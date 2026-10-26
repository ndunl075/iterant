"""Grade a run: execute the agent's output in a sandbox, then decide the verdict here.

Invariant 8 (verdicts are decided outside untrusted code) shapes the flow:

1. The agent's ``/out`` files go into a fresh, no-egress sandbox together with the task's
   ``run`` script, which holds no expected values. ``run`` executes the agent's output
   there and prints one JSON object.
2. The sandbox is destroyed. Its stdout is the only thing that crosses back. This process
   never executes, imports or unpickles it: it is parsed as strict JSON (at most
   ``MAX_RESULT_BYTES``, depth-limited, no NaN/Infinity, no duplicate keys).
3. The hidden ``check`` (JSON, the exact expected object) is read from the trusted pack
   here and compared here. It is never written into any sandbox.

Scoring:

* exit 126/127 means "could not run": an infra error. ``run`` may use it only before it
  touches agent code.
* any other non-zero exit, a timeout, or a kill (exit > 128, OOM) is 0.0: once ``run``
  executes agent code, the agent may be the cause.
* exit 0 with a missing, oversized or malformed result is 0.0. Exit 0 with a well-formed
  result scores 1.0 if it equals the expected object (types included: ``1`` is not
  ``true``), else 0.0.
* a broken pack file or sandbox is an infra error (``score is None``), never a score.

``GradeResult.error`` holds fixed messages only: sandbox output and paths are
candidate-controlled or hidden, and must never reach the proposer or analyst.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from iterant.contracts import ExecResult, SandboxBackend, Task

OUT_DIR = "/out"
RUN_PATH = "/run/run.sh"
MAX_RESULT_BYTES = 64 * 1024
MAX_RESULT_DEPTH = 32
_NOT_RUNNABLE = {126, 127}


@dataclass(frozen=True)
class GradeResult:
    """``score`` is set exactly when ``error`` is None."""

    score: float | None
    error: str | None
    wall_s: float

    def __post_init__(self) -> None:
        if (self.score is None) == (self.error is None):
            raise ValueError("exactly one of score and error must be set")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be within [0, 1]")

    @property
    def infra_err(self) -> bool:
        return self.error is not None


def _reject_constant(_name: str) -> Any:
    raise ValueError("non-finite number")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate key")
    return out


def _depth(value: Any) -> int:
    depth = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, level = stack.pop()
        children: list[Any]
        if isinstance(item, dict):
            children = list(cast("dict[str, Any]", item).values())
        elif isinstance(item, list):
            children = list(cast("list[Any]", item))
        else:
            continue
        depth = max(depth, level)
        stack.extend((c, level + 1) for c in children)
    return depth


def parse_result(text: str) -> dict[str, Any]:
    """Strictly parse a sandbox's stdout as one JSON object; raise ``ValueError`` if bad."""
    if len(text.encode("utf-8", "surrogatepass")) > MAX_RESULT_BYTES:
        raise ValueError("too large")
    try:
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except RecursionError as exc:  # nesting beyond the parser's own limit
        raise ValueError("too deep") from exc
    if not isinstance(value, dict) or _depth(value) > MAX_RESULT_DEPTH:
        raise ValueError("not an object, or too deep")
    return cast("dict[str, Any]", value)


def same(a: Any, b: Any) -> bool:
    """JSON equality that keeps types apart (``1`` != ``True``; ``1`` == ``1.0``)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        da, db = cast("dict[str, Any]", a), cast("dict[str, Any]", b)
        return da.keys() == db.keys() and all(same(da[k], db[k]) for k in da)
    if isinstance(a, list) and isinstance(b, list):
        la, lb = cast("list[Any]", a), cast("list[Any]", b)
        return len(la) == len(lb) and all(same(x, y) for x, y in zip(la, lb, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return isinstance(a, str) and isinstance(b, str) and a == b or a is None and b is None


def interpret(result: ExecResult, expected: Any) -> tuple[float | None, str | None]:
    """Map a ``run`` execution to ``(score, None)`` or ``(None, infra error)``."""
    code = result.exit_code
    if code in _NOT_RUNNABLE:
        return None, f"run script could not run (exit {code})"
    if result.timed_out or code != 0:
        return 0.0, None
    try:
        got = parse_result(result.stdout)
    except ValueError:
        return 0.0, None
    return (1.0 if same(got, expected) else 0.0), None


def _out_path(rel: str) -> str:
    path = PurePosixPath(rel)
    if not rel or "\\" in rel or "\0" in rel or path.is_absolute() or ".." in path.parts:
        raise ValueError("bad out file path")
    return f"{OUT_DIR}/{path.as_posix()}"


class Grader:
    """Grades tasks of the pack at ``pack_root`` using sandboxes from ``backend``."""

    def __init__(
        self,
        backend: SandboxBackend,
        image: str,
        pack_root: Path,
        *,
        timeout: float = 120.0,
        cpu: float = 1.0,
        mem: int = 1024,
        disk: int = 1024,
    ) -> None:
        self._backend = backend
        self._image = image
        self._root = pack_root
        self._timeout = timeout
        self._cpu, self._mem, self._disk = cpu, mem, disk

    def grade(self, task: Task, out_files: Mapping[str, bytes]) -> GradeResult:
        """Score one run. ``out_files`` maps paths relative to ``/out`` to their bytes."""
        start = time.monotonic()

        def done(score: float | None, error: str | None) -> GradeResult:
            return GradeResult(score, error, time.monotonic() - start)

        try:
            targets = {_out_path(rel): data for rel, data in out_files.items()}
        except ValueError:
            return done(None, "bad out file path")
        check = PurePosixPath(task.check)
        try:
            run = self._root.joinpath(*check.parent.parts, "run").read_bytes()
            expected = json.loads(self._root.joinpath(*check.parts).read_bytes())
        except (ValueError, OSError):
            return done(None, "cannot read pack files")

        try:
            sandbox = self._backend.create(
                self._image,
                self._cpu,
                self._mem,
                self._disk,
                self._timeout + 60.0,
                net="default-deny",
            )
        except Exception:
            return done(None, "sandbox create failed")
        try:
            for path, data in targets.items():
                sandbox.put(path, data)
            sandbox.put(RUN_PATH, run)
            cmd = f"mkdir -p {OUT_DIR} && cd {OUT_DIR} && sh {RUN_PATH}"
            result = sandbox.exec(cmd, self._timeout)
        except Exception as exc:
            return done(None, f"sandbox error: {type(exc).__name__}")
        finally:
            try:
                sandbox.destroy()
            except Exception:  # destroy is best-effort; the backend ttl reaps it
                pass
        score, error = interpret(result, expected)
        return done(score, error)
