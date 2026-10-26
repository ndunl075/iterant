"""Grade a run: execute the agent's output in a sandbox, then decide the verdict here.

Invariant 8 (verdicts are decided outside untrusted code) shapes the flow:

1. The agent's files go under ``AGENT_DIR`` in a fresh, no-egress sandbox, next to the
   task's ``run`` script (``RUN_PATH``), which holds no expected values. ``run`` executes
   the agent's output there and prints one JSON object. (Paths are under ``/work``: the
   sandbox root is read-only and ``/work`` is its only writable, executable disk.)
2. The sandbox is destroyed. Its stdout is the only thing that crosses back. This process
   never executes, imports or unpickles it: it is parsed as strict JSON (at most
   ``MAX_RESULT_BYTES``, depth-limited, finite numbers only, no duplicate keys).
3. The hidden ``check`` (JSON, the exact expected object) is read from the trusted pack
   here and compared here. It is never written into any sandbox.

An infra error (``score is None``) may come only from things the agent cannot influence:
unreadable pack files, sandbox create failure, a failed trusted write, a failed trusted
pre-flight exec, or a backend exception while running ``run``. Everything the agent
influences scores 0.0: an invalid output path, a failed write of its files, any non-zero
exit (including 126/127), a timeout or kill, and a missing, oversized or malformed result.
A timeout is checked first; a non-zero exit scores 0.0 even if a result was printed.
A well-formed result scores 1.0 if it equals the expected object (types included: ``1``
is not ``true``), else 0.0.

``GradeResult.error`` holds fixed messages only: sandbox output and paths are
candidate-controlled or hidden, and must never reach the proposer or analyst.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from iterant.contracts import ExecResult, SandboxBackend, Task
from iterant.grader.outfiles import OutputRejected, check_limits

AGENT_DIR = "/work/out"
RUN_PATH = "/work/.grader/run.sh"
MAX_RESULT_BYTES = 64 * 1024
MAX_RESULT_DEPTH = 32
DEFAULT_TIMEOUT_S = 120.0  # wall-clock limit of one check; ``Grader(timeout=...)`` overrides
PREFLIGHT_TIMEOUT_S = 30.0


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


def _validate(value: Any) -> None:
    """Raise ``ValueError`` if ``value`` nests too deep or holds a non-finite float."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, level = stack.pop()
        if level > MAX_RESULT_DEPTH:
            raise ValueError("too deep")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite number")
        if isinstance(item, dict):
            stack.extend((c, level + 1) for c in cast("dict[str, Any]", item).values())
        elif isinstance(item, list):
            stack.extend((c, level + 1) for c in cast("list[Any]", item))


def parse_json(text: str) -> dict[str, Any]:
    """Strictly parse ``text`` as one JSON object; raise ``ValueError`` if bad."""
    if len(text.encode("utf-8", "surrogatepass")) > MAX_RESULT_BYTES:
        raise ValueError("too large")
    try:
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except RecursionError as exc:  # nesting beyond the parser's own limit
        raise ValueError("too deep") from exc
    if not isinstance(value, dict):
        raise ValueError("not an object")
    _validate(value)
    return cast("dict[str, Any]", value)


parse_result = parse_json


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


def interpret(result: ExecResult, expected: Any) -> float:
    """Score one ``run`` execution (agent-influenced, so never an infra error)."""
    if result.timed_out or result.exit_code != 0:
        return 0.0
    try:
        got = parse_json(result.stdout)
    except ValueError:
        return 0.0
    return 1.0 if same(got, expected) else 0.0


def _agent_path(rel: str) -> str | None:
    """Sandbox path for an agent file, or None if ``rel`` is not a canonical relative path."""
    if not rel or "\\" in rel or "\0" in rel:
        return None
    path = PurePosixPath(rel)
    if path.is_absolute() or path.as_posix() != rel or {"", ".", ".."} & set(rel.split("/")):
        return None
    return f"{AGENT_DIR}/{rel}"


class Grader:
    """Grades tasks of the pack at ``pack_root`` using sandboxes from ``backend``."""

    def __init__(
        self,
        backend: SandboxBackend,
        image: str,
        pack_root: Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
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
        """Score one run. ``out_files`` maps canonical paths relative to the agent's out dir."""
        start = time.monotonic()

        def done(score: float | None, error: str | None) -> GradeResult:
            return GradeResult(score, error, time.monotonic() - start)

        targets: dict[str, bytes] = {}
        for rel, data in out_files.items():
            target = _agent_path(rel)
            if target is None:
                return done(0.0, None)
            targets[target] = data
        try:
            check_limits(out_files)
        except OutputRejected:
            return done(0.0, None)
        check = PurePosixPath(task.check)
        try:
            run = self._root.joinpath(*check.parent.parts, "run").read_bytes()
            expected = parse_json(self._root.joinpath(*check.parts).read_bytes().decode("utf-8"))
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
            # Trusted steps first: if any of them fails, the agent had no say in it.
            try:
                sandbox.put(RUN_PATH, run)
                pre = sandbox.exec(
                    f"mkdir -p {AGENT_DIR} && test -r {RUN_PATH}", PREFLIGHT_TIMEOUT_S
                )
            except Exception as exc:
                return done(None, f"sandbox error: {type(exc).__name__}")
            if pre.timed_out or pre.exit_code != 0:
                return done(None, "sandbox preflight failed")
            try:
                for path, data in targets.items():
                    sandbox.put(path, data)
            except Exception:
                return done(0.0, None)  # e.g. a file where another needs a directory
            try:
                result = sandbox.exec(f"cd {AGENT_DIR} && exec sh {RUN_PATH}", self._timeout)
            except Exception as exc:
                return done(None, f"sandbox error: {type(exc).__name__}")
        finally:
            try:
                sandbox.destroy()
            except Exception:  # destroy is best-effort; the backend ttl reaps it
                pass
        return done(interpret(result, expected), None)
