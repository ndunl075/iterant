"""Run a task's hidden check in a fresh sandbox and turn its output into a score.

Invariants this module upholds:

* 1/2: the check is read from the trusted pack and handed only to the grader's own
  sandbox, which is created fresh for every grade, has no egress and is destroyed
  afterwards. The agent's sandbox never sees it. Only this module's result counts.
* The check is never written to a file in the sandbox: candidate code that the
  check runs shares its uid and could rewrite a file (``sh`` reads scripts
  incrementally). It is passed base64-encoded inside the command and decoded into a
  shell variable that ``sh -c`` runs.
  # TODO(contract): ``Sandbox.exec`` has no stdin; with stdin this would be ``sh -s``.
  The command is bounded by the OS argument limit (~128 KiB per argument on Linux).
* A broken check or sandbox is an infra error (``score is None``), never a score.

Check protocol: the check runs as ``sh -c`` with ``/out`` as its working directory.

* exit 126/127 means "could not run" and the check must use it only before it
  touches candidate code: infra error.
* any other non-zero exit, a timeout, or a kill (exit > 128, OOM) is 0.0: once the
  check can run candidate code, the candidate may be the cause of any failure.
* exit 0 scores 1.0, unless the last non-blank stdout line is a float in [0, 1],
  which is then the score. Exit 0 with any other last line is an infra error.
  A non-zero exit never scores above 0.0, whatever it printed.

``GradeResult.error`` holds fixed messages only, never sandbox output: stderr is
candidate-controlled and may carry hidden-check text. Never show it to the proposer
or analyst.
"""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from iterant.contracts import ExecResult, SandboxBackend, Task

OUT_DIR = "/out"
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


def interpret(result: ExecResult) -> tuple[float | None, str | None]:
    """Map a check's ``ExecResult`` to ``(score, None)`` or ``(None, infra error)``."""
    code = result.exit_code
    if code in _NOT_RUNNABLE:
        return None, f"check could not run (exit {code})"
    if result.timed_out or code != 0:
        return 0.0, None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return 1.0, None
    try:
        score = float(lines[-1])
    except ValueError:
        score = math.nan
    if math.isfinite(score) and 0.0 <= score <= 1.0:
        return score, None
    return None, "malformed check output"


def _command(check: bytes) -> str:
    payload = base64.b64encode(check).decode("ascii")
    return (
        f"S=$(printf %s '{payload}' | base64 -d) || exit 127; "
        f"mkdir -p {OUT_DIR} && cd {OUT_DIR} || exit 127; "
        'sh -c "$S"'
    )


def _out_path(rel: str) -> str:
    path = PurePosixPath(rel)
    if not rel or "\\" in rel or "\0" in rel or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"bad out file path: {rel!r}")
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
            check = self._root.joinpath(*PurePosixPath(task.check).parts).read_bytes()
        except (ValueError, OSError) as exc:
            return done(None, f"cannot prepare grading: {exc}")

        try:
            sandbox = self._backend.create(
                self._image,
                self._cpu,
                self._mem,
                self._disk,
                self._timeout + 60.0,
                net="default-deny",
            )
        except Exception as exc:
            return done(None, f"sandbox create failed: {exc}")
        try:
            for path, data in targets.items():
                sandbox.put(path, data)
            result = sandbox.exec(_command(check), self._timeout)
        except Exception as exc:
            return done(None, f"sandbox error: {type(exc).__name__}")
        finally:
            try:
                sandbox.destroy()
            except Exception:  # destroy is best-effort; the backend ttl reaps it
                pass
        score, error = interpret(result)
        return done(score, error)
