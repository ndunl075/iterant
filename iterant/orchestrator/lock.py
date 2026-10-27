"""One writer per archive: an OS file lock, released by the OS if the process dies."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import TracebackType
from typing import IO, Self

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class OrchestratorBusy(RuntimeError):
    """Another orchestrator holds this archive. Two writers would race on batches and runs."""


class WriterLock:
    def __init__(self, repo: str | os.PathLike[str]) -> None:
        self._path = Path(repo) / ".iterant" / "orchestrator.lock"
        self._fh: IO[bytes] | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "a+b")  # noqa: SIM115 - held until release()
        try:
            if sys.platform == "win32":
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise OrchestratorBusy(
                f"another orchestrator is running on {self._path.parent}"
            ) from exc
        self._fh = fh

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
