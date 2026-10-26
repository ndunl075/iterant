"""Subprocess runner behind the container backends.

Injectable so unit tests can assert the exact argv without a container runtime.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import IO, Protocol

MAX_OUTPUT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class RunOutput:
    """Result of one runtime invocation. ``returncode`` is -1 when ``timed_out``."""

    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""
    timed_out: bool = False
    truncated: bool = False


class Runner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout: float | None = None,
        max_output: int = MAX_OUTPUT_BYTES,
    ) -> RunOutput: ...


def _drain(stream: IO[bytes], limit: int, sink: bytearray, truncated: list[bool]) -> None:
    # Keep the first `limit` bytes and keep reading so the child never blocks on a full pipe.
    while chunk := stream.read(65536):
        room = max(limit - len(sink), 0)
        sink += chunk[:room]
        if len(chunk) > room:
            truncated[0] = True


def _feed(stream: IO[bytes], data: bytes) -> None:
    try:
        stream.write(data)
        stream.close()
    except OSError:  # child exited early; its return code tells the story
        pass


def subprocess_runner(
    argv: Sequence[str],
    *,
    stdin: bytes | None = None,
    timeout: float | None = None,
    max_output: int = MAX_OUTPUT_BYTES,
) -> RunOutput:
    """Run ``argv`` (never through a host shell), capping captured output at ``max_output``
    bytes per stream, and kill it after ``timeout`` seconds. Raises ``OSError`` if the
    binary cannot be started."""
    proc = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    out, err = bytearray(), bytearray()
    out_cut, err_cut = [False], [False]
    threads = [
        threading.Thread(target=_drain, args=(proc.stdout, max_output, out, out_cut), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, max_output, err, err_cut), daemon=True),
    ]
    if stdin is not None:
        assert proc.stdin is not None
        threads.append(threading.Thread(target=_feed, args=(proc.stdin, stdin), daemon=True))
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for t in threads:
        t.join(timeout=5)
    return RunOutput(
        returncode=-1 if timed_out else proc.returncode,
        stdout=bytes(out),
        stderr=bytes(err),
        timed_out=timed_out,
        truncated=out_cut[0] or err_cut[0],
    )
