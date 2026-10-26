"""The real subprocess runner, exercised with the current Python interpreter."""

from __future__ import annotations

import sys
import time

import pytest

from iterant.sandbox import subprocess_runner


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_captures_output_and_exit_code() -> None:
    out = subprocess_runner(py("import sys; print('o'); print('e', file=sys.stderr); sys.exit(3)"))
    assert (out.returncode, out.stdout.strip(), out.stderr.strip()) == (3, b"o", b"e")
    assert not out.timed_out and not out.truncated


def test_stdin_bytes_roundtrip() -> None:
    data = bytes(range(256)) * 1000
    out = subprocess_runner(
        py("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"), stdin=data
    )
    assert out.stdout == data


def test_timeout_kills_and_flags() -> None:
    start = time.monotonic()
    out = subprocess_runner(py("import time; time.sleep(30)"), timeout=0.5)
    assert out.timed_out and out.returncode == -1
    assert time.monotonic() - start < 10


def test_output_is_capped_without_deadlock() -> None:
    out = subprocess_runner(py("import sys; sys.stdout.write('x' * 5_000_000)"), max_output=1000)
    assert out.truncated and len(out.stdout) == 1000 and out.returncode == 0


def test_missing_binary_raises_oserror() -> None:
    with pytest.raises(OSError):
        subprocess_runner(["definitely-not-a-binary-iterant"])
