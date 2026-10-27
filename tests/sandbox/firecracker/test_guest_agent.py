"""The guest agent's request handling, run on the host over socketpairs (Linux only).

``guest.py`` is the code that runs inside the VM; here it serves a temp dir instead of /work.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="guest.py runs on Linux")

GUEST_PATH = Path(__file__).resolve().parents[3] / "iterant/sandbox/firecracker/guest.py"


@pytest.fixture
def guest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location("fc_guest_under_test", GUEST_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(mod, "WORK", str(work))
    return mod


class Reply:
    def __init__(self, sock: socket.socket) -> None:
        self.f = sock.makefile("rb")
        line = self.f.readline()
        self.head: dict[str, Any] = json.loads(line) if line else {}
        self.eof = not line


def request(
    guest: Any, req: dict[str, Any], body: bytes = b"", raw: bytes | None = None
) -> tuple[Reply, socket.socket]:
    host, agent_side = socket.socketpair()
    threading.Thread(target=guest.handle, args=(agent_side, {}, False), daemon=True).start()
    host.sendall(raw if raw is not None else json.dumps(req).encode() + b"\n" + body)
    return Reply(host), host


def run_exec(guest: Any, cmd: str, timeout: float = 10) -> tuple[dict[str, Any], bytes, bytes]:
    rep, _ = request(guest, {"op": "exec", "cmd": cmd, "timeout": timeout})
    assert rep.head["ok"] is True, rep.head
    return rep.head, rep.f.read(rep.head["out"]), rep.f.read(rep.head["err"])


def test_ping_unknown_op_and_garbage(guest: Any) -> None:
    assert request(guest, {"op": "ping"})[0].head == {"ok": True}
    assert request(guest, {"op": "nope"})[0].head["ok"] is False
    assert request(guest, {}, raw=b"not json\n")[0].head["ok"] is False
    assert request(guest, {}, raw=b"[" * 60000 + b"\n")[0].head["ok"] is False  # no crash


def test_put_get_roundtrip_and_errors(guest: Any, tmp_path: Path) -> None:
    path = str(tmp_path / "work/a/b.bin")
    data = bytes(range(256)) * 50
    assert request(guest, {"op": "put", "path": path, "size": len(data)}, data)[0].head["ok"]
    rep, _ = request(guest, {"op": "get", "path": path})
    assert rep.head == {"ok": True, "size": len(data)} and rep.f.read(len(data)) == data
    assert request(guest, {"op": "get", "path": path + ".missing"})[0].head["errno"] == 2
    assert request(guest, {"op": "get", "path": str(tmp_path)})[0].head["errno"] == 21
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)  # pyright: ignore[reportAttributeAccessIssue]
    assert request(guest, {"op": "get", "path": str(fifo)})[0].head["errno"] == 22  # no hang
    bad = request(guest, {"op": "put", "path": "/proc/nope/x", "size": 3}, b"abc")[0].head
    assert bad["ok"] is False and bad["errno"]


def test_get_over_the_cap_is_refused(guest: Any, tmp_path: Path) -> None:
    guest.MAX_GET = 10
    big = tmp_path / "big"
    big.write_bytes(b"x" * 11)
    assert request(guest, {"op": "get", "path": str(big)})[0].head["errno"] == 27


def test_exec_result_exit_code_and_streams(guest: Any) -> None:
    head, out, err = run_exec(guest, "echo out; echo err >&2; exit 7")
    assert (head["exit"], head["timed_out"], head["truncated"]) == (7, False, False)
    assert (out, err) == (b"out\n", b"err\n")
    assert (head["out"], head["err"]) == (4, 4)  # header lengths match the bytes sent


def test_exec_runs_in_the_work_dir_with_a_clean_env(guest: Any, tmp_path: Path) -> None:
    _, out, _ = run_exec(guest, "pwd; echo $HOME")
    assert out.decode().split() == [str(tmp_path / "work"), str(tmp_path / "work")]


def test_exec_signal_death_is_128_plus_signal(guest: Any) -> None:
    assert run_exec(guest, "kill -9 $$")[0]["exit"] == 137


def test_exec_timeout_kills_the_whole_group(guest: Any, tmp_path: Path) -> None:
    start = time.monotonic()
    head, _, _ = run_exec(guest, "(sleep 2; touch late) & sleep 5", timeout=0.5)
    assert head["timed_out"] and head["exit"] == -1 and time.monotonic() - start < 3
    time.sleep(2.5)
    assert not (tmp_path / "work/late").exists()


def test_exec_truncates_output(guest: Any) -> None:
    guest.MAX_OUTPUT = 1000
    head, out, _ = run_exec(guest, "head -c 5000 /dev/zero | tr '\\0' x")
    assert head["truncated"] and head["out"] == len(out) == 1000 and head["exit"] == 0


def test_background_process_holding_the_pipes_does_not_hang_exec(guest: Any) -> None:
    start = time.monotonic()
    head, out, _ = run_exec(guest, "sleep 30 & echo started", timeout=20)
    assert time.monotonic() - start < 6  # output grace, not the sleep
    assert out == b"started\n" and not head["timed_out"] and head["exit"] == 0


def test_long_commands_go_through_a_script_file(
    guest: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(guest, "SHORT_CMD_MAX", 50)
    # Private script dir: the shared /tmp may hold other runs' .cmd-* files.
    private = tmp_path / "scripts"
    private.mkdir()
    real_mkstemp = guest.tempfile.mkstemp
    made: list[str] = []

    def mkstemp(*a: Any, **k: Any) -> tuple[int, str]:
        k["dir"] = str(private)
        fd, name = real_mkstemp(*a, **k)
        made.append(name)
        return fd, name

    monkeypatch.setattr(guest.tempfile, "mkstemp", mkstemp)
    cmd = "echo " + "x" * 5000
    _, out, _ = run_exec(guest, cmd)
    assert out == b"x" * 5000 + b"\n"
    assert len(made) == 1  # it did use a script file
    assert list(private.iterdir()) == []  # and cleaned it up


def test_spawn_failure_is_a_result_not_a_refusal(
    guest: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise OSError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(guest.subprocess, "Popen", boom)
    head, _, err = run_exec(guest, "true")
    assert head["ok"] is True and head["exit"] == 126 and b"cannot start" in err


def test_failure_after_the_command_started_drops_the_connection(
    guest: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(guest, "_pump", boom)
    rep, _ = request(guest, {"op": "exec", "cmd": "sleep 30; touch survived", "timeout": 10})
    assert rep.eof  # silence, never ok:false
    time.sleep(0.3)  # and the command was killed
    assert not (tmp_path / "work/survived").exists()
