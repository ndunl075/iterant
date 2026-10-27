"""Host side of the Firecracker backend against a fake guest agent (no KVM, no VM needed)."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from collections.abc import Callable

import pytest

from iterant.sandbox import FirecrackerSandbox, SandboxError, SandboxLost
from iterant.sandbox.firecracker.sandbox import Conn

Script = Callable[[socket.socket, dict[str, object], bytes], None]


class FakeVm:
    def __init__(self, exited: bool = False) -> None:
        self.exited = exited

    def poll(self) -> int | None:
        return 0 if self.exited else None

    def wait(self, timeout: float | None = None) -> int:
        if not self.exited:
            raise subprocess.TimeoutExpired("vm", timeout or 0)
        return 0


class Rig:
    def __init__(self, script: Script, vm: FakeVm | None = None, ttl: float = 300) -> None:
        self.vm = vm or FakeVm()
        self.script = script
        self.killed = 0
        self.cleaned = 0
        self.refuse = False
        self.sandbox = FirecrackerSandbox(
            name="t", vm=self.vm, connect=self._connect, ttl=ttl,
            kill_vm=self._kill, cleanup=self._clean,
        )  # fmt: skip
        self.sandbox.start_ttl()

    def _kill(self) -> None:
        self.killed += 1
        self.vm.exited = True

    def _clean(self) -> None:
        self.cleaned += 1

    def _connect(self, deadline: float) -> Conn:
        if self.refuse:
            raise ConnectionRefusedError("no agent")
        host, guest = socket.socketpair()
        threading.Thread(target=self._serve, args=(guest,), daemon=True).start()
        return Conn(host, deadline)

    def _serve(self, sock: socket.socket) -> None:
        try:
            f = sock.makefile("rb")
            req = json.loads(f.readline())
            rest = f.read(int(req["size"])) if req.get("op") == "put" else b""
            self.script(sock, req, rest)
        finally:
            sock.close()


def reply(sock: socket.socket, head: dict[str, object], *payload: bytes) -> None:
    sock.sendall(json.dumps(head).encode() + b"\n" + b"".join(payload))


def exec_ok(sock: socket.socket, req: dict[str, object], _: bytes) -> None:
    out, err = b"hello\n", b"warn\n"
    reply(sock, {"ok": True, "exit": 3, "timed_out": False, "truncated": False,
                 "out": len(out), "err": len(err)}, out, err)  # fmt: skip


def test_exec_result_is_mapped() -> None:
    rig = Rig(exec_ok)
    res = rig.sandbox.exec("whatever", 10)
    assert (res.exit_code, res.stdout, res.stderr) == (3, "hello\n", "warn\n")
    assert not res.timed_out and not res.truncated and not res.sandbox_lost


def test_exec_timeout_and_truncation_flags_pass_through() -> None:
    def script(sock: socket.socket, req: dict[str, object], _: bytes) -> None:
        reply(sock, {"ok": True, "exit": 0, "timed_out": True, "truncated": True,
                     "out": 0, "err": 0})  # fmt: skip

    res = Rig(script).sandbox.exec("x", 1)
    assert res.timed_out and res.exit_code == -1 and res.truncated and not res.sandbox_lost


def test_guest_dying_mid_exec_is_a_lost_sandbox() -> None:
    def script(sock: socket.socket, req: dict[str, object], _: bytes) -> None:
        pass  # closes the connection without answering: the agent was killed

    rig = Rig(script)
    res = rig.sandbox.exec("kill -9 -1", 10)  # must not raise
    assert res.sandbox_lost and not res.timed_out and res.exit_code != 0
    assert rig.killed == 1 and rig.cleaned == 1
    for call in (
        lambda: rig.sandbox.exec("true", 5),
        lambda: rig.sandbox.put("/work/f", b""),
        lambda: rig.sandbox.get("/work/f"),
    ):
        with pytest.raises(SandboxLost):
            call()
    rig.sandbox.destroy()
    rig.sandbox.destroy()
    assert rig.killed == 1 and rig.cleaned == 1


@pytest.mark.parametrize(
    "head",
    [
        {"ok": True, "exit": 0, "timed_out": False, "out": 1 << 30, "err": 0},
        {"ok": True, "exit": "0", "timed_out": False, "out": 0, "err": 0},
        {"ok": True, "exit": 0, "timed_out": False, "out": -1, "err": 0},
    ],
)
def test_garbage_from_the_guest_fails_closed(head: dict[str, object]) -> None:
    rig = Rig(lambda sock, req, _: reply(sock, head))
    assert rig.sandbox.exec("x", 5).sandbox_lost


def test_unparseable_header_fails_closed() -> None:
    rig = Rig(lambda sock, req, _: sock.sendall(b"not json\n"))
    assert rig.sandbox.exec("x", 5).sandbox_lost


def test_hung_guest_is_lost_after_the_answer_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iterant.sandbox.firecracker.sandbox.EXEC_ANSWER_SLACK_S", 0.3)
    release = threading.Event()

    def script(sock: socket.socket, req: dict[str, object], _: bytes) -> None:
        release.wait(10)

    rig = Rig(script)
    start = time.monotonic()
    res = rig.sandbox.exec("sleep 100", 0.2)
    release.set()
    assert res.sandbox_lost and res.timed_out and res.exit_code == -1
    assert time.monotonic() - start < 5


def test_unreachable_running_vm_is_an_infra_error_not_a_lost_sandbox() -> None:
    rig = Rig(exec_ok)
    rig.refuse = True
    with pytest.raises(SandboxError) as info:
        rig.sandbox.exec("x", 5)
    assert not isinstance(info.value, SandboxLost)
    assert rig.killed == 0


def test_vm_gone_without_destroy_is_lost() -> None:
    rig = Rig(exec_ok, vm=FakeVm(exited=True))
    rig.refuse = True
    assert rig.sandbox.exec("x", 5).sandbox_lost


def test_get_maps_errors() -> None:
    def script(sock: socket.socket, req: dict[str, object], _: bytes) -> None:
        path = str(req["path"])
        if path == "/data":
            reply(sock, {"ok": True, "size": 3}, b"abc")
        elif path == "/dir":
            reply(sock, {"ok": False, "errno": 21, "msg": "is a directory"})
        elif path == "/big":
            reply(sock, {"ok": False, "errno": 27, "msg": "too large"})
        else:
            reply(sock, {"ok": False, "errno": 2, "msg": "missing"})

    sb = Rig(script).sandbox
    assert sb.get("/data") == b"abc"
    with pytest.raises(FileNotFoundError):
        sb.get("/nope")
    with pytest.raises(IsADirectoryError):
        sb.get("/dir")
    with pytest.raises(SandboxError):
        sb.get("/big")
    with pytest.raises(ValueError):
        sb.get("relative")


def test_put_sends_the_payload_and_maps_errors() -> None:
    seen: list[tuple[str, bytes]] = []

    def script(sock: socket.socket, req: dict[str, object], body: bytes) -> None:
        seen.append((str(req["path"]), body))
        if str(req["path"]).startswith("/usr"):
            reply(sock, {"ok": False, "errno": 30, "msg": "read-only file system"})
        else:
            reply(sock, {"ok": True})

    sb = Rig(script).sandbox
    sb.put("/work/a/b.bin", bytes(range(256)) * 100)
    assert seen == [("/work/a/b.bin", bytes(range(256)) * 100)]
    with pytest.raises(RuntimeError):
        sb.put("/usr/x", b"x")


def test_ttl_expiry_destroys_and_exec_then_raises() -> None:
    rig = Rig(exec_ok, ttl=0.3)
    deadline = time.monotonic() + 5
    while rig.cleaned == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert rig.killed == 1 and rig.cleaned == 1
    with pytest.raises(SandboxError) as info:
        rig.sandbox.exec("true", 5)
    assert not isinstance(info.value, SandboxLost)


def test_exec_that_cannot_fit_in_the_ttl_is_a_timeout() -> None:
    res = Rig(exec_ok, ttl=3).sandbox.exec("x", 100)  # ttl minus the margin is <= 0
    assert res.timed_out and res.exit_code == -1
