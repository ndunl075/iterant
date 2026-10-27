"""One running Firecracker microVM: host side of the guest agent protocol (see ``guest.py``).

Trust: the guest agent runs as the sandbox user, so a command can kill or replace it. The host
therefore treats every guest answer as untrusted data and fails closed. A guest that vanishes,
hangs or answers garbage is the candidate's doing (``sandbox_lost``), never an infra error; only
failures on the host side of the line (no way to reach a VM that is still running) raise a
plain ``SandboxError``.
"""

# The host checker may run on a platform without these POSIX APIs (e.g. Windows).
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from iterant.contracts import ExecResult
from iterant.sandbox.container import SandboxError, SandboxLost

log = logging.getLogger(__name__)

CONTROL_PORT = 5000
PROXY_VSOCK_PORT = 1025
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
EXEC_TTL_MARGIN_S = 5.0  # exec never runs closer than this to the ttl
EXEC_ANSWER_SLACK_S = 15.0  # the agent enforces the timeout; this much extra before "hung"
VM_EXIT_WAIT_S = 3.0  # a guest whose agent died powers off within this long
_IO_TIMEOUT_S = 120.0
_HEADER_MAX = 65536


class VmProcess(Protocol):
    """The slice of ``subprocess.Popen`` the sandbox needs from the VM process."""

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...


class _Gone(Exception):
    """The guest answered nothing sensible: it died, hung or spoke garbage."""


def _require_abs(path: str) -> None:
    if not path.startswith("/") or "\0" in path:
        raise ValueError(f"sandbox path must be absolute, got {path!r}")


class Conn:
    """One request/response exchange over a stream socket, with an overall deadline."""

    def __init__(self, sock: socket.socket, deadline: float) -> None:
        self._sock = sock
        self._deadline = deadline
        self._buf = bytearray()

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def _fill(self) -> bool:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("guest did not answer in time")
        self._sock.settimeout(left)
        chunk = self._sock.recv(1 << 20)
        self._buf += chunk
        return bool(chunk)

    def send(self, data: bytes) -> None:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("guest did not accept the request in time")
        self._sock.settimeout(left)
        self._sock.sendall(data)

    def line(self) -> bytes:
        while (i := self._buf.find(b"\n")) < 0:
            if len(self._buf) > _HEADER_MAX:
                raise _Gone("oversized header line")
            if not self._fill():
                raise _Gone("connection closed before a full line")
        out = bytes(self._buf[:i])
        del self._buf[: i + 1]
        return out

    def exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            if not self._fill():
                raise _Gone("connection closed mid-payload")
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def header(self) -> dict[str, Any]:
        try:
            obj = json.loads(self.line())
        except ValueError as e:
            raise _Gone(f"unparseable header: {e}") from e
        if not isinstance(obj, dict):
            raise _Gone("header is not an object")
        return obj  # type: ignore[return-value]


def _int_field(head: dict[str, Any], key: str, limit: int) -> int:
    val = head.get(key)
    if not isinstance(val, int) or isinstance(val, bool) or not 0 <= val <= limit:
        raise _Gone(f"bad {key!r} in header: {val!r}")
    return val


def connect_vsock_uds(uds_path: str, deadline: float) -> Conn:
    """Open a stream to the guest's control port through Firecracker's vsock unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn = Conn(sock, deadline)
    try:
        sock.settimeout(max(deadline - time.monotonic(), 0.1))
        sock.connect(uds_path)
        conn.send(f"CONNECT {CONTROL_PORT}\n".encode())
        answer = conn.line()
        if not answer.startswith(b"OK"):
            raise ConnectionRefusedError(answer.decode(errors="replace"))
    except BaseException:
        conn.close()
        raise
    return conn


class FirecrackerSandbox:
    """A booted microVM. ``connect`` opens a control stream; ``cleanup`` releases host-side
    resources (VM process, proxy relay, sandbox directory) and must be safe to call twice."""

    def __init__(
        self,
        *,
        name: str,
        vm: VmProcess,
        connect: Callable[[float], Conn],
        ttl: float,
        kill_vm: Callable[[], None],
        cleanup: Callable[[], None],
        proxy_url: str | None = None,
    ) -> None:
        self._name = name
        self._vm = vm
        self._connect = connect
        self._kill_vm = kill_vm
        self._cleanup = cleanup
        self.proxy_url = proxy_url
        self._ttl = ttl
        self._deadline = time.monotonic() + ttl
        self._dead = False
        self._lost = False
        self._released = False
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    @property
    def resource_id(self) -> str:
        return self._name

    @property
    def vm_running(self) -> bool:
        return self._vm.poll() is None

    def start_ttl(self) -> None:
        """Start the ttl clock (called once the guest is up, so boot time is free)."""
        self._deadline = time.monotonic() + self._ttl
        self._timer = threading.Timer(self._ttl, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def _expire(self) -> None:
        try:
            self.destroy()
        except Exception:  # timer thread: nowhere to raise to
            log.exception("ttl teardown of %s failed", self._name)

    def _alive(self) -> None:
        if self._lost:
            raise SandboxLost(f"sandbox {self._name} was lost to the command that ran in it")
        if self._dead or time.monotonic() >= self._deadline:
            raise SandboxError(f"sandbox {self._name} is destroyed or past its ttl")

    def _open(self, budget: float) -> Conn:
        return self._connect(time.monotonic() + budget)

    def _vm_exited(self) -> bool:
        try:
            self._vm.wait(timeout=VM_EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _mark_lost(self) -> SandboxLost:
        self._lost = True
        try:
            self.destroy()
        except SandboxError:
            log.exception("could not tear down %s after losing it", self._name)
        return SandboxLost(f"sandbox {self._name} was lost to the command that ran in it")

    def _failure(self, what: str, err: BaseException) -> SandboxError:
        """Classify a transport failure. Only the host can fail to reach a *running* VM; a VM
        that is gone, hung or garbling was broken from inside, by a command."""
        if self._dead:
            return SandboxError(f"sandbox {self._name} was destroyed or expired during {what}")
        if isinstance(err, (_Gone, TimeoutError)) or self._vm_exited():
            return self._mark_lost()
        return SandboxError(f"{what} failed: cannot reach the running VM: {err}")

    # -- put / get -------------------------------------------------------------------------

    def put(self, path: str, data: bytes) -> None:
        _require_abs(path)
        self._alive()
        req = json.dumps({"op": "put", "path": path, "size": len(data)}).encode() + b"\n"
        try:
            conn = self._open(_IO_TIMEOUT_S)
        except (OSError, _Gone) as e:
            raise self._failure(f"put {path}", e) from e
        try:
            conn.send(req + data)
            head = conn.header()
        except (OSError, _Gone, TimeoutError) as e:
            raise self._failure(f"put {path}", e) from e
        finally:
            conn.close()
        if head.get("ok") is not True:
            raise SandboxError(f"put {path} failed: {head.get('msg')}")

    def get(self, path: str) -> bytes:
        _require_abs(path)
        self._alive()
        req = json.dumps({"op": "get", "path": path}).encode() + b"\n"
        try:
            conn = self._open(_IO_TIMEOUT_S)
        except (OSError, _Gone) as e:
            raise self._failure(f"get {path}", e) from e
        try:
            conn.send(req)
            head = conn.header()
            if head.get("ok") is True:
                data = conn.exact(_int_field(head, "size", MAX_OUTPUT_BYTES))
            else:
                data = b""
        except (OSError, _Gone, TimeoutError) as e:
            raise self._failure(f"get {path}", e) from e
        finally:
            conn.close()
        if head.get("ok") is not True:
            code = head.get("errno")
            if code == errno.ENOENT:
                raise FileNotFoundError(path)
            if code == errno.EISDIR:
                raise IsADirectoryError(path)
            if code == errno.EFBIG:
                raise SandboxError(f"get {path}: file larger than the output cap")
            raise SandboxError(f"get {path} failed: {head.get('msg')}")
        return data

    # -- exec ------------------------------------------------------------------------------

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        """Run ``cmd``; see ``Sandbox.exec`` for the contract.

        Returned, never raised: a timeout (the agent kills the command's whole process group),
        truncated output, a non-zero exit, an OOM kill, pids exhaustion, and the command
        killing the guest agent or the guest, or hanging it past the timeout (``sandbox_lost``;
        later calls raise ``SandboxLost``). A timeout also results when the ttl leaves no room.
        Raised as ``SandboxError``: the sandbox was already destroyed or past its ttl
        (including a destroy or ttl expiry racing this exec), or the host cannot reach a VM
        that is still running.
        """
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._alive()
        room = self._deadline - time.monotonic() - EXEC_TTL_MARGIN_S
        if room <= 0:
            return ExecResult(-1, "", "", True, 0.0)
        timeout = min(timeout, room)
        start = time.monotonic()
        req = json.dumps({"op": "exec", "cmd": cmd, "timeout": timeout}).encode() + b"\n"
        conn: Conn | None = None
        try:
            conn = self._open(_IO_TIMEOUT_S)
            conn.set_deadline(time.monotonic() + timeout + EXEC_ANSWER_SLACK_S)
            conn.send(req)
            head = conn.header()
            if head.get("ok") is not True:
                raise SandboxError(f"exec failed in the guest agent: {head.get('msg')}")
            n_out = _int_field(head, "out", MAX_OUTPUT_BYTES)
            n_err = _int_field(head, "err", MAX_OUTPUT_BYTES)
            code = head.get("exit")
            if not isinstance(code, int) or isinstance(code, bool):
                raise _Gone(f"bad exit code {code!r}")
            out, err = conn.exact(n_out), conn.exact(n_err)
        except (OSError, _Gone, TimeoutError) as e:
            wall = time.monotonic() - start
            failure = self._failure("exec", e)
            if isinstance(failure, SandboxLost):
                return ExecResult(-1 if isinstance(e, TimeoutError) else 137, "", "",
                                  isinstance(e, TimeoutError), wall, False, True)  # fmt: skip
            raise failure from e
        finally:
            if conn is not None:
                conn.close()
        wall = time.monotonic() - start
        if self._dead:
            raise SandboxError(f"sandbox {self._name} was destroyed or expired during exec")
        timed_out = head.get("timed_out") is True
        return ExecResult(
            -1 if timed_out else code,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            timed_out,
            wall,
            head.get("truncated") is True,
            False,
        )

    # -- lifecycle -------------------------------------------------------------------------

    def wait_ready(self, deadline: float) -> None:
        """Poll the agent until it answers; raises ``SandboxError`` on boot failure."""
        last: BaseException | None = None
        while time.monotonic() < deadline:
            if self._vm.poll() is not None:
                raise SandboxError("the microVM exited during boot")
            try:
                conn = self._connect(min(deadline, time.monotonic() + 2.0))
            except (OSError, _Gone) as e:
                last = e
                time.sleep(0.1)
                continue
            try:
                conn.send(b'{"op": "ping"}\n')
                if conn.header().get("ok") is True:
                    return
            except (OSError, _Gone, TimeoutError) as e:
                last = e
            finally:
                conn.close()
            time.sleep(0.1)
        raise SandboxError(f"the guest agent did not come up in time (last error: {last})")

    def destroy(self) -> None:
        with self._lock:
            if self._released:
                return
            self._dead = True
            if self._timer is not None:
                self._timer.cancel()
            errors: list[str] = []
            for step in (self._kill_vm, self._cleanup):
                try:
                    step()
                except (OSError, SandboxError, subprocess.SubprocessError) as e:
                    errors.append(str(e))
            if errors:
                raise SandboxError("teardown incomplete: " + "; ".join(errors))
            self._released = True


class ProxyRelay:
    """Host end of the one allowed guest route: the guest's vsock connections to the proxy
    port arrive on ``<uds>_<port>``; each is piped to the single proxy endpoint, nothing else."""

    def __init__(self, uds_path: str, host: str, port: int) -> None:
        self._target = (host, port)
        self._path = uds_path
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(uds_path)
        self._srv.listen(32)
        self._closed = False
        self._lock = threading.Lock()
        self._open: set[socket.socket] = set()
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                guest, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(guest,), daemon=True).start()

    def _serve(self, guest: socket.socket) -> None:
        try:
            up = socket.create_connection(self._target, timeout=10)
            up.settimeout(None)
        except OSError:
            guest.close()
            return
        with self._lock:
            if self._closed:
                guest.close()
                up.close()
                return
            self._open |= {guest, up}
        t = threading.Thread(target=_pipe, args=(up, guest), daemon=True)
        t.start()
        _pipe(guest, up)
        t.join()
        with self._lock:
            self._open -= {guest, up}
        guest.close()
        up.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = list(self._open)
        try:
            self._srv.close()
        except OSError:
            pass
        for s in conns:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        Path(self._path).unlink(missing_ok=True)


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while data := src.recv(65536):
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the VM's process group (the ``timeout`` wrapper and Firecracker) and reap it."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired as e:
        raise SandboxError(f"VM process {proc.pid} did not exit after SIGKILL") from e


def remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=False)
