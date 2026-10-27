"""One running Firecracker microVM: host side of the guest agent protocol (see ``guest.py``).

Trust: the guest agent runs as the sandbox user, so a command can kill or replace it. The host
therefore treats every guest answer as untrusted data and fails closed. Once a command has
been sent, a guest that vanishes, hangs, refuses or answers garbage is the candidate's doing
(``sandbox_lost``), never an infra error. Only failures on the host side of the line raise a
plain ``SandboxError``: a VM that is still running but unreachable, and a VM process that died
with a non-zero status (a guest reset exits Firecracker with status 0; a host-side kill or a
Firecracker crash does not).
"""

from __future__ import annotations

import errno
import json
import logging
import os
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from iterant.contracts import ExecResult
from iterant.sandbox.container import SandboxError, SandboxLost

log = logging.getLogger(__name__)

CONTROL_PORT = 5000
PROXY_VSOCK_PORT = 1025
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_COMMAND_CHARS = 1 << 20  # the guest runs anything over ~100 KB from a script file
EXEC_TTL_MARGIN_S = 5.0  # exec never runs closer than this to the ttl
EXEC_ANSWER_SLACK_S = 15.0  # the agent enforces the timeout; this much extra before "hung"
VM_EXIT_WAIT_S = 10.0  # a guest whose agent died powers off within this long (it syncs first)
MAX_REQUEST_BYTES = 8 * 1024 * 1024  # the guest agent's request-line limit (guest.MAX_REQUEST_LINE)
RELAY_MAX_CONNS = 64  # per sandbox: each costs two relay threads, two fds and a proxy thread
RELAY_IDLE_S = 300.0  # no byte either way for this long ends the connection
_IO_TIMEOUT_S = 120.0
_HEADER_MAX = 65536


class VmProcess(Protocol):
    """The slice of ``subprocess.Popen`` the sandbox needs from the VM process."""

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...


class _Gone(Exception):
    """The guest connection ended early: the agent (or the whole VM) is gone."""


class _Garbage(_Gone):
    """The guest answered something no agent would: only a replaced agent does that."""


def _require_abs(path: str) -> None:
    if not path.startswith("/") or "\0" in path:
        raise ValueError(f"sandbox path must be absolute, got {path!r}")


def _reset_as_gone(err: BaseException) -> BaseException:
    """A reset or broken pipe on an established agent stream means the agent went away
    mid-request. Treat it like a clean close: lost, once the VM has had time to exit (a
    guest that is still syncing before power-off must not read as an infra error)."""
    if isinstance(err, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
        return _Gone(f"connection reset by the guest: {type(err).__name__}")
    return err


def _unix_socket() -> socket.socket:
    if sys.platform == "win32":
        raise SandboxError("the firecracker backend needs Linux")
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


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
                raise _Garbage("oversized header line")
            if not self._fill():
                raise _Gone("connection closed before a full line")
        if i > _HEADER_MAX:
            raise _Garbage("oversized header line")
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
        """One JSON object line. The line is at most 64 KiB; anything the parser rejects
        (syntax, depth, size) is garbage."""
        try:
            obj = json.loads(self.line())
        except (ValueError, RecursionError, MemoryError) as e:
            raise _Garbage(f"unparseable header: {type(e).__name__}") from e
        if not isinstance(obj, dict):
            raise _Garbage("header is not an object")
        return cast("dict[str, Any]", obj)


def _int_field(head: dict[str, Any], key: str, low: int, high: int) -> int:
    val = head.get(key)
    if not isinstance(val, int) or isinstance(val, bool) or not low <= val <= high:
        raise _Garbage(f"bad {key!r} in header: {val!r}")
    return val


def _bool_field(head: dict[str, Any], key: str) -> bool:
    val = head.get(key)
    if not isinstance(val, bool):
        raise _Garbage(f"bad {key!r} in header: {val!r}")
    return val


def connect_vsock_uds(uds_path: str, deadline: float) -> Conn:
    """Open a stream to the guest's control port through Firecracker's vsock unix socket."""
    sock = _unix_socket()
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

    def _vm_exit_code(self) -> int | None:
        """Exit status of the VM process, waiting briefly for a dying guest; None if running."""
        try:
            return self._vm.wait(timeout=VM_EXIT_WAIT_S)
        except subprocess.TimeoutExpired:
            return None

    def _mark_lost(self) -> SandboxLost:
        self._lost = True
        try:
            self.destroy()
        except SandboxError:
            log.exception("could not tear down %s after losing it", self._name)
        return SandboxLost(f"sandbox {self._name} was lost to the command that ran in it")

    def _failure(self, what: str, err: BaseException) -> SandboxError:
        """Classify a transport failure.

        The guest powers itself off (Firecracker exits 0) when its agent dies, so a VM gone
        with status 0, a hung guest, or one answering garbage was broken from inside, by a
        command: lost. A VM process that died with a non-zero status, or a running VM that
        cannot be reached, is the host's problem: infra."""
        if self._dead:
            return SandboxError(f"sandbox {self._name} was destroyed or expired during {what}")
        if isinstance(err, (_Garbage, TimeoutError)):
            code = self._vm.poll()  # a hung or lying guest is still running
        else:
            code = self._vm_exit_code()
        if code is not None and code != 0:
            return SandboxError(f"{what} failed: the VM process died with status {code}")
        if code == 0 or isinstance(err, (_Gone, TimeoutError)):
            return self._mark_lost()
        return SandboxError(f"{what} failed: cannot reach the running VM: {err}")

    # -- put / get -------------------------------------------------------------------------

    def put(self, path: str, data: bytes) -> None:
        _require_abs(path)
        self._alive()
        req = json.dumps({"op": "put", "path": path, "size": len(data)}).encode() + b"\n"
        conn: Conn | None = None
        try:
            conn = self._open(_IO_TIMEOUT_S)
            try:
                conn.send(req + data)
                head = conn.header()
            except OSError as e:
                raise _reset_as_gone(e) from e
        except (OSError, _Gone, TimeoutError) as e:
            raise self._failure(f"put {path}", e) from e
        finally:
            if conn is not None:
                conn.close()
        if head.get("ok") is not True:
            raise SandboxError(f"put {path} failed: {head.get('msg')}")

    def get(self, path: str) -> bytes:
        _require_abs(path)
        self._alive()
        req = json.dumps({"op": "get", "path": path}).encode() + b"\n"
        conn: Conn | None = None
        try:
            conn = self._open(_IO_TIMEOUT_S)
            try:
                conn.send(req)
                head = conn.header()
                if head.get("ok") is True:
                    data = conn.exact(_int_field(head, "size", 0, MAX_OUTPUT_BYTES))
                else:
                    data = b""
            except OSError as e:
                raise _reset_as_gone(e) from e
        except (OSError, _Gone, TimeoutError) as e:
            raise self._failure(f"get {path}", e) from e
        finally:
            if conn is not None:
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
        killing the guest agent or the guest, or hanging, refusing or garbling it
        (``sandbox_lost``; later calls raise ``SandboxLost``). A timeout also results when the
        ttl leaves no room. Raised as ``SandboxError``: the sandbox was already destroyed or
        past its ttl (including a destroy or ttl expiry racing this exec), or the host cannot
        reach a VM that is still running or lost it to a host-side kill.
        """
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        if len(cmd) > MAX_COMMAND_CHARS:
            raise ValueError(f"command longer than {MAX_COMMAND_CHARS} characters")
        try:
            cmd.encode("utf-8")
        except UnicodeEncodeError as e:  # the guest could not run it and would answer ok:false
            raise ValueError("command is not valid text (lone surrogate)") from e
        self._alive()
        room = self._deadline - time.monotonic() - EXEC_TTL_MARGIN_S
        if room <= 0:
            return ExecResult(-1, "", "", True, 0.0)
        timeout = min(timeout, room)
        start = time.monotonic()
        req = json.dumps({"op": "exec", "cmd": cmd, "timeout": timeout}).encode() + b"\n"
        if len(req) > MAX_REQUEST_BYTES:  # JSON escaping can multiply the size
            raise ValueError(f"encoded exec request is {len(req)} bytes, over the guest limit")
        conn: Conn | None = None
        try:
            conn = self._open(_IO_TIMEOUT_S)
            conn.set_deadline(time.monotonic() + timeout + EXEC_ANSWER_SLACK_S)
            try:
                conn.send(req)
                head = conn.header()
                # The command is running: the agent answers with a result or not at all, so a
                # refusal or any out-of-range field is the agent's replacement talking.
                if head.get("ok") is not True:
                    raise _Garbage(f"agent refused a running command: {head.get('msg')!r}")
                n_out = _int_field(head, "out", 0, MAX_OUTPUT_BYTES)
                n_err = _int_field(head, "err", 0, MAX_OUTPUT_BYTES)
                timed_out = _bool_field(head, "timed_out")
                truncated = _bool_field(head, "truncated")
                code = _int_field(head, "exit", -1 if timed_out else 0, -1 if timed_out else 255)
                out, err = conn.exact(n_out), conn.exact(n_err)
            except OSError as e:
                raise _reset_as_gone(e) from e
        except (OSError, _Gone, TimeoutError) as e:
            wall = time.monotonic() - start
            failure = self._failure("exec", e)
            if isinstance(failure, SandboxLost):
                hung = isinstance(e, TimeoutError)
                return ExecResult(-1 if hung else 137, "", "", hung, wall, False, True)
            raise failure from e
        finally:
            if conn is not None:
                conn.close()
        wall = time.monotonic() - start
        if self._dead:
            raise SandboxError(f"sandbox {self._name} was destroyed or expired during exec")
        return ExecResult(
            code,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            timed_out,
            wall,
            truncated,
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
    port arrive on ``<uds>_<port>``; each is piped to the single proxy endpoint, nothing else.

    A command can open vsock connections to the host port directly, so the relay is bounded:
    at most ``max_conns`` at once (extra ones are closed at once) and a connection with no
    traffic for ``idle_s`` is closed. Without that, a flood would exhaust the orchestrator's
    threads and file descriptors for every sandbox."""

    def __init__(
        self,
        uds_path: str,
        host: str,
        port: int,
        *,
        max_conns: int = RELAY_MAX_CONNS,
        idle_s: float = RELAY_IDLE_S,
    ) -> None:
        self._target = (host, port)
        self._path = uds_path
        self._idle_s = idle_s
        self._slots = threading.BoundedSemaphore(max_conns)
        self._srv = _unix_socket()
        self._srv.bind(uds_path)
        self._srv.listen(max_conns)
        self._closed = False
        self._lock = threading.Lock()
        self._open: set[socket.socket] = set()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        while True:
            try:
                guest, _ = self._srv.accept()
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                guest.close()
                continue
            try:
                threading.Thread(target=self._serve, args=(guest,), daemon=True).start()
            except RuntimeError:  # cannot start a thread: refuse rather than die
                self._slots.release()
                guest.close()

    def _serve(self, guest: socket.socket) -> None:
        try:
            self._relay(guest)
        finally:
            self._slots.release()

    def _relay(self, guest: socket.socket) -> None:
        try:
            up = socket.create_connection(self._target, timeout=10)
        except OSError:
            guest.close()
            return
        with self._lock:
            if self._closed:
                guest.close()
                up.close()
                return
            self._open |= {guest, up}
        # One thread per connection: wait on both ends, end it after ``idle_s`` of silence.
        guest.settimeout(self._idle_s)
        up.settimeout(self._idle_s)
        sel = selectors.DefaultSelector()
        sel.register(guest, selectors.EVENT_READ, up)
        sel.register(up, selectors.EVENT_READ, guest)
        try:
            done = False
            while not done:
                ready = sel.select(self._idle_s)
                if not ready:
                    break
                for key, _ in ready:
                    data = cast(socket.socket, key.fileobj).recv(65536)
                    if not data:
                        done = True
                        break
                    cast(socket.socket, key.data).sendall(data)
        except OSError:
            pass
        finally:
            sel.close()
            with self._lock:
                self._open -= {guest, up}
            guest.close()
            up.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = list(self._open)
        _shutdown(self._srv)  # wakes the accept thread; close() alone does not
        try:
            self._srv.close()
        except OSError:
            pass
        _shutdown(*conns)
        self._thread.join(timeout=5)
        Path(self._path).unlink(missing_ok=True)


def _shutdown(*socks: socket.socket) -> None:
    for s in socks:
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the VM's process group (the ``timeout`` wrapper and Firecracker) and reap it."""
    if proc.poll() is None:
        if sys.platform == "win32":
            proc.kill()
        else:
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


def _open_dirs(root: Path) -> None:
    """Give every directory under ``root`` owner rwx, so it can be listed and emptied."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            if d.is_symlink() or not d.is_dir():
                continue
            os.chmod(d, 0o700)
            stack.extend(Path(e.path) for e in os.scandir(d))
        except OSError:
            pass


def remove_tree(path: str) -> None:
    """Remove a tree, including directories and files with mode 000."""
    _open_dirs(Path(path))
    shutil.rmtree(path)
