"""Guest-side init and exec agent, copied into the microVM rootfs as ``/.iterant/guest.py``.

Runs INSIDE the guest (Linux, python3 from the sandbox image, stdlib only; keep it free of
syntax newer than Python 3.8). The host code only reads its text; it never imports it.

PID 1 (root) mounts the pseudo file systems and the ``/work`` disk, and builds a cgroup-v2
leaf (``cmd``) carrying the cpu/memory/pids caps for commands. PID 1 and the agent stay
outside that leaf, so a command can neither starve nor OOM them. PID 1 then forks the agent and
drops it to the sandbox user. If the agent dies (a command killed it), PID 1 powers the VM
off: the host sees the VM gone and reports ``sandbox_lost``. The agent serves one request per
vsock connection:

    request  : one JSON line, then ``size`` raw bytes for ``put``
    response : one JSON line, then raw bytes named by the header (``get`` data, exec output)

Once a command has started, the agent never answers ``ok: false``: it either returns the
result or drops the connection (the host treats both a refusal and silence as a lost
sandbox). Everything the agent says is untrusted by the host: a command can kill or replace it.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import json
import os
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

if sys.platform == "win32":  # keeps the type checker from analysing Linux-only calls elsewhere
    raise ImportError("guest.py runs inside the Linux guest")

AF_VSOCK = 40
VMADDR_CID_ANY = 0xFFFFFFFF
VMADDR_CID_HOST = 2
CONTROL_PORT = 5000
PROXY_VSOCK_PORT = 1025
PROXY_LOCAL_PORT = 8080
SANDBOX_UID = 10001
WORK = "/work"
MAX_OUTPUT = 16 * 1024 * 1024
MAX_GET = MAX_OUTPUT
MAX_REQUEST_LINE = 8 * 1024 * 1024
SHORT_CMD_MAX = 100_000  # longer commands go through a script file (argv limit is 128 KiB)
EXIT_GRACE_S = 1.0  # after the command exits, how long to wait for output left in the pipes
CGROUP_FD = 9  # open on cmd/cgroup.procs; the command wrapper joins the cmd cgroup through it
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC, MS_REMOUNT, MS_BIND = 1, 2, 4, 8, 32, 4096


def _libc() -> Any:
    lib = ctypes.CDLL(None, use_errno=True)
    lib.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    ]
    return lib


LIBC: Any = _libc()


def mount(src: str, target: str, fstype: str | None, flags: int = 0, data: str = "") -> None:
    res = LIBC.mount(
        src.encode(),
        target.encode(),
        fstype.encode() if fstype else None,
        flags,
        data.encode() if data else None,
    )
    if res != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"mount {fstype or 'bind'} on {target}: {os.strerror(e)}")


def log(msg: str) -> None:
    try:
        sys.stderr.write("iterant-guest: " + msg + "\n")
        sys.stderr.flush()
    except OSError:
        pass


def read_config() -> dict[str, Any]:
    with open("/proc/cmdline") as f:
        for word in f.read().split():
            if word.startswith("iterant.cfg="):
                return json.loads(base64.urlsafe_b64decode(word.split("=", 1)[1]))
    raise RuntimeError("no iterant.cfg on the kernel command line")


def write_file(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def bring_up_loopback() -> None:
    import fcntl

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        fcntl.ioctl(sock.fileno(), 0x8914, struct.pack("16sh14x", b"lo", 1))  # SIOCSIFFLAGS: UP
    finally:
        sock.close()


def setup_cgroup(cfg: dict[str, Any]) -> int:
    """Create the ``cmd`` leaf with the caps and return an fd that moves a process into it.

    PID 1 and the agent stay in the root cgroup. The agent (sandbox user) may write
    ``cgroup.procs`` of the root and of ``cmd``; the bind mount below then hides the real root
    from every path lookup, so commands only ever see the ``cmd`` leaf as ``/sys/fs/cgroup``.
    """
    root = "/sys/fs/cgroup"
    mount("cgroup2", root, "cgroup2", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    write_file(root + "/cgroup.subtree_control", "+cpu +memory +pids")
    leaf = root + "/cmd"
    os.mkdir(leaf)
    period = 100000
    write_file(leaf + "/cpu.max", f"{max(int(cfg['cpu'] * period), 1000)} {period}")
    write_file(leaf + "/memory.max", str(int(cfg["mem"]) * 1024 * 1024))
    try:
        write_file(leaf + "/memory.swap.max", "0")
    except OSError:  # no swap support in the kernel: nothing to cap
        pass
    write_file(leaf + "/pids.max", str(int(cfg["pids"])))
    for procs in (root + "/cgroup.procs", leaf + "/cgroup.procs"):
        os.chown(procs, SANDBOX_UID, SANDBOX_UID)
    fd = os.open(leaf + "/cgroup.procs", os.O_WRONLY)
    # Show the leaf as the cgroup root, so `cat /sys/fs/cgroup/cpu.max` reports the caps.
    mount(leaf, root, None, MS_BIND)
    return fd


def init() -> None:
    try:  # root is already read-only; also refuse setuid binaries and device nodes on it
        mount("", "/", None, MS_REMOUNT | MS_BIND | MS_RDONLY | MS_NOSUID | MS_NODEV)
    except OSError as e:
        log(f"root remount failed: {e!r}")
    mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    mount("sysfs", "/sys", "sysfs", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    try:
        mount("devtmpfs", "/dev", "devtmpfs", MS_NOSUID | MS_NOEXEC, "mode=0755")
    except OSError as e:
        if e.errno != errno.EBUSY:  # the kernel may have mounted devtmpfs on /dev already
            raise
    try:  # console for diagnostics
        fd = os.open("/dev/ttyS0", os.O_WRONLY)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    except OSError:
        pass
    cfg = read_config()
    os.makedirs("/dev/shm", exist_ok=True)
    mount("tmpfs", "/dev/shm", "tmpfs", MS_NOSUID | MS_NODEV | MS_NOEXEC, "size=1m,mode=1777")
    mount("tmpfs", "/tmp", "tmpfs", MS_NOSUID | MS_NODEV, f"size={cfg['tmp_mib']}m,mode=1777")
    mount("/dev/vdb", WORK, "ext4", MS_NOSUID | MS_NODEV, "")
    os.chown(WORK, SANDBOX_UID, SANDBOX_UID)
    os.chmod(WORK, 0o700)
    bring_up_loopback()
    cg_fd = setup_cgroup(cfg)
    write_file("/proc/self/oom_score_adj", "-900")  # inherited by the agent, not by commands
    pid = os.fork()
    if pid == 0:
        try:
            if cg_fd != CGROUP_FD:
                os.dup2(cg_fd, CGROUP_FD, inheritable=True)
                os.close(cg_fd)
            os.setgroups([])
            os.setgid(SANDBOX_UID)
            os.setuid(SANDBOX_UID)
            agent(cfg, cgroup=True)
        except BaseException as e:
            log(f"agent failed: {e!r}")
        os._exit(1)
    os.close(cg_fd)
    while True:  # reap orphans; when the agent is gone the sandbox is over
        try:
            done, _ = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        if done == pid:
            break
    poweroff()


def poweroff() -> None:
    try:
        os.sync()
    except OSError:
        pass
    LIBC.reboot(0x01234567)  # LINUX_REBOOT_CMD_RESTART: Firecracker exits on a guest reset
    os._exit(1)


# ---- agent -------------------------------------------------------------------------------


class NoReply(Exception):
    """Drop the connection without an answer (the host reads that as a lost sandbox)."""


def send_header(conn: socket.socket, obj: dict[str, Any]) -> None:
    conn.sendall(json.dumps(obj).encode() + b"\n")


def do_put(conn: socket.socket, f: Any, req: dict[str, Any]) -> None:
    path, size = req["path"], int(req["size"])
    err: OSError | None = None
    fd = -1
    try:
        os.makedirs(os.path.dirname(path) or "/", exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    except OSError as e:
        err = e
    left = size
    while left > 0:  # always consume the payload so the host's send never breaks
        chunk = f.read(min(left, 1 << 20))
        if not chunk:
            raise EOFError("short put")
        left -= len(chunk)
        if err is None:
            try:
                os.write(fd, chunk)
            except OSError as e:
                err = e
    if fd >= 0:
        try:
            os.close(fd)
        except OSError as e:
            err = err or e
    if err is not None:
        send_header(conn, {"ok": False, "errno": err.errno, "msg": str(err)})
    else:
        send_header(conn, {"ok": True})


def do_get(conn: socket.socket, req: dict[str, Any]) -> None:
    try:
        mode = os.stat(req["path"]).st_mode
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(errno.EISDIR, "is a directory")
        if not stat.S_ISREG(mode):  # a fifo or device would block this thread forever
            raise OSError(errno.EINVAL, "not a regular file")
        with open(req["path"], "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size > MAX_GET:
                send_header(conn, {"ok": False, "errno": errno.EFBIG, "msg": "file too large"})
                return
            data = fh.read(MAX_GET + 1)
    except OSError as e:
        send_header(conn, {"ok": False, "errno": e.errno, "msg": str(e)})
        return
    if len(data) > MAX_GET:
        send_header(conn, {"ok": False, "errno": errno.EFBIG, "msg": "file too large"})
        return
    send_header(conn, {"ok": True, "size": len(data)})
    conn.sendall(data)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _pump(proc: subprocess.Popen[bytes], timeout: float) -> tuple[bytearray, bytearray, bool, bool]:
    """Collect stdout/stderr until the command ends. Returns (out, err, truncated, timed_out).

    One thread, no helper threads: nothing here can fail for lack of pids once the command
    exists. Each stream keeps its first MAX_OUTPUT bytes and the rest is read and dropped so
    the command never blocks on a pipe. After the main process exits, output still in the
    pipes is collected for EXIT_GRACE_S; a background process holding the pipes open is left
    running. A timeout kills the command's whole process group.
    """
    assert proc.stdout is not None and proc.stderr is not None
    out, err = bytearray(), bytearray()
    sinks = {proc.stdout.fileno(): out, proc.stderr.fileno(): err}
    cut = False
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)

    def collect(limit: float) -> bool:
        nonlocal cut
        for key, _ in sel.select(max(min(limit, 0.2), 0.0)):
            data = os.read(key.fd, 65536)
            if not data:
                sel.unregister(key.fileobj)
                continue
            sink = sinks[key.fd]
            room = max(MAX_OUTPUT - len(sink), 0)
            sink += data[:room]
            if len(data) > room:
                cut = True
        return bool(sel.get_map())

    deadline = time.monotonic() + timeout
    grace_end: float | None = None
    timed_out = False
    while sel.get_map():
        now = time.monotonic()
        if grace_end is None and proc.poll() is not None:
            grace_end = now + EXIT_GRACE_S
        stop = deadline if grace_end is None else grace_end
        if now >= stop:
            timed_out = grace_end is None
            break
        collect(stop - now)
    if not timed_out and proc.poll() is None:
        try:
            proc.wait(timeout=max(deadline - time.monotonic(), 0.001))
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        _kill_group(proc)
        proc.wait()
        end = time.monotonic() + 2.0  # the kill closes the pipes; take what is left
        while sel.get_map() and time.monotonic() < end:
            collect(end - time.monotonic())
    sel.close()
    return out, err, cut, timed_out


def do_exec(conn: socket.socket, req: dict[str, Any], cfg: dict[str, Any], cgroup: bool) -> None:
    cmd, timeout = req["cmd"], float(req["timeout"])
    env = {"PATH": PATH, "HOME": WORK, "LANG": "C.UTF-8"}
    if cfg.get("proxy_url"):
        env["ITERANT_PROXY_URL"] = cfg["proxy_url"]
    # Join the cmd cgroup (fail closed: a command outside its caps must not run), then become
    # first in line for the OOM killer (the agent runs at -900).
    join = f"echo 0 >&{CGROUP_FD} || exit 125; " if cgroup else ""
    prefix = join + f"exec {CGROUP_FD}>&-; " if cgroup else ""
    wrapper = prefix + 'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec /bin/sh "$@"'
    script: str | None = None
    argv = ["/bin/sh", "-c", wrapper, "sh"]
    try:
        if len(cmd.encode()) > SHORT_CMD_MAX:
            fd, script = tempfile.mkstemp(prefix=".cmd-", dir="/tmp")
            with os.fdopen(fd, "w") as sf:
                sf.write(cmd)
            argv += [script]
        else:
            argv += ["-c", cmd]
        proc = subprocess.Popen(
            argv,
            cwd=WORK,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(CGROUP_FD,) if cgroup else (),
        )
    except OSError as e:  # e.g. the pids limit refused the fork: the command's doing
        if script is not None:
            _unlink(script)
        msg = f"cannot start command: {e}".encode()
        header = {"ok": True, "exit": 126, "timed_out": False, "truncated": False}
        send_header(conn, dict(header, out=0, err=len(msg), wall=0.0))
        conn.sendall(msg)
        return
    # From here on the command is running: never answer ok:false, only a result or silence.
    start = time.monotonic()
    try:
        out, err, cut, timed_out = _pump(proc, timeout)
        out_b, err_b = bytes(out), bytes(err)  # snapshot: header and payload cannot disagree
        rc = proc.returncode
        code = -1 if timed_out else (128 - rc if rc < 0 else rc)
        header = {"ok": True, "exit": code, "timed_out": timed_out, "truncated": cut}
        send_header(
            conn, dict(header, out=len(out_b), err=len(err_b), wall=time.monotonic() - start)
        )
        conn.sendall(out_b)
        conn.sendall(err_b)
    except BaseException as e:
        _kill_group(proc)
        raise NoReply(repr(e)) from e
    finally:
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
        if script is not None:
            _unlink(script)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def handle(conn: socket.socket, cfg: dict[str, Any], cgroup: bool = False) -> None:
    try:
        f = conn.makefile("rb")
        req = json.loads(f.readline(MAX_REQUEST_LINE))
        op = req["op"]
        if op == "ping":
            send_header(conn, {"ok": True})
        elif op == "put":
            do_put(conn, f, req)
        elif op == "get":
            do_get(conn, req)
        elif op == "exec":
            do_exec(conn, req, cfg, cgroup)
        else:
            send_header(conn, {"ok": False, "errno": 0, "msg": "unknown op"})
    except NoReply:
        pass
    except Exception as e:  # a broken request must not take the agent down
        try:
            send_header(conn, {"ok": False, "errno": 0, "msg": repr(e)})
        except OSError:
            pass
    finally:
        conn.close()


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


RELAY_SLOTS = threading.BoundedSemaphore(64)  # a command can flood the local relay port


def relay_conn(local: socket.socket) -> None:
    try:
        _relay_conn(local)
    finally:
        RELAY_SLOTS.release()


def _relay_conn(local: socket.socket) -> None:
    try:
        up = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        up.connect((VMADDR_CID_HOST, PROXY_VSOCK_PORT))
    except OSError:
        local.close()
        return
    t = threading.Thread(target=pipe, args=(up, local), daemon=True)
    t.start()
    pipe(local, up)
    t.join()
    local.close()
    up.close()


def proxy_relay() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PROXY_LOCAL_PORT))
    srv.listen(64)
    while True:
        c, _ = srv.accept()
        if not RELAY_SLOTS.acquire(blocking=False):
            c.close()
            continue
        threading.Thread(target=relay_conn, args=(c,), daemon=True).start()


def agent(cfg: dict[str, Any], cgroup: bool = False) -> None:
    os.chdir(WORK)
    if cfg.get("proxy_url"):
        threading.Thread(target=proxy_relay, daemon=True).start()
    srv = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    srv.bind((VMADDR_CID_ANY, CONTROL_PORT))
    srv.listen(64)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn, cfg, cgroup), daemon=True).start()


if __name__ == "__main__":
    try:
        init()
    except BaseException as e:
        log(f"init failed: {e!r}")
        time.sleep(1)
        poweroff()
