"""Guest-side init and exec agent, copied into the microVM rootfs as ``/.iterant/guest.py``.

Runs INSIDE the guest (Linux, python3 from the sandbox image, stdlib only; keep it free of
syntax newer than Python 3.8). The host code only reads its text; it never imports it.

PID 1 (root) mounts the pseudo file systems, the ``/work`` disk and a cgroup-v2 leaf carrying
the cpu/memory/pids caps, then forks the agent and drops it to the sandbox user. If the agent
dies (a command killed it), PID 1 powers the VM off: the host sees the VM gone and reports
``sandbox_lost``. The agent serves one request per vsock connection:

    request  : one JSON line, then ``size`` raw bytes for ``put``
    response : one JSON line, then raw bytes named by the header (``get`` data, exec output)

Everything the agent says is untrusted by the host: a command can kill or replace it.
"""

# The guest is Linux-only; the host's type checker may run on another platform.
# pyright: basic
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import base64
import ctypes
import errno
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any

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


LIBC: Any = _libc() if sys.platform == "linux" else None


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


def setup_cgroup(cfg: dict[str, Any]) -> None:
    root = "/sys/fs/cgroup"
    mount("cgroup2", root, "cgroup2", MS_NOSUID | MS_NODEV | MS_NOEXEC)
    write_file(root + "/cgroup.subtree_control", "+cpu +memory +pids")
    leaf = root + "/sbx"
    os.mkdir(leaf)
    period = 100000
    write_file(leaf + "/cpu.max", f"{max(int(cfg['cpu'] * period), 1000)} {period}")
    write_file(leaf + "/memory.max", str(int(cfg["mem"]) * 1024 * 1024))
    try:
        write_file(leaf + "/memory.swap.max", "0")
    except OSError:  # no swap support in the kernel: nothing to cap
        pass
    write_file(leaf + "/pids.max", str(int(cfg["pids"])))
    write_file(leaf + "/cgroup.procs", str(os.getpid()))
    # Show the leaf as the cgroup root, so `cat /sys/fs/cgroup/cpu.max` reports the caps.
    mount(leaf, root, None, MS_BIND)


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
    setup_cgroup(cfg)
    write_file("/proc/self/oom_score_adj", "-900")  # inherited by the agent, not by commands
    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(SANDBOX_UID)
            os.setuid(SANDBOX_UID)
            agent(cfg)
        except BaseException as e:
            log(f"agent failed: {e!r}")
        os._exit(1)
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


def drain(stream: Any, sink: bytearray, cut: list[bool]) -> None:
    # Keep the first MAX_OUTPUT bytes and keep reading so the command never blocks on a pipe.
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        room = max(MAX_OUTPUT - len(sink), 0)
        sink += chunk[:room]
        if len(chunk) > room:
            cut[0] = True


def do_exec(conn: socket.socket, req: dict[str, Any], cfg: dict[str, Any]) -> None:
    cmd, timeout = req["cmd"], float(req["timeout"])
    env = {"PATH": PATH, "HOME": WORK, "LANG": "C.UTF-8"}
    if cfg.get("proxy_url"):
        env["ITERANT_PROXY_URL"] = cfg["proxy_url"]
    # Commands are first in line for the OOM killer (the agent runs at -900).
    wrapper = 'echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec /bin/sh -c "$0"'
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            ["/bin/sh", "-c", wrapper, cmd],
            cwd=WORK,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:  # e.g. the pids limit refused the fork: the command's doing
        msg = f"cannot start command: {e}".encode()
        header = {"ok": True, "exit": 126, "timed_out": False, "truncated": False}
        send_header(conn, dict(header, out=0, err=len(msg), wall=0.0))
        conn.sendall(msg)
        return
    out, err = bytearray(), bytearray()
    out_cut, err_cut = [False], [False]
    readers = [
        threading.Thread(target=drain, args=(proc.stdout, out, out_cut), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, err, err_cut), daemon=True),
    ]
    for t in readers:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.wait()
    for t in readers:
        t.join(timeout=5)
    rc = proc.returncode
    code = -1 if timed_out else (128 - rc if rc < 0 else rc)
    header = {"ok": True, "exit": code, "timed_out": timed_out}
    header["truncated"] = out_cut[0] or err_cut[0]
    send_header(conn, dict(header, out=len(out), err=len(err), wall=time.monotonic() - start))
    conn.sendall(bytes(out))
    conn.sendall(bytes(err))


def handle(conn: socket.socket, cfg: dict[str, Any]) -> None:
    try:
        f = conn.makefile("rb")
        req = json.loads(f.readline(65536))
        op = req["op"]
        if op == "ping":
            send_header(conn, {"ok": True})
        elif op == "put":
            do_put(conn, f, req)
        elif op == "get":
            do_get(conn, req)
        elif op == "exec":
            do_exec(conn, req, cfg)
        else:
            send_header(conn, {"ok": False, "errno": 0, "msg": "unknown op"})
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


def relay_conn(local: socket.socket) -> None:
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
        threading.Thread(target=relay_conn, args=(c,), daemon=True).start()


def agent(cfg: dict[str, Any]) -> None:
    os.chdir(WORK)
    if cfg.get("proxy_url"):
        threading.Thread(target=proxy_relay, daemon=True).start()
    srv = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    srv.bind((VMADDR_CID_ANY, CONTROL_PORT))
    srv.listen(64)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn, cfg), daemon=True).start()


if __name__ == "__main__":
    try:
        init()
    except BaseException as e:
        log(f"init failed: {e!r}")
        time.sleep(1)
        poweroff()
