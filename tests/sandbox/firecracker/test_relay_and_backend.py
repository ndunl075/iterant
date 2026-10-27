"""ProxyRelay limits, the vsock handshake and backend construction/cleanup (Linux only: they
use unix sockets and a stub firecracker script; no KVM needed)."""

from __future__ import annotations

import hashlib
import os
import socket
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from iterant.sandbox import FirecrackerBackend, SandboxError
from iterant.sandbox.firecracker import backend as fc_backend
from iterant.sandbox.firecracker import rootfs
from iterant.sandbox.firecracker.sandbox import ProxyRelay, connect_vsock_uds

linux_only = pytest.mark.skipif(sys.platform == "win32", reason="needs unix sockets")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="fcu-"))  # AF_UNIX paths are short
    yield d
    for f in d.glob("*"):
        f.unlink()
    d.rmdir()


class Upstream:
    """TCP server that counts live connections and never answers."""

    def __init__(self) -> None:
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(128)
        self.port: int = self.srv.getsockname()[1]
        self.conns: list[socket.socket] = []
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            self.conns.append(c)

    def alive(self) -> int:
        n = 0
        for c in self.conns:
            c.setblocking(False)
            try:
                if c.recv(1) != b"":
                    n += 1
            except BlockingIOError:
                n += 1
            except OSError:
                pass
        return n

    def close(self) -> None:
        self.srv.close()
        for c in self.conns:
            c.close()


def unix_client(path: str) -> socket.socket:
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # pyright: ignore[reportAttributeAccessIssue]
    c.connect(path)
    return c


def wait_for(cond: Callable[[], bool], seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@linux_only
def test_relay_bounds_concurrent_connections(short_dir: Path) -> None:
    up = Upstream()
    path = str(short_dir / "r.sock")
    relay = ProxyRelay(path, "127.0.0.1", up.port, max_conns=3, idle_s=60)
    clients = [unix_client(path) for _ in range(10)]
    try:
        assert wait_for(lambda: len(up.conns) >= 3)
        time.sleep(0.5)
        assert len(up.conns) == 3  # the rest were refused, not queued
        refused = 0
        for c in clients:
            c.settimeout(2)
            try:
                if c.recv(1) == b"":
                    refused += 1
            except OSError:
                pass
        assert refused >= 7
    finally:
        for c in clients:
            c.close()
        relay.close()
        up.close()


@linux_only
def test_relay_frees_a_slot_when_a_connection_ends(short_dir: Path) -> None:
    up = Upstream()
    path = str(short_dir / "r.sock")
    relay = ProxyRelay(path, "127.0.0.1", up.port, max_conns=1, idle_s=60)
    try:
        first = unix_client(path)
        assert wait_for(lambda: len(up.conns) == 1)
        first.close()
        assert wait_for(lambda: up.alive() == 0)
        second = unix_client(path)
        assert wait_for(lambda: len(up.conns) == 2)
        second.close()
    finally:
        relay.close()
        up.close()


@linux_only
def test_relay_closes_idle_connections(short_dir: Path) -> None:
    up = Upstream()
    path = str(short_dir / "r.sock")
    relay = ProxyRelay(path, "127.0.0.1", up.port, max_conns=4, idle_s=0.5)
    try:
        c = unix_client(path)
        c.settimeout(5)
        assert c.recv(1) == b""  # closed by the relay after the idle timeout
        assert wait_for(lambda: up.alive() == 0)
    finally:
        relay.close()
        up.close()


@linux_only
def test_relay_pipes_both_ways(short_dir: Path) -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def echo() -> None:
        c, _ = srv.accept()
        c.sendall(c.recv(100).upper())
        c.close()

    threading.Thread(target=echo, daemon=True).start()
    path = str(short_dir / "r.sock")
    relay = ProxyRelay(path, "127.0.0.1", srv.getsockname()[1])
    try:
        c = unix_client(path)
        c.sendall(b"hello")
        c.settimeout(5)
        assert c.recv(100) == b"HELLO"
    finally:
        relay.close()
        srv.close()


@linux_only
def test_relay_close_stops_the_accept_thread_and_removes_the_socket(short_dir: Path) -> None:
    before = threading.active_count()
    relays = [ProxyRelay(str(short_dir / f"r{i}.sock"), "127.0.0.1", 9) for i in range(5)]
    assert threading.active_count() >= before + 5
    for r in relays:
        r.close()
    assert wait_for(lambda: threading.active_count() <= before)
    assert not list(short_dir.glob("r*.sock"))


@linux_only
def test_non_ok_handshake_is_refused(short_dir: Path) -> None:
    path = str(short_dir / "v.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # pyright: ignore[reportAttributeAccessIssue]
    srv.bind(path)
    srv.listen(1)

    def serve() -> None:
        c, _ = srv.accept()
        c.recv(100)
        c.sendall(b"FAILED nobody listens\n")
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    try:
        with pytest.raises(ConnectionRefusedError):
            connect_vsock_uds(path, time.monotonic() + 5)
    finally:
        srv.close()


# -- backend construction and cleanup ------------------------------------------------------


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_backend_refuses_unpinned_artifacts(tmp_path: Path) -> None:
    fc, kernel = tmp_path / "firecracker", tmp_path / "vmlinux"
    fc.write_bytes(b"x")
    kernel.write_bytes(b"y")
    os.chmod(fc, 0o755)
    pins = (sha(fc), sha(kernel))
    FirecrackerBackend(firecracker_bin=str(fc), kernel=str(kernel), pins=pins)  # accepted
    with pytest.raises(SandboxError, match="pinned sha256"):
        FirecrackerBackend(firecracker_bin=str(fc), kernel=str(kernel), pins=("0" * 64, pins[1]))
    with pytest.raises(SandboxError, match="pinned sha256"):
        FirecrackerBackend(firecracker_bin=str(fc), kernel=str(kernel), pins=(pins[0], "0" * 64))
    with pytest.raises(SandboxError, match="not found"):
        FirecrackerBackend(firecracker_bin=str(tmp_path / "missing"), kernel=str(kernel), pins=pins)


def test_backend_has_no_path_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(fc_backend.BIN_ENV, raising=False)
    monkeypatch.delenv(fc_backend.KERNEL_ENV, raising=False)
    with pytest.raises(SandboxError, match="not found"):
        FirecrackerBackend()


@linux_only
def test_create_cleans_up_when_the_vm_dies_during_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fc, kernel = tmp_path / "firecracker", tmp_path / "vmlinux"
    fc.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    kernel.write_bytes(b"k")
    os.chmod(fc, stat.S_IRWXU)
    base = tmp_path / "base.ext4"
    base.write_bytes(b"")
    monkeypatch.setattr(fc_backend, "kvm_usable", lambda: True)
    monkeypatch.setattr(rootfs, "build_rootfs", lambda *a, **k: base)
    monkeypatch.setattr(rootfs, "make_ext4", lambda target, *a, **k: Path(target).write_bytes(b""))
    root = Path(tempfile.mkdtemp(prefix="fcb-"))
    try:
        backend = FirecrackerBackend(
            "http://127.0.0.1:9/", firecracker_bin=str(fc), kernel=str(kernel),
            pins=(sha(fc), sha(kernel)), sandbox_root=str(root), container_runtime="docker",
            boot_timeout=10,
        )  # fmt: skip
        before = threading.active_count()
        with pytest.raises(SandboxError, match="boom"):
            backend.create("img", 1, 64, 16, 60, "proxy-only")
        assert list(root.iterdir()) == []  # sandbox dir, sockets and logs are gone
        assert wait_for(lambda: threading.active_count() <= before)  # relay thread too
    finally:
        for f in root.glob("*"):
            f.unlink()
        root.rmdir()
