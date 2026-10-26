"""Backend conformance suite (§5): every sandbox backend must pass all of these.

Parametrized over ``BACKENDS`` below; real-runtime cases skip when the runtime is
missing. Probes run python3 inside the sandbox image.

To run the suite against another backend (e.g. the W2 Firecracker one), add a
``pytest.param`` to ``BACKENDS`` and a branch in the ``harness`` fixture. The backend must
export ``ITERANT_PROXY_URL`` inside ``proxy-only`` sandboxes, and the test image needs python3.
(Everything lives in this file because ``--import-mode=importlib`` cannot share modules.)
"""

from __future__ import annotations

import http.server
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from iterant.contracts import NetMode, Sandbox, SandboxBackend
from iterant.sandbox import DockerBackend, PodmanBackend, slugify

BACKENDS = [
    pytest.param("podman", marks=pytest.mark.podman, id="podman"),
    pytest.param("docker", marks=pytest.mark.docker, id="docker"),
]

TEST_IMAGE = os.environ.get("ITERANT_TEST_IMAGE", "docker.io/library/python:3.12-slim")


def _branch_slug() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return slugify(out.stdout.strip() or "iterant-test")
    except (OSError, subprocess.SubprocessError):
        return "iterant-test"


class Stub:
    """Localhost HTTP stub on an OS-picked port; counts requests."""

    def __init__(self, body: bytes) -> None:
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                stub.hits += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.hits = 0
        # 0.0.0.0 so the relay container can reach it through the host gateway.
        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        self.port: int = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@dataclass
class Harness:
    backend: SandboxBackend
    host_alias: str  # how a container reaches the host
    proxy: Stub
    other: Stub  # a second host service that proxy-only sandboxes must NOT reach
    created: list[Sandbox] = field(default_factory=lambda: [])

    def create(
        self,
        *,
        cpu: float = 1.0,
        mem: int = 256,
        disk: int = 64,
        ttl: float = 300,
        net: NetMode = "default-deny",
        image: str = TEST_IMAGE,
    ) -> Sandbox:
        sb = self.backend.create(image, cpu, mem, disk, ttl, net)
        self.created.append(sb)
        return sb


def _runtime_works(runtime: str) -> bool:
    if shutil.which(runtime) is None:
        return False
    try:
        out = subprocess.run([runtime, "info"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


@pytest.fixture(params=BACKENDS)
def harness(request: pytest.FixtureRequest) -> Iterator[Harness]:
    kind: str = request.param
    if not _runtime_works(kind):
        pytest.skip(f"{kind} is not installed or not usable here")
    proxy, other = Stub(b"proxy-stub"), Stub(b"other-stub")
    prefix = _branch_slug()
    if kind == "podman":
        alias = "host.containers.internal"
        backend: SandboxBackend = PodmanBackend(f"http://{alias}:{proxy.port}/", name_prefix=prefix)
    else:
        alias = "host.docker.internal"
        backend = DockerBackend(
            f"http://{alias}:{proxy.port}/", unsafe_container=True, name_prefix=prefix
        )
    h = Harness(backend, alias, proxy, other)
    try:
        yield h
    finally:
        for sb in h.created:
            try:
                sb.destroy()
            except Exception:  # keep tearing down the rest
                pass
        proxy.close()
        other.close()


CONNECT = """
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3).close()
    print("open")
except OSError:
    print("blocked")
"""

FETCH = """
import os, sys, time, urllib.request
url = os.environ.get("ITERANT_PROXY_URL", "")
deadline = time.time() + 15
while True:
    try:
        print(urllib.request.urlopen(url, timeout=3).read().decode())
        break
    except OSError as e:
        if time.time() > deadline:
            print("unreachable", e)
            break
        time.sleep(0.5)
"""


def probe(sb: Sandbox, code: str, *args: str, timeout: float = 60) -> str:
    sb.put("/work/probe.py", code.encode())
    res = sb.exec("python3 /work/probe.py " + " ".join(args), timeout)
    assert not res.timed_out, res
    return res.stdout.strip()


def test_runs_as_non_root(harness: Harness) -> None:
    sb = harness.create()
    res = sb.exec("id -u", 30)
    assert res.exit_code == 0 and res.stdout.strip() not in ("", "0")


def test_put_get_roundtrip(harness: Harness) -> None:
    sb = harness.create()
    data = bytes(range(256)) * 64
    sb.put("/work/deep/er/file.bin", data)
    assert sb.get("/work/deep/er/file.bin") == data
    with pytest.raises(FileNotFoundError):
        sb.get("/work/nope")


def test_no_host_files_visible(harness: Harness, tmp_path: Path) -> None:
    marker = tmp_path / "host-secret.txt"
    marker.write_text("secret")
    sb = harness.create()
    assert sb.exec(f"test -e '{marker.as_posix()}'", 30).exit_code != 0
    mounts = sb.exec("cat /proc/self/mountinfo", 30).stdout
    for bad in ("/home", "/host", "docker.sock", "podman.sock", str(tmp_path)):
        assert bad not in mounts, bad


def test_write_outside_work_dir_fails(harness: Harness) -> None:
    sb = harness.create()
    assert sb.exec("touch /work/ok", 30).exit_code == 0
    assert sb.exec("touch /usr/x", 30).exit_code != 0
    assert sb.exec("touch /etc/x", 30).exit_code != 0
    with pytest.raises(RuntimeError):
        sb.put("/usr/x", b"x")


def test_disk_cap(harness: Harness) -> None:
    sb = harness.create(disk=16)
    assert sb.exec("dd if=/dev/zero of=/work/big bs=1M count=8", 60).exit_code == 0
    assert sb.exec("dd if=/dev/zero of=/work/big2 bs=1M count=64", 60).exit_code != 0


def test_memory_cap(harness: Harness) -> None:
    sb = harness.create(mem=64)
    res = sb.exec("python3 -c \"b = b'x' * (400 << 20); print(len(b))\"", 60)
    assert res.exit_code != 0 and not res.timed_out
    assert sb.exec("true", 30).exit_code == 0  # sandbox survives the OOM kill


def test_cpu_and_pids_caps(harness: Harness) -> None:
    sb = harness.create(cpu=0.5)
    cpu = sb.exec("cat /sys/fs/cgroup/cpu.max", 30)
    if cpu.exit_code != 0:
        pytest.skip("cgroup v2 cpu.max not visible in the sandbox")
    quota, period = cpu.stdout.split()
    assert quota != "max" and int(quota) / int(period) == pytest.approx(0.5, abs=0.01)
    pids = sb.exec("cat /sys/fs/cgroup/pids.max", 30).stdout.strip()
    assert pids.isdigit() and int(pids) <= 4096


def test_exec_timeout_kills_the_process(harness: Harness) -> None:
    sb = harness.create()
    start = time.monotonic()
    res = sb.exec("sleep 3; echo late > /work/late", 1)
    assert res.timed_out and res.exit_code == -1
    assert time.monotonic() - start < 3
    time.sleep(4)
    with pytest.raises(FileNotFoundError):
        sb.get("/work/late")  # the process was killed, not just abandoned
    assert sb.exec("echo alive", 30).stdout.strip() == "alive"


def test_ttl_expiry_destroys_unused_sandbox(harness: Harness) -> None:
    sb = harness.create(ttl=3)
    assert sb.exec("true", 30).exit_code == 0
    time.sleep(5)
    with pytest.raises(RuntimeError):
        sb.exec("true", 5)


def test_destroy_is_idempotent(harness: Harness) -> None:
    sb = harness.create(net="proxy-only")
    sb.destroy()
    sb.destroy()
    with pytest.raises(RuntimeError):
        sb.exec("true", 5)


def test_default_deny_has_no_network(harness: Harness) -> None:
    sb = harness.create(net="default-deny")
    assert sb.exec("ls /sys/class/net", 30).stdout.split() == ["lo"]
    for host, port in ((harness.host_alias, harness.proxy.port), ("1.1.1.1", 443)):
        assert probe(sb, CONNECT, host, str(port)) == "blocked"
    assert harness.proxy.hits == 0


def test_proxy_only_reaches_only_the_proxy(harness: Harness) -> None:
    sb = harness.create(net="proxy-only")
    assert probe(sb, FETCH) == "proxy-stub"
    assert harness.proxy.hits >= 1
    # Same host, different port: not the proxy, so unreachable. Nor is the internet.
    assert probe(sb, CONNECT, harness.host_alias, str(harness.other.port)) == "blocked"
    assert probe(sb, CONNECT, "1.1.1.1", "443") == "blocked"
    assert harness.other.hits == 0
