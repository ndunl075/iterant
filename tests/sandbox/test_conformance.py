"""Backend conformance suite (§5): every sandbox backend must pass all of these.

Parametrized over ``BACKENDS`` below; real-runtime cases skip when the runtime is
missing. Probes run python3 inside the sandbox image.

To run the suite against another backend (e.g. the W2 Firecracker one), add a
``pytest.param`` to ``BACKENDS`` and a branch in the ``harness`` fixture. The backend must
export ``ITERANT_PROXY_URL`` inside ``proxy-only`` sandboxes, and the test image needs python3.
(Everything lives in this file because ``--import-mode=importlib`` cannot share modules.)
"""

from __future__ import annotations

import getpass
import http.server
import ipaddress
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from iterant.contracts import NetMode, Sandbox, SandboxBackend
from iterant.sandbox import ContainerSandbox, DockerBackend, PodmanBackend, SandboxLost, slugify

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
    runtime: str  # CLI used to inspect what the backend left behind
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

    def cli(self, *args: str) -> str:
        out = subprocess.run(
            [self.runtime, *args], capture_output=True, text=True, timeout=60, check=True
        )
        return out.stdout

    def rid(self, sb: Sandbox) -> str:
        return cast(ContainerSandbox, sb).resource_id

    def leftovers(self, sb: Sandbox) -> list[str]:
        """Containers (sandbox, relay) and networks the runtime still has for this sandbox."""
        sel = ["--filter", f"label=iterant.sandbox.id={self.rid(sb)}"]
        return (
            self.cli("ps", "-a", "--format", "{{.Names}}", *sel).split()
            + self.cli("network", "ls", "--format", "{{.Name}}", *sel).split()
        )

    def wait_gone(self, sb: Sandbox, within: float) -> list[str]:
        deadline = time.monotonic() + within
        left = self.leftovers(sb)
        while left and time.monotonic() < deadline:
            time.sleep(0.5)
            left = self.leftovers(sb)
        return left

    def addresses(self, sb: Sandbox) -> dict[str, str]:
        """Internal-net gateway plus the relay's outer-net address and gateway (the host side)."""
        rid = self.rid(sb)
        net = json.loads(self.cli("network", "inspect", f"{rid}-net"))[0]
        subnets = net.get("subnets") or net["IPAM"]["Config"]  # podman / docker
        first = subnets[0]
        cidr = ipaddress.ip_network(first.get("subnet") or first["Subnet"])
        internal_gw = first.get("gateway") or first.get("Gateway") or str(cidr[1])
        relay = json.loads(self.cli("container", "inspect", f"{rid}-relay"))[0]
        nets = relay["NetworkSettings"]["Networks"]
        outer = next(v for k, v in nets.items() if k != f"{rid}-net")
        return {
            "internal_gw": internal_gw,
            "relay_outer_ip": outer["IPAddress"],
            "outer_gw": outer["Gateway"],
        }


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
    h = Harness(backend, kind, alias, proxy, other)
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
except socket.gaierror:
    print("unresolved")  # never counts as "blocked": the connection was not even attempted
except OSError:
    print("blocked")
"""

RESOLVE = """
import socket, sys
try:
    socket.getaddrinfo(sys.argv[1], 80)
    print("resolved")
except OSError:
    print("failed")
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


def test_work_dir_owned_by_sandbox_user_with_mode_0700(harness: Harness) -> None:
    sb = harness.create(disk=64)
    res = sb.exec("stat -c '%u %a' /work", 30)
    assert res.exit_code == 0 and res.stdout.strip() == "10001 700"
    assert sb.exec("touch /work/ok && rm /work/ok", 30).exit_code == 0


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
    assert sb.exec(f"ls -A /home | grep -qx '{getpass.getuser()}'", 30).exit_code != 0
    for sock in ("/var/run/docker.sock", "/run/docker.sock", "/run/podman/podman.sock"):
        assert sb.exec(f"test -e {sock}", 30).exit_code != 0, sock


def test_write_outside_work_dir_fails(harness: Harness) -> None:
    sb = harness.create()
    assert sb.exec("touch /work/ok", 30).exit_code == 0
    for path in ("/usr/x", "/etc/x", "/var/tmp/x", "/run/x"):
        assert sb.exec(f"touch {path}", 30).exit_code != 0, path
    with pytest.raises(RuntimeError):
        sb.put("/usr/x", b"x")


def test_disk_cap_covers_every_writable_path(harness: Harness) -> None:
    sb = harness.create(disk=16)
    assert sb.exec("dd if=/dev/zero of=/work/big bs=1M count=8", 60).exit_code == 0
    assert sb.exec("dd if=/dev/zero of=/work/big2 bs=1M count=64", 60).exit_code != 0
    assert sb.exec("dd if=/dev/zero of=/dev/shm/big bs=1M count=8", 60).exit_code != 0
    assert sb.exec("dd if=/dev/zero of=/tmp/big bs=1M count=100", 60).exit_code != 0


def test_memory_cap(harness: Harness) -> None:
    sb = harness.create(mem=64)
    res = sb.exec("python3 -c \"b = b'x' * (400 << 20); print(len(b))\"", 60)
    assert res.exit_code != 0 and not res.timed_out
    assert sb.exec("true", 30).exit_code == 0  # sandbox survives the OOM kill


def test_truncated_stdout_is_flagged(harness: Harness) -> None:
    sb = harness.create()
    res = sb.exec("python3 -c \"import sys; sys.stdout.write('x' * (20 << 20))\"", 60)
    assert res.truncated and not res.timed_out and res.exit_code == 0
    assert 0 < len(res.stdout) <= 16 << 20
    small = sb.exec("echo hi", 30)
    assert small.stdout == "hi\n" and not small.truncated


FORK_UNTIL_REFUSED = """
import os, sys, time
for _ in range(5000):
    try:
        pid = os.fork()
    except OSError:
        sys.exit(42)  # the pids limit refused the fork
    if pid == 0:
        time.sleep(8)
        os._exit(0)
sys.exit(0)
"""


def test_pids_exhaustion_is_a_result_not_an_error(harness: Harness) -> None:
    sb = harness.create()
    sb.put("/work/forks.py", FORK_UNTIL_REFUSED.encode())
    res = sb.exec("python3 /work/forks.py", 60)  # must not raise
    assert res.exit_code == 42 and not res.timed_out
    time.sleep(10)  # let the children exit
    assert sb.exec("echo alive", 30).stdout.strip() == "alive"


def test_killing_pid_1_is_a_lost_sandbox_not_an_error(harness: Harness) -> None:
    sb = harness.create()
    # Kill everything we may signal, including the container's main process (`sleep`), so
    # the init process exits and the container goes away.
    res = sb.exec("kill -9 1; kill -9 -1; sleep 5", 30)  # must not raise
    assert res.sandbox_lost and not res.timed_out and res.exit_code != 0
    for call in (lambda: sb.exec("true", 10), lambda: sb.put("/work/f", b""), lambda: sb.get("/f")):
        with pytest.raises(SandboxLost):
            call()
    sb.destroy()  # idempotent and quiet for a container that is already gone
    sb.destroy()


def test_oom_is_a_result_not_an_error(harness: Harness) -> None:
    sb = harness.create(mem=64)
    # Several processes each take memory, so the cgroup OOM killer picks among many.
    script = (
        "for i in 1 2 3 4; do python3 -c \"b = b'x' * (200 << 20); print(len(b))\" & done; wait"
    )
    res = sb.exec(script, 60)  # must not raise
    assert not res.timed_out and not res.truncated
    assert res.stdout.split().count("209715200") < 4  # the 64 MiB cap stopped some of them


def test_cpu_and_pids_caps_are_applied(harness: Harness) -> None:
    sb = harness.create(cpu=0.5)  # create() itself fails if the caps did not apply
    quota, period = sb.exec("cat /sys/fs/cgroup/cpu.max", 30).stdout.split()
    assert quota != "max" and int(quota) / int(period) == pytest.approx(0.5, abs=0.01)
    pids = sb.exec("cat /sys/fs/cgroup/pids.max", 30).stdout.strip()
    assert pids.isdigit() and int(pids) <= 4096


def test_exec_timeout_kills_the_whole_process_group(harness: Harness) -> None:
    sb = harness.create()
    start = time.monotonic()
    res = sb.exec("(sleep 3; echo late > /work/late) & sleep 3; echo late2 > /work/late2", 1)
    assert res.timed_out and res.exit_code == -1
    assert time.monotonic() - start < 3
    time.sleep(4)
    for f in ("/work/late", "/work/late2"):
        with pytest.raises(FileNotFoundError):
            sb.get(f)  # killed, not just abandoned (including the backgrounded child)
    assert sb.exec("echo alive", 30).stdout.strip() == "alive"


def test_exec_timeout_does_not_touch_concurrent_execs(harness: Harness) -> None:
    sb = harness.create()
    box: list[str] = []
    t = threading.Thread(target=lambda: box.append(sb.exec("sleep 4; echo done", 30).stdout))
    t.start()
    time.sleep(1)
    assert sb.exec("sleep 30", 1).timed_out
    t.join(timeout=30)
    assert box == ["done\n"]


def test_ttl_expiry_removes_container_relay_and_network(harness: Harness) -> None:
    sb = harness.create(ttl=5, net="proxy-only")
    assert len(harness.leftovers(sb)) == 3  # sandbox, relay, internal network
    assert harness.wait_gone(sb, within=40) == []
    with pytest.raises(RuntimeError):
        sb.exec("true", 5)


def test_destroy_removes_everything_and_is_idempotent(harness: Harness) -> None:
    sb = harness.create(net="proxy-only")
    assert len(harness.leftovers(sb)) == 3
    sb.destroy()
    assert harness.leftovers(sb) == []
    sb.destroy()
    with pytest.raises(RuntimeError):
        sb.exec("true", 5)


def test_default_deny_has_no_network(harness: Harness) -> None:
    sb = harness.create(net="default-deny")
    assert sb.exec("ls /sys/class/net", 30).stdout.split() == ["lo"]
    for host, port in (("1.1.1.1", 443), ("169.254.169.254", 80)):
        assert probe(sb, CONNECT, host, str(port)) == "blocked"
    assert probe(sb, RESOLVE, "example.com") == "failed"


def test_proxy_only_reaches_only_the_proxy(harness: Harness) -> None:
    sb = harness.create(net="proxy-only")
    assert probe(sb, FETCH) == "proxy-stub"
    assert harness.proxy.hits >= 1
    addr = harness.addresses(sb)
    targets = [
        (addr["internal_gw"], harness.other.port),  # bridge gateway = the host
        (addr["internal_gw"], harness.proxy.port),
        (addr["outer_gw"], harness.other.port),
        (addr["outer_gw"], harness.proxy.port),
        (addr["relay_outer_ip"], harness.proxy.port),  # the relay, via its other network
        (addr["relay_outer_ip"], 22),
        ("169.254.169.254", 80),  # cloud metadata
        ("1.1.1.1", 443),
    ]
    for host, port in targets:
        assert probe(sb, CONNECT, host, str(port)) == "blocked", (host, port)
    assert harness.other.hits == 0
    # DNS on the internal net must not forward upstream (an exfiltration channel).
    # Control: the same lookup works from a container on the default network, otherwise
    # "failed" would only mean this host has no internet.
    ctl = subprocess.run(
        [harness.runtime, "run", "--rm", "--name", f"{harness.rid(sb)}-dnsctl",
         "--label", f"iterant.sandbox.id={harness.rid(sb)}", TEST_IMAGE,
         "python3", "-c", RESOLVE.replace("sys.argv[1]", "'example.com'")],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    if ctl.stdout.strip() != "resolved":
        pytest.skip(
            "control container cannot resolve example.com (no internet): DNS leak untestable"
        )
    assert probe(sb, RESOLVE, "example.com") == "failed"
    url = cast(ContainerSandbox, sb).proxy_url
    assert url is not None
    assert probe(sb, RESOLVE, url.split("/")[2].split(":")[0]) == "resolved"
