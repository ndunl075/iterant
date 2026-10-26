"""Unit tests: assert the exact runtime argv through a fake runner. No container runtime needed."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import cast

import pytest

from iterant.contracts import SandboxBackend
from iterant.sandbox import (
    ContainerSandbox,
    DockerBackend,
    PodmanBackend,
    RunOutput,
    SandboxError,
    UnsafeContainerError,
    slugify,
)

Handler = Callable[[list[str]], RunOutput]


class FakeRunner:
    def __init__(self, handler: Handler | None = None) -> None:
        self.calls: list[list[str]] = []
        self.stdins: list[bytes | None] = []
        self.handler = handler

    def __call__(
        self,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
        timeout: float | None = None,
        max_output: int = 0,
    ) -> RunOutput:
        self.calls.append(list(argv))
        self.stdins.append(stdin)
        return self.handler(list(argv)) if self.handler else RunOutput(0)

    def find(self, *words: str) -> list[list[str]]:
        return [c for c in self.calls if all(w in c for w in words)]


PROXY = "http://host.containers.internal:9000/v1"
FORBIDDEN = {"-v", "--volume", "--mount", "--privileged", "--pid=host", "--userns=host"}


def make(runner: FakeRunner, **kw: str) -> PodmanBackend:
    return PodmanBackend(PROXY, name_prefix="claude-w1-sandbox-test", runner=runner, **kw)


def opt(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def sandbox_run(runner: FakeRunner) -> list[str]:
    runs = [c for c in runner.calls if c[1] == "run"]
    assert len(runs) == 1
    return runs[0]


def test_backends_satisfy_protocol() -> None:
    assert isinstance(make(FakeRunner()), SandboxBackend)


def test_default_deny_argv() -> None:
    r = FakeRunner()
    make(r).create("img:1", cpu=1.5, mem=256, disk=128, ttl=90.2, net="default-deny")
    argv = sandbox_run(r)
    assert argv[0] == "podman"
    assert argv[argv.index("--user") + 1] == "10001:10001"  # non-root
    assert "--read-only" in argv
    tmpfs = [argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"]
    assert any(t.startswith("/work:") and "size=128m" in t and "nosuid" in t for t in tmpfs)
    assert "--cap-drop=ALL" in argv and "--security-opt=no-new-privileges" in argv
    assert opt(argv, "--cpus") == "1.5"
    assert opt(argv, "--memory") == "256m" and opt(argv, "--memory-swap") == "256m"
    assert opt(argv, "--pids-limit").isdigit()
    assert opt(argv, "--network") == "none"
    assert opt(argv, "--entrypoint") == "sleep"
    assert argv[-2:] == ["img:1", "91"]  # ttl rounded up, image before its args
    assert opt(argv, "--name").startswith("claude-w1-sandbox-test-")
    # default-deny needs no network, no relay
    assert not r.find("network") and len(r.calls) == 1


def test_proxy_only_isolation_argv() -> None:
    r = FakeRunner()
    sb = make(r).create("img:1", 1, 128, 64, 60, net="proxy-only")
    net_create = r.find("network", "create")[0]
    assert "--internal" in net_create  # no gateway/route off the private network
    netname = net_create[-1]
    relay = r.find("create", "--entrypoint")[0]
    assert relay[1] == "create" and opt(relay, "--network") == netname
    assert "TCP:host.containers.internal:9000" in relay
    assert "--read-only" in relay and "--cap-drop=ALL" in relay
    # relay is the only thing that also joins the outer network, and it joins it by name
    connect = r.find("network", "connect")[0]
    assert connect[-2] == "podman" and connect[-1] == opt(relay, "--name")
    argv = sandbox_run(r)
    assert opt(argv, "--network") == netname  # sandbox only ever on the internal net
    assert not r.find("network", "connect", opt(argv, "--name"))
    relay_name = opt(relay, "--name")
    assert f"ITERANT_PROXY_URL=http://{relay_name}:8080/v1" in argv
    assert cast(ContainerSandbox, sb).proxy_url == f"http://{relay_name}:8080/v1"
    # start order: network, relay created, connected, started, then the sandbox
    kinds = [c[1:3] if c[1] == "network" else c[1:2] for c in r.calls]
    assert kinds == [["network", "create"], ["create"], ["network", "connect"], ["start"], ["run"]]


def test_no_host_mounts_or_privilege_anywhere() -> None:
    r = FakeRunner()
    for mode in ("default-deny", "proxy-only"):
        sb = make(r).create("img", 1, 64, 64, 60, net=mode)  # type: ignore[arg-type]
        sb.put("/work/a", b"x")
        sb.get("/work/a")
        sb.exec("true", 5)
        sb.destroy()
    for call in r.calls:
        assert not FORBIDDEN & set(call), call
        assert not any(a.startswith(("--volume", "--mount", "-v")) for a in call), call
        assert "host" not in [opt(call, f) for f in ("--network", "--pid", "--ipc") if f in call]


def test_loopback_proxy_url_is_untouched_and_host_injection_rejected() -> None:
    for bad in (
        "http://a,fork:80",
        "http://a:80;b",
        "ftp://x:1",
        "http://u:p@x:1",
        "http://[::1]:1",
    ):
        with pytest.raises(ValueError):
            PodmanBackend(bad, runner=FakeRunner())


def test_proxy_only_without_proxy_url_raises() -> None:
    with pytest.raises(SandboxError, match="proxy_url"):
        PodmanBackend(runner=FakeRunner()).create("img", 1, 64, 64, 60)


def test_docker_requires_unsafe_flag() -> None:
    with pytest.raises(UnsafeContainerError):
        DockerBackend(PROXY, runner=FakeRunner())
    with pytest.raises(UnsafeContainerError):
        DockerBackend(PROXY, unsafe_container=False, runner=FakeRunner())
    r = FakeRunner()
    DockerBackend(PROXY, unsafe_container=True, runner=r).create("img", 1, 64, 64, 60)
    assert all(c[0] == "docker" for c in r.calls)
    assert r.find("network", "connect")[0][-2] == "bridge"
    assert "host.docker.internal:host-gateway" in r.find("create", "--entrypoint")[0]


@pytest.mark.parametrize(
    "kw",
    [
        {"image": "-v/:/h"},
        {"image": ""},
        {"cpu": 0},
        {"mem": 0},
        {"disk": -1},
        {"ttl": 0},
        {"net": "host"},
    ],
)
def test_create_rejects_bad_args(kw: dict[str, object]) -> None:
    args: dict[str, object] = dict(image="img", cpu=1, mem=64, disk=64, ttl=60, net="default-deny")
    args.update(kw)
    r = FakeRunner()
    with pytest.raises(ValueError):
        make(r).create(**args)  # type: ignore[arg-type]
    assert not r.calls


def test_bad_name_prefix_rejected() -> None:
    with pytest.raises(ValueError):
        PodmanBackend(name_prefix="Bad/Prefix", runner=FakeRunner())


def test_slugify() -> None:
    assert slugify("claude/W1_sandbox-e0fd71") == "claude-w1-sandbox-e0fd71"
    assert slugify("///") == "x"


def test_create_failure_cleans_up_what_it_made() -> None:
    def handler(argv: list[str]) -> RunOutput:
        return RunOutput(1, stderr=b"boom") if argv[1] == "run" else RunOutput(0)

    r = FakeRunner(handler)
    with pytest.raises(SandboxError, match="boom"):
        make(r).create("img", 1, 64, 64, 60)
    rm = [c for c in r.calls if c[1:3] == ["rm", "-f"]]
    assert len(rm) == 2 and r.find("network", "rm")


def test_missing_runtime_binary_is_sandbox_error() -> None:
    def handler(argv: list[str]) -> RunOutput:
        raise FileNotFoundError("podman")

    with pytest.raises(SandboxError, match="cannot run podman"):
        make(FakeRunner(handler)).create("img", 1, 64, 64, 60, net="default-deny")


def test_exec_argv_and_result() -> None:
    def handler(argv: list[str]) -> RunOutput:
        return RunOutput(3, b"out", b"err") if "echo hi" in argv else RunOutput(0)

    r = FakeRunner(handler)
    sb = make(r).create("img", 1, 64, 64, 60, net="default-deny")
    res = sb.exec("echo hi", 5)
    assert (res.exit_code, res.stdout, res.stderr, res.timed_out) == (3, "out", "err", False)
    c = r.find("echo hi")[0]
    assert c[:2] == ["podman", "exec"] and c[-3:] == ["sh", "-c", "echo hi"]  # never a host shell
    with pytest.raises(ValueError):
        sb.exec("true", 0)


def test_exec_timeout_kills_inside_container() -> None:
    def handler(argv: list[str]) -> RunOutput:
        return RunOutput(-1, b"part", b"", timed_out=True) if "sleep 99" in argv else RunOutput(0)

    r = FakeRunner(handler)
    sb = make(r).create("img", 1, 64, 64, 60, net="default-deny")
    res = sb.exec("sleep 99", 1)
    assert res.timed_out and res.exit_code == -1 and res.stdout == "part"
    assert r.calls[-1][-1] == "kill -9 -1"


def test_put_get_argv_and_errors() -> None:
    def handler(argv: list[str]) -> RunOutput:
        if "/missing" in argv:
            return RunOutput(44)
        if "/dir" in argv:
            return RunOutput(45)
        if "/big" in argv:
            return RunOutput(0, b"x", truncated=True)
        if "/ro/f" in argv:
            return RunOutput(1, stderr=b"Read-only file system")
        return RunOutput(0, b"\x00\xffdata")

    r = FakeRunner(handler)
    sb = make(r).create("img", 1, 64, 64, 60, net="default-deny")
    sb.put("/work/x y/$(z)", b"bytes")
    put = r.find("-i")[0]
    assert put[-1] == "/work/x y/$(z)" and r.stdins[-1] == b"bytes"  # path is an argument
    assert sb.get("/work/f") == b"\x00\xffdata"
    with pytest.raises(FileNotFoundError):
        sb.get("/missing")
    with pytest.raises(IsADirectoryError):
        sb.get("/dir")
    with pytest.raises(SandboxError, match="cap"):
        sb.get("/big")
    with pytest.raises(SandboxError, match="Read-only"):
        sb.put("/ro/f", b"")
    with pytest.raises(ValueError):
        sb.put("relative", b"")
    with pytest.raises(ValueError):
        sb.get("relative")


def test_destroy_idempotent_and_blocks_use() -> None:
    r = FakeRunner()
    sb = make(r).create("img", 1, 64, 64, 60, net="proxy-only")
    before = len(r.calls)
    sb.destroy()
    sb.destroy()
    teardown = r.calls[before:]
    assert [c[1:3] for c in teardown] == [["rm", "-f"], ["rm", "-f"], ["network", "rm"]]
    assert teardown[-1][-1].endswith("-net")
    for fn in (lambda: sb.exec("true", 1), lambda: sb.put("/a", b""), lambda: sb.get("/a")):
        with pytest.raises(SandboxError):
            fn()


def test_destroy_tolerates_already_gone_but_reports_real_errors() -> None:
    def gone(argv: list[str]) -> RunOutput:
        return (
            RunOutput(1, stderr=b"Error: no container with name or ID")
            if argv[1] == "rm"
            else RunOutput(0)
        )

    sb = make(FakeRunner(gone)).create("img", 1, 64, 64, 60, net="default-deny")
    sb.destroy()  # ttl --rm already removed it: fine

    fail = {"on": False}

    def handler(argv: list[str]) -> RunOutput:
        return (
            RunOutput(1, stderr=b"permission denied")
            if fail["on"] and argv[1] == "rm"
            else RunOutput(0)
        )

    sb2 = make(FakeRunner(handler)).create("img", 1, 64, 64, 60, net="default-deny")
    fail["on"] = True
    with pytest.raises(SandboxError, match="permission denied"):
        sb2.destroy()
    fail["on"] = False
    sb2.destroy()  # retry succeeds


def test_ttl_destroys_unused_sandbox() -> None:
    r = FakeRunner()
    sb = make(r).create("img", 1, 64, 64, 0.1, net="default-deny")
    deadline = time.monotonic() + 5
    while not r.find("rm", "-f") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert r.find("rm", "-f")
    with pytest.raises(SandboxError):
        sb.exec("true", 1)


def test_use_after_ttl_refused_even_before_timer_fires() -> None:
    sb = make(FakeRunner()).create("img", 1, 64, 64, 0.05, net="default-deny")
    time.sleep(0.1)
    with pytest.raises(SandboxError):
        sb.exec("true", 1)
