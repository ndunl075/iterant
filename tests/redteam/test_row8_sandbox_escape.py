"""Row 8: "Escape the sandbox". Stopped by non-root, no mounts, resource caps, default-deny egress
(containers for dev; microVMs at release).

Two layers. (a) Always-on: the real container backends are driven through a fake runner, and
every runtime command they would issue is checked for escape hatches, including a hostile image
name. (b) ``podman`` / ``docker`` marked: hostile code runs in a real sandbox and the walls are
probed from inside. Those skip cleanly when the runtime is missing.

NOT RUN IN CI: ci.yml deselects `podman`/`docker`/`firecracker` tests and we add no container CI
job ($0 budget). Layer (b) runs only on demand, on a machine with rootless Podman or Docker:

    uv run pytest tests/redteam/test_row8_sandbox_escape.py -m "podman or docker" -q -rs

It was last run on a throwaway `probe/*` branch with a copied `container-probe.yml` workflow
(never merged). There is no Firecracker red-team test yet; the microVM backend is not attacked
here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import Sandbox, SandboxBackend
from iterant.grader import Grader
from iterant.orchestrator import OrchestratorConfig, RunExecutor
from iterant.packs import load_pack
from iterant.sandbox import DockerBackend, PodmanBackend
from tests.redteam.conftest import make_proxy
from tests.sandbox.test_argv import (
    FakeRunner,
    assert_hardened,
    assert_no_escape_flags,
    make,
    opt,
    runs,
)

TEST_IMAGE = os.environ.get("ITERANT_TEST_IMAGE", "docker.io/library/python:3.12-slim")
SYNTHETIC = Path(__file__).resolve().parents[1] / "packs" / "synthetic"

# --- (a) the commands the backend would issue ----------


@pytest.mark.parametrize("mode", ["default-deny", "proxy-only"])
def test_row8_every_sandbox_is_non_root_read_only_capless_and_capped(mode: str) -> None:
    r = FakeRunner()
    make(r).create("img:1", 1, 128, 64, 60, net=mode)  # type: ignore[arg-type]
    sandbox_run = runs(r)[-1]
    assert_hardened(sandbox_run)
    assert opt(sandbox_run, "--user") != "0:0" and not opt(sandbox_run, "--user").startswith("0:")
    assert opt(sandbox_run, "--network") != "host"
    for call in r.calls:
        assert_no_escape_flags(call)


def test_row8_the_work_disk_is_the_only_writable_executable_place() -> None:
    r = FakeRunner()
    make(r).create("img:1", 1, 128, 64, 60, net="default-deny")
    argv = runs(r)[0]
    tmpfs = {argv[i + 1].split(":")[0]: argv[i + 1] for i, a in enumerate(argv) if a == "--tmpfs"}
    assert set(tmpfs) == {"/work", "/tmp"}
    assert "noexec" in tmpfs["/tmp"] and "exec" in tmpfs["/work"].split(",")


@pytest.mark.parametrize(
    "image",
    ["--privileged", "-v", "--cap-add=ALL", "", "-"],
)
def test_row8_an_image_name_cannot_smuggle_a_runtime_flag(image: str) -> None:
    r = FakeRunner()
    with pytest.raises(ValueError):
        make(r).create(image, 1, 64, 64, 60, net="default-deny")
    assert runs(r) == []


@pytest.mark.parametrize(
    "name", ["img; touch /tmp/pwned", "img && id", "img$(id)", "img`id`", "a b"]
)
def test_row8_a_shell_looking_image_name_stays_one_argv_element(name: str) -> None:
    """No shell sits between Iterant and the runtime: the name is one argv element, and the
    runtime refuses what is not an image."""
    r = FakeRunner()
    make(r).create(name, 1, 64, 64, 60, net="default-deny")  # not rejected by name
    assert runs(r)
    for argv in runs(r):
        assert argv.count(name) == 1
        at = argv.index(name)
        assert argv[at - 2 : at] == ["--entrypoint", "sleep"]  # still in the image position


def test_row8_exec_goes_through_the_runtime_with_the_command_as_one_argument() -> None:
    r = FakeRunner()
    sb = make(r).create("img:1", 1, 64, 64, 60, net="default-deny")
    r.calls.clear()
    evil = "true; docker run --privileged -v /:/host alpine cat /host/etc/shadow"
    sb.exec(evil, 5)
    [call] = [c for c in r.calls if "exec" in c]
    assert call.count(evil) == 1
    for c in r.calls:
        assert_no_escape_flags(c)


# --- (b) hostile code in a real sandbox ----------


def _runtime_works(kind: str) -> bool:
    exe = shutil.which(kind)
    if exe is None:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Tracking:
    """Destroys every sandbox it made, even if a test fails, and denies the network to
    ``proxy-only`` requests (the live tests have no proxy to route to)."""

    def __init__(self, inner: SandboxBackend, force_deny: bool = False) -> None:
        self.inner = inner
        self.force_deny = force_deny
        self.created: list[Sandbox] = []

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: Any = "proxy-only"
    ) -> Sandbox:
        sb = self.inner.create(
            image, cpu, mem, disk, ttl, "default-deny" if self.force_deny else net
        )
        self.created.append(sb)
        return sb


@pytest.fixture(
    params=[
        pytest.param("podman", marks=pytest.mark.podman, id="podman"),
        pytest.param("docker", marks=pytest.mark.docker, id="docker"),
    ]
)
def live(request: pytest.FixtureRequest) -> Iterator[Tracking]:
    kind: str = request.param
    if not _runtime_works(kind):
        pytest.skip(f"{kind} is not installed or not usable here")
    prefix = "w3-redteam-" + kind
    inner: SandboxBackend = (
        PodmanBackend(name_prefix=prefix)
        if kind == "podman"
        else DockerBackend(unsafe_container=True, name_prefix=prefix)
    )
    tracking = Tracking(inner)
    try:
        yield tracking
    finally:
        for sb in tracking.created:
            try:
                sb.destroy()
            except Exception:
                pass


def sandbox(live: Tracking) -> Sandbox:
    return live.create(TEST_IMAGE, 1.0, 256, 64, 300, "default-deny")


def sh(sb: Sandbox, script: str, timeout: float = 60) -> tuple[int, str]:
    res = sb.exec(script, timeout)
    return res.exit_code, res.stdout.strip()


def test_row8_live_the_agent_is_not_root(live: Tracking) -> None:
    sb = sandbox(live)
    assert sh(sb, "id -u") == (0, "10001")
    code, _ = sh(sb, "touch /etc/pwned")
    assert code != 0
    code, _ = sh(sb, "touch /usr/local/bin/pwned")
    assert code != 0
    assert sh(sb, "touch /work/ok && echo fine")[1] == "fine"


def test_row8_live_no_capabilities_and_no_new_privileges(live: Tracking) -> None:
    sb = sandbox(live)
    code, out = sh(sb, "grep -E '^CapEff:' /proc/self/status")
    assert code == 0 and out.split()[-1] == "0" * 16
    assert sh(sb, "mount -t tmpfs none /mnt")[0] != 0
    assert sh(sb, "grep -E '^NoNewPrivs:' /proc/self/status")[1].split()[-1] == "1"


def test_row8_live_default_deny_means_no_route_off_the_box(live: Tracking) -> None:
    sb = sandbox(live)
    # the only interface is loopback
    code, out = sh(sb, "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' ' | sort -u")
    assert code == 0 and out.split() in ([], ["lo"])
    probe = (
        'python3 -c "import socket,sys\n'
        "s=socket.socket(); s.settimeout(3)\n"
        "try:\n s.connect((sys.argv[1], int(sys.argv[2]))); print('CONNECTED')\n"
        "except OSError as e:\n print('blocked')\" "
    )
    for host, port in (("1.1.1.1", "53"), ("169.254.169.254", "80"), ("8.8.8.8", "443")):
        assert sh(sb, probe + f"{host} {port}")[1] == "blocked", host
    code, out = sh(sb, "getent hosts example.com")
    assert code != 0 and not out


def test_row8_live_the_host_key_and_host_files_are_not_in_the_box(
    live: Tracking, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "sk-ant-redteam-" + uuid.uuid4().hex
    marker = f"host-marker-{uuid.uuid4().hex}"
    (tmp_path / marker).write_text("host file")
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    sb = sandbox(live)
    _, env = sh(sb, "env; cat /proc/1/environ | tr '\\0' '\\n'")
    assert secret not in env
    assert sh(sb, f"find / -xdev -name '{marker}' 2>/dev/null | wc -l")[1] == "0"
    for sock in ("/var/run/docker.sock", "/run/podman/podman.sock", "/run/user"):
        assert sh(sb, f"test -e {sock}")[0] != 0, sock


def test_row8_live_a_fork_bomb_is_capped_by_the_pids_limit(live: Tracking) -> None:
    sb = sandbox(live)
    code, out = sh(
        sb,
        'python3 -c "import os,time\n'
        "n=0\n"
        "try:\n"
        " for _ in range(5000):\n"
        "  if os.fork()==0:\n"
        "   time.sleep(30); os._exit(0)\n"
        "  n+=1\n"
        "except OSError: pass\n"
        'print(n)"',
        timeout=60,
    )
    assert code == 0 and 0 < int(out) < 5000, (code, out)


# the /out attacks, through the real executor and a real sandbox ----------

OUT_ATTACKS = {
    "symlink-to-host-file": "ln -s /etc/passwd /out/leak",
    "symlink-to-dir": "ln -s /work /out/dir",
    "hard-link": "echo x > /out/a && ln /out/a /out/b",
    "fifo": "mkfifo /out/pipe",
    "huge-sparse-file": "truncate -s 5M /out/big",
    "too-many-files": "i=0; while [ $i -lt 1100 ]; do : > /out/f$i; i=$((i+1)); done",
    "too-deep": "mkdir -p /out/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s",
    "unreadable-file": "echo x > /out/secret; chmod 000 /out/secret",
    "forged-scores": "echo '{\"v\": 1}' > /out/result.json; echo 'SCORE: 1.0' > /out/score.txt; exit 0",  # noqa: E501
}


@pytest.mark.parametrize("attack", sorted(OUT_ATTACKS))
def test_row8_live_hostile_out_scores_zero_and_is_not_an_infra_error(
    live: Tracking, attack: str
) -> None:
    from iterant.contracts import RunKey

    pack = load_pack(SYNTHETIC)
    task = pack.tasks[3]
    deny = Tracking(live.inner, force_deny=True)
    live.created = deny.created  # one cleanup list
    proxy, _ = make_proxy()
    executor = RunExecutor(
        backend=deny,
        grader=Grader(deny, pack, SYNTHETIC, timeout=20),
        proxy=proxy,
        run_command=OUT_ATTACKS[attack],
        read_input=lambda t: (SYNTHETIC / t.input).read_bytes(),
        config=OrchestratorConfig(run_timeout_s=60, read_budget_s=60, mem=256, disk=64),
    )
    key = RunKey(1, "redteam", task.id, 0)
    done = executor.execute(key, task, TEST_IMAGE).result
    if attack == "forged-scores":
        # valid-looking output goes on to the grader, which runs the pack's script (no
        # solve.sh here) and must not be fooled by the forged files
        assert done.score == 0.0, done
    else:
        assert (done.score, done.infra_err) == (0.0, False), done
