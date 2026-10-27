"""A fake world for CLI tests.

Real: archive (SQLite), a git repo, the proxy ledger, the grader and a pack on disk, and the CLI
itself. Fake: the sandbox backend (agents are scripted by ``World.solver``), proposer, gate and
image source, plugged in through ``main(argv, factory=world.factory)``. No network, no container
runtime, no live model.

TODO(shared): this duplicates the world in ``tests/orchestrator/conftest.py`` (tests may not
import each other's conftest and nobody may add a root one); a shared fake belongs in a later
wave.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from iterant.archive import Archive
from iterant.cli.main import main
from iterant.cli.runtime import Runtime
from iterant.cli.settings import Project, Settings
from iterant.contracts import (
    ExecResult,
    GateResult,
    NetMode,
    Pack,
    Profile,
    Proposal,
    ProxyPolicy,
    Role,
    RoleCap,
    Sandbox,
    Split,
    TargetConfig,
)
from iterant.grader import Grader
from iterant.proxy import ModelPrice, Proxy, ProxyConfig, Usage
from iterant.sandbox import SandboxError

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
GRADER_IMAGE = "docker.io/library/python@sha256:" + "cd" * 32
RUN_COST = 4500
RUN_CAP = 100_000
PROPOSER_CAP = 50_000

TRAIN = [f"tr{i}" for i in range(4)]
SELECT = [f"se{i:02d}" for i in range(10)]
HOLDOUT = [f"ho{i}" for i in range(6)]
TEST = ["te0", "te1"]
CHAMP_SOLVES = {"tr0", "tr1", "se00", "se01", "se02", "ho0", "ho1"}

# Flags that shrink the acceptance rule to the fake pack.
SMALL = [
    "--select-tasks", "10", "--select-reps", "1",
    "--holdout-tasks", "6", "--holdout-reps", "1",
    "--smoke-tasks", "2",
]  # fmt: skip
# Calibration strong enough to unlock `run` (see calibrate.MIN_*).
CAL_ESTIMATE = ["--split", "select", "--tasks", "10", "--reps", "1"]
CAL_AA = ["--trials", "100", "--resamples", "1000"]
RUN_NOSKIP = [*SMALL, "--digest-tasks", "3"]
RUN_SMALL = [*RUN_NOSKIP, "--skip-calibration"]

INIT_ARGS = [
    "--profile", "claude", "--model", HAIKU, "--max-usd-per-run", "0.10",
    "--genome", "agent/**", "--pack", "evals/pack",
]  # fmt: skip


def default_solver(arm: str, task: str, attempt: int) -> str:
    if arm == "champ":
        return "ok" if task in CHAMP_SOLVES else "wrong"
    return "ok"


def diff(old: str, new: str) -> str:
    return (
        "diff --git a/agent/prompt.txt b/agent/prompt.txt\n"
        "--- a/agent/prompt.txt\n+++ b/agent/prompt.txt\n"
        f"@@ -1 +1 @@\n-{old}\n+{new}\n"
    )


@dataclass
class FakeSandbox:
    world: World
    image: str
    net: NetMode
    files: dict[str, bytes] = field(default_factory=lambda: {})

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def destroy(self) -> None:
        pass

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.world.cmds.append(cmd)
        ok = ExecResult(0, "", "", False, 1.0)
        if cmd == "python3 -V":
            if self.world.preflight_fails:
                return ExecResult(1, "", "", False, 0.1)
            return ExecResult(0, "Python 3.12.0", "", False, 0.1)
        if cmd.startswith("mkdir -p /work/out && test -r"):
            return ok
        if cmd.startswith("cd /work/out && exec sh"):
            good = self.files.get("/work/out/answer.txt") == b"ok"
            return ExecResult(0, json.dumps({"v": 1 if good else 0}), "", False, 0.1)
        if cmd == "true":
            return ok
        if cmd.startswith("find "):
            out = [
                f"f\t1\t{len(d)}\t{p.removeprefix('/work/out/')}\0"
                for p, d in self.files.items()
                if p.startswith("/work/out/")
            ]
            return ExecResult(0 if out else 1, "".join(out), "", False, 0.1)
        if "ANTHROPIC_API_KEY" in cmd:
            return self._agent(cmd)
        raise AssertionError(f"unexpected command {cmd!r}")

    def _agent(self, cmd: str) -> ExecResult:
        w = self.world
        token = re.search(r"ANTHROPIC_API_KEY='(itr_[^']+)'", cmd)
        assert token is not None
        session = w.proxy.sessions.resolve(token.group(1))
        assert session is not None and session.key is not None
        key = session.key
        w.exec_log.append(key.task)
        arm = "champ" if self.image.startswith("img:champion") else self.image.removeprefix("img:")
        outcome = w.solver(arm, key.task, key.attempt)
        body = json.dumps({"model": HAIKU, "max_tokens": 1000, "messages": []}).encode()
        adm = w.proxy.admit(token.group(1), body)
        assert not hasattr(adm, "status"), adm
        w.proxy.settle(adm, Usage(input=2000, output=500), False, None)  # type: ignore[arg-type]
        self.files["/work/out/answer.txt"] = b"ok" if outcome == "ok" else b"no"
        return ExecResult(0, "", "", False, 1.0)


class FakeBackend:
    def __init__(self, world: World) -> None:
        self.world = world
        self.sandboxes: list[FakeSandbox] = []

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        if self.world.create_fails:
            raise SandboxError("cannot create")
        sb = FakeSandbox(self.world, image, net)
        self.sandboxes.append(sb)
        return sb


class FakeProposer:
    def __init__(self, world: World) -> None:
        self.world = world
        self.queue: list[Proposal | None] = []
        self.digests: list[str] = []

    def propose(self, parent: str, workdir: Any, failure_digest: str, history: Any) -> Any:
        self.digests.append(failure_digest)
        w = self.world
        token = w.proxy.issue_call_token(Role.PROPOSER)
        body = json.dumps({"model": OPUS, "max_tokens": 1000, "messages": []}).encode()
        adm = w.proxy.admit(token, body)
        assert not hasattr(adm, "status"), adm
        w.proxy.settle(adm, Usage(input=1000, output=500), False, None)  # type: ignore[arg-type]
        return self.queue.pop(0)


class FakeGate:
    def check(
        self, proposal: Proposal, *, repo: Any, target: TargetConfig, pack: Pack
    ) -> GateResult:
        return GateResult(True)


class FakeImages:
    def champion(self, sha: str) -> str:
        return f"img:champion:{sha[:7]}"

    def candidate(self, cand_id: str, parent_sha: str, diff: str) -> str:
        return f"img:{cand_id}"


def _git(repo: Path, *args: str) -> str:
    import os

    out = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return out.stdout.strip()


class World:
    SMALL = SMALL
    RUN_SMALL = RUN_SMALL
    RUN_NOSKIP = RUN_NOSKIP
    INIT_ARGS = INIT_ARGS
    diff = staticmethod(diff)

    def __init__(self, tmp: Path) -> None:
        self.repo = tmp / "repo"
        (self.repo / "agent").mkdir(parents=True)
        (self.repo / "agent" / "prompt.txt").write_bytes(b"v0\n")
        (self.repo / "Containerfile").write_bytes(b"FROM scratch\n")
        self.pack_root = self.repo / "evals" / "pack"
        self._write_pack()
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "seed")
        self.solver: Callable[[str, str, int], str] = default_solver
        self.exec_log: list[str] = []
        self.cmds: list[str] = []
        self.preflight_fails = False
        self.create_fails = False
        self.queue: list[Proposal | None] = []
        self.proposer = FakeProposer(self)
        self.backend = FakeBackend(self)
        self.settings: list[Settings] = []
        self.runtimes: list[Runtime] = []
        self.proxy: Proxy = self._new_proxy(self.repo / ".iterant" / "archive.db")

    def _write_pack(self) -> None:
        root = self.pack_root
        splits: dict[str, str] = {}
        for split, ids in (
            ("train", TRAIN),
            ("select", SELECT),
            ("holdout", HOLDOUT),
            ("test", TEST),
        ):
            for tid in ids:
                d = root / "tasks" / tid
                d.mkdir(parents=True, exist_ok=True)
                (d / "input.json").write_bytes(json.dumps({"q": tid}).encode())
                (d / "run").write_bytes(b"echo run\n")
                (d / "check").write_bytes(b'{"v": 1}')
                splits[tid] = split
        (root / "pack.toml").write_text(
            f'name = "fake"\nversion = "1"\ngrader_image = "{GRADER_IMAGE}"\n'
        )
        (root / "splits.json").write_text(json.dumps({"assignment": splits}))

    def _new_proxy(self, db_path: Path, splits: dict[str, Split] | None = None) -> Proxy:
        from iterant.cli.runtime import _archive_sink  # pyright: ignore[reportPrivateUsage]

        policy = ProxyPolicy(
            Profile.CLAUDE,
            {
                Role.TARGET: RoleCap(
                    HAIKU,
                    "medium",
                    max_usd_micros_per_run=RUN_CAP,
                    max_usd_micros_per_gen=100_000_000,
                ),
                Role.PROPOSER: RoleCap(
                    OPUS,
                    "high",
                    max_usd_micros_per_run=PROPOSER_CAP,
                    max_usd_micros_per_gen=100_000_000,
                ),
            },
        )
        by_task = splits or {}
        return Proxy(
            ProxyConfig(
                policy=policy,
                sink=_archive_sink(db_path),
                prices={
                    HAIKU: ModelPrice(1_000_000, 5_000_000),
                    OPUS: ModelPrice(5_000_000, 25_000_000),
                },
                anthropic_base_url="http://upstream.invalid",
                anthropic_api_key="sk-ant-REAL-KEY",
                split_of=lambda k: by_task.get(k.task),
            )
        )

    def factory(self, settings: Settings, project: Project, db_path: Path) -> Runtime:
        """The ``RuntimeFactory`` the CLI is given: real archive/proxy/grader, fake the rest."""
        self.settings.append(settings)
        self.proxy = self._new_proxy(db_path, {t.id: t.split for t in project.pack.tasks})
        archive = Archive.open_path(db_path)
        self.proposer.queue = self.queue
        rt = Runtime(
            project=project,
            archive=archive,
            proxy=self.proxy,
            backend=self.backend,
            grader=Grader(self.backend, project.pack, project.pack_root, timeout=5),
            images=FakeImages(),
            proposer=self.proposer,
            gate=FakeGate(),
            analyst_client=None,
            analyst_model=HAIKU,
            analyst_effort="medium",
        )
        self.runtimes.append(rt)
        return rt

    def cal(self, kind: str) -> list[str]:
        """Flags that make `kind` strong enough to unlock a run."""
        return [*SMALL, *(CAL_ESTIMATE if kind == "estimate" else CAL_AA)]

    def args(self, command: str) -> list[str]:
        """Small-pack flags for ``command`` (run also skips calibration)."""
        return list(RUN_SMALL if command == "run" else SMALL)

    def calibrate(self) -> None:
        """A passing estimate and aa-test for the small pack (what `run` requires)."""
        assert self.cli("estimate", str(self.repo), *SMALL, *CAL_ESTIMATE) == 0
        assert self.cli("aa-test", str(self.repo), *SMALL, *CAL_AA) == 0

    def cli(self, *argv: str) -> int:
        return main(list(argv), factory=self.factory)

    def init(self) -> None:
        assert self.cli("init", str(self.repo), *INIT_ARGS) == 0

    def git(self, *args: str) -> str:
        return _git(self.repo, *args)


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    yield World(tmp_path)
