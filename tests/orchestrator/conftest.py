"""A whole fake world for orchestrator tests.

Real: archive (SQLite), git repo, proxy ledger, grader and its pack on disk. Fake: the sandbox
backend (agents are scripted by ``World.solver``), proposer, gate and image source. No network,
no container runtime, no live model.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from iterant.archive import Archive, archive_path
from iterant.contracts import (
    AcceptanceThresholds,
    Candidate,
    ExecResult,
    GateResult,
    NetMode,
    Pack,
    Profile,
    Proposal,
    ProxyCall,
    ProxyPolicy,
    Role,
    RoleCap,
    RunKey,
    Sandbox,
    Stage,
    Status,
    TargetConfig,
    TargetModel,
)
from iterant.grader import Grader
from iterant.orchestrator import Orchestrator, OrchestratorConfig, RunExecutor
from iterant.packs import load_pack
from iterant.proxy import ModelPrice, Proxy, ProxyConfig, Usage
from iterant.sandbox import SandboxError, SandboxLost

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
PROPOSER_CAP = 50_000
PROPOSER_COST = 17_500  # 1000 in + 500 out at $5 / $25 per Mtok
GRADER_IMAGE = "docker.io/library/python@sha256:" + "cd" * 32
RUN_COST = 4500  # 2000 input + 500 output tokens at $1 / $5 per Mtok
RUN_CAP = 100_000

TRAIN = [f"tr{i}" for i in range(4)]
SELECT = [f"se{i:02d}" for i in range(10)]
HOLDOUT = [f"ho{i}" for i in range(6)]
TEST = ["te0", "te1"]
CHAMP_SOLVES = {"tr0", "tr1", "se00", "se01", "se02", "ho0", "ho1"}

THRESHOLDS = AcceptanceThresholds(
    select_tasks=10, select_reps=1, holdout_tasks=6, holdout_reps=1, smoke_tasks=2
)


class Crash(BaseException):
    """Simulates the process dying (BaseException: nothing in the orchestrator may swallow it)."""


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
    lost: bool = False
    destroyed: int = 0

    def _alive(self) -> None:
        if self.lost:
            raise SandboxLost("sandbox is gone")

    def put(self, path: str, data: bytes) -> None:
        self._alive()
        if self.world.put_fails:
            raise SandboxError("put failed")
        self.files[path] = data

    def get(self, path: str) -> bytes:
        self._alive()
        if self.world.get_fails:
            raise SandboxError("get failed")
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def destroy(self) -> None:
        self.destroyed += 1

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self._alive()
        ok = ExecResult(0, "", "", False, 1.0)
        if cmd == "python3 -V":
            return ExecResult(1, "", "", False, 0.1) if self.world.preflight_fails else _py()
        if cmd.startswith("mkdir -p /work/out && test -r"):
            return ok
        if cmd.startswith("cd /work/out && exec sh"):
            good = self.files.get("/work/out/answer.txt") == b"ok"
            return ExecResult(0, json.dumps({"v": 1 if good else 0}), "", False, 0.1)
        if cmd == "true":
            if self.world.runtime_down:
                raise SandboxError("runtime is down")
            return ok
        if cmd.startswith("find "):
            return self._list()
        if "ANTHROPIC_API_KEY" in cmd:
            return self._agent(cmd, timeout)
        raise AssertionError(f"unexpected command {cmd!r}")

    def _list(self) -> ExecResult:
        if self.world.listing is not None:
            return ExecResult(0, self.world.listing, "", False, 0.1)
        out = [
            f"f\t1\t{len(d)}\t{p.removeprefix('/work/out/')}\0"
            for p, d in self.files.items()
            if p.startswith("/work/out/")
        ]
        return ExecResult(0 if out else 1, "".join(out), "", False, 0.1)

    def _agent(self, cmd: str, timeout: float) -> ExecResult:
        w = self.world
        token = re.search(r"ANTHROPIC_API_KEY='(itr_[^']+)'", cmd)
        assert token is not None and w.real_key not in cmd
        session = w.proxy.sessions.resolve(token.group(1))
        assert session is not None and session.key is not None
        key = session.key
        w.exec_log.append(key)
        if w.crash_at is not None and len(w.exec_log) == w.crash_at:
            raise Crash("process died")
        arm = "champ" if self.image.startswith("img:champion") else self.image.removeprefix("img:")
        outcome = w.solver(arm, key.task, key.attempt)
        if outcome == "infra":
            raise SandboxError("runtime broke")
        if outcome == "create-lost":
            raise SandboxLost("gone before it started")
        body = json.dumps({"model": HAIKU, "max_tokens": 1000, "messages": []}).encode()
        adm = w.proxy.admit(token.group(1), body)
        assert not hasattr(adm, "status"), adm
        w.proxy.settle(adm, Usage(input=2000, output=500), outcome == "refuse", None)  # type: ignore[arg-type]
        if outcome in ("ok", "refuse"):
            self.files["/work/out/answer.txt"] = b"ok"
        elif outcome == "wrong":
            self.files["/work/out/answer.txt"] = b"no"
        elif outcome == "timeout":
            return ExecResult(-1, "", "", True, timeout)
        elif outcome == "truncated":
            return ExecResult(0, "x", "", False, 1.0, truncated=True)
        elif outcome == "lost":
            self.lost = True
            return ExecResult(137, "", "", False, 1.0, sandbox_lost=True)
        elif outcome == "lost-raise":
            self.lost = True
            raise SandboxLost("killed")
        elif outcome == "crash":
            return ExecResult(1, "", "boom", False, 1.0)
        elif outcome == "ok-crash":  # right answer written, then a non-zero exit
            self.files["/work/out/answer.txt"] = b"ok"
            return ExecResult(1, "", "boom", False, 1.0)
        return ok_result()


def ok_result() -> ExecResult:
    return ExecResult(0, "", "", False, 1.0)


def _py() -> ExecResult:
    return ExecResult(0, "Python 3.12.0", "", False, 0.1)


class FakeBackend:
    def __init__(self, world: World) -> None:
        self.world = world
        self.ttls: list[float] = []
        self.sandboxes: list[FakeSandbox] = []

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        if self.world.create_fails:
            raise SandboxError("cannot create")
        self.ttls.append(ttl)
        sb = FakeSandbox(self.world, image, net)
        self.sandboxes.append(sb)
        return sb


class FakeProposer:
    def __init__(self, world: World | None = None) -> None:
        self.world = world
        self.queue: list[Proposal | None | Exception] = []
        self.calls: list[dict[str, Any]] = []

    def propose(
        self,
        parent: str,
        workdir: Any,
        failure_digest: str,
        history: Any,
    ) -> Proposal | None:
        root = Path(workdir)
        prompt = (root / "agent" / "prompt.txt").read_text()
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        self.calls.append(
            {
                "parent": parent,
                "prompt": prompt,
                "digest": failure_digest,
                "history": list(history),
                "files": files,
            }
        )
        if self.world is not None:  # the proposer's LLM call, through the proxy
            w = self.world
            token = w.proxy.issue_call_token(Role.PROPOSER)
            body = json.dumps({"model": OPUS, "max_tokens": 1000, "messages": []}).encode()
            adm = w.proxy.admit(token, body)
            assert not hasattr(adm, "status"), adm
            w.proxy.settle(adm, Usage(input=1000, output=500), False, None)  # type: ignore[arg-type]
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeGate:
    def __init__(self) -> None:
        self.fail: set[str] = set()
        self.boom: set[str] = set()
        self.seen: list[str] = []

    def check(
        self, proposal: Proposal, *, repo: Any, target: TargetConfig, pack: Pack
    ) -> GateResult:
        assert (Path(repo) / "agent" / "prompt.txt").is_file()
        self.seen.append(proposal.hypothesis)
        if proposal.hypothesis in self.boom:
            raise RuntimeError("auditor down")
        if proposal.hypothesis in self.fail:
            return GateResult(False, ("task id found in diff",))
        return GateResult(True)


class FakeImages:
    def __init__(self) -> None:
        self.fail: set[str] = set()
        self.built: list[str] = []

    def champion(self, sha: str) -> str:
        return f"img:champion:{sha[:7]}"

    def candidate(self, cand_id: str, parent_sha: str, diff: str) -> str:
        self.built.append(cand_id)
        assert diff.startswith("diff --git")
        if cand_id in self.fail:
            raise RuntimeError("build failed")
        return f"img:{cand_id}"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": __import__("os").environ["PATH"],
            "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return out.stdout.strip()


class World:
    Crash = Crash
    THRESHOLDS = THRESHOLDS
    SELECT = SELECT
    HOLDOUT = HOLDOUT
    TRAIN = TRAIN
    RUN_COST = RUN_COST
    PROPOSER_COST = PROPOSER_COST
    PROPOSER_CAP = PROPOSER_CAP
    RUN_CAP = RUN_CAP

    @staticmethod
    def proposal(old: str, new: str, hypothesis: str) -> Proposal:
        return Proposal(diff(old, new), hypothesis)

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.repo = tmp / "repo"
        (self.repo / "agent").mkdir(parents=True)
        (self.repo / "agent" / "prompt.txt").write_bytes(b"v0\n")
        (self.repo / "notes").mkdir()
        (self.repo / "notes" / "secret.txt").write_bytes(b"not for the proposer")
        (self.repo / ".containerignore").write_bytes(b"notes/")
        self.pack_root = self.repo / "evals" / "pack"  # the pack lives INSIDE the target repo
        self.write_pack()
        _git(self.repo, "init", "-q", "-b", "main")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "seed")
        self.pack = load_pack(self.pack_root)
        self.real_key = "sk-ant-REAL-KEY"
        self.solver: Callable[[str, str, int], str] = default_solver
        self.exec_log: list[RunKey] = []
        self.crash_at: int | None = None
        self.preflight_fails = False
        self.create_fails = False
        self.put_fails = False
        self.get_fails = False
        self.runtime_down = False
        self.redactions: list[list[str]] = []
        self.listing: str | None = None
        self.proposer = FakeProposer(self)
        self.gate = FakeGate()
        self.images = FakeImages()
        self.backend = FakeBackend(self)
        self.target = TargetConfig(
            genome=("agent/**",),
            image="Containerfile",
            run="python -m agent --task /task/input.json --out /out",
            model=TargetModel(HAIKU, "medium", max_usd_micros_per_run=RUN_CAP),
        )
        self.archives: list[Archive] = []
        self.proxy = self._new_proxy()

    def write_pack(self, input_suffix: str = "") -> None:
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
                (d / "input.json").write_bytes(json.dumps({"q": tid + input_suffix}).encode())
                (d / "run").write_bytes(b"echo run\n")
                (d / "check").write_bytes(b'{"v": 1}')
                splits[tid] = split
        (root / "pack.toml").write_text(
            f'name = "fake"\nversion = "1"\ngrader_image = "{GRADER_IMAGE}"\n'
        )
        (root / "splits.json").write_text(json.dumps({"assignment": splits}))

    def _sink(self, call: ProxyCall) -> None:
        """What the CLI wires: every proxy call lands in the archive (ids are process-local)."""
        with closing(Archive.open_path(archive_path(self.repo), create_schema=False)) as a:
            a.insert_proxy_call(replace(call, id=0))

    def _new_proxy(self) -> Proxy:
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
        return Proxy(
            ProxyConfig(
                policy=policy,
                sink=self._sink,
                prices={
                    HAIKU: ModelPrice(1_000_000, 5_000_000),
                    OPUS: ModelPrice(5_000_000, 25_000_000),
                },
                anthropic_base_url="http://upstream.invalid",
                anthropic_api_key=self.real_key,
            )
        )

    def reload_pack(self) -> None:
        self.pack = load_pack(self.pack_root)

    def orchestrator(self, **cfg: Any) -> Orchestrator:
        """A fresh orchestrator on fresh connections and a fresh proxy: what a restart sees."""
        self.proxy = self._new_proxy()
        archive = Archive.open(self.repo)
        self.archives.append(archive)
        cfg.setdefault("thresholds", THRESHOLDS)
        return Orchestrator(
            repo=self.repo,
            pack=self.pack,
            pack_root=self.pack_root,
            target=self.target,
            archive=archive,
            backend=self.backend,
            grader=Grader(self.backend, self.pack, self.pack_root, timeout=5),
            proxy=self.proxy,
            proposer=self.proposer,
            gate=self.gate,
            images=self.images,
            digest=self._digest,
            config=OrchestratorConfig(**cfg),
        )

    def _digest(self, redact: Any) -> str:
        self.redactions.append(list(redact))
        return "digest"

    def executor(self, **cfg_args: Any) -> RunExecutor:
        """An executor for direct runs; creates the batch and candidate rows its proxy calls
        reference."""
        db = self.db()
        if db.get_batch(1) is None:
            db.create_batch(1, self.pack.content_hash)
            sha = "0" * 64
            db.insert_candidate(Candidate("c", None, sha, "x", Stage.PROMOTE, Status.PROMOTED, 0))
        cfg = OrchestratorConfig(thresholds=THRESHOLDS, **cfg_args)
        return RunExecutor(
            backend=self.backend,
            grader=Grader(self.backend, self.pack, self.pack_root, timeout=5),
            proxy=self.proxy,
            run_command=self.target.run,
            read_input=lambda t: (self.pack_root / t.input).read_bytes(),
            config=cfg,
        )

    def db(self) -> Archive:
        a = Archive.open(self.repo)
        self.archives.append(a)
        return a

    def queue(self, *proposals: Proposal | None | Exception) -> None:
        self.proposer.queue.extend(proposals)

    def git(self, *args: str) -> str:
        return _git(self.repo, *args)

    def close(self) -> None:
        for a in self.archives:
            a.close()


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    w = World(tmp_path)
    yield w
    w.close()
