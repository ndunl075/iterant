"""Shared fixtures: a small target repo, a pack, a stub auditor and a fake sandbox backend."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

from iterant.contracts import (
    ExecResult,
    HiddenCheckPath,
    NetMode,
    Pack,
    Proposal,
    Split,
    TargetConfig,
    TargetModel,
    Task,
    VisibleInputPath,
)
from iterant.gate import AuditVerdict, DiffGate, GateReport

FILES = {
    "iterant.toml": "[target]\n",
    "Containerfile": "FROM scratch\n",
    "agent/loop.py": "def run(task):\n    return task\n",
    "prompts/system.md": "You are a careful assistant.\n",
    "tools/search.py": "def search(q):\n    return []\n",
    "secret.py": "KEY = 1\n",
    "evals/pack/pack.toml": 'name = "p"\n',
    "evals/pack/tasks/alpha-17/input.json": '{"q": 1}\n',
    "evals/pack/tasks/alpha-17/check": (
        "#!/bin/sh\ntest \"$(cat /out/answer.txt)\" = 'forty-two is the expected answer'\n"
    ),
    "evals/pack/tasks/beta-9/input.json": '{"q": 2}\n',
    "evals/pack/tasks/beta-9/check": "#!/bin/sh\ntest -f /out/done && exit 0\n",
}


@dataclass
class StubAuditor:
    verdict: AuditVerdict = field(default_factory=lambda: AuditVerdict(False, False, ()))
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=lambda: [])

    def audit(self, diff: str, policy: str) -> AuditVerdict:
        self.calls.append((diff, policy))
        if self.error is not None:
            raise self.error
        return self.verdict


@dataclass
class FakeSandbox:
    backend: FakeBackend
    destroyed: bool = False

    def put(self, path: str, data: bytes) -> None:
        self.backend.puts[path] = data

    def get(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.backend.cmds.append(cmd)
        for needle, result in self.backend.results.items():
            if needle in cmd:
                return result
        return ExecResult(0, "", "", False, 0.0)

    def destroy(self) -> None:
        self.destroyed = True


@dataclass
class FakeBackend:
    results: dict[str, ExecResult] = field(default_factory=lambda: {})
    cmds: list[str] = field(default_factory=lambda: [])
    puts: dict[str, bytes] = field(default_factory=lambda: {})
    created: list[FakeSandbox] = field(default_factory=lambda: [])
    nets: list[str] = field(default_factory=lambda: [])

    def create(
        self,
        image: str,
        cpu: float,
        mem: int,
        disk: int,
        ttl: float,
        net: NetMode = "proxy-only",
    ) -> FakeSandbox:
        self.nets.append(net)
        sb = FakeSandbox(self)
        self.created.append(sb)
        return sb


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return root


@pytest.fixture
def target() -> TargetConfig:
    return TargetConfig(
        genome=("prompts/**", "tools/**", "agent/loop.py"),
        image="Containerfile",
        run="python -m agent",
        model=TargetModel("m", "medium", max_usd_micros_per_run=1000),
    )


@pytest.fixture
def pack() -> Pack:
    def task(task_id: str, split: Split) -> Task:
        return Task(
            id=task_id,
            split=split,
            tags=(),
            input=VisibleInputPath(PurePosixPath("tasks", task_id, "input.json")),
            check=HiddenCheckPath(PurePosixPath("tasks", task_id, "check")),
        )

    return Pack(
        name="p",
        version="1",
        tasks=(task("alpha-17", Split.SELECT), task("beta-9", Split.TRAIN)),
        content_hash="0" * 64,
        grader_image="grader@sha256:" + "0" * 64,
    )


@pytest.fixture
def auditor() -> StubAuditor:
    return StubAuditor()


@pytest.fixture
def make_gate(repo: Path, auditor: StubAuditor) -> Callable[..., DiffGate]:
    def make(**kwargs: object) -> DiffGate:
        kwargs.setdefault("protected", ("evals/pack",))
        return DiffGate(auditor, repo / "evals" / "pack", **kwargs)  # type: ignore[arg-type]

    return make


Check = Callable[[str], GateReport]


@pytest.fixture
def run_gate(
    repo: Path, target: TargetConfig, pack: Pack, make_gate: Callable[..., DiffGate]
) -> Check:
    def run(diff: str) -> GateReport:
        return make_gate().check(
            Proposal(diff, "a hypothesis"), repo=repo, target=target, pack=pack
        )

    return run


def _git_diff(path: str, old: str | None, new: str | None) -> str:
    a = f"a/{path}" if old is not None else "/dev/null"
    b = f"b/{path}" if new is not None else "/dev/null"
    head = f"diff --git a/{path} b/{path}\n"
    if old is None:
        head += "new file mode 100644\n"
    if new is None:
        head += "deleted file mode 100644\n"
    body = "".join(
        difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=a,
            tofile=b,
        )
    )
    return head + body


@pytest.fixture
def edit(repo: Path) -> Callable[[str, str | None], str]:
    """``edit(path, new_text)``: a git diff turning the repo's file into ``new_text``
    (``new_text=None`` deletes it; a missing file is created)."""

    def make(path: str, new: str | None) -> str:
        file = repo / path
        old = file.read_text(encoding="utf-8") if file.exists() else None
        return _git_diff(path, old, new)

    return make


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()
