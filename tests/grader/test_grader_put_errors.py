"""Agent-file puts: only candidate-caused failures score 0.0; other sandbox errors are infra."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from iterant.contracts import (
    ExecResult,
    HiddenCheckPath,
    NetMode,
    Pack,
    Sandbox,
    Split,
    Task,
    VisibleInputPath,
)
from iterant.grader import Grader
from iterant.sandbox import SandboxError, SandboxLost

IMAGE = "docker.io/library/python@sha256:" + "ab" * 32
PACK = Pack("p", "1", (), "0" * 64, IMAGE)
AGENT_PREFIX = "/work/out/"


def _ok(stdout: str = "") -> ExecResult:
    return ExecResult(0, stdout, "", False, 0.1)


@dataclass
class PutFailSandbox:
    fail: Exception | None
    files: dict[str, bytes] = field(default_factory=lambda: {})

    def put(self, path: str, data: bytes) -> None:
        if self.fail is not None and path.startswith(AGENT_PREFIX):
            raise self.fail
        self.files[path] = data

    def get(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        return _ok(json.dumps({"v": 1}))

    def destroy(self) -> None:
        pass


class Backend:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.created = 0

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        self.created += 1
        return PutFailSandbox(self.fail)


def _task(root: Path) -> Task:
    d = root / "tasks" / "t1"
    d.mkdir(parents=True)
    (d / "run").write_bytes(b"echo run\n")
    (d / "check").write_bytes(b'{"v": 1}')
    return Task(
        "t1",
        Split.SELECT,
        (),
        VisibleInputPath(PurePosixPath("tasks/t1/input.json")),
        HiddenCheckPath(PurePosixPath("tasks/t1/check")),
    )


def test_infra_sandbox_error_on_put_is_not_a_zero_score(tmp_path: Path) -> None:
    backend = Backend(SandboxError("runtime exploded"))
    r = Grader(backend, PACK, tmp_path).grade(_task(tmp_path), {"a.txt": b"x"})
    assert r.score is None and r.infra_err


def test_sandbox_lost_on_put_scores_zero(tmp_path: Path) -> None:
    backend = Backend(SandboxLost("gone"))
    r = Grader(backend, PACK, tmp_path).grade(_task(tmp_path), {"a.txt": b"x"})
    assert r.score == 0.0 and r.error is None


def test_file_where_another_needs_a_directory_scores_zero_without_a_sandbox(
    tmp_path: Path,
) -> None:
    backend = Backend()
    r = Grader(backend, PACK, tmp_path).grade(_task(tmp_path), {"a": b"x", "a/b": b"y"})
    assert r.score == 0.0 and r.error is None
    assert backend.created == 0


def test_overlong_path_component_scores_zero(tmp_path: Path) -> None:
    backend = Backend()
    r = Grader(backend, PACK, tmp_path).grade(_task(tmp_path), {"x" * 300: b"x"})
    assert r.score == 0.0 and r.error is None
