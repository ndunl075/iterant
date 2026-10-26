from __future__ import annotations

import base64
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest

from iterant.contracts import (
    ExecResult,
    HiddenCheckPath,
    NetMode,
    Sandbox,
    SandboxBackend,
    Split,
    Task,
    VisibleInputPath,
)
from iterant.grader import Grader, GradeResult, interpret
from iterant.grader.grader import _command  # pyright: ignore[reportPrivateUsage]

OnExec = Callable[["FakeSandbox", str, float], ExecResult]


def res(code: int = 0, out: str = "", err: str = "", timed_out: bool = False) -> ExecResult:
    return ExecResult(code, out, err, timed_out, 0.1)


def ok(_sb: FakeSandbox, _cmd: str, _timeout: float) -> ExecResult:
    return res()


@dataclass
class FakeSandbox:
    on_exec: OnExec
    net: NetMode
    files: dict[str, bytes] = field(default_factory=lambda: {})
    cmds: list[str] = field(default_factory=lambda: [])
    destroyed: int = 0

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.cmds.append(cmd)
        return self.on_exec(self, cmd, timeout)

    def destroy(self) -> None:
        self.destroyed += 1


class FakeBackend:
    def __init__(self, on_exec: OnExec = ok, sandbox_cls: type[FakeSandbox] = FakeSandbox) -> None:
        self.on_exec = on_exec
        self.sandbox_cls = sandbox_cls
        self.sandboxes: list[FakeSandbox] = []
        self.create_args: list[tuple[str, float, int, int, float, NetMode]] = []
        self.fail_create: Exception | None = None

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        if self.fail_create:
            raise self.fail_create
        self.create_args.append((image, cpu, mem, disk, ttl, net))
        sb = self.sandbox_cls(self.on_exec, net)
        self.sandboxes.append(sb)
        return sb


def make_task(root: Path, check: bytes = b"echo 0.5\n") -> Task:
    d = root / "tasks" / "t1"
    d.mkdir(parents=True)
    (d / "input.json").write_bytes(b"{}")
    (d / "check").write_bytes(check)
    return Task(
        "t1",
        Split.SELECT,
        (),
        VisibleInputPath(PurePosixPath("tasks/t1/input.json")),
        HiddenCheckPath(PurePosixPath("tasks/t1/check")),
    )


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeBackend(), SandboxBackend)
    assert isinstance(FakeSandbox(ok, "default-deny"), Sandbox)


def decode_check(cmd: str) -> bytes:
    m = re.search(r"printf %s '([A-Za-z0-9+/=]*)' \| base64 -d", cmd)
    assert m, cmd
    return base64.b64decode(m.group(1))


def test_grade_uses_fresh_default_deny_sandbox_and_destroys_it(tmp_path: Path) -> None:
    task = make_task(tmp_path, b"SECRET-CHECK\n")
    backend = FakeBackend()
    grader = Grader(backend, "img", tmp_path, timeout=30)
    r1 = grader.grade(task, {"answer.txt": b"42"})
    r2 = grader.grade(task, {"answer.txt": b"43"})
    assert (r1.score, r2.score) == (1.0, 1.0)
    assert len(backend.sandboxes) == 2
    assert all(a[0] == "img" and a[5] == "default-deny" for a in backend.create_args)
    for sb in backend.sandboxes:
        assert sb.destroyed == 1
        assert len(sb.cmds) == 1 and decode_check(sb.cmds[0]) == b"SECRET-CHECK\n"
    assert backend.sandboxes[0].files["/out/answer.txt"] == b"42"
    assert backend.sandboxes[1].files["/out/answer.txt"] == b"43"  # nothing carries over


def test_check_is_never_written_into_the_sandbox(tmp_path: Path) -> None:
    backend = FakeBackend()
    Grader(backend, "img", tmp_path).grade(make_task(tmp_path, b"SECRET\n"), {"a/b.txt": b"x"})
    files = backend.sandboxes[0].files
    assert list(files) == ["/out/a/b.txt"]
    assert not any(b"SECRET" in data for data in files.values())
    assert "SECRET" not in backend.sandboxes[0].cmds[0]  # only its base64 form travels


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("base64") is None, reason="no sh")
def test_command_runs_the_check_without_a_file(tmp_path: Path) -> None:
    out = tmp_path / "out"
    cmd = _command(b'echo "ran in $(pwd)"; echo 0.5\n').replace("/out", out.as_posix())
    sh = shutil.which("sh")
    assert sh is not None
    p = subprocess.run([sh, "-c", cmd], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout.strip().splitlines()[-1] == "0.5"
    assert list(out.iterdir()) == []  # the check left no file in /out


def test_out_files_cannot_overwrite_the_check(tmp_path: Path) -> None:
    backend = FakeBackend()
    Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {"../grader/check": b"exit 0"})
    assert not backend.sandboxes  # refused before any sandbox exists


def test_exec_gets_timeout(tmp_path: Path) -> None:
    seen: list[float] = []

    def on_exec(_sb: FakeSandbox, _cmd: str, timeout: float) -> ExecResult:
        seen.append(timeout)
        return res()

    Grader(FakeBackend(on_exec), "img", tmp_path, timeout=7).grade(make_task(tmp_path), {})
    assert seen == [7]


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x", "a/../../x", "", "a\\b", "a\0b"])
def test_bad_out_paths_are_infra_errors(tmp_path: Path, bad: str) -> None:
    backend = FakeBackend()
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {bad: b""})
    assert r.infra_err and r.score is None and not backend.sandboxes


def test_missing_check_file_is_infra_error(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    (tmp_path / "tasks" / "t1" / "check").unlink()
    assert Grader(FakeBackend(), "img", tmp_path).grade(task, {}).infra_err


def test_create_failure_is_infra_error(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.fail_create = RuntimeError("no podman")
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {})
    assert r.error == "sandbox create failed: no podman" and r.score is None


def test_exec_failure_is_infra_error_and_sandbox_destroyed(tmp_path: Path) -> None:
    def boom(_sb: FakeSandbox, _cmd: str, _timeout: float) -> ExecResult:
        raise OSError("lost")

    backend = FakeBackend(boom)
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {})
    assert r.error == "sandbox error: OSError"
    assert backend.sandboxes[0].destroyed == 1


def test_destroy_failure_does_not_change_result(tmp_path: Path) -> None:
    class Bad(FakeSandbox):
        def destroy(self) -> None:
            raise RuntimeError("x")

    backend = FakeBackend(sandbox_cls=Bad)
    assert Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {}).score == 1.0


@pytest.mark.parametrize(
    ("result", "score", "is_error"),
    [
        (res(0), 1.0, False),
        (res(1), 0.0, False),
        (res(0, "log line\n0.75\n"), 0.75, False),
        (res(0, "0\n"), 0.0, False),
        (res(1, "0.25"), 0.0, False),  # a failing exit never scores
        (res(1, "1.0"), 0.0, False),
        (res(1, "some failure output\n"), 0.0, False),
        (res(0, "all good\n"), None, True),  # exit 0 with non-score output: cannot tell
        (res(0, "1.5"), None, True),
        (res(0, "-0.1"), None, True),
        (res(0, "nan"), None, True),
        (res(0, "inf"), None, True),
        (res(-1, "", "", True), 0.0, False),  # timeouts, kills and OOM are the candidate's doing
        (res(137), 0.0, False),
        (res(2), 0.0, False),
        (res(127, "", "sh: python3: not found"), None, True),  # could not run
        (res(126), None, True),
    ],
)
def test_interpret(result: ExecResult, score: float | None, is_error: bool) -> None:
    got, err = interpret(result)
    assert got == score and (err is not None) == is_error


def test_grade_result_validation() -> None:
    with pytest.raises(ValueError):
        GradeResult(None, None, 0.0)
    with pytest.raises(ValueError):
        GradeResult(0.5, "e", 0.0)
    with pytest.raises(ValueError):
        GradeResult(2.0, None, 0.0)


def test_errors_never_contain_sandbox_output(tmp_path: Path) -> None:
    secret = "HIDDEN-CHECK-TEXT"
    task = make_task(tmp_path)
    for result in (res(127, secret, secret), res(0, secret, secret)):
        backend = FakeBackend(lambda *_args, r=result: r)
        r = Grader(backend, "img", tmp_path).grade(task, {})
        assert r.error is not None and secret not in r.error

    def boom(_sb: FakeSandbox, _cmd: str, _timeout: float) -> ExecResult:
        raise OSError(secret)

    r = Grader(FakeBackend(boom), "img", tmp_path).grade(task, {})
    assert r.error is not None and secret not in r.error
