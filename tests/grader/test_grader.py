from __future__ import annotations

import json
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
from iterant.grader.grader import MAX_RESULT_BYTES, parse_result, same
from iterant.packs import load_pack

OnExec = Callable[["FakeSandbox", str, float], ExecResult]
SYNTHETIC = Path(__file__).resolve().parents[1] / "packs" / "synthetic"


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


def make_task(root: Path, expected: object | None = None, run: bytes = b"echo run\n") -> Task:
    d = root / "tasks" / "t1"
    d.mkdir(parents=True)
    (d / "input.json").write_bytes(b"{}")
    (d / "run").write_bytes(run)
    (d / "check").write_bytes(json.dumps({"v": 1} if expected is None else expected).encode())
    return Task(
        "t1",
        Split.SELECT,
        (),
        VisibleInputPath(PurePosixPath("tasks/t1/input.json")),
        HiddenCheckPath(PurePosixPath("tasks/t1/check")),
    )


def says(obj: object) -> OnExec:
    return lambda *_a: res(0, json.dumps(obj))


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeBackend(), SandboxBackend)
    assert isinstance(FakeSandbox(ok, "default-deny"), Sandbox)


def test_grade_uses_fresh_default_deny_sandbox_and_destroys_it(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    backend = FakeBackend(says({"v": 1}))
    grader = Grader(backend, "img", tmp_path, timeout=30)
    r1 = grader.grade(task, {"answer.txt": b"42"})
    r2 = grader.grade(task, {"answer.txt": b"43"})
    assert (r1.score, r2.score) == (1.0, 1.0)
    assert len(backend.sandboxes) == 2
    assert all(a[0] == "img" and a[5] == "default-deny" for a in backend.create_args)
    for sb in backend.sandboxes:
        assert sb.destroyed == 1 and len(sb.cmds) == 1
    assert backend.sandboxes[0].files["/out/answer.txt"] == b"42"
    assert backend.sandboxes[1].files["/out/answer.txt"] == b"43"  # nothing carries over


def test_expected_values_never_reach_the_sandbox(tmp_path: Path) -> None:
    task = make_task(tmp_path, {"secret": "HIDDEN-EXPECTED"})
    backend = FakeBackend(says({"secret": "HIDDEN-EXPECTED"}))
    r = Grader(backend, "img", tmp_path).grade(task, {"a/b.txt": b"x"})
    assert r.score == 1.0
    sb = backend.sandboxes[0]
    assert sorted(sb.files) == ["/out/a/b.txt", "/run/run.sh"]
    assert sb.files["/run/run.sh"] == b"echo run\n"
    blob = b"".join(sb.files.values()) + "".join(sb.cmds).encode()
    assert b"HIDDEN-EXPECTED" not in blob


def test_verdict_compares_in_this_process(tmp_path: Path) -> None:
    task = make_task(tmp_path, {"v": 1})
    assert Grader(FakeBackend(says({"v": 1})), "i", tmp_path).grade(task, {}).score == 1.0
    assert Grader(FakeBackend(says({"v": 2})), "i", tmp_path).grade(task, {}).score == 0.0
    # the sandbox cannot claim a score: only the object it prints is compared
    forged = says({"v": 2, "score": 1.0})
    assert Grader(FakeBackend(forged), "i", tmp_path).grade(task, {}).score == 0.0


def test_out_files_cannot_overwrite_the_run_script_or_check(tmp_path: Path) -> None:
    backend = FakeBackend()
    task = make_task(tmp_path)
    for bad in ("../run/run.sh", "../grader/check", "/run/run.sh"):
        assert Grader(backend, "img", tmp_path).grade(task, {bad: b"exit 0"}).infra_err
    assert not backend.sandboxes  # refused before any sandbox exists


def test_exec_gets_timeout(tmp_path: Path) -> None:
    seen: list[float] = []

    def on_exec(_sb: FakeSandbox, _cmd: str, timeout: float) -> ExecResult:
        seen.append(timeout)
        return res(0, "{}")

    Grader(FakeBackend(on_exec), "img", tmp_path, timeout=7).grade(make_task(tmp_path, {}), {})
    assert seen == [7]


@pytest.mark.parametrize("bad", ["/etc/passwd", "../x", "a/../../x", "", "a\\b", "a\0b"])
def test_bad_out_paths_are_infra_errors_without_the_path(tmp_path: Path, bad: str) -> None:
    backend = FakeBackend()
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {bad: b""})
    assert r.error == "bad out file path" and r.score is None and not backend.sandboxes


@pytest.mark.parametrize("victim", ["check", "run"])
def test_missing_pack_file_is_infra_error(tmp_path: Path, victim: str) -> None:
    task = make_task(tmp_path)
    (tmp_path / "tasks" / "t1" / victim).unlink()
    assert Grader(FakeBackend(), "img", tmp_path).grade(task, {}).error == "cannot read pack files"


def test_bad_check_json_is_infra_error(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    (tmp_path / "tasks" / "t1" / "check").write_bytes(b"{nope")
    assert Grader(FakeBackend(), "img", tmp_path).grade(task, {}).infra_err


def test_create_failure_is_infra_error_without_backend_text(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.fail_create = RuntimeError("no podman at /home/secret")
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {})
    assert r.error == "sandbox create failed" and r.score is None


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

    backend = FakeBackend(says({"v": 1}), sandbox_cls=Bad)
    assert Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {}).score == 1.0


@pytest.mark.parametrize(
    ("result", "score", "is_error"),
    [
        (res(0, '{"v": 1}'), 1.0, False),
        (res(0, ' {"v": 1.0}\n'), 1.0, False),
        (res(0, '{"v": 2}'), 0.0, False),
        (res(0, '{"v": true}'), 0.0, False),  # types stay apart
        (res(1, '{"v": 1}'), 0.0, False),  # a failing exit never scores
        (res(0, ""), 0.0, False),  # missing result
        (res(0, 'log line\n{"v": 1}'), 0.0, False),  # not exactly one object
        (res(0, '{"v": 1}{"v": 1}'), 0.0, False),
        (res(0, '{"v": NaN}'), 0.0, False),
        (res(0, '{"v": Infinity}'), 0.0, False),
        (res(0, '{"v": 1, "v": 1}'), 0.0, False),  # duplicate key
        (res(0, "[1]"), 0.0, False),  # not an object
        (res(0, "0.75"), 0.0, False),  # a bare score is not a result
        (res(-1, "", "", True), 0.0, False),  # timeouts, kills and OOM are the candidate's doing
        (res(137), 0.0, False),
        (res(2), 0.0, False),
        (res(127, "", "sh: not found"), None, True),  # could not run
        (res(126), None, True),
    ],
)
def test_interpret(result: ExecResult, score: float | None, is_error: bool) -> None:
    got, err = interpret(result, {"v": 1})
    assert got == score and (err is not None) == is_error


def test_result_size_and_depth_caps() -> None:
    big = json.dumps({"v": "x" * MAX_RESULT_BYTES})
    deep = '{"v": ' + "[" * 40 + "]" * 40 + "}"
    huge = '{"v": ' + "[" * 100000 + "]" * 100000 + "}"
    for bad in (big, deep, huge):
        with pytest.raises(ValueError):
            parse_result(bad)
    parse_result('{"v": ' + "[" * 20 + "]" * 20 + "}")


def test_same() -> None:
    assert same({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not same(1, True) and not same(0, False) and not same("1", 1)
    assert not same({"a": 1}, {"a": 1, "b": 2}) and not same([1], [1, 2])


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
        assert r.error is None or secret not in r.error

    def boom(_sb: FakeSandbox, _cmd: str, _timeout: float) -> ExecResult:
        raise OSError(secret)

    r = Grader(FakeBackend(boom), "img", tmp_path).grade(task, {})
    assert r.error is not None and secret not in r.error


class LocalBackend:
    """Runs the grader's command with the local ``sh``, mapping ``/out`` and ``/run``."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.count = 0

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        self.count += 1
        return LocalSandbox(ok, net, base=self.base / f"sb{self.count}")


@dataclass
class LocalSandbox(FakeSandbox):
    base: Path = field(default_factory=Path)

    def put(self, path: str, data: bytes) -> None:
        target = self.base / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        sh = shutil.which("sh")
        assert sh is not None
        local = cmd.replace("/out", (self.base / "out").as_posix())
        local = local.replace("/run/run.sh", (self.base / "run" / "run.sh").as_posix())
        p = subprocess.run([sh, "-c", local], capture_output=True, text=True, timeout=timeout)
        return ExecResult(p.returncode, p.stdout, p.stderr, False, 0.1)


@pytest.mark.skipif(shutil.which("sh") is None, reason="no sh")
def test_synthetic_pack_end_to_end(tmp_path: Path) -> None:
    task = load_pack(SYNTHETIC).tasks[3]
    data = json.loads((SYNTHETIC / task.input).read_bytes())
    total = data["a"] + data["b"]
    grader = Grader(LocalBackend(tmp_path), "img", SYNTHETIC)

    def solve(script: str) -> dict[str, bytes]:
        return {"solve.sh": script.encode()}

    good = grader.grade(task, solve(f"printf '{{\"sum\": {total}}}'\n"))
    wrong = grader.grade(task, solve("printf '{\"sum\": -1}'\n"))
    crash = grader.grade(task, solve("exit 3\n"))
    forged = grader.grade(task, solve('printf \'{"sum": 1, "score": 1}\'\n'))
    assert (good.score, wrong.score, crash.score, forged.score) == (1.0, 0.0, 0.0, 0.0)
