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
from iterant.grader.grader import DEFAULT_TIMEOUT_S, MAX_RESULT_BYTES, RUN_PATH, parse_result, same
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
    preflight: Callable[[float], ExecResult] = lambda _t: res()

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.cmds.append(cmd)
        if cmd.startswith("mkdir -p"):  # the grader's trusted preflight
            return self.preflight(timeout)
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
        self.preflight: Callable[[float], ExecResult] | None = None

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        if self.fail_create:
            raise self.fail_create
        self.create_args.append((image, cpu, mem, disk, ttl, net))
        sb = self.sandbox_cls(self.on_exec, net)
        if self.preflight is not None:
            sb.preflight = self.preflight
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
        assert sb.destroyed == 1 and len(sb.cmds) == 2
    assert backend.sandboxes[0].files["/work/out/answer.txt"] == b"42"
    assert backend.sandboxes[1].files["/work/out/answer.txt"] == b"43"  # nothing carries over


def test_expected_values_never_reach_the_sandbox(tmp_path: Path) -> None:
    task = make_task(tmp_path, {"secret": "HIDDEN-EXPECTED"})
    backend = FakeBackend(says({"secret": "HIDDEN-EXPECTED"}))
    r = Grader(backend, "img", tmp_path).grade(task, {"a/b.txt": b"x"})
    assert r.score == 1.0
    sb = backend.sandboxes[0]
    assert sorted(sb.files) == [RUN_PATH, "/work/out/a/b.txt"]
    assert sb.files[RUN_PATH] == b"echo run\n"
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
    for bad in ("../run/run.sh", "../grader/check", RUN_PATH):
        assert Grader(backend, "img", tmp_path).grade(task, {bad: b"exit 0"}).score == 0.0
    assert not backend.sandboxes  # refused before any sandbox exists


def test_exec_gets_timeout(tmp_path: Path) -> None:
    seen: list[float] = []

    def on_exec(_sb: FakeSandbox, _cmd: str, timeout: float) -> ExecResult:
        seen.append(timeout)
        return res(0, "{}")

    Grader(FakeBackend(on_exec), "img", tmp_path, timeout=7).grade(make_task(tmp_path, {}), {})
    assert seen == [7]


@pytest.mark.parametrize(
    "bad",
    ["/etc/passwd", "../x", "a/../../x", "", "a\\b", "a\0b", "a//b", "a/./b", ".", "a/", "./a"],
)
def test_bad_out_paths_score_zero(tmp_path: Path, bad: str) -> None:
    backend = FakeBackend()
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {bad: b""})
    assert r.score == 0.0 and r.error is None and not backend.sandboxes


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
        (res(127, "", "sh: not found"), 0.0, False),  # agent code can exit 126/127 too
        (res(126), 0.0, False),
        (res(-1, '{"v": 1}', "", True), 0.0, False),
        (res(0, '{"v": 1e999}'), 0.0, False),
    ],
)
def test_interpret(result: ExecResult, score: float | None, is_error: bool) -> None:
    assert interpret(result, {"v": 1}) == score and not is_error


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
        local = cmd.replace("/work", (self.base / "work").as_posix())
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


def test_oversized_output_scores_zero_without_a_sandbox(tmp_path: Path) -> None:
    backend = FakeBackend(says({"v": 1}))
    many = {f"f{i}": b"" for i in range(1001)}
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), many)
    assert r.score == 0.0 and not backend.sandboxes


def test_timeout_with_a_correct_result_scores_zero(tmp_path: Path) -> None:
    backend = FakeBackend(lambda *_a: res(-1, '{"v": 1}', "", True))
    assert Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {}).score == 0.0


def test_default_wall_clock_limit_is_a_named_constant(tmp_path: Path) -> None:
    seen: list[float] = []

    def on_exec(_sb: FakeSandbox, _cmd: str, timeout: float) -> ExecResult:
        seen.append(timeout)
        return res(0, "{}")

    Grader(FakeBackend(on_exec), "img", tmp_path).grade(make_task(tmp_path, {}), {})
    assert seen == [DEFAULT_TIMEOUT_S]


def test_agent_put_failure_scores_zero_but_trusted_failures_are_infra(tmp_path: Path) -> None:
    class PutFails(FakeSandbox):
        def put(self, path: str, data: bytes) -> None:
            if path.startswith("/work/out/"):
                raise OSError("agent file conflict")
            super().put(path, data)

    task = make_task(tmp_path)
    r = Grader(FakeBackend(says({"v": 1}), PutFails), "img", tmp_path).grade(task, {"a": b"x"})
    assert r.score == 0.0 and r.error is None

    class TrustedPutFails(FakeSandbox):
        def put(self, path: str, data: bytes) -> None:
            raise OSError("disk")

    r = Grader(FakeBackend(says({"v": 1}), TrustedPutFails), "img", tmp_path).grade(task, {})
    assert r.error == "sandbox error: OSError"


@pytest.mark.parametrize("pre", [res(1), res(-1, "", "", True)])
def test_failed_preflight_is_infra_error_and_agent_code_never_runs(
    tmp_path: Path, pre: ExecResult
) -> None:
    backend = FakeBackend(says({"v": 1}))
    backend.preflight = lambda _t: pre
    r = Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {"a": b"x"})
    assert r.error == "sandbox preflight failed"
    assert all(c.startswith("mkdir -p") for c in backend.sandboxes[0].cmds)
    assert "/work/out/a" not in backend.sandboxes[0].files


def test_agent_exit_126_127_cannot_hide_a_failure(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    for code in (126, 127):
        backend = FakeBackend(lambda *_a, c=code: res(c))
        r = Grader(backend, "img", tmp_path).grade(task, {})
        assert r.score == 0.0 and r.error is None


def test_agent_paths_stay_under_the_work_dir(tmp_path: Path) -> None:
    backend = FakeBackend(says({"v": 1}))
    Grader(backend, "img", tmp_path).grade(make_task(tmp_path), {"x/y.txt": b"1"})
    assert all(p.startswith("/work/") for p in backend.sandboxes[0].files)


def test_bad_check_file_values_are_infra_errors(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    for bad in (b'{"v": 1e999}', b'{"v": NaN}', b"[1]", b'{"a": 1, "a": 2}'):
        (tmp_path / "tasks" / "t1" / "check").write_bytes(bad)
        r = Grader(FakeBackend(), "img", tmp_path).grade(task, {})
        assert r.error == "cannot read pack files"
