import dataclasses
import sqlite3
from pathlib import PurePosixPath

import pytest

from iterant.contracts import (
    ARCHIVE_DDL,
    AcceptanceThresholds,
    Candidate,
    EvalConfig,
    ExecResult,
    HiddenCheckPath,
    IterantConfig,
    NetMode,
    Pack,
    Profile,
    Promotion,
    ProxyPolicy,
    Role,
    RoleCap,
    RunResult,
    Sandbox,
    SandboxBackend,
    Split,
    Stage,
    Status,
    TargetConfig,
    TargetModel,
    Task,
    VisibleInputPath,
)

SHA = "a" * 64


def make_task(tid: str = "t1") -> Task:
    return Task(
        id=tid,
        split=Split.TRAIN,
        tags=("easy",),
        input=VisibleInputPath(PurePosixPath(f"tasks/{tid}/input.json")),
        check=HiddenCheckPath(PurePosixPath(f"tasks/{tid}/check")),
    )


def make_candidate(cid: str = "c1", parent: str | None = None) -> Candidate:
    return Candidate(cid, parent, SHA, "h", Stage.PROPOSE, Status.ACTIVE, 0.0)


def make_run(cand: str = "c1", task: str = "t1", rep: int = 0) -> RunResult:
    return RunResult(cand, task, Split.SELECT, rep, 1.0, 10, 5, 0.01, 1.5, False, False)


def test_enum_values() -> None:
    assert [s.value for s in Split] == ["train", "select", "holdout", "test"]
    assert [s.value for s in Stage] == [
        "propose",
        "gate",
        "smoke",
        "select",
        "confirm",
        "promote",
    ]
    assert {r.value for r in Role} == {
        "target",
        "proposer",
        "analyst",
        "auditor",
        "transfer",
    }
    assert {p.value for p in Profile} == {"free", "claude"}
    assert {s.value for s in Status} == {"active", "rejected", "promoted", "error"}


def test_task_and_pack() -> None:
    pack = Pack("p", "1", (make_task("a"), make_task("b")), SHA)
    assert pack.tasks[0].id == "a"
    with pytest.raises(ValueError):
        Pack("p", "1", (make_task("a"), make_task("a")), SHA)
    with pytest.raises(ValueError):
        Pack("p", "1", (), "XYZ")
    same = PurePosixPath("a")
    with pytest.raises(ValueError):
        Task("t", Split.TRAIN, (), VisibleInputPath(same), HiddenCheckPath(same))
    with pytest.raises(ValueError):
        Task(
            "",
            Split.TRAIN,
            (),
            VisibleInputPath(same),
            HiddenCheckPath(PurePosixPath("b")),
        )


def test_config_types() -> None:
    model = TargetModel("claude-haiku-5-5", "medium", 0.10)
    target = TargetConfig(("prompts/**",), "Containerfile", "python -m agent", model)
    cfg = IterantConfig(target, EvalConfig("evals/support-pack"))
    assert cfg.target.model.effort == "medium"
    with pytest.raises(ValueError):
        TargetModel("m", "low", 0)
    with pytest.raises(ValueError):
        TargetConfig((), "c", "r", model)


def test_frozen() -> None:
    objs: list[object] = [
        make_task(),
        make_candidate(),
        make_run(),
        AcceptanceThresholds(),
        ExecResult(0, "", "", False, 0.1),
        Promotion(1, "c1", 0.1, 0.01, 0.1, 0.1),
        RoleCap("m", "low", max_tokens=1),
        ProxyPolicy(Profile.FREE),
        Pack("p", "1", (), SHA),
    ]
    for obj in objs:
        first = dataclasses.fields(obj)[0].name  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, first, None)


def test_validation() -> None:
    with pytest.raises(ValueError):
        Candidate("c", None, "bad", "h", Stage.GATE, Status.ACTIVE, 0)
    with pytest.raises(ValueError):
        RunResult("c", "t", Split.TRAIN, 0, 1.5, 0, 0, 0, 0, False, False)
    with pytest.raises(ValueError):
        Promotion(0, "c", 0, 0.5, 0, 0.5)
    with pytest.raises(ValueError):
        AcceptanceThresholds(select_reps=0)


def test_thresholds_defaults() -> None:
    t = AcceptanceThresholds()
    assert (t.delta_min, t.select_p, t.holdout_p) == (0.05, 0.05, 0.2)
    assert (t.max_cost_increase, t.max_wall_increase) == (0.15, 0.25)
    assert (t.select_tasks, t.select_reps, t.holdout_tasks, t.holdout_reps) == (
        80,
        2,
        40,
        2,
    )


def test_proxy_policy() -> None:
    free = RoleCap("llama", "medium", max_tokens=1000)
    paid = RoleCap("claude-opus-5-5", "high", max_usd=2.0)
    assert ProxyPolicy(Profile.FREE, {Role.PROPOSER: free}).roles[Role.PROPOSER] is free
    assert ProxyPolicy(Profile.CLAUDE, {Role.PROPOSER: paid}).profile is Profile.CLAUDE
    with pytest.raises(ValueError):
        ProxyPolicy(Profile.FREE, {Role.PROPOSER: paid})
    with pytest.raises(ValueError):
        ProxyPolicy(Profile.CLAUDE, {Role.PROPOSER: free})
    with pytest.raises(ValueError):
        RoleCap("m", "low")
    with pytest.raises(ValueError):
        RoleCap("m", "low", max_tokens=1, max_usd=1.0)


def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ARCHIVE_DDL)
    con.executescript(ARCHIVE_DDL)  # idempotent
    return con


def insert_run(con: sqlite3.Connection, r: RunResult) -> None:
    con.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            r.cand,
            r.task,
            r.split.value,
            r.rep,
            r.score,
            r.in_tok,
            r.out_tok,
            r.usd,
            r.wall_s,
            r.refusal,
            r.infra_err,
        ),
    )


def test_ddl_columns_match_dataclasses() -> None:
    con = db()
    for table, cls in (
        ("candidates", Candidate),
        ("runs", RunResult),
        ("promotions", Promotion),
    ):
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
        assert cols == [f.name for f in dataclasses.fields(cls)]


def test_runs_primary_key_rejects_duplicates() -> None:
    con = db()
    con.execute(
        "INSERT INTO candidates VALUES ('c1', NULL, ?, 'h', 'propose', 'active', 0)",
        (SHA,),
    )
    insert_run(con, make_run())
    insert_run(con, make_run(rep=1))
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(con, make_run())
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(con, make_run(cand="missing"))


def test_append_only() -> None:
    con = db()
    con.execute(
        "INSERT INTO candidates VALUES ('c1', NULL, ?, 'h', 'propose', 'active', 0)",
        (SHA,),
    )
    insert_run(con, make_run())
    con.execute("INSERT INTO promotions VALUES (1, 'c1', 0.1, 0.01, 0.1, 0.1)")
    con.execute(
        "UPDATE candidates SET stage = 'gate', status = 'rejected', usd = 1 WHERE id = 'c1'"
    )
    for sql in (
        "UPDATE runs SET score = 0",
        "DELETE FROM runs",
        "UPDATE promotions SET gen = 2",
        "DELETE FROM promotions",
        "DELETE FROM candidates",
        "UPDATE candidates SET hypothesis = 'x'",
        "UPDATE candidates SET stage = 'bogus'",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            con.execute(sql)


class FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        return self.files[path]

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        return ExecResult(0, cmd, "", False, 0.0)

    def destroy(self) -> None:
        self.files.clear()


class FakeBackend:
    def create(
        self,
        image: str,
        cpu: float,
        mem: int,
        disk: int,
        ttl: float,
        net: NetMode = "proxy-only",
    ) -> Sandbox:
        return FakeSandbox()


def test_fakes_satisfy_protocols() -> None:
    backend: SandboxBackend = FakeBackend()
    box: Sandbox = backend.create("img", 1, 512, 1024, 60)
    box.put("/task/input.json", b"{}")
    assert box.get("/task/input.json") == b"{}"
    assert box.exec("echo hi", 5).stdout == "echo hi"
    box.destroy()
    assert isinstance(box, Sandbox)
    assert isinstance(backend, SandboxBackend)
