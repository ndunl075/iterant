import dataclasses
import sqlite3
from pathlib import PurePosixPath

import pytest

from iterant.contracts import (
    ARCHIVE_DDL,
    AcceptanceThresholds,
    Batch,
    Candidate,
    EvalConfig,
    ExecResult,
    HiddenCheckPath,
    IterantConfig,
    NetMode,
    Pack,
    Profile,
    Promotion,
    ProxyCall,
    ProxyPolicy,
    Role,
    RoleCap,
    RunKey,
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
    return Candidate(cid, parent, SHA, "h", Stage.PROPOSE, Status.ACTIVE, 0)


def make_run(
    cand: str = "c1", task: str = "t1", rep: int = 0, attempt: int = 0, batch: int = 1
) -> RunResult:
    key = RunKey(batch, cand, task, rep, attempt)
    return RunResult(key, Split.SELECT, 1.0, 10, 5, 10_000, 1.5, False, False)


def test_enum_values() -> None:
    assert [s.value for s in Split] == ["train", "select", "holdout", "test"]
    assert [s.value for s in Stage] == ["propose", "gate", "smoke", "select", "confirm", "promote"]
    assert {r.value for r in Role} == {"target", "proposer", "analyst", "auditor", "transfer"}
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
        Task("", Split.TRAIN, (), VisibleInputPath(same), HiddenCheckPath(PurePosixPath("b")))


def test_config_types() -> None:
    model = TargetModel("claude-haiku-5-5", "medium", max_usd_micros_per_run=100_000)
    target = TargetConfig(("prompts/**",), "Containerfile", "python -m agent", model)
    cfg = IterantConfig(target, EvalConfig("evals/support-pack"))
    assert cfg.target.model.effort == "medium"
    assert TargetModel("m", "low", max_tokens_per_run=5).max_usd_micros_per_run is None
    with pytest.raises(ValueError):
        TargetModel("m", "low")
    with pytest.raises(ValueError):
        TargetModel("m", "low", max_usd_micros_per_run=0)
    with pytest.raises(ValueError):
        TargetConfig((), "c", "r", model)


def test_frozen() -> None:
    objs: list[object] = [
        make_task(),
        make_candidate(),
        make_run(),
        make_run().key,
        AcceptanceThresholds(),
        ExecResult(0, "", "", False, 0.1),
        Promotion(1, "c1", 0.1, 0.01, 0.1, 0.1),
        Batch(1, 1, SHA, "2026-01-01T00:00:00Z"),
        ProxyCall(1, None, Role.PROPOSER, "m", 1, 1, 1, False, "t"),
        RoleCap("m", "low", 1, 2),
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
        Candidate("c", None, SHA, "h", Stage.GATE, Status.ACTIVE, -1)
    with pytest.raises(ValueError):
        RunKey(-1, "c", "t", 0)
    with pytest.raises(ValueError):
        Promotion(0, "c", 0, 0.5, 0, 0.5)
    with pytest.raises(ValueError):
        AcceptanceThresholds(select_reps=0)
    with pytest.raises(ValueError):
        Batch(1, 1, "short", "t")


def test_run_score_null_iff_failed() -> None:
    key = RunKey(1, "c", "t", 0)
    with pytest.raises(ValueError):
        RunResult(key, Split.TRAIN, 1.5, 0, 0, 0, 0, False, False)
    with pytest.raises(ValueError):
        RunResult(key, Split.TRAIN, None, 0, 0, 0, 0, False, False)
    with pytest.raises(ValueError):
        RunResult(key, Split.TRAIN, 0.5, 0, 0, 0, 0, False, True)
    assert RunResult(key, Split.TRAIN, None, 0, 0, 0, 0, True, False).score is None


def test_thresholds_defaults() -> None:
    t = AcceptanceThresholds()
    assert (t.delta_min, t.select_p, t.holdout_p) == (0.05, 0.05, 0.2)
    assert (t.max_cost_increase, t.max_wall_increase) == (0.15, 0.25)
    assert (t.select_tasks, t.select_reps, t.holdout_tasks, t.holdout_reps) == (80, 2, 40, 2)
    assert (t.holdout_query_warn, t.max_diff_lines, t.smoke_tasks) == (20, 400, 5)


def test_proxy_policy() -> None:
    free = RoleCap("llama", "medium", max_tokens_per_run=1000, max_tokens_per_gen=50_000)
    paid = RoleCap(
        "claude-opus-5-5",
        "high",
        max_usd_micros_per_run=2_000_000,
        max_usd_micros_per_gen=9_000_000,
    )
    roles = {Role.PROPOSER: free}
    policy = ProxyPolicy(Profile.FREE, roles)
    assert policy.roles[Role.PROPOSER] is free
    roles[Role.TARGET] = free  # mutating the caller's dict must not leak in
    assert Role.TARGET not in policy.roles
    with pytest.raises(TypeError):
        policy.roles[Role.TARGET] = free  # type: ignore[index]
    assert ProxyPolicy(Profile.CLAUDE, {Role.PROPOSER: paid}).profile is Profile.CLAUDE
    with pytest.raises(ValueError):
        ProxyPolicy(Profile.FREE, {Role.PROPOSER: paid})
    with pytest.raises(ValueError):
        ProxyPolicy(Profile.CLAUDE, {Role.PROPOSER: free})


def test_role_cap_validation() -> None:
    with pytest.raises(ValueError):
        RoleCap("m", "low")  # no caps
    with pytest.raises(ValueError):
        RoleCap("m", "low", max_tokens_per_run=1)  # half a pair
    with pytest.raises(ValueError):
        RoleCap("m", "low", 1, 2, 1, 2)  # both pairs
    with pytest.raises(ValueError):
        RoleCap("m", "low", 5, 2)  # per_run > per_gen
    with pytest.raises(ValueError):
        RoleCap("m", "low", max_usd_micros_per_run=0, max_usd_micros_per_gen=1)


# --- archive DDL --------------------------------------------------------------


def db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(ARCHIVE_DDL)
    con.executescript(ARCHIVE_DDL)  # idempotent
    con.execute("INSERT INTO batches VALUES (1, 1, ?, '2026-01-01T00:00:00Z')", (SHA,))
    con.execute(
        "INSERT INTO candidates VALUES ('c1', NULL, ?, 'h', 'propose', 'active', 0)", (SHA,)
    )
    return con


def insert_run(con: sqlite3.Connection, r: RunResult, verb: str = "INSERT") -> None:
    k = r.key
    con.execute(
        f"{verb} INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (k.batch, k.cand, k.task, k.rep, k.attempt, r.split.value, r.score, r.in_tok, r.out_tok)
        + (r.usd_micros, r.wall_s, r.refusal, r.infra_err),
    )


def add_call(
    con: sqlite3.Connection, cid: int, key: tuple[int | str | None, ...] | None, role: str
) -> None:
    cols = key if key is not None else (None,) * 5
    con.execute(
        "INSERT INTO proxy_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, *cols, role, "m", 1, 1, 1, 0, "t"),
    )


def test_ddl_columns_match_dataclasses() -> None:
    def names(cls: type) -> list[str]:
        out: list[str] = []
        for f in dataclasses.fields(cls):
            if f.name == "key":
                out += [kf.name for kf in dataclasses.fields(RunKey)]
            else:
                out.append(f.name)
        return out

    con = db()
    tables = (
        ("batches", Batch),
        ("candidates", Candidate),
        ("runs", RunResult),
        ("promotions", Promotion),
        ("proxy_calls", ProxyCall),
    )
    for table, cls in tables:
        cols = [row[1] for row in con.execute(f"PRAGMA table_info({table})")]
        assert cols == names(cls), table


def test_tables_are_strict() -> None:
    con = db()
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO batches VALUES ('not-int', 1, ?, 't')", (SHA,))


def test_runs_primary_key_and_attempts() -> None:
    con = db()
    insert_run(con, make_run())
    insert_run(con, make_run(rep=1))
    insert_run(con, make_run(attempt=1))  # retry of the same (batch, cand, task, rep)
    insert_run(con, make_run(batch=1, cand="c1", task="t2"))
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(con, make_run())
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(con, make_run(cand="missing"))
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(con, make_run(batch=99))
    latest = con.execute(
        "SELECT MAX(attempt) FROM runs WHERE batch=1 AND cand='c1' AND task='t1' AND rep=0"
    ).fetchone()
    assert latest == (1,)


def test_champion_arm_can_repeat_across_batches() -> None:
    con = db()
    con.execute("INSERT INTO batches VALUES (2, 1, ?, 't')", (SHA,))
    insert_run(con, make_run(batch=1))
    insert_run(con, make_run(batch=2))


def test_score_null_check_in_sql() -> None:
    con = db()
    bad = (1, "c1", "t1", 0, 0, "select", None, 0, 0, 0, 0.0, 0, 0)  # NULL score, no failure
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", bad)
    bad2 = (1, "c1", "t1", 0, 0, "select", 0.5, 0, 0, 0, 0.0, 1, 0)  # score with refusal
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", bad2)


def test_insert_or_replace_cannot_overwrite() -> None:
    con = db()
    insert_run(con, make_run())
    con.execute("INSERT INTO promotions VALUES (1, 'c1', 0.1, 0.01, 0.1, 0.1)")
    add_call(con, 1, None, "proposer")
    for verb in ("INSERT OR REPLACE", "REPLACE", "INSERT OR IGNORE"):
        with pytest.raises(sqlite3.DatabaseError):
            insert_run(con, make_run(), verb)
        stmts = (
            f"{verb} INTO candidates VALUES ('c1', NULL, '{'b' * 64}', 'x', 'gate', 'rejected', 9)",
            f"{verb} INTO batches VALUES (1, 7, '{'b' * 64}', 'x')",
            f"{verb} INTO promotions VALUES (1, 'c1', 9, 0.5, 9, 0.5)",
            f"{verb} INTO proxy_calls VALUES (1, NULL, NULL, NULL, NULL, NULL, 'target',"
            " 'x', 9, 9, 9, 1, 'x')",
        )
        for sql in stmts:
            with pytest.raises(sqlite3.DatabaseError):
                con.execute(sql)
    assert con.execute("SELECT hypothesis, status, usd_micros FROM candidates").fetchall() == [
        ("h", "active", 0)
    ]
    assert con.execute("SELECT in_tok, usd_micros FROM runs").fetchall() == [(10, 10_000)]
    assert con.execute("SELECT select_delta FROM promotions").fetchall() == [(0.1,)]
    assert con.execute("SELECT gen FROM batches").fetchall() == [(1,)]
    assert con.execute("SELECT model, in_tok FROM proxy_calls").fetchall() == [("m", 1)]


def test_promotions_cand_unique() -> None:
    con = db()
    con.execute("INSERT INTO promotions VALUES (1, 'c1', 0.1, 0.01, 0.1, 0.1)")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("INSERT INTO promotions VALUES (2, 'c1', 0.1, 0.01, 0.1, 0.1)")


def test_proxy_calls_key_all_or_none() -> None:
    con = db()
    add_call(con, 1, (1, "c1", "t1", 0, 0), "target")
    add_call(con, 2, None, "proposer")
    with pytest.raises(sqlite3.IntegrityError):
        add_call(con, 3, (1, None, None, None, None), "target")
    with pytest.raises(sqlite3.IntegrityError):
        add_call(con, 4, None, "grader")


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE runs SET in_tok = 0",
        "DELETE FROM runs",
        "UPDATE batches SET gen = 2",
        "DELETE FROM batches",
        "UPDATE promotions SET gen = 2",
        "DELETE FROM promotions",
        "UPDATE proxy_calls SET in_tok = 0",
        "DELETE FROM proxy_calls",
        "DELETE FROM candidates",
        "UPDATE candidates SET hypothesis = 'x'",
        "UPDATE candidates SET parent = 'c1'",
        "UPDATE candidates SET stage = 'bogus'",
    ],
)
def test_append_only_statements_rejected(sql: str) -> None:
    con = db()
    insert_run(con, make_run())
    con.execute("INSERT INTO promotions VALUES (1, 'c1', 0.1, 0.01, 0.1, 0.1)")
    add_call(con, 1, None, "proposer")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute(sql)


def test_candidate_monotonic() -> None:
    con = db()
    con.execute("UPDATE candidates SET stage = 'gate', usd_micros = 5 WHERE id = 'c1'")
    con.execute("UPDATE candidates SET stage = 'gate', usd_micros = 5 WHERE id = 'c1'")  # no-op ok
    for sql in (
        "UPDATE candidates SET stage = 'propose'",  # backwards
        "UPDATE candidates SET usd_micros = 4",  # spend decreases
    ):
        with pytest.raises(sqlite3.DatabaseError):
            con.execute(sql)
    con.execute("UPDATE candidates SET stage = 'select', status = 'rejected', usd_micros = 7")
    for sql in (
        "UPDATE candidates SET status = 'active'",  # terminal -> active
        "UPDATE candidates SET status = 'promoted'",  # terminal -> other terminal
        "UPDATE candidates SET usd_micros = 8",  # terminal rows are frozen
        "UPDATE candidates SET stage = 'confirm'",
    ):
        with pytest.raises(sqlite3.DatabaseError):
            con.execute(sql)
    assert con.execute("SELECT stage, status, usd_micros FROM candidates").fetchall() == [
        ("select", "rejected", 7)
    ]


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
