import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from iterant.archive import (
    SCHEMA_VERSION,
    AppendOnlyViolation,
    Archive,
    ArchiveProxyLog,
    ConstraintViolation,
    DuplicateRowError,
    ForeignKeyViolation,
    InvalidTransition,
    PackHashMismatch,
    SchemaVersionError,
    UnknownRow,
    archive_path,
)
from iterant.contracts import (
    Batch,
    Candidate,
    Promotion,
    ProxyCall,
    Role,
    RunKey,
    RunResult,
    Split,
    Stage,
    Status,
)

H = "a" * 64
H2 = "b" * 64


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[Archive]:
    with Archive.open(tmp_path) as a:
        yield a


def cand(cid: str = "c1", parent: str | None = None, **kw: object) -> Candidate:
    base = dict(
        id=cid,
        parent=parent,
        diff_sha=H,
        hypothesis="h",
        stage=Stage.PROPOSE,
        status=Status.ACTIVE,
        usd_micros=0,
    )
    return Candidate(**{**base, **kw})  # type: ignore[arg-type]


def run(
    batch: int = 1,
    c: str = "c1",
    task: str = "t1",
    rep: int = 0,
    attempt: int = 0,
    split: Split = Split.SELECT,
    score: float | None = 1.0,
    refusal: bool = False,
    infra_err: bool = False,
) -> RunResult:
    return RunResult(
        key=RunKey(batch, c, task, rep, attempt),
        split=split,
        score=score,
        in_tok=1,
        out_tok=2,
        usd_micros=3,
        wall_s=0.5,
        refusal=refusal,
        infra_err=infra_err,
    )


def seed(a: Archive) -> None:
    a.insert_batch(Batch(id=1, gen=1, pack_hash=H, started_at="2026-01-01T00:00:00+00:00"))
    a.insert_candidate(cand())


def test_schema_created_idempotent_and_versioned(tmp_path: Path) -> None:
    with Archive.open(tmp_path) as a:
        assert a.schema_version == SCHEMA_VERSION
        a.insert_candidate(cand())
    assert archive_path(tmp_path).exists()
    with Archive.open(tmp_path) as a:  # reopen: no error, data intact
        assert a.get_candidate("c1") == cand()


def test_newer_schema_version_refused(tmp_path: Path) -> None:
    with Archive.open(tmp_path):
        pass
    con = sqlite3.connect(archive_path(tmp_path))
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    con.close()
    with pytest.raises(SchemaVersionError):
        Archive.open(tmp_path)


def test_foreign_keys_enforced(archive: Archive) -> None:
    with pytest.raises(ForeignKeyViolation):
        archive.insert_run(run())  # no batch, no candidate


def test_connection_closed_on_context_exit(tmp_path: Path) -> None:
    with Archive.open(tmp_path) as a:
        con = a._con  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_insert_roundtrip_and_duplicates(archive: Archive) -> None:
    seed(archive)
    assert archive.get_batch(1) == Batch(1, 1, H, "2026-01-01T00:00:00+00:00")
    archive.insert_run(run())
    with pytest.raises(DuplicateRowError):
        archive.insert_run(run())
    with pytest.raises(DuplicateRowError):
        archive.insert_candidate(cand())
    with pytest.raises(DuplicateRowError):
        archive.insert_batch(Batch(1, 2, H2, "x"))
    assert archive.get_candidate("nope") is None


def test_create_batch_assigns_ids(archive: Archive) -> None:
    b1 = archive.create_batch(1, H)
    b2 = archive.create_batch(1, H2)
    assert b2.id == b1.id + 1
    assert archive.batch_pack_hash(b2.id) == H2


def test_no_update_delete_paths_on_append_only_tables(archive: Archive) -> None:
    seed(archive)
    archive.insert_run(run())
    con = archive._con  # pyright: ignore[reportPrivateUsage]
    for sql in (
        "UPDATE runs SET score = 0",
        "DELETE FROM runs",
        "UPDATE batches SET gen = 5",
        "DELETE FROM candidates",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(sql)
    # and the translation maps them to our type
    from iterant.archive.errors import translate_integrity_error

    try:
        con.execute("DELETE FROM runs")
    except sqlite3.IntegrityError as exc:
        assert isinstance(translate_integrity_error(exc), AppendOnlyViolation)


def test_candidate_transitions(archive: Archive) -> None:
    seed(archive)
    archive.advance_stage("c1", Stage.GATE)
    archive.advance_stage("c1", Stage.GATE)  # same stage: no-op
    archive.advance_stage("c1", Stage.SELECT)
    with pytest.raises(InvalidTransition):
        archive.advance_stage("c1", Stage.SMOKE)
    assert archive.add_spend("c1", 40) == 40
    assert archive.add_spend("c1", 2) == 42
    with pytest.raises(ValueError):
        archive.add_spend("c1", -1)
    with pytest.raises(ValueError):
        archive.finish_candidate("c1", Status.ACTIVE)
    archive.finish_candidate("c1", Status.REJECTED)
    got = archive.get_candidate("c1")
    assert got is not None
    assert (got.stage, got.status, got.usd_micros) == (Stage.SELECT, Status.REJECTED, 42)
    for bad in (
        lambda: archive.advance_stage("c1", Stage.CONFIRM),
        lambda: archive.add_spend("c1", 1),
        lambda: archive.finish_candidate("c1", Status.ERROR),
    ):
        with pytest.raises(InvalidTransition):
            bad()


def test_promoted_requires_promote_stage(archive: Archive) -> None:
    seed(archive)
    with pytest.raises(ConstraintViolation):
        archive.finish_candidate("c1", Status.PROMOTED)
    archive.finish_candidate("c1", Status.PROMOTED, stage=Stage.PROMOTE)
    got = archive.get_candidate("c1")
    assert got is not None and got.status is Status.PROMOTED


def test_unknown_candidate(archive: Archive) -> None:
    for call in (
        lambda: archive.advance_stage("zz", Stage.GATE),
        lambda: archive.add_spend("zz", 1),
        lambda: archive.finish_candidate("zz", Status.ERROR),
    ):
        with pytest.raises(UnknownRow):
            call()


def test_promotion(archive: Archive) -> None:
    seed(archive)
    p = Promotion(
        gen=1, cand="c1", select_delta=0.1, select_p=0.01, holdout_delta=0.05, holdout_p=0.1
    )
    assert archive.latest_promotion() is None
    archive.insert_promotion(p)
    assert archive.get_promotion(1) == p
    assert archive.latest_promotion() == p
    with pytest.raises(DuplicateRowError):
        archive.insert_promotion(Promotion(2, "c1", 0.1, 0.01, 0.05, 0.1))  # cand reused


def test_latest_runs_takes_highest_attempt(archive: Archive) -> None:
    seed(archive)
    archive.insert_run(run(task="t1", score=None, refusal=True))
    archive.insert_run(run(task="t1", attempt=1, score=0.5))
    archive.insert_run(run(task="t2", score=0.0))
    archive.insert_run(run(task="t1", rep=1, score=1.0))
    latest = archive.latest_runs(1)
    assert set(latest) == {("c1", "t1", 0), ("c1", "t2", 0), ("c1", "t1", 1)}
    assert latest[("c1", "t1", 0)].key.attempt == 1
    assert latest[("c1", "t1", 0)].score == 0.5
    assert archive.latest_runs(99) == {}


def test_runs_for_candidate_by_split(archive: Archive) -> None:
    seed(archive)
    archive.insert_run(run(task="a", split=Split.SELECT))
    archive.insert_run(run(task="b", split=Split.HOLDOUT, score=None, infra_err=True))
    archive.insert_run(run(task="b", attempt=1, split=Split.HOLDOUT, score=0.25))
    assert {r.key.task for r in archive.runs_for_candidate("c1", split=Split.SELECT)} == {"a"}
    hold = archive.runs_for_candidate("c1", split=Split.HOLDOUT)
    assert [r.key.attempt for r in hold] == [1]
    full = archive.runs_for_candidate("c1", split=Split.HOLDOUT, latest_only=False)
    assert [r.key.attempt for r in full] == [0, 1]
    assert archive.runs_for_candidate("c1", batch=2) == []
    assert len(archive.runs_for_candidate("c1")) == 2


def test_pack_hash(archive: Archive) -> None:
    seed(archive)
    assert archive.batch_pack_hash(1) == H
    assert archive.batch_pack_hash(2) is None
    archive.require_pack_hash(1, H)
    with pytest.raises(PackHashMismatch):
        archive.require_pack_hash(1, H2)
    with pytest.raises(UnknownRow):
        archive.require_pack_hash(2, H)


def test_holdout_query_count_counts_batches(archive: Archive) -> None:
    archive.insert_candidate(cand())
    archive.insert_candidate(cand("c2", parent="c1"))
    for b in range(1, 22):
        archive.insert_batch(Batch(b, 1, H, "t"))
    assert archive.holdout_query_count() == 0
    for b in range(1, 21):
        # two candidates, two reps, a retry: still one query per batch
        for c in ("c1", "c2"):
            for rep in (0, 1):
                archive.insert_run(run(batch=b, c=c, rep=rep, split=Split.HOLDOUT))
        archive.insert_run(run(batch=b, split=Split.HOLDOUT, attempt=1))
        if b == 19:
            assert not archive.holdout_warning_due(20)
    assert archive.holdout_query_count() == 20
    assert archive.holdout_warning_due(20)
    archive.insert_run(run(batch=21, split=Split.SELECT))  # non-holdout doesn't count
    assert archive.holdout_query_count() == 20


def test_proxy_call_insert_and_sink(tmp_path: Path) -> None:
    with Archive.open(tmp_path) as a:
        seed(a)
        keyed = ProxyCall(7, RunKey(1, "c1", "t1", 0, 0), Role.TARGET, "m", 1, 2, 3, False, "t")
        free = ProxyCall(0, None, Role.PROPOSER, "m", 1, 2, 3, True, "t")
        assert a.insert_proxy_call(keyed) == 7
        auto = a.insert_proxy_call(free)
        assert auto == 8
        assert a.get_proxy_call(7) == keyed
        assert a.get_proxy_call(auto) == ProxyCall(
            auto, None, Role.PROPOSER, "m", 1, 2, 3, True, "t"
        )
        with pytest.raises(DuplicateRowError):
            a.insert_proxy_call(keyed)
        with pytest.raises(ForeignKeyViolation):
            a.insert_proxy_call(
                ProxyCall(0, RunKey(9, "c1", "t", 0, 0), Role.TARGET, "m", 0, 0, 0, False, "t")
            )

    sink = ArchiveProxyLog(archive_path(tmp_path))
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(5):
                sink(ProxyCall(0, None, Role.ANALYST, "m", 1, 1, 1, False, "t"))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    with Archive.open(tmp_path) as a:
        n = a._con.execute("SELECT COUNT(*) FROM proxy_calls").fetchone()  # pyright: ignore[reportPrivateUsage]
        assert n[0] == 2 + 20
