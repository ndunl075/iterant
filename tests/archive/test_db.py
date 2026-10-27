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
    ArchiveTampered,
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
    open_archive,
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
    assert archive.holdout_query_count() == 20
    assert not archive.holdout_warning_due(20)  # warn *after* 20
    archive.insert_run(run(batch=21, split=Split.SELECT))  # non-holdout doesn't count
    assert archive.holdout_query_count() == 20
    archive.insert_run(run(batch=21, split=Split.HOLDOUT, task="t9"))
    assert archive.holdout_warning_due(20)


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


def test_gitignore_written_once_and_not_overwritten(tmp_path: Path) -> None:
    gi = tmp_path / ".iterant" / ".gitignore"
    with Archive.open(tmp_path):
        assert gi.read_text() == "*\n"
    gi.write_text("custom\n")
    with Archive.open(tmp_path):
        pass
    assert gi.read_text() == "custom\n"


def test_archive_hidden_from_git_and_survives_clean(tmp_path: Path) -> None:
    import subprocess

    def g(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(tmp_path), *a], capture_output=True, text=True, check=True
        ).stdout

    g("init", "-q")
    with Archive.open(tmp_path) as a:
        a.insert_candidate(cand())
    assert g("status", "--porcelain") == ""
    g("add", "-A")
    assert g("ls-files") == ""


def test_busy_is_typed_and_retryable(tmp_path: Path) -> None:
    from iterant.archive import ArchiveBusy

    with Archive.open(tmp_path) as a:
        seed(a)
    blocker = sqlite3.connect(archive_path(tmp_path), isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        with Archive.open(tmp_path, busy_timeout=0.05) as a2:
            with pytest.raises(ArchiveBusy) as info:
                a2.insert_candidate(cand("c2"))
            assert info.value.retryable
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_sink_does_not_create_missing_archive(tmp_path: Path) -> None:
    from iterant.archive import ArchiveNotInitialized

    path = tmp_path / ".iterant" / "archive.db"
    sink = ArchiveProxyLog(path)
    call = ProxyCall(0, None, Role.ANALYST, "m", 1, 1, 1, False, "t")
    with pytest.raises(ArchiveNotInitialized):
        sink(call)
    assert not path.exists()
    path.parent.mkdir()
    sqlite3.connect(path).close()  # empty file, no schema
    with pytest.raises(ArchiveNotInitialized):
        sink(call)


def test_advance_stage_same_stage_noop_on_terminal(archive: Archive) -> None:
    seed(archive)
    archive.advance_stage("c1", Stage.GATE)
    archive.finish_candidate("c1", Status.REJECTED)
    archive.advance_stage("c1", Stage.GATE)  # replay: no-op
    with pytest.raises(InvalidTransition):
        archive.advance_stage("c1", Stage.SMOKE)


def _tamper(path: Path, *statements: str) -> None:
    con = sqlite3.connect(path, isolation_level=None)
    for sql in statements:
        con.execute(sql)
    con.close()


def _trigger_names(path: Path) -> list[str]:
    con = sqlite3.connect(path)
    try:
        return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")]
    finally:
        con.close()


def test_open_refuses_an_archive_with_a_dropped_trigger(tmp_path: Path) -> None:
    with Archive.open(tmp_path) as a:
        a.insert_candidate(cand())
    path = archive_path(tmp_path)
    _tamper(path, "DROP TRIGGER runs_no_update")
    with pytest.raises(ArchiveTampered, match="runs_no_update"):
        Archive.open(tmp_path)  # must not silently recreate the trigger
    assert "runs_no_update" not in _trigger_names(path)
    with pytest.raises(ArchiveTampered):
        Archive.open_path(path, create_schema=False)
    with pytest.raises(ArchiveTampered):
        open_archive(path)


def test_open_refuses_an_altered_trigger(tmp_path: Path) -> None:
    with Archive.open(tmp_path):
        pass
    path = archive_path(tmp_path)
    _tamper(
        path,
        "DROP TRIGGER runs_no_update",
        "CREATE TRIGGER runs_no_update BEFORE UPDATE ON runs WHEN 0"
        " BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END",
    )
    with pytest.raises(ArchiveTampered, match="runs_no_update"):
        Archive.open(tmp_path)


def test_open_refuses_triggers_deleted_via_writable_schema(tmp_path: Path) -> None:
    with Archive.open(tmp_path):
        pass
    path = archive_path(tmp_path)
    _tamper(
        path,
        "PRAGMA writable_schema=1",
        "DELETE FROM sqlite_master WHERE type='trigger'",
        "PRAGMA writable_schema=0",
    )
    with pytest.raises(ArchiveTampered):
        Archive.open(tmp_path)


def test_untampered_and_new_archives_open(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    open_archive(path).close()  # no tables yet: accepted, schema is created next
    with Archive.open(tmp_path / "r"):
        pass
    with Archive.open(tmp_path / "r") as a:
        assert a.schema_version == SCHEMA_VERSION


def _initialised(tmp_path: Path) -> Path:
    with Archive.open(tmp_path):
        pass
    return archive_path(tmp_path)


def test_open_refuses_an_extra_trigger_that_drops_inserts(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    _tamper(
        path,
        "CREATE TRIGGER runs_swallow BEFORE INSERT ON runs BEGIN SELECT RAISE(IGNORE); END",
    )
    with pytest.raises(ArchiveTampered, match="extra: trigger runs_swallow"):
        Archive.open(tmp_path)
    with pytest.raises(ArchiveTampered):
        open_archive(path)


def test_open_refuses_an_edited_table_definition(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    _tamper(
        path,
        "PRAGMA writable_schema=1",
        "UPDATE sqlite_master SET sql = replace(sql, 'CHECK (gen >= 0)', '')"
        " WHERE type = 'table' AND name = 'batches'",
        "PRAGMA writable_schema=0",
    )
    with pytest.raises(ArchiveTampered, match="altered: table batches"):
        Archive.open(tmp_path)


def test_open_refuses_an_initialised_archive_with_every_table_dropped(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    _tamper(
        path,
        "DROP TABLE proxy_calls",
        "DROP TABLE promotions",
        "DROP TABLE runs",
        "DROP TABLE candidates",
        "DROP TABLE batches",
    )
    with pytest.raises(ArchiveTampered, match="missing"):
        Archive.open(tmp_path)  # must not silently re-initialise
    assert _trigger_names(path) == []


def test_report_connect_refuses_a_tampered_archive(tmp_path: Path) -> None:
    from iterant.report import model

    path = _initialised(tmp_path)
    _tamper(path, "DROP TRIGGER runs_no_update")
    with pytest.raises(ArchiveTampered):
        model._connect(path)  # pyright: ignore[reportPrivateUsage]


def test_status_cli_reports_a_tampered_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from iterant.cli.main import main

    path = _initialised(tmp_path)
    _tamper(path, "DROP TRIGGER runs_no_update")
    assert main(["status", str(tmp_path)]) == 1
    assert "archive integrity check failed" in capsys.readouterr().err


def test_unknown_schema_version_with_a_dropped_trigger_is_refused_everywhere(
    tmp_path: Path,
) -> None:
    from iterant.report import model

    path = _initialised(tmp_path)
    _tamper(path, "DROP TRIGGER runs_no_update", "PRAGMA user_version = 99")
    with pytest.raises(ArchiveTampered):
        open_archive(path)
    with pytest.raises(ArchiveTampered):
        model._connect(path)  # pyright: ignore[reportPrivateUsage]
    for create in (False, True):  # the archive layer reports it as a schema version error
        with pytest.raises(SchemaVersionError):
            Archive.open_path(path, create_schema=create)


def test_a_trigger_named_like_an_sqlite_internal_is_refused(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    _tamper(
        path,
        "PRAGMA writable_schema=1",
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) VALUES"
        " ('trigger', 'sqlite_t', 'runs', 0,"
        " 'CREATE TRIGGER sqlite_t BEFORE INSERT ON runs BEGIN SELECT RAISE(IGNORE); END')",
        "PRAGMA writable_schema=0",
    )
    with pytest.raises(ArchiveTampered, match="sqlite_t"):
        Archive.open(tmp_path)


def test_analyze_statistics_tables_are_not_tampering(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    _tamper(path, "ANALYZE")
    with Archive.open(tmp_path):
        pass
