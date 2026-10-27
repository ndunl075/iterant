"""Budget stops and crash recovery: a restart neither duplicates nor skips runs."""

from __future__ import annotations

import pytest

from iterant.archive import Archive
from iterant.archive import PackHashMismatch as ArchivePackHashMismatch
from iterant.contracts import RunKey, Stage, Status
from iterant.orchestrator import BatchAborted
from iterant.packs.hashing import PackHashMismatch

TOTAL_RUNS = 36  # (smoke 2 + select 10 + holdout 6) x (candidate + champion)


def keys(world) -> set[RunKey]:
    db = world.db()
    return {
        r.key for cid in ("seed", "b1-0") for r in db.runs_for_candidate(cid, latest_only=False)
    }


def test_budget_exhaustion_stops_without_a_verdict_and_resume_finishes(world):
    world.queue(world.proposal("v0", "v1", "h-good"))
    res = world.orchestrator(budget_usd_micros=150_000).run_generation()

    assert res.stopped == "budget" and res.promoted is None
    assert res.spent_usd_micros <= 150_000 + world.RUN_COST  # stops before a run could overshoot
    assert res.spent_usd_micros == len(world.exec_log) * world.RUN_COST
    [cand] = res.candidates
    assert cand.status is Status.ACTIVE  # not rejected: money ran out, not the candidate
    first_runs = len(world.exec_log)
    assert 0 < first_runs < TOTAL_RUNS

    again = world.orchestrator().run_generation()  # more budget (none set)
    assert again.stopped == "done" and again.promoted == "b1-0"
    assert again.batch_id == 1  # resumed, not a new batch
    assert len(world.exec_log) == TOTAL_RUNS  # nothing run twice, nothing skipped
    assert len(set(world.exec_log)) == TOTAL_RUNS
    assert again.spent_usd_micros == TOTAL_RUNS * world.RUN_COST  # spend carried over


def test_a_budget_too_small_for_one_run_runs_nothing(world):
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator(budget_usd_micros=world.RUN_CAP - 1).run_generation()
    assert res.stopped == "budget" and not world.exec_log


@pytest.mark.parametrize("crash_at", [1, 4, 5, 24, 25, 36])
def test_crash_at_any_run_resumes_to_exactly_the_clean_result(world, crash_at):
    world.queue(world.proposal("v0", "v1", "h-good"))
    world.crash_at = crash_at
    with pytest.raises(world.Crash):
        world.orchestrator().run_generation()
    before = keys(world)
    assert len(before) == crash_at - 1  # the run that was in flight left no row
    n_before = len(world.exec_log)

    world.crash_at = None
    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0" and res.batch_id == 1
    after = keys(world)
    assert len(after) == TOTAL_RUNS
    redone = world.exec_log[n_before:]
    assert not set(redone) & before  # finished slots were skipped
    assert len(set(redone)) == len(redone) == TOTAL_RUNS - len(before)
    assert world.exec_log[n_before - 1] == redone[0]  # only the interrupted run repeats
    assert len(world.proposer.calls) == 1  # no second proposal for the same slot
    assert world.git("tag", "-l", "iterant/gen-1") == "iterant/gen-1"


def test_crash_between_the_git_promotion_and_the_db_write_is_repaired(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h-good"))
    real = Archive.record_promotion

    def die(self, promo):  # git is done, the DB write never happens
        raise world.Crash("died")

    monkeypatch.setattr(Archive, "record_promotion", die)
    with pytest.raises(world.Crash):
        world.orchestrator().run_generation()
    monkeypatch.setattr(Archive, "record_promotion", real)
    assert world.db().latest_promotion() is None
    assert world.git("tag", "-l", "iterant/gen-1") == "iterant/gen-1"
    runs_before = len(world.exec_log)

    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0" and res.gen == 1 and res.batch_id == 1
    assert len(world.exec_log) == runs_before  # no run repeated
    db = world.db()
    promo = db.get_promotion(1)
    cand = db.get_candidate("b1-0")
    assert promo is not None and cand is not None and cand.status is Status.PROMOTED
    assert sorted(world.git("tag", "-l", "iterant/gen-*").split()) == [
        "iterant/gen-0",
        "iterant/gen-1",
    ]


def test_crash_after_promoting_rejects_the_rest_of_the_batch(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h-a"), world.proposal("v0", "v1b", "h-b"))
    real = Archive.finish_candidate
    calls = []

    def die_after_winner(self, cand_id, status, *, stage=None):
        calls.append(cand_id)
        raise world.Crash("died")

    monkeypatch.setattr(Archive, "finish_candidate", die_after_winner)
    with pytest.raises(world.Crash):
        world.orchestrator(candidates_per_gen=2).run_generation()
    monkeypatch.setattr(Archive, "finish_candidate", real)
    db = world.db()
    assert db.latest_promotion() is not None  # the winner is recorded; the loser is still ACTIVE
    loser = next(c for c in db.iter_candidates() if c.id == calls[0])
    assert loser.status is Status.ACTIVE

    world.queue(world.proposal("v1b", "v2", "h-next"))
    res = world.orchestrator(candidates_per_gen=1).run_generation()

    loser = next(c for c in world.db().iter_candidates() if c.id == calls[0])
    assert (loser.stage, loser.status) == (Stage.PROMOTE, Status.REJECTED)
    assert res.gen == 2 and res.batch_id == 2  # a fresh batch, the old one is closed


def test_a_batch_pinned_to_another_pack_is_never_resumed(world):
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator(budget_usd_micros=150_000).run_generation()  # leaves batch 1 open
    world.write_pack(input_suffix="-edited")
    world.reload_pack()
    with pytest.raises(ArchivePackHashMismatch):
        world.orchestrator().run_generation()


def test_a_pack_that_changed_on_disk_aborts_before_anything_runs(world):
    world.queue(world.proposal("v0", "v1", "h"))
    (world.pack_root / "tasks" / "se00" / "check").write_bytes(b'{"v": 0}')
    with pytest.raises(PackHashMismatch):
        world.orchestrator().run_generation()
    assert not world.exec_log and world.db().get_batch(1) is None


def test_grader_preflight_failure_aborts_the_batch_as_infra(world):
    world.queue(world.proposal("v0", "v1", "h"))
    world.preflight_fails = True
    with pytest.raises(BatchAborted, match="preflight"):
        world.orchestrator().run_generation()
    assert not world.exec_log
    assert world.db().get_batch(1) is None  # no batch was opened
    assert not world.proposer.calls  # and no proposer money was spent
