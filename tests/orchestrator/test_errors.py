"""§6 error policy: infra errors retry once then drop the pair; candidate faults score 0."""

from __future__ import annotations

import pytest

from iterant.contracts import RunKey, Split, Stage, Status
from iterant.orchestrator import BatchAborted


def champ_default(task: str) -> str:
    return "ok" if task in {"tr0", "tr1", "se00", "se01", "se02", "ho0", "ho1"} else "wrong"


def solver_with(special):
    def solver(arm: str, task: str, attempt: int) -> str:
        hit = special(arm, task, attempt)
        if hit:
            return hit
        return champ_default(task) if arm == "champ" else "ok"

    return solver


def test_infra_error_is_retried_once_and_the_retry_counts(world):
    world.solver = solver_with(lambda a, t, n: "infra" if (a, t, n) == ("b1-0", "se00", 0) else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0"
    db = world.db()
    rows = [
        r
        for r in db.runs_for_candidate("b1-0", split=Split.SELECT, latest_only=False)
        if r.key.task == "se00"
    ]
    got = [(r.key.attempt, r.score, r.infra_err) for r in rows]
    assert got == [(0, None, True), (1, 1.0, False)]
    assert db.latest_runs(1)[("b1-0", "se00", 0)].key.attempt == 1


def test_a_second_infra_error_drops_the_pair_from_both_arms(world):
    # Both arms hit the infra error on se09 (twice), so the pair is dropped and the rest decides.
    world.solver = solver_with(lambda a, t, n: "infra" if t == "se09" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0"
    db = world.db()
    for arm in ("b1-0", "seed"):
        finals = [r for r in db.runs_for_candidate(arm, split=Split.SELECT) if r.key.task == "se09"]
        assert [(r.key.attempt, r.score, r.infra_err) for r in finals] == [(1, None, True)]
        every = db.runs_for_candidate(arm, split=Split.SELECT, latest_only=False)
        assert sum(1 for r in every if r.key.task == "se09") == 2  # tried once, retried once


def test_an_infra_error_only_the_candidate_hit_does_not_pass_the_infra_guard(world):
    world.solver = solver_with(lambda a, t, n: "infra" if (a, t) == ("b1-0", "se09") else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    [cand] = res.candidates
    assert (cand.stage, cand.status) == (Stage.SELECT, Status.REJECTED)
    assert any("infra" in n for n in cand.notes)


def test_a_refusal_is_retried_once_and_tracked_apart_from_infra(world):
    world.solver = solver_with(lambda a, t, n: "refuse" if (a, t, n) == ("b1-0", "se01", 0) else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0"
    rows = world.db().runs_for_candidate("b1-0", split=Split.SELECT, latest_only=False)
    first = next(r for r in rows if r.key.task == "se01" and r.key.attempt == 0)
    assert (first.refusal, first.infra_err, first.score) == (True, False, None)
    assert first.usd_micros == world.RUN_COST  # a refused first try still cost money


def test_too_many_infra_errors_abort_the_batch_and_leave_candidates_resumable(world):
    world.solver = solver_with(lambda a, t, n: "infra" if a == "champ" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    with pytest.raises(BatchAborted, match="infra"):
        world.orchestrator(max_infra_rate=0.4).run_generation()
    [cand] = [c for c in world.db().iter_candidates() if c.parent is not None]
    assert (cand.stage, cand.status) == (Stage.SMOKE, Status.ACTIVE)

    world.solver = solver_with(lambda a, t, n: "")  # the outage is over
    assert world.orchestrator().run_generation().promoted == "b1-0"


def task(world, tid: str = "se00"):
    return next(t for t in world.pack.tasks if t.id == tid)


@pytest.mark.parametrize("outcome", ["lost", "lost-raise", "create-lost", "timeout", "truncated"])
def test_candidate_caused_failures_score_zero_and_are_not_infra(world, outcome):
    world.solver = lambda arm, t, n: outcome
    key = RunKey(1, "c", "se00", 0)
    r = world.executor().execute(key, task(world), "img:c")
    assert (r.score, r.infra_err, r.refusal) == (0.0, False, False)
    assert all(sb.destroyed >= 1 for sb in world.backend.sandboxes)  # even a lost one


def test_a_lost_sandbox_is_not_graded(world):
    world.solver = lambda arm, t, n: "lost"
    world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert [sb.net for sb in world.backend.sandboxes] == ["proxy-only"]  # no grader sandbox


def test_runtime_failures_are_infra_not_zero(world):
    world.solver = lambda arm, t, n: "infra"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert (r.score, r.infra_err, r.refusal) == (None, True, False)


@pytest.mark.parametrize("flag", ["create_fails", "put_fails", "get_fails"])
def test_sandbox_failures_around_the_run_are_infra(world, flag):
    world.solver = lambda arm, t, n: "ok"
    setattr(world, flag, True)
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert (r.score, r.infra_err) == (None, True)


def test_a_refusal_wins_over_the_output_and_skips_grading(world):
    world.solver = lambda arm, t, n: "refuse"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert (r.score, r.refusal, r.infra_err) == (None, True, False)
    assert [sb.net for sb in world.backend.sandboxes] == ["proxy-only"]


def test_a_wrong_answer_scores_zero_and_a_right_one_one(world):
    world.solver = lambda arm, t, n: "wrong"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert r.score == 0.0 and not r.infra_err
    world.solver = lambda arm, t, n: "ok"
    r = world.executor().execute(RunKey(1, "c", "se00", 0, 1), task(world), "img:c")
    assert r.score == 1.0 and r.usd_micros == world.RUN_COST
