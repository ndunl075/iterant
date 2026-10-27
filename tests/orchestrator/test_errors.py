"""§6 error policy: infra errors retry once then drop the pair; candidate faults score 0."""

from __future__ import annotations

import pytest

from iterant.contracts import RunKey, Split, Stage, Status
from iterant.orchestrator import InfraRateExceeded, OrchestratorConfig


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
    # five of ten tasks (0 of 10 for the champion): clear on first-try rates (p = 0.031)
    hit = {f"se0{i}" for i in range(5, 10)}
    world.solver = solver_with(lambda a, t, n: "infra" if a == "b1-0" and t in hit else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    [cand] = res.candidates
    assert (cand.stage, cand.status) == (Stage.SELECT, Status.REJECTED)
    assert any("first-try infra-error rate" in n for n in cand.notes)


def test_a_single_infra_error_on_one_candidate_task_is_noise_not_a_veto(world):
    # the noise-tolerant guard (W3): 1 event is not significant, and 1 of 10 dropped is allowed
    world.solver = solver_with(lambda a, t, n: "infra" if (a, t) == ("b1-0", "se09") else "")
    world.queue(world.proposal("v0", "v1", "h"))
    assert world.orchestrator().run_generation().promoted == "b1-0"


def test_a_refusal_is_retried_once_and_tracked_apart_from_infra(world):
    world.solver = solver_with(lambda a, t, n: "refuse" if (a, t, n) == ("b1-0", "se01", 0) else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()

    assert res.promoted == "b1-0"
    rows = world.db().runs_for_candidate("b1-0", split=Split.SELECT, latest_only=False)
    first = next(r for r in rows if r.key.task == "se01" and r.key.attempt == 0)
    assert (first.refusal, first.infra_err, first.score) == (True, False, None)
    assert first.usd_micros == world.RUN_COST  # a refused first try still cost money


def test_infra_abort_closes_the_batch_and_the_next_call_opens_a_fresh_one(world):
    world.solver = solver_with(lambda a, t, n: "infra" if a == "champ" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    with pytest.raises(InfraRateExceeded, match="infra"):
        world.orchestrator(max_infra_rate=0.4).run_generation()

    db = world.db()
    cand = db.get_candidate("b1-0")
    marker = db.get_candidate("b1-closed")
    assert cand is not None and cand.status is Status.REJECTED
    assert marker is not None and marker.hypothesis.startswith("infra-abort")

    world.solver = solver_with(lambda a, t, n: "")  # the outage is over
    world.queue(world.proposal("v0", "v1", "h2"))
    res = world.orchestrator(max_infra_rate=0.4).run_generation()  # the SAME config
    assert (res.batch_id, res.gen, res.promoted) == (2, 1, "b2-0")


def test_a_persistent_outage_aborts_each_new_batch_too_not_wedges_one(world):
    world.solver = solver_with(lambda a, t, n: "infra" if a == "champ" else "")
    world.queue(world.proposal("v0", "v1", "h"), world.proposal("v0", "v1", "h2"))
    for batch in (1, 2):
        with pytest.raises(InfraRateExceeded):
            world.orchestrator(max_infra_rate=0.4).run_generation()
        assert world.db().get_batch(batch) is not None


def test_refusals_never_trip_the_infra_abort(world):
    world.solver = solver_with(lambda a, t, n: "refuse" if a == "champ" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator(max_infra_rate=0.4).run_generation()  # does not raise
    # the champion's refusals score 0.0 (symmetric with the candidate's), so the pairs are kept
    assert res.stopped == "done"


def test_a_candidates_own_infra_errors_do_not_abort_the_batch(world):
    world.solver = solver_with(lambda a, t, n: "infra" if a == "b1-0" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator(max_infra_rate=0.4).run_generation()
    assert res.promoted is None


def task(world, tid: str = "se00"):
    return next(t for t in world.pack.tasks if t.id == tid)


@pytest.mark.parametrize("outcome", ["lost", "lost-raise", "create-lost", "timeout", "truncated"])
def test_candidate_caused_failures_score_zero_and_are_not_infra(world, outcome):
    world.solver = lambda arm, t, n: outcome
    key = RunKey(1, "c", "se00", 0)
    r = world.executor().execute(key, task(world), "img:c").result
    assert (r.score, r.infra_err, r.refusal) == (0.0, False, False)
    assert all(sb.destroyed >= 1 for sb in world.backend.sandboxes)  # even a lost one


def test_a_lost_sandbox_is_not_graded(world):
    world.solver = lambda arm, t, n: "lost"
    world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    assert [sb.net for sb in world.backend.sandboxes] == ["proxy-only"]  # no grader sandbox


def test_runtime_failures_are_infra_not_zero(world):
    world.solver = lambda arm, t, n: "infra"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err, r.refusal) == (None, True, False)


@pytest.mark.parametrize("flag", ["create_fails", "put_fails"])
def test_sandbox_failures_before_the_run_are_infra(world, flag):
    world.solver = lambda arm, t, n: "ok"
    setattr(world, flag, True)
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (None, True)


def test_a_get_failure_the_agent_can_cause_scores_zero_when_the_sandbox_still_answers(world):
    world.solver = lambda arm, t, n: "ok"
    world.get_fails = True  # chmod 000 on an out file, a FIFO, a path too deep ...
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (0.0, False)


def test_a_wrecked_agent_sandbox_is_judged_by_a_fresh_one_not_by_itself(world):
    world.solver = lambda arm, t, n: "ok"
    world.get_fails = True
    world.agent_sandbox_broken = True  # the in-sandbox probe fails; a fresh sandbox is fine
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (0.0, False)
    assert len(world.backend.sandboxes) == 2  # the agent's and the fresh probe (no grading)


def test_a_get_failure_is_infra_only_if_the_runtime_is_down(world):
    world.solver = lambda arm, t, n: "ok"
    world.get_fails = True
    world.runtime_down = True
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (None, True)


def test_the_agent_sandbox_outlives_the_run_the_listing_and_the_read_budget(world):
    from iterant.orchestrator.runs import sandbox_ttl

    world.solver = lambda arm, t, n: "ok"
    ex = world.executor(run_timeout_s=50, read_budget_s=200)
    ex.execute(RunKey(1, "c", "se00", 0), task(world), "img:c")
    want = sandbox_ttl(OrchestratorConfig(run_timeout_s=50, read_budget_s=200))
    assert world.backend.ttls[0] == want >= 50 + 200


def test_a_key_rerun_in_one_process_is_not_double_counted(world):
    world.solver = lambda arm, t, n: "ok"
    ex = world.executor()
    key = RunKey(1, "c", "se00", 0)
    first = ex.execute(key, task(world), "img:c").result
    again = ex.execute(key, task(world), "img:c").result
    assert first.usd_micros == again.usd_micros == world.RUN_COST


def test_the_run_token_is_revoked_when_the_sandbox_cannot_be_created(world, monkeypatch):
    revoked: list[RunKey] = []
    real = world.proxy.sessions.revoke_run
    monkeypatch.setattr(world.proxy.sessions, "revoke_run", lambda k: (revoked.append(k), real(k)))
    world.create_fails = True
    key = RunKey(1, "c", "se00", 0)
    world.executor().execute(key, task(world), "img:c")
    assert key in revoked


def test_a_refusal_wins_over_the_output_and_skips_grading(world):
    world.solver = lambda arm, t, n: "refuse"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.refusal, r.infra_err) == (None, True, False)
    assert [sb.net for sb in world.backend.sandboxes] == ["proxy-only"]


def test_a_wrong_answer_scores_zero_and_a_right_one_one(world):
    world.solver = lambda arm, t, n: "wrong"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert r.score == 0.0 and not r.infra_err
    world.solver = lambda arm, t, n: "ok"
    r = world.executor().execute(RunKey(1, "c", "se00", 0, 1), task(world), "img:c").result
    assert r.score == 1.0 and r.usd_micros == world.RUN_COST


def test_a_non_sandbox_exception_after_the_agent_ran_is_infra_and_never_probed(world, monkeypatch):
    from iterant.orchestrator import runs

    def boom(*a, **k):
        raise TypeError("orchestrator bug")

    monkeypatch.setattr(runs, "read_out", boom)
    world.solver = lambda arm, t, n: "ok"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (None, True)
    assert len(world.backend.sandboxes) == 1  # no fresh-sandbox probe


def test_an_expired_agent_sandbox_is_infra_before_any_probe(world):
    from iterant.orchestrator.runs import sandbox_ttl

    world.solver = lambda arm, t, n: "ok"
    world.get_fails = True  # would score 0.0 if the sandbox were alive and the runtime healthy
    ex = world.executor()
    ttl = sandbox_ttl(OrchestratorConfig())
    ticks = iter([0.0, ttl + 1.0, ttl + 2.0, ttl + 3.0])
    ex._clock = lambda: next(ticks)  # pyright: ignore[reportPrivateUsage]
    r = ex.execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (None, True)
    assert len(world.backend.sandboxes) == 1
