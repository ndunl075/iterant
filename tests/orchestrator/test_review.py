"""Fixes from the PR #15 review: blind proposer view, spend repair, one writer, SMOKE rule."""

from __future__ import annotations

import dataclasses

import pytest

from iterant.archive import Archive
from iterant.contracts import RunKey, Stage, Status
from iterant.orchestrator import (
    Orchestrator,
    OrchestratorBusy,
    OrchestratorConfig,
    Outcomes,
    WriterLock,
)
from tests.orchestrator.test_errors import solver_with


def test_the_proposer_never_sees_the_pack_or_what_the_image_build_ignores(world):
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    files = world.proposer.calls[0]["files"]
    assert "agent/prompt.txt" in files
    assert not [f for f in files if f.startswith("evals/")]  # the pack, inside the repo
    assert not [f for f in files if f.startswith("notes/")]  # .containerignore


def test_the_analyst_is_told_to_redact_the_hidden_checks(world):
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    redact = world.redactions[0]
    assert redact  # hidden check paths and contents reach the analyst
    assert any("check" in r or "answer" in r or "ok" in r for r in redact)


def test_a_lost_spend_update_is_repaired_from_the_run_rows(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h"))
    real = Archive.add_spend
    calls = 0

    def flaky(self, cid, usd):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise world.Crash("died between insert_run and add_spend")
        return real(self, cid, usd)

    monkeypatch.setattr(Archive, "add_spend", flaky)
    with pytest.raises(world.Crash):
        world.orchestrator().run_generation()
    monkeypatch.setattr(Archive, "add_spend", real)

    res = world.orchestrator().run_generation()

    db = world.db()
    cand = db.get_candidate("b1-0")
    assert res.batch_id == 1 and cand is not None
    runs = db.runs_for_candidate("b1-0", batch=1, latest_only=False)
    assert cand.usd_micros == sum(r.usd_micros for r in runs)


def test_two_orchestrators_cannot_share_an_archive(world):
    world.queue(world.proposal("v0", "v1", "h"))
    with WriterLock(world.repo):
        with pytest.raises(OrchestratorBusy):
            world.orchestrator().run_generation()
    assert world.orchestrator().run_generation().promoted == "b1-0"  # free again


def test_smoke_kills_a_crash_or_timeout_the_champion_did_not_have(world):
    world.solver = solver_with(lambda a, t, n: "ok-crash" if a == "b1-0" and t == "tr0" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    [c] = res.candidates
    assert (c.stage, c.status) == (Stage.SMOKE, Status.REJECTED)
    assert c.notes and c.notes[0].startswith("smoke:")


def test_smoke_fails_closed_when_the_outcome_store_is_lost(world):
    world.queue(world.proposal("v0", "v1", "h"))
    (world.repo / ".iterant").mkdir(exist_ok=True)
    real = Outcomes.recorded_by_cand
    Outcomes.recorded_by_cand = lambda self, b: {}  # type: ignore[method-assign]
    try:
        res = world.orchestrator().run_generation()
    finally:
        Outcomes.recorded_by_cand = real  # type: ignore[method-assign]
    [c] = res.candidates
    assert (c.stage, c.status) == (Stage.SMOKE, Status.REJECTED)


def test_a_pack_at_the_repo_root_is_refused(world):
    with pytest.raises(ValueError, match="repo root"):
        Orchestrator(
            **{**world.orch_kwargs(), "pack_root": world.repo},
            config=OrchestratorConfig(thresholds=world.THRESHOLDS),
        )


def test_the_sandbox_ttl_outlasts_a_stalled_get(world):
    from iterant.orchestrator.runs import sandbox_ttl

    cfg = OrchestratorConfig(run_timeout_s=10, read_budget_s=50)
    assert sandbox_ttl(cfg) >= 10 + 30 + 50 + 120 + 10  # + backend IO timeout + probe


def test_the_analyst_is_not_called_when_its_worst_case_exceeds_the_budget(world):
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator(budget_usd_micros=world.PROPOSER_CAP + 5_000).run_generation()
    assert res.stopped == "budget" and not world.redactions


def test_smoke_forgives_a_fault_the_champion_shares(world):
    world.solver = solver_with(lambda a, t, n: "ok-crash" if t == "tr0" else "")
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    [c] = res.candidates
    assert c.stage is not Stage.SMOKE


def test_the_proposer_is_not_called_when_its_worst_case_exceeds_the_budget(world):
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator(budget_usd_micros=world.PROPOSER_CAP - 1).run_generation()
    assert res.stopped == "budget" and not world.proposer.calls


def test_a_restart_preloads_the_proxy_ledger_from_the_archive(world):
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator(budget_usd_micros=170_000).run_generation()  # stops part way
    first = world.exec_log[0]
    orch = world.orchestrator()  # a new process: the proxy knows nothing yet
    assert world.proxy.run_usage(first).usd_micros == 0
    orch.run_generation()
    assert world.proxy.run_usage(first).usd_micros == world.RUN_COST


def test_a_holdout_warning_is_surfaced_after_too_many_queries(world):
    world.queue(world.proposal("v0", "v1", "h1"), world.proposal("v1", "v2", "h2"))
    t = dataclasses.replace(world.THRESHOLDS, holdout_query_warn=1)  # one query per batch
    first = world.orchestrator(thresholds=t).run_generation()
    second = world.orchestrator(thresholds=t).run_generation()
    assert not first.warnings
    assert any("holdout" in w for w in second.warnings)


def test_run_keys_are_still_what_resume_matches_on(world):
    assert RunKey(1, "c", "t", 0) != RunKey(1, "c", "t", 0, 1)
