"""Whole generations against the fake world: promote and reject paths, shared champion arm."""

from __future__ import annotations

from iterant.contracts import Proposal, Split, Stage, Status
from iterant.orchestrator import SEED_ID

SELECT_RUNS = 10
HOLDOUT_RUNS = 6
SMOKE_RUNS = 2
PER_ARM = SMOKE_RUNS + SELECT_RUNS + HOLDOUT_RUNS


def test_full_generation_promotes_a_clear_win(world):
    world.queue(world.proposal("v0", "v1", "h-good"))
    res = world.orchestrator().run_generation()

    assert (res.gen, res.batch_id, res.promoted, res.stopped) == (1, 1, "b1-0", "done")
    db = world.db()
    promo = db.get_promotion(1)
    assert promo is not None and promo.cand == "b1-0"
    assert promo.select_delta > 0.05 and promo.select_p < 0.05
    assert promo.holdout_delta > 0 and promo.holdout_p < 0.2
    cand = db.get_candidate("b1-0")
    assert cand is not None
    assert (cand.stage, cand.status, cand.parent) == (Stage.PROMOTE, Status.PROMOTED, SEED_ID)
    assert cand.usd_micros == PER_ARM * world.RUN_COST
    # git: the champion branch carries the diff and the gen tag points at it
    assert world.git("show", "iterant/champion:agent/prompt.txt") == "v1"
    assert world.git("rev-parse", "iterant/gen-1^{commit}") == world.git(
        "rev-parse", "iterant/champion"
    )
    # the whole generation: both arms' runs plus the proposer's call, from the proxy log
    assert res.spent_usd_micros == 2 * PER_ARM * world.RUN_COST + world.PROPOSER_COST
    assert world.proposer.calls[0]["parent"] == SEED_ID
    assert world.proposer.calls[0]["prompt"] == "v0\n"
    assert world.proposer.calls[0]["digest"] == "digest"
    assert world.proposer.calls[0]["history"] == []
    assert world.gate.seen == ["h-good"]
    # every sandbox was torn down; agent ones had no egress but the proxy
    assert all(sb.destroyed >= 1 for sb in world.backend.sandboxes)
    nets = {sb.image.startswith("img:"): sb.net for sb in world.backend.sandboxes}
    assert nets == {True: "proxy-only", False: "default-deny"}


def test_runs_come_from_the_proxy_ledger_not_the_agent(world):
    world.queue(world.proposal("v0", "v1", "h-good"))
    world.orchestrator().run_generation()
    runs = world.db().runs_for_candidate("b1-0", batch=1)
    assert runs and all(r.usd_micros == world.RUN_COST and r.in_tok == 2000 for r in runs)
    assert all(r.out_tok == 500 and not r.infra_err and not r.refusal for r in runs)


def test_second_generation_builds_on_the_new_champion(world):
    world.queue(world.proposal("v0", "v1", "h1"), world.proposal("v1", "v2", "h2"))
    first = world.orchestrator().run_generation()
    second = world.orchestrator().run_generation()

    assert (first.promoted, second.promoted, second.gen, second.batch_id) == ("b1-0", "b2-0", 2, 2)
    assert world.proposer.calls[1]["parent"] == "b1-0"
    assert world.proposer.calls[1]["prompt"] == "v1\n"
    [tried] = world.proposer.calls[1]["history"]
    assert tried.candidate.id == "b1-0" and tried.diff.startswith("diff --git")
    assert world.git("show", "iterant/champion:agent/prompt.txt") == "v2"
    assert sorted(world.git("tag", "-l", "iterant/gen-*").split()) == [
        "iterant/gen-0",
        "iterant/gen-1",
        "iterant/gen-2",
    ]


def test_candidates_share_one_champion_arm_and_only_one_is_promoted(world):
    world.solver = lambda arm, task, attempt: (
        "wrong" if arm == "b1-0" and task == "ho5" else ("ok" if arm != "champ" else _champ(task))
    )
    world.queue(world.proposal("v0", "v1", "h-a"), world.proposal("v0", "v1b", "h-b"))
    res = world.orchestrator(candidates_per_gen=2).run_generation()

    db = world.db()
    assert res.promoted == "b1-1"
    loser = db.get_candidate("b1-0")
    assert loser is not None and (loser.stage, loser.status) == (Stage.PROMOTE, Status.REJECTED)
    # the champion ran each slot once for both candidates, not once per candidate
    assert len(db.runs_for_candidate(SEED_ID, split=Split.SELECT, batch=1)) == SELECT_RUNS
    assert len(db.runs_for_candidate(SEED_ID, split=Split.HOLDOUT, batch=1)) == HOLDOUT_RUNS
    assert len(db.runs_for_candidate("b1-0", split=Split.SELECT, batch=1)) == SELECT_RUNS
    assert len(db.runs_for_candidate("b1-1", split=Split.SELECT, batch=1)) == SELECT_RUNS


def _champ(task: str) -> str:
    return "ok" if task in {"tr0", "tr1", "se00", "se01", "se02", "ho0", "ho1"} else "wrong"


def test_every_kill_rule_rejects_and_the_champion_stays(world):
    def solver(arm: str, task: str, attempt: int) -> str:
        if arm == "b1-3":
            return "crash"
        if arm == "b1-4":
            return _champ(task)
        if arm == "b1-5":
            return "wrong" if task.startswith("ho") else "ok"
        return _champ(task) if arm == "champ" else "ok"

    big = Proposal("\n".join(f"+line {i}" for i in range(401)), "h-big")
    world.solver = solver
    world.gate.fail = {"h-gate"}
    world.queue(
        None,
        big,
        world.proposal("v0", "v1", "h-gate"),
        world.proposal("v0", "v2", "h-smoke"),
        world.proposal("v0", "v3", "h-null"),
        world.proposal("v0", "v4", "h-overfit"),
        RuntimeError("proposer exploded"),
    )
    res = world.orchestrator(candidates_per_gen=7).run_generation()

    where = {c.id: (c.stage, c.status) for c in res.candidates}
    assert where == {
        "b1-0": (Stage.PROPOSE, Status.REJECTED),  # no diff
        "b1-1": (Stage.PROPOSE, Status.REJECTED),  # over 400 lines
        "b1-2": (Stage.GATE, Status.REJECTED),
        "b1-3": (Stage.SMOKE, Status.REJECTED),
        "b1-4": (Stage.SELECT, Status.REJECTED),
        "b1-5": (Stage.CONFIRM, Status.REJECTED),
        "b1-6": (Stage.PROPOSE, Status.ERROR),
    }
    notes = {c.id: c.notes for c in res.candidates}
    assert notes["b1-2"] == ("task id found in diff",)
    assert notes["b1-4"] and notes["b1-4"][0].startswith("select:")
    assert res.promoted is None
    db = world.db()
    assert db.latest_promotion() is None
    assert world.git("rev-parse", "iterant/champion") == world.git("rev-parse", "main")
    # the rejects are all in the archive, big diff included
    big_row = db.get_candidate("b1-1")
    assert big_row is not None and big_row.hypothesis == "h-big"
    # killed candidates never ran later stages
    assert not db.runs_for_candidate("b1-3", split=Split.SELECT)
    assert not db.runs_for_candidate("b1-4", split=Split.HOLDOUT)
    assert not db.runs_for_candidate("b1-2")


def test_a_gate_that_raises_is_an_error_not_a_verdict(world):
    world.gate.boom = {"h-x"}
    world.queue(world.proposal("v0", "v1", "h-x"))
    res = world.orchestrator().run_generation()
    assert [(c.stage, c.status) for c in res.candidates] == [(Stage.GATE, Status.ERROR)]


def test_image_build_failure_errors_that_candidate_only(world):
    world.images.fail = {"b1-0"}
    world.queue(world.proposal("v0", "v1", "h-a"), world.proposal("v0", "v1b", "h-b"))
    res = world.orchestrator(candidates_per_gen=2).run_generation()
    where = {c.id: c.status for c in res.candidates}
    assert where == {"b1-0": Status.ERROR, "b1-1": Status.PROMOTED}
    assert res.promoted == "b1-1"
