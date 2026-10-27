"""init, then estimate, then a run of one generation, then status: end to end with fakes."""

from __future__ import annotations

import pytest

from iterant.archive import Archive
from iterant.contracts import Proposal, Stage, Status


def test_init_estimate_run_status(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()

    # estimate: baseline of the champion on the train split, nothing written to the real archive
    assert world.cli("estimate", str(world.repo), *world.SMALL, "--reps", "2") == 0
    out = capsys.readouterr().out
    assert "4 train tasks x 2 reps = 8 runs" in out
    assert "pass rate               50.0%" in out
    assert "smallest detectable gain" in out and "cost per candidate" in out
    assert "WARNING: only 4 train tasks were sampled" in out
    assert (world.repo / ".iterant" / "calibration.db").is_file()
    assert not (world.repo / ".iterant" / "archive.db").exists()
    assert world.exec_log and set(world.exec_log) == {"tr0", "tr1", "tr2", "tr3"}

    # calibration required before a real run (§4): a strong estimate and aa-test
    world.calibrate()
    capsys.readouterr()

    # run: one generation, the candidate wins and is promoted
    world.queue.append(Proposal(world.diff("v0", "v1"), "h-good"))
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    out = capsys.readouterr().out
    assert "generation 1 (batch 1): promoted b1-0" in out
    assert world.git("show", "iterant/champion:agent/prompt.txt") == "v1"
    with Archive.open(world.repo) as db:
        promo = db.get_promotion(1)
        cand = db.get_candidate("b1-0")
        assert promo is not None and promo.cand == "b1-0"
        assert cand is not None and (cand.stage, cand.status) == (Stage.PROMOTE, Status.PROMOTED)
        # the digest sample ran the champion on the first train tasks in the generation's batch,
        # and SMOKE reused those slots: each (champion, task) has one row
        slots = db.latest_runs(1)
        for task in ("tr0", "tr1", "tr2"):
            assert ("seed", task, 0) in slots
    assert world.proposer.digests == [world.proposer.digests[0]]
    assert "Train runs: 3" in world.proposer.digests[0]  # the digest saw the sampled train runs

    # status reads it back
    assert world.cli("status", str(world.repo)) == 0
    out = capsys.readouterr().out
    assert "champion: b1-0 (generation 1)" in out
    assert "gen 1: b1-0" in out
    assert "batches: 1; candidates: 1 (1 promoted)" in out
    assert "b1-0: promoted at promote" in out


def test_run_two_generations_stops_at_n(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    world.queue += [Proposal(world.diff("v0", "v1"), "h1"), Proposal(world.diff("v1", "v2"), "h2")]
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_SMALL, "--generations", "2") == 0
    out = capsys.readouterr().out
    assert "generation 1 (batch 1): promoted b1-0" in out
    assert "generation 2 (batch 2)" in out
    assert world.git("show", "iterant/champion:agent/prompt.txt") in ("v1", "v2")


def test_run_without_a_promotion_still_exits_zero(
    world, capsys: pytest.CaptureFixture[str]
) -> None:
    world.init()
    world.queue.append(None)  # the proposer found nothing to change
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 0
    out = capsys.readouterr().out
    assert "no candidate promoted" in out and "b1-0: rejected at propose" in out


def test_estimate_warns_when_pack_is_too_easy(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    world.solver = lambda arm, task, attempt: "ok"
    capsys.readouterr()
    assert world.cli("estimate", str(world.repo), *world.SMALL) == 0
    out = capsys.readouterr().out
    assert "WARNING: pack is too easy" in out


def test_estimate_warns_when_pack_is_too_hard_and_too_small(
    world, capsys: pytest.CaptureFixture[str]
) -> None:
    world.init()
    world.solver = lambda arm, task, attempt: "wrong"
    capsys.readouterr()
    # the defaults want 80 select and 40 holdout tasks; the fake pack has 10 and 6
    assert world.cli("estimate", str(world.repo), "--smoke-tasks", "2") == 0
    out = capsys.readouterr().out
    assert "WARNING: pack is too hard" in out
    assert "WARNING: pack is too small: 10 select tasks" in out
    assert "WARNING: pack is too small: 6 holdout tasks" in out


def test_aa_test_passes_on_a_stable_champion(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()
    assert (
        world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "20", "--resamples", "200")
        == 0
    )
    out = capsys.readouterr().out
    assert "false accepts        0 of 20" in out
    assert "A/A test passed." in out
    # the pool: select tasks only, 2 reps per arm slot; holdout is never queried
    assert len(world.exec_log) == 10 * 2
    assert all(t.startswith("se") for t in world.exec_log)
