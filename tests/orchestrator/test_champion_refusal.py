"""Champion-side refusal guard: a champion that refuses most tasks makes any candidate win
(refusals score 0), so the batch closes instead of promoting."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.contracts import AcceptanceThresholds, Status
from iterant.orchestrator import InfraRateExceeded
from tests.orchestrator.conftest import THRESHOLDS
from tests.orchestrator.test_errors import champ_default


def _solver(champ_refuses: set[str]):
    def solver(arm: str, task: str, attempt: int) -> str:
        if arm == "champ":
            return "refuse" if task in champ_refuses else champ_default(task)
        return "ok"

    return solver


def test_a_champion_that_refuses_most_tasks_closes_the_batch_and_promotes_nothing(world):
    world.solver = _solver({"se00", "se01", "se02", "se03", "se04", "se05", "se06", "se07"})
    world.queue(world.proposal("v0", "v1", "h"))
    with pytest.raises(InfraRateExceeded, match="refuses too often"):
        world.orchestrator().run_generation()

    db = world.db()
    marker = db.get_candidate("b1-closed")
    cand = db.get_candidate("b1-0")
    assert marker is not None and "refuses too often" in marker.hypothesis
    assert cand is not None and cand.status is not Status.PROMOTED
    assert db.latest_promotion() is None


def test_one_champion_refusal_is_under_the_default_threshold(world):
    world.solver = _solver({"se00"})
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()  # does not raise


def test_the_refusal_threshold_is_configurable(world):
    world.solver = _solver({"se00"})
    world.queue(world.proposal("v0", "v1", "h"))
    t = replace(THRESHOLDS, max_champion_refusal_rate=0.01)
    with pytest.raises(InfraRateExceeded, match="refuses too often"):
        world.orchestrator(thresholds=t).run_generation()


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="max_champion_refusal_rate"):
        AcceptanceThresholds(max_champion_refusal_rate=1.5)
