"""The champion refusal guard counts SELECT and CONFIRM only (not SMOKE's train tasks)."""

from __future__ import annotations

import pytest

from iterant.orchestrator import InfraRateExceeded
from tests.orchestrator.test_errors import champ_default


def test_the_close_message_names_the_champion_and_what_to_do(world):
    refuses = {f"se0{i}" for i in range(8)}
    world.solver = lambda arm, t, n: (
        "refuse"
        if arm == "champ" and t in refuses
        else (champ_default(t) if arm == "champ" else "ok")
    )
    world.queue(world.proposal("v0", "v1", "h"))
    with pytest.raises(InfraRateExceeded) as ei:
        world.orchestrator().run_generation()
    msg = str(ei.value)
    assert "current champion (gen-0" in msg and "reset iterant/champion" in msg


def test_a_champion_that_refuses_only_train_tasks_does_not_close_the_batch(world):
    # SMOKE runs on train tasks and never scores candidate refusals: the guard ignores it
    world.solver = lambda arm, t, n: (
        "refuse"
        if arm == "champ" and t in {"tr0", "tr1"}
        else (champ_default(t) if arm == "champ" else "ok")
    )
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()  # does not raise
    assert res.stopped == "done"


def test_a_train_refusing_candidate_cannot_entrench_itself(world):
    def solver(arm: str, task: str, attempt: int) -> str:
        if task.startswith("tr") and arm != "champ":
            return "refuse"  # the candidate refuses on the smoke tasks only
        return champ_default(task) if arm == "champ" else "ok"

    world.solver = solver
    world.queue(world.proposal("v0", "v1", "h"))
    assert world.orchestrator().run_generation().promoted == "b1-0"

    # next generation: the promoted candidate is the champion and still refuses train tasks
    world.solver = lambda arm, task, attempt: "refuse" if task.startswith("tr") else "ok"
    world.queue(world.proposal("v1", "v2", "h2"))
    res = world.orchestrator().run_generation()  # does not raise
    assert res.gen == 2 and res.stopped == "done"
