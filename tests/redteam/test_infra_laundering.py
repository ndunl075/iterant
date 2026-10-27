"""Cross-cutting attack on rows 5 and 6: make the candidate's own failures look like infra errors.

§6 drops a task pair from both arms when a run hits an infra error twice. A candidate that can
make its hard tasks end as "infra" (lose the sandbox, time out, break ``get``, ruin the
listing, trip the refusal classifier) would shed exactly the tasks it fails and keep only its
wins. Every failure the candidate can cause must score 0.0 as the candidate's doing; the
candidate must not be promoted.

The setup: the champion solves ``HARD`` (se00-02, ho0-1). The candidate solves every other
select/holdout task, so if its failures on ``HARD`` were dropped as infra, it would be
7-for-7 on select and 4-for-4 on holdout and be promoted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from iterant.contracts import ExecResult, Split, Stage, Status

HARD = {"se00", "se01", "se02", "ho0", "ho1"}
CHAMP_SOLVES = {"tr0", "tr1", "se00", "se01", "se02", "ho0", "ho1"}


def attack_solver(outcome: str) -> Callable[[str, str, int], str]:
    def solver(arm: str, task: str, attempt: int) -> str:
        if arm == "champ":
            return "ok" if task in CHAMP_SOLVES else "wrong"
        return outcome if task in HARD else "ok"

    return solver


def run_attack(world: Any, outcome: str) -> Any:
    world.solver = attack_solver(outcome)
    world.queue(world.proposal("v0", "v1", "h"))
    return world.orchestrator().run_generation()


def final_rows(world: Any, task_ids: set[str]) -> list[Any]:
    db = world.db()
    rows = db.runs_for_candidate("b1-0", batch=1)
    return [r for r in rows if r.key.task in task_ids]


def assert_not_laundered(world: Any, res: Any) -> None:
    assert res.promoted is None, "the candidate shed its hard tasks and was promoted"
    rows = final_rows(world, HARD)
    assert rows, "the candidate never ran its hard tasks"
    for r in rows:
        assert (r.score, r.infra_err, r.refusal) == (0.0, False, False), r


@pytest.mark.parametrize(
    "outcome", ["lost", "lost-raise", "create-lost", "timeout", "truncated", "crash"]
)
def test_row5_candidate_caused_agent_failures_score_zero_not_infra(world, outcome: str) -> None:  # type: ignore[no-untyped-def]
    """Killing PID 1, an OOM or pids kill, a hang, endless output or a crash: all its own."""
    res = run_attack(world, outcome)
    assert_not_laundered(world, res)


def hook_for(attack: str) -> Callable[..., Any]:
    def hook(sb: Any, key: Any, result: ExecResult) -> ExecResult | None:
        if key.cand != "b1-0" or key.task not in HARD:
            return None
        if attack == "get-fails":
            sb.get_broken = True  # chmod 000, a FIFO, a path too deep: get fails afterwards
        elif attack == "symlink-in-out":
            sb.listing_override = "l\t1\t5\tanswer.txt\0"
        elif attack == "hardlink-in-out":
            sb.listing_override = "f\t2\t2\tanswer.txt\0"
        elif attack == "fifo-in-out":
            sb.listing_override = "p\t1\t0\tanswer.txt\0"
        elif attack == "huge-file":
            sb.listing_override = f"f\t1\t{5 * 1024 * 1024}\tanswer.txt\0"
        elif attack == "too-many-files":
            sb.listing_override = "".join(f"f\t1\t1\tf{i}\0" for i in range(1500))
        elif attack == "too-many-entries":
            sb.listing_override = "".join(f"d\t1\t0\td{i}\0" for i in range(2500))
        elif attack == "garbage-listing":
            sb.listing_override = "not a find record\0"
        elif attack == "empty-listing":
            sb.listing_override = ""
        elif attack == "traversal-name":
            sb.listing_override = "f\t1\t2\t../../etc/passwd\0"
            sb.files["/work/out/../../etc/passwd"] = b"root"
        elif attack == "mangled-name":
            sb.listing_override = "f\t1\t2\tbad�name\0"
        elif attack == "vanishing-file":
            sb.listing_override = "f\t1\t2\tghost.txt\0"
        return None

    return hook


@pytest.mark.parametrize(
    "attack",
    [
        "get-fails",
        "symlink-in-out",
        "hardlink-in-out",
        "fifo-in-out",
        "huge-file",
        "too-many-files",
        "too-many-entries",
        "garbage-listing",
        "empty-listing",
        "traversal-name",
        "vanishing-file",
        "mangled-name",
    ],
)
def test_row6_candidate_caused_out_failures_score_zero_not_infra(world, attack: str) -> None:  # type: ignore[no-untyped-def]
    world.after_agent = hook_for(attack)
    res = run_attack(world, "ok")
    assert_not_laundered(world, res)


def test_row6_a_wrecked_sandbox_on_a_healthy_runtime_is_the_candidates_doing(world) -> None:  # type: ignore[no-untyped-def]
    """``get`` fails and the sandbox's own probe fails too, but a fresh sandbox is fine."""
    world.agent_sandbox_broken = True
    world.after_agent = hook_for("get-fails")
    res = run_attack(world, "ok")
    assert_not_laundered(world, res)


def test_row6_when_the_runtime_really_is_down_the_run_is_infra_and_the_candidate_does_not_win(  # type: ignore[no-untyped-def]
    world,
) -> None:
    """The mirror image: a real outage is infra. The candidate must still not be promoted by
    shedding the tasks the outage hit (the infra guard rejects an asymmetric outage)."""
    world.after_agent = hook_for("get-fails")
    world.runtime_down = True
    res = run_attack(world, "ok")
    assert res.promoted is None


def test_refusals_the_candidate_provokes_on_its_hard_tasks_do_not_get_it_promoted(world) -> None:  # type: ignore[no-untyped-def]
    """§6 drops a pair after a second refusal. A candidate whose prompt trips the classifier
    on exactly the tasks it would lose must be stopped by the refusal guard."""
    res = run_attack(world, "refuse")
    assert res.promoted is None
    [cand] = res.candidates
    assert (cand.stage, cand.status) == (Stage.SELECT, Status.REJECTED)
    assert any("refusal" in n for n in cand.notes), cand.notes


def test_a_candidate_cannot_use_an_infra_looking_first_try_to_get_a_second_chance_at_a_win(  # type: ignore[no-untyped-def]
    world,
) -> None:
    """Attempt 0 fails as a real infra error, attempt 1 is the candidate's honest answer. The
    retry counts (once); there is never a third try."""
    attempts: list[int] = []

    def solver(arm: str, task: str, attempt: int) -> str:
        if arm == "b1-0" and task == "se09":
            attempts.append(attempt)
            return "infra"
        return "ok" if arm != "champ" else ("ok" if task in CHAMP_SOLVES else "wrong")

    world.solver = solver
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    assert sorted(set(attempts)) == [0, 1] and len(attempts) == 2
    rows = world.db().runs_for_candidate("b1-0", split=Split.SELECT, latest_only=False)
    assert [r.key.attempt for r in rows if r.key.task == "se09"] == [0, 1]
