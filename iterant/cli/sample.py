"""Run one candidate's agent on a list of tasks through the orchestrator's own run machinery.

Used for the baseline runs of ``estimate`` and ``aa-test`` and for the failure-digest sample of
``run``. Rows follow the §1.6 rules: one row per attempt, the single §6 retry of a refused or
infra-errored run, a finished slot is never run twice.
"""

from __future__ import annotations

from collections.abc import Sequence

from iterant.archive import Archive
from iterant.cli.runtime import Runtime
from iterant.contracts import Role, RunKey, RunResult, Task
from iterant.orchestrator import (
    Budget,
    OrchestratorConfig,
    Outcomes,
    RunExecutor,
)


def worst_run_cost(rt: Runtime) -> int:
    """The most one target run may cost in micro-USD (0 for the token-capped free profile)."""
    cap = rt.proxy.config.policy.roles.get(Role.TARGET)
    return (cap.max_usd_micros_per_run or 0) if cap else 0


def make_executor(rt: Runtime, config: OrchestratorConfig) -> RunExecutor:
    project = rt.project
    return RunExecutor(
        backend=rt.backend,
        grader=rt.grader,
        proxy=rt.proxy,
        run_command=project.config.target.run,
        read_input=lambda t: (project.pack_root / t.input).read_bytes(),
        config=config,
    )


def run_slots(
    *,
    archive: Archive,
    executor: RunExecutor,
    outcomes: Outcomes | None,
    budget: Budget,
    worst: int,
    batch: int,
    cand: str,
    image: str,
    plan: Sequence[tuple[Task, int]],
) -> int:
    """Run every ``(task, rep)`` of ``plan`` that has no finished slot in ``batch``.

    Returns the micro-USD these runs cost. ``budget.check`` runs before each run and raises
    ``BudgetExhausted``; rows already written stay, so a rerun resumes.
    """
    done = archive.latest_runs(batch)
    spent = 0
    for task, rep in plan:
        prev = done.get((cand, task.id, rep))
        if prev is not None and (prev.score is not None or prev.key.attempt >= 1):
            continue
        attempt = 0 if prev is None else 1
        while True:
            budget.check(worst)
            key = RunKey(batch, cand, task.id, rep, attempt)
            ran = executor.execute(key, task, image)
            run: RunResult = ran.result
            if outcomes is not None:
                outcomes.record(key, ran.fault)
            archive.insert_run(run)
            spent += run.usd_micros
            if run.score is not None or attempt >= 1:
                break
            attempt += 1
    return spent
