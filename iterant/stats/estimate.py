"""``iterant estimate`` math: what a sample of champion runs says about the pack (§2, §4)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from iterant.contracts import AcceptanceThresholds, RunResult
from iterant.stats.pairing import final_attempts
from iterant.stats.power import smallest_detectable_gain


@dataclass(frozen=True)
class Estimate:
    """Numbers printed before any spend. Money is in dollars (inputs are micro-USD)."""

    n_runs: int  # final attempts in the sample
    n_tasks: int
    baseline_pass_rate: float  # mean over tasks of the task's mean score
    score_variance: float  # variance of task means across tasks
    run_variance: float  # variance of single-run scores
    discordance: float  # chance two runs of one task differ (see stats.power)
    discordance_from_reps: bool  # False: no task had 2+ scored reps, bound used instead
    refusal_rate: float
    infra_err_rate: float
    usd_per_run: float  # all attempts, retries included
    select_detectable_gain: float
    holdout_detectable_gain: float
    usd_per_candidate: float


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = math.fsum(xs) / n
    return math.fsum((x - m) ** 2 for x in xs) / (n - 1)


def estimate(
    runs: Sequence[RunResult],
    thresholds: AcceptanceThresholds | None = None,
    *,
    target_power: float = 0.8,
) -> Estimate:
    """Summarize a sample of champion runs (any split, several reps per task is best).

    Discordance is ``2 * within-task variance`` averaged over tasks with 2+ scored
    reps: the mean squared difference between two independent runs of one task. With
    no such task it falls back to ``2 * run_variance``, an upper bound. Detectable
    gains use the select and holdout sizes in ``thresholds`` at ``select_p`` and
    ``holdout_p``. ``usd_per_candidate`` assumes the candidate costs what the champion
    does, counts both arms in full (no shared-arm saving) and adds the smoke runs of
    the candidate only. Raises ``ValueError`` when no run was scored.
    """
    t = thresholds or AcceptanceThresholds()
    finals = list(final_attempts(runs).values())
    by_task: dict[str, list[float]] = {}
    for r in finals:
        if r.score is not None:
            by_task.setdefault(r.key.task, []).append(r.score)
    if not by_task:
        raise ValueError("no scored runs in the sample")
    scores = [s for xs in by_task.values() for s in xs]
    task_means = [math.fsum(xs) / len(xs) for xs in by_task.values()]
    run_var = _var(scores)
    within = [2 * _var(xs) for xs in by_task.values() if len(xs) >= 2]
    from_reps = bool(within)
    disc = math.fsum(within) / len(within) if from_reps else 2 * run_var
    disc = min(max(disc, 1e-6), 1.0)  # a perfectly stable pack still needs a finite answer
    usd_per_run = math.fsum(r.usd_micros for r in runs) / len(runs) / 1_000_000
    arm_runs = t.select_tasks * t.select_reps + t.holdout_tasks * t.holdout_reps
    return Estimate(
        n_runs=len(finals),
        n_tasks=len({r.key.task for r in finals}),
        baseline_pass_rate=math.fsum(task_means) / len(task_means),
        score_variance=_var(task_means),
        run_variance=run_var,
        discordance=disc,
        discordance_from_reps=from_reps,
        refusal_rate=sum(r.refusal for r in finals) / len(finals),
        infra_err_rate=sum(r.infra_err for r in finals) / len(finals),
        usd_per_run=usd_per_run,
        select_detectable_gain=smallest_detectable_gain(
            t.select_tasks, disc, t.select_reps, t.select_p, target_power
        ),
        holdout_detectable_gain=smallest_detectable_gain(
            t.holdout_tasks, disc, t.holdout_reps, t.holdout_p, target_power
        ),
        usd_per_candidate=usd_per_run * (2 * arm_runs + t.smoke_tasks),
    )
