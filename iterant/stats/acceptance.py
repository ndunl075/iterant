"""The acceptance rule (§4): SELECT, CONFIRM and the guards."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from iterant.contracts import AcceptanceThresholds, RunResult, Split
from iterant.stats.bootstrap import DEFAULT_RESAMPLES, paired_bootstrap
from iterant.stats.pairing import Pairing, final_attempts, pair_runs

# Guards compare sums of ints and means of floats; this absorbs float rounding at an
# exact boundary (115 vs 100 * 1.15 must pass).
_EPS = 1e-9


class Reason(StrEnum):
    NO_PAIRED_TASKS = "no_paired_tasks"
    DELTA_BELOW_MIN = "delta_below_min"  # SELECT: delta < delta_min
    NOT_POSITIVE = "not_positive"  # CONFIRM: delta <= 0
    P_TOO_HIGH = "p_too_high"
    COST_GUARD = "cost_guard"
    WALL_GUARD = "wall_guard"
    REFUSAL_GUARD = "refusal_guard"
    INFRA_GUARD = "infra_guard"


@dataclass(frozen=True)
class Decision:
    """Verdict of one stage. ``reasons`` is empty exactly when ``accepted``; every
    failing check is listed, not just the first. ``detail`` has one line per reason."""

    stage: Split
    accepted: bool
    delta: float
    p: float
    n_tasks: int
    n_dropped: int
    reasons: tuple[Reason, ...]
    detail: tuple[str, ...]


def _rate(runs: Sequence[RunResult], attr: str) -> float:
    finals = list(final_attempts(runs).values())
    return sum(getattr(r, attr) for r in finals) / len(finals) if finals else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _guards(
    pairing: Pairing,
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    t: AcceptanceThresholds,
) -> list[tuple[Reason, str]]:
    """Cost and wall time use the kept tasks' final runs (same tasks in both arms);
    refusal and infra-error rates use every final run, including dropped tasks."""
    out: list[tuple[Reason, str]] = []
    c_cost = _mean([r.usd_micros for r in pairing.candidate_runs])
    h_cost = _mean([r.usd_micros for r in pairing.champion_runs])
    if c_cost > h_cost * (1 + t.max_cost_increase) * (1 + _EPS):
        out.append(
            (
                Reason.COST_GUARD,
                f"cost per run {c_cost:.0f} > champion {h_cost:.0f} micro-USD "
                f"+ {t.max_cost_increase:.0%}",
            )
        )
    c_wall = _mean([r.wall_s for r in pairing.candidate_runs])
    h_wall = _mean([r.wall_s for r in pairing.champion_runs])
    if c_wall > h_wall * (1 + t.max_wall_increase) * (1 + _EPS):
        out.append(
            (
                Reason.WALL_GUARD,
                f"wall time per run {c_wall:.3f}s > champion {h_wall:.3f}s "
                f"+ {t.max_wall_increase:.0%}",
            )
        )
    for reason, attr, label in (
        (Reason.REFUSAL_GUARD, "refusal", "refusal"),
        (Reason.INFRA_GUARD, "infra_err", "infra-error"),
    ):
        c_rate, h_rate = _rate(candidate, attr), _rate(champion, attr)
        if c_rate > h_rate + _EPS:
            out.append((reason, f"{label} rate {c_rate:.1%} worse than champion {h_rate:.1%}"))
    return out


def _decide(
    stage: Split,
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    t: AcceptanceThresholds,
    seed: int,
    n_resamples: int,
) -> Decision:
    pairing = pair_runs(candidate, champion, split=stage)
    boot = paired_bootstrap(pairing.deltas, seed=seed, n_resamples=n_resamples)
    fails: list[tuple[Reason, str]] = []
    if boot.n == 0:
        fails.append((Reason.NO_PAIRED_TASKS, "no task has a scored run in both arms"))
    if stage is Split.SELECT:
        if boot.n and boot.delta < t.delta_min - _EPS:
            fails.append((Reason.DELTA_BELOW_MIN, f"delta {boot.delta:.4f} < {t.delta_min}"))
        p_max = t.select_p
    else:
        if boot.n and boot.delta <= 0:
            fails.append((Reason.NOT_POSITIVE, f"delta {boot.delta:.4f} is not > 0"))
        p_max = t.holdout_p
    if boot.n and not boot.p < p_max:
        fails.append((Reason.P_TOO_HIGH, f"p {boot.p:.4f} is not < {p_max}"))
    fails.extend(_guards(pairing, candidate, champion, t))
    return Decision(
        stage=stage,
        accepted=not fails,
        delta=boot.delta,
        p=boot.p,
        n_tasks=boot.n,
        n_dropped=pairing.n_dropped,
        reasons=tuple(r for r, _ in fails),
        detail=tuple(d for _, d in fails),
    )


def decide_select(
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    thresholds: AcceptanceThresholds | None = None,
    *,
    seed: int,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Decision:
    """SELECT: ``delta >= delta_min`` and ``p < select_p``, plus the guards.

    ``candidate`` and ``champion`` are the two arms' runs from one batch of
    ``select`` tasks.
    """
    return _decide(
        Split.SELECT, candidate, champion, thresholds or AcceptanceThresholds(), seed, n_resamples
    )


def decide_confirm(
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    thresholds: AcceptanceThresholds | None = None,
    *,
    seed: int,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Decision:
    """CONFIRM: holdout ``delta > 0`` and ``p < holdout_p``, plus the guards, on fresh
    holdout runs of both arms."""
    return _decide(
        Split.HOLDOUT, candidate, champion, thresholds or AcceptanceThresholds(), seed, n_resamples
    )


def accepted(select: Decision, confirm: Decision) -> bool:
    """Promote only if both stages passed."""
    return (
        select.stage is Split.SELECT
        and confirm.stage is Split.HOLDOUT
        and select.accepted
        and confirm.accepted
    )
