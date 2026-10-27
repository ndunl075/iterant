"""The acceptance rule (§4): SELECT, CONFIRM and the guards."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from iterant.contracts import AcceptanceThresholds, RunResult, Split
from iterant.stats.pairing import Pairing, pair_runs
from iterant.stats.signflip import DEFAULT_RESAMPLES, paired_signflip

# Guards compare sums of ints and means of floats; this absorbs float rounding at an
# exact boundary (115 vs 100 * 1.15 must pass).
_EPS = 1e-9

# More refused/infra-errored task pairs than this share of all tasks rejects the
# candidate: a verdict on few surviving tasks is not a verdict on the pack.
MAX_DROPPED_FRACTION = 0.10


class Reason(StrEnum):
    NO_PAIRED_TASKS = "no_paired_tasks"
    TOO_MANY_DROPPED = "too_many_dropped"
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


def _first_try_events(runs: Sequence[RunResult], attr: str) -> tuple[int, int]:
    """``(events, runs)`` over every first attempt (retries excluded: a retried run's final
    outcome is about the rate squared, which hides a real excess)."""
    firsts = [r for r in runs if r.key.attempt == 0]
    return sum(bool(getattr(r, attr)) for r in firsts), len(firsts)


def _binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, summed in log space (exact)."""
    if k <= 0:
        return 1.0
    if k > n or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        for i in range(k, n + 1)
    ]
    top = max(logs)
    return min(1.0, math.exp(top) * math.fsum(math.exp(x - top) for x in logs))


def noise_veto(
    cand_events: int, cand_runs: int, champ_events: int, champ_runs: int, t: AcceptanceThresholds
) -> tuple[bool, float, float]:
    """``(veto, p, excess)`` for one rate (refusal or infra error), first attempts only.

    ``p`` is the one-sided conditional binomial test: of all events in both arms, the
    candidate's share under H0 is its share of the runs. ``excess`` is the candidate's rate
    minus the champion's. Veto only when ``p < noise_guard_p`` and the excess is more than
    ``noise_guard_margin``."""
    if cand_runs == 0 or champ_runs == 0:
        return False, 1.0, 0.0
    total = cand_events + champ_events
    p = _binom_sf(cand_events, total, cand_runs / (cand_runs + champ_runs))
    excess = cand_events / cand_runs - champ_events / champ_runs
    return (p < t.noise_guard_p and excess > t.noise_guard_margin + _EPS), p, excess


def _guards(
    pairing: Pairing,
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    t: AcceptanceThresholds,
) -> list[tuple[Reason, str]]:
    """Cost and wall time count every attempt (retries included) of the kept tasks,
    per final run, so a refused first try that cost money still counts. Refusal and
    infra-error use every first attempt, including dropped tasks, and veto only a
    statistically clear and materially larger rate (``noise_veto``)."""
    out: list[tuple[Reason, str]] = []
    n_c, n_h = len(pairing.candidate_runs) or 1, len(pairing.champion_runs) or 1
    c_cost = sum(r.usd_micros for r in pairing.candidate_attempts) / n_c
    h_cost = sum(r.usd_micros for r in pairing.champion_attempts) / n_h
    if c_cost > h_cost * (1 + t.max_cost_increase) * (1 + _EPS):
        out.append(
            (
                Reason.COST_GUARD,
                f"cost per run {c_cost:.0f} > champion {h_cost:.0f} micro-USD "
                f"+ {t.max_cost_increase:.0%}",
            )
        )
    c_wall = sum(r.wall_s for r in pairing.candidate_attempts) / n_c
    h_wall = sum(r.wall_s for r in pairing.champion_attempts) / n_h
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
        k_c, m_c = _first_try_events(candidate, attr)
        k_h, m_h = _first_try_events(champion, attr)
        veto, p, excess = noise_veto(k_c, m_c, k_h, m_h, t)
        if veto:
            out.append(
                (
                    reason,
                    f"first-try {label} rate {k_c / m_c:.1%} worse than champion "
                    f"{k_h / m_h:.1%} (+{excess * 100:.1f} pts, p={p:.3f})",
                )
            )
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
    boot = paired_signflip(pairing.deltas, seed=seed, n_resamples=n_resamples)
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
    n_all = boot.n + pairing.n_dropped
    if pairing.n_dropped > MAX_DROPPED_FRACTION * n_all:
        fails.append(
            (
                Reason.TOO_MANY_DROPPED,
                f"{pairing.n_dropped} of {n_all} tasks dropped (refused/infra-error) "
                f"> {MAX_DROPPED_FRACTION:.0%}",
            )
        )
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

    ``candidate`` and ``champion`` are the two arms' runs (all attempts) from one batch
    of ``select`` tasks. ``seed`` should be derived per stage and per candidate, so
    SELECT and CONFIRM never reuse one random stream. Production uses the default
    ``n_resamples``.
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
    holdout runs of both arms. Use a different ``seed`` than SELECT's (see
    ``decide_select``)."""
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
