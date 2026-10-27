"""Power and smallest detectable gain for the paired SELECT/CONFIRM test (§4).

Model: per rep, the two arms disagree on a fraction ``discordance`` of tasks (the
mean squared difference between paired runs; for 0/1 scores, the chance the two
runs differ). With true gain ``g`` the per-task difference has variance ``d - g^2``.
Averaging ``reps`` reps divides that by ``reps``, which assumes the disagreement is
run noise, not a persistent per-task effect. If tasks differ persistently between
arms, real power at ``reps > 1`` is lower than reported here. The test is a
one-sided z test (the sign-flip test in use is close to it for these sizes).

Exact figures at 15% discordance, k=1, power 0.8, alpha 0.05: +10 pts needs 87
tasks, +5 needs 365, +3 needs 1,025. ARCHITECTURE.md section 4 rounds these to ~80, ~360
and ~1,000.
"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import NormalDist

from iterant.contracts import AcceptanceThresholds

_N = NormalDist()


def _check(discordance: float, reps: int, alpha: float, target: float = 0.5) -> None:
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must be in (0, 1]")
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if not (0.0 < alpha < 1.0 and 0.0 < target < 1.0):
        raise ValueError("alpha and power must be in (0, 1)")


def power(
    n_tasks: int, gain: float, discordance: float, reps: int = 1, alpha: float = 0.05
) -> float:
    """Chance a one-sided test at ``alpha`` detects a true gain of ``gain``."""
    _check(discordance, reps, alpha)
    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1")
    if not 0.0 < gain <= discordance:
        raise ValueError("gain must be in (0, discordance]")
    se = math.sqrt((discordance - gain**2) / (n_tasks * reps))
    if se == 0.0:
        return 1.0
    return _N.cdf(gain / se - _N.inv_cdf(1 - alpha))


def smallest_detectable_gain(
    n_tasks: int,
    discordance: float,
    reps: int = 1,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> float:
    """Smallest true gain detected with probability ``target_power`` (closed form)."""
    _check(discordance, reps, alpha, target_power)
    if n_tasks < 1:
        raise ValueError("n_tasks must be >= 1")
    z2 = (_N.inv_cdf(1 - alpha) + _N.inv_cdf(target_power)) ** 2
    return math.sqrt(z2 * discordance / (n_tasks * reps + z2))


def tasks_needed(
    gain: float,
    discordance: float,
    reps: int = 1,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> int:
    """Fewest tasks that detect ``gain`` with probability ``target_power``."""
    _check(discordance, reps, alpha, target_power)
    if not 0.0 < gain <= discordance:
        raise ValueError("gain must be in (0, discordance]")
    z2 = (_N.inv_cdf(1 - alpha) + _N.inv_cdf(target_power)) ** 2
    return math.ceil(z2 * (discordance - gain**2) / (reps * gain**2))


def rule_power(gain: float, discordance: float, t: AcceptanceThresholds) -> float:
    """Chance the full rule (SELECT then CONFIRM, on independent task draws) accepts a true
    ``gain``; at ``gain=0`` it is the A/A false-accept rate.

    SELECT needs the mean gain to reach ``delta_min`` and the one-sided test at ``select_p``;
    CONFIRM needs it positive and the test at ``holdout_p``. Normal approximations with the
    ``power()`` variance model (the sign-flip test is a little conservative, so this is a
    slight overstatement). Ignores the cost, wall and refusal guards, which an
    equal-behaviour candidate rarely trips."""
    if not 0.0 <= gain < 1.0 or not 0.0 < discordance <= 1.0:
        raise ValueError("gain must be in [0, 1) and discordance in (0, 1]")
    prob = 1.0
    for n, k, p, floor in (
        (t.select_tasks, t.select_reps, t.select_p, t.delta_min),
        (t.holdout_tasks, t.holdout_reps, t.holdout_p, 0.0),
    ):
        se0 = math.sqrt(discordance / (n * k))
        se_g = math.sqrt(max(discordance - gain**2, 1e-12) / (n * k))
        crit = max(floor, _N.inv_cdf(1 - p) * se0, 0.0)
        prob *= 1.0 - _N.cdf((crit - gain) / se_g)
    return prob


def _scaled(t: AcceptanceThresholds, tasks: int, reps: int) -> AcceptanceThresholds:
    """``t`` with the select size set and the holdout scaled in proportion."""
    return replace(
        t,
        select_tasks=max(1, tasks),
        holdout_tasks=max(1, round(tasks * t.holdout_tasks / t.select_tasks)),
        select_reps=reps,
        holdout_reps=max(1, round(reps * t.holdout_reps / t.select_reps)),
    )


def rule_tasks_needed(
    gain: float,
    discordance: float,
    t: AcceptanceThresholds,
    target: float = 0.5,
    limit: int = 100_000,
) -> int | None:
    """Fewest select tasks (holdout scaled in proportion, reps as in ``t``) at which
    ``rule_power`` reaches ``target``; ``None`` when ``limit`` tasks are not enough."""
    if rule_power(gain, discordance, _scaled(t, limit, t.select_reps)) < target:
        return None
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi) // 2
        if rule_power(gain, discordance, _scaled(t, mid, t.select_reps)) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def rule_reps_needed(
    gain: float,
    discordance: float,
    t: AcceptanceThresholds,
    target: float = 0.5,
    limit: int = 20,
) -> int | None:
    """Fewest SELECT reps per task (holdout reps scaled in proportion, tasks as in ``t``) at
    which ``rule_power`` reaches ``target``; ``None`` when ``limit`` reps are not enough.
    Assumes disagreement is run noise (see the module note)."""
    for reps in range(1, limit + 1):
        if rule_power(gain, discordance, _scaled(t, t.select_tasks, reps)) >= target:
            return reps
    return None
