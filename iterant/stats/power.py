"""Power and smallest detectable gain for the paired SELECT/CONFIRM test (§4).

Model: per rep, the two arms disagree on a fraction ``discordance`` of tasks (the
mean squared difference between paired runs; for 0/1 scores, the chance the two
runs differ). With true gain ``g`` the per-task difference has variance ``d - g^2``.
Averaging ``reps`` reps divides that by ``reps``, which assumes the disagreement is
run noise, not a persistent per-task effect. The test is a one-sided z test.
"""

from __future__ import annotations

import math
from statistics import NormalDist

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
