"""One-sided paired bootstrap for the mean per-task gain."""

from __future__ import annotations

import math
import operator
import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class BootstrapResult:
    delta: float  # mean per-task gain
    p: float  # one-sided, H0: true mean gain <= 0
    n: int  # tasks used


def paired_bootstrap(
    deltas: Sequence[float], *, seed: int, n_resamples: int = DEFAULT_RESAMPLES
) -> BootstrapResult:
    """Studentized one-sided bootstrap test of H0: mean(deltas) <= 0.

    The deltas are centered at zero to impose H0, resampled with replacement, and
    the resampled t statistic is compared with the observed one. ``p`` is
    ``(1 + #{t* >= t}) / (B + 1)``, so it is never 0. Deterministic for a given
    ``seed``. Fewer than two pairs give ``p = 1``.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    n = len(deltas)
    if n == 0:
        return BootstrapResult(0.0, 1.0, 0)
    mean = math.fsum(deltas) / n
    if n < 2:
        return BootstrapResult(mean, 1.0, n)
    var = math.fsum((x - mean) ** 2 for x in deltas) / (n - 1)
    if var <= 1e-18:  # every pair gained the same amount
        p = 1.0 if mean <= 0 else max(2.0**-n, 1.0 / (n_resamples + 1))
        return BootstrapResult(mean, p, n)
    t_obs = mean / math.sqrt(var / n)
    centered = [x - mean for x in deltas]
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        xs = rng.choices(centered, k=n)
        m = sum(xs) / n
        v = (sum(map(operator.mul, xs, xs)) - n * m * m) / (n - 1)
        if v <= 1e-18:
            t = math.copysign(math.inf, m) if abs(m) > 1e-12 else 0.0
        else:
            t = m / math.sqrt(v / n)
        if t >= t_obs:
            hits += 1
    return BootstrapResult(mean, (1 + hits) / (n_resamples + 1), n)
