"""One-sided paired sign-flip randomization test for the mean per-task gain."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import compress

# Production default. Callers derive a distinct seed per stage (SELECT vs CONFIRM)
# and per candidate; tests may lower this.
DEFAULT_RESAMPLES = 10_000
# With at most this many nonzero deltas all 2^n sign patterns are enumerated.
EXACT_MAX_N = 12
_TOL = 1e-9


@dataclass(frozen=True)
class PairedTestResult:
    delta: float  # mean per-task gain
    p: float  # one-sided, H0: gains are symmetric about 0
    n: int  # tasks used (zeros included)


def paired_signflip(
    deltas: Sequence[float], *, seed: int, n_resamples: int = DEFAULT_RESAMPLES
) -> PairedTestResult:
    """One-sided paired randomization test of H0: per-task gains are symmetric around 0.

    Under H0 each task's gain is as likely to be ``+x`` as ``-x``, so the observed sum
    is compared with the sum under random sign flips. Tasks with zero gain never
    change the sum and are ignored. With at most ``EXACT_MAX_N`` nonzero gains the
    p-value is exact, ``#{S* >= S} / 2^n`` (an all-positive gain over n tasks gives
    ``2^-n``). Otherwise it is Monte Carlo, ``(1 + #{S* >= S}) / (B + 1)``, which is
    never 0 and is valid for any B. Seeded and deterministic. Sums are compared with a
    1e-9 tolerance in the conservative direction.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    n = len(deltas)
    if n == 0:
        return PairedTestResult(0.0, 1.0, 0)
    mean = math.fsum(deltas) / n
    mags = [abs(x) for x in deltas if x != 0.0]
    obs = math.fsum(x for x in deltas if x != 0.0)
    if not mags:
        return PairedTestResult(mean, 1.0, n)
    if len(mags) <= EXACT_MAX_N:
        sums = [0.0]
        for a in mags:
            sums = [s + a for s in sums] + [s - a for s in sums]
        hits = sum(1 for s in sums if s >= obs - _TOL)
        return PairedTestResult(mean, hits / len(sums), n)
    total = math.fsum(mags)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        # sum of +/-a_i = 2 * (sum of the a_i flipped to +) - total
        s = 2.0 * sum(compress(mags, rng.choices((0, 1), k=len(mags)))) - total
        if s >= obs - _TOL:
            hits += 1
    return PairedTestResult(mean, (1 + hits) / (n_resamples + 1), n)
