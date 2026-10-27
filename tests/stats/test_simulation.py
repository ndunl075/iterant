"""Monte Carlo checks: A/A false-accept rate and power against ``power()``."""

from __future__ import annotations

import random

from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split
from iterant.stats import accepted, decide_confirm, decide_select, power

# Per-task pass probabilities: easy, hard and middling tasks, so most pairs agree.
_BASE = (0.05, 0.1, 0.1, 0.3, 0.5, 0.9, 0.95)


def _arm(
    rng: random.Random,
    probs: list[float],
    reps: int,
    cand: str,
    split: Split,
    batch: int,
) -> list[RunResult]:
    return [
        RunResult(
            key=RunKey(batch, cand, f"t{i}", rep),
            split=split,
            score=1.0 if rng.random() < p else 0.0,
            in_tok=1,
            out_tok=1,
            usd_micros=1000,
            wall_s=10.0,
            refusal=False,
            infra_err=False,
        )
        for i, p in enumerate(probs)
        for rep in range(reps)
    ]


def _tasks(rng: random.Random, n: int) -> list[float]:
    return [rng.choice(_BASE) for _ in range(n)]


def test_aa_full_select_and_confirm_false_accept_rate_at_most_5_percent() -> None:
    t = AcceptanceThresholds()  # 80x2 select, 40x2 holdout
    rng = random.Random(12345)
    trials, false_accepts, select_passes = 400, 0, 0
    for i in range(trials):
        sel_p = _tasks(rng, t.select_tasks)
        hold_p = _tasks(rng, t.holdout_tasks)
        s_c = _arm(rng, sel_p, t.select_reps, "c", Split.SELECT, 1)
        s_h = _arm(rng, sel_p, t.select_reps, "h", Split.SELECT, 1)
        sel = decide_select(s_c, s_h, t, seed=i, n_resamples=300)
        select_passes += sel.accepted
        h_c = _arm(rng, hold_p, t.holdout_reps, "c", Split.HOLDOUT, 2)
        h_h = _arm(rng, hold_p, t.holdout_reps, "h", Split.HOLDOUT, 2)
        conf = decide_confirm(h_c, h_h, t, seed=i, n_resamples=300)
        false_accepts += accepted(sel, conf)
    rate = false_accepts / trials
    assert rate <= 0.05, f"A/A false accept {rate:.3f} (select alone passed {select_passes})"
    assert select_passes / trials <= 0.08  # SELECT alone; p-value calibration is tested separately


def test_known_gain_detection_rate_matches_power() -> None:
    # Champion passes each task with p = 0.1; candidate with p = 0.2 (true gain 0.1).
    # Independent Bernoulli runs: P(runs differ) = 0.1*0.8 + 0.2*0.9 = 0.26.
    n, reps, gain, disc = 40, 2, 0.10, 0.26
    expected = power(n, gain, disc, reps, alpha=0.05)
    t = AcceptanceThresholds(select_tasks=n, select_reps=reps, delta_min=0.0001, select_p=0.05)
    rng = random.Random(777)
    trials, hits = 300, 0
    for i in range(trials):
        c = _arm(rng, [0.2] * n, reps, "c", Split.SELECT, 1)
        h = _arm(rng, [0.1] * n, reps, "h", Split.SELECT, 1)
        hits += decide_select(c, h, t, seed=i, n_resamples=300).accepted
    observed = hits / trials
    assert 0.35 < expected < 0.75  # a mid-range case, where a mismatch would show
    assert abs(observed - expected) < 0.12, f"observed {observed:.3f} vs power() {expected:.3f}"
