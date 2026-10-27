"""The A/A resampling: identical arms must rarely pass the §4 rule; a broken rule is caught."""

from __future__ import annotations

import random

from iterant.cli.calibrate import MAX_FALSE_ACCEPT, run_aa
from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split


def _pool(split: Split, n_tasks: int, reps: int, rate: float, seed: int = 1) -> list[RunResult]:
    """Champion runs where every run passes with chance ``rate`` (noisy, no real signal)."""
    rng = random.Random(seed)
    return [
        RunResult(
            RunKey(1, "champion", f"t{i:02d}", rep, 0),
            split,
            1.0 if rng.random() < rate else 0.0,
            10,
            10,
            100,
            1.0,
            False,
            False,
        )
        for i in range(n_tasks)
        for rep in range(reps)
    ]


def test_the_real_rule_has_few_false_accepts() -> None:
    t = AcceptanceThresholds(select_tasks=30, select_reps=1, holdout_tasks=20, holdout_reps=1)
    res = run_aa(
        _pool(Split.SELECT, 30, 2, 0.5),
        t,
        trials=100,
        seed=3,
        n_resamples=300,
    )
    assert res.trials == 100 and res.passed and res.rate <= MAX_FALSE_ACCEPT


def test_a_broken_rule_fails_the_test() -> None:
    # p thresholds of 1 and no minimum gain accept almost any non-negative noise
    t = AcceptanceThresholds(
        select_tasks=30,
        select_reps=1,
        holdout_tasks=20,
        holdout_reps=1,
        delta_min=0.0,
        select_p=1.0,
        holdout_p=1.0,
        max_cost_increase=10.0,
        max_wall_increase=10.0,
    )
    res = run_aa(
        _pool(Split.SELECT, 30, 2, 0.5),
        t,
        trials=60,
        seed=3,
        n_resamples=200,
    )
    assert not res.passed and res.rate > MAX_FALSE_ACCEPT


def test_same_seed_same_result() -> None:
    t = AcceptanceThresholds(select_tasks=15, select_reps=1, holdout_tasks=10, holdout_reps=1)
    args = (_pool(Split.SELECT, 15, 2, 0.5), t)
    a = run_aa(*args, trials=20, seed=9, n_resamples=100)
    b = run_aa(*args, trials=20, seed=9, n_resamples=100)
    assert a == b
