from __future__ import annotations

from collections.abc import Callable

import pytest

from iterant.contracts import AcceptanceThresholds, RunResult
from iterant.stats import estimate, power, smallest_detectable_gain, tasks_needed

MakeRun = Callable[..., RunResult]


def test_matches_architecture_power_table() -> None:
    # §4: 15% discordant, power 0.8, one-sided alpha 0.05, k=1
    assert 75 <= tasks_needed(0.10, 0.15) <= 90
    assert 340 <= tasks_needed(0.05, 0.15) <= 380
    assert 950 <= tasks_needed(0.03, 0.15) <= 1050


def test_hand_computed_values() -> None:
    # z(.95)+z(.8) = 1.644854 + 0.841621 = 2.486475; squared = 6.18256
    # n = 6.18256 * (0.15 - 0.01) / 0.01 = 86.56 -> 87
    assert tasks_needed(0.10, 0.15) == 87
    assert power(87, 0.10, 0.15) >= 0.8
    assert power(86, 0.10, 0.15) < 0.8
    assert power(80, 0.10, 0.15) == pytest.approx(0.772, abs=0.001)


def test_detectable_gain_inverts_power() -> None:
    for n, reps in ((80, 1), (80, 2), (40, 2), (360, 1)):
        g = smallest_detectable_gain(n, 0.15, reps)
        assert power(n, g, 0.15, reps) == pytest.approx(0.8, abs=1e-9)
    # default select size: 80 tasks x 2 reps detects roughly 7 pts
    assert 0.06 < smallest_detectable_gain(80, 0.15, 2) < 0.08


def test_power_is_monotone() -> None:
    assert power(100, 0.05, 0.15) < power(100, 0.10, 0.15) < power(400, 0.10, 0.15)
    assert power(100, 0.05, 0.15, reps=2) > power(100, 0.05, 0.15, reps=1)
    assert power(100, 0.05, 0.30) < power(100, 0.05, 0.15)


def test_power_input_validation() -> None:
    with pytest.raises(ValueError):
        power(0, 0.1, 0.15)
    with pytest.raises(ValueError):
        power(10, 0.2, 0.15)  # gain above discordance is impossible
    with pytest.raises(ValueError):
        smallest_detectable_gain(10, 0.0)
    with pytest.raises(ValueError):
        tasks_needed(0.1, 0.15, reps=0)


def test_estimate_hand_computed(mk: MakeRun) -> None:
    # t1: 1,1 (mean 1, var 0)   t2: 0,1 (mean .5, var .5)   t3: 0,0 (mean 0, var 0)
    runs = [
        mk("t1", 1.0, rep=0, usd=2000),
        mk("t1", 1.0, rep=1, usd=2000),
        mk("t2", 0.0, rep=0, usd=4000),
        mk("t2", 1.0, rep=1, usd=4000),
        mk("t3", 0.0, rep=0, usd=3000),
        mk("t3", 0.0, rep=1, usd=3000),
    ]
    t = AcceptanceThresholds(select_tasks=80, select_reps=2, holdout_tasks=40, holdout_reps=2)
    e = estimate(runs, t)
    assert e.n_runs == 6
    assert e.n_tasks == 3
    assert e.baseline_pass_rate == pytest.approx(0.5)
    assert e.score_variance == pytest.approx(0.25)  # var of [1, .5, 0]
    assert e.run_variance == pytest.approx(0.3)  # 1.5 / 5 for [1,1,0,1,0,0]
    assert e.discordance == pytest.approx((0 + 1.0 + 0) / 3)  # 2 * var_within, averaged
    assert e.discordance_from_reps
    assert e.usd_per_run == pytest.approx(0.003)  # 3000 micros mean
    assert e.select_detectable_gain == pytest.approx(
        smallest_detectable_gain(80, 1 / 3, 2, t.select_p)
    )
    assert e.holdout_detectable_gain == pytest.approx(
        smallest_detectable_gain(40, 1 / 3, 2, t.holdout_p)
    )
    # (2 * (80*2 + 40*2) + 5) runs
    assert e.usd_per_candidate == pytest.approx(0.003 * 485)


def test_estimate_single_rep_falls_back_and_ignores_failed_runs(mk: MakeRun) -> None:
    runs = [mk("a", 1.0), mk("b", 0.0), mk("c", None, refusal=True)]
    e = estimate(runs)
    assert not e.discordance_from_reps
    assert e.discordance == pytest.approx(2 * 0.5)  # 2 * var([1, 0])
    assert e.refusal_rate == pytest.approx(1 / 3)
    assert e.n_tasks == 3
    with pytest.raises(ValueError):
        estimate([mk("c", None, infra=True)])
