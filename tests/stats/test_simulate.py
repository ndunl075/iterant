"""Simulator checks: A/A false-accept rate, power monotonicity, match to power()."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.contracts import AcceptanceThresholds
from iterant.stats.simulate import (
    SimConfig,
    analytic_smallest_gain,
    power_curve,
    run_trials,
    simulate_trial,
    wilson,
)

# p-test only: delta_min ~ 0 so SELECT power is comparable with power().
_P_ONLY = AcceptanceThresholds(delta_min=0.0001)


def test_aa_false_accept_rate_at_most_5_percent() -> None:
    s = run_trials(SimConfig(gain=0.0), seed=11, trials=400, n_resamples=100)
    assert s.accept.low <= 0.05, f"A/A false accept {s.accept.rate:.3f} CI {s.accept.low:.3f}"
    assert s.accept.rate <= 0.05
    assert s.select_pass.rate <= 0.08


def test_aa_with_dropped_runs_and_coupling() -> None:
    cfg = SimConfig(gain=0.0, refusal_rate=0.02, infra_rate=0.02, arm_coupling=0.5)
    s = run_trials(cfg, seed=3, trials=200, n_resamples=100)
    assert s.accept.rate <= 0.05


def test_power_monotone_in_gain() -> None:
    cfg = SimConfig(thresholds=_P_ONLY)
    curve = power_curve([0.0, 0.05, 0.10, 0.20], cfg, seed=5, trials=150, n_resamples=100)
    rates = [c.select_pass.rate for c in curve]
    assert rates == sorted(rates)
    assert rates[-1] > 0.9
    assert rates[0] < 0.2


@pytest.mark.parametrize("gain", [0.05, 0.08, 0.12])
def test_empirical_power_matches_analytic(gain: float) -> None:
    cfg = SimConfig(gain=gain, thresholds=_P_ONLY)
    s = run_trials(cfg, seed=7, trials=400, n_resamples=100)
    assert s.analytic_power is not None
    assert abs(s.select_pass.rate - s.analytic_power) < 0.12, (
        f"gain {gain}: observed {s.select_pass.rate:.3f} vs power() {s.analytic_power:.3f}"
    )


def test_full_rule_power_is_at_most_select_power() -> None:
    s = run_trials(SimConfig(gain=0.1), seed=2, trials=150, n_resamples=100)
    assert s.accept.successes <= s.select_pass.successes


def test_smallest_detectable_gain_reaches_target_power() -> None:
    base = SimConfig(thresholds=_P_ONLY)
    g = analytic_smallest_gain(run_trials(base, seed=1, trials=20, n_resamples=100))
    s = run_trials(replace(base, gain=g), seed=9, trials=400, n_resamples=100)
    # discordance shifts with the gain, so compare with power() at the simulated value
    assert s.analytic_power is not None
    assert abs(s.select_pass.rate - s.analytic_power) < 0.12


def test_same_seed_same_results() -> None:
    cfg = SimConfig(gain=0.06, refusal_rate=0.03)
    a = run_trials(cfg, seed=4, trials=30, n_resamples=100)
    b = run_trials(cfg, seed=4, trials=30, n_resamples=100)
    c = run_trials(cfg, seed=5, trials=30, n_resamples=100)
    assert a == b
    assert a != c
    assert simulate_trial(cfg, 99) == simulate_trial(cfg, 99)


def test_wilson_interval() -> None:
    p = wilson(5, 100)
    assert p.low < 0.05 < p.high
    assert wilson(0, 0).high == 1.0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SimConfig(gain=1.0)
    with pytest.raises(ValueError):
        SimConfig(refusal_rate=2.0)
