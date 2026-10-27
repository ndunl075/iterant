"""Simulator checks: A/A false-accept rate, power vs power(), guards against real harm."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.contracts import AcceptanceThresholds
from iterant.stats.acceptance import Reason
from iterant.stats.simulate import (
    SimConfig,
    analytic_power_at,
    analytic_smallest_gain,
    power_curve,
    run_trials,
    simulate_trial,
    wilson,
)

# p-test only: delta_min ~ 0 so SELECT power is comparable with power().
_P_ONLY = AcceptanceThresholds(delta_min=0.0001)
# The sign-flip test on these discrete scores has a true size of about 3 to 4% at nominal 5%
# (measured A/A SELECT-alone rate, independent of the resample count), so power() is
# compared at this size. Without it the empirical rate sits ~8 pts below power().
_SIZE = 0.035
_R = 100  # resamples: far fewer than production's 10,000, to keep CI fast


def test_aa_false_accept_rate_at_most_2_percent() -> None:
    s = run_trials(SimConfig(), seed=11, trials=300, n_resamples=_R)
    assert s.accept.rate <= 0.02, f"A/A false accept {s.accept.rate:.3f}"
    assert s.select_pass.rate <= 0.06, f"A/A SELECT alone {s.select_pass.rate:.3f}"


def test_aa_with_task_effects_dropped_runs_and_coupling() -> None:
    cfg = SimConfig(task_effect_sd=0.1, refusal_rate=0.02, infra_rate=0.02, arm_coupling=0.5)
    s = run_trials(cfg, seed=3, trials=100, n_resamples=_R)
    assert s.accept.rate <= 0.02


def test_power_monotone_in_gain() -> None:
    cfg = SimConfig(thresholds=_P_ONLY)
    curve = power_curve(
        [0.0, 0.05, 0.10, 0.20], cfg, seed=5, trials=100, n_resamples=_R, select_only=True
    )
    rates = [c.select_pass.rate for c in curve]
    assert rates == sorted(rates)
    assert rates[-1] > 0.9
    assert rates[0] < 0.15


@pytest.mark.parametrize(("gain", "task_sd"), [(0.05, 0.0), (0.10, 0.0), (0.08, 0.15)])
def test_empirical_power_matches_analytic(gain: float, task_sd: float) -> None:
    """With ``task_sd > 0`` the gain is heterogeneous across tasks (mean still ``gain``)."""
    cfg = SimConfig(gain=gain, task_effect_sd=task_sd, thresholds=_P_ONLY)
    s = run_trials(cfg, seed=7, trials=200, n_resamples=_R, select_only=True)
    expected = analytic_power_at(s, _SIZE)
    assert expected is not None
    assert abs(s.select_pass.rate - expected) < 0.07, (
        f"gain {gain}: observed {s.select_pass.rate:.3f} vs power() at size {_SIZE}: {expected:.3f}"
    )
    # Never meaningfully above power() at the nominal 5% size.
    assert s.analytic_power is not None
    assert s.select_pass.rate < s.analytic_power + 0.03


def test_full_rule_power_is_at_most_select_power() -> None:
    s = run_trials(SimConfig(gain=0.1), seed=2, trials=100, n_resamples=_R)
    assert s.accept.successes <= s.select_pass.successes


def test_power_near_target_at_analytic_smallest_gain() -> None:
    """Run at the gain ``smallest_detectable_gain`` gives (at the discordance that gain
    itself produces); empirical power should be near the 0.8 target."""
    base = SimConfig(thresholds=_P_ONLY)
    gain = 0.0
    for i in range(2):  # discordance depends on the gain: iterate to a fixed point
        s = run_trials(
            replace(base, gain=gain), seed=1 + i, trials=20, n_resamples=_R, select_only=True
        )
        gain = analytic_smallest_gain(s)
    s = run_trials(replace(base, gain=gain), seed=9, trials=200, n_resamples=_R, select_only=True)
    assert s.analytic_power is not None
    assert abs(s.analytic_power - 0.8) < 0.03
    expected = analytic_power_at(s, _SIZE)
    assert expected is not None
    assert abs(s.select_pass.rate - expected) < 0.07


def test_candidate_that_refuses_more_is_rejected() -> None:
    good = SimConfig(gain=0.2)
    ok = run_trials(good, seed=4, trials=20, n_resamples=_R, select_only=True)
    assert ok.accept.rate > 0.8  # the gain alone would be accepted
    bad = replace(good, candidate_refusal_rate=0.25)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert Reason.REFUSAL_GUARD in d.reasons


def test_candidate_that_loses_many_tasks_is_rejected() -> None:
    # 40% refusals per attempt leaves ~16% of tasks with no scored run after the retry.
    bad = SimConfig(gain=0.2, candidate_refusal_rate=0.4)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert Reason.TOO_MANY_DROPPED in d.reasons


def test_candidate_with_more_infra_errors_is_rejected() -> None:
    bad = SimConfig(gain=0.2, candidate_infra_rate=0.25)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert Reason.INFRA_GUARD in d.reasons


def test_costlier_candidate_is_rejected() -> None:
    bad = SimConfig(gain=0.2, cost_multiplier=1.3)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert d.reasons == (Reason.COST_GUARD,)
    # Within the +15% allowance the same gain passes.
    near = run_trials(
        SimConfig(gain=0.2, cost_multiplier=1.1),
        seed=4,
        trials=25,
        n_resamples=_R,
        select_only=True,
    )
    assert near.accept.rate > 0.8


def test_slower_candidate_is_rejected() -> None:
    bad = SimConfig(gain=0.2, wall_multiplier=1.4)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert d.reasons == (Reason.WALL_GUARD,)


def test_guards_veto_equal_rate_candidate_often() -> None:
    """Characterises the rule (see PR "Rule observations"): at equal nonzero rates the
    refusal / infra guards fire on sampling noise alone, so a no-worse candidate is
    vetoed a lot."""
    s = run_trials(
        SimConfig(gain=0.12, refusal_rate=0.05, infra_rate=0.05), seed=6, trials=100, n_resamples=_R
    )
    assert s.guard_veto.low > 0.2


def test_same_seed_same_results() -> None:
    cfg = SimConfig(gain=0.06, refusal_rate=0.03, task_effect_sd=0.05)
    a = run_trials(cfg, seed=4, trials=20, n_resamples=_R)
    b = run_trials(cfg, seed=4, trials=20, n_resamples=_R)
    c = run_trials(cfg, seed=5, trials=20, n_resamples=_R)
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
    with pytest.raises(ValueError):
        SimConfig(candidate_infra_rate=-0.1)
    with pytest.raises(ValueError):
        SimConfig(cost_multiplier=0.0)
