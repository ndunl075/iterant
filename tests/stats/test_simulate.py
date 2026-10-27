"""Simulator checks: A/A false-accept rate, power vs power(), guards against real harm."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.contracts import AcceptanceThresholds
from iterant.stats.acceptance import Reason
from iterant.stats.presets import PRESETS
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
_P_ONLY = AcceptanceThresholds(delta_min=0.0001, select_p=0.05, holdout_p=0.2)
# The sign-flip test on these discrete scores has a true size of about 3 to 4% at nominal 5%
# (measured A/A SELECT-alone rate, independent of the resample count), so power() is
# compared at this size. Without it the empirical rate sits ~8 pts below power().
_SIZE = 0.035
_AA_TRIALS = 400
_POWER_TRIALS = 600  # standard error of a rate is <= 0.02; 200 trials left seeds borderline
_POWER_TOL = 0.08  # a bracket, not a point match: bias down to about -0.06 at task_sd > 0
_R = 100  # resamples: far fewer than production's 10,000, to keep CI fast


BALANCED = PRESETS["balanced"]
STRICT = PRESETS["strict"]


@pytest.mark.parametrize(
    ("name", "preset", "aa_max", "select_max"),
    [("balanced", BALANCED, 0.05, 0.09), ("strict", STRICT, 0.02, 0.06)],
)
def test_aa_false_accept_bounds_per_preset(
    name: str, preset: AcceptanceThresholds, aa_max: float, select_max: float
) -> None:
    """Full rule (SELECT then CONFIRM), identical arms. Fixed seed, CI-sized trial count.
    balanced is the looser preset (measured 2.7% at 1,500 trials); strict keeps the old
    bounds (0.5%, SELECT alone 4.4%)."""
    s = run_trials(SimConfig(thresholds=preset), seed=11, trials=_AA_TRIALS, n_resamples=_R)
    assert s.accept.rate <= aa_max, f"{name}: A/A false accept {s.accept.rate:.3f}"
    assert s.select_pass.rate <= select_max, f"{name}: A/A SELECT alone {s.select_pass.rate:.3f}"


@pytest.mark.parametrize("preset", [BALANCED, STRICT], ids=["balanced", "strict"])
def test_selective_refuser_does_not_raise_the_aa_false_accept(
    preset: AcceptanceThresholds,
) -> None:
    """Identical arms, but the candidate refuses (both attempts) on up to 4 slots per stage
    where it lost to the champion. A refusal scores 0.0, so shedding losses gains nothing."""
    cfg = SimConfig(selective_refusals=4, thresholds=preset)
    s = run_trials(cfg, seed=12, trials=_AA_TRIALS, n_resamples=_R)
    plain = run_trials(SimConfig(thresholds=preset), seed=12, trials=_AA_TRIALS, n_resamples=_R)
    assert s.accept.rate <= 0.05
    assert s.accept.rate <= plain.accept.rate + 0.01


def test_selective_refusals_are_actually_injected() -> None:
    d = simulate_trial(SimConfig(selective_refusals=4, mean_pass=0.5), 3, _R).select
    assert d.n_dropped == 0  # refusals are scored, not dropped


def test_balanced_power_at_plus_8_is_at_least_50_percent_at_default_sizes() -> None:
    s = run_trials(SimConfig(gain=0.08, thresholds=BALANCED), seed=21, trials=200, n_resamples=_R)
    assert s.accept.rate >= 0.5, f"balanced power at +8: {s.accept.rate:.3f}"


def test_aa_with_task_effects_dropped_runs_and_coupling() -> None:
    cfg = SimConfig(task_effect_sd=0.1, refusal_rate=0.02, infra_rate=0.02, arm_coupling=0.5)
    s = run_trials(cfg, seed=3, trials=100, n_resamples=_R)
    assert s.accept.rate <= 0.05


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
    s = run_trials(cfg, seed=7, trials=_POWER_TRIALS, n_resamples=_R, select_only=True)
    expected = analytic_power_at(s, _SIZE)
    assert expected is not None
    assert abs(s.select_pass.rate - expected) < _POWER_TOL, (
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
    s = run_trials(
        replace(base, gain=gain), seed=9, trials=_POWER_TRIALS, n_resamples=_R, select_only=True
    )
    assert s.analytic_power is not None
    assert abs(s.analytic_power - 0.8) < 0.03
    expected = analytic_power_at(s, _SIZE)
    assert expected is not None
    assert abs(s.select_pass.rate - expected) < _POWER_TOL


def test_candidate_that_refuses_more_is_rejected() -> None:
    good = SimConfig(gain=0.2)
    ok = run_trials(good, seed=4, trials=20, n_resamples=_R, select_only=True)
    assert ok.accept.rate > 0.8  # the gain alone would be accepted
    bad = replace(good, candidate_refusal_rate=0.25)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert not d.accepted
        assert Reason.REFUSAL_GUARD in d.reasons


def test_candidate_that_refuses_its_hard_tasks_gains_nothing() -> None:
    """A refusal that survives its retry scores 0.0 for the refuser (never a dropped pair),
    so 40% refusals cannot be used to shed tasks: no trial is accepted."""
    bad = SimConfig(gain=0.2, candidate_refusal_rate=0.4)
    for seed in range(8):
        d = simulate_trial(bad, seed, _R, select_only=True).select
        assert d.n_dropped == 0 and not d.accepted
        assert Reason.TOO_MANY_DROPPED not in d.reasons


def test_candidate_that_loses_many_tasks_to_infra_errors_is_rejected() -> None:
    # 40% infra errors per attempt leaves ~16% of tasks with no scored run after the retry.
    bad = SimConfig(gain=0.2, candidate_infra_rate=0.4)
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


def test_noise_veto_stays_low_at_equal_nonzero_rates() -> None:
    """Replaces the old zero-tolerance behaviour, where equal 5% rates vetoed a good
    candidate over a quarter of the time on sampling noise alone. One guard at a time it
    stays under 10%; with refusal and infra both at 5% the two one-sided tests add up
    (measured ~10.6% at 600 trials), so that case gets a looser bound."""
    for rates, bound in (
        (SimConfig(gain=0.12, refusal_rate=0.05), 0.10),
        (SimConfig(gain=0.12, infra_rate=0.05), 0.10),
        (SimConfig(gain=0.12, refusal_rate=0.05, infra_rate=0.05), 0.16),
    ):
        s = run_trials(rates, seed=6, trials=300, n_resamples=_R)
        assert s.guard_veto.rate <= bound, f"noise veto rate {s.guard_veto.rate:.3f}"


def test_candidate_with_2_to_7_percent_refusals_is_still_mostly_rejected() -> None:
    cfg = SimConfig(gain=0.2, refusal_rate=0.02, candidate_refusal_rate=0.07)
    s = run_trials(cfg, seed=8, trials=100, n_resamples=_R, select_only=True)
    assert s.accept.rate <= 0.4, f"2%->7% refusals accepted {s.accept.rate:.3f} of the time"
    # without the extra refusals the same gain is accepted almost always
    ok = run_trials(
        replace(cfg, candidate_refusal_rate=0.02),
        seed=8,
        trials=100,
        n_resamples=_R,
        select_only=True,
    )
    assert ok.accept.rate >= 0.8


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
