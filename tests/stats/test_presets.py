"""Presets, the full-rule power model and its agreement with the simulator."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.contracts import AcceptanceThresholds
from iterant.stats import (
    DEFAULT_PRESET,
    PRESETS,
    preset_thresholds,
    rule_power,
    rule_reps_needed,
    rule_tasks_needed,
)
from iterant.stats.simulate import SimConfig, run_trials

BALANCED, STRICT = PRESETS["balanced"], PRESETS["strict"]


def test_preset_values() -> None:
    assert DEFAULT_PRESET == "balanced" and set(PRESETS) == {"balanced", "strict"}
    assert (BALANCED.select_p, BALANCED.holdout_p, BALANCED.holdout_tasks) == (0.10, 0.5, 40)
    assert (STRICT.select_p, STRICT.holdout_p, STRICT.holdout_tasks) == (0.05, 0.2, 80)
    assert BALANCED == AcceptanceThresholds()  # the library default is balanced
    # everything else is as before, and delta_min is kept
    for t in PRESETS.values():
        assert (t.delta_min, t.select_tasks, t.select_reps, t.holdout_reps) == (0.05, 80, 2, 2)
        assert (t.max_cost_increase, t.max_wall_increase) == (0.15, 0.25)


def test_explicit_values_override_the_preset() -> None:
    t = preset_thresholds("strict", holdout_p=0.3, select_tasks=50)
    assert (t.select_p, t.holdout_p, t.holdout_tasks, t.select_tasks) == (0.05, 0.3, 80, 50)
    with pytest.raises(KeyError):
        preset_thresholds("lenient")
    with pytest.raises(ValueError):
        preset_thresholds("balanced", select_p=0.0)


def test_hard_limit_cannot_be_below_the_warning() -> None:
    with pytest.raises(ValueError, match="holdout_query_max"):
        AcceptanceThresholds(holdout_query_warn=30, holdout_query_max=10)


@pytest.mark.parametrize("preset", [BALANCED, STRICT], ids=["balanced", "strict"])
def test_model_aa_is_a_nominal_upper_bound_of_the_simulated_rate(
    preset: AcceptanceThresholds,
) -> None:
    """The closed form assumes an exact test; the sign-flip test on discrete scores is
    conservative, so the model sits above the simulated A/A rate (balanced ~5% vs ~3%)."""
    s = run_trials(SimConfig(thresholds=preset), seed=31, trials=300, n_resamples=100)
    model = rule_power(0.0, s.mean_discordance, preset)
    assert s.accept.rate <= model + 0.01
    assert model - s.accept.rate <= 0.05  # documented bias, not an unbounded one
    assert 0.0 < model <= 0.051


def test_model_power_is_monotone_and_balanced_beats_strict() -> None:
    gains = [0.0, 0.03, 0.05, 0.08, 0.12, 0.2]
    for t in (BALANCED, STRICT):
        ps = [rule_power(g, 0.3, t) for g in gains]
        assert ps == sorted(ps) and ps[-1] > 0.9
    assert all(rule_power(g, 0.3, BALANCED) >= rule_power(g, 0.3, STRICT) for g in gains)
    assert rule_power(0.08, 0.3, BALANCED) >= 0.5  # the default sizes reach 50% at +8


@pytest.mark.parametrize("preset", [BALANCED, STRICT], ids=["balanced", "strict"])
def test_model_overstates_simulated_power_by_a_bounded_amount(
    preset: AcceptanceThresholds,
) -> None:
    """Documented bias: the normal model ignores the sign-flip test's conservatism, so it
    overstates power, by up to ~0.12 at +8 for strict (measured: balanced +6/+3/+0 pts at
    +5/+8/+12, strict +9/+13/+5). It must never understate it by more than noise."""
    for gain in (0.05, 0.08, 0.12):
        s = run_trials(
            SimConfig(gain=gain, thresholds=preset), seed=31, trials=250, n_resamples=100
        )
        diff = rule_power(gain, s.mean_discordance, preset) - s.accept.rate
        assert -0.05 <= diff <= 0.16, (gain, diff)


def test_tasks_and_reps_needed_reach_the_target() -> None:
    small = replace(BALANCED, select_tasks=10, holdout_tasks=6, select_reps=1, holdout_reps=1)
    assert rule_power(0.08, 0.3, small) < 0.3
    n = rule_tasks_needed(0.08, 0.3, small, 0.5)
    assert n is not None and n > 10
    scaled = replace(small, select_tasks=n, holdout_tasks=round(n * 6 / 10))
    assert rule_power(0.08, 0.3, scaled) >= 0.5
    below = replace(small, select_tasks=n - 1, holdout_tasks=round((n - 1) * 6 / 10))
    assert rule_power(0.08, 0.3, below) < 0.5
    r = rule_reps_needed(0.08, 0.3, small, 0.5)
    assert r is not None and r > 1
    assert rule_tasks_needed(0.0, 0.3, small, 0.5) is None  # no gain: never reaches 50%
    assert rule_reps_needed(0.0, 0.3, small, 0.5) is None
