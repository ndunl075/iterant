"""``--preset``, the fingerprint, and the power table in ``estimate``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iterant.cli.calibrate import fingerprint, power_warnings
from iterant.cli.main import build_parser, main
from iterant.cli.settings import Settings, load_project, settings_from_args
from iterant.contracts import RunKey, RunResult, Split
from iterant.stats import PRESETS, Estimate, estimate


def _settings(*argv: str) -> Settings:
    return settings_from_args(build_parser().parse_args(["run", ".", *argv]))


@pytest.mark.parametrize("command", ["estimate", "aa-test", "run"])
def test_every_threshold_command_takes_preset(command: str) -> None:
    ns = build_parser().parse_args([command, ".", "--preset", "strict"])
    assert ns.preset == "strict"
    assert build_parser().parse_args([command, "."]).preset == "balanced"


def test_default_is_balanced_and_strict_applies() -> None:
    s = _settings()
    assert s.preset == "balanced" and s.thresholds == PRESETS["balanced"]
    s = _settings("--preset", "strict")
    assert s.preset == "strict" and s.thresholds == PRESETS["strict"]


def test_explicit_flags_override_the_preset() -> None:
    s = _settings("--preset", "strict", "--holdout-tasks", "30", "--select-p", "0.2")
    t = s.thresholds
    assert (t.holdout_tasks, t.select_p, t.holdout_p) == (30, 0.2, 0.2)
    s = _settings("--holdout-p", "0.4", "--delta-min", "0.03", "--holdout-query-max", "99")
    t = s.thresholds
    assert (t.holdout_p, t.select_p, t.delta_min, t.holdout_query_max) == (0.4, 0.10, 0.03, 99)


def test_unknown_preset_is_a_usage_error() -> None:
    assert main(["estimate", ".", "--preset", "lenient"]) == 2


def test_fingerprint_records_preset_and_effective_thresholds(world) -> None:  # type: ignore[no-untyped-def]
    world.init()
    project = load_project(world.repo)
    a = fingerprint(_settings(), project)
    b = fingerprint(_settings("--preset", "strict"), project)
    assert a["preset"] == "balanced" and b["preset"] == "strict"
    assert a["thresholds"] != b["thresholds"]
    # same preset name, different effective thresholds: still a different fingerprint
    c = fingerprint(_settings("--select-p", "0.2"), project)
    assert c["preset"] == "balanced" and c["thresholds"] != a["thresholds"]


def test_run_refuses_calibration_done_for_another_preset(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.calibrate()  # balanced
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP, "--preset", "strict") == 6
    err = capsys.readouterr().err
    assert "preset" in err and "changed" in err


def test_estimate_prints_power_table_aa_estimate_and_delta_min(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    capsys.readouterr()
    assert world.cli("estimate", str(world.repo), *world.SMALL, "--reps", "2") == 0
    out = capsys.readouterr().out
    assert "under the balanced preset" in out
    for gain in (3, 5, 8, 12):
        assert f"+{gain} pts" in out
    assert "A/A false-accept estimate" in out
    assert "minimum gain worth keeping  5 pts (delta_min)" in out
    assert world.cli("estimate", str(world.repo), *world.SMALL, "--preset", "strict") == 0
    assert "under the strict preset" in capsys.readouterr().out


def _est(power8: float) -> Estimate:
    runs = [
        RunResult(
            RunKey(1, "c", f"t{i}", r), Split.SELECT, float((i + r) % 2), 1, 1, 1, 1.0, False, False
        )
        for i in range(6)
        for r in range(2)
    ]
    return replace(estimate(runs), power_table=((0.08, power8),), discordance=0.3)


def test_power_warning_threshold_is_30_percent() -> None:
    t = PRESETS["balanced"]
    assert power_warnings(_est(0.30), t, "balanced") == []
    assert power_warnings(_est(0.31), t, "balanced") == []
    [w] = power_warnings(_est(0.29), t, "balanced")
    assert "too small" in w and "+8 pt" in w and "29%" in w and "50%" in w


def test_power_warning_says_how_much_more_is_needed_and_points_strict_at_balanced() -> None:
    t = replace(PRESETS["strict"], select_tasks=10, holdout_tasks=10, select_reps=1, holdout_reps=1)
    [w] = power_warnings(_est(0.1), t, "strict")
    assert "under the strict preset" in w and "select tasks (and" in w and "reps per task" in w
    assert "consider --preset balanced" in w
    [w] = power_warnings(_est(0.1), PRESETS["balanced"], "balanced")
    assert "consider --preset balanced" not in w
