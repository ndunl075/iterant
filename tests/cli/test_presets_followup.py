"""Review follow-ups: fingerprint limits, messages, status and report thresholds."""

from __future__ import annotations

import pytest

from iterant.cli.calibrate import fingerprint
from iterant.cli.main import build_parser
from iterant.cli.settings import Settings, load_project, settings_from_args


def _settings(*argv: str) -> Settings:
    return settings_from_args(build_parser().parse_args(["run", ".", *argv]))


def test_holdout_limits_are_not_part_of_the_fingerprint(world) -> None:  # type: ignore[no-untyped-def]
    world.init()
    project = load_project(world.repo)
    a = fingerprint(_settings(), project)
    b = fingerprint(_settings("--holdout-query-max", "99"), project)
    assert a == b and "holdout_query_max" not in a["thresholds"]  # type: ignore[operator]


def test_run_accepts_calibration_when_only_the_holdout_limit_changed(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.calibrate()
    capsys.readouterr()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP, "--holdout-query-max", "99") == 0


def test_a_changed_preset_message_names_the_preset_and_says_balanced_is_looser(  # type: ignore[no-untyped-def]
    world, capsys
) -> None:
    world.init()
    world.calibrate()
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP, "--preset", "strict") == 6
    err = capsys.readouterr().err
    assert "'strict'" in err and "looser than the old default" in err and "--preset strict" in err


def test_the_holdout_stop_message_and_exit(world, capsys, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from iterant.archive import Archive

    world.init()
    world.calibrate()
    capsys.readouterr()
    monkeypatch.setattr(Archive, "holdout_query_count", lambda self: 99)
    code = world.cli("run", str(world.repo), *world.RUN_NOSKIP, "--holdout-query-max", "60")
    err = capsys.readouterr().err
    assert code != 0
    assert "queried 99 times (limit 60)" in err
    assert "Raise --holdout-query-max" in err and "fresh archive" in err
    assert "rerun to resume" not in err


def test_status_shows_the_effective_holdout_limit(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)
    capsys.readouterr()
    assert world.cli("status", str(world.repo), "--holdout-query-max", "77") == 0
    assert "stops at 77" in capsys.readouterr().out


def test_report_thresholds_follow_flags_then_the_calibration_record(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.calibrate()  # records balanced with the small sizes
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_NOSKIP)
    capsys.readouterr()
    page = world.repo / ".iterant" / "report" / "index.html"

    def html(*flags: str) -> str:
        assert world.cli("report", str(world.repo), *flags) == 0
        return page.read_text(encoding="utf-8")

    h0 = html()
    assert "the thresholds of the last calibration record" in h0, h0[
        h0.find("Promoted rows") : h0.find("Promoted rows") + 500
    ]
    h = html("--preset", "strict")
    assert "the strict preset with your flags" in h and "p &lt; 0.05" in h
    assert "the balanced preset with your flags" in html("--select-tasks", "10")
