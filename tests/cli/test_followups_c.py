"""W3 follow-ups C: aa-test dropped-task warning, status holdout limit, newer-archive message."""

from __future__ import annotations

import sqlite3

from iterant.cli.calibrate import AAResult, dropped_warning, run_aa
from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split

from .test_aa import _pool  # pyright: ignore[reportPrivateUsage]


def test_under_sampled_tasks_are_counted_not_silently_dropped() -> None:
    t = AcceptanceThresholds(select_tasks=10, select_reps=1, holdout_tasks=5, holdout_reps=1)
    pool = _pool(Split.SELECT, 10, 2, 0.5)
    short = RunResult(
        RunKey(1, "champion", "extra", 0, 0), Split.SELECT, 1.0, 1, 1, 1, 1.0, False, False
    )
    res = run_aa(pool + [short], t, trials=3, seed=1, n_resamples=50)
    assert res.select_dropped == 1 and res.confirm_dropped == 1


def test_no_drops_when_every_task_has_enough_runs() -> None:
    t = AcceptanceThresholds(select_tasks=10, select_reps=1, holdout_tasks=5, holdout_reps=1)
    res = run_aa(_pool(Split.SELECT, 10, 2, 0.5), t, trials=3, seed=1, n_resamples=50)
    assert (res.select_dropped, res.confirm_dropped) == (0, 0)


def test_the_dropped_warning_names_the_counts() -> None:
    assert dropped_warning(AAResult(10, 0, 0)) is None
    text = dropped_warning(AAResult(10, 0, 0, 2, 3))
    assert text is not None and "2 select" in text and "3 confirm" in text
    assert text.startswith("WARNING")


def test_status_labels_the_default_holdout_limit(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)
    (world.repo / ".iterant" / "last_run.json").unlink()  # as if no run had recorded it
    capsys.readouterr()
    assert world.cli("status", str(world.repo)) == 0
    out = capsys.readouterr().out
    assert "stops at 60 (default" in out


def test_status_shows_the_limit_the_last_run_used(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL, "--holdout-query-max", "77")
    capsys.readouterr()
    assert world.cli("status", str(world.repo)) == 0
    out = capsys.readouterr().out
    assert "stops at 77 (used by the last run)" in out


def test_a_newer_archive_gets_an_upgrade_message(world, capsys) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)
    capsys.readouterr()
    from iterant.archive import archive_path

    con = sqlite3.connect(archive_path(world.repo))
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    assert world.cli("report", str(world.repo)) == 1
    err = capsys.readouterr().err
    assert "newer Iterant (schema 99)" in err and "upgrade Iterant" in err
    assert "integrity check failed" not in err
