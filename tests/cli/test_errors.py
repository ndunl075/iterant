"""Busy, budget and preflight failures are messages with exit codes, never tracebacks."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from iterant.archive import Archive
from iterant.cli import calibrate
from iterant.cli.main import main
from iterant.contracts import Proposal
from iterant.orchestrator import WriterLock


def _err(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


@pytest.mark.parametrize("command", ["run", "estimate", "aa-test"])
def test_busy_archive_is_exit_3(world, capsys: pytest.CaptureFixture[str], command: str) -> None:
    world.init()
    capsys.readouterr()
    with WriterLock(world.repo):  # another orchestrator holds the archive
        code = world.cli(command, str(world.repo), *world.args(command))
    err = _err(capsys)
    assert code == 3
    assert "another orchestrator is running" in err and "wait for it" in err
    assert not world.exec_log  # nothing ran


@pytest.mark.parametrize("command", ["run", "estimate", "aa-test"])
def test_preflight_failure_is_exit_5(
    world, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    world.init()
    world.preflight_fails = True
    capsys.readouterr()
    assert world.cli(command, str(world.repo), *world.args(command)) == 5
    err = _err(capsys)
    assert "grader preflight failed" in err and "python3 -V" in err
    assert not world.exec_log
    if command == "run":  # preflight comes before any batch
        with Archive.open(world.repo) as db:
            assert db.get_batch(1) is None


def test_runtime_failure_is_exit_5(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    world.create_fails = True  # the sandbox runtime does not answer at all
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 5
    assert "cannot create a sandbox" in _err(capsys)


def test_budget_exhausted_is_exit_4_and_resumable(
    world, capsys: pytest.CaptureFixture[str]
) -> None:
    world.init()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h"))
    capsys.readouterr()
    # one target run may cost 0.1 USD (the cap); half of that cannot start a run
    code = world.cli(
        "run", str(world.repo), *world.RUN_SMALL, "--profile", "claude", "--budget-usd", "0.06"
    )
    out, err = capsys.readouterr()
    assert code == 4
    assert "Traceback" not in err
    assert "budget exhausted" in err and "rerun with a larger --budget-usd" in err
    assert "[stopped: budget]" in out and "spent" in out
    # nothing was rejected for the budget: the batch resumes with more money
    code = world.cli(
        "run", str(world.repo), *world.RUN_SMALL, "--profile", "claude", "--budget-usd", "5"
    )
    assert code == 0
    assert "promoted b1-0" in capsys.readouterr().out


def test_budget_needs_the_claude_profile(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()
    assert world.cli("run", str(world.repo), "--profile", "free", "--budget-usd", "1") == 2
    assert "needs --profile claude" in _err(capsys)


def test_budget_must_be_a_positive_amount(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()
    for bad in ("0", "-1", "abc", "nan"):
        assert world.cli("run", str(world.repo), "--profile", "claude", "--budget-usd", bad) == 2
        assert "--budget-usd" in _err(capsys)


def test_docker_needs_unsafe_container(world, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()
    assert world.cli("run", str(world.repo), "--backend", "docker") == 2
    assert "--unsafe-container" in _err(capsys)


def test_missing_config_and_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["estimate", str(tmp_path)]) == 2
    assert "iterant init" in _err(capsys)
    (tmp_path / "iterant.toml").write_text("[target]\nbogus = 1\n")
    assert main(["run", str(tmp_path)]) == 2
    assert "bogus" in _err(capsys)


def test_missing_pack(world, capsys: pytest.CaptureFixture[str]) -> None:
    assert world.cli("init", str(world.repo), "--genome", "agent/**", "--pack", "nowhere") == 0
    capsys.readouterr()
    assert world.cli("run", str(world.repo)) == 2
    assert "pack" in _err(capsys)


def test_status_without_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(tmp_path)]) == 1
    assert "nothing has run here yet" in _err(capsys)


def test_usage_errors_are_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert main(["frobnicate"]) == 2
    assert main(["--help"]) == 0
    capsys.readouterr()


def test_aa_test_failure_is_exit_6(world, monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    world.init()
    capsys.readouterr()
    monkeypatch.setattr(calibrate, "run_aa", lambda *a, **k: calibrate.AAResult(40, 12, 8))
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "40") == 6
    err = _err(capsys)
    assert "A/A test FAILED" in err and "20.0%" in err and "stats or the pack are broken" in err


def test_report_is_pending_until_the_module_lands(
    world, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world.init()
    world.cli("run", str(world.repo), *world.RUN_SMALL)  # an archive to report on (no candidate)
    capsys.readouterr()
    monkeypatch.setitem(sys.modules, "iterant.report", None)  # as if the module is absent
    assert world.cli("report", str(world.repo)) == 1
    assert "pending W2.11" in _err(capsys)


def test_report_calls_render_with_archive_and_out_dir(
    world, monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)
    capsys.readouterr()
    calls: list[tuple[Path, Path]] = []

    def render(archive_path: Path, out_dir: Path) -> Path:
        calls.append((archive_path, out_dir))
        return out_dir / "index.html"

    fake = types.ModuleType("iterant.report")
    fake.render = render  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iterant.report", fake)
    out = tmp_path / "out"
    assert world.cli("report", str(world.repo), "--out", str(out)) == 0
    assert calls == [(world.repo / ".iterant" / "archive.db", out)]
    assert "index.html" in capsys.readouterr().out


def test_report_without_an_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", str(tmp_path)]) == 1
    assert "no archive" in _err(capsys)
