"""Fixes from the PR #17 review: budgets, holdout, resume digests, secrets, calibration, locks."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from iterant.archive import Archive
from iterant.cli import calibrate
from iterant.cli import run as run_module
from iterant.cli.lock import cli_lock
from iterant.cli.main import build_parser
from iterant.cli.runtime import build_runtime, is_loopback, openai_key_for
from iterant.cli.settings import load_project, settings_from_args
from iterant.contracts import Proposal
from iterant.orchestrator import OrchestratorBusy

REAL_KEY = "sk-ant-REAL-KEY"  # the proxy key the fake world holds


def _err(capsys: pytest.CaptureFixture[str]) -> str:
    err = capsys.readouterr().err
    assert "Traceback" not in err
    return err


# 1. the claude profile needs a budget


@pytest.mark.parametrize("command", ["estimate", "aa-test", "run"])
def test_claude_profile_requires_a_budget(world, capsys, command: str) -> None:
    world.init()
    capsys.readouterr()
    assert world.cli(command, str(world.repo), "--profile", "claude") == 2
    assert "--budget-usd" in _err(capsys)
    assert not world.exec_log and not world.cmds


# 2. holdout and test are never run for calibration


@pytest.mark.parametrize("split", ["holdout", "test"])
def test_estimate_split_is_train_or_select_only(world, capsys, split: str) -> None:
    world.init()
    capsys.readouterr()
    assert world.cli("estimate", str(world.repo), *world.SMALL, "--split", split) == 2
    assert not world.exec_log


def test_estimate_can_sample_select(world, capsys) -> None:
    world.init()
    assert world.cli("estimate", str(world.repo), *world.SMALL, "--split", "select") == 0
    assert world.exec_log and all(t.startswith("se") for t in world.exec_log)


def test_aa_test_runs_select_tasks_only(world, capsys) -> None:
    world.init()
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 0
    assert world.exec_log and all(t.startswith("se") for t in world.exec_log)
    with Archive.open_path(world.repo / ".iterant" / "calibration.db") as db:
        assert db.holdout_query_count() == 0


# 3. a resumed batch has no transcripts for the slots a dead process finished


def test_digest_after_resume_warns_and_omits_missing_transcripts(world, capsys) -> None:
    world.init()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h"))
    capsys.readouterr()
    claude = ["--profile", "claude"]
    # enough budget for exactly one digest run, then the budget stops the generation
    assert (
        world.cli("run", str(world.repo), *world.RUN_SMALL, *claude, "--budget-usd", "0.104") == 4
    )
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_SMALL, *claude, "--budget-usd", "5") == 0
    err = _err(capsys)
    assert "1 of 3 digest runs were finished by an earlier process" in err
    digest = world.proposer.digests[0]
    assert "1 sampled train run(s) from an earlier process have no transcript" in digest
    assert "Train runs: 2" in digest  # only the two with transcripts


# 4. --digest-tasks


@pytest.mark.parametrize("n", ["0", "-1"])
def test_digest_tasks_must_be_positive(world, capsys, n: str) -> None:
    world.init()
    capsys.readouterr()
    assert world.cli("run", str(world.repo), "--skip-calibration", "--digest-tasks", n) == 2
    assert "--digest-tasks must be >= 1" in _err(capsys)


# 5. OPENAI_API_KEY and the proxy bind address


def test_loopback_detection() -> None:
    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.1.2.3"):
        assert is_loopback(host)
    for host in ("0.0.0.0", "10.0.0.5", "example.com", "", None):
        assert not is_loopback(host)


def test_openai_key_goes_to_loopback_only(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    assert openai_key_for("http://127.0.0.1:11434/v1", False) == "sk-openai-test"
    assert openai_key_for("http://localhost:11434/v1", False) == "sk-openai-test"
    assert openai_key_for("https://api.example.com/v1", False) is None
    assert "not sent" in _err(capsys)
    assert openai_key_for("https://api.example.com/v1", True) == "sk-openai-test"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert openai_key_for("https://api.example.com/v1", False) is None


def test_non_loopback_proxy_bind_warns(world, capsys) -> None:
    assert world.cli("init", str(world.repo), "--genome", "agent/**") == 0  # free profile
    project = load_project(world.repo)
    args = build_parser().parse_args(["run", str(world.repo), "--proxy-bind", "0.0.0.0"])
    rt = build_runtime(settings_from_args(args), project, world.repo / ".iterant" / "archive.db")
    rt.close()
    assert "proxy listens on 0.0.0.0 (not loopback)" in _err(capsys)


# 7. calibration gates a real run


def test_run_refuses_without_a_passing_aa_test(world, capsys) -> None:
    world.init()
    world.queue.append(None)
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    err = _err(capsys)
    assert "iterant aa-test" in err and "--skip-calibration" in err
    assert not world.exec_log


def test_run_allowed_after_aa_test_and_refused_when_the_pack_changes(world, capsys) -> None:
    world.init()
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 0
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    (world.pack_root / "NOTES.txt").write_text("a new file changes the pack hash")
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    assert "pack changed since the last A/A test" in _err(capsys)


def test_skip_calibration_warns(world, capsys) -> None:
    world.init()
    world.queue.append(None)
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 0
    assert "running without calibration" in _err(capsys)


def test_a_failed_aa_test_blocks_run(world, monkeypatch, capsys) -> None:
    world.init()
    monkeypatch.setattr(calibrate, "run_aa", lambda *a, **k: calibrate.AAResult(40, 12, 8))
    assert world.cli("aa-test", str(world.repo), *world.SMALL) == 6
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    assert "last A/A test failed" in _err(capsys)


def test_aa_test_fails_on_a_degenerate_pack(world, capsys) -> None:
    world.init()
    world.solver = lambda arm, task, attempt: "ok"  # every score is identical
    capsys.readouterr()
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 6
    err = _err(capsys)
    assert "every champion run scored the same" in err
    record = json.loads((world.repo / ".iterant" / "calibration.json").read_text())
    assert record["passed"] is False
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6  # and it blocks a run


# 8. invariant 7 before every calibration batch


@pytest.mark.parametrize("command", ["estimate", "aa-test"])
def test_calibration_verifies_the_pack_hash(world, monkeypatch, capsys, command: str) -> None:
    world.init()
    real_load = calibrate.load_project

    def load_then_tamper(repo: Path):  # type: ignore[no-untyped-def]
        project = real_load(repo)
        (project.pack_root / "tasks" / "tr0" / "check").write_bytes(b'{"v": 2}')
        return project

    monkeypatch.setattr(calibrate, "load_project", load_then_tamper)
    capsys.readouterr()
    assert world.cli(command, str(world.repo), *world.SMALL) == 1
    assert "PackHashMismatch" in _err(capsys)
    assert not world.exec_log


# 11. one lock across the whole run loop


def test_cli_lock_is_held_between_generations(world, monkeypatch) -> None:
    world.init()
    world.queue += [Proposal(world.diff("v0", "v1"), "h1"), Proposal(world.diff("v1", "v2"), "h2")]
    real = run_module.Orchestrator
    probes: list[str] = []

    def probing(*a, **k):  # type: ignore[no-untyped-def]
        try:
            cli_lock(world.repo).acquire()
            probes.append("free")
        except OrchestratorBusy:
            probes.append("held")
        return real(*a, **k)

    monkeypatch.setattr(run_module, "Orchestrator", probing)
    assert world.cli("run", str(world.repo), *world.RUN_SMALL, "--generations", "2") == 0
    assert probes == ["held", "held"]  # before generation 1 and between 1 and 2


def test_a_status_probe_does_not_block_a_starting_run(world) -> None:
    world.init()
    world.queue.append(None)
    lock = cli_lock(world.repo)
    lock.acquire()  # what status holds for an instant
    threading.Timer(0.3, lock.release).start()
    t0 = time.monotonic()
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 0
    assert time.monotonic() - t0 >= 0.2


def test_status_reports_a_running_run(world, capsys) -> None:
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)
    capsys.readouterr()
    with cli_lock(world.repo):
        assert world.cli("status", str(world.repo)) == 0
    assert "a run is in progress" in capsys.readouterr().out
    assert world.cli("status", str(world.repo)) == 0
    assert "a run is in progress" not in capsys.readouterr().out


# 12. secrets stay out of everything; calibration leaves the real archive alone


def test_no_secret_in_output_archives_or_sandbox_commands(world, capsys) -> None:
    world.init()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h"))
    for cmd in ("estimate", "aa-test", "run", "status"):
        extra = ["--trials", "10"] if cmd == "aa-test" else []
        flags = [] if cmd == "status" else world.args(cmd)
        assert world.cli(cmd, str(world.repo), *flags, *extra) == 0
    out = capsys.readouterr()
    assert REAL_KEY not in out.out + out.err
    for db in (".iterant/archive.db", ".iterant/calibration.db"):
        assert REAL_KEY.encode() not in (world.repo / db).read_bytes()
    assert world.cmds
    assert all(REAL_KEY not in c for c in world.cmds)
    assert any("ANTHROPIC_API_KEY='itr_" in c for c in world.cmds)  # a session token only


def test_calibration_leaves_the_real_archive_byte_identical(world) -> None:
    world.init()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 0
    files = [
        world.repo / ".iterant" / n for n in ("archive.db", "archive.db-wal", "archive.db-shm")
    ]
    before = {f.name: f.read_bytes() for f in files if f.exists()}
    assert before
    assert world.cli("estimate", str(world.repo), *world.SMALL) == 0
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 0
    after = {f.name: f.read_bytes() for f in files if f.exists()}
    assert after == before
