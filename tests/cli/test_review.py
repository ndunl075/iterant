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
    assert "listens on 0.0.0.0, i.e. on every network interface" in _err(capsys)


# 7. calibration gates a real run


def test_run_refuses_without_a_passing_aa_test(world, capsys) -> None:
    world.init()
    world.queue.append(None)
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    err = _err(capsys)
    assert "iterant estimate" in err and "iterant aa-test" in err
    assert "--skip-calibration" in err
    assert not world.exec_log


def test_run_allowed_after_aa_test_and_refused_when_the_pack_changes(world, capsys) -> None:
    world.init()
    world.calibrate()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    (world.pack_root / "NOTES.txt").write_text("a new file changes the pack hash")
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    err = _err(capsys)
    assert "pack_hash" in err and "was run for something else" in err


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
    assert "the last `iterant aa-test` failed" in _err(capsys)


def test_aa_test_fails_on_a_degenerate_pack(world, capsys) -> None:
    world.init()
    world.solver = lambda arm, task, attempt: "ok"  # every score is identical
    capsys.readouterr()
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 6
    err = _err(capsys)
    assert "every champion run scored the same" in err
    record = json.loads((world.repo / ".iterant" / "calibration.json").read_text())
    assert record["aa-test"]["state"] == "fail"
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


# review round 3: bind address, fingerprinted records, estimate gate, CONFIRM shape


def test_default_bind_is_a_specific_non_loopback_address_or_a_clear_error() -> None:
    from iterant.cli.errors import CliError
    from iterant.cli.runtime import default_proxy_bind

    try:
        addr = default_proxy_bind("podman")
    except CliError as exc:  # no network route on this machine
        assert "--proxy-bind" in exc.message
    else:
        assert addr not in ("0.0.0.0", "::") and not is_loopback(addr)


def test_docker_default_bind_is_the_bridge_gateway(monkeypatch) -> None:
    import subprocess

    from iterant.cli import runtime
    from iterant.cli.errors import CliError

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        assert argv[:4] == ["docker", "network", "inspect", "bridge"]
        return subprocess.CompletedProcess(argv, 0, stdout="172.17.0.1\n", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    assert runtime.default_proxy_bind("docker") == "172.17.0.1"
    monkeypatch.setattr(
        runtime.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "")
    )
    with pytest.raises(CliError, match="--proxy-bind"):
        runtime.default_proxy_bind("docker")


def test_client_url_follows_the_bind_family() -> None:
    from iterant.cli.runtime import client_url

    assert client_url("0.0.0.0", "80") == "http://127.0.0.1:80"
    assert client_url("::", "80") == "http://[::1]:80"
    assert client_url("10.1.0.65", "80") == "http://10.1.0.65:80"
    assert client_url("fe80::1", "80") == "http://[fe80::1]:80"


def test_default_bind_is_used_and_loopback_gets_a_note(world, monkeypatch, capsys) -> None:
    from iterant.cli import runtime

    assert world.cli("init", str(world.repo), "--genome", "agent/**") == 0
    chosen: list[str] = []

    def pick(backend: str) -> str:
        chosen.append(backend)
        return "127.0.0.1"

    monkeypatch.setattr(runtime, "default_proxy_bind", pick)
    project = load_project(world.repo)
    args = build_parser().parse_args(["run", str(world.repo)])
    rt = build_runtime(settings_from_args(args), project, world.repo / ".iterant" / "archive.db")
    rt.close()
    assert chosen == ["podman"]
    assert "cannot reach a loopback-only proxy" in capsys.readouterr().err


def test_estimate_is_required_too(world, capsys) -> None:
    world.init()
    assert world.cli("aa-test", str(world.repo), *world.SMALL, "--trials", "10") == 0
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    err = _err(capsys)
    assert "no `iterant estimate` has been run" in err
    assert "no `iterant aa-test`" not in err


@pytest.mark.parametrize("flags", [["--delta-min", "0.2"], ["--select-reps", "2"]])
def test_run_refuses_when_the_rule_differs_from_the_calibrated_one(
    world, capsys, flags: list[str]
) -> None:
    world.init()
    world.calibrate()
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP, *flags) == 6
    assert "(thresholds changed)" in _err(capsys)


def test_run_refuses_when_profile_model_or_config_changed(world, capsys) -> None:
    world.init()
    world.calibrate()
    capsys.readouterr()
    toml = world.repo / "iterant.toml"
    text = toml.read_text()
    toml.write_text(text.replace("claude-haiku-5-5", "claude-opus-5-5"))
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    assert "(config_hash, target_model changed)" in _err(capsys)
    toml.write_text(text.replace('genome = ["agent/**"]', 'genome = ["agent/**", "extra.txt"]'))
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    assert "(config_hash changed)" in _err(capsys)
    toml.write_text(text)
    extra = ["--profile", "claude", "--budget-usd", "5"]
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP, *extra) == 6
    assert "(profile changed)" in _err(capsys)


@pytest.mark.parametrize("which", ["aa-test", "estimate"])
def test_a_newer_aborted_attempt_invalidates_an_older_pass(world, capsys, which: str) -> None:
    world.init()
    world.calibrate()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    world.preflight_fails = True  # the newer attempt aborts
    capsys.readouterr()
    assert world.cli(which, str(world.repo), *world.SMALL) == 5
    capsys.readouterr()
    world.preflight_fails = False
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 6
    assert f"the last `iterant {which}` did not finish" in _err(capsys)
    # rerunning it successfully re-enables run
    assert world.cli(which, str(world.repo), *world.SMALL) == 0
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0


def test_estimate_warnings_are_repeated_by_run(world, capsys) -> None:
    world.init()
    world.calibrate()  # the 4-task train sample always warns
    world.queue.append(None)
    capsys.readouterr()
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    assert "`iterant estimate` warned: " in _err(capsys)


def test_confirm_leg_has_the_holdout_shape(monkeypatch) -> None:
    from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split

    pool = [
        RunResult(
            RunKey(1, "champion", f"t{i:02d}", rep, 0),
            Split.SELECT,
            float(rep % 2),
            1,
            1,
            1,
            1.0,
            False,
            False,
        )
        for i in range(30)
        for rep in range(6)
    ]
    t = AcceptanceThresholds(select_tasks=30, select_reps=2, holdout_tasks=12, holdout_reps=3)
    seen: list[tuple[int, int, set[Split]]] = []
    real = calibrate.decide_confirm

    def spy(cand, champ, thresholds, **kw):  # type: ignore[no-untyped-def]
        n = len({r.key.task for r in cand})
        seen.append((n, len(cand) // n, {r.split for r in cand}))
        return real(cand, champ, thresholds, **kw)

    monkeypatch.setattr(calibrate, "decide_confirm", spy)
    calibrate.run_aa(pool, t, trials=3, seed=1, n_resamples=50)
    assert seen == [(12, 3, {Split.HOLDOUT})] * 3  # 12 tasks, 3 reps per arm, holdout-labelled
