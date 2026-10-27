"""The real wiring (``build_runtime``), profiles, prices and the digest's redaction. No container
runtime is started and no model is called: the podman backend only runs commands when a sandbox
is created."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from iterant.analyst import FailureAnalyst
from iterant.cli import run as run_module
from iterant.cli.errors import CliError
from iterant.cli.images import _apply  # pyright: ignore[reportPrivateUsage]
from iterant.cli.main import build_parser, main
from iterant.cli.profiles import build_policy, parse_price
from iterant.cli.runtime import build_runtime
from iterant.cli.settings import load_project, settings_from_args
from iterant.contracts import (
    AcceptanceThresholds,
    Profile,
    Proposal,
    Role,
    TargetModel,
)
from iterant.gate import DiffGate
from iterant.grader import Grader
from iterant.proposer import EditingProposer
from iterant.sandbox import PodmanBackend
from iterant.target import ImageBuildError


@pytest.fixture(autouse=True)
def _loopback_default_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    from iterant.cli import runtime

    monkeypatch.setattr(runtime, "default_proxy_bind", lambda backend: "127.0.0.1")


def _settings(world, *argv: str):  # type: ignore[no-untyped-def]
    args = build_parser().parse_args(["run", str(world.repo), *argv])
    return settings_from_args(args)


def test_free_profile_wiring(world) -> None:
    assert world.cli("init", str(world.repo), "--genome", "agent/**", "--model", "tiny:1b") == 0
    project = load_project(world.repo)
    settings = _settings(world)
    rt = build_runtime(settings, project, world.repo / ".iterant" / "archive.db")
    try:
        assert isinstance(rt.backend, PodmanBackend)
        assert isinstance(rt.grader, Grader) and isinstance(rt.proposer, EditingProposer)
        assert isinstance(rt.gate, DiffGate)
        policy = rt.proxy.config.policy
        assert policy.profile is Profile.FREE
        # one local model for every role, frozen by the proxy
        assert {c.model for c in policy.roles.values()} == {"tiny:1b"}
        assert set(policy.roles) == {Role.TARGET, Role.PROPOSER, Role.ANALYST, Role.AUDITOR}
        assert rt.proxy.config.openai_base_url == "http://127.0.0.1:11434/v1"
        assert rt.proxy.config.anthropic_api_key is None
        # the proxy's HTTP server is up and answers only to tokens
        base = rt.analyst_client  # a client over the proxy server
        assert base is not None
        with pytest.raises(Exception, match="401"):
            urllib.request.urlopen(  # noqa: S310
                urllib.request.Request(
                    f"{_proxy_url(rt)}/v1/messages",
                    data=b"{}",
                    headers={"x-api-key": "nope"},
                    method="POST",
                )
            )
    finally:
        rt.close()


def _proxy_url(rt: Any) -> str:
    return rt.analyst_client._base  # pyright: ignore[reportPrivateUsage]


def test_claude_profile_needs_a_key_and_is_opt_in(world, monkeypatch: pytest.MonkeyPatch) -> None:
    world.init()
    project = load_project(world.repo)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = _settings(world, "--profile", "claude", "--budget-usd", "5")
    with pytest.raises(CliError, match="ANTHROPIC_API_KEY"):
        build_runtime(settings, project, world.repo / ".iterant" / "archive.db")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    rt = build_runtime(settings, project, world.repo / ".iterant" / "archive.db")
    try:
        policy = rt.proxy.config.policy
        assert policy.profile is Profile.CLAUDE
        assert policy.roles[Role.PROPOSER].model == "claude-opus-5-5"
        assert policy.roles[Role.ANALYST].model == "claude-haiku-5-5"
        assert policy.roles[Role.AUDITOR].model == "claude-sonnet-5-5"
        assert policy.roles[Role.TARGET].max_usd_micros_per_run == 100_000
    finally:
        rt.close()


def test_claude_profile_target_model_needs_a_price(world, monkeypatch: pytest.MonkeyPatch) -> None:
    world.cli(
        "init",
        str(world.repo),
        "--profile",
        "claude",
        "--model",
        "my-model",
        "--genome",
        "agent/**",
    )
    project = load_project(world.repo)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    db = world.repo / ".iterant" / "archive.db"
    with pytest.raises(CliError, match="--price MODEL=IN:OUT"):
        build_runtime(_settings(world, "--profile", "claude", "--budget-usd", "5"), project, db)
    settings = _settings(
        world, "--profile", "claude", "--budget-usd", "5", "--price", "my-model=2:10"
    )
    build_runtime(settings, project, db).close()


def test_policy_caps() -> None:
    t = AcceptanceThresholds(select_tasks=10, holdout_tasks=6, smoke_tasks=2)
    model = TargetModel("claude-haiku-5-5", "medium", max_usd_micros_per_run=100_000)
    policy = build_policy(Profile.CLAUDE, model, t, 1, budget_usd_micros=2_000_000, extra_runs=3)
    cap = policy.roles[Role.TARGET]
    assert (cap.max_usd_micros_per_run, cap.max_usd_micros_per_gen) == (100_000, 2_000_000)
    with pytest.raises(CliError, match="below one target run"):
        build_policy(Profile.CLAUDE, model, t, 1, budget_usd_micros=50_000, extra_runs=3)
    with pytest.raises(CliError, match="max_tokens_per_run"):
        build_policy(Profile.FREE, model, t, 1, budget_usd_micros=None, extra_runs=3)
    free = TargetModel("m", "medium", max_tokens_per_run=1000)
    with pytest.raises(CliError, match="max_usd_per_run"):
        build_policy(Profile.CLAUDE, free, t, 1, budget_usd_micros=None, extra_runs=3)


def test_parse_price() -> None:
    name, price = parse_price("m=1.5:7.5")
    assert (name, price.input_per_mtok, price.output_per_mtok) == ("m", 1_500_000, 7_500_000)
    for bad in ("m", "m=1", "m=a:b", "=1:2", "m=-1:2"):
        with pytest.raises(CliError):
            parse_price(bad)


def test_apply_diff_to_a_plain_tree(tmp_path: Path) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt.txt").write_bytes(b"v0\n")
    d = "diff --git a/agent/prompt.txt b/agent/prompt.txt\n--- a/agent/prompt.txt\n"
    d += "+++ b/agent/prompt.txt\n@@ -1 +1 @@\n-v0\n+v1\n"
    _apply(d, tmp_path)
    assert (tmp_path / "agent" / "prompt.txt").read_bytes() == b"v1\n"
    with pytest.raises(ImageBuildError, match="does not apply"):
        _apply(d, tmp_path)  # already applied


def test_digest_gets_the_hidden_check_text_to_redact(
    world, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []
    real = FailureAnalyst

    def recording(*args: Any, **kwargs: Any) -> FailureAnalyst:
        seen.append(list(kwargs["redact"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_module, "FailureAnalyst", recording)
    world.init()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h"))
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) == 0
    assert len(seen) == 1
    redact = seen[0]
    assert '{"v": 1}' in redact  # the hidden check's text
    assert "tasks/tr0/check" in redact  # and its path, pack-relative and repo-relative
    assert "evals/pack/tasks/tr0/check" in redact


def test_cli_namespace_defaults() -> None:
    ns: argparse.Namespace = build_parser().parse_args(["estimate"])
    assert (ns.profile, ns.backend, ns.reps, ns.split) == ("free", "podman", 2, "train")
    assert main(["status", "--help"]) == 0
