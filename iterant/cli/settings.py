"""Command-line settings shared by estimate, aa-test and run, and the repo they point at."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from iterant.cli.errors import EXIT_USAGE, CliError
from iterant.cli.profiles import DEFAULT_PRICES, parse_price
from iterant.contracts import AcceptanceThresholds, IterantConfig, Pack, Profile
from iterant.packs import PackError, load_pack
from iterant.proxy import ModelPrice
from iterant.target import ConfigError, load_config

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1"
CONFIG_NAME = "iterant.toml"


@dataclass(frozen=True)
class Settings:
    repo: Path
    profile: Profile = Profile.FREE
    backend: str = "podman"  # "podman" or "docker"
    unsafe_container: bool = False
    openai_base_url: str = DEFAULT_OLLAMA_URL
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    candidates: int = 1
    budget_usd_micros: int | None = None
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    proxy_bind: str = "0.0.0.0"
    proxy_host: str | None = None  # the proxy as the sandbox relay sees it; default per backend
    digest_tasks: int = 10
    seed: int = 0
    send_openai_key: bool = False  # forward OPENAI_API_KEY to a non-loopback endpoint


def add_common_arguments(p: argparse.ArgumentParser) -> None:
    """Options of every command that runs the agent."""
    p.add_argument("repo", nargs="?", default=".", help="target repo (default: current directory)")
    p.add_argument(
        "--profile",
        choices=[x.value for x in Profile],
        default=Profile.FREE.value,
        help="free: local model through Ollama, $0 (default); claude: paid, opt-in",
    )
    p.add_argument("--backend", choices=["podman", "docker"], default="podman")
    p.add_argument(
        "--unsafe-container",
        action="store_true",
        help="allow the docker backend (the daemon is root-equivalent: dev machines only)",
    )
    p.add_argument(
        "--openai-base-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"free profile: OpenAI-compatible endpoint (default {DEFAULT_OLLAMA_URL})",
    )
    p.add_argument("--budget-usd", help="claude profile: stop when this many USD are spent")
    p.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="MODEL=IN:OUT",
        help="claude profile: USD per million tokens for a model not in the built-in table",
    )
    p.add_argument(
        "--proxy-bind",
        default="0.0.0.0",  # noqa: S104 - loopback is unreachable from rootless podman (probed)
        help="address the proxy listens on; sandboxes cannot reach a loopback-only proxy",
    )
    p.add_argument(
        "--proxy-host",
        help="host name sandboxes use to reach the proxy "
        "(default host.containers.internal for podman, host.docker.internal for docker)",
    )
    add_threshold_arguments(p)
    p.add_argument(
        "--send-openai-key",
        action="store_true",
        help="free profile: send OPENAI_API_KEY to a non-loopback --openai-base-url "
        "(by default it is sent to loopback endpoints only)",
    )
    p.add_argument("--seed", type=int, default=0, help="mixed into the bootstrap seeds")


def add_threshold_arguments(p: argparse.ArgumentParser) -> None:
    """The acceptance-rule knobs (§4)."""
    p.add_argument("--select-tasks", type=int, help="SELECT tasks (default 80)")
    p.add_argument("--select-reps", type=int, help="SELECT reps (default 2)")
    p.add_argument("--holdout-tasks", type=int, help="CONFIRM tasks (default 40)")
    p.add_argument("--holdout-reps", type=int, help="CONFIRM reps (default 2)")
    p.add_argument("--smoke-tasks", type=int, help="SMOKE tasks (default 5)")
    p.add_argument("--delta-min", type=float, help="smallest SELECT gain to accept (default 0.05)")


def thresholds_from_args(args: argparse.Namespace) -> AcceptanceThresholds:
    chosen = {
        name: getattr(args, name)
        for name in (
            "select_tasks",
            "select_reps",
            "holdout_tasks",
            "holdout_reps",
            "smoke_tasks",
            "delta_min",
        )
        if getattr(args, name, None) is not None
    }
    try:
        return AcceptanceThresholds(**chosen)
    except ValueError as exc:
        raise CliError(str(exc), EXIT_USAGE) from exc


def usd_to_micros(text: str, flag: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise CliError(f"{flag} {text!r}: expected a number of USD", EXIT_USAGE) from None
    micros = value * 1_000_000
    if not value.is_finite() or micros <= 0 or micros != micros.to_integral_value():
        raise CliError(f"{flag} {text!r}: must be a positive amount in whole micro-USD", EXIT_USAGE)
    return int(micros)


def _digest_tasks(args: argparse.Namespace) -> int:
    n: int = getattr(args, "digest_tasks", 10)
    if n < 1:
        raise CliError("--digest-tasks must be >= 1", EXIT_USAGE)
    return n


def settings_from_args(args: argparse.Namespace) -> Settings:
    profile = Profile(args.profile)
    budget = usd_to_micros(args.budget_usd, "--budget-usd") if args.budget_usd else None
    if budget is None and profile is Profile.CLAUDE:
        raise CliError(
            "the claude profile spends real money: pass --budget-usd to cap it "
            "(estimate, aa-test and run all spend)",
            EXIT_USAGE,
        )
    if budget is not None and profile is not Profile.CLAUDE:
        raise CliError(
            "--budget-usd needs --profile claude (the free profile costs $0)", EXIT_USAGE
        )
    if args.backend == "docker" and not args.unsafe_container:
        raise CliError(
            "the docker backend is not rootless: a container escape is a host root compromise. "
            "Use podman, or pass --unsafe-container to accept that on a dev machine.",
            EXIT_USAGE,
        )
    prices = dict(DEFAULT_PRICES)
    prices.update(parse_price(spec) for spec in args.price)
    return Settings(
        repo=Path(args.repo),
        profile=profile,
        backend=args.backend,
        unsafe_container=args.unsafe_container,
        openai_base_url=args.openai_base_url,
        thresholds=thresholds_from_args(args),
        candidates=getattr(args, "candidates", 1),
        budget_usd_micros=budget,
        prices=prices,
        proxy_bind=args.proxy_bind,
        proxy_host=args.proxy_host,
        digest_tasks=_digest_tasks(args),
        seed=args.seed,
        send_openai_key=args.send_openai_key,
    )


@dataclass(frozen=True)
class Project:
    """What ``iterant.toml`` and its pack say about a repo."""

    repo: Path
    config: IterantConfig
    pack: Pack
    pack_root: Path


def load_project(repo: Path) -> Project:
    if not repo.is_dir():
        raise CliError(f"{repo}: not a directory", EXIT_USAGE)
    toml = repo / CONFIG_NAME
    if not toml.is_file():
        raise CliError(f"{toml} not found; run `iterant init {repo}` first", EXIT_USAGE)
    try:
        config = load_config(toml)
    except ConfigError as exc:
        raise CliError(f"{toml}: {exc}", EXIT_USAGE) from exc
    pack_root = repo / config.evaluation.pack
    try:
        pack = load_pack(pack_root)
    except PackError as exc:
        raise CliError(f"pack {pack_root}: {exc}", EXIT_USAGE) from exc
    return Project(repo, config, pack, pack_root)
