"""Runtime model routing (§6): the ``free`` and ``claude`` profiles as proxy policies.

``free`` (default, $0): every role runs on the target's own local model through an
OpenAI-compatible endpoint (Ollama); caps are in tokens. ``claude`` (opt-in, paid by the user):
the §6 table; caps are in USD. Per-run caps come from ``iterant.toml`` for the target and are
fixed here for Iterant's own roles. Per-generation caps are generous upper bounds of what a
generation can legitimately run, so they only stop a runaway.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from iterant.cli.errors import EXIT_USAGE, CliError
from iterant.contracts import (
    AcceptanceThresholds,
    Effort,
    Profile,
    ProxyPolicy,
    Role,
    RoleCap,
    TargetModel,
)
from iterant.proxy import ModelPrice, target_cap

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"

# Micro-USD per million tokens. TODO: the Sonnet price is an assumption; check it against the
# provider's price list (an unlisted model is charged at the dearest listed price, so a wrong
# entry can only over-count). Other target models need ``--price``.
DEFAULT_PRICES: Mapping[str, ModelPrice] = {
    HAIKU: ModelPrice(1_000_000, 5_000_000),
    SONNET: ModelPrice(3_000_000, 15_000_000),
    OPUS: ModelPrice(5_000_000, 25_000_000),
}

# The largest request body each of Iterant's own roles sends, and its max_tokens. The proxy
# reserves ceil(bytes / 3) + max_tokens per call (``estimate_input``), so every per-run cap
# below must fit one such call; tests/cli/test_profiles.py asserts it for both profiles.
# Proposer: MAX_CONTEXT_CHARS (400k) + 20% for JSON escaping, system prompt and tool schemas.
PROPOSER_MAX_REQUEST_BYTES = 480_000
PROPOSER_MAX_TOKENS = 16_000
# Analyst: one 6k-char excerpt plus the system prompt, with escaping headroom.
ANALYST_MAX_REQUEST_BYTES = 16 * 1024
ANALYST_MAX_TOKENS = 1024
# Auditor: the gate refuses diffs over 128 KiB; the request is JSON, which escapes a control
# character to 6 bytes (``\u00XX``), so the worst case is 6x the diff, plus 40 KiB for the
# system prompt and policy note.
AUDITOR_MAX_REQUEST_BYTES = 6 * 128 * 1024 + 40 * 1024
# Per-generation proposer budgets are NOT sized for the worst case of max_turns (30) full
# 400k-char contexts per proposal (about 5.3M tokens, or ~$30 on Opus): a very long session
# can hit the generation cap and is then cut off (the proposer keeps the diff written so far).
AUDITOR_MAX_TOKENS = 1024

PROPOSER_EFFORT: Effort = "high"
ANALYST_EFFORT: Effort = "medium"
AUDITOR_EFFORT: Effort = "medium"


@dataclass(frozen=True)
class RoleModels:
    """Which model and effort each of Iterant's own roles asks for (must equal the cap)."""

    proposer: tuple[str, Effort]
    analyst: tuple[str, Effort]
    auditor: tuple[str, Effort]


def role_models(profile: Profile, target: TargetModel) -> RoleModels:
    if profile is Profile.FREE:
        local = target.name  # one local model for every role
        return RoleModels(
            (local, PROPOSER_EFFORT), (local, ANALYST_EFFORT), (local, AUDITOR_EFFORT)
        )
    return RoleModels((OPUS, PROPOSER_EFFORT), (HAIKU, ANALYST_EFFORT), (SONNET, AUDITOR_EFFORT))


def runs_upper_bound(t: AcceptanceThresholds, candidates: int, extra_runs: int) -> int:
    """More runs than one generation can need: both arms of every stage for every candidate,
    one retry each, plus the digest sample."""
    per_arm = t.smoke_tasks + t.select_tasks * t.select_reps + t.holdout_tasks * t.holdout_reps
    return 2 * (candidates + 1) * per_arm + extra_runs


def parse_price(spec: str) -> tuple[str, ModelPrice]:
    """``MODEL=IN:OUT`` in USD per million tokens, e.g. ``my-model=1.5:7.5``."""
    name, sep, rest = spec.partition("=")
    inp, sep2, out = rest.partition(":")
    if not (name and sep and sep2):
        raise CliError(
            f"--price {spec!r}: expected MODEL=IN:OUT (USD per million tokens)", EXIT_USAGE
        )
    try:
        micros = [int(Decimal(x) * 1_000_000) for x in (inp, out)]
    except (InvalidOperation, ValueError):
        raise CliError(f"--price {spec!r}: IN and OUT must be numbers", EXIT_USAGE) from None
    if min(micros) < 0:
        raise CliError(f"--price {spec!r}: prices must be >= 0", EXIT_USAGE)
    return name, ModelPrice(micros[0], micros[1])


def build_policy(
    profile: Profile,
    target: TargetModel,
    thresholds: AcceptanceThresholds,
    candidates: int,
    *,
    budget_usd_micros: int | None,
    extra_runs: int,
) -> ProxyPolicy:
    n = runs_upper_bound(thresholds, candidates, extra_runs)
    models = role_models(profile, target)
    if profile is Profile.FREE:
        if target.max_tokens_per_run is None:
            raise CliError(
                "the free profile needs target.model.max_tokens_per_run in iterant.toml "
                "(use --profile claude for a USD cap)",
                EXIT_USAGE,
            )
        roles: dict[Role, RoleCap] = {
            Role.TARGET: target_cap(target, profile, target.max_tokens_per_run * n),
            Role.PROPOSER: RoleCap(
                *models.proposer,
                max_tokens_per_run=200_000,
                max_tokens_per_gen=5_000_000 * candidates,
            ),
            Role.ANALYST: RoleCap(
                *models.analyst, max_tokens_per_run=8_000, max_tokens_per_gen=320_000
            ),
            Role.AUDITOR: RoleCap(
                *models.auditor,
                max_tokens_per_run=300_000,
                max_tokens_per_gen=1_280_000 * candidates,
            ),
        }
        return ProxyPolicy(profile, roles)
    per_run = target.max_usd_micros_per_run
    if per_run is None:
        raise CliError(
            "the claude profile needs target.model.max_usd_per_run in iterant.toml "
            "(use the free profile for a token cap)",
            EXIT_USAGE,
        )
    per_gen = per_run * n
    if budget_usd_micros is not None:
        per_gen = min(per_gen, budget_usd_micros)
    if per_gen < per_run:
        raise CliError(
            f"--budget-usd is below one target run (max_usd_per_run = {per_run / 1e6:.6f} USD)",
            EXIT_USAGE,
        )
    return ProxyPolicy(
        profile,
        {
            Role.TARGET: target_cap(target, profile, per_gen),
            Role.PROPOSER: RoleCap(
                *models.proposer,
                max_usd_micros_per_run=1_500_000,
                max_usd_micros_per_gen=20_000_000 * candidates,
            ),
            Role.ANALYST: RoleCap(
                *models.analyst, max_usd_micros_per_run=100_000, max_usd_micros_per_gen=3_000_000
            ),
            Role.AUDITOR: RoleCap(
                *models.auditor,
                max_usd_micros_per_run=900_000,
                max_usd_micros_per_gen=4_000_000 * candidates,
            ),
        },
    )
