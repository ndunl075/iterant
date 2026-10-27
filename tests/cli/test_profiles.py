"""Role caps must fit each role's own largest request (the proxy reserves input up front)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from iterant.cli import profiles
from iterant.cli.profiles import DEFAULT_PRICES, build_policy, role_models
from iterant.contracts import AcceptanceThresholds, Profile, Role, TargetModel
from iterant.proxy import InMemorySink, Proxy, ProxyConfig, Rejection
from iterant.proxy.ledger import (
    estimate_tokens,
    input_reserve_usd_micros,
    output_reserve_usd_micros,
)

THRESHOLDS = AcceptanceThresholds(select_tasks=10, holdout_tasks=6, smoke_tasks=2)

CALLS = {
    Role.PROPOSER: (profiles.PROPOSER_MAX_REQUEST_BYTES, profiles.PROPOSER_MAX_TOKENS),
    Role.ANALYST: (profiles.ANALYST_MAX_REQUEST_BYTES, profiles.ANALYST_MAX_TOKENS),
    Role.AUDITOR: (profiles.AUDITOR_MAX_REQUEST_BYTES, profiles.AUDITOR_MAX_TOKENS),
}


def _policy(profile: Profile):
    if profile is Profile.FREE:
        target = TargetModel("local-model", "medium", max_tokens_per_run=1000)
    else:
        target = TargetModel("claude-haiku-5-5", "medium", max_usd_micros_per_run=100_000)
    return target, build_policy(
        profile, target, THRESHOLDS, 2, budget_usd_micros=None, extra_runs=3
    )


@pytest.mark.parametrize("profile", [Profile.FREE, Profile.CLAUDE])
@pytest.mark.parametrize("role", list(CALLS))
def test_per_run_cap_fits_the_roles_largest_call(profile: Profile, role: Role) -> None:
    _, policy = _policy(profile)
    cap = policy.roles[role]
    size, max_tokens = CALLS[role]
    est = estimate_tokens(size)
    if cap.is_token_capped:
        need, limit, gen = est + max_tokens, cap.max_tokens_per_run, cap.max_tokens_per_gen
    else:
        price = DEFAULT_PRICES[cap.model]
        need = input_reserve_usd_micros(price, est) + output_reserve_usd_micros(price, max_tokens)
        limit, gen = cap.max_usd_micros_per_run, cap.max_usd_micros_per_gen
    assert limit is not None and gen is not None
    assert need <= limit <= gen


def test_proposer_limits_match_the_proposer_module() -> None:
    from iterant.proposer.agent import MAX_CONTEXT_CHARS

    assert profiles.PROPOSER_MAX_REQUEST_BYTES >= MAX_CONTEXT_CHARS


@pytest.mark.parametrize("profile", [Profile.FREE, Profile.CLAUDE])
@pytest.mark.parametrize("role", list(CALLS))
def test_a_near_max_request_is_admitted_end_to_end(profile: Profile, role: Role) -> None:
    target, policy = _policy(profile)
    size, max_tokens = CALLS[role]
    model, effort = getattr(role_models(profile, target), role.value)
    kw: dict[str, Any] = (
        {"openai_base_url": "http://127.0.0.1:1/v1"}
        if profile is Profile.FREE
        else {
            "anthropic_base_url": "http://127.0.0.1:1",
            "anthropic_api_key": "k",
            "prices": DEFAULT_PRICES,
        }
    )
    proxy = Proxy(ProxyConfig(policy=policy, sink=InMemorySink(), **kw))
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "output_config": {"effort": effort},
        "messages": [{"role": "user", "content": ""}],
    }
    overhead = len(json.dumps(body).encode())
    body["messages"][0]["content"] = "x" * (size - overhead)
    raw = json.dumps(body).encode()
    assert len(raw) == size
    adm = proxy.admit(proxy.issue_call_token(role), raw)
    assert not isinstance(adm, Rejection), adm
