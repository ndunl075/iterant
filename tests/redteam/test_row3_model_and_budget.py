"""Row 3: "Use a bigger model or more budget". Stopped by proxy role caps and the cost guard.

The candidate (or code it ships) tries to buy gains: a bigger model, higher effort, reroutes,
server tools, a request that reserves more than its cap, many calls past the cap, and a run
that outspends the generation budget. Real ``Proxy`` / ``ProxyServer`` and real orchestrator.
"""

from __future__ import annotations

import pytest

from iterant.contracts import Role, RunKey
from iterant.proxy import Admission, Proxy, Rejection, Usage
from tests.redteam.conftest import (
    GEN_CAP,
    HAIKU,
    KEY,
    OPUS,
    RUN_CAP,
    Stack,
    make_proxy,
    msg_body,
)


def rejected(result: Admission | Rejection, code: str) -> None:
    assert isinstance(result, Rejection), f"admitted a request that should fail with {code}"
    assert result.code == code, result


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (msg_body(model=OPUS), "model_not_allowed"),
        (msg_body(model="claude-haiku-5-5-latest"), "model_not_allowed"),
        (msg_body(model="CLAUDE-HAIKU-5-5"), "model_not_allowed"),
        (msg_body(model=" claude-haiku-5-5"), "model_not_allowed"),
        (msg_body(model=["claude-opus-5-5"]), "model_not_allowed"),
        (msg_body(effort="high"), "effort_not_allowed"),
        (msg_body(effort="max"), "effort_not_allowed"),
        (msg_body(fallbacks="default"), "fallback_forbidden"),
        (msg_body(fallbacks=[{"model": OPUS}]), "fallback_forbidden"),
        (msg_body(thinking={"type": "enabled", "budget_tokens": 60_000}), "field_not_allowed"),
        (msg_body(speed="fast"), "field_not_allowed"),
        (msg_body(service_tier="priority"), "field_not_allowed"),
        (msg_body(temperature=1.0), "field_not_allowed"),
        (
            msg_body(mcp_servers=[{"url": "https://attacker.example/mcp", "name": "x"}]),
            "field_not_allowed",
        ),
        (
            msg_body(tools=[{"type": "web_search_20250305", "name": "web_search"}]),
            "tool_not_allowed",
        ),
        (
            {**msg_body(), "output_config": {"effort": "medium", "format": {"type": "x"}}},
            "field_not_allowed",
        ),
    ],
)
def test_row3_a_target_cannot_change_model_effort_or_route(
    proxy: Proxy, body: dict[str, object], code: str
) -> None:
    token = proxy.issue_run_token(Role.TARGET, KEY)
    rejected(proxy.admit(token, __import__("json").dumps(body).encode()), code)


def test_row3_beta_headers_are_refused_for_the_target(proxy: Proxy) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    rejected(
        proxy.admit(token, json.dumps(msg_body()).encode(), "files-api-2025-04-14"),
        "beta_not_allowed",
    )


def test_row3_a_target_token_is_never_worth_the_proposers_model(proxy: Proxy) -> None:
    """Each role is frozen to its own model; swapping tokens or models between roles fails."""
    import json

    target_tok = proxy.issue_run_token(Role.TARGET, KEY)
    proposer_tok = proxy.issue_call_token(Role.PROPOSER)
    rejected(
        proxy.admit(target_tok, json.dumps(msg_body(OPUS, "high")).encode()), "model_not_allowed"
    )
    rejected(
        proxy.admit(proposer_tok, json.dumps(msg_body(HAIKU, "medium")).encode()),
        "model_not_allowed",
    )


def test_row3_the_effort_is_pinned_even_when_the_request_omits_it(stack: Stack) -> None:
    from iterant.proxy import anthropic_message

    stack.fake.add(None, anthropic_message("ok", model=HAIKU))
    token = stack.proxy.issue_run_token(Role.TARGET, KEY)
    status, _ = stack.post(token, msg_body(effort=None))
    assert status == 200
    assert stack.fake.requests[0].body["output_config"] == {"effort": "medium"}


def test_row3_a_huge_max_tokens_cannot_reserve_past_the_cap(proxy: Proxy) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    big = msg_body()
    big["max_tokens"] = 10_000_000  # $50 of output against a $0.10 cap
    result = proxy.admit(token, json.dumps(big).encode())
    assert isinstance(result, Rejection) and result.status == 403, result


@pytest.mark.parametrize("bad", [0, -5, "100", None, True, [100], {"n": 1}])
def test_row3_nonsense_max_tokens_never_admits(proxy: Proxy, bad: object) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    body = msg_body()
    body["max_tokens"] = bad
    result = proxy.admit(token, json.dumps(body).encode())
    assert isinstance(result, Rejection), result


@pytest.mark.xfail(
    strict=True,
    raises=(ValueError, OverflowError),
    reason="HOLE: max_tokens NaN/Infinity makes Proxy.admit raise (num() calls int(nan)) instead "
    "of returning a Rejection; iterant/proxy/jsonutil.py:24 via iterant/proxy/rules.py:176. "
    "Fails closed (no upstream call, no spend) but the server drops the connection with no reply.",
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_row3_non_finite_max_tokens_is_a_rejection_not_a_crash(proxy: Proxy, bad: float) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    body = msg_body()
    body["max_tokens"] = bad
    result = proxy.admit(token, json.dumps(body).encode())
    assert isinstance(result, Rejection), result


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,  # the final cap assertion; admit itself must not raise
    reason="HOLE: the per-run cap reserves only max_tokens of OUTPUT at admit "
    "(iterant/proxy/core.py:180-183); the request's input size is never reserved, so one huge "
    "prompt (up to the 8 MiB body limit, ~2M input tokens) is admitted and then settles far "
    "past the run cap. Concurrent big calls multiply the overshoot.",
)
def test_row3_one_huge_prompt_cannot_blow_through_the_run_cap(proxy: Proxy) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    body = msg_body()
    body["messages"] = [{"role": "user", "content": "x" * 2_000_000}]  # ~500k input tokens
    result = proxy.admit(token, json.dumps(body).encode())
    if isinstance(result, Rejection):
        return  # refused up front: the secure outcome
    # what a real upstream would bill for 2 MB of text (~4 bytes per token): $0.50 > $0.10 cap
    proxy.settle(result, Usage(input=500_000, output=1), False, None)
    assert proxy.run_usage(KEY).usd_micros <= RUN_CAP


def test_row3_many_small_calls_stop_at_the_run_cap(proxy: Proxy) -> None:
    import json

    token = proxy.issue_run_token(Role.TARGET, KEY)
    body = json.dumps(msg_body()).encode()  # reserves 100 output tokens = 500 micro-USD
    admitted = 0
    for _ in range(10_000):
        adm = proxy.admit(token, body)
        if isinstance(adm, Rejection):
            assert adm.code in ("run_cap_exceeded", "cap_exceeded") or adm.status == 403, adm
            break
        proxy.settle(adm, Usage(input=10_000, output=100), False, None)  # 10,500 micro-USD
        admitted += 1
    else:
        pytest.fail("the proxy never refused")
    assert 1 <= admitted < 20
    assert proxy.run_usage(KEY).usd_micros <= RUN_CAP + 10_500  # at most one call past the line


def test_row3_a_new_run_key_cannot_launder_a_runs_spend_past_the_generation_cap(
    proxy: Proxy,
) -> None:
    import json

    body = json.dumps(msg_body()).encode()
    spent = 0
    for task in range(1000):
        key = RunKey(1, "c1", f"t{task}", 0)
        token = proxy.issue_run_token(Role.TARGET, key)
        adm = proxy.admit(token, body)
        if isinstance(adm, Rejection):
            break
        proxy.settle(adm, Usage(input=40_000, output=100), False, None)
        spent += 40_500
    else:
        pytest.fail("the generation cap never bit")
    assert spent <= GEN_CAP + 40_500


def test_row3_a_restarted_proxy_remembers_what_was_spent() -> None:
    """Restarting the process must not hand the candidate a fresh budget."""
    import json

    old, calls = make_proxy()
    token = old.issue_run_token(Role.TARGET, KEY)
    body = json.dumps(msg_body()).encode()
    for usage in [Usage(input=10_000, output=100)] * 9 + [Usage(input=20_000, output=100)]:
        adm = old.admit(token, body)
        assert isinstance(adm, Admission)
        old.settle(adm, usage, False, None)
    assert old.run_usage(KEY).usd_micros > RUN_CAP  # spend is past the cap
    assert isinstance(old.admit(token, body), Rejection)  # control: the old process refuses

    fresh, _ = make_proxy()
    token2 = fresh.issue_run_token(Role.TARGET, KEY)
    assert isinstance(fresh.admit(token2, body), Admission)  # a fresh proxy knows nothing
    fresh2, _ = make_proxy()
    fresh2.preload(calls)  # what the orchestrator does on start, from the archived calls
    token3 = fresh2.issue_run_token(Role.TARGET, KEY)
    rejected(fresh2.admit(token3, body), "cap_exceeded_run")


# --- the budget, end to end ----------


def test_row3_the_generation_budget_stops_the_loop_before_it_is_exceeded(world) -> None:  # type: ignore[no-untyped-def]
    world.queue(world.proposal("v0", "v1", "h"))
    budget = 6 * world.RUN_COST + world.PROPOSER_CAP + 5_000
    res = world.orchestrator(budget_usd_micros=budget).run_generation()
    assert res.stopped == "budget"
    assert res.spent_usd_micros <= budget
    assert res.promoted is None


def test_row3_an_accurate_but_hungrier_candidate_fails_the_cost_guard(world) -> None:  # type: ignore[no-untyped-def]
    """The candidate solves everything, but spends 3x the champion's tokens per run (a bigger
    budget in disguise). The cost guard must reject it."""
    world.usage_scale["b1-0"] = 3
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    [cand] = res.candidates
    assert res.promoted is None
    assert cand.status.value == "rejected"
    assert any("cost" in note for note in cand.notes), cand.notes


def test_row3_a_slower_candidate_fails_the_wall_time_guard(world) -> None:  # type: ignore[no-untyped-def]
    """A candidate that wins by taking 10x the wall time (more compute) is rejected."""
    from iterant.contracts import ExecResult

    def slow(sb, key, result):  # type: ignore[no-untyped-def]
        if sb.image == "img:b1-0":
            return ExecResult(0, "", "", False, 10.0)  # the fake champion arm takes 1.0 s
        return None

    world.after_agent = slow
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    [cand] = res.candidates
    assert res.promoted is None
    assert any("wall" in note for note in cand.notes), cand.notes


def test_row3_a_negative_budget_is_refused(world) -> None:  # type: ignore[no-untyped-def]
    from iterant.orchestrator import OrchestratorConfig

    with pytest.raises(ValueError):
        OrchestratorConfig(budget_usd_micros=-1)
