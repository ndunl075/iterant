"""Unit tests: ledger, sessions, config, fake LLM."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from iterant.contracts import Profile, ProxyPolicy, Role, RoleCap, RunKey, TargetModel
from iterant.proxy import (
    FakeLLM,
    InMemorySink,
    ModelPrice,
    ProxyConfig,
    anthropic_message,
    target_cap,
)
from iterant.proxy.fake import request_key
from iterant.proxy.ledger import CapExceeded, Ledger, Usage, cost_usd_micros
from iterant.proxy.sessions import SessionRegistry

CAP = RoleCap("m", "low", max_usd_micros_per_run=100, max_usd_micros_per_gen=150)


def test_ledger_reservations_stop_concurrent_overspend():
    ledger = Ledger()
    key = RunKey(1, "c", "t", 0)
    a = ledger.reserve(Role.TARGET, key, CAP, 60)
    assert not isinstance(a, CapExceeded)
    b = ledger.reserve(Role.TARGET, key, CAP, 60)  # 60 in flight + 60 > 100
    assert isinstance(b, CapExceeded) and b.scope == "run"
    ledger.release(a)
    assert not isinstance(ledger.reserve(Role.TARGET, key, CAP, 60), CapExceeded)


def test_ledger_threads_never_exceed_gen_cap():
    ledger = Ledger()
    granted: list[object] = []

    def worker(i: int) -> None:
        r = ledger.reserve(Role.TARGET, RunKey(1, "c", "t", i), CAP, 40)
        if not isinstance(r, CapExceeded):
            granted.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 3  # 3 * 40 <= 150 < 4 * 40


def test_cost_rounds_up_once():
    p = ModelPrice(1_000_000, 5_000_000)
    assert cost_usd_micros(p, Usage(input=1, output=1)) == 6
    assert cost_usd_micros(ModelPrice(1, 1), Usage(input=1)) == 1  # never rounds to free


def test_sessions_expiry_uses_and_revoke():
    now = [0.0]
    reg = SessionRegistry(lambda: now[0], 10)
    key = RunKey(1, "c", "t", 0)
    run = reg.issue_run_token(Role.TARGET, key)
    call = reg.issue_call_token(Role.PROPOSER, uses=2)
    assert reg.resolve(run) and reg.resolve(call)
    assert reg.take_use(call) and reg.take_use(call) and not reg.take_use(call)
    assert reg.resolve(call) is None
    now[0] = 10.0
    assert reg.resolve(run) is None  # expired
    assert reg.resolve("itr_forged") is None and reg.resolve("") is None
    with pytest.raises(ValueError):
        reg.issue_call_token(Role.TARGET)
    with pytest.raises(ValueError):
        reg.issue_run_token(Role.AUDITOR, key)


def test_tokens_are_unique_and_unguessable():
    reg = SessionRegistry(lambda: 0.0, 10)
    toks = {reg.issue_call_token(Role.PROPOSER) for _ in range(100)}
    assert len(toks) == 100 and all(len(t) > 40 for t in toks)


def test_config_validation():
    sink = InMemorySink()
    claude = ProxyPolicy(Profile.CLAUDE, {Role.TARGET: CAP})
    with pytest.raises(ValueError, match="anthropic"):
        ProxyConfig(policy=claude, sink=sink, prices={"m": ModelPrice(1, 1)})
    with pytest.raises(ValueError, match="price"):
        ProxyConfig(policy=claude, sink=sink, anthropic_base_url="http://x", anthropic_api_key="k")
    with pytest.raises(ValueError, match="openai"):
        ProxyConfig(policy=ProxyPolicy(Profile.FREE), sink=sink)


def test_target_cap_from_target_model():
    tm = TargetModel("claude-haiku-5-5", "medium", max_usd_micros_per_run=100_000)
    cap = target_cap(tm, Profile.CLAUDE, 1_000_000)
    assert (cap.model, cap.effort) == ("claude-haiku-5-5", "medium")
    assert cap.max_usd_micros_per_run == 100_000
    with pytest.raises(ValueError):
        target_cap(tm, Profile.FREE, 1_000)


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_fake_matching_times_and_unmatched():
    fake = FakeLLM()
    fake.add({"messages": [{"role": "user", "content": "a"}]}, anthropic_message("A"), times=1)
    fake.add(lambda b: b.get("model") == "x", anthropic_message("X"))
    with fake:
        url = fake.base_url + "/v1/messages"
        a = {"model": "m", "messages": [{"role": "user", "content": "a"}]}
        assert _post(url, a)[1]["content"][0]["text"] == "A"
        assert _post(url, a)[0] == 404  # expired and model != x
        assert _post(url, {"model": "x"})[1]["content"][0]["text"] == "X"
    assert len(fake.requests) == 3 and len(fake.unmatched) == 1


def test_fake_from_file_and_exact_key(tmp_path):
    exact = {"model": "m", "messages": []}
    f = tmp_path / "fx.json"
    f.write_text(
        json.dumps(
            {
                "fixtures": [
                    {"key": request_key(exact), "response": anthropic_message("exact")},
                    {"match": {"model": "m"}, "response": anthropic_message("loose")},
                ]
            }
        )
    )
    with FakeLLM.from_file(f) as fake:
        url = fake.base_url + "/v1/messages"
        assert _post(url, exact)[1]["content"][0]["text"] == "exact"
        assert _post(url, {"model": "m", "x": 1})[1]["content"][0]["text"] == "loose"
