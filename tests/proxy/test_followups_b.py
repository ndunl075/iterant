"""Follow-ups B: non-finite max_tokens, input-token reservation, bad block types, admit crash."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from iterant.contracts import Profile, Role, RunKey
from iterant.proxy import Proxy, Rejection
from iterant.proxy.ledger import (
    DOCUMENT_TOKENS_FLOOR,
    IMAGE_TOKENS_UPPER,
    estimate_input,
    estimate_tokens,
)


def _raw(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode()  # allow_nan default: emits NaN / Infinity


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_max_tokens_is_a_400_rejection(make_stack, h, bad):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(max_tokens=bad)
    adm = s.proxy.admit(token, _raw(body))
    assert isinstance(adm, Rejection)
    assert adm.status == 400
    status, _ = s.post(token, body)
    assert status == 400


def test_huge_prompt_cannot_blow_the_run_cap(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(max_tokens=100)
    body["messages"][0]["content"] = "x" * 2_000_000  # ~$0.67 at the input price vs a $0.10 cap
    adm = s.proxy.admit(token, _raw(body))
    assert isinstance(adm, Rejection)
    assert adm.code == "cap_exceeded_run"


def test_small_prompt_still_admitted_and_reservation_includes_input(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    raw = _raw(h.msg_body(max_tokens=100))
    adm = s.proxy.admit(token, raw)
    assert not isinstance(adm, Rejection)
    assert adm.reservation.units == estimate_tokens(len(raw)) + 100 * 5
    s.proxy.release(adm)


def test_input_reservation_holds_under_concurrency(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(max_tokens=100)
    body["messages"][0]["content"] = "x" * 90_000  # ~30k micros each, cap 100k: at most 3 fit
    raw = _raw(body)
    results: list[Any] = []
    barrier = threading.Barrier(10)

    def go() -> None:
        barrier.wait()  # release all ten at once
        results.append(s.proxy.admit(token, raw))

    threads = [threading.Thread(target=go) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    admitted = [r for r in results if not isinstance(r, Rejection)]
    assert len(admitted) == 3


def test_free_profile_reserves_input_tokens(make_stack, h):
    s = make_stack(Profile.FREE)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(h.LOCAL, effort=None, max_tokens=10)
    body["messages"][0]["content"] = "x" * 9_000  # 3000 tokens est > 1000 run cap
    adm = s.proxy.admit(token, _raw(body))
    assert isinstance(adm, Rejection)
    assert adm.code == "cap_exceeded_run"


@pytest.mark.parametrize("kind", [["text"], {"a": 1}, None, 5])
def test_unhashable_or_invalid_block_type_is_400(make_stack, h, kind):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body()
    body["messages"][0]["content"] = [{"type": kind}]
    adm = s.proxy.admit(token, _raw(body))
    assert isinstance(adm, Rejection)
    assert adm.status == 400


def test_unexpected_admit_exception_is_a_clean_500(make_stack, h, monkeypatch):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(Proxy, "admit", boom)
    status, body = s.post(token, h.msg_body())
    assert status == 500
    assert b"internal error" in body


def test_deeply_nested_json_is_a_400_not_a_crash(make_stack, h) -> None:
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    adm = s.proxy.admit(token, b"[" * 100_000)
    assert isinstance(adm, Rejection) and adm.status == 400


def test_image_blocks_reserve_a_per_image_upper_bound() -> None:
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    body = {"messages": [{"role": "user", "content": [img, img, {"type": "text", "text": "x"}]}]}
    est = estimate_input(100, body)
    assert est.tokens == estimate_tokens(100) + 2 * IMAGE_TOKENS_UPPER


def test_document_blocks_reserve_their_base64_length() -> None:
    doc = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "A" * 900},
    }
    est = estimate_input(1000, {"messages": [{"role": "user", "content": [doc]}]})
    assert est.tokens == estimate_tokens(1000) + DOCUMENT_TOKENS_FLOOR + 900


def test_cache_control_is_priced_at_the_write_rate(make_stack, h) -> None:
    s = make_stack()
    plain = h.msg_body(max_tokens=100)
    plain["messages"][0]["content"] = "x" * 30_000
    cached = h.msg_body(max_tokens=100)
    cached["messages"][0]["content"] = [
        {"type": "text", "text": "x" * 30_000, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    a = s.proxy.admit(s.proxy.issue_run_token(Role.TARGET, h.KEY), _raw(plain))
    b = s.proxy.admit(s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t2", 0)), _raw(cached))
    assert not isinstance(a, Rejection) and not isinstance(b, Rejection)
    assert b.reservation.units - 500 >= 2 * (a.reservation.units - 500) * 0.95


def test_in_flight_calls_per_token_are_limited(make_stack, h) -> None:
    s = make_stack(max_inflight_per_token=2)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    raw = _raw(h.msg_body())
    first, second = s.proxy.admit(token, raw), s.proxy.admit(token, raw)
    assert not isinstance(first, Rejection) and not isinstance(second, Rejection)
    third = s.proxy.admit(token, raw)
    assert isinstance(third, Rejection) and third.status == 429
    s.proxy.release(first)  # a finished call frees a slot
    assert not isinstance(s.proxy.admit(token, raw), Rejection)
