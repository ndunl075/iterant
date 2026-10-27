"""Follow-ups B: non-finite max_tokens, input-token reservation, bad block types, admit crash."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from iterant.contracts import Profile, Role
from iterant.proxy import Proxy, Rejection
from iterant.proxy.ledger import estimate_input_tokens


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
    assert adm.reservation.units == estimate_input_tokens(len(raw)) * 1 + 100 * 5
    s.proxy.release(adm)


def test_input_reservation_holds_under_concurrency(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(max_tokens=100)
    body["messages"][0]["content"] = "x" * 90_000  # ~30k micros each, cap 100k: at most 3 fit
    raw = _raw(body)
    results: list[Any] = []

    def go() -> None:
        results.append(s.proxy.admit(token, raw))

    threads = [threading.Thread(target=go) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    admitted = [r for r in results if not isinstance(r, Rejection)]
    assert 1 <= len(admitted) <= 3


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
