"""End to end: sandbox-style client -> proxy -> fake Anthropic upstream."""

from __future__ import annotations

import json

import anthropic
import pytest

from iterant.contracts import Role, RunKey, Split
from iterant.proxy import anthropic_message


def sdk(stack, token):
    return anthropic.Anthropic(base_url=stack.server.base_url, api_key=token, max_retries=0)


def test_nonstream_passthrough_logs_call_and_hides_key(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message("hello", model=h.HAIKU, in_tok=10, out_tok=5))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body())
    assert status == 200
    assert json.loads(raw)["content"][0]["text"] == "hello"
    seen = s.fake.requests[0]
    assert seen.headers["x-api-key"] == h.REAL_KEY  # the real key goes out...
    assert token not in json.dumps(dict(seen.headers))  # ...the session token never does
    assert h.REAL_KEY.encode() not in raw
    (call,) = s.sink.calls
    assert (call.key, call.role, call.model) == (h.KEY, Role.TARGET, h.HAIKU)
    assert (call.in_tok, call.out_tok, call.refusal) == (10, 5, False)
    assert call.usd_micros == 35  # 10 * $1/M + 5 * $5/M
    assert call.created_at.endswith("Z")
    usage = s.proxy.run_usage(h.KEY)
    assert (usage.calls, usage.in_tok, usage.out_tok, usage.usd_micros) == (1, 10, 5, 35)


def test_effort_is_pinned_when_absent_and_body_otherwise_untouched(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.HAIKU))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(effort=None, system="be brief", metadata={"user_id": "u"})
    assert s.post(token, body)[0] == 200
    forwarded = s.fake.requests[0].body
    assert forwarded["output_config"] == {"effort": "medium"}
    assert {k: v for k, v in forwarded.items() if k != "output_config"} == body


@pytest.mark.parametrize(
    ("model", "effort", "code"),
    [
        ("claude-opus-5-5", "medium", "model_not_allowed"),
        ("claude-haiku-5-5", "high", "effort_not_allowed"),
    ],
)
def test_frozen_model_and_effort(make_stack, h, model, effort, code):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body(model=model, effort=effort))
    assert status == 403
    assert json.loads(raw)["error"]["iterant_code"] == code
    assert s.fake.requests == [] and s.sink.calls == []
    assert s.proxy.rejections[-1].code == code


def test_unknown_missing_and_expired_tokens_rejected(make_stack, h):
    s = make_stack()
    expired = s.proxy.issue_run_token(Role.TARGET, h.KEY, ttl_s=0)
    for tok in ("itr_nope", None, expired, h.REAL_KEY):
        status, raw = s.post(tok, h.msg_body())
        assert status == 401, tok
        assert json.loads(raw)["error"]["iterant_code"] == "unknown_token"
    assert s.fake.requests == []


def test_revoked_run_token(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    s.proxy.sessions.revoke_run(h.KEY)
    assert s.post(token, h.msg_body())[0] == 401


def test_upstream_error_is_scrubbed_and_not_retried(make_stack, h):
    s = make_stack()
    err = {
        "type": "error",
        "error": {"type": "authentication_error", "message": f"bad {h.REAL_KEY}"},
    }
    s.fake.add(None, err, status=401)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body())
    assert status == 401
    assert h.REAL_KEY.encode() not in raw
    assert len(s.fake.requests) == 1  # no retry, no fallback
    assert s.sink.calls == []
    assert s.proxy.run_usage(h.KEY).calls == 0


def test_unreachable_upstream_gives_502_without_leaking(make_stack, h):
    s = make_stack()
    s.fake.stop()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body())
    assert status == 502
    assert h.REAL_KEY.encode() not in raw


def test_streaming_through_official_sdk(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message("streamed reply", model=h.HAIKU, in_tok=7, out_tok=3))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    with sdk(s, token).messages.stream(
        model=h.HAIKU,
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"output_config": {"effort": "medium"}},
    ) as stream:
        final = stream.get_final_message()
    assert final.model_dump()["content"][0]["text"] == "streamed reply"
    (call,) = s.sink.calls
    assert (call.in_tok, call.out_tok, call.usd_micros) == (7, 3, 22)


def test_nonstream_through_official_sdk_sees_clear_cap_error(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    with pytest.raises(anthropic.PermissionDeniedError) as exc:
        sdk(s, token).messages.create(
            model=h.HAIKU,
            max_tokens=21_000,  # worst case 21_000 * $5/M = $0.105 > $0.10 run cap
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"output_config": {"effort": "medium"}},
        )
    assert "cap for this run" in str(exc.value)


def test_refusal_is_recorded_and_never_rerouted(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message("", model=h.HAIKU, stop_reason="refusal"))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body())
    assert status == 200 and json.loads(raw)["stop_reason"] == "refusal"
    assert s.sink.calls[0].refusal is True
    assert s.proxy.run_usage(h.KEY).refusals == 1
    assert len(s.fake.requests) == 1
    assert s.fake.requests[0].body["model"] == h.HAIKU


def test_streamed_refusal_is_recorded(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message("no", model=h.HAIKU, stop_reason="refusal"))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, _ = s.post(token, h.msg_body(stream=True))
    assert status == 200
    assert s.sink.calls[0].refusal is True


def test_target_fallbacks_forbidden_but_proposer_passes_through(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.OPUS))
    target = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    for kwargs, hdrs in (
        ({"fallbacks": "default"}, {}),
        ({}, {"anthropic-beta": "server-side-fallback-2026-07-01"}),
    ):
        status, raw = s.post(target, h.msg_body(**kwargs), headers=hdrs)
        assert status == 403
        assert json.loads(raw)["error"]["iterant_code"] == "fallback_forbidden"
    assert s.fake.requests == []
    proposer = s.proxy.issue_call_token(Role.PROPOSER)
    hdrs = {"anthropic-beta": "server-side-fallback-2026-07-01"}
    body = h.msg_body(model=h.OPUS, effort="high", fallbacks="default")
    assert s.post(proposer, body, headers=hdrs)[0] == 200
    assert s.fake.requests[0].body["fallbacks"] == "default"
    assert s.fake.requests[0].headers["anthropic-beta"] == "server-side-fallback-2026-07-01"


def test_cache_tokens_are_priced(make_stack, h):
    s = make_stack()
    s.fake.add(
        None,
        anthropic_message(
            model=h.HAIKU, in_tok=1000, out_tok=0, cache_write=100, cache_read=10_000
        ),
    )
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, h.msg_body())[0] == 200
    (call,) = s.sink.calls
    assert call.in_tok == 11_100
    assert call.usd_micros == 2125  # 1000*1 + 100*1.25 + 10000*0.1 (micro-USD per token)


def test_per_run_cap_enforced_after_spend(make_stack, h):
    s = make_stack()
    # 10_000 in + 18_000 out = 10_000 + 90_000 micro-USD = the whole $0.10 run cap
    s.fake.add(None, anthropic_message(model=h.HAIKU, in_tok=10_000, out_tok=18_000))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, h.msg_body())[0] == 200
    status, raw = s.post(token, h.msg_body())
    assert status == 403
    assert json.loads(raw)["error"]["iterant_code"] == "cap_exceeded_run"
    assert len(s.fake.requests) == 1
    other = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t1", 1))
    assert s.post(other, h.msg_body())[0] == 200  # a different run has its own budget


def test_retry_attempt_gets_a_fresh_run_budget(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.HAIKU, in_tok=10_000, out_tok=18_000))
    first = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    s.post(first, h.msg_body())
    retry = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t1", 0, attempt=1))
    assert s.post(retry, h.msg_body())[0] == 200


def test_per_generation_cap_and_reset(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.HAIKU, in_tok=10_000, out_tok=18_000))  # $0.10 each
    for rep in range(3):  # gen cap is $0.30
        tok = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t1", rep))
        assert s.post(tok, h.msg_body())[0] == 200
    tok = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t1", 3))
    status, raw = s.post(tok, h.msg_body())
    assert status == 403 and json.loads(raw)["error"]["iterant_code"] == "cap_exceeded_gen"
    s.proxy.begin_generation(1)
    assert s.post(tok, h.msg_body())[0] == 200


def test_call_token_is_single_use_and_logs_no_run_key(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.OPUS))
    token = s.proxy.issue_call_token(Role.PROPOSER)
    body = h.msg_body(model=h.OPUS, effort="high")
    assert s.post(token, body)[0] == 200
    assert s.sink.calls[0].key is None and s.sink.calls[0].role is Role.PROPOSER
    assert s.post(token, body)[0] == 401
    multi = s.proxy.issue_call_token(Role.PROPOSER, uses=2)
    assert [s.post(multi, body)[0] for _ in range(3)] == [200, 200, 401]


def test_call_cap_applies_per_call(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_call_token(Role.PROPOSER)
    body = h.msg_body(model=h.OPUS, effort="high", max_tokens=100_000)  # 100k * $25/M = $2.5
    assert s.post(token, body)[0] == 403
    assert s.fake.requests == []


def test_role_must_be_in_policy(make_stack, h):
    s = make_stack()
    with pytest.raises(ValueError):
        s.proxy.issue_call_token(Role.ANALYST)
    with pytest.raises(ValueError):
        s.proxy.issue_run_token(Role.PROPOSER, h.KEY)  # wrong kind of token for this role


def test_bad_requests(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, h.msg_body(), path="/v1/models")[0] == 404
    no_max = h.msg_body()
    del no_max["max_tokens"]
    assert s.post(token, no_max)[0] == 400
    assert s.post(token, [1, 2])[0] == 400  # type: ignore[arg-type]
    assert s.fake.requests == []


def test_sink_failure_fails_closed(make_stack, h):
    def broken(call):
        raise OSError("disk full")

    s = make_stack(sink=broken)
    s.fake.add(None, anthropic_message(model=h.HAIKU))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, h.msg_body())[0] == 500
    assert s.proxy.healthy is False
    status, raw = s.post(token, h.msg_body())
    assert status == 503 and json.loads(raw)["error"]["iterant_code"] == "ledger_failed"


def test_config_repr_hides_key(make_stack, h):
    s = make_stack()
    assert h.REAL_KEY not in repr(s.proxy.config)


def test_train_content_only(make_stack, h):
    splits = {"tr": Split.TRAIN, "se": Split.SELECT, "ho": Split.HOLDOUT, "te": Split.TEST}
    s = make_stack(splits=splits)
    s.fake.add(None, anthropic_message("answer", model=h.HAIKU))
    keys = {t: RunKey(1, "c1", t, 0) for t in (*splits, "unknown")}
    for key in keys.values():
        assert s.post(s.proxy.issue_run_token(Role.TARGET, key), h.msg_body(stream=True))[0] == 200
    content = s.proxy.train_content()
    assert [c.key.task for c in content] == ["tr"]
    assert content[0].request["messages"][0]["content"] == "hi"
    assert content[0].response["content"][0]["text"] == "answer"
    for task in ("se", "ho", "te", "unknown"):
        assert s.proxy.train_content(keys[task]) == []
    assert len(s.sink.calls) == 5  # all calls are still logged (metadata only)


def test_split_lookup_failure_is_not_train(make_stack, h):
    s = make_stack()
    s.proxy.config.split_of  # noqa: B018
    object.__setattr__(s.proxy.config, "split_of", lambda k: 1 / 0)
    s.fake.add(None, anthropic_message(model=h.HAIKU))
    s.post(s.proxy.issue_run_token(Role.TARGET, h.KEY), h.msg_body())
    assert s.proxy.train_content() == []


def test_split_changing_to_non_train_hides_stored_content(make_stack, h):
    s = make_stack(splits={"t1": Split.TRAIN})
    s.fake.add(None, anthropic_message(model=h.HAIKU))
    s.post(s.proxy.issue_run_token(Role.TARGET, h.KEY), h.msg_body())
    assert len(s.proxy.train_content()) == 1
    object.__setattr__(s.proxy.config, "split_of", lambda k: Split.SELECT)
    assert s.proxy.train_content() == []


def test_cut_off_call_without_usage_is_charged_worst_case(make_stack, h):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    adm = s.proxy.admit(token, json.dumps(h.msg_body(max_tokens=200)).encode())
    call = s.proxy.settle(adm, None, False, None)  # type: ignore[arg-type]
    assert call.out_tok == 200 and call.usd_micros == 1000  # 200 * $5/M
    assert s.proxy.run_usage(h.KEY).usd_micros == 1000
