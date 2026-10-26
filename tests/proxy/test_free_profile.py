"""Free profile: Anthropic in, OpenAI-compatible fake upstream out; caps are in tokens."""

from __future__ import annotations

import json

import anthropic

from iterant.contracts import Profile, Role, RunKey
from iterant.proxy import openai_completion


def body(h, **extra):
    return {
        "model": h.LOCAL,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def test_translates_request_and_response(make_stack, h):
    s = make_stack(Profile.FREE)
    s.fake.add(None, openai_completion("hello", model=h.LOCAL, in_tok=12, out_tok=4))
    token = s.proxy.issue_call_token(Role.PROPOSER)
    req = body(
        h,
        system=[{"type": "text", "text": "be brief"}],
        stop_sequences=["END"],
        temperature=0.2,
        top_k=5,
        output_config={"effort": "medium"},
        tools=[{"name": "calc", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "any"},
        messages=[
            {"role": "user", "content": "add"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hm", "signature": "s"},
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "tu1", "name": "calc", "input": {"a": 1}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [{"type": "text", "text": "2"}],
                    },
                    {"type": "text", "text": "thanks"},
                ],
            },
        ],
    )
    status, raw = s.post(token, req)
    assert status == 200
    out = json.loads(raw)
    assert out["content"] == [{"type": "text", "text": "hello"}]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 4}
    sent = s.fake.requests[0]
    assert sent.path == "/v1/chat/completions"
    b = sent.body
    assert b["stop"] == ["END"] and b["temperature"] == 0.2 and b["tool_choice"] == "required"
    assert "top_k" not in b and "output_config" not in b
    fn = b["tools"][0]["function"]
    assert fn == {"name": "calc", "description": "d", "parameters": {"type": "object"}}
    roles = [m["role"] for m in b["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user"]
    assert b["messages"][2]["tool_calls"][0]["function"] == {
        "name": "calc",
        "arguments": '{"a": 1}',
    }
    assert b["messages"][3] == {"role": "tool", "tool_call_id": "tu1", "content": "2"}
    (call,) = s.sink.calls
    assert (call.in_tok, call.out_tok, call.usd_micros, call.model) == (12, 4, 0, h.LOCAL)


def test_tool_call_response_and_finish_reasons(make_stack, h):
    s = make_stack(Profile.FREE)
    calls = [{"id": "c1", "name": "calc", "arguments": '{"a": 2}'}]
    tool = openai_completion("", model=h.LOCAL, finish_reason="tool_calls", tool_calls=calls)
    s.fake.add(None, tool, times=1)
    s.fake.add(None, openai_completion("x", model=h.LOCAL, finish_reason="length"), times=1)
    s.fake.add(None, openai_completion("", model=h.LOCAL, finish_reason="content_filter"), times=1)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    first = json.loads(s.post(token, body(h))[1])
    assert first["stop_reason"] == "tool_use"
    assert first["content"] == [{"type": "tool_use", "id": "c1", "name": "calc", "input": {"a": 2}}]
    assert json.loads(s.post(token, body(h))[1])["stop_reason"] == "max_tokens"
    refused = json.loads(s.post(token, body(h))[1])
    assert refused["stop_reason"] == "refusal"
    assert [c.refusal for c in s.sink.calls] == [False, False, True]


def test_streaming_text_via_sdk(make_stack, h):
    s = make_stack(Profile.FREE)
    s.fake.add(None, openai_completion("streamed local", model=h.LOCAL, in_tok=9, out_tok=3))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    client = anthropic.Anthropic(base_url=s.server.base_url, api_key=token, max_retries=0)
    with client.messages.stream(
        model=h.LOCAL, max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        final = stream.get_final_message()
    assert final.model_dump()["content"][0]["text"] == "streamed local"
    assert final.stop_reason == "end_turn"
    assert s.fake.requests[0].body["stream_options"] == {"include_usage": True}
    (call,) = s.sink.calls
    assert (call.in_tok, call.out_tok) == (9, 3)


def test_streaming_tool_call_via_sdk(make_stack, h):
    s = make_stack(Profile.FREE)
    calls = [{"id": "c1", "name": "calc", "arguments": '{"a": 2, "b": 3}'}]
    resp = openai_completion("let me", model=h.LOCAL, finish_reason="tool_calls", tool_calls=calls)
    s.fake.add(None, resp)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    client = anthropic.Anthropic(base_url=s.server.base_url, api_key=token, max_retries=0)
    with client.messages.stream(
        model=h.LOCAL,
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "calc", "input_schema": {"type": "object"}}],
    ) as stream:
        final = stream.get_final_message()
    assert [b.type for b in final.content] == ["text", "tool_use"]
    assert final.model_dump()["content"][1]["input"] == {"a": 2, "b": 3}
    assert final.stop_reason == "tool_use"


def test_missing_usage_is_estimated_not_free(make_stack, h):
    s = make_stack(Profile.FREE)
    resp = openai_completion("a reply of some length", model=h.LOCAL)
    del resp["usage"]
    s.fake.add(None, resp)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, body(h))[0] == 200
    assert s.post(token, body(h, stream=True))[0] == 200
    assert all(c.in_tok > 0 and c.out_tok > 0 for c in s.sink.calls)


def test_token_caps(make_stack, h):
    s = make_stack(Profile.FREE)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, body(h, max_tokens=2_000))[0] == 403  # worst case alone > 1000
    s.fake.add(None, openai_completion("x", model=h.LOCAL, in_tok=600, out_tok=300))
    assert s.post(token, body(h))[0] == 200  # 900 spent
    status, raw = s.post(token, body(h, max_tokens=200))  # 900 + 200 > 1000
    assert status == 403 and json.loads(raw)["error"]["iterant_code"] == "cap_exceeded_run"
    assert s.post(token, body(h, max_tokens=100))[0] == 200  # exactly fits: 1800 spent in gen
    other = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t2", 0))
    assert s.post(other, body(h))[0] == 200  # gen total 2700 > 2500 cap is reached next
    third = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t3", 0))
    status, raw = s.post(third, body(h))
    assert status == 403 and json.loads(raw)["error"]["iterant_code"] == "cap_exceeded_gen"


def test_frozen_model_applies_to_free_profile(make_stack, h):
    s = make_stack(Profile.FREE)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    assert s.post(token, body(h, model="some-bigger-local-model"))[0] == 403
    assert s.post(token, body(h, output_config={"effort": "max"}))[0] == 403
    assert s.fake.requests == []


def test_unsupported_content_and_upstream_errors(make_stack, h):
    s = make_stack(Profile.FREE)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    doc = {"role": "user", "content": [{"type": "document", "source": {}}]}
    proposer = s.proxy.issue_call_token(Role.PROPOSER)
    status, raw = s.post(proposer, body(h, messages=[doc]))
    assert status == 400
    assert "not supported by the free profile" in json.loads(raw)["error"]["message"]
    s.fake.add(None, {"error": {"message": "model not found"}}, status=404)
    status, raw = s.post(token, body(h))
    assert status == 404 and json.loads(raw)["type"] == "error"
    assert len(s.fake.requests) == 1 and s.sink.calls == []
