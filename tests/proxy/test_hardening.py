"""Review fixes: role allowlist, disconnect accounting, DoS limits, pricing by reported model."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from iterant.contracts import Profile, ProxyPolicy, Role, RoleCap, RunKey
from iterant.proxy import (
    Admission,
    InMemorySink,
    Proxy,
    ProxyConfig,
    Rejection,
    anthropic_message,
)
from iterant.proxy.fake import anthropic_sse

# --- target/transfer allowlist ------------------------------------------------

B64 = {"type": "base64", "media_type": "image/png", "data": "AAAA"}
BAD_BODIES = {
    "server_tool": {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
    "tool_without_schema": {"tools": [{"name": "t"}]},
    "mcp_servers": {"mcp_servers": [{"url": "https://attacker.example/mcp", "name": "x"}]},
    "speed": {"speed": "fast"},
    "container": {"container": "c1"},
    "service_tier": {"service_tier": "priority"},
    "temperature": {"temperature": 1.0},
    "top_k": {"top_k": 3},
    "thinking_enabled": {"thinking": {"type": "enabled", "budget_tokens": 5000}},
    "output_config_format": {"output_config": {"effort": "medium", "format": {"type": "x"}}},
    "tool_choice_any": {
        "tools": [{"name": "t", "input_schema": {}}],
        "tool_choice": {"type": "any"},
    },
    "image_url": {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": "http://x/i.png"}}],
            }
        ]
    },
    "document_file": {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": "f"}}],
            }
        ]
    },
    "nested_image_url": {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t",
                        "content": [
                            {"type": "image", "source": {"type": "url", "url": "http://x"}}
                        ],
                    }
                ],
            }
        ]
    },
    "server_tool_use_block": {
        "messages": [{"role": "assistant", "content": [{"type": "server_tool_use", "id": "x"}]}]
    },
    "mcp_block": {"system": [{"type": "mcp_tool_use", "id": "x"}]},
}


@pytest.mark.parametrize("name", list(BAD_BODIES))
def test_target_allowlist_rejects(make_stack, h, name):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = {**h.msg_body(), **BAD_BODIES[name]}
    status, raw = s.post(token, body)
    assert status == 400, raw
    assert json.loads(raw)["error"]["type"] == "invalid_request_error"
    assert s.fake.requests == []


@pytest.mark.parametrize("beta", ["mcp-client-2025-04-04", "context-1m-2025-08-07", "files-api"])
def test_target_beta_headers_rejected(make_stack, h, beta):
    s = make_stack()
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body(), headers={"anthropic-beta": beta})
    assert status == 400 and json.loads(raw)["error"]["iterant_code"] == "beta_not_allowed"
    assert s.fake.requests == []


def test_target_allowlist_accepts_normal_agent_requests(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.HAIKU))
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    body = h.msg_body(
        system=[{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"},
        metadata={"user_id": "u"},
        stop_sequences=["x"],
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "source": B64}, {"type": "text", "text": "a"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "t", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}],
            },
        ],
    )
    assert s.post(token, body)[0] == 200
    assert "anthropic-beta" not in s.fake.requests[0].headers


def test_proposer_requests_pass_through_unrestricted(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.OPUS))
    token = s.proxy.issue_call_token(Role.PROPOSER)
    body = h.msg_body(model=h.OPUS, effort="high", mcp_servers=[{"url": "https://x", "name": "x"}])
    assert s.post(token, body, headers={"anthropic-beta": "mcp-client-2025-04-04"})[0] == 200
    assert s.fake.requests[0].headers["anthropic-beta"] == "mcp-client-2025-04-04"
    assert s.fake.requests[0].body["mcp_servers"]


def test_free_translation_never_fetches_image_urls(make_stack, h):
    s = make_stack(Profile.FREE)
    token = s.proxy.issue_call_token(Role.PROPOSER)
    url_img = {"type": "image", "source": {"type": "url", "url": "http://x/i.png"}}
    msgs = [{"role": "user", "content": [url_img]}]
    status, _ = s.post(token, {"model": h.LOCAL, "max_tokens": 10, "messages": msgs})
    assert status == 400 and s.fake.requests == []


# --- limits ---------------------------------------------------------------------


def _raw(s, head: str) -> socket.socket:
    c = socket.create_connection(("127.0.0.1", s.server._httpd.server_address[1]), timeout=5)  # pyright: ignore
    c.sendall(head.encode())
    return c


def test_token_checked_before_body_is_read(make_stack, h):
    s = make_stack()
    c = _raw(s, "POST /v1/messages HTTP/1.1\r\nx-api-key: nope\r\ncontent-length: 5000000\r\n\r\n")
    assert c.recv(100).startswith(b"HTTP/1.0 401")  # replied without waiting for 5 MB
    c.close()


def test_body_limit_and_concurrency_cap(make_stack, h):
    s = make_stack(max_body_bytes=200, max_concurrent=1)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    big = h.msg_body(system="x" * 500)
    assert s.post(token, big)[0] == 413
    head = f"POST /v1/messages HTTP/1.1\r\nx-api-key: {token}\r\ncontent-length: 100\r\n\r\n{{"
    hold = _raw(s, head)  # takes the only slot and stalls mid-body
    try:
        other = s.proxy.issue_run_token(Role.TARGET, RunKey(1, "c1", "t1", 1))
        status, raw = s.post(other, h.msg_body())
        assert status == 503 and json.loads(raw)["error"]["type"] == "overloaded_error"
    finally:
        hold.close()


# --- pricing and keys -------------------------------------------------------------


def test_priced_and_logged_at_reported_model(make_stack, h):
    s = make_stack()
    s.fake.add(None, anthropic_message(model=h.OPUS, in_tok=10, out_tok=5), times=1)
    s.fake.add(None, anthropic_message(model="unlisted-model", in_tok=10, out_tok=5), times=1)
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    s.post(token, h.msg_body())
    s.post(token, h.msg_body())
    a, b = s.sink.calls
    assert (a.model, a.usd_micros) == (h.OPUS, 175)  # 10 * $5/M + 5 * $25/M, not haiku's 35
    assert (b.model, b.usd_micros) == ("unlisted-model", 175)  # unknown: dearest price


def test_run_balance_is_per_role_and_key(h):
    from iterant.proxy import ModelPrice

    cap = RoleCap(h.HAIKU, "medium", max_usd_micros_per_run=1000, max_usd_micros_per_gen=5000)
    policy = ProxyPolicy(Profile.CLAUDE, {Role.TARGET: cap, Role.TRANSFER: cap})
    proxy = Proxy(
        ProxyConfig(
            policy=policy,
            sink=InMemorySink(),
            prices={h.HAIKU: ModelPrice(1_000_000, 5_000_000)},
            anthropic_base_url="http://unused",
            anthropic_api_key="k",
        )
    )
    body = json.dumps(h.msg_body(max_tokens=150)).encode()  # reserves 750 of 1000
    t1 = proxy.issue_run_token(Role.TARGET, h.KEY)
    t2 = proxy.issue_run_token(Role.TRANSFER, h.KEY)  # same RunKey, other role
    assert isinstance(proxy.admit(t1, body), Admission)
    assert isinstance(proxy.admit(t1, body), Rejection)  # 750 held + 750 > 1000
    assert isinstance(proxy.admit(t2, body), Admission)  # same RunKey, own balance


def test_preload_restores_spend(make_stack, h):
    from iterant.contracts import ProxyCall

    s = make_stack()
    spent = ProxyCall(
        1, h.KEY, Role.TARGET, h.HAIKU, 10, 10, 100_000, False, "2026-01-01T00:00:00Z"
    )
    s.proxy.preload([spent])
    token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
    status, raw = s.post(token, h.msg_body())
    assert status == 403 and json.loads(raw)["error"]["iterant_code"] == "cap_exceeded_run"


# --- client disconnect and cut-off streams -------------------------------------------


class Stub:
    """Sends ``first``, then waits for ``release`` and sends ``rest`` (or closes if ``cut``)."""

    def __init__(self, first: list[bytes], rest: list[bytes], cut: bool) -> None:
        self.release = threading.Event()
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format, *args):
                return

            def do_POST(self):
                self.rfile.read(int(self.headers["content-length"]))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                for e in first:
                    self.wfile.write(e)
                self.wfile.flush()
                if cut:
                    return
                stub.release.wait(5)
                for e in rest:
                    self.wfile.write(e)
                    self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class SignalSink(InMemorySink):
    def __init__(self) -> None:
        super().__init__()
        self.got = threading.Event()

    def __call__(self, call) -> None:
        super().__call__(call)
        self.got.set()


def _read_until(c: socket.socket, needle: bytes) -> bytes:
    data = b""
    while needle not in data:
        part = c.recv(65536)
        if not part:
            break
        data += part
    return data


def _stream_request(s, token, body) -> socket.socket:
    payload = json.dumps(body).encode()
    head = (
        f"POST /v1/messages HTTP/1.1\r\nx-api-key: {token}\r\ncontent-type: application/json\r\n"
        f"content-length: {len(payload)}\r\n\r\n"
    )
    c = _raw(s, head)
    c.sendall(payload)
    return c


def _anthropic_events(h):
    msg = anthropic_message("answer text here", model=h.HAIKU, in_tok=150_000, out_tok=40)
    events = anthropic_sse(msg)
    return events[:1], events[1:]


@pytest.mark.parametrize("cut", [False, True])
def test_anthropic_disconnect_charges_real_or_seen_input(make_stack, h, cut):
    first, rest = _anthropic_events(h)
    stub = Stub(first, rest, cut)
    sink = SignalSink()
    try:
        s = make_stack(sink=sink, upstream=stub.base)
        token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
        c = _stream_request(s, token, h.msg_body(stream=True))
        assert b"message_start" in _read_until(c, b"message_start")
        c.close()  # client leaves mid-stream
        stub.release.set()
        assert sink.got.wait(10)
    finally:
        stub.close()
    (call,) = sink.calls
    assert call.in_tok == 150_000  # never zero
    if cut:
        assert call.out_tok == 100  # upstream died: assume the worst, max_tokens
        assert call.usd_micros == 150_000 + 500
    else:
        assert call.out_tok == 40  # drained to the real final usage
        assert call.usd_micros == 150_000 + 200
    assert s.proxy.run_usage(h.KEY).in_tok == 150_000


def _openai_events(with_usage: bool) -> tuple[list[bytes], list[bytes]]:
    def chunk(delta, finish=None, extra=None):
        d = {
            "id": "c",
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps({**d, **(extra or {})}).encode() + b"\n\n"

    first = [chunk({"role": "assistant", "content": "hel"})]
    rest = [chunk({"content": "lo"}), chunk({}, "stop")]
    if with_usage:
        usage = {"id": "c", "choices": [], "usage": {"prompt_tokens": 700, "completion_tokens": 30}}
        rest.append(b"data: " + json.dumps(usage).encode() + b"\n\n")
    rest.append(b"data: [DONE]\n\n")
    return first, rest


@pytest.mark.parametrize("cut", [False, True])
def test_free_disconnect_charges_real_or_estimated_input(make_stack, h, cut):
    first, rest = _openai_events(with_usage=True)
    stub = Stub(first, rest, cut)
    sink = SignalSink()
    try:
        s = make_stack(Profile.FREE, sink=sink, upstream=stub.base + "/v1")
        token = s.proxy.issue_run_token(Role.TARGET, h.KEY)
        body = {"model": h.LOCAL, "max_tokens": 100, "stream": True}
        body["messages"] = [{"role": "user", "content": "q" * 800}]
        c = _stream_request(s, token, body)
        assert b"message_start" in _read_until(c, b"message_start")
        c.close()
        stub.release.set()
        assert sink.got.wait(10)
    finally:
        stub.close()
    (call,) = sink.calls
    if cut:
        assert call.in_tok >= 200 and call.out_tok == 100  # estimated input + max_tokens
    else:
        assert (call.in_tok, call.out_tok) == (700, 30)


def test_sample_events_are_wellformed(h):
    first, rest = _anthropic_events(h)
    assert first[0].startswith(b"event: message_start") and rest[-1].startswith(
        b"event: message_stop"
    )
