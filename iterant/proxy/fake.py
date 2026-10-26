"""Recorded-fixture fake LLM: the shared upstream for tests (nothing here touches a network).

``FakeLLM`` is a localhost HTTP server (port 0) that speaks both upstream dialects:
``POST /v1/messages`` (Anthropic) and ``POST /v1/chat/completions`` (OpenAI-compatible).
It replays recorded responses; if the request has ``"stream": true`` it renders the
recorded (non-streaming) response as the matching SSE stream.

Usage::

    fake = FakeLLM()
    fake.add({"model": "m", "messages": [{"role": "user", "content": "hi"}]},
             anthropic_message("hello", model="m"))
    with fake:                       # or fake.start() / fake.stop()
        client = anthropic.Anthropic(base_url=fake.base_url, api_key="x")
        ...
    fake.requests                    # every request seen (path, headers, body)
    fake.unmatched                   # requests no fixture matched (they got a 404)

Matching: ``match`` is a mapping compared recursively as a *subset* of the request body
(dicts: every listed key must match; lists: same length, items match pairwise; anything
else: equal), or a ``Callable[[dict], bool]``, or None for "any request". Fixtures are
tried in the order added; the first live match wins. ``times=n`` makes a fixture expire
after ``n`` uses. ``path`` restricts a fixture to one endpoint.

Recorded files: ``FakeLLM.from_file(path)`` reads JSON
``{"fixtures": [{"match": {...}, "response": {...}, "status": 200, "path": "/v1/messages",
"times": 1}]}``. ``request_key(body)`` gives a stable sha256 of a request body for
recording tools that want exact keys: a fixture ``{"key": "<sha>"}`` matches only a body
with that exact key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from iterant.proxy.jsonutil import Obj, arr, obj, text

Matcher = Mapping[str, Any] | Callable[[Obj], bool] | None

ANTHROPIC_PATH = "/v1/messages"
OPENAI_PATH = "/v1/chat/completions"


def request_key(body: Obj) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _subset(want: object, got: object) -> bool:
    w, g = obj(want), obj(got)
    if w is not None:
        return g is not None and all(k in g and _subset(v, g[k]) for k, v in w.items())
    wl, gl = arr(want), arr(got)
    if wl is not None:
        return (
            gl is not None
            and len(wl) == len(gl)
            and all(_subset(a, b) for a, b in zip(wl, gl, strict=True))
        )
    return want == got


@dataclass
class Fixture:
    response: Obj
    match: Matcher = None
    status: int = 200
    path: str | None = None
    times: int | None = None
    key: str | None = None
    used: int = 0

    def matches(self, path: str, body: Obj) -> bool:
        if self.times is not None and self.used >= self.times:
            return False
        if self.path is not None and self.path != path:
            return False
        if self.key is not None and self.key != request_key(body):
            return False
        if self.match is None:
            return True
        if callable(self.match):
            return self.match(body)
        return _subset(self.match, body)


@dataclass(frozen=True)
class SeenRequest:
    path: str
    headers: Mapping[str, str]  # lower-cased names
    body: Obj


def anthropic_message(
    reply: str = "ok",
    *,
    model: str = "fake-model",
    in_tok: int = 10,
    out_tok: int = 5,
    stop_reason: str = "end_turn",
    cache_write: int = 0,
    cache_read: int = 0,
    tool_use: Obj | None = None,
) -> Obj:
    """A Messages API response. ``tool_use`` is ``{"id":..., "name":..., "input": {...}}``."""
    content: list[Obj] = []
    if reply:
        content.append({"type": "text", "text": reply})
    if tool_use:
        content.append({"type": "tool_use", **tool_use})
    usage: Obj = {"input_tokens": in_tok, "output_tokens": out_tok}
    if cache_write:
        usage["cache_creation_input_tokens"] = cache_write
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


def openai_completion(
    reply: str = "ok",
    *,
    model: str = "fake-model",
    in_tok: int = 10,
    out_tok: int = 5,
    finish_reason: str = "stop",
    tool_calls: list[Obj] | None = None,
) -> Obj:
    """A Chat Completions response. Each tool call is ``{"id", "name", "arguments": str}``."""
    message: Obj = {"role": "assistant", "content": reply or None}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for c in tool_calls
        ]
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


def _sse(event: str | None, data: Obj | str) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


def _halves(s: str) -> list[str]:
    mid = len(s) // 2
    return [p for p in (s[:mid], s[mid:]) if p]


def anthropic_sse(message: Obj) -> list[bytes]:
    """The Messages stream for a recorded response (text split in two deltas)."""
    usage = obj(message.get("usage")) or {}
    start_usage = {**usage, "output_tokens": 1}
    head = {k: v for k, v in message.items() if k not in ("content", "stop_reason")}
    out = [
        _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {**head, "content": [], "stop_reason": None, "usage": start_usage},
            },
        )
    ]
    for i, b in enumerate(arr(message.get("content")) or []):
        block = obj(b) or {}
        if block.get("type") == "tool_use":
            start: Obj = {**block, "input": {}}
            deltas: list[Obj] = [
                {"type": "input_json_delta", "partial_json": p}
                for p in _halves(json.dumps(block.get("input", {})))
            ]
        else:
            start = {**block, "text": ""}
            deltas = [
                {"type": "text_delta", "text": p} for p in _halves(str(block.get("text", "")))
            ]
        out.append(
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": start},
            )
        )
        for d in deltas:
            out.append(
                _sse("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d})
            )
        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": i}))
    out.append(
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": None},
                "usage": {"output_tokens": usage.get("output_tokens", 0)},
            },
        )
    )
    out.append(_sse("message_stop", {"type": "message_stop"}))
    return out


def openai_sse(completion: Obj) -> list[bytes]:
    """The Chat Completions stream for a recorded response (usage in a final chunk)."""
    choice = obj((arr(completion.get("choices")) or [None])[0]) or {}
    message = obj(choice.get("message")) or {}
    base = {
        "id": completion.get("id"),
        "object": "chat.completion.chunk",
        "model": completion.get("model"),
    }

    def chunk(delta: Obj, finish: str | None = None) -> bytes:
        return _sse(
            None, {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        )

    out = [chunk({"role": "assistant", "content": ""})]
    for p in _halves(text(message.get("content")) or ""):
        out.append(chunk({"content": p}))
    for i, c in enumerate(arr(message.get("tool_calls")) or []):
        call = obj(c) or {}
        fn = obj(call.get("function")) or {}
        first: Obj = {
            "index": i,
            "id": call.get("id"),
            "type": "function",
            "function": {"name": fn.get("name"), "arguments": ""},
        }
        out.append(chunk({"tool_calls": [first]}))
        for p in _halves(str(fn.get("arguments", ""))):
            out.append(chunk({"tool_calls": [{"index": i, "function": {"arguments": p}}]}))
    out.append(chunk({}, str(choice.get("finish_reason", "stop"))))
    if "usage" in completion:
        out.append(_sse(None, {**base, "choices": [], "usage": completion["usage"]}))
    out.append(b"data: [DONE]\n\n")
    return out


@dataclass
class FakeLLM:
    """See the module docstring. Thread-safe; one instance per fake upstream."""

    fixtures: list[Fixture] = field(default_factory=lambda: [])
    requests: list[SeenRequest] = field(default_factory=lambda: [])
    unmatched: list[SeenRequest] = field(default_factory=lambda: [])
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _httpd: ThreadingHTTPServer | None = field(default=None, repr=False)

    def add(
        self,
        match: Matcher,
        response: Obj,
        *,
        status: int = 200,
        path: str | None = None,
        times: int | None = None,
    ) -> FakeLLM:
        self.fixtures.append(Fixture(response, match, status, path, times))
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> FakeLLM:
        data = obj(json.loads(Path(path).read_text(encoding="utf-8"))) or {}
        fake = cls()
        for f in arr(data.get("fixtures")) or []:
            fx = obj(f) or {}
            fake.fixtures.append(
                Fixture(
                    response=obj(fx.get("response")) or {},
                    match=obj(fx.get("match")),
                    status=int(fx.get("status", 200)),
                    path=text(fx.get("path")),
                    times=fx.get("times"),
                    key=text(fx.get("key")),
                )
            )
        return fake

    def _find(self, req: SeenRequest) -> Fixture | None:
        with self._lock:
            self.requests.append(req)
            for fx in self.fixtures:
                if fx.matches(req.path, req.body):
                    fx.used += 1
                    return fx
            self.unmatched.append(req)
            return None

    # --- server -----------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Anthropic upstream base (no ``/v1``)."""
        assert self._httpd is not None, "start() first"
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    @property
    def openai_base_url(self) -> str:
        """OpenAI-compatible upstream base (with ``/v1``), as Ollama exposes it."""
        return self.base_url + "/v1"

    def start(self) -> FakeLLM:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                raw = self.rfile.read(int(self.headers.get("content-length", "0")))
                try:
                    body = obj(json.loads(raw)) or {}
                except ValueError:
                    body = {}
                req = SeenRequest(path, {k.lower(): v for k, v in self.headers.items()}, body)
                fx = fake._find(req)  # pyright: ignore[reportPrivateUsage]
                if path not in (ANTHROPIC_PATH, OPENAI_PATH) or fx is None:
                    self._json(404, {"error": {"message": "fake_llm: no fixture matched"}})
                elif body.get("stream") is True and fx.status == 200:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.end_headers()
                    events = (
                        anthropic_sse(fx.response)
                        if path == ANTHROPIC_PATH
                        else openai_sse(fx.response)
                    )
                    for e in events:
                        self.wfile.write(e)
                        self.wfile.flush()
                else:
                    self._json(fx.status, fx.response)

            def _json(self, status: int, data: Obj) -> None:
                payload = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> FakeLLM:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
