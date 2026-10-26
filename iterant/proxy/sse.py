"""Server-sent events: parse Anthropic stream events and rebuild the final message.

The proxy relays stream bytes untouched; this only watches them to learn usage, stop
reason and (for train-split runs) the content.
"""

from __future__ import annotations

import json
from typing import Any

from iterant.proxy.jsonutil import Obj, num, obj, text
from iterant.proxy.ledger import Usage


def sse_event(event: str, data: Obj) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


class StreamAccumulator:
    """Feed it raw SSE lines (or already-parsed events); read usage and the message after."""

    def __init__(self) -> None:
        self._message: Obj = {}
        self._blocks: dict[int, Obj] = {}
        self._partial_json: dict[int, list[str]] = {}
        self._usage: Obj = {}
        self.stop_reason: str | None = None
        self.stopped = False  # saw message_stop
        self.error: Obj | None = None
        self.saw_usage = False

    def feed_line(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return
        try:
            data = obj(json.loads(stripped[5:].strip()))
        except ValueError:
            return
        if data is not None:
            self.feed_event(data)

    def feed_event(self, ev: Obj) -> None:
        kind = ev.get("type")
        if kind == "message_start":
            message = obj(ev.get("message")) or {}
            self._message = {k: v for k, v in message.items() if k != "content"}
            self._merge_usage(obj(message.get("usage")))
        elif kind == "content_block_start":
            index = num(ev.get("index"))
            block = dict(obj(ev.get("content_block")) or {})
            self._blocks[index] = block
        elif kind == "content_block_delta":
            self._apply_delta(num(ev.get("index")), obj(ev.get("delta")) or {})
        elif kind == "content_block_stop":
            self._close_block(num(ev.get("index")))
        elif kind == "message_delta":
            delta = obj(ev.get("delta")) or {}
            self.stop_reason = text(delta.get("stop_reason")) or self.stop_reason
            self._message["stop_reason"] = self.stop_reason
            self._merge_usage(obj(ev.get("usage")))
        elif kind == "message_stop":
            self.stopped = True
        elif kind == "error":
            self.error = obj(ev.get("error")) or {}

    def _merge_usage(self, usage: Obj | None) -> None:
        if usage:
            self.saw_usage = True
            self._usage.update({k: v for k, v in usage.items() if isinstance(v, int)})

    def _apply_delta(self, index: int, delta: Obj) -> None:
        block = self._blocks.setdefault(index, {})
        kind = delta.get("type")
        if kind == "text_delta":
            block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
        elif kind == "thinking_delta":
            block["thinking"] = str(block.get("thinking", "")) + str(delta.get("thinking", ""))
        elif kind == "signature_delta":
            block["signature"] = delta.get("signature")
        elif kind == "input_json_delta":
            self._partial_json.setdefault(index, []).append(str(delta.get("partial_json", "")))

    def _close_block(self, index: int) -> None:
        block = self._blocks.get(index)
        parts = self._partial_json.pop(index, None)
        if block is not None and parts is not None:
            joined = "".join(parts)
            try:
                block["input"] = json.loads(joined) if joined else {}
            except ValueError:
                block["input"] = {"_unparsed": joined}

    @property
    def usage(self) -> Usage:
        u = self._usage
        return Usage(
            input=num(u.get("input_tokens")),
            output=num(u.get("output_tokens")),
            cache_write=num(u.get("cache_creation_input_tokens")),
            cache_read=num(u.get("cache_read_input_tokens")),
        )

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def message(self) -> Obj:
        for index in list(self._partial_json):
            self._close_block(index)
        content: list[Any] = [self._blocks[i] for i in sorted(self._blocks)]
        return {**self._message, "content": content, "usage": dict(self._usage)}


def usage_of(message: Obj) -> tuple[Usage, bool]:
    """Usage and refusal flag from a non-streaming Messages response."""
    u = obj(message.get("usage")) or {}
    usage = Usage(
        input=num(u.get("input_tokens")),
        output=num(u.get("output_tokens")),
        cache_write=num(u.get("cache_creation_input_tokens")),
        cache_read=num(u.get("cache_read_input_tokens")),
    )
    return usage, message.get("stop_reason") == "refusal"
