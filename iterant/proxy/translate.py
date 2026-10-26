"""Anthropic Messages API <-> OpenAI-compatible Chat Completions (the ``free`` profile).

The agent speaks the Anthropic API to the proxy; for ``free`` the proxy translates to an
OpenAI-compatible upstream (Ollama, llama.cpp).

Supported
---------
* ``system`` (string or text blocks), ``messages`` with string or block content.
* Blocks: ``text``, ``image`` (base64 and url sources), ``tool_use``, ``tool_result``
  (string or text-block content). ``thinking`` / ``redacted_thinking`` blocks in history
  are dropped.
* ``max_tokens``, ``temperature``, ``top_p``, ``stop_sequences``, ``stream``,
  ``tools`` (``input_schema`` becomes ``parameters``), ``tool_choice``
  (``auto``, ``any``, ``tool``, ``none``).
* Responses: text and tool calls, ``usage``, and ``finish_reason`` mapped to
  ``stop_reason`` (``stop`` -> ``end_turn``, ``length`` -> ``max_tokens``,
  ``tool_calls`` -> ``tool_use``, ``content_filter`` -> ``refusal``). Streaming is
  translated chunk by chunk into Anthropic events.

Dropped silently: ``top_k``, ``metadata``, ``thinking``, ``output_config``,
``cache_control``, ``service_tier``. Anything else unknown is dropped too.
Refused with a 400 (:class:`Unsupported`): ``document`` blocks and server tools
(tools without ``input_schema``).
"""

from __future__ import annotations

import json
from typing import Any

from iterant.proxy.jsonutil import Obj, arr, num, obj, text

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class Unsupported(ValueError):
    """The request uses something the OpenAI-compatible upstream cannot express."""


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for b in arr(content) or []:
        block = obj(b)
        if block is not None and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _image_part(block: Obj) -> Obj:
    source = obj(block.get("source")) or {}
    if source.get("type") == "base64":
        url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
    elif source.get("type") == "url":
        url = str(source.get("url", ""))
    else:
        raise Unsupported("image source type is not supported by the free profile")
    return {"type": "image_url", "image_url": {"url": url}}


def _user_messages(content: object) -> list[Obj]:
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    tool_msgs: list[Obj] = []
    parts: list[Obj] = []
    for b in arr(content) or []:
        block = obj(b)
        if block is None:
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif kind == "image":
            parts.append(_image_part(block))
        elif kind == "tool_result":
            body = _block_text(block.get("content"))
            if block.get("is_error"):
                body = "Error: " + body
            tool_msgs.append(
                {"role": "tool", "tool_call_id": str(block.get("tool_use_id", "")), "content": body}
            )
        else:
            raise Unsupported(
                f"user content block type {kind!r} is not supported by the free profile"
            )
    out = tool_msgs  # tool results must directly follow the assistant tool_calls
    if parts:
        only_text = all(p["type"] == "text" for p in parts)
        out.append(
            {"role": "user", "content": "".join(p["text"] for p in parts) if only_text else parts}
        )
    return out


def _assistant_message(content: object) -> Obj:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts: list[str] = []
    calls: list[Obj] = []
    for b in arr(content) or []:
        block = obj(b)
        if block is None:
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise Unsupported(
                f"assistant content block type {kind!r} is not supported by the free profile"
            )
    msg: Obj = {"role": "assistant", "content": "".join(texts) if texts else None}
    if calls:
        msg["tool_calls"] = calls
    return msg


def _tool_choice(choice: Obj) -> Any:
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool":
        return {"type": "function", "function": {"name": str(choice.get("name", ""))}}
    raise Unsupported(f"tool_choice type {kind!r} is not supported by the free profile")


def to_openai_request(body: Obj) -> Obj:
    messages: list[Obj] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _block_text(system)})
    for m in arr(body.get("messages")) or []:
        msg = obj(m)
        if msg is None:
            continue
        if msg.get("role") == "assistant":
            messages.append(_assistant_message(msg.get("content")))
        else:
            messages.extend(_user_messages(msg.get("content")))
    out: Obj = {"model": body["model"], "messages": messages}
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if src in body:
            out[dst] = body[src]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    tools: list[Obj] = []
    for t in arr(body.get("tools")) or []:
        tool = obj(t)
        if tool is None:
            continue
        if "input_schema" not in tool:
            raise Unsupported("server tools are not supported by the free profile")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
        )
    if tools:
        out["tools"] = tools
        choice = obj(body.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = _tool_choice(choice)
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _arguments(raw: object) -> Obj:
    if isinstance(raw, str):
        try:
            parsed = obj(json.loads(raw)) if raw else {}
        except ValueError:
            return {"_unparsed": raw}
        return parsed if parsed is not None else {"_unparsed": raw}
    return obj(raw) or {}


def from_openai_response(resp: Obj, model: str) -> Obj:
    choice = obj((arr(resp.get("choices")) or [None])[0]) or {}
    message = obj(choice.get("message")) or {}
    content: list[Obj] = []
    body_text = text(message.get("content"))
    if body_text:
        content.append({"type": "text", "text": body_text})
    for c in arr(message.get("tool_calls")) or []:
        call = obj(c) or {}
        fn = obj(call.get("function")) or {}
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "input": _arguments(fn.get("arguments")),
            }
        )
    usage = obj(resp.get("usage")) or {}
    return {
        "id": str(resp.get("id", "msg_free")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _FINISH_TO_STOP.get(str(choice.get("finish_reason")), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": num(usage.get("prompt_tokens")),
            "output_tokens": num(usage.get("completion_tokens")),
        },
    }


class OpenAIStreamTranslator:
    """Turns OpenAI chunks into Anthropic ``(event, data)`` pairs. Call ``finish`` at the end."""

    def __init__(self, model: str, estimated_input_tokens: int) -> None:
        self._model = model
        self._est_in = estimated_input_tokens
        self._started = False
        self._open_text: int | None = None
        self._tool_blocks: dict[int, int] = {}  # openai tool index -> anthropic block index
        self._next_block = 0
        self._finish: str | None = None
        self._usage: Obj | None = None
        self._chunks = 0
        self._id = "msg_free"

    def _start(self) -> list[tuple[str, Obj]]:
        if self._started:
            return []
        self._started = True
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self._id,
                        "type": "message",
                        "role": "assistant",
                        "model": self._model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _close_text(self) -> list[tuple[str, Obj]]:
        if self._open_text is None:
            return []
        index, self._open_text = self._open_text, None
        return [("content_block_stop", {"type": "content_block_stop", "index": index})]

    def feed(self, chunk: Obj) -> list[tuple[str, Obj]]:
        out = self._start()
        self._id = str(chunk.get("id", self._id))
        usage = obj(chunk.get("usage"))
        if usage:
            self._usage = usage
        choice = obj((arr(chunk.get("choices")) or [None])[0])
        if choice is None:
            return out
        delta = obj(choice.get("delta")) or {}
        piece = text(delta.get("content"))
        if piece:
            self._chunks += 1
            if self._open_text is None:
                self._open_text = self._next_block
                self._next_block += 1
                out.append(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self._open_text,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            out.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._open_text,
                        "delta": {"type": "text_delta", "text": piece},
                    },
                )
            )
        for c in arr(delta.get("tool_calls")) or []:
            call = obj(c) or {}
            oai_index = num(call.get("index"))
            fn = obj(call.get("function")) or {}
            if oai_index not in self._tool_blocks:
                out.extend(self._close_text())
                block = self._next_block
                self._next_block += 1
                self._tool_blocks[oai_index] = block
                out.append(
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block,
                            "content_block": {
                                "type": "tool_use",
                                "id": str(call.get("id", f"call_{oai_index}")),
                                "name": str(fn.get("name", "")),
                                "input": {},
                            },
                        },
                    )
                )
            fragment = text(fn.get("arguments"))
            if fragment:
                self._chunks += 1
                out.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._tool_blocks[oai_index],
                            "delta": {"type": "input_json_delta", "partial_json": fragment},
                        },
                    )
                )
        reason = text(choice.get("finish_reason"))
        if reason:
            self._finish = reason
        return out

    def finish(self) -> list[tuple[str, Obj]]:
        out = self._start()
        out.extend(self._close_text())
        for block in self._tool_blocks.values():
            out.append(("content_block_stop", {"type": "content_block_stop", "index": block}))
        self._tool_blocks.clear()
        if self._usage is not None:
            in_tok = num(self._usage.get("prompt_tokens"))
            out_tok = num(self._usage.get("completion_tokens"))
        else:  # upstream ignored include_usage: estimate, never report zero
            in_tok, out_tok = self._est_in, self._chunks
        out.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _FINISH_TO_STOP.get(str(self._finish), "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
                },
            )
        )
        out.append(("message_stop", {"type": "message_stop"}))
        return out
