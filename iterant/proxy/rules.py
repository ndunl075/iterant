"""Request rules (§6, invariants 3-5): what a role may ask for.

Pure functions over the parsed request body. The model and effort must equal the
role's cap, so a candidate cannot buy gains with a bigger model (invariant 5).

Target and transfer traffic runs candidate code, so it gets an explicit allowlist
(``check_restricted``): no server tools, MCP servers, remote image/document sources,
speed tiers or beta headers, and never fallbacks or rerouting. Anything else is a 400.
Proposer, analyst and auditor requests are Iterant's own code and pass through as sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from iterant.contracts import Role, RoleCap
from iterant.proxy.jsonutil import Obj, arr, num, obj

# Roles whose traffic must never be rerouted or given fallbacks (§6).
NO_FALLBACK_ROLES = frozenset({Role.TARGET, Role.TRANSFER})


@dataclass(frozen=True)
class Rejection:
    """A request the proxy refuses. ``status`` and ``kind`` follow Anthropic's error shape."""

    status: int
    kind: str  # Anthropic error type, e.g. "permission_error"
    code: str  # stable iterant code, e.g. "model_not_allowed"
    message: str


_ALLOWED_KEYS = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "stop_sequences",
        "stream",
        "tools",
        "tool_choice",
        "metadata",
        "output_config",
        "thinking",
    }
)
_ALLOWED_BLOCKS = frozenset(
    {"text", "image", "document", "tool_use", "tool_result", "thinking", "redacted_thinking"}
)


def _bad(code: str, message: str) -> Rejection:
    return Rejection(400, "invalid_request_error", code, f"iterant proxy: {message}")


def _check_blocks(content: object, where: str, *, allow_none: bool = False) -> Rejection | None:
    """``content`` must be a string or a list of block objects, all allowed. Fails closed:
    any other shape is a 400, never skipped."""
    if content is None and allow_none or isinstance(content, str):
        return None
    items = arr(content)
    if items is None:
        return _bad("malformed", f"{where} must be a string or a list of content blocks")
    for b in items:
        block = obj(b)
        if block is None:
            return _bad("malformed", f"{where} must contain only content block objects")
        kind = block.get("type")
        if kind not in _ALLOWED_BLOCKS:
            return _bad("block_not_allowed", f"content block type {kind!r} is not allowed")
        if kind in ("image", "document"):
            source = obj(block.get("source"))
            if source is None:
                return _bad("malformed", f"{kind} block needs a source object")
            if source.get("type") != "base64":
                return _bad(
                    "source_not_allowed",
                    f"{kind} source type {source.get('type')!r} is not allowed (base64 only)",
                )
        if kind == "tool_result":
            nested = _check_blocks(block.get("content"), "tool_result content", allow_none=True)
            if nested is not None:
                return nested
    return None


def check_restricted(body: Obj, beta_header: str) -> Rejection | None:
    """Allowlist for target and transfer requests (see module docstring).

    The proxy is trusted base, so this fails closed: wherever a string, list or object
    is expected, any other shape is rejected rather than skipped."""
    if beta_header.strip():
        return _bad("beta_not_allowed", "anthropic-beta headers are not allowed for this role")
    for key in body:
        if key not in _ALLOWED_KEYS:
            return _bad("field_not_allowed", f"request field {key!r} is not allowed for this role")
    if "output_config" in body:
        config = obj(body["output_config"])
        if config is None or set(config) - {"effort"}:
            return _bad("field_not_allowed", "output_config may only set effort for this role")
    if "metadata" in body and obj(body["metadata"]) is None:
        return _bad("malformed", "metadata must be an object")
    if "stop_sequences" in body:
        stops = arr(body["stop_sequences"])
        if stops is None or not all(isinstance(x, str) for x in stops):
            return _bad("malformed", "stop_sequences must be a list of strings")
    if "stream" in body and not isinstance(body["stream"], bool):
        return _bad("malformed", "stream must be a boolean")
    thinking = body.get("thinking")
    if thinking is not None and thinking != {"type": "adaptive"}:
        return _bad("field_not_allowed", 'thinking must be omitted or {"type": "adaptive"}')
    if "tools" in body:
        tools = arr(body["tools"])
        if tools is None:
            return _bad("malformed", "tools must be a list")
        for t in tools:
            tool = obj(t)
            if tool is None:
                return _bad("malformed", "tools must contain only objects")
            if tool.get("type", "custom") != "custom" or "input_schema" not in tool:
                return _bad(
                    "tool_not_allowed", "only client-defined tools are allowed (no server tools)"
                )
    if "tool_choice" in body:
        choice = obj(body["tool_choice"])
        if choice is None or choice.get("type") not in ("auto", "none"):
            return _bad("field_not_allowed", "tool_choice must be auto or none for this role")
    rejection = _check_blocks(body.get("system"), "system", allow_none=True)
    if rejection is not None:
        return rejection
    messages = arr(body.get("messages"))
    if messages is None:
        return _bad("malformed", "messages must be a list")
    for m in messages:
        msg = obj(m)
        if msg is None:
            return _bad("malformed", "messages must contain only objects")
        rejection = _check_blocks(msg.get("content"), "message content")
        if rejection is not None:
            return rejection
    return None


def check_request(role: Role, cap: RoleCap, body: Obj, beta_header: str) -> Rejection | None:
    model = body.get("model")
    if model != cap.model:
        return Rejection(
            403,
            "permission_error",
            "model_not_allowed",
            f"iterant proxy: role {role} is frozen to model {cap.model!r}, "
            f"request asked for {model!r}",
        )
    output_config = obj(body.get("output_config"))
    effort = output_config.get("effort") if output_config else None
    if output_config is not None and "effort" in output_config and effort != cap.effort:
        return Rejection(
            403,
            "permission_error",
            "effort_not_allowed",
            f"iterant proxy: role {role} is frozen to effort {cap.effort!r}, "
            f"request asked for {effort!r}",
        )
    if role in NO_FALLBACK_ROLES and ("fallbacks" in body or "fallback" in beta_header.lower()):
        return Rejection(
            403,
            "permission_error",
            "fallback_forbidden",
            f"iterant proxy: role {role} traffic is never rerouted; fallbacks are not allowed",
        )
    if role in NO_FALLBACK_ROLES:
        restricted = check_restricted(body, beta_header)
        if restricted is not None:
            return restricted
    if num(body.get("max_tokens")) < 1:
        return Rejection(
            400,
            "invalid_request_error",
            "max_tokens_required",
            "iterant proxy: max_tokens must be a positive integer",
        )
    return None


def with_effort(body: Obj, cap: RoleCap) -> Obj:
    """Copy of ``body`` whose ``output_config.effort`` is pinned to the cap's effort."""
    out = dict(body)
    config = dict(obj(out.get("output_config")) or {})
    config["effort"] = cap.effort
    out["output_config"] = config
    return out
