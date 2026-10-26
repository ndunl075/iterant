"""Request rules (§6, invariants 3 and 5): what a role may ask for.

Pure functions over the parsed request body. The model and effort must equal the
role's cap, so a candidate cannot buy gains with a bigger model (invariant 5).
Target-agent traffic is never rerouted: any ``fallbacks`` field or fallback beta is
refused outright. Proposer and auditor requests are passed through as sent.
"""

from __future__ import annotations

from dataclasses import dataclass

from iterant.contracts import Role, RoleCap
from iterant.proxy.jsonutil import Obj, num, obj

# Roles whose traffic must never be rerouted or given fallbacks (§6).
NO_FALLBACK_ROLES = frozenset({Role.TARGET, Role.TRANSFER})


@dataclass(frozen=True)
class Rejection:
    """A request the proxy refuses. ``status`` and ``kind`` follow Anthropic's error shape."""

    status: int
    kind: str  # Anthropic error type, e.g. "permission_error"
    code: str  # stable iterant code, e.g. "model_not_allowed"
    message: str


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
