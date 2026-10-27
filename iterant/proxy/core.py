"""The proxy's decisions, with no HTTP in it: admit a request, settle a call.

``Proxy.admit`` maps a token to ``(role, RunKey | None)``, applies the request rules and
reserves budget. ``Proxy.settle`` turns the upstream's usage into a ``ProxyCall``, writes
it through the injected sink and updates the ledger. ``server.py`` does the networking.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from iterant.contracts import Profile, ProxyCall, Role, RoleCap, RunKey, Split
from iterant.proxy.config import ModelPrice, ProxyConfig
from iterant.proxy.jsonutil import Obj, obj, text
from iterant.proxy.ledger import (
    CapExceeded,
    Ledger,
    Reservation,
    RunUsage,
    Usage,
    cost_usd_micros,
    estimate_input,
    input_reserve_usd_micros,
    output_reserve_usd_micros,
)
from iterant.proxy.rules import Rejection, check_request, with_effort
from iterant.proxy.sessions import Session, SessionRegistry


class LedgerError(RuntimeError):
    """The call sink failed. The proxy then refuses all new calls: no unlogged spend."""


@dataclass(frozen=True)
class Admission:
    token: str
    session: Session
    cap: RoleCap
    body: Obj  # parsed request, effort pinned for the claude profile
    stream: bool
    reservation: Reservation
    request_bytes: int


@dataclass(frozen=True)
class RejectedCall:
    """In-memory record of a refused request (no model call happened, so not a ``ProxyCall``)."""

    role: Role | None
    key: RunKey | None
    code: str
    message: str


@dataclass(frozen=True)
class CallContent:
    """Request and response of one train-split call (invariant 4)."""

    call_id: int
    key: RunKey
    role: Role
    request: Obj
    response: Obj | None


@dataclass
class _Content:
    items: list[CallContent] = field(default_factory=lambda: [])


class Proxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self.sessions = SessionRegistry(config.monotonic, config.token_ttl_s)
        self.ledger = Ledger(config.generation)
        self.rejections: deque[RejectedCall] = deque(maxlen=config.max_rejections)
        self._ids = itertools.count(config.first_call_id)
        self._call_run_ids = itertools.count()
        self._lock = threading.Lock()
        self._content = _Content()
        self._healthy = True
        self._inflight: dict[str, int] = {}  # admitted, unsettled calls per token

    # --- tokens -----------------------------------------------------------------

    def issue_run_token(self, role: Role, key: RunKey, ttl_s: float | None = None) -> str:
        self._require_role(role)
        return self.sessions.issue_run_token(role, key, ttl_s)

    def issue_call_token(self, role: Role, uses: int = 1, ttl_s: float | None = None) -> str:
        self._require_role(role)
        return self.sessions.issue_call_token(role, uses, ttl_s)

    def _require_role(self, role: Role) -> None:
        if role not in self.config.policy.roles:
            raise ValueError(f"role {role} is not in the proxy policy")

    def begin_generation(self, gen: int) -> None:
        self.ledger.begin_generation(gen)

    def check_token(self, token: str | None) -> Rejection | None:
        """Cheap pre-check so unauthenticated callers are refused before their body is read."""
        if token is not None and self.sessions.resolve(token) is not None:
            return None
        r = Rejection(
            401, "authentication_error", "unknown_token", "iterant proxy: unknown or expired token"
        )
        self.rejections.append(RejectedCall(None, None, r.code, r.message))
        return r

    # --- admit ------------------------------------------------------------------

    def admit(
        self, token: str | None, body_bytes: bytes, beta_header: str = ""
    ) -> Admission | Rejection:
        result = self._admit(token, body_bytes, beta_header)
        if isinstance(result, Rejection):
            session = self.sessions.resolve(token)
            self.rejections.append(
                RejectedCall(
                    session.role if session else None,
                    session.key if session else None,
                    result.code,
                    result.message,
                )
            )
        return result

    def _admit(
        self, token: str | None, body_bytes: bytes, beta_header: str
    ) -> Admission | Rejection:
        session = self.sessions.resolve(token)
        if token is None or session is None:
            return Rejection(
                401,
                "authentication_error",
                "unknown_token",
                "iterant proxy: unknown or expired token",
            )
        if not self._healthy:
            return Rejection(
                503,
                "api_error",
                "ledger_failed",
                "iterant proxy: call log failed; refusing new calls",
            )
        cap = self.config.policy.roles.get(session.role)
        if cap is None:
            return Rejection(
                403,
                "permission_error",
                "role_not_allowed",
                f"iterant proxy: role {session.role} has no cap",
            )
        try:
            body = obj(json.loads(body_bytes))
        except (ValueError, RecursionError):
            body = None
        if body is None:
            return Rejection(
                400,
                "invalid_request_error",
                "bad_body",
                "iterant proxy: body must be a JSON object",
            )
        rejection = check_request(session.role, cap, body, beta_header)
        if rejection is not None:
            return rejection
        claude = self.config.policy.profile is Profile.CLAUDE
        if claude:
            body = with_effort(body, cap)
        run_id = (
            (session.role, session.key)
            if session.key is not None
            else ("call", next(self._call_run_ids))
        )
        max_tokens = int(body["max_tokens"])
        # Reserve the worst case: a conservative input estimate plus the full output.
        estimate = estimate_input(len(body_bytes), body)
        est_input = estimate.tokens
        units = (
            est_input + max_tokens
            if cap.is_token_capped
            else input_reserve_usd_micros(self.config.prices[cap.model], est_input, estimate.cached)
            + output_reserve_usd_micros(self.config.prices[cap.model], max_tokens)
        )
        with self._lock:
            if self._inflight.get(token, 0) >= self.config.max_inflight_per_token:
                return Rejection(
                    429,
                    "rate_limit_error",
                    "too_many_in_flight",
                    f"iterant proxy: at most {self.config.max_inflight_per_token} calls "
                    "may be in flight per token",
                )
            self._inflight[token] = self._inflight.get(token, 0) + 1
        reservation = self.ledger.reserve(session.role, run_id, cap, units)
        if isinstance(reservation, CapExceeded):
            self._done(token)
            return Rejection(403, "permission_error", reservation.code, reservation.message)
        if not self.sessions.take_use(token):
            self._done(token)
            self.ledger.release(reservation)
            return Rejection(
                401,
                "authentication_error",
                "unknown_token",
                "iterant proxy: unknown or expired token",
            )
        return Admission(
            token, session, cap, body, body.get("stream") is True, reservation, len(body_bytes)
        )

    def release(self, adm: Admission) -> None:
        """The upstream call failed before any usage: nothing was spent."""
        self._done(adm.token)
        self.ledger.release(adm.reservation)

    def _done(self, token: str) -> None:
        with self._lock:
            left = self._inflight.get(token, 0) - 1
            if left > 0:
                self._inflight[token] = left
            else:
                self._inflight.pop(token, None)

    # --- settle -----------------------------------------------------------------

    def settle(
        self,
        adm: Admission,
        usage: Usage | None,
        refusal: bool,
        response: Obj | None,
    ) -> ProxyCall:
        """Log one completed (or cut-off) call. ``usage`` None means the upstream never said:
        charge the whole reservation rather than let it go uncounted."""
        cap, profile = adm.cap, self.config.policy.profile
        self._done(adm.token)
        if usage is None:
            usage = Usage(output=int(adm.body["max_tokens"]))
        # Price and log what the upstream says it ran (a fallback may differ from the cap).
        model = (text(response.get("model")) if response else None) or cap.model
        usd = cost_usd_micros(self._price(model), usage) if profile is Profile.CLAUDE else 0
        units = usage.in_total + usage.output if cap.is_token_capped else usd
        key = adm.session.key
        self.ledger.settle(adm.reservation, units, key, usage, usd, refusal)
        with self._lock:
            call = ProxyCall(
                id=next(self._ids),
                key=key,
                role=adm.session.role,
                model=model,
                in_tok=usage.in_total,
                out_tok=usage.output,
                usd_micros=usd,
                refusal=refusal,
                created_at=self.config.now_iso(),
            )
            try:
                self.config.sink(call)
            except Exception as exc:
                self._healthy = False
                raise LedgerError("call sink failed") from exc
            if key is not None and self._is_train(key):
                self._content.items.append(CallContent(call.id, key, call.role, adm.body, response))
        return call

    def _price(self, model: str) -> ModelPrice:
        """Price of ``model``; an unlisted model is charged at the dearest listed price."""
        prices = self.config.prices
        if model in prices:
            return prices[model]
        return max(prices.values(), key=lambda p: (p.output_per_mtok, p.input_per_mtok))

    def preload(self, calls: Iterable[ProxyCall]) -> None:
        """Seed spend after a restart from archived calls of the *current generation*.

        Hook for the orchestrator (W2): the ledger is in memory, so a restarted proxy
        would otherwise forget what this generation and these runs already spent.
        """
        for c in calls:
            cap = self.config.policy.roles.get(c.role)
            if cap is None:
                continue
            units = c.in_tok + c.out_tok if cap.is_token_capped else c.usd_micros
            run_id = (c.role, c.key) if c.key is not None else ("call", next(self._call_run_ids))
            usage = Usage(output=c.out_tok, input=c.in_tok)
            self.ledger.preload(c.role, run_id, units, c.key, usage, c.usd_micros, c.refusal)

    # --- queries ----------------------------------------------------------------

    def run_usage(self, key: RunKey) -> RunUsage:
        """Spend, tokens and refusals the proxy itself saw for ``key`` (invariant 2)."""
        return self.ledger.run_usage(key)

    def train_content(self, key: RunKey | None = None) -> list[CallContent]:
        """Call content of train-split runs only (invariant 4). Content of select, holdout
        and test runs is never stored, so it cannot be returned; the split is re-checked
        here as a second guard."""
        with self._lock:
            items = list(self._content.items)
        return [c for c in items if (key is None or c.key == key) and self._is_train(c.key)]

    @property
    def healthy(self) -> bool:
        return self._healthy

    def _is_train(self, key: RunKey) -> bool:
        try:
            return self.config.split_of(key) is Split.TRAIN
        except Exception:
            return False
