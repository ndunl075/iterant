"""Proxy configuration, pricing and the call-log sink.

The real provider key lives only in :class:`ProxyConfig`, in the proxy process. It is
excluded from ``repr`` and scrubbed from everything the proxy returns or logs.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol

from iterant.contracts import (
    Profile,
    ProxyCall,
    ProxyPolicy,
    Role,
    RoleCap,
    RunKey,
    Split,
    TargetModel,
)

MICROS_PER_MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Micro-USD per million tokens. No live lookups: the caller supplies the table.

    ``cache_write`` / ``cache_read`` default to 1.25x / 0.1x the input price
    (Anthropic's published multipliers) when left None.
    """

    input_per_mtok: int
    output_per_mtok: int
    cache_write_per_mtok: int | None = None
    cache_read_per_mtok: int | None = None

    def __post_init__(self) -> None:
        for v in (
            self.input_per_mtok,
            self.output_per_mtok,
            self.cache_write_per_mtok,
            self.cache_read_per_mtok,
        ):
            if v is not None and v < 0:
                raise ValueError("prices must be >= 0")

    @property
    def cache_write(self) -> int:
        if self.cache_write_per_mtok is not None:
            return self.cache_write_per_mtok
        return -(-self.input_per_mtok * 5 // 4)

    @property
    def cache_read(self) -> int:
        if self.cache_read_per_mtok is not None:
            return self.cache_read_per_mtok
        return -(-self.input_per_mtok // 10)


class CallSink(Protocol):
    """Receives every :class:`ProxyCall`. The archive module supplies the real one."""

    def __call__(self, call: ProxyCall) -> None: ...


class InMemorySink:
    """Test sink: keeps calls in a list."""

    def __init__(self) -> None:
        self.calls: list[ProxyCall] = []
        self._lock = threading.Lock()

    def __call__(self, call: ProxyCall) -> None:
        with self._lock:
            self.calls.append(call)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _no_split(_key: RunKey) -> Split | None:
    return None


def _empty_prices() -> Mapping[str, ModelPrice]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ProxyConfig:
    """Everything the proxy needs. Secrets are ``repr=False``.

    ``claude`` profile: ``anthropic_base_url`` (e.g. ``https://api.anthropic.com``),
    ``anthropic_api_key`` and a price for every policy model. ``free`` profile:
    ``openai_base_url`` including ``/v1`` (e.g. ``http://127.0.0.1:11434/v1``); the key
    is optional (Ollama ignores it).

    ``split_of`` maps a run to its task's split (invariant 4). Unknown or raising means
    "not train": that run's content is never kept.  # TODO(contract): ``RunKey`` has no
    split and the pack lives elsewhere, so the orchestrator injects this lookup.
    """

    policy: ProxyPolicy
    sink: CallSink
    prices: Mapping[str, ModelPrice] = field(default_factory=_empty_prices)
    anthropic_base_url: str | None = None
    anthropic_api_key: str | None = field(default=None, repr=False)
    openai_base_url: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    split_of: Callable[[RunKey], Split | None] = _no_split
    generation: int = 0
    token_ttl_s: float = 3600.0
    max_body_bytes: int = 8 * 1024 * 1024
    max_concurrent: int = 64  # requests in flight; more get a 503
    max_inflight_per_token: int = 4  # calls in flight on one token; more get a 429
    socket_timeout_s: float = 30.0  # client socket timeout per read/write
    request_deadline_s: float = 30.0  # total time to receive a request's headers and body
    max_rejections: int = 1000  # refused requests remembered in memory (oldest dropped)
    upstream_timeout_s: float = 600.0
    trust_env_proxy: bool = True
    first_call_id: int = 1  # ProxyCall.id is process-local; the archive sink should ignore it
    monotonic: Callable[[], float] = time.monotonic
    now_iso: Callable[[], str] = _utc_now_iso

    def __post_init__(self) -> None:
        if self.policy.profile is Profile.CLAUDE:
            if not self.anthropic_base_url or not self.anthropic_api_key:
                raise ValueError("claude profile needs anthropic_base_url and anthropic_api_key")
            missing = sorted({c.model for c in self.policy.roles.values()} - set(self.prices))
            if missing:
                raise ValueError(f"no price for model(s): {', '.join(missing)}")
        elif not self.openai_base_url:
            raise ValueError("free profile needs openai_base_url")
        if self.max_inflight_per_token < 1:
            raise ValueError("max_inflight_per_token must be >= 1")
        object.__setattr__(self, "prices", MappingProxyType(dict(self.prices)))

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.anthropic_api_key, self.openai_api_key) if s)


def target_cap(target: TargetModel, profile: Profile, max_per_gen: int) -> RoleCap:
    """The ``RoleCap`` for ``Role.TARGET`` derived from ``[target.model]`` (§1.5).

    ``max_per_gen`` is micro-USD for ``claude``, tokens for ``free``.
    """
    if profile is Profile.FREE:
        if target.max_tokens_per_run is None:
            raise ValueError("free profile needs max_tokens_per_run on the target model")
        return RoleCap(
            model=target.name,
            effort=target.effort,
            max_tokens_per_run=target.max_tokens_per_run,
            max_tokens_per_gen=max_per_gen,
        )
    if target.max_usd_micros_per_run is None:
        raise ValueError("claude profile needs max_usd_micros_per_run on the target model")
    return RoleCap(
        model=target.name,
        effort=target.effort,
        max_usd_micros_per_run=target.max_usd_micros_per_run,
        max_usd_micros_per_gen=max_per_gen,
    )


RUN_ROLES = frozenset({Role.TARGET, Role.TRANSFER})
CALL_ROLES = frozenset({Role.PROPOSER, Role.ANALYST, Role.AUDITOR})
