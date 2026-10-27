"""Spend accounting and cap enforcement (§6): per run and per generation.

Units follow the cap: tokens (``free``, input + output) or micro-USD (``claude``).
A request is admitted only if what is already spent, plus what other in-flight
requests may still spend, plus this request's worst-case input estimate and *output*, fits
both caps.
Input is reserved from a conservative size estimate at admit and replaced by actual
usage at settle; an estimate that is too low can overshoot a cap by that call only.
"""

from __future__ import annotations

import threading
from collections.abc import Hashable
from dataclasses import dataclass

from iterant.contracts import Role, RoleCap, RunKey
from iterant.proxy.config import MICROS_PER_MTOK, ModelPrice


@dataclass(frozen=True)
class Usage:
    """Token counts as reported by the upstream. ``in_total`` is what ``ProxyCall.in_tok`` logs."""

    input: int = 0
    output: int = 0
    cache_write: int = 0
    cache_read: int = 0

    @property
    def in_total(self) -> int:
        return self.input + self.cache_write + self.cache_read


def cost_usd_micros(price: ModelPrice, usage: Usage) -> int:
    """Exact integer cost, rounded up once on the total."""
    num = (
        usage.input * price.input_per_mtok
        + usage.output * price.output_per_mtok
        + usage.cache_write * price.cache_write
        + usage.cache_read * price.cache_read
    )
    return -(-num // MICROS_PER_MTOK)


def output_reserve_usd_micros(price: ModelPrice, max_tokens: int) -> int:
    return -(-max_tokens * price.output_per_mtok // MICROS_PER_MTOK)


def estimate_input_tokens(request_bytes: int) -> int:
    """Conservative input-token estimate from the raw request size (bytes / 3, rounded up).

    Real tokenisers average 3-4 bytes per token on text; JSON overhead and base64 only
    lower the count, so this over-reserves. ``settle`` replaces it with actual usage."""
    return -(-request_bytes // 3)


def input_reserve_usd_micros(price: ModelPrice, input_tokens: int) -> int:
    return -(-input_tokens * price.input_per_mtok // MICROS_PER_MTOK)


@dataclass(frozen=True)
class CapExceeded:
    scope: str  # "run" | "gen"
    spent: int
    reserved: int
    asked: int
    limit: int
    unit: str  # "tokens" | "usd_micros"

    @property
    def code(self) -> str:
        return f"cap_exceeded_{self.scope}"

    @property
    def message(self) -> str:
        where = "this run" if self.scope == "run" else "this generation"
        return (
            f"iterant proxy: {self.unit} cap for {where} would be exceeded "
            f"(limit {self.limit}, spent {self.spent}, in flight {self.reserved}, "
            f"this request needs up to {self.asked})"
        )


@dataclass(frozen=True)
class Reservation:
    role: Role
    run_id: Hashable
    gen: int
    units: int


@dataclass(frozen=True)
class RunUsage:
    """Everything the proxy logged for one run: the harness's source for spend and refusals."""

    calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    usd_micros: int = 0
    refusals: int = 0


class Ledger:
    def __init__(self, generation: int = 0) -> None:
        self._lock = threading.Lock()
        self.generation = generation
        self._run_spent: dict[Hashable, int] = {}
        self._run_held: dict[Hashable, int] = {}
        self._gen_spent: dict[tuple[int, Role], int] = {}
        self._gen_held: dict[tuple[int, Role], int] = {}
        self._usage: dict[RunKey, RunUsage] = {}

    def begin_generation(self, gen: int) -> None:
        with self._lock:
            self.generation = gen

    def reserve(
        self, role: Role, run_id: Hashable, cap: RoleCap, units: int
    ) -> Reservation | CapExceeded:
        token_capped = cap.is_token_capped
        per_run = cap.max_tokens_per_run if token_capped else cap.max_usd_micros_per_run
        per_gen = cap.max_tokens_per_gen if token_capped else cap.max_usd_micros_per_gen
        assert per_run is not None and per_gen is not None
        unit = "tokens" if token_capped else "usd_micros"
        with self._lock:
            gkey = (self.generation, role)
            rs, rh = self._run_spent.get(run_id, 0), self._run_held.get(run_id, 0)
            if rs + rh + units > per_run:
                return CapExceeded("run", rs, rh, units, per_run, unit)
            gs, gh = self._gen_spent.get(gkey, 0), self._gen_held.get(gkey, 0)
            if gs + gh + units > per_gen:
                return CapExceeded("gen", gs, gh, units, per_gen, unit)
            self._run_held[run_id] = rh + units
            self._gen_held[gkey] = gh + units
            return Reservation(role, run_id, self.generation, units)

    def preload(
        self,
        role: Role,
        run_id: Hashable,
        units: int,
        key: RunKey | None,
        usage: Usage,
        usd: int,
        refused: bool,
    ) -> None:
        """Count already-spent ``units`` (no reservation involved)."""
        r = Reservation(role, run_id, self.generation, 0)
        self.settle(r, units, key, usage, usd, refused)

    def release(self, r: Reservation) -> None:
        with self._lock:
            self._drop_hold(r)

    def settle(
        self,
        r: Reservation,
        spent_units: int,
        key: RunKey | None,
        usage: Usage,
        usd: int,
        refused: bool,
    ) -> None:
        with self._lock:
            self._drop_hold(r)
            gkey = (r.gen, r.role)
            self._run_spent[r.run_id] = self._run_spent.get(r.run_id, 0) + spent_units
            self._gen_spent[gkey] = self._gen_spent.get(gkey, 0) + spent_units
            if key is not None:
                u = self._usage.get(key, RunUsage())
                self._usage[key] = RunUsage(
                    u.calls + 1,
                    u.in_tok + usage.in_total,
                    u.out_tok + usage.output,
                    u.usd_micros + usd,
                    u.refusals + (1 if refused else 0),
                )

    def run_usage(self, key: RunKey) -> RunUsage:
        with self._lock:
            return self._usage.get(key, RunUsage())

    def _drop_hold(self, r: Reservation) -> None:
        gkey = (r.gen, r.role)
        self._run_held[r.run_id] = self._run_held.get(r.run_id, 0) - r.units
        self._gen_held[gkey] = self._gen_held.get(gkey, 0) - r.units
        if self._run_held[r.run_id] <= 0:
            del self._run_held[r.run_id]


__all__ = [
    "CapExceeded",
    "Ledger",
    "Reservation",
    "RunUsage",
    "Usage",
    "cost_usd_micros",
    "estimate_input_tokens",
    "input_reserve_usd_micros",
    "output_reserve_usd_micros",
]
