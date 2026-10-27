"""The generation's $ budget, fed by what the proxy logged (invariant 2)."""

from __future__ import annotations

from dataclasses import dataclass

from iterant.orchestrator.config import BudgetExhausted


@dataclass
class Budget:
    """``spent`` is micro-USD the proxy attributed to this generation's runs.

    ``limit`` None means unlimited. ``check`` is called before each run with the most that
    run may cost (the target's per-run cap, which the proxy enforces): a run that could push
    spend over the limit is not started.
    """

    limit: int | None
    spent: int = 0

    def check(self, worst_case: int) -> None:
        if self.limit is not None and self.spent + worst_case > self.limit:
            raise BudgetExhausted(
                f"budget {self.limit} micro-USD: spent {self.spent}, next run may cost {worst_case}"
            )

    def charge(self, usd_micros: int) -> None:
        if usd_micros < 0:
            raise ValueError("cannot charge a negative amount")
        self.spent += usd_micros
