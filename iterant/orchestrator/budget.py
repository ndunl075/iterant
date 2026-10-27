"""The generation's $ budget, measured from what the proxy logged (invariant 2)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from iterant.orchestrator.config import BudgetExhausted


@dataclass
class Budget:
    """``limit`` is micro-USD for the whole generation (every batch of it, every role); None
    means unlimited. ``spent`` reads the proxy's archived calls, so it is always current.

    ``check`` runs before each paid step with the most that step may cost (the proxy's per-call
    or per-run cap for its role, which the proxy enforces): a step that could pass the limit is
    not started.
    """

    limit: int | None
    spent: Callable[[], int]

    def check(self, worst_case: int) -> None:
        if self.limit is None:
            return
        spent = self.spent()
        if spent + worst_case > self.limit:
            raise BudgetExhausted(
                f"budget {self.limit} micro-USD: spent {spent}, next step may cost {worst_case}"
            )
