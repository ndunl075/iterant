from __future__ import annotations

from collections.abc import Callable

import pytest

from iterant.contracts import RunKey, RunResult, Split

MakeRun = Callable[..., RunResult]


@pytest.fixture
def mk() -> MakeRun:
    def make(
        task: str,
        score: float | None,
        *,
        cand: str = "c",
        rep: int = 0,
        attempt: int = 0,
        batch: int = 1,
        split: Split = Split.SELECT,
        usd: int = 1000,
        wall: float = 10.0,
        refusal: bool = False,
        infra: bool = False,
    ) -> RunResult:
        return RunResult(
            key=RunKey(batch, cand, task, rep, attempt),
            split=split,
            score=score,
            in_tok=1,
            out_tok=1,
            usd_micros=usd,
            wall_s=wall,
            refusal=refusal,
            infra_err=infra,
        )

    return make
