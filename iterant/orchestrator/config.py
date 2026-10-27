"""Orchestrator settings, the error types it raises and the seams it is built against."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from iterant.contracts import AcceptanceThresholds, Task
from iterant.grader import GradeResult

# TODO(contract): ARCHITECTURE §2 says /task and /out, but only /work is writable in a sandbox.
TASK_DIR = "/work/task"
OUT_DIR = "/work/out"

# Builds the proposer's failure digest from train trajectories. The argument is every string the
# digest must strip (hidden check paths and contents; the analyst's ``redact`` parameter).
FailureDigest = Callable[[Sequence[str]], str]


class BatchAborted(RuntimeError):
    """The batch cannot continue for an infrastructure reason. Unless it is an
    ``InfraRateExceeded`` (which closes the batch), nothing is recorded against the candidates,
    so the next ``run_generation`` resumes the same batch."""


class InfraRateExceeded(BatchAborted):
    """Too many of the champion arm's runs ended in infra errors. The batch is closed (its
    candidates are rejected with the reason recorded on a ``b<batch>-closed`` marker) and the
    next ``run_generation`` opens a fresh batch of the same generation."""


class BudgetExhausted(RuntimeError):
    """The next run might cost more than the budget has left. Nothing is rejected: the
    generation stops and a later call (with more budget) resumes it."""


class GraderLike(Protocol):
    def preflight(self) -> None: ...

    def grade(self, task: Task, out_files: dict[str, bytes]) -> GradeResult: ...


class ImageSource(Protocol):
    """Turns a champion commit or a candidate diff into a runnable target image.

    The real one (batch 2) wraps ``iterant.target.build_image``; tests use a fake.
    """

    def champion(self, sha: str) -> str:
        """Image of the repo at champion commit ``sha``."""
        ...

    def candidate(self, cand_id: str, parent_sha: str, diff: str) -> str:
        """Image of the repo at ``parent_sha`` with ``diff`` applied."""
        ...


@dataclass(frozen=True)
class OrchestratorConfig:
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    candidates_per_gen: int = 1
    budget_usd_micros: int | None = None  # None: no limit; else a cap on this generation's runs
    run_timeout_s: float = 600.0  # wall-clock limit of one agent run
    cpu: float = 1.0
    mem: int = 2048
    disk: int = 2048
    # close the batch when more of a stage's champion-arm runs than this are infra errors
    max_infra_rate: float = 0.5
    read_budget_s: float = 300.0  # time allowed to read /out once the agent has finished
    seed: int = 0  # mixed into the bootstrap seeds
    task_dir: str = TASK_DIR
    out_dir: str = OUT_DIR

    def __post_init__(self) -> None:
        if self.candidates_per_gen < 1:
            raise ValueError("candidates_per_gen must be >= 1")
        if self.budget_usd_micros is not None and self.budget_usd_micros < 0:
            raise ValueError("budget_usd_micros must be >= 0")
        if self.run_timeout_s <= 0:
            raise ValueError("run_timeout_s must be > 0")
        if self.read_budget_s <= 0:
            raise ValueError("read_budget_s must be > 0")
        if not 0.0 < self.max_infra_rate <= 1.0:
            raise ValueError("max_infra_rate must be within (0, 1]")
