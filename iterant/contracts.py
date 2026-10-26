"""Shared types and constants for every Iterant module.

Types and constants only: no behavior beyond trivial validation in
``__post_init__``. Stdlib only. Section numbers refer to ARCHITECTURE.md.

Conventions
-----------
* Scores are fractions in ``[0, 1]``; "5 pts" is ``0.05``.
* Memory and disk sizes are MiB; durations are seconds; money is USD.
* Digests (diff, pack) are lowercase sha256 hex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final, Literal, NewType, Protocol, runtime_checkable

# --- enums -------------------------------------------------------------------


class Split(StrEnum):
    """Task split (§2). ``train`` content may reach the proposer; the rest never (§1.4)."""

    TRAIN = "train"
    SELECT = "select"
    HOLDOUT = "holdout"
    TEST = "test"


class Stage(StrEnum):
    """Loop stages (§3), in order."""

    PROPOSE = "propose"
    GATE = "gate"
    SMOKE = "smoke"
    SELECT = "select"
    CONFIRM = "confirm"
    PROMOTE = "promote"


class Status(StrEnum):
    """Candidate status. ``stage`` says how far it got, ``status`` how it ended."""

    ACTIVE = "active"  # still moving through the stages
    REJECTED = "rejected"  # killed at ``stage``
    PROMOTED = "promoted"  # reached PROMOTE and became champion
    ERROR = "error"  # infrastructure failure, not a verdict on the candidate


class Role(StrEnum):
    """Proxy roles (§6). The grader has no role: it makes no LLM calls."""

    TARGET = "target"
    PROPOSER = "proposer"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    TRANSFER = "transfer"


class Profile(StrEnum):
    """``free``: one local model, caps in tokens. ``claude``: hosted, caps in USD."""

    FREE = "free"
    CLAUDE = "claude"


Effort = Literal["low", "medium", "high", "xhigh", "max"]
NetMode = Literal["proxy-only", "default-deny"]

# --- tasks and packs (§2) ----------------------------------------------------

# Distinct types so a visible input path can never be passed where the hidden
# check path is expected (pyright strict rejects the mix-up). Both are paths
# inside the pack on the trusted host, not paths inside a sandbox.
VisibleInputPath = NewType("VisibleInputPath", PurePosixPath)
HiddenCheckPath = NewType("HiddenCheckPath", PurePosixPath)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: str, what: str) -> None:
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError(f"{what} must be 64 lowercase hex chars, got {value!r}")


@dataclass(frozen=True)
class Task:
    """One pack task. ``input`` is shown to the agent; only the grader mounts ``check``."""

    id: str
    split: Split
    tags: tuple[str, ...]
    input: VisibleInputPath
    check: HiddenCheckPath

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("task id must be non-empty")
        if PurePosixPath(self.input) == PurePosixPath(self.check):
            raise ValueError("input and check must be different paths")


@dataclass(frozen=True)
class Pack:
    """A task pack. ``content_hash`` is checked before every batch (§1.7)."""

    name: str
    version: str
    tasks: tuple[Task, ...]
    content_hash: str

    def __post_init__(self) -> None:
        _require_sha256(self.content_hash, "content_hash")
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique within a pack")


# --- iterant.toml (§2). Types only; the parser lives in target/ --------------


@dataclass(frozen=True)
class TargetModel:
    """``[target.model]``: frozen by the proxy (§1.5)."""

    name: str
    effort: Effort
    max_usd_per_run: float

    def __post_init__(self) -> None:
        if self.max_usd_per_run <= 0:
            raise ValueError("max_usd_per_run must be > 0")


@dataclass(frozen=True)
class TargetConfig:
    """``[target]``. ``genome`` globs mark the only editable paths (§1.1)."""

    genome: tuple[str, ...]
    image: str  # path to the Containerfile, relative to the target repo
    run: str  # shell command that runs one task
    model: TargetModel

    def __post_init__(self) -> None:
        if not self.genome:
            raise ValueError("genome must list at least one glob")


@dataclass(frozen=True)
class EvalConfig:
    """``[eval]``: ``pack`` is the pack directory, relative to the target repo."""

    pack: str


@dataclass(frozen=True)
class IterantConfig:
    """Whole ``iterant.toml``. ``evaluation`` is the ``[eval]`` table (avoids shadowing ``eval``)."""

    target: TargetConfig
    evaluation: EvalConfig


# --- sandbox (§5) ------------------------------------------------------------


@dataclass(frozen=True)
class ExecResult:
    """Outcome of ``Sandbox.exec``. On timeout ``exit_code`` is -1 and ``timed_out`` is True."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float


@runtime_checkable
class Sandbox(Protocol):
    """A live disposable sandbox. Non-root, no host mounts, egress per ``net`` (§5)."""

    def put(self, path: str, data: bytes) -> None:
        """Write one file at absolute sandbox ``path``, creating parent dirs."""
        ...

    def get(self, path: str) -> bytes:
        """Read one file at absolute sandbox ``path``. Raises ``FileNotFoundError``."""
        ...

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        """Run ``cmd`` via the shell; kill it after ``timeout`` seconds."""
        ...

    def destroy(self) -> None:
        """Tear down. Idempotent."""
        ...


@runtime_checkable
class SandboxBackend(Protocol):
    """Creates sandboxes (podman, firecracker, e2b/modal)."""

    def create(
        self,
        image: str,
        cpu: float,
        mem: int,
        disk: int,
        ttl: float,
        net: NetMode = "proxy-only",
    ) -> Sandbox:
        """Start a sandbox. ``cpu`` is cores, ``mem``/``disk`` are MiB, ``ttl`` is seconds
        after which the backend destroys it regardless of use."""
        ...


# --- archive rows (§5) -------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A ``candidates`` row. ``parent`` is None only for the seed champion."""

    id: str
    parent: str | None
    diff_sha: str
    hypothesis: str
    stage: Stage
    status: Status
    usd: float

    def __post_init__(self) -> None:
        _require_sha256(self.diff_sha, "diff_sha")
        if self.usd < 0:
            raise ValueError("usd must be >= 0")


@dataclass(frozen=True)
class RunResult:
    """A ``runs`` row. ``score`` is None when the run never reached the grader."""

    cand: str
    task: str
    split: Split
    rep: int
    score: float | None
    in_tok: int
    out_tok: int
    usd: float
    wall_s: float
    refusal: bool
    infra_err: bool

    def __post_init__(self) -> None:
        if self.rep < 0:
            raise ValueError("rep must be >= 0")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        if min(self.in_tok, self.out_tok) < 0 or self.usd < 0 or self.wall_s < 0:
            raise ValueError("token, usd and wall_s values must be >= 0")


@dataclass(frozen=True)
class Promotion:
    """A ``promotions`` row: the evidence behind generation ``gen``."""

    gen: int
    cand: str
    select_delta: float
    select_p: float
    holdout_delta: float
    holdout_p: float

    def __post_init__(self) -> None:
        if self.gen < 1:
            raise ValueError("gen must be >= 1")
        if not (0.0 <= self.select_p <= 1.0 and 0.0 <= self.holdout_p <= 1.0):
            raise ValueError("p-values must be within [0, 1]")


def _in(values: type[StrEnum]) -> str:
    return ", ".join(f"'{v.value}'" for v in values)


# Candidates keep ``stage``/``status``/``usd`` as mutable columns (see PR notes);
# every other column is frozen by trigger. ``runs`` and ``promotions`` accept
# INSERT only. Foreign keys need ``PRAGMA foreign_keys = ON`` on the connection.
CANDIDATES_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS candidates (
    id         TEXT PRIMARY KEY,
    parent     TEXT REFERENCES candidates(id),
    diff_sha   TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    stage      TEXT NOT NULL CHECK (stage IN ({_in(Stage)})),
    status     TEXT NOT NULL CHECK (status IN ({_in(Status)})),
    usd        REAL NOT NULL DEFAULT 0 CHECK (usd >= 0)
);
CREATE TRIGGER IF NOT EXISTS candidates_no_delete BEFORE DELETE ON candidates
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
CREATE TRIGGER IF NOT EXISTS candidates_frozen_cols
BEFORE UPDATE OF id, parent, diff_sha, hypothesis ON candidates
BEGIN SELECT RAISE(ABORT, 'candidate identity columns are immutable'); END;
"""

RUNS_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS runs (
    cand      TEXT    NOT NULL REFERENCES candidates(id),
    task      TEXT    NOT NULL,
    split     TEXT    NOT NULL CHECK (split IN ({_in(Split)})),
    rep       INTEGER NOT NULL CHECK (rep >= 0),
    score     REAL    CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    in_tok    INTEGER NOT NULL CHECK (in_tok >= 0),
    out_tok   INTEGER NOT NULL CHECK (out_tok >= 0),
    usd       REAL    NOT NULL CHECK (usd >= 0),
    wall_s    REAL    NOT NULL CHECK (wall_s >= 0),
    refusal   INTEGER NOT NULL CHECK (refusal IN (0, 1)),
    infra_err INTEGER NOT NULL CHECK (infra_err IN (0, 1)),
    PRIMARY KEY (cand, task, rep)
);
CREATE TRIGGER IF NOT EXISTS runs_no_update BEFORE UPDATE ON runs
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
"""

PROMOTIONS_DDL: Final = """
CREATE TABLE IF NOT EXISTS promotions (
    gen           INTEGER PRIMARY KEY CHECK (gen >= 1),
    cand          TEXT NOT NULL REFERENCES candidates(id),
    select_delta  REAL NOT NULL,
    select_p      REAL NOT NULL CHECK (select_p >= 0 AND select_p <= 1),
    holdout_delta REAL NOT NULL,
    holdout_p     REAL NOT NULL CHECK (holdout_p >= 0 AND holdout_p <= 1)
);
CREATE TRIGGER IF NOT EXISTS promotions_no_update BEFORE UPDATE ON promotions
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
CREATE TRIGGER IF NOT EXISTS promotions_no_delete BEFORE DELETE ON promotions
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
"""

ARCHIVE_DDL: Final = CANDIDATES_DDL + RUNS_DDL + PROMOTIONS_DDL

# --- proxy roles and profiles (§6) -------------------------------------------


@dataclass(frozen=True)
class RoleCap:
    """What the proxy allows one role. Exactly one of the two caps is set:
    ``max_tokens`` (in+out, per generation) for ``free``, ``max_usd`` for ``claude``."""

    model: str
    effort: Effort
    max_tokens: int | None = None
    max_usd: float | None = None

    def __post_init__(self) -> None:
        if (self.max_tokens is None) == (self.max_usd is None):
            raise ValueError("set exactly one of max_tokens and max_usd")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.max_usd is not None and self.max_usd <= 0:
            raise ValueError("max_usd must be > 0")


@dataclass(frozen=True)
class ProxyPolicy:
    """Per-profile role table. ``free`` roles need token caps, ``claude`` roles USD caps."""

    profile: Profile
    roles: dict[Role, RoleCap] = field(default_factory=dict[Role, RoleCap])

    def __post_init__(self) -> None:
        wants_tokens = self.profile is Profile.FREE
        for role, cap in self.roles.items():
            if wants_tokens != (cap.max_tokens is not None):
                kind = "max_tokens" if wants_tokens else "max_usd"
                raise ValueError(f"{self.profile} profile needs {kind} for role {role}")


# --- acceptance rule (§4) ----------------------------------------------------


@dataclass(frozen=True)
class AcceptanceThresholds:
    """§4 defaults. Deltas are score fractions; ``max_*_increase`` are relative (0.15 = +15%)."""

    delta_min: float = 0.05
    select_p: float = 0.05
    holdout_p: float = 0.2
    max_cost_increase: float = 0.15
    max_wall_increase: float = 0.25
    select_tasks: int = 80
    select_reps: int = 2
    holdout_tasks: int = 40
    holdout_reps: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.delta_min <= 1.0:
            raise ValueError("delta_min must be within [0, 1]")
        for p in (self.select_p, self.holdout_p):
            if not 0.0 < p <= 1.0:
                raise ValueError("p thresholds must be within (0, 1]")
        if self.max_cost_increase < 0 or self.max_wall_increase < 0:
            raise ValueError("guard increases must be >= 0")
        counts = (
            self.select_tasks,
            self.select_reps,
            self.holdout_tasks,
            self.holdout_reps,
        )
        if min(counts) < 1:
            raise ValueError("task and rep counts must be >= 1")
