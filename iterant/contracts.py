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

import functools
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
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


# ``allowed``: the agent may execute code while solving a task. ``advisory-forbidden``: the
# pack asks the agent not to, but nothing enforces it (the sandbox cannot tell a script the
# agent wrote from one it ran to solve the task). The gate and the report read this field and
# must present the pack as unenforced; they must not treat it as a guarantee.
CodeExecPolicy = Literal["allowed", "advisory-forbidden"]
CODE_EXEC_POLICIES: Final[tuple[CodeExecPolicy, ...]] = ("allowed", "advisory-forbidden")

_PINNED_IMAGE = re.compile(r"[^\s@:/-][^\s@]*@sha256:[0-9a-f]{64}")


def require_pinned_image(image: str) -> None:
    """Raise ``ValueError`` unless ``image`` is ``name@sha256:<64 lowercase hex>``."""
    if not _PINNED_IMAGE.fullmatch(image):
        raise ValueError(f"grader image must be name@sha256:<64 hex>, got {image!r}")


@dataclass(frozen=True)
class Pack:
    """A task pack. ``content_hash`` is checked before every batch (§1.7).

    ``grader_image`` is the digest-pinned image the grader runs checks in. It comes from
    ``pack.toml`` (so the content hash covers it), never from the target repo: a candidate
    controls the target image and could plant files that change what the grader executes.
    """

    name: str
    version: str
    tasks: tuple[Task, ...]
    content_hash: str
    grader_image: str
    code_exec_policy: CodeExecPolicy = "allowed"

    def __post_init__(self) -> None:
        _require_sha256(self.content_hash, "content_hash")
        require_pinned_image(self.grader_image)
        if self.code_exec_policy not in CODE_EXEC_POLICIES:
            raise ValueError(f"unknown code_exec_policy {self.code_exec_policy!r}")
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique within a pack")


# --- iterant.toml (§2). Types only; the parser lives in target/ --------------


@dataclass(frozen=True)
class TargetModel:
    """``[target.model]``: frozen by the proxy (§1.5).

    Money is integer micro-USD (1 USD = 1_000_000). The toml parser converts the
    file's dollar amount (``max_usd_per_run = 0.10``) to micros. At least one
    per-run cap is set: USD for the ``claude`` profile, tokens for ``free``.
    The proxy derives ``RoleCap[Role.TARGET]`` from this; ``name`` and ``effort``
    must equal that cap's ``model`` and ``effort``.
    """

    name: str
    effort: Effort
    max_usd_micros_per_run: int | None = None
    max_tokens_per_run: int | None = None

    def __post_init__(self) -> None:
        if self.max_usd_micros_per_run is None and self.max_tokens_per_run is None:
            raise ValueError("set max_usd_micros_per_run or max_tokens_per_run")
        for cap in (self.max_usd_micros_per_run, self.max_tokens_per_run):
            if cap is not None and cap <= 0:
                raise ValueError("per-run caps must be > 0")


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
    """``iterant.toml``; ``evaluation`` holds the ``[eval]`` table (``eval`` is a builtin)."""

    target: TargetConfig
    evaluation: EvalConfig


# --- sandbox (§5) ------------------------------------------------------------


@dataclass(frozen=True)
class ExecResult:
    """Outcome of ``Sandbox.exec``. On timeout ``exit_code`` is -1 and ``timed_out`` is True.

    ``truncated`` is True when stdout or stderr hit the backend's capture cap and was cut.
    A truncated result is never trustworthy output: the grader scores it 0.0.
    ``sandbox_lost`` is True when the command made the sandbox unusable (it killed PID 1, or
    could not be killed). That is the candidate's doing: callers score the run 0.0 and must
    not treat the later ``SandboxLost`` from ``put``/``get``/``exec`` on that sandbox as an
    infra error.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    wall_s: float
    truncated: bool = False
    sandbox_lost: bool = False


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
        """Run ``cmd`` via the shell; kill it after ``timeout`` seconds.

        Candidate-caused outcomes are returned, never raised: a timeout, truncated output,
        a non-zero exit, an OOM kill, pids exhaustion, or the command killing PID 1 so the
        sandbox is gone. The result then has a non-zero ``exit_code`` (or
        ``timed_out``/``truncated``) and the caller scores the run 0.0. When the sandbox is
        unusable afterwards the result has ``sandbox_lost`` set and later calls raise
        ``SandboxLost`` (a ``SandboxError`` subclass in ``iterant.sandbox``); that is still
        the candidate's doing, so score 0.0 rather than dropping the run.

        Raise only for failures the command cannot cause: the runtime is missing or broken,
        or the sandbox was already destroyed or expired (a plain ``SandboxError``, not
        ``SandboxLost``). The caller treats such a raise as an infra error (§6: retry once,
        then drop the pair).
        """
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


# --- W2 protocols (§2, §3, §5) ----------------------------------------------
# Types only. The proposer and gate workers implement them.


# Every character ``str.splitlines`` treats as a line boundary (plus the lone "\r" of CRLF).
# ``gitrefs.promote`` rejects them in a commit subject, so reject them at proposal time.
_BAD_HYPOTHESIS_CHARS: Final = frozenset(
    [
        chr(c) for c in range(0x20) if c != 0x09
    ]  # C0 controls except tab: newline, CR, NUL, FS/GS/RS...
    + ["\x7f", "\x85", "\u2028", "\u2029"]
)


@dataclass(frozen=True)
class Proposal:
    """PROPOSE output: a unified diff against the parent plus a one-line hypothesis."""

    diff: str
    hypothesis: str

    def __post_init__(self) -> None:
        if not self.diff.strip():
            raise ValueError("diff must be non-empty (no diff is a PROPOSE kill, not a Proposal)")
        hyp = self.hypothesis
        if not hyp.strip() or hyp != hyp.strip() or any(ch in _BAD_HYPOTHESIS_CHARS for ch in hyp):
            raise ValueError(
                "hypothesis must be a single non-empty line with no surrounding whitespace"
                " or control characters"
            )


@dataclass(frozen=True)
class TriedCandidate:
    """One archive entry shown to the proposer: the candidate (its ``hypothesis``, and its
    ``stage``/``status`` as the outcome) and the diff text it applied."""

    candidate: Candidate
    diff: str


@runtime_checkable
class Proposer(Protocol):
    """Iterant's editing agent (§3 PROPOSE). Sees ``train`` data only (invariant 4)."""

    def propose(
        self,
        parent: str,
        workdir: str | os.PathLike[str],
        failure_digest: str,
        history: Sequence[TriedCandidate],
    ) -> Proposal | None:
        """Propose a change to candidate ``parent``'s genome.

        ``workdir`` is a copy of the target repo at ``parent``; edits are limited to the
        genome. ``failure_digest`` is built from ``train`` trajectories only; ``history`` is
        the archive's tried candidates with their diffs and outcomes. Return None when the
        proposer produced no diff.
        """
        ...


@dataclass(frozen=True)
class GateResult:
    """GATE verdict. ``passed`` holds exactly when there are no ``findings`` (any hit kills)."""

    passed: bool
    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.passed == bool(self.findings):
            raise ValueError("a gate result passes exactly when it has no findings")


@runtime_checkable
class Gate(Protocol):
    """GATE stage (§3): path allowlist, static checks, task-ID/check-path grep, diff audit."""

    def check(
        self,
        proposal: Proposal,
        *,
        repo: str | os.PathLike[str],
        target: TargetConfig,
        pack: Pack,
    ) -> GateResult:
        """Check ``proposal`` against the target repo at ``repo``. ``target.genome`` is the
        path allowlist; ``pack`` supplies the task IDs and check paths to grep for, and its
        ``code_exec_policy``. Raise only for infrastructure failures, not for a bad diff."""
        ...


# --- archive rows (§5) -------------------------------------------------------
# Money is integer micro-USD (``*usd_micros``): exact sums, no float drift.


@dataclass(frozen=True)
class RunKey:
    """Identity of one run attempt: ``(batch, cand, task, rep, attempt)``.

    The resume key is ``(batch, cand, task, rep)``; the highest ``attempt`` wins
    (attempt 1 is the single retry of a refused or infra-errored run, §6). The
    proxy issues a per-RunKey session token to the sandbox, so the real API key
    never enters it (invariant 3).
    """

    batch: int
    cand: str
    task: str
    rep: int
    attempt: int = 0

    def __post_init__(self) -> None:
        if min(self.batch, self.rep, self.attempt) < 0:
            raise ValueError("batch, rep and attempt must be >= 0")


@dataclass(frozen=True)
class Batch:
    """A ``batches`` row: one scheduled set of runs, pinned to a pack hash.

    ``pack_hash`` is checked before every batch (invariant 7). The orchestrator
    must refuse to resume a batch whose stored ``pack_hash`` differs from the
    current pack's ``content_hash``.
    """

    id: int
    gen: int
    pack_hash: str
    started_at: str  # ISO-8601 UTC

    def __post_init__(self) -> None:
        _require_sha256(self.pack_hash, "pack_hash")
        if self.gen < 0:
            raise ValueError("gen must be >= 0")


@dataclass(frozen=True)
class Candidate:
    """A ``candidates`` row. ``parent`` is None only for the seed champion.

    ``stage``, ``status`` and ``usd_micros`` advance monotonically (see DDL).
    """

    id: str
    parent: str | None
    diff_sha: str
    hypothesis: str
    stage: Stage
    status: Status
    usd_micros: int

    def __post_init__(self) -> None:
        _require_sha256(self.diff_sha, "diff_sha")
        if self.usd_micros < 0:
            raise ValueError("usd_micros must be >= 0")
        if self.status is Status.PROMOTED and self.stage is not Stage.PROMOTE:
            raise ValueError("a promoted candidate must be at the promote stage")


@dataclass(frozen=True)
class RunResult:
    """A ``runs`` row. ``score`` is None exactly when the run was refused or errored."""

    key: RunKey
    split: Split
    score: float | None
    in_tok: int
    out_tok: int
    usd_micros: int
    wall_s: float
    refusal: bool
    infra_err: bool

    def __post_init__(self) -> None:
        if (self.score is None) != (self.refusal or self.infra_err):
            raise ValueError("score is None exactly when refusal or infra_err")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be within [0, 1]")
        if min(self.in_tok, self.out_tok, self.usd_micros) < 0 or self.wall_s < 0:
            raise ValueError("token, usd_micros and wall_s values must be >= 0")


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


@dataclass(frozen=True)
class ProxyCall:
    """A ``proxy_calls`` row: one LLM call as logged by the proxy.

    ``key`` attributes the call to a run; it is None for calls that belong to no
    run (proposer, analyst, auditor). Scores and spend come from these rows, never
    from anything the agent reports (invariant 2). Select, holdout and test calls
    are never exposed to the proposer (invariant 4).
    """

    id: int
    key: RunKey | None
    role: Role
    model: str
    in_tok: int
    out_tok: int
    usd_micros: int
    refusal: bool
    created_at: str  # ISO-8601 UTC

    def __post_init__(self) -> None:
        if min(self.in_tok, self.out_tok, self.usd_micros) < 0:
            raise ValueError("token and usd_micros values must be >= 0")


class ArchiveTampered(RuntimeError):
    """The archive's schema, including its append-only triggers, was altered (invariant 6)."""


@functools.cache
def _expected_schema(version: int) -> frozenset[tuple[str, str, str, str]]:
    """(type, name, tbl_name, sql) of every schema object, as SQLite stores it after running
    the DDL for ``version`` in memory."""
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(SCHEMA_DDL_BY_VERSION[version])
        return frozenset(_schema_objects(con))
    finally:
        con.close()


def _schema_objects(con: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
    rows = con.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [(str(t), str(n), str(tn), str(q)) for t, n, tn, q in rows]


def verify_archive_schema(con: sqlite3.Connection) -> None:
    """Raise ``ArchiveTampered`` unless the database's whole schema (every table, index and
    trigger: type, name and SQL) equals the reference DDL for its ``user_version``.

    An extra, missing or altered object is tampering: a dropped or edited append-only trigger,
    a trigger that silently ignores inserts, a stripped CHECK constraint, or every table
    dropped from an initialised archive. A brand-new file (``user_version`` 0, no objects) is
    accepted: the schema is created next. A ``user_version`` newer than this build knows is
    left to the archive layer's ``SchemaVersionError``.

    Detection only: anyone who can write the file can also rewrite rows and restore the
    original schema text (a hash chain over rows is future work).
    """
    row = con.execute("PRAGMA user_version").fetchone()
    version = int(row[0])
    found = _schema_objects(con)
    if version == 0 and not found:
        return
    if version == 0:
        version = max(SCHEMA_DDL_BY_VERSION)  # schema written, version stamp not yet (crash)
    elif version not in SCHEMA_DDL_BY_VERSION:
        return
    expected = _expected_schema(version)
    got = frozenset(found)
    if got == expected:
        return
    exp = {(t, n): q for t, n, _tn, q in expected}
    act = {(t, n): q for t, n, _tn, q in got}
    missing = sorted(f"{t} {n}" for (t, n) in exp.keys() - act.keys())
    extra = sorted(f"{t} {n}" for (t, n) in act.keys() - exp.keys())
    altered = sorted(f"{t} {n}" for (t, n) in exp.keys() & act.keys() if exp[(t, n)] != act[(t, n)])
    parts = [
        f"{label}: {', '.join(v)}"
        for label, v in (("missing", missing), ("extra", extra), ("altered", altered))
        if v
    ]
    raise ArchiveTampered(
        "archive schema does not match the expected one ("
        + ("; ".join(parts) or "table name or sql differs")
        + "); refusing to open a possibly tampered archive"
    )


def open_archive(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open the archive database with foreign keys enforced.

    Raises ``ArchiveTampered`` if an existing archive's schema (tables, indexes,
    append-only triggers) differs from the expected one.

    Every connection to the archive must come from this function: SQLite ignores
    foreign keys unless ``PRAGMA foreign_keys = ON`` is set on each connection.
    It does not create the schema (``ARCHIVE_DDL`` does, via the archive module).
    """
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    if con.execute("PRAGMA foreign_keys").fetchone() != (1,):
        con.close()
        raise RuntimeError("could not enable SQLite foreign key enforcement")
    try:
        verify_archive_schema(con)
    except BaseException:
        con.close()
        raise
    return con


def _in(values: type[StrEnum]) -> str:
    return ", ".join(f"'{v.value}'" for v in values)


def _stage_rank(col: str) -> str:
    whens = " ".join(f"WHEN '{s.value}' THEN {i}" for i, s in enumerate(Stage))
    return f"(CASE {col} {whens} END)"


def _append_only(table: str, dup_cond: str) -> str:
    """Triggers making ``table`` insert-once: no UPDATE, no DELETE, and no INSERT
    (including INSERT OR REPLACE, which skips DELETE triggers) over an existing row."""
    return f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_dup BEFORE INSERT ON {table}
WHEN EXISTS (SELECT 1 FROM {table} WHERE {dup_cond})
BEGIN SELECT RAISE(ABORT, 'archive is append-only: row exists'); END;
CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
"""


# All tables are STRICT (SQLite >= 3.37). Foreign keys are only enforced on a
# connection from ``open_archive``.
BATCHES_DDL: Final = """
CREATE TABLE IF NOT EXISTS batches (
    id        INTEGER PRIMARY KEY,
    gen       INTEGER NOT NULL CHECK (gen >= 0),
    pack_hash TEXT    NOT NULL
        CHECK (length(pack_hash) = 64 AND pack_hash NOT GLOB '*[^0-9a-f]*'),
    started_at TEXT   NOT NULL CHECK (started_at != '')
) STRICT;
""" + _append_only("batches", "id = NEW.id")

# ``stage``, ``status`` and ``usd_micros`` change over time (see PR notes) but only
# monotonically: status leaves ``active`` once and the row is then frozen; stage
# only moves forward in §3 order; spend never decreases. Everything else is fixed.
CANDIDATES_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS candidates (
    id         TEXT PRIMARY KEY,
    parent     TEXT REFERENCES candidates(id),
    diff_sha   TEXT NOT NULL
        CHECK (length(diff_sha) = 64 AND diff_sha NOT GLOB '*[^0-9a-f]*'),
    hypothesis TEXT NOT NULL,
    stage      TEXT NOT NULL CHECK (stage IN ({_in(Stage)})),
    status     TEXT NOT NULL CHECK (status IN ({_in(Status)})),
    usd_micros INTEGER NOT NULL DEFAULT 0 CHECK (usd_micros >= 0),
    CHECK (status != 'promoted' OR stage = 'promote')
) STRICT;
CREATE TRIGGER IF NOT EXISTS candidates_no_dup BEFORE INSERT ON candidates
WHEN EXISTS (SELECT 1 FROM candidates WHERE id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'archive is append-only: row exists'); END;
CREATE TRIGGER IF NOT EXISTS candidates_no_delete BEFORE DELETE ON candidates
BEGIN SELECT RAISE(ABORT, 'archive is append-only'); END;
CREATE TRIGGER IF NOT EXISTS candidates_frozen_cols
BEFORE UPDATE OF id, parent, diff_sha, hypothesis ON candidates
BEGIN SELECT RAISE(ABORT, 'candidate identity columns are immutable'); END;
CREATE TRIGGER IF NOT EXISTS candidates_terminal_frozen BEFORE UPDATE ON candidates
WHEN OLD.status != 'active'
BEGIN SELECT RAISE(ABORT, 'terminal candidate is frozen'); END;
CREATE TRIGGER IF NOT EXISTS candidates_stage_forward BEFORE UPDATE OF stage ON candidates
WHEN {_stage_rank("NEW.stage")} < {_stage_rank("OLD.stage")}
BEGIN SELECT RAISE(ABORT, 'stage cannot move backwards'); END;
CREATE TRIGGER IF NOT EXISTS candidates_usd_monotonic BEFORE UPDATE OF usd_micros ON candidates
WHEN NEW.usd_micros < OLD.usd_micros
BEGIN SELECT RAISE(ABORT, 'spend cannot decrease'); END;
"""

# Resume key is (batch, cand, task, rep); the highest ``attempt`` wins.
RUNS_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS runs (
    batch     INTEGER NOT NULL REFERENCES batches(id),
    cand      TEXT    NOT NULL REFERENCES candidates(id),
    task      TEXT    NOT NULL,
    rep       INTEGER NOT NULL CHECK (rep >= 0),
    attempt   INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    split     TEXT    NOT NULL CHECK (split IN ({_in(Split)})),
    score     REAL    CHECK (score IS NULL OR (score >= 0 AND score <= 1)),
    in_tok    INTEGER NOT NULL CHECK (in_tok >= 0),
    out_tok   INTEGER NOT NULL CHECK (out_tok >= 0),
    usd_micros INTEGER NOT NULL CHECK (usd_micros >= 0),
    wall_s    REAL    NOT NULL CHECK (wall_s >= 0),
    refusal   INTEGER NOT NULL CHECK (refusal IN (0, 1)),
    infra_err INTEGER NOT NULL CHECK (infra_err IN (0, 1)),
    PRIMARY KEY (batch, cand, task, rep, attempt),
    CHECK ((score IS NULL) = (refusal = 1 OR infra_err = 1))
) STRICT;
""" + _append_only(
    "runs",
    "batch = NEW.batch AND cand = NEW.cand AND task = NEW.task"
    " AND rep = NEW.rep AND attempt = NEW.attempt",
)

PROMOTIONS_DDL: Final = """
CREATE TABLE IF NOT EXISTS promotions (
    gen           INTEGER PRIMARY KEY CHECK (gen >= 1),
    cand          TEXT NOT NULL UNIQUE REFERENCES candidates(id),
    select_delta  REAL NOT NULL,
    select_p      REAL NOT NULL CHECK (select_p >= 0 AND select_p <= 1),
    holdout_delta REAL NOT NULL,
    holdout_p     REAL NOT NULL CHECK (holdout_p >= 0 AND holdout_p <= 1)
) STRICT;
""" + _append_only("promotions", "gen = NEW.gen OR cand = NEW.cand")

# ``batch..attempt`` are all NULL (call belongs to no run) or all set. They carry
# no foreign key to ``runs``: a call is logged before its run row exists.
PROXY_CALLS_DDL: Final = f"""
CREATE TABLE IF NOT EXISTS proxy_calls (
    id         INTEGER PRIMARY KEY,
    batch      INTEGER REFERENCES batches(id),
    cand       TEXT REFERENCES candidates(id),
    task       TEXT,
    rep        INTEGER,
    attempt    INTEGER,
    role       TEXT NOT NULL CHECK (role IN ({_in(Role)})),
    model      TEXT NOT NULL,
    in_tok     INTEGER NOT NULL CHECK (in_tok >= 0),
    out_tok    INTEGER NOT NULL CHECK (out_tok >= 0),
    usd_micros INTEGER NOT NULL CHECK (usd_micros >= 0),
    refusal    INTEGER NOT NULL CHECK (refusal IN (0, 1)),
    created_at TEXT NOT NULL CHECK (created_at != ''),
    CHECK ((batch IS NULL) = (cand IS NULL) AND (batch IS NULL) = (task IS NULL)
       AND (batch IS NULL) = (rep IS NULL) AND (batch IS NULL) = (attempt IS NULL))
) STRICT;
""" + _append_only("proxy_calls", "id = NEW.id")

ARCHIVE_DDL: Final = BATCHES_DDL + CANDIDATES_DDL + RUNS_DDL + PROMOTIONS_DDL + PROXY_CALLS_DDL

# Expected schema by ``PRAGMA user_version``; see ``iterant.archive.db.SCHEMA_VERSION``.
SCHEMA_DDL_BY_VERSION: Final[Mapping[int, str]] = {1: ARCHIVE_DDL}

# --- proxy roles and profiles (§6) -------------------------------------------


@dataclass(frozen=True)
class RoleCap:
    """What the proxy allows one role, with hard caps per run and per generation (§6).

    Exactly one pair is set, matching the profile: ``max_tokens_per_run`` and
    ``max_tokens_per_gen`` for ``free``; ``max_usd_micros_per_run`` and
    ``max_usd_micros_per_gen`` for ``claude``. For a role with no runs (proposer,
    analyst, auditor) a "run" is one call.
    """

    model: str
    effort: Effort
    max_tokens_per_run: int | None = None
    max_tokens_per_gen: int | None = None
    max_usd_micros_per_run: int | None = None
    max_usd_micros_per_gen: int | None = None

    def __post_init__(self) -> None:
        tokens = (self.max_tokens_per_run, self.max_tokens_per_gen)
        usd = (self.max_usd_micros_per_run, self.max_usd_micros_per_gen)
        token_set = all(c is not None for c in tokens)
        usd_set = all(c is not None for c in usd)
        if token_set == usd_set or any(c is not None for c in (usd if token_set else tokens)):
            raise ValueError("set exactly one complete pair of per-run and per-gen caps")
        per_run, per_gen = tokens if token_set else usd
        assert per_run is not None and per_gen is not None
        if per_run <= 0 or per_gen < per_run:
            raise ValueError("caps must satisfy 0 < per_run <= per_gen")

    @property
    def is_token_capped(self) -> bool:
        return self.max_tokens_per_run is not None


def _empty_roles() -> Mapping[Role, RoleCap]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ProxyPolicy:
    """Per-profile role table. ``free`` roles need token caps, ``claude`` roles USD caps.

    ``roles`` is copied into a read-only mapping.
    """

    profile: Profile
    roles: Mapping[Role, RoleCap] = field(default_factory=_empty_roles, hash=False)

    def __post_init__(self) -> None:
        wants_tokens = self.profile is Profile.FREE
        for role, cap in self.roles.items():
            if wants_tokens != cap.is_token_capped:
                kind = "token" if wants_tokens else "USD"
                raise ValueError(f"{self.profile} profile needs {kind} caps for role {role}")
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))


# --- acceptance rule (§4) ----------------------------------------------------


@dataclass(frozen=True)
class AcceptanceThresholds:
    """§3/§4 defaults. Deltas are score fractions; ``max_*_increase`` are relative
    (0.15 = +15%).

    Exact comparisons: SELECT passes when ``delta >= delta_min`` and
    ``p < select_p``; CONFIRM passes when ``delta > 0`` and ``p < holdout_p``.
    """

    delta_min: float = 0.05
    select_p: float = 0.05
    holdout_p: float = 0.2
    max_cost_increase: float = 0.15
    max_wall_increase: float = 0.25
    select_tasks: int = 80
    select_reps: int = 2
    holdout_tasks: int = 40
    holdout_reps: int = 2
    holdout_query_warn: int = 20  # warn and ask for fresh holdout tasks after this many
    max_diff_lines: int = 400  # PROPOSE kill rule
    smoke_tasks: int = 5  # SMOKE runs this many train tasks, 1 rep

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
            self.holdout_query_warn,
            self.max_diff_lines,
            self.smoke_tasks,
        )
        if min(counts) < 1:
            raise ValueError("task, rep and limit counts must be >= 1")
