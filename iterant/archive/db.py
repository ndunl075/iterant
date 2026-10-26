"""SQLite data-access layer for the archive (ARCHITECTURE.md §5).

Every connection comes from ``contracts.open_archive`` so foreign keys are
enforced. There are no UPDATE/DELETE paths for append-only tables; the only
updates are the three monotonic candidate transitions (stage forward,
active -> terminal, spend up), and the triggers in ``contracts`` back them up.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, Self

from iterant.archive.errors import (
    PackHashMismatch,
    SchemaVersionError,
    UnknownRow,
    translate_integrity_error,
)
from iterant.contracts import (
    ARCHIVE_DDL,
    Batch,
    Candidate,
    Promotion,
    ProxyCall,
    Role,
    RunKey,
    RunResult,
    Split,
    Stage,
    Status,
    open_archive,
)

ARCHIVE_RELPATH: Final = Path(".iterant") / "archive.db"
# Stored in ``PRAGMA user_version``. Bump it and add a migration step in
# ``_ensure_schema`` when the DDL changes.
SCHEMA_VERSION: Final = 1

RunSlot = tuple[str, str, int]  # (cand, task, rep): the resume key within a batch

_RUN_COLS: Final = (
    "batch, cand, task, rep, attempt, split, score, in_tok, out_tok,"
    " usd_micros, wall_s, refusal, infra_err"
)
_CAND_COLS: Final = "id, parent, diff_sha, hypothesis, stage, status, usd_micros"
_CALL_COLS: Final = (
    "id, batch, cand, task, rep, attempt, role, model, in_tok, out_tok,"
    " usd_micros, refusal, created_at"
)


def archive_path(repo: str | os.PathLike[str]) -> Path:
    """Where the archive lives inside a target repo."""
    return Path(repo) / ARCHIVE_RELPATH


class ProxyCallSink(Protocol):
    """What the proxy plugs in: called once per logged LLM call."""

    def __call__(self, call: ProxyCall) -> None: ...


def _ensure_schema(con: sqlite3.Connection) -> None:
    row: Any = con.execute("PRAGMA user_version").fetchone()
    version = int(row[0])
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"archive schema version {version} is newer than supported {SCHEMA_VERSION}"
        )
    # All statements are IF NOT EXISTS, so re-running is harmless (idempotent, and
    # safe if two processes initialise at once).
    con.executescript(ARCHIVE_DDL)
    if version < SCHEMA_VERSION:
        # WAL lets the proxy's per-call connections write while the orchestrator reads.
        con.execute("PRAGMA journal_mode = WAL")
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


class Archive:
    """One open archive connection. Use as a context manager so it is always closed."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self._con = con

    # --- lifecycle -------------------------------------------------------------

    @classmethod
    def open(cls, repo: str | os.PathLike[str]) -> Self:
        """Open (creating if needed) ``<repo>/.iterant/archive.db``."""
        return cls.open_path(archive_path(repo))

    @classmethod
    def open_path(cls, db_path: str | os.PathLike[str], *, create_schema: bool = True) -> Self:
        path = Path(db_path)
        if create_schema:
            path.parent.mkdir(parents=True, exist_ok=True)
        con = open_archive(path)
        try:
            if create_schema:
                _ensure_schema(con)
        except BaseException:
            con.close()
            raise
        return cls(con)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row: Any = self._con.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    # --- internals -------------------------------------------------------------

    def _write(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Cursor:
        """Run one write in its own transaction, translating trigger aborts."""
        try:
            with self._con:
                return self._con.execute(sql, params)
        except sqlite3.IntegrityError as exc:
            raise translate_integrity_error(exc) from exc

    # --- inserts (append-only) -------------------------------------------------

    def insert_batch(self, batch: Batch) -> None:
        self._write(
            "INSERT INTO batches (id, gen, pack_hash, started_at) VALUES (?, ?, ?, ?)",
            (batch.id, batch.gen, batch.pack_hash, batch.started_at),
        )

    def create_batch(self, gen: int, pack_hash: str, *, started_at: str | None = None) -> Batch:
        """Insert a batch with an auto-assigned id and return it."""
        stamp = started_at or datetime.now(UTC).isoformat()
        Batch(id=0, gen=gen, pack_hash=pack_hash, started_at=stamp)  # validate first
        cur = self._write(
            "INSERT INTO batches (gen, pack_hash, started_at) VALUES (?, ?, ?)",
            (gen, pack_hash, stamp),
        )
        assert cur.lastrowid is not None
        return Batch(id=cur.lastrowid, gen=gen, pack_hash=pack_hash, started_at=stamp)

    def insert_candidate(self, cand: Candidate) -> None:
        self._write(
            f"INSERT INTO candidates ({_CAND_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                cand.id,
                cand.parent,
                cand.diff_sha,
                cand.hypothesis,
                cand.stage.value,
                cand.status.value,
                cand.usd_micros,
            ),
        )

    def insert_run(self, run: RunResult) -> None:
        k = run.key
        self._write(
            f"INSERT INTO runs ({_RUN_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                k.batch,
                k.cand,
                k.task,
                k.rep,
                k.attempt,
                run.split.value,
                run.score,
                run.in_tok,
                run.out_tok,
                run.usd_micros,
                run.wall_s,
                int(run.refusal),
                int(run.infra_err),
            ),
        )

    def insert_promotion(self, promo: Promotion) -> None:
        self._write(
            "INSERT INTO promotions (gen, cand, select_delta, select_p, holdout_delta, holdout_p)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                promo.gen,
                promo.cand,
                promo.select_delta,
                promo.select_p,
                promo.holdout_delta,
                promo.holdout_p,
            ),
        )

    def insert_proxy_call(self, call: ProxyCall) -> int:
        """Insert one proxy call and return its id.

        ``call.id <= 0`` means "assign the next id", so the proxy needn't coordinate ids.
        """
        k = call.key
        cur = self._write(
            f"INSERT INTO proxy_calls ({_CALL_COLS}) VALUES ({', '.join('?' * 13)})",
            (
                call.id if call.id > 0 else None,
                k.batch if k else None,
                k.cand if k else None,
                k.task if k else None,
                k.rep if k else None,
                k.attempt if k else None,
                call.role.value,
                call.model,
                call.in_tok,
                call.out_tok,
                call.usd_micros,
                int(call.refusal),
                call.created_at,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    # --- candidate transitions -------------------------------------------------

    def advance_stage(self, cand_id: str, stage: Stage) -> None:
        """Move an active candidate forward. Re-applying the same stage is a no-op
        (convenient on resume); moving backwards raises ``InvalidTransition``."""
        cur = self._write("UPDATE candidates SET stage = ? WHERE id = ?", (stage.value, cand_id))
        self._require_one(cur, cand_id)

    def finish_candidate(self, cand_id: str, status: Status, *, stage: Stage | None = None) -> None:
        """Active -> terminal (``rejected``/``promoted``/``error``), optionally also
        recording the stage reached. A terminal candidate can't change again."""
        if status is Status.ACTIVE:
            raise ValueError("finish_candidate needs a terminal status")
        cur = self._write(
            "UPDATE candidates SET status = ?, stage = COALESCE(?, stage) WHERE id = ?",
            (status.value, stage.value if stage else None, cand_id),
        )
        self._require_one(cur, cand_id)

    def add_spend(self, cand_id: str, usd_micros: int) -> int:
        """Add ``usd_micros`` (>= 0) to an active candidate's spend; return the new total.

        Record spend before finishing the candidate: terminal rows are frozen.
        """
        if usd_micros < 0:
            raise ValueError("spend can only increase")
        try:
            with self._con:
                row: Any = self._con.execute(
                    "UPDATE candidates SET usd_micros = usd_micros + ? WHERE id = ?"
                    " RETURNING usd_micros",
                    (usd_micros, cand_id),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise translate_integrity_error(exc) from exc
        if row is None:
            raise UnknownRow(f"no candidate {cand_id!r}")
        return int(row[0])

    @staticmethod
    def _require_one(cur: sqlite3.Cursor, cand_id: str) -> None:
        if cur.rowcount != 1:
            raise UnknownRow(f"no candidate {cand_id!r}")

    # --- point lookups ---------------------------------------------------------

    def get_candidate(self, cand_id: str) -> Candidate | None:
        row = self._con.execute(
            f"SELECT {_CAND_COLS} FROM candidates WHERE id = ?", (cand_id,)
        ).fetchone()
        return _candidate(row) if row else None

    def get_batch(self, batch_id: int) -> Batch | None:
        row = self._con.execute(
            "SELECT id, gen, pack_hash, started_at FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        return Batch(id=row[0], gen=row[1], pack_hash=row[2], started_at=row[3]) if row else None

    def get_promotion(self, gen: int) -> Promotion | None:
        row = self._con.execute(
            "SELECT gen, cand, select_delta, select_p, holdout_delta, holdout_p"
            " FROM promotions WHERE gen = ?",
            (gen,),
        ).fetchone()
        return _promotion(row) if row else None

    def latest_promotion(self) -> Promotion | None:
        """The current champion's evidence, or None before the first promotion."""
        row = self._con.execute(
            "SELECT gen, cand, select_delta, select_p, holdout_delta, holdout_p"
            " FROM promotions ORDER BY gen DESC LIMIT 1"
        ).fetchone()
        return _promotion(row) if row else None

    def get_proxy_call(self, call_id: int) -> ProxyCall | None:
        row = self._con.execute(
            f"SELECT {_CALL_COLS} FROM proxy_calls WHERE id = ?", (call_id,)
        ).fetchone()
        return _proxy_call(row) if row else None

    # --- queries ---------------------------------------------------------------

    def latest_runs(self, batch: int) -> dict[RunSlot, RunResult]:
        """Resume view of a batch: for each ``(cand, task, rep)`` the run with the
        highest ``attempt``. A slot is *done* when it appears here; whether a
        refused/errored latest attempt deserves its one retry is the caller's call."""
        rows = self._con.execute(
            f"SELECT {_RUN_COLS} FROM runs r WHERE r.batch = ? AND r.attempt = ("
            " SELECT MAX(attempt) FROM runs WHERE batch = r.batch AND cand = r.cand"
            " AND task = r.task AND rep = r.rep)",
            (batch,),
        ).fetchall()
        out: dict[RunSlot, RunResult] = {}
        for row in rows:
            run = _run(row)
            out[(run.key.cand, run.key.task, run.key.rep)] = run
        return out

    def runs_for_candidate(
        self,
        cand: str,
        *,
        split: Split | None = None,
        batch: int | None = None,
        latest_only: bool = True,
    ) -> list[RunResult]:
        """A candidate's runs, optionally limited to one split and/or batch.

        ``latest_only`` keeps only the highest attempt per ``(batch, task, rep)``,
        which is what statistics should use; pass False for the full history.
        """
        sql = f"SELECT {_RUN_COLS} FROM runs r WHERE r.cand = ?"
        params: list[Any] = [cand]
        if split is not None:
            sql += " AND r.split = ?"
            params.append(split.value)
        if batch is not None:
            sql += " AND r.batch = ?"
            params.append(batch)
        if latest_only:
            sql += (
                " AND r.attempt = (SELECT MAX(attempt) FROM runs WHERE batch = r.batch"
                " AND cand = r.cand AND task = r.task AND rep = r.rep)"
            )
        sql += " ORDER BY r.batch, r.task, r.rep, r.attempt"
        return [_run(row) for row in self._con.execute(sql, params).fetchall()]

    def batch_pack_hash(self, batch_id: int) -> str | None:
        row = self._con.execute(
            "SELECT pack_hash FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def require_pack_hash(self, batch_id: int, current_hash: str) -> None:
        """Raise ``PackHashMismatch`` unless the batch exists and was pinned to
        ``current_hash``. The orchestrator calls this before resuming (invariant 7)."""
        stored = self.batch_pack_hash(batch_id)
        if stored is None:
            raise UnknownRow(f"no batch {batch_id}")
        if stored != current_hash:
            raise PackHashMismatch(
                f"batch {batch_id} was pinned to pack {stored}, current pack is {current_hash}"
            )

    def holdout_query_count(self) -> int:
        """How many batches have touched the holdout split (§4 overfitting control).

        One query = one batch that ran any holdout task, however many candidates,
        reps or resumed attempts it contains.
        """
        row: Any = self._con.execute(
            "SELECT COUNT(DISTINCT batch) FROM runs WHERE split = ?", (Split.HOLDOUT.value,)
        ).fetchone()
        return int(row[0])

    def holdout_warning_due(self, warn_after: int) -> bool:
        """True once ``warn_after`` holdout queries have happened (default 20 in
        ``AcceptanceThresholds.holdout_query_warn``)."""
        return self.holdout_query_count() >= warn_after

    def iter_candidates(self, status: Status | None = None) -> Iterator[Candidate]:
        sql = f"SELECT {_CAND_COLS} FROM candidates"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        for row in self._con.execute(sql + " ORDER BY rowid", params).fetchall():
            yield _candidate(row)


class ArchiveProxyLog:
    """A ``ProxyCallSink`` that writes to the archive.

    It opens a short-lived connection per call, so it is safe to call from any
    thread and never shares a connection with the orchestrator.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        self._path = Path(db_path)

    def __call__(self, call: ProxyCall) -> None:
        with closing(Archive.open_path(self._path, create_schema=False)) as archive:
            archive.insert_proxy_call(call)


# --- row mappers -------------------------------------------------------------


def _candidate(row: Any) -> Candidate:
    return Candidate(
        id=row[0],
        parent=row[1],
        diff_sha=row[2],
        hypothesis=row[3],
        stage=Stage(row[4]),
        status=Status(row[5]),
        usd_micros=row[6],
    )


def _run(row: Any) -> RunResult:
    return RunResult(
        key=RunKey(batch=row[0], cand=row[1], task=row[2], rep=row[3], attempt=row[4]),
        split=Split(row[5]),
        score=row[6],
        in_tok=row[7],
        out_tok=row[8],
        usd_micros=row[9],
        wall_s=row[10],
        refusal=bool(row[11]),
        infra_err=bool(row[12]),
    )


def _promotion(row: Any) -> Promotion:
    return Promotion(
        gen=row[0],
        cand=row[1],
        select_delta=row[2],
        select_p=row[3],
        holdout_delta=row[4],
        holdout_p=row[5],
    )


def _proxy_call(row: Any) -> ProxyCall:
    key = (
        None
        if row[1] is None
        else RunKey(batch=row[1], cand=row[2], task=row[3], rep=row[4], attempt=row[5])
    )
    return ProxyCall(
        id=row[0],
        key=key,
        role=Role(row[6]),
        model=row[7],
        in_tok=row[8],
        out_tok=row[9],
        usd_micros=row[10],
        refusal=bool(row[11]),
        created_at=row[12],
    )
