"""Typed errors raised by the archive layer."""

from __future__ import annotations

import sqlite3


class ArchiveError(Exception):
    """Base class for every error this package raises on purpose."""


class AppendOnlyViolation(ArchiveError):
    """An UPDATE, DELETE or re-INSERT hit an append-only table (invariant 6)."""


class DuplicateRowError(AppendOnlyViolation):
    """The row (or its key) already exists; archive rows are inserted once."""


class InvalidTransition(ArchiveError):
    """A candidate change the monotonic triggers forbid.

    Stage moving backwards, any change to a terminal candidate, spend decreasing,
    or an edit to an identity column.
    """


class ForeignKeyViolation(ArchiveError):
    """The row references a batch or candidate that does not exist."""


class ConstraintViolation(ArchiveError):
    """Any other CHECK / NOT NULL / UNIQUE failure."""


class UnknownRow(ArchiveError, LookupError):
    """A write targeted a row that does not exist."""


class SchemaVersionError(ArchiveError):
    """The database was written by a newer Iterant than this one understands."""


class PackHashMismatch(ArchiveError):
    """A batch's stored pack hash differs from the current pack (invariants 6, 7)."""


class ArchiveBusy(ArchiveError):
    """The database is locked by another connection. Retryable."""

    retryable = True


class ArchiveNotInitialized(ArchiveError):
    """The archive file or its schema doesn't exist (and the caller must not create it)."""


class GitRefError(ArchiveError):
    """A git operation on the target repo failed or was refused."""


_TRANSITION_MARKERS = (
    "stage cannot move backwards",
    "spend cannot decrease",
    "terminal candidate is frozen",
    "candidate identity columns are immutable",
)


def translate_operational_error(exc: sqlite3.OperationalError) -> ArchiveError | None:
    """``ArchiveBusy`` for lock contention; None for any other operational error."""
    msg = str(exc).lower()
    if "locked" in msg or "busy" in msg:
        return ArchiveBusy(str(exc))
    return None


def translate_integrity_error(exc: sqlite3.IntegrityError) -> ArchiveError:
    """Map a SQLite trigger / constraint abort to a typed archive error."""
    msg = str(exc)
    if "row exists" in msg:
        return DuplicateRowError(msg)
    if any(marker in msg for marker in _TRANSITION_MARKERS):
        return InvalidTransition(msg)
    if "append-only" in msg:
        return AppendOnlyViolation(msg)
    if "FOREIGN KEY" in msg:
        return ForeignKeyViolation(msg)
    return ConstraintViolation(msg)
