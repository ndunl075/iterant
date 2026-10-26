"""Archive: the SQLite data-access layer and the git refs Iterant writes (§5)."""

from iterant.archive.db import (
    ARCHIVE_RELPATH,
    SCHEMA_VERSION,
    Archive,
    ArchiveProxyLog,
    ProxyCallSink,
    RunSlot,
    archive_path,
)
from iterant.archive.errors import (
    AppendOnlyViolation,
    ArchiveError,
    ConstraintViolation,
    DuplicateRowError,
    ForeignKeyViolation,
    GitRefError,
    InvalidTransition,
    PackHashMismatch,
    SchemaVersionError,
    UnknownRow,
)
from iterant.archive.gitrefs import (
    CHAMPION_BRANCH,
    champion_sha,
    diff_sha256,
    ensure_seed,
    gen_tag,
    gen_tag_sha,
    promote,
)

__all__ = [
    "ARCHIVE_RELPATH",
    "CHAMPION_BRANCH",
    "SCHEMA_VERSION",
    "AppendOnlyViolation",
    "Archive",
    "ArchiveError",
    "ArchiveProxyLog",
    "ConstraintViolation",
    "DuplicateRowError",
    "ForeignKeyViolation",
    "GitRefError",
    "InvalidTransition",
    "PackHashMismatch",
    "ProxyCallSink",
    "RunSlot",
    "SchemaVersionError",
    "UnknownRow",
    "archive_path",
    "champion_sha",
    "diff_sha256",
    "ensure_seed",
    "gen_tag",
    "gen_tag_sha",
    "promote",
]
