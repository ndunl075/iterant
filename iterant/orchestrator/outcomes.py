"""How each run ended, for the SMOKE rule (§3: "any crash or timeout the champion didn't have").

``RunResult`` only says the score; it cannot tell a crash from a wrong answer. This is a small
side store (``.iterant/outcomes.db``, never read by the stats) keyed like ``runs``. A missing row
means no fault was recorded.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from iterant.contracts import RunKey

_DDL = """
CREATE TABLE IF NOT EXISTS outcomes (
    batch INTEGER NOT NULL, cand TEXT NOT NULL, task TEXT NOT NULL,
    rep INTEGER NOT NULL, attempt INTEGER NOT NULL, fault TEXT NOT NULL,
    PRIMARY KEY (batch, cand, task, rep, attempt)
) STRICT;
"""


class Outcomes:
    def __init__(self, repo: str | os.PathLike[str]) -> None:
        self._path = Path(repo) / ".iterant" / "outcomes.db"

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self._path, timeout=5.0)
        con.executescript(_DDL)
        return con

    def record(self, key: RunKey, fault: str | None) -> None:
        """Note how ``key`` ended (``None``: no crash, timeout or lost sandbox). A re-run of the
        same key replaces the note."""
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?, ?, ?)",
                (key.batch, key.cand, key.task, key.rep, key.attempt, fault or ""),
            )

    def faults(self, batch: int, cand: str) -> dict[tuple[str, int], str]:
        """``{(task, rep): fault}`` for the final attempt of each slot that had a fault."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT task, rep, fault FROM outcomes o WHERE batch = ? AND cand = ?"
                " AND fault != ''"
                " AND attempt = (SELECT MAX(attempt) FROM outcomes WHERE batch = o.batch"
                " AND cand = o.cand AND task = o.task AND rep = o.rep)",
                (batch, cand),
            ).fetchall()
        return {(str(r[0]), int(r[1])): str(r[2]) for r in rows}
