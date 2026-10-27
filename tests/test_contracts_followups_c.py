"""W3 follow-ups C: stat-table DDL in the schema check, and ``clean_hypothesis`` edge cases."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iterant.contracts import (
    _SQLITE_STAT_DDL,  # pyright: ignore[reportPrivateUsage]
    ARCHIVE_DDL,
    ArchiveTampered,
    Proposal,
    clean_hypothesis,
    open_archive,
)
from iterant.proposer.agent import MAX_HYPOTHESIS_CHARS


def _initialised(tmp_path: Path) -> Path:
    path = tmp_path / "a.db"
    con = sqlite3.connect(path)
    con.executescript(ARCHIVE_DDL)
    con.execute("PRAGMA user_version = 1")
    con.commit()
    con.close()
    return path


def test_analyze_stat_tables_are_accepted(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    con = sqlite3.connect(path)
    con.execute("ANALYZE")
    con.commit()
    con.close()
    open_archive(path).close()


def test_a_rewritten_stat_table_is_flagged(tmp_path: Path) -> None:
    path = _initialised(tmp_path)
    con = sqlite3.connect(path, isolation_level=None)
    con.execute("ANALYZE")
    con.execute("PRAGMA writable_schema=1")
    con.execute(
        "UPDATE sqlite_master SET sql = 'CREATE TABLE sqlite_stat1(tbl,idx,stat,evil)'"
        " WHERE name = 'sqlite_stat1'"
    )
    con.execute("PRAGMA writable_schema=0")
    con.close()
    with pytest.raises(ArchiveTampered, match="sqlite_stat1"):
        open_archive(path)


def test_stat_table_canonical_form_matches_what_sqlite_writes() -> None:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t(a)")
    con.execute("INSERT INTO t VALUES (1)")
    con.execute("CREATE INDEX i ON t(a)")
    con.execute("ANALYZE")
    made = dict(con.execute("SELECT name, sql FROM sqlite_master WHERE name LIKE 'sqlite_stat%'"))
    con.close()
    assert made
    assert all(_SQLITE_STAT_DDL[n] == q for n, q in made.items())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "unspecified change"),
        ("   \t \n ", "unspecified change"),
        ("\x7f", "unspecified change"),
        ("\x00", "unspecified change"),
        ("a\x7fb", "a b"),
        ("a\x85b", "a b"),
        ("a b", "a b"),
        ("a\x00b", "a b"),
        ("  a   b  ", "a b"),
    ],
)
def test_clean_hypothesis(raw: str, expected: str) -> None:
    out = clean_hypothesis(raw)
    assert out == expected
    Proposal(diff="x", hypothesis=out)  # always a valid hypothesis


def test_clean_hypothesis_with_the_300_char_cap() -> None:
    assert MAX_HYPOTHESIS_CHARS == 300
    assert clean_hypothesis("x" * 300) == "x" * 300
    # the proposer cuts at the cap and cleans again: a cut that ends in a control character
    # or whitespace must still give a valid hypothesis
    raw = "x" * 299 + "\x00" + "tail"
    out = clean_hypothesis(clean_hypothesis(raw)[:MAX_HYPOTHESIS_CHARS])
    assert out == "x" * 299
    Proposal(diff="x", hypothesis=out)
