"""``_connect``: immutable open first, plain read-only fallback, first error chained."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from iterant.archive import Archive
from iterant.report import model


@pytest.fixture
def db(tmp_path: Path) -> Path:
    with Archive.open(tmp_path):
        pass
    return tmp_path / ".iterant" / "archive.db"


class Proxy:
    """A connection whose statements can be made to fail (sqlite3.Connection is final)."""

    def __init__(self, con: sqlite3.Connection, fail_on: str | None) -> None:
        self._con = con
        self._fail_on = fail_on

    def execute(self, sql: str, *a: Any) -> Any:
        if self._fail_on and sql.startswith(self._fail_on):
            raise sqlite3.DatabaseError(f"{self._fail_on} failed")
        return self._con.execute(sql, *a)

    def close(self) -> None:
        self._con.close()


def patch_connect(monkeypatch: pytest.MonkeyPatch, fail: dict[str, str | None]) -> list[str]:
    real = sqlite3.connect
    seen: list[str] = []

    def connect(target: str, *a: Any, **k: Any) -> Any:
        seen.append(target)
        key = "immutable" if "immutable=1" in target else "plain"
        if fail.get(key) == "connect":
            raise sqlite3.DatabaseError(f"{key} connect failed")
        return Proxy(real(target, *a, **k), fail.get(key))

    monkeypatch.setattr(model.sqlite3, "connect", connect)
    return seen


@pytest.mark.parametrize("stage", ["connect", "BEGIN", "SELECT"])
def test_an_immutable_failure_at_any_stage_falls_back_to_plain_read_only(
    db: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    seen = patch_connect(monkeypatch, {"immutable": stage})
    con = model._connect(db)  # pyright: ignore[reportPrivateUsage]
    con.close()
    assert len(seen) == 2 and "immutable=1" in seen[0] and seen[1].endswith("mode=ro")


def test_when_both_opens_fail_the_first_error_is_chained(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_connect(monkeypatch, {"immutable": "BEGIN", "plain": "SELECT"})
    with pytest.raises(sqlite3.DatabaseError, match="SELECT failed") as info:
        model._connect(db)  # pyright: ignore[reportPrivateUsage]
    cause = info.value.__cause__
    assert isinstance(cause, sqlite3.DatabaseError) and "BEGIN failed" in str(cause)
