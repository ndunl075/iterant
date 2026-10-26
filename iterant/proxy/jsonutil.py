"""Typed accessors for untyped JSON, so strict pyright stays quiet without ``Any`` leaking."""

from __future__ import annotations

from typing import Any, cast

Obj = dict[str, Any]


def obj(value: object) -> Obj | None:
    """``value`` as a JSON object, or None."""
    return cast(Obj, value) if isinstance(value, dict) else None


def arr(value: object) -> list[Any] | None:
    """``value`` as a JSON array, or None."""
    return cast(list[Any], value) if isinstance(value, list) else None


def num(value: object, default: int = 0) -> int:
    """``value`` as a non-negative int (bools and non-numbers give ``default``)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(int(value), 0)


def text(value: object) -> str | None:
    return value if isinstance(value, str) else None
