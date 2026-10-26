"""``iterant.toml`` parser (§2). Strict: unknown keys and wrong types are errors.

Every error names the key path (``target.model.effort: ...``). Floats are parsed
as ``Decimal`` so ``max_usd_per_run = 0.10`` becomes exactly 100_000 micro-USD.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from decimal import Decimal
from typing import cast, get_args

from iterant.contracts import (
    Effort,
    EvalConfig,
    IterantConfig,
    TargetConfig,
    TargetModel,
)
from iterant.target.genome import PathRejectedError, compile_glob, normalize_relative

_MICROS_PER_USD = 1_000_000


class ConfigError(ValueError):
    """``iterant.toml`` is invalid. The message starts with the offending key path."""


def load_config(path: str | os.PathLike[str]) -> IterantConfig:
    """Read and parse an ``iterant.toml`` file."""
    with open(path, encoding="utf-8") as fh:
        return parse_config(fh.read())


def parse_config(text: str) -> IterantConfig:
    try:
        raw: object = tomllib.loads(text, parse_float=Decimal)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"<toml>: {exc}") from exc
    top = _table(raw, "<root>", {"target", "eval"})
    target = _table(_required(top, "target", "target"), "target", _TARGET_KEYS)
    model = _table(_required(target, "model", "target.model"), "target.model", _MODEL_KEYS)
    evaluation = _table(_required(top, "eval", "eval"), "eval", {"pack"})

    genome = _genome(_required(target, "genome", "target.genome"))
    image = _rel_path(_required(target, "image", "target.image"), "target.image")
    run = _string(_required(target, "run", "target.run"), "target.run")
    pack = _rel_path(_required(evaluation, "pack", "eval.pack"), "eval.pack")

    name = _string(_required(model, "name", "target.model.name"), "target.model.name")
    effort = _string(_required(model, "effort", "target.model.effort"), "target.model.effort")
    if effort not in get_args(Effort):
        raise ConfigError(f"target.model.effort: must be one of {list(get_args(Effort))}")
    usd = (
        _usd_micros(model["max_usd_per_run"], "target.model.max_usd_per_run")
        if "max_usd_per_run" in model
        else None
    )
    tokens = (
        _positive_int(model["max_tokens_per_run"], "target.model.max_tokens_per_run")
        if "max_tokens_per_run" in model
        else None
    )
    if usd is None and tokens is None:
        raise ConfigError(
            "target.model: set max_usd_per_run (claude profile) or max_tokens_per_run (free)"
        )
    return IterantConfig(
        target=TargetConfig(
            genome=genome,
            image=image,
            run=run,
            model=TargetModel(
                name=name,
                effort=cast("Effort", effort),
                max_usd_micros_per_run=usd,
                max_tokens_per_run=tokens,
            ),
        ),
        evaluation=EvalConfig(pack=pack),
    )


_TARGET_KEYS = {"genome", "image", "run", "model"}
_MODEL_KEYS = {"name", "effort", "max_usd_per_run", "max_tokens_per_run"}


def _kind(value: object) -> str:
    return type(value).__name__


def _table(value: object, path: str, allowed: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a table, got {_kind(value)}")
    table = cast("Mapping[str, object]", value)
    prefix = "" if path == "<root>" else path + "."
    for key in sorted(table):
        if key not in allowed:
            raise ConfigError(f"{prefix}{key}: unknown key")
    return table


def _required(table: Mapping[str, object], key: str, path: str) -> object:
    if key not in table:
        raise ConfigError(f"{path}: required")
    return table[key]


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: expected a string, got {_kind(value)}")
    if not value.strip():
        raise ConfigError(f"{path}: must not be empty")
    return value


def _rel_path(value: object, path: str) -> str:
    try:
        return normalize_relative(_string(value, path), what=path)
    except PathRejectedError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _genome(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"target.genome: expected an array, got {_kind(value)}")
    items = cast("list[object]", value)
    if not items:
        raise ConfigError("target.genome: must list at least one glob")
    globs: list[str] = []
    for i, item in enumerate(items):
        path = f"target.genome[{i}]"
        pattern = _string(item, path)
        try:
            compile_glob(pattern)
        except PathRejectedError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        globs.append(pattern)
    return tuple(globs)


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer, got {_kind(value)}")
    if value <= 0:
        raise ConfigError(f"{path}: must be > 0")
    return value


def _usd_micros(value: object, path: str) -> int:
    """Dollars to integer micro-USD, exactly. Finer than 1 micro-USD is an error."""
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        raise ConfigError(f"{path}: expected a number, got {_kind(value)}")
    dec = Decimal(value)
    if not dec.is_finite():
        raise ConfigError(f"{path}: must be finite")
    sign, digits, exponent = dec.as_tuple()
    assert isinstance(exponent, int)  # finite
    coefficient = int("".join(map(str, digits)))
    shift = exponent + 6  # dollars -> micro-USD is a decimal shift by 6
    if shift >= 0:
        micros = coefficient * 10**shift
    else:
        micros, remainder = divmod(coefficient, 10**-shift)
        if remainder:
            raise ConfigError(f"{path}: {value} is finer than 1 micro-USD (0.000001)")
    if sign:
        micros = -micros
    if micros <= 0:
        raise ConfigError(f"{path}: must be > 0")
    return micros
