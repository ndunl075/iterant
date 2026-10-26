"""Deterministic, tag-stratified split assignment (§2, §4).

No ``random`` module: every task gets a key from ``sha256(seed, id)``, so the
result depends only on the seed and the task set, on any Python version.

Method: tasks are grouped into strata by their full sorted tag set. Within a
stratum they are ordered by hash; the i-th of m gets position ``(i + o) / m``
where ``o`` is a per-stratum hash offset in [0, 1). Sorting all tasks by that
position interleaves the strata evenly, and cutting the sequence into
select / holdout / test / train blocks gives every block a near-proportional
share of each stratum.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from iterant.contracts import AcceptanceThresholds, Split
from iterant.packs.hashing import PackError

SPLITS_FILE = "splits.json"
DEFAULT_TEST_FRACTION = 0.15
# Used when the pack is too small for the configured select/holdout sizes.
_FALLBACK_FRACTIONS = {Split.SELECT: 0.35, Split.HOLDOUT: 0.20, Split.TEST: 0.15}
MIN_TASKS = 4


@dataclass(frozen=True)
class SplitReport:
    """What the splitter did. ``honored`` is False when the pack was too small for the
    configured select/holdout sizes and proportions were used instead."""

    n_tasks: int
    counts: Mapping[Split, int]
    requested_select: int
    requested_holdout: int
    honored: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    seed: int
    assignment: Mapping[str, Split]
    report: SplitReport


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _counts(n: int, t: AcceptanceThresholds, test_fraction: float) -> tuple[dict[Split, int], bool]:
    test_n = max(1, math.ceil(test_fraction * n))
    if n - t.select_tasks - t.holdout_tasks - test_n >= t.smoke_tasks:
        return {
            Split.SELECT: t.select_tasks,
            Split.HOLDOUT: t.holdout_tasks,
            Split.TEST: test_n,
            Split.TRAIN: n - t.select_tasks - t.holdout_tasks - test_n,
        }, True
    counts = {s: max(1, round(f * n)) for s, f in _FALLBACK_FRACTIONS.items()}
    counts[Split.TRAIN] = n - sum(counts.values())
    while counts[Split.TRAIN] < 1:  # tiny packs: take from the largest other split
        biggest = max(_FALLBACK_FRACTIONS, key=lambda s: counts[s])
        counts[biggest] -= 1
        counts[Split.TRAIN] += 1
    return counts, False


def assign_splits(
    tasks: Sequence[tuple[str, tuple[str, ...]]],
    thresholds: AcceptanceThresholds | None = None,
    seed: int = 0,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> SplitPlan:
    """Assign each ``(task_id, tags)`` to a split. Pure and deterministic."""
    t = thresholds or AcceptanceThresholds()
    ids = [i for i, _ in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    if len(ids) < MIN_TASKS:
        raise ValueError(f"need at least {MIN_TASKS} tasks to fill every split, got {len(ids)}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be within (0, 1)")

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for task_id, tags in tasks:
        strata[tuple(sorted(set(tags)))].append(task_id)
    keyed: list[tuple[float, str]] = []
    for stratum, members in strata.items():
        label = json.dumps(stratum)
        offset = _unit(str(seed), "stratum", label)
        members.sort(key=lambda m: (_unit(str(seed), "task", m), m))
        keyed.extend(((i + offset) / len(members), m) for i, m in enumerate(members))
    keyed.sort()
    order = [m for _, m in keyed]

    counts, honored = _counts(len(order), t, test_fraction)
    assignment: dict[str, Split] = {}
    pos = 0
    for split in (Split.SELECT, Split.HOLDOUT, Split.TEST, Split.TRAIN):
        for task_id in order[pos : pos + counts[split]]:
            assignment[task_id] = split
        pos += counts[split]

    warnings: list[str] = []
    if not honored:
        warnings.append(
            f"pack has {len(order)} tasks, too few for select={t.select_tasks}, "
            f"holdout={t.holdout_tasks} (+test, +{t.smoke_tasks} train); "
            f"used proportions instead: {', '.join(f'{s.value}={counts[s]}' for s in Split)}"
        )
    report = SplitReport(
        n_tasks=len(order),
        counts=dict(counts),
        requested_select=t.select_tasks,
        requested_holdout=t.holdout_tasks,
        honored=honored,
        warnings=tuple(warnings),
    )
    return SplitPlan(seed=seed, assignment=assignment, report=report)


def dump_splits(plan: SplitPlan) -> bytes:
    """Canonical bytes of ``splits.json``: sorted keys, fixed indent, LF, trailing newline."""
    body = {
        "seed": plan.seed,
        "assignment": {k: v.value for k, v in sorted(plan.assignment.items())},
    }
    return (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf-8")


def parse_splits(data: bytes) -> dict[str, Split]:
    try:
        raw = json.loads(data)
        assignment = raw["assignment"]
        if not isinstance(assignment, dict):
            raise TypeError("assignment must be an object")
        return {str(k): Split(v) for k, v in assignment.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    except (ValueError, KeyError, TypeError) as exc:
        raise PackError(f"bad {SPLITS_FILE}: {exc}") from exc


def write_splits(root: Path, plan: SplitPlan) -> None:
    """Persist ``plan`` as ``<root>/splits.json`` (hashed with the pack)."""
    (root / SPLITS_FILE).write_bytes(dump_splits(plan))
