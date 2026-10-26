"""Pair candidate and champion runs by task (§3 SELECT, §6 retries)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from iterant.contracts import RunResult, Split


@dataclass(frozen=True)
class Pairing:
    """Per-task paired gains. ``deltas[i]`` is candidate minus champion for ``task_ids[i]``.

    ``dropped_tasks`` lists tasks removed from both arms: a final run (highest
    attempt) of either arm was refused or infra-errored, or a task was missing from
    one arm. ``candidate_runs`` and ``champion_runs`` are the final attempts of the
    kept tasks only, so guards compare like with like.
    """

    task_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    dropped_tasks: tuple[str, ...]
    candidate_runs: tuple[RunResult, ...]
    champion_runs: tuple[RunResult, ...]

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_tasks)

    @property
    def delta(self) -> float:
        """Mean per-task gain (0.0 when nothing is paired)."""
        return math.fsum(self.deltas) / len(self.deltas) if self.deltas else 0.0


def final_attempts(runs: Iterable[RunResult]) -> dict[tuple[str, int], RunResult]:
    """Latest attempt per ``(task, rep)``; the retry replaces the first try (§1.6, §6).

    Raises ``ValueError`` on a repeated ``(task, rep, attempt)``.
    """
    best: dict[tuple[str, int], RunResult] = {}
    seen: set[tuple[str, int, int]] = set()
    for r in runs:
        k = r.key
        full = (k.task, k.rep, k.attempt)
        if full in seen:
            raise ValueError(f"duplicate run {full}")
        seen.add(full)
        cur = best.get((k.task, k.rep))
        if cur is None or k.attempt > cur.key.attempt:
            best[(k.task, k.rep)] = r
    return best


def _by_task(finals: dict[tuple[str, int], RunResult]) -> dict[str, list[RunResult]]:
    out: dict[str, list[RunResult]] = {}
    for (task, _rep), r in sorted(finals.items()):
        out.setdefault(task, []).append(r)
    return out


def check_same_batch(
    candidate: Sequence[RunResult], champion: Sequence[RunResult], split: Split | None
) -> None:
    """Both arms must come from one batch (provider drift cancels only then) and, when
    ``split`` is given, from that split."""
    batches = {r.key.batch for r in (*candidate, *champion)}
    if len(batches) > 1:
        raise ValueError(f"arms must come from one batch, got batches {sorted(batches)}")
    if split is not None:
        bad = {r.split for r in (*candidate, *champion)} - {split}
        if bad:
            raise ValueError(f"expected only {split.value} runs, got {sorted(map(str, bad))}")


def pair_runs(
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    *,
    split: Split | None = None,
) -> Pairing:
    """Pair the two arms by task, averaging scores over reps.

    A task is dropped from both arms when any final run of either arm is refused or
    infra-errored, or when either arm has no run for it. Raises ``ValueError`` if the
    arms are not from one batch (or not all from ``split``).
    """
    check_same_batch(candidate, champion, split)
    cand = _by_task(final_attempts(candidate))
    champ = _by_task(final_attempts(champion))
    kept: list[str] = []
    deltas: list[float] = []
    dropped: list[str] = []
    cand_kept: list[RunResult] = []
    champ_kept: list[RunResult] = []
    for task in sorted(cand.keys() | champ.keys()):
        c_runs, h_runs = cand.get(task, []), champ.get(task, [])
        if not c_runs or not h_runs or any(r.score is None for r in (*c_runs, *h_runs)):
            dropped.append(task)
            continue
        c_mean = math.fsum(r.score for r in c_runs if r.score is not None) / len(c_runs)
        h_mean = math.fsum(r.score for r in h_runs if r.score is not None) / len(h_runs)
        kept.append(task)
        deltas.append(c_mean - h_mean)
        cand_kept.extend(c_runs)
        champ_kept.extend(h_runs)
    return Pairing(
        task_ids=tuple(kept),
        deltas=tuple(deltas),
        dropped_tasks=tuple(dropped),
        candidate_runs=tuple(cand_kept),
        champion_runs=tuple(champ_kept),
    )
