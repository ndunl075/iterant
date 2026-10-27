"""Pair candidate and champion runs by task (§3 SELECT, §6 retries)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from iterant.contracts import RunResult, Split


@dataclass(frozen=True)
class Pairing:
    """Per-task paired gains. ``deltas[i]`` is candidate minus champion for ``task_ids[i]``.

    ``dropped_tasks`` lists tasks removed from both arms because a final run (highest
    attempt) of either arm was infra-errored. A final refusal is not a drop: it scores 0.0
    for the arm that refused. ``candidate_runs`` and
    ``champion_runs`` are the final attempts of the kept tasks; ``*_attempts`` are
    every attempt of those tasks (retries included), which is what the cost and wall
    guards must count.
    """

    task_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    dropped_tasks: tuple[str, ...]
    candidate_runs: tuple[RunResult, ...]
    champion_runs: tuple[RunResult, ...]
    candidate_attempts: tuple[RunResult, ...] = ()
    champion_attempts: tuple[RunResult, ...] = ()

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


def _capped(items: Iterable[Any], limit: int = 10) -> str:
    xs = sorted(items)
    more = f" and {len(xs) - limit} more" if len(xs) > limit else ""
    return f"{xs[:limit]}{more}"


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


def _score(r: RunResult) -> float:
    """A refusal that survived its retry is the refusing arm's failure: it scores 0.0 (in
    either arm), so refusing on its worst tasks gains a candidate nothing."""
    return 0.0 if r.score is None else r.score


def pair_runs(
    candidate: Sequence[RunResult],
    champion: Sequence[RunResult],
    *,
    split: Split | None = None,
) -> Pairing:
    """Pair the two arms by task, averaging scores over reps.

    A task is dropped from both arms when any final run of either arm is infra-errored; a
    final refusal scores 0.0 for its arm and the pair is kept. Raises ``ValueError`` if the
    arms are not from one batch (or not all from ``split``), if an arm mixes candidates or
    both arms are the same candidate, or if a task, or a (task, rep), has runs in only one arm.
    """
    check_same_batch(candidate, champion, split)
    c_ids = {r.key.cand for r in candidate}
    h_ids = {r.key.cand for r in champion}
    if len(c_ids) > 1 or len(h_ids) > 1:
        raise ValueError(f"each arm must be one candidate, got {sorted(c_ids)} and {sorted(h_ids)}")
    if c_ids & h_ids:
        raise ValueError(f"candidate and champion arms are the same candidate {sorted(c_ids)}")
    c_final = final_attempts(candidate)
    h_final = final_attempts(champion)
    if c_final.keys() != h_final.keys():
        lone_tasks = {t for t, _ in c_final} ^ {t for t, _ in h_final}
        if lone_tasks:
            raise ValueError(f"tasks present in only one arm: {_capped(lone_tasks)}")
        lone = sorted(c_final.keys() ^ h_final.keys())
        raise ValueError(f"(task, rep) present in only one arm: {_capped(lone)}")
    cand = _by_task(c_final)
    champ = _by_task(h_final)
    c_all: dict[str, list[RunResult]] = {}
    h_all: dict[str, list[RunResult]] = {}
    for runs, acc in ((candidate, c_all), (champion, h_all)):
        for r in runs:
            acc.setdefault(r.key.task, []).append(r)
    kept: list[str] = []
    deltas: list[float] = []
    dropped: list[str] = []
    cand_kept: list[RunResult] = []
    champ_kept: list[RunResult] = []
    cand_att: list[RunResult] = []
    champ_att: list[RunResult] = []
    for task in sorted(cand):
        c_runs, h_runs = cand[task], champ[task]
        if any(r.score is None and not r.refusal for r in (*c_runs, *h_runs)):
            dropped.append(task)  # an infra error that survived its retry
            continue
        c_mean = math.fsum(_score(r) for r in c_runs) / len(c_runs)
        h_mean = math.fsum(_score(r) for r in h_runs) / len(h_runs)
        kept.append(task)
        deltas.append(c_mean - h_mean)
        cand_kept.extend(c_runs)
        champ_kept.extend(h_runs)
        cand_att.extend(c_all[task])
        champ_att.extend(h_all[task])
    return Pairing(
        task_ids=tuple(kept),
        deltas=tuple(deltas),
        dropped_tasks=tuple(dropped),
        candidate_runs=tuple(cand_kept),
        champion_runs=tuple(champ_kept),
        candidate_attempts=tuple(cand_att),
        champion_attempts=tuple(champ_att),
    )
