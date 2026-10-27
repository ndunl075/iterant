"""Read the archive into plain report data. Nothing here renders HTML.

Only aggregate statistics leave this module: no task ids, no hidden-check content and no
non-train task data (invariant 4). Diffs and hypotheses are carried verbatim but are
untrusted text; ``render`` escapes them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from iterant.archive import Archive
from iterant.contracts import (
    AcceptanceThresholds,
    Candidate,
    Pack,
    Promotion,
    RunResult,
    Split,
    Status,
    open_archive,
)
from iterant.orchestrator.workspace import DiffStore
from iterant.stats.acceptance import decide_confirm, decide_select
from iterant.stats.pairing import final_attempts

SEED_ID: Final = "seed"
# Recomputed p-values use this fixed seed. The orchestrator derives its own per-candidate seed
# from its config, which the archive does not store, so a Monte Carlo p may differ slightly.
REPORT_SEED: Final = 0
MAX_DIFF_LINES: Final = 2000

# HEURISTIC, not a verdict: a regex over the ADDED lines of a diff. It misses obfuscated or
# indirect execution (and can flag harmless mentions). It only tells a reader where to look.
CODE_EXEC_PATTERN: Final = re.compile(
    r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*)\b|\bPopen\b|\bpty\.\w+"
    r"|\b(?:exec|eval|compile)\s*\(|\bchild_process\b|\bshell\s*=\s*True"
    r"|\b(?:bash|sh|zsh|powershell|cmd)(?:\.exe)?\s+-c\b"
    r"|\b(?:code_interpreter|python_repl|run_python|execute_code|run_shell|run_command"
    r"|shell_tool|bash_tool)\b",
    re.IGNORECASE,
)


def adds_code_execution(diff: str) -> bool:
    """Heuristic: does any added line of ``diff`` match ``CODE_EXEC_PATTERN``?"""
    return any(
        CODE_EXEC_PATTERN.search(line)
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


@dataclass(frozen=True)
class ArmStats:
    arm: str  # candidate id (the champion arm is the parent's id)
    split: Split
    runs: int  # final attempts
    refusal_rate: float
    infra_rate: float
    mean_usd_micros: float  # per final run, retries included
    mean_wall_s: float


@dataclass(frozen=True)
class StageEvidence:
    split: Split
    stored: bool  # True: delta and p are the promotions row; False: recomputed from runs
    delta: float | None
    p: float | None
    n_tasks: int | None
    n_dropped: int | None
    accepted: bool | None  # recomputed verdict under the report's thresholds
    detail: tuple[str, ...]
    arms: tuple[ArmStats, ...]


@dataclass(frozen=True)
class CandidateReport:
    cand: Candidate
    diff: str
    diff_note: str  # why the diff text is missing or cut, else ""
    select: StageEvidence | None
    confirm: StageEvidence | None
    code_exec: bool


@dataclass(frozen=True)
class GenerationReport:
    gen: int
    batch_ids: tuple[int, ...]
    pack_hashes: tuple[str, ...]
    candidates: tuple[CandidateReport, ...]
    promotion: Promotion | None
    usd_micros: int
    in_tok: int
    out_tok: int
    by_role: tuple[tuple[str, int], ...]  # role, usd_micros


@dataclass(frozen=True)
class ReportData:
    generations: tuple[GenerationReport, ...]
    lineage: tuple[str, ...]  # champion cand ids, seed first
    thresholds: AcceptanceThresholds
    pack: Pack | None
    total_usd_micros: int
    holdout_queries: int
    warnings: tuple[str, ...]


def _arm_stats(runs: Sequence[RunResult], arm: str, split: Split) -> ArmStats:
    finals = list(final_attempts(runs).values())
    n = len(finals) or 1
    return ArmStats(
        arm=arm,
        split=split,
        runs=len(finals),
        refusal_rate=sum(r.refusal for r in finals) / n,
        infra_rate=sum(r.infra_err for r in finals) / n,
        mean_usd_micros=sum(r.usd_micros for r in runs) / n,
        mean_wall_s=sum(r.wall_s for r in runs) / n,
    )


def _stage(
    arc: Archive,
    cand: Candidate,
    champ_id: str,
    batch: int,
    split: Split,
    stored: tuple[float, float] | None,
    t: AcceptanceThresholds,
) -> StageEvidence | None:
    c_runs = arc.runs_for_candidate(cand.id, split=split, batch=batch, latest_only=False)
    h_runs = arc.runs_for_candidate(champ_id, split=split, batch=batch, latest_only=False)
    if not c_runs:
        return None
    arms = (_arm_stats(c_runs, cand.id, split), _arm_stats(h_runs, champ_id, split))
    decide = decide_select if split is Split.SELECT else decide_confirm
    try:
        d = decide(c_runs, h_runs, t, seed=REPORT_SEED)
    except ValueError as exc:
        note = f"arms not comparable: {exc}"
        if stored is not None:
            return StageEvidence(split, True, stored[0], stored[1], None, None, None, (note,), arms)
        return StageEvidence(split, False, None, None, None, None, None, (note,), arms)
    if stored is not None:
        return StageEvidence(
            split, True, stored[0], stored[1], d.n_tasks, d.n_dropped, d.accepted, d.detail, arms
        )
    return StageEvidence(
        split, False, d.delta, d.p, d.n_tasks, d.n_dropped, d.accepted, d.detail, arms
    )


def _spend(
    db: Path, gen: int, batch_ids: Sequence[int]
) -> tuple[int, int, int, tuple[tuple[str, int], ...]]:
    """Proxy-call spend of one generation: calls of its batches plus run-less calls (proposer,
    analyst, auditor) from its first batch's start until the next generation's first batch."""
    rows: list[tuple[str, int, int, int]] = []
    with closing(open_archive(db)) as con:
        ids = ",".join(str(int(b)) for b in batch_ids) or "NULL"
        rows.extend(
            (str(r[0]), int(r[1]), int(r[2]), int(r[3]))
            for r in con.execute(
                f"SELECT role, usd_micros, in_tok, out_tok FROM proxy_calls WHERE batch IN ({ids})"
            )
        )
        starts = {
            int(r[0]): datetime.fromisoformat(str(r[1]))
            for r in con.execute("SELECT gen, MIN(started_at) FROM batches GROUP BY gen")
        }
        if gen in starts:
            later = [s for g, s in starts.items() if g > gen]
            lo = starts[gen].replace(microsecond=0)
            hi = min(later).replace(microsecond=0) if later else None
            for r in con.execute(
                "SELECT role, usd_micros, in_tok, out_tok, created_at FROM proxy_calls"
                " WHERE batch IS NULL"
            ):
                at = datetime.fromisoformat(str(r[4]))
                if at >= lo and (hi is None or at < hi):
                    rows.append((str(r[0]), int(r[1]), int(r[2]), int(r[3])))
    roles: dict[str, int] = {}
    for role, usd, _i, _o in rows:
        roles[role] = roles.get(role, 0) + usd
    return (
        sum(r[1] for r in rows),
        sum(r[2] for r in rows),
        sum(r[3] for r in rows),
        tuple(sorted(roles.items())),
    )


def _candidate(
    arc: Archive,
    store: DiffStore,
    c: Candidate,
    batch: int,
    promo: Promotion | None,
    t: AcceptanceThresholds,
) -> CandidateReport:
    note = ""
    try:
        diff = store.get(c.diff_sha)
    except (FileNotFoundError, ValueError) as exc:
        diff, note = "", f"diff unavailable: {type(exc).__name__}"
    lines = diff.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        diff = "\n".join(lines[:MAX_DIFF_LINES])
        note = f"diff cut to the first {MAX_DIFF_LINES} of {len(lines)} lines"
    won = (
        promo if promo is not None and promo.cand == c.id and c.status is Status.PROMOTED else None
    )
    champ = c.parent or SEED_ID
    sel = _stage(
        arc, c, champ, batch, Split.SELECT, (won.select_delta, won.select_p) if won else None, t
    )
    conf = _stage(
        arc, c, champ, batch, Split.HOLDOUT, (won.holdout_delta, won.holdout_p) if won else None, t
    )
    return CandidateReport(c, diff, note, sel, conf, adds_code_execution(diff))


def collect(
    archive_path: Path,
    *,
    pack: Pack | None = None,
    thresholds: AcceptanceThresholds | None = None,
) -> ReportData:
    t = thresholds or AcceptanceThresholds()
    store = DiffStore(archive_path.parent.parent)
    with Archive.open_path(archive_path, create_schema=False) as arc:
        with closing(open_archive(archive_path)) as con:
            batches = [
                (int(r[0]), int(r[1]), str(r[2]))
                for r in con.execute("SELECT id, gen, pack_hash FROM batches ORDER BY id")
            ]
            promo_gens = [int(r[0]) for r in con.execute("SELECT gen FROM promotions ORDER BY gen")]
            total = int(
                con.execute("SELECT COALESCE(SUM(usd_micros), 0) FROM proxy_calls").fetchone()[0]
            )
        by_gen: dict[int, list[tuple[int, str]]] = {}
        for bid, gen, ph in batches:
            by_gen.setdefault(gen, []).append((bid, ph))
        cands = list(arc.iter_candidates())
        promotions = {g: p for g in promo_gens if (p := arc.get_promotion(g)) is not None}
        gens: list[GenerationReport] = []
        for gen in sorted(set(by_gen) | set(promotions)):
            members = by_gen.get(gen, [])
            batch_ids = tuple(b for b, _ in members)
            promo = promotions.get(gen)
            reports = [
                _candidate(arc, store, c, bid, promo, t)
                for bid in batch_ids
                for c in cands
                if c.id.startswith(f"b{bid}-") and not c.id.endswith("-closed")
            ]
            usd, i, o, roles = _spend(archive_path, gen, batch_ids)
            gens.append(
                GenerationReport(
                    gen=gen,
                    batch_ids=batch_ids,
                    pack_hashes=tuple(dict.fromkeys(h for _, h in members)),
                    candidates=tuple(reports),
                    promotion=promo,
                    usd_micros=usd,
                    in_tok=i,
                    out_tok=o,
                    by_role=roles,
                )
            )
        lineage = (SEED_ID, *(promotions[g].cand for g in sorted(promotions)))
        queries = arc.holdout_query_count()
    warnings: tuple[str, ...] = ()
    if queries > t.holdout_query_warn:
        warnings = (
            f"holdout queried {queries} times (warn after {t.holdout_query_warn}): "
            "add fresh holdout tasks",
        )
    return ReportData(tuple(gens), lineage, t, pack, total, queries, warnings)
