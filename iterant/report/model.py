"""Read the archive into plain report data. Nothing here renders HTML.

Only aggregate statistics leave this module: no task ids, no hidden-check content and no
non-train task data (invariant 4). Diffs and hypotheses are carried verbatim but are
untrusted text; ``render`` escapes them.

The archive is read through ONE read-only connection in ONE read transaction, so a report
never takes a write lock, never checkpoints and sees a consistent snapshot even while an
orchestrator is writing. (It deliberately does not use ``contracts.open_archive``: that
helper opens read-write and only exists to turn foreign keys on, which reads do not need.)
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from iterant.archive import ArchiveNotInitialized, diff_sha256
from iterant.contracts import (
    AcceptanceThresholds,
    Candidate,
    Pack,
    Promotion,
    RunKey,
    RunResult,
    Split,
    Stage,
    Status,
)
from iterant.stats.acceptance import decide_confirm, decide_select
from iterant.stats.pairing import final_attempts

SEED_ID: Final = "seed"
# Recomputed p-values use this fixed seed. The orchestrator derives its own per-candidate seed
# from its config, which the archive does not store, so a Monte Carlo p may differ slightly.
REPORT_SEED: Final = 0
MAX_DIFF_LINES: Final = 2000
MAX_DIFF_BYTES: Final = 512 * 1024

# HEURISTIC, not a verdict: a regex over the ADDED lines of a diff. It misses obfuscated or
# indirect execution (and can flag harmless mentions). It only tells a reader where to look.
CODE_EXEC_PATTERN: Final = re.compile(
    r"\bsubprocess\b|\bos\.(?:system|popen|exec\w*|spawn\w*)\b|\bPopen\b|\bpty\.\w+"
    r"|\bcreate_subprocess_\w+|\b(?:exec|eval|compile)\s*\(|\b__import__\b|\bimportlib\b"
    r"|\bctypes\b|\bchild_process\b|\bexecSync\b|\bexecFile\b|\bshell\s*=\s*True"
    r"|\b(?:bash|sh|zsh|powershell|cmd)(?:\.exe)?\s+-c\b"
    r"|\b(?:code_interpreter|python_repl|run_python|execute_code|run_shell|run_command"
    r"|shell_tool|bash_tool)\b",
    re.IGNORECASE,
)


def adds_code_execution(diff: str) -> bool:
    """Heuristic: does any added line of ``diff`` match ``CODE_EXEC_PATTERN``?

    Only a real file header (a ``+++ `` line right after a ``--- `` line) is skipped; an
    added line that merely starts with ``++`` is still checked.
    """
    prev = ""
    for line in diff.splitlines():
        header = line.startswith("+++ ") and prev.startswith("--- ")
        if line.startswith("+") and not header and CODE_EXEC_PATTERN.search(line):
            return True
        prev = line
    return False


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
    """One stage's numbers. ``stored`` rows come from the promotions table (the recorded
    decision); otherwise ``delta`` and ``p`` are recomputed from archived runs with the default
    thresholds and a fixed seed, and are NOT the recorded decision. No verdict either way."""

    split: Split
    stored: bool
    delta: float | None
    p: float | None
    n_tasks: int | None
    n_dropped: int | None
    note: str
    arms: tuple[ArmStats, ...]


@dataclass(frozen=True)
class CandidateReport:
    cand: Candidate
    diff: str
    diff_note: str  # why the diff text is missing or cut, else ""
    diff_missing: bool
    select: StageEvidence | None
    confirm: StageEvidence | None
    code_exec: bool


@dataclass(frozen=True)
class GenerationReport:
    gen: int
    batch_ids: tuple[int, ...]
    pack_hashes: tuple[str, ...]
    pack_matches: bool | None  # None: no usable pack supplied
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
    pack: Pack | None  # the supplied pack only if its hash matches some batch
    pack_warning: str  # why the pack name/policy is unavailable, else ""
    total_usd_micros: int
    unattributed_usd_micros: int  # total minus the per-generation sums
    holdout_queries: int
    warnings: tuple[str, ...]


def _connect(db: Path) -> sqlite3.Connection:
    """Open read-only. A WAL archive with no ``-wal`` file has no live writer, and a plain
    read-only open would create ``-wal``/``-shm`` beside it, so that case opens ``immutable``
    (no files, no locks). With a ``-wal`` present (a writer may be live) the open is a normal
    read-only one, which only reads the existing shared-memory file."""
    if not db.is_file():
        raise ArchiveNotInitialized(f"no archive at {db}")
    wal = db.with_name(db.name + "-wal")
    uri = db.resolve().as_uri()
    # a writer can appear between the exists() check and the open: if the immutable open then
    # reads a torn database, retry once as a plain read-only open
    plans = ["mode=ro"] if wal.exists() else ["mode=ro&immutable=1", "mode=ro"]
    last = len(plans) - 1
    for i, flags in enumerate(plans):
        con: sqlite3.Connection | None = None
        try:
            con = sqlite3.connect(f"{uri}?{flags}", uri=True, isolation_level=None)
            con.execute("BEGIN")  # one read transaction: a consistent snapshot
            con.execute("SELECT count(*) FROM sqlite_master").fetchone()
            return con
        except sqlite3.DatabaseError:
            if con is not None:
                con.close()
            if i == last:
                raise
    raise AssertionError("unreachable")  # pragma: no cover


def _candidate(row: Any) -> Candidate:
    return Candidate(row[0], row[1], row[2], row[3], Stage(row[4]), Status(row[5]), row[6])


def _run(row: Any) -> RunResult:
    return RunResult(
        key=RunKey(batch=row[0], cand=row[1], task=row[2], rep=row[3], attempt=row[4]),
        split=Split(row[5]),
        score=row[6],
        in_tok=row[7],
        out_tok=row[8],
        usd_micros=row[9],
        wall_s=row[10],
        refusal=bool(row[11]),
        infra_err=bool(row[12]),
    )


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
    c_runs: Sequence[RunResult],
    h_runs: Sequence[RunResult],
    cand: str,
    champ: str,
    split: Split,
    stored: tuple[float, float] | None,
    t: AcceptanceThresholds,
) -> StageEvidence | None:
    if not c_runs:
        return None
    arms = (_arm_stats(c_runs, cand, split), _arm_stats(h_runs, champ, split))
    if stored is not None:
        # The recorded decision: stored numbers only, nothing recomputed beside them.
        return StageEvidence(split, True, stored[0], stored[1], None, None, "", arms)
    decide = decide_select if split is Split.SELECT else decide_confirm
    try:
        d = decide(c_runs, h_runs, t, seed=REPORT_SEED)
    except ValueError as exc:
        return StageEvidence(split, False, None, None, None, None, f"not comparable: {exc}", arms)
    return StageEvidence(split, False, d.delta, d.p, d.n_tasks, d.n_dropped, "", arms)


def _read_diff(diffs_dir: Path, sha: str) -> tuple[str, str, bool]:
    """``(text, note, missing)``. Reads at most ``MAX_DIFF_BYTES`` of the file."""
    if sha == diff_sha256(b""):
        return "", "", False
    path = diffs_dir / f"{sha}.diff"
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_DIFF_BYTES + 1)
    except OSError as exc:
        return "", f"diff unavailable: {type(exc).__name__}", True
    note = ""
    if len(data) > MAX_DIFF_BYTES:
        data, note = data[:MAX_DIFF_BYTES], f"diff cut to the first {MAX_DIFF_BYTES} bytes"
    elif diff_sha256(data) != sha:
        return "", "diff unavailable: stored file does not match its hash", True
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > MAX_DIFF_LINES:
        text = "\n".join(lines[:MAX_DIFF_LINES])
        note = f"diff cut to the first {MAX_DIFF_LINES} of {len(lines)} lines"
    return text, note, False


def _gen_spend(
    gen: int,
    batch_ids: Sequence[int],
    calls: Sequence[tuple[int | None, str, int, int, int, datetime]],
    starts: dict[int, datetime],
) -> tuple[int, int, int, tuple[tuple[str, int], ...]]:
    """Proxy-call spend of one generation: calls of its batches plus run-less calls (proposer,
    analyst, auditor) from its first batch's start until the next generation's first batch."""
    mine = set(batch_ids)
    rows = [c for c in calls if c[0] in mine]
    if gen in starts:
        lo = starts[gen].replace(microsecond=0)
        later = [s for g, s in starts.items() if g > gen]
        hi = min(later).replace(microsecond=0) if later else None
        rows += [c for c in calls if c[0] is None and c[5] >= lo and (hi is None or c[5] < hi)]
    roles: dict[str, int] = {}
    for _b, role, usd, _i, _o, _t in rows:
        roles[role] = roles.get(role, 0) + usd
    return (
        sum(c[2] for c in rows),
        sum(c[3] for c in rows),
        sum(c[4] for c in rows),
        tuple(sorted(roles.items())),
    )


def collect(
    archive_path: Path,
    *,
    pack: Pack | None = None,
    thresholds: AcceptanceThresholds | None = None,
    diffs_dir: Path | None = None,
) -> ReportData:
    t = thresholds or AcceptanceThresholds()
    # ``<repo>/.iterant/diffs`` sits beside ``<repo>/.iterant/archive.db`` (DiffStore's layout).
    diffs = diffs_dir if diffs_dir is not None else archive_path.parent / "diffs"
    with closing(_connect(archive_path)) as con:
        batches = [
            (int(r[0]), int(r[1]), str(r[2]), datetime.fromisoformat(str(r[3])))
            for r in con.execute("SELECT id, gen, pack_hash, started_at FROM batches ORDER BY id")
        ]
        cands = [
            _candidate(r)
            for r in con.execute(
                "SELECT id, parent, diff_sha, hypothesis, stage, status, usd_micros"
                " FROM candidates ORDER BY rowid"
            )
        ]
        promotions = {
            int(r[0]): Promotion(int(r[0]), r[1], r[2], r[3], r[4], r[5])
            for r in con.execute(
                "SELECT gen, cand, select_delta, select_p, holdout_delta, holdout_p"
                " FROM promotions ORDER BY gen"
            )
        }
        by_slot: dict[tuple[int, str, Split], list[RunResult]] = {}
        for r in con.execute(
            "SELECT batch, cand, task, rep, attempt, split, score, in_tok, out_tok,"
            " usd_micros, wall_s, refusal, infra_err FROM runs ORDER BY batch, task, rep, attempt"
        ):
            run = _run(r)
            by_slot.setdefault((run.key.batch, run.key.cand, run.split), []).append(run)
        calls = [
            (
                None if r[0] is None else int(r[0]),
                str(r[1]),
                int(r[2]),
                int(r[3]),
                int(r[4]),
                datetime.fromisoformat(str(r[5])),
            )
            for r in con.execute(
                "SELECT batch, role, usd_micros, in_tok, out_tok, created_at FROM proxy_calls"
            )
        ]
        queries = int(
            con.execute(
                "SELECT COUNT(DISTINCT batch) FROM runs WHERE split = 'holdout'"
            ).fetchone()[0]
        )
        con.execute("COMMIT")

    all_hashes = {h for _, _, h, _ in batches}
    pack_warning = ""
    if pack is None:
        pack_warning = (
            "No pack was supplied to the report: the pack name and whether the agent is "
            "forbidden from executing code are UNKNOWN here."
        )
    elif pack.content_hash not in all_hashes:
        pack_warning = (
            "The supplied pack's hash matches none of the archive's batches, so it is ignored: "
            "pack name and code-execution policy are UNKNOWN for this archive."
        )
        pack = None

    by_gen: dict[int, list[tuple[int, str]]] = {}
    starts: dict[int, datetime] = {}
    for bid, gen, ph, at in batches:
        by_gen.setdefault(gen, []).append((bid, ph))
        starts[gen] = min(at, starts.get(gen, at))
    gens: list[GenerationReport] = []
    for gen in sorted(set(by_gen) | set(promotions)):
        members = by_gen.get(gen, [])
        batch_ids = tuple(b for b, _ in members)
        promo = promotions.get(gen)
        reports: list[CandidateReport] = []
        for bid in batch_ids:
            for c in cands:
                if not c.id.startswith(f"b{bid}-") or c.id.endswith("-closed"):
                    continue
                champ = c.parent or SEED_ID
                won = (
                    promo if promo and promo.cand == c.id and c.status is Status.PROMOTED else None
                )
                sel = _stage(
                    by_slot.get((bid, c.id, Split.SELECT), []),
                    by_slot.get((bid, champ, Split.SELECT), []),
                    c.id, champ, Split.SELECT,
                    (won.select_delta, won.select_p) if won else None, t,
                )  # fmt: skip
                conf = _stage(
                    by_slot.get((bid, c.id, Split.HOLDOUT), []),
                    by_slot.get((bid, champ, Split.HOLDOUT), []),
                    c.id, champ, Split.HOLDOUT,
                    (won.holdout_delta, won.holdout_p) if won else None, t,
                )  # fmt: skip
                diff, note, missing = _read_diff(diffs, c.diff_sha)
                reports.append(
                    CandidateReport(c, diff, note, missing, sel, conf, adds_code_execution(diff))
                )
        usd, i, o, roles = _gen_spend(gen, batch_ids, calls, starts)
        hashes = tuple(dict.fromkeys(h for _, h in members))
        gens.append(
            GenerationReport(
                gen=gen,
                batch_ids=batch_ids,
                pack_hashes=hashes,
                pack_matches=None if pack is None else all(h == pack.content_hash for h in hashes),
                candidates=tuple(reports),
                promotion=promo,
                usd_micros=usd,
                in_tok=i,
                out_tok=o,
                by_role=roles,
            )
        )
    lineage = (SEED_ID, *(promotions[g].cand for g in sorted(promotions)))
    total = sum(c[2] for c in calls)
    warnings: tuple[str, ...] = ()
    if queries > t.holdout_query_warn:
        warnings = (
            f"holdout queried {queries} times (warn after {t.holdout_query_warn}): "
            "add fresh holdout tasks",
        )
    return ReportData(
        tuple(gens),
        lineage,
        t,
        pack,
        pack_warning,
        total,
        total - sum(g.usd_micros for g in gens),
        queries,
        warnings,
    )
