"""``iterant status``: what the archive says (read-only)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from iterant.archive import Archive, ArchiveError, ArchiveNotInitialized, archive_path
from iterant.cli.errors import EXIT_ERROR, EXIT_OK, CliError
from iterant.contracts import AcceptanceThresholds, Batch, Candidate, open_archive
from iterant.orchestrator import OrchestratorBusy, WriterLock
from iterant.orchestrator.loop import CLOSED, SEED_ID


def _spend(path: Path) -> tuple[int, dict[str, int], int]:
    """Total micro-USD, micro-USD and calls by role, from the proxy log."""
    con = open_archive(path)
    try:
        rows = con.execute(
            "SELECT role, COUNT(*), COALESCE(SUM(usd_micros), 0) FROM proxy_calls GROUP BY role"
        ).fetchall()
    finally:
        con.close()
    by_role = {str(r[0]): int(r[2]) for r in rows}
    return sum(by_role.values()), by_role, sum(int(r[1]) for r in rows)


def _is_running(repo: Path) -> bool:
    lock = WriterLock(repo)
    try:
        lock.acquire()
    except OrchestratorBusy:
        return True
    lock.release()
    return False


def _members(cands: list[Candidate], batch_id: int) -> list[Candidate]:
    return [c for c in cands if c.id.startswith(f"b{batch_id}-")]


def render_status(repo: Path) -> str:
    path = archive_path(repo)
    try:
        archive = Archive.open_path(path, create_schema=False)
    except ArchiveNotInitialized:
        raise CliError(
            f"no archive at {path}: nothing has run here yet (try `iterant run`)", EXIT_ERROR
        ) from None
    except ArchiveError as exc:
        raise CliError(f"cannot open the archive at {path}: {exc}", EXIT_ERROR) from exc
    with archive:
        cands = list(archive.iter_candidates())
        batches: list[Batch] = []
        while (b := archive.get_batch(len(batches) + 1)) is not None:
            batches.append(b)
        promo = archive.latest_promotion()
        promos = [
            p for g in range(1, (promo.gen if promo else 0) + 1) if (p := archive.get_promotion(g))
        ]
        holdout_queries = archive.holdout_query_count()
    lines = [f"repo: {repo.resolve()}"]
    if _is_running(repo):
        lines.append("a run is in progress (the archive is locked)")
    if promo is None:
        lines.append("champion: seed (generation 0, the repo as it was at the first run)")
    else:
        lines.append(f"champion: {promo.cand} (generation {promo.gen})")
    for p in promos:
        lines.append(
            f"  gen {p.gen}: {p.cand}  select +{p.select_delta * 100:.1f} pts (p={p.select_p:.3f})"
            f"  holdout +{p.holdout_delta * 100:.1f} pts (p={p.holdout_p:.3f})"
        )
    real = [c for c in cands if c.id != SEED_ID and not c.id.endswith(f"-{CLOSED}")]
    by_status = Counter(c.status.value for c in real)
    lines.append(
        f"batches: {len(batches)}; candidates: {len(real)} "
        + "("
        + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
        + ")"
        if real
        else f"batches: {len(batches)}; candidates: 0"
    )
    warn = AcceptanceThresholds().holdout_query_warn
    lines.append(
        f"holdout queries: {holdout_queries} (warns after {warn})"
        + (": add fresh holdout tasks" if holdout_queries > warn else "")
    )
    total, by_role, calls = _spend(path)
    roles = ", ".join(f"{r} ${v / 1e6:.4f}" for r, v in sorted(by_role.items()))
    lines.append(
        f"spend: ${total / 1e6:.4f} over {calls} proxy calls" + (f" ({roles})" if roles else "")
    )
    if batches:
        last = batches[-1]
        lines.append(f"latest batch {last.id} (generation {last.gen}, pack {last.pack_hash[:12]}):")
        for c in _members(cands, last.id):
            note = f" -- {c.hypothesis}" if c.hypothesis else ""
            lines.append(f"  {c.id}: {c.status.value} at {c.stage.value}{note}")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    print(render_status(Path(args.repo)))
    return EXIT_OK
