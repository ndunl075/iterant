"""``iterant status``: what the archive says (read-only)."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from iterant.archive import Archive, ArchiveError, ArchiveNotInitialized, archive_path
from iterant.cli.errors import EXIT_ERROR, EXIT_OK, CliError
from iterant.cli.lock import cli_lock
from iterant.cli.settings import thresholds_from_args
from iterant.contracts import (
    AcceptanceThresholds,
    ArchiveTampered,
    Batch,
    Candidate,
    Promotion,
    Split,
    open_archive,
)
from iterant.orchestrator import OrchestratorBusy
from iterant.orchestrator.loop import CLOSED, SEED_ID

LAST_RUN = Path(".iterant") / "last_run.json"


def record_last_run(repo: Path, thresholds: AcceptanceThresholds) -> None:
    """Remember the holdout limits ``iterant run`` used, so ``status`` can show them."""
    target = repo / LAST_RUN
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(
                {
                    "holdout_query_max": thresholds.holdout_query_max,
                    "holdout_query_warn": thresholds.holdout_query_warn,
                }
            ),
            encoding="utf-8",
        )
        os.replace(tmp, target)  # atomic: status never reads a half-written file
    except OSError:
        pass  # status falls back to the default label


def _recorded_limits(repo: Path) -> tuple[int, int] | None:
    try:
        raw: object = json.loads((repo / LAST_RUN).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    mx: object = raw.get("holdout_query_max")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    warn: object = raw.get("holdout_query_warn")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(mx, int) and isinstance(warn, int) and not isinstance(mx, bool):
        return warn, mx
    return None


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
    lock = cli_lock(repo)
    try:
        lock.acquire()
    except OrchestratorBusy:
        return True
    lock.release()
    return False


def _members(cands: list[Candidate], batch_id: int) -> list[Candidate]:
    return [c for c in cands if c.id.startswith(f"b{batch_id}-")]


def batch_champion_id(archive: Archive, batch: Batch, promo: Promotion | None) -> str:
    """The champion arm of ``batch``: the champion when the batch opened. If the batch itself
    promoted (``promo.gen`` is the batch's generation) the current champion is its winner, a
    candidate of the batch, so the arm is the winner's parent."""
    if promo is None:
        return SEED_ID
    if promo.gen >= batch.gen:
        winner = archive.get_candidate(promo.cand)
        if winner is not None and winner.parent is not None:
            return winner.parent
        return SEED_ID
    return promo.cand


def render_status(
    repo: Path,
    thresholds: AcceptanceThresholds | None = None,
    *,
    explicit_limit: bool = False,
) -> str:
    """``explicit_limit``: the caller gave ``--holdout-query-max`` itself, so ``thresholds``
    wins over what the last run recorded."""
    path = archive_path(repo)
    try:
        archive = Archive.open_path(path, create_schema=False)
    except ArchiveNotInitialized:
        raise CliError(
            f"no archive at {path}: nothing has run here yet (try `iterant run`)", EXIT_ERROR
        ) from None
    except ArchiveTampered as exc:
        raise CliError(f"archive integrity check failed: {exc}", EXIT_ERROR) from exc
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
        champ_firsts = (
            [
                r
                for r in archive.runs_for_candidate(
                    batch_champion_id(archive, batches[-1], promo),
                    batch=batches[-1].id,
                    latest_only=False,
                )
                if r.key.attempt == 0 and r.split is not Split.TRAIN
            ]
            if batches
            else []
        )
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
    t = thresholds or AcceptanceThresholds()
    warn, hard = t.holdout_query_warn, t.holdout_query_max
    recorded = _recorded_limits(repo)
    if explicit_limit:
        which = "from your flag"
    elif recorded is not None:
        warn, hard = recorded
        which = "used by the last run"
    else:
        which = "default; no run recorded"
    lines.append(
        f"holdout queries: {holdout_queries} (warns after {warn}, stops at {hard} ({which}))"
        + (": add fresh holdout tasks" if holdout_queries > warn else "")
    )
    total, by_role, calls = _spend(path)
    roles = ", ".join(f"{r} ${v / 1e6:.4f}" for r, v in sorted(by_role.items()))
    lines.append(
        f"spend: ${total / 1e6:.4f} over {calls} proxy calls" + (f" ({roles})" if roles else "")
    )
    if champ_firsts:
        refused = sum(1 for r in champ_firsts if r.refusal)
        lines.append(
            f"champion refusals (latest batch, first attempts, select and holdout): "
            f"{refused} of {len(champ_firsts)} ({refused / len(champ_firsts):.0%})"
        )
    if batches:
        last = batches[-1]
        lines.append(f"latest batch {last.id} (generation {last.gen}, pack {last.pack_hash[:12]}):")
        for c in _members(cands, last.id):
            note = f" -- {c.hypothesis}" if c.hypothesis else ""
            lines.append(f"  {c.id}: {c.status.value} at {c.stage.value}{note}")
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    explicit = getattr(args, "holdout_query_max", None) is not None
    print(render_status(Path(args.repo), thresholds_from_args(args), explicit_limit=explicit))
    return EXIT_OK
