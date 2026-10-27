"""``iterant run``: wire the real objects and loop generations (§3).

Per generation a new ``Orchestrator`` is built over the same proxy, archive, backend, grader,
proposer and gate; only its $ budget changes (the remaining total). The orchestrator takes the
writer lock and runs ``Grader.preflight()`` itself; the CLI preflights once up front so a broken
grader image fails before any batch is opened.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from iterant.analyst import FailureAnalyst
from iterant.archive import archive_path, champion_sha
from iterant.cli.calibrate import calibration_problem
from iterant.cli.errors import (
    EXIT_BUDGET,
    EXIT_CALIBRATION,
    EXIT_INFRA,
    EXIT_OK,
    EXIT_USAGE,
    CliError,
)
from iterant.cli.lock import acquire_retrying, cli_lock
from iterant.cli.runtime import Runtime, RuntimeFactory
from iterant.cli.sample import make_executor, run_slots, worst_run_cost
from iterant.cli.settings import Project, Settings, load_project, settings_from_args
from iterant.contracts import Profile, Split, open_archive
from iterant.grader import GraderPreflightError
from iterant.orchestrator import (
    SEED_ID,
    Budget,
    FailureDigest,
    GenerationResult,
    Orchestrator,
    OrchestratorConfig,
    Outcomes,
    WriterLock,
)
from iterant.packs import tasks_in


def generation_spend(rt: Runtime, gen: int) -> int:
    """Micro-USD the proxy logged for the runs of generation ``gen``."""
    con = open_archive(archive_path(rt.project.repo))
    try:
        row = con.execute(
            "SELECT COALESCE(SUM(usd_micros), 0) FROM proxy_calls WHERE batch IN"
            " (SELECT id FROM batches WHERE gen = ?)",
            (gen,),
        ).fetchone()
        return int(row[0])
    finally:
        con.close()


def make_digest(rt: Runtime, settings: Settings, limit: int | None) -> FailureDigest:
    """The orchestrator's ``FailureDigest``: the failure digest of the champion on train tasks.

    The proxy keeps train content in memory only, so the sample runs here, through this
    process's proxy, in the generation's current batch as the champion's own slots. SMOKE
    then finds those slots done for the same first train tasks and does not run them twice.
    ``redact`` (hidden check paths and text, from the orchestrator) goes to the analyst."""
    project = rt.project

    def digest(redact: Sequence[str]) -> str:
        archive = rt.archive
        promo = archive.latest_promotion()
        champ = promo.cand if promo is not None else SEED_ID
        batch = 1
        while archive.get_batch(batch + 1) is not None:
            batch += 1
        sha = champion_sha(project.repo)
        if sha is None:
            raise CliError("no iterant/champion branch to sample", EXIT_INFRA)
        current = archive.get_batch(batch)
        assert current is not None
        tasks = tasks_in(project.pack, Split.TRAIN)[: settings.digest_tasks]
        cfg = OrchestratorConfig(thresholds=settings.thresholds, seed=settings.seed)
        run_slots(
            archive=archive,
            executor=make_executor(rt, cfg),
            outcomes=Outcomes(project.repo),
            budget=Budget(limit, lambda: generation_spend(rt, current.gen)),
            worst=worst_run_cost(rt),
            batch=batch,
            cand=champ,
            image=rt.images.champion(sha),
            plan=[(t, 0) for t in tasks],
        )
        results = archive.runs_for_candidate(champ, split=Split.TRAIN, batch=batch)
        content = rt.proxy.train_content()
        have = {c.key for c in content}
        # Train transcripts live in this process's memory only. Slots finished by an earlier
        # process (a resumed batch) have a score but no transcript: digest only what we can read.
        usable = [r for r in results if r.key in have]
        missing = len(results) - len(usable)
        note = ""
        if missing:
            note = (
                f"\n[{missing} sampled train run(s) from an earlier process have no transcript "
                "and are omitted from this digest]"
            )
            print(
                f"iterant: WARNING: {missing} of {len(results)} digest runs were finished by an "
                "earlier process and have no transcript; the failure digest omits them",
                file=sys.stderr,
            )
        analyst = FailureAnalyst(
            rt.analyst_client,
            model=rt.analyst_model,
            effort=rt.analyst_effort,
            redact=redact,
        )
        return analyst.digest(usable, content) + note

    return digest


def _money(micros: int, profile: Profile) -> str:
    return f"${micros / 1_000_000:.4f}" if profile is Profile.CLAUDE else "$0 (local model)"


def format_result(res: GenerationResult, profile: Profile) -> str:
    head = (
        f"generation {res.gen} (batch {res.batch_id}): "
        + (f"promoted {res.promoted}" if res.promoted else "no candidate promoted")
        + (" [stopped: budget]" if res.stopped == "budget" else "")
    )
    lines = [head]
    for c in res.candidates:
        why = f" -- {'; '.join(c.notes)}" if c.notes else ""
        lines.append(f"  {c.id}: {c.status.value} at {c.stage.value}{why}")
    lines.append(f"  spent {_money(res.spent_usd_micros, profile)}")
    lines.extend(f"  WARNING: {w}" for w in res.warnings)
    return "\n".join(lines)


def cmd_run(args: argparse.Namespace, factory: RuntimeFactory) -> int:
    settings = settings_from_args(args)
    if args.generations < 1:
        raise CliError("--generations must be >= 1", EXIT_USAGE)
    if settings.candidates < 1:
        raise CliError("--candidates must be >= 1", EXIT_USAGE)
    project = load_project(settings.repo)
    problem = calibration_problem(project.repo, project.pack.content_hash)
    if problem is not None:
        if not args.skip_calibration:
            raise CliError(
                f"{problem} (ARCHITECTURE §4: calibration must pass before a real run; "
                "--skip-calibration overrides)",
                EXIT_CALIBRATION,
            )
        print(f"iterant: WARNING: running without calibration ({problem})", file=sys.stderr)
    # Hold one lock across every generation (the orchestrator's own is per generation).
    lock = cli_lock(project.repo)
    acquire_retrying(lock)
    try:
        return _run_locked(args, factory, settings, project)
    finally:
        lock.release()


def _run_locked(
    args: argparse.Namespace, factory: RuntimeFactory, settings: Settings, project: Project
) -> int:
    # Fail fast and clearly when another run holds the archive (the orchestrator re-checks).
    with WriterLock(project.repo):
        pass
    rt = factory(settings, project, archive_path(project.repo))
    try:
        try:
            rt.grader.preflight()
        except GraderPreflightError as exc:
            raise CliError(f"grader preflight failed: {exc}", EXIT_INFRA) from exc
        return _loop(rt, settings, args.generations)
    finally:
        rt.close()


def _loop(rt: Runtime, settings: Settings, generations: int) -> int:
    project = rt.project
    total = settings.budget_usd_micros
    spent = 0
    for _ in range(generations):
        remaining = None if total is None else total - spent
        if remaining is not None and remaining <= 0:
            raise CliError(
                f"budget exhausted: {_money(spent, settings.profile)} spent", EXIT_BUDGET
            )
        cfg = OrchestratorConfig(
            thresholds=settings.thresholds,
            candidates_per_gen=settings.candidates,
            budget_usd_micros=remaining,
            seed=settings.seed,
        )
        orch = Orchestrator(
            repo=project.repo,
            pack=project.pack,
            pack_root=project.pack_root,
            target=project.config.target,
            archive=rt.archive,
            backend=rt.backend,
            grader=rt.grader,
            proxy=rt.proxy,
            proposer=rt.proposer,
            gate=rt.gate,
            images=rt.images,
            digest=make_digest(rt, settings, remaining),
            config=cfg,
        )
        res = orch.run_generation()
        print(format_result(res, settings.profile))
        spent += res.spent_usd_micros
        if res.stopped == "budget":
            raise CliError(
                f"budget exhausted after {_money(spent, settings.profile)}; "
                "nothing was rejected for it: rerun with a larger --budget-usd to resume "
                f"generation {res.gen}",
                EXIT_BUDGET,
            )
    return EXIT_OK
