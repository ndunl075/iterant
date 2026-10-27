"""``iterant estimate`` and ``iterant aa-test``: calibration before any real spend (§2, §4).

Both run the champion on the pack through the orchestrator's run machinery and keep the rows in
``.iterant/calibration.db``, a separate archive with the same schema, so calibration never
mixes with the real archive (the failure digest, the holdout query count and the batch
resume logic all read the real one). Both take the writer lock: not while ``run`` is going.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from iterant.archive import champion_sha, diff_sha256
from iterant.cli.errors import EXIT_CALIBRATION, EXIT_INFRA, EXIT_OK, EXIT_USAGE, CliError
from iterant.cli.lock import acquire_retrying, cli_lock
from iterant.cli.runtime import Runtime, RuntimeFactory
from iterant.cli.sample import make_executor, run_slots, worst_run_cost
from iterant.cli.settings import Project, Settings, load_project, settings_from_args
from iterant.contracts import (
    AcceptanceThresholds,
    Candidate,
    Pack,
    RunKey,
    RunResult,
    Split,
    Stage,
    Status,
    Task,
)
from iterant.grader import GraderPreflightError
from iterant.orchestrator import Budget, OrchestratorConfig, Outcomes, WriterLock
from iterant.packs import tasks_in, verify
from iterant.stats import Estimate, accepted, decide_confirm, decide_select, estimate
from iterant.stats.pairing import final_attempts

CALIBRATION_DB = Path(".iterant") / "calibration.db"
CHAMPION_ID = "champion"
MAX_FALSE_ACCEPT = 0.05  # §4: A/A false accepts must stay at or below 5%
EASY_PASS_RATE = 0.9
HARD_PASS_RATE = 0.1
SMALL_SAMPLE_TASKS = 20
HEALTHY_RATE = 0.05  # refusal and infra-error rates above this are worth a warning


def head_sha(repo: Path) -> str:
    """The champion's commit, or HEAD when no generation has run yet."""
    sha = champion_sha(repo)
    if sha is not None:
        return sha
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", "HEAD^{commit}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise CliError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        raise CliError(f"{repo} is not a git repo with at least one commit", EXIT_USAGE)
    return proc.stdout.strip()


@dataclass
class _Session:
    rt: Runtime
    batch: int
    image: str
    executor_config: OrchestratorConfig
    budget: Budget

    def run(self, plan: Sequence[tuple[Task, int]]) -> None:
        rt = self.rt
        run_slots(
            archive=rt.archive,
            executor=make_executor(rt, self.executor_config),
            outcomes=Outcomes(rt.project.repo),
            budget=self.budget,
            worst=worst_run_cost(rt),
            batch=self.batch,
            cand=CHAMPION_ID,
            image=self.image,
            plan=plan,
        )

    def runs(self) -> list[RunResult]:
        return self.rt.archive.runs_for_candidate(CHAMPION_ID, batch=self.batch, latest_only=False)


@contextmanager
def calibration_session(
    settings: Settings, project: Project, factory: RuntimeFactory
) -> Generator[_Session]:
    """Writer lock, runtime on the calibration archive, grader preflight, one new batch."""
    lock = cli_lock(project.repo)
    acquire_retrying(lock)
    try:
        yield from _session(settings, project, factory)
    finally:
        lock.release()


def _session(settings: Settings, project: Project, factory: RuntimeFactory) -> Generator[_Session]:
    with WriterLock(project.repo):
        # invariant 7: the pack on disk must still be the pack that was loaded
        verify(project.pack, project.pack.content_hash, project.pack_root)
        rt = factory(settings, project, project.repo / CALIBRATION_DB)
        try:
            try:
                rt.grader.preflight()
            except GraderPreflightError as exc:
                raise CliError(f"grader preflight failed: {exc}", EXIT_INFRA) from exc
            db = rt.archive
            batch = db.create_batch(0, project.pack.content_hash)
            if db.get_candidate(CHAMPION_ID) is None:
                db.insert_candidate(
                    Candidate(
                        CHAMPION_ID,
                        None,
                        diff_sha256(b""),
                        "champion baseline",
                        Stage.PROMOTE,
                        Status.PROMOTED,
                        0,
                    )
                )
            sha = head_sha(project.repo)
            try:
                image = rt.images.champion(sha)
            except Exception as exc:
                raise CliError(f"cannot build the champion image: {exc}", EXIT_INFRA) from exc
            cfg = OrchestratorConfig(thresholds=settings.thresholds, seed=settings.seed)
            budget = Budget(
                settings.budget_usd_micros,
                lambda: sum(
                    r.usd_micros
                    for r in db.runs_for_candidate(CHAMPION_ID, batch=batch.id, latest_only=False)
                ),
            )
            yield _Session(rt, batch.id, image, cfg, budget)
        finally:
            rt.close()


def effective_thresholds(pack: Pack, t: AcceptanceThresholds) -> AcceptanceThresholds:
    """The orchestrator slices each split to what the pack has; so does the math here."""
    n_sel = len(tasks_in(pack, Split.SELECT))
    n_hold = len(tasks_in(pack, Split.HOLDOUT))
    return replace(
        t,
        select_tasks=max(1, min(t.select_tasks, n_sel)),
        holdout_tasks=max(1, min(t.holdout_tasks, n_hold)),
    )


# --- estimate -----------------------------------------------------------------------


def pack_warnings(
    est: Estimate, pack: Pack, t: AcceptanceThresholds, *, sampled_split: Split
) -> list[str]:
    """Why this pack may not support a real run: too easy, too hard, too small, too noisy."""
    out: list[str] = []
    headroom = 1.0 - est.baseline_pass_rate
    if est.baseline_pass_rate >= EASY_PASS_RATE:
        out.append(
            f"pack is too easy: the champion already scores {est.baseline_pass_rate:.0%}, "
            f"leaving at most {headroom * 100:.0f} pts to gain. Candidates will win by noise "
            "or by gaming a weak check; add harder tasks."
        )
    if est.baseline_pass_rate <= HARD_PASS_RATE:
        out.append(
            f"pack is too hard: the champion scores {est.baseline_pass_rate:.0%}, so few tasks "
            "can flip and a candidate has almost nothing to learn from. Add easier tasks or "
            "check the agent runs at all."
        )
    sizes = {s: len(tasks_in(pack, s)) for s in (Split.TRAIN, Split.SELECT, Split.HOLDOUT)}
    for split, want in ((Split.SELECT, t.select_tasks), (Split.HOLDOUT, t.holdout_tasks)):
        have = sizes[split]
        if have == 0:
            out.append(
                f"pack is too small: it has no {split.value} tasks, so no candidate can pass"
            )
        elif have < want:
            out.append(
                f"pack is too small: {have} {split.value} tasks, the acceptance rule defaults "
                f"to {want}"
            )
    if sizes[Split.TRAIN] == 0:
        out.append("pack has no train tasks: the proposer would get no failure digest")
    if est.select_detectable_gain > t.delta_min:
        out.append(
            f"SELECT can only detect gains of {est.select_detectable_gain * 100:.1f} pts or more "
            f"(power 0.8), but delta_min is {t.delta_min * 100:.0f} pts: real gains near that "
            "will usually be missed. Add select tasks or reps."
        )
    if est.n_tasks < SMALL_SAMPLE_TASKS:
        out.append(
            f"only {est.n_tasks} {sampled_split.value} tasks were sampled: "
            "the pass-rate estimate is rough"
        )
    if not est.discordance_from_reps:
        out.append(
            "no task had 2 or more scored reps, so discordance is an upper bound and the "
            "detectable gain is pessimistic; rerun with --reps 2 or more"
        )
    for label, rate in (("refusal", est.refusal_rate), ("infra-error", est.infra_err_rate)):
        if rate > HEALTHY_RATE:
            out.append(f"{label} rate is {rate:.0%}: fix the setup before trusting any result")
    return out


def format_estimate(
    est: Estimate, pack: Pack, t: AcceptanceThresholds, *, split: Split, reps: int
) -> str:
    se = (est.score_variance / est.n_tasks) ** 0.5 if est.n_tasks else 0.0
    gain = est.select_detectable_gain * 100
    hold = est.holdout_detectable_gain * 100
    lines = [
        f"Baseline estimate for pack {pack.name} v{pack.version}: "
        f"{est.n_tasks} {split.value} tasks x {reps} reps = {est.n_runs} runs",
        f"  pass rate               {est.baseline_pass_rate:.1%} (+/- {se:.1%})",
        f"  variance                tasks {est.score_variance:.3f}, single run "
        f"{est.run_variance:.3f}",
        f"  discordance             {est.discordance:.3f} "
        f"({'measured from reps' if est.discordance_from_reps else 'upper bound, no reps'})",
        f"  refusals / infra errors {est.refusal_rate:.1%} / {est.infra_err_rate:.1%}",
        f"  cost per run            ${est.usd_per_run:.4f}",
        f"  smallest detectable gain  SELECT {gain:.1f} pts "
        f"({t.select_tasks} tasks x {t.select_reps} reps), "
        f"CONFIRM {hold:.1f} pts ({t.holdout_tasks} tasks x {t.holdout_reps} reps)",
        f"  cost per candidate      ${est.usd_per_candidate:.2f} "
        "(both arms through CONFIRM, plus smoke; excludes proposer and auditor)",
    ]
    return "\n".join(lines)


def _positive(value: int, flag: str) -> int:
    if value < 1:
        raise CliError(f"{flag} must be >= 1", EXIT_USAGE)
    return value


def cmd_estimate(args: argparse.Namespace, factory: RuntimeFactory) -> int:
    settings = settings_from_args(args)
    project = load_project(settings.repo)
    split = Split(args.split)
    if split not in (Split.TRAIN, Split.SELECT):
        raise CliError("only train and select tasks may be sampled for calibration", EXIT_USAGE)
    reps = _positive(args.reps, "--reps")
    n_tasks = _positive(args.tasks, "--tasks")
    pool = tasks_in(project.pack, split)
    if not pool:
        raise CliError(f"the pack has no {split.value} tasks to sample", EXIT_USAGE)
    tasks = pool[:n_tasks]
    with calibration_session(settings, project, factory) as session:
        plan = [(task, rep) for task in tasks for rep in range(reps)]
        session.run(plan)
        runs = session.runs()
    eff = effective_thresholds(project.pack, settings.thresholds)
    try:
        est = estimate(runs, eff)
    except ValueError as exc:
        raise CliError(
            f"no run was scored ({exc}): the agent or the grader is not working; "
            "check the target image, the run command and the grader image",
            EXIT_INFRA,
        ) from exc
    print(format_estimate(est, project.pack, eff, split=split, reps=reps))
    warnings = pack_warnings(est, project.pack, settings.thresholds, sampled_split=split)
    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print("No warnings: the pack looks usable for the configured acceptance rule.")
    return EXIT_OK


# --- aa-test ------------------------------------------------------------------------


@dataclass(frozen=True)
class AAResult:
    trials: int
    select_accepts: int
    false_accepts: int

    @property
    def rate(self) -> float:
        return self.false_accepts / self.trials if self.trials else 0.0

    @property
    def passed(self) -> bool:
        return self.rate <= MAX_FALSE_ACCEPT


def _split_arms(
    pool: Sequence[RunResult], per_arm: int, rng: random.Random
) -> tuple[list[RunResult], list[RunResult]]:
    """Per task, deal the champion's reps at random into two arms of ``per_arm`` each."""
    by_task: dict[str, list[RunResult]] = {}
    for r in final_attempts(pool).values():
        by_task.setdefault(r.key.task, []).append(r)
    a: list[RunResult] = []
    b: list[RunResult] = []
    for task, runs in sorted(by_task.items()):
        if len(runs) < 2 * per_arm:
            continue
        runs = rng.sample(runs, len(runs))
        for i in range(per_arm):
            a.append(replace(runs[i], key=RunKey(1, "aa-a", task, i, 0)))
            b.append(replace(runs[per_arm + i], key=RunKey(1, "aa-b", task, i, 0)))
    return a, b


def run_aa(
    pool: Sequence[RunResult],
    t: AcceptanceThresholds,
    *,
    trials: int,
    seed: int,
    n_resamples: int,
) -> AAResult:
    """Resample the champion's select reps into two arms, run the §4 rule on them, count accepts.

    Only select tasks are ever run (holdout queries are counted in the real archive and must
    not be spent on calibration). CONFIRM is simulated by a second, independent resample of
    the same pool relabelled as holdout, so ``false_accepts`` is the compounded SELECT then
    CONFIRM rate, as for a real null candidate.
    """
    rng = random.Random(seed)
    sel_ok = both = 0
    for i in range(trials):
        sa, sb = _split_arms(pool, t.select_reps, rng)
        ca, cb = _split_arms(pool, t.select_reps, rng)
        ca = [replace(r, split=Split.HOLDOUT) for r in ca]
        cb = [replace(r, split=Split.HOLDOUT) for r in cb]
        sel = decide_select(sa, sb, t, seed=seed * 1_000_003 + 2 * i, n_resamples=n_resamples)
        conf = decide_confirm(ca, cb, t, seed=seed * 1_000_003 + 2 * i + 1, n_resamples=n_resamples)
        sel_ok += sel.accepted
        both += accepted(sel, conf)
    return AAResult(trials, sel_ok, both)


RECORD = Path(".iterant") / "calibration.json"


def write_record(repo: Path, pack_hash: str, passed: bool, rate: float, trials: int) -> None:
    (repo / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (repo / RECORD).write_text(
        json.dumps({"pack_hash": pack_hash, "passed": passed, "rate": rate, "trials": trials}),
        encoding="utf-8",
    )


def calibration_problem(repo: Path, pack_hash: str) -> str | None:
    """Why ``run`` may not start (§4: the A/A test must pass first), or None if it may."""
    try:
        rec: object = json.loads((repo / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no A/A test has been run for this repo: run `iterant aa-test` first"
    if not isinstance(rec, dict):
        return "calibration record is unreadable: rerun `iterant aa-test`"
    fields: dict[str, object] = {str(k): v for k, v in rec.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if fields.get("pack_hash") != pack_hash:
        return "the pack changed since the last A/A test: rerun `iterant aa-test`"
    if fields.get("passed") is not True:
        return "the last A/A test failed: fix the pack or the stats, then rerun `iterant aa-test`"
    return None


def cmd_aa_test(args: argparse.Namespace, factory: RuntimeFactory) -> int:
    settings = settings_from_args(args)
    project = load_project(settings.repo)
    trials = _positive(args.trials, "--trials")
    resamples = _positive(args.resamples, "--resamples")
    eff = effective_thresholds(project.pack, settings.thresholds)
    sel_tasks = tasks_in(project.pack, Split.SELECT)[: eff.select_tasks]
    if not sel_tasks:
        raise CliError("the A/A test needs select tasks in the pack", EXIT_USAGE)
    with calibration_session(settings, project, factory) as session:
        session.run([(task, rep) for task in sel_tasks for rep in range(2 * eff.select_reps)])
        runs = session.runs()
    result = run_aa(runs, eff, trials=trials, seed=settings.seed, n_resamples=resamples)
    scored = {r.score for r in runs if r.score is not None}
    print(
        f"A/A test (champion vs champion) on pack {project.pack.name} v{project.pack.version}: "
        f"{len(runs)} champion select runs, {result.trials} resampled trials"
    )
    print(f"  SELECT passed alone  {result.select_accepts} of {result.trials}")
    print(
        f"  false accepts        {result.false_accepts} of {result.trials} "
        f"({result.rate:.1%}; limit {MAX_FALSE_ACCEPT:.0%}; 95% upper bound if none: "
        f"{min(1.0, 3 / result.trials):.1%})"
    )
    degenerate = len(scored) <= 1
    passed = result.passed and not degenerate
    write_record(project.repo, project.pack.content_hash, passed, result.rate, result.trials)
    if degenerate:
        raise CliError(
            "A/A test FAILED: every champion run scored the same, so no resample can differ and "
            "the test proves nothing. Use a pack where the champion sometimes fails and "
            "sometimes passes.",
            EXIT_CALIBRATION,
        )
    if not result.passed:
        raise CliError(
            f"A/A test FAILED: {result.rate:.1%} of identical-arm trials were accepted "
            f"(limit {MAX_FALSE_ACCEPT:.0%}). The stats or the pack are broken; do not run.",
            EXIT_CALIBRATION,
        )
    print("A/A test passed.")
    return EXIT_OK
