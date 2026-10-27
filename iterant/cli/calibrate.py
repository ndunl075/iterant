"""``iterant estimate`` and ``iterant aa-test``: calibration before any real spend (§2, §4).

Both run the champion on the pack through the orchestrator's run machinery and keep the rows in
``.iterant/calibration.db``, a separate archive with the same schema, so calibration never
mixes with the real archive (the failure digest, the holdout query count and the batch
resume logic all read the real one). Both take the writer lock: not while ``run`` is going.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
from iterant.stats import (
    DEFAULT_PRESET,
    Estimate,
    accepted,
    decide_confirm,
    decide_select,
    estimate,
    rule_reps_needed,
    rule_tasks_needed,
)
from iterant.stats.pairing import final_attempts

CALIBRATION_DB = Path(".iterant") / "calibration.db"
CHAMPION_ID = "champion"
MAX_FALSE_ACCEPT = 0.05  # §4: A/A false accepts must stay at or below 5%
# What `run` demands of the calibration it relies on (the one place these live). A weaker
# aa-test or estimate still runs and records, but does not unlock a real run.
MIN_AA_TRIALS = 100
MIN_AA_RESAMPLES = 1000
MIN_ESTIMATE_SPLIT = "select"  # the split the acceptance rule's SELECT stage draws from
# estimate must sample at least the configured select task count (or all the pack has) at
# least the configured select reps (see ``_param_problems``).
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
    settings: Settings, project: Project, factory: RuntimeFactory, kind: str
) -> Generator[_Session]:
    """Writer lock, runtime on the calibration archive, grader preflight, one new batch."""
    lock = cli_lock(project.repo)
    acquire_retrying(lock)
    try:
        yield from _session(settings, project, factory, kind)
    finally:
        lock.release()


def _session(
    settings: Settings, project: Project, factory: RuntimeFactory, kind: str
) -> Generator[_Session]:
    with WriterLock(project.repo):
        # invariant 7: the pack on disk must still be the pack that was loaded
        verify(project.pack, project.pack.content_hash, project.pack_root)
        # a newer attempt invalidates an older pass until it finishes (an abort leaves it so)
        set_record(project.repo, kind, fingerprint(settings, project), "running", "", {})
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


POWER_WARN_GAIN = 0.08  # warn when the rule rarely accepts a true gain this large...
POWER_WARN_BELOW = 0.30  # ...(below this chance)
POWER_TARGET = 0.50


def power_warnings(est: Estimate, t: AcceptanceThresholds, preset: str) -> list[str]:
    """Plain warning when the pack is too small to accept a real +8 pt gain."""
    at = dict(est.power_table).get(POWER_WARN_GAIN)
    if at is None or at >= POWER_WARN_BELOW:
        return []
    tasks = rule_tasks_needed(POWER_WARN_GAIN, est.discordance, t, POWER_TARGET)
    reps = rule_reps_needed(POWER_WARN_GAIN, est.discordance, t, POWER_TARGET)
    fix: list[str] = []
    if tasks is not None:
        hold = max(1, round(tasks * t.holdout_tasks / t.select_tasks))
        fix.append(f"about {tasks} select tasks (and {hold} holdout tasks) at {t.select_reps} reps")
    if reps is not None:
        fix.append(f"{reps} reps per task at the current task counts")
    msg = (
        f"the pack is too small: a true +{POWER_WARN_GAIN * 100:.0f} pt gain would be accepted "
        f"only {at:.0%} of the time under the {preset} preset."
    )
    if fix:
        msg += f" Reaching {POWER_TARGET:.0%} needs {' or '.join(fix)}."
    else:
        msg += f" Even a very large pack or many reps would not reach {POWER_TARGET:.0%}."
    if preset == "strict":
        msg += " The strict preset needs more tasks than balanced: consider --preset balanced."
    return [msg]


def format_estimate(
    est: Estimate,
    pack: Pack,
    t: AcceptanceThresholds,
    *,
    split: Split,
    reps: int,
    preset: str = DEFAULT_PRESET,
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
        f"  minimum gain worth keeping  {t.delta_min * 100:.0f} pts (delta_min)",
        f"Chance of accepting a true gain under the {preset} preset "
        f"(select p {t.select_p}, holdout p {t.holdout_p}; an approximation: a normal model "
        "that ignores the guards and runs a little above the simulator):",
    ]
    for g, pw in est.power_table:
        lines.append(f"  +{g * 100:.0f} pts  {pw:.0%}")
    lines.append(
        f"  A/A false-accept estimate  {est.aa_false_accept:.1%} (nominal upper bound; "
        "`iterant aa-test` measures it)"
    )
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
    with calibration_session(settings, project, factory, ESTIMATE) as session:
        plan = [(task, rep) for task in tasks for rep in range(reps)]
        session.run(plan)
        runs = session.runs()
    eff = effective_thresholds(project.pack, settings.thresholds)
    try:
        est = estimate(runs, eff)
    except ValueError as exc:
        set_record(project.repo, ESTIMATE, fingerprint(settings, project), "fail", str(exc), {})
        raise CliError(
            f"no run was scored ({exc}): the agent or the grader is not working; "
            "check the target image, the run command and the grader image",
            EXIT_INFRA,
        ) from exc
    print(format_estimate(est, project.pack, eff, split=split, reps=reps, preset=settings.preset))
    warnings = pack_warnings(est, project.pack, settings.thresholds, sampled_split=split)
    warnings += power_warnings(est, eff, settings.preset)
    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print("No warnings: the pack looks usable for the configured acceptance rule.")
    set_record(
        project.repo,
        ESTIMATE,
        fingerprint(settings, project),
        "warn" if warnings else "pass",
        " | ".join(warnings)[:1000],
        {"split": split.value, "tasks": len(tasks), "reps": reps},
    )
    return EXIT_OK


# --- aa-test ------------------------------------------------------------------------


@dataclass(frozen=True)
class AAResult:
    trials: int
    select_accepts: int
    false_accepts: int
    select_dropped: int = 0  # tasks with too few runs for the SELECT legs
    confirm_dropped: int = 0  # ... and for the simulated CONFIRM legs

    @property
    def rate(self) -> float:
        return self.false_accepts / self.trials if self.trials else 0.0

    @property
    def passed(self) -> bool:
        return self.rate <= MAX_FALSE_ACCEPT


def _by_task(pool: Sequence[RunResult]) -> dict[str, list[RunResult]]:
    by_task: dict[str, list[RunResult]] = {}
    for r in final_attempts(pool).values():
        by_task.setdefault(r.key.task, []).append(r)
    return by_task


def _count_under_sampled(pool: Sequence[RunResult], per_arm: int) -> int:
    """Tasks with fewer than ``2 * per_arm`` runs: ``_split_arms`` cannot use them."""
    return sum(1 for runs in _by_task(pool).values() if len(runs) < 2 * per_arm)


def _split_arms(
    pool: Sequence[RunResult], per_arm: int, n_tasks: int, rng: random.Random
) -> tuple[list[RunResult], list[RunResult]]:
    """Pick ``n_tasks`` tasks at random; per task, deal the champion's reps at random into two
    arms of ``per_arm`` each. Tasks with too few runs are skipped (``_count_under_sampled``
    counts them so the caller can say so)."""
    by_task = _by_task(pool)
    ids = sorted(t for t, runs in by_task.items() if len(runs) >= 2 * per_arm)
    chosen = sorted(rng.sample(ids, min(n_tasks, len(ids))))
    a: list[RunResult] = []
    b: list[RunResult] = []
    for task in chosen:
        runs = rng.sample(by_task[task], len(by_task[task]))
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
    not be spent on calibration). The SELECT leg has the select shape (``select_tasks`` tasks,
    ``select_reps`` per arm). The CONFIRM leg has the holdout shape (``holdout_tasks`` tasks,
    ``holdout_reps`` per arm), drawn independently from the same select pool and relabelled as
    holdout, so ``false_accepts`` is the compounded SELECT then CONFIRM rate of a real null
    candidate. The pool needs ``2 * max(select_reps, holdout_reps)`` reps per task.
    """
    rng = random.Random(seed)
    sel_ok = both = 0
    for i in range(trials):
        sa, sb = _split_arms(pool, t.select_reps, t.select_tasks, rng)
        ca, cb = _split_arms(pool, t.holdout_reps, t.holdout_tasks, rng)
        ca = [replace(r, split=Split.HOLDOUT) for r in ca]
        cb = [replace(r, split=Split.HOLDOUT) for r in cb]
        sel = decide_select(sa, sb, t, seed=seed * 1_000_003 + 2 * i, n_resamples=n_resamples)
        conf = decide_confirm(ca, cb, t, seed=seed * 1_000_003 + 2 * i + 1, n_resamples=n_resamples)
        sel_ok += sel.accepted
        both += accepted(sel, conf)
    return AAResult(
        trials,
        sel_ok,
        both,
        _count_under_sampled(pool, t.select_reps),
        _count_under_sampled(pool, t.holdout_reps),
    )


def dropped_warning(result: AAResult) -> str | None:
    """A line for aa-test output when tasks were left out of the resampling, else None."""
    if not (result.select_dropped or result.confirm_dropped):
        return None
    return (
        f"WARNING: {result.select_dropped} select and {result.confirm_dropped} confirm task(s) "
        "had too few champion runs and were left out of the A/A resampling (usually infra "
        "errors or refusals); the test used the remaining tasks"
    )


RECORD = Path(".iterant") / "calibration.json"
ESTIMATE = "estimate"
AA_TEST = "aa-test"
_OK_STATES = {ESTIMATE: ("pass", "warn"), AA_TEST: ("pass",)}


# Limits that do not change what a calibration measures.
_NOT_CALIBRATION = ("holdout_query_warn", "holdout_query_max", "max_champion_refusal_rate")


def _calibration_thresholds(t: AcceptanceThresholds) -> dict[str, object]:
    d: dict[str, object] = json.loads(json.dumps(asdict(t)))
    for name in _NOT_CALIBRATION:
        d.pop(name, None)
    return d


def recorded_thresholds(repo: Path) -> AcceptanceThresholds | None:
    """The thresholds the last calibration record (estimate, else aa-test) ran under, or None.
    The record omits the limits that do not affect calibration, so those take their defaults."""
    records = _load(repo)
    for kind in (ESTIMATE, AA_TEST):
        fp = _as_map(_as_map(records.get(kind)).get("fingerprint"))
        raw = _as_map(fp.get("thresholds"))
        if raw:
            try:
                return AcceptanceThresholds(**raw)  # pyright: ignore[reportArgumentType]
            except (TypeError, ValueError):
                return None
    return None


def fingerprint(settings: Settings, project: Project) -> dict[str, object]:
    """Everything a calibration result depends on. A record only counts for an identical one."""
    cfg = json.dumps(asdict(project.config), sort_keys=True, default=str)
    return {
        "pack_hash": project.pack.content_hash,
        "preset": settings.preset,
        "thresholds": _calibration_thresholds(settings.thresholds),
        "profile": settings.profile.value,
        "backend": settings.backend,
        "grader_image": project.pack.grader_image,
        "target_model": json.loads(json.dumps(asdict(project.config.target.model))),
        "config_hash": hashlib.sha256(cfg.encode()).hexdigest(),
    }


def _as_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _load(repo: Path) -> dict[str, object]:
    try:
        raw: object = json.loads((repo / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return _as_map(raw)


def set_record(
    repo: Path,
    kind: str,
    fp: dict[str, object],
    state: str,
    detail: str,
    params: dict[str, object],
) -> None:
    """``state`` is running (just started: invalidates any older result), pass, warn or fail.
    ``params`` are how the calibration was run (trials, resamples, seed; split, tasks, reps):
    ``run`` checks them against the minimums above."""
    records = _load(repo)
    records[kind] = {"fingerprint": fp, "state": state, "detail": detail, "params": params}
    (repo / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (repo / RECORD).write_text(json.dumps(records, sort_keys=True), encoding="utf-8")


def _int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _param_problems(
    kind: str, params: dict[str, object], settings: Settings, project: Project
) -> list[str]:
    """Why a passing record is too weak to unlock a real run."""
    out: list[str] = []
    if kind == AA_TEST:
        if _int(params.get("trials")) < MIN_AA_TRIALS:
            out.append(f"{_int(params.get('trials'))} trials (MIN_AA_TRIALS = {MIN_AA_TRIALS})")
        if _int(params.get("resamples")) < MIN_AA_RESAMPLES:
            out.append(
                f"{_int(params.get('resamples'))} resamples (MIN_AA_RESAMPLES = {MIN_AA_RESAMPLES})"
            )
        return out
    t = settings.thresholds
    need_tasks = min(t.select_tasks, len(tasks_in(project.pack, Split.SELECT)))
    if params.get("split") != MIN_ESTIMATE_SPLIT:
        out.append(f"split {params.get('split')!r} (MIN_ESTIMATE_SPLIT = {MIN_ESTIMATE_SPLIT!r})")
    if _int(params.get("tasks")) < need_tasks:
        out.append(f"{_int(params.get('tasks'))} tasks (the select task count, {need_tasks})")
    if _int(params.get("reps")) < t.select_reps:
        out.append(f"{_int(params.get('reps'))} reps (the select reps, {t.select_reps})")
    return out


def calibration_status(
    repo: Path, fp: dict[str, object], settings: Settings, project: Project
) -> tuple[list[str], list[str]]:
    """``(problems, notes)``: why ``run`` may not start (§4: estimate and aa-test must pass for
    exactly this pack, rule, profile, model and config), and non-blocking notes."""
    records = _load(repo)
    problems: list[str] = []
    notes: list[str] = []
    for kind in (ESTIMATE, AA_TEST):
        fields = _as_map(records.get(kind))
        if not fields:
            problems.append(f"no `iterant {kind}` has been run: run it first")
            continue
        old = _as_map(fields.get("fingerprint"))
        if old != fp:
            changed = sorted(k for k in fp if old.get(k) != fp[k]) or ["record"]
            msg = (
                f"`iterant {kind}` was run for something else ({', '.join(changed)} changed): "
                "rerun it"
            )
            if "preset" in changed or "thresholds" in changed:
                msg += (
                    f". The active preset is {fp.get('preset')!r}; the default `balanced` "
                    "(select p 0.10, holdout p 0.5) is looser than the old default "
                    "(0.05, 0.2). Pass `--preset strict` for the stricter p-values"
                )
            problems.append(msg)
            continue
        state = fields.get("state")
        if state not in _OK_STATES[kind]:
            why = "did not finish" if state == "running" else "failed"
            problems.append(f"the last `iterant {kind}` {why}: fix that, then rerun it")
        else:
            weak = _param_problems(kind, _as_map(fields.get("params")), settings, project)
            if weak:
                problems.append(
                    f"the last `iterant {kind}` was too weak to rely on ({'; '.join(weak)}): "
                    "rerun it with more"
                )
        if state == "warn":
            notes.append(f"`iterant estimate` warned: {fields.get('detail')}")
    return problems, notes


def cmd_aa_test(args: argparse.Namespace, factory: RuntimeFactory) -> int:
    settings = settings_from_args(args)
    project = load_project(settings.repo)
    trials = _positive(args.trials, "--trials")
    resamples = _positive(args.resamples, "--resamples")
    eff = effective_thresholds(project.pack, settings.thresholds)
    sel_tasks = tasks_in(project.pack, Split.SELECT)  # all of them: legs draw from the pool
    pool_reps = max(eff.select_reps, eff.holdout_reps)
    if not sel_tasks:
        raise CliError("the A/A test needs select tasks in the pack", EXIT_USAGE)
    if eff.holdout_tasks > len(sel_tasks):
        raise CliError(
            f"the A/A test simulates CONFIRM with {eff.holdout_tasks} tasks drawn from the select "
            f"tasks, but the pack has only {len(sel_tasks)} select tasks: add select tasks or "
            "lower --holdout-tasks",
            EXIT_USAGE,
        )
    with calibration_session(settings, project, factory, AA_TEST) as session:
        session.run([(task, rep) for task in sel_tasks for rep in range(2 * pool_reps)])
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
    warning = dropped_warning(result)
    if warning:
        print(f"  {warning}")
    degenerate = len(scored) <= 1
    passed = result.passed and not degenerate
    set_record(
        project.repo,
        AA_TEST,
        fingerprint(settings, project),
        "pass" if passed else "fail",
        f"{result.false_accepts}/{result.trials} false accepts",
        {"trials": trials, "resamples": resamples, "seed": settings.seed},
    )
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
