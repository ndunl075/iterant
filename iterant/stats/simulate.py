"""Synthetic paired runs with a known true gain, to check the §4 rule empirically.

No LLM and no sandbox. Each task has a champion pass probability drawn from a Beta
distribution (``difficulty_concentration`` sets how spread out task difficulty is);
the candidate passes with that probability plus ``gain`` (champion probability is
capped at ``1 - gain`` so the true gain is exactly ``gain`` on every task). Every run
is a Bernoulli draw. ``arm_coupling`` is the chance the two arms of one (task, rep)
share the same random draw (run noise common to both arms), which lowers the
discordance. Refused and infra-errored runs are drawn at the given rates and retried
once, as in §6. Everything is seeded and reproducible. Stdlib only.

``task_effect_sd`` adds a per-task shift to the candidate's pass probability (mean 0,
truncated symmetrically to stay inside [0, 1], so the true gain stays ``gain``): a null
diff then changes behaviour on particular tasks, which is what a real A/A looks like.
The two arms can have different refusal / infra rates, and the candidate can cost more
(``cost_multiplier``) or run longer (``wall_multiplier``), to exercise the guards
against real harm.

Scope limits. (1) The real orchestrator fixes the holdout set across candidates; here
every trial draws fresh SELECT and holdout tasks, so this does not model holdout
overfitting or its reuse. (2) The false-accept rate is per candidate: it is not per
generation and applies no multiple-comparison correction across candidates. (3) Trials
use far fewer resamples (200 by default, 2,000 in calibration) than production's
10,000.

``run_trials`` plays the full rule (``decide_select``, ``decide_confirm``,
``accepted``). ``analytic_power`` is ``power()`` at the discordance the simulation
actually produced, for the SELECT stage alone with the p-test only (a tiny
``delta_min``); the full rule is stricter, so compare its rate as an upper bound.
``power()`` assumes no persistent per-task effect, so with ``task_effect_sd > 0`` it
overstates power.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from statistics import NormalDist

from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split
from iterant.stats.acceptance import Decision, Reason, accepted, decide_confirm, decide_select
from iterant.stats.power import power, smallest_detectable_gain

CI_TRIALS = 300  # default: runs in CI in seconds
CI_RESAMPLES = 200
CALIBRATION_TRIALS = 5000  # ``calibration=True``
CALIBRATION_RESAMPLES = 2000

_CHAMP, _CAND = "champion", "candidate"
_Z = NormalDist().inv_cdf(0.975)
_GUARDS = {Reason.REFUSAL_GUARD, Reason.INFRA_GUARD}


@dataclass(frozen=True)
class SimConfig:
    gain: float = 0.0  # true per-task gain, as a score fraction (0.05 = +5 pts)
    mean_pass: float = 0.3  # mean champion pass probability across tasks
    difficulty_concentration: float = 2.0  # Beta(m*k, (1-m)*k); small k = very uneven tasks
    arm_coupling: float = 0.0  # chance both arms of a (task, rep) share one random draw
    task_effect_sd: float = 0.0  # sd of the mean-0 per-task shift of the candidate's p
    refusal_rate: float = 0.0  # per-attempt chance a run is refused (champion; the candidate
    infra_rate: float = 0.0  # too, unless its candidate_* rate below is set)
    candidate_refusal_rate: float | None = None
    candidate_infra_rate: float | None = None
    # Selective refuser: per stage, the candidate refuses (both attempts) on up to this many
    # (task, rep) slots where it failed and the champion passed, i.e. it tries to shed its losses.
    selective_refusals: int = 0
    cost_multiplier: float = 1.0  # candidate usd_micros relative to champion
    wall_multiplier: float = 1.0  # candidate wall_s relative to champion
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gain < 1.0:
            raise ValueError("gain must be in [0, 1)")
        if not 0.0 < self.mean_pass < 1.0 or self.difficulty_concentration <= 0:
            raise ValueError("mean_pass must be in (0, 1) and concentration > 0")
        rates = (
            self.arm_coupling,
            self.refusal_rate,
            self.infra_rate,
            self.candidate_refusal_rate,
            self.candidate_infra_rate,
        )
        if any(r is not None and not 0.0 <= r <= 1.0 for r in rates):
            raise ValueError("coupling and refusal / infra rates must be in [0, 1]")
        if self.task_effect_sd < 0 or min(self.cost_multiplier, self.wall_multiplier) <= 0:
            raise ValueError("task_effect_sd must be >= 0 and multipliers > 0")


@dataclass(frozen=True)
class TrialOutcome:
    select: Decision
    confirm: Decision | None  # None when the trial ran SELECT only
    accepted: bool
    discordance: float  # expected share of SELECT run pairs that differ in this trial


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    low: float  # Wilson 95% interval
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else 0.0


@dataclass(frozen=True)
class SimSummary:
    config: SimConfig
    trials: int
    accept: Proportion  # full rule: SELECT and CONFIRM
    select_pass: Proportion  # SELECT alone
    mean_discordance: float
    # Among trials where every check except the refusal / infra guards passed: the
    # share the refusal or infra guard vetoed.
    guard_veto: Proportion
    analytic_power: float | None  # None when ``gain`` is 0 or above the discordance


def wilson(successes: int, n: int, z: float = _Z) -> Proportion:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return Proportion(successes, n, 0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Proportion(successes, n, max(0.0, centre - half), min(1.0, centre + half))


def _draw_probs(rng: random.Random, cfg: SimConfig, n: int) -> list[tuple[float, float]]:
    """(champion p, candidate p) per task."""
    k, m = cfg.difficulty_concentration, cfg.mean_pass
    cap = 1.0 - cfg.gain
    out: list[tuple[float, float]] = []
    for _ in range(n):
        p = min(rng.betavariate(m * k, (1 - m) * k), cap)
        q = p + cfg.gain
        shift = rng.gauss(0.0, cfg.task_effect_sd) if cfg.task_effect_sd else 0.0
        room = min(q, 1.0 - q)  # truncate symmetrically so the shift keeps mean 0
        out.append((p, q + max(-room, min(room, shift))))
    return out


def _discordance(probs: Sequence[tuple[float, float]], cfg: SimConfig) -> float:
    """Expected share of run pairs that differ (ignores dropped runs)."""
    if not probs:
        return 0.0
    rho = cfg.arm_coupling
    return sum(rho * abs(q - p) + (1 - rho) * (p * (1 - q) + q * (1 - p)) for p, q in probs) / len(
        probs
    )


def _attempt(
    rng: random.Random,
    cfg: SimConfig,
    key: RunKey,
    split: Split,
    passed: bool,
    candidate: bool,
    force_refusal: bool = False,
) -> RunResult:
    refusal_rate, infra_rate = cfg.refusal_rate, cfg.infra_rate
    cost, wall = 1000, 10.0
    if candidate:
        if cfg.candidate_refusal_rate is not None:
            refusal_rate = cfg.candidate_refusal_rate
        if cfg.candidate_infra_rate is not None:
            infra_rate = cfg.candidate_infra_rate
        cost, wall = round(cost * cfg.cost_multiplier), wall * cfg.wall_multiplier
    u = rng.random()
    refusal = force_refusal or u < refusal_rate
    infra = not refusal and u < refusal_rate + infra_rate
    bad = refusal or infra
    return RunResult(
        key=key,
        split=split,
        score=None if bad else float(passed),
        in_tok=1,
        out_tok=1,
        usd_micros=cost,
        wall_s=wall,
        refusal=refusal,
        infra_err=infra,
    )


def _runs(
    rng: random.Random,
    cfg: SimConfig,
    probs: Sequence[tuple[float, float]],
    reps: int,
    split: Split,
    batch: int,
) -> tuple[list[RunResult], list[RunResult]]:
    cand: list[RunResult] = []
    champ: list[RunResult] = []
    shed = cfg.selective_refusals
    for i, (p, q) in enumerate(probs):
        for rep in range(reps):
            u_h = rng.random()
            u_c = u_h if rng.random() < cfg.arm_coupling else rng.random()
            refuse = shed > 0 and u_h < p and not u_c < q  # a slot the candidate would lose
            shed -= refuse
            for arm, out, passed, is_cand in (
                (_CHAMP, champ, u_h < p, False),
                (_CAND, cand, u_c < q, True),
            ):
                key = RunKey(batch, arm, f"t{i}", rep, 0)
                force = is_cand and refuse
                r = _attempt(rng, cfg, key, split, passed, is_cand, force)
                out.append(r)
                if r.score is None:  # one retry of a refused / errored run
                    retry = replace(key, attempt=1)
                    out.append(_attempt(rng, cfg, retry, split, passed, is_cand, force))
    return cand, champ


def simulate_trial(
    cfg: SimConfig,
    seed: int,
    n_resamples: int = CI_RESAMPLES,
    *,
    select_only: bool = False,
) -> TrialOutcome:
    """One full SELECT + CONFIRM round on fresh synthetic tasks. ``select_only`` skips
    CONFIRM (faster; ``accepted`` is then SELECT's verdict)."""
    t = cfg.thresholds
    rng = random.Random(seed)
    sel_p = _draw_probs(rng, cfg, t.select_tasks)
    hold_p = _draw_probs(rng, cfg, t.holdout_tasks)
    s_c, s_h = _runs(rng, cfg, sel_p, t.select_reps, Split.SELECT, 1)
    sel_seed, conf_seed = rng.randrange(2**31), rng.randrange(2**31)
    sel = decide_select(s_c, s_h, t, seed=sel_seed, n_resamples=n_resamples)
    disc = _discordance(sel_p, cfg)
    if select_only:
        return TrialOutcome(sel, None, sel.accepted, disc)
    h_c, h_h = _runs(rng, cfg, hold_p, t.holdout_reps, Split.HOLDOUT, 2)
    conf = decide_confirm(h_c, h_h, t, seed=conf_seed, n_resamples=n_resamples)
    return TrialOutcome(sel, conf, accepted(sel, conf), disc)


def _other_checks_pass(d: Decision | None) -> bool:
    return d is None or not set(d.reasons) - _GUARDS


def _guard_vetoed(d: Decision | None) -> bool:
    return d is not None and bool(set(d.reasons) & _GUARDS)


def run_trials(
    cfg: SimConfig,
    *,
    seed: int = 0,
    trials: int | None = None,
    n_resamples: int | None = None,
    calibration: bool = False,
    select_only: bool = False,
) -> SimSummary:
    """Run many seeded trials. Defaults are CI size; ``calibration=True`` is the big
    size. Explicit ``trials`` / ``n_resamples`` override either. ``select_only`` skips
    CONFIRM, so ``accept`` equals ``select_pass``."""
    n = trials if trials is not None else (CALIBRATION_TRIALS if calibration else CI_TRIALS)
    res = (
        n_resamples
        if n_resamples is not None
        else (CALIBRATION_RESAMPLES if calibration else CI_RESAMPLES)
    )
    outs = [
        simulate_trial(cfg, seed * 1_000_003 + i, res, select_only=select_only) for i in range(n)
    ]
    clean = [o for o in outs if _other_checks_pass(o.select) and _other_checks_pass(o.confirm)]
    vetoed = sum(_guard_vetoed(o.select) or _guard_vetoed(o.confirm) for o in clean)
    disc = sum(o.discordance for o in outs) / n if n else 0.0
    return SimSummary(
        config=cfg,
        trials=n,
        accept=wilson(sum(o.accepted for o in outs), n),
        select_pass=wilson(sum(o.select.accepted for o in outs), n),
        mean_discordance=disc,
        guard_veto=wilson(vetoed, len(clean)),
        analytic_power=_analytic(cfg, disc),
    )


def _analytic(cfg: SimConfig, discordance: float) -> float | None:
    if cfg.gain <= 0.0 or cfg.gain > discordance or discordance <= 0.0:
        return None
    t = cfg.thresholds
    return power(t.select_tasks, cfg.gain, discordance, t.select_reps, t.select_p)


def analytic_power_at(summary: SimSummary, alpha: float) -> float | None:
    """``power()`` at the simulated discordance, but at test size ``alpha``. The
    sign-flip test on these discrete scores rejects a true null less than 5% of the time
    (about 3 to 4% measured), so the matching analytic value uses that size."""
    cfg = summary.config
    disc = summary.mean_discordance
    if cfg.gain <= 0.0 or cfg.gain > disc:
        return None
    t = cfg.thresholds
    return power(t.select_tasks, cfg.gain, disc, t.select_reps, alpha)


def analytic_smallest_gain(summary: SimSummary) -> float:
    """``smallest_detectable_gain`` at the simulated discordance and SELECT size."""
    t = summary.config.thresholds
    return smallest_detectable_gain(
        t.select_tasks, summary.mean_discordance, t.select_reps, t.select_p
    )


def power_curve(
    gains: Sequence[float],
    base: SimConfig | None = None,
    *,
    seed: int = 0,
    trials: int | None = None,
    n_resamples: int | None = None,
    calibration: bool = False,
    select_only: bool = False,
) -> list[SimSummary]:
    """One ``run_trials`` summary per true gain, each with its own seed (``seed`` plus
    the index), so noise is independent across points."""
    cfg = base or SimConfig()
    return [
        run_trials(
            replace(cfg, gain=g),
            seed=seed + i,
            trials=trials,
            n_resamples=n_resamples,
            calibration=calibration,
            select_only=select_only,
        )
        for i, g in enumerate(gains)
    ]
