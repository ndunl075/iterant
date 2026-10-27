"""Synthetic paired runs with a known true gain, to check the §4 rule empirically.

No LLM and no sandbox. Each task has a champion pass probability drawn from a Beta
distribution (``difficulty_concentration`` sets how spread out task difficulty is);
the candidate passes with that probability plus ``gain`` (champion probability is
capped at ``1 - gain`` so the true gain is exactly ``gain`` on every task). Every run
is a Bernoulli draw. ``arm_coupling`` is the chance the two arms of one (task, rep)
share the same random draw (run noise common to both arms), which lowers the
discordance. Refused and infra-errored runs are drawn at the given rates and retried
once, as in §6. Everything is seeded and reproducible. Stdlib only.

``run_trials`` plays the full rule (``decide_select``, ``decide_confirm``,
``accepted``). ``analytic_power`` is ``power()`` at the discordance the simulation
actually produced, for the SELECT stage alone with the p-test only (a tiny
``delta_min``); the full rule is stricter, so compare its rate as an upper bound.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from statistics import NormalDist

from iterant.contracts import AcceptanceThresholds, RunKey, RunResult, Split
from iterant.stats.acceptance import Decision, accepted, decide_confirm, decide_select
from iterant.stats.power import power, smallest_detectable_gain

CI_TRIALS = 300  # default: runs in CI in seconds
CI_RESAMPLES = 200
CALIBRATION_TRIALS = 5000  # ``calibration=True``
CALIBRATION_RESAMPLES = 2000

_CHAMP, _CAND = "champion", "candidate"
_Z = NormalDist().inv_cdf(0.975)


@dataclass(frozen=True)
class SimConfig:
    gain: float = 0.0  # true per-task gain, as a score fraction (0.05 = +5 pts)
    mean_pass: float = 0.3  # mean champion pass probability across tasks
    difficulty_concentration: float = 2.0  # Beta(m*k, (1-m)*k); small k = very uneven tasks
    arm_coupling: float = 0.0  # chance both arms of a (task, rep) share one random draw
    refusal_rate: float = 0.0  # per-attempt chance a run is refused (both arms alike)
    infra_rate: float = 0.0  # per-attempt chance a run errors in infrastructure
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gain < 1.0:
            raise ValueError("gain must be in [0, 1)")
        if not 0.0 < self.mean_pass < 1.0 or self.difficulty_concentration <= 0:
            raise ValueError("mean_pass must be in (0, 1) and concentration > 0")
        for name in ("arm_coupling", "refusal_rate", "infra_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class TrialOutcome:
    select: Decision
    confirm: Decision
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


def _draw_probs(rng: random.Random, cfg: SimConfig, n: int) -> list[float]:
    k, m = cfg.difficulty_concentration, cfg.mean_pass
    cap = 1.0 - cfg.gain
    return [min(rng.betavariate(m * k, (1 - m) * k), cap) for _ in range(n)]


def _discordance(probs: Sequence[float], cfg: SimConfig) -> float:
    g, rho = cfg.gain, cfg.arm_coupling
    if not probs:
        return 0.0
    ind = [p * (1 - p - g) + (p + g) * (1 - p) for p in probs]
    return sum(rho * g + (1 - rho) * d for d in ind) / len(probs)


def _attempt(
    rng: random.Random, cfg: SimConfig, key: RunKey, split: Split, passed: bool
) -> RunResult:
    u = rng.random()
    refusal = u < cfg.refusal_rate
    infra = not refusal and u < cfg.refusal_rate + cfg.infra_rate
    bad = refusal or infra
    return RunResult(
        key=key,
        split=split,
        score=None if bad else float(passed),
        in_tok=1,
        out_tok=1,
        usd_micros=1000,
        wall_s=10.0,
        refusal=refusal,
        infra_err=infra,
    )


def _runs(
    rng: random.Random, cfg: SimConfig, probs: Sequence[float], reps: int, split: Split, batch: int
) -> tuple[list[RunResult], list[RunResult]]:
    cand: list[RunResult] = []
    champ: list[RunResult] = []
    for i, p in enumerate(probs):
        for rep in range(reps):
            u_h = rng.random()
            u_c = u_h if rng.random() < cfg.arm_coupling else rng.random()
            for arm, out, passed in (
                (_CHAMP, champ, u_h < p),
                (_CAND, cand, u_c < p + cfg.gain),
            ):
                r = _attempt(rng, cfg, RunKey(batch, arm, f"t{i}", rep, 0), split, passed)
                out.append(r)
                if r.score is None:  # one retry of a refused / errored run
                    out.append(_attempt(rng, cfg, replace(r.key, attempt=1), split, passed))
    return cand, champ


def simulate_trial(cfg: SimConfig, seed: int, n_resamples: int = CI_RESAMPLES) -> TrialOutcome:
    """One full SELECT + CONFIRM round on fresh synthetic tasks."""
    t = cfg.thresholds
    rng = random.Random(seed)
    sel_p = _draw_probs(rng, cfg, t.select_tasks)
    hold_p = _draw_probs(rng, cfg, t.holdout_tasks)
    s_c, s_h = _runs(rng, cfg, sel_p, t.select_reps, Split.SELECT, 1)
    h_c, h_h = _runs(rng, cfg, hold_p, t.holdout_reps, Split.HOLDOUT, 2)
    sel_seed, conf_seed = rng.randrange(2**31), rng.randrange(2**31)
    sel = decide_select(s_c, s_h, t, seed=sel_seed, n_resamples=n_resamples)
    conf = decide_confirm(h_c, h_h, t, seed=conf_seed, n_resamples=n_resamples)
    return TrialOutcome(sel, conf, accepted(sel, conf), _discordance(sel_p, cfg))


def run_trials(
    cfg: SimConfig,
    *,
    seed: int = 0,
    trials: int | None = None,
    n_resamples: int | None = None,
    calibration: bool = False,
) -> SimSummary:
    """Run many seeded trials. Defaults are CI size; ``calibration=True`` is the big
    size. Explicit ``trials`` / ``n_resamples`` override either."""
    n = trials if trials is not None else (CALIBRATION_TRIALS if calibration else CI_TRIALS)
    res = (
        n_resamples
        if n_resamples is not None
        else (CALIBRATION_RESAMPLES if calibration else CI_RESAMPLES)
    )
    outs = [simulate_trial(cfg, seed * 1_000_003 + i, res) for i in range(n)]
    disc = sum(o.discordance for o in outs) / n if n else 0.0
    return SimSummary(
        config=cfg,
        trials=n,
        accept=wilson(sum(o.accepted for o in outs), n),
        select_pass=wilson(sum(o.select.accepted for o in outs), n),
        mean_discordance=disc,
        analytic_power=_analytic(cfg, disc),
    )


def _analytic(cfg: SimConfig, discordance: float) -> float | None:
    if cfg.gain <= 0.0 or cfg.gain > discordance or discordance <= 0.0:
        return None
    t = cfg.thresholds
    return power(t.select_tasks, cfg.gain, discordance, t.select_reps, t.select_p)


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
) -> list[SimSummary]:
    """One ``run_trials`` summary per true gain (same seed for each, so the curve is
    smooth)."""
    cfg = base or SimConfig()
    return [
        run_trials(
            replace(cfg, gain=g),
            seed=seed,
            trials=trials,
            n_resamples=n_resamples,
            calibration=calibration,
        )
        for g in gains
    ]
