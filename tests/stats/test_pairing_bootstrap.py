from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from iterant.contracts import RunResult, Split
from iterant.stats import pair_runs, paired_signflip

MakeRun = Callable[..., RunResult]


def test_pairs_by_task_and_averages_reps(mk: MakeRun) -> None:
    cand = [mk("a", 1.0, rep=0), mk("a", 0.0, rep=1), mk("b", 1.0)]
    champ = [mk("a", 0.0, cand="h", rep=0), mk("a", 0.0, cand="h", rep=1), mk("b", 1.0, cand="h")]
    p = pair_runs(cand, champ)
    assert p.task_ids == ("a", "b")
    assert p.deltas == (0.5, 0.0)
    assert p.delta == 0.25
    assert p.n_dropped == 0


def test_drops_task_from_both_arms_when_either_side_refused_after_retry(mk: MakeRun) -> None:
    cand = [mk("a", None, refusal=True), mk("b", 1.0), mk("c", 1.0)]
    champ = [mk("a", 0.0, cand="h"), mk("b", None, cand="h", infra=True), mk("c", 0.0, cand="h")]
    p = pair_runs(cand, champ)
    assert p.task_ids == ("c",)
    assert p.dropped_tasks == ("a", "b")
    assert p.n_dropped == 2
    assert len(p.candidate_runs) == len(p.champion_runs) == 1


def test_retry_replaces_failed_first_attempt(mk: MakeRun) -> None:
    cand = [mk("a", None, infra=True, attempt=0), mk("a", 1.0, attempt=1)]
    champ = [mk("a", 0.0, cand="h")]
    p = pair_runs(cand, champ)
    assert p.deltas == (1.0,)
    assert p.n_dropped == 0
    # a retry that fails again drops the task
    cand2 = [mk("a", 1.0, attempt=0), mk("a", None, refusal=True, attempt=1)]
    assert pair_runs(cand2, champ).dropped_tasks == ("a",)


def test_task_missing_from_one_arm_is_an_error(mk: MakeRun) -> None:
    with pytest.raises(ValueError, match="only one arm.*'b'"):
        pair_runs([mk("a", 1.0), mk("b", 1.0)], [mk("a", 0.0, cand="h")])


def test_arms_must_be_single_distinct_candidates(mk: MakeRun) -> None:
    with pytest.raises(ValueError, match="same candidate"):
        pair_runs([mk("a", 1.0)], [mk("a", 0.0)])
    with pytest.raises(ValueError, match="one candidate"):
        pair_runs([mk("a", 1.0), mk("b", 1.0, cand="c2")], [mk("a", 0.0, cand="h")])


def test_pairing_keeps_all_attempts_of_kept_tasks(mk: MakeRun) -> None:
    cand = [mk("a", None, refusal=True, usd=5000), mk("a", 1.0, attempt=1, usd=100)]
    p = pair_runs(cand, [mk("a", 0.0, cand="h")])
    assert len(p.candidate_runs) == 1
    assert sum(r.usd_micros for r in p.candidate_attempts) == 5100


def test_rejects_mixed_batches_splits_and_duplicates(mk: MakeRun) -> None:
    with pytest.raises(ValueError, match="one batch"):
        pair_runs([mk("a", 1.0, batch=1)], [mk("a", 0.0, cand="h", batch=2)])
    with pytest.raises(ValueError, match="select"):
        pair_runs([mk("a", 1.0, split=Split.TRAIN)], [mk("a", 0.0, cand="h")], split=Split.SELECT)
    with pytest.raises(ValueError, match="duplicate"):
        pair_runs([mk("a", 1.0), mk("a", 1.0)], [mk("a", 0.0, cand="h")])


def test_signflip_exact_hand_cases() -> None:
    assert paired_signflip([], seed=0).p == 1.0
    assert paired_signflip([0.0] * 10, seed=0).p == 1.0
    assert paired_signflip([-0.1] * 5, seed=0).p == 1.0  # all-negative: every pattern is >=
    # all positive over n tasks: exactly 2**-n
    r = paired_signflip([0.1, 0.2, 0.3], seed=0)
    assert r.p == 1 / 8
    assert r.delta == pytest.approx(0.2)
    # [+1, +1, -1]: sums over 8 sign patterns are 3,1,1,-1,1,-1,-1,-3; observed 1 -> 4 of 8 >= 1
    assert paired_signflip([1.0, 1.0, -1.0], seed=0).p == 0.5
    # zeros are ignored but count towards the mean
    r = paired_signflip([1.0, 0.0, 0.0, 0.0], seed=0)
    assert r.p == 0.5 and r.delta == 0.25 and r.n == 4


def test_signflip_monte_carlo_path() -> None:
    assert paired_signflip([1.0] * 10 + [0.0] * 90, seed=2).p < 0.01
    assert paired_signflip([1.0] * 20 + [-1.0] * 20, seed=2).p > 0.4
    # many nonzero all-positive gains: p floors at 1/(B+1)
    assert paired_signflip([0.1] * 40, seed=0, n_resamples=99).p == 1 / 100


def test_signflip_is_deterministic_under_seed() -> None:
    d = [0.5, -0.5, 0.25, 1.0, 0.0, 0.5, -1.0, 0.1, 0.5, 0.3, -0.2, 0.7, 0.4, -0.1, 0.6]
    a = paired_signflip(d, seed=7, n_resamples=300)
    assert a == paired_signflip(d, seed=7, n_resamples=300)
    assert {paired_signflip(d, seed=s, n_resamples=300).p for s in range(5)} != {a.p}


def test_signflip_rejects_bad_resamples() -> None:
    with pytest.raises(ValueError):
        paired_signflip([0.1, 0.2], seed=0, n_resamples=0)


def test_signflip_p_value_is_calibrated_under_the_null() -> None:
    """The p-value alone (no delta_min, no CONFIRM): 0/1 scores, k=1, 15% discordant."""
    rng = random.Random(2024)
    sims, n = 2500, 80
    rejects = 0
    for i in range(sims):
        # each task: champion/candidate disagree w.p. 0.15, direction a fair coin
        deltas = [
            (1.0 if rng.random() < 0.5 else -1.0) if rng.random() < 0.15 else 0.0 for _ in range(n)
        ]
        rejects += paired_signflip(deltas, seed=i, n_resamples=100).p < 0.05
    assert rejects / sims <= 0.06, rejects / sims
