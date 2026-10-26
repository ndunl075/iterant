from __future__ import annotations

from collections.abc import Callable

import pytest

from iterant.contracts import RunResult, Split
from iterant.stats import pair_runs, paired_bootstrap

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
    champ = [mk("a", 0.0), mk("b", None, infra=True), mk("c", 0.0)]
    p = pair_runs(cand, champ)
    assert p.task_ids == ("c",)
    assert p.dropped_tasks == ("a", "b")
    assert p.n_dropped == 2
    assert len(p.candidate_runs) == len(p.champion_runs) == 1


def test_retry_replaces_failed_first_attempt(mk: MakeRun) -> None:
    cand = [mk("a", None, infra=True, attempt=0), mk("a", 1.0, attempt=1)]
    champ = [mk("a", 0.0)]
    p = pair_runs(cand, champ)
    assert p.deltas == (1.0,)
    assert p.n_dropped == 0
    # a retry that fails again drops the task
    cand2 = [mk("a", 1.0, attempt=0), mk("a", None, refusal=True, attempt=1)]
    assert pair_runs(cand2, champ).dropped_tasks == ("a",)


def test_task_missing_from_one_arm_is_dropped(mk: MakeRun) -> None:
    p = pair_runs([mk("a", 1.0), mk("b", 1.0)], [mk("a", 0.0)])
    assert p.task_ids == ("a",)
    assert p.dropped_tasks == ("b",)


def test_rejects_mixed_batches_splits_and_duplicates(mk: MakeRun) -> None:
    with pytest.raises(ValueError, match="one batch"):
        pair_runs([mk("a", 1.0, batch=1)], [mk("a", 0.0, batch=2)])
    with pytest.raises(ValueError, match="select"):
        pair_runs([mk("a", 1.0, split=Split.TRAIN)], [mk("a", 0.0)], split=Split.SELECT)
    with pytest.raises(ValueError, match="duplicate"):
        pair_runs([mk("a", 1.0), mk("a", 1.0)], [mk("a", 0.0)])


def test_bootstrap_hand_cases() -> None:
    assert paired_bootstrap([], seed=0).p == 1.0
    assert paired_bootstrap([0.3], seed=0).p == 1.0
    # all-zero and all-negative gains can never be significant
    assert paired_bootstrap([0.0] * 10, seed=0).p == 1.0
    assert paired_bootstrap([-0.1] * 10, seed=0).p == 1.0
    # constant positive gain: p floors at 2**-n (n=3 -> 0.125), above 1/(B+1)
    r = paired_bootstrap([0.1, 0.1, 0.1], seed=0, n_resamples=100)
    assert r.p == 0.125
    assert r.delta == pytest.approx(0.1)
    # mean of [1, 0, 1, 0, 1, 1] is 2/3, a clear win
    r = paired_bootstrap([1, 0, 1, 0, 1, 1] * 5, seed=1)
    assert r.delta == pytest.approx(2 / 3)
    assert r.p < 0.001
    # mostly zeros with a 10-task +1 win among 100: significant; among symmetric noise: not
    assert paired_bootstrap([1.0] * 10 + [0.0] * 90, seed=2).p < 0.01
    assert paired_bootstrap([1.0] * 5 + [-1.0] * 5 + [0.0] * 90, seed=2).p > 0.3


def test_bootstrap_is_deterministic_under_seed() -> None:
    d = [0.5, -0.5, 0.0, 1.0, 0.0, 0.5, -1.0, 0.0, 0.5, 0.0]
    a = paired_bootstrap(d, seed=7, n_resamples=300)
    b = paired_bootstrap(d, seed=7, n_resamples=300)
    assert a == b
    assert {paired_bootstrap(d, seed=s, n_resamples=300).p for s in range(5)} != {a.p}


def test_bootstrap_rejects_bad_resamples() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], seed=0, n_resamples=0)
