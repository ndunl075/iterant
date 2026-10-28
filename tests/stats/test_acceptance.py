from __future__ import annotations

from collections.abc import Callable

from iterant.contracts import AcceptanceThresholds, RunResult, Split
from iterant.stats import Reason, accepted, decide_confirm, decide_select
from iterant.stats.acceptance import _binom_sf, noise_veto  # pyright: ignore[reportPrivateUsage]

MakeRun = Callable[..., RunResult]


def arms(
    mk: MakeRun,
    gains: list[float],
    split: Split = Split.SELECT,
    **cand_kw: object,
) -> tuple[list[RunResult], list[RunResult]]:
    """Champion scores 0.5 on each task; candidate scores 0.5 + gain."""
    cand = [mk(f"t{i}", 0.5 + g, split=split, **cand_kw) for i, g in enumerate(gains)]
    champ = [mk(f"t{i}", 0.5, cand="h", split=split) for i in range(len(gains))]
    return cand, champ


def test_select_accepts_clear_win(mk: MakeRun) -> None:
    cand, champ = arms(mk, [0.2, 0.1] * 20 + [0.0] * 20)  # mean 0.1
    d = decide_select(cand, champ, seed=0)
    assert d.accepted and d.reasons == () and d.detail == ()
    assert d.stage is Split.SELECT
    assert round(d.delta, 6) == 0.1 and d.p < 0.05 and d.n_tasks == 60


def test_select_delta_boundary_is_inclusive(mk: MakeRun) -> None:
    # every task gains exactly 0.05 -> delta == delta_min passes (>=)
    cand, champ = arms(mk, [0.05] * 5 + [0.04, 0.06] * 10)
    assert decide_select(cand, champ, seed=0).accepted
    cand, champ = arms(mk, [0.04, 0.05] * 15)
    d = decide_select(cand, champ, seed=0)
    assert not d.accepted and Reason.DELTA_BELOW_MIN in d.reasons


def test_select_rejects_noise_even_with_big_mean(mk: MakeRun) -> None:
    # mean +0.08 from a few lucky tasks among many losers: delta clears, p does not
    cand, champ = arms(mk, [0.5, 0.5, -0.4, 0.3, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = decide_select(cand, champ, seed=0)
    assert d.delta >= 0.05
    assert not d.accepted and d.reasons == (Reason.P_TOO_HIGH,)


def test_p_threshold_is_strict(mk: MakeRun) -> None:
    cand, champ = arms(mk, [0.1] * 4)  # constant gain over 4 tasks: p = 2**-4 = 0.0625
    t = AcceptanceThresholds(select_p=0.0625)
    assert Reason.P_TOO_HIGH in decide_select(cand, champ, t, seed=0).reasons  # not <
    t = AcceptanceThresholds(select_p=0.07)
    assert decide_select(cand, champ, t, seed=0).accepted


def test_confirm_needs_positive_delta_and_p_below_holdout_p(mk: MakeRun) -> None:
    strict = AcceptanceThresholds(holdout_p=0.2)
    cand, champ = arms(mk, [0.1, 0.0, 0.1, 0.0] * 10, Split.HOLDOUT)
    d = decide_confirm(cand, champ, strict, seed=0)
    assert d.accepted and d.stage is Split.HOLDOUT
    cand, champ = arms(mk, [0.0] * 20, Split.HOLDOUT)
    d = decide_confirm(cand, champ, strict, seed=0)
    assert not d.accepted and Reason.NOT_POSITIVE in d.reasons
    cand, champ = arms(mk, [-0.1, 0.1] * 10 + [0.01], Split.HOLDOUT)
    assert Reason.P_TOO_HIGH in decide_confirm(cand, champ, strict, seed=0).reasons


def test_wrong_split_is_an_error_not_a_verdict(mk: MakeRun) -> None:
    import pytest

    cand, champ = arms(mk, [0.1] * 10, Split.HOLDOUT)
    with pytest.raises(ValueError):
        decide_select(cand, champ, seed=0)


def test_cost_guard_boundary(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand, champ = arms(mk, gains, usd=1150)  # +15% exactly against usd=1000 default
    assert decide_select(cand, champ, seed=0).accepted
    cand, champ = arms(mk, gains, usd=1151)
    d = decide_select(cand, champ, seed=0)
    assert d.reasons == (Reason.COST_GUARD,) and len(d.detail) == 1


def test_wall_guard_boundary(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand, champ = arms(mk, gains, wall=12.5)  # +25% exactly
    assert decide_select(cand, champ, seed=0).accepted
    cand, champ = arms(mk, gains, wall=12.6)
    assert decide_select(cand, champ, seed=0).reasons == (Reason.WALL_GUARD,)


def test_zero_cost_champion_allows_only_zero_cost(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand = [mk(f"t{i}", 0.5 + g, usd=1) for i, g in enumerate(gains)]
    champ = [mk(f"t{i}", 0.5, cand="h", usd=0) for i in range(len(gains))]
    assert Reason.COST_GUARD in decide_select(cand, champ, seed=0).reasons


def _first_try_failures(
    mk: MakeRun, runs: list[RunResult], idx: range, *, cand: str, refusal: bool
) -> None:
    """Make the first try of tasks ``idx`` refuse / infra-error, with a clean retry (same score)."""
    for i in idx:
        old = runs[i]
        runs[i] = mk(old.key.task, None, cand=cand, refusal=refusal, infra=not refusal)
        runs.append(mk(old.key.task, old.score, cand=cand, attempt=1))


def test_refusal_guard_vetoes_a_clear_first_try_excess_even_when_retries_succeed(
    mk: MakeRun,
) -> None:
    # 6 of 40 first tries refused (15%) vs none: p = 0.5**6 = 0.016, excess 15 pts. The
    # retries all succeed, so the *final* rates are equal: only first attempts show it.
    cand, champ = arms(mk, [0.2, 0.1] * 20)
    _first_try_failures(mk, cand, range(6), cand="c", refusal=True)
    d = decide_select(cand, champ, seed=0)
    assert d.reasons == (Reason.REFUSAL_GUARD,) and d.n_dropped == 0
    assert "first-try refusal rate 15.0%" in d.detail[0]


def test_infra_guard_vetoes_a_clear_first_try_excess(mk: MakeRun) -> None:
    cand, champ = arms(mk, [0.2, 0.1] * 20)
    _first_try_failures(mk, cand, range(6), cand="c", refusal=False)
    assert decide_select(cand, champ, seed=0).reasons == (Reason.INFRA_GUARD,)


def test_one_extra_refusal_is_noise_not_a_veto(mk: MakeRun) -> None:
    # 1 of 41 vs 0 of 41 first tries: excess 2.4 pts, but p ~ 0.5: not significant
    cand, champ = arms(mk, [0.2, 0.1] * 20)
    cand.append(mk("x", None, refusal=True))
    champ.append(mk("x", 0.5, cand="h"))
    d = decide_select(cand, champ, seed=0)
    assert d.n_dropped == 0 and Reason.REFUSAL_GUARD not in d.reasons and d.accepted


def test_a_refused_task_scores_zero_it_is_not_dropped(mk: MakeRun) -> None:
    """4 refusals the retry did not clear: each is a 0.0 against the champion's 0.5, so the
    candidate pays for them in delta instead of shedding the tasks."""
    cand, champ = arms(mk, [0.0] * 40)
    for i in range(4):
        cand[i] = mk(f"t{i}", None, refusal=True)
    d = decide_select(cand, champ, seed=0)
    assert d.n_dropped == 0 and d.n_tasks == 40
    assert round(d.delta, 6) == round(-4 * 0.5 / 40, 6)


def test_equal_first_try_rates_are_fine(mk: MakeRun) -> None:
    cand, champ = arms(mk, [0.2, 0.1] * 20)
    _first_try_failures(mk, cand, range(6), cand="c", refusal=True)
    _first_try_failures(mk, champ, range(6), cand="h", refusal=True)
    assert decide_select(cand, champ, seed=0).accepted


def test_noise_veto_needs_both_significance_and_margin() -> None:
    t = AcceptanceThresholds()
    # significant but only 2.0 pts worse (70 vs 50 of 1000): the margin is "more than 2 pts"
    veto, p, excess = noise_veto(70, 1000, 50, 1000, t)
    assert p < 0.05 and abs(excess - 0.02) < 1e-12 and not veto
    assert noise_veto(71, 1000, 50, 1000, t)[0]
    # large excess but not significant (2 of 10 vs 0 of 10): p = 0.25
    veto, p, excess = noise_veto(2, 10, 0, 10, t)
    assert excess == 0.2 and p == 0.25 and not veto
    # the fields are thresholds: a stricter margin / p turns the veto off, a looser one on
    assert not noise_veto(71, 1000, 50, 1000, AcceptanceThresholds(noise_guard_margin=0.05))[0]
    assert not noise_veto(6, 40, 0, 40, AcceptanceThresholds(noise_guard_p=0.01))[0]
    assert noise_veto(6, 40, 0, 40, AcceptanceThresholds(noise_guard_p=0.02))[0]
    # a better-than-champion or empty arm never vetoes
    assert not noise_veto(0, 40, 9, 40, t)[0] and not noise_veto(0, 0, 0, 40, t)[0]


def test_binomial_tail_is_exact() -> None:
    assert abs(_binom_sf(6, 6, 0.5) - 0.5**6) < 1e-15
    assert abs(_binom_sf(1, 3, 0.5) - 0.875) < 1e-15
    assert _binom_sf(0, 5, 0.3) == 1.0 and _binom_sf(6, 5, 0.3) == 0.0


def test_all_reasons_reported_and_empty_pairing(mk: MakeRun) -> None:
    cand, champ = arms(mk, [-0.1] * 20, usd=5000, wall=100.0)
    d = decide_select(cand, champ, seed=0)
    assert set(d.reasons) == {
        Reason.DELTA_BELOW_MIN,
        Reason.P_TOO_HIGH,
        Reason.COST_GUARD,
        Reason.WALL_GUARD,
    }
    assert len(d.detail) == len(d.reasons)
    d = decide_select([], [], seed=0)
    assert not d.accepted and d.reasons == (Reason.NO_PAIRED_TASKS,)


def test_accepted_requires_both_stages(mk: MakeRun) -> None:
    s_c, s_h = arms(mk, [0.2, 0.1] * 20)
    h_c, h_h = arms(mk, [0.1, 0.0, 0.1, 0.0] * 10, Split.HOLDOUT)
    sel, conf = decide_select(s_c, s_h, seed=0), decide_confirm(h_c, h_h, seed=0)
    assert accepted(sel, conf)
    bad_c, bad_h = arms(mk, [0.0] * 20, Split.HOLDOUT)
    assert not accepted(sel, decide_confirm(bad_c, bad_h, seed=0))
    assert not accepted(conf, sel)  # stages in the wrong slots


def test_cost_guard_counts_retried_attempts(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand, champ = arms(mk, gains)
    # a refused first try that cost 5000 on one task, then a clean retry
    cand[0] = mk("t0", None, refusal=True, usd=5000, attempt=0)
    cand.append(mk("t0", 0.7, attempt=1, usd=1000))
    d = decide_select(cand, champ, seed=0)
    # (39*1000 + 1000 + 5000) / 40 finals = 1125 <= 1150: still fine
    assert d.accepted
    cand.append(mk("t1", None, refusal=True, usd=20000, attempt=0))
    cand[1] = mk("t1", 0.6, attempt=1, usd=1000)
    assert Reason.COST_GUARD in decide_select(cand, champ, seed=0).reasons


def test_too_many_dropped_tasks_rejects(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand, champ = arms(mk, gains)
    for i in range(5):  # 5 of 45 tasks (11%) infra-errored in the candidate
        cand.append(mk(f"x{i}", None, infra=True))
        champ.append(mk(f"x{i}", 0.5, cand="h"))
    d = decide_select(cand, champ, seed=0)
    assert Reason.TOO_MANY_DROPPED in d.reasons and d.n_dropped == 5
    # 4 of 44 (9.1%) passes this check
    d = decide_select(cand[:-1], champ[:-1], seed=0)
    assert Reason.TOO_MANY_DROPPED not in d.reasons


def test_missing_task_and_identical_arms_raise(mk: MakeRun) -> None:
    import pytest

    cand, champ = arms(mk, [0.1] * 30)
    with pytest.raises(ValueError):
        decide_select(cand[:10], champ, seed=0)
    with pytest.raises(ValueError):
        decide_select(cand, cand, seed=0)


def test_missing_rep_in_one_arm_raises(mk: MakeRun) -> None:
    import pytest

    champ = [mk(f"t{i}", 0.0, cand="h", rep=r) for i in range(80) for r in (0, 1)]
    cand = [mk(f"t{i}", 1.0 if i < 10 else 0.0, rep=0) for i in range(80)]
    with pytest.raises(ValueError, match=r"\(task, rep\).*and 70 more"):
        decide_select(cand, champ, seed=0)
    many = [mk(f"u{i}", 1.0) for i in range(15)]
    with pytest.raises(ValueError, match="and 5 more"):
        decide_select(cand + many, champ, seed=0)


def _refusing(mk: MakeRun, n: int, refused: int, cand: str = "x") -> list[RunResult]:
    return [
        mk(f"t{i}", None if i < refused else 0.5, cand=cand, refusal=i < refused) for i in range(n)
    ]


def test_refusal_creep_is_caught_against_the_seed_but_not_step_by_step(mk: MakeRun) -> None:
    """Each generation refuses 2 points more than its champion (within the noise margin),
    so the champion comparison never vetoes; the seed anchor does, at gen 3."""
    n = 200
    seed_runs = _refusing(mk, n, 0, "seed")
    champ = seed_runs
    verdicts: list[bool] = []
    for gen in (1, 2, 3, 4):
        cand = _refusing(mk, n, 4 * gen, f"g{gen}")  # 2% per generation (n=200)
        base = decide_select(cand, champ, seed=0)
        assert Reason.REFUSAL_GUARD not in base.reasons  # step-wise: tolerated
        anchored = decide_select(cand, champ, seed=0, anchor=seed_runs)
        verdicts.append(Reason.REFUSAL_GUARD in anchored.reasons)
        champ = cand
    assert verdicts == [False, True, True, True]  # 2% tolerated, 4%+ is not (margin 2 pts)
    assert "seed champion" in anchored.detail[-1]
