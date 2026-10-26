from __future__ import annotations

from collections.abc import Callable

from iterant.contracts import AcceptanceThresholds, RunResult, Split
from iterant.stats import Reason, accepted, decide_confirm, decide_select

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
    cand, champ = arms(mk, [0.1, 0.0, 0.1, 0.0] * 10, Split.HOLDOUT)
    d = decide_confirm(cand, champ, seed=0)
    assert d.accepted and d.stage is Split.HOLDOUT
    cand, champ = arms(mk, [0.0] * 20, Split.HOLDOUT)
    d = decide_confirm(cand, champ, seed=0)
    assert not d.accepted and Reason.NOT_POSITIVE in d.reasons
    cand, champ = arms(mk, [-0.1, 0.1] * 10 + [0.01], Split.HOLDOUT)
    assert Reason.P_TOO_HIGH in decide_confirm(cand, champ, seed=0).reasons


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


def test_refusal_and_infra_rates_no_worse(mk: MakeRun) -> None:
    gains = [0.2, 0.1] * 20
    cand, champ = arms(mk, gains)
    cand.append(mk("x", None, refusal=True))  # dropped from pairing but still counted
    champ.append(mk("x", 0.5, cand="h"))
    d = decide_select(cand, champ, seed=0)
    assert d.reasons == (Reason.REFUSAL_GUARD,) and d.n_dropped == 1

    cand, champ = arms(mk, gains)
    cand.append(mk("x", None, infra=True))
    champ.append(mk("x", 0.5, cand="h"))
    assert decide_select(cand, champ, seed=0).reasons == (Reason.INFRA_GUARD,)

    # equal rates are fine
    cand.append(mk("y", 0.5))
    champ.append(mk("y", None, cand="h", infra=True))
    champ.append(mk("z", 0.5, cand="h"))
    cand.append(mk("z", 0.5))
    # candidate: 1 infra of 43 finals, champion: 1 of 43 -> equal
    assert decide_select(cand, champ, seed=0).accepted


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
    for i in range(5):  # 5 of 45 tasks (11%) refused in the candidate
        cand.append(mk(f"x{i}", None, refusal=True))
        champ.append(mk(f"x{i}", 0.5, cand="h"))
    d = decide_select(cand, champ, seed=0)
    assert Reason.TOO_MANY_DROPPED in d.reasons and d.n_dropped == 5
    # 4 of 44 (9.1%) passes this check (the refusal guard still fires)
    d = decide_select(cand[:-1], champ[:-1], seed=0)
    assert Reason.TOO_MANY_DROPPED not in d.reasons


def test_missing_task_and_identical_arms_raise(mk: MakeRun) -> None:
    import pytest

    cand, champ = arms(mk, [0.1] * 30)
    with pytest.raises(ValueError):
        decide_select(cand[:10], champ, seed=0)
    with pytest.raises(ValueError):
        decide_select(cand, cand, seed=0)
