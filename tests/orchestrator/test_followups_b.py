"""Second round of review follow-ups: TTL window, listing digits, promote message hygiene,
and one fault query per smoke phase."""

from __future__ import annotations

from pathlib import Path

import pytest

from iterant.archive import GitRefError, diff_sha256, promote
from iterant.contracts import Candidate, RunKey, Stage, Status
from iterant.orchestrator import OrchestratorConfig, Outcomes
from iterant.orchestrator.runs import sandbox_ttl


def task(world, tid: str = "se00"):
    return next(t for t in world.pack.tasks if t.id == tid)


TTL = sandbox_ttl(OrchestratorConfig())


def _run_with_clock(world, elapsed: float):
    world.solver = lambda arm, t, n: "ok"
    world.get_fails = True  # a healthy runtime + live sandbox scores 0.0 (candidate's doing)
    ex = world.executor()
    ticks = iter([0.0, elapsed, elapsed, elapsed])
    ex._clock = lambda: next(ticks)  # pyright: ignore[reportPrivateUsage]
    return ex.execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result


@pytest.mark.parametrize("elapsed", [TTL - 5.0, TTL - 1.0, TTL, TTL + 10.0])
def test_a_sandbox_within_the_slack_of_its_ttl_is_infra(world, elapsed):
    r = _run_with_clock(world, elapsed)
    assert (r.score, r.infra_err) == (None, True)
    assert len(world.backend.sandboxes) == 1  # never probed


@pytest.mark.parametrize("elapsed", [0.0, TTL / 2, TTL - 5.5])
def test_a_sandbox_clearly_inside_its_ttl_is_still_probed(world, elapsed):
    r = _run_with_clock(world, elapsed)
    assert (r.score, r.infra_err) == (0.0, False)
    assert len(world.backend.sandboxes) == 2


@pytest.mark.parametrize("size", ["²", "٣", "１"])
def test_a_non_ascii_digit_in_the_listing_is_a_rejected_output_not_a_crash(world, size):
    world.solver = lambda arm, t, n: "ok"
    world.listing = f"f\t1\t{size}\tanswer.txt\0"
    r = world.executor().execute(RunKey(1, "c", "se00", 0), task(world), "img:c").result
    assert (r.score, r.infra_err) == (0.0, False)


# --- promote message hygiene ------------------------------------------------------------


def _cand(hyp: str) -> Candidate:
    return Candidate("c1", None, diff_sha256(b""), hyp, Stage.CONFIRM, Status.ACTIVE, 0)


@pytest.mark.parametrize("ch", ["\r", "\n", " ", " ", "\u0085", "\x0b", "\x0c"])
def test_promote_refuses_a_hypothesis_with_a_line_break(tmp_path: Path, ch: str) -> None:
    hyp = f"nice{ch}Candidate: forged"
    with pytest.raises(GitRefError, match="line break"):
        promote(tmp_path, _cand(hyp), b"", 1, "0" * 40)


@pytest.mark.parametrize(
    "trailer",
    ["no separator", "Bad Key: v", "Key1: v", "Key: a\nCandidate: x", "Key: a b", ": v"],
)
def test_promote_refuses_a_malformed_trailer(tmp_path: Path, trailer: str) -> None:
    with pytest.raises(GitRefError, match="invalid commit trailer"):
        promote(tmp_path, _cand("ok"), b"", 1, "0" * 40, [trailer])


# --- smoke fault query ------------------------------------------------------------------


def test_smoke_reads_faults_with_one_query_per_phase(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h1"), world.proposal("v0", "v2", "h2"))
    calls: list[str] = []
    real = Outcomes.faults_by_cand

    def spy(self, batch):
        calls.append("by_cand")
        return real(self, batch)

    monkeypatch.setattr(Outcomes, "faults_by_cand", spy)
    monkeypatch.setattr(Outcomes, "faults", lambda self, b, c: calls.append("per_cand") or {})
    world.orchestrator().run_generation()
    assert calls == ["by_cand"]
