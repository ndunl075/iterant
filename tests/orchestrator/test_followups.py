"""Review follow-ups: promotion trailer, one smoke query per phase."""

from __future__ import annotations

import pytest

from iterant.archive import Archive
from iterant.orchestrator import Outcomes


def test_the_promotion_commit_records_the_batch_pack_hash(world):
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    assert res.promoted == "b1-0"
    body = world.git("log", "-1", "--format=%B", "iterant/gen-1").splitlines()
    assert f"Iterant-Pack-Hash: {world.pack.content_hash}" in body
    assert "Candidate: b1-0" in body  # resume still recognises the commit


def test_resume_reconciles_a_promotion_commit_that_has_the_pack_hash_trailer(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h-good"))
    real = Archive.record_promotion

    def die(self, promo):
        raise world.Crash("died")

    monkeypatch.setattr(Archive, "record_promotion", die)
    with pytest.raises(world.Crash):
        world.orchestrator().run_generation()
    monkeypatch.setattr(Archive, "record_promotion", real)
    assert "Iterant-Pack-Hash:" in world.git("log", "-1", "--format=%B", "iterant/gen-1")
    world.orchestrator().run_generation()
    promo = world.db().get_promotion(1)
    assert promo is not None and promo.cand == "b1-0"


def test_smoke_opens_the_outcome_store_once_for_the_whole_phase(world, monkeypatch):
    world.queue(world.proposal("v0", "v1", "h1"), world.proposal("v0", "v2", "h2"))
    calls: list[str] = []
    real = Outcomes.recorded_by_cand

    def spy(self, batch):
        calls.append("by_cand")
        return real(self, batch)

    monkeypatch.setattr(Outcomes, "recorded_by_cand", spy)
    monkeypatch.setattr(Outcomes, "recorded", lambda self, b, c: calls.append("per_cand") or set())
    world.orchestrator().run_generation()
    assert calls == ["by_cand"]  # one query for both candidates, none per task or candidate
