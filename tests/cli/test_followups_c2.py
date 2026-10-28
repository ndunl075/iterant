"""W3 follow-ups C, review round: SchemaVersionError message, last_run.json, status refusals."""

from __future__ import annotations

import sqlite3

from iterant.archive import archive_path


def _run_once(world) -> None:  # type: ignore[no-untyped-def]
    world.init()
    world.queue.append(None)
    world.cli("run", str(world.repo), *world.RUN_SMALL)


def test_a_newer_archive_through_run_gets_the_upgrade_message(world, capsys) -> None:  # type: ignore[no-untyped-def]
    _run_once(world)
    capsys.readouterr()
    con = sqlite3.connect(archive_path(world.repo))
    con.execute("PRAGMA user_version = 99")
    con.commit()
    con.close()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) != 0
    err = capsys.readouterr().err
    assert "newer Iterant (schema 99)" in err and "upgrade Iterant" in err
    assert "SchemaVersionError" not in err


def test_last_run_json_is_written_atomically_after_a_started_run(world) -> None:  # type: ignore[no-untyped-def]
    _run_once(world)
    d = world.repo / ".iterant"
    assert (d / "last_run.json").is_file()
    assert not list(d.glob("last_run.json.tmp"))


def test_last_run_json_is_not_written_when_preflight_fails(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from iterant.grader import Grader, GraderPreflightError

    def boom(self) -> None:  # type: ignore[no-untyped-def]
        raise GraderPreflightError("broken")

    monkeypatch.setattr(Grader, "preflight", boom)
    world.init()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) != 0
    assert not (world.repo / ".iterant" / "last_run.json").exists()


def test_status_shows_the_champion_refusal_rate(world, capsys) -> None:  # type: ignore[no-untyped-def]
    from iterant.contracts import Proposal

    world.init()
    world.calibrate()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h-good"))
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    capsys.readouterr()
    assert world.cli("status", str(world.repo)) == 0
    assert "champion refusals (latest batch, first attempts, select and holdout): 0 of" in (
        capsys.readouterr().out
    )


def test_status_refusals_use_the_arm_the_latest_batch_ran_against_after_a_promotion(  # type: ignore[no-untyped-def]
    world,
) -> None:
    from iterant.archive import Archive
    from iterant.cli.status import batch_champion_id
    from iterant.contracts import Proposal

    world.init()
    world.calibrate()
    world.queue.append(Proposal(world.diff("v0", "v1"), "h-good"))
    assert world.cli("run", str(world.repo), *world.RUN_NOSKIP) == 0
    with Archive.open_path(archive_path(world.repo), create_schema=False) as archive:
        promo = archive.latest_promotion()
        batch = archive.get_batch(1)
        assert promo is not None and batch is not None and promo.gen == batch.gen
        assert promo.cand != "seed"
        # The promoted winner is the current champion, but the batch's champion arm is the seed.
        assert batch_champion_id(archive, batch, promo) == "seed"
        nxt = archive.get_batch(2)
        assert nxt is None
        # A later batch (gen above the promotion) is judged against the current champion.
        later = type(batch)(2, promo.gen + 1, batch.pack_hash, batch.started_at)
        assert batch_champion_id(archive, later, promo) == promo.cand


def test_a_negative_user_version_does_not_claim_a_newer_iterant(world, capsys) -> None:  # type: ignore[no-untyped-def]
    _run_once(world)
    capsys.readouterr()
    con = sqlite3.connect(archive_path(world.repo))
    con.execute("PRAGMA user_version = -3")
    con.commit()
    con.close()
    world.queue.append(None)
    assert world.cli("run", str(world.repo), *world.RUN_SMALL) != 0
    assert "newer Iterant" not in capsys.readouterr().err
