import subprocess
from pathlib import Path

import pytest

from iterant.archive import (
    GitRefError,
    champion_sha,
    diff_sha256,
    ensure_seed,
    find_promoted_commit,
    gen_tag_sha,
    promote,
)
from iterant.contracts import Candidate, Stage, Status


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout
    return out.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "target"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "User")
    git(r, "config", "user.email", "u@example.com")
    (r / "prompt.txt").write_text("hello\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


def make_diff(repo: Path, new_text: str) -> bytes:
    (repo / "prompt.txt").write_text(new_text)
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary"], capture_output=True, check=True
    ).stdout
    git(repo, "checkout", "-q", "--", "prompt.txt")
    return diff


def candidate(diff: bytes, cid: str = "c1") -> Candidate:
    return Candidate(cid, None, diff_sha256(diff), "be nicer", Stage.CONFIRM, Status.ACTIVE, 0)


def test_seed_and_promote_leave_user_state_untouched(repo: Path) -> None:
    git(repo, "branch", "other")
    (repo / "wip.txt").write_text("uncommitted\n")
    git(repo, "add", "wip.txt")
    (repo / "prompt.txt").write_text("hello\nlocal edit\n")
    before = (
        git(repo, "rev-parse", "main", "other"),
        git(repo, "status", "--porcelain"),
        git(repo, "diff", "--cached"),
        (repo / "prompt.txt").read_text(),
    )
    main_sha = git(repo, "rev-parse", "main")

    seed = ensure_seed(repo)
    assert seed == main_sha == champion_sha(repo) == gen_tag_sha(repo, 0)
    assert ensure_seed(repo) == seed  # idempotent

    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD", "--", "prompt.txt"],
        capture_output=True,
        check=True,
    ).stdout
    # the user's local edit is in the working tree; build the candidate diff separately
    patch = b"--- a/prompt.txt\n+++ b/prompt.txt\n@@ -1 +1 @@\n-hello\n+hello world\n"
    patch = b"diff --git a/prompt.txt b/prompt.txt\n" + patch
    assert diff  # sanity: working tree really is dirty

    new = promote(repo, candidate(patch), patch, 1, seed)
    assert champion_sha(repo) == new == gen_tag_sha(repo, 1)
    assert git(repo, "rev-parse", f"{new}^") == seed
    assert git(repo, "show", f"{new}:prompt.txt") == "hello world"
    assert "Candidate: c1" in git(repo, "log", "-1", "--format=%B", new)

    after = (
        git(repo, "rev-parse", "main", "other"),
        git(repo, "status", "--porcelain"),
        git(repo, "diff", "--cached"),
        (repo / "prompt.txt").read_text(),
    )
    assert after == before
    assert git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_second_generation_builds_on_first(repo: Path) -> None:
    ensure_seed(repo)
    seed = ensure_seed(repo)
    d1 = make_diff(repo, "one\n")
    first = promote(repo, candidate(d1, "c1"), d1, 1, seed)
    # next diff must apply on top of the new champion, not on the user's HEAD
    d2 = (
        b"diff --git a/prompt.txt b/prompt.txt\n--- a/prompt.txt\n+++ b/prompt.txt\n"
        b"@@ -1 +1 @@\n-one\n+two\n"
    )
    second = promote(repo, candidate(d2, "c2"), d2, 2, first)
    assert git(repo, "rev-parse", f"{second}^") == first
    assert git(repo, "show", f"{second}:prompt.txt") == "two"
    assert git(repo, "show", "main:prompt.txt") == "hello"


def test_refusals_leave_refs_unchanged(repo: Path) -> None:
    seed = ensure_seed(repo)
    d1 = make_diff(repo, "one\n")
    good = candidate(d1)
    with pytest.raises(GitRefError, match="diff_sha"):
        promote(repo, candidate(b"other"), d1, 1, seed)
    bad = b"diff --git a/nope b/nope\n--- a/nope\n+++ b/nope\n@@ -1 +1 @@\n-x\n+y\n"
    with pytest.raises(GitRefError, match="does not apply"):
        promote(repo, candidate(bad), bad, 1, seed)
    with pytest.raises(ValueError):
        promote(repo, good, d1, 0, seed)
    assert champion_sha(repo) == seed
    assert gen_tag_sha(repo, 1) is None

    with pytest.raises(GitRefError, match="evaluated on"):
        promote(repo, good, d1, 1, "0" * 40)
    promote(repo, good, d1, 1, seed)
    other = candidate(d1, "someone-else")  # same diff, different candidate
    with pytest.raises(GitRefError, match="not made for candidate"):
        promote(repo, other, d1, 1, seed)


def test_promote_needs_seed_and_refuses_checked_out_champion(repo: Path) -> None:
    d = make_diff(repo, "x\n")
    with pytest.raises(GitRefError, match="ensure_seed"):
        promote(repo, candidate(d), d, 1, "0" * 40)
    seed = ensure_seed(repo)
    git(repo, "checkout", "-q", "iterant/champion")
    with pytest.raises(GitRefError, match="checked out"):
        promote(repo, candidate(d), d, 1, seed)
    assert champion_sha(repo) == seed


def test_ensure_seed_bad_base(repo: Path) -> None:
    with pytest.raises(GitRefError):
        ensure_seed(repo, "no-such-rev")


def test_promote_is_idempotent_after_lost_db_write(repo: Path, tmp_path: Path) -> None:
    from iterant.archive import Archive
    from iterant.contracts import Promotion

    seed = ensure_seed(repo)
    d = make_diff(repo, "one\n")
    cand = candidate(d)
    with Archive.open(repo) as a:
        a.insert_candidate(cand)
        first = promote(repo, cand, d, 1, seed)  # ...crash before record_promotion...
        assert find_promoted_commit(repo, cand, 1) == first
        got = a.get_candidate(cand.id)
        assert got is not None and got.status is Status.ACTIVE

        # resume: promote again returns the same commit, no new commit or error
        assert promote(repo, cand, d, 1, seed) == first
        assert git(repo, "rev-parse", f"{first}^") == seed
        assert champion_sha(repo) == first

        promo = Promotion(1, cand.id, 0.1, 0.01, 0.05, 0.1)
        a.record_promotion(promo)
        a.record_promotion(promo)  # replay is a no-op
        got = a.get_candidate(cand.id)
        assert got is not None
        assert (got.status, got.stage) == (Status.PROMOTED, Stage.PROMOTE)
        assert a.get_promotion(1) == promo


def test_record_promotion_is_atomic(repo: Path) -> None:
    from iterant.archive import Archive, ForeignKeyViolation
    from iterant.contracts import Promotion

    with Archive.open(repo) as a:
        with pytest.raises(ForeignKeyViolation):
            a.record_promotion(Promotion(1, "ghost", 0.1, 0.01, 0.05, 0.1))
        assert a.get_promotion(1) is None


def test_inherited_git_env_does_not_redirect(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    seed = ensure_seed(repo)
    monkeypatch.delenv("GIT_DIR")
    assert champion_sha(repo) == seed
    assert git(other, "branch", "--list", "iterant/*") == ""


def test_user_hooks_do_not_run(repo: Path) -> None:
    hook = repo / ".git" / "hooks" / "reference-transaction"
    marker = repo / ".git" / "hook-ran"
    hook.write_text(f"#!/bin/sh\ntouch '{marker.as_posix()}'\n")
    hook.chmod(0o755)
    ensure_seed(repo)
    assert not marker.exists()
