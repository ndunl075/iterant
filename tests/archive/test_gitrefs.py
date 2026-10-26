import subprocess
from pathlib import Path

import pytest

from iterant.archive import (
    GitRefError,
    champion_sha,
    diff_sha256,
    ensure_seed,
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

    new = promote(repo, candidate(patch), patch, 1)
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
    d1 = make_diff(repo, "one\n")
    first = promote(repo, candidate(d1, "c1"), d1, 1)
    # next diff must apply on top of the new champion, not on the user's HEAD
    d2 = (
        b"diff --git a/prompt.txt b/prompt.txt\n--- a/prompt.txt\n+++ b/prompt.txt\n"
        b"@@ -1 +1 @@\n-one\n+two\n"
    )
    second = promote(repo, candidate(d2, "c2"), d2, 2)
    assert git(repo, "rev-parse", f"{second}^") == first
    assert git(repo, "show", f"{second}:prompt.txt") == "two"
    assert git(repo, "show", "main:prompt.txt") == "hello"


def test_refusals_leave_refs_unchanged(repo: Path) -> None:
    seed = ensure_seed(repo)
    d1 = make_diff(repo, "one\n")
    good = candidate(d1)
    with pytest.raises(GitRefError, match="diff_sha"):
        promote(repo, candidate(b"other"), d1, 1)
    bad = b"diff --git a/nope b/nope\n--- a/nope\n+++ b/nope\n@@ -1 +1 @@\n-x\n+y\n"
    with pytest.raises(GitRefError, match="does not apply"):
        promote(repo, candidate(bad), bad, 1)
    with pytest.raises(ValueError):
        promote(repo, good, d1, 0)
    assert champion_sha(repo) == seed
    assert gen_tag_sha(repo, 1) is None

    promote(repo, good, d1, 1)
    with pytest.raises(GitRefError, match="already exists"):
        promote(repo, good, d1, 1)


def test_promote_needs_seed_and_refuses_checked_out_champion(repo: Path) -> None:
    d = make_diff(repo, "x\n")
    with pytest.raises(GitRefError, match="ensure_seed"):
        promote(repo, candidate(d), d, 1)
    seed = ensure_seed(repo)
    git(repo, "checkout", "-q", "iterant/champion")
    with pytest.raises(GitRefError, match="checked out"):
        promote(repo, candidate(d), d, 1)
    assert champion_sha(repo) == seed


def test_ensure_seed_bad_base(repo: Path) -> None:
    with pytest.raises(GitRefError):
        ensure_seed(repo, "no-such-rev")
