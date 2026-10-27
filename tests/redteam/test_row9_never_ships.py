"""Row 9: "Ship a bad change to production". Stopped because Iterant never merges: it only moves
``iterant/champion`` and tags ``iterant/gen-N``; the user merges after reading the evidence.

A candidate that wins (or one that tries to ride along with a win) must leave the user's branch,
working tree and remotes exactly as they were.
"""

from __future__ import annotations

import pytest

from iterant.contracts import Proposal


def test_row9_a_promotion_moves_only_the_champion_branch_and_tags(world) -> None:  # type: ignore[no-untyped-def]
    main_before = world.git("rev-parse", "main")
    head_before = world.git("symbolic-ref", "HEAD")
    refs_before = set(world.git("for-each-ref", "--format=%(refname)").split())

    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    assert res.promoted == "b1-0"

    assert world.git("rev-parse", "main") == main_before, "the user's branch moved"
    assert world.git("symbolic-ref", "HEAD") == head_before, "the user's checkout was switched"
    new_refs = set(world.git("for-each-ref", "--format=%(refname)").split()) - refs_before
    assert new_refs
    assert all(r.startswith(("refs/heads/iterant/", "refs/tags/iterant/")) for r in new_refs), (
        new_refs
    )


def test_row9_the_users_working_tree_and_index_are_untouched(world) -> None:  # type: ignore[no-untyped-def]
    before = (world.repo / "agent" / "prompt.txt").read_bytes()
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    assert (world.repo / "agent" / "prompt.txt").read_bytes() == before
    assert world.git("status", "--porcelain", "--untracked-files=no") == ""
    assert world.git("diff", "--cached", "--name-only") == ""


def test_row9_the_champion_differs_from_main_by_exactly_the_accepted_diff(world) -> None:  # type: ignore[no-untyped-def]
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    changed = world.git("diff", "--name-only", "main", "iterant/champion").split()
    assert changed == ["agent/prompt.txt"]
    assert world.git("show", "iterant/champion:agent/prompt.txt") == "v1"
    # one new commit on top of main, never a merge
    assert world.git("rev-list", "--count", "main..iterant/champion") == "1"
    assert world.git("rev-list", "--merges", "--count", "main..iterant/champion") == "0"


def test_row9_a_rejected_candidate_leaves_no_trace_on_the_champion_branch(world) -> None:  # type: ignore[no-untyped-def]
    world.solver = lambda arm, t, n: "wrong"
    world.queue(world.proposal("v0", "v1", "h"))
    res = world.orchestrator().run_generation()
    assert res.promoted is None
    assert world.git("rev-parse", "iterant/champion") == world.git("rev-parse", "main")
    assert world.git("tag", "-l", "iterant/gen-*").split() == ["iterant/gen-0"]


def test_row9_nothing_is_ever_pushed(world) -> None:  # type: ignore[no-untyped-def]
    assert world.git("remote") == ""
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    assert world.git("remote") == ""  # no remote was added to push to


@pytest.mark.parametrize(
    "hypothesis",
    [
        "--amend",
        "-m evil",
        "$(touch /tmp/pwned)",
        "`id`",
        "x\n\nCo-authored-by: Mallory <m@evil.example>",
        "x; git checkout main && git merge iterant/champion",
        "\x1b[31mred",
    ],
)
def test_row9_a_hostile_hypothesis_cannot_steer_git(world, hypothesis: str) -> None:  # type: ignore[no-untyped-def]
    """The hypothesis is candidate-written text that ends up in a commit message."""
    main_before = world.git("rev-parse", "main")
    diff = world.proposal("v0", "v1", "h").diff
    if "\n" in hypothesis:  # multi-line text cannot even be built into a Proposal
        with pytest.raises(ValueError):
            Proposal(diff, hypothesis)
        return
    world.queue(Proposal(diff, hypothesis))
    res = world.orchestrator().run_generation()
    assert res.promoted == "b1-0"
    assert world.git("rev-parse", "main") == main_before
    assert world.git("rev-list", "--count", "main..iterant/champion") == "1"
    assert world.git("symbolic-ref", "HEAD").endswith("main")
