"""Workspace.write refuses to overwrite an existing file the proposer has not read."""

from __future__ import annotations

from pathlib import Path

import pytest

from iterant.proposer.workspace import MAX_READ_BYTES, ToolError, Workspace
from iterant.target.genome import Genome


def ws(root: Path) -> Workspace:
    return Workspace(root, Genome(root, ["prompts/**"]), hidden=[])


def test_write_to_existing_unread_file_is_refused_and_file_unchanged(repo: Path) -> None:
    w = ws(repo)
    before = (repo / "prompts/system.md").read_bytes()
    with pytest.raises(ToolError, match=r"read_file prompts/system\.md before rewriting"):
        w.write("prompts/system.md", "stub\n")
    assert (repo / "prompts/system.md").read_bytes() == before
    assert w.touched == ()


def test_read_then_write_is_allowed(repo: Path) -> None:
    w = ws(repo)
    w.read("prompts/system.md")
    w.write("prompts/system.md", "new\n")
    assert (repo / "prompts/system.md").read_text(encoding="utf-8") == "new\n"


def test_creating_a_new_file_needs_no_read(repo: Path) -> None:
    w = ws(repo)
    w.write("prompts/brand-new.md", "hi\n")
    assert (repo / "prompts/brand-new.md").exists()


def test_writing_twice_is_allowed(repo: Path) -> None:
    w = ws(repo)
    w.write("prompts/brand-new.md", "one\n")
    w.write("prompts/brand-new.md", "two\n")
    w.read("prompts/system.md")
    w.write("prompts/system.md", "a\n")
    w.write("prompts/system.md", "b\n")
    assert (repo / "prompts/system.md").read_text(encoding="utf-8") == "b\n"


def test_a_failed_read_does_not_count(repo: Path) -> None:
    w = ws(repo)
    (repo / "prompts/big.md").write_text("x" * (MAX_READ_BYTES + 1), encoding="utf-8")
    with pytest.raises(ToolError):
        w.read("prompts/big.md")
    with pytest.raises(ToolError, match="before rewriting"):
        w.write("prompts/big.md", "small\n")
    with pytest.raises(ToolError):
        w.read("prompts/missing.md")
    w.write("prompts/missing.md", "created\n")  # still a new file


def test_a_case_variant_spelling_counts_as_read(repo: Path) -> None:
    if (repo / "PROMPTS/SYSTEM.MD").exists() is False:
        pytest.skip("case-sensitive filesystem")
    w = Workspace(repo, Genome(repo, ["prompts/**", "PROMPTS/**"]), hidden=[])
    w.read("prompts/system.md")
    w.write("PROMPTS/SYSTEM.MD", "new\n")
    assert (repo / "prompts/system.md").read_text(encoding="utf-8") == "new\n"
