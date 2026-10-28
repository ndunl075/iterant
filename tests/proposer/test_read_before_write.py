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
    with pytest.raises(ToolError):
        w.read("prompts/missing.md")
    with pytest.raises(ToolError, match="before rewriting"):
        w.write("prompts/system.md", "small\n")  # the failed read did not unlock other files
    w.write("prompts/missing.md", "created\n")  # still a new file


def test_a_case_variant_spelling_counts_as_read(repo: Path) -> None:
    if (repo / "PROMPTS/SYSTEM.MD").exists() is False:
        pytest.skip("case-sensitive filesystem")
    w = Workspace(repo, Genome(repo, ["prompts/**", "PROMPTS/**"]), hidden=[])
    w.read("prompts/system.md")
    w.write("PROMPTS/SYSTEM.MD", "new\n")
    assert (repo / "prompts/system.md").read_text(encoding="utf-8") == "new\n"


def test_a_file_too_large_to_read_can_still_be_overwritten(repo: Path) -> None:
    w = ws(repo)
    (repo / "prompts/big.md").write_text("x" * (MAX_READ_BYTES + 1), encoding="utf-8")
    with pytest.raises(ToolError):
        w.read("prompts/big.md")
    w.write("prompts/big.md", "small\n")
    assert (repo / "prompts/big.md").read_text(encoding="utf-8") == "small\n"


def test_a_non_utf8_file_keeps_its_not_utf8_message(repo: Path) -> None:
    w = ws(repo)
    (repo / "prompts/bin.md").write_bytes(b"\xff\xfe binary")
    with pytest.raises(ToolError, match="not UTF-8 text"):
        w.write("prompts/bin.md", "x\n")
    assert (repo / "prompts/bin.md").read_bytes() == b"\xff\xfe binary"
    assert w.touched == ()


def test_a_second_spelling_of_a_written_file_needs_no_read(repo: Path) -> None:
    w = ws(repo)
    w.read("prompts/system.md")
    w.write("prompts/system.md", "one\n")
    w.write("./prompts/system.md", "two\n")
    assert w.touched == ("prompts/system.md",)


def test_a_dot_segment_spelling_counts_as_read_on_any_filesystem(repo: Path) -> None:
    w = ws(repo)
    w.read("prompts/./system.md")
    w.write("prompts/system.md", "new\n")
    assert (repo / "prompts/system.md").read_text(encoding="utf-8") == "new\n"


def test_hidden_and_directory_reads_do_not_count(repo: Path) -> None:
    w = Workspace(repo, Genome(repo, ["prompts/**", "evals/**"]), hidden=["evals/pack"])
    with pytest.raises(ToolError):
        w.read("evals/pack/secret.txt")
    with pytest.raises(ToolError):
        w.read("prompts")
    with pytest.raises(ToolError):
        w.write("evals/pack/secret.txt", "x\n")
    assert (repo / "evals/pack/secret.txt").read_text(encoding="utf-8") == "HIDDEN-CHECK-CONTENT\n"


def test_a_refused_write_records_nothing(repo: Path) -> None:
    w = ws(repo)
    with pytest.raises(ToolError):
        w.write("prompts/system.md", "x\n")
    assert w.touched == () and w.diff() == ""
