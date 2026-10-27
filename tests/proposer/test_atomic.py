"""Workspace writes are atomic: a failure part-way leaves the original and no partial file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iterant.proposer.workspace import ToolError, Workspace
from iterant.target.genome import Genome


def ws(root: Path) -> Workspace:
    return Workspace(root, Genome(root, ["prompts/**"]), hidden=[])


def test_a_write_that_fails_part_way_leaves_the_original_intact(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = repo / "prompts" / "system.md"
    before = target.read_bytes()
    real = Path.write_bytes

    def half(self: Path, data: bytes) -> int:
        real(self, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", half)
    w = ws(repo)
    with pytest.raises(ToolError, match="cannot write"):
        w.write("prompts/system.md", "x" * 100)
    assert target.read_bytes() == before
    assert w.touched == () and w.diff() == ""
    assert sorted(p.name for p in target.parent.iterdir()) == ["system.md"]  # no temp left


def test_a_failed_replace_leaves_the_original_and_no_temp_file(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = repo / "prompts" / "system.md"
    before = target.read_bytes()

    def boom(src: object, dst: object) -> None:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", boom)
    w = ws(repo)
    with pytest.raises(ToolError):
        w.write("prompts/system.md", "new\n")
    assert target.read_bytes() == before
    assert w.touched == ()
    assert sorted(p.name for p in target.parent.iterdir()) == ["system.md"]


def test_a_successful_write_replaces_the_file_and_leaves_no_temp(repo: Path) -> None:
    w = ws(repo)
    w.write("prompts/system.md", "new\n")
    assert (repo / "prompts" / "system.md").read_text() == "new\n"
    assert sorted(p.name for p in (repo / "prompts").iterdir()) == ["system.md"]
    assert "+new" in w.diff()
