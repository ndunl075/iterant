"""Workspace writes are atomic: a failure part-way leaves the original and no partial file."""

from __future__ import annotations

import os
import shutil
import sys
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
    w.read("prompts/system.md")
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
    w.read("prompts/system.md")
    with pytest.raises(ToolError):
        w.write("prompts/system.md", "new\n")
    assert target.read_bytes() == before
    assert w.touched == ()
    assert sorted(p.name for p in target.parent.iterdir()) == ["system.md"]


def test_a_successful_write_replaces_the_file_and_leaves_no_temp(repo: Path) -> None:
    w = ws(repo)
    w.read("prompts/system.md")
    w.write("prompts/system.md", "new\n")
    assert (repo / "prompts" / "system.md").read_text() == "new\n"
    assert sorted(p.name for p in (repo / "prompts").iterdir()) == ["system.md"]
    assert "+new" in w.diff()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec bits")
def test_an_executable_file_stays_executable_and_the_diff_has_no_mode_line(repo: Path) -> None:
    target = repo / "prompts" / "run.sh"
    target.write_text("#!/bin/sh\necho 1\n")
    target.chmod(0o755)
    w = ws(repo)
    w.read("prompts/run.sh")
    w.write("prompts/run.sh", "#!/bin/sh\necho 2\n")
    assert target.stat().st_mode & 0o777 == 0o755
    assert "+echo 2" in w.diff()
    assert "old mode" not in w.diff() and "new mode" not in w.diff()


def test_the_original_mode_is_copied_onto_the_temp_file(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    real = shutil.copymode

    def spy(src: str, dst: str) -> None:
        calls.append((Path(src).name, Path(dst).name))
        real(src, dst)

    monkeypatch.setattr(shutil, "copymode", spy)
    w = ws(repo)
    w.read("prompts/system.md")
    w.write("prompts/system.md", "new\n")
    assert len(calls) == 1 and calls[0][0] == "system.md" and calls[0][1] != "system.md"


def test_a_copymode_failure_is_a_tool_error_and_leaves_the_original(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = repo / "prompts" / "system.md"
    before = target.read_bytes()

    def boom(a: object, b: object) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(shutil, "copymode", boom)
    w = ws(repo)
    w.read("prompts/system.md")
    with pytest.raises(ToolError, match="cannot write"):
        w.write("prompts/system.md", "new\n")
    assert target.read_bytes() == before and w.touched == ()
    assert sorted(p.name for p in target.parent.iterdir()) == ["system.md"]
