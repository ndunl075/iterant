"""Workspace write: OS errors become ToolError; junctions are refused like symlinks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from iterant.proposer.workspace import ToolError, Workspace
from iterant.target.genome import Genome


def ws(root: Path) -> Workspace:
    return Workspace(root, Genome(root, ["prompts/**"]), hidden=[])


def test_a_file_where_a_directory_is_needed_is_a_tool_error(repo: Path) -> None:
    w = ws(repo)
    (repo / "prompts" / "a").write_text("x\n", encoding="utf-8")
    with pytest.raises(ToolError, match="cannot write"):
        w.write("prompts/a/b.md", "y\n")
    assert w.touched == ()
    assert w.diff() == ""  # the failed write left no phantom touched file


def test_a_name_that_is_too_long_is_a_tool_error(repo: Path) -> None:
    w = ws(repo)
    with pytest.raises(ToolError):
        w.write("prompts/" + "n" * 5000, "y\n")
    assert w.diff() == ""


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows reparse points")
def test_write_through_a_junction_is_refused(repo: Path) -> None:
    outside = repo / "prompts" / "real"  # inside the genome: only the junction check can refuse
    outside.mkdir()
    link = repo / "prompts" / "j"
    done = subprocess.run(  # noqa: S603
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        pytest.skip("cannot create junctions here")
    try:
        with pytest.raises(ToolError, match="junction"):
            ws(repo).write("prompts/j/x.md", "y\n")
        assert not (outside / "x.md").exists()
    finally:
        link.rmdir()  # removes the link, not the target's contents
