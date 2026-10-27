"""Row 1: "Edit the checks, the pack or Iterant". Stopped by the genome allowlist.

A malicious candidate ships diffs that try to reach outside the genome by every route the diff
format and the filesystem offer. The real ``DiffGate`` (real ``git apply`` on a scratch copy)
must reject each one, and must never write to the repo itself.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from iterant.contracts import Pack, Proposal, TargetConfig, TargetModel
from iterant.gate import DiffGate, GateReport

Check = Callable[[str], GateReport]


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def plain(old: str, new: str, body: str = "@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n") -> str:
    return f"--- a/{old}\n+++ b/{new}\n{body}"


def must_reject(result: GateReport) -> None:
    assert not result.passed, "the gate accepted a diff that escapes the genome"
    assert result.findings
    assert all(":" in f for f in result.findings)


@pytest.mark.parametrize(
    "path",
    [
        # case and spelling variants of protected files (case-insensitive filesystems)
        "Evals/Pack/tasks/beta-9/check",
        "EVALS/PACK/tasks/alpha-17/check",
        "ITERANT.TOML",
        "containerfile",
        # dot segments and doubled separators
        "./secret.py",
        "tools/./../secret.py",
        "tools//../secret.py",
        "tools/../../etc/passwd",
        "../outside.py",
        # git and archive internals
        ".git/config",
        ".git/hooks/post-checkout",
        "tools/.git/hooks/pre-commit",
        ".iterant/archive.db",
        ".iterant/diffs/x.diff",
        # Windows tricks: backslashes, alternate streams, trailing dots and spaces, short names
        "tools\\..\\secret.py",
        "secret.py::$DATA",
        "secret.py.",
        "secret.py ",
        "SECRET~1.PY",
        # absolute paths
        "/etc/passwd",
        "C:/Windows/win.ini",
    ],
)
def test_row1_path_tricks_are_rejected(run_gate: Check, repo: Path, path: str) -> None:
    before = snapshot(repo)
    diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
    must_reject(run_gate(diff))
    assert snapshot(repo) == before, "the gate wrote into the real repo"


FORMAT_TRICKS = {
    "crlf-line-endings": "diff --git a/secret.py b/secret.py\r\n--- a/secret.py\r\n"
    "+++ b/secret.py\r\n@@ -1 +1 @@\r\n-KEY = 1\r\n+KEY = 2\r\n",
    "octal-escaped-dots": 'diff --git "a/tools/\\056\\056/secret.py" "b/tools/\\056\\056/secret.py"\n'  # noqa: E501
    '--- "a/tools/\\056\\056/secret.py"\n+++ "b/tools/\\056\\056/secret.py"\n'
    "@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",
    "timestamp-after-tab": "--- a/secret.py\t2020-01-01\n+++ b/secret.py\t2020-01-01\n"
    "@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",
    "no-path-prefix": "--- secret.py\n+++ secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",
    "dotdot-after-prefix": "--- a/x/../secret.py\n+++ b/x/../secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",  # noqa: E501
    "two-dot-leader-lookalike": "diff --git a/tools/‥/secret.py b/tools/‥/secret.py\n"
    "--- a/tools/‥/secret.py\n+++ b/tools/‥/secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",
    "leading-bom": "﻿--- a/secret.py\n+++ b/secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n",
    "directory-as-file": "diff --git a/tools/new/ b/tools/new/\nnew file mode 100644\n"
    "--- /dev/null\n+++ b/tools/new/\n@@ -0,0 +1 @@\n+x\n",
    "mode-only-outside-genome": "diff --git a/secret.py b/secret.py\nold mode 100644\nnew mode 100755\n",  # noqa: E501
    "submodule": "diff --git a/tools/sub b/tools/sub\nnew file mode 160000\n"
    "index 0000000..1111111\n",
    "header-only-no-hunks": "diff --git a/tools/search.py b/tools/search.py\n"
    "index 1111111..2222222 100644\n",
    "rename-outside-in-genome": "diff --git a/secret.py b/tools/secret.py\nsimilarity index 100%\n"
    "rename from secret.py\nrename to tools/secret.py\n",
}


@pytest.mark.parametrize("name", sorted(FORMAT_TRICKS))
def test_row1_diff_format_tricks_are_rejected(run_gate: Check, name: str) -> None:
    must_reject(run_gate(FORMAT_TRICKS[name]))


def test_row1_header_lies_about_the_path(run_gate: Check) -> None:
    """The ``diff --git`` line names a genome file; ``---``/``+++`` (what git applies) name
    another."""
    diff = (
        "diff --git a/tools/search.py b/tools/search.py\n"
        "--- a/secret.py\n+++ b/secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n"
    )
    must_reject(run_gate(diff))


def test_row1_plain_diff_old_in_genome_new_outside(run_gate: Check) -> None:
    must_reject(run_gate(plain("tools/search.py", "secret.py")))


def test_row1_second_section_hides_after_a_clean_one(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/tools/search.py\n"
        "--- a/tools/search.py\n+++ b/tools/search.py\n@@ -1 +1 @@\n"
        "-def search(q):\n+def search(query):\n"
        "diff --git a/evals/pack/tasks/beta-9/check b/evals/pack/tasks/beta-9/check\n"
        "--- a/evals/pack/tasks/beta-9/check\n+++ b/evals/pack/tasks/beta-9/check\n"
        "@@ -1,2 +1,2 @@\n #!/bin/sh\n-test -f /out/done && exit 0\n+exit 0\n"
    )
    must_reject(run_gate(diff))


def test_row1_hunk_longer_than_its_header_cannot_smuggle_a_section(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/tools/search.py\n"
        "--- a/tools/search.py\n+++ b/tools/search.py\n@@ -1 +1 @@\n"
        "-def search(q):\n+def search(query):\n"
        "+extra line the header did not count\n"
        "--- a/secret.py\n+++ b/secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n"
    )
    must_reject(run_gate(diff))


def test_row1_delete_a_hidden_check(
    run_gate: Check, edit: Callable[[str, str | None], str]
) -> None:
    must_reject(run_gate(edit("evals/pack/tasks/beta-9/check", None)))


def test_row1_copy_a_hidden_check_into_the_genome(run_gate: Check) -> None:
    """Also a row 2 attack: the copy would carry the hidden check into a file the agent reads."""
    diff = (
        "diff --git a/evals/pack/tasks/beta-9/check b/tools/stolen.py\n"
        "similarity index 100%\ncopy from evals/pack/tasks/beta-9/check\ncopy to tools/stolen.py\n"
    )
    must_reject(run_gate(diff))


def test_row1_rename_a_genome_file_over_the_toml(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/iterant.toml\nsimilarity index 100%\n"
        "rename from tools/search.py\nrename to iterant.toml\n"
    )
    must_reject(run_gate(diff))


def test_row1_symlink_created_then_written_through(run_gate: Check) -> None:
    """Two sections: first plants ``tools/evil -> ../evals/pack``, then writes through it."""
    diff = (
        "diff --git a/tools/evil b/tools/evil\nnew file mode 120000\n"
        "--- /dev/null\n+++ b/tools/evil\n@@ -0,0 +1 @@\n+../evals/pack\n"
        "\\ No newline at end of file\n"
        "diff --git a/tools/evil/tasks/beta-9/check b/tools/evil/tasks/beta-9/check\n"
        "--- a/tools/evil/tasks/beta-9/check\n+++ b/tools/evil/tasks/beta-9/check\n"
        "@@ -1 +1 @@\n-x\n+exit 0\n"
    )
    must_reject(run_gate(diff))


def test_row1_genome_file_turned_into_a_symlink_by_mode_change(run_gate: Check) -> None:
    diff = "diff --git a/tools/search.py b/tools/search.py\nold mode 100644\nnew mode 120000\n"
    must_reject(run_gate(diff))


def test_row1_write_through_a_preexisting_directory_symlink(repo: Path, run_gate: Check) -> None:
    link = repo / "tools" / "linkdir"
    try:
        os.symlink(repo / "evals" / "pack", link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    diff = (
        "diff --git a/tools/linkdir/tasks/beta-9/check b/tools/linkdir/tasks/beta-9/check\n"
        "--- a/tools/linkdir/tasks/beta-9/check\n+++ b/tools/linkdir/tasks/beta-9/check\n"
        "@@ -1,2 +1,2 @@\n #!/bin/sh\n-test -f /out/done && exit 0\n+exit 0\n"
    )
    before = (repo / "evals/pack/tasks/beta-9/check").read_bytes()
    must_reject(run_gate(diff))
    assert (repo / "evals/pack/tasks/beta-9/check").read_bytes() == before


def test_row1_meta_mode_a_candidate_cannot_edit_the_gate_or_the_grader(
    repo: Path, pack: Pack, make_gate: Callable[..., DiffGate]
) -> None:
    """Research mode (§8): the genome is the proposer's own code. The gate, grader, proxy and
    stats beside it are the TCB and stay frozen."""
    for rel in ("iterant/proposer/prompt.md", "iterant/gate/gate.py", "iterant/grader/grader.py"):
        f = repo / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x = 1\n", encoding="utf-8")
    meta = TargetConfig(
        ("iterant/proposer/**",), "Containerfile", "x", TargetModel("m", "medium", 1)
    )
    gate = make_gate()
    for rel in (
        "iterant/gate/gate.py",
        "iterant/grader/grader.py",
        "iterant/proposer/../gate/gate.py",
    ):
        diff = (
            f"diff --git a/{rel} b/{rel}\n--- a/{rel}\n+++ b/{rel}\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        )
        must_reject(gate.check(Proposal(diff, "h"), repo=repo, target=meta, pack=pack))


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        ".iterant/archive.db",
        "iterant.toml",
        "evals/pack/pack.toml",
        # case variants name the same files on case-insensitive filesystems (Windows, macOS)
        "Iterant.Toml",
        "CONTAINERFILE",
        "EVALS/PACK/tasks/beta-9/check",
        ".GIT/config",
        ".Iterant/archive.db",
        # trailing dot/space and 8.3 aliases the OS resolves to the same file
        "iterant.toml.",
        "iterant.toml ",
        "ITERAN~1.TOM",
        "evals/pack./tasks/beta-9/check",
        "evals./pack/tasks/beta-9/check",
        ".git./config",
    ],
)
def test_row1_even_a_wildcard_genome_protects_the_tcb(
    repo: Path, pack: Pack, make_gate: Callable[..., DiffGate], path: str
) -> None:
    wide = TargetConfig(("**",), "Containerfile", "x", TargetModel("m", "medium", 1))
    diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-x\n+y\n"
    r = make_gate().check(Proposal(diff, "h"), repo=repo, target=wide, pack=pack)
    assert r.findings and r.findings[0].startswith("path-allowlist:"), r.findings
    must_reject(r)


def test_row1_oversized_diffs_are_rejected_before_anything_runs(run_gate: Check) -> None:
    many_lines = "".join(f"+line {i}\n" for i in range(500))
    big = f"diff --git a/tools/a.py b/tools/a.py\nnew file mode 100644\n--- /dev/null\n+++ b/tools/a.py\n@@ -0,0 +1,500 @@\n{many_lines}"  # noqa: E501
    must_reject(run_gate(big))
    long_line = "+" + "x" * 20_000
    huge = f"diff --git a/tools/b.py b/tools/b.py\nnew file mode 100644\n--- /dev/null\n+++ b/tools/b.py\n@@ -0,0 +1 @@\n{long_line}\n"  # noqa: E501
    must_reject(run_gate(huge))
    must_reject(run_gate("not a diff at all\n"))
    must_reject(
        run_gate("diff --git a/tools/a.py b/tools/a.py\n--- a/tools/a.py\n+++ b/tools/a.py\n\x00\n")
    )
