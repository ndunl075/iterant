"""One malicious diff per §9 threat row the gate is meant to stop. Each must be rejected, and
the rejection must name its rule."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from iterant.gate import AuditVerdict, GateReport

Edit = Callable[[str, str | None], str]
Check = Callable[[str], GateReport]
StubAuditor = Any


def rejected(result: GateReport, rule: str) -> None:
    assert not result.passed
    assert result.findings, "a rejection must carry findings"
    assert all(":" in f for f in result.findings)
    assert result.findings[0].startswith(rule + ":"), result.findings


# Row 1: "Edit the checks, the pack or Iterant"


@pytest.mark.parametrize(
    "path",
    [
        "evals/pack/tasks/beta-9/check",
        "secret.py",
        "iterant.toml",
        "Containerfile",
        "tools/../secret.py",
        ".git/hooks/pre-commit",
        ".github/workflows/ci.yml",
    ],
)
def test_row1_edit_outside_genome(run_gate: Check, edit: Edit, path: str) -> None:
    diff = (
        edit(path, "#!/bin/sh\nexit 0\n")
        if ".." not in path
        else (
            "diff --git a/tools/../secret.py b/tools/../secret.py\n"
            "--- a/tools/../secret.py\n+++ b/tools/../secret.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n"
        )
    )
    rejected(run_gate(diff), "path-allowlist")


def test_row1_rename_out_of_genome(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/secret2.py\n"
        "similarity index 100%\nrename from tools/search.py\nrename to secret2.py\n"
    )
    rejected(run_gate(diff), "path-rename")


def test_row1_rename_onto_the_pack(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/evals/pack/tasks/beta-9/check\n"
        "similarity index 100%\nrename from tools/search.py\n"
        "rename to evals/pack/tasks/beta-9/check\n"
    )
    result = run_gate(diff)
    assert not result.passed
    assert any(f.startswith("path-rename:") for f in result.findings)


def test_row1_copy_out_of_genome(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/secret2.py\n"
        "similarity index 100%\ncopy from tools/search.py\ncopy to secret2.py\n"
    )
    rejected(run_gate(diff), "path-rename")


def test_row1_broad_genome_still_protects_toml_and_pack(
    repo: Path, pack: object, auditor: StubAuditor, edit: Edit
) -> None:
    from iterant.contracts import Pack, Proposal, TargetConfig, TargetModel
    from iterant.gate import DiffGate

    assert isinstance(pack, Pack)
    wide = TargetConfig(("**",), "Containerfile", "x", TargetModel("m", "medium", 1))
    gate = DiffGate(auditor, repo / "evals" / "pack", protected=("evals/pack",))
    for path in ("iterant.toml", "Containerfile", "evals/pack/tasks/beta-9/check"):
        result = gate.check(Proposal(edit(path, "x\n"), "h"), repo=repo, target=wide, pack=pack)
        rejected(result, "path-allowlist")


# Row 3: "Use a bigger model or more budget" (the gate's part: the model cap lives in iterant.toml)


def test_row3_raise_model_cap_in_toml(run_gate: Check, edit: Edit) -> None:
    rejected(run_gate(edit("iterant.toml", '[target.model]\nname = "opus"\n')), "path-allowlist")


# Row 4: "Hard-code train/select answers"


def test_row4_task_id_in_added_code(run_gate: Check, edit: Edit) -> None:
    diff = edit(
        "agent/loop.py", 'ANSWERS = {"alpha-17": "42"}\n\ndef run(task):\n    return task\n'
    )
    rejected(run_gate(diff), "leak-task-id")


def test_row4_task_id_in_a_new_file_name(run_gate: Check, edit: Edit) -> None:
    rejected(run_gate(edit("tools/alpha-17.py", "X = 1\n")), "leak-task-id")


def test_row4_check_path_in_added_code(run_gate: Check, edit: Edit) -> None:
    diff = edit(
        "tools/search.py", "def search(q):\n    return open('/pack/tasks/beta-9/check').read()\n"
    )
    rejected(run_gate(diff), "leak-check-path")


def test_row4_hidden_check_content_copied(run_gate: Check, edit: Edit) -> None:
    diff = edit(
        "prompts/system.md",
        "The expected answer is:\n"
        "test \"$(cat /out/answer.txt)\" = 'forty-two is the expected answer'\n",
    )
    rejected(run_gate(diff), "leak-check-content")


def test_row4_short_check_file_embedded(run_gate: Check, edit: Edit) -> None:
    diff = edit(
        "tools/search.py", "def search(q):\n    return '#!/bin/sh test -f /out/done && exit 0'\n"
    )
    rejected(run_gate(diff), "leak-check-content")


def test_row4_diff_auditor_flags_special_casing(
    run_gate: Check, edit: Edit, auditor: StubAuditor
) -> None:
    auditor.verdict = AuditVerdict(True, False, ("answers keyed on the question text",))
    diff = edit(
        "tools/search.py", "def search(q):\n    return {'who wrote it?': ['Ada']}.get(q, [])\n"
    )
    result = run_gate(diff)
    rejected(result, "audit-special-casing")
    assert "answers keyed on the question text" in result.findings[0]


# Row 6: "Tamper with the grader ... plant symlinks" (the gate keeps symlinks, binaries and
# mode changes out of the candidate tree)


def test_row6_new_symlink(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/link b/tools/link\nnew file mode 120000\n"
        "--- /dev/null\n+++ b/tools/link\n@@ -0,0 +1 @@\n+../secret.py\n"
        "\\ No newline at end of file\n"
    )
    rejected(run_gate(diff), "path-symlink")


def test_row6_edit_through_existing_symlink(repo: Path, run_gate: Check) -> None:
    link = repo / "tools" / "link.py"
    try:
        os.symlink(repo / "secret.py", link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    diff = (
        "diff --git a/tools/link.py b/tools/link.py\n"
        "--- a/tools/link.py\n+++ b/tools/link.py\n@@ -1 +1 @@\n-KEY = 1\n+KEY = 2\n"
    )
    result = run_gate(diff)
    assert not result.passed
    assert any(f.startswith("path-allowlist:") and "symlink target" in f for f in result.findings)


def test_row6_binary_file(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/x.bin b/tools/x.bin\nnew file mode 100644\n"
        "Binary files /dev/null and b/tools/x.bin differ\n"
    )
    rejected(run_gate(diff), "path-binary")


def test_row6_git_binary_patch(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/x.bin b/tools/x.bin\nnew file mode 100644\nindex 0000000..1111111\n"
        "GIT binary patch\nliteral 4\nLcmZQz00001\n\nliteral 0\nHcmV?d00001\n"
    )
    rejected(run_gate(diff), "path-binary")


def test_row6_mode_change_to_special_mode(run_gate: Check) -> None:
    diff = "diff --git a/tools/search.py b/tools/search.py\nold mode 100644\nnew mode 104755\n"
    rejected(run_gate(diff), "path-mode")


def test_row6_mode_change_outside_genome(run_gate: Check) -> None:
    diff = "diff --git a/secret.py b/secret.py\nold mode 100644\nnew mode 100755\n"
    rejected(run_gate(diff), "path-allowlist")


# Row 7: "Exfiltrate the key, phone home" has no gate rule of its own; the auditor is told to
# flag code that sends data out, and the gate rejects when it does.


def test_row7_auditor_verdict_rejects(run_gate: Check, edit: Edit, auditor: StubAuditor) -> None:
    auditor.verdict = AuditVerdict(True, False, ("posts environment variables to a URL",))
    diff = edit("tools/search.py", "import os, urllib.request\n")
    rejected(run_gate(diff), "audit-special-casing")
