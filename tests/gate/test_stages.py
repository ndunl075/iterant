"""Stage order, the clean path, size and apply rules, target lint/tests, and the audit stage."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import ExecResult, Gate, Pack, Proposal, TargetConfig
from iterant.gate import (
    AuditorError,
    AuditParseError,
    AuditVerdict,
    DiffGate,
    GateInfraError,
    GateReport,
    StaticChecks,
)

Edit = Callable[[str, str | None], str]
Check = Callable[[str], GateReport]
GOOD = "You are a careful assistant.\nAnswer in one sentence.\n"


def test_gate_satisfies_the_protocol(make_gate: Callable[..., DiffGate]) -> None:
    assert isinstance(make_gate(), Gate)


def test_clean_diff_passes_and_is_audited(run_gate: Check, edit: Edit, auditor: Any) -> None:
    diff = edit("prompts/system.md", GOOD)
    result = run_gate(diff)
    assert result.passed and result.findings == () and result.warnings == ()
    assert auditor.calls == [(diff, "allowed")]


def test_repo_is_never_written(repo: Path, run_gate: Check, edit: Edit) -> None:
    before = (repo / "prompts" / "system.md").read_text(encoding="utf-8")
    assert run_gate(edit("prompts/system.md", GOOD)).passed
    assert (repo / "prompts" / "system.md").read_text(encoding="utf-8") == before


def test_stops_at_the_first_failing_stage(run_gate: Check, edit: Edit, auditor: Any) -> None:
    # Outside the genome AND mentions a task id: only the allowlist reports.
    result = run_gate(edit("secret.py", 'X = "alpha-17"\n'))
    assert [f.split(":")[0] for f in result.findings] == ["path-allowlist"]
    assert auditor.calls == []


def test_leak_stage_runs_before_the_auditor(run_gate: Check, edit: Edit, auditor: Any) -> None:
    run_gate(edit("tools/search.py", "# beta-9\n"))
    assert auditor.calls == []


def test_size_limit(run_gate: Check, edit: Edit) -> None:
    result = run_gate(edit("tools/big.py", "x = 1\n" * 401))
    assert result.findings[0].startswith("diff-size:")
    assert run_gate(edit("tools/big.py", "x = 1\n" * 400)).passed


def test_size_counts_removed_lines_too(repo: Path, run_gate: Check, edit: Edit) -> None:
    (repo / "tools" / "big.py").write_text("y = 2\n" * 401, encoding="utf-8", newline="\n")
    assert run_gate(edit("tools/big.py", None)).findings[0].startswith("diff-size:")


@pytest.mark.parametrize(
    "diff",
    [
        "this is not a diff\n",
        "--- a/tools/search.py\n",
        "diff --git a/tools/search.py b/tools/search.py\n--- a/tools/search.py\n"
        "+++ b/tools/search.py\n@@ -1,2 +1,2 @@\n-def search(q):\n",
        "diff --git a/tools/x b/tools/x\nnew file mode 100644\n--- /dev/null\n+++ b/tools/x\n"
        "@@ -0,0 +1 @@\n+a\n+b\n",
        "diff --git a/tools/x b/tools/x\n\x00\n",
    ],
)
def test_malformed_diff(run_gate: Check, diff: str) -> None:
    assert run_gate(diff).findings[0].startswith("diff-malformed:")


def test_diff_that_does_not_apply(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/tools/search.py\n--- a/tools/search.py\n"
        "+++ b/tools/search.py\n@@ -1 +1 @@\n-no such line\n+replacement\n"
    )
    assert run_gate(diff).findings[0].startswith("diff-apply:")


def test_plain_unified_diff_is_accepted(run_gate: Check) -> None:
    diff = (
        "--- a/prompts/system.md\n+++ b/prompts/system.md\n@@ -1 +1 @@\n"
        "-You are a careful assistant.\n+Be brief.\n"
    )
    assert run_gate(diff).passed


def test_rename_inside_the_genome_is_allowed(run_gate: Check) -> None:
    diff = (
        "diff --git a/tools/search.py b/tools/find.py\nsimilarity index 100%\n"
        "rename from tools/search.py\nrename to tools/find.py\n"
    )
    assert run_gate(diff).passed


def test_exec_bit_flip_inside_the_genome_is_allowed(run_gate: Check) -> None:
    diff = "diff --git a/tools/search.py b/tools/search.py\nold mode 100644\nnew mode 100755\n"
    assert run_gate(diff).passed


# --- target lint and tests ----------------------------------------------------


def _static(backend: Any, **kw: Any) -> StaticChecks:
    return StaticChecks(backend=backend, image="target:img", **kw)


def test_lint_and_tests_run_in_a_sandbox(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
) -> None:
    gate = make_gate(static=_static(fake_backend, lint="ruff check .", test="pytest -q"))
    result = gate.check(
        Proposal(edit("prompts/system.md", GOOD), "h"), repo=repo, target=target, pack=pack
    )
    assert result.passed
    assert fake_backend.nets == ["default-deny"]
    assert [c for c in fake_backend.cmds if "ruff" in c or "pytest" in c] == [
        "cd /work && ruff check .",
        "cd /work && pytest -q",
    ]
    assert fake_backend.created[0].destroyed
    names = tarfile.open(fileobj=io.BytesIO(fake_backend.puts["/tmp/iterant-gate.tar"])).getnames()
    assert "prompts/system.md" in names and not any(
        n.startswith((".git", ".iterant")) for n in names
    )
    tar = tarfile.open(fileobj=io.BytesIO(fake_backend.puts["/tmp/iterant-gate.tar"]))
    member = tar.extractfile("prompts/system.md")
    assert member is not None and member.read().decode() == GOOD  # the patched tree


@pytest.mark.parametrize(
    ("kind", "result", "rule"),
    [
        ("lint", ExecResult(1, "", "E501 line too long", False, 0.1), "target-lint"),
        ("test", ExecResult(2, "1 failed", "", False, 0.1), "target-tests"),
        ("test", ExecResult(-1, "", "", True, 9.0), "target-tests"),
        ("lint", ExecResult(0, "", "", False, 0.1, truncated=True), "target-lint"),
        ("test", ExecResult(137, "", "", False, 0.1, sandbox_lost=True), "target-tests"),
    ],
)
def test_lint_or_test_failure_rejects(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
    kind: str,
    result: ExecResult,
    rule: str,
) -> None:
    fake_backend.results["ruff" if kind == "lint" else "pytest"] = result
    gate = make_gate(static=_static(fake_backend, lint="ruff check .", test="pytest -q"))
    out = gate.check(
        Proposal(edit("prompts/system.md", GOOD), "h"), repo=repo, target=target, pack=pack
    )
    assert out.findings[0].startswith(rule + ":")
    assert fake_backend.created[0].destroyed


def test_failing_lint_skips_tests_and_the_auditor(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
    auditor: Any,
) -> None:
    fake_backend.results["ruff"] = ExecResult(1, "", "bad", False, 0.1)
    gate = make_gate(static=_static(fake_backend, lint="ruff check .", test="pytest -q"))
    gate.check(Proposal(edit("prompts/system.md", GOOD), "h"), repo=repo, target=target, pack=pack)
    assert not any("pytest" in c for c in fake_backend.cmds) and auditor.calls == []


def test_no_sandbox_when_nothing_is_declared(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
) -> None:
    gate = make_gate(static=_static(fake_backend))
    gate.check(Proposal(edit("prompts/system.md", GOOD), "h"), repo=repo, target=target, pack=pack)
    assert fake_backend.created == []


def test_sandbox_is_not_used_for_a_diff_that_left_the_genome(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
) -> None:
    gate = make_gate(static=_static(fake_backend, test="pytest -q"))
    gate.check(Proposal(edit("secret.py", "X = 1\n"), "h"), repo=repo, target=target, pack=pack)
    assert fake_backend.created == []


def test_unpack_failure_is_infrastructure(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    fake_backend: Any,
) -> None:
    fake_backend.results["tar -xf"] = ExecResult(127, "", "tar: not found", False, 0.0)
    gate = make_gate(static=_static(fake_backend, test="pytest -q"))
    with pytest.raises(GateInfraError):
        gate.check(
            Proposal(edit("prompts/system.md", GOOD), "h"), repo=repo, target=target, pack=pack
        )
    assert fake_backend.created[0].destroyed


# --- audit stage ---------------------------------------------------------------


def test_advisory_policy_is_passed_to_the_auditor(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    auditor: Any,
) -> None:
    advisory = Pack(
        pack.name,
        pack.version,
        pack.tasks,
        pack.content_hash,
        pack.grader_image,
        "advisory-forbidden",
    )
    auditor.verdict = AuditVerdict(False, True, ("adds subprocess.run",))
    diff = edit("tools/search.py", "import subprocess\n\ndef search(q):\n    return []\n")
    result = make_gate().check(Proposal(diff, "h"), repo=repo, target=target, pack=advisory)
    assert auditor.calls[0][1] == "advisory-forbidden"
    assert result.passed and result.findings == ()
    assert len(result.warnings) == 1
    assert result.warnings[0].startswith("audit-code-exec:") and "unenforced" in result.warnings[0]


def test_code_execution_is_not_flagged_when_policy_allows_it(
    run_gate: Check, edit: Edit, auditor: Any
) -> None:
    auditor.verdict = AuditVerdict(False, True, ())
    result = run_gate(edit("tools/search.py", "import subprocess\n"))
    assert result.passed and result.warnings == ()


def test_special_casing_still_rejects_under_advisory_policy(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Callable[..., DiffGate],
    edit: Edit,
    auditor: Any,
) -> None:
    advisory = Pack(
        pack.name,
        pack.version,
        pack.tasks,
        pack.content_hash,
        pack.grader_image,
        "advisory-forbidden",
    )
    auditor.verdict = AuditVerdict(True, True, ("lookup table",))
    result = make_gate().check(
        Proposal(edit("tools/search.py", "T = {}\n"), "h"), repo=repo, target=target, pack=advisory
    )
    assert not result.passed and result.findings[0].startswith("audit-special-casing:")


def test_unparseable_audit_rejects(run_gate: Check, edit: Edit, auditor: Any) -> None:
    auditor.error = AuditParseError("reply is not JSON")
    assert run_gate(edit("tools/search.py", "x = 1\n")).findings[0].startswith("audit-unparseable:")


def test_unreachable_auditor_is_infrastructure(run_gate: Check, edit: Edit, auditor: Any) -> None:
    auditor.error = AuditorError("auditor call refused: HTTP 503")
    with pytest.raises(GateInfraError):
        run_gate(edit("tools/search.py", "x = 1\n"))
