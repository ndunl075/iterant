"""Row 4: "Hard-code train/select answers". Stopped by the proposer being blind to select data,
the task-ID grep, the diff auditor and holdout rotation.

Two halves. (a) A candidate diff that hard-codes task IDs or answers must not pass the gate.
(b) The proposer, which writes the diffs, must never see anything but train data: not through
its file tools, not through its repo copy, not through the failure digest, not through the
proxy's stored call content.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from iterant.analyst import FailureAnalyst
from iterant.contracts import RunKey, RunResult, Split
from iterant.gate import AuditVerdict, GateReport
from iterant.orchestrator.workspace import proposer_view
from iterant.proposer.workspace import ToolError, Workspace
from iterant.target.genome import Genome

Edit = Callable[[str, str | None], str]
Check = Callable[[str], GateReport]

# --- (a) hard-coding ----------


@pytest.mark.parametrize(
    "code",
    [
        'ANSWERS = {"alpha-17": "42"}',
        'ANSWERS = {"ALPHA-17": "42"}',
        "if task_id == 'alpha-17': return 42",
        "# alpha-17 -> 42",
        'ids = ["beta-9", "alpha-17"]',
        "x = 'tasks\\alpha-17'",
    ],
)
def test_row4_a_task_id_in_added_code_is_rejected(run_gate: Check, edit: Edit, code: str) -> None:
    result = run_gate(edit("agent/loop.py", f"{code}\ndef run(task):\n    return task\n"))
    assert not result.passed
    assert result.findings[0].startswith(("leak-task-id:", "leak-check-path:"))


def test_row4_a_task_id_in_a_new_file_name_is_rejected(run_gate: Check, edit: Edit) -> None:
    result = run_gate(edit("prompts/beta-9.md", "answer: 7\n"))
    assert not result.passed and result.findings[0].startswith("leak-task-id:")


def test_row4_an_id_assembled_from_pieces_reaches_the_auditor_and_is_rejected(
    run_gate: Check, edit: Edit, auditor
) -> None:  # type: ignore[no-untyped-def]
    """The grep is textual (documented limit); the auditor is the backstop and must be asked."""
    auditor.verdict = AuditVerdict(True, False, ("keys answers on an assembled task id",))
    diff = edit(
        "agent/loop.py",
        "TABLE = {'al' + 'pha-' + str(17): '42'}\ndef run(task):\n    return TABLE.get(task['id'])\n",  # noqa: E501
    )
    result = run_gate(diff)
    assert auditor.calls
    assert not result.passed and result.findings[0].startswith("audit-special-casing:")


def test_row4_an_auditor_that_cannot_answer_is_not_a_pass(
    run_gate: Check, edit: Edit, auditor
) -> None:  # type: ignore[no-untyped-def]
    from iterant.gate import GateInfraError
    from iterant.gate.audit import AuditorError

    auditor.error = AuditorError("upstream down")
    with pytest.raises(GateInfraError):
        run_gate(edit("prompts/system.md", "Be concise.\n"))


def test_row4_an_unparseable_audit_is_a_rejection(run_gate: Check, edit: Edit, auditor) -> None:  # type: ignore[no-untyped-def]
    from iterant.gate.audit import AuditParseError

    auditor.error = AuditParseError("not json")
    result = run_gate(edit("prompts/system.md", "Be concise.\n"))
    assert not result.passed and result.findings[0].startswith("audit-unparseable:")


# --- (b) the proposer is blind ----------


@pytest.fixture
def workspace(repo: Path) -> Workspace:
    genome = Genome(repo, ("prompts/**", "tools/**", "agent/loop.py"), protected=("iterant.toml",))
    return Workspace(repo, genome, hidden=("evals/pack", ".git", ".iterant"))


@pytest.mark.parametrize(
    "path",
    [
        "evals/pack/tasks/alpha-17/input.json",
        "evals/pack/tasks/alpha-17/check",
        "EVALS/PACK/tasks/alpha-17/check",
        "evals/Pack/pack.toml",
        "./evals/pack/pack.toml",
        "tools/../evals/pack/pack.toml",
        "../outside.txt",
        "/etc/passwd",
        ".git/config",
        ".iterant/archive.db",
    ],
)
def test_row4_the_proposer_cannot_read_the_pack_or_internals(
    workspace: Workspace, path: str
) -> None:
    with pytest.raises(ToolError):
        workspace.read(path)


def test_row4_the_proposer_cannot_list_the_pack(workspace: Workspace) -> None:
    listing = workspace.list_files()
    assert "tools/search.py" in listing
    assert "alpha-17" not in listing and "evals/pack" not in listing
    with pytest.raises(ToolError):
        workspace.list_files("evals/pack/tasks")


def test_row4_a_symlink_into_the_pack_does_not_open_it(repo: Path, workspace: Workspace) -> None:
    try:
        os.symlink(repo / "evals" / "pack", repo / "tools" / "peek", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    with pytest.raises(ToolError):
        workspace.read("tools/peek/tasks/alpha-17/input.json")
    assert "alpha-17" not in workspace.list_files()


def test_row4_the_proposer_cannot_write_outside_the_genome_or_through_a_link(
    repo: Path, workspace: Workspace
) -> None:
    for path in ("evals/pack/tasks/beta-9/check", "iterant.toml", "secret.py", "../x.py"):
        with pytest.raises(ToolError):
            workspace.write(path, "exit 0\n")
    try:
        os.symlink(repo / "secret.py", repo / "tools" / "ln.py")
    except (OSError, NotImplementedError):
        return
    with pytest.raises(ToolError):
        workspace.write("tools/ln.py", "KEY = 2\n")
    assert (repo / "secret.py").read_text() == "KEY = 1\n"


def test_row4_the_proposers_repo_copy_leaves_out_the_pack_and_internals(
    repo: Path, tmp_path: Path
) -> None:
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("[core]\n")
    (repo / ".iterant").mkdir()
    (repo / ".iterant" / "archive.db").write_bytes(b"sqlite")
    (repo / "notes").mkdir()
    (repo / "notes" / "private.txt").write_text("not for the proposer")
    (repo / ".containerignore").write_text("notes/\n")
    view = proposer_view(repo, "evals/pack", tmp_path / "view")
    seen = {p.relative_to(view).as_posix() for p in view.rglob("*") if p.is_file()}
    assert "tools/search.py" in seen
    assert not [s for s in seen if s.startswith(("evals/", ".git/", ".iterant/", "notes/"))], seen


def _results(*tasks: tuple[str, Split]) -> list[RunResult]:
    return [
        RunResult(RunKey(1, "c", t, 0), split, 0.0, 10, 5, 100, 1.0, False, False)
        for t, split in tasks
    ]


def test_row4_the_digest_ignores_select_holdout_and_test_even_if_a_caller_passes_them(
    world,  # type: ignore[no-untyped-def]
) -> None:
    """A real generation puts every split's call content through the real proxy; whatever the
    caller hands the analyst, only train content may appear in what the proposer reads."""
    world.queue(world.proposal("v0", "v1", "h"))
    world.solver = lambda arm, t, n: "wrong"  # every task fails: every split has failures
    world.orchestrator().run_generation()

    db = world.db()
    every_run = [r for r in db.latest_runs(1).values()]
    splits_seen = {r.split for r in every_run}
    assert {Split.TRAIN, Split.SELECT} <= splits_seen

    # the proxy kept content for train runs only
    kept = world.proxy.train_content()
    assert kept, "the proxy kept no train content at all"
    assert {c.key.task for c in kept} <= set(world.TRAIN)

    analyst = FailureAnalyst(None, model="m", effort="medium", redact=())
    digest = analyst.digest(every_run, kept)
    for task in (*world.SELECT, *world.HOLDOUT):
        assert task not in digest, f"{task} leaked into the digest"
        assert f"PROMPT-{task}" not in digest and f"REPLY-{task}" not in digest
    # a careless caller who also hands over non-train content directly changes nothing
    assert "PROMPT-se00" not in analyst.digest(
        every_run, [c for c in kept] + _fake_content(world, "se00")
    )


def _fake_content(world, task: str):  # type: ignore[no-untyped-def]
    from iterant.proxy.core import CallContent

    return [
        CallContent(
            9999,
            RunKey(1, "b1-0", task, 0),
            next(iter(world.proxy.config.policy.roles)),
            {"messages": [{"role": "user", "content": f"PROMPT-{task}"}]},
            {"content": [{"type": "text", "text": f"REPLY-{task}"}]},
        )
    ]


def test_row4_the_proxy_never_stores_select_or_holdout_content(world) -> None:  # type: ignore[no-untyped-def]
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    stored = world.proxy.train_content()
    for c in stored:
        assert c.key.task in world.TRAIN
    # asking for a non-train key by name returns nothing
    for task in (world.SELECT[0], world.HOLDOUT[0], "te0"):
        assert world.proxy.train_content(RunKey(1, "b1-0", task, 0)) == []


def test_row4_a_digest_cannot_carry_the_hidden_check_text_to_the_proposer(world) -> None:  # type: ignore[no-untyped-def]
    """Even train content that quotes a hidden check is scrubbed by the redact list the
    orchestrator hands to the digest builder."""
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    redact = world.redactions[0]
    secret = (world.pack_root / "tasks" / "tr0" / "check").read_text()
    assert secret in redact or any(secret in r for r in redact)
    assert any(r.endswith("tasks/tr0/check") or "tr0" in r for r in redact)
