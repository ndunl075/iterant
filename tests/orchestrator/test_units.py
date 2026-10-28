from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from iterant.contracts import ExecResult
from iterant.grader import OutputRejected
from iterant.orchestrator import (
    Budget,
    BudgetExhausted,
    DiffStore,
    OrchestratorConfig,
    agent_command,
    bind_paths,
    export_commit,
    read_out,
)
from iterant.sandbox import SandboxError


@dataclass
class ListSandbox:
    listing: str
    files: dict[str, bytes] = field(default_factory=lambda: {})
    exit_code: int = 0

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        return ExecResult(self.exit_code, self.listing, "", False, 0.1)

    def get(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


def test_read_out_returns_regular_files_and_skips_directories():
    listing = "d\t1\t0\tsub\0f\t1\t3\tsub/a.txt\0f\t1\t1\tb\0"
    sb = ListSandbox(listing, {"/o/sub/a.txt": b"abc", "/o/b": b"x"})
    assert read_out(sb, "/o") == {"sub/a.txt": b"abc", "b": b"x"}  # type: ignore[arg-type]


BIG = "".join(f"f\t1\t3000000\tf{i}\0" for i in range(6))


@pytest.mark.parametrize(
    "listing",
    [
        "l\t1\t0\tlink\0",  # symlink
        "f\t2\t3\thard\0",  # hard link
        "p\t1\t0\tfifo\0",  # special file
        "f\t1\t5000000\tbig\0",  # over the per-file cap
        BIG,  # over the total cap
        "garbage\0",
        "f\t1\tx\tname\0",
        "".join(f"d\t1\t0\td{i}\0" for i in range(2001)),  # too many entries
        "".join(f"f\t1\t0\tf{i}\0" for i in range(1001)),  # too many files
    ],
    ids="symlink hardlink fifo big-file big-total garbage bad-size entries files".split(),
)
def test_read_out_rejects_what_invariant_8_forbids(listing):
    with pytest.raises(OutputRejected):
        read_out(ListSandbox(listing), "/o")  # type: ignore[arg-type]


def test_read_out_rejects_an_unlistable_or_vanishing_output():
    with pytest.raises(OutputRejected):
        read_out(ListSandbox("", exit_code=1), "/o")  # type: ignore[arg-type]
    with pytest.raises(OutputRejected):
        read_out(ListSandbox("f\t1\t1\tgone\0"), "/o")  # type: ignore[arg-type]


def test_read_out_reraises_sandbox_errors_for_the_executor_to_judge():
    class Unreadable(ListSandbox):
        def get(self, path: str) -> bytes:
            raise SandboxError("permission denied")

    with pytest.raises(SandboxError) as info:
        read_out(Unreadable("f	1	1	a\0"), "/o")  # type: ignore[arg-type]
    assert not isinstance(info.value, OutputRejected)


def test_read_out_gives_up_after_its_budget():
    ticks = iter([0.0, 0.0, 10.0, 20.0, 30.0])
    sb = ListSandbox("f	1	1	a\0f	1	1	b\0", {"/o/a": b"x", "/o/b": b"y"})
    with pytest.raises(OutputRejected, match="too long"):
        read_out(sb, "/o", budget_s=5.0, clock=lambda: next(ticks))  # type: ignore[arg-type]


def test_bind_paths_moves_task_and_out_into_the_writable_dir_only():
    run = "python -m agent --task /task/input.json --out /out"
    assert bind_paths(run, "/work/task", "/work/out") == (
        "python -m agent --task /work/task/input.json --out /work/out"
    )
    untouched = "cat /usr/out/x ./out /output /outer a/task/b /opt/task/c"
    assert bind_paths(untouched, "/work/task", "/work/out") == untouched


def test_agent_command_carries_the_session_token_and_no_real_key():
    cmd = agent_command("run /out", "itr_abc-123", OrchestratorConfig())
    assert "ANTHROPIC_API_KEY='itr_abc-123'" in cmd
    assert 'ANTHROPIC_BASE_URL="${ITERANT_PROXY_URL}"' in cmd
    assert cmd.endswith("run /work/out") and "mkdir -p /work/out" in cmd


def test_budget_blocks_a_run_that_could_overshoot_and_counts_spend():
    spent = [0]
    b = Budget(1000, lambda: spent[0])
    b.check(1000)
    spent[0] = 400
    b.check(600)
    with pytest.raises(BudgetExhausted):
        b.check(601)
    Budget(None, lambda: 10**12).check(10**12)  # unlimited


@pytest.mark.parametrize(
    "bad",
    [
        {"candidates_per_gen": 0},
        {"budget_usd_micros": -1},
        {"run_timeout_s": 0},
        {"max_infra_rate": 0},
    ],
)
def test_config_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        OrchestratorConfig(**bad)


def test_export_commit_has_the_tree_and_no_git_dir(world, tmp_path):
    out = export_commit(world.repo, world.git("rev-parse", "HEAD"), tmp_path / "x")
    assert (out / "agent" / "prompt.txt").read_text() == "v0\n"
    assert not (out / ".git").exists()


def test_export_commit_fails_loudly_on_a_bad_sha(world, tmp_path):
    with pytest.raises(Exception, match="git archive"):
        export_commit(world.repo, "0" * 40, tmp_path / "x")


def test_diff_store_roundtrips_and_detects_corruption(tmp_path):
    store = DiffStore(tmp_path)
    sha = store.put("diff --git a b\n")
    assert store.put("diff --git a b\n") == sha  # idempotent
    assert store.get(sha) == "diff --git a b\n"
    (tmp_path / ".iterant" / "diffs" / f"{sha}.diff").write_text("tampered")
    with pytest.raises(ValueError, match="corrupt"):
        store.get(sha)
    with pytest.raises(FileNotFoundError):
        store.get("1" * 64)


def test_diff_store_knows_the_empty_diff(tmp_path):
    assert DiffStore(tmp_path).get(hashlib.sha256(b"").hexdigest()) == ""


def test_proposer_failure_note_is_clean() -> None:
    from iterant.orchestrator.loop import proposer_failure_note

    exc = RuntimeError(
        "bad\x00thing\x1b[31m at C:\\Users\\me\\AppData\\Temp\\iterant-propose-x\\work "
        "and /tmp/iterant-propose-y/work via http://127.0.0.1:5555/v1/messages\nnext"
    )
    note = proposer_failure_note(exc)
    assert note.startswith("proposer failed: RuntimeError: bad")
    assert "\x00" not in note and "\x1b" not in note and "\n" not in note
    for leaked in ("C:\\", "AppData", "/tmp", "127.0.0.1", "http://"):
        assert leaked not in note
    assert proposer_failure_note(ValueError("")) == "proposer failed: ValueError"


def test_proposer_failure_note_is_fast_on_huge_unbroken_text() -> None:
    import time

    from iterant.orchestrator.loop import proposer_failure_note

    start = time.perf_counter()
    note = proposer_failure_note(RuntimeError("a" * 50_000))
    assert time.perf_counter() - start < 1.0
    assert note.startswith("proposer failed: RuntimeError: aaa") and len(note) < 300
