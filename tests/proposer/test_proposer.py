from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from iterant.analyst import HttpMessagesClient
from iterant.contracts import Candidate, Proposer, Stage, Status, TriedCandidate
from iterant.proposer import EditingProposer
from iterant.proxy import FakeLLM, anthropic_message
from iterant.proxy.jsonutil import Obj, arr, obj

OPUS = "claude-opus-5-5"


def use(name: str, **args: object) -> Obj:
    return anthropic_message("", tool_use={"id": f"tu_{name}", "name": name, "input": args})


def turn(n: int):
    """Match the request that carries ``n`` messages (one per loop turn)."""
    return lambda body: len(body.get("messages", [])) == n


def script(fake: FakeLLM, *replies: Obj) -> None:
    for i, reply in enumerate(replies):
        fake.add(turn(1 + 2 * i), reply)


def make(fake: FakeLLM, config, **kw) -> EditingProposer:  # type: ignore[no-untyped-def]
    return EditingProposer(
        HttpMessagesClient(fake.base_url, "tok"), config, model=OPUS, effort="high", **kw
    )


def tried(i: int, hypothesis: str, diff: str, status: Status = Status.REJECTED) -> TriedCandidate:
    cand = Candidate(f"cand-{i}", None, "0" * 64, hypothesis, Stage.SELECT, status, 0)
    return TriedCandidate(cand, diff)


def test_implements_the_protocol(config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        assert isinstance(make(fake, config), Proposer)


def test_edit_inside_genome_returns_diff_and_hypothesis(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        script(
            fake,
            use("read_file", path="prompts/system.md"),
            use("write_file", path="prompts/system.md", content="You are helpful.\nBe terse.\n"),
            use("write_file", path="prompts/new.md", content="hello\n"),
            use("finish", hypothesis="Terse replies\nreduce rambling"),
        )
        proposal = make(fake, config).propose("seed", repo, "digest", [])
    assert proposal is not None
    assert proposal.hypothesis == "Terse replies reduce rambling"
    assert "diff --git a/prompts/system.md b/prompts/system.md" in proposal.diff
    assert "+Be terse." in proposal.diff
    assert "new file mode" in proposal.diff and "--- /dev/null" in proposal.diff
    assert (repo / "prompts" / "system.md").read_text() == "You are helpful.\nBe terse.\n"
    # tools are sent with the proposer's pinned model and effort
    first = fake.requests[0].body
    assert first["model"] == OPUS and first["output_config"] == {"effort": "high"}
    assert {t["name"] for t in first["tools"]} == {
        "list_files",
        "read_file",
        "write_file",
        "finish",
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_diff_applies_with_git(repo: Path, config, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    pristine = tmp_path / "pristine"
    shutil.copytree(repo, pristine)
    with FakeLLM() as fake:
        script(
            fake,
            use("write_file", path="agent/loop.py", content="def run():\n    return 2"),
            use("write_file", path="prompts/new.md", content="hello\n"),
            use("finish", hypothesis="change"),
        )
        proposal = make(fake, config).propose("seed", repo, "d", [])
    assert proposal is not None
    patch = tmp_path / "p.diff"
    patch.write_bytes(proposal.diff.encode())
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=pristine, check=True)


def test_edit_outside_genome_is_refused_and_nothing_written(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        script(
            fake,
            use("write_file", path="agent/other.py", content="X = 2\n"),
            use("write_file", path="iterant.toml", content="evil\n"),
            use("write_file", path="evals/pack/secret.txt", content="x\n"),
            use("write_file", path="../escape.txt", content="x\n"),
            use("write_file", path=".git/config", content="x\n"),
            use("finish", hypothesis="try forbidden edits"),
        )
        proposal = make(fake, config).propose("seed", repo, "d", [])
    assert proposal is None
    assert (repo / "agent" / "other.py").read_text() == "X = 1\n"
    assert (repo / "iterant.toml").read_text() == "# config\n"
    assert (repo / "evals" / "pack" / "secret.txt").read_text() == "HIDDEN-CHECK-CONTENT\n"
    assert not (repo.parent / "escape.txt").exists()
    last = fake.requests[-1].body["messages"][-1]["content"]
    assert all(obj(r) and obj(r).get("is_error") for r in last)  # type: ignore[union-attr]


def test_refused_edit_does_not_block_a_valid_one(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        script(
            fake,
            use("write_file", path="agent/other.py", content="X = 2\n"),
            use("write_file", path="agent/loop.py", content="def run():\n    return 3\n"),
            use("finish", hypothesis="loop change"),
        )
        proposal = make(fake, config).propose("seed", repo, "d", [])
    assert proposal is not None
    assert "agent/loop.py" in proposal.diff and "agent/other.py" not in proposal.diff


def test_none_when_there_is_no_diff(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        script(fake, use("finish", hypothesis="nothing to change"))
        assert make(fake, config).propose("seed", repo, "d", []) is None
    with FakeLLM() as fake:
        script(fake, anthropic_message("I would not change anything."))
        assert make(fake, config).propose("seed", repo, "d", []) is None
    with FakeLLM() as fake:  # rewriting a file with identical content is no diff either
        script(
            fake,
            use("write_file", path="prompts/system.md", content="You are helpful.\n"),
            use("finish", hypothesis="noop"),
        )
        assert make(fake, config).propose("seed", repo, "d", []) is None


def test_history_and_digest_reach_the_prompt(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    history = [
        tried(1, "Add a calculator hint", "-old line\n+NEW-HINT-LINE\n"),
        tried(2, "Shorten the system prompt", "-x\n+y\n", Status.ERROR),
    ]
    with FakeLLM() as fake:
        script(fake, use("finish", hypothesis="noop"))
        make(fake, config).propose("seed", repo, "DIGEST-MARKER: t1 failed", history)
    first = fake.requests[0].body["messages"][0]["content"]
    assert "DIGEST-MARKER: t1 failed" in first
    for text in ("cand-1", "Add a calculator hint", "NEW-HINT-LINE", "rejected at select"):
        assert text in first
    assert "Shorten the system prompt" in first and "error at select" in first
    assert "do not repeat" in first.lower()


def test_pack_and_git_are_not_readable_or_listed(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref")
    with FakeLLM() as fake:
        script(
            fake,
            use("read_file", path="evals/pack/secret.txt"),
            use("read_file", path="EVALS/pack/secret.txt"),
            use("read_file", path=".git/HEAD"),
            use("list_files"),
            use("finish", hypothesis="x"),
        )
        make(fake, config).propose("seed", repo, "d", [])
    seen = repr([r.body["messages"] for r in fake.requests])
    assert "HIDDEN-CHECK-CONTENT" not in seen
    listing = fake.requests[-1].body["messages"][-1]["content"][0]["content"]
    assert "agent/loop.py" in listing and "prompts/system.md" in listing
    assert "secret.txt" not in listing and ".git" not in listing


def test_max_turns_bounds_the_loop(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(None, use("list_files"))
        make(fake, config, max_turns=3).propose("seed", repo, "d", [])
    assert len(fake.requests) == 3


def test_refusal_yields_none(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("no", stop_reason="refusal"))
        assert make(fake, config).propose("seed", repo, "d", []) is None


def test_own_prompt_and_genome_files_exist() -> None:
    here = Path(__import__("iterant.proposer", fromlist=["x"]).__file__).parent
    assert (here / "prompt.md").read_text(encoding="utf-8").strip()
    assert 'genome = ["prompt.md"]' in (here / "genome.toml").read_text(encoding="utf-8")
    assert arr([1]) is not None
