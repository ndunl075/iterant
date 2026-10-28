"""Proposer hardening from the first live run: editable list, actionable refusals, nudges."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

import pytest

from iterant.analyst import HttpMessagesClient, LLMError
from iterant.proposer import EditingProposer
from iterant.proposer.agent import NUDGE
from iterant.proposer.workspace import ToolError, Workspace
from iterant.proxy import FakeLLM, anthropic_message
from iterant.proxy.jsonutil import Obj
from iterant.target.genome import genome_for

OPUS = "claude-opus-5-5"


def use(name: str, **args: object) -> Obj:
    return anthropic_message("", tool_use={"id": f"tu_{name}", "name": name, "input": args})


def make(fake: FakeLLM, config, **kw) -> EditingProposer:  # type: ignore[no-untyped-def]
    return EditingProposer(
        HttpMessagesClient(fake.base_url, "tok"), config, model=OPUS, effort="high", **kw
    )


def first_prompt(fake: FakeLLM) -> str:
    content = fake.requests[0].body["messages"][0]["content"]
    assert isinstance(content, str)
    return content


def test_prompt_lists_editable_files_only(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("no change"))
        make(fake, config).propose("seed", repo, "d", [])
        prompt = first_prompt(fake)
    assert "- prompts/system.md" in prompt and "- agent/loop.py" in prompt
    assert "agent/other.py" not in prompt  # outside the genome
    assert "iterant.toml" not in prompt and "secret.txt" not in prompt  # protected / pack
    assert "prompts/**" in prompt


def test_prompt_listing_is_capped(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    for i in range(200):
        (repo / "prompts" / f"p{i:03}.md").write_text("x\n")
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("no change"))
        make(fake, config).propose("seed", repo, "d", [])
        prompt = first_prompt(fake)
    assert prompt.count("- prompts/") <= 40


def test_refusal_names_editable_files(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    ws = Workspace(repo, genome_for(repo, config), hidden=("evals/pack",))
    with pytest.raises(ToolError) as w:
        ws.write("agent/other.py", "x")
    assert "editable files:" in str(w.value) and "agent/loop.py" in str(w.value)
    with pytest.raises(ToolError) as r:
        ws.read("agent/guessed.py")
    assert "agent/loop.py" in str(r.value)
    with pytest.raises(ToolError) as h:
        ws.read("evals/pack/secret.txt")
    assert "list_files" in str(h.value) and "HIDDEN" not in str(h.value)


def test_prose_reply_gets_nudged_then_edits(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(lambda b: len(b["messages"]) == 1, anthropic_message("I would change it"))
        fake.add(lambda b: len(b["messages"]) == 3, use("read_file", path="agent/loop.py"))
        fake.add(
            lambda b: len(b["messages"]) == 5,
            use("write_file", path="agent/loop.py", content="def run():\n    return 2\n"),
        )
        fake.add(lambda b: len(b["messages"]) == 7, use("finish", hypothesis="h"))
        proposal = make(fake, config).propose("seed", repo, "d", [])
        assert fake.requests[1].body["messages"][2] == {"role": "user", "content": NUDGE}
    assert proposal is not None and "return 2" in proposal.diff


def test_at_most_two_nudges_then_no_diff(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("prose again"))
        assert make(fake, config).propose("seed", repo, "d", []) is None
        assert len(fake.requests) == 3  # first reply plus two nudged retries


def test_nudges_count_within_max_turns(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(None, anthropic_message("prose"))
        assert make(fake, config, max_turns=2).propose("seed", repo, "d", []) is None
        assert len(fake.requests) == 2


def test_no_nudge_after_a_write(repo: Path, config) -> None:  # type: ignore[no-untyped-def]
    with FakeLLM() as fake:
        fake.add(lambda b: len(b["messages"]) == 1, use("read_file", path="agent/loop.py"))
        fake.add(
            lambda b: len(b["messages"]) == 3,
            use("write_file", path="agent/loop.py", content="y = 1\n"),
        )
        fake.add(lambda b: len(b["messages"]) == 5, anthropic_message("done"))
        proposal = make(fake, config).propose("seed", repo, "d", [])
        assert len(fake.requests) == 3
    assert proposal is not None


def test_timeout_message_names_the_limit() -> None:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    stop = threading.Event()
    conns: list[socket.socket] = []

    def hold() -> None:
        c, _ = srv.accept()
        conns.append(c)
        stop.wait(5)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    try:
        client = HttpMessagesClient(f"http://127.0.0.1:{srv.getsockname()[1]}", "tok", timeout=0.3)
        with pytest.raises(LLMError, match="timed out after 0.3s"):
            client.create({"x": 1})
    finally:
        stop.set()
        t.join()
        for c in conns:
            c.close()
        srv.close()
