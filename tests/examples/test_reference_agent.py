from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from iterant.target import genome_for, load_config

AGENT_DIR = Path(__file__).resolve().parents[2] / "examples" / "reference-agent"

EDITABLE = {
    "agent/loop.py",
    "prompts/system.md",
    "tools/__init__.py",
    "tools/calculate.py",
    "tools/lookup.py",
}


def _tracked_files() -> set[str]:
    return {
        p.relative_to(AGENT_DIR).as_posix()
        for p in AGENT_DIR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def test_toml_parses() -> None:
    cfg = load_config(AGENT_DIR / "iterant.toml")
    assert cfg.target.model.name == "llama3.1:8b"  # free profile, the default
    assert cfg.target.model.effort == "medium"
    assert cfg.target.model.max_tokens_per_run == 200_000
    assert (AGENT_DIR / cfg.target.image).is_file()


def test_genome_matches_exactly_the_intended_files() -> None:
    genome = genome_for(AGENT_DIR, load_config(AGENT_DIR / "iterant.toml"))
    editable = {f for f in _tracked_files() if genome.is_editable(f)}
    assert editable == EDITABLE
    for frozen in ["iterant.toml", "Containerfile", "agent/__main__.py", "kb/facts.json"]:
        assert not genome.is_editable(frozen)


def test_loop_has_no_hardcoded_model() -> None:
    src = (AGENT_DIR / "agent" / "loop.py").read_text(encoding="utf-8")
    assert "claude-" not in src and "llama" not in src


# --- the loop against a fake client ----------------------------------------------


@pytest.fixture
def loop() -> Iterator[ModuleType]:
    sys.path.insert(0, str(AGENT_DIR))
    try:
        yield importlib.import_module("agent.loop")
    finally:
        sys.path.remove(str(AGENT_DIR))
        for name in [
            m for m in sys.modules if m in ("agent", "tools") or m.startswith(("agent.", "tools."))
        ]:
            del sys.modules[name]


def _text(t: str) -> Any:
    return SimpleNamespace(type="text", text=t)


def _tool(id_: str, name: str, **args: Any) -> Any:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def _resp(stop: str, *blocks: Any) -> Any:
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClient:
    def __init__(self, *responses: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._responses.pop(0)


def test_model_and_effort_come_from_iterant_toml(loop: ModuleType, tmp_path: Path) -> None:
    cfg = load_config(AGENT_DIR / "iterant.toml")
    assert (loop.MODEL, loop.EFFORT) == (cfg.target.model.name, cfg.target.model.effort)
    other = tmp_path / "iterant.toml"
    other.write_text(
        "\n".join(
            ["[target.model]", "name = 'qwen2.5:7b'", "effort = 'high'", "max_tokens_per_run = 1"]
        ),
        encoding="utf-8",
    )
    assert loop.frozen_model(other) == ("qwen2.5:7b", "high")


def test_loop_sends_the_frozen_model(loop: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """What the loop sends is what the proxy will accept: the model frozen in ``iterant.toml``,
    even when the environment tries to say otherwise."""
    cfg = load_config(AGENT_DIR / "iterant.toml")
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    client = FakeClient(_resp("end_turn", _text("ok")))
    loop.run_task({"question": "q"}, client)
    (call,) = client.calls
    assert call["model"] == cfg.target.model.name
    assert call["extra_body"] == {"output_config": {"effort": cfg.target.model.effort}}


def test_loop_runs_tools_then_answers(loop: ModuleType) -> None:
    client = FakeClient(
        _resp("tool_use", _text("checking"), _tool("t1", "lookup_fact", key="widget_price_usd")),
        _resp("tool_use", _tool("t2", "calculate", expression="19.99 * 3")),
        _resp("end_turn", _text("Three widgets cost 59.97.")),
    )
    result = loop.run_task({"question": "Price of 3 widgets?"}, client)
    assert result.answer == "Three widgets cost 59.97."
    assert [e["output"] for e in result.transcript if e["role"] == "tool"] == ["19.99", "59.97"]

    first, second, third = client.calls
    assert first["model"] == "llama3.1:8b"
    assert first["extra_body"] == {"output_config": {"effort": "medium"}}
    assert "temperature" not in first and "thinking" not in first
    assert first["messages"] == [{"role": "user", "content": "Price of 3 widgets?"}]
    assert {t["name"] for t in first["tools"]} == {"calculate", "lookup_fact"}
    assert "lookup_fact" in first["system"]
    # tool results go back as a user turn keyed by tool_use_id
    assert second["messages"][-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "19.99"}],
    }
    assert third["messages"][-1]["content"][0]["tool_use_id"] == "t2"


def test_tool_errors_are_returned_to_the_model(loop: ModuleType) -> None:
    client = FakeClient(
        _resp("tool_use", _tool("t1", "calculate", expression="__import__('os').system('x')")),
        _resp("tool_use", _tool("t2", "nope")),
        _resp("end_turn", _text("done")),
    )
    result = loop.run_task({"question": "q"}, client)
    outputs = [e["output"] for e in result.transcript if e["role"] == "tool"]
    assert outputs[0].startswith("error:")
    assert outputs[1] == "error: unknown tool 'nope'"
    assert result.answer == "done"


def test_refusal_raises(loop: ModuleType) -> None:
    with pytest.raises(RuntimeError, match="refused"):
        loop.run_task({"question": "q"}, FakeClient(_resp("refusal", _text(""))))


def test_turn_limit_returns_empty_answer(loop: ModuleType) -> None:
    n = loop.MAX_TURNS
    client = FakeClient(
        *[_resp("tool_use", _tool(f"t{i}", "calculate", expression="1+1")) for i in range(n)]
    )
    assert loop.run_task({"question": "q"}, client).answer == ""
    assert len(client.calls) == n


def test_main_writes_outputs(
    loop: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = importlib.import_module("agent.__main__")

    client = FakeClient(_resp("end_turn", _text("42")))
    monkeypatch.setattr(entry, "make_client", lambda: client)
    task = tmp_path / "input.json"
    task.write_text(json.dumps({"question": "q"}), encoding="utf-8")
    out = tmp_path / "out"
    assert entry.main(["--task", str(task), "--out", str(out)]) == 0
    assert json.loads((out / "answer.json").read_text(encoding="utf-8")) == {"answer": "42"}
    assert (out / "transcript.json").is_file()


def test_calculate_rejects_non_arithmetic(loop: ModuleType) -> None:
    calculate = importlib.import_module("tools.calculate").calculate

    assert calculate("2 ** 3 + 1") == 9
    assert calculate("-(4 % 3)") == -1
    for bad in ["abs(1)", "a + 1", "2 ** 1000", "'x' * 3", "[1]"]:
        with pytest.raises((ValueError, SyntaxError)):
            calculate(bad)


def test_lookup_fact(loop: ModuleType) -> None:
    lookup_fact = importlib.import_module("tools.lookup").lookup_fact

    assert lookup_fact("founded") == "2024"
    assert "known keys" in lookup_fact("nope")


def test_make_client_lazy_import(loop: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the loop never imports the SDK; make_client does, lazily."""
    seen: dict[str, Any] = {}
    fake = ModuleType("anthropic")

    def anthropic_ctor(**kw: Any) -> str:
        seen.update(kw)
        return "client"

    fake.Anthropic = anthropic_ctor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert loop.make_client() == "client"
    assert seen == {"api_key": "iterant-sandbox"}


def test_containerfile_makes_the_agent_importable_from_any_directory() -> None:
    """The sandbox runs the agent with its own working directory (not WORKDIR), so
    ``python -m agent`` only works if the app directory is on PYTHONPATH."""
    text = (AGENT_DIR / "Containerfile").read_text(encoding="utf-8")
    assert "ENV PYTHONPATH=/app" in text
