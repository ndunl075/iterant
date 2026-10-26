"""The agent loop: ask the model, run the tools it requests, repeat until it answers.

The client is injected so tests can pass a fake. ``make_client`` imports the
Anthropic SDK lazily; inside the sandbox the SDK reads ``ANTHROPIC_BASE_URL``,
which points at the Iterant proxy. The proxy holds the real key and enforces the
model, effort and spend caps from ``iterant.toml``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from tools import SCHEMAS, run_tool

MODEL = "claude-haiku-5-5"
EFFORT = "medium"
MAX_TURNS = 8
MAX_TOKENS = 2048
PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "system.md"


@dataclass
class Result:
    answer: str
    transcript: list[dict] = field(default_factory=list)


def make_client():
    import anthropic  # lazy: only needed inside the container

    # The key is a placeholder; the proxy swaps in the real one.
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "iterant-sandbox"))


def _text(blocks) -> str:
    return "".join(b.text for b in blocks if b.type == "text").strip()


def run_task(task: dict, client) -> Result:
    system = PROMPT.read_text(encoding="utf-8")
    messages: list[dict] = [{"role": "user", "content": task["question"]}]
    transcript: list[dict] = []
    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=SCHEMAS,
            messages=messages,
            extra_body={"output_config": {"effort": EFFORT}},
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        transcript.append({"role": "assistant", "text": _text(resp.content)})
        calls = [b for b in resp.content if b.type == "tool_use"]
        if not calls:
            return Result(_text(resp.content), transcript)
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for call in calls:
            output = run_tool(call.name, call.input)
            transcript.append(
                {"role": "tool", "name": call.name, "input": call.input, "output": output}
            )
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": output})
        messages.append({"role": "user", "content": results})
    return Result("", transcript)  # ran out of turns
