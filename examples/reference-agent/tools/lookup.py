"""Look up a fact in the frozen knowledge base (kb/facts.json)."""

import json
from pathlib import Path

SCHEMA = {
    "name": "lookup_fact",
    "description": "Look up a fact about the demo company by key, e.g. 'founded'.",
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
}

KB = Path(__file__).resolve().parent.parent / "kb" / "facts.json"


def lookup_fact(key: str) -> str:
    facts = json.loads(KB.read_text(encoding="utf-8"))
    if key not in facts:
        return f"unknown key {key!r}; known keys: {', '.join(sorted(facts))}"
    return facts[key]
