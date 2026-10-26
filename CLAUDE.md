# Iterant

Iterant improves a user's AI agent by proposing changes to its editable files, testing them in disposable sandboxes and keeping only statistically proven wins.

## Before changing code
Read ARCHITECTURE.md §1 (invariants) first. Breaking one means the PR is rejected.

## Commands (all must pass locally and in CI)
```
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Rules
- Tests never call a live model. Use the recorded-fixture fake LLM.
- $0 budget: never call paid APIs or use real keys.
- `iterant/` is pyright strict; `tests/` is basic.
- No new dependencies without asking first.
- Worker rules: see docs/parallel-sessions.md.
