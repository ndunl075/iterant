# Contributing

Read [ARCHITECTURE.md](ARCHITECTURE.md) section 1 (invariants) first. A PR that breaks one is rejected. Rules for parallel worker sessions are in [docs/parallel-sessions.md](docs/parallel-sessions.md).

## Checks

All four must pass locally and in CI:

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

CI runs them on Linux with `-m "not podman and not docker and not firecracker"`.

## Rules

- **$0.** Never call a paid API or use a real key.
- **Tests never call a live model.** Use the recorded-fixture fake LLM. No internet from tests; a stub server on localhost (port 0) is fine.
- **Pyright is strict on `iterant/`** and basic on `tests/`.
- **No new dependencies** without asking first.
- `iterant/contracts.py`, `pyproject.toml` and `uv.lock` belong to the maintainer. Propose changes in the PR description instead of editing them.

## Tests that need a runtime (local only)

These are marked and skip when the runtime is missing. Run them by hand:

```sh
uv run pytest -m podman -q -rs        # rootless Podman
uv run pytest -m docker -q -rs        # Docker
uv run pytest -m firecracker -q -rs   # Linux with /dev/kvm
```

Red-team suite:

```sh
uv run pytest tests/redteam -q
uv run pytest tests/redteam/test_row8_sandbox_escape.py -m "podman or docker" -q -rs   # live sandbox cases
```

Prefix container, image and temp-dir names with your branch slug, and never run global cleanup such as `podman system prune`.

## Review gate

Every PR gets an independent zero-context review before merge. Fix the real findings on the same branch by merging main in; do not rebase or force-push.
