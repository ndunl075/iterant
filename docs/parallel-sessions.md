# Parallel sessions: rules for workers

Read all of this before you start. Your brief says what to build; this says how to work alongside other workers.

## Stack and commands
- Python 3.12, uv, pytest, ruff, pyright (strict on `iterant/`). Shared types live in `iterant/contracts.py`; import them, don't redefine them. If a contract is wrong or missing, say so in your PR's open questions instead of editing it (it has one owner per wave).
- Every PR must pass, locally and in CI:
  - `uv run pytest -q`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run pyright`
- Tests never call a live model or the network. Use the recorded-fixture fake LLM. The project has a $0 budget: never call a paid API or service.
- Read ARCHITECTURE.md §1 before writing code. A PR that breaks an invariant is rejected.

## Working
- Work in your own worktree and branch. Touch only the paths your brief says you own. If you need a change elsewhere, stop and ask in your PR's open questions.
- After 2 failed attempts at the same problem, stop and report what you tried, what you saw, and your best guess.
- Run long builds and tests in the background. Redirect long output to a file and read only the summary and failures.
- Before deleting a folder, remove any junctions or symlinks inside it first.
- Exclusive machine resources (container runtime state, fixed ports, `/dev/kvm`): use a unique name or port per test, and clean up in a `finally`/fixture teardown.
- Don't add a dependency you don't need. A new dependency goes in your PR's open questions with the reason the stdlib won't do.

## PRs
- Update an open PR branch by merging main into it. Never rebase or force-push.
- PR description, in this order:
  1. What changed.
  2. How it was tested: exact commands and their results.
  3. Files touched.
  4. Open questions.
- No self-assessment ("this is safe", "ready to merge"). The reviewer judges that.
- Stop when the PR is open and CI is green. Workers never merge.

## Local-only checks
CI runs the four commands above on Linux. These run only on a developer machine, and the architect runs them on the exact head before merging:
- The four commands on Windows (Nico's machine).
- Sandbox backends: tests marked `podman` need rootless Podman; tests marked `firecracker` need Linux with `/dev/kvm`. Mark such tests with the pytest marker and skip them cleanly when the runtime is missing.

## Review gate
Every PR gets an independent zero-context review before merge (Tier 1: Sonnet; Tier 2: Opus, for sandbox, proxy, target, gate, grader, archive, orchestrator). Findings come back to you in one message. For each: verify it in the code, fix the real ones, and give a one-line reason for each one you reject. Push the fixes to the same branch; the architect sends it back for re-review.
