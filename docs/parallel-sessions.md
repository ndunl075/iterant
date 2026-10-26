# Parallel sessions: rules for workers

Read all of this before you start. Your brief says what to build; this says how to work alongside other workers who are building at the same time on the same machine.

## Stack and commands
- Python 3.12, uv, pytest, ruff, pyright (strict on `iterant/`).
- Required reading: ARCHITECTURE.md §1 (invariants; a PR that breaks one is rejected) and the §5 text for your own module.
- Every PR must pass these, locally and in CI:
  - `uv run pytest -q`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run pyright`

## Shared files (nobody edits them in a wave)
- `iterant/contracts.py` holds the shared types. Import them; don't redefine them. The architect owns it. If a contract is wrong or missing, say so in your PR's open questions instead of editing it.
- `pyproject.toml` and `uv.lock`: don't edit them. Need a dependency? Put the request and the reason the stdlib won't do in your PR's open questions; the architect adds it.
- Tests go in `tests/<your module>/`. Never create or edit a root `tests/conftest.py`. The pytest markers `podman` and `firecracker` are already registered.
- Fake LLM: tests never call a live model. The recorded-fixture fake LLM upstream is owned by the proxy worker (`iterant/proxy/`). Until it lands, put any fake you need under your own `tests/<module>/` and flag it in open questions; don't build a second shared one.

## Tests, network and money
- $0 budget: never call a paid API or service.
- No internet from tests. A stub server on localhost (bind port 0, let the OS pick) is fine.
- Anything that needs a container runtime, an image pull or `/dev/kvm` is a local-only test: mark it `podman` or `firecracker` and make it skip cleanly when the runtime is missing. Expect those to skip on Windows; don't spend attempts making them run there.

## Working
- Work in your own worktree and branch. Touch only the paths your brief says you own, plus `tests/<your module>/`.
- After 2 failed attempts at the same problem, stop and report what you tried, what you saw, and your best guess.
- Run long builds and tests in the background. Redirect long output to a file and read only the summary and failures.
- Before deleting a folder, remove any junctions or symlinks inside it first.
- Machine resources are shared with other workers. Prefix every container, image, volume and temp-dir name with your branch name, bind port 0 for servers, and clean up in a fixture teardown. Never run global cleanup (`podman system prune`, `podman system reset`, deleting other workers' containers or images).
- Run headless only: nothing that opens a window.

## PRs
- Update an open PR branch by merging main into it. Never rebase or force-push.
- PR description, in this order:
  1. What changed.
  2. How it was tested: exact commands and their results, including which `podman`/`firecracker` tests skipped and why.
  3. Files touched.
  4. Open questions.
- No self-assessment ("this is safe", "ready to merge"). The reviewer judges that.
- Stop when the PR is open and CI is green. Workers never merge.

## Local-only checks
CI runs the four commands on Linux and skips the marked tests. The architect runs these on the exact head before merging:
- The four commands on Windows (Nico's machine).
- `podman`-marked tests on a machine with rootless Podman; `firecracker`-marked tests on Linux with `/dev/kvm`.

## Review gate
Every PR gets an independent zero-context review before merge (Tier 1: Sonnet; Tier 2: Opus, for sandbox, proxy, target, gate, grader, archive, orchestrator). Findings come back to you in one message. For each: verify it in the code, fix the real ones, and give a one-line reason for each one you reject. Push the fixes to the same branch; the architect sends it back for re-review.
