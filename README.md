# Iterant

**Iterant: your agent improves itself.**

Point Iterant at your AI agent and a set of tasks with automatic checks. It proposes changes to your agent's prompts, tools and logic, tests each one in disposable sandboxes, and keeps only the changes that beat the current version with statistical evidence. Iterant never merges into your code: it commits to an `iterant/champion` branch and writes an evidence report, and you decide whether to merge.

## Status

**Pre-release, v0.** The modules and the CLI are built and unit-tested against fakes. **No run with a real model has been done yet.** A one-off `iterant estimate` run on the reference agent (a throwaway GitHub Actions job, not part of CI) completed end to end on rootless Podman against a stub model server: image build, agent sandbox, proxy, grader, report. `aa-test`, `run` and the `claude` profile have not been run. Expect rough edges. The `claude` profile (paid) ships untested by us.

## How it differs

- **Statistically gated.** Each candidate is compared with the current champion on the same tasks in the same batch (paired bootstrap). It must pass a `select` split and then a held-out `holdout` split. `iterant aa-test` resamples the champion against itself and checks that the rule falsely accepts at most 5% of the time.
- **Sandboxed, with a trusted base.** Candidates can edit only the files `iterant.toml` marks as the genome. Sandboxes run as non-root with default-deny networking (Podman or Docker containers, or Firecracker microVMs on Linux with `/dev/kvm`). Hidden checks run only in a separate grader sandbox, and the API key never enters a sandbox. See [docs/threat-model.md](docs/threat-model.md) for what this stops and what it does not.
- **Works on your agent.** Any language, any repo: you supply a Containerfile, a run command and a task pack.
- **Free by default.** The `free` profile runs every role on one local model through an OpenAI-compatible server such as Ollama, at $0. Expect weaker proposals than a hosted model would give.

Design details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- A container runtime for sandboxes: rootless Podman (the default; development use), Docker with `--unsafe-container`, or `--backend firecracker` (microVMs, the release sandbox; Linux with `/dev/kvm` and the pinned assets from `iterant/sandbox/firecracker/fetch-assets.sh`).
- A local OpenAI-compatible model server for the `free` profile. The default endpoint is `http://127.0.0.1:11434/v1` (Ollama); change it with `--openai-base-url`. `iterant init` defaults to the model name `llama3.1:8b`.

## Quick start (reference agent, free profile)

The reference agent in [`examples/reference-agent/`](examples/reference-agent/) is a small question-answering agent. It ships with a free-profile `iterant.toml` (`llama3.1:8b`) and a 57-task reference pack in `evals/reference-pack/`, and it reads its model from `iterant.toml`, so no edits are needed. Run the steps from a checkout of this repo.

```sh
# 1. Install Iterant (from the Iterant checkout; run every `iterant` command from the Iterant checkout, not from ../my-agent).
uv sync

# 2. Copy the agent outside this repo; `iterant run` needs a git repo with a commit.
cp -r examples/reference-agent ../my-agent
(cd ../my-agent && git init && git add -A && git commit -m baseline)

# 3. Pull the model (Ollama must be running).
ollama pull llama3.1:8b

# 4. Baseline pass rate, variance, smallest detectable gain and cost
#    (the reference pack has 20 select tasks).
uv run iterant estimate --backend podman --split select --tasks 20 --reps 2 ../my-agent

# 5. A/A test: the champion against itself must be falsely accepted at most 5% of the time.
uv run iterant aa-test --backend podman ../my-agent

# 6. Run one generation.
uv run iterant run --backend podman --generations 1 ../my-agent

# 7. Inspect the archive and write the static HTML evidence report.
uv run iterant status ../my-agent
uv run iterant report ../my-agent        # writes ../my-agent/.iterant/report/
```

Notes:

- Every command takes an optional `repo` as its last argument (default: the current directory).
- `iterant run` refuses to start unless a matching `iterant estimate` (on the `select` split, with at least the select task and rep counts, or every select task in your pack if it has fewer than 80) and a passing `iterant aa-test` exist for this pack, profile, model and config. `--skip-calibration` overrides this (not recommended).
- With the defaults (`select` 80 tasks x 2 reps, `holdout` 40 x 2, `--delta-min 0.05`), a pack with fewer tasks gets smaller splits and cannot detect a 5-point gain. The reference pack is small (20 select, 11 holdout): it is a demo, not a benchmark. `estimate` reports the smallest gain it can detect. Flags such as `--select-tasks`, `--select-reps`, `--holdout-tasks`, `--holdout-reps`, `--smoke-tasks` and `--delta-min` change them. A calibration record only counts if these flags match the ones you pass to `run`.
- Use `--backend docker --unsafe-container` for Docker. The Docker daemon is root-equivalent, so use it on dev machines only.
- By default the proxy binds an address that sandboxes can reach through the container runtime. That address is reachable from your local network; requests are token-gated. Use `--proxy-bind` to change it.
- Run `uv run iterant <command> --help` for every flag.

## Not built yet

Holdout rotation, archive parent sampling (v1), eval bootstrap from logs (v1), and research mode `iterant meta` (v1).

## More

- [docs/threat-model.md](docs/threat-model.md): what the trusted base is, what each defense covers, known holes.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to build and test.

License: Apache-2.0
