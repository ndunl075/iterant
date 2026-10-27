# Iterant

**Iterant: your agent improves itself.**

Point Iterant at your AI agent and a set of tasks with automatic checks. It proposes changes to your agent's prompts, tools and logic, tests each one in disposable sandboxes, and keeps only the changes that beat the current version with statistical evidence. Iterant never merges into your code: it commits to an `iterant/champion` branch and writes an evidence report, and you decide whether to merge.

## Status

**Pre-release, v0.** The modules and the CLI are built and unit-tested against fakes. **No live end-to-end run has been done yet**: not on a real container runtime with a real local model, and not on the `claude` profile. Treat every step below as unverified end to end, and expect rough edges. The `claude` profile (paid) ships untested by us.

## How it differs

- **Statistically gated.** Each candidate is compared with the current champion on the same tasks in the same batch (paired bootstrap). It must pass a `select` split and then a held-out `holdout` split. `iterant aa-test` checks that the rule does not accept a champion compared with itself.
- **Hardened sandbox and trusted base.** Candidates can edit only the files `iterant.toml` marks as the genome. Hidden checks run only in a separate grader sandbox, and the API key never enters a sandbox. See [docs/threat-model.md](docs/threat-model.md) for what this stops and what it does not.
- **Works on your agent.** Any language, any repo: you supply a Containerfile, a run command and a task pack.
- **Free by default.** The `free` profile runs every role on one local model through an OpenAI-compatible server such as Ollama, at $0. Expect weaker proposals than a hosted model would give.

Design details: [ARCHITECTURE.md](ARCHITECTURE.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- A container runtime for sandboxes: rootless Podman (recommended), or Docker with `--unsafe-container`. The Firecracker microVM backend exists (Linux with `/dev/kvm`) but the CLI does not select it yet: `--backend` accepts only `podman` and `docker`.
- A local OpenAI-compatible model server for the `free` profile. The default endpoint is `http://127.0.0.1:11434/v1` (Ollama); change it with `--openai-base-url`. `iterant init` defaults to the model name `llama3.1:8b`.

## Quick start (reference agent, free profile)

The reference agent in [`examples/reference-agent/`](examples/reference-agent/) is a small question-answering agent. Run the steps from a checkout of this repo.

Two gaps to know about before you start:

1. **No task pack ships for the reference agent.** Its `iterant.toml` points at `evals/reference-pack`, which does not exist. The only pack in the repo is [`packs/cruxeval`](packs/cruxeval/README.md), which expects a different output format (`answer.txt`, a Python literal), so it does not fit the reference agent. You need a pack of your own (`pack.toml`, `splits.json`, `tasks/<id>/`; see ARCHITECTURE.md section 2).
2. **The reference agent hardcodes `MODEL = "claude-haiku-5-5"`** in `agent/loop.py`, and the proxy rejects any model that differs from the one frozen in `iterant.toml`. For the free profile, set `MODEL` to your local model name (for example `llama3.1:8b`) before the first run.

```sh
# 1. Install Iterant (from the Iterant checkout; run every `iterant` command from there).
uv sync

# 2. Copy the agent outside this repo; `iterant run` needs a git repo with a commit.
cp -r examples/reference-agent ../my-agent
(cd ../my-agent && git init && git add -A && git commit -m baseline)

# 3. Write iterant.toml for the free profile (overwrites the example's file).
uv run iterant init --force --profile free --model llama3.1:8b \
  --genome 'prompts/**' --genome 'tools/**' --genome agent/loop.py \
  --image Containerfile --pack evals/reference-pack ../my-agent

# 4. Add your pack at ../my-agent/evals/reference-pack, set MODEL in agent/loop.py, commit both.

# 5. Baseline pass rate, variance, smallest detectable gain and cost.
uv run iterant estimate --backend podman ../my-agent

# 6. A/A test: the champion against itself must not be accepted.
uv run iterant aa-test --backend podman ../my-agent

# 7. Run one generation.
uv run iterant run --backend podman --generations 1 ../my-agent

# 8. Inspect the archive and write the static HTML evidence report.
uv run iterant status ../my-agent
uv run iterant report ../my-agent        # writes ../my-agent/.iterant/report/
```

Notes:

- Each command takes an optional `repo` argument (default: the current directory) as its last argument.
- `iterant run` requires a passing `aa-test` for the pack unless you pass `--skip-calibration` (not recommended).
- With the defaults (`select` 80 tasks x 2 reps, `holdout` 40 x 2, `--delta-min 0.05`), your pack needs enough tasks for those splits. `estimate` tells you what gain your pack can actually detect. Flags such as `--select-tasks`, `--select-reps`, `--holdout-tasks`, `--holdout-reps`, `--smoke-tasks` and `--delta-min` change them; use the same values on every command.
- Use `--backend docker --unsafe-container` for Docker. The Docker daemon is root-equivalent, so use it on dev machines only.
- By default the proxy binds an address that sandboxes can reach through the container runtime. That address is reachable from your local network; requests are token-gated. Use `--proxy-bind` to change it.
- Run `uv run iterant <command> --help` for every flag.

## Not built yet

Holdout rotation, archive parent sampling (v1), eval bootstrap from logs (v1), research mode `iterant meta` (v1), and Firecracker as a selectable CLI backend.

## More

- [docs/threat-model.md](docs/threat-model.md): what the trusted base is, what each defense covers, known holes.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to build and test.

License: Apache-2.0
