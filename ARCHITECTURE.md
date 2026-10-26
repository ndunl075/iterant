# Iterant: architecture

**Your agent improves itself.** Point Iterant at your agent and your evals. It proposes changes to your agent's prompts, tools and logic, tests each one in disposable sandboxes, and keeps only the changes that beat the current version with statistical evidence.

- **Primary use:** improve *your* agent.
- **Side use:** a research platform for safe self-improvement. Iterant can also target its own proposer (§8).

## 1. Invariants (break one = reject the PR)

1. **Two zones.** The *genome* (the paths `iterant.toml` marks editable) is mutable. Everything else, including the rest of the target repo, Iterant itself and the task packs, is the trusted base (TCB). The TCB is never mounted into a sandbox writable.
2. **The candidate never grades itself.** Scores come only from the grader (a fresh sandbox, hidden checks) and the proxy logs. Nothing the agent reports counts.
3. **The key never enters a sandbox.** All LLM traffic goes through `proxy/`, which holds the key and enforces model, effort and USD caps for each role.
4. **The proposer is blind to selection data.** It sees `train` trajectories only. `select`, `holdout` and `test` return scores to the harness, never content.
5. **Frozen target model.** Gains must come from the agent's design, not from a bigger model or a bigger budget. The proxy enforces this.
6. **Append-only state.** Every candidate, run, proxy call and decision lands in the archive, rejects included. Rows are inserted once and never replaced or deleted (enforced by triggers); a candidate's `stage`, `status` and spend only advance. Runs are keyed by `(batch, cand, task, rep, attempt)`. A run resumes on `(batch, cand, task, rep)`, and the latest attempt wins. Each batch stores its pack hash, and a batch whose stored hash differs from the current pack is never resumed.
7. **Pinned tasks.** Task packs are hash-checked before every batch, and a mismatch aborts the batch.
8. **Verdicts are decided outside untrusted code.** Agent output (`/out`) is untrusted data. The grader sandbox that holds hidden checks and expected values never executes, imports or deserializes it: it reads `/out` only as regular files (no symlinks, hard links or special files; at most 1,000 files and 16 MiB total) and parses them only as text or strict JSON. If a check must execute agent-produced code, that code runs in a separate fresh sandbox with no hidden checks or expected values mounted, and only one JSON object crosses back (strict parser, at most 64 KiB, depth-limited). The verdict is a score in [0, 1]; NaN, infinity, a missing or malformed result, or a timeout of the check's wall-clock limit scores 0.0.

## 2. User contract

The user supplies two things: a target agent and a task pack.

**1. Target agent repo with `iterant.toml`:**
```toml
[target]
genome   = ["prompts/**", "tools/**", "agent/loop.py"]   # editable; all else frozen
image    = "Containerfile"                               # deps baked once, snapshotted
run      = "python -m agent --task /task/input.json --out /out"  # one task, any language
[target.model]          # frozen; the proxy rejects anything else
name = "claude-haiku-5-5"
effort = "medium"
max_usd_per_run = 0.10
[eval]
pack = "evals/support-pack"   # see below
```
- The agent reaches the LLM through `ANTHROPIC_BASE_URL`, which points at the proxy. The official SDKs read that variable, so the agent needs no code changes.
- The proxy speaks the Anthropic API and the OpenAI-compatible API, so local models (Ollama, llama.cpp) run the whole loop for $0. Hosted models are opt-in, and the user pays the provider.

**2. Task pack** (`pack.toml` + `tasks/<id>/`):
- `pack.toml` names the pack, its version and `grader_image`, a digest-pinned image (`name@sha256:<64 hex>`) that the grader runs checks in. It is never the candidate-built target image, which the candidate controls. The file is covered by the pack hash (§1.7). An optional `[policy] agent_code_exec_policy` is `allowed` (default) or `advisory-forbidden`: a request that the agent not execute code to solve tasks, which nothing enforces. The gate and report show it as unenforced.
- `input.json` is visible to the agent.
- `check` is hidden: a trusted pack command that reads `/out` as data and prints a score from 0 to 1, or exits 0 for pass (1.0) and non-zero for fail (0.0). It runs only in a fresh grader sandbox. Checks that must execute agent-produced code follow invariant 8.
- Each task is assigned a split: `train`, `select`, `holdout` or `test`. Iterant does this automatically and stratifies by the pack's `tags`.

**Onboarding helpers:**
- `iterant init` writes `iterant.toml` and a skeleton pack.
- `iterant estimate` measures baseline pass rate, variance and $ per run on a sample, then prints the smallest detectable gain and the $ per candidate *before* any spend.
- Starter packs (coding, extraction, support) plus a reference agent let people demo Iterant without their own evals.

## 3. Loop (one generation)

```
champion ──► PROPOSE ──► GATE ──► SMOKE ──► SELECT ──► CONFIRM ──► PROMOTE
 (git tag)   proposer    static   5 tasks   paired      holdout     tag gen-N
             edits       checks   no crash  stat test   stat test   archive row
             genome
                 ▲                                                    │
                 └──── failure digest (train split only) ◄────────────┘
```

| Stage | What | Kill rule |
|---|---|---|
| PROPOSE | The Iterant proposer runs in a sandbox on a copy of the target repo. Inputs are the train failure digest and archive history (tried/failed). Output is a diff plus a 1-line hypothesis. | No diff, or the diff is over 400 lines |
| GATE | Path allowlist (genome only), the target's own lint/tests if declared, grep for task IDs and check paths, LLM diff audit for special-casing | Any hit |
| SMOKE | 5 train tasks, 1 rep | Any crash or timeout the champion didn't have |
| SELECT | `select` split, k reps. Paired by task against a champion arm interleaved in the **same batch**, so provider drift cancels. Candidates in one batch share that arm. | See §4 |
| CONFIRM | `holdout` split, fresh runs of both arms | See §4 |
| PROMOTE | Commit to the `iterant/champion` branch of the target repo, tag `iterant/gen-N`, write an evidence report. The user merges when they choose. | n/a |

v0 picks parents greedily (champion only). In v1, parents are sampled from the archive (DGM-style), because pure hill-climbing stalls in local optima.

## 4. "Improvement" (acceptance rule)

Accept only if **all** of these hold:
- **SELECT:** paired per-task gain Δ ≥ δ_min (default 5 pts), and a one-sided paired bootstrap gives p < 0.05.
- **CONFIRM:** holdout Δ > 0 and p < 0.2. The two stages compound to a ~1% false accept per null candidate.
- **Guards:** cost per task ≤ +15%, wall time ≤ +25%, refusal and infra-error rate no worse. Every threshold is configurable.

Calibration gates (`iterant estimate` and `iterant aa-test`; both must pass before a real run):
- **A/A test.** Resample a pool of champion reps: false accepts ≤ 5%. If it fails, the stats or the pack are broken; stop.
- **Null-diff control.** A whitespace-only diff must be rejected.
- **Power.** The `select` size sets the smallest gain you can detect (measured at 15% discordant tasks, power 0.8):

| Gain to detect | Tasks needed (k=1) |
|---|---|
| +10 pts | ~80 |
| +5 pts | ~360 |
| +3 pts | ~1,000 |

  Defaults are `select` 80 × k=2 and `holdout` 40 × k=2. If the pack is too small for the gain the user wants, `estimate` says so up front.

Overfitting control: every holdout query is logged. After 20 queries, Iterant warns and asks for fresh holdout tasks. `test` is touched only for reports.

Sampling params are gone from the API, so runs are inherently noisy. Repeats and paired stats are mandatory, not optional.

## 5. Components

```
iterant/
  orchestrator/   generation loop, stage scheduler, $ budget, resume    [T2]
  target/         iterant.toml parser, genome allowlist, image build     [T2]
  sandbox/        Sandbox protocol + backends                            [T2]
  proxy/          LLM egress: key custody, role→model/effort, caps, log  [T2]
  gate/           allowlist, static checks, diff audit                   [T2]
  grader/         runs hidden checks on agent output in a fresh sandbox (invariant 8)  [T2]
  stats/          paired bootstrap, acceptance rule, power, estimate     [T1]
  archive/        SQLite + git refs in the target repo                   [T2]
  analyst/        trajectory → failure digest                            [T1]
  proposer/       Iterant's own editing agent (its genome, see §8)       [T1]
  packs/          pack format, hash check, splitter, starter packs       [T1]
  cli/            init | estimate | aa-test | run | status | report      [T1]
  report/         static HTML: lineage, Δ per gen, $ per gen, diffs      [T1]
examples/         reference agent + starter packs (demo and CI fixture)  [T1]
```
[T1] and [T2] are build review tiers (§7).

**Sandbox protocol:** `create(image, cpu, mem, disk, ttl, net="proxy-only")`, `put`, `get`, `exec(cmd, timeout)`, `destroy`. Backends:

| Backend | Use | Isolation |
|---|---|---|
| `podman` (rootless; `docker` accepted with `--unsafe-container`) | v0, laptops | Container, not a VM. Dev only. |
| `firecracker` | Default at public release. Needs `/dev/kvm`. | microVM, snapshot boot, CoW rootfs |
| `e2b` / `modal` (optional) | Hosts without KVM | Hosted microVM |

Every backend runs non-root with no host mounts, a CPU/mem/disk/time cap and default-deny egress, plus one route to the proxy.

`exec` returns an `ExecResult` for anything the command can cause: a timeout, output cut at the capture cap (`truncated`), a non-zero exit, an OOM kill, pids exhaustion, or killing PID 1 so the sandbox is gone. The grader scores all of these 0.0. `exec` raises only for failures the command cannot cause (runtime missing or broken, sandbox already destroyed or expired), and those are infra errors under the §6 policy.

**Archive (SQLite, in `.iterant/` of the target repo):**
- `batches(id, gen, pack_hash, started_at)`: one scheduled set of runs, pinned to a pack hash (§1.7)
- `candidates(id, parent, diff_sha, hypothesis, stage, status, usd_micros)`
- `runs(batch, cand, task, rep, attempt, split, score, in_tok, out_tok, usd_micros, wall_s, refusal, infra_err)`: `attempt` 1 is the single retry of a refused or infra-errored run (§6)
- `promotions(gen, cand, select_delta, select_p, holdout_delta, holdout_p)`
- `proxy_calls(id, batch, cand, task, rep, attempt, role, model, in_tok, out_tok, usd_micros, refusal, created_at)`: the proxy's own log; the run key is NULL for calls outside a run (proposer, analyst, auditor)

Money is integer micro-USD. All tables are STRICT. The full DDL, with its triggers, lives in `iterant/contracts.py`.

**Stack:** Python 3.12, uv, pytest, ruff, pyright, the official `anthropic` SDK for Claude, and the `openai` SDK only for OpenAI-compatible local endpoints. Tests use a recorded-fixture fake LLM and never call a live model.

## 6. Runtime model routing

| Role | Model | Effort | Calls | Why |
|---|---|---|---|---|
| Target agent | the user's choice, frozen by proxy (reference agent: `claude-haiku-5-5`) | user's choice (ref: medium) | ~245/candidate + ~240 shared champion arm | Most of the spend. Tune on a cheap model and optionally confirm on the production model. |
| Proposer | `claude-opus-5-5` | high | 1/candidate | Highest-leverage step and a small share of spend. In Wave 3, A/B it on accepts per $ against `claude-fable-5-1` high. |
| Failure analyst | `claude-haiku-5-5` | medium | 1/failed train run, Batch API (-50%) | Reading-heavy and high-volume. Output ≤ 200 tokens per digest. |
| Diff auditor (gate) | `claude-sonnet-5-5` | medium | 1/candidate | Adversarial judgment ("is this special-casing tasks?") at low cost |
| Transfer check (optional) | the user's production model | user's choice | `test` split per promotion | Confirms gains found on a cheap model hold on the production model |
| Grader | none | n/a | n/a | The pack's hidden checks. No LLM judge in v0 (judges are hackable). |

**Profiles:**
- `free` (default): every role runs on one local model through Ollama, at $0. Expect fewer accepted candidates, because a weak proposer makes weaker proposals.
- `claude`: the table above, paid by the user.

The proxy treats both identically. Caps are in tokens for `free` and USD for `claude`.

Request rules for Iterant's own calls (Claude API as of 2026-10):
- Omit `thinking` or send `{type:"adaptive"}`, and set `output_config.effort` explicitly (Opus and Haiku 5.5 default to medium). Disabling thinking or sending `budget_tokens` returns a 400.
- No `temperature`/`top_p`/`top_k`, no prefill, no forced `tool_choice` (use `auto` + `strict: true`).
- Stream long requests. Pass thinking blocks back unchanged. History is append-only.
- Put `cache_control` on the stable prefix and check that `usage.cache_read_input_tokens` > 0.
- Refusals: always check `stop_reason`. Proposer and auditor use `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`). **Target-agent traffic is never rerouted**, because a silent model swap corrupts the measurement. A refused run counts as `infra_err`: retry it once, then drop that task pair from both arms. Track refusal rate; self-improvement content may trip the Sonnet `frontier_llm` classifier.

`claude` profile spend for the reference agent is an estimate, measured by the first user who opts in:
- Target agent: ~$0.01–0.03 per run.
- Proposer: ~$1 per candidate.
- One candidate that reaches CONFIRM: ~$10–15.

For any other agent, `iterant estimate` gives the real figure. Hard caps are enforced by the proxy, both per run and per generation.

## 7. Build-time model routing (architect workflow)

| Who | Model | Effort | Does |
|---|---|---|---|
| Architect | Opus 5.5 | session default | Plans, splits, briefs, merges. Never builds. |
| Worker session, T1 module | Sonnet 5.5 | medium | One module → one PR |
| Worker session, T2 module | Sonnet 5.5 | high | target, sandbox, proxy, gate, grader, archive, orchestrator (isolation, money, persistence, recovery) |
| Research/explore subagent | Sonnet (Explore type to locate code) | default | Answers the architect's next question |
| Tier 0 batch and Tier 1 review, all re-reviews, CI triage | Sonnet | default | Zero-context review, findings only |
| Tier 2 review | opus-engineer (fallback: general-purpose + `model: opus`); the architect also reads the diff | default | Every T2 module PR plus its fix diffs |
| Counsel | 1 Opus + 1 Sonnet (stances) | default | Hard-to-reverse calls (data model, sandbox backend, acceptance rule) |
| Root cause after a failed Sonnet pass | Opus subagent | default | Escalation only |

## 8. Research mode (side track, v1)

`iterant meta` makes the target Iterant's own `proposer/`. The task pack is a set of *improvement tasks*: seeded agents with known headroom. Score = holdout-confirmed gain per $. The same loop, gate, TCB and stats apply unchanged.

This is the self-improvement story for researchers, and it dogfoods the platform.

## 9. Threat model (what the gate and TCB stop)

| Attack by a candidate | Stopped by |
|---|---|
| Edit the checks, the pack or Iterant | Genome allowlist; TCB is never mounted writable |
| Read hidden checks | Checks mount only in the grader sandbox |
| Use a bigger model or more budget | Proxy role caps; cost guard |
| Hard-code train/select answers | Proposer is blind to select; task-ID grep; diff auditor; holdout rotation |
| Forge logs or claim success | Scores come only from the grader and proxy |
| Tamper with the grader (patch its test framework, read its memory, plant symlinks or huge files in `/out`) | Invariant 8: agent output is plain data to the grader; agent code runs only in a separate fresh sandbox without expected values |
| Exfiltrate the key, phone home | Key lives in the proxy; default-deny egress |
| Escape the sandbox | microVM (release); non-root, no mounts, caps |
| Ship a bad change to production | Iterant never merges; the user merges `iterant/champion` after reading the evidence report |

Red-team suite (CI, Wave 3): a set of malicious candidate diffs, one per row above. Each must be rejected or contained.

**Main adoption risk:** users without evals. A weak pack teaches the agent to game weak checks. `estimate` reports pass rate and variance, and warns on packs that are too easy, too hard or too small. Eval bootstrap from logs (v1) must produce human-reviewed checks, never auto-accepted ones.
