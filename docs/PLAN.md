# Iterant: build plan (architect only; workers read ARCHITECTURE.md, not this)

## Positioning (decided 2026-10-26)

- **Primary: your agent improves itself.** Users bring an agent and evals, and Iterant keeps only the proven improvements.
- **Side: research mode** (`iterant meta`), Iterant improving its own proposer. This brings credibility and dogfooding.
- **Edge over DGM and SICA:** it works on *your* agent, it's statistically gated, it has a hardened TCB, and it costs ~$10 per candidate on cheap models.
- **Main adoption risk:** users without evals. Answer: starter packs, the reference agent, `iterant estimate` warnings, and eval bootstrap in v1.

## Decisions (counsel 2026-10-26: Opus skeptic + Sonnet simplest-possible)

| Topic | Skeptic | Simplest | Decision |
|---|---|---|---|
| Reference solver model | Sonnet 5.5 medium | Haiku 5.5 medium | **Haiku** for the reference agent. Users pick their own model, and the transfer check confirms on their production model. |
| v0 sandbox | rootless Podman / Firecracker | Docker | **Podman v0, Firecracker before release** |
| Threshold | +3 pts, 40 tasks × 3 | +8 pts, 30 × 3 | **+5 pts, 80 × 2 + holdout**, configurable. `estimate` reports what the pack can actually detect. |
| Baseline | same batch | same batch | **Adopted** |
| Null/A-A test, hidden checks, no judge, proxy key custody, Opus 5.5 high proposer | ✓ | ✓ | Unanimous |

## Open (Nico decides)
1. ~~License~~ Decided: Apache-2.0.
2. ~~Tagline~~ Decided: "Iterant: your agent improves itself." It is a placeholder; revisit at launch.
3. ~~Spend~~ Decided: **$0 budget.** The project never spends money. Statistical calibration uses simulation, end-to-end runs use local models, and the paid `claude` profile ships untested by us; contributors who use it report real costs.

## Waves

The current main SHA goes into every brief at launch. Workers own only the listed paths.

**W0: architect, sequential**
| # | PR | Tier |
|---|---|---|
| 0.1 | Scaffold (uv, ruff, pyright, pytest, CI), CLAUDE.md, ARCHITECTURE.md | 1 |
| 0.2 | `iterant/contracts.py`: Sandbox protocol, `iterant.toml` schema, Task/Pack, RunResult, Candidate, archive DDL, proxy role schema, all with type tests. This is the shared hotspot; landing it first is what lets W1 run in parallel. | 2 |
| 0.3 | `docs/parallel-sessions.md` (stack, commands, local-only checks such as Firecracker/KVM, review gate) | 1 |
| 0.4 | Sonnet research subagent picks the first starter pack: ≥200 tasks, hidden checks, under 2 min each, permissive license, Haiku baseline 30–60% | n/a |

**W1: 6 parallel workers** (needs W0 merged)
| # | Worker | Owns | Model / effort | Tier |
|---|---|---|---|---|
| 1 | sandbox: Podman + Docker-flagged backends, conformance suite | `iterant/sandbox/` | Sonnet high | 2 |
| 2 | proxy: key custody, Anthropic + OpenAI-compatible (Ollama) upstreams, role caps, token/USD ledger, call log | `iterant/proxy/` | Sonnet high | 2 |
| 3 | packs + grader: format, splitter, hash check, first starter pack, grader in a fresh sandbox | `iterant/packs/`, `iterant/grader/` | Sonnet high | 2 |
| 4 | target + reference agent: `iterant.toml` parser, genome allowlist, image build, `examples/` agent | `iterant/target/`, `examples/` | Sonnet high | 2 |
| 5 | stats: paired bootstrap, acceptance rule, `power()`, estimate math | `iterant/stats/` | Sonnet medium | 1 |
| 6 | archive: SQLite DAO, git refs in the target repo, resume keys | `iterant/archive/` | Sonnet high | 2 |

**W2: 6 parallel workers** (needs W1 merged)
| # | Worker | Owns | Model / effort | Tier |
|---|---|---|---|---|
| 7 | orchestrator: stages, batch-shared champion arm, budget, resume | `iterant/orchestrator/` | Sonnet high | 2 |
| 8 | gate: allowlist, target lint/tests, task-ID grep, diff-auditor call | `iterant/gate/` | Sonnet high | 2 |
| 9 | proposer + analyst: editing agent, digest via Batch API, archive history in input | `iterant/proposer/`, `iterant/analyst/` | Sonnet medium | 1 |
| 10 | CLI: init, estimate, aa-test, run, status | `iterant/cli/` | Sonnet medium | 1 |
| 11 | report: static HTML evidence report per promotion | `iterant/report/` | Sonnet medium | 1 |
| 12 | firecracker backend (passes the W1 conformance suite) | `iterant/sandbox/firecracker/` | Sonnet high | 2 |

**W3: calibration + red team** ($0)
- Worker 13: red-team CI suite, one malicious diff per threat row (Sonnet high, T2).
- Worker 14: a stats simulator that generates synthetic pass/fail runs with known true gains, used to verify the A/A false-accept rate, power and the acceptance rule. This needs no LLM (Sonnet medium, T1).
- Architect runs, on the reference agent with a local model: `estimate`, A/A via resampling, then a first run of 20 candidates.

**W4: release**
- README with a 5-minute demo on the reference agent, a threat-model doc, CONTRIBUTING.
- Firecracker becomes the default.
- Tier 0 batch review for the docs.

**v1:** archive parent sampling, more providers, eval bootstrap from logs (human-reviewed checks), research mode (`iterant meta`), more starter packs.

## Kill / pivot criteria
- A/A false accept above 5% → the stats are broken. Run nothing else until fixed.
- No accepts after 30 candidates on the reference agent → try a stronger local proposer model before blaming the loop.
- Select gains fail to confirm on holdout after ~30 candidates → the pack is too small or the idea doesn't work. Stop adding features and grow the pack, or stop.
- Three design partners can't produce a usable pack in a day → the eval barrier is the product. Build eval bootstrap before anything else.
