# Threat model

This expands ARCHITECTURE.md section 9. The attacker is a **candidate**: a change the proposer (an LLM) makes to your agent's genome, which may be malicious or may just learn to game the checks. Each row below lists the defenses and the red-team test that attacks it. Tests are in [`tests/redteam/`](../tests/redteam/README.md).

Status: pre-release. The red-team suite runs against the real gate, grader, orchestrator and proxy, with fakes only at the edges (sandbox runtime, model upstream, proposer, auditor). No run with a real model has been done; a one-off `iterant estimate` run against a stub model on Podman completed end to end.

## Trusted computing base (TCB)

The **genome** is the set of paths `iterant.toml` marks editable. It is the only thing a candidate may change. Everything else is the TCB and is never mounted into a sandbox writable: the rest of the target repo, Iterant itself, the task packs (including hidden checks), the grader image, the proxy and the archive.

Invariants the model rests on (ARCHITECTURE.md section 1):

- The candidate never grades itself. Scores come from the grader and the proxy log.
- The API key stays in the proxy and never enters a sandbox.
- The proposer sees `train` trajectories only.
- The target model is frozen. The proxy enforces it.
- The archive is append-only.
- Task packs are hash-checked before every batch.
- Agent output is untrusted data. The grader reads it only as plain files and strict JSON, and runs agent-produced code only in a separate sandbox with no expected values.

## Threats and defenses

| # | Attack by a candidate | Defenses | Red-team test |
|---|---|---|---|
| 1 | Edit the checks, the pack or Iterant | Genome allowlist in the gate; TCB never mounted writable; pack hash check; diff auditor | [`test_row1_genome_escape.py`](../tests/redteam/test_row1_genome_escape.py) |
| 2 | Read hidden checks | Checks mount only in the grader sandbox; task-ID and check-path grep; diff auditor | [`test_row2_hidden_checks.py`](../tests/redteam/test_row2_hidden_checks.py) |
| 3 | Use a bigger model or more budget | Proxy role caps on model, effort, tokens and USD, per run and per generation; cost and wall-time guards | [`test_row3_model_and_budget.py`](../tests/redteam/test_row3_model_and_budget.py) |
| 4 | Hard-code train or select answers | Proposer blind to non-train data; task-ID grep; diff auditor; query-count warning on the holdout | [`test_row4_hardcoding_and_blindness.py`](../tests/redteam/test_row4_hardcoding_and_blindness.py) |
| 5 | Forge logs or claim success | Scores come only from the grader and proxy log; append-only archive | [`test_row5_forged_scores.py`](../tests/redteam/test_row5_forged_scores.py) |
| 6 | Tamper with the grader (hostile `/out`, links, huge files, NaN scores) | Invariant 8: `/out` is plain data with size and type limits, strict JSON, non-finite or malformed results score 0.0, agent code runs in a separate sandbox | [`test_row6_grader_tampering.py`](../tests/redteam/test_row6_grader_tampering.py) |
| 7 | Exfiltrate the key, phone home | Key held only by the proxy; token-gated proxy requests; default-deny egress with one route to the proxy | [`test_row7_exfiltration.py`](../tests/redteam/test_row7_exfiltration.py) |
| 8 | Escape the sandbox | Non-root, no host mounts, capability and resource caps, default-deny network; microVM backend (release default; not selectable from the CLI yet, see Known holes) | [`test_row8_sandbox_escape.py`](../tests/redteam/test_row8_sandbox_escape.py) |
| 9 | Ship a bad change to production | Iterant never merges; you merge `iterant/champion` after reading the evidence report | [`test_row9_never_ships.py`](../tests/redteam/test_row9_never_ships.py) |
| 5, 6 | Make the candidate's own failures look like infra errors, so hard tasks are dropped | Lost sandboxes, timeouts and bad listings score 0.0 | [`test_infra_laundering.py`](../tests/redteam/test_infra_laundering.py) |

## What the tests do and do not prove

- Where a test says "the auditor rejects X" (rows 1, 2, 4, 7), the auditor is a stub with a canned verdict. Those tests prove the diff is sent to the auditor and that a flagging verdict stops the gate. They say nothing about whether the real LLM auditor would flag it.
- The live sandbox tests (`test_row8_live_*`, marked `podman` or `docker`) are not run in CI. They need a container runtime and are run by hand (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- **There is no Firecracker red-team test.** The microVM backend is not attacked by this suite.

## Known holes and missing defenses

Stated as in `tests/redteam/README.md`, which is the current list, plus notes from the code reviews. Some tests are strict `xfail`s that document a hole in product code until it is fixed.

- **Archive append-only triggers can be dropped or bypassed.** The triggers are plain SQLite triggers. `DROP TRIGGER`, or `PRAGMA writable_schema=1` with a delete from `sqlite_master`, removes them, and a later `UPDATE` on `runs` then succeeds undetected. This needs write access to the archive file, which a sandboxed candidate is not given. Documented by strict `xfail` tests in `test_row5_forged_scores.py`.
- **No holdout rotation.** Section 9 lists it as a defense for row 4, but it is not implemented. The orchestrator always takes the first `holdout_tasks` of the holdout split in id order. The only protection is a warning after `holdout_query_warn` queries, and nothing blocks on it. Row 4 rests on the proposer's blindness, the grep and the auditor.
- **The podman proxy is reachable from your LAN.** Loopback is unreachable from rootless Podman, so the proxy binds an address the sandbox can reach, and other machines on the network can reach it too. Requests are token-gated. `--proxy-bind` changes the address; `0.0.0.0` is an explicit opt-in that exposes every interface.
- **The Firecracker backend is not attacked by the red-team suite.** It is selectable with `--backend firecracker`. It also runs without Firecracker's jailer (deferred to the release work; see PR #18). Podman is a container, not a VM, and is for development only; the Docker backend needs `--unsafe-container` because the daemon is root-equivalent.
- **The CRUXEval code-exec policy is advisory and unenforced.** The pack sets `agent_code_exec_policy = "advisory-forbidden"` to ask the agent not to run code. Nothing can tell a script that solves the task from one that executes the function, so a candidate that runs code is neither blocked nor detected. The gate and report show the policy as unenforced. Treat a high score from a candidate with code-execution tools accordingly.
- **Proxy cap estimates are heuristic.** The non-finite `max_tokens` crash and the output-only reservation are fixed (PR #22). Admit now reserves an input estimate (bytes/3, plus fixed amounts for images and documents) and allows at most 4 calls in flight per run token, but text denser than 3 bytes per token can still settle somewhat past a cap.

## Adoption risk

A weak pack teaches the agent to game weak checks. `iterant estimate` reports pass rate and variance and warns on packs that are too easy, too hard or too small. The red-team suite cannot catch a badly written pack.
