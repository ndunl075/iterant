# Red-team suite (ARCHITECTURE.md §9)

Every test plays a malicious candidate and asserts that Iterant stops it. The real gate, grader,
orchestrator, proxy and sandbox backends run; fakes sit only at the edges (the sandbox runtime,
the model upstream, the proposer and the auditor). Fixtures reuse the module suites'
(`tests/gate/conftest.py`, `tests/orchestrator/conftest.py`), plus `HostileWorld` in
`conftest.py`, which lets a test hijack the agent's run inside the sandbox.

Run: `uv run pytest tests/redteam -q`. Tests marked `podman` / `docker` need a container runtime
and are deselected in CI's main job (they skip when the runtime is missing).

## Threat rows to tests

| §9 row | Attack | File |
|---|---|---|
| 1. Edit the checks, the pack or Iterant | path tricks, renames, copies, symlinks, mode changes, header lies, hidden second sections, wide genomes, meta mode | `test_row1_genome_escape.py` |
| 2. Read hidden checks | checks in agent sandboxes, grader sandbox contents, naming or copying check paths and content | `test_row2_hidden_checks.py` |
| 3. Use a bigger model or more budget | model, effort, fallbacks, beta, server tools, max_tokens, run and generation caps, restart, cost and wall guards | `test_row3_model_and_budget.py` |
| 4. Hard-code train/select answers | task-ID grep, auditor backstop; proposer blind to non-train (file tools, repo copy, digest, proxy content) | `test_row4_hardcoding_and_blindness.py` |
| 5. Forge logs or claim success | forged files in `/out`, agent stdout, usage claims, request metadata, aborted calls, archive rewrites | `test_row5_forged_scores.py` |
| 6. Tamper with the grader | hostile result JSON (NaN, inf, duplicates, depth, size), hostile `/out` (links, FIFOs, huge, deep), file names, pack and image tampering | `test_row6_grader_tampering.py` |
| 7. Exfiltrate the key, phone home | wrong, borrowed, replayed and other-role tokens, routes, header smuggling, key scans, network modes | `test_row7_exfiltration.py` |
| 8. Escape the sandbox | runtime argv, image-name injection (always on); non-root, caps, network, host files, fork bomb, hostile `/out` in a real sandbox (`podman` / `docker`) | `test_row8_sandbox_escape.py` |
| 9. Ship a bad change to production | main, HEAD, working tree, remotes untouched; hostile hypothesis text | `test_row9_never_ships.py` |
| Rows 5 and 6, cross-cutting | make the candidate's own failures look like infra errors so its hard tasks are dropped (lost sandbox, timeout, `get` failures, bad listings, refusals) | `test_infra_laundering.py` |

## Known holes

Tests marked `xfail(strict=True, reason="HOLE: ...")` document a real hole in product code. The
architect fixes the code; the fix turns the test green, strict xfail then fails until the marker
is removed.

| Test | Hole |
|---|---|
| `test_row3_non_finite_max_tokens_is_a_rejection_not_a_crash` | `max_tokens` NaN or Infinity raises inside `Proxy.admit` instead of returning a rejection |
| `test_row3_one_huge_prompt_cannot_blow_through_the_run_cap` | the run cap reserves output tokens only; a huge prompt settles far past the cap |
