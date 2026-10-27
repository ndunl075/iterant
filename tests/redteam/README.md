# Red-team suite (ARCHITECTURE.md §9)

Every test plays a malicious candidate and asserts that Iterant stops it. The real gate, grader,
orchestrator, proxy and sandbox backends run; fakes sit only at the edges (the sandbox runtime,
the model upstream, the proposer and the auditor). Fixtures reuse the module suites'
(`tests/gate/conftest.py`, `tests/orchestrator/conftest.py`), plus `HostileWorld` in
`conftest.py`, which lets a test hijack the agent's run inside the sandbox.

Run: `uv run pytest tests/redteam -q`.

**Live sandbox tests run only on demand, not in CI.** Tests marked `podman` / `docker` (the
`test_row8_live_*` cases) need a container runtime; ci.yml deselects them and we add no container
CI job ($0 budget). Run them by hand on a machine with rootless Podman or Docker:

    uv run pytest tests/redteam/test_row8_sandbox_escape.py -m "podman or docker" -q -rs

They last passed on a throwaway `probe/*` branch (workflow never merged). **There is no
Firecracker red-team test yet**, so the microVM backend is not attacked here.

**Auditor-stub tests prove less than they look.** Where a test says "the auditor rejects X" (rows
1, 2, 4, 7), the auditor is a stub returning a canned verdict. Those tests prove only that the diff
is *sent to* the auditor and that a flagging verdict stops the gate; they say nothing about whether
the real LLM auditor would flag it.

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
| `test_row3_non_finite_max_tokens_is_a_rejection_not_a_crash` | FIXED by PR #22 (marker removed): `max_tokens` NaN or Infinity now returns a 400 rejection from `Proxy.admit` |
| `test_row3_one_huge_prompt_cannot_blow_through_the_run_cap` | FIXED by PR #22 (marker removed): admit now reserves an input-token estimate plus `max_tokens` |

Closed as detection (row 5, holes 3 and 4): the archive refuses to open (`ArchiveTampered`) when
its schema (every table, index and trigger, compared with the reference DDL for its
`user_version`) has any extra, missing or altered object, so `DROP TRIGGER`, `writable_schema`,
an injected `RAISE(IGNORE)` trigger or dropped tables are caught on the next open
(`test_row5_dropping_the_triggers_is_detected_on_reopen`,
`test_row5_writable_schema_is_detected_on_reopen`). Not prevented: whoever can write the file can
still rewrite rows and restore the original schema text; a tamper-evident hash chain over run and
promotion rows is future work.

## Missing defenses (no test can pass; nothing to xfail)

- **Holdout rotation (§9 row 4) is not implemented.** The orchestrator always takes the first
  `holdout_tasks` of the holdout split, in id order (`iterant/orchestrator/loop.py:527`:
  `tasks_in(self._pack, Split.HOLDOUT)[: t.holdout_tasks]`). The only defense is the query-count
  warning after `holdout_query_warn` queries (`loop.py:530-533`), which nothing blocks on.
  Row 4 is therefore covered by the proposer's blindness, the grep and the auditor only.

## Open questions

- The fixtures are imported from other suites (`tests/gate/conftest.py`,
  `tests/orchestrator/conftest.py`, `tests/sandbox/test_argv.py`). Renaming or reshaping those
  helpers breaks this suite. Copying them would isolate it at the cost of duplication.
