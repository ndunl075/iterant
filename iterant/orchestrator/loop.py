"""One generation of the §3 loop: PROPOSE → GATE → SMOKE → SELECT → CONFIRM → PROMOTE.

State lives in the archive, so a restarted process picks up where the last one stopped:

* Candidate ids are ``b<batch>-<index>``; the ids say which batch a candidate belongs to. A batch
  that is abandoned (infra abort, changed pack) gets a ``b<batch>-closed`` marker whose
  hypothesis is the reason; a batch with a marker is never reopened.
* A candidate's ``stage`` is where it is *now* (``ACTIVE`` at ``SMOKE`` means the gate passed and
  smoke is pending or half done); ``REJECTED`` at a stage means it was killed there.
* A run slot ``(batch, cand, task, rep)`` is done once it has a scored row, or a second attempt
  (§6: one retry). Resume runs only the missing slots (and the missing retry), so rows are
  neither duplicated nor skipped.
* Stage verdicts are recomputed from the archived runs with fixed seeds, never stored.
* Promotion is git first, then one DB transaction; a lost DB write is repaired on the next start,
  even if the pack changed in between.
* One orchestrator per archive (``WriterLock``). Pass it a fresh ``Proxy``: on its first run it
  preloads the generation's archived calls into the proxy's in-memory ledger.

The champion arm is a normal candidate id (the champion's) whose runs sit in the same batch, so
every candidate of the batch is paired against the very same champion runs (§3).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from iterant.archive import (
    Archive,
    GitRefError,
    archive_path,
    champion_sha,
    diff_sha256,
    ensure_seed,
    find_promoted_commit,
    gen_tag_sha,
    promote,
)
from iterant.contracts import (
    Batch,
    Candidate,
    Gate,
    Pack,
    Promotion,
    Proposal,
    Proposer,
    ProxyCall,
    Role,
    RunKey,
    SandboxBackend,
    Split,
    Stage,
    Status,
    TargetConfig,
    Task,
    TriedCandidate,
    clean_hypothesis,
    open_archive,
)
from iterant.grader import GraderPreflightError
from iterant.orchestrator.budget import Budget
from iterant.orchestrator.config import (
    BatchAborted,
    BudgetExhausted,
    FailureDigest,
    GraderLike,
    ImageSource,
    InfraRateExceeded,
    OrchestratorConfig,
)
from iterant.orchestrator.lock import WriterLock
from iterant.orchestrator.outcomes import Outcomes
from iterant.orchestrator.runs import RunExecutor
from iterant.orchestrator.workspace import DiffStore, export_commit, proposer_view
from iterant.packs import tasks_in
from iterant.packs.hashing import verify
from iterant.proxy import Proxy
from iterant.stats import Decision, decide_confirm, decide_select

PACK_HASH_TRAILER = "Iterant-Pack-Hash"
SEED_ID = "seed"
CLOSED = "closed"


class HoldoutLimitReached(BatchAborted):
    """The holdout has been queried ``holdout_query_max`` times: the generation will not start
    (nothing is rejected). Raise the limit or start a fresh archive."""


@dataclass(frozen=True)
class CandidateOutcome:
    id: str
    stage: Stage
    status: Status
    # Why it was killed (this process only; a closed batch keeps its reason in the archive).
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GenerationResult:
    gen: int
    batch_id: int
    candidates: tuple[CandidateOutcome, ...]
    promoted: str | None
    stopped: Literal["done", "budget"]
    spent_usd_micros: int  # the whole generation so far (every batch, every role)
    warnings: tuple[str, ...] = ()


@dataclass
class _Gen:
    batch: Batch
    champ: Candidate
    champ_sha: str
    budget: Budget
    notes: dict[str, list[str]] = field(default_factory=lambda: {})
    images: dict[str, str] = field(default_factory=lambda: {})
    warnings: list[str] = field(default_factory=lambda: [])
    digest: str | None = None


def _is_marker(c: Candidate) -> bool:
    return c.id.endswith(f"-{CLOSED}")


class Orchestrator:
    def __init__(
        self,
        *,
        repo: str | Path,
        pack: Pack,
        pack_root: Path,
        target: TargetConfig,
        archive: Archive,
        backend: SandboxBackend,
        grader: GraderLike,
        proxy: Proxy,
        proposer: Proposer,
        gate: Gate,
        images: ImageSource,
        digest: FailureDigest,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._repo = Path(repo)
        self._pack = pack
        self._pack_root = pack_root
        self._target = target
        self._archive = archive
        self._grader = grader
        self._proxy = proxy
        self._proposer = proposer
        self._gate = gate
        self._images = images
        self._digest = digest
        self._cfg = config or OrchestratorConfig()
        self._store = DiffStore(self._repo)
        self._outcomes = Outcomes(self._repo)
        self._exec = RunExecutor(
            backend=backend,
            grader=grader,
            proxy=proxy,
            run_command=target.run,
            read_input=lambda t: (pack_root / t.input).read_bytes(),
            config=self._cfg,
        )
        self._preloaded = False
        target_cap = proxy.config.policy.roles.get(Role.TARGET)
        if self._cfg.budget_usd_micros is not None and (
            target_cap is None or target_cap.max_usd_micros_per_run is None
        ):
            raise ValueError("a $ budget needs the claude profile (USD caps on the target role)")
        self._worst_run = self._worst(Role.TARGET)
        try:
            self._pack_rel: str | None = (
                pack_root.resolve().relative_to(self._repo.resolve()).as_posix()
            )
        except ValueError:
            self._pack_rel = None  # the pack lives outside the repo
        if self._pack_rel == ".":
            raise ValueError("the pack root must not be the repo root: the proposer would see it")

    # --- entry point --------------------------------------------------------------

    def run_generation(self) -> GenerationResult:
        """Run (or resume) one generation.

        Raises ``OrchestratorBusy`` if another orchestrator holds the archive, ``BatchAborted``
        on an infra failure (``InfraRateExceeded`` closes the batch; the others leave it
        resumable), and the packs module's ``PackHashMismatch`` if the pack on disk is not the
        pack this orchestrator was given (invariant 7). A batch pinned to an older pack is
        closed and a new one opened.
        """
        with WriterLock(self._repo):
            return self._run()

    def _run(self) -> GenerationResult:
        self._reconcile_promotion()
        verify(self._pack, self._pack.content_hash, self._pack_root)
        champ, last_gen = self._champion()
        try:
            self._grader.preflight()
        except GraderPreflightError as exc:
            raise BatchAborted(f"grader preflight failed: {exc}") from exc
        sha = champion_sha(self._repo)
        if sha is None:
            raise BatchAborted("no iterant/champion branch")
        batch = self._open_batch(last_gen, champ)
        self._check_holdout_limit(batch, self._cfg.thresholds.holdout_query_max)
        self._repair_spend(batch)
        self._proxy.begin_generation(batch.gen)
        if not self._preloaded:
            self._proxy.preload(self._gen_calls(batch.gen))
            self._preloaded = True
        budget = Budget(self._cfg.budget_usd_micros, lambda: self._gen_spend(batch.gen))
        g = _Gen(batch, champ, sha, budget)
        stopped: Literal["done", "budget"] = "done"
        try:
            self._propose_all(g)
            self._gate_phase(g)
            self._smoke_phase(g)
            self._select_phase(g)
            self._confirm_phase(g)
            self._promote_phase(g)
        except BudgetExhausted:
            stopped = "budget"
        except InfraRateExceeded as exc:
            self._close(batch, champ, f"infra-abort: {exc}")
            raise
        return self._result(g, stopped)

    # --- champion, batch ------------------------------------------------------------

    def _champion(self) -> tuple[Candidate, int]:
        promo = self._archive.latest_promotion()
        if promo is not None:
            cand = self._archive.get_candidate(promo.cand)
            assert cand is not None
            return cand, promo.gen
        ensure_seed(self._repo)
        seed = self._archive.get_candidate(SEED_ID)
        if seed is None:
            seed = Candidate(
                SEED_ID, None, diff_sha256(b""), "seed champion", Stage.PROMOTE, Status.PROMOTED, 0
            )
            self._archive.insert_candidate(seed)
        return seed, 0

    def _latest_batch(self) -> Batch | None:
        latest, i = None, 1
        while (b := self._archive.get_batch(i)) is not None:
            latest, i = b, i + 1
        return latest

    def _all_members(self, batch_id: int) -> list[Candidate]:
        prefix = f"b{batch_id}-"
        return [c for c in self._archive.iter_candidates() if c.id.startswith(prefix)]

    def _members(self, batch_id: int) -> list[Candidate]:
        """The batch's real candidates (the closed marker is not one)."""
        return [c for c in self._all_members(batch_id) if not _is_marker(c)]

    def _close(self, batch: Batch, champ: Candidate, reason: str) -> None:
        """Abandon ``batch``: reject its live candidates and leave the reason on a marker."""
        members = self._all_members(batch.id)
        if any(_is_marker(c) for c in members):
            return
        for c in members:
            if c.status is Status.ACTIVE:
                self._archive.finish_candidate(c.id, Status.REJECTED)
        marker = Candidate(
            f"b{batch.id}-{CLOSED}",
            champ.id,
            diff_sha256(b""),
            reason[:500],
            Stage.PROPOSE,
            Status.ERROR,
            0,
        )
        self._archive.insert_candidate(marker)

    def _open_batch(self, last_gen: int, champ: Candidate) -> Batch:
        """The unfinished batch of the next generation, or a new one."""
        latest = self._latest_batch()
        if latest is not None:
            all_members = self._all_members(latest.id)
            members = [c for c in all_members if not _is_marker(c)]
            promo = self._archive.latest_promotion()
            if any(_is_marker(c) for c in all_members):
                pass
            elif promo is not None and promo.cand.startswith(f"b{latest.id}-"):
                # Crashed after promoting: the rest of that batch lost to the winner.
                for c in members:
                    if c.status is Status.ACTIVE:
                        self._archive.finish_candidate(c.id, Status.REJECTED, stage=Stage.PROMOTE)
            elif latest.gen == last_gen + 1 and (
                len(members) < self._cfg.candidates_per_gen
                or any(c.status is Status.ACTIVE for c in members)
            ):
                if latest.pack_hash == self._pack.content_hash:
                    return latest
                self._close(
                    latest,
                    champ,
                    f"pack-changed: batch pinned to {latest.pack_hash}, "
                    f"current pack is {self._pack.content_hash}",
                )
        return self._archive.create_batch(last_gen + 1, self._pack.content_hash)

    def _reconcile_promotion(self) -> None:
        """Finish a promotion whose git half happened but whose DB write was lost.

        Runs before the pack check: the evidence is in the archived runs, which do not depend on
        the pack on disk."""
        latest = self._latest_batch()
        if latest is None or self._archive.get_promotion(latest.gen) is not None:
            return
        pending = [
            c
            for c in self._members(latest.id)
            if c.status is Status.ACTIVE and c.stage is Stage.PROMOTE
        ]
        if not pending:
            return
        for c in pending:
            try:
                commit = find_promoted_commit(self._repo, c, latest.gen)
            except GitRefError:
                continue
            if commit is not None and c.parent is not None:
                sel = self._decide(latest.id, c.parent, c.id, Split.SELECT)
                conf = self._decide(latest.id, c.parent, c.id, Split.HOLDOUT)
                self._archive.record_promotion(
                    Promotion(latest.gen, c.id, sel.delta, sel.p, conf.delta, conf.p)
                )
                return
        if gen_tag_sha(self._repo, latest.gen) is not None:
            raise BatchAborted(
                f"tag iterant/gen-{latest.gen} exists but belongs to none of batch "
                f"{latest.id}'s candidates; delete the tag or fix the archive by hand, then rerun"
            )

    def _repair_spend(self, batch: Batch) -> None:
        """A crash between ``insert_run`` and ``add_spend`` leaves a candidate's spend short;
        bring it up to what its run rows say."""
        for c in self._members(batch.id):
            if c.status is not Status.ACTIVE:
                continue
            runs = self._archive.runs_for_candidate(c.id, batch=batch.id, latest_only=False)
            short = sum(r.usd_micros for r in runs) - c.usd_micros
            if short > 0:
                self._archive.add_spend(c.id, short)

    # --- spend (from the proxy's archived calls) --------------------------------------

    def _worst(self, role: Role) -> int:
        cap = self._proxy.config.policy.roles.get(role)
        return (cap.max_usd_micros_per_run or 0) if cap else 0

    def _gen_call_ids(self, gen: int) -> list[tuple[int, int]]:
        """``(call id, usd)`` of every proxy call of generation ``gen``: runs of its batches,
        plus calls outside any run made since its first batch started."""
        con = open_archive(archive_path(self._repo))
        try:
            batches = con.execute(
                "SELECT started_at FROM batches WHERE gen = ? ORDER BY id", (gen,)
            ).fetchall()
            if not batches:
                return []
            # proxy timestamps have whole-second precision, so compare at that precision
            since = datetime.fromisoformat(str(batches[0][0])).replace(microsecond=0)
            out = [
                (int(r[0]), int(r[1]))
                for r in con.execute(
                    "SELECT id, usd_micros FROM proxy_calls WHERE batch IN"
                    " (SELECT id FROM batches WHERE gen = ?)",
                    (gen,),
                )
            ]
            for row in con.execute(
                "SELECT id, usd_micros, created_at FROM proxy_calls WHERE batch IS NULL"
            ):
                if datetime.fromisoformat(str(row[2])) >= since:
                    out.append((int(row[0]), int(row[1])))
            return out
        finally:
            con.close()

    def _gen_spend(self, gen: int) -> int:
        return sum(usd for _id, usd in self._gen_call_ids(gen))

    def _gen_calls(self, gen: int) -> list[ProxyCall]:
        calls = (self._archive.get_proxy_call(i) for i, _ in self._gen_call_ids(gen))
        return [c for c in calls if c is not None]

    # --- PROPOSE --------------------------------------------------------------------

    def _redactions(self) -> list[str]:
        """Everything the failure digest must strip: hidden check paths and contents."""
        out: set[str] = set()
        for t in self._pack.tasks:
            check = Path(str(t.check))
            out.add(check.as_posix())
            if self._pack_rel:
                out.add(f"{self._pack_rel}/{check.as_posix()}")
            try:
                text = (self._pack_root / check).read_text(encoding="utf-8").strip()
            except (OSError, ValueError):
                continue
            if text:
                out.add(text)
        return sorted(out)

    def _propose_all(self, g: _Gen) -> None:
        for i in range(self._cfg.candidates_per_gen):
            cid = f"b{g.batch.id}-{i}"
            if self._archive.get_candidate(cid) is None:
                g.budget.check(self._worst(Role.PROPOSER))
                self._propose_one(g, cid)

    def _add(
        self, g: _Gen, cid: str, diff: str, hypothesis: str, stage: Stage, status: Status
    ) -> None:
        sha = self._store.put(diff) if diff else diff_sha256(b"")
        self._archive.insert_candidate(
            Candidate(cid, g.champ.id, sha, hypothesis, stage, status, 0)
        )

    def _propose_one(self, g: _Gen, cid: str) -> None:
        if g.digest is None:
            g.budget.check(self._worst(Role.ANALYST))
            g.digest = self._digest(self._redactions())
        with tempfile.TemporaryDirectory(prefix="iterant-propose-") as tmp:
            full = export_commit(self._repo, g.champ_sha, Path(tmp) / "full")
            work = proposer_view(full, self._pack_rel, Path(tmp) / "work")
            try:
                proposal = self._proposer.propose(g.champ.id, work, g.digest, self._history())
            except Exception as exc:
                msg = f"proposer failed: {type(exc).__name__}"
                self._add(g, cid, "", msg, Stage.PROPOSE, Status.ERROR)
                return
        if proposal is None:
            self._add(g, cid, "", "no diff", Stage.PROPOSE, Status.REJECTED)
            return
        lines = len(proposal.diff.splitlines())
        if lines > self._cfg.thresholds.max_diff_lines:
            g.notes[cid] = [f"diff has {lines} lines (max {self._cfg.thresholds.max_diff_lines})"]
            self._add(g, cid, proposal.diff, proposal.hypothesis, Stage.PROPOSE, Status.REJECTED)
            return
        self._add(g, cid, proposal.diff, proposal.hypothesis, Stage.GATE, Status.ACTIVE)

    def _history(self) -> list[TriedCandidate]:
        return [
            TriedCandidate(c, self._store.get(c.diff_sha))
            for c in self._archive.iter_candidates()
            if c.parent is not None and not _is_marker(c)
        ]

    # --- GATE -----------------------------------------------------------------------

    def _active(self, g: _Gen, stage: Stage) -> list[Candidate]:
        return [
            c for c in self._members(g.batch.id) if c.status is Status.ACTIVE and c.stage is stage
        ]

    def _gate_phase(self, g: _Gen) -> None:
        for c in self._active(g, Stage.GATE):
            g.budget.check(self._worst(Role.AUDITOR))
            proposal = Proposal(self._store.get(c.diff_sha), clean_hypothesis(c.hypothesis))
            try:
                with tempfile.TemporaryDirectory(prefix="iterant-gate-") as tmp:
                    repo = export_commit(self._repo, g.champ_sha, Path(tmp) / "repo")
                    result = self._gate.check(
                        proposal, repo=repo, target=self._target, pack=self._pack
                    )
            except Exception as exc:
                g.notes[c.id] = [f"gate failed: {type(exc).__name__}"]
                self._archive.finish_candidate(c.id, Status.ERROR, stage=Stage.GATE)
                continue
            if result.passed:
                self._archive.advance_stage(c.id, Stage.SMOKE)
            else:
                g.notes[c.id] = list(result.findings)
                self._archive.finish_candidate(c.id, Status.REJECTED, stage=Stage.GATE)

    # --- SMOKE ----------------------------------------------------------------------

    def _smoke_phase(self, g: _Gen) -> None:
        cands = self._with_images(g, self._active(g, Stage.SMOKE))
        if not cands:
            return
        tasks = tasks_in(self._pack, Split.TRAIN)[: self._cfg.thresholds.smoke_tasks]
        self._run_slots(g, tasks, 1, [g.champ.id, *(c.id for c in cands)], Split.TRAIN)
        all_faults = self._outcomes.faults_by_cand(g.batch.id)  # one query for the phase
        champ_faults = all_faults.get(g.champ.id, {})
        ids = {t.id for t in tasks}
        recorded = self._outcomes.recorded_by_cand(g.batch.id)  # one query for the phase
        for c in cands:
            # §3: kill on any crash or timeout the champion didn't have on the same task.
            mine = dict(all_faults.get(c.id, {}))
            # fail closed: a slot that ran but has no outcome record counts as a fault
            for task in tasks:
                if (task.id, 0) not in recorded.get(c.id, set()):
                    mine[(task.id, 0)] = "no-record"
            new = sorted(
                f"{task}: {fault}"
                for (task, rep), fault in mine.items()
                if task in ids and (task, rep) not in champ_faults
            )
            if new:
                g.notes[c.id] = [f"smoke: {', '.join(new)}"]
                self._archive.finish_candidate(c.id, Status.REJECTED, stage=Stage.SMOKE)
            else:
                self._archive.advance_stage(c.id, Stage.SELECT)

    # --- SELECT, CONFIRM ------------------------------------------------------------

    def _select_phase(self, g: _Gen) -> None:
        t = self._cfg.thresholds
        cands = self._with_images(g, self._active(g, Stage.SELECT))
        if not cands:
            return
        tasks = tasks_in(self._pack, Split.SELECT)[: t.select_tasks]
        arms = [g.champ.id, *(c.id for c in cands)]
        self._run_slots(g, tasks, t.select_reps, arms, Split.SELECT)
        for c in cands:
            self._judge(g, c, Split.SELECT, Stage.SELECT, Stage.CONFIRM)

    def _confirm_phase(self, g: _Gen) -> None:
        t = self._cfg.thresholds
        cands = self._with_images(g, self._active(g, Stage.CONFIRM))
        if not cands:
            return
        self._check_holdout_limit(g.batch, t.holdout_query_max)
        tasks = tasks_in(self._pack, Split.HOLDOUT)[: t.holdout_tasks]
        arms = [g.champ.id, *(c.id for c in cands)]
        self._run_slots(g, tasks, t.holdout_reps, arms, Split.HOLDOUT)
        if self._archive.holdout_warning_due(t.holdout_query_warn):
            g.warnings.append(
                f"holdout has been queried {self._archive.holdout_query_count()} times "
                f"(warn after {t.holdout_query_warn}): add fresh holdout tasks"
            )
        for c in cands:
            self._judge(g, c, Split.HOLDOUT, Stage.CONFIRM, Stage.PROMOTE)

    def _check_holdout_limit(self, batch: Batch, limit: int) -> None:
        """Hard stop: refuse to start a batch that would be holdout query number ``limit + 1``.

        Checked when the batch opens (so a run at the limit pays for nothing) and again before
        CONFIRM. A batch that already ran holdout tasks is counted in ``holdout_query_count``
        and may finish. Nothing is recorded against the candidates. The count covers the whole
        archive, so the ways out are to raise ``holdout_query_max`` or to start a fresh
        archive; adding or rotating holdout tasks alone does not reset it."""
        used = self._archive.holdout_query_count()
        started = any(
            r.split is Split.HOLDOUT for r in self._archive.latest_runs(batch.id).values()
        )
        if not started and used >= limit:
            raise HoldoutLimitReached(
                f"the holdout has been queried {used} times (limit {limit}): further CONFIRM "
                "runs would overfit it. Raise --holdout-query-max to continue, or start a "
                "fresh archive with refreshed holdout tasks"
            )

    def _decide(self, batch_id: int, champ_id: str, cid: str, split: Split) -> Decision:
        runs = {
            who: self._archive.runs_for_candidate(
                who, split=split, batch=batch_id, latest_only=False
            )
            for who in (cid, champ_id)
        }
        tag = f"{self._cfg.seed}:{batch_id}:{cid}:{split.value}"
        seed = int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")
        decide = decide_select if split is Split.SELECT else decide_confirm
        return decide(runs[cid], runs[champ_id], self._cfg.thresholds, seed=seed)

    def _judge(self, g: _Gen, c: Candidate, split: Split, stage: Stage, nxt: Stage) -> None:
        d = self._decide(g.batch.id, g.champ.id, c.id, split)
        if d.accepted:
            self._archive.advance_stage(c.id, nxt)
        else:
            g.notes[c.id] = [f"{stage.value}: {line}" for line in d.detail]
            self._archive.finish_candidate(c.id, Status.REJECTED, stage=stage)

    # --- PROMOTE --------------------------------------------------------------------

    def _promote_phase(self, g: _Gen) -> None:
        ready = self._active(g, Stage.PROMOTE)
        if not ready:
            return
        scored = [
            (
                self._decide(g.batch.id, g.champ.id, c.id, Split.HOLDOUT),
                self._decide(g.batch.id, g.champ.id, c.id, Split.SELECT),
                c,
            )
            for c in ready
        ]
        confirm, select, winner = max(scored, key=lambda s: (s[0].delta, s[1].delta, s[2].id))
        gen = g.batch.gen
        try:
            commit = find_promoted_commit(self._repo, winner, gen)
            if commit is None:
                diff = self._store.get(winner.diff_sha).encode("utf-8")
                promote(
                    self._repo,
                    winner,
                    diff,
                    gen,
                    g.champ_sha,
                    [f"{PACK_HASH_TRAILER}: {g.batch.pack_hash}"],
                )
        except GitRefError as exc:
            raise BatchAborted(f"cannot promote {winner.id}: {exc}") from exc
        self._archive.record_promotion(
            Promotion(gen, winner.id, select.delta, select.p, confirm.delta, confirm.p)
        )
        for c in ready:
            if c.id != winner.id:
                g.notes[c.id] = [f"superseded by {winner.id}"]
                self._archive.finish_candidate(c.id, Status.REJECTED, stage=Stage.PROMOTE)

    # --- runs -----------------------------------------------------------------------

    def _with_images(self, g: _Gen, cands: list[Candidate]) -> list[Candidate]:
        """Build images; a candidate whose image will not build is an error, not a verdict."""
        if g.champ.id not in g.images:
            try:
                g.images[g.champ.id] = self._images.champion(g.champ_sha)
            except Exception as exc:
                raise BatchAborted(f"champion image failed: {type(exc).__name__}") from exc
        ok: list[Candidate] = []
        for c in cands:
            if c.id not in g.images:
                try:
                    diff = self._store.get(c.diff_sha)
                    g.images[c.id] = self._images.candidate(c.id, g.champ_sha, diff)
                except Exception as exc:
                    g.notes[c.id] = [f"image build failed: {type(exc).__name__}"]
                    self._archive.finish_candidate(c.id, Status.ERROR)
                    continue
            ok.append(c)
        return ok

    def _run_slots(
        self, g: _Gen, tasks: tuple[Task, ...], reps: int, arm_ids: list[str], split: Split
    ) -> None:
        """Run every missing slot, interleaving the arms task by task (provider drift cancels).

        A slot with an infra-errored or refused first try gets exactly one more attempt (§6)."""
        done = self._archive.latest_runs(g.batch.id)
        for task in tasks:
            for rep in range(reps):
                for cid in arm_ids:
                    prev = done.get((cid, task.id, rep))
                    if prev is not None and (prev.score is not None or prev.key.attempt >= 1):
                        continue
                    attempt = 0 if prev is None else 1
                    while True:
                        g.budget.check(self._worst_run)
                        key = RunKey(g.batch.id, cid, task.id, rep, attempt)
                        ran = self._exec.execute(key, task, g.images[cid])
                        run = ran.result
                        self._outcomes.record(key, ran.fault)
                        self._archive.insert_run(run)
                        if cid != g.champ.id:
                            self._archive.add_spend(cid, run.usd_micros)
                        if run.score is not None or attempt >= 1:
                            break
                        attempt += 1
        self._guard_infra(g, split)
        self._guard_champion_refusals(g, split)

    def _guard_champion_refusals(self, g: _Gen, split: Split) -> None:
        """Close the batch when the *champion* refuses too many tasks on its first try.

        Refusals score 0, so a champion that refuses most tasks lets any candidate win. Only
        SELECT and HOLDOUT count: SMOKE (TRAIN) never scores a candidate's refusals, so a
        candidate could refuse there, win the scored stages, be promoted, and then have its
        own TRAIN refusals close every later batch. First attempts, because the retry hides
        most of a real excess (ARCHITECTURE §4). Closes through the infra-abort path."""
        if split is Split.TRAIN:
            return
        firsts = [
            r
            for r in self._archive.runs_for_candidate(
                g.champ.id, split=split, batch=g.batch.id, latest_only=False
            )
            if r.key.attempt == 0
        ]
        refused = sum(1 for r in firsts if r.refusal)
        if firsts and refused / len(firsts) > self._cfg.thresholds.max_champion_refusal_rate:
            raise InfraRateExceeded(
                f"the current champion (gen-{g.batch.gen - 1}, {g.champ.id}) refuses too often "
                f"({refused} of {len(firsts)} first-attempt {split.value} runs were refusals); "
                "check the model or the proxy, or reset iterant/champion to the previous "
                "generation"
            )

    def _guard_infra(self, g: _Gen, split: Split) -> None:
        """Close the batch when the *champion* arm keeps hitting infra errors.

        Only the champion's runs count: the candidate cannot cause those. Refusals are not
        counted (a candidate steers its own); the stats' refusal guard owns them."""
        finals = [
            r
            for (cid, _t, _r), r in self._archive.latest_runs(g.batch.id).items()
            if cid == g.champ.id and r.split is split
        ]
        bad = sum(1 for r in finals if r.infra_err)
        if finals and bad / len(finals) > self._cfg.max_infra_rate:
            raise InfraRateExceeded(
                f"{bad} of {len(finals)} champion {split.value} runs ended in an infra error"
            )

    # --- result ---------------------------------------------------------------------

    def _result(self, g: _Gen, stopped: Literal["done", "budget"]) -> GenerationResult:
        members = self._all_members(g.batch.id)
        promoted = next((c.id for c in members if c.status is Status.PROMOTED), None)
        return GenerationResult(
            gen=g.batch.gen,
            batch_id=g.batch.id,
            candidates=tuple(
                CandidateOutcome(c.id, c.stage, c.status, tuple(g.notes.get(c.id, ())))
                for c in members
            ),
            promoted=promoted,
            stopped=stopped,
            spent_usd_micros=self._gen_spend(g.batch.gen),
            warnings=tuple(g.warnings),
        )
