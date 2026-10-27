"""One agent run (§3, §6): sandbox from the target image, task in, agent out through the proxy,
``/out`` read back as data, then the grader scores it in its own fresh sandbox.

Who is to blame decides the result:

* The candidate's doing scores 0.0: ``ExecResult.sandbox_lost``, a ``SandboxLost`` from
  ``exec``/``get`` afterwards, a timeout, truncated output, ``/out`` that breaks the invariant 8
  limits (links, special files, too big, unlistable, too slow to read), and any ``get`` failure
  after the agent ran while the sandbox still answers (``chmod 000``, a FIFO, a path too deep).
* A refusal (the proxy saw one) is ``refusal=True``, no score.
* Everything else that goes wrong (sandbox create/put failures, a runtime that stops answering
  while ``/out`` is read, grader infra errors) is ``infra_err=True``, no score. The caller
  retries once, then drops the task pair.

Spend and tokens come from the proxy ledger, never from the agent (invariant 2).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from iterant.contracts import (
    ExecResult,
    Role,
    RunKey,
    RunResult,
    Sandbox,
    SandboxBackend,
    Task,
)
from iterant.grader import OutputRejected
from iterant.grader.outfiles import (
    MAX_OUT_ENTRIES,
    MAX_OUT_FILE_BYTES,
    MAX_OUT_FILES,
    MAX_OUT_TOTAL_BYTES,
)
from iterant.orchestrator.config import GraderLike, OrchestratorConfig
from iterant.proxy import Proxy
from iterant.sandbox import SandboxError, SandboxLost
from iterant.sandbox.container import PROXY_ENV

_LIST_TIMEOUT_S = 30.0
_PROBE_TIMEOUT_S = 10.0
# A get stalled on a FIFO ends at the container backend's 120 s IO timeout; the margin keeps
# the sandbox alive past that, plus the probe and some slack.
_IO_TIMEOUT_S = 120.0
_TTL_SLACK_S = 5.0  # treat a sandbox this close to its TTL as already expired
_TTL_MARGIN_S = _IO_TIMEOUT_S + _PROBE_TIMEOUT_S + 60.0
# One record per entry: type, hard-link count, size, path relative to the out dir, NUL-ended.
_LIST_CMD = r"find {out} -mindepth 1 -printf '%y\t%n\t%s\t%P\0'"
_TASK_REF = re.compile(r"(?<![\w./-])/task(?=/)")
_OUT_REF = re.compile(r"(?<![\w./-])/out(?![\w./-])")


def bind_paths(run: str, task_dir: str, out_dir: str) -> str:
    """Point the target's ``run`` command (written for ``/task`` and ``/out``, §2) at the
    sandbox's writable directories."""
    return _OUT_REF.sub(out_dir, _TASK_REF.sub(task_dir, run))


def agent_command(run: str, token: str, cfg: OrchestratorConfig) -> str:
    """The shell line that runs the agent with only a session token, never the real key."""
    return (
        f"export ANTHROPIC_BASE_URL=\"${{{PROXY_ENV}}}\" ANTHROPIC_API_KEY='{token}'; "
        f"mkdir -p {cfg.out_dir} && {bind_paths(run, cfg.task_dir, cfg.out_dir)}"
    )


def sandbox_ttl(cfg: OrchestratorConfig) -> float:
    """The agent sandbox must outlive the agent run, the listing and the whole read budget, so
    an expired sandbox (an infra error) can never be the agent's doing."""
    return cfg.run_timeout_s + _LIST_TIMEOUT_S + cfg.read_budget_s + _TTL_MARGIN_S


def read_out(
    sandbox: Sandbox,
    out_dir: str,
    *,
    budget_s: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, bytes]:
    """Read the agent's ``/out`` as plain data under the invariant 8 limits.

    Lists with ``find`` (types, link counts and sizes are checked before anything is read),
    then fetches regular single-link files one by one within ``budget_s``. Raises
    ``OutputRejected`` when the output breaks a rule, cannot be read, or takes too long (the
    agent's doing); ``SandboxLost`` if the sandbox is gone; any other ``SandboxError`` is
    re-raised for the caller to judge: only a fresh sandbox, out of the agent's reach, can say
    whether the runtime is down or the agent broke this one.
    """
    deadline = clock() + budget_s
    listing = sandbox.exec(_LIST_CMD.format(out=out_dir), _LIST_TIMEOUT_S)
    if listing.timed_out or listing.truncated or listing.sandbox_lost or listing.exit_code != 0:
        raise OutputRejected("output cannot be listed")
    entries = [e for e in listing.stdout.split("\0") if e]
    if len(entries) > MAX_OUT_ENTRIES:
        raise OutputRejected("too many output files")
    paths: list[str] = []
    total = 0
    for entry in entries:
        parts = entry.split("\t", 3)
        if len(parts) != 4:
            raise OutputRejected("unreadable output listing")
        kind, links, size_text, rel = parts
        if kind == "d":
            continue
        ascii_size = size_text.isascii() and size_text.isdigit()  # int() accepts other digits
        if kind != "f" or links != "1" or not ascii_size or not rel:
            raise OutputRejected("output contains a link or special file")
        size = int(size_text)
        if size > MAX_OUT_FILE_BYTES:
            raise OutputRejected("output file too large")
        total += size
        paths.append(rel)
    if len(paths) > MAX_OUT_FILES:
        raise OutputRejected("too many output files")
    if total > MAX_OUT_TOTAL_BYTES:
        raise OutputRejected("output too large")
    files: dict[str, bytes] = {}
    for rel in paths:
        if clock() > deadline:
            raise OutputRejected("output took too long to read")
        try:
            files[rel] = sandbox.get(f"{out_dir}/{rel}")
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise OutputRejected("output changed while reading") from exc
        except SandboxError:
            raise
    return files


@dataclass(frozen=True)
class Executed:
    """A run's result plus how the agent ended, for the SMOKE rule. ``fault`` is None unless
    the run crashed (non-zero exit), timed out, lost its sandbox or left unusable output."""

    result: RunResult
    fault: str | None


class RunExecutor:
    """Runs and scores single attempts. Stateless between calls."""

    def __init__(
        self,
        *,
        backend: SandboxBackend,
        grader: GraderLike,
        proxy: Proxy,
        run_command: str,
        read_input: Callable[[Task], bytes],
        config: OrchestratorConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._backend = backend
        self._grader = grader
        self._proxy = proxy
        self._run = run_command
        self._read_input = read_input
        self._cfg = config

    def _runtime_healthy(self, image: str) -> bool:
        """Judge the runtime from outside the agent's reach: a FRESH sandbox from the same
        image must start and run a trivial command."""
        cfg = self._cfg
        try:
            probe = self._backend.create(
                image, cfg.cpu, cfg.mem, cfg.disk, _PROBE_TIMEOUT_S + 50.0, "default-deny"
            )
        except Exception:
            return False
        try:
            res = probe.exec("true", _PROBE_TIMEOUT_S)
            return res.exit_code == 0 and not (res.timed_out or res.sandbox_lost or res.truncated)
        except Exception:
            return False
        finally:
            try:
                probe.destroy()
            except Exception:
                pass

    def execute(self, key: RunKey, task: Task, image: str) -> Executed:
        start = time.monotonic()
        cfg = self._cfg
        before = self._proxy.run_usage(key)  # a key re-run in this process must not double count

        def finish(
            score: float | None,
            *,
            infra: bool = False,
            wall: float | None = None,
            fault: str | None = None,
        ) -> Executed:
            now = self._proxy.run_usage(key)
            refused = now.refusals > before.refusals
            if refused:
                score, infra, fault = None, False, None
            elif score is None:
                infra = True
            return Executed(
                RunResult(
                    key=key,
                    split=task.split,
                    score=score,
                    in_tok=now.in_tok - before.in_tok,
                    out_tok=now.out_tok - before.out_tok,
                    usd_micros=now.usd_micros - before.usd_micros,
                    wall_s=wall if wall is not None else time.monotonic() - start,
                    refusal=refused,
                    infra_err=infra,
                ),
                fault,
            )

        token: str | None = None
        try:
            input_bytes = self._read_input(task)
            token = self._proxy.issue_run_token(Role.TARGET, key, sandbox_ttl(cfg))
            sandbox = self._backend.create(
                image, cfg.cpu, cfg.mem, cfg.disk, sandbox_ttl(cfg), "proxy-only"
            )
            created = self._clock()
        except Exception:
            if token is not None:
                self._proxy.sessions.revoke_run(key)
            return finish(None, infra=True)
        files: dict[str, bytes] | None = None
        agent_wall: float | None = None
        try:
            try:
                sandbox.put(f"{cfg.task_dir}/input.json", input_bytes)
            except Exception:
                return finish(None, infra=True)
            try:
                res = sandbox.exec(agent_command(self._run, token, cfg), cfg.run_timeout_s)
                agent_wall = res.wall_s
                bad = _fault(res)
                if bad is not None:
                    return finish(0.0, wall=agent_wall, fault=bad)
                files = read_out(sandbox, cfg.out_dir, budget_s=cfg.read_budget_s)
            except OutputRejected:
                return finish(0.0, wall=agent_wall, fault="bad-output")
            except SandboxLost:
                return finish(0.0, wall=agent_wall, fault="lost")
            except SandboxError:
                expired = self._clock() - created >= sandbox_ttl(cfg) - _TTL_SLACK_S
                if expired:
                    # the backend may have reaped it: an expired sandbox is never the agent's doing
                    return finish(None, infra=True, wall=agent_wall)
                if agent_wall is not None and self._runtime_healthy(image):
                    # the agent ran and a fresh sandbox is fine, so it broke this one
                    # (a delayed fork bomb, chmod 000, a FIFO ...): the candidate's doing
                    return finish(0.0, wall=agent_wall, fault="bad-output")
                return finish(None, infra=True, wall=agent_wall)
            except Exception:
                # not a sandbox problem at all: an orchestrator bug, never the candidate's fault
                return finish(None, infra=True, wall=agent_wall)
        finally:
            self._proxy.sessions.revoke_run(key)
            try:
                sandbox.destroy()
            except Exception:  # best-effort; the backend ttl reaps it
                pass
        if self._proxy.run_usage(key).refusals > before.refusals:
            return finish(None, wall=agent_wall)  # refused: no point grading
        graded = self._grader.grade(task, files)
        return finish(graded.score, wall=agent_wall, fault="crash" if res.exit_code != 0 else None)


def _fault(res: ExecResult) -> str | None:
    if res.sandbox_lost:
        return "lost"
    if res.timed_out:
        return "timeout"
    if res.truncated:
        return "truncated"
    return None
