"""One agent run (§3, §6): sandbox from the target image, task in, agent out through the proxy,
``/out`` read back as data, then the grader scores it in its own fresh sandbox.

Who is to blame decides the result:

* The candidate's doing scores 0.0: ``ExecResult.sandbox_lost``, a ``SandboxLost`` from
  ``exec``/``get`` afterwards, a timeout, truncated output, and ``/out`` that breaks the
  invariant 8 limits (links, special files, too big, unlistable).
* A refusal (the proxy saw one) is ``refusal=True``, no score.
* Everything else that goes wrong (sandbox create/put/get failures, grader infra errors) is
  ``infra_err=True``, no score. The caller retries once, then drops the task pair.

Spend and tokens come from the proxy ledger, never from the agent (invariant 2).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable

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
from iterant.sandbox import SandboxLost
from iterant.sandbox.container import PROXY_ENV

_LIST_TIMEOUT_S = 30.0
_TTL_MARGIN_S = 120.0
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


def read_out(sandbox: Sandbox, out_dir: str) -> dict[str, bytes]:
    """Read the agent's ``/out`` as plain data under the invariant 8 limits.

    Lists with ``find`` (types, link counts and sizes are checked before anything is read),
    then fetches regular single-link files one by one. Raises ``OutputRejected`` when the
    output breaks a rule or cannot be listed (the agent's doing), ``SandboxLost`` or another
    ``SandboxError`` from the sandbox itself.
    """
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
        if kind != "f" or links != "1" or not size_text.isdigit() or not rel:
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
        try:
            files[rel] = sandbox.get(f"{out_dir}/{rel}")
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise OutputRejected("output changed while reading") from exc
    return files


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
    ) -> None:
        self._backend = backend
        self._grader = grader
        self._proxy = proxy
        self._run = run_command
        self._read_input = read_input
        self._cfg = config

    def execute(self, key: RunKey, task: Task, image: str) -> RunResult:
        start = time.monotonic()
        cfg = self._cfg

        def finish(
            score: float | None, *, infra: bool = False, wall: float | None = None
        ) -> RunResult:
            usage = self._proxy.run_usage(key)
            refused = usage.refusals > 0
            if refused:
                score, infra = None, False
            elif score is None:
                infra = True
            return RunResult(
                key=key,
                split=task.split,
                score=score,
                in_tok=usage.in_tok,
                out_tok=usage.out_tok,
                usd_micros=usage.usd_micros,
                wall_s=wall if wall is not None else time.monotonic() - start,
                refusal=refused,
                infra_err=infra,
            )

        try:
            input_bytes = self._read_input(task)
            token = self._proxy.issue_run_token(Role.TARGET, key, cfg.run_timeout_s + _TTL_MARGIN_S)
            sandbox = self._backend.create(
                image, cfg.cpu, cfg.mem, cfg.disk, cfg.run_timeout_s + _TTL_MARGIN_S, "proxy-only"
            )
        except Exception:
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
                if _candidate_failed(res):
                    return finish(0.0, wall=agent_wall)
                files = read_out(sandbox, cfg.out_dir)
            except (OutputRejected, SandboxLost):
                return finish(0.0, wall=agent_wall)
            except Exception:
                return finish(None, infra=True, wall=agent_wall)
        finally:
            self._proxy.sessions.revoke_run(key)
            try:
                sandbox.destroy()
            except Exception:  # best-effort; the backend ttl reaps it
                pass
        if self._proxy.run_usage(key).refusals > 0:
            return finish(None, wall=agent_wall)  # refused: no point grading
        graded = self._grader.grade(task, files)
        return finish(graded.score, wall=agent_wall)


def _candidate_failed(res: ExecResult) -> bool:
    return res.sandbox_lost or res.timed_out or res.truncated
