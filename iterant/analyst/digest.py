"""Build the failure digest the proposer reads (§3 loop edge, §6 analyst row).

Inputs are ``RunResult`` rows and the proxy's call content (``Proxy.train_content``). Both
are filtered here as well, so a caller mistake cannot leak data (invariant 4): a result must
have ``split == train``, and call content is kept only when its key matches such a result.
``redact`` strings (for example hidden check text or paths) are removed from everything that
goes to the model and into the digest.

With a client, each failed run gets one analyst call (<= ~200 tokens out). Without one, or
when a call fails, is refused or returns nothing, that run gets a deterministic summary.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from iterant.analyst.client import MessagesClient, reply_text
from iterant.contracts import Effort, RunKey, RunResult, Split
from iterant.proxy.core import CallContent
from iterant.proxy.jsonutil import Obj, arr, obj, text

SYSTEM_PROMPT = (
    "You analyse one failed run of an AI agent on a task. Reply with 1-3 plain sentences "
    "(at most 200 tokens): what the agent did, and the most likely reason it failed. "
    "Describe behaviour only. Do not guess at hidden tests, graders or expected answers."
)
_REDACTED = "[redacted]"


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in arr(content) or []:
        block = obj(item)
        if block is None:
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(text(block.get("text")) or "")
        elif kind == "tool_use":
            parts.append(f"[tool_use {block.get('name')} {block.get('input')}]")
        elif kind == "tool_result":
            parts.append(f"[tool_result {_block_text(block.get('content'))}]")
    return "\n".join(p for p in parts if p)


def _clip(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = limit * 2 // 3
    tail = limit - head
    return f"{s[:head]}\n...[cut]...\n{s[len(s) - tail :]}"


def _final(results: Iterable[RunResult]) -> list[RunResult]:
    """The latest attempt per ``(batch, cand, task, rep)`` (§1.6)."""
    best: dict[tuple[int, str, str, int], RunResult] = {}
    for r in results:
        k = (r.key.batch, r.key.cand, r.key.task, r.key.rep)
        if k not in best or r.key.attempt > best[k].key.attempt:
            best[k] = r
    return list(best.values())


class FailureAnalyst:
    """``digest(results, content)`` returns the failure digest text.

    ``redact`` defaults to nothing. The caller must pass the pack's hidden check paths and
    check text (and anything else that must not reach the proposer); only those strings are
    scrubbed from model input and from the digest.
    """

    def __init__(
        self,
        client: MessagesClient | None,
        *,
        model: str,
        effort: Effort,
        redact: Sequence[str] = (),
        pass_threshold: float = 1.0,
        max_failures: int = 20,
        max_chars: int = 4000,
        excerpt_chars: int = 6000,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        # The orchestrator MUST pass hidden check paths/text here: nothing else removes them.
        self._redact = tuple(s for s in redact if s)
        self._threshold = pass_threshold
        self._max_failures = max_failures
        self._max_chars = max_chars
        self._excerpt_chars = excerpt_chars
        self._max_tokens = max_tokens

    def scrub(self, s: str) -> str:
        for needle in self._redact:
            s = s.replace(needle, _REDACTED)
        return s

    def digest(self, results: Sequence[RunResult], content: Sequence[CallContent]) -> str:
        train = _final(r for r in results if r.split is Split.TRAIN)
        keys = {r.key for r in train}
        calls: dict[RunKey, list[CallContent]] = {}
        for c in content:
            if c.key in keys:
                calls.setdefault(c.key, []).append(c)

        scored = [r for r in train if r.score is not None]
        refused = sum(1 for r in train if r.refusal)
        infra = sum(1 for r in train if r.infra_err)
        failed = sorted(
            (r for r in scored if (r.score or 0.0) < self._threshold),
            key=lambda r: (r.score or 0.0, r.key.task, r.key.cand, r.key.rep),
        )
        mean = sum(r.score or 0.0 for r in scored) / len(scored) if scored else 0.0
        lines = [
            f"Train runs: {len(train)} (scored {len(scored)}, refused {refused}, "
            f"infra errors {infra}). Mean score {mean:.2f}. Failed: {len(failed)}."
        ]
        shown = failed[: self._max_failures]
        excerpts = [self._excerpt(calls.get(r.key, [])) for r in shown]
        notes = self._notes(shown, excerpts)
        if shown:
            lines.append(f"Failures (showing {len(shown)} of {len(failed)}, worst first):")
        for r, excerpt, note in zip(shown, excerpts, notes, strict=True):
            body = note or self._fallback(excerpt)
            lines.append(f"- {r.key.task} rep{r.key.rep} score {(r.score or 0.0):.2f}: {body}")
        out = self.scrub("\n".join(lines))
        if len(out) > self._max_chars:
            out = out[: self._max_chars - 14].rstrip() + "\n[truncated]"
        return out

    # --- per-run pieces -----------------------------------------------------------

    def _excerpt(self, calls: list[CallContent]) -> str:
        """Readable transcript of one run: the task prompt, then each model reply."""
        parts: list[str] = []
        for i, c in enumerate(sorted(calls, key=lambda c: c.call_id)):
            if i == 0:
                messages = arr(c.request.get("messages")) or []
                first = obj(messages[0]) if messages else None
                if first is not None:
                    parts.append("TASK: " + _block_text(first.get("content")))
            if c.response is not None:
                parts.append("AGENT: " + _block_text(c.response.get("content")))
        return self.scrub(_clip("\n".join(parts), self._excerpt_chars))

    def _fallback(self, excerpt: str) -> str:
        replies = excerpt.split("\nAGENT: ")[1:] or [""]
        last = " ".join(replies[-1].split())
        return f"no analysis available; last agent output: {_clip(last, 200) or '(none)'}"

    def _notes(self, runs: list[RunResult], excerpts: list[str]) -> list[str]:
        if self._client is None or not runs:
            return [""] * len(runs)
        bodies: list[Obj] = [
            {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "output_config": {"effort": self._effort},
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Task {self.scrub(r.key.task)}, score {(r.score or 0.0):.2f}."
                            f"\n\n{excerpt}"
                        ),
                    }
                ],
            }
            for r, excerpt in zip(runs, excerpts, strict=True)
        ]
        notes: list[str] = []
        for resp in self._client.create_many(bodies):
            usable = resp is not None and resp.get("stop_reason") != "refusal"
            notes.append(" ".join(reply_text(resp).split()) if usable and resp else "")
        return notes
