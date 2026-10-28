"""The editing agent: a tool-use loop over the Messages API, through the proxy (proposer role).

Tools: ``list_files``, ``read_file``, ``write_file``, ``finish``. Writes are checked against the
genome by ``Workspace``; the returned diff covers exactly the files written. Each model call
is one proxy call and nudges count within ``max_turns``, so a proposer run makes at most
``max_turns`` calls; the CLI issues a fresh single-use call token per call.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from iterant.analyst.client import MessagesClient, reply_text
from iterant.contracts import Effort, IterantConfig, Proposal, TriedCandidate, clean_hypothesis
from iterant.proposer.workspace import ToolError, Workspace
from iterant.proxy.jsonutil import Obj, arr, obj, text
from iterant.target.genome import genome_for

PROMPT_PATH = Path(__file__).with_name("prompt.md")
MAX_HISTORY_CHARS = 20_000
MAX_DIFF_LINES_SHOWN = 60
MAX_HYPOTHESIS_CHARS = 300
MAX_CONTEXT_CHARS = 400_000
MAX_NUDGES = 2
NUDGE = (
    "You replied without calling a tool. Use the write_file tool to edit one of the editable "
    "files listed above; reply with tool calls only."
)

TOOLS: list[Obj] = [
    {
        "name": "list_files",
        "description": "List files in the repo, optionally under a directory prefix.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
    {
        "name": "read_file",
        "description": "Read one UTF-8 text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite one editable file with the full new content. "
            "Creating a new file needs no read; to overwrite an existing file, "
            "read_file it first or the write is refused."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "finish",
        "description": "Finish. Give a one-line hypothesis for the change.",
        "input_schema": {
            "type": "object",
            "properties": {"hypothesis": {"type": "string"}},
            "required": ["hypothesis"],
        },
    },
]


def render_history(history: Sequence[TriedCandidate]) -> str:
    """Tried candidates with outcome and diff, newest kept first when over the size cap."""
    if not history:
        return "Nothing has been tried yet."
    entries: list[str] = []
    for t in history:
        c = t.candidate
        diff_lines = t.diff.splitlines()
        shown = "\n".join(diff_lines[:MAX_DIFF_LINES_SHOWN])
        more = len(diff_lines) - MAX_DIFF_LINES_SHOWN
        cut = f"\n... ({more} more diff lines)" if more > 0 else ""
        entries.append(
            f"- {c.id}: {c.hypothesis} -> {c.status.value} at {c.stage.value}\n"
            f"```diff\n{shown}{cut}\n```"
        )
    kept: list[str] = []
    size = 0
    for entry in reversed(entries):
        size += len(entry)
        if size > MAX_HISTORY_CHARS and kept:
            break
        kept.append(entry)
    omitted = len(entries) - len(kept)
    head = f"({omitted} older entries omitted)\n" if omitted else ""
    return head + "\n".join(reversed(kept))


def render_editable(files: Sequence[str], patterns: Sequence[str]) -> str:
    listing = "\n".join(f"- {f}" for f in files) if files else "(no existing file matches)"
    return (
        f"## Files you may edit\n{listing}\n"
        f"Allowed path patterns: {', '.join(patterns)}\n"
        "Writes to any other path are refused.\n\n"
    )


def build_user_prompt(
    parent: str,
    digest: str,
    history: Sequence[TriedCandidate],
    editable: str = "",
) -> str:
    return (
        f"Current agent: candidate {parent}.\n\n"
        f"{editable}"
        f"## How it fails on training tasks\n{digest or '(no failure digest)'}\n\n"
        f"## Already tried (do not repeat)\n{render_history(history)}\n"
    )


def one_line(s: str) -> str:
    # Control characters become spaces so the result passes ``Proposal``'s strict check.
    return clean_hypothesis(clean_hypothesis(s)[:MAX_HYPOTHESIS_CHARS])


class EditingProposer:
    """``Proposer`` implementation. ``model`` and ``effort`` must equal the proposer role cap.

    ``propose`` returns None for no diff, a model refusal, or a conversation that outgrows
    ``max_context_chars``. Running out of ``max_turns`` keeps the diff written so far. An
    ``LLMError`` from the client propagates (an infra failure); ``workdir`` may then be
    half-written, so the caller must discard it.
    """

    def __init__(
        self,
        client: MessagesClient,
        config: IterantConfig,
        *,
        model: str,
        effort: Effort,
        max_turns: int = 30,
        max_tokens: int = 16_000,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._model = model
        self._effort: Effort = effort
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._max_context = max_context_chars
        self._system = (
            system_prompt if system_prompt is not None else PROMPT_PATH.read_text(encoding="utf-8")
        )

    def propose(
        self,
        parent: str,
        workdir: str | os.PathLike[str],
        failure_digest: str,
        history: Sequence[TriedCandidate],
    ) -> Proposal | None:
        ws = Workspace(
            workdir, genome_for(workdir, self._config), hidden=(self._config.evaluation.pack,)
        )
        messages: list[Obj] = [
            {
                "role": "user",
                "content": build_user_prompt(
                    parent,
                    failure_digest,
                    history,
                    render_editable(ws.editable_files(), self._config.target.genome),
                ),
            }
        ]
        nudges = 0
        hypothesis = ""
        last_text = ""
        for _ in range(self._max_turns):
            if len(json.dumps(messages)) > self._max_context:
                return None
            resp = self._client.create(
                {
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "output_config": {"effort": self._effort},
                    "system": self._system,
                    "tools": TOOLS,
                    "messages": messages,
                }
            )
            if resp.get("stop_reason") == "refusal":
                return None
            content = arr(resp.get("content")) or []
            messages.append({"role": "assistant", "content": content})
            last_text = reply_text(resp) or last_text
            uses = [b for b in (obj(x) for x in content) if b and b.get("type") == "tool_use"]
            if not uses:
                if not ws.touched and nudges < MAX_NUDGES:
                    nudges += 1
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                break
            results: list[Obj] = []
            finished = False
            for use in uses:
                name = text(use.get("name")) or ""
                args = obj(use.get("input")) or {}
                if name == "finish":
                    hypothesis = text(args.get("hypothesis")) or ""
                    finished = True
                result: Obj = {"type": "tool_result", "tool_use_id": use.get("id")}
                try:
                    result["content"] = self._run_tool(ws, name, args)
                except ToolError as exc:
                    result["content"] = str(exc)
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})
            if finished:
                break
        diff = ws.diff()
        if not diff:
            return None
        return Proposal(diff=diff, hypothesis=one_line(hypothesis or last_text))

    @staticmethod
    def _run_tool(ws: Workspace, name: str, args: Obj) -> str:
        path = text(args.get("path")) or ""
        if name == "list_files":
            return ws.list_files(path)
        if name == "read_file":
            return ws.read(path)
        if name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                raise ToolError("write_file needs string 'content'")
            return ws.write(path, content)
        if name == "finish":
            return "done"
        raise ToolError(f"unknown tool {name!r}")
