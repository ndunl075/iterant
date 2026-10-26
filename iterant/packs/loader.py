"""Task-pack loader (§2).

Layout::

    pack.toml                 name = "...", version = "...", grader_image = "name@sha256:<hex>",
                              optional [policy] agent_code_exec_policy = "allowed" |
                              "advisory-forbidden" (default "allowed")
    splits.json               seed + {task id: split}; written by ``split_pack``
    tasks/<id>/input.json     visible to the agent
    tasks/<id>/run            shell script run by the grader in a sandbox (cwd /out): it executes
                              the agent output and prints one JSON object. No expected values.
    tasks/<id>/check          hidden JSON: the exact object `run` must print. Read only by the
                              grader process, never copied into a sandbox.
    tasks/<id>/task.toml      optional: tags = ["a", "b"]

Any other files (LICENSE, NOTICE, README) are allowed and are covered by the hash.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, cast

from iterant.contracts import (
    CODE_EXEC_POLICIES,
    AcceptanceThresholds,
    CodeExecPolicy,
    HiddenCheckPath,
    Pack,
    Split,
    Task,
    VisibleInputPath,
    require_pinned_image,
)
from iterant.packs.hashing import PackError, compute_content_hash
from iterant.packs.splitter import (
    SPLITS_FILE,
    SplitReport,
    assign_splits,
    parse_splits,
    write_splits,
)

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PackError(f"cannot read {path.name}: {exc}") from exc


def _discover(root: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Every task directory as ``(id, tags)``, sorted by id."""
    tasks_dir = root / "tasks"
    if not tasks_dir.is_dir():
        raise PackError(f"{root}: no tasks/ directory")
    found: list[tuple[str, tuple[str, ...]]] = []
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            raise PackError(f"unexpected file in tasks/: {entry.name}")
        if not _ID.fullmatch(entry.name):
            raise PackError(f"bad task id {entry.name!r}")
        for required in ("input.json", "run", "check"):
            if not (entry / required).is_file():
                raise PackError(f"task {entry.name}: missing {required}")
        tags: tuple[str, ...] = ()
        if (entry / "task.toml").is_file():
            raw = _toml(entry / "task.toml").get("tags", [])
            if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):  # pyright: ignore[reportUnknownVariableType]
                raise PackError(f"task {entry.name}: tags must be a list of strings")
            tags = tuple(cast("list[str]", raw))
        found.append((entry.name, tags))
    if not found:
        raise PackError(f"{root}: pack has no tasks")
    return found


def split_pack(
    root: Path,
    thresholds: AcceptanceThresholds | None = None,
    seed: int = 0,
) -> SplitReport:
    """Assign splits for the tasks in ``root`` and write ``splits.json``.

    Overwrites an existing assignment, which changes the pack hash: do this when
    authoring a pack, never on a pack that is already pinned.
    """
    plan = assign_splits(_discover(root), thresholds, seed)
    write_splits(root, plan)
    return plan.report


def _code_exec_policy(table: object) -> CodeExecPolicy:
    fields = cast("dict[str, Any]", table) if isinstance(table, dict) else None
    if fields is None or set(fields) - {"agent_code_exec_policy"}:
        raise PackError("[policy] may only contain `agent_code_exec_policy`")
    value = fields.get("agent_code_exec_policy", "allowed")
    if value not in CODE_EXEC_POLICIES:
        raise PackError(f"agent_code_exec_policy must be one of {', '.join(CODE_EXEC_POLICIES)}")
    return value


def load_pack(root: Path) -> Pack:
    """Load and validate the pack at ``root``; ``content_hash`` is computed from disk."""
    meta = _toml(root / "pack.toml") if (root / "pack.toml").is_file() else None
    if meta is None:
        raise PackError(f"{root}: missing pack.toml")
    name, version = meta.get("name"), meta.get("version")
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise PackError("pack.toml needs non-empty string `name` and `version`")

    image = meta.get("grader_image")
    if not isinstance(image, str):
        raise PackError("pack.toml needs a string `grader_image` (name@sha256:<64 hex>)")
    try:
        require_pinned_image(image)
    except ValueError as exc:
        raise PackError(str(exc)) from exc
    policy = _code_exec_policy(meta.get("policy", {}))

    entries = _discover(root)
    if not (root / SPLITS_FILE).is_file():
        raise PackError(f"{root}: missing {SPLITS_FILE}; run split_pack first")
    assignment = parse_splits((root / SPLITS_FILE).read_bytes())
    if set(assignment) != {i for i, _ in entries}:
        raise PackError(f"{SPLITS_FILE} does not list exactly the tasks in tasks/")

    tasks = tuple(
        Task(
            id=task_id,
            split=assignment[task_id],
            tags=tags,
            input=VisibleInputPath(PurePosixPath("tasks", task_id, "input.json")),
            check=HiddenCheckPath(PurePosixPath("tasks", task_id, "check")),
        )
        for task_id, tags in entries
    )
    return Pack(
        name=name,
        version=version,
        tasks=tasks,
        content_hash=compute_content_hash(root),
        grader_image=image,
        code_exec_policy=policy,
    )


def tasks_in(pack: Pack, split: Split) -> tuple[Task, ...]:
    """The pack's tasks in ``split``, in id order."""
    return tuple(t for t in pack.tasks if t.split is split)
