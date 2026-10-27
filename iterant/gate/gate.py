"""GATE stage (§3): path allowlist, target lint/tests, leak grep, LLM diff audit.

Stages run in that order and stop at the first one that has findings, so a diff that
leaves the genome is never run in a sandbox and never sent to the auditor. Every finding
starts with the rule that produced it (``path-allowlist: ...``). Infrastructure failures
(no git, a broken sandbox runtime, an unreachable auditor) raise; they are not verdicts.

Checks that need the real tree run on a temporary copy of the repo with the diff applied
by ``git apply``; the repo itself is never written. The genome is checked twice: on the
paths the diff names, and on the files that actually differ after applying it.
"""

from __future__ import annotations

import filecmp
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

from iterant.contracts import (
    CodeExecPolicy,
    GateResult,
    Pack,
    Proposal,
    SandboxBackend,
    TargetConfig,
)
from iterant.gate.audit import AuditorError, AuditParseError, AuditVerdict
from iterant.gate.diff import DiffError, FileDiff, changed_line_count, parse_diff
from iterant.target.genome import Genome
from iterant.target.image import _Filter  # pyright: ignore[reportPrivateUsage]

MAX_DIFF_LINES: Final = 400  # §3 PROPOSE kill rule, enforced again here
MAX_DIFF_BYTES: Final = 128 * 1024
MAX_LINE_CHARS: Final = 2000  # a one-line 600 KB dict is still one changed line
ALLOWED_MODES: Final = frozenset({"100644", "100755"})
_SYMLINK_MODES: Final = frozenset({"120000", "160000"})
_MIN_CHECK_LINE: Final = 30  # shorter lines are too generic to count as a leak
_MIN_CHECK_FILE: Final = 20
_MIN_CHECK_VALUE: Final = 8  # decoded JSON string values shorter than this are too generic
_MAX_PER_RULE: Final = 5
_APPLY_TIMEOUT_S: Final = 60
_SKIP_DIRS: Final = (".git", ".iterant")


class GateInfraError(RuntimeError):
    """The gate could not run a check (git or the sandbox runtime is broken). Not a verdict."""


class Auditor(Protocol):
    """The diff auditor (see :class:`iterant.gate.audit.ProxyAuditor`)."""

    def audit(self, diff: str, policy: CodeExecPolicy) -> AuditVerdict: ...


@dataclass(frozen=True)
class StaticChecks:
    """The target's own lint and test commands, run in a sandbox on the patched tree.

    TODO(contract): ``iterant.toml`` has no field for these yet, so the caller supplies
    them. ``image`` is the built target image; the commands run via the shell in ``/work``.
    """

    backend: SandboxBackend
    image: str
    lint: str | None = None
    test: str | None = None
    timeout_s: float = 600.0


@dataclass(frozen=True)
class GateReport(GateResult):
    """A :class:`GateResult` that also carries non-blocking ``warnings``.

    TODO(contract): ``GateResult`` has no warnings field; the code-execution flag for
    ``advisory-forbidden`` packs travels here because nothing can enforce it.
    """

    warnings: tuple[str, ...] = ()


class DiffGate:
    """Implements :class:`iterant.contracts.Gate`.

    ``pack_root`` is the pack directory on the trusted host (for hidden-check content).
    ``protected`` adds repo paths that are never editable, beyond ``iterant.toml`` and the
    target image; pass the ``[eval] pack`` path here. Protected paths, the pack directory,
    ``.env*`` files, ``.git``/``.iterant`` at any depth and the repo's ``.containerignore``
    entries are also left out of the tree uploaded to the lint/test sandbox, so the hidden
    checks and local secrets never enter it.

    Limits, by design: the leak grep is textual, so a task id or check path assembled from
    pieces (``"alpha" + "-17"``, ``tasks/*/check``) gets past it. The auditor prompt names
    those evasions and is the backstop. Hidden-check content is matched as whole files,
    long lines and decoded JSON string values of 8+ characters; shorter expected values
    are too generic to grep for.

    Threat-model rows (§9) that are not gate stages: reading hidden checks (2) is stopped by
    the grader-only mount, and the gate adds the check-path grep; forged logs or success (5)
    are stopped by scoring only from the grader and proxy, and the gate adds the auditor,
    which is told to flag tampering with logs or scores.
    """

    def __init__(
        self,
        auditor: Auditor,
        pack_root: str | os.PathLike[str],
        *,
        static: StaticChecks | None = None,
        protected: Sequence[str] = (),
        max_diff_lines: int = MAX_DIFF_LINES,
    ) -> None:
        self._auditor = auditor
        self._pack_root = Path(pack_root)
        self._static = static
        self._protected = tuple(protected)
        self._max_lines = max_diff_lines

    def check(
        self,
        proposal: Proposal,
        *,
        repo: str | os.PathLike[str],
        target: TargetConfig,
        pack: Pack,
    ) -> GateReport:
        diff = proposal.diff if proposal.diff.endswith("\n") else proposal.diff + "\n"
        try:
            size = len(diff.encode("utf-8"))
        except UnicodeEncodeError:
            return _fail("diff-malformed: diff is not valid UTF-8 text")
        if size > MAX_DIFF_BYTES:
            return _fail(f"diff-size: {size} bytes, limit is {MAX_DIFF_BYTES}")
        if any(len(line) > MAX_LINE_CHARS for line in diff.split("\n")):
            return _fail(f"diff-size: a line is longer than {MAX_LINE_CHARS} characters")
        try:
            files = parse_diff(diff)
        except DiffError as exc:
            return _fail(f"diff-malformed: {exc}")
        lines = changed_line_count(files)
        if lines > self._max_lines:
            return _fail(f"diff-size: {lines} changed lines, limit is {self._max_lines}")

        root = Path(repo)
        protected = ("iterant.toml", target.image, *self._protected)

        def genome(at: Path) -> Genome:
            return Genome(at, target.genome, protected=protected)

        if findings := _path_findings(files, genome(root)):
            return _fail(*findings)
        excluded = _excluder(root, self._hidden(root))
        with tempfile.TemporaryDirectory(prefix="iterant-gate-", ignore_cleanup_errors=True) as tmp:
            tree = Path(tmp) / "tree"
            if error := _apply(root, tree, diff, Path(tmp), excluded):
                return _fail(f"diff-apply: {error}")
            findings, changed = _tree_findings(root, tree, genome(root), genome(tree), excluded)
            if findings:
                return _fail(*findings)
            if changed == 0 and any(f.added or f.removed or f.renamed or f.copied for f in files):
                return _fail("diff-apply: the diff changed nothing in the patched tree")
            if self._static is not None and (
                findings := _static_findings(self._static, tree, excluded)
            ):
                return _fail(*findings)
        if findings := _leak_findings(files, pack, self._pack_root):
            return _fail(*findings)
        return self._audit(proposal.diff, pack)

    def _hidden(self, root: Path) -> tuple[str, ...]:
        """Repo-relative paths kept out of the sandbox tree: protected paths and the pack."""
        hidden = list(self._protected)
        try:
            hidden.append(self._pack_root.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            pass
        return tuple(h for h in hidden if h and h != ".")

    def _audit(self, diff: str, pack: Pack) -> GateReport:
        try:
            verdict = self._auditor.audit(diff, pack.code_exec_policy)
        except AuditParseError as exc:
            return _fail(f"audit-unparseable: the auditor did not return a verdict ({exc})")
        except AuditorError as exc:
            raise GateInfraError(str(exc)) from exc
        if verdict.special_casing:
            why = "; ".join(verdict.reasons) or "no reason given"
            return _fail(f"audit-special-casing: {why}")
        if pack.code_exec_policy == "advisory-forbidden" and verdict.adds_code_execution:
            why = "; ".join(verdict.reasons) or "no reason given"
            return GateReport(
                True,
                warnings=(
                    "audit-code-exec: the diff adds code execution; this pack's policy is "
                    "advisory-forbidden and unenforced, so it is not rejected. " + why,
                ),
            )
        return GateReport(True)


def _fail(*findings: str) -> GateReport:
    return GateReport(False, findings)


# --- stage 1: path allowlist --------------------------------------------------


def _path_findings(files: list[FileDiff], genome: Genome) -> list[str]:
    out: list[str] = []
    for f in files:
        name = f.new or f.old or "?"
        if f.binary:
            out.append(f"path-binary: {name}: binary changes are not allowed")
        for mode in f.modes:
            if mode in _SYMLINK_MODES:
                out.append(f"path-symlink: {name}: mode {mode} (symlink or submodule)")
            elif mode not in ALLOWED_MODES:
                out.append(f"path-mode: {name}: mode {mode} is not allowed")
        rule = "path-rename" if f.renamed or f.copied else "path-allowlist"
        for path in f.paths:
            if not (decision := genome.check(path)).ok:
                out.append(f"{rule}: {path}: {decision.reason}")
    return list(dict.fromkeys(out))


def _excluder(root: Path, hidden: Sequence[str]) -> Callable[[str], bool]:
    """Which repo-relative paths stay out of the patched tree and the sandbox upload."""
    folded = tuple(h.casefold().strip("/") for h in hidden)
    flt = _Filter(root, hidden[0] if hidden else ".iterant")

    def excluded(rel: str) -> bool:
        low = rel.casefold()
        name = low.rsplit("/", 1)[-1]
        if name == ".env" or name.startswith(".env."):
            return True
        if any(low == h or low.startswith(h + "/") for h in folded):
            return True
        return flt.excluded(rel)

    return excluded


def _apply(
    repo: Path, tree: Path, diff: str, scratch: Path, excluded: Callable[[str], bool]
) -> str:
    """Copy ``repo`` to ``tree`` and apply ``diff``. Returns an error text, or '' on success."""

    def ignore(directory: str, names: list[str]) -> list[str]:
        rel = Path(os.path.relpath(directory, repo)).as_posix()
        prefix = "" if rel == "." else rel + "/"
        return [n for n in names if excluded(prefix + n)]

    shutil.copytree(repo, tree, symlinks=True, ignore=ignore)
    patch = scratch / "candidate.patch"
    patch.write_bytes(diff.encode("utf-8"))
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        # Without a ceiling, a scratch dir inside some repo makes git apply that repo's
        # root-relative paths, skip ours and exit 0.
        "GIT_CEILING_DIRECTORIES": str(scratch),
    }
    try:
        done = subprocess.run(  # noqa: S603
            ["git", "apply", "-p1", "--whitespace=nowarn", str(patch)],  # noqa: S607
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_APPLY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateInfraError(f"git apply could not run: {exc}") from exc
    return "" if done.returncode == 0 else (done.stderr.strip() or "git apply failed")[-500:]


def _entries(root: Path, excluded: Callable[[str], bool]) -> dict[str, str]:
    """Every kept entry under ``root`` as ``{posix path: 'file' | 'dir' | 'other'}``."""
    found: dict[str, str] = {}
    for base, dirs, names in os.walk(root, followlinks=False):
        rel_base = Path(base).relative_to(root)
        dirs[:] = [d for d in dirs if not excluded((rel_base / d).as_posix())]
        for d in dirs:
            p = Path(base, d)
            found[(rel_base / d).as_posix()] = "other" if p.is_symlink() else "dir"
        for n in names:
            rel = (rel_base / n).as_posix()
            if excluded(rel):
                continue
            p = Path(base, n)
            found[rel] = "file" if p.is_file() and not p.is_symlink() else "other"
    return found


def _tree_findings(
    before: Path, after: Path, old: Genome, new: Genome, excluded: Callable[[str], bool]
) -> tuple[list[str], int]:
    """Check the files that really differ once the diff is applied (§1.1).

    Returns the findings and how many paths differ."""
    a, b = _entries(before, excluded), _entries(after, excluded)
    out: list[str] = []
    changed = 0
    for path in sorted(set(a) | set(b)):
        ka, kb = a.get(path), b.get(path)
        if ka == kb == "dir":
            continue
        if ka == kb == "file" and filecmp.cmp(before / path, after / path, shallow=False):
            continue
        if ka == kb == "other" and os.readlink(before / path) == os.readlink(after / path):
            continue
        changed += 1
        if kb == "other":
            out.append(f"path-symlink: {path}: result is a symlink or special file")
            continue
        genome = new if kb is not None else old
        if not (decision := genome.check(path)).ok:
            out.append(f"path-allowlist: {path}: {decision.reason} (after applying the diff)")
    return out, changed


# --- stage 2: the target's own lint and tests ---------------------------------


def _static_findings(
    static: StaticChecks, tree: Path, excluded: Callable[[str], bool]
) -> list[str]:
    commands = [("target-lint", static.lint), ("target-tests", static.test)]
    if not any(cmd for _, cmd in commands):
        return []
    sandbox = static.backend.create(
        static.image, 1.0, 2048, 4096, static.timeout_s * 3, "default-deny"
    )
    try:
        sandbox.put("/tmp/iterant-gate.tar", _tar(tree, excluded))
        unpack = sandbox.exec("mkdir -p /work && tar -xf /tmp/iterant-gate.tar -C /work", 120)
        if unpack.exit_code != 0 or unpack.sandbox_lost:
            raise GateInfraError(
                f"could not unpack the tree in the sandbox: {unpack.stderr[-300:]}"
            )
        for rule, cmd in commands:
            if not cmd:
                continue
            res = sandbox.exec(f"cd /work && {cmd}", static.timeout_s)
            if res.timed_out or res.truncated or res.sandbox_lost or res.exit_code != 0:
                how = (
                    "timed out"
                    if res.timed_out
                    else "was lost"
                    if res.sandbox_lost
                    else "output was cut off"
                    if res.truncated
                    else f"exited {res.exit_code}"
                )
                # Sandbox output is deliberately not echoed: findings flow back toward the
                # proposer, and the output is candidate-influenced text.
                return [f"{rule}: `{cmd}` {how}"]
        return []
    finally:
        sandbox.destroy()


def _tar(tree: Path, excluded: Callable[[str], bool]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, kind in sorted(_entries(tree, excluded).items()):
            if kind == "file":
                info = tar.gettarinfo(str(tree / path), arcname=path)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(tree / path, "rb") as fh:
                    tar.addfile(info, fh)
    return buf.getvalue()


# --- stage 3: grep for leaks --------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split())


def _leak_findings(files: list[FileDiff], pack: Pack, pack_root: Path) -> list[str]:
    """Task IDs, check paths and hidden-check content in the paths or added lines."""
    paths = "\n".join(p.replace("\\", "/") for f in files for p in f.paths)
    added = [line.replace("\\", "/") for f in files for line in f.added]
    haystack = (paths + "\n" + "\n".join(added)).casefold()
    out: list[str] = []
    for task in pack.tasks:
        check_path = str(task.check).casefold()
        if check_path in haystack:
            out.append(f"leak-check-path: names the hidden check path {task.check}")
        elif _word(task.id).search(haystack) and not (task.id.isdigit() and len(task.id) < 4):
            out.append(f"leak-task-id: mentions task id {task.id!r}")
    if out:
        return out[:_MAX_PER_RULE]
    lines = {_norm(line) for line in added}
    raw_blob = "\n".join(added)
    blob = _norm(raw_blob)
    for task in pack.tasks:
        content = (pack_root / str(task.check)).read_text(encoding="utf-8", errors="replace")
        whole = _norm(content)
        long_lines = {s for s in map(_norm, content.splitlines()) if len(s) >= _MIN_CHECK_LINE}
        values = _json_strings(content)
        if (
            (len(whole) >= _MIN_CHECK_FILE and whole in blob)
            or lines & long_lines
            or any(v in raw_blob for v in values)
        ):
            out.append(f"leak-check-content: copies hidden check content of task {task.id!r}")
    return out[:_MAX_PER_RULE]


def _json_strings(text: str) -> set[str]:
    """String values (not keys) of ``text`` parsed as JSON, 8+ characters; empty if not JSON."""
    try:
        data: object = json.loads(text)
    except ValueError:
        return set()
    found: set[str] = set()
    stack: list[object] = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if len(item) >= _MIN_CHECK_VALUE:
                found.add(item)
        elif isinstance(item, list):
            stack.extend(item)  # pyright: ignore[reportUnknownArgumentType]
        elif isinstance(item, dict):
            stack.extend(item.values())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return found


_WORDS: dict[str, re.Pattern[str]] = {}


def _word(task_id: str) -> re.Pattern[str]:
    if task_id not in _WORDS:
        _WORDS[task_id] = re.compile(
            r"(?<![a-z0-9_])" + re.escape(task_id.casefold()) + r"(?![a-z0-9_])"
        )
    return _WORDS[task_id]


__all__ = [
    "Auditor",
    "DiffGate",
    "GateInfraError",
    "GateReport",
    "StaticChecks",
]
