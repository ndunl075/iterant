"""The file tools the editing agent gets: a guarded view of one target-repo copy.

Writes go through the genome allowlist (§1.1, ``iterant.target.genome``): a path outside the
genome, or one that resolves outside it through a symlink, is refused and nothing is written.
Reads are limited too: ``.git``, ``.iterant`` and the pack directory (hidden checks, and
select/holdout/test tasks) are invisible to the agent (invariant 4), so a workdir that still
holds them does not leak them.

The diff is computed from the files the agent wrote, against their content when first
touched, so it never depends on git being present.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

from iterant.target.genome import DENIED_DIRS, Genome, PathRejectedError, normalize_relative

MAX_READ_BYTES = 64 * 1024
MAX_WRITE_BYTES = 256 * 1024
MAX_LISTED = 500
MAX_EDITABLE_SHOWN = 40  # editable paths named in the prompt and in refusal messages
MAX_EDITABLE_CHARS = 2_000  # cap on the size of that listing
MAX_EDITABLE_SCAN = 20_000  # files looked at while searching for editable ones


def _lines(s: str) -> list[str]:
    """Split after each LF only, as git does; ``str.splitlines`` also breaks on FF, CR, U+2028."""
    return re.findall(r"[^\n]*\n|[^\n]+", s)


def _write_atomic(path: Path, data: bytes) -> None:
    """Temp file in the same directory, then ``os.replace``: a failure part-way leaves the
    original file untouched and no partial content behind."""
    tmp = path.with_name(f".{path.name}.iterant-{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        if path.exists():
            shutil.copymode(path, tmp)  # keep the exec bit: no mode change in the diff
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class ToolError(Exception):
    """A refused or failed tool call. ``str(error)`` goes back to the model."""


class Workspace:
    def __init__(self, root: str | os.PathLike[str], genome: Genome, hidden: Sequence[str]) -> None:
        self._root = Path(root)
        self._genome = genome
        self._hidden = tuple(tuple(p.casefold() for p in h.split("/")) for h in hidden)
        self._originals: dict[str, str | None] = {}
        self._read: set[str] = set()  # files read successfully this session

    # --- helpers ------------------------------------------------------------------

    def _norm(self, path: str) -> str:
        try:
            return normalize_relative(path)
        except PathRejectedError as exc:
            raise ToolError(f"refused: {exc}") from exc

    def _is_hidden(self, parts: Sequence[str]) -> bool:
        folded = tuple(p.casefold() for p in parts)
        return any(folded[: len(h)] == h for h in self._hidden)

    def _readable(self, path: str) -> Path:
        norm = self._norm(path)
        if self._is_hidden(norm.split("/")):
            raise ToolError("refused: this path is not available to the proposer; call list_files")
        full = self._root / norm
        try:
            real = Path(os.path.realpath(full)).relative_to(Path(os.path.realpath(self._root)))
        except ValueError as exc:
            raise ToolError("refused: path resolves outside the repo") from exc
        if self._is_hidden(real.parts) or any(p.casefold() in DENIED_DIRS for p in real.parts):
            raise ToolError("refused: this path is not available to the proposer; call list_files")
        return full

    # --- tools --------------------------------------------------------------------

    def _walk(self, base: Path) -> Iterator[str]:
        """Visible files under ``base`` as repo-relative paths, in sorted order."""
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = Path(dirpath).relative_to(self._root).parts
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d.casefold() not in DENIED_DIRS
                and not self._is_hidden((*rel_dir, d))
                and not (Path(dirpath) / d).is_symlink()
            )
            for name in sorted(filenames):
                if not self._is_hidden((*rel_dir, name)):
                    yield "/".join((*rel_dir, name))

    def list_files(self, prefix: str = "") -> str:
        base = self._readable(prefix) if prefix else self._root
        if not base.is_dir():
            raise ToolError(f"not a directory: {prefix}")
        found: list[str] = []
        for rel in self._walk(base):
            found.append(rel)
            if len(found) > MAX_LISTED:
                break
        shown = found[:MAX_LISTED]
        tail = "\n(list truncated)" if len(found) > MAX_LISTED else ""
        return "\n".join(shown) + tail if shown else "(no files)"

    def editable_files(self) -> list[str]:
        """Existing files the genome allows editing (capped by count and listing size), so the
        model can be told what it may write. Hidden paths and the pack never appear."""
        out: list[str] = []
        chars = 0
        for scanned, rel in enumerate(self._walk(self._root)):
            if scanned >= MAX_EDITABLE_SCAN:
                break
            if not self._genome.check(rel).ok:
                continue
            chars += len(rel) + 1
            if len(out) >= MAX_EDITABLE_SHOWN or chars > MAX_EDITABLE_CHARS:
                break
            out.append(rel)
        return out

    def editable_hint(self) -> str:
        files = self.editable_files()
        if not files:
            return "call list_files"
        return "editable files: " + ", ".join(files)

    def read(self, path: str) -> str:
        full = self._readable(path)
        if not full.is_file():
            raise ToolError(f"no such file: {path} ({self.editable_hint()})")
        data = full.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise ToolError(f"file is larger than {MAX_READ_BYTES} bytes")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("file is not UTF-8 text") from exc
        self._read.add(self._norm(path))
        return text

    def write(self, path: str, content: str) -> str:
        norm = self._norm(path)
        decision = self._genome.check(norm)
        if not decision.ok:
            raise ToolError(
                f"refused: {norm} is not editable ({decision.reason}); {self.editable_hint()}"
            )
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"content is larger than {MAX_WRITE_BYTES} bytes")
        full = self._root / norm
        added: str | None = None
        try:
            walked = self._root
            for part in norm.split("/"):
                walked = walked / part
                if walked.is_symlink() or os.path.isjunction(walked):
                    raise ToolError(f"refused: {norm} passes through a symlink or junction")
            if full.is_dir():
                raise ToolError(f"{norm} is a directory")
            norm = self._canonical(norm, full)
            if norm not in self._originals:
                if full.exists():
                    data_before = full.read_bytes()
                    try:
                        text_before = data_before.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ToolError(f"{norm} is not UTF-8 text; it cannot be edited") from exc
                    # A file read() refuses as too large can never be read: don't demand it.
                    if len(data_before) <= MAX_READ_BYTES and not self._was_read(full):
                        raise ToolError(
                            f"refused: read_file {norm} before rewriting it (it already exists)"
                        )
                    self._originals[norm] = text_before
                else:
                    self._originals[norm] = None
                added = norm
            full.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(full, data)
        except OSError as exc:
            if added is not None:
                del self._originals[added]  # the write is atomic: nothing changed, not touched
            raise ToolError(f"cannot write {norm}: {exc.strerror or type(exc).__name__}") from exc
        return f"wrote {norm} ({len(data)} bytes)"

    def _was_read(self, full: Path) -> bool:
        for known in self._read:
            other = self._root / known
            if other.exists() and os.path.samefile(full, other):
                return True
        return False

    def _canonical(self, norm: str, full: Path) -> str:
        """The key already used for this file, if ``norm`` names the same file by another
        spelling (case variants on a case-insensitive filesystem); else ``norm``."""
        if full.exists():
            for known in self._originals:
                other = self._root / known
                if other.exists() and os.path.samefile(full, other):
                    return known
        return norm

    # --- result -------------------------------------------------------------------

    @property
    def touched(self) -> tuple[str, ...]:
        return tuple(sorted(self._originals))

    def diff(self) -> str:
        """Git-style unified diff of every touched file that now differs; '' if none."""
        out: list[str] = []
        for norm in self.touched:
            before = self._originals[norm]
            after = (self._root / norm).read_bytes().decode("utf-8")
            if before == after:
                continue
            old = _lines(before or "")
            new = _lines(after)
            out.append(f"diff --git a/{norm} b/{norm}\n")
            if before is None:
                out.append("new file mode 100644\n")
            out.append("--- /dev/null\n" if before is None else f"--- a/{norm}\n")
            out.append(f"+++ b/{norm}\n")
            for line in difflib.unified_diff(old, new, n=3, lineterm="\n"):
                if line.startswith(("--- ", "+++ ")):
                    continue
                out.append(
                    line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                )
        return "".join(out)
