"""Strict parser for git-style and plain unified diffs.

The gate reads a candidate's diff before anything is applied, so the parser fails closed:
a line it does not recognise is an error, never skipped. Paths follow ``git apply -p1``
(the first path component is stripped), and ``diff --git`` headers must use the standard
``a/`` and ``b/`` prefixes. The gate also re-checks the tree that really results from
applying the diff, so a parser/applier disagreement cannot hide a path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_OCTAL = re.compile(r"\\([0-7]{3})")
_SIMPLE_ESC = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11, '"': 34, "\\": 92}
_IGNORED_HEADERS = ("similarity index ", "dissimilarity index ")


class DiffError(ValueError):
    """The diff is malformed or uses a form the gate does not accept."""


@dataclass
class FileDiff:
    """One file section. ``None`` for ``old``/``new`` means ``/dev/null``."""

    old: str | None = None
    new: str | None = None
    extra_paths: list[str] = field(
        default_factory=lambda: []
    )  # other splits of an ambiguous header
    modes: list[str] = field(default_factory=lambda: [])
    binary: bool = False
    renamed: bool = False
    copied: bool = False
    added: list[str] = field(default_factory=lambda: [])
    removed: list[str] = field(default_factory=lambda: [])

    @property
    def paths(self) -> list[str]:
        return [*(p for p in (self.old, self.new) if p is not None), *self.extra_paths]


def _unquote(raw: str) -> str:
    """Decode git's C-style quoting (``"caf\\303\\251"``); unquoted paths pass through."""
    if not raw.startswith('"'):
        return raw
    if len(raw) < 2 or not raw.endswith('"'):
        raise DiffError(f"unterminated quoted path: {raw!r}")
    body, out, i = raw[1:-1], bytearray(), 0
    while i < len(body):
        if body[i] != "\\":
            out += body[i].encode("utf-8")
            i += 1
        elif m := _OCTAL.match(body, i):
            out.append(int(m.group(1), 8) & 0xFF)
            i += 4
        elif i + 1 < len(body) and body[i + 1] in _SIMPLE_ESC:
            out.append(_SIMPLE_ESC[body[i + 1]])
            i += 2
        else:
            raise DiffError(f"bad escape in quoted path: {raw!r}")
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiffError(f"quoted path is not UTF-8: {raw!r}") from exc


def _strip1(path: str) -> str:
    """Drop the first path component, as ``git apply -p1`` does."""
    _, sep, rest = path.partition("/")
    if not sep or not rest:
        raise DiffError(f"path has no directory prefix to strip: {path!r}")
    return rest


def _operand(raw: str) -> str | None:
    """A ``---``/``+++`` operand: drop a trailing timestamp; ``/dev/null`` becomes None."""
    path = _unquote(raw.split("\t", 1)[0].rstrip("\r"))
    return None if path == "/dev/null" else _strip1(path)


def _git_header(rest: str) -> tuple[str, str, list[str]]:
    """Old and new path from ``diff --git a/X b/Y``; other splits of a spaced path are extras."""
    if rest.startswith('"'):
        end = 1
        while end < len(rest) and not (rest[end] == '"' and rest[end - 1] != "\\"):
            end += 1
        return _strip1(_unquote(rest[: end + 1])), _strip1(_unquote(rest[end + 2 :])), []
    pairs: list[tuple[str, str]] = []
    start = 0
    while (cut := rest.find(" b/", start)) != -1:
        pairs.append((rest[:cut], rest[cut + 1 :]))
        start = cut + 1
    if not pairs or not pairs[0][0].startswith("a/"):
        raise DiffError(f"cannot read diff header (need a/ and b/ prefixes): {rest[:80]!r}")
    for a, b in pairs:
        if a[2:] == b[2:]:
            return a[2:], b[2:], []
    stripped = [(a[2:], b[2:]) for a, b in pairs]
    return stripped[0][0], stripped[0][1], [p for pair in stripped[1:] for p in pair]


def parse_diff(text: str) -> list[FileDiff]:
    """Parse ``text`` into file sections or raise :class:`DiffError`."""
    if "\x00" in text:
        raise DiffError("diff contains a NUL byte")
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    old_left = new_left = 0
    seen_old = seen_new = False
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    for line in lines:
        if old_left > 0 or new_left > 0:
            assert cur is not None
            if line.startswith("+"):
                new_left -= 1
                cur.added.append(line[1:])
            elif line.startswith("-"):
                old_left -= 1
                cur.removed.append(line[1:])
            elif line.startswith(" ") or line in ("", "\r"):
                old_left -= 1
                new_left -= 1
            elif not line.startswith("\\"):
                raise DiffError(f"unexpected line inside a hunk: {line[:60]!r}")
            if old_left < 0 or new_left < 0:
                raise DiffError("hunk is longer than its header says")
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("diff --git ") or (line.startswith("--- ") and seen_old):
            if seen_old and not seen_new:
                raise DiffError("'---' without '+++'")
        if line.startswith("diff --git "):
            old, new, extra = _git_header(line[len("diff --git ") :])
            cur = FileDiff(old=old, new=new, extra_paths=extra)
            files.append(cur)
            seen_old = seen_new = False
        elif line.startswith("--- "):
            if cur is None or seen_old:  # plain unified diff: a new section
                cur = FileDiff()
                files.append(cur)
            cur.old, seen_old, seen_new = _operand(line[4:]), True, False
        elif line.startswith("+++ "):
            if cur is None or not seen_old:
                raise DiffError("'+++' without a preceding '---'")
            cur.new, seen_new = _operand(line[4:]), True
        elif line.startswith("@@ "):
            m = _HUNK.match(line)
            if m is None or cur is None or not (seen_old and seen_new):
                raise DiffError(f"bad hunk header: {line[:60]!r}")
            old_left = int(m.group(1) if m.group(1) is not None else 1)
            new_left = int(m.group(2) if m.group(2) is not None else 1)
        elif cur is not None and _header_line(cur, line):
            continue
        elif cur is not None and cur.binary:
            continue  # payload of a binary patch; binary is already a reject
        elif line.strip() == "":
            continue
        else:
            raise DiffError(f"unrecognised diff line: {line[:60]!r}")
    if old_left > 0 or new_left > 0:
        raise DiffError("diff ends inside a hunk")
    if seen_old and not seen_new:
        raise DiffError("'---' without '+++'")
    if not files:
        raise DiffError("no file sections found")
    if any(f.old is None and f.new is None for f in files):
        raise DiffError("file section names no path")
    return files


def _header_line(cur: FileDiff, line: str) -> bool:
    """Apply an extended git header line to ``cur``; False if ``line`` is not one."""
    for prefix in ("old mode ", "new mode ", "deleted file mode ", "new file mode "):
        if line.startswith(prefix):
            cur.modes.append(line[len(prefix) :].strip())
            if prefix == "deleted file mode ":
                cur.new = None
            elif prefix == "new file mode ":
                cur.old = None
            return True
    for prefix in ("rename from ", "copy from ", "rename to ", "copy to "):
        if line.startswith(prefix):
            path = _unquote(line[len(prefix) :])
            if prefix.endswith("from "):
                cur.old = path
            else:
                cur.new = path
            cur.renamed |= prefix.startswith("rename")
            cur.copied |= prefix.startswith("copy")
            return True
    if line.startswith("index "):
        parts = line.split(" ")
        if len(parts) > 2:
            cur.modes.append(parts[2].strip())  # ``index abc..def 100644`` carries the mode
        return True
    if line.startswith(("Binary files ", "GIT binary patch")):
        cur.binary = True
        return True
    return line.startswith(_IGNORED_HEADERS)


def changed_line_count(files: list[FileDiff]) -> int:
    return sum(len(f.added) + len(f.removed) for f in files)
