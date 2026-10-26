"""Genome allowlist: which target-repo paths a candidate may edit (§1.1, §9).

Security code. Everything outside the genome is the trusted base. A path is
editable only if all of these hold:

* it is a clean repo-relative POSIX path: no ``..``, no absolute form, no
  backslash, no ``:`` (drive letters, NTFS streams), no empty or trailing-dot
  or trailing-space segment, no trailing ``/``;
* no segment is ``.git`` or ``.iterant`` (any case, any depth, including the
  Windows 8.3 short names ``GIT~1`` / ``ITERANT~1``);
* it is not protected (``iterant.toml``, the Containerfile, the pack);
* it matches a genome glob;
* once symlinks and junctions are resolved on disk, the real path is still
  inside the repo and passes the same checks.

Glob syntax: ``*`` and ``?`` stay inside one segment, a whole-segment ``**``
spans any number of segments (``dir/**`` means everything below ``dir``, not
``dir`` itself). Anything else (``[``, ``{``, ``!``) is rejected, not guessed.

The check reads the filesystem at call time. Run it on the tree that will be
applied or mounted, and re-run it if that tree changes.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from iterant.contracts import IterantConfig

DENIED_DIRS = (".git", ".iterant")
# Any 8.3 short name (``ITERAN~1``, ``GIT~1.TXT``): it may alias a reserved directory.
_SHORT_NAME = re.compile(r"[^~]+~\d+(?:\.[^.]*)?")
_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
# Code points HFS+/APFS ignore when comparing names (".g<ZWNJ>it" is ".git" there),
# beyond category Cf: combining grapheme joiner, Mongolian and variation selectors, fillers.
_IGNORABLE = frozenset(
    {0x034F, 0x115F, 0x1160, 0x17B4, 0x17B5, 0x3164, 0xFFA0}
    | set(range(0x180B, 0x1810))
    | set(range(0xFE00, 0xFE10))
    | set(range(0xE0100, 0xE01F0))
)
# Never editable, whatever the globs say: root-relative paths, then names matched at any
# depth (git reads nested .gitattributes / .gitignore / .gitmodules too).
DEFAULT_PROTECTED = (".github", "iterant.toml", ".containerignore", ".dockerignore")
DEFAULT_PROTECTED_NAMES = (".gitmodules", ".gitattributes", ".gitignore")
_UNSUPPORTED_GLOB_CHARS = frozenset("[]{}!")


class PathRejectedError(ValueError):
    """A path or glob is malformed or unsafe. ``str(error)`` is the reason."""


def _split(raw: str, *, what: str, allow_wild: bool) -> tuple[str, ...]:
    """Validate ``raw`` and return its segments, with ``.`` segments dropped."""
    if not raw:
        raise PathRejectedError(f"{what} is empty")
    if any(ord(c) < 32 or ord(c) == 127 for c in raw):
        raise PathRejectedError(f"{what} contains a control character")
    if "\\" in raw:
        raise PathRejectedError(f"{what} contains a backslash; use '/'")
    if raw.startswith("/"):
        raise PathRejectedError(f"{what} is absolute")
    if ":" in raw:
        raise PathRejectedError(f"{what} contains ':' (drive letter or stream)")
    if raw.endswith("/"):
        raise PathRejectedError(f"{what} ends with '/'; name files, or use 'dir/**'")
    if any(unicodedata.category(c) == "Cf" or ord(c) in _IGNORABLE for c in raw):
        raise PathRejectedError(f"{what} contains an invisible Unicode character")
    raw = unicodedata.normalize("NFC", raw)
    parts: list[str] = []
    for seg in raw.split("/"):
        if seg == "":
            raise PathRejectedError(f"{what} has an empty segment ('//')")
        if seg == ".":
            continue
        if seg == "..":
            raise PathRejectedError(f"{what} contains '..'")
        if seg.endswith((".", " ")):
            raise PathRejectedError(f"{what} has a segment ending in '.' or space: {seg!r}")
        if _is_denied_segment(seg):
            raise PathRejectedError(f"{what} touches a reserved directory: {seg!r}")
        if seg.casefold().split(".")[0] in _DEVICE_NAMES:
            raise PathRejectedError(f"{what} uses a Windows device name: {seg!r}")
        if not allow_wild and any(c in seg for c in "*?"):
            raise PathRejectedError(f"{what} contains a wildcard")
        parts.append(seg)
    if not parts:
        raise PathRejectedError(f"{what} names the repo root")
    return tuple(parts)


def _is_denied_segment(seg: str) -> bool:
    folded = seg.casefold()
    return folded in DENIED_DIRS or _SHORT_NAME.fullmatch(folded) is not None


def normalize_relative(raw: str, *, what: str = "path") -> str:
    """Return ``raw`` as a clean repo-relative POSIX path, or raise ``PathRejectedError``.

    ``./a/b`` becomes ``a/b``. Used for the config's ``image`` and ``pack`` paths too.
    """
    return "/".join(_split(raw, what=what, allow_wild=False))


def compile_glob(pattern: str, *, case_insensitive: bool = False) -> re.Pattern[str]:
    """Compile one genome glob to a regex that must ``fullmatch`` a normalized path."""
    segs = _split(pattern, what=f"glob {pattern!r}", allow_wild=True)
    pieces: list[str] = []
    last = len(segs) - 1
    for i, seg in enumerate(segs):
        if bad := _UNSUPPORTED_GLOB_CHARS & set(seg):
            raise PathRejectedError(f"glob {pattern!r} uses unsupported {sorted(bad)[0]!r}")
        if "**" in seg and seg != "**":
            raise PathRejectedError(f"glob {pattern!r}: '**' must be a whole segment")
        if seg == "**":
            pieces.append("[^/]+(?:/[^/]+)*" if i == last else "(?:[^/]+/)*")
            continue
        rx = "".join("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c) for c in seg)
        pieces.append(rx if i == last else rx + "/")
    return re.compile("".join(pieces), re.IGNORECASE if case_insensitive else 0)


def _fold_always(parts: Sequence[str]) -> tuple[str, ...]:
    return tuple(s.casefold() for s in parts)


@dataclass(frozen=True)
class Decision:
    """Verdict for one path. ``reason`` is empty when ``ok``."""

    path: str
    ok: bool
    reason: str = ""


class Genome:
    """The editable zone of one target repo.

    ``protected`` paths (and everything under them), plus ``DEFAULT_PROTECTED`` and
    ``DEFAULT_PROTECTED_NAMES``, are never editable even if a glob matches. They are
    compared case-folded in every mode (fail closed: the repo may sit on a
    case-insensitive mount). ``case_insensitive`` only affects glob matching and
    defaults to True on Windows and macOS.

    The check sees the filesystem as it is now. A symlink a diff would create is
    invisible until the diff is applied, so run the check on the post-apply tree
    (the gate also rejects mode-120000 entries).
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        patterns: Sequence[str],
        *,
        protected: Sequence[str] = (),
        case_insensitive: bool | None = None,
    ) -> None:
        if not patterns:
            raise PathRejectedError("genome must list at least one glob")
        self._ci = (
            sys.platform in ("win32", "darwin") if case_insensitive is None else case_insensitive
        )
        self.root = Path(root)
        self.patterns = tuple(patterns)
        self._globs = tuple(compile_glob(p, case_insensitive=self._ci) for p in patterns)
        self._protected = tuple(
            _fold_always(_split(p, what=f"protected path {p!r}", allow_wild=False))
            for p in (*DEFAULT_PROTECTED, *protected)
        )

    def _fold(self, parts: Sequence[str]) -> tuple[str, ...]:
        return tuple(s.casefold() for s in parts) if self._ci else tuple(parts)

    def _verdict(self, parts: tuple[str, ...]) -> str:
        """Return a rejection reason for already-normalized segments, or ''."""
        folded = _fold_always(parts)
        for prot in self._protected:
            if folded[: len(prot)] == prot:
                return f"protected path: {'/'.join(prot)}"
        if named := frozenset(DEFAULT_PROTECTED_NAMES).intersection(folded):
            return f"protected path: {sorted(named)[0]}"
        joined = "/".join(parts)
        if not any(g.fullmatch(joined) for g in self._globs):
            return "outside the genome"
        return ""

    def _resolved_parts(self, parts: tuple[str, ...]) -> tuple[str, ...] | str:
        """Segments of the real location under the real root, or a rejection reason."""
        try:
            real_root = Path(os.path.realpath(self.root))
            real = Path(os.path.realpath(self.root.joinpath(*parts)))
            return real.relative_to(real_root).parts or "resolves to the repo root"
        except ValueError:
            return "resolves outside the repo"
        except OSError as exc:
            return f"cannot resolve: {exc}"

    def check(self, path: str) -> Decision:
        """Decide whether ``path`` (repo-relative, ``/``-separated) is editable."""
        try:
            parts = _split(path, what="path", allow_wild=False)
        except PathRejectedError as exc:
            return Decision(path, False, str(exc))
        if reason := self._verdict(parts):
            return Decision(path, False, reason)
        resolved = self._resolved_parts(parts)
        if isinstance(resolved, str):
            return Decision(path, False, resolved)
        if self._fold(resolved) != self._fold(parts):
            if any(_is_denied_segment(s) for s in resolved):
                return Decision(path, False, "resolves into a reserved directory")
            if reason := self._verdict(resolved):
                return Decision(path, False, f"symlink target {'/'.join(resolved)}: {reason}")
        return Decision(path, True)

    def is_editable(self, path: str) -> bool:
        return self.check(path).ok

    def violations(self, paths: Iterable[str]) -> list[Decision]:
        """Every rejected path in a diff's file list (include both sides of a rename)."""
        return [d for d in map(self.check, paths) if not d.ok]


def genome_for(root: str | os.PathLike[str], config: IterantConfig) -> Genome:
    """The genome of ``config``, with ``iterant.toml``, the image file and the pack protected.

    Without this a broad glob such as ``**`` would let a candidate rewrite its own
    model cap, its build file or the checks.
    """
    protected = ("iterant.toml", config.target.image, config.evaluation.pack)
    return Genome(root, config.target.genome, protected=protected)
