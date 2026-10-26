"""Build the target's Containerfile once and return an image reference (§2, §5).

The runner is injectable so unit tests assert the exact argv. The default
runner shells out to the container engine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from iterant.contracts import IterantConfig

BUILD_TIMEOUT_S = 1800
_ENGINES = ("podman", "docker")
_ALWAYS_EXCLUDED = (".git", ".iterant")
_IGNORE_FILES = (".containerignore", ".dockerignore")
_TAG = re.compile(r"[a-z0-9][a-z0-9_.-]*(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?")


class ImageBuildError(RuntimeError):
    """The image could not be built, or its id could not be read back."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str]], CommandResult]


@dataclass(frozen=True)
class Image:
    tag: str
    image_id: str


def slugify(name: str) -> str:
    """Lowercase ``name`` to ``a-z0-9-`` for container and image names."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def subprocess_runner(argv: Sequence[str]) -> CommandResult:
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=BUILD_TIMEOUT_S,
        )
    except FileNotFoundError as exc:
        raise ImageBuildError(f"container engine not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ImageBuildError(f"{argv[0]} timed out after {BUILD_TIMEOUT_S}s") from exc
    return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def _ignore_regex(pattern: str) -> re.Pattern[str]:
    """One .containerignore pattern as a regex over a repo-relative POSIX path (or a parent)."""
    pattern = pattern.strip("/")
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            c = pattern[i]
            out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c))
            i += 1
    return re.compile("".join(out) + "(?:/.*)?", re.DOTALL)


def _ignore_rules(root: Path) -> list[tuple[bool, re.Pattern[str]]]:
    """``(negated, regex)`` per line of the first .containerignore/.dockerignore, in order."""
    for name in _IGNORE_FILES:
        f = root / name
        if f.is_file() and not f.is_symlink():
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            rules = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
            return [(r.startswith("!"), _ignore_regex(r.lstrip("!"))) for r in rules]
    return []


class _Filter:
    """Decides what stays out of the build context. Hard exclusions cannot be re-included."""

    def __init__(self, root: Path, pack: str) -> None:
        self._hard = tuple(p.casefold() for p in (*_ALWAYS_EXCLUDED, pack))
        self._rules = _ignore_rules(root)

    def excluded(self, rel: str) -> bool:
        folded = rel.casefold()
        if any(seg in _ALWAYS_EXCLUDED for seg in folded.split("/")):  # nested repos too
            return True
        if any(folded == h or folded.startswith(h + "/") for h in self._hard):
            return True
        hit = False
        for negated, rx in self._rules:
            if rx.fullmatch(rel):
                hit = not negated
        return hit


def _copy_context(root: Path, dest: Path, flt: _Filter) -> None:
    """Copy ``root`` into ``dest`` minus excluded paths. Symlinks and junctions are skipped."""
    for cur, dirs, files in os.walk(root):
        rel_dir = Path(cur).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        dirs[:] = [
            d
            for d in dirs
            if not flt.excluded(prefix + d)
            and not os.path.islink(os.path.join(cur, d))
            and not os.path.isjunction(os.path.join(cur, d))
        ]
        (dest / rel_dir).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = os.path.join(cur, name)
            if flt.excluded(prefix + name) or os.path.islink(src):
                continue
            shutil.copyfile(src, dest / rel_dir / name)


def build_argv(engine: str, containerfile: Path, tag: str, context: Path) -> list[str]:
    return [engine, "build", "--file", str(containerfile), "--tag", tag, str(context)]


def inspect_argv(engine: str, tag: str) -> list[str]:
    return [engine, "image", "inspect", "--format", "{{.Id}}", tag]


def build_image(
    root: str | os.PathLike[str],
    config: IterantConfig,
    tag: str,
    *,
    engine: str = "podman",
    runner: Runner | None = None,
) -> Image:
    """Build ``config.target.image`` tagged ``tag`` from a filtered copy of ``root``.

    The repo itself is never the build context: it holds the hidden checks. The
    context is a temporary copy without ``.git/``, ``.iterant/``, the pack
    directory and anything the repo's ``.containerignore`` (or ``.dockerignore``)
    lists; symlinks are not copied. The Containerfile must survive that filter.

    ``tag`` is lowercase and should carry the caller's prefix (a branch slug in
    tests, so parallel workers never collide). Raises ``ImageBuildError`` on any
    failure; the build's stderr tail is in the message.
    """
    if engine not in _ENGINES:
        raise ImageBuildError(f"engine must be one of {_ENGINES}, got {engine!r}")
    if not _TAG.fullmatch(tag):
        raise ImageBuildError(f"invalid image tag {tag!r}")
    repo = Path(os.path.realpath(root))
    containerfile = Path(os.path.realpath(repo / config.target.image))
    if not containerfile.is_relative_to(repo):
        raise ImageBuildError(f"{config.target.image} resolves outside the repo")
    if not containerfile.is_file():
        raise ImageBuildError(f"{config.target.image} is not a file")

    rel_file = containerfile.relative_to(repo).as_posix()
    flt = _Filter(repo, config.evaluation.pack)
    if flt.excluded(rel_file):
        raise ImageBuildError(
            f"{rel_file} is in an excluded path (pack, .git, .iterant, ignore file)"
        )

    run = runner or subprocess_runner
    context = Path(tempfile.mkdtemp(prefix="iterant-ctx-"))
    try:
        _copy_context(repo, context, flt)
        built = run(build_argv(engine, context / rel_file, tag, context))
    finally:
        shutil.rmtree(context, ignore_errors=True)
    if built.returncode != 0:
        raise ImageBuildError(f"{engine} build failed ({built.returncode}): {built.stderr[-2000:]}")
    inspected = run(inspect_argv(engine, tag))
    image_id = inspected.stdout.strip()
    if inspected.returncode != 0 or not image_id:
        raise ImageBuildError(f"cannot read id of {tag}: {inspected.stderr[-500:]}")
    return Image(tag=tag, image_id=image_id)
