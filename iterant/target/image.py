"""Build the target's Containerfile once and return an image reference (§2, §5).

The runner is injectable so unit tests assert the exact argv. The default
runner shells out to the container engine.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from iterant.contracts import IterantConfig

BUILD_TIMEOUT_S = 1800
_ENGINES = ("podman", "docker")
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
    """Build ``config.target.image`` with ``root`` as context, tagged ``tag``.

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

    run = runner or subprocess_runner
    built = run(build_argv(engine, containerfile, tag, repo))
    if built.returncode != 0:
        raise ImageBuildError(f"{engine} build failed ({built.returncode}): {built.stderr[-2000:]}")
    inspected = run(inspect_argv(engine, tag))
    image_id = inspected.stdout.strip()
    if inspected.returncode != 0 or not image_id:
        raise ImageBuildError(f"cannot read id of {tag}: {inspected.stderr[-500:]}")
    return Image(tag=tag, image_id=image_id)
