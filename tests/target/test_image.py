from __future__ import annotations

import dataclasses
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from iterant.contracts import EvalConfig, IterantConfig
from iterant.target import CommandResult, ImageBuildError, build_image, parse_config, slugify

SLUG = "w1-target"
CONFIG_TEXT = """
[target]
genome = ["prompts/**"]
image = "Containerfile"
run = "python -m agent"
[target.model]
name = "claude-haiku-5-5"
effort = "medium"
max_usd_per_run = 0.10
[eval]
pack = "evals/pack"
"""


def _config() -> IterantConfig:
    return parse_config(CONFIG_TEXT)


class Recorder:
    def __init__(self, *results: CommandResult) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results)

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        self.calls.append(list(argv))
        return self._results.pop(0)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "Containerfile").write_text("FROM scratch\n", encoding="utf-8")
    return tmp_path


def test_exact_argv_and_id(tmp_path: Path) -> None:
    root = _repo(tmp_path).resolve()
    rec = Recorder(CommandResult(0), CommandResult(0, "sha256:abc123\n"))
    img = build_image(root, _config(), f"iterant-{SLUG}-t1", runner=rec)
    build, inspect = rec.calls
    context = build[-1]
    assert context != str(root)  # never the repo itself
    assert build == [
        "podman",
        "build",
        "--file",
        str(Path(context) / "Containerfile"),
        "--tag",
        f"iterant-{SLUG}-t1",
        context,
    ]
    assert inspect == ["podman", "image", "inspect", "--format", "{{.Id}}", f"iterant-{SLUG}-t1"]
    assert (img.tag, img.image_id) == (f"iterant-{SLUG}-t1", "sha256:abc123")
    assert not Path(context).exists()  # temp context cleaned up


class ContextSnapshot:
    """Runner that records the files in the build context while the build is running."""

    def __init__(self) -> None:
        self.files: set[str] = set()

    def __call__(self, argv: Sequence[str]) -> CommandResult:
        if argv[1] == "build":
            ctx = Path(argv[-1])
            self.files = {p.relative_to(ctx).as_posix() for p in ctx.rglob("*") if p.is_file()}
            return CommandResult(0)
        return CommandResult(0, "id")


def _populate(root: Path, files: Sequence[str]) -> None:
    for rel in files:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


def test_context_excludes_pack_git_iterant(tmp_path: Path) -> None:
    _populate(
        tmp_path,
        [
            "Containerfile",
            "agent/loop.py",
            "prompts/system.md",
            "evals/pack/tasks/t1/check",
            "evals/pack/pack.toml",
            ".git/config",
            ".iterant/archive.db",
            "vendor/.git/HEAD",
            "EVALS/PACK/x",
        ],
    )
    snap = ContextSnapshot()
    build_image(tmp_path, _config(), "iterant-x", runner=snap)
    assert snap.files == {"Containerfile", "agent/loop.py", "prompts/system.md"}


def test_context_honours_containerignore_with_negation(tmp_path: Path) -> None:
    _populate(
        tmp_path,
        ["Containerfile", "keep.txt", "big/a.bin", "big/keep.md", "x.pyc", "d/y.pyc", "d/z.py"],
    )
    (tmp_path / ".containerignore").write_text(
        "# comment\nbig\n!big/keep.md\n**/*.pyc\n", encoding="utf-8"
    )
    snap = ContextSnapshot()
    build_image(tmp_path, _config(), "iterant-x", runner=snap)
    # An excluded directory is pruned whole: `!big/keep.md` does not re-include (fails closed).
    assert snap.files == {"Containerfile", "keep.txt", "d/z.py", ".containerignore"}


def test_negation_cannot_reinclude_pack(tmp_path: Path) -> None:
    _populate(tmp_path, ["Containerfile", "evals/pack/check"])
    (tmp_path / ".dockerignore").write_text("!evals/pack/check\n", encoding="utf-8")
    snap = ContextSnapshot()
    build_image(tmp_path, _config(), "iterant-x", runner=snap)
    assert "evals/pack/check" not in snap.files


def test_symlinks_are_not_copied(tmp_path: Path) -> None:
    _populate(tmp_path, ["Containerfile", "evals/pack/check"])
    try:
        (tmp_path / "leak").symlink_to(tmp_path / "evals" / "pack", target_is_directory=True)
        (tmp_path / "leak.txt").symlink_to(tmp_path / "evals" / "pack" / "check")
    except OSError:
        pytest.skip("symlinks unavailable")
    snap = ContextSnapshot()
    build_image(tmp_path, _config(), "iterant-x", runner=snap)
    assert snap.files == {"Containerfile"}


def test_containerfile_in_excluded_path_refused(tmp_path: Path) -> None:
    _populate(tmp_path, ["evals/pack/Containerfile"])
    cfg = parse_config(
        CONFIG_TEXT.replace('image = "Containerfile"', 'image = "evals/pack/Containerfile"')
    )
    with pytest.raises(ImageBuildError, match="excluded path"):
        build_image(tmp_path, cfg, "iterant-x", runner=Recorder())


def test_containerfile_ignored_by_ignore_file_refused(tmp_path: Path) -> None:
    _populate(tmp_path, ["Containerfile"])
    (tmp_path / ".containerignore").write_text("Containerfile\n", encoding="utf-8")
    with pytest.raises(ImageBuildError, match="excluded path"):
        build_image(tmp_path, _config(), "iterant-x", runner=Recorder())


def test_context_cleaned_up_on_failure(tmp_path: Path) -> None:
    _repo(tmp_path)
    seen: list[str] = []

    def runner(argv: Sequence[str]) -> CommandResult:
        seen.append(argv[-1])
        return CommandResult(1, "", "no")

    with pytest.raises(ImageBuildError):
        build_image(tmp_path, _config(), "iterant-x", runner=runner)
    assert not Path(seen[0]).exists()


def test_docker_engine(tmp_path: Path) -> None:
    rec = Recorder(CommandResult(0), CommandResult(0, "id"))
    build_image(_repo(tmp_path), _config(), "iterant-x", engine="docker", runner=rec)
    assert [c[0] for c in rec.calls] == ["docker", "docker"]


def test_build_failure_includes_stderr(tmp_path: Path) -> None:
    rec = Recorder(CommandResult(125, "", "boom: no such base image"))
    with pytest.raises(ImageBuildError, match="boom: no such base image"):
        build_image(_repo(tmp_path), _config(), "iterant-x", runner=rec)
    assert len(rec.calls) == 1


def test_missing_id_is_error(tmp_path: Path) -> None:
    rec = Recorder(CommandResult(0), CommandResult(0, "  \n"))
    with pytest.raises(ImageBuildError, match="cannot read id"):
        build_image(_repo(tmp_path), _config(), "iterant-x", runner=rec)


@pytest.mark.parametrize("tag", ["", "UPPER", "-lead", "a b", "a;rm", "x:", "x/../y"])
def test_bad_tag_rejected_without_running(tmp_path: Path, tag: str) -> None:
    rec = Recorder()
    with pytest.raises(ImageBuildError, match="invalid image tag"):
        build_image(_repo(tmp_path), _config(), tag, runner=rec)
    assert rec.calls == []


def test_bad_engine(tmp_path: Path) -> None:
    with pytest.raises(ImageBuildError, match="engine"):
        build_image(_repo(tmp_path), _config(), "iterant-x", engine="rm", runner=Recorder())


def test_missing_containerfile(tmp_path: Path) -> None:
    with pytest.raises(ImageBuildError, match="is not a file"):
        build_image(tmp_path, _config(), "iterant-x", runner=Recorder())


def test_containerfile_symlink_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Containerfile").write_text("FROM scratch\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        (repo / "Containerfile").symlink_to(outside / "Containerfile")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ImageBuildError, match="outside the repo"):
        build_image(repo, _config(), "iterant-x", runner=Recorder())


def test_slugify() -> None:
    assert slugify("claude/W1_Target 459066!") == "claude-w1-target-459066"


@pytest.mark.podman
@pytest.mark.skipif(shutil.which("podman") is None, reason="podman not installed")
def test_real_build(tmp_path: Path) -> None:
    tag = f"iterant-{SLUG}-real-test"
    img = build_image(_repo(tmp_path), _config(), tag)
    try:
        assert img.image_id
    finally:
        subprocess.run(["podman", "rmi", "--force", tag], check=False, capture_output=True)


@pytest.mark.parametrize("pack", ["./evals/pack", "evals//pack", "evals/./pack"])
def test_unnormalised_pack_path_still_excludes_the_pack(tmp_path: Path, pack: str) -> None:
    _populate(tmp_path, ["Containerfile", "evals/pack/check", "agent/loop.py"])
    cfg = dataclasses.replace(_config(), evaluation=EvalConfig(pack=pack))
    snap = ContextSnapshot()
    if pack == "evals//pack":  # an empty segment is malformed: refused, never half-excluded
        with pytest.raises(ImageBuildError):
            build_image(tmp_path, cfg, "iterant-x", runner=snap)
        return
    build_image(tmp_path, cfg, "iterant-x", runner=snap)
    assert snap.files == {"Containerfile", "agent/loop.py"}


def test_context_file_count_is_capped(tmp_path: Path) -> None:
    _populate(tmp_path, ["Containerfile", *[f"d/f{i}" for i in range(6)]])
    rec = Recorder()
    with pytest.raises(ImageBuildError, match="limit"):
        build_image(tmp_path, _config(), "iterant-x", runner=rec, max_files=5)
    assert rec.calls == []  # nothing was built


def test_context_total_bytes_are_capped(tmp_path: Path) -> None:
    _populate(tmp_path, ["Containerfile"])
    (tmp_path / "big.bin").write_bytes(b"x" * 1000)
    rec = Recorder()
    with pytest.raises(ImageBuildError, match="limit"):
        build_image(tmp_path, _config(), "iterant-x", runner=rec, max_bytes=500)
    assert rec.calls == []
