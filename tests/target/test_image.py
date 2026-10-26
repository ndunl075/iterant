from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from iterant.contracts import IterantConfig
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
    assert rec.calls == [
        [
            "podman",
            "build",
            "--file",
            str(root / "Containerfile"),
            "--tag",
            f"iterant-{SLUG}-t1",
            str(root),
        ],
        ["podman", "image", "inspect", "--format", "{{.Id}}", f"iterant-{SLUG}-t1"],
    ]
    assert (img.tag, img.image_id) == (f"iterant-{SLUG}-t1", "sha256:abc123")


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
