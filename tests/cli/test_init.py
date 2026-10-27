"""``iterant init`` writes a config the rest of Iterant accepts."""

from __future__ import annotations

from pathlib import Path

import pytest

from iterant.cli.main import main
from iterant.target import load_config


def test_free_profile_is_the_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path)]) == 0
    cfg = load_config(tmp_path / "iterant.toml")
    assert cfg.target.model.max_tokens_per_run == 200_000
    assert cfg.target.model.max_usd_micros_per_run is None
    assert cfg.target.genome == ("prompts/**",)
    out = capsys.readouterr().out
    assert "free profile" in out
    assert "not a git repo" in out and "no task pack at evals/pack" in out


def test_claude_profile_and_options(tmp_path: Path) -> None:
    argv = [
        "init", str(tmp_path), "--profile", "claude", "--genome", "a/**", "--genome", "b.py",
        "--pack", "evals/mine", "--run", "node agent.js", "--image", "docker/Containerfile",
    ]  # fmt: skip
    assert main(argv) == 0
    cfg = load_config(tmp_path / "iterant.toml")
    assert cfg.target.model.name == "claude-haiku-5-5"
    assert cfg.target.model.max_usd_micros_per_run == 100_000
    assert cfg.target.genome == ("a/**", "b.py")
    assert (cfg.evaluation.pack, cfg.target.run) == ("evals/mine", "node agent.js")


def test_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "iterant.toml").write_text("# mine\n")
    assert main(["init", str(tmp_path)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert (tmp_path / "iterant.toml").read_text() == "# mine\n"
    assert main(["init", str(tmp_path), "--force"]) == 0
    load_config(tmp_path / "iterant.toml")


@pytest.mark.parametrize(
    "extra",
    [
        ["--genome", "../escape/**"],  # the parser's own rules apply
        ["--max-usd-per-run", "abc"],
        ["--max-usd-per-run", "0"],
        ["--pack", "/abs/pack"],
    ],
)
def test_invalid_values_never_write_a_file(tmp_path: Path, extra: list[str]) -> None:
    assert main(["init", str(tmp_path), "--profile", "claude", *extra]) == 2
    assert not (tmp_path / "iterant.toml").exists()


def test_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err
