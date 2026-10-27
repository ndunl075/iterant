"""Dotenv-style names stay out of the sandbox upload; look-alike words do not."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import Pack, Proposal, TargetConfig
from iterant.gate import DiffGate, StaticChecks
from iterant.gate.gate import _excluder  # pyright: ignore[reportPrivateUsage]

SECRET_NAMES = [
    ".env",
    ".env.local",
    ".env-production",
    ".env_local",
    ".env-dev",
    ".envrc",
    ".envrc.local",
    ".env~",
    ".envs",
    "prod.env",
    ".ENV.Local",
]


@pytest.mark.parametrize("name", SECRET_NAMES)
def test_secret_looking_names_are_excluded(name: str, tmp_path: Path) -> None:
    excluded = _excluder(tmp_path, [])
    assert excluded(name)
    assert excluded(f"tools/{name}")


@pytest.mark.parametrize("name", [".environment.md", ".envelope", "environment.md", "env.py"])
def test_look_alike_words_are_not_excluded(name: str, tmp_path: Path) -> None:
    assert not _excluder(tmp_path, [])(f"tools/{name}")


def test_the_envs_directory_is_pruned_from_the_upload(
    repo: Path,
    target: TargetConfig,
    pack: Pack,
    make_gate: Any,
    edit: Any,
    fake_backend: Any,
) -> None:
    d = repo / ".envs" / ".production"
    d.mkdir(parents=True)
    (d / ".django").write_bytes(b"SECRET=1\n")
    for name in (".env-production", ".env_local", ".envrc.local", ".env~"):
        (repo / "tools" / name).write_bytes(b"S=1\n")
    gate: DiffGate = make_gate(
        static=StaticChecks(backend=fake_backend, image="img", test="pytest -q")
    )
    good = edit("prompts/system.md", "You are a careful assistant, always.\n")
    assert gate.check(Proposal(good, "h"), repo=repo, target=target, pack=pack).passed
    names = tarfile.open(fileobj=io.BytesIO(fake_backend.puts["/tmp/iterant-gate.tar"])).getnames()
    assert "prompts/system.md" in names
    assert not [n for n in names if ".env" in n]
