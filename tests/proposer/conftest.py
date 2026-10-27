from __future__ import annotations

from pathlib import Path

import pytest

from iterant.contracts import (
    EvalConfig,
    IterantConfig,
    TargetConfig,
    TargetModel,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        "prompts/system.md": "You are helpful.\n",
        "agent/loop.py": "def run():\n    return 1\n",
        "agent/other.py": "X = 1\n",
        "iterant.toml": "# config\n",
        "evals/pack/pack.toml": "name = 'p'\n",
        "evals/pack/secret.txt": "HIDDEN-CHECK-CONTENT\n",
    }
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8", newline="")
    return root


@pytest.fixture
def config() -> IterantConfig:
    return IterantConfig(
        target=TargetConfig(
            genome=("prompts/**", "agent/loop.py"),
            image="Containerfile",
            run="python -m agent",
            model=TargetModel("claude-haiku-5-5", "medium", max_usd_micros_per_run=100_000),
        ),
        evaluation=EvalConfig(pack="evals/pack"),
    )
