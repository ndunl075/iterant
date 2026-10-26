"""Grade the synthetic pack end to end with a real container backend.

Needs ``iterant.sandbox`` (PR #7) and a runtime; skips otherwise. Local-only in CI
(deselected by the podman/docker markers); run with ``pytest -m "podman or docker"``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import Sandbox, SandboxBackend
from iterant.grader import Grader
from iterant.packs import load_pack

SYNTHETIC = Path(__file__).resolve().parents[1] / "packs" / "synthetic"
IMAGE = os.environ.get("ITERANT_TEST_IMAGE", "docker.io/library/python:3.12-slim")

BACKENDS = [
    pytest.param("podman", marks=pytest.mark.podman, id="podman"),
    pytest.param("docker", marks=pytest.mark.docker, id="docker"),
]


def _runtime_works(kind: str) -> bool:
    exe = shutil.which(kind)
    if exe is None:
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class Tracking:
    """Wraps a backend to destroy every sandbox it made, even if a test fails."""

    def __init__(self, inner: SandboxBackend) -> None:
        self.inner = inner
        self.created: list[Sandbox] = []

    def create(self, *args: Any, **kwargs: Any) -> Sandbox:
        sb = self.inner.create(*args, **kwargs)
        self.created.append(sb)
        return sb


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[Tracking]:
    kind: str = request.param
    sandbox = pytest.importorskip("iterant.sandbox")
    if not _runtime_works(kind):
        pytest.skip(f"{kind} is not installed or not usable here")
    prefix = "grader-int-" + kind
    inner: SandboxBackend = (
        sandbox.PodmanBackend(name_prefix=prefix)
        if kind == "podman"
        else sandbox.DockerBackend(unsafe_container=True, name_prefix=prefix)
    )
    tracking = Tracking(inner)
    try:
        yield tracking
    finally:
        for sb in tracking.created:
            try:
                sb.destroy()
            except Exception:
                pass


def solve(script: str) -> dict[str, bytes]:
    return {"solve.sh": script.encode()}


def test_synthetic_pack_graded_in_a_real_sandbox(backend: Tracking) -> None:
    task = load_pack(SYNTHETIC).tasks[3]
    data = json.loads((SYNTHETIC / task.input).read_bytes())
    total = data["a"] + data["b"]
    grader = Grader(backend, IMAGE, SYNTHETIC, timeout=20)

    good = grader.grade(task, solve(f"printf '{{\"sum\": {total}}}'\n"))
    wrong = grader.grade(task, solve("printf '{\"sum\": -1}'\n"))
    exit127 = grader.grade(task, solve("exit 127\n"))
    missing = grader.grade(task, {})  # no solve.sh at all: sh fails
    bad_json = grader.grade(task, solve("echo not json\n"))
    nonfinite = grader.grade(task, solve("printf '{\"sum\": NaN}'\n"))
    slow = Grader(backend, IMAGE, SYNTHETIC, timeout=5).grade(task, solve("sleep 60\n"))

    assert good.score == 1.0 and good.error is None
    for r in (wrong, exit127, missing, bad_json, nonfinite, slow):
        assert r.score == 0.0 and r.error is None, r
    assert len(backend.created) == 7
