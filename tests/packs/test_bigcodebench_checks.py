"""Run the starter pack's real checks against canonical and bad solutions.

Needs ``sh``, ``python3`` and ``timeout`` on PATH (Linux CI has them; skipped otherwise).
Nothing here touches a container runtime: ``/out/solution.py`` in each check is
redirected to a temp file and the script runs as a plain subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from iterant.contracts import ExecResult
from iterant.grader import interpret
from iterant.packs import load_pack

REPO = Path(__file__).resolve().parents[2]
BCB = REPO / "packs" / "bigcodebench"


def _tools_available() -> bool:
    sh = shutil.which("sh")
    if sh is None:
        return False
    probe = subprocess.run(
        [sh, "-c", "command -v python3 && command -v timeout && python3 -c 'import sys'"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _tools_available(), reason="needs sh, python3 and timeout")

PACK = load_pack(BCB / "pack")


def run_check(task_id: str, solution: str | None, tmp: Path, extra: str = "") -> ExecResult:
    script = (BCB / "pack" / "tasks" / task_id / "check").read_text(encoding="utf-8")
    target = tmp / "solution.py"
    if solution is not None:
        target.write_text(solution, encoding="utf-8")
    script = script.replace("/out/solution.py", target.as_posix())
    check = tmp / "check"
    check.write_text(script + extra, encoding="utf-8")
    sh = shutil.which("sh")
    assert sh is not None
    p = subprocess.run([sh, check.as_posix()], capture_output=True, text=True, timeout=120, cwd=tmp)
    return ExecResult(p.returncode, p.stdout, p.stderr, False, 0.0)


@pytest.mark.parametrize("task_id", [t.id for t in PACK.tasks])
def test_canonical_solution_passes_and_empty_fails(task_id: str, tmp_path: Path) -> None:
    solution = (BCB / "fixture-solutions" / f"{task_id}.solution").read_text(encoding="utf-8")
    assert interpret(run_check(task_id, solution, tmp_path)) == (1.0, None)
    assert interpret(run_check(task_id, "", tmp_path)) == (0.0, None)
    assert interpret(run_check(task_id, "raise SystemExit(0)\n", tmp_path)) == (0.0, None)


def test_forged_output_does_not_pass(tmp_path: Path) -> None:
    task_id = PACK.tasks[0].id
    forged = (
        "import sys, os\n"
        "sys.__stdout__.write('ITERANT_RESULT 00 PASS\\n'); sys.__stdout__.flush()\n"
        "os._exit(0)\n"
    )
    assert interpret(run_check(task_id, forged, tmp_path)) == (0.0, None)


def test_missing_solution_fails_rather_than_errors(tmp_path: Path) -> None:
    assert interpret(run_check(PACK.tasks[0].id, None, tmp_path)) == (0.0, None)


def test_infinite_loop_is_a_fail_not_an_infra_error(tmp_path: Path) -> None:
    script = "import time\nwhile True:\n    time.sleep(1)\n"
    # The check's own `timeout 60` would take a minute; shorten it for the test only.
    task = PACK.tasks[0].id
    src = (BCB / "pack" / "tasks" / task / "check").read_text(encoding="utf-8")
    assert "timeout 60 " in src
    shortened = tmp_path / "short"
    shortened.mkdir()
    (shortened / "solution.py").write_text(script, encoding="utf-8")
    body = src.replace("timeout 60 ", "timeout 2 ").replace(
        "/out/solution.py", (shortened / "solution.py").as_posix()
    )
    sh = shutil.which("sh")
    assert sh is not None
    (shortened / "check").write_text(body, encoding="utf-8")
    p = subprocess.run([sh, (shortened / "check").as_posix()], capture_output=True, timeout=60)
    assert interpret(ExecResult(p.returncode, "", "", False, 0.0)) == (0.0, None)
