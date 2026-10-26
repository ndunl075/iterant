"""Run the starter pack's real checks against canonical, bad and tampering solutions.

Needs ``sh``, ``python3``, ``timeout`` and ``base64`` on PATH (Linux CI has them;
skipped otherwise). Nothing here touches a container runtime: ``/out/solution.py`` in
each check is redirected to a temp file and the script runs as a plain subprocess.
"""

from __future__ import annotations

import os
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
        [
            sh,
            "-c",
            "command -v python3 && command -v timeout && command -v base64"
            " && python3 -I -S -B -c 'import unittest'",
        ],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _tools_available(), reason="needs sh, python3, timeout and base64"
)

PACK = load_pack(BCB / "pack")


def check_text(task_id: str) -> str:
    return (BCB / "pack" / "tasks" / task_id / "check").read_text(encoding="utf-8")


def canonical(task_id: str) -> str:
    return (BCB / "fixture-solutions" / f"{task_id}.solution").read_text(encoding="utf-8")


def run_check(
    task_id: str, solution: str | None, tmp: Path, script: str | None = None
) -> ExecResult:
    target = tmp / "solution.py"
    if solution is not None:
        target.write_text(solution, encoding="utf-8")
    body = (script or check_text(task_id)).replace("/out/solution.py", target.as_posix())
    check = tmp / "check"
    check.write_text(body, encoding="utf-8")
    sh = shutil.which("sh")
    assert sh is not None
    env = {**os.environ, "TMPDIR": str(tmp / "tmpdir")}  # the check's private temp dir
    (tmp / "tmpdir").mkdir(exist_ok=True)
    p = subprocess.run(
        [sh, check.as_posix()], capture_output=True, text=True, timeout=120, cwd=tmp, env=env
    )
    return ExecResult(p.returncode, p.stdout, p.stderr, False, 0.0)


@pytest.mark.parametrize("task_id", [t.id for t in PACK.tasks])
def test_canonical_solution_passes_and_empty_fails(task_id: str, tmp_path: Path) -> None:
    assert interpret(run_check(task_id, canonical(task_id), tmp_path)) == (1.0, None)
    assert interpret(run_check(task_id, "", tmp_path)) == (0.0, None)


ATTACKS = {
    "forged_marker": (
        "import sys, os\n"
        "sys.__stdout__.write('ITERANT_RESULT 00 PASS\\n'); sys.__stdout__.flush()\n"
        "os._exit(0)\n"
    ),
    "patch_testcase_run": (
        "import unittest\n"
        "unittest.TestCase.run = lambda self, result=None: (\n"
        "    result.startTest(self), result.addSuccess(self), result.stopTest(self))\n"
    ),
    "patch_addfailure": (
        "import unittest\n"
        "unittest.TestResult.addFailure = lambda self, test, err: None\n"
        "unittest.TestResult.addError = lambda self, test, err: None\n"
    ),
    "delete_tests": (
        "import unittest\nunittest.TestLoader.getTestCaseNames = lambda self, c: []\n"
    ),
    "replace_unittest_module": (
        "import sys, types\nsys.modules['unittest'] = types.ModuleType('unittest')\n"
    ),
    "patch_runner": (
        "import unittest\nunittest.TextTestRunner.run = lambda self, suite: unittest.TestResult()\n"
    ),
    "patch_build_class": (
        "import builtins\nbuiltins.__build_class__ = lambda *a, **k: type('X', (), {})\n"
    ),
    "sys_exit_zero": "import sys\nsys.exit(0)\n",
    "os_exit_zero": "import os\nos._exit(0)\n",
    "raise": "raise RuntimeError('boom')\n",
}


@pytest.mark.parametrize("attack", sorted(ATTACKS))
@pytest.mark.parametrize("task_idx", [0, 1])
def test_tampering_scores_zero(attack: str, task_idx: int, tmp_path: Path) -> None:
    task_id = PACK.tasks[task_idx].id
    # Tamper first, then the correct solution: only the tampering can explain a pass.
    solution = ATTACKS[attack] + "\n" + canonical(task_id)
    assert interpret(run_check(task_id, solution, tmp_path)) == (0.0, None), attack


def test_tests_are_not_on_disk_or_in_argv(tmp_path: Path) -> None:
    task_id = PACK.tasks[0].id
    script = check_text(task_id)
    assert "tests.py" not in script and "<<" not in script  # no test file, no heredocs
    assert "mktemp" not in script
    probe = "import sys\nassert len(sys.argv) == 2\n"  # only the solution path is passed
    assert interpret(run_check(task_id, probe + canonical(task_id), tmp_path)) == (1.0, None)
    assert list((tmp_path / "tmpdir").iterdir()) == []  # the check wrote nothing to temp


def test_missing_solution_fails_rather_than_errors(tmp_path: Path) -> None:
    assert interpret(run_check(PACK.tasks[0].id, None, tmp_path)) == (0.0, None)


def test_infinite_loop_is_a_fail_not_an_infra_error(tmp_path: Path) -> None:
    task_id = PACK.tasks[0].id
    script = check_text(task_id)
    assert "timeout -k 5 60 " in script
    shortened = script.replace("timeout -k 5 60 ", "timeout -k 1 2 ")  # test-only: 2 s, not 60 s
    loop = "import time\nwhile True:\n    time.sleep(1)\n"
    assert interpret(run_check(task_id, loop, tmp_path, shortened)) == (0.0, None)


def test_tool_missing_before_candidate_code_is_exit_127(tmp_path: Path) -> None:
    task_id = PACK.tasks[0].id
    script = check_text(task_id).replace("for tool in python3", "for tool in no-such-tool-xyz")
    result = run_check(task_id, canonical(task_id), tmp_path, script)
    assert result.exit_code == 127 and interpret(result)[0] is None
