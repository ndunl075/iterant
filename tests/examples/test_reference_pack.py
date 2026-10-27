"""The reference agent's task pack: it loads, hashes stably, is regenerable, and grades.

Grading runs the real ``Grader`` against a fake sandbox backend that executes a task's ``run``
script locally (the embedded Python program, under ``python -I``), so the real pack files, the
real hash check and the real verdict logic are all exercised without a container runtime.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from iterant.contracts import ExecResult, NetMode, Sandbox, Split
from iterant.grader import Grader
from iterant.grader.grader import AGENT_DIR
from iterant.packs import PackError, compute_content_hash, load_pack, tasks_in, verify
from iterant.packs.hashing import PackHashMismatch

REPO = Path(__file__).resolve().parents[2]
AGENT = REPO / "examples" / "reference-agent"
PACK = AGENT / "evals" / "reference-pack"
# Pinned: fails if any byte of the pack changes. Update on purpose only (regenerate first).
PINNED_HASH = "29c9086eff6c1c47f753e75a3f7a0b3733bc1b84adb8d5adab91407712799c09"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reference_pack_generate", PACK.parent / "generate.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- the pack on disk -------------------------------------------------------------


def test_pack_loads_with_a_pinned_hash() -> None:
    pack = load_pack(PACK)
    assert pack.name == "reference-qa"
    assert pack.content_hash == compute_content_hash(PACK) == PINNED_HASH
    verify(pack, PINNED_HASH, PACK)
    with pytest.raises(PackHashMismatch):
        verify(pack, "0" * 64, PACK)


def test_pack_size_and_splits() -> None:
    pack = load_pack(PACK)
    assert 40 <= len(pack.tasks) <= 60
    for split in Split:
        assert tasks_in(pack, split), split
    assert not (set(tasks_in(pack, Split.TRAIN)) & set(tasks_in(pack, Split.SELECT)))


def test_grader_image_is_the_cruxeval_digest() -> None:
    ours = tomllib.loads((PACK / "pack.toml").read_text(encoding="utf-8"))["grader_image"]
    theirs = tomllib.loads((REPO / "packs" / "cruxeval" / "pack.toml").read_text("utf-8"))[
        "grader_image"
    ]
    assert ours == theirs and re.search(r"@sha256:[0-9a-f]{64}$", ours)


def test_committed_pack_is_what_the_generator_writes(tmp_path: Path) -> None:
    _generator().main(tmp_path / "again")
    assert compute_content_hash(tmp_path / "again") == compute_content_hash(PACK)


def test_generator_facts_match_the_agents_knowledge_base() -> None:
    kb = json.loads((AGENT / "kb" / "facts.json").read_text(encoding="utf-8"))
    assert _generator().FACTS == kb


def test_pack_is_never_in_the_agent_image_context() -> None:
    ignored = (AGENT / ".containerignore").read_text(encoding="utf-8").split()
    assert "evals" in ignored


def test_pack_points_at_by_iterant_toml() -> None:
    cfg = tomllib.loads((AGENT / "iterant.toml").read_text(encoding="utf-8"))
    assert cfg["eval"]["pack"] == "evals/reference-pack"
    load_pack(AGENT / cfg["eval"]["pack"])


def test_questions_are_visible_and_answers_are_not() -> None:
    for task_dir in (PACK / "tasks").iterdir():
        question = json.loads((task_dir / "input.json").read_text(encoding="utf-8"))
        assert set(question) == {"question"}
        expected = json.loads((task_dir / "check").read_text(encoding="utf-8"))
        run = (task_dir / "run").read_text(encoding="utf-8")
        # the run script carries no per-task expected value
        for value in expected.values():
            if isinstance(value, str) and value not in _generator().FACTS.values():
                assert f'"{value}"' not in run


def test_expected_values_against_independent_arithmetic() -> None:
    def check(task: str) -> dict[str, object]:
        return json.loads((PACK / "tasks" / task / "check").read_text(encoding="utf-8"))

    assert check("arith-easy-01") == {"number": "43"}
    assert check("arith-easy-08") == {"number": "62.5"}
    assert check("arith-hard-01") == {"number": str(48213 * 379)}
    assert check("arith-hard-06") == {"number": "142.86"}
    assert check("arith-hard-07") == {"number": f"{1000 * 1.05**10:.2f}"}
    assert check("fact-math-07") == {"number": "10"}
    assert check("fact-01") == {"facts": ["lisbon"]}
    assert check("yes-no-06") == {"verdict": "no"}


# --- grading through the real Grader ------------------------------------------------


@dataclass
class LocalSandbox:
    """Runs ``cd <dir> && exec sh <run>`` by feeding the run script's heredoc program to
    ``python -I`` with the agent's files in a temp directory. Everything else is a no-op."""

    tmp: Path
    files: dict[str, bytes] = field(default_factory=lambda: {})

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        return self.files[path]

    def destroy(self) -> None:
        pass

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        if "exec sh " not in cmd:
            return ExecResult(0, "", "", False, 0.0)
        work = self.tmp / "work"
        work.mkdir(exist_ok=True)
        for path, data in self.files.items():
            if path.startswith(AGENT_DIR + "/"):
                (work / path.removeprefix(AGENT_DIR + "/")).write_bytes(data)
        script = next(d for p, d in self.files.items() if p.endswith("run.sh")).decode()
        head, sep, rest = script.partition("<<'ITERANT_PY'\n")
        assert sep and head.startswith("exec python3 -I -")
        program = rest.partition("\nITERANT_PY\n")[0] + "\n"
        done = subprocess.run(
            [sys.executable, "-I", "-"],
            input=program.encode(),
            cwd=work,
            capture_output=True,
            timeout=60,
            check=False,
        )
        return ExecResult(done.returncode, done.stdout.decode(), done.stderr.decode(), False, 0.0)


class LocalBackend:
    def __init__(self, tmp: Path) -> None:
        self._tmp = tmp
        self._n = 0

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        self._n += 1
        directory = self._tmp / f"sbx{self._n}"
        directory.mkdir()
        return LocalSandbox(directory)


@pytest.fixture
def grader(tmp_path: Path) -> Grader:
    return Grader(LocalBackend(tmp_path), load_pack(PACK), PACK)  # re-hashes the pack root


def _grade(grader: Grader, task_id: str, answer: str | None, *, raw: bytes | None = None) -> float:
    pack_task = next(t for t in load_pack(PACK).tasks if t.id == task_id)
    body = raw if raw is not None else json.dumps({"answer": answer}).encode()
    result = grader.grade(pack_task, {"answer.json": body})
    assert result.score is not None, result.error
    return result.score


@pytest.mark.parametrize(
    ("task", "good", "bad"),
    [
        ("arith-easy-01", "17 plus 26 is 43.", "It is 44."),
        ("arith-easy-08", "250 / 4 = 62.5", "That is 63."),
        ("arith-hard-01", f"The product is {48213 * 379:,}.", "About 18 million."),
        ("arith-hard-06", "1000 / 7 is about 142.86", "1000 / 7 is 142.9"),
        ("arith-medium-04", "The total is $64.97", "The total is 59.97 for the three items"),
        ("fact-math-07", "You can buy 10 units.", "You can buy 11 units."),
        ("fact-01", "Lisbon", "Porto"),
        ("fact-05", "The company has 42 employees.", "42 employees, founded in 2024."),
        ("fact-09", "$19.99", "29.99"),
        ("yes-no-01", "Yes, it is.", "No."),
        ("yes-no-06", "No, 91 = 7 * 13.", "Yes."),
    ],
)
def test_known_good_and_bad_answers(grader: Grader, task: str, good: str, bad: str) -> None:
    assert _grade(grader, task, good) == 1.0
    assert _grade(grader, task, bad) == 0.0


def test_unparseable_or_unsafe_output_scores_zero(grader: Grader) -> None:
    task = "arith-easy-01"
    assert _grade(grader, task, None, raw=b"43") == 0.0  # not an object
    assert _grade(grader, task, None, raw=b'{"answer": 43}') == 0.0  # not a string
    assert _grade(grader, task, None, raw=b'{"answer": "NaN", "x": NaN}') == 0.0
    assert _grade(grader, task, None, raw=b"not json") == 0.0
    assert _grade(grader, task, "no digits here") == 0.0
    # agent text is data: nothing in it is run, whatever it says
    assert _grade(grader, task, "__import__('os').system('true') 43") == 1.0


def test_grader_refuses_an_edited_pack(tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in PACK.rglob("*"):
        target = copy / path.relative_to(PACK)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.write_bytes(path.read_bytes())
    pack = load_pack(copy)
    (copy / "tasks" / "arith-easy-01" / "check").write_bytes(b'{"number": "44"}\n')
    with pytest.raises(PackError):
        Grader(LocalBackend(tmp_path), pack, copy)
