"""CRUXEval-O pack: converter, canonicalizer parity, parse-only grading, committed pack."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import ExecResult, Sandbox, SandboxBackend, Split
from iterant.grader import Grader
from iterant.grader.grader import interpret, parse_json
from iterant.packs import (
    PackError,
    compute_content_hash,
    cruxeval,
    cruxeval_canon,
    load_pack,
    tasks_in,
)

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "cruxeval_sample.jsonl"
COMMITTED = Path(__file__).resolve().parents[2] / "packs" / "cruxeval"
# Pinned: fails if any byte of the committed pack changes. Update on purpose only.
PINNED_COMMITTED_HASH = "367b734e9cd4ed8aef59aefab07db0d6f6a371bbe927434d48b1f5ec5116018b"
IMAGE = os.environ.get("ITERANT_TEST_IMAGE", "docker.io/library/python:3.12-slim")


def sample_rows() -> dict[str, dict[str, str]]:
    return {r["id"]: r for r in cruxeval.read_rows(SAMPLE.read_bytes())}


def program(run_text: str) -> str:
    """The Python program embedded in a ``run`` script."""
    head, sep, rest = run_text.partition("<<'ITERANT_PY'\n")
    assert sep and head.startswith("# ") and "exec python3 -I -" in head
    body, end, tail = rest.partition("\nITERANT_PY\n")
    assert end and tail == ""
    return body + "\n"


RUN_PROGRAM = program(cruxeval.run_script().decode())


def run_answer(tmp_path: Path, answer: bytes | None) -> ExecResult:
    """Execute the extracted ``run`` program with this interpreter, like the grader would."""
    work = tmp_path / "out"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    if answer is not None:
        (work / "answer.txt").write_bytes(answer)
    done = subprocess.run(
        [sys.executable, "-I", "-"],
        input=RUN_PROGRAM.encode(),
        cwd=work,
        capture_output=True,
        timeout=60,
    )
    return ExecResult(done.returncode, done.stdout.decode(), done.stderr.decode(), False, 0.0)


def score(tmp_path: Path, expected_literal: str, answer: bytes | None) -> float:
    expected = json.loads(cruxeval.expected_check(expected_literal))
    return interpret(run_answer(tmp_path, answer), expected)


# ---------------------------------------------------------------- canonicalizer / run script


def test_embedded_program_is_the_canonicalizer_module_text() -> None:
    assert RUN_PROGRAM == cruxeval.canon_source()
    assert "\r" not in RUN_PROGRAM


@pytest.mark.parametrize(
    "answer",
    [
        "1",
        "True",
        "1.0",
        "-0.0",
        "[1, 'a', (2, 3), {4, 5}, {'k': b'v'}]",
        "{'b': {2, 1}, 'a': None}",
        "'caf\\u00e9'",
        "1e999",
        "1+2j",
        "[" * 25 + "]" * 25,
        "1 x",
        "__import__('os')",
        "",
    ],
)
def test_run_script_matches_in_process_canonicalizer(tmp_path: Path, answer: str) -> None:
    got = run_answer(tmp_path, answer.encode())
    try:
        want = cruxeval_canon.encode_answer(cruxeval_canon.parse_answer(answer.encode())) + "\n"
    except Exception:
        assert got.exit_code != 0 and got.stdout == ""
    else:
        assert got.exit_code == 0 and got.stdout == want


@pytest.mark.parametrize(
    ("expected", "answer", "passes"),
    [
        # numbers: == semantics, but bool stays apart
        ("1", "1", True),
        ("1", "1.0", True),
        ("1.0", "1", True),
        ("1", "True", False),
        ("True", "1", False),
        ("False", "0", False),
        ("0", "-0.0", True),
        ("-0.0", "0", True),
        ("0.5", "0.5", True),
        ("0.1", "0.10000000000000002", False),
        ("1e999", "1e999", True),
        ("1e999", "-1e999", False),
        ("1+2j", "1+2j", True),
        ("1+2j", "1+3j", False),
        ("[1, 1.0, True]", "[1.0, 1, True]", True),
        ("[1, 1.0, True]", "[True, 1.0, 1]", False),
        # containers keep types apart
        ("[1]", "(1,)", False),
        ("(1,)", "[1]", False),
        ("{1}", "[1]", False),
        ("[1]", "{1}", False),
        ("{}", "[]", False),
        ("()", "[]", False),
        ("{'a': 1}", "{'a': True}", False),
        ("{1: 'a'}", "{True: 'a'}", False),
        # sets and dicts ignore order
        ("{3, 1, 2}", "{1, 2, 3}", True),
        ("[{2, 1}, {4, 3}]", "[{1, 2}, {3, 4}]", True),
        ("{(1, 2), (0, 5)}", "{(0, 5), (1, 2)}", True),
        ("{'a': 1, 'b': 2}", "{'b': 2, 'a': 1}", True),
        ("{'a': {2, 1}}", "{'a': {1, 2}}", True),
        ("[1, 2]", "[2, 1]", False),
        ("(1, 2)", "(2, 1)", False),
        ("{1, 2}", "{1, 3}", False),
        # strings and bytes
        ("'1'", "1", False),
        ("1", "'1'", False),
        ("'10'", "'1e1'", False),
        ("'0'", "'-0'", False),
        ("'caf\\u00e9'", "'caf\\xe9'", True),
        ("'caf\\u00e9'", "'cafe\\u0301'", False),
        ("'\\u4e2d'", "'\\u4e2d '", False),
        ("'a' 'b'", "'ab'", True),
        ("b'ab'", "'ab'", False),
        ("b'ab'", "b'ab'", True),
        ("None", "None", True),
        ("None", "'None'", False),
        ("''", "None", False),
        # big ints
        ("1" + "0" * 40, "1" + "0" * 40, True),
        ("1" + "0" * 40, "1" + "0" * 39, False),
        ("1" + "0" * 40, "1" + "0" * 39 + "1", False),
        ("1152921504606846976", "1152921504606846976.0", True),
        # layout around the literal
        ("[1, 2]", "  [1,\n 2]\n\n", True),
        ("[1, 2]", "[1, 2]  # note", True),
        ("[1, 2]", "```\n[1, 2]\n```", True),
        ("[1, 2]", "```python\n[1, 2]\n```\n", True),
        ("[1, 2]", "```py\n[1, 2]\n```", True),
        ("[1, 2]", "```Python\n[1, 2]\n```", True),
        # lone surrogates are not the code point their escapes spell
        ("'\\ud83d\\ude00'", "'\\U0001F600'", False),
        ("'\\U0001F600'", "'\\ud83d\\ude00'", False),
        ("'\\ud83d\\ude00'", "'\\ud83d\\ude00'", True),
        ("'\\U0001F600'", "'\\U0001F600'", True),
        ("'\\ud800'", "'\\ud800'", True),
        ("'\\ud800'", "'\\ud801'", False),
        ("[1, 2]", "```[1, 2]```", True),
        ("[1, 2]", "```bash\n[1, 2]\n```", False),
        ("[1, 2]", "```\n```\n[1, 2]\n```\n```", False),
        ("[1, 2]", "```\n[1, 2]", False),
        ("[1, 2]", "`[1, 2]`", False),
        ("[1, 2]", "The answer is [1, 2]", False),
        # junk and non-literals
        ("1", "1 2", False),
        ("1", "1\nprint(1)", False),
        ("1", "1; 2", False),
        ("[1]", "[1]]", False),
        ("[1]", "[1] x", False),
        ("[1]", "[1]\n[1]", False),
        ("2", "1+1", False),
        ("2", "len([1, 2])", False),
        ("[0, 1]", "[x for x in range(2)]", False),
        ("[0, 1]", "list(range(2))", False),
        ("{1, 2}", "{1} | {2}", False),
        ("1", "__import__('os').getcwd()", False),
        ("1", "x", False),
        ("1", "...", False),
        ("1", "", False),
        ("1", "   \n", False),
    ],
)
def test_grading_semantics(tmp_path: Path, expected: str, answer: str, passes: bool) -> None:
    assert score(tmp_path, expected, answer.encode()) == (1.0 if passes else 0.0)


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    assert score(tmp_path, "[1]", b"\xef\xbb\xbf[1]") == 1.0
    assert score(tmp_path, "[1]", b"\xef\xbb\xbf```\n[1]\n```") == 1.0


def test_planted_modules_in_the_out_dir_are_ignored(tmp_path: Path) -> None:
    forged = b"import sys\nprint('FORGED')\nsys.exit(0)\n"
    work = tmp_path / "out"
    for name in ("json.py", "ast.py", "sys.py", "sitecustomize.py", "usercustomize.py"):
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        (work / name).write_bytes(forged)
        (work / "answer.txt").write_bytes(b"[1]")
        done = subprocess.run(
            [sys.executable, "-I", "-"], input=RUN_PROGRAM.encode(), cwd=work, capture_output=True
        )
        assert done.stdout == b'{"answer":["list",["int","1"]]}\n', name


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")
def test_run_script_through_sh_with_a_stub_python3(tmp_path: Path) -> None:
    """The heredoc path: ``sh run`` finds ``python3`` on PATH and feeds it the program."""
    stub = tmp_path / "bin"
    stub.mkdir()
    exe = Path(sys.executable).as_posix()
    (stub / "python3").write_bytes(f'#!/bin/sh\nexec "{exe}" "$@"\n'.encode())
    (stub / "python3").chmod(0o755)
    run = tmp_path / "run"
    run.write_bytes(cruxeval.run_script())
    work = tmp_path / "work"
    work.mkdir()
    env = {**os.environ, "PATH": f"{stub.as_posix()}{os.pathsep}{os.environ['PATH']}"}

    def sh(answer: bytes | None) -> subprocess.CompletedProcess[bytes]:
        (work / "answer.txt").unlink(missing_ok=True)
        if answer is not None:
            (work / "answer.txt").write_bytes(answer)
        return subprocess.run(
            ["sh", run.as_posix()], cwd=work, env=env, capture_output=True, timeout=60
        )

    good = sh(b"{2, 1}")
    if good.returncode == 127:
        pytest.skip("stub python3 not runnable through sh on this platform")
    assert good.returncode == 0
    assert good.stdout.strip() == cruxeval_canon.encode_answer({1, 2}).encode()
    assert sh(b"1 x").returncode == 1
    assert sh(None).returncode == 1


def test_missing_file_is_a_nonzero_exit(tmp_path: Path) -> None:
    got = run_answer(tmp_path, None)
    assert got.exit_code != 0 and got.stdout == ""
    assert score(tmp_path, "1", None) == 0.0


PATHOLOGICAL = {
    "not-utf8": b"\xff\xfe1",
    "over-size-cap": b"1" + b" " * (cruxeval_canon.MAX_ANSWER_BYTES + 1),
    "deeper-than-cap": b"[" * 100 + b"]" * 100,
    "parser-limits": b"[" * 30_000 + b"]" * 30_000,
    "int-too-large": b"1" * 5000,
    "open-parens": b"(" * 50_000,
    "unterminated": b"'unterminated",
    "nul": b"\x00",
}


@pytest.mark.parametrize("answer", PATHOLOGICAL.values(), ids=PATHOLOGICAL.keys())
def test_pathological_answers_fail_cleanly(tmp_path: Path, answer: bytes) -> None:
    got = run_answer(tmp_path, answer)
    assert got.exit_code == 1 and got.stdout == ""
    assert "Traceback" not in got.stderr


def test_nesting_up_to_the_cap_is_accepted(tmp_path: Path) -> None:
    depth = cruxeval_canon.MAX_DEPTH
    text = "[" * depth + "]" * depth
    assert score(tmp_path, text, text.encode()) == 1.0
    deeper = "[" * (depth + 1) + "]" * (depth + 1)
    assert score(tmp_path, text, deeper.encode()) == 0.0


def test_encoding_depth_fits_the_grader_limit() -> None:
    depth = cruxeval_canon.MAX_DEPTH
    for open_, close in (("{1: ", "}"), ("(", ",)"), ("[", "]")):
        text = open_ * depth + "0" + close * depth
        parse_json(cruxeval_canon.encode_answer(ast.literal_eval(text)))


def test_nan_and_infinities() -> None:
    c = cruxeval_canon.canonical
    nan = float("nan")
    assert c(nan) == c(float("nan")) != c(float("inf"))
    assert c(float("inf")) != c(float("-inf"))
    assert c([nan]) != c((nan,))
    assert c({nan, 1.5}) == c({1.5, nan})


def test_canonical_rejects_unsupported_types() -> None:
    for bad in (object(), range(2), ..., len):
        with pytest.raises(TypeError):
            cruxeval_canon.canonical(bad)


def test_signed_zero_and_integral_floats_collapse() -> None:
    c = cruxeval_canon.canonical
    assert c(-0.0) == c(0) == c(0.0)
    assert c(True) != c(1)
    assert c(1) == c(1.0)
    assert c(2.5) != c(2)
    assert c(2**53 + 1) != c(float(2**53))


# ---------------------------------------------------------------- converter


def test_converter_is_deterministic(tmp_path: Path) -> None:
    cruxeval.convert(SAMPLE, tmp_path / "a")
    cruxeval.convert(SAMPLE, tmp_path / "b")
    assert compute_content_hash(tmp_path / "a") == compute_content_hash(tmp_path / "b")


def test_converter_refuses_to_overwrite(tmp_path: Path) -> None:
    cruxeval.convert(SAMPLE, tmp_path / "p")
    with pytest.raises(PackError):
        cruxeval.convert(SAMPLE, tmp_path / "p")


def test_converter_checks_dataset_hash(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="sha256"):
        cruxeval.convert(SAMPLE, tmp_path / "p", expected_sha256="0" * 64)


def test_converter_rejects_bad_rows(tmp_path: Path) -> None:
    good = {"id": "x1", "code": "def f(a):\n    return a", "input": "1", "output": "1"}
    for bad in (
        b"not json\n",
        b"[1]\n",
        json.dumps({**good, "output": 3}).encode(),
        json.dumps({**good, "id": "../x"}).encode(),
        json.dumps({**good, "output": "f(1)"}).encode(),  # not a literal
        json.dumps({**good, "output": "[" * 40 + "]" * 40}).encode(),  # too deep
        (json.dumps(good) + "\n" + json.dumps(good)).encode(),  # duplicate id
        b"",
    ):
        src = tmp_path / "bad.jsonl"
        src.write_bytes(bad)
        with pytest.raises(PackError):
            cruxeval.convert(src, tmp_path / "out")
        shutil.rmtree(tmp_path / "out", ignore_errors=True)


def test_input_never_leaks_the_output(tmp_path: Path) -> None:
    cruxeval.convert(SAMPLE, tmp_path / "p")
    rows = sample_rows()
    for task_dir in sorted((tmp_path / "p" / "tasks").iterdir()):
        row = rows[task_dir.name]
        visible = json.loads((task_dir / "input.json").read_bytes())
        assert set(visible) == {"code", "call", "instructions"}
        assert visible["code"] == row["code"]
        assert visible["call"] == f"f({row['input']})"
        assert visible["instructions"] == cruxeval.INSTRUCTIONS
        # the check holds the expected value; nothing visible or executed derives from it
        assert (task_dir / "check").read_bytes() == cruxeval.expected_check(row["output"])
        assert (task_dir / "run").read_bytes() == cruxeval.run_script()


def test_check_round_trips(tmp_path: Path) -> None:
    cruxeval.convert(SAMPLE, tmp_path / "p")
    for task_id, row in sample_rows().items():
        raw = (tmp_path / "p" / "tasks" / task_id / "check").read_bytes()
        assert raw.endswith(b"\n") and raw.isascii()
        expected = json.loads(raw)
        assert list(expected) == ["answer"]
        # the dataset's own output literal, written as the agent's answer, passes
        assert interpret(run_answer(tmp_path, row["output"].encode()), expected) == 1.0


def test_run_script_is_one_python_isolated_heredoc() -> None:
    text = cruxeval.run_script().decode()
    assert text.count("exec python3 -I -") == 1 and text.count("\nITERANT_PY\n") == 1
    assert "literal_eval" in text and "eval(" not in text.replace("literal_eval(", "")
    assert text.isascii()


def test_tags_policy_and_splits(tmp_path: Path) -> None:
    report = cruxeval.convert(SAMPLE, tmp_path / "p")
    pack = load_pack(tmp_path / "p")
    assert report.n_tasks == len(pack.tasks) == len(sample_rows())
    assert {t.tags for t in pack.tasks} <= {("short",), ("medium",), ("long",)}
    assert cruxeval.length_tag("x" * 99) == "short"
    assert cruxeval.length_tag("x" * 100) == "medium"
    assert cruxeval.length_tag("x" * 160) == "long"
    assert pack.name == "cruxeval-o"
    meta = tomllib.loads((tmp_path / "p" / "pack.toml").read_text())
    assert meta["policy"] == {"agent_code_exec": "forbidden"}


# ---------------------------------------------------------------- committed pack


def test_committed_pack_loads_with_a_stable_hash() -> None:
    pack = load_pack(COMMITTED)
    assert pack.name == "cruxeval-o" and len(pack.tasks) == 800
    assert pack.content_hash == compute_content_hash(COMMITTED) == PINNED_COMMITTED_HASH
    assert {t.tags for t in pack.tasks} == {("short",), ("medium",), ("long",)}
    for split in Split:
        assert tasks_in(pack, split)
    assert len(tasks_in(pack, Split.SELECT)) == 80 and len(tasks_in(pack, Split.HOLDOUT)) == 40
    for name in ("LICENSE", "NOTICE", "README.md"):
        assert (COMMITTED / name).is_file()
    assert "MIT License" in (COMMITTED / "LICENSE").read_text()


def test_committed_pack_files_are_uniform() -> None:
    run = cruxeval.run_script()
    for task_dir in sorted((COMMITTED / "tasks").iterdir()):
        assert (task_dir / "run").read_bytes() == run
        visible = json.loads((task_dir / "input.json").read_bytes())
        assert set(visible) == {"code", "call", "instructions"}
        assert visible["call"].startswith("f(") and visible["call"].endswith(")")
        assert list(json.loads((task_dir / "check").read_bytes())) == ["answer"]


def real_rows() -> list[dict[str, str]]:
    rows = [r for i, r in sample_rows().items() if i.startswith("sample_")]
    assert len(rows) == 5
    return rows


def test_committed_pack_matches_real_upstream_rows(tmp_path: Path) -> None:
    by_id = {t.id: t for t in load_pack(COMMITTED).tasks}
    for row in real_rows():
        task = by_id[row["id"]]
        visible = json.loads((COMMITTED / task.input).read_bytes())
        assert visible["code"] == row["code"] and visible["call"] == f"f({row['input']})"
        assert (COMMITTED / task.check).read_bytes() == cruxeval.expected_check(row["output"])
        assert score(tmp_path, row["output"], row["output"].encode()) == 1.0


# ---------------------------------------------------------------- real sandboxes

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
    prefix = "cruxeval-int-" + kind
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


def test_real_tasks_graded_in_a_real_sandbox(backend: Tracking) -> None:
    by_id = {t.id: t for t in load_pack(COMMITTED).tasks}
    grader = Grader(backend, IMAGE, COMMITTED, timeout=30)
    for row in real_rows():
        task = by_id[row["id"]]
        out = row["output"]
        good = grader.grade(task, {"answer.txt": out.encode()})
        fenced = grader.grade(task, {"answer.txt": f"```python\n{out}\n```\n".encode()})
        wrong = grader.grade(task, {"answer.txt": b"None" if out != "None" else b"0"})
        junk = grader.grade(task, {"answer.txt": out.encode() + b"\nprint(1)"})
        missing = grader.grade(task, {})
        assert good.score == 1.0 and good.error is None, row["id"]
        assert fenced.score == 1.0 and fenced.error is None, row["id"]
        for r in (wrong, junk, missing):
            assert r.score == 0.0 and r.error is None, (row["id"], r)
    assert len(backend.created) == 25
