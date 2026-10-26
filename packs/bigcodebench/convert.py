"""Convert BigCodeBench rows into an Iterant task pack.

BigCodeBench (https://github.com/bigcode-project/bigcodebench, Apache-2.0) gives each
task an instruction and a ``unittest`` suite. We keep only tasks that run on the
stdlib alone and need no network or GUI, so the check runs offline in a plain
Python image.

Usage (from the repo root; the rows come from the public Hugging Face dataset)::

    uv run python packs/bigcodebench/convert.py --fetch --out packs/bigcodebench/pack \\
        --limit 20 --solutions-dir packs/bigcodebench/fixture-solutions
    uv run python packs/bigcodebench/convert.py --rows rows.jsonl --out DIR --limit 0

``--limit 0`` converts every eligible task. The agent writes ``/out/solution.py``.

The check is a shell script that the grader runs as ``sh -c`` (never from a file in
the sandbox). It runs the candidate's code and the upstream tests under a timeout in
one hardened Python process (``-I -S -B``). The tests are compiled into the harness
before any candidate code runs and never touch disk. Defense in depth: a per-run
nonce, identity checks on the unittest machinery and builtins before and after, and an
expected test count taken from the test source. This stops casual and accidental
tampering (printing PASS, ``sys.exit(0)``, patching ``TestCase.run``), not a candidate
that attacks the harness itself (e.g. frame inspection); that residual risk is in
ARCHITECTURE.md section 9 and needs W3 red-team coverage.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

from iterant.packs import split_pack

DATASET = "bigcode/bigcodebench"
SPLIT = "v0.1.4"
REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
PAGE = 100

# Stdlib modules that touch the network, spawn processes or need a display.
DENIED = {
    "socket", "http", "urllib", "smtplib", "ftplib", "ssl", "imaplib", "poplib", "telnetlib",
    "socketserver", "xmlrpc", "wsgiref", "webbrowser", "asyncio", "select", "selectors",
    "subprocess", "multiprocessing", "threading", "concurrent", "ctypes", "signal", "cgi",
    "tkinter", "turtle", "platform", "sched", "pty", "fcntl", "resource", "idlelib",
}  # fmt: skip

HARNESS = r"""
import ast, builtins, io, sys, unittest

_nonce = sys.stdin.read().strip()
_real_out = sys.__stdout__
_unittest = unittest
_TESTS = @@TESTS@@
_CLASSES = (
    unittest.TestCase, unittest.TestSuite, unittest.BaseTestSuite, unittest.TestResult,
    unittest.TextTestResult, unittest.TextTestRunner, unittest.TestLoader,
)
_NAMES = (
    "TestCase", "TestSuite", "TestResult", "TextTestRunner", "TextTestResult", "TestLoader",
    "defaultTestLoader",
)
_class_snap = {c: dict(vars(c)) for c in _CLASSES}
_mod_snap = {n: getattr(unittest, n) for n in _NAMES}
_builtins_snap = dict(vars(builtins))
_TC, _Loader, _Runner, _Suite = (
    unittest.TestCase, unittest.TestLoader, unittest.TextTestRunner, unittest.TestSuite,
)
_tests_code = compile(_TESTS, "tests.py", "exec")  # tests never touch disk
_expected = {}
for _node in ast.parse(_TESTS).body:
    if isinstance(_node, ast.ClassDef) and any(
        (isinstance(b, ast.Name) and b.id == "TestCase")
        or (isinstance(b, ast.Attribute) and b.attr == "TestCase")
        for b in _node.bases
    ):
        _expected[_node.name] = {
            f.name
            for f in _node.body
            if isinstance(f, ast.FunctionDef) and f.name.startswith("test")
        }
if not any(_expected.values()):
    sys.exit(125)
print("ITERANT_READY", _nonce, file=_real_out, flush=True)


def _intact():
    if sys.modules.get("unittest") is not _unittest:
        return False
    if any(getattr(_unittest, n, None) is not v for n, v in _mod_snap.items()):
        return False
    for c, snap in _class_snap.items():
        now = vars(c)
        if set(now) != set(snap) or any(now[k] is not snap[k] for k in snap):
            return False
    now = vars(builtins)
    return all(k in now and now[k] is v for k, v in _builtins_snap.items())


_ok = False
try:
    sys.stdout = io.StringIO()
    _ns = {"__name__": "solution"}
    _solution_src = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    exec(compile(_solution_src, "solution.py", "exec"), _ns)
    if _intact():
        exec(_tests_code, _ns)
        _suite, _total = _Suite(), 0
        _good = _intact()
        for _name, _methods in _expected.items():
            _cls = _ns.get(_name)
            if not (isinstance(_cls, type) and issubclass(_cls, _TC)):
                _good = False
                break
            _found = _Loader().getTestCaseNames(_cls)
            _good = _good and _methods <= set(_found)
            _total += len(_found)
            _suite.addTests(_Loader().loadTestsFromTestCase(_cls))
        _result = _Runner(stream=io.StringIO()).run(_suite)
        _ok = (
            _good
            and _intact()
            and _result.testsRun == _total
            and _result.wasSuccessful()
            and not (_result.skipped or _result.expectedFailures or _result.unexpectedSuccesses)
        )
except BaseException:
    _ok = False
sys.stdout = _real_out
if _ok:
    print("ITERANT_RESULT", _nonce, "PASS", file=_real_out)
"""

CHECK = """#!/bin/sh
# Hidden check for {task_id}: reads /out/solution.py. The grader runs it with
# `sh -c`, never from a file in the sandbox. Exit 127 = could not run (infra);
# anything after the harness has printed READY is pass (0) or fail (1).
# The harness (base64) embeds the upstream tests, see packs/bigcodebench/convert.py.
for tool in python3 timeout base64; do
  command -v "$tool" >/dev/null 2>&1 || exit 127
done
H=$(printf '%s' '{harness}' | base64 -d) || exit 127
NONCE=$(python3 -I -S -B -c 'import secrets; print(secrets.token_hex(16))') || exit 127
OUT=$(printf '%s' "$NONCE" |
  timeout -k 5 60 python3 -I -S -B -c "$H" /out/solution.py 2>/dev/null)
case "$OUT" in
  *"ITERANT_READY $NONCE"*) ;;
  *) exit 127 ;;
esac
case "$OUT" in
  *"ITERANT_RESULT $NONCE PASS"*) exit 0 ;;
esac
exit 1
"""

NOTICE = """This pack is derived from BigCodeBench (https://github.com/bigcode-project/bigcodebench),
dataset {dataset} split {split} at revision {revision}, licensed under the Apache License 2.0
(see LICENSE.upstream).

Tasks were converted by packs/bigcodebench/convert.py: only tasks that use the Python
standard library alone and need no network or GUI were kept. The task instruction is the
upstream `instruct_prompt`; the check wraps the upstream `test` code unchanged.

Zhuo et al., "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and
Complex Instructions", arXiv:2406.15877.
"""


def fetch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows?dataset="
            f"{DATASET}&config=default&split={SPLIT}&offset={offset}&length={PAGE}"
        )
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (fixed https URL)
            page = json.load(resp)
        batch = [r["row"] for r in page["rows"]]
        rows.extend(batch)
        offset += PAGE
        if offset >= page["num_rows_total"]:
            return rows


def imported_modules(*sources: str) -> set[str]:
    mods: set[str] = set()
    for src in sources:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # upstream docstrings have invalid escapes
            tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


def eligible(row: dict[str, Any]) -> bool:
    """Stdlib only, nothing from DENIED, judged by the libs field and by actual imports."""
    try:
        libs = set(ast.literal_eval(row["libs"]))
        mods = imported_modules(
            row["complete_prompt"], row["test"], row["complete_prompt"] + row["canonical_solution"]
        )
    except (SyntaxError, ValueError):
        return False
    used = libs | mods
    return bool(used) and all(m in sys.stdlib_module_names and m not in DENIED for m in used)


def task_dir_name(row: dict[str, Any]) -> str:
    return f"bcb-{int(row['task_id'].split('/')[1]):04d}"


def write_task(root: Path, row: dict[str, Any]) -> None:
    task = root / "tasks" / task_dir_name(row)
    task.mkdir(parents=True)
    spec = {
        "task_id": row["task_id"],
        "instruction": row["instruct_prompt"],
        "output_file": "/out/solution.py",
    }
    (task / "input.json").write_bytes((json.dumps(spec, indent=1) + "\n").encode("utf-8"))
    harness = HARNESS.strip().replace("@@TESTS@@", repr(row["test"]))
    check = CHECK.format(
        task_id=row["task_id"], harness=base64.b64encode(harness.encode("utf-8")).decode("ascii")
    )
    (task / "check").write_bytes(check.encode("utf-8"))
    n_libs = len(ast.literal_eval(row["libs"]))
    tags = ["bigcodebench", f"libs-{min(n_libs, 3)}"]
    (task / "task.toml").write_bytes(f"tags = {json.dumps(tags)}\n".encode())


def pick(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Evenly spaced eligible rows in task-number order (deterministic)."""
    ok = sorted((r for r in rows if eligible(r)), key=lambda r: int(r["task_id"].split("/")[1]))
    if limit <= 0 or limit >= len(ok):
        return ok
    return [ok[i * len(ok) // limit] for i in range(limit)]


def convert(rows: list[dict[str, Any]], out: Path, limit: int, solutions: Path | None) -> int:
    chosen = pick(rows, limit)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; refusing to overwrite a pack")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.toml").write_bytes(b'name = "bigcodebench-mini"\nversion = "0.1.0"\n')
    notice = NOTICE.format(dataset=DATASET, split=SPLIT, revision=REVISION)
    (out / "NOTICE").write_bytes(notice.encode("utf-8"))
    licence = Path(__file__).with_name("LICENSE.upstream")
    (out / "LICENSE.upstream").write_bytes(licence.read_bytes())
    for row in chosen:
        write_task(out, row)
        if solutions is not None:
            solutions.mkdir(parents=True, exist_ok=True)
            code = row["complete_prompt"] + row["canonical_solution"] + "\n"
            (solutions / f"{task_dir_name(row)}.solution").write_bytes(code.encode("utf-8"))
    report = split_pack(out)
    print(f"converted {len(chosen)} tasks; splits {dict(report.counts)}")
    for w in report.warnings:
        print("warning:", w)
    return len(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rows", type=Path, help="JSON-lines file of dataset rows")
    src.add_argument("--fetch", action="store_true", help="download rows from Hugging Face")
    ap.add_argument("--out", type=Path, required=True, help="pack directory to create")
    ap.add_argument("--limit", type=int, default=20, help="tasks to keep (0 = all eligible)")
    ap.add_argument("--solutions-dir", type=Path, help="also write canonical solutions here")
    args = ap.parse_args()
    if args.fetch:
        rows = fetch_rows()
    else:
        rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
    convert(rows, args.out, args.limit, args.solutions_dir)


if __name__ == "__main__":
    main()
