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

The check is a shell script (the grader runs ``sh check`` with ``/out`` as cwd). It
runs the candidate's code and the upstream tests under a timeout in one Python
process and passes only if the process prints a per-run random nonce together with
PASS. A candidate that simply prints PASS or exits 0 does not pass. Candidate code
that inspects the harness's stack frames could still forge it; separating the two
needs a second user in the sandbox image and is left to the sandbox work.
"""

from __future__ import annotations

import argparse
import ast
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
import io, sys, unittest

nonce = sys.stdin.read().strip()
real_stdout = sys.__stdout__
tests_path, solution_path = sys.argv[1], sys.argv[2]
ns = {"__name__": "solution"}
ok = False
try:
    sys.stdout = io.StringIO()
    exec(compile(open(solution_path, encoding="utf-8").read(), "solution.py", "exec"), ns)
    exec(compile(open(tests_path, encoding="utf-8").read(), "tests.py", "exec"), ns)
    suite = unittest.TestSuite()
    for obj in list(ns.values()):
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    ok = result.wasSuccessful() and result.testsRun > 0
except BaseException:
    ok = False
sys.stdout = real_stdout
if ok:
    print("ITERANT_RESULT", nonce, "PASS")
"""

CHECK = """#!/bin/sh
# Hidden check for {task_id}. Reads /out/solution.py; see packs/bigcodebench/convert.py.
command -v python3 >/dev/null 2>&1 || exit 127
command -v timeout >/dev/null 2>&1 || exit 127
T=$(mktemp -d) || exit 127
cat > "$T/tests.py" <<'ITERANT_TESTS_EOF'
{test}
ITERANT_TESTS_EOF
cat > "$T/run.py" <<'ITERANT_RUN_EOF'
{harness}
ITERANT_RUN_EOF
NONCE=$(python3 -c 'import secrets; print(secrets.token_hex(16))') || exit 127
OUT=$(printf '%s' "$NONCE" |
  timeout 60 python3 "$T/run.py" "$T/tests.py" /out/solution.py 2>/dev/null)
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
    check = CHECK.format(
        task_id=row["task_id"], test=row["test"].rstrip("\n"), harness=HARNESS.strip()
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
