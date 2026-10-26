"""Convert the CRUXEval output-prediction set into an Iterant task pack (parse-only grading).

The agent sees a function and a call and writes a Python literal to ``answer.txt``. No
agent-produced code is ever executed: the per-task ``run`` script parses the file with
``ast.literal_eval`` and prints a canonical, type-tagged encoding (``cruxeval_canon``); the
hidden ``check`` holds the encoding of the dataset's expected output, computed here with
``ast.literal_eval`` too. The dataset is untrusted data: it is parsed as JSON and Python source
text, never imported or executed.

Usage: ``python -m iterant.packs.cruxeval cruxeval.jsonl packs/cruxeval [--sha256 HEX]``.
The output directory must not already hold ``pack.toml``, ``splits.json`` or ``tasks/``.
Output is deterministic: the same input gives byte-identical files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from iterant.packs import cruxeval_canon
from iterant.packs.hashing import PackError
from iterant.packs.loader import split_pack
from iterant.packs.splitter import SplitReport

PACK_NAME = "cruxeval-o"
PACK_VERSION = "1"
# python:3.12-slim (multi-arch index digest, looked up 2026-10-26). Grader sandboxes run `run`
# with this image's `python3 -I`; it is never the candidate-built target image.
GRADER_IMAGE = (
    "docker.io/library/python"
    "@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
)
INSTRUCTIONS = "Predict the exact return value. Write only a Python literal to /out/answer.txt."
_HEREDOC = "ITERANT_PY"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Function source length (characters) -> coarse difficulty tag used to stratify the splits.
_SHORT_BELOW = 100
_MEDIUM_BELOW = 160


def length_tag(code: str) -> str:
    if len(code) < _SHORT_BELOW:
        return "short"
    return "medium" if len(code) < _MEDIUM_BELOW else "long"


def canon_source() -> str:
    """Exact text of the canonicalizer module, with LF line endings."""
    text = Path(cruxeval_canon.__file__).read_bytes().decode("utf-8").replace("\r\n", "\n")
    if f"\n{_HEREDOC}\n" in f"\n{text}":
        raise PackError("canonicalizer contains the heredoc terminator")
    return text


def run_script() -> bytes:
    """The ``run`` script, identical for every task: the canonicalizer text in a heredoc."""
    body = canon_source()
    if not body.endswith("\n"):
        body += "\n"
    script = (
        "# CRUXEval-O parse-only check: answer.txt is parsed with ast.literal_eval, never run.\n"
        f"exec python3 -I - <<'{_HEREDOC}'\n{body}{_HEREDOC}\n"
    )
    return script.encode("utf-8")


def _dump(obj: object) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def read_rows(data: bytes) -> list[dict[str, str]]:
    """Validated dataset rows (``id``, ``code``, ``input``, ``output``), in file order."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(data.split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            raw: Any = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise PackError(f"line {number}: not JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise PackError(f"line {number}: not an object")
        obj = cast("dict[str, Any]", raw)
        row: dict[str, str] = {}
        for key in ("id", "code", "input", "output"):
            value = obj.get(key)
            if not isinstance(value, str):
                raise PackError(f"line {number}: {key!r} missing or not a string")
            row[key] = value
        if not _ID.fullmatch(row["id"]) or row["id"] in seen:
            raise PackError(f"line {number}: bad or duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise PackError("no rows")
    return rows


def expected_check(output_literal: str) -> bytes:
    """Hidden ``check`` bytes for a dataset output literal (``ast.literal_eval``, never eval)."""
    try:
        value: object = ast.literal_eval(output_literal)  # pyright: ignore[reportAny]
        text = cruxeval_canon.encode_answer(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise PackError(
            f"cannot canonicalize expected output {output_literal[:60]!r}: {exc}"
        ) from exc
    return (text + "\n").encode("ascii")


def convert(source: Path, out: Path, expected_sha256: str | None = None) -> SplitReport:
    """Write the pack for the dataset at ``source`` into ``out`` and assign its splits."""
    data = source.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise PackError("dataset sha256 does not match the expected value")
    rows = read_rows(data)
    for existing in ("pack.toml", "splits.json", "tasks"):
        if (out / existing).exists():
            raise PackError(f"{out / existing} already exists; refusing to overwrite")
    run = run_script()
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.toml").write_bytes(
        f'name = "{PACK_NAME}"\nversion = "{PACK_VERSION}"\n'
        f'grader_image = "{GRADER_IMAGE}"\n\n'
        '[policy]\nagent_code_exec_policy = "advisory-forbidden"\n'.encode()
    )
    for row in rows:
        task = out / "tasks" / row["id"]
        task.mkdir(parents=True)
        visible = {
            "code": row["code"],
            "call": f"f({row['input']})",
            "instructions": INSTRUCTIONS,
        }
        (task / "input.json").write_bytes(_dump(visible))
        (task / "run").write_bytes(run)
        (task / "check").write_bytes(expected_check(row["output"]))
        (task / "task.toml").write_bytes(f'tags = ["{length_tag(row["code"])}"]\n'.encode())
    return split_pack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="CRUXEval cruxeval.jsonl")
    parser.add_argument("out", type=Path, help="pack directory to create")
    parser.add_argument("--sha256", help="refuse to convert unless the dataset has this sha256")
    args = parser.parse_args()
    report = convert(args.source, args.out, args.sha256)
    print(
        f"{report.n_tasks} tasks: " + ", ".join(f"{s.value}={n}" for s, n in report.counts.items())
    )
    for warning in report.warnings:
        print("warning:", warning)


if __name__ == "__main__":
    main()
