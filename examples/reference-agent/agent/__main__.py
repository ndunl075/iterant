"""Entry point: ``python -m agent --task /task/input.json --out /out``."""

import argparse
import json
import sys
from pathlib import Path

from agent.loop import make_client, run_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent")
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    task = json.loads(args.task.read_text(encoding="utf-8"))
    result = run_task(task, make_client())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "answer.json").write_text(json.dumps({"answer": result.answer}), encoding="utf-8")
    (args.out / "transcript.json").write_text(json.dumps(result.transcript), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
