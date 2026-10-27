"""``iterant report``: a thin call into ``iterant.report.render`` (the static HTML report).

The archive stores pack hashes but not pack names or policy, so the loaded pack and the
acceptance thresholds are always passed: they put the pack name and the "NOT enforced" banner
(for ``advisory-forbidden`` packs) on the page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iterant.archive import archive_path
from iterant.cli.errors import EXIT_ERROR, EXIT_OK, CliError
from iterant.cli.settings import load_project, thresholds_from_args
from iterant.report import render


def cmd_report(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    db = archive_path(repo)
    if not db.is_file():
        raise CliError(f"no archive at {db}: nothing to report yet (try `iterant run`)", EXIT_ERROR)
    project = load_project(repo)
    out_dir = Path(args.out) if args.out else repo / ".iterant" / "report"
    page = render(
        db,
        out_dir,
        pack=project.pack,
        thresholds=thresholds_from_args(args),
        diffs_dir=repo / ".iterant" / "diffs",
    )
    print(f"report written to {page}")
    return EXIT_OK
