"""``iterant report``: a thin call into ``iterant.report.render`` (the static HTML report).

The archive stores pack hashes but not pack names or policy, so the loaded pack and the
acceptance thresholds are always passed: they put the pack name and the "NOT enforced" banner
(for ``advisory-forbidden`` packs) on the page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iterant.archive import archive_path
from iterant.cli.calibrate import recorded_thresholds
from iterant.cli.errors import EXIT_ERROR, EXIT_OK, CliError
from iterant.cli.settings import (
    OVERRIDE_FLAGS,
    load_project,
    preset_from_args,
    thresholds_from_args,
)
from iterant.contracts import AcceptanceThresholds
from iterant.report import render


def report_thresholds(args: argparse.Namespace, repo: Path) -> tuple[AcceptanceThresholds, str]:
    """Thresholds for the page and how to describe where they came from. The archive does not
    store the run's thresholds, so: explicit flags (``--preset`` or any override) win; else the
    thresholds of the last calibration record (what ``estimate`` and ``aa-test`` ran under);
    else the balanced defaults."""
    explicit = getattr(args, "preset", None) is not None or any(
        getattr(args, name, None) is not None for name in OVERRIDE_FLAGS
    )
    if explicit:
        return thresholds_from_args(args), f"the {preset_from_args(args)} preset with your flags"
    recorded = recorded_thresholds(repo)
    if recorded is not None:
        return recorded, "the thresholds of the last calibration record (.iterant/calibration.json)"
    return thresholds_from_args(args), "the default thresholds (balanced preset)"


def cmd_report(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    db = archive_path(repo)
    if not db.is_file():
        raise CliError(f"no archive at {db}: nothing to report yet (try `iterant run`)", EXIT_ERROR)
    project = load_project(repo)
    thresholds, source = report_thresholds(args, repo)
    out_dir = Path(args.out) if args.out else repo / ".iterant" / "report"
    page = render(
        db,
        out_dir,
        pack=project.pack,
        thresholds=thresholds,
        thresholds_source=source,
        diffs_dir=repo / ".iterant" / "diffs",
    )
    print(f"report written to {page}")
    return EXIT_OK
