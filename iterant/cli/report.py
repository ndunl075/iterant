"""``iterant report``: a thin call into ``iterant.report`` (the static HTML report).

PENDING W2.11: ``iterant/report/`` is built by another worker. This code is written against the
minimal signature ``render(archive_path, out_dir) -> Path`` (the path of the page it wrote) and
imports it lazily, so the CLI works, and says so clearly, until that module lands.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

from iterant.archive import archive_path
from iterant.cli.errors import EXIT_ERROR, EXIT_OK, CliError

Renderer = Callable[[Path, Path], Path]


def load_renderer() -> Renderer:
    try:
        module = importlib.import_module("iterant.report")
    except ImportError:
        raise CliError(
            "the report module is not available in this build (iterant/report is pending W2.11)",
            EXIT_ERROR,
        ) from None
    render = getattr(module, "render", None)
    if not callable(render):
        raise CliError("iterant.report has no render(archive_path, out_dir) function", EXIT_ERROR)
    return cast(Renderer, render)


def cmd_report(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    db = archive_path(repo)
    if not db.is_file():
        raise CliError(f"no archive at {db}: nothing to report yet (try `iterant run`)", EXIT_ERROR)
    out_dir = Path(args.out) if args.out else repo / ".iterant" / "report"
    page = load_renderer()(db, out_dir)
    print(f"report written to {page}")
    return EXIT_OK
