"""``iterant`` entry point: argument parsing, dispatch and error to exit-code mapping."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Sequence

from iterant.analyst import LLMError
from iterant.archive import ArchiveError, GitRefError
from iterant.cli.calibrate import cmd_aa_test, cmd_estimate
from iterant.cli.errors import (
    EXIT_BUDGET,
    EXIT_BUSY,
    EXIT_ERROR,
    EXIT_INFRA,
    EXIT_USAGE,
    CliError,
)
from iterant.cli.init import cmd_init
from iterant.cli.report import cmd_report
from iterant.cli.run import cmd_run
from iterant.cli.runtime import RuntimeFactory, build_runtime
from iterant.cli.settings import add_common_arguments, add_threshold_arguments
from iterant.cli.status import cmd_status
from iterant.contracts import Profile, Split
from iterant.gate import GateInfraError
from iterant.grader import GraderPreflightError
from iterant.orchestrator import BatchAborted, BudgetExhausted, OrchestratorBusy
from iterant.packs import PackError
from iterant.sandbox import SandboxError
from iterant.target import ConfigError, ImageBuildError

INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iterant",
        description="Improve your AI agent: propose changes, test them in sandboxes, keep only "
        "statistically proven wins.",
    )
    p.add_argument("--debug", action="store_true", help="show the traceback of unexpected errors")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    init = sub.add_parser("init", help="write a starter iterant.toml")
    init.add_argument("repo", nargs="?", default=".", help="target repo (default: current dir)")
    init.add_argument("--profile", choices=[x.value for x in Profile], default="free")
    init.add_argument("--genome", action="append", help="editable glob (repeatable)")
    init.add_argument("--image", default="Containerfile", help="Containerfile, relative to repo")
    init.add_argument("--run", default="python -m agent --task /task/input.json --out /out")
    init.add_argument("--pack", default="evals/pack", help="task pack directory, relative to repo")
    init.add_argument("--model", help="frozen target model (default depends on the profile)")
    init.add_argument(
        "--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"]
    )
    init.add_argument("--max-usd-per-run", help="per-run USD cap (claude profile)")
    init.add_argument("--max-tokens-per-run", type=int, help="per-run token cap (free profile)")
    init.add_argument("--force", action="store_true", help="overwrite an existing iterant.toml")
    init.set_defaults(handler="init")

    est = sub.add_parser("estimate", help="baseline pass rate, variance, detectable gain, $")
    add_common_arguments(est)
    est.add_argument("--tasks", type=int, default=20, help="tasks to sample (default 20)")
    est.add_argument("--reps", type=int, default=2, help="reps per task (default 2)")
    est.add_argument(
        "--split",
        choices=[Split.TRAIN.value, Split.SELECT.value],
        default="train",
        help="split to sample (holdout and test are never run for calibration)",
    )
    est.set_defaults(handler="estimate")

    aa = sub.add_parser("aa-test", help="A/A null test: champion vs champion through the rule")
    add_common_arguments(aa)
    aa.add_argument("--trials", type=int, default=200, help="resampled trials (default 200)")
    aa.add_argument("--resamples", type=int, default=2000, help="sign-flip resamples per test")
    aa.set_defaults(handler="aa-test")

    run = sub.add_parser("run", help="run generations of the improvement loop")
    add_common_arguments(run)
    run.add_argument("--generations", type=int, default=1, help="generations to run (default 1)")
    run.add_argument("--candidates", type=int, default=1, help="candidates per generation")
    run.add_argument("--digest-tasks", type=int, default=10, help="train tasks for the digest")
    run.add_argument(
        "--skip-calibration",
        action="store_true",
        help="run without a passing `iterant aa-test` for this pack (not recommended)",
    )
    run.set_defaults(handler="run")

    st = sub.add_parser("status", help="show the archive state")
    st.add_argument("repo", nargs="?", default=".")
    st.set_defaults(handler="status")

    rep = sub.add_parser("report", help="write the static HTML report")
    rep.add_argument("repo", nargs="?", default=".")
    rep.add_argument("--out", help="output directory (default <repo>/.iterant/report)")
    add_threshold_arguments(rep)
    rep.set_defaults(handler="report")
    return p


def _dispatch(args: argparse.Namespace, factory: RuntimeFactory) -> int:
    handlers: dict[str, Callable[[], int]] = {
        "init": lambda: cmd_init(args),
        "estimate": lambda: cmd_estimate(args, factory),
        "aa-test": lambda: cmd_aa_test(args, factory),
        "run": lambda: cmd_run(args, factory),
        "status": lambda: cmd_status(args),
        "report": lambda: cmd_report(args),
    }
    return handlers[args.handler]()


def _fail(message: str, code: int) -> int:
    print(f"iterant: error: {message}", file=sys.stderr)
    return code


def main(argv: Sequence[str] | None = None, *, factory: RuntimeFactory = build_runtime) -> int:
    """Run the CLI and return the exit code. ``factory`` wires the runtime (tests pass fakes)."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: 0 for --help, 2 for bad usage
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        return _dispatch(args, factory)
    except CliError as exc:
        return _fail(exc.message, exc.code)
    except OrchestratorBusy as exc:
        return _fail(
            f"{exc}. Another `iterant run` (or estimate/aa-test) holds this archive; wait for "
            "it to finish, or stop it, then rerun.",
            EXIT_BUSY,
        )
    except BudgetExhausted as exc:
        return _fail(
            f"budget exhausted ({exc}). Nothing was rejected for it; rerun with a larger "
            "--budget-usd to resume.",
            EXIT_BUDGET,
        )
    except BatchAborted as exc:
        return _fail(f"{exc}. The batch is intact or closed; rerun to resume.", EXIT_INFRA)
    except GraderPreflightError as exc:
        return _fail(f"grader preflight failed: {exc}", EXIT_INFRA)
    except (SandboxError, ImageBuildError, GateInfraError, LLMError) as exc:
        return _fail(f"{type(exc).__name__}: {exc}", EXIT_INFRA)
    except ConfigError as exc:
        return _fail(f"iterant.toml: {exc}", EXIT_USAGE)
    except (PackError, ArchiveError, GitRefError) as exc:
        return _fail(f"{type(exc).__name__}: {exc}", EXIT_ERROR)
    except KeyboardInterrupt:
        return _fail("interrupted; state is crash-safe, rerun to resume", INTERRUPTED)
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        return _fail(
            f"unexpected {type(exc).__name__}: {exc} (use --debug for a traceback)", EXIT_ERROR
        )


def entry() -> None:
    """Console-script entry point."""
    sys.exit(main())
