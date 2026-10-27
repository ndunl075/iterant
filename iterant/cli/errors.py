"""Exit codes and the one error type the CLI turns into a message (never a traceback)."""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1  # anything that went wrong with the repo, the pack, the archive or git
EXIT_USAGE = 2  # bad arguments or a bad iterant.toml (argparse uses 2 as well)
EXIT_BUSY = 3  # another orchestrator holds the archive
EXIT_BUDGET = 4  # the $ budget ran out; rerun with more budget to resume
EXIT_INFRA = 5  # sandbox runtime, image build, grader preflight or LLM endpoint failed
EXIT_CALIBRATION = 6  # the A/A test found too many false accepts


class CliError(Exception):
    """A failure with a user-facing message and the exit code to end the process with."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
