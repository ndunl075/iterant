"""The grader: the only source of scores (invariant 2) and the only place checks run."""

from iterant.grader.grader import (
    OUT_DIR,
    Grader,
    GradeResult,
    interpret,
)
from iterant.grader.outfiles import OutputRejected, collect_out_files

__all__ = [
    "OUT_DIR",
    "GradeResult",
    "Grader",
    "OutputRejected",
    "collect_out_files",
    "interpret",
]
