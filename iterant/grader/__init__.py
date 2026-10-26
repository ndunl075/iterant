"""The grader: the only source of scores (invariant 2) and the only place checks run."""

from iterant.grader.grader import (
    CHECK_PATH,
    OUT_DIR,
    Grader,
    GradeResult,
    interpret,
)

__all__ = ["CHECK_PATH", "OUT_DIR", "GradeResult", "Grader", "interpret"]
