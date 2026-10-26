"""Task packs: format, content hash and split assignment (§2, invariant 7)."""

from iterant.packs.hashing import PackError, PackHashMismatch, compute_content_hash, verify
from iterant.packs.loader import load_pack, split_pack, tasks_in
from iterant.packs.splitter import SplitPlan, SplitReport, assign_splits

__all__ = [
    "PackError",
    "PackHashMismatch",
    "SplitPlan",
    "SplitReport",
    "assign_splits",
    "compute_content_hash",
    "load_pack",
    "split_pack",
    "tasks_in",
    "verify",
]
