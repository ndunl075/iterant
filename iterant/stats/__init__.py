"""Paired sign-flip test, acceptance rule, power and estimate math (§4). Stdlib only."""

from iterant.stats.acceptance import Decision, Reason, accepted, decide_confirm, decide_select
from iterant.stats.estimate import Estimate, estimate
from iterant.stats.pairing import Pairing, pair_runs
from iterant.stats.power import power, smallest_detectable_gain, tasks_needed
from iterant.stats.signflip import DEFAULT_RESAMPLES, PairedTestResult, paired_signflip

__all__ = [
    "DEFAULT_RESAMPLES",
    "PairedTestResult",
    "Decision",
    "Estimate",
    "Pairing",
    "Reason",
    "accepted",
    "decide_confirm",
    "decide_select",
    "estimate",
    "pair_runs",
    "paired_signflip",
    "power",
    "smallest_detectable_gain",
    "tasks_needed",
]
