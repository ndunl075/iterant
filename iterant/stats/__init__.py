"""Paired bootstrap, acceptance rule, power and estimate math (§4). Stdlib only."""

from iterant.stats.acceptance import Decision, Reason, accepted, decide_confirm, decide_select
from iterant.stats.bootstrap import DEFAULT_RESAMPLES, BootstrapResult, paired_bootstrap
from iterant.stats.estimate import Estimate, estimate
from iterant.stats.pairing import Pairing, pair_runs
from iterant.stats.power import power, smallest_detectable_gain, tasks_needed

__all__ = [
    "DEFAULT_RESAMPLES",
    "BootstrapResult",
    "Decision",
    "Estimate",
    "Pairing",
    "Reason",
    "accepted",
    "decide_confirm",
    "decide_select",
    "estimate",
    "pair_runs",
    "paired_bootstrap",
    "power",
    "smallest_detectable_gain",
    "tasks_needed",
]
