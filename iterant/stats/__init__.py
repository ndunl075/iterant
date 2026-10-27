"""Paired sign-flip test, acceptance rule, power and estimate math (§4). Stdlib only."""

from iterant.stats.acceptance import Decision, Reason, accepted, decide_confirm, decide_select
from iterant.stats.estimate import Estimate, estimate
from iterant.stats.pairing import Pairing, pair_runs
from iterant.stats.power import (
    power,
    rule_power,
    rule_reps_needed,
    rule_tasks_needed,
    smallest_detectable_gain,
    tasks_needed,
)
from iterant.stats.presets import DEFAULT_PRESET, PRESETS, preset_thresholds
from iterant.stats.signflip import DEFAULT_RESAMPLES, PairedTestResult, paired_signflip

__all__ = [
    "DEFAULT_PRESET",
    "DEFAULT_RESAMPLES",
    "PRESETS",
    "preset_thresholds",
    "rule_power",
    "rule_reps_needed",
    "rule_tasks_needed",
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
