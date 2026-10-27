"""Acceptance presets (§4). The one place the preset values live."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from iterant.contracts import AcceptanceThresholds

DEFAULT_PRESET = "balanced"

# balanced: today's sizes with looser p-values (the A/A false accept stays <= 5%, and the pack
# needs far fewer tasks to accept a real gain). strict: the original p-values, and a larger
# CONFIRM to pay for them.
PRESETS: dict[str, AcceptanceThresholds] = {
    "balanced": AcceptanceThresholds(select_p=0.10, holdout_p=0.5),
    "strict": AcceptanceThresholds(select_p=0.05, holdout_p=0.2, holdout_tasks=80),
}


def preset_thresholds(name: str, **overrides: Any) -> AcceptanceThresholds:
    """The preset's thresholds with explicit values replacing the preset's.

    Raises ``KeyError`` for an unknown preset and ``ValueError`` for an invalid result."""
    return replace(PRESETS[name], **overrides)
