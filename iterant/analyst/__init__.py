"""Failure analyst: train-split trajectories and run results to a compact failure digest (§3, §5).

Only ``train`` data goes in (invariant 4): runs of any other split, and call content that does
not belong to a train run, are dropped. ``client`` holds the Messages client seam shared with
the proposer.
"""

from iterant.analyst.client import HttpMessagesClient, LLMError, MessagesClient
from iterant.analyst.digest import FailureAnalyst

__all__ = ["FailureAnalyst", "HttpMessagesClient", "LLMError", "MessagesClient"]
