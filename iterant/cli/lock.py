"""The CLI-level lock: held across a whole ``run`` loop, and by estimate and aa-test.

The orchestrator takes its own ``WriterLock`` per generation, so the CLI cannot hold that one
(a second lock on the same file from the same process fails). This lock is a separate file
(``.iterant/cli/.iterant/orchestrator.lock``: a ``WriterLock`` over the directory
``<repo>/.iterant/cli``), so nothing else can start between two generations. ``status`` probes
it briefly, so acquiring retries a few times instead of failing on that probe.
"""

from __future__ import annotations

import time
from pathlib import Path

from iterant.orchestrator import OrchestratorBusy, WriterLock

_ATTEMPTS = 5
_DELAY_S = 0.2


def cli_lock(repo: Path) -> WriterLock:
    return WriterLock(repo / ".iterant" / "cli")


def acquire_retrying(lock: WriterLock) -> None:
    for attempt in range(_ATTEMPTS):
        try:
            lock.acquire()
            return
        except OrchestratorBusy:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(_DELAY_S)
