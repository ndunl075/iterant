"""Session tokens: what a sandbox holds instead of the real key (invariant 3).

A run role (target, transfer) gets one token per ``RunKey``. A non-run role (proposer,
analyst, auditor) gets a token good for ``uses`` calls (default 1). Unknown, expired,
revoked and used-up tokens all resolve to None.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass

from iterant.contracts import Role, RunKey
from iterant.proxy.config import CALL_ROLES, RUN_ROLES

TOKEN_PREFIX = "itr_"


@dataclass(frozen=True)
class Session:
    role: Role
    key: RunKey | None


class _Entry:
    def __init__(self, session: Session, expires_at: float, uses_left: int | None) -> None:
        self.session = session
        self.expires_at = expires_at
        self.uses_left = uses_left  # None: unlimited until expiry


class SessionRegistry:
    def __init__(self, monotonic: Callable[[], float], default_ttl_s: float) -> None:
        self._now = monotonic
        self._ttl = default_ttl_s
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}

    def issue_run_token(self, role: Role, key: RunKey, ttl_s: float | None = None) -> str:
        if role not in RUN_ROLES:
            raise ValueError(f"role {role} does not run per-RunKey; use issue_call_token")
        return self._add(Session(role, key), ttl_s, None)

    def issue_call_token(self, role: Role, uses: int = 1, ttl_s: float | None = None) -> str:
        if role not in CALL_ROLES:
            raise ValueError(f"role {role} runs per-RunKey; use issue_run_token")
        if uses < 1:
            raise ValueError("uses must be >= 1")
        return self._add(Session(role, None), ttl_s, uses)

    def _add(self, session: Session, ttl_s: float | None, uses: int | None) -> str:
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        ttl = self._ttl if ttl_s is None else ttl_s
        with self._lock:
            self._entries[token] = _Entry(session, self._now() + ttl, uses)
        return token

    def resolve(self, token: str | None) -> Session | None:
        """Look a token up without using it up."""
        if not token:
            return None
        with self._lock:
            entry = self._live(token)
            return entry.session if entry else None

    def take_use(self, token: str) -> bool:
        """Spend one use. False if the token died since ``resolve``."""
        with self._lock:
            entry = self._live(token)
            if entry is None:
                return False
            if entry.uses_left is not None:
                entry.uses_left -= 1
                if entry.uses_left <= 0:
                    del self._entries[token]
            return True

    def revoke(self, token: str) -> None:
        with self._lock:
            self._entries.pop(token, None)

    def revoke_run(self, key: RunKey) -> None:
        with self._lock:
            for t in [t for t, e in self._entries.items() if e.session.key == key]:
                del self._entries[t]

    def _live(self, token: str) -> _Entry | None:
        entry = self._entries.get(token)
        if entry is None:
            return None
        if self._now() >= entry.expires_at:
            del self._entries[token]
            return None
        return entry
