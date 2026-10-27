"""Messages API client interface for Iterant's own calls (proposer, analyst).

``MessagesClient`` is the seam: ``create`` sends one request body and returns the response
object; ``create_many`` sends several and returns one entry per body (None where a call
failed). The analyst's volume calls go through ``create_many`` so a Batch API backed client
(-50%, §6) can replace ``HttpMessagesClient`` without touching the callers. Tests use
``HttpMessagesClient`` against ``FakeLLM``; nothing here reaches a real model.

Stdlib only: the runtime has no dependencies. The client talks to the proxy with a
per-call token, never a real key (invariant 3).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from iterant.proxy.jsonutil import Obj, arr, obj, text


class LLMError(RuntimeError):
    """A model call failed (HTTP error, bad JSON). ``str(error)`` is the reason."""


class MessagesClient(Protocol):
    def create(self, body: Obj) -> Obj:
        """POST one Messages request. Raises ``LLMError`` on failure."""
        ...

    def create_many(self, bodies: Sequence[Obj]) -> list[Obj | None]:
        """One response per body, in order; None for a call that failed."""
        ...


class HttpMessagesClient:
    """Synchronous Messages API client over ``urllib`` (``POST {base_url}/v1/messages``)."""

    def __init__(self, base_url: str, token: str, *, timeout: float = 300.0) -> None:
        self._url = base_url.rstrip("/") + "/v1/messages"
        self._token = token
        self._timeout = timeout

    def create(self, body: Obj) -> Obj:
        req = urllib.request.Request(
            self._url,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._token,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                parsed = obj(json.loads(resp.read()))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"model call failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"model call failed: {exc}") from exc
        if parsed is None:
            raise LLMError("model call failed: response is not a JSON object")
        return parsed

    def create_many(self, bodies: Sequence[Obj]) -> list[Obj | None]:
        out: list[Obj | None] = []
        for body in bodies:
            try:
                out.append(self.create(body))
            except LLMError:
                out.append(None)
        return out


def reply_text(response: Obj) -> str:
    """Concatenated text blocks of a Messages response."""
    parts: list[str] = []
    for block in arr(response.get("content")) or []:
        b = obj(block)
        if b is not None and b.get("type") == "text":
            parts.append(text(b.get("text")) or "")
    return "".join(parts).strip()
