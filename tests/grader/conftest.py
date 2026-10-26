"""Grader tests build fake packs whose hash does not match their files; skip the constructor's
hash check for them. Tests with ``real_verify`` in their name keep the real check."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from iterant.grader import grader as grader_module


@pytest.fixture(autouse=True)
def _skip_pack_verify(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if "real_verify" not in request.node.name:
        monkeypatch.setattr(grader_module, "verify", lambda *_a, **_k: None)
    yield
