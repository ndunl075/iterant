"""Shared fixtures: a fake upstream, a configured proxy and a server on port 0."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from iterant.contracts import Profile, ProxyPolicy, Role, RoleCap, RunKey, Split
from iterant.proxy import (
    FakeLLM,
    InMemorySink,
    ModelPrice,
    Proxy,
    ProxyConfig,
    ProxyServer,
)

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
LOCAL = "llama-test"
REAL_KEY = "sk-ant-REAL-SECRET-KEY-0123456789"

PRICES = {
    HAIKU: ModelPrice(1_000_000, 5_000_000),  # $1 / $5 per Mtok
    OPUS: ModelPrice(5_000_000, 25_000_000),
}

TARGET_RUN_CAP = 100_000  # $0.10
KEY = RunKey(batch=1, cand="c1", task="t1", rep=0)


def claude_policy() -> ProxyPolicy:
    return ProxyPolicy(
        Profile.CLAUDE,
        {
            Role.TARGET: RoleCap(
                HAIKU,
                "medium",
                max_usd_micros_per_run=TARGET_RUN_CAP,
                max_usd_micros_per_gen=300_000,
            ),
            Role.PROPOSER: RoleCap(
                OPUS, "high", max_usd_micros_per_run=2_000_000, max_usd_micros_per_gen=3_000_000
            ),
        },
    )


def free_policy() -> ProxyPolicy:
    return ProxyPolicy(
        Profile.FREE,
        {
            Role.TARGET: RoleCap(
                LOCAL, "medium", max_tokens_per_run=1_000, max_tokens_per_gen=2_500
            ),
            Role.PROPOSER: RoleCap(
                LOCAL, "medium", max_tokens_per_run=5_000, max_tokens_per_gen=5_000
            ),
        },
    )


@dataclass
class Stack:
    proxy: Proxy
    server: ProxyServer
    fake: FakeLLM
    sink: InMemorySink

    def post(
        self,
        token: str | None,
        body: dict[str, Any],
        path: str = "/v1/messages",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, bytes]:
        hdrs = {"content-type": "application/json", **(headers or {})}
        if token is not None:
            hdrs["x-api-key"] = token
        req = urllib.request.Request(
            self.server.base_url + path, data=json.dumps(body).encode(), headers=hdrs
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()


def msg_body(model: str = HAIKU, effort: str | None = "medium", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    if effort is not None:
        body["output_config"] = {"effort": effort}
    return body


MakeStack = Callable[..., Stack]


@pytest.fixture
def make_stack() -> Iterator[MakeStack]:
    stacks: list[Stack] = []

    def make(
        profile: Profile = Profile.CLAUDE,
        splits: dict[str, Split] | None = None,
        sink: Any = None,
    ) -> Stack:
        fake = FakeLLM().start()
        the_sink = InMemorySink()
        cfg_sink = sink if sink is not None else the_sink
        split_map = splits or {}
        common: dict[str, Any] = {
            "sink": cfg_sink,
            "split_of": lambda k: split_map.get(k.task),
            "trust_env_proxy": False,
        }
        if profile is Profile.CLAUDE:
            cfg = ProxyConfig(
                policy=claude_policy(),
                prices=PRICES,
                anthropic_base_url=fake.base_url,
                anthropic_api_key=REAL_KEY,
                **common,
            )
        else:
            cfg = ProxyConfig(policy=free_policy(), openai_base_url=fake.openai_base_url, **common)
        proxy = Proxy(cfg)
        server = ProxyServer(proxy).start()
        stack = Stack(proxy, server, fake, the_sink)
        stacks.append(stack)
        return stack

    yield make
    for s in stacks:
        s.server.stop()
        s.fake.stop()


@dataclass(frozen=True)
class Helpers:
    """Constants and builders for tests (importlib mode cannot import conftest directly)."""

    HAIKU: str = HAIKU
    OPUS: str = OPUS
    LOCAL: str = LOCAL
    REAL_KEY: str = REAL_KEY
    KEY: RunKey = KEY
    TARGET_RUN_CAP: int = TARGET_RUN_CAP

    msg_body = staticmethod(msg_body)


@pytest.fixture
def h() -> Helpers:
    return Helpers()
