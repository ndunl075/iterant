"""Shared fixtures for the red-team suite.

Real components, fakes only at the edges. The gate fixtures (a small target repo, a pack, a
stub auditor) and the orchestrator ``World`` (real archive, git repo, proxy ledger, grader and
pack on disk; fake sandbox backend and fake proposer) are reused from the module test suites
by import, so a red-team test attacks exactly what those suites build. ``HostileWorld`` adds
a hook that lets a test play a malicious candidate inside the agent sandbox, and a proxy that
records per-task markers so tests can see whose content it keeps.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import (
    ExecResult,
    NetMode,
    Profile,
    ProxyCall,
    ProxyPolicy,
    Role,
    RoleCap,
    RunKey,
    Sandbox,
    Split,
)
from iterant.proxy import (
    Admission,
    FakeLLM,
    ModelPrice,
    Proxy,
    ProxyConfig,
    ProxyServer,
    Rejection,
    Usage,
)
from iterant.sandbox import SandboxError
from tests.gate.conftest import (  # noqa: F401  (re-exported fixtures)
    FILES,
    auditor,
    edit,
    make_gate,
    pack,
    repo,
    run_gate,
    target,
)
from tests.orchestrator.conftest import FakeBackend, FakeSandbox, World


class HostileSandbox(FakeSandbox):
    """An agent sandbox whose agent run can be hijacked by ``world.after_agent``.

    The proxy call, the token handling and the scripted answer still happen exactly as in
    ``FakeSandbox``; the hook then edits the sandbox (plant files, break ``get``) and/or
    replaces the agent's ``ExecResult``.
    """

    get_broken = False
    listing_override: str | None = None  # what `find` on /out prints, set by the attack hook

    def _list(self) -> ExecResult:
        if self.listing_override is not None:
            return ExecResult(0, self.listing_override, "", False, 0.1)
        return super()._list()

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.__dict__.setdefault("cmds", []).append(cmd)  # every command, for key-custody scans
        return super().exec(cmd, timeout)

    def get(self, path: str) -> bytes:
        if self.get_broken:
            raise SandboxError("get failed")
        return super().get(path)

    def _agent(self, cmd: str, timeout: float) -> ExecResult:
        result = super()._agent(cmd, timeout)
        hook = self.world.after_agent  # type: ignore[attr-defined]
        if hook is None:
            return result
        replaced = hook(self, self.world.exec_log[-1], result)
        return replaced if replaced is not None else result


class HostileBackend(FakeBackend):
    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        if self.world.create_fails:
            raise SandboxError("cannot create")
        self.ttls.append(ttl)
        sb = HostileSandbox(self.world, image, net)
        self.sandboxes.append(sb)
        return sb


class MarkingProxy(Proxy):
    """The real proxy, except that every run-token request carries a per-task marker (what a
    task prompt and a model reply would hold), so a test can see which splits' content the
    proxy keeps. ``scale[cand]`` multiplies the usage the "upstream" reports for that arm."""

    scale: dict[str, int] = {}

    def admit(
        self, token: str | None, body_bytes: bytes, beta_header: str = ""
    ) -> Admission | Rejection:
        session = self.sessions.resolve(token)
        if session is not None and session.key is not None:
            body = json.loads(body_bytes)
            body["messages"] = [{"role": "user", "content": f"PROMPT-{session.key.task}"}]
            body_bytes = json.dumps(body).encode()
        return super().admit(token, body_bytes, beta_header)

    def settle(self, adm: Admission, usage: Usage | None, refusal: bool, response: Any) -> Any:
        key = adm.session.key
        if key is not None and usage is not None and key.cand in self.scale:
            n = self.scale[key.cand]
            usage = replace(usage, input=usage.input * n, output=usage.output * n)
        if key is not None and response is None:
            response = {
                "model": adm.cap.model,
                "content": [{"type": "text", "text": f"REPLY-{key.task}"}],
            }
        return super().settle(adm, usage, refusal, response)


AfterAgent = Callable[[HostileSandbox, Any, ExecResult], ExecResult | None]


class HostileWorld(World):
    def __init__(self, tmp: Path) -> None:
        self.after_agent: AfterAgent | None = None
        self.usage_scale: dict[str, int] = {}
        super().__init__(tmp)
        self.backend = HostileBackend(self)

    def _split_of(self, key: RunKey) -> Split | None:
        return next((t.split for t in self.pack.tasks if t.id == key.task), None)

    def _new_proxy(self) -> Proxy:
        base = super()._new_proxy()
        proxy = MarkingProxy(replace(base.config, split_of=self._split_of))
        proxy.scale = self.usage_scale
        return proxy


@pytest.fixture
def world(tmp_path: Path) -> Iterator[HostileWorld]:
    w = HostileWorld(tmp_path)
    yield w
    w.close()


# --- a standalone proxy stack (real Proxy + real HTTP server + recorded-fixture fake upstream) ---

HAIKU = "claude-haiku-5-5"
OPUS = "claude-opus-5-5"
REAL_KEY = "sk-ant-REAL-SECRET-KEY-0123456789"
RUN_CAP = 100_000  # micro-USD per target run
GEN_CAP = 300_000
KEY = RunKey(batch=1, cand="c1", task="t1", rep=0)


@dataclass
class Stack:
    proxy: Proxy
    server: ProxyServer
    fake: FakeLLM
    calls: list[ProxyCall]

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


def msg_body(model: Any = HAIKU, effort: str | None = "medium", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }
    if effort is not None:
        body["output_config"] = {"effort": effort}
    return body


def make_proxy(
    upstream: str = "http://upstream.invalid", **cfg: Any
) -> tuple[Proxy, list[ProxyCall]]:
    calls: list[ProxyCall] = []
    policy = ProxyPolicy(
        Profile.CLAUDE,
        {
            Role.TARGET: RoleCap(
                HAIKU,
                "medium",
                max_usd_micros_per_run=RUN_CAP,
                max_usd_micros_per_gen=GEN_CAP,
            ),
            Role.PROPOSER: RoleCap(
                OPUS, "high", max_usd_micros_per_run=2_000_000, max_usd_micros_per_gen=3_000_000
            ),
        },
    )
    proxy = Proxy(
        ProxyConfig(
            policy=policy,
            sink=lambda call: calls.append(call),
            prices={
                HAIKU: ModelPrice(1_000_000, 5_000_000),
                OPUS: ModelPrice(5_000_000, 25_000_000),
            },
            anthropic_base_url=upstream,
            anthropic_api_key=REAL_KEY,
            trust_env_proxy=False,
            **cfg,
        )
    )
    return proxy, calls


@pytest.fixture
def proxy() -> Proxy:
    return make_proxy()[0]


@pytest.fixture
def stack() -> Iterator[Stack]:
    fake = FakeLLM().start()
    proxy, calls = make_proxy(upstream=fake.base_url)
    server = ProxyServer(proxy).start()
    try:
        yield Stack(proxy, server, fake, calls)
    finally:
        server.stop()
        fake.stop()
