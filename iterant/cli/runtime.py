"""The objects a command runs on, and the factory that wires the real ones.

``Runtime`` is the seam tests use: ``main(argv, factory=...)`` takes any ``RuntimeFactory``, so
tests pass fakes (sandbox backend, proposer, gate, image source) and the same commands run
against them. ``build_runtime`` is the real wiring: archive, proxy and its HTTP server, the
container backend, grader, image builder, editing proposer, analyst client and diff gate.
"""

from __future__ import annotations

import ipaddress
import os
import sys
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from iterant.analyst import HttpMessagesClient, LLMError, MessagesClient
from iterant.archive import Archive
from iterant.cli.errors import EXIT_USAGE, CliError
from iterant.cli.images import ContainerImages
from iterant.cli.profiles import build_policy, role_models
from iterant.cli.settings import Project, Settings
from iterant.contracts import (
    Effort,
    Gate,
    Profile,
    Proposer,
    ProxyCall,
    Role,
    RunKey,
    SandboxBackend,
    Split,
)
from iterant.gate import DiffGate, ProxyAuditor
from iterant.grader import Grader
from iterant.orchestrator import GraderLike, ImageSource
from iterant.proposer import EditingProposer
from iterant.proxy import CallSink, Proxy, ProxyConfig, ProxyServer
from iterant.proxy.jsonutil import Obj
from iterant.sandbox import DockerBackend, PodmanBackend, slugify

ANTHROPIC_URL = "https://api.anthropic.com"
PROPOSER_MAX_TURNS = 30


@dataclass
class Runtime:
    project: Project
    archive: Archive
    proxy: Proxy
    backend: SandboxBackend
    grader: GraderLike
    images: ImageSource
    proposer: Proposer
    gate: Gate
    analyst_client: MessagesClient | None
    analyst_model: str
    analyst_effort: Effort
    _closers: list[Callable[[], None]] = field(default_factory=lambda: [])

    def on_close(self, fn: Callable[[], None]) -> None:
        self._closers.append(fn)

    def close(self) -> None:
        for fn in reversed(self._closers):
            try:
                fn()
            except Exception:  # keep closing the rest
                pass
        self.archive.close()


class RuntimeFactory(Protocol):
    def __call__(self, settings: Settings, project: Project, db_path: Path) -> Runtime:
        """Open the objects for ``project``, with the archive at ``db_path``."""
        ...


class _FreshTokenClient:
    """A ``MessagesClient`` that asks the proxy for a new single-use call token per request,
    so one client serves every call of a role (invariant 3: only tokens leave the proxy)."""

    def __init__(self, base_url: str, issue: Callable[[], str]) -> None:
        self._base = base_url
        self._issue = issue

    def create(self, body: Obj) -> Obj:
        return HttpMessagesClient(self._base, self._issue()).create(body)

    def create_many(self, bodies: Sequence[Obj]) -> list[Obj | None]:
        return [self._create_or_none(b) for b in bodies]

    def _create_or_none(self, body: Obj) -> Obj | None:
        try:
            return self.create(body)
        except LLMError:
            return None


def _archive_sink(db_path: Path) -> CallSink:
    """Every proxy call lands in the archive; ids are process-local, so the DB assigns them."""

    def sink(call: ProxyCall) -> None:
        with closing(Archive.open_path(db_path, create_schema=False)) as archive:
            archive.insert_proxy_call(replace(call, id=0))

    return sink


def is_loopback(host: str | None) -> bool:
    """True for ``localhost`` and loopback IP literals."""
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def openai_key_for(url: str, send_anyway: bool) -> str | None:
    """``OPENAI_API_KEY`` only for a loopback endpoint, or when the user opted in."""
    key = os.environ.get("OPENAI_API_KEY")
    if key and not (send_anyway or is_loopback(urlsplit(url).hostname)):
        print(
            "iterant: OPENAI_API_KEY is set but not sent: --openai-base-url is not a loopback "
            "address (pass --send-openai-key to send it there)",
            file=sys.stderr,
        )
        return None
    return key


def _proxy_config(settings: Settings, project: Project, db_path: Path) -> ProxyConfig:
    pack = project.pack
    splits = {t.id: t.split for t in pack.tasks}

    def split_of(key: RunKey) -> Split | None:
        return splits.get(key.task)

    policy = build_policy(
        settings.profile,
        project.config.target.model,
        settings.thresholds,
        settings.candidates,
        budget_usd_micros=settings.budget_usd_micros,
        extra_runs=settings.digest_tasks,
    )
    sink = _archive_sink(db_path)
    try:
        if settings.profile is Profile.CLAUDE:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise CliError(
                    "the claude profile spends real money and needs ANTHROPIC_API_KEY in the "
                    "environment; use --profile free for a local model",
                    EXIT_USAGE,
                )
            return ProxyConfig(
                policy=policy,
                sink=sink,
                prices=settings.prices,
                anthropic_base_url=ANTHROPIC_URL,
                anthropic_api_key=key,
                split_of=split_of,
            )
        return ProxyConfig(
            policy=policy,
            sink=sink,
            openai_base_url=settings.openai_base_url,
            openai_api_key=openai_key_for(settings.openai_base_url, settings.send_openai_key),
            split_of=split_of,
        )
    except ValueError as exc:
        hint = " (add it with --price MODEL=IN:OUT)" if "no price" in str(exc) else ""
        raise CliError(f"{exc}{hint}", EXIT_USAGE) from exc


def build_runtime(settings: Settings, project: Project, db_path: Path) -> Runtime:
    """The real wiring. Nothing here calls a model; calls happen when commands run."""
    config = project.config
    if not is_loopback(settings.proxy_bind):
        print(
            f"iterant: WARNING: --proxy-bind {settings.proxy_bind} is not a loopback address: "
            "the proxy accepts connections from other machines (it still needs a valid token)",
            file=sys.stderr,
        )
    proxy = Proxy(_proxy_config(settings, project, db_path))
    archive = Archive.open_path(db_path)
    server = ProxyServer(proxy, host=settings.proxy_bind).start()
    prefix = f"iterant-{slugify(project.repo.resolve().name) or 'repo'}"
    port = server.base_url.rsplit(":", 1)[1]
    try:
        if settings.backend == "docker":
            host = settings.proxy_host or "host.docker.internal"
            backend: SandboxBackend = DockerBackend(
                f"http://{host}:{port}/",
                unsafe_container=settings.unsafe_container,
                name_prefix=prefix,
            )
        else:
            host = settings.proxy_host or "host.containers.internal"
            backend = PodmanBackend(f"http://{host}:{port}/", name_prefix=prefix)
    except ValueError as exc:
        server.stop()
        archive.close()
        raise CliError(f"--proxy-host: {exc}", EXIT_USAGE) from exc
    models = role_models(settings.profile, config.target.model)
    base = server.base_url
    auditor = ProxyAuditor(
        base,
        lambda: proxy.issue_call_token(Role.AUDITOR),
        models.auditor[0],
        models.auditor[1],
    )
    rt = Runtime(
        project=project,
        archive=archive,
        proxy=proxy,
        backend=backend,
        grader=Grader(backend, project.pack, project.pack_root),
        images=ContainerImages(project.repo, config, engine=settings.backend, prefix=prefix),
        proposer=EditingProposer(
            _FreshTokenClient(base, lambda: proxy.issue_call_token(Role.PROPOSER)),
            config,
            model=models.proposer[0],
            effort=models.proposer[1],
            max_turns=PROPOSER_MAX_TURNS,
        ),
        gate=DiffGate(auditor, project.pack_root, protected=(config.evaluation.pack,)),
        analyst_client=_FreshTokenClient(base, lambda: proxy.issue_call_token(Role.ANALYST)),
        analyst_model=models.analyst[0],
        analyst_effort=models.analyst[1],
    )
    rt.on_close(server.stop)
    return rt
