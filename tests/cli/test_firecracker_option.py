"""``--backend firecracker``: selection, a clear failure without KVM or the pinned assets, and the
wiring (loopback proxy, image engine). The real backend is never built on a host without KVM:
tests use fakes and never boot a microVM."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from iterant.cli import runtime
from iterant.cli.errors import CliError
from iterant.cli.main import build_parser
from iterant.cli.runtime import build_runtime, default_proxy_bind, image_engine
from iterant.cli.settings import load_project, settings_from_args
from iterant.sandbox import SandboxError


class FakeFirecracker:
    """Stands in for ``FirecrackerBackend``: records how it was built."""

    built: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, proxy_url: str | None = None, **kwargs: Any) -> None:
        self.proxy_url = proxy_url
        FakeFirecracker.built.append((proxy_url or "", kwargs))


def _settings(world, *argv: str):  # type: ignore[no-untyped-def]
    args = build_parser().parse_args(["run", str(world.repo), "--backend", "firecracker", *argv])
    return settings_from_args(args)


def _build(world, settings):  # type: ignore[no-untyped-def]
    project = load_project(world.repo)
    return build_runtime(settings, project, world.repo / ".iterant" / "archive.db")


def test_backend_choices() -> None:
    parse = build_parser().parse_args
    assert parse(["run", ".", "--backend", "firecracker"]).backend == "firecracker"
    assert parse(["estimate", ".", "--backend", "firecracker"]).backend == "firecracker"
    assert parse(["run", "."]).backend == "podman"
    with pytest.raises(SystemExit):
        parse(["run", ".", "--backend", "qemu"])


def test_firecracker_needs_no_unsafe_flag(world) -> None:  # type: ignore[no-untyped-def]
    assert _settings(world).backend == "firecracker"  # only docker needs --unsafe-container


def test_default_bind_is_loopback() -> None:
    assert default_proxy_bind("firecracker") == "127.0.0.1"


def test_no_kvm_fails_clearly_before_anything_starts(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    world.cli("init", str(world.repo), "--genome", "agent/**")
    monkeypatch.setattr(runtime, "kvm_usable", lambda: False)
    monkeypatch.setattr(runtime, "FirecrackerBackend", FakeFirecracker)
    FakeFirecracker.built.clear()
    with pytest.raises(CliError, match=r"/dev/kvm.*--backend podman") as info:
        _build(world, _settings(world))
    assert info.value.code == 2
    assert FakeFirecracker.built == []


def _spy_cleanup(monkeypatch) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Count proxy-server stops and archive closes made by ``build_runtime``."""
    seen = {"stopped": 0, "closed": 0, "started": 0}
    real_server = runtime.ProxyServer

    class SpyServer(real_server):  # type: ignore[misc, valid-type]
        def start(self):  # type: ignore[no-untyped-def]
            seen["started"] += 1
            return super().start()

        def stop(self) -> None:
            seen["stopped"] += 1
            super().stop()

    real_open = runtime.Archive.open_path

    def spy_open(*a: Any, **k: Any):  # type: ignore[no-untyped-def]
        archive = real_open(*a, **k)
        close = archive.close

        def counted() -> None:
            seen["closed"] += 1
            close()

        archive.close = counted  # type: ignore[method-assign]
        return archive

    monkeypatch.setattr(runtime, "ProxyServer", SpyServer)
    monkeypatch.setattr(runtime.Archive, "open_path", staticmethod(spy_open))
    return seen


def _fake_host(world, monkeypatch, backend: type) -> None:  # type: ignore[no-untyped-def]
    world.cli("init", str(world.repo), "--genome", "agent/**")
    monkeypatch.setattr(runtime, "kvm_usable", lambda: True)
    monkeypatch.setattr(runtime, "image_engine", lambda _b: "podman")
    monkeypatch.setattr(runtime, "FirecrackerBackend", backend)


def test_missing_assets_fail_clearly_and_free_the_proxy(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class MissingAssets:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise SandboxError("firecracker not found (set ITERANT_FIRECRACKER_BIN)")

    _fake_host(world, monkeypatch, MissingAssets)
    seen = _spy_cleanup(monkeypatch)
    with pytest.raises(CliError, match=r"firecracker backend unavailable.*fetch-assets.sh") as info:
        _build(world, _settings(world))
    assert info.value.code == 2
    assert seen == {"started": 1, "stopped": 1, "closed": 1}


def test_constructor_value_error_is_not_blamed_on_proxy_host(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Bad:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise ValueError("name_prefix must match [a-z0-9][a-z0-9-]*")

    _fake_host(world, monkeypatch, Bad)
    seen = _spy_cleanup(monkeypatch)
    with pytest.raises(CliError, match="firecracker backend: name_prefix") as info:
        _build(world, _settings(world))
    assert "--proxy-host" not in info.value.message
    assert seen == {"started": 1, "stopped": 1, "closed": 1}


def test_any_exception_during_construction_frees_the_proxy(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Boom:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError("unexpected")

    _fake_host(world, monkeypatch, Boom)
    seen = _spy_cleanup(monkeypatch)
    with pytest.raises(RuntimeError, match="unexpected"):
        _build(world, _settings(world))
    assert seen == {"started": 1, "stopped": 1, "closed": 1}


def test_non_loopback_proxy_host_is_rejected_before_anything_starts(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _fake_host(world, monkeypatch, FakeFirecracker)
    seen = _spy_cleanup(monkeypatch)
    FakeFirecracker.built.clear()
    for host in ("10.0.0.5", "host.containers.internal"):
        with pytest.raises(CliError, match="loopback") as info:
            _build(world, _settings(world, "--proxy-host", host))
        assert info.value.code == 2
    assert seen["started"] == 0 and FakeFirecracker.built == []
    rt = _build(world, _settings(world, "--proxy-host", "localhost"))  # loopback is fine
    rt.close()


def test_real_backend_refuses_without_assets(world, monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """No fakes for the backend: with KVM claimed but no assets, the real constructor's refusal
    becomes the same clear message (so this works on any host)."""
    world.cli("init", str(world.repo), "--genome", "agent/**")
    monkeypatch.setattr(runtime, "kvm_usable", lambda: True)
    monkeypatch.setattr(runtime, "image_engine", lambda _b: "podman")
    monkeypatch.delenv("ITERANT_FIRECRACKER_BIN", raising=False)
    monkeypatch.delenv("ITERANT_FIRECRACKER_KERNEL", raising=False)
    with pytest.raises(CliError, match="firecracker backend unavailable"):
        _build(world, _settings(world))


def test_wiring_with_a_fake_backend(world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    world.cli("init", str(world.repo), "--genome", "agent/**", "--model", "tiny:1b")
    monkeypatch.setattr(runtime, "kvm_usable", lambda: True)
    monkeypatch.setattr(runtime, "FirecrackerBackend", FakeFirecracker)
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None
    )
    FakeFirecracker.built.clear()
    rt = _build(world, _settings(world))
    try:
        assert isinstance(rt.backend, FakeFirecracker)
        ((url, kwargs),) = FakeFirecracker.built
        # the host-side relay reaches the proxy on loopback
        assert url.startswith("http://127.0.0.1:") and url.endswith("/")
        assert kwargs["name_prefix"].startswith("iterant-")
        assert rt.images._engine == "podman"  # type: ignore[attr-defined]  # pyright: ignore
        assert rt.proxy.config.policy.roles
    finally:
        rt.close()


def test_image_engine(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert image_engine("podman") == "podman" and image_engine("docker") == "docker"
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/docker" if name == "docker" else None)
    assert image_engine("firecracker") == "docker"  # podman preferred, docker as the fallback
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(CliError, match="podman"):
        image_engine("firecracker")
