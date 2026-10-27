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


def test_missing_assets_fail_clearly_and_free_the_proxy(world, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    world.cli("init", str(world.repo), "--genome", "agent/**")
    monkeypatch.setattr(runtime, "kvm_usable", lambda: True)
    monkeypatch.setattr(runtime, "image_engine", lambda _b: "podman")
    for var in ("ITERANT_FIRECRACKER_BIN", "ITERANT_FIRECRACKER_KERNEL"):
        monkeypatch.delenv(var, raising=False)

    class MissingAssets:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise SandboxError("firecracker not found (set ITERANT_FIRECRACKER_BIN)")

    monkeypatch.setattr(runtime, "FirecrackerBackend", MissingAssets)
    with pytest.raises(CliError, match=r"firecracker backend unavailable.*fetch-assets.sh") as info:
        _build(world, _settings(world))
    assert info.value.code == 2


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
