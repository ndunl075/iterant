"""Firecracker microVM backend (§5): the default sandbox at release. Needs ``/dev/kvm``.

Each sandbox is one microVM booted from a pinned guest kernel:

* root: the image's file system as a shared read-only ext4 (see ``rootfs``); no host mounts.
* ``/work``: a private ext4 disk of ``disk`` MiB (the only persistent writable path), plus a
  64 MiB ``/tmp`` and a 1 MiB ``/dev/shm`` in guest RAM.
* cpu/memory/pids caps: a guest cgroup-v2 leaf holding every command; the VM itself gets
  ``ceil(cpu)`` vCPUs and ``mem`` MiB plus a fixed overhead for the guest kernel and agent.
* network: no network device at all (default-deny is structural). For ``proxy-only`` the guest
  reaches the host over vsock, where the single allowed route is piped to the proxy.
* control: put/get/exec travel over vsock to an agent in the guest (``guest.py``).
* ttl: a host timer destroys the VM, and a ``timeout`` wrapper kills it if the host dies.

Runs unprivileged: the only host needs are ``/dev/kvm`` access, the pinned ``firecracker``
binary, ``mke2fs`` and ``podman`` or ``docker`` to export the image.
"""

from __future__ import annotations

import base64
import json
import math
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from iterant.contracts import NetMode, Sandbox
from iterant.sandbox.container import _NAME_RE, SandboxError, _parse_proxy  # pyright: ignore[reportPrivateUsage]
from iterant.sandbox.firecracker import rootfs
from iterant.sandbox.firecracker.sandbox import (
    PROXY_VSOCK_PORT,
    Conn,
    FirecrackerSandbox,
    ProxyRelay,
    connect_vsock_uds,
    kill_process_group,
    remove_tree,
)

BIN_ENV = "ITERANT_FIRECRACKER_BIN"
KERNEL_ENV = "ITERANT_FIRECRACKER_KERNEL"
CACHE_ENV = "ITERANT_FIRECRACKER_CACHE"
DEBUG_ENV = "ITERANT_FIRECRACKER_DEBUG"
GUEST_OVERHEAD_MIB = 192  # guest kernel, init, agent and the 64 MiB /tmp on top of ``mem``
TMP_MIB = 64
PIDS_LIMIT = 512
PROXY_LOCAL_PORT = 8080
BOOT_TIMEOUT_S = 60.0
GUEST_CID = 3
SANDBOX_GRACE_S = 15  # the VM's own timeout outlives the host-side ttl by this much
_LOG_TAIL = 4000


def kvm_usable() -> bool:
    return os.path.exists("/dev/kvm") and os.access("/dev/kvm", os.R_OK | os.W_OK)


def _default_cache() -> Path:
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "iterant" / "firecracker"


class FirecrackerBackend:
    """Creates Firecracker microVM sandboxes. ``proxy_url`` is the proxy as seen from the host."""

    def __init__(
        self,
        # TODO(contract): SandboxBackend.create has no proxy address, so it is a constructor arg.
        proxy_url: str | None = None,
        *,
        name_prefix: str = "iterant",
        firecracker_bin: str | None = None,
        kernel: str | None = None,
        cache_dir: Path | None = None,
        container_runtime: str | None = None,
        sandbox_root: str | None = None,
        boot_timeout: float = BOOT_TIMEOUT_S,
    ) -> None:
        if not _NAME_RE.fullmatch(name_prefix):
            raise ValueError("name_prefix must match [a-z0-9][a-z0-9-]* (use slugify())")
        self._prefix = name_prefix
        self._proxy = _parse_proxy(proxy_url) if proxy_url is not None else None
        self._fc = firecracker_bin or os.environ.get(BIN_ENV) or shutil.which("firecracker")
        self._kernel = kernel or os.environ.get(KERNEL_ENV)
        self._cache = cache_dir or _default_cache()
        self._runtime = container_runtime
        # AF_UNIX paths are limited to ~100 bytes, so sandbox dirs live under a short root.
        self._root = sandbox_root or "/tmp"
        self._boot_timeout = boot_timeout

    def _container_runtime(self) -> str:
        found = self._runtime or shutil.which("podman") or shutil.which("docker")
        if not found:
            raise SandboxError("podman or docker is needed to export the image's file system")
        return found

    def _require_host(self) -> tuple[str, str]:
        if not self._fc or not os.access(self._fc, os.X_OK):
            raise SandboxError(f"firecracker binary not found (set {BIN_ENV})")
        if not self._kernel or not os.path.isfile(self._kernel):
            raise SandboxError(f"guest kernel not found (set {KERNEL_ENV})")
        if not kvm_usable():
            raise SandboxError("/dev/kvm is missing or not accessible to this user")
        return self._fc, self._kernel

    def create(
        self,
        image: str,
        cpu: float,
        mem: int,
        disk: int,
        ttl: float,
        net: NetMode = "proxy-only",
    ) -> Sandbox:
        if not image or image.startswith("-"):
            raise ValueError(f"bad image name {image!r}")
        if cpu <= 0 or mem <= 0 or disk <= 0 or ttl <= 0:
            raise ValueError("cpu, mem, disk and ttl must be > 0")
        if net not in ("proxy-only", "default-deny"):
            raise ValueError(f"unknown net mode {net!r}")
        if net == "proxy-only" and self._proxy is None:
            raise SandboxError("net='proxy-only' needs the backend built with proxy_url=...")
        fc, kernel = self._require_host()
        base_img = rootfs.build_rootfs(self._container_runtime(), image, self._cache, self._prefix)

        name = f"{self._prefix}-{secrets.token_hex(4)}"
        sbx_dir = Path(self._root) / name
        sbx_dir.mkdir(mode=0o700)
        relay: ProxyRelay | None = None
        proc: subprocess.Popen[bytes] | None = None
        try:
            rootfs.make_ext4(
                sbx_dir / "work.ext4", max(disk, 4), label="iterant-work",
                owner=f"{rootfs.SANDBOX_UID}:{rootfs.SANDBOX_UID}",
            )  # fmt: skip
            uds = str(sbx_dir / "v.sock")
            proxy_url: str | None = None
            if net == "proxy-only":
                assert self._proxy is not None
                host, port, path = self._proxy
                relay = ProxyRelay(f"{uds}_{PROXY_VSOCK_PORT}", host, port)
                proxy_url = f"http://127.0.0.1:{PROXY_LOCAL_PORT}{path}"
            cfg = {
                "cpu": cpu, "mem": mem, "pids": PIDS_LIMIT, "tmp_mib": TMP_MIB,
                "proxy_url": proxy_url,
            }  # fmt: skip
            debug = bool(os.environ.get(DEBUG_ENV))
            boot_args = " ".join(
                [
                    "console=ttyS0", "reboot=k", "panic=1", "pci=off", "root=/dev/vda", "ro",
                    "rootfstype=ext4", f"init={rootfs.GUEST_INIT}",
                    *([] if debug else ["quiet", "loglevel=3"]),
                    "iterant.cfg=" + base64.urlsafe_b64encode(json.dumps(cfg).encode()).decode(),
                ]
            )  # fmt: skip
            vm_cfg = {
                "boot-source": {"kernel_image_path": kernel, "boot_args": boot_args},
                "drives": [
                    {"drive_id": "rootfs", "path_on_host": str(base_img),
                     "is_root_device": True, "is_read_only": True},
                    {"drive_id": "work", "path_on_host": str(sbx_dir / "work.ext4"),
                     "is_root_device": False, "is_read_only": False},
                ],
                "machine-config": {
                    "vcpu_count": min(max(math.ceil(cpu), 1), 32),
                    "mem_size_mib": mem + GUEST_OVERHEAD_MIB,
                },
                "vsock": {"guest_cid": GUEST_CID, "uds_path": uds},
            }  # fmt: skip
            cfg_path = sbx_dir / "vm.json"
            cfg_path.write_text(json.dumps(vm_cfg))
            argv = [fc, "--no-api", "--config-file", str(cfg_path)]
            killer = shutil.which("timeout")
            if killer:  # the VM ends itself even if the host dies
                argv = [killer, "-s", "KILL", str(math.ceil(ttl) + SANDBOX_GRACE_S), *argv]
            with open(sbx_dir / "fc.log", "wb") as logf:
                proc = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                    cwd=sbx_dir, start_new_session=True,
                )  # fmt: skip
            vm = proc

            def connect(deadline: float) -> Conn:
                return connect_vsock_uds(uds, deadline)

            def kill_vm() -> None:
                kill_process_group(vm)

            def cleanup() -> None:
                if relay is not None:
                    relay.close()
                remove_tree(str(sbx_dir))

            sandbox = FirecrackerSandbox(
                name=name, vm=vm, connect=connect, ttl=ttl,
                kill_vm=kill_vm, cleanup=cleanup, proxy_url=proxy_url,
            )  # fmt: skip
            try:
                sandbox.wait_ready(time.monotonic() + self._boot_timeout)
            except SandboxError as e:
                raise SandboxError(
                    f"{e}\n--- firecracker log ---\n{_tail(sbx_dir / 'fc.log')}"
                ) from e
        except BaseException:
            if proc is not None:
                try:
                    kill_process_group(proc)
                except SandboxError:
                    pass
            if relay is not None:
                relay.close()
            shutil.rmtree(sbx_dir, ignore_errors=True)
            raise
        sandbox.start_ttl()
        return sandbox


def _tail(path: Path) -> str:
    try:
        return path.read_bytes()[-_LOG_TAIL:].decode(errors="replace")
    except OSError:
        return "(no log)"


__all__ = ["FirecrackerBackend", "kvm_usable"]
