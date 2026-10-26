"""Container sandbox backends (``podman`` rootless; ``docker`` only with ``unsafe_container``).

Isolation per ARCHITECTURE §5/§9: non-root user, read-only root fs, a size-capped tmpfs for
work files, no host mounts, cpu/mem/pids caps, a ttl, and no network except (``proxy-only``)
one relay to the proxy. Each sandbox gets its own internal network and its own relay.
"""

from __future__ import annotations

import logging
import math
import re
import secrets
import threading
import time
from collections.abc import Sequence
from urllib.parse import urlsplit

from iterant.contracts import ExecResult, NetMode, Sandbox
from iterant.sandbox.runner import Runner, RunOutput, subprocess_runner

log = logging.getLogger(__name__)

WORK_DIR = "/work"
SANDBOX_UID = 10001
PIDS_LIMIT = 512
TMP_TMPFS_MIB = 64
PROXY_ENV = "ITERANT_PROXY_URL"
RELAY_PORT = 8080
RELAY_GRACE_S = 30
# TODO(contract): pin by digest; the relay needs `timeout` and `socat` (busybox-based image).
DEFAULT_RELAY_IMAGE = "docker.io/alpine/socat:latest"

_IO_TIMEOUT_S = 120.0
_ADMIN_TIMEOUT_S = 60.0
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_GONE_MARKERS = ("no such", "not found", "no container", "does not exist", "not known")


class SandboxError(RuntimeError):
    """Sandbox could not be created or is no longer usable (destroyed, expired, runtime error)."""


class UnsafeContainerError(SandboxError):
    """The docker backend was constructed without ``unsafe_container=True``."""


def slugify(text: str, limit: int = 40) -> str:
    """Lowercase, only ``a-z0-9-``; used for the resource-name prefix (e.g. a branch name)."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-")
    return slug or "x"


def _text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _gone(out: RunOutput) -> bool:
    err = _text(out.stderr).lower()
    return any(m in err for m in _GONE_MARKERS)


def _require_abs(path: str) -> None:
    if not path.startswith("/") or "\0" in path:
        raise ValueError(f"sandbox path must be absolute, got {path!r}")


def _parse_proxy(proxy_url: str) -> tuple[str, str, int, str]:
    parts = urlsplit(proxy_url)
    host = parts.hostname or ""
    # The host lands in a socat address in the relay; accept only plain names/IPv4 so
    # a crafted URL cannot inject socat options.
    if parts.scheme not in ("http", "https") or not _HOST_RE.fullmatch(host):
        raise ValueError(
            f"proxy_url must be http(s)://<hostname-or-ipv4>[:port]/, got {proxy_url!r}"
        )
    if parts.username or parts.password:
        raise ValueError("proxy_url must not carry credentials")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.scheme, host, port, parts.path


class ContainerSandbox:
    """One running container (plus, for ``proxy-only``, its private network and relay)."""

    def __init__(
        self,
        *,
        runtime: str,
        runner: Runner,
        name: str,
        ttl: float,
        teardown: Sequence[Sequence[str]],
        proxy_url: str | None,
    ) -> None:
        self._rt = runtime
        self._runner = runner
        self._name = name
        self._teardown = [list(a) for a in teardown]
        self.proxy_url = proxy_url
        self._deadline = time.monotonic() + ttl
        self._dead = False
        self._released = False
        self._lock = threading.Lock()
        self._timer = threading.Timer(ttl, self._expire)
        self._timer.daemon = True

    def start_ttl(self) -> None:
        """Arm the ttl timer (called by the backend once the container is running)."""
        self._timer.start()

    def _expire(self) -> None:
        try:
            self.destroy()
        except Exception:  # timer thread: nowhere to raise to
            log.exception("ttl teardown of %s failed", self._name)

    def _alive(self) -> None:
        if self._dead or time.monotonic() >= self._deadline:
            raise SandboxError(f"sandbox {self._name} is destroyed or past its ttl")

    def _run(
        self, argv: Sequence[str], *, stdin: bytes | None = None, timeout: float | None = None
    ) -> RunOutput:
        try:
            return self._runner(argv, stdin=stdin, timeout=timeout)
        except OSError as e:
            raise SandboxError(f"cannot run {self._rt}: {e}") from e

    def _check(self, out: RunOutput, what: str) -> None:
        if out.returncode != 0:
            raise SandboxError(f"{what} failed ({out.returncode}): {_text(out.stderr).strip()}")

    def put(self, path: str, data: bytes) -> None:
        _require_abs(path)
        self._alive()
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        out = self._run(
            [self._rt, "exec", "-i", self._name, "sh", "-c", script, "sh", path],
            stdin=data,
            timeout=_IO_TIMEOUT_S,
        )
        self._check(out, f"put {path}")

    def get(self, path: str) -> bytes:
        _require_abs(path)
        self._alive()
        script = 'test -e "$1" || exit 44; test -d "$1" && exit 45; cat -- "$1"'
        out = self._run(
            [self._rt, "exec", self._name, "sh", "-c", script, "sh", path],
            timeout=_IO_TIMEOUT_S,
        )
        if out.returncode == 44:
            raise FileNotFoundError(path)
        if out.returncode == 45:
            raise IsADirectoryError(path)
        self._check(out, f"get {path}")
        if out.truncated:
            raise SandboxError(f"get {path}: file larger than the output cap")
        return out.stdout

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        if timeout <= 0:
            raise ValueError("timeout must be > 0")
        self._alive()
        start = time.monotonic()
        out = self._run(
            [self._rt, "exec", "--workdir", WORK_DIR, self._name, "sh", "-c", cmd],
            timeout=timeout,
        )
        wall = time.monotonic() - start
        if out.timed_out:
            # Killing the host-side client does not stop the process in the container.
            # Signal everything the sandbox user owns; pid 1 (sleep) ignores it.
            self._run(
                [self._rt, "exec", self._name, "sh", "-c", "kill -9 -1"], timeout=_ADMIN_TIMEOUT_S
            )
            return ExecResult(-1, _text(out.stdout), _text(out.stderr), True, wall)
        if out.returncode == 125 and _gone(out):
            raise SandboxError(f"sandbox {self._name} is gone: {_text(out.stderr).strip()}")
        return ExecResult(out.returncode, _text(out.stdout), _text(out.stderr), False, wall)

    def destroy(self) -> None:
        with self._lock:
            if self._released:
                return
            self._dead = True
            self._timer.cancel()
            errors: list[str] = []
            for argv in self._teardown:
                try:
                    out = self._runner(argv, timeout=_ADMIN_TIMEOUT_S)
                except OSError as e:
                    errors.append(str(e))
                    continue
                if out.returncode != 0 and not _gone(out):
                    errors.append(f"{' '.join(argv)}: {_text(out.stderr).strip()}")
            if errors:
                raise SandboxError("teardown incomplete: " + "; ".join(errors))
            self._released = True


class ContainerBackend:
    """Shared implementation; use ``PodmanBackend`` or ``DockerBackend``."""

    def __init__(
        self,
        runtime: str,
        *,
        name_prefix: str,
        proxy_url: str | None,
        relay_image: str,
        outer_network: str,
        relay_extra_args: Sequence[str],
        runner: Runner,
    ) -> None:
        if not _NAME_RE.fullmatch(name_prefix):
            raise ValueError("name_prefix must match [a-z0-9][a-z0-9-]* (use slugify())")
        self._rt = runtime
        self._prefix = name_prefix
        self._runner = runner
        self._relay_image = relay_image
        self._outer_network = outer_network
        self._relay_extra = list(relay_extra_args)
        self._proxy = _parse_proxy(proxy_url) if proxy_url is not None else None

    def _run(self, argv: Sequence[str], what: str) -> RunOutput:
        try:
            out = self._runner(argv, timeout=_ADMIN_TIMEOUT_S)
        except OSError as e:
            raise SandboxError(f"cannot run {self._rt}: {e}") from e
        if out.returncode != 0 or out.timed_out:
            raise SandboxError(f"{what} failed ({out.returncode}): {_text(out.stderr).strip()}")
        return out

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

        base = f"{self._prefix}-{secrets.token_hex(4)}"
        name, relay, netname = base, f"{base}-relay", f"{base}-net"
        label = ["--label", f"iterant.sandbox={self._prefix}"]
        rt = self._rt
        teardown: list[list[str]] = [[rt, "rm", "-f", name]]
        env: list[str] = []
        network = "none"
        proxy_url: str | None = None

        try:
            if net == "proxy-only":
                assert self._proxy is not None
                scheme, host, port, path = self._proxy
                network = netname
                proxy_url = f"{scheme}://{relay}:{RELAY_PORT}{path}"
                env = ["--env", f"{PROXY_ENV}={proxy_url}"]
                teardown += [[rt, "rm", "-f", relay], [rt, "network", "rm", netname]]
                self._run(
                    [rt, "network", "create", "--internal", *label, netname], "network create"
                )
                self._run(self.relay_argv(relay, netname, host, port, ttl, label), "relay create")
                self._run([rt, "network", "connect", self._outer_network, relay], "relay connect")
                self._run([rt, "start", relay], "relay start")
            self._run(self.run_argv(name, image, cpu, mem, disk, ttl, network, env, label), "run")
        except BaseException:
            for argv in reversed(teardown):
                try:
                    self._runner(argv, timeout=_ADMIN_TIMEOUT_S)
                except OSError:
                    pass
            raise

        sandbox = ContainerSandbox(
            runtime=rt,
            runner=self._runner,
            name=name,
            ttl=ttl,
            teardown=teardown,
            proxy_url=proxy_url,
        )
        sandbox.start_ttl()
        return sandbox

    def run_argv(
        self,
        name: str,
        image: str,
        cpu: float,
        mem: int,
        disk: int,
        ttl: float,
        network: str,
        env: Sequence[str],
        label: Sequence[str],
    ) -> list[str]:
        """The sandbox container. No host mounts: work files go in via ``put``/``get``."""
        return [
            self._rt, "run", "--detach", "--rm", "--name", name, *label,
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--read-only",
            # exec so agents can run scripts they write; the tmpfs is the only writable disk.
            "--tmpfs", f"{WORK_DIR}:rw,exec,nosuid,nodev,size={disk}m,mode=0700,"
            f"uid={SANDBOX_UID},gid={SANDBOX_UID}",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={TMP_TMPFS_MIB}m,mode=1777",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--cpus", f"{cpu:g}",
            "--memory", f"{mem}m", "--memory-swap", f"{mem}m",
            "--pids-limit", str(PIDS_LIMIT),
            "--network", network,
            "--workdir", WORK_DIR, "--env", f"HOME={WORK_DIR}",
            *env,
            # pid 1 is `sleep ttl`: the container ends itself even if the host process dies.
            "--entrypoint", "sleep", image, str(math.ceil(ttl)),
        ]  # fmt: skip

    def relay_argv(
        self, relay: str, netname: str, host: str, port: int, ttl: float, label: Sequence[str]
    ) -> list[str]:
        """TCP relay: listens on the sandbox's private net, forwards to the one proxy endpoint."""
        return [
            self._rt, "create", "--name", relay, *label,
            "--network", netname,
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--cpus", "0.5", "--memory", "64m", "--memory-swap", "64m", "--pids-limit", "64",
            *self._relay_extra,
            "--entrypoint", "timeout", self._relay_image, str(math.ceil(ttl) + RELAY_GRACE_S),
            "socat", f"TCP-LISTEN:{RELAY_PORT},fork,reuseaddr", f"TCP:{host}:{port}",
        ]  # fmt: skip


class PodmanBackend(ContainerBackend):
    """Rootless Podman. ``proxy_url`` is the proxy as seen from the relay container."""

    def __init__(
        self,
        # TODO(contract): SandboxBackend.create has no proxy address, so it is a constructor arg.
        proxy_url: str | None = None,
        *,
        name_prefix: str = "iterant",
        relay_image: str = DEFAULT_RELAY_IMAGE,
        runner: Runner = subprocess_runner,
    ) -> None:
        super().__init__(
            "podman",
            name_prefix=name_prefix,
            proxy_url=proxy_url,
            relay_image=relay_image,
            outer_network="podman",
            relay_extra_args=(),
            runner=runner,
        )


class DockerBackend(ContainerBackend):
    """Docker daemon (root-equivalent on the host): dev only, explicit opt-in."""

    def __init__(
        self,
        proxy_url: str | None = None,
        *,
        unsafe_container: bool = False,
        name_prefix: str = "iterant",
        relay_image: str = DEFAULT_RELAY_IMAGE,
        runner: Runner = subprocess_runner,
    ) -> None:
        if unsafe_container is not True:
            raise UnsafeContainerError(
                "docker is not rootless: the daemon runs as root, so a container escape "
                "is a host root compromise. Pass unsafe_container=True (--unsafe-container)."
            )
        super().__init__(
            "docker",
            name_prefix=name_prefix,
            proxy_url=proxy_url,
            relay_image=relay_image,
            outer_network="bridge",
            relay_extra_args=("--add-host", "host.docker.internal:host-gateway"),
            runner=runner,
        )
