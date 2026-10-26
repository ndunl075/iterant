"""Container sandbox backends (``podman`` rootless; ``docker`` only with ``unsafe_container``).

Isolation per ARCHITECTURE §5/§9: non-root user, read-only root fs, a size-capped tmpfs for
work files, no host mounts, cpu/mem/pids caps (verified after start), a ttl, and no network
except (``proxy-only``) one relay to the proxy. Each sandbox gets its own internal network
and its own relay.
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
SHM_SIZE = "1m"
PROXY_ENV = "ITERANT_PROXY_URL"
RELAY_PORT = 8080
RELAY_GRACE_S = 30
# alpine/socat 1.8.1.3 (multi-arch index digest). Needs busybox `timeout` and `socat`.
DEFAULT_RELAY_IMAGE = (
    "docker.io/alpine/socat@sha256:603174b77eebaba25ba8f8ebb15e1234d6600409554e95d6fb203fd106150287"
)

_IO_TIMEOUT_S = 120.0
_ADMIN_TIMEOUT_S = 60.0
_PULL_TIMEOUT_S = 900.0
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_GONE_MARKERS = (
    "no such",
    "not found",
    "no container",
    "does not exist",
    "not known",
    "in progress",
)

# Run in the fresh container: needed tools exist, and the cgroup caps really applied
# (rootless runtimes without cgroup-v2 delegation silently ignore --cpus/--pids-limit).
_VERIFY_SCRIPT = (
    "setsid -w true && command -v sleep >/dev/null || exit 3; "
    "cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/pids.max /sys/fs/cgroup/memory.max"
)


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


def _parse_proxy(proxy_url: str) -> tuple[str, int, str]:
    parts = urlsplit(proxy_url)
    host = parts.hostname or ""
    # The host lands in a socat address in the relay; accept only plain names/IPv4 so
    # a crafted URL cannot inject socat options. http only: the proxy terminates TLS upstream.
    if parts.scheme != "http" or not _HOST_RE.fullmatch(host):
        raise ValueError(f"proxy_url must be http://<hostname-or-ipv4>[:port]/, got {proxy_url!r}")
    if parts.username or parts.password:
        raise ValueError("proxy_url must not carry credentials")
    return host, parts.port or 80, parts.path


def _check_caps(stdout: str, cpu: float, mem: int) -> str | None:
    """Return a problem description, or None if the cgroup limits match what was asked."""
    lines = stdout.split("\n")
    try:
        quota, period = lines[0].split()
        ok_cpu = quota != "max" and abs(int(quota) / int(period) - cpu) <= 0.01 * cpu + 0.001
        ok_pids = int(lines[1]) == PIDS_LIMIT
        ok_mem = int(lines[2]) == mem * 1024 * 1024
    except (IndexError, ValueError, ZeroDivisionError):
        return f"cgroup limits unreadable: {stdout!r}"
    bad = [n for n, ok in (("cpu", ok_cpu), ("pids", ok_pids), ("memory", ok_mem)) if not ok]
    return f"cgroup limits not applied: {', '.join(bad)}" if bad else None


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

    @property
    def resource_id(self) -> str:
        """Value of the ``iterant.sandbox.id`` label on this sandbox's container/relay/network."""
        return self._name

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

    def _container_exists(self) -> bool:
        out = self._run([self._rt, "container", "inspect", self._name], timeout=_ADMIN_TIMEOUT_S)
        return out.returncode == 0

    def _confirmed_removed(self, name: str) -> bool:
        try:
            out = self._runner([self._rt, "container", "inspect", name], timeout=_ADMIN_TIMEOUT_S)
        except OSError:
            return False
        return out.returncode != 0 and _gone(out)

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
        pidfile = f"/tmp/.exec-{secrets.token_hex(6)}.pid"
        # Each exec is its own session/process group (setsid); the inner shell records the
        # group id, so a timeout can kill exactly this exec's processes and nothing else.
        # -w: a group-leader setsid otherwise forks and returns 0 at once, losing the result.
        wrapper = f'echo $$ > {pidfile}; sh -c "$0"; rc=$?; rm -f {pidfile}; exit $rc'
        start = time.monotonic()
        out = self._run(
            [
                self._rt, "exec", "--workdir", WORK_DIR, self._name,
                "setsid", "-w", "sh", "-c", wrapper, cmd,
            ],
            timeout=timeout,
        )  # fmt: skip
        wall = time.monotonic() - start
        if out.timed_out:
            if not self._kill_group(pidfile):
                # Cannot prove the process is gone (e.g. pids limit hit): fail closed.
                self._abort()
            return ExecResult(-1, _text(out.stdout), _text(out.stderr), True, wall)
        if out.returncode in (125, 126) and not self._container_exists():
            raise SandboxError(f"sandbox {self._name} is gone")
        return ExecResult(out.returncode, _text(out.stdout), _text(out.stderr), False, wall)

    def _kill_group(self, pidfile: str) -> bool:
        script = (
            f"p=$(cat {pidfile}) || exit 2; kill -9 -$p 2>/dev/null; i=0; "
            "while kill -0 -$p 2>/dev/null; do i=$((i+1)); [ $i -gt 30 ] && exit 1; "
            f"sleep 0.1; done; rm -f {pidfile}"
        )
        try:
            out = self._run([self._rt, "exec", self._name, "sh", "-c", script], timeout=30)
        except SandboxError:
            return False
        return out.returncode == 0 and not out.timed_out

    def _abort(self) -> None:
        try:
            self.destroy()
        except SandboxError:
            log.exception("could not tear down %s after a failed kill", self._name)

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
                    if argv[1:3] == ["rm", "-f"] and self._confirmed_removed(argv[3]):
                        # e.g. rootless podman fails killing its netns helper after the
                        # container itself is already deleted
                        log.warning("%s: %s", " ".join(argv), _text(out.stderr).strip())
                        continue
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
        sandbox_extra_args: Sequence[str],
        relay_extra_args: Sequence[str],
        work_tmpfs_owner: str,
        network_extra_args: Sequence[str] = (),
        runner: Runner,
    ) -> None:
        if not _NAME_RE.fullmatch(name_prefix):
            raise ValueError("name_prefix must match [a-z0-9][a-z0-9-]* (use slugify())")
        self._rt = runtime
        self._prefix = name_prefix
        self._runner = runner
        self._relay_image = relay_image
        self._outer_network = outer_network
        self._sandbox_extra = list(sandbox_extra_args)
        self._relay_extra = list(relay_extra_args)
        self._work_tmpfs_owner = work_tmpfs_owner
        self._network_extra = list(network_extra_args)
        self._proxy = _parse_proxy(proxy_url) if proxy_url is not None else None

    @property
    def runtime(self) -> str:
        return self._rt

    def _run(self, argv: Sequence[str], what: str, timeout: float = _ADMIN_TIMEOUT_S) -> RunOutput:
        try:
            out = self._runner(argv, timeout=timeout)
        except OSError as e:
            raise SandboxError(f"cannot run {self._rt}: {e}") from e
        if out.returncode != 0 or out.timed_out:
            raise SandboxError(f"{what} failed ({out.returncode}): {_text(out.stderr).strip()}")
        return out

    def _run_raw(self, argv: Sequence[str]) -> RunOutput:
        try:
            return self._runner(argv, timeout=_ADMIN_TIMEOUT_S)
        except OSError as e:
            raise SandboxError(f"cannot run {self._rt}: {e}") from e

    def _ensure_image(self, image: str) -> None:
        # Explicit pull with its own long timeout, instead of an implicit pull in `run`.
        try:
            have = self._runner([self._rt, "image", "inspect", image], timeout=_ADMIN_TIMEOUT_S)
        except OSError as e:
            raise SandboxError(f"cannot run {self._rt}: {e}") from e
        if have.returncode != 0:
            self._run([self._rt, "pull", "--quiet", image], f"pull {image}", _PULL_TIMEOUT_S)

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
        label = [
            "--label", f"iterant.sandbox={self._prefix}",
            "--label", f"iterant.sandbox.id={base}",
        ]  # fmt: skip
        rt = self._rt
        teardown: list[list[str]] = [[rt, "rm", "-f", name]]
        env: list[str] = []
        network = "none"
        proxy_url: str | None = None

        try:
            self._ensure_image(image)
            if net == "proxy-only":
                assert self._proxy is not None
                host, port, path = self._proxy
                network = netname
                proxy_url = f"http://{relay}:{RELAY_PORT}{path}"
                env = ["--env", f"{PROXY_ENV}={proxy_url}"]
                teardown += [[rt, "rm", "-f", relay], [rt, "network", "rm", netname]]
                self._ensure_image(self._relay_image)
                self._run(
                    [rt, "network", "create", "--internal", *self._network_extra, *label, netname],
                    "network create",
                )
                self._run(self.relay_argv(relay, netname, host, port, ttl, label), "relay run")
                self._run([rt, "network", "connect", self._outer_network, relay], "relay connect")
            self._run(self.run_argv(name, image, cpu, mem, disk, ttl, network, env, label), "run")
            check = self._run_raw([rt, "exec", name, "sh", "-c", _VERIFY_SCRIPT])
            if check.returncode == 3:
                raise SandboxError("image must provide `sleep` and util-linux `setsid` with -w")
            if check.returncode != 0:
                raise SandboxError(
                    "cannot read cgroup limits: needs cgroup v2 with cpu/pids/memory controller "
                    f"delegation ({_text(check.stderr).strip()})"
                )
            problem = _check_caps(_text(check.stdout), cpu, mem)
            if problem:
                raise SandboxError(
                    f"{problem} (rootless runtimes need cgroup-v2 delegation for cpu/pids limits)"
                )
        except BaseException:
            for argv in teardown:  # same order as destroy(): container, relay, then network
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

    def reap(self) -> None:
        """Remove leftover containers and networks carrying this backend's ``name_prefix``
        label (e.g. after a crash). Scoped by label, never global; it also removes sandboxes
        of any other live process that uses the same prefix."""
        selector = ["--filter", f"label=iterant.sandbox={self._prefix}"]
        rt = self._rt
        errors: list[str] = []
        for kind, listing, remove in (
            ("container", [rt, "ps", "-a", "--format", "{{.Names}}", *selector], [rt, "rm", "-f"]),
            ("network", [rt, "network", "ls", "--format", "{{.Name}}", *selector],
             [rt, "network", "rm"]),
        ):  # fmt: skip
            out = self._run(listing, f"list {kind}s")
            for item in _text(out.stdout).split():
                done = self._runner([*remove, item], timeout=_ADMIN_TIMEOUT_S)
                if done.returncode != 0 and not _gone(done):
                    errors.append(f"{kind} {item}: {_text(done.stderr).strip()}")
        if errors:
            raise SandboxError("reap incomplete: " + "; ".join(errors))

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
            self._rt, "run", "--detach", "--rm", "--init", "--name", name, *label,
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--read-only", *self._sandbox_extra,
            # exec so agents can run scripts they write; the tmpfs is the only writable disk.
            "--tmpfs", f"{WORK_DIR}:rw,exec,nosuid,nodev,size={disk}m,mode=0700,"
            f"{self._work_tmpfs_owner}",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={TMP_TMPFS_MIB}m,mode=1777",
            "--shm-size", SHM_SIZE,
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--cpus", f"{cpu:g}",
            "--memory", f"{mem}m", "--memory-swap", f"{mem}m",
            "--pids-limit", str(PIDS_LIMIT),
            "--network", network,
            "--workdir", WORK_DIR, "--env", f"HOME={WORK_DIR}",
            *env,
            # `sleep ttl` is the main process: the container ends itself even if the host dies.
            "--entrypoint", "sleep", image, str(math.ceil(ttl)),
        ]  # fmt: skip

    def relay_argv(
        self, relay: str, netname: str, host: str, port: int, ttl: float, label: Sequence[str]
    ) -> list[str]:
        """TCP relay: listens on the sandbox's private net, forwards to the one proxy endpoint.
        Started on the internal network only; the backend then connects the outer network."""
        return [
            self._rt, "run", "--detach", "--rm", "--name", relay, *label,
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
            # podman's --read-only otherwise leaves /run, /var/tmp writable (outside the disk cap)
            sandbox_extra_args=("--read-only-tmpfs=false",),
            relay_extra_args=(),
            # podman's --tmpfs rejects uid=/gid=; `U` chowns the mount to the container user
            work_tmpfs_owner="U",
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
            sandbox_extra_args=(),
            relay_extra_args=("--add-host", "host.docker.internal:host-gateway"),
            work_tmpfs_owner=f"uid={SANDBOX_UID},gid={SANDBOX_UID}",
            # no host-side gateway address on the internal net: the sandbox cannot reach host
            # services through the bridge IP
            network_extra_args=("--opt", "com.docker.network.bridge.gateway_mode_ipv4=isolated"),
            runner=runner,
        )
