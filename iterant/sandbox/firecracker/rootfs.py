"""Turn an OCI image into a read-only ext4 root file system for the microVM (no root needed).

``<podman|docker> export`` gives the flattened file tree; it is unpacked as the current user,
the guest init and agent are added, setuid/setgid bits are stripped, and ``mke2fs -d`` builds
the image from the directory. The image is cached by image id plus a hash of the guest code and
shared read-only by every sandbox of that image (the per-sandbox writable disk is separate).

Limits, by construction: file owners in the image are not preserved (the tree is owned by the
host user, so the guest sees a non-root, non-sandbox owner and the file system is read-only
anyway), and device nodes under ``/dev`` are dropped (the guest mounts devtmpfs).
"""

# The host checker may run on a platform without these POSIX APIs (e.g. Windows).
# pyright: reportAttributeAccessIssue=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

from iterant.sandbox.container import SandboxError

GUEST_DIR = "/.iterant"
GUEST_INIT = f"{GUEST_DIR}/init"
GUEST_PY = f"{GUEST_DIR}/guest.py"
SANDBOX_UID = 10001
_SHELL_INIT = (
    "#!/bin/sh\n"
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
    "export PATH\n"
    f"exec python3 {GUEST_PY}\n"
)
_PY_DIRS = ("usr/local/bin", "usr/bin", "bin")
_SKELETON = ("proc", "sys", "dev", "tmp", "work")
_BUILD_TIMEOUT_S = 900.0
_ADMIN_TIMEOUT_S = 120.0
_MKE2FS_CANDIDATES = ("/sbin/mke2fs", "/usr/sbin/mke2fs")


def guest_source() -> str:
    return (Path(__file__).with_name("guest.py")).read_text(encoding="utf-8")


def guest_digest() -> str:
    h = hashlib.sha256()
    h.update(_SHELL_INIT.encode())
    h.update(guest_source().encode())
    return h.hexdigest()[:12]


def find_mke2fs() -> str | None:
    found = shutil.which("mke2fs")
    if found:
        return found
    return next((p for p in _MKE2FS_CANDIDATES if os.access(p, os.X_OK)), None)


def _run(argv: list[str], what: str, timeout: float = _ADMIN_TIMEOUT_S) -> bytes:
    try:
        out = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        raise SandboxError(f"{what}: {e}") from e
    if out.returncode != 0:
        raise SandboxError(
            f"{what} failed ({out.returncode}): {out.stderr.decode(errors='replace')}"
        )
    return out.stdout


def make_ext4(
    target: Path, size_mib: int, *, label: str, source: Path | None = None, owner: str = "0:0"
) -> None:
    """Create an ext4 image of ``size_mib`` (sparse), optionally populated from ``source``."""
    mke2fs = find_mke2fs()
    if mke2fs is None:
        raise SandboxError("mke2fs (e2fsprogs) is not installed")
    with open(target, "wb") as f:
        f.truncate(size_mib * 1024 * 1024)
    argv = [mke2fs, "-q", "-F", "-t", "ext4", "-m", "0", "-O", "^has_journal", "-L", label]
    argv += ["-E", f"root_owner={owner}"]
    if source is not None:
        argv += ["-d", str(source)]
    _run([*argv, str(target)], "mke2fs")


def _tree_bytes(root: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        total += 4096 * (1 + len(dirnames))
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size + 4096
            except OSError:
                pass
    return total


def _export_tree(runtime: str, image: str, tree: Path, ctr: str) -> None:
    _run([runtime, "create", "--name", ctr, image, "/bin/true"], f"{runtime} create")
    try:
        exp = subprocess.Popen([runtime, "export", ctr], stdout=subprocess.PIPE)
        tar = subprocess.Popen(
            ["tar", "-x", "-p", "--no-same-owner", "--exclude=./dev/*", "--exclude=dev/*",
             "-C", str(tree)],
            stdin=exp.stdout,
            stderr=subprocess.PIPE,
        )  # fmt: skip
        assert exp.stdout is not None
        exp.stdout.close()
        try:
            _, err = tar.communicate(timeout=_BUILD_TIMEOUT_S)
            exp.wait(timeout=60)
        except subprocess.TimeoutExpired as e:
            tar.kill()
            exp.kill()
            raise SandboxError("exporting the image timed out") from e
        if tar.returncode != 0 or exp.returncode != 0:
            raise SandboxError(f"unpacking {image} failed: {err.decode(errors='replace')}")
    finally:
        subprocess.run([runtime, "rm", "-f", ctr], capture_output=True, timeout=_ADMIN_TIMEOUT_S)


def _prepare_tree(tree: Path) -> None:
    if os.getuid() == SANDBOX_UID:
        raise SandboxError(f"the host user must not have uid {SANDBOX_UID} (the guest sandbox uid)")
    if not any(os.path.lexists(tree / d / "python3") for d in _PY_DIRS):
        raise SandboxError("the image must provide python3 (the guest agent is Python)")
    if not os.path.lexists(tree / "bin" / "sh"):
        raise SandboxError("the image must provide /bin/sh")
    for name in _SKELETON:
        p = tree / name
        if p.is_symlink() or (p.exists() and not p.is_dir()):
            raise SandboxError(f"/{name} in the image is not a plain directory")
        p.mkdir(exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(tree):
        for name in (*dirnames, *filenames):
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            if not os.path.islink(path) and st.st_mode & 0o6000:
                os.chmod(path, st.st_mode & ~0o6000)
    guest = tree / GUEST_DIR.lstrip("/")
    guest.mkdir(mode=0o755)
    (guest / "init").write_text(_SHELL_INIT)
    os.chmod(guest / "init", 0o755)
    (guest / "guest.py").write_text(guest_source())
    os.chmod(guest / "guest.py", 0o644)


def image_id(runtime: str, image: str) -> str:
    have = subprocess.run(
        [runtime, "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        timeout=_ADMIN_TIMEOUT_S,
    )
    if have.returncode != 0:
        _run([runtime, "pull", "--quiet", image], f"pull {image}", _BUILD_TIMEOUT_S)
        have = subprocess.run(
            [runtime, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            timeout=_ADMIN_TIMEOUT_S,
        )
        if have.returncode != 0:
            raise SandboxError(
                f"cannot inspect image {image}: {have.stderr.decode(errors='replace')}"
            )
    iid = have.stdout.decode().strip().removeprefix("sha256:")
    if not iid or not all(c in "0123456789abcdef" for c in iid):
        raise SandboxError(f"unexpected image id {iid!r}")
    return iid


def build_rootfs(runtime: str, image: str, cache_dir: Path, name_prefix: str) -> Path:
    """Return the cached ext4 for ``image``, building it first if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    iid = image_id(runtime, image)
    final = cache_dir / f"rootfs-{iid[:32]}-{guest_digest()}.ext4"
    if final.exists():
        return final
    work = Path(tempfile.mkdtemp(prefix="build-", dir=cache_dir))
    try:
        tree = work / "tree"
        tree.mkdir()
        _export_tree(runtime, image, tree, f"{name_prefix}-rootfs-{secrets.token_hex(4)}")
        _prepare_tree(tree)
        size_mib = int(_tree_bytes(tree) * 1.3) // (1024 * 1024) + 64
        img = work / "rootfs.ext4"
        make_ext4(img, size_mib, label="iterant-root", source=tree)
        os.replace(img, final)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return final
