"""Rootfs tree preparation: setuid stripping, unreadable files, refusals, cache key."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from iterant.sandbox import SandboxError
from iterant.sandbox.firecracker import rootfs
from iterant.sandbox.firecracker.sandbox import remove_tree

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX modes")


@pytest.fixture(autouse=True)
def _host_uid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rootfs, "_host_uid", lambda: 1000)


def make_tree(root: Path, *, python: bool = True, sh: bool = True) -> Path:
    (root / "usr/local/bin").mkdir(parents=True)
    (root / "bin").mkdir()
    if python:
        (root / "usr/local/bin/python3").write_text("py")
    if sh:
        (root / "bin/sh").write_text("sh")
    return root


@posix_only
def test_prepare_strips_setuid_and_opens_unreadable_files(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "t")
    su = tree / "bin/su"
    su.write_text("su")
    os.chmod(su, 0o4755)
    sg = tree / "bin/sg"
    sg.write_text("sg")
    os.chmod(sg, 0o2755)
    secret = tree / "etc"
    secret.mkdir()
    (secret / "shadow").write_text("x")
    os.chmod(secret / "shadow", 0o000)
    os.chmod(secret, 0o000)  # a mode 000 directory holding a mode 000 file
    rootfs._prepare_tree(tree)  # pyright: ignore[reportPrivateUsage]
    assert not (os.lstat(su).st_mode & 0o6000) and (os.lstat(su).st_mode & 0o777) == 0o755
    assert not (os.lstat(sg).st_mode & 0o6000)
    assert os.lstat(secret / "shadow").st_mode & 0o600 == 0o600  # readable for mke2fs
    for name in ("proc", "sys", "dev", "tmp", "work"):
        assert (tree / name).is_dir()
    init = tree / ".iterant/init"
    assert init.read_text().startswith("#!/bin/sh") and os.access(init, os.X_OK)
    assert (tree / ".iterant/guest.py").read_text() == rootfs.guest_source()


def test_prepare_refuses_images_without_python_or_sh(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="python3"):
        rootfs._prepare_tree(make_tree(tmp_path / "a", python=False))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(SandboxError, match="/bin/sh"):
        rootfs._prepare_tree(make_tree(tmp_path / "b", sh=False))  # pyright: ignore[reportPrivateUsage]


def test_prepare_refuses_a_non_directory_skeleton_entry(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "t")
    (tree / "tmp").write_text("file")
    with pytest.raises(SandboxError, match="/tmp"):
        rootfs._prepare_tree(tree)  # pyright: ignore[reportPrivateUsage]


def test_prepare_refuses_a_preexisting_guest_dir(tmp_path: Path) -> None:
    tree = make_tree(tmp_path / "t")
    (tree / ".iterant").mkdir()
    with pytest.raises(SandboxError, match=".iterant"):
        rootfs._prepare_tree(tree)  # pyright: ignore[reportPrivateUsage]


def test_prepare_refuses_the_sandbox_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rootfs, "_host_uid", lambda: rootfs.SANDBOX_UID)
    with pytest.raises(SandboxError, match="uid"):
        rootfs._prepare_tree(make_tree(tmp_path / "t"))  # pyright: ignore[reportPrivateUsage]


def test_cache_key_covers_the_format_version(monkeypatch: pytest.MonkeyPatch) -> None:
    a = rootfs.guest_digest()
    monkeypatch.setattr(rootfs, "FORMAT_VERSION", rootfs.FORMAT_VERSION + 1)
    assert rootfs.guest_digest() != a


def test_cache_key_covers_rootfs_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = rootfs.guest_digest()
    edited = tmp_path / "rootfs.py"
    (tmp_path / "guest.py").write_bytes(rootfs.guest_source().encode())
    edited.write_bytes(Path(rootfs.__file__).read_bytes() + b"\n# format change\n")
    monkeypatch.setattr(rootfs, "__file__", str(edited))
    assert rootfs.guest_digest() != a


@posix_only
def test_remove_tree_handles_mode_000_directories(tmp_path: Path) -> None:
    tree = tmp_path / "t"
    (tree / "a/b").mkdir(parents=True)
    (tree / "a/b/f").write_text("x")
    os.chmod(tree / "a/b/f", 0)
    os.chmod(tree / "a/b", 0)
    os.chmod(tree / "a", stat.S_IREAD)
    remove_tree(str(tree))
    assert not tree.exists()
