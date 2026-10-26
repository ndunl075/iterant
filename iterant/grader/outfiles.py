"""Read and bound agent output (``/out``) as plain data (invariant 8).

``collect_out_files`` reads a directory tree without following links: only regular files
with a single hard link are accepted (no symlinks, junctions, hard links, devices, FIFOs
or sockets), within the caps below. Nothing here executes, imports or unpickles a file.
Every rejection is a fixed message: file names are agent-controlled.

Races: the agent's sandbox is stopped before this runs, but the checks are still made
race-free where the OS allows. On POSIX every directory and file is opened relative to its
parent's file descriptor with ``O_NOFOLLOW`` (and ``O_NONBLOCK``, so a swapped-in FIFO
cannot hang the read), then ``fstat`` must match the earlier ``lstat`` (same device and
inode, regular, one link), and at most cap+1 bytes are read. On Windows there is no
``O_NOFOLLOW``: links are detected by the reparse-point attribute and ``lstat``/``fstat``
are compared, which leaves a small window if something else swaps a path during the read.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

MAX_OUT_FILES = 1000
MAX_OUT_ENTRIES = 2000  # files and directories together, counted before anything is read
MAX_OUT_TOTAL_BYTES = 16 * 1024 * 1024
MAX_OUT_FILE_BYTES = 4 * 1024 * 1024
MAX_OUT_PATH_DEPTH = 16

_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: Windows junctions and symlinks
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
_DIR_FLAGS = os.O_RDONLY | _NOFOLLOW | getattr(os, "O_DIRECTORY", 0)


class OutputRejected(ValueError):
    """The agent's output breaks a rule; the run scores 0.0."""


def check_limits(files: Mapping[str, bytes]) -> None:
    """Raise ``OutputRejected`` if ``files`` exceeds the count, size or depth caps."""
    if len(files) > MAX_OUT_FILES:
        raise OutputRejected("too many output files")
    if sum(len(d) for d in files.values()) > MAX_OUT_TOTAL_BYTES:
        raise OutputRejected("output too large")
    if any(len(d) > MAX_OUT_FILE_BYTES for d in files.values()):
        raise OutputRejected("output file too large")
    if any(len(rel.split("/")) > MAX_OUT_PATH_DEPTH for rel in files):
        raise OutputRejected("output path too deep")


class _Acc:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.entries = 0
        self.total = 0


def _is_link(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _REPARSE_POINT) or stat.S_ISLNK(st.st_mode)


def _same_inode(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_ino == b.st_ino and a.st_dev == b.st_dev


def _read_fd(fd: int, lst: os.stat_result, rel: str, acc: _Acc) -> None:
    """Read the file open as ``fd`` if it is still what ``lst`` described; closes ``fd``."""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not _same_inode(st, lst):
            raise OutputRejected("output changed while reading")
        if st.st_nlink > 1:
            raise OutputRejected("output contains a hard link")
        if len(acc.files) >= MAX_OUT_FILES:
            raise OutputRejected("too many output files")
        if st.st_size > MAX_OUT_FILE_BYTES:
            raise OutputRejected("output file too large")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(MAX_OUT_FILE_BYTES + 1)  # the file may have grown since fstat
    except OSError as exc:
        raise OutputRejected("unreadable output") from exc
    finally:
        os.close(fd)
    if len(data) > MAX_OUT_FILE_BYTES:
        raise OutputRejected("output file too large")
    acc.total += len(data)
    if acc.total > MAX_OUT_TOTAL_BYTES:
        raise OutputRejected("output too large")
    acc.files[rel] = data


def _count(acc: _Acc) -> None:
    acc.entries += 1
    if acc.entries > MAX_OUT_ENTRIES:
        raise OutputRejected("too many output files")


def _walk_posix(fd: int, prefix: str, depth: int, acc: _Acc) -> None:
    if depth >= MAX_OUT_PATH_DEPTH:
        raise OutputRejected("output path too deep")
    names: list[str] = []
    with os.scandir(fd) as it:
        for entry in it:
            _count(acc)
            names.append(entry.name)
    for name in sorted(names):
        rel = prefix + name
        lst = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(lst.st_mode):
            raise OutputRejected("output contains a link")
        if stat.S_ISDIR(lst.st_mode):
            child = os.open(name, _DIR_FLAGS, dir_fd=fd)
            try:
                if not _same_inode(os.fstat(child), lst):
                    raise OutputRejected("output changed while reading")
                _walk_posix(child, rel + "/", depth + 1, acc)
            finally:
                os.close(child)
        elif stat.S_ISREG(lst.st_mode):
            try:
                file_fd = os.open(name, _FILE_FLAGS, dir_fd=fd)
            except OSError as exc:  # ELOOP: swapped for a symlink
                raise OutputRejected("output changed while reading") from exc
            _read_fd(file_fd, lst, rel, acc)
        else:
            raise OutputRejected("output contains a special file")


def _walk_windows(directory: Path, prefix: str, depth: int, acc: _Acc) -> None:
    if depth >= MAX_OUT_PATH_DEPTH:
        raise OutputRejected("output path too deep")
    names: list[str] = []
    with os.scandir(directory) as it:
        for entry in it:
            _count(acc)
            names.append(entry.name)
    for name in sorted(names):
        path, rel = directory / name, prefix + name
        lst = os.lstat(path)
        if _is_link(lst):
            raise OutputRejected("output contains a link")
        if stat.S_ISDIR(lst.st_mode):
            _walk_windows(path, rel + "/", depth + 1, acc)
        elif stat.S_ISREG(lst.st_mode):
            _read_fd(os.open(path, _FILE_FLAGS), lst, rel, acc)
        else:
            raise OutputRejected("output contains a special file")


def collect_out_files(root: Path) -> dict[str, bytes]:
    """Read ``root`` as ``{posix path relative to root: bytes}`` under the caps."""
    acc = _Acc()
    try:
        if os.name == "nt":
            if _is_link(os.lstat(root)):
                raise OutputRejected("output contains a link")
            _walk_windows(root, "", 0, acc)
        else:
            fd = os.open(root, _DIR_FLAGS)
            try:
                _walk_posix(fd, "", 0, acc)
            finally:
                os.close(fd)
    except OSError as exc:
        raise OutputRejected("unreadable output") from exc
    return dict(sorted(acc.files.items()))
