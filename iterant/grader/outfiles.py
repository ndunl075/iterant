"""Read and bound agent output (``/out``) as plain data (invariant 8).

``collect_out_files`` reads a directory without following links: only regular files with
a single hard link are accepted (no symlinks, junctions, hard links, devices, FIFOs or
sockets), within the caps below. Nothing here executes, imports or unpickles a file.
Every rejection is a fixed message: file names are agent-controlled.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

MAX_OUT_FILES = 1000
MAX_OUT_TOTAL_BYTES = 16 * 1024 * 1024
MAX_OUT_FILE_BYTES = 4 * 1024 * 1024
MAX_OUT_PATH_DEPTH = 16

_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: Windows junctions and symlinks


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


def collect_out_files(root: Path) -> dict[str, bytes]:
    """Read ``root`` as ``{posix path relative to root: bytes}`` under the caps."""
    files: dict[str, bytes] = {}
    total = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth >= MAX_OUT_PATH_DEPTH:
            raise OutputRejected("output path too deep")
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError as exc:
            raise OutputRejected("unreadable output") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                st = os.lstat(path)
            except OSError as exc:
                raise OutputRejected("unreadable output") from exc
            if getattr(st, "st_file_attributes", 0) & _REPARSE_POINT or stat.S_ISLNK(st.st_mode):
                raise OutputRejected("output contains a link")
            if stat.S_ISDIR(st.st_mode):
                stack.append((path, depth + 1))
                continue
            if not stat.S_ISREG(st.st_mode):
                raise OutputRejected("output contains a special file")
            if st.st_nlink > 1:
                raise OutputRejected("output contains a hard link")
            if len(files) >= MAX_OUT_FILES:
                raise OutputRejected("too many output files")
            total += st.st_size
            if st.st_size > MAX_OUT_FILE_BYTES:
                raise OutputRejected("output file too large")
            if total > MAX_OUT_TOTAL_BYTES:
                raise OutputRejected("output too large")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise OutputRejected("unreadable output") from exc
            files[path.relative_to(root).as_posix()] = data
    return files
