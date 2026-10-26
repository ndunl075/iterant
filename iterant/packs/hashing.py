"""Pack content hash and verification (invariant 7).

Canonical form of the hash, ``H = sha256``::

    H.update(b"iterant-pack-v1\\n")
    for every regular file under the pack root, sorted by its POSIX relative
    path encoded as UTF-8 (bytewise order):
        H.update(len(path).to_bytes(8, "big")); H.update(path)
        H.update(len(data).to_bytes(8, "big")); H.update(data)

Every file counts (``pack.toml``, ``splits.json``, each task's ``input.json``,
``check`` and ``task.toml``, notices), so any change to a task, a check, the tags
or the split assignment changes the hash. Lengths are prefixed, so no two
different listings share a byte stream. Symlinks are refused: a link could point
outside the pack and make the hash depend on files it does not cover.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from iterant.contracts import Pack

_DOMAIN = b"iterant-pack-v1\n"


class PackError(Exception):
    """A pack is malformed, or does not match its pinned hash."""


class PackHashMismatch(PackError):
    """The pack's content hash differs from the pinned one (invariant 7: abort the batch)."""


def _files(root: Path) -> list[tuple[bytes, Path]]:
    found: list[tuple[bytes, Path]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise PackError(f"symlink in pack: {entry}")
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                found.append((entry.relative_to(root).as_posix().encode("utf-8"), entry))
            else:
                raise PackError(f"not a regular file or directory: {entry}")
    found.sort(key=lambda item: item[0])
    return found


def compute_content_hash(root: Path) -> str:
    """Hash every file under ``root`` in the canonical form described in the module docstring."""
    if not root.is_dir():
        raise PackError(f"pack directory not found: {root}")
    digest = hashlib.sha256(_DOMAIN)
    for path_bytes, path in _files(root):
        data = path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def verify(pack: Pack, expected_hash: str, root: Path | None = None) -> None:
    """Raise ``PackHashMismatch`` unless ``pack`` matches ``expected_hash``.

    With ``root`` the pack directory is re-hashed from disk too, which catches
    edits made after ``load_pack``. The orchestrator passes ``root`` before every batch.
    """
    if pack.content_hash != expected_hash:
        raise PackHashMismatch(
            f"pack {pack.name!r} hash {pack.content_hash} != pinned {expected_hash}"
        )
    if root is not None:
        actual = compute_content_hash(root)
        if actual != expected_hash:
            raise PackHashMismatch(
                f"pack {pack.name!r} on disk hashes to {actual}, pinned {expected_hash}"
            )
