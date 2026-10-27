"""Host-side files the loop needs: a copy of the repo at a commit, and the diff store."""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

from iterant.archive import GitRefError, diff_sha256

_STRIPPED_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")


def export_commit(repo: str | os.PathLike[str], sha: str, dest: Path) -> Path:
    """Copy the tree of commit ``sha`` (no ``.git``, no working-tree changes) into ``dest``."""
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha],
            capture_output=True,
            env=env,
            check=False,
        )
    except OSError as exc:
        raise GitRefError(f"could not run git: {exc}") from exc
    if proc.returncode != 0:
        raise GitRefError(f"git archive {sha} failed: {proc.stderr.decode(errors='replace')}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return dest


class DiffStore:
    """Candidate diffs on disk, by sha256 (the archive keeps only the hash).

    Lives under ``<repo>/.iterant/diffs``, which the archive's ``.gitignore`` already hides.
    """

    def __init__(self, repo: str | os.PathLike[str]) -> None:
        self._dir = Path(repo) / ".iterant" / "diffs"

    def put(self, diff: str) -> str:
        data = diff.encode("utf-8")
        sha = diff_sha256(data)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{sha}.diff"
        if not path.exists():
            tmp = path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return sha

    def get(self, sha: str) -> str:
        """The diff with hash ``sha``; ``""`` for the empty diff (no file is stored)."""
        path = self._dir / f"{sha}.diff"
        if not path.is_file():
            if sha == diff_sha256(b""):
                return ""
            raise FileNotFoundError(f"diff {sha} is not in the store")
        data = path.read_bytes()
        if diff_sha256(data) != sha:
            raise ValueError(f"stored diff {sha} is corrupt")
        return data.decode("utf-8")
