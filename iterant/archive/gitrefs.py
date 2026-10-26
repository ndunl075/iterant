"""Git refs Iterant writes in the target repo (§3 PROMOTE).

Only plumbing commands are used, with a throwaway index file outside the repo,
so the user's working tree, their index and every branch except
``iterant/champion`` are never touched. The branch and the ``iterant/gen-N`` tag
move together in one atomic ref transaction.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from iterant.archive.errors import GitRefError
from iterant.contracts import Candidate

CHAMPION_BRANCH: Final = "iterant/champion"
CHAMPION_REF: Final = f"refs/heads/{CHAMPION_BRANCH}"
_IDENTITY: Final = {
    "GIT_AUTHOR_NAME": "Iterant",
    "GIT_AUTHOR_EMAIL": "iterant@localhost",
    "GIT_COMMITTER_NAME": "Iterant",
    "GIT_COMMITTER_EMAIL": "iterant@localhost",
}


def gen_tag(gen: int) -> str:
    """Short name of the tag for generation ``gen`` (gen 0 is the seed champion)."""
    return f"iterant/gen-{gen}"


def diff_sha256(diff: bytes) -> str:
    return hashlib.sha256(diff).hexdigest()


def _git(
    repo: Path,
    *args: str,
    input: bytes | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    full_env = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=input,
            capture_output=True,
            env=full_env,
            check=False,
        )
    except OSError as exc:
        raise GitRefError(f"could not run git: {exc}") from exc
    if check and proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        raise GitRefError(f"git {' '.join(args)} failed ({proc.returncode}): {err}")
    return proc


def _resolve(repo: Path, rev: str) -> str | None:
    proc = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def champion_sha(repo: str | os.PathLike[str]) -> str | None:
    """Tip of ``iterant/champion``, or None if it doesn't exist yet."""
    return _resolve(Path(repo), CHAMPION_REF)


def gen_tag_sha(repo: str | os.PathLike[str], gen: int) -> str | None:
    return _resolve(Path(repo), f"refs/tags/{gen_tag(gen)}")


def _checked_out_branches(repo: Path) -> set[str]:
    out = _git(repo, "worktree", "list", "--porcelain").stdout.decode()
    return {line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("branch ")}


def _update_refs(repo: Path, new: str, old_champion: str, gen: int) -> None:
    """Move the champion branch and create the gen tag as one all-or-nothing transaction."""
    zero = "0" * len(new)
    tag_ref = f"refs/tags/{gen_tag(gen)}"
    script = (
        "start\n"
        f"update {CHAMPION_REF} {new} {old_champion}\n"
        f"update {tag_ref} {new} {zero}\n"
        "prepare\n"
        "commit\n"
    )
    _git(repo, "update-ref", "--stdin", input=script.encode())


def ensure_seed(repo: str | os.PathLike[str], base: str = "HEAD") -> str:
    """Create ``iterant/champion`` and tag ``iterant/gen-0`` at ``base`` if missing.

    Returns the champion's tip. Does nothing (and ignores ``base``) when the branch
    already exists, so it is safe to call on every start.
    """
    root = Path(repo)
    existing = champion_sha(root)
    if existing is not None:
        return existing
    sha = _resolve(root, base)
    if sha is None:
        raise GitRefError(f"cannot resolve base revision {base!r}")
    _update_refs(root, sha, "0" * len(sha), 0)
    return sha


def promote(
    repo: str | os.PathLike[str],
    cand: Candidate,
    diff: bytes,
    gen: int,
) -> str:
    """Commit ``diff`` on top of ``iterant/champion`` and tag the result ``iterant/gen-<gen>``.

    ``diff`` must be the candidate's patch (``git diff`` format) and hash to
    ``cand.diff_sha``. Returns the new commit sha. Raises ``GitRefError`` if the
    diff doesn't match or apply, the tag already exists, the champion moved
    meanwhile, or the champion branch is checked out somewhere (moving it would
    desync that working tree).
    """
    root = Path(repo)
    if gen < 1:
        raise ValueError("gen must be >= 1")
    if diff_sha256(diff) != cand.diff_sha:
        raise GitRefError(f"diff does not match candidate {cand.id}'s diff_sha")
    parent = champion_sha(root)
    if parent is None:
        raise GitRefError("no iterant/champion branch; call ensure_seed first")
    if gen_tag_sha(root, gen) is not None:
        raise GitRefError(f"tag {gen_tag(gen)} already exists")
    if CHAMPION_REF in _checked_out_branches(root):
        raise GitRefError(f"{CHAMPION_BRANCH} is checked out; refusing to move it")

    with tempfile.TemporaryDirectory(prefix="iterant-archive-") as tmp:
        env = {**_IDENTITY, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git(root, "read-tree", parent, env=env)
        applied = _git(
            root, "apply", "--cached", "--whitespace=nowarn", "-", input=diff, env=env, check=False
        )
        if applied.returncode != 0:
            err = applied.stderr.decode(errors="replace").strip()
            raise GitRefError(f"diff for candidate {cand.id} does not apply: {err}")
        tree = _git(root, "write-tree", env=env).stdout.decode().strip()
        message = (
            f"iterant gen {gen}: {cand.hypothesis}\n\n"
            f"Candidate: {cand.id}\nDiff-sha256: {cand.diff_sha}\n"
        )
        new = (
            _git(
                root, "commit-tree", tree, "-p", parent, "-F", "-", input=message.encode(), env=env
            )
            .stdout.decode()
            .strip()
        )
    _update_refs(root, new, parent, gen)
    return new
