"""The real ``ImageSource``: build the target image for a champion commit or a candidate diff."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from iterant.archive import GitRefError
from iterant.contracts import IterantConfig
from iterant.orchestrator import export_commit
from iterant.target import ImageBuildError, build_image, slugify

_STRIPPED_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")
_APPLY_TIMEOUT_S = 60


def _apply(diff: str, tree: Path) -> None:
    """Apply ``diff`` to the plain directory ``tree`` (no git repo needed)."""
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    try:
        proc = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "apply", "--whitespace=nowarn", "-"],
            cwd=tree,
            input=diff.encode("utf-8"),
            capture_output=True,
            env=env,
            timeout=_APPLY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ImageBuildError(f"could not run git apply: {exc}") from exc
    if proc.returncode != 0:
        raise ImageBuildError(f"diff does not apply: {proc.stderr.decode(errors='replace')[-500:]}")


class ContainerImages:
    """Builds with ``podman`` or ``docker build`` from a temporary export of the commit (the
    repo itself is never the build context). Images are cached by commit or candidate id.

    ``prefix`` namespaces the image tags (lowercase ``a-z0-9-``).
    """

    def __init__(
        self, repo: str | os.PathLike[str], config: IterantConfig, *, engine: str, prefix: str
    ) -> None:
        self._repo = Path(repo)
        self._config = config
        self._engine = engine
        self._prefix = slugify(prefix) or "iterant"
        self._cache: dict[str, str] = {}

    def _build(self, tag: str, tree: Path) -> str:
        return build_image(tree, self._config, tag, engine=self._engine).tag

    def champion(self, sha: str) -> str:
        if sha not in self._cache:
            with tempfile.TemporaryDirectory(prefix="iterant-image-") as tmp:
                try:
                    tree = export_commit(self._repo, sha, Path(tmp) / "tree")
                except GitRefError as exc:
                    raise ImageBuildError(str(exc)) from exc
                self._cache[sha] = self._build(f"{self._prefix}-champion-{sha[:12]}", tree)
        return self._cache[sha]

    def candidate(self, cand_id: str, parent_sha: str, diff: str) -> str:
        if cand_id not in self._cache:
            with tempfile.TemporaryDirectory(prefix="iterant-image-") as tmp:
                try:
                    tree = export_commit(self._repo, parent_sha, Path(tmp) / "tree")
                except GitRefError as exc:
                    raise ImageBuildError(str(exc)) from exc
                _apply(diff, tree)
                self._cache[cand_id] = self._build(f"{self._prefix}-{slugify(cand_id)}", tree)
        return self._cache[cand_id]
