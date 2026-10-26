from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from iterant.target import Genome, PathRejectedError, genome_for, parse_config
from iterant.target.genome import compile_glob, normalize_relative

PATTERNS = ["prompts/**", "tools/**", "agent/loop.py", "docs/*.md", "a/**/z.txt"]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel in [
        "prompts/system.md",
        "prompts/deep/er/x.md",
        "tools/calc.py",
        "agent/loop.py",
        "agent/__main__.py",
        "docs/a.md",
        "docs/sub/b.md",
        "a/z.txt",
        "a/b/c/z.txt",
        "evals/pack/check.py",
        "iterant.toml",
        "Containerfile",
        ".git/config",
        ".iterant/archive.db",
        "secret.txt",
    ]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    return tmp_path


def _genome(repo: Path, *, ci: bool = False, protected: tuple[str, ...] = ()) -> Genome:
    return Genome(repo, PATTERNS, protected=protected, case_insensitive=ci)


@pytest.mark.parametrize(
    "path",
    [
        "prompts/system.md",
        "prompts/new.md",  # new file, not on disk
        "prompts/deep/er/x.md",
        "prompts/a/b/c/d/e.md",
        "tools/calc.py",
        "agent/loop.py",
        "./agent/loop.py",
        "./prompts/system.md",
        "prompts/./system.md",
        "././tools/calc.py",
        "docs/a.md",
        "a/z.txt",  # ** spans zero segments
        "a/b/c/z.txt",
    ],
)
def test_editable(repo: Path, path: str) -> None:
    d = _genome(repo).check(path)
    assert d.ok, d.reason


@pytest.mark.parametrize(
    ("path", "why"),
    [
        ("agent/__main__.py", "outside the genome"),
        ("agent/loop.pyc", "outside the genome"),
        ("agent/loop.py/x", "outside the genome"),
        ("prompts", "outside the genome"),  # dir/** needs something below
        ("docs/sub/b.md", "outside the genome"),  # * stays in one segment
        ("docs/a.txt", "outside the genome"),
        ("secret.txt", "outside the genome"),
        ("iterant.toml", "outside the genome"),
        ("a/z.txt.bak", "outside the genome"),
        ("PROMPTS/system.md", "outside the genome"),  # case-sensitive mode
        ("../prompts/system.md", ".."),
        ("prompts/../secret.txt", ".."),
        ("prompts/../../x", ".."),
        ("..", ".."),
        ("/prompts/system.md", "absolute"),
        ("/etc/passwd", "absolute"),
        ("C:/prompts/system.md", ":"),
        ("C:prompts/system.md", ":"),
        ("prompts/system.md:stream", ":"),
        ("prompts\\system.md", "backslash"),
        ("..\\x", "backslash"),
        ("\\\\host\\share\\x", "backslash"),
        ("prompts/", "ends with '/'"),
        ("prompts/deep/", "ends with '/'"),
        ("prompts//system.md", "empty segment"),
        ("", "empty"),
        (".", "repo root"),
        ("./", "ends with '/'"),
        ("prompts/system.md\x00.png", "control"),
        ("prompts/sys\nx.md", "control"),
        ("prompts/x.md.", "ending in '.'"),
        ("prompts/x.md ", "ending in '.' or space"),
        (".git/config", "reserved"),
        ("prompts/.git/hooks/x", "reserved"),
        ("tools/.git", "reserved"),
        (".iterant/archive.db", "reserved"),
        ("prompts/.iterant/x", "reserved"),
        (".GIT/config", "reserved"),
        (".Iterant/x", "reserved"),
        ("GIT~1/config", "reserved"),
        ("prompts/ITERANT~1/x", "reserved"),
        (".git./config", "ending in '.'"),
    ],
)
def test_rejected(repo: Path, path: str, why: str) -> None:
    d = _genome(repo).check(path)
    assert not d.ok
    assert why in d.reason, d.reason


def test_case_insensitive_mode_matches_windows_semantics(repo: Path) -> None:
    g = _genome(repo, ci=True)
    assert g.is_editable("PROMPTS/System.MD")
    assert g.is_editable("Agent/Loop.py")
    assert not g.is_editable("AGENT/__main__.py")
    assert not g.is_editable(".GIT/config")


def test_case_sensitive_mode(repo: Path) -> None:
    assert not _genome(repo, ci=False).is_editable("Prompts/system.md")


def test_protected_paths_beat_globs(repo: Path) -> None:
    g = Genome(
        repo,
        ["**"],
        protected=("iterant.toml", "evals/pack", "Containerfile"),
        case_insensitive=False,
    )
    assert g.is_editable("prompts/system.md")
    assert g.is_editable("secret.txt")
    for p in ["iterant.toml", "evals/pack/check.py", "evals/pack/new/x", "Containerfile"]:
        d = g.check(p)
        assert not d.ok and "protected" in d.reason
    assert not g.is_editable(".git/config")
    assert g.is_editable("evals/other.py")  # sibling with shared prefix is not protected
    assert g.is_editable("evals/packs/x.py")  # 'pack' != 'packs'


def test_protected_is_case_folded_when_ci(repo: Path) -> None:
    g = Genome(repo, ["**"], protected=("iterant.toml",), case_insensitive=True)
    assert not g.is_editable("ITERANT.toml")
    assert not g.is_editable("./Iterant.Toml")


def test_genome_for_protects_config_image_pack(repo: Path) -> None:
    cfg = parse_config(
        """
[target]
genome = ["**"]
image = "Containerfile"
run = "x"
[target.model]
name = "m"
effort = "low"
max_tokens_per_run = 5
[eval]
pack = "evals/pack"
"""
    )
    g = genome_for(repo, cfg)
    assert not g.is_editable("iterant.toml")
    assert not g.is_editable("Containerfile")
    assert not g.is_editable("evals/pack/check.py")
    assert g.is_editable("prompts/system.md")


def test_violations_lists_only_bad_paths(repo: Path) -> None:
    g = _genome(repo)
    bad = g.violations(["prompts/a.md", "../x", "secret.txt", "tools/t.py", ".git/config"])
    assert [d.path for d in bad] == ["../x", "secret.txt", ".git/config"]
    assert g.violations([]) == []


def test_rename_needs_both_sides_checked(repo: Path) -> None:
    g = _genome(repo)
    assert [d.path for d in g.violations(["prompts/a.md", "secret.txt"])] == ["secret.txt"]


def _symlink(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")


def test_symlink_file_to_outside_repo(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside") / "evil.txt"
    outside.write_text("x", encoding="utf-8")
    _symlink(repo / "prompts" / "link.md", outside, directory=False)
    d = _genome(repo).check("prompts/link.md")
    assert not d.ok and "outside the repo" in d.reason


def test_symlink_file_to_non_genome_file_in_repo(repo: Path) -> None:
    _symlink(repo / "prompts" / "link.md", repo / "secret.txt", directory=False)
    d = _genome(repo).check("prompts/link.md")
    assert not d.ok and "outside the genome" in d.reason


def test_symlink_file_to_checks(repo: Path) -> None:
    _symlink(repo / "tools" / "c.py", repo / "evals" / "pack" / "check.py", directory=False)
    g = Genome(repo, ["tools/**"], protected=("evals/pack",), case_insensitive=False)
    d = g.check("tools/c.py")
    assert not d.ok and "protected" in d.reason


def test_symlink_to_git_dir(repo: Path) -> None:
    _symlink(repo / "prompts" / "g", repo / ".git", directory=True)
    d = _genome(repo).check("prompts/g/config")
    assert not d.ok and "reserved" in d.reason


def test_symlinked_dir_to_outside_repo(
    repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "f.md").write_text("x", encoding="utf-8")
    _symlink(repo / "prompts" / "out", outside, directory=True)
    assert not _genome(repo).is_editable("prompts/out/f.md")
    assert not _genome(repo).is_editable("prompts/out/new.md")  # file does not exist yet


def test_symlink_within_genome_is_fine(repo: Path) -> None:
    _symlink(repo / "prompts" / "alias.md", repo / "prompts" / "system.md", directory=False)
    assert _genome(repo).is_editable("prompts/alias.md")


def test_symlink_loop_is_rejected_or_harmless(repo: Path) -> None:
    _symlink(repo / "prompts" / "loop", repo / "prompts" / "loop", directory=True)
    d = _genome(repo).check("prompts/loop/x.md")
    assert isinstance(d.ok, bool)  # must not hang or raise


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-only")
def test_junction_to_outside_repo(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    junction = repo / "prompts" / "junc"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
    )
    try:
        d = _genome(repo, ci=True).check("prompts/junc/f.md")
        assert not d.ok and "outside the repo" in d.reason
    finally:
        junction.rmdir()  # remove the link, never its target


# --- glob compiler --------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("*", "a", True),
        ("*", "a/b", False),
        ("*.py", ".py", True),
        ("*.py", "a.py", True),
        ("?.py", "ab.py", False),
        ("?.py", "a.py", True),
        ("**", "a", True),
        ("**", "a/b/c", True),
        ("d/**", "d", False),
        ("d/**", "d/x", True),
        ("d/**", "d/x/y", True),
        ("d/**", "dd/x", False),
        ("**/x.py", "x.py", True),
        ("**/x.py", "a/b/x.py", True),
        ("**/x.py", "ax.py", False),
        ("a/**/b", "a/b", True),
        ("a/**/b", "a/x/y/b", True),
        ("a/**/b", "a/xb", False),
        ("./d/**", "d/x", True),
        ("a.b", "aXb", False),  # '.' is literal
        ("a+b", "a+b", True),
        ("(x)", "(x)", True),
    ],
)
def test_glob_semantics(pattern: str, path: str, expected: bool) -> None:
    assert bool(compile_glob(pattern).fullmatch(path)) is expected


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        "/abs",
        "../x",
        "a/../b",
        "a/",
        "a//b",
        "a\\b",
        "[ab]",
        "{a,b}",
        "!x",
        "a**",
        "**b",
        "a**/b",
        ".git/**",
        "x/.iterant/*",
        "C:/x",
        ".",
    ],
)
def test_bad_globs_rejected(pattern: str) -> None:
    with pytest.raises(PathRejectedError):
        compile_glob(pattern)


def test_empty_genome_rejected(tmp_path: Path) -> None:
    with pytest.raises(PathRejectedError):
        Genome(tmp_path, [])


def test_normalize_relative() -> None:
    assert normalize_relative("./a/./b") == "a/b"
    with pytest.raises(PathRejectedError):
        normalize_relative("a/*")
