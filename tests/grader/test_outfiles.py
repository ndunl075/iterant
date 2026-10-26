from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from iterant.grader import OutputRejected, collect_out_files
from iterant.grader import outfiles as of


def test_reads_regular_files(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.txt").write_bytes(b"x")
    (tmp_path / "y.json").write_bytes(b"{}")
    assert collect_out_files(tmp_path) == {"a/b/x.txt": b"x", "y.json": b"{}"}


def test_symlink_rejected(tmp_path: Path) -> None:
    (tmp_path / "t").write_bytes(b"x")
    try:
        (tmp_path / "link").symlink_to(tmp_path / "t")
    except OSError:
        pytest.skip("symlinks not available")
    with pytest.raises(OutputRejected, match="link"):
        collect_out_files(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_junction_rejected(tmp_path: Path) -> None:
    out, target = tmp_path / "out", tmp_path / "elsewhere"
    out.mkdir()
    target.mkdir()
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(out / "jn"), str(target)], capture_output=True
    )
    if made.returncode != 0:
        pytest.skip("cannot create junction")
    try:
        with pytest.raises(OutputRejected, match="link"):
            collect_out_files(out)
    finally:
        (out / "jn").rmdir()  # remove the link, never the target's contents


def test_hard_link_rejected(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"x")
    try:
        os.link(tmp_path / "a", tmp_path / "b")
    except OSError:
        pytest.skip("hard links not available")
    with pytest.raises(OutputRejected, match="hard link"):
        collect_out_files(tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs")
def test_special_file_rejected(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(OutputRejected, match="special"):
        collect_out_files(tmp_path)


def test_file_count_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_FILES", 3)
    for i in range(3):
        (tmp_path / f"f{i}").write_bytes(b"")
    assert len(collect_out_files(tmp_path)) == 3
    (tmp_path / "f3").write_bytes(b"")
    with pytest.raises(OutputRejected, match="too many"):
        collect_out_files(tmp_path)
    with pytest.raises(OutputRejected, match="too many"):
        of.check_limits({f"f{i}": b"" for i in range(4)})


def test_total_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_TOTAL_BYTES", 10)
    (tmp_path / "a").write_bytes(b"123456")
    (tmp_path / "b").write_bytes(b"123456")
    with pytest.raises(OutputRejected, match="too large"):
        collect_out_files(tmp_path)
    with pytest.raises(OutputRejected, match="too large"):
        of.check_limits({"a": b"123456", "b": b"123456"})


def test_per_file_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_FILE_BYTES", 4)
    (tmp_path / "a").write_bytes(b"12345")
    with pytest.raises(OutputRejected, match="file too large"):
        collect_out_files(tmp_path)
    with pytest.raises(OutputRejected, match="file too large"):
        of.check_limits({"a": b"12345"})


def test_path_depth_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_PATH_DEPTH", 3)
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "ok").write_bytes(b"")
    assert collect_out_files(tmp_path) == {"a/b/ok": b""}
    (tmp_path / "a" / "b" / "c").mkdir()
    (tmp_path / "a" / "b" / "c" / "deep").write_bytes(b"")
    with pytest.raises(OutputRejected, match="too deep"):
        collect_out_files(tmp_path)
    with pytest.raises(OutputRejected, match="too deep"):
        of.check_limits({"a/b/c/d": b""})


def test_rejection_messages_hide_names(tmp_path: Path) -> None:
    (tmp_path / "SECRET-NAME").write_bytes(b"x")
    os.link(tmp_path / "SECRET-NAME", tmp_path / "SECRET-LINK")
    with pytest.raises(OutputRejected) as exc:
        collect_out_files(tmp_path)
    assert "SECRET" not in str(exc.value)


def test_entry_cap_counts_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_ENTRIES", 5)
    for i in range(6):
        (tmp_path / f"d{i}").mkdir()
    with pytest.raises(OutputRejected, match="too many"):
        collect_out_files(tmp_path)


def test_growth_after_fstat_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(of, "MAX_OUT_FILE_BYTES", 4)
    (tmp_path / "a").write_bytes(b"123456789")
    real = os.fstat

    def lying(fd: int) -> os.stat_result:
        st = list(real(fd))
        st[6] = 0  # st_size: the file "grew" after this was taken
        return os.stat_result(st)

    monkeypatch.setattr(of.os, "fstat", lying)
    with pytest.raises(OutputRejected, match="file too large"):
        collect_out_files(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="O_NOFOLLOW is POSIX-only")
def test_symlink_swapped_in_after_lstat_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out, secret = tmp_path / "out", tmp_path / "secret"
    out.mkdir()
    secret.write_bytes(b"SECRET")
    (out / "a").write_bytes(b"x")
    real = os.stat

    def swapping(path: str, *args: object, **kwargs: object) -> os.stat_result:
        st = real(path, *args, **kwargs)  # type: ignore[arg-type]
        if path == "a":
            (out / "a").unlink()
            (out / "a").symlink_to(secret)
        return st

    monkeypatch.setattr(of.os, "stat", swapping)
    with pytest.raises(OutputRejected, match="changed"):
        collect_out_files(out)
