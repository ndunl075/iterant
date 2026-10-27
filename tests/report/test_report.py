"""The evidence report, built from an archive populated by hand (no model, no sandbox)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iterant.archive import Archive, diff_sha256
from iterant.contracts import (
    Candidate,
    Pack,
    Promotion,
    ProxyCall,
    Role,
    RunKey,
    RunResult,
    Split,
    Stage,
    Status,
)
from iterant.orchestrator.workspace import DiffStore
from iterant.report import render
from iterant.report.model import adds_code_execution

PACK_HASH = "ab" * 32
IMAGE = "docker.io/library/python@sha256:" + "cd" * 32
XSS_HYP = "<script>alert('hyp')</script> make it \"better\""
XSS_DIFF = (
    "diff --git a/agent/p.txt b/agent/p.txt\n--- a/agent/p.txt\n+++ b/agent/p.txt\n"
    "@@ -1 +1,3 @@\n-old\n+<script>alert('diff')</script>\n+<img src=x onerror=alert(1)>\n"
    "+import subprocess\n"
)
PLAIN_DIFF = "--- a/p\n+++ b/p\n@@ -1 +1 @@\n-old\n+be concise\n"
N = 12


def pack(policy: str) -> Pack:
    return Pack(
        name="<b>mini</b>",
        version="1",
        tasks=(),
        content_hash=PACK_HASH,
        grader_image=IMAGE,
        code_exec_policy=policy,  # type: ignore[arg-type]
    )


def runs(arc: Archive, batch: int, cand: str, split: Split, solved: int) -> None:
    for i in range(N):
        arc.insert_run(
            RunResult(
                RunKey(batch, cand, f"{split.value}{i}", 0),
                split,
                1.0 if i < solved else 0.0,
                100,
                50,
                1000,
                1.0,
                False,
                False,
            )
        )


@pytest.fixture
def archive_file(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    store = DiffStore(repo)
    with Archive.open(repo) as arc:
        sha0 = diff_sha256(b"")
        arc.insert_candidate(
            Candidate("seed", None, sha0, "seed champion", Stage.PROMOTE, Status.PROMOTED, 0)
        )
        batch = arc.create_batch(1, PACK_HASH, started_at="2026-01-01T00:00:00+00:00")
        win = Candidate(
            "b1-0", "seed", store.put(XSS_DIFF), XSS_HYP, Stage.CONFIRM, Status.ACTIVE, 0
        )
        lose = Candidate(
            "b1-1", "seed", store.put(PLAIN_DIFF), "weak idea", Stage.SELECT, Status.ACTIVE, 0
        )
        arc.insert_candidate(win)
        arc.insert_candidate(lose)
        for split, champ, w, lo in ((Split.SELECT, 2, 10, 2), (Split.HOLDOUT, 2, 9, 2)):
            runs(arc, batch.id, "seed", split, champ)
            runs(arc, batch.id, "b1-0", split, w)
            if split is Split.SELECT:
                runs(arc, batch.id, "b1-1", split, lo)
        arc.finish_candidate("b1-1", Status.REJECTED, stage=Stage.SELECT)
        arc.record_promotion(Promotion(1, "b1-0", 0.6667, 0.0012, 0.5833, 0.0034))
        arc.insert_proxy_call(
            ProxyCall(0, RunKey(batch.id, "b1-0", "select0", 0), Role.TARGET, "m", 100, 50, 7000,
                      False, "2026-01-01T00:00:05+00:00")
        )  # fmt: skip
        arc.insert_proxy_call(
            ProxyCall(0, None, Role.PROPOSER, "m", 900, 400, 25000, False,
                      "2026-01-01T00:00:03+00:00")
        )  # fmt: skip
    return repo / ".iterant" / "archive.db"


def test_promotion_and_rejection(archive_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    index = render(archive_file, out)
    assert index == out / "index.html"
    page = (out / "gen-1.html").read_text(encoding="utf-8")
    text = index.read_text(encoding="utf-8")
    assert not (out / "gen-2.html").exists()
    # lineage, evidence as stored, thresholds, spend, pack hash
    assert "seed" in text and "gen-1.html" in text
    assert "+0.6667" in page and "0.0012" in page and "+0.5833" in page and "0.0034" in page
    assert "as stored in the promotions row" in page
    assert "delta &ge; 0.05" in page
    assert "$0.0320" in text  # 7000 + 25000 micro-USD
    assert PACK_HASH in text and "name not recorded" in text
    # the rejected candidate is listed, with its recomputed evidence and reason
    assert "b1-1" in text and "rejected at select" in text
    assert "delta 0.0000" in text or "recomputed from archived runs" in text
    assert "weak idea" in text and "be concise" in text
    # the promoted diff adds code execution (heuristic flag)
    assert "Possible code execution added" in page


def test_escaping(archive_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    render(archive_file, out, pack=pack("allowed"))
    for name in ("index.html", "gen-1.html"):
        text = (out / name).read_text(encoding="utf-8")
        assert "<script" not in text
        assert "<img" not in text
        assert "&lt;script&gt;alert(&#x27;hyp&#x27;)" in text
        assert "&lt;img src=x onerror=alert(1)&gt;" in text
        assert "<b>mini</b>" not in text and "&lt;b&gt;mini&lt;/b&gt;" in text


def test_advisory_banner(archive_file: Path, tmp_path: Path) -> None:
    adv = render(archive_file, tmp_path / "a", pack=pack("advisory-forbidden"))
    assert "NOT enforced" in adv.read_text(encoding="utf-8")
    assert "NOT enforced" in (tmp_path / "a" / "gen-1.html").read_text(encoding="utf-8")
    ok = render(archive_file, tmp_path / "b", pack=pack("allowed"))
    assert "NOT enforced" not in ok.read_text(encoding="utf-8")
    assert "mini" in ok.read_text(encoding="utf-8")  # pack name shown when the hash matches


def test_empty_archive(tmp_path: Path) -> None:
    with Archive.open(tmp_path):
        pass
    index = render(tmp_path / ".iterant" / "archive.db", tmp_path / "out")
    assert "Iterant evidence report" in index.read_text(encoding="utf-8")


def test_exec_heuristic() -> None:
    assert adds_code_execution("+import subprocess\n")
    assert adds_code_execution("+  r = eval(x)\n")
    assert not adds_code_execution("-import subprocess\n")  # removed, not added
    assert not adds_code_execution("+++ b/subprocess.py\n+be concise\n")
