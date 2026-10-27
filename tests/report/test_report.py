"""The evidence report, built from an archive populated by hand (no model, no sandbox)."""

from __future__ import annotations

import os
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
from iterant.report.model import MAX_DIFF_BYTES, adds_code_execution, collect
from iterant.stats.acceptance import decide_select

PACK_HASH = "ab" * 32
OTHER_HASH = "ef" * 32
IMAGE = "docker.io/library/python@sha256:" + "cd" * 32
XSS_HYP = "<script>alert('hyp')</script> make it \"better\""
XSS_DIFF = (
    "diff --git a/agent/p.txt b/agent/p.txt\n--- a/agent/p.txt\n+++ b/agent/p.txt\n"
    "@@ -1 +1,3 @@\n-old\n+<script>alert('diff')</script>\n+<img src=x onerror=alert(1)>\n"
    "+import subprocess\n"
)
PLAIN_DIFF = "--- a/p\n+++ b/p\n@@ -1 +1 @@\n-old\n+be concise\n"
N = 12
T0 = "2026-01-01T00:00:00+00:00"


def pack(
    policy: str = "allowed", name: str = "mini", version: str = "1", h: str = PACK_HASH
) -> Pack:
    return Pack(
        name=name,
        version=version,
        tasks=(),
        content_hash=h,
        grader_image=IMAGE,
        code_exec_policy=policy,  # type: ignore[arg-type]
    )


def runs(arc: Archive, batch: int, cand: str, split: Split, solved: int) -> None:
    for i in range(N):
        arc.insert_run(
            RunResult(
                RunKey(batch, cand, f"SECRET-{split.value}-{i}", 0),
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


def call(batch: int | None, cand: str | None, role: Role, usd: int, at: str) -> ProxyCall:
    key = None if batch is None or cand is None else RunKey(batch, cand, "SECRET-select-0", 0)
    return ProxyCall(0, key, role, "m", 100, 50, usd, False, at)


def seed(arc: Archive) -> None:
    arc.insert_candidate(
        Candidate("seed", None, diff_sha256(b""), "seed", Stage.PROMOTE, Status.PROMOTED, 0)
    )


def gen1(arc: Archive, store: DiffStore) -> None:
    """Batch 1 (gen 1): b1-0 promoted (hostile text), b1-1 rejected at SELECT."""
    batch = arc.create_batch(1, PACK_HASH, started_at=T0)
    arc.insert_candidate(
        Candidate("b1-0", "seed", store.put(XSS_DIFF), XSS_HYP, Stage.CONFIRM, Status.ACTIVE, 0)
    )
    arc.insert_candidate(
        Candidate(
            "b1-1", "seed", store.put(PLAIN_DIFF), "weak idea", Stage.SELECT, Status.ACTIVE, 0
        )
    )
    for split, champ, w in ((Split.SELECT, 2, 10), (Split.HOLDOUT, 2, 9)):
        runs(arc, batch.id, "seed", split, champ)
        runs(arc, batch.id, "b1-0", split, w)
    runs(arc, batch.id, "b1-1", Split.SELECT, 3)
    arc.finish_candidate("b1-1", Status.REJECTED, stage=Stage.SELECT)
    arc.record_promotion(Promotion(1, "b1-0", 0.6667, 0.0012, 0.5833, 0.0034))
    arc.insert_proxy_call(call(batch.id, "b1-0", Role.TARGET, 7000, "2026-01-01T00:00:05+00:00"))
    arc.insert_proxy_call(call(None, None, Role.PROPOSER, 25000, "2026-01-01T00:00:03+00:00"))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    with Archive.open(r) as arc:
        seed(arc)
        gen1(arc, DiffStore(r))
    return r


def db(repo: Path) -> Path:
    return repo / ".iterant" / "archive.db"


def test_promotion_and_rejection(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    index = render(db(repo), out, pack=pack())
    page = (out / "gen-1.html").read_text(encoding="utf-8")
    text = index.read_text(encoding="utf-8")
    assert not (out / "gen-2.html").exists()
    assert "gen-1.html" in text and "<code>seed</code>" in text
    # promoted: the recorded numbers only, no recomputed verdict
    assert "+0.6667" in page and "0.0012" in page and "+0.5833" in page and "0.0034" in page
    assert "recorded decision, as stored" in page
    assert "NOT the recorded decision" not in page
    assert "passes" not in text and "fails" not in text
    assert "$0.0320" in text  # 7000 + 25000 micro-USD
    assert PACK_HASH in text and "mini 1" in text
    assert "Possible code execution added" in page
    # rejected: recomputed numbers match a direct decide_select, labelled as not the decision
    with Archive.open_path(db(repo), create_schema=False) as arc:
        c = arc.runs_for_candidate("b1-1", split=Split.SELECT, batch=1, latest_only=False)
        h = arc.runs_for_candidate("seed", split=Split.SELECT, batch=1, latest_only=False)
    d = decide_select(c, h, seed=0)
    assert f"delta vs parent {d.delta:+.4f}, one-sided p {d.p:.4f}" in text
    assert "NOT the recorded decision" in text
    assert "weak idea" in text and "be concise" in text and "rejected at select" in text


def test_task_ids_and_checks_absent(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    render(db(repo), out)
    for f in out.glob("*.html"):
        assert "SECRET" not in f.read_text(encoding="utf-8")


def test_escaping(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    hostile = pack(name='x" onmouseover="alert(1)<script>', version="<b>1</b>")
    render(db(repo), out, pack=hostile)
    for name in ("index.html", "gen-1.html"):
        text = (out / name).read_text(encoding="utf-8")
        assert "<script" not in text and "<img" not in text and "<b>" not in text
        assert '" onmouseover=' not in text
        assert "&lt;script&gt;alert(&#x27;hyp&#x27;)" in text
        assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "&quot; onmouseover=&quot;alert(1)" in (out / "index.html").read_text(encoding="utf-8")


def test_banner_cases(repo: Path, tmp_path: Path) -> None:
    def text(sub: str, p: Pack | None) -> str:
        return render(db(repo), tmp_path / sub, pack=p).read_text(encoding="utf-8")

    adv = text("a", pack("advisory-forbidden"))
    assert "NOT enforced" in adv
    assert "NOT enforced" in (tmp_path / "a" / "gen-1.html").read_text(encoding="utf-8")
    ok = text("b", pack("allowed"))
    assert "NOT enforced" not in ok and "UNKNOWN" not in ok and "mini 1" in ok
    # no pack: a prominent warning, and no claim either way (fails closed)
    none = text("c", None)
    assert "UNKNOWN" in none and "NOT enforced" not in none and 'class="banner"' in none
    # a pack whose hash matches no batch is ignored: warning, name not applied
    bad = text("d", pack("advisory-forbidden", name="impostor", h=OTHER_HASH))
    assert "UNKNOWN" in bad and "impostor" not in bad and "NOT enforced" not in bad


def test_gen_level_pack_mismatch(tmp_path: Path) -> None:
    r = tmp_path / "repo"
    r.mkdir()
    with Archive.open(r) as arc:
        seed(arc)
        gen1(arc, DiffStore(r))
        b = arc.create_batch(2, OTHER_HASH, started_at="2026-01-02T00:00:00+00:00")
        arc.insert_candidate(
            Candidate("b2-0", "b1-0", diff_sha256(b""), "later", Stage.PROPOSE, Status.REJECTED, 0)
        )
        assert b.id == 2
    text = render(db(r), tmp_path / "out", pack=pack()).read_text(encoding="utf-8")
    assert text.count("PACK HASH MISMATCH") == 1


def test_read_only(repo: Path, tmp_path: Path) -> None:
    path = db(repo)
    before = (path.read_bytes(), path.stat().st_mtime_ns, sorted(os.listdir(path.parent)))
    render(path, tmp_path / "out")
    after = (path.read_bytes(), path.stat().st_mtime_ns, sorted(os.listdir(path.parent)))
    assert before == after
    assert not any(n.endswith(("-wal", "-shm")) for n in after[2])


def test_renders_beside_live_writer(repo: Path, tmp_path: Path) -> None:
    with Archive.open(repo) as arc:  # a live connection, WAL files present
        arc.insert_proxy_call(call(None, None, Role.ANALYST, 1, "2026-01-01T00:00:04+00:00"))
        index = render(db(repo), tmp_path / "out")
    assert "gen-1" in index.read_text(encoding="utf-8")


def test_multiple_gens_batches_and_spend_isolation(tmp_path: Path) -> None:
    r = tmp_path / "repo"
    r.mkdir()
    store = DiffStore(r)
    with Archive.open(r) as arc:
        seed(arc)
        arc.insert_proxy_call(call(None, None, Role.ANALYST, 111, "2025-12-31T23:00:00+00:00"))
        gen1(arc, store)  # proposer 25000 at 00:00:03, target 7000
        # gen 2 restarted: two batches share the number
        for bid, at in ((2, "2026-01-02T00:00:00+00:00"), (3, "2026-01-02T01:00:00+00:00")):
            b = arc.create_batch(2, PACK_HASH, started_at=at)
            assert b.id == bid
            arc.insert_candidate(
                Candidate(f"b{bid}-0", "b1-0", store.put(PLAIN_DIFF + f"+{bid}\n"), "h",
                          Stage.SELECT, Status.ACTIVE, 0)
            )  # fmt: skip
            runs(arc, bid, "b1-0", Split.SELECT, 5)
            runs(arc, bid, f"b{bid}-0", Split.SELECT, 5)
            arc.finish_candidate(f"b{bid}-0", Status.REJECTED, stage=Stage.SELECT)
            arc.insert_proxy_call(call(bid, f"b{bid}-0", Role.TARGET, 1000 * bid, at))
        arc.insert_proxy_call(call(None, None, Role.PROPOSER, 400, "2026-01-02T00:30:00+00:00"))
    out = tmp_path / "out"
    index = render(db(r), out, pack=pack())
    data = collect(db(r), pack=pack())
    by = {g.gen: g for g in data.generations}
    assert by[1].usd_micros == 7000 + 25000  # not gen 2's calls, not the pre-batch call
    assert by[2].usd_micros == 2000 + 3000 + 400
    assert by[2].batch_ids == (2, 3)
    assert data.total_usd_micros == 111 + 32000 + 5400
    assert data.unattributed_usd_micros == 111
    text = index.read_text(encoding="utf-8")
    assert "2 batches share this generation number" in text
    assert "in no generation" in text and "run spend only" in text


def test_missing_diff_store(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    render(db(repo), out, pack=pack(), diffs_dir=tmp_path / "nowhere")
    page = (out / "gen-1.html").read_text(encoding="utf-8")
    assert "diff of this promoted candidate is missing" in page
    assert "diff unavailable" in page


def test_diff_read_is_capped(tmp_path: Path) -> None:
    r = tmp_path / "repo"
    r.mkdir()
    big = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n" + "+x\n" * (MAX_DIFF_BYTES // 3 + 1000)
    with Archive.open(r) as arc:
        seed(arc)
        arc.create_batch(1, PACK_HASH, started_at=T0)
        arc.insert_candidate(
            Candidate("b1-0", "seed", DiffStore(r).put(big), "big", Stage.GATE, Status.ACTIVE, 0)
        )
    data = collect(db(r))
    c = data.generations[0].candidates[0]
    assert "cut" in c.diff_note and len(c.diff) <= MAX_DIFF_BYTES


def test_stale_out_dir(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "gen-9.html").write_text("stale", encoding="utf-8")
    (out / "notes.txt").write_text("keep", encoding="utf-8")
    render(db(repo), out)
    assert not (out / "gen-9.html").exists()
    assert (out / "notes.txt").exists() and (out / "gen-1.html").exists()


def test_empty_archive(tmp_path: Path) -> None:
    with Archive.open(tmp_path):
        pass
    index = render(db(tmp_path), tmp_path / "out")
    assert "Iterant evidence report" in index.read_text(encoding="utf-8")


def test_exec_heuristic() -> None:
    assert adds_code_execution("+import subprocess\n")
    assert adds_code_execution("+  r = eval(x)\n")
    for snippet in (
        "asyncio.create_subprocess_exec(x)",
        "cp.execSync(x)",
        "cp.execFile(x)",
        "__import__('os')",
        "importlib.import_module(n)",
        "import ctypes",
    ):
        assert adds_code_execution(f"+{snippet}\n"), snippet
    assert not adds_code_execution("-import subprocess\n")  # removed, not added
    header = "--- a/subprocess.py\n+++ b/subprocess.py\n+be concise\n"
    assert not adds_code_execution(header)  # a real file header is skipped
    assert adds_code_execution("+++ subprocess.run(x)\n")  # no `--- ` before it: an added line
    assert adds_code_execution("--- a/f\n+++ b/f\n++ os.system('x')\n")
