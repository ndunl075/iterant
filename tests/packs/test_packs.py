import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from iterant.contracts import AcceptanceThresholds, Split
from iterant.packs import (
    PackError,
    PackHashMismatch,
    assign_splits,
    compute_content_hash,
    load_pack,
    split_pack,
    tasks_in,
    verify,
)

REPO = Path(__file__).resolve().parents[2]
# Pinned: fails if any byte of the starter pack changes (line endings included).
# Update on purpose only, when the pack is regenerated.
PINNED_FIXTURE_HASH = "d514f87bddc7a99b07f53325349f7f3cf269ede24654d6ba051732de27646b84"


def make_pack(root: Path, n: int = 12, tags: tuple[str, ...] = ("a", "b")) -> Path:
    root.mkdir(parents=True)
    (root / "pack.toml").write_text('name = "demo"\nversion = "1"\n', encoding="utf-8")
    for i in range(n):
        d = root / "tasks" / f"t{i:03d}"
        d.mkdir(parents=True)
        (d / "input.json").write_text(json.dumps({"q": i}), encoding="utf-8")
        (d / "check").write_text("exit 0\n", encoding="utf-8")
        (d / "task.toml").write_text(f'tags = ["{tags[i % len(tags)]}"]\n', encoding="utf-8")
    split_pack(root, seed=1)
    return root


def test_load_roundtrip(tmp_path: Path) -> None:
    pack = load_pack(make_pack(tmp_path / "p"))
    assert (pack.name, pack.version, len(pack.tasks)) == ("demo", "1", 12)
    t = pack.tasks[0]
    assert str(t.input) == "tasks/t000/input.json" and str(t.check) == "tasks/t000/check"
    assert t.tags in (("a",), ("b",))
    assert sum(len(tasks_in(pack, s)) for s in Split) == 12


@pytest.mark.parametrize(
    "victim",
    ["tasks/t003/input.json", "tasks/t003/check", "tasks/t003/task.toml", "pack.toml", "NOTICE"],
)
def test_hash_changes_with_any_file(tmp_path: Path, victim: str) -> None:
    root = make_pack(tmp_path / "p")
    (root / "NOTICE").write_bytes(b"n")
    before = compute_content_hash(root)
    with (root / victim).open("ab") as f:
        f.write(b" ")
    assert compute_content_hash(root) != before


def test_hash_changes_with_split_assignment(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p")
    before = compute_content_hash(root)
    data = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    ids = sorted(data["assignment"])
    a = ids[0]
    b = next(i for i in ids if data["assignment"][i] != data["assignment"][a])
    data["assignment"][a], data["assignment"][b] = data["assignment"][b], data["assignment"][a]
    (root / "splits.json").write_text(json.dumps(data), encoding="utf-8")
    assert compute_content_hash(root) != before
    assert load_pack(root).content_hash != before


def test_hash_path_data_boundary_is_unambiguous(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
    (a / "x").write_bytes(b"yz")
    (b / "xy").write_bytes(b"z")
    assert compute_content_hash(a) != compute_content_hash(b)


def test_hash_ignores_location(tmp_path: Path) -> None:
    one = make_pack(tmp_path / "one")
    two = tmp_path / "elsewhere" / "two"
    two.parent.mkdir()
    shutil.copytree(one, two)
    assert compute_content_hash(one) == compute_content_hash(two)


def test_verify(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p")
    pack = load_pack(root)
    verify(pack, pack.content_hash)
    verify(pack, pack.content_hash, root)
    with pytest.raises(PackHashMismatch):
        verify(pack, "0" * 64)
    (root / "tasks" / "t000" / "check").write_text("exit 1\n", encoding="utf-8")
    verify(pack, pack.content_hash)  # the in-memory pack is stale on purpose...
    with pytest.raises(PackHashMismatch):  # ...so the on-disk re-check is what catches it
        verify(pack, pack.content_hash, root)


def test_symlink_rejected(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p")
    try:
        (root / "link").symlink_to(root / "pack.toml")
    except OSError:
        pytest.skip("symlinks not available")
    with pytest.raises(PackError, match="symlink"):
        compute_content_hash(root)


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-only")
def test_junction_rejected(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p")
    target = tmp_path / "elsewhere"
    target.mkdir()
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root / "jn"), str(target)], capture_output=True
    )
    if made.returncode != 0:
        pytest.skip("cannot create junction")
    try:
        with pytest.raises(PackError, match="junction"):
            compute_content_hash(root)
    finally:
        (root / "jn").rmdir()  # remove the link, never the target's contents


def test_load_errors(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p")
    (root / "tasks" / "t000" / "check").unlink()
    with pytest.raises(PackError, match="missing check"):
        load_pack(root)
    root = make_pack(tmp_path / "q")
    (root / "splits.json").unlink()
    with pytest.raises(PackError, match="splits.json"):
        load_pack(root)
    root = make_pack(tmp_path / "r")
    (root / "tasks" / "extra").mkdir()
    (root / "tasks" / "extra" / "input.json").write_bytes(b"{}")
    (root / "tasks" / "extra" / "check").write_bytes(b"")
    with pytest.raises(PackError, match="does not list exactly"):
        load_pack(root)
    with pytest.raises(PackError, match="pack.toml"):
        load_pack(tmp_path / "nowhere")


def tagged(n: int, tags: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    return [(f"id{i:04d}", (tags[i % len(tags)],)) for i in range(n)]


def test_splitter_is_deterministic_and_seeded() -> None:
    tasks = tagged(100, ["x", "y", "z"])
    a = assign_splits(tasks, seed=7).assignment
    assert a == assign_splits(list(reversed(tasks)), seed=7).assignment
    assert a != assign_splits(tasks, seed=8).assignment


def test_splitter_honors_sizes_when_big_enough() -> None:
    t = AcceptanceThresholds()
    plan = assign_splits(tagged(300, ["x", "y", "z"]), t, seed=0)
    c = Counter(plan.assignment.values())
    assert c[Split.SELECT] == t.select_tasks and c[Split.HOLDOUT] == t.holdout_tasks
    assert c[Split.TEST] == 45 and c[Split.TRAIN] == 300 - 80 - 40 - 45
    assert plan.report.honored and not plan.report.warnings


def test_splitter_reports_when_pack_too_small() -> None:
    plan = assign_splits(tagged(20, ["x", "y"]), AcceptanceThresholds(), seed=0)
    assert not plan.report.honored and plan.report.warnings
    assert plan.report.requested_select == 80
    c = Counter(plan.assignment.values())
    assert all(c[s] >= 1 for s in Split) and sum(c.values()) == 20


def test_splitter_honors_custom_sizes() -> None:
    t = AcceptanceThresholds(select_tasks=10, holdout_tasks=5, smoke_tasks=2)
    plan = assign_splits(tagged(60, ["x", "y"]), t, seed=3)
    assert plan.report.honored
    assert Counter(plan.assignment.values())[Split.SELECT] == 10


def test_splitter_stratifies_by_tags() -> None:
    tasks = [(f"a{i:03d}", ("a",)) for i in range(200)]
    tasks += [(f"b{i:03d}", ("b",)) for i in range(100)]
    plan = assign_splits(tasks, AcceptanceThresholds(select_tasks=60, holdout_tasks=30), seed=2)
    for split in Split:
        in_split = [i for i, s in plan.assignment.items() if s is split]
        share_a = sum(i.startswith("a") for i in in_split) / len(in_split)
        assert abs(share_a - 2 / 3) < 0.08, (split, share_a)


def test_splitter_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least"):
        assign_splits(tagged(3, ["x"]))
    with pytest.raises(ValueError, match="unique"):
        assign_splits([("a", ()), ("a", ()), ("b", ()), ("c", ()), ("d", ())])


def test_split_pack_persists_and_is_pinned_by_hash(tmp_path: Path) -> None:
    root = make_pack(tmp_path / "p", n=30)
    h = load_pack(root).content_hash
    split_pack(root, seed=1)  # same seed: identical file, identical hash
    assert load_pack(root).content_hash == h
    split_pack(root, seed=2)
    assert load_pack(root).content_hash != h


def test_starter_fixture_pack() -> None:
    root = REPO / "packs" / "bigcodebench" / "pack"
    pack = load_pack(root)
    assert pack.name == "bigcodebench-mini" and len(pack.tasks) == 20
    assert {t.split for t in pack.tasks} == set(Split)
    assert (root / "LICENSE.upstream").is_file() and (root / "NOTICE").is_file()
    assert pack.content_hash == PINNED_FIXTURE_HASH
