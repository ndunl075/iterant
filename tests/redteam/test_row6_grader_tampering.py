"""Row 6: "Tamper with the grader (patch its test framework, read its memory, plant symlinks or
huge files in /out)". Stopped by invariant 8: agent output is plain data to the grader.

The real ``Grader`` and the real ``/out`` reader run against hostile output. The sandbox is a
scripted fake that prints whatever the "agent's code" would print, so every malformed result a
hostile check script can emit is tried. Anything the agent controls must score 0.0, never
raise, and never be mistaken for an infra error.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from iterant.contracts import ExecResult, NetMode, Sandbox, Task
from iterant.grader import Grader, OutputRejected
from iterant.grader.grader import MAX_RESULT_BYTES, RUN_PATH
from iterant.grader.outfiles import (
    MAX_OUT_FILE_BYTES,
    MAX_OUT_FILES,
    MAX_OUT_PATH_DEPTH,
    collect_out_files,
)
from iterant.packs import load_pack
from iterant.packs.hashing import PackError, PackHashMismatch
from tests.orchestrator.conftest import GRADER_IMAGE, World

# --- a scripted grader backend ----------


@dataclass
class Box:
    on_run: Any
    net: NetMode
    image: str
    files: dict[str, bytes] = field(default_factory=lambda: {})
    cmds: list[str] = field(default_factory=lambda: [])
    destroyed: int = 0

    def put(self, path: str, data: bytes) -> None:
        self.files[path] = data

    def get(self, path: str) -> bytes:
        raise FileNotFoundError(path)

    def exec(self, cmd: str, timeout: float) -> ExecResult:
        self.cmds.append(cmd)
        if cmd == "python3 -V":
            return ExecResult(0, "Python 3.12.0", "", False, 0.1)
        if cmd.startswith("mkdir -p"):
            return ExecResult(0, "", "", False, 0.1)
        return self.on_run(self)

    def destroy(self) -> None:
        self.destroyed += 1


class Backend:
    def __init__(self, on_run: Any) -> None:
        self.on_run = on_run
        self.boxes: list[Box] = []

    def create(
        self, image: str, cpu: float, mem: int, disk: int, ttl: float, net: NetMode = "proxy-only"
    ) -> Sandbox:
        box = Box(self.on_run, net, image)
        self.boxes.append(box)
        return box  # type: ignore[return-value]


def says(stdout: str, **kw: Any) -> Any:
    return lambda _box: ExecResult(
        kw.pop("code", 0), stdout, "", kw.pop("timed_out", False), 0.1, **kw
    )


@pytest.fixture(scope="module")
def shared_world(tmp_path_factory: pytest.TempPathFactory) -> World:
    """One read-only world for the many tests that only grade (building one costs a git init)."""
    return World(tmp_path_factory.mktemp("row6"))


@pytest.fixture
def pack_world(shared_world: World) -> World:
    return shared_world


@pytest.fixture
def own_world(tmp_path: Path) -> World:
    """A private world for tests that damage the pack on disk."""
    return World(tmp_path)


def task_of(w: World, tid: str = "se00") -> Task:
    return next(t for t in w.pack.tasks if t.id == tid)


def grade(w: World, on_run: Any, files: dict[str, bytes] | None = None) -> tuple[Any, Backend]:
    backend = Backend(on_run)
    grader = Grader(backend, w.pack, w.pack_root, timeout=5)
    res = grader.grade(task_of(w), files if files is not None else {"solve.sh": b"x"})
    return res, backend


EXPECTED = '{"v": 1}'


def test_row6_control_the_right_object_scores_one(pack_world: World) -> None:
    res, backend = grade(pack_world, says(EXPECTED))
    assert (res.score, res.error) == (1.0, None)
    assert [b.net for b in backend.boxes] == ["default-deny"]
    assert backend.boxes[0].image == GRADER_IMAGE


@pytest.mark.parametrize(
    "stdout",
    [
        # non-finite numbers, in every spelling the stdlib parser accepts
        "NaN",
        "Infinity",
        "-Infinity",
        '{"v": NaN}',
        '{"v": Infinity}',
        '{"v": -Infinity}',
        '{"v": 1e999}',
        '{"v": 1, "x": NaN}',
        # right value, wrong type or shape
        '{"v": true}',
        '{"v": "1"}',
        '{"v": [1]}',
        '{"v": 1, "extra": 0}',
        "{}",
        "[]",
        '[{"v": 1}]',
        "null",
        "1",
        '"{\\"v\\": 1}"',
        # malformed JSON
        "",
        "   ",
        '{"v": 1',
        '{"v": 1,}',
        "{'v': 1}",
        '{"v": 1} trailing',
        '{"v": 1}\n{"v": 1}',
        '﻿{"v": 1}',
        # duplicate keys: the last would win in a lenient parser
        '{"v": 0, "v": 1}',
        '{"v": 1, "v": 0}',
        # oversized and over-deep
        '{"v": 1, "pad": "' + "x" * (MAX_RESULT_BYTES + 10) + '"}',
        "[" * 100_000 + "]" * 100_000,
        '{"a":' * 5_000 + "1" + "}" * 5_000,
        # code-like payloads must be inert data
        '{"v": 1}; rm -rf /',
        '__import__("os").system("true")',
        "{\"v\": __import__('os')}",
    ],
    ids=lambda s: repr(s[:24]),
)
def test_row6_malformed_or_hostile_results_score_zero(pack_world: World, stdout: str) -> None:
    res, _ = grade(pack_world, says(stdout))
    assert (res.score, res.error) == (0.0, None)


@pytest.mark.parametrize(
    "kw",
    [
        {"code": 1},
        {"code": 126},
        {"code": 127},
        {"code": 137},
        {"code": -9},
        {"timed_out": True},
        {"truncated": True},
        {"sandbox_lost": True},
        {"code": 1, "timed_out": True},
    ],
)
def test_row6_a_right_answer_does_not_save_a_bad_exit(
    pack_world: World, kw: dict[str, Any]
) -> None:
    res, _ = grade(pack_world, says(EXPECTED, **kw))
    assert (res.score, res.error) == (0.0, None)


def test_row6_a_backend_that_raises_during_the_run_is_infra_not_a_score(pack_world: World) -> None:
    """The one case an exception means infra: the exec itself failed (the command cannot cause
    it). The grader reports it as an error, never as a pass."""

    def boom(_box: Box) -> ExecResult:
        raise RuntimeError("runtime exploded")

    res, backend = grade(pack_world, boom)
    assert res.score is None and res.error is not None
    assert all(b.destroyed == 1 for b in backend.boxes)


# --- agent files: names and sizes ----------


@pytest.mark.parametrize(
    "name",
    [
        "../.grader/run.sh",
        "a/../../.grader/run.sh",
        "/work/.grader/run.sh",
        "/etc/passwd",
        "./x",
        "a//b",
        "a/./b",
        "a\\b",
        "a\0b",
        "",
        ".",
        "..",
        "x" * 300,
        "ok/" + "y" * 300,
    ],
)
def test_row6_hostile_file_names_score_zero_and_never_reach_a_sandbox(
    pack_world: World, name: str
) -> None:
    res, backend = grade(pack_world, says(EXPECTED), {name: b"data", "solve.sh": b"x"})
    assert (res.score, res.error) == (0.0, None)
    assert backend.boxes == [], "a sandbox was created for a rejected output"


def test_row6_a_file_where_another_needs_a_directory(pack_world: World) -> None:
    res, backend = grade(pack_world, says(EXPECTED), {"a": b"1", "a/b": b"2"})
    assert (res.score, res.error) == (0.0, None) and backend.boxes == []


def test_row6_too_many_files_or_too_deep_or_too_big(pack_world: World) -> None:
    cases: list[dict[str, bytes]] = [
        {f"f{i}": b"x" for i in range(MAX_OUT_FILES + 1)},
        {"/".join(["d"] * (MAX_OUT_PATH_DEPTH + 1)) + "/f": b"x"},
        {"big": b"x" * (MAX_OUT_FILE_BYTES + 1)},
        {f"p{i}": b"x" * (3 * 1024 * 1024) for i in range(6)},  # 18 MiB in total
    ]
    for files in cases:
        res, backend = grade(pack_world, says(EXPECTED), files)
        assert (res.score, res.error) == (0.0, None) and backend.boxes == []


def test_row6_the_agents_files_never_land_on_the_trusted_run_script(pack_world: World) -> None:
    res, backend = grade(
        pack_world, says(EXPECTED), {".grader/run.sh": b"exit 0\n", "run.sh": b"exit 0\n"}
    )
    [box] = backend.boxes
    assert res.score == 1.0  # it graded; the script that decided is the pack's
    assert box.files[RUN_PATH] == (pack_world.pack_root / "tasks/se00/run").read_bytes()
    assert box.files["/work/out/.grader/run.sh"] == b"exit 0\n"  # inert data under /out
    assert [p for p in box.files if p.startswith("/work/") and not p.startswith("/work/out/")] == [
        RUN_PATH
    ]


# --- /out on a real filesystem ----------


def _symlink(src: Path, dst: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(src, dst, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")


@pytest.fixture
def out(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    (d / "answer.txt").write_text("ok")
    return d


def test_row6_control_a_plain_out_is_read(out: Path) -> None:
    assert collect_out_files(out) == {"answer.txt": b"ok"}


def test_row6_a_symlink_to_a_host_secret_in_out(out: Path, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret")
    _symlink(secret, out / "leak")
    with pytest.raises(OutputRejected):
        collect_out_files(out)


def test_row6_a_symlinked_directory_in_out(out: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "x").write_text("x")
    _symlink(other, out / "dir", directory=True)
    with pytest.raises(OutputRejected):
        collect_out_files(out)


def test_row6_out_itself_is_a_symlink(out: Path, tmp_path: Path) -> None:
    link = tmp_path / "out-link"
    _symlink(out, link, directory=True)
    with pytest.raises(OutputRejected):
        collect_out_files(link)


def test_row6_a_hard_link_in_out(out: Path, tmp_path: Path) -> None:
    host = tmp_path / "host.txt"
    host.write_text("host file")
    try:
        os.link(host, out / "hl")
    except OSError:
        pytest.skip("cannot create hard links here")
    with pytest.raises(OutputRejected):
        collect_out_files(out)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs on this platform")
def test_row6_a_fifo_in_out_does_not_hang_the_reader(out: Path) -> None:
    os.mkfifo(out / "pipe")  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(OutputRejected):
        collect_out_files(out)


def test_row6_a_huge_file_in_out(out: Path) -> None:
    with open(out / "big", "wb") as f:
        f.truncate(MAX_OUT_FILE_BYTES + 1)  # sparse: cheap to make, over the cap
    with pytest.raises(OutputRejected):
        collect_out_files(out)


def test_row6_a_directory_bomb_in_out(out: Path) -> None:
    deep = out
    for _ in range(MAX_OUT_PATH_DEPTH + 2):
        deep = deep / "d"
        deep.mkdir()
    with pytest.raises(OutputRejected):
        collect_out_files(out)
    shutil.rmtree(out / "d")
    for i in range(2001):
        (out / f"e{i}").mkdir()
    with pytest.raises(OutputRejected):
        collect_out_files(out)


# --- the pack and the grader image ----------


@pytest.mark.parametrize(
    "tamper",
    [
        lambda root: (root / "tasks/se00/check").write_bytes(b'{"v": 0}'),
        lambda root: (root / "tasks/se00/run").write_bytes(b"echo '{\"v\": 1}'\n"),
        lambda root: (root / "tasks/se00/input.json").write_bytes(b"{}"),
        lambda root: (root / "extra.py").write_bytes(b"import os\n"),
        lambda root: (root / "pack.toml").write_text(
            'name = "fake"\nversion = "1"\ngrader_image = "docker.io/library/python@sha256:'
            + "ee" * 32
            + '"\n'
        ),
        lambda root: (root / "splits.json").write_text('{"assignment": {}}'),
    ],
    ids=["check", "run", "input", "extra-file", "grader-image", "splits"],
)
def test_row6_any_change_to_the_pack_after_pinning_stops_the_grader(
    own_world: World, tamper: Any
) -> None:
    tamper(own_world.pack_root)
    with pytest.raises(PackHashMismatch):
        Grader(Backend(says(EXPECTED)), own_world.pack, own_world.pack_root)


@pytest.mark.parametrize(
    "image",
    [
        "python:latest",
        "python",
        "docker.io/library/python:3.12",
        "x@sha256:abc",
        "x@md5:" + "a" * 32,
    ],
)
def test_row6_the_grader_image_must_be_pinned_by_digest(own_world: World, image: str) -> None:
    (own_world.pack_root / "pack.toml").write_text(
        f'name = "fake"\nversion = "1"\ngrader_image = "{image}"\n'
    )
    with pytest.raises(PackError):
        load_pack(own_world.pack_root)


def test_row6_a_symlink_planted_in_the_pack_is_refused(own_world: World, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not in the pack")
    _symlink(outside, own_world.pack_root / "tasks" / "se00" / "extra")
    with pytest.raises(PackError):
        load_pack(own_world.pack_root)


def test_row6_the_grader_never_runs_in_the_candidates_image(world) -> None:  # type: ignore[no-untyped-def]
    """Whatever the candidate built, the sandboxes that grade use the pack's pinned image."""
    world.queue(world.proposal("v0", "v1", "h"))
    world.orchestrator().run_generation()
    grading = [sb for sb in world.backend.sandboxes if sb.net == "default-deny"]
    assert grading
    assert {sb.image for sb in grading} == {GRADER_IMAGE}
    assert not {sb.image for sb in world.backend.sandboxes if sb.net == "proxy-only"} & {
        GRADER_IMAGE
    }


def test_row6_a_broken_grader_image_is_an_infra_error_not_a_wall_of_zeros(
    pack_world: World,
) -> None:
    from iterant.grader import GraderPreflightError

    backend = Backend(says("", code=127))
    backend.on_run = lambda _b: ExecResult(127, "", "python3: not found", False, 0.1)

    class Broken(Box):
        def exec(self, cmd: str, timeout: float) -> ExecResult:
            if cmd == "python3 -V":
                return ExecResult(127, "", "not found", False, 0.1)
            return super().exec(cmd, timeout)

    class BrokenBackend(Backend):
        def create(self, *a: Any, **k: Any) -> Sandbox:
            box = Broken(self.on_run, "default-deny", "x")
            self.boxes.append(box)
            return box  # type: ignore[return-value]

    grader = Grader(BrokenBackend(says("")), pack_world.pack, pack_world.pack_root)
    with pytest.raises(GraderPreflightError):
        grader.preflight()


def test_row6_agent_files_are_inert_bytes_whatever_they_say(pack_world: World) -> None:
    """The text of an agent file is never interpreted by the grading host: it is only ever
    stored in the sandbox. A file full of host-side code is just data."""
    payload = b"import os; os.system('echo pwned')\n__import__('pickle')\n" + bytes(range(256))
    res, backend = grade(
        pack_world, says(EXPECTED), {"sitecustomize.py": payload, "x.pkl": payload}
    )
    assert res.score == 1.0
    assert backend.boxes[0].files["/work/out/sitecustomize.py"] == payload
    assert json.loads(EXPECTED) == {"v": 1}
