from __future__ import annotations

from pathlib import Path

import pytest

from iterant.target import ConfigError, load_config, parse_config

GENOME = 'genome = ["prompts/**", "tools/**", "agent/loop.py"]'
RUN = 'run = "python -m agent --task /task/input.json --out /out"'
BASE = f"""
[target]
{GENOME}
image = "Containerfile"
{RUN}
[target.model]
name = "claude-haiku-5-5"
effort = "medium"
max_usd_per_run = 0.10
[eval]
pack = "evals/support-pack"
"""


def test_parses_spec_example() -> None:
    cfg = parse_config(BASE)
    assert cfg.target.genome == ("prompts/**", "tools/**", "agent/loop.py")
    assert cfg.target.image == "Containerfile"
    assert cfg.target.model.name == "claude-haiku-5-5"
    assert cfg.target.model.effort == "medium"
    assert cfg.target.model.max_usd_micros_per_run == 100_000
    assert cfg.target.model.max_tokens_per_run is None
    assert cfg.evaluation.pack == "evals/support-pack"


def test_load_config_reads_file(tmp_path: Path) -> None:
    (tmp_path / "iterant.toml").write_text(BASE, encoding="utf-8")
    assert load_config(tmp_path / "iterant.toml") == parse_config(BASE)


def _with_cap(line: str) -> str:
    return BASE.replace("max_usd_per_run = 0.10", line)


@pytest.mark.parametrize(
    ("literal", "micros"),
    [
        ("0.10", 100_000),
        ("0.1", 100_000),
        ("0.3", 300_000),
        ("0.07", 70_000),
        ("1", 1_000_000),
        ("12.345678", 12_345_678),
        ("0.000001", 1),
        ("1e-3", 1_000),
        ("999.5", 999_500_000),
        ("19.99", 19_990_000),
        ("1000", 1_000_000_000),
        ("0.1000000000000000000000000000000", 100_000),
    ],
)
def test_usd_to_micros_exact(literal: str, micros: int) -> None:
    cfg = parse_config(_with_cap(f"max_usd_per_run = {literal}"))
    assert cfg.target.model.max_usd_micros_per_run == micros


def test_excess_precision_is_error_not_rounded() -> None:
    # 31 significant digits: Decimal's default context would round this to 0.1.
    with pytest.raises(ConfigError, match="finer than 1 micro-USD"):
        parse_config(_with_cap("max_usd_per_run = 0.1000000000000000000000000000001"))


def test_tokens_cap_for_free_profile() -> None:
    cfg = parse_config(_with_cap("max_tokens_per_run = 50000"))
    assert cfg.target.model.max_tokens_per_run == 50_000
    assert cfg.target.model.max_usd_micros_per_run is None


def test_both_caps() -> None:
    cfg = parse_config(_with_cap("max_usd_per_run = 0.5\nmax_tokens_per_run = 9"))
    assert cfg.target.model.max_usd_micros_per_run == 500_000
    assert cfg.target.model.max_tokens_per_run == 9


def _sub(old: str, new: str) -> str:
    assert old in BASE
    return BASE.replace(old, new)


CASES: list[tuple[str, str]] = [
    (_with_cap("max_usd_per_run = 0.0000001"), "target.model.max_usd_per_run: "),
    (_with_cap("max_usd_per_run = 1e30"), "target.model.max_usd_per_run: must be at most 1000"),
    (_with_cap("max_usd_per_run = 1000.01"), "target.model.max_usd_per_run: must be at most 1000"),
    (_with_cap("max_usd_per_run = 1e20000000"), "target.model.max_usd_per_run: must be at most"),
    (_with_cap("max_usd_per_run = 1e-20000000"), "target.model.max_usd_per_run: "),
    (_with_cap("max_usd_per_run = 0"), "target.model.max_usd_per_run: must be > 0"),
    (_with_cap("max_usd_per_run = -1"), "target.model.max_usd_per_run: must be > 0"),
    (_with_cap("max_usd_per_run = nan"), "target.model.max_usd_per_run: must be finite"),
    (_with_cap("max_usd_per_run = inf"), "target.model.max_usd_per_run: must be finite"),
    (_with_cap("max_usd_per_run = true"), "target.model.max_usd_per_run: expected a number"),
    (_with_cap('max_usd_per_run = "0.1"'), "target.model.max_usd_per_run: expected a number"),
    (_with_cap("max_tokens_per_run = 0"), "target.model.max_tokens_per_run: must be > 0"),
    (_with_cap("max_tokens_per_run = 1.5"), "target.model.max_tokens_per_run: expected an int"),
    (_with_cap("max_tokens_per_run = true"), "target.model.max_tokens_per_run: expected an int"),
    (_with_cap("surprise = 1"), "target.model.surprise: unknown key"),
    (_sub("max_usd_per_run = 0.10", ""), "target.model: set max_usd_per_run"),
    (_sub('effort = "medium"', 'effort = "turbo"'), "target.model.effort: must be one of"),
    (_sub('effort = "medium"', "effort = 3"), "target.model.effort: expected a string"),
    (_sub('name = "claude-haiku-5-5"', 'name = ""'), "target.model.name: must not be empty"),
    (_sub('name = "claude-haiku-5-5"\n', ""), "target.model.name: required"),
    (_sub('image = "Containerfile"', "image = 7"), "target.image: expected a string"),
    (_sub('image = "Containerfile"', 'image = "../Containerfile"'), "target.image: "),
    (_sub('image = "Containerfile"', 'image = "/etc/Containerfile"'), "target.image: "),
    (_sub('pack = "evals/support-pack"', 'pack = "../packs"'), "eval.pack: "),
    (_sub('pack = "evals/support-pack"', 'pack = "C:/packs"'), "eval.pack: "),
    (_sub(RUN, "run = []"), "target.run: expected a string"),
    (_sub(GENOME, "genome = []"), "target.genome: must list"),
    (_sub(GENOME, 'genome = "prompts/**"'), "target.genome: expected an array"),
    (_sub('"tools/**"', '"../secrets/**"'), "target.genome[1]: "),
    (_sub('"tools/**"', '"/abs/**"'), "target.genome[1]: "),
    (_sub('"tools/**"', "3"), "target.genome[1]: expected a string"),
    (_sub('"tools/**"', '".git/**"'), "target.genome[1]: "),
    (_sub('"tools/**"', '"tools/[ab].py"'), "target.genome[1]: "),
    (BASE + "\n[extra]\nx = 1\n", "extra: unknown key"),
    (_sub("[eval]", "[evl]"), "evl: unknown key"),
    (_sub("[eval]\npack", "[eval]\nfoo = 1\npack"), "eval.foo: unknown key"),
    (_sub('[eval]\npack = "evals/support-pack"', ""), "eval: required"),
    (_sub("[target.model]", "[target.mdl]"), "target.mdl: unknown key"),
    (_sub("[target]", "[target]\nextra = 1"), "target.extra: unknown key"),
    ("eval = 3\n" + _sub('[eval]\npack = "evals/support-pack"', ""), "eval: expected a table"),
    ("not = [valid", "<toml>:"),
    ("", "target: required"),
]


@pytest.mark.parametrize(("text", "needle"), CASES)
def test_errors_name_the_key_path(text: str, needle: str) -> None:
    with pytest.raises(ConfigError) as err:
        parse_config(text)
    assert needle in str(err.value)


def test_target_must_be_table() -> None:
    with pytest.raises(ConfigError, match=r"target: expected a table, got str"):
        parse_config('target = "x"\n[eval]\npack = "p"\n')


def test_paths_are_normalized() -> None:
    cfg = parse_config(_sub('"Containerfile"', '"./docker/Containerfile"'))
    assert cfg.target.image == "docker/Containerfile"
