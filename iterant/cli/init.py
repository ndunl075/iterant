"""``iterant init``: write a starter ``iterant.toml`` (§2) into a target repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iterant.cli.errors import EXIT_OK, EXIT_USAGE, CliError
from iterant.cli.settings import CONFIG_NAME, usd_to_micros
from iterant.contracts import Profile
from iterant.target import ConfigError, parse_config

FREE_MODEL = "llama3.1:8b"
CLAUDE_MODEL = "claude-haiku-5-5"
FREE_MAX_TOKENS = 200_000
CLAUDE_MAX_USD = "0.10"
DEFAULT_RUN = "python -m agent --task /task/input.json --out /out"


def _q(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def render_config(
    *,
    genome: list[str],
    image: str,
    run: str,
    pack: str,
    model: str,
    effort: str,
    max_usd: str | None,
    max_tokens: int | None,
) -> str:
    lines = [
        "# Iterant starter config. Edit the genome globs to the files the proposer may change;",
        "# everything else in this repo stays frozen.",
        "[target]",
        f"genome = [{', '.join(_q(g) for g in genome)}]",
        f"image  = {_q(image)}",
        f"run    = {_q(run)}",
        "",
        "[target.model]  # frozen: the proxy rejects any other model",
        f"name   = {_q(model)}",
        f"effort = {_q(effort)}",
    ]
    if max_usd is not None:
        lines.append(f"max_usd_per_run = {max_usd}")
    if max_tokens is not None:
        lines.append(f"max_tokens_per_run = {max_tokens}")
    lines += ["", "[eval]", f"pack = {_q(pack)}", ""]
    return "\n".join(lines)


def cmd_init(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    if not repo.is_dir():
        raise CliError(f"{repo}: not a directory", EXIT_USAGE)
    profile = Profile(args.profile)
    free = profile is Profile.FREE
    max_usd = args.max_usd_per_run
    max_tokens = args.max_tokens_per_run
    if max_usd is None and max_tokens is None:
        max_usd, max_tokens = (None, FREE_MAX_TOKENS) if free else (CLAUDE_MAX_USD, None)
    if max_usd is not None:
        usd_to_micros(max_usd, "--max-usd-per-run")  # a clear error before the parser's
    text = render_config(
        genome=args.genome or ["prompts/**"],
        image=args.image,
        run=args.run,
        pack=args.pack,
        model=args.model or (FREE_MODEL if free else CLAUDE_MODEL),
        effort=args.effort,
        max_usd=max_usd,
        max_tokens=max_tokens,
    )
    try:
        parse_config(text)  # never write a file the rest of Iterant would refuse
    except ConfigError as exc:
        raise CliError(f"cannot write a valid {CONFIG_NAME}: {exc}", EXIT_USAGE) from exc
    target = repo / CONFIG_NAME
    if target.exists() and not args.force:
        raise CliError(f"{target} already exists; pass --force to overwrite it", EXIT_USAGE)
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {target} ({profile.value} profile)")
    for note in _notes(repo, args.image, args.pack):
        print(f"note: {note}")
    print(
        "next: edit the genome and run globs, add a task pack, then `iterant estimate`, "
        "`iterant aa-test` and `iterant run`."
    )
    return EXIT_OK


def _notes(repo: Path, image: str, pack: str) -> list[str]:
    notes: list[str] = []
    if not (repo / ".git").exists():
        notes.append(f"{repo} is not a git repo; `iterant run` needs one with a commit")
    if not (repo / image).is_file():
        notes.append(f"{image} does not exist yet: the target image is built from it")
    if not (repo / pack / "pack.toml").is_file():
        notes.append(f"no task pack at {pack} yet (pack.toml, splits.json, tasks/<id>/...)")
    return notes
