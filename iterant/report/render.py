"""Render the archive as static HTML: ``index.html`` plus one ``gen-N.html`` per promotion.

Stdlib only, inline CSS, no scripts, no external fetches. Every value that came from the
archive, a diff, a hypothesis or a pack goes through ``_e`` (``html.escape``), and a CSP
meta tag forbids scripts as a second layer.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Final

from iterant.contracts import AcceptanceThresholds, Pack
from iterant.report.model import (
    CODE_EXEC_PATTERN,
    ArmStats,
    CandidateReport,
    GenerationReport,
    ReportData,
    StageEvidence,
    collect,
)

_CSS: Final = (
    "body{font:15px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:62rem;padding:0 1rem;"
    "color:#1b1f24;background:#fff}table{border-collapse:collapse;margin:.5rem 0}"
    "th,td{border:1px solid #c8ccd0;padding:.25rem .6rem;text-align:left;vertical-align:top}"
    "pre{background:#f4f5f6;padding:.6rem;overflow:auto;font-size:13px}"
    ".banner{background:#fff3cd;border:2px solid #b58100;padding:.6rem 1rem;font-weight:600}"
    ".flag{background:#fde0dc;border:1px solid #b3261e;padding:.1rem .4rem}"
    ".muted{color:#5b636b}code{background:#f4f5f6;padding:0 .2rem}"
)
_CSP: Final = "default-src 'none'; style-src 'unsafe-inline'"
_STALE: Final = re.compile(r"gen-\d+\.html")


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _usd(micros: float) -> str:
    return f"${micros / 1_000_000:.4f}"


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>\n"
    )


def _banner(data: ReportData) -> str:
    """Fails closed: no usable pack means a prominent warning, never silence."""
    if data.pack is None:
        return f'<p class="banner">{_e(data.pack_warning)}</p>'
    if data.pack.code_exec_policy == "advisory-forbidden":
        return (
            '<p class="banner">This pack forbids the agent from executing code '
            "(agent_code_exec_policy = advisory-forbidden), but that is NOT enforced. "
            "Nothing stops a candidate from running code; promoted diffs that look like they "
            "add code execution are marked below.</p>"
        )
    return ""


def _thresholds(t: AcceptanceThresholds) -> str:
    return (
        f"SELECT: delta &ge; {_e(t.delta_min)} (the minimum gain worth keeping), "
        f"p &lt; {_e(t.select_p)}. "
        f"CONFIRM: delta &gt; 0, p &lt; {_e(t.holdout_p)}. "
        f"Guards: cost &le; +{_e(f'{t.max_cost_increase:.0%}')}, "
        f"wall time &le; +{_e(f'{t.max_wall_increase:.0%}')}, "
        "first-try refusal and infra rates not clearly worse."
    )


def _arms(arms: tuple[ArmStats, ...]) -> str:
    rows = "".join(
        f"<tr><td>{_e(a.arm)}</td><td>{_e(a.runs)}</td><td>{_e(f'{a.refusal_rate:.1%}')}</td>"
        f"<td>{_e(f'{a.infra_rate:.1%}')}</td><td>{_e(_usd(a.mean_usd_micros))}</td>"
        f"<td>{_e(f'{a.mean_wall_s:.2f}s')}</td></tr>"
        for a in arms
    )
    return (
        "<table><tr><th>arm</th><th>runs</th><th>refusal</th><th>infra error</th>"
        f"<th>cost/run</th><th>wall/run</th></tr>{rows}</table>"
    )


def _stage_html(ev: StageEvidence | None, label: str, source: str) -> str:
    if ev is None:
        return f"<h4>{_e(label)}</h4><p class='muted'>not reached</p>"
    if ev.stored:
        src = "recorded decision, as stored in the promotions row"
    else:
        src = (
            f"recomputed with {source} and a fixed seed, NOT the recorded decision; "
            "no verdict is implied"
        )
    delta = "n/a" if ev.delta is None else f"{ev.delta:+.4f}"
    p = "n/a" if ev.p is None else f"{ev.p:.4f}"
    tasks = "" if ev.n_tasks is None else f", {ev.n_tasks} paired tasks, {ev.n_dropped} dropped"
    note = f"<p class='muted'>{_e(ev.note)}</p>" if ev.note else ""
    return (
        f"<h4>{_e(label)}</h4><p>delta vs parent {_e(delta)}, one-sided p {_e(p)}{_e(tasks)} "
        f"<span class='muted'>({_e(src)})</span></p>{note}{_arms(ev.arms)}"
    )


def _diff_html(c: CandidateReport) -> str:
    note = f"<p class='muted'>{_e(c.diff_note)}</p>" if c.diff_note else ""
    body = f"<pre>{_e(c.diff)}</pre>" if c.diff else "<p class='muted'>(no diff text)</p>"
    return note + body


def _candidate_html(c: CandidateReport, *, detail: bool, source: str) -> str:
    k = c.cand
    promoted = k.status.value == "promoted"
    flags = ""
    if promoted and c.diff_missing:
        flags += (
            '<p class="banner">The diff of this promoted candidate is missing from the diff '
            "store: what was promoted cannot be shown here.</p>"
        )
    if promoted and c.code_exec:
        flags += (
            '<p><span class="flag">Possible code execution added</span> '
            "(heuristic pattern match on added lines, may miss or over-flag).</p>"
        )
    head = (
        f"<h3>{_e(k.id)} <span class='muted'>(parent {_e(k.parent)}, {_e(k.status.value)} at "
        f"{_e(k.stage.value)}, run spend {_e(_usd(k.usd_micros))})</span></h3>"
        f"<p>Hypothesis: {_e(k.hypothesis)}</p>{flags}"
    )
    ev = _stage_html(c.select, "SELECT", source) + _stage_html(c.confirm, "CONFIRM", source)
    if detail:
        return head + ev + f"<h4>Diff</h4>{_diff_html(c)}"
    return head + ev + f"<details><summary>Diff</summary>{_diff_html(c)}</details>"


def _pack_line(g: GenerationReport, data: ReportData) -> str:
    parts: list[str] = []
    for h in g.pack_hashes:
        name = "name unknown"
        if data.pack is not None and data.pack.content_hash == h:
            name = f"{data.pack.name} {data.pack.version}"
        parts.append(f"{_e(name)} (<code>{_e(h)}</code>)")
    line = "; ".join(parts) or "no batch"
    if g.pack_matches is False:
        line += (
            ' <span class="banner">PACK HASH MISMATCH: part of this generation ran on a '
            "different pack than the one supplied; its name and policy do not apply here.</span>"
        )
    return line


def _gen_section(g: GenerationReport, data: ReportData) -> str:
    title = "gen-0 (seed champion)" if g.gen == 0 else f"gen-{g.gen}"
    link = f' <a href="gen-{_e(g.gen)}.html">evidence page</a>' if g.promotion else ""
    roles = ", ".join(f"{_e(r)} {_e(_usd(u))}" for r, u in g.by_role)
    spend = (
        f"<p>Spend {_e(_usd(g.usd_micros))} ({_e(g.in_tok)} in / {_e(g.out_tok)} out tokens)"
        f"{f' by role: {roles}' if roles else ''}.</p>"
    )
    many = ""
    if len(g.batch_ids) > 1:
        many = (
            f"<p class='muted'>{_e(len(g.batch_ids))} batches share this generation number "
            "(earlier ones were abandoned or restarted); all are listed.</p>"
        )
    promo = ""
    if g.promotion:
        p = g.promotion
        promo = (
            f"<p>Promoted <code>{_e(p.cand)}</code>: SELECT delta {_e(f'{p.select_delta:+.4f}')} "
            f"(p {_e(f'{p.select_p:.4f}')}), CONFIRM delta {_e(f'{p.holdout_delta:+.4f}')} "
            f"(p {_e(f'{p.holdout_p:.4f}')}).</p>"
        )
    elif g.batch_ids:
        promo = "<p>No candidate promoted.</p>"
    cands = "".join(
        _candidate_html(c, detail=False, source=data.thresholds_source) for c in g.candidates
    )
    return (
        f"<section id='gen-{_e(g.gen)}'><h2>{_e(title)}{link}</h2>"
        f"<p>Batches: {_e(', '.join(map(str, g.batch_ids)) or 'none')}. "
        f"Pack: {_pack_line(g, data)}</p>{many}{promo}{spend}{cands}</section>"
    )


_SPEND_NOTE: Final = (
    "Spend comes from the proxy log. Per generation it is the calls of that generation's "
    "batches plus run-less calls (proposer, analyst, auditor) made from its first batch until "
    "the next generation's first batch. Run-less calls before the first batch belong to no "
    "generation, so the total can exceed the per-generation sum. A candidate's own figure is "
    "run spend only (its target runs), not proposer, analyst or auditor calls."
)


def _index(data: ReportData) -> str:
    promoted = [g for g in data.generations if g.promotion]
    steps = [f"<code>{_e(data.lineage[0])}</code>"]
    steps += [
        f"<code>{_e(c)}</code> (<a href='gen-{_e(g.gen)}.html'>gen-{_e(g.gen)}</a>)"
        for c, g in zip(data.lineage[1:], promoted, strict=True)
    ]
    lineage = " &rarr; ".join(steps)
    rows = "".join(
        f"<tr><td><a href='#gen-{_e(g.gen)}'>{_e(g.gen)}</a></td>"
        f"<td>{_e(g.promotion.cand if g.promotion else '-')}</td>"
        f"<td>{_e(f'{g.promotion.select_delta:+.4f}' if g.promotion else '-')}</td>"
        f"<td>{_e(f'{g.promotion.holdout_delta:+.4f}' if g.promotion else '-')}</td>"
        f"<td>{_e(_usd(g.usd_micros))}</td></tr>"
        for g in data.generations
    )
    warn = "".join(f"<p class='banner'>{_e(w)}</p>" for w in data.warnings)
    body = (
        "<h1>Iterant evidence report</h1>"
        f"{_banner(data)}{warn}"
        f"<p>Total spend {_e(_usd(data.total_usd_micros))} (of which "
        f"{_e(_usd(data.unattributed_usd_micros))} in no generation); holdout queried "
        f"{_e(data.holdout_queries)} times.</p>"
        f"<h2>Lineage</h2><p>{lineage}</p>"
        "<table><tr><th>gen</th><th>champion</th><th>SELECT delta</th><th>CONFIRM delta</th>"
        f"<th>spend</th></tr>{rows}</table>"
        f"<p class='muted'>{_e(_SPEND_NOTE)}</p>"
        "<p class='muted'>Promoted rows show the recorded decision's numbers. Other candidates "
        f"show delta and p recomputed from archived runs with {_e(data.thresholds_source)} "
        f"({_thresholds(data.thresholds)}) and a fixed seed; the archive does not store the "
        "run's thresholds or seed, so these are not the recorded decision. No confidence "
        "interval is stored.</p>" + "".join(_gen_section(g, data) for g in data.generations)
    )
    return _page("Iterant evidence report", body)


def _promotion_page(g: GenerationReport, data: ReportData) -> str:
    assert g.promotion is not None
    winner = next((c for c in g.candidates if c.cand.id == g.promotion.cand), None)
    body = (
        f"<p><a href='index.html'>&larr; report</a></p><h1>gen-{_e(g.gen)}</h1>"
        f"{_banner(data)}<p>Pack: {_pack_line(g, data)}</p>"
        f"<p>Spend {_e(_usd(g.usd_micros))} ({_e(g.in_tok)} in / {_e(g.out_tok)} out tokens).</p>"
        f"<p class='muted'>{_e(_SPEND_NOTE)}</p>"
    )
    if winner is None:
        body += f"<p>Promoted candidate {_e(g.promotion.cand)} is not in this archive.</p>"
    else:
        body += _candidate_html(winner, detail=True, source=data.thresholds_source)
        if winner.code_exec:
            pattern = _e(CODE_EXEC_PATTERN.pattern)
            body += f"<p class='muted'>Heuristic pattern: <code>{pattern}</code></p>"
    return _page(f"gen-{g.gen}", body)


def render(
    archive_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    pack: Pack | None = None,
    thresholds: AcceptanceThresholds | None = None,
    thresholds_source: str | None = None,
    diffs_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Write ``index.html`` and ``gen-N.html`` (one per promotion) into ``out_dir``.

    Returns the path of ``index.html``. The archive is opened read-only. Diffs are read from
    ``diffs_dir`` (default ``diffs/`` beside the archive file, i.e. ``<repo>/.iterant/diffs``).
    The archive stores pack hashes but not pack names or policy, so pass ``pack`` to show the
    name and the advisory-forbidden banner; without a pack whose hash matches a batch the
    report carries a prominent warning instead. Stale ``gen-N.html`` files in ``out_dir`` are
    removed first.
    """
    data = collect(
        Path(archive_path),
        pack=pack,
        thresholds=thresholds,
        diffs_dir=None if diffs_dir is None else Path(diffs_dir),
    )
    if thresholds_source is not None:
        data = replace(data, thresholds_source=thresholds_source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if _STALE.fullmatch(old.name) and (old.is_file() or old.is_symlink()):
            old.unlink()
    for g in data.generations:
        if g.promotion is not None:
            (out / f"gen-{g.gen}.html").write_text(_promotion_page(g, data), encoding="utf-8")
    index = out / "index.html"
    index.write_text(_index(data), encoding="utf-8")
    return index


__all__ = ["render"]
