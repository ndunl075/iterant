"""Render the archive as static HTML: ``index.html`` plus one ``gen-N.html`` per promotion.

Stdlib only, inline CSS, no scripts, no external fetches. Every value that came from the
archive, a diff or a hypothesis goes through ``_e`` (``html.escape``), and a CSP meta tag
forbids scripts as a second layer.
"""

from __future__ import annotations

import html
import os
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
    ".ok{color:#1a7f37}.bad{color:#b3261e}.muted{color:#5b636b}"
    "code{background:#f4f5f6;padding:0 .2rem}"
)
_CSP: Final = "default-src 'none'; style-src 'unsafe-inline'"


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


def _banner(pack: Pack | None) -> str:
    if pack is None:
        return (
            '<p class="muted">Pack policy not supplied to the report: whether the agent may '
            "execute code is unknown here.</p>"
        )
    if pack.code_exec_policy == "advisory-forbidden":
        return (
            '<p class="banner">This pack forbids the agent from executing code '
            "(agent_code_exec_policy = advisory-forbidden), but that is NOT enforced. "
            "Nothing stops a candidate from running code; promoted diffs that look like they "
            "add code execution are marked below.</p>"
        )
    return ""


def _thresholds(t: AcceptanceThresholds) -> str:
    return (
        f"SELECT: delta &ge; {_e(t.delta_min)}, p &lt; {_e(t.select_p)}. "
        f"CONFIRM: delta &gt; 0, p &lt; {_e(t.holdout_p)}. "
        f"Guards: cost &le; +{_e(f'{t.max_cost_increase:.0%}')}, "
        f"wall time &le; +{_e(f'{t.max_wall_increase:.0%}')}, refusal and infra rates no worse."
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


def _stage_html(ev: StageEvidence | None, label: str) -> str:
    if ev is None:
        return f"<h4>{_e(label)}</h4><p class='muted'>not reached</p>"
    src = "as stored in the promotions row" if ev.stored else "recomputed from archived runs"
    delta = "n/a" if ev.delta is None else f"{ev.delta:+.4f}"
    p = "n/a" if ev.p is None else f"{ev.p:.4f}"
    tasks = "" if ev.n_tasks is None else f", {ev.n_tasks} paired tasks, {ev.n_dropped} dropped"
    if ev.accepted is None:
        verdict = ""
    else:
        cls, word = ("ok", "passes") if ev.accepted else ("bad", "fails")
        verdict = f' <span class="{cls}">{word} thresholds + guards (recomputed)</span>'
    detail = "".join(f"<li>{_e(d)}</li>" for d in ev.detail)
    ul = f"<ul>{detail}</ul>" if detail else ""
    return (
        f"<h4>{_e(label)}</h4><p>delta vs parent {_e(delta)}, one-sided p {_e(p)}{_e(tasks)}"
        f"{verdict} <span class='muted'>({_e(src)})</span></p>{ul}{_arms(ev.arms)}"
    )


def _diff_html(c: CandidateReport) -> str:
    note = f"<p class='muted'>{_e(c.diff_note)}</p>" if c.diff_note else ""
    body = f"<pre>{_e(c.diff)}</pre>" if c.diff else "<p class='muted'>(empty diff)</p>"
    return note + body


def _exec_flag(c: CandidateReport) -> str:
    if not c.code_exec:
        return ""
    return (
        '<p><span class="flag">Possible code execution added</span> '
        "(heuristic pattern match on added lines, may miss or over-flag).</p>"
    )


def _candidate_html(c: CandidateReport, *, detail: bool) -> str:
    k = c.cand
    flag = _exec_flag(c) if k.status.value == "promoted" else ""
    head = (
        f"<h3>{_e(k.id)} <span class='muted'>(parent {_e(k.parent)}, {_e(k.status.value)} at "
        f"{_e(k.stage.value)}, {_e(_usd(k.usd_micros))})</span></h3>"
        f"<p>Hypothesis: {_e(k.hypothesis)}</p>{flag}"
    )
    ev = _stage_html(c.select, "SELECT") + _stage_html(c.confirm, "CONFIRM")
    diff = f"<h4>Diff</h4>{_diff_html(c)}"
    if detail:
        return head + ev + diff
    return head + ev + f"<details><summary>Diff</summary>{_diff_html(c)}</details>"


def _pack_line(gen: GenerationReport, pack: Pack | None) -> str:
    parts: list[str] = []
    for h in gen.pack_hashes:
        name = "name not recorded in archive"
        if pack is not None and pack.content_hash == h:
            name = f"{pack.name} {pack.version}"
        parts.append(f"{_e(name)} (<code>{_e(h)}</code>)")
    return "; ".join(parts) or "no batch"


def _gen_section(g: GenerationReport, data: ReportData) -> str:
    title = "gen-0 (seed champion)" if g.gen == 0 else f"gen-{g.gen}"
    link = f' <a href="gen-{g.gen}.html">evidence page</a>' if g.promotion else ""
    roles = ", ".join(f"{_e(r)} {_e(_usd(u))}" for r, u in g.by_role)
    spend = (
        f"<p>Spend {_e(_usd(g.usd_micros))} ({_e(g.in_tok)} in / {_e(g.out_tok)} out tokens)"
        f"{f' by role: {roles}' if roles else ''}.</p>"
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
    cands = "".join(_candidate_html(c, detail=False) for c in g.candidates)
    return (
        f"<section id='gen-{_e(g.gen)}'><h2>{_e(title)}{link}</h2>"
        f"<p>Batches: {_e(', '.join(map(str, g.batch_ids)) or 'none')}. "
        f"Pack: {_pack_line(g, data.pack)}</p>{promo}{spend}{cands}</section>"
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
        f"{_banner(data.pack)}{warn}"
        f"<p>Total spend {_e(_usd(data.total_usd_micros))}; holdout queried "
        f"{_e(data.holdout_queries)} times.</p>"
        f"<h2>Lineage</h2><p>{lineage}</p>"
        "<table><tr><th>gen</th><th>champion</th><th>SELECT delta</th><th>CONFIRM delta</th>"
        f"<th>spend</th></tr>{rows}</table>"
        f"<p class='muted'>Thresholds (defaults assumed; the archive does not store the run's "
        f"config): {_thresholds(data.thresholds)} Recomputed p-values use a fixed seed and may "
        "differ slightly from the orchestrator's. No confidence interval is stored.</p>"
        + "".join(_gen_section(g, data) for g in data.generations)
    )
    return _page("Iterant evidence report", body)


def _promotion_page(g: GenerationReport, data: ReportData) -> str:
    assert g.promotion is not None
    winner = next((c for c in g.candidates if c.cand.id == g.promotion.cand), None)
    body = (
        f"<p><a href='index.html'>&larr; report</a></p><h1>gen-{_e(g.gen)}</h1>"
        f"{_banner(data.pack)}<p>Pack: {_pack_line(g, data.pack)}</p>"
        f"<p>Spend {_e(_usd(g.usd_micros))} ({_e(g.in_tok)} in / {_e(g.out_tok)} out tokens).</p>"
        f"<p class='muted'>Thresholds: {_thresholds(data.thresholds)}</p>"
    )
    if winner is None:
        body += f"<p>Promoted candidate {_e(g.promotion.cand)} is not in this archive.</p>"
    else:
        body += _candidate_html(winner, detail=True)
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
) -> Path:
    """Write ``index.html`` and ``gen-N.html`` (one per promotion) into ``out_dir``.

    Returns the path of ``index.html``. ``archive_path`` is ``<repo>/.iterant/archive.db``;
    diffs are read from ``<repo>/.iterant/diffs``. The archive stores pack hashes but not
    pack names or policy, so pass ``pack`` to show the name and the advisory-forbidden banner,
    and ``thresholds`` if the run did not use the defaults.
    """
    data = collect(Path(archive_path), pack=pack, thresholds=thresholds)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in data.generations:
        if g.promotion is not None:
            (out / f"gen-{g.gen}.html").write_text(_promotion_page(g, data), encoding="utf-8")
    index = out / "index.html"
    index.write_text(_index(data), encoding="utf-8")
    return index


__all__ = ["render"]
