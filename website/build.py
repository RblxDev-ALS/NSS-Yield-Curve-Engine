"""Build the project website from a pipeline output directory.

The live workflow runs ``nss-engine run --source fred --figures --out site``
and then ``python website/build.py site``. This adds to ``site/``:

* ``index.html`` - landing page: the latest reading in words and as a table
  with five-year sparklines (from ``summary.json`` and the CSVs), the three
  charts, what did not work, and where the project came from;
* ``results.html``, ``methodology.html``, ``par-vs-zero.html``,
  ``changelog.html`` - the Markdown documents rendered to HTML, with a
  contents list and the equations typeset by MathJax;
* ``style.css``, ``favicon.svg``, ``404.html`` and ``img/social.png`` (the
  preview image that Reddit, Slack, Discord and messaging apps show for a
  link; needs matplotlib).

The site's navigation bar is also added to the interactive ``dashboard.html``.
The charts in ``img/`` and the shields.io badges in ``badges/`` are already
there. Needs ``pip install markdown``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

REPO = "https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine"
SITE = "https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/"
AUTHOR = "RblxDev-ALS"
AUTHOR_URL = "https://github.com/RblxDev-ALS"
ROOT = Path(__file__).resolve().parents[1]

#: (output page, source Markdown, navigation label)
PAGES = (
    ("results", "docs/results.md", "Results"),
    ("methodology", "docs/methodology.md", "Methodology"),
    ("par-vs-zero", "docs/par-vs-zero.md", "Par vs zero"),
    ("changelog", "CHANGELOG.md", "Changelog"),
)

#: Page description for search engines and link previews.
PITCH = (
    "The U.S. Treasury yield curve fitted every week since 1990: the term premium, breakeven "
    "inflation and recession odds, recomputed from Federal Reserve data each weekday. "
    "Open-source Python."
)

FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#16304f"/>'
    '<path d="M5 25 C9 13, 14 10, 27 8" fill="none" stroke="#f2c14e" stroke-width="3" '
    'stroke-linecap="round"/></svg>'
)

CSS = """
:root { color-scheme: light;
  --page:#f8f7f4; --surface:#ffffff; --ink:#151515; --ink-2:#4a4945; --muted:#7d7b75;
  --rule:#dddbd3; --rule-2:#ebe9e3; --accent:#1f5fae; --accent-ink:#ffffff; --code:#efeee9;
  --spark:#1f5fae; --up:#9a3b17; --down:#1f5f3a; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark;
  --page:#121212; --surface:#1b1b1a; --ink:#f1f0ec; --ink-2:#c4c2ba; --muted:#8f8d86;
  --rule:#34332f; --rule-2:#262624; --accent:#7fb0ef; --accent-ink:#0d1b2c; --code:#232321;
  --spark:#7fb0ef; --up:#f0a07c; --down:#8fd1a8; } }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--page); color:var(--ink);
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.serif, h1, h2, .lead, .doc { font-family: Charter, "Bitstream Charter", "Sitka Text", Cambria,
  Georgia, serif; }
a { color: var(--accent); text-underline-offset: 2px; }
a:hover { text-decoration-thickness: 2px; }

/* navigation: one line on a phone, scrolls sideways if it has to */
.nav { border-bottom: 1px solid var(--rule); background: var(--page); }
.nav .inner { max-width: 1040px; margin: 0 auto; padding: 0 16px; display:flex; align-items: center;
  gap: 20px; height: 52px; }
.nav .brand { font-weight: 650; color: var(--ink); text-decoration: none; white-space: nowrap;
  display:flex; align-items:center; gap: 8px; }
.nav .brand img { width: 20px; height: 20px; }
.nav .links { display:flex; gap: 18px; overflow-x: auto; scrollbar-width: none; white-space: nowrap;
  margin-left: auto; }
.nav .links::-webkit-scrollbar { display: none; }
.nav .links a { color: var(--ink-2); text-decoration: none; font-size: 15px; padding: 14px 0;
  border-bottom: 2px solid transparent; }
.nav .links a:hover { color: var(--ink); }
.nav .links a.here { color: var(--ink); border-bottom-color: var(--ink); }

main { max-width: 1040px; margin: 0 auto; padding: 40px 16px 72px; }
h1 { font-size: 40px; line-height: 1.15; margin: 0 0 18px; font-weight: 700; letter-spacing: -0.015em;
  max-width: 820px; }
h2 { font-size: 26px; line-height: 1.25; margin: 56px 0 10px; font-weight: 700; }
h3 { font-size: 18px; margin: 28px 0 6px; }
p, li { max-width: 700px; }
.dateline { color: var(--muted); font-size: 14px; margin: 0 0 10px; }
.lead { font-size: 21px; line-height: 1.5; color: var(--ink); max-width: 760px; margin: 0 0 22px; }
.lead b { font-weight: 700; }
.actions { display:flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin: 0 0 8px; }
.button { display:inline-block; padding: 10px 18px; border-radius: 6px; background: var(--ink);
  color: var(--page); text-decoration: none; font-weight: 600; }
.button:hover { opacity: .88; }
.install { display:inline-flex; align-items:center; gap: 8px; background: var(--code);
  border-radius: 6px; padding: 4px 4px 4px 12px; }
.install code { background: none; padding: 0; }
.install button { font: inherit; font-size: 13px; border: 1px solid var(--rule); background: var(--surface);
  color: var(--ink-2); border-radius: 4px; padding: 3px 9px; cursor: pointer; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 14px; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--code); padding: 12px 14px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }

/* the latest reading */
.ledger { display: table; width: 100%; max-width: 860px; border-collapse: collapse; margin: 14px 0 6px; font-variant-numeric: tabular-nums; }
.ledger th { font-size: 13px; font-weight: 500; color: var(--muted); text-align: right;
  border-bottom: 1px solid var(--rule); padding: 6px 10px; }
.ledger th:first-child { text-align: left; padding-left: 0; }
.ledger td { padding: 11px 10px; border-bottom: 1px solid var(--rule-2); text-align: right;
  vertical-align: middle; white-space: nowrap; }
.ledger td:first-child { text-align: left; white-space: normal; padding-left: 0; }
.ledger .name { font-weight: 550; }
.ledger .note { display:block; font-size: 13px; color: var(--muted); }
.ledger .value { font-size: 20px; font-weight: 650; }
.ledger td.chg { color: var(--ink-2); font-size: 14px; }
.ledger .spark { width: 132px; padding-right: 0; }
.ledger svg { display:block; margin-left: auto; overflow: visible; }
.foot { color: var(--muted); font-size: 13.5px; margin: 6px 0 0; }

figure { margin: 26px 0 36px; }
figure img { width: 100%; height: auto; display:block; border: 1px solid var(--rule); border-radius: 4px; }
figcaption { color: var(--ink-2); font-size: 15px; margin-top: 8px; max-width: 700px; }
.negatives li { margin-bottom: 8px; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums;
  display:block; overflow-x:auto; max-width:100%; margin: 14px 0; }
th, td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--rule-2); text-align: left; }
th { color: var(--ink-2); font-weight: 600; border-bottom-color: var(--rule); }
blockquote { margin: 14px 0; padding: 2px 16px; border-left: 3px solid var(--rule); color: var(--ink-2); }
img { max-width: 100%; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 40px 0; }

/* documents: prose column with a contents list beside it on wide screens */
.docwrap { display:grid; grid-template-columns: minmax(0, 1fr); gap: 0 48px; }
.doc { font-size: 17.5px; line-height: 1.65; }
.doc h1 { font-size: 36px; }
.doc h2 { font-size: 25px; margin-top: 48px; }
.doc table, .doc pre, .doc .MathJax { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.doc pre, .doc code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.doc img { border-radius: 4px; }
.doc p code, .doc li code, .doc td code { overflow-wrap: anywhere; }
mjx-container[display="true"] { display: block; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
.toc { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.toc p { margin: 0 0 8px; color: var(--ink-2); }
.toc summary { cursor: pointer; color: var(--ink-2); font-weight: 600; margin-bottom: 6px; }
.toc ol { list-style: none; padding: 0; margin: 0; }
.toc li { margin: 0 0 6px; line-height: 1.35; }
.toc a { color: var(--ink-2); text-decoration: none; }
.toc a:hover { color: var(--accent); }
.toc.inline { margin: 0 0 24px; border: 1px solid var(--rule); border-radius: 6px; padding: 10px 14px; }
.toc.inline summary { margin: 0; }
.toc.inline[open] summary { margin-bottom: 8px; }
.toc.side { display: none; }
@media (min-width: 1000px) {
  .docwrap.has-toc { grid-template-columns: minmax(0, 1fr) 220px; }
  .docwrap.has-toc .toc.side { display: block; grid-column: 2; grid-row: 1; position: sticky; top: 24px;
    align-self: start; max-height: calc(100vh - 48px); overflow-y: auto; }
  .docwrap.has-toc .doc { grid-column: 1; grid-row: 1; }
  .toc.inline { display: none; }
}

footer { margin-top: 72px; padding-top: 18px; border-top: 1px solid var(--rule); color: var(--muted);
  font-size: 13.5px; }
footer p { max-width: 760px; margin: 0 0 6px; }
footer a { color: var(--ink-2); }

@media (max-width: 640px) {
  main { padding-top: 28px; }
  h1 { font-size: 30px; } h2 { font-size: 23px; margin-top: 44px; }
  .lead { font-size: 18.5px; }
  .doc { font-size: 17px; }
  /* long inline equations scroll sideways; on a list, so the markers stay in its padding */
  .doc p, .doc ul, .doc ol { overflow-x: auto; overflow-y: hidden; }
  .nav .inner { gap: 14px; }
  .nav .brand span { display: none; }
  .ledger .chg, .ledger th.chg { display: none; }
  .ledger td { padding: 9px 6px; }
  .ledger .name { line-height: 1.3; display: block; }
  .ledger .note { display: none; }
  .ledger .spark { width: 84px; }
  .ledger svg { width: 80px; height: auto; }
  .ledger .value { font-size: 17px; }
}
"""

MATHJAX = """<script>window.MathJax = { tex: { inlineMath: [['\\\\(', '\\\\)']],
  displayMath: [['\\\\[', '\\\\]']] } };</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""

COPY_JS = """<script>
document.querySelectorAll('[data-copy]').forEach(function (b) {
  b.addEventListener('click', function () {
    navigator.clipboard.writeText(b.dataset.copy).then(function () {
      b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1500);
    });
  });
});
</script>"""


def nav_html(here: str) -> str:
    links = [("dashboard.html", "Dashboard", "dashboard")]
    links += [(f"{name}.html", label, name) for name, _, label in PAGES]
    items = "".join(
        f'<a href="{href}"{" class=here" if key == here else ""}>{html.escape(label)}</a>'
        for href, label, key in links
    )
    return (
        '<nav class="nav"><div class="inner"><a class="brand" href="index.html">'
        '<img src="favicon.svg" alt=""><span>NSS Yield Curve Engine</span></a>'
        f'<div class="links">{items}<a href="{REPO}">GitHub</a></div></div></nav>'
    )


def head_meta(title: str, description: str, path: str) -> str:
    """Title, description, favicon and the Open Graph / Twitter tags for link previews."""
    t, d = html.escape(title), html.escape(description)
    image = f"{SITE}img/social.png"
    return f"""<title>{t}</title>
<meta name="description" content="{d}">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<meta name="theme-color" content="#f8f7f4" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#121212" media="(prefers-color-scheme: dark)">
<meta property="og:type" content="website">
<meta property="og:site_name" content="NSS Yield Curve Engine">
<meta property="og:title" content="{t}">
<meta property="og:description" content="{d}">
<meta property="og:url" content="{SITE}{path}">
<meta property="og:image" content="{image}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{image}">"""


def footer_html(as_of: str | None = None) -> str:
    stamp = f" Data as of {html.escape(as_of)}." if as_of else ""
    return f"""<footer>
<p>Made by <a href="{AUTHOR_URL}">{AUTHOR}</a>. Code on <a href="{REPO}">GitHub</a> under the MIT
license; issues and corrections welcome.{stamp}</p>
<p>Data: Board of Governors of the Federal Reserve System (H.15, the GSW nominal and TIPS curves,
Kim-Wright) via FRED, Federal Reserve Bank of St. Louis; Federal Reserve Bank of Philadelphia
(Survey of Professional Forecasters); NBER. Not investment advice.</p>
</footer>"""


def page(
    title: str,
    body: str,
    here: str,
    math: bool = False,
    description: str = PITCH,
    as_of: str | None = None,
    script: str = "",
) -> str:
    path = "" if here == "index" else f"{here}.html"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{head_meta(title, description, path)}
<link rel="stylesheet" href="style.css">
{MATHJAX if math else ""}
</head><body>
{nav_html(here)}
<main>{body}
{footer_html(as_of)}
</main>{script}</body></html>
"""


# =============================================================================
# Markdown documents
# =============================================================================

_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE = re.compile(r"(?<![\\$\w])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])")
_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def render_document(text: str) -> tuple[str, bool, list[dict[str, Any]]]:
    """Markdown to HTML with ``$…$`` / ``$$…$$`` kept intact for MathJax.

    Returns the HTML, whether it has math, and the second-level headings
    (``{"id", "name"}``) for a contents list.
    """
    import markdown

    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    # code blocks first, so a $ inside code is not read as math
    text = _FENCE.sub(lambda m: keep(m.group(0)), text)
    math_parts: list[str] = []

    def put_math(m: re.Match[str], display: bool) -> str:
        tex = html.escape(m.group(1).strip())
        math_parts.append(f"\\[{tex}\\]" if display else f"\\({tex}\\)")
        return f"MATHPLACEHOLDER{len(math_parts) - 1}X"

    text = _DISPLAY.sub(lambda m: put_math(m, True), text)
    text = _INLINE.sub(lambda m: put_math(m, False), text)
    text = re.sub("\x00(\\d+)\x00", lambda m: stash[int(m.group(1))], text)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    out = md.convert(text)
    out = re.sub(r"MATHPLACEHOLDER(\d+)X", lambda m: math_parts[int(m.group(1))], out)
    sections: list[dict[str, Any]] = []
    for top in getattr(md, "toc_tokens", []):
        # documents have one h1 title; list its h2 children (or top-level h2s)
        sections.extend(top.get("children", []) if top.get("level") == 1 else [top])
    toc = [
        {"id": t["id"], "name": re.sub(r"MATHPLACEHOLDER\d+X", "", html.unescape(t["name"]))}
        for t in sections
    ]
    return out, bool(math_parts), toc


def render_markdown(text: str) -> tuple[str, bool]:
    """Markdown to HTML with ``$…$`` / ``$$…$$`` kept intact for MathJax."""
    out, has_math, _ = render_document(text)
    return out, has_math


def toc_html(toc: list[dict[str, Any]], inline: bool = False) -> str:
    """Contents list: a sidebar on wide screens, a collapsed box under the title on phones."""
    if len(toc) < 4:
        return ""
    items = "".join(
        f'<li><a href="#{html.escape(t["id"])}">{html.escape(t["name"])}</a></li>' for t in toc
    )
    if inline:
        return f'<details class="toc inline"><summary>Contents</summary><ol>{items}</ol></details>'
    return f'<nav class="toc side" aria-label="Contents"><p><b>On this page</b></p><ol>{items}</ol></nav>'


def with_contents(body: str, toc: list[dict[str, Any]]) -> str:
    """Put the phone contents list after the title and drop a hand-written "Contents:" line."""
    inline = toc_html(toc, inline=True)
    if not inline:
        return body
    body = re.sub(r"<p>Contents:.*?</p>\s*", "", body, count=1, flags=re.DOTALL)
    if "</h1>" in body:
        return body.replace("</h1>", "</h1>" + inline, 1)
    return inline + body


def rewrite_links(body: str) -> str:
    """Point links to the repository's documents at the site's pages, the rest at GitHub."""
    targets = {Path(src).name: f"{name}.html" for name, src, _ in PAGES}
    targets["README.md"] = "index.html"

    def fix(m: re.Match[str]) -> str:
        attr, url = m.group(1), m.group(2)
        if url.startswith(("#", "mailto:")):
            return m.group(0)
        path, _, anchor = url.partition("#")
        suffix = f"#{anchor}" if anchor else ""
        blob = f"{REPO}/blob/main/"
        if path.startswith(blob):
            path = path[len(blob) :]
        if path.startswith(("http://", "https://")):
            return m.group(0)
        name = Path(path).name
        if name in targets:
            return f'{attr}="{targets[name]}{suffix}"'
        if "docs/img/" in path or path.startswith("img/"):
            return f'{attr}="img/{name}"'
        clean = re.sub(r"^(\.\./)+", "", path)
        if not clean.startswith(
            ("docs/", "src/", "benchmarks/", "examples/", "tests/", ".github/")
        ):
            clean = f"docs/{clean}" if clean else clean
        return f'{attr}="{blob}{clean}{suffix}"'

    return re.sub(r'(href|src|srcset)="([^"]+)"', fix, body)


def _first_paragraph(text: str) -> str:
    """A plain-text description of a document for its link preview."""
    for block in text.split("\n\n"):
        block = block.strip()
        if block and not block.startswith(("#", "|", "```", "<", "*", "-", ">", "!")):
            plain = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", block)
            plain = re.sub(r"[*_`$\\]", "", plain)
            plain = " ".join(plain.split())
            return plain if len(plain) <= 200 else plain[:197].rsplit(" ", 1)[0] + "..."
    return PITCH


# =============================================================================
# Landing page
# =============================================================================


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _fmt(value: Any, spec: str, suffix: str = "") -> str:
    x = _num(value)
    return "–" if x is None else format(x, spec) + suffix


def _date(value: Any) -> str:
    """'2026-09-24' -> '24 September 2026' (as is if it does not parse)."""
    try:
        d = dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return str(value)
    return f"{d.day} {d:%B %Y}"


def _read_csv(site: Path, name: str) -> Any:
    path = site / name
    if not path.exists():
        return None
    import pandas as pd

    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:  # a malformed file should not take the site down
        return None


def _column(frame: Any, *names: str) -> Any:
    """The first of ``names`` that ``frame`` has, as a float series without gaps."""
    if frame is None:
        return None
    for name in names:
        if name in frame.columns:
            s = frame[name].astype(float).dropna()
            if len(s):
                return s
    return None


def sparkline(series: Any, years: int = 5, width: int = 120, height: int = 30) -> str:
    """An inline SVG line of the last ``years`` years, with a dot on the latest value."""
    if series is None or len(series) < 3:
        return ""
    import pandas as pd

    s = series[series.index >= series.index[-1] - pd.DateOffset(years=years)]
    if len(s) < 3:
        return ""
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo or 1.0
    pad = 3.0
    xs = [pad + (width - 2 * pad) * i / (len(s) - 1) for i in range(len(s))]
    ys = [pad + (height - 2 * pad) * (1 - (float(v) - lo) / span) for v in s]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    zero = ""
    if lo < 0 < hi:
        z = pad + (height - 2 * pad) * (1 - (0 - lo) / span)
        zero = (
            f'<line x1="0" x2="{width}" y1="{z:.1f}" y2="{z:.1f}" stroke="var(--rule)" '
            'stroke-width="1" stroke-dasharray="2 2"/>'
        )
    first, last = s.index[0], s.index[-1]
    label = f"{first:%b %Y} to {last:%b %Y}: low {lo:.2f}, high {hi:.2f}, latest {float(s.iloc[-1]):.2f}"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(label)}"><title>{html.escape(label)}</title>{zero}'
        f'<polyline points="{points}" fill="none" stroke="var(--spark)" stroke-width="1.5" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="2.5" fill="var(--spark)"/></svg>'
    )


def _year_change(series: Any, scale: float = 1.0, unit: str = "pp") -> str:
    """Change over the last year, e.g. '+0.42 pp'."""
    if series is None or len(series) < 2:
        return ""
    import pandas as pd

    past = series[series.index <= series.index[-1] - pd.DateOffset(years=1)]
    if not len(past):
        return ""
    d = (float(series.iloc[-1]) - float(past.iloc[-1])) * scale
    if unit == "pp" and abs(d) < 0.005:
        return "unchanged"
    return f"{d:+.2f} {unit}" if unit == "pp" else f"{d:+.0f} {unit}"


def _term_premium_series(site: Path) -> tuple[Any, Any, bool]:
    """(term premium, expected short rate, survey-anchored?) from term_premium.csv."""
    tp = _read_csv(site, "term_premium.csv")
    survey = _column(tp, "term_premium_survey")
    if survey is not None:
        return survey, _column(tp, "expected_short_rate_survey"), True
    return _column(tp, "term_premium"), _column(tp, "expected_short_rate"), False


def ledger(s: dict[str, Any], site: Path) -> str:
    """The latest reading as a table: value, change over a year, five-year sparkline."""
    from nss_engine.models import NSSCurve

    yields = _read_csv(site, "fitted_yields.csv")
    signals = _read_csv(site, "macro_signals.csv")
    be = _read_csv(site, "breakevens.csv")
    tp, esr, survey = _term_premium_series(site)

    rows: list[tuple[str, str, str, Any, float, str]] = []  # name, note, value, series, scale, unit
    y10 = _column(yields, "10Y")
    curve = NSSCurve(**{k: s["latest_curve"][k] for k in NSSCurve.__dataclass_fields__})
    rows.append(
        (
            "10-year Treasury yield",
            "par yield on the fitted curve",
            f"{float(curve.par_yield([10.0])[0]):.2f}%",
            y10,
            1.0,
            "pp",
        )
    )
    slope = _column(signals, "slope")
    regime = s.get("regime", {})
    rows.append(
        (
            f"Curve slope, {regime.get('slope_definition', '10Y − 3M')}",
            f"{regime.get('current', '').lower()} since {_date(regime.get('since', ''))}"
            if regime.get("current")
            else "",
            _fmt(s.get("spreads_latest_pct", {}).get("slope"), "+.2f", " pp"),
            slope,
            1.0,
            "pp",
        )
    )
    if "term_premium" in s:
        t = s["term_premium"]
        plain = t.get("plain_acm", {}).get("latest", {}).get("term_premium")
        note = "anchored to surveys of forecasters" if survey else "ACM model"
        if survey and plain is not None:
            note += f"; plain ACM {float(plain):+.2f}%"
        rows.append(
            (
                "10-year term premium",
                note,
                _fmt(t["latest"]["term_premium"], "+.2f", "%"),
                tp,
                1.0,
                "pp",
            )
        )
        rows.append(
            (
                "Expected short rate, next 10 years",
                "the rest of the 10-year zero yield",
                _fmt(t["latest"]["expected_short_rate"], ".2f", "%"),
                esr,
                1.0,
                "pp",
            )
        )
    if "inflation" in s:
        inf = s["inflation"]["latest"]
        rows.append(
            (
                "10-year breakeven inflation",
                f"real 10-year yield {_fmt(inf.get('real_10y'), '.2f', '%')}",
                _fmt(inf.get("be_10y"), ".2f", "%"),
                _column(be, "be_10y"),
                1.0,
                "pp",
            )
        )
        rows.append(
            (
                "5y5y forward breakeven",
                "expected inflation 5 to 10 years ahead",
                _fmt(inf.get("be_5y5y"), ".2f", "%"),
                _column(be, "be_5y5y"),
                1.0,
                "pp",
            )
        )
    if "recession_model" in s:
        rm = s["recession_model"]
        rt = s.get("recession_real_time_latest") or {}
        fwd = next((v for k, v in rt.items() if k.startswith("near-term")), None)
        note = "probit on the 10Y − 3M spread"
        if _num(fwd) is not None:
            note += f"; forward-spread model {float(fwd):.0%}"
        rows.append(
            (
                f"Recession odds, next {rm['horizon_months']} months",
                note,
                _fmt(rm["latest_probability"] * 100, ".0f", "%"),
                _column(signals, "recession_prob_12m"),
                100.0,
                "pts",
            )
        )

    body = "".join(
        f'<tr><td><span class="name">{html.escape(name)}</span>'
        f'<span class="note">{html.escape(note)}</span></td>'
        f'<td class="value">{html.escape(value)}</td>'
        f'<td class="chg">{html.escape(_year_change(series, scale, unit))}</td>'
        f'<td class="spark">{sparkline(series * scale if series is not None else None)}</td></tr>'
        for name, note, value, series, scale, unit in rows
    )
    return (
        '<table class="ledger"><thead><tr><th>Measure</th><th>Latest</th>'
        '<th class="chg">1-year change</th><th>5 years</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def headline(s: dict[str, Any]) -> str:
    """The latest reading in two or three plain sentences."""
    when = _date(s["as_of"])
    parts = []
    tp = s.get("term_premium", {}).get("latest")
    if tp and _num(tp.get("yield")) is not None:
        parts.append(
            f"On {when} the 10-year Treasury zero-coupon yield was <b>{float(tp['yield']):.2f}%</b>. "
            f"About <b>{float(tp['expected_short_rate']):.2f}%</b> of that is where short-term "
            f"rates are expected to be over the decade, and <b>{float(tp['term_premium']):.2f}%</b> "
            "is term premium, the extra yield investors ask for to tie money up for ten years."
        )
    else:
        parts.append(f"The latest curve is for {when}.")
    inf = s.get("inflation", {}).get("latest")
    if inf and _num(inf.get("be_10y")) is not None:
        parts.append(
            f"TIPS prices imply inflation averaging <b>{float(inf['be_10y']):.2f}%</b> a year over "
            "the same ten years."
        )
    rm = s.get("recession_model")
    if rm and _num(rm.get("latest_probability")) is not None:
        parts.append(
            "The slope of the curve puts the odds of a recession in the next twelve months at "
            f"<b>{float(rm['latest_probability']):.0%}</b>."
        )
    return " ".join(parts)


def _chart(name: str, alt: str, caption: str, img_dir: Path) -> str:
    if not (img_dir / f"{name}.png").exists():
        return ""
    dark = (
        f'<source media="(prefers-color-scheme: dark)" srcset="img/{name}-dark.png">'
        if (img_dir / f"{name}-dark.png").exists()
        else ""
    )
    return (
        f'<figure><picture>{dark}<img src="img/{name}.png" alt="{html.escape(alt)}" '
        f'width="1600" height="768" loading="lazy"></picture>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def track_record(s: dict[str, Any]) -> str:
    """How well the latest run agrees with independent estimates."""
    items = []
    fq = s["fit_quality_bp"]
    items.append(
        f"<li>{s['n_curves']:,} weekly curves since {s['sample_start'][:4]}; the median curve "
        f"misses its quotes by {fq['rmse_median']:.1f} basis points.</li>"
    )
    ref = s.get("reference_curve")
    if ref:
        items.append(
            f"<li>{ref['rmse_bp']:.1f} bp from {html.escape(ref.get('name', 'the reference curve'))} "
            "on zero rates from 1 to 30 years.</li>"
        )
    bench = s.get("term_premium", {}).get("benchmarks", {})
    for key, label in (
        ("ACM + SPF surveys (real time) vs Kim-Wright (Fed Board)", "Survey-anchored"),
        ("ACM on NSS curves (real time) vs Kim-Wright (Fed Board)", "Plain ACM"),
    ):
        if key in bench:
            b = bench[key]
            items.append(
                f"<li>{label} term premium, estimated in real time, against the Fed Board's "
                f"Kim-Wright estimate: correlation {b['corr_level']:.2f}, typical gap "
                f"{b['rmse_bp']:.0f} bp.</li>"
            )
    return "<ul>" + "".join(items) + "</ul>"


NEGATIVES = """
<h2>What didn't work</h2>
<p>These stay in the documentation, because they are as useful to know as the results that held up.</p>
<ul class="negatives">
<li>No forecasting model beat "yields stay where they are" one month ahead. Mixing a
model half-and-half with that guess only ties it at six to twelve months.</li>
<li>The textbook term premium model (ACM), re-estimated each month on the data available then,
jumps around too much to use. Anchoring it to surveys of forecasters fixed most of
that; a bias-corrected version made it worse.</li>
<li>Splitting the slope into expected rates and term premium predicted
recessions <i>worse</i> than the plain slope since 2005. The 2022-24 inversion pushed the real-time
model to 90% and no recession has followed so far.</li>
<li>Version 2.2 said the arbitrage-free model had better-calibrated
forecast intervals. A proper coverage test showed they are too wide, and the claim was withdrawn.</li>
</ul>
<p>The numbers and tests behind each are on the <a href="results.html">results page</a>.</p>
"""

ABOUT = f"""
<h2>Where this came from</h2>
<p>It started as a sophomore-year script: a Nelson-Siegel curve fitted to FRED data and a 3-D
Plotly surface. Each version since has fixed something the previous one got wrong. The biggest was
that FRED's yields are <a href="par-vs-zero.html">par yields, not zero rates</a>, which most
Nelson-Siegel code ignores. The <a href="changelog.html">changelog</a> lists every mistake and
its fix.</p>
<p>Everything runs from a GitHub Action every weekday evening. To run it yourself:</p>
<pre><code>pip install nss-engine
nss-engine run          # FRED data since 1990 -&gt; output/dashboard.html, report, CSVs</code></pre>
<p>No API key needed. The <a href="{REPO}#as-a-library">README</a> shows the Python API.</p>
"""


def landing(s: dict[str, Any], site: Path) -> str:
    img_dir = site / "img"
    charts = [
        _chart(
            "curve",
            "Heatmap of the fitted Treasury zero curve since 1990 and the latest curve",
            "Each column is one week: short maturities at the bottom, 30 years at the top, darker "
            "for higher yields. The pale stretches are the years of zero rates.",
            img_dir,
        ),
        _chart(
            "term_premium",
            "10-year term premium in real time: plain ACM, survey-anchored ACM and Kim-Wright",
            "Each point uses only data published by that date. Plain ACM (blue) swings with every "
            "re-estimation; the survey-anchored version (orange) stays close to the Fed Board's "
            "Kim-Wright estimate (grey).",
            img_dir,
        ),
        _chart(
            "recession",
            "Probability of a recession within 12 months from the yield curve, with NBER recessions",
            "The real-time line is what the model would have said at the time, knowing only the "
            "recessions that had already been dated.",
            img_dir,
        ),
    ]
    return f"""
<p class="dateline">Week of {html.escape(_date(s["as_of"]))} · rebuilt from Federal Reserve data every weekday</p>
<h1>The U.S. Treasury yield curve, taken apart every week since 1990</h1>
<p class="lead">{headline(s)}</p>
<div class="actions">
  <a class="button" href="dashboard.html">Open the interactive dashboard</a>
  <a href="results.html">How accurate is it?</a>
  <a href="{REPO}">Source code</a>
</div>

<h2>Latest reading</h2>
{ledger(s, site)}
<p class="foot">Changes are over the past year; hover a sparkline for its range. Term premium and
expected rate are monthly, the rest weekly.</p>

<h2>Since 1990</h2>
{"".join(charts)}

<h2>How it is checked</h2>
<p>Real markets have no answer key, so each model is first run on a simulated market where the
true curve, term premium and inflation are known, then compared with independent estimates from
the Federal Reserve. From the latest run:</p>
{track_record(s)}
{NEGATIVES}
{ABOUT}
"""


def social_card(s: dict[str, Any], path: Path) -> Path | None:
    """A 1200x630 preview image for links shared on social sites and in chats."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    from nss_engine.models import NSSCurve

    bg, ink, ink2, line = "#16304f", "#ffffff", "#b9c7d8", "#f2c14e"
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
    fig.text(0.06, 0.86, "NSS Yield Curve Engine", color=ink2, fontsize=22, family="serif")
    fig.text(
        0.06, 0.77, "The U.S. Treasury\ncurve, every week\nsince 1990", color=ink, fontsize=38,
        family="serif", weight="bold", va="top", linespacing=1.08,
    )  # fmt: skip
    rows = []
    tp = s.get("term_premium", {}).get("latest")
    if tp:
        rows.append(("10-year term premium", f"{float(tp['term_premium']):+.2f}%"))
    inf = s.get("inflation", {}).get("latest")
    if inf and _num(inf.get("be_10y")) is not None:
        rows.append(("10-year breakeven inflation", f"{float(inf['be_10y']):.2f}%"))
    rm = s.get("recession_model")
    if rm:
        rows.append(("Recession odds, 12 months", f"{float(rm['latest_probability']):.0%}"))
    for i, (label, value) in enumerate(rows):
        y = 0.31 - i * 0.075
        fig.text(0.06, y, label, color=ink2, fontsize=18)
        fig.text(0.50, y, value, color=ink, fontsize=20, weight="bold", ha="right")
    fig.text(0.06, 0.05, f"Data as of {_date(s['as_of'])}", color=ink2, fontsize=15)
    ax = fig.add_axes((0.64, 0.16, 0.31, 0.62), facecolor=bg)
    curve = NSSCurve(**{k: s["latest_curve"][k] for k in NSSCurve.__dataclass_fields__})
    tau = np.linspace(0.08, 30, 200)
    ax.plot(tau, curve.zero(tau), color=line, lw=4, solid_capstyle="round")
    ax.set_xlim(0, 30.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=14)
    ax.set_xticks([0, 10, 20, 30], ["0", "10", "20", "30y"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f%%"))
    ax.set_title("Latest zero curve", color=ink2, fontsize=15, loc="left", pad=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    return path


# =============================================================================
# Dashboard and other pages
# =============================================================================

_DASH_NAV_CSS = """<style>
.sitenav { border-bottom: 1px solid var(--border, rgba(0,0,0,.1)); }
.sitenav .in { max-width: 1180px; margin: 0 auto; padding: 0 16px; height: 48px; display: flex;
  align-items: center; gap: 18px; overflow-x: auto; white-space: nowrap; scrollbar-width: none; }
.sitenav a { color: var(--ink-2, #555); text-decoration: none; font-size: 15px; }
.sitenav a.brand { color: var(--ink, #111); font-weight: 650; margin-right: auto; display: flex;
  align-items: center; gap: 8px; }
.sitenav a.brand img { width: 20px; height: 20px; }
</style>"""


def add_nav_to_dashboard(path: Path) -> bool:
    """Put the site's navigation at the top of the pipeline's dashboard (once)."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if 'class="sitenav"' in text:
        return False
    links = "".join(f'<a href="{name}.html">{html.escape(label)}</a>' for name, _, label in PAGES)
    bar = (
        '<nav class="sitenav"><div class="in"><a class="brand" href="index.html">'
        '<img src="favicon.svg" alt="">NSS Yield Curve Engine</a>'
        f'{links}<a href="{REPO}">GitHub</a></div></nav>'
    )
    desc = (
        "Interactive dashboard of the U.S. Treasury yield curve: fits, term premium, "
        "breakevens, recession odds and forecasts."
    )
    image = f"{SITE}img/social.png"
    head_extra = (
        f'{_DASH_NAV_CSS}\n<link rel="icon" href="favicon.svg" type="image/svg+xml">\n'
        f'<meta name="description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        '<meta property="og:site_name" content="NSS Yield Curve Engine">\n'
        '<meta property="og:title" content="Dashboard · NSS Yield Curve Engine">\n'
        f'<meta property="og:description" content="{desc}">\n'
        f'<meta property="og:url" content="{SITE}dashboard.html">\n'
        f'<meta property="og:image" content="{image}">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:image" content="{image}">\n'
    )
    text = text.replace("</head>", head_extra + "</head>", 1)
    text = re.sub(r"<body([^>]*)>", lambda m: f"<body{m.group(1)}>{bar}", text, count=1)
    path.write_text(text, encoding="utf-8")
    return True


def not_found() -> str:
    """GitHub Pages serves this at any missing path, so links resolve from the site root."""
    text = page(
        "Page not found · NSS Yield Curve Engine",
        "<h1>Page not found</h1><p>That page does not exist (any more). Try the "
        '<a href="index.html">home page</a> or the <a href="dashboard.html">dashboard</a>.</p>',
        "404",
    )
    return text.replace("<head>", f'<head><base href="{SITE}">', 1)


def build(site: Path) -> list[Path]:
    """Write the website into ``site`` (a pipeline output directory)."""
    site.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in (("style.css", CSS), ("favicon.svg", FAVICON), ("404.html", not_found())):
        (site / name).write_text(content, encoding="utf-8")
        written.append(site / name)
    docs_img = ROOT / "docs" / "img"
    if docs_img.exists() and not (site / "img").exists():
        shutil.copytree(docs_img, site / "img")
    summary_path = site / "summary.json"
    as_of = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        as_of = _date(s["as_of"])
        body = landing(s, site)
        card = social_card(s, site / "img" / "social.png")
        if card is not None:
            written.append(card)
    else:
        body = f"<h1>NSS Yield Curve Engine</h1><p class=lead>{html.escape(PITCH)}</p>"
    index = site / "index.html"
    index.write_text(
        page(
            "NSS Yield Curve Engine: the U.S. Treasury curve, weekly since 1990",
            body,
            "index",
            as_of=as_of,
            script=COPY_JS,
        ),
        encoding="utf-8",
    )
    written.append(index)
    for name, src, label in PAGES:
        path = ROOT / src
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        body, has_math, toc = render_document(text)
        contents = toc_html(toc)
        if contents:
            body = with_contents(body, toc)
        wrap = "docwrap has-toc" if contents else "docwrap"
        out = site / f"{name}.html"
        out.write_text(
            page(
                f"{label} · NSS Yield Curve Engine",
                f'<div class="{wrap}">{contents}<div class="doc">{rewrite_links(body)}</div></div>',
                name,
                has_math,
                description=_first_paragraph(text),
                as_of=as_of,
            ),
            encoding="utf-8",
        )
        written.append(out)
    if add_nav_to_dashboard(site / "dashboard.html"):
        written.append(site / "dashboard.html")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site", type=Path, help="pipeline output directory (nss-engine run --out)")
    args = ap.parse_args()
    for path in build(args.site):
        print(path)


if __name__ == "__main__":
    main()
