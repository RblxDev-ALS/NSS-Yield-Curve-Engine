"""Build the project website from a pipeline output directory.

The live workflow runs ``nss-engine run --source fred --figures --out site``
and then ``python website/build.py site``. This adds to ``site/``:

* ``index.html`` - landing page: what the project is, the latest reading
  (from ``summary.json``), the three charts, links;
* ``results.html``, ``methodology.html``, ``par-vs-zero.html``,
  ``changelog.html`` - the Markdown documents rendered to HTML, with the
  equations typeset by MathJax;
* ``style.css``.

The interactive ``dashboard.html``, the charts in ``img/`` and the shields.io
badges in ``badges/`` are already there. Needs ``pip install markdown``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

REPO = "https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine"
ROOT = Path(__file__).resolve().parents[1]

#: (output page, source Markdown, navigation label)
PAGES = (
    ("results", "docs/results.md", "Results"),
    ("methodology", "docs/methodology.md", "Methodology"),
    ("par-vs-zero", "docs/par-vs-zero.md", "Par vs zero"),
    ("changelog", "CHANGELOG.md", "Changelog"),
)

PITCH = (
    "An open-source Python engine that fits the U.S. Treasury yield curve every week since "
    "1990, splits the 10-year yield into expected short rates and a survey-anchored term "
    "premium, reads breakeven inflation off the TIPS curve, and tests every claim out of "
    "sample, including the ones that failed."
)

CSS = """
:root { color-scheme: light;
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6; --code:#f0efec; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; --code:#262624; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.6; }
nav { border-bottom: 1px solid var(--border); background: var(--surface); }
nav .inner { max-width: 1080px; margin: 0 auto; padding: 10px 16px; display:flex; flex-wrap:wrap;
  gap: 4px 18px; align-items: baseline; }
nav .brand { font-weight: 700; color: var(--ink); text-decoration: none; margin-right: 8px; }
nav a { color: var(--ink-2); text-decoration: none; font-size: 15px; }
nav a:hover, nav a.here { color: var(--accent); }
main { max-width: 1080px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 34px; line-height: 1.2; margin: 8px 0 12px; letter-spacing: -0.01em; }
h2 { font-size: 22px; margin: 40px 0 8px; }
h3 { font-size: 18px; margin: 28px 0 6px; }
p, li { max-width: 780px; }
.pitch { font-size: 19px; color: var(--ink-2); max-width: 760px; margin: 0 0 20px; }
.actions { display:flex; flex-wrap: wrap; gap: 10px; margin: 0 0 8px; align-items: center; }
.button { display:inline-block; padding: 9px 16px; border-radius: 8px; background: var(--accent);
  color: #fff; text-decoration: none; font-weight: 600; }
.button.ghost { background: transparent; color: var(--accent); border: 1px solid var(--border); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 14px; }
code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--code); padding: 12px 14px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
  margin: 12px 0 8px; }
.tile { background: var(--surface); border:1px solid var(--border); border-radius: 12px;
  padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink-2); }
.tile .value { font-size: 28px; font-weight: 600; margin-top: 2px; }
.tile .sub { font-size: 12.5px; color: var(--muted); }
.asof { color: var(--muted); font-size: 14px; }
figure { margin: 18px 0 28px; }
figure img { width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--border); }
figcaption { color: var(--ink-2); font-size: 14.5px; margin-top: 6px; max-width: 780px; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums;
  display:block; overflow-x:auto; max-width:100%; margin: 12px 0; }
th, td { padding: 5px 10px; border-bottom: 1px solid var(--grid); text-align: left; }
th { color: var(--ink-2); }
blockquote { margin: 12px 0; padding: 4px 14px; border-left: 3px solid var(--grid); color: var(--ink-2); }
a { color: var(--accent); }
img { max-width: 100%; }
.doc img { border-radius: 8px; }
footer { margin-top: 56px; color: var(--muted); font-size: 13px; }
@media (max-width: 520px) { h1 { font-size: 27px; } .pitch { font-size: 17px; }
  .tiles { grid-template-columns: 1fr 1fr; } .tile .value { font-size: 22px; } }
"""

MATHJAX = """<script>window.MathJax = { tex: { inlineMath: [['\\\\(', '\\\\)']],
  displayMath: [['\\\\[', '\\\\]']] } };</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""


def page(title: str, body: str, here: str, math: bool = False) -> str:
    links = [("index.html", "Home", "index"), ("dashboard.html", "Live dashboard", "dashboard")]
    links += [(f"{name}.html", label, name) for name, _, label in PAGES]
    nav = "".join(
        f'<a href="{href}"{" class=here" if key == here else ""}>{html.escape(label)}</a>'
        for href, label, key in links
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(PITCH)}">
<link rel="stylesheet" href="style.css">
{MATHJAX if math else ""}
</head><body>
<nav><div class="inner"><a class="brand" href="index.html">NSS Yield Curve Engine</a>{nav}
<a href="{REPO}">GitHub</a></div></nav>
<main>{body}
<footer>Built by <a href="{REPO}">nss-engine</a> from public data: FRED (Federal Reserve Bank
of St. Louis), the Federal Reserve Board, the Philadelphia Fed's Survey of Professional
Forecasters and NBER. Educational project, not investment advice.</footer>
</main></body></html>
"""


# =============================================================================
# Markdown documents
# =============================================================================

_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE = re.compile(r"(?<![\\$\w])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])")
_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def render_markdown(text: str) -> tuple[str, bool]:
    """Markdown to HTML with ``$…$`` / ``$$…$$`` kept intact for MathJax."""
    import markdown

    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    # code blocks first, so a $ inside code is not read as math
    text = _FENCE.sub(lambda m: keep(m.group(0)), text)
    math: list[str] = []

    def put_math(m: re.Match[str], display: bool) -> str:
        tex = html.escape(m.group(1).strip())
        math.append(f"\\[{tex}\\]" if display else f"\\({tex}\\)")
        return f"MATHPLACEHOLDER{len(math) - 1}X"

    text = _DISPLAY.sub(lambda m: put_math(m, True), text)
    text = _INLINE.sub(lambda m: put_math(m, False), text)
    text = re.sub("\x00(\\d+)\x00", lambda m: stash[int(m.group(1))], text)
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    out = re.sub(r"MATHPLACEHOLDER(\d+)X", lambda m: math[int(m.group(1))], out)
    return out, bool(math)


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


# =============================================================================
# Landing page
# =============================================================================


def _fmt(value: Any, spec: str, suffix: str = "") -> str:
    try:
        return format(float(value), spec) + suffix
    except (TypeError, ValueError):
        return "–"


def _tile(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>'
    )


def latest_tiles(s: dict[str, Any]) -> str:
    """The "latest reading" panel from ``summary.json``."""
    from nss_engine.models import NSSCurve

    tiles = []
    curve = NSSCurve(**{k: s["latest_curve"][k] for k in NSSCurve.__dataclass_fields__})
    y10 = float(curve.par_yield([10.0])[0])
    tiles.append(_tile("10-year Treasury yield", f"{y10:.2f}%", "NSS fit, par"))
    slope = s.get("spreads_latest_pct", {}).get("slope")
    tiles.append(
        _tile(
            f"Curve slope ({s['regime']['slope_definition']})",
            _fmt(slope, "+.2f", " pp"),
            f"{s['regime']['current']} since {s['regime']['since']}",
        )
    )
    if "recession_model" in s:
        rm = s["recession_model"]
        sub = "NY Fed-style probit, 10Y−3M"
        rt = s.get("recession_real_time_latest") or {}
        fwd = next((v for k, v in rt.items() if k.startswith("near-term")), None)
        if fwd is not None:
            sub = f"real-time forward-spread model: {float(fwd):.0%}"
        tiles.append(
            _tile(
                f"Recession odds, next {rm['horizon_months']} months",
                _fmt(rm["latest_probability"] * 100, ".0f", "%"),
                sub,
            )
        )
    if "term_premium" in s:
        tp = s["term_premium"]
        sub = tp.get("method", "")
        plain = tp.get("plain_acm", {}).get("latest", {}).get("term_premium")
        if tp.get("method", "").startswith("survey") and plain is not None:
            sub = f"survey-anchored · plain ACM {float(plain):+.2f}%"
        tiles.append(
            _tile("10-year term premium", _fmt(tp["latest"]["term_premium"], "+.2f", "%"), sub)
        )
        tiles.append(
            _tile(
                "Expected short rate, next 10 years",
                _fmt(tp["latest"]["expected_short_rate"], ".2f", "%"),
                "the part of the 10-year yield that is not premium",
            )
        )
    if "inflation" in s:
        inf = s["inflation"]["latest"]
        tiles.append(
            _tile(
                "10-year breakeven inflation",
                _fmt(inf["be_10y"], ".2f", "%"),
                f"5y5y forward {_fmt(inf['be_5y5y'], '.2f', '%')} · real 10Y {_fmt(inf['real_10y'], '.2f', '%')}",
            )
        )
    return '<div class="tiles">' + "".join(tiles) + "</div>"


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
        f'loading="lazy"></picture><figcaption>{caption}</figcaption></figure>'
    )


def findings(s: dict[str, Any]) -> str:
    """A few live statistics from the latest run."""
    items = []
    fq = s["fit_quality_bp"]
    items.append(
        f"<li><strong>{s['n_curves']:,} weekly curves</strong> since {s['sample_start'][:4]}, "
        f"median fit error <strong>{fq['rmse_median']:.1f} bp</strong>.</li>"
    )
    ref = s.get("reference_curve")
    if ref:
        items.append(
            f"<li>Against the Federal Reserve's own curve: <strong>{ref['rmse_bp']:.1f} bp</strong> "
            "RMSE on zero rates, 1-30 years.</li>"
        )
    bench = s.get("term_premium", {}).get("benchmarks", {})
    for key, label in (
        ("ACM + SPF surveys (real time) vs Kim-Wright (Fed Board)", "survey-anchored"),
        ("ACM on NSS curves (real time) vs Kim-Wright (Fed Board)", "plain ACM"),
    ):
        if key in bench:
            b = bench[key]
            items.append(
                f"<li>Real-time 10-year term premium, {label}, vs the Fed Board's Kim-Wright: "
                f"correlation <strong>{b['corr_level']:.2f}</strong>, RMSE "
                f"<strong>{b['rmse_bp']:.0f} bp</strong>.</li>"
            )
    return "<ul>" + "".join(items) + "</ul>"


def landing(s: dict[str, Any], img_dir: Path) -> str:
    charts = "".join(
        [
            _chart(
                "curve",
                "Heatmap of the fitted Treasury zero curve since 1990 and the latest curve",
                "Every week since 1990, compressed: short rates at the bottom, 30 years at the top. "
                "Pale bands are the zero-rate years; the right panel is the latest curve and its quotes.",
                img_dir,
            ),
            _chart(
                "term_premium",
                "10-year term premium in real time: plain ACM, survey-anchored ACM and Kim-Wright",
                "The 10-year term premium as it would have been estimated at each date. Plain ACM "
                "swings with every re-estimation; anchoring expectations to the Survey of "
                "Professional Forecasters makes it stable and close to the Fed Board's estimate.",
                img_dir,
            ),
            _chart(
                "recession",
                "Recession probability from the yield curve vs NBER recessions",
                "Probability of a recession within 12 months. The real-time line is what the model "
                "would have said at each date, using only recessions known by then.",
                img_dir,
            ),
        ]
    )
    return f"""
<h1>What the Treasury curve says, measured honestly</h1>
<p class="pitch">{html.escape(PITCH)}</p>
<div class="actions">
  <a class="button" href="dashboard.html">Open the live dashboard</a>
  <a class="button ghost" href="results.html">Read the results</a>
  <a class="button ghost" href="{REPO}">Source on GitHub</a>
  <code>pip install nss-engine</code>
</div>
<h2>Latest reading</h2>
<p class="asof">Data as of {html.escape(str(s["as_of"]))}; rebuilt every weekday from FRED.</p>
{latest_tiles(s)}
{charts}
<h2>From the latest run</h2>
{findings(s)}
<p>Every number above is recomputed on each run. The <a href="results.html">results page</a>
has the full tests, including what did not work: the random walk still beats every forecasting
model, and plain ACM's real-time term premium is too unstable to use.</p>
"""


def build(site: Path) -> list[Path]:
    """Write the website into ``site`` (a pipeline output directory)."""
    site.mkdir(parents=True, exist_ok=True)
    written = []
    (site / "style.css").write_text(CSS, encoding="utf-8")
    written.append(site / "style.css")
    summary_path = site / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        body = landing(s, site / "img")
    else:
        body = f"<h1>NSS Yield Curve Engine</h1><p class=pitch>{html.escape(PITCH)}</p>"
    index = site / "index.html"
    index.write_text(page("NSS Yield Curve Engine", body, "index"), encoding="utf-8")
    written.append(index)
    for name, src, label in PAGES:
        path = ROOT / src
        if not path.exists():
            continue
        body, math = render_markdown(path.read_text(encoding="utf-8"))
        out = site / f"{name}.html"
        out.write_text(
            page(
                f"{label} · NSS Yield Curve Engine",
                f'<div class="doc">{rewrite_links(body)}</div>',
                name,
                math,
            ),
            encoding="utf-8",
        )
        written.append(out)
    docs_img = ROOT / "docs" / "img"
    if docs_img.exists() and not (site / "img").exists():
        shutil.copytree(docs_img, site / "img")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site", type=Path, help="pipeline output directory (nss-engine run --out)")
    args = ap.parse_args()
    for path in build(args.site):
        print(path)


if __name__ == "__main__":
    main()
