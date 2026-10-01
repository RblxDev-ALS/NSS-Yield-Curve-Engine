import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("markdown")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_build", ROOT / "website" / "build.py")
assert spec is not None and spec.loader is not None
site_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site_build)


def test_math_survives_markdown():
    html, has_math = site_build.render_markdown(
        "Loading $x_1 * y_2$ and\n\n$$a_{i} = b_{j}$$\n\n```\ncost = $5 and $6\n```\n"
    )
    assert has_math
    assert r"\(x_1 * y_2\)" in html and r"\[a_{i} = b_{j}\]" in html
    assert "<em>" not in html  # underscores and asterisks were not read as emphasis
    assert "cost = $5 and $6" in html  # code is left alone


def test_links_point_at_site_pages_or_github():
    body = (
        '<a href="methodology.md#6-term-premium">m</a> <a href="../README.md">r</a> '
        '<a href="https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/par-vs-zero.md">p</a> '
        '<a href="../benchmarks/real_data_studies.py">b</a> <img src="docs/img/curve.png"> '
        '<a href="#top">t</a>'
    )
    out = site_build.rewrite_links(body)
    assert 'href="methodology.html#6-term-premium"' in out
    assert 'href="index.html"' in out and 'href="par-vs-zero.html"' in out
    assert "blob/main/benchmarks/real_data_studies.py" in out
    assert 'src="img/curve.png"' in out and 'href="#top"' in out


def test_build_site(tmp_path):
    from nss_engine.pipeline import PipelineConfig, run_pipeline, write_outputs
    from nss_engine.synthetic import simulate_market

    mkt = simulate_market(start="2000-01-07", periods=52 * 8, seed=1)
    monthly = mkt.yields.resample("ME").last()
    truth = mkt.true_params.resample("ME").last()
    cfg = PipelineConfig(source="synthetic", freq="ME", run_forecasts=False)
    result = run_pipeline(cfg, data=(monthly, mkt.recession, truth))

    write_outputs(result, tmp_path, dashboard=True)
    written = site_build.build(tmp_path)
    names = {p.name for p in written}
    assert {"index.html", "methodology.html", "results.html", "style.css", "favicon.svg"} <= names
    assert {"404.html", "dashboard.html"} <= names
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    s = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "Latest reading" in index and site_build._date(s["as_of"]) in index
    assert "10-year term premium" in index and "breakeven" in index
    assert index.count("<polyline") >= 5  # sparklines from the CSVs
    assert 'property="og:image"' in index and "favicon.svg" in index
    methodology = (tmp_path / "methodology.html").read_text(encoding="utf-8")
    assert "MathJax" in methodology and "On this page" in methodology
    dashboard = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert dashboard.count('class="sitenav"') == 1
    for tag in ("og:title", "og:description", "og:url", "og:image", "twitter:image"):
        assert tag in dashboard  # link previews of the dashboard
    assert 'content="https://' in dashboard.split('property="og:url"')[1].split(">")[0]
    assert "_phone" in dashboard  # phone layout for the charts (legends under the plot)
    # wide equations and code scroll or wrap instead of widening the phone page
    css = (tmp_path / "style.css").read_text(encoding="utf-8")
    assert 'mjx-container[display="true"]' in css and "overflow-wrap: anywhere" in css
    site_build.build(tmp_path)  # rebuilding does not add a second navigation bar
    dashboard = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert dashboard.count('class="sitenav"') == 1


def test_sparkline_and_year_change():
    import pandas as pd

    idx = pd.date_range("2018-01-05", periods=400, freq="W-FRI")
    s = pd.Series(range(400), index=idx, dtype=float) / 100 - 3.5
    svg = site_build.sparkline(s)
    assert svg.startswith("<svg") and "stroke-dasharray" in svg  # crosses zero: zero line drawn
    assert site_build._year_change(s) == "+0.53 pp"  # 53 weeks back to a year ago
    assert site_build.sparkline(s.iloc[:2]) == ""


def test_headline_reads_the_summary():
    s = {
        "as_of": "2026-09-24",
        "term_premium": {
            "latest": {"yield": 5.18, "expected_short_rate": 4.02, "term_premium": 1.16}
        },
        "inflation": {"latest": {"be_10y": 2.33}},
        "recession_model": {"latest_probability": 0.08},
    }
    text = site_build.headline(s)
    assert "24 September 2026" in text and "<b>5.18%</b>" in text and "<b>1.16%</b>" in text
    assert "<b>2.33%</b>" in text and "<b>8%</b>" in text
