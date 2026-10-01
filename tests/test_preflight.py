import datetime as dt
import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts" / "preflight.py")
assert spec is not None and spec.loader is not None
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)

PAGE = """<!doctype html><html><head><title>t</title>
<meta name="description" content="d">
<meta property="og:title" content="Title &amp; more">
<meta property="og:description" content="Desc">
<meta property="og:url" content="https://example.org/">
<meta property="og:image" content="https://example.org/img/social.png">
<meta name="twitter:card" content="summary_large_image">
<meta property="og:image" content="https://example.org/second.png">
</head><body></body></html>"""


def png_header(width, height):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\0\0\0\0"


def test_parse_meta_and_complete_page():
    meta = preflight.parse_meta(PAGE)
    assert meta["og:title"] == "Title & more"  # entities decoded
    assert meta["og:image"].endswith("img/social.png")  # first tag wins
    assert meta["twitter:card"] == "summary_large_image"
    assert preflight.missing_meta(meta) == []


def test_missing_meta_reports_gaps_and_relative_urls():
    meta = preflight.parse_meta(
        '<meta property="og:title" content="x"><meta property="og:image" content="img/social.png">'
        '<meta property="og:url" content="  ">'
    )
    problems = preflight.missing_meta(meta)
    assert "og:description" in problems and "twitter:card" in problems and "og:url" in problems
    assert any(p.startswith("og:image") and "absolute" in p for p in problems)


def test_site_head_meta_passes_its_own_check():
    # the real tags written by website/build.py satisfy the checker
    pytest.importorskip("markdown")
    bspec = importlib.util.spec_from_file_location("site_build", ROOT / "website" / "build.py")
    assert bspec is not None and bspec.loader is not None
    build = importlib.util.module_from_spec(bspec)
    bspec.loader.exec_module(build)
    meta = preflight.parse_meta(build.head_meta("T", "D", "index.html"))
    assert preflight.missing_meta(meta) == []


def test_png_size():
    assert preflight.png_size(png_header(1200, 630)) == (1200, 630)
    assert preflight.png_size(png_header(1200, 630)[:24]) == (1200, 630)
    assert preflight.png_size(png_header(1200, 630)[:23]) is None
    assert preflight.png_size(b"GIF89a" + b"\0" * 40) is None
    assert preflight.png_size(b"") is None


def _badge(message, **kw):
    return json.dumps({"schemaVersion": 1, "label": "data as of", "message": message, **kw})


def test_shields_date_formats():
    assert preflight.shields_date(_badge("2026-09-24")) == (dt.date(2026, 9, 24), "data as of")
    assert preflight.shields_date(_badge("2026-09-24 00:00:00"))[0] == dt.date(2026, 9, 24)
    with pytest.raises(ValueError, match="ISO date"):
        preflight.shields_date(_badge("24 September 2026"))
    with pytest.raises(ValueError, match="valid JSON"):
        preflight.shields_date("<html>404</html>")
    with pytest.raises(ValueError, match="schemaVersion"):
        preflight.shields_date(json.dumps({"label": "a", "message": "2026-09-24"}))
    with pytest.raises(ValueError, match="label"):
        preflight.shields_date(json.dumps({"schemaVersion": 1, "message": "2026-09-24"}))


def test_shields_date_matches_what_pipeline_writes():
    pd = pytest.importorskip("pandas")
    assert preflight.shields_date(_badge(str(pd.Timestamp("2026-09-24"))))[0] == dt.date(
        2026, 9, 24
    )


def test_age_days():
    assert preflight.age_days(dt.date(2026, 9, 24), dt.date(2026, 10, 1)) == 7


README = """# T
[![CI](https://github.com/o/r/actions/workflows/ci.yml/badge.svg)](https://github.com/o/r)
[Site](https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/) and
[dash](https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/dashboard.html#top)
[results](docs/results.md "Results") [lic](LICENSE) [anchor](#top) [mail](mailto:a@b.c)
[spaced](docs/my%20file.md#x)
<p align="center"><img src="docs/img/curve.png" alt="c"></p>
<a href="https://example.org/x">x</a>
[ref]: docs/ref.md
```
[ignored](nope/in/code.md)
```
Inline `[also](ignored.md)` code.
[![e](https://img.shields.io/endpoint?url=https%3A%2F%2Frblxdev-als.github.io%2Fx.json)](docs/results.md)
"""


def test_readme_links_and_split():
    targets = preflight.readme_links(README)
    assert "nope/in/code.md" not in targets and "ignored.md" not in targets
    assert targets.count("docs/results.md") == 1  # title stripped; duplicates removed
    rel, site = preflight.split_links(targets)
    assert rel == [
        "docs/results.md",
        "LICENSE",
        "docs/my file.md",
        "docs/img/curve.png",
        "docs/ref.md",
    ]
    assert site == [
        "https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/",
        "https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/dashboard.html#top",
    ]


def test_missing_relative(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x")
    missing = preflight.missing_relative(["docs/a.md", "docs/b.md", "/docs/a.md"], tmp_path)
    assert missing == ["docs/b.md"]


def test_real_readme_relative_links_exist():
    # same code path as `preflight.py --offline`, run on this repository
    result = preflight.check_readme_offline(ROOT)[0]
    assert result.status == preflight.PASS, result.message


CFF_HANDLE = """title: "X"
authors:
  - alias: RblxDev-ALS
repository-code: "https://github.com/RblxDev-ALS/X"
"""
CFF_REAL = CFF_HANDLE.replace(
    "  - alias: RblxDev-ALS", "  - family-names: Doe\n    given-names: Jane\n    alias: RblxDev-ALS"
)


def test_cff_authors():
    assert preflight.cff_authors_are_handle(CFF_HANDLE)
    assert not preflight.cff_authors_are_handle(CFF_REAL)
    assert not preflight.cff_authors_are_handle("title: x\n")  # no authors block


def test_build_author():
    assert preflight.build_author_is_handle('X = 1\nAUTHOR = "RblxDev-ALS"\n') is True
    assert preflight.build_author_is_handle("AUTHOR = 'Jane Doe'\n") is False
    assert preflight.build_author_is_handle("nothing\n") is None


def test_pyproject_authors():
    assert preflight.pyproject_authors_are_handle({"project": {"name": "x"}}) is None
    handle_only = {"project": {"authors": [{"name": "RblxDev-ALS"}]}}
    assert preflight.pyproject_authors_are_handle(handle_only)
    named = {"project": {"authors": [{"name": "Jane Doe"}]}}
    assert not preflight.pyproject_authors_are_handle(named)


def test_check_authors_in_tmp_repo(tmp_path):
    (tmp_path / "website").mkdir()
    (tmp_path / "CITATION.cff").write_text(CFF_HANDLE)
    (tmp_path / "website" / "build.py").write_text('AUTHOR = "RblxDev-ALS"\n')
    py = {"project": {"authors": [{"name": "RblxDev-ALS"}]}}
    [warn] = preflight.check_authors(tmp_path, py)
    assert warn.status == preflight.WARN
    for place in ("CITATION.cff", "website/build.py", "pyproject.toml"):
        assert place in warn.message
    (tmp_path / "CITATION.cff").write_text(CFF_REAL)
    (tmp_path / "website" / "build.py").write_text('AUTHOR = "Jane Doe"\n')
    [ok] = preflight.check_authors(tmp_path, None)
    assert ok.status == preflight.PASS


def test_version_key():
    assert preflight.version_key("2.10.0") > preflight.version_key("2.4.0")
    assert preflight.version_key("2.4.0rc1") == (2, 4, 0)


def test_repo_metadata_results():
    good = {"homepage": "https://x/", "description": "fine"}
    assert [r.status for r in preflight.repo_metadata_results(good)] == [preflight.PASS]
    bad = {"homepage": "", "description": "The hidden heartbeat of bonds"}
    results = preflight.repo_metadata_results(bad)
    assert [r.status for r in results] == [preflight.WARN, preflight.WARN]
    assert preflight.SUGGESTED_DESCRIPTION in results[1].hint


def test_network_errors_do_not_raise(monkeypatch):
    def boom(url, max_bytes=None):
        raise preflight.FetchError("URLError: no network")

    monkeypatch.setattr(preflight, "fetch", boom)
    for results in (
        preflight.check_website(),
        preflight.check_freshness(),
        preflight.check_pages(),
        preflight.check_github_repo(),
        preflight.check_pypi({"project": {"version": "2.4.0"}}, ""),
    ):
        assert results and all(r.status in (preflight.WARN, preflight.FAIL) for r in results)
        assert all("no network" in r.message for r in results)


def test_github_rate_limit_is_a_warning(monkeypatch):
    monkeypatch.setattr(preflight, "fetch", lambda url, max_bytes=None: (403, {}, b""))
    [r] = preflight.check_github_repo()
    assert r.status == preflight.WARN and "rate limit" in r.message


def test_pypi_statuses(monkeypatch):
    def reply(status, body=b""):
        monkeypatch.setattr(preflight, "fetch", lambda url, max_bytes=None: (status, {}, body))

    py = {"project": {"version": "2.4.0"}}
    reply(404)
    [r] = preflight.check_pypi(py, "")
    assert r.status == preflight.FAIL and "docs/releasing.md" in r.hint
    reply(200, json.dumps({"info": {"version": "2.3.0"}}).encode())
    assert preflight.check_pypi(py, "")[0].status == preflight.WARN
    reply(200, json.dumps({"info": {"version": "2.4.0"}}).encode())
    assert preflight.check_pypi(py, "")[0].status == preflight.PASS


def test_image_checks(monkeypatch):
    def reply(status, ctype, body):
        monkeypatch.setattr(
            preflight, "fetch", lambda url, max_bytes=None: (status, {"content-type": ctype}, body)
        )

    reply(200, "image/png", png_header(1200, 630))
    assert preflight.check_image("https://x/i.png").status == preflight.PASS
    reply(200, "image/png", png_header(300, 157))
    assert preflight.check_image("https://x/i.png").status == preflight.WARN
    reply(200, "text/html", b"<html>")
    assert preflight.check_image("https://x/i.png").status == preflight.FAIL
    reply(404, "", b"")
    assert preflight.check_image("https://x/i.png").status == preflight.FAIL


def test_offline_run_does_not_touch_the_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network used in --offline mode")

    monkeypatch.setattr(preflight, "fetch", boom)
    results = preflight.run(offline=True, root=ROOT)
    assert results and not any(r.status == preflight.FAIL for r in results)
