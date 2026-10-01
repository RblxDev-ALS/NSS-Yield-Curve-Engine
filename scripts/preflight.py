#!/usr/bin/env python
"""Launch preflight: check that the project is ready to be posted publicly.

Run it on your own machine before posting::

    python scripts/preflight.py            # all checks (needs internet)
    python scripts/preflight.py --offline  # only the checks that read this repository

It prints one line per check (PASS / WARN / FAIL, with a fix hint) and exits
non-zero if any check FAILs. Standard library only.

The functions that look at text or bytes (``parse_meta``, ``png_size``,
``shields_date``, ``readme_links``, ``cff_authors_are_handle``, ...) do no I/O,
so the tests can run them without a network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, NamedTuple

try:
    import tomllib
except ImportError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "nss-engine"
HANDLE = "RblxDev-ALS"
REPO_SLUG = "RblxDev-ALS/NSS-Yield-Curve-Engine"
SITE = "https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/"
PYPI_JSON = f"https://pypi.org/pypi/{PACKAGE}/json"
GITHUB_API = f"https://api.github.com/repos/{REPO_SLUG}"
SUGGESTED_DESCRIPTION = (
    "The U.S. Treasury yield curve fitted weekly since 1990: term premium, breakeven "
    "inflation and recession odds, tested against the Fed's own numbers."
)
REQUIRED_META = ("og:title", "og:description", "og:image", "og:url", "twitter:card")
PAGES = ("dashboard.html", "results.html", "methodology.html")
MIN_IMAGE_WIDTH = 600
MAX_AGE_DAYS = 7
TIMEOUT = 15
USER_AGENT = "nss-engine-preflight/1.0 (+https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine)"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Result(NamedTuple):
    status: str
    name: str
    message: str
    hint: str = ""

    def line(self) -> str:
        text = f"{self.status:<4}  {self.name}: {self.message}"
        return f"{text}\n      fix: {self.hint}" if self.hint and self.status != PASS else text


# --------------------------------------------------------------------------- #
# Pure helpers (strings / bytes in, values out; no I/O)
# --------------------------------------------------------------------------- #


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        a = {k.lower(): v for k, v in attrs if v is not None}
        key = a.get("property") or a.get("name")
        if key and "content" in a:
            self.meta.setdefault(key.lower(), a["content"])  # first tag wins, as in crawlers


def parse_meta(html_text: str) -> dict[str, str]:
    """``{'og:title': ..., 'twitter:card': ...}`` from the ``<meta>`` tags of a page."""
    parser = _MetaParser()
    parser.feed(html_text)
    return parser.meta


def missing_meta(meta: dict[str, str]) -> list[str]:
    """Required link-preview tags that are absent or empty, or og:image / og:url not absolute."""
    problems = [k for k in REQUIRED_META if not meta.get(k, "").strip()]
    for key in ("og:image", "og:url"):
        value = meta.get(key, "").strip()
        if value and not value.lower().startswith(("http://", "https://")):
            problems.append(f"{key} (not an absolute URL: {value})")
    return problems


def png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR chunk, or None if ``data`` is not a PNG header."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def shields_date(text: str) -> tuple[dt.date, str]:
    """(date, label) from a shields.io endpoint JSON whose message starts with ISO date.

    Raises ValueError if it is not valid JSON, not a shields.io endpoint
    (``schemaVersion`` 1 with ``label`` and ``message``) or the message has no date.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc})") from exc
    if not isinstance(obj, dict) or obj.get("schemaVersion") != 1:
        raise ValueError("not a shields.io endpoint (schemaVersion 1 missing)")
    for key in ("label", "message"):
        if not isinstance(obj.get(key), str) or not obj[key]:
            raise ValueError(f"shields.io endpoint has no '{key}'")
    # pipeline.badges() writes str(pandas.Timestamp): "2026-09-24" or "2026-09-24 00:00:00"
    try:
        return dt.date.fromisoformat(obj["message"].strip()[:10]), obj["label"]
    except ValueError as exc:
        raise ValueError(f"message {obj['message']!r} does not start with an ISO date") from exc


def age_days(day: dt.date, today: dt.date) -> int:
    return (today - day).days


def _strip_code(markdown: str) -> str:
    markdown = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", markdown, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", markdown)


def readme_links(markdown: str) -> list[str]:
    """Every link and image target in Markdown / inline HTML, outside code, in order, unique."""
    text = _strip_code(markdown)
    found: list[str] = []
    for m in re.finditer(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[\"'][^)]*[\"'])?\s*\)", text):
        found.append(m.group(1))
    for m in re.finditer(r"""\b(?:src|href|srcset)\s*=\s*["']([^"']+)["']""", text):
        found.append(m.group(1).split()[0])
    for m in re.finditer(r"^\s*\[[^\]]+\]:\s*(\S+)", text, flags=re.M):
        found.append(m.group(1))
    return list(dict.fromkeys(found))


def split_links(targets: list[str], site: str = SITE) -> tuple[list[str], list[str]]:
    """(repo-relative paths, absolute URLs under ``site``); other external links are ignored."""
    relative: list[str] = []
    on_site: list[str] = []
    for t in targets:
        if t.startswith("#") or t.lower().startswith(("mailto:", "tel:", "data:", "javascript:")):
            continue
        if t.lower().startswith(("http://", "https://")):
            if t.startswith(site):
                on_site.append(t)
        elif t.startswith("//"):
            continue
        else:
            path = urllib.parse.unquote(re.split(r"[#?]", t, maxsplit=1)[0])
            if path:
                relative.append(path)
    return list(dict.fromkeys(relative)), list(dict.fromkeys(on_site))


def missing_relative(paths: list[str], root: Path) -> list[str]:
    return [p for p in paths if not (root / p.lstrip("/")).exists()]


def _is_handle(value: str) -> bool:
    return value.strip().strip("\"'").lower() == HANDLE.lower()


def cff_authors_are_handle(text: str) -> bool:
    """True if CITATION.cff lists authors and none has a name beyond the GitHub handle."""
    block = re.search(r"^authors:\s*\n((?:[ \t-].*\n?|\n)*)", text, flags=re.M)
    if not block:
        return False
    values = re.findall(
        r"^[ \t-]*(family-names|given-names|name|alias):\s*(.+?)\s*$", block.group(1), flags=re.M
    )
    if not values:
        return False
    return all(_is_handle(v) for _, v in values)


def build_author_is_handle(source: str) -> bool | None:
    """True if ``AUTHOR = "..."`` in website/build.py is the handle; None if not found."""
    m = re.search(r"^AUTHOR\s*=\s*([\"'])(.*?)\1", source, flags=re.M)
    return None if not m else _is_handle(m.group(2))


def pyproject_authors_are_handle(data: dict[str, Any]) -> bool | None:
    """True if ``project.authors`` has only the handle; None if there is no such field."""
    authors = data.get("project", {}).get("authors")
    if not authors:
        return None
    return all(_is_handle(str(a.get("name", ""))) for a in authors)


def version_key(version: str) -> tuple[int, ...]:
    """Sortable key for 'X.Y.Z' versions; pre-release suffixes are dropped."""
    return tuple(
        int(n) for n in re.findall(r"\d+", re.split(r"[a-zA-Z+-]", version, maxsplit=1)[0])
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


class FetchError(Exception):
    """Network-level failure (DNS, TLS, timeout, ...), not an HTTP status."""


def fetch(url: str, max_bytes: int | None = None) -> tuple[int, dict[str, str], bytes]:
    """GET ``url``: (status, lower-cased headers, up to ``max_bytes`` of body).

    HTTP error statuses are returned, not raised. Network errors raise FetchError.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read() if max_bytes is None else resp.read(max_bytes)
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, body
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return exc.code, headers, b""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        reason = getattr(exc, "reason", exc)
        raise FetchError(f"{type(exc).__name__}: {reason}") from exc


def _net_fail(name: str, url: str, exc: Exception, status: str = FAIL) -> Result:
    return Result(status, name, f"could not reach {url}: {exc}", "check your connection, retry")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def read_pyproject(root: Path = ROOT) -> tuple[dict[str, Any] | None, str]:
    """(parsed pyproject.toml, '') or (None, reason)."""
    if tomllib is None:
        return None, "Python 3.10 has no tomllib; run this script with Python 3.11 or newer"
    try:
        return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")), ""
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return None, f"cannot read pyproject.toml: {exc}"


def check_pypi(pyproject: dict[str, Any] | None, why: str) -> list[Result]:
    name = "pypi"
    if pyproject is None:
        return [Result(WARN, name, why, "run with Python 3.11+")]
    local = str(pyproject["project"]["version"])
    try:
        status, _, body = fetch(PYPI_JSON)
    except FetchError as exc:
        return [_net_fail(name, PYPI_JSON, exc)]
    if status == 404:
        return [
            Result(
                FAIL,
                name,
                f"{PACKAGE} is not on PyPI yet",
                "follow docs/releasing.md (trusted publisher, then push tag v" + local + ")",
            )
        ]
    if status != 200:
        return [Result(WARN, name, f"PyPI answered HTTP {status}", "retry in a minute")]
    try:
        latest = str(json.loads(body)["info"]["version"])
    except (ValueError, KeyError, TypeError) as exc:
        return [Result(WARN, name, f"unexpected PyPI response ({exc})", "retry in a minute")]
    if version_key(latest) == version_key(local):
        return [Result(PASS, name, f"latest release {latest} matches pyproject.toml")]
    if version_key(latest) < version_key(local):
        return [
            Result(
                WARN,
                name,
                f"PyPI has {latest} but pyproject.toml says {local}",
                f"tag v{local} and push it (docs/releasing.md), or lower the version",
            )
        ]
    return [
        Result(
            WARN,
            name,
            f"PyPI has {latest}, newer than pyproject.toml ({local})",
            "pull main; your checkout is behind",
        )
    ]


def check_website(site: str = SITE) -> list[Result]:
    name = "website"
    try:
        status, _, body = fetch(site)
    except FetchError as exc:
        return [_net_fail(name, site, exc)]
    if status != 200:
        return [
            Result(
                FAIL,
                name,
                f"{site} returned HTTP {status}",
                "run the 'Live dashboard' workflow and check Settings > Pages",
            )
        ]
    results = [Result(PASS, name, f"{site} returns 200")]
    meta = parse_meta(body.decode("utf-8", errors="replace"))
    problems = missing_meta(meta)
    if problems:
        results.append(
            Result(
                FAIL,
                "link-preview",
                "missing or invalid: " + ", ".join(problems),
                "fix head_meta() in website/build.py and redeploy",
            )
        )
    else:
        results.append(Result(PASS, "link-preview", "og:title/description/image/url, twitter:card"))
    image = meta.get("og:image", "").strip()
    if image.lower().startswith(("http://", "https://")):
        results.append(check_image(image))
    return results


def check_image(url: str) -> Result:
    name = "og:image"
    try:
        status, headers, head = fetch(url, max_bytes=64)
    except FetchError as exc:
        return _net_fail(name, url, exc)
    hint = "social_card() in website/build.py needs matplotlib; rerun the live workflow"
    if status != 200:
        return Result(FAIL, name, f"{url} returns HTTP {status}", hint)
    ctype = headers.get("content-type", "").split(";")[0].strip().lower()
    if not ctype.startswith("image/"):
        return Result(FAIL, name, f"content type is {ctype or 'missing'}, not an image", hint)
    size = png_size(head)
    if size is None:
        if ctype == "image/png":
            return Result(WARN, name, "served as PNG but the header is unreadable", hint)
        return Result(PASS, name, f"{ctype}, 200 (width not checked: not a PNG)")
    if size[0] < MIN_IMAGE_WIDTH:
        return Result(
            WARN,
            name,
            f"{size[0]}x{size[1]} px is narrower than {MIN_IMAGE_WIDTH}",
            "link previews want about 1200x630; see social_card() in website/build.py",
        )
    return Result(PASS, name, f"{ctype} {size[0]}x{size[1]} px")


def check_freshness(site: str = SITE, today: dt.date | None = None) -> list[Result]:
    name = "freshness"
    url = f"{site}badges/as_of.json"
    try:
        status, _, body = fetch(url)
    except FetchError as exc:
        return [_net_fail(name, url, exc, WARN)]
    if status != 200:
        return [
            Result(
                WARN, name, f"{url} returns HTTP {status}", "rerun the 'Live dashboard' workflow"
            )
        ]
    try:
        day, label = shields_date(body.decode("utf-8", errors="replace"))
    except ValueError as exc:
        return [Result(WARN, name, f"badges/as_of.json: {exc}", "see badges() in pipeline.py")]
    age = age_days(day, today or dt.date.today())
    if age > MAX_AGE_DAYS:
        return [
            Result(
                WARN,
                name,
                f"{label} {day} is {age} days old (limit {MAX_AGE_DAYS})",
                "check the latest 'Live dashboard' run in the Actions tab",
            )
        ]
    return [Result(PASS, name, f"{label} {day} ({age} days old)")]


def check_pages(site: str = SITE) -> list[Result]:
    results = []
    for page in PAGES:
        url = site + page
        try:
            status, _, _ = fetch(url, max_bytes=1)
        except FetchError as exc:
            results.append(_net_fail(f"page {page}", url, exc))
            continue
        if status == 200:
            results.append(Result(PASS, f"page {page}", "returns 200"))
        else:
            results.append(
                Result(
                    FAIL,
                    f"page {page}",
                    f"returns HTTP {status}",
                    "rebuild with website/build.py and redeploy",
                )
            )
    return results


def read_readme(root: Path = ROOT) -> str | None:
    try:
        return (root / "README.md").read_text(encoding="utf-8")
    except OSError:
        return None


def check_readme_offline(root: Path = ROOT) -> list[Result]:
    name = "readme paths"
    text = read_readme(root)
    if text is None:
        return [Result(FAIL, name, "README.md not found", "run from the repository checkout")]
    relative, _ = split_links(readme_links(text))
    missing = missing_relative(relative, root)
    if missing:
        return [
            Result(
                FAIL,
                name,
                f"{len(missing)} missing: " + ", ".join(missing),
                "fix the path in README.md or add the file",
            )
        ]
    return [Result(PASS, name, f"{len(relative)} relative links and images exist")]


def check_readme_online(root: Path = ROOT) -> list[Result]:
    name = "readme site links"
    text = read_readme(root)
    if text is None:
        return []
    _, urls = split_links(readme_links(text))
    bad: list[str] = []
    for url in urls:
        try:
            status, _, _ = fetch(url, max_bytes=1)
        except FetchError as exc:
            return [_net_fail(name, url, exc)]
        if status != 200:
            bad.append(f"{url} ({status})")
    if bad:
        return [Result(FAIL, name, "not resolving: " + ", ".join(bad), "fix the link or redeploy")]
    return [Result(PASS, name, f"{len(urls)} links to the site return 200")]


def check_authors(root: Path = ROOT, pyproject: dict[str, Any] | None = None) -> list[Result]:
    name = "author"
    places: list[str] = []
    try:
        cff = (root / "CITATION.cff").read_text(encoding="utf-8")
        if cff_authors_are_handle(cff):
            places.append("CITATION.cff (authors:)")
    except OSError:
        pass
    try:
        build = (root / "website" / "build.py").read_text(encoding="utf-8")
        if build_author_is_handle(build):
            places.append("website/build.py (AUTHOR)")
    except OSError:
        pass
    if pyproject is not None and pyproject_authors_are_handle(pyproject):
        places.append("pyproject.toml (project.authors)")
    if places:
        return [
            Result(
                WARN,
                name,
                f"still only the handle {HANDLE} in " + ", ".join(places),
                "add your real name in CITATION.cff (family-names / given-names), "
                "website/build.py (AUTHOR) and pyproject.toml ([project] authors)",
            )
        ]
    return [Result(PASS, name, "a real name is set (or no author fields found)")]


def check_github_repo() -> list[Result]:
    name = "github repo"
    try:
        status, _, body = fetch(GITHUB_API)
    except FetchError as exc:
        return [_net_fail(name, GITHUB_API, exc, WARN)]
    if status in (403, 429):
        return [
            Result(
                WARN,
                name,
                "GitHub API rate limit hit (unauthenticated)",
                "wait for the limit to reset, or set the About fields by hand: homepage "
                f"{SITE} and description: {SUGGESTED_DESCRIPTION}",
            )
        ]
    if status != 200:
        return [Result(WARN, name, f"GitHub API returned HTTP {status}", "retry in a minute")]
    try:
        repo = json.loads(body)
    except ValueError as exc:
        return [Result(WARN, name, f"unreadable GitHub response ({exc})", "retry in a minute")]
    return repo_metadata_results(repo)


def repo_metadata_results(repo: dict[str, Any]) -> list[Result]:
    name = "github repo"
    desc = repo.get("description") or ""
    hint = f'Edit "About" on the repo page. Suggested description: {SUGGESTED_DESCRIPTION}'
    results: list[Result] = []
    if not (repo.get("homepage") or "").strip():
        results.append(Result(WARN, name, "homepage is empty", f"set it to {SITE}"))
    if "hidden heartbeat" in desc.lower():
        results.append(Result(WARN, name, f"description is stale: {desc!r}", hint))
    elif not desc.strip():
        results.append(Result(WARN, name, "description is empty", hint))
    if not results:
        results.append(Result(PASS, name, "homepage and description are set"))
    return results


# --------------------------------------------------------------------------- #


def run(offline: bool, root: Path = ROOT) -> list[Result]:
    pyproject, why = read_pyproject(root)
    results: list[Result] = []
    if not offline:
        results += check_pypi(pyproject, why)
        results += check_website()
        results += check_freshness()
        results += check_pages()
    results += check_readme_offline(root)
    if not offline:
        results += check_readme_online(root)
    results += check_authors(root, pyproject)
    if not offline:
        results += check_github_repo()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the project is ready to be posted.")
    parser.add_argument(
        "--offline", action="store_true", help="run only the checks that need no network"
    )
    args = parser.parse_args(argv)
    results = run(args.offline)
    for r in results:
        print(r.line())
    counts = {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL)}
    print(f"\n{counts[PASS]} passed, {counts[WARN]} warnings, {counts[FAIL]} failed")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
