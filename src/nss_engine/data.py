"""Treasury yield data: FRED download, caching, parsing and resampling.

Yield panels used throughout the package are ``pandas.DataFrame`` objects with a
``DatetimeIndex`` and **columns equal to maturities in years** (floats), values
in percent. :data:`TREASURY_SERIES` maps the FRED constant-maturity series to
those maturities.

No API key is required: series are downloaded from FRED's public CSV endpoint.
If the environment variable ``FRED_API_KEY`` is set, the official JSON API is
used instead. Downloads are cached on disk (``~/.cache/nss_engine`` by default,
override with ``NSS_ENGINE_CACHE``) so repeated runs work offline.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

#: FRED daily Treasury constant-maturity series -> maturity in years.
TREASURY_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

#: NBER recession indicator (monthly, 1 = recession), published on FRED.
RECESSION_SERIES = "USREC"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
USER_AGENT = "nss-engine/1.0 (+https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine)"


class DataError(RuntimeError):
    """Raised when market data cannot be obtained or parsed."""


# =============================================================================
# Labels
# =============================================================================


def maturity_label(tau: float) -> str:
    """Human-readable tenor label: ``1/12 -> '1M'``, ``0.5 -> '6M'``, ``10 -> '10Y'``."""
    months = tau * 12
    if tau < 1 and abs(months - round(months)) < 1e-6:
        return f"{round(months)}M"
    if abs(tau - round(tau)) < 1e-6:
        return f"{round(tau)}Y"
    return f"{tau:g}Y"


def label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of a yield panel with tenor labels (``'2Y'``) instead of float maturities."""
    return df.rename(columns={c: maturity_label(float(c)) for c in df.columns})


# =============================================================================
# FRED access
# =============================================================================


def default_cache_dir() -> Path:
    env = os.environ.get("NSS_ENGINE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "nss_engine"


def parse_fred_csv(text: str, series_id: str | None = None) -> pd.Series:
    """Parse a FRED ``fredgraph.csv`` download into a float Series.

    Handles both header styles FRED has used (``DATE`` and ``observation_date``)
    and the ``'.'`` placeholder FRED uses for missing observations.
    """
    try:
        df = pd.read_csv(io.StringIO(text), na_values=[".", ""], keep_default_na=True)
    except Exception as exc:  # pragma: no cover - pandas raises many types
        raise DataError(f"could not parse FRED CSV: {exc}") from exc
    if df.shape[1] < 2:
        raise DataError("unexpected FRED CSV layout (need a date and a value column)")
    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[1]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name=str(value_col))
    s = s[s.index.notna()]
    s.index.name = "date"
    return s.sort_index()


def parse_fred_json(payload: str, series_id: str) -> pd.Series:
    """Parse a FRED API ``series/observations`` JSON response."""
    data = json.loads(payload)
    if "observations" not in data:
        raise DataError(f"FRED API error for {series_id}: {data.get('error_message', data)}")
    obs = data["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    values = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce")
    s = pd.Series(values.to_numpy(dtype=float), index=dates, name=series_id)
    s.index.name = "date"
    return s.sort_index()


def _http_get(url: str, timeout: float = 30.0, retries: int = 4) -> str:
    """GET with exponential back-off (1s, 2s, 4s, ...)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body.decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if (
                isinstance(exc, urllib.error.HTTPError)
                and 400 <= exc.code < 500
                and exc.code != 429
            ):
                break  # client errors will not fix themselves
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise DataError(f"download failed for {url}: {last_exc}")


def fetch_fred_series(
    series_id: str,
    *,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 12.0,
    refresh: bool = False,
    api_key: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Download the full history of one FRED series, using an on-disk cache.

    A cached copy younger than ``max_age_hours`` is returned without touching
    the network. If the download fails but any cached copy exists, the stale
    copy is returned (with a warning) rather than failing the whole pipeline.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / f"{series_id}.csv"
    if use_cache and not refresh and path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        if age_h <= max_age_hours:
            return _read_cached(path, series_id)

    api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
    try:
        if api_key:
            query = urllib.parse.urlencode(
                {"series_id": series_id, "api_key": api_key, "file_type": "json"}
            )
            series = parse_fred_json(_http_get(f"{FRED_API_URL}?{query}"), series_id)
        else:
            query = urllib.parse.urlencode({"id": series_id})
            series = parse_fred_csv(_http_get(f"{FRED_CSV_URL}?{query}"), series_id)
    except DataError:
        if use_cache and path.exists():
            import warnings

            warnings.warn(
                f"FRED download for {series_id} failed; using stale cache {path}",
                RuntimeWarning,
                stacklevel=2,
            )
            return _read_cached(path, series_id)
        raise
    series.name = series_id
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename_axis("date").to_csv(path, header=True)
    return series


def _read_cached(path: Path, series_id: str) -> pd.Series:
    s = parse_fred_csv(path.read_text(), series_id)
    s.name = series_id
    return s


def fetch_fred(
    series_ids: Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Download several FRED series and align them on a common date index."""
    frames = {sid: fetch_fred_series(sid, **kwargs) for sid in series_ids}  # type: ignore[arg-type]
    df = pd.DataFrame(frames)
    df.index.name = "date"
    return df.loc[slice(start, end)]


# =============================================================================
# Treasury panels
# =============================================================================


def resample_yields(
    df: pd.DataFrame, freq: str | None = "W-FRI", how: str = "last"
) -> pd.DataFrame:
    """Resample a daily panel. ``how='last'`` keeps each tenor's last quote in the
    period (what a trader would see at the close); ``how='mean'`` averages, which is
    the convention for monthly macro regressions (e.g. the NY Fed recession model).
    """
    if freq is None or freq.upper() in ("D", "B", "DAILY"):
        return df.dropna(how="all")
    if how not in ("last", "mean"):
        raise ValueError("how must be 'last' or 'mean'")
    observed = df.dropna(how="all")
    resampler = observed.resample(freq)
    out = resampler.last() if how == "last" else resampler.mean()
    # Label each period by its last *actual* observation date rather than the
    # period end, so a partial final week is not stamped with a future Friday.
    last_obs = pd.Series(observed.index, index=observed.index).resample(freq).max()
    out.index = pd.DatetimeIndex(last_obs.reindex(out.index).to_numpy(), name=df.index.name)
    return out.dropna(how="all")


def treasury_panel_from_fred_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FRED ``DGS*`` columns to maturities (years), sorted by maturity."""
    known = {c: TREASURY_SERIES[c] for c in df.columns if c in TREASURY_SERIES}
    if not known:
        raise DataError("no Treasury constant-maturity (DGS*) columns found")
    out = df[list(known)].rename(columns=known)
    return out.sort_index(axis=1)


def load_treasury_yields(
    start: str | pd.Timestamp | None = "1990-01-01",
    end: str | pd.Timestamp | None = None,
    freq: str | None = "W-FRI",
    how: str = "last",
    min_tenors: int = 6,
    **fetch_kwargs: object,
) -> pd.DataFrame:
    """U.S. Treasury constant-maturity yield panel from FRED.

    Parameters
    ----------
    start, end:
        Date range (inclusive).
    freq:
        Pandas offset alias for resampling (``'W-FRI'``, ``'ME'``, ``'B'``...),
        or ``None`` for daily data.
    how:
        ``'last'`` or ``'mean'`` aggregation within each period.
    min_tenors:
        Drop dates with fewer observed maturities than this.

    Notes
    -----
    Tenors enter and leave the panel over history (the 1-month bill starts in
    2001, the 20-year was not published 1987-1993 and the 30-year paused
    2002-2006). Missing tenors are kept as ``NaN``; the calibrator skips them.
    """
    raw = fetch_fred(TREASURY_SERIES, start=start, end=end, **fetch_kwargs)
    panel = treasury_panel_from_fred_columns(raw)
    panel = resample_yields(panel, freq, how)
    panel = panel[panel.notna().sum(axis=1) >= min_tenors]
    panel.index.name = "date"
    return panel


def load_recession_indicator(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **fetch_kwargs: object,
) -> pd.Series:
    """Monthly NBER recession indicator (``USREC``), 1 = recession month."""
    s = fetch_fred_series(RECESSION_SERIES, **fetch_kwargs)  # type: ignore[arg-type]
    s = s.loc[slice(start, end)].dropna().astype(int)
    s.index = s.index.to_period("M").to_timestamp("M")
    s.name = "recession"
    return s


# =============================================================================
# Federal Reserve (Gürkaynak-Sack-Wright) Svensson curve
# =============================================================================

#: The Fed's daily Svensson zero-curve parameters (Gürkaynak, Sack & Wright, 2007).
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"


def parse_gsw_csv(text: str) -> pd.DataFrame:
    """Parse ``feds200628.csv`` into NSS parameters in this package's convention.

    The file starts with a few lines of notes, then a header row beginning with
    ``Date`` and columns including ``BETA0..BETA3``, ``TAU1``, ``TAU2`` (and the
    fitted yields ``SVENY01..SVENY30``). GSW use time constants ``τ = 1/λ``;
    before 1980 they fit Nelson-Siegel, leaving ``BETA3``/``TAU2`` empty, which
    maps to ``beta3 = 0`` here. Missing values are ``NA``.

    Returns columns ``beta0 … lambda2`` indexed by date; rows that cannot form a
    valid curve are dropped.
    """
    lines = text.splitlines()
    header = next(
        (i for i, ln in enumerate(lines) if ln.split(",", 1)[0].strip().strip('"') == "Date"),
        None,
    )
    if header is None:
        raise DataError("could not find the 'Date' header row in the GSW file")
    df = pd.read_csv(
        io.StringIO("\n".join(lines[header:])), na_values=["NA", ".", ""], keep_default_na=True
    )
    df.columns = [str(c).strip().strip('"').upper() for c in df.columns]
    needed = {"DATE", "BETA0", "BETA1", "BETA2", "TAU1"}
    if not needed <= set(df.columns):
        raise DataError(f"GSW file lacks columns {sorted(needed - set(df.columns))}")
    num = df.drop(columns="DATE").apply(pd.to_numeric, errors="coerce")
    beta3 = num["BETA3"] if "BETA3" in num else pd.Series(0.0, index=num.index)
    tau2 = num["TAU2"] if "TAU2" in num else pd.Series(np.nan, index=num.index)
    ns = beta3.isna() | tau2.isna() | (tau2 <= 0)
    out = pd.DataFrame(
        {
            "beta0": num["BETA0"],
            "beta1": num["BETA1"],
            "beta2": num["BETA2"],
            "beta3": beta3.where(~ns, 0.0),
            "lambda1": 1.0 / num["TAU1"],
            "lambda2": (1.0 / tau2).where(~ns, 1.0 / num["TAU1"]),
        }
    )
    out.index = pd.DatetimeIndex(pd.to_datetime(df["DATE"], errors="coerce"), name="date")
    valid = out.notna().all(axis=1) & (out["lambda1"] > 0) & out.index.notna()
    return out[valid].sort_index()


def load_gsw_parameters(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 24.0,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (with caching) the Fed's Svensson curve parameters.

    This is an *independent* estimate of the same object this package fits:
    GSW fit off-the-run Treasury notes and bonds (no bills, no on-the-run
    issues, no 20-year bond) by minimising duration-weighted price errors. It
    is the natural benchmark for the zero curves estimated here from CMT par
    yields. Their curve is reliable from about 1 year to 30 years.
    """
    text = _cached_text("feds200628", GSW_URL, cache_dir, max_age_hours, refresh)
    return parse_gsw_csv(text).loc[slice(start, end)]


def _cached_text(
    key: str,
    url: str,
    cache_dir: Path | str | None,
    max_age_hours: float,
    refresh: bool,
) -> str:
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / f"{key}.csv"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 <= max_age_hours
    if fresh and not refresh:
        return path.read_text()
    try:
        text = _http_get(url, timeout=120.0)
    except DataError:
        if path.exists():
            import warnings

            warnings.warn(
                f"download of {url} failed; using stale cache {path}", RuntimeWarning, stacklevel=3
            )
            return path.read_text()
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def load_yields_csv(path: str | Path) -> pd.DataFrame:
    """Load a yield panel from CSV.

    The first column must be dates. Remaining columns may be FRED ids
    (``DGS10``), tenor labels (``10Y``, ``3M``) or numeric maturities in years.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True, na_values=[".", ""])
    rename: dict[str, float] = {}
    for col in df.columns:
        rename[col] = _parse_maturity(str(col))
    df = df.rename(columns=rename).sort_index(axis=1).sort_index()
    df.index.name = "date"
    return df.astype(float)


def _parse_maturity(col: str) -> float:
    c = col.strip().upper()
    if c in TREASURY_SERIES:
        return TREASURY_SERIES[c]
    try:
        if c.endswith("M"):
            return float(c[:-1]) / 12.0
        if c.endswith("Y"):
            return float(c[:-1])
        return float(c)
    except ValueError as exc:
        raise DataError(f"cannot interpret column {col!r} as a maturity") from exc


def describe_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tenor coverage summary: first/last date, observation count, mean, std."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        rows.append(
            {
                "tenor": maturity_label(float(col)),
                "first": s.index.min() if len(s) else pd.NaT,
                "last": s.index.max() if len(s) else pd.NaT,
                "obs": len(s),
                "mean": float(s.mean()) if len(s) else np.nan,
                "std": float(s.std()) if len(s) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("tenor")
