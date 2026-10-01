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
from numpy.typing import ArrayLike, NDArray

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

#: FRED daily TIPS constant-maturity (real) yields -> maturity in years. The
#: 5-, 7-, 10- and 20-year series start in January 2003; the 30-year in
#: February 2010.
TIPS_SERIES: dict[str, float] = {
    "DFII5": 5.0,
    "DFII7": 7.0,
    "DFII10": 10.0,
    "DFII20": 20.0,
    "DFII30": 30.0,
}

#: FRED's own breakeven inflation rates (percent, daily): the 5- and 10-year
#: nominal minus TIPS constant-maturity yields, and the 5-year, 5-year forward
#: rate computed from them.
BREAKEVEN_SERIES: dict[str, str] = {
    "T5YIE": "5Y breakeven",
    "T10YIE": "10Y breakeven",
    "T5YIFR": "5y5y forward breakeven",
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
    return _http_get_bytes(url, timeout=timeout, retries=retries).decode("utf-8")


def _http_get_bytes(url: str, timeout: float = 30.0, retries: int = 4) -> bytes:
    """Like :func:`_http_get`, returning the raw body (for binary files)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body: bytes = resp.read()
                return body
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
    s = parse_fred_csv(path.read_text(encoding="utf-8"), series_id)
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


def load_tips_yields(
    start: str | pd.Timestamp | None = "2003-01-01",
    end: str | pd.Timestamp | None = None,
    freq: str | None = "W-FRI",
    how: str = "last",
    min_tenors: int = 4,
    **fetch_kwargs: object,
) -> pd.DataFrame:
    """TIPS constant-maturity real yield panel from FRED (``DFII5`` … ``DFII30``).

    Same layout as :func:`load_treasury_yields`: columns are maturities in
    years, values in percent. Like the nominal CMT yields these are
    semi-annual par yields, of inflation-indexed notes and bonds.
    """
    raw = fetch_fred(TIPS_SERIES, start=start, end=end, **fetch_kwargs)
    known = {c: TIPS_SERIES[c] for c in raw.columns if c in TIPS_SERIES}
    panel = raw[list(known)].rename(columns=known).sort_index(axis=1)
    panel = resample_yields(panel, freq, how)
    panel = panel[panel.notna().sum(axis=1) >= min_tenors]
    panel.index.name = "date"
    return panel


def load_breakevens(
    start: str | pd.Timestamp | None = "2003-01-01",
    end: str | pd.Timestamp | None = None,
    **fetch_kwargs: object,
) -> pd.DataFrame:
    """FRED's breakeven inflation rates (``T5YIE``, ``T10YIE``, ``T5YIFR``), daily, percent."""
    raw = fetch_fred(BREAKEVEN_SERIES, start=start, end=end, **fetch_kwargs)
    return raw.dropna(how="all")


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


def load_kim_wright_term_premium(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    maturity: int = 10,
    **fetch_kwargs: object,
) -> pd.Series:
    """Kim & Wright (2005) term premium on a zero-coupon bond, percent (daily).

    The Federal Reserve Board's estimate from a three-factor affine model fitted
    to the Treasury curve and survey forecasts of short rates, published on FRED
    as ``THREEFYTP1`` … ``THREEFYTP10`` (the number is the maturity in years).
    It is an independent benchmark for :mod:`nss_engine.termpremium`.
    """
    if not 1 <= maturity <= 10:
        raise ValueError("Kim-Wright term premia exist for maturities 1 to 10 years")
    sid = f"THREEFYTP{maturity}"
    s = fetch_fred_series(sid, **fetch_kwargs)  # type: ignore[arg-type]
    s = s.loc[slice(start, end)].dropna()
    s.name = f"kim_wright_tp{maturity}"
    return s


# =============================================================================
# Adrian-Crump-Moench term premium (Federal Reserve Bank of New York)
# =============================================================================

#: The New York Fed's published ACM estimates (yields, expected short rates and
#: term premia, 1 to 10 years). The file has been an ``.xls`` workbook for
#: years; the ``.xlsx`` name is tried too in case it is converted.
ACM_URLS = (
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls",
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xlsx",
)


def parse_acm_frame(df: pd.DataFrame, maturity: int = 10) -> pd.DataFrame:
    """The New York Fed's ACM series for one maturity from its spreadsheet.

    Expects a ``DATE`` column and the columns ``ACMY{nn}`` (fitted yield),
    ``ACMRNY{nn}`` (risk-neutral yield, the average expected short rate) and
    ``ACMTP{nn}`` (term premium), percent, with ``nn`` the maturity in years
    (``01`` … ``10``). Returns them as ``yield``, ``expected_short_rate`` and
    ``term_premium`` on a sorted ``DatetimeIndex``.
    """
    cols = {str(c).strip().upper(): c for c in df.columns}
    if "DATE" not in cols:
        raise DataError("ACM file lacks a DATE column")
    nn = f"{maturity:02d}"
    names = {
        f"ACMY{nn}": "yield",
        f"ACMRNY{nn}": "expected_short_rate",
        f"ACMTP{nn}": "term_premium",
    }
    if f"ACMTP{nn}" not in cols:
        raise DataError(f"ACM file lacks ACMTP{nn}")
    dates = df[cols["DATE"]]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates.astype(str), format="mixed", errors="coerce")
    out = pd.DataFrame(
        {
            new: pd.to_numeric(df[cols[old]], errors="coerce")
            for old, new in names.items()
            if old in cols
        }
    )
    out.index = pd.DatetimeIndex(dates, name="date")
    out = out[out.index.notna()].sort_index()
    return out.dropna(how="all")


def load_acm_term_premium(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    maturity: int = 10,
    *,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 24.0 * 7,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adrian, Crump & Moench (2013) term premium as published by the New York Fed.

    The original model, estimated by its authors on the Fed's
    Gürkaynak-Sack-Wright zero curve since 1961; :mod:`nss_engine.termpremium`
    reimplements it, so this is a check of that implementation as well as a
    second benchmark next to Kim-Wright. Returns ``yield``,
    ``expected_short_rate`` and ``term_premium`` (percent) for ``maturity``
    years (1-10). Reading the ``.xls`` workbook needs ``xlrd``
    (``pip install "nss-engine[surveys]"``).
    """
    if not 1 <= maturity <= 10:
        raise ValueError("ACM term premia exist for maturities 1 to 10 years")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / "acm_term_premium.csv"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 <= max_age_hours
    if fresh and not refresh:
        raw = pd.read_csv(path)
    else:
        try:
            raw = _download_acm()
        except DataError:
            if not path.exists():
                raise
            import warnings

            warnings.warn(
                f"ACM download failed; using stale cache {path}", RuntimeWarning, stacklevel=2
            )
            raw = pd.read_csv(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(path, index=False)
    return parse_acm_frame(raw, maturity).loc[slice(start, end)]


def _download_acm() -> pd.DataFrame:
    errors = []
    for url in ACM_URLS:
        try:
            content = _http_get_bytes(url, timeout=120.0)
        except DataError as exc:
            errors.append(str(exc))
            continue
        try:
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
        except ImportError as exc:
            raise DataError("reading the ACM file needs xlrd: pip install xlrd") from exc
        except Exception as exc:  # pragma: no cover - many possible parser errors
            errors.append(f"could not parse {url}: {exc}")
            continue
        # prefer the daily sheet (the longest); any sheet with a DATE column works
        usable = [
            df for df in sheets.values() if "DATE" in {str(c).strip().upper() for c in df.columns}
        ]
        if usable:
            df = max(usable, key=len)
            date_col = next(c for c in df.columns if str(c).strip().upper() == "DATE")
            df[date_col] = pd.to_datetime(
                df[date_col].astype(str), format="mixed", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            return df
        errors.append(f"no sheet with a DATE column in {url}")
    raise DataError("; ".join(errors))


# =============================================================================
# Survey of Professional Forecasters (Federal Reserve Bank of Philadelphia)
# =============================================================================

SPF_URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/{statistic}_{variable}_level.xlsx"
)

#: SPF forecasts of the 3-month T-bill rate used as survey anchors, with the
#: months they average over, counted from the end of the survey's middle month
#: (``start``, ``end``; see :func:`spf_bill_windows`). ``TBILL3``-``TBILL6``
#: are quarterly averages 1-4 quarters ahead, ``TBILLB``-``TBILLD`` calendar-year
#: averages 1-3 years ahead, ``BILL10`` the average over the next ten years.
SPF_BILL_SERIES = ("TBILL3", "TBILL4", "TBILL5", "TBILL6", "TBILLB", "TBILLC", "TBILLD", "BILL10")


def spf_bill_windows(series: str, quarter: int) -> tuple[int, int]:
    """Months ``(start, end)`` after the survey date that an SPF bill forecast averages.

    The survey date is the end of the middle month of the survey quarter (the
    SPF is released in the middle of that month), so for a first-quarter survey
    month 1 is March. Quarterly forecasts ``TBILLk`` cover quarter ``q + k − 2``;
    annual ones (``TBILLB`` = next calendar year, ``C``, ``D``) the twelve
    months of that year; ``BILL10`` the next 120 months.
    """
    if series == "BILL10":
        return 1, 120
    if series.startswith("TBILL") and series[5:].isdigit():
        k = int(series[5:])
        if k < 3:
            raise ValueError(f"{series} is not a forecast of a future quarter")
        start = 3 * (k - 3) + 2
        return start, start + 2
    years_ahead = {"TBILLB": 1, "TBILLC": 2, "TBILLD": 3}.get(series)
    if years_ahead is None:
        raise ValueError(f"unknown SPF bill series {series!r}")
    start = 12 * years_ahead - 3 * quarter + 2
    return start, start + 11


def discount_to_continuous(rate: ArrayLike, days: int = 91) -> NDArray[np.float64]:
    """T-bill discount rate (percent) to a continuously compounded yield (percent).

    A bill quoted at discount ``d`` costs ``1 − d·days/360``; its continuously
    compounded yield is ``−ln(price)·365/days``. At 5% the two differ by 10 bp.
    """
    d = np.asarray(rate, dtype=float) / 100.0
    return -np.log1p(-d * days / 360.0) * 365.0 / days * 100.0


def parse_spf_bill_forecasts(
    wide: pd.DataFrame, series: Iterable[str] = SPF_BILL_SERIES
) -> pd.DataFrame:
    """Turn SPF level files (``YEAR``, ``QUARTER``, ``TBILL3`` …) into survey anchors.

    Returns one row per forecast with columns ``date`` (end of the survey's
    middle month), ``series``, ``start`` and ``end`` (the months it averages
    over, see :func:`spf_bill_windows`), ``value`` (continuously compounded
    percent) and ``quoted`` (the discount rate as published) - the format
    :func:`nss_engine.termpremium.fit_acm` takes as ``surveys``.
    """
    df = wide.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    if not {"YEAR", "QUARTER"} <= set(df.columns):
        raise DataError("SPF file lacks YEAR/QUARTER columns")
    rows = []
    for _, r in df.iterrows():
        year, quarter = r["YEAR"], r["QUARTER"]
        if pd.isna(year) or pd.isna(quarter):
            continue
        year, quarter = int(year), int(quarter)
        date = pd.Timestamp(year=year, month=3 * quarter - 1, day=1) + pd.offsets.MonthEnd(0)
        for name in series:
            value = pd.to_numeric(r.get(name, np.nan), errors="coerce")
            if pd.isna(value):
                continue
            start, end = spf_bill_windows(name, quarter)
            rows.append((date, name, start, end, float(value)))
    out = pd.DataFrame(rows, columns=["date", "series", "start", "end", "quoted"])
    out["value"] = discount_to_continuous(out["quoted"].to_numpy())
    out = out[["date", "series", "start", "end", "value", "quoted"]]
    return out.sort_values(["date", "start"]).reset_index(drop=True)


def load_spf_bill_forecasts(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    statistic: str = "median",
    series: Iterable[str] = SPF_BILL_SERIES,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 24.0 * 7,
    refresh: bool = False,
) -> pd.DataFrame:
    """Survey of Professional Forecasters' 3-month T-bill rate forecasts.

    The Philadelphia Fed publishes the median (or ``statistic='mean'``)
    forecast each quarter since 1981 for the next four quarters and the next
    few calendar years (``TBILL``), and since 1992, in first-quarter surveys, the
    average over the next ten years (``BILL10``). These are the survey anchors
    for :func:`nss_engine.termpremium.fit_acm`; see
    :func:`parse_spf_bill_forecasts` for the format. Reading the Excel files
    needs ``openpyxl`` (``pip install nss-engine[surveys]``).
    """
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / f"spf_{statistic}_bills.csv"
    fresh = path.exists() and (time.time() - path.stat().st_mtime) / 3600.0 <= max_age_hours
    if fresh and not refresh:
        wide = pd.read_csv(path)
    else:
        try:
            frames = [
                _read_spf_excel(_http_get_bytes(SPF_URL.format(statistic=statistic, variable=v)))
                for v in ("tbill", "bill10")
            ]
            wide = frames[0].merge(frames[1], on=["YEAR", "QUARTER"], how="outer")
        except DataError:
            if not path.exists():
                raise
            import warnings

            warnings.warn(
                f"SPF download failed; using stale cache {path}", RuntimeWarning, stacklevel=2
            )
            wide = pd.read_csv(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            wide.to_csv(path, index=False)
    out = parse_spf_bill_forecasts(wide, series)
    lo = pd.Timestamp(start) if start is not None else out["date"].min()
    hi = pd.Timestamp(end) if end is not None else out["date"].max()
    return out[(out["date"] >= lo) & (out["date"] <= hi)].reset_index(drop=True)


def _read_spf_excel(content: bytes) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(content))
    except ImportError as exc:
        raise DataError("reading the SPF files needs openpyxl: pip install openpyxl") from exc
    except Exception as exc:  # pragma: no cover - many possible parser errors
        raise DataError(f"could not parse SPF file: {exc}") from exc
    df.columns = [str(c).strip().upper() for c in df.columns]
    if not {"YEAR", "QUARTER"} <= set(df.columns):
        raise DataError("SPF file lacks YEAR/QUARTER columns")
    return df


# =============================================================================
# Federal Reserve (Gürkaynak-Sack-Wright) Svensson curve
# =============================================================================

#: The Fed's daily Svensson zero-curve parameters (Gürkaynak, Sack & Wright, 2007).
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

#: The Fed's daily Svensson curve for TIPS real yields (Gürkaynak, Sack & Wright, 2010).
GSW_TIPS_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"


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


def load_gsw_tips_parameters(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: Path | str | None = None,
    max_age_hours: float = 24.0,
    refresh: bool = False,
) -> pd.DataFrame:
    """The Fed's Svensson curve for TIPS real zero yields (Gürkaynak, Sack & Wright, 2010).

    Same file layout as :func:`load_gsw_parameters`. Subtracting it from the
    nominal GSW curve gives the Fed's own zero-coupon breakeven inflation, an
    independent benchmark for :mod:`nss_engine.inflation`. GSW consider it
    reliable from about 2 (early years: 5) to 20 years.
    """
    text = _cached_text("feds200805", GSW_TIPS_URL, cache_dir, max_age_hours, refresh)
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
        return path.read_text(encoding="utf-8")
    try:
        text = _http_get(url, timeout=120.0)
    except DataError:
        if path.exists():
            import warnings

            warnings.warn(
                f"download of {url} failed; using stale cache {path}", RuntimeWarning, stacklevel=3
            )
            return path.read_text(encoding="utf-8")
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
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
