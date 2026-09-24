"""Market data: FRED download, cleaning, resampling, CSV I/O and a synthetic
curve simulator for offline runs and tests.

The original script used ``pandas_datareader``, which is unmaintained and no
longer imports on recent Python/pandas versions. This module talks to FRED
directly with the standard library:

* with ``FRED_API_KEY`` set, the official JSON API is used;
* otherwise the public ``fredgraph.csv`` endpoint (no key required).

Downloads are cached on disk and retried with exponential back-off.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .model import NSSCurve, NSSParams, ParYieldPricer, cont_to_periodic, loading_matrix
from .tenors import FRED_CMT_SERIES, SERIES_TO_TENOR, maturities_of, parse_tenor, format_tenor

FREQ_ALIASES = {"D": None, "B": None, "W": "W-FRI", "M": "ME", "ME": "ME", "W-FRI": "W-FRI"}


class DataError(RuntimeError):
    """Raised when market data cannot be obtained or is unusable."""


class FREDClient:
    CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    API_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str | None = None, cache_dir: str | Path | None = ".fred_cache",
                 max_cache_age_hours: float = 12.0, timeout: float = 30.0, retries: int = 4):
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_age = max_cache_age_hours * 3600.0
        self.timeout = timeout
        self.retries = retries

    # -------------------------------------------------------------- download
    def _get(self, url: str) -> bytes:
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "nsscurve/2.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except Exception as exc:  # network errors vary by platform
                last = exc
                if attempt < self.retries - 1:
                    time.sleep(2 ** (attempt + 1))
        raise DataError(f"FRED request failed after {self.retries} attempts: {last}")

    def _download(self, series_id: str) -> pd.Series:
        if self.api_key:
            q = urllib.parse.urlencode({"series_id": series_id, "api_key": self.api_key,
                                        "file_type": "json"})
            payload = json.loads(self._get(f"{self.API_URL}?{q}"))
            obs = payload.get("observations", [])
            s = pd.Series({o["date"]: o["value"] for o in obs}, dtype=object)
        else:
            raw = self._get(f"{self.CSV_URL}?{urllib.parse.urlencode({'id': series_id})}")
            df = pd.read_csv(io.BytesIO(raw), na_values=["."])
            date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
            s = df.set_index(date_col).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return pd.to_numeric(s.replace(".", np.nan), errors="coerce").rename(series_id)

    def fetch_series(self, series_id: str) -> pd.Series:
        """Full history of one series, served from cache when fresh."""
        path = self.cache_dir / f"{series_id}.csv" if self.cache_dir else None
        if path and path.exists() and time.time() - path.stat().st_mtime < self.max_cache_age:
            return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(series_id)
        s = self._download(series_id)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(path)
        return s

    def fetch_curve(self, start=None, end=None,
                    tenors: Iterable[str] = FRED_CMT_SERIES) -> pd.DataFrame:
        """Daily CMT panel with tenor-label columns (``'1M' ... '30Y'``)."""
        cols = {}
        for tenor in tenors:
            sid = FRED_CMT_SERIES.get(tenor, tenor)
            cols[SERIES_TO_TENOR.get(sid, tenor)] = self.fetch_series(sid)
        df = pd.DataFrame(cols).sort_index()
        return df.loc[start:end] if (start is not None or end is not None) else df


# -----------------------------------------------------------------------------
# Cleaning / resampling / I/O
# -----------------------------------------------------------------------------

def order_tenors(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FRED ids to tenor labels and sort columns by maturity."""
    df = df.rename(columns=lambda c: SERIES_TO_TENOR.get(str(c).upper(), c))
    mats = maturities_of(df.columns)
    return df.iloc[:, np.argsort(mats, kind="stable")]


def clean_curve(df: pd.DataFrame, max_ffill: int = 5) -> pd.DataFrame:
    """Numeric coercion, drop empty rows/columns, bounded forward-fill.

    Unlike a blanket ``ffill().dropna()``, a tenor that is genuinely missing
    (e.g. the 20Y gap in 1987-1993 or the 1M before 2001) stays NaN; the
    calibrator simply ignores it instead of the whole date being dropped.
    """
    df = order_tenors(df.apply(pd.to_numeric, errors="coerce"))
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if max_ffill:
        df = df.ffill(limit=max_ffill)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def resample_curve(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Sample to period-end observations (``'D'``, ``'W'`` (Friday) or ``'M'``)."""
    rule = FREQ_ALIASES.get(freq.upper(), freq)
    if rule is None:
        return df
    return df.resample(rule).last().dropna(how="all")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a user-supplied rates file: first column dates, other columns
    tenor labels (``'2Y'``) or FRED ids (``'DGS2'``), values in percent."""
    df = pd.read_csv(path, index_col=0, parse_dates=True, na_values=[".", ""])
    return clean_curve(df)


# -----------------------------------------------------------------------------
# Synthetic data
# -----------------------------------------------------------------------------

def simulate_curve(start="2019-01-01", periods: int = 1300,
                   tenors: Sequence[str] = tuple(FRED_CMT_SERIES),
                   seed: int = 7, noise_bp: float = 1.0, pricing_error_bp: float = 3.0,
                   lambda1: float = 0.6, lambda2: float = 0.08, quote_type: str = "par",
                   return_factors: bool = False):
    """Simulate a business-daily CMT-style panel from a dynamic NSS model.

    Factors follow mean-reverting AR(1) processes, with a deterministic
    slope cycle that inverts the curve for part of the sample so regime
    logic has something to detect. Quotes are model zero rates converted to
    semi-annual compounding plus i.i.d. noise and a persistent, mean-
    reverting tenor-specific pricing error (the thing RV signals trade).

    ``quote_type='par'`` (default) produces CMT-style quotes: bills as
    zero-coupon bond-equivalent yields, coupon tenors as par yields.
    ``quote_type='zero'`` quotes semi-annually compounded zero rates.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    t = np.arange(periods) / 252.0
    mats = maturities_of(tenors)

    def ar1(mu, phi, sigma, x0=None):
        x = np.empty(periods)
        x[0] = mu if x0 is None else x0
        eps = rng.normal(0.0, sigma, periods)
        for i in range(1, periods):
            x[i] = mu + phi * (x[i - 1] - mu) + eps[i]
        return x

    level = ar1(4.0, 0.999, 0.03)
    slope = 1.0 + 1.6 * np.sin(2 * np.pi * t / 4.0) + ar1(0.0, 0.995, 0.03)  # long - short
    b2 = ar1(-1.0, 0.995, 0.08)
    b3 = ar1(0.8, 0.995, 0.06)
    b1 = -slope
    b1 = np.maximum(b1, 0.10 - level)            # keep the short rate positive

    factors = pd.DataFrame({"Beta0": level, "Beta1": b1, "Beta2": b2, "Beta3": b3,
                            "Lambda1": lambda1, "Lambda2": lambda2}, index=dates)
    betas = factors[["Beta0", "Beta1", "Beta2", "Beta3"]].to_numpy()
    if quote_type == "par":
        pricer = ParYieldPricer(mats)
        quotes = np.vstack([pricer(NSSCurve(NSSParams(*b, lambda1, lambda2))) for b in betas])
    elif quote_type == "zero":
        quotes = cont_to_periodic(betas @ loading_matrix(mats, lambda1, lambda2).T)
    else:
        raise ValueError("quote_type must be 'par' or 'zero'")

    persistent = np.column_stack([ar1(0.0, 0.97, pricing_error_bp / 100 * np.sqrt(1 - 0.97 ** 2))
                                  for _ in mats])
    quotes = quotes + persistent + rng.normal(0.0, noise_bp / 100.0, quotes.shape)
    df = pd.DataFrame(np.round(quotes, 2), index=dates, columns=list(tenors))
    df.index.name = "Date"
    return (df, factors) if return_factors else df


__all__ = ["FREDClient", "DataError", "clean_curve", "resample_curve", "load_csv",
           "simulate_curve", "order_tenors", "parse_tenor", "format_tenor"]
