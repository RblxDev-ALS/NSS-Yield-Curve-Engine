"""Curve analytics built on a calibrated :class:`~nsscurve.calibration.CurveHistory`."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .calibration import CurveHistory
from .model import NSSCurve
from .tenors import format_tenor

KEY_TENORS = (0.25, 2.0, 5.0, 10.0, 30.0)


def _col(df: pd.DataFrame, tenor: str) -> pd.Series | None:
    return df[tenor] if tenor in df.columns else None


def market_spreads(rates: pd.DataFrame) -> pd.DataFrame:
    """Benchmark spreads (percentage points) from raw market quotes, where available."""
    out = {}
    pairs = {"3M10Y": ("3M", "10Y"), "2Y10Y": ("2Y", "10Y"), "5Y30Y": ("5Y", "30Y"),
             "2Y5Y": ("2Y", "5Y")}
    for name, (a, b) in pairs.items():
        sa, sb = _col(rates, a), _col(rates, b)
        if sa is not None and sb is not None:
            out[name] = sb - sa
    if all(t in rates.columns for t in ("2Y", "5Y", "10Y")):
        out["Fly_2s5s10s"] = 2 * rates["5Y"] - rates["2Y"] - rates["10Y"]
    return pd.DataFrame(out, index=rates.index)


def curve_metrics(history: CurveHistory) -> pd.DataFrame:
    """Model-implied levels, spreads and shape descriptors for every date.

    Spreads are computed on the model *par* curve so they are directly
    comparable with the market spreads in :func:`market_spreads`.
    """
    mats = np.array([0.25, 2.0, 5.0, 10.0, 30.0])
    par = history.model_rates(mats, kind="par", labels=[format_tenor(m) for m in mats])
    p = history.params.loc[history.valid]
    out = pd.DataFrame(index=par.index)
    out["Level_Beta0"] = p["Beta0"]
    out["ShortRate"] = p["Beta0"] + p["Beta1"]
    out["Slope_negBeta1"] = -p["Beta1"]
    for c in par.columns:
        out[f"Par_{c}"] = par[c]
    out["Model_3M10Y"] = par["10Y"] - par["3M"]
    out["Model_2Y10Y"] = par["10Y"] - par["2Y"]
    out["Model_5Y30Y"] = par["30Y"] - par["5Y"]
    out["Model_Fly_2s5s10s"] = 2 * par["5Y"] - par["2Y"] - par["10Y"]
    out["Hump1_Years"] = 1.7932821329007607 / p["Lambda1"]
    if "Lambda2" in p:
        out["Hump2_Years"] = 1.7932821329007607 / p["Lambda2"]
    return out


def factor_validation(history: CurveHistory) -> pd.DataFrame:
    """Correlation of NSS factors with their empirical (model-free) proxies.

    A well-behaved calibration should show high correlations here; this is
    the quantitative version of the README claim that ``-beta1`` tracks the
    10Y-2Y spread.
    """
    mkt = market_spreads(history.market)
    p = history.params.loc[history.valid]
    rows = []

    def add(factor, series, proxy_name, proxy):
        if proxy is None:
            return
        both = pd.concat([series, proxy], axis=1).dropna()
        if len(both) > 2:
            lvl = both.corr().iloc[0, 1]
            chg = both.diff().dropna().corr().iloc[0, 1]
            rows.append({"Factor": factor, "Proxy": proxy_name,
                         "CorrLevels": lvl, "CorrChanges": chg})

    add("Beta0 (level)", p["Beta0"], "30Y yield", _col(history.market, "30Y"))
    add("-Beta1 (slope)", -p["Beta1"], "10Y-3M", mkt.get("3M10Y"))
    add("-Beta1 (slope)", -p["Beta1"], "10Y-2Y", mkt.get("2Y10Y"))
    add("Beta2 (curvature)", p["Beta2"], "2*5Y-2Y-10Y", mkt.get("Fly_2s5s10s"))
    return pd.DataFrame(rows)


def pca_decomposition(rates: pd.DataFrame, n_components: int = 3,
                      use_changes: bool = True) -> dict:
    """PCA of yield changes (default) or levels on dates with complete data.

    Returns explained-variance ratios and loadings; the first three
    components are the empirical level/slope/curvature that NSS factors
    parameterise.
    """
    x = rates.dropna()
    if use_changes:
        x = x.diff().dropna()
    if len(x) < n_components + 1:
        raise ValueError("not enough complete observations for PCA")
    x = x - x.mean()
    _, s, vt = np.linalg.svd(x.to_numpy(), full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    loadings = pd.DataFrame(vt[:n_components].T, index=x.columns,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    # Sign convention: level loads positively on average, slope positively on the long end.
    for i, c in enumerate(loadings.columns):
        ref = loadings[c].mean() if i == 0 else loadings[c].iloc[-1] - loadings[c].iloc[0]
        if ref < 0:
            loadings[c] *= -1
    return {"explained_variance": pd.Series(var[:n_components], index=loadings.columns),
            "loadings": loadings}


def carry_rolldown(curve: NSSCurve, tenors: Sequence[float] = (1, 2, 3, 5, 7, 10, 20, 30),
                   horizon: float = 0.25, funding_rate: float | None = None,
                   freq: int = 2) -> pd.DataFrame:
    """Approximate carry and roll-down (bp of return) of par bonds over ``horizon`` years.

    * carry    = (par yield - funding rate) * horizon
    * rolldown = modified duration at the horizon * (y(T) - y(T - horizon))

    Funding defaults to the model zero-coupon yield for the horizon (a
    term-repo proxy). Assumes an unchanged curve ("static curve" scenario).
    """
    tenors = np.asarray(tenors, dtype=float)
    if funding_rate is None:
        funding_rate = float(curve.par_yield(horizon, freq))
    y_t = np.atleast_1d(curve.par_yield(tenors, freq))
    rolled = np.maximum(tenors - horizon, 1e-6)
    y_r = np.atleast_1d(curve.par_yield(rolled, freq))
    yd = y_r / 100.0
    # Modified duration of a par bond with maturity T - h (annuity formula).
    with np.errstate(divide="ignore", invalid="ignore"):
        dur = np.where(yd > 1e-9,
                       (1 - (1 + yd / freq) ** (-freq * rolled)) / yd,
                       rolled)
    carry = (y_t - funding_rate) * horizon * 100.0
    roll = dur * (y_t - y_r) * 100.0
    return pd.DataFrame({"ParYield": y_t, "Carry_bp": carry, "Rolldown_bp": roll,
                         "Total_bp": carry + roll, "ModDuration": dur},
                        index=[format_tenor(t) for t in tenors])


def key_rate_table(curve: NSSCurve, tenors: Sequence[float] = (2, 5, 10, 30),
                   key_tenors: Sequence[float] = (0.25, 2, 5, 10, 20, 30),
                   freq: int = 2) -> pd.DataFrame:
    """Key-rate durations of par bonds priced off the curve (rows = bonds)."""
    rows = {}
    for T in tenors:
        c = float(curve.par_yield(T, freq))
        krd = curve.key_rate_durations(c, T, key_tenors, freq=freq)
        row = {format_tenor(k): v for k, v in krd.items()}
        row["Total"] = sum(krd.values())
        row["Coupon"] = c
        row["Price"] = curve.bond_price(c, T, freq)
        rows[format_tenor(T)] = row
    return pd.DataFrame(rows).T


def forward_curve(curve: NSSCurve, maturities: Sequence[float] | None = None) -> pd.DataFrame:
    """Zero, instantaneous-forward, par and discount curves on a maturity grid."""
    m = np.asarray(maturities if maturities is not None else np.linspace(0.1, 30, 120))
    return pd.DataFrame({"Zero": curve.zero(m), "Forward": curve.forward(m),
                         "Par": curve.par_yield(m), "Discount": curve.discount(m)},
                        index=pd.Index(m, name="Maturity"))


__all__ = ["market_spreads", "curve_metrics", "factor_validation", "pca_decomposition",
           "carry_rolldown", "key_rate_table", "forward_curve"]
