"""Relative-value signals from curve-fit residuals.

The residual of a tenor (market yield minus smooth model yield) measures
how rich or cheap that point is versus the rest of the curve. A residual is
only tradeable if it is (a) large relative to its own history and (b) mean
reverting, so both are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import market_spreads


def residual_zscores(residuals_bp: pd.DataFrame, window: int = 26,
                     min_periods: int | None = None) -> pd.DataFrame:
    """Rolling z-score of each tenor's residual against its *trailing* history.

    The window statistics exclude the current observation, so the score of
    date *t* uses only information available before *t* (no look-ahead).
    """
    min_periods = min_periods or max(window // 2, 5)
    past = residuals_bp.shift(1).rolling(window, min_periods=min_periods)
    std = past.std().where(lambda s: s > 1e-9)
    return (residuals_bp - past.mean()) / std


def half_life(series: pd.Series) -> float:
    """Mean-reversion half-life (periods) from an AR(1) fit ``x_t = a + phi x_{t-1}``.

    Returns ``inf`` when the series shows no mean reversion (``phi >= 1``)
    and NaN when there is too little data.
    """
    x = series.dropna().to_numpy()
    if len(x) < 10:
        return np.nan
    x0, x1 = x[:-1], x[1:]
    X = np.column_stack([np.ones_like(x0), x0])
    (_, phi), *_ = np.linalg.lstsq(X, x1, rcond=None)
    if phi <= 0:
        return 0.0
    if phi >= 1 - 1e-8:
        return np.inf
    return float(-np.log(2) / np.log(phi))


def rich_cheap_table(residuals_bp: pd.DataFrame, window: int = 26,
                     threshold: float = 2.0, max_half_life: float | None = None) -> pd.DataFrame:
    """Latest rich/cheap read per tenor.

    * ``CHEAP`` (z >= threshold): market yield above the fitted curve -> the
      bond is cheap; RV trade = buy it versus the curve / neighbours.
    * ``RICH`` (z <= -threshold): market yield below the curve -> sell.

    ``max_half_life`` (periods) optionally suppresses signals on tenors whose
    residuals do not mean-revert fast enough to be tradeable.
    """
    z = residual_zscores(residuals_bp, window)
    last = residuals_bp.index[-1]
    rows = []
    for tenor in residuals_bp.columns:
        r, zz = residuals_bp.at[last, tenor], z.at[last, tenor]
        hl = half_life(residuals_bp[tenor])
        signal = "FAIR"
        if np.isfinite(zz):
            if zz >= threshold:
                signal = "CHEAP (buy)"
            elif zz <= -threshold:
                signal = "RICH (sell)"
        if signal != "FAIR" and max_half_life is not None and not hl <= max_half_life:
            signal += " [slow reversion]"
        rows.append({"Tenor": tenor, "Residual_bp": r, "ZScore": zz,
                     "HalfLife_periods": hl, "Signal": signal})
    return pd.DataFrame(rows).set_index("Tenor")


def butterfly_signals(rates: pd.DataFrame, fitted: pd.DataFrame,
                      window: int = 26) -> pd.DataFrame:
    """Market vs model 2s5s10s butterfly and its z-scored deviation.

    The deviation isolates the part of the butterfly the smooth curve does
    not explain, i.e. the richness/cheapness of the 5Y body.
    """
    mkt = market_spreads(rates).get("Fly_2s5s10s")
    mdl = market_spreads(fitted).get("Fly_2s5s10s")
    if mkt is None or mdl is None:
        return pd.DataFrame()
    dev = (mkt - mdl) * 100.0
    z = residual_zscores(dev.to_frame("Fly"), window)["Fly"]
    return pd.DataFrame({"MarketFly_bp": mkt * 100.0, "ModelFly_bp": mdl * 100.0,
                         "Deviation_bp": dev, "ZScore": z})


def signal_backtest(residuals_bp: pd.DataFrame, window: int = 26, threshold: float = 2.0,
                    horizon: int = 4) -> pd.DataFrame:
    """Does the residual actually revert after a signal?

    For every (date, tenor) with ``|z| >= threshold`` measure the residual
    change over the next ``horizon`` periods in the direction of the trade
    (positive = the dislocation closed). Reports hit rate and average
    reversion in bp per tenor. This is a signal-quality check, not a P&L
    backtest (no DV01 weighting or transaction costs).
    """
    z = residual_zscores(residuals_bp, window)
    fwd = residuals_bp.shift(-horizon) - residuals_bp
    rows = []
    for tenor in residuals_bp.columns:
        trig = z[tenor].abs() >= threshold
        gain = (-np.sign(z[tenor]) * fwd[tenor])[trig].dropna()
        rows.append({"Tenor": tenor, "Signals": int(len(gain)),
                     "HitRate": float((gain > 0).mean()) if len(gain) else np.nan,
                     "AvgReversion_bp": float(gain.mean()) if len(gain) else np.nan})
    return pd.DataFrame(rows).set_index("Tenor")
