"""Macro regime classification.

Three complementary views:

* :func:`classify_slope_regime` - Steep / Flat / Inverted from the model
  slope, with hysteresis bands and a persistence filter so the label does
  not flicker when the slope hovers around a threshold (the original
  implementation re-labelled on every single observation).
* :func:`classify_curve_moves` - the trader's taxonomy of curve moves
  (bull/bear x steepener/flattener) over a look-back window.
* :func:`recession_probability` - the New York Fed yield-curve probit model
  (Estrella & Mishkin), mapping the 10Y-3M spread to a 12-month-ahead
  recession probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

STEEP, FLAT, INVERTED = "Expansionary (Steep)", "Transition (Flat)", "Recession Warning (Inverted)"
REGIME_COLORS = {STEEP: "#2ECC40", FLAT: "#AAAAAA", INVERTED: "#FF4136"}

# NY Fed model: P(recession in 12m) = Phi(alpha + beta * spread_10y3m)
NYFED_ALPHA, NYFED_BETA = -0.5333, -0.6330


def _raw_state(x: float, prev: str | None, steep: float, inverted: float, band: float) -> str:
    """Threshold classification with hysteresis around each boundary."""
    if prev is None:
        if x < inverted:
            return INVERTED
        return STEEP if x > steep else FLAT
    if prev == INVERTED:
        if x > steep + band:
            return STEEP
        return FLAT if x > inverted + band else INVERTED
    if prev == STEEP:
        if x < inverted - band:
            return INVERTED
        return FLAT if x < steep - band else STEEP
    # prev == FLAT
    if x < inverted - band:
        return INVERTED
    if x > steep + band:
        return STEEP
    return FLAT


def classify_slope_regime(slope: pd.Series, steep: float = 0.5, inverted: float = -0.1,
                          band: float = 0.05, min_persistence: int = 2) -> pd.Series:
    """Label each date Steep / Flat / Inverted from a slope series (long - short, %).

    ``band`` adds hysteresis: leaving a state requires crossing its
    threshold by ``band``. A new state is only adopted after it has been
    indicated for ``min_persistence`` consecutive observations.
    """
    if steep <= inverted:
        raise ValueError("steep threshold must exceed inverted threshold")
    labels = []
    current, candidate, count = None, None, 0
    for x in slope.to_numpy(dtype=float):
        if not np.isfinite(x):
            labels.append(current)
            continue
        raw = _raw_state(x, current, steep, inverted, band)
        if current is None:
            current = raw
        elif raw != current:
            count = count + 1 if raw == candidate else 1
            candidate = raw
            if count >= min_persistence:
                current, candidate, count = raw, None, 0
        else:
            candidate, count = None, 0
        labels.append(current)
    return pd.Series(labels, index=slope.index, name="Regime", dtype=object)


def classify_curve_moves(level: pd.Series, slope: pd.Series, window: int = 4,
                         threshold_bp: float = 5.0) -> pd.DataFrame:
    """Bull/bear steepener/flattener label over a ``window``-period look-back.

    *Bear* = yields up, *bull* = yields down; *steepener/flattener* = slope
    up/down. Moves smaller than ``threshold_bp`` on both axes are
    ``Range-bound``; if only one axis moves, the dominant move is reported
    (e.g. ``Parallel Sell-off`` / ``Pure Steepener``).
    """
    dl = level.diff(window) * 100.0
    ds = slope.diff(window) * 100.0
    out = []
    for a, b in zip(dl.to_numpy(), ds.to_numpy()):
        if not (np.isfinite(a) and np.isfinite(b)):
            out.append(None)
            continue
        big_l, big_s = abs(a) >= threshold_bp, abs(b) >= threshold_bp
        if not big_l and not big_s:
            out.append("Range-bound")
        elif big_l and not big_s:
            out.append("Parallel Sell-off" if a > 0 else "Parallel Rally")
        elif big_s and not big_l:
            out.append("Pure Steepener" if b > 0 else "Pure Flattener")
        else:
            out.append(f"{'Bear' if a > 0 else 'Bull'} {'Steepener' if b > 0 else 'Flattener'}")
    return pd.DataFrame({"LevelChange_bp": dl, "SlopeChange_bp": ds, "Move": out},
                        index=level.index)


def recession_probability(spread_10y3m: pd.Series, monthly: bool = True) -> pd.Series:
    """NY Fed probit: probability of a US recession 12 months ahead.

    ``spread_10y3m`` is the 10Y minus 3M Treasury spread in percentage
    points. The published model uses monthly averages, so by default the
    input is averaged by calendar month first. (The NY Fed converts the 3M
    bill to a bond-equivalent basis; FRED ``DGS3MO`` is already quoted that
    way.)
    """
    s = spread_10y3m.dropna()
    if monthly:
        s = s.resample("ME").mean()
    return pd.Series(norm.cdf(NYFED_ALPHA + NYFED_BETA * s.to_numpy()),
                     index=s.index, name="RecessionProb12m")


def regime_segments(regimes: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Collapse a label series into contiguous ``(start, end, label)`` spans.

    Used for chart shading: one rectangle per span instead of one per
    observation keeps the dashboard light.
    """
    segs = []
    r = regimes.dropna()
    if r.empty:
        return segs
    start, cur = r.index[0], r.iloc[0]
    prev = start
    for ts, lab in r.iloc[1:].items():
        if lab != cur:
            segs.append((start, ts, cur))
            start, cur = ts, lab
        prev = ts
    segs.append((start, prev, cur))
    return segs
