"""Macro regime detection from the yield curve.

The slope of the Treasury curve is one of the most reliable recession
predictors in macroeconomics (Estrella & Hardouvelis, 1991; Estrella &
Mishkin, 1998): every U.S. recession since the late 1960s was preceded by an
inversion of the 10-year / 3-month spread. This module turns a slope series
(model-implied or observed) into:

* a **level regime** - Inverted / Flat / Normal / Steep - with hysteresis so
  that noise around a threshold does not make the label flicker;
* a **dynamics regime** - bull/bear steepening/flattening - describing *how*
  the curve is moving;
* **inversion episodes** and their lead times to NBER recessions;
* a **probit recession-probability model** ``P(recession in 12m) = Φ(a + b·spread)``,
  the specification used by the Federal Reserve Bank of New York.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm

from .models import FloatArray

REGIMES = ("Inverted", "Flat", "Normal", "Steep")
#: Default regime boundaries on the spread, in percentage points.
DEFAULT_THRESHOLDS = (0.0, 0.5, 1.5)


# =============================================================================
# Level regimes
# =============================================================================


def classify_slope(
    spread: pd.Series,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    hysteresis: float = 0.10,
    labels: Sequence[str] = REGIMES,
) -> pd.Series:
    """Classify each date's spread into a regime, with hysteresis.

    A regime boundary ``b`` is only crossed upward once the spread exceeds
    ``b + hysteresis`` and downward once it falls below ``b - hysteresis``.
    With ``hysteresis=0`` this is plain bucketing.
    """
    bounds = np.asarray(thresholds, dtype=float)
    if len(labels) != bounds.size + 1:
        raise ValueError("need exactly one more label than thresholds")
    if np.any(np.diff(bounds) <= 0):
        raise ValueError("thresholds must be strictly increasing")
    values = spread.to_numpy(dtype=float)
    states = np.full(values.size, -1, dtype=int)
    state = -1
    for i, v in enumerate(values):
        if not np.isfinite(v):
            states[i] = state
            continue
        if state < 0:
            state = int(np.searchsorted(bounds, v, side="right"))
        else:
            while state < bounds.size and v > bounds[state] + hysteresis:
                state += 1
            while state > 0 and v < bounds[state - 1] - hysteresis:
                state -= 1
        states[i] = state
    cat = pd.Categorical.from_codes(states, categories=list(labels), ordered=True)
    return pd.Series(cat, index=spread.index, name="regime")


def regime_spells(regimes: pd.Series) -> pd.DataFrame:
    """Contiguous spells of each regime: start, end, length (periods)."""
    r = regimes.dropna()
    if r.empty:
        return pd.DataFrame(columns=["regime", "start", "end", "periods"])
    block = (r != r.shift()).cumsum()
    rows = [
        {"regime": grp.iloc[0], "start": grp.index[0], "end": grp.index[-1], "periods": len(grp)}
        for _, grp in r.groupby(block)
    ]
    return pd.DataFrame(rows)


def curve_dynamics(
    level: pd.Series, slope: pd.Series, window: int = 4, min_move: float = 0.05
) -> pd.Series:
    """Label curve moves over ``window`` periods.

    ============  ====================  =============================
    Level move    Slope move            Label
    ============  ====================  =============================
    down (bull)   up                    Bull steepener (e.g. Fed cuts)
    up (bear)     up                    Bear steepener (term premium)
    down (bull)   down                  Bull flattener (flight to duration)
    up (bear)     down                  Bear flattener (Fed hikes)
    ============  ====================  =============================

    Moves smaller than ``min_move`` (percentage points) in both level and slope
    are labelled ``Quiet``.
    """
    dl = level.diff(window)
    ds = slope.diff(window)
    out = pd.Series("Quiet", index=level.index, name="dynamics", dtype=object)
    active = (dl.abs() >= min_move) | (ds.abs() >= min_move)
    bull, steep = dl < 0, ds > 0
    out[active & bull & steep] = "Bull steepener"
    out[active & ~bull & steep] = "Bear steepener"
    out[active & bull & ~steep] = "Bull flattener"
    out[active & ~bull & ~steep] = "Bear flattener"
    out[dl.isna() | ds.isna()] = np.nan
    return out


# =============================================================================
# Inversions and recessions
# =============================================================================


def to_monthly(series: pd.Series, how: str = "mean") -> pd.Series:
    """Month-end-stamped monthly aggregate (``mean`` or ``last``)."""
    r = series.resample("ME")
    return (r.mean() if how == "mean" else r.last()).dropna()


def inversion_episodes(spread: pd.Series, min_periods: int = 1) -> pd.DataFrame:
    """Maximal runs of a negative spread lasting at least ``min_periods``."""
    s = spread.dropna()
    neg = s < 0
    rows = []
    for _, grp in s[neg].groupby((neg != neg.shift()).cumsum()[neg]):
        if len(grp) >= min_periods:
            rows.append(
                {
                    "start": grp.index[0],
                    "end": grp.index[-1],
                    "periods": len(grp),
                    "min_spread": float(grp.min()),
                    "date_of_min": grp.idxmin(),
                }
            )
    return pd.DataFrame(rows, columns=["start", "end", "periods", "min_spread", "date_of_min"])


def recession_starts(recession: pd.Series) -> pd.DatetimeIndex:
    """First month of each recession in a 0/1 indicator."""
    r = recession.fillna(0).astype(int)
    return pd.DatetimeIndex(r.index[(r == 1) & (r.shift(fill_value=0) == 0)])


def inversion_lead_times(
    spread_monthly: pd.Series,
    recession: pd.Series,
    min_months: int = 3,
    max_lead_months: int = 36,
) -> pd.DataFrame:
    """Match sustained inversions to the next recession.

    Returns one row per inversion episode of at least ``min_months`` months with
    the lead time (months) from inversion start to the next recession start
    within ``max_lead_months`` (``NaN`` = no recession followed: a false alarm).
    """
    episodes = inversion_episodes(spread_monthly, min_periods=min_months)
    starts = recession_starts(recession)
    leads = []
    for start in episodes["start"]:
        nxt = starts[starts > start]
        lead = np.nan
        if len(nxt):
            months = (nxt[0].year - start.year) * 12 + (nxt[0].month - start.month)
            if months <= max_lead_months:
                lead = months
        leads.append(lead)
    episodes["recession_start"] = [
        starts[starts > s][0] if not np.isnan(ld) else pd.NaT
        for s, ld in zip(episodes["start"], leads, strict=True)
    ]
    episodes["lead_months"] = leads
    return episodes


# =============================================================================
# Probit recession model
# =============================================================================


@dataclass(frozen=True)
class ProbitModel:
    """Fitted probit ``P(y=1 | x) = Φ(x·β)`` (first coefficient = intercept)."""

    coef: FloatArray
    stderr: FloatArray
    loglik: float
    loglik_null: float
    n_obs: int
    names: tuple[str, ...] = ("const", "spread")

    def predict(self, x: ArrayLike) -> FloatArray:
        X = _design(x)
        return np.asarray(norm.cdf(X @ self.coef), dtype=float)

    @property
    def pseudo_r2(self) -> float:
        """McFadden's pseudo-R²: 1 − LL/LL₀."""
        return 1.0 - self.loglik / self.loglik_null

    @property
    def zstats(self) -> FloatArray:
        return np.asarray(self.coef / self.stderr, dtype=float)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"coef": self.coef, "stderr": self.stderr, "z": self.zstats},
            index=list(self.names),
        )


def _design(x: ArrayLike) -> FloatArray:
    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_probit(x: ArrayLike, y: ArrayLike, max_iter: int = 100, tol: float = 1e-10) -> ProbitModel:
    """Maximum-likelihood probit via Newton-Raphson with analytic derivatives.

    The probit log-likelihood is globally concave, so Newton's method with step
    halving converges from zero. Standard errors come from the inverse observed
    information. (With overlapping monthly forecast horizons the errors are
    serially correlated, so these standard errors are optimistic; they are
    reported for reference, not formal inference.)
    """
    X = _design(x)
    yv = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != yv.size:
        raise ValueError("x and y must have the same length")
    if not set(np.unique(yv)).issubset({0.0, 1.0}):
        raise ValueError("y must be binary 0/1")
    q = 2.0 * yv - 1.0

    def loglik(b: FloatArray) -> float:
        return float(norm.logcdf(q * (X @ b)).sum())

    beta = np.zeros(X.shape[1])
    ll = loglik(beta)
    for _ in range(max_iter):
        xb = X @ beta
        # λ_i = q φ(q xb) / Φ(q xb), computed in log space for stability
        lam = q * np.exp(norm.logpdf(q * xb) - norm.logcdf(q * xb))
        grad = X.T @ lam
        hess = -(X * (lam * (lam + xb))[:, None]).T @ X
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-8:
            cand = beta - t * step
            ll_c = loglik(cand)
            if ll_c >= ll - 1e-12:
                break
            t /= 2.0
        converged = abs(ll_c - ll) < tol
        beta, ll = cand, ll_c
        if converged:
            break
    xb = X @ beta
    lam = q * np.exp(norm.logpdf(q * xb) - norm.logcdf(q * xb))
    info = (X * (lam * (lam + xb))[:, None]).T @ X
    stderr = np.sqrt(np.diag(np.linalg.pinv(info)))
    p_bar = float(np.clip(yv.mean(), 1e-12, 1 - 1e-12))
    ll_null = float(yv.size * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar)))
    return ProbitModel(beta, stderr, ll, ll_null, int(yv.size))


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve (probability a random positive outranks a random negative)."""
    s = np.asarray(scores, dtype=float)
    lab = np.asarray(labels, dtype=int)
    pos, neg = s[lab == 1], s[lab == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


@dataclass(frozen=True)
class RecessionModel:
    """Probit of 'recession ``horizon`` months ahead' on the monthly spread."""

    model: ProbitModel
    horizon: int
    fitted: pd.Series  #: in-sample probability, indexed by the *forecast origin* month
    latest_probability: float  #: probability implied by the latest spread
    latest_date: pd.Timestamp
    auc: float

    @property
    def target_date(self) -> pd.Timestamp:
        return self.latest_date + pd.offsets.MonthEnd(self.horizon)


def recession_probability_model(
    spread: pd.Series, recession: pd.Series, horizon: int = 12
) -> RecessionModel:
    """Estimate ``P(recession in month t+h) = Φ(a + b·spread_t)`` (NY Fed specification).

    ``spread`` may be weekly or daily; it is averaged to months. ``recession``
    is a monthly 0/1 NBER indicator.
    """
    s = to_monthly(spread, "mean")
    rec = recession.copy()
    rec.index = rec.index.to_period("M").to_timestamp("M")
    target = rec.shift(-horizon)
    df = pd.concat({"spread": s, "target": target}, axis=1).dropna()
    if df["target"].nunique() < 2:
        raise ValueError("need both recession and non-recession months to fit the model")
    model = fit_probit(df["spread"].to_numpy(), df["target"].to_numpy())
    fitted_all = pd.Series(model.predict(s.to_numpy()), index=s.index, name="recession_probability")
    auc = roc_auc(model.predict(df["spread"].to_numpy()), df["target"].to_numpy())
    return RecessionModel(
        model=model,
        horizon=horizon,
        fitted=fitted_all,
        latest_probability=float(fitted_all.iloc[-1]),
        latest_date=pd.Timestamp(s.index[-1]),
        auc=auc,
    )
