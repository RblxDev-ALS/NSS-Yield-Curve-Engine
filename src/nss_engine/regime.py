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
  the specification used by the Federal Reserve Bank of New York, optionally
  with several predictors;
* the **near-term forward spread** of Engstrom & Sharpe (2019), read straight
  off the fitted forward curve;
* a **pseudo-real-time evaluation** that re-estimates each probit every month
  using only outcomes known at the time, and scores the forecasts it would
  actually have produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm

from .models import FloatArray, nss_loadings

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


def fit_probit(
    x: ArrayLike, y: ArrayLike, max_iter: int = 100, tol: float = 1e-10, l2: float = 0.0
) -> ProbitModel:
    """Maximum-likelihood probit via Newton-Raphson with analytic derivatives.

    The probit log-likelihood is globally concave, so Newton's method with step
    halving converges from zero. ``l2 > 0`` adds a Gaussian prior
    ``−½·l2·Σ b_j²`` on the slope coefficients (not the intercept): when the
    classes are perfectly separable - common with only one or two recessions
    in a short sample - the unpenalised MLE diverges and forecasts become
    exactly 0 or 1. Standard errors come from the inverse observed
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

    prior = np.full(X.shape[1], l2)
    prior[0] = 0.0

    def loglik(b: FloatArray) -> float:
        return float(norm.logcdf(q * (X @ b)).sum() - 0.5 * np.sum(prior * b**2))

    beta = np.zeros(X.shape[1])
    ll = loglik(beta)
    for _ in range(max_iter):
        xb = X @ beta
        # λ_i = q φ(q xb) / Φ(q xb), computed in log space for stability
        lam = q * np.exp(norm.logpdf(q * xb) - norm.logcdf(q * xb))
        grad = X.T @ lam - prior * beta
        hess = -(X * (lam * (lam + xb))[:, None]).T @ X - np.diag(prior)
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
    info = (X * (lam * (lam + xb))[:, None]).T @ X + np.diag(prior)
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
    """Probit of 'recession ``horizon`` months ahead' on monthly predictors."""

    model: ProbitModel
    horizon: int
    fitted: pd.Series  #: in-sample probability, indexed by the *forecast origin* month
    latest_probability: float  #: probability implied by the latest predictors
    latest_date: pd.Timestamp
    auc: float

    @property
    def target_date(self) -> pd.Timestamp:
        return self.latest_date + pd.offsets.MonthEnd(self.horizon)


def _monthly_predictors(predictors: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = predictors.to_frame("spread") if isinstance(predictors, pd.Series) else predictors
    return frame.resample("ME").mean().dropna()


def _monthly_target(recession: pd.Series, horizon: int) -> pd.Series:
    rec = recession.copy()
    rec.index = rec.index.to_period("M").to_timestamp("M")
    return rec.shift(-horizon).rename("target")


def recession_probability_model(
    predictors: pd.Series | pd.DataFrame, recession: pd.Series, horizon: int = 12
) -> RecessionModel:
    """Estimate ``P(recession in month t+h) = Φ(a + b·x_t)`` (NY Fed specification).

    ``predictors`` is one series (e.g. the 10y−3m spread) or a frame of several;
    weekly or daily values are averaged to months. ``recession`` is a monthly
    0/1 NBER indicator.
    """
    X = _monthly_predictors(predictors)
    df = X.join(_monthly_target(recession, horizon), how="inner").dropna()
    if df["target"].nunique() < 2:
        raise ValueError("need both recession and non-recession months to fit the model")
    cols = list(X.columns)
    model = fit_probit(df[cols].to_numpy(), df["target"].to_numpy())
    model = ProbitModel(
        model.coef, model.stderr, model.loglik, model.loglik_null, model.n_obs, ("const", *cols)
    )
    fitted_all = pd.Series(model.predict(X.to_numpy()), index=X.index, name="recession_probability")
    auc = roc_auc(model.predict(df[cols].to_numpy()), df["target"].to_numpy())
    return RecessionModel(
        model=model,
        horizon=horizon,
        fitted=fitted_all,
        latest_probability=float(fitted_all.iloc[-1]),
        latest_date=pd.Timestamp(X.index[-1]),
        auc=auc,
    )


# =============================================================================
# Near-term forward spread and real-time evaluation
# =============================================================================


def near_term_forward_spread(
    params: pd.DataFrame, ahead: float = 1.5, tenor: float = 0.25
) -> pd.Series:
    """Engstrom & Sharpe (2019) near-term forward spread, in percentage points.

    The forward rate on a ``tenor``-year bill starting ``ahead`` years from now
    (6 quarters by default) minus today's ``tenor``-year rate, both from the
    fitted zero curve. It isolates what the market expects monetary policy to
    do over the next year and a half: a negative value means rate *cuts* are
    priced in, which historically precedes recessions. Engstrom & Sharpe show
    it dominates long-term spreads such as 10y−3m as a recession predictor.
    """
    tau = np.array([tenor, ahead, ahead + tenor])
    z = np.array(
        [
            nss_loadings(tau, r.lambda1, r.lambda2) @ np.array([r.beta0, r.beta1, r.beta2, r.beta3])
            for r in params.itertuples()
        ]
    )
    fwd = (z[:, 2] * tau[2] - z[:, 1] * tau[1]) / tenor
    return pd.Series(fwd - z[:, 0], index=params.index, name="near_term_forward_spread")


@dataclass(frozen=True)
class RealTimeEvaluation:
    """Pseudo-out-of-sample recession forecasts from an expanding-window probit."""

    probabilities: pd.Series  #: forecast made at each origin month
    outcomes: pd.Series  #: realised 0/1 recession ``horizon`` months later
    horizon: int

    @property
    def auc(self) -> float:
        return roc_auc(self.probabilities.to_numpy(), self.outcomes.to_numpy())

    @property
    def brier(self) -> float:
        """Mean squared error of the probabilities (lower is better)."""
        return float(np.mean((self.probabilities - self.outcomes) ** 2))

    @property
    def log_score(self) -> float:
        """Average predictive log-likelihood (higher is better)."""
        p = np.clip(self.probabilities.to_numpy(), 1e-6, 1 - 1e-6)
        y = self.outcomes.to_numpy()
        return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def real_time_evaluation(
    predictors: pd.Series | pd.DataFrame,
    recession: pd.Series,
    horizon: int = 12,
    min_train_months: int = 120,
    publication_lag: int = 0,
    l2: float = 1.0,
) -> RealTimeEvaluation:
    """Re-estimate the probit every month on information available at the time.

    At origin month ``t`` the model is fitted on origins ``s ≤ t − horizon −
    publication_lag``, whose outcomes (recession or not at ``s + horizon``)
    were known by ``t``; its forecast for ``t + horizon`` is recorded. NBER
    announces turning points with a lag of months to a year, so setting
    ``publication_lag`` (e.g. 12) gives a stricter, more realistic test. No
    information from after ``t`` enters any forecast. Early windows often
    contain a single recession, so the probit is fitted with a weak ridge
    prior ``l2`` (see :func:`fit_probit`).
    """
    X = _monthly_predictors(predictors)
    target = _monthly_target(recession, horizon)
    df = X.join(target, how="inner")
    cols = list(X.columns)
    known = df.dropna()
    probs, outs = {}, {}
    gap = horizon + publication_lag
    for i in range(len(df)):
        origin = df.index[i]
        if not np.isfinite(df["target"].iloc[i]):
            continue
        cutoff = origin - pd.offsets.MonthEnd(gap)
        train = known.loc[:cutoff]
        if len(train) < min_train_months or train["target"].nunique() < 2:
            continue
        m = fit_probit(train[cols].to_numpy(), train["target"].to_numpy(), l2=l2)
        probs[origin] = float(m.predict(df[cols].iloc[[i]].to_numpy())[0])
        outs[origin] = float(df["target"].iloc[i])
    if not probs:
        raise ValueError("not enough history for a real-time evaluation")
    return RealTimeEvaluation(
        pd.Series(probs, name="probability"), pd.Series(outs, name="outcome"), horizon
    )


def block_bootstrap_auc_difference(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    labels: ArrayLike,
    block: int = 24,
    n_boot: int = 2000,
    level: float = 0.9,
    seed: int = 0,
) -> tuple[float, float, float]:
    """AUC(a) − AUC(b) with a moving-block bootstrap confidence interval.

    Recession months come in episodes and 12-month-ahead targets overlap, so
    months are far from independent; resampling whole blocks of ``block``
    consecutive months (circularly) keeps that dependence. Both signals are
    scored on the same resampled months. Returns ``(difference, lower, upper)``.
    With only a few recessions in a sample the interval is wide - which is the
    point of reporting it.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = y.size
    diff = roc_auc(a, y) - roc_auc(b, y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        draws[k] = roc_auc(a[idx], y[idx]) - roc_auc(b[idx], y[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size < 0.5 * n_boot:
        return diff, float("nan"), float("nan")
    tail = (1 - level) / 2
    lo, hi = np.quantile(draws, [tail, 1 - tail])
    return float(diff), float(lo), float(hi)


def compare_recession_predictors(
    candidates: dict[str, pd.Series | pd.DataFrame],
    recession: pd.Series,
    horizon: int = 12,
    min_train_months: int = 120,
    publication_lag: int = 0,
) -> pd.DataFrame:
    """In-sample fit and pseudo-real-time accuracy of several probit specifications.

    All models are scored on the same forecast origins, so their out-of-sample
    numbers are comparable. Each later candidate's out-of-sample AUC is also
    compared with the first one's, with a 90% block-bootstrap interval
    (:func:`block_bootstrap_auc_difference`).
    """
    rows, evals = {}, {}
    for name, x in candidates.items():
        m = recession_probability_model(x, recession, horizon)
        rows[name] = {
            "pseudo_r2": m.model.pseudo_r2,
            "auc_in_sample": m.auc,
            "latest_probability": m.latest_probability,
        }
        evals[name] = real_time_evaluation(x, recession, horizon, min_train_months, publication_lag)
    common = None
    for ev in evals.values():
        idx = ev.probabilities.index
        common = idx if common is None else common.intersection(idx)
    base = None
    for name, ev in evals.items():
        sub = RealTimeEvaluation(ev.probabilities.loc[common], ev.outcomes.loc[common], ev.horizon)
        rows[name].update(
            auc_out_of_sample=sub.auc,
            brier_out_of_sample=sub.brier,
            log_score_out_of_sample=sub.log_score,
            n_forecasts=float(len(common)) if common is not None else 0.0,
        )
        if base is None:
            base = sub
            continue
        d, lo, hi = block_bootstrap_auc_difference(
            sub.probabilities.to_numpy(), base.probabilities.to_numpy(), sub.outcomes.to_numpy()
        )
        rows[name].update(auc_gain_vs_first=d, auc_gain_lo90=lo, auc_gain_hi90=hi)
    out = pd.DataFrame(rows).T
    out.index.name = "predictors"
    return out
