"""Yield curve forecasting with the dynamic Nelson-Siegel model.

Diebold & Li (2006) observed that if ``λ`` is fixed, the three Nelson-Siegel
factors can be extracted by a cross-sectional regression each month, and
forecasting the *curve* reduces to forecasting three well-behaved time series::

    ŷ_{t+h}(τ) = X(τ) · f̂_{t+h|t},     f̂_{t+h|t} from an AR(1) / VAR(1) on f_t

This module implements that model and an honest, rolling out-of-sample
evaluation against the random walk ("no change") - the benchmark that is
notoriously hard to beat in yield forecasting - including Diebold-Mariano
tests of equal predictive accuracy. It also scores the equal-weight average of
the model and the random walk: a combination with no estimated weights, so it
cannot overfit, and it gains whenever the two make partly offsetting errors
(Bates & Granger, 1969; Timmermann, 2006).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from .data import maturity_label
from .models import DIEBOLD_LI_LAMBDA, FloatArray, ns_loadings

FactorModel = Literal["ar1", "var1"]
FACTOR_NAMES = ("level", "slope", "curvature")


def extract_factors(yields: pd.DataFrame, lam: float = DIEBOLD_LI_LAMBDA) -> pd.DataFrame:
    """Nelson-Siegel factors with fixed ``λ``: one OLS regression per date.

    Missing tenors are skipped date by date. Returns columns
    ``level, slope, curvature`` (= β0, β1, β2).
    """
    mats = np.asarray(yields.columns, dtype=float)
    X_full = ns_loadings(mats, lam)
    out = np.full((len(yields), 3), np.nan)
    values = yields.to_numpy(dtype=float)
    for i, row in enumerate(values):
        m = np.isfinite(row)
        if m.sum() >= 3:
            out[i] = np.linalg.lstsq(X_full[m], row[m], rcond=None)[0]
    return pd.DataFrame(out, index=yields.index, columns=list(FACTOR_NAMES)).dropna()


@dataclass(frozen=True)
class VARModel:
    """``f_t = c + A f_{t-1} + e_t``. AR(1) per factor is the diagonal special case."""

    intercept: FloatArray
    coef: FloatArray
    resid_cov: FloatArray

    def forecast(self, last: FloatArray, h: int) -> FloatArray:
        f = np.asarray(last, dtype=float)
        for _ in range(h):
            f = self.intercept + self.coef @ f
        return f

    @property
    def unconditional_mean(self) -> FloatArray:
        k = self.coef.shape[0]
        return np.linalg.solve(np.eye(k) - self.coef, self.intercept)


def fit_factor_model(factors: pd.DataFrame | FloatArray, kind: FactorModel = "ar1") -> VARModel:
    """OLS estimate of an AR(1)-per-factor or a VAR(1) model."""
    F = np.asarray(factors, dtype=float)
    if F.shape[0] < 3:
        raise ValueError("need at least 3 observations")
    Y, Xlag = F[1:], F[:-1]
    k = F.shape[1]
    Z = np.column_stack([np.ones(len(Xlag)), Xlag])
    if kind == "var1":
        B = np.linalg.lstsq(Z, Y, rcond=None)[0]  # (1+k, k)
        c, A = B[0], B[1:].T
    elif kind == "ar1":
        c, A = np.zeros(k), np.zeros((k, k))
        for j in range(k):
            b = np.linalg.lstsq(Z[:, [0, 1 + j]], Y[:, j], rcond=None)[0]
            c[j], A[j, j] = b
    else:
        raise ValueError("kind must be 'ar1' or 'var1'")
    resid = Y - (c + Xlag @ A.T)
    cov = np.atleast_2d(np.cov(resid, rowvar=False))
    return VARModel(c, A, cov)


def forecast_curve(
    yields: pd.DataFrame,
    horizon: int,
    kind: FactorModel = "ar1",
    lam: float = DIEBOLD_LI_LAMBDA,
    maturities: Sequence[float] | None = None,
) -> pd.Series:
    """Forecast the curve ``horizon`` periods after the last date in ``yields``."""
    factors = extract_factors(yields, lam)
    model = fit_factor_model(factors, kind)
    f_h = model.forecast(factors.to_numpy()[-1], horizon)
    mats = np.asarray(maturities if maturities is not None else yields.columns, dtype=float)
    return pd.Series(ns_loadings(mats, lam) @ f_h, index=mats, name=f"forecast_h{horizon}")


@dataclass(frozen=True)
class ForecastEvaluation:
    """Out-of-sample accuracy by horizon (rows) and tenor (columns)."""

    rmse_model: pd.DataFrame  #: RMSE in bp
    rmse_random_walk: pd.DataFrame  #: RMSE in bp
    dm_stat: pd.DataFrame  #: Diebold-Mariano statistic (negative = model more accurate)
    dm_pvalue: pd.DataFrame  #: two-sided p-value (Harvey-Leybourne-Newbold corrected)
    n_forecasts: pd.Series
    kind: str
    rmse_combination: pd.DataFrame | None = None  #: RMSE (bp) of ½ model + ½ random walk

    @property
    def relative_rmse(self) -> pd.DataFrame:
        """Model RMSE / random-walk RMSE (< 1 means the model wins)."""
        return self.rmse_model / self.rmse_random_walk

    @property
    def relative_rmse_combination(self) -> pd.DataFrame:
        """Equal-weight combination RMSE / random-walk RMSE."""
        if self.rmse_combination is None:
            raise ValueError("no combination forecasts were evaluated")
        return self.rmse_combination / self.rmse_random_walk


def diebold_mariano(e_model: FloatArray, e_bench: FloatArray, h: int) -> tuple[float, float]:
    """Diebold-Mariano (1995) test with the Harvey, Leybourne & Newbold (1997) correction.

    Loss differential ``d = e_model² − e_bench²``; its long-run variance uses a
    rectangular window of ``h − 1`` lags (errors of h-step forecasts are MA(h−1)).
    """
    d = np.asarray(e_model, dtype=float) ** 2 - np.asarray(e_bench, dtype=float) ** 2
    d = d[np.isfinite(d)]
    n = d.size
    if n < max(10, 2 * h):
        return float("nan"), float("nan")
    d_bar = d.mean()
    dc = d - d_bar
    lrv = np.dot(dc, dc) / n
    for lag in range(1, h):
        lrv += 2.0 * np.dot(dc[lag:], dc[:-lag]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    dm = d_bar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    pval = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, pval


def evaluate_forecasts(
    yields: pd.DataFrame,
    horizons: Sequence[int] = (1, 6, 12),
    min_train: int = 60,
    kind: FactorModel = "ar1",
    lam: float = DIEBOLD_LI_LAMBDA,
    rolling_window: int | None = None,
) -> ForecastEvaluation:
    """Rolling-origin out-of-sample evaluation vs. the random walk.

    At every origin ``t ≥ min_train`` the factor model is re-estimated using
    data up to ``t`` only (expanding window, or the last ``rolling_window``
    observations), then ``h``-step forecasts are compared with realised yields.
    ``yields`` should be monthly for the classic Diebold-Li setting.
    """
    factors = extract_factors(yields, lam)
    Y = yields.reindex(factors.index)
    mats = np.asarray(Y.columns, dtype=float)
    X = ns_loadings(mats, lam)
    F = factors.to_numpy()
    obs = Y.to_numpy(dtype=float)
    T = len(F)
    labels = [maturity_label(m) for m in mats]

    err_m: dict[int, list[FloatArray]] = {h: [] for h in horizons}
    err_rw: dict[int, list[FloatArray]] = {h: [] for h in horizons}
    for t in range(min_train - 1, T - 1):
        lo = 0 if rolling_window is None else max(0, t + 1 - rolling_window)
        model = fit_factor_model(F[lo : t + 1], kind)
        for h in horizons:
            if t + h >= T:
                continue
            y_hat = X @ model.forecast(F[t], h)
            err_m[h].append((obs[t + h] - y_hat) * 100.0)
            err_rw[h].append((obs[t + h] - obs[t]) * 100.0)

    def rmse(errs: list[FloatArray]) -> FloatArray:
        if not errs:
            return np.full(mats.size, np.nan)
        a = np.array(errs)
        return np.sqrt(np.nanmean(a**2, axis=0))

    rm = pd.DataFrame([rmse(err_m[h]) for h in horizons], index=list(horizons), columns=labels)
    rr = pd.DataFrame([rmse(err_rw[h]) for h in horizons], index=list(horizons), columns=labels)
    rc = pd.DataFrame(
        [rmse(combination_errors(err_m[h], err_rw[h])) for h in horizons],
        index=list(horizons),
        columns=labels,
    )
    dm_s = pd.DataFrame(index=list(horizons), columns=labels, dtype=float)
    dm_p = pd.DataFrame(index=list(horizons), columns=labels, dtype=float)
    for h in horizons:
        if not err_m[h]:
            continue
        em, er = np.array(err_m[h]), np.array(err_rw[h])
        for j, lab in enumerate(labels):
            m = np.isfinite(em[:, j]) & np.isfinite(er[:, j])
            dm_s.loc[h, lab], dm_p.loc[h, lab] = diebold_mariano(em[m, j], er[m, j], h)
    for frame in (rm, rr, rc, dm_s, dm_p):
        frame.index.name = "horizon"
    n = pd.Series({h: len(err_m[h]) for h in horizons}, name="n_forecasts")
    return ForecastEvaluation(rm, rr, dm_s, dm_p, n, kind, rc)


def combination_errors(
    err_model: list[FloatArray], err_random_walk: list[FloatArray]
) -> list[FloatArray]:
    """Errors of the forecast ``½ model + ½ random walk``.

    ``y − (ŷ + y_t)/2 = (e_model + e_rw)/2``, so no forecasts need storing.
    """
    return [(a + b) / 2.0 for a, b in zip(err_model, err_random_walk, strict=True)]
