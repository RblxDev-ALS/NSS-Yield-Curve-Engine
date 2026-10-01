"""Bond excess returns: does an estimated term premium predict them?

A term premium is, by definition, the extra return investors *expect* for
holding a long bond instead of rolling over short ones. An estimate of it can
therefore be checked against something that is observed: the excess returns
that bonds actually earned afterwards. If a model says that the expected
one-year excess return on a 10-year bond is 2%, then, averaged over many
dates, the 10-year bond should have beaten one-year bills by about that much,
and more in years when the model said more. This module computes:

* realized h-month log excess returns on n-month zero-coupon bonds,
  ``rx_{t+h}(n) = p_{t+h}(n−h) − p_t(n) − h·y_t(h)/12``
  (:func:`excess_returns`);
* the excess return an affine model *expects* at each date
  (:func:`model_expected_excess_returns`,
  :meth:`~nss_engine.termpremium.ACMResult.expected_excess_returns`), and the
  same estimated in pseudo-real time, re-estimating the model each month on
  past data only (:func:`real_time_expected_returns`);
* the classic forecasting regressions, also in real time: Fama & Bliss's (1987)
  forward–spot spread and Cochrane & Piazzesi's (2005) single factor built
  from one-year forward rates (:func:`real_time_regression_forecasts`,
  :func:`forward_rates`, :func:`cochrane_piazzesi_forecasts`);
* out-of-sample scores against the expanding historical mean, the benchmark
  that return forecasts famously struggle to beat: Campbell & Thompson's
  (2008) out-of-sample R², Clark & West's (2007) test for nested models with a
  Newey–West variance (the h-month returns overlap), and a Mincer–Zarnowitz
  regression of realized on expected returns (:func:`evaluate_return_forecasts`).

A forecast made at month ``t`` may use only returns that were complete by
``t``: those formed at ``t − h`` or earlier. Every real-time function here
keeps to that rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from .models import FloatArray

if TYPE_CHECKING:  # pragma: no cover
    from .termpremium import ACMResult

#: Months per year; zero yields are percent a year, continuously compounded.
_MONTHLY = 1200.0

#: Bonds whose one-year excess returns are scored by default (months).
DEFAULT_MATURITIES = (24, 60, 120)


# =============================================================================
# Realized returns and forward rates
# =============================================================================


def _check_monthly(index: pd.Index) -> None:
    """Rows must be consecutive months: returns are found by counting rows."""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        months = index.to_period("M")
        steps = np.diff(months.year * 12 + months.month)
        if not (steps == 1).all():
            raise ValueError("the panel needs one row per month with no gaps")


def _months(zero_yields: pd.DataFrame) -> dict[int, str | float]:
    """Map whole months to the panel's column labels (maturities in years)."""
    out: dict[int, str | float] = {}
    for c in zero_yields.columns:
        m = float(c) * 12.0
        if abs(m - round(m)) < 1e-6:
            out[round(m)] = c
    return out


def excess_returns(
    zero_yields: pd.DataFrame,
    horizon: int = 12,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
) -> pd.DataFrame:
    """Realized ``horizon``-month log excess returns on zero-coupon bonds, percent.

    ``zero_yields`` is a monthly panel of zero yields (percent, continuously
    compounded) with columns in years on the monthly grid, as from
    :func:`~nss_engine.termpremium.zero_panel`. The return on the ``n``-month
    bond bought at ``t`` and sold at ``t + horizon`` as an ``(n − horizon)``-month
    bond, minus the ``horizon``-month yield at ``t``, is stored at the
    *formation* date ``t``; the last ``horizon`` rows are missing. Columns are
    the maturities in years.
    """
    _check_monthly(zero_yields.index)
    cols = _months(zero_yields)
    out = {}
    for n in maturities:
        if n <= horizon:
            raise ValueError("maturities must be longer than the holding period")
        needed = [n, horizon] + ([n - horizon] if n - horizon > 0 else [])
        missing = [m for m in needed if m not in cols]
        if missing:
            raise ValueError(f"zero panel lacks the {missing}-month yields")
        buy = zero_yields[cols[n]] * n
        sell = zero_yields[cols[n - horizon]].shift(-horizon) * (n - horizon)
        bill = zero_yields[cols[horizon]] * horizon
        out[n / 12.0] = (buy - sell - bill) / 12.0
    return pd.DataFrame(out, index=zero_yields.index)


def forward_rates(
    zero_yields: pd.DataFrame, years: Sequence[int] = (1, 2, 3, 4, 5)
) -> pd.DataFrame:
    """One-year forward rates ``f(n) = n·y(n) − (n−1)·y(n−1)`` (percent), ``f(1) = y(1)``.

    The rate for borrowing from year ``n − 1`` to year ``n``, locked in today;
    Cochrane & Piazzesi's predictors. Columns are ``"f1" … "f5"``.
    """
    cols = _months(zero_yields)
    out = {}
    for n in years:
        if 12 * n not in cols or (n > 1 and 12 * (n - 1) not in cols):
            raise ValueError(f"zero panel lacks the yields for the {n}-year forward")
        y_n = zero_yields[cols[12 * n]] * n
        y_prev = zero_yields[cols[12 * (n - 1)]] * (n - 1) if n > 1 else 0.0
        out[f"f{n}"] = y_n - y_prev
    return pd.DataFrame(out, index=zero_yields.index)


def forward_spot_spread(zero_yields: pd.DataFrame, maturity: int, horizon: int = 12) -> pd.Series:
    """Fama–Bliss predictor: the forward rate from ``n − h`` to ``n`` months minus the ``h``-month yield."""
    cols = _months(zero_yields)
    n, h = maturity, horizon
    missing = [m for m in (n, n - h, h) if m not in cols]
    if missing:
        raise ValueError(f"zero panel lacks the {missing}-month yields")
    fwd = (zero_yields[cols[n]] * n - zero_yields[cols[n - h]] * (n - h)) / h
    return (fwd - zero_yields[cols[h]]).rename(f"fb{n}")


# =============================================================================
# Model-implied expected returns
# =============================================================================


def model_expected_excess_returns(
    X: FloatArray,
    mu: FloatArray,
    phi: FloatArray,
    A: FloatArray,
    B: FloatArray,
    horizon: int,
    maturities: Sequence[int],
) -> FloatArray:
    """Expected ``horizon``-month log excess returns (percent) in an affine model.

    Log bond prices are ``p_t(n) = A_n + B_n' X_t`` (``A``, ``B`` indexed from
    ``n = 1``; risk-neutral loadings, as fitted to yields) and the state follows
    ``X_{t+1} = μ + Φ X_t + v`` under the real-world measure, so::

        E_t[rx_{t+h}(n)] = A_{n−h} + B_{n−h}' E_t[X_{t+h}] − p_t(n) + p_t(h)

    with ``E_t[X_{t+h}] = Σ_{j<h} Φ^j μ + Φ^h X_t``. Returns a ``T × len(maturities)``
    array.
    """
    k = phi.shape[0]
    drift = np.zeros(k)
    power = np.eye(k)
    for _ in range(horizon):
        drift = drift + power @ mu
        power = power @ phi
    EX = drift[None, :] + X @ power.T

    def logp(n: int, state: FloatArray) -> FloatArray:
        if n == 0:
            return np.zeros(state.shape[0])
        return np.asarray(A[n - 1] + state @ B[n - 1], dtype=float)

    out = np.empty((X.shape[0], len(maturities)))
    for j, n in enumerate(maturities):
        if not horizon < n <= A.size:
            raise ValueError(f"maturity {n} must exceed the horizon and be at most {A.size}")
        out[:, j] = logp(n - horizon, EX) - logp(n, X) + logp(horizon, X)
    return out * 100.0


def acm_expected_excess_returns(
    res: ACMResult, horizon: int = 12, maturities: Sequence[int] = DEFAULT_MATURITIES
) -> pd.DataFrame:
    """Expected excess returns implied by an estimated ACM model (percent).

    The prices of risk turn the real-world dynamics into the risk-neutral ones
    that price bonds; the expected excess return is what a bond earns, on
    average under the real-world dynamics, over the short rate. This is the
    quantity the term premium averages over a bond's life.
    """
    from .termpremium import affine_loadings

    n_max = round(float(res.fitted.columns[-1]) * 12)
    A, B = affine_loadings(
        n_max,
        res.mu - res.lambda0,
        res.phi - res.lambda1,
        res.sigma,
        res.sigma_e**2,
        res.delta0,
        res.delta1,
    )
    X = res.factors.to_numpy()
    values = model_expected_excess_returns(X, res.mu, res.phi, A, B, horizon, maturities)
    return pd.DataFrame(values, index=res.factors.index, columns=[n / 12.0 for n in maturities])


def real_time_expected_returns(
    zero_yields: pd.DataFrame,
    horizon: int = 12,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    min_train: int = 120,
    max_fit_error_bp: float = 10.0,
    **acm_kwargs: object,
) -> pd.DataFrame:
    """Expected excess returns estimated each month from data up to that month.

    Re-estimates :func:`~nss_engine.termpremium.fit_acm` every month (with
    ``acm_kwargs``, e.g. ``surveys=``; surveys published after a month are not
    used for it) and keeps the last row of :func:`acm_expected_excess_returns`.
    An estimate whose average yield fitting error exceeds ``max_fit_error_bp``
    is recorded as missing, as in
    :func:`~nss_engine.termpremium.real_time_decomposition`. Also returns the
    10-year term premium estimated at the same time (``term_premium``), when
    the panel reaches 10 years.
    """
    from .termpremium import fit_acm

    _check_monthly(zero_yields.index)
    cols = [n / 12.0 for n in maturities]
    rows = {}
    for t in range(min_train - 1, len(zero_yields)):
        res = fit_acm(zero_yields.iloc[: t + 1], **acm_kwargs)  # type: ignore[arg-type]
        ok = bool(res.fit_rmse_bp.mean() <= max_fit_error_bp)
        er = acm_expected_excess_returns(res, horizon, maturities).iloc[-1].to_numpy()
        tp = res.term_premium.iloc[-1]
        ten = tp.iloc[int(np.argmin(np.abs(tp.index.to_numpy(dtype=float) - 10.0)))]
        rows[zero_yields.index[t]] = [
            *(er if ok else np.full(len(cols), np.nan)),
            ten if ok else np.nan,
        ]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=[*cols, "term_premium"])
    out.index.name = zero_yields.index.name
    return out


# =============================================================================
# Forecasting regressions in real time
# =============================================================================


def real_time_regression_forecasts(
    target: pd.Series,
    predictors: pd.DataFrame | None,
    horizon: int = 12,
    min_train: int = 60,
) -> pd.Series:
    """Expanding-window OLS forecasts of ``target`` using only completed returns.

    ``target`` holds returns at their formation date (as from
    :func:`excess_returns`), ``predictors`` the signals known at each date. At
    origin ``t`` the regression uses formation dates up to ``t − horizon``
    (the returns realized by ``t``) and forecasts with the predictors at ``t``.
    ``predictors=None`` gives the expanding historical mean, the standard
    benchmark. At least ``min_train`` completed returns are required.
    """
    idx = target.index
    y = target.to_numpy(dtype=float)
    if predictors is None:
        Z = np.ones((len(idx), 1))
    else:
        P = predictors.reindex(idx).to_numpy(dtype=float)
        Z = np.column_stack([np.ones(len(idx)), P])
    out = np.full(len(idx), np.nan)
    for t in range(len(idx)):
        last = t - horizon  # last formation date whose return is known at t
        if last + 1 < min_train or not np.isfinite(Z[t]).all():
            continue
        keep = np.isfinite(y[: last + 1]) & np.isfinite(Z[: last + 1]).all(axis=1)
        if keep.sum() < max(min_train, Z.shape[1] + 2):
            continue
        coef, *_ = np.linalg.lstsq(Z[: last + 1][keep], y[: last + 1][keep], rcond=None)
        out[t] = Z[t] @ coef
    return pd.Series(out, index=idx, name=target.name)


def cochrane_piazzesi_forecasts(
    returns: pd.DataFrame,
    forwards: pd.DataFrame,
    horizon: int = 12,
    min_train: int = 60,
    factor_maturities: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
) -> pd.DataFrame:
    """Cochrane & Piazzesi's single-factor forecasts, estimated in real time.

    At each origin, using only completed returns: (1) regress the average
    excess return on the 2–5 year bonds on the forward rates, giving the
    "tent-shaped" factor ``γ'f``; (2) for each bond in ``returns``, regress its
    excess return on that factor; forecast with today's forward rates.
    ``returns`` must contain ``factor_maturities`` (years) as columns; the
    forecasts are returned for every column of ``returns``.
    """
    missing = [m for m in factor_maturities if m not in returns.columns]
    if missing:
        raise ValueError(f"returns need the columns {missing} for the factor")
    idx = returns.index
    avg = returns[list(factor_maturities)].mean(axis=1).to_numpy()
    F = np.column_stack([np.ones(len(idx)), forwards.reindex(idx).to_numpy(dtype=float)])
    R = returns.to_numpy(dtype=float)
    out = np.full(R.shape, np.nan)
    for t in range(len(idx)):
        last = t - horizon
        if last + 1 < min_train or not np.isfinite(F[t]).all():
            continue
        keep = np.isfinite(avg[: last + 1]) & np.isfinite(F[: last + 1]).all(axis=1)
        if keep.sum() < max(min_train, F.shape[1] + 2):
            continue
        gamma, *_ = np.linalg.lstsq(F[: last + 1][keep], avg[: last + 1][keep], rcond=None)
        cp = F[: last + 1] @ gamma
        Zc = np.column_stack([np.ones(keep.sum()), cp[keep]])
        for j in range(R.shape[1]):
            ok = np.isfinite(R[: last + 1, j])[keep]
            b, *_ = np.linalg.lstsq(Zc[ok], R[: last + 1, j][keep][ok], rcond=None)
            out[t, j] = b[0] + b[1] * (F[t] @ gamma)
    return pd.DataFrame(out, index=idx, columns=returns.columns)


# =============================================================================
# Scoring
# =============================================================================


def _newey_west_ols(y: FloatArray, x: FloatArray, lags: int) -> tuple[FloatArray, FloatArray]:
    """OLS of ``y`` on ``[1, x]`` with Newey–West (Bartlett) standard errors."""
    Z = np.column_stack([np.ones(y.size), x])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    u = y - Z @ coef
    g = Z * u[:, None]
    S = g.T @ g
    for lag in range(1, min(lags, y.size - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = g[lag:].T @ g[:-lag]
        S += w * (G + G.T)
    ZZi = np.linalg.inv(Z.T @ Z)
    cov = ZZi @ S @ ZZi
    return coef, np.sqrt(np.maximum(np.diag(cov), 0.0))


def _newey_west_mean(x: FloatArray, lags: int) -> tuple[float, float]:
    n = x.size
    xc = x - x.mean()
    lrv = float(xc @ xc) / n
    for lag in range(1, min(lags, n - 1) + 1):
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * float(xc[lag:] @ xc[:-lag]) / n
    return float(x.mean()), float(np.sqrt(max(lrv, 0.0) / n))


@dataclass(frozen=True)
class ReturnForecastScore:
    """How well one return forecast did out of sample."""

    n: int  #: forecast origins scored
    r2_oos: float  #: 1 − SSE(forecast)/SSE(benchmark); > 0 beats the benchmark
    clark_west: float  #: Clark–West statistic (positive favours the forecast)
    p_value: float  #: one-sided p-value of the Clark–West test
    mz_slope: float  #: slope of realized on forecast (1 = well calibrated)
    mz_slope_se: float  #: its Newey–West standard error
    mean_forecast: float  #: average forecast (percent)
    mean_realized: float  #: average realized return (percent)

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def evaluate_return_forecasts(
    realized: pd.Series,
    forecast: pd.Series,
    benchmark: pd.Series,
    horizon: int = 12,
) -> ReturnForecastScore:
    """Score a return forecast against a benchmark forecast (usually the historical mean).

    Uses the origins where all three are available. ``r2_oos`` is Campbell &
    Thompson's out-of-sample R². The Clark & West (2007) statistic corrects
    the MSE difference for the noise a larger model adds when the benchmark
    is nested in it; its long-run variance and the Mincer–Zarnowitz slope's
    standard error use Newey–West with ``horizon`` lags, since ``horizon``-month
    returns formed in consecutive months overlap.
    """
    df = pd.concat([realized, forecast, benchmark], axis=1, keys=["y", "f", "b"]).dropna()
    n = len(df)
    nan = float("nan")
    if n < 2 * horizon + 10:
        return ReturnForecastScore(n, nan, nan, nan, nan, nan, nan, nan)
    y, f, b = (df[c].to_numpy(dtype=float) for c in ("y", "f", "b"))
    e_f, e_b = y - f, y - b
    sse_b = float(e_b @ e_b)
    r2 = 1.0 - float(e_f @ e_f) / sse_b if sse_b > 0 else nan
    cw = e_b**2 - (e_f**2 - (b - f) ** 2)
    mean, se = _newey_west_mean(cw, horizon)
    stat = mean / se if se > 0 else nan
    p = float(stats.norm.sf(stat)) if np.isfinite(stat) else nan
    if np.std(f) > 0:
        coef, ses = _newey_west_ols(y, f, horizon)
        slope, slope_se = float(coef[1]), float(ses[1])
    else:
        slope, slope_se = nan, nan
    return ReturnForecastScore(
        n=n,
        r2_oos=r2,
        clark_west=float(stat),
        p_value=p,
        mz_slope=slope,
        mz_slope_se=slope_se,
        mean_forecast=float(f.mean()),
        mean_realized=float(y.mean()),
    )


# =============================================================================
# Were the surveys right?
# =============================================================================


def survey_forecast_errors(surveys: pd.DataFrame, bill_rate: pd.Series) -> pd.DataFrame:
    """Survey forecasts of the bill rate against the rate that followed.

    ``surveys`` is in the format of
    :func:`~nss_engine.data.load_spf_bill_forecasts`: each row forecasts, at
    month ``date``, the average 3-month rate over months ``start … end``
    ahead. ``bill_rate`` is the realized 3-month rate, monthly, in the same
    units (continuously compounded percent, e.g. the 3-month column of a
    zero panel). Returns the surveys whose window has passed, with
    ``realized`` and ``error`` (forecast − realized; positive = the survey
    expected higher rates than came). Survey-anchored expectations inherit
    these errors.
    """
    months = pd.DatetimeIndex(bill_rate.index).to_period("M")
    rate = pd.Series(bill_rate.to_numpy(dtype=float), index=months)
    rate = rate[~rate.index.duplicated(keep="last")]
    rows = []
    for r in surveys.itertuples(index=False):
        origin = pd.Timestamp(r.date).to_period("M")
        window = pd.period_range(origin + int(r.start), origin + int(r.end), freq="M")
        if window[-1] > rate.index[-1] or window[0] < rate.index[0]:
            continue
        realized = rate.reindex(window)
        if realized.isna().any():
            continue
        rows.append((r.date, r.series, r.start, r.end, r.value, float(realized.mean())))
    out = pd.DataFrame(rows, columns=["date", "series", "start", "end", "value", "realized"])
    out["error"] = out["value"] - out["realized"]
    return out
