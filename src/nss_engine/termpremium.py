"""Term premium: splitting yields into expected short rates and compensation for risk.

A 10-year yield is the average short rate investors expect over the next ten
years **plus a term premium**, the extra return they demand for locking money
up at a fixed rate instead of rolling over bills. The two parts mean very
different things: a rise in expected short rates says markets expect the Fed to
tighten; a rise in the term premium says investors want more compensation for
duration risk (inflation uncertainty, supply, volatility).

Neither part is observed. This module estimates them with the regression-based
affine term structure model of **Adrian, Crump & Moench (2013)** ("ACM"), the
model behind the Federal Reserve Bank of New York's published term premium:

1. Pricing factors ``X_t`` are the first ``K`` principal components of the
   zero-coupon yields (maturities 1 to 120 months).
2. A VAR(1) gives their dynamics under the real-world measure,
   ``X_{t+1} = μ + Φ X_t + v_{t+1}``, ``v ~ N(0, Σ)``.
3. One-month excess returns on bonds are regressed on the lagged factors and
   the VAR innovations, ``rx_{t+1} = a + β'v_{t+1} + c X_t + e_{t+1}``.
4. No-arbitrage restrictions turn those coefficients into the market prices of
   risk ``λ_t = λ0 + λ1 X_t``::

       λ1 = (ββ')⁻¹ β c
       λ0 = (ββ')⁻¹ β (a + ½(B* vec Σ + σ² ι))

5. The usual affine recursions price every maturity twice: with the estimated
   prices of risk (fitted yields) and with ``λ0 = λ1 = 0`` (the *risk-neutral*
   yield, i.e. the average expected short rate). The term premium is the
   difference.

Everything is ordinary least squares, so a full estimation takes milliseconds.
That makes it cheap to re-estimate the model every month on past data only, as
:func:`real_time_decomposition` does for the recession study.

The zero curves come from the NSS fits (:func:`zero_panel`); ACM themselves use
the Fed's Gürkaynak-Sack-Wright curve. Running the model on both shows how much
the curve estimate matters. The Kim & Wright (2005) term premium, published by
the Fed and available on FRED (``THREEFYTP10``), is an independent estimate from
a different model with survey data, and serves as the benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import FloatArray, NSSCurve, nss_loadings

#: Months per year; yields enter the model as monthly log yields in decimals.
_MONTHLY = 1200.0

#: Maturities (months) whose excess returns are used in ACM's regression.
DEFAULT_RETURN_MATURITIES = tuple(range(6, 121, 6))


@dataclass(frozen=True)
class ACMResult:
    """An estimated ACM model and its decomposition of the yield curve.

    All yields are in percent a year, continuously compounded; maturities of
    the frames are in years (``1/12, 2/12, …``).
    """

    observed: pd.DataFrame  #: input zero yields
    fitted: pd.DataFrame  #: model-implied yields
    risk_neutral: pd.DataFrame  #: yields with zero prices of risk (expected short rates)
    factors: pd.DataFrame  #: pricing factors (principal components)
    mu: FloatArray  #: VAR intercept (K)
    phi: FloatArray  #: VAR slope (K × K)
    sigma: FloatArray  #: VAR innovation covariance (K × K)
    lambda0: FloatArray  #: price of risk, constant (K)
    lambda1: FloatArray  #: price of risk, slope (K × K)
    delta0: float  #: short rate intercept (monthly, decimal)
    delta1: FloatArray  #: short rate loadings (K)
    sigma_e: float  #: pricing-error standard deviation of excess returns (monthly)
    return_maturities: tuple[int, ...]
    var_capped: bool = False  #: the real-world VAR was scaled back to stationarity

    @property
    def term_premium(self) -> pd.DataFrame:
        """Fitted yield minus risk-neutral yield, percent."""
        return self.fitted - self.risk_neutral

    @property
    def fit_rmse_bp(self) -> pd.Series:
        """Yield fitting error by maturity, basis points."""
        err = (self.fitted - self.observed) * 100.0
        return np.sqrt((err**2).mean())

    def decomposition(self, maturity: float = 10.0) -> pd.DataFrame:
        """``yield = expected short rate + term premium`` for one maturity (years)."""
        col = _nearest_column(self.fitted, maturity)
        return pd.DataFrame(
            {
                "yield": self.observed[col],
                "fitted": self.fitted[col],
                "expected_short_rate": self.risk_neutral[col],
                "term_premium": self.term_premium[col],
            }
        )

    @property
    def max_eigenvalue(self) -> float:
        """Largest absolute eigenvalue of the VAR (persistence of the factors)."""
        return float(np.max(np.abs(np.linalg.eigvals(self.phi))))


def _nearest_column(df: pd.DataFrame, maturity: float) -> float:
    cols = np.asarray(df.columns, dtype=float)
    return float(cols[np.argmin(np.abs(cols - maturity))])


# =============================================================================
# Inputs
# =============================================================================


def zero_panel(
    params: pd.DataFrame, max_months: int = 120, freq: str | None = "ME"
) -> pd.DataFrame:
    """Zero yields on the monthly maturity grid ``1..max_months`` from NSS parameters.

    ``params`` has the columns ``beta0 … lambda2`` (a :class:`~nss_engine.PanelFit`'s
    ``params``, or the Fed's curve from :func:`~nss_engine.data.load_gsw_parameters`).
    With ``freq='ME'`` the last curve of each month is used, the convention of
    ACM's monthly model; the index is then the month end.
    """
    p = params
    if freq is not None:
        p = params.resample(freq).last().dropna()
    tau = np.arange(1, max_months + 1) / 12.0
    betas = p[["beta0", "beta1", "beta2", "beta3"]].to_numpy()
    lam = p[["lambda1", "lambda2"]].to_numpy()
    out = np.empty((len(p), tau.size))
    for i in range(len(p)):
        out[i] = nss_loadings(tau, lam[i, 0], lam[i, 1]) @ betas[i]
    return pd.DataFrame(out, index=p.index, columns=tau)


def zero_panel_from_curves(curves: Sequence[NSSCurve], index: pd.Index) -> pd.DataFrame:
    """Like :func:`zero_panel`, from a list of curves (no resampling)."""
    params = pd.DataFrame([c.as_array() for c in curves], index=index)
    params.columns = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    return zero_panel(params, freq=None)


# =============================================================================
# Estimation
# =============================================================================


def fit_acm(
    zero_yields: pd.DataFrame,
    n_factors: int = 5,
    return_maturities: Sequence[int] = DEFAULT_RETURN_MATURITIES,
    max_eigenvalue: float | None = 0.998,
) -> ACMResult:
    """Estimate the Adrian-Crump-Moench model on a monthly panel of zero yields.

    Parameters
    ----------
    zero_yields:
        Continuously compounded zero yields in percent, one row per month and
        columns ``1/12, 2/12, …, N/12`` (every monthly maturity up to ``N``
        months, as produced by :func:`zero_panel`). No missing values.
    n_factors:
        Number of principal components used as pricing factors (ACM use 5).
    return_maturities:
        Maturities ``n`` (months) whose one-month holding-period returns enter
        the risk-price regression; each needs the ``n − 1`` month yield too.
    max_eigenvalue:
        Cap on the persistence of the real-world VAR. On short samples OLS can
        return an explosive VAR, which sends ten-year expectations to infinity.
        If its largest root exceeds the cap, ``Φ`` is scaled down to it (and
        ``μ`` reset so the factors keep their sample mean) while the
        risk-neutral dynamics ``Φ − λ1`` and ``μ − λ0`` are held fixed, so the
        fitted yields do not change; only the split into expectations and term
        premium does. ``None`` disables the cap.
    """
    Y, months = _validate(zero_yields)
    n_max = int(months[-1])
    ret_mats = tuple(int(n) for n in return_maturities if 2 <= n <= n_max)
    if len(ret_mats) < n_factors:
        raise ValueError("need at least as many return maturities as factors")
    T = Y.shape[0]
    min_months = 3 * n_factors + 10
    if min_months > T:
        raise ValueError(f"need more than {min_months} months of data, got {T}")

    y = Y / _MONTHLY  # monthly log yields, decimal
    # ---- factors: principal components of the yield levels -------------------------
    ybar = y.mean(axis=0)
    _, _, Vt = np.linalg.svd(y - ybar, full_matrices=False)
    W = Vt[:n_factors].T  # N × K loadings (orthonormal)
    X = (y - ybar) @ W  # T × K
    # sign convention: each factor positively related to the average yield
    signs = np.sign(W.sum(axis=0))
    signs[signs == 0] = 1.0
    W, X = W * signs, X * signs

    # ---- VAR(1) under the real-world measure ---------------------------------------
    Z = np.column_stack([np.ones(T - 1), X[:-1]])
    coef, *_ = np.linalg.lstsq(Z, X[1:], rcond=None)
    mu, phi = coef[0], coef[1:].T
    V = X[1:] - Z @ coef  # (T-1) × K innovations
    sigma = V.T @ V / (T - 1)

    # ---- excess returns ------------------------------------------------------------
    col = {int(m): j for j, m in enumerate(months)}
    logp = -y * months[None, :]  # log prices
    r = y[:, col[1]]  # one-month rate
    rx = np.column_stack(
        [logp[1:, col[n - 1]] - logp[:-1, col[n]] - r[:-1] for n in ret_mats]
    )  # (T-1) × N_r

    # ---- regression rx = a + β'v + c X_{t} + e -------------------------------------
    R = np.column_stack([np.ones(T - 1), V, X[:-1]])
    B, *_ = np.linalg.lstsq(R, rx, rcond=None)
    k = n_factors
    a = B[0]  # N_r
    beta = B[1 : 1 + k]  # K × N_r
    c = B[1 + k :].T  # N_r × K
    E = rx - R @ B
    sigma2 = float(np.sum(E**2) / E.size)

    # ---- prices of risk ------------------------------------------------------------
    bstar = np.array([np.outer(beta[:, i], beta[:, i]).ravel() for i in range(beta.shape[1])])
    bbt_inv_b = np.linalg.solve(beta @ beta.T, beta)  # (ββ')⁻¹β, K × N_r
    lambda1 = bbt_inv_b @ c
    lambda0 = bbt_inv_b @ (a + 0.5 * (bstar @ sigma.ravel() + sigma2))

    # ---- keep the real-world VAR stationary (fitted yields unchanged) ---------------
    rho = float(np.max(np.abs(np.linalg.eigvals(phi))))
    capped = max_eigenvalue is not None and rho > max_eigenvalue
    if max_eigenvalue is not None and capped:
        mu_q, phi_q = mu - lambda0, phi - lambda1
        phi = phi * (max_eigenvalue / rho)
        mu = (np.eye(k) - phi) @ X.mean(axis=0)
        lambda0, lambda1 = mu - mu_q, phi - phi_q

    # ---- short rate and pricing recursions -----------------------------------------
    Zr = np.column_stack([np.ones(T), X])
    d, *_ = np.linalg.lstsq(Zr, r, rcond=None)
    delta0, delta1 = float(d[0]), d[1:]
    A, Bn = affine_loadings(n_max, mu - lambda0, phi - lambda1, sigma, sigma2, delta0, delta1)
    A_rn, B_rn = affine_loadings(n_max, mu, phi, sigma, sigma2, delta0, delta1)

    n = np.arange(1, n_max + 1, dtype=float)
    fitted = -(A[None, :] + X @ Bn.T) / n[None, :] * _MONTHLY
    risk_neutral = -(A_rn[None, :] + X @ B_rn.T) / n[None, :] * _MONTHLY
    idx, cols = zero_yields.index, zero_yields.columns
    return ACMResult(
        observed=zero_yields.copy(),
        fitted=pd.DataFrame(fitted[:, months.astype(int) - 1], index=idx, columns=cols),
        risk_neutral=pd.DataFrame(risk_neutral[:, months.astype(int) - 1], index=idx, columns=cols),
        factors=pd.DataFrame(X, index=idx, columns=[f"PC{i + 1}" for i in range(k)]),
        mu=mu,
        phi=phi,
        sigma=sigma,
        lambda0=lambda0,
        lambda1=lambda1,
        delta0=delta0,
        delta1=delta1,
        sigma_e=float(np.sqrt(sigma2)),
        return_maturities=ret_mats,
        var_capped=bool(capped),
    )


def _validate(zero_yields: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    months = np.round(np.asarray(zero_yields.columns, dtype=float) * 12.0, 6)
    if not np.allclose(months, np.round(months)):
        raise ValueError("columns must be maturities on the monthly grid (k/12 years)")
    expected = np.arange(1, round(months.max()) + 1)
    if months.size != expected.size or not np.array_equal(np.round(months), expected):
        raise ValueError("columns must include every monthly maturity from 1 month up")
    Y = zero_yields.to_numpy(dtype=float)
    if not np.isfinite(Y).all():
        raise ValueError("zero yields must not contain missing values")
    return Y, np.round(months)


def affine_loadings(
    n_max: int,
    mu_q: FloatArray,
    phi_q: FloatArray,
    sigma: FloatArray,
    sigma2: float,
    delta0: float,
    delta1: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Affine bond-price loadings: ``log P_t^(n) = A_n + B_n' X_t`` for n = 1..n_max."""
    k = delta1.size
    A = np.empty(n_max)
    B = np.empty((n_max, k))
    a_prev, b_prev = 0.0, np.zeros(k)
    for i in range(n_max):
        # the (n-1)-month bond is cash when n = 1: no return, so no pricing error
        err = sigma2 if i > 0 else 0.0
        a_prev = a_prev + float(b_prev @ mu_q + 0.5 * (b_prev @ sigma @ b_prev + err)) - delta0
        b_prev = phi_q.T @ b_prev - delta1
        A[i], B[i] = a_prev, b_prev
    return A, B


# =============================================================================
# Pseudo-real-time decomposition
# =============================================================================


def real_time_decomposition(
    zero_yields: pd.DataFrame,
    maturity: float = 10.0,
    min_train: int = 120,
    n_factors: int = 5,
    max_eigenvalue: float | None = 0.998,
    max_fit_error_bp: float = 10.0,
) -> pd.DataFrame:
    """Re-estimate the model every month on data up to that month only.

    Returns, for each month from ``min_train`` on, the latest ``fitted``,
    ``expected_short_rate`` and ``term_premium`` for ``maturity`` that the
    model estimated *at the time*, and whether the stationarity cap bound
    (``var_capped``). A full-sample estimate uses the future to
    split past yields; this version does not, so it can be used to evaluate
    forecasts honestly.

    On short windows the smallest principal components can be almost pure
    noise, and the estimated risk-neutral dynamics then explode at long
    maturities. An estimate whose average yield fitting error exceeds
    ``max_fit_error_bp`` is recorded as missing (``NaN``) rather than used.
    """
    col = _nearest_column(zero_yields, maturity)
    j = list(zero_yields.columns).index(col)
    rows = {}
    for t in range(min_train - 1, len(zero_yields)):
        res = fit_acm(zero_yields.iloc[: t + 1], n_factors=n_factors, max_eigenvalue=max_eigenvalue)
        ok = bool(res.fit_rmse_bp.mean() <= max_fit_error_bp)
        rows[zero_yields.index[t]] = (
            res.fitted.iloc[-1, j] if ok else np.nan,
            res.risk_neutral.iloc[-1, j] if ok else np.nan,
            res.var_capped,
        )
    out = pd.DataFrame.from_dict(
        rows, orient="index", columns=["fitted", "expected_short_rate", "var_capped"]
    )
    out["term_premium"] = out["fitted"] - out["expected_short_rate"]
    out = out[["fitted", "expected_short_rate", "term_premium", "var_capped"]]
    out.index.name = zero_yields.index.name
    return out


# =============================================================================
# Comparison with a published term premium
# =============================================================================


def compare_term_premia(estimate: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Agreement between two monthly term-premium series (percent).

    Returns the correlation of levels and of 12-month changes, the RMSE, the
    mean gap (estimate − benchmark, bp) and the number of common months.
    """
    a = estimate.copy()
    b = benchmark.copy()
    a.index = pd.DatetimeIndex(a.index).to_period("M")
    b.index = pd.DatetimeIndex(b.index).to_period("M")
    b = b.groupby(level=0).mean()
    a = a.groupby(level=0).last()
    both = pd.concat([a.rename("est"), b.rename("ref")], axis=1).dropna()
    if len(both) < 24:
        raise ValueError("fewer than 24 common months")
    d12 = both.diff(12).dropna()
    gap = both["est"] - both["ref"]
    return {
        "corr_level": float(both["est"].corr(both["ref"])),
        "corr_change_12m": float(d12["est"].corr(d12["ref"])),
        "rmse_bp": float(np.sqrt(np.mean(gap**2)) * 100),
        "mean_gap_bp": float(gap.mean() * 100),
        "n_months": float(len(both)),
    }
