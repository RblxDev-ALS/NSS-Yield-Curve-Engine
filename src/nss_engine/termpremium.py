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
    #: how the real-world dynamics were estimated: ``"ols"``, ``"analytic"`` or
    #: ``"bootstrap"`` (bias-corrected), or ``"survey"`` (anchored to forecasts)
    p_dynamics: str = "ols"
    #: survey forecasts used, with the model's value for each (``model``)
    survey_fit: pd.DataFrame | None = None

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

    @property
    def survey_rmse(self) -> pd.Series:
        """Survey fitting error by series (percentage points); empty without surveys."""
        if self.survey_fit is None or self.survey_fit.empty:
            return pd.Series(dtype=float, name="survey_rmse")
        err = self.survey_fit["model"] - self.survey_fit["value"]
        out = np.sqrt((err**2).groupby(self.survey_fit["series"]).mean())
        return out.rename("survey_rmse")


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
    bias_correction: str | None = None,
    surveys: pd.DataFrame | None = None,
    survey_error: float | None = None,
    seed: int = 0,
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
    bias_correction:
        Correct the small-sample bias of the OLS real-world VAR, which makes
        rates look less persistent than they are (Bauer, Rudebusch & Wu 2012):
        ``"analytic"`` uses Pope's (1990) closed-form approximation,
        ``"bootstrap"`` BRW's inverse bootstrap (see :func:`bias_corrected_var`).
    surveys:
        Survey forecasts of the 3-month rate to anchor the real-world dynamics,
        in the format of :func:`~nss_engine.data.load_spf_bill_forecasts`
        (columns ``date``, ``series``, ``start``, ``end``, ``value``). The VAR
        is then estimated jointly on the factors and the surveys; see
        :func:`survey_anchored_var`.
    survey_error:
        Standard deviation (percentage points) of survey measurement errors.
        ``None`` estimates one per ``series`` from the fit (floored at 0.1).
    seed:
        Random seed of the bootstrap bias correction.

    In every variant the risk-neutral dynamics come from the cross-section
    of excess returns, as in ACM, so the fitted yields are identical; only the
    split into expected short rates and term premium changes.
    """
    if bias_correction is not None and surveys is not None:
        raise ValueError("use either a bias correction or survey anchors, not both")
    if bias_correction not in (None, "analytic", "bootstrap"):
        raise ValueError("bias_correction must be None, 'analytic' or 'bootstrap'")
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

    # ---- short rate and pricing under the risk-neutral measure ----------------------
    mu_q, phi_q = mu - lambda0, phi - lambda1
    Zr = np.column_stack([np.ones(T), X])
    d, *_ = np.linalg.lstsq(Zr, r, rcond=None)
    delta0, delta1 = float(d[0]), d[1:]
    A, Bn = affine_loadings(n_max, mu_q, phi_q, sigma, sigma2, delta0, delta1)

    # ---- real-world dynamics: bias correction or survey anchors --------------------
    p_mean = X.mean(axis=0)
    survey_fit = None
    if bias_correction is not None:
        phi = bias_corrected_var(X, phi, method=bias_correction, seed=seed)
        mu = (np.eye(k) - phi) @ p_mean
    elif surveys is not None:
        # the model's 3-month yield (percent) is c0 + c1'X
        c0, c1 = -A[2] / 3.0 * _MONTHLY, -Bn[2] / 3.0 * _MONTHLY
        xbar, phi_s, survey_fit = survey_anchored_var(
            X, zero_yields.index, phi, sigma, c0, c1, surveys, survey_error, max_eigenvalue
        )
        if not survey_fit.empty:  # no survey in the sample: keep OLS
            p_mean, phi = xbar, phi_s
            mu = (np.eye(k) - phi) @ p_mean

    # ---- keep the real-world VAR stationary (fitted yields unchanged) ---------------
    rho = float(np.max(np.abs(np.linalg.eigvals(phi))))
    capped = max_eigenvalue is not None and rho > max_eigenvalue
    if max_eigenvalue is not None and capped:
        phi = phi * (max_eigenvalue / rho)
        mu = (np.eye(k) - phi) @ p_mean
    lambda0, lambda1 = mu - mu_q, phi - phi_q
    if survey_fit is not None:
        survey_fit = _survey_model_values(survey_fit, X, mu, phi, A, Bn)
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
        p_dynamics=bias_correction or ("survey" if surveys is not None else "ols"),
        survey_fit=survey_fit,
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
# Real-world dynamics: small-sample bias correction
# =============================================================================


def var_bias(phi: FloatArray, sigma: FloatArray, n_obs: int) -> FloatArray:
    """Pope's (1990) approximation to the bias of the OLS VAR(1) slope.

    For ``X_{t+1} = μ + Φ X_t + v`` estimated by OLS with an intercept on
    ``n_obs`` transitions::

        E[Φ̂] − Φ ≈ −Σ [(I − Φ')⁻¹ + Φ'(I − Φ'²)⁻¹ + Σ_i λ_i (I − λ_i Φ')⁻¹] Γ₀⁻¹ / n_obs

    with ``λ_i`` the eigenvalues of ``Φ`` and ``Γ₀`` the unconditional variance
    of ``X``. In one dimension it is Kendall's ``−(1 + 3ρ)/n``. Needs a
    stationary ``Φ``.
    """
    from scipy.linalg import solve_discrete_lyapunov

    k = phi.shape[0]
    eye = np.eye(k)
    pt = phi.T
    gamma0 = solve_discrete_lyapunov(phi, sigma)
    term = np.linalg.inv(eye - pt) + pt @ np.linalg.inv(eye - pt @ pt)
    for lam in np.linalg.eigvals(phi):
        term = term + lam * np.linalg.inv(eye - lam * pt)
    bias = -sigma @ np.real(term) @ np.linalg.inv(gamma0) / n_obs
    return np.asarray(bias, dtype=float)


def _spectral_radius(phi: FloatArray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(phi))))


def _shrink_to_stationary(phi_hat: FloatArray, correction: FloatArray) -> FloatArray:
    """Kilian's (1998) rule: ``Φ̂ + δ·correction`` with the largest δ ≤ 1 that is stationary."""
    if _spectral_radius(phi_hat) >= 1.0:
        return phi_hat
    for delta in np.arange(1.0, 0.0, -0.01):
        cand = phi_hat + delta * correction
        if _spectral_radius(cand) < 1.0:
            return cand
    return phi_hat


def _ols_var_slopes(paths: FloatArray) -> FloatArray:
    """OLS VAR(1) slope (with intercept) of each of ``B`` paths, shape ``(B, T, K)``."""
    b, t, _ = paths.shape
    Z = np.concatenate([np.ones((b, t - 1, 1)), paths[:, :-1]], axis=2)
    ZZ = np.einsum("bti,btj->bij", Z, Z)
    ZY = np.einsum("bti,btj->bij", Z, paths[:, 1:])
    coef = np.linalg.solve(ZZ, ZY)
    return np.transpose(coef[:, 1:, :], (0, 2, 1))


def bias_corrected_var(
    X: FloatArray,
    phi: FloatArray,
    method: str = "analytic",
    n_boot: int = 200,
    n_iter: int = 8,
    seed: int = 0,
) -> FloatArray:
    """Bias-corrected slope of a VAR(1) fitted by OLS to ``X`` (``T × K``).

    OLS underestimates persistence in short samples, so expected future rates
    revert to their mean too quickly and a spurious share of long yields is
    attributed to the term premium (Bauer, Rudebusch & Wu, 2012).

    * ``"analytic"``: ``Φ̂ − bias(Φ̂)`` with Pope's formula (:func:`var_bias`).
    * ``"bootstrap"``: BRW's inverse bootstrap - the ``Φ`` whose OLS estimates
      on simulated samples average ``Φ̂``, found by the fixed-point iteration
      ``Φ ← Φ + (Φ̂ − mean OLS(Φ))`` with resampled residuals held fixed.

    Either way the correction is shrunk until the VAR is stationary (Kilian,
    1998); an explosive ``Φ̂`` is returned unchanged.
    """
    T, k = X.shape
    Z = np.column_stack([np.ones(T - 1), X[:-1]])
    coef, *_ = np.linalg.lstsq(Z, X[1:], rcond=None)
    V = X[1:] - Z @ coef
    sigma = V.T @ V / (T - 1)
    if _spectral_radius(phi) >= 1.0:
        return phi
    if method == "analytic":
        return _shrink_to_stationary(phi, -var_bias(phi, sigma, T - 1))
    if method != "bootstrap":
        raise ValueError("method must be 'analytic' or 'bootstrap'")
    rng = np.random.default_rng(seed)
    shocks = (V - V.mean(axis=0))[rng.integers(0, T - 1, size=(n_boot, T - 1))]
    d0 = X[0] - X.mean(axis=0)

    def mean_ols(ph: FloatArray) -> FloatArray:
        paths = np.empty((n_boot, T, k))
        paths[:, 0] = d0
        for t in range(1, T):
            paths[:, t] = paths[:, t - 1] @ ph.T + shocks[:, t - 1]
        return np.asarray(_ols_var_slopes(paths).mean(axis=0))

    current = phi.copy()
    for _ in range(n_iter):
        step = phi - mean_ols(current)
        current = _shrink_to_stationary(current, step)
    return _shrink_to_stationary(phi, current - phi)


# =============================================================================
# Real-world dynamics: survey anchors
# =============================================================================

#: Floor on estimated survey measurement errors (percentage points).
MIN_SURVEY_ERROR = 0.10


def _align_surveys(index: pd.Index, surveys: pd.DataFrame) -> pd.DataFrame:
    needed = {"date", "series", "start", "end", "value"}
    if not needed <= set(surveys.columns):
        raise ValueError(f"surveys need the columns {sorted(needed)}")
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("survey anchors need yields with a DatetimeIndex")
    months = index.to_period("M")
    pos = pd.Series(np.arange(len(index)), index=months)
    pos = pos[~pos.index.duplicated(keep="last")]
    sv = surveys.dropna(subset=["value"]).copy()
    sv["row"] = pos.reindex(pd.DatetimeIndex(sv["date"]).to_period("M")).to_numpy()
    sv = sv.dropna(subset=["row"])
    sv["row"] = sv["row"].astype(int)
    if (sv["start"] < 0).any() or (sv["end"] < sv["start"]).any():
        raise ValueError("survey windows need 0 <= start <= end")
    return sv.reset_index(drop=True)


def _window_loadings(
    phi: FloatArray, g: FloatArray, windows: FloatArray, horizon: int
) -> FloatArray:
    """``g' · mean(Φ^m for m in [start, end])`` for each window (rows of ``windows``)."""
    k = phi.shape[0]
    cum = np.empty((horizon + 2, k))  # cum[m] = g' Σ_{j<m} Φ^j
    cum[0] = 0.0
    row = g.copy()
    for m in range(horizon + 1):
        cum[m + 1] = cum[m] + row
        row = row @ phi
    s, e = windows[:, 0], windows[:, 1]
    out: FloatArray = (cum[e + 1] - cum[s]) / (e - s + 1)[:, None]
    return out


def survey_anchored_var(
    X: FloatArray,
    index: pd.Index,
    phi0: FloatArray,
    sigma: FloatArray,
    c0: float,
    c1: FloatArray,
    surveys: pd.DataFrame,
    survey_error: float | None = None,
    max_eigenvalue: float | None = 0.998,
) -> tuple[FloatArray, FloatArray, pd.DataFrame]:
    """Estimate the real-world VAR from the factors *and* survey forecasts.

    Minimises, over the factor mean ``x̄`` and slope ``Φ``,

        Σ_t v_t' Σ⁻¹ v_t + Σ_i ((f_i(x̄, Φ) − s_i) / σ_i)²,

    the Gaussian likelihood of the VAR plus survey measurement errors. Survey
    ``i`` reports, at month ``t_i``, the average 3-month yield expected over
    months ``start … end`` ahead; the model's value is
    ``f_i = c0 + c1'(x̄ + mean_m Φ^m (X_{t_i} − x̄))`` (``c0 + c1'X`` is the
    model's 3-month yield in percent). This is how Kim & Wright (2005) and
    Kim & Orphanides (2012) discipline expectations. Only surveys dated within
    the sample are used, so real-time estimates see only published surveys.

    Returns ``x̄``, ``Φ`` and the surveys used. With ``survey_error=None`` each
    series gets its own error, estimated from the fit (at least
    :data:`MIN_SURVEY_ERROR`). A penalty keeps the largest root of ``Φ`` below
    ``max_eigenvalue``: near a unit root the mean ``x̄`` stops mattering to
    the fit and can drift anywhere, which a later cap would expose.
    """
    from scipy.optimize import least_squares

    sv = _align_surveys(index, surveys)
    k = X.shape[1]
    xbar0 = X.mean(axis=0)
    if sv.empty:
        return xbar0, phi0, sv.assign(model=np.nan)
    # work in percent-like units so all parameters are O(1)
    Xs = X * _MONTHLY
    g = np.asarray(c1, dtype=float) / _MONTHLY
    chol = np.linalg.cholesky(sigma * _MONTHLY**2)
    rows = sv["row"].to_numpy()
    win = sv[["start", "end"]].to_numpy(dtype=int)
    uniq, inv = np.unique(win, axis=0, return_inverse=True)
    inv = np.ravel(inv)
    horizon = int(uniq[:, 1].max())
    values = sv["value"].to_numpy(dtype=float)
    series = sv["series"].astype(str).to_numpy()

    def residuals(theta: FloatArray, sd: FloatArray) -> FloatArray:
        xb, ph = theta[:k], theta[k:].reshape(k, k)
        D = Xs - xb
        V = D[1:] - D[:-1] @ ph.T
        rv = np.linalg.solve(chol, V.T).ravel()
        rho = _spectral_radius(ph)
        # trial steps can be explosive: bound the powers so Φ^120 stays finite
        G = _window_loadings(ph * min(1.0, 1.02 / rho), g, uniq, horizon)
        f = c0 + g @ xb + np.einsum("ij,ij->i", G[inv], D[rows])
        excess = 0.0 if max_eigenvalue is None else rho - max_eigenvalue
        return np.concatenate([rv, (f - values) / sd, [1e4 * max(excess, 0.0)]])

    theta = np.concatenate([xbar0 * _MONTHLY, phi0.ravel()])
    sd = np.full(values.size, 0.25 if survey_error is None else float(survey_error))
    for _ in range(1 if survey_error is not None else 3):
        sol = least_squares(residuals, theta, args=(sd,), method="trf", x_scale="jac")
        theta = sol.x
        if survey_error is None:
            err = residuals(theta, np.ones_like(sd))[-values.size - 1 : -1]
            rms = pd.Series(err**2).groupby(series).mean() ** 0.5
            sd = np.maximum(rms.reindex(series).to_numpy(), MIN_SURVEY_ERROR)
    return theta[:k] / _MONTHLY, theta[k:].reshape(k, k), sv.assign(error_sd=sd)


def _survey_model_values(
    sv: pd.DataFrame, X: FloatArray, mu: FloatArray, phi: FloatArray, A: FloatArray, Bn: FloatArray
) -> pd.DataFrame:
    """Model values of the surveys under the final real-world dynamics."""
    if sv.empty:
        return sv.drop(columns="row").assign(model=np.nan)
    k = phi.shape[0]
    c0, c1 = -A[2] / 3.0 * _MONTHLY, -Bn[2] / 3.0 * _MONTHLY
    try:
        xbar = np.linalg.solve(np.eye(k) - phi, mu)
    except np.linalg.LinAlgError:  # pragma: no cover - unit root without a cap
        return sv.assign(model=np.nan)
    win = sv[["start", "end"]].to_numpy(dtype=int)
    G = _window_loadings(phi, c1, win, int(win[:, 1].max()))
    D = X[sv["row"].to_numpy()] - xbar
    model = c0 + c1 @ xbar + np.einsum("ij,ij->i", G, D)
    out = sv.assign(model=model)
    return out.drop(columns="row")


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
    bias_correction: str | None = None,
    surveys: pd.DataFrame | None = None,
    survey_error: float | None = None,
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

    ``bias_correction``, ``surveys`` and ``survey_error`` are passed to
    :func:`fit_acm`; each month's estimate uses only the surveys published by
    then.
    """
    col = _nearest_column(zero_yields, maturity)
    j = list(zero_yields.columns).index(col)
    rows = {}
    for t in range(min_train - 1, len(zero_yields)):
        res = fit_acm(
            zero_yields.iloc[: t + 1],
            n_factors=n_factors,
            max_eigenvalue=max_eigenvalue,
            bias_correction=bias_correction,
            surveys=surveys,
            survey_error=survey_error,
        )
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
