"""Dynamic Nelson-Siegel as a state-space model, estimated by maximum likelihood.

Diebold, Rudebusch & Aruoba (2006) write the dynamic Nelson-Siegel model as a
linear Gaussian state-space system::

    y_t = Λ(λ) f_t + ε_t                ε_t ~ N(0, H),  H diagonal
    f_t − μ = A (f_{t−1} − μ) + η_t     η_t ~ N(0, Q)

with ``f_t = (level, slope, curvature)``. Unlike the two-step Diebold-Li
approach (:mod:`nss_engine.forecasting`), which fixes ``λ``, runs one
cross-sectional regression per date and then fits a VAR to those *estimates*,
the Kalman filter estimates ``λ``, the factor dynamics and the measurement
noise **jointly**. It also

* handles missing tenors exactly (an unobserved maturity simply drops out of
  that date's update),
* weighs each maturity by its own noise variance, and
* produces full **predictive distributions** - forecast intervals whose
  coverage can be checked out of sample.

Because ``H`` is diagonal, the update uses the information form
(Sherman-Morrison-Woodbury): every step inverts only 3 × 3 matrices, however
many maturities are observed.

``level_unit_root=True`` makes the level factor a random walk (``A[0, 0] = 1``,
no mean reversion). The level of U.S. rates is close to a unit root and a
mean-reverting level pulls forecasts toward the sample mean - one reason
Diebold-Li forecasts lose to the random walk after 2000.

Arbitrage-free Nelson-Siegel (AFNS)
-----------------------------------
The dynamic Nelson-Siegel model is not free of arbitrage: its factor dynamics
and its loadings are chosen separately. Christensen, Diebold & Rudebusch
(2011) show that one small change makes it arbitrage-free. If the factors
follow a Gaussian diffusion with volatility ``Σ`` and a particular
risk-neutral drift, bond yields have **exactly** the Nelson-Siegel loadings
plus a maturity-dependent constant::

    y_t(τ) = Λ(λ) f_t − C(τ)/τ,     C(τ)/τ = 1/(2τ) ∫₀^τ b(u)ᵀ ΣΣᵀ b(u) du
    b(u) = (u, (1 − e^{−λu})/λ, (1 − e^{−λu})/λ − u e^{−λu})

The *yield-adjustment term* ``C(τ)/τ`` is a convexity effect: volatile rates
make long bonds worth more, so their yields sit below what the loadings say.
It is negligible at short maturities and grows roughly with ``τ²`` (a few basis
points at 10 years, tens at 30). With ``arbitrage_free=True`` the state-space
model subtracts it from the measurement equation, with ``ΣΣᵀ`` taken from the
state innovation covariance (``Q / Δt``), so the cross-section and the
dynamics are estimated consistently. CDR find the AFNS restriction improves
forecasts, especially with independent factors (``independent=True``:
diagonal ``A`` and ``Q``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.stats import norm

from .data import maturity_label
from .forecasting import combination_errors, extract_factors, fit_factor_model
from .models import DIEBOLD_LI_LAMBDA, FloatArray, ns_loadings

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class DNSParameters:
    """Parameters of the state-space dynamic Nelson-Siegel model."""

    lam: float  #: decay rate λ (1/years)
    mu: FloatArray  #: factor means (level mean ignored when the level has a unit root)
    A: FloatArray  #: 3 × 3 transition matrix
    Q: FloatArray  #: 3 × 3 state innovation covariance
    h: FloatArray  #: measurement noise standard deviations (percent), one per maturity
    arbitrage_free: bool = False  #: subtract the AFNS yield-adjustment term
    dt: float = 1.0 / 12.0  #: length of one period in years (monthly data)

    def loadings(self, maturities: ArrayLike) -> FloatArray:
        return ns_loadings(maturities, self.lam)

    def intercept(self, maturities: ArrayLike) -> FloatArray:
        """Measurement intercept (percent): ``−C(τ)/τ`` for AFNS, zero otherwise."""
        tau = np.atleast_1d(np.asarray(maturities, dtype=float))
        if not self.arbitrage_free:
            return np.zeros(tau.size)
        # Q is in %² per period; ΣΣᵀ in decimal² per year is Q / 1e4 / dt, and the
        # adjustment (decimal) is linear in ΣΣᵀ, so in percent it is adj(Q/dt) / 100.
        return -afns_yield_adjustment(tau, self.lam, self.Q / self.dt) / 100.0


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(48)


def afns_adjustment_matrices(maturities: ArrayLike, lam: float) -> FloatArray:
    """``M(τ) = 1/(2τ) ∫₀^τ b(u) b(u)ᵀ du`` for each maturity, shape ``(n, 3, 3)``.

    The AFNS yield adjustment is ``C(τ)/τ = Σ_ij M_ij(τ) (ΣΣᵀ)_ij``. The integrand
    is smooth, so 48-point Gauss-Legendre quadrature is exact to rounding.
    """
    tau = np.atleast_1d(np.asarray(maturities, dtype=float))
    u = 0.5 * tau[:, None] * (_GL_NODES[None, :] + 1.0)  # n × q nodes on [0, τ]
    w = 0.5 * tau[:, None] * _GL_WEIGHTS[None, :]
    e = np.exp(-lam * u)
    b = np.stack([u, -np.expm1(-lam * u) / lam, -np.expm1(-lam * u) / lam - u * e], axis=-1)
    M = np.einsum("nq,nqi,nqj->nij", w, b, b)
    return M / (2.0 * tau[:, None, None])


def afns_yield_adjustment(maturities: ArrayLike, lam: float, omega: ArrayLike) -> FloatArray:
    """AFNS yield-adjustment term ``C(τ)/τ`` for factor covariance ``ΣΣᵀ = omega``.

    Units follow ``omega``: with ``ΣΣᵀ`` in decimal² per year the result is a
    decimal yield. Christensen, Diebold & Rudebusch (2011), eq. (22) in general
    form; the model yield is the Nelson-Siegel yield *minus* this term.
    """
    M = afns_adjustment_matrices(maturities, lam)
    return np.einsum("nij,ij->n", M, np.asarray(omega, dtype=float))


@dataclass(frozen=True)
class KalmanOutput:
    loglik: float
    filtered: FloatArray  #: T × 3 filtered factors E[f_t | y_1..t]
    filtered_cov: FloatArray  #: T × 3 × 3


def kalman_filter(y: FloatArray, maturities: FloatArray, p: DNSParameters) -> KalmanOutput:
    """Information-form Kalman filter; ``NaN`` entries of ``y`` are treated as missing.

    The covariance recursion does not depend on the data, so once it has
    converged for the current pattern of observed maturities the filter is in
    steady state: ``f_t = G f_{t−1} + b_t`` with a constant ``G``. Such
    stretches are run in one vectorised pass (``G`` is diagonalised and each
    mode is a scalar recursion evaluated by :func:`scipy.signal.lfilter`), so
    the cost is dominated by the few transient steps after each change in the
    set of observed maturities.
    """
    if p.arbitrage_free:
        y = y - p.intercept(maturities)[None, :]
    T = y.shape[0]
    L = p.loadings(maturities)
    A = p.A
    c = p.mu - A @ p.mu  # f_t = c + A f_{t-1} + η_t
    stationary = bool(np.all(np.abs(np.linalg.eigvals(A)) < 0.999))
    f = p.mu.copy()
    P = solve_discrete_lyapunov(A, p.Q) if stationary else np.eye(3) * 10.0
    inv_h2 = 1.0 / p.h**2
    log_h2 = np.log(p.h**2)
    observed = np.isfinite(y)
    y0 = np.where(observed, y, 0.0)
    keys = [row.tobytes() for row in observed]
    # end (exclusive) of the run of identical observation patterns containing t
    run_end = np.empty(T, dtype=int)
    end = T
    for t in range(T - 1, -1, -1):
        if t < T - 1 and keys[t] != keys[t + 1]:
            end = t + 1
        run_end[t] = end
    loglik = 0.0
    filt = np.empty((T, 3))
    filt_cov = np.empty((T, 3, 3))
    patterns: dict[bytes, tuple[FloatArray, FloatArray, float]] = {}
    t = 0
    while t < T:
        obs = observed[t]
        key = keys[t]
        n_obs = int(obs.sum())
        if key not in patterns:
            Lo = L[obs]
            w = inv_h2[obs]
            patterns[key] = (Lo.T @ (Lo * w[:, None]), Lo.T * w, float(log_h2[obs].sum()))
        info, LtW, logdet_h = patterns[key]
        if t > 0:
            f = c + A @ f
            P_pred = A @ P @ A.T + p.Q
        else:
            P_pred = P
        if n_obs == 0:
            P = P_pred
            filt[t], filt_cov[t] = f, P
            t += 1
            continue
        M_inv = np.linalg.inv(np.linalg.inv(P_pred) + info)
        logdet = float(logdet_h + np.linalg.slogdet(P_pred)[1] - np.linalg.slogdet(M_inv)[1])
        converged = t > 0 and np.max(np.abs(M_inv - P)) < 1e-12 * max(1.0, float(np.abs(P).max()))
        P = M_inv
        Lo = L[obs]
        w = inv_h2[obs]
        v = y0[t, obs] - Lo @ f
        LtWv = LtW @ v
        loglik += -0.5 * (n_obs * _LOG_2PI + logdet + float(v @ (w * v) - LtWv @ P @ LtWv))
        f = f + P @ LtWv
        filt[t], filt_cov[t] = f, P
        t += 1
        stop = run_end[t - 1]
        if converged and t < stop:
            f, block_ll = _steady_state_block(y0[t:stop][:, obs], f, A, c, P, Lo, LtW, w, filt, t)
            loglik += block_ll + (stop - t) * -0.5 * (n_obs * _LOG_2PI + logdet)
            filt_cov[t:stop] = P
            t = stop
    return KalmanOutput(float(loglik), filt, filt_cov)


def _steady_state_block(
    Y: FloatArray,
    f_prev: FloatArray,
    A: FloatArray,
    c: FloatArray,
    P: FloatArray,
    Lo: FloatArray,
    LtW: FloatArray,
    w: FloatArray,
    out: FloatArray,
    t0: int,
) -> tuple[FloatArray, float]:
    """Run a constant-gain stretch of the filter at once; returns (last f, Σ −½·quad)."""
    from scipy.signal import lfilter

    n = Y.shape[0]
    K = P @ LtW  # 3 × n_obs gain
    Gp = np.eye(3) - K @ Lo  # update applied to the prediction
    G = Gp @ A
    B = (Gp @ c)[None, :] + Y @ K.T  # b_t, n × 3
    evals, V = np.linalg.eig(G)
    if np.linalg.cond(V) > 1e8:  # (near-)defective G: plain recursion
        F = np.empty((n, 3))
        f = f_prev
        for i in range(n):
            f = G @ f + B[i]
            F[i] = f
    else:
        Vinv = np.linalg.inv(V)
        Z = B @ Vinv.T
        z0 = Vinv @ f_prev
        Zs = np.empty_like(Z)
        for j in range(3):
            Zs[:, j] = lfilter([1.0], [1.0, -evals[j]], Z[:, j], zi=[evals[j] * z0[j]])[0]
        F = np.real(Zs @ V.T)
    out[t0 : t0 + n] = F
    prev = np.vstack([f_prev[None, :], F[:-1]])
    pred = c[None, :] + prev @ A.T
    v = Y - pred @ Lo.T
    LtWv = v @ LtW.T
    quad = np.sum(w * v**2, axis=1) - np.einsum("ti,ij,tj->t", LtWv, P, LtWv)
    return F[-1], float(-0.5 * quad.sum())


# =============================================================================
# Parameter vector <-> DNSParameters
# =============================================================================


#: Lower bound on the measurement noise (percent, i.e. 1 bp). Without it the MLE
#: can drive one maturity's noise to zero - a degenerate optimum in which the
#: filter treats that yield as exact (seen on FRED data for the 3Y and 6M).
_H_FLOOR = 0.01


class _Transform:
    """Maps an unconstrained vector θ to model parameters."""

    def __init__(
        self,
        n: int,
        dynamics: str,
        fixed_lambda: float | None,
        level_unit_root: bool,
        arbitrage_free: bool = False,
        diagonal_q: bool = False,
    ) -> None:
        self.n, self.dynamics, self.fixed_lambda = n, dynamics, fixed_lambda
        self.level_unit_root = level_unit_root
        self.arbitrage_free, self.diagonal_q = arbitrage_free, diagonal_q

    def unpack(self, theta: FloatArray) -> DNSParameters:
        i = 0
        if self.fixed_lambda is None:
            lam = float(np.exp(theta[0]))
            i = 1
        else:
            lam = self.fixed_lambda
        mu = theta[i : i + 3].copy()
        i += 3
        if self.dynamics == "var":
            A = theta[i : i + 9].reshape(3, 3).copy()
            i += 9
        else:
            A = np.diag(theta[i : i + 3])
            i += 3
        if self.level_unit_root:
            A[0, :] = 0.0
            A[0, 0] = 1.0
        chol = np.zeros((3, 3))
        chol[np.tril_indices(3)] = theta[i : i + 6]
        chol[np.diag_indices(3)] = np.exp(chol[np.diag_indices(3)])
        if self.diagonal_q:
            chol = np.diag(np.diag(chol))
        i += 6
        Q = chol @ chol.T
        h = _H_FLOOR + np.exp(theta[i : i + self.n])
        return DNSParameters(lam, mu, A, Q, h, arbitrage_free=self.arbitrage_free)

    def n_free(self) -> int:
        """Number of free parameters (restricted entries are not counted)."""
        n_a = 9 if self.dynamics == "var" else 3
        if self.level_unit_root:
            n_a -= 3 if self.dynamics == "var" else 1
        n_q = 3 if self.diagonal_q else 6
        n_mu = 2 if self.level_unit_root else 3  # a random-walk level has no mean
        return (0 if self.fixed_lambda is not None else 1) + n_mu + n_a + n_q + self.n

    def pack(self, p: DNSParameters) -> FloatArray:
        parts = [] if self.fixed_lambda is not None else [np.array([np.log(p.lam)])]
        parts.append(p.mu)
        parts.append(p.A.ravel() if self.dynamics == "var" else np.diag(p.A))
        chol = np.linalg.cholesky(p.Q + 1e-12 * np.eye(3))
        low = chol[np.tril_indices(3)].copy()
        diag_pos = [0, 2, 5]  # positions of the diagonal in tril order
        low[diag_pos] = np.log(np.diag(chol))
        parts.append(low)
        parts.append(np.log(np.maximum(p.h - _H_FLOOR, 1e-6)))
        return np.concatenate(parts)


# =============================================================================
# Estimation and forecasting
# =============================================================================


@dataclass(frozen=True)
class DNSResult:
    params: DNSParameters
    maturities: FloatArray
    loglik: float
    n_obs: int
    factors: pd.DataFrame  #: filtered factors
    factor_cov: FloatArray  #: T × 3 × 3 filtered covariances
    converged: bool
    level_unit_root: bool
    n_free_params: int | None = None  #: number of estimated parameters, if known

    @property
    def n_params(self) -> int:
        if self.n_free_params is not None:
            return self.n_free_params
        return int(self.params.h.size + 3 + 6 + self.params.A.size + 1)

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.n_params * np.log(self.n_obs)

    def forecast(
        self, horizon: int, maturities: ArrayLike | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Mean and covariance of ``y_{T+h}`` given data up to ``T``."""
        p = self.params
        f = self.factors.to_numpy()[-1]
        P = self.factor_cov[-1]
        c = p.mu - p.A @ p.mu
        for _ in range(horizon):
            f = c + p.A @ f
            P = p.A @ P @ p.A.T + p.Q
        mats = self.maturities if maturities is None else np.asarray(maturities, dtype=float)
        L = p.loadings(mats)
        h = (
            p.h
            if maturities is None
            else np.interp(mats, self.maturities, p.h)  # noise of nearby maturities
        )
        return L @ f + p.intercept(mats), L @ P @ L.T + np.diag(h**2)

    def summary(self) -> pd.DataFrame:
        p = self.params
        rows = {
            "lambda": p.lam,
            "curvature peak (years)": 1.7933 / p.lam,
            "loglik": self.loglik,
            "BIC": self.bic,
        }
        for j, name in enumerate(("level", "slope", "curvature")):
            rows[f"A[{name},{name}]"] = float(p.A[j, j])
            rows[f"mean {name}"] = float(p.mu[j])
            rows[f"shock sd {name}"] = float(np.sqrt(p.Q[j, j]))
        for m, h in zip(self.maturities, p.h, strict=True):
            rows[f"noise sd {maturity_label(m)} (bp)"] = float(h * 100)
        if p.arbitrage_free:
            for m, a in zip((10.0, 30.0), p.intercept([10.0, 30.0]), strict=True):
                rows[f"yield adjustment {maturity_label(m)} (bp)"] = float(a * 100)
        return pd.Series(rows, name="value").to_frame()


def _starting_values(
    yields: pd.DataFrame, dynamics: str, lam: float, level_unit_root: bool
) -> DNSParameters:
    """Two-step Diebold-Li estimates: a good starting point for the MLE."""
    fac = extract_factors(yields, lam)
    model = fit_factor_model(fac, "var1" if dynamics == "var" else "ar1")
    A = model.coef.copy()
    # keep the start safely stationary so the Lyapunov initialisation exists
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if rho >= 0.995:
        A *= 0.995 / rho
    mu = fac.mean().to_numpy()
    Q = model.resid_cov + 1e-6 * np.eye(3)
    mats = np.asarray(yields.columns, dtype=float)
    fitted = fac.to_numpy() @ ns_loadings(mats, lam).T
    resid = yields.reindex(fac.index).to_numpy() - fitted
    h = np.sqrt(np.nanmean(resid**2, axis=0))
    h = np.where(np.isfinite(h) & (h > 1e-3), h, 0.05)
    if level_unit_root:
        A[0, :] = 0.0
        A[0, 0] = 1.0
    return DNSParameters(lam, mu, A, Q, h)


def fit_dns(
    yields: pd.DataFrame,
    dynamics: str = "var",
    level_unit_root: bool = False,
    fixed_lambda: float | None = None,
    start: DNSParameters | None = None,
    maxiter: int = 400,
    arbitrage_free: bool = False,
    independent: bool = False,
) -> DNSResult:
    """Maximum-likelihood estimate of the state-space dynamic Nelson-Siegel model.

    ``yields`` is a (typically monthly) panel with maturities as columns;
    ``NaN`` marks missing quotes. ``dynamics`` is ``"var"`` (full VAR(1)) or
    ``"diag"`` (independent AR(1) factors). ``start`` warm-starts the
    optimiser, e.g. from an estimate on a shorter sample.

    ``arbitrage_free=True`` estimates the AFNS model of Christensen, Diebold
    & Rudebusch (2011) (see the module notes); ``independent=True`` makes the
    factors fully independent (diagonal ``A`` *and* ``Q``), CDR's preferred
    forecasting specification.
    """
    if dynamics not in ("var", "diag"):
        raise ValueError("dynamics must be 'var' or 'diag'")
    if independent:
        dynamics = "diag"
    y = yields.to_numpy(dtype=float)
    mats = np.asarray(yields.columns, dtype=float)
    tr = _Transform(
        mats.size, dynamics, fixed_lambda, level_unit_root, arbitrage_free, diagonal_q=independent
    )
    lam0 = fixed_lambda if fixed_lambda is not None else DIEBOLD_LI_LAMBDA
    p0 = start or _starting_values(yields, dynamics, lam0, level_unit_root)
    theta0 = tr.pack(p0)

    def negloglik(theta: FloatArray) -> float:
        try:
            p = tr.unpack(theta)
            if not (0.02 < p.lam < 5.0):
                return 1e10
            val = -kalman_filter(y, mats, p).loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
        return val if np.isfinite(val) else 1e10

    res = minimize(negloglik, theta0, method="L-BFGS-B", options={"maxiter": maxiter})
    best = res.x if res.fun <= negloglik(theta0) else theta0
    p = tr.unpack(best)
    out = kalman_filter(y, mats, p)
    factors = pd.DataFrame(
        out.filtered, index=yields.index, columns=["level", "slope", "curvature"]
    )
    return DNSResult(
        params=p,
        maturities=mats,
        loglik=out.loglik,
        n_obs=int(np.isfinite(y).sum()),
        factors=factors,
        factor_cov=out.filtered_cov,
        converged=bool(res.success),
        level_unit_root=level_unit_root,
        n_free_params=tr.n_free(),
    )


# =============================================================================
# Out-of-sample evaluation
# =============================================================================


@dataclass(frozen=True)
class DNSForecastEvaluation:
    rmse_model: pd.DataFrame  #: horizon × tenor, bp
    rmse_random_walk: pd.DataFrame
    coverage: pd.DataFrame  #: share of outcomes inside the central interval
    interval: float
    n_forecasts: pd.Series
    rmse_combination: pd.DataFrame | None = None  #: RMSE (bp) of ½ model + ½ random walk

    @property
    def relative_rmse(self) -> pd.DataFrame:
        return self.rmse_model / self.rmse_random_walk

    @property
    def relative_rmse_combination(self) -> pd.DataFrame:
        if self.rmse_combination is None:
            raise ValueError("no combination forecasts were evaluated")
        return self.rmse_combination / self.rmse_random_walk


def evaluate_dns_forecasts(
    yields: pd.DataFrame,
    horizons: Sequence[int] = (1, 6, 12),
    min_train: int = 120,
    reestimate_every: int = 12,
    dynamics: str = "var",
    level_unit_root: bool = False,
    interval: float = 0.8,
    arbitrage_free: bool = False,
    independent: bool = False,
) -> DNSForecastEvaluation:
    """Rolling-origin forecasts vs the random walk, with interval coverage.

    Parameters are re-estimated every ``reestimate_every`` origins (warm-started
    from the previous estimate) on data up to the origin only; in between, the
    filter is simply run forward on the new observations - what a desk would do
    with a model re-estimated once a year.
    """
    mats = np.asarray(yields.columns, dtype=float)
    labels = [maturity_label(m) for m in mats]
    y = yields.to_numpy(dtype=float)
    T = len(y)
    z = float(norm.ppf(0.5 + interval / 2.0))
    err_m: dict[int, list[FloatArray]] = {h: [] for h in horizons}
    err_rw: dict[int, list[FloatArray]] = {h: [] for h in horizons}
    inside: dict[int, list[FloatArray]] = {h: [] for h in horizons}
    est: DNSResult | None = None
    params: DNSParameters | None = None
    for t in range(min_train - 1, T - 1):
        if params is None or (t - (min_train - 1)) % reestimate_every == 0:
            est = fit_dns(
                yields.iloc[: t + 1],
                dynamics=dynamics,
                level_unit_root=level_unit_root,
                start=params,
                maxiter=400 if params is None else 150,
                arbitrage_free=arbitrage_free,
                independent=independent,
            )
            params = est.params
        out = kalman_filter(y[: t + 1], mats, params)
        state = DNSResult(
            params,
            mats,
            out.loglik,
            0,
            pd.DataFrame(out.filtered),
            out.filtered_cov,
            True,
            level_unit_root,
        )
        for h in horizons:
            if t + h >= T:
                continue
            mean, cov = state.forecast(h)
            sd = np.sqrt(np.diag(cov))
            actual = y[t + h]
            err_m[h].append((actual - mean) * 100.0)
            err_rw[h].append((actual - y[t]) * 100.0)
            inside[h].append(np.where(np.isfinite(actual), np.abs(actual - mean) <= z * sd, np.nan))

    def agg(d: dict[int, list[FloatArray]], fn: str) -> pd.DataFrame:
        rows = []
        for h in horizons:
            a = np.array(d[h]) if d[h] else np.full((1, mats.size), np.nan)
            rows.append(
                np.sqrt(np.nanmean(a**2, axis=0)) if fn == "rmse" else np.nanmean(a, axis=0)
            )
        frame = pd.DataFrame(rows, index=list(horizons), columns=labels)
        frame.index.name = "horizon"
        return frame

    return DNSForecastEvaluation(
        rmse_model=agg(err_m, "rmse"),
        rmse_random_walk=agg(err_rw, "rmse"),
        coverage=agg(inside, "mean"),
        interval=interval,
        n_forecasts=pd.Series({h: len(err_m[h]) for h in horizons}, name="n_forecasts"),
        rmse_combination=agg(
            {h: combination_errors(err_m[h], err_rw[h]) for h in horizons}, "rmse"
        ),
    )
