"""Robust calibration of Nelson-Siegel(-Svensson) curves.

Why not just throw all six parameters at a generic optimiser?
--------------------------------------------------------------
The NSS least-squares surface is notoriously badly behaved (Gilli, Große &
Schumann, 2010): it has multiple local minima, long flat valleys in the decay
parameters, and near-perfect collinearity between the two curvature factors
whenever ``λ1 ≈ λ2``. A local optimiser started from a heuristic guess (the
approach of the original version of this project) therefore returns whichever
minimum is closest, and parameters jump around from one day to the next.

Separable least squares (variable projection)
---------------------------------------------
For *fixed* decay rates the model is **linear** in the betas::

    y = X(λ) β + ε          X(λ) = NSS loadings at the observed maturities

so the optimal betas have a closed form (weighted ridge regression)::

    β̂(λ) = (Xᵀ W X + R)⁻¹ Xᵀ W y

and the calibration collapses to a 2-D search over ``(λ1, λ2)`` only
(Golub & Pereyra's *variable projection*). That low-dimensional search is done
globally - an exhaustive, vectorised grid over ``log λ`` - and then polished
with a constrained local optimiser (SLSQP). The result is deterministic,
independent of starting values, and about as fast as a single local solve.

Identification
--------------
* ``λ1 ≥ min_lambda_ratio · λ2`` keeps the two curvature humps apart, removing
  the collinearity (and the label-switching between ``β2`` and ``β3``).
* Bounds on the ``λ`` place the curvature humps inside the observed maturity
  range, so ``β0`` keeps its meaning as the long-run level. The lower bound on
  ``λ2`` follows the data (``hump_within_data``): when the longest quotes are
  missing, the second hump may not peak beyond the longest one observed.
* A small ridge penalty on ``β2, β3`` regularises the remaining
  ill-conditioning without visibly biasing the fit.
* Across dates, an optional penalty on ``Δ log λ`` (``lambda_smoothing``)
  selects, among near-equivalent fits, the one closest to yesterday's decay
  rates - removing spurious jumps in the factor time series.

Par-yield fitting
-----------------
Treasury constant-maturity yields are *par* yields, while the NSS function is a
*zero* curve. With ``target="par"`` the calibrator first solves the zero-curve
problem above to get a good starting point, then refines all six parameters so
that the model's par yields (see :meth:`NSSCurve.par_yield`) match the quotes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, minimize

from .models import (
    DIEBOLD_LI_LAMBDA,
    PARAM_NAMES,
    FloatArray,
    NSSCurve,
    coupon_schedule,
    curvature_loading,
    nss_loadings,
    slope_loading,
)

Model = Literal["ns", "nss"]
Target = Literal["yield", "par"]


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings for :func:`calibrate` and :func:`calibrate_panel`.

    Attributes
    ----------
    model:
        ``"nss"`` (6 parameters, default) or ``"ns"`` (4 parameters).
    target:
        ``"par"`` (default) treats quotes as semi-annual par yields - how the
        Treasury constant-maturity (CMT) series on FRED are quoted - and fits the
        model's par curve to them. ``"yield"`` fits the continuously compounded
        zero curve directly to the quotes (the Diebold & Li 2006 shortcut). On
        par-quoted data the in-sample fit of both is identical, but the ``"yield"``
        zero curve is biased: ~3× further from the true curve on a simulated
        market, −5 bp at 30 years (``benchmarks/compare_legacy.py``). Use ``"yield"``
        for data that really are zero rates (e.g. the Fed's GSW curve).
    lambda1_bounds, lambda2_bounds:
        Box constraints on the decay rates (1/years). The defaults put the first
        hump between ~0.6 and ~12 years and the second between ~1.8 and ~30 years.
    min_lambda_ratio:
        Enforces ``λ1 ≥ ratio · λ2`` (NSS only).
    grid_size:
        Grid points per ``λ`` dimension for the global search.
    n_starts:
        Maximum number of grid-local minima (distinct basins) refined by the
        local optimiser.
    ridge:
        Ridge penalty on the curvature betas, in (%)² per (%)² of beta.
    lambda_smoothing:
        Penalty on ``Σ (log λ_t - log λ_{t-1})²`` used when a previous fit is
        supplied (panel calibration). ``0`` disables it.
    fixed_lambda1, fixed_lambda2:
        Fix the decay rates (e.g. the Diebold-Li value). The problem then becomes
        a single linear regression.
    min_points_nss:
        Minimum number of observed maturities to fit NSS; below it the
        calibrator falls back to Nelson-Siegel.
    min_points:
        Minimum number of observed maturities to attempt any fit.
    robust:
        Huber-type iteratively reweighted fitting. Quotes whose *standardised*
        residual ``r_i / √(1 − h_ii)`` exceeds ``huber_k`` robust standard
        deviations are down-weighted by ``huber_k / |u_i|`` and the curve is
        refitted. Using the leverage ``h_ii`` matters: NSS has 6 parameters
        for ~11 quotes and bends towards a bad quote at the ends of the curve,
        hiding it in the raw residuals.
    huber_k:
        Huber threshold in robust standard deviations.
    robust_scale_floor_bp:
        Lower bound on the robust residual scale, so a near-perfect fit does
        not flag sub-basis-point deviations.
    hump_within_data:
        Raise the lower bound on ``λ2`` so that the second curvature hump peaks
        no later than the longest *observed* maturity
        (``λ2 ≥ 1.793 / τ_max``). With all CMT tenors quoted (``τ_max = 30``)
        this is the default bound and changes nothing. When the long end is
        missing - the 30-year bond was not issued from 2002 to 2006 - it stops
        an unanchored hump from bending the extrapolated long end.
    """

    model: Model = "nss"
    target: Target = "par"
    lambda1_bounds: tuple[float, float] = (0.15, 3.0)
    lambda2_bounds: tuple[float, float] = (0.06, 1.0)
    min_lambda_ratio: float = 1.5
    grid_size: int = 24
    n_starts: int = 4
    ridge: float = 1e-5
    lambda_smoothing: float = 0.0
    fixed_lambda1: float | None = None
    fixed_lambda2: float | None = None
    min_points_nss: int = 7
    min_points: int = 4
    robust: bool = False
    huber_k: float = 3.0
    robust_scale_floor_bp: float = 2.0
    hump_within_data: bool = True

    def __post_init__(self) -> None:
        if self.model not in ("ns", "nss"):
            raise ValueError("model must be 'ns' or 'nss'")
        if self.target not in ("yield", "par"):
            raise ValueError("target must be 'yield' or 'par'")
        for lo, hi in (self.lambda1_bounds, self.lambda2_bounds):
            if not 0 < lo < hi:
                raise ValueError("lambda bounds must satisfy 0 < lower < upper")
        if self.min_lambda_ratio < 1:
            raise ValueError("min_lambda_ratio must be >= 1")
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if self.n_starts < 1:
            raise ValueError("n_starts must be >= 1")
        if self.ridge < 0 or self.lambda_smoothing < 0:
            raise ValueError("penalties must be non-negative")
        if self.huber_k <= 0 or self.robust_scale_floor_bp < 0:
            raise ValueError("huber_k must be positive and the scale floor non-negative")

    @classmethod
    def diebold_li(cls) -> CalibrationConfig:
        """Nelson-Siegel with ``λ`` fixed at the Diebold & Li (2006) value."""
        return cls(model="ns", target="yield", fixed_lambda1=DIEBOLD_LI_LAMBDA, ridge=0.0)


@dataclass(frozen=True)
class FitResult:
    """Outcome of a single-date calibration."""

    curve: NSSCurve
    maturities: FloatArray
    observed: FloatArray
    fitted: FloatArray
    model: str
    target: str
    loss: float
    success: bool
    message: str = ""
    n_evals: int = 0
    runtime_ms: float = 0.0
    #: Normalised weights used in the final fit (including robust down-weighting).
    weights: FloatArray | None = None
    #: Robust (Huber) weight multipliers in (0, 1]; all ones for a non-robust fit.
    robust_weights: FloatArray | None = None
    #: ∂fitted/∂[β0..β3, λ1, λ2] at the solution; columns of fixed parameters are 0.
    jacobian: FloatArray | None = None
    #: Which of the six parameters were estimated.
    active: tuple[bool, ...] = (True,) * 6
    #: Diagonal curvature of the ridge / λ-smoothing penalties in parameter space.
    penalty: FloatArray | None = None

    @property
    def residuals_bp(self) -> FloatArray:
        """Observed minus fitted, in basis points (positive = bond trades *cheap*)."""
        return (self.observed - self.fitted) * 100.0

    @property
    def rmse_bp(self) -> float:
        r = self.residuals_bp
        return float(np.sqrt(np.mean(r**2))) if r.size else float("nan")

    @property
    def max_abs_error_bp(self) -> float:
        r = self.residuals_bp
        return float(np.max(np.abs(r))) if r.size else float("nan")

    @property
    def n_params(self) -> int:
        return int(sum(self.active))

    @property
    def leverage(self) -> FloatArray:
        """Diagonal of the (weighted, penalised, linearised) hat matrix, ``h_ii`` in [0, 1].

        Without penalties ``Σ h_ii`` equals the number of fitted parameters; the
        ridge and λ-smoothing penalties pin parameters partly, lowering it. A
        tenor with ``h_ii`` near 1 essentially determines a parameter: its
        residual is small whatever its quote, so raw residuals cannot reveal a
        bad quote there.
        """
        if self.jacobian is None or self.weights is None:
            return np.full(self.maturities.size, np.nan)
        act = list(self.active)
        pen = None if self.penalty is None else self.penalty[act]
        return _leverage(self.jacobian[:, act], self.weights, pen)

    @property
    def dof(self) -> float:
        """Residual degrees of freedom: effective number of quotes minus ``Σ h_ii``."""
        n_eff = (
            float(np.sum(self.robust_weights))
            if self.robust_weights is not None
            else float(self.maturities.size)
        )
        return n_eff - float(np.nansum(self.leverage))

    def _unit_weights(self) -> FloatArray:
        """Weights rescaled so an ordinary (non-rejected) quote has weight ≈ 1."""
        assert self.weights is not None
        total = (
            float(np.sum(self.robust_weights))
            if self.robust_weights is not None
            else float(self.maturities.size)
        )
        return np.asarray(self.weights * total / self.weights.sum())

    @property
    def outliers(self) -> NDArray[np.bool_]:
        """Boolean mask of quotes the robust fit down-weighted."""
        if self.robust_weights is None:
            return np.zeros(self.maturities.size, dtype=bool)
        return np.asarray(self.robust_weights < 1.0, dtype=bool)

    @property
    def sigma_bp(self) -> float:
        """Estimated quote noise: weighted residual RMS with a degrees-of-freedom correction."""
        if self.weights is None or self.jacobian is None or not self.dof > 0.5:
            return float("nan")
        wn = self._unit_weights()
        return float(np.sqrt(np.sum(wn * self.residuals_bp**2) / self.dof))

    def covariance(self) -> FloatArray:
        """Asymptotic covariance of the six parameters (zeros for fixed ones).

        Penalised weighted least squares: ``σ̂² (Jᵀ W̃ J + S·P)⁻¹`` with ``W̃`` the
        weights scaled so an ordinary quote has weight 1 (``S = Σ W̃``), ``P``
        the penalty curvature and ``σ̂`` from :attr:`sigma_bp`. NSS parameters are poorly
        identified individually - expect large, strongly correlated errors on
        the betas - while the *curve* is pinned down tightly (see
        :meth:`confidence_band`).
        """
        cov = np.full((6, 6), np.nan)
        if self.jacobian is None or self.weights is None or not np.isfinite(self.sigma_bp):
            return cov
        act = np.flatnonzero(self.active)
        J = self.jacobian[:, act]
        wn = self._unit_weights()
        info = J.T @ (J * wn[:, None])
        if self.penalty is not None:
            info = info + wn.sum() * np.diag(self.penalty[act])
        sigma = self.sigma_bp / 100.0
        cov[:] = 0.0
        cov[np.ix_(act, act)] = sigma**2 * np.linalg.pinv(info)
        return cov

    def standard_errors(self) -> dict[str, float]:
        return dict(zip(PARAM_NAMES, np.sqrt(np.diag(self.covariance())), strict=True))

    def confidence_band(
        self, tau: ArrayLike, measure: str = "zero", level: float = 0.95
    ) -> tuple[FloatArray, FloatArray]:
        """Pointwise delta-method band ``(lower, upper)`` for ``zero``/``par``/``forward``.

        Uses a Student-t quantile with :attr:`dof` degrees of freedom: with 11
        quotes and 6 parameters there are only 5, and a normal quantile gives
        ~89% coverage for a nominal 95% band (checked by simulation in the
        tests). Reflects parameter estimation error only, not misspecification.
        """
        from scipy.stats import t as student_t

        dof = self.dof
        z = float(student_t.ppf(0.5 + level / 2.0, dof)) if dof > 0.5 else float("nan")
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        centre = self.curve.evaluate(tau, measure)
        cov = self.covariance()
        if not np.all(np.isfinite(cov)):
            nan = np.full_like(centre, np.nan)
            return nan, nan
        p = self.curve.as_array()
        grad = np.zeros((tau.size, 6))
        for j in np.flatnonzero(self.active):
            h = 1e-6 * max(1.0, abs(p[j]))
            up, dn = p.copy(), p.copy()
            up[j] += h
            dn[j] -= h
            grad[:, j] = (
                NSSCurve.from_array(up).evaluate(tau, measure)
                - NSSCurve.from_array(dn).evaluate(tau, measure)
            ) / (2 * h)
        se = np.sqrt(np.einsum("ni,ij,nj->n", grad, cov, grad))
        return centre - z * se, centre + z * se

    def summary(self) -> dict[str, float | str | bool]:
        out: dict[str, float | str | bool] = dict(self.curve.as_dict())
        out.update(
            rmse_bp=self.rmse_bp,
            max_abs_error_bp=self.max_abs_error_bp,
            n_points=float(self.maturities.size),
            model=self.model,
            target=self.target,
            success=self.success,
        )
        return out


# =============================================================================
# Internal helpers
# =============================================================================


class _ParOperator:
    """Fast, vectorised par-yield evaluation for a fixed set of maturities."""

    def __init__(self, maturities: FloatArray, freq: int = 2) -> None:
        self.maturities = maturities
        self.freq = freq
        self.is_bill = maturities <= 1.0
        times: list[float] = []
        rows: list[int] = []
        #: accrued fraction of the current coupon, subtracted from the annuity
        self.accrued = np.zeros(maturities.size)
        for i, t in enumerate(maturities):
            if self.is_bill[i]:
                continue
            sched, self.accrued[i] = coupon_schedule(float(t), freq)
            times.extend(sched)
            rows.extend([i] * sched.size)
        # Coupon dates of different bonds coincide (all on the half-year grid for
        # CMT tenors), so evaluate the curve once per *distinct* date.
        rounded = np.round(np.array(times, dtype=float), 10)
        self.times, col = np.unique(rounded, return_inverse=True)
        self.annuity_matrix = np.zeros((maturities.size, self.times.size))
        np.add.at(self.annuity_matrix, (np.array(rows, dtype=int), col), 1.0 / freq)

    def __call__(self, params: FloatArray) -> FloatArray:
        return self.value_and_jacobian(params, jacobian=False)[0]

    def value_and_jacobian(
        self, params: FloatArray, jacobian: bool = True
    ) -> tuple[FloatArray, FloatArray]:
        """Par yields and their analytic derivatives w.r.t. ``[β0..β3, λ1, λ2]`` (n × 6).

        Bills: ``P = 100 f (e^{z/(100 f)} − 1)`` so ``∂P/∂z = e^{z/(100 f)}``.
        Coupon bonds: ``P = 100 (1 − D_T) / A`` with ``A = Σ D(t_j)/f`` and
        ``∂D(t)/∂θ = −D(t) · t/100 · ∂z(t)/∂θ``.
        """
        b = params[:4]
        l1, l2 = float(params[4]), float(params[5])
        mats, f, bills = self.maturities, self.freq, self.is_bill
        n = mats.size
        # One pass over maturities and coupon dates together.
        tau = np.concatenate([mats, self.times])
        x1, x2 = l1 * tau, l2 * tau
        e1, e2 = np.exp(-x1), np.exp(-x2)
        s1, s2 = slope_loading(x1), slope_loading(x2)
        X = np.column_stack([np.ones_like(tau), s1, s1 - e1, s2 - e2])
        z = X @ b
        disc = np.exp(-z * tau / 100.0)
        z_mat, d_mat, d_cf = z[:n], disc[:n], disc[n:]
        out = np.empty_like(mats)
        growth = np.exp(z_mat[bills] / (100.0 * f))
        out[bills] = 100.0 * f * (growth - 1.0)
        cb = ~bills
        annuity = self.annuity_matrix @ d_cf - self.accrued / f
        out[cb] = 100.0 * (1.0 - d_mat[cb]) / annuity[cb]
        jac = np.zeros((n, 6))
        if not jacobian:
            return out, jac
        sp1, sp2 = _slope_loading_prime(x1), _slope_loading_prime(x2)
        dz = np.column_stack(
            [X, tau * (b[1] * sp1 + b[2] * (sp1 + e1)), tau * b[3] * (sp2 + e2)]
        )  # ∂z/∂[β0..β3, λ1, λ2]
        dD = -(disc * tau / 100.0)[:, None] * dz
        jac[bills] = growth[:, None] * dz[:n][bills]
        if cb.any():
            dA = self.annuity_matrix @ dD[n:]
            A = annuity[cb][:, None]
            jac[cb] = 100.0 * (-dD[:n][cb] * A - (1.0 - d_mat[cb])[:, None] * dA[cb]) / A**2
        return out, jac


def _batched_loadings(tau: FloatArray, lambdas: FloatArray, model: str) -> FloatArray:
    """Loadings for many λ vectors at once: ``lambdas`` (G, d) -> (G, n, k)."""
    x1 = lambdas[:, :1] * tau[None, :]
    cols = [np.ones_like(x1), slope_loading(x1), curvature_loading(x1)]
    if model == "nss":
        cols.append(curvature_loading(lambdas[:, 1:2] * tau[None, :]))
    return np.stack(cols, axis=-1)


def _slope_loading_prime(x: FloatArray) -> FloatArray:
    """d/dx [(1 - e^{-x}) / x] = ((1 + x) e^{-x} - 1) / x²."""
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    xs = x[small]
    out[small] = -0.5 + xs / 3.0 - xs**2 / 8.0
    xl = x[~small]
    out[~small] = ((1.0 + xl) * np.exp(-xl) - 1.0) / xl**2
    return out


def _loading_lambda_derivatives(tau: FloatArray, lam: FloatArray, model: str) -> FloatArray:
    """∂X/∂λ_j for each decay rate: array of shape (d, n, k)."""
    k = 3 if model == "ns" else 4
    d = 1 if model == "ns" else 2
    out = np.zeros((d, tau.size, k))
    x1 = lam[0] * tau
    s1p = _slope_loading_prime(x1)
    out[0, :, 1] = tau * s1p
    out[0, :, 2] = tau * (s1p + np.exp(-x1))
    if model == "nss":
        x2 = lam[1] * tau
        out[1, :, 3] = tau * (_slope_loading_prime(x2) + np.exp(-x2))
    return out


def _ridge_vector(model: str, ridge: float) -> FloatArray:
    return np.array([0.0, 0.0, ridge] + ([ridge] if model == "nss" else []))


def _solve_betas(
    X: FloatArray, y: FloatArray, w: FloatArray, ridge_vec: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Weighted ridge regression, batched over leading axes of ``X``.

    Returns ``(betas, loss)`` where loss = Σ w r² + Σ ridge β².
    """
    Xw = X * w[..., :, None]
    A = np.swapaxes(X, -1, -2) @ Xw + np.diag(ridge_vec)
    rhs = np.swapaxes(Xw, -1, -2) @ y
    try:
        betas = np.linalg.solve(A, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        betas = (
            np.linalg.lstsq(A, rhs, rcond=None)[0] if A.ndim == 2 else np.full(rhs.shape, np.nan)
        )
    resid = y - (X @ betas[..., None])[..., 0]
    loss = (w * resid**2).sum(axis=-1) + (ridge_vec * betas**2).sum(axis=-1)
    return betas, loss


def _full_params(betas: FloatArray, lambdas: FloatArray, model: str) -> FloatArray:
    if model == "ns":
        return np.array([*betas[:3], 0.0, lambdas[0], lambdas[0]])
    return np.array([*betas[:4], lambdas[0], lambdas[1]])


def _smoothing_penalty(
    log_lam: FloatArray, prev_log_lam: FloatArray | None, weight: float
) -> FloatArray | float:
    if prev_log_lam is None or weight == 0.0:
        return 0.0
    return weight * ((log_lam - prev_log_lam) ** 2).sum(axis=-1)


def _normalise_weights(weights: ArrayLike | None, n: int) -> FloatArray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be a non-negative vector matching the observed maturities")
    return w / w.sum()


# =============================================================================
# Public API
# =============================================================================


def calibrate(
    maturities: ArrayLike,
    yields: ArrayLike,
    config: CalibrationConfig | None = None,
    weights: ArrayLike | None = None,
    previous: NSSCurve | None = None,
) -> FitResult:
    """Calibrate a Nelson-Siegel(-Svensson) curve to one day of yields.

    Parameters
    ----------
    maturities:
        Maturities in years.
    yields:
        Observed yields in percent. ``NaN`` values (missing tenors) are ignored.
    config:
        Calibration settings; defaults to :class:`CalibrationConfig()`.
    weights:
        Optional non-negative weights per maturity (e.g. ``1/duration``); NaN
        tenors are dropped before normalisation.
    previous:
        Yesterday's curve. Used as an extra starting point and, when
        ``config.lambda_smoothing > 0``, as the anchor of the smoothing penalty.
    """
    cfg = config or CalibrationConfig()
    t0 = time.perf_counter()
    tau_all = np.asarray(maturities, dtype=float).ravel()
    y_all = np.asarray(yields, dtype=float).ravel()
    if tau_all.shape != y_all.shape:
        raise ValueError("maturities and yields must have the same length")
    mask = np.isfinite(y_all) & np.isfinite(tau_all)
    tau, y = tau_all[mask], y_all[mask]
    w_all = None if weights is None else np.asarray(weights, dtype=float).ravel()
    if w_all is not None and w_all.shape != tau_all.shape:
        raise ValueError("weights must match maturities")
    w = _normalise_weights(None if w_all is None else w_all[mask], max(tau.size, 1))

    if tau.size < cfg.min_points:
        return FitResult(
            curve=_nan_curve(),
            maturities=tau,
            observed=y,
            fitted=np.full_like(y, np.nan),
            model=cfg.model,
            target=cfg.target,
            loss=float("nan"),
            success=False,
            message=f"only {tau.size} observed maturities (need {cfg.min_points})",
            runtime_ms=(time.perf_counter() - t0) * 1e3,
        )

    model: str = cfg.model
    if model == "nss" and tau.size < cfg.min_points_nss:
        model = "ns"
    if model == "nss" and cfg.hump_within_data:
        cfg = _hump_bounds(cfg, float(tau.max()))

    target = cfg.target
    active = _active_params(cfg, model)

    def fit_once(
        w_fit: FloatArray, local: FloatArray | None = None
    ) -> tuple[FloatArray, float, int, bool, str, FloatArray, FloatArray]:
        if target == "par":
            out = (
                _calibrate_par(tau, y, w_fit, cfg, model, local, previous)
                if local is not None
                else _calibrate_par_multistart(tau, y, w_fit, cfg, model, previous)
            )
        else:
            out = _calibrate_zero(tau, y, w_fit, cfg, model, previous, local_start=local)
        fitted_vals, jac = _fit_jacobian(tau, out[0], target)
        return (*out, fitted_vals, jac)

    params, loss, n_evals, ok, msg, fitted, jac = fit_once(w)
    huber = np.ones(tau.size)
    w_used = w
    act = list(active)

    def penalty(p: FloatArray) -> FloatArray:
        return _penalty_diag(p, cfg, active, previous)

    if cfg.robust and tau.size > sum(active):
        # Huber first (convex, so it converges from the ordinary fit), then
        # Tukey's bisquare, which gives gross outliers zero weight.
        # Reweighting steps are local polishes; a global fit runs on the first
        # reweighting and again with the final weights.
        reweighted = False
        for kind in ("huber", "bisquare"):
            for _ in range(_ROBUST_MAX_ITER):
                lev = _leverage(jac[:, act], w_used, penalty(params)[act])
                new = _robust_weights(y - fitted, lev, cfg, kind)
                if (new > 0.5).sum() <= sum(active):  # too few clean quotes left
                    break
                if np.max(np.abs(new - huber)) < 1e-3:
                    break
                huber = new
                w_used = _normalise_weights(w * np.maximum(huber, 1e-9), tau.size)
                params, loss, extra, ok, msg, fitted, jac = fit_once(
                    w_used, params if reweighted else None
                )
                n_evals += extra
                reweighted = True
        if reweighted:
            final = fit_once(w_used)
            n_evals += final[2]
            if np.isfinite(final[1]) and final[1] <= loss:
                params, loss, _, ok, msg, fitted, jac = final

    if model == "ns" and previous is not None and np.isfinite(previous.lambda2):
        # β3 = 0, so λ2 is irrelevant for the curve; carry it forward to keep the
        # λ2 time series continuous across NS fallbacks.
        params[5] = previous.lambda2
    curve = NSSCurve.from_array(params)
    return FitResult(
        curve=curve,
        maturities=tau,
        observed=y,
        fitted=fitted,
        model=model,
        target=target,
        loss=float(loss),
        success=bool(ok),
        message=msg,
        n_evals=n_evals,
        runtime_ms=(time.perf_counter() - t0) * 1e3,
        weights=w_used,
        robust_weights=huber,
        jacobian=jac * np.asarray(active, dtype=float)[None, :],
        active=active,
        penalty=penalty(params),
    )


_ROBUST_MAX_ITER = 8

# The curvature loading (1 - e^-x)/x - e^-x peaks at x = λτ ≈ 1.7933.
HUMP_PEAK = 1.7932821325977144


def _hump_bounds(cfg: CalibrationConfig, tau_max: float) -> CalibrationConfig:
    """Tighten the ``λ2`` lower bound so the second hump peaks within the data."""
    lo, hi = cfg.lambda2_bounds
    # Stay feasible: below the upper bound and the ratio constraint's ceiling.
    ceiling = min(hi, cfg.lambda1_bounds[1] / cfg.min_lambda_ratio)
    new_lo = min(max(lo, HUMP_PEAK / tau_max), 0.5 * ceiling)
    if new_lo <= lo:
        return cfg
    return replace(cfg, lambda2_bounds=(new_lo, hi))


def _active_params(cfg: CalibrationConfig, model: str) -> tuple[bool, ...]:
    """Which of ``[β0, β1, β2, β3, λ1, λ2]`` are estimated."""
    nss = model == "nss"
    return (
        True,
        True,
        True,
        nss,
        cfg.fixed_lambda1 is None,
        nss and cfg.fixed_lambda2 is None,
    )


def _fit_jacobian(
    tau: FloatArray, params: FloatArray, target: str
) -> tuple[FloatArray, FloatArray]:
    """Fitted values and ``∂fitted/∂[β0..β3, λ1, λ2]`` (n × 6) for either target."""
    if not np.all(np.isfinite(params)):
        return np.full(tau.size, np.nan), np.full((tau.size, 6), np.nan)
    if target == "par":
        return _ParOperator(tau).value_and_jacobian(params)
    b, lam = params[:4], params[4:6]
    X = nss_loadings(tau, lam[0], lam[1])
    dX = _loading_lambda_derivatives(tau, lam, "nss")
    return X @ b, np.column_stack([X, dX[0] @ b, dX[1] @ b])


def _penalty_diag(
    params: FloatArray, cfg: CalibrationConfig, active: tuple[bool, ...], previous: NSSCurve | None
) -> FloatArray:
    """Curvature of the penalties w.r.t. ``[β0..β3, λ1, λ2]`` (diagonal).

    Ridge ``ρ β²`` contributes ``ρ``; smoothing ``s (log λ − log λ_prev)²``
    contributes ``s / λ²`` (Gauss-Newton), only when a previous curve anchors it.
    """
    pen = np.zeros(6)
    pen[2] = cfg.ridge
    pen[3] = cfg.ridge if active[3] else 0.0
    anchored = previous is not None and np.all(np.isfinite(previous.as_array()))
    if cfg.lambda_smoothing > 0 and anchored and np.all(np.isfinite(params[4:6])):
        pen[4:6] = cfg.lambda_smoothing / params[4:6] ** 2
    return pen * np.asarray(active, dtype=float)


def _leverage(J: FloatArray, w: FloatArray, penalty: FloatArray | None = None) -> FloatArray:
    """``diag(W^½ J (Jᵀ W J + P)⁻¹ Jᵀ W^½)`` via a thin QR of the augmented system."""
    if not np.all(np.isfinite(J)):
        return np.full(J.shape[0], np.nan)
    A = np.sqrt(w)[:, None] * J
    if penalty is not None and np.any(penalty > 0):
        A = np.vstack([A, np.diag(np.sqrt(penalty))])
    q, _ = np.linalg.qr(A)
    return np.clip(np.sum(q[: J.shape[0]] ** 2, axis=1), 0.0, 1.0)


def _robust_weights(
    resid: FloatArray, leverage: FloatArray, cfg: CalibrationConfig, kind: str
) -> FloatArray:
    """Huber or bisquare weights on leverage-standardised residuals (MAD scale).

    ``u_i = r_i / √(1 − h_ii)`` has the same variance for every quote, so the
    ends of the curve (high leverage, small raw residuals) are judged fairly.
    The scale is the MAD over quotes that are not already rejected.
    """
    u = resid / np.sqrt(np.clip(1.0 - leverage, 1e-3, 1.0))
    scale = max(1.4826 * float(np.median(np.abs(u))), cfg.robust_scale_floor_bp / 100.0)
    a = np.abs(u) / scale
    if kind == "huber":
        return np.where(a > cfg.huber_k, cfg.huber_k / np.maximum(a, 1e-12), 1.0)
    c = _BISQUARE_C * cfg.huber_k / 3.0  # 4.685 at the default huber_k = 3
    wts = np.where(a < c, (1.0 - (a / c) ** 2) ** 2, 0.0)
    # Keep quotes inside the Huber region at full weight: only the bisquare's
    # tail (rejection of gross errors) is wanted, not extra down-weighting of
    # ordinary noise.
    return np.where(a <= cfg.huber_k, 1.0, wts)


_BISQUARE_C = 4.685


def _nan_curve() -> NSSCurve:
    # NSSCurve validates positive lambdas; bypass for the "no fit" sentinel.
    curve = object.__new__(NSSCurve)
    for name in PARAM_NAMES:
        object.__setattr__(curve, name, float("nan"))
    return curve


def _lambda_space(cfg: CalibrationConfig, model: str) -> tuple[list[int], FloatArray]:
    """Indices of free λ's and the fixed λ vector (NaN where free)."""
    fixed = np.array(
        [
            np.nan if cfg.fixed_lambda1 is None else cfg.fixed_lambda1,
            np.nan if cfg.fixed_lambda2 is None else cfg.fixed_lambda2,
        ]
    )
    dims = 1 if model == "ns" else 2
    free = [i for i in range(dims) if np.isnan(fixed[i])]
    return free, fixed[:dims]


def _calibrate_zero(
    tau: FloatArray,
    y: FloatArray,
    w: FloatArray,
    cfg: CalibrationConfig,
    model: str,
    previous: NSSCurve | None,
    candidates: list[FloatArray] | None = None,
    local_start: FloatArray | None = None,
) -> tuple[FloatArray, float, int, bool, str]:
    """Variable-projection calibration of the zero curve to ``y``.

    If ``candidates`` is a list, the parameters of every locally refined basin
    are appended to it (used as starting points by the par-yield stage).
    ``local_start`` (full parameter vector) skips the global grid and only
    polishes from its decay rates.
    """
    ridge_vec = _ridge_vector(model, cfg.ridge)
    free, fixed = _lambda_space(cfg, model)
    bounds = np.log(np.array([cfg.lambda1_bounds, cfg.lambda2_bounds]))[: fixed.size]
    log_ratio = np.log(cfg.min_lambda_ratio)

    prev_log = None
    if previous is not None and np.all(np.isfinite(previous.as_array())):
        prev_full = np.log(np.array([previous.lambda1, previous.lambda2]))[: fixed.size]
        prev_log = np.clip(prev_full, bounds[:, 0], bounds[:, 1])
    smoothing = cfg.lambda_smoothing if prev_log is not None else 0.0

    def expand(theta: FloatArray) -> FloatArray:
        """Map free log-λ's to the full log-λ vector (batched)."""
        theta = np.atleast_2d(theta)
        full = np.tile(np.log(np.where(np.isnan(fixed), 1.0, fixed)), (theta.shape[0], 1))
        full[:, free] = theta
        return full

    def objective_batch(theta: FloatArray) -> tuple[FloatArray, FloatArray]:
        log_lam = expand(theta)
        X = _batched_loadings(tau, np.exp(log_lam), model)
        betas, loss = _solve_betas(X, y, w, ridge_vec)
        loss = loss + _smoothing_penalty(log_lam, prev_log, smoothing)
        return betas, loss

    n_evals = 0
    if not free:  # all decay rates fixed: one linear regression
        betas, loss = objective_batch(np.empty((1, 0)))
        lam = np.exp(expand(np.empty((1, 0)))[0])
        return _full_params(betas[0], lam, model), float(loss[0]), 1, True, "fixed lambda"

    # ---- 1. global grid search over log-λ ------------------------------------
    if local_start is not None:
        local_log = np.log(local_start[4:6])[: fixed.size]
        starts = [np.clip(local_log, bounds[:, 0], bounds[:, 1])[free]]
    else:
        axes = [np.linspace(bounds[i, 0], bounds[i, 1], cfg.grid_size) for i in free]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([g.ravel() for g in mesh], axis=-1)
        feasible_mask = np.ones(grid.shape[0], dtype=bool)
        if model == "nss" and len(free) == 2:
            feasible_mask = grid[:, 0] - grid[:, 1] >= log_ratio - 1e-12
        grid_loss = np.full(grid.shape[0], np.inf)
        _, grid_loss[feasible_mask] = objective_batch(grid[feasible_mask])
        n_evals += int(feasible_mask.sum())
        starts = _grid_local_minima(grid, grid_loss.reshape(mesh[0].shape), cfg.n_starts)
    if prev_log is not None and local_start is None:
        prev_theta = prev_log[free]
        feasible = model == "ns" or len(free) < 2 or prev_theta[0] - prev_theta[1] >= log_ratio
        if feasible:
            starts.append(prev_theta)

    # ---- 2. local polish (SLSQP handles the ratio constraint exactly) --------
    # By the envelope theorem the gradient of the *reduced* objective needs no
    # derivative of β̂(λ):  ∂L/∂λ_j = −2 Σ w r (∂X/∂λ_j · β̂)  (+ smoothing term).
    base_log = expand(np.zeros((1, len(free))))[0]

    def value_and_grad(theta: FloatArray) -> tuple[float, FloatArray]:
        log_lam = base_log.copy()
        log_lam[free] = theta
        lam = np.exp(log_lam)
        X = _batched_loadings(tau, lam[None, :], model)[0]
        betas, loss = _solve_betas(X, y, w, ridge_vec)
        wr = w * (y - X @ betas)
        dX = _loading_lambda_derivatives(tau, lam, model)  # (d, n, k)
        grad_full = -2.0 * np.einsum("n,dnk,k->d", wr, dX, betas) * lam
        val = float(loss) + float(_smoothing_penalty(log_lam, prev_log, smoothing))
        if smoothing and prev_log is not None:
            grad_full = grad_full + 2.0 * smoothing * (log_lam - prev_log)
        return val, grad_full[free]

    constraints: list[dict[str, object]] = []
    if model == "nss" and len(free) == 2:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda th: th[0] - th[1] - log_ratio,
                "jac": lambda th: np.array([1.0, -1.0]),
            }
        )
    best_theta, best_loss, ok, msg = starts[0], value_and_grad(starts[0])[0], True, "grid"
    for start in starts:
        res = minimize(
            value_and_grad,
            start,
            jac=True,
            method="SLSQP",
            bounds=[tuple(bounds[i]) for i in free],
            constraints=constraints,
            options={"ftol": 1e-15, "maxiter": 200},
        )
        n_evals += int(res.nfev)
        if candidates is not None and _feasible(res.x, bounds[free], constraints):
            b_c, _ = objective_batch(res.x)
            candidates.append(_full_params(b_c[0], np.exp(expand(res.x)[0]), model))
        if (
            np.isfinite(res.fun)
            and res.fun < best_loss
            and _feasible(res.x, bounds[free], constraints)
        ):
            best_theta, best_loss, ok, msg = (
                res.x,
                float(res.fun),
                bool(res.success),
                str(res.message),
            )

    betas, loss = objective_batch(best_theta)
    lam = np.exp(expand(best_theta)[0])
    return _full_params(betas[0], lam, model), float(loss[0]), n_evals, ok, msg


def _grid_local_minima(grid: FloatArray, loss: FloatArray, k: int) -> list[FloatArray]:
    """The ``k`` lowest *local* minima of a loss evaluated on a regular grid.

    A point is a local minimum if no neighbour (including diagonals) is lower.
    Refining every distinct basin - not just the globally lowest grid point -
    matters because the NSS surface can hide the true optimum in a basin much
    narrower than the grid spacing, next to broad basins of similar depth.
    """
    padded = np.pad(loss, 1, constant_values=np.inf)
    is_min = np.isfinite(loss)
    for offset in np.ndindex(*(3,) * loss.ndim):
        if all(o == 1 for o in offset):
            continue
        window = tuple(slice(o, o + n) for o, n in zip(offset, loss.shape, strict=True))
        is_min &= loss <= padded[window]
    flat = loss.ravel()
    idx = np.flatnonzero(is_min.ravel())
    if idx.size == 0:  # pragma: no cover - a finite grid always has a minimum
        idx = np.array([int(np.nanargmin(flat))])
    idx = idx[np.argsort(flat[idx])][:k]
    return [grid[i] for i in idx]


def _feasible(
    theta: FloatArray, bounds: FloatArray, constraints: Sequence[dict[str, object]]
) -> bool:
    tol = 1e-8
    if np.any(theta < bounds[:, 0] - tol) or np.any(theta > bounds[:, 1] + tol):
        return False
    for c in constraints:
        fun = c["fun"]
        assert callable(fun)
        if fun(theta) < -tol:
            return False
    return True


def _calibrate_par_multistart(
    tau: FloatArray,
    y: FloatArray,
    w: FloatArray,
    cfg: CalibrationConfig,
    model: str,
    previous: NSSCurve | None,
) -> tuple[FloatArray, float, int, bool, str]:
    """Global par-yield calibration.

    The par objective is not separable in β, so the global λ-grid runs on the
    zero-curve problem instead - but on *convexity-adjusted* quotes
    ``y − (par(ĉ) − zero(ĉ))``, where ``ĉ`` is a first zero-curve fit. That puts
    the grid search on (almost) the right problem. Every basin it finds, the
    unadjusted zero fit and last period's curve are then refined in par space,
    and the lowest par loss wins. Polishing only a single start lands in a
    worse basin on ~7% of simulated dates (median cost 2.7 bp RMSE).
    """
    # The convexity gap only needs to be roughly right: take it from last
    # period's curve, or from a coarse (unpolished) zero-curve grid fit.
    if previous is not None and np.all(np.isfinite(previous.as_array())):
        guess = previous
    else:
        guess = _coarse_zero_fit(tau, y, w, cfg, model)
    gap = guess.par_yield(tau) - guess.zero(tau)
    starts: list[FloatArray] = []
    first, _, n_evals, _, _ = _calibrate_zero(tau, y - gap, w, cfg, model, previous, starts)
    starts.insert(0, first)
    if not np.all(np.isfinite(first)):
        return first, float("nan"), n_evals, False, "zero-curve stage failed"
    if previous is not None and np.all(np.isfinite(previous.as_array())):
        prev = previous.as_array()
        if model == "ns":
            prev = np.array([*prev[:3], 0.0, prev[4], prev[4]])
        starts.append(prev)

    # Screen every start with a short polish, then refine only the winner.
    best: tuple[FloatArray, float, int, bool, str] | None = None
    candidates = _distinct(starts)
    for start in candidates:
        cand = _calibrate_par(
            tau, y, w, cfg, model, start, previous, max_nfev=8 if len(candidates) > 1 else None
        )
        n_evals += cand[2]
        if best is None or (np.isfinite(cand[1]) and cand[1] < best[1]):
            best = cand
    assert best is not None
    if len(candidates) > 1:
        final = _calibrate_par(tau, y, w, cfg, model, best[0], previous)
        n_evals += final[2]
        if np.isfinite(final[1]) and final[1] <= best[1]:
            best = final
        else:
            # A converged polish from the screened winner found nothing better,
            # which confirms it; report the polish's convergence status.
            best = (best[0], best[1], best[2], final[3], final[4])
    return best[0], best[1], n_evals, best[3], best[4]


def _coarse_zero_fit(
    tau: FloatArray, y: FloatArray, w: FloatArray, cfg: CalibrationConfig, model: str
) -> NSSCurve:
    """Best zero curve on a coarse λ grid (no local refinement)."""
    free, fixed = _lambda_space(cfg, model)
    axes = []
    for i, bounds in enumerate((cfg.lambda1_bounds, cfg.lambda2_bounds)[: fixed.size]):
        axes.append(np.geomspace(*bounds, 8) if i in free else np.array([fixed[i]]))
    lam = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=-1)
    if model == "nss":
        lam = lam[lam[:, 0] >= cfg.min_lambda_ratio * lam[:, 1] * (1 - 1e-12)]
    betas, loss = _solve_betas(
        _batched_loadings(tau, lam, model), y, w, _ridge_vector(model, cfg.ridge)
    )
    i = int(np.argmin(loss))
    return NSSCurve.from_array(_full_params(betas[i], lam[i], model))


def _distinct(starts: list[FloatArray], tol: float = 1e-6) -> list[FloatArray]:
    out: list[FloatArray] = []
    for s in starts:
        if not any(np.allclose(s, o, rtol=tol, atol=tol) for o in out):
            out.append(s)
    return out


def _calibrate_par(
    tau: FloatArray,
    y: FloatArray,
    w: FloatArray,
    cfg: CalibrationConfig,
    model: str,
    start: FloatArray,
    previous: NSSCurve | None,
    max_nfev: int | None = None,
) -> tuple[FloatArray, float, int, bool, str]:
    """Refine a zero-curve fit so that model *par* yields match the quotes.

    Solved as a bounded nonlinear least-squares problem (trust-region
    reflective). For NSS the decay rates are reparametrised as
    ``(log λ2, log λ1 - log λ2)`` so that the ratio constraint
    ``λ1 ≥ r·λ2`` becomes a simple box bound.
    """
    par = _ParOperator(tau)
    free, fixed = _lambda_space(cfg, model)
    k = 3 if model == "ns" else 4
    ridge_vec = _ridge_vector(model, cfg.ridge)
    bounds_log = np.log(np.array([cfg.lambda1_bounds, cfg.lambda2_bounds]))[: fixed.size]
    log_ratio = np.log(cfg.min_lambda_ratio)
    prev_log = None
    if previous is not None and np.all(np.isfinite(previous.as_array())):
        prev_log = np.log(np.array([previous.lambda1, previous.lambda2]))[: fixed.size]
    smoothing = cfg.lambda_smoothing if prev_log is not None else 0.0
    log_lam0 = np.log(start[4:6])[: fixed.size]
    ratio_param = model == "nss" and len(free) == 2
    sqrt_w, sqrt_ridge = np.sqrt(w), np.sqrt(ridge_vec)

    def unpack(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        log_lam = log_lam0.copy()
        if ratio_param:
            log_lam[1] = x[k]
            log_lam[0] = x[k] + x[k + 1]
        else:
            log_lam[free] = x[k:]
        return x[:k], log_lam

    def residuals(x: FloatArray) -> FloatArray:
        betas, log_lam = unpack(x)
        p = _full_params(betas, np.exp(log_lam), model)
        parts = [sqrt_w * (y - par(p)), sqrt_ridge * betas]
        if smoothing > 0 and prev_log is not None:
            parts.append(np.sqrt(smoothing) * (log_lam - prev_log))
        return np.concatenate(parts)

    n_smooth = fixed.size if smoothing > 0 and prev_log is not None else 0

    def jacobian(x: FloatArray) -> FloatArray:
        betas, log_lam = unpack(x)
        lam = np.exp(log_lam)
        _, dP = par.value_and_jacobian(_full_params(betas, lam, model))
        # Chain rule to the optimisation variables x = [β (k), transformed log-λ].
        dlog = np.zeros((fixed.size, x.size - k))  # ∂ log λ_i / ∂ x_λ
        if ratio_param:
            dlog[1, 0] = 1.0
            dlog[0, :] = 1.0
        else:
            for col, i in enumerate(free):
                dlog[i, col] = 1.0
        dP_dlog = dP[:, 4 : 4 + fixed.size] * lam[None, :]
        if model == "ns":  # λ2 mirrors λ1 but β3 = 0, so it has no effect
            dP_dlog = dP[:, 4:5] * lam[None, :1]
        J_fit = np.hstack([dP[:, :k], dP_dlog @ dlog])
        rows = [
            -sqrt_w[:, None] * J_fit,
            np.hstack([np.diag(sqrt_ridge), np.zeros((k, x.size - k))]),
        ]
        if n_smooth:
            rows.append(np.hstack([np.zeros((n_smooth, k)), np.sqrt(smoothing) * dlog]))
        return np.vstack(rows)

    if ratio_param:
        z0 = np.array([log_lam0[1], max(log_lam0[0] - log_lam0[1], log_ratio)])
        lo = [bounds_log[1, 0], log_ratio]
        hi = [bounds_log[1, 1], bounds_log[0, 1] - bounds_log[1, 0]]
    else:
        z0 = log_lam0[free]
        lo = [bounds_log[i, 0] for i in free]
        hi = [bounds_log[i, 1] for i in free]
    x0 = np.concatenate([start[:k], np.clip(z0, lo, hi)])
    res = least_squares(
        residuals,
        x0,
        jac=jacobian,
        bounds=([-np.inf] * k + lo, [np.inf] * k + hi),
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    loss0 = float(np.sum(residuals(x0) ** 2))
    x_best = res.x if np.isfinite(res.cost) and 2 * res.cost <= loss0 else x0
    betas, log_lam = unpack(x_best)
    lam = np.exp(log_lam)
    lo1, hi1 = cfg.lambda1_bounds
    if ratio_param and not (lo1 - 1e-9 <= lam[0] <= hi1 + 1e-9):
        # The reparametrised box bounds λ2 and λ1/λ2 but not λ1 itself. The
        # optimum wants λ1 beyond its bound, so that bound is active: fix λ1
        # there and re-solve for β and λ2 (with λ2 ≤ λ1 / ratio as a box bound).
        lam1 = float(np.clip(lam[0], lo1, hi1))
        hi2 = min(cfg.lambda2_bounds[1], lam1 / cfg.min_lambda_ratio)
        lo2 = min(cfg.lambda2_bounds[0], hi2 * (1 - 1e-9))
        sub = replace(cfg, fixed_lambda1=lam1, lambda2_bounds=(lo2, hi2))
        start_sub = _full_params(betas, np.array([lam1, min(max(lam[1], lo2), hi2)]), model)
        params_b, loss_b, nfev_b, ok_b, msg_b = _calibrate_par(
            tau, y, w, sub, model, start_sub, previous, max_nfev
        )
        return params_b, loss_b, int(res.nfev) + nfev_b, ok_b, f"lambda1 at its bound; {msg_b}"
    return (
        _full_params(betas, lam, model),
        float(np.sum(residuals(x_best) ** 2)),
        int(res.nfev),
        bool(res.success),
        str(res.message),
    )


# =============================================================================
# Panel calibration
# =============================================================================


@dataclass
class PanelFit:
    """Calibrated curves for a panel of dates.

    Attributes
    ----------
    params:
        One row per date with columns ``beta0 … lambda2``.
    diagnostics:
        ``rmse_bp``, ``max_abs_error_bp``, ``n_points``, ``model``, ``success``,
        ``runtime_ms``, ``sigma_bp`` (noise estimate), ``n_outliers`` and
        ``rmse_clean_bp`` (RMSE over quotes the robust fit kept) per date.
    fitted, residuals_bp:
        Model yields (percent) and observed-minus-fitted residuals (bp),
        aligned with the input panel (NaN where a tenor was not observed).
    outliers:
        Boolean panel of quotes down-weighted by a robust fit (all False
        otherwise).
    """

    params: pd.DataFrame
    diagnostics: pd.DataFrame
    fitted: pd.DataFrame
    residuals_bp: pd.DataFrame
    config: CalibrationConfig = field(default_factory=CalibrationConfig)
    outliers: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.params.index)

    def curve(self, date: object) -> NSSCurve:
        """Curve on ``date`` (or the last available date on or before it)."""
        row = (
            self.params.loc[slice(None, date)].iloc[-1]
            if not isinstance(date, int)
            else self.params.iloc[date]
        )
        return NSSCurve.from_mapping(row)

    def curves(self) -> list[NSSCurve]:
        return [NSSCurve.from_mapping(r) for _, r in self.params.iterrows()]

    def evaluate(self, tau: ArrayLike, measure: str = "zero") -> pd.DataFrame:
        """Evaluate every date's curve on a maturity grid -> DataFrame (dates × tau)."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        if measure == "zero":
            vals = np.array(
                [
                    nss_loadings(tau, r.lambda1, r.lambda2)
                    @ np.array([r.beta0, r.beta1, r.beta2, r.beta3])
                    for r in self.params.itertuples()
                ]
            )
        else:
            vals = np.array([c.evaluate(tau, measure) for c in self.curves()])
        return pd.DataFrame(vals, index=self.params.index, columns=tau)

    def spread(
        self, long: float = 10.0, short: float = 2.0, measure: str | None = None
    ) -> pd.Series:
        """Model-implied ``y(long) - y(short)`` (percentage points) through time."""
        measure = measure or ("par" if self.config.target == "par" else "zero")
        vals = self.evaluate([short, long], measure)
        return (vals[long] - vals[short]).rename(f"model_{long:g}y_{short:g}y")

    def to_frame(self) -> pd.DataFrame:
        return self.params.join(self.diagnostics)


def calibrate_panel(
    yields: pd.DataFrame,
    config: CalibrationConfig | None = None,
    weights: ArrayLike | None = None,
    warm_start: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> PanelFit:
    """Calibrate every row of a yield panel (index = dates, columns = maturities in years).

    Dates are processed in order; each fit receives the previous successful
    curve as a warm start (and smoothing anchor if ``config.lambda_smoothing``).
    """
    cfg = config or CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING)
    maturities = np.asarray(yields.columns, dtype=float)
    values = yields.to_numpy(dtype=float)
    param_rows, diag_rows = [], []
    fitted = np.full_like(values, np.nan)
    flagged = np.zeros(values.shape, dtype=bool)
    prev: NSSCurve | None = None
    n = len(values)
    for i in range(n):
        res = calibrate(
            maturities, values[i], cfg, weights=weights, previous=prev if warm_start else None
        )
        param_rows.append(res.curve.as_array())
        diag_rows.append(
            {
                "rmse_bp": res.rmse_bp,
                "max_abs_error_bp": res.max_abs_error_bp,
                "n_points": res.maturities.size,
                "model": res.model,
                "success": res.success,
                "message": res.message,
                "runtime_ms": res.runtime_ms,
                "sigma_bp": res.sigma_bp,
                "n_outliers": int(res.outliers.sum()),
                "rmse_clean_bp": float(np.sqrt(np.mean(res.residuals_bp[~res.outliers] ** 2)))
                if (~res.outliers).any()
                else float("nan"),
            }
        )
        if np.all(np.isfinite(res.curve.as_array())):
            mask = np.isfinite(values[i])
            fitted[i, mask] = res.fitted
            flagged[i, np.flatnonzero(mask)[res.outliers]] = True
            prev = res.curve
        if progress is not None:
            progress(i + 1, n)

    idx = yields.index
    params = pd.DataFrame(param_rows, index=idx, columns=list(PARAM_NAMES))
    diagnostics = pd.DataFrame(diag_rows, index=idx)
    fitted_df = pd.DataFrame(fitted, index=idx, columns=yields.columns)
    resid = (yields - fitted_df) * 100.0
    ok = params.notna().all(axis=1)
    return PanelFit(
        params=params[ok],
        diagnostics=diagnostics[ok],
        fitted=fitted_df[ok],
        residuals_bp=resid[ok],
        config=cfg,
        outliers=pd.DataFrame(flagged, index=idx, columns=yields.columns)[ok],
    )


#: Default Δlog-λ smoothing penalty used by :func:`calibrate_panel`. Chosen on
#: synthetic data with known true curves (``benchmarks/tune_regularisation.py``):
#: it minimises the error of the fitted curve against the *true* curve and cuts
#: week-to-week λ jumps roughly five-fold, at ~0.2 bp of in-sample RMSE.
DEFAULT_PANEL_SMOOTHING = 1e-3
