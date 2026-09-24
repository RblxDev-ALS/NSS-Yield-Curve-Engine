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
  range, so ``β0`` keeps its meaning as the long-run level.
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
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import least_squares, minimize

from .models import (
    DIEBOLD_LI_LAMBDA,
    PARAM_NAMES,
    FloatArray,
    NSSCurve,
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
        ``"yield"`` fits the zero curve directly to the quoted yields (standard
        practice, e.g. Diebold & Li 2006); ``"par"`` treats quotes as par yields.
    lambda1_bounds, lambda2_bounds:
        Box constraints on the decay rates (1/years). The defaults put the first
        hump between ~0.6 and ~12 years and the second between ~1.8 and ~30 years.
    min_lambda_ratio:
        Enforces ``λ1 ≥ ratio · λ2`` (NSS only).
    grid_size:
        Grid points per ``λ`` dimension for the global search.
    n_starts:
        Number of distinct grid basins refined by the local optimiser.
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
    """

    model: Model = "nss"
    target: Target = "yield"
    lambda1_bounds: tuple[float, float] = (0.15, 3.0)
    lambda2_bounds: tuple[float, float] = (0.06, 1.0)
    min_lambda_ratio: float = 1.5
    grid_size: int = 24
    n_starts: int = 3
    ridge: float = 1e-5
    lambda_smoothing: float = 0.0
    fixed_lambda1: float | None = None
    fixed_lambda2: float | None = None
    min_points_nss: int = 7
    min_points: int = 4

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

    @classmethod
    def diebold_li(cls) -> CalibrationConfig:
        """Nelson-Siegel with ``λ`` fixed at the Diebold & Li (2006) value."""
        return cls(model="ns", fixed_lambda1=DIEBOLD_LI_LAMBDA, ridge=0.0)


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
        for i, t in enumerate(maturities):
            if self.is_bill[i]:
                continue
            n = int(np.floor(t * freq + 1e-9))
            sched = t - np.arange(n) / freq
            times.extend(sched)
            rows.extend([i] * n)
        self.times = np.array(times, dtype=float)
        self.annuity_matrix = np.zeros((maturities.size, self.times.size))
        self.annuity_matrix[np.array(rows, dtype=int), np.arange(self.times.size)] = 1.0 / freq

    def __call__(self, params: FloatArray) -> FloatArray:
        b = params[:4]
        l1, l2 = params[4], params[5]
        mats = self.maturities
        d_mat = np.exp(-(nss_loadings(mats, l1, l2) @ b) * mats / 100.0)
        out = np.empty_like(mats)
        f = self.freq
        bills = self.is_bill
        out[bills] = 100.0 * f * (d_mat[bills] ** (-1.0 / (f * mats[bills])) - 1.0)
        if (~bills).any():
            d_cf = np.exp(-(nss_loadings(self.times, l1, l2) @ b) * self.times / 100.0)
            annuity = self.annuity_matrix @ d_cf
            out[~bills] = 100.0 * (1.0 - d_mat[~bills]) / annuity[~bills]
        return out


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

    params, loss, n_evals, ok, msg = _calibrate_zero(tau, y, w, cfg, model, previous)
    target = cfg.target
    if target == "par":
        params, loss, extra_evals, ok_par, msg = _calibrate_par(
            tau, y, w, cfg, model, params, previous
        )
        n_evals += extra_evals
        ok = ok and ok_par

    if model == "ns" and previous is not None and np.isfinite(previous.lambda2):
        # β3 = 0, so λ2 is irrelevant for the curve; carry it forward to keep the
        # λ2 time series continuous across NS fallbacks.
        params[5] = previous.lambda2
    curve = NSSCurve.from_array(params)
    fitted = curve.par_yield(tau) if target == "par" else curve.zero(tau)
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
    )


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
) -> tuple[FloatArray, float, int, bool, str]:
    """Variable-projection calibration of the zero curve to ``y``."""
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
    axes = [np.linspace(bounds[i, 0], bounds[i, 1], cfg.grid_size) for i in free]
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=-1)
    if model == "nss" and len(free) == 2:
        grid = grid[grid[:, 0] - grid[:, 1] >= log_ratio - 1e-12]
    _, grid_loss = objective_batch(grid)
    n_evals += grid.shape[0]

    step = np.array([(bounds[i, 1] - bounds[i, 0]) / (cfg.grid_size - 1) for i in free])
    starts = _distinct_minima(grid, grid_loss, min_separation=3.0 * step, k=cfg.n_starts)
    if prev_log is not None:
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


def _distinct_minima(
    grid: FloatArray, loss: FloatArray, min_separation: FloatArray, k: int
) -> list[FloatArray]:
    """Up to ``k`` lowest grid points that are pairwise well separated.

    Refining several *distinct* basins rather than only the single best grid
    point guards against the NSS surface's competing local minima.
    """
    chosen: list[FloatArray] = []
    for idx in np.argsort(loss):
        if not np.isfinite(loss[idx]):
            break
        cand = grid[idx]
        if all(np.any(np.abs(cand - c) > min_separation) for c in chosen):
            chosen.append(cand)
            if len(chosen) == k:
                break
    return chosen


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


def _calibrate_par(
    tau: FloatArray,
    y: FloatArray,
    w: FloatArray,
    cfg: CalibrationConfig,
    model: str,
    start: FloatArray,
    previous: NSSCurve | None,
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
        bounds=([-np.inf] * k + lo, [np.inf] * k + hi),
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    loss0 = float(np.sum(residuals(x0) ** 2))
    x_best = res.x if np.isfinite(res.cost) and 2 * res.cost <= loss0 else x0
    betas, log_lam = unpack(x_best)
    lam = np.exp(log_lam)
    if ratio_param and not (cfg.lambda1_bounds[0] - 1e-9 <= lam[0] <= cfg.lambda1_bounds[1] + 1e-9):
        # The reparametrised box does not bound λ1 itself; fall back to the
        # (always feasible) zero-curve starting point in that rare case.
        betas, log_lam = unpack(x0)
        lam = np.exp(log_lam)
        return (
            _full_params(betas, lam, model),
            loss0,
            int(res.nfev),
            False,
            "lambda1 left its bounds",
        )
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
        ``runtime_ms`` per date.
    fitted, residuals_bp:
        Model yields (percent) and observed-minus-fitted residuals (bp),
        aligned with the input panel (NaN where a tenor was not observed).
    """

    params: pd.DataFrame
    diagnostics: pd.DataFrame
    fitted: pd.DataFrame
    residuals_bp: pd.DataFrame
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

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
                "runtime_ms": res.runtime_ms,
            }
        )
        if np.all(np.isfinite(res.curve.as_array())):
            mask = np.isfinite(values[i])
            fitted[i, mask] = res.fitted
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
    )


#: Default Δlog-λ smoothing penalty used by :func:`calibrate_panel`. Chosen on
#: synthetic data with known true curves (``benchmarks/tune_regularisation.py``):
#: it minimises the error of the fitted curve against the *true* curve and cuts
#: week-to-week λ jumps roughly five-fold, at ~0.2 bp of in-sample RMSE.
DEFAULT_PANEL_SMOOTHING = 1e-3
