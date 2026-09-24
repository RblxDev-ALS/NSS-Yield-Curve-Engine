"""Robust NS / NSS calibration.

Why not just ``scipy.optimize.minimize`` on all six parameters?
---------------------------------------------------------------
The NSS least-squares surface is non-convex in the decay rates and nearly
flat along directions where the two hump loadings become collinear
(``lambda1 ~= lambda2``). A local optimiser started from yesterday's
solution therefore either gets stuck in a stale basin or swings wildly
between equivalent parameterisations ("beta2 = +30, beta3 = -30").

This module uses a three-stage approach:

1. **Variable projection over a decay-rate grid.** For fixed decay rates
   the model is *linear* in the betas, so the (regularised, weighted)
   optimum is a closed-form ridge solve. Evaluating that on a log-spaced
   grid of ``(lambda1, lambda2)`` pairs is a fully vectorised global search.
2. **Bounded non-linear refinement.** ``scipy.optimize.least_squares``
   (trust-region reflective) polishes the best grid point *and* the
   previous date's solution (warm start); the lower objective wins, so the
   warm start adds stability without being able to trap the fit.
3. **Identification by construction.** ``lambda1`` (short/medium hump) and
   ``lambda2`` (long-end hump) live in disjoint ranges, which rules out the
   collinear region and label switching between the two curvature factors.

Optional penalties: a ridge on the curvature betas (shrinks unnecessary
humps toward zero), a temporal smoothness prior pulling betas toward the
previous date, and a log-lambda smoothness prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .model import (NSSCurve, NSSParams, ParYieldPricer, PARAM_NAMES_NSS,
                    cont_to_periodic, loading_matrix, periodic_to_cont)
from .tenors import maturities_of

_JITTER = 1e-12


@dataclass(frozen=True)
class CalibrationConfig:
    """Calibration settings.

    Attributes
    ----------
    model: ``"nss"`` (6 parameters) or ``"ns"`` (4 parameters).
    fit_target: ``"zero"`` fits the model zero curve to the quotes converted
        to continuous compounding (fast, the textbook approximation);
        ``"par"`` fits model *par yields* to the quotes, which is the correct
        treatment of CMT coupon-bond yields (slower, more accurate zeros).
    ridge: L2 penalty on the curvature betas (``beta2``, ``beta3``).
    smoothness: L2 penalty pulling betas toward the previous date's betas.
    lambda_smoothness: L2 penalty on ``log(lambda)`` changes vs. the previous date.
    lambda1_bounds / lambda2_bounds: disjoint decay-rate ranges (per year).
        The defaults put the ``beta2`` hump between ~0.4Y and ~7Y and the
        ``beta3`` hump between ~7Y and ~60Y.
    beta_bounds: bounds for ``beta0..beta3`` (percent).
    grid_points: grid resolution for ``lambda1`` and ``lambda2``.
    weights: optional ``{maturity_in_years: weight}``; unlisted tenors get 1.
    min_tenors: minimum number of quoted tenors on a date (default: number
        of parameters).
    freq: compounding frequency of the market quotes (2 = semi-annual).
    refine: run the non-linear refinement stage.

    The defaults were chosen by simulation (see ``benchmarks/calibration_benchmark.py``):
    on synthetic CMT data with 1bp noise plus 3bp persistent pricing errors,
    par-target fitting with light ridge and temporal priors recovers the
    true zero curve to ~2.2bp (legacy v1 calibrator: ~6.1bp) and cuts
    week-over-week parameter churn >10x relative to unregularised fitting,
    while still tracking genuine curve moves (corr. of weekly 10Y changes
    with the truth ~0.99).
    """

    model: str = "nss"
    fit_target: str = "par"
    ridge: float = 1e-5
    smoothness: float = 1e-3
    lambda_smoothness: float = 1e-3
    lambda1_bounds: tuple[float, float] = (0.25, 5.0)
    lambda2_bounds: tuple[float, float] = (0.03, 0.25)
    beta_bounds: tuple[tuple[float, float], ...] = (
        (-5.0, 25.0), (-30.0, 30.0), (-40.0, 40.0), (-40.0, 40.0))
    grid_points: tuple[int, int] = (30, 16)
    weights: Mapping[float, float] | None = None
    min_tenors: int | None = None
    freq: int = 2
    refine: bool = True

    def __post_init__(self):
        if self.model not in ("ns", "nss"):
            raise ValueError("model must be 'ns' or 'nss'")
        if self.fit_target not in ("zero", "par"):
            raise ValueError("fit_target must be 'zero' or 'par'")
        for name in ("ridge", "smoothness", "lambda_smoothness"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for lo, hi in (self.lambda1_bounds, self.lambda2_bounds):
            if not 0 < lo < hi:
                raise ValueError("lambda bounds must satisfy 0 < lo < hi")
        if self.svensson and self.lambda2_bounds[1] > self.lambda1_bounds[0]:
            raise ValueError("lambda2_bounds must lie below lambda1_bounds "
                             "(disjoint ranges keep the two humps identified)")

    @property
    def svensson(self) -> bool:
        return self.model == "nss"

    @property
    def n_betas(self) -> int:
        return 4 if self.svensson else 3

    @property
    def n_params(self) -> int:
        return 6 if self.svensson else 4

    @property
    def required_tenors(self) -> int:
        return self.min_tenors if self.min_tenors is not None else self.n_params

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        b = list(self.beta_bounds[: self.n_betas]) + [self.lambda1_bounds]
        if self.svensson:
            b.append(self.lambda2_bounds)
        lo, hi = np.array(b, dtype=float).T
        return lo, hi


@dataclass
class FitResult:
    """Outcome of a single-date calibration."""

    params: NSSParams | None
    maturities: np.ndarray
    market: np.ndarray
    fitted: np.ndarray
    objective: float
    success: bool
    message: str = ""
    start: str = ""           # which candidate won: "grid" or "warm"

    @property
    def residuals_bp(self) -> np.ndarray:
        """Market minus model, basis points (positive = market yield above model = cheap)."""
        return (self.market - self.fitted) * 100.0

    @property
    def rmse_bp(self) -> float:
        r = self.residuals_bp
        return float(np.sqrt(np.nanmean(r ** 2))) if np.isfinite(r).any() else np.nan

    @property
    def max_abs_error_bp(self) -> float:
        r = self.residuals_bp
        return float(np.nanmax(np.abs(r))) if np.isfinite(r).any() else np.nan

    @property
    def curve(self) -> NSSCurve:
        if self.params is None:
            raise ValueError("calibration failed; no curve available")
        return NSSCurve(self.params)


class NSSCalibrator:
    """Calibrates NS/NSS curves to one date or a panel of dates."""

    def __init__(self, config: CalibrationConfig | None = None, **overrides):
        config = config or CalibrationConfig()
        self.config = replace(config, **overrides) if overrides else config
        cfg = self.config
        g1, g2 = cfg.grid_points
        l1 = np.geomspace(*cfg.lambda1_bounds, g1)
        if cfg.svensson:
            l2 = np.geomspace(*cfg.lambda2_bounds, g2)
            L1, L2 = np.meshgrid(l1, l2, indexing="ij")
            self._grid = (L1.ravel(), L2.ravel())
        else:
            self._grid = (l1, None)
        self._cache: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ utils
    def _weights(self, tau: np.ndarray) -> np.ndarray:
        w = np.ones_like(tau)
        if self.config.weights:
            for mat, wt in self.config.weights.items():
                w[np.isclose(tau, float(mat))] = float(wt)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        return w / w.sum()

    def _prepared(self, tau: np.ndarray):
        key = tuple(np.round(tau, 10))
        if key not in self._cache:
            L = loading_matrix(tau, *self._grid)                   # (G, n, k)
            pricer = ParYieldPricer(tau, self.config.freq)
            self._cache[key] = (L, pricer)
        return self._cache[key]

    def _penalty_matrices(self, prior: NSSParams | None):
        cfg = self.config
        k = cfg.n_betas
        R = np.zeros((k, k))
        R[2, 2] = cfg.ridge
        if cfg.svensson:
            R[3, 3] = cfg.ridge
        use_prior = prior is not None and cfg.smoothness > 0
        S = np.eye(k) * cfg.smoothness if use_prior else np.zeros((k, k))
        bp = self._prior_betas(prior) if use_prior else np.zeros(k)
        return R, S, bp

    def _prior_betas(self, prior: NSSParams) -> np.ndarray:
        b = np.array([prior.beta0, prior.beta1, prior.beta2, prior.beta3])
        return b[: self.config.n_betas]

    def _prior_lambdas(self, prior: NSSParams) -> np.ndarray:
        cfg = self.config
        l1 = np.clip(prior.lambda1, *cfg.lambda1_bounds)
        if not cfg.svensson:
            return np.array([l1])
        l2 = prior.lambda2 if prior.lambda2 is not None else np.sqrt(np.prod(cfg.lambda2_bounds))
        return np.array([l1, np.clip(l2, *cfg.lambda2_bounds)])

    # ------------------------------------------------------ stage 1: grid/VP
    def _grid_search(self, L, target, w, prior):
        cfg = self.config
        R, S, bp = self._penalty_matrices(prior)
        k = L.shape[-1]
        WL = L * w[:, None]
        A = np.einsum("gnk,gnj->gkj", WL, L) + R + S + _JITTER * np.eye(k)
        rhs = np.einsum("gnk,n->gk", WL, target) + S @ bp
        betas = np.linalg.solve(A, rhs[..., None])[..., 0]
        resid = np.einsum("gnk,gk->gn", L, betas) - target
        obj = (w * resid ** 2).sum(-1) + np.einsum("gk,kj,gj->g", betas, R, betas)
        d = betas - bp
        obj += np.einsum("gk,kj,gj->g", d, S, d)
        if prior is not None and cfg.lambda_smoothness > 0:
            log_prior = np.log(self._prior_lambdas(prior))
            grid = np.stack([g for g in self._grid if g is not None], axis=-1)
            obj += cfg.lambda_smoothness * ((np.log(grid) - log_prior) ** 2).sum(-1)
        lo, hi = cfg.bounds()
        feasible = np.all((betas >= lo[:k]) & (betas <= hi[:k]), axis=-1)
        pool = np.where(feasible, obj, np.inf) if feasible.any() else obj
        i = int(np.argmin(pool))
        lam = [self._grid[0][i]] + ([self._grid[1][i]] if cfg.svensson else [])
        vec = np.concatenate([betas[i], lam])
        return vec

    # ------------------------------------------------- stage 2: refinement
    def _residual_fn(self, tau, market, target, sw, pricer, prior):
        cfg = self.config
        sv = cfg.svensson
        k = cfg.n_betas
        sqrt_ridge = np.sqrt(cfg.ridge)
        use_smooth = prior is not None and cfg.smoothness > 0
        use_lsmooth = prior is not None and cfg.lambda_smoothness > 0
        if use_smooth:
            bp, sqrt_s = self._prior_betas(prior), np.sqrt(cfg.smoothness)
        if use_lsmooth:
            log_lp, sqrt_ls = np.log(self._prior_lambdas(prior)), np.sqrt(cfg.lambda_smoothness)

        def fn(theta):
            curve = NSSCurve(NSSParams.from_vector(theta, sv))
            if cfg.fit_target == "zero":
                parts = [sw * (curve.zero(tau) - target)]
            else:
                parts = [sw * (pricer(curve) - market)]
            parts.append(sqrt_ridge * theta[2:k])
            if use_smooth:
                parts.append(sqrt_s * (theta[:k] - bp))
            if use_lsmooth:
                parts.append(sqrt_ls * (np.log(theta[k:]) - log_lp))
            return np.concatenate(parts)

        n_p = cfg.n_params
        pen_rows = [np.eye(n_p)[2:k] * sqrt_ridge]
        if use_smooth:
            pen_rows.append(np.eye(n_p)[:k] * sqrt_s)
        pen_const = np.vstack(pen_rows)

        def jac(theta):
            curve = NSSCurve(NSSParams.from_vector(theta, sv))
            if cfg.fit_target == "zero":
                J = curve.zero_jacobian(tau)
            else:
                J = pricer.jacobian(curve)
            rows = [sw[:, None] * J, pen_const]
            if use_lsmooth:
                rows.append(np.hstack([np.zeros((n_p - k, k)),
                                       np.diag(sqrt_ls / theta[k:])]))
            return np.vstack(rows)

        return fn, jac

    def _polish(self, fn, jac, x0):
        lo, hi = self.config.bounds()
        span = hi - lo
        x0 = np.clip(x0, lo + 1e-9 * span, hi - 1e-9 * span)
        if not self.config.refine:
            r = fn(x0)
            return x0, float(r @ r), True, "grid only"
        try:
            sol = least_squares(fn, x0, jac=jac, bounds=(lo, hi), method="trf",
                                x_scale="jac", ftol=1e-10, xtol=1e-9, gtol=1e-10,
                                max_nfev=200)
            return sol.x, float(2.0 * sol.cost), True, sol.message
        except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover
            r = fn(x0)
            return x0, float(r @ r), False, f"refinement failed: {exc}"

    # --------------------------------------------------------------- public
    def fit(self, maturities: Sequence[float], yields: Sequence[float],
            prior: NSSParams | None = None) -> FitResult:
        """Calibrate to one curve of market quotes (percent, bond-equivalent).

        NaN quotes are ignored. ``prior`` (usually the previous date's
        solution) is used as an additional warm start and, if configured,
        as the centre of the smoothness penalties.
        """
        cfg = self.config
        tau_all = np.asarray(maturities, dtype=float)
        y_all = np.asarray(yields, dtype=float)
        mask = np.isfinite(y_all) & np.isfinite(tau_all) & (tau_all > 0)
        fitted = np.full_like(y_all, np.nan)
        if mask.sum() < cfg.required_tenors:
            return FitResult(None, tau_all, y_all, fitted, np.nan, False,
                             f"only {int(mask.sum())} valid tenors "
                             f"(need {cfg.required_tenors})")

        tau, market = tau_all[mask], y_all[mask]
        target = periodic_to_cont(market, cfg.freq)
        w = self._weights(tau)
        L, pricer = self._prepared(tau)
        fn, jac = self._residual_fn(tau, market, target, np.sqrt(w), pricer, prior)

        candidates = [("grid", self._grid_search(L, target, w, prior))]
        if prior is not None:
            candidates.append(("warm", np.concatenate([self._prior_betas(prior),
                                                        self._prior_lambdas(prior)])))
        best = None
        for label, x0 in candidates:
            x, obj, ok, msg = self._polish(fn, jac, x0)
            if np.all(np.isfinite(x)) and (best is None or obj < best[1] - 1e-15):
                best = (x, obj, ok, msg, label)

        x, obj, ok, msg, label = best
        params = NSSParams.from_vector(x, cfg.svensson)
        curve = NSSCurve(params)
        if cfg.fit_target == "par":
            fitted[mask] = pricer(curve)
        else:
            fitted[mask] = cont_to_periodic(curve.zero(tau), cfg.freq)
        return FitResult(params, tau_all, y_all, fitted, obj, ok, str(msg), label)

    def fit_history(self, rates: pd.DataFrame,
                    maturities: Sequence[float] | None = None,
                    progress: bool = False) -> "CurveHistory":
        """Sequentially calibrate every row of ``rates`` (index = dates,
        columns = tenors), warm-starting each date from the previous one."""
        mats = maturities_of(rates.columns) if maturities is None \
            else np.asarray(maturities, dtype=float)
        values = rates.to_numpy(dtype=float)
        rows_p, rows_f, rows_d = [], [], []
        prior = None
        n = len(rates)
        for i, y in enumerate(values):
            res = self.fit(mats, y, prior=prior)
            if res.params is not None:
                prior = res.params
                rows_p.append(res.params.to_dict())
            else:
                rows_p.append(dict.fromkeys(PARAM_NAMES_NSS, np.nan))
            rows_f.append(res.fitted)
            rows_d.append({"RMSE_bp": res.rmse_bp, "MaxAbsErr_bp": res.max_abs_error_bp,
                           "Objective": res.objective, "NTenors": int(np.isfinite(y).sum()),
                           "Converged": res.success, "Start": res.start})
            if progress and (i + 1) % 50 == 0:
                print(f"  calibrated {i + 1}/{n} dates")

        idx = rates.index
        params = pd.DataFrame(rows_p, index=idx, columns=list(PARAM_NAMES_NSS))
        if not self.config.svensson:
            params = params.drop(columns=["Beta3", "Lambda2"])
        fitted = pd.DataFrame(np.vstack(rows_f) if rows_f else np.empty((0, len(mats))),
                              index=idx, columns=rates.columns)
        diagnostics = pd.DataFrame(rows_d, index=idx)
        return CurveHistory(params=params, market=rates.copy(), fitted=fitted,
                            diagnostics=diagnostics, maturities=mats, config=self.config)


@dataclass
class CurveHistory:
    """Panel of calibrated curves."""

    params: pd.DataFrame
    market: pd.DataFrame
    fitted: pd.DataFrame
    diagnostics: pd.DataFrame
    maturities: np.ndarray
    config: CalibrationConfig = field(default_factory=CalibrationConfig)

    @property
    def residuals_bp(self) -> pd.DataFrame:
        """Market minus model yields (bp); positive = cheap to the curve."""
        return (self.market - self.fitted) * 100.0

    @property
    def valid(self) -> pd.Index:
        return self.params.dropna(subset=["Beta0"]).index

    def params_at(self, date=None) -> NSSParams:
        p = self.params.loc[self.valid]
        row = p.iloc[-1] if date is None else p.loc[:date].iloc[-1]
        return NSSParams.from_dict(row)

    def curve(self, date=None) -> NSSCurve:
        """Curve on ``date`` (or the last calibrated date on/before it; latest if None)."""
        return NSSCurve(self.params_at(date))

    def model_rates(self, maturities: Sequence[float], kind: str = "zero",
                    labels: Sequence[str] | None = None) -> pd.DataFrame:
        """Time series of model rates at arbitrary maturities.

        ``kind``: ``"zero"`` (continuous), ``"forward"`` (instantaneous),
        ``"par"`` (bond-equivalent par yields) or ``"market"`` (quote
        convention implied by the fit target).
        """
        mats = np.asarray(maturities, dtype=float)
        pricer = ParYieldPricer(mats, self.config.freq) if kind == "par" else None
        rows = []
        for _, row in self.params.loc[self.valid].iterrows():
            c = NSSCurve(NSSParams.from_dict(row))
            if kind == "zero":
                rows.append(c.zero(mats))
            elif kind == "forward":
                rows.append(c.forward(mats))
            elif kind == "par":
                rows.append(pricer(c))
            elif kind == "market":
                rows.append(c.market_yield(mats, self.config.fit_target, self.config.freq))
            else:
                raise ValueError(f"unknown kind {kind!r}")
        cols = list(labels) if labels is not None else list(mats)
        return pd.DataFrame(rows, index=self.valid, columns=cols)

    def summary(self) -> dict:
        d = self.diagnostics.loc[self.valid]
        p = self.params.loc[self.valid]
        # Day-over-day parameter changes: a direct measure of calibration stability.
        jumps = p.diff().abs().median().to_dict()
        return {
            "dates": int(len(self.params)),
            "calibrated": int(len(p)),
            "mean_rmse_bp": float(d["RMSE_bp"].mean()),
            "median_rmse_bp": float(d["RMSE_bp"].median()),
            "p95_rmse_bp": float(d["RMSE_bp"].quantile(0.95)),
            "worst_abs_error_bp": float(d["MaxAbsErr_bp"].max()),
            "median_abs_param_change": {k: float(v) for k, v in jumps.items()},
        }
