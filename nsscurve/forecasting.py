"""Dynamic NSS factor model for yield-curve forecasting (Diebold & Li, 2006).

1. Fix the decay rates across all dates (estimated jointly by minimising
   the pooled squared error over a lambda grid) so the betas become linear
   regression coefficients with stable interpretation over time.
2. Extract betas date by date (closed-form least squares).
3. Fit factor dynamics: independent AR(1)s (the Diebold-Li specification)
   or a VAR(1).
4. Forecast betas ``h`` steps ahead and map them back to yields.

:meth:`DieboldLiForecaster.backtest` runs an expanding-window, out-of-sample
evaluation against the random-walk benchmark (tomorrow's curve = today's),
which is notoriously hard to beat; results are reported honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import NSSCurve, NSSParams, cont_to_periodic, loading_matrix, periodic_to_cont
from .tenors import maturities_of


def _fit_dynamics(betas: np.ndarray, kind: str):
    """OLS for ``b_t = c + Phi b_{t-1} + e``; returns ``(c, Phi)``."""
    y, x = betas[1:], betas[:-1]
    k = betas.shape[1]
    if kind == "var":
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef[0], coef[1:].T
    c, phi = np.zeros(k), np.zeros((k, k))
    for j in range(k):
        X = np.column_stack([np.ones(len(x)), x[:, j]])
        (c[j], phi[j, j]), *_ = np.linalg.lstsq(X, y[:, j], rcond=None)
    return c, phi


def _iterate(beta_last: np.ndarray, c: np.ndarray, phi: np.ndarray, h: int) -> np.ndarray:
    path = np.empty((h, len(beta_last)))
    b = beta_last
    for i in range(h):
        b = c + phi @ b
        path[i] = b
    return path


@dataclass
class DieboldLiForecaster:
    """Fixed-lambda NS/NSS dynamic factor model.

    Parameters
    ----------
    model: ``"nss"`` or ``"ns"``.
    dynamics: ``"ar"`` (per-factor AR(1)) or ``"var"`` (VAR(1)).
    lambdas: fixed decay rates; estimated from the data if None.
    ridge: small L2 penalty on the curvature betas for numerical stability.
    freq: compounding frequency of the quotes.
    """

    model: str = "nss"
    dynamics: str = "ar"
    lambdas: tuple[float, ...] | None = None
    ridge: float = 1e-6
    freq: int = 2
    lambda1_bounds: tuple[float, float] = (0.25, 5.0)
    lambda2_bounds: tuple[float, float] = (0.03, 0.25)

    def __post_init__(self):
        if self.model not in ("ns", "nss"):
            raise ValueError("model must be 'ns' or 'nss'")
        if self.dynamics not in ("ar", "var"):
            raise ValueError("dynamics must be 'ar' or 'var'")
        self.betas_: pd.DataFrame | None = None

    # ----------------------------------------------------------- estimation
    def _loadings(self, mats, lambdas):
        return loading_matrix(mats, lambdas[0], lambdas[1] if self.model == "nss" else None)

    def estimate_lambdas(self, rates: pd.DataFrame, grid_points=(40, 20)) -> tuple[float, ...]:
        """Jointly optimal fixed decay rates (pooled SSE over complete dates)."""
        full = rates.dropna()
        if len(full) < 3:
            raise ValueError("need at least 3 dates with complete data")
        mats = maturities_of(full.columns)
        Y = periodic_to_cont(full.to_numpy(), self.freq)
        l1 = np.geomspace(*self.lambda1_bounds, grid_points[0])
        if self.model == "nss":
            L1, L2 = np.meshgrid(l1, np.geomspace(*self.lambda2_bounds, grid_points[1]),
                                 indexing="ij")
            grid = (L1.ravel(), L2.ravel())
        else:
            grid = (l1, None)
        X = loading_matrix(mats, *grid)                                    # (G, n, k)
        k = X.shape[-1]
        R = np.diag([0, 0] + [self.ridge] * (k - 2))
        A = np.einsum("gnk,gnj->gkj", X, X) + R
        B = np.linalg.solve(A, np.einsum("gnk,tn->gkt", X, Y))              # (G, k, T)
        resid = np.einsum("gnk,gkt->gtn", X, B) - Y[None]
        sse = (resid ** 2).sum(axis=(1, 2))
        i = int(np.argmin(sse))
        return (float(grid[0][i]),) + ((float(grid[1][i]),) if self.model == "nss" else ())

    def extract_betas(self, rates: pd.DataFrame, lambdas) -> pd.DataFrame:
        mats = maturities_of(rates.columns)
        X = self._loadings(mats, lambdas)
        k = X.shape[1]
        R = np.diag([0, 0] + [self.ridge] * (k - 2))
        rows = []
        for y in rates.to_numpy(dtype=float):
            m = np.isfinite(y)
            if m.sum() < k:
                rows.append(np.full(k, np.nan))
                continue
            Xm, ym = X[m], periodic_to_cont(y[m], self.freq)
            rows.append(np.linalg.solve(Xm.T @ Xm + R, Xm.T @ ym))
        cols = ["Beta0", "Beta1", "Beta2", "Beta3"][:k]
        return pd.DataFrame(rows, index=rates.index, columns=cols)

    def fit(self, rates: pd.DataFrame) -> "DieboldLiForecaster":
        self.columns_ = list(rates.columns)
        self.maturities_ = maturities_of(rates.columns)
        self.lambdas_ = tuple(self.lambdas) if self.lambdas else self.estimate_lambdas(rates)
        self.betas_ = self.extract_betas(rates, self.lambdas_).dropna()
        self.c_, self.phi_ = _fit_dynamics(self.betas_.to_numpy(), self.dynamics)
        return self

    # -------------------------------------------------------------- forecast
    def _curve(self, betas) -> NSSCurve:
        b = list(betas) + [0.0] * (4 - len(betas))
        l2 = self.lambdas_[1] if self.model == "nss" else None
        return NSSCurve(NSSParams(*b, self.lambdas_[0], l2))

    def forecast(self, horizon: int = 4) -> pd.DataFrame:
        """Forecast quotes (market convention) for each step 1..horizon."""
        if self.betas_ is None:
            raise RuntimeError("call fit() first")
        path = _iterate(self.betas_.iloc[-1].to_numpy(), self.c_, self.phi_, horizon)
        rows = [cont_to_periodic(self._curve(b).zero(self.maturities_), self.freq) for b in path]
        return pd.DataFrame(rows, index=pd.RangeIndex(1, horizon + 1, name="StepsAhead"),
                            columns=self.columns_)

    def factor_forecast(self, horizon: int = 4) -> pd.DataFrame:
        path = _iterate(self.betas_.iloc[-1].to_numpy(), self.c_, self.phi_, horizon)
        return pd.DataFrame(path, index=pd.RangeIndex(1, horizon + 1, name="StepsAhead"),
                            columns=self.betas_.columns)

    def persistence(self) -> pd.Series:
        """Own-lag coefficients (diagonal of Phi): near 1 = highly persistent factor.

        For ``dynamics='var'`` persistence also flows through cross terms;
        use :meth:`spectral_radius` for the system as a whole.
        """
        return pd.Series(np.diag(self.phi_), index=self.betas_.columns, name="Phi")

    def spectral_radius(self) -> float:
        """Largest |eigenvalue| of Phi; < 1 means the factor dynamics are stationary."""
        return float(np.max(np.abs(np.linalg.eigvals(self.phi_))))

    # -------------------------------------------------------------- backtest
    def backtest(self, rates: pd.DataFrame, horizon: int = 4, min_train: int = 104,
                 step: int = 1) -> dict:
        """Expanding-window out-of-sample RMSE (bp) vs a random walk, per tenor.

        Lambdas are estimated once on the first ``min_train`` observations
        only, so no future information leaks into the forecasts.
        """
        if len(rates) < min_train + horizon + 5:
            raise ValueError("series too short for the requested backtest")
        lambdas = tuple(self.lambdas) if self.lambdas else \
            self.estimate_lambdas(rates.iloc[:min_train])
        self.lambdas_ = lambdas
        mats = maturities_of(rates.columns)
        betas = self.extract_betas(rates, lambdas).to_numpy()
        values = rates.to_numpy(dtype=float)
        err_m, err_rw, dates = [], [], []
        for t in range(min_train - 1, len(rates) - horizon, step):
            hist = betas[: t + 1]
            hist = hist[np.isfinite(hist).all(axis=1)]
            if len(hist) < 20 or not np.isfinite(betas[t]).all():
                continue
            c, phi = _fit_dynamics(hist, self.dynamics)
            b_h = _iterate(betas[t], c, phi, horizon)[-1]
            pred = cont_to_periodic(self._curve(b_h).zero(mats), self.freq)
            actual = values[t + horizon]
            err_m.append((pred - actual) * 100.0)
            err_rw.append((values[t] - actual) * 100.0)
            dates.append(rates.index[t])
        em, er = np.array(err_m), np.array(err_rw)
        rmse_m = np.sqrt(np.nanmean(em ** 2, axis=0))
        rmse_rw = np.sqrt(np.nanmean(er ** 2, axis=0))
        table = pd.DataFrame({"RMSE_Model_bp": rmse_m, "RMSE_RandomWalk_bp": rmse_rw,
                              "Ratio": rmse_m / rmse_rw}, index=rates.columns)
        return {"table": table, "n_forecasts": len(dates), "horizon": horizon,
                "lambdas": lambdas,
                "errors_model": pd.DataFrame(em, index=dates, columns=rates.columns),
                "errors_random_walk": pd.DataFrame(er, index=dates, columns=rates.columns)}
