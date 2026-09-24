"""Benchmark: original (v0) calibration vs. the variable-projection calibrator.

The v0 engine fitted all six NSS parameters at once with L-BFGS-B, warm-started
from the previous week, with an L2 penalty of 0.005·(β2² + β3²) added to the
mean squared error. This script re-implements that routine verbatim and
compares it with the current calibrator on a synthetic market whose *true*
curves are known, so we can measure not only in-sample fit but how close each
method gets to the truth and how stable the parameters are.

Usage::

    python benchmarks/compare_legacy.py              # 2 seeds × 10 years weekly
    python benchmarks/compare_legacy.py --years 30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nss_engine.calibration import calibrate, calibrate_panel
from nss_engine.models import NSSCurve, nss_zero
from nss_engine.synthetic import simulate_market

GRID = np.linspace(0.25, 30, 60)


def legacy_calibrate(yields: np.ndarray, maturities: np.ndarray, last_params=None, l2_lambda=0.005):
    """The v0 ``NSSCurveModel.calibrate`` method, unchanged apart from taking arrays."""

    def objective(params):
        b0, b1, b2, b3, l1, l2 = params
        preds = nss_zero(maturities, b0, b1, b2, b3, l1, l2)
        mse = np.mean((yields - preds) ** 2)
        penalty = l2_lambda * (b2**2 + b3**2)
        return mse + penalty

    bounds = [(0, 15), (-15, 15), (-20, 20), (-20, 20), (0.2, 5.0), (0.2, 5.0)]
    init_guess = (
        last_params
        if last_params is not None
        else [yields[-1], yields[-1] - yields[0], 0, 0, 1.0, 2.0]
    )
    res = minimize(objective, init_guess, method="L-BFGS-B", bounds=bounds, tol=1e-7)
    return res.x if res.success else np.full(6, np.nan)


def run_legacy(panel: pd.DataFrame, l2_lambda: float) -> tuple[pd.DataFrame, float]:
    mats = np.asarray(panel.columns, dtype=float)
    out, prev = [], None
    t0 = time.perf_counter()
    for row in panel.to_numpy():
        p = legacy_calibrate(row, mats, prev, l2_lambda)
        prev = p if not np.isnan(p).any() else prev
        out.append(p)
    ms = (time.perf_counter() - t0) / len(panel) * 1e3
    cols = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    return pd.DataFrame(out, index=panel.index, columns=cols), ms


def run_single_date(panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    mats = np.asarray(panel.columns, dtype=float)
    t0 = time.perf_counter()
    rows = [calibrate(mats, r).curve.as_array() for r in panel.to_numpy()]
    ms = (time.perf_counter() - t0) / len(panel) * 1e3
    return pd.DataFrame(rows, index=panel.index, columns=GRID_COLS), ms


GRID_COLS = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]


def run_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    fit = calibrate_panel(panel)
    ms = (time.perf_counter() - t0) / len(panel) * 1e3
    return fit.params.reindex(panel.index), ms


def metrics(
    params: pd.DataFrame, panel: pd.DataFrame, truth: pd.DataFrame, ms: float
) -> dict[str, float]:
    mats = np.asarray(panel.columns, dtype=float)
    ok = params.notna().all(axis=1)
    p = params[ok].to_numpy()
    fitted = np.array(
        [NSSCurve.from_array(x).zero(mats) if x[4] > 0 else np.full(mats.size, np.nan) for x in p]
    )
    resid_bp = (panel[ok].to_numpy() - fitted) * 100
    rmse = np.sqrt(np.nanmean(resid_bp**2, axis=1))
    est = np.array([NSSCurve.from_array(x).zero(GRID) for x in p])
    tru = np.array([NSSCurve.from_array(x).zero(GRID) for x in truth[ok].to_numpy()])
    curve_err = np.sqrt(np.mean((est - tru) ** 2, axis=1)) * 100
    d = params[ok].diff().abs().median()
    return {
        "fit RMSE (bp)": float(np.mean(rmse)),
        "fit RMSE p99 (bp)": float(np.percentile(rmse, 99)),
        "error vs true curve (bp)": float(np.mean(curve_err)),
        "error vs true curve p99 (bp)": float(np.percentile(curve_err, 99)),
        "failed fits (%)": float((~ok).mean() * 100),
        "median |Δβ2| per week": float(d["beta2"]),
        "median |Δλ1| per week": float(d["lambda1"]),
        "ms per curve": ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--noise-bp", type=float, default=3.0)
    args = ap.parse_args()

    methods = {
        "v0: L-BFGS-B + ridge 0.005 (original)": lambda y: run_legacy(y, 0.005),
        "v0 without ridge": lambda y: run_legacy(y, 0.0),
        "v1: variable projection, per date": run_single_date,
        "v1: variable projection + λ smoothing (default)": run_panel,
    }
    results: dict[str, list[dict[str, float]]] = {k: [] for k in methods}
    for seed in args.seeds:
        mkt = simulate_market(
            periods=52 * args.years, seed=seed, noise_bp=args.noise_bp, missing_short_end_until=None
        )
        for name, fn in methods.items():
            params, ms = fn(mkt.yields)
            results[name].append(metrics(params, mkt.yields, mkt.true_params, ms))
            print(f"seed {seed}: {name} done", flush=True)

    table = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in results.items()}).T
    table.index.name = "method"
    pd.set_option("display.width", 200)
    print()
    print(
        f"{len(args.seeds)} seeds × {args.years} years of weekly curves, {args.noise_bp:g} bp quote noise"
    )
    print(table.round(3).to_markdown() if _has_tabulate() else table.round(3).to_string())


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
