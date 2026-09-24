"""Benchmark: legacy calibrator (v1 script) vs the new engine, against known truth.

Simulates CMT-style par-yield quotes from a dynamic NSS model with known
parameters, then measures for each calibration method:

* fit RMSE to the quotes (bp)
* error of the recovered zero and forward curves vs. the truth (bp)
* failures (dates returning NaN)
* week-over-week parameter churn (stability)
* runtime

Run:  python benchmarks/calibration_benchmark.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsscurve.calibration import CalibrationConfig, NSSCalibrator  # noqa: E402
from nsscurve.data import resample_curve, simulate_curve  # noqa: E402
from nsscurve.model import NSSCurve, NSSParams  # noqa: E402
from nsscurve.tenors import maturities_of  # noqa: E402


def legacy_fit_history(rates: pd.DataFrame, l2_lambda: float = 0.005) -> pd.DataFrame:
    """Faithful re-implementation of the original ``NSSCurveModel`` loop."""
    mats = maturities_of(rates.columns)

    def nss(tau, b0, b1, b2, b3, l1, l2):
        tau = np.maximum(tau, 1e-6)
        t1 = (1 - np.exp(-l1 * tau)) / (l1 * tau)
        t2 = t1 - np.exp(-l1 * tau)
        t3 = ((1 - np.exp(-l2 * tau)) / (l2 * tau)) - np.exp(-l2 * tau)
        return b0 + b1 * t1 + b2 * t2 + b3 * t3

    rows, prev = [], None
    for _, y in rates.iterrows():
        yv = y.to_numpy()

        def objective(p, yv=yv):
            mse = np.mean((yv - nss(mats, *p)) ** 2)
            return mse + l2_lambda * (p[2] ** 2 + p[3] ** 2)

        bounds = [(0, 15), (-15, 15), (-20, 20), (-20, 20), (0.2, 5.0), (0.2, 5.0)]
        x0 = prev if prev is not None else [y.iloc[-1], y.iloc[-1] - y.iloc[0], 0, 0, 1.0, 2.0]
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, tol=1e-7)
        p = res.x if res.success else np.full(6, np.nan)
        prev = p if not np.isnan(p).any() else prev
        rows.append(p)
    return pd.DataFrame(rows, index=rates.index,
                        columns=["Beta0", "Beta1", "Beta2", "Beta3", "Lambda1", "Lambda2"])


def evaluate(params: pd.DataFrame, truth: pd.DataFrame, rates: pd.DataFrame,
             legacy: bool) -> dict:
    grid = np.linspace(0.25, 30, 120)
    mats = maturities_of(rates.columns)
    ok = params.dropna().index
    z_err, f_err, fit = [], [], []
    for d in ok:
        p = params.loc[d].to_numpy()
        est = NSSCurve(NSSParams(*p))
        tru = NSSCurve(NSSParams(*truth.loc[d].to_numpy()))
        # The legacy model treats its output directly as the quote, i.e. as a
        # zero curve in quote units; compare like with like.
        z_err.append(est.zero(grid) - tru.zero(grid))
        f_err.append(est.forward(grid) - tru.forward(grid))
        model_quotes = est.zero(mats) if legacy else est.par_yield(mats)
        fit.append(rates.loc[d].to_numpy() - model_quotes)
    rms = lambda a: float(np.sqrt(np.nanmean(np.square(a))) * 100)
    churn = params.loc[ok].diff().abs().median()
    return {"fit_rmse_bp": rms(fit), "zero_err_bp": rms(z_err), "fwd_err_bp": rms(f_err),
            "failures": int(len(params) - len(ok)),
            "beta_churn": float(churn[["Beta0", "Beta1", "Beta2", "Beta3"]].mean()),
            "lambda1_churn": float(churn["Lambda1"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--periods", type=int, default=1300)
    args = ap.parse_args(argv)

    methods = {
        "legacy v1 (L-BFGS-B, ridge 0.005)": None,
        "new, zero target, no priors": CalibrationConfig(fit_target="zero", ridge=0,
                                                         smoothness=0, lambda_smoothness=0),
        "new, par target, no priors": CalibrationConfig(fit_target="par", ridge=0,
                                                        smoothness=0, lambda_smoothness=0),
        "new, defaults (par + priors)": CalibrationConfig(),
    }
    records = []
    for seed in range(args.seeds):
        daily, factors = simulate_curve(periods=args.periods, seed=100 + seed,
                                        return_factors=True)
        rates = resample_curve(daily, "W")
        truth = factors.resample("W-FRI").last().loc[rates.index]
        for name, cfg in methods.items():
            t0 = time.perf_counter()
            if cfg is None:
                params = legacy_fit_history(rates)
            else:
                params = NSSCalibrator(cfg).fit_history(rates).params
            elapsed = time.perf_counter() - t0
            rec = evaluate(params, truth, rates, legacy=cfg is None)
            rec.update(method=name, seed=seed, seconds=elapsed)
            records.append(rec)

    df = pd.DataFrame(records).groupby("method", sort=False).mean(numeric_only=True)
    df = df.drop(columns="seed")
    print(f"\n{args.seeds} simulated 5y weekly panels, 1bp noise + 3bp persistent pricing errors\n")
    with pd.option_context("display.width", 160):
        print(df.round(3).to_string())


if __name__ == "__main__":
    main()
