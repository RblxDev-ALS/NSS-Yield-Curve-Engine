"""Choose the ridge and λ-smoothing penalties on data with a known answer.

In-sample RMSE always prefers *no* regularisation. The right criterion is how
close the fitted curve is to the **true** curve across all maturities, which
we can only measure on synthetic data. This sweep reproduces the numbers
behind the defaults in :mod:`nss_engine.calibration`.

Usage::

    python benchmarks/tune_regularisation.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from nss_engine.calibration import CalibrationConfig, calibrate_panel
from nss_engine.models import NSSCurve
from nss_engine.synthetic import simulate_market

GRID = np.linspace(0.1, 30, 60)


def main(seeds=(1, 2), periods=300) -> None:
    rows = []
    for seed in seeds:
        mkt = simulate_market(periods=periods, seed=seed)
        truth = np.array([NSSCurve.from_array(p).zero(GRID) for p in mkt.true_params.to_numpy()])
        for ridge, smoothing in itertools.product([0.0, 1e-5, 1e-4], [0.0, 1e-4, 1e-3, 1e-2, 1e-1]):
            fit = calibrate_panel(
                mkt.yields, CalibrationConfig(ridge=ridge, lambda_smoothing=smoothing)
            )
            est = fit.evaluate(GRID).to_numpy()
            rows.append(
                {
                    "ridge": ridge,
                    "smoothing": smoothing,
                    "error vs truth (bp)": np.sqrt(np.mean((est - truth) ** 2)) * 100,
                    "fit RMSE (bp)": fit.diagnostics["rmse_bp"].mean(),
                    "median |Δλ1|": fit.params["lambda1"].diff().abs().median(),
                }
            )
    table = pd.DataFrame(rows).groupby(["ridge", "smoothing"]).mean()
    print(table.round(3).to_string())
    best = table["error vs truth (bp)"].idxmin()
    print(f"\nlowest error vs truth: ridge={best[0]:g}, smoothing={best[1]:g}")


if __name__ == "__main__":
    main()
