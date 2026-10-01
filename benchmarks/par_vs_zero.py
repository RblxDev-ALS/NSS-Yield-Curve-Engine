"""Numbers for the research note ``docs/par-vs-zero.md``.

1. **Anatomy of the gap.** For stylised zero curves, how far a semi-annual
   par yield sits from the continuously compounded zero rate of the same
   maturity, split into a compounding part and a coupon part. This is the error
   made by reading a CMT quote as a zero rate, before any fitting.
2. **Where the fitted curve goes wrong.** On the simulated par-quoted market
   (known true curves), the zero-curve error of a fit that reads the quotes as
   zero rates and of a par fit, by maturity and for the 5y5y forward rate.

Usage::

    python benchmarks/par_vs_zero.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nss_engine.calibration import DEFAULT_PANEL_SMOOTHING, CalibrationConfig, calibrate_panel
from nss_engine.models import NSSCurve
from nss_engine.synthetic import simulate_market

TENORS = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

CURVES = {
    "flat 5%": NSSCurve(5.0, 0.0, 0.0, 0.0, 0.5, 0.1),
    "flat 2%": NSSCurve(2.0, 0.0, 0.0, 0.0, 0.5, 0.1),
    "steep, 1% to 5% (2010s)": NSSCurve(5.2, -4.4, -1.0, 0.5, 0.6, 0.12),
    "inverted, 5% to 4% (2023)": NSSCurve(4.0, 1.4, -1.5, 0.5, 0.9, 0.15),
    "hump at 20Y, 1% to 3%": NSSCurve(2.8, -1.9, -2.0, 3.0, 0.6, 0.09),
}


def anatomy() -> pd.DataFrame:
    rows = {}
    for name, curve in CURVES.items():
        z = curve.zero(TENORS)
        q = curve.par_yield(TENORS)  # semi-annual par yield: the CMT quote
        q_cc = 200.0 * np.log1p(q / 200.0)  # the same coupon rate, continuously compounded
        for t, zi, qi, qci in zip(TENORS, z, q, q_cc, strict=True):
            rows[(name, f"{t:g}Y")] = {
                "zero (cc, %)": zi,
                "par quote (sa, %)": qi,
                "quote − zero (bp)": (qi - zi) * 100,
                "compounding (bp)": (qi - qci) * 100,
                "coupon (bp)": (qci - zi) * 100,
            }
    out = pd.DataFrame(rows).T
    out.index.names = ["curve", "maturity"]
    return out


def fitted_errors(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mkt = simulate_market(seed=seed)
    monthly = mkt.yields.resample("ME").last()
    truth = mkt.true_params.resample("ME").last()
    grid = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    true_z = np.array([NSSCurve.from_mapping(r).zero(grid) for _, r in truth.iterrows()])
    true_f = np.array([_fwd_5y5y(NSSCurve.from_mapping(r)) for _, r in truth.iterrows()])
    slope = np.array(
        [
            NSSCurve.from_mapping(r).zero([10.0])[0] - NSSCurve.from_mapping(r).zero([0.25])[0]
            for _, r in truth.iterrows()
        ]
    )
    level = true_z.mean(axis=1)
    design = np.column_stack([np.ones_like(slope), level**2, slope])
    bias, rmse, sens = {}, {}, {}
    for label, target in (("quotes read as zero rates", "yield"), ("par fit (default)", "par")):
        cfg = CalibrationConfig(target=target, lambda_smoothing=DEFAULT_PANEL_SMOOTHING)
        fit = calibrate_panel(monthly, cfg).params
        z = np.array([NSSCurve.from_mapping(r).zero(grid) for _, r in fit.iterrows()])
        f = np.array([_fwd_5y5y(NSSCurve.from_mapping(r)) for _, r in fit.iterrows()])
        err = (z - true_z) * 100
        ferr = (f - true_f) * 100
        cols = [f"{g:g}Y" for g in grid] + ["5y5y fwd"]
        bias[label] = dict(zip(cols, [*err.mean(axis=0), ferr.mean()], strict=True))
        rmse[label] = dict(
            zip(cols, [*np.sqrt((err**2).mean(axis=0)), np.sqrt((ferr**2).mean())], strict=True)
        )
        # error = a + b·level² + c·slope: what the compounding and coupon effects predict
        for j, g in enumerate(grid):
            if g in (2.0, 10.0, 30.0):
                coef, *_ = np.linalg.lstsq(design, err[:, j], rcond=None)
                resid = err[:, j] - design @ coef
                sens[(label, f"{g:g}Y")] = {
                    "bp per unit of level²": coef[1],
                    "bp per point of slope": coef[2],
                    "R²": 1 - resid.var() / err[:, j].var(),
                }
    sens_df = pd.DataFrame(sens).T
    sens_df.index.names = ["fit", "maturity"]
    return pd.DataFrame(bias).T, pd.DataFrame(rmse).T, sens_df


def _fwd_5y5y(curve: NSSCurve) -> float:
    z5, z10 = curve.zero([5.0, 10.0])
    return float((10 * z10 - 5 * z5) / 5)


def main() -> None:
    print("## 1. Par quote minus zero rate, by curve shape\n")
    print(
        "compounding = quote − the same rate continuously compounded; "
        "coupon = the rest (the par yield averages zero rates over the coupon dates).\n"
    )
    print(anatomy().round(2).to_markdown())
    bias, rmse, sens = fitted_errors()
    print("\n## 2. Fitted zero curve vs the true curve (simulated par-quoted market, month-ends)\n")
    print("Average error (fitted − true, bp):\n")
    print(bias.round(2).to_markdown())
    print("\nRMSE (bp):\n")
    print(rmse.round(2).to_markdown())
    print(
        "\nError regressed on the true level² (%², average zero rate) and slope "
        "(10Y − 3M zero, points):\n"
    )
    print(sens.round(3).to_markdown())


if __name__ == "__main__":
    main()
