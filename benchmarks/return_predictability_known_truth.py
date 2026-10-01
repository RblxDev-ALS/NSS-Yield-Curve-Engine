"""How much bond-return predictability can a 36-year sample reveal? A known-truth check.

Simulates arbitrage-free markets (:func:`nss_engine.synthetic.simulate_affine_market`)
of 440 months, the length of the FRED sample, with a near-unit-root level,
time-varying risk premia and SPF-style surveys. Because the true expected
excess returns are known, the study shows what out-of-sample R² even a perfect
estimate of the risk premium achieves on a sample this short, and how far the
real-time estimates fall below it:

* the true expected one-year excess return (no estimation at all);
* ACM's expected return, plain and survey-anchored, re-estimated each month
  on past data (first estimate after 60 months);
* Fama-Bliss and Cochrane-Piazzesi regressions, re-estimated each month.

All are scored against the expanding historical mean on the same forecast
origins, as in ``real_data_studies.py``. Run time is a few minutes per market.

Usage::

    python benchmarks/return_predictability_known_truth.py --markets 4
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from nss_engine import returns as bond_returns
from nss_engine.synthetic import simulate_affine_market


def one_market(seed: int, periods: int, horizon: int = 12) -> dict[tuple[str, str], dict]:
    mkt = simulate_affine_market(
        periods=periods, seed=seed, start="1990-01-31", level_persistence=0.99, noise_bp=2.0
    )
    zeros = mkt.yields
    surveys = mkt.surveys(seed=seed)
    mats = (24, 60, 120)
    rx = bond_returns.excess_returns(zeros, horizon, (24, 36, 48, 60, 120))
    cp = bond_returns.cochrane_piazzesi_forecasts(rx, bond_returns.forward_rates(zeros), horizon)
    true = mkt.expected_excess_returns(horizon, mats)
    plain = bond_returns.real_time_expected_returns(zeros, horizon, mats, min_train=60, n_factors=3)
    anchored = bond_returns.real_time_expected_returns(
        zeros, horizon, mats, min_train=60, n_factors=3, surveys=surveys
    )
    out = {}
    for n in mats:
        col = n / 12.0
        y = rx[col]
        bench = bond_returns.real_time_regression_forecasts(y, None, horizon)
        fcs = {
            "truth: true expected return": true[col],
            "ACM, plain (real time)": plain[col],
            "ACM, survey-anchored (real time)": anchored[col],
            "Fama-Bliss forward spread": bond_returns.real_time_regression_forecasts(
                y, bond_returns.forward_spot_spread(zeros, n, horizon).to_frame(), horizon
            ),
            "Cochrane-Piazzesi factor": cp[col],
        }
        common = pd.concat([y, bench, *fcs.values()], axis=1).dropna().index
        for name, f in fcs.items():
            sc = bond_returns.evaluate_return_forecasts(
                y.loc[common], f.loc[common], bench.loc[common], horizon
            )
            out[(f"{n // 12}Y", name)] = {
                "R² OOS (%)": 100 * sc.r2_oos,
                "CW p < 0.10": float(sc.p_value < 0.10),
                "slope": sc.mz_slope,
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--markets", type=int, default=4)
    ap.add_argument("--periods", type=int, default=440)
    args = ap.parse_args()
    t0 = time.perf_counter()
    runs = [pd.DataFrame(one_market(s, args.periods)).T for s in range(args.markets)]
    table = pd.concat(runs).groupby(level=[0, 1], sort=False).mean()
    table.index.names = ["bond", "forecast"]
    print(
        f"## Known truth: one-year excess returns, {args.markets} simulated markets of "
        f"{args.periods} months\n"
    )
    print(
        "Averages over markets: out-of-sample R² against the historical mean, the share of "
        "markets where the Clark-West test rejects at 10%, and the slope of realized on "
        "forecast returns (1 = calibrated).\n"
    )
    print(table.to_markdown(floatfmt=".2f"))
    print(f"\n({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
