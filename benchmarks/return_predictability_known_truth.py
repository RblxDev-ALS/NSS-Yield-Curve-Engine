"""How much bond-return predictability can a 36-year sample reveal? A known-truth check.

Simulates arbitrage-free markets (:func:`nss_engine.synthetic.simulate_affine_market`)
of 440 months, the length of the FRED sample, with a near-unit-root level,
time-varying risk premia and SPF-style surveys. Because the true expected
excess returns are known, the study shows what out-of-sample R² even a perfect
estimate of the risk premium achieves on a sample this short, and how far the
real-time estimates fall below it:

* the true expected one-year excess return (no estimation at all);
* ACM's expected return, plain and survey-anchored, re-estimated each month
  on past data (first estimate after 60 months), and anchored to surveys that
  expect rates half a point higher than the truth;
* Fama-Bliss and Cochrane-Piazzesi regressions, re-estimated each month.

All are scored against the expanding historical mean on the same forecast
origins, as in ``real_data_studies.py``. Run time is a few minutes per market.

Usage::

    python benchmarks/return_predictability_known_truth.py --markets 4 --jobs 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

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
    # forecasters who expect rates 0.5 points higher than the truth, as the SPF did in 2000-2025
    biased = bond_returns.real_time_expected_returns(
        zeros, horizon, mats, min_train=60, n_factors=3, surveys=mkt.surveys(seed=seed, bias_pp=0.5)
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
            "ACM, anchored to surveys biased +0.5 pp": biased[col],
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
    ap.add_argument("--jobs", type=int, default=1, help="markets simulated in parallel")
    args = ap.parse_args()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        results = pool.map(one_market, range(args.markets), [args.periods] * args.markets)
        runs = [pd.DataFrame(r).T for r in results]
    grouped = pd.concat(runs).groupby(level=[0, 1], sort=False)
    table = grouped.mean()
    table.insert(1, "R² OOS > 0", grouped["R² OOS (%)"].apply(lambda r: float((r > 0).mean())))
    table.insert(1, "sd across markets", grouped["R² OOS (%)"].std())
    table.index.names = ["bond", "forecast"]
    print(
        f"## Known truth: one-year excess returns, {args.markets} simulated markets of "
        f"{args.periods} months\n"
    )
    print(
        "Averages over markets: out-of-sample R² against the historical mean (with its "
        "standard deviation across markets and the share of markets where it is positive), "
        "the share of markets where the Clark-West test rejects at 10%, and the slope of "
        "realized on forecast returns (1 = calibrated).\n"
    )
    print(table.to_markdown(floatfmt=".2f"))
    print(f"\n({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
