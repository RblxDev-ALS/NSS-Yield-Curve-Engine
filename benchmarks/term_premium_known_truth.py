"""Term premium estimators on a simulated market whose term premium is known.

Each replication simulates 440 months (as long as the 1990-2026 FRED sample)
of an arbitrage-free three-factor market (:func:`simulate_affine_market`) with
a near-unit-root level, 1 bp of yield noise, and survey forecasts of the
3-month rate in the SPF's format: four quarterly horizons every quarter and the
ten-year average once a year. The ACM model is then estimated with its
real-world dynamics from

* OLS (the original ACM estimator),
* OLS with the small-sample bias removed (Pope's formula, or Bauer, Rudebusch
  & Wu's inverse bootstrap),
* the factors *and* the surveys (Kim & Orphanides style anchoring), with
  unbiased surveys, and with surveys that are 0.5 points too high throughout,

both on the full sample and in pseudo real time (re-estimated every month
from month 120 on, using only data and surveys published by then). Reported:
the 10-year term premium's RMSE against the truth, and how closely the
real-time series tracks the same estimator's full-sample series.

Usage::

    python benchmarks/term_premium_known_truth.py [--seeds 6] [--jobs 4]
"""

from __future__ import annotations

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from nss_engine.synthetic import simulate_affine_market
from nss_engine.termpremium import fit_acm, real_time_decomposition

WINDOWS = {"TBILL3": (2, 4), "TBILL4": (5, 7), "TBILL5": (8, 10), "TBILL6": (11, 13)}
WINDOWS["BILL10"] = (1, 120)


def replication(seed: int, periods: int = 440, min_train: int = 120) -> dict[str, dict[str, float]]:
    warnings.simplefilter("ignore")
    mkt = simulate_affine_market(
        periods=periods, seed=seed, noise_bp=1.0, level_persistence=0.99, start="1990-01-31"
    )
    methods: dict[str, dict[str, Any]] = {
        "OLS (ACM)": {},
        "bias-corrected, analytic": {"bias_correction": "analytic"},
        "bias-corrected, bootstrap (BRW)": {"bias_correction": "bootstrap"},
        "survey-anchored": {"surveys": mkt.surveys(WINDOWS, noise_pp=0.15, seed=seed)},
        "survey-anchored, surveys +0.5 pp": {
            "surveys": mkt.surveys(WINDOWS, noise_pp=0.15, bias_pp=0.5, seed=seed)
        },
    }
    truth = mkt.term_premium.iloc[:, -1]
    out = {}
    for name, kwargs in methods.items():
        full = fit_acm(mkt.yields, n_factors=3, **kwargs).term_premium.iloc[:, -1]
        rt = real_time_decomposition(mkt.yields, min_train=min_train, n_factors=3, **kwargs)
        tp = rt["term_premium"].dropna()
        out[name] = {
            "full: RMSE vs truth (bp)": _rmse(full, truth),
            "real time: RMSE vs truth (bp)": _rmse(tp, truth.reindex(tp.index)),
            "real time: RMSE vs own full sample (bp)": _rmse(tp, full.reindex(tp.index)),
            "real time: corr with own full sample": float(tp.corr(full.reindex(tp.index))),
            "real time: share discarded": float(rt["term_premium"].isna().mean()),
        }
    return out


def _rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()) * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(replication, range(args.seeds)))
    frames = [pd.DataFrame(r).T for r in results]
    mean = pd.concat(frames).groupby(level=0, sort=False).mean()
    worst = pd.concat(frames).groupby(level=0, sort=False)["real time: RMSE vs truth (bp)"].max()
    mean["real time: worst replication (bp)"] = worst
    mean.index.name = "real-world dynamics"
    print(
        f"## Known-truth term premium study ({args.seeds} replications, 440 months, "
        "level persistence 0.99)\n"
    )
    print("Averages over replications; 10-year premium.\n")
    print(mean.T.round(2).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
