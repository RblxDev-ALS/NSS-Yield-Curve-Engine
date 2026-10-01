"""Breakeven inflation on a simulated market whose breakeven curve is known.

Each replication simulates weekly nominal Treasury par yields (11 tenors,
3 bp noise) from 1990 and TIPS real par yields from 2003 (5, 7, 10 and 20
years, plus 30 years from 2010, 3 bp noise), with a true zero-coupon
breakeven curve that moves through time (``simulate_tips_market``). The
nominal curve is fitted with the default NSS calibration; the real curve
with several candidate settings. Reported: the error of the 5- and 10-year
zero-coupon breakevens and of the 5y5y forward breakeven against the truth,
for each setting and for FRED's formulas applied to the same quotes
(``T5YIE = DGS5 − DFII5``, ``T10YIE``, ``T5YIFR``).

Usage::

    python benchmarks/breakeven_known_truth.py [--seeds 4] [--noise-bp 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from nss_engine.calibration import DEFAULT_PANEL_SMOOTHING, CalibrationConfig, calibrate_panel
from nss_engine.inflation import breakevens, fred_style_breakevens, real_curve_config
from nss_engine.models import DIEBOLD_LI_LAMBDA
from nss_engine.synthetic import simulate_market, simulate_tips_market

REAL_CONFIGS = {
    "NS, hump 3-30y, λ smoothing (default)": real_curve_config(),
    "NS, nominal λ bounds, λ smoothing": CalibrationConfig(
        model="ns", lambda_smoothing=DEFAULT_PANEL_SMOOTHING
    ),
    "NS, fixed Diebold-Li λ": CalibrationConfig(model="ns", fixed_lambda1=DIEBOLD_LI_LAMBDA),
    "NS, fixed λ = 0.3 (hump at 6y)": CalibrationConfig(model="ns", fixed_lambda1=0.3),
}
TARGETS = ("be_5y", "be_10y", "be_5y5y")


def replication(seed: int, noise_bp: float = 3.0) -> dict[str, dict[str, float]]:
    nominal = simulate_market(start="2003-01-03", periods=52 * 23, seed=seed, noise_bp=noise_bp)
    tips = simulate_tips_market(nominal, seed=seed, noise_bp=noise_bp)
    truth = tips.true_breakevens()
    nom_fit = calibrate_panel(nominal.yields)
    out: dict[str, dict[str, float]] = {}
    for name, cfg in REAL_CONFIGS.items():
        real_fit = calibrate_panel(tips.yields, cfg)
        be = breakevens(nom_fit.params, real_fit.params)
        out[name] = _errors(be, truth)
        out[name]["real curve fit RMSE (bp)"] = float(real_fit.diagnostics["rmse_bp"].mean())
    fred = fred_style_breakevens(nominal.yields, tips.yields)
    fred = fred.rename(columns={"T5YIE": "be_5y", "T10YIE": "be_10y", "T5YIFR": "be_5y5y"})
    out["FRED formulas on the quotes"] = _errors(fred, truth)
    return out


def _errors(est: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    both = est[list(TARGETS)].join(truth, rsuffix="_true", how="inner")
    res = {}
    for t in TARGETS:
        gap = (both[t] - both[f"{t}_true"]) * 100
        res[f"{t}: RMSE (bp)"] = float(np.sqrt(np.mean(gap**2)))
        res[f"{t}: mean gap (bp)"] = float(gap.mean())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--noise-bp", type=float, default=3.0, help="quote noise, nominal and real")
    args = ap.parse_args()
    t0 = time.perf_counter()
    reps = [replication(s, args.noise_bp) for s in range(args.seeds)]
    table = pd.concat({i: pd.DataFrame(r).T for i, r in enumerate(reps)}).groupby(level=1).mean()
    table = table.loc[[*REAL_CONFIGS, "FRED formulas on the quotes"]]
    table.index.name = "method"
    print(
        f"## Breakeven inflation vs the known truth ({args.seeds} simulated markets, "
        f"2003-2025, {args.noise_bp:g} bp quote noise)\n"
    )
    print("Errors against the true zero-coupon breakevens and the true 5y5y forward breakeven.\n")
    print(table.round(2).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
