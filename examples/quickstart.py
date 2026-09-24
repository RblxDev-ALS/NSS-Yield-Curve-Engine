"""Minimal library usage: fit one curve, then analyse it.

Run with ``python examples/quickstart.py`` (downloads a few days of FRED data)
or ``python examples/quickstart.py --synthetic`` (offline).
"""

import sys

import numpy as np

from nss_engine import CalibrationConfig, calibrate
from nss_engine.analytics import Bond, carry_rolldown, risk_report
from nss_engine.data import load_treasury_yields, maturity_label
from nss_engine.synthetic import simulate_market

if "--synthetic" in sys.argv:
    yields = simulate_market(periods=10).yields
else:
    import pandas as pd

    yields = load_treasury_yields(start=pd.Timestamp.now() - pd.Timedelta(days=14), freq=None)

date, quotes = yields.index[-1], yields.iloc[-1]
fit = calibrate(np.asarray(quotes.index, dtype=float), quotes.to_numpy(), CalibrationConfig())
curve = fit.curve
print(f"{date:%Y-%m-%d}: RMSE {fit.rmse_bp:.2f} bp")
print(f"  level β0 = {curve.beta0:.2f}%, short rate = {curve.short_rate:.2f}%")
print(f"  10y-2y = {curve.spread(10, 2):+.2f} pp, 10y-3m = {curve.spread(10, 0.25):+.2f} pp")

print("\nForward rates:")
for t in (1, 2, 5, 10, 20):
    print(f"  {maturity_label(t):>4}: {curve.forward(t)[0]:.2f}%")

print("\nCarry & roll-down over 3 months:")
print(carry_rolldown(curve, [2, 5, 10, 30], horizon=0.25).round(1).to_string())

rep = risk_report(curve, Bond.par(curve, 10.0))
print(
    f"\n10y par bond: duration {rep.duration:.2f}, DV01 {rep.dv01:.4f}, convexity {rep.convexity:.1f}"
)
print("Factor durations:\n" + rep.factor_durations.round(3).to_string())
