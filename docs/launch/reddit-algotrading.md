# r/algotrading

**Title:** I tried to beat a random walk at forecasting Treasury yields with three Nelson–Siegel models. The random walk won.

**Body:**

Open-source Python project, public data, everything tested out of sample.
Not a trading system, and the main result is negative.

I re-estimated Diebold–Li, its Kalman-filter version and the arbitrage-free
version (AFNS) every month on past data only, 1990–2026. All three lost to
"yields don't change" at one month (Diebold–Mariano p < 0.001). At 6 and 12
months a 50/50 mix of model and random walk only ties it (p ≥ 0.49). The
arbitrage-free model's 80% intervals turned out too wide, covering 86–93%
of outcomes.

What the curve was useful for instead: measuring things. A weekly zero and
forward curve within 10 bp of the Fed's own, carry and roll-down, key-rate
durations, a term premium that is stable in real time once it's anchored
to surveys, and breakevens. The recession signal from the near-term forward
spread looked better than 10Y−3M out of sample (AUC 0.71 vs 0.61), but with
three recessions in the test window that difference isn't statistically
clear.

If you're thinking about signals: CMT yields are interpolated on-the-run par
yields, not executable prices, and FRED data arrives with a lag.

Code: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine
Dashboard, updated every weekday: https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/dashboard.html

If you've seen something beat the random walk on Treasury yields out of
sample, I'd like to know what, and how it was tested.
