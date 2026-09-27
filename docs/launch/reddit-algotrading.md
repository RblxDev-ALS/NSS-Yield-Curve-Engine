# r/algotrading

**Title:** I tried to beat a random walk at forecasting Treasury yields with three generations of Nelson–Siegel models. It still wins. Here's what the yield curve is useful for instead.

**Body:**

Open-source Python project, all public data, all tests out of sample. Up
front: **this is not a trading system**, and the headline result is negative.

**What does not work: forecasting yields.** Diebold–Li (2006), its
Kalman-filter state-space version, and the arbitrage-free version (AFNS),
each re-estimated every month on past data only, 1990–2026:

* Every model loses to "no change" at 1 month, significantly
  (Diebold–Mariano p < 0.001).
* At 6 and 12 months the losses are not significant, and averaging the model
  with the random walk only ties it (RMSE ratios 0.976–0.991, p ≥ 0.49).
* The arbitrage-free model's 80% intervals cover 86–93% of outcomes, i.e.
  they are too wide.

**What the curve is useful for:**

* **Measuring**, not predicting: a clean weekly zero/forward curve (10 bp from
  the Fed's own curve), carry and roll-down, key-rate and factor durations,
  rich/cheap residuals with rolling z-scores that use no future data.
* **Decomposing** the 10-year yield into expected short rates and a term
  premium; the survey-anchored version is stable in real time (0.91–0.94
  correlation with its full-sample estimate).
* **Recession risk**, with humility: the near-term forward spread had an
  out-of-sample AUC of 0.71 vs 0.61 for 10Y−3M, but with three recessions in
  the test window the difference is not statistically clear.
  Splitting the spread Rosenberg–Maurer style into expectations and term premium, both estimated in real time with survey anchors, was a negative result: from 2005 the expectations component was significantly *worse* than the plain spread (out-of-sample AUC 0.13 vs 0.45), and nothing beat a coin flip in that window (two recessions plus the 2022–24 inversion).
* **Breakeven inflation** from fitted TIPS and nominal zero curves.
  On real data, against the Fed's own TIPS curve, the engine's 5y5y breakeven is 20 bp RMSE away vs 27 bp for FRED's T5YIFR (which is 9 bp too low on average); at 5 years FRED's series is slightly closer (12 vs 14 bp).

**Caveats for anyone thinking of signals:** constant-maturity yields are
interpolated on-the-run par yields, not executable prices; residuals are
curve-shape signals; four recessions since 1990 are a tiny sample; and
anything published by FRED arrives with a lag.

Repo: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine
Dashboard (updated every weekday): https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/dashboard.html

Happy to hear about approaches that *have* beaten the random walk out of
sample, with the test that shows it.
