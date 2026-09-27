# r/econometrics

**Title:** How much of the 10-year Treasury yield is term premium? Anchoring ACM to survey forecasts, with real-time and known-truth tests

**Body:**

I maintain an open-source Python engine for the U.S. Treasury curve and the
latest version is mostly an econometrics exercise, so I'd value this sub's
view on the identification and the tests.

**The problem.** In an affine model like Adrian–Crump–Moench, the
cross-section of excess returns pins down the risk-neutral dynamics well (on
a simulated arbitrage-free market, exactly). The term premium depends on the
real-world VAR: where rates revert to and how fast. OLS on 1990–2026 data
learns that rates revert to their falling sample mean, so plain ACM puts the
10-year premium about a point above the Fed Board's Kim–Wright on average, and
re-estimated month by month it is very unstable (correlation 0.43 with its own
full-sample series after a 5-year start).

**Two fixes, same risk-neutral dynamics (so fitted yields do not change):**

1. Small-sample bias correction of the VAR: Pope's analytic formula and BRW's
   inverse bootstrap, shrunk to stationarity (Kilian).
2. Survey anchors: the joint likelihood of the VAR and the SPF's 3-month bill
   forecasts (1–4 quarters, 1–3 calendar years and 10 years ahead), each
   forecast's model value being the average expected short rate over its
   window, as in Kim & Orphanides (2012).

**Results.**

* Full sample vs Kim–Wright: RMSE 123 bp (plain), 113 / 117 bp
  (bias-corrected), **28 bp (survey-anchored)**. Correlation of 12-month
  changes 0.75, 0.46 / 0.37, **0.82**.
* Model-free check: 10-year zero yield minus the SPF's own 10-year bill
  forecast (35 first quarters, 1992–2026). Correlation with it: 0.89
  survey-anchored, 0.83 Kim–Wright, 0.80 plain ACM.
* Real time (first estimate after 5 / 10 / 15 years): survey-anchored
  correlates 0.94 / 0.91 / 0.92 with its own full-sample series, plain ACM
  0.43 / 0.58 / 0.70.
* Known truth, 8 simulated 440-month markets with level persistence 0.99:
  real-time RMSE 82 bp plain, 81 / 84 bp bias-corrected, **13 bp** with
  surveys, 53 bp with surveys biased by +0.5 pp throughout.
* Bias correction in real time is a negative result: the stationarity cap
  binds in 63–92% of months.

**Does it matter for recessions?** Rosenberg & Maurer (2008) found the
expectations component, not the premium, carries the spread's recession
signal. RECESSION_SPLIT_SENTENCE

**Forecast tests.** Diebold–Mariano (HLN-corrected, loss pooled over tenors)
and Newey–West tests of 80% interval coverage: the random walk beats every
model at 1 month, and the arbitrage-free model's intervals are significantly
too wide at every horizon (93 / 89 / 86% coverage).

Methodology with the equations:
https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/methodology.md
Results: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/results.md

Questions I'm unsure about: is a per-series survey error (estimated, floored
at 0.1 pp) reasonable, or should the 10-year forecast get a larger error? And
is the model-free survey premium a fair external check given that Kim–Wright
also uses surveys?
