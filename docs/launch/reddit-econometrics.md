# r/econometrics

**Title:** How much of the 10-year Treasury yield is term premium? Anchoring ACM to survey forecasts, tested in real time

**Body:**

I've been estimating the 10-year term premium with Adrian–Crump–Moench on
my own fitted Treasury curves, and I'd like this sub's view on the
identification and the tests.

The problem: OLS on 1990–2026 data learns that rates revert to a falling
sample mean, so plain ACM puts the premium about a point above the Fed
Board's Kim–Wright estimate, and re-estimated month by month it jumps
around (0.43 correlation with its own full-sample series after a 5-year
start).

What I tried, keeping the risk-neutral dynamics fixed so fitted yields don't
change:

1. Small-sample bias correction of the VAR (Pope's formula and the
   Bauer–Rudebusch–Wu bootstrap). It barely moved the level and, in real
   time, the stationarity cap binds in 63–92% of months.
2. Survey anchors, as in Kim & Orphanides (2012): the VAR is estimated
   jointly with SPF 3-month bill forecasts, each matched to the model's
   average expected short rate over its window.

The surveys did most of the work: 28 bp RMSE against Kim–Wright (123 bp
plain), 0.91–0.94 real-time agreement with the full-sample series, and
correlation 0.89 with a model-free premium (the 10-year yield minus the
SPF's own 10-year bill forecast). On simulated markets with a known premium
the real-time error fell from 82 to 13 bp.

One negative result: splitting the 10Y−3M spread into the expectations part
and the premium (Rosenberg & Maurer) did not help predict recessions. From
2005 the expectations part did worse than the plain spread out of sample,
though that window only has two recessions.

Two questions: is a per-series survey error (estimated, floored at 0.1 pp)
reasonable, or should the 10-year forecast get a larger one? And is the
model-free survey premium a fair check, given that Kim–Wright also uses
surveys?

Methodology with the equations: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/methodology.md
Results: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/results.md
