# r/quant

**Title:** Fitting the Treasury curve since 1990 and testing every claim out of sample: par-yield bias, a survey-anchored term premium, and a random walk that still wins

**Body:**

I've been building an open-source yield curve engine (Python, MIT) and I'd
like feedback on the methods and on how I've tested them. Everything runs on
public data (FRED, the Fed's GSW curves, Kim–Wright, the Philadelphia Fed's
SPF, NBER) and a GitHub Action reruns it every weekday.

**Curve fitting.** Nelson–Siegel–Svensson by variable projection (closed-form
betas, exhaustive grid over the two decay rates, analytic-Jacobian refinement
from every basin). The main lesson: FRED's CMT yields are semi-annual *par*
yields, and fitting a zero curve straight to them is biased in a way the
in-sample fit cannot show. Against the Fed's own curve, fitting par yields
cuts the error from 16.3 to 10.0 bp (1–30y zeros, 440 month-ends); on a
simulated market with a known truth, by 59%.

**Term premium.** ACM (the NY Fed's model) on the engine's curves tracks ACM
on the Fed's curve at 0.99 correlation, but plain ACM re-estimated each month
on past data is unusable in real time (0.43–0.70 correlation with its own
full-sample series). Anchoring expected short rates to SPF bill-rate forecasts
fixes most of it:

| 10y premium | RMSE vs Kim–Wright | real time vs own full sample | known-truth real-time RMSE |
|---|---:|---:|---:|
| plain ACM | 123 bp | 0.43–0.70 | 82 bp |
| bias-corrected (analytic / bootstrap) | 113 / 117 bp | worse vs KW | 81 / 84 bp |
| survey-anchored | 28 bp | 0.91–0.94 | 13 bp |

Bias-corrected VARs (Bauer–Rudebusch–Wu) were a negative result: the
stationarity cap binds in 63–92% of real-time months.

**Recessions.** Probits scored in pseudo-real time with only recessions known
at each date. The near-term forward spread beats 10Y−3M (AUC 0.71 vs 0.61),
but the 90% block-bootstrap interval on the gain includes zero.
RECESSION_SPLIT_SENTENCE

**Forecasting.** Diebold–Li, Kalman-filter DNS and AFNS all lose to the random
walk at 1 month (DM p < 0.001). Half model, half random walk ties it at 6–12
months (p ≥ 0.49). AFNS's intervals are significantly too wide at every
horizon; I originally claimed the opposite and the coverage test corrected me.

**Breakevens (new).** TIPS real curve and 5y5y forward breakevens from fitted
zero curves. On simulated markets FRED's T5YIE/T5YIFR formulas are off by
2.6 bp even with perfect quotes; the curve-based numbers are within 1 bp.
BREAKEVEN_SENTENCE

Repo: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine
Live dashboard: https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/dashboard.html
Detailed results: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/results.md

What I'd most like critique on: the survey error model (one σ per SPF series,
floored at 0.1 pp), whether 24-month blocks are sensible for the AUC
bootstrap, and what else you'd validate the term premium against.

*Posting notes: r/quant is strict about self-promotion; post on a weekday,
lead with the methods question, answer comments with numbers, and link the
results page rather than the dashboard first.*
