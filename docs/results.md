# Results

Every real-data number on this page comes from the
[live workflow](https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/actions/workflows/live-dashboard.yml),
which reruns the pipeline and `benchmarks/real_data_studies.py` on FRED data
on every push and every weekday. Simulated-market numbers come from the
benchmark scripts named in each section and can be rerun offline. Negative
results are reported next to the positive ones.

Contents: [fitting the curve](#fitting-the-curve) ·
[term premium](#term-premium) · [recessions](#recessions) ·
[breakeven inflation](#breakeven-inflation) · [forecasting](#forecasting) ·
[other findings](#other-findings)

## Fitting the curve

1,917 weekly curves, January 1990 to 24 September 2026:

| | |
|---|---|
| Median fit error | **3.8 bp** (95th percentile 8.0 bp), 100% of fits converged |
| Agreement with the **Federal Reserve's own curve** (1–30Y zero rates) | RMSE **10.0 bp** with par fitting vs 16.3 bp reading quotes as zero rates. Weekly changes correlate 0.97 |
| Calibration speed | ~26 ms per curve (par fit), so 36 years of weekly curves take about 70 s |
| Model-implied vs observed 10Y−3M spread | correlation **0.999**, mean absolute gap 5 bp |
| Latest reading (24 September 2026) | 10Y zero yield 5.18% = **4.02%** expected short rate + **1.16%** term premium (survey-anchored; plain ACM 1.91%). 10Y−3M +1.04 pp; recession odds 8% (NY Fed-style probit) |

### The quotes are par yields

FRED's constant-maturity yields are **semi-annual par yields**. Version 1.x,
like Diebold & Li's classic setup, fitted the continuously compounded *zero*
curve straight to them. The in-sample fit cannot reveal the error: both fits
match the quotes equally well. The zero curve that comes out is still wrong.
Checked against the Federal Reserve's independently estimated Svensson curve
(Gürkaynak, Sack & Wright; off-the-run bonds, a different method), over 440
month-ends since 1990:

| reading of the quotes | RMSE vs the Fed | after removing each maturity's average gap | average gap at 20Y | 5y5y forward gap |
|---|---:|---:|---:|---:|
| as zero rates (1.x) | 16.3 bp | 12.0 bp | −18.5 bp | −21.8 bp |
| **as par yields (2.0 default)** | **10.0 bp** | **7.8 bp** | −9.7 bp | −14.5 bp |
| as par yields, robust fit | 11.0 bp | 8.7 bp | −10.8 bp | −15.1 bp |

The par fit is **38% closer** to the Fed's curve. On a simulated market with a
known truth the same correction cuts the error 59% (below). What remains is
mostly a stable offset that is expected: the engine fits **on-the-run** issues,
which trade rich, while the Fed deliberately excludes them. The research note
[Par yields are not zero rates](par-vs-zero.md) works through the bias in
detail.

The third row is a negative result. Robust fitting (Huber/bisquare
reweighting) works on simulated bad quotes, where it cuts the error from 11.5
to 3.1 bp. On real data it mostly rejects the **20-year bond** (13.6% of weeks)
and the on-the-run 10-year (4.7%). Those are persistent market features, not
errors, and ignoring them moves the curve *away* from the Fed's. So it is
opt-in (`--robust`).

### Does NSS overfit? Leave-one-tenor-out cross-validation

In-sample error always favours the model with more parameters. The honest test
is out of sample: hide one maturity, fit the rest, and predict the hidden
yield. 441 month-end curves, 1990–2026:

| model | out-of-sample RMSE, 3M–20Y (interpolation) | 30Y (extrapolation) |
|---|---:|---:|
| Nelson–Siegel (4 parameters) | 9.31 bp | **17.2 bp** |
| NSS, each date fitted independently | 8.69 bp | 23.9 bp |
| **NSS + λ smoothing (default)** | **8.34 bp** | 23.4 bp |
| NSS + λ smoothing, zero target (1.x) | 8.31 bp | 29.2 bp |

NSS's extra parameters help **between** quoted maturities (10% lower
out-of-sample error than Nelson–Siegel), and the smoothing penalty helps a
little more. **Beyond** the last quote the extra flexibility hurts: with the
30Y hidden, NSS bends the long end more than the data supports, so for
extrapolation the simpler model is safer. Par fitting cuts that 30Y
extrapolation error from 29.2 to 23.4 bp. Keeping the second hump inside the
quoted maturities (2.1) moves it only from 23.36 to 23.30 bp.

### 2.0 vs 1.x vs the original algorithm, against a known truth

`benchmarks/compare_legacy.py` reruns the original (v0) calibration routine
unchanged and compares it with 1.x and 2.0 on a synthetic market where the
**true** curves are known and the quotes are par yields, like FRED's: 3 seeds ×
10 years of weekly curves, 3 bp of quote noise.

| method | fit RMSE | error vs **true** curve | worst 1% error vs truth |
|---|---:|---:|---:|
| v0: 6-D L-BFGS-B + ridge 0.005 (original) | 5.57 bp | 8.01 bp | 15.0 bp |
| v0 without the ridge penalty | 2.56 bp | 6.17 bp | 12.7 bp |
| 1.x: variable projection + λ smoothing, zero target | 2.15 bp | 5.74 bp | 12.4 bp |
| 2.0: par target, each week independent | 1.98 bp | 2.58 bp | 5.8 bp |
| **2.0: par target + λ smoothing (default)** | 2.15 bp | **2.37 bp** | **5.3 bp** |

2.0's curves are 59% closer to the truth than 1.x's and 70% closer than v0's,
with worst cases more than halved, at the same in-sample fit. v0's penalty was
larger than the fitting error itself, so it flattened genuine curvature. The
smoothing penalty *raises* in-sample RMSE slightly but *lowers* the error
against the truth: the extra in-sample fit was fitting noise.

## Term premium

A 10-year yield is the average short rate investors expect over ten years plus
a **term premium**. The engine splits them with the regression-based model of
**Adrian, Crump & Moench (2013)**, the method behind the New York Fed's
published term premium, run on its own zero curves (1–120 months, month-end,
five principal components).

**The engine's curves are good enough for term-structure modelling.** Run on
this engine's curves, the model tracks the same model run on the Fed's GSW
curve with correlation 0.99 (12-month changes 0.95), RMSE 31 bp: far less
than the disagreement between models.

### Where plain ACM goes wrong, and the fix

ACM learns where rates revert to from the sample average, which fell for most
of 1990–2020. So it attributes too much of the 1990s' high yields to the
premium: over 1990–2026 it averages 1.83%, a point above the Fed Board's
Kim–Wright estimate. Re-estimated each month on past data only, it is worse: a
premium that jumps whenever the model is re-estimated.

2.3 tried two fixes, both keeping ACM's pricing (the fitted yields do not
change, only the split): correcting the small-sample bias of the VAR (Bauer,
Rudebusch & Wu, 2012), and anchoring expected short rates to the Philadelphia
Fed's **Survey of Professional Forecasters** (Kim & Orphanides, 2012). In 2.4
the survey-anchored estimate is the headline; plain ACM is shown next to it.

| 10-year premium, full sample 1990–2026 | mean | corr. with Kim–Wright | corr. of 12-month changes | RMSE vs KW | mean gap vs KW |
|---|---:|---:|---:|---:|---:|
| plain ACM | 1.83% | 0.96 | 0.75 | 123 bp | +101 bp |
| bias-corrected, analytic | | | 0.46 | 113 bp | |
| bias-corrected, bootstrap | | | 0.37 | 117 bp | |
| **survey-anchored ACM** | **0.76%** | **0.95** | **0.82** | **28 bp** | **−7 bp** |

Bias correction brings the level a little closer to Kim–Wright but makes the
year-on-year changes much worse. Surveys fix both. They are fitted to 0.17 pp
RMSE one quarter ahead, 0.47 pp three years ahead and 0.41 pp for the
ten-year average.

**A model-free check.** Kim–Wright also uses surveys, so agreeing with it
could be circular. The SPF asks each first quarter for the average bill rate
over the next ten years; the 10-year zero yield minus that forecast is a term
premium with no model at all (35 first quarters, 1992–2026):

| estimate | corr. with the survey premium | mean gap |
|---|---:|---:|
| **survey-anchored ACM** | **0.89** | +11 bp |
| Kim–Wright | 0.83 | +22 bp |
| plain ACM | 0.80 | +118 bp |

### In real time

A full-sample estimate splits 1995's yields using what rates did through
2026. The real test is to re-estimate every month on data (and surveys)
published by then:

| real-time estimate, first after | 5 years | 10 years | 15 years |
|---|---:|---:|---:|
| corr. with its own full-sample series: plain ACM | 0.43 | 0.58 | 0.70 |
| … survey-anchored | **0.94** | **0.91** | **0.92** |
| corr. with Kim–Wright: plain ACM | 0.24 | 0.38 | 0.49 |
| … survey-anchored | **0.94** | **0.91** | **0.85** |
| RMSE vs Kim–Wright: plain ACM | 101–112 bp | | |
| … survey-anchored | 45–50 bp | | |

Plain ACM needs decades of data to learn where rates revert to; the surveys
tell it at every date. **Bias correction in real time is a negative result**:
it raises the correlation with its own full-sample series (0.58–0.92) but
lowers the agreement with Kim–Wright (0.00–0.45), and the stationarity cap
binds in 63–92% of months, so the corrected model is mostly the cap.

**Checked against a known truth.** `benchmarks/term_premium_known_truth.py`
simulates 8 arbitrage-free markets of 440 months with a near-unit-root level
(0.99) and SPF-style surveys:

| real-world dynamics | real-time RMSE vs the true premium | corr. of real time with own full sample |
|---|---:|---:|
| plain ACM (OLS) | 82 bp | 0.66 |
| bias-corrected, analytic | 81 bp | |
| bias-corrected, bootstrap | 84 bp | |
| **survey-anchored** | **13 bp** | **0.97** |
| survey-anchored, surveys biased +0.5 pp | 53 bp | |

The same ranking as on real data, and even surveys that are half a point too
high throughout beat no surveys.

## Recessions

Every U.S. recession since the late 1960s was preceded by an inverted curve.
Inversions of at least three months (10Y−3M) in the sample: 2000, 2006 and 2019,
each followed by a recession after **8, 18 and 10 months**; and 2022–24, the
deepest inversion in the sample (−1.8 pp, 25 months), with no recession so far.

In sample, the 10Y−3M probit has an AUC of 0.78. That flatters it: the model is
fitted on the same recessions it is scored on. The engine also runs a
**pseudo-real-time** test: each month it re-estimates the probit using only
recessions already known, then scores the forecast it would have made (297
forecast months):

| signal | AUC in sample | AUC out of sample | log score (higher = better) | Brier |
|---|---:|---:|---:|---:|
| 10Y−3M spread (NY Fed) | 0.779 | 0.613 | −0.395 | 0.088 |
| **near-term forward spread** | 0.774 | **0.712** | **−0.302** | 0.091 |
| both | 0.786 | 0.503 | −0.543 | 0.103 |

The **near-term forward spread** (Engstrom & Sharpe, 2019) is read straight off
the NSS forward curve: the 3-month rate expected 18 months ahead minus today's.
It measures whether markets expect the Fed to *cut*. Out of sample it ranks
recession risk better than the classic spread, but with only three recessions
in the forecast window the 90% block-bootstrap interval on the gain (24-month
blocks) includes zero:

| vs 10Y−3M, out of sample | AUC gain | 90% interval |
|---|---:|---:|
| near-term forward spread | +0.099 | [−0.035, +0.205] |
| both signals together | −0.110 | [−0.167, +0.008] |

Putting both signals in one model looks best in sample and is the worst out of
sample, the classic sign of overfitting with too few recessions.

### Expectations or term premium? A negative result

Rosenberg & Maurer (2008) found that the recession signal of the spread comes
from its expectations component (the spread minus the term premium), not from
the premium. 2.2 could not test this properly because plain ACM's real-time
premium was too noisy. With the survey-anchored premium, re-estimated each
month on data and surveys published by then, the split is finally stable, so
2.4 reruns the test. Month-end 10Y−3M spread, probits 12 months ahead, scored
in pseudo-real time on the 238 months (from 2005) where every signal has a
real-time value:

| signal | AUC out of sample | gain vs the spread, 90% interval |
|---|---:|---:|
| 10Y−3M spread | 0.45 | |
| expectations component, plain ACM | 0.41 | −0.05 [−0.09, +0.01] |
| term premium, plain ACM | 0.41 | −0.05 [−0.36, +0.28] |
| expectations component, **survey-anchored** | **0.13** | **−0.32 [−0.54, −0.02]** |
| term premium, survey-anchored | 0.46 | +0.00 [−0.61, +0.37] |
| both parts, survey-anchored | 0.28 | −0.18 [−0.27, +0.01] |

**Rosenberg & Maurer's result does not hold in this sample.** No curve signal
beats a coin flip from 2005 on, and the survey-anchored expectations component
is *significantly worse* than the spread it comes from. Waiting a year for NBER
to date each recession changes nothing (spread 0.50, expectations component
0.16, interval [−0.59, −0.01]; 226 months). The window holds only two
recessions, one caused by a pandemic, and the 2022–24 inversion, when markets
and forecasters expected rate cuts that came without a recession. That is the
most likely reason the expectations component misfires, and also why this
result says little about longer samples.

## Breakeven inflation

New in 2.4: the TIPS real curve is fitted like the nominal one (Nelson–Siegel
on the 4–5 TIPS par yields), and breakeven inflation is read off the two zero
curves; see [methodology §7](methodology.md#7-real-yields-and-breakeven-inflation).

**Checked against a known truth first** (`benchmarks/breakeven_known_truth.py`,
4 simulated markets, weekly 2003–2025, RMSE against the true zero-coupon
breakevens):

| method | 5Y breakeven | 10Y breakeven | 5y5y forward |
|---|---:|---:|---:|
| engine, 3 bp quote noise | **3.1 bp** | **3.6 bp** | **7.8 bp** |
| FRED's formulas on the same quotes | 5.4 bp | 5.3 bp | 10.2 bp |
| engine, no noise | **0.5 bp** | **0.3 bp** | **0.6 bp** |
| FRED's formulas, no noise | 3.2 bp | 3.1 bp | 3.5 bp |

FRED's `T5YIE`/`T10YIE` are differences of par yields and `T5YIFR` compounds
them as if they were zero rates; on noiseless quotes that alone is off by
2.6 bp at 5 years. The engine's error without noise is under 1 bp.

**On real data** (1,157 weekly curves, July 2004 to September 2026; FRED's TIPS
series only have four maturities from mid-2004): the real curve fits the TIPS
quotes to 1.15 bp (median; 95th percentile 5.3 bp). The Fed publishes its own
TIPS curve (Gürkaynak, Sack & Wright, 2010), so its zero-coupon breakevens are
an independent benchmark for both the engine and FRED's series (267 month-ends):

| vs the Fed's zero-coupon breakevens | corr. | RMSE | mean gap |
|---|---:|---:|---:|
| engine, 5Y | 0.975 | 13.8 bp | −1.5 bp |
| FRED `T5YIE` | 0.980 | 12.2 bp | −0.9 bp |
| **engine, 10Y** | **0.980** | **8.2 bp** | −1.0 bp |
| FRED `T10YIE` | 0.974 | 10.6 bp | −5.0 bp |
| **engine, 5y5y forward** | **0.885** | **20.4 bp** | **−0.5 bp** |
| FRED `T5YIFR` | 0.809 | 26.9 bp | −9.1 bp |

The engine is closer to the Fed at 10 years and for the 5y5y forward, where
FRED's shortcut is 9 bp too low on average; FRED's 5-year series is slightly
closer at 5 years. As a check of the machinery, the engine's *par* breakevens
reproduce FRED's `T5YIE` and `T10YIE` to 5.6 and 6.5 bp (correlation 0.996 and
0.988). In the sample the 5y5y breakeven ranged from 0.78% (December 2008) to
3.15% (April 2011); on 24 September 2026 it was 2.35%, with a 10-year real
yield of 2.85% and a 10-year breakeven of 2.33%.

## Forecasting

Diebold & Li's model beat "no change" at 12-month horizons in their original
1985–2000 sample, but not since. A model that mean-reverts to a historical
average struggles through decades of falling rates and the zero lower bound,
and the random walk is famously hard to beat (Duffee, 2002). Every model below
is re-estimated on past data only:

| model | 1 month | 6 months | 12 months | 80% interval coverage (1 / 6 / 12m) |
|---|---:|---:|---:|---:|
| Diebold–Li, AR(1) factors | 1.112 | 1.072 | 1.090 | – |
| Diebold–Li, VAR(1) factors | 1.089 | 1.018 | 1.028 | – |
| state-space, VAR(1) | 1.088 | 1.029 | 1.048 | 86% / 79% / 73% |
| state-space, random-walk level | 1.103 | 1.062 | 1.073 | 85% / 79% / 77% |
| **AFNS** (arbitrage-free), VAR(1) | 1.073 | 1.040 | 1.049 | 93% / 89% / 86% |
| AFNS, independent factors | 1.130 | 1.093 | 1.106 | 83% / 71% / 59% |
| ½ state-space VAR(1) + ½ random walk | **1.011** | **0.990** | **0.996** | – |
| ½ Diebold–Li VAR(1) + ½ random walk | 1.013 | 0.994 | 1.002 | – |
| ½ AFNS VAR(1) + ½ random walk | 1.008 | 0.997 | 1.000 | – |

(RMSE relative to the random walk, averaged over tenors; < 1 beats it.) The
state-space model is the Kalman-filter version (Diebold, Rudebusch & Aruoba,
2006); AFNS adds no-arbitrage (Christensen, Diebold & Rudebusch, 2011).

### Which differences are real? Diebold–Mariano tests

2.3 tests the differences, with the loss pooled over tenors at each forecast
origin (265–276 origins; Harvey–Leybourne–Newbold small-sample correction):

* **AFNS beats the state-space VAR at 1 month**: RMSE ratio 0.981, DM −2.47,
  p = 0.014. At 6 and 12 months the difference is not significant (p = 0.63 and
  0.79).
* **Both models lose to the random walk at 1 month**, significantly
  (p < 0.001). At 6 and 12 months their losses are not significant.
* **½ model + ½ random walk vs the random walk**: pooled RMSE ratios 0.976–0.991
  at 6 and 12 months, all p ≥ 0.49. **A tie, not a win.** At 1 month the
  state-space combination is significantly worse than the random walk
  (p = 0.034).

The practical lesson is the usual one: shrink model forecasts towards "no
change", and do not expect to beat it.

### Are the intervals calibrated? Newey–West coverage tests

The share of outcomes inside the 80% intervals, tested against 80% with
Newey–West standard errors (the monthly series overlap):

| model | 1 month | 6 months | 12 months |
|---|---:|---:|---:|
| state-space VAR(1) | 86% (too wide, p = 0.009) | 79% (p = 0.80) | 73% (p = 0.21) |
| AFNS VAR(1) | 93% (too wide) | 89% (too wide) | 86% (too wide, p = 0.044) |
| AFNS minus state-space | +7 points | +10 points | +14 points |

All AFNS-minus-state-space gaps have p ≤ 0.001. So AFNS's intervals are
**significantly too wide at every horizon**, while the unrestricted
state-space model's are too wide at 1 month and consistent with 80% at 6 and
12 months. (2.2 read AFNS's 86% at 12 months as "better calibrated" than 73%;
the test says the reverse: 73% is within sampling error of 80%, 86% is not.)
CDR found that *independent* factors forecast best; here they are the worst
specification at every horizon, with or without the restriction.

## Other findings

* **Raw NSS betas are not clean economic factors.** With the decay rates free,
  −β1 is a zero-to-infinity spread and correlates only 0.74 with the observed
  10Y−3M. Fixed-λ Diebold–Li factors correlate 0.995. So the regime engine
  reads the slope off the fitted curve (0.999) instead of using −β1.
* **The 20-year bond anomaly.** Since it was reintroduced in 2020, the 20-year
  has traded cheap relative to its neighbours (+9 bp above the fitted curve in
  the latest run).
* **Parameter uncertainty vs curve uncertainty.** Individual NSS betas have
  huge, strongly correlated standard errors. The fitted zero curve is pinned
  down to about ±10 bp (95%) at 2–10 years, widening to ±22 bp at 30 years,
  where only one quote anchors it.

To reproduce: `nss-engine run --source fred --start 1990-01-01` and
`python benchmarks/real_data_studies.py`.
