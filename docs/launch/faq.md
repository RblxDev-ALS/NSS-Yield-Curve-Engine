# Likely questions

The numbers you need to answer common questions in launch threads. Answer in
your own words; these are notes, not replies to paste. Every number is from
the README, [docs/results.md](../results.md) or the live workflow; update
them if a newer run changes them.

**"Nelson–Siegel is a 1987 model. Why not a spline / the Fed's curve / a neural net?"**
NSS is still what central banks publish (the Fed's GSW curve is Svensson) and
its six parameters mean something: level, slope, curvature. The point of the
project is not a new curve model but fitting a standard one carefully and
checking it: leave-one-tenor-out cross-validation shows NSS beats
Nelson–Siegel between quoted maturities (8.3 vs 9.3 bp out of sample) and is
worse beyond the last quote (23.4 vs 17.2 bp at 30 years). That trade-off is
in the results, not hidden.

**"Why should I trust your curve?"**
Don't; check it. It is compared every run with the Federal Reserve's own
independently estimated curve (off-the-run bonds, different method): 10.0 bp
RMSE on 1–30 year zero rates, weekly changes correlate 0.97. The remaining gap
is mostly a stable on-the-run vs off-the-run offset.

**"FRED yields are par yields, everyone knows that."**
Many public Nelson–Siegel implementations (and this project until 2.0) fit
zero curves straight to them. The bias is invisible in-sample and shows up
against an external truth: 16.3 → 10.0 bp against the Fed's curve, 59% less
error on a simulated market. The research note (docs/par-vs-zero.md) has the
decomposition.

**"Your term premium is just a model output."**
Yes, and models disagree by a point or more. That is why the headline number
is the survey-anchored one and why it is compared three ways: with the Fed
Board's Kim–Wright (RMSE 28 bp vs 123 bp for plain ACM), with a model-free
survey premium (10Y yield minus the SPF's own 10-year bill forecast,
correlation 0.89), and on simulated markets with a known premium (13 bp
real-time error vs 82 bp).

**"Anchoring to Kim–Wright's inputs and then agreeing with Kim–Wright is circular."**
Partly, which is why the model-free survey premium is there: it uses no term
structure model at all, and the survey-anchored estimate agrees with it better
(0.89) than Kim–Wright does (0.83). And on simulated data, where the truth is
known, surveys cut the real-time error from 82 to 13 bp, even surveys biased
by half a point help (53 bp).

**"Is your ACM even implemented correctly? The NY Fed publishes it."**
Yes, and the project checks against it. Run on the Fed's own curve since 1961,
as ACM do, this package's code moves one for one with the published 10-year
premium (correlation 1.000, 12-month changes 0.999) with a constant 20 bp
offset. Starting the same model in 1990 instead adds about 65 bp: that, not
the code, is why plain ACM on 1990–2026 data sits a point above Kim–Wright.

**"A term premium is an expected return. Does yours predict returns?"**
This is the 2.5 study; see docs/results.md, "Do term premia predict bond
returns?", for the latest numbers. Short version: plain ACM's premium has a
little out-of-sample power for 10-year bond returns, the survey-anchored one,
which matches Kim–Wright best, has none, because the surveys kept expecting
rate rises that did not come. The Cochrane–Piazzesi factor, famous in sample,
does far worse than the historical mean out of sample.

**"Did you try the bias-corrected VAR (Bauer–Rudebusch–Wu)?"**
Yes, analytic and bootstrap. On real data it barely moves the level (RMSE vs
Kim–Wright 113–117 bp) and makes 12-month changes worse (correlation 0.37–0.46
vs 0.75). In real time the stationarity cap binds in 63–92% of months. On
simulated markets it does not help either (81–84 vs 82 bp). Reported as a
negative result.

**"Your recession model has four recessions. Meaningless."**
Agreed that it is weak evidence, and the results say so: the forward spread
beats 10Y−3M out of sample (AUC 0.71 vs 0.61) but the 90% block-bootstrap
interval on the gain is [−0.04, +0.21]. Every claim about recessions is scored
in pseudo-real time with only recessions known at each date, and the 2022–24
inversion without a recession is in the sample.

**"Can I trade this?"**
Not as a forecast of yields: the random walk beats every model at 1 month
(Diebold–Mariano p < 0.001) and a 50/50 model/random-walk mix only ties it at
6–12 months (p ≥ 0.49). Residuals are curve-shape signals on interpolated
constant-maturity yields, not executable bond prices. It is a measurement and
research tool.

**"Breakevens are not inflation expectations."**
Right: they include an inflation risk premium and a TIPS liquidity discount
(large in 2008). The docs call them inflation compensation. What the engine
adds is reading them off fitted zero curves instead of differencing par
yields; on simulated markets FRED's own formulas are off by 2.6 bp at 5 years
even with perfect quotes.

**"Why only U.S. Treasuries?"**
Because the validation data (Fed curves, Kim–Wright, SPF, NBER) are U.S. The
calibrator takes any panel of par or zero yields (`load_yields_csv`), so other
curves work; they just are not validated here.

**"Is it just a wrapper around scipy.optimize?"**
The calibrator is variable projection: the betas are solved in closed form for
any decay rates, an exhaustive grid finds every basin of the remaining 2-D
problem, and each basin is refined with an analytic Jacobian. Property-based
tests found two real bugs where the optimum sat in a basin narrower than the
grid, which is why every basin is refined. A brute-force test checks global
optimality.

**"How long does it take?"**
About 26 ms per curve, so 36 years of weekly curves in about 70 s. The full
pipeline with the real-time survey-anchored term premium takes a few minutes
(`--fast` skips the month-by-month re-estimation).

**"Did you use AI to build this?"**
Answer this one honestly and in your own words: which tools you used, for
what, and what you did yourself (the idea, the design choices, the checks
against the Fed's numbers, the papers you read). Most people are fine with
AI-assisted code they can see was tested; what goes badly is a denial that
turns out not to be true.
