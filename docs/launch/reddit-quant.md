# r/quant

**Title:** I fitted the Treasury curve weekly since 1990 and checked it against the Fed's. Looking for critique of the term premium tests

**Body:**

I've been working on an open-source Python project that fits a
Nelson–Siegel–Svensson curve to FRED's Treasury yields every week since 1990
and estimates the term premium, breakevens and recession odds from it. It
reruns on public data every weekday. I'd like critique on the methods and
the testing.

A few results:

* FRED's constant-maturity yields are par yields. Fitting a zero curve
  straight to them (which I did until 2.0) puts the curve 16.3 bp from the
  Fed's GSW curve; fitting them as par yields gets it to 10.0 bp.
* Plain ACM re-estimated each month is too unstable to use in real time
  (0.43–0.70 correlation with its own full-sample series). Anchoring the
  expected short rates to SPF bill-rate forecasts gets 0.91–0.94, and 28 bp
  RMSE against Kim–Wright instead of 123 bp. On a simulated market with a
  known premium, the real-time error drops from 82 to 13 bp.
* Bias-corrected VARs didn't help; the stationarity cap binds in most
  real-time months.
* Nothing beats the random walk at forecasting yields one month ahead.

What I'm unsure about: the survey error model (one σ per SPF series, floored
at 0.1 pp), whether 24-month blocks make sense for the AUC bootstrap, and
what else you'd validate the term premium against.

Results and tests: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/blob/main/docs/results.md
Code: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine

---

*Notes: r/quant is strict about self-promotion; check the rules on the day,
lead with the question, and link the results page rather than the dashboard.*
