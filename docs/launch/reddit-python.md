# r/Python (Showcase)

**Title:** nss-engine: a tested Python engine for the U.S. Treasury yield curve (numpy/scipy, 226 tests, known-truth simulations, live GitHub Pages dashboard)

**Body:**

**What my project does.** `nss-engine` downloads U.S. Treasury yields from
FRED (no API key), fits a Nelson–Siegel–Svensson curve to every week since
1990, and builds an interactive dashboard, a Markdown report, CSVs and PNG
charts: the curve, recession probabilities, the term premium, breakeven
inflation and forecasts. A GitHub Action reruns it every weekday and
publishes the site.

```bash
pip install nss-engine
nss-engine run            # -> output/dashboard.html
```

**Target audience.** People who work with interest rates (students, quants,
economists) and want a transparent, reproducible pipeline instead of a
black box. It is a research and measurement tool, not a trading system.

**Comparison.** Most Nelson–Siegel packages fit one curve with a generic
optimizer. This one fits the quotes as par yields (what FRED actually
publishes), solves the linear part in closed form, and validates the output
against the Federal Reserve's own curve (10.0 bp RMSE) and other published
estimates.

**The engineering part I'm proudest of is the testing:**

* A **simulated market with a known truth** for every model: a dynamic NSS
  market for the calibrator, an arbitrage-free affine market with a known term
  premium and simulated surveys, and a TIPS market with a known breakeven
  curve. Tests assert accuracy against the truth, not just that code runs.
* **Hypothesis** property tests generate random curves; they found two real
  optimizer bugs (optima in basins narrower than the search grid).
* A **no-look-ahead** test for every real-time estimate: scramble all data
  published after a date and check that date's estimate is unchanged.
* Jacobians checked against finite differences to 1e-8; the fast Kalman
  filter against a textbook implementation to 1e-9.
* CI on Python 3.10–3.13, pandas 2 and 3, Linux and **Windows** (a cp1252
  encoding crash taught me to name `encoding="utf-8"` everywhere; a test now
  scans the package for any unencoded text I/O).
* ruff + mypy, trusted publishing to PyPI from a tag, and charts in the README
  drawn from real data by CI, never by hand.

Repo: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine
Site: https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/

Feedback on the API and the package layout is very welcome.

*Posting notes: r/Python requires the "What my project does / Target
audience / Comparison" sections for Showcase posts; use the Showcase flair.
Update the test count if it changed.*
