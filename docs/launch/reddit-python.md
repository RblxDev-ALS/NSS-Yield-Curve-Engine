# r/Python (Showcase)

**Title:** nss-engine: fits the U.S. Treasury yield curve from FRED data, with simulated-market tests for every model

**Body:**

**What my project does**

`nss-engine` downloads U.S. Treasury yields from FRED (no API key), fits a
Nelson–Siegel–Svensson curve to every week since 1990, and writes an
interactive dashboard, a Markdown report, CSVs and charts: the curve, the
term premium, breakeven inflation, recession probabilities and forecasts. A
GitHub Action reruns it every weekday and publishes a website.

```bash
pip install nss-engine
nss-engine run            # -> output/dashboard.html
```

**Target audience**

Students, economists and quants who want a reproducible yield curve pipeline
they can read. It's a research and measurement tool, not a trading system.

**Comparison**

Most Nelson–Siegel packages fit one curve with a generic optimizer and treat
the quotes as zero rates. This one fits FRED's quotes as the par yields they
are, solves the linear part in closed form, and checks its output against
the Federal Reserve's own curve (10.0 bp RMSE).

The testing is the part I spent the most time on. Each model has a
simulated market where the true answer is known, and the tests check
accuracy against it. Hypothesis property tests found two real optimizer
bugs. Every real-time estimate has a no-look-ahead test: scramble all data
published after a date and that date's estimate must not change. CI runs on
Python 3.10–3.13, pandas 2 and 3, Linux and Windows (a cp1252 crash taught
me to pass `encoding="utf-8"` everywhere).

Code: https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine
Site: https://rblxdev-als.github.io/NSS-Yield-Curve-Engine/

I'd welcome feedback on the API and the package layout.

---

*Notes: use the Showcase flair; the three section headings are required.*
