# Institutional NSS Yield Curve Engine & Macro Regime Tracker

A quantitative engine that parameterizes the U.S. Treasury term structure with the
**Nelson-Siegel-Svensson (NSS)** model. It turns the fitted curve into relative-value
signals, macro regime labels, risk measures and factor forecasts.

Version 2 rewrites the original single-script prototype as a tested Python package
(`nsscurve`). The main changes are a new calibration algorithm, pricing that follows
market conventions, and several new analytics layers.

---

## What's new in v2

| Area | v1 (script) | v2 (`nsscurve` package) |
|---|---|---|
| **Calibration** | Local L-BFGS-B from yesterday's solution; can get stuck in a stale basin | Global **variable-projection grid search** plus bounded trust-region refinement; the warm start competes with the global candidate instead of trapping the fit |
| **Identification** | λ1 and λ2 share one range, so the two humps can swap roles or become collinear | **Disjoint λ ranges**: short/medium hump vs long-end hump. Label switching can't happen |
| **Regularization** | Heavy ridge (0.005) on β2, β3, which biases the curve | Light ridge plus **temporal smoothness priors** on the betas and on log-λ |
| **Quote conventions** | CMT yields treated as zero rates | CMT handled correctly: bills as zero-coupon, notes/bonds as **semi-annual par yields**, with exact stub accrual. Optional fast zero-approximation mode |
| **Speed** | Numerical gradients | **Analytic Jacobians** for both the zero and par objectives |
| **Data** | `pandas_datareader` (unmaintained, breaks on modern Python); `ffill().dropna()` drops whole dates | Direct FRED client (CSV or JSON API) with caching and retries; bounded forward-fill; missing tenors handled date by date |
| **Analytics** | 2D curve, −β1 plot, surface | Zero, forward, par and discount curves; **key-rate durations**; **carry and roll-down**; PCA; factor validation |
| **Signals** | None | **Rich/cheap residual z-scores** (no look-ahead), mean-reversion half-lives, a signal hit-rate check, 2s5s10s butterfly |
| **Regimes** | Single-threshold label on β1 | Hysteresis plus persistence filter; **bull/bear steepener/flattener** labels; **NY Fed recession-probability probit** |
| **Forecasting** | None | **Diebold-Li** dynamic factor model (AR(1) or VAR(1)) with an expanding-window, out-of-sample **backtest against a random walk** |
| **Engineering** | Top-level script, duplicate legacy code ran on import | Package, CLI, 65 unit tests, CI, offline synthetic mode, 20 output files including a standalone HTML dashboard |

### Benchmark: v1 vs v2 against known truth

`benchmarks/calibration_benchmark.py` simulates 5 years of weekly CMT-style par quotes from a
dynamic NSS model with **known** parameters. The quotes include 1bp of noise and 3bp of
persistent, mean-reverting pricing errors. Each calibrator is then scored on how well it
recovers the true curves (mean over 3 seeds):

| Method | Fit RMSE | Zero-curve error | Forward-curve error | Median weekly Δβ | Runtime |
|---|---|---|---|---|---|
| v1 (L-BFGS-B, ridge 0.005) | 3.8bp | 6.1bp | 8.6bp | 0.040 | 2.0s |
| v2, zero target, no priors | 4.3bp | 7.1bp | 13.4bp | 0.80 | 4.7s |
| v2, par target, no priors | 2.2bp | 3.1bp | 14.2bp | 1.08 | 11.5s |
| **v2 defaults (par + priors)** | **2.5bp** | **2.2bp** | **5.5bp** | 0.060 | 4.4s |

Two findings drove the defaults:

1. **Treating par yields as zero rates is biased.** On a typical curve it misstates the
   30Y zero rate by about 12bp. That error flows into every discount factor and forward
   rate. A regression test (`test_zero_approximation_is_biased_on_par_quotes`) keeps it
   visible.
2. **Unregularized NSS overfits noise.** The parameters churn about 15x more from week to
   week, and the forward curve gets worse. v2's temporal priors restore stability without
   v1's shrinkage bias, and weekly 10Y changes still correlate about 0.99 with the truth.

v1 is faster. Its stability comes from a ridge that flattens real curvature, which is why
its zero-curve error is roughly 3x higher.

---

## The model

$$
y(\tau) = \beta_0 + \beta_1\,\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau}
 + \beta_2\left(\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau}-e^{-\lambda_1\tau}\right)
 + \beta_3\left(\frac{1-e^{-\lambda_2\tau}}{\lambda_2\tau}-e^{-\lambda_2\tau}\right)
$$

* $y(\tau)$ is the continuously compounded zero rate.
* $\beta_0$ is the long rate and $\beta_0+\beta_1$ is the instantaneous short rate, so the
  **curve slope (long minus short) is $-\beta_1$**.
* $\beta_2$ and $\beta_3$ are humps that peak at $1.7933/\lambda_1$ and $1.7933/\lambda_2$
  years.
* The instantaneous forward rate has the closed form
  $f(\tau)=\beta_0+\beta_1e^{-\lambda_1\tau}+\beta_2\lambda_1\tau e^{-\lambda_1\tau}+\beta_3\lambda_2\tau e^{-\lambda_2\tau}$.

**Calibration** minimizes

$$
\sum_i w_i\,(\hat q_i(\theta)-q_i)^2 \;+\; \rho(\beta_2^2+\beta_3^2)
\;+\; s\,\lVert\beta-\beta_{t-1}\rVert^2 \;+\; s_\lambda\lVert\log\lambda-\log\lambda_{t-1}\rVert^2
$$

where $\hat q_i$ is the model **par yield** (or the zero rate in `--fit-target zero` mode).
For fixed λ the zero-target problem is linear in β. The optimal betas are therefore a closed-form ridge solve, which
is evaluated on a vectorized log-spaced (λ1, λ2) grid. The best grid point and the previous
date's solution are then both polished with `scipy.optimize.least_squares` using analytic
Jacobians, and the lower objective wins.

---

## Quick start

```bash
git clone https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine.git
cd NSS-Yield-Curve-Engine
pip install -r requirements.txt        # or: pip install -e .[dev]

python nss_engine.py                   # live FRED data, weekly, 5 years
python nss_engine.py --offline         # synthetic data, no network needed
python nss_engine.py --show            # also open the dashboard in a browser
```

Everything is written to `output/`: `dashboard.html`, `summary.json` and CSV tables. A
console report prints the regime, rich/cheap table, carry and roll-down, key-rate
durations, factor validation and the forecast backtest.

### Useful options

```bash
python nss_engine.py --freq D --years 10              # daily, 10-year history
python nss_engine.py --model ns                       # classic 3-factor Nelson-Siegel
python nss_engine.py --fit-target zero                # fast zero-rate approximation
python nss_engine.py --smoothness 0 --lambda-smoothness 0   # pure date-by-date fits
python nss_engine.py --source csv --csv my_rates.csv  # your own data (tenor or FRED-id columns)
python nss_engine.py --horizon 12 --dynamics var      # 12-period VAR(1) forecast
python nss_engine.py --fallback-synthetic             # use synthetic data if FRED is unreachable
python nss_engine.py --help
```

Set `FRED_API_KEY` to use the official FRED JSON API. Without a key, the public CSV
endpoint is used. Downloads are cached in `.fred_cache/` for 12 hours.

### Python API

```python
from nsscurve import NSSCalibrator, CalibrationConfig, DieboldLiForecaster
from nsscurve.data import FREDClient, clean_curve, resample_curve
from nsscurve import analytics, signals, regimes

rates = resample_curve(clean_curve(FREDClient().fetch_curve(start="2020-01-01")), "W")

history = NSSCalibrator(CalibrationConfig(model="nss", fit_target="par")).fit_history(rates)
curve = history.curve()                      # latest NSSCurve
curve.zero([2, 10, 30]); curve.forward(5.0); curve.par_yield(10); curve.discount(7)
curve.key_rate_durations(coupon=4.25, maturity=10)

signals.rich_cheap_table(history.residuals_bp)       # CHEAP / RICH / FAIR per tenor
analytics.carry_rolldown(curve, horizon=0.25)        # 3M carry + roll-down (bp)
regimes.recession_probability(rates["10Y"] - rates["3M"])

dl = DieboldLiForecaster(dynamics="ar").fit(rates)
dl.forecast(horizon=4)
dl.backtest(rates, horizon=4)["table"]               # RMSE vs random walk, per tenor
```

---

## Outputs

| File | Contents |
|---|---|
| `dashboard.html` | Standalone interactive dashboard (see below) |
| `summary.json` | As-of snapshot: config, fit statistics, latest parameters, regime, recession probability, active signals, forecast skill |
| `nss_macro_signals.csv` | Daily/weekly NSS parameters (β0–β3, λ1, λ2); same filename as v1 |
| `fit_diagnostics.csv` | RMSE, max error, tenor count and winning start per date |
| `market_rates.csv`, `fitted_rates.csv`, `residuals_bp.csv` | Quotes, model quotes and residuals (market − model) |
| `curve_metrics.csv` | Model par yields, 3M10Y / 2Y10Y / 5Y30Y spreads, 2s5s10s fly, short rate, hump locations |
| `regimes.csv` | Model slope, Steep/Flat/Inverted regime, bull/bear steepener/flattener move |
| `recession_probability.csv` | NY Fed probit, monthly |
| `rich_cheap.csv`, `signal_quality.csv`, `butterfly_2s5s10s.csv` | RV signals, half-lives, hit rates |
| `carry_rolldown.csv`, `key_rate_durations.csv` | Carry, roll-down and key-rate risk on the latest curve |
| `factor_validation.csv`, `pca_*.csv` | Factor–proxy correlations; empirical PCA of yield changes |
| `forecast.csv`, `forecast_backtest.csv` | Diebold-Li forecast and out-of-sample skill vs random walk |

### Dashboard

* Market quotes vs the NSS fit, with the instantaneous forward curve, the fit
  four periods earlier (one month on weekly data) and the Diebold-Li forecast.
* Rich/cheap residual bars annotated with z-scores.
* Model slope −β1 vs the market 10Y−2Y and 10Y−3M spreads, with inverted and flat
  regimes shaded.
* NSS factor time series.
* NY Fed 12-month recession probability.
* Calibration RMSE and maximum error over time.
* 3D zero-curve surface.

---

## Project layout

```
nss_engine.py            CLI entry point (python nss_engine.py --help)
nsscurve/
  model.py               NSS math: loadings, zero/forward/discount/par curves, bond pricing,
                         key-rate durations, analytic Jacobians, compounding conversions
  calibration.py         Variable-projection grid + least-squares calibrator, CurveHistory
  data.py                FRED client, cleaning, resampling, CSV loader, synthetic simulator
  tenors.py              Tenor labels <-> years, FRED CMT series map
  analytics.py           Spreads, curve metrics, factor validation, PCA, carry/roll, KRDs
  signals.py             Residual z-scores, half-lives, rich/cheap, butterfly, signal check
  regimes.py             Slope regimes with hysteresis, curve-move labels, NY Fed probit
  forecasting.py         Diebold-Li dynamic factor model + out-of-sample backtest
  dashboard.py           Plotly dashboard
  pipeline.py            End-to-end orchestration and exports
  cli.py                 Argument parsing and console report
benchmarks/              v1-vs-v2 calibration benchmark against known truth
tests/                   pytest suite (65 tests)
```

## Development

```bash
pip install -r requirements-dev.txt
pytest -q                                     # ~15s
ruff check nsscurve tests benchmarks
python benchmarks/calibration_benchmark.py    # ~1 min
```

## Notes and limitations

* **FRED CMT yields are par yields on on-the-run securities.** NSS fits a smooth curve
  through them, so residuals mix genuine rich/cheapness with on-the-run and liquidity
  effects. Use `signal_quality.csv` to check whether a tenor's residual actually
  mean-reverts before trading it.
* **Carry and roll-down use a static-curve approximation** with a par-bond duration.
  Funding defaults to the model 3M rate.
* **Beating a random walk is hard.** The Diebold-Li backtest reports the RMSE ratio
  honestly; a ratio near 1 is typical at short horizons.
* **The recession probability uses the published NY Fed coefficients**
  (α = −0.5333, β = −0.6330 on the monthly-average 10Y−3M spread). It is a statistical
  indicator, not a forecast of this engine.
* **Default penalty strengths were tuned on synthetic data.** Weekly sampling was the
  reference case. For daily data, somewhat stronger smoothing may be appropriate.
