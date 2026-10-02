# Changelog

## 2.5.1 — author name

* The package metadata, `CITATION.cff` and the website name the author, Musa
  Jafri (2.5.0 on PyPI lists only the GitHub handle). No code changes.

## 2.5.0 — do term premia predict bond returns? ACM checked against the New York Fed

### Added
* **Do term premia predict bond returns?** (`nss_engine.returns`). A term
  premium is an expected excess return, so it can be scored against the
  excess returns bonds went on to earn: realized one-year returns on 2-, 5-
  and 10-year zero-coupon bonds (`excess_returns`), the returns the ACM model
  expects (`ACMResult.expected_excess_returns`), the same re-estimated each
  month on past data (`real_time_expected_returns`), and real-time Fama–Bliss
  and Cochrane–Piazzesi regressions. Forecasts are scored against the
  historical mean with the out-of-sample R², Clark–West tests and
  Mincer–Zarnowitz slopes, all with Newey–West variances for overlapping
  returns. Real-data study in `benchmarks/real_data_studies.py`; known-truth
  study in `benchmarks/return_predictability_known_truth.py`.
* **The New York Fed's published ACM term premium**
  (`data.load_acm_term_premium`; the `.xls` file needs `xlrd`, now in the
  `[surveys]` extra). The pipeline and dashboard show it next to Kim–Wright,
  and a study runs this package's ACM code on the Fed's curve since 1961, as
  ACM do, to check the implementation against the published series.
* **Releases from the browser**: *Actions → Release → Run workflow* on `main`
  tags `v<version>`, uploads to PyPI and creates the GitHub release
  ([docs/releasing.md](docs/releasing.md)).
* `scripts/preflight.py`: checks PyPI, the website and its link preview, the
  freshness of the data, README links, the author name and the GitHub About
  box before the project is posted (`--offline` for the repository checks).
* Package metadata: an author on PyPI, `py.typed` and the `Typing :: Typed`
  classifier. A test checks that `pyproject.toml`, `CITATION.cff` and the
  website name the same author.
* The tour notebook compares the 10-year bond's realized excess returns with
  the model's expected ones.

### Results on FRED data
* **ACM replication.** On the Fed's curve since 1961, as ACM estimate it, this
  package's code tracks the New York Fed's published 10-year premium with
  correlation 1.000 (12-month changes 0.999) and a constant offset of 20 bp.
  The same code from 1990 averages 1.72% instead of 0.86%: plain ACM's point
  gap to Kim–Wright on 1990–2026 data comes mostly from the sample start.
* **Bond returns, 2000–2025** (one-year excess returns, forecasts in real
  time, against the historical mean): for the 10-year bond, plain ACM's
  real-time 10-year premium (R² OOS 8.8%, Clark–West p = 0.005) and the
  Fama–Bliss forward spread (7.6%, p = 0.017) beat the mean, in both halves
  of the sample. The survey-anchored premium does not (its expected return,
  −0.2% a year, R² OOS −44%): the SPF expected higher bill rates than came at
  every horizon, 65–76% of the time, and the ten-year forecast was too high
  in all 16 surveys whose window has passed (by 2.2 pp on average). The
  Cochrane–Piazzesi factor explains 21–27% of returns in sample and does far
  worse than the mean out of sample (−43% to −69%), a negative result.
* **Known truth** (24 simulated markets of 440 months): the true expected
  return reaches an out-of-sample R² of 11% for the 10-year bond, the
  survey-anchored premium with accurate surveys 11%, with surveys biased
  +0.5 pp −8%, plain ACM 0%; Fama–Bliss and Cochrane–Piazzesi lose (−10%,
  −15%).
* The survey-anchored premium stays the headline: it is the better estimate
  of what investors expected (Kim–Wright, the model-free survey premium);
  docs/results.md explains why that is a different question from what bonds
  then earned.

### Fixed
* Website on phones: display equations, long inline maths and inline code no
  longer widen the page, and the dashboard's charts move their legends under
  the plot, wrap their titles and zoom the 3-D surface out below 520 px. The
  dashboard page has full link-preview tags. The landing page shows the
  install command with a copy button.

## 2.4.0 — survey-anchored term premium by default, breakeven inflation, website

### Changed (affects results)
* **The survey-anchored term premium is the headline estimate** whenever the
  SPF surveys can be downloaded: in the pipeline, report, dashboard, badges
  and CLI. Plain ACM is reported next to it (`summary["term_premium"]["plain_acm"]`).
  On FRED data the survey-anchored premium is closer to Kim–Wright (RMSE 28 bp
  vs 123 bp), to a model-free survey premium (correlation 0.89 vs 0.80) and,
  estimated in real time, to its own full-sample series (0.91–0.94 vs
  0.43–0.70); see 2.3.0 below and [docs/results.md](docs/results.md).
* The pipeline also re-estimates the survey-anchored premium every month on
  past data (`term_premium_real_time_survey`, about three minutes on FRED data;
  `--fast` skips it), and the recession test of the expectations component
  and the term premium (Rosenberg & Maurer) uses that series.
* The AFNS forecast-interval claim is reversed (see 2.3.0 below).

### Added
* **Real yields and breakeven inflation** (`nss_engine.inflation`): the TIPS
  real curve (FRED `DFII5`–`DFII30`, Nelson–Siegel on par yields), zero-coupon
  breakevens, the 5y5y forward breakeven and par breakevens, compared with
  FRED's `T5YIE`, `T10YIE`, `T5YIFR` and with the Fed's TIPS curve
  (`data.load_gsw_tips_parameters`, Gürkaynak, Sack & Wright 2010). Dashboard
  section, report section, `breakevens.csv`, `tips_nss_parameters.csv`, two
  badges, `--no-inflation`.
* **Known-truth test for breakevens** (`synthetic.simulate_tips_market`,
  `benchmarks/breakeven_known_truth.py`). Over 4 simulated markets the
  engine's 5Y / 10Y / 5y5y breakevens miss the truth by 3.1 / 3.6 / 7.8 bp
  with 3 bp quote noise, against 5.4 / 5.3 / 10.2 bp for FRED's formulas on the
  same quotes; without noise 0.5 / 0.3 / 0.6 bp against 3.2 / 3.1 / 3.5 bp.
* **Real-time recession probabilities up to today**
  (`regime.real_time_probabilities`), in the dashboard and `macro_signals.csv`.
* **Charts for the README** (`nss_engine.figures`, `nss-engine run --figures`,
  `pip install "nss-engine[figures]"`): the curve since 1990, the real-time
  term premium (plain vs survey-anchored vs Kim–Wright) and recession
  probabilities, in light and dark versions, drawn from FRED data by the live
  workflow.
* **Project website** (`website/build.py`): a landing page that states the
  latest reading in words and as a table with year-on-year changes and
  five-year sparklines, the three charts, what did not work, and the
  project's history; the dashboard with the site's navigation; and the
  results, methodology, research note and changelog as HTML with a contents
  list. A favicon, a 404 page and a preview image with the latest numbers
  (`img/social.png`) for links shared on social sites and in chats.
  Published to GitHub Pages from `main`.
* Real-data studies: the recession split test with plain and survey-anchored
  real-time premia (also with NBER dates known only 12 months late), and a
  breakeven study against FRED and the Fed's TIPS curve.

### Results on FRED data
* **Breakevens** (weekly, July 2004 – September 2026): the real curve fits the
  TIPS quotes to 1.15 bp (median). Against the Fed's own zero-coupon
  breakevens the engine's 10-year and 5y5y breakevens have RMSE 8.2 and
  20.4 bp, FRED's `T10YIE` and `T5YIFR` 10.6 and 26.9 bp (the latter 9 bp too
  low on average); at 5 years FRED is slightly closer (12.2 vs 13.8 bp).
* **Recessions, expectations vs term premium** (a negative result): with the
  survey-anchored real-time split, the expectations component predicts
  recessions *worse* than the 10Y−3M spread out of sample (AUC 0.13 vs 0.45,
  90% interval on the difference [−0.54, −0.02], 238 months from 2005); no
  curve signal beats a coin flip in that window. Rosenberg & Maurer's result
  is not confirmed.
* Latest reading (24 September 2026): 10Y zero yield 5.18% = 4.02% expected
  short rate + 1.16% term premium (survey-anchored; plain ACM 1.91%); 10-year
  breakeven 2.33%, 5y5y 2.35%, 10-year real yield 2.85%.
* `docs/results.md` collects the detailed results; the README is shorter.
* MIT license metadata in the package.

### Fixed
* The live workflow's runs on different branches no longer cancel each other.

## 2.3.0 — survey-anchored term premium, significance tests, PyPI

### Added
* **Survey-anchored term premium** (`fit_acm(surveys=...)`): the real-world
  dynamics are estimated jointly from the pricing factors and the Survey of
  Professional Forecasters' 3-month bill forecasts (1–4 quarters, 1–3 calendar
  years and 10 years ahead), as Kim & Wright and Kim & Orphanides discipline
  expectations with surveys. The cross-section, and so the fitted yields, is
  unchanged; only the split into expectations and premium moves.
  `data.load_spf_bill_forecasts` downloads and caches the Philadelphia Fed's
  files (`pip install nss-engine[surveys]` for `openpyxl`).
* **Bias-corrected VAR** (`fit_acm(bias_correction="analytic" | "bootstrap")`):
  Pope's closed-form small-sample bias, or Bauer, Rudebusch & Wu's inverse
  bootstrap, each shrunk to stationarity as in Kilian (1998).
* **Known-truth test** of all four estimators
  (`benchmarks/term_premium_known_truth.py`): `simulate_affine_market` now
  stores its true dynamics, simulates SPF-style surveys, and takes a
  `level_persistence`.
* **Diebold–Mariano tests between forecasts** (`forecasting.compare_forecasts`,
  pooled over tenors) and Newey–West tests of interval coverage
  (`forecasting.hac_mean_test`); the state-space evaluation keeps every
  forecast error (`DNSForecastEvaluation.errors`).
* **PyPI release workflow** (trusted publishing on a `v*` tag, with a build
  check on every packaging change) and [docs/releasing.md](docs/releasing.md).

### Results on FRED data (1990–2026)
* Survey-anchored 10-year premium: mean 0.76%, correlation 0.95 with
  Kim–Wright (12-month changes 0.82), RMSE 28 bp, mean gap −7 bp. Plain ACM:
  mean 1.83%, correlation 0.96 (changes 0.75), RMSE 123 bp, gap +101 bp.
  Bias correction lowers the RMSE only to 113 bp (analytic) or 117 bp
  (bootstrap) and worsens the 12-month changes (0.46, 0.37).
* Against a model-free survey premium (10Y zero yield minus the SPF 10-year
  bill forecast, 35 first quarters): correlation 0.89 survey-anchored, 0.83
  Kim–Wright, 0.80 plain ACM.
* In real time (first estimate after 5 / 10 / 15 years) the survey-anchored
  premium correlates 0.94 / 0.91 / 0.92 with its own full-sample series and
  0.94 / 0.91 / 0.85 with Kim–Wright (plain ACM: 0.43 / 0.58 / 0.70 and
  0.24 / 0.38 / 0.49); RMSE vs Kim–Wright 45–50 bp against 101–112 bp.
  Real-time bias correction is a negative result: agreement with Kim–Wright
  falls to 0.00–0.45 and the stationarity cap binds in 63–92% of months.
* Known truth (8 simulated markets, level persistence 0.99): real-time RMSE
  82 bp plain, 81 / 84 bp bias-corrected, 13 bp with surveys, 53 bp with
  surveys biased by +0.5 pp.
* Diebold–Mariano: AFNS beats the state-space VAR at 1 month (p = 0.014), not
  at 6 or 12; both lose to the random walk at 1 month (p < 0.001); the
  ½ model + ½ random walk combinations tie the random walk at 6 and 12 months
  (p ≥ 0.49).
* Coverage of 80% intervals: AFNS (93 / 89 / 86%) is significantly too wide
  at every horizon; the state-space VAR (86 / 79 / 73%) only at 1 month. 2.2's
  "AFNS has far better calibrated intervals" was wrong and is withdrawn.

### Fixed
* `simulate_affine_market(level_persistence=0.99)` would have made the level
  and slope load identically on every yield, leaving part of the state
  invisible to any yield-based model; the risk-neutral dynamics no longer
  depend on the real-world persistence.

## 2.2.0 — term premium, arbitrage-free dynamics, easier to use

### Added
* **Term premium** (`termpremium.fit_acm`): the Adrian, Crump & Moench (2013)
  regression-based affine model on the NSS zero curves splits each yield into
  the average expected short rate and a term premium. On FRED data the 10-year
  premium tracks the same model run on the Fed's GSW curve with correlation 0.99
  (RMSE 31 bp) and Kim–Wright with correlation 0.96, about a point higher on
  average. Report section, dashboard chart and tile, `term_premium.csv`,
  `--no-term-premium`.
* **Pseudo-real-time term premium** (`real_time_decomposition`), re-estimated
  every month on past data only, and recession probits on the expectations and
  term-premium parts of the 10y−3m spread. On FRED data the real-time
  premium is noisy (correlation 0.43 with the full-sample estimate after a
  5-year start, 0.70 after 15), and on the 2006–2026 origins no curve signal
  beats a coin flip out of sample.
* **Arbitrage-free Nelson–Siegel** in the state-space model
  (`fit_dns(arbitrage_free=True)`, Christensen, Diebold & Rudebusch 2011): the
  yield-adjustment term for any factor covariance, tied to the state shocks, so
  the restriction adds no parameters; `independent=True` for CDR's diagonal
  specification. On FRED data AFNS has the best one-month forecast of any
  single model (1.073 of the random walk's RMSE) and far better calibrated
  12-month intervals (86% coverage for 80%, against 73%), but still does not
  beat the random walk; independent factors are the worst specification.
* **A market with a known term premium** (`synthetic.simulate_affine_market`)
  for testing: the ACM estimator recovers its risk-neutral dynamics exactly.
* `data.load_kim_wright_term_premium` (FRED `THREEFYTP1`–`10`).
* **Colab notebook** (`examples/tour.ipynb`), **live status badges** in
  shields.io endpoint format next to the dashboard, and `CITATION.cff`.

### Fixed
* `DNSResult.n_params` counted restricted entries (a diagonal `A`, a
  random-walk level) as free parameters, overstating the BIC penalty.

## 2.1.0 — honest error bars, forecast combination, long-end guard

### Changed (affects results)
* **The second NSS hump stays inside the data** (`hump_within_data=True`). The
  lower bound on λ2 becomes 1.7933/τmax for the longest *observed* maturity τmax,
  so the hump cannot peak where the curve is only extrapolating. With every
  tenor quoted this is the old bound and fits are bit-for-bit unchanged. On FRED
  data the 30Y leave-one-out error moves from 23.36 to 23.30 bp: correct in
  principle, negligible in practice.

### Added
* **Confidence interval on the recession-signal comparison.** Each signal's
  out-of-sample AUC gain over 10y−3m gets a 90% moving-block bootstrap
  interval (`regime.block_bootstrap_auc_difference`). On FRED data the
  near-term forward spread's +0.099 gain has the interval [−0.035, +0.205],
  so 2.0's "clearly better" was overstated; the README now says so.
* **Equal-weight forecast combination** (½ model + ½ random walk), scored in
  both forecast evaluations (`relative_rmse_combination`). On FRED data it
  gives the project's first ratios below 1 (0.990 at 6 months, 0.996 at 12 for
  the state-space model): a tie with the random walk, not a significant win.
* `ReferenceComparison.subset()`; real-data studies report the fixed-bound
  variant and the months without a 30-year quote.

### Fixed
* A docstring pointed to a benchmark file that does not exist.

## 2.0.0 — par-yield fitting, validation against the Fed, state-space model

### Changed (affects results)
* **Quotes are fitted as par yields by default** (`target="par"`). FRED CMT
  yields are semi-annual par yields; 1.x fitted the continuously compounded zero
  curve straight to them. The in-sample fit of the two is the same, but the old
  zero curve is biased. It was 2.9× further from the truth on a simulated
  par-quoted market (5.3 vs 1.9 bp; −5 bp at 30 years; −7.5 bp on the 5y5y
  forward). On 1990–2026 FRED data it was 16.3 bp from the Federal Reserve's own
  curve, against 10.0 bp for the par fit. `--target yield` restores the old behaviour.
* The synthetic market quotes par yields (`quote="par"`) like FRED, so benchmarks
  and tests exercise the realistic case.

### Fixed
* **Par yields at maturities that are not whole half-years** dropped the stub
  coupon and ignored accrued interest. Plotted par curves were a sawtooth, and
  off-grid `par_yield` values were wrong. CMT tenors were unaffected.
* **Par calibration got stuck in the wrong basin** on ~7% of dates, because it
  refined one start from the zero-curve fit (median cost 2.7 bp RMSE). It now
  refines every basin of a grid search run on convexity-adjusted quotes, and
  matches a brute-force reference.
* When the par optimum pushed λ1 outside its bounds, the par fit was discarded
  and reported as failed. λ1 is now fixed at the active bound and the rest
  re-solved. On FRED data the success rate rose from 93.9% to 100%.
* The state-space MLE could drive a maturity's measurement noise to zero (seen
  for the 3Y and 6M). The noise now has a 1 bp floor.

### Added
* Analytic Jacobian for par fitting and de-duplicated coupon dates: ~20 ms per
  weekly curve, down from ~70 ms.
* **Robust fitting** (`--robust`): Huber then bisquare reweighting on
  leverage-standardized residuals, using the penalized hat matrix. It is opt-in
  because on FRED data it mainly rejects the persistent 20Y/10Y dislocations.
* **Uncertainty**: leverage, effective degrees of freedom, σ̂, parameter
  covariance, and delta-method confidence bands with t quantiles for
  zero/par/forward curves (93–97% coverage for nominal 95%, tested).
* **Validation against the Fed's GSW Svensson curve** (`data.load_gsw_parameters`,
  `validation.compare_to_reference`). It reports bias, demeaned RMSE and change
  correlation per maturity, in every FRED run.
* **Near-term forward spread** (Engstrom & Sharpe, 2019) from the NSS forward curve.
* **Pseudo-real-time recession evaluation**: probits re-estimated monthly on
  outcomes known at the time, scored by out-of-sample AUC, Brier and log score,
  with a ridge prior against perfect separation. The report compares 10y−3m,
  the forward spread, and both together.
* **State-space dynamic Nelson–Siegel** (`statespace.fit_dns`), a Kalman filter
  and maximum-likelihood model after Diebold, Rudebusch & Aruoba (2006), with
  exact missing-data handling, predictive intervals and an optional random-walk
  level. The steady-state filter is vectorized: one fit takes ~3 s instead of ~60 s.
* Real-data studies: comparison with the GSW curve by fitting target, and
  state-space forecasts with interval coverage.
* Dashboard: confidence band and rejected quotes on today's curve, forward
  spread vs 10y−3m, gap to the Fed's curve, and a state-space forecast fan.
* Diagnostics: optimizer messages per date, and the share of fits with a decay rate at a bound.

## 1.0.0 — rebuilt as a tested package

The original version was a single Colab-exported script. Version 1.0 rebuilds
it as an installable, tested Python package and fixes several methodological
problems in the original.

### Fixed (problems in v0)
* **Regularization dominated the fit.** v0 added `0.005·(β2² + β3²)` to a mean
  squared error measured in %². For a typical curve that penalty was larger
  than the fitting error itself, so curvature was shrunk toward zero. On synthetic
  data with known true curves, v0's curves were about 3× further from the truth
  than v1's (see `benchmarks/compare_legacy.py`).
* **Local optimizer on a multi-modal surface.** A single L-BFGS-B run over all
  six parameters converges to whichever basin is nearest the starting point.
  Even without the penalty it fits worse than a global search on the same data.
* **Wrong sign in the initial guess.** v0 initialized `β1 = y_long − y_short`,
  but `β1 = short − long`.
* **Unidentified decay parameters.** Both λ's shared the range [0.2, 5], so the
  two curvature terms could be collinear or swap roles.
* **Missing data.** v0 forward-filled and then dropped every date with any missing
  tenor, which discards all data before 2001 (the 1-month bill did not exist).
  Also, when fewer than 5 weeks were available, the "1 month ago" comparison fell back to a row with a mismatched shape.
* **Code ran on import**, contained two concatenated scripts, and silently
  dropped failed fits.

### Added
* Variable-projection calibrator: closed-form betas, global grid over the
  decay rates, analytic-gradient SLSQP refinement from every grid-local minimum,
  identification constraints, ridge and λ-smoothing penalties tuned on known truth
* Par-yield fitting (`--target par`), Nelson–Siegel mode, fixed-λ Diebold–Li mode
* FRED client with caching, retries, stale-cache fallback and optional API key
* Forward, discount and par curves
* Bond analytics: DV01, duration, convexity, key-rate and factor durations,
  carry and roll-down
* Relative value: residual z-scores with no look-ahead, mean-reversion half-lives
* Regimes with hysteresis, bull/bear steepener/flattener detection, inversion
  lead times, and a probit recession model on NBER data
* Diebold–Li forecasting with rolling out-of-sample evaluation and
  Diebold–Mariano tests
* PCA validation of the latent factors
* Interactive dashboard (light/dark), Markdown report, CSV/JSON exports, CLI
* Synthetic market simulator with known true parameters
* 117 tests (property-based tests and a real market curve included), 95% coverage, ruff + mypy, CI on
  Python 3.10–3.13, and a scheduled job that runs the pipeline on live FRED data

## 0.1 — original prototype
* Nelson–Siegel and NSS fits to FRED constant-maturity yields, a 2×2 Plotly
  dashboard, and CSV export of the factors.
