# Changelog

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
