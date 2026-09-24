# Changelog

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
* 112 tests (property-based tests included), 95% coverage, ruff + mypy, CI on
  Python 3.10–3.13, and a scheduled job that runs the pipeline on live FRED data

## 0.1 — original prototype
* Nelson–Siegel and NSS fits to FRED constant-maturity yields, a 2×2 Plotly
  dashboard, and CSV export of the factors.
