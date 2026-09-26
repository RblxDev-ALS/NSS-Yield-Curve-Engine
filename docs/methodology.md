# Methodology

This document explains the mathematics behind every component of the engine and
the reasoning behind each design decision. Notation: maturities $\tau$ in years,
rates in percent.

## 1. The Nelson–Siegel–Svensson curve

Nelson & Siegel (1987) proposed describing the whole zero-coupon curve with
three factors whose *loadings* are fixed functions of maturity:

$$
z(\tau) = \beta_0
+ \beta_1 \underbrace{\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau}}_{\text{slope loading}}
+ \beta_2 \underbrace{\left(\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau} - e^{-\lambda_1\tau}\right)}_{\text{curvature loading}}
+ \beta_3 \left(\frac{1-e^{-\lambda_2\tau}}{\lambda_2\tau} - e^{-\lambda_2\tau}\right)
$$

Svensson (1994) added the $\beta_3$ term so the long end can bend independently of
the short-end hump. Setting $\beta_3 = 0$ recovers Nelson–Siegel.

| Factor | Loading at $\tau=0$ | Loading as $\tau\to\infty$ | Interpretation |
|---|---|---|---|
| $\beta_0$ | 1 | 1 | **Level**: $z(\infty) = \beta_0$ |
| $\beta_1$ | 1 | 0 | **Slope**: $z(0) = \beta_0+\beta_1$, so long − short $= -\beta_1$ |
| $\beta_2$ | 0 | 0 | **Curvature**: hump peaking at $\tau^* = 1.7933/\lambda_1$ |
| $\beta_3$ | 0 | 0 | Second hump at $1.7933/\lambda_2$ |

The constant 1.7933 is the root of $(x^2+x+1)e^{-x}=1$, the first-order condition
for the maximum of the curvature loading (`models.curvature_peak`). With the
Diebold–Li value $\lambda = 0.0609$/month $= 0.7308$/year, the hump sits at
30 months.

**Numerical stability.** For small $x=\lambda\tau$, $1-e^{-x}$ loses precision
through cancellation. The engine uses `expm1` and switches to the Taylor series
$1 - x/2 + x^2/6$ (slope) and $x/2 - x^2/3 + x^3/8$ (curvature) below
$|x| < 10^{-4}$. The tests check continuity across the switch and agreement with a
high-precision reference.

### Forward rates, discount factors and par yields

Because $z(\tau)\tau = \int_0^\tau f(s)\,ds$, the instantaneous forward curve has a
closed form:

$$
f(\tau) = \frac{d}{d\tau}\big[\tau z(\tau)\big]
= \beta_0 + \beta_1 e^{-\lambda_1\tau} + \beta_2 \lambda_1\tau e^{-\lambda_1\tau} + \beta_3\lambda_2\tau e^{-\lambda_2\tau}.
$$

Discount factors are $D(\tau) = e^{-z(\tau)\tau/100}$. A bond paying a coupon $c$
semi-annually prices at par (clean) when

$$
c = 2\,\frac{1 - D(T)}{\sum_i D(t_i) - a}, \qquad t_i = T, T-\tfrac12, \dots > 0 ,
$$

where $a = 1 - 2\min_i t_i$ is the accrued fraction of the current coupon
period (zero for maturities that are whole half-years; see §2.4).

Maturities of one year or less are quoted like bills, on a bond-equivalent basis:
$y = 2\,(D^{-1/(2T)} - 1)$. The tests verify that the forward curve integrates
back to the zero curve, that par bonds price at exactly 100, and that a flat
continuously compounded curve at $r$ has par yield $2(e^{r/2}-1)$.

## 2. Calibration

### 2.1 Why the naive approach fails

Given observed yields $y_i$ at maturities $\tau_i$, calibration minimizes
$\sum_i w_i\,(y_i - z(\tau_i;\theta))^2$ over six parameters. Treating all six
symmetrically with a local optimizer (the approach in v0 of this project) has
three problems:

1. **Multiple local minima.** The surface is not convex in $(\lambda_1,\lambda_2)$.
   Different starting points give different answers (Gilli, Große & Schumann, 2010).
2. **Collinearity.** When $\lambda_1 \approx \lambda_2$ the two curvature loadings are
   nearly identical, so $\beta_2$ and $\beta_3$ can take huge values of opposite sign.
3. **Flat valleys.** Very different $\lambda$ values fit about equally well, so
   warm-starting from last week does not stop the parameters from wandering.

### 2.2 Variable projection

The model is *linear in the betas once the lambdas are fixed*: $y = X(\lambda)\beta + \varepsilon$.
So for any $\lambda$ the optimal betas have a closed form, a weighted ridge regression:

$$
\hat\beta(\lambda) = \big(X^\top W X + R\big)^{-1} X^\top W y,
\qquad R = \mathrm{diag}(0, 0, \rho, \rho),
$$

Substituting $\hat\beta(\lambda)$ back leaves a **two-dimensional** problem in the
lambdas alone (Golub & Pereyra, 1973):

$$
L(\lambda) = \sum_i w_i\,\big(y_i - X_i(\lambda)\hat\beta(\lambda)\big)^2 + \rho\,(\hat\beta_2^2+\hat\beta_3^2)
+ \kappa\,\lVert\log\lambda - \log\lambda_{t-1}\rVert^2 .
$$

The engine minimizes $L$ in two stages:

1. **Global search.** An exhaustive grid in $\log\lambda$, restricted to the feasible
   region below. All grid points are evaluated in one batched `numpy.linalg.solve`.
2. **Local refinement.** SLSQP runs from the three best *well-separated* grid points
   (and from last week's $\lambda$), with the ratio constraint imposed exactly.

**Analytic gradient.** Since $\hat\beta$ minimizes the inner problem, the envelope
theorem says the derivative of $\hat\beta(\lambda)$ is not needed:

$$
\frac{\partial L}{\partial \lambda_j} = -2\sum_i w_i r_i \left(\frac{\partial X_i}{\partial\lambda_j}\hat\beta\right)
+ 2\kappa\,\frac{\log\lambda_j-\log\lambda_{j,t-1}}{\lambda_j},
$$

using $\frac{d}{dx}\frac{1-e^{-x}}{x} = \frac{(1+x)e^{-x}-1}{x^2}$. This halved the
calibration time. The loading derivatives are checked against finite differences to
$10^{-10}$ in the tests.

**Global optimality is tested.** One test compares the calibrator's loss with an
exhaustive 300 × 300 grid and requires the calibrator to be at least as good. A
property-based test (Hypothesis) checks that random noise-free curves are
recovered to within 0.01 bp. That test found a real counterexample: a single
refinement start converged to a competing basin, which is why the calibrator now
refines several distinct starts.

### 2.3 Identification constraints

* $\lambda_1 \ge 1.5\,\lambda_2$ keeps the two humps apart. This removes the
  collinearity and stops $\beta_2$ and $\beta_3$ from swapping roles.
* $\lambda_1 \in [0.15, 3]$ and $\lambda_2 \in [0.06, 1]$ place the humps between
  about 0.6 and 30 years, inside the observed maturity range. Without this, a
  hump beyond 30 years acts as an extra slope and $\beta_0$ loses its meaning as the long-run level.
* The same logic applies when the long end is **missing**. The curvature loading
  $\frac{1-e^{-x}}{x}-e^{-x}$ peaks at $x=\lambda\tau\approx1.7933$, so the lower
  bound on $\lambda_2$ is raised to $1.7933/\tau_{\max}$, where $\tau_{\max}$ is the
  longest *observed* maturity (`hump_within_data`). With the 30-year quoted this
  is the fixed bound above and nothing changes. Without it (February 2002 to
  February 2006, when the Treasury did not issue the 30-year bond, or when
  cross-validation hides it) the second hump cannot peak in the region the curve
  is extrapolating into.
* The ridge penalty $\rho = 10^{-5}$ and the smoothing penalty $\kappa = 10^{-3}$
  were chosen by minimizing the error against the **true** curve on synthetic data
  (`benchmarks/tune_regularisation.py`). In-sample RMSE always prefers no
  regularization, so it is the wrong criterion.

### 2.4 Fitting par yields (the default)

Treasury constant-maturity (CMT) yields are semi-annual *par* yields, but
$z(\tau)$ is a continuously compounded *zero* curve. Fitting $z$ directly to CMT
quotes (the common shortcut, and Diebold–Li's choice, `--target yield`) looks
just as good in sample: on a simulated par-quoted market the in-sample RMSE of the
two approaches is identical (1.36 vs 1.37 bp). The zero curve, however, is
biased: it is 2.9 times further from the true curve (5.3 vs 1.9 bp), −5 bp at
30 years, and the 5y5y forward is off by −7.5 bp. On real data the same
correction moves the fitted curve 38% closer to the Federal Reserve's
independently estimated curve (§3). The engine therefore matches **model par
yields** to the quotes by default.

For a bond with coupon dates $t_1 < \dots < t_n = T$ (every half year, counting
back from $T$), the par coupon makes the *clean* price 100:

$$
c = 2\,\frac{1 - D(T)}{\sum_i D(t_i) - a}, \qquad a = 1 - 2\,t_1 ,
$$

where $a$ is the accrued fraction of the current coupon period ($a=0$ for whole
half-years, as for every CMT tenor). Bills (≤ 1 year) are quoted on a
bond-equivalent basis, $y = 2(e^{z/200}-1)$.

**Calibration.** The par objective is not separable in β, so variable projection
does not apply directly. The engine:

1. estimates the *convexity gap* $g_i = \text{par}_i - z_i$ from last week's curve
   (or a coarse grid fit),
2. runs the global variable-projection search of §2.2 on the adjusted quotes
   $y - g$, which puts the grid on (almost) the right problem,
3. refines **every** basin it finds, plus last week's curve, in par space with
   bounded trust-region least squares, screening them with a short polish and
   fully refining the winner.

The lambdas are reparametrized as $(\log\lambda_2,\ \log\lambda_1 - \log\lambda_2)$,
which turns the ratio constraint into a box bound; if the optimum then wants
$\lambda_1$ outside its own bounds, $\lambda_1$ is fixed at the bound and the rest
re-solved. The Jacobian is analytic:

$$
\frac{\partial P}{\partial\theta} = 100\,\frac{-\partial_\theta D(T)\,A - (1-D(T))\,\partial_\theta A}{A^2},
\qquad \partial_\theta D(t) = -D(t)\,\frac{t}{100}\,\partial_\theta z(t),
$$

with $A = (\sum_i D(t_i) - a)/2$, and coupon dates shared between bonds are
evaluated once (60 instead of 154 curve points for the CMT tenors). Refining a
single start from the zero-curve fit landed in a worse basin on 7% of simulated
dates, costing a median 2.7 bp of RMSE; the multi-start matches a brute-force
reference (a polish from every point of a λ grid) on 149 of 150 dates, the last
within 0.001 bp. A par fit takes about 20 ms per week in panel mode.

### 2.5 Missing data

FRED tenors come and go over history: the 1-month bill starts in 2001, the
20-year was not published from 1987 to 1993, and the 30-year paused from 2002 to
2006. Each date is fitted on the tenors it actually has. Below 7 tenors the engine
falls back to Nelson–Siegel, and below 4 it records a failed fit.

### 2.6 Robust fitting (optional)

`--robust` guards against bad quotes. It iteratively reweights the fit, first with
Huber weights and then with Tukey's bisquare, which gives gross outliers zero
weight. With 11 quotes and 6 parameters, NSS bends towards a bad quote at the
ends of the curve and hides it in the raw residuals. The weights are therefore
computed on **leverage-standardized** residuals $u_i = r_i/\sqrt{1-h_{ii}}$, where
$h_{ii}$ is the diagonal of the *penalized* hat matrix

$$
H = W^{1/2} J\,(J^\top W J + P)^{-1} J^\top W^{1/2},
$$

$J$ is the Jacobian of the fitted quotes and $P$ the curvature of the ridge and
λ-smoothing penalties. Omitting $P$ overstated the leverage of the 30-year point
in panel fits and flagged 12% of *clean* 30-year quotes; with it, 0.2% of clean
quotes are flagged. On a simulated market with a 50 bp error injected into one
random quote per date, the robust fit cuts the zero-curve error from 11.5 to
3.1 bp (1.7 bp without errors).

On FRED data it is **off by default**. The quotes it rejects there are mostly
persistent dislocations, not errors: the 20-year bond (13.6% of weeks) and the
on-the-run 10-year (4.7%). Ignoring them moved the curve *away* from the Fed's
curve (RMSE 11.0 vs 10.0 bp).

### 2.7 Uncertainty

Each fit reports the parameter covariance
$\hat\sigma^2 (J^\top \tilde W J + S P)^{-1}$ and pointwise confidence bands for
zero, par and forward rates by the delta method. The noise estimate
$\hat\sigma$ uses the effective degrees of freedom $n - \operatorname{tr} H$, and the
bands use a Student-t quantile. With 5 residual degrees of freedom a normal
quantile gives only ~89% coverage for a nominal 95% band; the t quantile gives
93–97% in simulation (tested). The individual betas are poorly identified, with
large, strongly correlated standard errors, while the curve itself is pinned
down to a few basis points.

## 3. Factor validation

* **PCA.** Litterman & Scheinkman (1991) showed that three principal components
  (level, slope, curvature) explain almost all Treasury yield variation. The
  dashboard plots the empirical PCA loadings next to the Nelson–Siegel loadings.
* **Model-free proxies** (Diebold & Li, 2006): level $=(y_{3m}+y_{2y}+y_{10y})/3$,
  slope $=y_{10y}-y_{3m}$, curvature $=2y_{2y}-y_{3m}-y_{10y}$. On 1990–2026 data,
  the fixed-λ Diebold–Li factors track these proxies closely (correlations 0.84,
  0.995 and 0.98). The free-λ NSS betas track them less closely (0.59, 0.72 and
  0.79). With λ free, $\beta_0$ is an asymptote beyond the data and $-\beta_1$ is a
  zero-to-infinity spread, and both trade off against $\beta_3$ and $\lambda$ from
  week to week. This is why the regime engine reads its slope off the fitted curve
  (the model-implied 10Y−3M, correlation 0.999 with the observed spread) rather
  than using $-\beta_1$ directly.

### Agreement with the Federal Reserve's curve

The Fed publishes its own daily Svensson curve (Gürkaynak, Sack & Wright, 2007).
It is fitted to *off-the-run* notes and bonds, with no bills, no on-the-run
issues and no 20-year bond, by minimizing duration-weighted price errors. That
makes it an independent estimate of the same zero curve. `validation.py` compares
the two on common dates, per maturity: bias, demeaned RMSE and the correlation
of changes. Every FRED run of the pipeline reports it. Some persistent gap is
expected, because on-the-run issues trade rich, so CMT-based curves sit a few
basis points lower.

## 4. Macro regimes

The slope used is the **model-implied 10-year minus 3-month spread**. This is the
spread studied by Estrella & Mishkin (1998) and used by the New York Fed. It is
read off the fitted curve, so it is defined even when a tenor is missing.

* **Level regimes** Inverted / Flat / Normal / Steep, with boundaries at 0, 0.5 and 1.5 pp.
  **Hysteresis** of 0.1 pp: a boundary is crossed only once the spread clears it
  by that margin, so noise around 0 does not make the label flicker.
* **Dynamics**: the signs of the 4-week changes in level and slope give the
  classic trader taxonomy (bull/bear × steepener/flattener).
* **Recession probit**:
  $P(\text{recession in month } t+12) = \Phi(\alpha + \beta\cdot \text{spread}_t)$,
  fitted by Newton–Raphson on the concave probit log-likelihood with analytic
  gradient and Hessian. NBER recession months come from FRED (`USREC`).
  The engine reports McFadden's pseudo-R² and the ROC AUC. Standard errors are
  optimistic because the 12-month forecast windows overlap, and they are labeled as such.
* **Near-term forward spread** (Engstrom & Sharpe, 2019): the 3-month forward
  rate six quarters ahead minus today's 3-month rate,
  $\frac{z(1.75)\cdot1.75 - z(1.5)\cdot 1.5}{0.25} - z(0.25)$, read off the fitted
  curve. It measures the policy path the market expects; a negative value means
  cuts are priced in.
* **Pseudo-real-time evaluation.** In-sample AUCs flatter a model. The engine
  re-estimates each probit every month on origins whose outcome was already
  known ($s \le t - h$, optionally minus an NBER publication lag) and scores the
  forecast it would have made (AUC, Brier score, log score). Early windows
  contain one or two recessions, so the classes are often perfectly separable
  and the maximum-likelihood slope diverges to ±∞. A weak Gaussian prior on the
  slope coefficients (ridge, $\ell_2 = 1$) keeps those forecasts inside (0, 1).
  The 10Y−3M spread, the near-term forward spread and both together are scored
  on the same forecast months.
* **How sure is the ranking?** The out-of-sample window holds only a few
  recessions, and neighbouring months are strongly dependent (recessions last
  months, and 12-month-ahead targets overlap). The AUC gain of each signal over
  the first is therefore given a 90% **moving-block bootstrap** interval
  (Künsch, 1989): 24-month blocks of forecast months are resampled circularly,
  both signals are scored on the same resample, and the 5th and 95th percentiles
  of the AUC difference are reported.
* **Lead times**: each sustained inversion (≥ 3 months) is matched to the next
  recession start within 36 months. Inversions with no recession within that window are counted as false alarms.

## 5. Forecasting (Diebold–Li, 2006)

With $\lambda$ fixed, the factors $f_t = (\beta_{0t},\beta_{1t},\beta_{2t})$ are
estimated each month by OLS and forecast with AR(1) (or VAR(1)) models:
$\hat y_{t+h}(\tau) = X(\tau)\hat f_{t+h|t}$. The evaluation is strictly out of sample:

* At every origin the model is re-estimated using data **up to that month only**.
* The benchmark is the random walk ("no change"), which is notoriously hard to beat
  (Duffee, 2002).
* Differences are tested with the **Diebold–Mariano** test, using the
  Harvey–Leybourne–Newbold small-sample correction and $h-1$ autocovariance lags
  for $h$-step forecasts.
* The **equal-weight combination** $\tfrac12\hat y^{\text{model}}_{t+h} + \tfrac12 y_t$
  is scored too. Its weights are not estimated, so it cannot overfit, and its
  error is the average of the two errors, so by Minkowski's inequality its RMSE
  is never above the average of the two RMSEs. It beats both whenever their
  errors are not too strongly correlated (Bates & Granger, 1969; Timmermann, 2006).

### 5.1 State-space dynamic Nelson–Siegel

Diebold, Rudebusch & Aruoba (2006) estimate the model in one step as a linear
Gaussian state-space system:

$$
y_t = \Lambda(\lambda) f_t + \varepsilon_t,\ \varepsilon_t\sim N(0,H);\qquad
f_t - \mu = A(f_{t-1}-\mu) + \eta_t,\ \eta_t \sim N(0,Q),
$$

with $H$ diagonal. The Kalman filter gives the exact Gaussian likelihood, and
λ, μ, $A$, $Q$ and $H$ are estimated jointly by maximum likelihood (L-BFGS-B on
an unconstrained parametrization, started from the two-step estimates). Compared
with the two-step approach it handles missing tenors exactly, weights maturities
by their own noise, and yields **predictive distributions**.

*Speed.* Because $H$ is diagonal, the update is done in information form
(Woodbury), so only 3×3 matrices are inverted. The covariance recursion does not
depend on the data. Once it has converged for the current set of observed
maturities, the filter is in steady state, $f_t = G f_{t-1} + b_t$, and the whole
stretch is run at once: $G$ is diagonalized and each mode is a scalar recursion
(`scipy.signal.lfilter`). The log-likelihood matches a textbook covariance-form
filter to $10^{-9}$, including missing and blank dates, and one fit on 400
months takes about 3 s instead of about 60 s.

The measurement noise has a 1 bp floor. Without it the MLE drove the 3-year and
6-month noise on FRED data to zero, a known degenerate optimum in which the
filter treats one yield as exact.

`level_unit_root=True` makes the level a random walk. Out-of-sample forecasts are
re-estimated every 12 months on past data only. The evaluation reports RMSE
against the random walk and the coverage of 80% intervals. On data simulated
from the model itself, with 20 years of training data, the intervals cover
77–84%. They ignore parameter uncertainty, so with short training samples they
are too narrow.

### 5.2 Arbitrage-free Nelson–Siegel (AFNS)

The dynamic Nelson–Siegel model chooses its loadings and its factor dynamics
separately, so nothing rules out arbitrage between bonds. Christensen, Diebold
& Rudebusch (2011) show that if the factors follow a Gaussian diffusion
$dX_t = K^Q(\theta^Q - X_t)\,dt + \Sigma\,dW^Q_t$ with the short rate
$r_t = X_{1,t} + X_{2,t}$ and

$$
K^Q = \begin{pmatrix}0&0&0\\0&\lambda&-\lambda\\0&0&\lambda\end{pmatrix},
$$

then zero-coupon yields have **exactly** the Nelson–Siegel loadings plus a
maturity-dependent constant:

$$
y_t(\tau) = \Lambda(\lambda) X_t - \frac{C(\tau)}{\tau},\qquad
\frac{C(\tau)}{\tau} = \frac{1}{2\tau}\int_0^\tau b(u)^\top \Sigma\Sigma^\top b(u)\,du,
$$

with $b(u) = \big(u,\ \tfrac{1-e^{-\lambda u}}{\lambda},\
\tfrac{1-e^{-\lambda u}}{\lambda} - u e^{-\lambda u}\big)$. The *yield
adjustment* $C(\tau)/\tau$ is a convexity effect: it is negligible at short
maturities and grows roughly like $\tau^2$ through the level term
$\sigma_{11}^2\tau^2/6$.

The engine evaluates the integral for any (full) $\Sigma\Sigma^\top$ with
48-point Gauss–Legendre quadrature, which is exact to rounding because the
integrand is smooth. In the state-space model $\Sigma\Sigma^\top = Q/\Delta t$
(the monthly innovation covariance per year), so the same parameters drive the
dynamics and the cross-section and the restriction adds no parameters.
`fit_dns(..., arbitrage_free=True)` subtracts the adjustment in the
measurement equation; `independent=True` makes $A$ and $Q$ diagonal, CDR's
preferred forecasting specification.

*Tests.* The quadrature matches CDR's closed form for diagonal $\Sigma$ to
$10^{-10}$. Independently, zero-coupon bonds are priced exactly under the
risk-neutral diffusion above for a full $\Sigma$: appending
$Y_t=\int_0^t r_s\,ds$ to the state, $E[Y_\tau]$ and $\operatorname{Var}(Y_\tau)$
follow from a matrix exponential (Van Loan, 1978), and
$-\log P(\tau)/\tau = (E[Y_\tau] - \tfrac12\operatorname{Var}Y_\tau)/\tau$ agrees
with the AFNS yield to $10^{-12}$.

## 6. Term premium (Adrian, Crump & Moench, 2013)

A long yield is the average short rate expected over its life plus a **term
premium**. With monthly log zero yields $y^{(n)}_t$ ($n$ = 1…120 months, in
decimals per month) from the NSS curves:

1. **Factors.** $X_t$ = the first $K=5$ principal components of the yields.
2. **Real-world dynamics.** $X_{t+1} = \mu + \Phi X_t + v_{t+1}$ by OLS,
   $\Sigma = \operatorname{Cov}(v)$.
3. **Excess returns.** For $n$ = 6, 12, …, 120,
   $rx^{(n-1)}_{t+1} = \log P^{(n-1)}_{t+1} - \log P^{(n)}_t - y^{(1)}_t$ is
   regressed on $(1, v_{t+1}, X_t)$:
   $rx = a\iota^\top + \beta^\top V + cX_- + E$, with $\sigma^2$ the variance
   of $E$.
4. **Prices of risk.** No arbitrage implies
   $\lambda_1 = (\beta\beta^\top)^{-1}\beta c$ and
   $\lambda_0 = (\beta\beta^\top)^{-1}\beta\big(a + \tfrac12(B^\*\operatorname{vec}\Sigma + \sigma^2\iota)\big)$,
   where row $n$ of $B^\*$ is $\operatorname{vec}(\beta^{(n)}\beta^{(n)\top})^\top$.
5. **Pricing.** $\log P^{(n)}_t = A_n + B_n^\top X_t$ with $A_0 = 0$, $B_0 = 0$,
   $$A_n = A_{n-1} + B_{n-1}^\top(\mu - \lambda_0) + \tfrac12\big(B_{n-1}^\top\Sigma B_{n-1} + \sigma^2\big) - \delta_0,\qquad
   B_n^\top = B_{n-1}^\top(\Phi - \lambda_1) - \delta_1^\top,$$
   where $y^{(1)}_t = \delta_0 + \delta_1^\top X_t$ by OLS (the $\sigma^2$ term is
   dropped for $n=1$, whose "$(n-1)$-month bond" is cash). The same recursion
   with $\lambda_0=\lambda_1=0$ gives the **risk-neutral yield**, the average
   expected short rate. The term premium is the difference.

*What is identified, and what is not.* The excess-return regression pins down
the risk-neutral dynamics $\Phi - \lambda_1$ exactly: on exact data from a
simulated arbitrage-free market (`synthetic.simulate_affine_market`) its
eigenvalues are recovered to $10^{-6}$ and yields are priced to $10^{-3}$ bp. The
error in the premium comes entirely from the real-world VAR, and shrinks with
the sample: 17 bp RMSE at 50 years of monthly data, 8 bp at 1,700 years. The
term premium is only as good as the estimate of how persistent rates are.

*Stationarity.* On short samples OLS can return an explosive $\Phi$, which sends
ten-year expectations to infinity. If its largest root exceeds 0.998, $\Phi$ is
scaled down to it and $\mu$ reset so the factors keep their sample mean, while
$\Phi - \lambda_1$ and $\mu - \lambda_0$ are held fixed. Fitted yields are then
unchanged; only the split between expectations and premium moves.

*Real time.* `real_time_decomposition` re-estimates the model every month on
data up to that month (after a 60-month start). With so few months the smallest
principal components can be pure noise and the risk-neutral dynamics explode at
long maturities; an estimate that misprices that month's curve by more than
10 bp on average is discarded (one month in 317 on the synthetic market). On
FRED data the real-time premium correlates only 0.43 with the full-sample one
after a 5-year start, 0.58 after 10 years and 0.70 after 15, while the
stationarity cap binds in 0.3% of months: the noise is statistical, not
numerical. Where rates revert to takes decades of data to learn.

*Benchmarks.* The Kim & Wright (2005) premium (FRED `THREEFYTP10`), from a
three-factor affine model fitted with survey forecasts of short rates, and the
same ACM model run on the Fed's GSW curve, as ACM do.

*Recessions.* Rosenberg & Maurer (2008) find that the expectations component
of the spread, not the term premium, carries its recession signal. The engine
tests this with the pseudo-real-time probits of §4: the 10y−3m spread, the
spread minus the real-time 10-year term premium, and the premium alone.

## 7. Risk and relative value

* **Pricing** discounts each cash flow on the NSS zero curve.
* **Effective duration and convexity** come from ±1 bp parallel bumps.
* **Key-rate durations** (Ho, 1992) use triangular bumps at 3M/2Y/5Y/10Y/30Y. They
  sum to the effective duration, which the tests verify.
* **Factor durations** (Willner, 1996): $-\frac{1}{P}\frac{\partial P}{\partial\beta_k}$,
  computed analytically. This is the bond's exposure to a 1 pp move in level, slope
  or curvature. The level factor duration equals the effective duration exactly.
* **Carry and roll-down** of a zero-coupon bond over horizon $h$ with an unchanged curve:
  $z(T)T - z(T-h)(T-h) = \underbrace{z(T)h}_{\text{carry}} + \underbrace{[z(T)-z(T-h)](T-h)}_{\text{roll-down}}$.
* **Rich/cheap signals**: residual = observed − fitted. A positive residual is a
  yield above the curve, so the bond is **cheap**. Rolling z-scores use only past
  data (the test perturbs the last observation and checks earlier values do not
  change). AR(1) half-lives measure how quickly mispricings correct.

## 8. Limitations

* CMT yields are interpolated on-the-run par yields, not prices of individual
  bonds. Residuals are therefore a *curve-shape* signal, not directly tradeable
  mispricings.
* The cross-sectional NSS fit is a statistical curve, not an arbitrage-free
  model. The state-space model can impose no-arbitrage (AFNS, §5.2), and the
  term premium model (§6) is arbitrage-free by construction, but the curves
  they start from are not.
* Term premium estimates are model-dependent and uncertain; different models
  disagree by tens of basis points, mostly because long-run expectations hinge
  on how persistent rates are estimated to be.
* The probit rests on a handful of recessions (four since 1990). Its
  probabilities are indicative and have wide uncertainty, which the
  pseudo-real-time evaluation makes visible.

## References

* Adrian, T., Crump, R. & Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics*.
* Bates, J. & Granger, C. (1969). The combination of forecasts. *Operational Research Quarterly*.
* Beaton, A. & Tukey, J. (1974). The fitting of power series, meaning polynomials, illustrated on band-spectroscopic data. *Technometrics*.
* Christensen, J., Diebold, F. & Rudebusch, G. (2011). The affine arbitrage-free class of Nelson–Siegel term structure models. *Journal of Econometrics*.
* Diebold, F. & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*.
* Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
* Diebold, F., Rudebusch, G. & Aruoba, S. B. (2006). The macroeconomy and the yield curve: a dynamic latent factor approach. *Journal of Econometrics*.
* Duffee, G. (2002). Term premia and interest rate forecasts in affine models. *Journal of Finance*.
* Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*, 2nd ed. Oxford University Press.
* Engstrom, E. & Sharpe, S. (2019). The near-term forward yield spread as a leading indicator: a less distorted mirror. *Financial Analysts Journal*.
* Estrella, A. & Mishkin, F. (1998). Predicting U.S. recessions: financial variables as leading indicators. *Review of Economics and Statistics*.
* Gilli, M., Große, S. & Schumann, E. (2010). Calibrating the Nelson–Siegel–Svensson model. COMISEF working paper.
* Golub, G. & Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. *SIAM Journal on Numerical Analysis*.
* Gürkaynak, R., Sack, B. & Wright, J. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics*.
* Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
* Ho, T. (1992). Key rate durations: measures of interest rate risks. *Journal of Fixed Income*.
* Huber, P. (1964). Robust estimation of a location parameter. *Annals of Mathematical Statistics*.
* Kim, D. & Wright, J. (2005). An arbitrage-free three-factor term structure model and the recent behavior of long-term yields and distant-horizon forward rates. Federal Reserve Board FEDS 2005-33.
* Künsch, H. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*.
* Litterman, R. & Scheinkman, J. (1991). Common factors affecting bond returns. *Journal of Fixed Income*.
* Nelson, C. & Siegel, A. (1987). Parsimonious modeling of yield curves. *Journal of Business*.
* Rosenberg, J. & Maurer, S. (2008). Signal or noise? Implications of the term premium for recession forecasting. *FRBNY Economic Policy Review*.
* Svensson, L. (1994). Estimating and interpreting forward interest rates: Sweden 1992–1994. NBER Working Paper 4871.
* Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting*, vol. 1. Elsevier.
* Van Loan, C. (1978). Computing integrals involving the matrix exponential. *IEEE Transactions on Automatic Control*.
* Willner, R. (1996). A new tool for portfolio managers: level, slope, and curvature durations. *Journal of Fixed Income*.
