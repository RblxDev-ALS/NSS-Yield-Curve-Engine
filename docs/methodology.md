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
semi-annually prices at par when

$$
c = 2\,\frac{1 - D(T)}{\sum_i D(t_i)}, \qquad t_i = T, T-\tfrac12, \dots > 0 .
$$

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
* The ridge penalty $\rho = 10^{-5}$ and the smoothing penalty $\kappa = 10^{-3}$
  were chosen by minimizing the error against the **true** curve on synthetic data
  (`benchmarks/tune_regularisation.py`). In-sample RMSE always prefers no
  regularization, so it is the wrong criterion.

### 2.4 Fitting par yields

Treasury constant-maturity (CMT) yields are *par* yields, but $z(\tau)$ is a *zero*
curve. Fitting $z$ directly to CMT quotes (the common shortcut, and Diebold–Li's
choice) is a good approximation. With `--target par` the engine instead
matches **model par yields** to the quotes. It starts from the variable-projection
solution and refines all six parameters with bounded trust-region least squares.
The lambdas are reparametrized as $(\log\lambda_2,\ \log\lambda_1 - \log\lambda_2)$,
which turns the ratio constraint into a simple bound. A test shows that fitting
par quotes as if they were zero rates distorts the zero curve, while the par
target recovers it exactly.

### 2.5 Missing data

FRED tenors come and go over history: the 1-month bill starts in 2001, the
20-year was not published from 1987 to 1993, and the 30-year paused from 2002 to
2006. Each date is fitted on the tenors it actually has. Below 7 tenors the engine
falls back to Nelson–Siegel, and below 4 it records a failed fit.

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

## 6. Risk and relative value

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

## 7. Limitations

* CMT yields are interpolated on-the-run par yields, not prices of individual
  bonds. Residuals are therefore a *curve-shape* signal, not directly tradeable
  mispricings.
* The NSS model is a statistical fit, not an arbitrage-free model. For
  arbitrage-free dynamics see the AFNS model of Christensen, Diebold & Rudebusch (2011).
* The probit uses one predictor and a handful of recessions. Its probabilities
  are indicative and have wide uncertainty.

## References

* Christensen, J., Diebold, F. & Rudebusch, G. (2011). The affine arbitrage-free class of Nelson–Siegel term structure models. *Journal of Econometrics*.
* Diebold, F. & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*.
* Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*.
* Duffee, G. (2002). Term premia and interest rate forecasts in affine models. *Journal of Finance*.
* Estrella, A. & Mishkin, F. (1998). Predicting U.S. recessions: financial variables as leading indicators. *Review of Economics and Statistics*.
* Gilli, M., Große, S. & Schumann, E. (2010). Calibrating the Nelson–Siegel–Svensson model. COMISEF working paper.
* Golub, G. & Pereyra, V. (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate. *SIAM Journal on Numerical Analysis*.
* Gürkaynak, R., Sack, B. & Wright, J. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics*.
* Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
* Ho, T. (1992). Key rate durations: measures of interest rate risks. *Journal of Fixed Income*.
* Litterman, R. & Scheinkman, J. (1991). Common factors affecting bond returns. *Journal of Fixed Income*.
* Nelson, C. & Siegel, A. (1987). Parsimonious modeling of yield curves. *Journal of Business*.
* Svensson, L. (1994). Estimating and interpreting forward interest rates: Sweden 1992–1994. NBER Working Paper 4871.
* Willner, R. (1996). A new tool for portfolio managers: level, slope, and curvature durations. *Journal of Fixed Income*.
