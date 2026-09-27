# Par yields are not zero rates

### A quiet bias in Nelson–Siegel fits to Treasury constant-maturity yields

*NSS Yield Curve Engine, research note, version 2.3 (September 2026).
Every number can be reproduced with the commands in the appendix.*

---

**Abstract.** Nelson–Siegel models of the U.S. Treasury curve are usually fitted
straight to the Federal Reserve's constant-maturity (CMT) yields, as if they
were continuously compounded zero-coupon rates. They are semi-annual *par*
yields. This note measures what that shortcut costs. The gap between a par
quote and the zero rate of the same maturity has two parts: a compounding
term of about $y^2/4$ (6 bp at 5% rates) at every maturity, and a coupon term
proportional to the slope of the curve that reaches −13 to −22 bp at 10–30
years on a steep curve. Fitting the zero curve to the quotes absorbs both into
the curve. The in-sample fit cannot reveal this, because both readings of the
quotes fit them equally well. Against an external truth the difference is
large. On a simulated par-quoted market with known true curves, fitting model
par yields instead cuts the zero-curve error from 5.7 to 2.4 bp (−59%). On
1990–2026 FRED data, it cuts the distance to the Federal Reserve's
independently estimated Svensson curve (Gürkaynak, Sack & Wright, 2007) from
16.3 to 10.0 bp (−38%), and the 30-year leave-one-out extrapolation error
from 29.2 to 23.4 bp. A robust (outlier-downweighting) fit, which helps on
simulated bad quotes, moves the curve *away* from the Fed's on real data,
because the quotes it rejects are persistent market features rather than
errors.

---

## 1. Introduction

The H.15 release and FRED publish Treasury yields at eleven constant
maturities, from one month to thirty years. They are the most convenient
source of U.S. yield curve data, and the starting point of a large literature
that follows Diebold & Li (2006): fit the Nelson–Siegel (NS) or
Nelson–Siegel–Svensson (NSS) zero curve

$$
z(\tau) = \beta_0 + \beta_1\,\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau}
+ \beta_2\left(\frac{1-e^{-\lambda_1\tau}}{\lambda_1\tau}-e^{-\lambda_1\tau}\right)
+ \beta_3\left(\frac{1-e^{-\lambda_2\tau}}{\lambda_2\tau}-e^{-\lambda_2\tau}\right)
$$

to each day's quotes by least squares, and use the fitted curve, or its
factors, for forecasting, term-structure models and risk.

The CMT yields are not zero rates. Beyond one year they are read off a curve
through the bid yields of on-the-run notes and bonds, which trade near par, and
are quoted as semi-annual bond-equivalent yields. Bills (one year and less)
are quoted on a bond-equivalent basis too. So a CMT quote at maturity $T$ is
the coupon $c(T)$ that makes a semi-annual bond maturing at $T$ price at par,
while $z(T)$ is the continuously compounded yield of a single payment at $T$.

The difference is not new: it is why the Federal Reserve builds its own zero
curve from bond prices (Gürkaynak, Sack & Wright, 2007, "GSW"). But treating
CMT quotes as zeros is common practice, and its cost is rarely measured. Version
1.x of this engine made the same shortcut. Version 2.0 fixed it; this note
documents why the fix matters and how large the effect is.

## 2. Anatomy of the par–zero gap

With discount factors $D(t) = e^{-z(t)\,t}$ (rates as decimals here) and coupon
dates $t_1 < \dots < t_n = T$ every half year, the par coupon solves

$$
1 = \frac{c}{2}\sum_{i=1}^{n} D(t_i) + D(T)
\quad\Longrightarrow\quad
c(T) = 2\,\frac{1 - D(T)}{\sum_i D(t_i)} .
$$

(For maturities that are not whole half-years the engine also handles the stub
coupon and accrued interest; the CMT tenors are all whole half-years.) Write
$c^{cc} = 2\ln(1 + c/2)$ for the same coupon rate continuously compounded.
The error from reading the quote as a zero rate splits into two parts:

$$
\underbrace{c(T) - z(T)}_{\text{reading error}}
= \underbrace{c(T) - c^{cc}(T)}_{\text{compounding}}
+ \underbrace{c^{cc}(T) - z(T)}_{\text{coupon}} .
$$

**Compounding.** For a flat curve at $r$, $c = 2(e^{r/2} - 1)$, so
$c - r \approx r^2/4$: 6.3 bp at 5%, 1.0 bp at 2%, 25 bp at 10%. It raises the
quote above the zero rate at every maturity, by an amount that moves with
the *level* of rates.

**Coupon.** Linearize the pricing equation around a flat curve at $c^{cc}$. A
perturbation $\delta z(t)$ keeps the bond at par only if
$\sum_i w_i\,\delta z(t_i) = 0$ with $w_i \propto t_i\,\text{CF}_i\,D(t_i)$,
the weights of the bond's Macaulay duration. So, to first order,

$$
c^{cc}(T) \approx \sum_i w_i\, z(t_i),
\qquad
z(T) - c^{cc}(T) \approx \sum_i w_i\,\big(z(T) - z(t_i)\big):
$$

the par yield is a duration-weighted average of the zero rates up to $T$. On
an upward-sloping curve it sits *below* the zero rate, by an amount that grows
with maturity and with the *slope*; on an inverted curve it sits slightly
above.

Table 1 evaluates the exact gap on stylized curves.

**Table 1.** Par quote minus zero rate (bp), split into its two parts.

| curve | maturity | zero rate | par quote | quote − zero | compounding | coupon |
|---|---:|---:|---:|---:|---:|---:|
| flat 5% | any | 5.00% | 5.06% | +6.3 | +6.3 | 0.0 |
| flat 2% | any | 2.00% | 2.01% | +1.0 | +1.0 | 0.0 |
| steep, 1% → 5% | 2Y | 2.41% | 2.41% | +0.4 | +1.4 | −1.0 |
| | 10Y | 4.45% | 4.36% | −8.3 | +4.7 | −13.0 |
| | 30Y | 5.02% | 4.86% | −15.9 | +5.8 | −21.7 |
| inverted, 5% → 4% | 2Y | 4.26% | 4.32% | +5.5 | +4.6 | +0.9 |
| | 10Y | 4.14% | 4.18% | +4.5 | +4.3 | +0.2 |
| | 30Y | 4.10% | 4.16% | +5.6 | +4.3 | +1.4 |
| hump at 20Y, 1% → 3% | 10Y | 2.91% | 2.86% | −5.3 | +2.0 | −7.3 |
| | 20Y | 3.37% | 3.27% | −9.6 | +2.7 | −12.2 |

Two features matter for what follows. First, the gap is not small compared with
the fitting error of a good NSS fit (a median of 3.8 bp on FRED data). Second,
it is not a constant: it moves with the level and the slope of the curve, so it
contaminates the *dynamics* of the fitted factors, not just their average.

## 3. What the fit does with the wrong reading

Fitting $z(\tau;\theta)$ to the quotes $c_i$ finds the NSS curve closest to the
points $(\tau_i, c_i)$. The fitted curve is then, roughly, the true zero curve
plus the NSS-shaped projection of the gap in Table 1. Because the gap is smooth
in maturity, NSS can absorb almost all of it: the fit is excellent, and wrong.

This has an uncomfortable consequence: **nothing computed from the quotes
alone can detect the error.** Both readings produce a six-parameter curve that
passes within noise of eleven quotes. On a simulated par-quoted market the
in-sample RMSE of the two fits is the same (1.36 vs 1.37 bp in one test;
2.15 bp for both with the default smoothing in Table 2). Leave-one-tenor-out
cross-validation of *interior* maturities does not help either, as §5.3 shows,
because the held-out quote is predicted on the same basis the model was fitted
on. The error is in the mapping from quotes to the curve, and only an external
reference can see it: a simulated market where the true curve is known, or an
independent estimate of the same curve.

## 4. The fix: fit model par yields

The engine instead matches **model par yields** to the quotes: it minimizes
$\sum_i w_i\big(c_i - \mathrm{par}(\tau_i;\theta)\big)^2$ plus small ridge and
week-to-week smoothing penalties, with bills priced on their bond-equivalent
basis $2(e^{z/200}-1)$. This objective is no longer linear in the betas for
fixed decay rates, so the variable-projection trick that makes zero fits fast
and globally reliable does not apply directly. The engine keeps its benefits
in three steps:

1. estimate the *convexity gap* $g_i = \mathrm{par}_i - z_i$ from last week's
   curve;
2. run the global variable-projection grid search over $(\lambda_1, \lambda_2)$
   on the adjusted quotes $c_i - g_i$, which is (almost) the right problem;
3. refine every basin the grid finds, plus last week's curve, in par space by
   bounded trust-region least squares with an analytic Jacobian,

$$
\frac{\partial\,\mathrm{par}}{\partial\theta}
= \frac{-2\,\partial_\theta D(T)\,S - 2\,(1-D(T))\,\partial_\theta S}{S^2},
\qquad S = \sum_i D(t_i),\quad
\partial_\theta D(t) = -D(t)\,t\,\partial_\theta z(t).
$$

Refining only the start implied by the zero fit landed in a worse basin on 7%
of simulated dates (median cost 2.7 bp); the multi-start matches a brute-force
search over a λ grid on 149 of 150 test dates (the last within 0.001 bp). A par
fit takes about 20 ms per weekly curve, against 10 ms for a zero fit, so 36
years of weekly data take about a minute.

## 5. Evidence

### 5.1 A simulated market with known curves

The engine's synthetic market draws daily NSS parameters from a dynamic model
with a zero lower bound and realistic inversions, and quotes semi-annual par
yields at the eleven CMT maturities with 3 bp of i.i.d. noise. Table 2 compares
calibration methods on three seeds × ten years of weekly curves.

**Table 2.** Fit to the quotes and error against the true zero curve (bp).

| method | fit RMSE | error vs truth | worst 1% vs truth |
|---|---:|---:|---:|
| v0: 6-D L-BFGS-B + ridge (the original script) | 5.57 | 8.01 | 15.0 |
| 1.x: variable projection + smoothing, quotes as zeros | 2.15 | 5.74 | 12.4 |
| 2.0: par target, each week independent | 1.98 | 2.58 | 5.8 |
| **2.0: par target + smoothing (default)** | **2.15** | **2.37** | **5.3** |

The fit to the quotes is identical for 1.x and 2.0 (2.15 bp); the error against
the truth falls by 59% and the worst cases by more than half.

Table 3 shows where the zero reading goes wrong, on 30 years of month-end
curves from one seed. Its RMSE is about 6 bp at every maturity out to seven
years and rises to 10.5 bp at 30 years, where the coupon term grows. The
average one-year error, +4.8 bp, is the compounding term: the mean of $y^2/4$
over this market's one-year rates is 4.9 bp. The par fit is
at the noise floor (1.7–2.3 bp) until the long end, where only one quote
anchors the curve.

**Table 3.** RMSE of the fitted zero curve against the truth, by maturity (bp).

| fit | 1Y | 2Y | 5Y | 10Y | 20Y | 30Y | 5y5y forward |
|---|---:|---:|---:|---:|---:|---:|---:|
| quotes read as zero rates | 6.0 | 6.1 | 5.7 | 6.6 | 8.6 | 10.5 | 9.0 |
| par fit | 1.8 | 1.7 | 1.7 | 2.1 | 2.3 | 4.1 | 3.9 |

The mechanism of §2 shows up directly in these errors. Regressing each date's
error on the true level squared and the 10Y−3M slope explains 68–82% of the
zero reading's error, and none of the par fit's (Table 4). At two years the
level coefficient is 0.24 bp per %², matching the $y^2/4$ compounding term
(0.25).

**Table 4.** Fitted-curve error regressed on level² (%²) and slope (points).

| fit | maturity | bp per unit of level² | bp per point of slope | R² |
|---|---:|---:|---:|---:|
| quotes read as zero rates | 2Y | 0.24 | −1.1 | 0.75 |
| | 10Y | 0.47 | −2.0 | 0.82 |
| | 30Y | 1.20 | +1.0 | 0.68 |
| par fit | 2Y | −0.01 | 0.00 | 0.00 |
| | 10Y | 0.01 | 0.10 | 0.00 |
| | 30Y | 0.02 | 0.01 | 0.00 |

(At long maturities the fitted zero-reading error is not the raw gap of
Table 1: NSS spreads the gap across maturities in whatever way best fits all
eleven quotes, which is why the slope coefficient changes sign at 30 years.)

### 5.2 The Federal Reserve's curve

Real data has no known truth, but it has an independent estimate of the same
object. GSW fit a Svensson curve to *off-the-run* Treasury notes and bonds
(excluding bills, on-the-run issues and the 20-year bond) by minimizing
duration-weighted price errors. Table 5 compares zero curves fitted from FRED
CMT quotes with the GSW curve over 440 month-ends, 1990–2026, from one to
thirty years.

**Table 5.** Engine zero curves against the Fed's GSW curve (bp).

| reading of the quotes | RMSE | RMSE after removing each maturity's mean gap | mean gap at 20Y | 5y5y forward, mean gap |
|---|---:|---:|---:|---:|
| as zero rates (1.x) | 16.3 | 12.0 | −18.5 | −21.8 |
| **as par yields (2.0 default)** | **10.0** | **7.8** | −9.7 | −14.5 |
| as par yields, robust fit | 11.0 | 8.7 | −10.8 | −15.1 |

The par reading is 38% closer to the Fed's curve overall and halves the gap at
20 years. Its weekly changes correlate 0.97 with GSW's. Part of the remaining
difference is expected and stable, not error: the CMT curve runs through
on-the-run issues, which trade rich (lower yields) because they are the most
liquid, while GSW deliberately excludes them. That is consistent with the
engine's curve sitting below GSW's at the long end.

The effect reaches everything built on the curve. The 5y5y forward, a common
gauge of long-run inflation and policy expectations, moves 7 bp closer to the
Fed's. Term-structure models inherit the improvement: an Adrian–Crump–Moench
term premium estimated on the engine's par-fitted curves tracks the same model
estimated on the GSW curve with correlation 0.99 (RMSE 31 bp).

### 5.3 Leave-one-tenor-out cross-validation

Table 6 hides each maturity in turn, fits the rest, and predicts the hidden
quote (441 month-end curves). For interior maturities the two readings are
indistinguishable (8.31 vs 8.34 bp), as §3 predicted: the prediction is made on
the same basis the model was fitted on, so the mapping error cancels. Beyond the
last quote it does not cancel. With the 30-year quote hidden, the curve must be
extrapolated from 20 years, and the zero reading's distorted long end costs
5.8 bp.

**Table 6.** Out-of-sample RMSE of the held-out quote (bp).

| model | 3M–20Y (interpolation) | 30Y (extrapolation) |
|---|---:|---:|
| Nelson–Siegel (4 parameters) | 9.31 | **17.2** |
| NSS, each date fitted independently | 8.69 | 23.9 |
| **NSS + λ smoothing (default)** | 8.34 | 23.4 |
| NSS + λ smoothing, quotes as zeros | **8.31** | 29.2 |

The table also carries a separate lesson: NSS's extra flexibility helps between
quotes (10% lower error than NS) and hurts beyond them, where the simpler
Nelson–Siegel extrapolates best.

### 5.4 A negative result: robust fitting

With eleven quotes and six parameters, one bad quote can bend the whole curve.
The engine therefore offers iteratively reweighted fitting (Huber, then Tukey's
bisquare) on leverage-standardized residuals. On a simulated market with a
50 bp error injected into one random quote per date it works as intended,
cutting the zero-curve error from 11.5 to 3.1 bp (1.7 bp with no errors). On FRED
data it moves the curve *away* from the Fed's (Table 5: 11.0 vs 10.0 bp). The
quotes it rejects are the 20-year bond (13.6% of weeks), which has traded cheap
to its neighbours since it was reintroduced, and the on-the-run 10-year (4.7%).
These are persistent market features, not data errors; discarding them throws
away real information about the curve. Robust fitting is therefore opt-in
(`--robust`).

## 6. Discussion

**When it matters.** The compounding term scales with the square of the level
and the coupon term with the slope, so the bias is largest when rates are
high and the curve is steep, as in the early 1990s. It matters again now: at
4–5% rates the compounding term alone is 4–6 bp at every maturity, more than a
typical NSS fitting error.

**What it contaminates.** Because the gap moves with level and slope, the
factors of a dynamic Nelson–Siegel model fitted to CMT quotes carry a
state-dependent bias: the level factor is overstated when rates are high, and
the long end is understated when the curve is steep. Forward rates, which
differentiate the curve, are hit hardest (Table 5: 22 bp at 5y5y). Any model
that reads expectations or premia off the long end (term premium models,
breakeven inflation, forward-guidance studies) inherits the error.

**What it does not fix.** Fitting par yields removes a mapping error; it does not
make CMT quotes into bond prices. They remain interpolated yields of a few
on-the-run securities, with their liquidity premia, and the fitted curve
remains a statistical curve rather than an arbitrage-free model. The remaining
7.8 bp demeaned gap to GSW is a mix of those effects and of GSW's own
estimation error.

**A cheap fix.** Matching model par yields costs about 10 ms more per curve
than matching zero rates, needs no extra data, and leaves the in-sample fit
unchanged. For anyone fitting yield curves to CMT data it should be the
default.

## 7. Conclusion

Reading Treasury CMT quotes as zero rates adds a compounding bias of about
$y^2/4$ and a slope-dependent coupon bias that reaches −22 bp at 30 years on a
steep curve. Both are invisible to the in-sample fit and to interior
cross-validation. Fitting model par yields removes them: on a simulated market
with known curves the error falls by 59%, and on 36 years of FRED data the
fitted curve moves 38% closer to the Federal Reserve's independently estimated
curve. Robust down-weighting of "outlier" quotes, by contrast, hurts on real
data, where the outliers are persistent market features.

## Appendix: reproducing the numbers

```bash
pip install -e ".[dev]"
python benchmarks/par_vs_zero.py            # Tables 1, 3, 4 (simulated, ~15 s)
python benchmarks/compare_legacy.py --seeds 0 1 2   # Table 2 (simulated)
python benchmarks/real_data_studies.py      # Tables 5 and 6 (FRED and the Fed's
                                            # GSW curve; needs internet)
```

The real-data tables are regenerated on every push by the repository's
[Live dashboard workflow](https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine/actions/workflows/live-dashboard.yml),
which prints them in the job summary. The formulas for par yields, the
Jacobian and the robust weights are derived in
[methodology.md](methodology.md), §§1 and 2.

## References

* Diebold, F. & Li, C. (2006). Forecasting the term structure of government bond
  yields. *Journal of Econometrics*, 130(2), 337–364.
* Gürkaynak, R., Sack, B. & Wright, J. (2007). The U.S. Treasury yield curve:
  1961 to the present. *Journal of Monetary Economics*, 54(8), 2291–2304.
* Adrian, T., Crump, R. & Moench, E. (2013). Pricing the term structure with
  linear regressions. *Journal of Financial Economics*, 110(1), 110–138.
* Huber, P. (1964). Robust estimation of a location parameter. *Annals of
  Mathematical Statistics*, 35(1), 73–101.
* Nelson, C. & Siegel, A. (1987). Parsimonious modeling of yield curves.
  *Journal of Business*, 60(4), 473–489.
* Svensson, L. (1994). Estimating and interpreting forward interest rates:
  Sweden 1992–1994. NBER Working Paper 4871.
* Board of Governors of the Federal Reserve System. Selected Interest Rates
  (H.15): Treasury constant maturities, via FRED, Federal Reserve Bank of
  St. Louis.
