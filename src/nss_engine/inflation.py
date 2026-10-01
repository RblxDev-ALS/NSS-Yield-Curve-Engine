"""Real yields and breakeven inflation from the nominal and TIPS curves.

Treasury Inflation-Protected Securities (TIPS) pay a real coupon on a
principal indexed to CPI, so their yields are *real* yields. The gap between a
nominal and a real yield of the same maturity is **breakeven inflation**: the
inflation rate at which the two bonds return the same. It is the market's
inflation compensation, which equals expected inflation plus an inflation risk
premium minus a TIPS liquidity premium, so it is not a pure forecast.

FRED publishes breakevens as simple differences of constant-maturity par
yields (``T5YIE = DGS5 − DFII5``) and a 5-year, 5-year forward rate computed
from those two par differences (``T5YIFR``). A par-yield difference mixes the
coupons of two different bonds and is not a point on any zero or forward
curve. Here both curves are fitted first (the nominal curve with NSS, the real
curve with the same calibrator) and breakevens are read off the zero curves:

* zero-coupon breakeven ``b(τ) = z_nominal(τ) − z_real(τ)`` (continuous
  compounding), and
* the 5y5y forward breakeven ``(10·b(10) − 5·b(5)) / 5``, the average
  instantaneous forward breakeven between 5 and 10 years.

**Few quotes.** The TIPS curve has only four (2003-2010) or five quotes, from
5 to 30 years. Nelson-Siegel-Svensson's six parameters would outnumber them,
and the calibrator falls back to Nelson-Siegel (four parameters) below seven
quotes anyway. :func:`real_curve_config` also keeps the curvature hump inside
the quoted range; the benchmark ``benchmarks/breakeven_known_truth.py`` checks
the choice against a simulated market with a known breakeven curve. Nothing
below 5 years is quoted, so short real rates and breakevens are
extrapolations and are not reported.

Known simplifications: TIPS accrue with a 3-month CPI indexation lag, CPI is
seasonal, and the deflation floor has value when inflation is low. None of
these is modelled; they matter mostly at short maturities, which are not
reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .calibration import DEFAULT_PANEL_SMOOTHING, CalibrationConfig, PanelFit, calibrate_panel
from .models import NSSCurve

#: Maturities (years) of FRED's TIPS constant-maturity series.
TIPS_MATURITIES = (5.0, 7.0, 10.0, 20.0, 30.0)

#: Bounds on the real curve's decay rate: the curvature hump peaks between
#: 1.79/0.6 ≈ 3 and 1.79/0.06 ≈ 30 years, inside or next to the quoted range.
REAL_LAMBDA_BOUNDS = (0.06, 0.6)


def real_curve_config(smoothing: float = DEFAULT_PANEL_SMOOTHING) -> CalibrationConfig:
    """Calibration settings for the TIPS real curve: Nelson-Siegel on par yields."""
    return CalibrationConfig(
        model="ns",
        lambda1_bounds=REAL_LAMBDA_BOUNDS,
        lambda_smoothing=smoothing,
        min_points=4,
    )


def fit_real_curve(real_yields: pd.DataFrame, config: CalibrationConfig | None = None) -> PanelFit:
    """Fit the real (TIPS) curve on every date of a panel of real par yields."""
    return calibrate_panel(real_yields, config or real_curve_config())


def _aligned(
    nominal: pd.DataFrame, real: pd.DataFrame, tolerance_days: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nom = nominal.sort_index()
    rl = real.sort_index()
    nom = nom.reindex(rl.index, method="ffill", tolerance=pd.Timedelta(days=tolerance_days))
    ok = nom.notna().all(axis=1) & rl.notna().all(axis=1)
    return nom[ok], rl[ok]


def breakevens(
    nominal_params: pd.DataFrame,
    real_params: pd.DataFrame,
    maturities: Sequence[float] = (5.0, 10.0),
) -> pd.DataFrame:
    """Breakeven inflation from nominal and real NSS parameters (percent).

    Both frames have the columns ``beta0 … lambda2`` (e.g. the ``params`` of
    two :class:`~nss_engine.PanelFit`); they are aligned on the real curve's
    dates. Returns, per date, the zero-coupon breakeven ``be_<m>y`` for each
    maturity, the 5y5y forward breakeven ``be_5y5y`` (when 5 and 10 are among
    the maturities), the par-yield breakeven ``be_par_<m>y`` (the model's
    nominal minus real par yield, the quantity FRED's ``T5YIE``/``T10YIE``
    measure) and the real zero yield ``real_<m>y``.
    """
    nom, rl = _aligned(nominal_params, real_params)
    mats = np.asarray(list(maturities), dtype=float)
    rows = []
    for (_, n), (_, r) in zip(nom.iterrows(), rl.iterrows(), strict=True):
        cn, cr = NSSCurve.from_mapping(n), NSSCurve.from_mapping(r)
        zn, zr = cn.zero(mats), cr.zero(mats)
        pn, pr = cn.par_yield(mats), cr.par_yield(mats)
        rows.append(np.concatenate([zn - zr, pn - pr, zr]))
    labels = [f"{m:g}y" for m in mats]
    cols = [f"be_{m}" for m in labels] + [f"be_par_{m}" for m in labels]
    cols += [f"real_{m}" for m in labels]
    out = pd.DataFrame(rows, index=rl.index, columns=cols)
    if 5.0 in mats and 10.0 in mats:
        out.insert(len(mats), "be_5y5y", (10.0 * out["be_10y"] - 5.0 * out["be_5y"]) / 5.0)
    out.index.name = "date"
    return out


def fred_style_breakevens(nominal_quotes: pd.DataFrame, real_quotes: pd.DataFrame) -> pd.DataFrame:
    """FRED's formulas applied to quotes: par-yield differences and their 5y5y forward.

    ``T5YIE = DGS5 − DFII5``, ``T10YIE = DGS10 − DFII10`` and
    ``T5YIFR = ((1 + T10YIE/100)^10 / (1 + T5YIE/100)^5)^(1/5) − 1``, in
    percent. Useful to compare the curve-based estimates with FRED's on
    simulated data, where the truth is known.
    """
    nom, rl = nominal_quotes.sort_index(), real_quotes.sort_index()
    nom = nom.reindex(rl.index, method="ffill", tolerance=pd.Timedelta(days=4))
    be5 = nom[5.0] - rl[5.0]
    be10 = nom[10.0] - rl[10.0]
    fwd = (((1 + be10 / 100) ** 10 / (1 + be5 / 100) ** 5) ** 0.2 - 1) * 100
    out = pd.DataFrame({"T5YIE": be5, "T10YIE": be10, "T5YIFR": fwd}).dropna()
    out.index.name = "date"
    return out


def compare_series(estimate: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Agreement of two daily or weekly series, on month-end values (percent).

    Returns the correlation of levels and of monthly changes, the RMSE and
    mean gap (estimate − benchmark, bp) and the number of common months.
    """
    a = estimate.dropna().copy()
    b = benchmark.dropna().copy()
    a.index = pd.DatetimeIndex(a.index).to_period("M")
    b.index = pd.DatetimeIndex(b.index).to_period("M")
    a = a.groupby(level=0).last()
    b = b.groupby(level=0).last()
    both = pd.concat([a.rename("est"), b.rename("ref")], axis=1).dropna()
    if len(both) < 24:
        raise ValueError("fewer than 24 common months")
    d1 = both.diff().dropna()
    gap = both["est"] - both["ref"]
    return {
        "corr_level": float(both["est"].corr(both["ref"])),
        "corr_change_1m": float(d1["est"].corr(d1["ref"])),
        "rmse_bp": float(np.sqrt(np.mean(gap**2)) * 100),
        "mean_gap_bp": float(gap.mean() * 100),
        "n_months": float(len(both)),
    }
