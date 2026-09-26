"""Model-selection studies on real (or synthetic) Treasury data.

1. **Leave-one-tenor-out cross-validation.** For every date and every tenor,
   fit the curve *without* that tenor and predict it. In-sample RMSE always
   favours the model with more parameters; this measures what a curve is
   actually used for - pricing maturities between the quoted points - and so
   reveals overfitting. Compares Nelson-Siegel, NSS fitted date by date, and
   NSS with the default λ smoothing.
2. **Forecast design.** Diebold-Li forecasts with AR(1) vs VAR(1) factor
   dynamics, on an expanding vs a rolling 10-year estimation window, relative
   to the random walk.
3. **Agreement with the Federal Reserve's curve.** Zero curves fitted from
   CMT quotes are compared with the Fed's independently estimated Svensson
   curve (Gürkaynak, Sack & Wright, 2007) for each way of reading the quotes.
   With ``--source synthetic`` the reference is the known true curve.
4. **Term premium.** The Adrian-Crump-Moench 10-year premium with 3-6
   factors, on this engine's curves and on the Fed's, against Kim-Wright.
5. **State-space dynamic Nelson-Siegel.** Kalman-filter/MLE forecasts
   (Diebold, Rudebusch & Aruoba, 2006), with a mean-reverting or a
   random-walk level, independent factors, and the arbitrage-free AFNS
   restriction (Christensen, Diebold & Rudebusch, 2011), against the random
   walk - point accuracy and the coverage of 80% forecast intervals.

Usage::

    python benchmarks/real_data_studies.py                 # FRED, monthly since 1990
    python benchmarks/real_data_studies.py --source synthetic
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from nss_engine.calibration import (
    DEFAULT_PANEL_SMOOTHING,
    CalibrationConfig,
    calibrate,
    calibrate_panel,
)
from nss_engine.data import (
    DataError,
    load_gsw_parameters,
    load_kim_wright_term_premium,
    load_treasury_yields,
    maturity_label,
)
from nss_engine.forecasting import evaluate_forecasts
from nss_engine.statespace import evaluate_dns_forecasts
from nss_engine.synthetic import simulate_market
from nss_engine.termpremium import (
    compare_term_premia,
    fit_acm,
    real_time_decomposition,
    zero_panel,
)
from nss_engine.validation import compare_to_reference

INTERIOR = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0)


def cross_validate(yields: pd.DataFrame, cfg: CalibrationConfig, panel: bool) -> pd.Series:
    """Out-of-sample RMSE (bp) per held-out tenor."""
    mats = np.asarray(yields.columns, dtype=float)
    # Predict the held-out quote on the same basis the model was fitted on.
    measure = "par" if cfg.target == "par" else "zero"
    errors: dict[float, list[float]] = {t: [] for t in mats}
    for j, held_out in enumerate(mats):
        train = yields.copy()
        train.iloc[:, j] = np.nan
        if panel:
            curves = calibrate_panel(train, cfg).params
            pred = {d: _predict(row, held_out, measure) for d, row in curves.iterrows()}
        else:
            pred = {}
            for d, row in train.iterrows():
                res = calibrate(mats, row.to_numpy(), cfg)
                if res.success:
                    pred[d] = float(res.curve.evaluate(held_out, measure)[0])
        actual = yields.iloc[:, j]
        for d, p in pred.items():
            if np.isfinite(actual.get(d, np.nan)):
                errors[held_out].append((actual[d] - p) * 100)
    return pd.Series(
        {maturity_label(t): float(np.sqrt(np.mean(np.square(e)))) for t, e in errors.items() if e},
        name="cv_rmse_bp",
    )


def _predict(row: pd.Series, tau: float, measure: str) -> float:
    from nss_engine.models import NSSCurve

    return float(NSSCurve.from_mapping(row).evaluate(tau, measure)[0])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default="fred", choices=["fred", "synthetic"])
    ap.add_argument("--start", default="1990-01-01")
    args = ap.parse_args()

    reference = None
    if args.source == "fred":
        monthly = load_treasury_yields(args.start, freq="ME", how="last")
        try:
            reference = load_gsw_parameters(start=args.start)
        except DataError as exc:
            print(f"(GSW reference curve unavailable: {exc})\n")
    else:
        mkt = simulate_market(seed=0)
        monthly = mkt.yields.resample("ME").last()
        reference = mkt.true_params.resample("ME").last()
        reference.index = monthly.index
    print(
        f"Data: {args.source}, {len(monthly)} month-end curves "
        f"{monthly.index[0]:%Y-%m} to {monthly.index[-1]:%Y-%m}\n"
    )

    # ---- 1. cross-validation ------------------------------------------------------
    t0 = time.perf_counter()
    models = {
        "Nelson-Siegel": (CalibrationConfig(model="ns"), False),
        "NSS, each date independent": (CalibrationConfig(), False),
        "NSS + λ smoothing (default)": (
            CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING),
            True,
        ),
        "NSS + λ smoothing, fixed λ2 bound (2.0)": (
            CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING, hump_within_data=False),
            True,
        ),
        "NSS + λ smoothing, zero target": (
            CalibrationConfig(target="yield", lambda_smoothing=DEFAULT_PANEL_SMOOTHING),
            True,
        ),
    }
    cv = pd.DataFrame(
        {name: cross_validate(monthly, cfg, panel) for name, (cfg, panel) in models.items()}
    ).T
    interior = [maturity_label(t) for t in INTERIOR if maturity_label(t) in cv.columns]
    cv.insert(0, "interior mean", cv[interior].mean(axis=1))
    cv.index.name = "model"
    print("## Leave-one-tenor-out cross-validation (out-of-sample RMSE, bp)\n")
    print("'interior mean' averages 3M-20Y; 1M and 30Y are extrapolations.\n")
    print(cv.round(2).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)\n")

    # ---- 2. forecasting design ---------------------------------------------------------
    core = monthly.dropna(axis=1, thresh=int(0.9 * len(monthly))).dropna()
    rows = {}
    for kind in ("ar1", "var1"):
        for window, label in ((None, "expanding"), (120, "rolling 10y")):
            ev = evaluate_forecasts(
                core, horizons=(1, 6, 12), min_train=120, kind=kind, rolling_window=window
            )
            rel = ev.relative_rmse
            rows[f"{kind.upper()}, {label}"] = {f"h={h}m": rel.loc[h].mean() for h in rel.index}
            if window is None:
                comb = ev.relative_rmse_combination
                rows[f"½ {kind.upper()} + ½ random walk"] = {
                    f"h={h}m": comb.loc[h].mean() for h in comb.index
                }
    fc = pd.DataFrame(rows).T
    fc.index.name = "factor model"
    print(
        "## Diebold-Li forecasts: RMSE relative to random walk, averaged over tenors (<1 = better)\n"
    )
    print(f"First forecast origin: {core.index[119]:%Y-%m}; {core.shape[1]} tenors.\n")
    print(fc.round(3).to_markdown())

    # ---- 3. agreement with the Fed's (GSW) curve -------------------------------------
    if reference is not None:
        gsw_study(monthly, reference, args.source)

    # ---- 4. term premium robustness ---------------------------------------------------
    if args.source == "fred":
        term_premium_study(monthly, reference)

    # ---- 5. state-space dynamic Nelson-Siegel ------------------------------------------
    dns_study(core, fc)


def term_premium_study(monthly: pd.DataFrame, reference: pd.DataFrame | None) -> None:
    """ACM 10-year term premium vs Kim-Wright, by number of factors and source curve."""
    try:
        kw = load_kim_wright_term_premium(start=monthly.index[0])
    except DataError as exc:
        print(f"\n(Kim-Wright term premium unavailable: {exc})\n")
        return
    fit = calibrate_panel(monthly, CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING))
    curves = {"NSS curves (this engine)": zero_panel(fit.params)}
    if reference is not None:
        curves["Fed GSW curve"] = zero_panel(reference.loc[monthly.index[0] :])
    rows = {}
    for curve_name, zeros in curves.items():
        for k in (3, 4, 5, 6):
            tp = fit_acm(zeros, n_factors=k).decomposition(10)["term_premium"]
            stats = compare_term_premia(tp, kw)
            rows[(curve_name, k)] = {
                "mean TP (%)": float(tp.mean()),
                "latest TP (%)": float(tp.iloc[-1]),
                **stats,
            }
    table = pd.DataFrame(rows).T.drop(columns="n_months")
    table.index.names = ["curve", "factors"]
    rt_rows = {}
    nss = curves["NSS curves (this engine)"]
    full = fit_acm(nss).decomposition(10)["term_premium"]
    for min_train in (60, 120, 180):
        rt = real_time_decomposition(nss, 10.0, min_train=min_train)
        tp = rt["term_premium"].dropna()
        vs_kw = compare_term_premia(tp, kw)
        rt_rows[f"start after {min_train} months"] = {
            "first estimate": f"{tp.index[0]:%Y-%m}",
            "corr with full sample": float(tp.corr(full.reindex(tp.index))),
            "corr with Kim-Wright": vs_kw["corr_level"],
            "RMSE vs Kim-Wright (bp)": vs_kw["rmse_bp"],
            "sd (%)": float(tp.std()),
            "share VAR capped": float(rt["var_capped"].mean()),
            "discarded": int(rt["term_premium"].isna().sum()),
        }
    print("\n## Term premium (ACM, 10-year) vs Kim-Wright\n")
    print(
        "Correlation of monthly levels and 12-month changes, RMSE and mean gap "
        "(ACM − Kim-Wright, bp). ACM's own choice is 5 factors on the Fed's curve.\n"
    )
    print(table.round(3).to_markdown())
    print("\nPseudo-real-time 10-year premium (5 factors, NSS curves, re-estimated monthly):\n")
    print(pd.DataFrame(rt_rows).T.to_markdown(floatfmt=".3f"))


def dns_study(core: pd.DataFrame, two_step: pd.DataFrame) -> None:
    t0 = time.perf_counter()
    rows, cover = {}, {}
    specs: dict[str, dict[str, bool]] = {
        "VAR(1)": {},
        "VAR(1), random-walk level": {"level_unit_root": True},
        "independent factors": {"independent": True},
        "AFNS, VAR(1)": {"arbitrage_free": True},
        "AFNS, independent factors": {"arbitrage_free": True, "independent": True},
    }
    for label, kwargs in specs.items():
        ev = evaluate_dns_forecasts(
            core, horizons=(1, 6, 12), min_train=120, reestimate_every=12, **kwargs
        )
        rel = ev.relative_rmse
        rows[f"state-space {label}"] = {f"h={h}m": rel.loc[h].mean() for h in rel.index}
        comb = ev.relative_rmse_combination
        rows[f"½ state-space {label} + ½ random walk"] = {
            f"h={h}m": comb.loc[h].mean() for h in comb.index
        }
        cover[f"state-space {label}"] = {f"h={h}m": ev.coverage.loc[h].mean() for h in rel.index}
    table = pd.concat([two_step, pd.DataFrame(rows).T])
    table.index.name = "model"
    print("\n## State-space dynamic Nelson-Siegel (Kalman filter, MLE)\n")
    print(
        "RMSE relative to the random walk, averaged over tenors (<1 = better); parameters "
        "re-estimated every 12 months on data up to each origin. AFNS rows impose "
        "no-arbitrage (Christensen, Diebold & Rudebusch, 2011); 'independent factors' "
        "means diagonal factor dynamics and shocks.\n"
    )
    print(table.round(3).to_markdown())
    print("\nCoverage of 80% forecast intervals (share of outcomes inside):\n")
    cov = pd.DataFrame(cover).T
    cov.index.name = "model"
    print(cov.round(3).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)")


def gsw_study(monthly: pd.DataFrame, reference: pd.DataFrame, source: str) -> None:
    ref = reference.reindex(monthly.index, method="ffill", tolerance=pd.Timedelta(days=4))
    ref = ref.dropna()
    name = "the true curve" if source == "synthetic" else "the Fed's GSW curve"
    configs = {
        "zero target (quotes read as zero rates)": CalibrationConfig(
            target="yield", lambda_smoothing=DEFAULT_PANEL_SMOOTHING
        ),
        "par target (default)": CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING),
        "par target + robust": CalibrationConfig(
            lambda_smoothing=DEFAULT_PANEL_SMOOTHING, robust=True
        ),
        "par target, fixed λ2 bound (2.0)": CalibrationConfig(
            lambda_smoothing=DEFAULT_PANEL_SMOOTHING, hump_within_data=False
        ),
    }
    # Months without a 30-year quote (the Treasury suspended the bond 2002-2006):
    # the fitted long end is an extrapolation from the 20-year.
    no_long = monthly.index[monthly[monthly.columns.max()].isna()]
    rows, bias_rows, gap_rows = {}, {}, {}
    for label, cfg in configs.items():
        fit = calibrate_panel(monthly, cfg)
        cmp = compare_to_reference(fit, ref)
        rows[label] = cmp.overall()
        bias_rows[label] = cmp.summary()["bias_bp"]
        gap = cmp.subset(no_long)
        if gap.n_dates:
            gap_rows[label] = gap.summary()["rmse_bp"]
    print(f"\n## Zero curves vs {name}, 1Y-30Y ({int(rows[label]['n_dates'])} month-ends)\n")
    print(
        "RMSE includes any constant offset; 'demeaned' removes each maturity's average gap; "
        "'change corr' is the correlation of monthly changes.\n"
    )
    print(pd.DataFrame(rows).T.drop(columns="n_dates").round(3).to_markdown())
    print("\nAverage gap (engine − reference, bp) by maturity:\n")
    print(pd.DataFrame(bias_rows).T.round(1).to_markdown())
    if gap_rows:
        n = len(ref.index.intersection(no_long))
        print(f"\nRMSE (bp) in the {n} months without a 30-year quote:\n")
        print(pd.DataFrame(gap_rows).T.round(1).to_markdown())
    else:
        print("\n(No month in this sample lacks a 30-year quote.)")


if __name__ == "__main__":
    main()
