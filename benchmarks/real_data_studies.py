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
5. **Which part of the slope predicts recessions?** The 10Y−3M spread split
   into expected short rates and term premium, both estimated in real time,
   with and without survey anchors (Rosenberg & Maurer, 2008).
6. **Breakeven inflation.** The TIPS real curve and breakevens against FRED's
   T5YIE / T10YIE / T5YIFR and the Fed's own TIPS curve (Gürkaynak, Sack &
   Wright, 2010).
7. **State-space dynamic Nelson-Siegel.** Kalman-filter/MLE forecasts
   (Diebold, Rudebusch & Aruoba, 2006), with a mean-reverting or a
   random-walk level, independent factors, and the arbitrage-free AFNS
   restriction (Christensen, Diebold & Rudebusch, 2011), against the random
   walk - point accuracy and the coverage of 80% forecast intervals.
8. **Replicating the New York Fed's ACM term premium.** This package's ACM
   code on the Fed's curve since 1961 (as ACM do), and on this engine's
   curves since 1990, against the series the New York Fed publishes.
9. **Do term premia predict bond returns?** One-year excess returns on 2-,
   5- and 10-year zero-coupon bonds against the returns that ACM (plain and
   survey-anchored) expected in real time, Fama-Bliss and Cochrane-Piazzesi
   regressions, and the historical mean: out-of-sample R², Clark-West tests
   and Mincer-Zarnowitz slopes.

Usage::

    python benchmarks/real_data_studies.py                 # FRED, monthly since 1990
    python benchmarks/real_data_studies.py --source synthetic
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd

from nss_engine import returns as bond_returns
from nss_engine.calibration import (
    DEFAULT_PANEL_SMOOTHING,
    CalibrationConfig,
    calibrate,
    calibrate_panel,
)
from nss_engine.data import (
    DataError,
    load_acm_term_premium,
    load_breakevens,
    load_gsw_parameters,
    load_gsw_tips_parameters,
    load_kim_wright_term_premium,
    load_recession_indicator,
    load_spf_bill_forecasts,
    load_tips_yields,
    load_treasury_yields,
    maturity_label,
)
from nss_engine.forecasting import compare_forecasts, evaluate_forecasts, hac_mean_test
from nss_engine.inflation import (
    breakevens,
    compare_series,
    fit_real_curve,
    fred_style_breakevens,
)
from nss_engine.regime import compare_recession_predictors
from nss_engine.statespace import DNSForecastEvaluation, evaluate_dns_forecasts
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

    # ---- 4. term premium robustness, and 5. its recession test --------------------------
    if args.source == "fred":
        real_time = term_premium_study(monthly, reference)
        if real_time:
            recession_split_study(monthly, real_time, args.start)

    # ---- 6. breakeven inflation -----------------------------------------------------------
    if args.source == "fred":
        breakeven_study(args.start)

    # ---- 5. state-space dynamic Nelson-Siegel ------------------------------------------
    dns_study(core, fc)


def term_premium_study(
    monthly: pd.DataFrame, reference: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """ACM 10-year term premium vs Kim-Wright, by number of factors and source curve.

    Returns the real-time decompositions started after 60 months, by method.
    """
    try:
        kw = load_kim_wright_term_premium(start=monthly.index[0])
    except DataError as exc:
        print(f"\n(Kim-Wright term premium unavailable: {exc})\n")
        return {}
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
    print("\n## Term premium (ACM, 10-year) vs Kim-Wright\n")
    print(
        "Correlation of monthly levels and 12-month changes, RMSE and mean gap "
        "(ACM − Kim-Wright, bp). ACM's own choice is 5 factors on the Fed's curve.\n"
    )
    print(table.round(3).to_markdown())
    first_rt = anchored_term_premium_study(curves["NSS curves (this engine)"], kw)
    spf = _surveys_or_none()
    acm_replication_study(curves["NSS curves (this engine)"], kw, spf)
    return_predictability_study(curves["NSS curves (this engine)"], kw, spf)
    return first_rt


def _surveys_or_none() -> pd.DataFrame | None:
    try:
        return load_spf_bill_forecasts()
    except (DataError, ImportError):
        return None


def _month_end(s: pd.Series) -> pd.Series:
    out = s.groupby(pd.DatetimeIndex(s.index).to_period("M")).last()
    out.index = out.index.to_timestamp("M")
    return out


def acm_replication_study(nss: pd.DataFrame, kw: pd.Series, spf: pd.DataFrame | None) -> None:
    """This package's ACM code against the New York Fed's published ACM term premium."""
    t0 = time.perf_counter()
    try:
        nyfed = load_acm_term_premium()
    except (DataError, ImportError) as exc:
        print(f"\n(New York Fed ACM term premium unavailable: {exc})\n")
        return
    ny_tp = _month_end(nyfed["term_premium"].dropna())
    ny_rn = (
        _month_end(nyfed["expected_short_rate"].dropna())
        if "expected_short_rate" in nyfed
        else None
    )
    print(
        "\n## Replicating the New York Fed's ACM term premium (10-year)\n\n"
        f"New York Fed series: {ny_tp.index[0]:%Y-%m} to {ny_tp.index[-1]:%Y-%m}, "
        f"mean {ny_tp.loc[nss.index[0] :].mean():.2f}% since {nss.index[0]:%Y-%m}. "
        "All rows are compared over the months since then; gap = row − New York Fed.\n"
    )
    estimates: dict[str, tuple[pd.Series, pd.Series | None]] = {}
    try:
        gsw_all = load_gsw_parameters()
    except DataError as exc:
        print(f"(GSW curve unavailable: {exc})")
        gsw_all = None
    if gsw_all is not None:
        long = zero_panel(gsw_all).dropna()
        for label, zeros in (
            (f"ACM code, Fed curve since {long.index[0]:%Y} (ACM's design)", long),
            ("ACM code, Fed curve since 1990", long.loc[nss.index[0] :]),
        ):
            if len(zeros) < 120:
                continue
            d = fit_acm(zeros).decomposition(10)
            estimates[label] = (d["term_premium"], d["expected_short_rate"])
    d = fit_acm(nss).decomposition(10)
    estimates["ACM code, this engine's curves since 1990"] = (
        d["term_premium"],
        d["expected_short_rate"],
    )
    if spf is not None:
        d = fit_acm(nss, surveys=spf).decomposition(10)
        estimates["survey-anchored, this engine's curves"] = (
            d["term_premium"],
            d["expected_short_rate"],
        )
    estimates["Kim-Wright"] = (kw, None)
    rows = {}
    start = nss.index[0]
    for label, (tp, rn) in estimates.items():
        stats = compare_term_premia(tp.loc[start:], ny_tp.loc[start:])
        row = {
            "mean TP (%)": float(tp.loc[start:].mean()),
            "corr": stats["corr_level"],
            "corr 12m chg": stats["corr_change_12m"],
            "RMSE (bp)": stats["rmse_bp"],
            "mean gap (bp)": stats["mean_gap_bp"],
            "months": int(stats["n_months"]),
        }
        if rn is not None and ny_rn is not None:
            row["RMSE exp. short rate (bp)"] = compare_term_premia(
                rn.loc[start:], ny_rn.loc[start:]
            )["rmse_bp"]
        rows[label] = row
    print(pd.DataFrame(rows).T.to_markdown(floatfmt=".3f"))
    print(f"\n({time.perf_counter() - t0:.0f} s)")


def return_predictability_study(
    nss: pd.DataFrame, kw: pd.Series, spf: pd.DataFrame | None, horizon: int = 12
) -> None:
    """Do the term premium estimates predict the excess returns bonds went on to earn?"""
    t0 = time.perf_counter()
    mats = (24, 60, 120)
    rx = bond_returns.excess_returns(nss, horizon, (24, 36, 48, 60, 120))
    fwd = bond_returns.forward_rates(nss)
    cp = bond_returns.cochrane_piazzesi_forecasts(rx, fwd, horizon)
    kw_m = _month_end(kw).reindex(nss.index)
    rt: dict[str, pd.DataFrame] = {
        "ACM, plain": bond_returns.real_time_expected_returns(nss, horizon, mats, min_train=60)
    }
    if spf is not None:
        rt["ACM, survey-anchored"] = bond_returns.real_time_expected_returns(
            nss, horizon, mats, min_train=60, surveys=spf
        )
    full = {"ACM, plain": fit_acm(nss).expected_excess_returns(horizon, mats)}
    if spf is not None:
        full["ACM, survey-anchored"] = fit_acm(nss, surveys=spf).expected_excess_returns(
            horizon, mats
        )
    print(
        "\n## Do term premia predict bond returns?\n\n"
        f"Realized {horizon}-month log excess returns on zero-coupon bonds (this engine's curves) "
        "against forecasts made at the start of each holding period with data available then. "
        "Benchmark: the historical mean of returns completed by then. R² OOS > 0 beats it; "
        "Clark-West p-values are one-sided; slope = realized on forecast (1 = calibrated), "
        "Newey-West standard errors for overlapping returns. Kim-Wright's parameters are "
        "estimated on the full sample, so its row is not strictly real time.\n"
    )
    rows = {}
    for n in mats:
        col = n / 12.0
        y = rx[col]
        bench = bond_returns.real_time_regression_forecasts(y, None, horizon)
        fcs: dict[str, pd.Series] = {}
        for name, frame in rt.items():
            fcs[f"{name}: model's expected return"] = frame[col]
            fcs[f"{name}: regression on it"] = bond_returns.real_time_regression_forecasts(
                y, frame[[col]], horizon
            )
            fcs[f"{name}: regression on 10Y term premium"] = (
                bond_returns.real_time_regression_forecasts(y, frame[["term_premium"]], horizon)
            )
        fcs["Fama-Bliss forward spread"] = bond_returns.real_time_regression_forecasts(
            y, bond_returns.forward_spot_spread(nss, n, horizon).to_frame(), horizon
        )
        fcs["Cochrane-Piazzesi factor"] = cp[col]
        fcs["Kim-Wright 10Y term premium (regression)"] = (
            bond_returns.real_time_regression_forecasts(y, kw_m.to_frame(), horizon)
        )
        # score every forecast on the same origins
        common = pd.concat([y, bench, *fcs.values()], axis=1).dropna().index
        for name, f in fcs.items():
            sc = bond_returns.evaluate_return_forecasts(
                y.loc[common], f.loc[common], bench.loc[common], horizon
            )
            half = common[len(common) // 2]
            early, late = (
                bond_returns.evaluate_return_forecasts(
                    y.loc[idx], f.loc[idx], bench.loc[idx], horizon
                ).r2_oos
                for idx in (common[common < half], common[common >= half])
            )
            rows[(f"{n // 12}Y", name)] = {
                "R² OOS (%)": 100 * sc.r2_oos,
                "1st half": 100 * early,
                "2nd half": 100 * late,
                "CW p": sc.p_value,
                "slope": sc.mz_slope,
                "slope se": sc.mz_slope_se,
                "mean fcst (%)": sc.mean_forecast,
                "mean realized (%)": sc.mean_realized,
                "origins": sc.n,
            }
        first, last = common[0], common[-1]
    table = pd.DataFrame(rows).T
    table.index.names = ["bond", "forecast"]
    print(
        f"Forecast origins {first:%Y-%m} to {last:%Y-%m}; the halves split them at {half:%Y-%m}.\n"
    )
    print(table.to_markdown(floatfmt=".3f"))
    insample = {}
    for n in mats:
        col = n / 12.0
        y = rx[col]
        row = {}
        for name, er in full.items():
            both = pd.concat([y, er[col]], axis=1).dropna()
            row[f"{name} (full-sample model)"] = float(both.corr().iloc[0, 1] ** 2)
        both = pd.concat([y, fwd], axis=1).dropna()
        Z = np.column_stack([np.ones(len(both)), both[fwd.columns].to_numpy()])
        coef, *_ = np.linalg.lstsq(Z, both[col].to_numpy(), rcond=None)
        resid = both[col].to_numpy() - Z @ coef
        row["Cochrane-Piazzesi forwards (in sample)"] = float(
            1 - resid.var() / both[col].to_numpy().var()
        )
        insample[f"{n // 12}Y"] = row
    print("\nIn-sample R² for comparison (estimated on the whole sample, so with look-ahead):\n")
    print(pd.DataFrame(insample).round(3).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)")


def anchored_term_premium_study(nss: pd.DataFrame, kw: pd.Series) -> dict[str, pd.DataFrame]:
    """Real-world dynamics from OLS, bias-corrected OLS or survey anchors, full and real time."""
    t0 = time.perf_counter()
    try:
        spf = load_spf_bill_forecasts()
    except (DataError, ImportError) as exc:
        print(f"\n(SPF survey forecasts unavailable: {exc})\n")
        spf = None
    if spf is not None:
        first = spf.groupby("series")["date"].min().dt.strftime("%Y-%m")
        print(
            f"\nSPF 3-month bill forecasts: {len(spf)} forecasts, "
            f"{spf['date'].min():%Y-%m} to {spf['date'].max():%Y-%m}; first per series: "
            + ", ".join(f"{k} {v}" for k, v in first.items())
        )
    methods: dict[str, dict[str, Any]] = {
        "OLS (ACM)": {},
        "bias-corrected, analytic": {"bias_correction": "analytic"},
        "bias-corrected, bootstrap (BRW)": {"bias_correction": "bootstrap"},
    }
    if spf is not None:
        methods["survey-anchored (SPF)"] = {"surveys": spf}
    full, rows = {}, {}
    for name, kwargs in methods.items():
        res = fit_acm(nss, **kwargs)
        tp = res.decomposition(10)["term_premium"]
        full[name] = tp
        stats = compare_term_premia(tp, kw)
        rows[name] = {
            "mean TP (%)": float(tp.mean()),
            "sd TP (%)": float(tp.std()),
            "latest TP (%)": float(tp.iloc[-1]),
            "max root": res.max_eigenvalue,
            "corr KW": stats["corr_level"],
            "corr KW 12m chg": stats["corr_change_12m"],
            "RMSE KW (bp)": stats["rmse_bp"],
            "mean gap KW (bp)": stats["mean_gap_bp"],
        }
        if res.survey_fit is not None:
            print(f"\nSurvey fit, {name} (RMSE by series, pp):")
            print(res.survey_rmse.round(3).to_markdown())
    print("\n## Term premium: real-world dynamics (full sample, 10-year, NSS curves)\n")
    print(pd.DataFrame(rows).T.round(3).to_markdown())

    if spf is not None:
        # model-free: 10-year zero yield minus the survey's 10-year average bill rate
        bill10 = spf[spf["series"] == "BILL10"].set_index("date")["value"]
        y10 = nss[_ten_year(nss)]
        months = y10.index.to_period("M")
        y10m = pd.Series(y10.to_numpy(), index=months)
        b10 = pd.Series(bill10.to_numpy(), index=bill10.index.to_period("M"))
        common = b10.index.intersection(y10m.index)
        if len(common) >= 10:
            survey_tp = (y10m[common] - b10[common]).rename("survey TP")
            kw_m = kw.groupby(kw.index.to_period("M")).mean()
            comp = {"Kim-Wright": kw_m}
            comp.update(
                {k: pd.Series(v.to_numpy(), index=v.index.to_period("M")) for k, v in full.items()}
            )
            srow = {}
            for k, v in comp.items():
                both = pd.concat([survey_tp, v.rename("x")], axis=1).dropna()
                srow[k] = {
                    "corr": float(both.iloc[:, 0].corr(both["x"])),
                    "mean gap (bp)": float((both["x"] - both.iloc[:, 0]).mean() * 100),
                    "n": len(both),
                }
            print(
                "\nModel-free survey premium (10Y zero yield − SPF BILL10, each first quarter, "
                f"{common.min()} to {common.max()}, mean {survey_tp.mean():.2f}%): agreement of "
                "each estimate with it (gap = estimate − survey premium)\n"
            )
            print(pd.DataFrame(srow).T.round(3).to_markdown())

    rt_rows = {}
    first_rt: dict[str, pd.DataFrame] = {}
    for min_train in (60, 120, 180):
        for name, kwargs in methods.items():
            if kwargs.get("bias_correction") == "bootstrap":
                continue  # ~5x slower than the analytic correction, and no better on known truth
            rt = real_time_decomposition(nss, 10.0, min_train=min_train, **kwargs)
            if min_train == 60:
                first_rt[name] = rt
            tp = rt["term_premium"].dropna()
            vs_kw = compare_term_premia(tp, kw)
            own = full[name].reindex(tp.index)
            ols = full["OLS (ACM)"].reindex(tp.index)
            rt_rows[(f"after {min_train} months ({tp.index[0]:%Y-%m})", name)] = {
                "corr own full sample": float(tp.corr(own)),
                "RMSE own full sample (bp)": float(np.sqrt(((tp - own) ** 2).mean()) * 100),
                "corr OLS full sample": float(tp.corr(ols)),
                "corr KW": vs_kw["corr_level"],
                "corr KW 12m chg": vs_kw["corr_change_12m"],
                "RMSE KW (bp)": vs_kw["rmse_bp"],
                "sd (%)": float(tp.std()),
                "share capped": float(rt["var_capped"].mean()),
                "discarded": int(rt["term_premium"].isna().sum()),
            }
    print("\n## Real-time 10-year term premium, re-estimated monthly on past data\n")
    print(
        "Each estimate uses only yields (and surveys) published by then. 'own full sample' "
        "is the same method estimated on all data.\n"
    )
    table = pd.DataFrame(rt_rows).T
    table.index.names = ["first estimate", "real-world dynamics"]
    print(table.to_markdown(floatfmt=".3f"))
    print(f"\n({time.perf_counter() - t0:.0f} s)")
    return first_rt


def recession_split_study(
    monthly: pd.DataFrame, real_time: dict[str, pd.DataFrame], start: str
) -> None:
    """Rosenberg & Maurer (2008): expectations component vs term premium as recession signals.

    The 10Y−3M spread (month-end CMT quotes) is split into the 10-year term
    premium and the rest, the expectations component, using the real-time
    decompositions, so no split uses data from after its date. Every
    candidate is scored in pseudo-real time on the same forecast months.
    """
    try:
        rec = load_recession_indicator(start)
    except DataError as exc:
        print(f"\n(NBER recession dates unavailable: {exc})\n")
        return
    spread = (monthly[10.0] - monthly[0.25]).rename("spread")
    spread.index = spread.index.to_period("M").to_timestamp("M")
    candidates: dict[str, pd.Series | pd.DataFrame] = {}
    labels = {"OLS (ACM)": "plain ACM", "survey-anchored (SPF)": "survey-anchored"}
    frames = {}
    for name, rt in real_time.items():
        if name not in labels:
            continue
        tp = rt["term_premium"].copy()
        tp.index = pd.DatetimeIndex(tp.index).to_period("M").to_timestamp("M")
        frames[labels[name]] = tp
    if not frames:
        return
    both = pd.concat([spread, *[v.rename(k) for k, v in frames.items()]], axis=1).dropna()
    candidates["10Y−3M spread"] = both["spread"]
    for label in frames:
        candidates[f"expectations component ({label})"] = both["spread"] - both[label]
        candidates[f"term premium ({label})"] = both[label]
        candidates[f"both parts ({label})"] = pd.DataFrame(
            {"exp": both["spread"] - both[label], "tp": both[label]}
        )
    for lag in (0, 12):
        try:
            table = compare_recession_predictors(candidates, rec, 12, publication_lag=lag)
        except ValueError as exc:
            print(f"\n(recession split test failed: {exc})")
            return
        cols = [
            "auc_in_sample",
            "auc_out_of_sample",
            "log_score_out_of_sample",
            "brier_out_of_sample",
            "auc_gain_vs_first",
            "auc_gain_lo90",
            "auc_gain_hi90",
        ]
        when = "" if lag == 0 else ", NBER dates known only 12 months late"
        print(f"\n## Which part of the slope predicts recessions? (real-time split{when})\n")
        print(
            "Rosenberg & Maurer (2008): spread = expectations component + 10-year term premium, "
            "both from the ACM model re-estimated every month on past data (first estimate after "
            f"60 months). Probits 12 months ahead, pseudo-real time, "
            f"{int(table['n_forecasts'].iloc[0])} common forecast months "
            f"from {both.index[0] + pd.offsets.MonthEnd(120):%Y-%m} or later; AUC gains vs the "
            "spread with 90% block-bootstrap intervals.\n"
        )
        print(table[cols].round(3).to_markdown())


def breakeven_study(start: str) -> None:
    """TIPS real curve and breakeven inflation vs FRED's measures and the Fed's TIPS curve."""
    t0 = time.perf_counter()
    try:
        real = load_tips_yields(max(pd.Timestamp(start), pd.Timestamp("2003-01-01")), freq="W-FRI")
        nominal = load_treasury_yields(real.index[0] - pd.Timedelta(days=7), freq="W-FRI")
    except DataError as exc:
        print(f"\n(TIPS yields unavailable: {exc})\n")
        return
    nom_fit = calibrate_panel(nominal)
    real_fit = fit_real_curve(real)
    be = breakevens(nom_fit.params, real_fit.params)
    diag = real_fit.diagnostics
    print(
        f"\n## Breakeven inflation ({len(be)} weekly curves, {be.index[0]:%Y-%m} to "
        f"{be.index[-1]:%Y-%m})\n"
    )
    print(
        f"TIPS real curve (Nelson-Siegel on 4-5 par yields): median fit RMSE "
        f"{diag['rmse_bp'].median():.2f} bp, 95th percentile {diag['rmse_bp'].quantile(0.95):.2f} bp, "
        f"{diag['success'].mean():.1%} converged.\n"
    )
    last = be.iloc[-1]
    print(
        f"Latest ({be.index[-1]:%Y-%m-%d}): real 10Y {last['real_10y']:.2f}%, breakevens "
        f"5Y {last['be_5y']:.2f}%, 10Y {last['be_10y']:.2f}%, 5y5y {last['be_5y5y']:.2f}%; "
        f"par-yield breakevens 5Y {last['be_par_5y']:.2f}%, 10Y {last['be_par_10y']:.2f}%.\n"
    )
    refs: dict[str, pd.Series] = {}
    try:
        fred = load_breakevens(be.index[0])
        refs.update({f"FRED {c}": fred[c] for c in fred.columns})
    except DataError as exc:
        print(f"(FRED breakevens unavailable: {exc})")
    try:
        gsw_n = load_gsw_parameters(start=be.index[0])
        gsw_r = load_gsw_tips_parameters(start=be.index[0])
        gsw_be = breakevens(gsw_n, gsw_r)
        refs.update({f"Fed GSW {c}": gsw_be[c] for c in ("be_5y", "be_10y", "be_5y5y")})
    except (DataError, ValueError) as exc:
        print(f"(Fed TIPS curve unavailable: {exc})")
    quotes = fred_style_breakevens(nominal, real)
    pairs = {
        ("engine be_par_5y", "FRED T5YIE"): (be["be_par_5y"], "FRED T5YIE"),
        ("engine be_par_10y", "FRED T10YIE"): (be["be_par_10y"], "FRED T10YIE"),
        ("FRED formula on weekly quotes (T5YIFR)", "FRED T5YIFR"): (
            quotes["T5YIFR"],
            "FRED T5YIFR",
        ),
        ("engine be_5y5y", "FRED T5YIFR"): (be["be_5y5y"], "FRED T5YIFR"),
        ("engine be_5y", "Fed GSW be_5y"): (be["be_5y"], "Fed GSW be_5y"),
        ("engine be_10y", "Fed GSW be_10y"): (be["be_10y"], "Fed GSW be_10y"),
        ("engine be_5y5y", "Fed GSW be_5y5y"): (be["be_5y5y"], "Fed GSW be_5y5y"),
        ("FRED T5YIE", "Fed GSW be_5y"): ("FRED T5YIE", "Fed GSW be_5y"),
        ("FRED T10YIE", "Fed GSW be_10y"): ("FRED T10YIE", "Fed GSW be_10y"),
        ("FRED T5YIFR", "Fed GSW be_5y5y"): ("FRED T5YIFR", "Fed GSW be_5y5y"),
    }
    rows = {}
    for key, (est, ref) in pairs.items():
        e = refs.get(est) if isinstance(est, str) else est
        if e is None or ref not in refs:
            continue
        try:
            rows[key] = compare_series(e, refs[ref])
        except ValueError:
            continue
    if rows:
        table = pd.DataFrame(rows).T
        table.index.names = ["estimate", "benchmark"]
        print(
            "Agreement on month-end values (RMSE and mean gap in bp; gap = estimate − benchmark). "
            "`be_par` is the engine's nominal minus real *par* yield, what FRED's T5YIE/T10YIE "
            "measure; `be_5y`, `be_10y` are zero-coupon breakevens; the Fed GSW rows subtract the "
            "Fed's TIPS zero curve from its nominal one.\n"
        )
        print(table.round(3).to_markdown())
    print(f"\n({time.perf_counter() - t0:.0f} s)")


def _ten_year(zeros: pd.DataFrame) -> float:
    cols = np.asarray(zeros.columns, dtype=float)
    return float(cols[np.argmin(np.abs(cols - 10.0))])


def dns_study(core: pd.DataFrame, two_step: pd.DataFrame) -> None:
    t0 = time.perf_counter()
    rows, cover, evs = {}, {}, {}
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
        evs[label] = ev
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
    significance_study(evs)
    print(f"\n({time.perf_counter() - t0:.0f} s)")


def significance_study(evs: dict[str, DNSForecastEvaluation]) -> None:
    """Diebold-Mariano tests between forecasts, and HAC tests of interval coverage."""
    pairs = {
        "AFNS vs state-space, VAR(1)": (("AFNS, VAR(1)", "model"), ("VAR(1)", "model")),
        "state-space VAR(1) vs random walk": (("VAR(1)", "model"), ("VAR(1)", "random_walk")),
        "AFNS VAR(1) vs random walk": (("AFNS, VAR(1)", "model"), ("VAR(1)", "random_walk")),
        "½ state-space VAR(1) + ½ RW vs RW": (
            ("VAR(1)", "combination"),
            ("VAR(1)", "random_walk"),
        ),
        "½ AFNS VAR(1) + ½ RW vs RW": (
            ("AFNS, VAR(1)", "combination"),
            ("VAR(1)", "random_walk"),
        ),
    }
    horizons = list(evs["VAR(1)"].rmse_model.index)
    rows = {}
    for name, ((a, fa), (b, fb)) in pairs.items():
        for h in horizons:
            res = compare_forecasts(evs[a].errors(h, fa), evs[b].errors(h, fb), h)
            rows[(name, f"{h}m")] = res
    table = pd.DataFrame(rows).T
    table.index.names = ["A vs B", "horizon"]
    print("\n## Are the differences significant? Diebold-Mariano tests\n")
    print(
        "Loss = squared error averaged over tenors at each origin; RMSE in bp; "
        "DM statistic with the Harvey-Leybourne-Newbold correction "
        "(negative = A more accurate), two-sided p-value.\n"
    )
    print(table.drop(columns="n").round(3).to_markdown())
    print(f"\n({int(table['n'].min())}-{int(table['n'].max())} forecast origins per test)")

    cov_rows = {}
    for h in horizons:
        per_origin = {k: evs[k].inside[h].mean(axis=1) for k in ("VAR(1)", "AFNS, VAR(1)")}
        for k, x in per_origin.items():
            t = hac_mean_test(x.to_numpy(), value=evs[k].interval, lags=max(h - 1, 6))
            cov_rows[(f"state-space {k}", f"{h}m")] = t
        diff = (per_origin["AFNS, VAR(1)"] - per_origin["VAR(1)"]).dropna()
        cov_rows[("AFNS − state-space", f"{h}m")] = hac_mean_test(
            diff.to_numpy(), 0.0, lags=max(h - 1, 6)
        )
    print(
        "\nCoverage of 80% intervals: mean, Newey-West s.e., test against 80% (or 0 for the gap)\n"
    )
    ct = pd.DataFrame(cov_rows).T
    ct.index.names = ["model", "horizon"]
    print(ct.round(3).to_markdown())


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
