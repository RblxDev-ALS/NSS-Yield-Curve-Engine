"""Plain-text / Markdown rendering of pipeline results."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import maturity_label

if TYPE_CHECKING:
    from .pipeline import PipelineResult


def _table(df: pd.DataFrame | pd.Series, floatfmt: str = ".2f") -> str:
    """Minimal GitHub-flavoured Markdown table (no ``tabulate`` dependency)."""
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    headers = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in frame.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, float):
                cells.append("–" if pd.isna(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(r: PipelineResult) -> str:
    s = r.summary
    fq = s["fit_quality_bp"]
    curve = r.fit.curve(-1)
    lines: list[str] = []
    add = lines.append

    add(f"# NSS Yield Curve Report — {s['as_of']}")
    add("")
    add(
        f"Source: **{r.config.source}** · {s['n_curves']} curves from {s['sample_start']} to {s['as_of']} "
        f"· model **{r.config.calibration.model.upper()}** fitted to "
        f"**{'par yields' if r.config.calibration.target == 'par' else 'quoted yields (zero-curve fit)'}**."
    )
    if r.config.source == "synthetic":
        add("")
        add(
            "> **Synthetic data.** These results come from a simulated market with known true "
            "parameters, not from real Treasury yields."
        )
    add("")

    add("## Current curve")
    add("")
    params = pd.Series(curve.as_dict()).rename("value").to_frame()
    params.index.name = "parameter"
    add(_table(params, ".4f"))
    add("")
    add(
        f"Short rate (β0+β1): **{curve.short_rate:.2f}%** · long-run level (β0): **{curve.long_rate:.2f}%** "
        f"· curvature hump at **{s['latest_curve']['curvature_peak_years']:.1f}y**."
    )
    add("")

    obs = r.yields.iloc[-1]
    tenors = list(r.yields.columns)
    tbl = pd.DataFrame(
        {
            "observed_pct": obs.to_numpy(),
            "fitted_pct": r.fit.fitted.iloc[-1].to_numpy(),
            "residual_bp": r.fit.residuals_bp.iloc[-1].to_numpy(),
            "zero_pct": curve.zero(tenors),
            "forward_pct": curve.forward(tenors),
        },
        index=pd.Index([maturity_label(float(t)) for t in tenors], name="tenor"),
    )
    lf = r.latest_fit
    if lf is not None and lf.success:
        lo, hi = lf.confidence_band(tenors)
        tbl["zero_95ci_bp"] = [f"±{v:.1f}" for v in (hi - lo) / 2 * 100]
        flagged = set(np.round(lf.maturities[lf.outliers], 8))
        tbl["quote"] = ["rejected" if round(float(t), 8) in flagged else "" for t in tenors]
    add(_table(tbl))
    add("")
    if lf is not None and lf.outliers.any():
        add(
            "*Rejected* quotes were down-weighted by the robust fit; their residual shows the "
            "full dislocation from the curve implied by the other tenors."
        )
        add("")

    add("## Fit quality")
    add("")
    add(
        f"* Median RMSE **{fq['rmse_median']:.2f} bp** (mean {fq['rmse_mean']:.2f}, 95th pct {fq['rmse_p95']:.2f})"
    )
    add(f"* Median max-abs error {fq['max_abs_error_median']:.2f} bp")
    add(
        f"* Median calibration time {fq['runtime_ms_median']:.1f} ms per curve; "
        f"optimiser success rate {fq['success_rate']:.1%}"
    )
    bounds = fq.get("share_lambda_at_bound") or {}
    at_bound = {k: v for k, v in bounds.items() if v and v > 0.01}
    if at_bound:
        add(
            "* Decay rates at a bound: "
            + ", ".join(f"{k.replace('_', ' ')} {v:.1%}" for k, v in at_bound.items())
        )
    if fq.get("not_converged_reasons"):
        add(
            "* Optimiser status on non-converged dates: "
            + "; ".join(f"{msg} ({n})" for msg, n in fq["not_converged_reasons"].items())
        )
    if fq.get("share_ns_fallback"):
        add(
            f"* {fq['share_ns_fallback']:.1%} of dates had too few tenors for NSS and used Nelson-Siegel"
        )
    if fq.get("rmse_clean_median") is not None and r.config.calibration.robust:
        add(
            f"* Median RMSE over quotes the robust fit kept: **{fq['rmse_clean_median']:.2f} bp**; "
            f"median estimated quote noise σ̂ {fq['sigma_median']:.2f} bp"
        )
    if "outliers" in s:
        ol = s["outliers"]
        add(
            f"* Robust fit rejected **{ol['share_of_quotes']:.2%}** of quotes. Share by tenor: "
            + ", ".join(
                f"{k} {v:.1%}" for k, v in ol["share_by_tenor"].items() if v is not None and v > 0
            )
        )
    if "synthetic_truth" in s:
        add(
            f"* Fitted vs **true** curve RMSE: {s['synthetic_truth']['curve_rmse_vs_truth_bp']:.2f} bp"
        )
    add("")

    if r.reference is not None:
        ov = s["reference_curve"]
        add(f"## Validation against {r.reference_name}")
        add("")
        add(
            f"Zero curves compared on {int(ov['n_dates'])} common dates, 1–30 years: RMSE "
            f"**{ov['rmse_bp']:.1f} bp** ({ov['demeaned_rmse_bp']:.1f} bp after removing each "
            f"maturity's average gap); period-to-period changes correlate "
            f"**{ov['mean_change_corr']:.3f}** on average."
        )
        add("")
        add(_table(r.reference.summary().T, ".2f"))
        add("")
        if "GSW" in (r.reference_name or ""):
            add(
                "The Fed's curve is fitted to off-the-run notes and bonds; this engine fits "
                "on-the-run constant-maturity par yields. On-the-run issues trade rich, so a "
                "small negative bias is expected - the change correlation measures tracking."
            )
            add("")

    add("## Macro regime")
    add("")
    rg = s["regime"]
    add(
        f"* Current regime: **{rg['current']}** since {rg['since']} ({rg['weeks_in_regime']} periods) "
        f"· slope = {rg['slope_definition']} = **{r.spreads['slope'].iloc[-1]:+.2f} pp**"
    )
    if rg.get("curve_dynamics"):
        add(f"* Recent curve move: **{rg['curve_dynamics']}**")
    tr = s.get("spread_tracking", {})
    if tr:
        add(
            f"* Model-implied vs observed spreads: 10y-3m corr {tr.get('10y3m_corr', float('nan')):.3f} "
            f"(MAE {tr.get('10y3m_mae_bp', float('nan')):.1f} bp); 10y-2y corr "
            f"{tr.get('10y2y_corr', float('nan')):.3f} (MAE {tr.get('10y2y_mae_bp', float('nan')):.1f} bp)"
        )
    share = pd.Series(rg["share_of_time"], name="share of sample")
    share.index.name = "regime"
    add("")
    add(_table(share.to_frame(), ".1%"))
    add("")

    if r.recession_model is not None:
        rm = s["recession_model"]
        add("### Recession probability (probit, NY Fed specification)")
        add("")
        add(
            f"P(recession in {rm['horizon_months']} months) = Φ({rm['coef_const']:.3f} "
            f"{rm['coef_spread']:+.3f} × spread) · pseudo-R² {rm['pseudo_r2']:.3f} · AUC {rm['auc']:.3f}"
        )
        add("")
        add(
            f"**Latest: {rm['latest_probability']:.1%}** probability of recession in {rm['target_month']}."
        )
        add("")
    if r.recession_comparison is not None:
        rc = r.recession_comparison.copy()
        add("### Which curve signal predicts recessions? (pseudo-real time)")
        add("")
        add(
            f"Each probit is re-estimated every month using only recessions known at the time and "
            f"scored on the forecasts it would have made ({int(rc['n_forecasts'].iloc[0])} common "
            "forecast months). The near-term forward spread (Engstrom & Sharpe, 2019) is the "
            "3-month forward rate 18 months ahead minus today's 3-month rate, from the NSS "
            "forward curve. Brier: lower is better; log score and AUC: higher is better."
        )
        add("")
        add(
            _table(
                rc[
                    [
                        "auc_in_sample",
                        "pseudo_r2",
                        "auc_out_of_sample",
                        "brier_out_of_sample",
                        "log_score_out_of_sample",
                        "latest_probability",
                    ]
                ],
                ".3f",
            )
        )
        add("")
        add(
            f"Latest near-term forward spread: **{r.spreads['near_term_fwd'].iloc[-1]:+.2f} pp** "
            "(negative = rate cuts priced in)."
        )
        add("")
    if r.lead_times is not None and not r.lead_times.empty:
        lt = r.lead_times.copy()
        for c in ("start", "end", "recession_start", "date_of_min"):
            lt[c] = pd.to_datetime(lt[c]).dt.strftime("%Y-%m")
        lt.index = pd.RangeIndex(1, len(lt) + 1, name="#")
        add("### Sustained inversions (≥ 3 months) and subsequent recessions")
        add("")
        add(
            _table(
                lt[["start", "end", "periods", "min_spread", "recession_start", "lead_months"]],
                ".2f",
            )
        )
        add("")

    if r.acm is not None:
        _term_premium_section(r, add)

    add("## Factor validation")
    add("")
    ev = r.pca.explained_variance_ratio
    add(
        f"PCA of yield changes: first three components explain "
        f"{ev.iloc[0]:.1%} / {ev.iloc[1]:.1%} / {ev.iloc[2]:.1%} (total {ev.sum():.1%})."
    )
    add("")
    add("Correlation of each factor with its model-free proxy (Diebold & Li, 2006):")
    add("")
    add(_table(r.proxy_correlations, ".3f"))
    add("")
    add(
        "With free decay rates, β0 is the curve's asymptote beyond 30 years and −β1 the spread "
        "between infinite and zero maturity, so raw NSS betas track the proxies less closely "
        "than fixed-λ factors. The regime engine therefore uses the *model-implied* 10y−3m "
        "spread, which matches the observed spread almost perfectly."
    )
    add("")

    if r.forecast_eval is not None:
        fe = r.forecast_eval
        add("## Out-of-sample forecasts (Diebold-Li, AR(1) factors)")
        add("")
        add(
            "Relative RMSE vs random walk (< 1 = model more accurate); monthly data, expanding window."
        )
        add("")
        add(_table(fe.relative_rmse, ".3f"))
        add("")
        add("Diebold-Mariano p-values (H0: equal accuracy):")
        add("")
        add(_table(fe.dm_pvalue, ".3f"))
        add("")

    if r.dns is not None:
        d = r.dns
        add("## State-space dynamic Nelson-Siegel (Kalman filter)")
        add("")
        add(
            f"Maximum-likelihood fit on {len(d.factors)} months (Diebold, Rudebusch & Aruoba, "
            f"2006): λ = {d.params.lam:.3f} (curvature peak at {1.7933 / d.params.lam:.1f}y), "
            f"level persistence {d.params.A[0, 0]:.3f}/month, slope {d.params.A[1, 1]:.3f}, "
            f"curvature {d.params.A[2, 2]:.3f}. Measurement noise by tenor (bp): "
            + ", ".join(
                f"{maturity_label(float(m))} {h * 100:.0f}"
                for m, h in zip(d.maturities, d.params.h, strict=True)
            )
            + "."
        )
        add("")
        if r.dns_forecast is not None:
            add("12-month-ahead forecast with 80% interval:")
            add("")
            add(_table(r.dns_forecast, ".2f"))
            add("")

    add("## Relative value (latest)")
    add("")
    rc = r.rich_cheap.join(r.half_life.rename("half_life_weeks"))
    add(_table(rc, ".2f"))
    add("")
    add("Positive residual = yield above the curve = **cheap**.")
    add("")

    add("## Carry & roll-down (3-month horizon, unchanged curve)")
    add("")
    add(_table(r.carry, ".1f"))
    add("")

    add("## Risk of par bonds on the current curve")
    add("")
    risk = pd.DataFrame(
        {
            name: {
                "price": rep.price,
                "DV01": rep.dv01,
                "duration": rep.duration,
                "convexity": rep.convexity,
                **{f"factor dur {k}": v for k, v in rep.factor_durations.items()},
            }
            for name, rep in r.risk.items()
        }
    ).T
    risk.index.name = "bond"
    add(_table(risk, ".3f"))
    add("")
    return "\n".join(lines)


def _term_premium_section(r: PipelineResult, emit: Callable[[str], None]) -> None:
    """Term premium decomposition: latest split, benchmarks, recession test."""
    assert r.acm is not None
    tp = r.summary["term_premium"]
    latest = tp["latest"]
    emit("## Term premium (Adrian-Crump-Moench)")
    emit("")
    emit(
        "A 10-year yield is the average short rate investors expect over ten years plus a "
        "**term premium**, the extra return demanded for holding a long bond. The ACM model "
        "(five principal components of the NSS zero curves, 1-120 months) separates the two "
        "using only regressions."
    )
    emit("")
    emit(
        f"**Latest 10Y zero yield {latest['fitted']:.2f}% = expected short rate "
        f"{latest['expected_short_rate']:.2f}% + term premium {latest['term_premium']:+.2f}%.** "
        f"Sample range of the premium: {tp['min']:+.2f}% ({tp['min_date']}) to "
        f"{tp['max']:+.2f}% ({tp['max_date']}); model fitting error {tp['fit_rmse_bp_mean']:.1f} bp."
    )
    emit("")
    if r.term_premium_comparison is not None:
        cmp = r.term_premium_comparison.copy()
        cmp.index = [f"{e} vs {b}" for e, b in cmp.index]
        cmp.index.name = "10-year term premium"
        emit("Agreement with other estimates (monthly):")
        emit("")
        emit(_table(cmp, ".2f"))
        emit("")
    if r.term_premium_recession is not None:
        rc = r.term_premium_recession
        emit("### Which part of the slope predicts recessions?")
        emit("")
        emit(
            "The spread is split into its expectations component (spread minus the 10-year "
            "term premium) and the term premium itself, both estimated in real time. "
            f"Pseudo-real-time probits, {int(rc['n_forecasts'].iloc[0])} common forecast months."
        )
        emit("")
        cols = ["auc_in_sample", "auc_out_of_sample", "brier_out_of_sample"]
        cols += ["log_score_out_of_sample", "latest_probability"]
        emit(_table(rc[cols], ".3f"))
        emit("")
