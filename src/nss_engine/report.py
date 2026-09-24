"""Plain-text / Markdown rendering of pipeline results."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    add(_table(tbl))
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
    if fq.get("share_ns_fallback"):
        add(
            f"* {fq['share_ns_fallback']:.1%} of dates had too few tenors for NSS and used Nelson-Siegel"
        )
    if "synthetic_truth" in s:
        add(
            f"* Fitted vs **true** curve RMSE: {s['synthetic_truth']['curve_rmse_vs_truth_bp']:.2f} bp"
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
