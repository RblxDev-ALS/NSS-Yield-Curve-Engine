"""Interactive Plotly dashboard.

Every figure is available on its own (``fig_*`` functions, handy in notebooks)
and :func:`build_dashboard` assembles them into a single self-contained HTML
page with headline stat tiles, table views and automatic light/dark theming.

Design notes: one y-axis per chart (never dual axes), a fixed colorblind-safe
categorical order, a single-hue ramp for magnitude and a blue↔red diverging
ramp with a neutral midpoint for signed quantities (residuals, relative RMSE).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import maturity_label
from .models import ns_loadings, nss_loadings
from .regime import REGIMES, regime_spells

if TYPE_CHECKING:
    from .pipeline import PipelineResult

# ---- design tokens (validated categorical order; see README "Design") --------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
RECESSION_FILL = "rgba(137,135,129,0.18)"
DIVERGING = [
    [0.0, "#1c5cab"],
    [0.25, "#6da7ec"],
    [0.5, "#f0efec"],
    [0.75, "#ec8f8e"],
    [1.0, "#b8302f"],
]
REGIME_COLORS = {"Inverted": "#d03b3b", "Flat": "#ec835a", "Normal": "#86b6ef", "Steep": "#2a78d6"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, showline=True, ticks=""),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, showline=False, ticks=""),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(font=dict(family=FONT)),
        margin=dict(l=56, r=24, t=64, b=48),
    )
)


def _fig(**layout: object) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=_TEMPLATE, **layout)
    return fig


def _add_recessions(
    fig: go.Figure, recession: pd.Series | None, start: pd.Timestamp, **kw: object
) -> None:
    if recession is None or recession.sum() == 0:
        return
    rec = recession[recession.index >= start - pd.offsets.MonthEnd(1)]
    spells = regime_spells(rec.map({1: "rec", 0: "exp"}))
    for _, sp in spells[spells["regime"] == "rec"].iterrows():
        fig.add_vrect(
            x0=sp["start"] - pd.offsets.MonthBegin(1),
            x1=sp["end"],
            fillcolor=RECESSION_FILL,
            line_width=0,
            layer="below",
            **kw,
        )


# =============================================================================
# Individual figures
# =============================================================================


def fig_curve_snapshot(r: PipelineResult) -> go.Figure:
    """Latest observed yields, the fitted curve, its forward curve, and 1 year ago."""
    fit = r.fit
    date = r.as_of
    curve = fit.curve(-1)
    grid = np.linspace(1 / 12, 30, 240)
    measure = "par" if r.config.calibration.target == "par" else "zero"
    fig = _fig(
        title=f"Treasury curve on {date:%d %b %Y}",
        height=500,
        hovermode="x unified",
        # many series: put the legend under the plot so it cannot run into the title
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=56, r=24, t=56, b=120),
    )
    fig.update_xaxes(title="Maturity (years)")
    fig.update_yaxes(title="Yield (%)")
    ago = fit.params.index[fit.params.index <= date - pd.DateOffset(years=1)]
    if len(ago):
        c_ago = fit.curve(ago[-1])
        fig.add_scatter(
            x=grid,
            y=c_ago.evaluate(grid, measure),
            name=f"Fitted, {ago[-1]:%b %Y}",
            line=dict(color=MUTED, width=2),
            hovertemplate="%{y:.2f}%",
        )
    fig.add_scatter(
        x=grid,
        y=curve.forward(grid),
        name="Instantaneous forward",
        line=dict(color=SERIES[1], width=2),
        hovertemplate="%{y:.2f}%",
    )
    lf = r.latest_fit
    if lf is not None and lf.success:
        lo, hi = lf.confidence_band(grid, measure)
        if np.all(np.isfinite(lo)):
            fig.add_scatter(
                x=np.concatenate([grid, grid[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor="rgba(42,120,214,0.14)",
                line=dict(width=0),
                name="95% confidence band",
                hoverinfo="skip",
            )
    fig.add_scatter(
        x=grid,
        y=curve.evaluate(grid, measure),
        name=f"NSS fit ({measure})",
        line=dict(color=SERIES[0], width=2.5),
        hovertemplate="%{y:.2f}%",
    )
    if lf is not None and lf.outliers.any():
        fig.add_scatter(
            x=lf.maturities[lf.outliers],
            y=lf.observed[lf.outliers],
            mode="markers",
            name="Rejected by robust fit",
            marker=dict(
                color=SERIES[7], size=14, symbol="x-thin", line=dict(width=3, color=SERIES[7])
            ),
            hovertemplate="%{y:.2f}%",
        )
    obs = r.yields.loc[:date].iloc[-1].dropna()
    fig.add_scatter(
        x=obs.index.astype(float),
        y=obs.values,
        mode="markers",
        name="Observed CMT",
        marker=dict(color=SERIES[0], size=9, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{y:.2f}%",
    )
    if r.forecast_curve is not None:
        h = max(r.config.forecast_horizons)
        fc = r.forecast_curve
        fig.add_scatter(
            x=fc.index.astype(float),
            y=fc.values,
            mode="lines+markers",
            name=f"Diebold-Li forecast, +{h}m",
            line=dict(color=SERIES[2], width=2),
            marker=dict(size=8),
            hovertemplate="%{y:.2f}%",
        )
    return fig


def fig_curve_animation(r: PipelineResult, max_frames: int = 160) -> go.Figure:
    """Month-by-month animation of observed yields and the fitted curve."""
    fit = r.fit
    monthly = fit.params.groupby(fit.params.index.to_period("M")).tail(1)
    step = max(1, int(np.ceil(len(monthly) / max_frames)))
    dates = monthly.index[::step]
    if dates[-1] != monthly.index[-1]:
        dates = dates.append(monthly.index[-1:])
    grid = np.linspace(1 / 12, 30, 120)
    measure = "par" if r.config.calibration.target == "par" else "zero"
    ymin = float(np.nanmin(r.yields.to_numpy())) - 0.5
    ymax = float(np.nanmax(r.yields.to_numpy())) + 0.5

    def traces(d: pd.Timestamp) -> list[go.Scatter]:
        c = fit.curve(d)
        o = r.yields.loc[d].dropna()
        return [
            go.Scatter(
                x=grid,
                y=c.evaluate(grid, measure),
                mode="lines",
                name="NSS fit",
                line=dict(color=SERIES[0], width=2.5),
            ),
            go.Scatter(
                x=o.index.astype(float),
                y=o.values,
                mode="markers",
                name="Observed",
                marker=dict(color=SERIES[0], size=9, line=dict(color=SURFACE, width=2)),
            ),
        ]

    fig = _fig(height=470)
    for t in traces(dates[-1]):
        fig.add_trace(t)
    fig.frames = [go.Frame(data=traces(d), name=f"{d:%Y-%m}") for d in dates]
    fig.update_xaxes(title="Maturity (years)", range=[0, 30.5])
    fig.update_yaxes(title="Yield (%)", range=[ymin, ymax])
    fig.update_layout(
        title_text="The curve month by month (drag the slider or press play)",
        sliders=[
            dict(
                active=len(dates) - 1,
                currentvalue=dict(prefix="", font=dict(size=14)),
                pad=dict(t=40),
                steps=[
                    dict(
                        method="animate",
                        label=f"{d:%Y-%m}",
                        args=[
                            [f"{d:%Y-%m}"],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for d in dates
                ],
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=1,
                y=1.14,
                xanchor="right",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=80, redraw=False),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                    ),
                ],
            )
        ],
        margin=dict(b=120),
    )
    return fig


def fig_surface(r: PipelineResult, max_dates: int = 240) -> go.Figure:
    fit = r.fit
    step = max(1, len(fit.params) // max_dates)
    sub = fit.params.iloc[::step]
    grid = np.linspace(0.25, 30, 50)
    z = np.array(
        [
            nss_loadings(grid, p.lambda1, p.lambda2)
            @ np.array([p.beta0, p.beta1, p.beta2, p.beta3])
            for p in sub.itertuples()
        ]
    )
    seq = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
    fig = _fig(title="Term structure surface (fitted zero rates)", height=560)
    fig.add_surface(
        x=grid,
        y=sub.index,
        z=z,
        colorscale=seq,
        showscale=True,
        colorbar=dict(title="%", thickness=12, len=0.6),
        hovertemplate="%{y|%b %Y}<br>%{x:.1f}y: %{z:.2f}%<extra></extra>",
    )
    axis = dict(gridcolor=GRID, backgroundcolor=SURFACE, showbackground=True)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Maturity (y)", **axis),
            yaxis=dict(title="", **axis),
            zaxis=dict(title="Yield (%)", **axis),
            aspectratio=dict(x=1.0, y=1.7, z=0.7),
            camera=dict(eye=dict(x=1.35, y=-1.45, z=0.75)),
        ),
        margin=dict(l=0, r=0, t=56, b=0),
    )
    return fig


def fig_factors(r: PipelineResult) -> go.Figure:
    """Small multiples of the four NSS factors with recession shading."""
    p = r.fit.params
    rows = [
        ("β0 · level (long-run rate)", p["beta0"]),
        ("−β1 · slope (long minus short)", -p["beta1"]),
        ("β2 · curvature", p["beta2"]),
        ("β3 · second curvature", p["beta3"]),
    ]
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[t for t, _ in rows],
    )
    for i, (name, s) in enumerate(rows, start=1):
        fig.add_scatter(
            x=s.index,
            y=s.values,
            name=name,
            line=dict(color=SERIES[0], width=1.6),
            showlegend=False,
            hovertemplate="%{y:.2f}",
            row=i,
            col=1,
        )
        _add_recessions(fig, r.recession, p.index[0], row=i, col=1)
    fig.update_layout(
        template=_TEMPLATE,
        height=760,
        hovermode="x unified",
        title="Latent factors through time (grey bands: NBER recessions)",
    )
    fig.update_annotations(font=dict(size=13, color=INK_2), x=0, xanchor="left")
    return fig


def fig_slope_regime(r: PipelineResult) -> go.Figure:
    """Model-implied vs observed slope, with the regime strip beneath."""
    sp = r.spreads
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.86, 0.14], vertical_spacing=0.04
    )
    lbl = f"{r.config.slope_long:g}y−{maturity_label(r.config.slope_short).lower()}"
    if "obs_10y3m" in sp and r.config.slope_short == 0.25 and r.config.slope_long == 10:
        fig.add_scatter(
            x=sp.index,
            y=sp["obs_10y3m"],
            name="Observed 10y−3m",
            line=dict(color=MUTED, width=1.4),
            hovertemplate="%{y:.2f} pp",
            row=1,
            col=1,
        )
    fig.add_scatter(
        x=sp.index,
        y=sp["slope"],
        name=f"NSS-implied {lbl}",
        line=dict(color=SERIES[0], width=2),
        hovertemplate="%{y:.2f} pp",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=sp.index,
        y=sp["model_10y2y"],
        name="NSS-implied 10y−2y",
        line=dict(color=SERIES[1], width=1.6),
        hovertemplate="%{y:.2f} pp",
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color=INK_2, width=1), row=1, col=1)
    _add_recessions(fig, r.recession, sp.index[0], row=1, col=1)

    codes = r.regimes.cat.codes.to_numpy()
    n = len(REGIMES)
    scale = []
    for i, name in enumerate(REGIMES):
        scale += [[i / n, REGIME_COLORS[name]], [(i + 1) / n, REGIME_COLORS[name]]]
    fig.add_heatmap(
        x=r.regimes.index,
        y=["Regime"],
        z=[codes],
        zmin=-0.5,
        zmax=n - 0.5,
        colorscale=scale,
        showscale=False,
        customdata=[r.regimes.astype(str).to_numpy()],
        hovertemplate="%{x|%d %b %Y}: %{customdata}<extra></extra>",
        row=2,
        col=1,
    )
    for name in REGIMES:  # legend entries for the strip
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=name,
            marker=dict(symbol="square", size=12, color=REGIME_COLORS[name]),
            row=2,
            col=1,
        )
    fig.update_yaxes(title="Spread (pp)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_layout(
        template=_TEMPLATE, height=520, hovermode="x unified", title="Curve slope and macro regime"
    )
    return fig


def fig_recession_probability(r: PipelineResult) -> go.Figure | None:
    if r.recession_model is None:
        return None
    m = r.recession_model
    prob = m.fitted * 100
    fig = _fig(
        title=f"Probability of recession {m.horizon} months ahead (probit on the slope)",
        height=380,
        hovermode="x unified",
    )
    fig.add_scatter(
        x=prob.index,
        y=prob.values,
        name="Probability",
        line=dict(color=SERIES[0], width=2),
        fill="tozeroy",
        fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="%{y:.0f}%",
        showlegend=False,
    )
    _add_recessions(fig, r.recession, prob.index[0])
    fig.update_yaxes(title="%", range=[0, 100])
    fig.add_annotation(
        x=prob.index[-1],
        y=prob.iloc[-1],
        text=f"{prob.iloc[-1]:.0f}%",
        showarrow=True,
        arrowhead=0,
        ax=0,
        ay=-30,
        font=dict(color=INK, size=13),
    )
    return fig


def fig_forward_spread(r: PipelineResult) -> go.Figure:
    """Near-term forward spread (Engstrom & Sharpe) against the 10y−3m slope."""
    sp = r.spreads
    fig = _fig(
        title="Near-term forward spread vs the long-term slope",
        height=400,
        hovermode="x unified",
    )
    fig.add_scatter(
        x=sp.index,
        y=sp["slope"],
        name="10y − 3m",
        line=dict(color=SERIES[0], width=1.8),
        hovertemplate="%{y:+.2f} pp",
    )
    fig.add_scatter(
        x=sp.index,
        y=sp["near_term_fwd"],
        name="Forward 3m in 18m − 3m",
        line=dict(color=SERIES[1], width=1.8),
        hovertemplate="%{y:+.2f} pp",
    )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    _add_recessions(fig, r.recession, sp.index[0])
    fig.update_yaxes(title="percentage points")
    return fig


def fig_reference_gap(r: PipelineResult) -> go.Figure | None:
    """Engine zero curve minus the reference (Fed GSW or true) curve through time."""
    if r.reference is None:
        return None
    d = r.reference.difference_bp
    cols = [c for c in ("2Y", "10Y", "30Y") if c in d.columns]
    fig = _fig(
        title=f"Engine zero curve minus {r.reference_name}", height=380, hovermode="x unified"
    )
    for i, c in enumerate(cols):
        fig.add_scatter(
            x=d.index,
            y=d[c].rolling(4, min_periods=1).mean(),
            name=c,
            line=dict(color=SERIES[i], width=1.6),
            hovertemplate="%{y:+.1f} bp",
        )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig.update_yaxes(title="bp (4-period average)")
    return fig


def fig_dns_forecast(r: PipelineResult) -> go.Figure | None:
    """State-space DNS forecast across maturities with its 80% interval."""
    if r.dns_forecast is None:
        return None
    fc = r.dns_forecast
    x = r.dns.maturities if r.dns is not None else np.arange(len(fc))
    h = max(r.config.forecast_horizons)
    fig = _fig(
        title=f"State-space forecast of the curve, {h} months ahead",
        height=400,
        hovermode="x unified",
    )
    fig.update_xaxes(title="Maturity (years)")
    fig.update_yaxes(title="Yield (%)")
    fig.add_scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([fc["upper_80_pct"], fc["lower_80_pct"][::-1]]),
        fill="toself",
        fillcolor="rgba(27,175,122,0.16)",
        line=dict(width=0),
        name="80% interval",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=x,
        y=fc["forecast_pct"],
        name="Forecast",
        mode="lines+markers",
        line=dict(color=SERIES[2], width=2.5),
        hovertemplate="%{y:.2f}%",
    )
    fig.add_scatter(
        x=x,
        y=fc["latest_pct"],
        name="Latest month (average)",
        mode="lines+markers",
        line=dict(color=MUTED, width=2, dash="dot"),
        hovertemplate="%{y:.2f}%",
    )
    return fig


def fig_term_premium(r: PipelineResult) -> go.Figure | None:
    """10-year yield split into expected short rates and term premium, with benchmarks."""
    if r.acm is None:
        return None
    dec = r.acm.decomposition(10)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            "10-year zero yield = expected short rate + term premium",
            "10-year term premium: this engine vs other estimates",
        ),
    )
    fig.add_scatter(
        x=dec.index,
        y=dec["fitted"],
        name="10Y zero yield",
        line=dict(color=INK_2, width=1.8),
        hovertemplate="%{y:.2f}%",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=dec.index,
        y=dec["expected_short_rate"],
        name="Expected short rate (10y avg)",
        line=dict(color=SERIES[0], width=1.8),
        hovertemplate="%{y:.2f}%",
        row=1,
        col=1,
    )
    fig.add_scatter(
        x=dec.index,
        y=dec["term_premium"],
        name="Term premium (ACM on NSS curves)",
        line=dict(color=SERIES[1], width=2.2),
        hovertemplate="%{y:+.2f}%",
        row=2,
        col=1,
    )
    if r.term_premium_real_time is not None:
        rt = r.term_premium_real_time["term_premium"]
        fig.add_scatter(
            x=rt.index,
            y=rt,
            name="…estimated in real time",
            line=dict(color=SERIES[1], width=1.4, dash="dot"),
            hovertemplate="%{y:+.2f}%",
            row=2,
            col=1,
        )
    for i, (name, series) in enumerate(r.term_premium_benchmarks.items()):
        monthly = series.resample("ME").mean().loc[dec.index[0] :]
        fig.add_scatter(
            x=monthly.index,
            y=monthly,
            name=name,
            line=dict(color=SERIES[2 + i], width=1.6),
            hovertemplate="%{y:+.2f}%",
            row=2,
            col=1,
        )
    fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=2, col=1)
    for row in (1, 2):
        _add_recessions(fig, r.recession, dec.index[0], row=row)
    fig.update_yaxes(title="%", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    fig.update_layout(
        template=_TEMPLATE,
        height=620,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        margin=dict(t=60, b=110),
    )
    fig.update_annotations(font=dict(size=13, color=INK_2))
    return fig


def fig_fit_quality(r: PipelineResult) -> go.Figure:
    d = r.fit.diagnostics
    roll = d["rmse_bp"].rolling(13, min_periods=1).median()
    fig = _fig(title="Calibration error through time", height=360, hovermode="x unified")
    fig.add_scatter(
        x=d.index,
        y=d["rmse_bp"],
        name="RMSE per curve",
        mode="lines",
        line=dict(color=AXIS, width=1),
        hovertemplate="%{y:.2f} bp",
    )
    fig.add_scatter(
        x=d.index,
        y=roll,
        name="13-period median",
        line=dict(color=SERIES[0], width=2),
        hovertemplate="%{y:.2f} bp",
    )
    fig.update_yaxes(title="RMSE (bp)", rangemode="tozero")
    return fig


def fig_residual_heatmap(r: PipelineResult) -> go.Figure:
    res = r.fit.residuals_bp
    lim = float(np.nanpercentile(np.abs(res.to_numpy()), 99)) or 1.0
    fig = _fig(
        title="Pricing residuals by tenor (observed − fitted, bp; red = cheap, blue = rich)",
        height=380,
    )
    fig.add_heatmap(
        x=res.index,
        y=[maturity_label(float(c)) for c in res.columns],
        z=res.T.to_numpy(),
        zmin=-lim,
        zmax=lim,
        colorscale=DIVERGING,
        colorbar=dict(title="bp", thickness=12),
        hovertemplate="%{x|%d %b %Y} · %{y}: %{z:.1f} bp<extra></extra>",
    )
    return fig


def fig_rich_cheap(r: PipelineResult) -> go.Figure:
    rc = r.rich_cheap
    colors = [
        REGIME_COLORS["Inverted"] if s == "CHEAP" else SERIES[0] if s == "RICH" else AXIS
        for s in rc["signal"]
    ]
    fig = _fig(title="Relative value today: residual z-score (±2 = signal)", height=360)
    fig.add_bar(
        x=rc.index,
        y=rc["zscore"],
        marker=dict(color=colors, cornerradius=4),
        customdata=np.column_stack([rc["residual_bp"], rc["signal"]]),
        hovertemplate="%{x}: z = %{y:.2f}<br>residual %{customdata[0]:.1f} bp · %{customdata[1]}"
        "<extra></extra>",
    )
    for y in (2, -2):
        fig.add_hline(y=y, line=dict(color=MUTED, width=1))
    lim = max(2.5, float(np.nanmax(np.abs(rc["zscore"].to_numpy(dtype=float)), initial=0)) + 0.5)
    fig.update_yaxes(title="z-score", zeroline=True, range=[-lim, lim])
    return fig


def fig_forecast_skill(r: PipelineResult) -> go.Figure | None:
    if r.forecast_eval is None:
        return None
    rel = r.forecast_eval.relative_rmse
    z = rel.to_numpy(dtype=float)
    dev = float(np.nanmax(np.abs(z - 1))) or 0.1
    fig = _fig(title="Out-of-sample RMSE relative to random walk (<1 = model better)", height=320)
    fig.add_heatmap(
        x=list(rel.columns),
        y=[f"{h}-month" for h in rel.index],
        z=z,
        zmin=1 - dev,
        zmax=1 + dev,
        colorscale=[[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#b8302f"]],
        text=np.vectorize(lambda v: f"{v:.2f}")(z),
        texttemplate="%{text}",
        colorbar=dict(title="ratio", thickness=12),
        hovertemplate="%{y} horizon, %{x}: %{z:.3f}<extra></extra>",
    )
    fig.update_yaxes(automargin=True, title="Horizon")
    return fig


def fig_pca(r: PipelineResult) -> go.Figure:
    """Empirical PCA loadings next to the Nelson-Siegel loadings they mirror."""
    ld = r.pca.loadings
    mats = np.asarray(ld.index, dtype=float)
    lam = float(r.fit.params["lambda1"].median())
    ns = ns_loadings(mats, lam)
    names = ["Level", "Slope", "Curvature"]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{n}: PC{i + 1} explains {r.pca.explained_variance_ratio.iloc[i]:.1%}"
            for i, n in enumerate(names)
        ],
    )
    for i in range(3):
        pc = ld.iloc[:, i].to_numpy()
        nl = (
            ns[:, i] if i != 1 else -ns[:, 1]
        )  # slope loading falls with maturity; flip to match PC2
        nl = nl - nl.mean() if i > 0 else nl
        nl = nl / np.linalg.norm(nl) * np.sign(np.dot(nl, pc) or 1)
        fig.add_scatter(
            x=mats,
            y=pc,
            name="PCA loading",
            mode="lines+markers",
            line=dict(color=SERIES[0], width=2),
            marker=dict(size=8),
            showlegend=i == 0,
            row=1,
            col=i + 1,
        )
        fig.add_scatter(
            x=mats,
            y=nl,
            name="NS loading (normalised)",
            line=dict(color=SERIES[1], width=2),
            showlegend=i == 0,
            row=1,
            col=i + 1,
        )
    fig.update_xaxes(title="Maturity (y)")
    fig.update_layout(
        template=_TEMPLATE,
        height=420,
        title="Factor validation: PCA vs Nelson-Siegel loadings",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(t=90, b=100),
    )
    fig.update_annotations(font=dict(size=13, color=INK_2))
    return fig


# =============================================================================
# Page assembly
# =============================================================================

_CSS = """
:root { color-scheme: light;
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --border:rgba(11,11,11,.10); --accent:#2a78d6; --critical:#d03b3b; --good:#006300; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; --critical:#e66767; --good:#0ca30c; } }
:root[data-theme="dark"] { color-scheme: dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --border:rgba(255,255,255,.10); --accent:#3987e5; --critical:#e66767; --good:#0ca30c; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.5; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }
header h1 { font-size: 28px; margin: 0 0 4px; letter-spacing: -0.01em; }
header p { margin: 0; color: var(--ink-2); }
.banner { margin-top: 12px; padding: 10px 14px; border-radius: 8px; border:1px solid var(--border);
  background: var(--surface); color: var(--ink-2); }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 24px 0; }
.tile { background: var(--surface); border:1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink-2); }
.tile .value { font-size: 28px; font-weight: 600; margin-top: 2px; }
.tile .sub { font-size: 12px; color: var(--muted); }
section { margin-top: 36px; }
section h2 { font-size: 20px; margin: 0 0 4px; }
section > p { margin: 0 0 12px; color: var(--ink-2); max-width: 820px; }
.card { background: var(--surface); border:1px solid var(--border); border-radius: 12px; padding: 8px;
  margin-bottom: 12px; overflow: hidden; }
details { margin: 4px 0 12px; }
summary { cursor: pointer; color: var(--accent); font-size: 14px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; margin-top: 8px;
  display: block; overflow-x: auto; max-width: 100%; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 600; }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } }
@media (max-width: 480px) { .tiles { grid-template-columns: 1fr 1fr; }
  .tile .value { font-size: 22px; white-space: nowrap; } header h1 { font-size: 24px; } }
footer { margin-top: 48px; color: var(--muted); font-size: 13px; }
a { color: var(--accent); }
"""

# Re-colour Plotly figures when the page is in dark mode (Plotly bakes colours in).
_DARK_JS = """
(function(){
  const dark = () => document.documentElement.dataset.theme === 'dark' ||
    (document.documentElement.dataset.theme !== 'light' &&
     window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  function paint(){
    const d = dark();
    const bg = d ? '#1a1a19' : '#fcfcfb', ink = d ? '#c3c2b7' : '#52514e',
          title = d ? '#ffffff' : '#0b0b0b', grid = d ? '#2c2c2a' : '#e1e0d9', axis = d ? '#383835' : '#c3c2b7';
    document.querySelectorAll('.js-plotly-plot').forEach(el => {
      const upd = {paper_bgcolor: bg, plot_bgcolor: bg, 'font.color': ink, 'title.font.color': title};
      Object.keys(el.layout || {}).forEach(k => {
        if (/^[xy]axis\\d*$/.test(k)) { upd[k + '.gridcolor'] = grid; upd[k + '.linecolor'] = axis;
                                         upd[k + '.zerolinecolor'] = axis; }
        if (/^scene\\d*$/.test(k)) { ['xaxis','yaxis','zaxis'].forEach(a => {
          upd[k + '.' + a + '.backgroundcolor'] = bg; upd[k + '.' + a + '.gridcolor'] = grid; }); }
      });
      (el.layout.annotations || []).forEach((a, i) => {
        upd['annotations[' + i + '].font.color'] = a.showarrow ? title : ink; });
      (el.layout.sliders || []).forEach((_, i) => {
        upd['sliders[' + i + '].bgcolor'] = grid; upd['sliders[' + i + '].bordercolor'] = axis;
        upd['sliders[' + i + '].tickcolor'] = axis; upd['sliders[' + i + '].font.color'] = ink;
        upd['sliders[' + i + '].currentvalue.font.color'] = title; });
      (el.layout.updatemenus || []).forEach((_, i) => {
        upd['updatemenus[' + i + '].bgcolor'] = bg; upd['updatemenus[' + i + '].bordercolor'] = axis;
        upd['updatemenus[' + i + '].font.color'] = ink; });
      Plotly.relayout(el, upd);
    });
  }
  window.addEventListener('load', paint);
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paint);
})();
"""


def _table_html(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    return df.to_html(float_format=lambda v: floatfmt.format(v), border=0, na_rep="–", escape=True)


def _tile(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="sub">{html.escape(sub)}</div></div>'
    )


def build_dashboard(
    r: PipelineResult, path: str | Path, include_plotlyjs: str | bool = "cdn"
) -> Path:
    """Write the full dashboard to ``path`` and return it."""
    s = r.summary
    first = [True]

    def embed(fig: go.Figure | None) -> str:
        if fig is None:
            return ""
        js: str | bool = include_plotlyjs if first[0] else False
        first[0] = False
        return (
            '<div class="card">'
            + fig.to_html(
                full_html=False,
                include_plotlyjs=js,
                auto_play=False,
                config={"displaylogo": False, "responsive": True},
            )
            + "</div>"
        )

    curve = r.fit.curve(-1)
    slope = r.spreads["slope"].iloc[-1]
    tiles = [
        _tile("Regime", s["regime"]["current"], f"since {s['regime']['since']}"),
        _tile(f"Slope ({s['regime']['slope_definition']})", f"{slope:+.2f} pp", "NSS-implied"),
        _tile(
            "10-year yield",
            f"{float(curve.evaluate([10.0], 'par' if r.config.calibration.target == 'par' else 'zero')[0]):.2f}%",
            f"short rate {curve.short_rate:.2f}%",
        ),
        _tile(
            "Median fit error",
            f"{s['fit_quality_bp']['rmse_median']:.1f} bp",
            f"{s['n_curves']} curves calibrated",
        ),
    ]
    if r.recession_model is not None:
        tiles.insert(
            2,
            _tile(
                f"Recession odds, {r.recession_model.horizon}m",
                f"{r.recession_model.latest_probability:.0%}",
                "probit on the slope",
            ),
        )

    if r.acm is not None:
        tp_latest = s["term_premium"]["latest"]
        tiles.append(
            _tile(
                "10Y term premium",
                f"{tp_latest['term_premium']:+.2f}%",
                f"expected short rate {tp_latest['expected_short_rate']:.2f}%",
            )
        )

    obs = r.yields.iloc[-1]
    today = pd.DataFrame(
        {
            "Observed (%)": obs.to_numpy(),
            "Fitted (%)": r.fit.fitted.iloc[-1].to_numpy(),
            "Residual (bp)": r.fit.residuals_bp.iloc[-1].to_numpy(),
            "Forward (%)": curve.forward(list(obs.index)),
        },
        index=pd.Index([maturity_label(float(t)) for t in obs.index], name="Tenor"),
    )

    synthetic = r.config.source == "synthetic"
    banner = (
        (
            '<div class="banner"><strong>Synthetic data.</strong> This page was generated from a simulated '
            "market with known true parameters, not from real Treasury yields.</div>"
        )
        if synthetic
        else ""
    )

    lead_html = ""
    if r.lead_times is not None and not r.lead_times.empty:
        lt = r.lead_times[
            ["start", "end", "periods", "min_spread", "recession_start", "lead_months"]
        ].copy()
        for c in ("start", "end", "recession_start"):
            lt[c] = pd.to_datetime(lt[c]).dt.strftime("%b %Y")
        lt.columns = [
            "Inversion start",
            "End",
            "Months",
            "Deepest (pp)",
            "Next recession",
            "Lead (months)",
        ]
        lead_html = (
            "<details><summary>Inversion episodes table</summary>"
            + _table_html(lt.set_index("Inversion start"), "{:.1f}")
            + "</details>"
        )

    fc_html = ""
    if r.forecast_eval is not None:
        fc_html = (
            "<details><summary>Diebold-Mariano p-values</summary>"
            + _table_html(r.forecast_eval.dm_pvalue, "{:.3f}")
            + "</details>"
        )

    # The first embedded figure carries plotly.js, so render the page's first
    # figure before any section that is assembled ahead of the body.
    snapshot_html = embed(fig_curve_snapshot(r))

    rec_cmp_html = ""
    if r.recession_comparison is not None:
        cmp = r.recession_comparison[
            ["auc_in_sample", "auc_out_of_sample", "brier_out_of_sample", "latest_probability"]
        ]
        rec_cmp_html = (
            "<details><summary>Which signal predicts recessions best (pseudo-real time)?</summary>"
            + _table_html(cmp, "{:.3f}")
            + "</details>"
        )

    tp_section = ""
    if r.acm is not None:
        tp = s["term_premium"]["latest"]
        tp_tables = ""
        if r.term_premium_comparison is not None:
            cmp = r.term_premium_comparison.copy()
            cmp.index = [f"{e} vs {b}" for e, b in cmp.index]
            cmp.columns = [
                "corr (level)",
                "corr (12m change)",
                "RMSE (bp)",
                "mean gap (bp)",
                "months",
            ]
            tp_tables += (
                "<details><summary>Agreement with other estimates</summary>"
                + _table_html(cmp, "{:.2f}")
                + "</details>"
            )
        if r.term_premium_recession is not None:
            rc = r.term_premium_recession[
                ["auc_in_sample", "auc_out_of_sample", "brier_out_of_sample", "latest_probability"]
            ]
            tp_tables += (
                "<details><summary>Which part of the slope predicts recessions?</summary>"
                + _table_html(rc, "{:.3f}")
                + "</details>"
            )
        tp_section = (
            "<section><h2>Expected rates vs term premium</h2>"
            "<p>A 10-year yield is the average short rate investors expect over the next decade plus a "
            "<em>term premium</em>, the extra return demanded for holding a long bond. The "
            "Adrian-Crump-Moench model (the method behind the New York Fed's published premium) "
            f"separates the two. Latest: {tp['fitted']:.2f}% = {tp['expected_short_rate']:.2f}% "
            f"expected {'+' if tp['term_premium'] >= 0 else '−'} {abs(tp['term_premium']):.2f}% term premium.</p>"
            f"{embed(fig_term_premium(r))}{tp_tables}</section>"
        )

    ref_section = ""
    if r.reference is not None:
        ov = s["reference_curve"]
        ref_section = (
            f"<section><h2>Validation against {html.escape(r.reference_name or '')}</h2>"
            f"<p>Zero curves agree to {ov['rmse_bp']:.1f} bp RMSE over {int(ov['n_dates'])} dates; "
            f"period-to-period changes correlate {ov['mean_change_corr']:.3f}. The Fed fits off-the-run "
            "notes and bonds, this engine on-the-run par yields, so small persistent gaps are expected."
            f"</p>{embed(fig_reference_gap(r))}"
            "<details><summary>Table view</summary>"
            + _table_html(r.reference.summary().T, "{:.2f}")
            + "</details></section>"
        )

    body = f"""
<header>
  <h1>NSS Yield Curve Engine</h1>
  <p>U.S. Treasury curve as of <strong>{r.as_of:%d %B %Y}</strong> ·
     {"Nelson-Siegel-Svensson" if r.config.calibration.model == "nss" else "Nelson-Siegel"} ·
     source: {html.escape(r.config.source)}</p>
  {banner}
</header>
<div class="tiles">{"".join(tiles)}</div>

<section><h2>Today's curve</h2>
<p>Dots are observed constant-maturity yields; the line is the calibrated NSS curve. The forward curve shows
the rates the market implies for future short-term borrowing.</p>
{snapshot_html}
<details><summary>Table view</summary>{_table_html(today)}</details>
</section>

<section><h2>How the curve has moved</h2>
<p>Every week since {r.fit.params.index[0]:%Y} compressed into a slider, and as a surface.</p>
{embed(fig_curve_animation(r))}
{embed(fig_surface(r))}
</section>

<section><h2>Slope, regimes and recessions</h2>
<p>An inverted curve (long rates below short rates) has preceded every U.S. recession since the 1960s.
The regime strip classifies the model-implied slope with hysteresis so noise near a threshold does not
flip the label.</p>
{embed(fig_slope_regime(r))}
{lead_html}
{embed(fig_recession_probability(r))}
<p>The near-term forward spread (Engstrom &amp; Sharpe, 2019) reads the market's expected path of the
Fed funds rate off the NSS forward curve: below zero, cuts are priced in.</p>
{embed(fig_forward_spread(r))}
{rec_cmp_html}
</section>

<section><h2>Latent factors</h2>
<p>Level, slope and curvature summarise the whole curve in a handful of numbers.</p>
{embed(fig_factors(r))}
{embed(fig_pca(r))}
<details><summary>Correlation with model-free proxies</summary>{_table_html(r.proxy_correlations, "{:.3f}")}</details>
</section>

{tp_section}

{ref_section}

<section><h2>Model quality and relative value</h2>
<p>Residuals are where individual tenors sit relative to the smooth curve. Positive (red) means the yield
is above the curve - the bond is cheap relative to its neighbours.</p>
{embed(fig_fit_quality(r))}
{embed(fig_residual_heatmap(r))}
{embed(fig_rich_cheap(r))}
<details><summary>Table view</summary>{_table_html(r.rich_cheap.join(r.half_life.rename("half-life (periods)")))}</details>
</section>

<section><h2>Forecasting</h2>
<p>Diebold-Li dynamic Nelson-Siegel forecasts, re-estimated each month on past data only, compared with
the random-walk ("no change") benchmark.</p>
{embed(fig_forecast_skill(r))}
{fc_html}
<p>The state-space version of the model (Kalman filter, all parameters estimated jointly by maximum
likelihood) also gives forecast <em>intervals</em>.</p>
{embed(fig_dns_forecast(r))}
</section>

<footer>Generated by <a href="https://github.com/RblxDev-ALS/NSS-Yield-Curve-Engine">nss-engine</a>.
Data: Federal Reserve Bank of St. Louis (FRED), H.15 constant-maturity Treasury yields; NBER recession dates.
Not investment advice.</footer>
"""
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSS Yield Curve Dashboard</title>
<style>{_CSS}</style></head>
<body><main>{body}</main><script>{_DARK_JS}</script></body></html>"""
    out = Path(path)
    out.write_text(page, encoding="utf-8")
    return out
