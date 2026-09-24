"""Interactive Plotly dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .calibration import CurveHistory
from .regimes import FLAT, INVERTED, REGIME_COLORS, regime_segments

_FACTOR_COLORS = {"Beta0": "#4FC3F7", "-Beta1": "#FFA726", "Beta2": "#AB47BC", "Beta3": "#66BB6A"}


def build_dashboard(history: CurveHistory, regimes: pd.Series,
                    recession_prob: pd.Series | None = None,
                    market_spreads: pd.DataFrame | None = None,
                    rich_cheap: pd.DataFrame | None = None,
                    forecast: pd.DataFrame | None = None,
                    lookback_periods: int = 4) -> go.Figure:
    """Assemble the multi-panel dashboard figure."""
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "surface", "colspan": 2}, None]],
        row_heights=[0.22, 0.22, 0.18, 0.38],
        vertical_spacing=0.06, horizontal_spacing=0.08,
        subplot_titles=(
            "Market quotes vs NSS fit (with forward curve)",
            "Rich / cheap: latest residuals (bp)",
            "Slope: model -β1 vs 10Y-2Y / 10Y-3M (red = inverted, grey = flat)",
            "NSS factors",
            "NY Fed 12m recession probability",
            "Calibration quality (RMSE, bp)",
            "Zero-curve surface"),
    )
    valid = history.valid
    last = valid[-1]
    mats = history.maturities
    grid = np.linspace(max(mats.min(), 1 / 12), mats.max(), 150)
    curve = history.curve(last)

    # 1. market vs fit ------------------------------------------------------
    fig.add_trace(go.Scatter(x=mats, y=history.market.loc[last], mode="markers",
                             name=f"Market {last:%Y-%m-%d}",
                             marker=dict(size=9, color="cyan")), row=1, col=1)
    target = history.config.fit_target
    fig.add_trace(go.Scatter(x=grid, y=curve.market_yield(grid, target), name="NSS fit",
                             line=dict(color="cyan", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=grid, y=curve.forward(grid), name="Inst. forward",
                             line=dict(color="#FFD54F", width=1.5, dash="dash")), row=1, col=1)
    if len(valid) > lookback_periods:
        prev = valid[-1 - lookback_periods]
        fig.add_trace(go.Scatter(x=grid, y=history.curve(prev).market_yield(grid, target),
                                 name=f"Fit {prev:%Y-%m-%d}",
                                 line=dict(color="gray", dash="dot")), row=1, col=1)
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=mats, y=forecast.iloc[-1], mode="lines+markers",
                                 name=f"DL forecast (+{forecast.index[-1]})",
                                 line=dict(color="#EF5350", dash="dashdot")), row=1, col=1)

    # 2. residual bars ------------------------------------------------------
    resid = history.residuals_bp.loc[last]
    colors = np.where(resid > 0, "#2ECC40", "#FF4136")
    text = None
    if rich_cheap is not None and "ZScore" in rich_cheap:
        text = [f"z={z:+.1f}" if np.isfinite(z) else "" for z in rich_cheap["ZScore"]]
    fig.add_trace(go.Bar(x=list(resid.index), y=resid.to_numpy(), marker_color=colors,
                         text=text, textposition="outside", name="Residual (mkt-model)",
                         showlegend=False), row=1, col=2)

    # 3. slope & inversion shading -----------------------------------------
    p = history.params.loc[valid]
    fig.add_trace(go.Scatter(x=p.index, y=-p["Beta1"], name="-β1 (model slope)",
                             line=dict(color="orange", width=2.5)), row=2, col=1)
    if market_spreads is not None:
        for col, color in (("2Y10Y", "white"), ("3M10Y", "#90A4AE")):
            if col in market_spreads:
                fig.add_trace(go.Scatter(x=market_spreads.index, y=market_spreads[col],
                                         name=f"{col[:2]}-{col[2:]} (market)",
                                         line=dict(color=color, dash="dot", width=1)),
                              row=2, col=1)

    # Shapes are added after the traces: plotly skips add_vrect on empty subplots.
    for start, end, label in regime_segments(regimes):
        if label in (INVERTED, FLAT):
            fig.add_vrect(x0=start, x1=end, fillcolor=REGIME_COLORS[label],
                          opacity=0.22 if label == INVERTED else 0.08,
                          layer="below", line_width=0, row=2, col=1)

    # 4. factors -------------------------------------------------------------
    factors = {"Beta0": p["Beta0"], "-Beta1": -p["Beta1"], "Beta2": p["Beta2"]}
    if "Beta3" in p:
        factors["Beta3"] = p["Beta3"]
    for name, s in factors.items():
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name,
                                 line=dict(color=_FACTOR_COLORS[name], width=1.5)), row=2, col=2)

    # 5. recession probability ---------------------------------------------
    if recession_prob is not None and not recession_prob.empty:
        fig.add_trace(go.Scatter(x=recession_prob.index, y=recession_prob * 100,
                                 name="P(recession, 12m) %", fill="tozeroy",
                                 line=dict(color="#EF5350")), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="gray", dash="dot", width=1), row=3, col=1)

    # 6. fit quality ---------------------------------------------------------
    d = history.diagnostics.loc[valid]
    fig.add_trace(go.Scatter(x=d.index, y=d["RMSE_bp"], name="RMSE (bp)",
                             line=dict(color="#E0E0E0", width=1.5)), row=3, col=2)
    fig.add_trace(go.Scatter(x=d.index, y=d["MaxAbsErr_bp"], name="Max |err| (bp)",
                             line=dict(color="#7E57C2", width=1)), row=3, col=2)

    # 7. surface -------------------------------------------------------------
    surf_mats = np.linspace(0.1, mats.max(), 40)
    step = max(1, len(p) // 250)                      # keep the surface light
    sub = p.iloc[::step]
    z = np.vstack([history.curve(ts).zero(surf_mats) for ts in sub.index])
    fig.add_trace(go.Surface(z=z, x=surf_mats, y=sub.index, colorscale="Viridis",
                             showscale=False, name="Zero surface"), row=4, col=1)

    regime_now = regimes.dropna().iloc[-1] if regimes.notna().any() else "n/a"
    color = REGIME_COLORS.get(regime_now, "white")
    headline = (f"CURRENT REGIME: <span style='color:{color}'>{str(regime_now).upper()}</span>"
                f"  |  fit RMSE {d['RMSE_bp'].iloc[-1]:.1f}bp")
    if recession_prob is not None and not recession_prob.empty:
        headline += f"  |  P(recession 12m) {recession_prob.iloc[-1]:.0%}"
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=1.045, showarrow=False,
                       text=headline, font=dict(size=17, color="white"))
    fig.update_layout(height=1500, template="plotly_dark", margin=dict(t=110, l=50, r=30, b=30),
                      legend=dict(orientation="h", y=-0.02),
                      scene=dict(xaxis_title="Maturity (Y)", yaxis_title="Date",
                                 zaxis_title="Zero (%)"))
    fig.update_xaxes(title_text="Maturity (years)", row=1, col=1)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="bp", row=1, col=2)
    fig.update_yaxes(title_text="%", range=[0, 100], row=3, col=1)
    return fig


def save_dashboard(fig: go.Figure, path: str | Path, show: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()
    return path
