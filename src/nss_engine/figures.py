"""Static charts (PNG) for the README and the project website.

:func:`make_figures` draws three charts from a :class:`~nss_engine.pipeline.PipelineResult`:

* ``curve.png`` - 36 years of the fitted zero curve as a heatmap, next to the
  latest par curve and its quotes;
* ``term_premium.png`` - the 10-year term premium estimated in real time,
  plain ACM against survey-anchored ACM, with Kim-Wright for reference;
* ``recession.png`` - the probability of a recession within 12 months, against
  NBER recessions.

Each comes in a light and a ``-dark`` variant. The live workflow runs this on
FRED data (``nss-engine run --figures``), so the charts are never drawn by
hand. Needs matplotlib: ``pip install "nss-engine[figures]"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .models import nss_loadings

if TYPE_CHECKING:
    from .pipeline import PipelineResult

#: Sequential blue ramp (light to dark) for the yield heatmap.
_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@dataclass(frozen=True)
class _Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    recession: str
    ramp: tuple[str, ...]


LIGHT = _Theme(
    "light",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    ("#2a78d6", "#eb6834", "#1baf7a"),
    "#ecebe6",
    tuple(_RAMP),
)
DARK = _Theme(
    "dark",
    "#1a1a19",
    "#ffffff",
    "#c3c2b7",
    "#898781",
    "#2c2c2a",
    "#383835",
    ("#3987e5", "#d95926", "#199e70"),
    "#2a2a28",
    tuple(reversed(_RAMP)),
)

SOURCE_NOTE = "Data: FRED (H.15, TIPS, Kim-Wright, NBER), Philadelphia Fed SPF · nss-engine"


def make_figures(result: PipelineResult, out_dir: str | Path, dark: bool = True) -> dict[str, Path]:
    """Write the charts to ``out_dir`` and return their paths.

    Charts whose inputs are missing (no recession data, no term premium) are
    skipped. With ``dark=True`` each chart also gets a ``-dark`` variant.
    """
    import matplotlib

    matplotlib.use("Agg")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    themes = (LIGHT, DARK) if dark else (LIGHT,)
    for theme in themes:
        suffix = "" if theme is LIGHT else "-dark"
        for name, draw in (
            ("curve", _curve_chart),
            ("term_premium", _term_premium_chart),
            ("recession", _recession_chart),
        ):
            fig = draw(result, theme)
            if fig is None:
                continue
            path = out / f"{name}{suffix}.png"
            fig.savefig(path, dpi=160, facecolor=theme.surface)
            _close(fig)
            paths[f"{name}{suffix}"] = path
    return paths


# =============================================================================
# Helpers
# =============================================================================


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _style(ax: Any, t: _Theme) -> None:
    ax.set_facecolor(t.surface)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t.axis)
    ax.tick_params(colors=t.ink2, labelsize=9, length=0)
    ax.grid(axis="y", color=t.grid, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(t: _Theme, title: str, subtitle: str, width: float = 10.0, height: float = 4.8) -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig = plt.figure(figsize=(width, height), facecolor=t.surface)
    fig.text(0.012, 0.955, title, fontsize=15, fontweight="bold", color=t.ink, va="top")
    fig.text(0.012, 0.885, subtitle, fontsize=10, color=t.ink2, va="top")
    fig.text(0.012, 0.018, SOURCE_NOTE, fontsize=7.5, color=t.muted, va="bottom")
    return fig


def _shade_recessions(ax: Any, recession: pd.Series | None, start: pd.Timestamp, t: _Theme) -> bool:
    if recession is None or recession.sum() == 0:
        return False
    rec = recession[recession.index >= start]
    on = rec.astype(bool).to_numpy()
    idx = rec.index
    shaded = False
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            ax.axvspan(idx[i] - pd.offsets.MonthBegin(1), idx[j], color=t.recession, lw=0, zorder=0)
            shaded = True
            i = j + 1
        else:
            i += 1
    return shaded


def _end_label(ax: Any, s: pd.Series, text: str, t: _Theme, color: str, dy: float = 0.0) -> None:
    s = s.dropna()
    if s.empty:
        return
    x, y = s.index[-1], float(s.iloc[-1])
    ax.plot([x], [y], "o", ms=5, color=color, mec=t.surface, mew=1.5, zorder=5)
    ax.annotate(
        text,
        (x, y),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=t.ink,
        annotation_clip=False,
    )


def _monthly(s: pd.Series) -> pd.Series:
    out = s.dropna().copy()
    out.index = pd.DatetimeIndex(out.index).to_period("M").to_timestamp("M")
    return out.groupby(level=0).last()


# =============================================================================
# Charts
# =============================================================================


def _curve_chart(r: PipelineResult, t: _Theme) -> Any:
    from matplotlib import dates as mdates
    from matplotlib.colors import LinearSegmentedColormap

    params = r.fit.params.resample("ME").last().dropna()
    grid = np.linspace(0.25, 30.0, 120)
    z = np.array(
        [
            nss_loadings(grid, p.lambda1, p.lambda2) @ [p.beta0, p.beta1, p.beta2, p.beta3]
            for p in params.itertuples()
        ]
    ).T
    start, end = params.index[0], params.index[-1]
    fig = _figure(
        t,
        f"The U.S. Treasury curve, {start:%Y}-{end:%Y}",
        "Nelson-Siegel-Svensson zero curve fitted to FRED par yields every week (shown monthly); "
        "latest curve on the right",
    )
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2.35, 1], left=0.06, right=0.985, top=0.8, bottom=0.14, wspace=0.24
    )
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("ramp", list(t.ramp))
    x0, x1 = mdates.date2num(start), mdates.date2num(end)
    im = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=(x0, x1, grid[0], grid[-1]),
        cmap=cmap,
        vmin=0,
        vmax=max(8.0, float(np.nanmax(z))),
        interpolation="nearest",
    )
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style(ax, t)
    ax.grid(False)
    ax.set_ylabel("maturity (years)", color=t.ink2, fontsize=9)
    ax.set_yticks([0.25, 5, 10, 20, 30], ["3M", "5Y", "10Y", "20Y", "30Y"])
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t.ink2, labelsize=8, length=0)
    cb.ax.set_title("%", color=t.ink2, fontsize=8, loc="left")

    ax2 = fig.add_subplot(gs[1])
    _style(ax2, t)
    measure = "par" if r.config.calibration.target == "par" else "zero"
    fine = np.linspace(1 / 12, 30, 200)
    latest = r.fit.curve(-1)
    ax2.plot(
        fine, latest.evaluate(fine, measure), color=t.series[0], lw=2, label=f"{r.as_of:%d %b %Y}"
    )
    year_ago = r.as_of - pd.DateOffset(years=1)
    if r.fit.params.index[0] <= year_ago:
        old = r.fit.curve(year_ago)
        ax2.plot(fine, old.evaluate(fine, measure), color=t.muted, lw=1.5, label="a year earlier")
    obs = r.yields.loc[r.as_of].dropna()
    ax2.plot(
        np.asarray(obs.index, dtype=float),
        obs.to_numpy(),
        "o",
        ms=5.5,
        color=t.series[0],
        mec=t.surface,
        mew=1.5,
        label="quotes",
        zorder=5,
    )
    ax2.set_xlabel("maturity (years)", color=t.ink2, fontsize=9)
    ax2.set_ylabel(f"{measure} yield (%)", color=t.ink2, fontsize=9)
    ax2.set_xlim(0, 31)
    leg = ax2.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=t.ink2)
    for h in leg.legend_handles:
        if h is not None:
            h.set_alpha(1)
    return fig


def _term_premium_chart(r: PipelineResult, t: _Theme) -> Any:
    if r.acm is None:
        return None
    series: list[tuple[str, pd.Series, str, float]] = []
    real_time = r.term_premium_real_time
    if real_time is not None:
        series.append(("Plain ACM", _monthly(real_time["term_premium"]), t.series[0], 1.6))
    if r.term_premium_real_time_survey is not None:
        rts = _monthly(r.term_premium_real_time_survey["term_premium"])
        series.append(("Survey-anchored ACM", rts, t.series[1], 2.4))
    if not series:
        return None
    start = min(s.index[0] for _, s, _, _ in series)
    kw = r.term_premium_benchmarks.get("Kim-Wright (Fed Board)")
    if kw is not None:
        series.append(("Kim-Wright (Fed Board)", _monthly(kw).loc[start:], t.ink2, 1.3))
    fig = _figure(
        t,
        "What investors demand for holding a 10-year bond",
        "10-year term premium, re-estimated every month using only the yields and surveys "
        "published by then (%)",
    )
    ax = fig.add_axes((0.06, 0.12, 0.8, 0.62))
    _style(ax, t)
    shaded = _shade_recessions(ax, r.recession, start, t)
    ax.axhline(0, color=t.axis, lw=1)
    for name, s, color, lw in series:
        ax.plot(s.index, s.to_numpy(), color=color, lw=lw, label=name, solid_capstyle="round")
    _clip_outliers(ax, series, t)
    ends = sorted(
        ((float(s.dropna().iloc[-1]), name, s, c) for name, s, c, _ in series), key=lambda v: v[0]
    )
    offsets = _spread_labels([e[0] for e in ends], ax)
    for (_, _name, s, c), dy in zip(ends, offsets, strict=True):
        _end_label(ax, s, f"{float(s.dropna().iloc[-1]):+.2f}", t, c, dy)
    handles, labels = ax.get_legend_handles_labels()
    if shaded:
        from matplotlib.patches import Patch

        handles.append(Patch(color=t.recession, lw=0))
        labels.append("NBER recession")
    _legend(fig, handles, labels, t)
    return fig


def _clip_outliers(ax: Any, series: list[tuple[str, pd.Series, str, float]], t: _Theme) -> None:
    """Fit the y-axis to the bulk of the data and label the points left off scale.

    The first real-time estimates of plain ACM, from five years of data, can be
    several points away from everything else; one such month would flatten the
    whole chart. They are not hidden: each series' most extreme off-scale value
    is written at the edge of the plot.
    """
    values = np.concatenate([s.dropna().to_numpy() for _, s, _, _ in series])
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = 0.12 * (hi - lo)
    lo, hi = min(lo - pad, 0.0), hi + pad
    ax.set_ylim(lo, hi)
    for name, s, color, _ in series:
        s = s.dropna()
        for side, mask in (("below", s < lo), ("above", s > hi)):
            if not mask.any():
                continue
            worst = s[mask].idxmin() if side == "below" else s[mask].idxmax()
            y = lo if side == "below" else hi
            ax.annotate(
                f"{name}: {s[worst]:+.1f}% in {worst:%b %Y} (off scale)",
                (worst, y),
                xytext=(6, 8 if side == "below" else -8),
                textcoords="offset points",
                va="bottom" if side == "below" else "top",
                fontsize=8.5,
                color=t.ink2,
            )
            ax.plot([worst], [y], marker="v" if side == "below" else "^", ms=6, color=color)


def _legend(fig: Any, handles: list[Any], labels: list[str], t: _Theme) -> None:
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.052, 0.835),
        labelcolor=t.ink2,
        ncols=len(labels),
        handlelength=1.6,
        columnspacing=1.4,
    )


def _spread_labels(values: list[float], ax: Any, min_gap_pt: float = 11.0) -> list[float]:
    """Vertical offsets (points) that keep sorted end labels from overlapping."""
    lo, hi = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    per_unit = height_pt / (hi - lo) if hi > lo else 1.0
    pos = [v * per_unit for v in values]
    placed: list[float] = []
    for p in pos:
        placed.append(max(p, placed[-1] + min_gap_pt) if placed else p)
    return [q - p for p, q in zip(pos, placed, strict=True)]


def _recession_chart(r: PipelineResult, t: _Theme) -> Any:
    if r.recession_model is None:
        return None
    lines: list[tuple[str, pd.Series, str, float]] = []
    rt = r.recession_real_time
    if rt is not None:
        fwd = [c for c in rt.columns if c.startswith("near-term")]
        if fwd:
            lines.append(
                (
                    "near-term forward spread, real time",
                    _monthly(rt[fwd[0]] * 100),
                    t.series[1],
                    2.2,
                )
            )
    lines.insert(
        0,
        (
            "10Y−3M spread, in sample (NY Fed model)",
            _monthly(r.recession_model.fitted * 100),
            t.series[0],
            1.6,
        ),
    )
    start = lines[0][1].index[0]
    horizon = r.recession_model.horizon
    fig = _figure(
        t,
        "Does the yield curve see recessions coming?",
        f"Probability of a U.S. recession within {horizon} months from probit models on the "
        "curve (%); shaded: NBER recessions",
    )
    ax = fig.add_axes((0.06, 0.12, 0.8, 0.62))
    _style(ax, t)
    shaded = _shade_recessions(ax, r.recession, start, t)
    for name, s, color, lw in lines:
        ax.plot(s.index, s.to_numpy(), color=color, lw=lw, label=name)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ends = sorted(((float(s.dropna().iloc[-1]), s, c) for _, s, c, _ in lines), key=lambda v: v[0])
    for (v, s, c), dy in zip(ends, _spread_labels([e[0] for e in ends], ax), strict=True):
        _end_label(ax, s, f"{v:.0f}%", t, c, dy)
    handles, labels = ax.get_legend_handles_labels()
    if shaded:
        from matplotlib.patches import Patch

        handles.append(Patch(color=t.recession, lw=0))
        labels.append("NBER recession")
    _legend(fig, handles, labels, t)
    return fig
