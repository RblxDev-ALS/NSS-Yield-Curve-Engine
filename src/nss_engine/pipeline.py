"""End-to-end pipeline: data -> calibration -> analytics -> regimes -> forecasts -> outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analytics, forecasting, regime
from .calibration import DEFAULT_PANEL_SMOOTHING, CalibrationConfig, PanelFit, calibrate_panel
from .data import (
    DataError,
    label_columns,
    load_recession_indicator,
    load_treasury_yields,
    load_yields_csv,
    maturity_label,
)
from .models import NSSCurve, curvature_peak
from .synthetic import simulate_market

#: Standard tenors reported in risk / carry tables.
REPORT_TENORS = (2.0, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for :func:`run_pipeline`.

    ``source`` is ``"fred"`` (download), ``"synthetic"`` (simulated market with
    known parameters) or a path to a CSV file (see :func:`load_yields_csv`).
    """

    source: str = "fred"
    start: str | None = "1990-01-01"
    end: str | None = None
    freq: str = "W-FRI"
    calibration: CalibrationConfig = field(
        default_factory=lambda: CalibrationConfig(lambda_smoothing=DEFAULT_PANEL_SMOOTHING)
    )
    slope_long: float = 10.0
    slope_short: float = 0.25  # 10y-3m: the NY Fed / Estrella-Mishkin spread
    forecast_horizons: tuple[int, ...] = (1, 6, 12)
    forecast_min_train: int = 60
    recession_horizon: int = 12
    rv_window: int = 52
    seed: int = 0
    synthetic_years: int = 34
    run_forecasts: bool = True


@dataclass
class PipelineResult:
    config: PipelineConfig
    yields: pd.DataFrame
    fit: PanelFit
    spreads: pd.DataFrame
    regimes: pd.Series
    dynamics: pd.Series
    recession: pd.Series | None
    recession_model: regime.RecessionModel | None
    lead_times: pd.DataFrame | None
    pca: analytics.PCAResult
    proxy_correlations: pd.DataFrame
    forecast_eval: forecasting.ForecastEvaluation | None
    forecast_curve: pd.Series | None
    rich_cheap: pd.DataFrame
    half_life: pd.Series
    carry: pd.DataFrame
    risk: dict[str, analytics.RiskReport]
    true_params: pd.DataFrame | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.fit.params.index[-1])


def load_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame | None]:
    """Return ``(yields, recession_indicator, true_params)`` for the configured source."""
    if cfg.source == "synthetic":
        periods = int(cfg.synthetic_years * 12 * _periods_per_month(cfg.freq))
        start = cfg.start or "1990-01-05"
        mkt = simulate_market(start=start, periods=periods, freq=cfg.freq, seed=cfg.seed)
        return mkt.yields, mkt.recession, mkt.true_params
    if cfg.source == "fred":
        yields = load_treasury_yields(cfg.start, cfg.end, freq=cfg.freq)
        try:
            rec: pd.Series | None = load_recession_indicator(cfg.start, cfg.end)
        except DataError:
            rec = None
        return yields, rec, None
    path = Path(cfg.source)
    if not path.exists():
        raise DataError(f"unknown source {cfg.source!r} (use 'fred', 'synthetic' or a CSV path)")
    yields = load_yields_csv(path).loc[slice(cfg.start, cfg.end)]
    return yields, None, None


def run_pipeline(
    cfg: PipelineConfig | None = None,
    progress: Callable[[str], None] | None = None,
    data: tuple[pd.DataFrame, pd.Series | None, pd.DataFrame | None] | None = None,
) -> PipelineResult:
    """Run the full analysis. ``data`` can be passed to skip loading (useful in tests)."""
    cfg = cfg or PipelineConfig()
    say = progress or (lambda _msg: None)

    say(f"loading data (source={cfg.source})")
    yields, recession, true_params = data if data is not None else load_data(cfg)
    if yields.empty:
        raise DataError("no yield data in the requested range")
    say(
        f"calibrating {len(yields)} curves ({cfg.calibration.model.upper()}, target={cfg.calibration.target})"
    )
    fit = calibrate_panel(yields, cfg.calibration)
    if fit.params.empty:
        raise RuntimeError("calibration failed on every date")

    # ---- spreads & regimes ------------------------------------------------------
    say("classifying regimes")
    measure = "par" if cfg.calibration.target == "par" else "zero"
    tenors = sorted({0.25, 2.0, 5.0, 10.0, cfg.slope_short, cfg.slope_long})
    model_on = fit.evaluate(tenors, measure)
    spreads = pd.DataFrame(
        {
            "model_10y3m": model_on[10.0] - model_on[0.25],
            "model_10y2y": model_on[10.0] - model_on[2.0],
        }
    )
    obs = yields.reindex(fit.params.index)
    for name, (lo, hi) in {"obs_10y3m": (0.25, 10.0), "obs_10y2y": (2.0, 10.0)}.items():
        if lo in obs.columns and hi in obs.columns:
            spreads[name] = obs[hi] - obs[lo]
    spreads["slope"] = model_on[cfg.slope_long] - model_on[cfg.slope_short]
    regimes = regime.classify_slope(spreads["slope"])
    level = model_on[[2.0, 5.0, 10.0]].mean(axis=1)
    dynamics = regime.curve_dynamics(
        level, spreads["model_10y2y"], window=_periods_per_month(cfg.freq)
    )

    rec_model = None
    lead_times = None
    if recession is not None and recession.sum() > 0:
        say("fitting recession probit")
        try:
            rec_model = regime.recession_probability_model(
                spreads["slope"], recession, cfg.recession_horizon
            )
        except ValueError:
            rec_model = None
        lead_times = regime.inversion_lead_times(regime.to_monthly(spreads["slope"]), recession)

    # ---- factor validation --------------------------------------------------------
    say("validating factors (PCA)")
    pca_res = analytics.pca(yields.dropna(axis=1, thresh=int(0.8 * len(yields))))
    # Free-λ NSS betas vs fixed-λ Diebold-Li factors: with λ free, β0 is an
    # asymptote beyond the data and −β1 an infinite-maturity spread, so they track
    # the textbook proxies less closely - which is why regimes use the
    # model-implied 10y−3m spread rather than −β1.
    dl = forecasting.extract_factors(yields).set_axis(["beta0", "beta1", "beta2"], axis=1)
    proxy_corr = pd.DataFrame(
        {
            "NSS (free λ)": analytics.factor_proxy_correlations(fit.params, yields),
            "Diebold-Li (fixed λ)": analytics.factor_proxy_correlations(dl, yields),
        }
    )
    proxy_corr.index.name = "factor ~ proxy"

    # ---- forecasts ------------------------------------------------------------------
    fc_eval, fc_curve = None, None
    if cfg.run_forecasts:
        monthly = yields.resample("ME").mean().dropna(how="all")
        core = monthly.dropna(axis=1, thresh=int(0.9 * len(monthly))).dropna()
        if len(core) > cfg.forecast_min_train + max(cfg.forecast_horizons):
            say(f"evaluating Diebold-Li forecasts on {len(core)} months")
            fc_eval = forecasting.evaluate_forecasts(
                core, cfg.forecast_horizons, cfg.forecast_min_train
            )
            fc_curve = forecasting.forecast_curve(
                core, max(cfg.forecast_horizons), maturities=list(yields.columns)
            )

    # ---- relative value, carry, risk -------------------------------------------------
    say("computing relative value and risk")
    rich_cheap = analytics.rich_cheap(fit.residuals_bp, window=cfg.rv_window)
    half_life = analytics.residual_half_life(fit.residuals_bp)
    last_curve = fit.curve(-1)
    carry = analytics.carry_rolldown(last_curve, REPORT_TENORS, horizon=0.25)
    risk = {
        f"{t:g}Y par bond": analytics.risk_report(last_curve, analytics.Bond.par(last_curve, t))
        for t in REPORT_TENORS
    }

    result = PipelineResult(
        config=cfg,
        yields=yields,
        fit=fit,
        spreads=spreads,
        regimes=regimes,
        dynamics=dynamics,
        recession=recession,
        recession_model=rec_model,
        lead_times=lead_times,
        pca=pca_res,
        proxy_correlations=proxy_corr,
        forecast_eval=fc_eval,
        forecast_curve=fc_curve,
        rich_cheap=rich_cheap,
        half_life=half_life,
        carry=carry,
        risk=risk,
        true_params=true_params,
    )
    result.summary = build_summary(result)
    say("done")
    return result


def _periods_per_month(freq: str) -> int:
    f = freq.upper()
    if f.startswith("W"):
        return 4
    if f in ("B", "D"):
        return 21
    return 1


# =============================================================================
# Summary & outputs
# =============================================================================


def _clean(obj: Any) -> Any:
    """Make an object JSON-serialisable (NaN -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj).date())
    return obj


def build_summary(r: PipelineResult) -> dict[str, Any]:
    """Headline numbers for the report / JSON export."""
    diag = r.fit.diagnostics
    curve = r.fit.curve(-1)
    spells = regime.regime_spells(r.regimes)
    current = spells.iloc[-1]
    s: dict[str, Any] = {
        "as_of": r.as_of,
        "source": r.config.source,
        "n_curves": len(r.fit.params),
        "sample_start": r.fit.params.index[0],
        "calibration": asdict(r.config.calibration),
        "latest_curve": curve.as_dict()
        | {
            "short_rate": curve.short_rate,
            "curvature_peak_years": curvature_peak(curve.lambda1),
        },
        "fit_quality_bp": {
            "rmse_median": diag["rmse_bp"].median(),
            "rmse_mean": diag["rmse_bp"].mean(),
            "rmse_p95": diag["rmse_bp"].quantile(0.95),
            "max_abs_error_median": diag["max_abs_error_bp"].median(),
            "runtime_ms_median": diag["runtime_ms"].median(),
            "success_rate": float(diag["success"].mean()),
            "share_ns_fallback": float((diag["model"] == "ns").mean()),
        },
        "regime": {
            "current": str(current["regime"]),
            "since": current["start"],
            "weeks_in_regime": int(current["periods"]),
            "curve_dynamics": r.dynamics.dropna().iloc[-1] if r.dynamics.notna().any() else None,
            "slope_definition": f"{maturity_label(r.config.slope_long)} − {maturity_label(r.config.slope_short)}",
            "share_of_time": r.regimes.value_counts(normalize=True).to_dict(),
        },
        "spreads_latest_pct": r.spreads.iloc[-1].to_dict(),
        "spread_tracking": _spread_tracking(r.spreads),
        "pca_explained_variance": r.pca.explained_variance_ratio.to_dict(),
        "factor_proxy_correlations": r.proxy_correlations.to_dict(orient="index"),
        "rich_cheap_latest": r.rich_cheap.reset_index().to_dict(orient="records"),
        "residual_half_life_weeks": r.half_life.to_dict(),
    }
    if r.recession_model is not None:
        m = r.recession_model
        s["recession_model"] = {
            "horizon_months": m.horizon,
            "coef_const": m.model.coef[0],
            "coef_spread": m.model.coef[1],
            "z_spread": m.model.zstats[1],
            "pseudo_r2": m.model.pseudo_r2,
            "auc": m.auc,
            "latest_probability": m.latest_probability,
            "target_month": m.target_date,
        }
    if r.lead_times is not None and not r.lead_times.empty:
        lt = r.lead_times
        s["inversions"] = {
            "episodes": len(lt),
            "followed_by_recession": int(lt["lead_months"].notna().sum()),
            "median_lead_months": lt["lead_months"].median(),
        }
    if r.forecast_eval is not None:
        rel = r.forecast_eval.relative_rmse
        s["forecast_relative_rmse"] = {f"h={h}": rel.loc[h].to_dict() for h in rel.index}
    if r.true_params is not None:
        s["synthetic_truth"] = _truth_errors(r)
    return _clean(s)


def _spread_tracking(spreads: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for tenor in ("10y3m", "10y2y"):
        m, o = f"model_{tenor}", f"obs_{tenor}"
        if m in spreads and o in spreads:
            diff = (spreads[m] - spreads[o]).dropna()
            out[f"{tenor}_corr"] = float(spreads[m].corr(spreads[o]))
            out[f"{tenor}_mae_bp"] = float(diff.abs().mean() * 100)
    return out


def _truth_errors(r: PipelineResult) -> dict[str, float]:
    """For synthetic data: accuracy of the fitted curves against the true curves."""
    grid = np.linspace(0.25, 30, 60)
    tp = r.true_params.reindex(r.fit.params.index) if r.true_params is not None else None
    if tp is None:
        return {}
    true = np.array([NSSCurve.from_array(p).zero(grid) for p in tp.to_numpy()])
    est = r.fit.evaluate(grid).to_numpy()
    return {"curve_rmse_vs_truth_bp": float(np.sqrt(np.mean((est - true) ** 2)) * 100)}


def write_outputs(
    result: PipelineResult, out_dir: str | Path, dashboard: bool = True
) -> dict[str, Path]:
    """Write CSV / JSON / Markdown / HTML outputs. Returns the paths written."""
    from .report import render_markdown

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["parameters"] = out / "nss_parameters.csv"
    result.fit.to_frame().to_csv(paths["parameters"], float_format="%.6f")

    paths["fitted"] = out / "fitted_yields.csv"
    label_columns(result.fit.fitted).to_csv(paths["fitted"], float_format="%.4f")

    paths["residuals"] = out / "residuals_bp.csv"
    label_columns(result.fit.residuals_bp).to_csv(paths["residuals"], float_format="%.3f")

    signals = result.spreads.copy()
    signals["regime"] = result.regimes.astype(str)
    signals["dynamics"] = result.dynamics
    if result.recession_model is not None:
        prob = result.recession_model.fitted
        signals = signals.join(
            prob.reindex(signals.index, method="ffill").rename("recession_prob_12m"), how="left"
        )
    paths["signals"] = out / "macro_signals.csv"
    signals.to_csv(paths["signals"], float_format="%.4f")

    paths["summary"] = out / "summary.json"
    paths["summary"].write_text(json.dumps(result.summary, indent=2, default=str))

    paths["report"] = out / "report.md"
    paths["report"].write_text(render_markdown(result))

    if dashboard:
        from .viz import build_dashboard

        paths["dashboard"] = out / "dashboard.html"
        build_dashboard(result, paths["dashboard"])
    return paths
