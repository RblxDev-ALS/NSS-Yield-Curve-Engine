"""End-to-end pipeline: data -> calibration -> analytics -> signals -> outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, regimes, signals
from .calibration import CalibrationConfig, CurveHistory, NSSCalibrator
from .data import FREDClient, clean_curve, load_csv, resample_curve, simulate_curve
from .forecasting import DieboldLiForecaster


@dataclass
class PipelineConfig:
    source: str = "fred"                   # "fred" | "csv" | "synthetic"
    csv_path: str | None = None
    years: float = 5.0
    freq: str = "W"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    zscore_window: int = 26
    zscore_threshold: float = 2.0
    regime_steep: float = 0.5
    regime_inverted: float = -0.1
    regime_band: float = 0.05
    regime_persistence: int = 2
    move_window: int = 4
    forecast_horizon: int = 4
    forecast_dynamics: str = "ar"
    run_backtest: bool = True
    output_dir: str = "output"
    seed: int = 7
    verbose: bool = True


@dataclass
class PipelineResult:
    rates: pd.DataFrame
    history: CurveHistory
    metrics: pd.DataFrame
    market_spreads: pd.DataFrame
    regimes: pd.DataFrame
    recession_prob: pd.Series
    rich_cheap: pd.DataFrame
    signal_quality: pd.DataFrame
    butterfly: pd.DataFrame
    carry_roll: pd.DataFrame
    key_rates: pd.DataFrame
    factor_validation: pd.DataFrame
    pca: dict
    forecast: pd.DataFrame | None
    backtest: dict | None
    summary: dict


def load_rates(cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.source == "synthetic":
        periods = int(cfg.years * 252)
        start = pd.Timestamp.today().normalize() - pd.offsets.BDay(periods)
        raw = simulate_curve(start=start, periods=periods, seed=cfg.seed)
        rates = clean_curve(raw)
    elif cfg.source == "csv":
        if not cfg.csv_path:
            raise ValueError("csv source requires csv_path")
        rates = load_csv(cfg.csv_path)
    elif cfg.source == "fred":
        start = pd.Timestamp.today().normalize() - pd.DateOffset(days=int(cfg.years * 365.25))
        rates = clean_curve(FREDClient().fetch_curve(start=start))
    else:
        raise ValueError(f"unknown source {cfg.source!r}")
    rates = resample_curve(rates, cfg.freq)
    if rates.empty:
        raise ValueError("no market data after cleaning/resampling")
    return rates


def run_pipeline(cfg: PipelineConfig, rates: pd.DataFrame | None = None) -> PipelineResult:
    log = print if cfg.verbose else (lambda *a, **k: None)
    if rates is None:
        log(f"--- Loading data (source={cfg.source}, {cfg.years}y, freq={cfg.freq}) ---")
        rates = load_rates(cfg)
    log(f"Data: {len(rates)} observations x {rates.shape[1]} tenors "
        f"({rates.index[0]:%Y-%m-%d} -> {rates.index[-1]:%Y-%m-%d})")

    c = cfg.calibration
    log(f"--- Calibrating {c.model.upper()} (target={c.fit_target}, ridge={c.ridge:g}, "
        f"smoothness={c.smoothness:g}, lambda_smoothness={c.lambda_smoothness:g}) ---")
    history = NSSCalibrator(c).fit_history(rates, progress=cfg.verbose)
    if len(history.valid) == 0:
        raise RuntimeError("calibration failed on every date")
    fit_summary = history.summary()
    log(f"Fit: mean RMSE {fit_summary['mean_rmse_bp']:.2f}bp, "
        f"p95 {fit_summary['p95_rmse_bp']:.2f}bp, worst point {fit_summary['worst_abs_error_bp']:.1f}bp")

    metrics = analytics.curve_metrics(history)
    spreads = analytics.market_spreads(rates)

    # --- regimes -------------------------------------------------------------
    slope = -history.params.loc[history.valid, "Beta1"]
    slope_regime = regimes.classify_slope_regime(
        slope, cfg.regime_steep, cfg.regime_inverted, cfg.regime_band, cfg.regime_persistence)
    level = (metrics["Par_2Y"] + metrics["Par_10Y"]) / 2
    moves = regimes.classify_curve_moves(level, metrics["Model_2Y10Y"], cfg.move_window)
    regime_df = pd.concat([slope.rename("ModelSlope"), slope_regime, moves], axis=1)
    spread_3m10y = spreads["3M10Y"] if "3M10Y" in spreads else metrics["Model_3M10Y"]
    rec_prob = regimes.recession_probability(spread_3m10y)

    # --- RV signals --------------------------------------------------------------
    resid = history.residuals_bp.loc[history.valid]
    rich_cheap = signals.rich_cheap_table(resid, cfg.zscore_window, cfg.zscore_threshold)
    quality = signals.signal_backtest(resid, cfg.zscore_window, cfg.zscore_threshold)
    fly = signals.butterfly_signals(rates.loc[history.valid], history.fitted.loc[history.valid],
                                    cfg.zscore_window)

    # --- risk / carry on the latest curve ------------------------------------------
    curve = history.curve()
    carry = analytics.carry_rolldown(curve)
    krd = analytics.key_rate_table(curve)
    validation = analytics.factor_validation(history)
    try:
        pca = analytics.pca_decomposition(rates)
    except ValueError:
        pca = {}

    # --- forecasting -----------------------------------------------------------------
    forecast, backtest = None, None
    try:
        dl = DieboldLiForecaster(model=c.model, dynamics=cfg.forecast_dynamics).fit(rates)
        forecast = dl.forecast(cfg.forecast_horizon)
        if cfg.run_backtest:
            min_train = max(52, len(rates) // 2)
            backtest = DieboldLiForecaster(model=c.model, dynamics=cfg.forecast_dynamics) \
                .backtest(rates, cfg.forecast_horizon, min_train=min_train)
    except ValueError as exc:
        log(f"Forecasting skipped: {exc}")

    last = history.valid[-1]
    summary = {
        "as_of": f"{last:%Y-%m-%d}",
        "config": {k: v for k, v in asdict(cfg).items() if k != "calibration"},
        "calibration_config": asdict(c),
        "fit": fit_summary,
        "latest_params": history.params.loc[last].dropna().to_dict(),
        "regime": slope_regime.iloc[-1],
        "curve_move": moves["Move"].iloc[-1],
        "recession_probability_12m": float(rec_prob.iloc[-1]) if len(rec_prob) else None,
        "rich_cheap_signals": rich_cheap.loc[rich_cheap.Signal != "FAIR", "Signal"].to_dict(),
        "forecast_rmse_ratio_vs_random_walk": (
            backtest["table"]["Ratio"].round(3).to_dict() if backtest else None),
    }
    return PipelineResult(rates, history, metrics, spreads, regime_df, rec_prob, rich_cheap,
                          quality, fly, carry, krd, validation, pca, forecast, backtest, summary)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def export_results(result: PipelineResult, output_dir: str | Path) -> dict[str, Path]:
    """Write every result table to ``output_dir``; returns name -> path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h = result.history
    tables = {
        # Historic file name kept for downstream compatibility.
        "nss_macro_signals.csv": h.params.loc[h.valid],
        "fit_diagnostics.csv": h.diagnostics,
        "market_rates.csv": result.rates,
        "fitted_rates.csv": h.fitted,
        "residuals_bp.csv": h.residuals_bp,
        "curve_metrics.csv": result.metrics,
        "regimes.csv": result.regimes,
        "recession_probability.csv": result.recession_prob.to_frame(),
        "rich_cheap.csv": result.rich_cheap,
        "signal_quality.csv": result.signal_quality,
        "butterfly_2s5s10s.csv": result.butterfly,
        "carry_rolldown.csv": result.carry_roll,
        "key_rate_durations.csv": result.key_rates,
        "factor_validation.csv": result.factor_validation,
    }
    if result.forecast is not None:
        tables["forecast.csv"] = result.forecast
    if result.backtest:
        tables["forecast_backtest.csv"] = result.backtest["table"]
    if result.pca:
        tables["pca_loadings.csv"] = result.pca["loadings"]
        tables["pca_explained_variance.csv"] = result.pca["explained_variance"].to_frame("Ratio")
    paths = {}
    for name, df in tables.items():
        p = out / name
        df.to_csv(p)
        paths[name] = p
    p = out / "summary.json"
    p.write_text(json.dumps(result.summary, indent=2, default=_json_default))
    paths["summary.json"] = p
    return paths
