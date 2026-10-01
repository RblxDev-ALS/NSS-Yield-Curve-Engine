"""End-to-end pipeline: data -> calibration -> analytics -> regimes -> forecasts -> outputs."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import analytics, forecasting, inflation, regime, statespace, termpremium
from .calibration import (
    DEFAULT_PANEL_SMOOTHING,
    CalibrationConfig,
    FitResult,
    PanelFit,
    calibrate,
    calibrate_panel,
)
from .data import (
    DataError,
    label_columns,
    load_breakevens,
    load_gsw_parameters,
    load_gsw_tips_parameters,
    load_kim_wright_term_premium,
    load_recession_indicator,
    load_spf_bill_forecasts,
    load_tips_yields,
    load_treasury_yields,
    load_yields_csv,
    maturity_label,
)
from .models import NSSCurve, curvature_peak
from .synthetic import simulate_market, simulate_tips_market
from .validation import ReferenceComparison, compare_to_reference

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
    #: Compare with the Fed's GSW curve (FRED source) or the true curve (synthetic).
    reference_curve: bool = True
    #: Split yields into expected short rates and a term premium (ACM model).
    term_premium: bool = True
    #: Months of data before the first pseudo-real-time term premium estimate.
    term_premium_min_train: int = 60
    #: Anchor the term premium's expectations to SPF survey forecasts (FRED source).
    #: When the surveys are available this is the headline estimate; plain ACM is
    #: reported next to it.
    term_premium_surveys: bool = True
    #: Also re-estimate the survey-anchored premium every month on past data (about
    #: three minutes on 1990-2026 data); the recession test then uses it.
    term_premium_real_time_surveys: bool = True
    #: Fit the TIPS real curve and derive breakeven inflation (FRED and synthetic sources).
    inflation: bool = True


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
    latest_fit: FitResult | None = None
    recession_comparison: pd.DataFrame | None = None
    dns: statespace.DNSResult | None = None
    dns_forecast: pd.DataFrame | None = None
    reference: ReferenceComparison | None = None
    reference_name: str | None = None
    acm: termpremium.ACMResult | None = None
    #: The same model with expectations anchored to SPF surveys (FRED source only).
    acm_survey: termpremium.ACMResult | None = None
    #: Pseudo-real-time 10-year decomposition (each month estimated on past data only).
    term_premium_real_time: pd.DataFrame | None = None
    #: The same with survey anchors (each month using only surveys published by then).
    term_premium_real_time_survey: pd.DataFrame | None = None
    #: Other estimates of the 10-year term premium (Kim-Wright, ACM on the Fed's curve).
    term_premium_benchmarks: dict[str, pd.Series] = field(default_factory=dict)
    term_premium_comparison: pd.DataFrame | None = None
    #: Recession probits on the expectations and term-premium parts of the spread.
    term_premium_recession: pd.DataFrame | None = None
    #: Pseudo-real-time recession probabilities by signal (monthly, 12 months ahead).
    recession_real_time: pd.DataFrame | None = None
    #: Real (TIPS) curve fits.
    real_fit: PanelFit | None = None
    #: Breakeven inflation from the nominal and real curves (see :mod:`.inflation`).
    breakevens: pd.DataFrame | None = None
    #: Other breakeven measures: FRED's, the Fed's GSW curves, or the truth (synthetic).
    breakeven_benchmarks: pd.DataFrame | None = None
    breakeven_comparison: pd.DataFrame | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def as_of(self) -> pd.Timestamp:
        return pd.Timestamp(self.fit.params.index[-1])

    @property
    def headline_acm(self) -> termpremium.ACMResult | None:
        """The headline term-premium model: survey-anchored when surveys were available."""
        return self.acm_survey if self.acm_survey is not None else self.acm

    @property
    def term_premium_method(self) -> str:
        return "survey-anchored ACM" if self.acm_survey is not None else "ACM"

    @property
    def headline_real_time(self) -> pd.DataFrame | None:
        """Real-time 10-year decomposition of the headline model (plain ACM as fallback)."""
        if self.acm_survey is not None and self.term_premium_real_time_survey is not None:
            return self.term_premium_real_time_survey
        return self.term_premium_real_time


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
    latest_fit = _refit_latest(yields, fit, cfg.calibration)

    reference, reference_name = None, None
    ref_params = None
    if cfg.reference_curve:
        ref_params, reference_name = _reference_params(cfg, true_params)
        if ref_params is not None:
            say(f"comparing with {reference_name}")
            try:
                reference = compare_to_reference(fit, ref_params)
            except ValueError:
                reference = None

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
    spreads["near_term_fwd"] = regime.near_term_forward_spread(fit.params)
    regimes = regime.classify_slope(spreads["slope"])
    level = model_on[[2.0, 5.0, 10.0]].mean(axis=1)
    dynamics = regime.curve_dynamics(
        level, spreads["model_10y2y"], window=_periods_per_month(cfg.freq)
    )

    rec_model = None
    lead_times = None
    rec_comparison = None
    rec_rt = None
    if recession is not None and recession.sum() > 0:
        say("fitting recession probit")
        try:
            rec_model = regime.recession_probability_model(
                spreads["slope"], recession, cfg.recession_horizon
            )
        except ValueError:
            rec_model = None
        say("evaluating recession predictors in pseudo-real time")
        try:
            rec_comparison = regime.compare_recession_predictors(
                {
                    f"{_spread_label(cfg)} spread": spreads["slope"],
                    "near-term forward spread": spreads["near_term_fwd"],
                    "both": spreads[["slope", "near_term_fwd"]],
                },
                recession,
                cfg.recession_horizon,
            )
        except ValueError:
            rec_comparison = None
        lead_times = regime.inversion_lead_times(regime.to_monthly(spreads["slope"]), recession)
        rec_rt = _recession_real_time(spreads, recession, cfg)

    # ---- term premium ---------------------------------------------------------------
    acm, acm_survey, tp_rt, tp_rt_sv, tp_cmp, tp_rec = None, None, None, None, None, None
    tp_bench: dict[str, pd.Series] = {}
    if cfg.term_premium:
        say("decomposing yields into expected short rates and term premium (ACM)")
        gsw = ref_params if cfg.source == "fred" and true_params is None else None
        acm, acm_survey, tp_rt, tp_rt_sv, tp_bench, tp_cmp = _term_premium(fit, cfg, gsw, say)
        headline_rt = tp_rt_sv if acm_survey is not None and tp_rt_sv is not None else tp_rt
        if (
            acm is not None
            and headline_rt is not None
            and recession is not None
            and recession.sum() > 0
        ):
            say("testing the expectations and term-premium parts of the spread as predictors")
            tp_rec = _term_premium_recession(spreads["slope"], headline_rt, recession, cfg)

    # ---- real yields and breakeven inflation ---------------------------------------------
    real_fit, be, be_bench, be_cmp = None, None, None, None
    if cfg.inflation:
        real_fit, be, be_bench, be_cmp = _inflation(fit, cfg, true_params, ref_params, say)

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
    dns, dns_forecast = None, None
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
            say("fitting the state-space dynamic Nelson-Siegel model")
            dns, dns_forecast = _fit_dns(core, max(cfg.forecast_horizons))

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
        latest_fit=latest_fit,
        recession_comparison=rec_comparison,
        dns=dns,
        dns_forecast=dns_forecast,
        reference=reference,
        reference_name=reference_name if reference is not None else None,
        acm=acm,
        term_premium_real_time=tp_rt,
        acm_survey=acm_survey,
        term_premium_benchmarks=tp_bench,
        term_premium_comparison=tp_cmp,
        term_premium_recession=tp_rec,
        term_premium_real_time_survey=tp_rt_sv,
        recession_real_time=rec_rt,
        real_fit=real_fit,
        breakevens=be,
        breakeven_benchmarks=be_bench,
        breakeven_comparison=be_cmp,
    )
    result.summary = build_summary(result)
    say("done")
    return result


def _fit_dns(
    monthly: pd.DataFrame, horizon: int
) -> tuple[statespace.DNSResult | None, pd.DataFrame | None]:
    """State-space DNS fit and an ``horizon``-month forecast with an 80% interval."""
    try:
        dns = statespace.fit_dns(monthly)
    except (ValueError, np.linalg.LinAlgError):
        return None, None
    mean, cov = dns.forecast(horizon)
    sd = np.sqrt(np.diag(cov))
    z = 1.2816  # 80% central interval
    last = monthly.iloc[-1].to_numpy()
    table = pd.DataFrame(
        {
            "latest_pct": last,
            "forecast_pct": mean,
            "lower_80_pct": mean - z * sd,
            "upper_80_pct": mean + z * sd,
        },
        index=pd.Index([maturity_label(float(m)) for m in monthly.columns], name="tenor"),
    )
    return dns, table


def _term_premium(
    fit: PanelFit,
    cfg: PipelineConfig,
    gsw: pd.DataFrame | None,
    say: Callable[[str], None] = lambda _msg: None,
) -> tuple[
    termpremium.ACMResult | None,
    termpremium.ACMResult | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, pd.Series],
    pd.DataFrame | None,
]:
    """ACM decomposition of the fitted curves, plain and survey-anchored, in real time
    too, plus benchmarks."""
    zeros = termpremium.zero_panel(fit.params)
    try:
        acm = termpremium.fit_acm(zeros)
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None, None, {}, None
    acm_survey, surveys = None, None
    if cfg.source == "fred" and cfg.term_premium_surveys:
        with contextlib.suppress(DataError, ImportError, ValueError, np.linalg.LinAlgError):
            surveys = load_spf_bill_forecasts(end=cfg.end)
            acm_survey = termpremium.fit_acm(zeros, surveys=surveys)
    tp_rt, tp_rt_sv = None, None
    if len(zeros) > cfg.term_premium_min_train + 12:
        tp_rt = termpremium.real_time_decomposition(zeros, 10.0, cfg.term_premium_min_train)
        if acm_survey is not None and cfg.term_premium_real_time_surveys:
            say("re-estimating the survey-anchored term premium month by month (real time)")
            with contextlib.suppress(ValueError, np.linalg.LinAlgError):
                tp_rt_sv = termpremium.real_time_decomposition(
                    zeros, 10.0, cfg.term_premium_min_train, surveys=surveys
                )
    bench: dict[str, pd.Series] = {}
    if cfg.source == "fred":
        with contextlib.suppress(DataError):
            bench["Kim-Wright (Fed Board)"] = load_kim_wright_term_premium(cfg.start, cfg.end)
    if gsw is not None:
        try:
            gsw_acm = termpremium.fit_acm(termpremium.zero_panel(gsw.loc[fit.params.index[0] :]))
            bench["ACM on the Fed's GSW curve"] = gsw_acm.decomposition(10)["term_premium"]
        except (ValueError, np.linalg.LinAlgError):
            pass
    estimates = {"ACM on NSS curves (full sample)": acm.decomposition(10)["term_premium"]}
    if acm_survey is not None:
        estimates["ACM + SPF surveys (full sample)"] = acm_survey.decomposition(10)["term_premium"]
    if tp_rt is not None:
        estimates["ACM on NSS curves (real time)"] = tp_rt["term_premium"]
    if tp_rt_sv is not None:
        estimates["ACM + SPF surveys (real time)"] = tp_rt_sv["term_premium"]
    rows = {}
    for b_name, b in bench.items():
        for e_name, e in estimates.items():
            try:
                rows[(e_name, b_name)] = termpremium.compare_term_premia(e, b)
            except ValueError:
                continue
    cmp = None
    if rows:
        cmp = pd.DataFrame(rows).T
        cmp.index.names = ["estimate", "benchmark"]
    return acm, acm_survey, tp_rt, tp_rt_sv, bench, cmp


def _recession_real_time(
    spreads: pd.DataFrame, recession: pd.Series, cfg: PipelineConfig
) -> pd.DataFrame | None:
    """Pseudo-real-time recession probabilities of the two single-signal probits."""
    out = {}
    for name, col in (
        (f"{_spread_label(cfg)} spread", "slope"),
        ("near-term forward spread", "near_term_fwd"),
    ):
        with contextlib.suppress(ValueError):
            out[name] = regime.real_time_probabilities(
                spreads[col], recession, cfg.recession_horizon
            )
    return pd.DataFrame(out) if out else None


def _inflation(
    fit: PanelFit,
    cfg: PipelineConfig,
    true_params: pd.DataFrame | None,
    ref_params: pd.DataFrame | None,
    say: Callable[[str], None],
) -> tuple[PanelFit | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Fit the TIPS real curve; breakevens and their comparison with other measures."""
    bench: pd.DataFrame | None = None
    if cfg.source == "synthetic" and true_params is not None:
        try:
            tips = simulate_tips_market(true_params, seed=cfg.seed)
        except ValueError:  # sample ends before TIPS exist
            return None, None, None, None
        real = tips.yields
        bench = tips.true_breakevens().add_prefix("true ")
    elif cfg.source == "fred":
        start = max(pd.Timestamp(cfg.start or "2003-01-01"), pd.Timestamp("2003-01-01"))
        try:
            real = load_tips_yields(start, cfg.end, freq=cfg.freq)
        except DataError:
            return None, None, None, None
    else:
        return None, None, None, None
    if len(real) < 2:
        return None, None, None, None
    say(f"fitting the TIPS real curve on {len(real)} dates; breakeven inflation")
    real_fit = inflation.fit_real_curve(real)
    be = inflation.breakevens(fit.params, real_fit.params)
    if be.empty:
        return real_fit, None, None, None
    if cfg.source == "fred":
        frames = []
        with contextlib.suppress(DataError):
            fred = load_breakevens(be.index[0], cfg.end)
            frames.append(fred.rename(columns=lambda c: f"FRED {c}"))
        if ref_params is not None:
            with contextlib.suppress(DataError, ValueError):
                gsw_real = load_gsw_tips_parameters(be.index[0], cfg.end)
                gsw_be = inflation.breakevens(ref_params, gsw_real)
                frames.append(gsw_be[["be_5y", "be_10y", "be_5y5y"]].add_prefix("Fed GSW "))
        bench = pd.concat(frames, axis=1) if frames else None
    cmp = _breakeven_comparison(be, bench) if bench is not None else None
    return real_fit, be, bench, cmp


#: Which engine column each benchmark column measures.
_BREAKEVEN_PAIRS = {
    "FRED T5YIE": "be_par_5y",
    "FRED T10YIE": "be_par_10y",
    "FRED T5YIFR": "be_5y5y",
    "Fed GSW be_5y": "be_5y",
    "Fed GSW be_10y": "be_10y",
    "Fed GSW be_5y5y": "be_5y5y",
    "true be_5y": "be_5y",
    "true be_10y": "be_10y",
    "true be_5y5y": "be_5y5y",
}


def _breakeven_comparison(be: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame | None:
    rows = {}
    for b_col, e_col in _BREAKEVEN_PAIRS.items():
        if b_col in bench and e_col in be:
            with contextlib.suppress(ValueError):
                rows[(e_col, b_col)] = inflation.compare_series(be[e_col], bench[b_col])
    if not rows:
        return None
    out = pd.DataFrame(rows).T
    out.index.names = ["engine", "benchmark"]
    return out


def _term_premium_recession(
    slope: pd.Series, tp_rt: pd.DataFrame, recession: pd.Series, cfg: PipelineConfig
) -> pd.DataFrame | None:
    """Probits on the spread, its expectations component and the term premium alone.

    The 10-year term premium estimated in real time is subtracted from the
    10y−3m spread; what is left is the part of the slope explained by expected
    short rates (Rosenberg & Maurer, 2008). All candidates are scored on the
    same origins, which start later than the main comparison because the
    real-time premium needs its own training window.
    """
    monthly = regime.to_monthly(slope)
    tp = tp_rt["term_premium"].copy()
    tp.index = pd.DatetimeIndex(tp.index).to_period("M").to_timestamp("M")
    monthly.index = pd.DatetimeIndex(monthly.index).to_period("M").to_timestamp("M")
    frame = pd.concat([monthly.rename("spread"), tp.rename("tp")], axis=1).dropna()
    try:
        return regime.compare_recession_predictors(
            {
                f"{_spread_label(cfg)} spread": frame["spread"],
                "expectations component": frame["spread"] - frame["tp"],
                "term premium": frame["tp"],
            },
            recession,
            cfg.recession_horizon,
        )
    except ValueError:
        return None


def _spread_label(cfg: PipelineConfig) -> str:
    return f"{maturity_label(cfg.slope_long)}−{maturity_label(cfg.slope_short)}"


def _refit_latest(yields: pd.DataFrame, fit: PanelFit, cfg: CalibrationConfig) -> FitResult:
    """Re-run the last date's fit to get its full diagnostics (bands, leverage, weights)."""
    date = fit.params.index[-1]
    previous = fit.curve(-2) if len(fit.params) > 1 else None
    row = yields.loc[date]
    return calibrate(np.asarray(row.index, dtype=float), row.to_numpy(), cfg, previous=previous)


def _reference_params(
    cfg: PipelineConfig, true_params: pd.DataFrame | None
) -> tuple[pd.DataFrame | None, str | None]:
    if true_params is not None:
        return true_params, "the true curve (synthetic market)"
    if cfg.source == "fred":
        try:
            return load_gsw_parameters(cfg.start, cfg.end), "the Federal Reserve's GSW curve"
        except DataError:
            return None, None
    return None, None


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
            "rmse_clean_median": diag["rmse_clean_bp"].median(),
            "sigma_median": diag["sigma_bp"].median(),
            "share_lambda_at_bound": _share_at_bounds(r.fit),
            "not_converged_reasons": diag.loc[~diag["success"].astype(bool), "message"]
            .value_counts()
            .head(3)
            .to_dict(),
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
    if r.recession_comparison is not None:
        s["recession_predictors"] = r.recession_comparison.to_dict(orient="index")
    if r.lead_times is not None and not r.lead_times.empty:
        lt = r.lead_times
        s["inversions"] = {
            "episodes": len(lt),
            "followed_by_recession": int(lt["lead_months"].notna().sum()),
            "median_lead_months": lt["lead_months"].median(),
        }
    if r.dns is not None:
        s["state_space_dns"] = {
            "lambda": r.dns.params.lam,
            "loglik": r.dns.loglik,
            "persistence": np.diag(r.dns.params.A).tolist(),
            "noise_bp": dict(
                zip(
                    [maturity_label(float(m)) for m in r.dns.maturities],
                    (r.dns.params.h * 100).tolist(),
                    strict=True,
                )
            ),
        }
        if r.dns_forecast is not None:
            s["state_space_dns"]["forecast"] = r.dns_forecast.to_dict(orient="index")
    if r.forecast_eval is not None:
        rel = r.forecast_eval.relative_rmse
        s["forecast_relative_rmse"] = {f"h={h}": rel.loc[h].to_dict() for h in rel.index}
    if r.acm is not None:
        head = r.headline_acm
        assert head is not None
        rt = r.headline_real_time
        tp = head.decomposition(10)["term_premium"]
        s["term_premium"] = {
            "model": (
                "Adrian-Crump-Moench (2013), 5 principal components, real-world dynamics "
                "anchored to SPF surveys"
                if r.acm_survey is not None
                else "Adrian-Crump-Moench (2013), 5 principal components"
            ),
            "method": r.term_premium_method,
            "maturity_years": 10,
            "latest": head.decomposition(10).iloc[-1].to_dict(),
            "latest_real_time": _latest_real_time(rt),
            "real_time_share_var_capped": float(rt["var_capped"].mean())
            if rt is not None
            else None,
            "real_time_months_discarded": int(rt["fitted"].isna().sum())
            if rt is not None
            else None,
            "mean": float(tp.mean()),
            "min": float(tp.min()),
            "min_date": tp.idxmin(),
            "max": float(tp.max()),
            "max_date": tp.idxmax(),
            "fit_rmse_bp_mean": float(head.fit_rmse_bp.mean()),
            "var_max_eigenvalue": head.max_eigenvalue,
        }
        plain = r.acm.decomposition(10)
        s["term_premium"]["plain_acm"] = {
            "latest": plain.iloc[-1].to_dict(),
            "latest_real_time": _latest_real_time(r.term_premium_real_time),
            "mean": float(plain["term_premium"].mean()),
            "var_max_eigenvalue": r.acm.max_eigenvalue,
        }
        if r.acm_survey is not None:
            s["term_premium"]["survey_anchored"] = {
                "latest": r.acm_survey.decomposition(10).iloc[-1].to_dict(),
                "latest_real_time": _latest_real_time(r.term_premium_real_time_survey),
                "mean": float(r.acm_survey.decomposition(10)["term_premium"].mean()),
                "survey_rmse_pp": r.acm_survey.survey_rmse.to_dict(),
                "n_surveys": len(r.acm_survey.survey_fit)
                if r.acm_survey.survey_fit is not None
                else 0,
            }
        if r.term_premium_comparison is not None:
            s["term_premium"]["benchmarks"] = {
                f"{e} vs {b}": row.to_dict() for (e, b), row in r.term_premium_comparison.iterrows()
            }
    if r.term_premium_recession is not None:
        s["term_premium_recession_predictors"] = r.term_premium_recession.to_dict(orient="index")
        s["term_premium_recession_method"] = r.term_premium_method
    if r.recession_real_time is not None:
        s["recession_real_time_latest"] = r.recession_real_time.iloc[-1].to_dict()
    if r.breakevens is not None and r.real_fit is not None:
        be = r.breakevens
        s["inflation"] = {
            "as_of": be.index[-1],
            "latest": be.iloc[-1].to_dict(),
            "sample_start": be.index[0],
            "n_curves": len(be),
            "real_fit_rmse_bp_median": r.real_fit.diagnostics["rmse_bp"].median(),
            "real_fit_model": "Nelson-Siegel (4-5 TIPS quotes)",
            "be_5y5y_min": float(be["be_5y5y"].min()),
            "be_5y5y_min_date": be["be_5y5y"].idxmin(),
            "be_5y5y_max": float(be["be_5y5y"].max()),
            "be_5y5y_max_date": be["be_5y5y"].idxmax(),
        }
        if r.breakeven_comparison is not None:
            s["inflation"]["benchmarks"] = {
                f"{e} vs {b}": row.to_dict() for (e, b), row in r.breakeven_comparison.iterrows()
            }
    if r.true_params is not None:
        s["synthetic_truth"] = _truth_errors(r)
    if r.config.calibration.robust and r.fit.outliers is not None:
        flags = r.fit.outliers
        observed = r.fit.residuals_bp.notna()
        by_tenor = (flags.sum() / observed.sum().replace(0, np.nan)).rename(
            index=lambda c: maturity_label(float(c))
        )
        s["outliers"] = {
            "share_of_quotes": float(flags.to_numpy().sum() / max(observed.to_numpy().sum(), 1)),
            "share_by_tenor": by_tenor.to_dict(),
            "latest": [maturity_label(float(c)) for c in flags.columns[flags.iloc[-1].to_numpy()]],
        }
    if r.reference is not None:
        s["reference_curve"] = {
            "name": r.reference_name,
            **r.reference.overall(),
            "bias_bp": r.reference.summary()["bias_bp"].to_dict(),
        }
    return _clean(s)


def _latest_real_time(rt: pd.DataFrame | None) -> dict[str, float] | None:
    if rt is None:
        return None
    last = rt.drop(columns="var_capped").dropna()
    return last.iloc[-1].astype(float).to_dict() if not last.empty else None


def _share_at_bounds(fit: PanelFit) -> dict[str, float]:
    """Share of dates on which each decay rate sits at one of its bounds."""
    cfg = fit.config
    out = {}
    for name, (lo, hi) in (("lambda1", cfg.lambda1_bounds), ("lambda2", cfg.lambda2_bounds)):
        lam = fit.params[name]
        out[f"{name}_lower"] = float((lam <= lo * (1 + 1e-6)).mean())
        out[f"{name}_upper"] = float((lam >= hi * (1 - 1e-6)).mean())
    return out


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


def badges(result: PipelineResult) -> dict[str, dict[str, Any]]:
    """Live-status badges in the shields.io *endpoint* format.

    Published with the dashboard, they let a README show the latest reading,
    e.g. ``https://img.shields.io/endpoint?url=<site>/badges/regime.json``.
    """
    s = result.summary
    regime_colors = {"Inverted": "red", "Flat": "orange", "Normal": "blue", "Steep": "blue"}
    out: dict[str, dict[str, Any]] = {
        "regime": {
            "label": f"curve ({s['regime']['slope_definition'].replace(' ', '')})",
            "message": f"{s['regime']['current']} {result.spreads['slope'].iloc[-1]:+.2f}pp",
            "color": regime_colors.get(s["regime"]["current"], "lightgrey"),
        },
        "as_of": {"label": "data as of", "message": str(s["as_of"]), "color": "informational"},
    }
    if "recession_model" in s:
        p = s["recession_model"]["latest_probability"]
        out["recession"] = {
            "label": f"recession odds {s['recession_model']['horizon_months']}m",
            "message": f"{p:.0%}",
            "color": "red" if p >= 0.4 else "orange" if p >= 0.2 else "green",
        }
    if "term_premium" in s:
        tp = s["term_premium"]["latest"]["term_premium"]
        out["term_premium"] = {
            "label": "10Y term premium",
            "message": f"{tp:+.2f}%",
            "color": "informational",
        }
    if "inflation" in s:
        be = s["inflation"]["latest"]
        out["breakeven"] = {
            "label": "10Y breakeven",
            "message": f"{be['be_10y']:.2f}%",
            "color": "informational",
        }
        out["breakeven_5y5y"] = {
            "label": "5y5y breakeven",
            "message": f"{be['be_5y5y']:.2f}%",
            "color": "informational",
        }
    for badge in out.values():
        badge["schemaVersion"] = 1
    return out


def write_outputs(
    result: PipelineResult, out_dir: str | Path, dashboard: bool = True, offline: bool = False
) -> dict[str, Path]:
    """Write CSV / JSON / Markdown / HTML outputs. Returns the paths written.

    ``offline=True`` embeds plotly.js in the dashboard (about 4.5 MB) so it opens
    without internet access; by default it is loaded from the Plotly CDN.
    """
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
    if result.recession_real_time is not None:
        rt = result.recession_real_time.add_prefix("real-time recession prob, ")
        signals = signals.join(rt.reindex(signals.index, method="ffill"), how="left")
    paths["signals"] = out / "macro_signals.csv"
    signals.to_csv(paths["signals"], float_format="%.4f")

    if result.fit.outliers is not None and result.config.calibration.robust:
        paths["outliers"] = out / "outliers.csv"
        label_columns(result.fit.outliers.astype(int)).to_csv(paths["outliers"])

    if result.acm is not None:
        dec = result.acm.decomposition(10)
        if result.term_premium_real_time is not None:
            rt = result.term_premium_real_time.add_suffix("_real_time")
            dec = dec.join(rt, how="left")
        if result.acm_survey is not None:
            sv = result.acm_survey.decomposition(10)[["expected_short_rate", "term_premium"]]
            dec = dec.join(sv.add_suffix("_survey"), how="left")
        if result.term_premium_real_time_survey is not None:
            rts = result.term_premium_real_time_survey[["expected_short_rate", "term_premium"]]
            dec = dec.join(rts.add_suffix("_survey_real_time"), how="left")
        paths["term_premium"] = out / "term_premium.csv"
        dec.to_csv(paths["term_premium"], float_format="%.4f")

    if result.breakevens is not None:
        be = result.breakevens
        if result.breakeven_benchmarks is not None:
            bench = result.breakeven_benchmarks.reindex(
                be.index, method="ffill", tolerance=pd.Timedelta(days=4)
            )
            be = be.join(bench)
        paths["breakevens"] = out / "breakevens.csv"
        be.to_csv(paths["breakevens"], float_format="%.4f")
    if result.real_fit is not None:
        paths["real_parameters"] = out / "tips_nss_parameters.csv"
        result.real_fit.to_frame().to_csv(paths["real_parameters"], float_format="%.6f")

    if result.reference is not None:
        paths["reference"] = out / "reference_comparison.csv"
        result.reference.summary().to_csv(paths["reference"], float_format="%.3f")

    paths["summary"] = out / "summary.json"
    paths["summary"].write_text(json.dumps(result.summary, indent=2, default=str), encoding="utf-8")

    badge_dir = out / "badges"
    badge_dir.mkdir(exist_ok=True)
    for name, badge in badges(result).items():
        paths[f"badge_{name}"] = badge_dir / f"{name}.json"
        paths[f"badge_{name}"].write_text(json.dumps(badge), encoding="utf-8")

    paths["report"] = out / "report.md"
    paths["report"].write_text(render_markdown(result), encoding="utf-8")

    if dashboard:
        from .viz import build_dashboard

        paths["dashboard"] = out / "dashboard.html"
        build_dashboard(result, paths["dashboard"], include_plotlyjs=True if offline else "cdn")
    return paths
