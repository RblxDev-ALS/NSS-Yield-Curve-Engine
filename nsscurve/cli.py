"""Command-line interface: ``python nss_engine.py --help``."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .calibration import CalibrationConfig
from .dashboard import build_dashboard, save_dashboard
from .data import DataError
from .pipeline import PipelineConfig, export_results, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nss_engine",
        description="Nelson-Siegel-Svensson yield curve engine & macro regime tracker.")
    src = p.add_argument_group("data")
    src.add_argument("--source", choices=["fred", "csv", "synthetic"], default="fred",
                     help="market data source (default: fred)")
    src.add_argument("--offline", action="store_true",
                     help="shortcut for --source synthetic (no network needed)")
    src.add_argument("--csv", dest="csv_path", help="rates CSV for --source csv")
    src.add_argument("--years", type=float, default=5.0, help="history length (default 5)")
    src.add_argument("--freq", choices=["D", "W", "M"], default="W",
                     help="sampling frequency (default W = Friday close)")
    src.add_argument("--fallback-synthetic", action="store_true",
                     help="use synthetic data if FRED cannot be reached")

    cal = p.add_argument_group("calibration")
    cal.add_argument("--model", choices=["nss", "ns"], default="nss")
    cal.add_argument("--fit-target", choices=["par", "zero"], default="par",
                     help="fit par yields exactly (default) or treat quotes as zeros")
    cal.add_argument("--ridge", type=float, default=CalibrationConfig.ridge)
    cal.add_argument("--smoothness", type=float, default=CalibrationConfig.smoothness)
    cal.add_argument("--lambda-smoothness", type=float,
                     default=CalibrationConfig.lambda_smoothness)
    cal.add_argument("--no-refine", action="store_true", help="grid search only (fastest)")

    sig = p.add_argument_group("signals & forecasting")
    sig.add_argument("--zscore-window", type=int, default=26)
    sig.add_argument("--zscore-threshold", type=float, default=2.0)
    sig.add_argument("--horizon", type=int, default=4, help="forecast horizon (periods)")
    sig.add_argument("--dynamics", choices=["ar", "var"], default="ar")
    sig.add_argument("--no-backtest", action="store_true")

    out = p.add_argument_group("output")
    out.add_argument("--output-dir", default="output")
    out.add_argument("--show", action="store_true", help="open the dashboard in a browser")
    out.add_argument("--no-dashboard", action="store_true")
    out.add_argument("--quiet", action="store_true")
    out.add_argument("--seed", type=int, default=7, help="seed for synthetic data")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cal = CalibrationConfig(model=args.model, fit_target=args.fit_target, ridge=args.ridge,
                            smoothness=args.smoothness,
                            lambda_smoothness=args.lambda_smoothness,
                            refine=not args.no_refine)
    return PipelineConfig(
        source="synthetic" if args.offline else args.source, csv_path=args.csv_path,
        years=args.years, freq=args.freq, calibration=cal,
        zscore_window=args.zscore_window, zscore_threshold=args.zscore_threshold,
        forecast_horizon=args.horizon, forecast_dynamics=args.dynamics,
        run_backtest=not args.no_backtest, output_dir=args.output_dir, seed=args.seed,
        verbose=not args.quiet)


def _print_report(result, log) -> None:
    s = result.summary
    with pd.option_context("display.width", 140, "display.max_columns", 20,
                           "display.float_format", "{:,.3f}".format):
        log(f"\n=== As of {s['as_of']} ===")
        log(f"Regime: {s['regime']}   |   recent curve move: {s['curve_move']}")
        if s["recession_probability_12m"] is not None:
            log(f"NY Fed-model recession probability (12m): {s['recession_probability_12m']:.1%}")
        log("\nLatest NSS parameters:")
        log(result.history.params.loc[result.history.valid].tail(3).to_string())
        log("\nRich/cheap (residual = market - model, bp):")
        log(result.rich_cheap.to_string())
        log("\nCarry + roll-down, 3M horizon (bp):")
        log(result.carry_roll.to_string())
        log("\nKey-rate durations of par bonds:")
        log(result.key_rates.to_string())
        if len(result.factor_validation):
            log("\nFactor validation (correlation with empirical proxies):")
            log(result.factor_validation.to_string(index=False))
        if result.backtest:
            bt = result.backtest
            log(f"\nDiebold-Li {bt['horizon']}-step forecast backtest "
                f"({bt['n_forecasts']} out-of-sample forecasts; ratio < 1 beats random walk):")
            log(bt["table"].to_string())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    log = (lambda *a, **k: None) if args.quiet else print
    try:
        result = run_pipeline(cfg)
    except DataError as exc:
        if not args.fallback_synthetic:
            print(f"ERROR: {exc}\nHint: re-run with --offline or --fallback-synthetic.",
                  file=sys.stderr)
            return 2
        log(f"FRED unavailable ({exc}); falling back to synthetic data.")
        cfg.source = "synthetic"
        result = run_pipeline(cfg)

    paths = export_results(result, cfg.output_dir)
    if not args.no_dashboard:
        fig = build_dashboard(result.history, result.regimes["Regime"], result.recession_prob,
                              result.market_spreads, result.rich_cheap, result.forecast)
        paths["dashboard.html"] = save_dashboard(fig, f"{cfg.output_dir}/dashboard.html",
                                                 show=args.show)
    _print_report(result, log)
    log(f"\nWrote {len(paths)} files to {cfg.output_dir}/ "
        f"({', '.join(sorted(p.name for p in paths.values())[:4])}, ...)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
