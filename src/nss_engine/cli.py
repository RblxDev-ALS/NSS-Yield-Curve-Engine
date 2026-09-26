"""Command-line interface: ``nss-engine <command>`` (or ``python -m nss_engine``)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from . import __version__
from .calibration import DEFAULT_PANEL_SMOOTHING, CalibrationConfig, calibrate
from .data import DataError, label_columns, load_treasury_yields, load_yields_csv, maturity_label
from .pipeline import PipelineConfig, run_pipeline, write_outputs
from .synthetic import simulate_market


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source",
        default="fred",
        help="'fred' (default, downloads from FRED), 'synthetic', or a path to a CSV file",
    )
    p.add_argument("--start", default="1990-01-01", help="first date (default 1990-01-01)")
    p.add_argument("--end", default=None, help="last date (default: latest available)")
    p.add_argument(
        "--freq", default="W-FRI", help="pandas frequency for resampling (default W-FRI)"
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --source synthetic")
    p.add_argument("--years", type=int, default=34, help="length of the --source synthetic sample")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", choices=["nss", "ns"], default="nss")
    p.add_argument(
        "--target",
        choices=["yield", "par"],
        default="par",
        help="treat quotes as par yields (default; correct for FRED CMT data) or fit the "
        "zero curve to them directly",
    )
    p.add_argument("--ridge", type=float, default=CalibrationConfig.ridge)
    p.add_argument(
        "--smoothing",
        type=float,
        default=DEFAULT_PANEL_SMOOTHING,
        help="penalty on week-to-week log-lambda changes (0 disables)",
    )
    p.add_argument(
        "--robust",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="down-weight bad quotes (Huber/bisquare reweighting). Off by default: on "
        "FRED data it mostly rejects persistent 10Y/20Y dislocations, and the curve then "
        "agrees less well with the Fed's",
    )


def _calibration_config(args: argparse.Namespace) -> CalibrationConfig:
    return CalibrationConfig(
        model=args.model,
        target=args.target,
        ridge=args.ridge,
        lambda_smoothing=args.smoothing,
        robust=args.robust,
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        source=args.source,
        start=args.start,
        end=args.end,
        freq=args.freq,
        seed=args.seed,
        synthetic_years=args.years,
        calibration=_calibration_config(args),
        run_forecasts=not args.no_forecast,
        reference_curve=not args.no_reference,
        term_premium=not args.no_term_premium,
    )
    t0 = time.perf_counter()
    result = run_pipeline(
        cfg, progress=lambda m: print(f"[{time.perf_counter() - t0:6.1f}s] {m}", file=sys.stderr)
    )
    paths = write_outputs(result, args.out, dashboard=not args.no_dashboard, offline=args.offline)
    s = result.summary
    print(
        f"\nAs of {s['as_of']}: regime {s['regime']['current']} "
        f"(slope {result.spreads['slope'].iloc[-1]:+.2f} pp), median fit RMSE "
        f"{s['fit_quality_bp']['rmse_median']:.2f} bp over {s['n_curves']} curves."
    )
    if "recession_model" in s:
        print(f"Recession probability (12m): {s['recession_model']['latest_probability']:.1%}")
    if "term_premium" in s:
        tp = s["term_premium"]["latest"]
        print(
            f"10Y zero yield {tp['fitted']:.2f}% = expected short rate "
            f"{tp['expected_short_rate']:.2f}% + term premium {tp['term_premium']:+.2f}%"
        )
    print("\nOutputs:")
    for name, path in paths.items():
        print(f"  {name:<13} {path}")
    if args.print_report:
        print("\n" + paths["report"].read_text())
    return 0


def cmd_curve(args: argparse.Namespace) -> int:
    """Fit and print a single day's curve."""
    if args.source == "synthetic":
        yields = simulate_market(seed=args.seed).yields
    elif args.source == "fred":
        start = (
            (pd.Timestamp(args.date) - pd.Timedelta(days=14))
            if args.date
            else pd.Timestamp.now() - pd.Timedelta(days=30)
        )
        yields = load_treasury_yields(start=start, end=args.date, freq=None, min_tenors=4)
    else:
        yields = load_yields_csv(args.source)
    if args.date:
        yields = yields.loc[: args.date]
    if yields.empty:
        raise DataError("no data on or before the requested date")
    row = yields.iloc[-1]
    cfg = replace(_calibration_config(args), lambda_smoothing=0.0)
    res = calibrate(np.asarray(row.index, dtype=float), row.to_numpy(), cfg)
    c = res.curve
    print(
        f"{args.model.upper()} curve for {yields.index[-1]:%Y-%m-%d}  "
        f"(RMSE {res.rmse_bp:.2f} bp, max |err| {res.max_abs_error_bp:.2f} bp, {res.runtime_ms:.1f} ms)\n"
    )
    for name, val in c.as_dict().items():
        print(f"  {name:<8} {val:9.4f}")
    print(f"\n  {'tenor':>5} {'observed':>9} {'fitted':>8} {'resid bp':>9} {'forward':>8}")
    for tau, o, f, r in zip(
        res.maturities, res.observed, res.fitted, res.residuals_bp, strict=True
    ):
        print(f"  {maturity_label(tau):>5} {o:9.3f} {f:8.3f} {r:9.2f} {c.forward(tau)[0]:8.3f}")
    print(f"\n  10y-2y {c.spread(10, 2):+.2f} pp · 10y-3m {c.spread(10, 0.25):+.2f} pp")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    yields = load_treasury_yields(args.start, args.end, freq=args.freq)
    label_columns(yields).to_csv(args.out, float_format="%.4f")
    print(f"wrote {len(yields)} rows × {yields.shape[1]} tenors to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nss-engine",
        description="Nelson-Siegel-Svensson yield curve engine for U.S. Treasuries.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="full pipeline: calibrate, analyse, write report + dashboard"
    )
    _add_data_args(p_run)
    _add_model_args(p_run)
    p_run.add_argument("--out", default="output", help="output directory (default ./output)")
    p_run.add_argument("--no-dashboard", action="store_true", help="skip the HTML dashboard")
    p_run.add_argument("--no-forecast", action="store_true", help="skip the forecast evaluation")
    p_run.add_argument(
        "--no-term-premium",
        action="store_true",
        help="skip the term premium decomposition (ACM model)",
    )
    p_run.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the comparison with the Federal Reserve's (GSW) curve",
    )
    p_run.add_argument(
        "--offline",
        action="store_true",
        help="embed plotly.js so the dashboard works without internet",
    )
    p_run.add_argument("--print-report", action="store_true", help="print the Markdown report")
    p_run.set_defaults(func=cmd_run)

    p_curve = sub.add_parser("curve", help="fit and print one day's curve")
    _add_data_args(p_curve)
    _add_model_args(p_curve)
    p_curve.add_argument("--date", default=None, help="date (default: latest)")
    p_curve.set_defaults(func=cmd_curve)

    p_dl = sub.add_parser("download", help="download Treasury yields from FRED to CSV")
    _add_data_args(p_dl)
    p_dl.add_argument("--out", default="treasury_yields.csv")
    p_dl.set_defaults(func=cmd_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
