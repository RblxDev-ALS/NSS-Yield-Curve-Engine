import json

import pandas as pd
import pytest

from nsscurve.calibration import CalibrationConfig
from nsscurve.cli import main
from nsscurve.pipeline import PipelineConfig, export_results, run_pipeline


def test_cli_offline_end_to_end(tmp_path, capsys):
    rc = main(["--offline", "--years", "2", "--output-dir", str(tmp_path), "--quiet"])
    assert rc == 0
    for name in ["nss_macro_signals.csv", "residuals_bp.csv", "rich_cheap.csv", "regimes.csv",
                 "forecast.csv", "forecast_backtest.csv", "carry_rolldown.csv",
                 "key_rate_durations.csv", "summary.json", "dashboard.html"]:
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["fit"]["mean_rmse_bp"] < 5
    assert summary["regime"] in {"Expansionary (Steep)", "Transition (Flat)",
                                 "Recession Warning (Inverted)"}
    params = pd.read_csv(tmp_path / "nss_macro_signals.csv", index_col=0)
    assert list(params.columns) == ["Beta0", "Beta1", "Beta2", "Beta3", "Lambda1", "Lambda2"]
    assert capsys.readouterr().out == ""


def test_pipeline_ns_monthly_from_csv(tmp_path):
    from nsscurve.data import simulate_curve
    csv = tmp_path / "rates.csv"
    simulate_curve(start="2018-01-01", periods=800, seed=2).to_csv(csv)
    cfg = PipelineConfig(source="csv", csv_path=str(csv), freq="M", verbose=False,
                         calibration=CalibrationConfig(model="ns"), run_backtest=False,
                         zscore_window=12)
    res = run_pipeline(cfg)
    assert "Beta3" not in res.history.params.columns
    assert res.forecast is not None
    paths = export_results(res, tmp_path / "out")
    assert paths["summary.json"].exists()


def test_cli_reports_data_errors(monkeypatch, tmp_path, capsys):
    from nsscurve import pipeline
    from nsscurve.data import DataError

    def fail(cfg):
        raise DataError("network down")

    monkeypatch.setattr(pipeline, "load_rates", fail)
    rc = main(["--output-dir", str(tmp_path), "--no-dashboard"])
    assert rc == 2
    assert "network down" in capsys.readouterr().err


def test_bad_source():
    with pytest.raises(ValueError):
        run_pipeline(PipelineConfig(source="bloomberg", verbose=False))
