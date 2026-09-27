import json

import numpy as np
import pandas as pd
import pytest

from nss_engine import cli
from nss_engine.calibration import CalibrationConfig
from nss_engine.pipeline import PipelineConfig, run_pipeline, write_outputs


@pytest.fixture(scope="module")
def result(long_market):
    monthly = long_market.yields.resample("ME").last()
    cfg = PipelineConfig(
        source="synthetic",
        freq="ME",
        forecast_horizons=(1, 12),
        calibration=CalibrationConfig(lambda_smoothing=1e-3, robust=True),
    )
    return run_pipeline(
        cfg, data=(monthly, long_market.recession, long_market.true_params.resample("ME").last())
    )


def test_pipeline_result(result):
    s = result.summary
    assert s["n_curves"] == len(result.fit.params) > 300
    assert s["fit_quality_bp"]["rmse_median"] < 4
    assert s["regime"]["current"] in {"Inverted", "Flat", "Normal", "Steep"}
    assert 0 <= s["recession_model"]["latest_probability"] <= 1
    assert s["recession_model"]["coef_spread"] < 0
    assert s["inversions"]["episodes"] >= 1
    assert s["spread_tracking"]["10y3m_corr"] > 0.99
    assert s["synthetic_truth"]["curve_rmse_vs_truth_bp"] < 4
    assert set(result.forecast_eval.relative_rmse.index) == {1, 12}
    assert result.forecast_curve is not None
    assert "10Y par bond" in result.risk
    # Validation against the known truth, robust-fit statistics, latest-fit bands
    ref = s["reference_curve"]
    assert "true curve" in ref["name"] and ref["rmse_bp"] < 5 and ref["mean_change_corr"] > 0.8
    assert abs(ref["bias_bp"]["10Y"]) < 2
    assert 0 <= s["outliers"]["share_of_quotes"] < 0.02
    preds = s["recession_predictors"]
    assert {"10Y−3M spread", "near-term forward spread", "both"} <= set(preds)
    assert 0 <= preds["both"]["auc_out_of_sample"] <= 1
    assert "near_term_fwd" in result.spreads
    dns = s["state_space_dns"]
    assert 0.1 < dns["lambda"] < 3
    fc = pd.DataFrame(dns["forecast"]).T
    assert (fc["lower_80_pct"] < fc["forecast_pct"]).all()
    assert (fc["forecast_pct"] < fc["upper_80_pct"]).all()
    assert s["fit_quality_bp"]["rmse_clean_median"] <= s["fit_quality_bp"]["rmse_median"]
    assert set(s["fit_quality_bp"]["share_lambda_at_bound"]) == {
        "lambda1_lower",
        "lambda1_upper",
        "lambda2_lower",
        "lambda2_upper",
    }
    lo, hi = result.latest_fit.confidence_band([2, 10])
    assert np.all(hi - lo > 0)
    tp = s["term_premium"]
    latest = tp["latest"]
    assert latest["fitted"] == pytest.approx(
        latest["expected_short_rate"] + latest["term_premium"], abs=1e-4
    )
    assert tp["fit_rmse_bp_mean"] < 5
    assert tp["real_time_months_discarded"] <= 3
    assert result.term_premium_real_time is not None
    assert result.term_premium_real_time.index[0] > result.acm.fitted.index[0]
    assert {"10Y−3M spread", "expectations component", "term premium"} <= set(
        s["term_premium_recession_predictors"]
    )
    json.dumps(s)  # fully serialisable


def test_write_outputs(result, tmp_path):
    paths = write_outputs(result, tmp_path)
    for key in ("parameters", "fitted", "residuals", "signals", "summary", "report", "dashboard"):
        assert paths[key].exists() and paths[key].stat().st_size > 0
    params = pd.read_csv(paths["parameters"], index_col=0, parse_dates=True)
    assert {"beta0", "lambda2", "rmse_bp"} <= set(params.columns)
    signals = pd.read_csv(paths["signals"], index_col=0)
    assert {"regime", "slope", "recession_prob_12m"} <= set(signals.columns)
    report = paths["report"].read_text(encoding="utf-8")
    assert "Synthetic data" in report and "Recession probability" in report
    assert "Validation against the true curve" in report and "zero_95ci_bp" in report
    assert "pseudo-real time" in report and "State-space dynamic Nelson-Siegel" in report
    assert paths["outliers"].exists() and paths["reference"].exists()
    assert "Term premium (Adrian-Crump-Moench)" in report
    badge = json.loads(paths["badge_regime"].read_text(encoding="utf-8"))
    assert badge["schemaVersion"] == 1
    assert badge["message"].split()[0] in {"Inverted", "Flat", "Normal", "Steep"}
    assert {"badge_recession", "badge_term_premium", "badge_as_of"} <= set(paths)
    assert json.loads(paths["badge_recession"].read_text(encoding="utf-8"))["message"].endswith("%")
    tp = pd.read_csv(paths["term_premium"], index_col=0)
    assert {"expected_short_rate", "term_premium", "term_premium_real_time"} <= set(tp.columns)
    html = paths["dashboard"].read_text(encoding="utf-8")
    assert html.count("plotly-graph-div") >= 8
    assert 'src="https://cdn.plot.ly' in html  # default: load plotly.js from the CDN
    # plotly.js must load before the first chart on the page
    assert html.index('src="https://cdn.plot.ly') < html.index("plotly-graph-div")
    assert "Validation against" in html and "Near-term forward spread" in html
    assert "prefers-color-scheme: dark" in html


def test_offline_dashboard_embeds_plotly(result, tmp_path):
    paths = write_outputs(result, tmp_path, offline=True)
    html = paths["dashboard"].read_text(encoding="utf-8")
    assert 'src="https://cdn.plot.ly' not in html and len(html) > 3_000_000


def test_par_target_pipeline(long_market):
    y = long_market.yields.resample("QE").last().iloc[-40:]
    cfg = PipelineConfig(
        source="synthetic",
        freq="QE",
        calibration=CalibrationConfig(target="par"),
        run_forecasts=False,
    )
    r = run_pipeline(cfg, data=(y, None, None))
    assert r.recession_model is None and r.forecast_eval is None
    assert np.isfinite(r.summary["fit_quality_bp"]["rmse_median"])


def test_gsw_reference_for_fred_source(monkeypatch, small_market):
    from nss_engine import pipeline
    from nss_engine.data import DataError

    cfg = PipelineConfig(source="fred", run_forecasts=False)
    monkeypatch.setattr(pipeline, "load_gsw_parameters", lambda *a, **k: small_market.true_params)
    params, name = pipeline._reference_params(cfg, None)
    assert "GSW" in name and params is small_market.true_params

    def offline(*a, **k):
        raise DataError("offline")

    monkeypatch.setattr(pipeline, "load_gsw_parameters", offline)
    assert pipeline._reference_params(cfg, None) == (None, None)
    r = run_pipeline(cfg, data=(small_market.yields.iloc[:30], None, None))
    assert r.reference is None and "reference_curve" not in r.summary


def _offline(*args, **kwargs):
    from nss_engine.data import DataError

    raise DataError("offline")


def test_term_premium_benchmarks_for_fred_source(monkeypatch, long_market, tmp_path):
    from nss_engine import pipeline
    from nss_engine.data import parse_spf_bill_forecasts
    from nss_engine.report import render_markdown

    monthly = long_market.yields.resample("ME").last()
    truth = long_market.true_params
    kw = pd.Series(np.linspace(2.0, 0.0, len(truth)), index=truth.index, name="kim_wright_tp10")
    monkeypatch.setattr(pipeline, "load_gsw_parameters", lambda *a, **k: truth)
    monkeypatch.setattr(pipeline, "load_kim_wright_term_premium", lambda *a, **k: kw)
    # survey forecasts: next quarter's bill rate = today's 3M yield, ten-year = today's 10Y
    q = monthly[monthly.index.month.isin([2, 5, 8, 11])]
    wide = pd.DataFrame(
        {
            "YEAR": q.index.year,
            "QUARTER": (q.index.month + 1) // 3,
            "TBILL3": q[0.25].to_numpy(),
            "BILL10": np.where(q.index.month == 2, q[10.0].to_numpy(), np.nan),
        }
    )
    surveys = parse_spf_bill_forecasts(wide)
    monkeypatch.setattr(pipeline, "load_spf_bill_forecasts", lambda *a, **k: surveys)
    cfg = PipelineConfig(source="fred", freq="ME", run_forecasts=False)
    r = run_pipeline(cfg, data=(monthly, None, None))
    assert set(r.term_premium_benchmarks) == {
        "Kim-Wright (Fed Board)",
        "ACM on the Fed's GSW curve",
    }
    cmp = r.term_premium_comparison
    assert len(cmp) == 6  # (full sample, survey-anchored, real time) x (two benchmarks)
    assert r.acm_survey is not None and r.acm_survey.p_dynamics == "survey"
    sv = r.summary["term_premium"]["survey_anchored"]
    assert sv["n_surveys"] == len(surveys)
    pd.testing.assert_frame_equal(r.acm_survey.fitted, r.acm.fitted)
    report = render_markdown(r)
    assert "Anchored to surveys" in report
    paths = write_outputs(r, tmp_path, dashboard=False)
    tp_csv = pd.read_csv(paths["term_premium"], index_col=0)
    assert {"term_premium_survey", "expected_short_rate_survey"} <= set(tp_csv.columns)
    # without surveys (download failed, or switched off) the rest still runs
    monkeypatch.setattr(pipeline, "load_spf_bill_forecasts", _offline)
    assert run_pipeline(cfg, data=(monthly, None, None)).acm_survey is None
    # the engine's curves are close to the "Fed" curve here, so the premia agree
    row = cmp.loc[("ACM on NSS curves (full sample)", "ACM on the Fed's GSW curve")]
    assert row["corr_level"] > 0.9
    assert r.term_premium_recession is None  # no recession data
    assert "benchmarks" in r.summary["term_premium"]


def test_csv_source(tmp_path, small_market):
    from nss_engine.data import label_columns

    path = tmp_path / "yields.csv"
    label_columns(small_market.yields.iloc[:30]).to_csv(path)
    r = run_pipeline(PipelineConfig(source=str(path), start=None, run_forecasts=False))
    assert len(r.fit.params) == 30


def test_unknown_source():
    from nss_engine.data import DataError

    with pytest.raises(DataError):
        run_pipeline(PipelineConfig(source="nowhere.csv"))


class TestCLI:
    def test_run_synthetic(self, tmp_path, capsys):
        code = cli.main(
            [
                "run",
                "--source",
                "synthetic",
                "--freq",
                "ME",
                "--years",
                "20",
                "--out",
                str(tmp_path),
                "--no-dashboard",
                "--print-report",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "median fit RMSE" in out and "# NSS Yield Curve Report" in out
        assert (tmp_path / "summary.json").exists() and not (tmp_path / "dashboard.html").exists()

    def test_curve(self, capsys):
        assert cli.main(["curve", "--source", "synthetic", "--date", "1995-06-30"]) == 0
        out = capsys.readouterr().out
        assert "NSS curve for 1995-06-30" in out and "10Y" in out

    def test_curve_ns_model(self, capsys):
        assert cli.main(["curve", "--source", "synthetic", "--model", "ns"]) == 0
        assert "NS curve" in capsys.readouterr().out

    def test_data_error_exit_code(self, capsys):
        assert cli.main(["curve", "--source", "synthetic", "--date", "1900-01-01"]) == 2
        assert "error:" in capsys.readouterr().err

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["--version"])
        assert "nss-engine" in capsys.readouterr().out


def test_text_files_are_written_as_utf8_on_any_platform():
    # Windows defaults to cp1252, which cannot encode the report's Greek letters
    # and minus signs: every text read or write must name its encoding.
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "nss_engine"
    offenders = []
    for path in src.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"\.(write_text|read_text)\(|\bopen\(", text):
            depth, j = 1, m.end()
            while depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            call = text[m.start() : j]
            if "encoding=" not in call and "urlopen" not in text[m.start() - 7 : m.end()]:
                offenders.append(f"{path.name}: {call[:60]}")
    assert not offenders, offenders
