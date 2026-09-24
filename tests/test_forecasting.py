import numpy as np
import pytest

from nsscurve.data import resample_curve, simulate_curve
from nsscurve.forecasting import DieboldLiForecaster


@pytest.fixture(scope="module")
def zero_quoted():
    d = simulate_curve(start="2015-01-01", periods=1500, seed=5, quote_type="zero",
                       noise_bp=0.5, pricing_error_bp=0.0)
    return resample_curve(d, "W")


def test_lambda_estimation_recovers_truth(zero_quoted):
    lam = DieboldLiForecaster().estimate_lambdas(zero_quoted)
    assert lam[0] == pytest.approx(0.6, rel=0.15)
    assert lam[1] == pytest.approx(0.08, rel=0.3)


def test_fit_and_forecast_shapes(zero_quoted):
    dl = DieboldLiForecaster(dynamics="var").fit(zero_quoted)
    f = dl.forecast(6)
    assert f.shape == (6, zero_quoted.shape[1])
    assert list(f.columns) == list(zero_quoted.columns)
    assert dl.factor_forecast(3).shape == (3, 4)
    assert 0.9 < dl.spectral_radius() < 1.05
    assert np.isfinite(f.to_numpy()).all()
    p = DieboldLiForecaster(dynamics="ar").fit(zero_quoted).persistence()
    # Level/slope/curvature are persistent; the long-hump beta3 is noisier under fixed lambdas.
    assert (p[["Beta0", "Beta1", "Beta2"]] > 0.8).all() and (p < 1.05).all()


def test_ns_variant_and_fixed_lambdas(zero_quoted):
    dl = DieboldLiForecaster(model="ns", lambdas=(0.7,)).fit(zero_quoted)
    assert dl.lambdas_ == (0.7,)
    assert dl.betas_.shape[1] == 3


def test_backtest_is_out_of_sample_and_reports_ratio(zero_quoted):
    bt = DieboldLiForecaster().backtest(zero_quoted, horizon=4, min_train=150, step=2)
    t = bt["table"]
    assert bt["n_forecasts"] > 50
    assert set(t.columns) == {"RMSE_Model_bp", "RMSE_RandomWalk_bp", "Ratio"}
    assert (t["Ratio"] > 0.5).all() and (t["Ratio"] < 2.0).all()
    assert bt["errors_model"].index[0] >= zero_quoted.index[148]


def test_validation_errors(zero_quoted):
    with pytest.raises(ValueError):
        DieboldLiForecaster(dynamics="garch")
    with pytest.raises(RuntimeError):
        DieboldLiForecaster().forecast()
    with pytest.raises(ValueError):
        DieboldLiForecaster().backtest(zero_quoted.iloc[:45], min_train=40)
