import numpy as np
import pytest

from nsscurve.calibration import CalibrationConfig, NSSCalibrator
from nsscurve.model import NSSCurve, NSSParams, cont_to_periodic

EXACT = dict(ridge=0.0, smoothness=0.0, lambda_smoothness=0.0)


@pytest.mark.parametrize("target", ["zero", "par"])
def test_exact_recovery_noiseless(mats, true_params, target):
    c = NSSCurve(true_params)
    quotes = c.par_yield(mats) if target == "par" else cont_to_periodic(c.zero(mats))
    res = NSSCalibrator(CalibrationConfig(fit_target=target, **EXACT)).fit(mats, quotes)
    assert res.success
    assert res.rmse_bp < 1e-6
    np.testing.assert_allclose(res.params.to_vector(), true_params.to_vector(), atol=1e-6)


def test_ns_model_recovery(mats):
    truth = NSSParams(5.0, -1.0, 2.0, lambda1=0.9)
    quotes = cont_to_periodic(NSSCurve(truth).zero(mats))
    res = NSSCalibrator(CalibrationConfig(model="ns", fit_target="zero", **EXACT)).fit(mats, quotes)
    assert not res.params.is_svensson
    np.testing.assert_allclose(res.params.to_vector(), truth.to_vector(), atol=1e-6)


def test_zero_approximation_is_biased_on_par_quotes(mats, true_params):
    """Treating par yields as zeros (the original approach) mis-states long zeros."""
    quotes = NSSCurve(true_params).par_yield(mats)
    zero_fit = NSSCalibrator(CalibrationConfig(fit_target="zero", **EXACT)).fit(mats, quotes)
    par_fit = NSSCalibrator(CalibrationConfig(fit_target="par", **EXACT)).fit(mats, quotes)
    true_30 = NSSCurve(true_params).zero(30.0)
    assert abs(par_fit.curve.zero(30.0) - true_30) * 100 < 1e-4
    assert abs(zero_fit.curve.zero(30.0) - true_30) * 100 > 5.0     # > 5bp bias


def test_missing_tenors_are_ignored(mats, true_params):
    quotes = NSSCurve(true_params).par_yield(mats)
    quotes[[0, 9]] = np.nan                     # e.g. no 1M and no 20Y
    res = NSSCalibrator(CalibrationConfig(**EXACT)).fit(mats, quotes)
    assert res.success and res.rmse_bp < 1e-4
    assert np.isnan(res.fitted[[0, 9]]).all()
    assert np.isnan(res.residuals_bp[0])


def test_too_few_tenors_fails_gracefully(mats):
    quotes = np.full(len(mats), np.nan)
    quotes[:3] = [1.0, 1.1, 1.2]
    res = NSSCalibrator().fit(mats, quotes)
    assert res.params is None and not res.success
    assert "valid tenors" in res.message
    with pytest.raises(ValueError):
        _ = res.curve


def test_parameters_stay_within_bounds(mats):
    rng = np.random.default_rng(0)
    cfg = CalibrationConfig()
    lo, hi = cfg.bounds()
    cal = NSSCalibrator(cfg)
    for _ in range(10):
        quotes = 3 + np.cumsum(rng.normal(0, 0.3, len(mats)))
        x = cal.fit(mats, quotes).params.to_vector()
        assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
        assert x[4] > x[5]                         # humps identified: lambda1 > lambda2


def test_weights_shift_fit_toward_heavy_tenors(mats, true_params):
    quotes = NSSCurve(true_params).par_yield(mats)
    quotes[8] += 0.10                             # 10bp kink at 10Y
    base = NSSCalibrator(CalibrationConfig(**EXACT)).fit(mats, quotes)
    heavy = NSSCalibrator(CalibrationConfig(weights={10: 50.0}, **EXACT)).fit(mats, quotes)
    assert abs(heavy.residuals_bp[8]) < abs(base.residuals_bp[8])


def test_config_validation():
    with pytest.raises(ValueError):
        CalibrationConfig(model="foo")
    with pytest.raises(ValueError):
        CalibrationConfig(fit_target="spot")
    with pytest.raises(ValueError):
        CalibrationConfig(ridge=-1)
    with pytest.raises(ValueError):
        CalibrationConfig(lambda1_bounds=(0.1, 5), lambda2_bounds=(0.05, 0.5))
    assert NSSCalibrator(ridge=0.0).config.ridge == 0.0


def test_history_fit_quality_and_stability(weekly_panel):
    rates, _ = weekly_panel
    hist = NSSCalibrator().fit_history(rates)
    s = hist.summary()
    assert s["calibrated"] == len(rates)
    assert s["mean_rmse_bp"] < 4.0
    # Temporal priors keep week-to-week parameter changes small.
    assert s["median_abs_param_change"]["Beta0"] < 0.25
    assert s["median_abs_param_change"]["Lambda1"] < 0.1
    assert hist.residuals_bp.shape == rates.shape
    assert list(hist.params.columns) == ["Beta0", "Beta1", "Beta2", "Beta3", "Lambda1", "Lambda2"]


def test_history_recovers_true_zero_curve(weekly_panel):
    rates, factors = weekly_panel
    hist = NSSCalibrator().fit_history(rates)
    grid = np.linspace(0.5, 30, 60)
    est = hist.model_rates(grid, "zero").to_numpy()
    true = np.vstack([NSSCurve(NSSParams(*row)).zero(grid) for row in factors.to_numpy()])
    assert np.sqrt(np.mean((est - true) ** 2)) * 100 < 4.0      # bp


def test_smoothness_reduces_parameter_churn(weekly_panel):
    rates, _ = weekly_panel
    rough = NSSCalibrator(ridge=0.0, smoothness=0.0, lambda_smoothness=0.0).fit_history(rates)
    smooth = NSSCalibrator().fit_history(rates)
    churn = lambda h: h.params.diff().abs().median()[["Beta0", "Beta1", "Beta2", "Beta3"]].mean()
    assert churn(smooth) < 0.5 * churn(rough)


def test_curve_history_accessors(weekly_panel):
    rates, _ = weekly_panel
    hist = NSSCalibrator(fit_target="zero").fit_history(rates.iloc[:20])
    assert hist.curve().params == hist.params_at()
    mid = rates.index[10]
    assert hist.curve(mid).params == NSSParams.from_dict(hist.params.loc[mid])
    for kind in ("zero", "forward", "par", "market"):
        assert hist.model_rates([2, 10], kind).shape == (20, 2)
    with pytest.raises(ValueError):
        hist.model_rates([2], "bogus")
