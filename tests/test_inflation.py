import numpy as np
import pandas as pd
import pytest

from nss_engine import inflation
from nss_engine.calibration import calibrate_panel
from nss_engine.models import PARAM_NAMES, NSSCurve
from nss_engine.synthetic import simulate_market, simulate_tips_market


@pytest.fixture(scope="module")
def markets():
    nominal = simulate_market(start="2003-01-03", periods=52 * 6, seed=2)
    return nominal, simulate_tips_market(nominal, seed=2)


def _frame(curves, index):
    return pd.DataFrame([c.as_array() for c in curves], index=index, columns=list(PARAM_NAMES))


def test_breakevens_are_nominal_minus_real_zero_rates():
    idx = pd.date_range("2020-01-03", periods=3, freq="W-FRI")
    nom = NSSCurve(4.5, -2.0, -1.0, 1.0, 0.9, 0.2)
    real = NSSCurve(2.0, -1.5, 0.5, 0.0, 0.4, 0.4)
    be = inflation.breakevens(_frame([nom] * 3, idx), _frame([real] * 3, idx))
    expected = nom.zero([5.0, 10.0]) - real.zero([5.0, 10.0])
    np.testing.assert_allclose(be[["be_5y", "be_10y"]].iloc[0], expected, atol=1e-12)
    # 5y5y: the average forward breakeven from 5 to 10 years
    fwd = (nom.forward_rate(5.0, 10.0) - real.forward_rate(5.0, 10.0))[0]
    assert be["be_5y5y"].iloc[0] == pytest.approx(fwd, abs=1e-10)
    par = nom.par_yield([5.0]) - real.par_yield([5.0])
    assert be["be_par_5y"].iloc[0] == pytest.approx(par[0])
    assert be["real_10y"].iloc[0] == pytest.approx(real.zero([10.0])[0])


def test_breakevens_align_on_real_dates_with_tolerance():
    nom_idx = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    real_idx = nom_idx[1:] + pd.Timedelta(days=1)  # off by a day
    nom = _frame([NSSCurve(4.0, -1.0, 0.0, 0.0, 0.8, 0.2)] * 4, nom_idx)
    real = _frame([NSSCurve(1.5, -0.5, 0.0, 0.0, 0.4, 0.4)] * 3, real_idx)
    be = inflation.breakevens(nom, real)
    assert list(be.index) == list(real_idx)


def test_real_curve_config_is_nelson_siegel_inside_the_data():
    cfg = inflation.real_curve_config()
    assert cfg.model == "ns" and cfg.min_points == 4
    lo, hi = cfg.lambda1_bounds
    assert 1.79 / hi > 2.5 and 1.79 / lo <= 30.0


def test_simulated_tips_market(markets):
    nominal, tips = markets
    assert list(tips.yields.columns) == [5.0, 7.0, 10.0, 20.0, 30.0]
    assert tips.yields.index[0] >= pd.Timestamp("2003-01-01")
    # the 30-year starts in 2010, as on FRED
    assert tips.yields[30.0].isna().all()
    # real + breakeven = nominal, exactly, at every maturity
    i = 10
    tau = np.array([2.0, 5.0, 10.0, 30.0])
    real = NSSCurve.from_array(tips.true_real_params.iloc[i].to_numpy())
    be = NSSCurve.from_array(tips.true_breakeven_params.iloc[i].to_numpy())
    nom = NSSCurve.from_array(nominal.true_params.loc[tips.yields.index[i]].to_numpy())
    np.testing.assert_allclose(real.zero(tau) + be.zero(tau), nom.zero(tau), atol=1e-12)
    truth = tips.true_breakevens()
    assert set(truth.columns) == {"be_5y", "be_10y", "be_5y5y"}
    assert 1.0 < truth["be_10y"].mean() < 4.0
    with pytest.raises(ValueError, match="no dates"):
        simulate_tips_market(nominal.true_params.loc[:"2002-12-31"].iloc[:0])


def test_breakevens_recover_the_truth(markets):
    nominal, tips = markets
    nom_fit = calibrate_panel(nominal.yields)
    real_fit = inflation.fit_real_curve(tips.yields)
    assert (real_fit.diagnostics["model"] == "ns").all()
    assert real_fit.diagnostics["rmse_bp"].median() < 4.0
    be = inflation.breakevens(nom_fit.params, real_fit.params)
    truth = tips.true_breakevens().reindex(be.index)
    rmse = np.sqrt(((be[truth.columns] - truth) ** 2).mean()) * 100
    assert rmse["be_5y"] < 6 and rmse["be_10y"] < 6 and rmse["be_5y5y"] < 12
    # and they beat FRED's par-yield formulas on the same quotes
    fred = inflation.fred_style_breakevens(nominal.yields, tips.yields)
    fred.columns = ["be_5y", "be_10y", "be_5y5y"]
    fred_rmse = np.sqrt(((fred - truth.reindex(fred.index)) ** 2).mean()) * 100
    assert rmse["be_5y"] < fred_rmse["be_5y"]


def test_fred_style_breakevens_formula():
    idx = pd.date_range("2020-01-03", periods=2, freq="W-FRI")
    nom = pd.DataFrame({5.0: [3.0, 3.0], 10.0: [4.0, 4.0]}, index=idx)
    real = pd.DataFrame({5.0: [1.0, 1.0], 10.0: [1.5, 1.5]}, index=idx)
    out = inflation.fred_style_breakevens(nom, real)
    assert out["T5YIE"].iloc[0] == pytest.approx(2.0)
    assert out["T10YIE"].iloc[0] == pytest.approx(2.5)
    assert out["T5YIFR"].iloc[0] == pytest.approx(((1.025**10 / 1.02**5) ** 0.2 - 1) * 100)


def test_compare_series():
    idx = pd.date_range("2010-01-01", periods=900, freq="D")
    ref = pd.Series(np.sin(np.arange(900) / 50.0) + 2.0, index=idx)
    est = ref + 0.1
    stats = inflation.compare_series(est, ref)
    assert stats["corr_level"] == pytest.approx(1.0)
    assert stats["mean_gap_bp"] == pytest.approx(10.0)
    assert stats["rmse_bp"] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="24"):
        inflation.compare_series(est.iloc[:40], ref.iloc[:40])
