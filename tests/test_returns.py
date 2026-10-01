import dataclasses

import numpy as np
import pandas as pd
import pytest

from nss_engine import returns as ret
from nss_engine.synthetic import simulate_affine_market
from nss_engine.termpremium import fit_acm


def _flat_panel(level: float = 4.0, months: int = 120, periods: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=periods, freq="ME")
    cols = np.arange(1, months + 1) / 12.0
    return pd.DataFrame(level, index=idx, columns=cols)


def test_flat_constant_curve_earns_no_excess_return():
    rx = ret.excess_returns(_flat_panel())
    assert list(rx.columns) == [2.0, 5.0, 10.0]
    assert rx.iloc[:-12].abs().to_numpy().max() < 1e-12
    assert rx.iloc[-12:].isna().all().all()  # not yet realized
    fwd = ret.forward_rates(_flat_panel())
    np.testing.assert_allclose(fwd.to_numpy(), 4.0)


def test_excess_return_by_hand():
    zeros = _flat_panel(periods=24)
    zeros.iloc[12, :] = 5.0  # all yields up 1 point a year after the first date
    rx = ret.excess_returns(zeros, maturities=(24,))
    # buy the 2-year at 4%, sell it as a 1-year at 5%, minus the 1-year at 4%
    assert rx.iloc[0, 0] == pytest.approx(2 * 4.0 - 1 * 5.0 - 4.0)
    with pytest.raises(ValueError):
        ret.excess_returns(zeros, maturities=(12,))


def test_true_expected_returns_are_unbiased_forecasts():
    mkt = simulate_affine_market(periods=20000, seed=3)
    true = mkt.expected_excess_returns()
    realized = ret.excess_returns(mkt.yields)
    for col in realized.columns:
        df = pd.concat([realized[col], true[col]], axis=1).dropna()
        slope = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
        assert 0.8 < slope < 1.2
        gap = df.iloc[:, 0] - df.iloc[:, 1]  # forecast errors; overlapping, so ~n/12 independent
        assert abs(gap.mean()) < 3 * gap.std() / np.sqrt(len(df) / 12)


def test_acm_recovers_expected_returns_and_expectations_hypothesis_gives_none():
    mkt = simulate_affine_market(periods=3000, seed=1)
    res = fit_acm(mkt.yields, n_factors=3)
    est = res.expected_excess_returns()
    true = mkt.expected_excess_returns()
    for col in true.columns:
        assert np.corrcoef(est[col], true[col])[0, 1] > 0.98
    assert (est - true).abs().mean().max() < 0.2
    # no prices of risk: the expected excess return is only a convexity term
    k = res.phi.shape[0]
    eh = dataclasses.replace(res, lambda0=np.zeros(k), lambda1=np.zeros((k, k)))
    assert eh.expected_excess_returns().abs().to_numpy().max() < 0.05


def test_regression_forecasts_use_only_completed_returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-31", periods=200, freq="ME")
    x = pd.DataFrame({"x": rng.normal(size=200)}, index=idx)
    y = pd.Series(0.5 * x["x"].to_numpy() + rng.normal(size=200), index=idx)
    base = ret.real_time_regression_forecasts(y, x, horizon=12, min_train=40)
    mean = ret.real_time_regression_forecasts(y, None, horizon=12, min_train=40)
    t = 150
    # the historical mean at t is the mean of returns formed up to t - 12
    assert mean.iloc[t] == pytest.approx(y.iloc[: t - 12 + 1].mean())
    # scramble everything not yet realized at t: the forecast at t is unchanged
    y2 = y.copy()
    y2.iloc[t - 11 :] = rng.normal(size=len(y) - (t - 11)) * 100
    x2 = x.copy()
    x2.iloc[t + 1 :] = 1e6
    again = ret.real_time_regression_forecasts(y2, x2, horizon=12, min_train=40)
    assert again.iloc[t] == pytest.approx(base.iloc[t])
    assert base.iloc[: 40 + 12 - 1].isna().all()


def test_cochrane_piazzesi_forecasts_without_look_ahead():
    mkt = simulate_affine_market(periods=300, seed=2)
    rx = ret.excess_returns(mkt.yields, maturities=(24, 36, 48, 60, 120))
    fwd = ret.forward_rates(mkt.yields)
    base = ret.cochrane_piazzesi_forecasts(rx, fwd, min_train=60)
    t = 200
    rx2 = rx.copy()
    rx2.iloc[t - 11 :] = 50.0
    again = ret.cochrane_piazzesi_forecasts(rx2, fwd, min_train=60)
    pd.testing.assert_series_equal(base.iloc[t], again.iloc[t])
    assert base.iloc[t].notna().all()


def test_scores_reward_the_true_expectation_and_punish_noise():
    mkt = simulate_affine_market(periods=1200, seed=4)
    realized = ret.excess_returns(mkt.yields)[10.0]
    true = mkt.expected_excess_returns()[10.0]
    bench = ret.real_time_regression_forecasts(realized, None, min_train=60)
    good = ret.evaluate_return_forecasts(realized, true, bench)
    assert good.r2_oos > 0.02 and good.p_value < 0.05
    assert good.n == realized.notna().sum() - 60 - 11
    noise = pd.Series(np.random.default_rng(1).normal(0, 3, len(true)), index=true.index)
    bad = ret.evaluate_return_forecasts(realized, noise, bench)
    assert bad.r2_oos < 0
    same = ret.evaluate_return_forecasts(realized, bench, bench)
    assert same.r2_oos == pytest.approx(0.0)
    assert set(good.as_dict()) >= {"r2_oos", "p_value", "mz_slope"}


def test_real_time_expected_returns_match_a_fit_on_past_data():
    mkt = simulate_affine_market(periods=160, seed=5)
    rt = ret.real_time_expected_returns(mkt.yields, min_train=150, n_factors=3)
    assert len(rt) == 11 and list(rt.columns) == [2.0, 5.0, 10.0, "term_premium"]
    t = 155
    past = fit_acm(mkt.yields.iloc[: t + 1], n_factors=3)
    np.testing.assert_allclose(
        rt.loc[mkt.yields.index[t], [2.0, 5.0, 10.0]].to_numpy(dtype=float),
        past.expected_excess_returns().iloc[-1].to_numpy(),
    )
    future = mkt.yields.copy()
    future.iloc[t + 1 :] += 3.0
    rt2 = ret.real_time_expected_returns(future, min_train=150, n_factors=3)
    pd.testing.assert_series_equal(rt.loc[mkt.yields.index[t]], rt2.loc[mkt.yields.index[t]])


def test_gaps_in_the_monthly_panel_are_refused():
    zeros = _flat_panel().drop(index=pd.Timestamp("2001-06-30"))
    with pytest.raises(ValueError, match="no gaps"):
        ret.excess_returns(zeros)
    with pytest.raises(ValueError):
        ret.forward_spot_spread(_flat_panel(months=60), 120)
