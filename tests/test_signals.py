import numpy as np
import pandas as pd
import pytest

from nsscurve.signals import (butterfly_signals, half_life, residual_zscores,
                              rich_cheap_table, signal_backtest)


def _ar1(phi, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return pd.Series(x, index=pd.date_range("2000-01-07", periods=n, freq="W-FRI"))


def test_zscore_has_no_lookahead():
    s = _ar1(0.8).to_frame("5Y")
    z1 = residual_zscores(s, 26)
    s2 = s.copy()
    s2.iloc[-50:] += 100.0                          # change the future
    z2 = residual_zscores(s2, 26)
    pd.testing.assert_frame_equal(z1.iloc[:-50], z2.iloc[:-50])


def test_half_life_matches_ar_coefficient():
    phi = 0.8
    assert half_life(_ar1(phi)) == pytest.approx(-np.log(2) / np.log(phi), rel=0.15)
    assert half_life(pd.Series(np.arange(100.0))) == np.inf
    assert np.isnan(half_life(pd.Series([1.0, 2.0])))


def test_rich_cheap_signs():
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"2Y": rng.normal(0, 1, 60), "10Y": rng.normal(0, 1, 60),
                       "30Y": rng.normal(0, 1, 60)}, index=idx)
    df.iloc[-1] = [8.0, -8.0, 0.0]
    t = rich_cheap_table(df, window=26, threshold=2.0)
    assert t.loc["2Y", "Signal"].startswith("CHEAP")
    assert t.loc["10Y", "Signal"].startswith("RICH")
    assert t.loc["30Y", "Signal"] == "FAIR"


def test_signal_backtest_detects_mean_reversion():
    s = _ar1(0.5).to_frame("7Y")
    q = signal_backtest(s, window=52, threshold=1.5, horizon=4)
    assert q.loc["7Y", "Signals"] > 20
    assert q.loc["7Y", "HitRate"] > 0.7
    assert q.loc["7Y", "AvgReversion_bp"] > 0


def test_butterfly_signals():
    idx = pd.date_range("2024-01-05", periods=40, freq="W-FRI")
    mkt = pd.DataFrame({"2Y": 4.0, "5Y": 4.1, "10Y": 4.3}, index=idx)
    mdl = mkt.copy()
    mdl["5Y"] = 4.05
    f = butterfly_signals(mkt, mdl, window=10)
    assert f["MarketFly_bp"].iloc[0] == pytest.approx(-10.0)
    assert f["Deviation_bp"].iloc[0] == pytest.approx(10.0)
    assert butterfly_signals(mkt[["2Y"]], mdl[["2Y"]]).empty
