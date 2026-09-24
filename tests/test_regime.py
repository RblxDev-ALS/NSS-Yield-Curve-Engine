import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nss_engine import regime


def _series(values, freq="W-FRI"):
    return pd.Series(
        values, index=pd.date_range("2020-01-03", periods=len(values), freq=freq), dtype=float
    )


class TestClassify:
    def test_plain_bucketing(self):
        s = _series([-0.5, 0.2, 1.0, 2.0])
        out = regime.classify_slope(s, hysteresis=0.0)
        assert list(out.astype(str)) == ["Inverted", "Flat", "Normal", "Steep"]

    def test_hysteresis_prevents_flicker(self):
        s = _series([0.3, -0.05, 0.05, -0.05, 0.05, -0.2, -0.05, 0.05, 0.2])
        no_h = regime.classify_slope(s, hysteresis=0.0).astype(str)
        with_h = regime.classify_slope(s, hysteresis=0.1).astype(str)
        changes = lambda r: int((r != r.shift()).sum() - 1)  # noqa: E731
        assert changes(no_h) == 6
        assert list(with_h) == ["Flat"] * 5 + ["Inverted"] * 3 + ["Flat"]
        assert changes(with_h) == 2

    def test_multi_step_jump(self):
        out = regime.classify_slope(_series([-1.0, 3.0]))
        assert list(out.astype(str)) == ["Inverted", "Steep"]

    def test_nan_keeps_state(self):
        out = regime.classify_slope(_series([1.0, np.nan, 1.0]))
        assert list(out.astype(str)) == ["Normal"] * 3

    def test_validation(self):
        with pytest.raises(ValueError):
            regime.classify_slope(_series([1.0]), thresholds=(1.0, 0.0, 2.0))
        with pytest.raises(ValueError):
            regime.classify_slope(_series([1.0]), labels=("a", "b"))

    def test_spells(self):
        r = regime.classify_slope(_series([-1, -1, 1, 1, 1, -1]), hysteresis=0)
        sp = regime.regime_spells(r)
        assert list(sp["periods"]) == [2, 3, 1]
        assert list(sp["regime"]) == ["Inverted", "Normal", "Inverted"]


def test_curve_dynamics():
    level = _series([4.0, 3.5, 3.5, 4.0, 4.0])
    slope = _series([1.0, 1.5, 1.5, 1.0, 1.02])
    out = regime.curve_dynamics(level, slope, window=1)
    assert out.iloc[0] is np.nan or pd.isna(out.iloc[0])
    assert list(out.iloc[1:]) == ["Bull steepener", "Quiet", "Bear flattener", "Quiet"]
    lvl2, slp2 = _series([4.0, 4.5, 3.9]), _series([1.0, 1.3, 1.1])
    assert list(regime.curve_dynamics(lvl2, slp2, 1).iloc[1:]) == [
        "Bear steepener",
        "Bull flattener",
    ]


def test_inversion_episodes_and_lead_times():
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    spread = pd.Series(1.0, index=idx)
    spread.iloc[5:10] = -0.3  # 5-month inversion
    spread.iloc[7] = -0.8
    spread.iloc[30:31] = -0.1  # 1-month blip, ignored with min_months=3
    rec = pd.Series(0, index=idx)
    rec.iloc[20:28] = 1
    ep = regime.inversion_episodes(spread)
    assert len(ep) == 2 and ep.iloc[0]["periods"] == 5 and ep.iloc[0]["min_spread"] == -0.8
    lt = regime.inversion_lead_times(spread, rec, min_months=3)
    assert len(lt) == 1 and lt.iloc[0]["lead_months"] == 15
    assert regime.recession_starts(rec)[0] == idx[20]


class TestProbit:
    def test_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        x = rng.normal(1.0, 1.2, 20_000)
        a, b = -0.5, -0.7
        y = (rng.uniform(size=x.size) < norm.cdf(a + b * x)).astype(int)
        m = regime.fit_probit(x, y)
        np.testing.assert_allclose(m.coef, [a, b], atol=0.05)
        assert np.all(m.stderr > 0) and m.pseudo_r2 > 0
        assert m.summary().shape == (2, 3)
        assert np.all(np.diff(m.predict([-1.0, 0.0, 2.0])) < 0)

    def test_matches_scipy_mle(self):
        from scipy.optimize import minimize

        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        y = (x + rng.normal(size=400) > 0.3).astype(int)
        m = regime.fit_probit(x, y)
        X = np.column_stack([np.ones_like(x), x])
        nll = lambda b: -norm.logcdf((2 * y - 1) * (X @ b)).sum()  # noqa: E731
        ref = minimize(nll, np.zeros(2), method="BFGS")
        np.testing.assert_allclose(m.coef, ref.x, atol=1e-4)
        assert m.loglik == pytest.approx(-ref.fun, abs=1e-6)

    def test_validation(self):
        with pytest.raises(ValueError):
            regime.fit_probit([1, 2], [0, 2])
        with pytest.raises(ValueError):
            regime.fit_probit([1, 2, 3], [0, 1])


def test_auc():
    assert regime.roc_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert regime.roc_auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert regime.roc_auc([0.5, 0.5], [1, 0]) == 0.5
    assert np.isnan(regime.roc_auc([0.5], [1]))


def test_recession_model_on_synthetic(long_market):
    y = long_market.yields
    spread = y[10.0] - y[0.25]
    m = regime.recession_probability_model(spread, long_market.recession, horizon=12)
    assert m.model.coef[1] < 0, "a lower spread must raise recession odds"
    assert m.auc > 0.8
    assert 0 <= m.latest_probability <= 1
    assert m.target_date > m.latest_date
