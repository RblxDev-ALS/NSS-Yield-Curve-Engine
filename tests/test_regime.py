import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from nss_engine import regime
from nss_engine.models import PARAM_NAMES


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


class TestNearTermForwardSpread:
    def test_matches_forward_rate(self, humped_curve, inverted_curve):
        params = pd.DataFrame(
            [humped_curve.as_array(), inverted_curve.as_array()],
            index=pd.to_datetime(["2020-01-31", "2023-06-30"]),
            columns=list(PARAM_NAMES),
        )
        ntfs = regime.near_term_forward_spread(params)
        for curve, value in zip((humped_curve, inverted_curve), ntfs, strict=True):
            expected = curve.forward_rate(1.5, 1.75)[0] - curve.zero(0.25)[0]
            assert value == pytest.approx(expected)
        assert ntfs.iloc[0] > 0 > ntfs.iloc[1]  # hikes priced vs cuts priced

    def test_flat_curve_is_zero(self):
        flat = pd.DataFrame(
            [[4.0, 0.0, 0.0, 0.0, 0.7, 0.2]],
            index=[pd.Timestamp("2024-01-31")],
            columns=list(PARAM_NAMES),
        )
        assert regime.near_term_forward_spread(flat).iloc[0] == pytest.approx(0.0, abs=1e-12)


class TestRealTimeRecessionModel:
    @pytest.fixture
    def data(self, long_market):
        mats = np.asarray(long_market.yields.columns, dtype=float)
        y = long_market.yields
        spread = y[mats[np.argmin(abs(mats - 10))]] - y[mats[np.argmin(abs(mats - 0.25))]]
        return spread, long_market.recession

    def test_no_look_ahead(self, data):
        spread, rec = data
        base = regime.real_time_evaluation(spread, rec, horizon=12, min_train_months=60)
        origin = base.probabilities.index[len(base.probabilities) // 2]
        # Scramble every outcome that was not yet known at `origin`.
        scrambled = rec.copy()
        unknown = scrambled.index > origin
        scrambled[unknown] = 1 - scrambled[unknown]
        alt = regime.real_time_evaluation(spread, scrambled, horizon=12, min_train_months=60)
        assert alt.probabilities[origin] == pytest.approx(base.probabilities[origin])

    def test_publication_lag_uses_less_data(self, data):
        spread, rec = data
        fast = regime.real_time_evaluation(spread, rec, min_train_months=60)
        slow = regime.real_time_evaluation(spread, rec, min_train_months=60, publication_lag=12)
        assert slow.probabilities.index[0] > fast.probabilities.index[0]

    def test_scores(self, data):
        spread, rec = data
        ev = regime.real_time_evaluation(spread, rec, min_train_months=60)
        p, y = ev.probabilities.to_numpy(), ev.outcomes.to_numpy()
        assert np.all((p > 0) & (p < 1))  # the ridge prior prevents 0/1 forecasts
        assert ev.brier == pytest.approx(np.mean((p - y) ** 2))
        assert ev.log_score == pytest.approx(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        assert ev.auc == pytest.approx(regime.roc_auc(p, y))

    def test_ridge_prior_tames_separation(self):
        x = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
        y = np.array([1, 1, 1, 0, 0, 0])  # perfectly separable
        loose = regime.fit_probit(x, y)
        tight = regime.fit_probit(x, y, l2=1.0)
        assert abs(loose.coef[1]) > 3 * abs(tight.coef[1])
        assert 0.5 < tight.predict([-1.0])[0] < 0.99

    def test_compare_predictors(self, data):
        spread, rec = data
        noise = pd.Series(
            np.random.default_rng(0).normal(size=len(spread)), index=spread.index, name="noise"
        )
        table = regime.compare_recession_predictors(
            {"spread": spread, "noise": noise, "both": pd.concat([spread, noise], axis=1)},
            rec,
            min_train_months=60,
        )
        assert table.loc["spread", "auc_out_of_sample"] > table.loc["noise", "auc_out_of_sample"]
        assert len(set(table["n_forecasts"])) == 1  # scored on common origins
        assert table.loc["both", "pseudo_r2"] >= table.loc["spread", "pseudo_r2"] - 1e-9
        # Gains are measured against the first candidate, with an interval around them.
        assert np.isnan(table.loc["spread", "auc_gain_vs_first"])
        gain = table.loc["noise"]
        assert gain["auc_gain_vs_first"] == pytest.approx(
            gain["auc_out_of_sample"] - table.loc["spread", "auc_out_of_sample"]
        )
        assert gain["auc_gain_lo90"] <= gain["auc_gain_vs_first"] <= gain["auc_gain_hi90"]

    def test_block_bootstrap_auc_difference(self):
        rng = np.random.default_rng(1)
        n = 400
        y = (np.sin(np.arange(n) / 15) > 0.7).astype(int)  # episodes, like recessions
        good = y + rng.normal(0, 0.5, n)
        weak = y + rng.normal(0, 3.0, n)
        d, lo, hi = regime.block_bootstrap_auc_difference(good, weak, y, n_boot=500)
        assert d > 0 and lo > 0 and lo <= d <= hi  # a real gap is detected
        d0, lo0, hi0 = regime.block_bootstrap_auc_difference(good, good, y, n_boot=200)
        assert d0 == 0 and lo0 == 0 and hi0 == 0
        # No recessions at all: no interval rather than a misleading one
        none = np.zeros(n, dtype=int)
        assert np.isnan(regime.block_bootstrap_auc_difference(good, weak, none, n_boot=50)).all()

    def test_multi_predictor_names(self, data):
        spread, rec = data
        X = pd.concat({"a": spread, "b": spread**2}, axis=1)
        m = regime.recession_probability_model(X, rec)
        assert m.model.names == ("const", "a", "b")
        assert m.model.coef.size == 3

    def test_too_little_history(self, data):
        spread, rec = data
        with pytest.raises(ValueError):
            regime.real_time_evaluation(spread, rec, min_train_months=10_000)
