import numpy as np
import pandas as pd
import pytest

from nss_engine import analytics
from nss_engine.analytics import Bond, carry_rolldown, price, risk_report
from nss_engine.calibration import calibrate_panel
from nss_engine.models import NSSCurve


class TestBonds:
    def test_zero_coupon_price_and_duration(self, humped_curve):
        b = Bond(7.0, 0.0)
        assert price(humped_curve, b) == pytest.approx(100 * humped_curve.discount(7.0)[0])
        rep = risk_report(humped_curve, b)
        # Under continuous compounding a zero's effective duration equals its maturity.
        assert rep.duration == pytest.approx(7.0, rel=1e-6)
        assert rep.convexity == pytest.approx(49.0, rel=1e-4)

    def test_par_bond(self, humped_curve):
        b = Bond.par(humped_curve, 10.0)
        assert price(humped_curve, b) == pytest.approx(100.0, abs=1e-9)

    def test_cashflows(self):
        t, cf = Bond(2.0, 5.0).cashflows()
        np.testing.assert_allclose(t, [0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(cf, [2.5, 2.5, 2.5, 102.5])

    def test_key_rate_durations_sum_to_duration(self, humped_curve):
        rep = risk_report(humped_curve, Bond.par(humped_curve, 12.0))
        assert rep.key_rate_durations.sum() == pytest.approx(rep.duration, rel=1e-6)

    def test_factor_durations(self, humped_curve):
        b = Bond.par(humped_curve, 10.0)
        rep = risk_report(humped_curve, b)
        # β0 shifts every zero rate equally, so its factor duration is the duration.
        assert rep.factor_durations.iloc[0] == pytest.approx(rep.duration, rel=1e-6)
        # And it matches a finite-difference bump of β1.
        h = 1e-4
        up = NSSCurve(*(humped_curve.as_array() + np.array([0, h, 0, 0, 0, 0])))
        dn = NSSCurve(*(humped_curve.as_array() - np.array([0, h, 0, 0, 0, 0])))
        fd = -(price(up, b) - price(dn, b)) / (2 * h) / price(humped_curve, b) * 100
        assert rep.factor_durations.iloc[1] == pytest.approx(fd, rel=1e-6)

    def test_dv01(self, humped_curve):
        rep = risk_report(humped_curve, Bond.par(humped_curve, 5.0))
        assert rep.dv01 == pytest.approx(rep.duration * rep.price / 1e4, rel=1e-6)


class TestCarry:
    def test_identity(self, humped_curve):
        cr = carry_rolldown(humped_curve, [2, 10], horizon=0.25)
        np.testing.assert_allclose(cr["total_bp"], cr["carry_bp"] + cr["rolldown_bp"])
        # Exact log-return of a zero over the horizon with an unchanged curve.
        T, h = 10.0, 0.25
        direct = humped_curve.zero(T)[0] * T - humped_curve.zero(T - h)[0] * (T - h)
        assert cr.loc["10Y", "total_bp"] == pytest.approx(direct * 100)

    def test_flat_curve_has_no_rolldown(self):
        flat = NSSCurve(4.0, 0, 0, 0, 0.5, 0.1)
        cr = carry_rolldown(flat, [5.0], 0.5)
        assert cr["rolldown_bp"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert cr["excess_bp"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_upward_curve_has_positive_rolldown(self, humped_curve):
        assert (carry_rolldown(humped_curve, [2, 5, 10], 0.25)["rolldown_bp"] > 0).all()

    def test_horizon_validation(self, humped_curve):
        with pytest.raises(ValueError):
            carry_rolldown(humped_curve, [0.25], 0.5)


class TestFactors:
    def test_pca_three_factor_structure(self, long_market):
        res = analytics.pca(long_market.yields, changes=False)
        assert res.cumulative.iloc[-1] > 0.99
        assert (res.loadings["PC1"] > 0).all()  # level: everything moves together
        ld = res.loadings["PC2"]
        assert ld.iloc[-1] > ld.iloc[0]  # slope sign convention

    def test_pca_requires_data(self):
        with pytest.raises(ValueError):
            analytics.pca(pd.DataFrame({1.0: [1.0], 2.0: [2.0]}))

    def test_proxy_correlations(self, long_market):
        fit = calibrate_panel(long_market.yields.iloc[::8])
        corr = analytics.factor_proxy_correlations(fit.params, long_market.yields)
        assert corr["-beta1 ~ slope"] > 0.8
        assert corr["beta0 ~ level"] > 0.5


class TestRelativeValue:
    def test_zscore_has_no_lookahead(self):
        idx = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
        rng = np.random.default_rng(0)
        res = pd.DataFrame({2.0: rng.normal(size=120)}, index=idx)
        z1 = analytics.residual_zscores(res, window=52)
        res2 = res.copy()
        res2.iloc[-1] = 100.0  # changing the future must not change the past
        z2 = analytics.residual_zscores(res2, window=52)
        pd.testing.assert_frame_equal(z1.iloc[:-1], z2.iloc[:-1])

    def test_half_life(self):
        rng = np.random.default_rng(1)
        n, phi = 5000, 0.8
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal()
        idx = pd.date_range("2000-01-07", periods=n, freq="W-FRI")
        hl = analytics.residual_half_life(
            pd.DataFrame({5.0: x, 10.0: rng.normal(size=n)}, index=idx)
        )
        assert hl["5Y"] == pytest.approx(np.log(0.5) / np.log(phi), rel=0.1)
        assert hl["10Y"] < 0.5  # white noise reverts immediately

    def test_rich_cheap_signals(self):
        idx = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
        rng = np.random.default_rng(2)
        res = pd.DataFrame(
            {2.0: rng.normal(size=80), 5.0: rng.normal(size=80), 10.0: rng.normal(size=80)},
            index=idx,
        )
        res.iloc[-1] = [10.0, -10.0, 0.0]
        rc = analytics.rich_cheap(res)
        assert list(rc["signal"]) == ["CHEAP", "RICH", "fair"]
