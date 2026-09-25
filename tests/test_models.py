import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nss_engine.models import (
    DIEBOLD_LI_LAMBDA,
    NSSCurve,
    curvature_loading,
    curvature_peak,
    ns_loadings,
    nss_loadings,
    slope_loading,
)


class TestLoadings:
    def test_limits_at_zero(self):
        assert slope_loading(0.0) == pytest.approx(1.0)
        assert curvature_loading(0.0) == pytest.approx(0.0)

    def test_limits_at_infinity(self):
        assert slope_loading(1e6) == pytest.approx(0.0, abs=1e-5)
        assert curvature_loading(1e6) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("x", [1e-12, 1e-8, 1e-5, 9.9e-5, 1.01e-4, 1e-3])
    def test_series_branch_matches_exact_formula(self, x):
        # Compare with a high-precision evaluation via the series expansion.
        import math

        exact_slope = sum((-x) ** k / math.factorial(k + 1) for k in range(12))
        exact_curv = exact_slope - math.exp(-x)
        assert slope_loading(x) == pytest.approx(exact_slope, rel=1e-12)
        assert curvature_loading(x) == pytest.approx(exact_curv, rel=1e-8, abs=1e-15)

    def test_loadings_are_continuous_across_series_cutoff(self):
        x = np.array([1e-4 * (1 - 1e-9), 1e-4 * (1 + 1e-9)])
        assert abs(np.diff(slope_loading(x))[0]) < 1e-12
        assert abs(np.diff(curvature_loading(x))[0]) < 1e-12

    def test_shapes(self):
        tau = np.linspace(0, 30, 7)
        assert ns_loadings(tau, 0.7).shape == (7, 3)
        assert nss_loadings(tau, 0.7, 0.2).shape == (7, 4)
        assert nss_loadings(5.0, 0.7, 0.2).shape == (1, 4)

    def test_curvature_peak(self):
        lam = 0.6
        grid = np.linspace(0.01, 40, 400_001)
        numeric_peak = grid[np.argmax(curvature_loading(lam * grid))]
        assert curvature_peak(lam) == pytest.approx(numeric_peak, abs=1e-3)
        # Diebold-Li chose λ so the curvature loading peaks at 30 months.
        assert curvature_peak(DIEBOLD_LI_LAMBDA) == pytest.approx(2.5, abs=0.05)


class TestCurve:
    def test_short_and_long_limits(self, humped_curve):
        c = humped_curve
        assert c.zero(0.0)[0] == pytest.approx(c.beta0 + c.beta1)
        assert c.zero(1e4)[0] == pytest.approx(c.beta0, abs=1e-3)
        assert c.forward(0.0)[0] == pytest.approx(c.short_rate)
        assert c.forward(1e4)[0] == pytest.approx(c.long_rate, abs=1e-3)

    def test_forward_is_derivative_of_tau_times_zero(self, humped_curve):
        tau = np.linspace(0.1, 30, 50)
        h = 1e-6
        numeric = (
            (tau + h) * humped_curve.zero(tau + h) - (tau - h) * humped_curve.zero(tau - h)
        ) / (2 * h)
        np.testing.assert_allclose(humped_curve.forward(tau), numeric, atol=1e-6)

    def test_zero_is_average_of_forward(self, inverted_curve):
        # z(T) = (1/T) ∫_0^T f(s) ds
        from scipy.integrate import quad

        for T in (0.5, 2.0, 10.0, 30.0):
            avg = quad(lambda s: inverted_curve.forward(s)[0], 0, T)[0] / T
            assert inverted_curve.zero(T)[0] == pytest.approx(avg, abs=1e-9)

    def test_discount_factor(self, humped_curve):
        assert humped_curve.discount(0.0)[0] == pytest.approx(1.0)
        d = humped_curve.discount(np.linspace(0.1, 30, 30))
        assert np.all(np.diff(d) < 0), "positive rates imply decreasing discount factors"

    def test_forward_rate_between_dates(self, humped_curve):
        t1, t2 = 2.0, 5.0
        f = humped_curve.forward_rate(t1, t2)[0]
        d1, d2 = humped_curve.discount([t1, t2])
        assert f == pytest.approx(100 * np.log(d1 / d2) / (t2 - t1))
        with pytest.raises(ValueError):
            humped_curve.forward_rate(5.0, 2.0)

    def test_par_yield_of_flat_curve(self):
        r = 4.0  # continuously compounded
        flat = NSSCurve(r, 0.0, 0.0, 0.0, 0.5, 0.1)
        expected = 100 * 2 * (np.exp(r / 100 / 2) - 1)  # semi-annual equivalent
        np.testing.assert_allclose(flat.par_yield([0.5, 1, 2, 5, 10, 30]), expected, rtol=1e-12)

    def test_par_bond_prices_at_par(self, humped_curve):
        for T in (2.0, 7.0, 30.0):
            c = humped_curve.par_yield(T)[0] / 100
            times = T - np.arange(int(T * 2)) / 2
            price = c / 2 * humped_curve.discount(times).sum() + humped_curve.discount(T)[0]
            assert price == pytest.approx(1.0, abs=1e-12)

    def test_nelson_siegel_factory(self):
        ns = NSSCurve.nelson_siegel(5, -1, 2, 0.5)
        tau = np.linspace(0, 30, 11)
        np.testing.assert_allclose(ns.zero(tau), ns_loadings(tau, 0.5) @ [5, -1, 2])

    def test_round_trips(self, humped_curve):
        assert NSSCurve.from_array(humped_curve.as_array()) == humped_curve
        assert NSSCurve.from_mapping(humped_curve.as_dict()) == humped_curve

    def test_validation(self):
        with pytest.raises(ValueError):
            NSSCurve(4, -1, 1, 0, -0.5, 0.1)
        with pytest.raises(ValueError):
            NSSCurve.from_array([1, 2, 3])
        with pytest.raises(ValueError):
            NSSCurve(4, -1, 1).evaluate(1.0, "bogus")

    def test_spread(self, inverted_curve):
        s = inverted_curve.spread(10, 2)
        assert s == pytest.approx(inverted_curve.zero(10)[0] - inverted_curve.zero(2)[0])
        assert inverted_curve.spread(10, 0.25) < 0


@settings(max_examples=60, deadline=None)
@given(
    b0=st.floats(1, 8),
    b1=st.floats(-5, 5),
    b2=st.floats(-6, 6),
    b3=st.floats(-6, 6),
    l1=st.floats(0.2, 3),
    l2=st.floats(0.05, 0.19),
)
def test_forward_identity_property(b0, b1, b2, b3, l1, l2):
    """f(τ) = z(τ) + τ z'(τ) for any parameters."""
    c = NSSCurve(b0, b1, b2, b3, l1, l2)
    tau = np.array([0.3, 1.7, 6.0, 18.0])
    h = 1e-5
    dz = (c.zero(tau + h) - c.zero(tau - h)) / (2 * h)
    np.testing.assert_allclose(c.forward(tau), c.zero(tau) + tau * dz, atol=1e-6)


class TestOffGridParYields:
    """Maturities that are not whole coupon periods: stub coupon and accrued interest."""

    def test_coupon_schedule(self):
        from nss_engine.models import coupon_schedule

        times, accrued = coupon_schedule(1.3)
        np.testing.assert_allclose(times, [0.3, 0.8, 1.3])
        assert accrued == pytest.approx(0.4)  # 0.2 of the 0.5-year period has elapsed
        times, accrued = coupon_schedule(10.0)
        assert times.size == 20 and times[0] == pytest.approx(0.5) and accrued == 0.0

    def test_par_curve_is_smooth(self):
        """Dropping the stub coupon / ignoring accrued made the par curve a sawtooth."""
        curve = NSSCurve(4.5, -2.0, -1.5, 2.0, 0.9, 0.15)
        grid = np.linspace(1.01, 30, 3000)
        par = curve.par_yield(grid)
        jumps = np.abs(np.diff(par))
        assert jumps.max() < 0.005  # < 0.5 bp between points 1 day apart

    def test_off_grid_par_bond_has_clean_price_100(self):
        from nss_engine.analytics import Bond, price
        from nss_engine.models import coupon_schedule

        curve = NSSCurve(4.5, -2.0, -1.5, 2.0, 0.9, 0.15)
        for maturity in (1.3, 7.8, 12.25):
            bond = Bond.par(curve, maturity)
            _, accrued = coupon_schedule(maturity)
            clean = price(curve, bond) - accrued * bond.coupon / bond.freq
            assert clean == pytest.approx(100.0, abs=1e-9)

    def test_calibration_operator_agrees_off_grid(self):
        from nss_engine.calibration import _ParOperator

        curve = NSSCurve(4.5, -2.0, -1.5, 2.0, 0.9, 0.15)
        tau = np.array([0.25, 1.3, 2.75, 7.8, 12.25, 30.0])
        op = _ParOperator(tau)
        p = curve.as_array()
        value, jac = op.value_and_jacobian(p)
        np.testing.assert_allclose(value, curve.par_yield(tau), atol=1e-12)
        fd = np.column_stack([(op(p + e) - op(p - e)) / 2e-6 for e in np.eye(6) * 1e-6])
        np.testing.assert_allclose(jac, fd, atol=1e-6 * np.abs(fd).max())
