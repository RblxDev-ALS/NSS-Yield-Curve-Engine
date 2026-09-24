import numpy as np
import pytest
from scipy.optimize import approx_fprime

from nsscurve.model import (NSSCurve, NSSParams, ParYieldPricer, cont_to_periodic,
                            hump_loading, loading_matrix, periodic_to_cont, slope_loading)


def test_loading_limits():
    assert slope_loading(0.0) == pytest.approx(1.0)
    assert hump_loading(0.0) == pytest.approx(0.0)
    assert slope_loading(1e-10) == pytest.approx(1.0)
    assert slope_loading(200.0) == pytest.approx(1 / 200.0)
    # hump peaks at x ~= 1.7933
    x = np.linspace(0.01, 10, 100001)
    assert x[np.argmax(hump_loading(x))] == pytest.approx(1.7933, abs=1e-3)


def test_loading_matrix_broadcasts_over_grid(mats):
    L = loading_matrix(mats, np.array([0.5, 1.0, 2.0]), np.array([0.05, 0.1, 0.2]))
    assert L.shape == (3, len(mats), 4)
    np.testing.assert_allclose(L[1], loading_matrix(mats, 1.0, 0.1))
    assert loading_matrix(mats, 1.0).shape == (len(mats), 3)


def test_short_and_long_rate_limits(true_params):
    c = NSSCurve(true_params)
    assert c.zero(1e-9) == pytest.approx(true_params.beta0 + true_params.beta1, abs=1e-6)
    assert c.zero(1e5) == pytest.approx(true_params.beta0, abs=1e-2)
    assert c.forward(0.0) == pytest.approx(c.short_rate)
    assert c.short_rate == pytest.approx(2.0)
    assert c.hump_locations == pytest.approx((1.7933 / 0.6, 1.7933 / 0.08), rel=1e-4)


def test_forward_is_derivative_of_tau_times_zero(true_params):
    c = NSSCurve(true_params)
    t = np.linspace(0.1, 30, 50)
    h = 1e-5
    numeric = ((t + h) * c.zero(t + h) - (t - h) * c.zero(t - h)) / (2 * h)
    np.testing.assert_allclose(c.forward(t), numeric, atol=1e-6)


def test_scalar_and_vector_inputs(true_params):
    c = NSSCurve(true_params)
    assert isinstance(c.zero(5.0), float)
    assert isinstance(c.par_yield(5.0), float)
    assert c.zero(np.array([5.0])).shape == (1,)


def test_compounding_round_trip():
    r = np.array([-0.5, 0.0, 1.0, 5.0, 12.0])
    np.testing.assert_allclose(periodic_to_cont(cont_to_periodic(r)), r, atol=1e-12)
    # 5% semi-annual = 2*ln(1.025) continuous
    assert periodic_to_cont(5.0) == pytest.approx(200 * np.log(1.025))


def test_par_yield_of_flat_curve_equals_flat_rate():
    flat_cc = 4.0
    c = NSSCurve(NSSParams(flat_cc, 0.0, 0.0, 0.0, 1.0, 0.1))
    expected = cont_to_periodic(flat_cc, 2)
    np.testing.assert_allclose(c.par_yield([0.5, 1, 2, 7, 30]), expected, atol=1e-10)
    # With a stub period, linear accrued interest vs. compounded discounting
    # moves the clean par yield by a fraction of a basis point.
    np.testing.assert_allclose(c.par_yield([2.3, 7.75]), expected, atol=5e-3)


def test_par_bond_prices_at_par(true_params):
    c = NSSCurve(true_params)
    for T in (2, 5, 10, 30):
        assert c.bond_price(c.par_yield(T), T) == pytest.approx(100.0, abs=1e-9)


def test_key_rate_durations_sum_to_effective_duration(true_params):
    c = NSSCurve(true_params)
    coupon, T = 4.0, 10.0
    krd = c.key_rate_durations(coupon, T)
    h = 1e-4
    up = c.bond_price(coupon, T, zero_shift=lambda t: np.full_like(t, h * 100))
    dn = c.bond_price(coupon, T, zero_shift=lambda t: np.full_like(t, -h * 100))
    eff = (dn - up) / (2 * c.bond_price(coupon, T) * h)
    assert sum(krd.values()) == pytest.approx(eff, rel=1e-4)
    assert max(krd, key=krd.get) == 10.0
    assert krd[30.0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [[4.0, -2.0, -1.5, 1.2, 0.6, 0.08],
                                   [2.0, 1.0, 3.0, -2.0, 2.5, 0.2]])
def test_analytic_jacobians_match_finite_differences(mats, theta):
    theta = np.array(theta)
    m = np.append(mats, 2.3)                      # includes a stub-period maturity
    c = NSSCurve(NSSParams.from_vector(theta))
    pricer = ParYieldPricer(m)

    def fd(func):
        return np.array([approx_fprime(theta, lambda t, i=i: func(t)[i], 1e-7)
                         for i in range(len(m))])

    zj = fd(lambda t: NSSCurve(NSSParams.from_vector(t)).zero(m))
    pj = fd(lambda t: pricer(NSSCurve(NSSParams.from_vector(t))))
    np.testing.assert_allclose(c.zero_jacobian(m), zj, atol=5e-5)
    np.testing.assert_allclose(pricer.jacobian(c), pj, atol=5e-5)


def test_params_round_trips(true_params):
    assert NSSParams.from_vector(true_params.to_vector()) == true_params
    assert NSSParams.from_dict(true_params.to_dict()) == true_params
    ns = NSSParams(4.0, -2.0, 1.0, lambda1=0.7)
    assert not ns.is_svensson
    assert NSSParams.from_vector(ns.to_vector(), svensson=False) == ns
    assert NSSParams.from_dict(ns.to_dict()) == ns
    assert np.isnan(ns.to_dict()["Lambda2"])
