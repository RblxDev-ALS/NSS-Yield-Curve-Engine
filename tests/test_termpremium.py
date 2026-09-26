import numpy as np
import pandas as pd
import pytest

from nss_engine.models import NSSCurve
from nss_engine.synthetic import simulate_affine_market
from nss_engine.termpremium import (
    compare_term_premia,
    fit_acm,
    real_time_decomposition,
    zero_panel,
)


@pytest.fixture(scope="module")
def affine():
    return simulate_affine_market(periods=600, seed=1)


def test_exact_data_is_priced_exactly(affine):
    res = fit_acm(affine.yields, n_factors=3)
    assert res.fit_rmse_bp.max() < 1e-3
    assert res.sigma_e < 1e-12


def test_risk_neutral_dynamics_are_recovered_exactly(affine):
    # With exact data the excess-return regression pins down the Q dynamics
    # (Φ − λ1) exactly; only the real-world VAR carries sampling error.
    res = fit_acm(affine.yields, n_factors=3)
    q_eig = np.sort(np.abs(np.linalg.eigvals(res.phi - res.lambda1)))
    np.testing.assert_allclose(q_eig, [0.85, 0.93, 0.97], atol=1e-6)


def test_one_month_premium_is_zero(affine):
    res = fit_acm(affine.yields, n_factors=3)
    np.testing.assert_allclose(res.term_premium.iloc[:, 0], 0.0, atol=1e-10)
    # and the one-month fitted yield is the short-rate regression
    np.testing.assert_allclose(res.fitted.iloc[:, 0], affine.yields.iloc[:, 0], atol=1e-8)


def test_term_premium_recovered(affine):
    res = fit_acm(affine.yields, n_factors=3)
    est, true = res.term_premium.iloc[:, -1], affine.term_premium.iloc[:, -1]
    assert est.corr(true) > 0.8
    assert np.sqrt(((est - true) ** 2).mean()) < 0.3  # percent


def test_term_premium_error_shrinks_with_sample_length():
    # The remaining error comes from estimating the real-world VAR, so it is
    # consistent: a much longer sample gets much closer to the truth.
    errs = []
    for periods in (600, 20000):
        mkt = simulate_affine_market(periods=periods, seed=1)
        res = fit_acm(mkt.yields, n_factors=3)
        diff = res.term_premium.iloc[:, -1] - mkt.term_premium.iloc[:, -1]
        errs.append(float(np.sqrt((diff**2).mean())))
    assert errs[1] < 0.6 * errs[0]


def test_noisy_data_and_five_factors():
    mkt = simulate_affine_market(periods=600, seed=2, noise_bp=1.0)
    res = fit_acm(mkt.yields, n_factors=5)
    assert res.fit_rmse_bp.mean() < 1.5
    est, true = res.term_premium.iloc[:, -1], mkt.term_premium.iloc[:, -1]
    assert est.corr(true) > 0.7
    dec = res.decomposition(10)
    np.testing.assert_allclose(
        dec["fitted"], dec["expected_short_rate"] + dec["term_premium"], atol=1e-12
    )


def test_zero_panel_matches_curves():
    idx = pd.to_datetime(["2020-01-15", "2020-01-31", "2020-02-14"])
    curves = [
        NSSCurve(4.5, -1.5, -2.0, 1.0, 0.9, 0.15),
        NSSCurve(4.4, -1.4, -2.0, 1.0, 0.9, 0.15),
        NSSCurve(4.3, -1.3, -1.0, 0.5, 0.8, 0.2),
    ]
    params = pd.DataFrame(
        [c.as_array() for c in curves],
        index=idx,
        columns=["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"],
    )
    panel = zero_panel(params, max_months=24)
    assert list(panel.index) == list(pd.to_datetime(["2020-01-31", "2020-02-29"]))
    np.testing.assert_allclose(panel.iloc[0], curves[1].zero(np.arange(1, 25) / 12))
    np.testing.assert_allclose(panel.iloc[1], curves[2].zero(np.arange(1, 25) / 12))


def test_input_validation(affine):
    with pytest.raises(ValueError, match="every monthly maturity"):
        fit_acm(affine.yields.drop(columns=affine.yields.columns[5]))
    bad = affine.yields.copy()
    bad.iloc[3, 4] = np.nan
    with pytest.raises(ValueError, match="missing"):
        fit_acm(bad)
    with pytest.raises(ValueError, match="more than"):
        fit_acm(affine.yields.iloc[:20])


def test_real_time_decomposition_has_no_look_ahead(affine):
    y = affine.yields.iloc[:200]
    base = real_time_decomposition(y, min_train=150, n_factors=3)
    assert len(base) == 51
    scrambled = y.copy()
    scrambled.iloc[180:] += 1.0  # change the future
    alt = real_time_decomposition(scrambled, min_train=150, n_factors=3)
    pd.testing.assert_frame_equal(base.iloc[:30], alt.iloc[:30])
    assert not np.allclose(base["term_premium"].iloc[-1], alt["term_premium"].iloc[-1])


def test_compare_term_premia():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    ref_daily_idx = pd.date_range("2000-01-01", idx[-1], freq="B")
    x = pd.Series(np.sin(np.arange(60) / 6.0), index=idx)
    ref = x.reindex(ref_daily_idx, method="bfill")
    out = compare_term_premia(x + 0.1, ref)
    assert out["corr_level"] > 0.99
    assert out["mean_gap_bp"] == pytest.approx(10.0, abs=1.0)
    assert out["n_months"] == 60
    with pytest.raises(ValueError):
        compare_term_premia(x.iloc[:10], ref)


def test_stationarity_cap_keeps_fitted_yields(affine):
    free = fit_acm(affine.yields, n_factors=3, max_eigenvalue=None)
    capped = fit_acm(affine.yields, n_factors=3, max_eigenvalue=0.9)
    assert free.max_eigenvalue > 0.9
    assert capped.max_eigenvalue == pytest.approx(0.9)
    # the cross-section (risk-neutral dynamics) is untouched ...
    pd.testing.assert_frame_equal(free.fitted, capped.fitted, atol=1e-9, rtol=0)
    np.testing.assert_allclose(free.phi - free.lambda1, capped.phi - capped.lambda1, atol=1e-12)
    # ... only the split into expectations and premium moves
    assert not np.allclose(free.term_premium.iloc[:, -1], capped.term_premium.iloc[:, -1])
    # a cap above the estimated persistence changes nothing
    loose = fit_acm(affine.yields, n_factors=3, max_eigenvalue=0.9999)
    pd.testing.assert_frame_equal(free.term_premium, loose.term_premium)


def test_real_time_discards_failed_estimates(affine):
    rt = real_time_decomposition(affine.yields.iloc[:80], min_train=70, max_fit_error_bp=-1.0)
    assert rt.isna().all().all()
