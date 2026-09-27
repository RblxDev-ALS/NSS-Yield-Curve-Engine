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
    assert capped.var_capped and not free.var_capped
    # a cap above the estimated persistence changes nothing
    loose = fit_acm(affine.yields, n_factors=3, max_eigenvalue=0.9999)
    pd.testing.assert_frame_equal(free.term_premium, loose.term_premium)


def test_real_time_discards_failed_estimates(affine):
    rt = real_time_decomposition(affine.yields.iloc[:80], min_train=70, max_fit_error_bp=-1.0)
    assert rt[["fitted", "expected_short_rate", "term_premium"]].isna().all().all()
    assert rt["var_capped"].dtype == bool


# ---- real-world dynamics: bias correction and survey anchors -------------------------


def test_pope_bias_matches_kendall_in_one_dimension():
    from nss_engine.termpremium import var_bias

    bias = var_bias(np.array([[0.9]]), np.array([[1.0]]), 100)
    assert bias[0, 0] == pytest.approx(-(1 + 3 * 0.9) / 100)


@pytest.mark.parametrize("method", ["analytic", "bootstrap"])
def test_bias_correction_removes_most_of_the_ols_bias(method):
    # Monte Carlo on a persistent bivariate VAR: OLS is biased towards zero
    # persistence; the corrected estimator is much closer on average.
    from nss_engine.termpremium import bias_corrected_var

    rng = np.random.default_rng(0)
    phi = np.array([[0.95, 0.05], [0.0, 0.8]])
    ols, corrected = [], []
    for _ in range(60):
        X = np.zeros((120, 2))
        for t in range(1, 120):
            X[t] = phi @ X[t - 1] + rng.standard_normal(2)
        Z = np.column_stack([np.ones(119), X[:-1]])
        coef, *_ = np.linalg.lstsq(Z, X[1:], rcond=None)
        ph = coef[1:].T
        ols.append(ph)
        corrected.append(bias_corrected_var(X, ph, method=method, n_boot=100, n_iter=4))
    err_ols = np.abs(np.mean(ols, axis=0) - phi).sum()
    err_bc = np.abs(np.mean(corrected, axis=0) - phi).sum()
    assert err_bc < 0.5 * err_ols
    assert all(np.max(np.abs(np.linalg.eigvals(p))) < 1 for p in corrected)


def test_bias_correction_leaves_explosive_and_bad_input_alone():
    from nss_engine.termpremium import bias_corrected_var

    X = np.cumsum(np.ones((50, 1)), axis=0) ** 1.5
    phi = np.array([[1.02]])
    assert bias_corrected_var(X, phi) is phi
    with pytest.raises(ValueError):
        bias_corrected_var(
            np.random.default_rng(0).standard_normal((50, 1)), np.array([[0.5]]), "x"
        )


def test_real_world_variants_keep_fitted_yields(affine):
    base = fit_acm(affine.yields, n_factors=3)
    surveys = affine.surveys(noise_pp=0.1)
    for kwargs in (
        {"bias_correction": "analytic"},
        {"bias_correction": "bootstrap"},
        {"surveys": surveys},
    ):
        res = fit_acm(affine.yields, n_factors=3, **kwargs)
        pd.testing.assert_frame_equal(base.fitted, res.fitted, atol=1e-9, rtol=0)
        np.testing.assert_allclose(base.phi - base.lambda1, res.phi - res.lambda1, atol=1e-10)
        assert res.p_dynamics == kwargs.get("bias_correction", "survey")
    assert base.p_dynamics == "ols" and base.survey_fit is None and base.survey_rmse.empty
    with pytest.raises(ValueError, match="either"):
        fit_acm(affine.yields, bias_correction="analytic", surveys=surveys)
    with pytest.raises(ValueError, match="bias_correction"):
        fit_acm(affine.yields, bias_correction="kilian")


def test_bias_correction_raises_persistence(affine):
    ols = fit_acm(affine.yields.iloc[:240], n_factors=3)
    bc = fit_acm(affine.yields.iloc[:240], n_factors=3, bias_correction="analytic")
    assert bc.max_eigenvalue > ols.max_eigenvalue


def test_synthetic_surveys_are_true_expectations(affine):
    exact = affine.surveys(noise_pp=0.0)
    assert set(exact["series"]) == {"Q1", "Q2", "Q4", "BILL10"}
    # the ten-year survey is annual, the others quarterly
    assert (exact["series"] == "BILL10").sum() == 50
    assert (exact["series"] == "Q1").sum() == 200
    # a one-month-ahead expectation of the 3-month yield, from the true dynamics
    e1 = affine.expected_bill_rate(1, 1)
    X = affine.factors.to_numpy()
    c0, c1 = affine.bill_loadings
    np.testing.assert_allclose(e1.iloc[:-1], c0 + (X[:-1] @ affine.phi.T + affine.mu) @ c1)
    biased = affine.surveys(noise_pp=0.0, bias_pp=0.5)
    np.testing.assert_allclose(biased["value"] - exact["value"], 0.5)


def test_exact_surveys_pin_down_the_term_premium():
    # A short sample leaves OLS far from the true persistence; unbiased surveys
    # of expected short rates fix most of the error in the premium.
    errs = {"ols": [], "survey": []}
    for seed in range(4):
        mkt = simulate_affine_market(periods=180, seed=seed, level_persistence=0.99)
        true = mkt.term_premium.iloc[:, -1]
        for name, kw in (("ols", {}), ("survey", {"surveys": mkt.surveys(noise_pp=0.1)})):
            tp = fit_acm(mkt.yields, n_factors=3, **kw).term_premium.iloc[:, -1]
            errs[name].append(float(np.sqrt(((tp - true) ** 2).mean())))
    assert np.mean(errs["survey"]) < 0.6 * np.mean(errs["ols"])


def test_survey_fit_and_error_options():
    mkt = simulate_affine_market(periods=240, seed=4, level_persistence=0.99)
    sv = mkt.surveys(noise_pp=0.2, seed=1)
    res = fit_acm(mkt.yields, n_factors=3, surveys=sv)
    fit = res.survey_fit
    assert fit is not None and len(fit) == len(sv)
    assert {"model", "error_sd"} <= set(fit.columns)
    assert (fit["error_sd"] >= 0.1).all()
    # the surveys are fitted to within about their noise (0.2 points)
    assert res.survey_rmse.mean() < 0.3
    fixed = fit_acm(mkt.yields, n_factors=3, surveys=sv, survey_error=0.2)
    assert (fixed.survey_fit["error_sd"] == 0.2).all()
    # surveys outside the sample are ignored; none at all leaves OLS in place
    late = sv.assign(date=sv["date"] + pd.DateOffset(years=100))
    none = fit_acm(mkt.yields, n_factors=3, surveys=late)
    pd.testing.assert_frame_equal(none.term_premium, fit_acm(mkt.yields, n_factors=3).term_premium)
    assert none.survey_fit is not None and none.survey_fit.empty


def test_survey_input_validation(affine):
    sv = affine.surveys()
    with pytest.raises(ValueError, match="columns"):
        fit_acm(affine.yields, surveys=sv.drop(columns="end"))
    with pytest.raises(ValueError, match="start"):
        fit_acm(affine.yields, surveys=sv.assign(start=5, end=2))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        fit_acm(affine.yields.reset_index(drop=True), surveys=sv)


def test_real_time_survey_anchor_has_no_look_ahead():
    mkt = simulate_affine_market(periods=160, seed=5, level_persistence=0.99)
    sv = mkt.surveys(noise_pp=0.1)
    base = real_time_decomposition(mkt.yields, min_train=140, n_factors=3, surveys=sv)
    cutoff = mkt.yields.index[149]
    later = sv["date"] > cutoff
    moved = sv.copy()
    moved.loc[later, "value"] += 2.0  # change surveys published after month 150
    alt = real_time_decomposition(mkt.yields, min_train=140, n_factors=3, surveys=moved)
    pd.testing.assert_frame_equal(base.loc[:cutoff], alt.loc[:cutoff])
    assert not np.allclose(base["term_premium"].iloc[-1], alt["term_premium"].iloc[-1])
