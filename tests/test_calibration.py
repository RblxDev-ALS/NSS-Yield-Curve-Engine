import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nss_engine.calibration import (
    CalibrationConfig,
    _batched_loadings,
    _loading_lambda_derivatives,
    _ParOperator,
    _solve_betas,
    calibrate,
    calibrate_panel,
)
from nss_engine.models import DIEBOLD_LI_LAMBDA, NSSCurve, ns_loadings

EXACT = CalibrationConfig(ridge=0.0)


def assert_curve_close(a: NSSCurve, b: NSSCurve, atol: float = 1e-5) -> None:
    np.testing.assert_allclose(a.as_array(), b.as_array(), atol=atol)


class TestExactRecovery:
    @pytest.mark.parametrize("fixture", ["humped_curve", "inverted_curve"])
    def test_nss_zero_target(self, request, maturities, fixture):
        true = request.getfixturevalue(fixture)
        res = calibrate(maturities, true.zero(maturities), EXACT)
        assert res.success
        assert res.rmse_bp < 1e-4
        assert_curve_close(res.curve, true)

    def test_nss_par_target(self, maturities, humped_curve):
        cfg = CalibrationConfig(ridge=0.0, target="par")
        res = calibrate(maturities, humped_curve.par_yield(maturities), cfg)
        assert res.rmse_bp < 1e-4
        assert_curve_close(res.curve, humped_curve, atol=1e-4)
        # The fitted values are par yields, not zero rates.
        np.testing.assert_allclose(res.fitted, res.curve.par_yield(maturities))

    def test_par_target_beats_zero_target_on_par_quotes(self, maturities, humped_curve):
        quotes = humped_curve.par_yield(maturities)
        zero_fit = calibrate(maturities, quotes, EXACT)
        par_fit = calibrate(maturities, quotes, CalibrationConfig(ridge=0.0, target="par"))
        grid = np.linspace(0.5, 30, 60)
        err_zero = np.abs(zero_fit.curve.zero(grid) - humped_curve.zero(grid)).max()
        err_par = np.abs(par_fit.curve.zero(grid) - humped_curve.zero(grid)).max()
        assert err_par < 1e-4 < err_zero, "treating par yields as zero rates biases the zero curve"

    def test_nelson_siegel(self, maturities):
        true = NSSCurve.nelson_siegel(5.0, -2.0, 1.5, 0.6)
        res = calibrate(maturities, true.zero(maturities), CalibrationConfig(model="ns", ridge=0.0))
        assert res.model == "ns"
        assert res.curve.beta3 == 0.0
        np.testing.assert_allclose(res.curve.as_array()[:3], true.as_array()[:3], atol=1e-6)
        assert res.curve.lambda1 == pytest.approx(0.6, abs=1e-6)

    def test_fixed_lambda_is_plain_ols(self, maturities, humped_curve):
        y = humped_curve.zero(maturities)
        res = calibrate(maturities, y, CalibrationConfig.diebold_li())
        ols = np.linalg.lstsq(ns_loadings(maturities, DIEBOLD_LI_LAMBDA), y, rcond=None)[0]
        np.testing.assert_allclose(res.curve.as_array()[:3], ols, atol=1e-10)
        assert res.curve.lambda1 == DIEBOLD_LI_LAMBDA


class TestRobustness:
    def test_global_optimality_against_brute_force(self, maturities):
        """The loss must be no worse than an exhaustive 300×300 λ grid."""
        rng = np.random.default_rng(11)
        true = NSSCurve(5.0, -1.0, -4.0, 3.0, 1.4, 0.12)
        y = true.zero(maturities) + rng.normal(0, 0.04, maturities.size)
        cfg = CalibrationConfig()
        res = calibrate(maturities, y, cfg)

        l1 = np.exp(np.linspace(*np.log(cfg.lambda1_bounds), 300))
        l2 = np.exp(np.linspace(*np.log(cfg.lambda2_bounds), 300))
        L1, L2 = np.meshgrid(l1, l2, indexing="ij")
        lam = np.column_stack([L1.ravel(), L2.ravel()])
        lam = lam[lam[:, 0] >= cfg.min_lambda_ratio * lam[:, 1]]
        X = _batched_loadings(maturities, lam, "nss")
        w = np.full(maturities.size, 1 / maturities.size)
        _, loss = _solve_betas(X, y, w, np.array([0, 0, cfg.ridge, cfg.ridge]))
        assert res.loss <= loss.min() + 1e-12

    def test_noisy_fit_is_at_noise_level(self, maturities, humped_curve):
        rng = np.random.default_rng(0)
        sigma_bp = 3.0
        rmses = []
        for _ in range(30):
            y = humped_curve.zero(maturities) + rng.normal(0, sigma_bp / 100, maturities.size)
            rmses.append(calibrate(maturities, y).rmse_bp)
        # With 11 points and 6 parameters the expected in-sample RMSE is σ·sqrt(5/11) ≈ 2.0 bp.
        assert 1.5 < np.mean(rmses) < 2.6

    def test_missing_tenors(self, maturities, humped_curve):
        y = humped_curve.zero(maturities)
        y[[0, 9]] = np.nan
        res = calibrate(maturities, y, EXACT)
        assert res.maturities.size == 9
        assert res.rmse_bp < 1e-4

    def test_falls_back_to_ns_with_few_points(self, humped_curve):
        tau = np.array([0.25, 2, 5, 10, 30])
        res = calibrate(tau, humped_curve.zero(tau))
        assert res.model == "ns" and res.curve.beta3 == 0.0 and res.success

    def test_too_few_points(self):
        res = calibrate([1, 10, 30], [4.0, np.nan, 4.5])
        assert not res.success
        assert np.isnan(res.curve.beta0)
        assert "only 2" in res.message

    def test_zero_weight_ignores_outlier(self, maturities, humped_curve):
        y = humped_curve.zero(maturities)
        y[7] += 0.5  # a 50 bp bad quote on the 7Y
        w = np.ones_like(y)
        w[7] = 0.0
        res = calibrate(maturities, y, EXACT, weights=w)
        assert_curve_close(res.curve, humped_curve, atol=1e-4)
        assert res.residuals_bp[7] == pytest.approx(50, abs=0.1)

    def test_input_validation(self, maturities):
        with pytest.raises(ValueError):
            calibrate(maturities, np.ones(3))
        with pytest.raises(ValueError):
            calibrate(maturities, np.ones_like(maturities), weights=np.ones(2))
        with pytest.raises(ValueError):
            CalibrationConfig(model="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            CalibrationConfig(lambda1_bounds=(2.0, 1.0))
        with pytest.raises(ValueError):
            CalibrationConfig(min_lambda_ratio=0.5)

    def test_smoothing_anchors_lambdas(self, maturities, humped_curve):
        rng = np.random.default_rng(5)
        y = humped_curve.zero(maturities) + rng.normal(0, 0.05, maturities.size)
        prev = NSSCurve(4.5, -2.0, -1.5, 2.0, 0.5, 0.2)
        free = calibrate(maturities, y, CalibrationConfig(), previous=prev)
        tied = calibrate(maturities, y, CalibrationConfig(lambda_smoothing=10.0), previous=prev)

        def dist(c):
            return abs(np.log(c.lambda1 / prev.lambda1)) + abs(np.log(c.lambda2 / prev.lambda2))

        assert dist(tied.curve) < 0.05
        assert dist(tied.curve) <= dist(free.curve)


@settings(max_examples=40, deadline=None)
@given(
    b0=st.floats(2, 8),
    b1=st.floats(-4, 3),
    b2=st.floats(-5, 5),
    b3=st.floats(-5, 5),
    l1=st.floats(0.35, 2.5),
    l2=st.floats(0.07, 0.22),
    noise=st.floats(0, 0.05),
)
def test_constraints_always_hold(b0, b1, b2, b3, l1, l2, noise):
    tau = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rng = np.random.default_rng(0)
    y = NSSCurve(b0, b1, b2, b3, l1, l2).zero(tau) + rng.normal(0, noise, tau.size)
    cfg = CalibrationConfig()
    c = calibrate(tau, y, cfg).curve
    assert cfg.lambda1_bounds[0] - 1e-9 <= c.lambda1 <= cfg.lambda1_bounds[1] + 1e-9
    assert cfg.lambda2_bounds[0] - 1e-9 <= c.lambda2 <= cfg.lambda2_bounds[1] + 1e-9
    assert c.lambda1 >= cfg.min_lambda_ratio * c.lambda2 * (1 - 1e-9)


@settings(max_examples=40, deadline=None)
@given(
    b0=st.floats(2, 8),
    b1=st.floats(-4, 3),
    b2=st.floats(-5, 5),
    b3=st.floats(-5, 5),
    l1=st.floats(0.35, 2.5),
    l2=st.floats(0.07, 0.22),
)
def test_noiseless_curves_are_fitted_exactly(b0, b1, b2, b3, l1, l2):
    tau = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true = NSSCurve(b0, b1, b2, b3, l1, l2)
    res = calibrate(tau, true.zero(tau), EXACT)
    assert res.rmse_bp < 0.01


def test_lambda_derivatives_match_finite_differences():
    tau = np.array([0.0, 1e-5, 0.08, 1.0, 5.0, 30.0])
    lam = np.array([0.7, 0.2])
    d = _loading_lambda_derivatives(tau, lam, "nss")
    h = 1e-6
    for j in range(2):
        e = np.eye(2)[j] * h
        num = (
            _batched_loadings(tau, (lam + e)[None], "nss")[0]
            - _batched_loadings(tau, (lam - e)[None], "nss")[0]
        ) / (2 * h)
        np.testing.assert_allclose(d[j], num, atol=1e-8)


def test_par_operator_matches_curve(maturities, humped_curve):
    op = _ParOperator(maturities)
    np.testing.assert_allclose(
        op(humped_curve.as_array()), humped_curve.par_yield(maturities), atol=1e-12
    )


class TestPanel:
    def test_panel_fit(self, small_market):
        fit = calibrate_panel(small_market.yields)
        n = len(small_market.yields)
        assert len(fit.params) == n
        assert fit.diagnostics["success"].all()
        assert 1.0 < fit.diagnostics["rmse_bp"].median() < 3.0  # 3 bp noise
        np.testing.assert_allclose(
            fit.residuals_bp.to_numpy(),
            (small_market.yields - fit.fitted).to_numpy() * 100,
            equal_nan=True,
        )

    def test_missing_tenor_preserved(self, small_market):
        y = small_market.yields.copy()
        y.iloc[:10, 3] = np.nan  # the 1Y tenor
        fit = calibrate_panel(y)
        assert fit.fitted.iloc[:10, 3].isna().all()
        assert fit.fitted.iloc[10:, 3].notna().all()
        assert fit.fitted.iloc[:, 0].isna().all()  # 1M does not exist before 2001

    def test_panel_accessors(self, small_market):
        fit = calibrate_panel(small_market.yields.iloc[:20])
        d = fit.params.index[5]
        assert fit.curve(d) == NSSCurve.from_mapping(fit.params.loc[d])
        assert fit.curve(d + pd.Timedelta(days=2)) == fit.curve(d)  # as-of lookup
        assert fit.curve(-1) == fit.curves()[-1]
        surf = fit.evaluate([1, 10])
        assert surf.shape == (20, 2)
        spread = fit.spread(10, 1)
        np.testing.assert_allclose(spread, surf[10.0] - surf[1.0])
        assert set(fit.to_frame().columns) >= {"beta0", "rmse_bp"}

    def test_smoothing_reduces_parameter_jitter(self, small_market):
        rough = calibrate_panel(small_market.yields, CalibrationConfig(lambda_smoothing=0.0))
        smooth = calibrate_panel(small_market.yields)
        jitter = lambda f: f.params["lambda1"].diff().abs().median()  # noqa: E731
        assert jitter(smooth) < 0.5 * jitter(rough)
        # ... at a small cost in fit.
        assert smooth.diagnostics["rmse_bp"].mean() < rough.diagnostics["rmse_bp"].mean() + 0.5

    def test_smoothed_fit_is_closer_to_truth(self, small_market):
        grid = np.linspace(0.25, 30, 40)
        truth = np.array(
            [NSSCurve.from_array(p).zero(grid) for p in small_market.true_params.to_numpy()]
        )

        def err(cfg):
            est = calibrate_panel(small_market.yields, cfg).evaluate(grid).to_numpy()
            return np.sqrt(np.mean((est - truth) ** 2))

        assert err(CalibrationConfig(lambda_smoothing=1e-3)) < err(
            CalibrationConfig(lambda_smoothing=0.0, ridge=0.0)
        )
