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
from nss_engine.models import DIEBOLD_LI_LAMBDA, NSSCurve, ns_loadings, nss_loadings
from nss_engine.synthetic import simulate_market

EXACT = CalibrationConfig(ridge=0.0, target="yield")


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
        res = calibrate(
            maturities,
            true.zero(maturities),
            CalibrationConfig(model="ns", ridge=0.0, target="yield"),
        )
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


class TestParCalibration:
    """Par-yield fitting: the default, because FRED CMT yields are par yields."""

    def test_par_operator_matches_curve_and_finite_differences(self, maturities, humped_curve):
        op = _ParOperator(maturities)
        p = humped_curve.as_array()
        value, jac = op.value_and_jacobian(p)
        np.testing.assert_allclose(value, humped_curve.par_yield(maturities), atol=1e-12)
        fd = np.column_stack([(op(p + e) - op(p - e)) / 2e-6 for e in np.eye(6) * 1e-6])
        np.testing.assert_allclose(jac, fd, atol=1e-6 * np.abs(fd).max())

    def test_coupon_dates_are_deduplicated(self, maturities):
        op = _ParOperator(maturities)
        assert op.times.size == 60  # every half-year to 30y, once
        # each bond's annuity weights still sum to (number of coupons) / freq
        np.testing.assert_allclose(
            op.annuity_matrix.sum(axis=1)[maturities > 1], maturities[maturities > 1]
        )

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"model": "ns"}, {"fixed_lambda2": 0.2}, {"fixed_lambda1": 0.7, "fixed_lambda2": 0.2}],
    )
    def test_residual_jacobian_is_exact(self, maturities, humped_curve, kwargs, monkeypatch):
        import nss_engine.calibration as cal

        captured = {}
        real = cal.least_squares

        def spy(fun, x0, jac=None, **kw):
            captured.update(fun=fun, jac=jac, x=np.asarray(x0) + 0.01)
            return real(fun, x0, jac=jac, **kw)

        monkeypatch.setattr(cal, "least_squares", spy)
        cfg = CalibrationConfig(target="par", lambda_smoothing=1e-3, **kwargs)
        y = humped_curve.par_yield(maturities) + 0.01 * np.sin(maturities)
        calibrate(maturities, y, cfg, previous=humped_curve)
        f, j, x = captured["fun"], captured["jac"], captured["x"]
        fd = np.column_stack([(f(x + e) - f(x - e)) / 2e-7 for e in np.eye(x.size) * 1e-7])
        np.testing.assert_allclose(j(x), fd, atol=1e-6 * max(np.abs(fd).max(), 1.0))

    def test_par_fit_is_globally_optimal(self, maturities):
        """No par polish started anywhere on a λ grid beats the calibrator."""
        from nss_engine.calibration import _calibrate_par

        rng = np.random.default_rng(5)
        cfg = CalibrationConfig()
        w = np.full(maturities.size, 1 / maturities.size)
        for true in (
            NSSCurve(5.0, -1.0, -4.0, 3.0, 1.4, 0.12),
            NSSCurve(3.9, 1.6, -2.5, 1.2, 1.2, 0.2),
        ):
            y = true.par_yield(maturities) + rng.normal(0, 0.03, maturities.size)
            res = calibrate(maturities, y, cfg)
            best = np.inf
            for l1 in np.geomspace(*cfg.lambda1_bounds, 7):
                for l2 in np.geomspace(*cfg.lambda2_bounds, 7):
                    if l1 < cfg.min_lambda_ratio * l2:
                        continue
                    b = np.linalg.lstsq(nss_loadings(maturities, l1, l2), y, rcond=None)[0]
                    start = np.array([*b, l1, l2])
                    best = min(best, _calibrate_par(maturities, y, w, cfg, "nss", start, None)[1])
            assert res.loss <= best * (1 + 1e-6)

    def test_zero_target_is_biased_on_par_quotes(self):
        """In-sample fit cannot tell the two targets apart; distance to the truth can."""
        mkt = simulate_market(periods=40, seed=2, noise_bp=2.0, missing_short_end_until=None)
        grid = np.linspace(0.5, 30, 40)
        truth = np.array([NSSCurve.from_array(p).zero(grid) for p in mkt.true_params.to_numpy()])
        err, rmse = {}, {}
        for target in ("yield", "par"):
            fit = calibrate_panel(mkt.yields, CalibrationConfig(target=target))
            err[target] = np.sqrt(np.mean((fit.evaluate(grid).to_numpy() - truth) ** 2)) * 100
            rmse[target] = fit.diagnostics["rmse_bp"].mean()
        assert abs(rmse["par"] - rmse["yield"]) < 0.2
        assert err["par"] < 0.5 * err["yield"]


class TestRobustness:
    def test_global_optimality_against_brute_force(self, maturities):
        """The loss must be no worse than an exhaustive 300×300 λ grid."""
        rng = np.random.default_rng(11)
        true = NSSCurve(5.0, -1.0, -4.0, 3.0, 1.4, 0.12)
        y = true.zero(maturities) + rng.normal(0, 0.04, maturities.size)
        cfg = CalibrationConfig(target="yield")
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


@settings(max_examples=int(__import__("os").environ.get("HYPOTHESIS_EXAMPLES", 40)), deadline=None)
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


@settings(max_examples=int(__import__("os").environ.get("HYPOTHESIS_EXAMPLES", 40)), deadline=None)
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
    # 0.05 bp is 20x finer than the 1 bp precision of FRED quotes. Exact (1e-8)
    # recovery is not guaranteed: when β3 is tiny, λ2 is nearly unidentified and
    # the loss surface is flat to within a few hundredths of a basis point.
    assert res.rmse_bp < 0.05


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
        spread = fit.spread(10, 1, measure="zero")
        np.testing.assert_allclose(spread, surf[10.0] - surf[1.0])
        par = fit.evaluate([1, 10], "par")
        np.testing.assert_allclose(fit.spread(10, 1), par[10.0] - par[1.0])  # par by default
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


# U.S. Treasury constant-maturity yields on 2026-09-23 (FRED H.15, percent).
REAL_2026_09_23 = {
    1 / 12: 3.99, 0.25: 4.19, 0.5: 4.31, 1: 4.49, 2: 4.85, 3: 4.97,
    5: 4.99, 7: 5.05, 10: 5.11, 20: 5.45, 30: 5.40,
}  # fmt: skip


class TestRealCurve:
    """Regression tests on an actual market curve (no network needed)."""

    tau = np.array(list(REAL_2026_09_23))
    y = np.array(list(REAL_2026_09_23.values()))

    def test_fit_quality(self):
        res = calibrate(self.tau, self.y)
        assert res.success
        assert res.rmse_bp < 8.0  # typical NSS fit to CMT data is a few bp
        assert abs(res.curve.spread(10, 0.25) - (5.11 - 4.19)) < 0.10

    def test_nss_beats_ns_in_sample(self):
        nss = calibrate(self.tau, self.y)
        ns = calibrate(self.tau, self.y, CalibrationConfig(model="ns"))
        assert nss.rmse_bp < ns.rmse_bp

    def test_twenty_year_trades_cheap(self):
        # The 20Y bond has sat above the fitted curve since its 2020 reissue.
        res = calibrate(self.tau, self.y)
        resid = dict(zip(res.maturities, res.residuals_bp, strict=True))
        assert resid[20.0] == max(resid.values()) and resid[20.0] > 3.0

    def test_par_fit(self):
        res = calibrate(self.tau, self.y, CalibrationConfig(target="par"))
        assert res.success and res.rmse_bp < 8.0


class TestRobustFitting:
    def test_rejects_a_bad_quote_and_recovers_the_curve(self, maturities, humped_curve):
        rng = np.random.default_rng(3)
        y = humped_curve.par_yield(maturities) + rng.normal(0, 0.02, maturities.size)
        clean = calibrate(maturities, y.copy())
        y[6] += 0.40  # a 40 bp bad 5Y quote
        plain = calibrate(maturities, y)
        robust = calibrate(maturities, y, CalibrationConfig(robust=True))
        assert robust.outliers[6] and robust.outliers.sum() == 1
        assert robust.robust_weights[6] < 0.05
        assert plain.outliers.sum() == 0  # a non-robust fit never flags
        grid = np.linspace(0.5, 30, 40)
        err = {
            name: np.abs(f.curve.zero(grid) - humped_curve.zero(grid)).max() * 100
            for name, f in (("clean", clean), ("plain", plain), ("robust", robust))
        }
        # As accurate as if the bad quote had never been there; the plain fit is not.
        assert err["robust"] < 1.2 * err["clean"]
        assert err["plain"] > 2 * err["robust"]
        # The rejected quote's residual shows the full dislocation.
        assert robust.residuals_bp[6] == pytest.approx(40, abs=5)

    def test_high_leverage_outlier_at_the_short_end(self, maturities, humped_curve):
        """A bad 1M bill: NSS bends towards it, so raw residuals understate it."""
        y = humped_curve.par_yield(maturities)
        y[0] -= 0.30
        plain = calibrate(maturities, y)
        assert plain.leverage[0] > 0.5
        robust = calibrate(maturities, y, CalibrationConfig(robust=True))
        assert robust.outliers[0]
        assert abs(robust.residuals_bp[0]) > 2 * abs(plain.residuals_bp[0])

    def test_clean_data_is_left_alone(self, maturities, humped_curve):
        y = humped_curve.par_yield(maturities)
        plain = calibrate(maturities, y)
        robust = calibrate(maturities, y, CalibrationConfig(robust=True))
        assert not robust.outliers.any()
        np.testing.assert_allclose(robust.curve.as_array(), plain.curve.as_array())

    def test_zero_target_and_panel(self, small_market):
        y = small_market.yields.iloc[:30].copy()
        y.iloc[10, 5] += 0.5
        fit = calibrate_panel(y, CalibrationConfig(robust=True))
        assert fit.outliers.iloc[10, 5]
        # ~2% of clean quotes are flagged at huber_k = 3 (3 bp noise, 2 bp scale floor)
        assert fit.outliers.to_numpy().sum() <= 0.03 * y.notna().to_numpy().sum()
        assert "n_outliers" in fit.diagnostics
        row = y.iloc[10].to_numpy()
        zero = calibrate(np.asarray(y.columns), row, CalibrationConfig(robust=True, target="yield"))
        observed_cols = np.flatnonzero(np.isfinite(row))
        assert observed_cols[zero.outliers].tolist() == [5]

    def test_rejects_bad_settings(self):
        with pytest.raises(ValueError):
            CalibrationConfig(huber_k=0)


class TestUncertainty:
    def test_leverage_sums_to_number_of_parameters(self, maturities, humped_curve):
        y = humped_curve.par_yield(maturities) + 0.01 * np.cos(maturities)
        res = calibrate(maturities, y, CalibrationConfig(ridge=0.0))
        assert res.leverage.sum() == pytest.approx(6.0, abs=1e-8)
        assert res.dof == pytest.approx(maturities.size - 6)
        ns = calibrate(maturities, y, CalibrationConfig(model="ns", ridge=0.0))
        assert ns.n_params == 4 and ns.leverage.sum() == pytest.approx(4.0, abs=1e-8)
        # Penalties pin parameters partly, so they use up less than a full degree of freedom.
        ridged = calibrate(maturities, y)
        assert 5.0 < ridged.leverage.sum() < 6.0
        smoothed = calibrate(
            maturities, y, CalibrationConfig(lambda_smoothing=0.1), previous=res.curve
        )
        assert smoothed.leverage.sum() < ridged.leverage.sum()

    def test_bands_have_nominal_coverage(self, maturities, humped_curve):
        rng = np.random.default_rng(0)
        grid = np.array([0.5, 2, 5, 10, 25])
        hits, sigmas = [], []
        for _ in range(120):
            y = humped_curve.par_yield(maturities) + rng.normal(0, 0.03, maturities.size)
            res = calibrate(maturities, y, CalibrationConfig(ridge=0.0))
            lo, hi = res.confidence_band(grid)
            truth = humped_curve.zero(grid)
            hits.append((lo <= truth) & (truth <= hi))
            sigmas.append(res.sigma_bp)
        assert 0.89 < np.mean(hits) < 0.99
        assert np.mean(sigmas) == pytest.approx(3.0, rel=0.15)

    def test_covariance_shape_and_fixed_parameters(self, maturities, humped_curve):
        y = humped_curve.zero(maturities) + 0.01 * np.sin(3 * maturities)
        res = calibrate(maturities, y, CalibrationConfig.diebold_li())
        cov = res.covariance()
        assert cov.shape == (6, 6)
        np.testing.assert_array_equal(cov[3:, :], 0.0)  # β3, λ1, λ2 not estimated
        se = res.standard_errors()
        assert se["beta0"] > 0 and se["lambda1"] == 0
        lo, hi = res.confidence_band([1, 5, 10], measure="forward")
        assert np.all(lo < hi)

    def test_exact_fit_has_no_band(self, maturities, humped_curve):
        tau = maturities[:6]
        res = calibrate(tau, humped_curve.par_yield(tau), CalibrationConfig(model="ns"))
        assert res.n_params == 4
        few = calibrate(tau[:4], humped_curve.par_yield(tau[:4]), CalibrationConfig(model="ns"))
        assert np.isnan(few.sigma_bp)
        assert np.all(np.isnan(few.confidence_band([1, 2])[0]))


def test_par_fit_with_lambda1_at_its_upper_bound(maturities):
    """A very short hump (λ1 = 5 > 3) must pin λ1 at its bound, not abandon the par fit."""
    true = NSSCurve(4.0, -2.0, 6.0, -3.0, 4.0, 1.0)
    res = calibrate(maturities, true.par_yield(maturities))
    assert res.success
    assert res.curve.lambda1 == pytest.approx(CalibrationConfig().lambda1_bounds[1])
    assert res.curve.lambda1 >= CalibrationConfig().min_lambda_ratio * res.curve.lambda2 - 1e-9
    # ... and match the best par fit with λ1 fixed at that bound
    pinned = calibrate(maturities, true.par_yield(maturities), CalibrationConfig(fixed_lambda1=3.0))
    assert res.loss <= pinned.loss * (1 + 1e-6)
