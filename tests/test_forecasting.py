import numpy as np
import pandas as pd
import pytest

from nss_engine import forecasting
from nss_engine.models import DIEBOLD_LI_LAMBDA, ns_loadings


def _var_sim(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    c = np.array([0.05, -0.02, 0.0])
    A = np.array([[0.98, 0.0, 0.0], [0.05, 0.9, 0.0], [0.0, 0.0, 0.8]])
    f = np.zeros((n, 3))
    f[0] = np.linalg.solve(np.eye(3) - A, c)
    for t in range(1, n):
        f[t] = c + A @ f[t - 1] + rng.normal(0, 0.1, 3)
    return f, c, A


def test_var_recovers_coefficients():
    f, c, A = _var_sim()
    m = forecasting.fit_factor_model(f, "var1")
    np.testing.assert_allclose(m.coef, A, atol=0.03)
    np.testing.assert_allclose(m.unconditional_mean, np.linalg.solve(np.eye(3) - A, c), atol=0.5)


def test_ar1_is_diagonal():
    f, _, A = _var_sim()
    m = forecasting.fit_factor_model(f, "ar1")
    assert np.count_nonzero(m.coef - np.diag(np.diag(m.coef))) == 0
    assert m.coef[0, 0] == pytest.approx(A[0, 0], abs=0.02)
    with pytest.raises(ValueError):
        forecasting.fit_factor_model(f, "garch")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        forecasting.fit_factor_model(f[:2])


def test_forecast_converges_to_mean():
    f, _, _ = _var_sim()
    m = forecasting.fit_factor_model(f, "var1")
    far = m.forecast(np.array([10.0, 10.0, 10.0]), 2000)
    np.testing.assert_allclose(far, m.unconditional_mean, atol=1e-6)


def test_extract_factors_exact():
    mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true = np.array([[5.0, -2.0, 1.0], [4.0, 0.5, -1.0]])
    Y = true @ ns_loadings(mats, DIEBOLD_LI_LAMBDA).T
    df = pd.DataFrame(Y, columns=mats, index=pd.date_range("2020-01-31", periods=2, freq="ME"))
    df.iloc[1, 0] = np.nan
    np.testing.assert_allclose(forecasting.extract_factors(df).to_numpy(), true, atol=1e-10)


def test_diebold_mariano():
    rng = np.random.default_rng(3)
    e_bench = rng.normal(0, 1.0, 600)
    good = rng.normal(0, 0.7, 600)
    stat, p = forecasting.diebold_mariano(good, e_bench, h=1)
    assert stat < 0 and p < 0.01
    stat, p = forecasting.diebold_mariano(e_bench, e_bench + rng.normal(0, 1e-3, 600), h=1)
    assert p > 0.05
    assert np.isnan(forecasting.diebold_mariano(good[:5], e_bench[:5], 1)[0])


def test_evaluate_forecasts_beats_random_walk_on_mean_reverting_curve():
    """Strongly mean-reverting factors: the model must beat 'no change' at long horizons."""
    rng = np.random.default_rng(4)
    n = 360
    mu = np.array([5.0, -2.0, 0.0])
    f = np.zeros((n, 3))
    f[0] = mu + 2
    for t in range(1, n):
        f[t] = mu + 0.85 * (f[t - 1] - mu) + rng.normal(0, 0.15, 3)
    mats = np.array([0.25, 1, 2, 5, 10, 30])
    Y = f @ ns_loadings(mats, DIEBOLD_LI_LAMBDA).T + rng.normal(0, 0.02, (n, mats.size))
    df = pd.DataFrame(Y, columns=mats, index=pd.date_range("1990-01-31", periods=n, freq="ME"))
    ev = forecasting.evaluate_forecasts(df, horizons=(1, 12), min_train=60)
    assert (ev.relative_rmse.loc[12] < 0.9).all()
    assert (ev.dm_stat.loc[12] < 0).all()
    assert ev.n_forecasts[1] == n - 60 and ev.n_forecasts[12] == n - 60 - 11
    rolled = forecasting.evaluate_forecasts(
        df, horizons=(6,), min_train=60, rolling_window=120, kind="var1"
    )
    assert rolled.rmse_model.shape == (1, mats.size)


def test_forecast_curve_shape(long_market):
    monthly = long_market.yields.resample("ME").mean().dropna(axis=1, thresh=300).dropna()
    fc = forecasting.forecast_curve(monthly, 12, maturities=[1, 5, 10])
    assert list(fc.index) == [1, 5, 10] and fc.notna().all()
