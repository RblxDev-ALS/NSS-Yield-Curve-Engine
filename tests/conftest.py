import numpy as np
import pytest

from nss_engine.models import NSSCurve
from nss_engine.synthetic import simulate_market

#: The 11 FRED constant-maturity tenors, in years.
CMT = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)


@pytest.fixture
def maturities() -> np.ndarray:
    return CMT.copy()


@pytest.fixture
def humped_curve() -> NSSCurve:
    """A realistic upward-sloping curve with a short-end hump and long-end bend."""
    return NSSCurve(beta0=4.5, beta1=-2.0, beta2=-1.5, beta3=2.0, lambda1=0.9, lambda2=0.15)


@pytest.fixture
def inverted_curve() -> NSSCurve:
    """A 2023-style inverted curve."""
    return NSSCurve(beta0=3.9, beta1=1.6, beta2=-2.5, beta3=1.2, lambda1=1.2, lambda2=0.2)


@pytest.fixture(scope="session")
def small_market():
    return simulate_market(periods=156, seed=3)


@pytest.fixture(scope="session")
def long_market():
    """34 years of weekly data with four stylised recessions (seed 0)."""
    return simulate_market(seed=0)
