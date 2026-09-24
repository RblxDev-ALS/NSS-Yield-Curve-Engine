import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsscurve.data import resample_curve, simulate_curve  # noqa: E402
from nsscurve.model import NSSParams  # noqa: E402

CMT_MATS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


@pytest.fixture
def mats():
    return CMT_MATS.copy()


@pytest.fixture
def true_params():
    return NSSParams(4.0, -2.0, -1.5, 1.2, 0.6, 0.08)


@pytest.fixture(scope="session")
def weekly_panel():
    """~2 years of weekly synthetic CMT quotes plus the true factors."""
    daily, factors = simulate_curve(start="2020-01-01", periods=520, seed=11,
                                    return_factors=True)
    weekly = resample_curve(daily, "W")
    return weekly, factors.resample("W-FRI").last().loc[weekly.index]
