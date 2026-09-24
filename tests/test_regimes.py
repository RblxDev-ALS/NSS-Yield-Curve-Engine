import numpy as np
import pandas as pd
import pytest

from nsscurve.regimes import (FLAT, INVERTED, STEEP, classify_curve_moves,
                              classify_slope_regime, recession_probability, regime_segments)


def _series(vals):
    return pd.Series(vals, index=pd.date_range("2024-01-05", periods=len(vals), freq="W-FRI"))


def test_basic_classification():
    r = classify_slope_regime(_series([1.0, 1.0, 0.2, 0.2, -0.5, -0.5]), min_persistence=1)
    assert r.tolist() == [STEEP, STEEP, FLAT, FLAT, INVERTED, INVERTED]


def test_hysteresis_prevents_flicker():
    # Oscillates around the inversion threshold (-0.1) inside the +/-0.05 band.
    vals = [-0.3, -0.08, -0.12, -0.07, -0.13, -0.09]
    r = classify_slope_regime(_series(vals), band=0.05, min_persistence=1)
    assert set(r) == {INVERTED}
    no_band = classify_slope_regime(_series(vals), band=0.0, min_persistence=1)
    assert FLAT in set(no_band)


def test_persistence_filter_ignores_one_off_spikes():
    vals = [1.0, 1.0, -0.5, 1.0, 1.0, -0.5, -0.5, -0.5]
    r = classify_slope_regime(_series(vals), min_persistence=2)
    assert r.iloc[2] == STEEP                 # single-period spike ignored
    assert r.iloc[6] == INVERTED              # confirmed after 2 observations


def test_nan_carries_previous_state_and_validation():
    r = classify_slope_regime(_series([1.0, np.nan, 1.0]))
    assert r.tolist() == [STEEP, STEEP, STEEP]
    with pytest.raises(ValueError):
        classify_slope_regime(_series([1.0]), steep=-1, inverted=0)


def test_curve_moves():
    level = _series([4.0, 4.0, 4.3, 4.3, 3.9, 3.9, 3.9])
    slope = _series([0.5, 0.5, 0.7, 0.7, 0.3, 0.3, 0.31])
    m = classify_curve_moves(level, slope, window=1, threshold_bp=5)["Move"]
    assert pd.isna(m.iloc[0])
    assert m.iloc[2] == "Bear Steepener"
    assert m.iloc[4] == "Bull Flattener"
    assert m.iloc[5] == "Range-bound"
    m2 = classify_curve_moves(_series([4.0, 4.2]), _series([0.5, 0.5]), 1)["Move"]
    assert m2.iloc[1] == "Parallel Sell-off"


def test_recession_probability_matches_nyfed_model():
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    p = recession_probability(pd.Series(0.0, index=idx))
    assert len(p) == 3                                     # monthly averages
    assert p.iloc[0] == pytest.approx(0.2969, abs=1e-3)    # Phi(-0.5333)
    probs = recession_probability(pd.Series([-1.0, 0.0, 1.0, 2.0], index=idx[:4]), monthly=False)
    assert probs.is_monotonic_decreasing


def test_regime_segments():
    r = _series(["A", "A", "B", "B", "B", "A"])
    segs = regime_segments(r)
    assert [s[2] for s in segs] == ["A", "B", "A"]
    assert segs[0][0] == r.index[0] and segs[-1][1] == r.index[-1]
    assert regime_segments(pd.Series([], dtype=object)) == []
