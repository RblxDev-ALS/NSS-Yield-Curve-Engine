import numpy as np
import pandas as pd
import pytest

from nss_engine import data
from nss_engine.calibration import calibrate_panel
from nss_engine.data import DataError, load_gsw_parameters, parse_gsw_csv
from nss_engine.models import NSSCurve
from nss_engine.validation import compare_to_reference

# Mimics feds200628.csv: a preamble of notes, then a header row starting with Date.
GSW_TEXT = """\
"The yield curve fitting method is described in Gurkaynak, Sack and Wright (2007)."
"Note: ..."

Date,BETA0,BETA1,BETA2,BETA3,SVEN1F01,SVENY01,SVENY10,TAU1,TAU2
1975-01-02,7.5,-1.2,0.8,NA,7.1,6.9,7.4,1.5,NA
2024-01-02,4.1,-0.9,-1.8,3.2,4.6,4.5,3.9,1.2,11.0
2024-01-03,NA,NA,NA,NA,NA,NA,NA,NA,NA
2024-01-04,4.2,-0.8,-1.7,3.0,4.6,4.5,4.0,1.3,10.5
"""


def test_parse_gsw_csv():
    p = parse_gsw_csv(GSW_TEXT)
    assert list(p.columns) == ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    assert len(p) == 3  # the all-NA row is dropped
    old = p.loc["1975-01-02"]
    assert old["beta3"] == 0.0 and old["lambda2"] == pytest.approx(1 / 1.5)  # NS era
    new = p.loc["2024-01-02"]
    assert new["lambda1"] == pytest.approx(1 / 1.2) and new["lambda2"] == pytest.approx(1 / 11)
    # GSW τ are time constants: our loadings with λ = 1/τ reproduce their formula.
    c = NSSCurve.from_mapping(new)
    x = 10 / 1.2
    manual = 4.1 - 0.9 * (1 - np.exp(-x)) / x - 1.8 * ((1 - np.exp(-x)) / x - np.exp(-x))
    x2 = 10 / 11.0
    manual += 3.2 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))
    assert c.zero(10.0)[0] == pytest.approx(manual)


def test_parse_gsw_rejects_other_files():
    with pytest.raises(DataError):
        parse_gsw_csv("a,b\n1,2\n")
    with pytest.raises(DataError):
        parse_gsw_csv("Date,BETA0\n2024-01-02,4.0\n")


def test_load_gsw_uses_cache(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout=30.0, retries=4):
        calls.append(url)
        return GSW_TEXT

    monkeypatch.setattr(data, "_http_get", fake_get)
    a = load_gsw_parameters(start="2024-01-01", cache_dir=tmp_path)
    b = load_gsw_parameters(start="2024-01-01", cache_dir=tmp_path)
    assert len(calls) == 1 and "feds200628" in calls[0]
    assert len(a) == 2
    pd.testing.assert_frame_equal(a, b)

    def failing(url, timeout=30.0, retries=4):
        raise DataError("offline")

    monkeypatch.setattr(data, "_http_get", failing)
    with pytest.warns(RuntimeWarning, match="stale cache"):
        c = load_gsw_parameters(cache_dir=tmp_path, refresh=True)
    assert len(c) == 3
    with pytest.raises(DataError):
        load_gsw_parameters(cache_dir=tmp_path / "empty")


class TestCompareToReference:
    def test_fit_agrees_with_truth(self, small_market):
        fit = calibrate_panel(small_market.yields.iloc[:40])
        cmp = compare_to_reference(fit, small_market.true_params)
        assert cmp.n_dates == 40
        ov = cmp.overall()
        assert ov["rmse_bp"] < 4.0
        assert ov["mean_change_corr"] > 0.9
        s = cmp.summary()
        assert {"bias_bp", "std_bp", "rmse_bp", "change_corr"} <= set(s.columns)
        assert "5y5y fwd" in s.index and "10Y" in s.index

    def test_constant_offset_is_bias_not_tracking_error(self, small_market):
        truth = small_market.true_params.iloc[:30]
        shifted = truth.copy()
        shifted["beta0"] += 0.05  # +5 bp at every maturity
        cmp = compare_to_reference(truth, shifted)
        s = cmp.summary()
        np.testing.assert_allclose(s["bias_bp"], -5.0, atol=1e-9)
        assert cmp.overall()["demeaned_rmse_bp"] < 1e-9
        np.testing.assert_allclose(s["change_corr"], 1.0)

    def test_subset(self, small_market):
        truth = small_market.true_params.iloc[:30]
        cmp = compare_to_reference(truth, truth)
        sub = cmp.subset(truth.index[5:12].append(pd.Index([pd.Timestamp("1800-01-01")])))
        assert sub.n_dates == 7
        assert list(sub.engine.columns) == list(cmp.engine.columns)

    def test_no_common_dates(self, small_market):
        p = small_market.true_params
        with pytest.raises(ValueError):
            compare_to_reference(p.iloc[:5], p.iloc[5:10])
