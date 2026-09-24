import numpy as np
import pandas as pd
import pytest

from nsscurve import data
from nsscurve.data import (DataError, FREDClient, clean_curve, load_csv, resample_curve,
                           simulate_curve)
from nsscurve.tenors import format_tenor, maturities_of, parse_tenor


@pytest.mark.parametrize("label,years", [("1M", 1 / 12), ("3M", 0.25), ("6m", 0.5),
                                         ("10Y", 10.0), ("DGS3MO", 0.25), ("DGS30", 30.0),
                                         (7, 7.0), ("2W", 14 / 365)])
def test_parse_tenor(label, years):
    assert parse_tenor(label) == pytest.approx(years)


def test_format_tenor_round_trip():
    for lab in ["1M", "3M", "6M", "1Y", "2Y", "30Y"]:
        assert format_tenor(parse_tenor(lab)) == lab
    with pytest.raises(ValueError):
        parse_tenor("banana")


def test_clean_curve_keeps_structural_gaps_and_orders_columns():
    idx = pd.bdate_range("2024-01-01", periods=12)
    df = pd.DataFrame({"DGS10": np.linspace(4, 4.1, 12), "DGS2": np.linspace(4.5, 4.4, 12),
                       "DGS20": np.nan}, index=idx)
    df.iloc[3, 0] = np.nan                                 # one-day hole -> filled
    df.loc[idx[5], :] = np.nan                             # holiday row -> dropped
    out = clean_curve(df, max_ffill=2)
    assert list(out.columns) == ["2Y", "10Y"]              # empty 20Y dropped, sorted
    assert len(out) == 11
    assert out["10Y"].notna().all()


def test_ffill_is_bounded():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({"2Y": [1.0] + [np.nan] * 8 + [2.0], "10Y": 3.0}, index=idx)
    out = clean_curve(df, max_ffill=3)
    assert out["2Y"].isna().sum() == 5


def test_resample_weekly_uses_friday_close():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({"2Y": np.arange(10.0)}, index=idx)
    w = resample_curve(df, "W")
    assert all(d.dayofweek == 4 for d in w.index)
    assert w["2Y"].tolist() == [4.0, 9.0]
    assert resample_curve(df, "D") is df


def test_load_csv_accepts_fred_ids(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("DATE,DGS2,DGS10\n2024-01-02,4.3,3.9\n2024-01-03,.,3.95\n")
    df = load_csv(p)
    assert list(df.columns) == ["2Y", "10Y"]
    assert df["2Y"].iloc[1] == 4.3                         # '.' treated as missing then filled


def test_simulate_curve_is_deterministic_and_realistic():
    a, f = simulate_curve(periods=300, seed=3, return_factors=True)
    b = simulate_curve(periods=300, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a.shape == (300, 11)
    assert (f["Beta0"] + f["Beta1"] >= 0.1 - 1e-12).all()   # positive short rate
    assert a.abs().max().max() < 20
    with pytest.raises(ValueError):
        simulate_curve(periods=10, quote_type="bogus")


@pytest.mark.parametrize("header", ["observation_date", "DATE"])
def test_fred_client_parses_csv_and_caches(tmp_path, monkeypatch, header):
    calls = []

    def fake_get(self, url):
        calls.append(url)
        return f"{header},DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,4.01\n".encode()

    monkeypatch.setattr(FREDClient, "_get", fake_get)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    client = FREDClient(cache_dir=tmp_path)
    s = client.fetch_series("DGS10")
    assert s.iloc[0] == 3.95 and np.isnan(s.iloc[1])
    client.fetch_series("DGS10")                          # served from cache
    assert len(calls) == 1
    curve = client.fetch_curve(tenors=["10Y"])
    assert list(curve.columns) == ["10Y"]


def test_fred_client_json_api(tmp_path, monkeypatch):
    payload = b'{"observations": [{"date": "2024-01-02", "value": "4.1"},' \
              b' {"date": "2024-01-03", "value": "."}]}'
    monkeypatch.setattr(FREDClient, "_get", lambda self, url: payload)
    s = FREDClient(api_key="k", cache_dir=None).fetch_series("DGS2")
    assert s.iloc[0] == 4.1 and np.isnan(s.iloc[1])


def test_fred_client_retries_then_raises(monkeypatch):
    attempts = []

    def boom(*a, **k):
        attempts.append(1)
        raise OSError("blocked")

    monkeypatch.setattr(data.urllib.request, "urlopen", boom)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    with pytest.raises(DataError):
        FREDClient(cache_dir=None, retries=3)._get("http://example.invalid")
    assert len(attempts) == 3


def test_maturities_of():
    np.testing.assert_allclose(maturities_of(["3M", "2Y", "DGS30"]), [0.25, 2, 30])
