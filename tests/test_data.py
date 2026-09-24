import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from nss_engine import data
from nss_engine.data import (
    DataError,
    describe_panel,
    label_columns,
    load_yields_csv,
    maturity_label,
    parse_fred_csv,
    parse_fred_json,
    resample_yields,
    treasury_panel_from_fred_columns,
)

NEW_STYLE = "observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,3.99\n"
OLD_STYLE = "DATE,DGS2\n2024-01-02,4.33\n2024-01-03,4.30\n"


def test_parse_fred_csv_new_header_and_missing_values():
    s = parse_fred_csv(NEW_STYLE, "DGS10")
    assert list(s.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert s.iloc[0] == 3.95 and np.isnan(s.iloc[1])


def test_parse_fred_csv_old_header():
    s = parse_fred_csv(OLD_STYLE)
    assert s.name == "DGS2" and len(s) == 2


def test_parse_fred_csv_rejects_garbage():
    with pytest.raises(DataError):
        parse_fred_csv("just_one_column\n1\n2\n")


def test_parse_fred_json():
    payload = json.dumps(
        {
            "observations": [
                {"date": "2024-01-02", "value": "3.95"},
                {"date": "2024-01-03", "value": "."},
            ]
        }
    )
    s = parse_fred_json(payload, "DGS10")
    assert s.iloc[0] == 3.95 and np.isnan(s.iloc[1])
    with pytest.raises(DataError):
        parse_fred_json(json.dumps({"error_message": "bad key"}), "DGS10")


@pytest.mark.parametrize(
    ("tau", "label"),
    [(1 / 12, "1M"), (0.25, "3M"), (0.5, "6M"), (1.0, "1Y"), (10.0, "10Y"), (1.5, "1.5Y")],
)
def test_maturity_label(tau, label):
    assert maturity_label(tau) == label


class TestFetchWithCache:
    def test_downloads_then_uses_cache(self, tmp_path, monkeypatch):
        calls = []

        def fake_get(url, timeout=30.0, retries=4):
            calls.append(url)
            return NEW_STYLE

        monkeypatch.setattr(data, "_http_get", fake_get)
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        s1 = data.fetch_fred_series("DGS10", cache_dir=tmp_path)
        s2 = data.fetch_fred_series("DGS10", cache_dir=tmp_path)
        assert len(calls) == 1, "second call must be served from the cache"
        assert "id=DGS10" in calls[0]
        pd.testing.assert_series_equal(s1, s2, check_freq=False)
        assert (tmp_path / "DGS10.csv").exists()

    def test_refresh_and_expiry(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(data, "_http_get", lambda url, **k: calls.append(url) or NEW_STYLE)
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        data.fetch_fred_series("DGS10", cache_dir=tmp_path)
        data.fetch_fred_series("DGS10", cache_dir=tmp_path, refresh=True)
        old = time.time() - 48 * 3600
        os.utime(tmp_path / "DGS10.csv", (old, old))
        data.fetch_fred_series("DGS10", cache_dir=tmp_path)
        assert len(calls) == 3

    def test_stale_cache_used_when_download_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        monkeypatch.setattr(data, "_http_get", lambda url, **k: NEW_STYLE)
        data.fetch_fred_series("DGS10", cache_dir=tmp_path)

        def boom(url, **k):
            raise DataError("offline")

        monkeypatch.setattr(data, "_http_get", boom)
        with pytest.warns(RuntimeWarning, match="stale cache"):
            s = data.fetch_fred_series("DGS10", cache_dir=tmp_path, refresh=True)
        assert len(s) == 3

    def test_no_cache_no_network_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        def boom(url, **k):
            raise DataError("offline")

        monkeypatch.setattr(data, "_http_get", boom)
        with pytest.raises(DataError):
            data.fetch_fred_series("DGS10", cache_dir=tmp_path)

    def test_api_key_uses_json_endpoint(self, tmp_path, monkeypatch):
        seen = []
        payload = json.dumps({"observations": [{"date": "2024-01-02", "value": "4.0"}]})
        monkeypatch.setattr(data, "_http_get", lambda url, **k: seen.append(url) or payload)
        s = data.fetch_fred_series("DGS10", cache_dir=tmp_path, api_key="abc")
        assert "api_key=abc" in seen[0] and s.iloc[0] == 4.0

    def test_load_treasury_yields(self, tmp_path, monkeypatch):
        dates = pd.bdate_range("2024-01-01", periods=15)

        def fake_get(url, **k):
            sid = url.split("id=")[1]
            tau = data.TREASURY_SERIES[sid]
            vals = [f"{4 + 0.05 * tau + 0.01 * i:.2f}" for i in range(len(dates))]
            rows = "\n".join(f"{d:%Y-%m-%d},{v}" for d, v in zip(dates, vals, strict=True))
            return f"observation_date,{sid}\n{rows}\n"

        monkeypatch.setattr(data, "_http_get", fake_get)
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        panel = data.load_treasury_yields("2024-01-01", None, freq="W-FRI", cache_dir=tmp_path)
        assert list(panel.columns) == sorted(data.TREASURY_SERIES.values())
        assert len(panel) == 3
        assert panel.index.name == "date"


def test_resample_last_and_mean():
    idx = pd.bdate_range("2024-01-01", periods=10)
    df = pd.DataFrame({2.0: np.arange(10.0)}, index=idx)
    df.iloc[4, 0] = np.nan  # Friday missing -> 'last' uses Thursday's quote
    weekly = resample_yields(df, "W-FRI", "last")
    assert weekly.iloc[0, 0] == 3.0
    assert resample_yields(df, "W-FRI", "mean").iloc[1, 0] == pytest.approx(7.0)
    assert len(resample_yields(df, None)) == 9  # the all-NaN day is dropped
    with pytest.raises(ValueError):
        resample_yields(df, "W-FRI", "median")


def test_panel_from_fred_columns():
    df = pd.DataFrame({"DGS10": [4.0], "DGS2": [4.5], "OTHER": [1.0]})
    out = treasury_panel_from_fred_columns(df)
    assert list(out.columns) == [2.0, 10.0]
    with pytest.raises(DataError):
        treasury_panel_from_fred_columns(pd.DataFrame({"X": [1.0]}))


def test_load_yields_csv_accepts_several_column_styles(tmp_path):
    path = tmp_path / "y.csv"
    path.write_text("date,DGS10,3M,2Y,0.5\n2024-01-05,4.0,5.3,4.4,5.2\n2024-01-12,4.1,.,4.3,5.1\n")
    df = load_yields_csv(path)
    assert list(df.columns) == [0.25, 0.5, 2.0, 10.0]
    assert np.isnan(df.loc["2024-01-12", 0.25])
    bad = tmp_path / "bad.csv"
    bad.write_text("date,foo\n2024-01-05,1\n")
    with pytest.raises(DataError):
        load_yields_csv(bad)


def test_label_and_describe(small_market):
    labelled = label_columns(small_market.yields)
    assert "10Y" in labelled.columns and "1M" in labelled.columns
    desc = describe_panel(small_market.yields)
    assert desc.loc["1M", "obs"] < desc.loc["10Y", "obs"]  # 1M blanked before 2001
