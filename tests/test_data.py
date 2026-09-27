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
    # Periods are labelled by their last actual observation, never a future date.
    assert weekly.index[-1] == idx[-1] and weekly.index[0] == idx[3]  # Friday was missing
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
    path.write_text(
        "date,DGS10,3M,2Y,0.5\n2024-01-05,4.0,5.3,4.4,5.2\n2024-01-12,4.1,.,4.3,5.1\n",
        encoding="utf-8",
    )
    df = load_yields_csv(path)
    assert list(df.columns) == [0.25, 0.5, 2.0, 10.0]
    assert np.isnan(df.loc["2024-01-12", 0.25])
    bad = tmp_path / "bad.csv"
    bad.write_text("date,foo\n2024-01-05,1\n", encoding="utf-8")
    with pytest.raises(DataError):
        load_yields_csv(bad)


def test_label_and_describe(small_market):
    labelled = label_columns(small_market.yields)
    assert "10Y" in labelled.columns and "1M" in labelled.columns
    desc = describe_panel(small_market.yields)
    assert desc.loc["1M", "obs"] < desc.loc["10Y", "obs"]  # 1M blanked before 2001


def test_load_kim_wright_term_premium(monkeypatch):
    calls = []

    def fake(series_id, **kwargs):
        calls.append(series_id)
        idx = pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"])
        return pd.Series([0.5, np.nan, 0.4], index=idx)

    monkeypatch.setattr(data, "fetch_fred_series", fake)
    s = data.load_kim_wright_term_premium(start="2020-01-01", maturity=5)
    assert calls == ["THREEFYTP5"]
    assert s.name == "kim_wright_tp5" and list(s) == [0.4]
    with pytest.raises(ValueError):
        data.load_kim_wright_term_premium(maturity=30)


# ---- Survey of Professional Forecasters ---------------------------------------------


def test_spf_bill_windows():
    from nss_engine.data import spf_bill_windows

    # a first-quarter survey is dated end-February: month 1 = March
    assert spf_bill_windows("TBILL3", 1) == (2, 4)  # Q2 = April-June
    assert spf_bill_windows("TBILL6", 3) == (11, 13)
    assert spf_bill_windows("TBILLB", 1) == (11, 22)  # next January-December
    assert spf_bill_windows("TBILLB", 4) == (2, 13)  # survey end-November
    assert spf_bill_windows("TBILLD", 2) == (32, 43)
    assert spf_bill_windows("BILL10", 1) == (1, 120)
    for bad in ("TBILL2", "TBILLA", "CPI10"):
        with pytest.raises(ValueError):
            spf_bill_windows(bad, 1)


def test_discount_to_continuous():
    from nss_engine.data import discount_to_continuous

    np.testing.assert_allclose(discount_to_continuous([5.0])[0], 5.102, atol=1e-3)
    assert discount_to_continuous(0.0) == 0.0
    # always above the discount rate
    r = np.linspace(0.1, 15, 20)
    assert np.all(discount_to_continuous(r) > r)


SPF_WIDE = pd.DataFrame(
    {
        "YEAR": [1992, 1992, 1993],
        "QUARTER": [1, 2, 1],
        "TBILL1": [4.0, 3.9, 3.0],
        "TBILL3": [4.1, 3.8, 3.2],
        "TBILL6": [4.6, 4.2, 3.9],
        "TBILLB": [5.0, 4.8, np.nan],
        "BILL10": [5.5, np.nan, 5.0],
    }
)


def test_parse_spf_bill_forecasts():
    from nss_engine.data import discount_to_continuous, parse_spf_bill_forecasts

    out = parse_spf_bill_forecasts(SPF_WIDE)
    assert list(out.columns) == ["date", "series", "start", "end", "value", "quoted"]
    assert len(out) == 3 + 3 + 2 + 2  # TBILL3, TBILL6, TBILLB, BILL10 where present
    first = out[out["date"] == pd.Timestamp("1992-02-29")]
    assert set(first["series"]) == {"TBILL3", "TBILL6", "TBILLB", "BILL10"}
    row = out[(out["series"] == "TBILL3") & (out["date"] == pd.Timestamp("1992-05-31"))].iloc[0]
    assert (row["start"], row["end"], row["quoted"]) == (2, 4, 3.8)
    assert row["value"] == pytest.approx(discount_to_continuous(3.8))
    assert "TBILL1" not in set(out["series"])  # past quarters are not forecasts
    with pytest.raises(DataError):
        parse_spf_bill_forecasts(SPF_WIDE.drop(columns="YEAR"))


def test_load_spf_bill_forecasts_downloads_caches_and_falls_back(tmp_path, monkeypatch):
    pytest.importorskip("openpyxl")
    import io

    def xlsx(df):
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        return buf.getvalue()

    files = {
        "median_tbill_level": xlsx(SPF_WIDE.drop(columns="BILL10")),
        "median_bill10_level": xlsx(SPF_WIDE[["YEAR", "QUARTER", "BILL10"]]),
    }
    calls = []

    def fake(url, **kwargs):
        calls.append(url)
        return files[url.rsplit("/", 1)[1].removesuffix(".xlsx")]

    monkeypatch.setattr(data, "_http_get_bytes", fake)
    out = data.load_spf_bill_forecasts(cache_dir=tmp_path)
    assert len(calls) == 2 and len(out) == 10
    again = data.load_spf_bill_forecasts(cache_dir=tmp_path, start="1993-01-01")
    assert len(calls) == 2 and set(again["date"]) == {pd.Timestamp("1993-02-28")}

    def boom(url, **kwargs):
        raise DataError("offline")

    monkeypatch.setattr(data, "_http_get_bytes", boom)
    with pytest.warns(RuntimeWarning, match="stale"):
        stale = data.load_spf_bill_forecasts(cache_dir=tmp_path, refresh=True)
    pd.testing.assert_frame_equal(stale, out)
    with pytest.raises(DataError):
        data.load_spf_bill_forecasts(cache_dir=tmp_path / "empty")
    with pytest.raises(ValueError):
        data.load_spf_bill_forecasts(statistic="mode")
    monkeypatch.setattr(data, "_http_get_bytes", lambda url, **k: b"not a spreadsheet")
    with pytest.raises(DataError):
        data.load_spf_bill_forecasts(cache_dir=tmp_path / "bad")
