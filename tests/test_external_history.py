# ==============================================================================
# tests/test_external_history.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

import binance_vision
import external_history


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.csv", csv_text)
    return buf.getvalue()


def _minute_klines(start: str, periods: int, quote=100.0, buy_ratio=0.5) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC", name="OpenTime")
    if callable(quote):
        quote_values = np.asarray([quote(i, ts) for i, ts in enumerate(idx)], dtype=float)
    else:
        quote_values = np.full(periods, float(quote))
    if callable(buy_ratio):
        ratios = np.asarray([buy_ratio(i, ts) for i, ts in enumerate(idx)], dtype=float)
    else:
        ratios = np.full(periods, float(buy_ratio))
    ratios = np.clip(ratios, 0.0, 1.0)
    close = np.full(periods, 100.0)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": quote_values / close,
            "QuoteVolume": quote_values,
            "Trades": np.ones(periods),
            "TakerBuyBase": (quote_values * ratios) / close,
            "TakerBuyQuote": quote_values * ratios,
        },
        index=idx,
    )


def _write_micro_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_vpin_bucket_splitting_and_drops_incomplete_tail():
    prepared = pd.DataFrame(
        {
            "quote_volume": [60.0, 80.0, 60.0, 30.0],
            "buy_quote": [60.0, 0.0, 60.0, 30.0],
        }
    )

    vpin = external_history.compute_vpin_from_minute_bars(prepared, bucket_capacity=100.0, bucket_count=2)

    assert vpin == pytest.approx(0.2)


def test_vpin_reconstruction_is_causal_for_minutes_after_t():
    klines = _minute_klines("2026-01-01 00:00", 6, quote=100.0, buy_ratio=0.5)
    klines.iloc[4, klines.columns.get_loc("TakerBuyQuote")] = 100.0

    at_t = external_history.reconstruct_vpin_at(
        klines,
        "2026-01-01 00:04:00+00:00",
        bucket_capacity=100.0,
        require_full_window=False,
        bucket_count=4,
    )
    after_t = external_history.reconstruct_vpin_at(
        klines,
        "2026-01-01 00:05:00+00:00",
        bucket_capacity=100.0,
        require_full_window=False,
        bucket_count=4,
    )

    assert at_t["vpin"] == pytest.approx(0.0)
    assert after_t["vpin"] > at_t["vpin"]


def test_dynamic_capacity_is_causal_and_requires_seven_days():
    base = _minute_klines("2026-01-01 00:00", 7 * 24 * 60, quote=100.0, buy_ratio=0.5)
    with_future = pd.concat([base, _minute_klines("2026-01-08 00:00", 1, quote=100000.0, buy_ratio=1.0)])

    t = pd.Timestamp("2026-01-08 00:00", tz="UTC")
    base_rec = external_history.reconstruct_vpin_at(base, t)
    future_rec = external_history.reconstruct_vpin_at(with_future, t)
    too_early = external_history.reconstruct_vpin_at(base, pd.Timestamp("2026-01-07 23:00", tz="UTC"))

    assert base_rec["bucket_capacity"] == pytest.approx(300.0)
    assert future_rec["bucket_capacity"] == pytest.approx(base_rec["bucket_capacity"])
    assert future_rec["vpin"] == pytest.approx(base_rec["vpin"])
    assert np.isnan(too_early["vpin"])


def test_reconstruction_is_independent_of_input_origin_and_mid_hour_timestamp():
    klines = _minute_klines(
        "2026-01-01 00:00",
        30 * 24 * 60,
        quote=lambda i, ts: 100.0 + float((i // 60) % 5),
        buy_ratio=lambda i, ts: 0.2 + 0.6 * ((i // 180) % 2),
    )
    t = pd.Timestamp("2026-01-25 12:30", tz="UTC")
    long_slice = klines.loc[klines.index >= t - pd.Timedelta(days=30)]
    short_slice = klines.loc[klines.index >= t - pd.Timedelta(days=10)]

    long_rec = external_history.reconstruct_vpin_at(long_slice, t)
    short_rec = external_history.reconstruct_vpin_at(short_slice, t)

    assert long_rec["vpin"] == pytest.approx(short_rec["vpin"])

    future = klines.copy()
    future.loc[pd.Timestamp("2026-01-25 12:30", tz="UTC"), "QuoteVolume"] = 1_000_000.0
    future.loc[pd.Timestamp("2026-01-25 12:30", tz="UTC"), "TakerBuyQuote"] = 1_000_000.0
    unchanged = external_history.reconstruct_vpin_at(future, t)
    assert unchanged["vpin"] == pytest.approx(long_rec["vpin"])

    small = _minute_klines("2026-01-01", 10, quote=10.0, buy_ratio=0.5)
    assert np.isnan(external_history.compute_vpin_from_minute_bars(small, bucket_capacity=100.0, bucket_count=20))


def test_funding_z_uses_previous_90_records_and_flags_current_record():
    idx = pd.date_range("2026-01-01", periods=95, freq="8h", tz="UTC")
    baseline = np.linspace(-0.001, 0.001, 90)
    values = np.r_[baseline, 0.01, np.zeros(4)]
    raw = pd.DataFrame({"calc_time": (idx.view("int64") // 1_000_000).astype(np.int64), "last_funding_rate": values})

    out = external_history.normalize_funding_frame(raw)
    expected_z = (values[90] - baseline.mean()) / baseline.std(ddof=0)

    assert out.iloc[:90]["funding_z"].isna().all()
    assert out.iloc[:90]["funding_extreme_pos"].eq(0).all()
    assert out.iloc[90]["funding_z"] == pytest.approx(expected_z)
    assert out.iloc[90]["funding_extreme_pos"] == 1
    assert out.iloc[91]["funding_extreme_pos"] == 0


def test_validation_report_corr_dedup_low_n_missing_and_constant(monkeypatch, tmp_path):
    klines = _minute_klines(
        "2026-01-01 00:00",
        10 * 24 * 60,
        quote=lambda i, ts: 100.0 + 20.0 * np.sin(i / 97.0),
        buy_ratio=lambda i, ts: 0.5 + 0.4 * np.sin(i / 331.0),
    )
    monkeypatch.setattr(external_history.binance_vision, "download_klines", lambda *args, **kwargs: klines.copy())
    times = pd.date_range("2026-01-08 00:00", periods=60, freq="1h", tz="UTC")
    rows = []
    for ts in times:
        vpin = external_history.reconstruct_vpin_at(klines, ts)["vpin"]
        rows.append({"schema_version": 2, "ts": ts.isoformat(), "symbol": "BTC-USD", "vpin": vpin})
    rows.append(dict(rows[-1]))
    path = tmp_path / "micro.jsonl"
    _write_micro_history(path, rows)

    report = external_history.validate_vpin_reconstruction(
        ["BTC-USD"],
        micro_history_file=path,
        data_dir=str(tmp_path),
        output_dir=tmp_path,
        symbol_map={"BTC-USD": "BTCUSDT"},
        material_differences=[],
    )

    item = report["symbols"]["BTC-USD"]
    assert report["deduplicated_rows"] == 1
    assert item["n_common"] == 60
    assert item["pearson"] == pytest.approx(1.0)
    assert item["spearman"] == pytest.approx(1.0)
    assert item["verdict"] == "approved"

    low_path = tmp_path / "micro_low.jsonl"
    _write_micro_history(low_path, rows[:10])
    low_report = external_history.validate_vpin_reconstruction(
        ["BTC-USD"],
        micro_history_file=low_path,
        output_dir=tmp_path,
        symbol_map={"BTC-USD": "BTCUSDT"},
        material_differences=[],
    )
    assert low_report["symbols"]["BTC-USD"]["verdict"] == "insufficient_overlap"

    missing = external_history.validate_vpin_reconstruction(["BTC-USD"], micro_history_file=tmp_path / "missing.jsonl", output_dir=tmp_path)
    assert missing["status"] == "skipped"

    constant_path = tmp_path / "micro_constant.jsonl"
    constant_rows = [{**row, "vpin": 0.5} for row in rows[:-1]]
    _write_micro_history(constant_path, constant_rows)
    constant_report = external_history.validate_vpin_reconstruction(
        ["BTC-USD"],
        micro_history_file=constant_path,
        output_dir=tmp_path,
        symbol_map={"BTC-USD": "BTCUSDT"},
        material_differences=[],
    )
    assert constant_report["symbols"]["BTC-USD"]["verdict"] == "insufficient_variation"


def test_funding_downloader_skips_missing_month(monkeypatch, tmp_path):
    csv_text = "calc_time,symbol,last_funding_rate\n1767225600000,BTCUSDT,0.0001\n"
    payload = _zip_bytes(csv_text)
    calls = []

    def fake_urlopen(url, timeout=30):
        calls.append(url)
        if "2026-01" in url:
            return _FakeResponse(payload)
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(binance_vision, "urlopen", fake_urlopen)

    raw, skipped = binance_vision.download_funding_rates(
        "BTCUSDT",
        "2026-01-01",
        "2026-02-28",
        data_dir=str(tmp_path),
        return_skipped=True,
    )

    assert raw.shape[0] == 1
    assert skipped == ["2026-02"]
    assert len(calls) == 2
