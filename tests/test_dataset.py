# ==============================================================================
# tests/test_dataset.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import binance_vision
import dataset
from TripleBarrier import compute_causal_volatility


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.payload


def _synthetic_klines(rows: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    rng = np.random.default_rng(41)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, rows)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(rows, 1000.0)
    volume[200] = 0.0
    taker_buy = volume * 0.55
    taker_buy[200] = 123.0
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "QuoteVolume": volume * close,
            "Trades": np.arange(rows) + 100,
            "TakerBuyBase": taker_buy,
            "TakerBuyQuote": taker_buy * close,
        },
        index=idx,
    ).rename_axis("OpenTime")


def _cfg(tmp_path: Path) -> dict:
    return {
        "BINANCE_SYMBOL_MAP": {"BTC-USD": "BTCUSDT"},
        "DATASET_DATA_DIR": str(tmp_path / "binance_vision"),
        "DATASET_OUTPUT_DIR": str(tmp_path / "datasets"),
        "DATASET_DEFAULT_MONTHS": 1,
        "DATASET_DEFAULT_TIMEFRAME": "1h",
        "TRIPLE_BARRIER_HORIZON": 12,
        "TRIPLE_BARRIER_PROFIT_MULT": 2.0,
        "TRIPLE_BARRIER_LOSS_MULT": 1.0,
        "TRIPLE_BARRIER_VOL_WINDOW": 24,
        "HMM_FEATURE_MODE": "fractional_diff",
    }


def test_dataset_builds_nonempty_with_required_columns_and_utc_index(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset.binance_vision, "download_klines", lambda *args, **kwargs: _synthetic_klines())
    monkeypatch.setattr("fractional.build_fractional_hmm_features", lambda df, config: pd.DataFrame({"fd_return": df["Close"].pct_change().fillna(0.0)}, index=df.index))
    out = dataset.build_training_dataset(["BTC-USD"], start="2026-01-01", end="2026-01-12", horizon=12, config=_cfg(tmp_path), strict=True)
    assert out.shape[0] > 0
    for col in [*dataset.BASE_FEATURE_COLUMNS, "tb_label", "tb_uniqueness"]:
        assert col in out.columns
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_strict_false_records_fractional_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset.binance_vision, "download_klines", lambda *args, **kwargs: _synthetic_klines())

    def fail_fractional(df, config):
        raise RuntimeError("forced fractional failure")

    monkeypatch.setattr("fractional.build_fractional_hmm_features", fail_fractional)
    out = dataset.build_training_dataset(["BTCUSDT"], start="2026-01-01", end="2026-01-12", horizon=12, config=_cfg(tmp_path), strict=False)
    meta = out.attrs["metadata_by_symbol"]["BTCUSDT"]
    assert meta["feature_path"] == "fallback"
    assert "forced fractional failure" in meta["fallback_reason"]


def test_strict_true_fractional_failure_is_readable(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset.binance_vision, "download_klines", lambda *args, **kwargs: _synthetic_klines())
    monkeypatch.setattr("fractional.build_fractional_hmm_features", lambda df, config: (_ for _ in ()).throw(RuntimeError("forced fractional failure")))
    with pytest.raises(RuntimeError, match="Fractional feature generation failed in strict mode"):
        dataset.build_training_dataset(["BTCUSDT"], start="2026-01-01", end="2026-01-12", horizon=12, config=_cfg(tmp_path), strict=True)


def test_zero_volume_flow_guard():
    df = _synthetic_klines()
    features = dataset.build_causal_features(df, _cfg(Path("/tmp")))
    idx = df.index[200]
    next_idx = df.index[201]
    assert pd.isna(features.loc[idx, "volume_change"])
    assert pd.isna(features.loc[next_idx, "volume_change"])
    assert features.loc[idx, "taker_buy_ratio"] == 0.5
    assert features.loc[idx, "flow_imb"] == 0.0


def _write_zip(path: Path, csv_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.csv", csv_text)


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload.csv", csv_text)
    return buf.getvalue()


def _kline_csv(open_time: int, header: bool) -> str:
    columns = ",".join(binance_vision.KLINE_COLUMNS)
    row = f"{open_time},100,101,99,100.5,10,{open_time + 1},1000,20,5,500,0"
    return f"{columns}\n{row}\n" if header else f"{row}\n"


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("micros", [False, True])
def test_parse_klines_zip_headerless_and_header_ms_and_microseconds(tmp_path, header, micros):
    ts = 1767225600000000 if micros else 1767225600000
    path = tmp_path / f"klines_{header}_{micros}.zip"
    _write_zip(path, _kline_csv(ts, header))
    out = binance_vision.read_klines_zip(path)
    assert out.shape[0] == 1
    assert out.index.name == "OpenTime"
    assert str(out.index.tz) == "UTC"
    assert float(out.iloc[0]["TakerBuyQuote"]) == 500.0


def test_download_agg_trades_uses_cached_fake_zip_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(binance_vision, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network should not be used")))
    path = tmp_path / "aggTrades" / "FAKEUSDT" / "FAKEUSDT-aggTrades-2026-01.zip"
    csv_text = "aggTradeId,price,quantity,firstTradeId,lastTradeId,timestamp,isBuyerMaker,isBestMatch\n1,100,0.5,1,1,1767225600000,true,true\n"
    _write_zip(path, csv_text)
    out = binance_vision.download_agg_trades("FAKEUSDT", "2026-01-01", "2026-01-02", data_dir=str(tmp_path))
    assert out.shape[0] == 1
    assert set(["timestamp", "price", "quantity", "isBuyerMaker"]).issubset(out.columns)


def test_corrupt_cached_zip_is_replaced_atomically_without_tmp_left(tmp_path, monkeypatch):
    path = tmp_path / "klines" / "BTCUSDT" / "1h" / "BTCUSDT-1h-2026-01.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a zip")
    payload = _zip_bytes(_kline_csv(1767225600000, header=True))
    monkeypatch.setattr(binance_vision, "urlopen", lambda *args, **kwargs: _FakeResponse(payload))

    out_path = binance_vision._download_cached("https://example.test/file.zip", path, "klines symbol=BTCUSDT interval=1h month=2026-01")
    assert out_path == path
    assert path.exists()
    assert not list(path.parent.glob("*.tmp"))
    parsed = binance_vision.read_klines_zip(path)
    assert parsed.shape[0] == 1
    assert float(parsed.iloc[0]["Close"]) == 100.5


def test_atomic_download_removes_tmp_on_invalid_download(tmp_path, monkeypatch):
    path = tmp_path / "klines" / "BTCUSDT" / "1h" / "BTCUSDT-1h-2026-01.zip"
    monkeypatch.setattr(binance_vision, "urlopen", lambda *args, **kwargs: _FakeResponse(b"broken zip payload"))

    with pytest.raises(binance_vision.BinanceVisionError):
        binance_vision._download_cached("https://example.test/file.zip", path, "klines symbol=BTCUSDT interval=1h month=2026-01")

    assert not path.exists()
    assert not list(path.parent.glob("*.tmp"))


def test_no_lookahead_features_match_prefix_calculation():
    df = _synthetic_klines()
    features = dataset.build_causal_features(df, _cfg(Path("/tmp")))
    for t in [180, 210]:
        idx = df.index[t]
        expected_vol = compute_causal_volatility(df["Close"].iloc[: t + 1], span=24).iloc[-1]
        assert abs(float(features.loc[idx, "volatility"]) - float(expected_vol)) <= 1e-12
        expected_dist = np.log(df["Close"].iloc[t] / df["Close"].iloc[max(0, t - 167) : t + 1].max())
        assert abs(float(features.loc[idx, "dist_from_max_168"]) - float(expected_dist)) <= 1e-12
