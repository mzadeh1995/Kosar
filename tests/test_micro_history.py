# ==============================================================================
# tests/test_micro_history.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

import market
from config import SCHEMA_VERSION


def _cfg(path):
    return {
        "MICROSTRUCTURE_FEATURES_ENABLED": True,
        "MICROSTRUCTURE_HISTORY_FILE": path,
    }


def test_micro_history_records_finite_numeric_values(monkeypatch, tmp_path):
    calls = []
    path = str(tmp_path / "micro.jsonl")

    monkeypatch.setattr(market, "append_jsonl", lambda p, item: calls.append((p, item)))

    micro = {
        "ofi_l1": np.float64(1.25),
        "vpin_bucket_count": np.int64(4),
        "vpin_last_trade_ts": np.int64(1_700_000_000_000),
        "bad_nan": np.nan,
        "bad_inf": np.inf,
        "flag": True,
        "reason": "ok",
        "none_value": None,
    }

    assert market._record_microstructure_history("BTC-USD", micro, _cfg(path)) is True

    assert len(calls) == 1
    written_path, event = calls[0]
    assert written_path == path
    assert event["schema_version"] == SCHEMA_VERSION
    assert event["symbol"] == "BTC-USD"
    assert event["ts"] == datetime.fromtimestamp(1_700_000_000_000 / 1000.0, timezone.utc).isoformat()
    assert datetime.fromisoformat(event["ts"]).tzinfo is not None
    assert event["ofi_l1"] == 1.25
    assert type(event["ofi_l1"]) is float
    assert event["vpin_bucket_count"] == 4
    assert type(event["vpin_bucket_count"]) is int
    assert event["vpin_last_trade_ts"] == 1_700_000_000_000
    assert "bad_nan" not in event
    assert "bad_inf" not in event
    assert "flag" not in event
    assert "reason" not in event
    assert "none_value" not in event


def test_micro_history_skips_empty_none_and_all_nan(monkeypatch, tmp_path):
    calls = []
    path = str(tmp_path / "micro.jsonl")
    monkeypatch.setattr(market, "append_jsonl", lambda p, item: calls.append((p, item)))

    assert market._record_microstructure_history("BTC-USD", {}, _cfg(path)) is False
    assert market._record_microstructure_history("BTC-USD", None, _cfg(path)) is False
    assert market._record_microstructure_history("BTC-USD", {"a": np.nan, "b": np.inf, "flag": False}, _cfg(path)) is False

    assert calls == []


def test_micro_history_missing_path_skips_without_exception(monkeypatch):
    calls = []
    warnings = []
    monkeypatch.setattr(market, "append_jsonl", lambda p, item: calls.append((p, item)))
    monkeypatch.setattr(market.logger, "warning", lambda msg: warnings.append(msg))

    assert market._record_microstructure_history("BTC-USD", {"ofi_l1": 1.0}, _cfg(None)) is False
    assert market._record_microstructure_history("BTC-USD", {"ofi_l1": 1.0}, _cfg("")) is False

    assert calls == []
    assert len(warnings) == 2


def test_micro_history_write_error_is_fail_open(monkeypatch, tmp_path):
    warnings = []

    def raise_on_write(path, item):
        raise RuntimeError("disk is unhappy")

    monkeypatch.setattr(market, "append_jsonl", raise_on_write)
    monkeypatch.setattr(market.logger, "warning", lambda msg: warnings.append(msg))

    result = market._record_microstructure_history("ETH-USD", {"vpin": 0.42}, _cfg(str(tmp_path / "micro.jsonl")))

    assert result is False
    assert len(warnings) == 1
    assert "Microstructure history write failed" in warnings[0]


def test_micro_history_respects_disabled_feature_flag(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(market, "append_jsonl", lambda p, item: calls.append((p, item)))
    cfg = {
        "MICROSTRUCTURE_FEATURES_ENABLED": False,
        "MICROSTRUCTURE_HISTORY_FILE": str(tmp_path / "micro.jsonl"),
    }

    assert market._record_microstructure_history("BTC-USD", {"ofi_l1": 1.0}, cfg) is False
    assert calls == []
