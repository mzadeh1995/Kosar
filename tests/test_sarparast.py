# ==============================================================================
# tests/test_sarparast.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import asyncio
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

import Sarparast


def _df_from_closes(closes) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=idx)


@pytest.fixture(autouse=True)
def _disable_sarparast_file_telemetry(monkeypatch):
    monkeypatch.setitem(Sarparast.CONFIG, "TELEMETRY_SENATE_EVENTS", False)
    monkeypatch.setattr(Sarparast, "append_jsonl", lambda *args, **kwargs: None)


def _final(deliberation: dict) -> dict:
    return deliberation["final_analyses"]["Sarparast"]


def test_sarparast_schema_and_determinism():
    df = _df_from_closes(np.linspace(100.0, 130.0, 30))

    first = Sarparast.sarparast_decide(None, "cycle-1", "BTCUSDT", df, {}, {})
    second = Sarparast.sarparast_decide(None, "cycle-1", "BTCUSDT", df, {}, {})

    assert first == second
    decision, deliberation = first
    assert decision == "BUY"
    assert set(deliberation.keys()) == {"decision_engine", "initial_analyses", "final_analyses"}
    assert set(deliberation["initial_analyses"].keys()) == {"Sarparast"}
    assert set(deliberation["final_analyses"].keys()) == {"Sarparast"}

    final = _final(deliberation)
    assert set(final.keys()) == {"final_vote", "final_confidence", "changed_opinion", "reason_for_final_decision"}
    assert final["final_vote"] in {"BUY", "SELL", "HOLD"}
    assert final["final_confidence"] == 68
    assert isinstance(final["final_confidence"], int)
    assert 0 <= final["final_confidence"] <= 100
    assert final["changed_opinion"] is False


@pytest.mark.parametrize(
    ("closes", "expected_vote", "expected_confidence", "reason_text"),
    [
        (np.linspace(100.0, 130.0, 30), "BUY", 68, "mom="),
        (np.linspace(130.0, 100.0, 30), "HOLD", 68, "mom="),
        (np.full(30, 100.0), "HOLD", 68, "mom="),
        ([100.0] * 25, "HOLD", 60, "insufficient data"),
        ([100.0] * 24, "HOLD", 60, "insufficient data"),
    ],
)
def test_sarparast_rules_and_never_sell(closes, expected_vote, expected_confidence, reason_text):
    decision, deliberation = Sarparast.sarparast_decide(None, "cycle-1", "ETHUSDT", _df_from_closes(closes), {}, {})
    final = _final(deliberation)

    assert decision == expected_vote
    assert decision != "SELL"
    assert final["final_vote"] == expected_vote
    assert final["final_vote"] != "SELL"
    assert final["final_confidence"] == expected_confidence
    assert reason_text in final["reason_for_final_decision"]


def test_sarparast_ignores_latest_forming_candle_for_momentum():
    closed_closes = np.linspace(130.0, 100.0, 25)
    df = _df_from_closes([*closed_closes, 1000.0])

    decision, deliberation = Sarparast.sarparast_decide(None, "cycle-1", "ETHUSDT", df, {}, {})
    final = _final(deliberation)

    assert decision == "HOLD"
    assert final["final_confidence"] == 68
    assert "mom=" in final["reason_for_final_decision"]
    assert "<= 0" in final["reason_for_final_decision"]


def test_sarparast_missing_close_is_fail_safe_hold():
    decision, deliberation = Sarparast.sarparast_decide(None, "cycle-1", "XRPUSDT", pd.DataFrame({"Open": [1, 2]}), {}, {})
    final = _final(deliberation)

    assert decision == "HOLD"
    assert final["final_confidence"] == 60
    assert "insufficient data" in final["reason_for_final_decision"]


def test_sarparast_branch_does_not_touch_or_import_senate(monkeypatch):
    sys.modules.pop("senate", None)
    main = importlib.import_module("main")
    monkeypatch.setitem(main.CONFIG, "DECISION_ENGINE", "sarparast")

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("senate branch should not be called")

    monkeypatch.setattr(main, "_convene_senate_decision", fail_if_called)
    monkeypatch.setattr(main, "_sarparast_decision", lambda *args, **kwargs: ("HOLD", {"final_analyses": {}}))

    result = asyncio.run(main.run_decision_engine(None, "cycle-1", "BTCUSDT", _df_from_closes([100.0] * 30), {}, {}))

    assert result[0] == "HOLD"
    assert "senate" not in sys.modules


def test_senate_branch_calls_existing_path_with_mock(monkeypatch):
    main = importlib.import_module("main")
    monkeypatch.setitem(main.CONFIG, "DECISION_ENGINE", "senate")
    called = {"value": False}

    async def fake_senate(*args, **kwargs):
        called["value"] = True
        return "HOLD", {"final_analyses": {"Mock": {"final_vote": "HOLD", "final_confidence": 70}}}

    monkeypatch.setattr(main, "_convene_senate_decision", fake_senate)

    decision, deliberation = asyncio.run(main.run_decision_engine(None, "cycle-1", "BTCUSDT", _df_from_closes([100.0] * 30), {}, {}))

    assert called["value"] is True
    assert decision == "HOLD"
    assert "Mock" in deliberation["final_analyses"]


def test_invalid_decision_engine_is_readable(monkeypatch):
    main = importlib.import_module("main")
    monkeypatch.setitem(main.CONFIG, "DECISION_ENGINE", "broken")

    with pytest.raises(ValueError, match="DECISION_ENGINE.*senate.*sarparast"):
        main.get_decision_engine()
