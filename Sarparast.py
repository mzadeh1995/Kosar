# ==============================================================================
# Sarparast.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Temporary deterministic caretaker decision engine. Not an API client.
# ==============================================================================

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd

from config import CONFIG, SCHEMA_VERSION, TZ
from telemetry_utils import append_jsonl, assert_feature_contract_event, logger


ENGINE_NAME = "Sarparast"
BUY = "BUY"
HOLD = "HOLD"
CONFIDENCE_NORMAL = 68
CONFIDENCE_FAILSAFE = 60


def _final_analysis(vote: str, confidence: int, reason: str) -> dict[str, Any]:
    return {
        "final_vote": vote,
        "final_confidence": int(confidence),
        "changed_opinion": False,
        "reason_for_final_decision": reason,
    }


def _initial_analysis(vote: str, confidence: int, reason: str) -> dict[str, Any]:
    return {
        "vote": vote,
        "confidence": int(confidence),
        "reason": reason,
    }


def _close_series(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["Close"], errors="coerce").dropna()


def _closed_close_series(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    # Binance kline fetch can include the currently forming candle as the final row.
    return _close_series(df.iloc[:-1])


def _emit_decision_event(cycle_id: str, symbol: str, parsed: dict[str, Any], reason: str) -> None:
    ev = {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(TZ).isoformat(),
        "cycle_id": cycle_id,
        "symbol": symbol,
        "phase": 0,
        "debate_round": False,
        "senator": ENGINE_NAME,
        "provider": "local",
        "model_type": "deterministic_momentum",
        "model_id": "sarparast_momentum_24",
        "location": None,
        "attempt": 1,
        "ok": True,
        "fail_stage": None,
        "error": None,
        "http_status": None,
        "response_preview": reason,
        "parsed": parsed,
    }
    assert_feature_contract_event(ev)
    if CONFIG.get("TELEMETRY_SENATE_EVENTS", True):
        append_jsonl(CONFIG["SENATE_EVENTS_FILE"], ev)


def sarparast_decide(
    session: Any,
    cycle_id: str,
    symbol: str,
    df: pd.DataFrame,
    market_data: dict[str, Any] | None,
    weights: dict[str, float] | None,
) -> tuple[str, dict[str, Any]]:
    """Return a deterministic spot-only vote from 24-bar closed-candle momentum.

    The final input row is discarded first because Binance kline fetch can
    include the currently forming candle.
    mom = log(last closed Close / Close from 24 closed bars before it).
    Positive momentum returns BUY with confidence 68; zero/negative momentum
    returns HOLD with confidence 68. If 25 closed candles cannot be extracted
    from the supplied payload, it returns HOLD with confidence 60 and does not
    raise. Sarparast never emits SELL and never fetches data.
    """

    del session, market_data, weights

    closes = _closed_close_series(df)
    if len(closes) < 25:
        reason = (
            "Sarparast insufficient data: fewer than 25 closed candles available "
            "after dropping the latest forming candle."
        )
        decision = HOLD
        confidence = CONFIDENCE_FAILSAFE
        mom_text = None
    else:
        latest = float(closes.iloc[-1])
        prior = float(closes.iloc[-25])
        if latest > 0.0 and prior > 0.0:
            mom = math.log(latest / prior)
            mom_text = f"{mom:.10f}"
            if mom > 0.0:
                decision = BUY
                reason = f"Sarparast momentum rule: mom={mom_text} > 0, voting BUY."
            else:
                decision = HOLD
                reason = f"Sarparast momentum rule: mom={mom_text} <= 0, voting HOLD."
            confidence = CONFIDENCE_NORMAL
        else:
            reason = "Sarparast insufficient data: non-positive close encountered."
            decision = HOLD
            confidence = CONFIDENCE_FAILSAFE
            mom_text = None

    final = _final_analysis(decision, confidence, reason)
    initial = _initial_analysis(decision, confidence, reason)
    deliberation_data = {
        "decision_engine": ENGINE_NAME,
        "initial_analyses": {ENGINE_NAME: initial},
        "final_analyses": {ENGINE_NAME: final},
    }
    _emit_decision_event(str(cycle_id), str(symbol), final, reason)
    logger.info(f"🧭 {ENGINE_NAME} decision for {symbol}: {decision} confidence={confidence} mom={mom_text}")
    return decision, deliberation_data
