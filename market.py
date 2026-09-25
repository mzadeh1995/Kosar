# ==============================================================================
# 📈 market.py
# ------------------------------------------------------------------------------
# Market screener and correlation orchestration.
# v49.1: Added "Calibration" and Patch
# Binance provider-specific REST/session/rate-limit logic was split into provider.py.
# ==============================================================================

from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

import pandas as pd
import pandas_ta as ta

from config import CONFIG, TZ, SCHEMA_VERSION
from hmm import infer_hmm_regime
from telemetry_utils import logger, log_stats, write_json, append_jsonl
from portfolio import pm
from provider import binance_download_single, get_reliable_price, build_microstructure_features

# 📈 SCREENER (DETAILS -> ./log/screener_runs + latest ./log/screener_details.json)
# -------------------------
def sell_filter_active() -> bool:
    return bool(CONFIG.get("SELL_FILTER_ENABLED", False)) and (CONFIG.get("TRADING_MODE") == "spot")


def find_recent_swing_low_high(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())


def _coerce_micro_history_number(value):
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    return None


def _micro_history_ts(micro: dict) -> str:
    raw_ts = _coerce_micro_history_number(micro.get("vpin_last_trade_ts"))
    if raw_ts is not None and raw_ts > 0:
        seconds = float(raw_ts) / 1000.0 if raw_ts > 10_000_000_000 else float(raw_ts)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def _record_microstructure_history(symbol: str, micro: dict, config: dict | None = None) -> bool:
    cfg = CONFIG if config is None else config
    if not bool(cfg.get("MICROSTRUCTURE_FEATURES_ENABLED", False)):
        return False
    if not isinstance(micro, dict) or not micro:
        return False

    numeric_micro = {}
    for key, value in micro.items():
        coerced = _coerce_micro_history_number(value)
        if coerced is not None:
            numeric_micro[str(key)] = coerced

    if not numeric_micro:
        return False

    path = cfg.get("MICROSTRUCTURE_HISTORY_FILE")
    if not path:
        logger.warning("⚠️ MICROSTRUCTURE_HISTORY_FILE is empty; skipping microstructure history record.")
        return False

    event = {
        "schema_version": SCHEMA_VERSION,
        "ts": _micro_history_ts(micro),
        "symbol": str(symbol),
        **numeric_micro,
    }

    try:
        append_jsonl(path, event)
        return True
    except Exception as e:
        logger.warning(f"⚠️ Microstructure history write failed for {symbol}: {type(e).__name__}: {e}")
        return False


async def screen_candidates(universe: List[str], cycle_id: str) -> List[dict]:
    logger.info(f"🕵️‍♂️ Screening {len(universe)} assets...")

    funnel = {
        "universe_total": len(universe),
        "skip_open_position": 0,
        "download_error": 0,
        "df_empty_or_none": 0,
        "df_too_short_<250": 0,
        "fail_ema200": 0,
        "fail_sell_filter": 0,  # ✅ new
        "fail_adx": 0,
        "fail_rsi": 0,
        "fail_rvol_sma": 0,
        "fail_rvol_last": 0,
        "fail_rvol_threshold": 0,
        "fail_atr": 0,
        "passed_screener": 0,
    }

    per_symbol: List[dict] = []
    candidates: List[dict] = []

    for sym in universe:
        sym_rec = {"cycle_id": cycle_id, "symbol": sym, "status": "unknown", "fail_reason": None, "metrics": {}}

        if pm.is_position_open(sym):
            funnel["skip_open_position"] += 1
            sym_rec["status"] = "skipped_open_position"
            per_symbol.append(sym_rec)
            continue

        try:
            df = await binance_download_single(sym, "2mo", CONFIG["TIMEFRAME"])
            await asyncio.sleep(CONFIG["DOWNLOAD_DELAY_SECONDS"])

            if df is None or df.empty:
                funnel["df_empty_or_none"] += 1
                sym_rec["status"] = "fail"
                sym_rec["fail_reason"] = "df_empty_or_none"
                per_symbol.append(sym_rec)
                continue

            # Drop incomplete last bar if volume invalid
            try:
                if "Volume" in df.columns and len(df) >= 2:
                    last_vol = df["Volume"].iloc[-1]
                    if pd.isna(last_vol) or last_vol <= 0:
                        df = df.iloc[:-1]
            except Exception:
                pass

            if df is None or df.empty:
                funnel["df_empty_or_none"] += 1
                sym_rec["status"] = "fail"
                sym_rec["fail_reason"] = "df_empty_or_none_after_trim"
                per_symbol.append(sym_rec)
                continue

            if len(df) < 250:
                funnel["df_too_short_<250"] += 1
                sym_rec["status"] = "fail"
                sym_rec["fail_reason"] = "df_too_short_<250"
                sym_rec["metrics"]["rows"] = int(len(df))
                per_symbol.append(sym_rec)
                continue

            # EMA metrics (kept for market_data / telemetry; not used as a hard filter)
            ema200 = ta.ema(df["Close"], 200)
            ema_last = ema200.iloc[-1] if ema200 is not None and not ema200.empty else float("nan")
            ema_len = int(CONFIG.get("SELL_FILTER_EMA_LENGTH", 50))
            ema_fast = ta.ema(df["Close"], ema_len)
            close_last = float(df["Close"].iloc[-1])

            # SELL FILTER is intentionally excluded from screener decision gating.

            # ADX (metric-only, no screener gating)
            adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
            adx_raw = adx.get("ADX_14", pd.Series([float("nan")])).iloc[-1] if adx is not None and not adx.empty else float("nan")
            adx_value = 0.0 if pd.isna(adx_raw) else float(adx_raw)

            # RSI (metric-only, no screener gating)
            rsi_raw = ta.rsi(df["Close"], 14).iloc[-1]
            rsi_val = 50.0 if pd.isna(rsi_raw) else float(rsi_raw)

            # RVOL (metric-only, no screener gating)
            sma_vol = ta.sma(df["Volume"], 20).iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            if pd.isna(sma_vol) or float(sma_vol) <= 1e-9 or pd.isna(last_vol) or float(last_vol) <= 0:
                rvol = 0.0
            else:
                rvol = float(last_vol) / float(sma_vol)

            # ATR (metric-only, no screener gating)
            atr_raw = ta.atr(df["High"], df["Low"], df["Close"], 14).iloc[-1]
            if pd.isna(atr_raw) or float(atr_raw) <= 1e-9:
                try:
                    hl = pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")
                    atr_fallback = float(hl.tail(14).mean())
                    if pd.isna(atr_fallback) or atr_fallback <= 1e-9:
                        atr_fallback = max(close_last * 0.005, 1e-6)
                    atr_val = float(atr_fallback)
                except Exception:
                    atr_val = max(close_last * 0.005, 1e-6)
            else:
                atr_val = float(atr_raw)

            market_data = {
                "rsi": round(float(rsi_val), 2),
                "rvol": round(float(rvol), 2),
                "adx": round(float(adx_value), 2),
                "atr": float(atr_val),
                "ema50": None if ema_fast is None or ema_fast.empty or pd.isna(ema_fast.iloc[-1]) else round(float(ema_fast.iloc[-1]), 6),
                "ema200": None if pd.isna(ema_last) else round(float(ema_last), 6),
            }

            candidates.append({"symbol": sym, "df": df, "rvol": rvol, "market_data": market_data})

            funnel["passed_screener"] += 1
            sym_rec["status"] = "pass"
            sym_rec["metrics"] = market_data
            per_symbol.append(sym_rec)

        except Exception as e:
            funnel["download_error"] += 1
            sym_rec["status"] = "fail"
            sym_rec["fail_reason"] = f"exception: {type(e).__name__}"
            per_symbol.append(sym_rec)

    pre_trunc = len(candidates)
    candidates.sort(key=lambda x: x["rvol"], reverse=True)
    topN = candidates[:CONFIG["TOP_CANDIDATES_COUNT"]]

    # HMM runs only on shortlisted candidates (topN), not on the full universe.
    for cand in topN:
        sym = cand["symbol"]
        df_cand = cand["df"]
        market_data = cand["market_data"]
        try:
            hmm_source_df = df_cand
            if CONFIG.get("HMM_SEPARATE_DOWNLOAD", True):
                hmm_source_df = await binance_download_single(
                    sym,
                    CONFIG.get("HMM_LOOKBACK_PERIOD", "1y"),
                    CONFIG["TIMEFRAME"],
                )
                await asyncio.sleep(CONFIG["DOWNLOAD_DELAY_SECONDS"])

            hmm_result = infer_hmm_regime(hmm_source_df, CONFIG, symbol=sym)
            if not bool(hmm_result.get("hmm_ok", False)):
                logger.warning(f"⚠️ HMM inference issue for {sym}: {hmm_result.get('hmm_reason')}")

            market_data.update({
                "hmm_state": hmm_result.get("hmm_state"),
                "hmm_regime": hmm_result.get("hmm_regime"),
                "hmm_confidence": hmm_result.get("hmm_confidence"),
                "hmm_bull_prob": hmm_result.get("hmm_bull_prob"),
                "hmm_neutral_prob": hmm_result.get("hmm_neutral_prob"),
                "hmm_bear_prob": hmm_result.get("hmm_bear_prob"),
                "hmm_policy": hmm_result.get("hmm_policy"),
                "hmm_reason": hmm_result.get("hmm_reason"),
                "hmm_backend": hmm_result.get("hmm_backend"),
                "hmm_anchor_method": hmm_result.get("hmm_anchor_method"),
                "hmm_persistence": hmm_result.get("hmm_persistence"),
                "hmm_switch_margin": hmm_result.get("hmm_switch_margin"),
                "hmm_prev_regime": hmm_result.get("hmm_prev_regime"),
                "hmm_regime_changed": hmm_result.get("hmm_regime_changed"),
                "hmm_regime_age_bars": hmm_result.get("hmm_regime_age_bars"),
                "hmm_filtered_confidence": hmm_result.get("hmm_filtered_confidence"),
                "hmm_feature_mode": hmm_result.get("hmm_feature_mode"),
                "hmm_feature_rows": hmm_result.get("hmm_feature_rows"),
                "hmm_feature_columns": hmm_result.get("hmm_feature_columns"),
                "hmm_trend_col": hmm_result.get("hmm_trend_col"),
                "hmm_vol_col": hmm_result.get("hmm_vol_col"),
                "hmm_range_col": hmm_result.get("hmm_range_col"),
                "hmm_fracdiff_method": hmm_result.get("hmm_fracdiff_method"),
                "hmm_fracdiff_d_used": hmm_result.get("hmm_fracdiff_d_used"),
                "hmm_frac_diff_d": hmm_result.get("hmm_frac_diff_d"),
                "hmm_fracdiff_window": hmm_result.get("hmm_fracdiff_window"),
                "hmm_fracdiff_threshold": hmm_result.get("hmm_fracdiff_threshold"),
                "hmm_frac_diff_threshold": hmm_result.get("hmm_frac_diff_threshold"),
                "hmm_fracdiff_use_adf": hmm_result.get("hmm_fracdiff_use_adf"),
                "hmm_fracdiff_adf_selected": hmm_result.get("hmm_fracdiff_adf_selected"),
                "hmm_fracdiff_adf_pvalue": hmm_result.get("hmm_fracdiff_adf_pvalue"),
                "hmm_fracdiff_corr_with_base": hmm_result.get("hmm_fracdiff_corr_with_base"),
            })
        except Exception as e:
            logger.warning(f"⚠️ HMM enrichment failed for {sym}: {type(e).__name__}: {e}")

        try:
            micro = await build_microstructure_features(sym, CONFIG)
            if isinstance(micro, dict):
                market_data.update(micro)
                _record_microstructure_history(sym, micro, CONFIG)
        except Exception as e:
            logger.warning(f"⚠️ Microstructure enrichment failed for {sym}: {type(e).__name__}: {e}")

        for rec in per_symbol:
            if rec.get("status") == "pass" and rec.get("symbol") == sym:
                rec["metrics"] = dict(market_data)
                break

    trunc = {
        "candidates_pre_truncation": pre_trunc,
        "returned_topN": len(topN),
        "truncation_dropped": max(0, pre_trunc - len(topN)),
        "TOP_CANDIDATES_COUNT": CONFIG["TOP_CANDIDATES_COUNT"],
    }

    screener_summary = {
        "universe_total": len(universe),
        "selected_for_senate": len(topN),
    }
    log_stats("Screener Summary", screener_summary)

    screener_report = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "ts": datetime.now(TZ).isoformat(),
        "funnel": funnel,
        "truncation": trunc,
        "per_symbol": per_symbol,
        "top_candidates": [{"symbol": c["symbol"], "rvol": c["rvol"], "market_data": c["market_data"]} for c in topN],
    }

    if CONFIG["TELEMETRY_SCREENER_DETAILS"]:
        write_json(CONFIG["SCREENER_LATEST_FILE"], screener_report)
        hist_path = os.path.join(CONFIG["SCREENER_RUNS_DIR"], f"{cycle_id}.json")
        write_json(hist_path, screener_report)

    if CONFIG["TELEMETRY_SCREENER_EVENTS"]:
        append_jsonl(CONFIG["SCREENER_EVENTS_FILE"], {"schema_version": SCHEMA_VERSION, "ts": screener_report["ts"], "cycle_id": cycle_id, "type": "screener_summary", "funnel": funnel, "truncation": trunc})

    return topN


async def _close_series_for_correlation(symbol: str) -> Optional[pd.Series]:
    df = await binance_download_single(symbol, "1mo", CONFIG["TIMEFRAME"])
    if df is None or df.empty or "Close" not in df:
        return None

    series = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if series.empty:
        return None

    series.name = symbol
    return series


# 🔗 CORRELATION FILTER
# -------------------------
async def is_highly_correlated(new_symbol: str, open_positions: dict, threshold: float) -> Tuple[bool, bool]:
    if not open_positions:
        return False, False

    open_symbols = list(open_positions.keys())
    symbols_to_check = open_symbols + [new_symbol]

    try:
        tasks = [_close_series_for_correlation(sym) for sym in symbols_to_check]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        close_map: Dict[str, pd.Series] = {}
        for sym, res in zip(symbols_to_check, results):
            if isinstance(res, Exception) or res is None:
                continue
            close_map[sym] = res

        missing = [sym for sym in symbols_to_check if sym not in close_map]
        if missing:
            logger.warning(
                f"Correlation check failed for {new_symbol}: Missing close series for {missing}. "
                "Blocking trade for safety."
            )
            return True, False

        close_prices = pd.concat([close_map[s] for s in symbols_to_check], axis=1, join="inner")

        if close_prices is None or close_prices.empty or len(getattr(close_prices, "columns", [])) < 2:
            logger.warning(f"Correlation check failed for {new_symbol}: Not enough data. Blocking trade for safety.")
            return True, False

        returns = close_prices.pct_change()
        corr = returns.corr(min_periods=100)

        for open_sym in open_symbols:
            if new_symbol not in corr.index or open_sym not in corr.columns:
                logger.warning(
                    f"Correlation check failed for {new_symbol}: Not enough overlap with {open_sym}. "
                    "Blocking trade for safety."
                )
                return True, False

            c = corr.loc[new_symbol, open_sym]
            if pd.isna(c):
                logger.warning(f"🛡️ Correlation is NaN between {new_symbol} and {open_sym}. Blocking trade for safety.")
                return True, True
            if float(c) > threshold:
                logger.warning(f"🛡️ Correlation Filter: {new_symbol} highly correlated with {open_sym} (Corr: {float(c):.2f}). Blocking trade.")
                return True, False

    except Exception as e:
        logger.error(f"Correlation check failed: {e}. Blocking trade for safety.")
        return True, False

    return False, False
