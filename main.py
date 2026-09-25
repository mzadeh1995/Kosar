# ==============================================================================
# 🚀 main.py
# ------------------------------------------------------------------------------
# Orchestrator (production cycle + main loop).
# v49.1: Added "Calibration" and Patch
# core orchestration/HMM/gating/portfolio behavior preserved.
# ==============================================================================

from __future__ import annotations

import os

# Cap OpenMP threads before any native import (phase-6 prep: xgboost will join
# the live process where pomegranate already runs; they conflict over libomp on
# macOS). setdefault on purpose: server-side env override stays possible.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
from datetime import datetime, timedelta
from typing import Dict

import aiohttp

from config import VERSION, CONFIG, TZ, SCHEMA_VERSION, BASE_DIR

from telemetry_utils import (
    logger,
    now_ts,
    log_stats,
    append_json_array,
    enforce_non_negotiables_or_exit,
    check_api_keys,
)

from portfolio import (
    pm,
    compute_total_equity_and_drawdown,
    compute_trade_allocation,
)

from market import (
    get_reliable_price,
    screen_candidates,
    is_highly_correlated,
    find_recent_swing_low_high,
    sell_filter_active,
)

VALID_DECISION_ENGINES = {"senate", "sarparast"}


def get_decision_engine(config: dict | None = None) -> str:
    cfg = CONFIG if config is None else config
    engine = str(cfg.get("DECISION_ENGINE", "senate")).strip().lower()
    if engine not in VALID_DECISION_ENGINES:
        raise ValueError('Invalid DECISION_ENGINE; expected "senate" or "sarparast"')
    return engine


def load_decision_weights(config: dict | None = None) -> Dict[str, float]:
    if get_decision_engine(config) == "senate":
        from senate import load_or_recalculate_weights

        return load_or_recalculate_weights()
    return {"Sarparast": 1.0}


async def _convene_senate_decision(session, cycle_id: str, symbol: str, df, market_data: dict, weights: Dict[str, float]):
    from senate import convene_deliberative_senate

    return await convene_deliberative_senate(session, cycle_id, symbol, df, market_data, weights)


def _sarparast_decision(session, cycle_id: str, symbol: str, df, market_data: dict, weights: Dict[str, float]):
    from Sarparast import sarparast_decide

    return sarparast_decide(session, cycle_id, symbol, df, market_data, weights)


async def run_decision_engine(session, cycle_id: str, symbol: str, df, market_data: dict, weights: Dict[str, float]):
    engine = get_decision_engine()
    if engine == "senate":
        return await _convene_senate_decision(session, cycle_id, symbol, df, market_data, weights)
    return _sarparast_decision(session, cycle_id, symbol, df, market_data, weights)

# -------------------------
# 🚀 PRODUCTION CYCLE
# -------------------------
async def run_production_cycle(weights: Dict[str, float]):
    cycle_id = now_ts()

    logger.info("=" * 50)
    logger.info(f"🚀 Cycle Start: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S %Z')} 🚀")
    logger.info("=" * 50)

    # Circuit breaker state
    try:
        cb_state = pm._read_json(pm.circuit_breaker_file, {"is_tripped": False, "tripped_until": None})
        if cb_state.get("is_tripped"):
            tripped_until = datetime.fromisoformat(cb_state["tripped_until"])
            if tripped_until.tzinfo is None:
                tripped_until = TZ.localize(tripped_until)
            if datetime.now(TZ) < tripped_until:
                logger.warning(f"🚨 CIRCUIT BREAKER ACTIVE! Trading paused until {tripped_until.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
                return
            else:
                logger.info("Circuit breaker cooldown ended. Resuming.")
                pm._write_json(pm.circuit_breaker_file, {"is_tripped": False, "tripped_until": None})
    except Exception as e:
        logger.error(f"Corrupted circuit breaker state: {e}. Resetting for safety.")
        pm._write_json(pm.circuit_breaker_file, {"is_tripped": False, "tripped_until": None})

    # ✅ Equity computation (supports futures/shorts; still useful in spot for DD protection)
    total_equity, peak_equity, drawdown, eq_details = await compute_total_equity_and_drawdown()
    log_stats("Equity Snapshot", eq_details)

    if CONFIG["TELEMETRY_EQUITY_CURVE"]:
        append_json_array(
            pm.equity_curve_file,
            {"schema_version": SCHEMA_VERSION, "ts": datetime.now(TZ).isoformat(), **eq_details},
            max_len=20000,
        )

    if drawdown >= CONFIG["MAX_DRAWDOWN_PCT"]:
        logger.critical(f"🚨🚨🚨 MAX DRAWDOWN REACHED ({drawdown:.2%})! ACTIVATING CIRCUIT BREAKER! 🚨🚨🚨")
        cooldown_until = datetime.now(TZ) + timedelta(hours=CONFIG["CIRCUIT_BREAKER_COOLDOWN_HOURS"])
        pm._write_json(pm.circuit_breaker_file, {"is_tripped": True, "tripped_until": cooldown_until.isoformat()})

        logger.critical("Liquidating all open positions...")
        for sym, pos in list(pm.get_open_positions().items()):
            price = await get_reliable_price(sym)
            pm.close_position(sym, price if price else float(pos["entry_price"]), "MAX_DRAWDOWN_HIT")
        return

    # Manage open positions (breakeven move + dynamic trailing + SL/TP exits)
    open_positions = pm.get_open_positions()
    for sym, pos in list(open_positions.items()):
        try:
            current_price = await get_reliable_price(sym)
            if not current_price:
                continue

            entry_price = float(pos["entry_price"])
            initial_sl = float(pos.get("initial_sl", pos["sl"]))
            current_sl = float(pos["sl"])
            current_tp = float(pos["tp"])
            current_sl_level = int(pos.get("sl_level", -1))
            initial_risk_dist = abs(entry_price - initial_sl)

            if initial_risk_dist > 1e-9:
                updated_sl = None
                updated_tp = None
                updated_sl_level = current_sl_level
                update_note = None

                if current_sl_level < 0:
                    if pos["side"] == "BUY" and current_price >= entry_price + initial_risk_dist:
                        updated_sl = entry_price
                        updated_tp = current_price + (3.0 * initial_risk_dist)
                        updated_sl_level = 0
                        update_note = "Breakeven move + TP push"
                    elif pos["side"] == "SELL" and current_price <= entry_price - initial_risk_dist:
                        updated_sl = entry_price
                        updated_tp = current_price - (3.0 * initial_risk_dist)
                        updated_sl_level = 0
                        update_note = "Breakeven move + TP push"
                else:
                    trail_dist = 1.5 * initial_risk_dist
                    if pos["side"] == "BUY" and current_price > entry_price + initial_risk_dist:
                        candidate_sl = current_price - trail_dist
                        if candidate_sl > current_sl + 1e-9:
                            updated_sl = candidate_sl
                            updated_tp = current_price + (3.0 * initial_risk_dist)
                            updated_sl_level = current_sl_level + 1
                            update_note = "Trailing move + TP push"
                    elif pos["side"] == "SELL" and current_price < entry_price - initial_risk_dist:
                        candidate_sl = current_price + trail_dist
                        if candidate_sl < current_sl - 1e-9:
                            updated_sl = candidate_sl
                            updated_tp = current_price - (3.0 * initial_risk_dist)
                            updated_sl_level = current_sl_level + 1
                            update_note = "Trailing move + TP push"

                if updated_sl is not None:
                    if pm.update_position_sl(
                        sym,
                        updated_sl,
                        new_tp=updated_tp,
                        new_sl_level=updated_sl_level,
                        note=update_note or "Risk update",
                    ):
                        current_sl = float(updated_sl)
                        current_tp = float(updated_tp)
                        current_sl_level = int(updated_sl_level)

            exit_reason = None
            if pos["side"] == "BUY" and current_price >= current_tp:
                exit_reason = "TP_HIT"
            elif pos["side"] == "BUY" and current_price <= current_sl:
                exit_reason = f"SL_{current_sl_level}_HIT"
            elif pos["side"] == "SELL" and current_price <= current_tp:
                exit_reason = "TP_HIT"
            elif pos["side"] == "SELL" and current_price >= current_sl:
                exit_reason = f"SL_{current_sl_level}_HIT"

            if exit_reason:
                logger.info(f"🎯 Closing {sym} at {current_price}. Reason: {exit_reason}")
                pm.close_position(sym, current_price, exit_reason)

        except Exception as e:
            logger.error(f"Failed to update position for {sym}: {e}")

    # New trades
    if len(pm.get_open_positions()) >= CONFIG["MAX_OPEN_TRADES"]:
        logger.info("🛡️ Max open trades reached. Skipping new trade screening.")
        return

    top_candidates = await screen_candidates(CONFIG["UNIVERSE"], cycle_id)
    if not top_candidates:
        logger.info("😴 No suitable candidates found.")
        return

    corr_stats = {"candidates_in_from_screener": len(top_candidates), "corr_checked": 0, "corr_blocked": 0, "corr_nan_blocked": 0, "corr_passed": 0}

    leverage = float(CONFIG["FUTURES_LEVERAGE"]) if CONFIG["TRADING_MODE"] == "futures" else 1.0

    async with aiohttp.ClientSession() as session:
        for cand in top_candidates:
            if len(pm.get_open_positions()) >= CONFIG["MAX_OPEN_TRADES"]:
                logger.info("🛡️ Max open trades reached during loop. Stopping.")
                break

            sym, df, market_data = cand["symbol"], cand["df"], cand["market_data"]

            corr_stats["corr_checked"] += 1
            is_corr, is_nan = await is_highly_correlated(sym, pm.get_open_positions(), CONFIG["CORRELATION_THRESHOLD"])
            if is_nan:
                corr_stats["corr_nan_blocked"] += 1
                continue
            if is_corr:
                corr_stats["corr_blocked"] += 1
                continue
            corr_stats["corr_passed"] += 1

            decision, deliberation_data = await run_decision_engine(session, cycle_id, sym, df, market_data, weights)

            # ✅ Spot mode: ignore SELL
            if CONFIG["TRADING_MODE"] == "spot" and decision == "SELL":
                logger.info(f"🟦 SPOT MODE: SELL decision ignored for {sym}. Skipping.")
                continue

            # HMM gating layer (after senate decision, before execution)
            if CONFIG.get("HMM_GATING_ENABLED", True) and decision in ["BUY", "SELL"]:
                hmm_policy = str(market_data.get("hmm_policy", "caution")).lower()
                hmm_regime = str(market_data.get("hmm_regime", "unknown")).lower()
                hmm_reason = str(market_data.get("hmm_reason", ""))
                try:
                    hmm_conf = float(market_data.get("hmm_confidence", 0.0))
                except Exception:
                    hmm_conf = 0.0
                try:
                    hmm_persistence = float(market_data.get("hmm_persistence", 0.0))
                except Exception:
                    hmm_persistence = 0.0
                try:
                    hmm_switch_margin = float(market_data.get("hmm_switch_margin", 0.0))
                except Exception:
                    hmm_switch_margin = 0.0
                try:
                    hmm_regime_age_bars = int(market_data.get("hmm_regime_age_bars", 0))
                except Exception:
                    hmm_regime_age_bars = 0
                raw_regime_changed = market_data.get("hmm_regime_changed", False)
                if isinstance(raw_regime_changed, bool):
                    hmm_regime_changed = raw_regime_changed
                else:
                    hmm_regime_changed = str(raw_regime_changed).strip().lower() in {"1", "true", "yes", "y", "on"}

                if hmm_policy == "block":
                    logger.warning(
                        f"🛑 HMM BLOCK: {sym} {decision} rejected | "
                        f"Regime={hmm_regime} | Confidence={hmm_conf:.4f} | Reason={hmm_reason}"
                    )
                    continue

                if (
                    CONFIG["TRADING_MODE"] == "spot"
                    and bool(CONFIG.get("HMM_BLOCK_BEAR_IN_SPOT", True))
                    and decision == "BUY"
                    and hmm_regime == "bear"
                ):
                    logger.warning(
                        f"🛑 HMM SPOT BLOCK: {sym} BUY rejected in bear regime | "
                        f"Confidence={hmm_conf:.4f} | Reason={hmm_reason}"
                    )
                    continue

                if hmm_policy == "caution":
                    stricter_threshold = min(0.95, float(CONFIG["CONSENSUS_THRESHOLD"]) + 0.10)
                    min_persistence = float(CONFIG.get("HMM_MIN_STATE_PERSISTENCE", 0.55))
                    min_switch_margin = float(CONFIG.get("HMM_SWITCH_PROB_MARGIN", 0.10))
                    fresh_switch_bars = int(CONFIG.get("HMM_SWITCH_BLOCK_FRESH_BARS", 2))
                    conviction_proxy = 0.0
                    try:
                        final_analyses = deliberation_data.get("final_analyses", {}) if isinstance(deliberation_data, dict) else {}
                        aligned_score = 0.0
                        total_possible = 0.0
                        for senator_name, vote_data in final_analyses.items():
                            w = float(weights.get(senator_name, 1.0))
                            total_possible += (100.0 * w)
                            vote = vote_data.get("final_vote")
                            conf = int(vote_data.get("final_confidence", 0))
                            if vote == decision and conf >= int(CONFIG["MIN_VOTE_CONFIDENCE"]):
                                aligned_score += (w * conf)
                        if total_possible > 0:
                            conviction_proxy = aligned_score / total_possible
                    except Exception:
                        conviction_proxy = 0.0

                    exceptionally_strong = conviction_proxy >= min(0.99, stricter_threshold + 0.10)
                    if conviction_proxy < stricter_threshold:
                        logger.warning(
                            f"⚠️ HMM CAUTION GATE: {sym} {decision} rejected | "
                            f"ProxyConviction={conviction_proxy:.4f} < Required={stricter_threshold:.4f} | "
                            f"Regime={hmm_regime} | Reason={hmm_reason}"
                        )
                        continue

                    if hmm_persistence < min_persistence:
                        logger.warning(
                            f"⚠️ HMM CAUTION GATE: {sym} {decision} rejected | "
                            f"Persistence={hmm_persistence:.4f} < Required={min_persistence:.4f} | "
                            f"Regime={hmm_regime} | Reason={hmm_reason}"
                        )
                        continue

                    if hmm_switch_margin < min_switch_margin:
                        logger.warning(
                            f"⚠️ HMM CAUTION GATE: {sym} {decision} rejected | "
                            f"SwitchMargin={hmm_switch_margin:.4f} < Required={min_switch_margin:.4f} | "
                            f"Regime={hmm_regime} | Reason={hmm_reason}"
                        )
                        continue

                    if (
                        hmm_regime_changed
                        and hmm_regime_age_bars < fresh_switch_bars
                        and not exceptionally_strong
                    ):
                        logger.warning(
                            f"⚠️ HMM CAUTION GATE: {sym} {decision} rejected | "
                            f"FreshSwitchAge={hmm_regime_age_bars} < Required={fresh_switch_bars} | "
                            f"Regime={hmm_regime} | Reason={hmm_reason}"
                        )
                        continue

            if decision in ["BUY", "SELL"]:
                entry = float(df["Close"].iloc[-1])

                swing_low, swing_high = find_recent_swing_low_high(df, CONFIG["SWING_LOOKBACK_PERIOD"])
                atr = float(market_data["atr"])

                if decision == "BUY":
                    sl_swing = swing_low * 0.998
                    sl_atr = entry - (2.5 * atr)
                    sl = max(sl_swing, sl_atr)
                else:
                    sl_swing = swing_high * 1.002
                    sl_atr = entry + (2.5 * atr)
                    sl = min(sl_swing, sl_atr)

                adx = float(market_data["adx"])
                if adx > 40:
                    reward_ratio = 2.5
                elif adx > 30:
                    reward_ratio = 2.0
                else:
                    reward_ratio = 1.6

                risk_dist = abs(entry - sl)
                if risk_dist <= 1e-9:
                    logger.warning(f"Invalid risk distance for {sym}. Skipping.")
                    continue

                tp = entry + (risk_dist * reward_ratio) if decision == "BUY" else entry - (risk_dist * reward_ratio)

                # ✅ Position sizing (risk based on TOTAL equity, not only free cash)
                total_equity, _, _, eq_details2 = await compute_total_equity_and_drawdown()
                free_cash = float(eq_details2["free_cash"])

                size, margin_used, notional, risk_used, dbg = compute_trade_allocation(
                    total_equity=total_equity,
                    free_cash=free_cash,
                    entry=entry,
                    sl=sl,
                    risk_per_trade_pct=float(CONFIG["RISK_PER_TRADE_PCT"]),
                    leverage=leverage,
                )

                if margin_used < float(CONFIG["MIN_TRADE_MARGIN_USD"]):
                    logger.warning(
                        f"⚠️ Trade approved but free cash too small for {sym}. "
                        f"Margin=${margin_used:.2f} < MIN_TRADE_MARGIN_USD=${CONFIG['MIN_TRADE_MARGIN_USD']}. Skipping."
                    )
                    continue

                if dbg.get("scaled_down"):
                    logger.warning(
                        f"🧩 Scaling down position due to allocation/cash cap for {sym}: "
                        f"DesiredMargin=${dbg['margin_risk']:.2f} -> UsedMargin=${margin_used:.2f} | "
                        f"CapMargin=${dbg['cap_margin']:.2f} | FreeCash=${dbg['free_cash']:.2f}"
                    )

                logger.info(
                    f"⚡ EXECUTING {decision} on {sym} | Entry: {entry:.6f} | SL: {sl:.6f} | TP: {tp:.6f} | "
                    f"Notional=${notional:.2f} | Margin=${margin_used:.2f} | RiskUsed=${risk_used:.2f}"
                )

                pm.record_new_trade(
                    sym,
                    decision,
                    entry,
                    sl,
                    tp,
                    size,
                    margin_used,
                    notional,
                    leverage,
                    risk_used,
                    market_data,
                    deliberation_data,
                )

    log_stats("Correlation Filter", corr_stats)


# -------------------------
# 🧠 MAIN
# -------------------------
async def main():
    logger.info("=" * 60)
    logger.info(f"🤖 LAUNCHING COGNITIVE AI TRADING SYSTEM v{VERSION} 🤖")
    logger.info(f"Mode: {CONFIG['TRADING_MODE'].upper()} | Futures Leverage: {CONFIG['FUTURES_LEVERAGE']} | Max Trades: {CONFIG['MAX_OPEN_TRADES']}")
    logger.info(f"Consensus: {CONFIG['CONSENSUS_THRESHOLD']:.0%} | MinVoteConfidence: {CONFIG['MIN_VOTE_CONFIDENCE']}%")
    logger.info(f"Risk/Trade: {CONFIG['RISK_PER_TRADE_PCT']:.0%} of TOTAL EQUITY | Allocation Cap: {CONFIG['CAPITAL_ALLOCATION_MAX_PCT']:.0%} per trade")
    decision_engine = get_decision_engine()
    logger.info(f"DecisionEngine: {decision_engine}")
    logger.info(f"SellFilterActive: {sell_filter_active()} (enabled={CONFIG.get('SELL_FILTER_ENABLED')})")
    logger.info(
        f"HMM: enabled={CONFIG.get('HMM_ENABLED')} | lookback={CONFIG.get('HMM_LOOKBACK_PERIOD')} | "
        f"min_conf={float(CONFIG.get('HMM_MIN_CONFIDENCE', 0.70)):.2f} | gating={CONFIG.get('HMM_GATING_ENABLED')} | "
        f"backend={CONFIG.get('HMM_EMISSION_BACKEND')} | filtering_only={CONFIG.get('HMM_FILTERING_ONLY')} | "
        f"min_persistence={float(CONFIG.get('HMM_MIN_STATE_PERSISTENCE', 0.55)):.2f}"
    )
    logger.info(f"MAX_TOKENS: {CONFIG['MAX_TOKENS']} (locked)")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"LOG_DIR : {CONFIG['LOG_DIR']}")
    logger.info(f"DATA_DIR: {CONFIG['DATA_DIR']}")
    logger.info("=" * 60)

    # Must be after logger is ready
    enforce_non_negotiables_or_exit()

    if decision_engine == "senate":
        check_api_keys()
    else:
        logger.info("Skipping LLM API key check because DECISION_ENGINE=sarparast.")

    try:
        while True:
            weights = load_decision_weights()
            await run_production_cycle(weights)
            sleep_duration = 3600
            logger.info(f"Cycle complete. Sleeping for {sleep_duration / 60} minutes... 💤")
            await asyncio.sleep(sleep_duration)

    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown initiated. Exiting...")
    except Exception as e:
        logger.critical(f"☠️ A fatal, unhandled error occurred in the main loop: {e}", exc_info=True)


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
