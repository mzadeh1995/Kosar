# ==============================================================================
# 💼 portfolio.py
# ------------------------------------------------------------------------------
# Portfolio state, persistence, equity computation, and position sizing.
# v49.1: Added "Calibration" and Patch
# portfolio/risk behavior preserved.
# ==============================================================================

from __future__ import annotations

import os
import json
import uuid
from typing import Any, Tuple, Optional
from datetime import datetime, timedelta

from config import CONFIG, TZ
from telemetry_utils import logger, write_json

# -------------------------
# 🏦 PORTFOLIO MANAGER (DATA -> ./data)
# -------------------------
class PortfolioManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.wallet_file = os.path.join(data_dir, "wallet.json")
        self.positions_file = os.path.join(data_dir, "open_positions.json")
        self.history_file = os.path.join(data_dir, "trade_history.json")
        self.weights_file = os.path.join(data_dir, "senator_weights.json")
        self.circuit_breaker_file = os.path.join(data_dir, "circuit_breaker_state.json")
        self.equity_tracker_file = os.path.join(data_dir, "equity_tracker.json")
        self.equity_curve_file = os.path.join(data_dir, "equity_curve.json")  # JSON array
        self.initialize_storage()

    def initialize_storage(self):
        files_to_init = [
            (self.wallet_file, {"balance": CONFIG["INITIAL_BALANCE_USD"], "updated_at": datetime.now(TZ).isoformat()}),
            (self.positions_file, {}),
            (self.history_file, []),
            (self.weights_file, {}),
            (self.circuit_breaker_file, {"is_tripped": False, "tripped_until": None}),
            (self.equity_tracker_file, {"peak_equity": CONFIG["INITIAL_BALANCE_USD"]}),
            (self.equity_curve_file, []),
        ]
        for fp, dc in files_to_init:
            if not os.path.exists(fp):
                write_json(fp, dc)

    def _read_json(self, fp: str, dv: Any):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error reading JSON from {fp}: {e}")
            return dv

    def _write_json(self, fp: str, data: Any):
        try:
            write_json(fp, data)
        except Exception as e:
            logger.error(f"❌ Error writing JSON to {fp}: {e}")

    def get_balance(self) -> float:
        return float(self._read_json(self.wallet_file, {}).get("balance", 0.0))

    def get_open_positions(self) -> dict:
        return self._read_json(self.positions_file, {})

    def get_trade_history(self) -> list:
        return self._read_json(self.history_file, [])

    def is_position_open(self, symbol: str) -> bool:
        return symbol in self.get_open_positions()

    def update_position_sl(
        self,
        symbol: str,
        new_sl: float,
        new_tp: Optional[float] = None,
        new_sl_level: Optional[int] = None,
        note: str = "Risk update",
    ) -> bool:
        positions = self.get_open_positions()
        if symbol in positions:
            old_sl = float(positions[symbol].get("sl", new_sl))
            old_tp = float(positions[symbol].get("tp", new_tp if new_tp is not None else 0.0))
            old_level = int(positions[symbol].get("sl_level", -1))
            positions[symbol]["sl"] = float(new_sl)
            if new_tp is not None:
                positions[symbol]["tp"] = float(new_tp)
            if new_sl_level is not None:
                positions[symbol]["sl_level"] = int(new_sl_level)
            self._write_json(self.positions_file, positions)
            logger.info(
                f"✅ SL/TP for {symbol} updated | "
                f"SL: {old_sl:.6f} -> {float(positions[symbol]['sl']):.6f} | "
                f"TP: {old_tp:.6f} -> {float(positions[symbol].get('tp', old_tp)):.6f} | "
                f"SL Level: {old_level} -> {int(positions[symbol].get('sl_level', old_level))} | "
                f"{note}"
            )
            return True
        return False

    def record_new_trade(
        self,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        size: float,
        margin_used: float,
        notional_usd: float,
        leverage: float,
        risk_amount_usd: float,
        market_data: dict,
        senator_deliberations: dict,
    ) -> bool:
        current_balance = self.get_balance()

        if margin_used > current_balance + 1e-9:
            logger.critical(f"⛔ Margin exceeds free cash for {symbol}! Margin=${margin_used:.2f}, FreeCash=${current_balance:.2f}")
            return False

        # Lock margin
        self._write_json(self.wallet_file, {"balance": current_balance - margin_used, "updated_at": datetime.now(TZ).isoformat()})

        positions = self.get_open_positions()
        trade_id = f"{symbol}_{uuid.uuid4().hex[:8]}"

        risk_dist = abs(entry - sl)
        reward_dist = abs(tp - entry)
        initial_rr = round(reward_dist / risk_dist, 2) if risk_dist > 0 else 0.0
        hmm_regime_at_entry = market_data.get("hmm_regime")
        hmm_policy_at_entry = market_data.get("hmm_policy")
        hmm_prev_regime_at_entry = market_data.get("hmm_prev_regime")
        try:
            hmm_confidence_at_entry = float(market_data.get("hmm_confidence")) if market_data.get("hmm_confidence") is not None else None
        except Exception:
            hmm_confidence_at_entry = None
        try:
            hmm_persistence_at_entry = float(market_data.get("hmm_persistence")) if market_data.get("hmm_persistence") is not None else None
        except Exception:
            hmm_persistence_at_entry = None
        raw_changed = market_data.get("hmm_regime_changed", False)
        if isinstance(raw_changed, bool):
            hmm_regime_changed_at_entry = raw_changed
        else:
            hmm_regime_changed_at_entry = str(raw_changed).strip().lower() in {"1", "true", "yes", "y", "on"}
        try:
            hmm_regime_age_bars_at_entry = int(market_data.get("hmm_regime_age_bars", 0))
        except Exception:
            hmm_regime_age_bars_at_entry = 0

        positions[symbol] = {
            "trade_id": trade_id,
            "side": side,
            "entry_price": float(entry),
            "initial_sl": float(sl),
            "sl": float(sl),
            "sl_level": -1,
            "tp": float(tp),
            "size": float(size),
            "margin_used_usd": float(margin_used),
            "notional_usd": float(notional_usd),
            "leverage": float(leverage),
            "risk_amount_usd": float(risk_amount_usd),
            "initial_rr": float(initial_rr),
            "market_data_at_entry": market_data,
            "hmm_regime_at_entry": hmm_regime_at_entry,
            "hmm_confidence_at_entry": hmm_confidence_at_entry,
            "hmm_policy_at_entry": hmm_policy_at_entry,
            "hmm_persistence_at_entry": hmm_persistence_at_entry,
            "hmm_regime_changed_at_entry": hmm_regime_changed_at_entry,
            "hmm_regime_age_bars_at_entry": hmm_regime_age_bars_at_entry,
            "hmm_prev_regime_at_entry": hmm_prev_regime_at_entry,
            "open_time": datetime.now(TZ).isoformat(),
            "senator_deliberations": senator_deliberations,
        }
        self._write_json(self.positions_file, positions)
        logger.info(f"✅ Trade Recorded: {side} {symbol} | ID: {trade_id} | Notional=${notional_usd:.2f} | Margin=${margin_used:.2f} | RR: {initial_rr}")
        return True

    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        positions = self.get_open_positions()
        if symbol not in positions:
            return

        pos = positions.pop(symbol)

        entry = float(pos["entry_price"])
        size = float(pos["size"])
        side = pos["side"]
        margin_used = float(pos.get("margin_used_usd", pos.get("cost_usd", 0.0)))

        pnl_usd = (exit_price - entry) * size if side == "BUY" else (entry - exit_price) * size
        realized_r = round(pnl_usd / float(pos["risk_amount_usd"]), 2) if float(pos["risk_amount_usd"]) > 0 else 0.0

        trade_outcome = "WIN" if pnl_usd > 0 else "LOSS" if pnl_usd < 0 else "NEUTRAL"

        # Release margin + pnl
        self._write_json(
            self.wallet_file,
            {"balance": self.get_balance() + margin_used + pnl_usd, "updated_at": datetime.now(TZ).isoformat()}
        )
        self._write_json(self.positions_file, positions)

        deliberations = pos.get("senator_deliberations", {})
        final_analyses = deliberations.get("final_analyses", {})
        graded_votes = final_analyses.copy()

        for senator_name, vote_info in graded_votes.items():
            voted_side = vote_info.get("final_vote")
            vote_info["aligned"] = (voted_side == side)
            if voted_side not in ["BUY", "SELL"]:
                vote_info["correct"] = None
            elif trade_outcome == "WIN":
                vote_info["correct"] = (voted_side == side)
            elif trade_outcome == "LOSS":
                vote_info["correct"] = (voted_side != side)
            else:
                vote_info["correct"] = None

            if "reason_for_final_decision" in vote_info:
                vote_info["reason_short"] = str(vote_info["reason_for_final_decision"])[:200]

        deliberations["final_analyses_graded"] = graded_votes

        history_record = {
            "trade_id": pos["trade_id"],
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "sl": float(pos["sl"]),
            "sl_level": int(pos.get("sl_level", -1)),
            "tp": float(pos["tp"]),
            "exit_reason": exit_reason,
            "pnl_usd": round(pnl_usd, 2),
            "risk_amount_usd": round(float(pos["risk_amount_usd"]), 2),
            "initial_rr": float(pos["initial_rr"]),
            "realized_r": realized_r,
            "outcome": trade_outcome,
            "market_data": pos["market_data_at_entry"],
            "hmm_regime_at_entry": pos.get("hmm_regime_at_entry"),
            "hmm_confidence_at_entry": pos.get("hmm_confidence_at_entry"),
            "hmm_policy_at_entry": pos.get("hmm_policy_at_entry"),
            "hmm_persistence_at_entry": pos.get("hmm_persistence_at_entry"),
            "hmm_regime_changed_at_entry": pos.get("hmm_regime_changed_at_entry"),
            "hmm_regime_age_bars_at_entry": pos.get("hmm_regime_age_bars_at_entry"),
            "hmm_prev_regime_at_entry": pos.get("hmm_prev_regime_at_entry"),
            "open_time": pos["open_time"],
            "close_time": datetime.now(TZ).isoformat(),
            "senator_deliberations": deliberations,
        }

        history = self.get_trade_history()
        history.append(history_record)
        self._write_json(self.history_file, history)

        logger.info(f"🏁 Position Closed: {symbol} | PnL: ${pnl_usd:.2f} | R: {realized_r} | Outcome: {trade_outcome}")


pm = PortfolioManager(CONFIG["DATA_DIR"])


# 🧮 Equity computation (supports futures margin model + shorts)
# -------------------------
async def compute_total_equity_and_drawdown() -> Tuple[float, float, float, dict]:
    """
    total_equity = free_cash + sum(margin_used) + sum(unrealized_pnl)
    This works for BUY and SELL when margin is locked from wallet.
    """
    free_cash = pm.get_balance()
    open_positions = pm.get_open_positions()
    # Local import to avoid circular imports (market.py imports pm from portfolio.py)
    from market import get_reliable_price

    total_margin = 0.0
    total_unreal_pnl = 0.0

    for sym, pos in open_positions.items():
        entry = float(pos["entry_price"])
        size = float(pos["size"])
        side = pos["side"]
        margin_used = float(pos.get("margin_used_usd", 0.0))
        total_margin += margin_used

        price = await get_reliable_price(sym)
        if price is None:
            price = entry

        unreal = (price - entry) * size if side == "BUY" else (entry - price) * size
        total_unreal_pnl += unreal

    total_equity = free_cash + total_margin + total_unreal_pnl

    equity_tracker = pm._read_json(pm.equity_tracker_file, {"peak_equity": CONFIG["INITIAL_BALANCE_USD"]})
    peak_equity = float(equity_tracker.get("peak_equity", CONFIG["INITIAL_BALANCE_USD"]))

    if total_equity > peak_equity:
        peak_equity = total_equity
        pm._write_json(pm.equity_tracker_file, {"peak_equity": peak_equity})

    drawdown = (peak_equity - total_equity) / peak_equity if peak_equity > 0 else 0.0

    details = {
        "free_cash": round(free_cash, 4),
        "total_margin": round(total_margin, 4),
        "unrealized_pnl": round(total_unreal_pnl, 4),
        "total_equity": round(total_equity, 4),
        "peak_equity": round(peak_equity, 4),
        "drawdown": round(drawdown, 6),
    }
    return total_equity, peak_equity, drawdown, details


# -------------------------
# 💰 Position sizing (leverage=1 + capital allocation + scale-down)
# -------------------------
def compute_trade_allocation(
    total_equity: float,
    free_cash: float,
    entry: float,
    sl: float,
    risk_per_trade_pct: float,
    leverage: float,
) -> Tuple[float, float, float, float, dict]:
    """
    Returns: (size, margin_used, notional, risk_amount_used, debug)
    - size is quantity
    - margin_used locked from wallet
    - notional is position notional = entry * size
    - risk_amount_used is $ risk implied by SL distance for that size
    """
    risk_dist = abs(entry - sl)
    if risk_dist <= 1e-9 or entry <= 0:
        return 0.0, 0.0, 0.0, 0.0, {"reason": "invalid_risk_dist_or_entry"}

    risk_target = max(0.0, float(total_equity) * float(risk_per_trade_pct))

    size_risk = risk_target / risk_dist
    notional_risk = size_risk * entry
    margin_risk = notional_risk / leverage

    cap_margin = float(total_equity) * float(CONFIG["CAPITAL_ALLOCATION_MAX_PCT"])
    max_margin_this_trade = min(free_cash, cap_margin)

    margin_used = min(margin_risk, max_margin_this_trade)

    notional = margin_used * leverage
    size = notional / entry
    risk_used = size * risk_dist

    dbg = {
        "risk_target": risk_target,
        "risk_used": risk_used,
        "risk_dist": risk_dist,
        "size_risk": size_risk,
        "notional_risk": notional_risk,
        "margin_risk": margin_risk,
        "cap_margin": cap_margin,
        "free_cash": free_cash,
        "max_margin_this_trade": max_margin_this_trade,
        "margin_used": margin_used,
        "notional_used": notional,
        "size_used": size,
        "scaled_down": margin_used + 1e-9 < margin_risk,
    }
    return size, margin_used, notional, risk_used, dbg
