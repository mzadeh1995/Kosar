# ==============================================================================
# 🧮 OFI.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Order Flow Imbalance (OFI) feature engineering utilities.
# ==============================================================================

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ofi_l1",
    "depth_imbalance",
    "spread_bps",
    "liquidity_usd",
    "mid_price",
    "last_update_id",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return None


def _safe_int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def normalize_depth_snapshot(raw_depth: dict) -> dict:
    out = {
        "last_update_id": None,
        "bids": [],
        "asks": [],
    }

    if not isinstance(raw_depth, dict):
        return out

    out["last_update_id"] = _safe_int_or_none(raw_depth.get("lastUpdateId"))

    bids_raw = raw_depth.get("bids", [])
    asks_raw = raw_depth.get("asks", [])

    if isinstance(bids_raw, list):
        bids = []
        for row in bids_raw:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            price = _finite_or_none(row[0])
            qty = _finite_or_none(row[1])
            if price is None or qty is None or price <= 0 or qty < 0:
                continue
            bids.append((float(price), float(qty)))
        bids.sort(key=lambda x: x[0], reverse=True)
        out["bids"] = bids

    if isinstance(asks_raw, list):
        asks = []
        for row in asks_raw:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            price = _finite_or_none(row[0])
            qty = _finite_or_none(row[1])
            if price is None or qty is None or price <= 0 or qty < 0:
                continue
            asks.append((float(price), float(qty)))
        asks.sort(key=lambda x: x[0])
        out["asks"] = asks

    return out


def _best_l1(snapshot: dict) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
    bids = snapshot.get("bids", []) if isinstance(snapshot, dict) else []
    asks = snapshot.get("asks", []) if isinstance(snapshot, dict) else []

    best_bid = bids[0] if isinstance(bids, list) and bids else None
    best_ask = asks[0] if isinstance(asks, list) and asks else None
    return best_bid, best_ask


def compute_l1_ofi(prev_snapshot: dict, curr_snapshot: dict) -> float:
    prev_norm = normalize_depth_snapshot(prev_snapshot)
    curr_norm = normalize_depth_snapshot(curr_snapshot)

    prev_bid, prev_ask = _best_l1(prev_norm)
    curr_bid, curr_ask = _best_l1(curr_norm)

    if prev_bid is None or prev_ask is None or curr_bid is None or curr_ask is None:
        return 0.0

    prev_bid_price, prev_bid_qty = prev_bid
    prev_ask_price, prev_ask_qty = prev_ask
    curr_bid_price, curr_bid_qty = curr_bid
    curr_ask_price, curr_ask_qty = curr_ask

    e_n = (
        (1.0 if curr_bid_price >= prev_bid_price else 0.0) * curr_bid_qty
        - (1.0 if curr_bid_price <= prev_bid_price else 0.0) * prev_bid_qty
        - (1.0 if curr_ask_price <= prev_ask_price else 0.0) * curr_ask_qty
        + (1.0 if curr_ask_price >= prev_ask_price else 0.0) * prev_ask_qty
    )

    return _safe_float(e_n, 0.0)


def compute_depth_imbalance(snapshot: dict, levels: int = 10) -> float:
    norm = normalize_depth_snapshot(snapshot)
    lvl = max(1, int(levels))

    bids = norm["bids"][:lvl]
    asks = norm["asks"][:lvl]

    bid_qty = float(np.sum([q for _, q in bids])) if bids else 0.0
    ask_qty = float(np.sum([q for _, q in asks])) if asks else 0.0

    denom = bid_qty + ask_qty
    if denom <= 0:
        return 0.0

    value = (bid_qty - ask_qty) / denom
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def compute_spread_bps(snapshot: dict) -> float:
    norm = normalize_depth_snapshot(snapshot)
    best_bid, best_ask = _best_l1(norm)
    if best_bid is None or best_ask is None:
        return 0.0

    bid = _safe_float(best_bid[0], 0.0)
    ask = _safe_float(best_ask[0], 0.0)
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0

    spread = ((ask - bid) / mid) * 10000.0
    if not np.isfinite(spread):
        return 0.0
    return float(spread)


def compute_liquidity_usd(snapshot: dict, levels: int = 10) -> float:
    norm = normalize_depth_snapshot(snapshot)
    lvl = max(1, int(levels))

    bids = norm["bids"][:lvl]
    asks = norm["asks"][:lvl]

    bid_notional = float(np.sum([p * q for p, q in bids])) if bids else 0.0
    ask_notional = float(np.sum([p * q for p, q in asks])) if asks else 0.0
    liquidity = bid_notional + ask_notional
    if not np.isfinite(liquidity):
        return 0.0
    return float(max(0.0, liquidity))


def _compute_mid_price(snapshot: dict) -> float:
    norm = normalize_depth_snapshot(snapshot)
    best_bid, best_ask = _best_l1(norm)
    if best_bid is None or best_ask is None:
        return 0.0
    mid = (_safe_float(best_bid[0], 0.0) + _safe_float(best_ask[0], 0.0)) / 2.0
    if not np.isfinite(mid):
        return 0.0
    return float(max(0.0, mid))


def compute_ofi_series(depth_snapshots: list[dict], levels: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    if not isinstance(depth_snapshots, list) or not depth_snapshots:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    prev_norm: Optional[dict] = None
    lvl = max(1, int(levels))

    for raw in depth_snapshots:
        norm = normalize_depth_snapshot(raw)
        ofi_l1 = compute_l1_ofi(prev_norm, norm) if prev_norm is not None else 0.0

        row = {
            "ofi_l1": _safe_float(ofi_l1, 0.0),
            "depth_imbalance": _safe_float(compute_depth_imbalance(norm, levels=lvl), 0.0),
            "spread_bps": _safe_float(compute_spread_bps(norm), 0.0),
            "liquidity_usd": _safe_float(compute_liquidity_usd(norm, levels=lvl), 0.0),
            "mid_price": _safe_float(_compute_mid_price(norm), 0.0),
            "last_update_id": norm.get("last_update_id"),
        }
        rows.append(row)
        prev_norm = norm

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    for col in ["ofi_l1", "depth_imbalance", "spread_bps", "liquidity_usd", "mid_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _local_zscore(series: pd.Series, window: int) -> float:
    if series is None or series.empty:
        return 0.0
    tail = pd.to_numeric(series.tail(max(2, int(window))), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if tail.shape[0] < 2:
        return 0.0
    mean = float(tail.mean())
    std = float(tail.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    z = (float(tail.iloc[-1]) - mean) / std
    return _safe_float(z, 0.0)


def build_ofi_features(depth_snapshots: list[dict], config: dict) -> dict:
    cfg = config or {}
    levels = max(1, int(cfg.get("OFI_LEVELS", 10)))
    min_liq_usd = max(0.0, _safe_float(cfg.get("OFI_MIN_LIQUIDITY_USD", 0.0), 0.0))
    z_window = max(2, int(cfg.get("OFI_ZSCORE_WINDOW", 50)))

    neutral = {
        "ofi_ok": False,
        "ofi_reason": "no_snapshots",
        "ofi_l1": 0.0,
        "ofi_l1_norm": 0.0,
        "ofi_l1_norm_denom_qty": 0.0,
        "ofi_l1_z": 0.0,
        "ofi_depth_imbalance": 0.0,
        "ofi_spread_bps": 0.0,
        "ofi_liquidity_usd": 0.0,
        "ofi_snapshot_count": 0,
        "ofi_levels": int(levels),
    }

    try:
        df = compute_ofi_series(depth_snapshots if isinstance(depth_snapshots, list) else [], levels=levels)
        if df.empty:
            return neutral

        last = df.iloc[-1]
        snapshot_count = int(len(df))

        latest_liquidity = _safe_float(last.get("liquidity_usd"), 0.0)
        latest_ofi = _safe_float(last.get("ofi_l1"), 0.0)

        latest_snapshot = normalize_depth_snapshot(depth_snapshots[-1] if depth_snapshots else {})
        best_bid, best_ask = _best_l1(latest_snapshot)
        best_bid_qty = _safe_float(best_bid[1], 0.0) if best_bid is not None else 0.0
        best_ask_qty = _safe_float(best_ask[1], 0.0) if best_ask is not None else 0.0
        denom_qty = max(0.0, best_bid_qty) + max(0.0, best_ask_qty)

        if denom_qty > 1e-12:
            ofi_l1_norm = latest_ofi / denom_qty
        else:
            ofi_l1_norm = 0.0
        ofi_l1_norm = float(np.clip(_safe_float(ofi_l1_norm, 0.0), -1.0, 1.0))

        reason = None
        ofi_ok = True
        if snapshot_count < 2:
            ofi_ok = False
            reason = "insufficient_snapshots_for_l1"
        if latest_liquidity < min_liq_usd:
            ofi_ok = False
            reason = "liquidity_below_min"

        out = {
            "ofi_ok": bool(ofi_ok),
            "ofi_reason": reason,
            "ofi_l1": _safe_float(latest_ofi, 0.0),
            "ofi_l1_norm": _safe_float(ofi_l1_norm, 0.0),
            "ofi_l1_norm_denom_qty": _safe_float(denom_qty, 0.0),
            "ofi_l1_z": _safe_float(_local_zscore(df["ofi_l1"], z_window), 0.0),
            "ofi_depth_imbalance": _safe_float(last.get("depth_imbalance"), 0.0),
            "ofi_spread_bps": _safe_float(last.get("spread_bps"), 0.0),
            "ofi_liquidity_usd": _safe_float(latest_liquidity, 0.0),
            "ofi_snapshot_count": snapshot_count,
            "ofi_levels": int(levels),
        }

        for k in ["ofi_l1", "ofi_l1_norm", "ofi_l1_norm_denom_qty", "ofi_l1_z", "ofi_depth_imbalance", "ofi_spread_bps", "ofi_liquidity_usd"]:
            out[k] = _safe_float(out.get(k, 0.0), 0.0)

        return out
    except Exception:
        return neutral
