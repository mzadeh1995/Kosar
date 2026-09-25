# ==============================================================================
# 📊 VPIN.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# VPIN (Volume-Synchronized Probability of Informed Trading) utilities.
# ==============================================================================

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "timestamp_ms",
    "price",
    "quantity_base",
    "quantity_quote",
    "buyer_initiated_volume_quote",
    "seller_initiated_volume_quote",
    "buyer_initiated_volume_base",
    "seller_initiated_volume_base",
    "is_buyer_initiated",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def normalize_agg_trades(raw_trades: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    if not isinstance(raw_trades, list):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    for rec in raw_trades:
        if not isinstance(rec, dict):
            continue

        try:
            ts = int(rec.get("T"))
        except Exception:
            continue

        price = _safe_float(rec.get("p"), float("nan"))
        qty_base = _safe_float(rec.get("q"), float("nan"))
        if (not np.isfinite(price)) or (not np.isfinite(qty_base)) or price <= 0 or qty_base <= 0 or ts < 0:
            continue

        qty_quote = price * qty_base
        if not np.isfinite(qty_quote) or qty_quote <= 0:
            continue

        is_buyer_initiated = not bool(rec.get("m", False))

        buy_quote = qty_quote if is_buyer_initiated else 0.0
        sell_quote = qty_quote if not is_buyer_initiated else 0.0
        buy_base = qty_base if is_buyer_initiated else 0.0
        sell_base = qty_base if not is_buyer_initiated else 0.0

        rows.append(
            {
                "timestamp_ms": int(ts),
                "price": float(price),
                "quantity_base": float(qty_base),
                "quantity_quote": float(qty_quote),
                "buyer_initiated_volume_quote": float(buy_quote),
                "seller_initiated_volume_quote": float(sell_quote),
                "buyer_initiated_volume_base": float(buy_base),
                "seller_initiated_volume_base": float(sell_base),
                "is_buyer_initiated": bool(is_buyer_initiated),
            }
        )

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in [
        "price",
        "quantity_base",
        "quantity_quote",
        "buyer_initiated_volume_quote",
        "seller_initiated_volume_quote",
        "buyer_initiated_volume_base",
        "seller_initiated_volume_base",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(
        subset=[
            "timestamp_ms",
            "price",
            "quantity_base",
            "quantity_quote",
            "buyer_initiated_volume_quote",
            "seller_initiated_volume_quote",
            "buyer_initiated_volume_base",
            "seller_initiated_volume_base",
        ],
        inplace=True,
    )

    return df


def resolve_bucket_volume(trades_df: pd.DataFrame, config: dict) -> tuple[float, str]:
    cfg = config or {}
    mode = str(cfg.get("VPIN_BUCKET_VOLUME_MODE", "dynamic_quote")).strip().lower()
    bucket_count = max(1, int(cfg.get("VPIN_BUCKET_COUNT", 20)))

    if mode == "fixed_quote":
        return max(0.0, _safe_float(cfg.get("VPIN_FIXED_BUCKET_VOLUME_QUOTE", 100000.0), 100000.0)), "quote"

    if mode == "fixed_base":
        return max(0.0, _safe_float(cfg.get("VPIN_FIXED_BUCKET_VOLUME_BASE", 10.0), 10.0)), "base"

    total_quote = 0.0
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and "quantity_quote" in trades_df.columns:
        total_quote = _safe_float(pd.to_numeric(trades_df["quantity_quote"], errors="coerce").sum(), 0.0)

    return (total_quote / float(bucket_count)) if total_quote > 0 else 0.0, "quote"


def build_volume_buckets(trades_df: pd.DataFrame, bucket_volume: float, volume_unit: str) -> pd.DataFrame:
    cols = ["buy_volume", "sell_volume", "total_volume", "imbalance_abs", "start_ts", "end_ts"]
    if trades_df is None or not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        return pd.DataFrame(columns=cols)

    bucket_vol = _safe_float(bucket_volume, 0.0)
    if bucket_vol <= 1e-12:
        return pd.DataFrame(columns=cols)

    unit = str(volume_unit).strip().lower()
    if unit not in {"quote", "base"}:
        unit = "quote"

    if unit == "base":
        buy_col = "buyer_initiated_volume_base"
        sell_col = "seller_initiated_volume_base"
    else:
        buy_col = "buyer_initiated_volume_quote"
        sell_col = "seller_initiated_volume_quote"

    records: list[dict] = []
    cur_buy = 0.0
    cur_sell = 0.0
    cur_total = 0.0
    cur_start_ts = None
    cur_end_ts = None

    for _, row in trades_df.iterrows():
        buy_left = max(0.0, _safe_float(row.get(buy_col), 0.0))
        sell_left = max(0.0, _safe_float(row.get(sell_col), 0.0))
        trade_total_left = buy_left + sell_left

        if trade_total_left <= 1e-12:
            continue

        ts = row.get("timestamp_ms")
        try:
            ts_int = int(ts)
        except Exception:
            ts_int = None

        while trade_total_left > 1e-12:
            remaining_capacity = bucket_vol - cur_total
            if remaining_capacity <= 1e-12:
                imbalance = abs(cur_buy - cur_sell)
                records.append(
                    {
                        "buy_volume": float(cur_buy),
                        "sell_volume": float(cur_sell),
                        "total_volume": float(cur_total),
                        "imbalance_abs": float(imbalance),
                        "start_ts": cur_start_ts,
                        "end_ts": cur_end_ts,
                    }
                )
                cur_buy = 0.0
                cur_sell = 0.0
                cur_total = 0.0
                cur_start_ts = None
                cur_end_ts = None
                remaining_capacity = bucket_vol

            take = min(remaining_capacity, trade_total_left)
            if take <= 0:
                break

            buy_take = take * (buy_left / trade_total_left) if trade_total_left > 0 else 0.0
            sell_take = take * (sell_left / trade_total_left) if trade_total_left > 0 else 0.0

            cur_buy += buy_take
            cur_sell += sell_take
            cur_total += take

            buy_left = max(0.0, buy_left - buy_take)
            sell_left = max(0.0, sell_left - sell_take)
            trade_total_left = buy_left + sell_left

            if cur_start_ts is None:
                cur_start_ts = ts_int
            cur_end_ts = ts_int

            if cur_total >= (bucket_vol - 1e-12):
                imbalance = abs(cur_buy - cur_sell)
                records.append(
                    {
                        "buy_volume": float(cur_buy),
                        "sell_volume": float(cur_sell),
                        "total_volume": float(cur_total),
                        "imbalance_abs": float(imbalance),
                        "start_ts": cur_start_ts,
                        "end_ts": cur_end_ts,
                    }
                )
                cur_buy = 0.0
                cur_sell = 0.0
                cur_total = 0.0
                cur_start_ts = None
                cur_end_ts = None

    if not records:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(records, columns=cols)
    for col in ["buy_volume", "sell_volume", "total_volume", "imbalance_abs"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_vpin_from_buckets(buckets_df: pd.DataFrame, bucket_volume: float, min_buckets: int) -> dict:
    out = {
        "vpin_ok": False,
        "vpin_reason": "insufficient_buckets",
        "vpin": 0.0,
        "vpin_bucket_count": 0,
        "vpin_buy_volume": 0.0,
        "vpin_sell_volume": 0.0,
        "vpin_total_volume": 0.0,
        "vpin_imbalance_sum": 0.0,
        "vpin_denominator": 0.0,
        "vpin_lower_bound": 0.0,
        "vpin_formula_check_ok": False,
    }

    if buckets_df is None or not isinstance(buckets_df, pd.DataFrame) or buckets_df.empty:
        return out

    bucket_vol = _safe_float(bucket_volume, 0.0)
    if bucket_vol <= 1e-12:
        out["vpin_reason"] = "invalid_bucket_volume"
        return out

    min_count = max(1, int(min_buckets))
    n = int(len(buckets_df))

    buy_series = pd.to_numeric(buckets_df["buy_volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sell_series = pd.to_numeric(buckets_df["sell_volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total_series = pd.to_numeric(buckets_df["total_volume"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    buy_sum = _safe_float(buy_series.sum(), 0.0)
    sell_sum = _safe_float(sell_series.sum(), 0.0)
    total_sum = _safe_float(total_series.sum(), 0.0)

    # VPIN numerator must be sum(abs(bucket_buy - bucket_sell)).
    imbalance_series = (buy_series - sell_series).abs()
    numerator = _safe_float(imbalance_series.sum(), 0.0)
    denom = float(n) * bucket_vol

    vpin = (numerator / denom) if denom > 1e-12 else 0.0
    if not np.isfinite(vpin):
        vpin = 0.0
    vpin = float(np.clip(vpin, 0.0, 1.0))
    lower_bound = (abs(buy_sum - sell_sum) / denom) if denom > 1e-12 else 0.0
    lower_bound = _safe_float(lower_bound, 0.0)

    vpin_ok = bool(n >= min_count)
    formula_check_ok = True
    if vpin_ok and n > 0:
        formula_check_ok = bool(vpin >= (lower_bound - 1e-9))
    reason = None if vpin_ok else "insufficient_buckets"
    if vpin_ok and not formula_check_ok:
        reason = "formula_check_failed"

    out.update(
        {
            "vpin_ok": vpin_ok,
            "vpin_reason": reason,
            "vpin": float(vpin),
            "vpin_bucket_count": n,
            "vpin_buy_volume": float(max(0.0, buy_sum)),
            "vpin_sell_volume": float(max(0.0, sell_sum)),
            "vpin_total_volume": float(max(0.0, total_sum)),
            "vpin_imbalance_sum": float(max(0.0, numerator)),
            "vpin_denominator": float(max(0.0, denom)),
            "vpin_lower_bound": float(max(0.0, lower_bound)),
            "vpin_formula_check_ok": bool(formula_check_ok),
        }
    )

    return out


def _local_zscore(series: pd.Series, window: int) -> float:
    if series is None or series.empty:
        return 0.0
    tail = pd.to_numeric(series.tail(max(2, int(window))), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if tail.shape[0] < 2:
        return 0.0
    mean = _safe_float(tail.mean(), 0.0)
    std = _safe_float(tail.std(ddof=0), 0.0)
    if std <= 1e-12:
        return 0.0
    return _safe_float((float(tail.iloc[-1]) - mean) / std, 0.0)


def build_vpin_features(raw_trades: list[dict], config: dict) -> dict:
    cfg = config or {}
    neutral = {
        "vpin_ok": False,
        "vpin_reason": "no_trades",
        "vpin": 0.0,
        "vpin_z": 0.0,
        "vpin_bucket_count": 0,
        "vpin_bucket_volume": 0.0,
        "vpin_volume_unit": "quote",
        "vpin_buy_volume": 0.0,
        "vpin_sell_volume": 0.0,
        "vpin_total_volume": 0.0,
        "vpin_imbalance_sum": 0.0,
        "vpin_denominator": 0.0,
        "vpin_lower_bound": 0.0,
        "vpin_formula_check_ok": False,
        "vpin_last_trade_ts": None,
    }

    try:
        trades_df = normalize_agg_trades(raw_trades)
        if trades_df.empty:
            return neutral

        bucket_volume, volume_unit = resolve_bucket_volume(trades_df, cfg)
        if bucket_volume <= 1e-12:
            neutral["vpin_reason"] = "invalid_bucket_volume"
            neutral["vpin_volume_unit"] = str(volume_unit)
            return neutral

        buckets_df = build_volume_buckets(trades_df, bucket_volume=bucket_volume, volume_unit=volume_unit)
        core = compute_vpin_from_buckets(
            buckets_df,
            bucket_volume=bucket_volume,
            min_buckets=max(1, int(cfg.get("VPIN_MIN_BUCKETS", 5))),
        )

        ratio_series = pd.Series(dtype=float)
        if not buckets_df.empty:
            ratio_series = pd.to_numeric(buckets_df["imbalance_abs"], errors="coerce") / max(bucket_volume, 1e-12)

        out = {
            "vpin_ok": bool(core.get("vpin_ok", False)),
            "vpin_reason": core.get("vpin_reason"),
            "vpin": _safe_float(core.get("vpin", 0.0), 0.0),
            "vpin_z": _local_zscore(ratio_series, max(2, int(cfg.get("VPIN_ZSCORE_WINDOW", 50)))),
            "vpin_bucket_count": int(core.get("vpin_bucket_count", 0)),
            "vpin_bucket_volume": _safe_float(bucket_volume, 0.0),
            "vpin_volume_unit": str(volume_unit),
            "vpin_buy_volume": _safe_float(core.get("vpin_buy_volume", 0.0), 0.0),
            "vpin_sell_volume": _safe_float(core.get("vpin_sell_volume", 0.0), 0.0),
            "vpin_total_volume": _safe_float(core.get("vpin_total_volume", 0.0), 0.0),
            "vpin_imbalance_sum": _safe_float(core.get("vpin_imbalance_sum", 0.0), 0.0),
            "vpin_denominator": _safe_float(core.get("vpin_denominator", 0.0), 0.0),
            "vpin_lower_bound": _safe_float(core.get("vpin_lower_bound", 0.0), 0.0),
            "vpin_formula_check_ok": bool(core.get("vpin_formula_check_ok", False)),
            "vpin_last_trade_ts": int(trades_df["timestamp_ms"].iloc[-1]) if not trades_df.empty else None,
        }

        for key in [
            "vpin",
            "vpin_z",
            "vpin_bucket_volume",
            "vpin_buy_volume",
            "vpin_sell_volume",
            "vpin_total_volume",
            "vpin_imbalance_sum",
            "vpin_denominator",
            "vpin_lower_bound",
        ]:
            out[key] = _safe_float(out.get(key, 0.0), 0.0)

        # Hard invariant on final returned VPIN value.
        vpin_ok_final = bool(out.get("vpin_ok", False))
        bucket_count_final = max(0, int(out.get("vpin_bucket_count", 0)))
        denom_final = _safe_float(out.get("vpin_denominator", 0.0), 0.0)
        final_vpin = float(np.clip(_safe_float(out.get("vpin", 0.0), 0.0), 0.0, 1.0))
        out["vpin"] = final_vpin

        formula_ok_final = True
        if vpin_ok_final and bucket_count_final > 0:
            lower_bound_final = _safe_float(out.get("vpin_lower_bound", 0.0), 0.0)
            formula_ok_final = bool(final_vpin >= (lower_bound_final - 1e-9))

        if vpin_ok_final and denom_final > 1e-12:
            expected_vpin = _safe_float(out.get("vpin_imbalance_sum", 0.0), 0.0) / denom_final
            expected_vpin = float(np.clip(_safe_float(expected_vpin, 0.0), 0.0, 1.0))
            out["vpin"] = expected_vpin
            output_match = abs(float(out["vpin"]) - expected_vpin) <= 1e-9
            if not output_match:
                out["vpin_formula_check_ok"] = False
                out["vpin_reason"] = "formula_output_mismatch"
            formula_ok_final = bool(formula_ok_final and output_match)

        if out.get("vpin_reason") == "formula_output_mismatch":
            out["vpin_formula_check_ok"] = False
        else:
            out["vpin_formula_check_ok"] = bool(formula_ok_final and bool(out.get("vpin_formula_check_ok", False)))

        return out
    except Exception:
        return neutral
