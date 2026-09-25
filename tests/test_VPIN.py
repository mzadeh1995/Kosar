# ==============================================================================
# ✅ tests/test_VPIN.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Unit tests for VPIN feature engineering utilities.
# ==============================================================================

import math

import pandas as pd

import VPIN


def test_normalize_agg_trades_and_trade_direction_classification():
    raw = [
        {"p": "100", "q": "2", "T": 2000, "m": False},  # buyer initiated
        {"p": "100", "q": "3", "T": 3000, "m": True},   # seller initiated
    ]

    df = VPIN.normalize_agg_trades(raw)
    assert list(df["timestamp_ms"]) == [2000, 3000]
    assert bool(df.iloc[0]["is_buyer_initiated"]) is True
    assert bool(df.iloc[1]["is_buyer_initiated"]) is False

    assert df.iloc[0]["buyer_initiated_volume_quote"] == 200.0
    assert df.iloc[0]["seller_initiated_volume_quote"] == 0.0
    assert df.iloc[1]["buyer_initiated_volume_quote"] == 0.0
    assert df.iloc[1]["seller_initiated_volume_quote"] == 300.0


def test_build_volume_buckets_with_trade_splitting_across_boundaries():
    trades_df = pd.DataFrame(
        [
            {
                "timestamp_ms": 1,
                "buyer_initiated_volume_quote": 60.0,
                "seller_initiated_volume_quote": 0.0,
                "buyer_initiated_volume_base": 0.6,
                "seller_initiated_volume_base": 0.0,
            },
            {
                "timestamp_ms": 2,
                "buyer_initiated_volume_quote": 0.0,
                "seller_initiated_volume_quote": 70.0,
                "buyer_initiated_volume_base": 0.0,
                "seller_initiated_volume_base": 0.7,
            },
            {
                "timestamp_ms": 3,
                "buyer_initiated_volume_quote": 0.0,
                "seller_initiated_volume_quote": 70.0,
                "buyer_initiated_volume_base": 0.0,
                "seller_initiated_volume_base": 0.7,
            },
        ]
    )

    buckets = VPIN.build_volume_buckets(trades_df, bucket_volume=100.0, volume_unit="quote")
    assert len(buckets) == 2

    b1 = buckets.iloc[0]
    b2 = buckets.iloc[1]

    assert math.isclose(float(b1["buy_volume"]), 60.0, rel_tol=1e-9)
    assert math.isclose(float(b1["sell_volume"]), 40.0, rel_tol=1e-9)
    assert math.isclose(float(b1["total_volume"]), 100.0, rel_tol=1e-9)
    assert math.isclose(float(b1["imbalance_abs"]), 20.0, rel_tol=1e-9)

    assert math.isclose(float(b2["buy_volume"]), 0.0, rel_tol=1e-9)
    assert math.isclose(float(b2["sell_volume"]), 100.0, rel_tol=1e-9)
    assert math.isclose(float(b2["total_volume"]), 100.0, rel_tol=1e-9)
    assert math.isclose(float(b2["imbalance_abs"]), 100.0, rel_tol=1e-9)


def test_compute_vpin_formula():
    buckets = pd.DataFrame(
        [
            {"buy_volume": 60.0, "sell_volume": 40.0, "total_volume": 100.0, "imbalance_abs": 20.0, "start_ts": 1, "end_ts": 2},
            {"buy_volume": 0.0, "sell_volume": 100.0, "total_volume": 100.0, "imbalance_abs": 100.0, "start_ts": 2, "end_ts": 3},
        ]
    )
    out = VPIN.compute_vpin_from_buckets(buckets, bucket_volume=100.0, min_buckets=1)

    assert out["vpin_ok"] is True
    assert math.isclose(out["vpin"], 0.6, rel_tol=1e-9)
    assert out["vpin_bucket_count"] == 2
    assert math.isclose(out["vpin_imbalance_sum"], 120.0, rel_tol=1e-9)
    assert math.isclose(out["vpin_denominator"], 200.0, rel_tol=1e-9)
    assert math.isclose(out["vpin_lower_bound"], 0.4, rel_tol=1e-9)
    assert out["vpin_formula_check_ok"] is True


def test_build_vpin_features_buy_sell_imbalance_produces_positive_vpin():
    # Strong buy/sell imbalance across trades -> VPIN must be > 0 when buckets exist.
    raw_trades = [
        {"p": "100", "q": "1", "T": 1000, "m": False},  # buy 100 quote
        {"p": "100", "q": "1", "T": 2000, "m": False},  # buy 100 quote
        {"p": "100", "q": "1", "T": 3000, "m": True},   # sell 100 quote
        {"p": "100", "q": "1", "T": 4000, "m": False},  # buy 100 quote
    ]
    cfg = {
        "VPIN_BUCKET_VOLUME_MODE": "fixed_quote",
        "VPIN_FIXED_BUCKET_VOLUME_QUOTE": 100.0,
        "VPIN_MIN_BUCKETS": 1,
        "VPIN_ZSCORE_WINDOW": 50,
    }

    out = VPIN.build_vpin_features(raw_trades, cfg)
    assert out["vpin_ok"] is True
    assert out["vpin_bucket_count"] > 0
    assert out["vpin_buy_volume"] != out["vpin_sell_volume"]
    assert out["vpin"] > 0.0
    assert out["vpin_formula_check_ok"] is True


def test_vpin_regression_reported_case_values():
    # Regression case matching reported live aggregates where prior output had vpin=0 incorrectly.
    # Here we build synthetic buckets with:
    # imbalance_sum=755936.1941437796, denominator=974742.0430951, lower_bound=0.1287072185668101.
    bucket_count = 20
    bucket_volume = 48737.102154755
    d_pos = 44069.626566534
    d_neg = 31523.992847844

    records = []
    # 10 buckets with strong buy dominance
    for i in range(10):
        buy = (bucket_volume + d_pos) / 2.0
        sell = (bucket_volume - d_pos) / 2.0
        records.append(
            {
                "buy_volume": buy,
                "sell_volume": sell,
                "total_volume": bucket_volume,
                # Intentionally wrong to ensure implementation recomputes imbalance from buy/sell.
                "imbalance_abs": 0.0,
                "start_ts": i,
                "end_ts": i,
            }
        )
    # 10 buckets with sell dominance
    for i in range(10, 20):
        buy = (bucket_volume - d_neg) / 2.0
        sell = (bucket_volume + d_neg) / 2.0
        records.append(
            {
                "buy_volume": buy,
                "sell_volume": sell,
                "total_volume": bucket_volume,
                "imbalance_abs": 0.0,
                "start_ts": i,
                "end_ts": i,
            }
        )

    buckets = pd.DataFrame(records)
    out = VPIN.compute_vpin_from_buckets(buckets, bucket_volume=bucket_volume, min_buckets=1)

    expected_ratio = out["vpin_imbalance_sum"] / out["vpin_denominator"]
    expected_ratio = max(0.0, min(1.0, expected_ratio))

    assert out["vpin"] > 0.0
    assert math.isclose(out["vpin"], expected_ratio, rel_tol=1e-12)
    assert out["vpin_formula_check_ok"] is True


def test_resolve_bucket_volume_dynamic_quote():
    trades_df = pd.DataFrame([{"quantity_quote": 200.0}, {"quantity_quote": 300.0}])
    bucket_volume, unit = VPIN.resolve_bucket_volume(
        trades_df,
        {"VPIN_BUCKET_VOLUME_MODE": "dynamic_quote", "VPIN_BUCKET_COUNT": 5},
    )

    assert unit == "quote"
    assert math.isclose(bucket_volume, 100.0, rel_tol=1e-9)


def test_malformed_inputs_are_safe_and_neutral():
    empty_df = VPIN.normalize_agg_trades(raw_trades=[{"bad": "row"}, "x"])
    assert empty_df.empty

    out = VPIN.build_vpin_features(raw_trades=[{"bad": "row"}], config={})
    assert out["vpin_ok"] is False
    assert out["vpin"] == 0.0
    assert out["vpin_z"] == 0.0
    assert out["vpin_bucket_count"] == 0
