# ==============================================================================
# ✅ tests/test_OFI.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Unit tests for OFI feature engineering utilities.
# ==============================================================================

import math
import asyncio

import OFI


def test_normalize_depth_snapshot_sorts_and_drops_invalid_rows():
    raw = {
        "lastUpdateId": "123",
        "bids": [["100", "2"], ["101", "1.5"], ["bad", "1"], ["99", "-2"]],
        "asks": [["103", "4"], ["102", "3"], ["0", "1"], ["x", "1"]],
    }
    out = OFI.normalize_depth_snapshot(raw)

    assert out["last_update_id"] == 123
    assert out["bids"] == [(101.0, 1.5), (100.0, 2.0)]
    assert out["asks"] == [(102.0, 3.0), (103.0, 4.0)]


def test_compute_l1_ofi_bid_price_up_case():
    prev_snap = {"bids": [["100", "5"]], "asks": [["101", "4"]]}
    curr_snap = {"bids": [["101", "6"]], "asks": [["101", "4"]]}
    assert OFI.compute_l1_ofi(prev_snap, curr_snap) == 6.0


def test_compute_l1_ofi_bid_price_down_case():
    prev_snap = {"bids": [["100", "5"]], "asks": [["101", "4"]]}
    curr_snap = {"bids": [["99", "2"]], "asks": [["101", "4"]]}
    assert OFI.compute_l1_ofi(prev_snap, curr_snap) == -5.0


def test_compute_l1_ofi_ask_price_down_case():
    prev_snap = {"bids": [["100", "5"]], "asks": [["101", "4"]]}
    curr_snap = {"bids": [["100", "5"]], "asks": [["100", "3"]]}
    assert OFI.compute_l1_ofi(prev_snap, curr_snap) == -3.0


def test_compute_l1_ofi_ask_price_up_case():
    prev_snap = {"bids": [["100", "5"]], "asks": [["101", "4"]]}
    curr_snap = {"bids": [["100", "5"]], "asks": [["102", "6"]]}
    assert OFI.compute_l1_ofi(prev_snap, curr_snap) == 4.0


def test_compute_spread_bps():
    snap = {"bids": [["100", "2"]], "asks": [["101", "3"]]}
    spread = OFI.compute_spread_bps(snap)
    expected = (1.0 / 100.5) * 10000.0
    assert math.isclose(spread, expected, rel_tol=1e-9)


def test_depth_imbalance_clipped_to_range():
    only_bids = {"bids": [["100", "10"]], "asks": []}
    only_asks = {"bids": [], "asks": [["101", "10"]]}

    assert OFI.compute_depth_imbalance(only_bids, levels=10) == 1.0
    assert OFI.compute_depth_imbalance(only_asks, levels=10) == -1.0


def test_collect_depth_snapshots_bypasses_depth_cache(monkeypatch):
    import provider

    calls = []

    async def fake_fetch_binance_depth(internal_symbol, limit=None, use_cache=True, force_refresh=False):
        calls.append(
            {
                "internal_symbol": internal_symbol,
                "limit": limit,
                "use_cache": use_cache,
                "force_refresh": force_refresh,
            }
        )
        idx = len(calls)
        return {
            "lastUpdateId": idx,
            "bids": [["100", str(10 + idx)]],
            "asks": [["101", str(20 + idx)]],
        }

    monkeypatch.setattr(provider, "fetch_binance_depth", fake_fetch_binance_depth)

    cfg = {
        "OFI_SNAPSHOT_COUNT": 3,
        "OFI_SNAPSHOT_INTERVAL_SECONDS": 0.0,
        "OFI_DEPTH_LIMIT": 100,
        "OFI_BYPASS_DEPTH_CACHE": True,
    }
    snapshots = asyncio.run(provider.collect_depth_snapshots("BTC-USD", cfg))

    assert len(snapshots) == 3
    assert len({snap.get("lastUpdateId") for snap in snapshots}) == 3
    assert all(call["use_cache"] is False for call in calls)


def test_ofi_l1_norm_uses_quantity_denominator_not_usd_liquidity():
    snap_a = {
        "lastUpdateId": 1,
        "bids": [["100", "10"]],
        "asks": [["101", "20"]],
    }
    snap_b = {
        "lastUpdateId": 2,
        "bids": [["100", "15"]],
        "asks": [["101", "18"]],
    }

    features = OFI.build_ofi_features([snap_a, snap_b], {"OFI_LEVELS": 10, "OFI_ZSCORE_WINDOW": 50})
    ofi_l1 = OFI.compute_l1_ofi(snap_a, snap_b)
    qty_denom = 15.0 + 18.0
    expected_norm = max(-1.0, min(1.0, ofi_l1 / qty_denom))

    usd_liquidity = OFI.compute_liquidity_usd(snap_b, levels=10)
    usd_norm = ofi_l1 / usd_liquidity if usd_liquidity > 0 else 0.0

    assert math.isclose(features["ofi_l1"], ofi_l1, rel_tol=1e-9)
    assert math.isclose(features["ofi_l1_norm"], expected_norm, rel_tol=1e-9)
    assert math.isclose(features["ofi_l1_norm_denom_qty"], qty_denom, rel_tol=1e-9)
    assert not math.isclose(features["ofi_l1_norm"], usd_norm, rel_tol=1e-9)


def test_ofi_l1_norm_zero_denominator_is_safe():
    snap_a = {
        "lastUpdateId": 1,
        "bids": [["100", "0"]],
        "asks": [["101", "0"]],
    }
    snap_b = {
        "lastUpdateId": 2,
        "bids": [["100", "0"]],
        "asks": [["101", "0"]],
    }

    features = OFI.build_ofi_features([snap_a, snap_b], {"OFI_LEVELS": 10})
    assert features["ofi_l1_norm"] == 0.0
    assert features["ofi_l1_norm_denom_qty"] == 0.0


def test_single_snapshot_marks_ofi_not_ok():
    single_snapshot = {
        "lastUpdateId": 1,
        "bids": [["100", "10"]],
        "asks": [["101", "20"]],
    }
    features = OFI.build_ofi_features([single_snapshot], {"OFI_LEVELS": 10})
    assert features["ofi_ok"] is False
    assert features["ofi_reason"] == "insufficient_snapshots_for_l1"
    assert features["ofi_l1"] == 0.0
    assert features["ofi_l1_norm"] == 0.0


def test_malformed_inputs_are_safe_and_neutral():
    assert OFI.compute_l1_ofi(None, None) == 0.0
    assert OFI.compute_spread_bps({"bids": [], "asks": []}) == 0.0
    assert OFI.compute_liquidity_usd({"bids": "bad", "asks": "bad"}) == 0.0

    features = OFI.build_ofi_features(depth_snapshots=[], config={"OFI_LEVELS": 10})
    assert features["ofi_ok"] is False
    assert features["ofi_l1"] == 0.0
    assert features["ofi_l1_norm"] == 0.0
    assert features["ofi_l1_z"] == 0.0
