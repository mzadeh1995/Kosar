# ==============================================================================
# ✅ tests/scripts/live_microstructure_smoke.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Live smoke test for microstructure feature retrieval and sanity checks.
# ==============================================================================

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from provider import build_microstructure_features, close_binance_http_session


async def _run_symbol(symbol: str) -> dict:
    features = await build_microstructure_features(symbol, CONFIG)
    print(json.dumps({"symbol": symbol, "features": features}, ensure_ascii=False, indent=2))

    assert features.get("microstructure_ok") is not None, f"{symbol}: microstructure_ok is None"

    ofi_depth_imbalance = float(features.get("ofi_depth_imbalance", 0.0) or 0.0)
    assert -1.0 <= ofi_depth_imbalance <= 1.0, f"{symbol}: ofi_depth_imbalance out of range: {ofi_depth_imbalance}"

    ofi_spread_bps = float(features.get("ofi_spread_bps", 0.0) or 0.0)
    assert ofi_spread_bps >= 0.0, f"{symbol}: ofi_spread_bps negative: {ofi_spread_bps}"

    vpin = float(features.get("vpin", 0.0) or 0.0)
    assert 0.0 <= vpin <= 1.0, f"{symbol}: vpin out of range: {vpin}"

    bucket_count = int(features.get("vpin_bucket_count", 0) or 0)
    buy_volume = float(features.get("vpin_buy_volume", 0.0) or 0.0)
    sell_volume = float(features.get("vpin_sell_volume", 0.0) or 0.0)

    if bucket_count > 0 and abs(buy_volume - sell_volume) > 1e-9:
        assert vpin > 0.0, (
            f"{symbol}: vpin must be > 0 when buy/sell differ and buckets exist | "
            f"buy={buy_volume}, sell={sell_volume}, buckets={bucket_count}, vpin={vpin}"
        )

    return features


async def main() -> None:
    try:
        for symbol in ["BTC-USD", "ETH-USD"]:
            await _run_symbol(symbol)
    finally:
        await close_binance_http_session()


if __name__ == "__main__":
    asyncio.run(main())
