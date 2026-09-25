# ==============================================================================
# analytics/select_features.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# CLI runner for offline Random Forest feature-selection diagnostics only.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from primary_features import load_symbol_dataset, select_features_cv


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
DIAGNOSTIC_NOTE = (
    "This list is diagnostic only; binding feature selection in phase 3-b must be "
    "performed per fold and only with that fold's train rows."
)


def _dataset_path(dataset_dir: Path, symbol: str, timeframe: str) -> Path:
    stem = f"{symbol}_{timeframe}"
    for suffix in [".csv", ".parquet"]:
        candidate = dataset_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset file not found for {symbol} {timeframe} in {dataset_dir}")


def _series_to_float_dict(series: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in series.items()}


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _aggregate_symbol_results(symbol_results: dict[str, dict], config: dict) -> dict:
    importance_by_symbol = {
        symbol: result["mean_importances"]
        for symbol, result in symbol_results.items()
        if int(result["valid_folds"]) > 0
    }
    if not importance_by_symbol:
        return {
            "mean_importance": {},
            "mean_rank": {},
            "selected_features_diagnostic": [],
            "note": DIAGNOSTIC_NOTE,
        }

    importance_df = pd.DataFrame(importance_by_symbol)
    mean_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    ranks = importance_df.rank(axis=0, ascending=False, method="average")
    mean_rank = ranks.mean(axis=1).sort_values(ascending=True)

    keep_ratio = float(config.get("PRIMARY_FS_KEEP_RATIO", 0.5))
    keep_count = max(1, int(math.ceil(len(mean_importance) * keep_ratio)))
    selected = list(mean_importance.head(keep_count).index)

    return {
        "mean_importance": _series_to_float_dict(mean_importance),
        "mean_rank": _series_to_float_dict(mean_rank),
        "selected_features_diagnostic": selected,
        "note": DIAGNOSTIC_NOTE,
    }


def run(symbols: list[str], timeframe: str, dataset_dir: Path, config: dict) -> dict:
    symbol_results: dict[str, dict] = {}
    serializable_symbols: dict[str, dict] = {}

    for symbol in symbols:
        path = _dataset_path(dataset_dir, symbol, timeframe)
        df = load_symbol_dataset(path, dataset_dir=dataset_dir)
        result = select_features_cv(df, config)
        symbol_results[symbol] = result

        top5 = result["mean_importances"].sort_values(ascending=False).head(5)
        top5_text = ", ".join(f"{name}={value:.6f}" for name, value in top5.items())
        print(f"{symbol} | events={result['event_rows']} usable={result['usable_rows']} | valid_folds={result['valid_folds']}")
        print(f"  top5: {top5_text}")
        print(f"  non_positive: {result['non_positive_features']}")
        print(f"  selected: {result['selected_features']}")

        serializable_symbols[symbol] = {
            "dataset_path": str(path),
            "events": int(result["event_rows"]),
            "usable_rows": int(result["usable_rows"]),
            "dropped_rows": int(result["dropped_rows"]),
            "valid_folds": int(result["valid_folds"]),
            "mean_importances": _series_to_float_dict(result["mean_importances"].sort_values(ascending=False)),
            "non_positive_features": list(result["non_positive_features"]),
            "selected_features": list(result["selected_features"]),
            "fold_used_ranges": list(result["fold_used_ranges"]),
            "perm_weighted": result["perm_weighted"],
            "perm_weighted_by_fold": list(result["perm_weighted_by_fold"]),
        }

    aggregate = _aggregate_symbol_results(symbol_results, config)
    print("\nAggregate mean rank:")
    for feature, rank in aggregate["mean_rank"].items():
        print(f"  {feature}: {rank:.3f}")
    print(f"\nAggregate selected diagnostic: {aggregate['selected_features_diagnostic']}")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "sklearn_version": sklearn.__version__,
        "symbols": serializable_symbols,
        "aggregate": aggregate,
        "params": {
            "PRIMARY_CV_SPLITS": int(config.get("PRIMARY_CV_SPLITS", 5)),
            "PRIMARY_RANDOM_SEED": int(config.get("PRIMARY_RANDOM_SEED", 41)),
            "PRIMARY_RF_PARAMS": dict(config.get("PRIMARY_RF_PARAMS", {})),
            "PRIMARY_FS_VAL_FRACTION": float(config.get("PRIMARY_FS_VAL_FRACTION", 0.2)),
            "PRIMARY_FS_PERM_REPEATS": int(config.get("PRIMARY_FS_PERM_REPEATS", 5)),
            "PRIMARY_FS_KEEP_RATIO": float(config.get("PRIMARY_FS_KEEP_RATIO", 0.5)),
        },
        "seed": int(config.get("PRIMARY_RANDOM_SEED", 41)),
        "perm_weighted": {
            "all_valid_folds_weighted": all(
                bool(v)
                for result in serializable_symbols.values()
                for v in result["perm_weighted_by_fold"]
            )
            if any(result["perm_weighted_by_fold"] for result in serializable_symbols.values())
            else None
        },
    }

    out_dir = Path(config.get("PRIMARY_MODEL_DIR", ROOT / "data" / "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_selection_summary_{timeframe}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\nSaved summary: {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RF + permutation feature-selection diagnostics.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframe", default=CONFIG.get("DATASET_DEFAULT_TIMEFRAME", "1h"))
    parser.add_argument("--dataset-dir", default=CONFIG.get("DATASET_OUTPUT_DIR"))
    args = parser.parse_args()

    run(
        symbols=[str(symbol).upper() for symbol in args.symbols],
        timeframe=str(args.timeframe),
        dataset_dir=Path(args.dataset_dir),
        config=dict(CONFIG),
    )


if __name__ == "__main__":
    main()
