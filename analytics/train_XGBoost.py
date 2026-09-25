# ==============================================================================
# analytics/train_XGBoost.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# CLI runner for offline XGBoost primary-model training only.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from primary_features import load_symbol_dataset
from XGBoost import run_purged_cv, save_oof, train_final_model, xgboost


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]


def _dataset_path(dataset_dir: Path, symbol: str, timeframe: str) -> Path:
    stem = f"{symbol}_{timeframe}"
    for suffix in [".csv", ".parquet"]:
        candidate = dataset_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Dataset file not found for {symbol} {timeframe} in {dataset_dir}")


def _json_default(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _threshold_diagnostic(folds: list[dict], thresholds: list[float]) -> dict:
    rows = []
    for threshold in thresholds:
        key = f"{float(threshold):.2f}"
        model_sum = sum(float(fold["B"]["thresholds"][key]["sum_net"]) for fold in folds)
        baseline_sum = sum(float(fold["B"]["baseline"]["sum_net"]) for fold in folds)
        trade_count = sum(int(fold["B"]["thresholds"][key]["count"]) for fold in folds)
        vertical_count = sum(
            int(fold["B"]["thresholds"][key]["count"])
            * float(fold["B"]["thresholds"][key]["vertical_rate"] or 0.0)
            for fold in folds
        )
        rows.append(
            {
                "threshold": float(threshold),
                "model_sum_net": float(model_sum),
                "baseline_sum_net": float(baseline_sum),
                "vertical_rate": float(vertical_count / trade_count) if trade_count else None,
                "trade_count": int(trade_count),
            }
        )
    return max(rows, key=lambda row: row["model_sum_net"]) if rows else {}


def run(symbols: list[str], timeframe: str, dataset_dir: Path, config: dict) -> dict:
    symbol_rows: list[dict] = []
    for symbol in symbols:
        path = _dataset_path(dataset_dir, symbol, timeframe)
        df = load_symbol_dataset(path, dataset_dir=dataset_dir)
        cv_result = run_purged_cv(df, config)
        oof_path = save_oof(cv_result["oof"], symbol, timeframe, config)

        final_config = dict(config)
        final_config["PRIMARY_TRAIN_SYMBOL"] = symbol
        final_config["PRIMARY_TRAIN_TIMEFRAME"] = timeframe
        final_config["PRIMARY_CV_SUMMARY"] = cv_result["summary"]
        final_result = train_final_model(df, final_config)

        counts = [len(fold["selected_features_b"]) for fold in cv_result["folds"]]
        diagnostic = _threshold_diagnostic(
            cv_result["folds"],
            list(config.get("PRIMARY_PROB_THRESHOLDS", [])),
        )
        row = {
            "symbol": symbol,
            "auc_mean_a": cv_result["summary"]["A"]["mean_auc"],
            "auc_mean_b": cv_result["summary"]["B"]["mean_auc"],
            "auc_worst_b": cv_result["summary"]["B"]["worst_auc"],
            "selected_feature_count_range_b": [min(counts), max(counts)] if counts else None,
            "best_threshold_diagnostic": diagnostic,
            "valid_folds": len(cv_result["folds"]),
            "oof_path": str(oof_path),
            "model_path": str(final_result["model_path"]),
            "metadata_path": str(final_result["metadata_path"]),
        }
        symbol_rows.append(row)
        print(
            f"{symbol} | A mean={row['auc_mean_a']} | B mean={row['auc_mean_b']} "
            f"| B worst={row['auc_worst_b']} | features={row['selected_feature_count_range_b']} "
            f"| threshold={diagnostic.get('threshold')} | model net={diagnostic.get('model_sum_net')} "
            f"| baseline net={diagnostic.get('baseline_sum_net')} "
            f"| vertical={diagnostic.get('vertical_rate')}"
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "params": {
            "PRIMARY_XGB_PARAMS": dict(config.get("PRIMARY_XGB_PARAMS", {})),
            "PRIMARY_RF_PARAMS": dict(config.get("PRIMARY_RF_PARAMS", {})),
            "PRIMARY_CV_SPLITS": int(config.get("PRIMARY_CV_SPLITS", 5)),
            "PRIMARY_FS_KEEP_RATIO": float(config.get("PRIMARY_FS_KEEP_RATIO", 0.5)),
            "PRIMARY_PROB_THRESHOLDS": list(config.get("PRIMARY_PROB_THRESHOLDS", [])),
        },
        "seed": int(config.get("PRIMARY_RANDOM_SEED", 41)),
        "ab_policy": (
            "A/B is diagnostic only. Arm B was fixed in advance and no model or "
            "feature choice was made from test-fold metrics."
        ),
        "symbols": symbol_rows,
    }
    out_dir = Path(config.get("PRIMARY_MODEL_DIR", ROOT / "data" / "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"primary_training_summary_{timeframe}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Saved training summary: {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline XGBoost primary-model training.")
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
