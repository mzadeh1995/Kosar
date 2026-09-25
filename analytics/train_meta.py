# ==============================================================================
# analytics/train_meta.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meta_model import (  # noqa: E402
    CANONICAL_DATASET_DIR,
    CANONICAL_EXTERNAL_DIR,
    CANONICAL_OUTPUT_DIR,
    CANONICAL_SYMBOLS,
    run_meta_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the offline 4h CatBoost meta-label model with past-only thresholds."
    )
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    parser.add_argument("--dataset-dir", default=str(CANONICAL_DATASET_DIR))
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--external-dir", default=str(CANONICAL_EXTERNAL_DIR))
    parser.add_argument("--hmm-dir", default=None)
    parser.add_argument("--output-dir", default=str(CANONICAL_OUTPUT_DIR))
    args = parser.parse_args()

    summary = run_meta_training(
        symbols=args.symbols,
        dataset_dir=args.dataset_dir,
        external_dir=args.external_dir,
        hmm_dir=args.hmm_dir,
        output_dir=args.output_dir,
        timeframe=args.timeframe,
    )
    concise = {
        "is_real_data_run": summary["is_real_data_run"],
        "is_canonical_output": summary["is_canonical_output"],
        "symbols": summary["symbols"],
        "feature_columns": summary["feature_columns"],
        "fallback_counts": summary["fallback_counts"],
        "baseline_anchor": summary["walk_forward"]["baseline_anchor"],
        "tau_final_raw_diagnostic": summary["walk_forward"]["tau_final_raw_diagnostic"],
        "output_paths": summary["output_paths"],
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
