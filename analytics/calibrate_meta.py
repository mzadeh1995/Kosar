# ==============================================================================
# analytics/calibrate_meta.py
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

from calibration import (  # noqa: E402
    CANONICAL_META_OOF_DIR,
    CANONICAL_SYMBOLS,
    CalibrationError,
    run_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the offline 4h meta/primary OOF scores and evaluate the "
            "pre-registered additivity gate."
        )
    )
    parser.add_argument("--symbols", nargs="+", default=list(CANONICAL_SYMBOLS))
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--meta-oof-dir", default=str(CANONICAL_META_OOF_DIR))
    parser.add_argument("--output-dir", default=str(CANONICAL_META_OOF_DIR))
    args = parser.parse_args()

    try:
        summary = run_calibration(
            symbols=args.symbols,
            timeframe=args.timeframe,
            meta_oof_dir=args.meta_oof_dir,
            output_dir=args.output_dir,
        )
    except CalibrationError as exc:
        details = exc.details if isinstance(exc.details, dict) else {}
        rollback_succeeded = details.get("rollback_succeeded")
        managed_outputs_state = (
            "unknown_or_partial" if rollback_succeeded is False else "none"
        )
        failure = {
            "status": "failed",
            "error_code": exc.code,
            "message": str(exc),
            "details": exc.details,
            "managed_outputs_state": managed_outputs_state,
            "managed_outputs_written": (
                None if managed_outputs_state == "unknown_or_partial" else False
            ),
        }
        print(json.dumps(failure, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
        return 2

    transaction = summary.get("transaction", {})
    concise = {
        "status": (
            "complete"
            if transaction.get("cleanup_success") is True
            else "committed_with_cleanup_warnings"
        ),
        "is_real_data_run": summary.get("is_real_data_run"),
        "symbols": summary.get("symbols"),
        "selected_methods": summary.get("selected_methods"),
        "additivity_gate": summary.get("additivity_gate"),
        "output_paths": summary.get("output_paths"),
        "report_sha256": summary.get("report_sha256_after_commit"),
        "report_size_bytes": summary.get("report_size_bytes_after_commit"),
        "transaction": {
            "report_committed_last": transaction.get("report_committed_last"),
            "commit_order": transaction.get("commit_order"),
            "staging_validation": transaction.get("staging_validation"),
            "cleanup_success": transaction.get("cleanup_success"),
            "cleanup_errors": transaction.get("cleanup_errors"),
            "remaining_debris": transaction.get("remaining_debris"),
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
