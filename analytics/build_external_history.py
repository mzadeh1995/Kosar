# ==============================================================================
# analytics/build_external_history.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
import external_history


def _resolve_symbol(symbol: str) -> str:
    mapped = CONFIG.get("BINANCE_SYMBOL_MAP", {}).get(symbol)
    if mapped:
        return str(mapped).strip().upper()
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def _default_output_dir() -> str:
    return os.path.join(str(CONFIG.get("DATA_DIR", str(ROOT / "data"))), "external")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external funding and reconstructed VPIN histories.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Internal symbols, e.g. BTC-USD ETH-USD")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument("--skip-vpin", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--micro-history-file", default=CONFIG.get("MICROSTRUCTURE_HISTORY_FILE"))
    parser.add_argument("--data-dir", default=CONFIG.get("DATASET_DATA_DIR"))
    parser.add_argument("--output-dir", default=_default_output_dir())
    args = parser.parse_args()

    if not args.validate_only and not args.start:
        parser.error("--start is required unless --validate-only is used")

    symbol_map = {sym: _resolve_symbol(sym) for sym in args.symbols}
    end = args.end or external_history._parse_time(None).isoformat()

    if not args.validate_only:
        for internal_symbol, binance_symbol in symbol_map.items():
            if not args.skip_funding:
                result = external_history.build_funding_history(
                    binance_symbol,
                    args.start,
                    end,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                )
                print(f"funding {internal_symbol} -> {result['csv_path']}")
            if not args.skip_vpin:
                result = external_history.build_reconstructed_vpin_history(
                    binance_symbol,
                    args.start,
                    end,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                )
                print(f"vpin {internal_symbol} -> {result['csv_path']}")

    report = external_history.validate_vpin_reconstruction(
        list(args.symbols),
        start=args.start,
        end=args.end,
        micro_history_file=args.micro_history_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        symbol_map=symbol_map,
    )
    print(f"validation status={report.get('status')} report={report.get('report_path')}")
    for symbol, item in report.get("symbols", {}).items():
        print(
            f"validation {symbol}: n={item.get('n_common')} "
            f"pearson={item.get('pearson')} spearman={item.get('spearman')} "
            f"verdict={item.get('verdict')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
