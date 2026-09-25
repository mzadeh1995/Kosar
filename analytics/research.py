# ==============================================================================
# 🔬 analytics/research.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Offline runner for feature isolation diagnostics (not part of live trading).
# ==============================================================================

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import CONFIG as _PROJECT_CONFIG
except Exception as exc:  # pragma: no cover - environment dependent
    _CONFIG_IMPORT_ERROR = exc
    CONFIG = {
        "UNIVERSE": ["BTC-USD", "ETH-USD", "BNB-USD"],
        "TIMEFRAME": "1h",
        "FEATURE_ISOLATION_OUTPUT_DIR": str(ROOT / "log" / "feature_isolation"),
        "MICROSTRUCTURE_HISTORY_FILE": str(ROOT / "log" / "microstructure_history.jsonl"),
        "RESEARCH_DEFAULT_PERIOD": "1mo",
        "RESEARCH_MIN_ROWS": 200,
        "FEATURE_ISOLATION_HORIZON_BARS": [1, 3, 6, 12],
    }
else:
    _CONFIG_IMPORT_ERROR = None
    CONFIG = dict(_PROJECT_CONFIG)

try:
    import numpy as np
    import pandas as pd
except Exception as exc:  # pragma: no cover - environment dependent
    np = None
    pd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


REQUIRED_FEATURE_COLUMNS = ["ofi_l1_norm", "ofi_l1_z", "ofi_depth_imbalance", "vpin", "vpin_z"]
DIAGNOSTIC_COLUMNS = [
    "vpin_formula_check_ok",
    "vpin_denominator",
    "vpin_imbalance_sum",
    "vpin_lower_bound",
    "vpin_ok",
    "ofi_ok",
    "microstructure_ok",
]
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _dependencies_ready() -> bool:
    return (np is not None) and (pd is not None)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isfinite(out):
        return out
    return None


def _parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None:
        return pd.NaT

    # Numeric UNIX timestamps (seconds or milliseconds).
    numeric_candidate = None
    if isinstance(value, (int, float)):
        numeric_candidate = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return pd.NaT
        try:
            numeric_candidate = float(s)
        except Exception:
            numeric_candidate = None

    if numeric_candidate is not None and np.isfinite(numeric_candidate):
        try:
            abs_v = abs(numeric_candidate)
            unit = "ms" if abs_v >= 1e11 else "s"
            return pd.to_datetime(numeric_candidate, unit=unit, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _nested_dict_candidates(row: dict) -> list[dict]:
    out: list[dict] = [row]
    for key in ("market_data", "microstructure", "features", "data", "payload"):
        nested = row.get(key)
        if isinstance(nested, dict):
            out.append(nested)
    return out


def _extract_symbol(row: dict) -> str | None:
    for blob in _nested_dict_candidates(row):
        for key in ("symbol", "sym", "internal_symbol"):
            value = blob.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_timestamp(row: dict) -> pd.Timestamp | pd.NaT:
    for blob in _nested_dict_candidates(row):
        for key in ("ts", "timestamp", "datetime", "time"):
            ts = _parse_timestamp(blob.get(key))
            if not pd.isna(ts):
                return ts
    return pd.NaT


def _extract_value(row: dict, key: str) -> Any:
    for blob in _nested_dict_candidates(row):
        if key in blob:
            return blob.get(key)
    return None


def load_microstructure_history(path: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    symbol_list = [str(s).strip() for s in symbols if str(s).strip()]
    symbol_set = set(symbol_list)
    empty = pd.DataFrame(columns=REQUIRED_FEATURE_COLUMNS + DIAGNOSTIC_COLUMNS)
    per_symbol_rows: dict[str, list[dict[str, Any]]] = {sym: [] for sym in symbol_list}

    history_path = Path(path)
    if not history_path.exists() or not history_path.is_file():
        return {sym: empty.copy() for sym in symbol_list}

    with history_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue

            symbol = _extract_symbol(row)
            if symbol not in symbol_set:
                continue

            ts = _extract_timestamp(row)
            if pd.isna(ts):
                continue

            rec: dict[str, Any] = {"timestamp": ts}
            for col in REQUIRED_FEATURE_COLUMNS:
                rec[col] = _safe_float(_extract_value(row, col))

            for col in DIAGNOSTIC_COLUMNS:
                v = _extract_value(row, col)
                if col.endswith("_ok") and isinstance(v, (bool, int)):
                    rec[col] = bool(v)
                else:
                    fv = _safe_float(v)
                    rec[col] = fv if fv is not None else v

            per_symbol_rows[symbol].append(rec)

    out: dict[str, pd.DataFrame] = {}
    for sym in symbol_list:
        rows = per_symbol_rows.get(sym, [])
        if not rows:
            out[sym] = empty.copy()
            continue

        df = pd.DataFrame(rows)
        if "timestamp" not in df.columns:
            out[sym] = empty.copy()
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]

        for col in REQUIRED_FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df[REQUIRED_FEATURE_COLUMNS] = df[REQUIRED_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

        # Feature isolation requires valid historical microstructure rows.
        finite_mask = df[REQUIRED_FEATURE_COLUMNS].notna().all(axis=1)
        df = df.loc[finite_mask]

        keep_cols = REQUIRED_FEATURE_COLUMNS + [c for c in DIAGNOSTIC_COLUMNS if c in df.columns]
        out[sym] = df[keep_cols].copy()

    return out


async def load_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    if not _dependencies_ready():
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    from provider import binance_download_single

    raw = await binance_download_single(symbol, period, interval)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    missing = [c for c in OHLCV_COLUMNS if c not in raw.columns]
    if missing:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = raw[OHLCV_COLUMNS].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
        except Exception:
            pass

    df = df[~df.index.isna()]
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    return df


async def run_research_for_symbol(
    symbol: str,
    microstructure_df: pd.DataFrame,
    period: str,
    interval: str,
    config: dict,
    min_rows: int,
) -> dict:
    if not _dependencies_ready():
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "missing_dependencies",
            "rows_available": 0,
        }

    rows_available = int(microstructure_df.shape[0]) if isinstance(microstructure_df, pd.DataFrame) else 0
    if rows_available < max(1, int(min_rows)):
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "insufficient_microstructure_history",
            "rows_available": rows_available,
        }

    ohlcv_df = await load_ohlcv(symbol, period, interval)
    if ohlcv_df.empty:
        return {
            "symbol": symbol,
            "status": "skipped",
            "reason": "no_ohlcv_data",
            "rows_available": rows_available,
        }

    report = _run_feature_isolation_report_safe(symbol, ohlcv_df, microstructure_df, config)
    return {
        "symbol": symbol,
        "status": "ok",
        "rows_available": rows_available,
        "report": report,
    }


def _run_feature_isolation_report_safe(symbol: str, ohlcv_df: pd.DataFrame, microstructure_df: pd.DataFrame, config: dict) -> dict:
    from feature_isolation import run_feature_isolation_report

    report = run_feature_isolation_report(
        symbol=symbol,
        ohlcv_df=ohlcv_df,
        microstructure_df=microstructure_df,
        config=config,
    )
    return report


def _top_ic_results(report: dict, top_k: int = 5) -> list[str]:
    ic_table = report.get("ic_table", {}) if isinstance(report, dict) else {}
    scored: list[tuple[float, str]] = []

    for feat, targets in ic_table.items():
        if not isinstance(targets, dict):
            continue
        for target, metrics in targets.items():
            if not isinstance(metrics, dict):
                continue
            if str(metrics.get("status")) != "ok":
                continue
            p = abs(float(metrics.get("pearson_ic", 0.0) or 0.0))
            s = abs(float(metrics.get("spearman_ic", 0.0) or 0.0))
            score = max(p, s)
            line = (
                f"{feat} -> {target} | abs(IC)={score:.4f} "
                f"(pearson={float(metrics.get('pearson_ic', 0.0) or 0.0):.4f}, "
                f"spearman={float(metrics.get('spearman_ic', 0.0) or 0.0):.4f}, "
                f"n={int(metrics.get('n', 0) or 0)})"
            )
            scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored[:max(1, int(top_k))]]


def _print_symbol_report(symbol: str, result: dict) -> None:
    print("\n" + "=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Rows available: {int(result.get('rows_available', 0) or 0)}")

    if result.get("status") != "ok":
        print(f"SKIPPED: {result.get('reason', 'unknown')}")
        return

    report = result.get("report", {}) if isinstance(result.get("report"), dict) else {}
    coverage = report.get("feature_coverage", {})
    print(f"Feature coverage: {json.dumps(coverage, ensure_ascii=False)}")

    json_path = report.get("output_json")
    print(f"Report JSON path: {json_path}")

    csv_path = None
    if isinstance(json_path, str) and json_path.strip():
        candidate = Path(json_path).with_suffix(".csv")
        if candidate.exists():
            csv_path = str(candidate)
    print(f"Report CSV path: {csv_path if csv_path else 'N/A'}")

    top_ic = _top_ic_results(report, top_k=5)
    if top_ic:
        print("Top IC results:")
        for line in top_ic:
            print(f"- {line}")
    else:
        print("Top IC results: N/A")

    print(f"Verdict per feature: {json.dumps(report.get('verdict', {}), ensure_ascii=False)}")
    warnings = report.get("warnings", [])
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"- {w}")


def _make_synthetic_ohlcv_and_microstructure(rows: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = max(120, int(rows))
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=rows, freq="h", tz="UTC")

    rng = np.random.default_rng(42)
    ret = rng.normal(0.0, 0.004, rows)
    close = 100.0 * np.cumprod(1.0 + ret)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0008, 0.0006, rows)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0008, 0.0006, rows)))
    volume = np.abs(rng.normal(1500.0, 250.0, rows))

    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )

    ret_s = pd.Series(close, index=idx).pct_change().fillna(0.0)
    noise = pd.Series(rng.normal(0.0, 0.08, rows), index=idx)

    ofi_l1_norm = (ret_s.shift(1).fillna(0.0) * 25.0 + noise).clip(-1.0, 1.0)
    ofi_l1_z = ((ofi_l1_norm - ofi_l1_norm.rolling(60, min_periods=20).mean()) /
                ofi_l1_norm.rolling(60, min_periods=20).std()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    ofi_depth_imbalance = (0.6 * ofi_l1_norm + 0.4 * noise).clip(-1.0, 1.0)

    vpin_raw = (ret_s.abs().rolling(12, min_periods=4).mean().fillna(0.0) * 80.0) + np.abs(noise) * 0.2
    vpin = pd.Series(vpin_raw, index=idx).clip(0.0, 1.0)
    vpin_z = ((vpin - vpin.rolling(60, min_periods=20).mean()) /
              vpin.rolling(60, min_periods=20).std()).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    micro = pd.DataFrame(
        {
            "ofi_l1_norm": ofi_l1_norm.astype(float),
            "ofi_l1_z": ofi_l1_z.astype(float),
            "ofi_depth_imbalance": ofi_depth_imbalance.astype(float),
            "vpin": vpin.astype(float),
            "vpin_z": vpin_z.astype(float),
            "vpin_formula_check_ok": True,
            "vpin_ok": True,
            "ofi_ok": True,
            "microstructure_ok": True,
        },
        index=idx,
    )

    return ohlcv, micro


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline feature isolation research runner")
    parser.add_argument("--symbols", nargs="+", help="Internal symbols, e.g. BTC-USD ETH-USD")
    parser.add_argument("--history-file", type=str, help="JSONL path for historical microstructure rows")
    parser.add_argument("--period", type=str, help="Price history period, e.g. 1mo")
    parser.add_argument("--interval", type=str, help="Price interval, e.g. 1h")
    parser.add_argument("--min-rows", type=int, help="Minimum microstructure rows per symbol")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run in-memory synthetic smoke diagnostics")
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    universe = CONFIG.get("UNIVERSE", [])
    default_symbols = [str(s) for s in universe[:3]] if isinstance(universe, list) else []
    symbols = args.symbols if args.symbols else default_symbols
    symbols = [str(s).strip() for s in symbols if str(s).strip()]

    default_history = str(
        CONFIG.get("MICROSTRUCTURE_HISTORY_FILE", ROOT / "log" / "microstructure_history.jsonl")
    )
    history_file = str(args.history_file or default_history)
    period = str(args.period or CONFIG.get("RESEARCH_DEFAULT_PERIOD", "1mo"))
    interval = str(args.interval or CONFIG.get("TIMEFRAME", "1h"))
    min_rows = int(args.min_rows if args.min_rows is not None else CONFIG.get("RESEARCH_MIN_ROWS", 200))

    if not symbols:
        print("No symbols provided and CONFIG['UNIVERSE'] is empty.")
        return 0

    print("Offline Feature Isolation Research Runner")
    print(f"Symbols: {symbols}")
    print(f"Period: {period}")
    print(f"Interval: {interval}")
    print(f"Min rows: {min_rows}")
    if _CONFIG_IMPORT_ERROR is not None:
        print(
            "Config import warning: using fallback research defaults because "
            f"`config.py` failed to import ({type(_CONFIG_IMPORT_ERROR).__name__}: {_CONFIG_IMPORT_ERROR})."
        )

    if not _dependencies_ready():
        print(
            "Missing dependency for offline research runner: "
            f"{type(_IMPORT_ERROR).__name__ if _IMPORT_ERROR else 'ImportError'}: {_IMPORT_ERROR}"
        )
        print("Install required packages first (numpy, pandas, and project dependencies).")
        return 0

    if args.synthetic_smoke:
        print("Mode: SYNTHETIC SMOKE (in-memory only, not valid for trading decisions)")
        for sym in symbols:
            synthetic_symbol = f"SYNTHETIC-{sym}"
            ohlcv_df, micro_df = _make_synthetic_ohlcv_and_microstructure(max(220, min_rows + 20))
            report = _run_feature_isolation_report_safe(synthetic_symbol, ohlcv_df, micro_df, CONFIG)
            result = {
                "symbol": synthetic_symbol,
                "status": "ok",
                "rows_available": int(micro_df.shape[0]),
                "report": report,
            }
            _print_symbol_report(synthetic_symbol, result)
        return 0

    print(f"History file: {history_file}")
    hist = load_microstructure_history(history_file, symbols)

    if all(df.empty for df in hist.values()):
        print(
            "No usable historical microstructure rows were found. "
            "Collect live OFI/VPIN telemetry over time first, then rerun this script."
        )

    for sym in symbols:
        micro_df = hist.get(sym, pd.DataFrame(columns=REQUIRED_FEATURE_COLUMNS))
        result = await run_research_for_symbol(sym, micro_df, period, interval, CONFIG, min_rows)
        _print_symbol_report(sym, result)

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_async_main(args))
    finally:
        try:
            from provider import close_binance_http_session

            asyncio.run(close_binance_http_session())
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
