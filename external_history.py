# ==============================================================================
# external_history.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Not used by live trading.
# ==============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import binance_vision


FUNDING_COLUMN_MAP = {
    "settlement_time": ("calc_time", "fundingTime", "funding_time", "time", "timestamp"),
    "funding_rate": ("last_funding_rate", "fundingRate", "funding_rate", "rate"),
    "symbol": ("symbol",),
}
MICRO_HISTORY_COLUMN_MAP = {
    "schema_version": "schema_version",
    "ts": "ts",
    "symbol": "symbol",
    "vpin": "vpin",
}

VPIN_BUCKET_COUNT_HIST = 20
VPIN_WINDOW_DAYS = 7
VPIN_Z_LOOKBACK = 50
FUNDING_Z_LOOKBACK = 90

LIVE_VPIN_CONTRACT = {
    "source": "Binance aggTrades REST via provider.fetch_binance_agg_trades",
    "lookback_minutes": 60,
    "limit": 1000,
    "side_classification": "buyer_initiated = not isBuyerMaker (m == False)",
    "volume_unit": "quote by default when VPIN_BUCKET_VOLUME_MODE=dynamic_quote",
    "bucket_capacity": "sum(quantity_quote over fetched trades) / VPIN_BUCKET_COUNT",
    "bucket_count": 20,
    "min_buckets": 5,
    "incomplete_bucket": "final incomplete bucket is dropped",
    "timestamp": "vpin_last_trade_ts from the last normalized aggTrade; v46 history uses it as ts when present",
}
LIVE_VPIN_MATERIAL_DIFFERENCES = [
    "Live VPIN uses aggTrades over the latest 60 minutes, while reconstruction uses 1m klines over 7 days.",
    "Live bucket capacity is total fetched quote volume / 20, while reconstruction uses median complete hourly quote volume over 7 days / 20.",
    "Live VPIN aggregates all complete buckets in the fetched window with min_buckets=5, while reconstruction averages the last 20 complete buckets.",
    "Live uses trade-level buyer-maker classification; reconstruction uses 1m TakerBuyQuote and QuoteVolume aggregates.",
]


def _parse_time(value: Any, default: pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None:
        if default is not None:
            return default
        return pd.Timestamp.now(tz="UTC")
    text = str(value).strip()
    if len(text) == 7 and text[4] == "-":
        return pd.Timestamp(f"{text}-01", tz="UTC")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date/time value: {value!r}")
    return pd.Timestamp(ts).tz_convert("UTC")


def _default_output_dir() -> Path:
    return Path(os.getcwd()) / "data" / "external"


def _safe_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label=df.index.name or "timestamp")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Missing expected column; candidates={candidates}; actual={list(df.columns)}")


def _timestamp_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if not finite.empty:
        unit = "us" if float(finite.abs().max()) > 1e14 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(values, utc=True, errors="coerce")


def _causal_z(series: pd.Series, lookback: int) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    base = x.shift(1)
    mean = base.rolling(lookback, min_periods=lookback).mean()
    std = base.rolling(lookback, min_periods=lookback).std(ddof=0)
    z = (x - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def normalize_funding_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=["funding_rate", "funding_z", "funding_extreme_pos"])

    time_col = _resolve_column(raw, FUNDING_COLUMN_MAP["settlement_time"])
    rate_col = _resolve_column(raw, FUNDING_COLUMN_MAP["funding_rate"])
    out = pd.DataFrame(
        {
            "settlement_time": _timestamp_series(raw[time_col]),
            "funding_rate": pd.to_numeric(raw[rate_col], errors="coerce"),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["settlement_time", "funding_rate"])
    if out.empty:
        return pd.DataFrame(columns=["funding_rate", "funding_z", "funding_extreme_pos"])

    out = out.sort_values("settlement_time").drop_duplicates(subset=["settlement_time"], keep="last")
    out.set_index("settlement_time", inplace=True)
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC")
    out.index.name = "settlement_time"
    out["funding_rate"] = out["funding_rate"].astype(float)
    out["funding_z"] = _causal_z(out["funding_rate"], FUNDING_Z_LOOKBACK)
    out["funding_extreme_pos"] = ((out["funding_rate"] > 0.0) & (out["funding_z"] >= 2.0)).astype(int)
    return out[["funding_rate", "funding_z", "funding_extreme_pos"]]


def build_funding_history(
    symbol: str,
    start: Any,
    end: Any,
    data_dir: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    save: bool = True,
) -> dict:
    sym = _safe_symbol(symbol)
    start_ts = _parse_time(start)
    end_ts = _parse_time(end)
    raw, skipped_months = binance_vision.download_funding_rates(
        sym,
        start_ts,
        end_ts,
        data_dir=data_dir,
        return_skipped=True,
    )
    funding = normalize_funding_frame(raw)
    if not funding.empty:
        funding = funding.loc[(funding.index >= start_ts) & (funding.index <= end_ts)].copy()

    out_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    csv_path = out_dir / f"funding_{sym}.csv"
    meta_path = out_dir / f"funding_{sym}.json"
    metadata = {
        "symbol": sym,
        "requested_start": start_ts.isoformat(),
        "requested_end": end_ts.isoformat(),
        "observed_start": None if funding.empty else funding.index.min().isoformat(),
        "observed_end": None if funding.empty else funding.index.max().isoformat(),
        "rows": int(funding.shape[0]),
        "skipped_months": list(skipped_months),
        "csv_path": str(csv_path),
        "metadata_path": str(meta_path),
    }
    if save:
        _write_csv(funding, csv_path)
        _write_json(meta_path, metadata)
    return {"data": funding, "metadata": metadata, "csv_path": csv_path, "metadata_path": meta_path}


def prepare_minute_klines(klines: pd.DataFrame) -> pd.DataFrame:
    if klines is None or not isinstance(klines, pd.DataFrame) or klines.empty:
        return pd.DataFrame(columns=["open_time", "close_time", "quote_volume", "buy_quote", "sell_quote"])

    out = klines.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()

    quote = pd.to_numeric(out.get("QuoteVolume"), errors="coerce")
    buy = pd.to_numeric(out.get("TakerBuyQuote"), errors="coerce")
    prepared = pd.DataFrame(
        {
            "open_time": pd.DatetimeIndex(out.index),
            "close_time": pd.DatetimeIndex(out.index) + pd.Timedelta(minutes=1),
            "quote_volume": quote,
            "buy_quote": buy,
        }
    )
    prepared["sell_quote"] = prepared["quote_volume"] - prepared["buy_quote"]
    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(subset=["open_time", "close_time", "quote_volume", "buy_quote", "sell_quote"])
    prepared = prepared.loc[prepared["quote_volume"] > 0.0].copy()
    prepared["buy_quote"] = prepared["buy_quote"].clip(lower=0.0, upper=prepared["quote_volume"])
    prepared["sell_quote"] = (prepared["quote_volume"] - prepared["buy_quote"]).clip(lower=0.0)
    prepared.reset_index(drop=True, inplace=True)
    return prepared


def compute_vpin_from_minute_bars(
    minute_bars: pd.DataFrame,
    bucket_capacity: float,
    bucket_count: int = VPIN_BUCKET_COUNT_HIST,
) -> float:
    prepared = prepare_minute_klines(minute_bars) if "quote_volume" not in minute_bars.columns else minute_bars.copy()
    if prepared.empty:
        return float("nan")

    capacity = float(bucket_capacity)
    if not np.isfinite(capacity) or capacity <= 1e-12:
        return float("nan")

    total = pd.to_numeric(prepared["quote_volume"], errors="coerce").to_numpy(dtype=float)
    buy = pd.to_numeric(prepared["buy_quote"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(total) & np.isfinite(buy) & (total > 0.0)
    total = total[valid]
    buy = np.clip(buy[valid], 0.0, total)
    if total.size == 0:
        return float("nan")

    n_full = int(np.floor(float(total.sum()) / capacity))
    if n_full < int(bucket_count):
        return float("nan")

    cum_total = np.concatenate([[0.0], np.cumsum(total)])
    cum_buy = np.concatenate([[0.0], np.cumsum(buy)])
    boundaries = np.arange(n_full + 1, dtype=float) * capacity
    buy_at_boundaries = np.interp(boundaries, cum_total, cum_buy)
    bucket_buy = np.diff(buy_at_boundaries)
    imbalance = np.abs(bucket_buy - (capacity - bucket_buy)) / capacity
    return float(np.mean(imbalance[-int(bucket_count) :]))


def _window_has_full_support(prepared: pd.DataFrame, ts: pd.Timestamp, window: pd.Timedelta) -> bool:
    if prepared.empty:
        return False
    start = ts - window
    min_close = prepared["close_time"].min()
    max_close = prepared["close_time"].max()
    return bool(min_close <= start + pd.Timedelta(minutes=1) and max_close >= ts)


def _capacity_at(prepared: pd.DataFrame, ts: pd.Timestamp, bucket_count: int, window: pd.Timedelta) -> float:
    window_start = ts - window
    w = prepared.loc[(prepared["close_time"] > window_start) & (prepared["close_time"] <= ts)]
    if w.empty:
        return float("nan")
    hourly = (
        pd.Series(w["quote_volume"].to_numpy(dtype=float), index=pd.DatetimeIndex(w["close_time"]))
        .resample("1h", label="right", closed="right")
        .agg(["sum", "count"])
    )
    full = hourly.loc[(hourly["count"] == 60) & (hourly.index <= ts), "sum"]
    if full.empty:
        return float("nan")
    return float(full.median() / float(bucket_count))


def reconstruct_vpin_at(
    minute_klines: pd.DataFrame,
    ts: Any,
    bucket_capacity: float | None = None,
    require_full_window: bool = True,
    bucket_count: int = VPIN_BUCKET_COUNT_HIST,
    window_days: int = VPIN_WINDOW_DAYS,
) -> dict:
    prepared = prepare_minute_klines(minute_klines) if "close_time" not in minute_klines.columns else minute_klines.copy()
    t = _parse_time(ts)
    window = pd.Timedelta(days=int(window_days))
    if require_full_window and not _window_has_full_support(prepared, t, window):
        return {"vpin": float("nan"), "bucket_capacity": float("nan")}

    windowed = prepared.loc[(prepared["close_time"] > (t - window)) & (prepared["close_time"] <= t)]
    if windowed.empty:
        return {"vpin": float("nan"), "bucket_capacity": float("nan")}

    capacity = float(bucket_capacity) if bucket_capacity is not None else _capacity_at(prepared, t, int(bucket_count), window)
    return {
        "vpin": compute_vpin_from_minute_bars(windowed, capacity, bucket_count=int(bucket_count)),
        "bucket_capacity": capacity,
    }


def _hourly_checkpoints(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DatetimeIndex:
    first = start_ts.ceil("1h")
    last = end_ts.floor("1h")
    if last < first:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, last, freq="1h", tz="UTC")


def _missing_minute_ratio(prepared: pd.DataFrame) -> float | None:
    if prepared.empty:
        return None
    open_times = pd.DatetimeIndex(prepared["open_time"]).drop_duplicates().sort_values()
    expected = int(((open_times.max() - open_times.min()) / pd.Timedelta(minutes=1))) + 1
    if expected <= 0:
        return None
    return float(max(0, expected - len(open_times)) / expected)


def build_reconstructed_vpin_history(
    symbol: str,
    start: Any,
    end: Any,
    data_dir: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    save: bool = True,
) -> dict:
    sym = _safe_symbol(symbol)
    start_ts = _parse_time(start)
    end_ts = _parse_time(end)
    request_start = start_ts - pd.Timedelta(days=VPIN_WINDOW_DAYS)
    klines = binance_vision.download_klines(sym, "1m", request_start, end_ts, data_dir=data_dir)
    prepared = prepare_minute_klines(klines)

    rows = []
    for ts in _hourly_checkpoints(start_ts, end_ts):
        rec = reconstruct_vpin_at(prepared, ts)
        rows.append({"timestamp": ts, "vpin": rec["vpin"], "bucket_capacity": rec["bucket_capacity"]})

    history = pd.DataFrame(rows)
    if history.empty:
        history = pd.DataFrame(columns=["vpin", "vpin_z", "bucket_capacity"])
        history.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    else:
        history.set_index("timestamp", inplace=True)
        history.index = pd.DatetimeIndex(history.index).tz_convert("UTC")
        history.index.name = "timestamp"
        history["vpin_z"] = _causal_z(history["vpin"], VPIN_Z_LOOKBACK)
        history = history[["vpin", "vpin_z", "bucket_capacity"]]

    out_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    csv_path = out_dir / f"vpin_{sym}_1h.csv"
    meta_path = out_dir / f"vpin_{sym}_1h.json"
    metadata = {
        "symbol": sym,
        "requested_start": start_ts.isoformat(),
        "requested_end": end_ts.isoformat(),
        "observed_1m_start": None if prepared.empty else pd.Timestamp(prepared["open_time"].min()).isoformat(),
        "observed_1m_end": None if prepared.empty else pd.Timestamp(prepared["open_time"].max()).isoformat(),
        "missing_minute_ratio": _missing_minute_ratio(prepared),
        "checkpoint_count": int(history.shape[0]),
        "bucket_params": {
            "bucket_count": VPIN_BUCKET_COUNT_HIST,
            "capacity_window_days": VPIN_WINDOW_DAYS,
            "capacity_definition": "median complete hourly quote volume in [T-7d, T] / 20",
        },
        "csv_path": str(csv_path),
        "metadata_path": str(meta_path),
    }
    if save:
        _write_csv(history, csv_path)
        _write_json(meta_path, metadata)
    return {"data": history, "metadata": metadata, "csv_path": csv_path, "metadata_path": meta_path}


def _read_micro_history(path: str | os.PathLike[str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    rows = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _corr_values(a: pd.Series, b: pd.Series) -> tuple[float | None, float | None, bool]:
    if len(a) < 2 or len(b) < 2:
        return None, None, False
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return None, None, True
    pearson = a.corr(b, method="pearson")
    spearman = a.corr(b, method="spearman")
    return (
        None if pd.isna(pearson) else float(pearson),
        None if pd.isna(spearman) else float(spearman),
        False,
    )


def _verdict(n: int, pearson: float | None, insufficient_variation: bool, material_differences: list[str]) -> str:
    if n < 50:
        return "insufficient_overlap"
    if insufficient_variation:
        return "insufficient_variation"
    if material_differences:
        return "reference_only"
    if pearson is None:
        return "insufficient_variation"
    if pearson >= 0.7:
        return "approved"
    if pearson >= 0.4:
        return "caution"
    return "rejected"


def validate_vpin_reconstruction(
    symbols: list[str],
    start: Any | None = None,
    end: Any | None = None,
    micro_history_file: str | os.PathLike[str] | None = None,
    data_dir: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    symbol_map: dict[str, str] | None = None,
    material_differences: list[str] | None = None,
    save: bool = True,
) -> dict:
    out_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
    report_path = out_dir / "vpin_validation_report.json"
    differences = LIVE_VPIN_MATERIAL_DIFFERENCES if material_differences is None else list(material_differences)
    report = {
        "status": "ok",
        "micro_history_file": None if micro_history_file is None else str(micro_history_file),
        "deduplicated_rows": 0,
        "live_vpin_contract": LIVE_VPIN_CONTRACT,
        "material_differences": differences,
        "symbols": {},
        "report_path": str(report_path),
    }

    if not micro_history_file:
        report["status"] = "skipped"
        report["reason"] = "micro_history_file_missing_or_empty; history accumulates on the server; rerun later with --validate-only"
        if save:
            _write_json(report_path, report)
        return report

    raw = _read_micro_history(micro_history_file)
    if raw.empty:
        report["status"] = "skipped"
        report["reason"] = "micro_history_file_missing_or_empty; history accumulates on the server; rerun later with --validate-only"
        if save:
            _write_json(report_path, report)
        return report

    ts_col = MICRO_HISTORY_COLUMN_MAP["ts"]
    symbol_col = MICRO_HISTORY_COLUMN_MAP["symbol"]
    vpin_col = MICRO_HISTORY_COLUMN_MAP["vpin"]
    missing = [col for col in [ts_col, symbol_col, vpin_col] if col not in raw.columns]
    if missing:
        report["status"] = "skipped"
        report["reason"] = f"micro_history_missing_columns:{missing}"
        if save:
            _write_json(report_path, report)
        return report

    df = raw.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df[vpin_col] = pd.to_numeric(df[vpin_col], errors="coerce")
    df = df.dropna(subset=[ts_col, symbol_col, vpin_col])
    if start is not None:
        df = df.loc[df[ts_col] >= _parse_time(start)]
    if end is not None:
        df = df.loc[df[ts_col] <= _parse_time(end)]

    before = int(df.shape[0])
    df = df.drop_duplicates(subset=[symbol_col, ts_col], keep="last").sort_values([symbol_col, ts_col])
    report["deduplicated_rows"] = before - int(df.shape[0])

    resolved_map = symbol_map or {}
    for symbol in symbols:
        live_symbol = str(symbol)
        binance_symbol = _safe_symbol(resolved_map.get(live_symbol, live_symbol))
        live = df.loc[df[symbol_col] == live_symbol, [ts_col, vpin_col]].copy()
        if live.empty:
            report["symbols"][live_symbol] = {
                "binance_symbol": binance_symbol,
                "n_common": 0,
                "verdict": "insufficient_overlap",
                "reason": "no_live_records",
            }
            continue

        min_ts = pd.Timestamp(live[ts_col].min())
        max_ts = pd.Timestamp(live[ts_col].max())
        klines = binance_vision.download_klines(
            binance_symbol,
            "1m",
            min_ts - pd.Timedelta(days=VPIN_WINDOW_DAYS),
            max_ts,
            data_dir=data_dir,
        )
        prepared = prepare_minute_klines(klines)
        reconstructed = []
        capacities = []
        for ts in live[ts_col]:
            rec = reconstruct_vpin_at(prepared, ts)
            reconstructed.append(rec["vpin"])
            capacities.append(rec["bucket_capacity"])

        compare = pd.DataFrame(
            {
                "ts": live[ts_col].to_numpy(),
                "live_vpin": live[vpin_col].to_numpy(dtype=float),
                "reconstructed_vpin": reconstructed,
                "bucket_capacity": capacities,
            }
        ).replace([np.inf, -np.inf], np.nan)
        compare = compare.dropna(subset=["live_vpin", "reconstructed_vpin"])
        n = int(compare.shape[0])
        pearson, spearman, constant = _corr_values(compare["live_vpin"], compare["reconstructed_vpin"]) if n else (None, None, False)
        bias = None if n == 0 else float(np.median(compare["live_vpin"] - compare["reconstructed_vpin"]))
        verdict = _verdict(n, pearson, constant, differences)
        report["symbols"][live_symbol] = {
            "binance_symbol": binance_symbol,
            "n_common": n,
            "pearson": pearson,
            "spearman": spearman,
            "median_bias_live_minus_reconstructed": bias,
            "verdict": verdict,
            "reference_only": bool(verdict == "reference_only"),
        }

    if save:
        _write_json(report_path, report)
    return report
