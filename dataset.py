# ==============================================================================
# dataset.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Not used by live trading.
# ==============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import binance_vision
from TripleBarrier import apply_triple_barrier, compute_causal_volatility, compute_uniqueness

BASE_FEATURE_COLUMNS = [
    "return",
    "log_return",
    "volatility",
    "log_range",
    "volume_change",
    "trend",
    "taker_buy_ratio",
    "flow_imb",
    "flow_imb_z",
    "quote_vol_z",
    "trades_z",
    "ret_autocorr_24",
    "dist_from_max_168",
    "dist_from_min_168",
    "hour_of_day",
    "day_of_week",
]

ENRICHED_FEATURE_COLUMNS = [
    "close_z_4h",
    "vol_4h",
    "vol_ratio_4h_1d",
    "trend_4h",
    "trend_strength_4h",
    "trend_1d",
    "dist_from_max_1d",
    "btc_trend_24",
    "btc_volatility",
    "btc_dist_from_max_168",
    "rel_strength",
]

HMM_FEATURE_COLUMNS = [
    "hmm_bull_prob",
    "hmm_neutral_prob",
    "hmm_bear_prob",
    "hmm_confidence",
    "hmm_policy_code",
    "hmm_regime_age",
]

HMM_POLICY_CODE_MAP = {"block": 0, "caution": 1, "allow": 2}
HMM_WF_COLUMN_MAP = {
    "timestamp": "timestamp",
    "regime": "hmm_regime",
    "policy": "hmm_policy",
    "bull_prob": "hmm_bull_prob",
    "neutral_prob": "hmm_neutral_prob",
    "bear_prob": "hmm_bear_prob",
    "confidence": "hmm_confidence",
}

ENRICHMENT_TIMEFRAME_DELTAS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def _project_config(config: dict | None = None) -> dict:
    if config is not None:
        return dict(config)
    try:
        from config import CONFIG

        return dict(CONFIG)
    except Exception:
        cwd = os.getcwd()
        return {
            "BINANCE_SYMBOL_MAP": {},
            "DATASET_DATA_DIR": os.path.join(cwd, "data", "binance_vision"),
            "DATASET_OUTPUT_DIR": os.path.join(cwd, "data", "datasets"),
            "DATASET_DEFAULT_MONTHS": 24,
            "DATASET_DEFAULT_TIMEFRAME": "1h",
            "TRIPLE_BARRIER_HORIZON": 24,
            "TRIPLE_BARRIER_PROFIT_MULT": 2.0,
            "TRIPLE_BARRIER_LOSS_MULT": 1.0,
            "TRIPLE_BARRIER_VOL_WINDOW": 24,
        }


def resolve_binance_symbol(symbol: str, config: dict | None = None) -> str:
    cfg = _project_config(config)
    text = str(symbol).strip()
    mapped = cfg.get("BINANCE_SYMBOL_MAP", {}).get(text)
    if mapped:
        return str(mapped).strip().upper()
    return text.upper().replace("/", "").replace("-", "")


def _parse_time(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 7 and text[4] == "-":
        return pd.Timestamp(f"{text}-01", tz="UTC")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid dataset date/time value: {value!r}")
    return pd.Timestamp(ts).tz_convert("UTC")


def resolve_dataset_range(start: Any, end: Any, config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    if end is None:
        end_ts = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(nanoseconds=1)
    else:
        end_ts = _parse_time(end)
        if isinstance(end, str):
            text = end.strip()
            if len(text) == 7 and text[4] == "-":
                end_ts = end_ts + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            elif len(text) == 10 and text[4] == "-" and text[7] == "-":
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if start is None:
        months = max(1, int(config.get("DATASET_DEFAULT_MONTHS", 24)))
        start_ts = (end_ts - pd.DateOffset(months=months)).normalize()
    else:
        start_ts = _parse_time(start)
    return start_ts, end_ts


def _rolling_z(series: pd.Series, window: int = 50) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mean = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std(ddof=0)
    return ((x - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_autocorr_lag1(series: pd.Series, window: int = 24) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")

    def _acf(vals: np.ndarray) -> float:
        arr = pd.Series(vals).dropna()
        if arr.shape[0] < 3:
            return np.nan
        out = arr.autocorr(lag=1)
        return float(out) if np.isfinite(out) else np.nan

    return x.rolling(window=window, min_periods=window).apply(_acf, raw=True)


def _normalize_klines_index(klines: pd.DataFrame) -> pd.DataFrame:
    out = klines.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out = out.sort_index()
    out.index.name = "OpenTime"
    return out


def _load_klines(
    raw_symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    config: dict,
) -> pd.DataFrame:
    klines = binance_vision.download_klines(
        raw_symbol,
        timeframe,
        start_ts,
        end_ts,
        data_dir=str(config.get("DATASET_DATA_DIR")),
    )
    return _normalize_klines_index(klines)


def _enrichment_timeframe_key(timeframe: str) -> str:
    return str(timeframe).strip().lower()


def _enrichment_timeframe_delta(timeframe: str) -> pd.Timedelta:
    key = _enrichment_timeframe_key(timeframe)
    if key not in ENRICHMENT_TIMEFRAME_DELTAS:
        raise ValueError("enrichment فعلاً فقط 1h و 4h را پشتیبانی می‌کند")
    return ENRICHMENT_TIMEFRAME_DELTAS[key]


def _resample_origin(index: pd.DatetimeIndex) -> pd.Timestamp:
    if index.tz is None:
        return pd.Timestamp("1970-01-01")
    return pd.Timestamp("1970-01-01", tz=index.tz)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    out = ohlcv.resample(
        rule,
        label="left",
        closed="left",
        origin=_resample_origin(pd.DatetimeIndex(ohlcv.index)),
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _rolling_log_return_std(close: pd.Series, window: int) -> pd.Series:
    log_return = np.log(pd.to_numeric(close, errors="coerce") / pd.to_numeric(close, errors="coerce").shift(1))
    return log_return.rolling(window=window, min_periods=window).std(ddof=0)


def _merge_closed_higher_frame(
    target_index: pd.DatetimeIndex,
    features: pd.DataFrame,
    higher_frame_delta: pd.Timedelta,
    row_delta: pd.Timedelta,
) -> pd.DataFrame:
    columns = list(features.columns)
    if len(target_index) == 0:
        return pd.DataFrame(index=target_index, columns=columns, dtype=float)

    target_index = pd.DatetimeIndex(target_index)
    left = pd.DataFrame(
        {
            "_row_pos": np.arange(len(target_index)),
            "OpenTime": target_index,
            "_row_close_time": target_index + row_delta,
        }
    ).sort_values("_row_close_time")

    if features.empty:
        out = pd.DataFrame(index=target_index, columns=columns, dtype=float)
        out.index.name = "OpenTime"
        return out

    right = features.copy()
    right["_feature_close_time"] = pd.DatetimeIndex(right.index) + higher_frame_delta
    right = right.sort_values("_feature_close_time")

    merged = pd.merge_asof(
        left,
        right[["_feature_close_time", *columns]],
        left_on="_row_close_time",
        right_on="_feature_close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_pos").set_index("OpenTime")
    merged.index.name = "OpenTime"
    return merged[columns]


def build_multitimeframe_features(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """Build closed-candle 4h/1d context features for a base dataset."""

    row_delta = _enrichment_timeframe_delta(timeframe)
    klines = _normalize_klines_index(df)

    frame_4h = _resample_ohlcv(klines, "4h")
    close_4h = pd.to_numeric(frame_4h["Close"], errors="coerce")
    features_4h = pd.DataFrame(index=frame_4h.index)
    features_4h["close_z_4h"] = _rolling_z(close_4h, 20)
    features_4h["vol_4h"] = _rolling_log_return_std(close_4h, 20)
    features_4h["trend_4h"] = np.log(close_4h / close_4h.shift(12))
    features_4h["trend_strength_4h"] = (features_4h["trend_4h"].abs() / (features_4h["vol_4h"] + 1e-12)).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    frame_1d = _resample_ohlcv(klines, "24h")
    close_1d = pd.to_numeric(frame_1d["Close"], errors="coerce")
    features_1d = pd.DataFrame(index=frame_1d.index)
    features_1d["_vol_1d"] = _rolling_log_return_std(close_1d, 14)
    features_1d["trend_1d"] = np.log(close_1d / close_1d.shift(7))
    rolling_max_1d = close_1d.rolling(window=30, min_periods=1).max()
    features_1d["dist_from_max_1d"] = np.log(close_1d / rolling_max_1d)

    joined_4h = _merge_closed_higher_frame(klines.index, features_4h, pd.Timedelta(hours=4), row_delta)
    joined_1d = _merge_closed_higher_frame(klines.index, features_1d, pd.Timedelta(days=1), row_delta)

    out = pd.DataFrame(index=klines.index)
    for col in ["close_z_4h", "vol_4h", "trend_4h", "trend_strength_4h"]:
        out[col] = joined_4h[col]
    out["vol_ratio_4h_1d"] = (joined_4h["vol_4h"] / (joined_1d["_vol_1d"] + 1e-12)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    out["trend_1d"] = joined_1d["trend_1d"]
    out["dist_from_max_1d"] = joined_1d["dist_from_max_1d"]
    return out.reindex(columns=ENRICHED_FEATURE_COLUMNS[:7]).replace([np.inf, -np.inf], np.nan)


def build_btc_context_features(symbol_features: pd.DataFrame, btc_klines: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    btc_base = build_causal_features(_normalize_klines_index(btc_klines), config)
    out = pd.DataFrame(index=symbol_features.index)
    out["btc_trend_24"] = btc_base["trend"].reindex(symbol_features.index)
    out["btc_volatility"] = btc_base["volatility"].reindex(symbol_features.index)
    out["btc_dist_from_max_168"] = btc_base["dist_from_max_168"].reindex(symbol_features.index)
    out["rel_strength"] = pd.to_numeric(symbol_features["trend"], errors="coerce") - out["btc_trend_24"]
    return out.replace([np.inf, -np.inf], np.nan)


def _hmm_walkforward_path(hmm_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(hmm_dir) / f"hmm_walkforward_{symbol}_{timeframe}_pomegranate_normal.csv"


def _hmm_regime_age(regimes: pd.Series) -> pd.Series:
    clean = regimes.astype("string").fillna("unknown")
    changes = clean.ne(clean.shift()).fillna(True).to_numpy(dtype=bool)
    group_id = changes.cumsum()
    age = pd.Series(np.arange(len(group_id))).groupby(group_id).cumcount()
    return pd.Series(age.to_numpy(), index=regimes.index, dtype=int)


def load_hmm_walkforward_features(
    symbol: str,
    timeframe: str,
    target_index: pd.DatetimeIndex,
    hmm_dir: str | Path,
) -> pd.DataFrame:
    """Join HMM walk-forward rows only after their production timestamp."""

    if _enrichment_timeframe_key(timeframe) != "1h":
        raise ValueError("فایل‌های walk-forward رژیم فقط 1h اند")

    path = _hmm_walkforward_path(hmm_dir, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"HMM walk-forward file not found for {symbol} {timeframe}: {path}")

    wf = pd.read_csv(path)
    required_cols = list(HMM_WF_COLUMN_MAP.values())
    missing = [col for col in required_cols if col not in wf.columns]
    if missing:
        raise ValueError(f"HMM walk-forward file {path} is missing required columns: {missing}")

    timestamp_col = HMM_WF_COLUMN_MAP["timestamp"]
    wf["_hmm_ready_time"] = pd.to_datetime(wf[timestamp_col], utc=True, errors="coerce")
    if wf["_hmm_ready_time"].isna().any():
        raise ValueError(f"HMM walk-forward file {path} contains unparseable timestamp values")
    wf = wf.sort_values("_hmm_ready_time").drop_duplicates(subset=["_hmm_ready_time"], keep="last").copy()
    wf["hmm_regime_age"] = _hmm_regime_age(wf[HMM_WF_COLUMN_MAP["regime"]])

    prepared = pd.DataFrame(
        {
            "_hmm_ready_time": wf["_hmm_ready_time"],
            "hmm_bull_prob": pd.to_numeric(wf[HMM_WF_COLUMN_MAP["bull_prob"]], errors="coerce"),
            "hmm_neutral_prob": pd.to_numeric(wf[HMM_WF_COLUMN_MAP["neutral_prob"]], errors="coerce"),
            "hmm_bear_prob": pd.to_numeric(wf[HMM_WF_COLUMN_MAP["bear_prob"]], errors="coerce"),
            "hmm_confidence": pd.to_numeric(wf[HMM_WF_COLUMN_MAP["confidence"]], errors="coerce"),
            "hmm_policy_code": wf[HMM_WF_COLUMN_MAP["policy"]].astype("string").str.lower().map(HMM_POLICY_CODE_MAP),
            "hmm_regime_age": wf["hmm_regime_age"],
        }
    ).sort_values("_hmm_ready_time")

    target_index = pd.DatetimeIndex(target_index)
    row_delta = _enrichment_timeframe_delta(timeframe)
    left = pd.DataFrame(
        {
            "_row_pos": np.arange(len(target_index)),
            "OpenTime": target_index,
            "_row_close_time": target_index + row_delta,
        }
    ).sort_values("_row_close_time")

    merged = pd.merge_asof(
        left,
        prepared[["_hmm_ready_time", *HMM_FEATURE_COLUMNS]],
        left_on="_row_close_time",
        right_on="_hmm_ready_time",
        direction="backward",
        tolerance=pd.Timedelta(hours=6),
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_pos").set_index("OpenTime")
    merged.index.name = "OpenTime"
    return merged[HMM_FEATURE_COLUMNS]


def _hmm_coverage(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df[HMM_FEATURE_COLUMNS].notna().all(axis=1).mean())


def build_causal_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = _project_config(config)
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    quote_volume = pd.to_numeric(df["QuoteVolume"], errors="coerce")
    trades = pd.to_numeric(df["Trades"], errors="coerce")
    taker_buy_base = pd.to_numeric(df["TakerBuyBase"], errors="coerce")

    out["return"] = close.pct_change()
    out["log_return"] = np.log(close / close.shift(1))
    out["volatility"] = compute_causal_volatility(close, span=int(cfg.get("TRIPLE_BARRIER_VOL_WINDOW", 24)))
    out["log_range"] = np.log(high / low)
    valid_volume_pair = (volume > 0) & (volume.shift(1) > 0)
    out["volume_change"] = np.nan
    out.loc[valid_volume_pair, "volume_change"] = np.log(volume.loc[valid_volume_pair] / volume.shift(1).loc[valid_volume_pair])
    out["trend"] = np.log(close / close.shift(24))

    ratio = pd.Series(0.5, index=df.index, dtype=float)
    valid_volume = volume > 0
    ratio.loc[valid_volume] = (taker_buy_base.loc[valid_volume] / volume.loc[valid_volume]).clip(0.0, 1.0)
    out["taker_buy_ratio"] = ratio.fillna(0.5)
    out["flow_imb"] = (2.0 * out["taker_buy_ratio"]) - 1.0
    out["flow_imb_z"] = _rolling_z(out["flow_imb"], 50)
    out["quote_vol_z"] = _rolling_z(quote_volume, 50)
    out["trades_z"] = _rolling_z(trades, 50)
    out["ret_autocorr_24"] = _rolling_autocorr_lag1(out["log_return"], 24)

    rolling_max = close.rolling(window=168, min_periods=1).max()
    rolling_min = close.rolling(window=168, min_periods=1).min()
    out["dist_from_max_168"] = np.log(close / rolling_max)
    out["dist_from_min_168"] = np.log(close / rolling_min)
    out["hour_of_day"] = pd.DatetimeIndex(df.index).hour.astype(int)
    out["day_of_week"] = pd.DatetimeIndex(df.index).dayofweek.astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def _fractional_features(df: pd.DataFrame, config: dict, strict: bool) -> tuple[pd.DataFrame, str, str | None]:
    try:
        import fractional

        frac = fractional.build_fractional_hmm_features(df, config)
        if not isinstance(frac, pd.DataFrame) or frac.empty:
            raise RuntimeError("fractional.build_fractional_hmm_features returned empty output")
        return frac.copy(), "fractional", None
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        if strict:
            raise RuntimeError(f"Fractional feature generation failed in strict mode: {reason}") from exc
        return pd.DataFrame(index=df.index), "fallback", reason


def _metadata_for(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    feature_path: str,
    fallback_reason: str | None,
    tb_params: dict,
    hmm_coverage: float | None,
    hmm_enabled: bool,
) -> dict:
    events = df.loc[df["tb_label"].notna()] if "tb_label" in df.columns else pd.DataFrame()
    label_counts = {}
    if not events.empty:
        label_counts = {str(int(k)): int(v) for k, v in events["tb_label"].value_counts(dropna=True).sort_index().items()}
    mean_uniqueness = None
    if "tb_uniqueness" in df.columns:
        vals = pd.to_numeric(df["tb_uniqueness"], errors="coerce").dropna()
        mean_uniqueness = float(vals.mean()) if not vals.empty else None
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": df.index.min().isoformat() if not df.empty else None,
        "end": df.index.max().isoformat() if not df.empty else None,
        "rows": int(df.shape[0]),
        "events": int(events.shape[0]),
        "label_distribution": label_counts,
        "mean_tb_uniqueness": mean_uniqueness,
        "feature_path": feature_path,
        "fallback_reason": fallback_reason,
        "new_feature_columns": list(ENRICHED_FEATURE_COLUMNS) + (list(HMM_FEATURE_COLUMNS) if hmm_enabled else []),
        "enriched_feature_columns": list(ENRICHED_FEATURE_COLUMNS),
        "hmm_feature_columns": list(HMM_FEATURE_COLUMNS) if hmm_enabled else [],
        "hmm_coverage": hmm_coverage,
        "triple_barrier": tb_params,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet" and _has_pyarrow():
        df.to_parquet(path, index=True)
    else:
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        df.to_csv(path, index=True, index_label="OpenTime")
    return path


def _write_metadata(path: Path, metadata: dict) -> None:
    meta_path = path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, default=str)


def _build_one_symbol(
    input_symbol: str,
    timeframe: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    horizon: int,
    config: dict,
    strict: bool,
    btc_klines: pd.DataFrame,
    hmm_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    raw_symbol = resolve_binance_symbol(input_symbol, config)
    klines = _normalize_klines_index(btc_klines) if raw_symbol == "BTCUSDT" else _load_klines(raw_symbol, timeframe, start_ts, end_ts, config)

    base_features = build_causal_features(klines, config)
    multiframe_features = build_multitimeframe_features(klines, timeframe=timeframe)
    btc_context_features = build_btc_context_features(base_features, btc_klines, config)
    frac_features, feature_path, fallback_reason = _fractional_features(klines, config, strict)
    frac_features = frac_features.reindex(klines.index)

    feature_df = pd.concat([base_features, multiframe_features, btc_context_features, frac_features], axis=1)
    feature_columns = [c for c in feature_df.columns if c not in klines.columns]
    wide = pd.concat([klines, feature_df], axis=1)
    wide = wide.replace([np.inf, -np.inf], np.nan)
    required_feature_columns = [col for col in feature_columns if col not in ENRICHED_FEATURE_COLUMNS]
    required = ["Open", "High", "Low", "Close", *required_feature_columns]
    wide = wide.dropna(subset=[c for c in required if c in wide.columns]).copy()
    wide["Symbol"] = raw_symbol

    hmm_coverage_value = None
    if hmm_dir is not None:
        hmm_features = load_hmm_walkforward_features(raw_symbol, timeframe, wide.index, hmm_dir)
        wide = pd.concat([wide, hmm_features], axis=1)
        hmm_coverage_value = _hmm_coverage(wide)

    tb_params = {
        "horizon": int(horizon),
        "profit_mult": float(config.get("TRIPLE_BARRIER_PROFIT_MULT", 2.0)),
        "loss_mult": float(config.get("TRIPLE_BARRIER_LOSS_MULT", 1.0)),
        "vol_window": int(config.get("TRIPLE_BARRIER_VOL_WINDOW", 24)),
    }
    labeled = apply_triple_barrier(
        wide,
        price_col="Close",
        volatility_col="volatility",
        horizon=int(horizon),
        profit_mult=tb_params["profit_mult"],
        loss_mult=tb_params["loss_mult"],
    )
    labeled["tb_uniqueness"] = compute_uniqueness(labeled)
    labeled.index.name = "OpenTime"
    metadata = _metadata_for(
        labeled,
        raw_symbol,
        timeframe,
        feature_path,
        fallback_reason,
        tb_params,
        hmm_coverage_value,
        hmm_enabled=hmm_dir is not None,
    )
    return labeled, metadata


def build_training_dataset(
    symbols,
    timeframe: str = "1h",
    start=None,
    end=None,
    horizon: int | None = None,
    output_path: str | None = None,
    output_dir: str | Path | None = None,
    hmm_dir: str | Path | None = None,
    config: dict | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    cfg = _project_config(config)
    timeframe = str(timeframe or cfg.get("DATASET_DEFAULT_TIMEFRAME", "1h"))
    _enrichment_timeframe_delta(timeframe)
    if hmm_dir is not None and _enrichment_timeframe_key(timeframe) != "1h":
        raise ValueError("فایل‌های walk-forward رژیم فقط 1h اند")
    horizon = int(horizon if horizon is not None else cfg.get("TRIPLE_BARRIER_HORIZON", 24))
    start_ts, end_ts = resolve_dataset_range(start, end, cfg)

    if isinstance(symbols, str):
        symbol_list = [symbols]
    else:
        symbol_list = [str(s) for s in symbols]
    symbol_list = [s for s in symbol_list if s.strip()]
    if not symbol_list:
        raise ValueError("build_training_dataset requires at least one symbol")

    output_dir_path = Path(output_dir) if output_dir is not None else Path(str(cfg.get("DATASET_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "datasets"))))
    btc_klines = _load_klines("BTCUSDT", timeframe, start_ts, end_ts, cfg)
    frames: list[pd.DataFrame] = []
    metadata_by_symbol: dict[str, dict] = {}
    for sym in symbol_list:
        df_sym, meta = _build_one_symbol(
            sym,
            timeframe,
            start_ts,
            end_ts,
            horizon,
            cfg,
            strict,
            btc_klines=btc_klines,
            hmm_dir=hmm_dir,
        )
        raw_symbol = str(meta["symbol"])
        per_symbol_path = output_dir_path / f"{raw_symbol}_{timeframe}.csv"
        written = _write_table(df_sym, per_symbol_path)
        _write_metadata(written, meta)
        frames.append(df_sym)
        metadata_by_symbol[raw_symbol] = meta

    combined = pd.concat(frames, axis=0).sort_index()
    combined.index.name = "OpenTime"
    combined.attrs["metadata_by_symbol"] = metadata_by_symbol

    if output_path:
        out_path = _write_table(combined, Path(output_path))
        combined_meta = {
            "symbols": list(metadata_by_symbol.keys()),
            "timeframe": timeframe,
            "rows": int(combined.shape[0]),
            "metadata_by_symbol": metadata_by_symbol,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_metadata(out_path, combined_meta)

    return combined


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline training datasets from Binance Vision archives")
    parser.add_argument("--symbols", nargs="+", required=True, help="Internal or raw Binance symbols, e.g. BTC-USD BTCUSDT")
    parser.add_argument("--timeframe", default=None, help="Kline interval, default from config")
    parser.add_argument("--start", default=None, help="YYYY-MM or ISO timestamp")
    parser.add_argument("--end", default=None, help="YYYY-MM or ISO timestamp")
    parser.add_argument("--horizon", type=int, default=None, help="Triple-barrier horizon in bars")
    parser.add_argument("--output-path", default=None, help="Optional combined output CSV/parquet path")
    parser.add_argument("--output-dir", default=None, help="Directory for per-symbol dataset CSV/metadata files")
    parser.add_argument("--hmm-dir", default=None, help="Optional HMM walk-forward output directory to enrich datasets")
    parser.add_argument("--no-strict", action="store_true", help="Allow fallback when fractional features fail")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cfg = _project_config()
        timeframe = args.timeframe or str(cfg.get("DATASET_DEFAULT_TIMEFRAME", "1h"))
        df = build_training_dataset(
            args.symbols,
            timeframe=timeframe,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            output_path=args.output_path,
            output_dir=args.output_dir,
            hmm_dir=args.hmm_dir,
            config=cfg,
            strict=not args.no_strict,
        )
        print(f"Built dataset rows={df.shape[0]} symbols={sorted(df['Symbol'].dropna().unique().tolist())}")
        return 0
    except Exception as exc:
        print(f"dataset build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
