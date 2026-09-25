# ==============================================================================
# binance_vision.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Not used by live trading.
# ==============================================================================

from __future__ import annotations

import io
import os
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision"
urlopen = urllib_request.urlopen

KLINE_COLUMNS = [
    "OpenTime",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "CloseTime",
    "QuoteVolume",
    "Trades",
    "TakerBuyBase",
    "TakerBuyQuote",
    "Ignore",
]
KLINE_OUTPUT_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "QuoteVolume",
    "Trades",
    "TakerBuyBase",
    "TakerBuyQuote",
]

AGG_TRADE_COLUMNS = [
    "aggTradeId",
    "price",
    "quantity",
    "firstTradeId",
    "lastTradeId",
    "timestamp",
    "isBuyerMaker",
    "isBestMatch",
]
FUNDING_RATE_FALLBACK_COLUMNS = [
    "calc_time",
    "symbol",
    "last_funding_rate",
]


class BinanceVisionError(RuntimeError):
    pass


def _default_data_dir() -> str:
    try:
        from config import CONFIG

        return str(CONFIG.get("DATASET_DATA_DIR", os.path.join(os.getcwd(), "data", "binance_vision")))
    except Exception:
        return os.path.join(os.getcwd(), "data", "binance_vision")


def _safe_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=1, tz="UTC")


def _parse_time(value: Any, default: pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None:
        if default is not None:
            return default
        return pd.Timestamp.now(tz="UTC").normalize()
    text = str(value).strip()
    if len(text) == 7 and text[4] == "-":
        return pd.Timestamp(f"{text}-01", tz="UTC")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date/time value: {value!r}")
    return pd.Timestamp(ts).tz_convert("UTC")


def _month_iter(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    cur = _month_start(start)
    last = _month_start(end)
    out: list[str] = []
    while cur <= last:
        out.append(cur.strftime("%Y-%m"))
        cur = cur + pd.DateOffset(months=1)
    return out


def _days_in_month(month_tag: str, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    month = pd.Timestamp(f"{month_tag}-01", tz="UTC")
    next_month = month + pd.DateOffset(months=1)
    first = max(start.normalize(), month)
    last = min(end.normalize(), next_month - pd.Timedelta(days=1))
    days = []
    cur = first
    while cur <= last:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += pd.Timedelta(days=1)
    return days


def _kline_monthly_url(symbol: str, interval: str, month_tag: str) -> tuple[str, str]:
    filename = f"{symbol}-{interval}-{month_tag}.zip"
    url = f"{BASE_URL}/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
    return url, filename


def _kline_daily_url(symbol: str, interval: str, day_tag: str) -> tuple[str, str]:
    filename = f"{symbol}-{interval}-{day_tag}.zip"
    url = f"{BASE_URL}/data/spot/daily/klines/{symbol}/{interval}/{filename}"
    return url, filename


def _agg_monthly_url(symbol: str, month_tag: str) -> tuple[str, str]:
    filename = f"{symbol}-aggTrades-{month_tag}.zip"
    url = f"{BASE_URL}/data/spot/monthly/aggTrades/{symbol}/{filename}"
    return url, filename


def _agg_daily_url(symbol: str, day_tag: str) -> tuple[str, str]:
    filename = f"{symbol}-aggTrades-{day_tag}.zip"
    url = f"{BASE_URL}/data/spot/daily/aggTrades/{symbol}/{filename}"
    return url, filename


def _funding_monthly_url(symbol: str, month_tag: str) -> tuple[str, str]:
    filename = f"{symbol}-fundingRate-{month_tag}.zip"
    url = f"{BASE_URL}/data/futures/um/monthly/fundingRate/{symbol}/{filename}"
    return url, filename


def _is_missing_archive_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "not found" in text


def _download_cached(url: str, path: Path, context: str, timeout: int = 30) -> Path:
    if path.exists() and path.is_file() and path.stat().st_size > 0:
        try:
            _validate_zip_light(path)
            return path
        except BinanceVisionError:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with urlopen(url, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError) as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise BinanceVisionError(f"Binance Vision download failed for {context}; url={url}; error={type(exc).__name__}: {exc}") from exc
    if not payload:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise BinanceVisionError(f"Binance Vision download returned empty payload for {context}; url={url}")
    try:
        tmp_path.write_bytes(payload)
        _validate_zip_full(tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        if isinstance(exc, BinanceVisionError):
            raise BinanceVisionError(f"Binance Vision downloaded invalid zip for {context}; url={url}; error={exc}") from exc
        raise
    return path


def _validate_zip_light(path: str | os.PathLike[str]) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            if not [n for n in zf.namelist() if not n.endswith("/")]:
                raise BinanceVisionError(f"Zip archive has no CSV payload: {path}")
    except zipfile.BadZipFile as exc:
        raise BinanceVisionError(f"Invalid zip archive: {path}") from exc


def _validate_zip_full(path: str | os.PathLike[str]) -> None:
    try:
        with zipfile.ZipFile(path) as zf:
            if not [n for n in zf.namelist() if not n.endswith("/")]:
                raise BinanceVisionError(f"Zip archive has no CSV payload: {path}")
            bad_member = zf.testzip()
            if bad_member is not None:
                raise BinanceVisionError(f"Zip archive failed integrity check at member {bad_member}: {path}")
    except zipfile.BadZipFile as exc:
        raise BinanceVisionError(f"Invalid zip archive: {path}") from exc


def _read_first_csv_from_zip(path: str | os.PathLike[str]) -> bytes:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise BinanceVisionError(f"Zip archive has no CSV payload: {path}")
            with zf.open(names[0]) as fh:
                return fh.read()
    except zipfile.BadZipFile as exc:
        raise BinanceVisionError(f"Invalid zip archive: {path}") from exc


def _read_csv_no_schema(path: str | os.PathLike[str]) -> pd.DataFrame:
    payload = _read_first_csv_from_zip(path)
    if not payload:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(payload), header=None)


def _read_csv_with_optional_header(path: str | os.PathLike[str], fallback_columns: list[str]) -> pd.DataFrame:
    raw = _read_csv_no_schema(path)
    if raw.empty:
        return pd.DataFrame(columns=fallback_columns)

    first = [str(x).strip() for x in raw.iloc[0].tolist()]
    has_header = any(not _looks_numeric(x) for x in first)
    if has_header:
        out = raw.iloc[1:].reset_index(drop=True).copy()
        out.columns = first
        return out

    out = raw.copy()
    columns = list(fallback_columns)
    if out.shape[1] > len(columns):
        columns.extend([f"extra_{i}" for i in range(out.shape[1] - len(columns))])
    out.columns = columns[: out.shape[1]]
    return out


def _drop_header_row(raw: pd.DataFrame, first_timestamp_col: int = 0) -> pd.DataFrame:
    if raw.empty:
        return raw
    first = str(raw.iloc[0, first_timestamp_col]).strip().lower()
    try:
        float(first)
        return raw
    except Exception:
        return raw.iloc[1:].reset_index(drop=True)


def _looks_numeric(value: Any) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def _timestamp_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
    unit = "us" if float(finite.abs().max()) > 1e14 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")


def read_klines_zip(path: str | os.PathLike[str]) -> pd.DataFrame:
    raw = _read_csv_no_schema(path)
    if raw.empty:
        return pd.DataFrame(columns=KLINE_OUTPUT_COLUMNS)
    raw = _drop_header_row(raw, first_timestamp_col=0)
    if raw.shape[1] < len(KLINE_COLUMNS):
        raise BinanceVisionError(f"Kline CSV has {raw.shape[1]} columns, expected at least {len(KLINE_COLUMNS)}: {path}")

    df = raw.iloc[:, : len(KLINE_COLUMNS)].copy()
    df.columns = KLINE_COLUMNS
    df["OpenTime"] = _timestamp_series(df["OpenTime"])
    for col in KLINE_OUTPUT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["OpenTime", *KLINE_OUTPUT_COLUMNS])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=KLINE_OUTPUT_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=KLINE_OUTPUT_COLUMNS)

    out = df.set_index("OpenTime")[KLINE_OUTPUT_COLUMNS].sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC")
    out.index.name = "OpenTime"
    out = out[~out.index.duplicated(keep="last")]
    return out


def _parse_bool_series(values: pd.Series) -> pd.Series:
    return values.map(lambda x: str(x).strip().lower() in {"1", "true", "t", "yes", "y"})


def read_agg_trades_zip(path: str | os.PathLike[str]) -> pd.DataFrame:
    raw = _read_csv_no_schema(path)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "price", "quantity", "isBuyerMaker"])
    raw = _drop_header_row(raw, first_timestamp_col=0)
    if raw.shape[1] < len(AGG_TRADE_COLUMNS):
        raise BinanceVisionError(f"aggTrades CSV has {raw.shape[1]} columns, expected at least {len(AGG_TRADE_COLUMNS)}: {path}")

    df = raw.iloc[:, : len(AGG_TRADE_COLUMNS)].copy()
    df.columns = AGG_TRADE_COLUMNS
    df["timestamp"] = _timestamp_series(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["isBuyerMaker"] = _parse_bool_series(df["isBuyerMaker"])
    out = df[["timestamp", "price", "quantity", "isBuyerMaker"]].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["timestamp", "price", "quantity"]).sort_values("timestamp").reset_index(drop=True)
    return out


def read_funding_rate_zip(path: str | os.PathLike[str]) -> pd.DataFrame:
    return _read_csv_with_optional_header(path, FUNDING_RATE_FALLBACK_COLUMNS)


def _concat_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def download_klines(
    symbol: str,
    interval: str,
    start: Any,
    end: Any,
    data_dir: str | None = None,
) -> pd.DataFrame:
    sym = _safe_symbol(symbol)
    start_ts = _parse_time(start)
    end_ts = _parse_time(end)
    if end_ts < start_ts:
        raise ValueError(f"end must be >= start for {sym}: start={start_ts}, end={end_ts}")

    base = Path(data_dir or _default_data_dir()) / "klines" / sym / str(interval)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for month_tag in _month_iter(start_ts, end_ts):
        url, filename = _kline_monthly_url(sym, str(interval), month_tag)
        context = f"klines symbol={sym} interval={interval} month={month_tag}"
        try:
            path = _download_cached(url, base / filename, context)
            frames.append(read_klines_zip(path))
            continue
        except BinanceVisionError as exc:
            errors.append(str(exc))

        for day_tag in _days_in_month(month_tag, start_ts, end_ts):
            day_url, day_filename = _kline_daily_url(sym, str(interval), day_tag)
            day_context = f"klines symbol={sym} interval={interval} day={day_tag}"
            try:
                path = _download_cached(day_url, base / day_filename, day_context)
                frames.append(read_klines_zip(path))
            except BinanceVisionError as exc:
                errors.append(str(exc))

    out = _concat_frames(frames, KLINE_OUTPUT_COLUMNS)
    if out.empty:
        detail = errors[-3:] if errors else ["no files selected"]
        raise BinanceVisionError(f"No Binance Vision klines loaded for symbol={sym}, interval={interval}, start={start}, end={end}. Errors: {detail}")
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)].copy()
    return out


def download_agg_trades(
    symbol: str,
    start: Any,
    end: Any,
    data_dir: str | None = None,
) -> pd.DataFrame:
    sym = _safe_symbol(symbol)
    start_ts = _parse_time(start)
    end_ts = _parse_time(end)
    if end_ts < start_ts:
        raise ValueError(f"end must be >= start for {sym}: start={start_ts}, end={end_ts}")

    base = Path(data_dir or _default_data_dir()) / "aggTrades" / sym
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for month_tag in _month_iter(start_ts, end_ts):
        url, filename = _agg_monthly_url(sym, month_tag)
        context = f"aggTrades symbol={sym} month={month_tag}"
        try:
            path = _download_cached(url, base / filename, context)
            frames.append(read_agg_trades_zip(path))
            continue
        except BinanceVisionError as exc:
            errors.append(str(exc))

        for day_tag in _days_in_month(month_tag, start_ts, end_ts):
            day_url, day_filename = _agg_daily_url(sym, day_tag)
            day_context = f"aggTrades symbol={sym} day={day_tag}"
            try:
                path = _download_cached(day_url, base / day_filename, day_context)
                frames.append(read_agg_trades_zip(path))
            except BinanceVisionError as exc:
                errors.append(str(exc))

    if not frames:
        detail = errors[-3:] if errors else ["no files selected"]
        raise BinanceVisionError(f"No Binance Vision aggTrades loaded for symbol={sym}, start={start}, end={end}. Errors: {detail}")
    out = pd.concat(frames, axis=0).sort_values("timestamp").drop_duplicates(subset=["timestamp", "price", "quantity", "isBuyerMaker"])
    return out.loc[(out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)].reset_index(drop=True)


def download_funding_rates(
    symbol: str,
    start: Any,
    end: Any,
    data_dir: str | None = None,
    return_skipped: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    sym = _safe_symbol(symbol)
    start_ts = _parse_time(start)
    end_ts = _parse_time(end)
    if end_ts < start_ts:
        raise ValueError(f"end must be >= start for {sym}: start={start_ts}, end={end_ts}")

    base = Path(data_dir or _default_data_dir()) / "fundingRate" / sym
    frames: list[pd.DataFrame] = []
    skipped_months: list[str] = []
    errors: list[str] = []
    for month_tag in _month_iter(start_ts, end_ts):
        url, filename = _funding_monthly_url(sym, month_tag)
        context = f"fundingRate symbol={sym} month={month_tag}"
        try:
            path = _download_cached(url, base / filename, context)
            frame = read_funding_rate_zip(path)
            if not frame.empty:
                frame["_source_month"] = month_tag
                frames.append(frame)
        except BinanceVisionError as exc:
            if _is_missing_archive_error(exc):
                skipped_months.append(month_tag)
                continue
            errors.append(str(exc))
            raise

    if not frames:
        out = pd.DataFrame()
    else:
        out = pd.concat(frames, axis=0).reset_index(drop=True)
    if return_skipped:
        return out, skipped_months
    return out
