# ==============================================================================
# 🛰️ provider.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# symbol mapping, shared HTTP session, bounded concurrency, retry/backoff,
# OHLCV fetch, and reliable price retrieval.
# ==============================================================================

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple

import aiohttp
import pandas as pd

from config import CONFIG
from telemetry_utils import logger

# -------------------------
# 📥 BINANCE REST HELPERS
# -------------------------
_BINANCE_EXCHANGE_INFO_CACHE: Optional[dict] = None
_BINANCE_EXCHANGE_INFO_TS: float = 0.0
_BINANCE_SYMBOL_META: Dict[str, dict] = {}
_BINANCE_SHARED_SESSION: Optional[aiohttp.ClientSession] = None
_BINANCE_SESSION_LOCK: Optional[asyncio.Lock] = None
_BINANCE_REQUEST_SEMAPHORE: Optional[asyncio.Semaphore] = None
_BINANCE_SPACING_LOCK: Optional[asyncio.Lock] = None
_BINANCE_LAST_REQUEST_TS: float = 0.0

_BOOK_TICKER_CACHE: Dict[str, Tuple[float, dict]] = {}
_PRICE_TICKER_CACHE: Dict[str, Tuple[float, dict]] = {}
_DEPTH_CACHE: Dict[str, Tuple[float, dict]] = {}
_AGGTRADES_CACHE: Dict[str, Tuple[float, list]] = {}

_WARNED_UNMAPPED_SYMBOLS: set[str] = set()
_WARNED_UNAVAILABLE_SYMBOLS: set[str] = set()
_WARNED_NON_TRADING_SYMBOLS: set[str] = set()


def _warn_once(bucket: set[str], key: str, message: str) -> None:
    if key not in bucket:
        bucket.add(key)
        logger.warning(message)


def _to_binance_symbol(internal_symbol: str) -> Optional[str]:
    sym_map = CONFIG.get("BINANCE_SYMBOL_MAP", {})
    mapped = None
    if isinstance(sym_map, dict):
        mapped = sym_map.get(internal_symbol)

    if mapped:
        return str(mapped).strip().upper()

    # Fallback for simple internal symbols like XXX-USD -> XXXUSDT
    if isinstance(internal_symbol, str) and internal_symbol.endswith("-USD"):
        return f"{internal_symbol[:-4]}USDT".upper()

    return None


def _interval_to_ms(interval: str) -> Optional[int]:
    iv = str(interval).strip().lower()
    if not iv:
        return None

    unit = iv[-1]
    try:
        num = int(iv[:-1])
    except Exception:
        return None

    if num <= 0:
        return None

    if unit == "m":
        return num * 60 * 1000
    if unit == "h":
        return num * 60 * 60 * 1000
    if unit == "d":
        return num * 24 * 60 * 60 * 1000
    if unit == "w":
        return num * 7 * 24 * 60 * 60 * 1000

    return None


def _period_to_required_candles(period: str, interval: str) -> int:
    iv_ms = _interval_to_ms(interval)
    if not iv_ms:
        return 0

    p = str(period).strip().lower()
    qty = None
    duration_ms = None

    try:
        if p.endswith("mo"):
            qty = int(p[:-2])
            duration_ms = qty * 30 * 24 * 60 * 60 * 1000
        elif p.endswith("wk"):
            qty = int(p[:-2])
            duration_ms = qty * 7 * 24 * 60 * 60 * 1000
        elif p.endswith("y"):
            qty = int(p[:-1])
            duration_ms = qty * 365 * 24 * 60 * 60 * 1000
        elif p.endswith("d"):
            qty = int(p[:-1])
            duration_ms = qty * 24 * 60 * 60 * 1000
    except Exception:
        duration_ms = None

    if duration_ms is None or duration_ms <= 0:
        return 0

    # +2 to stay close to previous provider semantics around boundary bars.
    return max(1, int(duration_ms // iv_ms) + 2)


def _binance_base_url() -> str:
    base = str(CONFIG.get("BINANCE_PUBLIC_BASE_URL", "https://data-api.binance.vision")).strip()
    return base.rstrip("/")


def _get_binance_session_lock() -> asyncio.Lock:
    global _BINANCE_SESSION_LOCK
    if _BINANCE_SESSION_LOCK is None:
        _BINANCE_SESSION_LOCK = asyncio.Lock()
    return _BINANCE_SESSION_LOCK


def _get_binance_spacing_lock() -> asyncio.Lock:
    global _BINANCE_SPACING_LOCK
    if _BINANCE_SPACING_LOCK is None:
        _BINANCE_SPACING_LOCK = asyncio.Lock()
    return _BINANCE_SPACING_LOCK


def _get_binance_request_semaphore() -> asyncio.Semaphore:
    global _BINANCE_REQUEST_SEMAPHORE
    if _BINANCE_REQUEST_SEMAPHORE is None:
        max_conc = max(1, int(CONFIG.get("BINANCE_MAX_CONCURRENT_REQUESTS", 4)))
        _BINANCE_REQUEST_SEMAPHORE = asyncio.Semaphore(max_conc)
    return _BINANCE_REQUEST_SEMAPHORE


async def _get_shared_binance_session() -> aiohttp.ClientSession:
    global _BINANCE_SHARED_SESSION

    if _BINANCE_SHARED_SESSION is not None and not _BINANCE_SHARED_SESSION.closed:
        return _BINANCE_SHARED_SESSION

    lock = _get_binance_session_lock()
    async with lock:
        if _BINANCE_SHARED_SESSION is not None and not _BINANCE_SHARED_SESSION.closed:
            return _BINANCE_SHARED_SESSION

        max_conc = max(1, int(CONFIG.get("BINANCE_MAX_CONCURRENT_REQUESTS", 4)))
        connector = aiohttp.TCPConnector(
            limit=max(8, max_conc * 2),
            limit_per_host=max(4, max_conc),
            ttl_dns_cache=300,
        )
        _BINANCE_SHARED_SESSION = aiohttp.ClientSession(connector=connector, raise_for_status=False)
        return _BINANCE_SHARED_SESSION


async def close_binance_http_session() -> None:
    global _BINANCE_SHARED_SESSION
    if _BINANCE_SHARED_SESSION is not None and not _BINANCE_SHARED_SESSION.closed:
        await _BINANCE_SHARED_SESSION.close()
    _BINANCE_SHARED_SESSION = None


async def _enforce_min_request_spacing() -> None:
    global _BINANCE_LAST_REQUEST_TS

    spacing = max(0.0, float(CONFIG.get("BINANCE_MIN_REQUEST_SPACING_SECONDS", 0.0)))
    if spacing <= 0.0:
        return

    lock = _get_binance_spacing_lock()
    async with lock:
        now = time.monotonic()
        elapsed = now - _BINANCE_LAST_REQUEST_TS
        if elapsed < spacing:
            await asyncio.sleep(spacing - elapsed)
        _BINANCE_LAST_REQUEST_TS = time.monotonic()


def _parse_retry_after_seconds(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None

    try:
        seconds = float(str(raw).strip())
        if seconds >= 0:
            return seconds
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(str(raw).strip())
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds = (dt - now).total_seconds()
        if seconds >= 0:
            return seconds
    except Exception:
        pass

    return None


def _compute_backoff_seconds(attempt: int, retry_after_seconds: Optional[float] = None) -> float:
    respect_retry_after = bool(CONFIG.get("BINANCE_RESPECT_RETRY_AFTER", True))
    if respect_retry_after and retry_after_seconds is not None and retry_after_seconds >= 0:
        return float(retry_after_seconds)

    base = max(0.05, float(CONFIG.get("BINANCE_BACKOFF_BASE_SECONDS", 0.75)))
    max_backoff = max(base, float(CONFIG.get("BINANCE_BACKOFF_MAX_SECONDS", 8.0)))

    exp_wait = min(max_backoff, base * (2 ** max(0, int(attempt))))
    jitter = random.uniform(0.0, min(0.35, exp_wait * 0.2))
    return min(max_backoff, exp_wait + jitter)


async def _binance_get_json(path: str, params: Optional[dict] = None) -> Any:
    timeout_sec = float(CONFIG.get("BINANCE_TIMEOUT_SECONDS", 10))
    retries = max(0, int(CONFIG.get("BINANCE_RETRY_COUNT", 2)))
    url = f"{_binance_base_url()}{path}"
    sem = _get_binance_request_semaphore()

    for attempt in range(retries + 1):
        should_retry = False
        sleep_seconds = 0.0

        try:
            async with sem:
                await _enforce_min_request_spacing()
                session = await _get_shared_binance_session()
                timeout = aiohttp.ClientTimeout(total=timeout_sec)

                async with session.get(url, params=params, timeout=timeout) as resp:
                    if 200 <= resp.status < 300:
                        try:
                            return await resp.json(content_type=None)
                        except Exception:
                            logger.warning(f"⚠️ Binance REST invalid JSON on {path}.")
                            return None

                    status = int(resp.status)

                    if status in {418, 429}:
                        retry_after = _parse_retry_after_seconds(resp.headers.get("Retry-After"))
                        should_retry = attempt < retries
                        sleep_seconds = _compute_backoff_seconds(attempt, retry_after_seconds=retry_after)
                        logger.warning(
                            f"⚠️ Binance REST throttle response {status} on {path}. "
                            f"{'Retrying' if should_retry else 'Retries exhausted'} "
                            f"(wait={sleep_seconds:.2f}s)."
                        )
                    elif status in {500, 502, 503, 504}:
                        should_retry = attempt < retries
                        sleep_seconds = _compute_backoff_seconds(attempt)
                        logger.warning(
                            f"⚠️ Binance REST transient response {status} on {path}. "
                            f"{'Retrying' if should_retry else 'Retries exhausted'} "
                            f"(wait={sleep_seconds:.2f}s)."
                        )
                    else:
                        return None

        except asyncio.TimeoutError:
            should_retry = attempt < retries
            sleep_seconds = _compute_backoff_seconds(attempt)
            if should_retry:
                logger.warning(f"⚠️ Binance REST timeout on {path}. Retrying in {sleep_seconds:.2f}s.")
            else:
                logger.warning(f"⚠️ Binance REST timeout on {path}. Retries exhausted.")
        except aiohttp.ClientError as e:
            should_retry = attempt < retries
            sleep_seconds = _compute_backoff_seconds(attempt)
            if should_retry:
                logger.warning(f"⚠️ Binance REST client error on {path}: {type(e).__name__}. Retrying in {sleep_seconds:.2f}s.")
            else:
                logger.warning(f"⚠️ Binance REST client error on {path}: {type(e).__name__}. Retries exhausted.")
        except Exception as e:
            should_retry = attempt < retries
            sleep_seconds = _compute_backoff_seconds(attempt)
            if should_retry:
                logger.warning(f"⚠️ Binance REST unexpected error on {path}: {type(e).__name__}. Retrying in {sleep_seconds:.2f}s.")
            else:
                logger.warning(f"⚠️ Binance REST unexpected error on {path}: {type(e).__name__}. Retries exhausted.")

        if should_retry:
            await asyncio.sleep(max(0.0, float(sleep_seconds)))
            continue
        return None

    return None


async def _get_binance_symbol_meta(binance_symbol: str) -> Optional[dict]:
    global _BINANCE_EXCHANGE_INFO_CACHE, _BINANCE_EXCHANGE_INFO_TS, _BINANCE_SYMBOL_META

    now = time.time()
    ttl = max(30, int(CONFIG.get("BINANCE_EXCHANGE_INFO_TTL_SECONDS", 3600)))
    cache_expired = (now - _BINANCE_EXCHANGE_INFO_TS) >= ttl

    if _BINANCE_EXCHANGE_INFO_CACHE is None or cache_expired:
        data = await _binance_get_json("/api/v3/exchangeInfo")
        if isinstance(data, dict):
            _BINANCE_EXCHANGE_INFO_CACHE = data
            _BINANCE_EXCHANGE_INFO_TS = now
            symbols = data.get("symbols", [])
            if isinstance(symbols, list):
                parsed: Dict[str, dict] = {}
                for rec in symbols:
                    if not isinstance(rec, dict):
                        continue
                    k = str(rec.get("symbol", "")).upper()
                    if k:
                        parsed[k] = rec
                _BINANCE_SYMBOL_META = parsed

    return _BINANCE_SYMBOL_META.get(binance_symbol)


async def _resolve_binance_symbol(internal_symbol: str) -> Optional[str]:
    binance_symbol = _to_binance_symbol(internal_symbol)
    if not binance_symbol:
        _warn_once(
            _WARNED_UNMAPPED_SYMBOLS,
            internal_symbol,
            f"⚠️ No Binance symbol mapping configured for internal symbol {internal_symbol}. Skipping symbol.",
        )
        return None

    meta = await _get_binance_symbol_meta(binance_symbol)
    if not isinstance(meta, dict):
        _warn_once(
            _WARNED_UNAVAILABLE_SYMBOLS,
            internal_symbol,
            f"⚠️ Binance spot symbol metadata not found for {internal_symbol} -> {binance_symbol}. Skipping symbol.",
        )
        return None

    status = str(meta.get("status", "")).upper()
    is_spot_allowed = bool(meta.get("isSpotTradingAllowed", True))

    perms = meta.get("permissions")
    if isinstance(perms, list) and perms:
        is_spot_allowed = is_spot_allowed and ("SPOT" in perms)

    if status != "TRADING" or not is_spot_allowed:
        _warn_once(
            _WARNED_NON_TRADING_SYMBOLS,
            internal_symbol,
            f"⚠️ Binance symbol not tradable on Spot right now for {internal_symbol} -> {binance_symbol} "
            f"(status={status}, spot_allowed={is_spot_allowed}). Skipping symbol.",
        )
        return None

    return binance_symbol


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
        if pd.isna(out):
            return None
        return out
    except Exception:
        return None


def _read_ttl_cache(
    cache: Dict[str, Tuple[float, Any]],
    key: str,
    ttl_seconds: float,
    expected_type: type | None = None,
) -> Optional[Any]:
    if ttl_seconds <= 0:
        return None

    cached = cache.get(key)
    if not cached:
        return None

    ts, payload = cached
    if (time.monotonic() - ts) <= ttl_seconds:
        if expected_type is not None and not isinstance(payload, expected_type):
            return None
        return payload

    cache.pop(key, None)
    return None


def _write_ttl_cache(cache: Dict[str, Tuple[float, Any]], key: str, payload: Any, ttl_seconds: float) -> None:
    if ttl_seconds <= 0 or payload is None:
        return
    cache[key] = (time.monotonic(), payload)


def _klines_to_ohlcv_df(rows: list) -> pd.DataFrame:
    parsed: list[tuple[int, float, float, float, float, float]] = []

    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) < 6:
            continue

        try:
            open_time = int(r[0])
        except Exception:
            continue

        o = _to_float(r[1])
        h = _to_float(r[2])
        l = _to_float(r[3])
        c = _to_float(r[4])
        v = _to_float(r[5])

        if None in (o, h, l, c, v):
            continue

        parsed.append((open_time, o, h, l, c, v))

    if not parsed:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(parsed, columns=["OpenTime", "Open", "High", "Low", "Close", "Volume"])
    df.sort_values("OpenTime", inplace=True)
    df.drop_duplicates(subset=["OpenTime"], keep="last", inplace=True)
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df.set_index("OpenTime", inplace=True)

    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out.dropna(inplace=True)
    return out


async def _fetch_binance_klines_chunk(
    binance_symbol: str,
    interval: str,
    limit: int,
    end_time_ms: Optional[int] = None,
) -> list:
    params: Dict[str, Any] = {
        "symbol": binance_symbol,
        "interval": interval,
        "limit": max(1, min(1000, int(limit))),
    }
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    data = await _binance_get_json("/api/v3/klines", params=params)
    return data if isinstance(data, list) else []


async def binance_download_single(symbol: str, period: str, interval: str) -> pd.DataFrame:
    binance_symbol = await _resolve_binance_symbol(symbol)
    if not binance_symbol:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    need = _period_to_required_candles(period, interval)
    if need <= 0:
        logger.warning(f"⚠️ Unsupported period/interval for Binance klines: period={period}, interval={interval}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    rows: list = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    try:
        while len(rows) < need:
            remaining = need - len(rows)
            chunk_limit = min(1000, max(1, remaining))

            chunk = await _fetch_binance_klines_chunk(binance_symbol, interval, chunk_limit, end_time_ms=end_ms)
            if not chunk:
                break

            rows = chunk + rows

            try:
                oldest_open_time = int(chunk[0][0])
            except Exception:
                break

            next_end = oldest_open_time - 1
            if next_end <= 0 or next_end >= end_ms:
                break

            end_ms = next_end

            if len(chunk) < chunk_limit:
                break

            await asyncio.sleep(float(CONFIG.get("DOWNLOAD_DELAY_SECONDS", 0.15)))
    except Exception as e:
        logger.warning(f"⚠️ Binance klines fetch failed for {symbol} ({binance_symbol}): {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if len(rows) > need:
        rows = rows[-need:]

    return _klines_to_ohlcv_df(rows)


def _neutral_microstructure_features(
    config: dict | None,
    reason: str,
    ofi_reason: str | None = None,
    vpin_reason: str | None = None,
) -> dict:
    cfg = config or {}
    try:
        levels = int(cfg.get("OFI_LEVELS", 10))
    except Exception:
        levels = 10

    r = str(reason) if reason else "not_computed"
    return {
        "microstructure_ok": False,
        "microstructure_reason": r,
        "ofi_ok": False,
        "ofi_reason": ofi_reason if ofi_reason is not None else r,
        "ofi_l1": 0.0,
        "ofi_l1_norm": 0.0,
        "ofi_l1_norm_denom_qty": 0.0,
        "ofi_l1_z": 0.0,
        "ofi_depth_imbalance": 0.0,
        "ofi_spread_bps": 0.0,
        "ofi_liquidity_usd": 0.0,
        "ofi_snapshot_count": 0,
        "ofi_levels": int(max(1, levels)),
        "vpin_ok": False,
        "vpin_reason": vpin_reason if vpin_reason is not None else r,
        "vpin": 0.0,
        "vpin_z": 0.0,
        "vpin_bucket_count": 0,
        "vpin_bucket_volume": 0.0,
        "vpin_volume_unit": "quote",
        "vpin_buy_volume": 0.0,
        "vpin_sell_volume": 0.0,
        "vpin_total_volume": 0.0,
        "vpin_imbalance_sum": 0.0,
        "vpin_denominator": 0.0,
        "vpin_lower_bound": 0.0,
        "vpin_formula_check_ok": False,
        "vpin_last_trade_ts": None,
    }


async def fetch_binance_depth(
    internal_symbol: str,
    limit: int | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Optional[dict]:
    binance_symbol = await _resolve_binance_symbol(internal_symbol)
    if not binance_symbol:
        return None

    if force_refresh:
        use_cache = False

    cfg_limit = int(CONFIG.get("OFI_DEPTH_LIMIT", 100))
    depth_limit = max(1, min(5000, int(limit if limit is not None else cfg_limit)))
    ttl = max(0.0, float(CONFIG.get("BINANCE_DEPTH_TTL_SECONDS", 1.0)))
    cache_key = f"{binance_symbol}:{depth_limit}"

    if use_cache:
        cached = _read_ttl_cache(_DEPTH_CACHE, cache_key, ttl, expected_type=dict)
        if cached is not None:
            return cached

    data = await _binance_get_json(
        "/api/v3/depth",
        params={"symbol": binance_symbol, "limit": depth_limit},
    )
    if isinstance(data, dict):
        _write_ttl_cache(_DEPTH_CACHE, cache_key, data, ttl)
        return data
    return None


async def fetch_binance_agg_trades(
    internal_symbol: str,
    lookback_minutes: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    binance_symbol = await _resolve_binance_symbol(internal_symbol)
    if not binance_symbol:
        return []

    cfg_lookback = int(CONFIG.get("VPIN_LOOKBACK_MINUTES", 60))
    cfg_limit = int(CONFIG.get("VPIN_AGGTRADES_LIMIT", 1000))
    lookback = max(1, int(lookback_minutes if lookback_minutes is not None else cfg_lookback))
    trades_limit = max(1, min(1000, int(limit if limit is not None else cfg_limit)))

    ttl = max(0.0, float(CONFIG.get("BINANCE_AGGTRADES_TTL_SECONDS", 2.0)))
    cache_key = f"{binance_symbol}:{lookback}:{trades_limit}"
    cached = _read_ttl_cache(_AGGTRADES_CACHE, cache_key, ttl, expected_type=list)
    if cached is not None:
        return cached

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = max(0, now_ms - (lookback * 60 * 1000))

    data = await _binance_get_json(
        "/api/v3/aggTrades",
        params={
            "symbol": binance_symbol,
            "startTime": int(start_ms),
            "endTime": int(now_ms),
            "limit": int(trades_limit),
        },
    )

    if isinstance(data, list):
        payload = [rec for rec in data if isinstance(rec, dict)]
        _write_ttl_cache(_AGGTRADES_CACHE, cache_key, payload, ttl)
        return payload
    return []


async def collect_depth_snapshots(internal_symbol: str, config: dict) -> list[dict]:
    cfg = config or {}
    count = max(1, int(cfg.get("OFI_SNAPSHOT_COUNT", 3)))
    interval_seconds = max(0.0, float(cfg.get("OFI_SNAPSHOT_INTERVAL_SECONDS", 0.35)))
    depth_limit = max(1, int(cfg.get("OFI_DEPTH_LIMIT", 100)))
    bypass_depth_cache = bool(cfg.get("OFI_BYPASS_DEPTH_CACHE", True))

    snapshots: list[dict] = []
    for idx in range(count):
        try:
            snap = await fetch_binance_depth(
                internal_symbol,
                limit=depth_limit,
                use_cache=(not bypass_depth_cache),
            )
            if isinstance(snap, dict):
                snapshots.append(snap)
        except Exception as e:
            logger.warning(f"⚠️ Depth snapshot fetch failed for {internal_symbol}: {type(e).__name__}: {e}")

        if idx < (count - 1) and interval_seconds > 0:
            await asyncio.sleep(interval_seconds)

    return snapshots


async def build_microstructure_features(internal_symbol: str, config: dict) -> dict:
    cfg = config or {}
    enabled = bool(cfg.get("MICROSTRUCTURE_FEATURES_ENABLED", False))
    fail_open = bool(cfg.get("MICROSTRUCTURE_FAIL_OPEN", True))

    if not enabled:
        return _neutral_microstructure_features(
            cfg,
            reason="microstructure_disabled",
            ofi_reason="ofi_disabled",
            vpin_reason="vpin_disabled",
        )

    try:
        import OFI
        import VPIN

        out = _neutral_microstructure_features(cfg, reason="not_computed")
        ofi_enabled = bool(cfg.get("OFI_ENABLED", True))
        vpin_enabled = bool(cfg.get("VPIN_ENABLED", True))

        ofi_features: dict = _neutral_microstructure_features(cfg, reason="ofi_not_computed")
        vpin_features: dict = _neutral_microstructure_features(cfg, reason="vpin_not_computed")

        if ofi_enabled:
            try:
                depth_snapshots = await collect_depth_snapshots(internal_symbol, cfg)
                built = OFI.build_ofi_features(depth_snapshots, cfg)
                if isinstance(built, dict):
                    ofi_features = built
                else:
                    ofi_features = {"ofi_ok": False, "ofi_reason": "ofi_invalid_output"}
            except Exception as e:
                ofi_features = {"ofi_ok": False, "ofi_reason": f"ofi_exception:{type(e).__name__}"}
        else:
            ofi_features = {"ofi_ok": False, "ofi_reason": "ofi_disabled"}

        if vpin_enabled:
            try:
                raw_trades = await fetch_binance_agg_trades(
                    internal_symbol,
                    lookback_minutes=int(cfg.get("VPIN_LOOKBACK_MINUTES", 60)),
                    limit=int(cfg.get("VPIN_AGGTRADES_LIMIT", 1000)),
                )
                built = VPIN.build_vpin_features(raw_trades, cfg)
                if isinstance(built, dict):
                    vpin_features = built
                else:
                    vpin_features = {"vpin_ok": False, "vpin_reason": "vpin_invalid_output"}
            except Exception as e:
                vpin_features = {"vpin_ok": False, "vpin_reason": f"vpin_exception:{type(e).__name__}"}
        else:
            vpin_features = {"vpin_ok": False, "vpin_reason": "vpin_disabled"}

        out.update({k: v for k, v in ofi_features.items() if str(k).startswith("ofi_")})
        # Keep "vpin" and all "vpin_*" keys.
        out.update({k: v for k, v in vpin_features.items() if (k == "vpin" or str(k).startswith("vpin_"))})

        ofi_ok = bool(out.get("ofi_ok", False)) if ofi_enabled else True
        vpin_ok = bool(out.get("vpin_ok", False)) if vpin_enabled else True
        micro_ok = bool(ofi_ok and vpin_ok)

        reasons: list[str] = []
        if ofi_enabled and not ofi_ok:
            reasons.append(str(out.get("ofi_reason") or "ofi_not_ok"))
        if vpin_enabled and not vpin_ok:
            reasons.append(str(out.get("vpin_reason") or "vpin_not_ok"))

        out["microstructure_ok"] = micro_ok
        out["microstructure_reason"] = None if micro_ok else (";".join(reasons) if reasons else "not_ok")
        return out
    except Exception as e:
        logger.warning(f"⚠️ Microstructure feature build failed for {internal_symbol}: {type(e).__name__}: {e}")
        if fail_open:
            return _neutral_microstructure_features(
                cfg,
                reason=f"microstructure_exception:{type(e).__name__}",
                ofi_reason=f"microstructure_exception:{type(e).__name__}",
                vpin_reason=f"microstructure_exception:{type(e).__name__}",
            )
        raise


async def _fetch_book_ticker(binance_symbol: str) -> Optional[dict]:
    ttl = max(0.0, float(CONFIG.get("BINANCE_BOOK_TICKER_TTL_SECONDS", 2.0)))
    cached = _read_ttl_cache(_BOOK_TICKER_CACHE, binance_symbol, ttl)
    if cached is not None:
        return cached

    data = await _binance_get_json("/api/v3/ticker/bookTicker", params={"symbol": binance_symbol})
    if isinstance(data, dict):
        _write_ttl_cache(_BOOK_TICKER_CACHE, binance_symbol, data, ttl)
        return data
    return None


async def _fetch_ticker_price(binance_symbol: str) -> Optional[dict]:
    ttl = max(0.0, float(CONFIG.get("BINANCE_PRICE_TTL_SECONDS", 2.0)))
    cached = _read_ttl_cache(_PRICE_TICKER_CACHE, binance_symbol, ttl)
    if cached is not None:
        return cached

    data = await _binance_get_json("/api/v3/ticker/price", params={"symbol": binance_symbol})
    if isinstance(data, dict):
        _write_ttl_cache(_PRICE_TICKER_CACHE, binance_symbol, data, ttl)
        return data
    return None


async def _fetch_ticker_24hr(binance_symbol: str) -> Optional[dict]:
    data = await _binance_get_json("/api/v3/ticker/24hr", params={"symbol": binance_symbol})
    return data if isinstance(data, dict) else None


async def _last_closed_candle_price(binance_symbol: str, interval: str) -> Optional[float]:
    rows = await _fetch_binance_klines_chunk(binance_symbol, interval, limit=3, end_time_ms=None)
    if not rows:
        return None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for row in reversed(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        close_price = _to_float(row[4])
        if close_price is None or close_price <= 0:
            continue
        try:
            close_time = int(row[6])
        except Exception:
            close_time = 0

        if close_time < now_ms:
            return close_price

    # Last fallback: take final row close if present.
    try:
        return _to_float(rows[-1][4])
    except Exception:
        return None


async def get_reliable_price(symbol: str) -> Optional[float]:
    binance_symbol = await _resolve_binance_symbol(symbol)
    if not binance_symbol:
        return None

    try:
        bt = await _fetch_book_ticker(binance_symbol)
        if bt:
            bid = _to_float(bt.get("bidPrice"))
            ask = _to_float(bt.get("askPrice"))

            if bid is not None and ask is not None and bid > 0 and ask > 0:
                return float((bid + ask) / 2.0)
            if bid is not None and bid > 0:
                return float(bid)
            if ask is not None and ask > 0:
                return float(ask)
    except Exception:
        pass

    try:
        tp = await _fetch_ticker_price(binance_symbol)
        if tp:
            p = _to_float(tp.get("price"))
            if p is not None and p > 0:
                return float(p)
    except Exception:
        pass

    try:
        t24 = await _fetch_ticker_24hr(binance_symbol)
        if t24:
            p = _to_float(t24.get("lastPrice"))
            if p is not None and p > 0:
                return float(p)
    except Exception:
        pass

    try:
        p = await _last_closed_candle_price(binance_symbol, CONFIG["TIMEFRAME"])
        if p is not None and p > 0:
            return float(p)
    except Exception:
        pass

    return None
