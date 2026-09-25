# ==============================================================================
# tests/hmm_walkforward.py
# ------------------------------------------------------------------------------
# HMM walk-forward diagnostics and forward-return regime scoring.
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from hmm import infer_hmm_regime  # noqa: E402


BACKENDS = ["pomegranate_normal", "pomegranate_gmm"]
ROW_FIELDS = [
    "timestamp",
    "symbol",
    "timeframe",
    "backend",
    "close",
    "fwd_log_return",
    "hmm_ok",
    "hmm_regime",
    "hmm_policy",
    "hmm_confidence",
    "hmm_bull_prob",
    "hmm_neutral_prob",
    "hmm_bear_prob",
    "hmm_reason",
    "hmm_state_prob_max",
    "hmm_state_prob_min",
    "hmm_state_prob_margin",
    "hmm_regime_prob_margin",
    "hmm_state_prob_entropy",
    "hmm_effective_emission_backend",
    "hmm_mapping_method",
    "fit_seconds",
]
SUMMARY_FIELDS = [
    "backend",
    "summary_scope",
    "rows",
    "hmm_ok_rate",
    "bull_pct",
    "neutral_pct",
    "bear_pct",
    "allow_pct",
    "caution_pct",
    "block_pct",
    "avg_confidence",
    "regime_switch_count",
    "fallback_pct",
    "avg_state_prob_margin",
    "avg_regime_prob_margin",
    "avg_fit_seconds",
    "max_fit_seconds",
    "total_fit_seconds",
    "mapping_matched_pct",
    "mapping_score_order_pct",
    "mapping_none_pct",
    "mapping_other_pct",
    "fwd_rows",
    "fwd_bull_n",
    "fwd_bull_mean",
    "fwd_bull_hit",
    "fwd_neutral_n",
    "fwd_neutral_mean",
    "fwd_neutral_hit",
    "fwd_bear_n",
    "fwd_bear_mean",
    "fwd_bear_hit",
    "fwd_regime_spread",
    "fwd_allow_mean",
    "fwd_block_mean",
    "fwd_policy_spread",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _binance_symbol(symbol: str) -> str:
    cleaned = str(symbol or "").strip().upper()
    if not cleaned:
        return "BTCUSDT"
    if cleaned.endswith("-USD"):
        return f"{cleaned[:-4]}USDT"
    return cleaned.replace("/", "").replace("-", "")


def _fetch_klines_direct(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    base_url = str(CONFIG.get("BINANCE_PUBLIC_BASE_URL", "https://data-api.binance.vision")).rstrip("/")
    binance_symbol = _binance_symbol(symbol)
    remaining = max(1, int(limit))
    end_time = None
    rows: list[list[Any]] = []

    while remaining > 0:
        batch_limit = min(1000, remaining)
        params: dict[str, Any] = {
            "symbol": binance_symbol,
            "interval": timeframe,
            "limit": batch_limit,
        }
        if end_time is not None:
            params["endTime"] = int(end_time)

        url = f"{base_url}/api/v3/klines?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        if not isinstance(payload, list) or not payload:
            break

        rows = payload + rows
        remaining -= len(payload)

        try:
            oldest_open = int(payload[0][0])
        except Exception:
            break
        next_end = oldest_open - 1
        if next_end <= 0 or next_end == end_time:
            break
        end_time = next_end
        time.sleep(0.12)

        if len(payload) < batch_limit:
            break

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if len(rows) > limit:
        rows = rows[-limit:]

    parsed: list[tuple[int, float, float, float, float, float]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            parsed.append(
                (
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                )
            )
        except Exception:
            continue

    df = pd.DataFrame(parsed, columns=["OpenTime", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.drop_duplicates(subset=["OpenTime"], keep="last", inplace=True)
    df.sort_values("OpenTime", inplace=True)
    df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
    df.set_index("OpenTime", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()


def fetch_ohlcv(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return _fetch_klines_direct(symbol=symbol, timeframe=timeframe, limit=limit)


def _walk_backend(df: pd.DataFrame, symbol: str, timeframe: str, backend: str, window: int, step: int, horizon: int) -> list[dict[str, Any]]:
    cfg = dict(CONFIG)
    cfg["HMM_EMISSION_BACKEND"] = backend
    cfg["HMM_GMM_FALLBACK_TO_NORMAL"] = True
    # Keep this diagnostic usable with short walk-forward windows. Production
    # config remains untouched; only the local inspection run is relaxed.
    cfg["HMM_MIN_FEATURE_ROWS"] = min(
        int(cfg.get("HMM_MIN_FEATURE_ROWS", 300)),
        max(60, int(window * 0.5)),
    )

    rows: list[dict[str, Any]] = []
    total = max(0, len(df) - window)
    for n, i in enumerate(range(window, len(df), max(1, int(step))), start=1):
        df_window = df.iloc[i - window : i]
        t0 = time.perf_counter()
        result = infer_hmm_regime(df_window, cfg, symbol=f"{symbol}:{timeframe}:{backend}:walkforward")
        fit_seconds = time.perf_counter() - t0
        ts = df.index[i]
        close_now = _safe_float(df["Close"].iloc[i])
        fwd_log_return = None
        if i + horizon < len(df):
            close_future = _safe_float(df["Close"].iloc[i + horizon])
            if close_now > 0.0 and close_future > 0.0:
                fwd_log_return = math.log(close_future / close_now)
        rows.append(
            {
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "symbol": symbol,
                "timeframe": timeframe,
                "backend": backend,
                "close": close_now,
                "fwd_log_return": fwd_log_return,
                "hmm_ok": bool(result.get("hmm_ok", False)),
                "hmm_regime": result.get("hmm_regime", "unknown"),
                "hmm_policy": result.get("hmm_policy", "caution"),
                "hmm_confidence": _safe_float(result.get("hmm_confidence"), 0.0),
                "hmm_bull_prob": _safe_float(result.get("hmm_bull_prob"), 0.0),
                "hmm_neutral_prob": _safe_float(result.get("hmm_neutral_prob"), 1.0),
                "hmm_bear_prob": _safe_float(result.get("hmm_bear_prob"), 0.0),
                "hmm_reason": str(result.get("hmm_reason", "")),
                "hmm_state_prob_max": _safe_float(result.get("hmm_state_prob_max"), 0.0),
                "hmm_state_prob_min": _safe_float(result.get("hmm_state_prob_min"), 0.0),
                "hmm_state_prob_margin": _safe_float(result.get("hmm_state_prob_margin"), 0.0),
                "hmm_regime_prob_margin": _safe_float(result.get("hmm_regime_prob_margin"), 0.0),
                "hmm_state_prob_entropy": _safe_float(result.get("hmm_state_prob_entropy"), 0.0),
                "hmm_effective_emission_backend": result.get("hmm_effective_emission_backend"),
                "hmm_mapping_method": result.get("hmm_mapping_method", "none"),
                "fit_seconds": fit_seconds,
            }
        )
        if n == 1 or n % 50 == 0 or n >= total:
            print(f"{backend}: {n}/{total} windows", flush=True)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _pct(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def _fwd_group_stats(rows: list[dict[str, Any]], field: str, value: str) -> tuple[int, float, float]:
    values = [
        _safe_float(row.get("fwd_log_return"), 0.0)
        for row in rows
        if str(row.get(field)) == value and math.isfinite(_safe_float(row.get("fwd_log_return"), float("nan")))
    ]
    n = len(values)
    if n <= 0:
        return 0, 0.0, 0.0
    return n, sum(values) / n, _pct(sum(1 for item in values if item > 0.0), n)


def _fwd_policy_mean(rows: list[dict[str, Any]], policy: str) -> tuple[int, float]:
    values = [
        _safe_float(row.get("fwd_log_return"), 0.0)
        for row in rows
        if str(row.get("hmm_policy")) == policy and math.isfinite(_safe_float(row.get("fwd_log_return"), float("nan")))
    ]
    if not values:
        return 0, 0.0
    return len(values), sum(values) / len(values)


def summarize(rows: list[dict[str, Any]], backend: str, summary_scope: str = "all") -> dict[str, Any]:
    total = len(rows)
    regimes = [str(row.get("hmm_regime", "unknown")) for row in rows]
    policies = [str(row.get("hmm_policy", "caution")) for row in rows]
    switch_count = sum(
        1
        for prev, curr in zip(rows, rows[1:])
        if bool(prev.get("hmm_ok", False))
        and bool(curr.get("hmm_ok", False))
        and str(prev.get("hmm_regime", "unknown")) != str(curr.get("hmm_regime", "unknown"))
    )
    fallback_count = sum(
        1
        for row in rows
        if str(row.get("backend")) == "pomegranate_gmm"
        and str(row.get("hmm_effective_emission_backend")) != "pomegranate_gmm"
    )
    avg_conf = sum(_safe_float(row.get("hmm_confidence"), 0.0) for row in rows) / total if total else 0.0
    fit_seconds = [_safe_float(row.get("fit_seconds"), 0.0) for row in rows]
    mapping_methods = [str(row.get("hmm_mapping_method", "none")) for row in rows]
    known_mapping = {"matched", "score_order", "none"}
    fwd_rows = [
        row
        for row in rows
        if bool(row.get("hmm_ok", False)) and math.isfinite(_safe_float(row.get("fwd_log_return"), float("nan")))
    ]
    fwd_bull_n, fwd_bull_mean, fwd_bull_hit = _fwd_group_stats(fwd_rows, "hmm_regime", "bull")
    fwd_neutral_n, fwd_neutral_mean, fwd_neutral_hit = _fwd_group_stats(fwd_rows, "hmm_regime", "neutral")
    fwd_bear_n, fwd_bear_mean, fwd_bear_hit = _fwd_group_stats(fwd_rows, "hmm_regime", "bear")
    fwd_allow_n, fwd_allow_mean = _fwd_policy_mean(fwd_rows, "allow")
    fwd_block_n, fwd_block_mean = _fwd_policy_mean(fwd_rows, "block")
    return {
        "backend": backend,
        "summary_scope": summary_scope,
        "rows": total,
        "hmm_ok_rate": _pct(sum(1 for row in rows if bool(row.get("hmm_ok", False))), total),
        "bull_pct": _pct(regimes.count("bull"), total),
        "neutral_pct": _pct(regimes.count("neutral"), total),
        "bear_pct": _pct(regimes.count("bear"), total),
        "allow_pct": _pct(policies.count("allow"), total),
        "caution_pct": _pct(policies.count("caution"), total),
        "block_pct": _pct(policies.count("block"), total),
        "avg_confidence": avg_conf,
        "regime_switch_count": switch_count,
        "fallback_pct": _pct(fallback_count, total),
        "avg_state_prob_margin": sum(_safe_float(row.get("hmm_state_prob_margin"), 0.0) for row in rows) / total if total else 0.0,
        "avg_regime_prob_margin": sum(_safe_float(row.get("hmm_regime_prob_margin"), 0.0) for row in rows) / total if total else 0.0,
        "avg_fit_seconds": sum(fit_seconds) / total if total else 0.0,
        "max_fit_seconds": max(fit_seconds) if fit_seconds else 0.0,
        "total_fit_seconds": sum(fit_seconds),
        "mapping_matched_pct": _pct(mapping_methods.count("matched"), total),
        "mapping_score_order_pct": _pct(mapping_methods.count("score_order"), total),
        "mapping_none_pct": _pct(mapping_methods.count("none"), total),
        "mapping_other_pct": _pct(sum(1 for method in mapping_methods if method not in known_mapping), total),
        "fwd_rows": len(fwd_rows),
        "fwd_bull_n": fwd_bull_n,
        "fwd_bull_mean": fwd_bull_mean,
        "fwd_bull_hit": fwd_bull_hit,
        "fwd_neutral_n": fwd_neutral_n,
        "fwd_neutral_mean": fwd_neutral_mean,
        "fwd_neutral_hit": fwd_neutral_hit,
        "fwd_bear_n": fwd_bear_n,
        "fwd_bear_mean": fwd_bear_mean,
        "fwd_bear_hit": fwd_bear_hit,
        "fwd_regime_spread": fwd_bull_mean - fwd_bear_mean if fwd_bull_n >= 1 and fwd_bear_n >= 1 else 0.0,
        "fwd_allow_mean": fwd_allow_mean,
        "fwd_block_mean": fwd_block_mean,
        "fwd_policy_spread": fwd_allow_mean - fwd_block_mean if fwd_allow_n >= 1 and fwd_block_n >= 1 else 0.0,
    }


def _warnings(summary: dict[str, Any]) -> list[str]:
    out: list[str] = []
    rows = int(summary.get("rows", 0) or 0)
    if rows <= 0:
        return ["no walk-forward rows produced"]

    switch_ratio = _safe_float(summary.get("regime_switch_count"), 0.0) / max(1, rows)
    if switch_ratio > 0.20:
        out.append("regime may be unstable: switch count is high")

    max_regime = max(
        _safe_float(summary.get("bull_pct"), 0.0),
        _safe_float(summary.get("neutral_pct"), 0.0),
        _safe_float(summary.get("bear_pct"), 0.0),
    )
    if max_regime > 0.80:
        out.append("model may be collapsed: one regime dominates more than 80%")

    if _safe_float(summary.get("hmm_ok_rate"), 0.0) < 0.90:
        out.append("backend is not production-reliable yet: hmm_ok_rate is below 90%")

    if str(summary.get("backend")) == "pomegranate_gmm" and _safe_float(summary.get("fallback_pct"), 0.0) > 0.50:
        out.append("GMM appears unstable: more than 50% of rows fell back to normal")
    return out


def _plot_rows(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mpl_config = path.parent / ".mplconfig"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skipping PNG: {type(exc).__name__}: {exc}", flush=True)
        return False

    if not rows:
        return False

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["timestamp", "close"], inplace=True)
    if df.empty:
        return False

    colors = {"bull": "#18864b", "neutral": "#7a7a7a", "bear": "#c83e3e", "unknown": "#111111"}
    fig, (ax_price, ax_regime) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax_price.plot(df["timestamp"], df["close"], color="#222222", linewidth=1.1)
    ax_price.set_title(str(df["backend"].iloc[0]))
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.25)

    for regime, group in df.groupby("hmm_regime"):
        ax_price.scatter(group["timestamp"], group["close"], s=8, color=colors.get(regime, "#111111"), label=str(regime), alpha=0.75)

    regime_values = {"bear": -1, "neutral": 0, "bull": 1, "unknown": 0}
    ax_regime.scatter(
        df["timestamp"],
        [regime_values.get(str(x), 0) for x in df["hmm_regime"]],
        s=10,
        c=[colors.get(str(x), "#111111") for x in df["hmm_regime"]],
        alpha=0.8,
    )
    ax_regime.set_yticks([-1, 0, 1], ["bear", "neutral", "bull"])
    ax_regime.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"backend: {summary['backend']}")
    print(f"summary_scope: {summary.get('summary_scope', 'all')}")
    for field in SUMMARY_FIELDS[1:]:
        if field == "summary_scope":
            continue
        value = summary.get(field)
        if isinstance(value, float):
            print(f"{field}: {value:.4f}")
        else:
            print(f"{field}: {value}")
    warnings = _warnings(summary)
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("warnings: none")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward HMM regime inspection tool")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=24, help="Forward scoring horizon in candles")
    parser.add_argument("--output-dir", default="log/hmm_walkforward")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    symbol = _binance_symbol(args.symbol)
    timeframe = str(args.timeframe)
    limit = max(1, int(args.limit))
    window = max(20, int(args.window))
    step = max(1, int(args.step))
    horizon = max(1, int(args.horizon))
    output_dir = Path(args.output_dir)

    print(f"Fetching {symbol} {timeframe} limit={limit}...")
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df.empty:
        print("No OHLCV data fetched.")
        return 1
    if len(df) <= window:
        print(f"Not enough rows for walk-forward: rows={len(df)} window={window}")
        return 1

    summaries: list[dict[str, Any]] = []
    for backend in BACKENDS:
        print("\n" + "-" * 72)
        print(f"Running backend: {backend}")
        rows = _walk_backend(df, symbol, timeframe, backend, window, step, horizon)
        csv_path = output_dir / f"hmm_walkforward_{symbol}_{timeframe}_{backend}.csv"
        png_path = output_dir / f"hmm_walkforward_{symbol}_{timeframe}_{backend}.png"
        _write_csv(csv_path, rows, ROW_FIELDS)
        png_ok = _plot_rows(png_path, rows)
        print(f"CSV: {csv_path}")
        print(f"PNG: {png_path if png_ok else 'not created'}")
        summary_blocks = [summarize(rows, backend, "all")]
        if backend == "pomegranate_gmm":
            pure_rows = [row for row in rows if str(row.get("hmm_effective_emission_backend")) == "pomegranate_gmm"]
            fallback_rows = [row for row in rows if str(row.get("hmm_effective_emission_backend")) != "pomegranate_gmm"]
            summary_blocks.append(summarize(pure_rows, backend, "pure_gmm"))
            summary_blocks.append(summarize(fallback_rows, backend, "fallback"))
        for summary in summary_blocks:
            summary["csv_path"] = str(csv_path)
            summary["png_path"] = str(png_path) if png_ok else ""
            summaries.append(summary)
            _print_summary(summary)

    summary_path = output_dir / f"hmm_walkforward_summary_{symbol}_{timeframe}.csv"
    _write_csv(summary_path, summaries, SUMMARY_FIELDS)
    print("\n" + "=" * 72)
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
