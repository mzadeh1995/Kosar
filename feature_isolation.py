# ==============================================================================
# 📊 feature_isolation.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Standalone diagnostics for OFI/VPIN feature isolation analysis.
# ==============================================================================

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["ofi_l1_norm", "ofi_l1_z", "ofi_depth_imbalance", "vpin", "vpin_z"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        except Exception:
            pass
    return out


def compute_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index if isinstance(df, pd.DataFrame) else pd.Index([]))
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df.columns:
        return out

    close = pd.to_numeric(df["Close"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    horizon_vals = []
    for h in horizons if isinstance(horizons, list) else []:
        try:
            hi = int(h)
            if hi > 0:
                horizon_vals.append(hi)
        except Exception:
            continue
    horizon_vals = sorted(set(horizon_vals))

    for h in horizon_vals:
        fwd = (close.shift(-h) / close) - 1.0
        abs_fwd = fwd.abs()

        roll_window = max(50, 10 * h)
        rolling_q = abs_fwd.rolling(window=roll_window, min_periods=max(20, h * 5)).quantile(0.90).shift(1)
        global_q = _safe_float(abs_fwd.quantile(0.90), 0.0)
        threshold = rolling_q.fillna(global_q)

        jump = pd.Series(np.nan, index=abs_fwd.index, dtype=float)
        valid = abs_fwd.notna() & threshold.notna()
        jump.loc[valid] = (abs_fwd.loc[valid] > threshold.loc[valid]).astype(float)

        out[f"fwd_return_{h}"] = fwd
        out[f"fwd_abs_return_{h}"] = abs_fwd
        out[f"jump_label_{h}"] = jump

    return out


def information_coefficient(features_df, labels_df) -> dict:
    results: dict[str, dict[str, dict[str, Any]]] = {}

    if features_df is None or labels_df is None:
        return results

    feats = pd.DataFrame(features_df).copy()
    labels = pd.DataFrame(labels_df).copy()

    try:
        from scipy import stats as scipy_stats  # optional
    except Exception:
        scipy_stats = None

    label_cols = [c for c in labels.columns if str(c).startswith(("fwd_return_", "fwd_abs_return_", "jump_label_"))]

    for feat in FEATURE_COLUMNS:
        if feat not in feats.columns:
            results[feat] = {"status": "skipped_missing_feature"}
            continue

        feat_results: dict[str, dict[str, Any]] = {}
        x = pd.to_numeric(feats[feat], errors="coerce")

        for tgt in label_cols:
            y = pd.to_numeric(labels[tgt], errors="coerce")
            aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

            if aligned.shape[0] < 10:
                feat_results[tgt] = {
                    "n": int(aligned.shape[0]),
                    "pearson_ic": 0.0,
                    "spearman_ic": 0.0,
                    "pearson_pvalue": None,
                    "spearman_pvalue": None,
                    "status": "insufficient_data",
                }
                continue

            pearson_ic = _safe_float(aligned["x"].corr(aligned["y"], method="pearson"), 0.0)
            spearman_ic = _safe_float(aligned["x"].corr(aligned["y"], method="spearman"), 0.0)

            pearson_p = None
            spearman_p = None
            if scipy_stats is not None and aligned.shape[0] >= 3:
                try:
                    pearson_p = _safe_float(scipy_stats.pearsonr(aligned["x"], aligned["y"]).pvalue, np.nan)
                except Exception:
                    pearson_p = None
                try:
                    spearman_p = _safe_float(scipy_stats.spearmanr(aligned["x"], aligned["y"]).pvalue, np.nan)
                except Exception:
                    spearman_p = None

                if pearson_p is not None and not np.isfinite(pearson_p):
                    pearson_p = None
                if spearman_p is not None and not np.isfinite(spearman_p):
                    spearman_p = None

            feat_results[tgt] = {
                "n": int(aligned.shape[0]),
                "pearson_ic": pearson_ic,
                "spearman_ic": spearman_ic,
                "pearson_pvalue": pearson_p,
                "spearman_pvalue": spearman_p,
                "status": "ok",
            }

        results[feat] = feat_results

    return results


def granger_tests(features_df, target_series, max_lag=6) -> dict:
    try:
        from statsmodels.tsa.stattools import grangercausalitytests  # optional
    except Exception:
        return {
            "ok": False,
            "reason": "statsmodels_unavailable",
            "results": {},
        }

    features = pd.DataFrame(features_df).copy() if features_df is not None else pd.DataFrame()
    target = pd.to_numeric(pd.Series(target_series), errors="coerce") if target_series is not None else pd.Series(dtype=float)

    if features.empty or target.empty:
        return {
            "ok": False,
            "reason": "insufficient_data",
            "results": {},
        }

    lag = max(1, int(max_lag))
    out: dict[str, Any] = {}

    for feat in FEATURE_COLUMNS:
        if feat not in features.columns:
            out[feat] = {"status": "skipped_missing_feature"}
            continue

        x = pd.to_numeric(features[feat], errors="coerce")
        aligned = pd.concat([target.rename("target"), x.rename("feature")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

        if aligned.shape[0] < (lag + 10):
            out[feat] = {
                "status": "insufficient_data",
                "n": int(aligned.shape[0]),
            }
            continue

        if float(aligned["feature"].std(ddof=0)) <= 1e-12:
            out[feat] = {
                "status": "constant_feature",
                "n": int(aligned.shape[0]),
            }
            continue

        try:
            test_res = grangercausalitytests(aligned[["target", "feature"]], maxlag=lag, verbose=False)
            pvals = {}
            for lag_i in range(1, lag + 1):
                if lag_i not in test_res:
                    continue
                try:
                    pval = float(test_res[lag_i][0]["ssr_ftest"][1])
                    if np.isfinite(pval):
                        pvals[str(lag_i)] = pval
                except Exception:
                    continue

            min_pvalue = min(pvals.values()) if pvals else None
            out[feat] = {
                "status": "ok",
                "n": int(aligned.shape[0]),
                "max_lag": lag,
                "pvalues": pvals,
                "min_pvalue": min_pvalue,
            }
        except Exception as exc:
            out[feat] = {
                "status": "error",
                "n": int(aligned.shape[0]),
                "error": f"{type(exc).__name__}:{str(exc)[:180]}",
            }

    return {
        "ok": True,
        "reason": None,
        "results": out,
    }


def _coverage_ratio(series: pd.Series) -> float:
    if series is None or series.shape[0] == 0:
        return 0.0
    valid = series.replace([np.inf, -np.inf], np.nan).notna().sum()
    return _safe_float(valid / max(1, series.shape[0]), 0.0)


def _flatten_ic_table(ic_table: dict) -> pd.DataFrame:
    rows = []
    for feat, by_target in (ic_table or {}).items():
        if not isinstance(by_target, dict):
            continue
        for target, metrics in by_target.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                {
                    "feature": feat,
                    "target": target,
                    "status": metrics.get("status"),
                    "n": metrics.get("n"),
                    "pearson_ic": metrics.get("pearson_ic"),
                    "spearman_ic": metrics.get("spearman_ic"),
                    "pearson_pvalue": metrics.get("pearson_pvalue"),
                    "spearman_pvalue": metrics.get("spearman_pvalue"),
                }
            )
    return pd.DataFrame(rows)


def _feature_verdict(feature: str, coverage: float, ic_table: dict, min_n: int = 50) -> str:
    if coverage <= 0.0:
        return "skipped"
    if coverage < 0.20:
        return "insufficient_data"

    feat_metrics = ic_table.get(feature)
    if not isinstance(feat_metrics, dict):
        return "skipped"

    best_abs_ic = 0.0
    best_n = 0

    for _, metrics in feat_metrics.items():
        if not isinstance(metrics, dict):
            continue
        n = int(metrics.get("n", 0) or 0)
        if n > best_n:
            best_n = n
        p = abs(_safe_float(metrics.get("pearson_ic"), 0.0))
        s = abs(_safe_float(metrics.get("spearman_ic"), 0.0))
        best_abs_ic = max(best_abs_ic, p, s)

    if best_n < min_n:
        return "insufficient_data"
    return "useful" if best_abs_ic >= 0.05 else "weak"


def _json_default(obj: Any):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def run_feature_isolation_report(symbol: str, ohlcv_df: pd.DataFrame, microstructure_df: pd.DataFrame, config: dict) -> dict:
    cfg = config or {}
    horizons = cfg.get("FEATURE_ISOLATION_HORIZON_BARS", [1, 3, 6, 12])
    output_dir = str(cfg.get("FEATURE_ISOLATION_OUTPUT_DIR", os.path.join(os.getcwd(), "log", "feature_isolation")))

    warnings: list[str] = []

    price_df = _ensure_datetime_index(ohlcv_df)
    micro_df = _ensure_datetime_index(microstructure_df)

    if price_df.empty:
        report = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_size": 0,
            "feature_coverage": {},
            "ic_table": {},
            "granger_results": {"ok": False, "reason": "no_ohlcv_data", "results": {}},
            "warnings": ["no_ohlcv_data"],
            "verdict": {f: "skipped" for f in FEATURE_COLUMNS},
        }
        return report

    labels = compute_forward_returns(price_df, list(horizons) if isinstance(horizons, list) else [1, 3, 6, 12])

    if micro_df.empty:
        warnings.append("no_microstructure_data")
        micro_aligned = pd.DataFrame(index=labels.index)
    else:
        micro_aligned = micro_df.reindex(labels.index)

    feature_coverage = {}
    for feat in FEATURE_COLUMNS:
        if feat in micro_aligned.columns:
            feature_coverage[feat] = _coverage_ratio(pd.to_numeric(micro_aligned[feat], errors="coerce"))
        else:
            feature_coverage[feat] = 0.0
            warnings.append(f"missing_feature:{feat}")

    ic_table = information_coefficient(micro_aligned, labels)

    target_col = None
    fwd_cols = [c for c in labels.columns if str(c).startswith("fwd_return_")]
    if "fwd_return_1" in labels.columns:
        target_col = "fwd_return_1"
    elif fwd_cols:
        target_col = fwd_cols[0]

    if target_col is None:
        granger_result = {"ok": False, "reason": "no_target_label", "results": {}}
        warnings.append("no_forward_return_target")
    else:
        granger_result = granger_tests(micro_aligned, labels[target_col], max_lag=6)

    verdict = {
        feat: _feature_verdict(feat, _safe_float(feature_coverage.get(feat, 0.0), 0.0), ic_table)
        for feat in FEATURE_COLUMNS
    }

    sample_size = int(len(labels))

    report = {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_size": sample_size,
        "feature_coverage": feature_coverage,
        "ic_table": ic_table,
        "granger_results": granger_result,
        "warnings": warnings,
        "verdict": verdict,
    }

    os.makedirs(output_dir, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_sym = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(symbol))

    json_path = os.path.join(output_dir, f"{safe_sym}__feature_isolation__{ts_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    try:
        ic_df = _flatten_ic_table(ic_table)
        if not ic_df.empty:
            csv_path = os.path.join(output_dir, f"{safe_sym}__feature_isolation__{ts_tag}.csv")
            ic_df.to_csv(csv_path, index=False)
    except Exception:
        pass

    report["output_json"] = json_path
    return report
