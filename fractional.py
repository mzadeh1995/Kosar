# ==============================================================================
# 🧮 fractional.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Fractional differentiation utilities for HMM feature engineering.
# Includes fixed-width fractional differencing (FFD) and fractional HMM features.
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "fd_log_close",
    "fd_return",
    "fd_volatility",
    "fd_log_volume",
    "fd_log_range",
]


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _safe_int(x: Any, default: int) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return int(default)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_ffd_weights(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    d_val = _safe_float(d, 0.0)
    thr = max(0.0, _safe_float(threshold, 1e-5))
    size_limit = max(1, _safe_int(max_size, 10000))

    weights = [1.0]
    for k in range(1, size_limit):
        prev = weights[-1]
        wk = -prev * ((d_val - k + 1.0) / float(k))
        if abs(wk) < thr:
            break
        weights.append(float(wk))
    return np.asarray(weights, dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5, max_size: int = 10000) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)

    numeric = _to_numeric_series(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)

    valid = numeric.dropna()
    if valid.empty:
        return result

    weights = get_ffd_weights(d=d, threshold=threshold, max_size=max_size)
    width = len(weights)
    if width <= 0 or len(valid) < width:
        return result

    vals = valid.to_numpy(dtype=float)
    out_vals = np.full(vals.shape[0], np.nan, dtype=float)
    weights_rev = weights[::-1]

    for i in range(width - 1, len(vals)):
        window = vals[i - width + 1 : i + 1]
        out_vals[i] = float(np.dot(weights_rev, window))

    result.loc[valid.index] = out_vals
    return result


def select_fracdiff_d_adf(
    series: pd.Series,
    d_grid: Iterable[float] | None = None,
    pvalue_threshold: float = 0.05,
    threshold: float = 1e-5,
    max_size: int = 10000,
    min_corr: float = 0.30,
    fallback_d: float = 0.45,
) -> Tuple[float, Dict[str, Any]]:
    base = _to_numeric_series(series)
    fallback = _safe_float(fallback_d, 0.45)
    pv_thr = max(1e-9, min(1.0, _safe_float(pvalue_threshold, 0.05)))
    min_abs_corr = max(0.0, min(1.0, _safe_float(min_corr, 0.30)))

    meta: Dict[str, Any] = {
        "adf_enabled": True,
        "adf_selected": False,
        "adf_pvalue": None,
        "corr_with_base": None,
        "selected_d": fallback,
        "reason": "fallback",
    }

    if base.dropna().shape[0] < 40:
        meta["reason"] = "insufficient_rows_for_adf"
        return fallback, meta

    try:
        from statsmodels.tsa.stattools import adfuller  # optional dependency
    except Exception as exc:
        meta["reason"] = f"statsmodels_unavailable:{type(exc).__name__}"
        return fallback, meta

    grid_raw = list(d_grid) if d_grid is not None else [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    grid: list[float] = []
    for d_val in grid_raw:
        try:
            d_num = float(d_val)
            if np.isfinite(d_num):
                grid.append(d_num)
        except Exception:
            continue
    if not grid:
        meta["reason"] = "empty_d_grid"
        return fallback, meta

    grid = sorted(set(grid))
    for d_val in grid:
        try:
            fd = frac_diff_ffd(base, d=d_val, threshold=threshold, max_size=max_size)
            aligned = pd.concat(
                [base.rename("base"), fd.rename("fd")],
                axis=1,
            ).replace([np.inf, -np.inf], np.nan).dropna()

            if len(aligned) < 40:
                continue

            corr = float(aligned["base"].corr(aligned["fd"]))
            if not np.isfinite(corr) or abs(corr) < min_abs_corr:
                continue

            # adfuller returns tuple where index 1 is p-value
            adf_pvalue = float(adfuller(aligned["fd"].to_numpy(dtype=float), autolag="AIC")[1])
            if not np.isfinite(adf_pvalue):
                continue

            if adf_pvalue <= pv_thr:
                meta.update(
                    {
                        "adf_selected": True,
                        "adf_pvalue": adf_pvalue,
                        "corr_with_base": corr,
                        "selected_d": float(d_val),
                        "reason": "selected",
                    }
                )
                return float(d_val), meta
        except Exception:
            continue

    meta["reason"] = "no_candidate_met_constraints"
    return fallback, meta


def build_fractional_hmm_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or {}
    required = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    if not set(required).issubset(df.columns):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    eps = max(1e-15, _safe_float(cfg.get("HMM_RET_EPS", 1e-12), 1e-12))
    vol_window = max(5, _safe_int(cfg.get("HMM_VOL_WINDOW", 24), 24))
    d_fixed = _safe_float(cfg.get("HMM_FRAC_DIFF_D", 0.45), 0.45)
    fd_threshold = max(0.0, _safe_float(cfg.get("HMM_FRAC_DIFF_THRESHOLD", 1e-5), 1e-5))
    fd_max_size = max(1, _safe_int(cfg.get("HMM_FRAC_DIFF_MAX_SIZE", 10000), 10000))
    use_adf = bool(cfg.get("HMM_FRAC_DIFF_USE_ADF", False))
    adf_pv = _safe_float(cfg.get("HMM_FRAC_DIFF_ADF_PVALUE", 0.05), 0.05)
    d_grid = cfg.get("HMM_FRAC_DIFF_D_GRID", [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    min_corr = _safe_float(cfg.get("HMM_FRAC_DIFF_MIN_CORR", 0.30), 0.30)

    close = _to_numeric_series(df["Close"])
    high = _to_numeric_series(df["High"])
    low = _to_numeric_series(df["Low"])
    volume = _to_numeric_series(df["Volume"])

    log_close = np.log(close.clip(lower=eps))
    log_volume = np.log(volume.clip(lower=eps))
    log_range = np.log(high.clip(lower=eps) / low.clip(lower=eps))

    d_used = d_fixed
    adf_meta: Dict[str, Any] = {
        "adf_enabled": False,
        "adf_selected": False,
        "adf_pvalue": None,
        "corr_with_base": None,
        "selected_d": d_fixed,
        "reason": "disabled",
    }
    if use_adf:
        d_used, adf_meta = select_fracdiff_d_adf(
            series=log_close,
            d_grid=d_grid,
            pvalue_threshold=adf_pv,
            threshold=fd_threshold,
            max_size=fd_max_size,
            min_corr=min_corr,
            fallback_d=d_fixed,
        )

    fd_log_close = frac_diff_ffd(log_close, d=d_used, threshold=fd_threshold, max_size=fd_max_size)
    fd_log_volume = frac_diff_ffd(log_volume, d=d_used, threshold=fd_threshold, max_size=fd_max_size)
    fd_log_range = frac_diff_ffd(log_range, d=d_used, threshold=fd_threshold, max_size=fd_max_size)

    fd_return = fd_log_close.diff()
    fd_volatility = fd_return.rolling(vol_window, min_periods=vol_window).std()

    features = pd.DataFrame(
        {
            "fd_log_close": fd_log_close,
            "fd_return": fd_return,
            "fd_volatility": fd_volatility,
            "fd_log_volume": fd_log_volume,
            "fd_log_range": fd_log_range,
        },
        index=df.index,
    )
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)

    weights = get_ffd_weights(d=d_used, threshold=fd_threshold, max_size=fd_max_size)
    features.attrs["fracdiff_meta"] = {
        "feature_mode": "fractional_diff",
        "method": "ffd",
        "d_fixed": d_fixed,
        "d_used": float(d_used),
        "threshold": float(fd_threshold),
        "max_size": int(fd_max_size),
        "window_size": int(len(weights)),
        "adf_enabled": bool(use_adf),
        "adf_selected": bool(adf_meta.get("adf_selected", False)),
        "adf_pvalue": adf_meta.get("adf_pvalue"),
        "corr_with_base": adf_meta.get("corr_with_base"),
        "adf_reason": adf_meta.get("reason"),
    }
    return features
