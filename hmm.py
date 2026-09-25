# ==============================================================================
# hmm.py
# ------------------------------------------------------------------------------
# Lightweight market-regime gate.
# v49.1: Added "Calibration" and Patch
#
# HMM is not a trade decision engine. It only classifies the current market
# regime and returns a policy gate: allow, caution, or block.
# ==============================================================================

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
REGIMES = {"bull", "neutral", "bear"}
POLICIES = {"allow", "caution", "block"}

DEFAULT_BACKEND = "pomegranate"
DEFAULT_EMISSION_BACKEND = "pomegranate_normal"
DEFAULT_ANCHOR_METHOD = "return_trend_volatility"

_ANCHOR_STATE: dict[str, dict[str, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_probs(values: dict[str, float] | None = None) -> dict[str, float]:
    raw = {"bull": 0.0, "neutral": 1.0, "bear": 0.0}
    if isinstance(values, dict):
        for key in raw:
            raw[key] = max(0.0, _safe_float(values.get(key), raw[key]))

    total = sum(raw.values())
    if (not np.isfinite(total)) or total <= 1e-12:
        return {"bull": 0.0, "neutral": 1.0, "bear": 0.0}
    return {key: float(value / total) for key, value in raw.items()}


def _prob_margin(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    top = np.sort(arr)[-2:]
    return max(0.0, float(top[-1] - top[-2]))


def _prob_entropy(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def _base_result(
    *,
    enabled: bool,
    ok: bool,
    reason: str,
    backend: str,
    anchor_method: str,
    feature_meta: dict[str, Any] | None = None,
    regime: str = "neutral",
    policy: str = "caution",
    state: int | None = None,
    confidence: float = 0.0,
    probs: dict[str, float] | None = None,
    persistence: float = 0.0,
    switch_margin: float = 0.0,
    prev_regime: str | None = None,
    regime_changed: bool = False,
    regime_age_bars: int = 0,
    filtered_confidence: float | None = None,
    debug_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_probs = _normalize_probs(probs)
    clean_regime = regime if regime in {"bull", "neutral", "bear", "unknown"} else "neutral"
    clean_policy = policy if policy in POLICIES else "caution"

    meta = {
        "hmm_feature_mode": "unknown",
        "hmm_feature_rows": 0,
        "hmm_feature_columns": [],
        "hmm_trend_col": None,
        "hmm_vol_col": None,
        "hmm_range_col": None,
        "hmm_state_prob_max": 0.0,
        "hmm_state_prob_min": 0.0,
        "hmm_state_prob_margin": 0.0,
        "hmm_regime_prob_margin": 0.0,
        "hmm_state_prob_entropy": 0.0,
        "hmm_effective_emission_backend": None,
        "hmm_mapping_method": "none",
        "hmm_fracdiff_method": None,
        "hmm_fracdiff_d_used": None,
        "hmm_frac_diff_d": None,
        "hmm_fracdiff_window": None,
        "hmm_fracdiff_threshold": None,
        "hmm_frac_diff_threshold": None,
        "hmm_fracdiff_use_adf": False,
        "hmm_fracdiff_adf_selected": False,
        "hmm_fracdiff_adf_pvalue": None,
        "hmm_fracdiff_corr_with_base": None,
    }
    if isinstance(feature_meta, dict):
        meta.update(feature_meta)
    if isinstance(debug_meta, dict):
        meta.update(debug_meta)

    feature_columns = meta.get("hmm_feature_columns", [])
    if not isinstance(feature_columns, list):
        feature_columns = list(feature_columns) if isinstance(feature_columns, tuple) else []

    conf = max(0.0, min(1.0, _safe_float(confidence, 0.0)))
    filtered_conf = conf if filtered_confidence is None else max(0.0, min(1.0, _safe_float(filtered_confidence, conf)))

    out = {
        "hmm_enabled": bool(enabled),
        "hmm_ok": bool(ok),
        "hmm_state": None if state is None else _safe_int(state, 0),
        "hmm_regime": clean_regime,
        "hmm_confidence": float(conf),
        "hmm_bull_prob": clean_probs["bull"],
        "hmm_neutral_prob": clean_probs["neutral"],
        "hmm_bear_prob": clean_probs["bear"],
        "hmm_policy": clean_policy,
        "hmm_reason": str(reason),
        "hmm_backend": str(backend),
        "hmm_anchor_method": str(anchor_method),
        "hmm_persistence": max(0.0, min(1.0, _safe_float(persistence, 0.0))),
        "hmm_switch_margin": _safe_float(switch_margin, 0.0),
        "hmm_prev_regime": prev_regime if prev_regime in REGIMES else None,
        "hmm_regime_changed": bool(regime_changed),
        "hmm_regime_age_bars": max(0, _safe_int(regime_age_bars, 0)),
        "hmm_filtered_confidence": float(filtered_conf),
        "hmm_feature_mode": str(meta.get("hmm_feature_mode", "unknown")),
        "hmm_feature_rows": max(0, _safe_int(meta.get("hmm_feature_rows", 0), 0)),
        "hmm_feature_columns": [str(col) for col in feature_columns],
        "hmm_trend_col": meta.get("hmm_trend_col"),
        "hmm_vol_col": meta.get("hmm_vol_col"),
        "hmm_range_col": meta.get("hmm_range_col"),
        "hmm_state_prob_max": _safe_float(meta.get("hmm_state_prob_max"), 0.0),
        "hmm_state_prob_min": _safe_float(meta.get("hmm_state_prob_min"), 0.0),
        "hmm_state_prob_margin": _safe_float(meta.get("hmm_state_prob_margin"), 0.0),
        "hmm_regime_prob_margin": _safe_float(meta.get("hmm_regime_prob_margin"), 0.0),
        "hmm_state_prob_entropy": _safe_float(meta.get("hmm_state_prob_entropy"), 0.0),
        "hmm_effective_emission_backend": meta.get("hmm_effective_emission_backend"),
        "hmm_mapping_method": meta.get("hmm_mapping_method") if meta.get("hmm_mapping_method") in {"matched", "score_order", "none"} else "none",
    }

    # Backward-compatible optional fields consumed by market.py/senate.py via .get().
    for key in [
        "hmm_fracdiff_method",
        "hmm_fracdiff_d_used",
        "hmm_frac_diff_d",
        "hmm_fracdiff_window",
        "hmm_fracdiff_threshold",
        "hmm_frac_diff_threshold",
        "hmm_fracdiff_use_adf",
        "hmm_fracdiff_adf_selected",
        "hmm_fracdiff_adf_pvalue",
        "hmm_fracdiff_corr_with_base",
    ]:
        out[key] = meta.get(key)

    return out


def _neutral_result(
    enabled: bool,
    reason: str,
    config: dict | None,
    feature_meta: dict[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    resolved_backend = backend or str(cfg.get("HMM_BACKEND", DEFAULT_BACKEND))
    anchor_method = str(cfg.get("HMM_ANCHOR_METHOD", DEFAULT_ANCHOR_METHOD))
    fail_policy = str(cfg.get("HMM_FAIL_POLICY", "caution")).lower()
    if fail_policy not in POLICIES:
        fail_policy = "caution"
    return _base_result(
        enabled=enabled,
        ok=False,
        reason=reason,
        backend=resolved_backend,
        anchor_method=anchor_method,
        feature_meta=feature_meta,
        regime="neutral",
        policy=fail_policy,
        confidence=0.0,
        probs={"bull": 0.0, "neutral": 1.0, "bear": 0.0},
    )


def trim_ohlcv_for_hmm(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=REQUIRED_OHLCV)
    if not set(REQUIRED_OHLCV).issubset(df.columns):
        return pd.DataFrame(columns=REQUIRED_OHLCV)

    out = df[REQUIRED_OHLCV].copy()
    for col in REQUIRED_OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out["Close"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Volume"] >= 0)]
    return out


def _legacy_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    eps = max(1e-15, _safe_float(config.get("HMM_RET_EPS", 1e-12), 1e-12))
    vol_window = max(5, _safe_int(config.get("HMM_VOL_WINDOW", 24), 24))
    trend_window = max(3, _safe_int(config.get("HMM_TREND_WINDOW", 8), 8))

    close = pd.to_numeric(df["Close"], errors="coerce").clip(lower=eps)
    high = pd.to_numeric(df["High"], errors="coerce").clip(lower=eps)
    low = pd.to_numeric(df["Low"], errors="coerce").clip(lower=eps)
    volume = pd.to_numeric(df["Volume"], errors="coerce").clip(lower=eps)

    log_close = np.log(close)
    fd_return = log_close.diff()
    out = pd.DataFrame(
        {
            "fd_return": fd_return,
            "fd_volatility": fd_return.rolling(vol_window, min_periods=vol_window).std(),
            "fd_log_range": np.log(high / low),
            "fd_log_volume": np.log(volume).diff(),
            "fd_trend": fd_return.rolling(trend_window, min_periods=trend_window).mean(),
        },
        index=df.index,
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def build_hmm_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = config or {}
    mode = str(cfg.get("HMM_FEATURE_MODE", "fractional_diff")).strip().lower()

    if mode == "fractional_diff":
        try:
            from fractional import build_fractional_hmm_features

            features = build_fractional_hmm_features(df, cfg)
            if isinstance(features, pd.DataFrame) and not features.empty:
                out = features.copy()
                if "fd_return" in out.columns and "fd_trend" not in out.columns:
                    out["fd_trend"] = pd.to_numeric(out["fd_return"], errors="coerce").rolling(8, min_periods=3).mean()
                out = out.replace([np.inf, -np.inf], np.nan).dropna()
                out.attrs["feature_mode"] = "fractional_diff"
                out.attrs["fracdiff_meta"] = dict(getattr(features, "attrs", {}).get("fracdiff_meta", {}))
                return out
        except Exception:
            pass

    out = _legacy_features(df, cfg)
    out.attrs["feature_mode"] = "legacy"
    out.attrs["fracdiff_meta"] = {}
    return out


def _feature_meta(features: pd.DataFrame, config: dict) -> dict[str, Any]:
    attrs = dict(getattr(features, "attrs", {}) or {})
    frac_meta = dict(attrs.get("fracdiff_meta", {}) or {})
    mode = str(attrs.get("feature_mode") or frac_meta.get("feature_mode") or config.get("HMM_FEATURE_MODE", "legacy"))
    cols = [str(col) for col in features.columns]

    trend_col = str(config.get("HMM_REGIME_TREND_COL", "fd_trend"))
    if trend_col not in features.columns:
        trend_col = "fd_trend" if "fd_trend" in features.columns else "fd_return" if "fd_return" in features.columns else None
    vol_col = str(config.get("HMM_REGIME_VOL_COL", "fd_volatility"))
    if vol_col not in features.columns:
        vol_col = "fd_volatility" if "fd_volatility" in features.columns else None
    range_col = str(config.get("HMM_REGIME_RANGE_COL", "fd_log_range"))
    if range_col not in features.columns:
        range_col = "fd_log_range" if "fd_log_range" in features.columns else None

    return {
        "hmm_feature_mode": mode,
        "hmm_feature_rows": int(len(features)),
        "hmm_feature_columns": cols,
        "hmm_trend_col": trend_col,
        "hmm_vol_col": vol_col,
        "hmm_range_col": range_col,
        "hmm_fracdiff_method": frac_meta.get("method"),
        "hmm_fracdiff_d_used": frac_meta.get("d_used"),
        "hmm_frac_diff_d": frac_meta.get("d_used"),
        "hmm_fracdiff_window": frac_meta.get("window_size"),
        "hmm_fracdiff_threshold": frac_meta.get("threshold"),
        "hmm_frac_diff_threshold": frac_meta.get("threshold"),
        "hmm_fracdiff_use_adf": bool(frac_meta.get("adf_enabled", False)),
        "hmm_fracdiff_adf_selected": bool(frac_meta.get("adf_selected", False)),
        "hmm_fracdiff_adf_pvalue": frac_meta.get("adf_pvalue"),
        "hmm_fracdiff_corr_with_base": frac_meta.get("corr_with_base"),
    }


def _prepare_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str | None]:
    cleaned = trim_ohlcv_for_hmm(df)
    if cleaned.empty:
        return pd.DataFrame(), pd.DataFrame(), {"hmm_feature_mode": str(config.get("HMM_FEATURE_MODE", "unknown"))}, "insufficient_data"

    max_rows = max(50, _safe_int(config.get("HMM_MAX_FEATURE_ROWS", 2000), 2000))
    raw_features = build_hmm_features(cleaned, config)
    if raw_features is None or raw_features.empty:
        return pd.DataFrame(), pd.DataFrame(), _feature_meta(pd.DataFrame(), config), "insufficient_data"

    preferred = ["fd_return", "fd_volatility", "fd_log_range", "fd_log_volume", "fd_trend"]
    columns = [col for col in preferred if col in raw_features.columns]
    if len(columns) < 2:
        columns = list(raw_features.columns[: min(6, len(raw_features.columns))])
    raw_features = raw_features[columns].replace([np.inf, -np.inf], np.nan).dropna().tail(max_rows)

    min_rows = max(20, _safe_int(config.get("HMM_MIN_FEATURE_ROWS", 300), 300))
    meta = _feature_meta(raw_features, config)
    if len(raw_features) < min_rows:
        return raw_features, pd.DataFrame(), meta, "insufficient_data"

    clipped = raw_features.copy()
    for col in clipped.columns:
        series = pd.to_numeric(clipped[col], errors="coerce")
        lo = series.quantile(0.01)
        hi = series.quantile(0.99)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            clipped[col] = series.clip(lo, hi)
        else:
            clipped[col] = series

    scaled = pd.DataFrame(index=clipped.index)
    for col in clipped.columns:
        series = pd.to_numeric(clipped[col], errors="coerce")
        median = _safe_float(series.median(), 0.0)
        q75 = _safe_float(series.quantile(0.75), median)
        q25 = _safe_float(series.quantile(0.25), median)
        scale = q75 - q25
        if (not np.isfinite(scale)) or scale <= 1e-12:
            std = _safe_float(series.std(ddof=0), 1.0)
            scale = std if std > 1e-12 else 1.0
        scaled[col] = ((series - median) / scale).clip(-8.0, 8.0)

    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna()
    raw_features = clipped.loc[scaled.index]
    meta = _feature_meta(raw_features, config)
    if len(scaled) < min_rows:
        return raw_features, pd.DataFrame(), meta, "insufficient_data"
    return raw_features, scaled, meta, None


def _import_pomegranate(include_gmm: bool = False) -> tuple[Any, Any, Any | None, str | None]:
    try:
        hmm_mod = importlib.import_module("pomegranate.hmm")
        dist_mod = importlib.import_module("pomegranate.distributions")
        gmm_cls = None
        if include_gmm:
            gmm_mod = importlib.import_module("pomegranate.gmm")
            gmm_cls = getattr(gmm_mod, "GeneralMixtureModel")
        return getattr(hmm_mod, "DenseHMM"), getattr(dist_mod, "Normal"), gmm_cls, None
    except Exception as exc:
        return None, None, None, f"pomegranate_unavailable:{type(exc).__name__}:{str(exc)[:120]}"


def _make_normal(Normal: Any, means: np.ndarray, variances: np.ndarray) -> Any:
    means = np.asarray(means, dtype=np.float32)
    variances = np.asarray(variances, dtype=np.float32)
    attempts = [
        lambda: Normal(means=means, covs=variances, covariance_type="diag"),
        lambda: Normal(means, variances, covariance_type="diag"),
        lambda: Normal(means=means, covs=variances),
        lambda: Normal(means, variances),
    ]
    last_error: Exception | None = None
    for build in attempts:
        try:
            return build()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"normal_distribution_init_failed:{type(last_error).__name__ if last_error else 'unknown'}")


def _min_cov(config: dict) -> float:
    return max(1e-8, _safe_float(config.get("HMM_GMM_MIN_COV", 1e-3), 1e-3))


def _iter_normal_distributions(distribution: Any, seen: set[int] | None = None):
    if distribution is None:
        return
    if seen is None:
        seen = set()
    ident = id(distribution)
    if ident in seen:
        return
    seen.add(ident)

    if hasattr(distribution, "means") and hasattr(distribution, "covs"):
        yield distribution

    for attr in ("distributions", "components"):
        children = getattr(distribution, attr, None)
        if children is None:
            continue
        try:
            iterator = list(children)
        except Exception:
            continue
        for child in iterator:
            yield from _iter_normal_distributions(child, seen)


def _clamp_covariance_param(covs: Any, min_cov: float) -> None:
    if covs is None:
        return
    if hasattr(covs, "data") and hasattr(covs.data, "clamp_"):
        data = covs.data
        try:
            data.nan_to_num_(nan=float(min_cov), posinf=float(min_cov), neginf=float(min_cov))
        except Exception:
            pass
        data.clamp_(min=float(min_cov))
        return

    arr = np.asarray(covs, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=float(min_cov), posinf=float(min_cov), neginf=float(min_cov))
    arr = np.maximum(arr, float(min_cov)).astype(np.float32)
    try:
        covs[...] = arr
    except Exception:
        if hasattr(covs, "copy_"):
            covs.copy_(arr)


def _clamp_distribution_variances(distribution: Any, config: dict) -> str | None:
    min_cov = _min_cov(config)
    try:
        normals = list(_iter_normal_distributions(distribution))
        for normal in normals:
            try:
                if hasattr(normal, "min_cov"):
                    normal.min_cov = min_cov
            except Exception:
                pass

            _clamp_covariance_param(getattr(normal, "covs", None), min_cov)
            for attr in ("means", "covs"):
                value = getattr(normal, attr, None)
                if value is None:
                    continue
                arr = _to_numpy(value)
                if not np.isfinite(arr).all():
                    return f"non_finite_{attr}_after_variance_clamp"
            covs = getattr(normal, "covs", None)
            if covs is not None and np.any(_to_numpy(covs) < min_cov - 1e-12):
                return "covariance_below_min_after_variance_clamp"
        return None
    except Exception as exc:
        return f"variance_clamp_failed:{type(exc).__name__}:{str(exc)[:120]}"


def _make_normal_from_chunk(Normal: Any, chunk: np.ndarray, config: dict) -> Any:
    min_cov = _min_cov(config)
    means = np.mean(chunk, axis=0).astype(np.float32)
    variances = np.var(chunk, axis=0).astype(np.float32)
    variances = np.maximum(variances, min_cov).astype(np.float32)
    return _make_normal(Normal, means, variances)


def _gmm_split_index(feature_columns: list[str] | None, config: dict, n_features: int) -> int:
    columns = [str(col) for col in (feature_columns or [])]
    candidates = [
        str(config.get("HMM_GMM_SPLIT_COL", "fd_volatility")),
        str(config.get("_HMM_FEATURE_VOL_COL", "")),
        str(config.get("HMM_REGIME_VOL_COL", "fd_volatility")),
        "fd_volatility",
    ]
    for candidate in candidates:
        if candidate in columns:
            return columns.index(candidate)
    if n_features > 1:
        return 1
    return 0


def _make_gmm_distribution(
    GeneralMixtureModel: Any,
    Normal: Any,
    chunk: np.ndarray,
    config: dict,
    feature_columns: list[str] | None = None,
) -> Any:
    min_rows = max(2, _safe_int(config.get("HMM_GMM_MIN_COMPONENT_ROWS", 30), 30))
    n_components = max(1, _safe_int(config.get("HMM_GMM_COMPONENTS", 2), 2))
    if chunk.shape[0] < min_rows or n_components <= 1:
        return _make_normal_from_chunk(Normal, chunk, config)

    n_components = min(n_components, max(1, chunk.shape[0] // min_rows))
    if n_components <= 1:
        return _make_normal_from_chunk(Normal, chunk, config)

    split_idx = _gmm_split_index(feature_columns, config, chunk.shape[1])
    order = np.argsort(chunk[:, split_idx]) if chunk.shape[1] else np.arange(chunk.shape[0])
    groups = np.array_split(order, n_components)
    distributions = []
    for group in groups:
        if len(group) < 2:
            return _make_normal_from_chunk(Normal, chunk, config)
        distributions.append(_make_normal_from_chunk(Normal, chunk[group], config))

    priors = np.full(len(distributions), 1.0 / len(distributions), dtype=np.float32)
    max_iter = max(1, _safe_int(config.get("HMM_GMM_MAX_ITER", 30), 30))
    tol = max(1e-8, _safe_float(config.get("HMM_GMM_TOL", 1e-3), 1e-3))
    gmm = GeneralMixtureModel(
        distributions,
        priors=priors,
        max_iter=max_iter,
        tol=tol,
        verbose=False,
    )
    gmm.fit(np.asarray(chunk, dtype=np.float32))
    clamp_error = _clamp_distribution_variances(gmm, config)
    if clamp_error:
        raise RuntimeError(clamp_error)
    return gmm


def _initial_state_groups(x: np.ndarray, trend_values: np.ndarray, n_states: int) -> list[np.ndarray]:
    order = np.argsort(trend_values)
    groups = np.array_split(order, n_states)
    fallback = np.arange(x.shape[0])
    return [group if len(group) else fallback for group in groups]


def _build_pomegranate_model(
    DenseHMM: Any,
    Normal: Any,
    GeneralMixtureModel: Any | None,
    x: np.ndarray,
    config: dict,
    emission_backend: str,
    feature_columns: list[str] | None = None,
) -> Any:
    n_states = max(2, _safe_int(config.get("HMM_N_STATES", 3), 3))
    n_states = min(n_states, max(2, x.shape[0] // 20))
    trend_values = x[:, 0] if x.shape[1] else np.zeros(x.shape[0])
    groups = _initial_state_groups(x, trend_values, n_states)

    distributions = []
    for group in groups:
        chunk = x[group]
        if emission_backend == "pomegranate_gmm":
            if GeneralMixtureModel is None:
                raise RuntimeError("general_mixture_model_unavailable")
            distributions.append(_make_gmm_distribution(GeneralMixtureModel, Normal, chunk, config, feature_columns))
        else:
            distributions.append(_make_normal_from_chunk(Normal, chunk, config))

    sticky = max(0.34, min(0.98, _safe_float(config.get("HMM_STICKY_SELF_TRANSITION", 0.88), 0.88)))
    off_diag = (1.0 - sticky) / max(1, n_states - 1)
    edges = np.full((n_states, n_states), off_diag, dtype=np.float32)
    np.fill_diagonal(edges, sticky)
    starts = np.full(n_states, 1.0 / n_states, dtype=np.float32)
    ends = np.full(n_states, 1.0 / n_states, dtype=np.float32)
    max_iter = max(1, _safe_int(config.get("HMM_MAX_ITER", 50), 50))
    tol = max(1e-8, _safe_float(config.get("HMM_TOL", 1e-3), 1e-3))

    attempts = [
        lambda: DenseHMM(distributions=distributions, edges=edges, starts=starts, ends=ends, max_iter=max_iter, tol=tol, verbose=False),
        lambda: DenseHMM(distributions, edges=edges, starts=starts, ends=ends, max_iter=max_iter, tol=tol),
        lambda: DenseHMM(distributions, edges, starts, ends, max_iter=max_iter, tol=tol),
        lambda: DenseHMM(distributions=distributions, edges=edges, starts=starts, max_iter=max_iter, tol=tol),
    ]
    last_error: Exception | None = None
    for build in attempts:
        try:
            return build()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"dense_hmm_init_failed:{type(last_error).__name__ if last_error else 'unknown'}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _fit_predict_pomegranate(
    scaled_features: pd.DataFrame,
    config: dict,
    emission_backend: str,
) -> tuple[np.ndarray | None, str | None]:
    include_gmm = emission_backend == "pomegranate_gmm"
    DenseHMM, Normal, GeneralMixtureModel, import_error = _import_pomegranate(include_gmm=include_gmm)
    if import_error:
        return None, import_error

    x2d = scaled_features.to_numpy(dtype=np.float32)
    if x2d.ndim != 2 or x2d.shape[0] < 2 or x2d.shape[1] < 1:
        return None, "insufficient_data"
    x3d = x2d.reshape(1, x2d.shape[0], x2d.shape[1]).astype(np.float32, copy=False)

    fit_inputs = [x3d, [x2d]]
    predict_inputs = [x3d, [x2d], x2d]
    feature_columns = [str(col) for col in scaled_features.columns]
    last_error: Exception | None = None

    for fit_input in fit_inputs:
        try:
            model = _build_pomegranate_model(
                DenseHMM,
                Normal,
                GeneralMixtureModel,
                x2d,
                config,
                emission_backend,
                feature_columns,
            )
            model.fit(fit_input)
            clamp_error = _clamp_distribution_variances(model, config)
            if clamp_error:
                raise RuntimeError(clamp_error)
            for pred_input in predict_inputs:
                try:
                    probs = _to_numpy(model.predict_proba(pred_input))
                    if probs.ndim == 3:
                        probs = probs[0]
                    if probs.ndim != 2 or probs.shape[0] != x2d.shape[0]:
                        last_error = RuntimeError("invalid_probability_shape")
                        continue
                    if not np.isfinite(probs).all():
                        last_error = RuntimeError("non_finite_probabilities")
                        continue
                    row_sums = probs.sum(axis=1, keepdims=True)
                    if (not np.isfinite(row_sums).all()) or np.any(row_sums <= 1e-12):
                        last_error = RuntimeError("zero_probability_rows")
                        continue
                    probs = probs / row_sums
                    return probs, None
                except Exception as exc:
                    last_error = exc
        except Exception as exc:
            last_error = exc

    prefix = "pomegranate_gmm_failed" if emission_backend == "pomegranate_gmm" else "pomegranate_failed"
    return None, f"{prefix}:{type(last_error).__name__ if last_error else 'unknown'}:{str(last_error)[:120] if last_error else ''}"


def _state_stats(raw_features: pd.DataFrame, state_probs: np.ndarray, meta: dict[str, Any]) -> tuple[dict[int, dict[str, float]], np.ndarray]:
    hard_states = np.argmax(state_probs, axis=1)
    trend_col = meta.get("hmm_trend_col")
    vol_col = meta.get("hmm_vol_col")
    range_col = meta.get("hmm_range_col")

    stats: dict[int, dict[str, float]] = {}
    n_states = state_probs.shape[1]
    for state in range(n_states):
        mask = hard_states == state
        subset = raw_features.loc[mask]
        if subset.empty:
            stats[state] = {"trend": 0.0, "return": 0.0, "volatility": 0.0, "range": 0.0, "occupancy": 0.0, "persistence": 0.0}
            continue
        trend = _safe_float(subset[trend_col].mean(), 0.0) if trend_col in subset else 0.0
        ret = _safe_float(subset["fd_return"].mean(), trend) if "fd_return" in subset else trend
        vol = _safe_float(subset[vol_col].mean(), 0.0) if vol_col in subset else 0.0
        rng = _safe_float(subset[range_col].mean(), 0.0) if range_col in subset else 0.0
        stats[state] = {
            "trend": trend,
            "return": ret,
            "volatility": abs(vol),
            "range": abs(rng),
            "occupancy": float(mask.mean()),
            "persistence": 0.0,
        }

    for state in range(n_states):
        idx = np.where(hard_states[:-1] == state)[0]
        if len(idx):
            stats[state]["persistence"] = float(np.mean(hard_states[idx + 1] == state))
    return stats, hard_states


def _map_states_to_regimes(stats: dict[int, dict[str, float]]) -> dict[int, str]:
    states = list(stats.keys())
    if not states:
        return {}
    if len(states) == 1:
        return {states[0]: "neutral"}

    scores = {
        state: _safe_float(values.get("trend"), 0.0) + 0.5 * _safe_float(values.get("return"), 0.0)
        for state, values in stats.items()
    }
    ordered = sorted(states, key=lambda s: scores[s])
    mapping = {state: "neutral" for state in states}
    mapping[ordered[0]] = "bear"
    mapping[ordered[-1]] = "bull"

    if len(states) == 2:
        return mapping

    # If the middle state is directionally close to zero, keep it neutral. If
    # all scores are nearly identical, use the highest volatility/range state
    # as the defensive bear regime and keep the rest neutral/bull by score.
    spread = abs(scores[ordered[-1]] - scores[ordered[0]])
    if spread <= 1e-8:
        defensive = max(states, key=lambda s: stats[s].get("volatility", 0.0) + stats[s].get("range", 0.0))
        mapping = {state: "neutral" for state in states}
        mapping[defensive] = "bear"
        remaining = [state for state in states if state != defensive]
        if remaining:
            mapping[max(remaining, key=lambda s: stats[s].get("occupancy", 0.0))] = "bull"
    return mapping


def _state_score_spread(stats: dict[int, dict[str, float]]) -> float:
    scores = [
        _safe_float(values.get("trend"), 0.0) + 0.5 * _safe_float(values.get("return"), 0.0)
        for values in stats.values()
    ]
    if len(scores) < 2:
        return 0.0
    return float(max(scores) - min(scores))


def _mapping_cache_key(symbol: str | None, emission_backend: str, n_states: int, feature_columns: list[str]) -> str:
    clean_symbol = str(symbol or "__global__")
    clean_backend = str(emission_backend or DEFAULT_EMISSION_BACKEND)
    clean_columns = ",".join(str(col) for col in feature_columns)
    return f"state_map::{clean_symbol}::{clean_backend}::{int(n_states)}::{clean_columns}"


def _soft_state_centroids(scaled_features: pd.DataFrame, state_probs: np.ndarray) -> np.ndarray:
    x = scaled_features.to_numpy(dtype=np.float64)
    probs = np.asarray(state_probs, dtype=np.float64)
    n_states = probs.shape[1]
    centroids = np.zeros((n_states, x.shape[1]), dtype=np.float64)
    global_mean = np.nanmean(x, axis=0) if x.size else np.zeros(x.shape[1], dtype=np.float64)
    global_mean = np.nan_to_num(global_mean, nan=0.0, posinf=0.0, neginf=0.0)

    for state in range(n_states):
        weights = probs[:, state]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        total = float(weights.sum())
        if total <= 1e-12:
            centroids[state] = global_mean
        else:
            centroids[state] = np.sum(x * weights[:, None], axis=0) / total
    return np.nan_to_num(centroids, nan=0.0, posinf=0.0, neginf=0.0)


def _greedy_state_match(previous: np.ndarray, current: np.ndarray) -> tuple[dict[int, int], float]:
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != current.shape or previous.ndim != 2:
        return {}, float("inf")

    distances = np.linalg.norm(current[:, None, :] - previous[None, :, :], axis=2)
    candidates: list[tuple[float, int, int]] = []
    for new_state in range(distances.shape[0]):
        for old_state in range(distances.shape[1]):
            candidates.append((float(distances[new_state, old_state]), new_state, old_state))
    candidates.sort(key=lambda item: item[0])

    matched_new: set[int] = set()
    matched_old: set[int] = set()
    mapping: dict[int, int] = {}
    max_distance = 0.0
    for distance, new_state, old_state in candidates:
        if new_state in matched_new or old_state in matched_old:
            continue
        matched_new.add(new_state)
        matched_old.add(old_state)
        mapping[new_state] = old_state
        max_distance = max(max_distance, distance)
        if len(mapping) == current.shape[0]:
            break

    if len(mapping) != current.shape[0]:
        return {}, float("inf")
    return mapping, max_distance


def _select_state_regime_mapping(
    *,
    symbol: str | None,
    emission_backend: str,
    requested_n_states: int,
    scaled_features: pd.DataFrame,
    state_probs: np.ndarray,
    stats: dict[int, dict[str, float]],
    config: dict,
) -> tuple[dict[int, str], str]:
    score_mapping = _map_states_to_regimes(stats)
    mapping = dict(score_mapping)
    method = "score_order"

    centroids = _soft_state_centroids(scaled_features, state_probs)
    feature_columns = [str(col) for col in scaled_features.columns]
    cache_key = _mapping_cache_key(symbol, emission_backend, requested_n_states, feature_columns)
    cached = _ANCHOR_STATE.get(cache_key)

    max_dist_threshold = max(0.0, _safe_float(config.get("HMM_MAP_MATCH_MAX_DIST", 2.0), 2.0))
    rebase_spread = max(0.0, _safe_float(config.get("HMM_MAP_REBASE_MIN_SPREAD", 0.5), 0.5))

    if isinstance(cached, dict):
        previous_centroids = np.asarray(cached.get("centroids"), dtype=float)
        previous_mapping = cached.get("state_to_regime")
        if isinstance(previous_mapping, dict) and previous_centroids.shape == centroids.shape:
            matched_states, max_distance = _greedy_state_match(previous_centroids, centroids)
            if matched_states and max_distance <= max_dist_threshold:
                matched_mapping = {
                    new_state: previous_mapping.get(old_state, "neutral")
                    for new_state, old_state in matched_states.items()
                }
                if matched_mapping != score_mapping and _state_score_spread(stats) > rebase_spread:
                    mapping = dict(score_mapping)
                    method = "score_order"
                else:
                    mapping = matched_mapping
                    method = "matched"

    _ANCHOR_STATE[cache_key] = {
        "centroids": centroids,
        "state_to_regime": dict(mapping),
    }
    return mapping, method


def _regime_probabilities(last_state_probs: np.ndarray, state_to_regime: dict[int, str]) -> dict[str, float]:
    out = {"bull": 0.0, "neutral": 0.0, "bear": 0.0}
    for state, prob in enumerate(last_state_probs):
        regime = state_to_regime.get(state, "neutral")
        if regime in out:
            out[regime] += _safe_float(prob, 0.0)
    return _normalize_probs(out)


def _apply_anchor(symbol: str | None, proposed_regime: str, regime_probs: dict[str, float], config: dict) -> tuple[str, str | None, bool, int, float]:
    key = str(symbol or "__global__")
    anchor = _ANCHOR_STATE.get(key, {})
    prev_regime = anchor.get("last_regime") if anchor.get("last_regime") in REGIMES else None
    prev_age = max(0, _safe_int(anchor.get("regime_age_bars", 0), 0))
    switch_confirm = max(1, _safe_int(config.get("HMM_SWITCH_CONFIRM_BARS", 2), 2))

    proposed_prob = _safe_float(regime_probs.get(proposed_regime), 0.0)
    prev_prob = _safe_float(regime_probs.get(prev_regime), 0.0) if prev_regime else 0.0
    switch_margin = proposed_prob - prev_prob

    accepted = proposed_regime
    changed = False
    age = 1
    pending_regime = None
    pending_count = 0

    if prev_regime is None:
        accepted = proposed_regime
        age = 1
    elif proposed_regime == prev_regime:
        accepted = prev_regime
        age = prev_age + 1
    else:
        old_pending = anchor.get("pending_regime")
        old_count = _safe_int(anchor.get("pending_count", 0), 0)
        pending_regime = proposed_regime
        pending_count = old_count + 1 if old_pending == proposed_regime else 1
        if pending_count >= switch_confirm:
            accepted = proposed_regime
            changed = True
            age = 1
            pending_regime = None
            pending_count = 0
        else:
            accepted = prev_regime
            age = prev_age + 1

    _ANCHOR_STATE[key] = {
        "last_regime": accepted,
        "regime_age_bars": age,
        "pending_regime": pending_regime,
        "pending_count": pending_count,
    }
    return accepted, prev_regime, changed, age, switch_margin


def _policy_for_regime(regime: str, confidence: float, regime_changed: bool, regime_age_bars: int, config: dict) -> str:
    min_conf = _safe_float(config.get("HMM_MIN_CONFIDENCE", 0.50), 0.50)
    if confidence < min_conf:
        policy = "caution"
    elif regime == "bull":
        policy = "allow"
    elif regime == "bear":
        policy = "block"
    else:
        policy = "caution"

    fresh_bars = max(0, _safe_int(config.get("HMM_SWITCH_BLOCK_FRESH_BARS", 2), 2))
    if regime_changed and regime_age_bars < fresh_bars:
        policy = "caution"
    return policy


def infer_hmm_regime(
    df: pd.DataFrame,
    config: dict,
    symbol: str | None = None,
    cycle_id: str | None = None,
    senate_candidate: Any | None = None,
) -> dict[str, Any]:
    del cycle_id, senate_candidate  # Kept for backward compatibility.

    cfg = config or {}
    enabled = _as_bool(cfg.get("HMM_ENABLED", True), True)
    backend = str(cfg.get("HMM_BACKEND", DEFAULT_BACKEND))
    emission_backend = str(cfg.get("HMM_EMISSION_BACKEND", DEFAULT_EMISSION_BACKEND))
    anchor_method = str(cfg.get("HMM_ANCHOR_METHOD", DEFAULT_ANCHOR_METHOD))

    if not enabled:
        return _neutral_result(False, "hmm_disabled", cfg, backend=backend)
    if backend != "pomegranate":
        return _neutral_result(True, f"unsupported_backend:{backend}", cfg, backend=backend)
    if emission_backend not in {"pomegranate_normal", "pomegranate_gmm"}:
        return _neutral_result(True, f"unsupported_emission_backend:{emission_backend}", cfg, backend=backend)

    try:
        raw_features, scaled_features, meta, feature_error = _prepare_features(df, cfg)
        if feature_error is not None:
            return _neutral_result(True, feature_error, cfg, feature_meta=meta, backend=backend)

        fit_cfg = dict(cfg)
        if meta.get("hmm_vol_col"):
            fit_cfg["_HMM_FEATURE_VOL_COL"] = meta.get("hmm_vol_col")

        effective_emission = emission_backend
        reason_detail = f"backend={backend};emission={emission_backend};gate=regime_filter"

        state_probs, hmm_error = _fit_predict_pomegranate(scaled_features, fit_cfg, emission_backend)
        if hmm_error is not None and emission_backend == "pomegranate_gmm":
            gmm_error_text = str(hmm_error)[:180]
            if _as_bool(cfg.get("HMM_GMM_FALLBACK_TO_NORMAL", True), True):
                normal_probs, normal_error = _fit_predict_pomegranate(scaled_features, fit_cfg, "pomegranate_normal")
                if normal_error is None and normal_probs is not None:
                    state_probs = normal_probs
                    hmm_error = None
                    effective_emission = "pomegranate_normal"
                    reason_detail = (
                        f"backend={backend};emission=pomegranate_gmm;"
                        f"fallback=pomegranate_normal;gmm_error={gmm_error_text}"
                    )
                else:
                    return _neutral_result(
                        True,
                        normal_error or f"pomegranate_gmm_failed:{hmm_error}",
                        cfg,
                        feature_meta=meta,
                        backend=backend,
                    )
            else:
                reason = hmm_error if str(hmm_error).startswith("pomegranate_gmm_failed") else f"pomegranate_gmm_failed:{hmm_error}"
                return _neutral_result(True, reason, cfg, feature_meta=meta, backend=backend)

        if hmm_error is not None or state_probs is None:
            return _neutral_result(True, hmm_error or "pomegranate_failed:unknown", cfg, feature_meta=meta, backend=backend)

        if state_probs.ndim != 2 or state_probs.shape[0] != len(raw_features):
            return _neutral_result(True, "pomegranate_failed:invalid_probability_shape", cfg, feature_meta=meta, backend=backend)

        last_state_probs = np.asarray(state_probs[-1], dtype=float)
        if not np.isfinite(last_state_probs).all() or last_state_probs.sum() <= 1e-12:
            return _neutral_result(True, "pomegranate_failed:invalid_probabilities", cfg, feature_meta=meta, backend=backend)
        last_state_probs = last_state_probs / last_state_probs.sum()

        stats, _hard_states = _state_stats(raw_features, state_probs, meta)
        requested_n_states = max(2, _safe_int(cfg.get("HMM_N_STATES", state_probs.shape[1]), state_probs.shape[1]))
        state_to_regime, mapping_method = _select_state_regime_mapping(
            symbol=symbol,
            emission_backend=effective_emission,
            requested_n_states=requested_n_states,
            scaled_features=scaled_features,
            state_probs=state_probs,
            stats=stats,
            config=cfg,
        )
        regime_probs = _regime_probabilities(last_state_probs, state_to_regime)
        regime_margin = _prob_margin(list(regime_probs.values()))
        debug_meta = {
            "hmm_state_prob_max": float(np.max(last_state_probs)),
            "hmm_state_prob_min": float(np.min(last_state_probs)),
            "hmm_state_prob_margin": _prob_margin(last_state_probs),
            "hmm_regime_prob_margin": regime_margin,
            "hmm_state_prob_entropy": _prob_entropy(last_state_probs),
            "hmm_effective_emission_backend": effective_emission,
            "hmm_mapping_method": mapping_method,
        }

        proposed_regime = max(regime_probs, key=regime_probs.get)
        filtered_confidence = _safe_float(regime_probs.get(proposed_regime), 0.0)
        min_margin = max(0.0, _safe_float(cfg.get("HMM_MIN_REGIME_PROB_MARGIN", 0.05), 0.05))
        low_information = regime_margin < min_margin
        if low_information:
            accepted_regime = "neutral"
            prev_regime = None
            changed = False
            age = 0
            switch_margin = regime_margin
            confidence = filtered_confidence
            policy = "caution"
            reason_detail = f"{reason_detail};low_information_posterior"
        else:
            accepted_regime, prev_regime, changed, age, switch_margin = _apply_anchor(symbol, proposed_regime, regime_probs, cfg)
            confidence = _safe_float(regime_probs.get(accepted_regime), filtered_confidence)
            policy = _policy_for_regime(accepted_regime, confidence, changed, age, cfg)

        current_state = int(np.argmax(last_state_probs))
        accepted_states = [state for state, regime in state_to_regime.items() if regime == accepted_regime]
        if accepted_states:
            persistence_state = max(accepted_states, key=lambda s: last_state_probs[s])
        else:
            persistence_state = current_state
        persistence = _safe_float(stats.get(persistence_state, {}).get("persistence"), 0.0)

        return _base_result(
            enabled=True,
            ok=True,
            reason=reason_detail,
            backend=backend,
            anchor_method=anchor_method,
            feature_meta=meta,
            regime=accepted_regime,
            policy=policy,
            state=current_state,
            confidence=confidence,
            probs=regime_probs,
            persistence=persistence,
            switch_margin=switch_margin,
            prev_regime=prev_regime,
            regime_changed=changed,
            regime_age_bars=age,
            filtered_confidence=filtered_confidence,
            debug_meta=debug_meta,
        )
    except Exception as exc:
        return _neutral_result(True, f"hmm_exception:{type(exc).__name__}:{str(exc)[:160]}", cfg, backend=backend)
