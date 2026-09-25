# ==============================================================================
# XGBoost.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Offline primary-model training and evaluation. Not used by live trading.
# ==============================================================================

from __future__ import annotations

import importlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from TripleBarrier import PurgedWalkForwardCV
from primary_features import build_xy, select_features_fold


def _load_xgboost_package():
    """Avoid XGBoost.py shadowing the lowercase package on case-insensitive disks."""
    project_root = Path(__file__).resolve().parent
    original_path = list(sys.path)
    try:
        sys.path[:] = [
            entry
            for entry in sys.path
            if Path(entry or Path.cwd()).resolve() != project_root
        ]
        return importlib.import_module("xgboost")
    finally:
        sys.path[:] = original_path


xgboost = _load_xgboost_package()
XGBClassifier = xgboost.XGBClassifier


def _project_config(config: dict | None = None) -> dict:
    if config is not None:
        return dict(config)
    from config import CONFIG

    return dict(CONFIG)


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ordered_intersection(left: pd.Index, right: pd.Index) -> pd.Index:
    right_values = set(right)
    return pd.Index([value for value in left if value in right_values])


def _selected_features(importances: pd.Series, keep_ratio: float) -> list[str]:
    if importances.empty:
        return []
    ratio = min(max(float(keep_ratio), 0.0), 1.0)
    keep_count = max(1, int(math.ceil(len(importances) * ratio)))
    return list(importances.sort_values(ascending=False).head(keep_count).index)


def _scale_pos_weight(y: pd.Series, w: pd.Series) -> float | None:
    y_values = pd.Series(y).astype(int)
    weights = pd.Series(np.asarray(w, dtype=float), index=y_values.index)
    positive_weight = float(weights.loc[y_values == 1].sum())
    if positive_weight <= 0.0:
        return None
    negative_weight = float(weights.loc[y_values == 0].sum())
    return negative_weight / positive_weight


def round_trip_cost(config: dict | None = None) -> float:
    cfg = _project_config(config)
    fee = float(cfg.get("PRIMARY_FEE_BPS_PER_SIDE", 10.0))
    slippage = float(cfg.get("PRIMARY_SLIPPAGE_BPS_PER_SIDE", 5.0))
    return 2.0 * (fee + slippage) / 10000.0


def fit_fold_model(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    w_tr: pd.Series,
    config: dict | None = None,
    feature_subset: list[str] | None = None,
):
    cfg = _project_config(config)
    columns = list(feature_subset) if feature_subset is not None else list(X_tr.columns)
    missing = [column for column in columns if column not in X_tr.columns]
    if missing:
        raise ValueError(f"feature_subset contains missing columns: {missing}")

    scale_pos_weight = _scale_pos_weight(y_tr, w_tr)
    if scale_pos_weight is None:
        return None

    params = dict(cfg.get("PRIMARY_XGB_PARAMS", {}))
    params["random_state"] = int(cfg.get("PRIMARY_RANDOM_SEED", 41))
    params["scale_pos_weight"] = float(scale_pos_weight)
    model = XGBClassifier(**params)
    model.fit(X_tr.loc[:, columns], y_tr, sample_weight=w_tr)
    model.primary_feature_columns_ = columns
    model.primary_scale_pos_weight_ = float(scale_pos_weight)
    return model


def _trade_stats(y: pd.Series, ret: pd.Series, event: pd.Series, mask: np.ndarray, cost: float) -> dict:
    selected = np.asarray(mask, dtype=bool)
    count = int(selected.sum())
    if count == 0:
        return {
            "count": 0,
            "win_rate": None,
            "mean_net": None,
            "sum_net": 0.0,
            "vertical_rate": None,
        }

    y_selected = np.asarray(y, dtype=int)[selected]
    ret_selected = np.asarray(ret, dtype=float)[selected]
    event_selected = np.asarray(event, dtype=object)[selected]
    net = ret_selected - float(cost)
    return {
        "count": count,
        "win_rate": float(np.mean(y_selected == 1)),
        "mean_net": float(np.mean(net)),
        "sum_net": float(np.sum(net)),
        "vertical_rate": float(np.mean(event_selected == "vertical")),
    }


def evaluate_fold(
    model,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    ret_te: pd.Series,
    event_te: pd.Series,
    config: dict | None = None,
) -> dict:
    cfg = _project_config(config)
    columns = list(getattr(model, "primary_feature_columns_", X_te.columns))
    probabilities = np.asarray(model.predict_proba(X_te.loc[:, columns])[:, 1], dtype=float)
    y_values = pd.Series(y_te).astype(int)
    auc = float(roc_auc_score(y_values, probabilities)) if y_values.nunique() >= 2 else None
    cost = round_trip_cost(cfg)

    thresholds: dict[str, dict] = {}
    for threshold in cfg.get("PRIMARY_PROB_THRESHOLDS", [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]):
        threshold_value = float(threshold)
        thresholds[f"{threshold_value:.2f}"] = _trade_stats(
            y_values,
            ret_te,
            event_te,
            probabilities >= threshold_value,
            cost,
        )

    return {
        "auc": auc,
        "base_positive_rate": float(y_values.mean()) if len(y_values) else None,
        "thresholds": thresholds,
        "baseline": _trade_stats(
            y_values,
            ret_te,
            event_te,
            np.ones(len(y_values), dtype=bool),
            cost,
        ),
        "probabilities": probabilities,
    }


def _auc_summary(folds: list[dict], arm: str) -> dict:
    values = [
        float(fold[arm]["auc"])
        for fold in folds
        if fold[arm]["auc"] is not None
    ]
    return {
        "mean_auc": float(np.mean(values)) if values else None,
        "worst_auc": float(np.min(values)) if values else None,
        "valid_auc_folds": int(len(values)),
    }


def run_purged_cv(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = _project_config(config)
    xy = build_xy(df, cfg)
    xy_index = pd.Index(xy["index"])
    cv = PurgedWalkForwardCV(
        n_splits=int(cfg.get("PRIMARY_CV_SPLITS", 5)),
        embargo_bars=int(cfg.get("TRIPLE_BARRIER_EMBARGO_BARS", 24)),
    )

    folds: list[dict] = []
    oof_parts: list[pd.DataFrame] = []
    for fold_number, (train_pos, test_pos) in enumerate(cv.split(df), start=1):
        train_index = _ordered_intersection(pd.Index(df.index[train_pos]), xy_index)
        test_index = _ordered_intersection(pd.Index(df.index[test_pos]), xy_index)
        if len(train_index) == 0 or len(test_index) == 0:
            continue

        X_tr = xy["X"].loc[train_index]
        y_tr = xy["y"].loc[train_index]
        w_tr = xy["w"].loc[train_index]
        X_te = xy["X"].loc[test_index]
        y_te = xy["y"].loc[test_index]
        w_te = xy["w"].loc[test_index]
        ret_te = xy["ret"].loc[test_index]
        event_te = df.loc[test_index, "tb_event"]
        end_times_tr = pd.Series(
            pd.to_datetime(df.loc[train_index, "tb_exit_index"], utc=True, errors="coerce"),
            index=train_index,
        )

        selection = select_features_fold(X_tr, y_tr, w_tr, end_times_tr, cfg)
        if selection is None:
            continue
        selected = _selected_features(
            selection["importances"],
            float(cfg.get("PRIMARY_FS_KEEP_RATIO", 0.5)),
        )
        if not selected:
            continue

        model_a = fit_fold_model(X_tr, y_tr, w_tr, cfg)
        model_b = fit_fold_model(X_tr, y_tr, w_tr, cfg, feature_subset=selected)
        if model_a is None or model_b is None:
            continue

        metrics_a = evaluate_fold(model_a, X_te, y_te, ret_te, event_te, cfg)
        metrics_b = evaluate_fold(model_b, X_te, y_te, ret_te, event_te, cfg)
        probabilities_b = metrics_b.pop("probabilities")
        metrics_a.pop("probabilities")

        scale_pos_weight = float(model_b.primary_scale_pos_weight_)
        folds.append(
            {
                "fold": int(fold_number),
                "train_range": (_timestamp(train_index.min()), _timestamp(train_index.max())),
                "test_range": (_timestamp(test_index.min()), _timestamp(test_index.max())),
                "train_events": int(len(train_index)),
                "test_events": int(len(test_index)),
                "selected_features_b": selected,
                "scale_pos_weight": scale_pos_weight,
                "A": metrics_a,
                "B": metrics_b,
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "fold": int(fold_number),
                    "p_primary": probabilities_b,
                    "y": y_te.to_numpy(dtype=int),
                    "tb_return": ret_te.to_numpy(dtype=float),
                    "tb_uniqueness": w_te.to_numpy(dtype=float),
                },
                index=pd.DatetimeIndex(test_index, name="OpenTime"),
            )
        )

    oof = (
        pd.concat(oof_parts).sort_index()
        if oof_parts
        else pd.DataFrame(columns=["fold", "p_primary", "y", "tb_return", "tb_uniqueness"])
    )
    if not oof.empty:
        oof.index.name = "OpenTime"
        assert not oof.index.has_duplicates, "OOF OpenTime index contains duplicates"

    return {
        "folds": folds,
        "summary": {
            "A": _auc_summary(folds, "A"),
            "B": _auc_summary(folds, "B"),
        },
        "oof": oof,
    }


def save_oof(
    oof_frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: dict | None = None,
) -> Path:
    cfg = _project_config(config)
    out_dir = Path(cfg.get("PRIMARY_MODEL_DIR", Path("data") / "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = oof_frame.copy()
    if "OpenTime" in frame.columns:
        frame = frame.set_index("OpenTime")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True), name="OpenTime")
    assert not frame.index.has_duplicates, "OOF OpenTime index contains duplicates"
    required = ["fold", "p_primary", "y", "tb_return", "tb_uniqueness"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"OOF frame is missing columns: {missing}")
    out_path = out_dir / f"primary_oof_{str(symbol).upper()}_{timeframe}.csv"
    frame.loc[:, required].to_csv(out_path, index=True)
    return out_path


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _infer_symbol(df: pd.DataFrame, config: dict) -> str:
    if "Symbol" in df.columns:
        symbols = pd.Series(df["Symbol"]).dropna().astype(str).unique()
        if len(symbols) == 1:
            return str(symbols[0]).upper()
    return str(config.get("PRIMARY_TRAIN_SYMBOL", "UNKNOWN")).upper()


def train_final_model(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = _project_config(config)
    xy = build_xy(df, cfg)
    end_times = pd.Series(
        pd.to_datetime(df.loc[xy["index"], "tb_exit_index"], utc=True, errors="coerce"),
        index=xy["index"],
    )
    selection = select_features_fold(xy["X"], xy["y"], xy["w"], end_times, cfg)
    if selection is None:
        raise ValueError("Final feature selection could not produce a valid result")
    selected = _selected_features(
        selection["importances"],
        float(cfg.get("PRIMARY_FS_KEEP_RATIO", 0.5)),
    )
    model = fit_fold_model(
        xy["X"],
        xy["y"],
        xy["w"],
        cfg,
        feature_subset=selected,
    )
    if model is None:
        raise ValueError("Final model training requires positive-class train weight")

    symbol = _infer_symbol(df, cfg)
    timeframe = str(cfg.get("PRIMARY_TRAIN_TIMEFRAME", cfg.get("DATASET_DEFAULT_TIMEFRAME", "1h")))
    out_dir = Path(cfg.get("PRIMARY_MODEL_DIR", Path("data") / "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"primary_{symbol}_{timeframe}.json"
    meta_path = out_dir / f"primary_{symbol}_{timeframe}_meta.json"
    model.save_model(model_path)

    cv_summary = cfg.get("PRIMARY_CV_SUMMARY", {})
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "final_features": selected,
        "params": model.get_params(),
        "scale_pos_weight": float(model.primary_scale_pos_weight_),
        "seed": int(cfg.get("PRIMARY_RANDOM_SEED", 41)),
        "data_range": [
            _timestamp(xy["index"].min()).isoformat(),
            _timestamp(xy["index"].max()).isoformat(),
        ],
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "cv_summary": cv_summary,
        "note": "این مدل روی کل داده آموزش دیده؛ متریک‌های معتبر فقط از CV هستند. متصل به live نیست.",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=_json_default)

    return {
        "model": model,
        "model_path": model_path,
        "metadata_path": meta_path,
        "selected_features": selected,
        "scale_pos_weight": float(model.primary_scale_pos_weight_),
    }
