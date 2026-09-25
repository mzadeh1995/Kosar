# ==============================================================================
# primary_features.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Offline primary-model feature selection infrastructure. Not used by live trading.
# ==============================================================================

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from TripleBarrier import PurgedWalkForwardCV


RAW_MARKET_COLUMNS = {
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "QuoteVolume",
    "Trades",
    "TakerBuyBase",
    "TakerBuyQuote",
}

def _project_config(config: dict | None = None) -> dict:
    if config is not None:
        return dict(config)
    from config import CONFIG

    return dict(CONFIG)


def _utc_datetime_index(values: Any, name: str = "OpenTime") -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError(f"{name} contains values that cannot be parsed as UTC datetimes")
    return pd.DatetimeIndex(parsed, name=name)


def load_symbol_dataset(path: str | Path, dataset_dir: str | Path | None = None) -> pd.DataFrame:
    path = Path(path)
    if dataset_dir is None:
        dataset_dir = _project_config().get("DATASET_OUTPUT_DIR")
    if not path.is_absolute() and not path.exists() and dataset_dir is not None:
        candidate = Path(dataset_dir) / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        if "OpenTime" not in df.columns:
            raise ValueError(f"CSV dataset must contain an OpenTime column: {path}")
        df["OpenTime"] = _utc_datetime_index(df["OpenTime"])
        df = df.set_index("OpenTime")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        if "OpenTime" in df.columns:
            df["OpenTime"] = _utc_datetime_index(df["OpenTime"])
            df = df.set_index("OpenTime")
        else:
            df.index = _utc_datetime_index(df.index)
    else:
        raise ValueError(f"Unsupported dataset suffix for {path}; expected .csv or .parquet")

    df.index = _utc_datetime_index(df.index)
    df = df.sort_index()
    if df.index.has_duplicates:
        duplicate_mask = df.index.duplicated(keep=False)
        duplicate_count = int(duplicate_mask.sum())
        duplicate_samples = df.index[duplicate_mask].unique()[:3].tolist()
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate OpenTime values in {path}; "
            f"sample duplicate timestamps: {duplicate_samples}"
        )

    if "tb_exit_index" not in df.columns:
        raise ValueError("Dataset must contain tb_exit_index")
    df["tb_exit_index"] = pd.to_datetime(df["tb_exit_index"], utc=True, errors="coerce")

    if "Symbol" in df.columns:
        symbols = pd.Series(df["Symbol"]).dropna().unique()
        if len(symbols) > 1:
            raise ValueError("load_symbol_dataset expects one symbol per file; found multiple Symbol values")

    return df


def _feature_columns(config: dict) -> list[str]:
    cols = list(config.get("PRIMARY_FEATURE_COLUMNS", []))
    if not cols:
        raise ValueError("PRIMARY_FEATURE_COLUMNS is empty or missing")
    return [str(col) for col in cols]


def _validate_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    tb_cols = [col for col in feature_cols if col.startswith("tb_")]
    if tb_cols:
        raise ValueError(f"Leakage guard: tb_* columns cannot be used as X features: {tb_cols}")

    hmm_cols = [col for col in feature_cols if col.startswith("hmm_")]
    if hmm_cols:
        raise ValueError(f"Leakage guard: hmm_* columns cannot be used as X features: {hmm_cols}")

    raw_cols = [col for col in feature_cols if col in RAW_MARKET_COLUMNS]
    if raw_cols:
        raise ValueError(f"Leakage guard: raw market/OHLCV columns cannot be used as X features: {raw_cols}")

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing PRIMARY_FEATURE_COLUMNS in dataset: {missing}")


def build_xy(df: pd.DataFrame, config: dict | None = None) -> dict:
    """Build spot buy/no-buy labels from Triple Barrier rows.

    The positive class is `tb_label == 1`; both vertical (`0`) and stop-loss
    (`-1`) events map to the negative class because the live system is spot-only
    and the practical primary decision is buy vs. do not buy.
    """

    cfg = _project_config(config)
    feature_cols = _feature_columns(cfg)
    _validate_feature_columns(df, feature_cols)

    required = ["tb_label", "tb_return", "tb_uniqueness"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Dataset is missing required Triple Barrier columns: {missing_required}")

    events = df.loc[df["tb_label"].notna()].copy()
    X = events[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    assert not any(str(col).startswith("tb_") for col in X.columns), "Leakage guard: tb_* columns reached X"
    assert not any(str(col).startswith("hmm_") for col in X.columns), "Leakage guard: hmm_* columns reached X"
    assert not any(str(col) in RAW_MARKET_COLUMNS for col in X.columns), "Leakage guard: raw market columns reached X"
    y = pd.Series((pd.to_numeric(events["tb_label"], errors="coerce") == 1.0).astype(int), index=events.index, name="y")
    w = pd.Series(pd.to_numeric(events["tb_uniqueness"], errors="coerce"), index=events.index, name="w")
    ret = pd.Series(pd.to_numeric(events["tb_return"], errors="coerce"), index=events.index, name="ret")

    valid = X.notna().all(axis=1) & w.notna() & ret.notna()
    dropped_rows = int((~valid).sum())

    X = X.loc[valid].astype(float)
    y = y.loc[valid].astype(int)
    w = w.loc[valid].astype(float)
    ret = ret.loc[valid].astype(float)

    assert X.index.equals(y.index)
    assert X.index.equals(w.index)
    assert X.index.equals(ret.index)

    return {
        "X": X,
        "y": y,
        "w": w,
        "ret": ret,
        "index": X.index,
        "dropped_rows": dropped_rows,
        "event_rows": int(events.shape[0]),
    }


def _has_two_classes(y: pd.Series) -> bool:
    return int(pd.Series(y).dropna().nunique()) >= 2


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _rf_params(config: dict) -> dict:
    params = dict(config.get("PRIMARY_RF_PARAMS", {}))
    params.setdefault("n_estimators", 400)
    params.setdefault("max_features", "sqrt")
    params.setdefault("n_jobs", -1)
    params["random_state"] = int(config.get("PRIMARY_RANDOM_SEED", 41))
    return params


def select_features_fold(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    w_tr: pd.Series,
    end_times_tr: pd.Series,
    config: dict | None = None,
) -> dict | None:
    cfg = _project_config(config)
    if not X_tr.index.equals(y_tr.index) or not X_tr.index.equals(w_tr.index):
        raise ValueError("X_tr, y_tr, and w_tr must share the same index and order")

    end_times = pd.Series(pd.to_datetime(end_times_tr, utc=True, errors="coerce"), index=X_tr.index)
    if end_times.isna().any():
        raise ValueError("end_times_tr contains unparseable timestamps")

    X_tr = X_tr.sort_index()
    y_tr = y_tr.reindex(X_tr.index)
    w_tr = w_tr.reindex(X_tr.index)
    end_times = end_times.reindex(X_tr.index)

    n_rows = int(X_tr.shape[0])
    if n_rows < 4:
        warnings.warn("Skipping feature-selection fold: not enough train rows for internal validation")
        return None

    val_fraction = float(cfg.get("PRIMARY_FS_VAL_FRACTION", 0.2))
    val_fraction = min(max(val_fraction, 0.01), 0.99)
    val_size = max(1, int(math.ceil(n_rows * val_fraction)))
    split_pos = n_rows - val_size
    if split_pos <= 0:
        warnings.warn("Skipping feature-selection fold: internal subtrain split is empty")
        return None

    val_start_ts = _timestamp(X_tr.index[split_pos])
    sub_candidate_index = X_tr.index[:split_pos]
    sub_index = sub_candidate_index[end_times.loc[sub_candidate_index] < val_start_ts]
    val_index = X_tr.index[split_pos:]

    if len(sub_index) == 0 or len(val_index) == 0:
        warnings.warn("Skipping feature-selection fold: internal purge left an empty subtrain or validation slice")
        return None

    X_sub = X_tr.loc[sub_index]
    y_sub = y_tr.loc[sub_index]
    w_sub = w_tr.loc[sub_index]
    X_val = X_tr.loc[val_index]
    y_val = y_tr.loc[val_index]
    w_val = w_tr.loc[val_index]

    if not _has_two_classes(y_sub):
        warnings.warn("Skipping feature-selection fold: internal subtrain slice has only one class")
        return None
    if not _has_two_classes(y_val):
        warnings.warn("Skipping feature-selection fold: internal validation slice has only one class")
        return None

    rf = RandomForestClassifier(**_rf_params(cfg))
    rf.fit(X_sub, y_sub, sample_weight=w_sub)

    repeats = max(1, int(cfg.get("PRIMARY_FS_PERM_REPEATS", 5)))
    seed = int(cfg.get("PRIMARY_RANDOM_SEED", 41))
    perm_weighted = True
    try:
        perm = permutation_importance(
            rf,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=repeats,
            random_state=seed,
            sample_weight=w_val,
        )
    except TypeError:
        perm_weighted = False
        perm = permutation_importance(
            rf,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=repeats,
            random_state=seed,
        )

    importances = pd.Series(perm.importances_mean, index=X_tr.columns, name="importance", dtype=float)
    used_index = X_sub.index.append(X_val.index)
    return {
        "importances": importances,
        "used_range": (_timestamp(used_index.min()), _timestamp(used_index.max())),
        "perm_weighted": perm_weighted,
    }


def _ordered_intersection(left: pd.Index, right: pd.Index) -> pd.Index:
    right_set = set(right)
    return pd.Index([value for value in left if value in right_set])


def _top_features(importances: pd.Series, keep_ratio: float) -> list[str]:
    if importances.empty:
        return []
    keep_ratio = min(max(float(keep_ratio), 0.0), 1.0)
    keep_count = max(1, int(math.ceil(len(importances) * keep_ratio)))
    return list(importances.sort_values(ascending=False).head(keep_count).index)


def select_features_cv(df: pd.DataFrame, config: dict | None = None) -> dict:
    cfg = _project_config(config)
    xy = build_xy(df, cfg)
    feature_cols = list(xy["X"].columns)
    cv = PurgedWalkForwardCV(
        n_splits=int(cfg.get("PRIMARY_CV_SPLITS", 5)),
        embargo_bars=int(cfg.get("TRIPLE_BARRIER_EMBARGO_BARS", 24)),
    )

    importances_by_fold: list[pd.Series] = []
    fold_used_ranges: list[dict] = []
    perm_weighted_by_fold: list[bool] = []
    xy_index = pd.Index(xy["index"])

    for fold_number, (train_pos, _test_pos) in enumerate(cv.split(df), start=1):
        raw_train_index = pd.Index(df.index[train_pos])
        train_index = _ordered_intersection(raw_train_index, xy_index)
        if len(train_index) == 0:
            warnings.warn(f"Skipping feature-selection fold {fold_number}: no usable train rows after build_xy filtering")
            continue

        X_tr = xy["X"].loc[train_index]
        y_tr = xy["y"].loc[train_index]
        w_tr = xy["w"].loc[train_index]
        end_times_tr = pd.Series(pd.to_datetime(df.loc[train_index, "tb_exit_index"], utc=True, errors="coerce"), index=train_index)

        result = select_features_fold(X_tr, y_tr, w_tr, end_times_tr, cfg)
        if result is None:
            continue

        importances_by_fold.append(result["importances"])
        fold_used_ranges.append(
            {
                "fold": int(fold_number),
                "used_range": result["used_range"],
            }
        )
        perm_weighted_by_fold.append(bool(result["perm_weighted"]))

    if importances_by_fold:
        mean_importances = pd.concat(importances_by_fold, axis=1).mean(axis=1).reindex(feature_cols)
    else:
        mean_importances = pd.Series(0.0, index=feature_cols, name="importance", dtype=float)
    mean_importances.name = "mean_importance"

    selected_features = _top_features(mean_importances, float(cfg.get("PRIMARY_FS_KEEP_RATIO", 0.5))) if importances_by_fold else []
    non_positive = list(mean_importances[mean_importances <= 0.0].index)

    return {
        "mean_importances": mean_importances,
        "valid_folds": int(len(importances_by_fold)),
        "selected_features": selected_features,
        "non_positive_features": non_positive,
        "fold_used_ranges": fold_used_ranges,
        "perm_weighted": bool(all(perm_weighted_by_fold)) if perm_weighted_by_fold else None,
        "perm_weighted_by_fold": perm_weighted_by_fold,
        "event_rows": int(xy["event_rows"]),
        "usable_rows": int(xy["X"].shape[0]),
        "dropped_rows": int(xy["dropped_rows"]),
    }
