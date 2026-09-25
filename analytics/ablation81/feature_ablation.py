# ==============================================================================
# analytics/ablation81/feature_ablation.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Frozen column-only transformations for ablation81 phase four."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

import meta_model


BASE_FEATURES = tuple(meta_model.META_FEATURE_COLUMNS) + tuple(
    meta_model.META_HMM_FEATURE_COLUMNS
)
BASE_CATEGORICAL = ("symbol", "hmm_regime")
CALENDAR_FEATURES = ("hour_of_day", "day_of_week")
MICRO_FUNDING_FEATURES = (
    "vpin",
    "vpin_z",
    "funding_z",
    "funding_extreme_pos",
)
HMM_FEATURES = tuple(meta_model.META_HMM_FEATURE_COLUMNS)
E6_FEATURES = ("btc_dist_from_max_168", "close_z_4h")
MODEL_ARMS = ("A5", "A6", "E3", "E4", "E5", "E6")
SESSION_CATEGORIES = ("Asia", "Europe", "America")


class FeatureAblationError(RuntimeError):
    """Raised when a phase-four arm changes more than its registered columns."""


def _remove_exact(base: Sequence[str], removed: Sequence[str]) -> list[str]:
    missing = sorted(set(removed) - set(base))
    if missing:
        raise FeatureAblationError(f"cannot remove absent columns: {missing}")
    return [column for column in base if column not in set(removed)]


def materialize_session(decision_ts: pd.Series) -> pd.Series:
    """Map the pinned UTC decision hours to the three registered sessions."""

    timestamps = pd.to_datetime(decision_ts, utc=True, errors="raise")
    hours = timestamps.dt.hour
    unexpected = sorted(set(hours.unique().tolist()) - {0, 4, 8, 12, 16, 20})
    if unexpected:
        raise FeatureAblationError(
            f"session arm encountered non-4h decision hours: {unexpected}"
        )
    mapped = hours.map(
        {0: "Asia", 4: "Asia", 8: "Europe", 12: "Europe", 16: "America", 20: "America"}
    )
    if mapped.isna().any():
        raise FeatureAblationError("session materialization produced null values")
    return pd.Series(
        pd.Categorical(mapped, categories=list(SESSION_CATEGORIES)),
        index=decision_ts.index,
        name="session",
    )


def build_frozen_arm_spec(
    arm: str,
    *,
    config: Mapping[str, Any],
    tree_count: int,
    expected_config: Mapping[str, Any],
    expected_tree_count: int,
    base_features: Sequence[str],
    base_categorical: Sequence[str],
) -> dict[str, Any]:
    """Return one arm spec after proving config/tree count are unchanged."""

    name = str(arm)
    if name not in MODEL_ARMS:
        raise FeatureAblationError(f"unknown phase-four model arm: {name}")
    if dict(config) != dict(expected_config):
        raise FeatureAblationError(f"{name} changed the frozen model configuration")
    if int(tree_count) != int(expected_tree_count):
        raise FeatureAblationError(f"{name} changed the frozen tree count")
    if tuple(base_features) != BASE_FEATURES:
        raise FeatureAblationError("base feature schema differs from frozen A4")
    if tuple(base_categorical) != BASE_CATEGORICAL:
        raise FeatureAblationError("base categorical schema differs from frozen A4")

    features = list(BASE_FEATURES)
    categorical = list(BASE_CATEGORICAL)
    removed: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    if name == "A5":
        removed = CALENDAR_FEATURES
        features = _remove_exact(features, removed)
    elif name == "A6":
        removed = MICRO_FUNDING_FEATURES
        features = _remove_exact(features, removed)
    elif name == "E3":
        removed = CALENDAR_FEATURES
        added = ("session",)
        insert_at = features.index(CALENDAR_FEATURES[0])
        features = _remove_exact(features, removed)
        features.insert(insert_at, "session")
        categorical.append("session")
    elif name == "E4":
        removed = ("symbol",)
        features = _remove_exact(features, removed)
        categorical.remove("symbol")
    elif name == "E5":
        removed = HMM_FEATURES
        features = _remove_exact(features, removed)
        categorical.remove("hmm_regime")
    elif name == "E6":
        added = E6_FEATURES
        if set(added) & set(features):
            raise FeatureAblationError("E6 additions already exist in frozen A4")
        features.extend(added)

    if len(features) != len(set(features)):
        raise FeatureAblationError(f"{name} produced duplicate feature columns")
    if not set(categorical).issubset(set(features)):
        raise FeatureAblationError(f"{name} categorical columns are not features")
    expected = (set(BASE_FEATURES) - set(removed)) | set(added)
    if set(features) != expected:
        raise FeatureAblationError(f"{name} changed unregistered columns")
    return {
        "arm": name,
        "feature_columns": features,
        "categorical_features": categorical,
        "removed_columns": list(removed),
        "added_columns": list(added),
        "config": dict(config),
        "tree_count": int(tree_count),
        "eval_set_used": False,
        "early_stopping_used": False,
        "search_or_reselection_used": False,
    }


def project_arm_frame(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    """Project both fit and prediction inputs to the exact registered schema."""

    columns = list(spec.get("feature_columns", []))
    if not columns or len(columns) != len(set(columns)):
        raise FeatureAblationError("arm projection received an invalid feature list")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise FeatureAblationError(f"arm projection misses columns: {missing}")
    projected = frame.loc[:, columns].copy()
    if list(projected.columns) != columns:
        raise FeatureAblationError("arm projection changed registered column order")
    return projected
