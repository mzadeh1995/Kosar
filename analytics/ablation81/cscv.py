# ==============================================================================
# analytics/ablation81/cscv.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Group-safe CSCV prediction and matrix construction for phase three."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .nested import CONFIG_KEYS, OUTER_FOLDS, NestedSearchError


PREDICTION_COLUMNS = (
    "config_id",
    "outer_fold",
    "symbol",
    "decision_ts",
    "p_raw",
)
BLOCK_COLUMNS = tuple(
    f"fold_{fold}_block_{block}"
    for fold in OUTER_FOLDS
    for block in range(1, 5)
)
MATRIX_COLUMNS = (
    "config_id",
    *CONFIG_KEYS,
    *BLOCK_COLUMNS,
)


class CSCVError(RuntimeError):
    """Raised when CSCV rows or block semantics violate the contract."""


def assign_group_safe_blocks(
    evaluated_events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Assign four deterministic, decision-time-safe blocks to every outer fold."""

    required = {"fold", "symbol", "decision_ts"}
    missing = sorted(required - set(evaluated_events.columns))
    if missing:
        raise CSCVError(f"block population misses columns: {missing}")
    work = evaluated_events.loc[:, ["fold", "symbol", "decision_ts"]].copy()
    work["outer_fold"] = pd.to_numeric(
        work.pop("fold"), errors="raise"
    ).astype(int)
    work["symbol"] = work["symbol"].astype(str)
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    if work.duplicated(["outer_fold", "symbol", "decision_ts"]).any():
        raise CSCVError("block population keys are not unique")
    observed_folds = tuple(sorted(int(item) for item in work["outer_fold"].unique()))
    if observed_folds != OUTER_FOLDS:
        raise CSCVError(f"block population folds changed: {observed_folds}")

    parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for fold in OUTER_FOLDS:
        current = work.loc[work["outer_fold"].eq(fold)].copy()
        group_counts = (
            current.groupby("decision_ts", sort=True, observed=True)
            .size()
            .rename("n_rows")
            .reset_index()
        )
        if len(group_counts) < 4:
            raise CSCVError(f"fold {fold} has fewer than four decision-time groups")
        cumulative = group_counts["n_rows"].cumsum().to_numpy(dtype=np.int64)
        total_rows = int(cumulative[-1])
        boundary_end_indices: list[int] = []
        for block in (1, 2, 3):
            matches = np.flatnonzero(4 * cumulative >= block * total_rows)
            if matches.size == 0:
                raise CSCVError(f"fold {fold} block {block} has no boundary")
            boundary_end_indices.append(int(matches[0]))
        if not all(
            left < right
            for left, right in zip(
                boundary_end_indices, boundary_end_indices[1:]
            )
        ):
            raise CSCVError(
                f"fold {fold} group-safe boundaries would create an empty block"
            )
        group_positions = np.arange(len(group_counts), dtype=np.int64)
        group_blocks = (
            np.searchsorted(
                np.asarray(boundary_end_indices, dtype=np.int64),
                group_positions,
                side="left",
            )
            + 1
        )
        group_to_block = pd.Series(
            group_blocks,
            index=group_counts["decision_ts"],
        )
        current["block"] = (
            current["decision_ts"].map(group_to_block).astype(int)
        )
        if sorted(current["block"].unique().tolist()) != [1, 2, 3, 4]:
            raise CSCVError(f"fold {fold} does not have four nonempty blocks")
        if int(current.groupby("decision_ts")["block"].nunique().max()) != 1:
            raise CSCVError(f"fold {fold} split a decision_ts across blocks")
        for block in range(1, 5):
            selected = current.loc[current["block"].eq(block)]
            audit.append(
                {
                    "fold": fold,
                    "block": block,
                    "column": f"fold_{fold}_block_{block}",
                    "n_rows": int(len(selected)),
                    "n_unique_decision_ts": int(
                        selected["decision_ts"].nunique()
                    ),
                    "start_ts": selected["decision_ts"].min(),
                    "end_ts": selected["decision_ts"].max(),
                    "no_timestamp_split": True,
                }
            )
        parts.append(current)
    labels = pd.concat(parts, ignore_index=True)
    labels["block_column"] = [
        f"fold_{fold}_block_{block}"
        for fold, block in zip(
            labels["outer_fold"], labels["block"], strict=True
        )
    ]
    if len(audit) != 16:
        raise CSCVError(f"expected 16 block audits, observed {len(audit)}")
    return labels, audit


def validate_cscv_predictions(
    predictions: pd.DataFrame,
    evaluated_events: pd.DataFrame,
    *,
    config_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate the exact long prediction panel without using it for selection."""

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise CSCVError(
            "CSCV prediction schema mismatch: "
            f"expected={PREDICTION_COLUMNS}, observed={tuple(predictions.columns)}"
        )
    work = predictions.copy()
    work["config_id"] = work["config_id"].astype(str)
    work["outer_fold"] = pd.to_numeric(
        work["outer_fold"], errors="raise"
    ).astype(int)
    work["symbol"] = work["symbol"].astype(str)
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    work["p_raw"] = pd.to_numeric(work["p_raw"], errors="raise")
    probabilities = work["p_raw"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise CSCVError("CSCV predictions contain invalid probabilities")
    key_columns = ["config_id", "outer_fold", "symbol", "decision_ts"]
    if work.duplicated(key_columns).any():
        raise CSCVError("CSCV prediction keys are not unique")
    expected_ids = [str(item) for item in config_ids]
    if len(expected_ids) != 108 or len(set(expected_ids)) != 108:
        raise CSCVError("CSCV validation requires 108 unique config ids")
    if sorted(work["config_id"].unique().tolist()) != sorted(expected_ids):
        raise CSCVError("CSCV prediction config ids differ from the catalog")

    reference = evaluated_events.loc[
        :, ["fold", "symbol", "decision_ts"]
    ].copy()
    reference["outer_fold"] = pd.to_numeric(
        reference.pop("fold"), errors="raise"
    ).astype(int)
    reference["symbol"] = reference["symbol"].astype(str)
    reference["decision_ts"] = pd.to_datetime(
        reference["decision_ts"], utc=True, errors="raise"
    )
    if reference.duplicated(["outer_fold", "symbol", "decision_ts"]).any():
        raise CSCVError("evaluated reference keys are not unique")
    if tuple(sorted(reference["outer_fold"].unique().tolist())) != OUTER_FOLDS:
        raise CSCVError("evaluated reference must contain folds two through five")
    expected_per_config = int(len(reference))
    counts = work["config_id"].value_counts()
    if not counts.eq(expected_per_config).all():
        raise CSCVError(
            "not every CSCV config covers the complete evaluated population"
        )
    joined = work.merge(
        reference,
        on=["outer_fold", "symbol", "decision_ts"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise CSCVError("CSCV predictions include keys outside evaluated population")
    fold_counts = {
        str(int(fold)): int(count)
        for fold, count in work.groupby("outer_fold").size().sort_index().items()
    }
    expected_fold_counts = {
        str(fold): int(
            108 * reference["outer_fold"].eq(fold).sum()
        )
        for fold in OUTER_FOLDS
    }
    if fold_counts != expected_fold_counts:
        raise CSCVError(
            f"CSCV fold counts changed: {fold_counts} != {expected_fold_counts}"
        )
    return {
        "status": "passed",
        "row_count": int(len(work)),
        "config_count": 108,
        "evaluated_rows_per_config": expected_per_config,
        "fold_row_counts": fold_counts,
        "unique_key_count": int(len(work)),
        "probabilities_finite_and_bounded": True,
    }


def canonicalize_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen long-file row and timestamp ordering."""

    work = predictions.loc[:, list(PREDICTION_COLUMNS)].copy()
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return work.sort_values(
        ["config_id", "outer_fold", "decision_ts", "symbol"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_cscv_matrix(
    predictions: pd.DataFrame,
    evaluated_events: pd.DataFrame,
    *,
    catalog: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute all 108×16 negative weighted-logloss block values."""

    catalog_frame = pd.DataFrame([dict(record) for record in catalog])
    expected_catalog_columns = ("config_id", *CONFIG_KEYS)
    if tuple(catalog_frame.columns) != expected_catalog_columns:
        raise CSCVError(
            "configuration catalog schema mismatch: "
            f"{tuple(catalog_frame.columns)}"
        )
    validation = validate_cscv_predictions(
        predictions,
        evaluated_events,
        config_ids=catalog_frame["config_id"].astype(str).tolist(),
    )
    labels, block_audit = assign_group_safe_blocks(evaluated_events)
    reference = evaluated_events.loc[
        :, ["fold", "symbol", "decision_ts", "meta_y", "tb_uniqueness"]
    ].copy()
    reference["outer_fold"] = pd.to_numeric(
        reference.pop("fold"), errors="raise"
    ).astype(int)
    reference["symbol"] = reference["symbol"].astype(str)
    reference["decision_ts"] = pd.to_datetime(
        reference["decision_ts"], utc=True, errors="raise"
    )
    reference = reference.merge(
        labels.loc[
            :, ["outer_fold", "symbol", "decision_ts", "block_column"]
        ],
        on=["outer_fold", "symbol", "decision_ts"],
        how="left",
        validate="one_to_one",
    )
    joined = predictions.copy()
    joined["decision_ts"] = pd.to_datetime(
        joined["decision_ts"], utc=True, errors="raise"
    )
    joined = joined.merge(
        reference,
        on=["outer_fold", "symbol", "decision_ts"],
        how="left",
        validate="many_to_one",
    )
    if joined[["meta_y", "tb_uniqueness", "block_column"]].isna().any(axis=None):
        raise CSCVError("CSCV matrix join produced missing reference values")
    probability = np.clip(
        joined["p_raw"].to_numpy(dtype=float), 1e-15, 1.0 - 1e-15
    )
    target = joined["meta_y"].to_numpy(dtype=float)
    weight = joined["tb_uniqueness"].to_numpy(dtype=float)
    if (
        not np.isfinite(target).all()
        or not set(np.unique(target).tolist()).issubset({0.0, 1.0})
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
    ):
        raise CSCVError("CSCV target/weight reference values are invalid")
    joined["_weighted_loss"] = (
        -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        * weight
    )
    joined["_weight"] = weight
    grouped = (
        joined.groupby(
            ["config_id", "block_column"], sort=True, observed=True
        )[["_weighted_loss", "_weight"]]
        .sum()
        .reset_index()
    )
    if (grouped["_weight"] <= 0.0).any():
        raise CSCVError("CSCV block contains non-positive total weight")
    grouped["value"] = -grouped["_weighted_loss"] / grouped["_weight"]
    values = grouped.pivot(
        index="config_id", columns="block_column", values="value"
    )
    if values.shape != (108, 16) or values.isna().any(axis=None):
        raise CSCVError(f"CSCV matrix value panel has shape {values.shape}")
    values = values.loc[:, list(BLOCK_COLUMNS)].reset_index()
    matrix = catalog_frame.merge(
        values, on="config_id", how="left", validate="one_to_one", sort=False
    )
    matrix = matrix.loc[:, list(MATRIX_COLUMNS)]
    numeric_values = matrix.loc[:, list(BLOCK_COLUMNS)].to_numpy(dtype=float)
    if (
        matrix.shape != (108, 22)
        or not np.isfinite(numeric_values).all()
        or np.any(numeric_values > 0.0)
    ):
        raise CSCVError("final CSCV matrix failed shape/finite/sign checks")
    return matrix, {
        "status": "passed",
        "prediction_validation": validation,
        "matrix_rows": 108,
        "matrix_block_columns": 16,
        "matrix_semantics": "negative_tb_uniqueness_weighted_raw_logloss",
        "blocks": block_audit,
    }
