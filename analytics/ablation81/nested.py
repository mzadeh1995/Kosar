# ==============================================================================
# analytics/ablation81/nested.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Pinned nested-CV search and deterministic winner refits for A4."""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import meta_model

from .quarantine import (
    assert_outer_evaluation_population,
    assert_outer_training_population,
)


FROZEN_THREAD_COUNT = 3
OUTER_FOLDS = (2, 3, 4, 5)
SEED_STABILITY_SEEDS = (42, 0, 1, 2, 3)
SEARCH_ITERATIONS = 1_500
EARLY_STOPPING_ROUNDS = 50
SELECTION_TIE_TOLERANCE = 1e-12
DEPTHS = (3, 4, 5)
L2_VALUES = (3, 10, 30)
LEARNING_RATES = (0.01, 0.03, 0.05)
BOOSTING_TYPES = ("Ordered", "Plain")
BOOTSTRAPS = ("default", "bernoulli_subsample_0.8")
CONFIG_KEYS = (
    "depth",
    "l2_leaf_reg",
    "learning_rate",
    "boosting_type",
    "bootstrap",
)
LEDGER_REQUIRED_KEYS = {
    "config_id",
    "config",
    "inner_validation_weighted_raw_logloss",
    "tree_count",
    "best_iteration",
    "scale_pos_weight",
    "weighted_positive_sum",
    "weighted_negative_sum",
    "inner_fit_cpu_seconds",
    "inner_fit_wall_seconds",
}
LEDGER_FORBIDDEN_KEYS = {
    "p_raw",
    "outer_logloss",
    "outer_metric",
    "outer_predictions",
}


class NestedSearchError(RuntimeError):
    """Raised on a nested-CV contract violation."""


def build_config_catalog() -> list[dict[str, Any]]:
    """Return all and only the 108 frozen configurations."""

    configs: list[dict[str, Any]] = []
    for position, values in enumerate(
        itertools.product(
            DEPTHS,
            L2_VALUES,
            LEARNING_RATES,
            BOOSTING_TYPES,
            BOOTSTRAPS,
        )
    ):
        config = dict(zip(CONFIG_KEYS, values, strict=True))
        configs.append({"config_id": f"cfg_{position:03d}", **config})
    identities = {
        tuple(record[key] for key in CONFIG_KEYS) for record in configs
    }
    if len(configs) != 108 or len(identities) != 108:
        raise NestedSearchError("configuration catalog is not exactly 108 unique rows")
    return configs


def simplicity_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the pinned deterministic tie-break key."""

    bootstrap = str(config["bootstrap"])
    boosting = str(config["boosting_type"])
    if bootstrap not in BOOTSTRAPS or boosting not in BOOSTING_TYPES:
        raise NestedSearchError("configuration contains an unknown categorical value")
    return (
        int(config["depth"]),
        -float(config["l2_leaf_reg"]),
        float(config["learning_rate"]),
        0 if boosting == "Ordered" else 1,
        0 if bootstrap == "default" else 1,
    )


def weighted_raw_logloss(
    targets: Sequence[float],
    probabilities: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Compute the contract's uniqueness-weighted raw-score logloss."""

    y = np.asarray(targets, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if not (len(y) == len(p) == len(w)) or len(y) == 0:
        raise NestedSearchError("weighted logloss inputs must be nonempty and aligned")
    if (
        not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or not np.isfinite(w).all()
        or np.any(w < 0.0)
        or float(w.sum()) <= 0.0
        or not set(np.unique(y).tolist()).issubset({0.0, 1.0})
    ):
        raise NestedSearchError("weighted logloss inputs are invalid")
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    losses = -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))
    return float(np.average(losses, weights=w))


def inner_temporal_split_and_purge(
    selection_events: pd.DataFrame,
    *,
    outer_fold: int,
    is_real_data_run: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Take the last floor(25%) rows as validation and purge its past."""

    past_gate = assert_outer_training_population(
        selection_events, outer_fold=int(outer_fold)
    )
    if selection_events.empty:
        raise NestedSearchError("outer history cannot be empty")
    ordered = selection_events.sort_values(
        ["decision_ts", "symbol"], ascending=[True, True], kind="mergesort"
    )
    n_total = int(len(ordered))
    n_validation = int(math.floor(0.25 * n_total))
    if n_validation <= 0 or n_validation >= n_total:
        raise NestedSearchError(
            f"inner split is not viable: total={n_total}, validation={n_validation}"
        )
    split_position = n_total - n_validation
    pre_purge_train = ordered.iloc[:split_position].copy()
    validation = ordered.iloc[split_position:].copy()
    validation_start = pd.to_datetime(
        validation["decision_ts"], utc=True, errors="raise"
    ).min()
    purged_train, purge_metadata = meta_model.purge_training_events(
        pre_purge_train,
        validation_start,
        is_real_data_run=bool(is_real_data_run),
    )
    if purged_train.empty:
        raise NestedSearchError("inner purge removed the complete training frame")
    if set(purged_train.index) & set(validation.index):
        raise NestedSearchError("inner train and validation indices overlap")
    exits = pd.to_datetime(
        purged_train["tb_exit_index"], utc=True, format="mixed", errors="raise"
    )
    purge_boundary = validation_start - pd.Timedelta(hours=96)
    if not ((exits + pd.Timedelta(hours=4)) < purge_boundary).all():
        raise NestedSearchError("inner purge/embargo strict inequality failed")
    metadata = {
        "status": "passed",
        "outer_fold": int(outer_fold),
        "selection_population_gate": past_gate,
        "sort": ["decision_ts ascending", "symbol ascending", "mergesort"],
        "total_history_rows": n_total,
        "inner_validation_floor_25pct_rows": n_validation,
        "inner_pre_purge_train_rows": int(len(pre_purge_train)),
        "inner_post_purge_train_rows": int(len(purged_train)),
        "inner_validation_rows": int(len(validation)),
        "inner_validation_start": validation_start,
        "purge": purge_metadata,
        "strict_purge_assertion": True,
    }
    return purged_train, validation, metadata


def _model_params(
    config: Mapping[str, Any],
    *,
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    loss_function = meta_model.META_CATBOOST_PARAMS["loss_function"]
    if loss_function != "Logloss":
        raise NestedSearchError(
            f"production loss function changed: {loss_function!r}"
        )
    params: dict[str, Any] = {
        "loss_function": loss_function,
        "iterations": int(iterations),
        "learning_rate": float(config["learning_rate"]),
        "depth": int(config["depth"]),
        "l2_leaf_reg": float(config["l2_leaf_reg"]),
        "boosting_type": str(config["boosting_type"]),
        "random_seed": int(random_seed),
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "CPU",
        "thread_count": FROZEN_THREAD_COUNT,
    }
    bootstrap = str(config["bootstrap"])
    if bootstrap == "bernoulli_subsample_0.8":
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = 0.8
    elif bootstrap != "default":
        raise NestedSearchError(f"unknown bootstrap option: {bootstrap}")
    if bootstrap == "default" and (
        "bootstrap_type" in params or "subsample" in params
    ):
        raise NestedSearchError("default bootstrap must omit all bootstrap parameters")
    return params


def _pool(
    X: pd.DataFrame,
    events: pd.DataFrame,
    *,
    categorical_features: Sequence[str],
) -> Pool:
    return Pool(
        data=X.loc[events.index],
        label=events["meta_y"].astype(int),
        weight=events["tb_uniqueness"].astype(float),
        cat_features=list(categorical_features),
    )


def _predict(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(X, thread_count=FROZEN_THREAD_COUNT)[:, 1],
        dtype=float,
    )
    if (
        probabilities.shape != (len(X),)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise NestedSearchError("CatBoost produced invalid raw probabilities")
    return probabilities


def select_winning_config(
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select solely from inner-validation records, never outer predictions."""

    if not ledger:
        raise NestedSearchError("selection ledger cannot be empty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(ledger):
        keys = set(raw)
        missing = sorted(LEDGER_REQUIRED_KEYS - keys)
        contaminated = sorted(
            LEDGER_FORBIDDEN_KEYS & keys
            | {
                key
                for key in keys
                if key.startswith("outer_") or key.startswith("cscv_")
            }
        )
        if missing:
            raise NestedSearchError(
                f"selection ledger row {position} misses keys: {missing}"
            )
        if contaminated:
            raise NestedSearchError(
                f"selection ledger row {position} is outer-contaminated: "
                f"{contaminated}"
            )
        config_id = str(raw["config_id"])
        if config_id in seen:
            raise NestedSearchError(f"duplicate selection config_id: {config_id}")
        seen.add(config_id)
        loss = float(raw["inner_validation_weighted_raw_logloss"])
        if not math.isfinite(loss):
            raise NestedSearchError(f"non-finite selection loss for {config_id}")
        config = dict(raw["config"])
        if set(config) != set(CONFIG_KEYS):
            raise NestedSearchError(f"malformed configuration for {config_id}")
        normalized.append(dict(raw))
    minimum = min(
        float(record["inner_validation_weighted_raw_logloss"])
        for record in normalized
    )
    tied = [
        record
        for record in normalized
        if abs(
            float(record["inner_validation_weighted_raw_logloss"]) - minimum
        )
        <= SELECTION_TIE_TOLERANCE
    ]
    winner = min(tied, key=lambda record: simplicity_key(record["config"]))
    return {
        "config_id": str(winner["config_id"]),
        "config": dict(winner["config"]),
        "inner_validation_weighted_raw_logloss": float(
            winner["inner_validation_weighted_raw_logloss"]
        ),
        "tree_count": int(winner["tree_count"]),
        "best_iteration": int(winner["best_iteration"]),
        "minimum_observed_loss": float(minimum),
        "tie_candidate_count": int(len(tied)),
        "tie_tolerance": SELECTION_TIE_TOLERANCE,
        "simplicity_key": list(simplicity_key(winner["config"])),
    }


def search_outer_fold(
    selection_events: pd.DataFrame,
    outer_events: pd.DataFrame,
    X: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    outer_fold: int,
    configs: Sequence[Mapping[str, Any]],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fit every inner model, ledger its loss, and separately sink CSCV scores."""

    outer = int(outer_fold)
    assert_outer_training_population(selection_events, outer_fold=outer)
    outer_gate = assert_outer_evaluation_population(
        outer_events, outer_fold=outer
    )
    inner_train, inner_validation, split_audit = inner_temporal_split_and_purge(
        selection_events,
        outer_fold=outer,
        is_real_data_run=True,
    )
    catalog = [dict(config) for config in configs]
    if len(catalog) != 108 or len({item["config_id"] for item in catalog}) != 108:
        raise NestedSearchError("outer search requires exactly 108 unique configs")

    train_pool = _pool(
        X, inner_train, categorical_features=categorical_features
    )
    validation_pool = _pool(
        X, inner_validation, categorical_features=categorical_features
    )
    outer_X = X.loc[outer_events.index, list(feature_columns)]
    scale_pos_weight, positive, negative = meta_model.weighted_scale_pos_weight(
        inner_train["meta_y"], inner_train["tb_uniqueness"]
    )
    ledger: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    resolved_params_by_config: dict[str, dict[str, Any]] = {}
    search_cpu_start = time.process_time()
    search_wall_start = time.perf_counter()
    for position, catalog_record in enumerate(catalog, start=1):
        config_id = str(catalog_record["config_id"])
        config = {key: catalog_record[key] for key in CONFIG_KEYS}
        params = _model_params(config, iterations=SEARCH_ITERATIONS, random_seed=42)
        params["scale_pos_weight"] = scale_pos_weight
        fit_cpu_start = time.process_time()
        fit_wall_start = time.perf_counter()
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=validation_pool,
            use_best_model=True,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        fit_cpu = time.process_time() - fit_cpu_start
        fit_wall = time.perf_counter() - fit_wall_start
        validation_p = _predict(
            model, X.loc[inner_validation.index, list(feature_columns)]
        )
        inner_loss = weighted_raw_logloss(
            inner_validation["meta_y"],
            validation_p,
            inner_validation["tb_uniqueness"],
        )
        best_iteration = int(model.get_best_iteration())
        tree_count = int(model.tree_count_)
        if best_iteration < 0 or tree_count != best_iteration + 1:
            raise NestedSearchError(
                f"{config_id} best iteration/tree count mismatch: "
                f"{best_iteration}/{tree_count}"
            )
        if tree_count < 1 or tree_count > SEARCH_ITERATIONS:
            raise NestedSearchError(f"{config_id} invalid tree count: {tree_count}")
        ledger.append(
            {
                "config_id": config_id,
                "config": config,
                "inner_validation_weighted_raw_logloss": inner_loss,
                "tree_count": tree_count,
                "best_iteration": best_iteration,
                "scale_pos_weight": float(scale_pos_weight),
                "weighted_positive_sum": float(positive),
                "weighted_negative_sum": float(negative),
                "inner_fit_cpu_seconds": float(fit_cpu),
                "inner_fit_wall_seconds": float(fit_wall),
            }
        )
        resolved_params_by_config[config_id] = dict(model.get_all_params())

        outer_p = _predict(model, outer_X)
        prediction_parts.append(
            pd.DataFrame(
                {
                    "config_id": config_id,
                    "outer_fold": outer,
                    "symbol": outer_events["symbol"].astype(str).to_numpy(),
                    "decision_ts": pd.to_datetime(
                        outer_events["decision_ts"], utc=True, errors="raise"
                    ).to_numpy(),
                    "p_raw": outer_p,
                }
            )
        )
        if progress is not None and (
            position == 1 or position % 12 == 0 or position == len(catalog)
        ):
            progress(
                {
                    "event": "phase3_search_progress",
                    "outer_fold": outer,
                    "completed_configs": position,
                    "total_configs": len(catalog),
                    "latest_config_id": config_id,
                    "latest_inner_logloss": inner_loss,
                    "elapsed_wall_seconds": time.perf_counter()
                    - search_wall_start,
                }
            )
    winner = select_winning_config(ledger)
    search_cpu = time.process_time() - search_cpu_start
    search_wall = time.perf_counter() - search_wall_start
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_prediction_rows = len(catalog) * len(outer_events)
    if len(predictions) != expected_prediction_rows:
        raise NestedSearchError(
            "CSCV prediction sink row count mismatch: "
            f"expected={expected_prediction_rows}, observed={len(predictions)}"
        )
    return {
        "outer_fold": outer,
        "winner": winner,
        "winner_inner_resolved_params": resolved_params_by_config[
            winner["config_id"]
        ],
        "selection_ledger": ledger,
        "cscv_predictions": predictions,
        "inner_split": split_audit,
        "outer_evaluation_gate": outer_gate,
        "timing": {
            "search_cpu_seconds": float(search_cpu),
            "search_wall_seconds": float(search_wall),
        },
        "selection_input_schema_excludes_outer_predictions": True,
    }


def refit_winner_for_seed(
    selection_events: pd.DataFrame,
    outer_events: pd.DataFrame,
    X: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    outer_fold: int,
    config: Mapping[str, Any],
    tree_count: int,
    seed: int,
) -> dict[str, Any]:
    """Refit one frozen winner on its purged full history and predict its outer fold."""

    outer = int(outer_fold)
    history_gate = assert_outer_training_population(
        selection_events, outer_fold=outer
    )
    evaluation_gate = assert_outer_evaluation_population(
        outer_events, outer_fold=outer
    )
    outer_start = pd.to_datetime(
        outer_events["decision_ts"], utc=True, errors="raise"
    ).min()
    purged, purge_metadata = meta_model.purge_training_events(
        selection_events,
        outer_start,
        is_real_data_run=True,
    )
    if purged.empty:
        raise NestedSearchError("winner refit purge removed all history")
    scale_pos_weight, positive, negative = meta_model.weighted_scale_pos_weight(
        purged["meta_y"], purged["tb_uniqueness"]
    )
    params = _model_params(
        config, iterations=int(tree_count), random_seed=int(seed)
    )
    params["scale_pos_weight"] = scale_pos_weight
    train_pool = _pool(X, purged, categorical_features=categorical_features)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    model = CatBoostClassifier(**params)
    model.fit(train_pool, verbose=False)
    fit_cpu = time.process_time() - cpu_start
    fit_wall = time.perf_counter() - wall_start
    if int(model.tree_count_) != int(tree_count):
        raise NestedSearchError(
            f"winner refit tree count changed: {model.tree_count_} != {tree_count}"
        )
    probabilities = _predict(
        model, X.loc[outer_events.index, list(feature_columns)]
    )
    return {
        "outer_fold": outer,
        "seed": int(seed),
        "probabilities": probabilities,
        "metadata": {
            "history_gate": history_gate,
            "outer_evaluation_gate": evaluation_gate,
            "pre_purge_history_rows": int(len(selection_events)),
            "post_purge_history_rows": int(len(purged)),
            "outer_start": outer_start,
            "purge": purge_metadata,
            "tree_count": int(tree_count),
            "config": dict(config),
            "scale_pos_weight": float(scale_pos_weight),
            "weighted_positive_sum": float(positive),
            "weighted_negative_sum": float(negative),
            "fit_cpu_seconds": float(fit_cpu),
            "fit_wall_seconds": float(fit_wall),
            "resolved_model_params": dict(model.get_all_params()),
            "eval_set_used": False,
            "early_stopping_used": False,
            "use_best_model_used": False,
        },
    }
