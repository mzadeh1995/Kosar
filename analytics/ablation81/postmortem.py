# ==============================================================================
# analytics/ablation81/postmortem.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Decision-inert post-mortem diagnostics for rejected A4 scores."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import calibration

from .metrics import REPORT_SCOPES
from .nested import weighted_raw_logloss


PIPELINE_REJECTION_STATUS = "production_pipeline_rejected_scores"
NOT_EVALUABLE_REASON = "not_evaluable_due_to_production_pipeline_rejection"
SECTION_8_1_NULL_FIELDS = (
    "n_trades",
    "sum_net",
    "mean_net",
    "win_rate",
    "max_drawdown",
    "entry_rate",
    "holding_hours_mean",
    "holding_hours_median",
    "holding_hours_sum",
    "expected_shortfall_5pct",
    "expected_shortfall_tail_n",
    "auc_unweighted_raw",
    "logloss_tb_uniqueness_weighted",
)


class PostMortemError(RuntimeError):
    """Raised when a diagnostic-only post-mortem invariant fails."""


def _normalized_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "decision_ts", "fold", "p_raw"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise PostMortemError(f"prediction frame misses columns: {missing}")
    work = predictions.loc[:, ["symbol", "decision_ts", "fold", "p_raw"]].copy()
    work["symbol"] = work["symbol"].astype(str)
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["p_raw"] = pd.to_numeric(work["p_raw"], errors="raise")
    probability = work["p_raw"].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise PostMortemError("raw post-mortem probabilities are invalid")
    if work.duplicated(["symbol", "decision_ts", "fold"]).any():
        raise PostMortemError("raw post-mortem prediction keys are not unique")
    return work


def attach_raw_predictions(
    evaluated_events: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one seed's raw predictions to the exact evaluated population."""

    required = {
        "symbol",
        "decision_ts",
        "fold",
        "meta_y",
        "tb_uniqueness",
    }
    missing = sorted(required - set(evaluated_events.columns))
    if missing:
        raise PostMortemError(f"evaluated events miss columns: {missing}")
    events = evaluated_events.copy()
    events["symbol"] = events["symbol"].astype(str)
    events["decision_ts"] = pd.to_datetime(
        events["decision_ts"], utc=True, errors="raise"
    )
    events["fold"] = pd.to_numeric(events["fold"], errors="raise").astype(int)
    if events.duplicated(["symbol", "decision_ts", "fold"]).any():
        raise PostMortemError("evaluated event keys are not unique")
    raw = _normalized_predictions(predictions)
    merged = events.merge(
        raw,
        on=["symbol", "decision_ts", "fold"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    membership = {
        str(key): int(value)
        for key, value in merged["_merge"].value_counts(dropna=False).items()
    }
    if (
        membership.get("both", 0) != len(events)
        or membership.get("left_only", 0) != 0
        or membership.get("right_only", 0) != 0
    ):
        raise PostMortemError(
            "raw predictions do not exactly cover evaluated events: "
            f"{membership}"
        )
    lookup = raw.set_index(["symbol", "decision_ts", "fold"])["p_raw"]
    ordered_keys = pd.MultiIndex.from_frame(
        events.loc[:, ["symbol", "decision_ts", "fold"]]
    )
    attached = lookup.reindex(ordered_keys)
    if attached.isna().any() or len(attached) != len(events):
        raise PostMortemError(
            "validated raw predictions could not preserve event order"
        )
    events["p_raw"] = attached.to_numpy(dtype=float)
    return events


def score_dispersion(probabilities: Sequence[float]) -> dict[str, Any]:
    """Return the owner-requested deterministic raw-score summary."""

    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise PostMortemError("score dispersion requires finite nonempty values")
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation_population_ddof_0": float(np.std(values, ddof=0)),
        "unique_value_count": int(np.unique(values).size),
        "row_count": int(values.size),
    }


def raw_fold_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute decision-inert raw AUC, weighted logloss, and dispersion."""

    y = pd.to_numeric(frame["meta_y"], errors="raise").to_numpy(dtype=float)
    p = pd.to_numeric(frame["p_raw"], errors="raise").to_numpy(dtype=float)
    w = pd.to_numeric(
        frame["tb_uniqueness"], errors="raise"
    ).to_numpy(dtype=float)
    if len(y) == 0:
        raise PostMortemError("raw fold diagnostics require rows")
    auc = (
        None
        if np.unique(y).size != 2
        else float(roc_auc_score(y, p))
    )
    return {
        "auc_unweighted_raw": auc,
        "logloss_tb_uniqueness_weighted_raw": weighted_raw_logloss(y, p, w),
        "score_dispersion": score_dispersion(p),
        "decision_role": "diagnostic_only",
    }


def synthetic_platt_diagnostic(
    history: pd.DataFrame,
    *,
    calibrator_name: str,
    training_folds: Sequence[int],
) -> dict[str, Any]:
    """Fit the explicitly synthetic Platt diagnostic and report fit metadata."""

    expected_folds = [int(item) for item in training_folds]
    observed_folds = sorted(
        int(item) for item in pd.to_numeric(
            history["fold"], errors="raise"
        ).unique()
    )
    if observed_folds != expected_folds:
        raise PostMortemError(
            f"{calibrator_name} training folds changed: "
            f"expected={expected_folds}, observed={observed_folds}"
        )
    _, metadata = calibration.fit_score_calibrator(
        history["p_raw"].to_numpy(dtype=float),
        history["meta_y"].to_numpy(dtype=int),
        history["tb_uniqueness"].to_numpy(dtype=float),
        method="platt",
        input_domain=calibration.SCORE_DOMAINS["p_meta"],
        is_real_data_run=False,
    )
    return {
        "calibrator": str(calibrator_name),
        "training_folds": expected_folds,
        "n_train": int(len(history)),
        "requested_method": str(metadata["requested_method"]),
        "method": str(metadata["effective_method"]),
        "fallback": bool(metadata["fallback"]),
        "fallback_reason": metadata["fallback_reason"],
        "coefficient_from_fit_metadata": (
            None
            if metadata["coefficient"] is None
            else float(metadata["coefficient"])
        ),
        "intercept_from_fit_metadata": (
            None
            if metadata["intercept"] is None
            else float(metadata["intercept"])
        ),
        "fit_metadata_source": (
            "calibration.fit_score_calibrator returned metadata; "
            "not calibrator.params"
        ),
        "is_real_data_run": False,
        "allow_fallback_argument": "omitted_default",
        "decision_role": "diagnostic_only",
    }


def build_seed_postmortem(
    evaluated_events: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    seed: int,
    tree_count_by_fold: Mapping[int, int],
) -> dict[str, Any]:
    """Build all per-fold raw and synthetic diagnostics for one frozen seed."""

    attached = attach_raw_predictions(evaluated_events, predictions)
    per_fold: dict[str, Any] = {}
    for fold in (2, 3, 4, 5):
        current = attached.loc[attached["fold"].eq(fold)]
        record: dict[str, Any] = {
            "fold": fold,
            "tree_count": int(tree_count_by_fold[fold]),
            "raw_model_diagnostics": raw_fold_diagnostics(current),
        }
        if fold == 2:
            record["synthetic_platt"] = {
                "calibrator": None,
                "training_folds": [],
                "n_train": 0,
                "status": "not_applicable_first_evaluation_fold",
                "coefficient_from_fit_metadata": None,
                "intercept_from_fit_metadata": None,
                "method": None,
                "fallback": None,
                "fallback_reason": "fold_without_past_evaluated_history",
                "decision_role": "diagnostic_only",
            }
        elif fold == 3 or int(seed) == 42:
            training_folds = list(range(2, fold))
            history = attached.loc[attached["fold"].isin(training_folds)]
            record["synthetic_platt"] = synthetic_platt_diagnostic(
                history,
                calibrator_name=f"C{fold}",
                training_folds=training_folds,
            )
        else:
            record["synthetic_platt"] = {
                "calibrator": f"C{fold}",
                "training_folds": list(range(2, fold)),
                "status": "not_requested_outside_canonical_seed",
                "coefficient_from_fit_metadata": None,
                "intercept_from_fit_metadata": None,
                "method": None,
                "fallback": None,
                "fallback_reason": None,
                "decision_role": "diagnostic_only",
            }
        per_fold[str(fold)] = record
    return {
        "seed": int(seed),
        "role": (
            "canonical_A4" if int(seed) == 42 else "E10_diagnostic_seed_only"
        ),
        "per_fold": per_fold,
        "decision_role": "diagnostic_only",
    }


def weighted_logloss_with_scale_pos_weight(
    targets: Sequence[float],
    probabilities: Sequence[float],
    uniqueness_weights: Sequence[float],
    *,
    scale_pos_weight: float,
) -> float:
    """Compute validation logloss using uniqueness and positive-class scaling."""

    y = np.asarray(targets, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    uniqueness = np.asarray(uniqueness_weights, dtype=float).reshape(-1)
    scale = float(scale_pos_weight)
    if (
        not math.isfinite(scale)
        or scale <= 0.0
        or not (len(y) == len(p) == len(uniqueness))
    ):
        raise PostMortemError("scaled validation logloss inputs are invalid")
    combined = uniqueness * np.where(y == 1.0, scale, 1.0)
    return weighted_raw_logloss(y, p, combined)


def null_section_8_1_metrics() -> dict[str, Any]:
    """Return explicit nulls for an arm with no defined trade set."""

    scope_record = {
        field: None for field in SECTION_8_1_NULL_FIELDS
    }
    scope_record["reason"] = NOT_EVALUABLE_REASON
    return {
        scope: dict(scope_record)
        for scope in REPORT_SCOPES
    }


def rejected_confirmatory_semantics() -> dict[str, Any]:
    """Return the owner-adjudicated measurement and decision fields."""

    return {
        "pipeline_status": PIPELINE_REJECTION_STATUS,
        "construction_status_fa": "ناموفق در مرحله‌ی ساخت",
        "trade_set_defined": False,
        "economic_abstention": False,
        "placebo_test_failed": False,
        "confirmatory_test_status": (
            "not_executable_due_to_production_pipeline_rejection"
        ),
        "confirmatory_metrics_reason": NOT_EVALUABLE_REASON,
        "placebo_sampling_executed": False,
        "placebo_percentile_rank": None,
        "placebo_monte_carlo_p": None,
        "E12": {
            "value": None,
            "reason": NOT_EVALUABLE_REASON,
        },
        "decision_rule_outcome": "case_c",
        "decision_definition_fa": (
            "بازویی که ساخته نشد، به‌تعریف هیچ دروازه‌ای را پاس نمی‌کند"
        ),
        "mandatory_decision_label_fa": (
            "A4 در مرحله‌ی ساخت شکست خورد و هرگز به آزمون تأییدی نرسید — "
            "به مکانیزم ردِ خط لوله، نه شکست از placebo"
        ),
    }
