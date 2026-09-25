# ==============================================================================
# analytics/ablation81/a4.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""A4 walk-forward calibration, replay, metrics, and rowwise hand-off."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

import calibration

from .adjudication import derive_affected_mask_from_pinned
from .metrics import (
    REPORT_SCOPES,
    model_metrics,
    prepare_evaluated_events,
    trade_metrics,
)
from .quarantine import assert_selection_population


A4_OOF_COLUMNS = (
    "symbol",
    "decision_ts",
    "fold",
    "p_raw",
    "p_effective",
    "tau",
    "cal_is_raw",
    "traded_A4",
)


class A4PipelineError(RuntimeError):
    """Raised when A4 calibration/replay cannot be reconstructed exactly."""


def _normalized_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "decision_ts", "fold", "p_raw"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise A4PipelineError(f"A4 raw predictions miss columns: {missing}")
    work = predictions.loc[:, ["symbol", "decision_ts", "fold", "p_raw"]].copy()
    work["symbol"] = work["symbol"].astype(str)
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["p_raw"] = pd.to_numeric(work["p_raw"], errors="raise")
    probabilities = work["p_raw"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise A4PipelineError("A4 raw predictions are invalid")
    if work.duplicated(["symbol", "decision_ts"]).any():
        raise A4PipelineError("A4 raw prediction keys are not unique")
    if sorted(work["fold"].unique().tolist()) != [2, 3, 4, 5]:
        raise A4PipelineError("A4 raw predictions must cover folds two through five")
    return work


def _attach_predictions(
    pinned_events: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    pinned = pinned_events.copy()
    pinned["symbol"] = pinned["symbol"].astype(str)
    pinned["decision_ts"] = pd.to_datetime(
        pinned["decision_ts"], utc=True, errors="raise"
    )
    if pinned.duplicated(["symbol", "decision_ts"]).any():
        raise A4PipelineError("pinned A4 population keys are not unique")
    evaluated = pinned["meta_eval_status"].eq("evaluated")
    expected = pinned.loc[
        evaluated, ["symbol", "decision_ts", "fold"]
    ].copy()
    expected["fold"] = pd.to_numeric(expected["fold"], errors="raise").astype(int)
    raw = _normalized_prediction_frame(predictions)
    merged = expected.merge(
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
        membership.get("both", 0) != int(evaluated.sum())
        or membership.get("left_only", 0) != 0
        or membership.get("right_only", 0) != 0
    ):
        raise A4PipelineError(
            f"A4 raw predictions do not cover evaluated keys: {membership}"
        )
    lookup = raw.set_index(["symbol", "decision_ts"])["p_raw"]
    evaluated_keys = pd.MultiIndex.from_frame(
        pinned.loc[evaluated, ["symbol", "decision_ts"]]
    )
    pinned.loc[evaluated, "p_meta"] = lookup.reindex(
        evaluated_keys
    ).to_numpy(dtype=float)
    if pinned.loc[evaluated, "p_meta"].isna().any():
        raise A4PipelineError("A4 attachment produced missing evaluated p_meta")
    return pinned


def _a4_metrics(
    prepared: pd.DataFrame,
    selected: pd.Series,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope, folds in REPORT_SCOPES.items():
        population = prepared.loc[prepared["fold"].isin(folds)]
        aligned = selected.reindex(population.index)
        if aligned.isna().any():
            raise A4PipelineError(f"A4 mask cannot align to scope {scope}")
        record = trade_metrics(population, aligned.astype(bool))
        record["model_metrics"] = model_metrics(
            population, score_column="p_meta"
        )
        result[scope] = record
    return result


def _fit_record_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    calibrator = record["calibrator"]
    return {
        "fold": int(record["fold"]),
        "score": str(record["score"]),
        "score_name": str(record["score_name"]),
        "method": str(record["method"]),
        "train_folds": [int(item) for item in record["train_folds"]],
        "n_train": int(record["n_train"]),
        "n_eval": int(record["n_eval"]),
        "calibrator_params": dict(calibrator.params),
        "fit_metadata": dict(record["fit_metadata"]),
    }


def run_a4_calibration_pipeline(
    pinned_events: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    *,
    economic_cost: float,
) -> dict[str, Any]:
    """Run the exact production calibration/replay sequence and audit its mask."""

    if abs(float(economic_cost) - float(calibration.CAL_ECONOMIC_COST)) > 1e-15:
        raise A4PipelineError(
            "assembly and calibration economic costs differ: "
            f"{economic_cost} != {calibration.CAL_ECONOMIC_COST}"
        )
    events_a4 = _attach_predictions(pinned_events, raw_predictions)
    evaluated_input = events_a4["meta_eval_status"].eq("evaluated")
    threshold_selection_quarantine: dict[str, Any] = {}
    evaluated_folds = sorted(
        int(item) for item in events_a4.loc[evaluated_input, "fold"].unique()
    )
    for fold in evaluated_folds:
        history = events_a4.loc[
            evaluated_input & events_a4["fold"].lt(fold)
        ]
        threshold_selection_quarantine[f"fold_{fold}"] = (
            assert_selection_population(history)
        )
    walk_forward = calibration.walk_forward_calibrate(
        events_a4, is_real_data_run=True
    )
    calibrated = walk_forward["events"]
    replay = calibration.replay_threshold_mechanism(
        calibrated,
        score="p_meta",
        method="platt",
        fit_records=walk_forward["fit_records"],
    )
    replay_by_fold = {
        int(record["fold"]): record for record in replay["folds"]
    }
    evaluated = calibrated["meta_eval_status"].eq("evaluated")
    effective = pd.Series(np.nan, index=calibrated.index, dtype=float)
    tau = pd.Series(np.nan, index=calibrated.index, dtype=float)
    selected = pd.Series(False, index=calibrated.index, dtype=bool)
    reconstruction: dict[str, Any] = {}
    for fold in sorted(int(item) for item in calibrated.loc[evaluated, "fold"].unique()):
        current = evaluated & calibrated["fold"].eq(fold)
        is_raw_values = calibrated.loc[current, "cal_is_raw"].astype(int)
        if is_raw_values.nunique() != 1:
            raise A4PipelineError(f"fold {fold} mixes raw/calibrated rows")
        is_raw = bool(int(is_raw_values.iloc[0]))
        score_column = "p_meta" if is_raw else "p_meta_cal_platt"
        fold_scores = pd.to_numeric(
            calibrated.loc[current, score_column], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(fold_scores).all():
            raise A4PipelineError(f"fold {fold} effective A4 scores are invalid")
        fold_tau = float(replay_by_fold[fold]["tau"])
        fold_selected = fold_scores >= fold_tau
        effective.loc[current] = fold_scores
        tau.loc[current] = fold_tau
        selected.loc[current] = fold_selected
        net = (
            calibrated.loc[current, "tb_return"].to_numpy(dtype=float)
            - float(economic_cost)
        )
        reconstructed_n = int(fold_selected.sum())
        reconstructed_sum = (
            float(net[fold_selected].sum()) if fold_selected.any() else 0.0
        )
        replay_n = int(replay_by_fold[fold]["n_trades"])
        replay_sum = float(replay_by_fold[fold]["sum_net"])
        if reconstructed_n != replay_n or abs(reconstructed_sum - replay_sum) > 1e-9:
            raise A4PipelineError(
                f"fold {fold} replay reconstruction mismatch: "
                f"n={reconstructed_n}/{replay_n}, "
                f"sum={reconstructed_sum}/{replay_sum}"
            )
        reconstruction[f"fold_{fold}"] = {
            "fold": fold,
            "cal_is_raw": is_raw,
            "score_column": score_column,
            "tau": fold_tau,
            "reconstructed_n_trades": reconstructed_n,
            "replay_n_trades": replay_n,
            "reconstructed_sum_net": reconstructed_sum,
            "replay_sum_net": replay_sum,
            "absolute_sum_net_difference": abs(reconstructed_sum - replay_sum),
            "status": "passed",
        }
    if effective.loc[evaluated].isna().any() or tau.loc[evaluated].isna().any():
        raise A4PipelineError("A4 reconstruction left evaluated rows unassigned")
    calibrated = calibrated.copy()
    calibrated["traded_meta"] = False
    calibrated.loc[evaluated, "traded_meta"] = selected.loc[evaluated]
    prepared = prepare_evaluated_events(
        calibrated, economic_cost=float(economic_cost)
    )
    selected_prepared = prepared["traded_meta"].astype(bool)
    metrics = _a4_metrics(prepared, selected_prepared)

    affected_full, affected_audit = derive_affected_mask_from_pinned(calibrated)
    affected_traded = affected_full & calibrated["traded_meta"].astype(bool)
    affected_sum_net = float(
        (
            calibrated.loc[affected_traded, "tb_return"].astype(float)
            - float(economic_cost)
        ).sum()
    )
    overlap = {
        "affected_population_rows": int(affected_full.sum()),
        "n_trades": int(affected_traded.sum()),
        "sum_net": affected_sum_net,
        "affected_mask_audit": affected_audit,
    }
    oof = pd.DataFrame(
        {
            "symbol": calibrated.loc[evaluated, "symbol"].astype(str).to_numpy(),
            "decision_ts": pd.to_datetime(
                calibrated.loc[evaluated, "decision_ts"],
                utc=True,
                errors="raise",
            ).to_numpy(),
            "fold": calibrated.loc[evaluated, "fold"].astype(int).to_numpy(),
            "p_raw": calibrated.loc[evaluated, "p_meta"].to_numpy(dtype=float),
            "p_effective": effective.loc[evaluated].to_numpy(dtype=float),
            "tau": tau.loc[evaluated].to_numpy(dtype=float),
            "cal_is_raw": calibrated.loc[evaluated, "cal_is_raw"]
            .astype(int)
            .to_numpy(),
            "traded_A4": selected.loc[evaluated].astype(bool).to_numpy(),
        }
    )
    oof = oof.sort_values(
        ["fold", "decision_ts", "symbol"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if tuple(oof.columns) != A4_OOF_COLUMNS or len(oof) != int(evaluated.sum()):
        raise A4PipelineError("A4 rowwise hand-off schema/count failed")
    return {
        "oof": oof,
        "metrics": metrics,
        "funding_affected_fold5_overlap": overlap,
        "calibration": {
            "call_order": [
                "calibration.walk_forward_calibrate",
                "calibration.replay_threshold_mechanism",
            ],
            "is_real_data_run": True,
            "joblib_artifacts_deserialized_or_scored": False,
            "threshold_selection_quarantine": threshold_selection_quarantine,
            "fold_records": list(walk_forward["fold_records"]),
            "fit_records": [
                _fit_record_summary(record)
                for record in walk_forward["fit_records"]
            ],
        },
        "replay": replay,
        "reconstruction": {
            "absolute_tolerance_sum_net": 1e-9,
            "n_trades_exact": True,
            "folds": reconstruction,
            "status": "passed",
        },
    }
