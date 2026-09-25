# ==============================================================================
# analytics/ablation81/phase3_complete.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Complete phase three after the exact owner-adjudicated Platt rejection."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import calibration
import meta_model

from .a4 import run_a4_calibration_pipeline
from .adjudication_platt import (
    EXPECTED_ADJUDICATED_SIGNATURE,
    EXPECTED_PHASE3_STOP_IDENTITY,
    assert_allowed_family_rejection_code,
    assert_exact_authorized_rejection,
    verify_owner_adjudication_platt,
)
from .cscv import (
    BLOCK_COLUMNS,
    MATRIX_COLUMNS,
    PREDICTION_COLUMNS,
    build_cscv_matrix,
    canonicalize_prediction_rows,
)
from .integrity import IntegrityError, file_identity, load_calibrated_events, write_json_once
from .metrics import ES_SEMANTICS, MAXDD_SEMANTICS
from .nested import (
    CONFIG_KEYS,
    EARLY_STOPPING_ROUNDS,
    FROZEN_THREAD_COUNT,
    OUTER_FOLDS,
    SEARCH_ITERATIONS,
    SEED_STABILITY_SEEDS,
    SELECTION_TIE_TOLERANCE,
    _model_params,
    _pool,
    _predict,
    build_config_catalog,
    inner_temporal_split_and_purge,
    refit_winner_for_seed,
    simplicity_key,
    weighted_raw_logloss,
)
from .phase3 import (
    DEVIATIONS,
    OUTPUT_DIR,
    PHASE_REPORT_PATH,
    PHASE_STOP_REPORT_PATH,
    PROJECT_ROOT,
    _append_task_log,
    _identity_subset,
    _json_ready,
    _progress,
    _run_pytest,
    _run_reentry_protocol,
    _utc_now,
    _verify_forbidden_owner_rule,
    _verify_prior_frozen_outputs,
)
from .phase3_gates import run_pretraining_gates
from .postmortem import (
    NOT_EVALUABLE_REASON,
    PIPELINE_REJECTION_STATUS,
    build_seed_postmortem,
    null_section_8_1_metrics,
    rejected_confirmatory_semantics,
    weighted_logloss_with_scale_pos_weight,
)
from .quarantine import assert_outer_evaluation_population


SELECTED_CONFIGS_PATH = OUTPUT_DIR / "selected_configs.json"
SEED_STABILITY_PATH = OUTPUT_DIR / "seed_stability_4h.json"
CSCV_PREDICTIONS_PATH = OUTPUT_DIR / "cscv_predictions_4h.csv"
CSCV_MATRIX_PATH = OUTPUT_DIR / "cscv_matrix_4h.csv"
A4_OOF_PATH = OUTPUT_DIR / "a4_oof_4h.csv"

SCIENTIFIC_DESTINATIONS = {
    "selected_configs": SELECTED_CONFIGS_PATH,
    "seed_stability": SEED_STABILITY_PATH,
    "cscv_predictions": CSCV_PREDICTIONS_PATH,
    "cscv_matrix": CSCV_MATRIX_PATH,
    "a4_oof": A4_OOF_PATH,
}
A4_REJECTED_OOF_COLUMNS = (
    "symbol",
    "decision_ts",
    "fold",
    "p_raw",
    "p_effective",
    "tau",
    "traded_A4",
    "pipeline_status",
)
EXPECTED_EVALUATED_ROWS = 9_048
EXPECTED_POPULATION_ROWS = 11_029
EXPECTED_SEARCH_FIT_COUNT = 432
EXPECTED_REFIT_COUNT = 20
EXPECTED_CANONICAL_EXCEPTION_MESSAGE = (
    "Platt coefficient must be positive; observed -0.000843584342876648"
)
HISTORICAL_A2_PLATT_COEFFICIENT = 0.05452992058508968
PINNED_PRODUCTION_REJECTION_SENTENCE_FA = (
    "خط لوله‌ی تولید طبق طراحی صریح خودش این امتیازها را رد کرد و همین رد، "
    "مانع ورود یک کالیبره‌گر با شیب منفی (وارونه‌کننده‌ی رتبه‌بندی) به "
    "نسخه‌ی ۵۰ شد"
)
PINNED_LN2_CAVEAT_FA = (
    "۰.۶۲۳۹ زیان بهترین پیش‌بین ثابت زیر همین وزن‌دهی است، ولی مبنای سنجش "
    "یادگیری ln2=۰.۶۹۳۱۴۷ است، چون خروجی مدل به‌واسطه‌ی scale_pos_weight "
    "حول ۰.۵ متوازن می‌شود و فاصله تا ۰.۶۲۳۹ اثر همان توازن است نه نشانه‌ی "
    "بدتربودن از بدیهی"
)
PINNED_A5_A6_CAVEAT_FA = (
    "پیکربندی‌های منجمد A4 در foldهای ۲ و ۴ مدل تک‌درختی‌اند؛ تفسیر حذف "
    "ویژگی از چنین مدلی محدود است"
)
PINNED_DECISION_LABEL_FA = (
    "A4 در مرحله‌ی ساخت شکست خورد و هرگز به آزمون تأییدی نرسید — "
    "به مکانیزم ردِ خط لوله، نه شکست از placebo"
)


class Phase3CompletionError(IntegrityError):
    """Raised when the adjudicated phase-three completion changes semantics."""


def _identity_pair(path: Path) -> dict[str, Any]:
    observed = file_identity(path)
    return {
        "size_bytes": int(observed["size_bytes"]),
        "sha256": str(observed["sha256"]),
    }


def _load_stop_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity_pair(PHASE_STOP_REPORT_PATH)
    if identity != EXPECTED_PHASE3_STOP_IDENTITY:
        raise Phase3CompletionError(
            "phase3 stop authority identity changed: "
            f"expected={EXPECTED_PHASE3_STOP_IDENTITY}, observed={identity}"
        )
    try:
        stop_report = json.loads(PHASE_STOP_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3CompletionError(f"cannot read phase3 stop authority: {exc}") from exc
    stop = stop_report.get("stop", {})
    expected_stop = {
        "canonical_seed": 42,
        "calibration_training_folds": [2],
        "score": "p_meta",
        "method": "platt",
        "observed_platt_coefficient": -0.000843584342876648,
        "exception_class": "calibration.CalibrationError",
        "exception_code": "nonpositive_platt_coefficient",
        "message": EXPECTED_CANONICAL_EXCEPTION_MESSAGE,
    }
    for key, expected in expected_stop.items():
        if stop.get(key) != expected:
            raise Phase3CompletionError(
                f"phase3 stop signature changed at {key}: "
                f"expected={expected!r}, observed={stop.get(key)!r}"
            )
    completed = stop_report.get("completed_before_stop", {})
    winners = completed.get("selected_winners_in_memory")
    if (
        not isinstance(winners, dict)
        or sorted(winners) != ["2", "3", "4", "5"]
        or int(completed.get("search_fit_count", -1)) != EXPECTED_SEARCH_FIT_COUNT
        or int(completed.get("winner_refit_count", -1)) != EXPECTED_REFIT_COUNT
    ):
        raise Phase3CompletionError("phase3 stop winner authority is incomplete")
    source_records: list[dict[str, Any]] = []
    source_identities = stop_report.get("source_identities", {})
    if not isinstance(source_identities, dict) or len(source_identities) != 7:
        raise Phase3CompletionError("phase3 stop source-identity set changed")
    for relative, expected in sorted(source_identities.items()):
        observed = _identity_pair(PROJECT_ROOT / relative)
        expected_pair = {
            "size_bytes": int(expected["size_bytes"]),
            "sha256": str(expected["sha256"]),
        }
        if observed != expected_pair:
            raise Phase3CompletionError(
                f"frozen stopped-run source changed: {relative}; "
                f"expected={expected_pair}, observed={observed}"
            )
        source_records.append(
            {"path": relative, **observed, "status": "passed"}
        )
    return stop_report, {
        "status": "passed",
        "phase3_stop_report": {
            "path": PHASE_STOP_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            **identity,
        },
        "exact_stop_signature": expected_stop,
        "frozen_source_count": len(source_records),
        "frozen_sources": source_records,
        "winner_authority": (
            "hash-gated phase3_stop_report.json; reproduction has no "
            "selection authority"
        ),
    }


def _assert_destinations_absent() -> dict[str, Any]:
    paths = {**SCIENTIFIC_DESTINATIONS, "phase3_report": PHASE_REPORT_PATH}
    observed = {
        key: not path.exists()
        for key, path in paths.items()
    }
    if not all(observed.values()):
        existing = [key for key, absent in observed.items() if not absent]
        raise Phase3CompletionError(
            f"refusing to overwrite frozen phase3 completion outputs: {existing}"
        )
    return {
        "status": "passed",
        "all_destinations_absent": True,
        "paths": {
            key: path.relative_to(PROJECT_ROOT).as_posix()
            for key, path in paths.items()
        },
    }


def _frozen_winner(
    stop_report: Mapping[str, Any],
    outer_fold: int,
) -> dict[str, Any]:
    winner = dict(
        stop_report["completed_before_stop"]["selected_winners_in_memory"][
            str(int(outer_fold))
        ]
    )
    if set(winner["config"]) != set(CONFIG_KEYS):
        raise Phase3CompletionError(
            f"frozen winner fold {outer_fold} configuration schema changed"
        )
    return winner


def _reproduce_outer_fold_for_cscv(
    selection_events: pd.DataFrame,
    outer_events: pd.DataFrame,
    X: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    outer_fold: int,
    catalog: Sequence[Mapping[str, Any]],
    frozen_winner: Mapping[str, Any],
) -> dict[str, Any]:
    """Reproduce CSCV fits while treating the stopped-run winner as immutable."""

    outer = int(outer_fold)
    outer_gate = assert_outer_evaluation_population(
        outer_events, outer_fold=outer
    )
    inner_train, inner_validation, split_audit = inner_temporal_split_and_purge(
        selection_events,
        outer_fold=outer,
        is_real_data_run=True,
    )
    records = [dict(item) for item in catalog]
    if len(records) != 108 or len({item["config_id"] for item in records}) != 108:
        raise Phase3CompletionError("CSCV reproduction requires the exact catalog")
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
    winner_capture: dict[str, Any] | None = None
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for position, catalog_record in enumerate(records, start=1):
        config_id = str(catalog_record["config_id"])
        config = {key: catalog_record[key] for key in CONFIG_KEYS}
        params = _model_params(
            config,
            iterations=SEARCH_ITERATIONS,
            random_seed=42,
        )
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
            model,
            X.loc[inner_validation.index, list(feature_columns)],
        )
        raw_loss = weighted_raw_logloss(
            inner_validation["meta_y"],
            validation_p,
            inner_validation["tb_uniqueness"],
        )
        best_iteration = int(model.get_best_iteration())
        tree_count = int(model.tree_count_)
        if best_iteration < 0 or tree_count != best_iteration + 1:
            raise Phase3CompletionError(
                f"{config_id} best_iteration/tree_count mismatch"
            )
        row = {
            "config_id": config_id,
            "config": config,
            "inner_validation_weighted_raw_logloss": raw_loss,
            "tree_count": tree_count,
            "best_iteration": best_iteration,
            "scale_pos_weight": float(scale_pos_weight),
            "weighted_positive_sum": float(positive),
            "weighted_negative_sum": float(negative),
            "inner_fit_cpu_seconds": float(fit_cpu),
            "inner_fit_wall_seconds": float(fit_wall),
        }
        ledger.append(row)
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
        if config_id == str(frozen_winner["config_id"]):
            scaled_loss = weighted_logloss_with_scale_pos_weight(
                inner_validation["meta_y"],
                validation_p,
                inner_validation["tb_uniqueness"],
                scale_pos_weight=float(scale_pos_weight),
            )
            best_scores = model.get_best_score()
            catboost_validation_loss = (
                None
                if "validation" not in best_scores
                or "Logloss" not in best_scores["validation"]
                else float(best_scores["validation"]["Logloss"])
            )
            winner_capture = {
                "ledger_record": row,
                "resolved_model_params_get_all_params": dict(
                    model.get_all_params()
                ),
                "inner_objective_post_mortem": {
                    "same_raw_validation_probabilities": True,
                    "validation_rows": int(len(inner_validation)),
                    "tb_uniqueness_weighted_raw_logloss": raw_loss,
                    "tb_uniqueness_times_scale_pos_weight_for_positive_logloss": (
                        scaled_loss
                    ),
                    "scale_pos_weight": float(scale_pos_weight),
                    "catboost_reported_validation_logloss": (
                        catboost_validation_loss
                    ),
                    "manual_scaled_minus_catboost_reported": (
                        None
                        if catboost_validation_loss is None
                        else float(scaled_loss - catboost_validation_loss)
                    ),
                    "decision_role": "diagnostic_only",
                    "extra_fit_performed": False,
                },
            }
        if position == 1 or position % 12 == 0 or position == len(records):
            _progress(
                {
                    "event": "phase3_cscv_reproduction_progress",
                    "outer_fold": outer,
                    "completed_configs": position,
                    "total_configs": len(records),
                    "latest_config_id": config_id,
                    "elapsed_wall_seconds": (
                        time.perf_counter() - wall_start
                    ),
                    "selection_authority": False,
                }
            )
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    if winner_capture is None:
        raise Phase3CompletionError(
            f"frozen winner fold {outer} was not present in catalog"
        )
    captured = winner_capture["ledger_record"]
    exact_keys = ("config_id", "config", "tree_count", "best_iteration")
    for key in exact_keys:
        if captured[key] != frozen_winner[key]:
            raise Phase3CompletionError(
                f"reproduced frozen winner fold {outer} differs at {key}: "
                f"{captured[key]!r} != {frozen_winner[key]!r}"
            )
    expected_loss = float(
        frozen_winner["inner_validation_weighted_raw_logloss"]
    )
    reproduced_loss = float(
        captured["inner_validation_weighted_raw_logloss"]
    )
    loss_difference = reproduced_loss - expected_loss
    if abs(loss_difference) > 1e-15:
        raise Phase3CompletionError(
            f"reproduced frozen winner fold {outer} loss changed by "
            f"{loss_difference}"
        )
    minimum = min(
        float(item["inner_validation_weighted_raw_logloss"])
        for item in ledger
    )
    tied = [
        item
        for item in ledger
        if abs(
            float(item["inner_validation_weighted_raw_logloss"]) - minimum
        )
        <= SELECTION_TIE_TOLERANCE
    ]
    audit_winner = min(tied, key=lambda item: simplicity_key(item["config"]))
    if (
        str(audit_winner["config_id"]) != str(frozen_winner["config_id"])
        or len(tied) != int(frozen_winner["tie_candidate_count"])
        or abs(minimum - float(frozen_winner["minimum_observed_loss"])) > 1e-15
    ):
        raise Phase3CompletionError(
            f"non-authoritative equality audit failed for fold {outer}"
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    expected_prediction_rows = 108 * len(outer_events)
    if len(predictions) != expected_prediction_rows:
        raise Phase3CompletionError(
            f"fold {outer} CSCV prediction row count changed"
        )
    return {
        "outer_fold": outer,
        "frozen_winner": dict(frozen_winner),
        "winner_capture": winner_capture,
        "selection_ledger_reproduced_for_equality_audit": ledger,
        "inner_split": split_audit,
        "outer_evaluation_gate": outer_gate,
        "cscv_predictions": predictions,
        "timing": {
            "cpu_seconds": float(cpu_seconds),
            "wall_seconds": float(wall_seconds),
        },
        "authority_and_scope": {
            "purpose": "cscv_reproduction_no_selection_authority",
            "frozen_winner_source": (
                "hash-gated data/models/ablation81/"
                "phase3_stop_report.json"
            ),
            "select_winning_config_called": False,
            "outer_predictions_used_for_selection": False,
            "equality_audit": {
                "status": "passed",
                "reproduced_winner_config_id": str(
                    audit_winner["config_id"]
                ),
                "minimum_observed_loss": float(minimum),
                "tie_candidate_count": int(len(tied)),
                "winner_loss_difference_vs_frozen": float(loss_difference),
                "loss_tolerance": 1e-15,
            },
        },
    }


def _seed_prediction_frame(
    parts: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    frame = pd.concat(list(parts), ignore_index=True)
    frame["decision_ts"] = pd.to_datetime(
        frame["decision_ts"], utc=True, errors="raise"
    )
    if (
        len(frame) != EXPECTED_EVALUATED_ROWS
        or frame.duplicated(["symbol", "decision_ts", "fold"]).any()
        or sorted(frame["fold"].astype(int).unique().tolist())
        != list(OUTER_FOLDS)
    ):
        raise Phase3CompletionError("seed prediction population changed")
    return frame.sort_values(
        ["fold", "decision_ts", "symbol"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _exception_coefficient(message: str) -> float | None:
    match = re.search(r"coefficient=?(?: must be positive; observed )?([+-]?[0-9.eE-]+)", message)
    if match is None:
        match = re.search(r"observed ([+-]?[0-9.eE-]+)", message)
    return None if match is None else float(match.group(1))


def _run_seed_pipeline(
    pinned_events: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    *,
    seed: int,
    economic_cost: float,
    postmortem: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one production attempt and apply only the two narrow owner codes."""

    try:
        pipeline = run_a4_calibration_pipeline(
            pinned_events,
            raw_predictions,
            economic_cost=float(economic_cost),
        )
    except calibration.CalibrationError as exc:
        code_gate = assert_allowed_family_rejection_code(exc.code)
        exception_class = (
            f"{exc.__class__.__module__}.{exc.__class__.__name__}"
        )
        coefficient = _exception_coefficient(str(exc))
        rejection_gate = None
        if int(seed) == 42:
            synthetic_c3 = postmortem["per_fold"]["3"]["synthetic_platt"]
            synthetic_coefficient = synthetic_c3[
                "coefficient_from_fit_metadata"
            ]
            if synthetic_coefficient != EXPECTED_ADJUDICATED_SIGNATURE[
                "coefficient"
            ]:
                raise Phase3CompletionError(
                    "canonical synthetic C3 coefficient differs from "
                    "owner signature"
                )
            if (
                str(exc) != EXPECTED_CANONICAL_EXCEPTION_MESSAGE
                or coefficient != EXPECTED_ADJUDICATED_SIGNATURE[
                    "coefficient"
                ]
            ):
                raise Phase3CompletionError(
                    "canonical production exception text/coefficient changed"
                )
            rejection_gate = assert_exact_authorized_rejection(
                seed=seed,
                calibrator="C3",
                training_folds=[2],
                score="p_meta",
                method="platt",
                coefficient=float(coefficient),
                exception_class=exception_class,
                exception_code=str(exc.code),
            )
        semantics = rejected_confirmatory_semantics()
        return {
            **semantics,
            "seed": int(seed),
            "exception": {
                "class": exception_class,
                "code": str(exc.code),
                "message": str(exc),
                "details": exc.details,
                "parsed_coefficient": coefficient,
            },
            "allowed_code_gate": code_gate,
            "exact_canonical_signature_gate": rejection_gate,
            "call_order": [
                "calibration.walk_forward_calibrate",
                "calibration.replay_threshold_mechanism (not reached)",
            ],
            "replay_status": "not_reached_due_to_production_pipeline_rejection",
            "fallback_attempted": False,
            "isotonic_substitution_attempted": False,
            "coefficient_sign_change_attempted": False,
            "automatic_raw_mode_attempted": False,
            "coefficient_floor_attempted": False,
            "fold_deletion_attempted": False,
            "production_code_changed": False,
            "section_8_1_metrics": null_section_8_1_metrics(),
            "funding_affected_fold5_overlap": {
                "affected_population_rows": None,
                "n_trades": None,
                "sum_net": None,
                "reason": NOT_EVALUABLE_REASON,
            },
            "oof_available": False,
        }
    if int(seed) == 42:
        raise Phase3CompletionError(
            "canonical seed 42 unexpectedly passed the production pipeline"
        )
    oof = pipeline.pop("oof")
    return {
        "seed": int(seed),
        "pipeline_status": "production_pipeline_completed",
        "construction_status_fa": "موفق در مرحله‌ی ساخت",
        "trade_set_defined": True,
        "confirmatory_test_status": "not_applicable_E10_diagnostic_seed",
        "metrics": pipeline["metrics"],
        "funding_affected_fold5_overlap": pipeline[
            "funding_affected_fold5_overlap"
        ],
        "calibration": pipeline["calibration"],
        "reconstruction": pipeline["reconstruction"],
        "call_order": pipeline["calibration"]["call_order"],
        "fallback_attempted": False,
        "oof_available": True,
        "oof_row_count": int(len(oof)),
        "decision_role": "E10_diagnostic_seed_only",
    }


def _canonical_rejected_oof(raw_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = raw_predictions.loc[
        :, ["symbol", "decision_ts", "fold", "p_raw"]
    ].copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["decision_ts"] = pd.to_datetime(
        frame["decision_ts"], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["p_raw"] = pd.to_numeric(frame["p_raw"], errors="raise")
    frame["p_effective"] = np.nan
    frame["tau"] = np.nan
    frame["traded_A4"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    frame["pipeline_status"] = PIPELINE_REJECTION_STATUS
    frame = frame.loc[:, list(A4_REJECTED_OOF_COLUMNS)].sort_values(
        ["fold", "decision_ts", "symbol"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if (
        len(frame) != EXPECTED_EVALUATED_ROWS
        or tuple(frame.columns) != A4_REJECTED_OOF_COLUMNS
        or not np.isfinite(frame["p_raw"].to_numpy(dtype=float)).all()
        or not frame[["p_effective", "tau", "traded_A4"]].isna().all(axis=None)
        or not frame["pipeline_status"].eq(PIPELINE_REJECTION_STATUS).all()
    ):
        raise Phase3CompletionError("canonical rejected A4 OOF schema failed")
    return frame


def _stage_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            _json_ready(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stage_csv(path: Path, frame: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(
            handle,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
            na_rep="",
        )
        handle.flush()
        os.fsync(handle.fileno())


def _assert_float_exact(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    for column in columns:
        left = expected[column].to_numpy(dtype=float)
        right = observed[column].to_numpy(dtype=float)
        if not np.array_equal(left, right):
            raise Phase3CompletionError(
                f"CSV exact float round-trip changed {column}"
            )


def _commit_scientific_artifacts(
    *,
    selected_configs: Mapping[str, Any],
    seed_stability: Mapping[str, Any],
    cscv_predictions: pd.DataFrame,
    cscv_matrix: pd.DataFrame,
    a4_oof: pd.DataFrame,
) -> dict[str, Any]:
    existing = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in SCIENTIFIC_DESTINATIONS.values()
        if path.exists()
    ]
    if existing:
        raise Phase3CompletionError(
            f"refusing to overwrite phase3 scientific artifacts: {existing}"
        )
    payloads = {
        "selected_configs": selected_configs,
        "seed_stability": seed_stability,
    }
    frames = {
        "cscv_predictions": cscv_predictions,
        "cscv_matrix": cscv_matrix,
        "a4_oof": a4_oof,
    }
    with tempfile.TemporaryDirectory(
        prefix=".phase3-complete-stage-",
        dir=OUTPUT_DIR,
    ) as temporary_text:
        temporary = Path(temporary_text)
        staged = {
            key: temporary / destination.name
            for key, destination in SCIENTIFIC_DESTINATIONS.items()
        }
        for key, payload in payloads.items():
            _stage_json(staged[key], payload)
            observed = json.loads(staged[key].read_text(encoding="utf-8"))
            if observed != _json_ready(payload):
                raise Phase3CompletionError(
                    f"{key} staged JSON round-trip changed"
                )
        for key, frame in frames.items():
            _stage_csv(staged[key], frame)

        observed_predictions = pd.read_csv(
            staged["cscv_predictions"],
            float_precision="round_trip",
            low_memory=False,
        )
        if (
            tuple(observed_predictions.columns) != PREDICTION_COLUMNS
            or len(observed_predictions) != len(cscv_predictions)
            or not observed_predictions.loc[
                :, ["config_id", "outer_fold", "symbol", "decision_ts"]
            ].astype(str).equals(
                cscv_predictions.loc[
                    :, ["config_id", "outer_fold", "symbol", "decision_ts"]
                ].astype(str)
            )
        ):
            raise Phase3CompletionError("CSCV prediction staged keys changed")
        _assert_float_exact(
            cscv_predictions, observed_predictions, ("p_raw",)
        )

        observed_matrix = pd.read_csv(
            staged["cscv_matrix"],
            float_precision="round_trip",
            low_memory=False,
        )
        if (
            tuple(observed_matrix.columns) != MATRIX_COLUMNS
            or len(observed_matrix) != len(cscv_matrix)
            or not observed_matrix.loc[
                :, ["config_id", "boosting_type", "bootstrap"]
            ].astype(str).equals(
                cscv_matrix.loc[
                    :, ["config_id", "boosting_type", "bootstrap"]
                ].astype(str)
            )
        ):
            raise Phase3CompletionError("CSCV matrix staged schema changed")
        _assert_float_exact(
            cscv_matrix,
            observed_matrix,
            (
                "depth",
                "l2_leaf_reg",
                "learning_rate",
                *BLOCK_COLUMNS,
            ),
        )

        observed_oof = pd.read_csv(
            staged["a4_oof"],
            float_precision="round_trip",
            low_memory=False,
        )
        if (
            tuple(observed_oof.columns) != A4_REJECTED_OOF_COLUMNS
            or len(observed_oof) != EXPECTED_EVALUATED_ROWS
            or not observed_oof.loc[
                :, ["symbol", "decision_ts", "fold", "pipeline_status"]
            ].astype(str).equals(
                a4_oof.loc[
                    :, ["symbol", "decision_ts", "fold", "pipeline_status"]
                ].astype(str)
            )
            or not observed_oof[
                ["p_effective", "tau", "traded_A4"]
            ].isna().all(axis=None)
            or not observed_oof["pipeline_status"].eq(
                PIPELINE_REJECTION_STATUS
            ).all()
        ):
            raise Phase3CompletionError(
                "rejected A4 OOF staged null/schema contract changed"
            )
        _assert_float_exact(a4_oof, observed_oof, ("p_raw",))
        raw_text = staged["a4_oof"].read_text(encoding="utf-8")
        if ",nan," in raw_text.lower() or ",<na>," in raw_text.lower():
            raise Phase3CompletionError(
                "rejected A4 OOF uses a textual null sentinel"
            )

        committed: list[Path] = []
        try:
            for key, destination in SCIENTIFIC_DESTINATIONS.items():
                if destination.exists():
                    raise Phase3CompletionError(
                        f"phase3 destination appeared during staging: {destination}"
                    )
                os.replace(staged[key], destination)
                committed.append(destination)
        except BaseException:
            rollback_failures: list[str] = []
            for path in reversed(committed):
                try:
                    path.unlink()
                except OSError as exc:
                    rollback_failures.append(f"{path}: {exc}")
            if rollback_failures:
                raise Phase3CompletionError(
                    "partial artifact commit rollback failed: "
                    + " | ".join(rollback_failures)
                )
            raise
    identities = {
        key: {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            **_identity_pair(path),
        }
        for key, path in SCIENTIFIC_DESTINATIONS.items()
    }
    return {
        "status": "passed",
        "transaction": "stage_validate_atomic_replace_with_rollback",
        "output_count": len(identities),
        "outputs": identities,
    }


def _accepted_deviations() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in DEVIATIONS:
        record = dict(item)
        record["status"] = "accepted_by_owner"
        records.append(record)
    if len(records) != 4:
        raise Phase3CompletionError("phase3 deviation count changed")
    return records


def _winner_scientific_table(
    stop_report: Mapping[str, Any],
) -> dict[str, Any]:
    ln2 = math.log(2.0)
    rows: dict[str, Any] = {}
    for fold in OUTER_FOLDS:
        winner = _frozen_winner(stop_report, fold)
        loss = float(winner["inner_validation_weighted_raw_logloss"])
        rows[str(fold)] = {
            "outer_fold": int(fold),
            "config_id": str(winner["config_id"]),
            "config": dict(winner["config"]),
            "tree_count": int(winner["tree_count"]),
            "best_iteration": int(winner["best_iteration"]),
            "inner_validation_tb_uniqueness_weighted_raw_logloss": loss,
            "signed_loss_minus_ln2": float(loss - ln2),
            "tie_candidate_count": int(winner["tie_candidate_count"]),
        }
    return {
        "ln2": ln2,
        "constant_predictor_weighted_loss_context": 0.6239,
        "pinned_context_sentence_fa": PINNED_LN2_CAVEAT_FA,
        "rows": rows,
        "folds_2_and_4_are_single_tree": True,
        "folds_2_and_4_use_grid_minimum_learning_rate_0_01": True,
    }


def run_phase3_completion() -> dict[str, Any]:
    """Complete phase three and stop before every phase-four mechanism."""

    phase_cpu_start = time.process_time()
    phase_wall_start = time.perf_counter()
    started_at = _utc_now()
    absence_gate = _assert_destinations_absent()
    _append_task_log(
        {
            "event": "phase3_owner_adjudicated_completion_started",
            "owner_artifact_already_registered_as_first_resume_action": True,
        }
    )
    owner_gate = verify_owner_adjudication_platt(PROJECT_ROOT)
    stop_report, stop_authority_gate = _load_stop_authority()
    reentry = _run_reentry_protocol()
    _progress(
        {
            "event": "phase3_completion_reentry_passed",
            "owner_adjudication": owner_gate["owner_artifact"],
            "phase3_stop_report": stop_authority_gate[
                "phase3_stop_report"
            ],
        }
    )

    assembly, pretraining_gates = run_pretraining_gates(PROJECT_ROOT)
    population = pretraining_gates["population_alignment"]
    membership = population["key_membership_counts"]
    if membership != {"both": 11_029, "left_only": 0, "right_only": 0}:
        raise Phase3CompletionError(
            f"population gate signature changed: {membership}"
        )
    _progress(
        {
            "event": "phase3_completion_pretraining_gates_passed",
            "execution_order": pretraining_gates["execution_order"],
            "population_membership": membership,
        }
    )
    pytest_before = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }

    events = assembly["events"].copy()
    events["fold"] = pd.to_numeric(
        events["fold"], errors="raise"
    ).astype(int)
    events["decision_ts"] = pd.to_datetime(
        events["decision_ts"], utc=True, errors="raise"
    )
    if len(events) != EXPECTED_POPULATION_ROWS:
        raise Phase3CompletionError("assembled population row count changed")
    X = assembly["X"]
    feature_columns = list(assembly["feature_columns"])
    categorical_features = list(assembly["categorical_features"])
    catalog = build_config_catalog()

    reproduction_reports: dict[str, Any] = {}
    cscv_parts: list[pd.DataFrame] = []
    for outer_fold in OUTER_FOLDS:
        _progress(
            {
                "event": "phase3_cscv_reproduction_outer_started",
                "outer_fold": int(outer_fold),
                "selection_authority": False,
            }
        )
        reproduction = _reproduce_outer_fold_for_cscv(
            events.loc[events["fold"].lt(outer_fold)].copy(),
            events.loc[events["fold"].eq(outer_fold)].copy(),
            X,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            outer_fold=outer_fold,
            catalog=catalog,
            frozen_winner=_frozen_winner(stop_report, outer_fold),
        )
        cscv_parts.append(reproduction.pop("cscv_predictions"))
        reproduction_reports[str(outer_fold)] = reproduction
        _progress(
            {
                "event": "phase3_cscv_reproduction_outer_completed",
                "outer_fold": int(outer_fold),
                "timing": reproduction["timing"],
                "equality_audit": reproduction["authority_and_scope"][
                    "equality_audit"
                ],
            }
        )

    evaluated_events = events.loc[events["fold"].isin(OUTER_FOLDS)].copy()
    if len(evaluated_events) != EXPECTED_EVALUATED_ROWS:
        raise Phase3CompletionError("evaluated population row count changed")
    cscv_predictions_memory = pd.concat(cscv_parts, ignore_index=True).loc[
        :, list(PREDICTION_COLUMNS)
    ]
    cscv_matrix, cscv_audit = build_cscv_matrix(
        cscv_predictions_memory,
        evaluated_events,
        catalog=catalog,
    )
    cscv_predictions = canonicalize_prediction_rows(
        cscv_predictions_memory
    )

    seed_parts: dict[int, list[pd.DataFrame]] = {
        int(seed): [] for seed in SEED_STABILITY_SEEDS
    }
    refit_records: dict[str, dict[str, Any]] = {
        str(fold): {} for fold in OUTER_FOLDS
    }
    refit_cpu_total = 0.0
    refit_wall_total = 0.0
    for outer_fold in OUTER_FOLDS:
        frozen = _frozen_winner(stop_report, outer_fold)
        selection_events = events.loc[events["fold"].lt(outer_fold)].copy()
        outer_events = events.loc[events["fold"].eq(outer_fold)].copy()
        for seed in SEED_STABILITY_SEEDS:
            refit = refit_winner_for_seed(
                selection_events,
                outer_events,
                X,
                feature_columns=feature_columns,
                categorical_features=categorical_features,
                outer_fold=outer_fold,
                config=frozen["config"],
                tree_count=int(frozen["tree_count"]),
                seed=int(seed),
            )
            probability = refit.pop("probabilities")
            metadata = refit["metadata"]
            refit_cpu_total += float(metadata["fit_cpu_seconds"])
            refit_wall_total += float(metadata["fit_wall_seconds"])
            seed_parts[int(seed)].append(
                pd.DataFrame(
                    {
                        "symbol": outer_events["symbol"].astype(str).to_numpy(),
                        "decision_ts": outer_events["decision_ts"].to_numpy(),
                        "fold": int(outer_fold),
                        "p_raw": probability,
                    }
                )
            )
            refit_records[str(outer_fold)][str(seed)] = metadata
        _progress(
            {
                "event": "phase3_frozen_winner_refits_completed",
                "outer_fold": int(outer_fold),
                "seeds": list(SEED_STABILITY_SEEDS),
            }
        )

    seed_predictions = {
        int(seed): _seed_prediction_frame(seed_parts[int(seed)])
        for seed in SEED_STABILITY_SEEDS
    }
    pinned_events = load_calibrated_events(PROJECT_ROOT)
    postmortem_population = pinned_events.loc[
        pinned_events["meta_eval_status"].eq("evaluated")
    ].copy()
    if len(postmortem_population) != EXPECTED_EVALUATED_ROWS:
        raise Phase3CompletionError(
            "pinned post-mortem population row count changed"
        )
    seed_results: dict[str, Any] = {}
    canonical_rejection: dict[str, Any] | None = None
    for seed in SEED_STABILITY_SEEDS:
        tree_counts = {
            int(fold): int(_frozen_winner(stop_report, fold)["tree_count"])
            for fold in OUTER_FOLDS
        }
        diagnostic = build_seed_postmortem(
            postmortem_population,
            seed_predictions[int(seed)],
            seed=int(seed),
            tree_count_by_fold=tree_counts,
        )
        pipeline = _run_seed_pipeline(
            pinned_events,
            seed_predictions[int(seed)],
            seed=int(seed),
            economic_cost=float(assembly["metadata"]["cost_round_trip"]),
            postmortem=diagnostic,
        )
        seed_results[str(seed)] = {
            "seed": int(seed),
            "tree_count_by_outer_fold": {
                str(key): value for key, value in tree_counts.items()
            },
            "diagnostic_post_mortem": diagnostic,
            "production_pipeline": pipeline,
            "refit_metadata_by_outer_fold": {
                str(fold): refit_records[str(fold)][str(seed)]
                for fold in OUTER_FOLDS
            },
        }
        if int(seed) == 42:
            canonical_rejection = pipeline
        _progress(
            {
                "event": "phase3_seed_pipeline_classified",
                "seed": int(seed),
                "pipeline_status": pipeline["pipeline_status"],
            }
        )
    if (
        canonical_rejection is None
        or canonical_rejection["pipeline_status"]
        != PIPELINE_REJECTION_STATUS
        or canonical_rejection["mandatory_decision_label_fa"]
        != PINNED_DECISION_LABEL_FA
    ):
        raise Phase3CompletionError(
            "canonical A4 rejection semantics were not reproduced"
        )
    a4_oof = _canonical_rejected_oof(seed_predictions[42])

    original_search_timings = {
        str(fold): stop_report["completed_before_stop"][
            "outer_fold_completion_records"
        ][str(fold)]["search_timing"]
        for fold in OUTER_FOLDS
    }
    original_timing_summary = dict(stop_report["known_timing"])
    reproduction_cpu = float(
        sum(
            reproduction_reports[str(fold)]["timing"]["cpu_seconds"]
            for fold in OUTER_FOLDS
        )
    )
    reproduction_wall = float(
        sum(
            reproduction_reports[str(fold)]["timing"]["wall_seconds"]
            for fold in OUTER_FOLDS
        )
    )
    selected_configs_payload = {
        "schema_version": 2,
        "contract_version": 7,
        "phase": 3,
        "frozen": True,
        "thread_count": FROZEN_THREAD_COUNT,
        "selection_authority": {
            "source": (
                "data/models/ablation81/phase3_stop_report.json"
            ),
            **EXPECTED_PHASE3_STOP_IDENTITY,
            "reproduction_has_selection_authority": False,
            "select_winning_config_called_during_completion": False,
        },
        "selection_objective": "tb_uniqueness_weighted_raw_logloss_only",
        "selection_tie_tolerance": SELECTION_TIE_TOLERANCE,
        "config_catalog": catalog,
        "outer_folds": reproduction_reports,
        "original_frozen_search_timing": {
            "per_outer_fold": original_search_timings,
            **original_timing_summary,
        },
        "cscv_reproduction_timing": {
            "per_outer_fold": {
                str(fold): reproduction_reports[str(fold)]["timing"]
                for fold in OUTER_FOLDS
            },
            "total_cpu_seconds": reproduction_cpu,
            "total_wall_seconds": reproduction_wall,
        },
        "winner_refit_timing": {
            "total_cpu_seconds": float(refit_cpu_total),
            "total_wall_seconds": float(refit_wall_total),
        },
        "fit_counts": {
            "cscv_reproduction_grid_fits": EXPECTED_SEARCH_FIT_COUNT,
            "frozen_winner_seed_refits": EXPECTED_REFIT_COUNT,
            "new_search_or_reselection_fits": 0,
        },
    }
    seed_stability_payload = {
        "schema_version": 2,
        "contract_version": 7,
        "phase": 3,
        "arm": "A4",
        "exploratory": "E10",
        "frozen": True,
        "seeds": list(SEED_STABILITY_SEEDS),
        "canonical_a4_seed": 42,
        "metric_semantics": {
            "maxdd_semantics": MAXDD_SEMANTICS,
            "es_semantics": ES_SEMANTICS,
            "official_A4_metrics_when_rejected": (
                "all null with "
                "not_evaluable_due_to_production_pipeline_rejection"
            ),
            "raw_diagnostics_decision_eligible": False,
        },
        "canonical_confirmatory_and_decision_semantics": (
            rejected_confirmatory_semantics()
        ),
        "results_by_seed": seed_results,
        "post_mortem_limits": {
            "new_grid_search_performed": False,
            "synthetic_calibrators_used_for_scores": False,
            "synthetic_calibrators_used_for_thresholds": False,
            "synthetic_calibrators_used_for_trades": False,
            "synthetic_calibrators_used_for_decision": False,
            "production_rejection_can_rescue_arm": False,
        },
    }

    pytest_before_commit = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    precommit_forbidden = _verify_forbidden_owner_rule()
    precommit_prior = _verify_prior_frozen_outputs()
    verify_owner_adjudication_platt(PROJECT_ROOT)
    _load_stop_authority()
    output_commit = _commit_scientific_artifacts(
        selected_configs=selected_configs_payload,
        seed_stability=seed_stability_payload,
        cscv_predictions=cscv_predictions,
        cscv_matrix=cscv_matrix,
        a4_oof=a4_oof,
    )
    _progress(
        {
            "event": "phase3_owner_adjudicated_artifacts_frozen",
            "outputs": output_commit["outputs"],
        }
    )

    pytest_after = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    forbidden_after = _verify_forbidden_owner_rule()
    prior_after = _verify_prior_frozen_outputs()
    owner_after = verify_owner_adjudication_platt(PROJECT_ROOT)
    _, stopped_sources_after = _load_stop_authority()
    new_source_identities = {
        relative: {
            "path": relative,
            **_identity_pair(PROJECT_ROOT / relative),
        }
        for relative in (
            "analytics/ablation81/adjudication_platt.py",
            "analytics/ablation81/postmortem.py",
            "analytics/ablation81/phase3_complete.py",
        )
    }

    winner_table = _winner_scientific_table(stop_report)
    coefficient_ratio = float(
        abs(HISTORICAL_A2_PLATT_COEFFICIENT)
        / abs(EXPECTED_ADJUDICATED_SIGNATURE["coefficient"])
    )
    phase_cpu = time.process_time() - phase_cpu_start
    phase_wall = time.perf_counter() - phase_wall_start
    report = {
        "schema_version": 2,
        "contract_version": 7,
        "phase": 3,
        "status": (
            "completed_with_scientific_production_pipeline_rejection_"
            "waiting_for_owner_continue"
        ),
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "resume_registration": {
            "owner_adjudication_platt_was_first_resume_action": True,
            "owner_adjudication": owner_gate,
            "phase3_stop_authority": stop_authority_gate,
            "destination_absence_before_completion": absence_gate,
        },
        "reentry_protocol": reentry,
        "pretraining_hard_gates": pretraining_gates,
        "pretraining_gate_required_details": {
            "execution_order": [
                "population_alignment",
                "input_files_lineage",
                "owner_adjudicated_funding_signature",
            ],
            "population_alignment": {
                "both": int(membership["both"]),
                "left_only": int(membership["left_only"]),
                "right_only": int(membership["right_only"]),
                "four_columns": [
                    "p_primary",
                    "tb_return",
                    "tb_uniqueness",
                    "fold",
                ],
                "absolute_tolerance": 1e-9,
            },
            "input_files_lineage": pretraining_gates[
                "input_files_lineage"
            ],
            "funding_signature": pretraining_gates["funding_signature"],
        },
        "pytest": {
            "before_scientific_completion": pytest_before,
            "before_artifact_commit": pytest_before_commit,
            "after_artifacts": pytest_after,
            "phase3_pinned_tests_implemented_and_green": [9, 10, 12, 15],
        },
        "scientific_results": {
            "classification": "scientific_result",
            "canonical_A4_pipeline": canonical_rejection,
            "confirmatory_test_status": (
                "not_executable_due_to_production_pipeline_rejection"
            ),
            "decision_rule_outcome": "case_c",
            "decision_definition_fa": (
                "بازویی که ساخته نشد، به‌تعریف هیچ دروازه‌ای را پاس نمی‌کند"
            ),
            "mandatory_decision_label_fa": PINNED_DECISION_LABEL_FA,
            "production_rejection_pinned_sentence_fa": (
                PINNED_PRODUCTION_REJECTION_SENTENCE_FA
            ),
            "winner_table": winner_table,
            "historical_A2_coefficient_comparison": {
                "historical_A2_final_platt_coefficient": (
                    HISTORICAL_A2_PLATT_COEFFICIENT
                ),
                "observed_A4_rejected_coefficient": (
                    EXPECTED_ADJUDICATED_SIGNATURE["coefficient"]
                ),
                "historical_A2_magnitude_over_A4_magnitude": (
                    coefficient_ratio
                ),
                "approximately": "65x",
            },
            "folds_2_and_4_minimum_grid_learning_rate": {
                "fold_2_learning_rate": 0.01,
                "fold_4_learning_rate": 0.01,
                "grid_minimum_learning_rate": 0.01,
                "status": "passed",
            },
            "diagnostic_post_mortem_by_seed": {
                seed: seed_results[seed]["diagnostic_post_mortem"]
                for seed in seed_results
            },
            "seed_pipeline_status": {
                seed: seed_results[seed]["production_pipeline"][
                    "pipeline_status"
                ]
                for seed in seed_results
            },
            "cscv": cscv_audit,
            "a4_oof_post_hoc_schema": {
                "path": A4_OOF_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "columns": list(A4_REJECTED_OOF_COLUMNS),
                "row_population": (
                    "all evaluated events in folds 2 through 5"
                ),
                "row_count": EXPECTED_EVALUATED_ROWS,
                "p_raw": "finite and populated for every row",
                "p_effective": "true null for every row",
                "tau": "true null for every row",
                "traded_A4": "true null for every row",
                "pipeline_status": PIPELINE_REJECTION_STATUS,
            },
            "A4_trade_set_defined": False,
            "A4_section_8_1_metrics": null_section_8_1_metrics(),
            "A4_E12": {
                "value": None,
                "reason": NOT_EVALUABLE_REASON,
            },
            "A4_placebo": {
                "sampling_executed": False,
                "percentile_rank": None,
                "monte_carlo_p": None,
                "reason": NOT_EVALUABLE_REASON,
            },
        },
        "timing": {
            "original_frozen_search_per_outer_fold": original_search_timings,
            "original_frozen_search_summary": original_timing_summary,
            "cscv_reproduction_per_outer_fold": {
                str(fold): reproduction_reports[str(fold)]["timing"]
                for fold in OUTER_FOLDS
            },
            "cscv_reproduction_total_cpu_seconds": reproduction_cpu,
            "cscv_reproduction_total_wall_seconds": reproduction_wall,
            "winner_refit_total_cpu_seconds": float(refit_cpu_total),
            "winner_refit_total_wall_seconds": float(refit_wall_total),
            "phase_cpu_seconds": float(phase_cpu),
            "phase_wall_seconds": float(phase_wall),
        },
        "output_commit": output_commit,
        "supporting_artifacts": [
            "data/models/ablation81/owner_adjudication_platt.json",
            "data/models/ablation81/phase3_stop_report.json",
            "data/models/ablation81/phase3_integrity_report.json",
            "data/models/ablation81/a4_oof_4h.csv",
        ],
        "execution_incidents": [
            {
                "classification": (
                    "false_diagnostic_guard_failure_not_scientific_deviation"
                ),
                "stage": "synthetic_C3_exact_coefficient_check",
                "cause": (
                    "A three-key outer merge proved exact membership but "
                    "changed the frozen population row order before the "
                    "synthetic diagnostic fit."
                ),
                "observed_production_result": (
                    "The actual real-data production call reproduced the "
                    "exact owner-adjudicated exception and coefficient."
                ),
                "scientific_artifacts_committed_before_failure": False,
                "scientific_method_or_production_code_changed": False,
                "correction_scope": (
                    "After the same one-to-one three-key membership proof, "
                    "attach p_raw in the original frozen event order."
                ),
                "post_correction_exact_synthetic_coefficient": (
                    -0.000843584342876648
                ),
            }
        ],
        "deviations": _accepted_deviations(),
        "self_audit_doubts": [
            {
                "doubt": (
                    "CSCV reproduction could silently become a second "
                    "configuration selection."
                ),
                "check": (
                    "Winners came only from the exact hash-gated stop report; "
                    "all 432 fits were used for CSCV and equality audit, and "
                    "select_winning_config was never called."
                ),
                "result": "passed",
            },
            {
                "doubt": (
                    "A synthetic fallback could accidentally rescue A4 or "
                    "produce thresholds/trades."
                ),
                "check": (
                    "Synthetic fits used is_real_data_run=False only in the "
                    "decision-inert post-mortem; their outputs were never "
                    "transformed into scores, thresholds, trades, or decisions."
                ),
                "result": "no rescue and no decision use",
            },
            {
                "doubt": (
                    "Blank A4 fields could be coerced into zero or False."
                ),
                "check": (
                    "The staged and committed OOF round-trip requires all "
                    "p_effective, tau, and traded_A4 cells to remain true nulls."
                ),
                "result": "passed",
            },
            {
                "doubt": (
                    "The canonical error might be a different seed, fold, "
                    "score, method, coefficient, exception class, or code."
                ),
                "check": (
                    "The actual production exception and synthetic C3 metadata "
                    "were asserted against the exact owner signature."
                ),
                "result": "passed",
            },
            {
                "doubt": (
                    "CatBoost's fitted objective weighting could be confused "
                    "with the preregistered selection loss."
                ),
                "check": (
                    "For every frozen winner, both tb_uniqueness-only raw "
                    "logloss and tb_uniqueness times positive-class scale "
                    "logloss were calculated on the same inner-validation "
                    "probabilities without an extra fit."
                ),
                "result": "reported as decision-inert objective post-mortem",
            },
        ],
        "integrity_after_outputs": {
            "forbidden_files_gate": forbidden_after,
            "prior_frozen_output_gate": prior_after,
            "owner_adjudication_gate": owner_after,
            "stopped_run_sources_gate": stopped_sources_after,
            "new_source_identities": new_source_identities,
            "precommit_forbidden_files_gate": precommit_forbidden,
            "precommit_prior_frozen_output_gate": precommit_prior,
            "fallback_prohibition": {
                "isotonic_substitution": False,
                "coefficient_sign_change": False,
                "forced_raw_mode": False,
                "coefficient_floor": False,
                "fold_deletion": False,
                "production_code_change": False,
                "pinned_joblib_used": False,
            },
        },
        "carry_forward_to_final_report": {
            "production_rejection_sentence_fa": (
                PINNED_PRODUCTION_REJECTION_SENTENCE_FA
            ),
            "winner_ln2_context_sentence_fa": PINNED_LN2_CAVEAT_FA,
            "A5_A6_interpretation_limit_fa": PINNED_A5_A6_CAVEAT_FA,
            "mandatory_decision_label_fa": PINNED_DECISION_LABEL_FA,
            "recommendation_sentences_allowed": False,
        },
        "owner_future_requirements": {
            "all_preregistered_arms_A1_through_A7_remain_required": True,
            "A5_A6_caveat": PINNED_A5_A6_CAVEAT_FA,
            "phase5_stage_rows_A2_A7_copied_without_recomputation": True,
            "phase5_exact_250000_row_sum_net_assertion": True,
            "phase5_stage_exact_alignment_keys": [
                "null_type",
                "matching",
                "arm",
                "scope",
                "seed",
            ],
            "phase5_stage_sum_net_tolerance": 0.0,
            "phase5_feature_null_rows_full_status_required": [
                {
                    "symbol": "ADAUSDT",
                    "decision_ts": "2024-02-05T16:00:00Z",
                    "fold": 1,
                    "evaluated_in_any_metric": False,
                    "feature_block": "funding",
                },
                {
                    "symbol": "BTCUSDT",
                    "decision_ts": "2024-08-06T08:00:00Z",
                    "fold": 2,
                    "evaluated_in_metrics": True,
                    "traded_baseline": True,
                    "traded_meta": False,
                    "feature_block": "HMM",
                },
            ],
            "phase5_E11_prominent_pinned_sentences_required": [
                "مقدار p برابر 0.00009999 کفِ تفکیک‌پذیری ده هزار بذر است، نه اندازه‌ی اثر",
                "بازوی E11 تشخیصی است، نه تأییدی",
                "پرسش این نال، مقایسه‌ی زیرمجموعه‌ی تأییدشده‌ی متا با زیرمجموعه‌ی تصادفی هم‌شمار از معاملات A1 است",
            ],
            "phase5_E11_recommendation_sentences_allowed": False,
            "no_next_phase_mechanism_before_continue": True,
        },
        "phase3_completed": True,
        "phase4_ready": True,
        "phase4_mechanism_executed": False,
        "next_phase_requires_separate_owner_continue_message": True,
        "no_recommendation_sentence_in_phase3_report": True,
    }
    ready_report = _json_ready(report)
    write_json_once(PHASE_REPORT_PATH, ready_report)
    observed_report = json.loads(PHASE_REPORT_PATH.read_text(encoding="utf-8"))
    if observed_report != ready_report:
        raise Phase3CompletionError("phase3 report JSON round-trip changed")
    report_identity = {
        "path": PHASE_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        **_identity_pair(PHASE_REPORT_PATH),
    }
    _append_task_log(
        {
            "event": "phase3_completed_waiting_for_separate_owner_continue",
            "phase_report": report_identity,
            "phase4_mechanism_executed": False,
        }
    )
    return {
        "status": report["status"],
        "phase_report": report_identity,
        "scientific_outputs": output_commit["outputs"],
        "canonical_pipeline_status": canonical_rejection[
            "pipeline_status"
        ],
        "confirmatory_test_status": report["scientific_results"][
            "confirmatory_test_status"
        ],
        "decision_rule_outcome": report["scientific_results"][
            "decision_rule_outcome"
        ],
        "mandatory_decision_label_fa": PINNED_DECISION_LABEL_FA,
        "timing": report["timing"],
        "pytest_after": pytest_after,
        "phase4_ready": True,
        "phase4_mechanism_executed": False,
        "next_phase_requires_separate_owner_continue_message": True,
    }
