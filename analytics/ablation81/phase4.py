# ==============================================================================
# analytics/ablation81/phase4.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Execute registered phase-four feature ablations and stop before phase five."""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import binance_vision
import calibration
import dataset
import meta_model

from .a4 import run_a4_calibration_pipeline
from .adjudication import derive_affected_mask_from_pinned
from .adjudication_platt import assert_allowed_family_rejection_code
from .adjudication_restore import verify_owner_adjudicated_restore
from .feature_ablation import (
    BASE_CATEGORICAL,
    BASE_FEATURES,
    MODEL_ARMS,
    build_frozen_arm_spec,
    materialize_session,
    project_arm_frame,
)
from .integrity import (
    IntegrityError,
    collect_environment_fingerprint,
    file_identity,
    load_calibrated_events,
    verify_environment_fingerprint,
    verify_pinned_artifacts,
    verify_production_constants,
    verify_research_input_lineage,
    verify_s1_anchors,
    write_json_once,
)
from .metrics import REPORT_SCOPES, model_metrics, prepare_evaluated_events, trade_metrics
from .nested import FROZEN_THREAD_COUNT, OUTER_FOLDS, refit_winner_for_seed
from .phase3 import (
    ENVIRONMENT_PATH,
    LINEAGE_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
    _append_task_log,
    _json_ready,
    _run_pytest,
)
from .phase3_gates import run_pretraining_gates
from .postmortem import (
    NOT_EVALUABLE_REASON,
    PIPELINE_REJECTION_STATUS,
    null_section_8_1_metrics,
)


PHASE4_REPORT_PATH = OUTPUT_DIR / "phase4_integrity_report.json"
SELECTED_CONFIGS_PATH = OUTPUT_DIR / "selected_configs.json"
FINGERPRINT_PATH = OUTPUT_DIR / "forbidden_files_fingerprint.json"
TASK_LOG_PATH = OUTPUT_DIR / "ablation81_run.log"
E6_START = pd.Timestamp("2022-07-01T00:00:00Z")
E6_END = pd.Timestamp("2026-07-09T20:00:00Z")
EXPECTED_POPULATION_ROWS = 11_029
EXPECTED_EVALUATED_ROWS = 9_048
EXPECTED_ARCHIVE_COUNT = 399
E6_COMPARE_ATOL = 1e-12
PINNED_A5_A6_CAVEAT_FA = (
    "پیکربندی‌های منجمد A4 در foldهای ۲ و ۴ مدل تک‌درختی‌اند؛ تفسیر حذف "
    "ویژگی از چنین مدلی محدود است"
)

PHASE3_FROZEN_IDENTITIES = {
    "data/models/ablation81/phase3_integrity_report.json": {
        "size_bytes": 118_751,
        "sha256": "9d44b8fee698553726e12fef4587794b328b8ae34a81c48cdab95cc1d48217f8",
    },
    "data/models/ablation81/selected_configs.json": {
        "size_bytes": 334_483,
        "sha256": "89a54e414058fe5e6649c8f603b3038fd947558776bafc739abc5dbdef083f0b",
    },
    "data/models/ablation81/seed_stability_4h.json": {
        "size_bytes": 178_577,
        "sha256": "8a57f1511791bc697594b39b3f40e0b28de79d9ece3aea2cd9ba1d5cdc26bf44",
    },
    "data/models/ablation81/cscv_predictions_4h.csv": {
        "size_bytes": 57_700_970,
        "sha256": "63b5e9b24f1a357a17e0744280ea2531430b0c3095c11cf8e1ca2f6d0fa85cdf",
    },
    "data/models/ablation81/cscv_matrix_4h.csv": {
        "size_bytes": 41_933,
        "sha256": "3af663d3b8e1885f72292791ec419bd7b80b4648c8001ceffffa5acf76ce7e68",
    },
    "data/models/ablation81/a4_oof_4h.csv": {
        "size_bytes": 814_377,
        "sha256": "f6d912d6a56f63c38058f8a30bfffd64396bf1d718f699294e7f6d8a7597d7a9",
    },
    "data/models/ablation81/owner_adjudication_platt.json": {
        "size_bytes": 12_933,
        "sha256": "ca7e03c46f51e8f61f111c2643dd3f1fc51b23f83f40fb03155b30bee91c834f",
    },
}

PHASE3_FROZEN_SOURCES = {
    "analytics/ablation81/a4.py": (12_184, "a1986290a5aeec26ce9eca99fdf34ed5761ed898d4db79ff84bf4c4ce7f2a951"),
    "analytics/ablation81/cscv.py": (13_319, "f2f83baa98a7e312e313c1f53359292d3b4a291a5fc9291a146f390b450c12c2"),
    "analytics/ablation81/nested.py": (20_142, "21b4a1ec8c469c449af77311070cf52298761b4fab5746b104c10aa475f63a90"),
    "analytics/ablation81/phase3.py": (52_530, "2e7b73e0de11b29c19214594678f56ec599a39f74f2d3f00cba96a772997891c"),
    "analytics/ablation81/phase3_gates.py": (12_009, "0d175426c1880fde51cd538e30afa71c36ce4349251eef227c77819ebf53aa99"),
    "analytics/ablation81/quarantine.py": (3_019, "0477802bb5e7cbe7a92a0b32cc1da262752d6edcca477741b7d96a18a86ebc9f"),
    "analytics/ablation81/adjudication_platt.py": (8_497, "3ad96af10c40f5dd4cefff86705f18b1f9469cbd7a9c5a5de2aec8c3f0c4705e"),
    "analytics/ablation81/phase3_complete.py": (61_880, "6c7079a85979dcd4d6dd29ba4af8623ba65849989de3a1ee6410b75eac71a29c"),
    "analytics/ablation81/postmortem.py": (12_523, "48036da4da79cc62695dd574ee64e4fe5008639f540b0c11916f817a89892b7c"),
}


class Phase4Error(IntegrityError):
    """Raised when phase four departs from its registered construction."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_pair(path: Path) -> dict[str, Any]:
    observed = file_identity(path)
    return {
        "size_bytes": int(observed["size_bytes"]),
        "sha256": str(observed["sha256"]),
    }


def _verify_exact(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = _identity_pair(path)
    normalized = {
        "size_bytes": int(expected["size_bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if observed != normalized:
        raise Phase4Error(
            f"frozen identity changed: {path}; expected={normalized}, observed={observed}"
        )
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        **observed,
        "status": "passed",
    }


def _verify_phase3_frozen() -> dict[str, Any]:
    artifacts = [
        _verify_exact(PROJECT_ROOT / relative, expected)
        for relative, expected in PHASE3_FROZEN_IDENTITIES.items()
    ]
    sources = []
    for relative, (size, digest) in PHASE3_FROZEN_SOURCES.items():
        sources.append(
            _verify_exact(
                PROJECT_ROOT / relative,
                {"size_bytes": size, "sha256": digest},
            )
        )
    return {
        "status": "passed",
        "artifact_count": len(artifacts),
        "source_count": len(sources),
        "artifacts": artifacts,
        "sources": sources,
        "phase3_test_source_note": (
            "the phase3 test identity was verified before the required phase4 "
            "test-11 edit; the evolved test source is hash-recorded and rerun below"
        ),
    }


def _reentry() -> dict[str, Any]:
    environment = verify_environment_fingerprint(
        ENVIRONMENT_PATH,
        collect_environment_fingerprint(thread_count=FROZEN_THREAD_COUNT),
    )
    pinned = verify_pinned_artifacts(PROJECT_ROOT)
    lineage = verify_research_input_lineage(LINEAGE_PATH, base_dir=PROJECT_ROOT)
    production = verify_production_constants()
    s1 = verify_s1_anchors(load_calibrated_events(PROJECT_ROOT))
    phase3 = _verify_phase3_frozen()
    restore = verify_owner_adjudicated_restore(PROJECT_ROOT)
    return {
        "status": "passed",
        "environment": environment,
        "pinned_artifacts": pinned,
        "research_input_lineage": lineage,
        "production_constants": production,
        "s1": s1,
        "phase3_frozen": phase3,
        "backup_restore": restore,
        "original_forbidden_fingerprint_rebuilt_or_deleted": False,
    }


def _load_selected_winners() -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    _verify_exact(SELECTED_CONFIGS_PATH, PHASE3_FROZEN_IDENTITIES[
        "data/models/ablation81/selected_configs.json"
    ])
    document = json.loads(SELECTED_CONFIGS_PATH.read_text(encoding="utf-8"))
    records: dict[int, dict[str, Any]] = {}
    for fold in OUTER_FOLDS:
        record = dict(document["outer_folds"][str(fold)]["frozen_winner"])
        if int(record["tree_count"]) != int(record["best_iteration"]) + 1:
            raise Phase4Error(f"fold {fold} winner tree-count identity failed")
        records[int(fold)] = record
    signature = {
        str(fold): {
            "config_id": records[fold]["config_id"],
            "config": records[fold]["config"],
            "tree_count": int(records[fold]["tree_count"]),
            "best_iteration": int(records[fold]["best_iteration"]),
        }
        for fold in OUTER_FOLDS
    }
    expected = {
        "2": {"config_id": "cfg_062", "tree_count": 1, "best_iteration": 0},
        "3": {"config_id": "cfg_083", "tree_count": 24, "best_iteration": 23},
        "4": {"config_id": "cfg_075", "tree_count": 1, "best_iteration": 0},
        "5": {"config_id": "cfg_009", "tree_count": 13, "best_iteration": 12},
    }
    for key, fixed in expected.items():
        for field, value in fixed.items():
            if signature[key][field] != value:
                raise Phase4Error(f"frozen winner signature changed at {key}/{field}")
    return records, {
        "status": "passed",
        "selection_or_reselection_executed": False,
        "source": "data/models/ablation81/selected_configs.json",
        "outer_folds": signature,
    }


def _frozen_fingerprint_records() -> dict[str, dict[str, Any]]:
    document = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    return {str(item["path"]): dict(item) for item in document["records"]}


def _archive_inputs() -> tuple[dict[str, list[Path]], dict[str, Any]]:
    frozen = _frozen_fingerprint_records()
    by_symbol: dict[str, list[Path]] = {}
    audit_records: list[dict[str, Any]] = []
    for symbol in meta_model.CANONICAL_SYMBOLS:
        directory = PROJECT_ROOT / "data" / "binance_vision" / "klines" / symbol / "4h"
        paths = sorted(directory.glob(f"{symbol}-4h-*.zip"))
        if len(paths) != 57:
            raise Phase4Error(f"E6 archive count changed for {symbol}: {len(paths)}")
        by_symbol[symbol] = paths
        for path in paths:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            expected = frozen.get(relative)
            if expected is None:
                raise Phase4Error(f"E6 archive absent from phase-zero fingerprint: {relative}")
            observed = _identity_pair(path)
            expected_pair = {
                "size_bytes": int(expected["size_bytes"]),
                "sha256": str(expected["sha256"]),
            }
            if observed != expected_pair:
                raise Phase4Error(
                    f"E6 archive byte identity changed: {relative}; "
                    f"expected={expected_pair}, observed={observed}"
                )
            audit_records.append({"path": relative, **observed, "status": "passed"})
    if len(audit_records) != EXPECTED_ARCHIVE_COUNT:
        raise Phase4Error(
            f"E6 total archive count changed: {len(audit_records)}"
        )
    return by_symbol, {
        "status": "passed",
        "archive_count": len(audit_records),
        "per_symbol_count": 57,
        "network_access_attempted": False,
        "alternative_source_used": False,
        "records": audit_records,
    }


def _read_archives(paths: list[Path]) -> pd.DataFrame:
    frames = [binance_vision.read_klines_zip(path) for path in paths]
    if len(frames) != 57:
        raise Phase4Error("E6 archive reader did not consume exactly 57 files")
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[(out.index >= E6_START) & (out.index <= E6_END)].copy()
    if out.empty:
        raise Phase4Error("E6 reconstructed kline frame is empty")
    return out


def _comparison_record(stored: pd.Series, rebuilt: pd.Series, *, label: str) -> dict[str, Any]:
    left = pd.to_numeric(stored, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(rebuilt, errors="coerce").to_numpy(dtype=float)
    if left.shape != right.shape:
        raise Phase4Error(f"E6 {label} shape changed")
    nan_mismatch = np.isnan(left) != np.isnan(right)
    finite = np.isfinite(left) & np.isfinite(right)
    diff = np.abs(left[finite] - right[finite])
    mismatch = int(nan_mismatch.sum()) + int(np.count_nonzero(diff > E6_COMPARE_ATOL))
    if mismatch:
        raise Phase4Error(f"E6 {label} reconstruction mismatch count={mismatch}")
    return {
        "status": "passed",
        "row_count": int(len(left)),
        "finite_pair_count": int(finite.sum()),
        "nan_pattern_mismatch_count": int(nan_mismatch.sum()),
        "mismatch_count_above_tolerance": int(np.count_nonzero(diff > E6_COMPARE_ATOL)),
        "maximum_absolute_difference": float(diff.max(initial=0.0)),
        "absolute_tolerance": E6_COMPARE_ATOL,
    }


def _build_e6_features(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    archive_paths, archive_audit = _archive_inputs()
    raw = {symbol: _read_archives(paths) for symbol, paths in archive_paths.items()}
    btc = raw["BTCUSDT"]
    lookup_parts: list[pd.DataFrame] = []
    comparisons: dict[str, Any] = {}
    for symbol in meta_model.CANONICAL_SYMBOLS:
        causal = dataset.build_causal_features(raw[symbol])
        multi = dataset.build_multitimeframe_features(raw[symbol], timeframe="4h")
        btc_context = dataset.build_btc_context_features(causal, btc)
        rebuilt = pd.DataFrame(
            {
                "OpenTime": raw[symbol].index,
                "close_z_4h": multi["close_z_4h"].reindex(raw[symbol].index).to_numpy(),
                "btc_dist_from_max_168": btc_context["btc_dist_from_max_168"].reindex(raw[symbol].index).to_numpy(),
            }
        )
        stored = pd.read_csv(
            PROJECT_ROOT / "data" / "datasets_4h" / f"{symbol}_4h.csv",
            low_memory=False,
        )
        stored["OpenTime"] = pd.to_datetime(stored["OpenTime"], utc=True, errors="raise")
        joined = stored.loc[:, ["OpenTime", "close_z_4h", "btc_dist_from_max_168"]].merge(
            rebuilt,
            on="OpenTime",
            how="left",
            validate="one_to_one",
            suffixes=("_stored", "_rebuilt"),
        )
        comparisons[symbol] = {
            "close_z_4h": _comparison_record(
                joined["close_z_4h_stored"], joined["close_z_4h_rebuilt"],
                label=f"{symbol}/close_z_4h",
            ),
            "btc_dist_from_max_168": _comparison_record(
                joined["btc_dist_from_max_168_stored"],
                joined["btc_dist_from_max_168_rebuilt"],
                label=f"{symbol}/btc_dist_from_max_168",
            ),
        }
        rebuilt["symbol"] = symbol
        lookup_parts.append(rebuilt)
    lookup = pd.concat(lookup_parts, ignore_index=True)
    keys = events.loc[:, ["symbol", "OpenTime", "decision_ts"]].copy()
    keys["OpenTime"] = pd.to_datetime(keys["OpenTime"], utc=True, errors="raise")
    keys["decision_ts"] = pd.to_datetime(keys["decision_ts"], utc=True, errors="raise")
    joined_events = keys.merge(
        lookup,
        on=["symbol", "OpenTime"],
        how="left",
        validate="one_to_one",
    )
    if len(joined_events) != EXPECTED_POPULATION_ROWS:
        raise Phase4Error("E6 event feature join changed population size")
    feature_values = joined_events.loc[:, ["btc_dist_from_max_168", "close_z_4h"]]
    if not np.isfinite(feature_values.to_numpy(dtype=float)).all():
        raise Phase4Error("E6 event features contain non-finite values")
    ready_ts = joined_events["OpenTime"] + pd.Timedelta(hours=4)
    future = ready_ts > joined_events["decision_ts"]
    if future.any():
        raise Phase4Error(f"E6 P0 future timestamp violations: {int(future.sum())}")
    exact_ready = ready_ts.eq(joined_events["decision_ts"])
    return feature_values.set_axis(events.index), {
        "status": "passed",
        "executable": True,
        "production_imports": [
            "binance_vision.read_klines_zip",
            "dataset.build_causal_features",
            "dataset.build_multitimeframe_features",
            "dataset.build_btc_context_features",
        ],
        "range": {"start": E6_START, "end": E6_END},
        "dataset_reproduction": comparisons,
        "event_population_rows": int(len(joined_events)),
        "event_feature_finite_rows": int(
            np.isfinite(feature_values.to_numpy(dtype=float)).all(axis=1).sum()
        ),
        "p0_lookahead": {
            "status": "passed",
            "feature_ready_semantics": "source_4h_OpenTime + 4h <= decision_ts",
            "future_timestamp_violation_count": int(future.sum()),
            "exact_ready_timestamp_count": int(exact_ready.sum()),
            "rows_scanned": int(len(joined_events)),
        },
        "archive_lineage": archive_audit,
        "network_access_attempted": False,
        "source_substitution_attempted": False,
    }


def _prediction_frame(outer_events: pd.DataFrame, probabilities: np.ndarray, fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": outer_events["symbol"].astype(str).to_numpy(),
            "decision_ts": pd.to_datetime(
                outer_events["decision_ts"], utc=True, errors="raise"
            ).to_numpy(),
            "fold": int(fold),
            "p_raw": np.asarray(probabilities, dtype=float),
        }
    )


def _parse_coefficient(message: str) -> float | None:
    match = re.search(r"coefficient(?:=| must be positive; observed )([^, ]+)", message)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _rowwise_handoff(oof: pd.DataFrame) -> dict[str, Any]:
    fold5 = oof.loc[oof["fold"].eq(5)].copy()
    fold5["decision_ts"] = pd.to_datetime(
        fold5["decision_ts"], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    records = fold5.loc[
        :, ["symbol", "decision_ts", "p_raw", "p_effective", "tau", "traded_A4"]
    ].to_dict(orient="records")
    return {
        "status": "available_for_phase5_reporting_only",
        "row_count": len(records),
        "rows": records,
    }


def _run_model_arm(
    arm: str,
    events: pd.DataFrame,
    X: pd.DataFrame,
    winners: Mapping[int, Mapping[str, Any]],
    pinned_events: pd.DataFrame,
    *,
    economic_cost: float,
) -> dict[str, Any]:
    parts: list[pd.DataFrame] = []
    fold_records: dict[str, Any] = {}
    cpu_total = 0.0
    wall_total = 0.0
    for fold in OUTER_FOLDS:
        frozen = winners[int(fold)]
        spec = build_frozen_arm_spec(
            arm,
            config=frozen["config"],
            tree_count=int(frozen["tree_count"]),
            expected_config=frozen["config"],
            expected_tree_count=int(frozen["tree_count"]),
            base_features=BASE_FEATURES,
            base_categorical=BASE_CATEGORICAL,
        )
        selection = events.loc[events["fold"].lt(fold)].copy()
        outer = events.loc[events["fold"].eq(fold)].copy()
        refit = refit_winner_for_seed(
            selection,
            outer,
            project_arm_frame(X, spec),
            feature_columns=spec["feature_columns"],
            categorical_features=spec["categorical_features"],
            outer_fold=int(fold),
            config=frozen["config"],
            tree_count=int(frozen["tree_count"]),
            seed=42,
        )
        probabilities = refit.pop("probabilities")
        metadata = refit["metadata"]
        cpu_total += float(metadata["fit_cpu_seconds"])
        wall_total += float(metadata["fit_wall_seconds"])
        parts.append(_prediction_frame(outer, probabilities, int(fold)))
        fold_records[str(fold)] = {
            "frozen_config_id": frozen["config_id"],
            "column_spec": spec,
            "refit_metadata": metadata,
        }
    predictions = pd.concat(parts, ignore_index=True)
    if len(predictions) != EXPECTED_EVALUATED_ROWS:
        raise Phase4Error(f"{arm} prediction count changed: {len(predictions)}")
    try:
        pipeline = run_a4_calibration_pipeline(
            pinned_events,
            predictions,
            economic_cost=float(economic_cost),
        )
    except calibration.CalibrationError as exc:
        allowed = assert_allowed_family_rejection_code(exc.code)
        return {
            "arm": arm,
            "pipeline_status": PIPELINE_REJECTION_STATUS,
            "construction_status_fa": "ناموفق در مرحله‌ی ساخت",
            "trade_set_defined": False,
            "calibration_exception": {
                "class": "calibration.CalibrationError",
                "code": str(exc.code),
                "message": str(exc),
                "details": exc.details,
                "parsed_coefficient": _parse_coefficient(str(exc)),
                "allowed_family_code_gate": allowed,
            },
            "fallback_attempted": False,
            "isotonic_substitution_attempted": False,
            "sign_change_attempted": False,
            "raw_forcing_attempted": False,
            "coefficient_floor_attempted": False,
            "fold_deletion_attempted": False,
            "production_code_changed": False,
            "metrics": null_section_8_1_metrics(),
            "funding_affected_fold5_overlap": {
                "affected_population_rows": 103,
                "n_trades": None,
                "sum_net": None,
                "reason": NOT_EVALUABLE_REASON,
            },
            "placebo_sampling_executed": False,
            "rowwise_reporting_handoff": {
                "status": "not_available",
                "reason": NOT_EVALUABLE_REASON,
            },
            "fold_refits": fold_records,
            "timing": {
                "refit_cpu_seconds": cpu_total,
                "refit_wall_seconds": wall_total,
            },
        }
    oof = pipeline.pop("oof")
    return {
        "arm": arm,
        "pipeline_status": "production_pipeline_completed",
        "construction_status_fa": "موفق در مرحله‌ی ساخت",
        "trade_set_defined": True,
        "metrics": pipeline["metrics"],
        "funding_affected_fold5_overlap": pipeline[
            "funding_affected_fold5_overlap"
        ],
        "calibration": pipeline["calibration"],
        "replay": pipeline["replay"],
        "reconstruction": pipeline["reconstruction"],
        "fallback_attempted": False,
        "placebo_sampling_executed": False,
        "rowwise_reporting_handoff": _rowwise_handoff(oof),
        "fold_refits": fold_records,
        "timing": {
            "refit_cpu_seconds": cpu_total,
            "refit_wall_seconds": wall_total,
        },
    }


def _metrics_for_mask(
    prepared: pd.DataFrame,
    selected: pd.Series,
    *,
    score_column: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for scope, folds in REPORT_SCOPES.items():
        population = prepared.loc[prepared["fold"].isin(folds)]
        mask = selected.reindex(population.index)
        if mask.isna().any():
            raise Phase4Error(f"E7 mask alignment failed for {scope}")
        record = trade_metrics(population, mask.astype(bool))
        record["model_metrics"] = model_metrics(population, score_column=score_column)
        report[scope] = record
    return report


def _run_e7(pinned_events: pd.DataFrame, *, economic_cost: float) -> dict[str, Any]:
    work = pinned_events.copy()
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], utc=True, errors="raise")
    work["net_return"] = pd.to_numeric(work["tb_return"], errors="raise") - float(economic_cost)
    evaluated = work["meta_eval_status"].eq("evaluated")
    work["traded_meta"] = False
    expected_value = pd.Series(np.nan, index=work.index, dtype=float)
    fold_records: dict[str, Any] = {}
    for fold in OUTER_FOLDS:
        history = work.loc[work["fold"].lt(fold)]
        wins = history.loc[history["net_return"].gt(0.0), "net_return"]
        losses = history.loc[history["net_return"].le(0.0), "net_return"]
        if wins.empty or losses.empty:
            raise Phase4Error(f"E7 fold {fold} lacks wins or losses in history")
        mu_plus = float(wins.mean())
        mu_minus = float(losses.mean())
        current = evaluated & work["fold"].eq(fold)
        probabilities = pd.to_numeric(
            work.loc[current, "p_primary_cal_platt"], errors="raise"
        )
        if not np.isfinite(probabilities.to_numpy(dtype=float)).all():
            raise Phase4Error(f"E7 fold {fold} primary calibrated score is invalid")
        values = probabilities * mu_plus + (1.0 - probabilities) * mu_minus
        expected_value.loc[current] = values
        work.loc[current, "traded_meta"] = values.gt(0.0).to_numpy(dtype=bool)
        fold_records[str(fold)] = {
            "fold": int(fold),
            "history_folds": sorted(int(item) for item in history["fold"].unique()),
            "history_rows": int(len(history)),
            "history_win_rows": int(len(wins)),
            "history_loss_rows": int(len(losses)),
            "mu_plus_mean_net": mu_plus,
            "mu_minus_mean_net": mu_minus,
            "score": "p_primary_cal_platt",
            "rule": "p*mu_plus + (1-p)*mu_minus > 0",
            "selected_rows": int(work.loc[current, "traded_meta"].sum()),
            "selection_population_uses_only_folds_lt_k": True,
        }
    if expected_value.loc[evaluated].isna().any():
        raise Phase4Error("E7 left evaluated expected values undefined")
    prepared = prepare_evaluated_events(work, economic_cost=float(economic_cost))
    prepared["p_primary_raw_for_model_metrics"] = prepared["p_primary"].astype(float)
    selected = prepared["traded_meta"].astype(bool)
    metrics = _metrics_for_mask(
        prepared, selected, score_column="p_primary_raw_for_model_metrics"
    )
    affected, affected_audit = derive_affected_mask_from_pinned(work)
    affected_traded = affected & work["traded_meta"].astype(bool)
    overlap = {
        "affected_population_rows": int(affected.sum()),
        "n_trades": int(affected_traded.sum()),
        "sum_net": float(work.loc[affected_traded, "net_return"].sum()),
        "affected_mask_audit": affected_audit,
    }
    fold5 = work.loc[evaluated & work["fold"].eq(5)].copy()
    rows = pd.DataFrame(
        {
            "symbol": fold5["symbol"].astype(str),
            "decision_ts": fold5["decision_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "p_primary_cal_platt": fold5["p_primary_cal_platt"].astype(float),
            "expected_value": expected_value.loc[fold5.index].astype(float),
            "traded_E7": fold5["traded_meta"].astype(bool),
        }
    ).to_dict(orient="records")
    return {
        "arm": "E7",
        "pipeline_status": "deterministic_expected_value_threshold_completed",
        "construction_status_fa": "موفق در مرحله‌ی ساخت",
        "trade_set_defined": True,
        "model_training_executed": False,
        "calibration_refit_executed": False,
        "threshold_rule": "p_primary_cal_platt*mu_plus + (1-p_primary_cal_platt)*mu_minus > 0",
        "history_includes_fold1": True,
        "fold_records": fold_records,
        "metrics": metrics,
        "funding_affected_fold5_overlap": overlap,
        "placebo_sampling_executed": False,
        "rowwise_reporting_handoff": {
            "status": "available_for_phase5_reporting_only",
            "row_count": len(rows),
            "rows": rows,
        },
    }


def _phase4_x(arm: str, base_X: pd.DataFrame, events: pd.DataFrame, e6: pd.DataFrame) -> pd.DataFrame:
    X = base_X.copy()
    if tuple(X.columns) != BASE_FEATURES:
        raise Phase4Error("canonical X schema changed before phase-four transforms")
    if arm == "E3":
        X["session"] = materialize_session(events["decision_ts"]).astype(str)
    elif arm == "E6":
        X["btc_dist_from_max_168"] = e6["btc_dist_from_max_168"].astype(float)
        X["close_z_4h"] = e6["close_z_4h"].astype(float)
    return X


def run_phase4() -> dict[str, Any]:
    """Execute phase four once and leave phase five untouched."""

    if PHASE4_REPORT_PATH.exists():
        raise Phase4Error("refusing to overwrite frozen phase4 report")
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    started = _utc_now()
    _append_task_log({"event": "phase4_started", "thread_count": FROZEN_THREAD_COUNT})
    reentry_before = _reentry()
    assembly, pretraining = run_pretraining_gates(PROJECT_ROOT)
    if pretraining["population_alignment"]["key_membership_counts"] != {
        "both": 11_029,
        "left_only": 0,
        "right_only": 0,
    }:
        raise Phase4Error("phase4 population gate signature changed")
    tests_before = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    winners, winner_gate = _load_selected_winners()
    events = assembly["events"].copy()
    events["fold"] = pd.to_numeric(events["fold"], errors="raise").astype(int)
    events["decision_ts"] = pd.to_datetime(events["decision_ts"], utc=True, errors="raise")
    if len(events) != EXPECTED_POPULATION_ROWS:
        raise Phase4Error("phase4 assembled row count changed")
    base_X = assembly["X"].copy()
    if tuple(assembly["feature_columns"]) != BASE_FEATURES:
        raise Phase4Error("phase4 base feature schema changed")
    if tuple(assembly["categorical_features"]) != BASE_CATEGORICAL:
        raise Phase4Error("phase4 base categorical schema changed")
    e6_values, e6_audit = _build_e6_features(events)
    pinned = load_calibrated_events(PROJECT_ROOT)
    if len(pinned) != EXPECTED_POPULATION_ROWS:
        raise Phase4Error("pinned row count changed before phase4 fitting")

    arms: dict[str, Any] = {}
    for arm in MODEL_ARMS:
        arm_cpu = time.process_time()
        arm_wall = time.perf_counter()
        _append_task_log({"event": "phase4_arm_started", "arm": arm})
        X_arm = _phase4_x(arm, base_X, events, e6_values)
        result = _run_model_arm(
            arm,
            events,
            X_arm,
            winners,
            pinned,
            economic_cost=float(assembly["metadata"]["cost_round_trip"]),
        )
        result["timing"]["arm_total_cpu_seconds"] = time.process_time() - arm_cpu
        result["timing"]["arm_total_wall_seconds"] = time.perf_counter() - arm_wall
        if arm in {"A5", "A6"}:
            result["pinned_interpretation_limitation_fa"] = PINNED_A5_A6_CAVEAT_FA
        arms[arm] = result
        _append_task_log(
            {
                "event": "phase4_arm_completed",
                "arm": arm,
                "pipeline_status": result["pipeline_status"],
                "timing": result["timing"],
            }
        )
    e7_cpu = time.process_time()
    e7_wall = time.perf_counter()
    arms["E7"] = _run_e7(
        pinned,
        economic_cost=float(assembly["metadata"]["cost_round_trip"]),
    )
    arms["E7"]["timing"] = {
        "cpu_seconds": time.process_time() - e7_cpu,
        "wall_seconds": time.perf_counter() - e7_wall,
    }

    tests_after = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    reentry_after = _reentry()
    source_identities = {
        relative: {
            "path": relative,
            **_identity_pair(PROJECT_ROOT / relative),
        }
        for relative in (
            "analytics/ablation81/adjudication_restore.py",
            "analytics/ablation81/feature_ablation.py",
            "analytics/ablation81/phase4.py",
            "tests/test_ablation81.py",
        )
    }
    completed = _utc_now()
    report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 4,
        "status": "completed",
        "phase4_completed": True,
        "phase5_mechanism_executed": False,
        "next_phase_requires_separate_owner_continue_message": True,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "owner_backup_restore_adjudication": reentry_after["backup_restore"],
        "reentry_before_scientific_work": reentry_before,
        "reentry_after_scientific_work": reentry_after,
        "pretraining_hard_gates": pretraining,
        "frozen_winner_gate": winner_gate,
        "test_11": {
            "status": "implemented_and_green",
            "name": "test_11_frozen_config_ablation_changes_only_columns",
            "purpose": (
                "prove every model arm changes only registered columns while "
                "config, tree count, no-search, no-eval-set, and no-early-stopping remain frozen"
            ),
        },
        "pytest": {"before_scientific_work": tests_before, "after_scientific_work": tests_after},
        "e6_exact_production_reconstruction": e6_audit,
        "scientific_results": {
            "required_arms": ["A5", "A6", "E3", "E4", "E5", "E6", "E7"],
            "all_required_arms_executed": set(arms) == {"A5", "A6", "E3", "E4", "E5", "E6", "E7"},
            "E8_status": "disabled_not_activated_by_owner",
            "arms": arms,
            "A5_A6_pinned_interpretation_limitation_fa": PINNED_A5_A6_CAVEAT_FA,
            "placebo_sampling_executed_in_phase4": False,
            "search_or_reselection_executed_in_phase4": False,
            "joblib_artifacts_deserialized_or_scored": False,
        },
        "phase5_reporting_handoff": {
            "purpose": (
                "rowwise fold5 carry-forward only; no sensitivity, placebo, PBO, "
                "decision-rule, or other phase5 calculation was executed"
            ),
            "owner_required_103_row_exclusion_calculated_now": False,
            "arms": {
                arm: result["rowwise_reporting_handoff"]
                for arm, result in arms.items()
            },
        },
        "deviations": [
            {
                "origin_phase": 4,
                "status": "accepted_by_owner_before_phase4_start",
                "text_fa": (
                    "مالک امضای دقیق بازیابی بکاپ را، شامل ۱۹۸۳ اختلاف فقط-mtime "
                    "و یک تغییر .pytest_cache/v/cache/nodeids، برای آغاز فاز چهار "
                    "نادیده گرفت؛ اثر انگشت اصلی بازسازی یا حذف نشد و این حکم "
                    "هیچ استثنای عمومی ایجاد نمی‌کند."
                ),
                "owner_message_verbatim_fa": (
                    "از نظر من اشکالی ندارد. اینها مشکل ساز نیستند. این موارد را "
                    "نادیده بگیر. فاز ۴ را شروع کن."
                ),
                "exact_owner_artifact": reentry_after["backup_restore"]["owner_artifact"],
            },
            {
                "origin_phase": 4,
                "status": "registered_supporting_artifact",
                "text_fa": (
                    "فایل phase4_integrity_report.json گزارش فازی پشتیبان است و "
                    "در محاسبات علمی مصرف نمی‌شود."
                ),
            },
        ],
        "execution_incidents": [
            {
                "origin_phase": 4,
                "status": "corrected_before_any_artifact_or_result_was_produced_or_consumed",
                "announced_verbatim_fa": (
                    "اجرای نخست در همان شروع A5 و پیش از ساخت هر artifact علمی متوقف شد: "
                    "لایه‌ی جدید فاز ۴، هنگام آموزش، DataFrame کامل A4 را به تابع refit "
                    "داده بود و فقط هنگام پیش‌بینی زیرمجموعه‌ی ستون‌ها را اعمال می‌کرد؛ "
                    "در نتیجه جایگاه ستون categorical ناسازگار شد. این خطا مربوط به "
                    "wrapper تازه‌ی فاز ۴ است، نه داده یا کد تولید. اصلاح را فقط به "
                    "همان wrapper محدود می‌کنم، رخداد را در JSON فاز ۴ ثبت می‌کنم و "
                    "تست‌ها و دروازه‌ها را دوباره اجرا می‌کنم."
                ),
                "failure": (
                    "CatBoost prediction interpreted symbol as numeric because the "
                    "training Pool had received the unprojected base frame"
                ),
                "correction_scope": (
                    "phase4 wrapper now projects X to the registered arm feature list "
                    "before both training Pool construction and prediction"
                ),
                "production_code_changed": False,
                "frozen_input_changed": False,
                "scientific_artifact_produced": False,
                "scientific_result_consumed": False,
                "search_or_reselection_executed": False,
            }
        ],
        "self_audit_doubts": [
            {
                "topic": "single_tree_interpretation",
                "status": "open_interpretive_limitation",
                "text_fa": PINNED_A5_A6_CAVEAT_FA,
            },
            {
                "topic": "production_pipeline_rejection_is_arm_specific",
                "status": "resolved_by_owner_rule",
                "text_fa": (
                    "رد کالیبراسیون هر بازو مستقل طبقه‌بندی شده و هیچ fallback، "
                    "تغییر علامت، حذف fold یا نجات خودکار انجام نشده است."
                ),
            },
            {
                "topic": "E6_exactness",
                "status": "resolved_by_full_reproduction_and_P0_gate",
                "text_fa": (
                    "دو ویژگی E6 از ۳۹۹ آرشیو محلی منجمد بازسازی، علیه تمام "
                    "ردیف‌های هفت دیتاست مقایسه و از نظر زمان آماده‌شدن ممیزی شدند."
                ),
            },
            {
                "topic": "backup_restore_metadata",
                "status": "narrow_owner_adjudication_only",
                "text_fa": (
                    "پذیرش اختلاف بازیابی فقط به امضای دقیق ثبت‌شده محدود است؛ "
                    "هر اختلاف دیگری توقف‌ساز می‌ماند."
                ),
            },
        ],
        "supporting_artifacts": [
            {
                "path": "data/models/ablation81/phase4_integrity_report.json",
                "role": "phase4_integrity_and_scientific_report",
                "frozen_once_written": True,
                "consumed_in_scientific_calculation": False,
            },
            reentry_after["backup_restore"]["owner_artifact"],
        ],
        "new_source_identities": source_identities,
        "timing": {
            "phase_cpu_seconds": time.process_time() - cpu_start,
            "phase_wall_seconds": time.perf_counter() - wall_start,
            "arm_timing": {
                arm: result["timing"] for arm, result in arms.items()
            },
        },
        "completion_boundary": (
            "phase four is complete; phase five has not started and requires a separate owner continue message"
        ),
    }
    write_json_once(PHASE4_REPORT_PATH, _json_ready(report))
    report_identity = _identity_pair(PHASE4_REPORT_PATH)
    loaded = json.loads(PHASE4_REPORT_PATH.read_text(encoding="utf-8"))
    if loaded.get("status") != "completed" or loaded.get("phase") != 4:
        raise Phase4Error("phase4 report readback verification failed")
    _append_task_log(
        {
            "event": "phase4_completed",
            "phase4_report": {
                "path": PHASE4_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                **report_identity,
            },
            "phase5_mechanism_executed": False,
        }
    )
    return {"report": report, "report_identity": report_identity}


if __name__ == "__main__":
    result = run_phase4()
    summary = {
        "status": result["report"]["status"],
        "phase": 4,
        "report": {
            "path": PHASE4_REPORT_PATH.relative_to(PROJECT_ROOT).as_posix(),
            **result["report_identity"],
        },
        "arm_statuses": {
            arm: record["pipeline_status"]
            for arm, record in result["report"]["scientific_results"]["arms"].items()
        },
        "phase5_started": False,
    }
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True))
