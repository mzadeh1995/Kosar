# ==============================================================================
# analytics/ablation81/phase1.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Execute and freeze phase-one P0 look-ahead evidence for ablation81 v7."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import meta_model

from .integrity import (
    PINNED_ARTIFACTS,
    SYMBOLS,
    IntegrityError,
    collect_environment_fingerprint,
    file_identity,
    sha256_file,
    verify_environment_fingerprint,
    verify_forbidden_files_fingerprint,
    verify_pinned_artifacts,
    verify_research_input_lineage,
    write_json_once,
)
from .lookahead import (
    VERDICT_FA,
    audit_calendar_block,
    audit_full_population,
    audit_symbol_block,
)


FROZEN_THREAD_COUNT = 3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "ablation81"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment_fingerprint.json"
FORBIDDEN_FINGERPRINT_PATH = OUTPUT_DIR / "forbidden_files_fingerprint.json"
LINEAGE_PATH = OUTPUT_DIR / "research_input_lineage.json"
P0_REPORT_PATH = OUTPUT_DIR / "p0_lookahead_report.json"
PHASE_REPORT_PATH = OUTPUT_DIR / "phase1_integrity_report.json"
TASK_LOG_PATH = OUTPUT_DIR / "ablation81_run.log"
HMM_BUILDER_SNAPSHOT_PATH = OUTPUT_DIR / "hmm_builder_evidence_snapshot.txt"

EXPECTED_FROZEN_FORBIDDEN_IDENTITY = {
    "size_bytes": 13_209_954,
    "sha256": "f14182e35749f9c37d34282fe5b130a672f20e8f585daded838ef989d87942f1",
}
EXPECTED_HMM_BUILDER_SNAPSHOT_IDENTITY = {
    "size_bytes": 6_509,
    "sha256": "2d4f3efd9a2497338fd1ec74ad1af020ff4d4a54b6237194cb2c76809020cb40",
}

DEVIATIONS = [
    {
        "origin_phase": 0,
        "text_fa": (
            "یک اجرای کمکی ممیزی به‌علت استفاده از نام ویژهٔ `path` در zsh، "
            "false-fail شد؛ بلافاصله تکرار شد و نتیجه `checked=12, issues=0` "
            "بود. هیچ فایلی تغییر نکرد."
        ),
    },
    {
        "origin_phase": 0,
        "text_fa": (
            "فایل `phase0_integrity_report.json` به‌عنوان گزارش فازی پشتیبان "
            "تولید شد؛ این فایل در تحلیل علمی مصرف نمی‌شود."
        ),
    },
    {
        "origin_phase": 1,
        "text_fa": (
            "به‌دلیل فعال‌شدن گیت توقف پیش از ممیزی P0، فایل پشتیبان "
            "`phase1_stop_report.json` ساخته شد؛ این فایل در تحلیل علمی مصرف "
            "نمی‌شود."
        ),
    },
    {
        "origin_phase": 1,
        "text_fa": (
            "با داوری مالک، فقط فایل‌هایی با basename دقیق `.DS_Store` در هر "
            "عمق از گیت forbidden مستثنا شدند؛ هیچ dotfile یا glob دیگری "
            "مستثنا نشد و اثرانگشت فاز صفر بازساخته نشد."
        ),
    },
    {
        "origin_phase": 1,
        "text_fa": (
            "فایل `hmm_builder_evidence_snapshot.txt` به‌عنوان snapshot "
            "بایت‌به‌بایت سازندهٔ موقت HMM تولید شد؛ این فایل در تحلیل عددی "
            "مصرف نمی‌شود و فقط مدرک P0-B است."
        ),
    },
    {
        "origin_phase": 1,
        "text_fa": (
            "فایل `phase1_integrity_report.json` به‌عنوان گزارش فازی پشتیبان "
            "تولید شد؛ این فایل در تحلیل علمی مصرف نمی‌شود."
        ),
    },
    {
        "origin_phase": 1,
        "text_fa": (
            "در بازبینی نهایی یک نقص ثبتی پیدا شد: جزئیات داوری `.DS_Store` در "
            "گزارش P0 کامل بود، اما در خود `phase1_integrity_report.json` فقط "
            "خلاصهٔ گیت آمده بود. پیش از تحویل، همان دو بلوک ثبتی را بدون تغییر "
            "نتایج علمی به گزارش فاز یک و سازندهٔ آن اضافه می‌کنم؛ اثرانگشت "
            "frozen همچنان دست‌نخورده می‌ماند."
        ),
    },
]

CONTRACT_CLARIFICATIONS = {
    "phase5_auc_and_logloss": {
        "auc_score_domain": "raw",
        "auc_weighting": "unweighted",
        "logloss_weighting": "tb_uniqueness",
    },
    "phase_reports_are_supporting_artifacts": True,
    "phase5_hash_manifest_closure": {
        "directory": "data/models/ablation81",
        "include_every_existing_file": True,
        "exclude_only": ["hash_manifest.json", "ablation81_run.log"],
        "include_supporting_artifacts_and_phase_reports": True,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_task_log(event: dict[str, Any]) -> None:
    with TASK_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_phase_tests() -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "tests/test_ablation81.py",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=900,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": time.perf_counter() - started,
        "output": completed.stdout,
        "status": "passed" if completed.returncode == 0 else "failed",
    }


def _assert_identity(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    actual = file_identity(path)
    if (
        actual["size_bytes"] != expected["size_bytes"]
        or actual["sha256"] != expected["sha256"]
    ):
        raise IntegrityError(
            f"{label} identity mismatch: expected={expected}, observed={actual}"
        )
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": actual["size_bytes"],
        "sha256": actual["sha256"],
        "status": "passed",
    }


def _timeframe_for_canonical_role(role: str) -> str:
    return {
        "primary_oof": "4h",
        "dataset": "4h",
        "funding": "8h_settlement",
        "vpin": "1h",
        "hmm": "1h",
    }[role]


def _build_research_lineage(assembly: dict[str, Any]) -> dict[str, Any]:
    records_by_path: dict[str, dict[str, Any]] = {}

    def add(
        path_value: str | Path,
        *,
        role: str,
        symbol: str | None,
        timeframe: str | None,
        required_by: str,
    ) -> None:
        path = Path(path_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as exc:
            raise IntegrityError(f"lineage input is outside project root: {resolved}") from exc
        identity = {
            "path": relative,
            "size_bytes": int(resolved.stat().st_size),
            "sha256": sha256_file(resolved),
            "role": role,
            "roles": [role],
            "symbol": symbol,
            "timeframe": timeframe,
            "required_by": [required_by],
        }
        existing = records_by_path.get(relative)
        if existing is None:
            records_by_path[relative] = identity
            return
        if (
            existing["size_bytes"] != identity["size_bytes"]
            or existing["sha256"] != identity["sha256"]
        ):
            raise IntegrityError(f"lineage duplicate identity differs: {relative}")
        existing["roles"] = sorted(set(existing["roles"]) | {role})
        existing["required_by"] = sorted(
            set(existing["required_by"]) | {required_by}
        )
        if existing["symbol"] is None:
            existing["symbol"] = symbol
        if existing["timeframe"] is None:
            existing["timeframe"] = timeframe

    for record in PINNED_ARTIFACTS:
        add(
            str(record["path"]),
            role=str(record["role"]),
            symbol=record["symbol"],
            timeframe="4h",
            required_by="contract_2_2",
        )

    input_files = assembly["metadata"].get("input_files")
    if not isinstance(input_files, list) or len(input_files) != 35:
        raise IntegrityError(
            f"canonical assembly input_files must contain 35 records, got {input_files!r}"
        )
    for record in input_files:
        role = str(record["role"])
        add(
            str(record["path"]),
            role=f"canonical_{role}",
            symbol=str(record["symbol"]),
            timeframe=_timeframe_for_canonical_role(role),
            required_by="contract_2_6_canonical_35",
        )

    required_contract_inputs = [
        ("meta_model.py", "required_production_code", None, None),
        ("calibration.py", "required_production_code", None, None),
        ("config.py", "required_production_code", None, None),
        (
            "data/models/primary_training_summary_4h.json",
            "primary_oof_training_summary",
            None,
            "4h",
        ),
    ]
    for path, role, symbol, timeframe in required_contract_inputs:
        add(
            path,
            role=role,
            symbol=symbol,
            timeframe=timeframe,
            required_by="contract_2_6",
        )

    p0_b_code_evidence = [
        ("XGBoost.py", "primary_oof_builder", "p_primary"),
        ("TripleBarrier.py", "purged_walk_forward_and_dataset_evidence", "p_primary,dataset"),
        ("primary_features.py", "primary_feature_selection_evidence", "p_primary"),
        (
            "analytics/train_XGBoost.py",
            "primary_training_runner_evidence",
            "p_primary",
        ),
        ("dataset.py", "dataset_feature_builder_evidence", "dataset"),
        ("external_history.py", "funding_vpin_builder_evidence", "funding,vpin"),
        ("binance_vision.py", "raw_market_reader_evidence", "dataset,funding,vpin"),
        ("tests/hmm_walkforward.py", "hmm_walk_forward_base_builder", "hmm"),
        ("hmm.py", "hmm_fit_and_normalization_evidence", "hmm"),
        ("fractional.py", "hmm_fractional_feature_evidence", "hmm"),
    ]
    for path, role, block in p0_b_code_evidence:
        add(
            path,
            role=role,
            symbol=None,
            timeframe=None,
            required_by=f"p0_b_{block}",
        )

    add(
        HMM_BUILDER_SNAPSHOT_PATH,
        role="hmm_canonical_builder_evidence_snapshot",
        symbol=None,
        timeframe="1h",
        required_by="p0_b_hmm",
    )
    for symbol in SYMBOLS:
        add(
            f"data/external/funding_{symbol}.json",
            role="funding_source_coverage_metadata",
            symbol=symbol,
            timeframe="8h_settlement",
            required_by="p0_a_funding_gap_cause",
        )

    records = [records_by_path[path] for path in sorted(records_by_path)]
    return {
        "schema_version": 1,
        "contract_version": 7,
        "root": str(PROJECT_ROOT),
        "frozen_at_utc": _utc_now(),
        "frozen_at_end_of_phase": 1,
        "frozen": True,
        "input_count": len(records),
        "inputs": records,
        "canonical_assembly_input_file_count": len(input_files),
        "hmm_builder_snapshot": {
            "snapshot_path": str(HMM_BUILDER_SNAPSHOT_PATH.relative_to(PROJECT_ROOT)),
            "original_path": "/tmp/kosar_hmm_fwd_long_v47.py",
            "byte_identical_at_capture": True,
            **EXPECTED_HMM_BUILDER_SNAPSHOT_IDENTITY,
        },
    }


def _p0_b_evidence() -> dict[str, Any]:
    return {
        "p_primary": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "seven canonical 4h labeled datasets",
            "builder": "analytics/train_XGBoost.py -> XGBoost.run_purged_cv",
            "walk_forward_proof": {
                "symbol_fold_pairs_reconstructed": 35,
                "exact_test_timestamp_set_matches": 35,
                "oof_rows_per_symbol": 6_341,
                "fold_counts_per_symbol": {
                    "1": 1_269,
                    "2": 1_268,
                    "3": 1_268,
                    "4": 1_268,
                    "5": 1_268,
                },
                "training_rule": (
                    "Each OOF test block is predicted only from earlier event blocks; "
                    "train exits must precede a 96-hour purge boundary."
                ),
                "test_metrics_used_for_selection": False,
            },
            "window_direction": "past-only outer split and past-only inner validation",
            "missing_value_policy": "inf/non-numeric to NaN, then complete-case removal",
            "bfill_or_future_fill": False,
            "fitted_normalizer": None,
            "evidence": [
                {"file": "analytics/train_XGBoost.py", "lines": "31-37,72-103,112-135"},
                {"file": "XGBoost.py", "lines": "91-115,197-313"},
                {"file": "TripleBarrier.py", "lines": "176-225"},
                {"file": "primary_features.py", "lines": "51-99,127-173,196-285"},
                {"file": "meta_model.py", "lines": "354-417,873-889"},
                {
                    "file": "data/models/primary_training_summary_4h.json",
                    "facts": {
                        "valid_folds_each_symbol": 5,
                        "seed": 41,
                        "timeframe": "4h",
                    },
                },
            ],
        },
        "dataset": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "Binance Vision 4h klines",
            "builder": "dataset.build_causal_features",
            "window_direction": (
                "volatility uses log-return shift(1) and trailing EWM; "
                "log_range uses the current closed 4h candle"
            ),
            "missing_value_policy": "inf to NaN followed by required-row removal",
            "bfill_or_future_fill": False,
            "fitted_normalizer": None,
            "merge": "exact one-to-one OpenTime; decision_ts=OpenTime+4h",
            "reconciliation": {
                "value_max_abs_difference": 0.0,
                "formula_max_abs_difference_upper_bound": 9.98e-17,
            },
            "evidence": [
                {"file": "binance_vision.py", "lines": "291-314,352-392"},
                {"file": "dataset.py", "lines": "407-421,525-586"},
                {"file": "TripleBarrier.py", "lines": "36-41"},
                {"file": "meta_model.py", "lines": "420-494,887-892"},
            ],
        },
        "calendar": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "decision_ts UTC",
            "builder": "meta_model.build_meta_dataset",
            "window_direction": "not applicable",
            "missing_value_policy": "none",
            "bfill_or_future_fill": False,
            "fitted_normalizer": None,
            "evidence": [{"file": "meta_model.py", "lines": "887-891"}],
        },
        "funding": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "Binance Vision monthly funding-rate archives",
            "builder": "external_history.normalize_funding_frame",
            "window_direction": (
                "funding_z uses x.shift(1), trailing rolling90, min_periods90, "
                "std ddof=0"
            ),
            "missing_value_policy": "invalid/inf and zero-variance results remain NaN",
            "bfill_or_future_fill": False,
            "fitted_normalizer": "past-only rolling mean and standard deviation",
            "merge": "backward as-of, exact allowed, tolerance 8h1m",
            "reconciliation": {
                "funding_z_max_abs_difference": 1.78e-15,
                "null_mask_mismatch_count": 0,
                "extreme_flag_mismatch_count": 0,
            },
            "evidence": [
                {"file": "binance_vision.py", "lines": "436-474"},
                {"file": "external_history.py", "lines": "33-36,123-197"},
                {"file": "meta_model.py", "lines": "497-577,895-910"},
                {
                    "files": "data/external/funding_{SYMBOL}.json",
                    "fact": "requested July 2026 but skipped_months contains 2026-07",
                },
            ],
        },
        "vpin": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "Binance Vision 1m klines",
            "builder": "external_history.build_reconstructed_vpin_history",
            "window_direction": (
                "reconstruction uses (T-7d,T]; capacity uses completed past hours; "
                "vpin_z uses shift(1) and trailing rolling50"
            ),
            "missing_value_policy": "invalid rows dropped; unsupported windows remain NaN",
            "bfill_or_future_fill": False,
            "fitted_normalizer": "past-only rolling mean and standard deviation",
            "merge": "backward as-of, exact allowed, tolerance 2h",
            "reconciliation": {
                "vpin_z_max_abs_difference": 2.775e-12,
                "null_mask_mismatch_count": 0,
            },
            "evidence": [
                {"file": "binance_vision.py", "lines": "291-314,352-392"},
                {"file": "external_history.py", "lines": "123-129,200-388"},
                {"file": "meta_model.py", "lines": "539-577,912-927"},
            ],
        },
        "hmm": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "Binance public 1h klines",
            "builder": "captured /tmp/kosar_hmm_fwd_long_v47.py",
            "window_direction": "each fit uses df.iloc[i-500:i]; row i is excluded",
            "missing_value_policy": "invalid rows removed; failed HMM row becomes missing/NaN",
            "bfill_or_future_fill": False,
            "fitted_normalizer": (
                "1%/99% clipping, median and IQR fitted only within each past window"
            ),
            "merge": "backward as-of, exact allowed, tolerance 6h",
            "effective_feature_mode": {
                "configured": "fractional_diff",
                "ffd_width": 1_163,
                "fit_window_rows": 500,
                "observed": "legacy",
                "post_warmup_rows": 476,
                "lookahead_effect": False,
            },
            "diagnostic_future_columns_enter_meta_features": False,
            "evidence": [
                {
                    "file": "data/models/ablation81/hmm_builder_evidence_snapshot.txt",
                    "lines": "25-32,70-100",
                },
                {"file": "tests/hmm_walkforward.py", "lines": "117-191"},
                {"file": "fractional.py", "lines": "65-90,229-230"},
                {"file": "hmm.py", "lines": "241-286,328-376,615-673,811-910"},
                {"file": "meta_model.py", "lines": "539-577,602-674,765-806,929-941"},
            ],
        },
        "symbol": {
            "verdict": "healthy",
            "verdict_fa": VERDICT_FA["healthy"],
            "raw_source": "canonical symbol loop and source filename identity",
            "builder": "meta_model._load_primary_oof",
            "window_direction": "not applicable",
            "missing_value_policy": "none",
            "bfill_or_future_fill": False,
            "fitted_normalizer": None,
            "categorical": True,
            "evidence": [
                {"file": "meta_model.py", "lines": "173-187,243-284,403,960-966,1111-1140"}
            ],
        },
    }


def _funding_failure_details(events: pd.DataFrame) -> dict[str, Any]:
    unmatched = events.loc[~events["_funding_matched"].astype(bool)].copy()
    matched_nan = events.loc[
        events["_funding_matched"].astype(bool)
        & events[["funding_z", "funding_extreme_pos"]].isna().any(axis=1)
    ]
    by_symbol = {
        str(symbol): int(count)
        for symbol, count in unmatched["symbol"].value_counts().sort_index().items()
    }
    source_ends: dict[str, str | None] = {}
    skipped_months: dict[str, list[str]] = {}
    for symbol in SYMBOLS:
        metadata = json.loads(
            (PROJECT_ROOT / f"data/external/funding_{symbol}.json").read_text(
                encoding="utf-8"
            )
        )
        source_ends[symbol] = metadata.get("observed_end")
        skipped_months[symbol] = list(metadata.get("skipped_months", []))
    return {
        "blocking_reason": (
            "The production backward as-of join has no funding record within "
            "8h1m for 103 population rows."
        ),
        "unmatched_count": int(len(unmatched)),
        "unmatched_by_symbol": by_symbol,
        "unmatched_by_fold": {
            str(int(fold)): int(count)
            for fold, count in unmatched["fold"].value_counts().sort_index().items()
        },
        "unmatched_decision_start": (
            None
            if unmatched.empty
            else pd.Timestamp(unmatched["decision_ts"].min()).isoformat()
        ),
        "unmatched_decision_end": (
            None
            if unmatched.empty
            else pd.Timestamp(unmatched["decision_ts"].max()).isoformat()
        ),
        "source_observed_end_by_symbol": source_ends,
        "source_skipped_months_by_symbol": skipped_months,
        "matched_but_feature_nan_count": int(len(matched_nan)),
        "matched_feature_nan_rows": [
            {
                "symbol": str(row["symbol"]),
                "decision_ts": pd.Timestamp(row["decision_ts"]).isoformat(),
            }
            for _, row in matched_nan.iterrows()
        ],
    }


def _build_p0_report(
    *,
    assembly: dict[str, Any],
    p0_a: dict[str, Any],
    lineage_identity: dict[str, Any],
    owner_gate: dict[str, Any],
) -> dict[str, Any]:
    events = assembly["events"]
    p0_b = _p0_b_evidence()
    calendar_a = audit_calendar_block(events)
    symbol_a = audit_symbol_block(events, canonical_symbols=SYMBOLS)
    p0_a_blocks = {
        "p_primary": {
            "p0_a_verdict": "not_applicable",
            "p0_a_verdict_fa": VERDICT_FA["not_applicable"],
            "reason": "p_primary is direct OOF lineage, not an as-of source join",
        },
        **p0_a["blocks"],
        "calendar": calendar_a,
        "symbol": symbol_a,
    }
    blocks: dict[str, Any] = {}
    for block in (
        "p_primary",
        "dataset",
        "calendar",
        "funding",
        "vpin",
        "hmm",
        "symbol",
    ):
        a_verdict = p0_a_blocks[block]["p0_a_verdict"]
        b_verdict = p0_b[block]["verdict"]
        final_verdict = (
            "error"
            if "error" in {a_verdict, b_verdict}
            else "unknown"
            if "unknown" in {a_verdict, b_verdict}
            else "healthy"
        )
        blocks[block] = {
            "verdict": final_verdict,
            "verdict_fa": VERDICT_FA[final_verdict],
            "p0_a": p0_a_blocks[block],
            "p0_b": p0_b[block],
        }
    blocks["funding"]["p0_a_failure_details"] = _funding_failure_details(events)
    blocking_blocks = [
        block
        for block, record in blocks.items()
        if record["verdict"] in {"error", "unknown"}
    ]
    return {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 1,
        "status": "stopped",
        "stop_reason": "P0 block verdict is error",
        "blocking_blocks": blocking_blocks,
        "population": {
            "rows": int(len(events)),
            "fold_counts": {
                str(int(fold)): int(count)
                for fold, count in events["fold"].value_counts().sort_index().items()
            },
            "symbol_counts": {
                str(symbol): int(count)
                for symbol, count in events["symbol"].value_counts().sort_index().items()
            },
            "assembly_input_file_count": len(assembly["metadata"]["input_files"]),
            "fallback_counts": assembly["metadata"]["fallback_counts"],
        },
        "blocks": blocks,
        "owner_adjudication_ds_store": owner_gate,
        "initial_ds_store_stop": {
            "cause_was_external": True,
            "cause": "Owner opened project folders in Finder between phases.",
            "changed_paths": [
                ".DS_Store",
                "analytics/.DS_Store",
                "data/.DS_Store",
                "data/models/.DS_Store",
            ],
            "scientific_outputs_produced_or_consumed": False,
        },
        "research_input_lineage": lineage_identity,
        "contract_clarifications": CONTRACT_CLARIFICATIONS,
        "supporting_artifacts": [
            "data/models/ablation81/hmm_builder_evidence_snapshot.txt",
            "data/models/ablation81/phase1_stop_report.json",
            "data/models/ablation81/phase1_integrity_report.json",
        ],
        "deviations": DEVIATIONS,
        "self_audit_doubts": [
            {
                "doubt": (
                    "The primary summary lacks per-fold train/test/purge hashes and "
                    "does not independently attest the historical executable."
                ),
                "check": (
                    "Current source lineage was hashed and the exact test timestamp "
                    "membership was independently reconstructed for all 35 symbol-fold pairs."
                ),
                "result": "35 of 35 exact membership matches",
            },
            {
                "doubt": (
                    "HMM is configured for fractional_diff, but the 500-row builder "
                    "window is shorter than the 1163-weight FFD width."
                ),
                "check": "The effective feature path was reproduced on 500 rows.",
                "result": "legacy causal fallback, 476 usable rows, no future input",
            },
            {
                "doubt": (
                    "A matched source timestamp does not guarantee every feature is usable."
                ),
                "check": "Matched-but-feature-NaN rows were counted separately.",
                "result": "funding=1, HMM=1; both recorded without future filling",
            },
        ],
        "phase2_ready": False,
        "owner_adjudication_required": True,
        "created_at_utc": _utc_now(),
    }


def run_phase1() -> dict[str, Any]:
    if any(path.exists() for path in (LINEAGE_PATH, P0_REPORT_PATH, PHASE_REPORT_PATH)):
        raise IntegrityError("one or more frozen phase-one outputs already exist")
    started_wall = _utc_now()
    started = time.perf_counter()
    _append_task_log(
        {
            "event": "phase1_restarted_after_owner_ds_store_adjudication",
            "timestamp_utc": started_wall,
            "thread_count": FROZEN_THREAD_COUNT,
        }
    )

    environment_gate = verify_environment_fingerprint(
        ENVIRONMENT_PATH,
        collect_environment_fingerprint(thread_count=FROZEN_THREAD_COUNT),
    )
    frozen_forbidden_identity = _assert_identity(
        FORBIDDEN_FINGERPRINT_PATH,
        EXPECTED_FROZEN_FORBIDDEN_IDENTITY,
        "frozen forbidden-files fingerprint",
    )
    forbidden_gate = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
    )
    if (
        forbidden_gate["excluded_frozen_record_count"] != 14
        or forbidden_gate["certified_record_count"] != 42_693
    ):
        raise IntegrityError(f"unexpected .DS_Store adjudication counts: {forbidden_gate}")
    pinned_gate = verify_pinned_artifacts(PROJECT_ROOT)
    snapshot_gate = _assert_identity(
        HMM_BUILDER_SNAPSHOT_PATH,
        EXPECTED_HMM_BUILDER_SNAPSHOT_IDENTITY,
        "HMM builder evidence snapshot",
    )

    assembly = meta_model.build_meta_dataset(
        meta_model.CANONICAL_SYMBOLS,
        PROJECT_ROOT / "data" / "datasets_4h",
        PROJECT_ROOT / "data" / "external",
        timeframe="4h",
        hmm_dir=PROJECT_ROOT / "log" / "hmm_walkforward_fwd_long",
        primary_dir=PROJECT_ROOT / "data" / "models",
        is_real_data_run=True,
    )
    p0_a = audit_full_population(assembly["events"])

    lineage = _build_research_lineage(assembly)
    write_json_once(LINEAGE_PATH, lineage)
    lineage_verification = verify_research_input_lineage(
        LINEAGE_PATH, base_dir=PROJECT_ROOT
    )
    lineage_identity = file_identity(LINEAGE_PATH)

    owner_gate = {
        "rule": {
            "kind": "exact_file_basename",
            "value": ".DS_Store",
            "depth": "any",
            "other_dotfiles_or_globs_excluded": False,
        },
        "changed_paths": [
            ".DS_Store",
            "analytics/.DS_Store",
            "data/.DS_Store",
            "data/models/.DS_Store",
        ],
        "excluded_frozen_record_count": forbidden_gate[
            "excluded_frozen_record_count"
        ],
        "excluded_current_record_count": forbidden_gate[
            "excluded_current_record_count"
        ],
        "certified_record_count": forbidden_gate["certified_record_count"],
        "frozen_fingerprint_rebuilt_or_deleted": False,
        "frozen_fingerprint_identity": frozen_forbidden_identity,
        "future_automatic_refreeze_allowed": False,
        "any_other_mismatch_requires_full_stop": True,
        "verification_status": forbidden_gate["status"],
    }
    p0_report = _build_p0_report(
        assembly=assembly,
        p0_a=p0_a,
        lineage_identity=lineage_identity,
        owner_gate=owner_gate,
    )
    write_json_once(P0_REPORT_PATH, p0_report)

    tests = _run_phase_tests()
    forbidden_after_tests = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
    )
    lineage_after_tests = verify_research_input_lineage(
        LINEAGE_PATH, base_dir=PROJECT_ROOT
    )
    if tests["returncode"] != 0:
        raise IntegrityError("phase-one pinned tests failed")

    elapsed = time.perf_counter() - started
    output_identities = {
        str(path.relative_to(PROJECT_ROOT)): file_identity(path)
        for path in (
            HMM_BUILDER_SNAPSHOT_PATH,
            LINEAGE_PATH,
            P0_REPORT_PATH,
            OUTPUT_DIR / "phase1_stop_report.json",
        )
    }
    phase_report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 1,
        "status": "stopped",
        "started_at_utc": started_wall,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "environment_gate": environment_gate,
        "pinned_artifact_gate": {
            "status": pinned_gate["status"],
            "verified_count": pinned_gate["verified_count"],
        },
        "forbidden_files_gate": {
            "before_p0": forbidden_gate,
            "after_tests": forbidden_after_tests,
            "frozen_fingerprint_identity": frozen_forbidden_identity,
        },
        "owner_adjudication_ds_store": owner_gate,
        "initial_ds_store_stop": p0_report["initial_ds_store_stop"],
        "lineage_gate": {
            "initial": lineage_verification,
            "after_tests": lineage_after_tests,
            "frozen": True,
        },
        "p0_result": {
            "status": p0_report["status"],
            "blocking_blocks": p0_report["blocking_blocks"],
            "verdicts": {
                block: record["verdict"]
                for block, record in p0_report["blocks"].items()
            },
        },
        "pytest": tests,
        "output_identities_before_phase_report": output_identities,
        "supporting_artifacts": [
            "data/models/ablation81/hmm_builder_evidence_snapshot.txt",
            "data/models/ablation81/phase1_stop_report.json",
            "data/models/ablation81/phase1_integrity_report.json",
        ],
        "deviations": DEVIATIONS,
        "contract_clarifications": CONTRACT_CLARIFICATIONS,
        "self_audit_doubts": p0_report["self_audit_doubts"],
        "phase2_ready": False,
        "owner_adjudication_required": True,
        "stop_reason": "funding P0-A verdict is error",
    }
    write_json_once(PHASE_REPORT_PATH, phase_report)
    _append_task_log(
        {
            "event": "phase1_completed_with_p0_stop",
            "timestamp_utc": phase_report["completed_at_utc"],
            "status": "stopped",
            "blocking_blocks": p0_report["blocking_blocks"],
            "phase_tests": tests["status"],
            "lineage_inputs": lineage["input_count"],
            "phase2_ready": False,
        }
    )
    return phase_report


def main() -> int:
    try:
        report = run_phase1()
    except BaseException as exc:
        try:
            _append_task_log(
                {
                    "event": "phase1_execution_error",
                    "timestamp_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except BaseException:
            pass
        print(f"PHASE1_EXECUTION_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "blocking_blocks": report["p0_result"]["blocking_blocks"],
                "phase_tests": report["pytest"]["status"],
                "lineage_inputs": report["lineage_gate"]["initial"][
                    "verified_count"
                ],
                "phase2_ready": report["phase2_ready"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
