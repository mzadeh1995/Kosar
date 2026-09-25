# ==============================================================================
# analytics/ablation81/phase2.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Execute phase-two non-training arms and placebo audits for ablation81 v7."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adjudication import (
    derive_affected_mask_from_pinned,
    verify_owner_adjudication_funding,
)
from .integrity import (
    IntegrityError,
    collect_environment_fingerprint,
    file_identity,
    load_calibrated_events,
    verify_environment_fingerprint,
    verify_forbidden_files_fingerprint,
    verify_pinned_artifacts,
    verify_production_constants,
    verify_research_input_lineage,
    verify_s1_anchors,
    write_json_once,
)
from .metrics import (
    ES_SEMANTICS,
    MAXDD_SEMANTICS,
    REPORT_SCOPES,
    arm_masks,
    calculate_arm_metrics,
    equal_n_report,
    overlap_report,
    pool_composition,
    prepare_evaluated_events,
)
from .placebo import (
    MATCHINGS,
    NULL_TYPES,
    PLACEBO_SCOPES,
    build_sampling_plan,
    run_placebo_plan,
    summarize_placebo_ranks,
)


FROZEN_THREAD_COUNT = 3
PLACEBO_SEEDS = tuple(range(10_000))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "ablation81"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment_fingerprint.json"
FORBIDDEN_FINGERPRINT_PATH = OUTPUT_DIR / "forbidden_files_fingerprint.json"
LINEAGE_PATH = OUTPUT_DIR / "research_input_lineage.json"
OWNER_ADJUDICATION_PATH = OUTPUT_DIR / "owner_adjudication_funding.json"
PHASE2_PLACEBO_PATH = OUTPUT_DIR / "phase2_placebo_distributions_4h.csv"
FINAL_PLACEBO_PATH = OUTPUT_DIR / "placebo_distributions_4h.csv"
PHASE_REPORT_PATH = OUTPUT_DIR / "phase2_integrity_report.json"
PHASE_STOP_REPORT_PATH = OUTPUT_DIR / "phase2_stop_report.json"
TASK_LOG_PATH = OUTPUT_DIR / "ablation81_run.log"

BASE_PRIOR_FROZEN_IDENTITIES = {
    "data/models/ablation81/environment_fingerprint.json": {
        "size_bytes": 456,
        "sha256": "653c9efa1a08052c16019a0dd887c591727ebe8f91589e049188572dd20d1ac8",
    },
    "data/models/ablation81/forbidden_files_fingerprint.json": {
        "size_bytes": 13_209_954,
        "sha256": "f14182e35749f9c37d34282fe5b130a672f20e8f585daded838ef989d87942f1",
    },
    "data/models/ablation81/phase0_integrity_report.json": {
        "size_bytes": 19_165,
        "sha256": "ac900762ec945516f5e41398b125ff76d070c84ed95d6560663ec7c6f95f44fd",
    },
    "data/models/ablation81/hmm_builder_evidence_snapshot.txt": {
        "size_bytes": 6_509,
        "sha256": "2d4f3efd9a2497338fd1ec74ad1af020ff4d4a54b6237194cb2c76809020cb40",
    },
    "data/models/ablation81/phase1_stop_report.json": {
        "size_bytes": 3_583,
        "sha256": "ffdb78af51b56f6e0364dd1eca7c826e6149dfe7023dd4707151a2a54761f747",
    },
    "data/models/ablation81/research_input_lineage.json": {
        "size_bytes": 27_732,
        "sha256": "b527aaed547c7dabf51a9a2d613dac3b27cd34e631b7cec385be17a534f40612",
    },
    "data/models/ablation81/p0_lookahead_report.json": {
        "size_bytes": 43_244,
        "sha256": "6df201b08f5107b396d04608ac89c70e6c7ad848ea6b1516377e9a1b71772f5d",
    },
    "data/models/ablation81/phase1_integrity_report.json": {
        "size_bytes": 36_775,
        "sha256": "2822a68123e7c30459332b290f8b6eb498c25577e1e30f74601ca49304a4211f",
    },
}

RESUME_FROZEN_IDENTITIES = {
    **BASE_PRIOR_FROZEN_IDENTITIES,
    "data/models/ablation81/owner_adjudication_funding.json": {
        "size_bytes": 8_088,
        "sha256": "be1c7aeea2fbab564d906cc244c2623a15a8bc2ec43cd32d53a5253650f3b601",
    },
    "data/models/ablation81/phase2_stop_report.json": {
        "size_bytes": 11_012,
        "sha256": "3614c67a2ac92ab6dc23b4083f7e8c4afc1e69413e75c6fee560753d44b5e54d",
    },
}

EXPECTED_ARCHIVE_IDENTITY_FROM_STOP = {
    "path": str(PROJECT_ROOT / "data" / "external" / "Archive.zip"),
    "size_bytes": 474_260,
    "sha256": "8ad2de23a3029ebbb083845a5f49661d34dac0dae88236d43848ef1e9d7e6fe7",
}

DEVIATIONS = [
    {
        "origin_phase": 2,
        "text_fa": (
            "فایل `phase2_integrity_report.json` به‌عنوان گزارش فازی پشتیبان "
            "تولید شد؛ این فایل در تحلیل علمی مصرف نمی‌شود."
        ),
    },
    {
        "origin_phase": 2,
        "text_fa": (
            "فایل `phase2_placebo_distributions_4h.csv` به‌عنوان artifact "
            "مرحله‌ایِ منجمدِ فاز دو ساخته می‌شود تا "
            "`placebo_distributions_4h.csv` در فاز پنج، پس از آماده‌شدن "
            "ردیف‌های A4، فقط یک‌بار و به‌صورت کامل ساخته شود؛ فایل مرحله‌ای "
            "بازنویسی نمی‌شود."
        ),
    },
    {
        "origin_phase": 2,
        "status": "superseded_by_correction",
        "text_fa": (
            "در بازبینی فاز دو، یک اجرای فقط‌خواندنیِ "
            "`build_meta_dataset` صرفاً برای آزمودن گیت نرم‌افزاری داوری "
            "مالک انجام شد؛ هیچ خروجی فاز سه نوشته نشد، هیچ آموزش یا انتخابی "
            "انجام نشد و نتیجه در هیچ تصمیم علمی مصرف نشد."
        ),
    },
    {
        "origin_phase": 2,
        "status": "active_correction",
        "text_fa": (
            "تصحیح ثبتی: در بازبینی فاز دو، سه اجرای فقط‌خواندنیِ "
            "`build_meta_dataset` انجام شد: یک اجرای گذر واقعی گیت و دو "
            "اجرای fault-injection برای آزمودن شکست گیت؛ هیچ خروجی فاز سه "
            "نوشته نشد، هیچ آموزش یا انتخابی انجام نشد و هیچ نتیجه‌ای در "
            "تصمیم علمی مصرف نشد. ثبت قبلیِ «یک اجرا» با این متن جایگزین "
            "می‌شود."
        ),
    },
    {
        "origin_phase": 2,
        "status": "active",
        "text_fa": (
            "فایل `phase2_stop_report.json` به‌عنوان گزارش توقف پشتیبان "
            "تولید شد؛ این فایل در تحلیل علمی مصرف نمی‌شود."
        ),
    },
    {
        "origin_phase": 2,
        "status": "active",
        "text_fa": (
            "فایل بیرونی `data/external/Archive.zip` را مالک برای ارسال "
            "چهارده فایل `funding_*` به ممیزی مستقل ساخته و سپس از درخت "
            "پروژه خارج کرد؛ هیچ استثنایی افزوده نشد، اثرانگشت منجمد "
            "بازساخته یا حذف نشد و هیچ خروجی علمی تولید یا مصرف نشد."
        ),
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_task_log(event: dict[str, Any]) -> None:
    with TASK_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_prior_frozen_outputs(
    expected_identities: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    identities = (
        BASE_PRIOR_FROZEN_IDENTITIES
        if expected_identities is None
        else expected_identities
    )
    records: list[dict[str, Any]] = []
    for relative, expected in identities.items():
        path = PROJECT_ROOT / relative
        actual = file_identity(path)
        observed = {
            "size_bytes": int(actual["size_bytes"]),
            "sha256": str(actual["sha256"]),
        }
        if observed != expected:
            raise IntegrityError(
                f"prior frozen output changed: {relative}; "
                f"expected={expected}, observed={observed}"
            )
        records.append({"path": relative, **observed, "status": "passed"})
    return {
        "status": "passed",
        "verified_count": len(records),
        "records": records,
    }


def _verify_forbidden_exact_counts() -> dict[str, Any]:
    result = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH,
        project_root=PROJECT_ROOT,
    )
    expected = {
        "status": "passed",
        "excluded_frozen_record_count": 14,
        "excluded_current_record_count": 14,
        "certified_record_count": 42_693,
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected:
        raise IntegrityError(
            "forbidden exact-count gate changed: "
            f"expected={expected}, observed={observed}"
        )
    expected_rule = {
        "kind": "exact_file_basename",
        "value": ".DS_Store",
        "glob_or_other_dotfiles_excluded": False,
    }
    if result.get("ignored_rule") != expected_rule:
        raise IntegrityError(
            "forbidden exception rule changed: "
            f"expected={expected_rule}, observed={result.get('ignored_rule')}"
        )
    return result


def _verify_phase2_stop_report() -> dict[str, Any]:
    identity = file_identity(PHASE_STOP_REPORT_PATH)
    observed_identity = {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }
    expected_identity = RESUME_FROZEN_IDENTITIES[
        "data/models/ablation81/phase2_stop_report.json"
    ]
    if observed_identity != expected_identity:
        raise IntegrityError(
            "phase2 stop report identity changed: "
            f"expected={expected_identity}, observed={observed_identity}"
        )
    try:
        document = json.loads(PHASE_STOP_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read frozen phase2 stop report: {exc}") from exc
    if document.get("status") != "stopped_before_scientific_computation":
        raise IntegrityError("phase2 stop report status changed")
    archive_identity = document.get("blocking_file", {}).get("identity")
    if archive_identity != EXPECTED_ARCHIVE_IDENTITY_FROM_STOP:
        raise IntegrityError(
            "archive identity in frozen stop report changed: "
            f"expected={EXPECTED_ARCHIVE_IDENTITY_FROM_STOP}, "
            f"observed={archive_identity}"
        )
    scientific = document.get("scientific_execution", {})
    if scientific.get("scientific_outputs_present") != []:
        raise IntegrityError(
            "phase2 stop report no longer records zero scientific outputs"
        )
    archive_path = PROJECT_ROOT / "data" / "external" / "Archive.zip"
    if archive_path.exists():
        raise IntegrityError(
            "owner-removed data/external/Archive.zip is present at re-entry"
        )
    return {
        "status": "passed",
        "phase2_stop_report_identity": {
            "path": "data/models/ablation81/phase2_stop_report.json",
            **observed_identity,
        },
        "blocking_file_identity": archive_identity,
        "archive_present_at_reentry": False,
    }


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


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise IntegrityError(f"refusing to overwrite frozen CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if path.exists():
            raise IntegrityError(f"refusing to overwrite frozen CSV: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_forbidden_archive_stop_report(error_text: str) -> dict[str, Any]:
    """Freeze the phase-two pre-science stop caused by the added archive."""

    archive_relative = Path("data/external/Archive.zip")
    archive_path = PROJECT_ROOT / archive_relative
    scientific_outputs = (
        PHASE2_PLACEBO_PATH,
        FINAL_PLACEBO_PATH,
        PHASE_REPORT_PATH,
    )
    present_scientific_outputs = [
        str(path.relative_to(PROJECT_ROOT))
        for path in scientific_outputs
        if path.exists()
    ]
    if present_scientific_outputs:
        raise IntegrityError(
            "cannot register pre-science stop after phase-two outputs exist: "
            f"{present_scientific_outputs}"
        )
    if PHASE_STOP_REPORT_PATH.exists():
        raise IntegrityError(
            f"refusing to overwrite frozen stop report: {PHASE_STOP_REPORT_PATH}"
        )
    if not archive_path.is_file():
        raise IntegrityError(f"blocking archive disappeared: {archive_path}")

    archive_stat = archive_path.stat()
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        members = [
            {
                "name": item.filename,
                "uncompressed_size_bytes": int(item.file_size),
                "compressed_size_bytes": int(item.compress_size),
                "crc32_hex": f"{item.CRC:08x}",
            }
            for item in archive.infolist()
        ]

    recorded_deviations: list[dict[str, Any]] = []
    for entry in DEVIATIONS:
        record = dict(entry)
        text_fa = str(record["text_fa"])
        if "`phase2_integrity_report.json`" in text_fa:
            record["realization_status"] = "not_realized_due_forbidden_gate_stop"
        elif "`phase2_placebo_distributions_4h.csv`" in text_fa:
            record["realization_status"] = "not_realized_due_forbidden_gate_stop"
        elif record.get("status") == "superseded_by_correction":
            record["realization_status"] = "superseded"
        else:
            record["realization_status"] = "realized_read_only_diagnostic"
        recorded_deviations.append(record)
    recorded_deviations.append(
        {
            "origin_phase": 2,
            "status": "active",
            "realization_status": "realized",
            "text_fa": (
                "فایل `phase2_stop_report.json` به‌عنوان گزارش توقف پشتیبان "
                "تولید شد؛ این فایل در تحلیل علمی مصرف نمی‌شود."
            ),
        }
    )

    owner_gate = verify_owner_adjudication_funding(PROJECT_ROOT)
    prior_gate = _verify_prior_frozen_outputs()
    report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 2,
        "status": "stopped_before_scientific_computation",
        "timestamp_utc": _utc_now(),
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "blocking_gate": "forbidden_files_fingerprint",
        "blocking_error_type": "ForbiddenMutationError",
        "blocking_error": str(error_text),
        "mismatch": {
            "added": [archive_relative.as_posix()],
            "added_count": 1,
            "changed": [],
            "changed_count": 0,
            "removed": [],
            "removed_count": 0,
            "ds_store_exception_applicable": False,
            "automatic_exemption_allowed": False,
            "automatic_refreeze_allowed": False,
        },
        "blocking_file": {
            "identity": file_identity(archive_path),
            "mtime_ns": int(archive_stat.st_mtime_ns),
            "birthtime_utc": datetime.fromtimestamp(
                archive_stat.st_birthtime, tz=timezone.utc
            ).isoformat(),
            "zip_member_count": len(members),
            "zip_members_metadata_only": members,
            "content_extracted": False,
            "consumed_as_scientific_input": False,
        },
        "gate_progress_before_stop": {
            "owner_adjudication": owner_gate,
            "prior_frozen_outputs": prior_gate,
            "environment_fingerprint": "passed",
            "forbidden_files_fingerprint": "failed",
            "pinned_artifacts_checked": False,
            "research_input_lineage_checked": False,
        },
        "scientific_execution": {
            "raw_pinned_events_loaded_by_official_runner": False,
            "arm_metrics_computed_by_official_runner": False,
            "placebo_computed_by_official_runner": False,
            "models_trained": False,
            "selection_performed": False,
            "scientific_outputs_present": present_scientific_outputs,
        },
        "owner_adjudication_required": True,
        "phase2_ready": False,
        "phase3_ready": False,
        "deviations": recorded_deviations,
        "self_audit_doubts": [
            {
                "doubt": (
                    "The added archive contains funding-named files and could be "
                    "mistaken for an authorized repair source."
                ),
                "check": (
                    "Only ZIP central-directory metadata was listed; the archive "
                    "was not extracted or used, and the official runner stopped "
                    "before loading pinned events."
                ),
                "result": "no scientific output produced or consumed",
            }
        ],
        "required_owner_ruling": (
            "adjudicate data/external/Archive.zip; no deletion, exemption, "
            "restoration, or fingerprint refreeze is authorized automatically"
        ),
    }
    write_json_once(PHASE_STOP_REPORT_PATH, report)
    _append_task_log(
        {
            "event": "phase2_stopped_before_science",
            "timestamp_utc": report["timestamp_utc"],
            "status": "stopped",
            "blocking_gate": report["blocking_gate"],
            "blocking_file": archive_relative.as_posix(),
            "report": str(PHASE_STOP_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "owner_adjudication_required": True,
        }
    )
    return report


def _scope_population(events: pd.DataFrame, folds: tuple[int, ...]) -> pd.DataFrame:
    return events.loc[events["fold"].isin(folds)]


def _pool_composition_by_scope(
    events: pd.DataFrame,
    *,
    pool_mask: pd.Series,
    unique_to_arm_mask: pd.Series,
) -> dict[str, Any]:
    return {
        scope: pool_composition(
            _scope_population(events, folds),
            pool_mask=pool_mask,
            unique_to_arm_mask=unique_to_arm_mask,
        )
        for scope, folds in REPORT_SCOPES.items()
    }


def _without_equal_n_event_ids(report: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(report))
    for record in result["per_fold"].values():
        record.pop("selected_event_ids", None)
    return result


def _affected_overlap(
    events: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> dict[str, Any]:
    affected, signature = derive_affected_mask_from_pinned(events)
    arms: dict[str, Any] = {}
    for arm in ("A1", "A2", "A7"):
        selected = affected & masks[arm]
        arms[arm] = {
            "n_trades": int(selected.sum()),
            "sum_net": float(events.loc[selected, "net_return"].sum()),
        }
    if arms["A2"]["n_trades"] != 0 or arms["A7"]["n_trades"] != 0:
        raise IntegrityError(
            f"owner adjudication requires A2=A7=0 on affected rows: {arms}"
        )
    return {
        "affected_row_signature": signature,
        "arms": arms,
    }


def _actual_sum_by_scope(
    arm_metrics: dict[str, Any],
    arm: str,
) -> dict[str, float]:
    return {
        scope: float(arm_metrics[arm][scope]["sum_net"])
        for scope in PLACEBO_SCOPES
    }


def _build_placebo_results(
    events: pd.DataFrame,
    masks: dict[str, pd.Series],
    arm_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    distributions: list[pd.DataFrame] = []
    a2_results: dict[str, Any] = {
        "distributions": {},
        "pool_composition": {},
    }
    unique_a2 = masks["A2"] & ~masks["A1"]
    for null_type in NULL_TYPES:
        a2_results["distributions"][null_type] = {}
        representative_pool: pd.Series | None = None
        for matching in MATCHINGS:
            plan, plan_audit = build_sampling_plan(
                events,
                arm="A2",
                arm_mask=masks["A2"],
                a1_mask=masks["A1"],
                null_type=null_type,
                matching=matching,
            )
            distribution, exposure = run_placebo_plan(
                events, plan, seeds=PLACEBO_SEEDS
            )
            distributions.append(distribution)
            representative_pool = plan.pool_mask
            a2_results["distributions"][null_type][matching] = {
                "ranks": summarize_placebo_ranks(
                    distribution,
                    observed_sum_net_by_scope=_actual_sum_by_scope(
                        arm_metrics, "A2"
                    ),
                ),
                "holding_hours": exposure,
                "sampling_plan": plan_audit,
            }
        if representative_pool is None:
            raise IntegrityError(f"no sampling plan built for {null_type}")
        a2_results["pool_composition"][null_type] = _pool_composition_by_scope(
            events,
            pool_mask=representative_pool,
            unique_to_arm_mask=unique_a2,
        )

    e11_plan, e11_plan_audit = build_sampling_plan(
        events,
        arm="A7",
        arm_mask=masks["A7"],
        a1_mask=masks["A1"],
        null_type="primary_informed",
        matching="per_fold",
    )
    e11_distribution, e11_exposure = run_placebo_plan(
        events, e11_plan, seeds=PLACEBO_SEEDS
    )
    distributions.append(e11_distribution)
    combined = pd.concat(distributions, ignore_index=True)
    expected_columns = [
        "null_type",
        "matching",
        "arm",
        "scope",
        "seed",
        "sum_net",
    ]
    if list(combined.columns) != expected_columns:
        raise IntegrityError("combined placebo schema differs from contract")
    if len(combined) != 250_000:
        raise IntegrityError(
            f"phase-two placebo row count expected=250000 observed={len(combined)}"
        )
    group_counts = {
        "|".join(map(str, key)): int(value)
        for key, value in combined.groupby(
            ["arm", "null_type", "matching"], sort=True
        ).size().items()
    }
    expected_groups = {
        "A2|primary_informed|fold_symbol": 50_000,
        "A2|primary_informed|per_fold": 50_000,
        "A2|universe|fold_symbol": 50_000,
        "A2|universe|per_fold": 50_000,
        "A7|primary_informed|per_fold": 50_000,
    }
    if group_counts != expected_groups:
        raise IntegrityError(
            f"placebo group counts differ: {group_counts} != {expected_groups}"
        )
    e11 = {
        "encoding": {
            "null_type": "primary_informed",
            "matching": "per_fold",
            "set_identity": "A1 union A7 equals A1 because A7 is a subset of A1",
            "third_null_type_added": False,
        },
        "ranks": summarize_placebo_ranks(
            e11_distribution,
            observed_sum_net_by_scope=_actual_sum_by_scope(arm_metrics, "A7"),
        ),
        "holding_hours": e11_exposure,
        "sampling_plan": e11_plan_audit,
        "pool_composition": _pool_composition_by_scope(
            events,
            pool_mask=e11_plan.pool_mask,
            unique_to_arm_mask=(masks["A7"] & ~masks["A1"]),
        ),
    }
    return combined, {
        "A2_A3": a2_results,
        "A7_E11": e11,
        "full_distribution_row_count": int(len(combined)),
        "group_counts": group_counts,
        "seed_contract": {
            "count": 10_000,
            "first": 0,
            "last": 9_999,
            "rng": "numpy.random.default_rng(seed)",
        },
    }


def run_phase2() -> dict[str, Any]:
    if (
        PHASE2_PLACEBO_PATH.exists()
        or FINAL_PLACEBO_PATH.exists()
        or PHASE_REPORT_PATH.exists()
    ):
        raise IntegrityError("one or more phase-two outputs already exist")
    started_epoch_ns = time.time_ns()
    started_wall = _utc_now()
    started = time.perf_counter()

    # Complete the no-write re-entry protocol before loading any events.
    forbidden_before = _verify_forbidden_exact_counts()
    prior_before = _verify_prior_frozen_outputs(RESUME_FROZEN_IDENTITIES)
    environment_gate = verify_environment_fingerprint(
        ENVIRONMENT_PATH,
        collect_environment_fingerprint(thread_count=FROZEN_THREAD_COUNT),
    )
    pinned_gate = verify_pinned_artifacts(PROJECT_ROOT)
    lineage_before = verify_research_input_lineage(
        LINEAGE_PATH, base_dir=PROJECT_ROOT
    )
    owner_gate = verify_owner_adjudication_funding(PROJECT_ROOT)
    archive_stop_gate = _verify_phase2_stop_report()
    constants_gate = verify_production_constants()
    owner_mtime_ns = int(OWNER_ADJUDICATION_PATH.stat().st_mtime_ns)
    if owner_mtime_ns >= started_epoch_ns:
        raise IntegrityError(
            "owner adjudication artifact was not registered before phase-two runner"
        )
    _append_task_log(
        {
            "event": "phase2_resumed_after_owner_archive_adjudication",
            "timestamp_utc": started_wall,
            "thread_count": FROZEN_THREAD_COUNT,
            "owner_artifact": owner_gate["owner_artifact"],
            "phase2_stop_report": archive_stop_gate[
                "phase2_stop_report_identity"
            ],
            "forbidden_counts": {
                "excluded_frozen_record_count": forbidden_before[
                    "excluded_frozen_record_count"
                ],
                "certified_record_count": forbidden_before[
                    "certified_record_count"
                ],
            },
        }
    )

    raw_events = load_calibrated_events(PROJECT_ROOT)
    s1_gate = verify_s1_anchors(raw_events)
    evaluated = prepare_evaluated_events(
        raw_events,
        economic_cost=float(constants_gate["economic_cost"]),
    )
    expected_fold_counts = {2: 1_745, 3: 2_119, 4: 2_197, 5: 2_987}
    observed_fold_counts = {
        int(fold): int(count)
        for fold, count in evaluated["fold"].value_counts().sort_index().items()
    }
    if len(evaluated) != 9_048 or observed_fold_counts != expected_fold_counts:
        raise IntegrityError(
            f"phase-two evaluated population changed: rows={len(evaluated)}, "
            f"folds={observed_fold_counts}"
        )
    masks = arm_masks(evaluated)
    arm_metric_report = calculate_arm_metrics(evaluated, masks)
    e1 = overlap_report(
        evaluated, a1_mask=masks["A1"], a2_mask=masks["A2"]
    )
    e2 = _without_equal_n_event_ids(
        equal_n_report(evaluated, a2_mask=masks["A2"])
    )
    affected_overlap = _affected_overlap(evaluated, masks)
    placebo_frame, placebo_results = _build_placebo_results(
        evaluated, masks, arm_metric_report
    )

    tests = _run_phase_tests()
    if tests["returncode"] != 0:
        raise IntegrityError("phase-two pinned tests failed")
    forbidden_after_tests = _verify_forbidden_exact_counts()
    lineage_after_tests = verify_research_input_lineage(
        LINEAGE_PATH, base_dir=PROJECT_ROOT
    )
    prior_after_tests = _verify_prior_frozen_outputs(RESUME_FROZEN_IDENTITIES)
    owner_after_tests = verify_owner_adjudication_funding(PROJECT_ROOT)
    archive_stop_after_tests = _verify_phase2_stop_report()

    _write_csv_once(PHASE2_PLACEBO_PATH, placebo_frame)
    placebo_identity = file_identity(PHASE2_PLACEBO_PATH)
    if (
        int(
            pd.read_csv(
                PHASE2_PLACEBO_PATH,
                usecols=["seed"],
            ).shape[0]
        )
        != 250_000
    ):
        raise IntegrityError("persisted placebo row count differs from 250000")

    prior_after_output = _verify_prior_frozen_outputs(RESUME_FROZEN_IDENTITIES)
    owner_after_output = verify_owner_adjudication_funding(PROJECT_ROOT)
    archive_stop_after_output = _verify_phase2_stop_report()
    forbidden_after_output = _verify_forbidden_exact_counts()
    elapsed = time.perf_counter() - started
    completed_wall = _utc_now()
    phase_report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 2,
        "status": "passed",
        "started_at_utc": started_wall,
        "completed_at_utc": completed_wall,
        "elapsed_seconds": elapsed,
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "owner_adjudication_gate": {
            "initial": owner_gate,
            "after_tests": owner_after_tests,
            "after_output": owner_after_output,
            "artifact_existed_before_phase2_runner": True,
            "artifact_mtime_ns": owner_mtime_ns,
            "no_scientific_computation_preceded_registration": True,
        },
        "pre_resume_archive_stop_incident": {
            "status": "owner_adjudicated_and_reentry_passed",
            "phase2_stop_report_identity": archive_stop_gate[
                "phase2_stop_report_identity"
            ],
            "blocking_file_identity": archive_stop_gate[
                "blocking_file_identity"
            ],
            "owner_created_at_utc": "2026-07-19T02:09:52Z",
            "owner_purpose": (
                "external Fable audit transfer of fourteen funding_* files"
            ),
            "cause_was_external": True,
            "removed_by_owner": True,
            "exemption_added": False,
            "fingerprint_rebuilt_or_deleted": False,
            "scientific_outputs_produced_or_consumed": False,
            "archive_present_at_reentry": False,
            "forbidden_rule_changed": False,
            "frozen_stop_report_verification": {
                "before_science": archive_stop_gate,
                "after_tests": archive_stop_after_tests,
                "after_output": archive_stop_after_output,
            },
        },
        "reentry_protocol": {
            "status": "passed",
            "contract_full_text_read": True,
            "execution_plan_full_text_read": True,
            "phase_reports_read": [
                "data/models/ablation81/phase0_integrity_report.json",
                "data/models/ablation81/phase1_stop_report.json",
                "data/models/ablation81/phase1_integrity_report.json",
                "data/models/ablation81/phase2_stop_report.json",
            ],
            "frozen_output_count": prior_before["verified_count"],
            "pinned_artifact_count": pinned_gate["verified_count"],
            "lineage_record_count": lineage_before["verified_count"],
            "write_before_reentry_completed": False,
        },
        "prior_frozen_output_gate": {
            "before_science": prior_before,
            "after_tests": prior_after_tests,
            "after_output": prior_after_output,
        },
        "environment_gate": environment_gate,
        "forbidden_files_gate": {
            "before_science": forbidden_before,
            "after_tests": forbidden_after_tests,
            "after_output": forbidden_after_output,
        },
        "pinned_artifact_gate": {
            "status": pinned_gate["status"],
            "verified_count": pinned_gate["verified_count"],
        },
        "research_input_lineage_gate": {
            "before_science": {
                "status": lineage_before["status"],
                "verified_count": lineage_before["verified_count"],
            },
            "after_tests": {
                "status": lineage_after_tests["status"],
                "verified_count": lineage_after_tests["verified_count"],
            },
        },
        "s1_gate": s1_gate,
        "production_constants_gate": constants_gate,
        "population": {
            "evaluated_rows": int(len(evaluated)),
            "fold_counts": {
                str(fold): count
                for fold, count in observed_fold_counts.items()
            },
            "deterministic_order": "decision_ts ascending, then symbol ascending",
        },
        "metric_semantics": {
            "maxdd_semantics": MAXDD_SEMANTICS,
            "maxdd_initial_zero_prepended": False,
            "es_semantics": ES_SEMANTICS,
            "auc": "unweighted on raw score",
            "logloss": "tb_uniqueness-weighted on raw score clipped to [1e-15,1-1e-15]",
        },
        "scientific_results": {
            "arms": arm_metric_report,
            "E1_overlap": e1,
            "E2_equal_n": e2,
            "A3_and_E11_placebo": placebo_results,
            "owner_affected_103_overlap": affected_overlap,
        },
        "owner_future_requirements": {
            "phase3_exact_signature_gate": (
                "analytics.ablation81.adjudication."
                "assert_owner_adjudicated_funding_signature"
            ),
            "funding_files_must_match_frozen_lineage": True,
            "alternative_funding_sources_allowed": False,
            "microstructure_history_used_as_funding_source": False,
            "phase5_fold5_sensitivity": {
                "required": True,
                "reporting_only": True,
                "arms": [
                    "A1",
                    "A2",
                    "A4",
                    "A5",
                    "A6",
                    "A7",
                    "model_based_exploratories",
                ],
                "full_population_metrics_required": True,
                "exclude_103_from_trades_and_entry_rate_denominator": True,
                "placebo_pools_reexecuted_without_103_rows": False,
                "decision_rule_reexecuted_without_103_rows": False,
            },
            "every_fold5_arm_reports_overlap_on_103_rows": {
                "required": True,
                "metrics": ["n_trades", "sum_net"],
            },
            "final_report_feature_nan_rows": [
                owner_gate["signature"]["funding_matched_but_feature_nan"],
                owner_gate["signature"]["hmm_matched_but_feature_nan"],
            ],
            "funding_feature_nan_fold1_excluded_from_all_evaluated_metrics": True,
            "final_report_prominent_ffd_legacy_finding": {
                "required": True,
                "configured": "fractional_diff",
                "effective": "legacy",
                "builder_window_rows": 500,
                "ffd_width": 1_163,
            },
        },
        "contract_clarifications": {
            "E11_csv_encoding": (
                "arm=A7, null_type=primary_informed, matching=per_fold; "
                "A1 union A7 equals A1"
            ),
            "placebo_csv_phase2_state": (
                "phase2_placebo_distributions_4h.csv contains A2 A3 and A7 E11 "
                "rows and freezes in phase2"
            ),
            "placebo_csv_final_state": (
                "placebo_distributions_4h.csv is created once in phase5 from the "
                "verified frozen phase2 stage plus contract-required A4 rows"
            ),
            "phase2_stage_is_never_rewritten": True,
            "final_contract_csv_exists_in_phase2": False,
            "primary_informed_pools_are_arm_dependent": True,
            "A2_and_A4_percentiles_share_one_scale": False,
            "A4_not_computed_in_phase2": True,
        },
        "pytest": tests,
        "output_identities_before_phase_report": {
            "data/models/ablation81/owner_adjudication_funding.json": file_identity(
                OWNER_ADJUDICATION_PATH
            ),
            "data/models/ablation81/phase2_stop_report.json": file_identity(
                PHASE_STOP_REPORT_PATH
            ),
            "data/models/ablation81/phase2_placebo_distributions_4h.csv": (
                placebo_identity
            ),
            "analytics/ablation81/adjudication.py": file_identity(
                PROJECT_ROOT / "analytics" / "ablation81" / "adjudication.py"
            ),
            "analytics/ablation81/metrics.py": file_identity(
                PROJECT_ROOT / "analytics" / "ablation81" / "metrics.py"
            ),
            "analytics/ablation81/placebo.py": file_identity(
                PROJECT_ROOT / "analytics" / "ablation81" / "placebo.py"
            ),
            "analytics/ablation81/phase2.py": file_identity(
                PROJECT_ROOT / "analytics" / "ablation81" / "phase2.py"
            ),
            "tests/test_ablation81.py": file_identity(
                PROJECT_ROOT / "tests" / "test_ablation81.py"
            ),
        },
        "supporting_artifacts": [
            "data/models/ablation81/owner_adjudication_funding.json",
            "data/models/ablation81/phase2_stop_report.json",
            "data/models/ablation81/phase2_placebo_distributions_4h.csv",
            "data/models/ablation81/phase2_integrity_report.json",
        ],
        "deviations": DEVIATIONS,
        "self_audit_doubts": [
            {
                "doubt": (
                    "Count matching does not equal time-in-market matching and can "
                    "leave exposure differences."
                ),
                "check": (
                    "Total holding hours were computed for every placebo iteration "
                    "and summarized at p05/p50/p95 and mean."
                ),
                "result": "recorded for A2 A3 and A7 E11 distributions",
            },
            {
                "doubt": (
                    "The owner-authorized 103-row mask is inferred from the exact "
                    "registered interval on pinned rows during phase two."
                ),
                "check": (
                    "Count, symbol split, fold, start, and end were asserted; phase "
                    "three has a separate full assembled-source signature gate."
                ),
                "result": "103 exact pinned rows; full source gate reserved for phase3",
            },
            {
                "doubt": (
                    "MaxDD could be misimplemented by silently prepending an initial "
                    "zero not stated as part of the additive trade curve."
                ),
                "check": (
                    "The implementation and hand test use maximum.accumulate(cumsum) "
                    "minus cumsum without prepending zero."
                ),
                "result": "literal section-4 semantics enforced",
            },
        ],
        "phase3_ready": True,
        "phase3_readiness": {
            "technical_ready": True,
            "owner_continue_required": True,
            "owner_acceptance_of_phase2_deviations_required": True,
        },
        "next_phase_requires_owner_continue_message": True,
    }
    write_json_once(PHASE_REPORT_PATH, phase_report)
    _append_task_log(
        {
            "event": "phase2_completed",
            "timestamp_utc": completed_wall,
            "status": "passed",
            "placebo_rows": int(len(placebo_frame)),
            "phase_tests": tests["status"],
            "phase3_ready": True,
        }
    )
    return phase_report


def main() -> int:
    try:
        report = run_phase2()
    except BaseException as exc:
        try:
            _append_task_log(
                {
                    "event": "phase2_execution_error",
                    "timestamp_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except BaseException:
            pass
        print(f"PHASE2_EXECUTION_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "placebo_rows": report["scientific_results"][
                    "A3_and_E11_placebo"
                ]["full_distribution_row_count"],
                "phase_tests": report["pytest"]["status"],
                "phase3_ready": report["phase3_ready"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
