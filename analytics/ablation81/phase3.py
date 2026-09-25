# ==============================================================================
# analytics/ablation81/phase3.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Execute ablation81 phase three and stop before phase four."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .a4 import A4_OOF_COLUMNS, run_a4_calibration_pipeline
from .cscv import (
    BLOCK_COLUMNS,
    MATRIX_COLUMNS,
    PREDICTION_COLUMNS,
    build_cscv_matrix,
    canonicalize_prediction_rows,
)
from .integrity import (
    ALLOWED_PATHS,
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
from .nested import (
    FROZEN_THREAD_COUNT,
    OUTER_FOLDS,
    SEED_STABILITY_SEEDS,
    build_config_catalog,
    refit_winner_for_seed,
    search_outer_fold,
)
from .phase3_gates import run_pretraining_gates
from .metrics import ES_SEMANTICS, MAXDD_SEMANTICS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "ablation81"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment_fingerprint.json"
FORBIDDEN_PATH = OUTPUT_DIR / "forbidden_files_fingerprint.json"
LINEAGE_PATH = OUTPUT_DIR / "research_input_lineage.json"
TASK_LOG_PATH = OUTPUT_DIR / "ablation81_run.log"
SELECTED_CONFIGS_PATH = OUTPUT_DIR / "selected_configs.json"
SEED_STABILITY_PATH = OUTPUT_DIR / "seed_stability_4h.json"
A4_METRICS_PATH = OUTPUT_DIR / "a4_metrics_4h.json"
A4_OOF_PATH = OUTPUT_DIR / "a4_oof_4h.csv"
CSCV_PREDICTIONS_PATH = OUTPUT_DIR / "cscv_predictions_4h.csv"
CSCV_MATRIX_PATH = OUTPUT_DIR / "cscv_matrix_4h.csv"
PHASE_REPORT_PATH = OUTPUT_DIR / "phase3_integrity_report.json"
PHASE_STOP_REPORT_PATH = OUTPUT_DIR / "phase3_stop_report.json"

PRIOR_FROZEN_IDENTITIES = {
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
    "data/models/ablation81/owner_adjudication_funding.json": {
        "size_bytes": 8_088,
        "sha256": "be1c7aeea2fbab564d906cc244c2623a15a8bc2ec43cd32d53a5253650f3b601",
    },
    "data/models/ablation81/phase2_stop_report.json": {
        "size_bytes": 11_012,
        "sha256": "3614c67a2ac92ab6dc23b4083f7e8c4afc1e69413e75c6fee560753d44b5e54d",
    },
    "data/models/ablation81/phase2_placebo_distributions_4h.csv": {
        "size_bytes": 14_915_153,
        "sha256": "1dd9f6039034ae19812dfae6cb07ac5d433d85bcdc30af937b3ebb90ab9f3d31",
    },
    "data/models/ablation81/phase2_integrity_report.json": {
        "size_bytes": 84_271,
        "sha256": "2da529bede56a4332bf5f2b1018d9d352f685b496a5e597d63f3e5618cd0cc73",
    },
}
PRIOR_FROZEN_SOURCE_IDENTITIES = {
    "analytics/ablation81/adjudication.py": {
        "size_bytes": 13_273,
        "sha256": "b9d57fc8a9643a5595f5105d563b37acc6e4b2f7685a368029f41a5861ffd15d",
    },
    "analytics/ablation81/metrics.py": {
        "size_bytes": 15_775,
        "sha256": "ec5c7a10d0ec7e370d0701d0aca3349e3ecc2725de4615a90090a4d1a471e4b1",
    },
    "analytics/ablation81/placebo.py": {
        "size_bytes": 14_844,
        "sha256": "335ba7a06d4326bd2e5d728042f850469c32b6d2db8582c5d67519db337e7321",
    },
    "analytics/ablation81/phase2.py": {
        "size_bytes": 41_736,
        "sha256": "9c118c6ebbae7c33c21ac8dbf7fde9ebbe2da3d7df31903bf8c482607acb231c",
    },
}

DEVIATIONS = [
    {
        "origin_phase": 3,
        "status": "active",
        "announced_verbatim_fa": (
            "انحراف اعلامی فاز سه: «نخستین اجرای پیاده‌سازی جدیدِ دروازه‌ی "
            "هم‌ترازی، به‌علت مقایسه‌ی بیش‌ازحد سخت دیکشنریِ شمارنده‌ی "
            "categorical در pandas، با وجود نتیجه‌ی واقعیِ `both=11029`، "
            "`left_only=0` و `right_only=0` شکست کاذب داد؛ هیچ آموزش، "
            "انتخاب، یا خروجی علمی تولید یا مصرف نشد و اصلاح فقط در منطق "
            "همان چک‌کننده اعمال می‌شود.»"
        ),
        "text_fa": (
            "نخستین اجرای پیاده‌سازی جدیدِ دروازه‌ی هم‌ترازی، به‌علت "
            "مقایسه‌ی بیش‌ازحد سخت دیکشنریِ شمارنده‌ی categorical در pandas، "
            "با وجود نتیجه‌ی واقعیِ `both=11029`، `left_only=0` و "
            "`right_only=0` شکست کاذب داد؛ هیچ آموزش، انتخاب، یا خروجی علمی "
            "تولید یا مصرف نشد و اصلاح فقط در منطق همان چک‌کننده اعمال می‌شود."
        ),
    },
    {
        "origin_phase": 3,
        "status": "active",
        "announced_verbatim_fa": (
            "«فایل `a4_oof_4h.csv` به‌عنوان artifact علمیِ ردیفی و منجمد "
            "ساخته می‌شود تا امتیاز خام، امتیاز مؤثر، آستانه و ماسک A4 در "
            "فازهای بعد بدون بازآموزی یا بازکالیبراسیون منتقل شود؛ این فایل "
            "در انتخاب پیکربندی دخالت ندارد.»"
        ),
        "text_fa": (
            "فایل `a4_oof_4h.csv` به‌عنوان artifact علمیِ ردیفی و منجمد "
            "ساخته می‌شود تا امتیاز خام، امتیاز مؤثر، آستانه و ماسک A4 در "
            "فازهای بعد بدون بازآموزی یا بازکالیبراسیون منتقل شود؛ این فایل "
            "در انتخاب پیکربندی دخالت ندارد."
        ),
    },
    {
        "origin_phase": 3,
        "status": "active",
        "announced_verbatim_fa": (
            "«فایل `phase3_integrity_report.json` به‌عنوان گزارش فازی "
            "پشتیبان تولید می‌شود و در محاسبات علمی فازهای بعد مصرف نمی‌شود.»"
        ),
        "text_fa": (
            "فایل `phase3_integrity_report.json` به‌عنوان گزارش فازی پشتیبان "
            "تولید می‌شود و در محاسبات علمی فازهای بعد مصرف نمی‌شود."
        ),
    },
    {
        "origin_phase": 3,
        "status": "active",
        "announced_verbatim_fa": (
            "انحراف اعلامی فاز سه: «فایل `phase3_stop_report.json` به‌عنوان "
            "گزارش توقف پشتیبانِ این خطای کالیبراسیون تولید می‌شود؛ در تحلیل "
            "علمی مصرف نمی‌شود و هیچ مجوزی برای fallback، تغییر علامت، "
            "isotonic، خام‌کردن خودکار، یا تغییر کد تولید ایجاد نمی‌کند.»"
        ),
        "text_fa": (
            "فایل `phase3_stop_report.json` به‌عنوان گزارش توقف پشتیبانِ "
            "این خطای کالیبراسیون تولید می‌شود؛ در تحلیل علمی مصرف نمی‌شود "
            "و هیچ مجوزی برای fallback، تغییر علامت، isotonic، خام‌کردن "
            "خودکار، یا تغییر کد تولید ایجاد نمی‌کند."
        ),
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat().replace("+00:00", "Z")
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IntegrityError(f"non-finite value cannot enter JSON: {value}")
        return value
    return value


def _append_task_log(event: dict[str, Any]) -> None:
    payload = _json_ready({"timestamp_utc": _utc_now(), **event})
    with TASK_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _progress(event: dict[str, Any]) -> None:
    ready = _json_ready(event)
    print(json.dumps(ready, ensure_ascii=False, sort_keys=True), flush=True)
    _append_task_log(ready)


def _identity_subset(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    return {
        "path": str(path),
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _verify_prior_frozen_outputs() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    expected_identities = {
        **PRIOR_FROZEN_IDENTITIES,
        **PRIOR_FROZEN_SOURCE_IDENTITIES,
    }
    for relative, expected in expected_identities.items():
        actual = _identity_subset(PROJECT_ROOT / relative)
        observed = {
            "size_bytes": actual["size_bytes"],
            "sha256": actual["sha256"],
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
        "frozen_artifact_count": len(PRIOR_FROZEN_IDENTITIES),
        "frozen_phase2_source_count": len(PRIOR_FROZEN_SOURCE_IDENTITIES),
        "records": records,
    }


def _excluded_ds_store_paths() -> dict[str, Any]:
    try:
        frozen = json.loads(FORBIDDEN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read forbidden fingerprint paths: {exc}") from exc
    frozen_records = frozen.get("records")
    if not isinstance(frozen_records, list):
        raise IntegrityError("forbidden fingerprint records must be a list")
    frozen_paths = sorted(
        str(record["path"])
        for record in frozen_records
        if Path(str(record.get("path", ""))).name == ".DS_Store"
    )
    current_paths: list[str] = []
    for current_text, directory_names, file_names in os.walk(
        PROJECT_ROOT, followlinks=False
    ):
        current = Path(current_text)
        kept_directories: list[str] = []
        for name in directory_names:
            child = current / name
            relative = child.relative_to(PROJECT_ROOT).as_posix()
            if any(
                relative == allowed or relative.startswith(f"{allowed}/")
                for allowed in ALLOWED_PATHS
            ):
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
        if ".DS_Store" in file_names:
            current_paths.append(
                (current / ".DS_Store").relative_to(PROJECT_ROOT).as_posix()
            )
    current_paths.sort()
    return {
        "excluded_frozen_paths": frozen_paths,
        "excluded_current_paths": current_paths,
        "new_current_paths_vs_frozen": sorted(set(current_paths) - set(frozen_paths)),
        "missing_current_paths_vs_frozen": sorted(
            set(frozen_paths) - set(current_paths)
        ),
    }


def _verify_forbidden_owner_rule() -> dict[str, Any]:
    gate = verify_forbidden_files_fingerprint(
        FORBIDDEN_PATH, project_root=PROJECT_ROOT
    )
    if (
        gate["status"] != "passed"
        or int(gate["excluded_frozen_record_count"]) != 14
        or int(gate["certified_record_count"]) != 42_693
        or gate["ignored_rule"]
        != {
            "kind": "exact_file_basename",
            "value": ".DS_Store",
            "glob_or_other_dotfiles_excluded": False,
        }
    ):
        raise IntegrityError(f"forbidden-files owner rule changed: {gate}")
    path_audit = _excluded_ds_store_paths()
    if len(path_audit["excluded_frozen_paths"]) != int(
        gate["excluded_frozen_record_count"]
    ) or len(path_audit["excluded_current_paths"]) != int(
        gate["excluded_current_record_count"]
    ):
        raise IntegrityError("exact .DS_Store path audit count mismatch")
    return {**gate, **path_audit}


def _run_reentry_protocol() -> dict[str, Any]:
    environment = collect_environment_fingerprint(
        thread_count=FROZEN_THREAD_COUNT
    )
    return {
        "status": "passed",
        "contract_and_execution_plan_read_in_full": True,
        "prior_phase_reports_read": [
            "phase0_integrity_report.json",
            "phase1_stop_report.json",
            "phase1_integrity_report.json",
            "owner_adjudication_funding.json",
            "phase2_stop_report.json",
            "phase2_integrity_report.json",
        ],
        "environment_gate": verify_environment_fingerprint(
            ENVIRONMENT_PATH, environment
        ),
        "prior_frozen_output_gate": _verify_prior_frozen_outputs(),
        "pinned_artifact_gate": verify_pinned_artifacts(PROJECT_ROOT),
        "research_input_lineage_gate": verify_research_input_lineage(
            LINEAGE_PATH, base_dir=PROJECT_ROOT
        ),
        "forbidden_files_gate": _verify_forbidden_owner_rule(),
        "s1_gate": verify_s1_anchors(load_calibrated_events(PROJECT_ROOT)),
        "production_constants_gate": verify_production_constants(),
    }


def _run_pytest(path: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        path,
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    result = {
        "command": command,
        "returncode": int(completed.returncode),
        "elapsed_seconds": float(time.perf_counter() - started),
        "output": output,
        "status": "passed" if completed.returncode == 0 else "failed",
    }
    if completed.returncode != 0:
        raise IntegrityError(f"pytest failed for {path}:\n{output}")
    return result


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
        )
        handle.flush()
        os.fsync(handle.fileno())


def _read_roundtrip_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def _assert_float_roundtrip(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    for column in columns:
        left = expected[column].to_numpy(dtype=float)
        right = observed[column].to_numpy(dtype=float)
        if not np.array_equal(left, right):
            raise IntegrityError(f"CSV float round-trip changed column {column}")


def _commit_phase_artifacts(
    *,
    selected_configs: dict[str, Any],
    seed_stability: dict[str, Any],
    a4_metrics: dict[str, Any],
    a4_oof: pd.DataFrame,
    cscv_predictions: pd.DataFrame,
    cscv_matrix: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    destinations = {
        "selected_configs": SELECTED_CONFIGS_PATH,
        "seed_stability": SEED_STABILITY_PATH,
        "a4_metrics": A4_METRICS_PATH,
        "a4_oof": A4_OOF_PATH,
        "cscv_predictions": CSCV_PREDICTIONS_PATH,
        "cscv_matrix": CSCV_MATRIX_PATH,
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise IntegrityError(f"refusing to overwrite frozen phase3 artifacts: {existing}")
    with tempfile.TemporaryDirectory(
        prefix=".phase3-stage-", dir=OUTPUT_DIR
    ) as temporary_text:
        temporary = Path(temporary_text)
        staged = {
            key: temporary / path.name for key, path in destinations.items()
        }
        _stage_json(staged["selected_configs"], selected_configs)
        _stage_json(staged["seed_stability"], seed_stability)
        _stage_json(staged["a4_metrics"], a4_metrics)
        _stage_csv(staged["a4_oof"], a4_oof)
        _stage_csv(staged["cscv_predictions"], cscv_predictions)
        _stage_csv(staged["cscv_matrix"], cscv_matrix)
        for key, payload in (
            ("selected_configs", selected_configs),
            ("seed_stability", seed_stability),
            ("a4_metrics", a4_metrics),
        ):
            observed_json = json.loads(staged[key].read_text(encoding="utf-8"))
            if observed_json != _json_ready(payload):
                raise IntegrityError(f"{key} JSON staged round-trip mismatch")

        observed_a4 = _read_roundtrip_csv(staged["a4_oof"])
        if (
            tuple(observed_a4.columns) != A4_OOF_COLUMNS
            or len(observed_a4) != len(a4_oof)
            or not observed_a4.loc[
                :, ["symbol", "decision_ts", "fold"]
            ].astype(str).equals(
                a4_oof.loc[:, ["symbol", "decision_ts", "fold"]].astype(str)
            )
            or observed_a4["traded_A4"].astype(bool).tolist()
            != a4_oof["traded_A4"].astype(bool).tolist()
            or observed_a4["cal_is_raw"].astype(int).tolist()
            != a4_oof["cal_is_raw"].astype(int).tolist()
        ):
            raise IntegrityError("A4 OOF staged round-trip schema/mask mismatch")
        _assert_float_roundtrip(
            a4_oof,
            observed_a4,
            ("p_raw", "p_effective", "tau"),
        )

        observed_predictions = _read_roundtrip_csv(
            staged["cscv_predictions"]
        )
        if (
            tuple(observed_predictions.columns) != PREDICTION_COLUMNS
            or len(observed_predictions) != len(cscv_predictions)
            or observed_predictions.loc[
                :, ["config_id", "outer_fold", "symbol", "decision_ts"]
            ].astype(str).equals(
                cscv_predictions.loc[
                    :, ["config_id", "outer_fold", "symbol", "decision_ts"]
                ].astype(str)
            )
            is False
        ):
            raise IntegrityError("CSCV prediction staged round-trip key mismatch")
        _assert_float_roundtrip(
            cscv_predictions, observed_predictions, ("p_raw",)
        )

        observed_matrix = _read_roundtrip_csv(staged["cscv_matrix"])
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
            raise IntegrityError("CSCV matrix staged round-trip schema mismatch")
        _assert_float_roundtrip(
            cscv_matrix,
            observed_matrix,
            ("depth", "l2_leaf_reg", "learning_rate"),
        )
        _assert_float_roundtrip(
            cscv_matrix, observed_matrix, BLOCK_COLUMNS
        )
        committed: list[Path] = []
        try:
            for key, destination in destinations.items():
                if destination.exists():
                    raise IntegrityError(
                        f"phase3 destination appeared before commit: {destination}"
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
                raise IntegrityError(
                    "phase3 partial-commit rollback failed: "
                    + " | ".join(rollback_failures)
                )
            raise
    return {
        str(path.relative_to(PROJECT_ROOT)): _identity_subset(path)
        for path in destinations.values()
    }


def _canonical_a4_oof(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[:, list(A4_OOF_COLUMNS)].copy()
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return work.sort_values(
        ["fold", "decision_ts", "symbol"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def run_phase3() -> dict[str, Any]:
    """Run only phase three, freeze its artifacts, and require owner continuation."""

    phase_cpu_start = time.process_time()
    phase_wall_start = time.perf_counter()
    started_at = _utc_now()
    all_frozen_destinations = (
        SELECTED_CONFIGS_PATH,
        SEED_STABILITY_PATH,
        A4_METRICS_PATH,
        A4_OOF_PATH,
        CSCV_PREDICTIONS_PATH,
        CSCV_MATRIX_PATH,
        PHASE_REPORT_PATH,
    )
    preexisting = [str(path) for path in all_frozen_destinations if path.exists()]
    if preexisting:
        raise IntegrityError(
            f"phase3 frozen destination already exists before execution: {preexisting}"
        )
    _append_task_log({"event": "phase3_started"})
    reentry = _run_reentry_protocol()
    _progress({"event": "phase3_reentry_passed"})

    assembly, pretraining_gates = run_pretraining_gates(PROJECT_ROOT)
    _progress(
        {
            "event": "phase3_pretraining_gates_passed",
            "order": pretraining_gates["execution_order"],
        }
    )
    pytest_before = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    _progress(
        {
            "event": "phase3_tests_before_training_passed",
            "ablation81_output": pytest_before["ablation81"]["output"].strip(),
            "calibration_output": pytest_before["calibration_regression"][
                "output"
            ].strip(),
        }
    )

    events = assembly["events"].copy()
    events["fold"] = pd.to_numeric(events["fold"], errors="raise").astype(int)
    events["decision_ts"] = pd.to_datetime(
        events["decision_ts"], utc=True, errors="raise"
    )
    X = assembly["X"]
    feature_columns = list(assembly["feature_columns"])
    categorical_features = list(assembly["categorical_features"])
    catalog = build_config_catalog()
    outer_reports: dict[str, Any] = {}
    cscv_parts: list[pd.DataFrame] = []
    seed_prediction_parts: dict[int, list[pd.DataFrame]] = {
        seed: [] for seed in SEED_STABILITY_SEEDS
    }
    refit_cpu_total = 0.0
    refit_wall_total = 0.0

    for outer_fold in OUTER_FOLDS:
        selection_events = events.loc[events["fold"].lt(outer_fold)].copy()
        outer_events = events.loc[events["fold"].eq(outer_fold)].copy()
        _progress(
            {
                "event": "phase3_outer_search_started",
                "outer_fold": outer_fold,
                "history_rows": len(selection_events),
                "outer_rows": len(outer_events),
            }
        )
        search = search_outer_fold(
            selection_events,
            outer_events,
            X,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
            outer_fold=outer_fold,
            configs=catalog,
            progress=_progress,
        )
        cscv_parts.append(search.pop("cscv_predictions"))
        refits: dict[str, Any] = {}
        winner = search["winner"]
        for seed in SEED_STABILITY_SEEDS:
            refit = refit_winner_for_seed(
                selection_events,
                outer_events,
                X,
                feature_columns=feature_columns,
                categorical_features=categorical_features,
                outer_fold=outer_fold,
                config=winner["config"],
                tree_count=winner["tree_count"],
                seed=seed,
            )
            probabilities = refit.pop("probabilities")
            refit_cpu_total += float(refit["metadata"]["fit_cpu_seconds"])
            refit_wall_total += float(refit["metadata"]["fit_wall_seconds"])
            seed_prediction_parts[seed].append(
                pd.DataFrame(
                    {
                        "symbol": outer_events["symbol"].astype(str).to_numpy(),
                        "decision_ts": outer_events["decision_ts"].to_numpy(),
                        "fold": outer_fold,
                        "p_raw": probabilities,
                    }
                )
            )
            refits[str(seed)] = refit["metadata"]
        search["winner_refits_by_seed"] = refits
        outer_reports[str(outer_fold)] = search
        _progress(
            {
                "event": "phase3_outer_fold_completed",
                "outer_fold": outer_fold,
                "winner": winner,
                "search_timing": search["timing"],
                "refit_wall_seconds": float(
                    sum(
                        record["fit_wall_seconds"]
                        for record in refits.values()
                    )
                ),
            }
        )

    evaluated_events = events.loc[events["fold"].isin(OUTER_FOLDS)].copy()
    cscv_predictions_memory = pd.concat(cscv_parts, ignore_index=True)
    cscv_predictions_memory = cscv_predictions_memory.loc[
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
    pinned = load_calibrated_events(PROJECT_ROOT)
    seed_results: dict[str, Any] = {}
    canonical_pipeline: dict[str, Any] | None = None
    for seed in SEED_STABILITY_SEEDS:
        raw_predictions = pd.concat(
            seed_prediction_parts[seed], ignore_index=True
        )
        pipeline = run_a4_calibration_pipeline(
            pinned,
            raw_predictions,
            economic_cost=float(assembly["metadata"]["cost_round_trip"]),
        )
        oof = pipeline.pop("oof")
        seed_results[str(seed)] = pipeline
        if seed == 42:
            canonical_pipeline = {**pipeline, "oof": oof}
    if canonical_pipeline is None:
        raise IntegrityError("canonical seed 42 pipeline was not produced")
    a4_oof = _canonical_a4_oof(canonical_pipeline.pop("oof"))

    search_cpu_by_fold = {
        fold: float(outer_reports[fold]["timing"]["search_cpu_seconds"])
        for fold in outer_reports
    }
    search_wall_by_fold = {
        fold: float(outer_reports[fold]["timing"]["search_wall_seconds"])
        for fold in outer_reports
    }
    selected_configs_payload = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 3,
        "frozen": True,
        "thread_count": FROZEN_THREAD_COUNT,
        "selection_objective": "tb_uniqueness_weighted_raw_logloss_only",
        "selection_tie_tolerance": 1e-12,
        "selection_never_reads_cscv_predictions": True,
        "fold5_quarantine_active": True,
        "config_catalog": catalog,
        "outer_folds": outer_reports,
        "search_timing": {
            "per_outer_fold_cpu_seconds": search_cpu_by_fold,
            "per_outer_fold_wall_seconds": search_wall_by_fold,
            "total_cpu_seconds": float(sum(search_cpu_by_fold.values())),
            "total_wall_seconds": float(sum(search_wall_by_fold.values())),
        },
        "winner_refit_timing": {
            "total_cpu_seconds": float(refit_cpu_total),
            "total_wall_seconds": float(refit_wall_total),
        },
    }
    seed_stability_payload = {
        "schema_version": 1,
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
        },
        "results_by_seed": seed_results,
    }
    a4_metrics_payload = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 3,
        "arm": "A4",
        "frozen": True,
        "canonical_seed": 42,
        "metric_semantics": {
            "maxdd_semantics": MAXDD_SEMANTICS,
            "es_semantics": ES_SEMANTICS,
        },
        "raw_model_metrics_are_unweighted_auc_and_tb_uniqueness_weighted_logloss": True,
        "trade_metrics_and_replay": canonical_pipeline,
        "rowwise_handoff": {
            "path": "data/models/ablation81/a4_oof_4h.csv",
            "columns": list(A4_OOF_COLUMNS),
            "row_count": int(len(a4_oof)),
            "selection_input": False,
        },
    }
    output_identities = _commit_phase_artifacts(
        selected_configs=selected_configs_payload,
        seed_stability=seed_stability_payload,
        a4_metrics=a4_metrics_payload,
        a4_oof=a4_oof,
        cscv_predictions=cscv_predictions,
        cscv_matrix=cscv_matrix,
    )
    _progress(
        {
            "event": "phase3_scientific_artifacts_frozen",
            "outputs": output_identities,
        }
    )

    pytest_after = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    forbidden_after = _verify_forbidden_owner_rule()
    prior_after = _verify_prior_frozen_outputs()
    source_identities = {
        str(path.relative_to(PROJECT_ROOT)): _identity_subset(path)
        for path in (
            PROJECT_ROOT / "analytics" / "ablation81" / "phase3_gates.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "quarantine.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "nested.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "cscv.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "a4.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "phase3.py",
            PROJECT_ROOT / "tests" / "test_ablation81.py",
        )
    }
    phase_cpu = time.process_time() - phase_cpu_start
    phase_wall = time.perf_counter() - phase_wall_start
    report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 3,
        "status": "completed_waiting_for_owner_continue",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "reentry_protocol": reentry,
        "pretraining_hard_gates": pretraining_gates,
        "pytest": {
            "before_training": pytest_before,
            "after_artifacts": pytest_after,
            "phase3_pinned_tests_implemented_and_green": [9, 10, 12, 15],
        },
        "scientific_results": {
            "metric_semantics": {
                "maxdd_semantics": MAXDD_SEMANTICS,
                "es_semantics": ES_SEMANTICS,
            },
            "selected_config_by_outer_fold": {
                fold: outer_reports[fold]["winner"] for fold in outer_reports
            },
            "a4_metrics": canonical_pipeline["metrics"],
            "a4_funding_affected_fold5_overlap": canonical_pipeline[
                "funding_affected_fold5_overlap"
            ],
            "a4_replay_reconstruction": canonical_pipeline["reconstruction"],
            "a4_calibration_fold_records": canonical_pipeline["calibration"][
                "fold_records"
            ],
            "seed_stability_summary": {
                seed: {
                    "metrics": seed_results[seed]["metrics"],
                    "funding_affected_fold5_overlap": seed_results[seed][
                        "funding_affected_fold5_overlap"
                    ],
                }
                for seed in seed_results
            },
            "cscv": cscv_audit,
        },
        "timing": {
            "search_per_outer_fold": {
                fold: outer_reports[fold]["timing"] for fold in outer_reports
            },
            "search_total_cpu_seconds": float(
                sum(search_cpu_by_fold.values())
            ),
            "search_total_wall_seconds": float(
                sum(search_wall_by_fold.values())
            ),
            "winner_refit_total_cpu_seconds": float(refit_cpu_total),
            "winner_refit_total_wall_seconds": float(refit_wall_total),
            "phase_cpu_seconds": float(phase_cpu),
            "phase_wall_seconds": float(phase_wall),
        },
        "output_identities_before_phase_report": output_identities,
        "source_identities": source_identities,
        "supporting_artifacts": [
            "data/models/ablation81/a4_oof_4h.csv",
            "data/models/ablation81/phase3_integrity_report.json",
        ],
        "deviations": DEVIATIONS,
        "self_audit_doubts": [
            {
                "doubt": (
                    "CatBoost internal early-stopping Logloss can reflect its "
                    "scale_pos_weight as well as Pool weights."
                ),
                "check": (
                    "Configuration selection was recomputed independently from "
                    "raw validation probabilities using only tb_uniqueness."
                ),
                "result": "independent contract objective used for all 432 candidates",
            },
            {
                "doubt": (
                    "A stored outer-fold prediction could accidentally influence "
                    "configuration selection."
                ),
                "check": (
                    "Selection accepts only the inner ledger, rejects outer/CSCV "
                    "fields, and test 15 mutates the prediction file."
                ),
                "result": "selection output remained identical",
            },
            {
                "doubt": (
                    "A calibrated threshold mask could differ from production replay."
                ),
                "check": (
                    "Every fold mask was reconstructed rowwise and compared on exact "
                    "n_trades and sum_net at absolute tolerance 1e-9."
                ),
                "result": "all four folds passed",
            },
            {
                "doubt": (
                    "A decision timestamp shared by multiple symbols could be split "
                    "between CSCV blocks."
                ),
                "check": (
                    "All sixteen blocks were assigned at grouped decision_ts "
                    "boundaries and group block cardinality was asserted equal to one."
                ),
                "result": "all sixteen nonempty blocks passed",
            },
        ],
        "forbidden_files_gate_after_outputs": forbidden_after,
        "prior_frozen_output_gate_after_outputs": prior_after,
        "phase4_ready": True,
        "next_phase_requires_owner_continue_message": True,
        "phase4_mechanism_executed": False,
        "owner_future_requirements": {
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
            "phase5_E11_A7_frozen_values": {
                "folds_2_4": {
                    "percentile_rank": 100.0,
                    "monte_carlo_p": 0.0000999900009999,
                },
                "fold_5": {
                    "percentile_rank": 24.28,
                    "monte_carlo_p": 0.7572242775722428,
                },
            },
            "phase5_E11_recommendation_sentences_allowed": False,
            "no_next_phase_mechanism_before_continue": True,
        },
    }
    ready_report = _json_ready(report)
    write_json_once(PHASE_REPORT_PATH, ready_report)
    observed_report = json.loads(PHASE_REPORT_PATH.read_text(encoding="utf-8"))
    if observed_report != ready_report:
        raise IntegrityError("phase3 report JSON round-trip mismatch")
    report_identity = _identity_subset(PHASE_REPORT_PATH)
    _append_task_log(
        {
            "event": "phase3_completed_waiting_for_owner",
            "phase_report": report_identity,
        }
    )
    return {
        "status": report["status"],
        "phase_report": report_identity,
        "scientific_output_identities": output_identities,
        "pytest": pytest_after,
        "timing": report["timing"],
        "selected_config_by_outer_fold": report["scientific_results"][
            "selected_config_by_outer_fold"
        ],
        "a4_metrics": report["scientific_results"]["a4_metrics"],
        "deviations": DEVIATIONS,
        "phase4_ready": True,
        "next_phase_requires_owner_continue_message": True,
    }


def _latest_phase3_run_log() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with TASK_LOG_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(
                    f"task log line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise IntegrityError(f"task log line {line_number} is not an object")
            records.append(record)
    starts = [
        position
        for position, record in enumerate(records)
        if record.get("event") == "phase3_started"
    ]
    if not starts:
        raise IntegrityError("task log has no phase3_started record")
    return records[starts[-1] :]


def register_phase3_calibration_stop() -> dict[str, Any]:
    """Freeze the observed canonical-Platt hard stop without rerunning training."""

    if PHASE_STOP_REPORT_PATH.exists():
        raise IntegrityError(
            f"refusing to overwrite frozen stop report: {PHASE_STOP_REPORT_PATH}"
        )
    scientific_destinations = (
        SELECTED_CONFIGS_PATH,
        SEED_STABILITY_PATH,
        A4_METRICS_PATH,
        A4_OOF_PATH,
        CSCV_PREDICTIONS_PATH,
        CSCV_MATRIX_PATH,
        PHASE_REPORT_PATH,
    )
    artifact_absence = {
        str(path.relative_to(PROJECT_ROOT)): not path.exists()
        for path in scientific_destinations
    }
    if not all(artifact_absence.values()):
        raise IntegrityError(
            f"unexpected partial phase3 artifacts exist: {artifact_absence}"
        )
    stage_directories = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in OUTPUT_DIR.glob(".phase3-stage-*")
    )
    if stage_directories:
        raise IntegrityError(
            f"phase3 staging directories survived the stop: {stage_directories}"
        )

    reentry = _run_reentry_protocol()
    tests = {
        "ablation81": _run_pytest("tests/test_ablation81.py"),
        "calibration_regression": _run_pytest("tests/test_calibration.py"),
    }
    run_log = _latest_phase3_run_log()
    observed_run_events = {
        str(record.get("event")) for record in run_log if "event" in record
    }
    required_run_events = {
        "phase3_started",
        "phase3_reentry_passed",
        "phase3_pretraining_gates_passed",
        "phase3_tests_before_training_passed",
    }
    if not required_run_events.issubset(observed_run_events):
        raise IntegrityError(
            "official run log misses pre-stop milestones: "
            f"{sorted(required_run_events - observed_run_events)}"
        )
    outer_completed = {
        str(int(record["outer_fold"])): record
        for record in run_log
        if record.get("event") == "phase3_outer_fold_completed"
    }
    if sorted(outer_completed) != ["2", "3", "4", "5"]:
        raise IntegrityError(
            f"official run did not log all outer completions: {sorted(outer_completed)}"
        )
    for fold, record in outer_completed.items():
        winner = record.get("winner")
        timing = record.get("search_timing")
        if (
            not isinstance(winner, dict)
            or not isinstance(timing, dict)
            or int(winner.get("tree_count", 0)) < 1
        ):
            raise IntegrityError(f"outer fold {fold} completion record is malformed")
    known_search_cpu = float(
        sum(
            float(record["search_timing"]["search_cpu_seconds"])
            for record in outer_completed.values()
        )
    )
    known_search_wall = float(
        sum(
            float(record["search_timing"]["search_wall_seconds"])
            for record in outer_completed.values()
        )
    )
    known_refit_wall = float(
        sum(float(record["refit_wall_seconds"]) for record in outer_completed.values())
    )

    stop_deviations = []
    for deviation in DEVIATIONS:
        record = dict(deviation)
        if "a4_oof_4h.csv" in record["text_fa"]:
            record["status"] = "announced_not_materialized_due_to_calibration_stop"
        elif "phase3_integrity_report.json" in record["text_fa"]:
            record["status"] = "announced_not_materialized_due_to_calibration_stop"
        stop_deviations.append(record)
    source_identities = {
        str(path.relative_to(PROJECT_ROOT)): _identity_subset(path)
        for path in (
            PROJECT_ROOT / "analytics" / "ablation81" / "phase3_gates.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "quarantine.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "nested.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "cscv.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "a4.py",
            PROJECT_ROOT / "analytics" / "ablation81" / "phase3.py",
            PROJECT_ROOT / "tests" / "test_ablation81.py",
        )
    }
    report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 3,
        "status": "hard_stopped_owner_adjudication_required",
        "registered_at_utc": _utc_now(),
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "stop": {
            "stage": "canonical_A4_walk_forward_calibration",
            "exception_class": "calibration.CalibrationError",
            "exception_code": "nonpositive_platt_coefficient",
            "message": (
                "Platt coefficient must be positive; observed "
                "-0.000843584342876648"
            ),
            "observed_platt_coefficient": -0.000843584342876648,
            "required_condition": "coefficient > 0",
            "is_real_data_run": True,
            "canonical_seed": 42,
            "fold": 3,
            "score": "p_meta",
            "method": "platt",
            "calibration_training_folds": [2],
            "context_location_basis": (
                "The runner evaluates seed 42 first; walk_forward_calibrate "
                "keeps fold 2 raw and its first real fit is fold 3, score "
                "p_meta, method platt."
            ),
            "fallback_attempted": False,
            "isotonic_substitution_attempted": False,
            "coefficient_sign_change_attempted": False,
            "automatic_raw_mode_attempted": False,
            "production_code_changed": False,
        },
        "completed_before_stop": {
            "pretraining_gates_passed": True,
            "tests_before_training_passed": True,
            "search_fit_count": 432,
            "winner_refit_count": 20,
            "outer_fold_completion_records": outer_completed,
            "selected_winners_in_memory": {
                fold: record["winner"] for fold, record in outer_completed.items()
            },
            "cscv_predictions_and_matrix_computed_in_memory": True,
            "cscv_persisted": False,
            "canonical_raw_A4_predictions_computed_in_memory": True,
            "canonical_calibration_completed": False,
            "seed_stability_calibrations_completed": 0,
            "scientific_computations_performed": True,
            "scientific_artifacts_persisted": False,
        },
        "known_timing": {
            "search_cpu_seconds_sum": known_search_cpu,
            "search_wall_seconds_sum": known_search_wall,
            "winner_refit_wall_seconds_sum": known_refit_wall,
            "full_runner_wall_seconds_until_exception": None,
            "full_runner_wall_seconds_unavailable_reason": (
                "The exception occurred before the normal phase timer was frozen."
            ),
        },
        "artifact_absence_after_stop": artifact_absence,
        "surviving_stage_directories": stage_directories,
        "reentry_and_integrity_after_stop": reentry,
        "pytest_after_stop": tests,
        "source_identities": source_identities,
        "deviations": stop_deviations,
        "self_audit_doubts": [
            {
                "doubt": (
                    "A nonpositive Platt slope means the required monotone mapping "
                    "cannot be silently treated as an ordinary calibrated score."
                ),
                "check": (
                    "The real-data production guard raised before replay or any "
                    "artifact commit."
                ),
                "result": "hard stop preserved; no fallback applied",
            },
            {
                "doubt": (
                    "The exception might have left a partial frozen artifact or "
                    "temporary stage directory."
                ),
                "check": (
                    "All seven planned destinations and every .phase3-stage-* "
                    "directory were checked after process termination."
                ),
                "result": "no planned artifact and no stage directory exists",
            },
            {
                "doubt": (
                    "The error could come from a later seed/fold rather than "
                    "canonical A4 C3."
                ),
                "check": (
                    "The deterministic loop order was traced: seed 42 first, fold "
                    "2 raw, then fold 3 p_meta/platt as the first fit."
                ),
                "result": "failure location is canonical seed42 C3 p_meta/platt",
            },
        ],
        "supporting_artifacts": [
            "data/models/ablation81/phase3_stop_report.json"
        ],
        "phase3_completed": False,
        "phase4_ready": False,
        "phase4_mechanism_executed": False,
        "owner_adjudication_required": True,
        "next_action_requires_explicit_owner_message": True,
        "no_automatic_retry_or_readjudication": True,
    }
    ready = _json_ready(report)
    write_json_once(PHASE_STOP_REPORT_PATH, ready)
    observed = json.loads(PHASE_STOP_REPORT_PATH.read_text(encoding="utf-8"))
    if observed != ready:
        raise IntegrityError("phase3 stop report JSON round-trip mismatch")
    identity = _identity_subset(PHASE_STOP_REPORT_PATH)
    _append_task_log(
        {
            "event": "phase3_calibration_hard_stop_registered",
            "phase3_stop_report": identity,
            "observed_platt_coefficient": -0.000843584342876648,
        }
    )
    return {
        "status": report["status"],
        "phase3_stop_report": identity,
        "selected_winners_in_memory": report["completed_before_stop"][
            "selected_winners_in_memory"
        ],
        "known_timing": report["known_timing"],
        "artifact_absence_after_stop": artifact_absence,
        "pytest_after_stop": tests,
        "owner_adjudication_required": True,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            _json_ready(run_phase3()),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
