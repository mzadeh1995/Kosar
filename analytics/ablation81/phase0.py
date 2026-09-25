# ==============================================================================
# analytics/ablation81/phase0.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Execute and record phase zero of the ablation81 v7 contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import (
    IntegrityError,
    atomic_write_json,
    calculate_s2,
    capture_forbidden_files_fingerprint,
    collect_environment_fingerprint,
    file_identity,
    load_calibrated_events,
    reconcile_s2,
    verify_environment_fingerprint,
    verify_forbidden_files_fingerprint,
    verify_pinned_artifacts,
    verify_production_constants,
    verify_s1_anchors,
    write_json_once,
)


FROZEN_THREAD_COUNT = 3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "ablation81"
ENVIRONMENT_PATH = OUTPUT_DIR / "environment_fingerprint.json"
FORBIDDEN_FINGERPRINT_PATH = OUTPUT_DIR / "forbidden_files_fingerprint.json"
PHASE_REPORT_PATH = OUTPUT_DIR / "phase0_integrity_report.json"
TASK_LOG_PATH = OUTPUT_DIR / "ablation81_run.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_task_log(event: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TASK_LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_pytest(target: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        target,
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


def _initialize_frozen_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_environment = collect_environment_fingerprint(
        thread_count=FROZEN_THREAD_COUNT
    )
    if ENVIRONMENT_PATH.exists():
        environment_check = verify_environment_fingerprint(
            ENVIRONMENT_PATH, observed_environment
        )
    else:
        write_json_once(ENVIRONMENT_PATH, observed_environment)
        environment_check = verify_environment_fingerprint(
            ENVIRONMENT_PATH, observed_environment
        )

    if FORBIDDEN_FINGERPRINT_PATH.exists():
        forbidden_check = verify_forbidden_files_fingerprint(
            FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
        )
    else:
        forbidden_fingerprint = capture_forbidden_files_fingerprint(PROJECT_ROOT)
        write_json_once(FORBIDDEN_FINGERPRINT_PATH, forbidden_fingerprint)
        forbidden_check = verify_forbidden_files_fingerprint(
            FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
        )
    return environment_check, forbidden_check


def run_phase0() -> dict[str, Any]:
    if PHASE_REPORT_PATH.exists():
        raise IntegrityError(f"phase-zero report is already frozen: {PHASE_REPORT_PATH}")

    phase_started_wall = _utc_now()
    phase_started = time.perf_counter()
    environment_check, forbidden_initial = _initialize_frozen_artifacts()
    _append_task_log(
        {
            "event": "phase0_started",
            "timestamp_utc": phase_started_wall,
            "thread_count": FROZEN_THREAD_COUNT,
        }
    )

    pinned = verify_pinned_artifacts(PROJECT_ROOT)
    events = load_calibrated_events(PROJECT_ROOT)
    s1 = verify_s1_anchors(events)
    production_constants = verify_production_constants()
    s2_observed = calculate_s2(
        events, economic_cost=production_constants["economic_cost"]
    )
    s2_reconciliation = reconcile_s2(s2_observed)

    forbidden_after_integrity = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
    )
    calibration_regression = _run_pytest("tests/test_calibration.py")
    forbidden_after_calibration_tests = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
    )
    phase_tests = _run_pytest("tests/test_ablation81.py")
    forbidden_after_phase_tests = verify_forbidden_files_fingerprint(
        FORBIDDEN_FINGERPRINT_PATH, project_root=PROJECT_ROOT
    )

    if calibration_regression["returncode"] != 0:
        raise IntegrityError("pre-task calibration regression failed")
    if phase_tests["returncode"] != 0:
        raise IntegrityError("phase-zero pinned tests failed")

    protected_paths = (
        "PROJECT_CONTEXT.md",
        "config.py",
        "log/main.log",
        "log/main_v47_5.log",
    )
    protected_identities = {
        relative: file_identity(PROJECT_ROOT / relative) for relative in protected_paths
    }
    elapsed = time.perf_counter() - phase_started
    report = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 0,
        "status": "passed",
        "started_at_utc": phase_started_wall,
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "frozen_thread_count": FROZEN_THREAD_COUNT,
        "environment_gate": environment_check,
        "forbidden_files_gate": {
            "initial": forbidden_initial,
            "after_integrity_gates": forbidden_after_integrity,
            "after_calibration_regression": forbidden_after_calibration_tests,
            "after_phase_tests": forbidden_after_phase_tests,
        },
        "protected_file_identities": protected_identities,
        "pinned_artifact_gate": pinned,
        "s1_hard_anchors": s1,
        "production_constants_gate": production_constants,
        "s2_observed": s2_observed,
        "s2_reconciliation": s2_reconciliation,
        "pytest": {
            "pre_task_calibration_regression": calibration_regression,
            "phase0_pinned_tests": phase_tests,
        },
        "deviations": [
            {
                "kind": "additional_supporting_artifact",
                "path": str(FORBIDDEN_FINGERPRINT_PATH.relative_to(PROJECT_ROOT)),
                "effect_on_scientific_analysis": "none",
                "purpose": (
                    "Machine-verifiable final proof that every file outside the "
                    "three contract-authorized paths remains unchanged."
                ),
            }
        ],
        "self_audit_doubts": [
            {
                "doubt": (
                    "Read-only imports or pytest could silently rewrite bytecode, "
                    "cache, or protected logs."
                ),
                "check": (
                    "All subprocesses used -B and disabled pytest cache; the full "
                    "forbidden-file fingerprint was rechecked after each command."
                ),
                "result": "no forbidden file or metadata change detected",
            }
        ],
        "phase1_ready": True,
        "next_phase_requires_literal_continue_message": True,
    }
    write_json_once(PHASE_REPORT_PATH, report)
    _append_task_log(
        {
            "event": "phase0_completed",
            "timestamp_utc": report["completed_at_utc"],
            "elapsed_seconds": elapsed,
            "status": "passed",
            "s2_decision_rule_suspended": s2_reconciliation[
                "decision_rule_suspended"
            ],
            "pytest": {
                "pre_task_calibration_regression": calibration_regression["status"],
                "phase0_pinned_tests": phase_tests["status"],
            },
        }
    )
    return report


def main() -> int:
    try:
        report = run_phase0()
    except BaseException as exc:
        try:
            _append_task_log(
                {
                    "event": "phase0_stopped",
                    "timestamp_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except BaseException:
            pass
        print(f"PHASE0_STOP {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "pinned_artifacts": report["pinned_artifact_gate"][
                    "verified_count"
                ],
                "s1_status": report["s1_hard_anchors"]["status"],
                "s2_status": report["s2_reconciliation"]["status"],
                "decision_rule_suspended": report["s2_reconciliation"][
                    "decision_rule_suspended"
                ],
                "phase0_tests": report["pytest"]["phase0_pinned_tests"]["status"],
                "calibration_regression": report["pytest"][
                    "pre_task_calibration_regression"
                ]["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
