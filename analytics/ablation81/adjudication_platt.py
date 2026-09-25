# ==============================================================================
# analytics/ablation81/adjudication_platt.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Exact owner-adjudication gate for the canonical A4 Platt rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .integrity import file_identity


OWNER_ARTIFACT_RELATIVE = Path(
    "data/models/ablation81/owner_adjudication_platt.json"
)
PHASE3_STOP_REPORT_RELATIVE = Path(
    "data/models/ablation81/phase3_stop_report.json"
)
EXPECTED_OWNER_ARTIFACT_IDENTITY = {
    "size_bytes": 12_933,
    "sha256": "ca7e03c46f51e8f61f111c2643dd3f1fc51b23f83f40fb03155b30bee91c834f",
}
EXPECTED_OWNER_MESSAGE_IDENTITY = {
    "characters": 7_027,
    "utf8_bytes": 11_570,
    "sha256": "24c14bf599008e003cc9046d2a249c6b7448a36f8b44a0ce25e1593203e459e9",
}
EXPECTED_PHASE3_STOP_IDENTITY = {
    "size_bytes": 44_357,
    "sha256": "7cb2fb30e907f10ff7b80e407396eeedf2245823ee2bb7b292324145e70f19ea",
}
EXPECTED_ADJUDICATED_SIGNATURE = {
    "canonical_seed": 42,
    "calibrator": "C3",
    "calibration_training_folds": [2],
    "score": "p_meta",
    "method": "platt",
    "coefficient": -0.000843584342876648,
    "exception_class": "calibration.CalibrationError",
    "exception_code": "nonpositive_platt_coefficient",
}
EXPECTED_AUTHORIZATION_SCOPE = {
    "exact_signature_only": True,
    "classification": "scientific_result",
    "phase3_completion_allowed": True,
    "fallback_allowed": False,
    "automatic_expansion_or_readjudication_allowed": False,
    "allowed_phase_level_rejection_codes": [
        "nonpositive_platt_coefficient",
        "nonfinite_platt_parameters",
    ],
    "phase4_requires_separate_owner_continue": True,
}
AUTHORIZED_REJECTION_CODES = frozenset(
    EXPECTED_AUTHORIZATION_SCOPE["allowed_phase_level_rejection_codes"]
)


class PlattOwnerAdjudicationMismatch(RuntimeError):
    """Raised when the exact owner-authorized Platt signature changes."""


def _identity_subset(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    return {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def verify_owner_adjudication_platt(project_root: Path) -> dict[str, Any]:
    """Verify artifact bytes, verbatim message, signature, scope, and stop link."""

    root = project_root.resolve(strict=True)
    owner_path = root / OWNER_ARTIFACT_RELATIVE
    owner_identity = _identity_subset(owner_path)
    if owner_identity != EXPECTED_OWNER_ARTIFACT_IDENTITY:
        raise PlattOwnerAdjudicationMismatch(
            "owner Platt artifact identity changed: "
            f"expected={EXPECTED_OWNER_ARTIFACT_IDENTITY}, "
            f"observed={owner_identity}"
        )
    try:
        document = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlattOwnerAdjudicationMismatch(
            f"cannot read owner Platt adjudication: {exc}"
        ) from exc
    required_top_level = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 3,
        "frozen": True,
        "status": "owner_adjudication_registered",
    }
    for key, expected in required_top_level.items():
        if document.get(key) != expected:
            raise PlattOwnerAdjudicationMismatch(
                f"owner Platt {key} expected={expected!r}, "
                f"observed={document.get(key)!r}"
            )
    message = document.get("owner_message_verbatim_fa")
    if not isinstance(message, str):
        raise PlattOwnerAdjudicationMismatch(
            "owner_message_verbatim_fa must be text"
        )
    encoded = message.encode("utf-8")
    message_identity = {
        "characters": len(message),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if (
        message_identity != EXPECTED_OWNER_MESSAGE_IDENTITY
        or document.get("owner_message_identity") != EXPECTED_OWNER_MESSAGE_IDENTITY
    ):
        raise PlattOwnerAdjudicationMismatch(
            "verbatim owner Platt message identity changed"
        )
    if document.get("adjudicated_signature") != EXPECTED_ADJUDICATED_SIGNATURE:
        raise PlattOwnerAdjudicationMismatch(
            "owner-adjudicated Platt signature changed"
        )
    if document.get("authorization_scope") != EXPECTED_AUTHORIZATION_SCOPE:
        raise PlattOwnerAdjudicationMismatch(
            "owner Platt authorization scope changed"
        )

    stop_path = root / PHASE3_STOP_REPORT_RELATIVE
    stop_identity = _identity_subset(stop_path)
    expected_stop_record = {
        "path": PHASE3_STOP_REPORT_RELATIVE.as_posix(),
        **EXPECTED_PHASE3_STOP_IDENTITY,
    }
    if (
        stop_identity != EXPECTED_PHASE3_STOP_IDENTITY
        or document.get("phase3_stop_report_identity") != expected_stop_record
    ):
        raise PlattOwnerAdjudicationMismatch(
            "phase3 stop report differs from the adjudicated report"
        )
    try:
        stop_document = json.loads(stop_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlattOwnerAdjudicationMismatch(
            f"cannot read adjudicated phase3 stop report: {exc}"
        ) from exc
    stop_signature = stop_document.get("stop", {})
    expected_stop_signature = {
        "canonical_seed": 42,
        "calibration_training_folds": [2],
        "score": "p_meta",
        "method": "platt",
        "observed_platt_coefficient": -0.000843584342876648,
        "exception_class": "calibration.CalibrationError",
        "exception_code": "nonpositive_platt_coefficient",
        "message": (
            "Platt coefficient must be positive; observed "
            "-0.000843584342876648"
        ),
    }
    for key, expected in expected_stop_signature.items():
        if stop_signature.get(key) != expected:
            raise PlattOwnerAdjudicationMismatch(
                "phase3 stop-report signature changed at "
                f"{key}: expected={expected!r}, "
                f"observed={stop_signature.get(key)!r}"
            )
    return {
        "status": "passed",
        "owner_artifact": {
            "path": OWNER_ARTIFACT_RELATIVE.as_posix(),
            **owner_identity,
        },
        "owner_message": message_identity,
        "adjudicated_signature": EXPECTED_ADJUDICATED_SIGNATURE,
        "authorization_scope": EXPECTED_AUTHORIZATION_SCOPE,
        "phase3_stop_report": {
            "path": PHASE3_STOP_REPORT_RELATIVE.as_posix(),
            **stop_identity,
        },
        "phase3_stop_signature": expected_stop_signature,
    }


def assert_exact_authorized_rejection(
    *,
    seed: int,
    calibrator: str,
    training_folds: list[int],
    score: str,
    method: str,
    coefficient: float,
    exception_class: str,
    exception_code: str,
) -> dict[str, Any]:
    """Allow only the exact adjudicated C3 rejection signature."""

    observed = {
        "canonical_seed": int(seed),
        "calibrator": str(calibrator),
        "calibration_training_folds": [int(item) for item in training_folds],
        "score": str(score),
        "method": str(method),
        "coefficient": float(coefficient),
        "exception_class": str(exception_class),
        "exception_code": str(exception_code),
    }
    if observed != EXPECTED_ADJUDICATED_SIGNATURE:
        raise PlattOwnerAdjudicationMismatch(
            "observed Platt rejection differs from exact owner signature: "
            f"expected={EXPECTED_ADJUDICATED_SIGNATURE}, observed={observed}"
        )
    return {
        "status": "passed",
        "exact_signature_match": True,
        "signature": observed,
    }


def assert_allowed_family_rejection_code(code: str) -> dict[str, Any]:
    """Authorize only the two owner-enumerated construction-rejection codes."""

    observed = str(code)
    if observed not in AUTHORIZED_REJECTION_CODES:
        raise PlattOwnerAdjudicationMismatch(
            f"unadjudicated calibration exception code: {observed!r}"
        )
    return {
        "status": "passed",
        "code": observed,
        "allowed_codes": sorted(AUTHORIZED_REJECTION_CODES),
        "automatic_expansion_allowed": False,
    }
