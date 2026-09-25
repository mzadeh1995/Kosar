# ==============================================================================
# analytics/ablation81/adjudication_restore.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Narrow owner-adjudication gate for the exact backup-restore signature."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .integrity import capture_forbidden_files_fingerprint, file_identity


OWNER_ARTIFACT_RELATIVE = Path(
    "data/models/ablation81/owner_adjudication_backup_restore.json"
)
FINGERPRINT_RELATIVE = Path(
    "data/models/ablation81/forbidden_files_fingerprint.json"
)
EXPECTED_OWNER_ARTIFACT_IDENTITY = {
    "size_bytes": 1_635_271,
    "sha256": "95ebc2b1d02e43fd4201e478fef1ef82006b8c93b21d08579dce252b38b13013",
}
EXPECTED_FINGERPRINT_IDENTITY = {
    "size_bytes": 13_209_954,
    "sha256": "f14182e35749f9c37d34282fe5b130a672f20e8f585daded838ef989d87942f1",
}
EXPECTED_MISMATCH_SIGNATURE = {
    "frozen_certified_record_count": 42_693,
    "current_certified_record_count": 42_693,
    "added_count": 0,
    "removed_count": 0,
    "changed_count": 1_984,
    "mtime_only_count": 1_983,
    "byte_or_content_identity_changed_count": 1,
    "byte_or_content_identity_changed_paths": [
        ".pytest_cache/v/cache/nodeids"
    ],
    "mode_changed_count": 0,
    "canonical_mismatch_manifest_sha256": (
        "1d8234a000eb4a455896f16d0942af1c80f00efa94ccc53b047142fb74022e2a"
    ),
    "canonical_mismatch_manifest_utf8_bytes": 1_165_217,
}


class RestoreAdjudicationMismatch(RuntimeError):
    """Raised when the owner-authorized restore signature changes."""


def _identity_pair(path: Path) -> dict[str, Any]:
    observed = file_identity(path)
    return {
        "size_bytes": int(observed["size_bytes"]),
        "sha256": str(observed["sha256"]),
    }


def _canonical_mismatch(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fingerprint_path = project_root / FINGERPRINT_RELATIVE
    frozen = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    frozen_records = {
        str(record["path"]): record
        for record in frozen["records"]
        if Path(str(record.get("path", ""))).name != ".DS_Store"
    }
    observed = capture_forbidden_files_fingerprint(project_root)
    current_records = {
        str(record["path"]): record for record in observed["records"]
    }
    added = sorted(set(current_records) - set(frozen_records))
    removed = sorted(set(frozen_records) - set(current_records))
    changed: list[dict[str, Any]] = []
    for relative in sorted(set(frozen_records) & set(current_records)):
        before = frozen_records[relative]
        after = current_records[relative]
        if before == after:
            continue
        differing = sorted(
            key
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        )
        changed.append(
            {
                "path": relative,
                "differing_fields": differing,
                "frozen_record": before,
                "observed_record": after,
            }
        )
    canonical = {"added": added, "removed": removed, "changed": changed}
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    byte_changed = [
        record["path"]
        for record in changed
        if any(
            field in record["differing_fields"]
            for field in ("sha256", "size_bytes", "type", "link_target")
        )
    ]
    signature = {
        "frozen_certified_record_count": len(frozen_records),
        "current_certified_record_count": len(current_records),
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "mtime_only_count": sum(
            record["differing_fields"] == ["mtime_ns"] for record in changed
        ),
        "byte_or_content_identity_changed_count": len(byte_changed),
        "byte_or_content_identity_changed_paths": byte_changed,
        "mode_changed_count": sum(
            "mode" in record["differing_fields"] for record in changed
        ),
        "canonical_mismatch_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_mismatch_manifest_utf8_bytes": len(encoded),
    }
    return canonical, {
        "signature": signature,
        "ignored_current_exact_ds_store_count": int(
            observed["ignored_current_record_count"]
        ),
    }


def verify_owner_adjudicated_restore(project_root: Path) -> dict[str, Any]:
    """Require the exact owner artifact and exact live restore mismatch."""

    root = project_root.resolve(strict=True)
    owner_path = root / OWNER_ARTIFACT_RELATIVE
    owner_identity = _identity_pair(owner_path)
    if owner_identity != EXPECTED_OWNER_ARTIFACT_IDENTITY:
        raise RestoreAdjudicationMismatch(
            "owner restore artifact identity changed: "
            f"expected={EXPECTED_OWNER_ARTIFACT_IDENTITY}, "
            f"observed={owner_identity}"
        )
    document = json.loads(owner_path.read_text(encoding="utf-8"))
    if document.get("status") != "owner_adjudication_registered":
        raise RestoreAdjudicationMismatch("owner restore artifact status changed")
    if document.get("authorization_scope") != {
        "exact_signature_only": True,
        "phase4_start_authorized": True,
        "frozen_fingerprint_rebuilt_or_deleted": False,
        "writes_outside_allowed_paths_authorized": False,
        "general_mtime_exception_added": False,
        "general_cache_exception_added": False,
        "automatic_expansion_or_readjudication_allowed": False,
        "any_other_mismatch_requires_new_hard_stop": True,
    }:
        raise RestoreAdjudicationMismatch(
            "owner restore authorization scope changed"
        )
    fingerprint_path = root / FINGERPRINT_RELATIVE
    fingerprint_identity = _identity_pair(fingerprint_path)
    if fingerprint_identity != EXPECTED_FINGERPRINT_IDENTITY:
        raise RestoreAdjudicationMismatch(
            "frozen forbidden fingerprint identity changed"
        )
    canonical, live = _canonical_mismatch(root)
    signature = live["signature"]
    if signature != EXPECTED_MISMATCH_SIGNATURE:
        raise RestoreAdjudicationMismatch(
            "live backup-restore mismatch differs from the exact owner signature: "
            f"expected={EXPECTED_MISMATCH_SIGNATURE}, observed={signature}"
        )
    if canonical != document.get("canonical_mismatch_manifest"):
        raise RestoreAdjudicationMismatch(
            "owner artifact mismatch manifest differs from live exact records"
        )
    artifact_signature = dict(document.get("adjudicated_exact_restore_signature", {}))
    artifact_signature.pop("ignored_current_exact_ds_store_count", None)
    if artifact_signature != EXPECTED_MISMATCH_SIGNATURE:
        raise RestoreAdjudicationMismatch(
            "owner artifact restore signature changed"
        )
    return {
        "status": "passed_by_exact_owner_adjudication",
        "owner_artifact": {
            "path": OWNER_ARTIFACT_RELATIVE.as_posix(),
            **owner_identity,
        },
        "frozen_forbidden_fingerprint": {
            "path": FINGERPRINT_RELATIVE.as_posix(),
            **fingerprint_identity,
            "rebuilt_or_deleted": False,
        },
        "exact_signature": signature,
        "ignored_current_exact_ds_store_count": live[
            "ignored_current_exact_ds_store_count"
        ],
        "general_exception_added": False,
        "any_other_mismatch_requires_new_hard_stop": True,
    }

