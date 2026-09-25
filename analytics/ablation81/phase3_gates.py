# ==============================================================================
# analytics/ablation81/phase3_gates.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Hard-stop assembly gates for ablation81 phase three."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import meta_model

from .adjudication import (
    assert_owner_adjudicated_funding_signature,
    verify_owner_adjudication_funding,
)
from .integrity import (
    file_identity,
    load_calibrated_events,
    verify_research_input_lineage,
)


EXPECTED_PAIR_COUNT = 11_029
ALIGNMENT_COLUMNS = ("p_primary", "tb_return", "tb_uniqueness", "fold")
ALIGNMENT_ATOL = 1e-9
EXPECTED_CANONICAL_INPUT_COUNT = 35
EXPECTED_LINEAGE_IDENTITY = {
    "size_bytes": 27_732,
    "sha256": "b527aaed547c7dabf51a9a2d613dac3b27cd34e631b7cec385be17a534f40612",
}


class Phase3GateError(RuntimeError):
    """Raised when any phase-three pre-training gate fails."""


def _identity_subset(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    return {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def build_canonical_assembly(project_root: Path) -> dict[str, Any]:
    """Import and call the production assembly routine without modifying it."""

    root = project_root.resolve(strict=True)
    assembly = meta_model.build_meta_dataset(
        meta_model.CANONICAL_SYMBOLS,
        root / "data" / "datasets_4h",
        root / "data" / "external",
        timeframe="4h",
        hmm_dir=root / "log" / "hmm_walkforward_fwd_long",
        primary_dir=root / "data" / "models",
        is_real_data_run=True,
    )
    required = {"events", "X", "feature_columns", "categorical_features", "metadata"}
    missing = sorted(required - set(assembly))
    if missing:
        raise Phase3GateError(f"canonical assembly misses keys: {missing}")
    expected_features = tuple(meta_model.META_FEATURE_COLUMNS) + tuple(
        meta_model.META_HMM_FEATURE_COLUMNS
    )
    if tuple(assembly["feature_columns"]) != expected_features:
        raise Phase3GateError("assembled feature columns differ from production constants")
    if tuple(assembly["categorical_features"]) != ("symbol", "hmm_regime"):
        raise Phase3GateError("assembled categorical features differ from production")
    return assembly


def _normalized_key_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"symbol", "decision_ts", *ALIGNMENT_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Phase3GateError(f"{label} population misses columns: {missing}")
    normalized = frame.loc[:, ["symbol", "decision_ts", *ALIGNMENT_COLUMNS]].copy()
    normalized["symbol"] = normalized["symbol"].astype(str)
    normalized["decision_ts"] = pd.to_datetime(
        normalized["decision_ts"], utc=True, errors="raise"
    )
    if normalized.duplicated(["symbol", "decision_ts"]).any():
        duplicates = int(normalized.duplicated(["symbol", "decision_ts"], keep=False).sum())
        raise Phase3GateError(
            f"{label} population contains duplicate normalized keys: {duplicates}"
        )
    return normalized


def assert_population_alignment(
    assembled_events: pd.DataFrame,
    pinned_events: pd.DataFrame,
) -> dict[str, Any]:
    """Require exact keys and rowwise agreement with the seven pinned files."""

    assembled = _normalized_key_frame(assembled_events, label="assembled")
    pinned = _normalized_key_frame(pinned_events, label="pinned")
    if len(assembled) != EXPECTED_PAIR_COUNT or len(pinned) != EXPECTED_PAIR_COUNT:
        raise Phase3GateError(
            "population pair count changed: "
            f"assembled={len(assembled)}, pinned={len(pinned)}, "
            f"expected={EXPECTED_PAIR_COUNT}"
        )
    merged = assembled.merge(
        pinned,
        on=["symbol", "decision_ts"],
        how="outer",
        suffixes=("_assembled", "_pinned"),
        indicator=True,
        validate="one_to_one",
        sort=False,
    )
    membership_counts = {
        str(key): int(value)
        for key, value in merged["_merge"].value_counts(dropna=False).items()
    }
    if (
        membership_counts.get("both", 0) != EXPECTED_PAIR_COUNT
        or membership_counts.get("left_only", 0) != 0
        or membership_counts.get("right_only", 0) != 0
    ):
        raise Phase3GateError(
            f"population key-set mismatch: {membership_counts}"
        )

    max_diff: dict[str, float] = {}
    mismatch_count: dict[str, int] = {}
    for column in ALIGNMENT_COLUMNS:
        left = pd.to_numeric(
            merged[f"{column}_assembled"], errors="raise"
        ).to_numpy(dtype=float)
        right = pd.to_numeric(
            merged[f"{column}_pinned"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise Phase3GateError(f"population alignment {column} is non-finite")
        difference = np.abs(left - right)
        max_diff[column] = float(difference.max(initial=0.0))
        mismatch_count[column] = int(np.count_nonzero(difference > ALIGNMENT_ATOL))
        if mismatch_count[column]:
            raise Phase3GateError(
                f"population alignment {column} has "
                f"{mismatch_count[column]} differences above {ALIGNMENT_ATOL}"
            )
    return {
        "status": "passed",
        "pair_count": EXPECTED_PAIR_COUNT,
        "key_membership_counts": membership_counts,
        "absolute_tolerance": ALIGNMENT_ATOL,
        "relative_tolerance": 0.0,
        "max_abs_diff_by_column": max_diff,
        "mismatch_count_by_column": mismatch_count,
    }


def _unique_identity_map(
    records: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    expected: bool,
) -> tuple[dict[tuple[str, str, str], tuple[int, str]], Counter[str]]:
    result: dict[tuple[str, str, str], tuple[int, str]] = {}
    roles: Counter[str] = Counter()
    for position, record in enumerate(records):
        required = {"path", "role", "symbol", "size_bytes", "sha256"}
        missing = sorted(required - set(record))
        if missing:
            raise Phase3GateError(
                f"input identity record {position} misses keys: {missing}"
            )
        role = str(record["role"])
        if expected:
            if not role.startswith("canonical_"):
                raise Phase3GateError(
                    f"lineage canonical role lacks prefix: {role!r}"
                )
            role = role.removeprefix("canonical_")
            relative = Path(str(record["path"])).as_posix()
        else:
            absolute = Path(str(record["path"])).resolve(strict=True)
            try:
                relative = absolute.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise Phase3GateError(
                    f"assembly input is outside project root: {absolute}"
                ) from exc
        key = (role, str(record["symbol"]), relative)
        if key in result:
            raise Phase3GateError(f"duplicate input identity key: {key}")
        size = record["size_bytes"]
        sha256 = str(record["sha256"])
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or len(sha256) != 64
        ):
            raise Phase3GateError(f"malformed input identity for {relative}")
        result[key] = (int(size), sha256)
        roles[role] += 1
    return result, roles


def assert_input_files_lineage(
    metadata: Mapping[str, Any],
    *,
    project_root: Path,
    lineage_path: Path,
) -> dict[str, Any]:
    """Require the literal assembly ``input_files`` key and its frozen identity."""

    if not isinstance(metadata, Mapping):
        raise Phase3GateError("assembly metadata is not a mapping")
    if "input_files" not in metadata:
        raise Phase3GateError(
            "assembly metadata lacks required literal input_files key"
        )
    actual_records = metadata["input_files"]
    if not isinstance(actual_records, list):
        raise Phase3GateError("assembly metadata input_files must be a list")

    observed_lineage_identity = _identity_subset(lineage_path)
    if observed_lineage_identity != EXPECTED_LINEAGE_IDENTITY:
        raise Phase3GateError(
            "frozen lineage identity changed: "
            f"expected={EXPECTED_LINEAGE_IDENTITY}, "
            f"observed={observed_lineage_identity}"
        )
    lineage_gate = verify_research_input_lineage(
        lineage_path, base_dir=project_root
    )
    document = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage_records = document.get("inputs")
    if not isinstance(lineage_records, list):
        raise Phase3GateError("frozen lineage inputs must be a list")
    expected_records = [
        record
        for record in lineage_records
        if "contract_2_6_canonical_35" in record.get("required_by", [])
    ]
    if len(expected_records) != EXPECTED_CANONICAL_INPUT_COUNT:
        raise Phase3GateError(
            "frozen canonical input subset changed: "
            f"observed={len(expected_records)}"
        )
    actual_map, actual_roles = _unique_identity_map(
        actual_records,
        project_root=project_root,
        expected=False,
    )
    expected_map, expected_roles = _unique_identity_map(
        expected_records,
        project_root=project_root,
        expected=True,
    )
    if actual_map != expected_map:
        actual_keys = set(actual_map)
        expected_keys = set(expected_map)
        changed = sorted(
            key
            for key in actual_keys & expected_keys
            if actual_map[key] != expected_map[key]
        )
        raise Phase3GateError(
            "assembly input_files differ from frozen canonical lineage: "
            f"added={sorted(actual_keys - expected_keys)}, "
            f"missing={sorted(expected_keys - actual_keys)}, changed={changed}"
        )
    if actual_roles != expected_roles:
        raise Phase3GateError(
            f"assembly input role counts changed: {dict(actual_roles)}"
        )
    return {
        "status": "passed",
        "input_file_count": len(actual_map),
        "lineage_verified_count": int(lineage_gate["verified_count"]),
        "lineage_identity": observed_lineage_identity,
        "role_counts": dict(sorted(actual_roles.items())),
        "literal_input_files_key_present": True,
        "exact_map_match": True,
    }


def run_pretraining_gates(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble once and execute the three hard gates in contract order."""

    root = project_root.resolve(strict=True)
    assembly = build_canonical_assembly(root)
    pinned = load_calibrated_events(root)
    population_gate = assert_population_alignment(assembly["events"], pinned)
    lineage_gate = assert_input_files_lineage(
        assembly["metadata"],
        project_root=root,
        lineage_path=root / "data" / "models" / "ablation81"
        / "research_input_lineage.json",
    )
    owner_gate = verify_owner_adjudication_funding(root)
    funding_gate = assert_owner_adjudicated_funding_signature(assembly["events"])
    return assembly, {
        "status": "passed",
        "execution_order": [
            "population_alignment",
            "input_files_lineage",
            "owner_adjudicated_funding_signature",
        ],
        "population_alignment": population_gate,
        "input_files_lineage": lineage_gate,
        "owner_adjudication_artifact": owner_gate,
        "funding_signature": funding_gate,
    }
