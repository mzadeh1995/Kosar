# ==============================================================================
# analytics/ablation81/integrity.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Machine-enforced integrity gates for the ablation81 v7 contract."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
)

CALIBRATED_COLUMNS = (
    "OpenTime",
    "decision_ts",
    "fold",
    "y",
    "meta_y",
    "p_primary",
    "p_meta",
    "tb_return",
    "tb_exit_index",
    "tb_uniqueness",
    "tau_meta_used",
    "tau_baseline_used",
    "tau_meta_forced_used",
    "tau_baseline_forced_used",
    "traded_meta",
    "traded_baseline",
    "meta_eval_status",
    "p_meta_cal_platt",
    "p_meta_cal_iso",
    "p_primary_cal_platt",
    "p_primary_cal_iso",
    "cal_is_raw",
)

PINNED_ARTIFACTS = (
    {
        "path": "data/models/meta_oof_cal_BTCUSDT_4h.csv",
        "size_bytes": 488_913,
        "sha256": "6edd5166ad7cbf8470995baaa68c4e18d250fed02c655ee9311b4c595dc0045b",
        "symbol": "BTCUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_ETHUSDT_4h.csv",
        "size_bytes": 401_355,
        "sha256": "2c7ccfbd3d78a19bc71a8f571cf57495ab2d28d9c569a90e0defd44cbd0f3801",
        "symbol": "ETHUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_BNBUSDT_4h.csv",
        "size_bytes": 440_885,
        "sha256": "e2cafec8922db718ef31742edd43786e0094ad812c0b0b3701e1d72c7f93f8de",
        "symbol": "BNBUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_SOLUSDT_4h.csv",
        "size_bytes": 507_478,
        "sha256": "59942b854d8851459e0f949b53bf07ef532227caa0b76b2e4e9a2ec02eb5e5d0",
        "symbol": "SOLUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_XRPUSDT_4h.csv",
        "size_bytes": 339_335,
        "sha256": "595fce504d940b84b612a9bb1ad387909a697e98b6e976115a899ec66810e3e4",
        "symbol": "XRPUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_ADAUSDT_4h.csv",
        "size_bytes": 329_514,
        "sha256": "bb143b3259500babcaf93155486b1f7d7da4326b782fe66a2a5f43cc64a8c38b",
        "symbol": "ADAUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/meta_oof_cal_AVAXUSDT_4h.csv",
        "size_bytes": 398_699,
        "sha256": "08b7125a6e970d3b80dd9fbb69560844058888ac235fbeb337d3ccde7489fe54",
        "symbol": "AVAXUSDT",
        "role": "calibrated_oof_population",
    },
    {
        "path": "data/models/calibrator_meta_4h.joblib",
        "size_bytes": 1_073,
        "sha256": "c9168dd3c99536837246779b669397d85ee578f4c3b11491f9e096b494575491",
        "symbol": None,
        "role": "pinned_calibrator",
    },
    {
        "path": "data/models/calibrator_primary_4h.joblib",
        "size_bytes": 1_073,
        "sha256": "479cf10eacb2e0ab80370251b4c28a3d4472ce5b055e1c79e954601f3d573b46",
        "symbol": None,
        "role": "pinned_calibrator",
    },
    {
        "path": "data/models/calibration_report_4h.json",
        "size_bytes": 199_107,
        "sha256": "073882f65881e2f069c42a71c30a4e01ed4c38142f1fc9eb5b952ecd6bf57ab5",
        "symbol": None,
        "role": "pinned_calibration_report",
    },
    {
        "path": "data/models/calibrator_meta_4h.json",
        "size_bytes": 32_602,
        "sha256": "5b4735c05ef1465a1f7cfc81a1b439ceac63ea7f5e0f5aac2518c0b05ac81d75",
        "symbol": None,
        "role": "pinned_calibrator_metadata",
    },
    {
        "path": "data/models/calibrator_primary_4h.json",
        "size_bytes": 32_735,
        "sha256": "efd0cd28d3ec0d7e18f24619d7039562b2632d227474ff24d9f9cf6b771cbc5d",
        "symbol": None,
        "role": "pinned_calibrator_metadata",
    },
)

S1_EXPECTED = {
    "total_rows": 11_029,
    "fold_counts": {1: 1_981, 2: 1_745, 3: 2_119, 4: 2_197, 5: 2_987},
    "evaluated_rows": 9_048,
    "evaluated_meta_y_positive": 3_293,
    "unweighted_base_rate_7dp": 0.3639478,
    "weighted_base_rate_7dp": 0.3160750,
}

S2_EXPECTED = {
    "overlap": {
        "common": {"n": 873, "sum_net": 3.9478},
        "meta_only": {"n": 629, "sum_net": -2.5732},
        "primary_only": {"n": 2_561, "sum_net": -5.8359},
    },
    "equal_n": {"n": 1_502, "meta_sum_net": 1.3747, "primary_sum_net": 6.0273},
    "auc": {
        "pooled_meta": 0.5138696,
        "pooled_primary": 0.5262030,
        "fold5_meta": 0.5011941,
        "fold5_primary": 0.4926582,
    },
    "fold5": {
        "primary_n": 740,
        "primary_sum_net": -4.4260,
        "primary_tau": 0.75,
        "meta_n": 193,
        "meta_sum_net": -0.8148,
    },
}

ALLOWED_PATHS = (
    "analytics/ablation81",
    "data/models/ablation81",
    "tests/test_ablation81.py",
)
FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME = ".DS_Store"


class IntegrityError(RuntimeError):
    """Base class for a contract-enforced stop."""


class ManifestMismatch(IntegrityError):
    """Raised when a recorded file identity does not match the file."""


class EnvironmentMismatch(IntegrityError):
    """Raised when the frozen environment differs from the observed one."""


class S1AnchorError(IntegrityError):
    """Raised when any hard S1 population anchor fails."""


class ForbiddenMutationError(IntegrityError):
    """Raised when a file outside the allowed paths changes."""


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Return the byte identity of a regular file."""

    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ManifestMismatch(f"not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": int(info.st_size),
        "sha256": sha256_file(resolved),
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically serialize JSON with deterministic formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_once(path: Path, payload: Any) -> None:
    """Create a frozen JSON artifact and refuse to overwrite it."""

    if path.exists():
        raise IntegrityError(f"refusing to overwrite frozen artifact: {path}")
    atomic_write_json(path, payload)


def collect_environment_fingerprint(*, thread_count: int) -> dict[str, Any]:
    """Collect the exact stable fields required by the environment gate."""

    if not isinstance(thread_count, int) or isinstance(thread_count, bool) or thread_count < 1:
        raise IntegrityError(f"invalid frozen thread_count: {thread_count!r}")
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "catboost": importlib.metadata.version("catboost"),
        "joblib": importlib.metadata.version("joblib"),
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "thread_count": thread_count,
    }


def verify_environment_fingerprint(
    fingerprint_path: Path,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact equality, including missing or additional fields."""

    try:
        frozen = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentMismatch(f"cannot read environment fingerprint: {exc}") from exc
    observed_plain = json.loads(json.dumps(observed, sort_keys=True))
    if frozen != observed_plain:
        raise EnvironmentMismatch(
            "environment fingerprint mismatch: "
            f"frozen={json.dumps(frozen, sort_keys=True)} "
            f"observed={json.dumps(observed_plain, sort_keys=True)}"
        )
    return {"status": "passed", "fingerprint": frozen}


def _resolve_recorded_path(
    recorded_path: str,
    *,
    document_path: Path,
    base_dir: Path | None,
    document_root: str | None,
) -> Path:
    candidate = Path(recorded_path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        return base_dir / candidate
    if document_root is not None:
        root = Path(document_root)
        if not root.is_absolute():
            root = document_path.parent / root
        return root / candidate
    return document_path.parent / candidate


def _normalize_identity_records(
    records: Any,
    *,
    document_label: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ManifestMismatch(f"{document_label} must contain a non-empty record list")
    normalized: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ManifestMismatch(f"{document_label} record {position} is not an object")
        missing = {"path", "size_bytes", "sha256"} - set(record)
        if missing:
            raise ManifestMismatch(
                f"{document_label} record {position} misses {sorted(missing)}"
            )
        path_text = str(record["path"])
        if path_text in seen:
            raise ManifestMismatch(f"{document_label} has duplicate path: {path_text}")
        seen.add(path_text)
        normalized.append(record)
    return normalized


def _verify_identity_document(
    document_path: Path,
    *,
    record_key: str,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestMismatch(f"cannot read {document_path.name}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ManifestMismatch(f"{document_path.name} must be a JSON object")
    records = _normalize_identity_records(
        document.get(record_key), document_label=document_path.name
    )
    document_root = document.get("root")
    if document_root is not None and not isinstance(document_root, str):
        raise ManifestMismatch(f"{document_path.name} root must be a string")
    verified: list[dict[str, Any]] = []
    for record in records:
        path = _resolve_recorded_path(
            str(record["path"]),
            document_path=document_path,
            base_dir=base_dir,
            document_root=document_root,
        )
        try:
            actual = file_identity(path)
        except (OSError, ManifestMismatch) as exc:
            raise ManifestMismatch(f"{document_path.name}: {record['path']}: {exc}") from exc
        expected_size = record["size_bytes"]
        expected_sha = str(record["sha256"])
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or len(expected_sha) != 64
        ):
            raise ManifestMismatch(
                f"{document_path.name}: malformed identity for {record['path']}"
            )
        if actual["size_bytes"] != expected_size or actual["sha256"] != expected_sha:
            raise ManifestMismatch(
                f"{document_path.name}: identity mismatch for {record['path']}; "
                f"expected size={expected_size}, sha256={expected_sha}; "
                f"observed size={actual['size_bytes']}, sha256={actual['sha256']}"
            )
        verified.append(
            {
                "path": str(record["path"]),
                "size_bytes": actual["size_bytes"],
                "sha256": actual["sha256"],
            }
        )
    return {"status": "passed", "verified_count": len(verified), "records": verified}


def verify_hash_manifest(
    manifest_path: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify every file identity recorded under ``files``."""

    return _verify_identity_document(
        manifest_path, record_key="files", base_dir=base_dir
    )


def verify_research_input_lineage(
    lineage_path: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify every file identity recorded under ``inputs``."""

    return _verify_identity_document(
        lineage_path, record_key="inputs", base_dir=base_dir
    )


def verify_pinned_artifacts(project_root: Path) -> dict[str, Any]:
    """Machine-verify the twelve v7-pinned artifacts."""

    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for expected in PINNED_ARTIFACTS:
        path = project_root / str(expected["path"])
        try:
            actual = file_identity(path)
        except (OSError, ManifestMismatch) as exc:
            mismatches.append(f"{expected['path']}: {exc}")
            continue
        size_matches = actual["size_bytes"] == expected["size_bytes"]
        hash_matches = actual["sha256"] == expected["sha256"]
        record = {
            **dict(expected),
            "observed_size_bytes": actual["size_bytes"],
            "observed_sha256": actual["sha256"],
            "status": "passed" if size_matches and hash_matches else "failed",
        }
        records.append(record)
        if not size_matches or not hash_matches:
            mismatches.append(
                f"{expected['path']}: expected size={expected['size_bytes']}, "
                f"sha256={expected['sha256']}; observed size={actual['size_bytes']}, "
                f"sha256={actual['sha256']}"
            )
    if mismatches:
        raise ManifestMismatch("pinned artifact gate failed: " + " | ".join(mismatches))
    return {"status": "passed", "verified_count": len(records), "records": records}


def load_calibrated_events(project_root: Path) -> pd.DataFrame:
    """Load the seven pinned populations after exact per-file schema checks."""

    frames: list[pd.DataFrame] = []
    for record in PINNED_ARTIFACTS[: len(SYMBOLS)]:
        path = project_root / str(record["path"])
        frame = pd.read_csv(path, low_memory=False)
        actual_columns = tuple(frame.columns)
        if actual_columns != CALIBRATED_COLUMNS:
            raise S1AnchorError(
                f"{record['path']} schema mismatch: "
                f"expected={CALIBRATED_COLUMNS}, observed={actual_columns}"
            )
        frame["decision_ts"] = pd.to_datetime(
            frame["decision_ts"], utc=True, errors="raise"
        )
        frame["symbol"] = str(record["symbol"])
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True, sort=False)
    events.attrs["source_schemas_validated"] = True
    return events


def _strict_bool(series: pd.Series, *, label: str) -> pd.Series:
    if series.dtype == bool:
        return series
    non_null = set(series.dropna().unique().tolist())
    if not non_null.issubset({True, False, 0, 1}):
        raise S1AnchorError(f"{label} contains non-boolean values: {sorted(non_null)!r}")
    if series.isna().any():
        raise S1AnchorError(f"{label} contains null values")
    return series.astype(bool)


def verify_s1_anchors(
    events: pd.DataFrame,
    *,
    expected: Mapping[str, Any] = S1_EXPECTED,
) -> dict[str, Any]:
    """Reproduce every hard S1 anchor and stop on any mismatch."""

    expected_combined_columns = CALIBRATED_COLUMNS + ("symbol",)
    actual_combined_columns = tuple(events.columns)
    if actual_combined_columns != expected_combined_columns:
        raise S1AnchorError(
            "combined population schema mismatch: "
            f"expected={expected_combined_columns}, "
            f"observed={actual_combined_columns}"
        )

    failures: list[str] = []
    total_rows = int(len(events))
    if total_rows != int(expected["total_rows"]):
        failures.append(
            f"total_rows expected={expected['total_rows']} observed={total_rows}"
        )

    folds_numeric = pd.to_numeric(events["fold"], errors="coerce")
    if folds_numeric.isna().any():
        failures.append("fold contains non-numeric or null values")
    actual_fold_counts = {
        int(key): int(value)
        for key, value in folds_numeric.value_counts(dropna=False).sort_index().items()
        if not pd.isna(key)
    }
    expected_fold_counts = {
        int(key): int(value) for key, value in expected["fold_counts"].items()
    }
    if actual_fold_counts != expected_fold_counts:
        failures.append(
            f"fold_counts expected={expected_fold_counts} observed={actual_fold_counts}"
        )

    statuses = set(events["meta_eval_status"].dropna().unique().tolist())
    expected_statuses = {"evaluated", "not_evaluated_no_prior_fold"}
    if statuses != expected_statuses or events["meta_eval_status"].isna().any():
        failures.append(
            f"meta_eval_status expected={sorted(expected_statuses)} "
            f"observed={sorted(str(item) for item in statuses)}"
        )

    evaluated_mask = events["meta_eval_status"].eq("evaluated")
    expected_evaluated_mask = folds_numeric.between(2, 5, inclusive="both")
    evaluated_rows = int(evaluated_mask.sum())
    if evaluated_rows != int(expected["evaluated_rows"]):
        failures.append(
            f"evaluated_rows expected={expected['evaluated_rows']} "
            f"observed={evaluated_rows}"
        )
    if not evaluated_mask.equals(expected_evaluated_mask):
        disagreement = int((evaluated_mask != expected_evaluated_mask).sum())
        failures.append(
            "evaluated population is not exactly folds 2..5; "
            f"disagreement_rows={disagreement}"
        )

    evaluated = events.loc[evaluated_mask]
    meta_y = pd.to_numeric(evaluated["meta_y"], errors="coerce")
    uniqueness = pd.to_numeric(evaluated["tb_uniqueness"], errors="coerce")
    if meta_y.isna().any():
        failures.append(f"evaluated meta_y null_count={int(meta_y.isna().sum())}")
    if uniqueness.isna().any() or not np.isfinite(uniqueness.to_numpy()).all():
        failures.append("evaluated tb_uniqueness contains null or non-finite values")

    meta_y_positive = int(meta_y.sum()) if not meta_y.isna().any() else -1
    if meta_y_positive != int(expected["evaluated_meta_y_positive"]):
        failures.append(
            "evaluated_meta_y_positive "
            f"expected={expected['evaluated_meta_y_positive']} "
            f"observed={meta_y_positive}"
        )

    unweighted_rate = float(meta_y.mean())
    if round(unweighted_rate, 7) != float(expected["unweighted_base_rate_7dp"]):
        failures.append(
            "unweighted_base_rate_7dp "
            f"expected={expected['unweighted_base_rate_7dp']} "
            f"observed={round(unweighted_rate, 7)}"
        )

    if uniqueness.isna().any() or float(uniqueness.sum()) <= 0.0:
        weighted_rate = float("nan")
    else:
        weighted_rate = float(np.average(meta_y.to_numpy(), weights=uniqueness.to_numpy()))
    if not math.isfinite(weighted_rate) or round(weighted_rate, 7) != float(
        expected["weighted_base_rate_7dp"]
    ):
        failures.append(
            "weighted_base_rate_7dp "
            f"expected={expected['weighted_base_rate_7dp']} "
            f"observed={round(weighted_rate, 7) if math.isfinite(weighted_rate) else weighted_rate}"
        )

    evaluated_nulls = {
        column: int(evaluated[column].isna().sum())
        for column in ("p_meta", "p_primary_cal_platt")
    }
    for column, count in evaluated_nulls.items():
        if count:
            failures.append(f"evaluated {column} null_count={count}")

    if failures:
        raise S1AnchorError("S1 hard-anchor gate failed: " + " | ".join(failures))

    return {
        "status": "passed",
        "total_rows": total_rows,
        "fold_counts": actual_fold_counts,
        "evaluated_rows": evaluated_rows,
        "evaluated_folds_exact": True,
        "meta_eval_status_values": sorted(statuses),
        "evaluated_meta_y_positive": meta_y_positive,
        "unweighted_base_rate": unweighted_rate,
        "unweighted_base_rate_7dp": round(unweighted_rate, 7),
        "weighted_base_rate": weighted_rate,
        "weighted_base_rate_7dp": round(weighted_rate, 7),
        "evaluated_null_counts": evaluated_nulls,
        "source_schemas_validated": bool(
            events.attrs.get("source_schemas_validated", False)
        ),
        "column_order": list(CALIBRATED_COLUMNS),
    }


def verify_production_constants() -> dict[str, Any]:
    """Import and reconcile production-owned cost, embargo, and threshold constants."""

    import calibration
    import meta_model

    derived_cost, embargo = meta_model.derive_cost_and_embargo()
    calibration_cost = float(calibration.CAL_ECONOMIC_COST)
    if not math.isclose(derived_cost, calibration_cost, rel_tol=0.0, abs_tol=1e-15):
        raise IntegrityError(
            f"cost sources differ: meta_model={derived_cost!r}, "
            f"calibration={calibration_cost!r}"
        )
    if not math.isclose(derived_cost, 0.0030, rel_tol=0.0, abs_tol=1e-12):
        raise IntegrityError(f"cost contract requires 0.0030, observed {derived_cost!r}")
    if embargo != pd.Timedelta(hours=96):
        raise IntegrityError(f"embargo contract requires 96h, observed {embargo}")

    selection_grid = tuple(calibration.CAL_THRESHOLD_SELECTION_GRID)
    first_fold = float(calibration.CAL_THRESHOLD_FIRST_FOLD)
    abstain = float(calibration.CAL_THRESHOLD_ABSTAIN)
    minimum_trades = int(calibration.CAL_THRESHOLD_MIN_TRADES)
    numeric_grid = tuple(float(value) for value in selection_grid)
    if (
        not numeric_grid
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in numeric_grid)
        or any(right <= left for left, right in zip(numeric_grid, numeric_grid[1:]))
    ):
        raise IntegrityError(f"invalid imported selection grid: {selection_grid!r}")
    if (
        not math.isfinite(first_fold)
        or not 0.0 <= first_fold <= 1.0
        or not math.isfinite(abstain)
        or abstain <= numeric_grid[-1]
        or minimum_trades < 1
    ):
        raise IntegrityError(
            "invalid imported production threshold constants: "
            f"first_fold={first_fold}, abstain={abstain}, "
            f"minimum_trades={minimum_trades}"
        )
    return {
        "status": "passed",
        "economic_cost": float(derived_cost),
        "cost_sources_equal": True,
        "embargo_hours": int(embargo / pd.Timedelta(hours=1)),
        "selection_grid": list(numeric_grid),
        "first_fold_threshold": first_fold,
        "abstain_threshold": abstain,
        "minimum_trades": minimum_trades,
    }


def _selection_summary(net: pd.Series, mask: pd.Series) -> dict[str, Any]:
    selected = net.loc[mask]
    return {"n": int(mask.sum()), "sum_net": float(selected.sum())}


def calculate_s2(events: pd.DataFrame, *, economic_cost: float) -> dict[str, Any]:
    """Reproduce the contract's soft historical anchors from pinned CSV rows."""

    evaluated = events.loc[events["meta_eval_status"].eq("evaluated")].copy()
    evaluated["decision_ts"] = pd.to_datetime(
        evaluated["decision_ts"], utc=True, errors="raise"
    )
    meta_mask = _strict_bool(evaluated["traded_meta"], label="traded_meta")
    primary_mask = _strict_bool(
        evaluated["traded_baseline"], label="traded_baseline"
    )
    net = pd.to_numeric(evaluated["tb_return"], errors="raise") - economic_cost

    common = meta_mask & primary_mask
    meta_only = meta_mask & ~primary_mask
    primary_only = ~meta_mask & primary_mask
    overlap = {
        "common": _selection_summary(net, common),
        "meta_only": _selection_summary(net, meta_only),
        "primary_only": _selection_summary(net, primary_only),
    }

    equal_n_folds: list[dict[str, Any]] = []
    equal_n_total = 0
    equal_n_meta_sum = 0.0
    equal_n_primary_sum = 0.0
    for fold in (2, 3, 4, 5):
        current = evaluated.loc[evaluated["fold"].eq(fold)].copy()
        current["net"] = (
            pd.to_numeric(current["tb_return"], errors="raise") - economic_cost
        )
        n_selected = int(
            _strict_bool(current["traded_meta"], label=f"fold{fold}.traded_meta").sum()
        )
        fold_sums: dict[str, float] = {}
        for score in ("p_meta", "p_primary"):
            ranked = current.sort_values(
                [score, "decision_ts", "symbol"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            fold_sums[score] = float(ranked.head(n_selected)["net"].sum())
        equal_n_folds.append(
            {
                "fold": fold,
                "n": n_selected,
                "meta_sum_net": fold_sums["p_meta"],
                "primary_sum_net": fold_sums["p_primary"],
            }
        )
        equal_n_total += n_selected
        equal_n_meta_sum += fold_sums["p_meta"]
        equal_n_primary_sum += fold_sums["p_primary"]

    target = pd.to_numeric(evaluated["meta_y"], errors="raise")
    fold5 = evaluated.loc[evaluated["fold"].eq(5)].copy()
    fold5_target = pd.to_numeric(fold5["meta_y"], errors="raise")
    auc = {
        "pooled_meta": float(roc_auc_score(target, evaluated["p_meta"])),
        "pooled_primary": float(roc_auc_score(target, evaluated["p_primary"])),
        "fold5_meta": float(roc_auc_score(fold5_target, fold5["p_meta"])),
        "fold5_primary": float(roc_auc_score(fold5_target, fold5["p_primary"])),
    }

    fold5_net = pd.to_numeric(fold5["tb_return"], errors="raise") - economic_cost
    fold5_meta_mask = _strict_bool(fold5["traded_meta"], label="fold5.traded_meta")
    fold5_primary_mask = _strict_bool(
        fold5["traded_baseline"], label="fold5.traded_baseline"
    )
    primary_tau_values = sorted(
        float(value) for value in fold5["tau_baseline_used"].dropna().unique()
    )
    meta_tau_values = sorted(
        float(value) for value in fold5["tau_meta_used"].dropna().unique()
    )
    fold5_primary = _selection_summary(fold5_net, fold5_primary_mask)
    fold5_meta = _selection_summary(fold5_net, fold5_meta_mask)
    primary_sum = float(fold5_primary["sum_net"])
    meta_sum = float(fold5_meta["sum_net"])
    loss_reduction = (
        float((1.0 - abs(meta_sum) / abs(primary_sum)) * 100.0)
        if primary_sum != 0.0
        else float("nan")
    )
    return {
        "overlap": overlap,
        "equal_n": {
            "folds": equal_n_folds,
            "n": equal_n_total,
            "meta_sum_net": equal_n_meta_sum,
            "primary_sum_net": equal_n_primary_sum,
        },
        "auc": auc,
        "fold5": {
            "primary_n": fold5_primary["n"],
            "primary_sum_net": primary_sum,
            "primary_tau_values": primary_tau_values,
            "meta_n": fold5_meta["n"],
            "meta_sum_net": meta_sum,
            "meta_tau_values": meta_tau_values,
            "loss_reduction_percent": loss_reduction,
        },
    }


def reconcile_s2(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Apply zero/count, ±0.001/sum, and ±0.0002/AUC tolerances."""

    checks: list[dict[str, Any]] = []

    def check(label: str, actual: float | int, expected: float | int, tolerance: float) -> None:
        passed = abs(float(actual) - float(expected)) <= tolerance
        checks.append(
            {
                "label": label,
                "observed": actual,
                "expected": expected,
                "tolerance": tolerance,
                "status": "passed" if passed else "failed",
                "absolute_difference": abs(float(actual) - float(expected)),
            }
        )

    for region in ("common", "meta_only", "primary_only"):
        check(
            f"overlap.{region}.n",
            observed["overlap"][region]["n"],
            S2_EXPECTED["overlap"][region]["n"],
            0.0,
        )
        check(
            f"overlap.{region}.sum_net",
            observed["overlap"][region]["sum_net"],
            S2_EXPECTED["overlap"][region]["sum_net"],
            0.001,
        )
    check("equal_n.n", observed["equal_n"]["n"], S2_EXPECTED["equal_n"]["n"], 0.0)
    check(
        "equal_n.meta_sum_net",
        observed["equal_n"]["meta_sum_net"],
        S2_EXPECTED["equal_n"]["meta_sum_net"],
        0.001,
    )
    check(
        "equal_n.primary_sum_net",
        observed["equal_n"]["primary_sum_net"],
        S2_EXPECTED["equal_n"]["primary_sum_net"],
        0.001,
    )
    for label in ("pooled_meta", "pooled_primary", "fold5_meta", "fold5_primary"):
        check(
            f"auc.{label}",
            observed["auc"][label],
            S2_EXPECTED["auc"][label],
            0.0002,
        )
    for label in ("primary_n", "meta_n"):
        check(
            f"fold5.{label}",
            observed["fold5"][label],
            S2_EXPECTED["fold5"][label],
            0.0,
        )
    for label in ("primary_sum_net", "meta_sum_net"):
        check(
            f"fold5.{label}",
            observed["fold5"][label],
            S2_EXPECTED["fold5"][label],
            0.001,
        )
    primary_tau_values = observed["fold5"]["primary_tau_values"]
    tau_passed = primary_tau_values == [S2_EXPECTED["fold5"]["primary_tau"]]
    checks.append(
        {
            "label": "fold5.primary_tau_values",
            "observed": primary_tau_values,
            "expected": [S2_EXPECTED["fold5"]["primary_tau"]],
            "tolerance": 0.0,
            "status": "passed" if tau_passed else "failed",
            "absolute_difference": 0.0 if tau_passed else None,
        }
    )
    failed = [item for item in checks if item["status"] == "failed"]
    return {
        "status": "reconciled" if not failed else "unreconciled",
        "decision_rule_suspended": bool(failed),
        "checks": checks,
        "cause_analysis": (
            {"status": "not_required", "failed_checks": []}
            if not failed
            else {
                "status": "requires_owner_adjudication",
                "failed_checks": failed,
                "note": (
                    "Observed-minus-expected differences are recorded per check; "
                    "the decision rule remains suspended until owner adjudication."
                ),
            }
        ),
    }


def _is_allowed_relative(relative_path: str) -> bool:
    normalized = relative_path.strip("/")
    return any(
        normalized == allowed or normalized.startswith(f"{allowed}/")
        for allowed in ALLOWED_PATHS
    )


def _filesystem_record(path: Path, relative_path: str) -> dict[str, Any]:
    before = path.lstat()
    common = {
        "path": relative_path,
        "mode": stat.S_IMODE(before.st_mode),
        "mtime_ns": int(before.st_mtime_ns),
    }
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(path)
        encoded = os.fsencode(target)
        return {
            **common,
            "type": "symlink",
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "link_target": target,
        }
    if stat.S_ISREG(before.st_mode):
        digest = sha256_file(path)
        after = path.lstat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_mode != after.st_mode
        ):
            raise ForbiddenMutationError(f"file changed while fingerprinting: {path}")
        return {
            **common,
            "type": "regular",
            "size_bytes": int(before.st_size),
            "sha256": digest,
        }
    return {
        **common,
        "type": "special",
        "size_bytes": int(before.st_size),
        "sha256": None,
    }


def capture_forbidden_files_fingerprint(project_root: Path) -> dict[str, Any]:
    """Fingerprint forbidden files except the owner-exempted exact .DS_Store basename."""

    root = project_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    ignored_current_record_count = 0
    for current_text, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_text)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            child = current / name
            relative = child.relative_to(root).as_posix()
            if _is_allowed_relative(relative):
                continue
            if child.is_symlink():
                records.append(_filesystem_record(child, relative))
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            child = current / name
            relative = child.relative_to(root).as_posix()
            if _is_allowed_relative(relative):
                continue
            if name == FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME:
                ignored_current_record_count += 1
                continue
            records.append(_filesystem_record(child, relative))
    records.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "root": str(root),
        "excluded_allowed_paths": list(ALLOWED_PATHS),
        "ignored_exact_file_basename": FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME,
        "ignored_current_record_count": ignored_current_record_count,
        "record_count": len(records),
        "records": records,
    }


def verify_forbidden_files_fingerprint(
    fingerprint_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Re-scan and require exact byte and metadata equality for forbidden files."""

    try:
        frozen = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForbiddenMutationError(
            f"cannot read forbidden-files fingerprint: {exc}"
        ) from exc
    observed = capture_forbidden_files_fingerprint(project_root)
    frozen_records = frozen.get("records", [])
    if not isinstance(frozen_records, list):
        raise ForbiddenMutationError("frozen forbidden-files records must be a list")
    exempted_frozen_records = [
        record
        for record in frozen_records
        if Path(str(record.get("path", ""))).name
        == FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME
    ]
    certified_frozen_records = [
        record
        for record in frozen_records
        if Path(str(record.get("path", ""))).name
        != FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME
    ]
    fixed_fields_match = (
        frozen.get("schema_version") == observed.get("schema_version")
        and frozen.get("root") == observed.get("root")
        and frozen.get("excluded_allowed_paths")
        == observed.get("excluded_allowed_paths")
    )
    records_match = certified_frozen_records == observed.get("records")
    if fixed_fields_match and records_match:
        total_bytes = sum(
            int(record["size_bytes"])
            for record in certified_frozen_records
            if record["type"] == "regular"
        )
        return {
            "status": "passed",
            "ignored_rule": {
                "kind": "exact_file_basename",
                "value": FORBIDDEN_FINGERPRINT_IGNORED_EXACT_BASENAME,
                "glob_or_other_dotfiles_excluded": False,
            },
            "excluded_frozen_record_count": len(exempted_frozen_records),
            "excluded_current_record_count": int(
                observed.get("ignored_current_record_count", 0)
            ),
            "certified_record_count": len(certified_frozen_records),
            "regular_file_bytes": total_bytes,
        }

    frozen_by_path = {item["path"]: item for item in certified_frozen_records}
    observed_by_path = {item["path"]: item for item in observed.get("records", [])}
    added = sorted(set(observed_by_path) - set(frozen_by_path))
    removed = sorted(set(frozen_by_path) - set(observed_by_path))
    changed = sorted(
        path
        for path in set(frozen_by_path) & set(observed_by_path)
        if frozen_by_path[path] != observed_by_path[path]
    )
    detail = {
        "added": added[:50],
        "removed": removed[:50],
        "changed": changed[:50],
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
    }
    raise ForbiddenMutationError(
        "forbidden-files fingerprint mismatch: "
        + json.dumps(detail, ensure_ascii=False, sort_keys=True)
    )
