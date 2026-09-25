# ==============================================================================
# calibration.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Offline v49 score calibration, diagnostics, and transactional publication.

This module is intentionally an island: it neither imports CatBoost nor the v48
``meta_model`` module.  The CatBoost file is an opaque, hashed lineage input.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
CANONICAL_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
)
CANONICAL_META_OOF_DIR = ROOT / "data" / "models"
CANONICAL_OUTPUT_DIR = ROOT / "data" / "models"

OOF_COLUMNS = (
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
)
CALIBRATION_COLUMNS = (
    "p_meta_cal_platt",
    "p_meta_cal_iso",
    "p_primary_cal_platt",
    "p_primary_cal_iso",
    "cal_is_raw",
)
CALIBRATED_OOF_COLUMNS = OOF_COLUMNS + CALIBRATION_COLUMNS

CAL_MIN_TRAIN_N = 300
CAL_CLIP_EPS = 1e-6
CAL_LOGLOSS_EPS = 1e-15
CAL_IMPROVEMENT_MIN = 0.001
CAL_METHOD_TIE_TOLERANCE = 0.001
CAL_PLATT_C = 1e6
CAL_PLATT_SOLVER = "lbfgs"
CAL_PLATT_MAX_ITER = 1000

# mirror of config, frozen for this analysis
CAL_ECONOMIC_COST = 0.0030
# mirror of config, frozen for this analysis
CAL_THRESHOLD_SELECTION_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
# mirror of config, frozen for this analysis
CAL_THRESHOLD_DIAGNOSTIC_GRID = (0.80, 0.85, 0.90)
# mirror of config, frozen for this analysis
CAL_THRESHOLD_ABSTAIN = 1.01
CAL_THRESHOLD_FIRST_FOLD = 0.55
CAL_THRESHOLD_MIN_TRADES = 30
CAL_THRESHOLD_TIE_TOL = 1e-12

SCORE_DOMAINS: dict[str, tuple[float, float]] = {
    "p_meta": (0.0, 1.0),
    "p_primary": (0.55, 1.0),
}
SCORE_SHORT_NAMES = {"p_meta": "meta", "p_primary": "primary"}

CANONICAL_TOTAL_ROWS = 11029
CANONICAL_FOLD_COUNTS = {1: 1981, 2: 1745, 3: 2119, 4: 2197, 5: 2987}
CANONICAL_AUC_ANCHORS = {
    "aggregate": {"meta": 0.5138696, "primary": 0.5262030},
    "fold_5": {"meta": 0.5011941, "primary": 0.4926582},
}
CANONICAL_EQUAL_N_ANCHORS = {
    2: {"n": 537, "meta": 1.6071, "primary": 2.2083},
    3: {"n": 468, "meta": 1.3479, "primary": 2.7687},
    4: {"n": 304, "meta": -0.7655, "primary": 2.1513},
    5: {"n": 193, "meta": -0.8148, "primary": -1.1010},
    "aggregate": {"n": 1502, "meta": 1.3747, "primary": 6.0273},
}
CANONICAL_RAW_THRESHOLD_ANCHORS = {
    2: {"tau": 0.55, "n": 537, "sum_net": 1.6071},
    3: {"tau": 0.60, "n": 468, "sum_net": 1.3479},
    4: {"tau": 0.60, "n": 304, "sum_net": -0.7655},
    5: {"tau": 0.65, "n": 193, "sum_net": -0.8148},
}


class CalibrationError(RuntimeError):
    """A structured calibration-contract violation."""

    def __init__(self, message: str, *, code: str = "calibration_error", details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


class PreflightError(CalibrationError):
    """An input, lineage, schema, or canonical-anchor violation."""

    def __init__(self, message: str, *, details: Any = None):
        super().__init__(
            message,
            code="invalid_anchor_or_input",
            details=details,
        )


class AnchorError(PreflightError):
    """A canonical economic or statistical anchor violation."""


class TransactionError(CalibrationError):
    """A failed staging, commit, or rollback operation."""

    def __init__(self, message: str, *, details: Any = None):
        super().__init__(message, code="transaction_error", details=details)


@dataclass
class ScoreCalibrator:
    """Safe persisted wrapper whose public input is always a raw score."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    clip_eps: float = CAL_CLIP_EPS
    input_domain: tuple[float, float] = (0.0, 1.0)
    estimator: Any | None = None

    def __post_init__(self) -> None:
        if self.method not in {"platt", "isotonic", "identity"}:
            raise ValueError(f"unknown calibration method: {self.method!r}")
        if len(self.input_domain) != 2 or self.input_domain[0] > self.input_domain[1]:
            raise ValueError(f"invalid input_domain: {self.input_domain!r}")
        if self.method == "identity" and self.estimator is not None:
            raise ValueError("identity calibrator must not carry an estimator")
        if self.method != "identity" and self.estimator is None:
            raise ValueError(f"{self.method} calibrator requires a fitted estimator")

    def predict(self, raw_scores: Any) -> np.ndarray:
        """Validate raw scores, apply the fitted mapping, and preserve shape."""

        values = np.asarray(raw_scores, dtype=float)
        original_shape = values.shape
        flat = values.reshape(-1)
        if not np.all(np.isfinite(flat)):
            raise ValueError("raw_scores must be finite")
        lower, upper = map(float, self.input_domain)
        if np.any(flat < lower) or np.any(flat > upper):
            observed = (
                float(np.min(flat)) if flat.size else None,
                float(np.max(flat)) if flat.size else None,
            )
            raise ValueError(
                f"raw_scores outside input domain [{lower}, {upper}]: observed={observed}"
            )
        if self.method == "identity":
            predicted = flat.copy()
        elif self.method == "platt":
            clipped = np.clip(flat, self.clip_eps, 1.0 - self.clip_eps)
            logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            predicted = np.asarray(self.estimator.predict_proba(logit)[:, 1], dtype=float)
        else:
            predicted = np.asarray(self.estimator.predict(flat), dtype=float)
        if predicted.size != flat.size:
            raise CalibrationError(
                "calibrator prediction did not preserve element count",
                details={"input_size": int(flat.size), "output_size": int(predicted.size)},
            )
        return predicted.reshape(original_shape)


def predict_calibrated(calibrator: ScoreCalibrator, raw_scores: Any) -> np.ndarray:
    """Stable public prediction entry point for persisted wrappers."""

    if not isinstance(calibrator, ScoreCalibrator):
        raise TypeError("calibrator must be a ScoreCalibrator")
    return calibrator.predict(raw_scores)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _parse_utc(series: pd.Series, column: str, path: Path) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="raise", utc=False)
    except Exception as exc:
        raise PreflightError(f"{path.name}: invalid {column}: {exc}") from exc
    timestamps = [pd.Timestamp(item) for item in parsed]
    if any(item.tzinfo is None or item.utcoffset() is None for item in timestamps):
        raise PreflightError(f"{path.name}: {column} must be timezone-aware UTC")
    if any(item.utcoffset().total_seconds() != 0 for item in timestamps):
        raise PreflightError(f"{path.name}: {column} must use UTC offset")
    return pd.to_datetime(series, errors="raise", utc=True)


def _finite_numeric(frame: pd.DataFrame, column: str, path: Path) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise PreflightError(f"{path.name}: {column} must be numeric: {exc}") from exc
    if not np.all(np.isfinite(values)):
        raise PreflightError(f"{path.name}: {column} must be finite")
    return values


def load_oof_file(
    path: str | Path,
    expected_symbol: str,
    *,
    is_real_data_run: bool = False,
) -> pd.DataFrame:
    """Load one v48 OOF CSV and enforce the exact seventeen-column contract."""

    del is_real_data_run  # The loader contract is equally strict in both modes.
    csv_path = Path(path)
    if not csv_path.is_file():
        raise PreflightError(f"missing OOF input: {csv_path}")
    expected_name = f"meta_oof_{expected_symbol}_4h.csv"
    if csv_path.name != expected_name:
        raise PreflightError(
            f"OOF filename/symbol mismatch: expected {expected_name!r}, got {csv_path.name!r}"
        )
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise PreflightError(f"failed to parse {csv_path}: {exc}") from exc
    if tuple(frame.columns) != OOF_COLUMNS:
        raise PreflightError(
            f"{csv_path.name}: exact OOF schema mismatch",
            details={"expected": list(OOF_COLUMNS), "observed": list(frame.columns)},
        )
    frame = frame.copy()
    frame["OpenTime"] = _parse_utc(frame["OpenTime"], "OpenTime", csv_path)
    frame["decision_ts"] = _parse_utc(frame["decision_ts"], "decision_ts", csv_path)
    frame["tb_exit_index"] = _parse_utc(
        frame["tb_exit_index"], "tb_exit_index", csv_path
    )
    if not (frame["decision_ts"] == frame["OpenTime"] + pd.Timedelta(hours=4)).all():
        bad = frame.index[
            frame["decision_ts"] != frame["OpenTime"] + pd.Timedelta(hours=4)
        ].tolist()[:10]
        raise PreflightError(
            f"{csv_path.name}: decision_ts must equal OpenTime + 4h; bad rows={bad}"
        )

    frame["symbol"] = str(expected_symbol)
    duplicate = frame.duplicated(["symbol", "OpenTime"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["symbol", "OpenTime"]].head(10).to_dict("records")
        raise PreflightError(
            f"{csv_path.name}: duplicate (symbol, OpenTime)", details=examples
        )

    for target in ("y", "meta_y"):
        values = pd.to_numeric(frame[target], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise PreflightError(f"{csv_path.name}: {target} must contain only 0 and 1")
        frame[target] = values.astype(int)

    folds = pd.to_numeric(frame["fold"], errors="coerce")
    if folds.isna().any() or not np.equal(folds, np.floor(folds)).all():
        raise PreflightError(f"{csv_path.name}: fold must contain integers")
    frame["fold"] = folds.astype(int)
    if not frame["fold"].isin([1, 2, 3, 4, 5]).all():
        raise PreflightError(f"{csv_path.name}: folds must be in 1..5")

    weights = _finite_numeric(frame, "tb_uniqueness", csv_path)
    if np.any(weights <= 0.0):
        raise PreflightError(f"{csv_path.name}: tb_uniqueness must be strictly positive")
    primary = _finite_numeric(frame, "p_primary", csv_path)
    if np.any(primary < 0.55) or np.any(primary > 1.0):
        raise PreflightError(f"{csv_path.name}: p_primary outside [0.55, 1.0]")

    try:
        p_meta = pd.to_numeric(frame["p_meta"], errors="raise")
    except Exception as exc:
        raise PreflightError(f"{csv_path.name}: p_meta must be numeric or empty: {exc}") from exc
    fold_one = frame["fold"].eq(1)
    evaluated = frame["meta_eval_status"].eq("evaluated")
    expected_status = np.where(fold_one, "not_evaluated_no_prior_fold", "evaluated")
    if not np.array_equal(frame["meta_eval_status"].to_numpy(dtype=str), expected_status):
        raise PreflightError(f"{csv_path.name}: fold/meta_eval_status inconsistency")
    if not p_meta.loc[fold_one].isna().all():
        raise PreflightError(f"{csv_path.name}: p_meta must be NaN on every fold=1 row")
    if p_meta.loc[~fold_one].isna().any():
        raise PreflightError(f"{csv_path.name}: p_meta may be NaN only on fold=1")
    evaluated_meta = p_meta.loc[evaluated].to_numpy(dtype=float)
    if not np.all(np.isfinite(evaluated_meta)):
        raise PreflightError(f"{csv_path.name}: evaluated p_meta must be finite")
    if np.any(evaluated_meta < 0.0) or np.any(evaluated_meta > 1.0):
        raise PreflightError(f"{csv_path.name}: evaluated p_meta outside [0, 1]")
    frame["p_meta"] = p_meta
    _finite_numeric(frame, "tb_return", csv_path)
    for tau_column in (
        "tau_meta_used",
        "tau_baseline_used",
        "tau_meta_forced_used",
        "tau_baseline_forced_used",
    ):
        try:
            tau = pd.to_numeric(frame[tau_column], errors="raise")
        except Exception as exc:
            raise PreflightError(
                f"{csv_path.name}: {tau_column} must be numeric or empty: {exc}"
            ) from exc
        if not tau.loc[fold_one].isna().all():
            raise PreflightError(f"{csv_path.name}: {tau_column} must be NaN on fold=1")
        if tau.loc[evaluated].isna().any() or not np.all(
            np.isfinite(tau.loc[evaluated].to_numpy(dtype=float))
        ):
            raise PreflightError(
                f"{csv_path.name}: evaluated {tau_column} must be finite"
            )
        frame[tau_column] = tau
    for traded_column in ("traded_meta", "traded_baseline"):
        values = frame[traded_column]
        if pd.api.types.is_bool_dtype(values):
            parsed_bool = values.astype(bool)
        else:
            normalized = values.astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false"]).all():
                raise PreflightError(
                    f"{csv_path.name}: {traded_column} must contain only booleans"
                )
            parsed_bool = normalized.eq("true")
        frame[traded_column] = parsed_bool
        if parsed_bool.loc[fold_one].any():
            raise PreflightError(
                f"{csv_path.name}: {traded_column} must be false on non-evaluated fold=1"
            )
    return frame


def _artifact_record(summary: Mapping[str, Any], filename: str) -> Mapping[str, Any] | None:
    artifacts = summary.get("output_artifacts")
    if isinstance(artifacts, Mapping):
        record = artifacts.get(filename)
        return record if isinstance(record, Mapping) else None
    if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        for item in artifacts:
            if isinstance(item, Mapping) and Path(str(item.get("path", ""))).name == filename:
                return item
    return None


def _claimed_fold_counts(summary: Mapping[str, Any]) -> dict[int, int] | None:
    direct = summary.get("fired_event_count_by_fold")
    if isinstance(direct, Mapping):
        try:
            return {int(key): int(value) for key, value in direct.items()}
        except (TypeError, ValueError):
            return None
    coverage = summary.get("coverage")
    try:
        by_fold = coverage["dataset"]["by_fold"]
        return {int(key): int(value["event_count"]) for key, value in by_fold.items()}
    except (KeyError, TypeError, ValueError):
        return None


def _claimed_total_rows(summary: Mapping[str, Any]) -> int | None:
    counts = _claimed_fold_counts(summary)
    if counts is not None:
        return int(sum(counts.values()))
    for key in ("row_count", "n_rows", "total_rows"):
        if key in summary:
            try:
                return int(summary[key])
            except (TypeError, ValueError):
                return None
    try:
        return int(summary["coverage"]["dataset"]["overall"]["event_count"])
    except (KeyError, TypeError, ValueError):
        return None


def preflight_inputs(
    meta_oof_dir: str | Path,
    symbols: Sequence[str],
    *,
    timeframe: str = "4h",
    is_real_data_run: bool,
) -> dict[str, Any]:
    """Validate nine input files, lineage, schema, and row-count claims."""

    directory = Path(meta_oof_dir).resolve()
    symbol_tuple = tuple(str(item) for item in symbols)
    if timeframe != "4h":
        raise PreflightError(f"timeframe must be '4h', got {timeframe!r}")
    summary_path = directory / "meta_training_summary_4h.json"
    model_path = directory / "meta_4h.cbm"
    oof_paths = {symbol: directory / f"meta_oof_{symbol}_4h.csv" for symbol in symbol_tuple}
    required = [summary_path, model_path, *oof_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PreflightError("missing required calibration inputs", details={"missing": missing})
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"failed to parse {summary_path}: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise PreflightError("meta training summary must be a JSON object")

    if "timeframe" in summary and summary.get("timeframe") != timeframe:
        raise PreflightError("summary timeframe disagrees with requested timeframe")
    if "symbols" in summary and tuple(summary.get("symbols", ())) != symbol_tuple:
        raise PreflightError("summary symbols disagree with requested symbols/order")

    if is_real_data_run:
        checks = {
            "is_real_data_run": summary.get("is_real_data_run") is True,
            "is_canonical_output": summary.get("is_canonical_output") is True,
            "deployment_ready_false": summary.get("deployment", {}).get("deployment_ready") is False,
            "timeframe": summary.get("timeframe") == "4h",
            "symbols": tuple(summary.get("symbols", ())) == tuple(CANONICAL_SYMBOLS),
            "requested_symbols": symbol_tuple == tuple(CANONICAL_SYMBOLS),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise PreflightError(
                f"canonical summary identity mismatch: {failed}", details=checks
            )
        matched_n = _extract_matched_n(summary)
        matched_failures: list[dict[str, Any]] = []
        if matched_n is None:
            matched_failures.append({"field": "matched_n", "observed": None})
        else:
            if matched_n["meta_n"] != 3434 or matched_n["primary_n"] != 3434:
                matched_failures.append(
                    {
                        "field": "matched_n.n",
                        "expected": 3434,
                        "observed": [matched_n["meta_n"], matched_n["primary_n"]],
                    }
                )
            if abs(matched_n["meta_sum_net"] - (-5.3508)) > 5e-4:
                matched_failures.append(
                    {
                        "field": "matched_n.meta_sum_net",
                        "expected": -5.3508,
                        "observed": matched_n["meta_sum_net"],
                    }
                )
            if abs(matched_n["primary_sum_net"] - (-1.8881)) > 5e-4:
                matched_failures.append(
                    {
                        "field": "matched_n.primary_sum_net",
                        "expected": -1.8881,
                        "observed": matched_n["primary_sum_net"],
                    }
                )
        if matched_failures:
            raise PreflightError(
                "canonical matched-N summary anchor mismatch", details=matched_failures
            )

    lineage: dict[str, dict[str, Any]] = {}
    for path in [*oof_paths.values(), model_path]:
        actual = _file_identity(path)
        lineage[path.name] = actual
        claimed = _artifact_record(summary, path.name)
        if claimed is None:
            raise PreflightError(f"summary output_artifacts missing {path.name}")
        problems: list[str] = []
        if str(claimed.get("sha256")) != actual["sha256"]:
            problems.append("sha256/hash")
        try:
            if int(claimed.get("size_bytes")) != actual["size_bytes"]:
                problems.append("size_bytes")
        except (TypeError, ValueError):
            problems.append("size_bytes")
        if problems:
            raise PreflightError(
                f"lineage mismatch for {path.name}: {', '.join(problems)}",
                details={"claimed": dict(claimed), "actual": actual},
            )

    frames = {
        symbol: load_oof_file(path, symbol, is_real_data_run=is_real_data_run)
        for symbol, path in oof_paths.items()
    }
    events = pd.concat(frames.values(), ignore_index=True)
    duplicate = events.duplicated(["symbol", "OpenTime"], keep=False)
    if duplicate.any():
        raise PreflightError("duplicate (symbol, OpenTime) across OOF inputs")
    observed_fold_counts = {
        int(key): int(value) for key, value in events["fold"].value_counts().sort_index().items()
    }
    claimed_fold_counts = _claimed_fold_counts(summary)
    if claimed_fold_counts is not None and observed_fold_counts != claimed_fold_counts:
        raise PreflightError(
            "OOF fold row counts disagree with summary",
            details={"claimed": claimed_fold_counts, "observed": observed_fold_counts},
        )
    claimed_total = _claimed_total_rows(summary)
    if claimed_total is not None and len(events) != claimed_total:
        raise PreflightError(
            "OOF total row count disagrees with summary",
            details={"claimed": claimed_total, "observed": int(len(events))},
        )
    per_symbol_claims = summary.get("fired_event_counts")
    if isinstance(per_symbol_claims, Mapping):
        for symbol in symbol_tuple:
            claim = per_symbol_claims.get(symbol)
            if not isinstance(claim, Mapping):
                raise PreflightError(f"summary fired_event_counts missing {symbol}")
            observed_frame = frames[symbol]
            try:
                claimed_symbol_total = int(claim["total"])
                claimed_symbol_folds = {
                    int(key): int(value) for key, value in claim["by_fold"].items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise PreflightError(
                    f"summary fired_event_counts malformed for {symbol}"
                ) from exc
            observed_symbol_folds = {
                int(key): int(value)
                for key, value in observed_frame["fold"].value_counts().sort_index().items()
            }
            if claimed_symbol_total != len(observed_frame) or claimed_symbol_folds != observed_symbol_folds:
                raise PreflightError(
                    f"summary per-symbol row claims disagree for {symbol}",
                    details={
                        "claimed_total": claimed_symbol_total,
                        "observed_total": int(len(observed_frame)),
                        "claimed_folds": claimed_symbol_folds,
                        "observed_folds": observed_symbol_folds,
                    },
                )
    if is_real_data_run:
        if len(events) != CANONICAL_TOTAL_ROWS or observed_fold_counts != CANONICAL_FOLD_COUNTS:
            raise PreflightError(
                "canonical OOF row-count anchor mismatch",
                details={
                    "expected_total": CANONICAL_TOTAL_ROWS,
                    "observed_total": int(len(events)),
                    "expected_folds": CANONICAL_FOLD_COUNTS,
                    "observed_folds": observed_fold_counts,
                },
            )
    lineage[summary_path.name] = _file_identity(summary_path)
    return {
        "summary": dict(summary),
        "events": events,
        "frames": frames,
        "input_paths": {
            "summary": summary_path,
            "model": model_path,
            "oof": oof_paths,
        },
        "lineage": lineage,
        "fold_counts": observed_fold_counts,
        "total_rows": int(len(events)),
        "is_real_data_run": bool(is_real_data_run),
        "checks": {
            "nine_inputs_present": len(required) == len(symbol_tuple) + 2,
            "lineage_verified": True,
            "loader_validated": True,
            "row_counts_verified": True,
        },
    }


def _fit_guard_reason(
    scores: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> str | None:
    if scores.ndim != 1 or targets.ndim != 1 or weights.ndim != 1:
        return "scores, targets, and sample_weight must be one-dimensional"
    if not (len(scores) == len(targets) == len(weights)) or len(scores) == 0:
        return "scores, targets, and sample_weight must have equal nonzero length"
    if not np.all(np.isfinite(scores)):
        return "scores must be finite"
    if not np.all(np.isfinite(weights)):
        return "sample_weight must be finite"
    if np.any(weights <= 0.0):
        return "sample_weight must be strictly positive"
    if not np.all(np.isfinite(targets)):
        return "target must be finite"
    classes = np.unique(targets)
    if len(classes) != 2 or not np.array_equal(classes, np.array([0.0, 1.0])):
        return "both target classes 0 and 1 must be present"
    if np.unique(scores).size < 2:
        return "at least two distinct scores are required; score is constant"
    return None


def _identity_fit_metadata(
    *,
    requested_method: str,
    n_train: int,
    fallback: bool,
    fallback_reason: str | None,
    coefficient: float | None = None,
    intercept: float | None = None,
) -> dict[str, Any]:
    return {
        "requested_method": requested_method,
        "effective_method": "identity",
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
        "coefficient": coefficient,
        "intercept": intercept,
        "n_train": int(n_train),
        "params": {},
    }


def fit_score_calibrator(
    raw_scores: Any,
    targets: Any,
    sample_weight: Any | None = None,
    *,
    method: str,
    input_domain: tuple[float, float],
    is_real_data_run: bool,
    allow_fallback: bool | None = None,
) -> tuple[ScoreCalibrator, dict[str, Any]]:
    """Fit one weighted method, returning the safe wrapper and fit metadata."""

    requested_method = str(method).lower()
    if requested_method == "iso":
        requested_method = "isotonic"
    if requested_method not in {"platt", "isotonic", "identity"}:
        raise ValueError(f"unknown calibration method: {method!r}")
    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    y = np.asarray(targets, dtype=float).reshape(-1)
    weights = (
        np.ones(len(scores), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float).reshape(-1)
    )
    fallback_allowed = (not is_real_data_run) if allow_fallback is None else bool(allow_fallback)
    domain = (float(input_domain[0]), float(input_domain[1]))

    guard_reason = _fit_guard_reason(scores, y, weights)
    if requested_method == "identity":
        if guard_reason is not None:
            raise CalibrationError(
                f"identity fit guard failed: {guard_reason}", code="fit_guard_failed"
            )
        if np.any(scores < domain[0]) or np.any(scores > domain[1]):
            raise CalibrationError("identity scores outside input domain")
        wrapper = ScoreCalibrator(
            method="identity",
            params={},
            clip_eps=CAL_CLIP_EPS,
            input_domain=domain,
            estimator=None,
        )
        return wrapper, _identity_fit_metadata(
            requested_method="identity",
            n_train=len(scores),
            fallback=False,
            fallback_reason=None,
        )

    if guard_reason is not None:
        if is_real_data_run or not fallback_allowed:
            raise CalibrationError(
                f"{requested_method} fit guard failed: {guard_reason}",
                code="fit_guard_failed",
            )
        wrapper = ScoreCalibrator(
            method="identity",
            params={"requested_method": requested_method, "fallback_reason": guard_reason},
            clip_eps=CAL_CLIP_EPS,
            input_domain=domain,
            estimator=None,
        )
        return wrapper, _identity_fit_metadata(
            requested_method=requested_method,
            n_train=len(scores),
            fallback=True,
            fallback_reason=guard_reason,
        )
    if np.any(scores < domain[0]) or np.any(scores > domain[1]):
        reason = f"scores outside input domain [{domain[0]}, {domain[1]}]"
        if is_real_data_run or not fallback_allowed:
            raise CalibrationError(reason, code="fit_guard_failed")
        wrapper = ScoreCalibrator(
            method="identity",
            params={"requested_method": requested_method, "fallback_reason": reason},
            input_domain=domain,
        )
        return wrapper, _identity_fit_metadata(
            requested_method=requested_method,
            n_train=len(scores),
            fallback=True,
            fallback_reason=reason,
        )

    if requested_method == "platt":
        clipped = np.clip(scores, CAL_CLIP_EPS, 1.0 - CAL_CLIP_EPS)
        design = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        estimator = LogisticRegression(
            C=CAL_PLATT_C,
            solver=CAL_PLATT_SOLVER,
            max_iter=CAL_PLATT_MAX_ITER,
        )
        estimator.fit(design, y.astype(int), sample_weight=weights)
        coefficient = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])
        params = {
            "C": CAL_PLATT_C,
            "solver": CAL_PLATT_SOLVER,
            "max_iter": CAL_PLATT_MAX_ITER,
            "coefficient": coefficient,
            "intercept": intercept,
        }
        if not np.isfinite(coefficient) or not np.isfinite(intercept):
            reason = (
                "Platt parameters must be finite; observed "
                f"coefficient={coefficient}, intercept={intercept}"
            )
            if is_real_data_run or not fallback_allowed:
                raise CalibrationError(reason, code="nonfinite_platt_parameters")
            wrapper = ScoreCalibrator(
                method="identity",
                params={"requested_method": "platt", "fallback_reason": reason},
                clip_eps=CAL_CLIP_EPS,
                input_domain=domain,
                estimator=None,
            )
            metadata = _identity_fit_metadata(
                requested_method="platt",
                n_train=len(scores),
                fallback=True,
                fallback_reason=reason,
                coefficient=coefficient,
                intercept=intercept,
            )
            metadata["params"] = params
            return wrapper, metadata
        if coefficient <= 0.0:
            reason = f"Platt coefficient must be positive; observed {coefficient}"
            if is_real_data_run or not fallback_allowed:
                raise CalibrationError(reason, code="nonpositive_platt_coefficient")
            wrapper = ScoreCalibrator(
                method="identity",
                params={"requested_method": "platt", "fallback_reason": reason},
                clip_eps=CAL_CLIP_EPS,
                input_domain=domain,
                estimator=None,
            )
            metadata = _identity_fit_metadata(
                requested_method="platt",
                n_train=len(scores),
                fallback=True,
                fallback_reason=reason,
                coefficient=coefficient,
                intercept=intercept,
            )
            metadata["params"] = params
            return wrapper, metadata
        wrapper = ScoreCalibrator(
            method="platt",
            params=params,
            clip_eps=CAL_CLIP_EPS,
            input_domain=domain,
            estimator=estimator,
        )
        return wrapper, {
            "requested_method": "platt",
            "effective_method": "platt",
            "fallback": False,
            "fallback_reason": None,
            "coefficient": coefficient,
            "intercept": intercept,
            "n_train": int(len(scores)),
            "params": params,
        }

    estimator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    estimator.fit(scores, y.astype(int), sample_weight=weights)
    params = {"out_of_bounds": "clip", "increasing": True}
    wrapper = ScoreCalibrator(
        method="isotonic",
        params=params,
        clip_eps=CAL_CLIP_EPS,
        input_domain=domain,
        estimator=estimator,
    )
    _validate_isotonic_mapping(wrapper, scores)
    return wrapper, {
        "requested_method": "isotonic",
        "effective_method": "isotonic",
        "fallback": False,
        "fallback_reason": None,
        "coefficient": None,
        "intercept": None,
        "n_train": int(len(scores)),
        "params": params,
    }


def _validate_isotonic_mapping(calibrator: ScoreCalibrator, raw_scores: Any) -> None:
    unique = np.unique(np.asarray(raw_scores, dtype=float).reshape(-1))
    predicted = calibrator.predict(unique)
    if np.any(np.diff(predicted) < -1e-15):
        raise CalibrationError("isotonic mapping is not nondecreasing")
    # Distinct raw values may tie after calibration, but must never invert.
    running_max = np.maximum.accumulate(predicted)
    if np.any(predicted < running_max - 1e-15):
        raise CalibrationError("isotonic mapping introduces an inversion")


def _safe_auc(targets: Any, probabilities: Any) -> float | None:
    y = np.asarray(targets, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) == 0 or np.unique(y).size != 2:
        return None
    return float(roc_auc_score(y, p))


def walk_forward_calibrate(
    events: pd.DataFrame,
    *,
    is_real_data_run: bool,
    min_train_n: int = CAL_MIN_TRAIN_N,
) -> dict[str, Any]:
    """Apply past-only Ck mappings to folds 3..5 and retain fold-2 raw scores."""

    required = {
        "fold",
        "meta_y",
        "p_meta",
        "p_primary",
        "tb_uniqueness",
        "meta_eval_status",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise CalibrationError(f"walk-forward events missing columns: {missing}")
    output = events.copy()
    for column in CALIBRATION_COLUMNS[:-1]:
        output[column] = np.nan
    output["cal_is_raw"] = np.zeros(len(output), dtype=int)
    evaluated = output["meta_eval_status"].eq("evaluated")
    if output.loc[evaluated, "p_meta"].isna().any():
        raise CalibrationError("evaluated population contains missing p_meta")

    fit_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    evaluated_folds = sorted(int(item) for item in output.loc[evaluated, "fold"].unique())
    for fold in evaluated_folds:
        current_mask = evaluated & output["fold"].eq(fold)
        if fold == min(evaluated_folds):
            for score in ("p_meta", "p_primary"):
                output.loc[current_mask, f"{score}_cal_platt"] = output.loc[current_mask, score]
                output.loc[current_mask, f"{score}_cal_iso"] = output.loc[current_mask, score]
            output.loc[current_mask, "cal_is_raw"] = 1
            fold_records.append(
                {
                    "fold": fold,
                    "n_eval": int(current_mask.sum()),
                    "n_train": 0,
                    "train_folds": [],
                    "cal_is_raw": True,
                    "raw_reason": "fold_without_past_evaluated_history",
                }
            )
            continue

        train_mask = evaluated & output["fold"].lt(fold)
        train_indices = output.index[train_mask].tolist()
        train_folds = sorted(int(item) for item in output.loc[train_mask, "fold"].unique())
        n_train = int(train_mask.sum())
        if n_train < int(min_train_n):
            if is_real_data_run:
                raise CalibrationError(
                    f"real-data walk-forward fold {fold} unexpectedly has only {n_train} training rows",
                    code="real_data_min_train_guard",
                )
            for score in ("p_meta", "p_primary"):
                output.loc[current_mask, f"{score}_cal_platt"] = output.loc[current_mask, score]
                output.loc[current_mask, f"{score}_cal_iso"] = output.loc[current_mask, score]
            output.loc[current_mask, "cal_is_raw"] = 1
            fold_records.append(
                {
                    "fold": fold,
                    "n_eval": int(current_mask.sum()),
                    "n_train": n_train,
                    "train_folds": train_folds,
                    "cal_is_raw": True,
                    "raw_reason": "training_sample_below_minimum",
                }
            )
            continue

        for score in ("p_meta", "p_primary"):
            for method, suffix in (("platt", "platt"), ("isotonic", "iso")):
                calibrator, metadata = fit_score_calibrator(
                    output.loc[train_mask, score].to_numpy(dtype=float),
                    output.loc[train_mask, "meta_y"].to_numpy(dtype=int),
                    output.loc[train_mask, "tb_uniqueness"].to_numpy(dtype=float),
                    method=method,
                    input_domain=SCORE_DOMAINS[score],
                    is_real_data_run=is_real_data_run,
                    allow_fallback=False,
                )
                predictions = calibrator.predict(
                    output.loc[current_mask, score].to_numpy(dtype=float)
                )
                output.loc[current_mask, f"{score}_cal_{suffix}"] = predictions
                if method == "platt":
                    raw_auc = _safe_auc(
                        output.loc[current_mask, "meta_y"], output.loc[current_mask, score]
                    )
                    calibrated_auc = _safe_auc(
                        output.loc[current_mask, "meta_y"], predictions
                    )
                    if raw_auc is not None and abs(calibrated_auc - raw_auc) > 1e-12:
                        raise CalibrationError(
                            f"fold {fold} {score}: Platt AUC invariance failed",
                            details={"raw_auc": raw_auc, "calibrated_auc": calibrated_auc},
                        )
                fit_records.append(
                    {
                        "fold": fold,
                        "score": score,
                        "score_name": SCORE_SHORT_NAMES[score],
                        "method": method,
                        "train_folds": train_folds,
                        "train_indices": train_indices,
                        "n_train": n_train,
                        "n_eval": int(current_mask.sum()),
                        "calibrator": calibrator,
                        "fit_metadata": metadata,
                    }
                )
        fold_records.append(
            {
                "fold": fold,
                "n_eval": int(current_mask.sum()),
                "n_train": n_train,
                "train_folds": train_folds,
                "cal_is_raw": False,
                "raw_reason": None,
            }
        )
    output["cal_is_raw"] = output["cal_is_raw"].astype(int)
    return {"events": output, "fit_records": fit_records, "fold_records": fold_records}


def equal_count_bins(
    probabilities: Any,
    decision_ts: Any,
    symbols: Any,
    *,
    max_bins: int = 10,
) -> list[np.ndarray]:
    """Return deterministic equal-count positional bins, robust to score ties."""

    p = np.asarray(probabilities, dtype=float).reshape(-1)
    ts = pd.to_datetime(np.asarray(decision_ts).reshape(-1), utc=True, errors="raise")
    sym = np.asarray(symbols, dtype=str).reshape(-1)
    n = len(p)
    if not (len(ts) == len(sym) == n):
        raise ValueError("probabilities, decision_ts, and symbols must have equal length")
    if n == 0:
        return []
    if max_bins <= 0:
        raise ValueError("max_bins must be positive")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite")
    positions = np.arange(n, dtype=int)
    # Stable least-significant-to-most-significant sorting gives the specified
    # (p, decision_ts, symbol) order while explicitly using mergesort.
    order = positions[np.argsort(sym[positions], kind="mergesort")]
    ts_ns = np.asarray(ts.asi8, dtype=np.int64)
    order = order[np.argsort(ts_ns[order], kind="mergesort")]
    order = order[np.argsort(p[order], kind="mergesort")]
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, min(max_bins, n))]


def calibration_metrics(
    targets: Any,
    probabilities: Any,
    sample_weight: Any,
    decision_ts: Any,
    symbols: Any,
    *,
    max_bins: int = 10,
) -> dict[str, Any]:
    """Compute weighted/unweighted Brier, explicit-clipped log loss, ECE, and reliability."""

    y = np.asarray(targets, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(y) == 0 or not (len(y) == len(p) == len(weights)):
        raise ValueError("metrics inputs must have equal nonzero length")
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("targets must be finite binary values")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("probabilities must be finite in [0, 1]")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample_weight must be finite and strictly positive")
    squared_error = (p - y) ** 2
    clipped_positive = np.clip(p, CAL_LOGLOSS_EPS, 1.0 - CAL_LOGLOSS_EPS)
    clipped_negative = np.clip(1.0 - p, CAL_LOGLOSS_EPS, 1.0 - CAL_LOGLOSS_EPS)
    point_log_loss = -(
        y * np.log(clipped_positive) + (1.0 - y) * np.log(clipped_negative)
    )
    bins = equal_count_bins(p, decision_ts, symbols, max_bins=max_bins)
    reliability: list[dict[str, Any]] = []
    ece_unweighted = 0.0
    ece_weighted = 0.0
    total_weight = float(np.sum(weights))
    for bin_number, indices in enumerate(bins, start=1):
        bin_y = y[indices]
        bin_p = p[indices]
        bin_w = weights[indices]
        weight_sum = float(np.sum(bin_w))
        mean_p = float(np.mean(bin_p))
        event_rate = float(np.mean(bin_y))
        weighted_mean_p = float(np.average(bin_p, weights=bin_w))
        weighted_event_rate = float(np.average(bin_y, weights=bin_w))
        ece_unweighted += (len(indices) / len(y)) * abs(mean_p - event_rate)
        ece_weighted += (weight_sum / total_weight) * abs(
            weighted_mean_p - weighted_event_rate
        )
        reliability.append(
            {
                "bin": bin_number,
                "n": int(len(indices)),
                "weight_sum": weight_sum,
                "mean_p_unweighted": mean_p,
                "event_rate_unweighted": event_rate,
                "mean_p_weighted": weighted_mean_p,
                "event_rate_weighted": weighted_event_rate,
            }
        )
    return {
        "n": int(len(y)),
        "weight_sum": total_weight,
        "brier_unweighted": float(np.mean(squared_error)),
        "brier_weighted": float(np.average(squared_error, weights=weights)),
        "log_loss_unweighted": float(np.mean(point_log_loss)),
        "log_loss_weighted": float(np.average(point_log_loss, weights=weights)),
        "ece_unweighted": float(ece_unweighted),
        "ece_weighted": float(ece_weighted),
        "auc": _safe_auc(y, p),
        "reliability": reliability,
        "log_loss_clip_eps": CAL_LOGLOSS_EPS,
    }


def _version_columns() -> dict[str, dict[str, str]]:
    return {
        "meta": {
            "identity": "p_meta",
            "platt": "p_meta_cal_platt",
            "isotonic": "p_meta_cal_iso",
        },
        "primary": {
            "identity": "p_primary",
            "platt": "p_primary_cal_platt",
            "isotonic": "p_primary_cal_iso",
        },
    }


def _metrics_for_view(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for score_name, methods in _version_columns().items():
        result[score_name] = {}
        for method, column in methods.items():
            result[score_name][method] = calibration_metrics(
                frame["meta_y"],
                frame[column],
                frame["tb_uniqueness"],
                frame["decision_ts"],
                frame["symbol"],
            )
    return result


def evaluate_calibration(events: pd.DataFrame) -> dict[str, Any]:
    """Evaluate all six score versions per fold and on both aggregate slices."""

    evaluated = events[events["meta_eval_status"].eq("evaluated")].copy()
    if evaluated.empty:
        raise CalibrationError("no evaluated events available for calibration metrics")
    per_fold: dict[str, Any] = {}
    platt_auc_checks: list[dict[str, Any]] = []
    isotonic_auc_changes: list[dict[str, Any]] = []
    for fold in sorted(int(item) for item in evaluated["fold"].unique()):
        fold_frame = evaluated[evaluated["fold"].eq(fold)]
        fold_metrics = _metrics_for_view(fold_frame)
        per_fold[str(fold)] = fold_metrics
        for score_name in ("meta", "primary"):
            raw_auc = fold_metrics[score_name]["identity"]["auc"]
            platt_auc = fold_metrics[score_name]["platt"]["auc"]
            iso_auc = fold_metrics[score_name]["isotonic"]["auc"]
            delta_platt = None if raw_auc is None else float(platt_auc - raw_auc)
            if delta_platt is not None and abs(delta_platt) > 1e-12:
                raise CalibrationError(
                    f"fold {fold} {score_name}: Platt AUC invariance exceeds 1e-12"
                )
            platt_auc_checks.append(
                {
                    "fold": fold,
                    "score": score_name,
                    "raw_auc": raw_auc,
                    "platt_auc": platt_auc,
                    "delta": delta_platt,
                    "passed": delta_platt is None or abs(delta_platt) <= 1e-12,
                }
            )
            isotonic_auc_changes.append(
                {
                    "fold": fold,
                    "score": score_name,
                    "raw_auc": raw_auc,
                    "isotonic_auc": iso_auc,
                    "delta": None if raw_auc is None else float(iso_auc - raw_auc),
                }
            )
    calibrated_only = evaluated[evaluated["cal_is_raw"].eq(0)]
    aggregate: dict[str, Any] = {"all_evaluated": _metrics_for_view(evaluated)}
    aggregate["calibrated_only"] = (
        _metrics_for_view(calibrated_only) if not calibrated_only.empty else None
    )
    return {
        "target": "meta_y",
        "per_fold": per_fold,
        "aggregate": aggregate,
        "auc_contract": {
            "platt_per_fold_invariance_tolerance": 1e-12,
            "platt_per_fold_checks": platt_auc_checks,
            "isotonic_per_fold_changes_report_only": isotonic_auc_changes,
            "aggregate_calibrated_auc_interpretation": (
                "Different past-only mappings are used per fold and fold 2 remains raw; "
                "pooled calibrated AUC movement is expected and is not a ranking metric."
            ),
        },
    }


def select_calibration_method(method_to_brier: Mapping[str, float]) -> dict[str, Any]:
    """Apply the preregistered eligible-set and deterministic priority rule."""

    normalized: dict[str, float] = {}
    aliases = {"raw": "identity", "iso": "isotonic"}
    for method, value in method_to_brier.items():
        normalized[aliases.get(str(method).lower(), str(method).lower())] = float(value)
    required = {"identity", "platt", "isotonic"}
    if set(normalized) != required or not all(np.isfinite(list(normalized.values()))):
        raise ValueError(f"method_to_brier must contain finite values for {sorted(required)}")
    minimum = min(normalized.values())
    eligible = [
        method
        for method in ("platt", "identity", "isotonic")
        if normalized[method] <= minimum + CAL_METHOD_TIE_TOLERANCE
    ]
    selected = eligible[0]
    improvement = float(normalized["identity"] - normalized[selected])
    demonstrated = bool(improvement > CAL_IMPROVEMENT_MIN)
    semantics = (
        "uniqueness_weighted_calibrated_probability"
        if selected in {"platt", "isotonic"} and demonstrated
        else "uncalibrated_probability_like_score"
    )
    return {
        "selected_method": selected,
        "m_min": float(minimum),
        "eligible_methods": eligible,
        "method_brier_weighted": normalized,
        "tie_tolerance": CAL_METHOD_TIE_TOLERANCE,
        "priority_order": ["platt", "identity", "isotonic"],
        "improvement_brier": improvement,
        "improvement_threshold": CAL_IMPROVEMENT_MIN,
        "calibration_improvement_demonstrated": demonstrated,
        "probability_ready_for_v50": False,
        "weighted_calibration_artifact_available": True,
        "method_selection_uses_walk_forward_oof": True,
        "method_selection_has_post_selection_holdout": False,
        "selected_method_metrics_are_post_selection": True,
        "output_semantics": semantics,
        "kelly_input_semantics_decision": "pending_v50",
    }


def top_n_indices(events: pd.DataFrame, score: str | Sequence[float], n: int) -> np.ndarray:
    """Return original index labels for deterministic descending-score top-N."""

    if not {"decision_ts", "symbol"}.issubset(events.columns):
        raise ValueError("events must contain decision_ts and symbol")
    if isinstance(score, str):
        if score not in events.columns:
            raise ValueError(f"unknown score column: {score}")
        probabilities = events[score].to_numpy(dtype=float)
    else:
        probabilities = np.asarray(score, dtype=float).reshape(-1)
    if len(probabilities) != len(events):
        raise ValueError("score length must match events")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("top-N scores must be finite")
    if int(n) != n or n < 0 or n > len(events):
        raise ValueError(f"n must be an integer in [0, {len(events)}]")
    ranked = pd.DataFrame(
        {
            "_score": probabilities,
            "_decision_ts": pd.to_datetime(events["decision_ts"], utc=True),
            "_symbol": events["symbol"].astype(str).to_numpy(),
            "_index": events.index.to_numpy(),
        }
    )
    ranked = ranked.sort_values(
        ["_score", "_decision_ts", "_symbol"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked["_index"].iloc[: int(n)].to_numpy()


def top_n_comparison(
    events: pd.DataFrame,
    raw_score: str,
    calibrated_score: str,
    n: int,
) -> dict[str, Any]:
    """Compare raw and calibrated event sets and their event-unit economics."""

    raw_indices = top_n_indices(events, raw_score, n)
    calibrated_indices = top_n_indices(events, calibrated_score, n)
    raw_set = set(raw_indices.tolist())
    calibrated_set = set(calibrated_indices.tolist())
    net = events["tb_return"].astype(float) - CAL_ECONOMIC_COST
    return {
        "n": int(n),
        "overlap_n": int(len(raw_set & calibrated_set)),
        "displaced_n": int(len(raw_set - calibrated_set)),
        "sets_identical": raw_set == calibrated_set,
        "raw_sum_net": float(net.loc[raw_indices].sum()),
        "calibrated_sum_net": float(net.loc[calibrated_indices].sum()),
        "delta_sum_net": float(
            net.loc[calibrated_indices].sum() - net.loc[raw_indices].sum()
        ),
    }


def _raw_auc_comparison(events: pd.DataFrame) -> dict[str, Any]:
    evaluated = events[events["meta_eval_status"].eq("evaluated")]
    per_fold: dict[str, Any] = {}
    for fold in sorted(int(item) for item in evaluated["fold"].unique()):
        current = evaluated[evaluated["fold"].eq(fold)]
        per_fold[str(fold)] = {
            "n": int(len(current)),
            "meta_auc": _safe_auc(current["meta_y"], current["p_meta"]),
            "primary_auc": _safe_auc(current["meta_y"], current["p_primary"]),
        }
    return {
        "target": "meta_y",
        "per_fold": per_fold,
        "aggregate": {
            "n": int(len(evaluated)),
            "meta_auc": _safe_auc(evaluated["meta_y"], evaluated["p_meta"]),
            "primary_auc": _safe_auc(evaluated["meta_y"], evaluated["p_primary"]),
        },
    }


def _equal_n_raw_comparison(events: pd.DataFrame) -> dict[str, Any]:
    evaluated = events[events["meta_eval_status"].eq("evaluated")]
    per_fold: dict[str, Any] = {}
    total_n = 0
    meta_total = 0.0
    primary_total = 0.0
    for fold in sorted(int(item) for item in evaluated["fold"].unique()):
        current = evaluated[evaluated["fold"].eq(fold)]
        n = int(current["traded_meta"].astype(bool).sum())
        meta_indices = top_n_indices(current, "p_meta", n)
        primary_indices = top_n_indices(current, "p_primary", n)
        net = current["tb_return"].astype(float) - CAL_ECONOMIC_COST
        meta_sum = float(net.loc[meta_indices].sum())
        primary_sum = float(net.loc[primary_indices].sum())
        per_fold[str(fold)] = {
            "n": n,
            "meta_sum_net": meta_sum,
            "primary_sum_net": primary_sum,
        }
        total_n += n
        meta_total += meta_sum
        primary_total += primary_sum
    return {
        "tie_break": "score descending, decision_ts ascending, symbol ascending; mergesort",
        "per_fold": per_fold,
        "aggregate": {
            "n": int(total_n),
            "meta_sum_net": float(meta_total),
            "primary_sum_net": float(primary_total),
        },
    }


def _threshold_candidate_table(scores: np.ndarray, net: np.ndarray) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    all_thresholds = (
        *CAL_THRESHOLD_SELECTION_GRID,
        *CAL_THRESHOLD_DIAGNOSTIC_GRID,
        CAL_THRESHOLD_ABSTAIN,
    )
    for tau in all_thresholds:
        selected = scores >= float(tau)
        n_trades = int(selected.sum())
        sum_net = float(net[selected].sum()) if n_trades else 0.0
        table.append(
            {
                "tau": float(tau),
                "n_trades": n_trades,
                "sum_net": sum_net,
                "selection_candidate": tau in CAL_THRESHOLD_SELECTION_GRID
                or tau == CAL_THRESHOLD_ABSTAIN,
                "eligible": bool(
                    tau == CAL_THRESHOLD_ABSTAIN or n_trades >= CAL_THRESHOLD_MIN_TRADES
                ),
            }
        )
    return table


def _choose_threshold(scores: np.ndarray, net: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    table = _threshold_candidate_table(scores, net)
    candidates = [
        item for item in table if item["selection_candidate"] and item["eligible"]
    ]
    if not candidates:
        raise CalibrationError("threshold selection produced no eligible candidate")
    best = candidates[0]
    for candidate in candidates[1:]:
        improvement = float(candidate["sum_net"] - best["sum_net"])
        if improvement > CAL_THRESHOLD_TIE_TOL:
            best = candidate
            continue
        if abs(improvement) <= CAL_THRESHOLD_TIE_TOL:
            if candidate["tau"] > best["tau"]:
                best = candidate
            elif candidate["tau"] == best["tau"] and candidate["n_trades"] < best["n_trades"]:
                best = candidate
    return float(best["tau"]), table


def _fit_record_calibrator(
    fit_records: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    score: str,
    method: str,
) -> ScoreCalibrator | None:
    for record in fit_records:
        if (
            int(record["fold"]) == int(fold)
            and record["score"] == score
            and record["method"] == method
        ):
            return record["calibrator"]
    return None


def replay_threshold_mechanism(
    events: pd.DataFrame,
    *,
    score: str = "p_meta",
    method: str = "identity",
    fit_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Replay v48 thresholds, optionally using a same-scale Ck on history/current."""

    if method not in {"identity", "platt", "isotonic"}:
        raise ValueError(f"unknown replay method: {method}")
    evaluated = events[events["meta_eval_status"].eq("evaluated")]
    folds = sorted(int(item) for item in evaluated["fold"].unique())
    rows: list[dict[str, Any]] = []
    for position, fold in enumerate(folds):
        current = evaluated[evaluated["fold"].eq(fold)]
        raw_current = current[score].to_numpy(dtype=float)
        history_table: list[dict[str, Any]] | None = None
        calibrator: ScoreCalibrator | None = None
        if position == 0:
            tau = CAL_THRESHOLD_FIRST_FOLD
            current_scores = raw_current
            mapping = "raw_first_evaluation_fold"
        else:
            history = evaluated[evaluated["fold"].lt(fold)]
            raw_history = history[score].to_numpy(dtype=float)
            if method == "identity":
                history_scores = raw_history
                current_scores = raw_current
                mapping = "identity_raw"
            else:
                calibrator = _fit_record_calibrator(
                    fit_records, fold=fold, score=score, method=method
                )
                if calibrator is None:
                    raise CalibrationError(
                        f"missing C{fold} fit for same-scale {score}/{method} replay"
                    )
                history_scores = calibrator.predict(raw_history)
                current_scores = calibrator.predict(raw_current)
                mapping = f"C{fold}_{method}_applied_to_history_and_current"
            history_net = history["tb_return"].to_numpy(dtype=float) - CAL_ECONOMIC_COST
            tau, history_table = _choose_threshold(history_scores, history_net)
        selected = current_scores >= tau
        current_net = current["tb_return"].to_numpy(dtype=float) - CAL_ECONOMIC_COST
        rows.append(
            {
                "fold": fold,
                "tau": float(tau),
                "n_trades": int(selected.sum()),
                "sum_net": float(current_net[selected].sum()) if selected.any() else 0.0,
                "mapping": mapping,
                "history_threshold_table": history_table,
                "calibrator_params": None if calibrator is None else calibrator.params,
            }
        )
    return {
        "score": score,
        "method": method,
        "folds": rows,
        "selection_grid": list(CAL_THRESHOLD_SELECTION_GRID),
        "diagnostic_only_grid": list(CAL_THRESHOLD_DIAGNOSTIC_GRID),
        "abstain": CAL_THRESHOLD_ABSTAIN,
        "min_trades": CAL_THRESHOLD_MIN_TRADES,
        "tie_tolerance": CAL_THRESHOLD_TIE_TOL,
        "same_scale_protocol": method != "identity",
        "interpretation": (
            "The same past-fitted Ck mapping is applied to both history and current fold. "
            "The replay is causal in Ck and tau, but retrospective because the global method "
            "was selected post hoc; it is diagnostic, not a deployable simulation."
            if method != "identity"
            else "Raw-score reproduction of the frozen v48 threshold mechanism."
        ),
    }


def _selected_top_n_diagnostic(
    events: pd.DataFrame,
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evaluated = events[events["meta_eval_status"].eq("evaluated")]
    output: dict[str, Any] = {}
    for score_name, raw_column in (("meta", "p_meta"), ("primary", "p_primary")):
        method = str(selections[score_name]["selected_method"])
        selected_column = {
            "identity": raw_column,
            "platt": f"{raw_column}_cal_platt",
            "isotonic": f"{raw_column}_cal_iso",
        }[method]
        per_fold: dict[str, Any] = {}
        for fold in sorted(int(item) for item in evaluated["fold"].unique()):
            current = evaluated[evaluated["fold"].eq(fold)]
            n = int(current["traded_meta"].astype(bool).sum())
            comparison = top_n_comparison(current, raw_column, selected_column, n)
            if method in {"platt", "identity"} and not comparison["sets_identical"]:
                raise CalibrationError(
                    f"fold {fold} {score_name}: {method} changed the raw top-N event set"
                )
            per_fold[str(fold)] = comparison
        output[score_name] = {
            "selected_method": method,
            "per_fold": per_fold,
            "guard_applied": method in {"platt", "identity"},
        }
    return output


def _extract_matched_n(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        pooled = summary["evaluation"]["paired_tables"]["matched_n"][
            "pooled_folds_2_5"
        ]
        meta_n = int(pooled["meta_matched_n"]["n_trades"])
        primary_n = int(pooled["baseline_matched_reference"]["n_trades"])
        return {
            "source": "meta_training_summary_4h.json",
            "meta_sum_net": float(
                pooled["meta_matched_n"]["sum_net_event_units"]
            ),
            "primary_sum_net": float(
                pooled["baseline_matched_reference"]["sum_net_event_units"]
            ),
            "n": meta_n if meta_n == primary_n else None,
            "meta_n": meta_n,
            "primary_n": primary_n,
            "diagnostic_only": True,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _validate_canonical_anchors(
    events: pd.DataFrame,
    auc_comparison: Mapping[str, Any],
    equal_n: Mapping[str, Any],
    raw_threshold_replay: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    aggregate_auc = auc_comparison["aggregate"]
    fold_five_auc = auc_comparison["per_fold"]["5"]
    for observed, expected, label in (
        (aggregate_auc["meta_auc"], CANONICAL_AUC_ANCHORS["aggregate"]["meta"], "aggregate meta AUC"),
        (aggregate_auc["primary_auc"], CANONICAL_AUC_ANCHORS["aggregate"]["primary"], "aggregate primary AUC"),
        (fold_five_auc["meta_auc"], CANONICAL_AUC_ANCHORS["fold_5"]["meta"], "fold 5 meta AUC"),
        (fold_five_auc["primary_auc"], CANONICAL_AUC_ANCHORS["fold_5"]["primary"], "fold 5 primary AUC"),
    ):
        if observed is None or abs(float(observed) - float(expected)) > 1e-6:
            failures.append({"anchor": label, "expected": expected, "observed": observed})

    for key, expected in CANONICAL_EQUAL_N_ANCHORS.items():
        observed = equal_n["aggregate"] if key == "aggregate" else equal_n["per_fold"][str(key)]
        observed_n = int(observed["n"])
        observed_meta = float(observed["meta_sum_net"])
        observed_primary = float(observed["primary_sum_net"])
        if observed_n != expected["n"]:
            failures.append(
                {"anchor": f"equal-N {key} n", "expected": expected["n"], "observed": observed_n}
            )
        if abs(observed_meta - expected["meta"]) > 5e-4:
            failures.append(
                {"anchor": f"equal-N {key} meta", "expected": expected["meta"], "observed": observed_meta}
            )
        if abs(observed_primary - expected["primary"]) > 5e-4:
            failures.append(
                {"anchor": f"equal-N {key} primary", "expected": expected["primary"], "observed": observed_primary}
            )

    replay_by_fold = {int(item["fold"]): item for item in raw_threshold_replay["folds"]}
    for fold, expected in CANONICAL_RAW_THRESHOLD_ANCHORS.items():
        observed = replay_by_fold.get(fold)
        if observed is None:
            failures.append({"anchor": f"raw threshold fold {fold}", "observed": None})
            continue
        if float(observed["tau"]) != expected["tau"]:
            failures.append(
                {"anchor": f"raw threshold fold {fold} tau", "expected": expected["tau"], "observed": observed["tau"]}
            )
        if int(observed["n_trades"]) != expected["n"]:
            failures.append(
                {"anchor": f"raw threshold fold {fold} n", "expected": expected["n"], "observed": observed["n_trades"]}
            )
        if abs(float(observed["sum_net"]) - expected["sum_net"]) > 5e-4:
            failures.append(
                {"anchor": f"raw threshold fold {fold} sum", "expected": expected["sum_net"], "observed": observed["sum_net"]}
            )

    evaluated = events[events["meta_eval_status"].eq("evaluated")]
    unweighted_rate = float(evaluated["meta_y"].mean())
    weighted_rate = float(
        np.average(evaluated["meta_y"], weights=evaluated["tb_uniqueness"])
    )
    if abs(unweighted_rate - 0.3640) > 1e-4:
        failures.append(
            {"anchor": "unweighted base rate", "expected": 0.3640, "observed": unweighted_rate}
        )
    if abs(weighted_rate - 0.3161) > 1e-4:
        failures.append(
            {"anchor": "weighted base rate", "expected": 0.3161, "observed": weighted_rate}
        )
    if failures:
        raise AnchorError("canonical calibration anchors failed", details=failures)
    return {
        "passed": True,
        "auc_tolerance": 1e-6,
        "sum_net_tolerance": 5e-4,
        "base_rate_tolerance": 1e-4,
        "base_rates": {
            "unweighted": unweighted_rate,
            "weighted": weighted_rate,
        },
        "comparisons_verified": ["A_raw_auc", "B_raw_equal_n", "raw_threshold_replay"],
    }


def evaluate_additivity_gate(
    events: pd.DataFrame,
    *,
    summary: Mapping[str, Any],
    is_real_data_run: bool,
) -> dict[str, Any]:
    """Compute both raw comparisons and the preregistered real-data verdict."""

    auc_comparison = _raw_auc_comparison(events)
    equal_n = _equal_n_raw_comparison(events)
    matched_n = _extract_matched_n(summary)
    if not is_real_data_run:
        return {
            "applicable": False,
            "verdict": "not_applicable_non_real_data",
            "reason": "additivity anchors are canonical-real-data only",
        }
    meta_auc = float(auc_comparison["aggregate"]["meta_auc"])
    primary_auc = float(auc_comparison["aggregate"]["primary_auc"])
    meta_sum = float(equal_n["aggregate"]["meta_sum_net"])
    primary_sum = float(equal_n["aggregate"]["primary_sum_net"])
    if meta_auc <= primary_auc and meta_sum <= primary_sum:
        verdict = "not_demonstrated"
    elif meta_auc > primary_auc and meta_sum > primary_sum:
        verdict = "deterministic_gate_pass"
    else:
        verdict = "mixed_inconclusive"
    return {
        "applicable": True,
        "verdict": verdict,
        "comparison_A_raw_auc": auc_comparison,
        "comparison_B_raw_equal_n": equal_n,
        "diagnostics": {
            "matched_n_from_v48_summary": matched_n,
            "fold_5_regime_conditional_exception": {
                "meta_auc": auc_comparison["per_fold"]["5"]["meta_auc"],
                "primary_auc": auc_comparison["per_fold"]["5"]["primary_auc"],
            },
        },
        "deployment_ready": False,
        "does_not_automatically_authorize_v50": True,
    }


def transactional_write(
    output_dir: str | Path,
    writers: Mapping[str, Callable[[Path], None]],
    *,
    report_name: str = "calibration_report_4h.json",
    validate_staging: Callable[[Path], Any] | None = None,
    failure_injector: Callable[[str, str | None], None] | None = None,
) -> dict[str, Any]:
    """Stage, validate, commit report-last, and fully roll back any failure."""

    directory = Path(output_dir)
    if report_name not in writers:
        raise TransactionError(f"writers must include report marker {report_name!r}")
    names = list(writers)
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise TransactionError("managed artifact names must be unique plain filenames")
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".calibration_staging_", dir=directory))
    token = staging.name.removeprefix(".calibration_staging_")
    backups: dict[str, Path] = {}
    replaced: list[str] = []
    validation_result: Any = None

    def inject(stage: str, filename: str | None = None) -> None:
        if failure_injector is not None:
            failure_injector(stage, filename)

    def backup_path(name: str) -> Path:
        return directory / f".calibration_backup_{token}_{name}"

    def remove_staging() -> list[str]:
        errors: list[str] = []
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except Exception as exc:  # pragma: no cover - OS-level forensic path
                errors.append(f"remove staging {staging}: {exc}")
        return errors

    try:
        # The report writer runs last so it may hash the other staged artifacts.
        stage_order = [name for name in names if name != report_name] + [report_name]
        for name in stage_order:
            writers[name](staging / name)
            if not (staging / name).is_file():
                raise RuntimeError(f"writer did not create staged artifact {name}")
            inject("after_stage_file", name)
        if validate_staging is not None:
            validation_result = validate_staging(staging)
        inject("after_staging", None)
    except Exception as exc:
        cleanup_errors = remove_staging()
        raise TransactionError(
            f"staging or staging validation failed: {exc}",
            details={"cleanup_errors": cleanup_errors},
        ) from exc

    try:
        inject("before_commit", None)
        # Remove an old commit marker from service before touching other files.
        report_destination = directory / report_name
        if report_destination.exists():
            report_backup = backup_path(report_name)
            os.replace(report_destination, report_backup)
            backups[report_name] = report_backup
        inject("after_report_backup", report_name)

        commit_order = sorted(name for name in names if name != report_name) + [report_name]
        for name in commit_order:
            destination = directory / name
            if name != report_name and destination.exists():
                saved = backup_path(name)
                os.replace(destination, saved)
                backups[name] = saved
            os.replace(staging / name, destination)
            replaced.append(name)
            inject("after_replace", str(name))
        inject("after_commit", None)
    except Exception as exc:
        rollback_errors: list[str] = []
        # First remove every freshly committed file, including first-run files
        # that never had a backup.  Then restore all moved originals.
        for name in reversed(replaced):
            destination = directory / name
            if destination.exists():
                try:
                    destination.unlink()
                except Exception as rollback_exc:  # pragma: no cover - OS failure
                    rollback_errors.append(f"remove new {name}: {rollback_exc}")
        for name, saved in backups.items():
            destination = directory / name
            if destination.exists():
                try:
                    destination.unlink()
                except Exception as rollback_exc:  # pragma: no cover
                    rollback_errors.append(f"clear destination {name}: {rollback_exc}")
            if saved.exists():
                try:
                    os.replace(saved, destination)
                except Exception as rollback_exc:  # pragma: no cover
                    rollback_errors.append(f"restore {name}: {rollback_exc}")
        if not rollback_errors:
            rollback_errors.extend(remove_staging())
            for saved in backups.values():
                if saved.exists():
                    try:
                        saved.unlink()
                    except Exception as rollback_exc:  # pragma: no cover
                        rollback_errors.append(f"remove backup {saved}: {rollback_exc}")
        raise TransactionError(
            f"commit failed and rollback was attempted: {exc}",
            details={
                "rollback_succeeded": not rollback_errors,
                "rollback_errors": rollback_errors,
                "forensic_staging": str(staging) if rollback_errors else None,
            },
        ) from exc

    cleanup_errors: list[str] = []
    for saved in backups.values():
        if saved.exists():
            try:
                saved.unlink()
            except Exception as exc:  # pragma: no cover - commit remains valid
                cleanup_errors.append(f"remove backup {saved}: {exc}")
    cleanup_errors.extend(remove_staging())
    remaining_debris = [
        str(path)
        for path in directory.iterdir()
        if path.name.startswith(".calibration_staging_")
        or path.name.startswith(".calibration_backup_")
    ]
    if remaining_debris:
        cleanup_errors.append(f"remaining transaction debris: {remaining_debris}")
    identities = {name: _file_identity(directory / name) for name in names}
    return {
        "committed": True,
        "report_committed_last": replaced[-1] == report_name,
        "commit_order": replaced,
        "staging_validation": validation_result,
        "artifact_identities": identities,
        "cleanup_success": not cleanup_errors,
        "cleanup_errors": cleanup_errors,
        "remaining_debris": remaining_debris,
        "cleanup": {
            "success": not cleanup_errors,
            "errors": cleanup_errors,
            "remaining_debris": remaining_debris,
        },
    }


def _selection_inputs(evaluation: Mapping[str, Any], score_name: str) -> dict[str, float]:
    calibrated_only = evaluation["aggregate"]["calibrated_only"]
    if calibrated_only is None:
        raise CalibrationError("method selection requires at least one cal_is_raw=0 row")
    return {
        method: float(calibrated_only[score_name][method]["brier_weighted"])
        for method in ("identity", "platt", "isotonic")
    }


def _fit_final_artifacts(
    events: pd.DataFrame,
    *,
    selections: Mapping[str, Mapping[str, Any]],
    evaluation: Mapping[str, Any],
    fit_records: Sequence[Mapping[str, Any]],
    preflight: Mapping[str, Any],
    is_real_data_run: bool,
) -> tuple[dict[str, ScoreCalibrator], dict[str, dict[str, Any]]]:
    evaluated = events[events["meta_eval_status"].eq("evaluated")].copy()
    weights = evaluated["tb_uniqueness"].to_numpy(dtype=float)
    targets = evaluated["meta_y"].to_numpy(dtype=int)
    base_rates = {
        "unweighted_meta_y": float(np.mean(targets)),
        "uniqueness_weighted_meta_y": float(np.average(targets, weights=weights)),
        "n_evaluated": int(len(evaluated)),
        "positive_count": int(targets.sum()),
    }
    environment = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    final_calibrators: dict[str, ScoreCalibrator] = {}
    metadata_by_score: dict[str, dict[str, Any]] = {}
    for score_name, score_column in (("meta", "p_meta"), ("primary", "p_primary")):
        method = str(selections[score_name]["selected_method"])
        raw = evaluated[score_column].to_numpy(dtype=float)
        calibrator, final_fit = fit_score_calibrator(
            raw,
            targets,
            weights,
            method=method,
            input_domain=SCORE_DOMAINS[score_column],
            is_real_data_run=is_real_data_run,
            allow_fallback=False,
        )
        unweighted_calibrator, unweighted_fit = fit_score_calibrator(
            raw,
            targets,
            np.ones(len(raw), dtype=float),
            method=method,
            input_domain=SCORE_DOMAINS[score_column],
            is_real_data_run=is_real_data_run,
            allow_fallback=False,
        )
        unweighted_predictions = unweighted_calibrator.predict(raw)
        unweighted_metrics = calibration_metrics(
            targets,
            unweighted_predictions,
            weights,
            evaluated["decision_ts"],
            evaluated["symbol"],
        )
        reconstructible = method in {"platt", "identity"}
        unweighted_refit: dict[str, Any] = {
            "diagnostic_only": True,
            "deployable": False,
            "artifact_written": False,
            "method": method,
            "reconstructible": reconstructible,
            "mean_prediction": float(np.mean(unweighted_predictions)),
            "mean_prediction_unweighted": float(np.mean(unweighted_predictions)),
            "mean_prediction_weighted_tb_uniqueness": float(
                np.average(unweighted_predictions, weights=weights)
            ),
            "metrics": unweighted_metrics,
        }
        if reconstructible:
            unweighted_refit["fit"] = unweighted_fit

        platt_fits = []
        for record in fit_records:
            if record["score"] == score_column and record["method"] == "platt":
                platt_fits.append(
                    {
                        "fold": int(record["fold"]),
                        "train_folds": list(record["train_folds"]),
                        "n_train": int(record["n_train"]),
                        "coefficient": record["fit_metadata"]["coefficient"],
                        "intercept": record["fit_metadata"]["intercept"],
                    }
                )
        diagnostic_only = score_name == "primary"
        deployment_reason = (
            "Not bound to future p_primary-producing models; lineage binding and promotion "
            "are responsibilities of the consuming task."
            if diagnostic_only
            else None
        )
        selected_metrics = {
            "per_fold": {
                fold: metrics[score_name][method]
                for fold, metrics in evaluation["per_fold"].items()
            },
            "aggregate_all_evaluated": evaluation["aggregate"]["all_evaluated"][
                score_name
            ][method],
            "aggregate_calibrated_only": evaluation["aggregate"]["calibrated_only"][
                score_name
            ][method],
        }
        metadata_by_score[score_name] = {
            "version": "v49",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "score": score_column,
            "target": "meta_y",
            "selected_method": method,
            "final_fit": final_fit,
            "final_calibrator_has_no_evaluation_fold": True,
            "walk_forward_platt_fits": platt_fits,
            "walk_forward_oof_metrics": selected_metrics,
            "selection": dict(selections[score_name]),
            "base_rates": base_rates,
            "output_semantics": selections[score_name]["output_semantics"],
            "kelly_input_semantics_decision": "pending_v50",
            "input_contract": {
                "raw_score_required": True,
                "finite_required": True,
                "shape_preserved": True,
                "input_domain": list(SCORE_DOMAINS[score_column]),
                "conditional_population": (
                    "Scores are conditional on the fired-event population; primary domain "
                    "is [0.55, 1.0]."
                ),
            },
            "deployment": {
                "diagnostic_only": diagnostic_only,
                "deployable_candidate": not diagnostic_only,
                "reason": deployment_reason,
                "deployment_ready": False,
                "probability_ready_for_v50": False,
            },
            "unweighted_refit": unweighted_refit,
            "training": {
                "n": int(len(evaluated)),
                "open_time_min": evaluated["OpenTime"].min(),
                "open_time_max": evaluated["OpenTime"].max(),
                "folds": sorted(int(item) for item in evaluated["fold"].unique()),
                "sample_weight": "tb_uniqueness",
            },
            "input_lineage": preflight["lineage"],
            "environment": environment,
            "scale_transfer_assumption": {
                "calibrator_trained_on_oof_fold_model_scores": True,
                "intended_for_refit_final_model_scores": True,
                "refit_score_scale_transfer_assumption": True,
                "refit_score_scale_independently_validated": False,
            },
        }
        final_calibrators[score_name] = calibrator
    return final_calibrators, metadata_by_score


def _managed_artifact_names(symbols: Sequence[str]) -> list[str]:
    return [
        *(f"meta_oof_cal_{symbol}_4h.csv" for symbol in symbols),
        "calibrator_meta_4h.joblib",
        "calibrator_primary_4h.joblib",
        "calibrator_meta_4h.json",
        "calibrator_primary_4h.json",
        "calibration_report_4h.json",
    ]


def _assert_core_columns_exact(
    actual: pd.DataFrame,
    original: pd.DataFrame,
    name: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            original,
            check_dtype=False,
            check_names=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise TransactionError(
            f"{name}: original non-timestamp columns changed: {exc}"
        ) from exc


def _validate_calibration_staging(
    staging: Path,
    *,
    symbols: Sequence[str],
    input_paths: Mapping[str, Any],
    calibrators: Mapping[str, ScoreCalibrator],
) -> dict[str, Any]:
    names = _managed_artifact_names(symbols)
    missing = [name for name in names if not (staging / name).is_file()]
    if missing:
        raise TransactionError("staging is missing managed artifacts", details=missing)
    csv_checks: dict[str, Any] = {}
    for symbol in symbols:
        name = f"meta_oof_cal_{symbol}_4h.csv"
        actual = pd.read_csv(staging / name)
        original = pd.read_csv(input_paths["oof"][symbol])
        if tuple(actual.columns) != CALIBRATED_OOF_COLUMNS:
            raise TransactionError(f"{name}: staged CSV schema mismatch")
        if len(actual) != len(original):
            raise TransactionError(f"{name}: staged CSV row count mismatch")
        timestamp_columns = ("OpenTime", "decision_ts", "tb_exit_index")
        for column in timestamp_columns:
            try:
                actual_time = pd.to_datetime(actual[column], errors="raise", utc=True)
                original_time = pd.to_datetime(original[column], errors="raise", utc=True)
            except Exception as exc:
                raise TransactionError(
                    f"{name}: could not parse staged/source {column} as UTC: {exc}"
                ) from exc
            if not actual_time.equals(original_time):
                raise TransactionError(f"{name}: original timestamp column {column} changed")
        non_timestamp_columns = [
            column for column in OOF_COLUMNS if column not in timestamp_columns
        ]
        _assert_core_columns_exact(
            actual.loc[:, non_timestamp_columns],
            original.loc[:, non_timestamp_columns],
            name,
        )
        csv_checks[name] = {"rows": int(len(actual)), "core_columns_unchanged": True}
    smoke: dict[str, Any] = {}
    for score_name, domain in (("meta", SCORE_DOMAINS["p_meta"]), ("primary", SCORE_DOMAINS["p_primary"])):
        name = f"calibrator_{score_name}_4h.joblib"
        restored = joblib.load(staging / name)
        if not isinstance(restored, ScoreCalibrator):
            raise TransactionError(f"{name}: joblib did not contain ScoreCalibrator")
        query = np.array([(domain[0] + domain[1]) / 2.0], dtype=float)
        prediction = restored.predict(query)
        if prediction.shape != query.shape or not np.all(np.isfinite(prediction)):
            raise TransactionError(f"{name}: smoke prediction failed")
        expected = calibrators[score_name].predict(query)
        if not np.array_equal(prediction, expected):
            raise TransactionError(f"{name}: round-trip prediction mismatch")
        smoke[name] = {"raw": query.tolist(), "prediction": prediction.tolist()}
    for name in (
        "calibrator_meta_4h.json",
        "calibrator_primary_4h.json",
        "calibration_report_4h.json",
    ):
        try:
            parsed = json.loads((staging / name).read_text(encoding="utf-8"))
        except Exception as exc:
            raise TransactionError(f"{name}: staged JSON parse failed: {exc}") from exc
        if not isinstance(parsed, Mapping):
            raise TransactionError(f"{name}: staged JSON must contain an object")
    report = json.loads((staging / "calibration_report_4h.json").read_text(encoding="utf-8"))
    registry = report.get("output_artifacts")
    expected_registry = set(names) - {"calibration_report_4h.json"}
    if not isinstance(registry, Mapping) or set(registry) != expected_registry:
        raise TransactionError("report output_artifacts must register exactly eleven peers")
    for name in expected_registry:
        actual_identity = _file_identity(staging / name)
        record = registry[name]
        if (
            record.get("sha256") != actual_identity["sha256"]
            or int(record.get("size_bytes", -1)) != actual_identity["size_bytes"]
        ):
            raise TransactionError(f"report output_artifacts mismatch for {name}")
    return {
        "all_managed_files_present": True,
        "csv_checks": csv_checks,
        "joblib_smoke_predictions": smoke,
        "json_parse_checks": True,
        "eleven_peer_hashes_verified": True,
    }


def run_calibration(
    *,
    symbols: Sequence[str] = CANONICAL_SYMBOLS,
    timeframe: str = "4h",
    meta_oof_dir: str | Path = CANONICAL_META_OOF_DIR,
    output_dir: str | Path | None = None,
    is_real_data_run: bool | None = None,
    failure_injector: Callable[[str, str | None], None] | None = None,
) -> dict[str, Any]:
    """Run the complete v49 pipeline and atomically publish twelve artifacts."""

    symbol_tuple = tuple(str(item) for item in symbols)
    if timeframe != "4h":
        raise CalibrationError("v49 calibration supports timeframe='4h' only")
    input_directory = Path(meta_oof_dir).resolve()
    destination = Path(CANONICAL_OUTPUT_DIR if output_dir is None else output_dir).resolve()
    detected_real = bool(
        input_directory == CANONICAL_META_OOF_DIR.resolve()
        and symbol_tuple == tuple(CANONICAL_SYMBOLS)
        and timeframe == "4h"
    )
    if is_real_data_run is not None and bool(is_real_data_run) != detected_real:
        raise CalibrationError(
            "is_real_data_run override disagrees with resolved canonical identity"
        )
    real_run = detected_real
    if destination == Path(CANONICAL_OUTPUT_DIR).resolve() and not real_run:
        raise CalibrationError(
            "noncanonical input may not write to the canonical/default output directory"
        )
    if not real_run and destination == input_directory:
        raise CalibrationError("noncanonical inputs are read-only; choose a separate output_dir")

    # Required ordering: load/validate, WF calibration, metrics, every real-data
    # anchor, verdict, and only then creation of staging or any managed output.
    preflight = preflight_inputs(
        input_directory,
        symbol_tuple,
        timeframe=timeframe,
        is_real_data_run=real_run,
    )
    walk_forward = walk_forward_calibrate(
        preflight["events"],
        is_real_data_run=real_run,
        min_train_n=CAL_MIN_TRAIN_N,
    )
    calibrated_events = walk_forward["events"]
    evaluation = evaluate_calibration(calibrated_events)
    selections = {
        score_name: select_calibration_method(_selection_inputs(evaluation, score_name))
        for score_name in ("meta", "primary")
    }

    auc_comparison = _raw_auc_comparison(calibrated_events)
    equal_n = _equal_n_raw_comparison(calibrated_events)
    raw_threshold = replay_threshold_mechanism(
        calibrated_events, score="p_meta", method="identity"
    )
    anchor_validation: dict[str, Any]
    if real_run:
        anchor_validation = _validate_canonical_anchors(
            calibrated_events, auc_comparison, equal_n, raw_threshold
        )
    else:
        anchor_validation = {
            "passed": True,
            "canonical_anchors_applicable": False,
            "reason": "canonical anchors are real-data only",
        }
    additivity_gate = evaluate_additivity_gate(
        calibrated_events,
        summary=preflight["summary"],
        is_real_data_run=real_run,
    )

    top_n_diagnostic = _selected_top_n_diagnostic(calibrated_events, selections)
    selected_meta_method = selections["meta"]["selected_method"]
    calibrated_threshold = replay_threshold_mechanism(
        calibrated_events,
        score="p_meta",
        method=selected_meta_method,
        fit_records=walk_forward["fit_records"],
    )
    final_calibrators, calibrator_metadata = _fit_final_artifacts(
        calibrated_events,
        selections=selections,
        evaluation=evaluation,
        fit_records=walk_forward["fit_records"],
        preflight=preflight,
        is_real_data_run=real_run,
    )
    evaluated = calibrated_events[
        calibrated_events["meta_eval_status"].eq("evaluated")
    ]
    base_rates = {
        "unweighted": float(evaluated["meta_y"].mean()),
        "weighted_tb_uniqueness": float(
            np.average(evaluated["meta_y"], weights=evaluated["tb_uniqueness"])
        ),
    }
    fit_records_for_report = [
        {
            key: value
            for key, value in record.items()
            if key not in {"calibrator", "train_indices"}
        }
        for record in walk_forward["fit_records"]
    ]
    report: dict[str, Any] = {
        "project": "Kosar",
        "version": "v49",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_real_data_run": real_run,
        "is_canonical_output": destination == Path(CANONICAL_OUTPUT_DIR).resolve(),
        "identity": {
            "symbols": list(symbol_tuple),
            "timeframe": timeframe,
            "meta_oof_dir": str(input_directory),
            "output_dir": str(destination),
        },
        "pipeline_order": [
            "load",
            "validate",
            "walk_forward_calibration",
            "metrics",
            "verify_all_applicable_anchors",
            "determine_verdict",
            "create_and_validate_staging",
            "transactional_commit_report_last",
        ],
        "preflight": {
            "checks": preflight["checks"],
            "fold_counts": preflight["fold_counts"],
            "total_rows": preflight["total_rows"],
            "input_lineage": preflight["lineage"],
            "anchor_validation": anchor_validation,
        },
        "target": "meta_y",
        "base_rates": base_rates,
        "walk_forward": {
            "min_train_n": CAL_MIN_TRAIN_N,
            "fold_records": walk_forward["fold_records"],
            "fit_records": fit_records_for_report,
        },
        "calibration_evaluation": evaluation,
        "selections": selections,
        "economics": {
            "top_n_rank_invariance": top_n_diagnostic,
            "raw_threshold_replay": raw_threshold,
            "selected_calibrated_same_scale_threshold_replay": calibrated_threshold,
            "interpretation": (
                "Fixed-grid trade-count movement after calibration is a scale finding, not "
                "an economic signal; top-N is the ranking-economic diagnostic."
            ),
        },
        "raw_comparisons": {
            "A_auc": auc_comparison,
            "B_equal_n": equal_n,
            "matched_n_from_v48_summary": _extract_matched_n(preflight["summary"]),
        },
        "additivity_gate": additivity_gate,
        "deployment": {
            "deployment_ready": False,
            "probability_ready_for_v50": False,
            "kelly_input_semantics_decision": "pending_v50",
            "next_step_not_automatically_authorized": True,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "transaction_contract": {
            "managed_artifact_count": 12,
            "report_is_commit_marker_and_committed_last": True,
            "report_self_hash_excluded": True,
            "peer_artifact_registry_count": 11,
            "post_commit_cleanup_reporting": (
                "Actual cleanup_success, cleanup_errors, and remaining_debris are returned "
                "by run_calibration for terminal reporting because cleanup occurs after the "
                "report commit marker is written."
            ),
        },
        "output_artifacts": {},
    }

    managed_names = _managed_artifact_names(symbol_tuple)
    if len(managed_names) != 12 or len(set(managed_names)) != 12:
        raise CalibrationError(
            f"managed namespace must contain exactly 12 files; observed {managed_names}"
        )
    writers: dict[str, Callable[[Path], None]] = {}
    for symbol in symbol_tuple:
        name = f"meta_oof_cal_{symbol}_4h.csv"
        symbol_frame = calibrated_events[
            calibrated_events["symbol"].eq(symbol)
        ].sort_values("OpenTime", kind="mergesort")
        output_frame = symbol_frame.loc[:, CALIBRATED_OOF_COLUMNS].copy()
        writers[name] = (
            lambda path, frame=output_frame: frame.to_csv(path, index=False)
        )
    for score_name in ("meta", "primary"):
        writers[f"calibrator_{score_name}_4h.joblib"] = (
            lambda path, wrapper=final_calibrators[score_name]: joblib.dump(wrapper, path)
        )
        writers[f"calibrator_{score_name}_4h.json"] = (
            lambda path, payload=calibrator_metadata[score_name]: _write_json(path, payload)
        )

    def write_report(path: Path) -> None:
        registry: dict[str, Any] = {}
        for name in managed_names:
            if name == "calibration_report_4h.json":
                continue
            staged_path = path.parent / name
            if not staged_path.is_file():
                raise TransactionError(f"cannot hash missing staged peer {name}")
            identity = _file_identity(staged_path)
            registry[name] = {
                "path": str((destination / name).resolve()),
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        if len(registry) != 11:
            raise TransactionError("report peer registry must contain exactly eleven artifacts")
        report["output_artifacts"] = registry
        _write_json(path, report)

    writers["calibration_report_4h.json"] = write_report
    transaction = transactional_write(
        destination,
        writers,
        report_name="calibration_report_4h.json",
        validate_staging=lambda staging: _validate_calibration_staging(
            staging,
            symbols=symbol_tuple,
            input_paths=preflight["input_paths"],
            calibrators=final_calibrators,
        ),
        failure_injector=failure_injector,
    )
    report_path = destination / "calibration_report_4h.json"
    report_hash = _sha256(report_path)
    return {
        **report,
        "symbols": list(symbol_tuple),
        "selected_methods": {
            score_name: selection["selected_method"]
            for score_name, selection in selections.items()
        },
        "output_paths": {
            name: str((destination / name).resolve()) for name in managed_names
        },
        "transaction": transaction,
        "report_path": str(report_path),
        "report_sha256": report_hash,
        "report_sha256_after_commit": report_hash,
        "report_size_bytes_after_commit": int(report_path.stat().st_size),
    }
