# ==============================================================================
# meta_model.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Offline 4h meta-label training. Not used by live trading.
# ==============================================================================

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import CONFIG


META_PROJECT_ROOT = Path(__file__).resolve().parent
CANONICAL_PRIMARY_DIR = (META_PROJECT_ROOT / "data" / "models").resolve()
CANONICAL_DATASET_DIR = (META_PROJECT_ROOT / "data" / "datasets_4h").resolve()
CANONICAL_EXTERNAL_DIR = (META_PROJECT_ROOT / "data" / "external").resolve()
CANONICAL_HMM_DIR = (META_PROJECT_ROOT / "log" / "hmm_walkforward_fwd_long").resolve()
CANONICAL_OUTPUT_DIR = (META_PROJECT_ROOT / "data" / "models").resolve()

CANONICAL_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
)

META_BASE_THRESHOLD = 0.55
META_MIN_TRAIN_EVENTS = 50
META_MIN_TRADES_FOR_TAU = 30
META_THRESHOLDS = tuple(round(0.40 + 0.05 * i, 2) for i in range(8))
BASELINE_THRESHOLDS = tuple(round(0.55 + 0.05 * i, 2) for i in range(5))
DIAGNOSTIC_THRESHOLDS = (0.80, 0.85, 0.90)
ABSTAIN_THRESHOLD = 1.01
TAU_TIE_TOLERANCE = 1e-12

DATASET_COLUMNS = ["volatility", "log_range", "tb_exit_index"]
META_FEATURE_COLUMNS = [
    "p_primary",
    "volatility",
    "log_range",
    "hour_of_day",
    "day_of_week",
    "funding_z",
    "funding_extreme_pos",
    "vpin",
    "vpin_z",
    "symbol",
]
META_HMM_FEATURE_COLUMNS = [
    "hmm_regime",
    "hmm_bull_prob",
    "hmm_neutral_prob",
    "hmm_bear_prob",
    "hmm_confidence",
    "hmm_policy_code",
    "hmm_regime_age_hours",
]
META_HMM_COLUMN_MAP = {
    "timestamp": "timestamp",
    "ok": "hmm_ok",
    "regime": "hmm_regime",
    "policy": "hmm_policy",
    "bull_prob": "hmm_bull_prob",
    "neutral_prob": "hmm_neutral_prob",
    "bear_prob": "hmm_bear_prob",
    "confidence": "hmm_confidence",
}
HMM_POLICY_CODE_MAP = {"block": 0, "caution": 1, "allow": 2}
HMM_REGIMES = {"bull", "neutral", "bear"}

META_CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 5,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU",
    "thread_count": 1,
    "boosting_type": "Ordered",
}

META_OOF_COLUMNS = [
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
]

EXPECTED_BASELINE_ANCHOR = {
    2: {"tau": 0.55, "n_trades": 1745, "sum_net": -2.860},
    3: {"tau": 0.75, "n_trades": 467, "sum_net": 2.773},
    4: {"tau": 0.75, "n_trades": 482, "sum_net": 2.624},
    5: {"tau": 0.75, "n_trades": 740, "sum_net": -4.426},
}

_SOURCE_FEATURES = {
    "dataset": ["volatility", "log_range", "tb_exit_index"],
    "funding": ["funding_z", "funding_extreme_pos"],
    "vpin": ["vpin", "vpin_z"],
    "hmm": list(META_HMM_FEATURE_COLUMNS),
}


class BaselineAnchorError(RuntimeError):
    """Raised when the independently supplied baseline anchor does not match."""


def _project_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return dict(CONFIG if config is None else config)


def derive_cost_and_embargo(config: Mapping[str, Any] | None = None) -> tuple[float, pd.Timedelta]:
    """Derive and verify the locked v48 economic and purge contracts."""

    cfg = _project_config(config)
    required = [
        "PRIMARY_FEE_BPS_PER_SIDE",
        "PRIMARY_SLIPPAGE_BPS_PER_SIDE",
        "TRIPLE_BARRIER_EMBARGO_BARS",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"v48 requires config keys: {missing}")

    fee = float(cfg["PRIMARY_FEE_BPS_PER_SIDE"])
    slippage = float(cfg["PRIMARY_SLIPPAGE_BPS_PER_SIDE"])
    cost = (fee + slippage) * 2.0 / 10000.0
    embargo = pd.Timedelta(hours=4 * int(cfg["TRIPLE_BARRIER_EMBARGO_BARS"]))
    if not math.isclose(cost, 0.0030, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"v48 cost contract requires 0.0030, got {cost!r}")
    if embargo != pd.Timedelta(hours=96):
        raise ValueError(f"v48 embargo contract requires 96h, got {embargo}")
    return float(cost), embargo


COST_ROUND_TRIP, EMBARGO = derive_cost_and_embargo(CONFIG)


def _normalize_symbol(value: str) -> str:
    return str(value).strip().upper().replace("/", "").replace("-", "")


def normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols]
    if not normalized:
        raise ValueError("At least one symbol is required")
    if any(not symbol for symbol in normalized):
        raise ValueError("Symbols must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate symbols are not allowed: {normalized}")
    known = [symbol for symbol in CANONICAL_SYMBOLS if symbol in normalized]
    unknown = sorted(symbol for symbol in normalized if symbol not in CANONICAL_SYMBOLS)
    return known + unknown


def _resolved(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def resolve_run_identity(
    symbols: Sequence[str],
    dataset_dir: str | os.PathLike[str],
    external_dir: str | os.PathLike[str],
    hmm_dir: str | os.PathLike[str] | None,
    output_dir: str | os.PathLike[str],
    timeframe: str,
) -> dict[str, Any]:
    """Resolve the independent real-data and canonical-output identities."""

    if str(timeframe) != "4h":
        raise ValueError(f"v48 supports timeframe='4h' only; got {timeframe!r}")

    ordered_symbols = normalize_symbols(symbols)
    dataset_path = _resolved(dataset_dir)
    external_path = _resolved(external_dir)
    hmm_path = None if hmm_dir is None else _resolved(hmm_dir)
    output_path = _resolved(output_dir)

    canonical_base = (
        dataset_path == CANONICAL_DATASET_DIR
        and external_path == CANONICAL_EXTERNAL_DIR
        and ordered_symbols == list(CANONICAL_SYMBOLS)
        and str(timeframe) == "4h"
    )
    if canonical_base and hmm_path is None:
        raise ValueError("Canonical real-data v48 runs require --hmm-dir")

    is_real_data_run = bool(canonical_base and hmm_path == CANONICAL_HMM_DIR)
    is_canonical_output = bool(is_real_data_run and output_path == CANONICAL_OUTPUT_DIR)
    if output_path == CANONICAL_OUTPUT_DIR and not is_canonical_output:
        raise ValueError(
            "Non-canonical input identity cannot write to the default canonical output directory; "
            "provide a separate --output-dir"
        )

    return {
        "symbols": ordered_symbols,
        "timeframe": "4h",
        "primary_dir": str(CANONICAL_PRIMARY_DIR),
        "dataset_dir": str(dataset_path),
        "external_dir": str(external_path),
        "hmm_dir": None if hmm_path is None else str(hmm_path),
        "output_dir": str(output_path),
        "is_real_data_run": is_real_data_run,
        "is_canonical_output": is_canonical_output,
    }


def _input_paths(
    symbols: Sequence[str],
    primary_dir: Path,
    dataset_dir: Path,
    external_dir: Path,
    hmm_dir: Path | None,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for symbol in symbols:
        files.extend(
            [
                {
                    "role": "primary_oof",
                    "symbol": symbol,
                    "path": primary_dir / f"primary_oof_{symbol}_4h.csv",
                },
                {
                    "role": "dataset",
                    "symbol": symbol,
                    "path": dataset_dir / f"{symbol}_4h.csv",
                },
                {
                    "role": "funding",
                    "symbol": symbol,
                    "path": external_dir / f"funding_{symbol}.csv",
                },
                {
                    "role": "vpin",
                    "symbol": symbol,
                    "path": external_dir / f"vpin_{symbol}_1h.csv",
                },
            ]
        )
        if hmm_dir is not None:
            files.append(
                {
                    "role": "hmm",
                    "symbol": symbol,
                    "path": hmm_dir / f"hmm_walkforward_{symbol}_1h_pomegranate_normal.csv",
                }
            )
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_input_files(
    symbols: Sequence[str],
    primary_dir: str | os.PathLike[str],
    dataset_dir: str | os.PathLike[str],
    external_dir: str | os.PathLike[str],
    hmm_dir: str | os.PathLike[str] | None,
) -> list[dict[str, Any]]:
    """Require every requested input before reading or fitting any model."""

    entries = _input_paths(
        symbols,
        _resolved(primary_dir),
        _resolved(dataset_dir),
        _resolved(external_dir),
        None if hmm_dir is None else _resolved(hmm_dir),
    )
    missing = [str(entry["path"].resolve()) for entry in entries if not entry["path"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required v48 input files ({len(missing)}): {missing}")

    metadata: list[dict[str, Any]] = []
    for entry in entries:
        path = entry["path"].resolve()
        metadata.append(
            {
                "role": entry["role"],
                "symbol": entry["symbol"],
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return metadata


def parse_mixed_utc(values: pd.Series, context: str) -> tuple[pd.Series, int]:
    """Parse the required mixed ISO timestamp format and count invalid values."""

    parsed = pd.to_datetime(values, utc=True, format="mixed", errors="coerce")
    invalid_count = int(parsed.isna().sum())
    if len(parsed) > 0 and invalid_count == len(parsed):
        raise ValueError(f"{context} has no valid timestamps after mixed UTC parsing")
    return parsed, invalid_count


def _require_columns(frame: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _finite_numeric(series: pd.Series, context: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} must contain only finite numeric values")
    return numeric


def _load_primary_oof(
    path: Path,
    symbol: str,
    is_real_data_run: bool,
) -> tuple[pd.DataFrame, dict[str, Any], int]:
    raw = pd.read_csv(path)
    schema = list(raw.columns)
    required = ["OpenTime", "fold", "p_primary", "y", "tb_return"]
    _require_columns(raw, required, f"primary OOF {path}")

    parsed, invalid_ts = parse_mixed_utc(raw["OpenTime"], f"primary OOF {path} OpenTime")
    raw = raw.assign(OpenTime=parsed).loc[parsed.notna()].copy()
    if raw.empty:
        raise ValueError(f"primary OOF {path} is empty after timestamp parsing")
    raw = raw.sort_values("OpenTime", kind="mergesort")
    if raw["OpenTime"].duplicated().any():
        raise ValueError(f"primary OOF {path} contains duplicate OpenTime values")

    fold_numeric = _finite_numeric(raw["fold"], f"primary OOF {path} fold")
    if not np.equal(fold_numeric.to_numpy(), np.floor(fold_numeric.to_numpy())).all():
        raise ValueError(f"primary OOF {path} fold values must be integers")
    raw["fold"] = fold_numeric.astype(int)
    observed_folds = set(raw["fold"].unique().tolist())
    if observed_folds != {1, 2, 3, 4, 5}:
        raise ValueError(f"primary OOF {path} folds must be exactly 1..5; got {sorted(observed_folds)}")

    y = _finite_numeric(raw["y"], f"primary OOF {path} y")
    if not set(y.astype(int).unique().tolist()).issubset({0, 1}) or not np.equal(y, y.astype(int)).all():
        raise ValueError(f"primary OOF {path} y must contain only 0/1")
    raw["y"] = y.astype(int)

    raw["p_primary"] = _finite_numeric(raw["p_primary"], f"primary OOF {path} p_primary")
    if not raw["p_primary"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError(f"primary OOF {path} p_primary must be in [0, 1]")
    raw["tb_return"] = _finite_numeric(raw["tb_return"], f"primary OOF {path} tb_return")

    weight_fallback_count = 0
    if "tb_uniqueness" not in raw.columns:
        if is_real_data_run:
            raise ValueError(f"real-data OOF {path} requires tb_uniqueness")
        raw["tb_uniqueness"] = 1.0
        weight_fallback_count = int(len(raw))
    raw["tb_uniqueness"] = _finite_numeric(
        raw["tb_uniqueness"], f"primary OOF {path} tb_uniqueness"
    )
    if not (raw["tb_uniqueness"] > 0.0).all():
        raise ValueError(f"primary OOF {path} tb_uniqueness must be strictly positive")

    fired = raw.loc[raw["p_primary"] >= META_BASE_THRESHOLD].copy()
    fired["symbol"] = symbol
    diagnostics = {
        "path": str(path.resolve()),
        "schema": schema,
        "rows": int(len(raw)),
        "invalid_timestamp_count": invalid_ts,
        "source_start": raw["OpenTime"].min(),
        "source_end": raw["OpenTime"].max(),
        "fold_counts": {str(k): int(v) for k, v in raw["fold"].value_counts().sort_index().items()},
        "fired_count": int(len(fired)),
        "fired_fold_counts": {
            str(k): int(v) for k, v in fired["fold"].value_counts().sort_index().items()
        },
    }
    return fired, diagnostics, weight_fallback_count


def _load_dataset_slice(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    schema = list(raw.columns)
    selected_columns = ["OpenTime", *DATASET_COLUMNS]
    if len(selected_columns) != 4:
        raise ValueError(
            f"dataset slice contract violation for {path}: expected exactly 4 columns, "
            f"got {selected_columns}"
        )
    _require_columns(raw, selected_columns, f"dataset {path}")
    sliced = raw.loc[:, selected_columns].copy()
    if list(sliced.columns) != selected_columns:
        raise ValueError(
            f"dataset slice contract violation for {path}: expected {selected_columns}, got {list(sliced.columns)}"
        )

    parsed, invalid_ts = parse_mixed_utc(sliced["OpenTime"], f"dataset {path} OpenTime")
    sliced["OpenTime"] = parsed
    sliced = sliced.loc[sliced["OpenTime"].notna()].sort_values("OpenTime", kind="mergesort")
    if sliced.empty:
        raise ValueError(f"dataset {path} is empty after timestamp parsing")
    if sliced["OpenTime"].duplicated().any():
        raise ValueError(f"dataset {path} contains duplicate OpenTime values")

    diagnostics = {
        "path": str(path.resolve()),
        "schema": schema,
        "selected_schema": list(sliced.columns),
        "rows": int(len(sliced)),
        "invalid_timestamp_count": invalid_ts,
        "raw_tb_exit_missing_count": int(sliced["tb_exit_index"].isna().sum()),
        "source_start": sliced["OpenTime"].min(),
        "source_end": sliced["OpenTime"].max(),
    }
    return sliced, diagnostics


def _join_dataset_events(
    events: pd.DataFrame,
    dataset: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    right = dataset.copy()
    right["_dataset_matched"] = True
    joined = events.merge(right, on="OpenTime", how="left", validate="one_to_one", sort=False)
    joined["_dataset_matched"] = joined["_dataset_matched"].fillna(False).astype(bool)
    if not joined["_dataset_matched"].all():
        missing = joined.loc[~joined["_dataset_matched"], "OpenTime"].head(5).tolist()
        raise ValueError(f"dataset {path} has no exact row for fired OOF events: {missing}")

    exit_raw = joined["tb_exit_index"]
    parsed_exit = pd.to_datetime(exit_raw, utc=True, format="mixed", errors="coerce")
    invalid_nonempty = int((exit_raw.notna() & parsed_exit.isna()).sum())
    joined["tb_exit_index"] = parsed_exit
    if joined["tb_exit_index"].isna().any():
        missing = joined.loc[joined["tb_exit_index"].isna(), "OpenTime"].head(5).tolist()
        raise ValueError(f"joined fired events from {path} require tb_exit_index: {missing}")

    lower = joined["OpenTime"] + pd.Timedelta(hours=4)
    upper = joined["OpenTime"] + pd.Timedelta(hours=96)
    bad_exit = (joined["tb_exit_index"] < lower) | (joined["tb_exit_index"] > upper)
    if bad_exit.any():
        sample = joined.loc[bad_exit, ["OpenTime", "tb_exit_index"]].head(5).to_dict("records")
        raise ValueError(f"tb_exit_index outside [OpenTime+4h, OpenTime+96h] in {path}: {sample}")

    for column in ["volatility", "log_range"]:
        joined[column] = _finite_numeric(joined[column], f"joined dataset {path} {column}")

    diagnostics = {
        "matched_count": int(joined["_dataset_matched"].sum()),
        "outside_tolerance_count": 0,
        "invalid_tb_exit_timestamp_count": invalid_nonempty,
        "matched_but_feature_nan_count": 0,
    }
    return joined, diagnostics


def _prepare_numeric_source(
    path: Path,
    source: str,
    time_column: str,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    schema = list(raw.columns)
    _require_columns(raw, [time_column, *feature_columns], f"{source} source {path}")
    parsed, invalid_ts = parse_mixed_utc(raw[time_column], f"{source} source {path} {time_column}")
    raw = raw.assign(**{time_column: parsed}).loc[parsed.notna()].copy()
    if raw.empty:
        raise ValueError(f"{source} source {path} is empty after timestamp parsing")
    raw = raw.sort_values(time_column, kind="mergesort")
    before_dedup = int(len(raw))
    raw = raw.drop_duplicates(subset=[time_column], keep="last")

    prepared = raw.loc[:, [time_column, *feature_columns]].copy()
    inf_counts: dict[str, int] = {}
    for column in feature_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        values = prepared[column].to_numpy(dtype=float)
        inf_mask = np.isinf(values)
        inf_counts[column] = int(inf_mask.sum())
        if inf_mask.any():
            prepared.loc[inf_mask, column] = np.nan

    ready_column = f"_{source}_source_ts"
    prepared = prepared.rename(columns={time_column: ready_column})
    diagnostics = {
        "path": str(path.resolve()),
        "schema": schema,
        "rows_after_timestamp_filter": before_dedup,
        "deduplicated_count": before_dedup - int(len(prepared)),
        "invalid_timestamp_count": invalid_ts,
        "source_start": prepared[ready_column].min(),
        "source_end": prepared[ready_column].max(),
        "inf_to_nan_count_by_feature": inf_counts,
    }
    return prepared, diagnostics


def _merge_asof_source(
    events: pd.DataFrame,
    prepared: pd.DataFrame,
    source: str,
    feature_columns: Sequence[str],
    tolerance: pd.Timedelta,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ready_column = f"_{source}_source_ts"
    left = events.copy()
    left["_merge_order"] = np.arange(len(left), dtype=int)
    left = left.sort_values("decision_ts", kind="mergesort")
    right = prepared.sort_values(ready_column, kind="mergesort")
    merged = pd.merge_asof(
        left,
        right.loc[:, [ready_column, *feature_columns]],
        left_on="decision_ts",
        right_on=ready_column,
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_merge_order", kind="mergesort").drop(columns=["_merge_order"])
    match_column = f"_{source}_matched"
    merged[match_column] = merged[ready_column].notna()

    if (merged.loc[merged[match_column], ready_column] > merged.loc[merged[match_column], "decision_ts"]).any():
        raise ValueError(f"Causality violation: {source} source timestamp is after decision_ts")

    if source == "hmm":
        usable = merged["hmm_regime"].notna() & merged["hmm_regime"].ne("missing")
        usable &= merged[[col for col in feature_columns if col != "hmm_regime"]].notna().all(axis=1)
    else:
        usable = merged[list(feature_columns)].notna().all(axis=1)
    diagnostics = {
        "matched_count": int(merged[match_column].sum()),
        "outside_tolerance_count": int((~merged[match_column]).sum()),
        "matched_but_feature_nan_count": int((merged[match_column] & ~usable).sum()),
    }
    return merged, diagnostics


def parse_hmm_ok(value: Any) -> bool:
    """Parse only the explicitly allowed HMM success representations."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value) in {0.0, 1.0}:
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
    raise ValueError(f"Invalid hmm_ok value: {value!r}")


def _prepare_hmm_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    schema = list(raw.columns)
    source_columns = [
        META_HMM_COLUMN_MAP["timestamp"],
        META_HMM_COLUMN_MAP["ok"],
        META_HMM_COLUMN_MAP["regime"],
        META_HMM_COLUMN_MAP["policy"],
        META_HMM_COLUMN_MAP["bull_prob"],
        META_HMM_COLUMN_MAP["neutral_prob"],
        META_HMM_COLUMN_MAP["bear_prob"],
        META_HMM_COLUMN_MAP["confidence"],
    ]
    _require_columns(raw, source_columns, f"HMM source {path}")

    timestamp_column = META_HMM_COLUMN_MAP["timestamp"]
    parsed, invalid_ts = parse_mixed_utc(raw[timestamp_column], f"HMM source {path} timestamp")
    raw = raw.assign(**{timestamp_column: parsed}).loc[parsed.notna()].copy()
    if raw.empty:
        raise ValueError(f"HMM source {path} is empty after timestamp parsing")
    raw = raw.sort_values(timestamp_column, kind="mergesort")
    before_dedup = int(len(raw))
    raw = raw.drop_duplicates(subset=[timestamp_column], keep="last").copy()

    try:
        raw["_hmm_ok_parsed"] = [parse_hmm_ok(value) for value in raw[META_HMM_COLUMN_MAP["ok"]]]
    except ValueError as exc:
        raise ValueError(f"HMM source {path}: {exc}") from exc

    true_mask = raw["_hmm_ok_parsed"]
    true_rows = raw.loc[true_mask].copy()
    if not true_rows.empty:
        true_rows["_regime"] = true_rows[META_HMM_COLUMN_MAP["regime"]].astype("string").str.lower()
        true_rows["_policy"] = true_rows[META_HMM_COLUMN_MAP["policy"]].astype("string").str.lower()
        if not true_rows["_regime"].isin(HMM_REGIMES).all():
            bad = true_rows.loc[~true_rows["_regime"].isin(HMM_REGIMES), "_regime"].unique().tolist()
            raise ValueError(f"HMM source {path} has invalid true-row regimes: {bad}")
        if not true_rows["_policy"].isin(HMM_POLICY_CODE_MAP).all():
            bad = true_rows.loc[~true_rows["_policy"].isin(HMM_POLICY_CODE_MAP), "_policy"].unique().tolist()
            raise ValueError(f"HMM source {path} has invalid true-row policies: {bad}")

        probability_columns = [
            META_HMM_COLUMN_MAP["bull_prob"],
            META_HMM_COLUMN_MAP["neutral_prob"],
            META_HMM_COLUMN_MAP["bear_prob"],
        ]
        for column in [*probability_columns, META_HMM_COLUMN_MAP["confidence"]]:
            true_rows[column] = _finite_numeric(true_rows[column], f"HMM source {path} {column}")
            if not true_rows[column].between(0.0, 1.0, inclusive="both").all():
                raise ValueError(f"HMM source {path} true-row {column} must be in [0, 1]")
        probability_sum = true_rows[probability_columns].sum(axis=1)
        if not np.isclose(probability_sum.to_numpy(dtype=float), 1.0, atol=1e-3, rtol=0.0).all():
            raise ValueError(f"HMM source {path} true-row probabilities must sum to 1 within 1e-3")

        regime_change = true_rows["_regime"].ne(true_rows["_regime"].shift()).fillna(True)
        regime_group = regime_change.cumsum()
        block_start = true_rows.groupby(regime_group, sort=False)[timestamp_column].transform("min")
        true_rows["_regime_age_hours"] = (
            (true_rows[timestamp_column] - block_start) / pd.Timedelta(hours=1)
        ).astype(float)

    prepared = pd.DataFrame(index=raw.index)
    prepared["_hmm_source_ts"] = raw[timestamp_column]
    prepared["_hmm_ok_matched"] = raw["_hmm_ok_parsed"].astype(bool)
    prepared["hmm_regime"] = "missing"
    for column in META_HMM_FEATURE_COLUMNS[1:]:
        prepared[column] = np.nan

    if not true_rows.empty:
        idx = true_rows.index
        prepared.loc[idx, "hmm_regime"] = true_rows["_regime"].astype(str)
        prepared.loc[idx, "hmm_bull_prob"] = true_rows[META_HMM_COLUMN_MAP["bull_prob"]].to_numpy(dtype=float)
        prepared.loc[idx, "hmm_neutral_prob"] = true_rows[META_HMM_COLUMN_MAP["neutral_prob"]].to_numpy(dtype=float)
        prepared.loc[idx, "hmm_bear_prob"] = true_rows[META_HMM_COLUMN_MAP["bear_prob"]].to_numpy(dtype=float)
        prepared.loc[idx, "hmm_confidence"] = true_rows[META_HMM_COLUMN_MAP["confidence"]].to_numpy(dtype=float)
        prepared.loc[idx, "hmm_policy_code"] = true_rows["_policy"].map(HMM_POLICY_CODE_MAP).to_numpy(dtype=float)
        prepared.loc[idx, "hmm_regime_age_hours"] = true_rows["_regime_age_hours"].to_numpy(dtype=float)

    prepared = prepared.sort_values("_hmm_source_ts", kind="mergesort").reset_index(drop=True)
    diffs = prepared["_hmm_source_ts"].diff().dropna() / pd.Timedelta(hours=1)
    cadence = None
    if not diffs.empty:
        modes = diffs.mode(dropna=True)
        cadence = None if modes.empty else float(modes.iloc[0])

    diagnostics = {
        "path": str(path.resolve()),
        "schema": schema,
        "rows_after_timestamp_filter": before_dedup,
        "deduplicated_count": before_dedup - int(len(prepared)),
        "invalid_timestamp_count": invalid_ts,
        "source_start": prepared["_hmm_source_ts"].min(),
        "source_end": prepared["_hmm_source_ts"].max(),
        "hmm_ok_true_count": int(prepared["_hmm_ok_matched"].sum()),
        "hmm_ok_false_count": int((~prepared["_hmm_ok_matched"]).sum()),
        "cadence_mode_hours": cadence,
    }
    return prepared, diagnostics


def _feature_usable(series: pd.Series, feature: str) -> pd.Series:
    usable = series.notna()
    if feature == "hmm_regime":
        usable &= series.astype("string").ne("missing")
    return usable


def coverage_stats(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    """Return separate source-match and per-feature usability coverage."""

    features = _SOURCE_FEATURES[source]
    match_column = f"_{source}_matched"
    if match_column not in frame.columns:
        raise ValueError(f"Coverage source {source!r} has no match column {match_column!r}")
    total = int(len(frame))
    matched = frame[match_column].fillna(False).astype(bool)
    feature_coverage: dict[str, Any] = {}
    all_usable = pd.Series(True, index=frame.index, dtype=bool)
    for feature in features:
        if feature not in frame.columns:
            raise ValueError(f"Coverage source {source!r} is missing feature {feature!r}")
        usable = _feature_usable(frame[feature], feature)
        all_usable &= usable
        count = int(usable.sum())
        feature_coverage[feature] = {
            "usable_count": count,
            "usable_rate": None if total == 0 else float(count / total),
        }
    match_count = int(matched.sum())
    all_count = int(all_usable.sum())
    return {
        "event_count": total,
        "match_count": match_count,
        "match_rate": None if total == 0 else float(match_count / total),
        "all_features_usable_count": all_count,
        "all_features_usable_rate": None if total == 0 else float(all_count / total),
        "features": feature_coverage,
    }


def build_coverage_report(events: pd.DataFrame, include_hmm: bool) -> dict[str, Any]:
    sources = ["dataset", "funding", "vpin"] + (["hmm"] if include_hmm else [])
    report: dict[str, Any] = {}
    for source in sources:
        by_symbol = {
            symbol: coverage_stats(group, source)
            for symbol, group in events.groupby("symbol", sort=True)
        }
        by_fold = {
            str(int(fold)): coverage_stats(group, source)
            for fold, group in events.groupby("fold", sort=True)
        }
        by_symbol_and_fold: dict[str, Any] = {}
        for symbol, symbol_frame in events.groupby("symbol", sort=True):
            by_symbol_and_fold[symbol] = {
                str(int(fold)): coverage_stats(group, source)
                for fold, group in symbol_frame.groupby("fold", sort=True)
            }
        report[source] = {
            "overall": coverage_stats(events, source),
            "by_symbol": by_symbol,
            "by_fold": by_fold,
            "by_symbol_and_fold": by_symbol_and_fold,
        }
    return report


def _guard_feature_whitelist(X: pd.DataFrame, expected: Sequence[str], config: Mapping[str, Any]) -> None:
    if list(X.columns) != list(expected):
        raise ValueError(f"Meta X must exactly match the whitelist; expected={list(expected)}, got={list(X.columns)}")

    direct_forbidden = {
        "y",
        "meta_y",
        "fold",
        "tb_return",
        "decision_ts",
        "fwd_log_return",
        "close",
    }
    forbidden_present = sorted(column for column in X.columns if column in direct_forbidden or column.startswith("tb_"))
    if forbidden_present:
        raise ValueError(f"Forbidden leakage columns in meta X: {forbidden_present}")

    allowed_primary_overlap = {"volatility", "log_range", "hour_of_day", "day_of_week"}
    primary_features = set(config.get("PRIMARY_FEATURE_COLUMNS", []))
    bad_primary_overlap = sorted((set(X.columns) & primary_features) - allowed_primary_overlap)
    if bad_primary_overlap:
        raise ValueError(f"Primary-model features are forbidden in meta X: {bad_primary_overlap}")

    forbidden_hmm_source = {
        "timestamp",
        "backend",
        "timeframe",
        "fwd_log_return",
        "close",
        "hmm_reason",
        "hmm_state_prob_max",
        "hmm_state_prob_min",
        "hmm_state_prob_margin",
        "hmm_regime_prob_margin",
        "hmm_state_prob_entropy",
        "hmm_effective_emission_backend",
        "hmm_mapping_method",
        "fit_seconds",
    }
    leaked_hmm = sorted(set(X.columns) & forbidden_hmm_source)
    if leaked_hmm:
        raise ValueError(f"Forbidden raw HMM columns in meta X: {leaked_hmm}")


def _critical_finite_guard(events: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    for column in columns:
        if column not in events.columns:
            raise ValueError(f"{context} is missing critical column {column!r}")
        values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{context} critical column {column!r} contains NaN/inf")


def build_meta_dataset(
    symbols: Sequence[str],
    dataset_dir: str | os.PathLike[str],
    external_dir: str | os.PathLike[str],
    timeframe: str = "4h",
    hmm_dir: str | os.PathLike[str] | None = None,
    *,
    primary_dir: str | os.PathLike[str] | None = None,
    is_real_data_run: bool | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable v48 event table and exact CatBoost feature matrix."""

    if str(timeframe) != "4h":
        raise ValueError(f"v48 supports timeframe='4h' only; got {timeframe!r}")
    cfg = _project_config(config)
    cost, _ = derive_cost_and_embargo(cfg)
    ordered_symbols = normalize_symbols(symbols)
    primary_path = CANONICAL_PRIMARY_DIR if primary_dir is None else _resolved(primary_dir)
    dataset_path = _resolved(dataset_dir)
    external_path = _resolved(external_dir)
    hmm_path = None if hmm_dir is None else _resolved(hmm_dir)

    computed_real = bool(
        dataset_path == CANONICAL_DATASET_DIR
        and external_path == CANONICAL_EXTERNAL_DIR
        and hmm_path == CANONICAL_HMM_DIR
        and ordered_symbols == list(CANONICAL_SYMBOLS)
        and str(timeframe) == "4h"
    )
    real_run = computed_real if is_real_data_run is None else bool(is_real_data_run)
    if real_run and hmm_path is None:
        raise ValueError("Real-data meta dataset assembly requires hmm_dir")

    input_files = preflight_input_files(
        ordered_symbols,
        primary_path,
        dataset_path,
        external_path,
        hmm_path,
    )

    parts: list[pd.DataFrame] = []
    source_diagnostics: dict[str, Any] = {}
    input_schemas: dict[str, dict[str, list[str]]] = {
        "primary_oof": {},
        "dataset": {},
        "funding": {},
        "vpin": {},
        "hmm": {},
    }
    weight_fallback_count = 0
    fired_counts: dict[str, Any] = {}
    source_inf_to_nan_count: dict[str, int] = {}

    for symbol in ordered_symbols:
        oof_path = primary_path / f"primary_oof_{symbol}_4h.csv"
        dataset_file = dataset_path / f"{symbol}_4h.csv"
        funding_path = external_path / f"funding_{symbol}.csv"
        vpin_path = external_path / f"vpin_{symbol}_1h.csv"

        events, oof_diag, symbol_weight_fallback = _load_primary_oof(oof_path, symbol, real_run)
        weight_fallback_count += symbol_weight_fallback
        fired_counts[symbol] = {
            "total": oof_diag["fired_count"],
            "by_fold": oof_diag["fired_fold_counts"],
        }
        input_schemas["primary_oof"][symbol] = oof_diag["schema"]

        dataset_slice, dataset_diag = _load_dataset_slice(dataset_file)
        events, dataset_join_diag = _join_dataset_events(events, dataset_slice, dataset_file)
        events["decision_ts"] = events["OpenTime"] + pd.Timedelta(hours=4)
        events["hour_of_day"] = events["decision_ts"].dt.hour.astype(int)
        events["day_of_week"] = events["decision_ts"].dt.dayofweek.astype(int)
        events["meta_y"] = ((events["tb_return"] - cost) > 0.0).astype(int)
        input_schemas["dataset"][symbol] = dataset_diag["schema"]

        funding, funding_diag = _prepare_numeric_source(
            funding_path,
            "funding",
            "settlement_time",
            ["funding_z", "funding_extreme_pos"],
        )
        for column, count in funding_diag["inf_to_nan_count_by_feature"].items():
            source_inf_to_nan_count[column] = source_inf_to_nan_count.get(column, 0) + int(count)
        events, funding_join_diag = _merge_asof_source(
            events,
            funding,
            "funding",
            ["funding_z", "funding_extreme_pos"],
            pd.Timedelta(hours=8, minutes=1),
        )
        input_schemas["funding"][symbol] = funding_diag["schema"]

        vpin, vpin_diag = _prepare_numeric_source(
            vpin_path,
            "vpin",
            "timestamp",
            ["vpin", "vpin_z"],
        )
        for column, count in vpin_diag["inf_to_nan_count_by_feature"].items():
            source_inf_to_nan_count[column] = source_inf_to_nan_count.get(column, 0) + int(count)
        events, vpin_join_diag = _merge_asof_source(
            events,
            vpin,
            "vpin",
            ["vpin", "vpin_z"],
            pd.Timedelta(hours=2),
        )
        input_schemas["vpin"][symbol] = vpin_diag["schema"]

        hmm_diag: dict[str, Any] | None = None
        hmm_join_diag: dict[str, Any] | None = None
        if hmm_path is not None:
            hmm_file = hmm_path / f"hmm_walkforward_{symbol}_1h_pomegranate_normal.csv"
            hmm, hmm_diag = _prepare_hmm_source(hmm_file)
            events, hmm_join_diag = _merge_asof_source(
                events,
                hmm,
                "hmm",
                META_HMM_FEATURE_COLUMNS,
                pd.Timedelta(hours=6),
            )
            events["hmm_regime"] = events["hmm_regime"].astype("string").fillna("missing").astype(str)
            input_schemas["hmm"][symbol] = hmm_diag["schema"]

        source_diagnostics[symbol] = {
            "primary_oof": oof_diag,
            "dataset": {**dataset_diag, **dataset_join_diag},
            "funding": {**funding_diag, **funding_join_diag},
            "vpin": {**vpin_diag, **vpin_join_diag},
            "hmm": None if hmm_diag is None else {**hmm_diag, **(hmm_join_diag or {})},
        }
        parts.append(events)

    if not parts:
        raise ValueError("No fired primary events were assembled")
    pooled = pd.concat(parts, axis=0, ignore_index=True)
    pooled = pooled.sort_values(["decision_ts", "symbol"], kind="mergesort").reset_index(drop=True)
    pooled["_row_id"] = np.arange(len(pooled), dtype=int)
    pooled["_dataset_matched"] = pooled["_dataset_matched"].astype(bool)

    feature_columns = list(META_FEATURE_COLUMNS)
    categorical_features = ["symbol"]
    if hmm_path is not None:
        feature_columns.extend(META_HMM_FEATURE_COLUMNS)
        categorical_features.append("hmm_regime")

    pooled["symbol"] = pooled["symbol"].astype(str)
    inf_to_nan_count: dict[str, int] = dict(source_inf_to_nan_count)
    for column in feature_columns:
        if column in categorical_features:
            continue
        numeric = pd.to_numeric(pooled[column], errors="coerce")
        inf_mask = np.isinf(numeric.to_numpy(dtype=float))
        inf_to_nan_count[column] = inf_to_nan_count.get(column, 0) + int(inf_mask.sum())
        if inf_mask.any():
            numeric.loc[inf_mask] = np.nan
        pooled[column] = numeric

    _critical_finite_guard(
        pooled,
        ["meta_y", "y", "p_primary", "tb_return", "fold", "tb_uniqueness"],
        "assembled meta events",
    )
    X = pooled.loc[:, feature_columns].copy()
    _guard_feature_whitelist(X, feature_columns, cfg)

    coverage = build_coverage_report(pooled, include_hmm=hmm_path is not None)
    fallback_counts = {
        "tb_uniqueness_weight_fallback_count": int(weight_fallback_count),
        "tb_exit_index_fallback_count": 0,
    }
    if real_run and any(fallback_counts.values()):
        raise ValueError(f"Real-data assembly forbids fallbacks: {fallback_counts}")

    metadata = {
        "symbols": ordered_symbols,
        "timeframe": "4h",
        "is_real_data_run": real_run,
        "rows": int(len(pooled)),
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "input_files": input_files,
        "input_schemas": input_schemas,
        "source_diagnostics": source_diagnostics,
        "coverage": coverage,
        "fired_event_counts": fired_counts,
        "fired_event_count_by_fold": {
            str(int(fold)): int(count)
            for fold, count in pooled["fold"].value_counts().sort_index().items()
        },
        "fallback_counts": fallback_counts,
        "inf_to_nan_count_by_feature": inf_to_nan_count,
        "cost_round_trip": cost,
        "meta_base_threshold": META_BASE_THRESHOLD,
    }
    return {
        "events": pooled,
        "X": X,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "metadata": metadata,
    }


def weighted_scale_pos_weight(y: pd.Series, weights: pd.Series) -> tuple[float, float, float]:
    """Return negative/positive weighted mass; CatBoost uses their ratio."""

    y_values = pd.Series(y).astype(int)
    weight_values = pd.Series(np.asarray(weights, dtype=float), index=y_values.index)
    if not np.isfinite(weight_values.to_numpy(dtype=float)).all():
        raise ValueError("Training weights must be finite")
    positive = float(weight_values.loc[y_values == 1].sum())
    negative = float(weight_values.loc[y_values == 0].sum())
    if positive <= 0.0 or negative <= 0.0:
        raise ValueError(
            f"Both weighted classes must be positive; weighted_positive={positive}, weighted_negative={negative}"
        )
    return float(negative / positive), positive, negative


def purge_training_events(
    train_events: pd.DataFrame,
    eval_start: Any,
    *,
    is_real_data_run: bool,
    embargo: pd.Timedelta = EMBARGO,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the global v48 purge; synthetic data alone may infer missing exits."""

    if train_events.empty:
        raise ValueError("Cannot purge an empty training frame")
    eval_start_ts = pd.Timestamp(eval_start)
    if eval_start_ts.tzinfo is None:
        eval_start_ts = eval_start_ts.tz_localize("UTC")
    else:
        eval_start_ts = eval_start_ts.tz_convert("UTC")
    if embargo != pd.Timedelta(hours=96):
        raise ValueError(f"v48 purge requires a 96h embargo, got {embargo}")

    if "tb_exit_index" in train_events.columns:
        exits = pd.to_datetime(
            train_events["tb_exit_index"],
            utc=True,
            format="mixed",
            errors="coerce",
        )
    else:
        exits = pd.Series(pd.NaT, index=train_events.index, dtype="datetime64[ns, UTC]")
    missing = exits.isna()
    fallback_count = int(missing.sum())
    if fallback_count:
        if is_real_data_run:
            raise ValueError(
                f"Real-data purge forbids tb_exit_index fallback; missing_count={fallback_count}"
            )
        if "decision_ts" not in train_events.columns:
            raise ValueError("Synthetic tb_exit_index fallback requires decision_ts")
        fallback = pd.to_datetime(
            train_events["decision_ts"], utc=True, format="mixed", errors="coerce"
        ) + pd.Timedelta(hours=96)
        if fallback.loc[missing].isna().any():
            raise ValueError("Synthetic tb_exit_index fallback has invalid decision_ts")
        exits = exits.copy()
        exits.loc[missing] = fallback.loc[missing]

    purge_boundary = eval_start_ts - embargo
    allowed = (exits + pd.Timedelta(hours=4)) < purge_boundary
    kept = train_events.loc[allowed.to_numpy(dtype=bool)].copy()
    metadata = {
        "purge_removed_count": int((~allowed).sum()),
        "tb_exit_index_fallback_count": fallback_count,
        "eval_start_k": eval_start_ts,
        "purge_boundary_ts": purge_boundary,
        "purge_rule": "tb_exit_index + 4h < eval_start_k - 96h",
        "equivalent_primary_rule": "tb_exit_index < first_test_OpenTime - 96h",
    }
    return kept, metadata


def _training_health(frame: pd.DataFrame) -> tuple[float, float, float]:
    if len(frame) < META_MIN_TRAIN_EVENTS:
        raise ValueError(
            f"Meta training requires at least {META_MIN_TRAIN_EVENTS} events; got {len(frame)}"
        )
    _critical_finite_guard(frame, ["meta_y", "tb_uniqueness"], "meta training frame")
    classes = set(frame["meta_y"].astype(int).unique().tolist())
    if classes != {0, 1}:
        raise ValueError(f"Meta training requires both classes; got {sorted(classes)}")
    return weighted_scale_pos_weight(frame["meta_y"], frame["tb_uniqueness"])


def _model_frame(frame: pd.DataFrame, feature_columns: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    X = frame.loc[:, list(feature_columns)].copy()
    for column in categorical:
        if column not in X.columns:
            raise ValueError(f"Missing categorical feature {column!r}")
        if X[column].isna().any():
            raise ValueError(f"CatBoost categorical feature {column!r} cannot contain NaN")
        X[column] = X[column].astype(str)
    return X


def fit_meta_catboost(
    train_events: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    """Fit one deterministic weighted CatBoost model on a validated train frame."""

    scale_pos_weight, positive, negative = _training_health(train_events)
    params = dict(META_CATBOOST_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    if params.get("allow_writing_files") is not False:
        raise ValueError("Every v48 CatBoost fit requires allow_writing_files=False")
    model = CatBoostClassifier(**params)
    X = _model_frame(train_events, feature_columns, categorical_features)
    model.fit(
        X,
        train_events["meta_y"].astype(int),
        sample_weight=train_events["tb_uniqueness"].astype(float),
        cat_features=list(categorical_features),
    )
    metadata = {
        "train_event_count": int(len(train_events)),
        "weighted_positive_sum": positive,
        "weighted_negative_sum": negative,
        "scale_pos_weight": scale_pos_weight,
        "train_date_range": [train_events["decision_ts"].min(), train_events["decision_ts"].max()],
        "feature_columns": list(feature_columns),
        "categorical_features": list(categorical_features),
        "feature_importance": {
            feature: float(value)
            for feature, value in zip(feature_columns, model.get_feature_importance(), strict=True)
        },
    }
    return model, metadata


def _predict_meta(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(_model_frame(frame, feature_columns, categorical_features))[:, 1],
        dtype=float,
    )
    if probabilities.shape != (len(frame),) or not np.isfinite(probabilities).all():
        raise ValueError("CatBoost produced invalid p_meta values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("CatBoost p_meta must remain in [0, 1]")
    return probabilities


def auc_result(y: pd.Series, score: pd.Series | np.ndarray) -> dict[str, Any]:
    y_values = pd.Series(y).astype(int).to_numpy()
    score_values = np.asarray(score, dtype=float)
    if len(y_values) == 0:
        return {"value": None, "reason": "empty"}
    if not np.isfinite(score_values).all():
        raise ValueError("AUC score input contains NaN/inf")
    if len(np.unique(y_values)) < 2:
        return {"value": None, "reason": "single_class"}
    return {"value": float(roc_auc_score(y_values, score_values)), "reason": None}


def _candidate_row(
    history: pd.DataFrame,
    score_column: str,
    tau: float,
    *,
    diagnostic_only: bool,
    abstain: bool,
) -> dict[str, Any]:
    if abstain:
        return {
            "tau": ABSTAIN_THRESHOLD,
            "n_trades": 0,
            "sum_net": 0.0,
            "mean_net_bps": None,
            "effective_n": 0.0,
            "eligible": True,
            "diagnostic_only": False,
            "abstain": True,
        }
    selected = history[score_column].to_numpy(dtype=float) >= float(tau)
    count = int(selected.sum())
    net = history["_net"].to_numpy(dtype=float)[selected]
    weights = history["tb_uniqueness"].to_numpy(dtype=float)[selected]
    return {
        "tau": float(tau),
        "n_trades": count,
        "sum_net": float(net.sum()) if count else 0.0,
        "mean_net_bps": None if count == 0 else float(net.mean() * 10000.0),
        "effective_n": float(weights.sum()) if count else 0.0,
        "eligible": bool(count >= META_MIN_TRADES_FOR_TAU),
        "diagnostic_only": bool(diagnostic_only),
        "abstain": False,
    }


def threshold_candidate_table(
    history: pd.DataFrame,
    score_column: str,
    selection_thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    _require_columns(history, [score_column, "_net", "tb_uniqueness"], "threshold history")
    rows = [
        _candidate_row(
            history,
            score_column,
            tau,
            diagnostic_only=False,
            abstain=False,
        )
        for tau in selection_thresholds
    ]
    rows.extend(
        _candidate_row(
            history,
            score_column,
            tau,
            diagnostic_only=True,
            abstain=False,
        )
        for tau in DIAGNOSTIC_THRESHOLDS
    )
    rows.append(
        _candidate_row(
            history,
            score_column,
            ABSTAIN_THRESHOLD,
            diagnostic_only=False,
            abstain=True,
        )
    )
    return rows


def _best_by_sum(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select from an empty candidate list")
    max_sum = max(float(row["sum_net"]) for row in rows)
    tied = [row for row in rows if abs(float(row["sum_net"]) - max_sum) <= TAU_TIE_TOLERANCE]
    tied.sort(key=lambda row: (-float(row["tau"]), int(row["n_trades"])))
    return tied[0]


def select_threshold(
    history: pd.DataFrame,
    score_column: str,
    selection_thresholds: Sequence[float],
    *,
    first_evaluated_fold: bool,
    forced: bool,
) -> dict[str, Any]:
    """Select tau from prior folds only, with an independent forced view."""

    table = threshold_candidate_table(history, score_column, selection_thresholds)
    if first_evaluated_fold:
        return {
            "selected_tau": 0.55,
            "selection_reason": "first_evaluated_fold_fixed",
            "forced": bool(forced),
            "candidates": table,
        }

    eligible_real = [
        row
        for row in table
        if row["eligible"] and not row["diagnostic_only"] and not row["abstain"]
    ]
    if forced:
        if not eligible_real:
            return {
                "selected_tau": 0.55,
                "selection_reason": "forced_fallback",
                "forced": True,
                "candidates": table,
            }
        selected = _best_by_sum(eligible_real)
        return {
            "selected_tau": float(selected["tau"]),
            "selection_reason": "forced_best_eligible",
            "forced": True,
            "candidates": table,
        }

    if not eligible_real:
        return {
            "selected_tau": ABSTAIN_THRESHOLD,
            "selection_reason": "no_eligible_real_threshold",
            "forced": False,
            "candidates": table,
        }
    abstain_row = next(row for row in table if row["abstain"])
    selected = _best_by_sum([*eligible_real, abstain_row])
    if selected["abstain"]:
        reason = "abstain_non_positive_history"
    else:
        reason = "best_eligible_positive_history"
    return {
        "selected_tau": float(selected["tau"]),
        "selection_reason": reason,
        "forced": False,
        "candidates": table,
    }


def trade_metrics(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(frame),):
        raise ValueError("Trade mask length does not match frame")
    count = int(selected.sum())
    if count == 0:
        return {
            "n_trades": 0,
            "pt_hit_rate": None,
            "net_win_rate": None,
            "mean_net_bps": None,
            "sum_net_event_units": 0.0,
            "effective_n": 0.0,
        }
    y = frame["y"].to_numpy(dtype=int)[selected]
    net = frame["_net"].to_numpy(dtype=float)[selected]
    weights = frame["tb_uniqueness"].to_numpy(dtype=float)[selected]
    return {
        "n_trades": count,
        "pt_hit_rate": float(np.mean(y == 1)),
        "net_win_rate": float(np.mean(net > 0.0)),
        "mean_net_bps": float(np.mean(net) * 10000.0),
        "sum_net_event_units": float(np.sum(net)),
        "effective_n": float(np.sum(weights)),
    }


def _top_n_meta_mask(frame: pd.DataFrame, n: int) -> pd.Series:
    count = max(0, min(int(n), int(len(frame))))
    mask = pd.Series(False, index=frame.index, dtype=bool)
    if count == 0:
        return mask
    ordered = frame.sort_values(
        ["p_meta", "decision_ts", "symbol"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    mask.loc[ordered.head(count).index] = True
    return mask


def _quantiles(values: pd.Series) -> dict[str, float]:
    q = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": float(q.loc[0.0]),
        "q25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "q75": float(q.loc[0.75]),
        "max": float(q.loc[1.0]),
    }


def _validate_baseline_anchor(
    events: pd.DataFrame,
    fold_reports: Mapping[str, Any],
    threshold_selection: Mapping[str, Any],
    is_real_data_run: bool,
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    passed = True
    for fold, expected in EXPECTED_BASELINE_ANCHOR.items():
        report = fold_reports[str(fold)]
        metrics = report["realized"]["baseline"]
        forced_metrics = report["realized"]["baseline_forced"]
        actual = {
            "tau": float(report["taus"]["baseline"]),
            "n_trades": int(metrics["n_trades"]),
            "sum_net": float(metrics["sum_net_event_units"]),
        }
        forced_actual = {
            "tau": float(report["taus"]["baseline_forced"]),
            "n_trades": int(forced_metrics["n_trades"]),
            "sum_net": float(forced_metrics["sum_net_event_units"]),
        }
        fold_passed = bool(
            actual["tau"] == expected["tau"]
            and actual["n_trades"] == expected["n_trades"]
            and abs(actual["sum_net"] - expected["sum_net"]) <= 5e-4
            and forced_actual["tau"] == expected["tau"]
            and forced_actual["n_trades"] == expected["n_trades"]
            and abs(forced_actual["sum_net"] - expected["sum_net"]) <= 5e-4
        )
        passed &= fold_passed
        comparisons[str(fold)] = {
            "expected": dict(expected),
            "actual": actual,
            "forced_actual": forced_actual,
            "passed": fold_passed,
        }

    result = {"passed": bool(passed), "tolerance_sum_net": 5e-4, "folds": comparisons}
    if is_real_data_run and not passed:
        payload = {
            "message": "Real-data baseline anchor mismatch; stop for human adjudication",
            "anchor": result,
            "fired_event_count_by_fold": {
                str(int(fold)): int(count)
                for fold, count in events["fold"].value_counts().sort_index().items()
            },
            "baseline_threshold_tables": threshold_selection["baseline"],
            "baseline_forced_threshold_tables": threshold_selection["baseline_forced"],
        }
        raise BaselineAnchorError(json.dumps(_json_native(payload), ensure_ascii=False, allow_nan=False))
    return result


def run_meta_walk_forward(
    events: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_features: Sequence[str],
    *,
    is_real_data_run: bool,
    include_hmm: bool,
    cost: float = COST_ROUND_TRIP,
    embargo: pd.Timedelta = EMBARGO,
) -> dict[str, Any]:
    """Train folds 2..5 and select every threshold only from prior evaluated folds."""

    work = events.copy().sort_values(["decision_ts", "symbol"], kind="mergesort").reset_index(drop=True)
    work["_net"] = work["tb_return"].astype(float) - float(cost)
    work["p_meta"] = np.nan
    for column in [
        "tau_meta_used",
        "tau_baseline_used",
        "tau_meta_forced_used",
        "tau_baseline_forced_used",
    ]:
        work[column] = np.nan
    work["traded_meta"] = False
    work["traded_baseline"] = False
    work["_traded_meta_forced"] = False
    work["_traded_baseline_forced"] = False
    work["_traded_meta_matched_n"] = False
    work["_traded_baseline_matched_reference"] = False
    work["meta_eval_status"] = "not_evaluated_no_prior_fold"

    fold_reports: dict[str, Any] = {}
    threshold_selection: dict[str, dict[str, Any]] = {
        "meta": {},
        "baseline": {},
        "meta_forced": {},
        "baseline_forced": {},
    }
    total_exit_fallback = 0

    for fold in range(2, 6):
        eval_mask = work["fold"].eq(fold)
        eval_frame = work.loc[eval_mask].copy()
        if eval_frame.empty:
            raise ValueError(f"Meta evaluation fold {fold} is empty")
        eval_start = eval_frame["decision_ts"].min()
        train_candidates = work.loc[work["fold"] < fold].copy()
        train_frame, purge_meta = purge_training_events(
            train_candidates,
            eval_start,
            is_real_data_run=is_real_data_run,
            embargo=embargo,
        )
        total_exit_fallback += int(purge_meta["tb_exit_index_fallback_count"])
        if is_real_data_run and purge_meta["tb_exit_index_fallback_count"] != 0:
            raise ValueError(f"Real-data fold {fold} used an event-end fallback")

        model, fit_meta = fit_meta_catboost(train_frame, feature_columns, categorical_features)
        train_prob = _predict_meta(model, train_frame, feature_columns, categorical_features)
        eval_prob = _predict_meta(model, eval_frame, feature_columns, categorical_features)
        work.loc[eval_mask, "p_meta"] = eval_prob
        work.loc[eval_mask, "meta_eval_status"] = "evaluated"

        history = work.loc[work["fold"].between(2, fold - 1, inclusive="both")].copy()
        first = fold == 2
        meta_selection = select_threshold(
            history,
            "p_meta",
            META_THRESHOLDS,
            first_evaluated_fold=first,
            forced=False,
        )
        baseline_selection = select_threshold(
            history,
            "p_primary",
            BASELINE_THRESHOLDS,
            first_evaluated_fold=first,
            forced=False,
        )
        meta_forced = select_threshold(
            history,
            "p_meta",
            META_THRESHOLDS,
            first_evaluated_fold=first,
            forced=True,
        )
        baseline_forced = select_threshold(
            history,
            "p_primary",
            BASELINE_THRESHOLDS,
            first_evaluated_fold=first,
            forced=True,
        )
        selections = {
            "meta": meta_selection,
            "baseline": baseline_selection,
            "meta_forced": meta_forced,
            "baseline_forced": baseline_forced,
        }
        for arm, selection in selections.items():
            threshold_selection[arm][str(fold)] = selection

        tau_meta = float(meta_selection["selected_tau"])
        tau_baseline = float(baseline_selection["selected_tau"])
        tau_meta_forced = float(meta_forced["selected_tau"])
        tau_baseline_forced = float(baseline_forced["selected_tau"])
        work.loc[eval_mask, "tau_meta_used"] = tau_meta
        work.loc[eval_mask, "tau_baseline_used"] = tau_baseline
        work.loc[eval_mask, "tau_meta_forced_used"] = tau_meta_forced
        work.loc[eval_mask, "tau_baseline_forced_used"] = tau_baseline_forced

        current = work.loc[eval_mask]
        meta_mask = current["p_meta"] >= tau_meta
        baseline_mask = current["p_primary"] >= tau_baseline
        meta_forced_mask = current["p_meta"] >= tau_meta_forced
        baseline_forced_mask = current["p_primary"] >= tau_baseline_forced
        work.loc[current.index, "traded_meta"] = meta_mask.to_numpy(dtype=bool)
        work.loc[current.index, "traded_baseline"] = baseline_mask.to_numpy(dtype=bool)
        work.loc[current.index, "_traded_meta_forced"] = meta_forced_mask.to_numpy(dtype=bool)
        work.loc[current.index, "_traded_baseline_forced"] = baseline_forced_mask.to_numpy(dtype=bool)

        baseline_abstained = math.isclose(tau_baseline, ABSTAIN_THRESHOLD, abs_tol=1e-12)
        reference_mask = baseline_forced_mask if baseline_abstained else baseline_mask
        reference_source = "baseline_forced" if baseline_abstained else "baseline"
        target_n = int(reference_mask.sum())
        matched_mask = _top_n_meta_mask(current, target_n)
        work.loc[current.index, "_traded_meta_matched_n"] = matched_mask.to_numpy(dtype=bool)
        work.loc[current.index, "_traded_baseline_matched_reference"] = reference_mask.to_numpy(dtype=bool)

        train_auc_meta = auc_result(train_frame["meta_y"], train_prob)
        train_auc_primary = auc_result(train_frame["meta_y"], train_frame["p_primary"])
        eval_auc_meta = auc_result(eval_frame["meta_y"], eval_prob)
        eval_auc_primary = auc_result(eval_frame["meta_y"], eval_frame["p_primary"])
        fit_meta.update(purge_meta)
        fit_meta["train_auc_meta"] = train_auc_meta
        fit_meta["train_auc_primary"] = train_auc_primary
        fit_meta["eval_auc_meta"] = eval_auc_meta
        fit_meta["eval_auc_primary"] = eval_auc_primary
        fit_meta["hmm_coverage_train"] = coverage_stats(train_frame, "hmm") if include_hmm else None
        fit_meta["hmm_coverage_eval"] = coverage_stats(eval_frame, "hmm") if include_hmm else None

        fold_reports[str(fold)] = {
            "fold": fold,
            "model": fit_meta,
            "p_meta_quantiles": _quantiles(pd.Series(eval_prob)),
            "taus": {
                "meta": tau_meta,
                "baseline": tau_baseline,
                "meta_forced": tau_meta_forced,
                "baseline_forced": tau_baseline_forced,
            },
            "matched_n": {
                "n": target_n,
                "baseline_reference_source": reference_source,
                "used_forced_baseline_fallback": baseline_abstained,
                "diagnostic_only_not_live_executable": True,
            },
            "realized": {
                "meta": trade_metrics(current, meta_mask),
                "baseline": trade_metrics(current, baseline_mask),
                "meta_forced": trade_metrics(current, meta_forced_mask),
                "baseline_forced": trade_metrics(current, baseline_forced_mask),
                "meta_matched_n": trade_metrics(current, matched_mask),
                "baseline_matched_reference": trade_metrics(current, reference_mask),
            },
        }

    evaluated = work["fold"].between(2, 5, inclusive="both")
    if work.loc[evaluated, "p_meta"].isna().any() or not np.isfinite(
        work.loc[evaluated, "p_meta"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Every evaluated fold must have finite p_meta")
    if work.loc[work["fold"].eq(1), "p_meta"].notna().any():
        raise ValueError("Fold 1 p_meta must remain NaN")
    if is_real_data_run and total_exit_fallback != 0:
        raise ValueError(f"Real-data walk-forward used {total_exit_fallback} event-end fallbacks")

    anchor = _validate_baseline_anchor(
        work,
        fold_reports,
        threshold_selection,
        is_real_data_run,
    )
    all_history = work.loc[evaluated].copy()
    tau_final = select_threshold(
        all_history,
        "p_meta",
        META_THRESHOLDS,
        first_evaluated_fold=False,
        forced=False,
    )
    tau_final["diagnostic_only_not_for_live"] = True

    return {
        "events": work,
        "folds": fold_reports,
        "threshold_selection": threshold_selection,
        "baseline_anchor": anchor,
        "tau_final_raw_diagnostic": tau_final,
        "tb_exit_index_fallback_count": total_exit_fallback,
    }


_ARM_COLUMNS = {
    "meta": "traded_meta",
    "baseline": "traded_baseline",
    "meta_forced": "_traded_meta_forced",
    "baseline_forced": "_traded_baseline_forced",
    "meta_matched_n": "_traded_meta_matched_n",
    "baseline_matched_reference": "_traded_baseline_matched_reference",
}


def _view_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "pooled_folds_2_5": events["fold"].between(2, 5, inclusive="both"),
        "folds_2_4": events["fold"].between(2, 4, inclusive="both"),
        "fold_5": events["fold"].eq(5),
    }


def build_arm_reports(events: pd.DataFrame) -> dict[str, Any]:
    views = _view_masks(events)
    arm_metrics: dict[str, Any] = {}
    for view_name, view_mask in views.items():
        frame = events.loc[view_mask].copy()
        arm_metrics[view_name] = {
            arm: trade_metrics(frame, frame[column])
            for arm, column in _ARM_COLUMNS.items()
        }

    paired: dict[str, Any] = {"main": {}, "forced": {}, "matched_n": {}}
    for view_name, metrics in arm_metrics.items():
        paired["main"][view_name] = {
            "meta": metrics["meta"],
            "baseline": metrics["baseline"],
        }
        paired["forced"][view_name] = {
            "meta_forced": metrics["meta_forced"],
            "baseline_forced": metrics["baseline_forced"],
        }
        paired["matched_n"][view_name] = {
            "meta_matched_n": metrics["meta_matched_n"],
            "baseline_matched_reference": metrics["baseline_matched_reference"],
            "diagnostic_only_not_live_executable": True,
        }

    evaluated = events.loc[events["fold"].between(2, 5, inclusive="both")]
    per_symbol: dict[str, Any] = {}
    for symbol, frame in evaluated.groupby("symbol", sort=True):
        per_symbol[symbol] = {
            arm: trade_metrics(frame, frame[column])
            for arm, column in _ARM_COLUMNS.items()
        }
    return {
        "arm_metrics_by_view": arm_metrics,
        "paired_tables": paired,
        "per_symbol": per_symbol,
    }


def concurrency_diagnostics(frame: pd.DataFrame, mask: pd.Series | np.ndarray) -> dict[str, Any]:
    selected = frame.loc[np.asarray(mask, dtype=bool)].copy()
    count = int(len(selected))
    if count == 0:
        return {
            "n_trades": 0,
            "max_concurrent_positions": 0,
            "btc_share": None,
            "multi_signal_decision_ts_count": 0,
        }

    starts = pd.to_datetime(selected["decision_ts"], utc=True, format="mixed", errors="coerce")
    ends = pd.to_datetime(selected["tb_exit_index"], utc=True, format="mixed", errors="coerce") + pd.Timedelta(hours=4)
    if starts.isna().any() or ends.isna().any():
        raise ValueError("Concurrency diagnostics require valid decision and exit timestamps")
    points: list[tuple[pd.Timestamp, int]] = []
    for start, end in zip(starts, ends, strict=True):
        points.append((pd.Timestamp(start), 1))
        points.append((pd.Timestamp(end), -1))
    points.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    maximum = 0
    for _, delta in points:
        active += delta
        maximum = max(maximum, active)

    decision_counts = selected.groupby("decision_ts", sort=True).size()
    return {
        "n_trades": count,
        "max_concurrent_positions": int(maximum),
        "btc_share": float(selected["symbol"].eq("BTCUSDT").mean()),
        "multi_signal_decision_ts_count": int((decision_counts > 1).sum()),
    }


def build_concurrency_report(events: pd.DataFrame) -> dict[str, Any]:
    evaluated = events.loc[events["fold"].between(2, 5, inclusive="both")].copy()
    return {
        arm: concurrency_diagnostics(evaluated, evaluated[column])
        for arm, column in _ARM_COLUMNS.items()
    }


def _reconciliation_table(frame: pd.DataFrame) -> dict[str, int]:
    y = frame["y"].astype(int)
    net_win = frame["_net"] > 0.0
    return {
        "y0_net_nonpositive": int(((y == 0) & ~net_win).sum()),
        "y0_net_positive": int(((y == 0) & net_win).sum()),
        "y1_net_nonpositive": int(((y == 1) & ~net_win).sum()),
        "y1_net_positive": int(((y == 1) & net_win).sum()),
        "total": int(len(frame)),
    }


def build_y_net_reconciliation(events: pd.DataFrame) -> dict[str, Any]:
    by_symbol_and_fold: dict[str, Any] = {}
    for symbol, symbol_frame in events.groupby("symbol", sort=True):
        by_symbol_and_fold[symbol] = {
            str(int(fold)): _reconciliation_table(fold_frame)
            for fold, fold_frame in symbol_frame.groupby("fold", sort=True)
        }
    return {
        "pooled": _reconciliation_table(events),
        "by_fold": {
            str(int(fold)): _reconciliation_table(frame)
            for fold, frame in events.groupby("fold", sort=True)
        },
        "by_symbol": {
            symbol: _reconciliation_table(frame)
            for symbol, frame in events.groupby("symbol", sort=True)
        },
        "by_symbol_and_fold": by_symbol_and_fold,
    }


def build_auc_report(events: pd.DataFrame, fold_reports: Mapping[str, Any]) -> dict[str, Any]:
    evaluated = events.loc[events["fold"].between(2, 5, inclusive="both")]
    return {
        "pooled_evaluated": {
            "meta": auc_result(evaluated["meta_y"], evaluated["p_meta"]),
            "primary": auc_result(evaluated["meta_y"], evaluated["p_primary"]),
        },
        "per_fold": {
            fold: {
                "train_meta": report["model"]["train_auc_meta"],
                "train_primary": report["model"]["train_auc_primary"],
                "eval_meta": report["model"]["eval_auc_meta"],
                "eval_primary": report["model"]["eval_auc_primary"],
            }
            for fold, report in fold_reports.items()
        },
    }


def _json_native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.timedelta64):
        return str(pd.Timedelta(value))
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_native(item) for item in list(value)]
    if pd.isna(value):
        return None
    return value


def _write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
    native = _json_native(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(native, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("r", encoding="utf-8") as handle:
        json.load(handle)


def _managed_artifacts(output_dir: Path) -> list[Path]:
    managed: dict[str, Path] = {}
    for name in ["meta_4h.cbm", "meta_training_summary_4h.json"]:
        path = output_dir / name
        if path.is_file():
            managed[path.name] = path
    for path in output_dir.glob("meta_oof_*_4h.csv"):
        if path.is_file():
            managed[path.name] = path
    return [managed[name] for name in sorted(managed)]


def _commit_staged_artifacts(
    temp_dir: Path,
    output_dir: Path,
    manifest: Sequence[str],
) -> None:
    """Commit summary last and restore every managed destination on exceptions."""

    backup_dir = temp_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    existing = _managed_artifacts(output_dir)
    backed_up: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for path in existing:
            backup = backup_dir / path.name
            os.replace(path, backup)
            backed_up.append((path, backup))
        for name in manifest:
            source = temp_dir / name
            destination = output_dir / name
            os.replace(source, destination)
            committed.append(destination)
    except Exception:
        for path in reversed(committed):
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass
        rollback_errors: list[str] = []
        for original, backup in reversed(backed_up):
            try:
                if backup.exists():
                    os.replace(backup, original)
            except Exception as exc:
                rollback_errors.append(f"{original}: {type(exc).__name__}: {exc}")
        if rollback_errors:
            raise RuntimeError(f"v48 rollback failed: {rollback_errors}")
        raise


def write_outputs_transactional(
    model: CatBoostClassifier,
    events: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | os.PathLike[str],
    symbols: Sequence[str],
) -> dict[str, Any]:
    """Stage all managed outputs, validate JSON, then commit summary last."""

    output_path = _resolved(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / f".meta_v48_tmp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=False, exist_ok=False)

    model_name = "meta_4h.cbm"
    summary_name = "meta_training_summary_4h.json"
    oof_names = [f"meta_oof_{symbol}_4h.csv" for symbol in symbols]
    manifest_without_summary = [model_name, *oof_names]
    full_manifest = [*manifest_without_summary, summary_name]
    existing_names = {path.name for path in _managed_artifacts(output_path)}
    removed_stale = sorted(existing_names - set(full_manifest))

    try:
        model.save_model(str(temp_dir / model_name))
        for symbol, name in zip(symbols, oof_names, strict=True):
            frame = events.loc[events["symbol"].eq(symbol), META_OOF_COLUMNS].copy()
            frame = frame.sort_values("OpenTime", kind="mergesort")
            if frame["OpenTime"].duplicated().any():
                raise ValueError(f"Meta OOF {symbol} has duplicate OpenTime")
            frame.to_csv(temp_dir / name, index=False)

        output_artifacts: dict[str, Any] = {}
        for name in manifest_without_summary:
            path = temp_dir / name
            output_artifacts[name] = {
                "path": str((output_path / name).resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        summary["output_artifacts"] = output_artifacts
        summary["removed_stale_artifacts"] = removed_stale
        _write_strict_json(temp_dir / summary_name, summary)
        _commit_staged_artifacts(temp_dir, output_path, full_manifest)
    except Exception as exc:
        rollback_failed = isinstance(exc, RuntimeError) and str(exc).startswith("v48 rollback failed:")
        if not rollback_failed:
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    shutil.rmtree(temp_dir)
    return {
        "output_dir": str(output_path),
        "model_path": str((output_path / model_name).resolve()),
        "summary_path": str((output_path / summary_name).resolve()),
        "oof_paths": [str((output_path / name).resolve()) for name in oof_names],
        "removed_stale_artifacts": removed_stale,
    }


def run_meta_training(
    *,
    symbols: Sequence[str] = CANONICAL_SYMBOLS,
    dataset_dir: str | os.PathLike[str] = CANONICAL_DATASET_DIR,
    external_dir: str | os.PathLike[str] = CANONICAL_EXTERNAL_DIR,
    hmm_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] = CANONICAL_OUTPUT_DIR,
    timeframe: str = "4h",
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute v48 offline training and transactionally publish its artifacts."""

    cfg = _project_config(config)
    cost, embargo = derive_cost_and_embargo(cfg)
    identity = resolve_run_identity(
        symbols,
        dataset_dir,
        external_dir,
        hmm_dir,
        output_dir,
        timeframe,
    )
    assembly = build_meta_dataset(
        identity["symbols"],
        identity["dataset_dir"],
        identity["external_dir"],
        timeframe=identity["timeframe"],
        hmm_dir=identity["hmm_dir"],
        primary_dir=identity["primary_dir"],
        is_real_data_run=identity["is_real_data_run"],
        config=cfg,
    )
    events = assembly["events"]
    walk_forward = run_meta_walk_forward(
        events,
        assembly["feature_columns"],
        assembly["categorical_features"],
        is_real_data_run=identity["is_real_data_run"],
        include_hmm=identity["hmm_dir"] is not None,
        cost=cost,
        embargo=embargo,
    )
    oof_events = walk_forward["events"]

    # The final model has no evaluation fold, so no purge is applied here.
    final_model, final_fit = fit_meta_catboost(
        oof_events,
        assembly["feature_columns"],
        assembly["categorical_features"],
    )
    final_train_prob = _predict_meta(
        final_model,
        oof_events,
        assembly["feature_columns"],
        assembly["categorical_features"],
    )
    final_no_eval_reason = "final_model_has_no_evaluation_fold"
    final_fit.update(
        {
            "purge_applied": False,
            "purge_reason": final_no_eval_reason,
            "purge_removed_count": 0,
            "tb_exit_index_fallback_count": 0,
            "eval_start_k": None,
            "eval_start_k_reason": final_no_eval_reason,
            "purge_boundary_ts": None,
            "purge_boundary_ts_reason": final_no_eval_reason,
            "train_auc_meta": auc_result(oof_events["meta_y"], final_train_prob),
            "train_auc_primary": auc_result(
                oof_events["meta_y"], oof_events["p_primary"]
            ),
            "eval_auc_meta": {"value": None, "reason": final_no_eval_reason},
            "eval_auc_primary": {"value": None, "reason": final_no_eval_reason},
            "hmm_coverage_train": coverage_stats(oof_events, "hmm")
            if identity["hmm_dir"] is not None
            else None,
            "hmm_coverage_eval": None,
            "hmm_coverage_eval_reason": final_no_eval_reason,
        }
    )

    fallback_counts = dict(assembly["metadata"]["fallback_counts"])
    fallback_counts["tb_exit_index_fallback_count"] = int(
        walk_forward["tb_exit_index_fallback_count"]
    )
    if identity["is_real_data_run"] and any(fallback_counts.values()):
        raise ValueError(f"Real-data v48 run forbids all fallbacks: {fallback_counts}")

    arm_reports = build_arm_reports(oof_events)
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc),
        "project": "Kosar",
        "version": "v48",
        "timeframe": "4h",
        "identity": identity,
        "symbols": identity["symbols"],
        "is_real_data_run": identity["is_real_data_run"],
        "is_canonical_output": identity["is_canonical_output"],
        "input_files": assembly["metadata"]["input_files"],
        "input_schemas": assembly["metadata"]["input_schemas"],
        "source_diagnostics": assembly["metadata"]["source_diagnostics"],
        "coverage": assembly["metadata"]["coverage"],
        "fired_event_counts": assembly["metadata"]["fired_event_counts"],
        "fired_event_count_by_fold": assembly["metadata"]["fired_event_count_by_fold"],
        "feature_columns": assembly["feature_columns"],
        "categorical_features": assembly["categorical_features"],
        "inf_to_nan_count_by_feature": assembly["metadata"]["inf_to_nan_count_by_feature"],
        "fallback_counts": fallback_counts,
        "economic_contract": {
            "meta_y": "(tb_return - COST_ROUND_TRIP > 0).astype(int)",
            "cost_round_trip": cost,
            "embargo": embargo,
            "base_fire_threshold": META_BASE_THRESHOLD,
            "event_window_hours": 96,
        },
        "model_contract": {
            "catboost_version": catboost.__version__,
            "params": META_CATBOOST_PARAMS,
            "has_time": False,
            "p_meta_semantics": (
                "Raw probability-like output under weighted Logloss with scale_pos_weight; "
                "not an absolute calibrated probability and not live-usable before v49."
            ),
            "omp_num_threads_environment": os.environ.get("OMP_NUM_THREADS"),
            "observed_omp_conflict": None,
        },
        "walk_forward": {
            "folds": walk_forward["folds"],
            "threshold_selection": walk_forward["threshold_selection"],
            "baseline_anchor": walk_forward["baseline_anchor"],
            "tau_final_raw_diagnostic": walk_forward["tau_final_raw_diagnostic"],
            "auc": build_auc_report(oof_events, walk_forward["folds"]),
        },
        "evaluation": {
            **arm_reports,
            "concurrency": build_concurrency_report(oof_events),
            "y_net_reconciliation": build_y_net_reconciliation(oof_events),
        },
        "final_model": final_fit,
        "deployment": {
            "deployment_ready": False,
            "calibration_status": "pending_v49",
            "deployment_threshold": None,
            "tau_final_raw_diagnostic_live_use_forbidden": True,
        },
        "transaction_contract": (
            "Python exceptions restore the managed destination to its previous state. "
            "SIGKILL during commit is not atomic, but a mixed state is detectable because "
            "the summary commit marker is written last."
        ),
    }
    outputs = write_outputs_transactional(
        final_model,
        oof_events,
        summary,
        identity["output_dir"],
        identity["symbols"],
    )
    summary["output_paths"] = outputs
    return _json_native(summary)
