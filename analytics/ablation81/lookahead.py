# ==============================================================================
# analytics/ablation81/lookahead.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Full-population look-ahead alignment checks for ablation81 phase one."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


VERDICT_FA = {
    "healthy": "سالم",
    "error": "خطا",
    "unknown": "نامشخص",
    "not_applicable": "نامربوط",
}


class LookaheadAuditError(RuntimeError):
    """Raised when a look-ahead audit cannot be evaluated deterministically."""


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LookaheadAuditError(f"{context} misses required columns: {missing}")


def _strict_match_mask(series: pd.Series, *, context: str) -> pd.Series:
    if series.isna().any():
        raise LookaheadAuditError(f"{context} match mask contains null values")
    if series.dtype == bool:
        return series
    values = set(series.unique().tolist())
    if not values.issubset({True, False, 0, 1}):
        raise LookaheadAuditError(
            f"{context} match mask contains non-boolean values: {values!r}"
        )
    return series.astype(bool)


def audit_asof_alignment(
    frame: pd.DataFrame,
    *,
    source: str,
    decision_column: str,
    source_timestamp_column: str,
    match_column: str,
    tolerance: pd.Timedelta,
    feature_columns: Sequence[str] = (),
    symbol_column: str | None = None,
    expected_lag: pd.Timedelta | None = None,
    require_all_rows_matched: bool = True,
    include_by_symbol: bool = True,
) -> dict[str, Any]:
    """Scan every row and issue the phase-one P0-A verdict for one source."""

    if tolerance < pd.Timedelta(0):
        raise LookaheadAuditError(f"{source} tolerance must be non-negative")
    required = [
        decision_column,
        source_timestamp_column,
        match_column,
        *feature_columns,
    ]
    if symbol_column is not None:
        required.append(symbol_column)
    _required_columns(frame, required, source)

    decisions = pd.to_datetime(frame[decision_column], utc=True, errors="coerce")
    source_timestamps = pd.to_datetime(
        frame[source_timestamp_column], utc=True, errors="coerce"
    )
    matched = _strict_match_mask(frame[match_column], context=source)
    invalid_decision = decisions.isna()
    matched_without_timestamp = matched & source_timestamps.isna()
    valid_match = matched & ~invalid_decision & source_timestamps.notna()
    lag = decisions.loc[valid_match] - source_timestamps.loc[valid_match]
    future = lag < pd.Timedelta(0)
    older_than_tolerance = lag > tolerance
    unmatched = ~matched

    expected_lag_mismatch_count = 0
    if expected_lag is not None:
        expected_lag_mismatch_count = int((lag != expected_lag).sum())

    matched_but_feature_nan_count = 0
    if feature_columns:
        usable = frame.loc[:, list(feature_columns)].notna().all(axis=1)
        matched_but_feature_nan_count = int((matched & ~usable).sum())

    blocking_count = (
        int(invalid_decision.sum())
        + int(matched_without_timestamp.sum())
        + int(future.sum())
        + int(older_than_tolerance.sum())
        + expected_lag_mismatch_count
    )
    if require_all_rows_matched:
        blocking_count += int(unmatched.sum())
    verdict = "healthy" if blocking_count == 0 else "error"

    result: dict[str, Any] = {
        "source": source,
        "p0_a_verdict": verdict,
        "p0_a_verdict_fa": VERDICT_FA[verdict],
        "rows_scanned": int(len(frame)),
        "matched_count": int(matched.sum()),
        "unmatched_or_outside_tolerance_count": int(unmatched.sum()),
        "invalid_decision_timestamp_count": int(invalid_decision.sum()),
        "matched_without_source_timestamp_count": int(
            matched_without_timestamp.sum()
        ),
        "future_timestamp_violation_count": int(future.sum()),
        "matched_older_than_tolerance_count": int(older_than_tolerance.sum()),
        "expected_lag_mismatch_count": expected_lag_mismatch_count,
        "matched_but_feature_nan_count": matched_but_feature_nan_count,
        "require_all_rows_matched": bool(require_all_rows_matched),
        "tolerance_seconds": float(tolerance.total_seconds()),
        "expected_lag_seconds": (
            None if expected_lag is None else float(expected_lag.total_seconds())
        ),
        "lag_min_seconds": (
            None if lag.empty else float(lag.min().total_seconds())
        ),
        "lag_max_seconds": (
            None if lag.empty else float(lag.max().total_seconds())
        ),
        "blocking_violation_count": blocking_count,
    }
    if include_by_symbol and symbol_column is not None:
        result["by_symbol"] = {
            str(symbol): audit_asof_alignment(
                group,
                source=source,
                decision_column=decision_column,
                source_timestamp_column=source_timestamp_column,
                match_column=match_column,
                tolerance=tolerance,
                feature_columns=feature_columns,
                symbol_column=None,
                expected_lag=expected_lag,
                require_all_rows_matched=require_all_rows_matched,
                include_by_symbol=False,
            )
            for symbol, group in frame.groupby(symbol_column, sort=True)
        }
    return result


def audit_full_population(events: pd.DataFrame) -> dict[str, Any]:
    """Run P0-A over every row for the four time-aligned source blocks."""

    if len(events) != 11_029:
        raise LookaheadAuditError(
            f"phase-one population must contain 11029 rows, observed {len(events)}"
        )
    work = events.copy()
    work["_dataset_source_ts_for_audit"] = pd.to_datetime(
        work["OpenTime"], utc=True, errors="coerce"
    )
    blocks = {
        "dataset": audit_asof_alignment(
            work,
            source="dataset",
            decision_column="decision_ts",
            source_timestamp_column="_dataset_source_ts_for_audit",
            match_column="_dataset_matched",
            tolerance=pd.Timedelta(hours=4),
            expected_lag=pd.Timedelta(hours=4),
            feature_columns=("volatility", "log_range"),
            symbol_column="symbol",
        ),
        "funding": audit_asof_alignment(
            work,
            source="funding",
            decision_column="decision_ts",
            source_timestamp_column="_funding_source_ts",
            match_column="_funding_matched",
            tolerance=pd.Timedelta(hours=8, minutes=1),
            feature_columns=("funding_z", "funding_extreme_pos"),
            symbol_column="symbol",
        ),
        "vpin": audit_asof_alignment(
            work,
            source="vpin",
            decision_column="decision_ts",
            source_timestamp_column="_vpin_source_ts",
            match_column="_vpin_matched",
            tolerance=pd.Timedelta(hours=2),
            feature_columns=("vpin", "vpin_z"),
            symbol_column="symbol",
        ),
        "hmm": audit_asof_alignment(
            work,
            source="hmm",
            decision_column="decision_ts",
            source_timestamp_column="_hmm_source_ts",
            match_column="_hmm_matched",
            tolerance=pd.Timedelta(hours=6),
            feature_columns=(
                "hmm_regime",
                "hmm_bull_prob",
                "hmm_neutral_prob",
                "hmm_bear_prob",
                "hmm_confidence",
                "hmm_policy_code",
                "hmm_regime_age_hours",
            ),
            symbol_column="symbol",
        ),
    }
    return {
        "population_rows": int(len(work)),
        "blocks": blocks,
        "error_blocks": sorted(
            key for key, value in blocks.items() if value["p0_a_verdict"] == "error"
        ),
    }


def audit_calendar_block(events: pd.DataFrame) -> dict[str, Any]:
    """Verify direct UTC calendar derivation on all rows."""

    _required_columns(
        events, ["decision_ts", "hour_of_day", "day_of_week"], "calendar"
    )
    decisions = pd.to_datetime(events["decision_ts"], utc=True, errors="coerce")
    hour = pd.to_numeric(events["hour_of_day"], errors="coerce")
    day = pd.to_numeric(events["day_of_week"], errors="coerce")
    invalid = int(decisions.isna().sum())
    hour_mismatch = int((hour != decisions.dt.hour).sum())
    day_mismatch = int((day != decisions.dt.dayofweek).sum())
    verdict = "healthy" if invalid + hour_mismatch + day_mismatch == 0 else "error"
    return {
        "p0_a_verdict": verdict,
        "p0_a_verdict_fa": VERDICT_FA[verdict],
        "rows_scanned": int(len(events)),
        "invalid_decision_timestamp_count": invalid,
        "hour_of_day_mismatch_count": hour_mismatch,
        "day_of_week_mismatch_count": day_mismatch,
        "null_feature_count": int(
            events[["hour_of_day", "day_of_week"]].isna().any(axis=1).sum()
        ),
    }


def audit_symbol_block(
    events: pd.DataFrame,
    *,
    canonical_symbols: Sequence[str],
) -> dict[str, Any]:
    """Verify the categorical symbol domain and per-row availability."""

    _required_columns(events, ["symbol"], "symbol")
    values = events["symbol"].astype("string")
    null_count = int(values.isna().sum())
    observed = sorted(values.dropna().astype(str).unique().tolist())
    expected = sorted(str(item) for item in canonical_symbols)
    out_of_domain = int((~values.isin(expected) & values.notna()).sum())
    verdict = (
        "healthy"
        if null_count == 0 and out_of_domain == 0 and observed == expected
        else "error"
    )
    return {
        "p0_a_verdict": verdict,
        "p0_a_verdict_fa": VERDICT_FA[verdict],
        "rows_scanned": int(len(events)),
        "null_count": null_count,
        "out_of_domain_count": out_of_domain,
        "expected_values": expected,
        "observed_values": observed,
        "counts": {
            str(key): int(value)
            for key, value in values.value_counts().sort_index().items()
        },
    }


def finite_max_abs_difference(left: pd.Series, right: pd.Series) -> float | None:
    """Return a deterministic finite-only reconciliation diagnostic."""

    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if left_values.shape != right_values.shape:
        raise LookaheadAuditError(
            f"reconciliation shape mismatch: {left_values.shape} != {right_values.shape}"
        )
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if not mask.any():
        return None
    return float(np.max(np.abs(left_values[mask] - right_values[mask])))
