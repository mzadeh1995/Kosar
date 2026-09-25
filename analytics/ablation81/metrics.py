# ==============================================================================
# analytics/ablation81/metrics.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Deterministic shared calculators for ablation81 v7."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


MAXDD_SEMANTICS = "event_level_additive_unit_trade_curve"
ES_SEMANTICS = "event_level_unit_trade_tail_mean"
ANALYSIS_FOLDS = (2, 3, 4, 5)
REPORT_SCOPES: Mapping[str, tuple[int, ...]] = {
    "fold_2": (2,),
    "fold_3": (3,),
    "fold_4": (4,),
    "fold_5": (5,),
    "folds_2_4": (2, 3, 4),
    "folds_2_5": (2, 3, 4, 5),
}


class MetricsContractError(RuntimeError):
    """Raised when a shared-calculator precondition is violated."""


def strict_bool(series: pd.Series, *, label: str) -> pd.Series:
    """Return a null-free boolean series without truthy coercion."""

    if series.isna().any():
        raise MetricsContractError(f"{label} contains null values")
    if series.dtype == bool:
        return series.astype(bool)
    observed = set(series.unique().tolist())
    if not observed.issubset({True, False, 0, 1}):
        raise MetricsContractError(
            f"{label} contains non-boolean values: {sorted(observed, key=str)!r}"
        )
    return series.astype(bool)


def prepare_evaluated_events(
    events: pd.DataFrame,
    *,
    economic_cost: float,
) -> pd.DataFrame:
    """Select evaluated rows and attach deterministic net/holding columns."""

    required = {
        "decision_ts",
        "fold",
        "symbol",
        "tb_return",
        "tb_exit_index",
        "tb_uniqueness",
        "meta_y",
        "p_meta",
        "p_primary",
        "traded_meta",
        "traded_baseline",
        "meta_eval_status",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise MetricsContractError(f"evaluated population misses columns: {missing}")
    if not math.isfinite(float(economic_cost)):
        raise MetricsContractError("economic_cost must be finite")

    work = events.loc[events["meta_eval_status"].eq("evaluated")].copy()
    work["decision_ts"] = pd.to_datetime(
        work["decision_ts"], utc=True, errors="raise"
    )
    work["tb_exit_index"] = pd.to_datetime(
        work["tb_exit_index"], utc=True, errors="raise"
    )
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["symbol"] = work["symbol"].astype(str)
    for column in (
        "tb_return",
        "tb_uniqueness",
        "meta_y",
        "p_meta",
        "p_primary",
    ):
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise MetricsContractError(f"{column} contains non-finite values")
    if not set(work["fold"].unique().tolist()).issubset(set(ANALYSIS_FOLDS)):
        raise MetricsContractError(
            f"evaluated rows must be folds 2..5, got {sorted(work['fold'].unique())}"
        )
    if work.duplicated(["symbol", "decision_ts"]).any():
        raise MetricsContractError("evaluated symbol/decision_ts pairs are not unique")

    work = work.sort_values(
        ["decision_ts", "symbol"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    work["_event_id"] = np.arange(len(work), dtype=np.int64)
    work["net_return"] = (
        work["tb_return"].to_numpy(dtype=float) - float(economic_cost)
    )
    holding = (
        work["tb_exit_index"] - work["decision_ts"]
    ) / pd.Timedelta(hours=1)
    work["holding_hours"] = holding.to_numpy(dtype=float)
    if (
        not np.isfinite(work["holding_hours"].to_numpy(dtype=float)).all()
        or (work["holding_hours"] < 0.0).any()
    ):
        raise MetricsContractError("holding_hours must be finite and non-negative")
    return work


def arm_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    """Build A1/A2/A7 only from their pinned boolean columns."""

    if "traded_baseline" not in events or "traded_meta" not in events:
        raise MetricsContractError("arm masks require pinned trade columns")
    a1 = strict_bool(events["traded_baseline"], label="traded_baseline")
    a2 = strict_bool(events["traded_meta"], label="traded_meta")
    return {
        "A1": a1,
        "A2": a2,
        "A7": a1 & a2,
    }


def additive_equity_curve(net_returns: Sequence[float]) -> np.ndarray:
    """Return the contract's additive unit-trade cumulative curve."""

    values = np.asarray(net_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise MetricsContractError("net_returns must be a finite one-dimensional array")
    return np.cumsum(values, dtype=float)


def max_drawdown(net_returns: Sequence[float]) -> float | None:
    """Compute peak-to-current drawdown on the stated additive curve."""

    curve = additive_equity_curve(net_returns)
    if curve.size == 0:
        return None
    running_peak = np.maximum.accumulate(curve)
    return float(np.max(running_peak - curve))


def expected_shortfall_5pct(
    net_returns: Sequence[float],
) -> tuple[float | None, int]:
    """Return the event-level five-percent tail mean and tail count."""

    values = np.asarray(net_returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise MetricsContractError("net_returns must be finite for expected shortfall")
    if values.size == 0:
        return None, 0
    tail_count = max(1, int(math.ceil(0.05 * values.size)))
    tail = np.partition(values, tail_count - 1)[:tail_count]
    return float(np.mean(tail)), tail_count


def trade_metrics(
    population: pd.DataFrame,
    selected_mask: pd.Series | Sequence[bool],
) -> dict[str, Any]:
    """Calculate all section 8.1 trade metrics for one population slice."""

    if isinstance(selected_mask, pd.Series):
        if not selected_mask.index.equals(population.index):
            raise MetricsContractError(
                "selected mask index differs from population index"
            )
        candidate = selected_mask.copy()
    else:
        if len(selected_mask) != len(population):
            raise MetricsContractError(
                "selected mask length differs from population"
            )
        candidate = pd.Series(selected_mask, index=population.index)
    mask = strict_bool(candidate, label="selected_mask")
    selected = population.loc[mask].sort_values(
        ["decision_ts", "symbol"], ascending=[True, True], kind="mergesort"
    )
    net = selected["net_return"].to_numpy(dtype=float)
    holding = selected["holding_hours"].to_numpy(dtype=float)
    n_trades = int(len(selected))
    es, es_tail_n = expected_shortfall_5pct(net)
    return {
        "n_trades": n_trades,
        "sum_net": float(np.sum(net, dtype=float)),
        "mean_net": None if n_trades == 0 else float(np.mean(net)),
        "win_rate": None if n_trades == 0 else float(np.mean(net > 0.0)),
        "max_drawdown": max_drawdown(net),
        "entry_rate": (
            None if len(population) == 0 else float(n_trades / len(population))
        ),
        "holding_hours_mean": (
            None if n_trades == 0 else float(np.mean(holding))
        ),
        "holding_hours_median": (
            None if n_trades == 0 else float(np.median(holding))
        ),
        "holding_hours_sum": float(np.sum(holding, dtype=float)),
        "expected_shortfall_5pct": es,
        "expected_shortfall_tail_n": es_tail_n,
    }


def model_metrics(
    population: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, float | None]:
    """Compute unweighted raw-score AUC and uniqueness-weighted logloss."""

    if population.empty:
        return {"auc_unweighted_raw": None, "logloss_tb_uniqueness_weighted": None}
    score = pd.to_numeric(population[score_column], errors="raise").to_numpy(
        dtype=float
    )
    target = pd.to_numeric(population["meta_y"], errors="raise").to_numpy(
        dtype=float
    )
    weight = pd.to_numeric(
        population["tb_uniqueness"], errors="raise"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(score).all()
        or not np.isfinite(target).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0.0)
        or float(weight.sum()) <= 0.0
    ):
        raise MetricsContractError("model-metric inputs are invalid")
    clipped = np.clip(score, 1e-15, 1.0 - 1e-15)
    row_loss = -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    auc = (
        None
        if np.unique(target).size < 2
        else float(roc_auc_score(target, score))
    )
    return {
        "auc_unweighted_raw": auc,
        "logloss_tb_uniqueness_weighted": float(np.average(row_loss, weights=weight)),
    }


def _scope_population(events: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame:
    return events.loc[events["fold"].isin(tuple(folds))]


def calculate_arm_metrics(
    events: pd.DataFrame,
    masks: Mapping[str, pd.Series],
) -> dict[str, Any]:
    """Calculate phase-two A1/A2/A7 metrics across every required slice."""

    report: dict[str, Any] = {}
    for arm in ("A1", "A2", "A7"):
        if arm not in masks:
            raise MetricsContractError(f"missing arm mask {arm}")
        arm_report: dict[str, Any] = {}
        for scope, folds in REPORT_SCOPES.items():
            population = _scope_population(events, folds)
            selected = masks[arm].reindex(population.index)
            if selected.isna().any():
                raise MetricsContractError(f"{arm}/{scope} mask cannot be aligned")
            metrics = trade_metrics(population, selected.astype(bool))
            if arm == "A2":
                metrics["model_metrics"] = model_metrics(
                    population, score_column="p_meta"
                )
            arm_report[scope] = metrics
        report[arm] = arm_report
    return report


def equal_n_fold(
    events: pd.DataFrame,
    *,
    fold: int,
    a2_mask: pd.Series,
) -> dict[str, Any]:
    """Compute the deterministic top-N comparison for one fold."""

    current = events.loc[events["fold"].eq(int(fold))].copy()
    current_mask = a2_mask.reindex(current.index)
    if current_mask.isna().any():
        raise MetricsContractError(f"fold {fold} A2 mask cannot be aligned")
    n = int(current_mask.astype(bool).sum())
    if n > len(current):
        raise MetricsContractError(f"fold {fold} equal-N exceeds population")
    sums: dict[str, float] = {}
    selected_ids: dict[str, list[int]] = {}
    for score in ("p_meta", "p_primary"):
        if current[score].isna().any():
            raise MetricsContractError(f"fold {fold} {score} contains nulls")
        ranked = current.sort_values(
            [score, "decision_ts", "symbol"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        chosen = ranked.head(n)
        sums[score] = float(chosen["net_return"].sum())
        selected_ids[score] = [int(item) for item in chosen["_event_id"].tolist()]
    return {
        "fold": int(fold),
        "n": n,
        "meta_sum_net": sums["p_meta"],
        "primary_sum_net": sums["p_primary"],
        "selected_event_ids": selected_ids,
    }


def equal_n_report(
    events: pd.DataFrame,
    *,
    a2_mask: pd.Series,
) -> dict[str, Any]:
    """Return E2 per-fold values and foldwise aggregate comparisons."""

    per_fold = {
        f"fold_{fold}": equal_n_fold(events, fold=fold, a2_mask=a2_mask)
        for fold in ANALYSIS_FOLDS
    }

    def aggregate(folds: Sequence[int]) -> dict[str, Any]:
        records = [per_fold[f"fold_{fold}"] for fold in folds]
        return {
            "folds": [int(fold) for fold in folds],
            "n": int(sum(record["n"] for record in records)),
            "meta_sum_net": float(sum(record["meta_sum_net"] for record in records)),
            "primary_sum_net": float(
                sum(record["primary_sum_net"] for record in records)
            ),
        }

    return {
        "per_fold": per_fold,
        "folds_2_4": aggregate((2, 3, 4)),
        "folds_2_5": aggregate((2, 3, 4, 5)),
    }


def overlap_report(
    events: pd.DataFrame,
    *,
    a1_mask: pd.Series,
    a2_mask: pd.Series,
) -> dict[str, Any]:
    """Return E1 three-region overlap numbers per fold and in aggregates."""

    regions = {
        "common": a1_mask & a2_mask,
        "meta_only": ~a1_mask & a2_mask,
        "primary_only": a1_mask & ~a2_mask,
    }
    result: dict[str, Any] = {}
    for scope, folds in REPORT_SCOPES.items():
        population = _scope_population(events, folds)
        scope_record: dict[str, Any] = {}
        for name, mask in regions.items():
            aligned = mask.reindex(population.index)
            if aligned.isna().any():
                raise MetricsContractError(f"{scope}/{name} mask cannot be aligned")
            selected = population.loc[aligned.astype(bool)]
            scope_record[name] = {
                "n_trades": int(len(selected)),
                "sum_net": float(selected["net_return"].sum()),
            }
        result[scope] = scope_record
    return result


def percentile_rank(
    placebo_values: Sequence[float],
    observed_value: float,
) -> float:
    """Apply the pinned mid-rank convention exactly."""

    values = np.asarray(placebo_values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not math.isfinite(float(observed_value))
    ):
        raise MetricsContractError("percentile inputs must be finite and non-empty")
    less = int(np.count_nonzero(values < float(observed_value)))
    equal = int(np.count_nonzero(values == float(observed_value)))
    return float((less + 0.5 * equal) / values.size * 100.0)


def monte_carlo_p(
    placebo_values: Sequence[float],
    observed_value: float,
) -> float:
    """Return the pinned plus-one upper-tail Monte Carlo p value."""

    values = np.asarray(placebo_values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not math.isfinite(float(observed_value))
    ):
        raise MetricsContractError("Monte Carlo inputs must be finite and non-empty")
    greater_equal = int(np.count_nonzero(values >= float(observed_value)))
    return float((1 + greater_equal) / (values.size + 1))


def pool_composition(
    events: pd.DataFrame,
    *,
    pool_mask: pd.Series,
    unique_to_arm_mask: pd.Series,
) -> dict[str, Any]:
    """Summarize one null pool and its arm-unique-row sensitivity."""

    pool = pool_mask.reindex(events.index)
    unique = unique_to_arm_mask.reindex(events.index)
    if pool.isna().any() or unique.isna().any():
        raise MetricsContractError("pool composition masks cannot be aligned")
    pool = pool.astype(bool)
    unique = unique.astype(bool)
    selected = events.loc[pool, "net_return"].to_numpy(dtype=float)
    without_unique = events.loc[pool & ~unique, "net_return"].to_numpy(dtype=float)
    return {
        "pool_size": int(selected.size),
        "pool_mean_net": None if selected.size == 0 else float(np.mean(selected)),
        "pool_median_net": None if selected.size == 0 else float(np.median(selected)),
        "pool_mean_net_without_arm_unique_trades": (
            None if without_unique.size == 0 else float(np.mean(without_unique))
        ),
        "removed_arm_unique_trade_count": int((pool & unique).sum()),
    }
