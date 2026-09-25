# ==============================================================================
# analytics/ablation81/placebo.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Seed-pinned count-matched placebo sampling for ablation81 v7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    ANALYSIS_FOLDS,
    MetricsContractError,
    monte_carlo_p,
    percentile_rank,
    strict_bool,
)


PLACEBO_SCOPES = ("fold_2", "fold_3", "fold_4", "fold_5", "folds_2_4")
NULL_TYPES = ("primary_informed", "universe")
MATCHINGS = ("per_fold", "fold_symbol")


class PlaceboContractError(RuntimeError):
    """Raised when a null pool or sampling plan violates the frozen contract."""


@dataclass(frozen=True)
class SamplingCell:
    """One deterministic RNG call in the pinned traversal order."""

    fold: int
    symbol: str | None
    pool_positions: np.ndarray
    sample_size: int


@dataclass(frozen=True)
class SamplingPlan:
    """Precomputed cells for one arm/null/matching combination."""

    arm: str
    null_type: str
    matching: str
    pool_mask: pd.Series
    arm_mask: pd.Series
    cells: tuple[SamplingCell, ...]


def _aligned_bool(
    mask: pd.Series | Sequence[bool],
    events: pd.DataFrame,
    *,
    label: str,
) -> pd.Series:
    if isinstance(mask, pd.Series):
        if not mask.index.equals(events.index):
            raise PlaceboContractError(
                f"{label} index differs from population index"
            )
        series = mask.copy()
    else:
        if len(mask) != len(events):
            raise PlaceboContractError(
                f"{label} length differs from population"
            )
        series = pd.Series(mask, index=events.index)
    try:
        return strict_bool(series, label=label)
    except MetricsContractError as exc:
        raise PlaceboContractError(str(exc)) from exc


def expected_pool_mask(
    events: pd.DataFrame,
    *,
    arm_mask: pd.Series,
    a1_mask: pd.Series,
    null_type: str,
) -> pd.Series:
    """Build the exact equality target for one registered null."""

    if null_type == "primary_informed":
        return a1_mask | arm_mask
    if null_type == "universe":
        return pd.Series(True, index=events.index, dtype=bool)
    raise PlaceboContractError(f"unsupported null_type: {null_type!r}")


def assert_placebo_pool_contract(
    events: pd.DataFrame,
    *,
    pool_mask: pd.Series | Sequence[bool],
    arm_mask: pd.Series | Sequence[bool],
    a1_mask: pd.Series | Sequence[bool],
    null_type: str,
) -> dict[str, Any]:
    """Require exact pool equality, evaluated membership, scores, and subset."""

    required = {
        "fold",
        "symbol",
        "meta_eval_status",
        "p_meta",
        "p_primary",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise PlaceboContractError(f"placebo population misses columns: {missing}")
    pool = _aligned_bool(pool_mask, events, label="pool_mask")
    arm = _aligned_bool(arm_mask, events, label="arm_mask")
    a1 = _aligned_bool(a1_mask, events, label="a1_mask")
    expected = expected_pool_mask(
        events, arm_mask=arm, a1_mask=a1, null_type=null_type
    )
    if not pool.equals(expected):
        disagreement = int((pool != expected).sum())
        raise PlaceboContractError(
            f"{null_type} pool is not exactly equal to its contract; "
            f"disagreement_rows={disagreement}"
        )
    if not events.loc[pool, "meta_eval_status"].eq("evaluated").all():
        raise PlaceboContractError("placebo pool contains non-evaluated rows")
    if events.loc[pool, ["p_meta", "p_primary"]].isna().any(axis=None):
        raise PlaceboContractError("placebo pool contains null required scores")
    if bool((arm & ~pool).any()):
        raise PlaceboContractError("arm is not a subset of its placebo pool")

    fold_records: dict[str, Any] = {}
    for fold in ANALYSIS_FOLDS:
        fold_mask = events["fold"].eq(fold)
        pool_count = int((pool & fold_mask).sum())
        arm_count = int((arm & fold_mask).sum())
        if arm_count > pool_count:
            raise PlaceboContractError(
                f"fold {fold} arm count {arm_count} exceeds pool {pool_count}"
            )
        fold_records[str(fold)] = {
            "pool_count": pool_count,
            "arm_count": arm_count,
        }
    return {
        "status": "passed",
        "null_type": null_type,
        "exact_pool_equality": True,
        "arm_subset": True,
        "folds": fold_records,
    }


def build_sampling_plan(
    events: pd.DataFrame,
    *,
    arm: str,
    arm_mask: pd.Series,
    a1_mask: pd.Series,
    null_type: str,
    matching: str,
) -> tuple[SamplingPlan, dict[str, Any]]:
    """Precompute every pool cell in the exact RNG traversal order."""

    if matching not in MATCHINGS:
        raise PlaceboContractError(f"unsupported matching: {matching!r}")
    if not events.index.equals(pd.RangeIndex(len(events))):
        raise PlaceboContractError(
            "placebo events must have a deterministic zero-based RangeIndex"
        )
    arm_bool = _aligned_bool(arm_mask, events, label=f"{arm}.arm_mask")
    a1_bool = _aligned_bool(a1_mask, events, label="A1.arm_mask")
    pool = expected_pool_mask(
        events, arm_mask=arm_bool, a1_mask=a1_bool, null_type=null_type
    )
    pool_audit = assert_placebo_pool_contract(
        events,
        pool_mask=pool,
        arm_mask=arm_bool,
        a1_mask=a1_bool,
        null_type=null_type,
    )
    symbols = sorted(events["symbol"].astype(str).unique().tolist())
    cells: list[SamplingCell] = []
    cell_audit: list[dict[str, Any]] = []
    for fold in ANALYSIS_FOLDS:
        fold_mask = events["fold"].eq(fold)
        cell_symbols: tuple[str | None, ...] = (
            (None,) if matching == "per_fold" else tuple(symbols)
        )
        for symbol in cell_symbols:
            cell_mask = fold_mask.copy()
            if symbol is not None:
                cell_mask &= events["symbol"].astype(str).eq(symbol)
            positions = np.flatnonzero((pool & cell_mask).to_numpy(dtype=bool))
            sample_size = int((arm_bool & cell_mask).sum())
            if sample_size > positions.size:
                raise PlaceboContractError(
                    f"{arm}/{null_type}/{matching}/fold={fold}/symbol={symbol} "
                    f"requires {sample_size} from pool {positions.size}"
                )
            positions.setflags(write=False)
            cells.append(
                SamplingCell(
                    fold=int(fold),
                    symbol=symbol,
                    pool_positions=positions,
                    sample_size=sample_size,
                )
            )
            cell_audit.append(
                {
                    "fold": int(fold),
                    "symbol": symbol,
                    "pool_count": int(positions.size),
                    "sample_count": sample_size,
                }
            )
    plan = SamplingPlan(
        arm=arm,
        null_type=null_type,
        matching=matching,
        pool_mask=pool,
        arm_mask=arm_bool,
        cells=tuple(cells),
    )
    return plan, {
        "pool_contract": pool_audit,
        "cell_count": len(cells),
        "cells": cell_audit,
        "traversal": (
            "fold ascending"
            if matching == "per_fold"
            else "fold ascending then symbol alphabetical ascending"
        ),
        "without_replacement": True,
    }


def sample_seed(
    plan: SamplingPlan,
    *,
    seed: int,
    net_returns: np.ndarray,
    holding_hours: np.ndarray,
    return_selected_positions: bool = False,
) -> dict[str, Any]:
    """Execute exactly one seed with one independent default_rng instance."""

    net = np.asarray(net_returns, dtype=float)
    holding = np.asarray(holding_hours, dtype=float)
    if (
        net.ndim != 1
        or holding.shape != net.shape
        or not np.isfinite(net).all()
        or not np.isfinite(holding).all()
    ):
        raise PlaceboContractError("sampling value arrays are invalid")
    rng = np.random.default_rng(int(seed))
    sum_by_fold = np.zeros(len(ANALYSIS_FOLDS), dtype=float)
    holding_by_fold = np.zeros(len(ANALYSIS_FOLDS), dtype=float)
    selected_by_fold: dict[int, list[int]] = {fold: [] for fold in ANALYSIS_FOLDS}
    for cell in plan.cells:
        chosen = np.asarray(
            rng.choice(
                cell.pool_positions,
                size=cell.sample_size,
                replace=False,
            ),
            dtype=np.int64,
        )
        # RangeIndex positions are already in the contract's canonical
        # decision_ts/symbol order.  Sort only after RNG selection so the
        # sampled set is unchanged while floating-point accumulation follows
        # that canonical order.
        chosen.sort()
        selected_by_fold[cell.fold].extend(int(item) for item in chosen.tolist())
    for position, fold in enumerate(ANALYSIS_FOLDS):
        selected = sorted(selected_by_fold[fold])
        selected_by_fold[fold] = selected
        expected = int(
            sum(
                cell.sample_size
                for cell in plan.cells
                if cell.fold == fold
            )
        )
        if len(selected) != expected or len(set(selected)) != expected:
            raise PlaceboContractError(
                f"seed {seed} fold {fold} violates count/no-replacement"
            )
        canonical_positions = np.asarray(selected, dtype=np.int64)
        sum_by_fold[position] = float(net[canonical_positions].sum())
        holding_by_fold[position] = float(holding[canonical_positions].sum())
    return {
        "sum_net_by_fold": sum_by_fold,
        "holding_hours_by_fold": holding_by_fold,
        "selected_positions_by_fold": (
            selected_by_fold if return_selected_positions else None
        ),
    }


def _scope_arrays(values_by_fold: np.ndarray) -> dict[str, np.ndarray]:
    if values_by_fold.ndim != 2 or values_by_fold.shape[1] != len(ANALYSIS_FOLDS):
        raise PlaceboContractError("fold result matrix has the wrong shape")
    return {
        "fold_2": values_by_fold[:, 0],
        "fold_3": values_by_fold[:, 1],
        "fold_4": values_by_fold[:, 2],
        "fold_5": values_by_fold[:, 3],
        "folds_2_4": values_by_fold[:, :3].sum(axis=1),
    }


def run_placebo_plan(
    events: pd.DataFrame,
    plan: SamplingPlan,
    *,
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a precomputed plan and return full sums plus holding summaries."""

    ordered_seeds = np.asarray([int(seed) for seed in seeds], dtype=np.int64)
    if ordered_seeds.ndim != 1 or np.unique(ordered_seeds).size != ordered_seeds.size:
        raise PlaceboContractError("seeds must be a unique one-dimensional sequence")
    net = events["net_return"].to_numpy(dtype=float)
    holding = events["holding_hours"].to_numpy(dtype=float)
    sums = np.zeros((len(ordered_seeds), len(ANALYSIS_FOLDS)), dtype=float)
    holding_sums = np.zeros_like(sums)
    first_seed_audit: dict[str, Any] | None = None
    for row, seed in enumerate(ordered_seeds.tolist()):
        sampled = sample_seed(
            plan,
            seed=seed,
            net_returns=net,
            holding_hours=holding,
            return_selected_positions=(row == 0),
        )
        sums[row, :] = sampled["sum_net_by_fold"]
        holding_sums[row, :] = sampled["holding_hours_by_fold"]
        if row == 0:
            first_seed_audit = {
                "seed": int(seed),
                "selected_counts": {
                    str(fold): len(
                        sampled["selected_positions_by_fold"][fold]
                    )
                    for fold in ANALYSIS_FOLDS
                },
                "unique_selected_counts": {
                    str(fold): len(
                        set(sampled["selected_positions_by_fold"][fold])
                    )
                    for fold in ANALYSIS_FOLDS
                },
            }
    sum_scopes = _scope_arrays(sums)
    holding_scopes = _scope_arrays(holding_sums)
    frames: list[pd.DataFrame] = []
    holding_summary: dict[str, Any] = {}
    for scope in PLACEBO_SCOPES:
        values = sum_scopes[scope]
        frames.append(
            pd.DataFrame(
                {
                    "null_type": plan.null_type,
                    "matching": plan.matching,
                    "arm": plan.arm,
                    "scope": scope,
                    "seed": ordered_seeds,
                    "sum_net": values,
                }
            )
        )
        exposure = holding_scopes[scope]
        holding_summary[scope] = {
            "mean": float(np.mean(exposure)),
            "p05": float(np.percentile(exposure, 5, method="linear")),
            "p50": float(np.percentile(exposure, 50, method="linear")),
            "p95": float(np.percentile(exposure, 95, method="linear")),
            "percentile_method": "linear",
        }
    distribution = pd.concat(frames, ignore_index=True)
    expected_columns = [
        "null_type",
        "matching",
        "arm",
        "scope",
        "seed",
        "sum_net",
    ]
    if list(distribution.columns) != expected_columns:
        raise PlaceboContractError("placebo distribution schema drifted")
    return distribution, {
        "holding_hours_sum_distribution": holding_summary,
        "first_seed_no_replacement_audit": first_seed_audit,
        "seed_count": int(len(ordered_seeds)),
        "seed_min": None if len(ordered_seeds) == 0 else int(ordered_seeds.min()),
        "seed_max": None if len(ordered_seeds) == 0 else int(ordered_seeds.max()),
    }


def summarize_placebo_ranks(
    distribution: pd.DataFrame,
    *,
    observed_sum_net_by_scope: dict[str, float],
) -> dict[str, Any]:
    """Calculate the pinned percentile and p value for each available scope."""

    result: dict[str, Any] = {}
    for scope, observed in observed_sum_net_by_scope.items():
        values = distribution.loc[
            distribution["scope"].eq(scope), "sum_net"
        ].to_numpy(dtype=float)
        if values.size == 0:
            raise PlaceboContractError(f"distribution misses scope {scope}")
        result[scope] = {
            "observed_sum_net": float(observed),
            "placebo_count": int(values.size),
            "percentile_rank": percentile_rank(values, observed),
            "monte_carlo_p": monte_carlo_p(values, observed),
        }
    return result
