# ==============================================================================
# analytics/ablation81/quarantine.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Selection-population guards for the frozen fold-five quarantine."""

from __future__ import annotations

from typing import Any

import pandas as pd


class FutureFoldViolation(RuntimeError):
    """Raised when an outer search receives a current or future fold."""


class QuarantineViolation(FutureFoldViolation):
    """Raised when fold five enters any selection population."""


def _numeric_folds(frame: pd.DataFrame) -> pd.Series:
    if "fold" not in frame.columns:
        raise FutureFoldViolation("selection population lacks fold")
    try:
        folds = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    except (TypeError, ValueError) as exc:
        raise FutureFoldViolation("selection population fold is not integral") from exc
    return folds


def assert_selection_population(frame: pd.DataFrame) -> dict[str, Any]:
    """Hard-stop if even one fold-five row reaches selection."""

    folds = _numeric_folds(frame)
    fold5_count = int(folds.eq(5).sum())
    if fold5_count:
        raise QuarantineViolation(
            f"fold-five quarantine violation: row_count={fold5_count}"
        )
    return {
        "status": "passed",
        "row_count": int(len(frame)),
        "folds": sorted(int(item) for item in folds.unique()),
        "fold5_count": 0,
    }


def assert_outer_training_population(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
) -> dict[str, Any]:
    """Require every supplied selection row to precede the outer fold."""

    outer = int(outer_fold)
    if outer not in (2, 3, 4, 5):
        raise FutureFoldViolation(f"invalid outer fold: {outer}")
    quarantine = assert_selection_population(frame)
    folds = _numeric_folds(frame)
    invalid = folds.ge(outer)
    if invalid.any():
        counts = {
            str(int(fold)): int(count)
            for fold, count in folds.loc[invalid].value_counts().sort_index().items()
        }
        raise FutureFoldViolation(
            f"outer fold {outer} received current/future rows: {counts}"
        )
    return {
        **quarantine,
        "outer_fold": outer,
        "past_only": True,
    }


def assert_outer_evaluation_population(
    frame: pd.DataFrame,
    *,
    outer_fold: int,
) -> dict[str, Any]:
    """Validate evaluation rows while explicitly allowing outer fold five."""

    outer = int(outer_fold)
    folds = _numeric_folds(frame)
    invalid = ~folds.eq(outer)
    if invalid.any():
        raise FutureFoldViolation(
            f"outer evaluation contains rows outside fold {outer}"
        )
    return {
        "status": "passed",
        "outer_fold": outer,
        "row_count": int(len(frame)),
        "selection_use_allowed": False,
    }
