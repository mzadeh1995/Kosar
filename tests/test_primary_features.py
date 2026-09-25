# ==============================================================================
# tests/test_primary_features.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from primary_features import build_xy, load_symbol_dataset, select_features_cv, select_features_fold
from TripleBarrier import PurgedWalkForwardCV


def _cfg(**overrides) -> dict:
    cfg = {
        "PRIMARY_FEATURE_COLUMNS": list(CONFIG["PRIMARY_FEATURE_COLUMNS"]),
        "PRIMARY_CV_SPLITS": 4,
        "PRIMARY_RANDOM_SEED": 41,
        "PRIMARY_RF_PARAMS": {
            "n_estimators": 50,
            "max_depth": None,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
        },
        "PRIMARY_FS_VAL_FRACTION": 0.25,
        "PRIMARY_FS_PERM_REPEATS": 3,
        "PRIMARY_FS_KEEP_RATIO": 0.5,
        "TRIPLE_BARRIER_EMBARGO_BARS": 2,
    }
    cfg.update(overrides)
    return cfg


def _synthetic_labeled(rows: int = 420, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC", name="OpenTime")
    signal = rng.normal(0.0, 1.0, rows)
    y = (signal + rng.normal(0.0, 0.15, rows) > 0.0).astype(int)

    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, rows)))
    df = pd.DataFrame(index=idx)
    df["Open"] = np.r_[close[0], close[:-1]]
    df["High"] = np.maximum(df["Open"], close) * 1.005
    df["Low"] = np.minimum(df["Open"], close) * 0.995
    df["Close"] = close
    df["Volume"] = 1000.0 + rng.normal(0.0, 10.0, rows)
    df["QuoteVolume"] = df["Volume"] * close
    df["Trades"] = 1000 + rng.integers(0, 100, rows)
    df["TakerBuyBase"] = df["Volume"] * 0.51
    df["TakerBuyQuote"] = df["TakerBuyBase"] * close
    df["Symbol"] = "BTCUSDT"

    for col in CONFIG["PRIMARY_FEATURE_COLUMNS"]:
        df[col] = rng.normal(0.0, 1.0, rows)
    df["log_return"] = signal

    df["tb_label"] = np.where(y == 1, 1.0, -1.0)
    df["tb_return"] = np.where(y == 1, 0.012, -0.008)
    df["tb_horizon"] = 2
    df["tb_upper_barrier"] = close * 1.02
    df["tb_lower_barrier"] = close * 0.99
    df["tb_event"] = np.where(y == 1, "pt", "sl")
    df["tb_exit_index"] = idx + pd.Timedelta(hours=2)
    df["tb_uniqueness"] = rng.uniform(0.5, 1.0, rows)
    return df


def test_signal_feature_is_selected_and_has_highest_importance():
    result = select_features_cv(_synthetic_labeled(), _cfg())
    importances = result["mean_importances"]
    assert result["valid_folds"] > 0
    assert "log_return" in result["selected_features"]
    assert importances["log_return"] == importances.max()


def test_pure_noise_feature_stays_well_below_signal():
    result = select_features_cv(_synthetic_labeled(), _cfg())
    importances = result["mean_importances"]
    assert importances["volume_change"] < importances["log_return"] * 0.5


def test_feature_selection_used_ranges_stay_before_each_test_fold():
    df = _synthetic_labeled()
    cfg = _cfg()
    result = select_features_cv(df, cfg)
    cv = PurgedWalkForwardCV(n_splits=cfg["PRIMARY_CV_SPLITS"], embargo_bars=cfg["TRIPLE_BARRIER_EMBARGO_BARS"])
    test_starts = {
        fold_number: df.index[test_idx[0]]
        for fold_number, (_train_idx, test_idx) in enumerate(cv.split(df), start=1)
        if len(test_idx)
    }
    assert result["fold_used_ranges"]
    for item in result["fold_used_ranges"]:
        used_start, used_end = item["used_range"]
        test_start = test_starts[item["fold"]]
        assert used_start < test_start
        assert used_end < test_start


def test_build_xy_rejects_tb_feature_leakage():
    df = _synthetic_labeled()
    df["tb_fake"] = 1.0
    cfg = _cfg(PRIMARY_FEATURE_COLUMNS=["tb_fake", *CONFIG["PRIMARY_FEATURE_COLUMNS"]])
    with pytest.raises(ValueError, match="Leakage guard"):
        build_xy(df, cfg)


def test_build_xy_missing_feature_column_is_readable():
    df = _synthetic_labeled().drop(columns=["flow_imb_z"])
    with pytest.raises(ValueError, match="Missing PRIMARY_FEATURE_COLUMNS"):
        build_xy(df, _cfg())


def test_feature_selection_is_reproducible_with_same_seed():
    df = _synthetic_labeled()
    first = select_features_cv(df, _cfg())
    second = select_features_cv(df, _cfg())
    assert first["selected_features"] == second["selected_features"]
    pd.testing.assert_series_equal(first["mean_importances"], second["mean_importances"], check_exact=True)


def _fold_inputs_for_single_class(case: str):
    cfg = _cfg(PRIMARY_FS_VAL_FRACTION=0.25)
    rng = np.random.default_rng(123)
    idx = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC", name="OpenTime")
    X = pd.DataFrame(rng.normal(size=(40, len(cfg["PRIMARY_FEATURE_COLUMNS"]))), index=idx, columns=cfg["PRIMARY_FEATURE_COLUMNS"])
    w = pd.Series(1.0, index=idx)
    end_times = pd.Series(idx + pd.Timedelta(hours=1), index=idx)

    if case == "subtrain":
        y_values = np.r_[np.zeros(30, dtype=int), np.tile([0, 1], 5)]
    elif case == "validation":
        y_values = np.r_[np.tile([0, 1], 15), np.ones(10, dtype=int)]
    else:
        raise AssertionError(case)
    y = pd.Series(y_values, index=idx)
    return X, y, w, end_times, cfg


def test_single_class_internal_subtrain_fold_is_skipped_without_crash():
    X, y, w, end_times, cfg = _fold_inputs_for_single_class("subtrain")
    with pytest.warns(UserWarning, match="subtrain slice has only one class"):
        result = select_features_fold(X, y, w, end_times, cfg)
    assert result is None


def test_single_class_internal_validation_fold_is_skipped_without_crash():
    X, y, w, end_times, cfg = _fold_inputs_for_single_class("validation")
    with pytest.warns(UserWarning, match="validation slice has only one class"):
        result = select_features_fold(X, y, w, end_times, cfg)
    assert result is None


def test_load_symbol_dataset_rejects_duplicate_opentime(tmp_path):
    path = tmp_path / "duplicate_opentime.csv"
    pd.DataFrame(
        {
            "OpenTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "tb_exit_index": ["2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match=rf"(?i)duplicate.*{path}"):
        load_symbol_dataset(path)


def test_load_symbol_dataset_accepts_unique_opentime(tmp_path):
    path = tmp_path / "unique_opentime.csv"
    pd.DataFrame(
        {
            "OpenTime": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "tb_exit_index": ["2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
        }
    ).to_csv(path, index=False)

    loaded = load_symbol_dataset(path)

    assert len(loaded) == 2
    assert loaded.index.is_unique
