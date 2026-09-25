# ==============================================================================
# tests/test_XGBoost.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import CONFIG
from primary_features import build_xy
from TripleBarrier import PurgedWalkForwardCV
from XGBoost import evaluate_fold, round_trip_cost, run_purged_cv, save_oof


def _config(tmp_path=None, **overrides) -> dict:
    cfg = {
        "PRIMARY_FEATURE_COLUMNS": list(CONFIG["PRIMARY_FEATURE_COLUMNS"]),
        "PRIMARY_CV_SPLITS": 3,
        "PRIMARY_RANDOM_SEED": 41,
        "PRIMARY_RF_PARAMS": {
            "n_estimators": 40,
            "max_depth": None,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
        },
        "PRIMARY_FS_VAL_FRACTION": 0.25,
        "PRIMARY_FS_PERM_REPEATS": 3,
        "PRIMARY_FS_KEEP_RATIO": 0.5,
        "TRIPLE_BARRIER_EMBARGO_BARS": 2,
        "PRIMARY_FEE_BPS_PER_SIDE": 10.0,
        "PRIMARY_SLIPPAGE_BPS_PER_SIDE": 5.0,
        "PRIMARY_PROB_THRESHOLDS": [0.30, 0.50],
        "PRIMARY_XGB_PARAMS": {
            "n_estimators": 40,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "n_jobs": 1,
        },
    }
    if tmp_path is not None:
        cfg["PRIMARY_MODEL_DIR"] = str(tmp_path)
    cfg.update(overrides)
    return cfg


def _synthetic_labeled(rows: int = 480, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC", name="OpenTime")
    signal = rng.normal(size=rows)
    y = (signal + rng.normal(scale=0.18, size=rows) > 0.0).astype(int)
    close = 100.0 * np.exp(np.cumsum(rng.normal(scale=0.002, size=rows)))

    df = pd.DataFrame(index=index)
    df["Symbol"] = "BTCUSDT"
    for column in CONFIG["PRIMARY_FEATURE_COLUMNS"]:
        df[column] = rng.normal(size=rows)
    df["log_return"] = signal
    df["tb_label"] = np.where(y == 1, 1.0, -1.0)
    df["tb_return"] = np.where(y == 1, 0.012, -0.008)
    df["tb_event"] = np.where(y == 1, "pt", "sl")
    df["tb_exit_index"] = index + pd.Timedelta(hours=2)
    df["tb_uniqueness"] = rng.uniform(0.5, 1.0, rows)
    df["Close"] = close
    return df


@pytest.fixture(scope="module")
def synthetic_result():
    df = _synthetic_labeled()
    cfg = _config()
    return df, cfg, run_purged_cv(df, cfg)


def test_planted_signal_gives_usable_auc_in_both_arms(synthetic_result):
    _df, _cfg_value, result = synthetic_result
    assert result["summary"]["A"]["mean_auc"] > 0.6
    assert result["summary"]["B"]["mean_auc"] > 0.6
    assert result["folds"]
    assert all("log_return" in fold["selected_features_b"] for fold in result["folds"])


def test_train_range_ends_before_test_range_starts(synthetic_result):
    _df, _cfg_value, result = synthetic_result
    for fold in result["folds"]:
        assert fold["train_range"][1] < fold["test_range"][0]


def test_scale_pos_weight_is_computed_from_fold_train_only(synthetic_result):
    df, cfg, result = synthetic_result
    xy = build_xy(df, cfg)
    cv = PurgedWalkForwardCV(
        n_splits=cfg["PRIMARY_CV_SPLITS"],
        embargo_bars=cfg["TRIPLE_BARRIER_EMBARGO_BARS"],
    )
    split_by_number = {
        number: train_pos
        for number, (train_pos, _test_pos) in enumerate(cv.split(df), start=1)
    }

    for fold in result["folds"]:
        raw_train_index = pd.Index(df.index[split_by_number[fold["fold"]]])
        train_index = pd.Index([value for value in raw_train_index if value in set(xy["index"])])
        y_train = xy["y"].loc[train_index]
        w_train = xy["w"].loc[train_index]
        expected = float(w_train.loc[y_train == 0].sum() / w_train.loc[y_train == 1].sum())
        assert fold["scale_pos_weight"] == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_save_oof_has_required_columns_unique_index_and_valid_row_count(tmp_path, synthetic_result):
    _df, cfg, result = synthetic_result
    path = save_oof(result["oof"], "BTCUSDT", "1h", _config(tmp_path))
    saved = pd.read_csv(path, parse_dates=["OpenTime"])
    assert list(saved.columns) == [
        "OpenTime",
        "fold",
        "p_primary",
        "y",
        "tb_return",
        "tb_uniqueness",
    ]
    assert not saved["OpenTime"].duplicated().any()
    assert len(saved) == sum(fold["test_events"] for fold in result["folds"])


def test_same_seed_reproduces_auc_and_oof_probabilities_exactly():
    df = _synthetic_labeled(rows=360)
    cfg = _config()
    first = run_purged_cv(df, cfg)
    second = run_purged_cv(df, cfg)
    assert first["summary"]["A"]["mean_auc"] == second["summary"]["A"]["mean_auc"]
    assert first["summary"]["B"]["mean_auc"] == second["summary"]["B"]["mean_auc"]
    np.testing.assert_array_equal(
        first["oof"]["p_primary"].to_numpy(),
        second["oof"]["p_primary"].to_numpy(),
    )


class _FixedProbabilityModel:
    primary_feature_columns_ = ["signal"]

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, _X):
        return np.column_stack([1.0 - self.probabilities, self.probabilities])


def test_round_trip_cost_and_net_are_exactly_point_zero_zero_three():
    cfg = _config()
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    X = pd.DataFrame({"signal": [0.0, 1.0]}, index=index)
    result = evaluate_fold(
        _FixedProbabilityModel([0.8, 0.9]),
        X,
        pd.Series([1, 0], index=index),
        pd.Series([0.013, -0.007], index=index),
        pd.Series(["pt", "sl"], index=index),
        cfg,
    )
    assert round_trip_cost(cfg) == pytest.approx(0.003)
    stats = result["thresholds"]["0.50"]
    assert stats["sum_net"] == pytest.approx((0.013 - 0.003) + (-0.007 - 0.003))
    assert stats["mean_net"] == pytest.approx(0.0)


def test_single_class_test_auc_is_none_without_crash():
    cfg = _config()
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    X = pd.DataFrame({"signal": [0.0, 1.0, 2.0]}, index=index)
    result = evaluate_fold(
        _FixedProbabilityModel([0.2, 0.5, 0.8]),
        X,
        pd.Series([1, 1, 1], index=index),
        pd.Series([0.01, 0.02, 0.03], index=index),
        pd.Series(["pt", "vertical", "pt"], index=index),
        cfg,
    )
    assert result["auc"] is None
    assert result["thresholds"]["0.50"]["count"] == 2
