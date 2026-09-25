# ==============================================================================
# tests/test_triple_barrier.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from TripleBarrier import (
    TB_COLUMNS,
    PurgedWalkForwardCV,
    apply_triple_barrier,
    compute_causal_volatility,
    compute_uniqueness,
)


def _base_df(rows: int = 8, close: float = 100.0, vol: float = 0.01) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "Open": np.full(rows, close),
            "High": np.full(rows, close * 1.001),
            "Low": np.full(rows, close * 0.999),
            "Close": np.full(rows, close),
            "Volume": np.full(rows, 1000.0),
            "vol": np.full(rows, vol),
        },
        index=idx,
    )


def test_profit_barrier_first_hit():
    df = _base_df()
    upper = 100.0 * math.exp(0.01)
    df.iloc[1, df.columns.get_loc("High")] = upper * 1.001
    out = apply_triple_barrier(df, volatility_col="vol", horizon=3, profit_mult=1.0, loss_mult=1.0)
    assert out.iloc[0]["tb_label"] == 1
    assert out.iloc[0]["tb_event"] == "pt"


def test_loss_barrier_first_hit():
    df = _base_df()
    lower = 100.0 * math.exp(-0.01)
    df.iloc[1, df.columns.get_loc("Low")] = lower * 0.999
    out = apply_triple_barrier(df, volatility_col="vol", horizon=3, profit_mult=1.0, loss_mult=1.0)
    assert out.iloc[0]["tb_label"] == -1
    assert out.iloc[0]["tb_event"] == "sl"


def test_vertical_barrier_without_price_hit():
    df = _base_df()
    out = apply_triple_barrier(df, volatility_col="vol", horizon=3, profit_mult=5.0, loss_mult=5.0)
    assert out.iloc[0]["tb_label"] == 0
    assert out.iloc[0]["tb_event"] == "vertical"


def test_tail_rows_without_full_horizon_are_nan():
    df = _base_df(rows=5)
    out = apply_triple_barrier(df, volatility_col="vol", horizon=3)
    for col in TB_COLUMNS:
        assert out.iloc[2:][col].isna().all()


def test_ambiguous_candle_uses_stop_policy(monkeypatch):
    monkeypatch.setitem(__import__("config").CONFIG, "TRIPLE_BARRIER_AMBIGUOUS_POLICY", "stop")
    df = _base_df()
    upper = 100.0 * math.exp(0.01)
    lower = 100.0 * math.exp(-0.01)
    df.iloc[1, df.columns.get_loc("High")] = upper * 1.001
    df.iloc[1, df.columns.get_loc("Low")] = lower * 0.999
    out = apply_triple_barrier(df, volatility_col="vol", horizon=3, profit_mult=1.0, loss_mult=1.0)
    assert out.iloc[0]["tb_label"] == -1
    assert out.iloc[0]["tb_event"] == "sl"


def test_internal_volatility_is_causal_no_lookahead():
    rng = np.random.default_rng(41)
    rows = 90
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, rows)))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.5,
            "Low": close * 0.5,
            "Close": close,
            "Volume": 1000.0,
        },
        index=idx,
    )
    out = apply_triple_barrier(df, horizon=5, profit_mult=2.0, loss_mult=1.0)
    for t in [30, 45, 60]:
        expected = compute_causal_volatility(df["Close"].iloc[: t + 1]).iloc[-1]
        upper = float(out.iloc[t]["tb_upper_barrier"])
        entry = float(df.iloc[t]["Close"])
        actual = math.log(upper / entry) / 2.0
        assert math.isclose(actual, max(float(expected), 1e-6), rel_tol=0.0, abs_tol=1e-12)


def test_labels_are_stable_before_truncation_boundary():
    rows = 100
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = 100.0 + np.sin(np.arange(rows) / 3.0)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 3.0,
            "Low": close - 3.0,
            "Close": close,
            "Volume": 1000.0,
            "vol": 0.01,
        },
        index=idx,
    )
    full = apply_triple_barrier(df, volatility_col="vol", horizon=6, profit_mult=1.0, loss_mult=1.0)
    m = 70
    trunc = apply_triple_barrier(df.iloc[:m], volatility_col="vol", horizon=6, profit_mult=1.0, loss_mult=1.0)
    boundary = df.index[m - 1]
    has_full_horizon_in_trunc = full.index <= df.index[m - 1 - 6]
    stable_idx = full.index[full["tb_exit_index"].notna() & (full["tb_exit_index"] <= boundary) & has_full_horizon_in_trunc]
    common = stable_idx.intersection(trunc.index)
    pd.testing.assert_series_equal(full.loc[common, "tb_label"], trunc.loc[common, "tb_label"], check_names=False)


def test_tb_columns_exist():
    out = apply_triple_barrier(_base_df(), volatility_col="vol", horizon=3)
    assert set(TB_COLUMNS).issubset(out.columns)


def test_uniqueness_range_and_non_overlapping_event_weight():
    idx = pd.date_range("2026-01-01", periods=7, freq="h", tz="UTC")
    labeled = pd.DataFrame(index=idx)
    labeled["tb_label"] = np.nan
    labeled["tb_exit_index"] = pd.Series([pd.NA] * len(labeled), index=idx, dtype="object")
    labeled.loc[idx[0], ["tb_label", "tb_exit_index"]] = [1.0, idx[1]]
    labeled.loc[idx[3], ["tb_label", "tb_exit_index"]] = [1.0, idx[4]]
    weights = compute_uniqueness(labeled)
    assert weights.dropna().between(0.0, 1.0, inclusive="right").all()
    assert weights.loc[idx[0]] == 1.0
    assert weights.loc[idx[3]] == 1.0


def test_purged_walk_forward_cv_respects_embargo():
    df = _base_df(rows=80)
    labeled = apply_triple_barrier(df, volatility_col="vol", horizon=4, profit_mult=3.0, loss_mult=3.0)
    cv = PurgedWalkForwardCV(n_splits=4, embargo_bars=4)
    for train_idx, test_idx in cv.split(labeled):
        if len(test_idx) == 0:
            continue
        test_start_pos = int(test_idx[0])
        boundary = labeled.index[max(0, test_start_pos - 4)]
        if len(train_idx):
            train_exits = pd.to_datetime(labeled.iloc[train_idx]["tb_exit_index"], utc=True)
            assert (train_exits < boundary).all()
        assert set(train_idx).isdisjoint(set(test_idx))


def test_purged_walk_forward_cv_rejects_combined_multi_symbol_immediately():
    df = _base_df(rows=80)
    labeled = apply_triple_barrier(df, volatility_col="vol", horizon=4, profit_mult=3.0, loss_mult=3.0)
    labeled["Symbol"] = ["BTCUSDT" if i % 2 == 0 else "ETHUSDT" for i in range(len(labeled))]
    cv = PurgedWalkForwardCV(n_splits=4, embargo_bars=4)
    with pytest.raises(ValueError, match="PurgedWalkForwardCV must be run per symbol"):
        cv.split(labeled)


def test_purged_walk_forward_cv_accepts_single_symbol_column():
    df = _base_df(rows=80)
    labeled = apply_triple_barrier(df, volatility_col="vol", horizon=4, profit_mult=3.0, loss_mult=3.0)
    cv = PurgedWalkForwardCV(n_splits=4, embargo_bars=4)
    without_symbol = list(cv.split(labeled))
    labeled_with_symbol = labeled.copy()
    labeled_with_symbol["Symbol"] = "BTCUSDT"
    with_symbol = list(cv.split(labeled_with_symbol))
    assert len(without_symbol) > 0
    assert len(with_symbol) == len(without_symbol)
