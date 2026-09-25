# ==============================================================================
# TripleBarrier.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# Not used by live trading.
# ==============================================================================

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


TB_COLUMNS = [
    "tb_label",
    "tb_return",
    "tb_event",
    "tb_exit_index",
    "tb_horizon",
    "tb_upper_barrier",
    "tb_lower_barrier",
]


def _config_value(key: str, default):
    try:
        from config import CONFIG

        return CONFIG.get(key, default)
    except Exception:
        return default


def compute_causal_volatility(price: pd.Series, span: int | None = None) -> pd.Series:
    window = int(span if span is not None else _config_value("TRIPLE_BARRIER_VOL_WINDOW", 24))
    window = max(1, window)
    px = pd.to_numeric(pd.Series(price), errors="coerce")
    log_ret = np.log(px / px.shift(1))
    return log_ret.ewm(span=window, min_periods=window, adjust=False).std()


def apply_triple_barrier(
    df: pd.DataFrame,
    price_col: str = "Close",
    volatility_col: str | None = None,
    horizon: int = 24,
    profit_mult: float = 2.0,
    loss_mult: float = 1.0,
    min_return: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in ["tb_label", "tb_return", "tb_horizon", "tb_upper_barrier", "tb_lower_barrier"]:
        out[col] = np.nan
    out["tb_event"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    out["tb_exit_index"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    if out.empty:
        return out
    if price_col not in out.columns or "High" not in out.columns or "Low" not in out.columns:
        raise ValueError("apply_triple_barrier requires price_col plus High and Low columns")

    horizon = max(1, int(horizon))
    profit_mult = float(profit_mult)
    loss_mult = float(loss_mult)
    min_return = max(0.0, float(min_return))
    ambiguous_policy = str(_config_value("TRIPLE_BARRIER_AMBIGUOUS_POLICY", "stop")).strip().lower()

    close = pd.to_numeric(out[price_col], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    if volatility_col:
        if volatility_col not in out.columns:
            raise ValueError(f"volatility_col not found: {volatility_col}")
        vol = pd.to_numeric(out[volatility_col], errors="coerce")
    else:
        vol = compute_causal_volatility(close)

    index = out.index
    n = len(out)
    for i in range(n):
        if i + horizon >= n:
            continue
        entry = float(close.iloc[i]) if np.isfinite(close.iloc[i]) else np.nan
        vol_i = float(vol.iloc[i]) if np.isfinite(vol.iloc[i]) else np.nan
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(vol_i):
            continue

        vol_i = max(vol_i, 1e-6)
        if vol_i * max(profit_mult, loss_mult) < min_return:
            continue

        upper = entry * float(np.exp(profit_mult * vol_i))
        lower = entry * float(np.exp(-loss_mult * vol_i))
        label = 0
        event = "vertical"
        exit_pos = i + horizon
        tb_return = float(np.log(close.iloc[exit_pos] / entry)) if close.iloc[exit_pos] > 0 else np.nan

        for j in range(i + 1, i + horizon + 1):
            hi = float(high.iloc[j]) if np.isfinite(high.iloc[j]) else np.nan
            lo = float(low.iloc[j]) if np.isfinite(low.iloc[j]) else np.nan
            hit_upper = np.isfinite(hi) and hi >= upper
            hit_lower = np.isfinite(lo) and lo <= lower
            if hit_upper and hit_lower:
                exit_pos = j
                if ambiguous_policy == "stop":
                    label = -1
                    event = "sl"
                    tb_return = float(np.log(lower / entry))
                else:
                    label = 1
                    event = "pt"
                    tb_return = float(np.log(upper / entry))
                break
            if hit_upper:
                exit_pos = j
                label = 1
                event = "pt"
                tb_return = float(np.log(upper / entry))
                break
            if hit_lower:
                exit_pos = j
                label = -1
                event = "sl"
                tb_return = float(np.log(lower / entry))
                break

        row_key = index[i]
        out.at[row_key, "tb_label"] = float(label)
        out.at[row_key, "tb_return"] = tb_return
        out.at[row_key, "tb_event"] = event
        out.at[row_key, "tb_exit_index"] = index[exit_pos]
        out.at[row_key, "tb_horizon"] = int(horizon)
        out.at[row_key, "tb_upper_barrier"] = upper
        out.at[row_key, "tb_lower_barrier"] = lower

    return out


def _event_positions(df_labeled: pd.DataFrame) -> list[tuple[int, int]]:
    if df_labeled.empty or "tb_label" not in df_labeled.columns or "tb_exit_index" not in df_labeled.columns:
        return []
    index = pd.Index(df_labeled.index)
    out: list[tuple[int, int]] = []
    events = df_labeled["tb_label"].notna() & df_labeled["tb_exit_index"].notna()
    for pos, (_, row) in enumerate(df_labeled.iterrows()):
        if not bool(events.iloc[pos]):
            continue
        exit_pos_arr = index.get_indexer([row["tb_exit_index"]])
        if len(exit_pos_arr) and int(exit_pos_arr[0]) >= 0:
            exit_pos = int(exit_pos_arr[0])
            if exit_pos > pos:
                out.append((pos, exit_pos))
    return out


def compute_uniqueness(df_labeled: pd.DataFrame) -> pd.Series:
    weights = pd.Series(np.nan, index=df_labeled.index, name="tb_uniqueness", dtype=float)
    events = _event_positions(df_labeled)
    if not events:
        return weights

    concurrency = np.zeros(len(df_labeled), dtype=float)
    for start_pos, exit_pos in events:
        concurrency[start_pos + 1 : exit_pos + 1] += 1.0

    for start_pos, exit_pos in events:
        coverage = concurrency[start_pos + 1 : exit_pos + 1]
        valid = coverage[coverage > 0]
        weights.iloc[start_pos] = float(np.mean(1.0 / valid)) if valid.size else 1.0

    return weights.clip(lower=0.0, upper=1.0)


class PurgedWalkForwardCV:
    def __init__(self, n_splits: int = 5, embargo_bars: int | None = None):
        self.n_splits = max(1, int(n_splits))
        if embargo_bars is None:
            embargo_bars = int(_config_value("TRIPLE_BARRIER_EMBARGO_BARS", 24))
        self.embargo_bars = max(0, int(embargo_bars))

    def split(self, df_labeled: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if "Symbol" in df_labeled.columns:
            unique_symbols = pd.Series(df_labeled["Symbol"]).dropna().unique()
            if len(unique_symbols) > 1:
                raise ValueError("PurgedWalkForwardCV must be run per symbol, not on a combined multi-symbol dataset.")
        return self._iter_splits(df_labeled)

    def _iter_splits(self, df_labeled: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if df_labeled.empty or "tb_label" not in df_labeled.columns or "tb_exit_index" not in df_labeled.columns:
            return

        event_mask = df_labeled["tb_label"].notna() & df_labeled["tb_exit_index"].notna()
        event_positions = np.flatnonzero(event_mask.to_numpy())
        if event_positions.size < 2:
            return

        folds = np.array_split(event_positions, self.n_splits + 1)
        index = pd.Index(df_labeled.index)
        for fold_i in range(1, len(folds)):
            test_pos = folds[fold_i]
            if test_pos.size == 0:
                continue
            test_start_pos = int(test_pos[0])
            boundary_pos = max(0, test_start_pos - self.embargo_bars)
            purge_boundary = index[boundary_pos]

            train_candidates = event_positions[event_positions < test_start_pos]
            if train_candidates.size:
                train_exits = pd.to_datetime(df_labeled.iloc[train_candidates]["tb_exit_index"], utc=True, errors="coerce")
                boundary_ts = pd.Timestamp(purge_boundary)
                if boundary_ts.tzinfo is None:
                    boundary_ts = boundary_ts.tz_localize("UTC")
                train_pos = train_candidates[train_exits < boundary_ts]
            else:
                train_pos = np.array([], dtype=int)

            if train_pos.size:
                train_exit_values = pd.to_datetime(df_labeled.iloc[train_pos]["tb_exit_index"], utc=True, errors="coerce")
                boundary_ts = pd.Timestamp(purge_boundary)
                if boundary_ts.tzinfo is None:
                    boundary_ts = boundary_ts.tz_localize("UTC")
                assert bool((train_exit_values < boundary_ts).all())
                assert len(set(train_pos).intersection(set(test_pos))) == 0

            yield train_pos.astype(int), test_pos.astype(int)
