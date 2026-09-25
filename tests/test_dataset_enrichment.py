# ==============================================================================
# tests/test_dataset_enrichment.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dataset
from config import CONFIG
from primary_features import build_xy


def _klines(index: pd.DatetimeIndex, close_values) -> pd.DataFrame:
    close = np.asarray(close_values, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(len(index), 1000.0)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "QuoteVolume": volume * close,
            "Trades": np.arange(len(index)) + 100,
            "TakerBuyBase": volume * 0.52,
            "TakerBuyQuote": volume * close * 0.52,
        },
        index=index,
    ).rename_axis("OpenTime")


def _cfg(tmp_path: Path) -> dict:
    return {
        "BINANCE_SYMBOL_MAP": {"BTC-USD": "BTCUSDT"},
        "DATASET_DATA_DIR": str(tmp_path / "binance_vision"),
        "DATASET_OUTPUT_DIR": str(tmp_path / "default_datasets"),
        "DATASET_DEFAULT_MONTHS": 1,
        "DATASET_DEFAULT_TIMEFRAME": "1h",
        "TRIPLE_BARRIER_HORIZON": 12,
        "TRIPLE_BARRIER_PROFIT_MULT": 4.0,
        "TRIPLE_BARRIER_LOSS_MULT": 2.0,
        "TRIPLE_BARRIER_VOL_WINDOW": 24,
        "HMM_FEATURE_MODE": "fractional_diff",
    }


def _write_hmm_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "backend": "pomegranate_normal",
        "close": 100.0,
        "fwd_log_return": 0.0,
        "hmm_ok": True,
        "hmm_reason": "test",
        "hmm_state_prob_max": 1.0,
        "hmm_state_prob_min": 0.0,
        "hmm_state_prob_margin": 1.0,
        "hmm_regime_prob_margin": 1.0,
        "hmm_state_prob_entropy": 0.0,
        "hmm_effective_emission_backend": "pomegranate_normal",
        "hmm_mapping_method": "score_order",
        "fit_seconds": 0.01,
    }
    pd.DataFrame([{**defaults, **row} for row in rows]).to_csv(path, index=False)


def _labeled_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(41)
    idx = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC", name="OpenTime")
    df = pd.DataFrame(index=idx)
    for col in CONFIG["PRIMARY_FEATURE_COLUMNS"]:
        df[col] = rng.normal(size=rows)
    for col in dataset.ENRICHED_FEATURE_COLUMNS:
        df[col] = rng.normal(size=rows)
    for col in dataset.HMM_FEATURE_COLUMNS:
        df[col] = rng.normal(size=rows)
    df["tb_label"] = np.tile([1.0, -1.0], rows // 2)
    df["tb_return"] = np.where(df["tb_label"] == 1.0, 0.02, -0.01)
    df["tb_uniqueness"] = 1.0
    df["tb_exit_index"] = idx + pd.Timedelta(hours=2)
    return df


def test_resample_features_use_closed_higher_timeframe_candles_only():
    idx_4h = pd.date_range("2026-01-01", periods=140, freq="h", tz="UTC")
    base_close_4h = 100.0 + np.arange(len(idx_4h)) * 0.01
    extreme_close_4h = base_close_4h.copy()
    extreme_close_4h[idx_4h.get_loc(pd.Timestamp("2026-01-05 03:00", tz="UTC"))] = 250.0

    base_features_4h = dataset.build_multitimeframe_features(_klines(idx_4h, base_close_4h))
    extreme_features_4h = dataset.build_multitimeframe_features(_klines(idx_4h, extreme_close_4h))

    before_4h_close = pd.Timestamp("2026-01-05 02:00", tz="UTC")
    at_4h_close = pd.Timestamp("2026-01-05 03:00", tz="UTC")
    assert extreme_features_4h.loc[before_4h_close, "close_z_4h"] == pytest.approx(
        base_features_4h.loc[before_4h_close, "close_z_4h"]
    )
    assert extreme_features_4h.loc[before_4h_close, "trend_4h"] == pytest.approx(
        base_features_4h.loc[before_4h_close, "trend_4h"]
    )
    assert extreme_features_4h.loc[at_4h_close, "close_z_4h"] != pytest.approx(
        base_features_4h.loc[at_4h_close, "close_z_4h"]
    )
    assert extreme_features_4h.loc[at_4h_close, "trend_4h"] != pytest.approx(
        base_features_4h.loc[at_4h_close, "trend_4h"]
    )

    idx_1d = pd.date_range("2026-01-01", periods=12 * 24, freq="h", tz="UTC")
    base_close_1d = 100.0 + np.arange(len(idx_1d)) * 0.02
    extreme_close_1d = base_close_1d.copy()
    extreme_close_1d[idx_1d.get_loc(pd.Timestamp("2026-01-10 23:00", tz="UTC"))] = 300.0

    base_features_1d = dataset.build_multitimeframe_features(_klines(idx_1d, base_close_1d))
    extreme_features_1d = dataset.build_multitimeframe_features(_klines(idx_1d, extreme_close_1d))

    before_1d_close = pd.Timestamp("2026-01-10 22:00", tz="UTC")
    at_1d_close = pd.Timestamp("2026-01-10 23:00", tz="UTC")
    assert extreme_features_1d.loc[before_1d_close, "trend_1d"] == pytest.approx(
        base_features_1d.loc[before_1d_close, "trend_1d"]
    )
    assert extreme_features_1d.loc[at_1d_close, "trend_1d"] != pytest.approx(
        base_features_1d.loc[at_1d_close, "trend_1d"]
    )


def test_resample_features_use_4h_base_close_time_for_daily_features():
    idx = pd.date_range("2026-01-01", periods=12 * 6, freq="4h", tz="UTC", name="OpenTime")
    base_close = 100.0 * np.exp(np.linspace(0.0, 0.20, len(idx)))
    extreme_close = base_close.copy()
    extreme_idx = idx.get_loc(pd.Timestamp("2026-01-10 20:00", tz="UTC"))
    extreme_close[extreme_idx] = base_close[extreme_idx] * 4.0

    base_features = dataset.build_multitimeframe_features(_klines(idx, base_close), timeframe="4h")
    extreme_features = dataset.build_multitimeframe_features(_klines(idx, extreme_close), timeframe="4h")

    before_daily_close = pd.Timestamp("2026-01-10 16:00", tz="UTC")
    at_daily_close = pd.Timestamp("2026-01-10 20:00", tz="UTC")
    assert pd.notna(base_features.loc[before_daily_close, "trend_1d"])
    assert pd.notna(base_features.loc[at_daily_close, "trend_1d"])
    assert pd.notna(extreme_features.loc[before_daily_close, "trend_1d"])
    assert pd.notna(extreme_features.loc[at_daily_close, "trend_1d"])
    assert extreme_features.loc[before_daily_close, "trend_1d"] == pytest.approx(
        base_features.loc[before_daily_close, "trend_1d"]
    )
    assert extreme_features.loc[at_daily_close, "trend_1d"] != pytest.approx(
        base_features.loc[at_daily_close, "trend_1d"]
    )


def test_4h_context_is_identity_on_4h_base_without_one_bar_lag():
    idx = pd.date_range("2026-01-01", periods=60, freq="4h", tz="UTC", name="OpenTime")
    close = 100.0 * np.exp(np.linspace(0.0, 0.30, len(idx)))
    features = dataset.build_multitimeframe_features(_klines(idx, close), timeframe="4h")
    expected = pd.Series(np.log(close / pd.Series(close, index=idx).shift(12).to_numpy()), index=idx)

    comparison = pd.DataFrame({"actual": features["trend_4h"], "expected": expected}).iloc[12:].dropna()
    assert not comparison.empty
    np.testing.assert_allclose(comparison["actual"].to_numpy(), comparison["expected"].to_numpy(), rtol=0.0, atol=1e-12)


def test_hmm_dir_with_4h_timeframe_is_rejected_before_download(tmp_path):
    with pytest.raises(ValueError, match="walk-forward.*1h"):
        dataset.build_training_dataset(
            ["BTCUSDT"],
            timeframe="4h",
            hmm_dir=tmp_path / "hmm",
            config=_cfg(tmp_path),
        )


def test_enrichment_rejects_unsupported_timeframe():
    idx = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC", name="OpenTime")
    with pytest.raises(ValueError, match="enrichment.*1h.*4h"):
        dataset.build_multitimeframe_features(_klines(idx, np.linspace(100.0, 110.0, len(idx))), timeframe="15m")


def test_hmm_join_uses_backward_asof_with_six_hour_tolerance(tmp_path):
    hmm_dir = tmp_path / "hmm"
    _write_hmm_csv(
        hmm_dir / "hmm_walkforward_BTCUSDT_1h_pomegranate_normal.csv",
        [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "hmm_regime": "bull",
                "hmm_policy": "allow",
                "hmm_confidence": 0.8,
                "hmm_bull_prob": 0.8,
                "hmm_neutral_prob": 0.1,
                "hmm_bear_prob": 0.1,
            }
        ],
    )
    target_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31 22:30", tz="UTC"),
            pd.Timestamp("2026-01-01 04:59", tz="UTC"),
            pd.Timestamp("2026-01-01 05:01", tz="UTC"),
        ],
        name="OpenTime",
    )

    out = dataset.load_hmm_walkforward_features("BTCUSDT", "1h", target_index, hmm_dir)

    assert pd.isna(out.iloc[0]["hmm_bull_prob"])
    assert out.iloc[1]["hmm_bull_prob"] == pytest.approx(0.8)
    assert pd.isna(out.iloc[2]["hmm_bull_prob"])
    assert out.iloc[1]["hmm_policy_code"] == 2


def test_hmm_regime_age_resets_on_regime_change(tmp_path):
    hmm_dir = tmp_path / "hmm"
    _write_hmm_csv(
        hmm_dir / "hmm_walkforward_BTCUSDT_1h_pomegranate_normal.csv",
        [
            {"timestamp": "2026-01-01T00:00:00+00:00", "hmm_regime": "bull", "hmm_policy": "allow", "hmm_confidence": 0.8, "hmm_bull_prob": 0.8, "hmm_neutral_prob": 0.1, "hmm_bear_prob": 0.1},
            {"timestamp": "2026-01-01T06:00:00+00:00", "hmm_regime": "bull", "hmm_policy": "allow", "hmm_confidence": 0.7, "hmm_bull_prob": 0.7, "hmm_neutral_prob": 0.2, "hmm_bear_prob": 0.1},
            {"timestamp": "2026-01-01T12:00:00+00:00", "hmm_regime": "bear", "hmm_policy": "block", "hmm_confidence": 0.9, "hmm_bull_prob": 0.1, "hmm_neutral_prob": 0.0, "hmm_bear_prob": 0.9},
            {"timestamp": "2026-01-01T18:00:00+00:00", "hmm_regime": "bear", "hmm_policy": "block", "hmm_confidence": 0.6, "hmm_bull_prob": 0.2, "hmm_neutral_prob": 0.2, "hmm_bear_prob": 0.6},
            {"timestamp": "2026-01-02T00:00:00+00:00", "hmm_regime": "neutral", "hmm_policy": "caution", "hmm_confidence": 0.5, "hmm_bull_prob": 0.25, "hmm_neutral_prob": 0.5, "hmm_bear_prob": 0.25},
        ],
    )
    target_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-12-31 23:00", tz="UTC"),
            pd.Timestamp("2026-01-01 05:00", tz="UTC"),
            pd.Timestamp("2026-01-01 11:00", tz="UTC"),
            pd.Timestamp("2026-01-01 17:00", tz="UTC"),
            pd.Timestamp("2026-01-01 23:00", tz="UTC"),
        ],
        name="OpenTime",
    )

    out = dataset.load_hmm_walkforward_features("BTCUSDT", "1h", target_index, hmm_dir)

    assert out["hmm_regime_age"].tolist() == [0, 1, 0, 1, 0]


def test_btc_context_rel_strength_matches_symbol_minus_btc_and_btc_self_is_zero():
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC", name="OpenTime")
    btc = _klines(idx, 100.0 * np.exp(np.linspace(0.0, 0.08, len(idx))))
    symbol = _klines(idx, 90.0 * np.exp(np.linspace(0.0, 0.12, len(idx))))

    symbol_features = dataset.build_causal_features(symbol)
    btc_context = dataset.build_btc_context_features(symbol_features, btc)
    expected = symbol_features["trend"] - btc_context["btc_trend_24"]
    pd.testing.assert_series_equal(btc_context["rel_strength"].dropna(), expected.dropna(), check_names=False)

    btc_features = dataset.build_causal_features(btc)
    btc_self_context = dataset.build_btc_context_features(btc_features, btc)
    assert float(btc_self_context["rel_strength"].dropna().abs().max()) <= 1e-12


def test_hmm_columns_are_not_used_by_build_xy():
    df = _labeled_frame()
    xy = build_xy(df, {"PRIMARY_FEATURE_COLUMNS": list(CONFIG["PRIMARY_FEATURE_COLUMNS"])})
    assert len(xy["X"].columns) == len(CONFIG["PRIMARY_FEATURE_COLUMNS"])
    assert not any(str(col).startswith("hmm_") for col in xy["X"].columns)


def test_output_dir_writes_per_symbol_files_without_touching_default(monkeypatch, tmp_path):
    idx = pd.date_range("2026-01-01", periods=320, freq="h", tz="UTC", name="OpenTime")
    klines = _klines(idx, 100.0 * np.exp(np.linspace(0.0, 0.05, len(idx))))
    cfg = _cfg(tmp_path)
    custom_dir = tmp_path / "custom_datasets"

    monkeypatch.setattr(dataset.binance_vision, "download_klines", lambda *args, **kwargs: klines.copy())
    monkeypatch.setattr(
        "fractional.build_fractional_hmm_features",
        lambda df, config: pd.DataFrame({"fd_return": df["Close"].pct_change().fillna(0.0)}, index=df.index),
    )

    dataset.build_training_dataset(["BTC-USD"], start="2026-01-01", end="2026-01-12", output_dir=custom_dir, config=cfg)

    assert (custom_dir / "BTCUSDT_1h.csv").exists()
    assert (custom_dir / "BTCUSDT_1h.json").exists()
    assert not (Path(cfg["DATASET_OUTPUT_DIR"]) / "BTCUSDT_1h.csv").exists()


def test_vol_ratio_stays_finite_when_daily_volatility_is_zero():
    idx = pd.date_range("2026-01-01", periods=25 * 24, freq="h", tz="UTC", name="OpenTime")
    features = dataset.build_multitimeframe_features(_klines(idx, np.full(len(idx), 100.0)))
    latest = features["vol_ratio_4h_1d"].dropna().iloc[-1]
    assert np.isfinite(latest)
    assert latest == pytest.approx(0.0)
