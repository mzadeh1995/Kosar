# ==============================================================================
# tests/test_hmm_contract.py
# ------------------------------------------------------------------------------
# Contract tests for the HMM regime gate.
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import importlib
import importlib.util
import math

import numpy as np
import pandas as pd

import hmm


REQUIRED_KEYS = {
    "hmm_enabled",
    "hmm_ok",
    "hmm_state",
    "hmm_regime",
    "hmm_confidence",
    "hmm_bull_prob",
    "hmm_neutral_prob",
    "hmm_bear_prob",
    "hmm_policy",
    "hmm_reason",
    "hmm_backend",
    "hmm_anchor_method",
    "hmm_persistence",
    "hmm_switch_margin",
    "hmm_prev_regime",
    "hmm_regime_changed",
    "hmm_regime_age_bars",
    "hmm_filtered_confidence",
    "hmm_feature_mode",
    "hmm_feature_rows",
    "hmm_feature_columns",
    "hmm_trend_col",
    "hmm_vol_col",
    "hmm_range_col",
    "hmm_mapping_method",
}


def _synthetic_ohlcv(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    drift = np.r_[np.full(rows // 3, 0.001), np.full(rows // 3, -0.001), np.zeros(rows - 2 * (rows // 3))]
    noise = rng.normal(0.0, 0.006, rows)
    close = 100.0 * np.cumprod(1.0 + drift + noise)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0015, 0.0005, rows)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0015, 0.0005, rows)))
    volume = rng.lognormal(mean=7.0, sigma=0.25, size=rows)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def _config(**overrides):
    cfg = {
        "HMM_ENABLED": True,
        "HMM_BACKEND": "pomegranate",
        "HMM_EMISSION_BACKEND": "pomegranate_normal",
        "HMM_N_STATES": 3,
        "HMM_MIN_FEATURE_ROWS": 80,
        "HMM_MAX_FEATURE_ROWS": 300,
        "HMM_MIN_CONFIDENCE": 0.50,
        "HMM_SWITCH_BLOCK_FRESH_BARS": 2,
        "HMM_FEATURE_MODE": "legacy",
        "HMM_GMM_COMPONENTS": 2,
        "HMM_GMM_MIN_COMPONENT_ROWS": 20,
        "HMM_GMM_MIN_COV": 1e-3,
        "HMM_GMM_MAX_ITER": 5,
        "HMM_GMM_TOL": 1e-3,
        "HMM_GMM_FALLBACK_TO_NORMAL": True,
        "HMM_GMM_SPLIT_COL": "fd_volatility",
        "HMM_MAP_MATCH_MAX_DIST": 2.0,
        "HMM_MAP_REBASE_MIN_SPREAD": 0.5,
    }
    cfg.update(overrides)
    return cfg


def _assert_contract(result: dict) -> None:
    assert isinstance(result, dict)
    assert REQUIRED_KEYS.issubset(result.keys())
    assert result["hmm_policy"] in {"allow", "caution", "block"}
    assert result["hmm_regime"] in {"bull", "neutral", "bear", "unknown"}
    assert result["hmm_mapping_method"] in {"matched", "score_order", "none"}

    probs = [result["hmm_bull_prob"], result["hmm_neutral_prob"], result["hmm_bear_prob"]]
    assert all(math.isfinite(float(p)) for p in probs)
    assert all(0.0 <= float(p) <= 1.0 for p in probs)
    assert math.isclose(sum(float(p) for p in probs), 1.0, rel_tol=1e-6, abs_tol=1e-6)

    for key in ["hmm_confidence", "hmm_persistence", "hmm_switch_margin", "hmm_filtered_confidence"]:
        assert math.isfinite(float(result[key]))


def test_uniform_posterior_is_neutral_caution(monkeypatch):
    def uniform_fit_predict(scaled_features, config, emission_backend):
        n = len(scaled_features)
        return np.full((n, 3), 1.0 / 3.0), None

    monkeypatch.setattr(hmm, "_fit_predict_pomegranate", uniform_fit_predict)

    result = hmm.infer_hmm_regime(
        _synthetic_ohlcv(),
        _config(HMM_MIN_REGIME_PROB_MARGIN=0.05),
        symbol="UNIFORM-POSTERIOR",
    )
    _assert_contract(result)
    assert result["hmm_ok"] is True
    assert result["hmm_regime"] == "neutral"
    assert result["hmm_policy"] == "caution"
    assert "low_information_posterior" in result["hmm_reason"]
    assert math.isclose(result["hmm_regime_prob_margin"], 0.0, abs_tol=1e-12)


def test_zero_probability_rows_are_invalid_not_uniform(monkeypatch):
    class ZeroProbModel:
        def fit(self, _input):
            return self

        def predict_proba(self, pred_input):
            if isinstance(pred_input, list):
                n = len(pred_input[0])
            elif getattr(pred_input, "ndim", 0) == 3:
                n = pred_input.shape[1]
            else:
                n = len(pred_input)
            return np.zeros((n, 3), dtype=float)

    monkeypatch.setattr(hmm, "_import_pomegranate", lambda include_gmm=False: (object, object, None, None))
    monkeypatch.setattr(hmm, "_build_pomegranate_model", lambda *args, **kwargs: ZeroProbModel())

    features = pd.DataFrame(np.random.default_rng(3).normal(size=(20, 3)), columns=["a", "b", "c"])
    probs, err = hmm._fit_predict_pomegranate(features, _config(), "pomegranate_normal")

    assert probs is None
    assert err is not None
    assert "zero_probability_rows" in err


def test_infer_hmm_regime_contract_with_synthetic_ohlcv():
    result = hmm.infer_hmm_regime(_synthetic_ohlcv(), _config(HMM_EMISSION_BACKEND="pomegranate_normal"), symbol="SYNTH")
    _assert_contract(result)


def test_infer_hmm_regime_pomegranate_gmm_backend_is_not_placeholder():
    result = hmm.infer_hmm_regime(
        _synthetic_ohlcv(),
        _config(HMM_EMISSION_BACKEND="pomegranate_gmm", HMM_GMM_FALLBACK_TO_NORMAL=False),
        symbol="SYNTH-GMM",
    )
    _assert_contract(result)
    assert result["hmm_reason"] != "pomegranate_gmm_todo"
    assert "pomegranate_gmm_todo" not in result["hmm_reason"]

    if importlib.util.find_spec("pomegranate") is not None:
        assert result["hmm_ok"] is True
        assert "emission=pomegranate_gmm" in result["hmm_reason"]
        assert "fallback=" not in result["hmm_reason"]


def test_infer_hmm_regime_gmm_falls_back_to_normal(monkeypatch):
    def forced_gmm_failure(*args, **kwargs):
        raise RuntimeError("forced_gmm_failure")

    monkeypatch.setattr(hmm, "_make_gmm_distribution", forced_gmm_failure)

    result = hmm.infer_hmm_regime(
        _synthetic_ohlcv(),
        _config(HMM_EMISSION_BACKEND="pomegranate_gmm", HMM_GMM_FALLBACK_TO_NORMAL=True),
        symbol="SYNTH-GMM-FALLBACK",
    )
    _assert_contract(result)

    if importlib.util.find_spec("pomegranate") is not None:
        assert result["hmm_ok"] is True
        assert "emission=pomegranate_gmm" in result["hmm_reason"]
        assert "fallback=pomegranate_normal" in result["hmm_reason"]
        assert "forced_gmm_failure" in result["hmm_reason"]


def test_infer_hmm_regime_insufficient_data_is_neutral_caution():
    result = hmm.infer_hmm_regime(_synthetic_ohlcv(rows=12), _config(HMM_MIN_FEATURE_ROWS=80), symbol="TOO-SHORT")
    _assert_contract(result)
    assert result["hmm_ok"] is False
    assert result["hmm_regime"] == "neutral"
    assert result["hmm_policy"] == "caution"
    assert result["hmm_reason"] == "insufficient_data"


def test_infer_hmm_regime_missing_pomegranate_does_not_crash(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if str(name).startswith("pomegranate"):
            raise ImportError("forced missing pomegranate")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    result = hmm.infer_hmm_regime(_synthetic_ohlcv(), _config(), symbol="NO-POMEGRANATE")
    _assert_contract(result)
    assert result["hmm_ok"] is False
    assert result["hmm_regime"] == "neutral"
    assert result["hmm_policy"] == "caution"
    assert "pomegranate" in result["hmm_reason"].lower()
    assert result["hmm_mapping_method"] == "none"


def _segmented_probs(rows: int, states: int) -> np.ndarray:
    states = max(2, int(states))
    low = 0.01 / max(1, states - 1)
    probs = np.full((rows, states), low, dtype=float)
    for i in range(rows):
        state = min(states - 1, int(i * states / max(1, rows)))
        probs[i, :] = low
        probs[i, state] = 0.99
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def test_state_mapping_reuses_previous_symbol_cache(monkeypatch):
    hmm._ANCHOR_STATE.clear()

    def fake_fit_predict(scaled_features, config, emission_backend):
        return _segmented_probs(len(scaled_features), int(config.get("HMM_N_STATES", 3))), None

    monkeypatch.setattr(hmm, "_fit_predict_pomegranate", fake_fit_predict)

    cfg = _config(HMM_EMISSION_BACKEND="pomegranate_gmm")
    first = hmm.infer_hmm_regime(_synthetic_ohlcv(), cfg, symbol="MAP-CONTINUITY")
    second = hmm.infer_hmm_regime(_synthetic_ohlcv(), cfg, symbol="MAP-CONTINUITY")

    _assert_contract(first)
    _assert_contract(second)
    assert first["hmm_mapping_method"] == "score_order"
    assert second["hmm_mapping_method"] == "matched"


def test_state_mapping_cache_invalidates_when_signature_changes(monkeypatch):
    hmm._ANCHOR_STATE.clear()
    columns_holder = {"columns": ["fd_return", "fd_volatility", "fd_log_range"]}

    def fake_prepare_features(df, config):
        del df, config
        rows = 120
        idx = pd.date_range("2025-03-01", periods=rows, freq="h", tz="UTC")
        trend = np.linspace(-2.0, 2.0, rows)
        data = {}
        for col in columns_holder["columns"]:
            if col == "fd_return":
                data[col] = trend
            elif col == "fd_volatility":
                data[col] = np.abs(trend) + 0.1
            elif col == "fd_log_range":
                data[col] = np.linspace(0.2, 0.5, rows)
            else:
                data[col] = np.linspace(-0.5, 0.5, rows)
        raw = pd.DataFrame(data, index=idx)
        scale = (raw.quantile(0.75) - raw.quantile(0.25)).replace(0, 1.0)
        scaled = (raw - raw.median()) / scale
        meta = {
            "hmm_feature_mode": "legacy",
            "hmm_feature_rows": len(raw),
            "hmm_feature_columns": list(raw.columns),
            "hmm_trend_col": "fd_return" if "fd_return" in raw.columns else raw.columns[0],
            "hmm_vol_col": "fd_volatility" if "fd_volatility" in raw.columns else None,
            "hmm_range_col": "fd_log_range" if "fd_log_range" in raw.columns else None,
        }
        return raw, scaled, meta, None

    def fake_fit_predict(scaled_features, config, emission_backend):
        return _segmented_probs(len(scaled_features), int(config.get("HMM_N_STATES", 3))), None

    monkeypatch.setattr(hmm, "_prepare_features", fake_prepare_features)
    monkeypatch.setattr(hmm, "_fit_predict_pomegranate", fake_fit_predict)

    base_cfg = _config(HMM_FEATURE_MODE="legacy", HMM_N_STATES=3)
    first = hmm.infer_hmm_regime(_synthetic_ohlcv(), base_cfg, symbol="MAP-INVALIDATE")
    second = hmm.infer_hmm_regime(_synthetic_ohlcv(), base_cfg, symbol="MAP-INVALIDATE")
    changed_states = hmm.infer_hmm_regime(
        _synthetic_ohlcv(),
        _config(HMM_FEATURE_MODE="legacy", HMM_N_STATES=4),
        symbol="MAP-INVALIDATE",
    )

    columns_holder["columns"] = ["fd_return", "fd_volatility", "fd_log_range", "extra_feature"]
    changed_columns = hmm.infer_hmm_regime(_synthetic_ohlcv(), base_cfg, symbol="MAP-INVALIDATE")

    _assert_contract(first)
    _assert_contract(second)
    _assert_contract(changed_states)
    _assert_contract(changed_columns)
    assert first["hmm_mapping_method"] == "score_order"
    assert second["hmm_mapping_method"] == "matched"
    assert changed_states["hmm_mapping_method"] == "score_order"
    assert changed_columns["hmm_mapping_method"] == "score_order"


def test_hmm_exception_result_includes_mapping_method(monkeypatch):
    def forced_prepare_failure(*args, **kwargs):
        raise RuntimeError("forced_prepare_failure")

    monkeypatch.setattr(hmm, "_prepare_features", forced_prepare_failure)
    result = hmm.infer_hmm_regime(_synthetic_ohlcv(), _config(), symbol="HMM-EXCEPTION")

    _assert_contract(result)
    assert result["hmm_ok"] is False
    assert result["hmm_regime"] == "neutral"
    assert result["hmm_mapping_method"] == "none"
    assert "hmm_exception" in result["hmm_reason"]
