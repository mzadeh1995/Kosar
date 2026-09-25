# ==============================================================================
# tests/test_meta_model.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_model as mm
from config import CONFIG


def _config(**overrides) -> dict:
    config = {
        "PRIMARY_FEE_BPS_PER_SIDE": 10.0,
        "PRIMARY_SLIPPAGE_BPS_PER_SIDE": 5.0,
        "TRIPLE_BARRIER_EMBARGO_BARS": 24,
        "PRIMARY_FEATURE_COLUMNS": list(CONFIG["PRIMARY_FEATURE_COLUMNS"]),
    }
    config.update(overrides)
    return config


def _hmm_frame(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    regimes = np.resize(np.array(["bull", "neutral", "bear"], dtype=object), len(timestamps))
    probabilities = {
        "bull": (0.7, 0.2, 0.1),
        "neutral": (0.2, 0.6, 0.2),
        "bear": (0.1, 0.2, 0.7),
    }
    probs = np.asarray([probabilities[regime] for regime in regimes], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "backend": "pomegranate_normal",
            "close": 100.0 + np.arange(len(timestamps), dtype=float),
            "fwd_log_return": 0.001,
            "hmm_ok": True,
            "hmm_regime": regimes,
            "hmm_policy": "allow",
            "hmm_confidence": probs.max(axis=1),
            "hmm_bull_prob": probs[:, 0],
            "hmm_neutral_prob": probs[:, 1],
            "hmm_bear_prob": probs[:, 2],
        }
    )


def _write_bundle(
    root: Path,
    *,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    rows_per_fold: int = 2,
    include_hmm: bool = True,
    include_uniqueness: bool = True,
    dataset_extras: dict[str, object] | None = None,
) -> dict:
    primary_dir = root / "primary"
    dataset_dir = root / "datasets"
    external_dir = root / "external"
    hmm_dir = root / "hmm"
    for directory in [primary_dir, dataset_dir, external_dir, hmm_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for symbol_number, symbol in enumerate(symbols):
        total = rows_per_fold * 5
        open_time = pd.date_range(
            "2024-01-01",
            periods=total,
            freq="4h",
            tz="UTC",
            name="OpenTime",
        )
        position = np.arange(total)
        folds = np.repeat(np.arange(1, 6), rows_per_fold)
        p_primary = 0.56 + 0.38 * ((position + symbol_number) % 11) / 10.0
        tb_return = np.where((position + symbol_number) % 2 == 0, 0.018, -0.008)
        y = ((position + symbol_number) % 3 == 0).astype(int)

        oof = pd.DataFrame(
            {
                "OpenTime": open_time,
                "fold": folds,
                "p_primary": p_primary,
                "y": y,
                "tb_return": tb_return,
            }
        )
        if include_uniqueness:
            oof["tb_uniqueness"] = 0.5 + 0.01 * ((position + symbol_number) % 20)
        oof.to_csv(primary_dir / f"primary_oof_{symbol}_4h.csv", index=False)

        dataset = pd.DataFrame(
            {
                "OpenTime": open_time,
                "volatility": 0.01 + 0.0001 * position,
                "log_range": 0.02 + 0.0002 * position,
                "tb_exit_index": open_time + pd.Timedelta(hours=8),
            }
        )
        for column, value in (dataset_extras or {}).items():
            dataset[column] = value
        dataset.to_csv(dataset_dir / f"{symbol}_4h.csv", index=False)

        decision_time = open_time + pd.Timedelta(hours=4)
        funding = pd.DataFrame(
            {
                "settlement_time": decision_time,
                "funding_rate": 0.0001,
                "funding_z": np.sin(position / 7.0),
                "funding_extreme_pos": (position % 9 == 0).astype(int),
            }
        )
        funding.to_csv(external_dir / f"funding_{symbol}.csv", index=False)

        vpin = pd.DataFrame(
            {
                "timestamp": decision_time,
                "vpin": 0.2 + 0.001 * (position % 30),
                "vpin_z": np.cos(position / 9.0),
                "bucket_capacity": 1000.0,
            }
        )
        vpin.to_csv(external_dir / f"vpin_{symbol}_1h.csv", index=False)

        if include_hmm:
            hmm = _hmm_frame(pd.DatetimeIndex(decision_time))
            hmm["symbol"] = symbol
            hmm.to_csv(
                hmm_dir / f"hmm_walkforward_{symbol}_1h_pomegranate_normal.csv",
                index=False,
            )

    return {
        "root": root,
        "symbols": list(symbols),
        "primary_dir": primary_dir,
        "dataset_dir": dataset_dir,
        "external_dir": external_dir,
        "hmm_dir": hmm_dir if include_hmm else None,
    }


def _assemble(bundle: dict, **overrides) -> dict:
    arguments = {
        "symbols": bundle["symbols"],
        "dataset_dir": bundle["dataset_dir"],
        "external_dir": bundle["external_dir"],
        "timeframe": "4h",
        "hmm_dir": bundle["hmm_dir"],
        "primary_dir": bundle["primary_dir"],
        "is_real_data_run": False,
        "config": _config(),
    }
    arguments.update(overrides)
    return mm.build_meta_dataset(**arguments)


def _walk_events(*, rows_per_fold: int = 80, all_negative: bool = False) -> pd.DataFrame:
    total = rows_per_fold * 5
    open_time = pd.date_range("2024-01-01", periods=total, freq="4h", tz="UTC")
    position = np.arange(total)
    tb_return = (
        np.full(total, -0.01)
        if all_negative
        else np.where(position % 2 == 0, 0.018, -0.008)
    )
    return pd.DataFrame(
        {
            "OpenTime": open_time,
            "decision_ts": open_time + pd.Timedelta(hours=4),
            "tb_exit_index": open_time + pd.Timedelta(hours=8),
            "fold": np.repeat(np.arange(1, 6), rows_per_fold),
            "y": (position % 3 == 0).astype(int),
            "meta_y": ((tb_return - mm.COST_ROUND_TRIP) > 0.0).astype(int),
            "p_primary": 0.40 + 0.05 * (position % 11),
            "mock_p_meta": 0.42 + 0.05 * ((position * 3) % 10),
            "tb_return": tb_return,
            "tb_uniqueness": 0.5 + 0.01 * (position % 20),
            "symbol": "BTCUSDT",
        }
    )


def _run_fake_walk(monkeypatch: pytest.MonkeyPatch, events: pd.DataFrame) -> dict:
    def fake_fit(frame, _features, _categorical):
        return object(), {"train_event_count": int(len(frame)), "feature_importance": {}}

    def fake_predict(_model, frame, _features, _categorical):
        return frame["mock_p_meta"].to_numpy(dtype=float)

    monkeypatch.setattr(mm, "fit_meta_catboost", fake_fit)
    monkeypatch.setattr(mm, "_predict_meta", fake_predict)
    return mm.run_meta_walk_forward(
        events,
        ["mock_p_meta"],
        [],
        is_real_data_run=False,
        include_hmm=False,
    )


def _managed_snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in mm._managed_artifacts(directory)}


def _all_file_snapshot(directory: Path) -> dict[str, tuple[int, str]]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): (int(path.stat().st_size), mm._sha256(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _tree_snapshot(directory: Path) -> list[tuple[str, str, bytes | None]]:
    if not directory.exists():
        return []
    snapshot: list[tuple[str, str, bytes | None]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: str(item.relative_to(directory))):
        relative = str(path.relative_to(directory))
        snapshot.append((relative, "dir", None) if path.is_dir() else (relative, "file", path.read_bytes()))
    return snapshot


def _stable_summary(summary: dict) -> dict:
    stable = copy.deepcopy(summary)
    stable.pop("created_at", None)
    stable.pop("output_artifacts", None)
    return stable


class _FakeSavedModel:
    def save_model(self, path: str) -> None:
        Path(path).write_bytes(b"synthetic-catboost-model")


class _FailingSavedModel:
    def save_model(self, _path: str) -> None:
        raise RuntimeError("synthetic staging failure")


def _transaction_events(symbol: str = "BTCUSDT") -> pd.DataFrame:
    open_time = pd.Timestamp("2025-01-01T00:00:00Z")
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "OpenTime": open_time,
                "decision_ts": open_time + pd.Timedelta(hours=4),
                "fold": 2,
                "y": 1,
                "meta_y": 1,
                "p_primary": 0.8,
                "p_meta": 0.7,
                "tb_return": 0.02,
                "tb_exit_index": open_time + pd.Timedelta(hours=8),
                "tb_uniqueness": 0.75,
                "tau_meta_used": 0.55,
                "tau_baseline_used": 0.55,
                "tau_meta_forced_used": 0.55,
                "tau_baseline_forced_used": 0.55,
                "traded_meta": True,
                "traded_baseline": True,
                "meta_eval_status": "evaluated",
            }
        ]
    )


def _seed_managed_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_4h.cbm").write_bytes(b"old-model")
    (output_dir / "meta_oof_BTCUSDT_4h.csv").write_bytes(b"old-oof")
    (output_dir / "meta_training_summary_4h.json").write_bytes(b'{"old": true}')
    (output_dir / "unmanaged.txt").write_bytes(b"keep-me")


@pytest.fixture(scope="module")
def trained_pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("meta_ordering")
    bundle = _write_bundle(
        root / "inputs",
        symbols=("BTCUSDT", "ETHUSDT"),
        rows_per_fold=60,
    )
    output_dir = root / "output"
    canonical_before = _all_file_snapshot(mm.CANONICAL_OUTPUT_DIR)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(mm, "CANONICAL_PRIMARY_DIR", bundle["primary_dir"].resolve())
    patcher.setitem(mm.META_CATBOOST_PARAMS, "iterations", 8)
    patcher.setitem(mm.META_CATBOOST_PARAMS, "allow_writing_files", False)
    try:
        first_return = mm.run_meta_training(
            symbols=bundle["symbols"],
            dataset_dir=bundle["dataset_dir"],
            external_dir=bundle["external_dir"],
            hmm_dir=bundle["hmm_dir"],
            output_dir=output_dir,
            config=_config(),
        )
        first_summary = json.loads((output_dir / "meta_training_summary_4h.json").read_text())
        first_csv = {
            symbol: (output_dir / f"meta_oof_{symbol}_4h.csv").read_bytes()
            for symbol in bundle["symbols"]
        }
        first_frames = {
            symbol: pd.read_csv(output_dir / f"meta_oof_{symbol}_4h.csv")
            for symbol in bundle["symbols"]
        }

        second_return = mm.run_meta_training(
            symbols=list(reversed(bundle["symbols"])),
            dataset_dir=bundle["dataset_dir"],
            external_dir=bundle["external_dir"],
            hmm_dir=bundle["hmm_dir"],
            output_dir=output_dir,
            config=_config(),
        )
        second_summary = json.loads((output_dir / "meta_training_summary_4h.json").read_text())
        second_csv = {
            symbol: (output_dir / f"meta_oof_{symbol}_4h.csv").read_bytes()
            for symbol in bundle["symbols"]
        }
        second_frames = {
            symbol: pd.read_csv(output_dir / f"meta_oof_{symbol}_4h.csv")
            for symbol in bundle["symbols"]
        }
    finally:
        patcher.undo()

    return {
        "root": root,
        "output_dir": output_dir,
        "symbols": bundle["symbols"],
        "first_return": first_return,
        "second_return": second_return,
        "first_summary": first_summary,
        "second_summary": second_summary,
        "first_csv": first_csv,
        "second_csv": second_csv,
        "first_frames": first_frames,
        "second_frames": second_frames,
        "canonical_unchanged": canonical_before == _all_file_snapshot(mm.CANONICAL_OUTPUT_DIR),
        "catboost_info": list(root.rglob("catboost_info")),
    }


def test_01_timeframe_lock_rejects_every_non_4h_entrypoint(tmp_path):
    with pytest.raises(ValueError, match="timeframe='4h'"):
        mm.build_meta_dataset([], tmp_path, tmp_path, timeframe="1h")
    with pytest.raises(ValueError, match="timeframe='4h'"):
        mm.resolve_run_identity([], tmp_path, tmp_path, None, tmp_path / "out", "1d")


def test_02_mixed_timestamp_parser_counts_drops_and_rejects_empty():
    parsed, invalid = mm.parse_mixed_utc(
        pd.Series(["2025-01-01T00:00:00Z", "2025-01-01T04:00:00.123Z", "broken"]),
        "mixed test",
    )
    assert invalid == 1
    assert parsed.notna().tolist() == [True, True, False]
    assert isinstance(parsed.dtype, pd.DatetimeTZDtype)
    assert str(parsed.dt.tz) == "UTC"
    with pytest.raises(ValueError, match="no valid timestamps"):
        mm.parse_mixed_utc(pd.Series(["broken", None]), "empty test")


def test_03_funding_join_is_backward_exact_and_tolerance_bounded(tmp_path):
    path = tmp_path / "funding.csv"
    pd.DataFrame(
        {
            "settlement_time": ["2025-01-01T11:00:00Z", "2025-01-01T13:00:00.001Z"],
            "funding_z": [1.0, 9.0],
            "funding_extreme_pos": [0, 1],
        }
    ).to_csv(path, index=False)
    source, _ = mm._prepare_numeric_source(
        path,
        "funding",
        "settlement_time",
        ["funding_z", "funding_extreme_pos"],
    )
    events = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                ["2025-01-01T12:00:00Z", "2025-01-01T13:00:00.001Z", "2025-01-01T22:00:01Z"],
                utc=True,
                format="mixed",
            )
        }
    )
    merged, diagnostics = mm._merge_asof_source(
        events,
        source,
        "funding",
        ["funding_z", "funding_extreme_pos"],
        pd.Timedelta(hours=8, minutes=1),
    )
    assert merged.loc[0, "funding_z"] == 1.0
    assert merged.loc[1, "funding_z"] == 9.0
    assert pd.isna(merged.loc[2, "funding_z"])
    assert diagnostics == {
        "matched_count": 2,
        "outside_tolerance_count": 1,
        "matched_but_feature_nan_count": 0,
    }
    assert (merged.loc[merged["_funding_matched"], "_funding_source_ts"] <= merged.loc[merged["_funding_matched"], "decision_ts"]).all()


def test_04_vpin_hmm_causality_and_hmm_true_false_validation_scope(tmp_path):
    vpin_path = tmp_path / "vpin.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T10:00:00Z", "2025-01-01T13:00:00Z"],
            "vpin": [0.2, 0.9],
            "vpin_z": [-0.5, 1.5],
        }
    ).to_csv(vpin_path, index=False)
    vpin, _ = mm._prepare_numeric_source(vpin_path, "vpin", "timestamp", ["vpin", "vpin_z"])
    vpin_events = pd.DataFrame(
        {"decision_ts": pd.to_datetime(["2025-01-01T12:00:00Z", "2025-01-01T13:00:00Z"], utc=True)}
    )
    vpin_joined, _ = mm._merge_asof_source(
        vpin_events,
        vpin,
        "vpin",
        ["vpin", "vpin_z"],
        pd.Timedelta(hours=2),
    )
    assert vpin_joined["vpin"].tolist() == [0.2, 0.9]

    hmm_path = tmp_path / "hmm.csv"
    hmm = _hmm_frame(pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC"))
    hmm.loc[0, "hmm_regime"] = "bull"
    hmm.loc[1, ["hmm_ok", "hmm_regime", "hmm_policy"]] = [False, "invalid", "invalid"]
    hmm.loc[1, ["hmm_bull_prob", "hmm_neutral_prob", "hmm_bear_prob", "hmm_confidence"]] = [9.0, -3.0, 7.0, 2.0]
    hmm.loc[2, "hmm_regime"] = "bull"
    hmm.to_csv(hmm_path, index=False)
    prepared, diagnostics = mm._prepare_hmm_source(hmm_path)
    assert diagnostics["hmm_ok_false_count"] == 1
    assert prepared.loc[1, "hmm_regime"] == "missing"
    assert prepared.loc[1, mm.META_HMM_FEATURE_COLUMNS[1:]].isna().all()
    assert prepared.loc[2, "hmm_regime_age_hours"] == 8.0

    decisions = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                ["2025-01-01T03:00:00Z", "2025-01-01T04:00:00Z", "2025-01-01T08:00:00Z"],
                utc=True,
            )
        }
    )
    joined, join_diagnostics = mm._merge_asof_source(
        decisions,
        prepared,
        "hmm",
        mm.META_HMM_FEATURE_COLUMNS,
        pd.Timedelta(hours=6),
    )
    assert joined["hmm_regime"].tolist() == ["bull", "missing", "bull"]
    assert join_diagnostics["matched_but_feature_nan_count"] == 1
    assert (joined["_hmm_source_ts"] <= joined["decision_ts"]).all()

    valid_true = _hmm_frame(pd.date_range("2025-01-02", periods=1, freq="4h", tz="UTC"))
    invalid_cases = [
        ("hmm_policy", "invalid"),
        ("hmm_regime", "invalid"),
        ("hmm_bull_prob", 0.95),
        ("hmm_confidence", 1.5),
    ]
    for column, value in invalid_cases:
        candidate = valid_true.copy()
        candidate.loc[0, column] = value
        candidate.to_csv(hmm_path, index=False)
        with pytest.raises(ValueError):
            mm._prepare_hmm_source(hmm_path)


def test_05_hmm_ok_parser_accepts_only_explicit_representations():
    assert mm.parse_hmm_ok("False") is False
    assert mm.parse_hmm_ok("1") is True
    assert mm.parse_hmm_ok(np.int64(0)) is False
    assert mm.parse_hmm_ok(np.bool_(True)) is True
    with pytest.raises(ValueError, match="Invalid hmm_ok"):
        mm.parse_hmm_ok(np.nan)
    with pytest.raises(ValueError, match="Invalid hmm_ok"):
        mm.parse_hmm_ok("yes")


def test_06_purge_embargo_boundary_and_synthetic_fallback_contract():
    eval_start = pd.Timestamp("2025-01-10T00:00:00Z")
    frame = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True
            ),
            "tb_exit_index": pd.to_datetime(
                ["2025-01-05T19:00:00Z", "2025-01-05T20:00:00Z"], utc=True
            ),
            "fold": [3, 3],
        }
    )
    kept, metadata = mm.purge_training_events(frame, eval_start, is_real_data_run=False)
    assert kept.index.tolist() == [0]
    assert metadata["purge_removed_count"] == 1
    assert metadata["purge_boundary_ts"] == pd.Timestamp("2025-01-06T00:00:00Z")

    missing = frame.iloc[[0]].drop(columns="tb_exit_index")
    fallback_kept, fallback_metadata = mm.purge_training_events(
        missing,
        eval_start,
        is_real_data_run=False,
    )
    assert fallback_metadata["tb_exit_index_fallback_count"] == 1
    assert len(fallback_kept) == 1
    with pytest.raises(ValueError, match="forbids tb_exit_index fallback"):
        mm.purge_training_events(missing, eval_start, is_real_data_run=True)


def test_07_dataset_raw_empty_ends_are_allowed_but_joined_exits_are_strict(tmp_path):
    open_time = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
    path = tmp_path / "dataset.csv"
    dataset = pd.DataFrame(
        {
            "OpenTime": open_time,
            "volatility": [0.1, 0.2, 0.3],
            "log_range": [0.2, 0.3, 0.4],
            "tb_exit_index": [pd.NaT, open_time[1] + pd.Timedelta(hours=8), pd.NaT],
        }
    )
    dataset.to_csv(path, index=False)
    sliced, diagnostics = mm._load_dataset_slice(path)
    assert diagnostics["raw_tb_exit_missing_count"] == 2
    joined, _ = mm._join_dataset_events(pd.DataFrame({"OpenTime": [open_time[1]]}), sliced, path)
    assert joined.loc[0, "tb_exit_index"] == open_time[1] + pd.Timedelta(hours=8)

    with pytest.raises(ValueError, match="require tb_exit_index"):
        mm._join_dataset_events(pd.DataFrame({"OpenTime": [open_time[0]]}), sliced, path)

    for invalid_exit in [open_time[1] + pd.Timedelta(hours=3), open_time[1] + pd.Timedelta(hours=97)]:
        invalid = dataset.copy()
        invalid.loc[1, "tb_exit_index"] = invalid_exit
        invalid.to_csv(path, index=False)
        invalid_slice, _ = mm._load_dataset_slice(path)
        with pytest.raises(ValueError, match=r"outside \[OpenTime\+4h, OpenTime\+96h\]"):
            mm._join_dataset_events(
                pd.DataFrame({"OpenTime": [open_time[1]]}),
                invalid_slice,
                path,
            )


def test_08_meta_target_spw_and_config_derived_cost_embargo(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle", rows_per_fold=2)
    assembly = _assemble(bundle)
    events = assembly["events"]
    expected = ((events["tb_return"] - 0.003) > 0.0).astype(int)
    pd.testing.assert_series_equal(events["meta_y"], expected, check_names=False)
    assert "y" not in assembly["X"].columns
    assert "meta_y" not in assembly["X"].columns

    ratio, positive, negative = mm.weighted_scale_pos_weight(
        events["meta_y"], events["tb_uniqueness"]
    )
    assert positive == pytest.approx(events.loc[events["meta_y"].eq(1), "tb_uniqueness"].sum())
    assert negative == pytest.approx(events.loc[events["meta_y"].eq(0), "tb_uniqueness"].sum())
    assert ratio == pytest.approx(negative / positive)
    assert mm.derive_cost_and_embargo(_config()) == (0.003, pd.Timedelta(hours=96))
    with pytest.raises(ValueError, match="cost contract"):
        mm.derive_cost_and_embargo(_config(PRIMARY_FEE_BPS_PER_SIDE=11.0))
    with pytest.raises(ValueError, match="embargo contract"):
        mm.derive_cost_and_embargo(_config(TRIPLE_BARRIER_EMBARGO_BARS=23))


def test_09_dataset_slice_blocks_hmm_collision_and_extra_selected_column(tmp_path, monkeypatch):
    bundle = _write_bundle(
        tmp_path / "bundle",
        rows_per_fold=2,
        dataset_extras={"hmm_bull_prob": 999.0, "trend": 123.0},
    )
    hmm_path = bundle["hmm_dir"] / "hmm_walkforward_BTCUSDT_1h_pomegranate_normal.csv"
    hmm = pd.read_csv(hmm_path)
    hmm["hmm_bull_prob"] = 0.2
    hmm["hmm_neutral_prob"] = 0.5
    hmm["hmm_bear_prob"] = 0.3
    hmm.to_csv(hmm_path, index=False)
    assembly = _assemble(bundle)
    assert assembly["X"]["hmm_bull_prob"].eq(0.2).all()
    assert assembly["metadata"]["source_diagnostics"]["BTCUSDT"]["dataset"]["selected_schema"] == [
        "OpenTime",
        "volatility",
        "log_range",
        "tb_exit_index",
    ]

    dataset_path = bundle["dataset_dir"] / "BTCUSDT_4h.csv"
    monkeypatch.setattr(mm, "DATASET_COLUMNS", ["volatility", "log_range", "tb_exit_index", "trend"])
    with pytest.raises(ValueError, match="dataset slice contract"):
        mm._load_dataset_slice(dataset_path)


def test_10_hmm_leakage_columns_and_non_whitelisted_features_are_rejected(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle", rows_per_fold=2)
    assembly = _assemble(bundle)
    expected = [*mm.META_FEATURE_COLUMNS, *mm.META_HMM_FEATURE_COLUMNS]
    assert assembly["feature_columns"] == expected
    assert list(assembly["X"].columns) == expected
    assert "fwd_log_return" not in assembly["X"].columns
    assert "close" not in assembly["X"].columns

    bad = assembly["X"].assign(trend=1.0)
    with pytest.raises(ValueError, match="Primary-model features are forbidden"):
        mm._guard_feature_whitelist(bad, [*expected, "trend"], _config())


def test_11_hmm_missing_sentinel_and_age_survive_sort_and_dedup(tmp_path):
    path = tmp_path / "hmm.csv"
    timestamps = pd.to_datetime(
        [
            "2025-01-01T08:00:00Z",
            "2025-01-01T04:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-01-01T04:00:00Z",
        ],
        utc=True,
    )
    hmm = _hmm_frame(pd.DatetimeIndex(timestamps))
    hmm["hmm_regime"] = ["bull", "bear", "bull", "bull"]
    hmm.to_csv(path, index=False)
    prepared, diagnostics = mm._prepare_hmm_source(path)
    assert diagnostics["deduplicated_count"] == 1
    assert prepared["_hmm_source_ts"].is_monotonic_increasing
    assert prepared["hmm_regime"].tolist() == ["bull", "bull", "bull"]
    assert prepared["hmm_regime_age_hours"].tolist() == [0.0, 4.0, 8.0]

    bundle = _write_bundle(tmp_path / "bundle", rows_per_fold=1)
    hmm_path = bundle["hmm_dir"] / "hmm_walkforward_BTCUSDT_1h_pomegranate_normal.csv"
    future = _hmm_frame(pd.date_range("2030-01-01", periods=2, freq="4h", tz="UTC"))
    future.to_csv(hmm_path, index=False)
    assembly = _assemble(bundle)
    assert assembly["X"]["hmm_regime"].eq("missing").all()
    assert assembly["X"][mm.META_HMM_FEATURE_COLUMNS[1:]].isna().all().all()


def test_12_threshold_history_is_past_only_and_fold_one_is_excluded(monkeypatch):
    events = _walk_events()
    result = _run_fake_walk(monkeypatch, events)
    for arm in ["meta", "baseline", "meta_forced", "baseline_forced"]:
        selection = result["threshold_selection"][arm]["2"]
        assert selection["selected_tau"] == 0.55
        assert all(candidate["n_trades"] == 0 for candidate in selection["candidates"])

    fold_two = result["events"].loc[result["events"]["fold"].eq(2)]
    expected = int((fold_two["p_primary"] >= 0.55).sum())
    candidates = result["threshold_selection"]["baseline"]["3"]["candidates"]
    tau_055 = next(row for row in candidates if row["tau"] == 0.55)
    assert tau_055["n_trades"] == expected
    assert tau_055["n_trades"] < int((result["events"]["p_primary"] >= 0.55).sum())


def test_13_abstain_no_eligible_and_tie_break_choose_higher_tau():
    negative = pd.DataFrame(
        {"score": np.full(40, 0.6), "_net": np.full(40, -0.01), "tb_uniqueness": 1.0}
    )
    abstain = mm.select_threshold(
        negative,
        "score",
        [0.5, 0.6],
        first_evaluated_fold=False,
        forced=False,
    )
    assert abstain["selected_tau"] == mm.ABSTAIN_THRESHOLD
    assert abstain["selection_reason"] == "abstain_non_positive_history"
    assert int((negative["score"] >= abstain["selected_tau"]).sum()) == 0

    too_small = negative.head(20)
    no_eligible = mm.select_threshold(
        too_small,
        "score",
        [0.5, 0.6],
        first_evaluated_fold=False,
        forced=False,
    )
    assert no_eligible["selected_tau"] == mm.ABSTAIN_THRESHOLD
    assert no_eligible["selection_reason"] == "no_eligible_real_threshold"

    tied = pd.DataFrame(
        {
            "score": [0.5] * 30 + [0.6] * 30,
            "_net": [5e-13 / 30.0] * 30 + [1.0 / 30.0] * 30,
            "tb_uniqueness": 1.0,
        }
    )
    tie_result = mm.select_threshold(
        tied,
        "score",
        [0.5, 0.6],
        first_evaluated_fold=False,
        forced=False,
    )
    assert tie_result["selected_tau"] == 0.6


def test_14_forced_and_matched_n_diagnostic_arms_are_independent(monkeypatch):
    negative = pd.DataFrame(
        {"score": np.full(40, 0.6), "_net": np.full(40, -0.01), "tb_uniqueness": 1.0}
    )
    forced = mm.select_threshold(
        negative,
        "score",
        [0.5, 0.6],
        first_evaluated_fold=False,
        forced=True,
    )
    assert forced["selected_tau"] == 0.6
    assert forced["selection_reason"] == "forced_best_eligible"
    assert int((negative["score"] >= forced["selected_tau"]).sum()) == 40

    fallback = mm.select_threshold(
        negative.head(20),
        "score",
        [0.5, 0.6],
        first_evaluated_fold=False,
        forced=True,
    )
    assert fallback["selected_tau"] == 0.55
    assert fallback["selection_reason"] == "forced_fallback"

    tied = pd.DataFrame(
        {
            "p_meta": [0.9, 0.8, 0.8, 0.7],
            "decision_ts": pd.to_datetime(
                ["2025-01-01T04:00:00Z", "2025-01-01T08:00:00Z", "2025-01-01T04:00:00Z", "2025-01-01T00:00:00Z"],
                utc=True,
            ),
            "symbol": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BTCUSDT"],
        },
        index=[10, 11, 12, 13],
    )
    assert mm._top_n_meta_mask(tied, 2).loc[[10, 11, 12, 13]].tolist() == [True, False, True, False]

    result = _run_fake_walk(monkeypatch, _walk_events(all_negative=True))
    fold_three = result["folds"]["3"]
    assert fold_three["taus"]["baseline"] == mm.ABSTAIN_THRESHOLD
    assert fold_three["matched_n"]["used_forced_baseline_fallback"] is True
    assert fold_three["realized"]["baseline"]["n_trades"] == 0
    assert fold_three["matched_n"]["n"] == fold_three["realized"]["baseline_forced"]["n_trades"]
    assert fold_three["realized"]["meta_matched_n"]["n_trades"] == fold_three["matched_n"]["n"]


def test_15_meta_oof_schema_and_fold_one_contract(trained_pair):
    assert mm.META_OOF_COLUMNS == [
        "OpenTime",
        "decision_ts",
        "fold",
        "y",
        "meta_y",
        "p_primary",
        "p_meta",
        "tb_return",
        "tb_exit_index",
        "tb_uniqueness",
        "tau_meta_used",
        "tau_baseline_used",
        "tau_meta_forced_used",
        "tau_baseline_forced_used",
        "traded_meta",
        "traded_baseline",
        "meta_eval_status",
    ]
    for frame in trained_pair["second_frames"].values():
        assert list(frame.columns) == mm.META_OOF_COLUMNS
        fold_one = frame.loc[frame["fold"].eq(1)]
        assert fold_one["p_meta"].isna().all()
        assert fold_one[
            [
                "tau_meta_used",
                "tau_baseline_used",
                "tau_meta_forced_used",
                "tau_baseline_forced_used",
            ]
        ].isna().all().all()
        assert not fold_one[["traded_meta", "traded_baseline"]].any().any()
        assert fold_one["meta_eval_status"].eq("not_evaluated_no_prior_fold").all()
        assert frame.loc[frame["fold"].between(2, 5), "meta_eval_status"].eq("evaluated").all()


def test_16_identity_guards_resolve_aliases_and_preserve_canonical_output(tmp_path, trained_pair):
    canonical_before = _all_file_snapshot(mm.CANONICAL_OUTPUT_DIR)
    with pytest.raises(ValueError, match="Non-canonical input identity"):
        mm.resolve_run_identity(
            ["BTCUSDT"],
            tmp_path / "datasets",
            tmp_path / "external",
            None,
            mm.CANONICAL_OUTPUT_DIR,
            "4h",
        )

    dataset_alias = mm.CANONICAL_DATASET_DIR / ".." / mm.CANONICAL_DATASET_DIR.name
    external_alias = mm.CANONICAL_EXTERNAL_DIR / ".." / mm.CANONICAL_EXTERNAL_DIR.name
    hmm_alias = mm.CANONICAL_HMM_DIR / ".." / mm.CANONICAL_HMM_DIR.name
    identity = mm.resolve_run_identity(
        list(reversed(mm.CANONICAL_SYMBOLS)),
        dataset_alias,
        external_alias,
        hmm_alias,
        tmp_path / "dryrun",
        "4h",
    )
    assert identity["is_real_data_run"] is True
    assert identity["is_canonical_output"] is False

    command = [
        sys.executable,
        str(mm.META_PROJECT_ROOT / "analytics" / "train_meta.py"),
        "--symbols",
        "BTCUSDT",
        "--dataset-dir",
        str(tmp_path / "datasets"),
        "--external-dir",
        str(tmp_path / "external"),
    ]
    completed = subprocess.run(
        command,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert completed.returncode != 0
    assert "Non-canonical input identity" in completed.stderr
    assert canonical_before == _all_file_snapshot(mm.CANONICAL_OUTPUT_DIR)
    assert trained_pair["canonical_unchanged"] is True


def test_17_reversed_symbol_order_produces_identical_csv_and_stable_summary(trained_pair):
    assert trained_pair["first_return"]["symbols"] == trained_pair["symbols"]
    assert trained_pair["second_return"]["symbols"] == trained_pair["symbols"]
    assert trained_pair["first_csv"] == trained_pair["second_csv"]
    assert _stable_summary(trained_pair["first_summary"]) == _stable_summary(
        trained_pair["second_summary"]
    )


def test_18_identical_training_reproduces_p_meta_exactly(trained_pair):
    for symbol in trained_pair["symbols"]:
        first = trained_pair["first_frames"][symbol]["p_meta"].to_numpy(dtype=float)
        second = trained_pair["second_frames"][symbol]["p_meta"].to_numpy(dtype=float)
        np.testing.assert_array_equal(first, second)


def test_19_nan_inf_and_weight_fallback_guards(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle", rows_per_fold=1)
    funding_path = bundle["external_dir"] / "funding_BTCUSDT.csv"
    funding = pd.read_csv(funding_path)
    funding.loc[0, "funding_z"] = np.inf
    funding.to_csv(funding_path, index=False)
    assembly = _assemble(bundle)
    assert len(assembly["events"]) == 5
    assert pd.isna(assembly["X"].iloc[0]["funding_z"])
    assert assembly["metadata"]["inf_to_nan_count_by_feature"]["funding_z"] == 1

    original_oof = pd.read_csv(bundle["primary_dir"] / "primary_oof_BTCUSDT_4h.csv")
    critical_path = tmp_path / "critical.csv"
    critical = original_oof.copy()
    critical.loc[0, "tb_return"] = np.inf
    critical.to_csv(critical_path, index=False)
    with pytest.raises(ValueError, match="finite numeric"):
        mm._load_primary_oof(critical_path, "BTCUSDT", False)

    missing_weight_path = tmp_path / "missing_weight.csv"
    original_oof.drop(columns="tb_uniqueness").to_csv(missing_weight_path, index=False)
    loaded, _, fallback_count = mm._load_primary_oof(missing_weight_path, "BTCUSDT", False)
    assert loaded["tb_uniqueness"].eq(1.0).all()
    assert fallback_count == len(original_oof)
    with pytest.raises(ValueError, match="requires tb_uniqueness"):
        mm._load_primary_oof(missing_weight_path, "BTCUSDT", True)

    with pytest.raises(ValueError, match="Both weighted classes"):
        mm.weighted_scale_pos_weight(pd.Series([1, 1]), pd.Series([1.0, 1.0]))
    with pytest.raises(ValueError, match="weights must be finite"):
        mm.weighted_scale_pos_weight(pd.Series([0, 1]), pd.Series([1.0, np.inf]))


def test_20_missing_input_preflight_fails_before_any_csv_read(tmp_path, monkeypatch):
    def forbidden_read(*_args, **_kwargs):
        raise AssertionError("preflight must fail before pd.read_csv")

    monkeypatch.setattr(mm.pd, "read_csv", forbidden_read)
    with pytest.raises(FileNotFoundError, match="Missing required v48 input files"):
        mm.preflight_input_files(
            ["BTCUSDT"],
            tmp_path / "primary",
            tmp_path / "dataset",
            tmp_path / "external",
            tmp_path / "hmm",
        )


def test_21_strict_json_native_types_and_single_class_auc(tmp_path, trained_pair):
    path = tmp_path / "strict.json"
    mm._write_strict_json(
        path,
        {
            "integer": np.int64(7),
            "floating": np.float64(1.5),
            "boolean": np.bool_(True),
            "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
            "undefined": np.nan,
        },
    )
    loaded = json.loads(path.read_text())
    assert loaded == {
        "integer": 7,
        "floating": 1.5,
        "boolean": True,
        "timestamp": "2025-01-01T00:00:00+00:00",
        "undefined": None,
    }
    assert mm.auc_result(pd.Series([1, 1]), np.array([0.2, 0.8])) == {
        "value": None,
        "reason": "single_class",
    }

    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant: {value}")

    summary_text = (trained_pair["output_dir"] / "meta_training_summary_4h.json").read_text()
    summary = json.loads(summary_text, parse_constant=reject_constant)
    final_model = summary["final_model"]
    for field in ["train_auc_meta", "train_auc_primary"]:
        assert set(final_model[field]) == {"value", "reason"}
        assert isinstance(final_model[field]["value"], float)
        assert 0.0 <= final_model[field]["value"] <= 1.0
        assert final_model[field]["reason"] is None

    no_eval_reason = "final_model_has_no_evaluation_fold"
    for field in ["eval_auc_meta", "eval_auc_primary"]:
        assert final_model[field] == {"value": None, "reason": no_eval_reason}
    for field in ["eval_start_k", "purge_boundary_ts", "hmm_coverage_eval"]:
        assert final_model[field] is None
        assert final_model[f"{field}_reason"] == no_eval_reason


def test_22_transaction_staging_and_mid_commit_failures_restore_existing_destination(
    tmp_path,
    monkeypatch,
):
    events = _transaction_events()
    staging_output = tmp_path / "staging"
    _seed_managed_output(staging_output)
    before_staging = _tree_snapshot(staging_output)
    with pytest.raises(RuntimeError, match="staging failure"):
        mm.write_outputs_transactional(
            _FailingSavedModel(),
            events,
            {"run": "staging"},
            staging_output,
            ["BTCUSDT"],
        )
    assert _tree_snapshot(staging_output) == before_staging

    real_replace = os.replace
    for fail_on in [2, 3]:
        output = tmp_path / f"commit_{fail_on}"
        _seed_managed_output(output)
        before = _tree_snapshot(output)
        state = {"commit_calls": 0, "failed": False}

        def flaky_replace(source, destination):
            source_path = Path(source)
            if source_path.parent.name.startswith(".meta_v48_tmp_") and not state["failed"]:
                state["commit_calls"] += 1
                if state["commit_calls"] == fail_on:
                    state["failed"] = True
                    raise OSError(f"synthetic replace failure {fail_on}")
            return real_replace(source, destination)

        with monkeypatch.context() as patch:
            patch.setattr(mm.os, "replace", flaky_replace)
            with pytest.raises(OSError, match="synthetic replace failure"):
                mm.write_outputs_transactional(
                    _FakeSavedModel(),
                    events,
                    {"run": fail_on},
                    output,
                    ["BTCUSDT"],
                )
        assert _tree_snapshot(output) == before


def test_23_transaction_first_run_failure_is_empty_and_success_removes_stale(
    tmp_path,
    monkeypatch,
):
    events = _transaction_events()
    empty_output = tmp_path / "empty"
    real_replace = os.replace
    state = {"commit_calls": 0, "failed": False}

    def fail_second_commit(source, destination):
        source_path = Path(source)
        if source_path.parent.name.startswith(".meta_v48_tmp_") and not state["failed"]:
            state["commit_calls"] += 1
            if state["commit_calls"] == 2:
                state["failed"] = True
                raise OSError("synthetic empty-run failure")
        return real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(mm.os, "replace", fail_second_commit)
        with pytest.raises(OSError, match="empty-run failure"):
            mm.write_outputs_transactional(
                _FakeSavedModel(),
                events,
                {"run": "empty"},
                empty_output,
                ["BTCUSDT"],
            )
    assert _tree_snapshot(empty_output) == []

    stale_output = tmp_path / "stale"
    stale_output.mkdir()
    stale_name = "meta_oof_OLD_4h.csv"
    (stale_output / stale_name).write_bytes(b"stale")
    result = mm.write_outputs_transactional(
        _FakeSavedModel(),
        events,
        {"run": "success"},
        stale_output,
        ["BTCUSDT"],
    )
    assert not (stale_output / stale_name).exists()
    assert result["removed_stale_artifacts"] == [stale_name]
    saved_summary = json.loads((stale_output / "meta_training_summary_4h.json").read_text())
    assert saved_summary["removed_stale_artifacts"] == [stale_name]


def test_24_every_catboost_fit_disables_writing_and_leaves_no_catboost_info(
    tmp_path,
    monkeypatch,
    trained_pair,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(mm.META_CATBOOST_PARAMS, "iterations", 5)
    monkeypatch.setitem(mm.META_CATBOOST_PARAMS, "allow_writing_files", False)
    count = 60
    decision_ts = pd.date_range("2025-01-01", periods=count, freq="4h", tz="UTC")
    train = pd.DataFrame(
        {
            "decision_ts": decision_ts,
            "signal": np.linspace(-1.0, 1.0, count),
            "symbol": np.where(np.arange(count) % 2 == 0, "BTCUSDT", "ETHUSDT"),
            "meta_y": (np.arange(count) % 2).astype(int),
            "tb_uniqueness": 0.5 + 0.01 * (np.arange(count) % 20),
        }
    )
    model, _ = mm.fit_meta_catboost(train, ["signal", "symbol"], ["symbol"])
    params = model.get_params()
    assert params["iterations"] <= 50
    assert params["allow_writing_files"] is False
    assert not (tmp_path / "catboost_info").exists()
    assert trained_pair["catboost_info"] == []
