# ==============================================================================
# tests/test_calibration.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Offline, synthetic, and deterministic tests for the v49 calibration island."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import calibration as cal


EXPECTED_INPUT_COLUMNS = [
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

CAL_COLUMNS = [
    "p_meta_cal_platt",
    "p_meta_cal_iso",
    "p_primary_cal_platt",
    "p_primary_cal_iso",
]

CANONICAL_FOLD_COUNTS = {1: 1981, 2: 1745, 3: 2119, 4: 2197, 5: 2987}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(directory: Path) -> dict[str, bytes]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _transaction_debris(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in directory.rglob("*")
        if any(token in path.name.lower() for token in ("staging", "backup", ".bak"))
    ]


def _symbol_events(
    symbol: str,
    fold_sizes: dict[int, int],
    *,
    seed: int,
) -> pd.DataFrame:
    """Build one valid 17-column OOF frame plus the internal symbol column."""
    rng = np.random.default_rng(seed)
    folds = np.concatenate(
        [np.full(fold_sizes[fold], fold, dtype=int) for fold in range(1, 6)]
    )
    n = len(folds)
    position = np.arange(n, dtype=int)
    open_time = pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC")

    # Exact class presence in every non-trivial slice, with seeded score jitter.
    meta_y = ((position + seed) % 4 != 0).astype(int)
    y = 1 - meta_y
    meta_noise = rng.normal(0.0, 0.045, n)
    primary_noise = rng.normal(0.0, 0.025, n)
    p_meta = np.clip(0.16 + 0.64 * meta_y + meta_noise, 0.01, 0.99)
    p_primary = np.clip(0.57 + 0.35 * meta_y + primary_noise, 0.55, 1.0)
    p_meta = p_meta.astype(float)
    p_meta[folds == 1] = np.nan
    evaluated = folds >= 2
    tau = np.where(evaluated, 0.55, np.nan)
    uniqueness = 0.35 + rng.random(n)
    traded_meta = evaluated & (np.nan_to_num(p_meta, nan=-1.0) >= 0.70)
    traded_baseline = evaluated & (p_primary >= 0.80)

    frame = pd.DataFrame(
        {
            "OpenTime": open_time,
            "decision_ts": open_time + pd.Timedelta(hours=4),
            "fold": folds,
            "y": y,
            "meta_y": meta_y,
            "p_primary": p_primary,
            "p_meta": p_meta,
            "tb_return": np.where(meta_y == 1, 0.015, -0.008),
            "tb_exit_index": open_time + pd.Timedelta(hours=8),
            "tb_uniqueness": uniqueness,
            "tau_meta_used": tau,
            "tau_baseline_used": tau,
            "tau_meta_forced_used": tau,
            "tau_baseline_forced_used": tau,
            "traded_meta": traded_meta,
            "traded_baseline": traded_baseline,
            "meta_eval_status": np.where(
                evaluated,
                "evaluated",
                "not_evaluated_no_prior_fold",
            ),
        },
        columns=EXPECTED_INPUT_COLUMNS,
    )
    frame["symbol"] = symbol
    return frame


def _events(
    *,
    rows_per_fold: int = 40,
    symbols: tuple[str, ...] = ("BTCUSDT",),
    seed: int = 4101,
) -> pd.DataFrame:
    frames = [
        _symbol_events(
            symbol,
            {fold: rows_per_fold for fold in range(1, 6)},
            seed=seed + number * 97,
        )
        for number, symbol in enumerate(symbols)
    ]
    return pd.concat(frames, ignore_index=True)


def _write_bundle(
    directory: Path,
    *,
    symbols: tuple[str, ...],
    rows_per_fold: int = 40,
    canonical_counts: bool = False,
    real_flags: bool = False,
) -> dict[str, object]:
    """Write a self-consistent synthetic v48 input bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    symbol_count = len(symbols)
    for symbol_number, symbol in enumerate(symbols):
        if canonical_counts:
            fold_sizes = {
                fold: total // symbol_count
                + int(symbol_number < (total % symbol_count))
                for fold, total in CANONICAL_FOLD_COUNTS.items()
            }
        else:
            fold_sizes = {fold: rows_per_fold for fold in range(1, 6)}
        frame = _symbol_events(symbol, fold_sizes, seed=7001 + 101 * symbol_number)
        frames[symbol] = frame
        frame[EXPECTED_INPUT_COLUMNS].to_csv(
            directory / f"meta_oof_{symbol}_4h.csv",
            index=False,
        )

    model_path = directory / "meta_4h.cbm"
    model_path.write_bytes(b"synthetic-catboost-bytes-never-loaded")
    output_artifacts: dict[str, dict[str, object]] = {}
    for path in [
        model_path,
        *(directory / f"meta_oof_{symbol}_4h.csv" for symbol in symbols),
    ]:
        output_artifacts[path.name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    fold_counts = {
        str(fold): int(sum(frame["fold"].eq(fold).sum() for frame in frames.values()))
        for fold in range(1, 6)
    }
    summary = {
        "version": "v48-synthetic-for-v49-tests",
        "is_real_data_run": real_flags,
        "is_canonical_output": real_flags,
        "timeframe": "4h",
        "symbols": list(symbols),
        "deployment": {"deployment_ready": False},
        "fired_event_count_by_fold": fold_counts,
        "fired_event_counts": {
            symbol: {
                "total": int(len(frame)),
                "by_fold": {
                    str(fold): int(frame["fold"].eq(fold).sum())
                    for fold in range(1, 6)
                },
            }
            for symbol, frame in frames.items()
        },
        "output_artifacts": output_artifacts,
        "evaluation": {
            "paired_tables": {
                "matched_n": {
                    "pooled_folds_2_5": {
                        "meta_matched_n": {
                            "n_trades": 3434,
                            "sum_net_event_units": -5.350815572565586,
                        },
                        "baseline_matched_reference": {
                            "n_trades": 3434,
                            "sum_net_event_units": -1.888085447671394,
                        },
                    }
                }
            }
        },
        "walk_forward": {"folds": {}},
    }
    (directory / "meta_training_summary_4h.json").write_text(
        json.dumps(summary, allow_nan=False),
        encoding="utf-8",
    )
    return {"directory": directory, "frames": frames, "summary": summary}


def _fit(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    method: str,
    domain: tuple[float, float] = (0.0, 1.0),
    weights: np.ndarray | None = None,
    is_real_data_run: bool = False,
):
    if weights is None:
        weights = np.ones(len(scores), dtype=float)
    return cal.fit_score_calibrator(
        scores,
        targets,
        weights,
        method=method,
        input_domain=domain,
        is_real_data_run=is_real_data_run,
    )


def test_01_walk_forward_training_indices_are_strictly_past_only():
    events = _events(rows_per_fold=24)
    result = cal.walk_forward_calibrate(
        events,
        is_real_data_run=False,
        min_train_n=10,
    )
    fold_four_records = [record for record in result["fit_records"] if record["fold"] == 4]
    assert len(fold_four_records) == 4
    expected_indices = set(events.index[events["fold"].isin([2, 3])])
    for record in fold_four_records:
        assert record["train_folds"] == [2, 3]
        assert set(record["train_indices"]) == expected_indices
        assert not set(record["train_indices"]) & set(events.index[events["fold"] >= 4])


def test_02_platt_preserves_auc_inside_each_evaluation_fold():
    events = _events(rows_per_fold=60)
    for score, domain in (("p_meta", (0.0, 1.0)), ("p_primary", (0.55, 1.0))):
        for fold in (3, 4, 5):
            train = events[events["fold"].between(2, fold - 1)]
            current = events[events["fold"].eq(fold)]
            calibrator, metadata = _fit(
                train[score].to_numpy(float),
                train["meta_y"].to_numpy(int),
                method="platt",
                domain=domain,
                weights=train["tb_uniqueness"].to_numpy(float),
            )
            calibrated = cal.predict_calibrated(calibrator, current[score].to_numpy(float))
            raw_auc = roc_auc_score(current["meta_y"], current[score])
            calibrated_auc = roc_auc_score(current["meta_y"], calibrated)
            assert metadata["coefficient"] > 0.0
            assert abs(calibrated_auc - raw_auc) <= 1e-12


def test_03_isotonic_mapping_is_nondecreasing_and_has_no_inversions():
    scores = np.linspace(0.02, 0.98, 240)
    targets = ((np.arange(240) % 7) < (1 + 5 * scores)).astype(int)
    calibrator, _ = _fit(scores, targets, method="isotonic")
    query = np.sort(np.concatenate([np.linspace(0.02, 0.98, 500), [0.4, 0.4]]))
    predicted = calibrator.predict(query)
    assert np.all(np.diff(predicted) >= -1e-15)
    for left in range(len(query) - 1):
        if query[left] < query[left + 1]:
            assert predicted[left] <= predicted[left + 1]
    tied = calibrator.predict(np.array([0.4, 0.4]))
    assert tied[0] == tied[1]


def test_04_isotonic_improves_deliberately_miscalibrated_brier_with_margin():
    rng = np.random.default_rng(4904)
    train_x = rng.uniform(0.0, 1.0, 1800)
    eval_x = rng.uniform(0.0, 1.0, 2400)
    train_y = rng.binomial(1, 0.08 + 0.78 * train_x)
    eval_y = rng.binomial(1, 0.08 + 0.78 * eval_x)
    train_raw = np.clip(0.01 + 0.97 * train_x**3, 0.0, 1.0)
    eval_raw = np.clip(0.01 + 0.97 * eval_x**3, 0.0, 1.0)
    calibrator, _ = _fit(train_raw, train_y, method="isotonic")
    calibrated = calibrator.predict(eval_raw)
    raw_brier = float(np.mean((eval_raw - eval_y) ** 2))
    calibrated_brier = float(np.mean((calibrated - eval_y) ** 2))
    assert raw_brier - calibrated_brier > 0.025


def test_05_fold_two_and_low_sample_history_are_raw_and_flagged():
    events = _events(rows_per_fold=20)
    result = cal.walk_forward_calibrate(
        events,
        is_real_data_run=False,
        min_train_n=300,
    )
    calibrated = result["events"]
    fold_one = calibrated["fold"].eq(1)
    fold_two = calibrated["fold"].eq(2)
    low_history = calibrated["fold"].isin([3, 4, 5])
    assert calibrated.loc[fold_one, CAL_COLUMNS].isna().all().all()
    assert calibrated.loc[fold_one, "cal_is_raw"].eq(0).all()
    assert calibrated.loc[fold_two | low_history, "cal_is_raw"].eq(1).all()
    for fold_mask in (fold_two, *[calibrated["fold"].eq(fold) for fold in (3, 4, 5)]):
        assert np.array_equal(
            calibrated.loc[fold_mask, "p_meta_cal_platt"],
            calibrated.loc[fold_mask, "p_meta"],
        )
        assert np.array_equal(
            calibrated.loc[fold_mask, "p_meta_cal_iso"],
            calibrated.loc[fold_mask, "p_meta"],
        )
        assert np.array_equal(
            calibrated.loc[fold_mask, "p_primary_cal_platt"],
            calibrated.loc[fold_mask, "p_primary"],
        )
        assert np.array_equal(
            calibrated.loc[fold_mask, "p_primary_cal_iso"],
            calibrated.loc[fold_mask, "p_primary"],
        )


def test_06_equal_count_bins_handle_heavy_ties_and_small_inputs():
    probabilities = np.array([0.2] * 50 + list(np.linspace(0.21, 0.95, 50)))
    decision_ts = pd.date_range("2025-01-01", periods=100, freq="4h", tz="UTC")
    symbols = np.resize(np.array(["ETHUSDT", "BTCUSDT"], dtype=object), 100)
    bins = cal.equal_count_bins(probabilities, decision_ts, symbols, max_bins=10)
    flattened = np.concatenate(bins)
    assert sorted(flattened.tolist()) == list(range(100))
    assert max(map(len, bins)) - min(map(len, bins)) <= 1

    small_bins = cal.equal_count_bins(
        np.array([0.5, 0.5, 0.7]),
        pd.date_range("2025-06-01", periods=3, freq="4h", tz="UTC"),
        np.array(["B", "A", "A"]),
        max_bins=10,
    )
    assert len(small_bins) == 3
    assert [len(bin_indices) for bin_indices in small_bins] == [1, 1, 1]


def test_07_weighted_ece_matches_manual_formula_and_logloss_clips_only_itself():
    y = np.array([0, 1, 1, 0], dtype=int)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    weights = np.array([1.0, 3.0, 2.0, 4.0])
    timestamps = pd.date_range("2025-01-01", periods=4, freq="4h", tz="UTC")
    symbols = np.array(["A", "A", "B", "B"])
    metrics = cal.calibration_metrics(
        y,
        p,
        weights,
        timestamps,
        symbols,
        max_bins=2,
    )
    bin_one_gap = abs((1.0 * 0.1 + 3.0 * 0.2) / 4.0 - 3.0 / 4.0)
    bin_two_gap = abs((2.0 * 0.8 + 4.0 * 0.9) / 6.0 - 2.0 / 6.0)
    manual_weighted_ece = (4.0 / 10.0) * bin_one_gap + (6.0 / 10.0) * bin_two_gap
    assert metrics["ece_weighted"] == pytest.approx(manual_weighted_ece, abs=1e-15)

    exact = cal.calibration_metrics(
        np.array([1, 0]),
        np.array([0.0, 1.0]),
        np.ones(2),
        pd.date_range("2025-02-01", periods=2, freq="4h", tz="UTC"),
        np.array(["A", "B"]),
    )
    assert np.isfinite(exact["log_loss_unweighted"])
    assert np.isfinite(exact["log_loss_weighted"])
    assert exact["log_loss_unweighted"] == pytest.approx(-np.log(cal.CAL_LOGLOSS_EPS))
    assert exact["brier_unweighted"] == 1.0
    assert exact["brier_weighted"] == 1.0
    assert exact["ece_unweighted"] == 1.0
    assert exact["ece_weighted"] == 1.0


def test_08_platt_and_isotonic_fit_receive_sample_weight(monkeypatch):
    scores = np.linspace(0.05, 0.95, 80)
    targets = (scores > 0.48).astype(int)
    weights = np.linspace(0.3, 2.1, len(scores))
    observed: dict[str, np.ndarray] = {}
    original_platt_fit = cal.LogisticRegression.fit
    original_iso_fit = cal.IsotonicRegression.fit

    def spy_platt(self, x, y, sample_weight=None):
        observed["platt"] = np.asarray(sample_weight, dtype=float).copy()
        return original_platt_fit(self, x, y, sample_weight=sample_weight)

    def spy_iso(self, x, y, sample_weight=None):
        observed["isotonic"] = np.asarray(sample_weight, dtype=float).copy()
        return original_iso_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(cal.LogisticRegression, "fit", spy_platt)
    monkeypatch.setattr(cal.IsotonicRegression, "fit", spy_iso)
    _fit(scores, targets, method="platt", weights=weights)
    _fit(scores, targets, method="isotonic", weights=weights)
    assert np.array_equal(observed["platt"], weights)
    assert np.array_equal(observed["isotonic"], weights)


def test_09_wrapper_round_trip_accepts_raw_scores_for_all_methods(tmp_path):
    scores = np.linspace(0.03, 0.97, 120).reshape(12, 10)
    flat_scores = scores.ravel()
    targets = (flat_scores > 0.45).astype(int)
    for method in ("platt", "isotonic", "identity"):
        calibrator, _ = _fit(flat_scores, targets, method=method)
        expected = calibrator.predict(scores)
        path = tmp_path / f"{method}.joblib"
        joblib.dump(calibrator, path)
        restored = joblib.load(path)
        actual = cal.predict_calibrated(restored, scores)
        assert actual.shape == scores.shape
        assert np.array_equal(actual, expected)


def test_10_walk_forward_uses_only_evaluated_population_and_meta_y_target(monkeypatch):
    events = _events(rows_per_fold=28)
    assert (events["y"] != events["meta_y"]).all()
    assert events.loc[events["fold"].eq(1), "p_meta"].isna().all()
    observed_targets: list[np.ndarray] = []
    original_fit = cal.fit_score_calibrator

    def spy_fit(raw_scores, targets, sample_weight=None, **kwargs):
        observed_targets.append(np.asarray(targets, dtype=int).copy())
        return original_fit(raw_scores, targets, sample_weight, **kwargs)

    monkeypatch.setattr(cal, "fit_score_calibrator", spy_fit)
    result = cal.walk_forward_calibrate(
        events,
        is_real_data_run=False,
        min_train_n=10,
    )
    assert len(observed_targets) == len(result["fit_records"])
    for seen, record in zip(observed_targets, result["fit_records"], strict=True):
        training = events.loc[record["train_indices"]]
        assert training["meta_eval_status"].eq("evaluated").all()
        assert training["fold"].between(2, record["fold"] - 1).all()
        assert np.array_equal(seen, training["meta_y"].to_numpy(int))
        assert not np.array_equal(seen, training["y"].to_numpy(int))
    fold_one = result["events"]["fold"].eq(1)
    assert result["events"].loc[fold_one, CAL_COLUMNS].isna().all().all()


def test_11_primary_is_calibrated_symmetrically_and_enforces_conditional_domain():
    events = _events(rows_per_fold=32)
    result = cal.walk_forward_calibrate(
        events,
        is_real_data_run=False,
        min_train_n=10,
    )
    calibrated = result["events"]
    assert {"p_primary_cal_platt", "p_primary_cal_iso"}.issubset(calibrated.columns)
    assert calibrated.loc[calibrated["fold"].eq(5), "p_primary_cal_platt"].notna().all()
    primary, _ = _fit(
        np.linspace(0.55, 1.0, 100),
        (np.arange(100) % 3 != 0).astype(int),
        method="identity",
        domain=(0.55, 1.0),
    )
    with pytest.raises(ValueError, match="domain"):
        primary.predict(np.array([0.549999]))


def test_12_single_class_and_constant_score_guards_have_synthetic_only_fallbacks():
    varying = np.linspace(0.1, 0.9, 40)
    single_class = np.ones(40, dtype=int)
    with pytest.raises(cal.CalibrationError):
        _fit(varying, single_class, method="platt", is_real_data_run=True)
    fallback, metadata = _fit(varying, single_class, method="platt")
    assert fallback.method == "identity"
    assert metadata["fallback"] is True
    assert "class" in metadata["fallback_reason"].lower()

    constant = np.full(40, 0.6)
    two_classes = np.resize(np.array([0, 1]), 40)
    with pytest.raises(cal.CalibrationError):
        _fit(constant, two_classes, method="isotonic", is_real_data_run=True)
    fallback, metadata = _fit(constant, two_classes, method="isotonic")
    assert fallback.method == "identity"
    assert metadata["fallback"] is True
    assert any(word in metadata["fallback_reason"].lower() for word in ("distinct", "constant"))

    pipeline_events = _events(rows_per_fold=24)
    pipeline_events.loc[pipeline_events["fold"].ge(2), "p_meta"] = 0.6
    with pytest.raises(cal.CalibrationError):
        cal.walk_forward_calibrate(
            pipeline_events,
            is_real_data_run=False,
            min_train_n=10,
        )


def test_13_nonpositive_platt_coefficient_is_fatal_real_and_reasoned_synthetic():
    scores = np.linspace(0.03, 0.97, 200)
    reverse_ranked_target = (scores < 0.50).astype(int)
    with pytest.raises(cal.CalibrationError):
        _fit(
            scores,
            reverse_ranked_target,
            method="platt",
            is_real_data_run=True,
        )
    fallback, metadata = _fit(scores, reverse_ranked_target, method="platt")
    assert fallback.method == "identity"
    assert metadata["fallback"] is True
    assert metadata["coefficient"] <= 0.0
    assert any(word in metadata["fallback_reason"].lower() for word in ("coefficient", "positive"))


def test_14_preflight_hash_and_strict_loader_schema_duplicate_guards(tmp_path):
    symbols = tuple(cal.CANONICAL_SYMBOLS)
    bundle = _write_bundle(
        tmp_path / "real_like_inputs",
        symbols=symbols,
        canonical_counts=True,
        real_flags=True,
    )
    input_dir = bundle["directory"]
    altered = input_dir / f"meta_oof_{symbols[0]}_4h.csv"
    altered.write_bytes(altered.read_bytes() + b"\n")
    with pytest.raises(cal.PreflightError, match="(?i)(sha256|hash|size_bytes)"):
        cal.preflight_inputs(
            input_dir,
            symbols,
            timeframe="4h",
            is_real_data_run=True,
        )
    assert not list(input_dir.glob("meta_oof_cal_*"))
    assert not (input_dir / "calibration_report_4h.json").exists()

    source = bundle["frames"][symbols[0]][EXPECTED_INPUT_COLUMNS]
    invalid_cases = {
        "missing": source.drop(columns="p_meta"),
        "extra": source.assign(unexpected_column=1),
        "duplicate": pd.concat([source, source.iloc[[0]]], ignore_index=True),
    }
    for case, frame in invalid_cases.items():
        case_dir = tmp_path / case
        case_dir.mkdir()
        path = case_dir / f"meta_oof_{symbols[0]}_4h.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(cal.PreflightError):
            cal.load_oof_file(path, symbols[0], is_real_data_run=False)

    mismatch_path = tmp_path / "meta_oof_ETHUSDT_4h.csv"
    source.to_csv(mismatch_path, index=False)
    with pytest.raises(cal.PreflightError, match="(?i)symbol"):
        cal.load_oof_file(mismatch_path, symbols[0], is_real_data_run=False)


def test_15_noncanonical_identity_guard_and_separate_output_preserve_inputs(
    tmp_path,
    monkeypatch,
):
    symbols = tuple(cal.CANONICAL_SYMBOLS)
    bundle = _write_bundle(
        tmp_path / "synthetic_inputs",
        symbols=symbols,
        rows_per_fold=52,
    )
    input_dir = bundle["directory"]
    before = _snapshot(input_dir)
    fake_default = tmp_path / "canonical_default_output"
    monkeypatch.setattr(cal, "CANONICAL_OUTPUT_DIR", fake_default)
    with pytest.raises((ValueError, cal.CalibrationError, cal.PreflightError)):
        cal.run_calibration(
            symbols=symbols,
            timeframe="4h",
            meta_oof_dir=input_dir,
            output_dir=fake_default,
        )
    assert not fake_default.exists() or _snapshot(fake_default) == {}

    output_dir = tmp_path / "separate_output"
    cal.run_calibration(
        symbols=symbols,
        timeframe="4h",
        meta_oof_dir=input_dir,
        output_dir=output_dir,
    )
    assert _snapshot(input_dir) == before
    report = json.loads((output_dir / "calibration_report_4h.json").read_text())
    assert report["additivity_gate"] == {
        "applicable": False,
        "verdict": "not_applicable_non_real_data",
        "reason": "additivity anchors are canonical-real-data only",
    }
    assert len(report["output_artifacts"]) == 11
    assert len([path for path in output_dir.iterdir() if path.is_file()]) == 12
    for symbol in symbols:
        original = pd.read_csv(input_dir / f"meta_oof_{symbol}_4h.csv")
        calibrated = pd.read_csv(output_dir / f"meta_oof_cal_{symbol}_4h.csv")
        assert list(calibrated.columns[:17]) == EXPECTED_INPUT_COLUMNS
        pd.testing.assert_frame_equal(
            calibrated[EXPECTED_INPUT_COLUMNS],
            original[EXPECTED_INPUT_COLUMNS],
            check_dtype=False,
        )


def test_16_platt_and_identity_preserve_raw_top_n_event_set():
    events = _events(rows_per_fold=70)
    current = events[events["fold"].eq(5)].copy()
    train = events[events["fold"].between(2, 4)]
    platt, metadata = _fit(
        train["p_meta"].to_numpy(float),
        train["meta_y"].to_numpy(int),
        method="platt",
        weights=train["tb_uniqueness"].to_numpy(float),
    )
    identity, _ = _fit(
        train["p_meta"].to_numpy(float),
        train["meta_y"].to_numpy(int),
        method="identity",
        weights=train["tb_uniqueness"].to_numpy(float),
    )
    assert metadata["coefficient"] > 0.0
    current["platt"] = platt.predict(current["p_meta"].to_numpy(float))
    current["identity"] = identity.predict(current["p_meta"].to_numpy(float))
    raw_top = set(cal.top_n_indices(current, "p_meta", 25).tolist())
    assert set(cal.top_n_indices(current, "platt", 25).tolist()) == raw_top
    assert set(cal.top_n_indices(current, "identity", 25).tolist()) == raw_top


def test_17_transaction_failures_restore_full_and_partial_prior_states(tmp_path):
    names = ["artifact_a.bin", "artifact_b.bin", "calibration_report_4h.json"]

    def make_writers(prefix: str) -> dict[str, Callable[[Path], None]]:
        payloads = {
            "artifact_a.bin": f"{prefix}-a".encode(),
            "artifact_b.bin": f"{prefix}-b".encode(),
            "calibration_report_4h.json": json.dumps({"payload": prefix}).encode(),
        }
        return {
            name: (lambda path, data=data: path.write_bytes(data))
            for name, data in payloads.items()
        }

    def validate(staging_dir: Path) -> None:
        assert all((staging_dir / name).is_file() for name in names)
        json.loads((staging_dir / "calibration_report_4h.json").read_text())

    for mode in ("all_existing", "partial_first_run"):
        output_dir = tmp_path / mode
        output_dir.mkdir()
        if mode == "all_existing":
            for name, writer in make_writers("old").items():
                writer(output_dir / name)
        else:
            (output_dir / "artifact_a.bin").write_bytes(b"partial-old-a")
        initial = _snapshot(output_dir)

        staging_failure_writers = make_writers("staging-failure")

        def fail_writer(_path: Path) -> None:
            raise RuntimeError("injected staging failure")

        staging_failure_writers["artifact_b.bin"] = fail_writer
        with pytest.raises(cal.TransactionError):
            cal.transactional_write(
                output_dir,
                staging_failure_writers,
                report_name="calibration_report_4h.json",
                validate_staging=validate,
            )
        assert _snapshot(output_dir) == initial
        assert _transaction_debris(output_dir) == []

        def fail_validation(_staging_dir: Path) -> None:
            raise RuntimeError("injected staging validation failure")

        with pytest.raises(cal.TransactionError):
            cal.transactional_write(
                output_dir,
                make_writers("validation-failure"),
                report_name="calibration_report_4h.json",
                validate_staging=fail_validation,
            )
        assert _snapshot(output_dir) == initial
        assert _transaction_debris(output_dir) == []

        target = "artifact_a.bin" if mode == "all_existing" else "artifact_b.bin"
        injected = {"raised": False}

        def fail_mid_commit(stage: str, filename: str | None) -> None:
            if (
                stage == "after_replace"
                and filename == target
                and not injected["raised"]
            ):
                injected["raised"] = True
                raise RuntimeError(f"injected after replacing {filename}")

        with pytest.raises(cal.TransactionError) as excinfo:
            cal.transactional_write(
                output_dir,
                make_writers("mid-commit-failure"),
                report_name="calibration_report_4h.json",
                validate_staging=validate,
                failure_injector=fail_mid_commit,
            )
        assert injected["raised"] is True
        assert excinfo.value.details["rollback_succeeded"] is True
        if mode == "partial_first_run":
            assert not (output_dir / "artifact_b.bin").exists()
            assert (output_dir / "artifact_a.bin").read_bytes() == b"partial-old-a"
        assert _snapshot(output_dir) == initial
        assert _transaction_debris(output_dir) == []

        commit_events: list[tuple[str, str | None]] = []

        def record_commit(stage: str, filename: str | None) -> None:
            commit_events.append((stage, filename))

        cal.transactional_write(
            output_dir,
            make_writers("success"),
            report_name="calibration_report_4h.json",
            validate_staging=validate,
            failure_injector=record_commit,
        )
        replacements = [filename for stage, filename in commit_events if stage == "after_replace"]
        assert replacements[-1] == "calibration_report_4h.json"
        assert _snapshot(output_dir) == {
            "artifact_a.bin": b"success-a",
            "artifact_b.bin": b"success-b",
            "calibration_report_4h.json": json.dumps({"payload": "success"}).encode(),
        }
        assert _transaction_debris(output_dir) == []


def test_18_identity_selection_round_trip_honesty_flags_and_tie_break(tmp_path):
    targets = np.resize(np.array([0, 1], dtype=int), 100)
    raw_scores = np.where(targets == 1, 0.90, 0.10).astype(float)
    deliberately_worse_platt = np.where(targets == 1, 0.70, 0.30).astype(float)
    deliberately_worse_isotonic = np.full(len(targets), 0.50, dtype=float)
    deliberately_raw_better = {
        "identity": float(np.mean((raw_scores - targets) ** 2)),
        "platt": float(np.mean((deliberately_worse_platt - targets) ** 2)),
        "isotonic": float(np.mean((deliberately_worse_isotonic - targets) ** 2)),
    }
    selection = cal.select_calibration_method(deliberately_raw_better)
    assert selection["selected_method"] == "identity"
    assert selection["calibration_improvement_demonstrated"] is False
    assert selection["probability_ready_for_v50"] is False
    assert selection["output_semantics"] == "uncalibrated_probability_like_score"

    identity, _ = _fit(raw_scores, targets, method=selection["selected_method"])
    path = tmp_path / "identity.joblib"
    joblib.dump(identity, path)
    restored = joblib.load(path)
    assert np.array_equal(restored.predict(raw_scores), raw_scores)

    near_tie = cal.select_calibration_method(
        {
            "identity": 0.2004,
            "platt": 0.2007,
            "isotonic": 0.2000,
        }
    )
    assert set(near_tie["eligible_methods"]) == {"platt", "identity", "isotonic"}
    assert near_tie["selected_method"] == "platt"


def test_19_nonfinite_platt_parameters_stop_real_and_reasoned_synthetic(monkeypatch):
    scores = np.array([0.20, 0.80, 0.30, 0.70], dtype=float)
    targets = np.array([0, 1, 0, 1], dtype=int)
    weights = np.ones(len(scores), dtype=float)

    for coefficient, intercept in ((np.nan, 0.0), (0.1, np.nan)):
        calls = {"fit": 0}

        class StubLogisticRegression:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, design, y, sample_weight=None):
                calls["fit"] += 1
                self.coef_ = np.array([[coefficient]], dtype=float)
                self.intercept_ = np.array([intercept], dtype=float)
                return self

            def predict_proba(self, design):
                positive = np.full(len(design), 0.5, dtype=float)
                return np.column_stack((1.0 - positive, positive))

        monkeypatch.setattr(cal, "LogisticRegression", StubLogisticRegression)

        with pytest.raises(cal.CalibrationError) as excinfo:
            cal.fit_score_calibrator(
                scores,
                targets,
                weights,
                method="platt",
                input_domain=(0.0, 1.0),
                is_real_data_run=True,
            )
        assert calls["fit"] == 1
        assert excinfo.value.code == "nonfinite_platt_parameters"

        wrapper, metadata = cal.fit_score_calibrator(
            scores,
            targets,
            weights,
            method="platt",
            input_domain=(0.0, 1.0),
            is_real_data_run=False,
        )
        assert calls["fit"] == 2
        assert wrapper.method == "identity"
        assert metadata["fallback"] is True
        assert "finite" in metadata["fallback_reason"].lower()


def test_20_staging_core_columns_exact_lock(tmp_path, monkeypatch):
    original = pd.DataFrame(
        {
            "tb_return": [0.125, np.nan],
            "fold": [2, 1],
        }
    )
    cal._assert_core_columns_exact(original.copy(), original, "synthetic.csv")

    drifted = original.copy()
    drifted.loc[0, "tb_return"] += 1e-9
    with pytest.raises(cal.TransactionError):
        cal._assert_core_columns_exact(drifted, original, "synthetic.csv")

    reordered = original.iloc[::-1].reset_index(drop=True)
    with pytest.raises(cal.TransactionError):
        cal._assert_core_columns_exact(reordered, original, "synthetic.csv")

    symbols = tuple(cal.CANONICAL_SYMBOLS)
    bundle = _write_bundle(
        tmp_path / "synthetic_inputs",
        symbols=symbols,
        rows_per_fold=52,
    )
    original_helper = cal._assert_core_columns_exact
    calls = {"count": 0}

    def counting_helper(actual, expected, name):
        calls["count"] += 1
        return original_helper(actual, expected, name)

    monkeypatch.setattr(cal, "_assert_core_columns_exact", counting_helper)
    result = cal.run_calibration(
        symbols=symbols,
        timeframe="4h",
        meta_oof_dir=bundle["directory"],
        output_dir=tmp_path / "separate_output",
    )
    assert result["transaction"]["committed"] is True
    assert calls["count"] == 7


def test_21_cli_failure_state_honesty(monkeypatch, capsys):
    module_path = Path(cal.__file__).resolve().parent / "analytics" / "calibrate_meta.py"
    spec = importlib.util.spec_from_file_location("calibrate_meta_v49_1_test", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def fail_rollback(**_kwargs):
        raise cal.TransactionError(
            "boom",
            details={
                "rollback_succeeded": False,
                "rollback_errors": ["injected"],
                "forensic_staging": "/tmp/x",
            },
        )

    monkeypatch.setattr(cli, "run_calibration", fail_rollback)
    monkeypatch.setattr(sys, "argv", [str(module_path)])
    assert cli.main() == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "failed"
    assert failure["managed_outputs_state"] == "unknown_or_partial"
    assert failure["managed_outputs_written"] is None

    def fail_preflight(**_kwargs):
        raise cal.PreflightError("bad input")

    monkeypatch.setattr(cli, "run_calibration", fail_preflight)
    monkeypatch.setattr(sys, "argv", [str(module_path)])
    assert cli.main() == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["managed_outputs_state"] == "none"
    assert failure["managed_outputs_written"] is False
