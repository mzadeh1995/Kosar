# ==============================================================================
# tests/test_ablation81.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Pinned tests implemented through the current ablation81 phase."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics.ablation81.integrity import (
    EnvironmentMismatch,
    ForbiddenMutationError,
    ManifestMismatch,
    S1AnchorError,
    atomic_write_json,
    capture_forbidden_files_fingerprint,
    collect_environment_fingerprint,
    file_identity,
    load_calibrated_events,
    verify_environment_fingerprint,
    verify_forbidden_files_fingerprint,
    verify_hash_manifest,
    verify_research_input_lineage,
    verify_s1_anchors,
)
from analytics.ablation81.feature_ablation import (
    BASE_CATEGORICAL,
    BASE_FEATURES,
    FeatureAblationError,
    build_frozen_arm_spec,
    materialize_session,
    project_arm_frame,
)
from analytics.ablation81.lookahead import audit_asof_alignment
from analytics.ablation81.metrics import (
    additive_equity_curve,
    arm_masks,
    equal_n_fold,
    max_drawdown,
    monte_carlo_p,
    percentile_rank,
    prepare_evaluated_events,
    trade_metrics,
)
from analytics.ablation81.placebo import (
    PlaceboContractError,
    assert_placebo_pool_contract,
    build_sampling_plan,
    run_placebo_plan,
    sample_seed,
)
from analytics.ablation81.nested import (
    NestedSearchError,
    inner_temporal_split_and_purge,
    select_winning_config,
)
from analytics.ablation81.quarantine import (
    FutureFoldViolation,
    QuarantineViolation,
    assert_outer_evaluation_population,
    assert_outer_training_population,
    assert_selection_population,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _flip_hash_nibble(document: dict, record_key: str) -> dict:
    tampered = copy.deepcopy(document)
    original = tampered[record_key][0]["sha256"]
    replacement = "0" if original[0] != "0" else "1"
    tampered[record_key][0]["sha256"] = replacement + original[1:]
    return tampered


def test_01_p0_detects_planted_future_leak() -> None:
    frame = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "source_ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T03:01:00Z",
                ],
                utc=True,
            ),
            "matched": [True, True, True],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    result = audit_asof_alignment(
        frame,
        source="synthetic",
        decision_column="decision_ts",
        source_timestamp_column="source_ts",
        match_column="matched",
        tolerance=pd.Timedelta(hours=2),
        feature_columns=("feature",),
    )
    assert result["rows_scanned"] == 3
    assert result["future_timestamp_violation_count"] == 1
    assert result["p0_a_verdict"] == "error"


def test_02_p0_passes_clean_synthetic_join() -> None:
    frame = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "source_ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:30:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "matched": [True, True, True],
            "feature": [1.0, 2.0, 3.0],
        }
    )
    result = audit_asof_alignment(
        frame,
        source="synthetic",
        decision_column="decision_ts",
        source_timestamp_column="source_ts",
        match_column="matched",
        tolerance=pd.Timedelta(hours=1),
        feature_columns=("feature",),
    )
    assert result["rows_scanned"] == 3
    assert result["blocking_violation_count"] == 0
    assert result["p0_a_verdict"] == "healthy"


def test_03_manifest_verification_catches_tamper(tmp_path: Path) -> None:
    manifest_payload = tmp_path / "manifest_payload.bin"
    manifest_payload.write_bytes(b"phase-zero-manifest-payload")
    manifest_record = file_identity(manifest_payload)
    manifest_record["path"] = manifest_payload.name
    manifest_document = {
        "schema_version": 1,
        "root": str(tmp_path),
        "files": [manifest_record],
    }
    manifest_path = tmp_path / "hash_manifest.json"
    atomic_write_json(manifest_path, manifest_document)
    assert verify_hash_manifest(manifest_path)["status"] == "passed"

    atomic_write_json(
        manifest_path, _flip_hash_nibble(manifest_document, "files")
    )
    with pytest.raises(ManifestMismatch):
        verify_hash_manifest(manifest_path)
    atomic_write_json(manifest_path, manifest_document)
    manifest_payload.write_bytes(b"Phase-zero-manifest-payload")
    with pytest.raises(ManifestMismatch):
        verify_hash_manifest(manifest_path)

    lineage_payload = tmp_path / "lineage_payload.bin"
    lineage_payload.write_bytes(b"phase-zero-lineage-payload")
    lineage_record = {
        **file_identity(lineage_payload),
        "path": lineage_payload.name,
        "role": "synthetic_test_input",
        "symbol": None,
        "timeframe": "4h",
    }
    lineage_document = {
        "schema_version": 1,
        "root": str(tmp_path),
        "inputs": [lineage_record],
    }
    lineage_path = tmp_path / "research_input_lineage.json"
    atomic_write_json(lineage_path, lineage_document)
    assert verify_research_input_lineage(lineage_path)["status"] == "passed"

    atomic_write_json(
        lineage_path, _flip_hash_nibble(lineage_document, "inputs")
    )
    with pytest.raises(ManifestMismatch):
        verify_research_input_lineage(lineage_path)
    atomic_write_json(lineage_path, lineage_document)
    lineage_payload.write_bytes(b"Phase-zero-lineage-payload")
    with pytest.raises(ManifestMismatch):
        verify_research_input_lineage(lineage_path)

    observed_environment = collect_environment_fingerprint(thread_count=3)
    environment_path = tmp_path / "environment_fingerprint.json"
    atomic_write_json(environment_path, observed_environment)
    assert (
        verify_environment_fingerprint(environment_path, observed_environment)[
            "status"
        ]
        == "passed"
    )
    mismatched_environment = copy.deepcopy(observed_environment)
    mismatched_environment["numpy"] = f"{observed_environment['numpy']}.tampered"
    with pytest.raises(EnvironmentMismatch):
        verify_environment_fingerprint(environment_path, mismatched_environment)

    forbidden_root = tmp_path / "forbidden_scope"
    forbidden_root.mkdir()
    finder_metadata = forbidden_root / ".DS_Store"
    other_dotfile = forbidden_root / ".hidden"
    finder_metadata.write_bytes(b"finder-metadata-v1")
    other_dotfile.write_bytes(b"other-dotfile-v1")
    forbidden_document = capture_forbidden_files_fingerprint(forbidden_root)
    assert forbidden_document["ignored_current_record_count"] == 1
    assert [record["path"] for record in forbidden_document["records"]] == [
        ".hidden"
    ]
    forbidden_path = tmp_path / "forbidden_files_fingerprint.json"
    atomic_write_json(forbidden_path, forbidden_document)

    finder_metadata.write_bytes(b"finder-metadata-v2")
    exact_exemption = verify_forbidden_files_fingerprint(
        forbidden_path, project_root=forbidden_root
    )
    assert exact_exemption["status"] == "passed"
    assert exact_exemption["excluded_current_record_count"] == 1

    other_dotfile.write_bytes(b"other-dotfile-v2")
    with pytest.raises(ForbiddenMutationError):
        verify_forbidden_files_fingerprint(
            forbidden_path, project_root=forbidden_root
        )


def test_04_s1_anchor_mismatch_stops() -> None:
    events = load_calibrated_events(PROJECT_ROOT)
    baseline = verify_s1_anchors(events)
    assert baseline["status"] == "passed"
    assert baseline["total_rows"] == 11_029

    tampered = events.copy(deep=True)
    evaluated_index = tampered.index[
        tampered["meta_eval_status"].eq("evaluated")
    ][0]
    tampered.loc[evaluated_index, "p_meta"] = np.nan
    with pytest.raises(S1AnchorError):
        verify_s1_anchors(tampered)


def test_05_shared_calculators_match_hand_values() -> None:
    decision = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T04:00:00Z",
            "2026-01-01T08:00:00Z",
            "2026-01-01T12:00:00Z",
        ],
        utc=True,
    )
    raw = pd.DataFrame(
        {
            "decision_ts": decision,
            "fold": [2, 2, 2, 2],
            "symbol": ["BTCUSDT", "ETHUSDT", "BTCUSDT", "ETHUSDT"],
            "tb_return": [0.013, -0.017, 0.023, -0.007],
            "tb_exit_index": decision
            + pd.to_timedelta([4, 8, 12, 4], unit="h"),
            "tb_uniqueness": [1.0, 1.0, 1.0, 1.0],
            "meta_y": [1, 0, 1, 0],
            "p_meta": [0.9, 0.9, 0.8, 0.7],
            "p_primary": [0.1, 0.2, 0.9, 0.8],
            "traded_meta": [True, True, False, False],
            "traded_baseline": [True, True, True, True],
            "meta_eval_status": ["evaluated"] * 4,
        }
    )
    events = prepare_evaluated_events(raw, economic_cost=0.003)
    expected_net = np.array([0.01, -0.02, 0.02, -0.01])
    np.testing.assert_allclose(events["net_return"], expected_net, atol=1e-15)
    np.testing.assert_allclose(
        additive_equity_curve(expected_net),
        np.array([0.01, -0.01, 0.01, 0.0]),
        atol=1e-15,
    )
    assert max_drawdown(expected_net) == pytest.approx(0.02)

    metrics = trade_metrics(events, pd.Series(True, index=events.index))
    assert metrics["n_trades"] == 4
    assert metrics["sum_net"] == pytest.approx(0.0, abs=1e-15)
    assert metrics["max_drawdown"] == pytest.approx(0.02)
    assert metrics["holding_hours_mean"] == pytest.approx(7.0)
    assert metrics["holding_hours_median"] == pytest.approx(6.0)
    assert metrics["holding_hours_sum"] == pytest.approx(28.0)
    assert metrics["expected_shortfall_5pct"] == pytest.approx(-0.02)
    assert metrics["expected_shortfall_tail_n"] == 1

    masks = arm_masks(events)
    equal_n = equal_n_fold(events, fold=2, a2_mask=masks["A2"])
    assert equal_n["n"] == 2
    assert equal_n["meta_sum_net"] == pytest.approx(-0.01)
    assert equal_n["primary_sum_net"] == pytest.approx(0.01)


def test_06_placebo_sampling_is_seeded_and_count_matched() -> None:
    rows: list[dict] = []
    for fold in (2, 3, 4, 5):
        for offset, (symbol, a1, a2) in enumerate(
            [
                ("ADAUSDT", True, True),
                ("ADAUSDT", True, False),
                ("BTCUSDT", False, True),
                ("BTCUSDT", False, False),
            ]
        ):
            rows.append(
                {
                    "fold": fold,
                    "symbol": symbol,
                    "meta_eval_status": "evaluated",
                    "p_meta": 0.51 + 0.01 * offset,
                    "p_primary": 0.61 + 0.01 * offset,
                    "net_return": float(fold * 10 + offset),
                    "holding_hours": float(4 + offset),
                    "traded_baseline": a1,
                    "traded_meta": a2,
                }
            )
    events = pd.DataFrame(rows).reset_index(drop=True)
    masks = arm_masks(events)
    primary_pool = masks["A1"] | masks["A2"]
    universe_pool = pd.Series(True, index=events.index)
    assert (
        assert_placebo_pool_contract(
            events,
            pool_mask=primary_pool,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="primary_informed",
        )["status"]
        == "passed"
    )
    assert (
        assert_placebo_pool_contract(
            events,
            pool_mask=universe_pool,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="universe",
        )["status"]
        == "passed"
    )

    wrong_primary_superpool = universe_pool.copy()
    with pytest.raises(PlaceboContractError):
        assert_placebo_pool_contract(
            events,
            pool_mask=wrong_primary_superpool,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="primary_informed",
        )
    wrong_universe = universe_pool.copy()
    wrong_universe.iloc[-1] = False
    with pytest.raises(PlaceboContractError):
        assert_placebo_pool_contract(
            events,
            pool_mask=wrong_universe,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="universe",
        )
    non_evaluated = events.copy()
    non_evaluated.loc[0, "meta_eval_status"] = "not_evaluated"
    with pytest.raises(PlaceboContractError):
        assert_placebo_pool_contract(
            non_evaluated,
            pool_mask=primary_pool,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="primary_informed",
        )
    null_score = events.copy()
    null_score.loc[0, "p_meta"] = np.nan
    with pytest.raises(PlaceboContractError):
        assert_placebo_pool_contract(
            null_score,
            pool_mask=primary_pool,
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type="primary_informed",
        )

    seeds = [0, 1, 2]
    expected_seed_zero = {
        ("primary_informed", "per_fold"): {
            2: [1, 2],
            3: [4, 6],
            4: [8, 10],
            5: [13, 14],
        },
        ("primary_informed", "fold_symbol"): {
            2: [1, 2],
            3: [5, 6],
            4: [9, 10],
            5: [12, 14],
        },
        ("universe", "per_fold"): {
            2: [2, 3],
            3: [4, 5],
            4: [8, 11],
            5: [14, 15],
        },
        ("universe", "fold_symbol"): {
            2: [1, 3],
            3: [5, 6],
            4: [8, 10],
            5: [12, 14],
        },
    }
    for null_type, matching in (
        ("primary_informed", "per_fold"),
        ("primary_informed", "fold_symbol"),
        ("universe", "per_fold"),
        ("universe", "fold_symbol"),
    ):
        plan, audit = build_sampling_plan(
            events,
            arm="A2",
            arm_mask=masks["A2"],
            a1_mask=masks["A1"],
            null_type=null_type,
            matching=matching,
        )
        assert audit["without_replacement"] is True
        first, first_summary = run_placebo_plan(events, plan, seeds=seeds)
        second, second_summary = run_placebo_plan(events, plan, seeds=seeds)
        assert first.to_csv(index=False, lineterminator="\n").encode() == second.to_csv(
            index=False, lineterminator="\n"
        ).encode()
        assert first_summary == second_summary
        selected = sample_seed(
            plan,
            seed=0,
            net_returns=events["net_return"].to_numpy(),
            holding_hours=events["holding_hours"].to_numpy(),
            return_selected_positions=True,
        )["selected_positions_by_fold"]
        assert selected == expected_seed_zero[(null_type, matching)]
        for fold in (2, 3, 4, 5):
            assert len(selected[fold]) == int(
                (masks["A2"] & events["fold"].eq(fold)).sum()
            )
            assert len(selected[fold]) == len(set(selected[fold]))
            selected_pool = (
                primary_pool if null_type == "primary_informed" else universe_pool
            )
            assert selected_pool.loc[selected[fold]].all()


def test_07_percentile_rank_tie_convention() -> None:
    distribution = np.array([1.0, 2.0, 2.0, 3.0])
    assert percentile_rank(distribution, 2.0) == pytest.approx(50.0)
    assert monte_carlo_p(distribution, 2.0) == pytest.approx(0.8)


def test_08_arm_masks_read_only_from_columns() -> None:
    frame = pd.DataFrame(
        {
            "traded_baseline": [True, True, False, False],
            "traded_meta": [True, False, True, False],
            "p_primary": [0.0, 1.0, 0.0, 1.0],
            "p_meta": [1.0, 0.0, 1.0, 0.0],
        }
    )
    masks = arm_masks(frame)
    assert masks["A1"].tolist() == [True, True, False, False]
    assert masks["A2"].tolist() == [True, False, True, False]
    assert masks["A7"].tolist() == [True, False, False, False]

    score_tampered = frame.copy()
    score_tampered["p_primary"] = [1.0, 0.0, 1.0, 0.0]
    score_tampered["p_meta"] = [0.0, 1.0, 0.0, 1.0]
    tampered_masks = arm_masks(score_tampered)
    for arm in ("A1", "A2", "A7"):
        assert tampered_masks[arm].equals(masks[arm])


def test_09_nested_search_never_sees_future_folds() -> None:
    for outer_fold in (2, 3, 4, 5):
        healthy = pd.DataFrame(
            {
                "fold": list(range(1, outer_fold)),
                "decision_ts": pd.date_range(
                    "2025-01-01",
                    periods=outer_fold - 1,
                    freq="4h",
                    tz="UTC",
                ),
            }
        )
        result = assert_outer_training_population(
            healthy, outer_fold=outer_fold
        )
        assert result["past_only"] is True
        assert all(fold < outer_fold for fold in result["folds"])

        for planted_fold in (outer_fold, min(5, outer_fold + 1)):
            poisoned = pd.concat(
                [
                    healthy,
                    pd.DataFrame(
                        {
                            "fold": [planted_fold],
                            "decision_ts": [
                                pd.Timestamp("2026-01-01", tz="UTC")
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )
            with pytest.raises(FutureFoldViolation):
                assert_outer_training_population(
                    poisoned, outer_fold=outer_fold
                )


def test_10_inner_split_respects_purge_embargo() -> None:
    decision_ts = pd.date_range(
        "2025-01-01T00:00:00Z", periods=8, freq="24h"
    )
    validation_start = decision_ts[6]
    purge_boundary = validation_start - pd.Timedelta(hours=96)
    frame = pd.DataFrame(
        {
            "fold": [1] * 8,
            "symbol": [
                "BTCUSDT",
                "ETHUSDT",
                "BTCUSDT",
                "ETHUSDT",
                "BTCUSDT",
                "ETHUSDT",
                "BTCUSDT",
                "ETHUSDT",
            ],
            "decision_ts": decision_ts,
            "tb_exit_index": [
                purge_boundary - pd.Timedelta(hours=5),
                purge_boundary - pd.Timedelta(hours=4),
                purge_boundary - pd.Timedelta(hours=3),
                purge_boundary,
                purge_boundary + pd.Timedelta(hours=4),
                purge_boundary + pd.Timedelta(hours=8),
                decision_ts[6] + pd.Timedelta(hours=4),
                decision_ts[7] + pd.Timedelta(hours=4),
            ],
        }
    )
    train, validation, audit = inner_temporal_split_and_purge(
        frame,
        outer_fold=2,
        is_real_data_run=True,
    )
    assert validation.index.tolist() == [6, 7]
    assert train.index.tolist() == [0]
    assert audit["inner_validation_floor_25pct_rows"] == 2
    assert audit["inner_pre_purge_train_rows"] == 6
    assert audit["inner_post_purge_train_rows"] == 1
    assert audit["inner_validation_start"] == validation_start
    assert audit["purge"]["purge_boundary_ts"] == purge_boundary
    assert (
        audit["purge"]["purge_rule"]
        == "tb_exit_index + 4h < eval_start_k - 96h"
    )
    assert (
        pd.to_datetime(train["tb_exit_index"], utc=True)
        + pd.Timedelta(hours=4)
        < purge_boundary
    ).all()


def test_11_frozen_config_ablation_changes_only_columns() -> None:
    config = {
        "depth": 3,
        "l2_leaf_reg": 30,
        "learning_rate": 0.01,
        "boosting_type": "Ordered",
        "bootstrap": "default",
    }
    expected_changes = {
        "A5": ({"hour_of_day", "day_of_week"}, set()),
        "A6": (
            {"vpin", "vpin_z", "funding_z", "funding_extreme_pos"},
            set(),
        ),
        "E3": ({"hour_of_day", "day_of_week"}, {"session"}),
        "E4": ({"symbol"}, set()),
        "E5": (
            {
                "hmm_regime",
                "hmm_bull_prob",
                "hmm_neutral_prob",
                "hmm_bear_prob",
                "hmm_confidence",
                "hmm_policy_code",
                "hmm_regime_age_hours",
            },
            set(),
        ),
        "E6": (set(), {"btc_dist_from_max_168", "close_z_4h"}),
    }
    for arm, (removed, added) in expected_changes.items():
        spec = build_frozen_arm_spec(
            arm,
            config=config,
            tree_count=1,
            expected_config=config,
            expected_tree_count=1,
            base_features=BASE_FEATURES,
            base_categorical=BASE_CATEGORICAL,
        )
        assert set(spec["removed_columns"]) == removed
        assert set(spec["added_columns"]) == added
        assert set(spec["feature_columns"]) == (
            set(BASE_FEATURES) - removed
        ) | added
        assert spec["config"] == config
        assert spec["tree_count"] == 1
        assert spec["eval_set_used"] is False
        assert spec["early_stopping_used"] is False
        assert spec["search_or_reselection_used"] is False
        synthetic = pd.DataFrame(
            {column: [0.0] for column in set(BASE_FEATURES) | added}
        )
        projected = project_arm_frame(synthetic, spec)
        assert projected.columns.tolist() == spec["feature_columns"]

    assert "symbol" not in build_frozen_arm_spec(
        "E4",
        config=config,
        tree_count=1,
        expected_config=config,
        expected_tree_count=1,
        base_features=BASE_FEATURES,
        base_categorical=BASE_CATEGORICAL,
    )["categorical_features"]
    assert "hmm_regime" not in build_frozen_arm_spec(
        "E5",
        config=config,
        tree_count=1,
        expected_config=config,
        expected_tree_count=1,
        base_features=BASE_FEATURES,
        base_categorical=BASE_CATEGORICAL,
    )["categorical_features"]
    session = materialize_session(
        pd.Series(pd.date_range("2026-01-01", periods=6, freq="4h", tz="UTC"))
    )
    assert session.astype(str).tolist() == [
        "Asia",
        "Asia",
        "Europe",
        "Europe",
        "America",
        "America",
    ]

    changed = dict(config)
    changed["depth"] = 4
    with pytest.raises(FeatureAblationError):
        build_frozen_arm_spec(
            "A5",
            config=changed,
            tree_count=1,
            expected_config=config,
            expected_tree_count=1,
            base_features=BASE_FEATURES,
            base_categorical=BASE_CATEGORICAL,
        )
    with pytest.raises(FeatureAblationError):
        build_frozen_arm_spec(
            "A5",
            config=config,
            tree_count=2,
            expected_config=config,
            expected_tree_count=1,
            base_features=BASE_FEATURES,
            base_categorical=BASE_CATEGORICAL,
        )
    with pytest.raises(FeatureAblationError):
        build_frozen_arm_spec(
            "A5",
            config=config,
            tree_count=1,
            expected_config=config,
            expected_tree_count=1,
            base_features=BASE_FEATURES[:-1],
            base_categorical=BASE_CATEGORICAL,
        )


def test_12_fold5_quarantine_guard_raises() -> None:
    healthy = pd.DataFrame({"fold": [1, "2", 4]})
    assert assert_selection_population(healthy)["status"] == "passed"

    poisoned = pd.DataFrame({"fold": [1, "2", "5"]})
    with pytest.raises(QuarantineViolation):
        assert_selection_population(poisoned)
    with pytest.raises(QuarantineViolation):
        assert_outer_training_population(poisoned, outer_fold=5)

    fold5_evaluation = pd.DataFrame({"fold": ["5", 5]})
    result = assert_outer_evaluation_population(
        fold5_evaluation, outer_fold=5
    )
    assert result["status"] == "passed"
    assert result["selection_use_allowed"] is False


def test_15_cscv_predictions_never_enter_selection(tmp_path: Path) -> None:
    def ledger_row(
        config_id: str,
        loss: float,
        *,
        depth: int,
        l2_leaf_reg: int,
    ) -> dict:
        return {
            "config_id": config_id,
            "config": {
                "depth": depth,
                "l2_leaf_reg": l2_leaf_reg,
                "learning_rate": 0.03,
                "boosting_type": "Ordered",
                "bootstrap": "default",
            },
            "inner_validation_weighted_raw_logloss": loss,
            "tree_count": 17,
            "best_iteration": 16,
            "scale_pos_weight": 1.0,
            "weighted_positive_sum": 2.0,
            "weighted_negative_sum": 2.0,
            "inner_fit_cpu_seconds": 0.1,
            "inner_fit_wall_seconds": 0.1,
        }

    ledger = [
        ledger_row("cfg_000", 0.4, depth=3, l2_leaf_reg=30),
        ledger_row("cfg_001", 0.3, depth=5, l2_leaf_reg=3),
    ]
    predictions_path = tmp_path / "cscv_predictions_4h.csv"
    predictions = pd.DataFrame(
        {
            "config_id": ["cfg_000", "cfg_001"],
            "outer_fold": [5, 5],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "decision_ts": ["2026-01-01T00:00:00Z"] * 2,
            "p_raw": [0.01, 0.99],
        }
    )
    predictions.to_csv(predictions_path, index=False)
    before = select_winning_config(ledger)

    poisoned_predictions = predictions.copy()
    poisoned_predictions["p_raw"] = [1.0, 0.0]
    poisoned_predictions.to_csv(predictions_path, index=False)
    after = select_winning_config(ledger)
    assert before == after
    assert before["config_id"] == "cfg_001"

    contaminated = copy.deepcopy(ledger)
    contaminated[0]["p_raw"] = [0.99]
    with pytest.raises(NestedSearchError):
        select_winning_config(contaminated)
