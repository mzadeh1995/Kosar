# ==============================================================================
# analytics/ablation81/adjudication.py
# ------------------------------------------------------------------------------
# v49.1: Added "Calibration" and Patch
# ==============================================================================

"""Narrow owner-adjudication gate for the frozen funding coverage signature."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .integrity import file_identity
from .metrics import MetricsContractError, strict_bool


OWNER_ARTIFACT_RELATIVE = Path(
    "data/models/ablation81/owner_adjudication_funding.json"
)
P0_REPORT_RELATIVE = Path("data/models/ablation81/p0_lookahead_report.json")
EXPECTED_OWNER_ARTIFACT_IDENTITY = {
    "size_bytes": 8_088,
    "sha256": "be1c7aeea2fbab564d906cc244c2623a15a8bc2ec43cd32d53a5253650f3b601",
}
EXPECTED_OWNER_MESSAGE_IDENTITY = {
    "characters": 4_024,
    "utf8_bytes": 6_640,
    "sha256": "5cce2f142a7c24004eee86017c642446d02e1365a79a8ada943218cc08566955",
}
EXPECTED_P0_IDENTITY = {
    "size_bytes": 43_244,
    "sha256": "6df201b08f5107b396d04608ac89c70e6c7ad848ea6b1516377e9a1b71772f5d",
}
EXPECTED_ADJUDICATED_SIGNATURE = {
    "unmatched_count": 103,
    "unmatched_decision_start": "2026-07-01T04:00:00Z",
    "unmatched_decision_end": "2026-07-06T00:00:00Z",
    "unmatched_by_symbol": {
        "ADA": 11,
        "AVAX": 29,
        "BNB": 2,
        "BTC": 21,
        "ETH": 11,
        "SOL": 20,
        "XRP": 9,
    },
    "affected_fold_counts": {"5": 103},
    "future_timestamp_violation_count": 0,
    "funding_matched_but_feature_nan": {
        "symbol": "ADAUSDT",
        "decision_ts": "2024-02-05T16:00:00Z",
        "fold": 1,
    },
    "hmm_matched_but_feature_nan": {
        "symbol": "BTCUSDT",
        "decision_ts": "2024-08-06T08:00:00Z",
        "fold": 2,
    },
}


class OwnerAdjudicationMismatch(RuntimeError):
    """Raised whenever the one authorized funding signature changes."""


def _identity_subset(path: Path) -> dict[str, Any]:
    identity = file_identity(path)
    return {
        "size_bytes": int(identity["size_bytes"]),
        "sha256": str(identity["sha256"]),
    }


def _utc_z(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _short_symbol(value: Any) -> str:
    symbol = str(value)
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def verify_owner_adjudication_funding(
    project_root: Path,
) -> dict[str, Any]:
    """Verify the frozen artifact, verbatim-message bytes, signature, and P0 link."""

    root = project_root.resolve(strict=True)
    owner_path = root / OWNER_ARTIFACT_RELATIVE
    owner_identity = _identity_subset(owner_path)
    if owner_identity != EXPECTED_OWNER_ARTIFACT_IDENTITY:
        raise OwnerAdjudicationMismatch(
            "owner adjudication artifact identity changed: "
            f"expected={EXPECTED_OWNER_ARTIFACT_IDENTITY}, observed={owner_identity}"
        )
    try:
        document = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnerAdjudicationMismatch(
            f"cannot read owner adjudication artifact: {exc}"
        ) from exc
    required_top_level = {
        "schema_version": 1,
        "contract_version": 7,
        "phase": 2,
        "frozen": True,
        "status": "owner_adjudication_registered",
        "reentry_protocol_required": True,
    }
    for key, expected in required_top_level.items():
        if document.get(key) != expected:
            raise OwnerAdjudicationMismatch(
                f"owner adjudication {key} expected={expected!r}, "
                f"observed={document.get(key)!r}"
            )
    message = document.get("owner_message_verbatim_fa")
    if not isinstance(message, str):
        raise OwnerAdjudicationMismatch("owner_message_verbatim_fa must be text")
    message_identity = {
        "characters": len(message),
        "utf8_bytes": len(message.encode("utf-8")),
        "sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }
    if message_identity != EXPECTED_OWNER_MESSAGE_IDENTITY:
        raise OwnerAdjudicationMismatch(
            "verbatim owner message changed: "
            f"expected={EXPECTED_OWNER_MESSAGE_IDENTITY}, observed={message_identity}"
        )
    if document.get("adjudicated_signature") != EXPECTED_ADJUDICATED_SIGNATURE:
        raise OwnerAdjudicationMismatch("registered adjudicated signature changed")
    expected_scope = {
        "exact_signature_only": True,
        "p0_error_verdict_remains_frozen": True,
        "automatic_expansion_or_readjudication_allowed": False,
        "funding_repair_or_alternative_source_allowed": False,
        "phase2_continuation_allowed": True,
        "phase3_requires_machine_exact_signature_assertion": True,
    }
    if document.get("authorization_scope") != expected_scope:
        raise OwnerAdjudicationMismatch("authorization scope changed")

    p0_path = root / P0_REPORT_RELATIVE
    p0_identity = _identity_subset(p0_path)
    registered_p0 = document.get("p0_lookahead_report_identity")
    expected_registered_p0 = {
        "path": P0_REPORT_RELATIVE.as_posix(),
        **EXPECTED_P0_IDENTITY,
    }
    if (
        p0_identity != EXPECTED_P0_IDENTITY
        or registered_p0 != expected_registered_p0
    ):
        raise OwnerAdjudicationMismatch(
            "P0 identity does not match the adjudicated frozen report"
        )
    return {
        "status": "passed",
        "owner_artifact": {
            "path": OWNER_ARTIFACT_RELATIVE.as_posix(),
            **owner_identity,
        },
        "owner_message": message_identity,
        "p0_lookahead_report": {
            "path": P0_REPORT_RELATIVE.as_posix(),
            **p0_identity,
        },
        "signature": EXPECTED_ADJUDICATED_SIGNATURE,
        "authorization_scope": expected_scope,
    }


def derive_affected_mask_from_pinned(
    events: pd.DataFrame,
) -> tuple[pd.Series, dict[str, Any]]:
    """Identify the registered edge rows in pinned data and require exact counts."""

    required = {"decision_ts", "symbol", "fold"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise OwnerAdjudicationMismatch(
            f"pinned population misses adjudication columns: {missing}"
        )
    decisions = pd.to_datetime(events["decision_ts"], utc=True, errors="raise")
    folds = pd.to_numeric(events["fold"], errors="raise").astype(int)
    start = pd.Timestamp(
        EXPECTED_ADJUDICATED_SIGNATURE["unmatched_decision_start"]
    )
    end = pd.Timestamp(EXPECTED_ADJUDICATED_SIGNATURE["unmatched_decision_end"])
    affected = folds.eq(5) & decisions.between(start, end, inclusive="both")
    selected = events.loc[affected].copy()
    selected_decisions = decisions.loc[affected]
    signature = {
        "unmatched_count": int(len(selected)),
        "unmatched_decision_start": (
            None if selected.empty else _utc_z(selected_decisions.min())
        ),
        "unmatched_decision_end": (
            None if selected.empty else _utc_z(selected_decisions.max())
        ),
        "unmatched_by_symbol": {
            _short_symbol(symbol): int(count)
            for symbol, count in selected["symbol"].value_counts().sort_index().items()
        },
        "affected_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in folds.loc[affected].value_counts().sort_index().items()
        },
    }
    expected = {
        key: EXPECTED_ADJUDICATED_SIGNATURE[key]
        for key in (
            "unmatched_count",
            "unmatched_decision_start",
            "unmatched_decision_end",
            "unmatched_by_symbol",
            "affected_fold_counts",
        )
    }
    if signature != expected:
        raise OwnerAdjudicationMismatch(
            f"pinned affected-row signature changed: expected={expected}, "
            f"observed={signature}"
        )
    return affected, {"status": "passed", **signature}


def assert_owner_adjudicated_funding_signature(
    assembled_events: pd.DataFrame,
) -> dict[str, Any]:
    """Phase-three hard gate: require the exact full assembled signature."""

    required = {
        "decision_ts",
        "symbol",
        "fold",
        "_funding_matched",
        "_funding_source_ts",
        "funding_z",
        "funding_extreme_pos",
        "_hmm_matched",
        "hmm_regime",
        "hmm_bull_prob",
        "hmm_neutral_prob",
        "hmm_bear_prob",
        "hmm_confidence",
        "hmm_policy_code",
        "hmm_regime_age_hours",
    }
    missing = sorted(required - set(assembled_events.columns))
    if missing:
        raise OwnerAdjudicationMismatch(
            f"assembled signature gate misses columns: {missing}"
        )
    decisions = pd.to_datetime(
        assembled_events["decision_ts"], utc=True, errors="raise"
    )
    funding_source = pd.to_datetime(
        assembled_events["_funding_source_ts"], utc=True, errors="coerce"
    )
    try:
        funding_matched = strict_bool(
            assembled_events["_funding_matched"],
            label="_funding_matched",
        )
        hmm_matched = strict_bool(
            assembled_events["_hmm_matched"],
            label="_hmm_matched",
        )
    except MetricsContractError as exc:
        raise OwnerAdjudicationMismatch(str(exc)) from exc
    unmatched = ~funding_matched
    expected_unmatched, expected_unmatched_audit = (
        derive_affected_mask_from_pinned(assembled_events)
    )
    if not unmatched.equals(expected_unmatched):
        disagreement = int((unmatched != expected_unmatched).sum())
        raise OwnerAdjudicationMismatch(
            "owner-authorized funding row set changed: "
            f"row_disagreement_count={disagreement}"
        )
    if funding_source.loc[funding_matched].isna().any():
        raise OwnerAdjudicationMismatch(
            "matched funding rows contain null or invalid source timestamps"
        )
    if not assembled_events.loc[
        unmatched,
        ["funding_z", "funding_extreme_pos"],
    ].isna().all(axis=None):
        raise OwnerAdjudicationMismatch(
            "owner-authorized unmatched funding rows contain repaired features"
        )
    unmatched_rows = assembled_events.loc[unmatched]
    unmatched_decisions = decisions.loc[unmatched]
    future_count = int(
        (
            funding_source.loc[funding_matched]
            > decisions.loc[funding_matched]
        ).sum()
    )
    funding_nan = funding_matched & assembled_events[
        ["funding_z", "funding_extreme_pos"]
    ].isna().any(axis=1)
    hmm_nan = hmm_matched & assembled_events[
        [
            "hmm_regime",
            "hmm_bull_prob",
            "hmm_neutral_prob",
            "hmm_bear_prob",
            "hmm_confidence",
            "hmm_policy_code",
            "hmm_regime_age_hours",
        ]
    ].isna().any(axis=1)

    def single_identifier(mask: pd.Series, label: str) -> dict[str, Any]:
        rows = assembled_events.loc[mask]
        if len(rows) != 1:
            raise OwnerAdjudicationMismatch(
                f"{label} expected one row, observed {len(rows)}"
            )
        index = rows.index[0]
        return {
            "symbol": str(rows.loc[index, "symbol"]),
            "decision_ts": _utc_z(decisions.loc[index]),
            "fold": int(rows.loc[index, "fold"]),
        }

    observed = {
        "unmatched_count": int(unmatched.sum()),
        "unmatched_decision_start": (
            None if not unmatched.any() else _utc_z(unmatched_decisions.min())
        ),
        "unmatched_decision_end": (
            None if not unmatched.any() else _utc_z(unmatched_decisions.max())
        ),
        "unmatched_by_symbol": {
            _short_symbol(symbol): int(count)
            for symbol, count in unmatched_rows["symbol"]
            .value_counts()
            .sort_index()
            .items()
        },
        "affected_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in unmatched_rows["fold"]
            .value_counts()
            .sort_index()
            .items()
        },
        "future_timestamp_violation_count": future_count,
        "funding_matched_but_feature_nan": single_identifier(
            funding_nan, "funding matched-but-feature-NaN"
        ),
        "hmm_matched_but_feature_nan": single_identifier(
            hmm_nan, "HMM matched-but-feature-NaN"
        ),
    }
    if observed != EXPECTED_ADJUDICATED_SIGNATURE:
        raise OwnerAdjudicationMismatch(
            "owner-authorized funding signature changed: "
            f"expected={EXPECTED_ADJUDICATED_SIGNATURE}, observed={observed}"
        )
    return {
        "status": "passed",
        "exact_signature_match": True,
        "exact_affected_row_mask_match": True,
        "affected_row_mask_audit": expected_unmatched_audit,
        "signature": observed,
    }
