"""Supervised model ladder (`ML_REGIME_CONTRACT_PLAN.md` §2; brief §7B.2).

One immutable view + one fold set feed every rung. Before any delta is
computed the runner asserts identical ``oos_row_id`` sets and identical
``(candidate_id, target, training_prevalence)`` tuples across rungs
(acceptance 7B.22-1) — the rows are the SAME out-of-sample rows, so rung
differences are model differences and nothing else.

Rungs are requested by registered protocol **id only**; planned protocols
refuse fail-closed with their registered reason, and the ladder never
exposes a parameter, threshold, calibrator, or feature-subset enumeration
surface (brief §7B.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..context_experiment_contracts import (
    ContextFeatureTier,
    canonical_contract_sha256,
)
from ..context_feature_view import CandidateFeatureView, features_for_tier
from ..context_folds import ContextFoldSet
from ..context_model import run_context_fold_models
from ..context_statistics import binary_prediction_report, block_bootstrap_interval
from .calibration_policies import assert_calibration_policy_executable
from .logistic_model import LogisticModelRun, run_logistic_fold_models
from .model_protocols import (
    CATBOOST_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    PREVALENCE_PROTOCOL_ID,
    assert_model_protocol_available,
    assert_single_frozen_selection,
)

__all__ = [
    "DEFAULT_LADDER_PROTOCOLS",
    "LadderRung",
    "SupervisedLadderRun",
    "run_supervised_ladder",
    "paired_cell_delta_report",
]

DEFAULT_LADDER_PROTOCOLS: tuple[str, ...] = (
    PREVALENCE_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    CATBOOST_PROTOCOL_ID,
)

#: The exact prediction-row columns every rung emits (order-exact; the
#: schema is pinned by the CatBoost lane and mirrored by every other rung).
_PREDICTION_COLUMNS = (
    "oos_row_id",
    "candidate_id",
    "setup_id",
    "trading_day",
    "fold_index",
    "training_prevalence",
    "target",
    "probability",
    "gross_r",
    "net_r",
)


@dataclass(frozen=True, slots=True)
class LadderRung:
    protocol_id: str
    resolved_protocol_hash: str
    predictions: pd.DataFrame
    fold_reports: tuple[dict[str, Any], ...]
    feature_importance: pd.DataFrame
    prediction_report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SupervisedLadderRun:
    ladder_id: str
    view_id: str
    tier: str
    calibration_policy_id: str
    rungs: tuple[LadderRung, ...]
    parity: dict[str, Any]
    paired_deltas: dict[str, dict[str, Any]]

    def rung(self, protocol_id: str) -> LadderRung:
        for rung in self.rungs:
            if rung.protocol_id == protocol_id:
                return rung
        raise KeyError(protocol_id)


def _run_prevalence_reference(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Reference 0 — the fold-local training prevalence as the probability.

    No fit code: the rows are exactly the fitted rungs' rows (same merge,
    same valid-fold loop, same ``oos_row_id``), with
    ``probability = fold.training_prevalence``.
    """

    labels = labeled_candidates.set_index("candidate_id", verify_integrity=True)
    feature_frame = view.frame.set_index("candidate_id", verify_integrity=True)
    if not set(labels.index.astype(str)).issubset(set(feature_frame.index.astype(str))):
        raise ValueError("labels contain candidate IDs outside the immutable feature view")
    merged = feature_frame.join(
        labels[["binary_target", "gross_r", "net_r", "trading_day", "setup_id"]],
        how="left",
        rsuffix="_label",
        validate="one_to_one",
    )
    predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in folds.folds:
        report: dict[str, Any] = {
            "fold_index": fold.fold_index,
            "valid": fold.valid,
            "invalid_reason": fold.invalid_reason,
            "training_prevalence": fold.training_prevalence,
            "train_candidates": len(fold.train_candidate_ids),
            "test_candidates": len(fold.test_candidate_ids),
        }
        if not fold.valid:
            fold_reports.append(report)
            continue
        test = merged.loc[list(fold.test_candidate_ids)]
        for candidate_id in test.index.astype(str):
            source = test.loc[candidate_id]
            predictions.append(
                {
                    "oos_row_id": canonical_contract_sha256(
                        {
                            "view_id": view.view_id,
                            "fold_index": fold.fold_index,
                            "candidate_id": candidate_id,
                        }
                    ),
                    "candidate_id": candidate_id,
                    "setup_id": source.get("setup_id_label", source.get("setup_id")),
                    "trading_day": source.get(
                        "trading_day_label", source.get("trading_day")
                    ),
                    "fold_index": fold.fold_index,
                    "training_prevalence": fold.training_prevalence,
                    "target": int(source["binary_target"]),
                    "probability": float(fold.training_prevalence),
                    "gross_r": float(source["gross_r"]),
                    "net_r": float(source["net_r"]),
                }
            )
        report["model_fitted"] = False  # the reference has no fit stage
        fold_reports.append(report)
    frame = pd.DataFrame(predictions)
    if not frame.empty and frame["oos_row_id"].duplicated().any():
        # same guard the fitted lanes carry (adversarial footnote): the
        # set-based parity gate must never mask duplicate reference rows
        raise ValueError("prevalence reference emitted duplicate OOS row IDs")
    return frame, tuple(fold_reports)


def _parity_key_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["oos_row_id", "candidate_id", "target", "training_prevalence"])
    return predictions[
        ["oos_row_id", "candidate_id", "target", "training_prevalence"]
    ].copy()


def _assert_identical_rows(rungs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The 7B.22-1 gate: identical rows/folds across every rung."""

    reference_name = next(iter(rungs))
    reference = _parity_key_frame(rungs[reference_name])
    reference_ids = set(reference["oos_row_id"].astype(str))
    reference_tuples = {
        (
            str(row.oos_row_id),
            str(row.candidate_id),
            int(row.target),
            float(row.training_prevalence),
        )
        for row in reference.itertuples()
    }
    for name, frame in rungs.items():
        keys = _parity_key_frame(frame)
        ids = set(keys["oos_row_id"].astype(str))
        if ids != reference_ids:
            raise ValueError(
                f"ladder rung {name!r} does not share the reference rung's exact "
                "OOS row-id set; rung deltas are undefined (brief §7B.2)"
            )
        tuples = {
            (
                str(row.oos_row_id),
                str(row.candidate_id),
                int(row.target),
                float(row.training_prevalence),
            )
            for row in keys.itertuples()
        }
        if tuples != reference_tuples:
            raise ValueError(
                f"ladder rung {name!r} disagrees with the reference rung on "
                "(candidate_id, target, training_prevalence) for at least one "
                "shared OOS row"
            )
    return {
        "identical_rows": True,
        "oos_row_count": len(reference_ids),
        "rung_ids": sorted(rungs),
    }


def paired_cell_delta_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    value_column: str = "net_r",
) -> dict[str, Any]:
    """Generalized paired delta keyed on ``oos_row_id`` (DT §4.2).

    The same identical-OOS-ids gate as ``paired_tier_delta_report``, keyed on
    the model-independent row id so ladder rungs (same view, same folds)
    pair exactly; the interval is the fixed trading-day block bootstrap.
    """

    keys = ("oos_row_id", "trading_day", value_column)
    if not set(keys).issubset(left) or not set(keys).issubset(right):
        raise ValueError("cell delta inputs lack paired oos-row/day/value columns")
    if set(left["oos_row_id"].astype(str)) != set(right["oos_row_id"].astype(str)):
        raise ValueError("cell delta requires identical OOS row IDs")
    paired = left[list(keys)].merge(
        right[list(keys)],
        on="oos_row_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if not (
        paired["trading_day_left"].astype(str)
        == paired["trading_day_right"].astype(str)
    ).all():
        raise ValueError("paired cell rows disagree on trading day")
    paired["trading_day"] = paired["trading_day_left"].astype(str)
    paired["delta"] = pd.to_numeric(
        paired[f"{value_column}_right"], errors="raise"
    ) - pd.to_numeric(paired[f"{value_column}_left"], errors="raise")
    return block_bootstrap_interval(
        paired,
        cluster_column="trading_day",
        value_column="delta",
    )


def _with_brier_loss(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["brier_loss"] = (
        pd.to_numeric(frame["target"], errors="raise")
        - pd.to_numeric(frame["probability"], errors="raise")
    ) ** 2
    return frame


def run_supervised_ladder(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    tier: ContextFeatureTier,
    protocols: tuple[str, ...] = DEFAULT_LADDER_PROTOCOLS,
    manual_feature_overrides: dict[str, Any] | None = None,
    calibration_policy_id: str = "raw_probability_diagnostics_v1",
) -> SupervisedLadderRun:
    """Run every requested rung on identical rows/folds and pair the deltas.

    ``protocols`` accepts registered protocol IDS only (unknown → ValueError;
    planned → fail-closed refusal). The prevalence reference is mandatory —
    a ladder without its reference rung has no skill baseline.
    """

    if len(set(protocols)) != len(protocols):
        raise ValueError("ladder protocols contain duplicates")
    for protocol_id in protocols:
        assert_model_protocol_available(protocol_id)
    if PREVALENCE_PROTOCOL_ID not in protocols:
        raise ValueError(
            "the training-prevalence reference rung is mandatory for every ladder"
        )
    assert_single_frozen_selection("calibrator", (calibration_policy_id,))
    assert_calibration_policy_executable(calibration_policy_id)

    features = features_for_tier(tier)
    rungs: list[LadderRung] = []
    frames: dict[str, pd.DataFrame] = {}
    for protocol_id in protocols:
        if protocol_id == PREVALENCE_PROTOCOL_ID:
            predictions, fold_reports = _run_prevalence_reference(
                view, labeled_candidates, folds
            )
            resolved_hash = canonical_contract_sha256(
                {
                    "protocol_id": PREVALENCE_PROTOCOL_ID,
                    "source": "fold_local_training_prevalence",
                    "view_id": view.view_id,
                }
            )
            importance = pd.DataFrame()
        elif protocol_id == LOGISTIC_PROTOCOL_ID:
            run: LogisticModelRun = run_logistic_fold_models(
                view,
                labeled_candidates,
                folds,
                features=features,
                manual_feature_overrides=manual_feature_overrides,
            )
            predictions = run.predictions
            fold_reports = run.fold_reports
            resolved_hash = run.protocol.resolved_hash
            importance = run.feature_importance
        elif protocol_id == CATBOOST_PROTOCOL_ID:
            context_run = run_context_fold_models(
                view,
                labeled_candidates,
                folds,
                tier=tier,
                manual_feature_overrides=manual_feature_overrides,
            )
            predictions = context_run.predictions
            fold_reports = context_run.fold_reports
            resolved_hash = context_run.protocol.resolved_hash
            importance = context_run.feature_importance
        else:  # pragma: no cover - registry alignment makes this unreachable
            raise ValueError(f"protocol {protocol_id!r} has no ladder rung wiring")
        if not predictions.empty:
            missing_columns = [
                column for column in _PREDICTION_COLUMNS if column not in predictions
            ]
            if missing_columns:
                raise ValueError(
                    f"rung {protocol_id!r} predictions lack columns {missing_columns}"
                )
        frames[protocol_id] = predictions
        rungs.append(
            LadderRung(
                protocol_id=protocol_id,
                resolved_protocol_hash=resolved_hash,
                predictions=predictions,
                fold_reports=fold_reports,
                feature_importance=importance,
                prediction_report=binary_prediction_report(
                    predictions
                    if not predictions.empty
                    else pd.DataFrame(columns=list(_PREDICTION_COLUMNS))
                ),
            )
        )

    parity = _assert_identical_rows(frames)

    paired_deltas: dict[str, dict[str, Any]] = {}
    reference_frame = frames[PREVALENCE_PROTOCOL_ID]
    if not reference_frame.empty:
        reference_loss = _with_brier_loss(reference_frame)
        for protocol_id, frame in frames.items():
            if protocol_id == PREVALENCE_PROTOCOL_ID or frame.empty:
                continue
            paired_deltas[f"{PREVALENCE_PROTOCOL_ID}__vs__{protocol_id}"] = (
                paired_cell_delta_report(
                    reference_loss,
                    _with_brier_loss(frame),
                    value_column="brier_loss",
                )
            )
        if (
            LOGISTIC_PROTOCOL_ID in frames
            and CATBOOST_PROTOCOL_ID in frames
            and not frames[LOGISTIC_PROTOCOL_ID].empty
            and not frames[CATBOOST_PROTOCOL_ID].empty
        ):
            paired_deltas[f"{LOGISTIC_PROTOCOL_ID}__vs__{CATBOOST_PROTOCOL_ID}"] = (
                paired_cell_delta_report(
                    _with_brier_loss(frames[LOGISTIC_PROTOCOL_ID]),
                    _with_brier_loss(frames[CATBOOST_PROTOCOL_ID]),
                    value_column="brier_loss",
                )
            )

    # adversarial m-9: the ladder id binds the LABELS and FOLDS it ran on,
    # not just the view/protocols — two labelings can never share one id
    label_content_hash = canonical_contract_sha256(
        {
            "labeled_pairs": sorted(
                (str(candidate_id), None if pd.isna(target) else int(target))
                for candidate_id, target in zip(
                    labeled_candidates["candidate_id"],
                    labeled_candidates["binary_target"],
                    strict=True,
                )
            )
        }
    )
    fold_set_hash = canonical_contract_sha256(
        {
            "folds": [
                {
                    "fold_index": fold.fold_index,
                    "valid": fold.valid,
                    "train_candidate_ids": list(fold.train_candidate_ids),
                    "test_candidate_ids": list(fold.test_candidate_ids),
                }
                for fold in folds.folds
            ]
        }
    )
    ladder_id = canonical_contract_sha256(
        {
            "view_id": view.view_id,
            "tier": tier.value,
            "calibration_policy_id": calibration_policy_id,
            "label_content_hash": label_content_hash,
            "fold_set_hash": fold_set_hash,
            "rungs": {
                rung.protocol_id: rung.resolved_protocol_hash for rung in rungs
            },
        }
    )
    return SupervisedLadderRun(
        ladder_id=ladder_id,
        view_id=view.view_id,
        tier=tier.value,
        calibration_policy_id=calibration_policy_id,
        rungs=tuple(rungs),
        parity=parity,
        paired_deltas=paired_deltas,
    )
