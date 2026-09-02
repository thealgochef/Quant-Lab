"""Supervised model ladder (`ML_REGIME_CONTRACT_PLAN.md` §2; brief §7B.2).

One immutable view + one fold set feed every rung. Before any delta is
computed the runner asserts identical row-identity sets and identical
``(candidate_id, target, training_prevalence)`` tuples across rungs
(acceptance 7B.22-1) — the rows are the SAME out-of-sample rows, so rung
differences are model differences and nothing else. R6.1 (D13): the row
identity is the bundle-independent ``comparison_row_id`` (schedule, fold
set, fold, candidate, label artifact) so baseline and challenger ARMS with
different ``view_id``s pair exactly; the legacy view-scoped ``oos_row_id``
stays as a column for M0–M3 compatibility and never pairs arms.

Rungs are requested by registered protocol **id only**; planned protocols
refuse fail-closed with their registered reason, and the ladder never
exposes a parameter, threshold, calibrator, or feature-subset enumeration
surface (brief §7B.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from .catboost_bundle_model import (
    CatBoostBundleModelRun,
    bundle_rung_categorical_features,
    run_catboost_bundle_fold_models,
)
from .comparison_rows import (
    COMPARISON_ROW_IDENTITY_KEY,
    LEGACY_ROW_IDENTITY_KEY,
    RegimeFoldFeatureSource,
    candidate_fold_set_id,
    comparison_row_id,
    default_fold_schedule_id,
    label_artifact_content_id,
    label_content_hash,
    with_comparison_row_ids,
)
from .fold_set_artifact import fold_set_id as _fold_set_hash
from .logistic_model import LogisticModelRun, run_logistic_fold_models
from .model_protocols import (
    CATBOOST_BUNDLE_PROTOCOL_ID,
    CATBOOST_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    PREVALENCE_PROTOCOL_ID,
    assert_model_protocol_available,
    assert_single_frozen_selection,
)

__all__ = [
    "DEFAULT_LADDER_PROTOCOLS",
    "DEFAULT_BUNDLE_LADDER_PROTOCOLS",
    "CATBOOST_BUNDLE_REFUSAL",
    "CATBOOST_BUNDLE_RUNG_TIER_REFUSAL",
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

#: R6.1: the bundle-parametrized ladder — prevalence + logistic + the
#: bundle-aware CatBoost rung on identical comparison rows.
DEFAULT_BUNDLE_LADDER_PROTOCOLS: tuple[str, ...] = (
    PREVALENCE_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    CATBOOST_BUNDLE_PROTOCOL_ID,
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
    #: What supplied the feature list: the frozen tier registry (R5) or one
    #: exact resolved feature bundle (R5B bundle-parametrized rungs).
    feature_source: dict[str, Any] = field(default_factory=dict)
    #: R6.1-FIX §3.6 (review RA-05): where the label identity the rungs bound
    #: came from — ``label_artifact`` (the caller passed the exact persisted
    #: label artifact id, e.g. S07's) or ``content_hash_unpersisted`` (the
    #: helper form: the FULL consumed-column content hash; such a run can
    #: never feed a persisted study).
    label_identity_source: str = "label_artifact"

    def rung(self, protocol_id: str) -> LadderRung:
        for rung in self.rungs:
            if rung.protocol_id == protocol_id:
                return rung
        raise KeyError(protocol_id)


def _run_prevalence_reference(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    fold_schedule_id: str,
    fold_set: str,
    label_artifact_id: str,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    """Reference 0 — the fold-local training prevalence as the probability.

    No fit code: the rows are exactly the fitted rungs' rows (same merge,
    same valid-fold loop, same ``oos_row_id`` / ``comparison_row_id``), with
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
                    "comparison_row_id": comparison_row_id(
                        fold_schedule_id=fold_schedule_id,
                        candidate_fold_set_id=fold_set,
                        fold_index=fold.fold_index,
                        candidate_id=candidate_id,
                        label_artifact_id=label_artifact_id,
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


def _parity_key_frame(
    predictions: pd.DataFrame, key: str = COMPARISON_ROW_IDENTITY_KEY
) -> pd.DataFrame:
    columns = [key, "candidate_id", "target", "training_prevalence"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    return predictions[columns].copy()


def _assert_identical_rows(
    rungs: dict[str, pd.DataFrame], *, key: str = COMPARISON_ROW_IDENTITY_KEY
) -> dict[str, Any]:
    """The 7B.22-1 gate: identical rows/folds across every rung, keyed on
    the bundle-independent ``comparison_row_id`` (D13)."""

    reference_name = next(iter(rungs))
    reference = _parity_key_frame(rungs[reference_name], key)
    reference_ids = set(reference[key].astype(str))
    reference_tuples = {
        (
            str(getattr(row, key)),
            str(row.candidate_id),
            int(row.target),
            float(row.training_prevalence),
        )
        for row in reference.itertuples()
    }
    for name, frame in rungs.items():
        keys = _parity_key_frame(frame, key)
        ids = set(keys[key].astype(str))
        if ids != reference_ids:
            raise ValueError(
                f"ladder rung {name!r} does not share the reference rung's exact "
                f"{key} set; rung deltas are undefined (brief §7B.2)"
            )
        tuples = {
            (
                str(getattr(row, key)),
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
        # R5-FIX (gate finding 5): with zero OOS rows there is nothing to
        # compare — the claim is "not evaluable", never "parity held"
        "status": "held" if reference_ids else "not_evaluable",
        "rung_ids": sorted(rungs),
        "row_identity_key": key,
    }


def paired_cell_delta_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    value_column: str = "net_r",
    key_column: str | None = None,
) -> dict[str, Any]:
    """Generalized paired delta keyed on the row identity (DT §4.2).

    R6.1 (D13): the key is the bundle-independent ``comparison_row_id``
    whenever both frames carry it (rungs AND arms pair exactly); frames
    without it (the frozen M0–M3 tier lane's own reports) fall back to the
    legacy view-scoped ``oos_row_id``. ``key_column`` pins the key
    explicitly. The interval is the fixed trading-day block bootstrap; the
    report records ``row_identity_key``.
    """

    if key_column is None:
        both = set(left.columns) & set(right.columns)
        key_column = (
            COMPARISON_ROW_IDENTITY_KEY
            if COMPARISON_ROW_IDENTITY_KEY in both
            else LEGACY_ROW_IDENTITY_KEY
        )
    keys = (key_column, "trading_day", value_column)
    if not set(keys).issubset(left) or not set(keys).issubset(right):
        raise ValueError("cell delta inputs lack paired row-identity/day/value columns")
    if set(left[key_column].astype(str)) != set(right[key_column].astype(str)):
        raise ValueError(f"cell delta requires identical OOS row IDs ({key_column})")
    paired = left[list(keys)].merge(
        right[list(keys)],
        on=key_column,
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
    report = block_bootstrap_interval(
        paired,
        cluster_column="trading_day",
        value_column="delta",
    )
    return {**report, "row_identity_key": key_column}


def _with_brier_loss(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["brier_loss"] = (
        pd.to_numeric(frame["target"], errors="raise")
        - pd.to_numeric(frame["probability"], errors="raise")
    ) ** 2
    return frame


#: The exact fail-closed reason for the FROZEN-LANE CatBoost rung requested
#: under a bundle-parametrized ladder: that fold runner lives in the frozen
#: M0–M3 lane and is tier-locked — never modified in v1. R6.1: the
#: bundle-aware ``ifvg_context_catboost_bundle_v1`` rung is the lawful
#: nonlinear challenger on bundle paths.
CATBOOST_BUNDLE_REFUSAL = (
    "the ifvg_context_catboost_binary_v1 fold runner is tier-locked inside "
    "the frozen M0-M3 lane (never modified in v1); bundle-parametrized "
    "ladders run the prevalence reference, the logistic protocol, and the "
    "bundle-aware ifvg_context_catboost_bundle_v1 rung"
)

#: …and the mirror refusal: the bundle-aware rung needs a resolved bundle
#: identity and has no wiring on the frozen-tier path.
CATBOOST_BUNDLE_RUNG_TIER_REFUSAL = (
    "the bundle-aware ifvg_context_catboost_bundle_v1 rung runs only on the "
    "bundle-parametrized path (bundle_features + bundle_ref); frozen-tier "
    "ladders run ifvg_context_catboost_binary_v1"
)


def run_supervised_ladder(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    tier: ContextFeatureTier | None = None,
    bundle_features: tuple[str, ...] | None = None,
    bundle_ref: str | None = None,
    bundle_evidence_ref: str | None = None,
    protocols: tuple[str, ...] = DEFAULT_LADDER_PROTOCOLS,
    manual_feature_overrides: dict[str, Any] | None = None,
    calibration_policy_id: str = "raw_probability_diagnostics_v1",
    bundle_categorical_features: tuple[str, ...] | None = None,
    fold_local_features: RegimeFoldFeatureSource | None = None,
    fold_schedule_id: str | None = None,
    label_artifact_id: str | None = None,
    label_identity_source: str | None = None,
) -> SupervisedLadderRun:
    """Run every requested rung on identical rows/folds and pair the deltas.

    ``protocols`` accepts registered protocol IDS only (unknown → ValueError;
    planned → fail-closed refusal). The prevalence reference is mandatory —
    a ladder without its reference rung has no skill baseline.

    Feature parametrization is EXACTLY one of:

    * ``tier`` — the R5 frozen-tier registry path (all three rungs);
    * ``bundle_features`` + ``bundle_ref`` — the R5B bundle-parametrized
      path (DECISIONS_TAKEN #41): the exact resolved feature names of one
      resolved bundle, whose ``resolved_feature_bundle_id`` (``bundle_ref``)
      enters the ladder identity, together with ``bundle_evidence_ref``
      (review F2) — the exact evidence artifact (e.g. the MBP-1 feature
      artifact id) whose joined columns the arm's frame carries, so two
      ladders over different evidence can never share one ``ladder_id``.
      The frozen-lane CatBoost rung refuses fail-closed under this path —
      its fold runner is tier-locked; the bundle-aware
      ``ifvg_context_catboost_bundle_v1`` rung is the nonlinear challenger
      here (R6.1 §6.J). ``bundle_categorical_features`` are the
      block-declared categoricals of the bundle (D8; default: the frozen
      registry ∪ the fold-local source's categoricals); ``fold_local_features``
      is the fold-local regime feature seam (D7); ``fold_schedule_id`` /
      ``label_artifact_id`` pin the D13 comparison-row identity (defaults:
      derived from the labeled trading days / the label content hash).
    """

    if (tier is None) == (bundle_features is None):
        raise ValueError(
            "exactly one feature parametrization is lawful: tier XOR "
            "(bundle_features + bundle_ref)"
        )
    if bundle_features is not None and not bundle_ref:
        raise ValueError(
            "bundle-parametrized ladders require the resolved bundle id "
            "(bundle_ref) inside the ladder identity"
        )
    if tier is not None and bundle_evidence_ref is not None:
        raise ValueError(
            "an evidence artifact reference is lawful only on the "
            "bundle-parametrized path"
        )
    if len(set(protocols)) != len(protocols):
        raise ValueError("ladder protocols contain duplicates")
    for protocol_id in protocols:
        assert_model_protocol_available(protocol_id)
    if PREVALENCE_PROTOCOL_ID not in protocols:
        raise ValueError(
            "the training-prevalence reference rung is mandatory for every ladder"
        )
    if bundle_features is not None and CATBOOST_PROTOCOL_ID in protocols:
        raise ValueError(CATBOOST_BUNDLE_REFUSAL)
    if tier is not None and CATBOOST_BUNDLE_PROTOCOL_ID in protocols:
        raise ValueError(CATBOOST_BUNDLE_RUNG_TIER_REFUSAL)
    if tier is not None and (
        fold_local_features is not None or bundle_categorical_features is not None
    ):
        raise ValueError(
            "fold-local features and block-declared categoricals are lawful only on "
            "the bundle-parametrized path"
        )
    assert_single_frozen_selection("calibrator", (calibration_policy_id,))
    assert_calibration_policy_executable(calibration_policy_id)

    features = features_for_tier(tier) if tier is not None else tuple(bundle_features)
    # D13: the row identity every rung shares — schedule, fold set, labels
    schedule_id = fold_schedule_id or default_fold_schedule_id(folds, labeled_candidates)
    # R6.1-FIX §3.6 (review RA-05): without the exact persisted label artifact
    # id the ladder binds the FULL consumed-column content hash and is stamped
    # unpersistable — the narrow (candidate, target) pair hash is never the
    # label identity of a rung
    if label_artifact_id is not None:
        label_id = str(label_artifact_id)
        resolved_label_source = label_identity_source or "label_artifact"
    else:
        label_id = label_artifact_content_id(None, labeled_candidates)
        resolved_label_source = "content_hash_unpersisted"
    if resolved_label_source not in ("label_artifact", "content_hash_unpersisted"):
        raise ValueError(
            "label_identity_source must be 'label_artifact' or 'content_hash_unpersisted'"
        )
    fold_set = candidate_fold_set_id(folds)
    if bundle_features is not None:
        block_declared = tuple(bundle_categorical_features or ()) + (
            tuple(fold_local_features.categorical_features)
            if fold_local_features is not None
            else ()
        )
        rung_categoricals = bundle_rung_categorical_features(
            features, block_declared=tuple(dict.fromkeys(block_declared))
        )
    else:
        rung_categoricals = ()
    rungs: list[LadderRung] = []
    frames: dict[str, pd.DataFrame] = {}
    for protocol_id in protocols:
        if protocol_id == PREVALENCE_PROTOCOL_ID:
            predictions, fold_reports = _run_prevalence_reference(
                view,
                labeled_candidates,
                folds,
                fold_schedule_id=schedule_id,
                fold_set=fold_set,
                label_artifact_id=label_id,
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
                fold_local_features=fold_local_features,
                fold_schedule_id=schedule_id,
                label_artifact_id=label_id,
            )
            predictions = run.predictions
            fold_reports = run.fold_reports
            resolved_hash = run.protocol.resolved_hash
            importance = run.feature_importance
        elif protocol_id == CATBOOST_BUNDLE_PROTOCOL_ID:
            bundle_run: CatBoostBundleModelRun = run_catboost_bundle_fold_models(
                view,
                labeled_candidates,
                folds,
                features=features,
                resolved_feature_bundle_id=str(bundle_ref),
                categorical_features=rung_categoricals,
                fold_local_features=fold_local_features,
                fold_schedule_id=schedule_id,
                label_artifact_id=label_id,
                manual_feature_overrides=manual_feature_overrides,
            )
            predictions = bundle_run.predictions
            fold_reports = bundle_run.fold_reports
            resolved_hash = bundle_run.protocol.resolved_hash
            importance = bundle_run.feature_importance
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
        # D13: every rung carries (and agrees on) the comparison-row identity —
        # the frozen M0–M3 CatBoost lane emits only the legacy id, so the
        # ladder derives it from the same fold/candidate under the same
        # schedule/fold set/label artifact
        predictions = with_comparison_row_ids(
            predictions,
            fold_schedule_id=schedule_id,
            candidate_fold_set_id=fold_set,
            label_artifact_id=label_id,
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
        for challenger_id in (CATBOOST_PROTOCOL_ID, CATBOOST_BUNDLE_PROTOCOL_ID):
            if (
                LOGISTIC_PROTOCOL_ID in frames
                and challenger_id in frames
                and not frames[LOGISTIC_PROTOCOL_ID].empty
                and not frames[challenger_id].empty
            ):
                paired_deltas[f"{LOGISTIC_PROTOCOL_ID}__vs__{challenger_id}"] = (
                    paired_cell_delta_report(
                        _with_brier_loss(frames[LOGISTIC_PROTOCOL_ID]),
                        _with_brier_loss(frames[challenger_id]),
                        value_column="brier_loss",
                    )
                )

    # adversarial m-9: the ladder id binds the LABELS and FOLDS it ran on,
    # not just the view/protocols — two labelings can never share one id
    labels_hash = label_content_hash(labeled_candidates)
    # R6.1: the ONE legacy row-population hash (``fold_set_artifact.fold_set_id``)
    fold_set_hash = _fold_set_hash(folds)
    feature_source: dict[str, Any] = (
        {"kind": "frozen_tier", "tier": tier.value}
        if tier is not None
        else {
            "kind": "resolved_bundle",
            "resolved_feature_bundle_id": bundle_ref,
            "feature_names": list(features),
            # review F2: the exact evidence artifact behind any joined
            # columns rides the identity — None means the bundle's columns
            # come entirely from the immutable candidate view
            "evidence_ref": bundle_evidence_ref,
            # R6.1 (D7/D8): the fold-local regime feature artifact joined per
            # fold and the block-declared categoricals of the bundle
            "fold_local_feature_artifact_id": (
                str(fold_local_features.artifact_id)
                if fold_local_features is not None
                else None
            ),
            "block_declared_categorical_features": list(rung_categoricals),
            # D13: the comparison-row identity the rungs share
            "fold_schedule_id": schedule_id,
            "label_artifact_id": label_id,
        }
    )
    ladder_id = canonical_contract_sha256(
        {
            "view_id": view.view_id,
            "feature_source": feature_source,
            "calibration_policy_id": calibration_policy_id,
            "label_content_hash": labels_hash,
            "fold_set_hash": fold_set_hash,
            "rungs": {
                rung.protocol_id: rung.resolved_protocol_hash for rung in rungs
            },
        }
    )
    return SupervisedLadderRun(
        ladder_id=ladder_id,
        view_id=view.view_id,
        tier=tier.value if tier is not None else "resolved_bundle",
        calibration_policy_id=calibration_policy_id,
        rungs=tuple(rungs),
        parity=parity,
        paired_deltas=paired_deltas,
        feature_source=feature_source,
        label_identity_source=resolved_label_source,
    )
