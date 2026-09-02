"""Controlled Baseline vs Baseline+MBP-1 study workflow (R5B deliverable 10).

Two arms over IDENTICAL profile/candidate rows, labels, folds, and model
protocol: the challenger bundle must carry the activated MBP-1 order-flow
block, and its baseline is the challenger's OWN base bundle (composition by
extension — the bundle graph is the preregistered block ablation, the
permitted feature-selection form of brief §7B.3). Both arms run the
bundle-parametrized supervised ladder — the prevalence reference, the
logistic protocol, and (R6.1 §6.J) the bundle-aware CatBoost rung
``ifvg_context_catboost_bundle_v1`` (the frozen-lane CatBoost runner stays
tier-locked) — on identical rows/labels/folds/purge/embargo/costs/seed/
hyperparameters, and are gated by the same identical-rows assertion on the
bundle-independent ``comparison_row_id`` (D13) before any delta is
computed. Paired OOS Brier deltas are reported per rung; the pinned
``headline_protocol_id`` selects the headline comparison.

The persisted study artifact pins every input identity (resolved bundle
ids, the candidate view, the exact MBP-1 feature artifact, label and fold
hashes, the model protocol) plus the paired Brier delta with its
trading-day block-bootstrap interval, and carries the permanent
``research_only_offline`` boundary stamp (owner decision R-6). The
prevalence reference arms must be IDENTICAL across arms — a differing
reference means the arms did not share rows/folds and the study refuses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import Field

from ..context_feature_view import CandidateFeatureView
from ..context_folds import ContextFoldSet
from ..features.bundle_feature_view import (
    build_bundle_feature_view,
    mbp1_block_keys_in_bundle,
)
from ..features.feature_bundles import FEATURE_BUNDLE_REGISTRY, resolve_bundle
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
    register_identity_pair,
)
from ..search.store import (
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .comparison_rows import (
    COMPARISON_ROW_IDENTITY_KEY,
    default_fold_schedule_id,
    label_artifact_content_id,
)
from .fold_set_artifact import fold_set_id as _legacy_fold_set_id
from .model_protocols import (
    CATBOOST_BUNDLE_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    PREVALENCE_PROTOCOL_ID,
)
from .supervised_ladder import (
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    SupervisedLadderRun,
    paired_cell_delta_report,
    run_supervised_ladder,
)

__all__ = [
    "CONTROLLED_STUDY_STORE",
    "HEADLINE_PROTOCOL_IDS",
    "assert_cross_arm_identity",
    "ControlledFeatureStudyPayload",
    "ControlledFeatureStudyEnvelope",
    "ControlledFeatureStudyRun",
    "run_controlled_mbp1_study",
    "save_controlled_feature_study",
    "load_controlled_feature_study",
    "load_controlled_study_detail",
]

CONTROLLED_STUDY_STORE = "controlled_feature_studies"
_DETAIL_SIDECAR = "controlled_study_detail.json"
#: The rungs a controlled study may pin as its headline comparison.
HEADLINE_PROTOCOL_IDS: tuple[str, ...] = (LOGISTIC_PROTOCOL_ID, CATBOOST_BUNDLE_PROTOCOL_ID)
_Scalar = str | int | float | bool | None


def _jsonable(value: Any) -> Any:
    """NaN/Inf-safe JSON projection (typed nulls, never invalid JSON)."""

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (pd.isna(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ControlledFeatureStudyPayload(FrozenContract):
    baseline_bundle_key: str
    challenger_bundle_key: str
    baseline_resolved_bundle_id: str = Field(pattern=SHA256_PATTERN)
    challenger_resolved_bundle_id: str = Field(pattern=SHA256_PATTERN)
    view_id: str = Field(pattern=SHA256_PATTERN)
    mbp1_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    label_content_hash: str = Field(pattern=SHA256_PATTERN)
    fold_set_hash: str = Field(pattern=SHA256_PATTERN)
    model_protocol_id: str
    calibration_policy_id: str
    baseline_ladder_id: str = Field(pattern=SHA256_PATTERN)
    challenger_ladder_id: str = Field(pattern=SHA256_PATTERN)
    oos_row_count: int = Field(ge=0)
    parity_status: Literal["held", "not_evaluable"]
    #: the HEADLINE rung's summaries / delta (``model_protocol_id``)
    baseline_summary: ImmutableMap[str, Any]
    challenger_summary: ImmutableMap[str, Any]
    paired_brier_delta: ImmutableMap[str, Any] | None
    #: R6.1: every rung of both arms — prevalence + logistic + the
    #: bundle-aware CatBoost rung — on identical comparison rows (D13)
    ladder_protocol_ids: tuple[str, ...] = (LOGISTIC_PROTOCOL_ID,)
    row_identity_key: Literal["comparison_row_id"] = COMPARISON_ROW_IDENTITY_KEY
    fold_schedule_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    #: R6.1-FIX §3.6 (F-08): the EXACT label artifact identity — mandatory,
    #: 64-hex; ``label_identity_source`` says whether it is the persisted
    #: label artifact (S07's policy-bearing id over every consumed column) or
    #: the unpersisted helper's full consumed-column content hash. Only the
    #: former may be saved, compared as an immutable study, or promoted.
    label_artifact_id: str = Field(pattern=SHA256_PATTERN)
    #: REQUIRED (review RA-05): never a self-declared default
    label_identity_source: Literal["label_artifact", "content_hash_unpersisted"]
    baseline_rung_summaries: ImmutableMap[str, ImmutableMap[str, _Scalar]] = ImmutableMap()
    challenger_rung_summaries: ImmutableMap[str, ImmutableMap[str, _Scalar]] = ImmutableMap()
    paired_brier_deltas: ImmutableMap[str, ImmutableMap[str, _Scalar] | None] = ImmutableMap()
    research_boundary: Literal["research_only_offline"] = "research_only_offline"


class ControlledFeatureStudyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "controlled_feature_study_id"

    controlled_feature_study_id: str = Field(pattern=SHA256_PATTERN)
    payload: ControlledFeatureStudyPayload
    #: post-materialization fact binding the full diagnostics sidecar
    detail_sha256: str = Field(pattern=SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class ControlledFeatureStudyRun:
    envelope: ControlledFeatureStudyEnvelope
    baseline: SupervisedLadderRun
    challenger: SupervisedLadderRun
    detail_bytes: bytes


def _summary(run: SupervisedLadderRun, protocol_id: str = LOGISTIC_PROTOCOL_ID) -> dict[str, Any]:
    rung = run.rung(protocol_id)
    report = dict(rung.prediction_report or {})
    return _jsonable(
        {
            "oos_row_count": run.parity.get("oos_row_count", 0),
            "status": report.get("status"),
            "brier_score": report.get("brier_score"),
            "reference_brier_score": report.get("reference_brier_score"),
            "brier_skill_score": report.get("brier_skill_score"),
            "log_loss": report.get("log_loss"),
            "auc": report.get("auc"),
        }
    )


def _label_hash(labeled_candidates: pd.DataFrame) -> str:
    return canonical_contract_sha256(
        {
            "labeled_pairs": sorted(
                (str(cid), None if pd.isna(target) else int(target))
                for cid, target in zip(
                    labeled_candidates["candidate_id"],
                    labeled_candidates["binary_target"],
                    strict=True,
                )
            )
        }
    )


def _fold_hash(folds: ContextFoldSet) -> str:
    """The ONE legacy row-population hash (R6.1 delegation)."""

    return _legacy_fold_set_id(folds)


def assert_cross_arm_identity(
    baseline: SupervisedLadderRun,
    challenger: SupervisedLadderRun,
    *,
    protocols: tuple[str, ...] = DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    key: str = COMPARISON_ROW_IDENTITY_KEY,
) -> None:
    """The two arms must be the SAME rows/folds — proven, never assumed.

    R6.1 (D13): keyed on the bundle-independent ``comparison_row_id`` —
    the arms are different bundle views (different ``view_id``s), so the
    legacy ``oos_row_id`` can never pair them.
    """

    for protocol_id in protocols:
        left = baseline.rung(protocol_id).predictions
        right = challenger.rung(protocol_id).predictions
        left_keys = {
            (
                str(getattr(r, key)),
                str(r.candidate_id),
                int(r.target),
                float(r.training_prevalence),
            )
            for r in left.itertuples()
        }
        right_keys = {
            (
                str(getattr(r, key)),
                str(r.candidate_id),
                int(r.target),
                float(r.training_prevalence),
            )
            for r in right.itertuples()
        }
        if left_keys != right_keys:
            raise ValueError(
                f"controlled-study arms disagree on the exact OOS rows for "
                f"{protocol_id} (keyed on {key}); the paired delta is undefined "
                "(identical rows/labels/folds are the study's premise)"
            )
    # the prevalence REFERENCE must be numerically identical across arms:
    # features cannot change it, so a difference means the arms diverged
    left = baseline.rung(PREVALENCE_PROTOCOL_ID).predictions
    right = challenger.rung(PREVALENCE_PROTOCOL_ID).predictions
    if not left.empty:
        merged = left[[key, "probability"]].merge(
            right[[key, "probability"]],
            on=key,
            validate="one_to_one",
            suffixes=("_baseline", "_challenger"),
        )
        if not (
            merged["probability_baseline"] == merged["probability_challenger"]
        ).all():
            raise ValueError(
                "the prevalence reference differs across arms — the arms did "
                "not share fold-local training rows"
            )


#: R5B name kept for callers; the gate keys on ``comparison_row_id`` now.
_assert_cross_arm_identity = assert_cross_arm_identity


def _with_brier(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["brier_loss"] = (
        pd.to_numeric(out["target"], errors="raise")
        - pd.to_numeric(out["probability"], errors="raise")
    ) ** 2
    return out


def run_controlled_mbp1_study(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    challenger_bundle_key: str,
    mbp1_features: pd.DataFrame,
    mbp1_feature_artifact,
    calibration_policy_id: str = "raw_probability_diagnostics_v1",
    headline_protocol_id: str = LOGISTIC_PROTOCOL_ID,
    fold_schedule_id: str | None = None,
    label_artifact_id: str | None = None,
) -> ControlledFeatureStudyRun:
    """``mbp1_feature_artifact`` is the ``Mbp1FeatureArtifactEnvelope`` whose
    table the supplied frame must BE — the binding is verified by rehash at
    the bundle-view seam (review F1), and its id rides the challenger
    ladder's identity (review F2). Both arms run prevalence + logistic + the
    bundle-aware CatBoost rung; ``headline_protocol_id`` (logistic or the
    bundle rung) selects the headline summaries/delta; ``fold_schedule_id``
    / ``label_artifact_id`` pin the D13 comparison-row identity (the
    pipeline passes S08's / S07's ids; defaults derive from the inputs)."""

    if headline_protocol_id not in HEADLINE_PROTOCOL_IDS:
        raise ValueError(
            f"headline protocol {headline_protocol_id!r} is not a controlled-study "
            f"rung; lawful: {HEADLINE_PROTOCOL_IDS}"
        )
    challenger_env = resolve_bundle(challenger_bundle_key)
    if not mbp1_block_keys_in_bundle(challenger_env):
        raise ValueError(
            f"bundle {challenger_bundle_key!r} carries no MBP-1 order-flow "
            "features; the controlled MBP-1 study requires an order-flow "
            "challenger"
        )
    definition = FEATURE_BUNDLE_REGISTRY[challenger_bundle_key]
    if definition.base_bundle_key is None:
        raise ValueError(
            f"bundle {challenger_bundle_key!r} has no base bundle; the "
            "controlled study baseline is the challenger's own base"
        )
    baseline_env = resolve_bundle(definition.base_bundle_key)
    if mbp1_block_keys_in_bundle(baseline_env):
        raise ValueError(
            "the challenger's base bundle also carries MBP-1 features; the "
            "controlled contrast would not isolate the block"
        )

    mbp1_feature_artifact_id = mbp1_feature_artifact.mbp1_feature_artifact_id
    baseline_view_env, baseline_frame = build_bundle_feature_view(view, baseline_env)
    challenger_view_env, challenger_frame = build_bundle_feature_view(
        view,
        challenger_env,
        mbp1_features=mbp1_features,
        mbp1_feature_artifact=mbp1_feature_artifact,
    )
    # both arms keep the SAME view_id: oos_row_id = hash{view_id, fold,
    # candidate} is deliberately feature-independent so rows pair exactly;
    # the evidence divergence is pinned by the view payload's and the
    # ladder identity's mbp1_feature_artifact_id (reviews F1/F2)
    baseline_arm_view = replace(view, frame=baseline_frame)
    challenger_arm_view = replace(view, frame=challenger_frame)
    arm_protocols = DEFAULT_BUNDLE_LADDER_PROTOCOLS
    schedule_id = fold_schedule_id or default_fold_schedule_id(folds, labeled_candidates)
    # R6.1-FIX §3.6: a persisting caller passes S07's exact label artifact id;
    # a helper run without one carries the FULL consumed-column content hash
    # and is stamped unpersistable
    if label_artifact_id is not None:
        label_id = str(label_artifact_id)
        label_identity_source = "label_artifact"
    else:
        label_id = label_artifact_content_id(None, labeled_candidates)
        label_identity_source = "content_hash_unpersisted"
    baseline_run = run_supervised_ladder(
        baseline_arm_view,
        labeled_candidates,
        folds,
        bundle_features=tuple(baseline_env.payload.resolved_feature_names),
        bundle_ref=baseline_env.resolved_feature_bundle_id,
        protocols=arm_protocols,
        calibration_policy_id=calibration_policy_id,
        fold_schedule_id=schedule_id,
        label_artifact_id=label_id,
        label_identity_source=label_identity_source,
    )
    challenger_run = run_supervised_ladder(
        challenger_arm_view,
        labeled_candidates,
        folds,
        bundle_features=tuple(challenger_env.payload.resolved_feature_names),
        bundle_ref=challenger_env.resolved_feature_bundle_id,
        bundle_evidence_ref=mbp1_feature_artifact_id,
        protocols=arm_protocols,
        calibration_policy_id=calibration_policy_id,
        fold_schedule_id=schedule_id,
        label_artifact_id=label_id,
        label_identity_source=label_identity_source,
    )
    assert_cross_arm_identity(baseline_run, challenger_run, protocols=arm_protocols)

    paired_deltas: dict[str, ImmutableMap | None] = {}
    for protocol_id in arm_protocols:
        if protocol_id == PREVALENCE_PROTOCOL_ID:
            continue
        left = baseline_run.rung(protocol_id).predictions
        right = challenger_run.rung(protocol_id).predictions
        if left.empty or right.empty:
            paired_deltas[protocol_id] = None
            continue
        paired_deltas[protocol_id] = ImmutableMap(
            _jsonable(
                paired_cell_delta_report(
                    _with_brier(left), _with_brier(right), value_column="brier_loss"
                )
            )
        )
    paired_delta = paired_deltas[headline_protocol_id]

    payload = ControlledFeatureStudyPayload(
        baseline_bundle_key=baseline_env.payload.feature_bundle_key,
        challenger_bundle_key=challenger_bundle_key,
        baseline_resolved_bundle_id=baseline_env.resolved_feature_bundle_id,
        challenger_resolved_bundle_id=challenger_env.resolved_feature_bundle_id,
        view_id=view.view_id,
        mbp1_feature_artifact_id=mbp1_feature_artifact_id,
        label_content_hash=_label_hash(labeled_candidates),
        fold_set_hash=_fold_hash(folds),
        model_protocol_id=headline_protocol_id,
        calibration_policy_id=calibration_policy_id,
        baseline_ladder_id=baseline_run.ladder_id,
        challenger_ladder_id=challenger_run.ladder_id,
        oos_row_count=int(baseline_run.parity.get("oos_row_count", 0)),
        parity_status=baseline_run.parity.get("status", "not_evaluable"),
        baseline_summary=_summary(baseline_run, headline_protocol_id),
        challenger_summary=_summary(challenger_run, headline_protocol_id),
        paired_brier_delta=dict(paired_delta) if paired_delta is not None else None,
        ladder_protocol_ids=tuple(arm_protocols),
        fold_schedule_id=schedule_id,
        label_artifact_id=label_id,
        label_identity_source=label_identity_source,
        baseline_rung_summaries=ImmutableMap(
            {
                protocol_id: ImmutableMap(_summary(baseline_run, protocol_id))
                for protocol_id in arm_protocols
                if protocol_id != PREVALENCE_PROTOCOL_ID
            }
        ),
        challenger_rung_summaries=ImmutableMap(
            {
                protocol_id: ImmutableMap(_summary(challenger_run, protocol_id))
                for protocol_id in arm_protocols
                if protocol_id != PREVALENCE_PROTOCOL_ID
            }
        ),
        paired_brier_deltas=ImmutableMap(paired_deltas),
    )
    detail = {
        "baseline_view_envelope": baseline_view_env.model_dump(mode="json"),
        "challenger_view_envelope": challenger_view_env.model_dump(mode="json"),
        "baseline": {
            "parity": _jsonable(baseline_run.parity),
            "paired_deltas": _jsonable(baseline_run.paired_deltas),
            "rungs": {
                rung.protocol_id: _jsonable(rung.prediction_report)
                for rung in baseline_run.rungs
            },
        },
        "challenger": {
            "parity": _jsonable(challenger_run.parity),
            "paired_deltas": _jsonable(challenger_run.paired_deltas),
            "rungs": {
                rung.protocol_id: _jsonable(rung.prediction_report)
                for rung in challenger_run.rungs
            },
        },
    }
    detail_bytes = (json.dumps(detail, sort_keys=True) + "\n").encode("utf-8")
    envelope = ControlledFeatureStudyEnvelope.from_payload(
        payload, detail_sha256=hashlib.sha256(detail_bytes).hexdigest()
    )
    return ControlledFeatureStudyRun(
        envelope=envelope,
        baseline=baseline_run,
        challenger=challenger_run,
        detail_bytes=detail_bytes,
    )


def save_controlled_feature_study(root: Path, run: ControlledFeatureStudyRun) -> tuple:
    """Persist ONLY a study bound to the exact label artifact (R6.1-FIX §3.6):
    an unpersisted helper run (``content_hash_unpersisted``) is refused."""

    if run.envelope.payload.label_identity_source != "label_artifact":
        raise PermissionError(
            "a controlled feature study whose label identity is content_hash_unpersisted "
            "cannot be saved, compared as an immutable study, or promoted; pass the exact "
            "persisted label artifact id (S07) to run_controlled_mbp1_study"
        )
    return save_or_reuse_envelope(
        Path(root),
        CONTROLLED_STUDY_STORE,
        run.envelope,
        extra_files={_DETAIL_SIDECAR: run.detail_bytes},
    )


def load_controlled_feature_study(
    root: Path, study_id: str
) -> ControlledFeatureStudyEnvelope:
    return load_verified_envelope(
        Path(root), CONTROLLED_STUDY_STORE, study_id, ControlledFeatureStudyEnvelope
    )


def load_controlled_study_detail(
    root: Path, envelope: ControlledFeatureStudyEnvelope
) -> dict[str, Any]:
    data = load_sidecar_bytes(
        Path(root),
        CONTROLLED_STUDY_STORE,
        envelope.controlled_feature_study_id,
        _DETAIL_SIDECAR,
    )
    if hashlib.sha256(data).hexdigest() != envelope.detail_sha256:
        raise ValueError("controlled-study detail sidecar fails the envelope hash check")
    return json.loads(data.decode("utf-8"))


def _example_controlled_study_payload() -> ControlledFeatureStudyPayload:
    return ControlledFeatureStudyPayload(
        baseline_bundle_key="B0_CORE",
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        baseline_resolved_bundle_id="a" * 64,
        challenger_resolved_bundle_id="b" * 64,
        view_id="c" * 64,
        mbp1_feature_artifact_id="d" * 64,
        label_content_hash="e" * 64,
        fold_set_hash="f" * 64,
        model_protocol_id=LOGISTIC_PROTOCOL_ID,
        calibration_policy_id="raw_probability_diagnostics_v1",
        baseline_ladder_id="1" * 64,
        challenger_ladder_id="2" * 64,
        oos_row_count=0,
        parity_status="not_evaluable",
        baseline_summary={},
        challenger_summary={},
        paired_brier_delta=None,
        label_artifact_id="3" * 64,
        label_identity_source="label_artifact",
    )


register_identity_pair(
    name="ControlledFeatureStudy",
    envelope_cls=ControlledFeatureStudyEnvelope,
    payload_cls=ControlledFeatureStudyPayload,
    id_field="controlled_feature_study_id",
    example_factory=_example_controlled_study_payload,
    extra_envelope_fields=("detail_sha256",),
)
