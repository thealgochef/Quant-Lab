"""Regime study requests, readiness, and the deterministic S10 decisions
(R6.1 §6.E; D1, D4, D14).

A :class:`RegimeStudyRequest` is the FROZEN, semantic-identity-bearing
description of one regime study inside a pipeline specification: the
algorithm, grain (candidate rows or a 5m/15m context-bar panel), the input
bundle + features, k, winsorization, bootstrap budget, the stratified
reporting classes requested, and — for MODEL-BEARING studies only
(``feature_only`` / ``cohort_model``) — the EXACT frozen promotion decision,
owner-decision artifact, and capability assessment ids that authorize them
(D1). Descriptive requests omit the three refs; no execution path ever
resolves a mutable "latest" status.

Readiness (:func:`regime_study_block_reason`) re-runs every registry /
algorithm / bundle / input / leakage / grain-coherence / stage-plan check
by resolving a PROBE protocol through ``resolve_kmeans_protocol`` (the same
gate S05 uses), and verified-loads the frozen authority artifacts for
model-bearing requests when a store root is given — a plan referencing an
absent, superseded, mismatched, or under-status authority fails BEFORE any
path is constructed.

S10 (:func:`build_s10_decisions`) deterministically derives the initial
status (DESCRIPTIVE_ONLY or a BLOCKED state) from the assessment and — when
the ratification-free structural gates pass and descriptive classes were
requested — the STRATIFICATION_READY decision (D4/D6); ``decided_at`` is the
evidence as-of instant (the maximum observation timestamp), never the wall
clock, so a re-run mints identical decision ids and every stage REUSES.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field, model_validator

from ..features.context_bar_panel_contract import (
    CONTEXT_BAR_PANEL_BUNDLE_KEY,
    PANEL_AS_OF_POLICY_ID_V1,
    PANEL_AS_OF_POLICY_REGISTRY,
    PANEL_INTERVALS_SECONDS_V1,
)
from ..features.feature_blocks import AvailabilityStage, BlockUnavailableError
from ..search.identities import SHA256_PATTERN, FrozenContract
from ..search.store import SearchStoreError
from .regime_algorithms import (
    KMEANS_ALGORITHM_KEY,
    RegimeAlgorithmUnavailableError,
    assert_regime_algorithm_fittable,
)
from .regime_contracts import (
    ObservationGranularity,
    RegimeCapabilityAssessmentEnvelope,
    RegimeLeakageError,
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeProtocolEnvelope,
    RegimeRole,
    RegimeStatus,
)
from .regime_fold_features import HARD_ID_ENCODING_NONE, HardIdEncoding

__all__ = [
    "COMPARISON_CLASSES",
    "DESCRIPTIVE_CLASSES",
    "SUPERVISED_CLASSES",
    "RegimeStudyRequest",
    "RegimeStudyStagePlanFacts",
    "regime_study_block_reason",
    "resolve_study_protocol",
    "derive_initial_regime_status",
    "regime_evidence_as_of",
    "build_s10_decisions",
    "verify_frozen_authority",
    "FrozenAuthority",
]

#: The five ML §5.5 comparison classes (owner planning decision Q4).
COMPARISON_CLASSES: tuple[str, ...] = (
    "cohort_descriptive",
    "feature_only",
    "cohort_model",
    "stratified_prop",
    "stratified_frontier",
)
#: Classes that FIT models on regime features — FEATURE_ELIGIBLE required.
SUPERVISED_CLASSES: tuple[str, ...] = ("feature_only", "cohort_model")
#: Classes that only DESCRIBE — STRATIFICATION_READY suffices.
DESCRIPTIVE_CLASSES: tuple[str, ...] = (
    "cohort_descriptive",
    "stratified_prop",
    "stratified_frontier",
)


class RegimeStudyRequest(FrozenContract):
    """The frozen regime-study request inside a pipeline specification (D1)."""

    algorithm_key: str = KMEANS_ALGORITHM_KEY
    observation_granularity: ObservationGranularity = ObservationGranularity.CANDIDATE_STAGE_ROW
    observation_stage: AvailabilityStage = AvailabilityStage.ENTRY_DECISION
    panel_interval_seconds: int | None = None
    panel_as_of_policy_id: str | None = None
    input_feature_bundle_key: str
    resolved_input_features: tuple[str, ...] = Field(min_length=1)
    resolved_cluster_count: int = Field(default=3, ge=2)
    winsorization_policy: Literal["none", "clip_p01_p99_train_fitted_v1"] = "none"
    bootstrap_refits: int = Field(default=50, ge=1)
    stratified_reporting_requested: bool = False
    comparison_classes_requested: tuple[str, ...] = ()
    supervised_bundle_key: str | None = None
    #: the ONE hard-id vocabulary (F11) — shared with the fold-feature artifact
    hard_id_encoding: HardIdEncoding = HARD_ID_ENCODING_NONE
    #: exact frozen authority (model-bearing requests only; D1/D4)
    regime_promotion_decision_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    owner_decision_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    required_capability_assessment_id: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @property
    def requires_supervision(self) -> bool:
        return any(name in SUPERVISED_CLASSES for name in self.comparison_classes_requested)

    @property
    def is_panel(self) -> bool:
        return self.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL

    @property
    def descriptive_classes(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.comparison_classes_requested if name in DESCRIPTIVE_CLASSES
        )

    @property
    def authority_refs(self) -> tuple[str | None, str | None, str | None]:
        return (
            self.regime_promotion_decision_id,
            self.owner_decision_artifact_id,
            self.required_capability_assessment_id,
        )

    def stage_plan_problems(
        self,
        *,
        feature_bundle_ids: tuple[str, ...],
        stage_values: tuple[str, ...],
        label_policy_id: str | None,
        model_protocol_id: str | None,
    ) -> list[str]:
        """The STRUCTURAL lawful-plan rules of a regime study (cheap; shared by
        the semantic-spec validator and readiness): required stages, the
        bundle membership, the panel prerequisites, the stratified-reporting
        stage, and the supervised prerequisites."""

        stages = set(stage_values)
        problems: list[str] = []
        missing = [stage for stage in _REQUIRED_STAGES if stage not in stages]
        if missing:
            problems.append(f"a regime study requires stages {missing} in the plan")
        for key in (self.input_feature_bundle_key, self.supervised_bundle_key):
            if key is not None and key not in feature_bundle_ids:
                problems.append(
                    f"regime study bundle {key!r} must be one of the plan's feature "
                    f"bundles {tuple(feature_bundle_ids)}"
                )
        if self.is_panel:
            if _STAGE_S04 not in stages:
                problems.append(
                    "a panel-grain regime study requires 04_build_or_reuse_replay_charts"
                )
            if self.input_feature_bundle_key != CONTEXT_BAR_PANEL_BUNDLE_KEY:
                problems.append(
                    "a panel-grain regime study must use the "
                    f"{CONTEXT_BAR_PANEL_BUNDLE_KEY} bundle"
                )
        if self.stratified_reporting_requested and _STAGE_S14 not in stages:
            problems.append(
                "stratified regime reporting requires 14_build_frontier_and_insights"
            )
        if self.requires_supervision:
            if _STAGE_S07 not in stages or not label_policy_id:
                problems.append(
                    "feature_only / cohort_model regime studies require 07_derive_labels "
                    "+ a label policy"
                )
            if not model_protocol_id:
                problems.append(
                    "feature_only / cohort_model regime studies require a supervised "
                    "model protocol"
                )
        return problems

    @model_validator(mode="after")
    def _lawful(self):
        grain = ObservationGranularity(self.observation_granularity)
        if grain is ObservationGranularity.DECISION_ROW:
            raise ValueError(
                "decision_row grain is refused: no decision-row observation source is "
                "wired in V1 (candidate_stage_row or context_bar_panel only)"
            )
        if grain is ObservationGranularity.CONTEXT_BAR_PANEL:
            if self.panel_interval_seconds not in PANEL_INTERVALS_SECONDS_V1:
                raise ValueError(
                    f"panel_interval_seconds={self.panel_interval_seconds} is not "
                    f"owner-registered; registered: {PANEL_INTERVALS_SECONDS_V1}"
                )
            if self.panel_as_of_policy_id not in PANEL_AS_OF_POLICY_REGISTRY:
                raise ValueError(
                    f"panel_as_of_policy_id {self.panel_as_of_policy_id!r} is not registered; "
                    f"registered: {tuple(PANEL_AS_OF_POLICY_REGISTRY)}"
                )
        elif self.panel_interval_seconds is not None or self.panel_as_of_policy_id is not None:
            raise ValueError("panel fields are legal only for the context_bar_panel grain")
        if len(set(self.resolved_input_features)) != len(self.resolved_input_features):
            raise ValueError("resolved_input_features must not repeat a feature")
        unknown = sorted(set(self.comparison_classes_requested) - set(COMPARISON_CLASSES))
        if unknown:
            raise ValueError(f"unregistered comparison classes {unknown}")
        if len(set(self.comparison_classes_requested)) != len(self.comparison_classes_requested):
            raise ValueError("comparison_classes_requested must not repeat a class")
        if self.comparison_classes_requested and not self.stratified_reporting_requested:
            raise ValueError(
                "comparison classes require stratified_reporting_requested=True"
            )
        if self.stratified_reporting_requested and not self.comparison_classes_requested:
            raise ValueError("stratified reporting requires at least one comparison class")
        refs = self.authority_refs
        present = tuple(ref is not None for ref in refs)
        if self.requires_supervision:
            if self.supervised_bundle_key is None:
                raise ValueError(
                    "feature_only / cohort_model studies require supervised_bundle_key"
                )
            if not all(present):
                raise ValueError(
                    "model-bearing regime studies (feature_only / cohort_model) must freeze "
                    "the EXACT regime_promotion_decision_id, owner_decision_artifact_id, and "
                    "required_capability_assessment_id — no latest-status lookup exists"
                )
        else:
            if any(present):
                raise ValueError(
                    "descriptive-only regime studies must omit the promotion / owner / "
                    "assessment authority refs (they enter model-bearing identities only)"
                )
            if self.supervised_bundle_key is not None:
                raise ValueError("supervised_bundle_key is legal only with supervised classes")
        return self


@dataclass(frozen=True, slots=True)
class RegimeStudyStagePlanFacts:
    """The pipeline-level facts readiness needs (no pipeline import here)."""

    feature_bundle_ids: tuple[str, ...]
    stage_values: tuple[str, ...]
    label_policy_id: str | None
    model_protocol_id: str | None


_REQUIRED_STAGES = (
    "05_materialize_feature_views",
    "06_validate_feature_coverage",
    "08_build_folds",
    "09_train_models",
    "10_generate_predictions_and_diagnostics",
)
_STAGE_S04 = "04_build_or_reuse_replay_charts"
_STAGE_S07 = "07_derive_labels"
_STAGE_S14 = "14_build_frontier_and_insights"


def resolve_study_protocol(
    request: RegimeStudyRequest, *, observation_source_artifact_id: str | None
) -> RegimeProtocolEnvelope:
    """The concrete protocol for THIS request over THIS verified observation
    source (the panel grain pins the loaded panel artifact id)."""

    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415
    from .regime_service import resolve_kmeans_protocol  # noqa: PLC0415

    bundle = resolve_bundle(request.input_feature_bundle_key)
    panel = request.is_panel
    if panel and observation_source_artifact_id is None:
        raise ValueError("the panel grain requires the loaded context_bar_panel_artifact_id")
    return resolve_kmeans_protocol(
        input_feature_bundle_ref=bundle.resolved_feature_bundle_id,
        resolved_input_features=request.resolved_input_features,
        observation_granularity=request.observation_granularity,
        observation_stage=request.observation_stage,
        panel_interval_seconds=request.panel_interval_seconds if panel else None,
        panel_source_artifact_id=observation_source_artifact_id if panel else None,
        panel_as_of_policy_id=request.panel_as_of_policy_id if panel else None,
        resolved_cluster_count=request.resolved_cluster_count,
        winsorization_policy=request.winsorization_policy,
    )


def _probe_protocol(request: RegimeStudyRequest) -> RegimeProtocolEnvelope:
    """Readiness probe: the candidate grain's protocol is exact; the panel
    grain's uses a placeholder source id (the real one exists after S05)."""

    return resolve_study_protocol(
        request, observation_source_artifact_id=("0" * 64) if request.is_panel else None
    )


@dataclass(frozen=True, slots=True)
class FrozenAuthority:
    promotion: RegimePromotionDecisionEnvelope
    owner: Any
    assessment: RegimeCapabilityAssessmentEnvelope


def verify_frozen_authority(
    store_root: Path,
    request: RegimeStudyRequest,
    *,
    expected_protocol_id: str | None,
    run_scope: str,
) -> FrozenAuthority:
    """Verified-load the EXACT frozen promotion / owner / assessment refs of a
    model-bearing request and prove they cohere (D1/D4/D9): the decision is
    FEATURE_ELIGIBLE (or beyond), references exactly the required assessment
    and owner artifact, the owner artifact authorizes that decision over that
    assessment, nothing is superseded, and — when known — the protocol is the
    decision's. Never resolves a "latest" decision."""

    from ..search.owner_decisions import (  # noqa: PLC0415
        OwnerDecisionRefusalError,
        assert_owner_decision_authorizes,
        load_owner_decision,
        load_supersession_chain,
        transition_key,
    )
    from .regime_store import (  # noqa: PLC0415
        load_regime_assessment,
        load_regime_promotion,
        load_regime_protocol,
    )

    root = Path(store_root)
    decision_id, owner_id, assessment_id = request.authority_refs
    if decision_id is None or owner_id is None or assessment_id is None:
        raise ValueError("a descriptive request carries no frozen authority to verify")
    try:
        decision = load_regime_promotion(root, decision_id)
    except SearchStoreError as error:
        raise ValueError(
            "frozen regime_promotion_decision_id is not a verified entry of this store"
        ) from error
    payload = decision.payload
    if RegimeStatus(payload.status) not in (
        RegimeStatus.FEATURE_ELIGIBLE,
        RegimeStatus.MODEL_FEATURE,
    ):
        raise ValueError(
            f"frozen promotion decision has status {payload.status.value}; model-bearing "
            "regime studies require FEATURE_ELIGIBLE"
        )
    if payload.capability_assessment_ref != assessment_id:
        raise ValueError(
            "frozen promotion decision references a different capability assessment than "
            "required_capability_assessment_id"
        )
    if payload.owner_ratification_ref != owner_id:
        raise ValueError(
            "frozen promotion decision references a different owner-decision artifact than "
            "owner_decision_artifact_id"
        )
    if expected_protocol_id is not None and (
        payload.resolved_regime_protocol_id != expected_protocol_id
    ):
        raise ValueError(
            "frozen promotion decision was taken over a different resolved regime protocol "
            "than this request resolves to"
        )
    try:
        assessment = load_regime_assessment(root, assessment_id)
    except SearchStoreError as error:
        raise ValueError(
            "required_capability_assessment_id is not a verified entry of this store"
        ) from error
    if assessment.payload.resolved_regime_protocol_id != payload.resolved_regime_protocol_id:
        raise ValueError("the frozen assessment does not belong to the decision's protocol")
    try:
        owner = load_owner_decision(root, owner_id)
    except SearchStoreError as error:
        raise ValueError(
            "owner_decision_artifact_id is not a verified owner-decision artifact of this store"
        ) from error
    try:
        protocol = load_regime_protocol(root, payload.resolved_regime_protocol_id)
    except SearchStoreError as error:
        raise ValueError("the decision's regime protocol is not a verified store entry") from error
    try:
        assert_owner_decision_authorizes(
            owner,
            protocol_envelope=protocol,
            assessment_envelope=assessment,
            transition=transition_key(payload.previous_status, payload.status),
            as_of=payload.decided_at,
            run_scope=run_scope,  # type: ignore[arg-type]
            supersession_chain=load_supersession_chain(root),
        )
    except OwnerDecisionRefusalError as error:
        raise ValueError(f"frozen owner evidence refused: {error}") from error
    return FrozenAuthority(promotion=decision, owner=owner, assessment=assessment)


def regime_study_block_reason(
    request: RegimeStudyRequest,
    facts: RegimeStudyStagePlanFacts,
    *,
    store_root: Path | None = None,
    run_scope: str = "full_authorized_development",
) -> str | None:
    """The capability-readiness reason for a regime study, or None when it
    is launchable (registry / algorithm / bundle / input / leakage / grain /
    stage-plan checks; the frozen authority verified-loaded when a store root
    is given)."""

    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415

    try:
        assert_regime_algorithm_fittable(request.algorithm_key)
    except (RegimeAlgorithmUnavailableError, ValueError) as error:
        return str(error)
    for key in (request.input_feature_bundle_key, request.supervised_bundle_key):
        if key is None:
            continue
        try:
            resolve_bundle(key)
        except (BlockUnavailableError, ValueError) as error:
            return f"regime study bundle {key!r} is unresolvable: {error}"
        if key not in facts.feature_bundle_ids:
            return (
                f"regime study bundle {key!r} must be one of the plan's feature bundles "
                f"{tuple(facts.feature_bundle_ids)}"
            )
    try:
        _probe_protocol(request)
    except (RegimeLeakageError, ValueError, PermissionError) as error:
        return f"regime protocol cannot be resolved: {error}"
    problems = request.stage_plan_problems(
        feature_bundle_ids=tuple(facts.feature_bundle_ids),
        stage_values=tuple(facts.stage_values),
        label_policy_id=facts.label_policy_id,
        model_protocol_id=facts.model_protocol_id,
    )
    if problems:
        return "; ".join(problems)
    if request.requires_supervision and store_root is not None:
        try:
            verify_frozen_authority(
                Path(store_root),
                request,
                expected_protocol_id=(
                    None
                    if request.is_panel
                    else _probe_protocol(request).resolved_regime_protocol_id
                ),
                run_scope=run_scope,
            )
        except ValueError as error:
            return f"frozen regime authority refused: {error}"
    return None


def derive_initial_regime_status(assessment: RegimeCapabilityAssessmentEnvelope) -> RegimeStatus:
    """The deterministic first decision after a fit: DESCRIPTIVE_ONLY, or a
    typed BLOCKED state when no OOS assignment or the coverage gates fail."""

    payload = assessment.payload
    if not payload.oos_assignment_available:
        return RegimeStatus.BLOCKED_NO_OOS_ASSIGNMENT
    if not payload.coverage.coverage_gates_passed:
        return RegimeStatus.BLOCKED_INSUFFICIENT_COVERAGE
    return RegimeStatus.DESCRIPTIVE_ONLY


def regime_evidence_as_of(assignments: pd.DataFrame, *, fallback: str) -> str:
    """The evidence as-of instant = the maximum observation timestamp of the
    run (ISO-8601, UTC); ``fallback`` (e.g. the last authorized day at 17:00
    ET) when the run carries no timestamps (zero valid folds)."""

    if "observation_ts_utc" in assignments.columns and len(assignments):
        stamps = pd.to_datetime(assignments["observation_ts_utc"], utc=True, errors="coerce")
        latest = stamps.max()
        if pd.notna(latest):
            return latest.isoformat().replace("+00:00", "Z")
    return fallback


def build_s10_decisions(
    store_root: Path,
    *,
    protocol: RegimeProtocolEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    request: RegimeStudyRequest,
    decided_at: str,
) -> tuple[RegimePromotionDecisionEnvelope, ...]:
    """S10's deterministic, ratification-free decisions (D4/D6): the initial
    status, then STRATIFICATION_READY iff coverage gates passed AND an OOS
    assignment exists AND descriptive classes were requested. Persisted
    through ``persist_regime_promotion`` (which re-verifies the assessment);
    identical inputs mint identical decision ids."""

    from .regime_store import persist_regime_promotion  # noqa: PLC0415

    root = Path(store_root)
    initial_status = derive_initial_regime_status(assessment)
    first = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            role=RegimeRole.DESCRIPTIVE_ONLY,
            status=initial_status,
            previous_status=RegimeStatus.PLANNED,
            previous_decision_ref=None,
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=None,
            decided_at=decided_at,
        )
    )
    persist_regime_promotion(root, first)
    decisions = [first]
    structural = (
        initial_status is RegimeStatus.DESCRIPTIVE_ONLY
        and assessment.payload.coverage.coverage_gates_passed
        and assessment.payload.oos_assignment_available
    )
    if structural and request.descriptive_classes:
        ready = RegimePromotionDecisionEnvelope.from_payload(
            RegimePromotionDecision(
                resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
                role=RegimeRole.STRATIFICATION_ONLY,
                status=RegimeStatus.STRATIFICATION_READY,
                previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_decision_ref=first.regime_promotion_decision_id,
                capability_assessment_ref=assessment.regime_capability_assessment_id,
                owner_ratification_ref=None,
                decided_at=decided_at,
            )
        )
        persist_regime_promotion(root, ready)
        decisions.append(ready)
    return tuple(decisions)


_ = PANEL_AS_OF_POLICY_ID_V1  # re-exported default for request builders
