"""Regime contracts (ML_REGIME_CONTRACT_PLAN §3; V3 P1-3; Amendment P1-B).

The protocol/fit/capability/promotion split, enforced structurally:

* ``RegimeProtocolPayload`` — the ALGORITHM PROTOCOL. No role, no status,
  no fold or fit state; hashed → ``resolved_regime_protocol_id``. The
  logical ``algorithm_key`` (``kmeans_v1`` …) is never confused with the
  resolved identity (V3 P1-2); the registry's pinned parameters enter the
  identity through ``pinned_parameters_hash`` (CS §0.2 — a registry pin
  edit without a version bump can never keep an old protocol id).
* ``RegimeFitPayload`` — ONE fold's fit under one protocol; role-free. It
  pins the verified source artifacts (never empty) AND the hash of the
  training feature matrix, so two fits over different feature values can
  never share a ``regime_fit_id``.
* ``RegimeCapabilityAssessment`` — coverage/occupancy/stability facts over
  an exact fit set + fold set, with the applied gate values (all
  ``proposed_protocol_default``).
* ``RegimePromotionDecision`` — status changes WITHOUT implying any
  numerical fit changed. The ladder (planned → descriptive →
  stratification → feature-eligible → model-feature) is enforced by the
  contract itself: one forward step at a time, ``previous_decision_ref``
  chaining, ISO-8601 ``decided_at``, a 64-hex owner ratification reference
  (the ``OwnerDecisionEvidenceRef`` content hash) from ``FEATURE_ELIGIBLE``
  onward, and a ROLE ladder under which the execution-side roles
  (decision policy, execution-gate candidate, frozen execution gate) are
  unrepresentable in V1 — they carry the exact S11 blocked reason. The
  PASSING-assessment half of the gate is enforced where the assessment is
  loadable: ``regime_store.persist_regime_promotion`` refuses any decision
  whose referenced assessment does not exist, names another protocol, or
  did not pass its gates.

Panel-grain validation (P1-B): ``CONTEXT_BAR_PANEL`` requires an
owner-registered ``panel_interval_seconds``, a ``panel_source_artifact_id``,
and a completed-bars-only ``panel_as_of_policy_id``; for
``CANDIDATE_STAGE_ROW`` / ``DECISION_ROW`` all three must be None — any
other combination fails validation.

Leakage is unrepresentable at the input seam (acceptance 7B.22-3):
``assert_no_regime_leakage`` refuses every outcome/label/future column AND
— by default, from the feature-block registries — every feature whose
block availability stage lies after the protocol's observation stage;
unregistered feature names are refused because their stage is unprovable.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..features.feature_blocks import AvailabilityStage
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)

__all__ = [
    "RegimeStatus",
    "RegimeRole",
    "ObservationGranularity",
    "INITIALIZATION_POLICIES",
    "RegimeProtocolPayload",
    "RegimeProtocolEnvelope",
    "RegimeFitPayload",
    "RegimeFitEnvelope",
    "RegimeFitArtifact",
    "SIDECAR_REFERENCE_PATTERN",
    "RegimeAssignmentColumns",
    "REGIME_ASSIGNMENT_MISSING_REASONS",
    "RegimeCoverageReport",
    "RegimeStabilityReport",
    "RegimeCapabilityAssessment",
    "RegimeCapabilityAssessmentEnvelope",
    "RegimePromotionDecision",
    "RegimePromotionDecisionEnvelope",
    "PROMOTION_SEQUENCE",
    "ROLE_MINIMUM_STATUS",
    "V1_UNREPRESENTABLE_ROLES",
    "assert_lawful_promotion",
    "assert_lawful_role",
    "PROHIBITED_REGIME_INPUT_COLUMNS",
    "PROHIBITED_REGIME_INPUT_TOKENS",
    "RegimeLeakageError",
    "registered_feature_stages",
    "assert_no_regime_leakage",
    "REGIME_PROPOSED_DEFAULTS",
    "sample_adequacy_minimum",
]


class RegimeStatus(StrEnum):
    PLANNED = "planned"
    DESCRIPTIVE_ONLY = "descriptive_only"
    STRATIFICATION_READY = "stratification_ready"
    FEATURE_ELIGIBLE = "feature_eligible"
    MODEL_FEATURE = "model_feature"
    EXPERIMENTAL = "experimental"
    BLOCKED_NO_OOS_ASSIGNMENT = "blocked_no_oos_assignment"
    BLOCKED_INSUFFICIENT_COVERAGE = "blocked_insufficient_coverage"
    SUPERSEDED = "superseded"


class RegimeRole(StrEnum):
    DESCRIPTIVE_ONLY = "descriptive_only"
    STRATIFICATION_ONLY = "stratification_only"
    FEATURE_GENERATOR = "feature_generator"
    PREDICTIVE_MODEL = "predictive_model"
    DECISION_POLICY = "decision_policy"
    EXECUTION_GATE_CANDIDATE = "execution_gate_candidate"
    FROZEN_EXECUTION_GATE = "frozen_execution_gate"
    MONITORING_ONLY = "monitoring_only"


class ObservationGranularity(StrEnum):
    CANDIDATE_STAGE_ROW = "candidate_stage_row"
    DECISION_ROW = "decision_row"
    CONTEXT_BAR_PANEL = "context_bar_panel"


#: The registered initialization policies — one per registry algorithm
#: (ML plan §4). The protocol schema accepts exactly this set, so every
#: planned algorithm's protocol is EXPRESSIBLE in V1 (its activation is an
#: implementation event, not a schema change); executability is decided by
#: the registry (``regime_algorithms.assert_protocol_executable``).
INITIALIZATION_POLICIES: tuple[str, ...] = (
    "k-means++_n_init_10_v1",
    "minibatch_k-means++_n_init_10_v1",
    "gmm_kmeans_init_n_init_5_v1",
    "spectral_kmeans_label_assignment_v1",
    "nystroem_rbf_then_k-means++_n_init_10_v1",
    "surrogate_logistic_v1",
)


class RegimeProtocolPayload(FrozenContract):
    """The algorithm protocol — no role/status, no fold/fit state (P1-3)."""

    algorithm_key: str
    algorithm_version: str
    pinned_parameters_hash: str = Field(pattern=SHA256_PATTERN)
    input_feature_bundle_ref: str = Field(pattern=SHA256_PATTERN)
    resolved_input_features: tuple[str, ...] = Field(min_length=1)
    observation_granularity: ObservationGranularity
    panel_interval_seconds: int | None
    panel_source_artifact_id: str | None
    panel_as_of_policy_id: str | None
    observation_stage: AvailabilityStage
    missingness_policy: Literal["median_impute_with_indicator_v1"]
    winsorization_policy: Literal["none", "clip_p01_p99_train_fitted_v1"]
    scaler_policy: Literal["standard_scaler_v1"]
    dimensionality_reduction_policy: Literal["none", "pca_fixed_components_v1"]
    kernel_or_affinity_policy: str | None
    cluster_count_policy: Literal["fixed_k", "inner_train_only_selection"]
    resolved_cluster_count: int = Field(ge=2)
    random_seed: Literal[7] = 7
    initialization_policy: str
    out_of_sample_assignment_policy: Literal[
        "centroid_predict_v1",
        "gmm_posterior_v1",
        "nystrom_transform_kmeans_predict_v1",
        "none_training_only",
        "surrogate_logistic_v1",
    ]
    cluster_label_alignment_policy: Literal["centroid_min_distance_hungarian_v1"]
    fit_scope: Literal["per_training_fold"]
    software_versions: ImmutableMap[str, str]
    formula_version: str

    @model_validator(mode="after")
    def _registered_policies(self):
        if self.initialization_policy not in INITIALIZATION_POLICIES:
            raise ValueError(
                f"initialization_policy {self.initialization_policy!r} is not "
                f"registered; registered: {INITIALIZATION_POLICIES}"
            )
        if len(set(self.resolved_input_features)) != len(self.resolved_input_features):
            raise ValueError("resolved_input_features must not repeat a feature")
        return self

    @model_validator(mode="after")
    def _panel_grain_fields_agree(self):
        """Amendment P1-B: the panel grain is in the ACTUAL schema."""

        panel_fields = (
            self.panel_interval_seconds,
            self.panel_source_artifact_id,
            self.panel_as_of_policy_id,
        )
        if self.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL:
            if any(value is None for value in panel_fields):
                raise ValueError(
                    "CONTEXT_BAR_PANEL requires panel_interval_seconds, "
                    "panel_source_artifact_id, AND panel_as_of_policy_id"
                )
            if self.panel_interval_seconds is not None and self.panel_interval_seconds < 60:
                raise ValueError(
                    "panel_interval_seconds must be at least 60 — regime "
                    "models are never fitted at tick or MBP-event scale"
                )
        elif any(value is not None for value in panel_fields):
            raise ValueError(
                f"{self.observation_granularity.value} carries no panel fields; "
                "all three must be None"
            )
        return self


class RegimeProtocolEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "resolved_regime_protocol_id"

    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeProtocolPayload


_SHA256_RE = re.compile(SHA256_PATTERN)


class RegimeFitPayload(FrozenContract):
    """ONE fold's fit under one protocol — role-free, status-free.

    ``source_artifact_ids`` pins the verified upstream evidence (the
    observation view / panel artifact ids — never empty, CS §0.1) and
    ``training_feature_matrix_hash`` binds the fold's training INPUT values:
    a fit over different feature values is a different fit identity.
    """

    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    fold_index: int = Field(ge=0)
    fit_start: str | None
    fit_end: str | None
    training_row_ids_hash: str = Field(pattern=SHA256_PATTERN)
    training_feature_matrix_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _verified_source_ids(self):
        bad = [ref for ref in self.source_artifact_ids if not _SHA256_RE.fullmatch(ref)]
        if bad:
            raise ValueError(
                "source_artifact_ids must be verified 64-hex artifact identities; "
                f"refused: {bad}"
            )
        if len(set(self.source_artifact_ids)) != len(self.source_artifact_ids):
            raise ValueError("source_artifact_ids must not repeat an identity")
        return self


class RegimeFitEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_fit_id"

    regime_fit_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeFitPayload


#: Manifest-relative sidecar references: the store's own sidecar-name
#: whitelist (a bare file name inside the artifact directory — no
#: separators, no traversal, no drive/ADS colons, no leading dot), minus
#: the two store-owned files.
SIDECAR_REFERENCE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_STORE_OWNED_FILES = ("envelope.json", "manifest.json")


class RegimeFitArtifact(FrozenContract):
    """The materialized numerical facts of one fit (post-materialization —
    never inside a pre-run identity). ``preprocessing_pipeline_ref`` is a
    MANIFEST-RELATIVE sidecar name (revision P1-4), never a path."""

    regime_fit_id: str = Field(pattern=SHA256_PATTERN)
    fold_index: int = Field(ge=0)
    artifact_schema_version: int = Field(default=1, ge=1)
    fitted_parameter_payload_hash: str = Field(pattern=SHA256_PATTERN)
    preprocessing_pipeline_ref: str
    fold_local_cluster_ids: tuple[int, ...]
    centroid_or_component_descriptors: ImmutableMap[int, tuple[tuple[str, float], ...]]
    training_row_ids_hash: str = Field(pattern=SHA256_PATTERN)
    training_feature_matrix_hash: str = Field(pattern=SHA256_PATTERN)
    training_row_count: int = Field(ge=0)
    inertia_or_loglik: float
    software_versions: ImmutableMap[str, str]

    @model_validator(mode="after")
    def _relative_ref(self):
        ref = self.preprocessing_pipeline_ref
        if not SIDECAR_REFERENCE_PATTERN.fullmatch(ref) or ref in _STORE_OWNED_FILES:
            raise ValueError(
                "preprocessing_pipeline_ref must be a manifest-relative sidecar "
                "name — no paths, separators, traversal, drive/ADS colons, or "
                "store-owned file names"
            )
        return self


#: The exact assignment-frame columns (one row per (row, fold)); persisted
#: as a frame beside each fit. Typed reasons per 7B.22-10 — rows preserved.
#: ``distances`` is the distance to EVERY fold-local centroid (the §4
#: kmeans_v1 output); ``assignment_entropy``/``log_density``/``outlier_score``
#: are carried as NaN by kmeans_v1 (schema stable for the post-V1
#: algorithms; GMM adds ``probabilities`` with its release).
RegimeAssignmentColumns: tuple[str, ...] = (
    "resolved_regime_protocol_id",
    "regime_fit_id",
    "row_id",
    "fold_index",
    "partition",
    "observation_ts_utc",
    "fold_local_cluster_id",
    "canonical_reporting_cluster_id",
    "distances",
    "assigned_distance",
    "assignment_margin",
    "assignment_entropy",
    "log_density",
    "outlier_score",
    "valid",
    "missing_reason",
)

REGIME_ASSIGNMENT_MISSING_REASONS: tuple[str, ...] = (
    "source_feature_missing",
    "fold_invalid",
    "training_only_algorithm",
    "below_confidence_floor",
    "coverage_gap",
)


class RegimeCoverageReport(FrozenContract):
    rows_total: int = Field(ge=0)
    rows_assigned: int = Field(ge=0)
    rows_typed_null: ImmutableMap[str, int]
    per_fold_coverage: ImmutableMap[int, float]
    oos_assignment_coverage: float = Field(ge=0.0, le=1.0)
    per_cluster_occupancy: ImmutableMap[int, float]
    minimum_training_observations_gate: int = Field(ge=0)
    minimum_training_observations_observed: int = Field(ge=0)
    coverage_gates_passed: bool
    gate_failures: tuple[str, ...]


class RegimeStabilityReport(FrozenContract):
    bootstrap_refit_count: int = Field(ge=0)
    bootstrap_seed_policy: Literal["rng7_resample_refit_seed_1000_plus_i_v1"] = (
        "rng7_resample_refit_seed_1000_plus_i_v1"
    )
    bootstrap_aligned_ami_mean: float | None
    bootstrap_aligned_ami_low: float | None
    per_cluster_agreement: ImmutableMap[int, float]
    #: Temporal facts are computed over OUT-OF-SAMPLE test rows ordered by
    #: the observation's as-of timestamp, WITHIN each fold (no cross-fold
    #: concatenation; no row-id ordering) — a true point-in-time timeline.
    temporal_order_policy: Literal["oos_test_rows_by_observation_ts_within_fold_v1"] = (
        "oos_test_rows_by_observation_ts_within_fold_v1"
    )
    temporal_transition_count: int = Field(default=0, ge=0)
    temporal_persistence: float | None
    transition_matrix: tuple[tuple[float, ...], ...]
    #: Cross-fold centroid comparisons happen in the scaled INPUT-feature
    #: coordinates only (missing-indicator columns are fold-dependent and
    #: never enter an alignment).
    alignment_space: Literal["scaled_input_features_v1"] = "scaled_input_features_v1"
    fold_to_fold_recurrence: float | None
    separation_min_centroid_distance: float | None
    silhouette_descriptive: float | None
    semantic_descriptors: ImmutableMap[int, tuple[tuple[str, float], ...]]
    stability_gates_passed: bool
    gate_failures: tuple[str, ...]


class RegimeCapabilityAssessment(FrozenContract):
    """Coverage/occupancy/stability over EXACT fit + fold identities."""

    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    regime_fit_ids: tuple[str, ...]
    fold_set_id: str = Field(pattern=SHA256_PATTERN)
    coverage: RegimeCoverageReport
    occupancy: ImmutableMap[int, float]
    stability: RegimeStabilityReport
    oos_assignment_available: bool
    minimum_cluster_occupancy_gate: float
    minimum_cluster_rows_gate: int
    minimum_assignment_confidence_gate: float | None
    gates_passed: bool
    gate_failures: tuple[str, ...]


class RegimeCapabilityAssessmentEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_capability_assessment_id"

    regime_capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeCapabilityAssessment


#: The lawful promotion ladder (forward one step at a time). The
#: NON-PROMOTING states — DESCRIPTIVE_ONLY, EXPERIMENTAL, BLOCKED_* — are
#: always reachable (demotion/blocking is never gated; none of them earns a
#: role above descriptive/monitoring); SUPERSEDED is terminal — reachable
#: from anywhere, never left.
PROMOTION_SEQUENCE: tuple[RegimeStatus, ...] = (
    RegimeStatus.PLANNED,
    RegimeStatus.DESCRIPTIVE_ONLY,
    RegimeStatus.STRATIFICATION_READY,
    RegimeStatus.FEATURE_ELIGIBLE,
    RegimeStatus.MODEL_FEATURE,
)

_ALWAYS_REACHABLE = {
    RegimeStatus.DESCRIPTIVE_ONLY,
    RegimeStatus.EXPERIMENTAL,
    RegimeStatus.BLOCKED_NO_OOS_ASSIGNMENT,
    RegimeStatus.BLOCKED_INSUFFICIENT_COVERAGE,
    RegimeStatus.SUPERSEDED,
}

#: The ROLE ladder (ML plan §1): a role is lawful only at or above the
#: status that earns it. ``None`` = lawful at every status.
ROLE_MINIMUM_STATUS: MappingProxyType[RegimeRole, RegimeStatus | None] = MappingProxyType(
    {
        RegimeRole.DESCRIPTIVE_ONLY: None,
        RegimeRole.MONITORING_ONLY: None,
        RegimeRole.STRATIFICATION_ONLY: RegimeStatus.STRATIFICATION_READY,
        RegimeRole.FEATURE_GENERATOR: RegimeStatus.FEATURE_ELIGIBLE,
        RegimeRole.PREDICTIVE_MODEL: RegimeStatus.MODEL_FEATURE,
    }
)

#: Execution-side roles are UNREPRESENTABLE in V1: a regime artifact can
#: never become an FSM guard, execution gate, decision policy, or prop-risk
#: control without a separately frozen protocol plus the downstream
#: sequential gated replay (S11 — blocked with the exact registered reason).
V1_UNREPRESENTABLE_ROLES: tuple[RegimeRole, ...] = (
    RegimeRole.DECISION_POLICY,
    RegimeRole.EXECUTION_GATE_CANDIDATE,
    RegimeRole.FROZEN_EXECUTION_GATE,
)


def assert_lawful_promotion(
    current: RegimeStatus,
    proposed: RegimeStatus,
    *,
    owner_ratification_ref: str | None,
    gates_passed: bool,
) -> None:
    """The status ladder: one forward step at a time; FEATURE_ELIGIBLE and
    beyond require BOTH passing capability gates AND the owner's
    ratification reference (P1-3 — nothing is promoted merely because
    fitting succeeded). SUPERSEDED is terminal; PLANNED is never a target."""

    current = RegimeStatus(current)
    proposed = RegimeStatus(proposed)
    if current is RegimeStatus.SUPERSEDED and proposed is not RegimeStatus.SUPERSEDED:
        raise ValueError("a superseded protocol is terminal; it cannot be re-promoted")
    if proposed is RegimeStatus.PLANNED:
        raise ValueError("planned is the pre-decision state, never a decision target")
    if proposed in _ALWAYS_REACHABLE:
        return
    if proposed not in PROMOTION_SEQUENCE:
        raise ValueError(f"status {proposed.value} is not a promotion target")
    # blocked/experimental states resume the ladder from the bottom
    current_index = (
        PROMOTION_SEQUENCE.index(current) if current in PROMOTION_SEQUENCE else 0
    )
    proposed_index = PROMOTION_SEQUENCE.index(proposed)
    if proposed_index > current_index + 1:
        raise ValueError(
            f"promotion {current.value} → {proposed.value} skips the sequence "
            f"({' → '.join(status.value for status in PROMOTION_SEQUENCE)})"
        )
    if proposed_index >= PROMOTION_SEQUENCE.index(RegimeStatus.FEATURE_ELIGIBLE):
        if not gates_passed:
            raise ValueError(
                f"promotion to {proposed.value} requires a PASSING capability "
                "assessment (sample adequacy, occupancy, coverage, stability)"
            )
        if not owner_ratification_ref:
            raise ValueError(
                f"promotion to {proposed.value} requires the owner's "
                "ratification reference — proposed_protocol_default values "
                "carry no research weight (P1-3)"
            )


def assert_lawful_role(role: RegimeRole, status: RegimeStatus) -> None:
    """The role ladder: execution-side roles are unrepresentable in V1
    (exact S11 reason); every other role requires the status that earns it."""

    from .decision_policies import S11_BLOCKED_REASON  # noqa: PLC0415

    role = RegimeRole(role)
    status = RegimeStatus(status)
    if role in V1_UNREPRESENTABLE_ROLES:
        raise ValueError(
            f"role {role.value} is unrepresentable in V1 — {S11_BLOCKED_REASON}"
        )
    minimum = ROLE_MINIMUM_STATUS[role]
    if minimum is None:
        return
    if status not in PROMOTION_SEQUENCE or (
        PROMOTION_SEQUENCE.index(status) < PROMOTION_SEQUENCE.index(minimum)
    ):
        raise ValueError(
            f"role {role.value} requires status {minimum.value} or later; "
            f"decision status is {status.value}"
        )


def _parse_decided_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("decided_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("decided_at must carry an explicit UTC offset")
    return parsed


class RegimePromotionDecision(FrozenContract):
    """A status change that can never rewrite numerical identity (V3 P1-3).

    The ladder is enforced HERE (one step; ratified from FEATURE_ELIGIBLE;
    role ladder; terminal SUPERSEDED; ISO-8601 decision time; chained
    ``previous_decision_ref``). The passing-assessment half is enforced at
    persistence, where the referenced assessment is loaded and verified.
    """

    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    role: RegimeRole
    status: RegimeStatus
    previous_status: RegimeStatus
    previous_decision_ref: str | None = Field(default=None, pattern=SHA256_PATTERN)
    capability_assessment_ref: str = Field(pattern=SHA256_PATTERN)
    #: The ``OwnerDecisionEvidenceRef.content_hash`` of the owner's ratified
    #: regime decision (grain + cluster policy) — required from
    #: FEATURE_ELIGIBLE onward (P1-3).
    owner_ratification_ref: str | None = Field(default=None, pattern=SHA256_PATTERN)
    decided_at: str

    @model_validator(mode="after")
    def _lawful_decision(self):
        _parse_decided_at(self.decided_at)
        if (self.previous_decision_ref is None) != (
            self.previous_status is RegimeStatus.PLANNED
        ):
            raise ValueError(
                "previous_decision_ref is required exactly when previous_status "
                "is not planned (the ladder is a chain of persisted decisions)"
            )
        assert_lawful_promotion(
            self.previous_status,
            self.status,
            owner_ratification_ref=self.owner_ratification_ref,
            gates_passed=True,  # the assessment half is verified at persistence
        )
        assert_lawful_role(self.role, self.status)
        return self


class RegimePromotionDecisionEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_promotion_decision_id"

    regime_promotion_decision_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimePromotionDecision


# ── input leakage (brief §7B.6; acceptance 7B.22-3) ──────────────────────────

PROHIBITED_REGIME_INPUT_COLUMNS: tuple[str, ...] = (
    "binary_target",
    "label",
    "gross_r",
    "net_r",
    "mfe_r",
    "mae_r",
    "realized_ticks",
    "realized_r",
    "resolution",
    "resolution_ts_utc",
    "bars_after_entry_to_resolution",
)

PROHIBITED_REGIME_INPUT_TOKENS: tuple[str, ...] = ("payout", "breach", "passed")


class RegimeLeakageError(PermissionError):
    """An outcome/label/future column reached the regime input surface."""


def registered_feature_stages() -> dict[str, AvailabilityStage]:
    """feature name → block availability stage, from the block registries
    (every resolvable block's feature names carry its definition's stage)."""

    from ..features.feature_blocks import (  # noqa: PLC0415
        FEATURE_BLOCK_REGISTRY,
        FEATURE_BLOCK_RESOLUTION_REGISTRY,
    )

    stages: dict[str, AvailabilityStage] = {}
    for block_key, envelope in FEATURE_BLOCK_RESOLUTION_REGISTRY.items():
        definition = FEATURE_BLOCK_REGISTRY.get(block_key)
        if definition is None:
            continue
        for name in envelope.payload.feature_names:
            stages[name] = definition.availability_stage
    return stages


def assert_no_regime_leakage(
    resolved_input_features: tuple[str, ...],
    *,
    observation_stage: AvailabilityStage,
    feature_stage_for: dict[str, AvailabilityStage] | None = None,
) -> None:
    """Refuse prohibited inputs and any feature whose block availability
    stage lies AFTER the protocol's observation stage (point-in-time by
    construction). The stage check is ALWAYS applied: an explicit map is
    consulted first, then the block registries; a feature with no
    registered stage is refused because its point-in-time availability is
    unprovable."""

    offenders = [
        name
        for name in resolved_input_features
        if name in PROHIBITED_REGIME_INPUT_COLUMNS
        or any(token in name.lower() for token in PROHIBITED_REGIME_INPUT_TOKENS)
    ]
    if offenders:
        raise RegimeLeakageError(
            f"prohibited regime inputs (outcome/label/future evidence): {sorted(offenders)}"
        )
    stages = tuple(AvailabilityStage)
    limit = stages.index(AvailabilityStage(observation_stage))
    registry = None
    late: list[str] = []
    unknown: list[str] = []
    for name in resolved_input_features:
        stage = (feature_stage_for or {}).get(name)
        if stage is None:
            if registry is None:
                registry = registered_feature_stages()
            stage = registry.get(name)
        if stage is None:
            unknown.append(name)
        elif stages.index(AvailabilityStage(stage)) > limit:
            late.append(name)
    if unknown:
        raise RegimeLeakageError(
            "features with no registered availability stage cannot be proven "
            f"point-in-time: {sorted(unknown)}"
        )
    if late:
        raise RegimeLeakageError(
            f"features available only after {observation_stage.value}: {sorted(late)}"
        )


# ── proposal stamps (revision P1-3) ──────────────────────────────────────────


def _freeze(
    entries: dict[str, dict[str, Any]],
) -> MappingProxyType[str, MappingProxyType[str, Any]]:
    return MappingProxyType(
        {key: MappingProxyType(dict(value)) for key, value in entries.items()}
    )


#: Every scientific default in this lane, stamped. Nothing here carries any
#: research weight before the owner's decision evidence exists. Frozen —
#: the SINGLE source the service gates, the diagnostics floors, and the UI
#: stamp table all read (review C7/C9).
REGIME_PROPOSED_DEFAULTS: MappingProxyType[str, MappingProxyType[str, Any]] = _freeze({
    "algorithm_baseline": {
        "value": "kmeans_v1",
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "25",
    },
    "fixed_cluster_count": {
        "value": 3,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "29",
    },
    "minimum_cluster_occupancy_fraction": {
        "value": 0.05,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_cluster_rows_per_fold": {
        "value": 25,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "bootstrap_aligned_ami_advisory_floor": {
        "value": 0.5,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_training_observations_candidate_stage": {
        "value": 150,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_training_observations_decision_row": {
        "value": 150,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_training_observations_panel": {
        "value": 300,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "observation_grain_baseline": {
        "value": "candidate_stage_row@entry_decision",
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "28",
    },
})

_SAMPLE_ADEQUACY_KEYS: MappingProxyType[ObservationGranularity, str] = MappingProxyType(
    {
        ObservationGranularity.CANDIDATE_STAGE_ROW: (
            "minimum_training_observations_candidate_stage"
        ),
        ObservationGranularity.DECISION_ROW: "minimum_training_observations_decision_row",
        ObservationGranularity.CONTEXT_BAR_PANEL: "minimum_training_observations_panel",
    }
)


def sample_adequacy_minimum(grain: ObservationGranularity) -> int:
    """The stamped minimum training observations per fold for a grain —
    read from ``REGIME_PROPOSED_DEFAULTS`` (never restated elsewhere)."""

    key = _SAMPLE_ADEQUACY_KEYS[ObservationGranularity(grain)]
    return int(REGIME_PROPOSED_DEFAULTS[key]["value"])


register_identity_pair(
    name="RegimeProtocol",
    envelope_cls=RegimeProtocolEnvelope,
    payload_cls=RegimeProtocolPayload,
    id_field="resolved_regime_protocol_id",
    example_factory=lambda: RegimeProtocolPayload(
        algorithm_key="kmeans_v1",
        algorithm_version="1",
        pinned_parameters_hash="f" * 64,
        input_feature_bundle_ref="a" * 64,
        resolved_input_features=("distance_to_htf_ticks",),
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        panel_interval_seconds=None,
        panel_source_artifact_id=None,
        panel_as_of_policy_id=None,
        observation_stage=AvailabilityStage.ENTRY_DECISION,
        missingness_policy="median_impute_with_indicator_v1",
        winsorization_policy="none",
        scaler_policy="standard_scaler_v1",
        dimensionality_reduction_policy="none",
        kernel_or_affinity_policy=None,
        cluster_count_policy="fixed_k",
        resolved_cluster_count=3,
        initialization_policy="k-means++_n_init_10_v1",
        out_of_sample_assignment_policy="centroid_predict_v1",
        cluster_label_alignment_policy="centroid_min_distance_hungarian_v1",
        fit_scope="per_training_fold",
        software_versions={"scikit-learn": "0"},
        formula_version="regime_lane_v1",
    ),
)

register_identity_pair(
    name="RegimeFit",
    envelope_cls=RegimeFitEnvelope,
    payload_cls=RegimeFitPayload,
    id_field="regime_fit_id",
    example_factory=lambda: RegimeFitPayload(
        resolved_regime_protocol_id="a" * 64,
        source_artifact_ids=("b" * 64,),
        fold_index=0,
        fit_start="2026-01-05",
        fit_end="2026-02-27",
        training_row_ids_hash="c" * 64,
        training_feature_matrix_hash="d" * 64,
    ),
)

register_identity_pair(
    name="RegimeCapabilityAssessment",
    envelope_cls=RegimeCapabilityAssessmentEnvelope,
    payload_cls=RegimeCapabilityAssessment,
    id_field="regime_capability_assessment_id",
    example_factory=lambda: RegimeCapabilityAssessment(
        resolved_regime_protocol_id="a" * 64,
        regime_fit_ids=("b" * 64,),
        fold_set_id="c" * 64,
        coverage=RegimeCoverageReport(
            rows_total=0,
            rows_assigned=0,
            rows_typed_null={},
            per_fold_coverage={},
            oos_assignment_coverage=0.0,
            per_cluster_occupancy={},
            minimum_training_observations_gate=150,
            minimum_training_observations_observed=0,
            coverage_gates_passed=False,
            gate_failures=("sample_adequacy",),
        ),
        occupancy={},
        stability=RegimeStabilityReport(
            bootstrap_refit_count=0,
            bootstrap_aligned_ami_mean=None,
            bootstrap_aligned_ami_low=None,
            per_cluster_agreement={},
            temporal_persistence=None,
            transition_matrix=(),
            fold_to_fold_recurrence=None,
            separation_min_centroid_distance=None,
            silhouette_descriptive=None,
            semantic_descriptors={},
            stability_gates_passed=False,
            gate_failures=("no_valid_folds",),
        ),
        oos_assignment_available=False,
        minimum_cluster_occupancy_gate=0.05,
        minimum_cluster_rows_gate=25,
        minimum_assignment_confidence_gate=None,
        gates_passed=False,
        gate_failures=("sample_adequacy",),
    ),
)

register_identity_pair(
    name="RegimePromotionDecision",
    envelope_cls=RegimePromotionDecisionEnvelope,
    payload_cls=RegimePromotionDecision,
    id_field="regime_promotion_decision_id",
    example_factory=lambda: RegimePromotionDecision(
        resolved_regime_protocol_id="a" * 64,
        role=RegimeRole.DESCRIPTIVE_ONLY,
        status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_status=RegimeStatus.PLANNED,
        previous_decision_ref=None,
        capability_assessment_ref="b" * 64,
        owner_ratification_ref=None,
        decided_at="2026-08-26T00:00:00Z",
    ),
)
