"""Locked contracts for leakage-safe IFVG context experiments.

These types intentionally do not inherit from, or widen, the legacy
``IfvgExperimentConfig`` contract.  The immutable v2/v3 artifact pair is the
only supported input to this lane.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "ProfileCapabilityStatus",
    "ArtifactPreparationStatus",
    "ContextFeatureTier",
    "ProfileCapability",
    "PROFILE_CAPABILITY_REGISTRY",
    "profile_capability",
    "ArtifactReference",
    "PairedIfvgArtifactReference",
    "IfvgContextExperimentDatasetConfig",
    "CandidateStageContextLink",
    "IfvgContextLabelConfig",
    "IfvgContextExperimentConfig",
    "IfvgContextExperimentResult",
    "IfvgContextFoldDefinition",
    "IfvgContextFeatureIdentity",
    "IfvgContextRunReconciliation",
    "IFVG_CONTEXT_MODEL_PROTOCOL_ID",
    "IFVG_CONTEXT_MODEL_PARAMETERS",
    "IFVG_CONTEXT_CALIBRATION_POLICY_ID",
    "THRESHOLD_REPORT_GRID",
    "canonical_contract_sha256",
    "context_run_identity",
]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def canonical_contract_sha256(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_run_identity(
    *,
    config_hash: str,
    view_id: str,
    label_derivation_id: str,
    folds: list[dict[str, Any]],
    model_protocol_hash: str | None,
    predictions: list[dict[str, Any]],
    status: str,
) -> str:
    """Hash the deterministic run protocol and exact OOS prediction stream."""

    normalized_predictions = []
    for row in predictions:
        normalized_predictions.append(
            {
                "oos_row_id": str(row["oos_row_id"]),
                "candidate_id": str(row["candidate_id"]),
                "setup_id": str(row["setup_id"]),
                "trading_day": str(row["trading_day"]),
                "fold_index": int(row["fold_index"]),
                "training_prevalence": float(row["training_prevalence"]),
                "target": int(row["target"]),
                "probability": float(row["probability"]),
                "gross_r": float(row["gross_r"]),
                "net_r": float(row["net_r"]),
            }
        )
    return canonical_contract_sha256(
        {
            "config_hash": config_hash,
            "view_id": view_id,
            "label_derivation_id": label_derivation_id,
            "folds": folds,
            "model_protocol_hash": model_protocol_hash,
            "oos_predictions": normalized_predictions,
            "status": status,
        }
    )


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProfileCapabilityStatus(StrEnum):
    RUNNABLE = "runnable"
    BLOCKED = "blocked"
    ANALYSIS_ONLY = "analysis_only"
    LEGACY_READ_ONLY = "legacy_read_only"


class ArtifactPreparationStatus(StrEnum):
    NOT_PREPARED = "not_prepared"
    PREPARING = "preparing"
    FAILED = "failed"
    CONTEXT_READY = "context_ready"
    SUPERSEDED = "superseded"
    LEGACY_ONLY = "legacy_only"
    BLOCKED = "blocked"


class ContextFeatureTier(StrEnum):
    M0 = "M0"
    M1_PRIMARY = "M1_PRIMARY"
    M1_PLUS_240_EXPERIMENTAL = "M1_PLUS_240_EXPERIMENTAL"
    M2 = "M2"
    M3 = "M3"


class ProfileCapability(_FrozenModel):
    profile_name: str
    status: ProfileCapabilityStatus
    reason: str | None = None


PROFILE_CAPABILITY_REGISTRY = MappingProxyType(
    {
        "ifvg_v2_doc_default_fresh_static_1r": ProfileCapability(
            profile_name="ifvg_v2_doc_default_fresh_static_1r",
            status=ProfileCapabilityStatus.RUNNABLE,
        ),
        "ifvg_v2_ict_clean_fresh_static_1r": ProfileCapability(
            profile_name="ifvg_v2_ict_clean_fresh_static_1r",
            status=ProfileCapabilityStatus.BLOCKED,
            reason="same_leg_locality_sweep_semantics_unresolved",
        ),
        "ifvg_v2_ict_clean_pure_retest_static_1r": ProfileCapability(
            profile_name="ifvg_v2_ict_clean_pure_retest_static_1r",
            status=ProfileCapabilityStatus.BLOCKED,
            reason="owner_trigger_not_selected",
        ),
        "ifvg_v2_weak_counter_displacement_research": ProfileCapability(
            profile_name="ifvg_v2_weak_counter_displacement_research",
            status=ProfileCapabilityStatus.ANALYSIS_ONLY,
            reason="analysis_only_profile",
        ),
        "ifvg_v1_legacy_candidate_stream": ProfileCapability(
            profile_name="ifvg_v1_legacy_candidate_stream",
            status=ProfileCapabilityStatus.LEGACY_READ_ONLY,
            reason="legacy_contract",
        ),
    }
)


def profile_capability(
    profile_name: str,
    *,
    canonical_short_enabled: bool = False,
) -> ProfileCapability:
    if canonical_short_enabled:
        return ProfileCapability(
            profile_name=profile_name,
            status=ProfileCapabilityStatus.BLOCKED,
            reason="canonical_short_profile_not_ratified",
        )
    try:
        return PROFILE_CAPABILITY_REGISTRY[profile_name]
    except KeyError as error:
        raise ValueError(f"unregistered IFVG profile {profile_name!r}") from error


class ArtifactReference(_FrozenModel):
    artifact_id: str = Field(pattern=_SHA256_PATTERN)
    manifest_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_kind: Literal["v2", "v3"]
    dataset_schema_version: int
    preparation_status: ArtifactPreparationStatus = ArtifactPreparationStatus.CONTEXT_READY
    profile_hash: str | None = None
    feature_formula_version: str | None = None


class PairedIfvgArtifactReference(_FrozenModel):
    v2: ArtifactReference
    v3: ArtifactReference

    @model_validator(mode="after")
    def _kinds_are_exact(self) -> PairedIfvgArtifactReference:
        if self.v2.artifact_kind != "v2" or self.v3.artifact_kind != "v3":
            raise ValueError("paired artifact requires one v2 and one v3 reference")
        return self


class IfvgContextExperimentDatasetConfig(_FrozenModel):
    artifact_pair: PairedIfvgArtifactReference
    profile_name: str = "ifvg_v2_doc_default_fresh_static_1r"
    candidate_stage: str = "entry_candidate"
    include_warmup_candidates: Literal[False] = False
    cutoff_ts_utc: datetime = datetime(2026, 6, 10, 21, tzinfo=UTC)

    @field_validator("cutoff_ts_utc")
    @classmethod
    def _cutoff_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("cutoff_ts_utc must be timezone-aware UTC")
        return value


class CandidateStageContextLink(_FrozenModel):
    candidate_id: str
    setup_id: str
    stage: str
    geometry_evidence_id: str
    geometry_evidence_cursor: str
    context_capture_id: str
    context_state_id: str
    context_as_of_ts: datetime
    feature_as_of_ts: datetime

    @model_validator(mode="after")
    def _causal(self) -> CandidateStageContextLink:
        for field_name in ("context_as_of_ts", "feature_as_of_ts"):
            value = getattr(self, field_name)
            if value.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware")
        if self.context_as_of_ts > self.feature_as_of_ts:
            raise ValueError("context.as_of_ts exceeds feature_as_of_ts")
        for field_name in (
            "candidate_id",
            "setup_id",
            "stage",
            "geometry_evidence_id",
            "geometry_evidence_cursor",
            "context_capture_id",
            "context_state_id",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        return self


class IfvgContextLabelConfig(_FrozenModel):
    label_family: Literal["fixed_r", "fixed_sl_tp"] = "fixed_r"
    reward_r: float = 1.0
    fixed_stop_ticks: int | None = None
    fixed_target_ticks: int | None = None
    horizon_bars: int | None = None
    cost_per_trade_r: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def _valid_barriers(self) -> IfvgContextLabelConfig:
        if self.label_family == "fixed_r":
            if self.reward_r not in (1.0, 1.5, 2.0):
                raise ValueError("fixed_r reward_r must be 1.0, 1.5, or 2.0")
            if self.fixed_stop_ticks is not None or self.fixed_target_ticks is not None:
                raise ValueError("fixed_r cannot include fixed SL/TP overrides")
        else:
            if not self.fixed_stop_ticks or not self.fixed_target_ticks:
                raise ValueError("fixed_sl_tp requires positive stop and target ticks")
        if self.horizon_bars is not None and self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        return self

    @property
    def identity(self) -> str:
        return canonical_contract_sha256(self)


IFVG_CONTEXT_MODEL_PROTOCOL_ID = "ifvg_context_catboost_binary_v1"
IFVG_CONTEXT_CALIBRATION_POLICY_ID = "raw_probability_diagnostics_v1"
THRESHOLD_REPORT_GRID = (0.40, 0.50, 0.60, 0.70)
IFVG_CONTEXT_MODEL_PARAMETERS = MappingProxyType(
    {
        "iterations": 200,
        "depth": 4,
        "learning_rate": 0.08,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "l2_leaf_reg": 3,
        "random_seed": 7,
        "random_strength": 0,
        "bootstrap_type": "No",
        "boosting_type": "Plain",
        "grow_policy": "SymmetricTree",
        "task_type": "CPU",
        "thread_count": 1,
        "class_weights": None,
        "use_best_model": False,
        "allow_writing_files": False,
        "verbose": False,
    }
)


class IfvgContextExperimentConfig(_FrozenModel):
    dataset: IfvgContextExperimentDatasetConfig
    feature_tier: ContextFeatureTier
    label: IfvgContextLabelConfig = Field(default_factory=IfvgContextLabelConfig)
    observation_filters: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    model_protocol_id: Literal["ifvg_context_catboost_binary_v1"] = (
        IFVG_CONTEXT_MODEL_PROTOCOL_ID
    )
    train_days: Literal[40] = 40
    test_days: Literal[5] = 5
    step_days: Literal[5] = 5
    embargo_days: Literal[2] = 2
    minimum_train_candidates: Literal[30] = 30
    bootstrap_repetitions: Literal[10_000] = 10_000
    random_seed: Literal[7] = 7

    @model_validator(mode="after")
    def _registered_and_ready(self) -> IfvgContextExperimentConfig:
        capability = profile_capability(self.dataset.profile_name)
        if capability.status is not ProfileCapabilityStatus.RUNNABLE:
            raise ValueError(
                f"profile is {capability.status.value}: {capability.reason}"
            )
        if self.dataset.artifact_pair.v2.preparation_status is not (
            ArtifactPreparationStatus.CONTEXT_READY
        ):
            raise ValueError("v2 artifact is not context_ready")
        if self.dataset.artifact_pair.v3.preparation_status is not (
            ArtifactPreparationStatus.CONTEXT_READY
        ):
            raise ValueError("v3 artifact is not context_ready")
        if self.feature_tier in {ContextFeatureTier.M2, ContextFeatureTier.M3}:
            formula = self.dataset.artifact_pair.v3.feature_formula_version
            if formula != "ifvg_context_formula_v2":
                raise ValueError("M2/M3 require ifvg_context_formula_v2")
        return self

    @property
    def identity(self) -> str:
        return canonical_contract_sha256(self)


class IfvgContextFoldDefinition(_FrozenModel):
    fold_index: int
    train_days: tuple[str, ...]
    test_days: tuple[str, ...]
    train_candidate_ids: tuple[str, ...]
    test_candidate_ids: tuple[str, ...]
    excluded_boundary_setup_ids: tuple[str, ...] = ()
    purged_candidate_ids: tuple[str, ...] = ()
    embargoed_candidate_ids: tuple[str, ...] = ()
    valid: bool
    invalid_reason: str | None = None
    training_prevalence: float | None = None

    @model_validator(mode="after")
    def _reason_matches_validity(self) -> IfvgContextFoldDefinition:
        if self.valid == (self.invalid_reason is not None):
            raise ValueError("valid folds have no reason; invalid folds require one")
        if set(self.train_candidate_ids) & set(self.test_candidate_ids):
            raise ValueError("fold train/test candidate IDs overlap")
        return self


class IfvgContextFeatureIdentity(_FrozenModel):
    view_id: str = Field(pattern=_SHA256_PATTERN)
    artifact_pair_hash: str = Field(pattern=_SHA256_PATTERN)
    feature_registry_hash: str = Field(pattern=_SHA256_PATTERN)
    categorical_registry_hash: str = Field(pattern=_SHA256_PATTERN)
    feature_tier: ContextFeatureTier
    ordered_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    manual_feature_overrides_hash: str = Field(pattern=_SHA256_PATTERN)


class IfvgContextRunReconciliation(_FrozenModel):
    artifact_pair_match: bool
    cohort_match: bool
    label_match: bool
    folds_match: bool
    model_protocol_match: bool
    calibration_protocol_match: bool
    bootstrap_protocol_match: bool
    oos_row_ids_match: bool
    registered_tier_delta_only: bool = False
    compatible_for_metric_delta: bool
    differing_fields: tuple[str, ...] = ()


class IfvgContextExperimentResult(_FrozenModel):
    run_id: str = Field(pattern=_SHA256_PATTERN)
    config_hash: str = Field(pattern=_SHA256_PATTERN)
    view_id: str = Field(pattern=_SHA256_PATTERN)
    label_derivation_id: str = Field(pattern=_SHA256_PATTERN)
    model_protocol_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    status: str
    folds: tuple[IfvgContextFoldDefinition, ...]
    oos_row_ids: tuple[str, ...]
    candidate_research_report: dict[str, Any]
    actual_execution_report: dict[str, Any]
    feature_coverage_report: dict[str, Any]
    reconciliation_audit_report: dict[str, Any]
    model_report: dict[str, Any] | None = None
