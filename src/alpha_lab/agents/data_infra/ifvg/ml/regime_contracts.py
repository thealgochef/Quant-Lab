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

import math
import re
from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import Field, model_validator

from ..features.arrow_tables import arrow_schema_hash, frame_from_arrow_bytes, frame_to_arrow_bytes
from ..features.context_bar_panel_contract import (
    PANEL_AS_OF_POLICY_REGISTRY,
    PANEL_ASSIGNMENT_MISSING_REASONS,
    PANEL_INTERVALS_SECONDS_V1,
    PANEL_PROPOSED_STAMPS,
)
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
    "FIT_ASSIGNMENT_SCHEMA",
    "FIT_ASSIGNMENT_SCHEMA_HASH",
    "ASSIGNMENT_ROW_KINDS",
    "AssignmentRowKind",
    "FitAssignmentRef",
    "fit_assignment_table_bytes",
    "fit_assignment_frame_from_bytes",
    "validate_assignment_rows",
    "RegimeCoverageReport",
    "FoldBootstrapStability",
    "TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2",
    "TEMPORAL_ORDER_POLICY_PANEL_V2",
    "RegimeStabilityReport",
    "RegimeCapabilityAssessment",
    "RegimeCapabilityAssessmentEnvelope",
    "RegimePromotionDecision",
    "RegimePromotionDecisionEnvelope",
    "PROMOTION_SEQUENCE",
    "MODEL_FEATURE_PROMOTION_REFUSAL",
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


_SHA256_RE = re.compile(SHA256_PATTERN)


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
            # R6.1 (owner Q3 #3): only the owner-registered intervals; the
            # source is a verified 64-hex artifact id; the as-of policy is
            # registered — never a free string
            if self.panel_interval_seconds not in PANEL_INTERVALS_SECONDS_V1:
                raise ValueError(
                    f"panel_interval_seconds={self.panel_interval_seconds} is not "
                    f"owner-registered; registered: {PANEL_INTERVALS_SECONDS_V1}"
                )
            if not _SHA256_RE.fullmatch(self.panel_source_artifact_id or ""):
                raise ValueError(
                    "panel_source_artifact_id must be the verified 64-hex "
                    "context_bar_panel_artifact_id"
                )
            if self.panel_as_of_policy_id not in PANEL_AS_OF_POLICY_REGISTRY:
                raise ValueError(
                    f"panel_as_of_policy_id {self.panel_as_of_policy_id!r} is not "
                    f"registered; registered: {tuple(PANEL_AS_OF_POLICY_REGISTRY)}"
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

#: R6.1-FIX (plan §3.4, F-05): the ENFORCED Arrow schema of the per-fit
#: assignment sidecar (``regime_fits/<id>/assignments.arrow``) — exactly
#: ``RegimeAssignmentColumns`` in order, pandas metadata stripped, so the
#: persisted bytes depend on the values and this declared schema only (never
#: on an inferred frame schema). Its hash is a semantic input of every
#: artifact that binds a fit's assignment sidecar.
FIT_ASSIGNMENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("resolved_regime_protocol_id", pa.large_string()),
        pa.field("regime_fit_id", pa.large_string()),
        pa.field("row_id", pa.large_string()),
        pa.field("fold_index", pa.int64()),
        pa.field("partition", pa.large_string()),
        pa.field("observation_ts_utc", pa.large_string()),
        pa.field("fold_local_cluster_id", pa.int64()),
        pa.field("canonical_reporting_cluster_id", pa.int64()),
        pa.field("distances", pa.list_(pa.float64())),
        pa.field("assigned_distance", pa.float64()),
        pa.field("assignment_margin", pa.float64()),
        pa.field("assignment_entropy", pa.float64()),
        pa.field("log_density", pa.float64()),
        pa.field("outlier_score", pa.float64()),
        pa.field("valid", pa.bool_()),
        pa.field("missing_reason", pa.large_string()),
    ]
)
if tuple(FIT_ASSIGNMENT_SCHEMA.names) != tuple(RegimeAssignmentColumns):
    raise AssertionError("FIT_ASSIGNMENT_SCHEMA must cover RegimeAssignmentColumns in order")
FIT_ASSIGNMENT_SCHEMA_HASH = arrow_schema_hash(FIT_ASSIGNMENT_SCHEMA)

#: The three row kinds an assignment table may carry (plan §3.4):
#: ``fit`` — the per-fit sidecar (train + test rows of ONE fit);
#: ``descriptive`` — the OOS-assignment artifact (test rows only, canonical
#: reporting id required); ``model_facing`` — the fold-local feature rows
#: (train + test rows; the canonical id is reporting-only and may be null).
ASSIGNMENT_ROW_KINDS: tuple[str, ...] = ("fit", "descriptive", "model_facing")
AssignmentRowKind = Literal["fit", "descriptive", "model_facing"]

_SHA256_RE_STRICT = re.compile(SHA256_PATTERN)
#: RA-06: tolerance of the valid-row arithmetic (assigned distance / margin)
_ASSIGNMENT_TOLERANCE = 1e-9
_ASSIGNMENT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "fold_local_cluster_id",
    "canonical_reporting_cluster_id",
    "distances",
    "assigned_distance",
    "assignment_margin",
)
_ASSIGNMENT_ROW_INVARIANT_COLUMNS: tuple[str, ...] = (
    "regime_fit_id",
    "fold_index",
    "partition",
    "fold_local_cluster_id",
    "canonical_reporting_cluster_id",
    "distances",
    "assigned_distance",
    "assignment_margin",
    "valid",
    "missing_reason",
)


class FitAssignmentRef(FrozenContract):
    """The exact persisted assignment evidence of ONE fit (plan §3.1): the
    fit identity, the manifest-verified sidecar SHA-256, and the enforced
    schema hash. Every artifact derived from a fit's assignments binds one
    of these per fit, so the artifact id determines the bytes it consumed."""

    regime_fit_id: str = Field(pattern=SHA256_PATTERN)
    assignments_sidecar_sha256: str = Field(pattern=SHA256_PATTERN)
    assignment_schema_hash: str = Field(pattern=SHA256_PATTERN)


def _fit_assignment_frame_for_schema(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RegimeAssignmentColumns) - set(frame.columns))
    if missing:
        raise ValueError(f"assignment frame lacks the enforced schema columns: {missing}")
    out = frame.loc[:, list(RegimeAssignmentColumns)].copy().reset_index(drop=True)
    for column in ("fold_index",):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
    for column in ("fold_local_cluster_id", "canonical_reporting_cluster_id"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype("Int64")
    for column in (
        "resolved_regime_protocol_id",
        "regime_fit_id",
        "row_id",
        "partition",
        "observation_ts_utc",
        "missing_reason",
    ):
        out[column] = out[column].astype(object).where(out[column].notna(), None)
        out[column] = out[column].map(lambda v: None if v is None else str(v))
    for column in (
        "assigned_distance",
        "assignment_margin",
        "assignment_entropy",
        "log_density",
        "outlier_score",
    ):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    out["distances"] = out["distances"].map(
        lambda v: None if v is None or (isinstance(v, float) and math.isnan(v))
        else [float(x) for x in v]
    )
    out["valid"] = out["valid"].astype(bool)
    return out


def fit_assignment_table_bytes(frame: pd.DataFrame) -> bytes:
    """Arrow IPC bytes of a fit's assignment frame under the ENFORCED schema."""

    return frame_to_arrow_bytes(_fit_assignment_frame_for_schema(frame), FIT_ASSIGNMENT_SCHEMA)


def fit_assignment_frame_from_bytes(data: bytes) -> pd.DataFrame:
    """The verified sidecar bytes → frame; the bytes must carry exactly the
    enforced schema (a differently typed sidecar is refused)."""

    from io import BytesIO  # noqa: PLC0415

    import pyarrow.ipc  # noqa: PLC0415

    with pyarrow.ipc.open_file(BytesIO(data)) as reader:
        table = reader.read_all()
    stored = arrow_schema_hash(table.schema)
    if stored != FIT_ASSIGNMENT_SCHEMA_HASH:
        raise ValueError(
            "fit assignment sidecar does not carry the enforced FIT_ASSIGNMENT_SCHEMA "
            f"(schema hash {stored[:12]}… != {FIT_ASSIGNMENT_SCHEMA_HASH[:12]}…)"
        )
    frame = frame_from_arrow_bytes(data)
    frame["distances"] = frame["distances"].map(
        lambda v: None if v is None else [float(x) for x in v]
    )
    return frame


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float | np.floating):
        return bool(np.isnan(value))
    try:
        return bool(pd.isna(value)) if not isinstance(value, list | tuple | np.ndarray) else False
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def validate_assignment_rows(
    frame: pd.DataFrame,
    *,
    cluster_count: int,
    kind: AssignmentRowKind = "descriptive",
    registered_reasons: Iterable[str] | None = None,
) -> None:
    """Plan §3.4 cross-field invariants over an assignment table.

    A ``valid=true`` row carries the COMPLETE assignment: a 64-hex fit id,
    ``fold_index >= 0``, a lawful partition (``test`` only for the
    descriptive kind; ``train``/``test`` otherwise), a local cluster id in
    ``[0, k)``, a finite distance vector of length ``k``, finite assigned
    distance and margin, ``missing_reason`` null, and — except for the
    model-facing kind, whose canonical id is reporting-only — the canonical
    reporting id; it is arithmetically self-consistent (adversarial RA-06):
    ``assigned_distance == distances[fold_local_cluster_id] == min(distances)``
    and ``assignment_margin == second_smallest − smallest >= 0`` (1e-9 —
    exactly what the kernel produces and the PIT / candidate rules copy).
    A ``valid=false`` row carries NO assignment output (every output column
    null) and exactly one registered missing reason, but KEEPS its
    reconciliation linkage: the key column (``row_id`` for the fit /
    model-facing kinds, ``candidate_id`` for the descriptive kind) is never
    null, and a non-null partition / fit id / fold index must be lawful. Any
    other cross-field state is refused; there is no optional-column fallback.
    """

    if kind not in ASSIGNMENT_ROW_KINDS:
        raise ValueError(f"unknown assignment row kind {kind!r}; lawful: {ASSIGNMENT_ROW_KINDS}")
    k = int(cluster_count)
    if k < 2:
        raise ValueError("cluster_count must be at least 2")
    # the linkage key: the descriptive table keys on ``candidate_id`` (the OOS
    # schema; a fit-schema frame validated under the descriptive rule keys on
    # its ``row_id``), the fit / model-facing kinds on ``row_id``
    if kind == "descriptive":
        key_column = "candidate_id" if "candidate_id" in frame.columns else "row_id"
    else:
        key_column = "row_id"
    missing_columns = sorted(
        (set(_ASSIGNMENT_ROW_INVARIANT_COLUMNS) | {key_column}) - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(f"assignment table lacks required columns: {missing_columns}")
    if kind == "descriptive":
        lawful_partitions = {"test"}
        canonical_required = True
        reasons = set(
            registered_reasons
            if registered_reasons is not None
            else (
                *REGIME_ASSIGNMENT_MISSING_REASONS,
                *PANEL_ASSIGNMENT_MISSING_REASONS,
                "no_oos_assignment",
            )
        )
    elif kind == "fit":
        lawful_partitions = {"train", "test"}
        canonical_required = True
        reasons = set(
            registered_reasons
            if registered_reasons is not None
            else REGIME_ASSIGNMENT_MISSING_REASONS
        )
    else:
        lawful_partitions = {"train", "test"}
        canonical_required = False
        reasons = set(
            registered_reasons
            if registered_reasons is not None
            else (*REGIME_ASSIGNMENT_MISSING_REASONS, *PANEL_ASSIGNMENT_MISSING_REASONS)
        )
    for position, row in enumerate(frame.to_dict("records")):
        valid = row.get("valid")
        if _is_null(valid) or not isinstance(valid, bool | np.bool_):
            raise ValueError(f"assignment row {position}: valid must be a boolean")
        key = row.get(key_column)
        if _is_null(key) or str(key) == "":
            raise ValueError(
                f"assignment row {position}: every row requires a non-null {key_column} "
                "(reconciliation linkage)"
            )
        if bool(valid):
            fit_id = row.get("regime_fit_id")
            if _is_null(fit_id) or not _SHA256_RE_STRICT.fullmatch(str(fit_id)):
                raise ValueError(f"assignment row {position}: a valid row requires a 64-hex fit id")
            fold_index = row.get("fold_index")
            if _is_null(fold_index) or int(fold_index) < 0:
                raise ValueError(f"assignment row {position}: a valid row requires fold_index >= 0")
            partition = row.get("partition")
            if _is_null(partition) or str(partition) not in lawful_partitions:
                raise ValueError(
                    f"assignment row {position}: partition {partition!r} is not lawful for the "
                    f"{kind} kind ({sorted(lawful_partitions)})"
                )
            local = row.get("fold_local_cluster_id")
            if _is_null(local) or not 0 <= int(local) < k:
                raise ValueError(
                    f"assignment row {position}: a valid row requires a local cluster id in "
                    f"[0, {k})"
                )
            canonical = row.get("canonical_reporting_cluster_id")
            if canonical_required and _is_null(canonical):
                raise ValueError(
                    f"assignment row {position}: a valid {kind} row requires the canonical "
                    "reporting cluster id"
                )
            if not _is_null(canonical) and int(canonical) < 0:
                raise ValueError(f"assignment row {position}: canonical id must be non-negative")
            distances = row.get("distances")
            if distances is None or _is_null(distances):
                raise ValueError(
                    f"assignment row {position}: a valid row requires the complete distance vector"
                )
            values = list(distances)
            if len(values) != k or not all(_finite(v) for v in values):
                raise ValueError(
                    f"assignment row {position}: a valid row requires {k} finite distances; "
                    f"got {len(values)}"
                )
            if not _finite(row.get("assigned_distance")) or not _finite(
                row.get("assignment_margin")
            ):
                raise ValueError(
                    f"assignment row {position}: a valid row requires a finite assigned distance "
                    "and margin"
                )
            # RA-06 (b): the assignment outputs are one arithmetic fact — the
            # local id is the argmin, the assigned distance is that minimum,
            # the margin is the runner-up minus the minimum
            assigned = float(row.get("assigned_distance"))
            margin = float(row.get("assignment_margin"))
            ordered = sorted(float(v) for v in values)
            if (
                abs(float(values[int(local)]) - assigned) > _ASSIGNMENT_TOLERANCE
                or abs(ordered[0] - assigned) > _ASSIGNMENT_TOLERANCE
            ):
                raise ValueError(
                    f"assignment row {position}: assigned distance {assigned!r} is not the "
                    f"distance of local cluster {int(local)} (== the minimum of the vector)"
                )
            expected_margin = ordered[1] - ordered[0]
            if margin < -_ASSIGNMENT_TOLERANCE or abs(margin - expected_margin) > (
                _ASSIGNMENT_TOLERANCE
            ):
                raise ValueError(
                    f"assignment row {position}: assignment margin {margin!r} is not the "
                    f"runner-up distance minus the minimum ({expected_margin!r} >= 0)"
                )
            if not _is_null(row.get("missing_reason")):
                raise ValueError(
                    f"assignment row {position}: a valid row carries no missing_reason"
                )
            continue
        # RA-06 (a): an invalid row keeps its linkage — nulls are lawful (the
        # candidate never reached a fit), non-null values must be lawful
        partition = row.get("partition")
        if not _is_null(partition) and str(partition) not in lawful_partitions:
            raise ValueError(
                f"assignment row {position}: an invalid row's partition {partition!r} is not "
                f"lawful for the {kind} kind ({sorted(lawful_partitions)})"
            )
        fit_id = row.get("regime_fit_id")
        if not _is_null(fit_id) and not _SHA256_RE_STRICT.fullmatch(str(fit_id)):
            raise ValueError(
                f"assignment row {position}: an invalid row's fit id must be 64-hex when present"
            )
        fold_index = row.get("fold_index")
        if not _is_null(fold_index) and int(fold_index) < 0:
            raise ValueError(
                f"assignment row {position}: an invalid row's fold_index must be >= 0 when present"
            )
        for column in _ASSIGNMENT_OUTPUT_COLUMNS:
            value = row.get(column)
            if column == "distances":
                if value is not None and not _is_null(value):
                    raise ValueError(
                        f"assignment row {position}: an invalid row may not carry {column}"
                    )
                continue
            if not _is_null(value):
                raise ValueError(
                    f"assignment row {position}: an invalid row may not carry {column}"
                )
        reason = row.get("missing_reason")
        if _is_null(reason) or str(reason) not in reasons:
            raise ValueError(
                f"assignment row {position}: an invalid row requires one registered missing "
                f"reason; got {reason!r}"
            )


class RegimeCoverageReport(FrozenContract):
    rows_total: int = Field(ge=0)
    rows_assigned: int = Field(ge=0)
    rows_typed_null: ImmutableMap[str, int]
    per_fold_coverage: ImmutableMap[int, float]
    oos_assignment_coverage: float = Field(ge=0.0, le=1.0)
    per_cluster_occupancy: ImmutableMap[int, float]
    #: R6.1: occupancy per (fold, canonical reporting id) — key "fold:cluster"
    #: — so the occupancy and rows-per-fold gates share one key space
    per_fold_canonical_occupancy: ImmutableMap[str, float] = ImmutableMap()
    minimum_training_observations_gate: int = Field(ge=0)
    minimum_training_observations_observed: int = Field(ge=0)
    coverage_gates_passed: bool
    gate_failures: tuple[str, ...]


#: R6.1 temporal-order policies (D11): candidate/decision grains carry a
#: CANDIDATE-EVENT transition matrix (elapsed seconds, trading-day + named-
#: session resets, a stamped maximum gap); only a regular panel carries
#: ordinary temporal persistence/transition semantics.
TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2 = (
    "oos_test_rows_by_observation_ts_within_fold_day_session_reset_max_gap_v2"
)
TEMPORAL_ORDER_POLICY_PANEL_V2 = "oos_test_bars_consecutive_within_trading_day_v2"


class FoldBootstrapStability(FrozenContract):
    """Bootstrap stability of ONE valid fold (D10 — every fold, not only the
    reference fold): resamples of that fold's TRAINING matrix refitted with
    the registry-pinned parameters, Hungarian-aligned onto the fold's fit."""

    fold_index: int = Field(ge=0)
    refit_count: int = Field(ge=0)
    aligned_ami_mean: float | None
    aligned_ami_p05: float | None
    per_cluster_agreement: ImmutableMap[int, float]
    undefined_reason: str | None = None


class RegimeStabilityReport(FrozenContract):
    #: applied refits per fold (the reference fold's count; every fold uses
    #: the same applied count under the total-refit cap)
    bootstrap_refit_count: int = Field(ge=0)
    bootstrap_seed_policy: Literal[
        "rng7_plus_fold_resample_refit_seed_1000_plus_1000fold_plus_i_v2"
    ] = "rng7_plus_fold_resample_refit_seed_1000_plus_1000fold_plus_i_v2"
    bootstrap_refits_per_fold_requested: int = Field(default=0, ge=0)
    bootstrap_total_refit_cap: int = Field(default=0, ge=0)
    bootstrap_refits_per_fold_applied: int = Field(default=0, ge=0)
    #: REFERENCE-FOLD aliases (the R6 scalars keep their meaning: the first
    #: valid fold's bootstrap facts) — never the protocol-wide gate input
    bootstrap_aligned_ami_mean: float | None
    bootstrap_aligned_ami_low: float | None
    per_cluster_agreement: ImmutableMap[int, float]
    reference_fold_bootstrap_stability: FoldBootstrapStability | None = None
    per_fold_bootstrap_stability: tuple[FoldBootstrapStability, ...] = ()
    #: fraction of valid folds whose bootstrap AMI is defined
    bootstrap_fold_coverage: float | None = None
    protocol_min_bootstrap_aligned_ami_mean: float | None = None
    protocol_min_bootstrap_aligned_ami_p05: float | None = None
    bootstrap_gate_scope: Literal["protocol_wide_minimum_fold_mean_v1"] = (
        "protocol_wide_minimum_fold_mean_v1"
    )
    minimum_bootstrap_aligned_ami_mean_applied: float = 0.5
    #: Temporal facts are computed over OUT-OF-SAMPLE test rows ordered by
    #: the observation's as-of timestamp, WITHIN each fold (no cross-fold
    #: concatenation; no row-id ordering) — a true point-in-time timeline.
    #: Exactly one grain's fields are populated (validator below).
    temporal_order_policy: Literal[
        "oos_test_rows_by_observation_ts_within_fold_day_session_reset_max_gap_v2",
        "oos_test_bars_consecutive_within_trading_day_v2",
    ] = TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2
    session_scheme_id: str = "ifvg_doc_session_scheme_et_v1"
    #: ── panel grain: ordinary temporal semantics on a regular panel ──
    temporal_transition_count: int = Field(default=0, ge=0)
    temporal_persistence: float | None
    transition_matrix: tuple[tuple[float, ...], ...]
    panel_pairs_dropped_gap: int = Field(default=0, ge=0)
    panel_pairs_dropped_boundary: int = Field(default=0, ge=0)
    #: ── candidate/decision grains: the CANDIDATE-EVENT transition matrix ──
    candidate_event_transition_matrix: tuple[tuple[float, ...], ...] = ()
    candidate_event_persistence: float | None = None
    candidate_event_pairs_counted: int = Field(default=0, ge=0)
    candidate_event_pairs_dropped_gap: int = Field(default=0, ge=0)
    candidate_event_pairs_dropped_boundary: ImmutableMap[str, int] = ImmutableMap()
    candidate_event_elapsed_seconds_summary: ImmutableMap[str, float] = ImmutableMap()
    candidate_event_maximum_gap_seconds: int = Field(default=0, ge=0)
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

    @model_validator(mode="after")
    def _one_grain_populated(self):
        candidate_populated = bool(self.candidate_event_transition_matrix) or (
            self.candidate_event_pairs_counted > 0
        )
        panel_populated = bool(self.transition_matrix) or self.temporal_transition_count > 0
        if self.temporal_order_policy == TEMPORAL_ORDER_POLICY_PANEL_V2 and candidate_populated:
            raise ValueError("a panel-grain report carries no candidate-event transition facts")
        if (
            self.temporal_order_policy == TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2
            and panel_populated
        ):
            raise ValueError(
                "a candidate/decision-grain report carries no panel temporal facts — "
                "candidate events are not a regular time series"
            )
        return self


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

#: V1 (R6.1 S5): the top of the ladder is UNPERSISTABLE through every path —
#: the store and the promotion CLI share this exact refusal. MODEL_FEATURE
#: stays representable at the contract (the ladder/role tests exercise it)
#: but no V1 artifact can carry it.
MODEL_FEATURE_PROMOTION_REFUSAL = (
    "model_feature is refused: it requires the activated IFVG_REGIME_CONTEXT_V1 "
    "block and a completed controlled feature study"
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
    # R6.1: renamed from the R6 "advisory floor" — a gate that blocks
    # promotion is a minimum, not advice (owner amendment); it applies to
    # the PROTOCOL-WIDE minimum fold mean aligned AMI (D10)
    "minimum_bootstrap_aligned_ami_mean": {
        "value": 0.5,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "bootstrap_refits_per_fold": {
        "value": 50,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "bootstrap_total_refit_cap": {
        "value": 400,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "bootstrap_gate_scope": {
        "value": "protocol_wide_minimum_fold_mean_v1",
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "candidate_event_maximum_gap_seconds": {
        "value": 7200,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_trades_per_regime_stratum": {
        "value": 20,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "minimum_training_rows_per_regime_stratum": {
        "value": 60,
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "prop_event_attribution_policy": {
        "value": "source_trade_then_pit_v1",
        "stamp": "proposed_protocol_default",
        "owner_ratification_required_before_feature_eligible": True,
        "owner_decision": "30",
    },
    "regime_feature_hard_id_encoding_default": {
        "value": "none",
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
    # R6.1: the context-bar panel stamps (owner Q3; decision 28) — values
    # imported from the single leaf source, never restated
    **{
        name: {
            "value": value,
            "stamp": "proposed_protocol_default",
            "owner_ratification_required_before_feature_eligible": True,
            "owner_decision": "28",
        }
        for name, value in PANEL_PROPOSED_STAMPS.items()
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
