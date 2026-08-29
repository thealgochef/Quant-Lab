"""Pipeline contracts and executors (CS §7; IMPLEMENTATION_PLAN §13; PHASED R5).

The 16-stage operator pipeline, split into the research-bearing
:class:`PipelineSemanticIdentity` and the operational
:class:`ExecutionAttemptIdentity` (P0-3): stage and result identities key on
the semantic id; retries and resource changes mint new attempts, never new
science. Stage S11 is BLOCKED with the exact registered reason until
`RejectedCandidatePolicy` is owner-ratified and the sequential golden tests
pass. Operator availability is capability-scoped by the selected stage plan
(V3 P1-5): a strategy-only plan never waits for MBP-1 activation or regime
fitting, an MBP-1 study plan refuses before R5B, and post-V1 regime plans
refuse until the expansion release.

Execution composes the SAME primitives as the study-lane orchestrator —
`enumerate_children`, the reuse/neutrality/publication semantics, the
costed-evaluation cache, `merge_prop_vectors` (one shared implementation of
ALL-legs feasibility + the worst-firm merge), `build_frontier` — so the two
lanes cannot drift. Every stage executor is idempotent and store-reusing:
a retry re-executes the stage, and a stage whose freshly-minted semantic
result identity equals the prior attempt's verified identity is marked
``REUSED`` — reuse is PROVEN by identity, never assumed from a status flag.

The mutable per-pipeline state file mirrors the search state-file
conventions (atomic writes, O_EXCL lock with heartbeat, a
``cancel.requested`` sentinel honored at stage boundaries only).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from ..data_access import allowlist_sha256
from ..ml.decision_policies import S11_BLOCKED_REASON
from ..ml.regime_study import RegimeStudyRequest
from ..preparation import _write_json_atomic
from . import pipeline_regime as _regime
from .authorization import SyntheticAuthorizationMarker
from .charter import SearchCharterEnvelope, SimulationProtocol
from .failure import FailureReason, sanitize_failure_message
from .frontier import ObjectiveSpec, build_frontier
from .gates import evaluate_strategy_gates
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    SearchChildMembership,
    canonical_contract_sha256,
    register_identity_pair,
)
from .insights import InsightPanelEnvelope, InsightPanelPayload, render_insight_panel
from .lineage import (
    build_native_lineage_map,
    deserialize_native_lineage_map,
    persist_lineage_uniqueness,
    serialize_native_lineage_map,
)
from .orchestrator import (
    OBJECTIVE_DIRECTIONS,
    SearchChildMembershipEnvelope,
    SearchFrontierEnvelope,
    SearchFrontierPayload,
    _child_evaluation_envelope,
    _load_child_evaluation,
    _publish_child_evaluation,
    _search_lock,
    enumerate_children,
    merge_prop_vectors,
)
from .store import (
    has_envelope,
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .strategy_metrics import compute_strategy_metrics
from .verification import (
    ControlFlowGateReport,
    VerificationRunEnvelope,
    evaluate_control_flow_gates,
    validate_verification_run,
    verification_report_stamps,
)

__all__ = [
    "S11_BLOCKED_REASON",
    "PipelineRunScope",
    "QuantLabPipelineStage",
    "StageStatus",
    "WorkerPolicy",
    "PipelineSemanticSpecPayload",
    "PipelineSemanticIdentity",
    "ExecutionAttemptIdentity",
    "PipelineStageResultPayload",
    "PipelineStageResultEnvelope",
    "PipelineResultPayload",
    "PipelineResultEnvelope",
    "CANONICAL_STAGE_ORDER",
    "STAGE_DEPENDENCIES",
    "FOLD_PROTOCOL_ID_V1",
    "BOOTSTRAP_PROTOCOL_ID_DEFAULT",
    "POST_V1_REGIME_ALGORITHM_KEYS",
    "StageReadiness",
    "StagePlanReadinessReport",
    "derive_stage_plan_readiness",
    "StagePlanBlockedError",
    "assert_stage_plan_launchable",
    "PipelineWiring",
    "PipelineRunResult",
    "run_pipeline",
    "read_pipeline_state",
    "request_pipeline_cancel",
    "PUBLICATION_GATE_IDS",
    "run_publication_gates",
    "activate_pipeline_result",
    "PublicationError",
]

#: The frozen walk-forward protocol id (matches the study-cell Literal).
FOLD_PROTOCOL_ID_V1 = "ifvg_context_walkforward_40_5_5_2_v1"

#: R5 engineering default for the bootstrap-path horizon family (the horizon
#: rides the simulation identity — DECISIONS_TAKEN R3 #24). Recorded in
#: DECISIONS_TAKEN; unratified for research like every engineering default.
BOOTSTRAP_PROTOCOL_ID_DEFAULT = "day_block_bootstrap_h90_v1"

#: Post-V1 regime-expansion algorithm keys (ML plan §4). Named here as a
#: REFUSAL vocabulary only — no fit implementation exists or is callable in
#: V1; R6 lands `kmeans_v1` and the post-V1 expansion release lands these.
POST_V1_REGIME_ALGORITHM_KEYS: tuple[str, ...] = (
    "minibatch_kmeans_v1",
    "gaussian_mixture_v1",
    "spectral_clustering_train_only_v1",
    "nystrom_kmeans_v1",
    "surrogate_assignment_logistic_v1",
)

_STATE_FILENAME = "pipeline_state.json"
_CANCEL_SENTINEL = "cancel.requested"
_PROP_VECTORS_SIDECAR = "prop_vectors.json"
#: R6.1: core_replay_id -> {account_simulation_id: [firm_label, mode]} of the
#: simulations THIS stage persisted (attempt-invariant; reused attempts
#: recover it from the prior stage result — never a store scan)
_ACCOUNT_SIMULATIONS_SIDECAR = "account_simulations.json"


class PipelineRunScope(StrEnum):
    VERIFICATION_5D = "verification_5d"
    FULL_AUTHORIZED_DEVELOPMENT = "full_authorized_development"


class QuantLabPipelineStage(StrEnum):
    S00_VALIDATE_INPUTS = "00_validate_inputs"
    S01_PREPARE_STRATEGY_PROFILES = "01_prepare_strategy_profiles"
    S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS = "02_run_or_reuse_sequential_replays"
    S03_BUILD_OR_REUSE_FSM_AUDIT = "03_build_or_reuse_fsm_audit"
    S04_BUILD_OR_REUSE_REPLAY_CHARTS = "04_build_or_reuse_replay_charts"
    S05_MATERIALIZE_FEATURE_VIEWS = "05_materialize_feature_views"
    S06_VALIDATE_FEATURE_COVERAGE = "06_validate_feature_coverage"
    S07_DERIVE_LABELS = "07_derive_labels"
    S08_BUILD_FOLDS = "08_build_folds"
    S09_TRAIN_MODELS = "09_train_models"
    S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS = "10_generate_predictions_and_diagnostics"
    S11_RUN_FROZEN_MODEL_GATED_REPLAYS = "11_run_frozen_model_gated_replays"
    S12_RUN_PROP_HISTORICAL_REPLAYS = "12_run_prop_historical_replays"
    S13_RUN_BOOTSTRAP_AND_STRESS = "13_run_bootstrap_and_stress"
    S14_BUILD_FRONTIER_AND_INSIGHTS = "14_build_frontier_and_insights"
    S15_VERIFY_AND_PUBLISH = "15_verify_and_publish"


class StageStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    REUSED = "reused"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED_AT_SAFE_BOUNDARY = "cancelled_at_safe_boundary"
    BLOCKED = "blocked"


CANONICAL_STAGE_ORDER: tuple[QuantLabPipelineStage, ...] = tuple(QuantLabPipelineStage)

#: In-plan prerequisites per stage — a plan naming a stage must name its
#: prerequisites, so capability-scoped subsets stay coherent.
STAGE_DEPENDENCIES: Mapping[QuantLabPipelineStage, tuple[QuantLabPipelineStage, ...]] = {
    QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES: (
        QuantLabPipelineStage.S00_VALIDATE_INPUTS,
    ),
    QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS: (
        QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
    ),
    QuantLabPipelineStage.S03_BUILD_OR_REUSE_FSM_AUDIT: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S04_BUILD_OR_REUSE_REPLAY_CHARTS: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S06_VALIDATE_FEATURE_COVERAGE: (
        QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS,
    ),
    QuantLabPipelineStage.S07_DERIVE_LABELS: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S08_BUILD_FOLDS: (QuantLabPipelineStage.S07_DERIVE_LABELS,),
    QuantLabPipelineStage.S09_TRAIN_MODELS: (
        QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS,
        QuantLabPipelineStage.S07_DERIVE_LABELS,
        QuantLabPipelineStage.S08_BUILD_FOLDS,
    ),
    QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS: (
        QuantLabPipelineStage.S09_TRAIN_MODELS,
    ),
    QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS: (
        QuantLabPipelineStage.S09_TRAIN_MODELS,
    ),
    QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S13_RUN_BOOTSTRAP_AND_STRESS: (
        QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS,
    ),
    QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS: (
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    ),
    QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH: (
        QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS,
    ),
}


class WorkerPolicy(FrozenContract):
    """Operational resources — attempt-scoped, never semantic (P0-3)."""

    max_workers: int = Field(ge=1, le=4)
    max_tasks_per_child: int = Field(ge=1)
    memory_budget_bytes: int = Field(ge=0)
    start_method: Literal["spawn"] = "spawn"


class PipelineSemanticSpecPayload(FrozenContract):
    """Every research-bearing pipeline field — and nothing operational."""

    run_scope: PipelineRunScope
    date_allowlist: tuple[str, ...]
    allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    warmup_policy_id: str
    search_charter_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    source_artifact_ids: tuple[str, ...]
    feature_bundle_ids: tuple[str, ...]
    label_policy_id: str | None
    fold_protocol_id: str | None
    model_protocol_id: str | None
    cost_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    account_policy_set_ids: tuple[str, ...]
    portfolio_policy_ids: tuple[str, ...]
    simulation_protocol: SimulationProtocol
    software_commits: ImmutableMap[str, str]
    stage_plan: tuple[QuantLabPipelineStage, ...]
    #: R6.1 (D1): the frozen regime study request — None for every plan that
    #: runs no regime study (its absence is part of the identity too).
    regime_study: RegimeStudyRequest | None = None

    @model_validator(mode="after")
    def _lawful(self):
        if self.allowlist_hash != allowlist_sha256(self.date_allowlist):
            raise ValueError("allowlist_hash does not hash the date allowlist")
        if self.run_scope is PipelineRunScope.VERIFICATION_5D and len(self.date_allowlist) > 5:
            raise ValueError("the verification scope admits at most five real days")
        plan = self.stage_plan
        if not plan:
            raise ValueError("the stage plan cannot be empty")
        if len(set(plan)) != len(plan):
            raise ValueError("the stage plan contains duplicate stages")
        positions = {stage: index for index, stage in enumerate(CANONICAL_STAGE_ORDER)}
        if [positions[stage] for stage in plan] != sorted(positions[stage] for stage in plan):
            raise ValueError("the stage plan must follow the canonical stage order")
        if QuantLabPipelineStage.S00_VALIDATE_INPUTS not in plan:
            raise ValueError("every stage plan begins with 00_validate_inputs")
        planned = set(plan)
        for stage in plan:
            for prerequisite in STAGE_DEPENDENCIES.get(stage, ()):
                if prerequisite not in planned:
                    raise ValueError(
                        f"stage {stage.value} requires {prerequisite.value} in the plan"
                    )
        if QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS in planned and (
            not self.feature_bundle_ids
        ):
            raise ValueError("feature stages require at least one feature bundle")
        if self.feature_bundle_ids and (
            QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS not in planned
        ):
            # adversarial m-12: an inert bundle claim inside the semantic
            # identity (e.g. an MBP-1 bundle a strategy-only plan never
            # touches) is refused rather than carried
            raise ValueError(
                "feature bundles are declared but no feature stage is planned"
            )
        if QuantLabPipelineStage.S07_DERIVE_LABELS in planned and not self.label_policy_id:
            raise ValueError("07_derive_labels requires a label policy id")
        if QuantLabPipelineStage.S08_BUILD_FOLDS in planned and not self.fold_protocol_id:
            raise ValueError("08_build_folds requires a fold protocol id")
        if (
            QuantLabPipelineStage.S09_TRAIN_MODELS in planned
            and not self.model_protocol_id
            and self.regime_study is None
        ):
            raise ValueError("09_train_models requires a model protocol id (or a regime study)")
        if self.regime_study is not None:
            # R6.1 (D14): computation-path-scoped stage rules — a descriptive
            # study runs S09a only; model-bearing classes require the labels,
            # a supervised protocol, and the exact frozen authority (validated
            # by the request itself)
            problems = self.regime_study.stage_plan_problems(
                feature_bundle_ids=tuple(self.feature_bundle_ids),
                stage_values=tuple(stage.value for stage in plan),
                label_policy_id=self.label_policy_id,
                model_protocol_id=self.model_protocol_id,
            )
            if problems:
                raise ValueError("regime study: " + "; ".join(problems))
        return self


class PipelineSemanticIdentity(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "pipeline_semantic_id"

    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    payload: PipelineSemanticSpecPayload


class ExecutionAttemptIdentity(FrozenContract):
    """One operational attempt — deliberately NOT an identity envelope.

    Attempt facts (workers, host, timestamps, retry reason) belong to the
    forbidden-payload vocabulary by design: they can never enter a hashed
    scientific identity, so this contract is not registered with the
    identity-projection audit.
    """

    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    worker_policy: WorkerPolicy
    attempt_number: int = Field(ge=1)
    host_environment: ImmutableMap[str, str]
    started_at: str
    ended_at: str | None
    operational_retry_reason: str | None


class PipelineStageResultPayload(FrozenContract):
    """One stage's semantic result: the exact artifacts it produced/reused."""

    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    stage: QuantLabPipelineStage
    output_artifact_ids: tuple[str, ...]


class PipelineStageResultEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "stage_result_id"

    stage_result_id: str = Field(pattern=SHA256_PATTERN)
    payload: PipelineStageResultPayload


class PipelineResultPayload(FrozenContract):
    """The immutable pipeline result (S15). Publication/activation state is
    OPERATIONAL and lives in the mutable state file + catalog, never here."""

    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    run_scope: PipelineRunScope
    stage_result_ids: ImmutableMap[str, str]
    frontier_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    control_flow_gates: ControlFlowGateReport | None
    verification_stamps: ImmutableMap[str, Any] | None


class PipelineResultEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "pipeline_result_id"

    pipeline_result_id: str = Field(pattern=SHA256_PATTERN)
    payload: PipelineResultPayload


# ─────────────────────────────────────────────────────────────────────────────
# Capability-scoped readiness (V3 P1-5; TEST_MATRIX §3.9)
# ─────────────────────────────────────────────────────────────────────────────


class StageReadiness(FrozenContract):
    stage: QuantLabPipelineStage
    state: Literal["available", "blocked_terminal", "blocked_capability"]
    reason: str | None

    @model_validator(mode="after")
    def _blocked_has_reason(self):
        if self.state != "available" and not self.reason:
            raise ValueError("a blocked stage requires a reason")
        return self


class StagePlanReadinessReport(FrozenContract):
    entries: tuple[StageReadiness, ...]
    launchable: bool

    def entry(self, stage: QuantLabPipelineStage) -> StageReadiness:
        for candidate in self.entries:
            if candidate.stage is stage:
                return candidate
        raise KeyError(stage.value)


class StagePlanBlockedError(PermissionError):
    """The selected stage plan references unavailable capabilities."""


def derive_stage_plan_readiness(
    spec: PipelineSemanticSpecPayload,
    *,
    store_root: Path | None = None,
    run_scope: str | None = None,
) -> StagePlanReadinessReport:
    """Per-stage availability for the SELECTED plan (capability-scoped).

    ``blocked_terminal`` marks a stage that runs to a designed terminal
    BLOCKED state (S11 in V1) — it never prevents a launch.
    ``blocked_capability`` marks a spec that references an unavailable
    capability (a planned feature block such as `IFVG_ORDER_FLOW_MBP1_V1`
    before R5B, a planned model protocol, a post-V1 regime algorithm) —
    the plan refuses to launch until that capability's release activates it.
    """

    from ..features.feature_blocks import BlockUnavailableError  # noqa: PLC0415
    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415
    from ..ml.model_protocols import (  # noqa: PLC0415
        CATBOOST_BUNDLE_PROTOCOL_ID,
        CATBOOST_PROTOCOL_ID,
        LOGISTIC_PROTOCOL_ID,
        MODEL_PROTOCOL_REGISTRY,
        ModelProtocolStatus,
    )

    bundle_block_reason: str | None = None
    for bundle_key in spec.feature_bundle_ids:
        try:
            resolve_bundle(bundle_key)
        except (BlockUnavailableError, ValueError) as error:
            bundle_block_reason = sanitize_failure_message(str(error))
            break
    model_block_reason: str | None = None
    bundle_parametrized = bool(_mbp1_bearing_bundles(spec.feature_bundle_ids))
    if (
        bundle_block_reason is None
        and spec.model_protocol_id == CATBOOST_PROTOCOL_ID
        and bundle_parametrized
    ):
        # R5B/R6.1: the frozen-lane CatBoost fold runner is tier-locked in the
        # M0-M3 lane; bundle-parametrized plans run the logistic protocol or
        # the bundle-aware CatBoost rung (research-only)
        model_block_reason = (
            f"{CATBOOST_PROTOCOL_ID} has no bundle-parametrized wiring for "
            "MBP-1-bearing bundles (the CatBoost fold runner is tier-locked "
            f"in the frozen M0-M3 lane); select {LOGISTIC_PROTOCOL_ID} or the "
            f"bundle-aware {CATBOOST_BUNDLE_PROTOCOL_ID}"
        )
    elif (
        bundle_block_reason is None
        and spec.model_protocol_id == CATBOOST_BUNDLE_PROTOCOL_ID
        and not bundle_parametrized
    ):
        # R6.1: the bundle-aware rung needs a resolved-bundle identity; the
        # frozen-tier path has no wiring for it
        model_block_reason = (
            f"{CATBOOST_BUNDLE_PROTOCOL_ID} is the bundle-aware CatBoost rung of "
            "bundle-parametrized (MBP-1-bearing) plans only; frozen-tier bundles "
            f"pin {CATBOOST_PROTOCOL_ID}"
        )
    elif spec.model_protocol_id:
        if spec.model_protocol_id in POST_V1_REGIME_ALGORITHM_KEYS:
            model_block_reason = (
                f"{spec.model_protocol_id} is a post-V1 regime-expansion "
                "algorithm; no fit implementation is callable in V1"
            )
        else:
            entry = MODEL_PROTOCOL_REGISTRY.get(spec.model_protocol_id)
            if entry is None:
                model_block_reason = (
                    f"model protocol {spec.model_protocol_id!r} is not registered"
                )
            elif entry.status is not ModelProtocolStatus.AVAILABLE:
                model_block_reason = (
                    f"model protocol {spec.model_protocol_id!r} is "
                    f"{entry.status.value}: {entry.reason}"
                )
    # R6.1: a regime study blocks its own stages until every registry /
    # bundle / leakage / grain / stage-plan / frozen-authority check passes
    regime_block_reason = _regime.regime_readiness_reason(
        spec, store_root=store_root, run_scope=run_scope
    )
    regime_stages = {
        stage
        for stage in QuantLabPipelineStage
        if stage.value in _regime.REGIME_STAGE_VALUES
        and (
            stage is not QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS
            or (spec.regime_study is not None and spec.regime_study.stratified_reporting_requested)
        )
    }
    bundle_stages = {
        QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS,
        QuantLabPipelineStage.S06_VALIDATE_FEATURE_COVERAGE,
        QuantLabPipelineStage.S09_TRAIN_MODELS,
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
    }
    model_stages = {
        QuantLabPipelineStage.S09_TRAIN_MODELS,
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
    }
    entries: list[StageReadiness] = []
    for stage in spec.stage_plan:
        if stage is QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS:
            entries.append(
                StageReadiness(
                    stage=stage, state="blocked_terminal", reason=S11_BLOCKED_REASON
                )
            )
        elif bundle_block_reason and stage in bundle_stages:
            entries.append(
                StageReadiness(
                    stage=stage, state="blocked_capability", reason=bundle_block_reason
                )
            )
        elif model_block_reason and stage in model_stages:
            entries.append(
                StageReadiness(
                    stage=stage, state="blocked_capability", reason=model_block_reason
                )
            )
        elif regime_block_reason and stage in regime_stages:
            entries.append(
                StageReadiness(
                    stage=stage,
                    state="blocked_capability",
                    reason=sanitize_failure_message(regime_block_reason),
                )
            )
        else:
            entries.append(StageReadiness(stage=stage, state="available", reason=None))
    launchable = all(entry.state != "blocked_capability" for entry in entries)
    return StagePlanReadinessReport(entries=tuple(entries), launchable=launchable)


def assert_stage_plan_launchable(
    spec: PipelineSemanticSpecPayload,
    *,
    store_root: Path | None = None,
    run_scope: str | None = None,
) -> StagePlanReadinessReport:
    report = derive_stage_plan_readiness(spec, store_root=store_root, run_scope=run_scope)
    if not report.launchable:
        blocked = [
            f"{entry.stage.value}: {entry.reason}"
            for entry in report.entries
            if entry.state == "blocked_capability"
        ]
        raise StagePlanBlockedError(
            "the selected stage plan references unavailable capabilities — "
            + "; ".join(blocked)
        )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Wiring, state, and the stage machine
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineWiring:
    """Injectable heavy seams. Every builder a planned stage needs must be
    wired or S00 refuses the launch — a stage never silently no-ops."""

    identity_resolver: Callable[..., object]
    child_runner: Callable[..., Any]
    prewarm: Callable[..., None] | None = None
    cost_points: float | None = None
    audit_builder: Callable[..., tuple[str, ...]] | None = None
    chart_builder: Callable[..., tuple[str, ...]] | None = None
    candidate_view_source: Callable[[], Any] | None = None
    label_builder: Callable[..., tuple[str, Any]] | None = None
    firm_specs: tuple[Any, ...] = ()
    bar_observations_for: Callable[..., Any] | None = None
    verification_run: VerificationRunEnvelope | None = None
    verification_authorization: object | None = None
    #: R5B: the offline MBP-1 evidence seam — a zero-argument callable
    #: returning ``(Mbp1SourceArtifactEnvelope, events_by_day, anchor_frame)``.
    #: Required whenever the plan's bundles resolve MBP-1 order-flow
    #: features; the stage never fabricates order-flow evidence.
    mbp1_evidence_source: Callable[[], tuple[Any, Mapping[str, Any], Any]] | None = None
    #: OPTIONAL caller cross-check only (safety review F3). R5-FIX finding 7:
    #: a caller-provided string is NOT evidence — the real scope requires
    #: ``loaded_seed_snapshot_id_source`` and refuses without it; when both
    #: are set they must agree.
    expected_seed_snapshot_id: str | None = None
    #: R5-FIX finding 7: LOADS the seed-snapshot artifact the runner will
    #: actually use (verified store load: manifest + file hashes +
    #: id-hashes-payload + seed-bytes rehash + profile binding) and returns
    #: its content-derived envelope id. The real verification scope REQUIRES
    #: this seam; its result — never a caller string — is what
    #: ``validate_verification_run`` checks against the authorization.
    loaded_seed_snapshot_id_source: Callable[[], str] | None = None
    #: R6.1: the VERIFIED replay-chart seam for panel-grain regime studies —
    #: ``chart_id → VerifiedReplayChartArtifact`` (the panel is materialized
    #: from the artifact's rehashed ``bars_tf.parquet`` only; never a frame).
    context_bar_source: Callable[[str], Any] | None = None


@dataclass
class PipelineRunResult:
    pipeline_semantic_id: str
    attempt: ExecutionAttemptIdentity
    stage_statuses: dict[str, str]
    state_path: Path
    result_envelope: PipelineResultEnvelope | None = None


@dataclass
class _RunContext:
    semantic: PipelineSemanticIdentity
    charter: SearchCharterEnvelope
    wiring: PipelineWiring
    store_root: Path
    state_root: Path
    state: dict[str, Any]
    specs: tuple[Any, ...] | None = None
    children: list[dict[str, Any]] = field(default_factory=list)
    tables_by_child: dict[str, Any] = field(default_factory=dict)
    metrics_by_child: dict[str, Any] = field(default_factory=dict)
    gates_passed: dict[str, dict[str, float]] = field(default_factory=dict)
    neutrality_by_child: dict[str, bool] = field(default_factory=dict)
    view: Any = None
    bundle_views: dict[str, Any] = field(default_factory=dict)
    bundle_frames: dict[str, Any] = field(default_factory=dict)
    labeled: Any = None
    label_artifact_id: str | None = None
    folds: Any = None
    ladder: Any = None
    mbp1_evidence: dict[str, Any] = field(default_factory=dict)
    controlled_study: Any = None
    prop_vectors: dict[str, dict[str, Any]] = field(default_factory=dict)
    frontier_id: str | None = None
    insight_ids: tuple[str, ...] = ()
    comparison_result_ids: tuple[str, ...] = ()
    stage_sidecars: dict[str, bytes] = field(default_factory=dict)
    lineage_sidecars: dict[str, dict] = field(default_factory=dict)
    #: safety review F1: any drive's zero-counter assertion tripping is
    #: recorded here so S15's gate is DERIVED from the run, never asserted
    forbidden_access_detected: bool = False
    synthetic: bool = False
    result_envelope: PipelineResultEnvelope | None = None
    #: R6.1: the regime study's run-scoped state (observation ref, protocol,
    #: fold sets, execution result, decisions) — see ``pipeline_regime``
    regime: dict[str, Any] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_isna(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _state_dir(state_root: Path, pipeline_semantic_id: str) -> Path:
    return Path(state_root) / pipeline_semantic_id


def read_pipeline_state(state_root: Path, pipeline_semantic_id: str) -> dict | None:
    path = _state_dir(state_root, pipeline_semantic_id) / _STATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def request_pipeline_cancel(state_root: Path, pipeline_semantic_id: str) -> Path:
    directory = _state_dir(state_root, pipeline_semantic_id)
    directory.mkdir(parents=True, exist_ok=True)
    sentinel = directory / _CANCEL_SENTINEL
    sentinel.write_text("cancel requested\n", encoding="utf-8")
    return sentinel


def _initial_state(semantic: PipelineSemanticIdentity) -> dict[str, Any]:
    planned = set(semantic.payload.stage_plan)
    return {
        "schema_version": 1,
        "pipeline_semantic_id": semantic.pipeline_semantic_id,
        "run_scope": semantic.payload.run_scope.value,
        "search_charter_id": semantic.payload.search_charter_id,
        "current_stage": None,
        "stages": {
            stage.value: {
                "in_plan": stage in planned,
                "status": StageStatus.PENDING.value if stage in planned else None,
                "explanation": (
                    "" if stage in planned else "not required by this stage plan"
                ),
                "stage_result_id": None,
                "output_artifact_ids": [],
                "started_at": None,
                "ended_at": None,
            }
            for stage in CANONICAL_STAGE_ORDER
        },
        "attempts": [],
        "children": [],
        "warnings": [],
        "publication": {"state": "not_prepared", "gates": None, "activated": False},
        # derived (safety review F7): only a full_authorized_development
        # run — a separate, explicitly authorized operator action — could
        # ever flip this
        "full_pipeline_not_run": (
            semantic.payload.run_scope
            is not PipelineRunScope.FULL_AUTHORIZED_DEVELOPMENT
        ),
    }


def _checkpoint(context: _RunContext, *, heartbeat: Path | None = None) -> Path:
    if heartbeat is not None:
        with suppress(OSError):
            os.utime(heartbeat)
    directory = _state_dir(context.state_root, context.semantic.pipeline_semantic_id)
    path = directory / _STATE_FILENAME
    _write_json_atomic(path, context.state)
    return path


def _record_children(context: _RunContext) -> None:
    context.state["children"] = list(context.children)


def _resolve_core_id(context: _RunContext, spec_child) -> tuple[str, object | None]:
    resolved = context.wiring.identity_resolver(spec_child)
    if isinstance(resolved, str):
        return resolved, None
    return resolved.core_replay_id, resolved


def _ensure_specs(context: _RunContext) -> tuple[Any, ...]:
    if context.specs is None:
        context.specs = enumerate_children(
            context.charter,
            identity_resolver=lambda spec_child: _resolve_core_id(context, spec_child)[0],
        )
    return context.specs


# ─────────────────────────────────────────────────────────────────────────────
# Stage executors — each returns (output_artifact_ids, explanation)
# ─────────────────────────────────────────────────────────────────────────────


def _loaded_seed_snapshot_id_for_real_scope(wiring: PipelineWiring) -> str:
    """The seed-snapshot id the runner ACTUALLY loads (R5-FIX finding 7).

    The real verification scope refuses without a wired
    ``loaded_seed_snapshot_id_source`` — a caller-provided
    ``expected_seed_snapshot_id`` string is a cross-check, never evidence.
    The source performs a VERIFIED artifact load and returns the loaded
    envelope's content-derived id; a wired-but-disagreeing caller
    expectation refuses before any source path is constructed.
    """

    if wiring.loaded_seed_snapshot_id_source is None:
        raise PermissionError(
            "the real verification scope requires the loaded-seed-snapshot "
            "source (PipelineWiring.loaded_seed_snapshot_id_source): the "
            "expected seed must be derived from the verified artifact the "
            "runner will load — a caller-provided id is not evidence "
            "(fail-before-path)"
        )
    loaded = wiring.loaded_seed_snapshot_id_source()
    if (
        wiring.expected_seed_snapshot_id is not None
        and wiring.expected_seed_snapshot_id != loaded
    ):
        raise PermissionError(
            "the caller's expected seed snapshot id disagrees with the "
            "verified loaded artifact; refused before any source path "
            f"(expected {wiring.expected_seed_snapshot_id[:12]}…, loaded "
            f"{loaded[:12]}…)"
        )
    return loaded


def _mbp1_bearing_bundles(feature_bundle_ids: tuple[str, ...]) -> tuple[str, ...]:
    """The subset of RESOLVABLE bundles that carry MBP-1 order-flow features."""

    from ..features.bundle_feature_view import mbp1_block_keys_in_bundle  # noqa: PLC0415
    from ..features.feature_blocks import BlockUnavailableError  # noqa: PLC0415
    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415

    bearing: list[str] = []
    for bundle_key in feature_bundle_ids:
        try:
            envelope = resolve_bundle(bundle_key)
        except (BlockUnavailableError, ValueError):
            continue  # unresolvable bundles are the readiness layer's refusal
        if mbp1_block_keys_in_bundle(envelope):
            bearing.append(bundle_key)
    return tuple(bearing)


def _stage_s00_validate(context: _RunContext) -> tuple[tuple[str, ...], str]:
    spec = context.semantic.payload
    charter = context.charter
    problems: list[str] = []
    if spec.search_charter_id != charter.search_id:
        problems.append("pipeline spec does not reference the supplied charter")
    if spec.date_allowlist != charter.payload.date_policy.replay_dates:
        problems.append("date allowlist disagrees with the charter date policy")
    if spec.cost_policy_sha256 != canonical_contract_sha256(charter.payload.cost_policy):
        problems.append("cost policy hash disagrees with the charter cost policy")
    if spec.simulation_protocol != charter.payload.simulation_protocol:
        problems.append(
            "simulation protocol disagrees with the charter simulation protocol"
        )
    planned = set(spec.stage_plan)
    wiring = context.wiring
    for stage, attribute in (
        (QuantLabPipelineStage.S03_BUILD_OR_REUSE_FSM_AUDIT, "audit_builder"),
        (QuantLabPipelineStage.S04_BUILD_OR_REUSE_REPLAY_CHARTS, "chart_builder"),
        (QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS, "candidate_view_source"),
        (QuantLabPipelineStage.S07_DERIVE_LABELS, "label_builder"),
        # adversarial m-10: the label builder receives the candidate view,
        # so a plan with S07 needs the view source even without S05
        (QuantLabPipelineStage.S07_DERIVE_LABELS, "candidate_view_source"),
    ):
        if stage in planned and getattr(wiring, attribute) is None:
            problems.append(f"{stage.value} planned but no {attribute} is wired")
    if (
        QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS in planned
        and _mbp1_bearing_bundles(spec.feature_bundle_ids)
        and wiring.mbp1_evidence_source is None
    ):
        problems.append(
            "an MBP-1-bearing bundle is planned but no mbp1_evidence_source "
            "is wired (order-flow evidence is never fabricated)"
        )
    problems.extend(_regime.regime_wiring_problems(spec, wiring))
    prop_planned = QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS in planned
    if prop_planned and not wiring.firm_specs:
        problems.append("12_run_prop_historical_replays planned but no firm specs wired")
    if prop_planned:
        expected_policy_ids = tuple(
            sorted(
                firm_spec.policy_set_envelope().account_policy_set_id
                for firm_spec in wiring.firm_specs
            )
        )
        if tuple(sorted(spec.account_policy_set_ids)) != expected_policy_ids:
            problems.append(
                "account_policy_set_ids do not match the wired firm specs' "
                "resolved policy-set envelope ids"
            )
    if "historical_1m_scenario" in charter.payload.simulation_protocol.modes and (
        wiring.bar_observations_for is None
    ):
        problems.append("scenario mode planned but no bar-observation source is wired")
    if spec.fold_protocol_id not in (None, FOLD_PROTOCOL_ID_V1):
        problems.append(
            f"fold protocol {spec.fold_protocol_id!r} is not the frozen "
            f"{FOLD_PROTOCOL_ID_V1}"
        )
    if problems:
        raise ValueError("input validation failed: " + "; ".join(problems))

    context.synthetic = isinstance(
        charter.payload.owner_authorization, SyntheticAuthorizationMarker
    )
    # R6.1: readiness verified-loads a model-bearing study's frozen authority
    # from THIS store before any path is constructed
    readiness = assert_stage_plan_launchable(
        spec, store_root=context.store_root, run_scope=_regime.regime_run_scope(context)
    )
    if spec.run_scope is PipelineRunScope.VERIFICATION_5D:
        if context.synthetic:
            branch = (
                "synthetic control-flow verification: the charter carries the "
                "typed synthetic marker and no real source path is constructed; "
                "the REAL five-day slice stays blocked on the owner's "
                "VerificationAuthorizationRef"
            )
        else:
            if wiring.verification_run is None:
                raise PermissionError(
                    "the real verification scope requires a VerificationRunEnvelope"
                )
            run_payload = wiring.verification_run.payload
            if tuple(run_payload.allowlist) != tuple(spec.date_allowlist) or (
                run_payload.allowlist_hash != spec.allowlist_hash
            ):
                raise PermissionError(
                    "the verification run envelope's allowlist does not match "
                    "the pipeline spec's date allowlist (adversarial m-16; "
                    "fail-before-path)"
                )
            validate_verification_run(
                wiring.verification_run,
                expected_pipeline_semantic_id=context.semantic.pipeline_semantic_id,
                expected_baseline_profile_id=charter.payload.baseline_profile_name,
                expected_baseline_section_config_hash=(
                    charter.payload.baseline_section_config_hash
                ),
                expected_seed_snapshot_id=_loaded_seed_snapshot_id_for_real_scope(
                    wiring
                ),
                authorization=wiring.verification_authorization,
            )
            branch = "real verification authorization validated before any source path"
    else:
        if context.synthetic:
            raise PermissionError(
                "full_authorized_development can never run under the synthetic "
                "authorization marker"
            )
        branch = "development scope: owner authorization bundle present on the charter"
    context.stage_sidecars["stage_plan_readiness.json"] = (
        json.dumps(readiness.model_dump(mode="json"), sort_keys=True) + "\n"
    ).encode("utf-8")
    return (), f"inputs validated; {branch}"


def _regime_needs_child_tables(context: _RunContext) -> bool:
    """R6.1: stratified regime reports read each child's executed-trade table."""

    request = _regime.regime_request(context.semantic.payload)
    return request is not None and bool(request.stratified_reporting_requested)


def _stage_s01_prepare(context: _RunContext) -> tuple[tuple[str, ...], str]:
    specs = _ensure_specs(context)
    if len(specs) > context.charter.payload.max_child_count:
        raise ValueError("enumerated children exceed the charter ceiling")
    context.children = [
        {
            "ordinal": spec_child.ordinal,
            "core_replay_id": _resolve_core_id(context, spec_child)[0],
            "axis_value_ids": dict(spec_child.axis_value_ids),
            "comparison_role": spec_child.comparison_role,
            "state": "queued",
            "failure_reason": None,
            "explanation": "",
            "replay_invocations": 0,
        }
        for spec_child in specs
    ]
    _record_children(context)
    blocked = sum(
        1
        for spec_child in specs
        if spec_child.capability is not None
        and spec_child.capability.status != "generated_runnable"
    )
    return (), (
        f"{len(specs)} children enumerated (deduped on core replay identity); "
        f"{blocked} blocked by generated-profile capability"
    )


def _stage_s02_replays(context: _RunContext) -> tuple[tuple[str, ...], str]:

    specs = _ensure_specs(context)
    charter_payload = context.charter.payload
    cost = (
        charter_payload.cost_policy.cost_points_round_turn
        if context.wiring.cost_points is None
        else context.wiring.cost_points
    )
    if context.wiring.prewarm is not None:
        context.wiring.prewarm(specs)
    outputs: list[str] = []
    rows_by_ordinal = {row["ordinal"]: row for row in context.children}
    for spec_child in specs:
        row = rows_by_ordinal[spec_child.ordinal]
        core_replay_id = row["core_replay_id"]
        if spec_child.capability is not None and (
            spec_child.capability.status != "generated_runnable"
        ):
            row["state"] = "blocked"
            row["failure_reason"] = FailureReason.BLOCKED_AXIS.value
            row["explanation"] = (
                f"generated profile blocked before replay: "
                f"{spec_child.capability.status} — {spec_child.capability.reason}"
            )
            _record_children(context)
            continue
        _, envelope = _resolve_core_id(context, spec_child)
        if has_envelope(context.store_root, "core_replays", core_replay_id):
            row["state"] = "reused"
            row["explanation"] = (
                "verified reuse: an immutable replay with this exact identity "
                "already exists (zero replay invocations)"
            )
            if _regime_needs_child_tables(context):
                # R6.1: the regime study's stratified reports need this child's
                # executed-trade tables, which a reused replay does not carry —
                # re-derive them and REQUIRE the reproduction to match the
                # persisted costed evaluation exactly (verified reuse by
                # reproduction; the immutable replay is never rewritten)
                try:
                    result = context.wiring.child_runner(
                        spec=spec_child, core_replay_id=core_replay_id
                    )
                    row["replay_invocations"] = 1
                    persisted = _load_child_evaluation(
                        context.store_root,
                        _child_evaluation_envelope(
                            core_replay_id, charter_payload.cost_policy
                        ).costed_evaluation_id,
                    )
                    if persisted is None:
                        # adversarial R6.1 S4: nothing persisted can verify the
                        # reproduction under THIS cost policy — the re-derived
                        # tables are NOT adopted as this run's evidence (the
                        # child stays reused; its gates are not evaluated this
                        # run and it is left out of the stratified reports)
                        row["explanation"] = (
                            "verified reuse: the immutable replay exists; the child was "
                            "re-derived for the regime study's stratified reports but "
                            "this cost policy has no persisted costed evaluation to "
                            "reproduce — reproduction unverifiable; the re-derived "
                            "tables are not this run's evidence"
                        )
                    else:
                        reproduced = compute_strategy_metrics(
                            result.tables,
                            cost_points=cost,
                            evaluation_config_hash=canonical_contract_sha256(
                                {
                                    "core_replay_id": core_replay_id,
                                    "cost_policy": charter_payload.cost_policy.model_dump(
                                        mode="json"
                                    ),
                                }
                            ),
                        )
                        if reproduced.model_dump(mode="json") != persisted.model_dump(
                            mode="json"
                        ):
                            raise RuntimeError(
                                "the re-derived child tables do not reproduce the "
                                "persisted costed evaluation; refusing to treat the "
                                "reused replay as this run's evidence"
                            )
                        context.tables_by_child[core_replay_id] = result
                        row["explanation"] = (
                            "verified reuse by reproduction: the immutable replay "
                            "exists; the child was re-derived for the regime study's "
                            "stratified reports and reproduced its persisted costed "
                            "evaluation"
                        )
                except Exception as error:  # noqa: BLE001 — per-child containment
                    row["state"] = "failed"
                    row["failure_reason"] = FailureReason.REPLAY.value
                    row["explanation"] = sanitize_failure_message(str(error))
                    _record_children(context)
                    _checkpoint(context)
                    continue
        else:
            row["state"] = "running"
            _record_children(context)
            _checkpoint(context)
            try:
                result = context.wiring.child_runner(
                    spec=spec_child, core_replay_id=core_replay_id
                )
                row["replay_invocations"] = 1
                neutrality = getattr(result, "neutrality", None)
                if neutrality is not None and not neutrality.passed:
                    raise RuntimeError(
                        "child audit-neutrality FAILED - the child cannot "
                        "publish (core tables differ with the audit channel "
                        "enabled, or audit stamps broke referential integrity)"
                    )
                context.neutrality_by_child[core_replay_id] = bool(
                    neutrality.passed if neutrality is not None else True
                )
                context.tables_by_child[core_replay_id] = result
                if envelope is not None:
                    save_or_reuse_envelope(context.store_root, "core_replays", envelope)
                row["state"] = "completed"
            except Exception as error:  # noqa: BLE001 — per-child containment
                if isinstance(error, AssertionError) and (
                    "IFVG source" in str(error) or "source access" in str(error)
                ):
                    # a drive's zero-counter assertion tripped: S15's
                    # zero_forbidden_counters gate derives False from this
                    context.forbidden_access_detected = True
                row["state"] = "failed"
                row["failure_reason"] = FailureReason.REPLAY.value
                row["explanation"] = sanitize_failure_message(str(error))
                _record_children(context)
                _checkpoint(context)
                continue
        membership_envelope = SearchChildMembershipEnvelope.from_payload(
            SearchChildMembership(
                parent_search_id=context.charter.search_id,
                child_ordinal=spec_child.ordinal,
                axis_value_ids=dict(spec_child.axis_value_ids),
                core_replay_id=core_replay_id,
                comparison_role=spec_child.comparison_role,  # type: ignore[arg-type]
            )
        )
        save_or_reuse_envelope(context.store_root, "memberships", membership_envelope)
        outputs.append(core_replay_id)

        evaluation = _child_evaluation_envelope(core_replay_id, charter_payload.cost_policy)
        result = context.tables_by_child.get(core_replay_id)
        if result is not None:
            metrics = compute_strategy_metrics(
                result.tables,
                cost_points=cost,
                evaluation_config_hash=canonical_contract_sha256(
                    {
                        "core_replay_id": core_replay_id,
                        "cost_policy": charter_payload.cost_policy.model_dump(mode="json"),
                    }
                ),
            )
            _publish_child_evaluation(context.store_root, evaluation, metrics)
            # DEV-R4-16 closure: persist the profile-independent lineage
            # evidence — the uniqueness report immutably, the delta-capable
            # key projection as this stage's sidecar.
            lineage_map = build_native_lineage_map(
                result.tables, core_replay_id=core_replay_id
            )
            persist_lineage_uniqueness(context.store_root, lineage_map.uniqueness_report)
            serialized = serialize_native_lineage_map(lineage_map)
            context.lineage_sidecars[core_replay_id] = serialized
            context.stage_sidecars[f"lineage_map_{core_replay_id}.json"] = (
                json.dumps(serialized, sort_keys=True) + "\n"
            ).encode("utf-8")
            day_funnels = getattr(result, "day_funnels", None)
            if day_funnels:
                context.stage_sidecars[f"day_funnels_{core_replay_id}.json"] = (
                    json.dumps(
                        {
                            str(day): {
                                str(key): int(value)
                                for key, value in dict(counters).items()
                                if isinstance(value, (int, float))
                                and not _safe_isna(value)
                            }
                            for day, counters in dict(day_funnels).items()
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
        else:
            metrics = _load_child_evaluation(
                context.store_root, evaluation.costed_evaluation_id
            )
            if metrics is None:
                if "reproduction unverifiable" not in str(row.get("explanation") or ""):
                    row["explanation"] = (
                        "reused replay has no published costed evaluation for "
                        "this cost policy; gates were not evaluated this run"
                    )
                _record_children(context)
                continue
        context.metrics_by_child[core_replay_id] = metrics
        report = evaluate_strategy_gates(
            metrics, charter_payload.objective_policy.feasibility_gates
        )
        if report.passed:
            objective_values = {
                objective: getattr(metrics, objective)
                for objective in charter_payload.objective_policy.pareto_objectives
                if hasattr(metrics, objective)
                and getattr(metrics, objective) is not None
            }
            context.gates_passed[core_replay_id] = objective_values
        else:
            row["failure_reason"] = (
                report.failure_reason.value if report.failure_reason else None
            )
            row["explanation"] = report.human_explanation
        _record_children(context)
        _checkpoint(context)
    completed = sum(
        1 for row in context.children if row["state"] in ("completed", "reused")
    )
    return tuple(sorted(set(outputs))), (
        f"{completed}/{len(context.children)} children completed or reused; "
        f"{len(context.gates_passed)} passed the strategy gates"
    )


def _companion_stage(
    context: _RunContext, builder: Callable[..., tuple[str, ...]], kind: str
) -> tuple[tuple[str, ...], str]:
    outputs: list[str] = []
    built = 0
    for row in context.children:
        if row["state"] not in ("completed", "reused"):
            continue
        result = context.tables_by_child.get(row["core_replay_id"])
        outputs.extend(builder(row, result))
        built += 1
    return tuple(sorted(set(outputs))), (
        f"{kind} companions built or reused for {built} children"
    )


def _stage_s03_audit(context: _RunContext) -> tuple[tuple[str, ...], str]:
    assert context.wiring.audit_builder is not None
    return _companion_stage(context, context.wiring.audit_builder, "audit")


def _stage_s04_charts(context: _RunContext) -> tuple[tuple[str, ...], str]:
    assert context.wiring.chart_builder is not None
    outputs: list[str] = []
    charts_by_child: dict[str, tuple[str, ...]] = {}
    built = 0
    for row in context.children:
        if row["state"] not in ("completed", "reused"):
            continue
        core_replay_id = str(row["core_replay_id"])
        result = context.tables_by_child.get(core_replay_id)
        chart_ids = tuple(str(chart_id) for chart_id in context.wiring.chart_builder(row, result))
        charts_by_child[core_replay_id] = chart_ids
        outputs.extend(chart_ids)
        built += 1
    unique = tuple(sorted(set(outputs)))
    # R6.1: the panel-grain regime study materializes its panel from ONE of
    # these VERIFIED replay-chart artifacts at S05 — selected by the declared
    # chart-selection policy over the per-child mapping (adversarial F13)
    context.regime["chart_ids"] = unique
    context.regime["charts_by_child"] = charts_by_child
    return unique, f"replay-chart companions built or reused for {built} children"


def _ensure_mbp1_evidence(context: _RunContext) -> dict[str, Any]:
    """Materialize + immutably persist the MBP-1 evidence exactly once.

    Runs the offline materializer over the wired evidence (source artifact,
    its per-day events, and the candidate stage anchors), persists the
    source artifact, feature artifact, and coverage report through the
    verified stores (save-or-reuse — identical evidence reuses one
    artifact), and caches the joinable frame for S05/S09.
    """

    if context.mbp1_evidence:
        return context.mbp1_evidence
    from ..features.feature_blocks import (  # noqa: PLC0415
        FEATURE_BLOCK_RESOLUTION_REGISTRY,
    )
    from ..features.mbp1_coverage import (  # noqa: PLC0415
        build_mbp1_coverage_report,
        save_mbp1_coverage_report,
    )
    from ..features.mbp1_feature_materializer import (  # noqa: PLC0415
        materialize_mbp1_features,
        save_mbp1_feature_artifact,
    )
    from ..features.mbp1_source_artifact import (  # noqa: PLC0415
        assert_evidence_provenance_permitted,
        save_mbp1_source_artifact,
    )

    assert context.wiring.mbp1_evidence_source is not None
    source_envelope, events_by_day, anchors = context.wiring.mbp1_evidence_source()
    # R5B.1: synthetic coverage evidence is lawful only under the synthetic
    # marker — a real scope refuses it before the artifact is trusted
    assert_evidence_provenance_permitted(
        source_envelope.payload.ordered_partitions, synthetic_scope=context.synthetic
    )
    resolved_block = FEATURE_BLOCK_RESOLUTION_REGISTRY["IFVG_ORDER_FLOW_MBP1_V1"]
    feature_envelope, feature_frame, evidence_frame = materialize_mbp1_features(
        source_envelope,
        anchors,
        resolved_block=resolved_block,
        events_by_day=events_by_day,
    )
    coverage_envelope = build_mbp1_coverage_report(
        source_envelope, feature_envelope, feature_frame, evidence_frame
    )
    save_mbp1_source_artifact(
        context.store_root,
        source_envelope,
        {day: _canonical_day_bytes(events_by_day[day]) for day in events_by_day}
        if source_envelope.events_stored
        else {},
    )
    save_mbp1_feature_artifact(
        context.store_root, feature_envelope, feature_frame, evidence_frame
    )
    save_mbp1_coverage_report(context.store_root, coverage_envelope)
    context.mbp1_evidence = {
        "source_id": source_envelope.mbp1_source_artifact_id,
        "feature_artifact_id": feature_envelope.mbp1_feature_artifact_id,
        "feature_envelope": feature_envelope,
        "coverage_report_id": coverage_envelope.mbp1_coverage_report_id,
        "feature_frame": feature_frame,
        "resolved_block_id": resolved_block.resolved_feature_block_id,
    }
    return context.mbp1_evidence


def _canonical_day_bytes(frame) -> bytes:
    from ..features.mbp1_source_artifact import _canonical_event_bytes  # noqa: PLC0415

    return _canonical_event_bytes(frame)


def _stage_s05_feature_views(context: _RunContext) -> tuple[tuple[str, ...], str]:
    from ..features.bundle_feature_view import (  # noqa: PLC0415
        mbp1_block_keys_in_bundle,
        resolve_available_bundle_view,
    )
    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415

    assert context.wiring.candidate_view_source is not None
    context.view = context.wiring.candidate_view_source()
    outputs: list[str] = []
    dumped: dict[str, Any] = {}
    mbp1_note = ""
    regime_request = _regime.regime_request(context.semantic.payload)
    for bundle_key in context.semantic.payload.feature_bundle_ids:
        if (
            regime_request is not None
            and regime_request.is_panel
            and bundle_key == regime_request.input_feature_bundle_key
        ):
            # the panel bundle is a PANEL artifact, never a candidate view
            continue
        if mbp1_block_keys_in_bundle(resolve_bundle(bundle_key)):
            evidence = _ensure_mbp1_evidence(context)
            envelope, frame = resolve_available_bundle_view(
                context.view,
                bundle_key,
                mbp1_features=evidence["feature_frame"],
                mbp1_feature_artifact=evidence["feature_envelope"],
            )
            for artifact_id in (
                evidence["source_id"],
                evidence["feature_artifact_id"],
                evidence["coverage_report_id"],
            ):
                if artifact_id not in outputs:
                    outputs.append(artifact_id)
            mbp1_note = (
                "; MBP-1 evidence materialized offline (research-only, "
                "owner decision R-6) and joined one-to-one with typed nulls"
            )
        else:
            envelope, frame = resolve_available_bundle_view(context.view, bundle_key)
        context.bundle_views[bundle_key] = envelope
        context.bundle_frames[bundle_key] = frame
        outputs.append(envelope.bundle_feature_view_id)
        dumped[bundle_key] = envelope.model_dump(mode="json")
    if context.mbp1_evidence:
        dumped["__mbp1_evidence__"] = {
            key: context.mbp1_evidence[key]
            for key in ("source_id", "feature_artifact_id", "coverage_report_id")
        }
    regime_outputs, regime_record, regime_note = _regime.s05_regime_observation(context)
    for artifact_id in regime_outputs:
        if artifact_id not in outputs:
            outputs.append(artifact_id)
    dumped.update(regime_record)
    context.stage_sidecars["bundle_feature_views.json"] = (
        json.dumps(dumped, sort_keys=True) + "\n"
    ).encode("utf-8")
    return tuple(outputs), (
        f"{len(context.bundle_views)} bundle feature views materialized over "
        f"the immutable candidate view (available blocks only){mbp1_note}{regime_note}"
    )


def _stage_s06_coverage(context: _RunContext) -> tuple[tuple[str, ...], str]:
    report: dict[str, Any] = {}
    for bundle_key, envelope in context.bundle_views.items():
        frame = context.bundle_frames[bundle_key]
        names = envelope.payload.resolved_feature_names
        coverage = {
            name: (float(frame[name].notna().mean()) if len(frame) else 0.0)
            for name in names
        }
        report[bundle_key] = {
            "rows": int(len(frame)),
            "minimum_feature_coverage": min(coverage.values()) if coverage else None,
            "per_feature_nonnull_fraction": coverage,
        }
    context.stage_sidecars["feature_coverage.json"] = (
        json.dumps(report, sort_keys=True) + "\n"
    ).encode("utf-8")
    regime_note = ""
    regime_preview = _regime.s06_regime_coverage(context)
    if regime_preview is not None:
        context.stage_sidecars["regime_sample_adequacy_preview.json"] = (
            _regime.canonical_json_bytes(regime_preview)
        )
        regime_note = (
            f"; regime inputs: {regime_preview['rows_with_inputs']}/"
            f"{regime_preview['rows_total']} rows carry an input (floor "
            f"{regime_preview['floors']['minimum_training_observations']} training rows)"
        )
    return (), (
        f"feature coverage evaluated for {len(report)} bundle view(s); "
        f"full report in the stage sidecar{regime_note}"
    )


def _stage_s07_labels(context: _RunContext) -> tuple[tuple[str, ...], str]:
    assert context.wiring.label_builder is not None
    if context.view is None and context.wiring.candidate_view_source is not None:
        context.view = context.wiring.candidate_view_source()
    label_policy_id, labeled = context.wiring.label_builder(context.view)
    if label_policy_id != context.semantic.payload.label_policy_id:
        raise ValueError(
            f"label builder produced policy {label_policy_id!r} but the "
            f"pipeline pins {context.semantic.payload.label_policy_id!r}"
        )
    context.labeled = labeled
    # adversarial M-1: the id hashes SORTED (candidate_id, target) PAIRS —
    # binding candidate to target (two different labelings can never share
    # an id) and order-invariantly (a row reorder can never fork it)
    context.label_artifact_id = canonical_contract_sha256(
        {
            "label_policy_id": label_policy_id,
            "labeled_pairs": sorted(
                (
                    str(candidate_id),
                    None if _safe_isna(target) else int(target),
                )
                for candidate_id, target in zip(
                    labeled["candidate_id"], labeled["binary_target"], strict=True
                )
            ),
        }
    )
    return (context.label_artifact_id,), (
        f"labels derived for {len(labeled)} candidates under {label_policy_id}"
    )


def _stage_s08_folds(context: _RunContext) -> tuple[tuple[str, ...], str]:
    from ..context_folds import build_context_folds  # noqa: PLC0415

    regime_request = _regime.regime_request(context.semantic.payload)
    if context.labeled is None and regime_request is None:
        raise ValueError("fold construction requires the derived labels (run 07)")
    if context.labeled is None:
        # R6.1 (D14): a descriptive regime study without derived labels builds
        # its candidate folds label-free under the same frozen schedule
        context.folds = None
        regime_outputs, regime_record, regime_note = _regime.s08_regime_folds(context)
        context.stage_sidecars["fold_sample_adequacy.json"] = _regime.canonical_json_bytes(
            regime_record
        )
        return regime_outputs, (
            f"label-free candidate folds under {FOLD_PROTOCOL_ID_V1}{regime_note}"
        )
    # R6.1 (adversarial F7): ONE day source when a regime study is planned —
    # the candidate view's observed trading days (charter allowlist ∩
    # observed) drive BOTH the labeled candidate folds and the persisted
    # fold schedule, so a label/view day divergence is a typed fold fact
    # (a thinner fold), never a schedule/window mismatch at S08
    days = (
        tuple(sorted(set(context.view.frame["trading_day"].astype(str))))
        if regime_request is not None
        else tuple(sorted(set(context.labeled["trading_day"].astype(str))))
    )
    context.folds = build_context_folds(context.labeled, authorized_trading_days=days)
    valid = sum(1 for fold in context.folds.folds if fold.valid)
    if not context.folds.folds:
        explanation = (
            f"0 folds constructible under the frozen {FOLD_PROTOCOL_ID_V1} "
            f"protocol on {len(days)} trading days — a legitimate safe-failure "
            "state on short (verification) windows"
        )
    else:
        invalid_reasons = sorted(
            {
                fold.invalid_reason
                for fold in context.folds.folds
                if fold.invalid_reason is not None
            }
        )
        explanation = (
            f"{len(context.folds.folds)} folds under {FOLD_PROTOCOL_ID_V1}; "
            f"{valid} valid"
        )
        if invalid_reasons:
            explanation += f"; invalid reasons: {', '.join(invalid_reasons)}"
    if regime_request is None:
        return (), explanation
    regime_outputs, regime_record, regime_note = _regime.s08_regime_folds(context)
    context.stage_sidecars["fold_sample_adequacy.json"] = _regime.canonical_json_bytes(
        regime_record
    )
    return regime_outputs, explanation + regime_note


def _stage_s09_train(context: _RunContext) -> tuple[tuple[str, ...], str]:
    """S09 = the supervised ladder (when a model protocol is pinned) + the
    regime study's S09a (and S09b/S09c for model-bearing requests) — every
    fit of the run happens here (D14; S14 performs zero fitting)."""

    spec = context.semantic.payload
    regime_request = _regime.regime_request(spec)
    outputs: list[str] = []
    if spec.model_protocol_id is not None:
        ladder_outputs, explanation = _stage_s09_supervised_ladder(context)
        outputs.extend(ladder_outputs)
    elif regime_request is None:
        raise ValueError("training requires a pinned model protocol or a regime study")
    else:
        explanation = "no supervised model protocol pinned (regime study only)"
    if regime_request is not None:
        regime_outputs, regime_record, regime_note = _regime.s09_regime_fit(context)
        for artifact_id in regime_outputs:
            if artifact_id not in outputs:
                outputs.append(artifact_id)
        context.stage_sidecars["regime_run.json"] = _regime.canonical_json_bytes(regime_record)
        explanation += regime_note
    return tuple(outputs), explanation


def _stage_s09_supervised_ladder(context: _RunContext) -> tuple[tuple[str, ...], str]:
    from ..features.bundle_feature_view import frozen_tier_for_bundle  # noqa: PLC0415
    from ..ml.supervised_ladder import run_supervised_ladder  # noqa: PLC0415

    if context.folds is None or context.labeled is None:
        raise ValueError("training requires labels and folds (run 07/08)")
    primary_bundle = context.semantic.payload.feature_bundle_ids[0]
    envelope = context.bundle_views[primary_bundle]
    if envelope.payload.mbp1_feature_artifact_id is not None:
        return _run_controlled_mbp1_stage(context, primary_bundle)
    tier = frozen_tier_for_bundle(envelope.payload.resolved_feature_names)
    if tier is None:
        raise ValueError(
            f"bundle {primary_bundle} does not resolve to a frozen tier "
            "feature set and carries no MBP-1 evidence; no ladder wiring "
            "exists for it (fail closed)"
        )
    schedule = context.regime.get("schedule")
    context.ladder = run_supervised_ladder(
        context.view,
        context.labeled,
        context.folds,
        tier=tier,
        # D13: the exact S08 schedule id and S07 label artifact id key the
        # comparison rows when a regime study persisted them
        fold_schedule_id=schedule.fold_schedule_id if schedule is not None else None,
        label_artifact_id=context.label_artifact_id,
    )
    oos_rows = context.ladder.parity["oos_row_count"]
    # R5-FIX finding 5: zero OOS rows means the parity claim is NOT
    # evaluable — saying "held over 0 rows" would overstate the evidence
    parity_clause = (
        f"identical-rows parity held over {oos_rows} OOS rows"
        if oos_rows
        else "identical-rows parity not evaluable (0 OOS rows)"
    )
    return (context.ladder.ladder_id,), (
        f"supervised ladder ran {len(context.ladder.rungs)} rungs on tier "
        f"{tier.value}; {parity_clause}"
    )


def _run_controlled_mbp1_stage(
    context: _RunContext, primary_bundle: str
) -> tuple[tuple[str, ...], str]:
    """S09's R5B path: the controlled Baseline vs Baseline+MBP-1 study.

    Both arms run bundle-parametrized — the prevalence reference, the
    logistic protocol, and (R6.1) the bundle-aware CatBoost rung; the
    frozen-lane CatBoost fold runner stays tier-locked — on identical rows,
    labels, and folds keyed by the D13 ``comparison_row_id``; the pinned
    protocol is the headline comparison; the persisted study envelope
    carries the paired Brier deltas and the ``research_only_offline`` boundary.
    """

    from ..ml.controlled_feature_study import (  # noqa: PLC0415
        HEADLINE_PROTOCOL_IDS,
        run_controlled_mbp1_study,
        save_controlled_feature_study,
    )

    pinned = context.semantic.payload.model_protocol_id
    if pinned not in HEADLINE_PROTOCOL_IDS:
        raise ValueError(
            "the controlled MBP-1 study runs the prevalence reference, "
            f"{HEADLINE_PROTOCOL_IDS[0]!r}, and the bundle-aware "
            f"{HEADLINE_PROTOCOL_IDS[1]!r} on both arms; the pinned protocol "
            f"{pinned!r} has no bundle-parametrized wiring (the "
            "ifvg_context_catboost_binary_v1 fold runner is tier-locked in the "
            "frozen M0-M3 lane)"
        )
    evidence = context.mbp1_evidence
    if not evidence:
        raise ValueError("MBP-1 evidence was not materialized (run stage 05)")
    study = run_controlled_mbp1_study(
        context.view,
        context.labeled,
        context.folds,
        challenger_bundle_key=primary_bundle,
        mbp1_features=evidence["feature_frame"],
        mbp1_feature_artifact=evidence["feature_envelope"],
        headline_protocol_id=pinned,
        # D13: S07's exact label artifact id keys the comparison rows
        label_artifact_id=context.label_artifact_id,
        fold_schedule_id=(
            context.regime["schedule"].fold_schedule_id
            if context.regime.get("schedule") is not None
            else None
        ),
    )
    save_controlled_feature_study(context.store_root, study)
    context.controlled_study = study
    context.ladder = study.challenger
    oos_rows = study.challenger.parity["oos_row_count"]
    parity_clause = (
        f"identical-rows parity held over {oos_rows} OOS rows in both arms"
        if oos_rows
        else "identical-rows parity not evaluable (0 OOS rows; the legitimate "
        "safe-failure shape on verification windows)"
    )
    return (
        (
            study.envelope.controlled_feature_study_id,
            study.baseline.ladder_id,
            study.challenger.ladder_id,
        ),
        (
            f"controlled Baseline vs Baseline+MBP-1 study over "
            f"{study.envelope.payload.baseline_bundle_key} → {primary_bundle} "
            "(research-only offline; prevalence + logistic + bundle-aware CatBoost "
            f"rungs on identical comparison rows; headline {pinned}); {parity_clause}"
        ),
    )


def _stage_s10_diagnostics(context: _RunContext) -> tuple[tuple[str, ...], str]:
    regime_request = _regime.regime_request(context.semantic.payload)
    outputs: list[str] = []
    if context.ladder is not None:
        ladder_outputs, explanation = _stage_s10_ladder_diagnostics(context)
        outputs.extend(ladder_outputs)
    elif regime_request is None:
        raise ValueError("diagnostics require the trained ladder (run 09)")
    else:
        explanation = "no supervised ladder (regime study only)"
    if regime_request is not None:
        regime_outputs, diagnostics, regime_note = _regime.s10_regime_diagnostics(context)
        for artifact_id in regime_outputs:
            if artifact_id not in outputs:
                outputs.append(artifact_id)
        context.stage_sidecars["regime_diagnostics.json"] = _regime.canonical_json_bytes(
            diagnostics
        )
        explanation += regime_note
    return tuple(outputs), explanation


def _stage_s10_ladder_diagnostics(context: _RunContext) -> tuple[tuple[str, ...], str]:
    if context.ladder is None:
        raise ValueError("diagnostics require the trained ladder (run 09)")
    ladder = context.ladder
    diagnostics = {
        "ladder_id": ladder.ladder_id,
        "view_id": ladder.view_id,
        "tier": ladder.tier,
        "feature_source": ladder.feature_source,
        "calibration_policy_id": ladder.calibration_policy_id,
        "parity": ladder.parity,
        "rungs": {
            rung.protocol_id: {
                "resolved_protocol_hash": rung.resolved_protocol_hash,
                "prediction_report": rung.prediction_report,
                "fold_reports": list(rung.fold_reports),
            }
            for rung in ladder.rungs
        },
        "paired_deltas": ladder.paired_deltas,
    }
    if context.controlled_study is not None:
        diagnostics["controlled_feature_study"] = context.controlled_study.envelope.model_dump(
            mode="json"
        )
    context.stage_sidecars["supervised_ladder.json"] = (
        json.dumps(diagnostics, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    return (ladder.ladder_id,), (
        f"per-rung prediction reports and paired deltas persisted "
        f"({ladder.parity['oos_row_count']} OOS rows; empty-fold reports are "
        "the legitimate safe-failure shape on verification windows)"
    )


def _historical_modes(protocol: SimulationProtocol) -> tuple[str, ...]:
    return tuple(
        mode
        for mode in protocol.modes
        if mode in ("historical_closed_trade", "historical_1m_scenario")
    )


def _resample_modes(protocol: SimulationProtocol) -> tuple[str, ...]:
    return tuple(
        mode for mode in protocol.modes if mode in ("day_block_bootstrap", "stress")
    )


@dataclass
class _PropSubject:
    core_replay_id: str


def _prior_stage_vectors(
    context: _RunContext, stage: QuantLabPipelineStage
) -> dict[str, dict[str, Any]]:
    """Verified prop vectors from a PRIOR attempt's stage result sidecar."""

    from alpha_lab.propsim.prop_metrics import PayoutReliabilityVector  # noqa: PLC0415

    entry = context.state["stages"][stage.value]
    stage_result_id = entry.get("stage_result_id")
    if not stage_result_id or not has_envelope(
        context.store_root, "pipeline_stage_results", stage_result_id
    ):
        return {}
    try:
        raw = load_sidecar_bytes(
            context.store_root,
            "pipeline_stage_results",
            stage_result_id,
            _PROP_VECTORS_SIDECAR,
        )
    except Exception:  # noqa: BLE001 — a prior attempt without prop vectors
        return {}
    decoded = json.loads(raw.decode("utf-8"))
    return {
        core_id: {
            label: PayoutReliabilityVector.model_validate(vector)
            for label, vector in per_label.items()
        }
        for core_id, per_label in decoded.items()
    }


def _prior_stage_simulations(
    context: _RunContext, stage: QuantLabPipelineStage
) -> dict[str, dict[str, list[str]]]:
    """The account-simulation ids a PRIOR attempt's stage result recorded."""

    entry = context.state["stages"][stage.value]
    stage_result_id = entry.get("stage_result_id")
    if not stage_result_id or not has_envelope(
        context.store_root, "pipeline_stage_results", stage_result_id
    ):
        return {}
    try:
        raw = load_sidecar_bytes(
            context.store_root,
            "pipeline_stage_results",
            stage_result_id,
            _ACCOUNT_SIMULATIONS_SIDECAR,
        )
    except Exception:  # noqa: BLE001 — a prior attempt without the sidecar
        return {}
    return json.loads(raw.decode("utf-8"))


def _run_prop_modes(
    context: _RunContext,
    modes: tuple[str, ...],
    *,
    n_paths: int,
    stage: QuantLabPipelineStage,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    from alpha_lab.propsim.search_bridge import make_prop_simulator  # noqa: PLC0415

    protocol = context.charter.payload.simulation_protocol
    charter_payload = context.charter.payload
    simulations_sink: dict[str, dict[str, list[str]]] = {}
    prior_simulations = _prior_stage_simulations(context, stage)
    simulator = make_prop_simulator(
        context.wiring.firm_specs,
        event_detail_persistence_policy_id=protocol.event_detail_persistence_policy_id,
        tick_size=charter_payload.cost_policy.tick_size,
        costed_evaluation_id_for=lambda core_id: _child_evaluation_envelope(
            core_id, charter_payload.cost_policy
        ).costed_evaluation_id,
        cost_points_round_turn=charter_payload.cost_policy.cost_points_round_turn,
        seed=protocol.bootstrap_seed,
        simulation_modes=modes,
        intrabar_scenario_policy_id=(
            "bar_adverse_extreme_first_v1"
            if "historical_1m_scenario" in modes
            else None
        ),
        bootstrap_protocol_id=(
            BOOTSTRAP_PROTOCOL_ID_DEFAULT if "day_block_bootstrap" in modes else None
        ),
        stress_scenario_ids=(protocol.stress_scenario_ids if "stress" in modes else ()),
        n_paths=n_paths,
        bar_observations_for=context.wiring.bar_observations_for,
        store_root=context.store_root,
        on_simulation_persisted=_record_account_simulation(simulations_sink),
    )
    prior_vectors = _prior_stage_vectors(context, stage)
    vectors_out: dict[str, dict[str, Any]] = {}
    simulated = 0
    reused = 0
    rows_by_id = {row["core_replay_id"]: row for row in context.children}
    for core_replay_id in sorted(context.gates_passed):
        result = context.tables_by_child.get(core_replay_id)
        row = rows_by_id[core_replay_id]
        if result is None:
            prior = prior_vectors.get(core_replay_id)
            if prior:
                vectors_out[core_replay_id] = prior
                if core_replay_id in prior_simulations:
                    simulations_sink[core_replay_id] = dict(prior_simulations[core_replay_id])
                reused += 1
            else:
                row["explanation"] = (
                    "prop simulation skipped: this child was REUSED without "
                    "rebuilt tables and no prior attempt's verified vectors "
                    "exist; rerun the replay stage to simulate it"
                )
            continue
        try:
            vectors = simulator(
                outcome=_PropSubject(core_replay_id=core_replay_id), result=result
            )
        except Exception as error:  # noqa: BLE001 — per-child containment
            row["failure_reason"] = FailureReason.REPLAY.value
            row["explanation"] = "prop simulation failed: " + sanitize_failure_message(
                str(error)
            )
            context.gates_passed.pop(core_replay_id, None)
            continue
        vectors_out[core_replay_id] = vectors
        simulated += 1
    context.stage_sidecars[_PROP_VECTORS_SIDECAR] = (
        json.dumps(
            {
                core_id: {
                    label: vector.model_dump(mode="json")
                    for label, vector in per_label.items()
                }
                for core_id, per_label in vectors_out.items()
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    context.stage_sidecars[_ACCOUNT_SIMULATIONS_SIDECAR] = (
        json.dumps(simulations_sink, sort_keys=True) + "\n"
    ).encode("utf-8")
    by_child = context.regime.setdefault("account_simulations", {})
    for core_id, simulations in simulations_sink.items():
        by_child.setdefault(core_id, {}).update(
            {simulation_id: tuple(value) for simulation_id, value in simulations.items()}
        )
    return vectors_out, simulated, reused


def _record_account_simulation(sink: dict[str, dict[str, list[str]]]):
    """R6.1: record ``core_replay_id -> {account_simulation_id: [firm, mode]}``
    for the stratified-prop reports (exact persisted ids, never a store scan)."""

    def _hook(core_replay_id: str, simulation_id: str, firm_label: str, mode: str) -> None:
        sink.setdefault(core_replay_id, {})[simulation_id] = [firm_label, mode]

    return _hook


def _persist_policy_set_envelopes(context: _RunContext) -> tuple[str, ...]:
    """Unconditional S12 writer (adversarial m-5): the stage's outputs name
    these envelopes, so they persist even when zero children simulate."""

    policy_ids = []
    for firm_spec in context.wiring.firm_specs:
        envelope = firm_spec.policy_set_envelope()
        save_or_reuse_envelope(context.store_root, "account_policy_sets", envelope)
        policy_ids.append(envelope.account_policy_set_id)
    return tuple(sorted(policy_ids))


def _stage_s12_prop_historical(context: _RunContext) -> tuple[tuple[str, ...], str]:
    modes = _historical_modes(context.charter.payload.simulation_protocol)
    if not modes:
        return (), "no historical prop modes in the simulation protocol"
    policy_ids = _persist_policy_set_envelopes(context)
    vectors, simulated, reused = _run_prop_modes(
        context,
        modes,
        n_paths=1,
        stage=QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS,
    )
    for core_id, per_label in vectors.items():
        context.prop_vectors.setdefault(core_id, {}).update(per_label)
    note = f"historical prop simulations ({', '.join(modes)}) ran for {simulated}"
    if reused:
        note += f" (+{reused} reused from a prior verified attempt)"
    note += (
        " gates-passing children; resolved AccountPolicySetEnvelopes "
        "persisted to the account_policy_sets store"
    )
    return policy_ids, note


def _stage_s13_bootstrap_stress(context: _RunContext) -> tuple[tuple[str, ...], str]:
    protocol = context.charter.payload.simulation_protocol
    modes = _resample_modes(protocol)
    if not modes:
        return (), "no bootstrap/stress modes in the simulation protocol"
    vectors, simulated, reused = _run_prop_modes(
        context,
        modes,
        n_paths=protocol.bootstrap_n_paths,
        stage=QuantLabPipelineStage.S13_RUN_BOOTSTRAP_AND_STRESS,
    )
    for core_id, per_label in vectors.items():
        context.prop_vectors.setdefault(core_id, {}).update(per_label)
    note = (
        f"resampled simulations ({', '.join(modes)}; {protocol.bootstrap_n_paths} "
        f"paths, seed {protocol.bootstrap_seed}, {BOOTSTRAP_PROTOCOL_ID_DEFAULT}) "
        f"ran for {simulated} children"
    )
    if reused:
        note += f" (+{reused} reused from a prior verified attempt)"
    return (), note


def _stage_s14_frontier_insights(context: _RunContext) -> tuple[tuple[str, ...], str]:
    charter_payload = context.charter.payload
    prop_planned = (
        QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS
        in set(context.semantic.payload.stage_plan)
    )
    feasible: dict[str, dict[str, float]] = {}
    rows_by_id = {row["core_replay_id"]: row for row in context.children}
    for core_replay_id, strategy_objectives in sorted(context.gates_passed.items()):
        row = rows_by_id[core_replay_id]
        if prop_planned:
            merge = merge_prop_vectors(
                context.prop_vectors.get(core_replay_id, {}),
                prop_thresholds=charter_payload.objective_policy.prop_feasibility_gates,
                pareto_objectives=charter_payload.objective_policy.pareto_objectives,
                strategy_objectives=strategy_objectives,
            )
            if merge.verdict != "ok":
                if merge.failure_reason is not None:
                    row["failure_reason"] = merge.failure_reason.value
                row["explanation"] = merge.explanation or ""
                continue
            feasible[core_replay_id] = dict(merge.merged_objectives or {})
        else:
            missing = [
                metric
                for metric in charter_payload.objective_policy.pareto_objectives
                if metric not in strategy_objectives
            ]
            if missing:
                row["explanation"] = (
                    "passed all strategy gates but objective metric(s) "
                    f"{missing} are unavailable; excluded from the frontier"
                )
                continue
            feasible[core_replay_id] = dict(strategy_objectives)
    _record_children(context)
    outputs: list[str] = []
    explanation_parts: list[str] = []
    if feasible:
        objectives = tuple(
            ObjectiveSpec(metric=metric, direction=OBJECTIVE_DIRECTIONS[metric])
            for metric in charter_payload.objective_policy.pareto_objectives
        )
        frontier = build_frontier(
            feasible,
            objectives=objectives,
            lexicographic_tie_breaks=(
                charter_payload.objective_policy.lexicographic_tie_breaks
            ),
        )
        frontier_envelope, _ = save_or_reuse_envelope(
            context.store_root,
            "frontiers",
            SearchFrontierEnvelope.from_payload(
                SearchFrontierPayload(
                    search_id=context.charter.search_id, frontier=frontier
                )
            ),
        )
        context.frontier_id = frontier_envelope.frontier_id
        outputs.append(frontier_envelope.frontier_id)
        explanation_parts.append(
            f"frontier persisted over {len(feasible)} feasible children"
        )
        insight_ids = _persist_insight_panels(context, feasible, rows_by_id)
        context.insight_ids = insight_ids
        outputs.extend(insight_ids)
        explanation_parts.append(f"{len(insight_ids)} insight panels persisted")
    else:
        explanation_parts.append(
            "no feasible children; no frontier or insights to persist"
        )
    explanation_parts.append(_persist_cross_profile_deltas(context))
    outputs.extend(context.comparison_result_ids)
    regime_outputs, regime_record, regime_note = _regime.s14_regime_reports(context)
    if regime_record:
        context.stage_sidecars["regime_stratified_reports.json"] = (
            _regime.canonical_json_bytes(regime_record)
        )
        outputs.extend(artifact_id for artifact_id in regime_outputs if artifact_id not in outputs)
        explanation_parts.append(regime_note)
    return tuple(outputs), "; ".join(explanation_parts)


def _persist_insight_panels(
    context: _RunContext,
    feasible: Mapping[str, Mapping[str, float]],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Deterministic per-child insight panels into the insights store (S14)."""

    baseline_id = next(
        (
            row["core_replay_id"]
            for row in context.children
            if row["comparison_role"] == "baseline"
        ),
        None,
    )
    baseline_metrics = context.metrics_by_child.get(baseline_id) if baseline_id else None
    insight_ids: list[str] = []
    for core_replay_id in sorted(feasible):
        metrics = context.metrics_by_child.get(core_replay_id)
        if metrics is None:
            continue
        row = rows_by_id[core_replay_id]
        changed_axes = tuple(
            sorted(
                key
                for key, value_id in row["axis_value_ids"].items()
                if baseline_id is not None
                and rows_by_id[baseline_id]["axis_value_ids"].get(key) != value_id
            )
        )
        panel = render_insight_panel(
            changed_axis_labels=changed_axes,
            child_id=core_replay_id,
            metrics=metrics,
            baseline_metrics=(
                baseline_metrics if core_replay_id != baseline_id else None
            ),
        )
        envelope = InsightPanelEnvelope.from_payload(
            InsightPanelPayload(
                search_id=context.charter.search_id,
                subject_core_replay_id=core_replay_id,
                panel=panel,
            )
        )
        save_or_reuse_envelope(context.store_root, "insights", envelope)
        insight_ids.append(envelope.insight_panel_id)
    return tuple(insight_ids)


def _persist_cross_profile_deltas(context: _RunContext) -> str:
    """DEV-R4-16 closure: build + persist the baseline↔challenger comparison
    results from the S02 lineage evidence (this attempt's, or a prior
    verified attempt's stage sidecars)."""

    from ..study.comparison_contracts import (  # noqa: PLC0415
        ComparisonCompatibility,
        ComparisonResult,
        ComparisonResultEnvelope,
        SearchDerivationComparisonSubject,
    )
    from ..study_providers import prepare_cross_profile_deltas  # noqa: PLC0415

    maps: dict[str, Any] = {
        core_id: deserialize_native_lineage_map(payload)
        for core_id, payload in context.lineage_sidecars.items()
    }
    if not maps:
        entry = context.state["stages"][
            QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value
        ]
        stage_result_id = entry.get("stage_result_id")
        if stage_result_id and has_envelope(
            context.store_root, "pipeline_stage_results", stage_result_id
        ):
            for row in context.children:
                with suppress(Exception):
                    raw = load_sidecar_bytes(
                        context.store_root,
                        "pipeline_stage_results",
                        stage_result_id,
                        f"lineage_map_{row['core_replay_id']}.json",
                    )
                    maps[row["core_replay_id"]] = deserialize_native_lineage_map(
                        json.loads(raw.decode("utf-8"))
                    )
    baseline_row = next(
        (row for row in context.children if row["comparison_role"] == "baseline"),
        None,
    )
    if baseline_row is None or baseline_row["core_replay_id"] not in maps:
        return "cross-profile deltas skipped: no baseline lineage evidence"
    baseline_map = maps[baseline_row["core_replay_id"]]
    persisted = 0
    for row in context.children:
        if row["comparison_role"] == "baseline":
            continue
        challenger_map = maps.get(row["core_replay_id"])
        if challenger_map is None:
            continue
        changed_axes = tuple(
            sorted(
                key
                for key, value_id in row["axis_value_ids"].items()
                if baseline_row["axis_value_ids"].get(key) != value_id
            )
        )
        deltas = prepare_cross_profile_deltas(
            context.store_root,
            baseline_map=baseline_map,
            challenger_map=challenger_map,
            changed_axis_keys=changed_axes,
        )
        result = ComparisonResult(
            # R5-FIX finding 6: the search lane's reference is TYPED as a
            # derivation, never as a study-cell ComparisonEnvelope id
            subject=SearchDerivationComparisonSubject(
                derivation_id=canonical_contract_sha256(
                    {
                        "kind": "cross_profile_population_delta_v1",
                        "search_id": context.charter.search_id,
                        "baseline_core_replay_id": baseline_row["core_replay_id"],
                        "challenger_core_replay_id": row["core_replay_id"],
                        "changed_axis_keys": list(changed_axes),
                    }
                )
            ),
            compatibility=ComparisonCompatibility(
                per_dimension_match={
                    f"strategy_profile.{axis}": False for axis in changed_axes
                },
                required_equalities_satisfied=deltas.lineage_valid,
                registered_single_axis_delta_only=len(changed_axes) == 1,
                compatible_for_metric_delta=deltas.lineage_valid,
                compatibility_status=(
                    "compatible" if deltas.lineage_valid else "config_diff_only"
                ),
                incompatibility_reasons=(
                    ()
                    if deltas.lineage_valid
                    else (deltas.lineage_invalid_reason or "lineage_invalid",)
                ),
                differing_fields=tuple(
                    f"strategy_profile.{axis}" for axis in changed_axes
                ),
            ),
            delta_reports={
                kind: report.model_dump(mode="json")
                for kind, report in deltas.reports.items()
            },
            config_diff=None,
            evidence_links=(
                deltas.baseline_lineage_report_id,
                deltas.challenger_lineage_report_id,
            ),
        )
        envelope = ComparisonResultEnvelope.from_payload(result)
        save_or_reuse_envelope(context.store_root, "search_results", envelope)
        context.comparison_result_ids = (
            *context.comparison_result_ids,
            envelope.comparison_result_id,
        )
        persisted += 1
    if persisted == 0:
        return "cross-profile deltas skipped: no challenger lineage evidence"
    return f"{persisted} cross-profile comparison result(s) persisted"


def _stage_s15_verify_publish(context: _RunContext) -> tuple[tuple[str, ...], str]:
    stages = context.state["stages"]
    replay_entry = stages[QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value]
    terminal_children = all(
        row["state"] in ("completed", "reused", "blocked", "failed")
        for row in context.children
    )
    completed_children = [
        row for row in context.children if row["state"] in ("completed", "reused")
    ]
    reload_ok = True
    if context.frontier_id is not None:
        try:
            load_verified_envelope(
                context.store_root,
                "frontiers",
                context.frontier_id,
                SearchFrontierEnvelope,
            )
        except Exception:  # noqa: BLE001 — the gate result is the evidence
            reload_ok = False
    for insight_id in context.insight_ids:
        try:
            load_verified_envelope(
                context.store_root, "insights", insight_id, InsightPanelEnvelope
            )
        except Exception:  # noqa: BLE001
            reload_ok = False
    # R6.1: every regime artifact of the run must reload through the stores
    reload_ok = reload_ok and _regime.s15_regime_reload_ok(context)
    chart_entry = stages[QuantLabPipelineStage.S04_BUILD_OR_REUSE_REPLAY_CHARTS.value]
    verifier_link = bool(chart_entry["in_plan"]) and bool(
        chart_entry["output_artifact_ids"]
    )
    gates = evaluate_control_flow_gates(
        {
            "replay_completed": replay_entry["status"]
            in (StageStatus.COMPLETED.value, StageStatus.REUSED.value)
            and terminal_children
            and bool(completed_children),
            "invariants_passed": all(
                context.neutrality_by_child.get(row["core_replay_id"], True)
                for row in completed_children
            ),
            "artifacts_published_and_reloaded": reload_ok,
            "neutrality_passed": (
                all(context.neutrality_by_child.values())
                if context.neutrality_by_child
                else bool(completed_children)
            ),
            "verifier_link_resolves": verifier_link,
            # DERIVED (safety review F1): every drive asserts its own zero
            # counters before returning; a tripped assertion is recorded at
            # the per-child containment and fails this gate. On the
            # synthetic branch no real source path is constructible, so an
            # untripped run is zero by construction.
            "zero_forbidden_counters": not context.forbidden_access_detected,
        }
    )
    stamps = None
    if context.semantic.payload.run_scope is PipelineRunScope.VERIFICATION_5D:
        # safety review F2: synthetic fixture days are NEVER counted as
        # real dates — the real_date fields describe real-data consumption
        stamps = dict(
            verification_report_stamps(
                allowlist=(
                    ()
                    if context.synthetic
                    else context.semantic.payload.date_allowlist
                ),
                synthetic_fixture_ids=(
                    (context.charter.search_id,) if context.synthetic else ()
                ),
            )
        )
        if context.synthetic:
            stamps["synthetic_date_count"] = len(
                context.semantic.payload.date_allowlist
            )
    stage_result_ids = {
        stage_value: entry["stage_result_id"]
        for stage_value, entry in stages.items()
        if entry.get("stage_result_id")
        # S15's own id postdates this payload — including a prior attempt's
        # would make the result self-referential and drift across attempts
        and stage_value != QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH.value
    }
    result_envelope = PipelineResultEnvelope.from_payload(
        PipelineResultPayload(
            pipeline_semantic_id=context.semantic.pipeline_semantic_id,
            run_scope=context.semantic.payload.run_scope,
            stage_result_ids=stage_result_ids,
            frontier_id=context.frontier_id,
            control_flow_gates=gates,
            verification_stamps=stamps,
        )
    )
    save_or_reuse_envelope(context.store_root, "search_results", result_envelope)
    context.state["publication"] = {
        "state": "prepared_not_published",
        "gates": None,
        "activated": False,
        "pipeline_result_id": result_envelope.pipeline_result_id,
        "control_flow_gates_passed": gates.passed,
    }
    context.result_envelope = result_envelope
    return (result_envelope.pipeline_result_id,), (
        f"pipeline result persisted (control-flow gates "
        f"{'passed' if gates.passed else 'NOT passed'}); publication state is "
        "prepared_not_published — activation is a separate explicit action"
    )


_STAGE_EXECUTORS: Mapping[
    QuantLabPipelineStage, Callable[[_RunContext], tuple[tuple[str, ...], str]]
] = {
    QuantLabPipelineStage.S00_VALIDATE_INPUTS: _stage_s00_validate,
    QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES: _stage_s01_prepare,
    QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS: _stage_s02_replays,
    QuantLabPipelineStage.S03_BUILD_OR_REUSE_FSM_AUDIT: _stage_s03_audit,
    QuantLabPipelineStage.S04_BUILD_OR_REUSE_REPLAY_CHARTS: _stage_s04_charts,
    QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS: _stage_s05_feature_views,
    QuantLabPipelineStage.S06_VALIDATE_FEATURE_COVERAGE: _stage_s06_coverage,
    QuantLabPipelineStage.S07_DERIVE_LABELS: _stage_s07_labels,
    QuantLabPipelineStage.S08_BUILD_FOLDS: _stage_s08_folds,
    QuantLabPipelineStage.S09_TRAIN_MODELS: _stage_s09_train,
    QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS: _stage_s10_diagnostics,
    QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS: _stage_s12_prop_historical,
    QuantLabPipelineStage.S13_RUN_BOOTSTRAP_AND_STRESS: _stage_s13_bootstrap_stress,
    QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS: _stage_s14_frontier_insights,
    QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH: _stage_s15_verify_publish,
}


def run_pipeline(
    semantic: PipelineSemanticIdentity,
    charter: SearchCharterEnvelope,
    *,
    store_root: Path,
    state_root: Path,
    wiring: PipelineWiring,
    worker_policy: WorkerPolicy,
    operational_retry_reason: str | None = None,
    stale_lock_seconds: float = 86_400.0,
) -> PipelineRunResult:
    """Run (or resume) one frozen pipeline specification as a new attempt.

    Retrying the same semantic pipeline mints a NEW
    :class:`ExecutionAttemptIdentity` and reuses verified semantic outputs;
    changing any research-bearing field changes ``pipeline_semantic_id`` and
    therefore the state directory, the stage identities, and the stores —
    a resource change can never mint a new scientific result (P0-3).
    """

    store_root = Path(store_root)
    state_root = Path(state_root)
    pipeline_id = semantic.pipeline_semantic_id
    with _search_lock(
        state_root, pipeline_id, stale_lock_seconds=stale_lock_seconds
    ) as lock_path:
        state_dir = _state_dir(state_root, pipeline_id)
        state_dir.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.unlink(state_dir / _CANCEL_SENTINEL)
        prior = read_pipeline_state(state_root, pipeline_id)
        state = prior if prior is not None else _initial_state(semantic)
        if state["pipeline_semantic_id"] != pipeline_id:
            raise ValueError("state directory belongs to a different semantic id")
        attempt = ExecutionAttemptIdentity(
            pipeline_semantic_id=pipeline_id,
            worker_policy=worker_policy,
            attempt_number=len(state.get("attempts", ())) + 1,
            host_environment={"os": os.name},
            started_at=_now(),
            ended_at=None,
            operational_retry_reason=operational_retry_reason,
        )
        state.setdefault("attempts", []).append(attempt.model_dump(mode="json"))
        context = _RunContext(
            semantic=semantic,
            charter=charter,
            wiring=wiring,
            store_root=store_root,
            state_root=state_root,
            state=state,
        )
        _checkpoint(context, heartbeat=lock_path)
        sentinel = state_dir / _CANCEL_SENTINEL
        planned = [
            stage
            for stage in CANONICAL_STAGE_ORDER
            if stage in set(semantic.payload.stage_plan)
        ]
        halted = False
        for index, stage in enumerate(planned):
            entry = state["stages"][stage.value]
            if sentinel.exists():
                with suppress(OSError):
                    os.unlink(sentinel)
                for later in planned[index:]:
                    later_entry = state["stages"][later.value]
                    later_entry["status"] = StageStatus.CANCELLED_AT_SAFE_BOUNDARY.value
                    later_entry["explanation"] = (
                        "safe cancel honored at the stage boundary"
                    )
                halted = True
                _checkpoint(context, heartbeat=lock_path)
                break
            if stage is QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS:
                entry["status"] = StageStatus.BLOCKED.value
                entry["explanation"] = S11_BLOCKED_REASON
                entry["started_at"] = entry["started_at"] or _now()
                entry["ended_at"] = _now()
                state["current_stage"] = stage.value
                _checkpoint(context, heartbeat=lock_path)
                continue
            prior_result_id = entry.get("stage_result_id")
            entry["status"] = StageStatus.RUNNING.value
            entry["started_at"] = _now()
            state["current_stage"] = stage.value
            _checkpoint(context, heartbeat=lock_path)
            context.stage_sidecars = {}
            try:
                outputs, explanation = _STAGE_EXECUTORS[stage](context)
            except Exception as error:  # noqa: BLE001 — sanitized stage failure
                entry["status"] = StageStatus.FAILED.value
                entry["explanation"] = sanitize_failure_message(str(error))
                entry["ended_at"] = _now()
                halted = True
                _checkpoint(context, heartbeat=lock_path)
                break
            stage_result = PipelineStageResultEnvelope.from_payload(
                PipelineStageResultPayload(
                    pipeline_semantic_id=pipeline_id,
                    stage=stage,
                    output_artifact_ids=tuple(outputs),
                )
            )
            try:
                save_or_reuse_envelope(
                    store_root,
                    "pipeline_stage_results",
                    stage_result,
                    extra_files=dict(context.stage_sidecars) or None,
                )
            except Exception as error:  # noqa: BLE001 — adversarial m-11
                # a store refusal (e.g. sidecar byte divergence across
                # attempts) fails the STAGE with sanitized evidence instead
                # of crashing the runner mid-state
                entry["status"] = StageStatus.FAILED.value
                entry["explanation"] = (
                    "stage-result publication refused: "
                    + sanitize_failure_message(str(error))
                )
                entry["ended_at"] = _now()
                halted = True
                _checkpoint(context, heartbeat=lock_path)
                break
            if prior_result_id == stage_result.stage_result_id:
                # proven byte-identical to the prior attempt's verified result
                entry["status"] = StageStatus.REUSED.value
                entry["explanation"] = (
                    "verified reuse: this attempt re-derived the identical "
                    "semantic stage result"
                )
            else:
                entry["status"] = StageStatus.COMPLETED.value
                entry["explanation"] = explanation
            entry["stage_result_id"] = stage_result.stage_result_id
            entry["output_artifact_ids"] = list(outputs)
            entry["ended_at"] = _now()
            _checkpoint(context, heartbeat=lock_path)
        state["attempts"][-1]["ended_at"] = _now()
        if not halted:
            state["current_stage"] = None
        path = _checkpoint(context, heartbeat=lock_path)
        return PipelineRunResult(
            pipeline_semantic_id=pipeline_id,
            attempt=attempt,
            stage_statuses={
                stage_value: entry["status"]
                for stage_value, entry in state["stages"].items()
                if entry["in_plan"]
            },
            state_path=path,
            result_envelope=context.result_envelope,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Publication gates and activation (FUX §30.6; verify-then-activate)
# ─────────────────────────────────────────────────────────────────────────────


class PublicationError(PermissionError):
    """Publication/activation refused (fail-closed)."""


PUBLICATION_GATE_IDS: tuple[str, ...] = (
    "all_planned_stages_terminal",
    "no_failed_stages",
    "pipeline_result_persisted",
    "control_flow_gates_passed",
)


def run_publication_gates(
    state_root: Path, pipeline_semantic_id: str, *, store_root: Path
) -> dict[str, Any]:
    """Evaluate the publication checklist and record it in the state file."""

    with _search_lock(
        Path(state_root), pipeline_semantic_id, stale_lock_seconds=86_400.0
    ):
        state = read_pipeline_state(state_root, pipeline_semantic_id)
        if state is None:
            raise PublicationError("no pipeline state exists for this semantic id")
        planned_entries = [
            entry for entry in state["stages"].values() if entry["in_plan"]
        ]
        terminal = all(
            entry["status"]
            in (
                StageStatus.COMPLETED.value,
                StageStatus.REUSED.value,
                StageStatus.BLOCKED.value,
            )
            for entry in planned_entries
        )
        no_failures = all(
            entry["status"] != StageStatus.FAILED.value for entry in planned_entries
        )
        publication = state.get("publication") or {}
        result_id = publication.get("pipeline_result_id")
        result_persisted = bool(result_id) and has_envelope(
            Path(store_root), "search_results", result_id
        )
        gates = {
            "all_planned_stages_terminal": terminal,
            "no_failed_stages": no_failures,
            "pipeline_result_persisted": result_persisted,
            "control_flow_gates_passed": bool(
                publication.get("control_flow_gates_passed")
            ),
        }
        publication["gates"] = gates
        publication["state"] = (
            "gates_passed" if all(gates.values()) else "prepared_not_published"
        )
        state["publication"] = publication
        _write_json_atomic(
            _state_dir(Path(state_root), pipeline_semantic_id) / _STATE_FILENAME, state
        )
        return gates


def activate_pipeline_result(
    state_root: Path, pipeline_semantic_id: str, *, store_root: Path
) -> str:
    """Activate the published research catalog entry (development scope only).

    Verification-only artifacts can NEVER activate a research catalog entry —
    the refusal is scope-based and has no override.
    """

    with _search_lock(
        Path(state_root), pipeline_semantic_id, stale_lock_seconds=86_400.0
    ):
        state = read_pipeline_state(state_root, pipeline_semantic_id)
        if state is None:
            raise PublicationError("no pipeline state exists for this semantic id")
        if state["run_scope"] == PipelineRunScope.VERIFICATION_5D.value:
            raise PublicationError(
                "verification-only artifacts can never activate a research "
                "catalog entry (verify-then-activate boundary)"
            )
        publication = state.get("publication") or {}
        gates = publication.get("gates")
        if not gates or not all(gates.values()):
            raise PublicationError(
                "activation requires every publication gate to pass first — "
                "run the publication gates"
            )
        from .catalog import append_catalog_event  # noqa: PLC0415

        result_id = publication["pipeline_result_id"]
        append_catalog_event(
            Path(store_root),
            kind="note",
            artifact_id=result_id,
            payload={"note": "research catalog entry activated (verify-then-activate)"},
        )
        publication["state"] = "published_activated"
        publication["activated"] = True
        state["publication"] = publication
        _write_json_atomic(
            _state_dir(Path(state_root), pipeline_semantic_id) / _STATE_FILENAME, state
        )
        return result_id


def _example_pipeline_spec() -> PipelineSemanticSpecPayload:
    return PipelineSemanticSpecPayload(
        run_scope=PipelineRunScope.VERIFICATION_5D,
        date_allowlist=("2026-06-04", "2026-06-05"),
        allowlist_hash=allowlist_sha256(("2026-06-04", "2026-06-05")),
        warmup_policy_id="zero_real_warmup_seed_snapshot_v1",
        search_charter_id="a" * 64,
        source_artifact_ids=(),
        feature_bundle_ids=(),
        label_policy_id=None,
        fold_protocol_id=None,
        model_protocol_id=None,
        cost_policy_sha256="b" * 64,
        account_policy_set_ids=(),
        portfolio_policy_ids=(),
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="simulated_clock_v1",
        ),
        software_commits={"quant_lab": "c" * 40, "strategy_core": "d" * 40},
        stage_plan=(
            QuantLabPipelineStage.S00_VALIDATE_INPUTS,
            QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
            QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
        ),
    )


def _example_stage_result() -> PipelineStageResultPayload:
    return PipelineStageResultPayload(
        pipeline_semantic_id="a" * 64,
        stage=QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
        output_artifact_ids=("b" * 64,),
    )


def _example_pipeline_result() -> PipelineResultPayload:
    return PipelineResultPayload(
        pipeline_semantic_id="a" * 64,
        run_scope=PipelineRunScope.VERIFICATION_5D,
        stage_result_ids={"02_run_or_reuse_sequential_replays": "b" * 64},
        frontier_id=None,
        control_flow_gates=None,
        verification_stamps=None,
    )


register_identity_pair(
    name="PipelineSemanticSpec",
    envelope_cls=PipelineSemanticIdentity,
    payload_cls=PipelineSemanticSpecPayload,
    id_field="pipeline_semantic_id",
    example_factory=_example_pipeline_spec,
)
register_identity_pair(
    name="PipelineStageResult",
    envelope_cls=PipelineStageResultEnvelope,
    payload_cls=PipelineStageResultPayload,
    id_field="stage_result_id",
    example_factory=_example_stage_result,
)
register_identity_pair(
    name="PipelineResult",
    envelope_cls=PipelineResultEnvelope,
    payload_cls=PipelineResultPayload,
    id_field="pipeline_result_id",
    example_factory=_example_pipeline_result,
)
