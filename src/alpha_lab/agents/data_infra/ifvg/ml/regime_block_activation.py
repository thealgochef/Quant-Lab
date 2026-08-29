"""Status-gated, versioned activation of ``IFVG_REGIME_CONTEXT_V1`` (R6.1 D9).

The regime block stays PLANNED at import (the module registries are never
touched here). Its activation is a PURE, versioned event
(``features.feature_blocks.with_activated_block``) whose resolution payload
binds the EXACT frozen ``regime_promotion_decision_id``,
``owner_decision_artifact_id``, ``capability_assessment_id``, resolved
regime protocol, and fold-local feature artifact — the block never asks
for a protocol's "current / latest" decision. Activation VERIFIED-LOADS
every one of those artifacts through the stores and refuses any protocol,
assessment, supersession, status, or request mismatch with a message that
names the frozen decision (nothing here promotes; a below-FEATURE_ELIGIBLE
decision cannot activate the block).

The model feature names are the fit-local id (when the artifact encodes it),
the margin, the assigned distance, and the local distance vector; the
canonical aligned reporting fields are excluded (D8); ``as_of_policy =
fold_local_fit_partition_assignment_v1``; join keys ``(candidate_id,
fold_index)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..features.feature_blocks import (
    FeatureBlockResolutionEnvelope,
    FeatureBlockResolutionPayload,
    feature_block_registry_hash,
    with_activated_block,
)
from ..search.identities import canonical_contract_sha256
from ..search.owner_decisions import (
    OwnerDecisionArtifactEnvelope,
    OwnerDecisionRefusalError,
    assert_owner_decision_authorizes,
    load_owner_decision,
    load_supersession_chain,
    transition_key,
)
from ..search.store import SearchStoreError
from .regime_contracts import (
    ObservationGranularity,
    RegimeCapabilityAssessmentEnvelope,
    RegimePromotionDecisionEnvelope,
    RegimeProtocolEnvelope,
    RegimeStatus,
)
from .regime_fold_features import (
    REGIME_FOLD_FEATURE_FORMULA_VERSION,
    REGIME_FOLD_FEATURE_MATERIALIZER_VERSION,
    RegimeFoldFeatureArtifactEnvelope,
    load_regime_fold_features,
)
from .regime_store import (
    load_regime_assessment,
    load_regime_promotion,
    load_regime_protocol,
)

__all__ = [
    "REGIME_BLOCK_KEY",
    "REGIME_ACTIVATION_AS_OF_POLICY",
    "REGIME_ACTIVATION_JOIN_KEYS",
    "REGIME_ACTIVATION_SOURCE_INTERVAL_POLICY",
    "FEATURE_ELIGIBLE_STATUSES",
    "RegimeActivationRefusalError",
    "RegimeBlockActivation",
    "regime_activation_resolution_payload",
    "assert_activation_authority",
    "activate_regime_context_block",
]

REGIME_BLOCK_KEY = "IFVG_REGIME_CONTEXT_V1"
REGIME_ACTIVATION_AS_OF_POLICY = "fold_local_fit_partition_assignment_v1"
REGIME_ACTIVATION_JOIN_KEYS: tuple[str, ...] = ("candidate_id", "fold_index")
REGIME_ACTIVATION_SOURCE_INTERVAL_POLICY = "fold_local_regime_fit_v1"
#: The statuses that authorize a regime output as a model feature source.
FEATURE_ELIGIBLE_STATUSES: frozenset[RegimeStatus] = frozenset(
    {RegimeStatus.FEATURE_ELIGIBLE, RegimeStatus.MODEL_FEATURE}
)


class RegimeActivationRefusalError(PermissionError):
    """The frozen authority does not authorize activating the regime block."""


@dataclass(frozen=True)
class RegimeBlockActivation:
    """The pure activation event's result: new registry mappings, the minted
    resolution envelope, and the verified authority it binds."""

    definitions: MappingProxyType
    resolutions: MappingProxyType
    envelope: FeatureBlockResolutionEnvelope
    protocol: RegimeProtocolEnvelope
    fold_feature_artifact: RegimeFoldFeatureArtifactEnvelope
    promotion_decision: RegimePromotionDecisionEnvelope
    owner_decision: OwnerDecisionArtifactEnvelope
    assessment: RegimeCapabilityAssessmentEnvelope
    registry_hash: str

    @property
    def resolved_feature_block_id(self) -> str:
        return self.envelope.resolved_feature_block_id


def _frozen(decision: RegimePromotionDecisionEnvelope) -> str:
    return f"frozen decision {decision.regime_promotion_decision_id[:12]}…"


def assert_activation_authority(
    *,
    protocol: RegimeProtocolEnvelope,
    fold_feature_artifact: RegimeFoldFeatureArtifactEnvelope,
    promotion_decision: RegimePromotionDecisionEnvelope,
    owner_decision: OwnerDecisionArtifactEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    run_scope: str,
    supersession_chain=(),
) -> None:
    """Every refusal names the frozen decision; nothing here promotes."""

    decision = promotion_decision.payload
    protocol_id = protocol.resolved_regime_protocol_id
    if RegimeStatus(decision.status) not in FEATURE_ELIGIBLE_STATUSES:
        raise RegimeActivationRefusalError(
            f"activation refused: {_frozen(promotion_decision)} carries status "
            f"{RegimeStatus(decision.status).value}, below FEATURE_ELIGIBLE — nothing here "
            "promotes"
        )
    if decision.resolved_regime_protocol_id != protocol_id:
        raise RegimeActivationRefusalError(
            f"activation refused: {_frozen(promotion_decision)} names a different regime "
            "protocol than the request"
        )
    if decision.owner_ratification_ref != owner_decision.owner_decision_artifact_id:
        raise RegimeActivationRefusalError(
            f"activation refused: {_frozen(promotion_decision)} was ratified by a different "
            "owner-decision artifact than the request"
        )
    if decision.capability_assessment_ref != assessment.regime_capability_assessment_id:
        raise RegimeActivationRefusalError(
            f"activation refused: {_frozen(promotion_decision)} references a different "
            "capability assessment than the request"
        )
    if assessment.payload.resolved_regime_protocol_id != protocol_id:
        raise RegimeActivationRefusalError(
            f"activation refused: the assessment behind {_frozen(promotion_decision)} belongs "
            "to another protocol"
        )
    if not assessment.payload.gates_passed:
        raise RegimeActivationRefusalError(
            f"activation refused: the assessment behind {_frozen(promotion_decision)} did not "
            "pass its capability gates"
        )
    artifact = fold_feature_artifact.payload
    if artifact.resolved_regime_protocol_id != protocol_id:
        raise RegimeActivationRefusalError(
            f"activation refused: the fold-feature artifact belongs to another protocol "
            f"({_frozen(promotion_decision)})"
        )
    fit_ids = {ref.regime_fit_id for ref in artifact.regime_fit_ids_by_fold if ref.regime_fit_id}
    if not fit_ids:
        raise RegimeActivationRefusalError(
            f"activation refused: the fold-feature artifact carries no regime fit "
            f"({_frozen(promotion_decision)})"
        )
    uncovered = sorted(fit_ids - set(assessment.payload.regime_fit_ids))
    if uncovered:
        raise RegimeActivationRefusalError(
            f"activation refused: {len(uncovered)} fold-feature fit(s) are not covered by the "
            f"assessment behind {_frozen(promotion_decision)}"
        )
    try:
        assert_owner_decision_authorizes(
            owner_decision,
            protocol_envelope=protocol,
            assessment_envelope=assessment,
            transition=transition_key(decision.previous_status, decision.status),
            as_of=decision.decided_at,
            run_scope=run_scope,  # type: ignore[arg-type]
            supersession_chain=tuple(supersession_chain),
        )
    except OwnerDecisionRefusalError as error:
        raise RegimeActivationRefusalError(
            f"activation refused ({_frozen(promotion_decision)}): {error}"
        ) from error


def regime_activation_resolution_payload(
    *,
    protocol: RegimeProtocolEnvelope,
    fold_feature_artifact: RegimeFoldFeatureArtifactEnvelope,
    promotion_decision: RegimePromotionDecisionEnvelope,
    owner_decision: OwnerDecisionArtifactEnvelope,
    definitions=None,
) -> FeatureBlockResolutionPayload:
    """The block's first resolution (block_version = planned + 1 = 2)."""

    from ..features.feature_blocks import FEATURE_BLOCK_REGISTRY  # noqa: PLC0415

    registry = definitions if definitions is not None else FEATURE_BLOCK_REGISTRY
    definition = registry.get(REGIME_BLOCK_KEY)
    if definition is None:
        raise ValueError(f"unregistered feature block {REGIME_BLOCK_KEY!r}")
    artifact = fold_feature_artifact.payload
    columns = artifact.columns
    names = tuple(artifact.model_feature_names)
    categorical = tuple(artifact.categorical_model_features)
    payload = protocol.payload
    timeframes: tuple[int, ...] = ()
    if ObservationGranularity(payload.observation_granularity) is (
        ObservationGranularity.CONTEXT_BAR_PANEL
    ):
        timeframes = (int(payload.panel_interval_seconds or 0),)
    return FeatureBlockResolutionPayload(
        feature_block_key=REGIME_BLOCK_KEY,
        block_version=definition.block_version + 1,
        formula_version=REGIME_FOLD_FEATURE_FORMULA_VERSION,
        source_artifact_refs=(
            protocol.resolved_regime_protocol_id,
            fold_feature_artifact.regime_fold_feature_artifact_id,
            promotion_decision.payload.capability_assessment_ref,
            promotion_decision.regime_promotion_decision_id,
            owner_decision.owner_decision_artifact_id,
        ),
        source_schema_hash=artifact.feature_schema_hash,
        feature_schema_hash=canonical_contract_sha256({"features": list(names)}),
        materializer_version=REGIME_FOLD_FEATURE_MATERIALIZER_VERSION,
        feature_names=names,
        numeric_features=tuple(name for name in names if name not in categorical),
        categorical_features=categorical,
        validity_fields=(columns.valid,),
        missing_reason_fields=(columns.missing_reason,),
        source_timeframes=timeframes,
        source_interval_policy=REGIME_ACTIVATION_SOURCE_INTERVAL_POLICY,
        as_of_policy=REGIME_ACTIVATION_AS_OF_POLICY,
        join_keys=REGIME_ACTIVATION_JOIN_KEYS,
        join_policy="one_to_one_typed_null_on_missing",
        direction_normalization="none",
        session_normalization="none",
        warmup_requirement="exclude_warmup_v1",
        coverage_requirements={},
    )


def activate_regime_context_block(
    root: Path,
    *,
    resolved_regime_protocol_id: str,
    regime_fold_feature_artifact_id: str,
    regime_promotion_decision_id: str,
    owner_decision_artifact_id: str,
    run_scope: str,
    definitions=None,
    resolutions=None,
) -> RegimeBlockActivation:
    """Verified-load the exact frozen authority and mint the pure activation
    event over ``definitions`` / ``resolutions`` (the published registries by
    default). The module registries are never mutated; the caller decides
    what to do with the returned mappings."""

    root = Path(root)
    try:
        protocol = load_regime_protocol(root, resolved_regime_protocol_id)
        fold_feature_artifact = load_regime_fold_features(root, regime_fold_feature_artifact_id)
        promotion_decision = load_regime_promotion(root, regime_promotion_decision_id)
        owner_decision = load_owner_decision(root, owner_decision_artifact_id)
        assessment = load_regime_assessment(
            root, promotion_decision.payload.capability_assessment_ref
        )
    except SearchStoreError as error:
        raise RegimeActivationRefusalError(
            "activation refused: a frozen authority artifact is not a verified store entry "
            f"({error})"
        ) from error
    assert_activation_authority(
        protocol=protocol,
        fold_feature_artifact=fold_feature_artifact,
        promotion_decision=promotion_decision,
        owner_decision=owner_decision,
        assessment=assessment,
        run_scope=run_scope,
        supersession_chain=load_supersession_chain(root),
    )
    payload = regime_activation_resolution_payload(
        protocol=protocol,
        fold_feature_artifact=fold_feature_artifact,
        promotion_decision=promotion_decision,
        owner_decision=owner_decision,
        definitions=definitions,
    )
    new_definitions, new_resolutions, envelope = with_activated_block(
        feature_block_key=REGIME_BLOCK_KEY,
        resolution_payload=payload,
        definitions=definitions,
        resolutions=resolutions,
    )
    frozen_definitions = MappingProxyType(dict(new_definitions))
    frozen_resolutions = MappingProxyType(dict(new_resolutions))
    return RegimeBlockActivation(
        definitions=frozen_definitions,
        resolutions=frozen_resolutions,
        envelope=envelope,
        protocol=protocol,
        fold_feature_artifact=fold_feature_artifact,
        promotion_decision=promotion_decision,
        owner_decision=owner_decision,
        assessment=assessment,
        registry_hash=feature_block_registry_hash(frozen_definitions, frozen_resolutions),
    )
