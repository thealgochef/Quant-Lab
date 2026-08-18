"""Feature-block contract (DELTA_TAXONOMY.md §6; brief §7A.11).

The stable registry definition (:class:`FeatureBlockDefinition`) and one
resolved materialization (:class:`FeatureBlockResolutionPayload/Envelope`) are
separate objects. Blocks partition the frozen tier ladder DISJOINTLY, and the
union of the available blocks reproduces ``TIER_FEATURE_REGISTRY[M3]``
feature-for-feature (module assertion + unit test). Activation of a planned
block publishes a new definition version and mints its first resolved id — a
registry-hash-changing event, never a bare status flip (revision P1-5).
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Literal

from pydantic import Field

from ..context_experiment_contracts import ContextFeatureTier
from ..context_feature_view import (
    M0_FEATURES,
    TIER_FEATURE_REGISTRY,
)
from ..context_model import categorical_features_for
from ..context_schemas import IFVG_CONTEXT_ARROW_REGISTRY_HASH
from ..contracts import IFVG_CAPTURE_SCHEMA_VERSION, IFVG_DATASET_SCHEMA_VERSION
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
    register_identity_pair,
)
from .mbp1_source_contract import (
    R5B_WINDOW_SPECS,
    Mbp1FeatureWindowSpec,
    assert_no_deep_book_identifiers,
    mbp1_feature_names,
)

__all__ = [
    "FeatureBlockStatus",
    "AvailabilityStage",
    "FeatureBlockDefinition",
    "FeatureBlockResolutionPayload",
    "FeatureBlockResolutionEnvelope",
    "FEATURE_BLOCK_REGISTRY",
    "FEATURE_BLOCK_RESOLUTION_REGISTRY",
    "resolve_block_definition",
    "resolve_available_block",
    "feature_block_registry_hash",
    "with_activated_block",
    "BlockUnavailableError",
    "SESSION_FEATURES",
    "CORE_BASELINE_FEATURES",
]


class FeatureBlockStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_OWNER_DECISION = "blocked_owner_decision"
    EXPERIMENTAL = "experimental"
    SUPERSEDED = "superseded"


class AvailabilityStage(StrEnum):
    HTF_TAP = "htf_tap"
    PARENT_LOCK = "parent_lock"
    OPPOSING_CONFIRMATION = "opposing_confirmation"
    INVERSION = "inversion"
    ENTRY_DECISION = "entry_decision"


class BlockUnavailableError(PermissionError):
    """A planned/blocked block was requested for materialization."""


class FeatureBlockDefinition(FrozenContract):
    feature_block_key: str
    block_version: int = Field(ge=1)
    human_name: str
    status: FeatureBlockStatus
    status_reason: str | None
    feature_family: str
    source_kind: Literal[
        "v2_candidate",
        "v3_context",
        "mbp1_parquet",
        "regime_artifact",
        "bars_1m",
        "planned_external",
    ]
    availability_stage: AvailabilityStage
    dependencies: tuple[str, ...] = ()
    incompatible_blocks: tuple[str, ...] = ()
    experimental_flags: tuple[str, ...] = ()
    requires_strategy_replay: bool = False
    requires_feature_materialization: bool = True
    requires_model_retrain: bool = True
    can_affect_execution: bool = False
    expected_computation_path: str = "feature_materialization_plus_model_refit"


class FeatureBlockResolutionPayload(FrozenContract):
    feature_block_key: str
    block_version: int = Field(ge=1)
    formula_version: str
    source_artifact_refs: tuple[str, ...]
    source_schema_hash: str = Field(pattern=SHA256_PATTERN)
    feature_schema_hash: str = Field(pattern=SHA256_PATTERN)
    materializer_version: str
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    validity_fields: tuple[str, ...]
    missing_reason_fields: tuple[str, ...]
    source_timeframes: tuple[int, ...]
    source_interval_policy: str
    as_of_policy: str
    join_keys: tuple[str, ...]
    join_policy: Literal["one_to_one_required", "one_to_one_typed_null_on_missing"]
    direction_normalization: str
    session_normalization: str
    warmup_requirement: str
    coverage_requirements: ImmutableMap[str, float]
    mbp1_feature_windows: tuple[Mbp1FeatureWindowSpec, ...] = ()


class FeatureBlockResolutionEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "resolved_feature_block_id"

    resolved_feature_block_id: str = Field(pattern=SHA256_PATTERN)
    payload: FeatureBlockResolutionPayload


SESSION_FEATURES: tuple[str, ...] = ("entry_session", "in_engine_session", "in_doc_session")
CORE_BASELINE_FEATURES: tuple[str, ...] = tuple(
    name for name in M0_FEATURES if name not in SESSION_FEATURES
)

_PRIMARY_TFS = (60, 180, 300, 600, 900, 1800, 3600)
_M3 = TIER_FEATURE_REGISTRY[ContextFeatureTier.M3]
_STRUCTURE_PRIMARY = tuple(
    name
    for name in _M3
    if name.startswith("ctx_structure_") and not name.startswith("ctx_structure_14400s_")
)
_STRUCTURE_240 = tuple(
    name
    for name in TIER_FEATURE_REGISTRY[ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL]
    if name.startswith("ctx_structure_14400s_")
)
_DISPLACEMENT = tuple(name for name in _M3 if name.startswith("ctx_displacement_"))
_LIQUIDITY = tuple(
    name for name in _M3 if name.startswith(("ctx_pool_", "ctx_sweep_"))
)


def _validity_split(names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    validity = tuple(
        name for name in names if name.endswith(("_valid", "_source_available"))
    )
    missing = tuple(name for name in names if name.endswith("_missing_reason"))
    return validity, missing


def _definition(
    key: str,
    human_name: str,
    status: FeatureBlockStatus,
    *,
    family: str,
    source_kind: str,
    stage: AvailabilityStage = AvailabilityStage.ENTRY_DECISION,
    reason: str | None = None,
    flags: tuple[str, ...] = (),
) -> FeatureBlockDefinition:
    return FeatureBlockDefinition(
        feature_block_key=key,
        block_version=1,
        human_name=human_name,
        status=status,
        status_reason=reason,
        feature_family=family,
        source_kind=source_kind,  # type: ignore[arg-type]
        availability_stage=stage,
        experimental_flags=flags,
    )


def _source_schema_hash(source_kind_descriptor: str) -> str:
    """The ACTUAL schema authority per source kind (C17 resolution).

    v3-context blocks pin the real locked Arrow registry hash; v2-candidate
    blocks pin the exact versioned capture/dataset schema surface. Block
    resolutions are deliberately artifact-independent — study cells pin the
    concrete artifacts via their data lineage.
    """

    if source_kind_descriptor == "v3_context":
        return IFVG_CONTEXT_ARROW_REGISTRY_HASH
    return canonical_contract_sha256(
        {
            "source_kind": source_kind_descriptor,
            "capture_schema_version": IFVG_CAPTURE_SCHEMA_VERSION,
            "dataset_schema_version": IFVG_DATASET_SCHEMA_VERSION,
        }
    )


def _resolution(
    key: str,
    names: tuple[str, ...],
    *,
    formula_version: str,
    source_kind_descriptor: str,
    source_timeframes: tuple[int, ...] = (),
    join_policy: str = "one_to_one_typed_null_on_missing",
    direction_normalization: str = "none",
) -> FeatureBlockResolutionEnvelope:
    validity, missing = _validity_split(names)
    categorical = categorical_features_for(names)
    payload = FeatureBlockResolutionPayload(
        feature_block_key=key,
        block_version=1,
        formula_version=formula_version,
        source_artifact_refs=(),
        source_schema_hash=_source_schema_hash(source_kind_descriptor),
        feature_schema_hash=canonical_contract_sha256({"features": list(names)}),
        materializer_version="build_candidate_feature_view_v1",
        feature_names=names,
        numeric_features=tuple(
            name for name in names if name not in categorical
        ),
        categorical_features=tuple(categorical),
        validity_fields=validity,
        missing_reason_fields=missing,
        source_timeframes=source_timeframes,
        source_interval_policy="event_driven_context_v2",
        as_of_policy="candidate_stage_as_of_v1",
        join_keys=("candidate_id",),
        join_policy=join_policy,  # type: ignore[arg-type]
        direction_normalization=direction_normalization,
        session_normalization="none",
        warmup_requirement="exclude_warmup_v1",
        coverage_requirements={},
    )
    return FeatureBlockResolutionEnvelope.from_payload(payload)


_DEFINITIONS: dict[str, FeatureBlockDefinition] = {
    definition.feature_block_key: definition
    for definition in (
        _definition(
            "IFVG_CORE_BASELINE_V1",
            "IFVG lifecycle geometry core",
            FeatureBlockStatus.AVAILABLE,
            family="lifecycle_geometry",
            source_kind="v2_candidate",
        ),
        _definition(
            "IFVG_SESSION_CONTEXT_V1",
            "Session context",
            FeatureBlockStatus.AVAILABLE,
            family="session",
            source_kind="v2_candidate",
        ),
        _definition(
            "IFVG_STRUCTURE_CONTEXT_V1",
            "Market-structure context (primary TFs)",
            FeatureBlockStatus.AVAILABLE,
            family="structure",
            source_kind="v3_context",
        ),
        _definition(
            "IFVG_STRUCTURE_CONTEXT_240_V1",
            "Market-structure context (4H, experimental)",
            FeatureBlockStatus.EXPERIMENTAL,
            family="structure",
            source_kind="v3_context",
            reason="Q-40 240m anchor status is experimental",
            flags=("anchor_240m_experimental",),
        ),
        _definition(
            "IFVG_DISPLACEMENT_CONTEXT_V2",
            "Displacement windows",
            FeatureBlockStatus.AVAILABLE,
            family="displacement",
            source_kind="v3_context",
        ),
        _definition(
            "IFVG_LIQUIDITY_CONTEXT_V1",
            "Equal-level pools and sweeps",
            FeatureBlockStatus.AVAILABLE,
            family="liquidity",
            source_kind="v3_context",
        ),
        _definition(
            "IFVG_VOLATILITY_CONTEXT_V1",
            "Volatility context (planned)",
            FeatureBlockStatus.PLANNED,
            family="volatility",
            source_kind="bars_1m",
            reason="reserved ctx_vol_*; requires the SC formula-v3 change (post-V1)",
        ),
        _definition(
            "IFVG_ORDER_FLOW_MBP1_V1",
            "MBP-1 order-flow (planned through R5; research-only offline at R5B)",
            FeatureBlockStatus.PLANNED,
            family="order_flow",
            source_kind="mbp1_parquet",
            reason=(
                "activation is the R5B versioned event; the activated block is "
                "research_only_offline (owner decision R-6)"
            ),
        ),
        _definition(
            "IFVG_REGIME_CONTEXT_V1",
            "Regime context features (planned)",
            FeatureBlockStatus.PLANNED,
            family="regime",
            source_kind="regime_artifact",
            reason="depends on an OOS-capable, owner-ratified regime artifact",
        ),
        _definition(
            "IFVG_KEY_LEVEL_CONTEXT_V1",
            "Key-level distances (planned)",
            FeatureBlockStatus.PLANNED,
            family="key_levels",
            source_kind="planned_external",
            reason="reserved ctx_keylevel_* (PDH/PDL/session extremes)",
        ),
        _definition(
            "IFVG_EXECUTION_LIQUIDITY_V1",
            "Execution-liquidity features (planned)",
            FeatureBlockStatus.PLANNED,
            family="execution_liquidity",
            source_kind="mbp1_parquet",
            reason=(
                "shares the MBP-1 source artifact with a SEPARATE semantic "
                "identity (§7A.6.L); reserved exliq_*"
            ),
        ),
    )
}

_RESOLUTIONS: dict[str, FeatureBlockResolutionEnvelope] = {
    "IFVG_CORE_BASELINE_V1": _resolution(
        "IFVG_CORE_BASELINE_V1",
        CORE_BASELINE_FEATURES,
        formula_version="ifvg_v2_capture_row_v1",
        source_kind_descriptor="v2_candidate",
        join_policy="one_to_one_required",
    ),
    "IFVG_SESSION_CONTEXT_V1": _resolution(
        "IFVG_SESSION_CONTEXT_V1",
        SESSION_FEATURES,
        formula_version="ifvg_v2_capture_row_v1",
        source_kind_descriptor="v2_candidate",
        join_policy="one_to_one_required",
    ),
    "IFVG_STRUCTURE_CONTEXT_V1": _resolution(
        "IFVG_STRUCTURE_CONTEXT_V1",
        _STRUCTURE_PRIMARY,
        formula_version="ifvg_context_formula_v2",
        source_kind_descriptor="v3_context",
        source_timeframes=_PRIMARY_TFS,
    ),
    "IFVG_STRUCTURE_CONTEXT_240_V1": _resolution(
        "IFVG_STRUCTURE_CONTEXT_240_V1",
        _STRUCTURE_240,
        formula_version="ifvg_context_formula_v2",
        source_kind_descriptor="v3_context",
        source_timeframes=(14400,),
    ),
    "IFVG_DISPLACEMENT_CONTEXT_V2": _resolution(
        "IFVG_DISPLACEMENT_CONTEXT_V2",
        _DISPLACEMENT,
        formula_version="ifvg_context_formula_v2",
        source_kind_descriptor="v3_context",
        direction_normalization="thesis_directional_v1",
    ),
    "IFVG_LIQUIDITY_CONTEXT_V1": _resolution(
        "IFVG_LIQUIDITY_CONTEXT_V1",
        _LIQUIDITY,
        formula_version="ifvg_context_formula_v2",
        source_kind_descriptor="v3_context",
    ),
}

FEATURE_BLOCK_REGISTRY: MappingProxyType[str, FeatureBlockDefinition] = MappingProxyType(
    dict(_DEFINITIONS)
)
FEATURE_BLOCK_RESOLUTION_REGISTRY: MappingProxyType[str, FeatureBlockResolutionEnvelope] = (
    MappingProxyType(dict(_RESOLUTIONS))
)


def resolve_block_definition(feature_block_key: str) -> FeatureBlockDefinition:
    definition = FEATURE_BLOCK_REGISTRY.get(feature_block_key)
    if definition is None:
        raise ValueError(f"unregistered feature block {feature_block_key!r}")
    return definition


def resolve_available_block(
    feature_block_key: str,
    *,
    definitions: MappingProxyType[str, FeatureBlockDefinition] | dict | None = None,
    resolutions: MappingProxyType[str, FeatureBlockResolutionEnvelope] | dict | None = None,
    allow_experimental: bool = False,
) -> FeatureBlockResolutionEnvelope:
    """The ONLY route to a materializable block — planned/blocked fail closed."""

    registry = definitions if definitions is not None else FEATURE_BLOCK_REGISTRY
    resolved = resolutions if resolutions is not None else FEATURE_BLOCK_RESOLUTION_REGISTRY
    definition = registry.get(feature_block_key)
    if definition is None:
        raise ValueError(f"unregistered feature block {feature_block_key!r}")
    if definition.status is FeatureBlockStatus.EXPERIMENTAL and not allow_experimental:
        raise BlockUnavailableError(
            f"block {feature_block_key} is experimental: {definition.status_reason}"
        )
    if definition.status not in (
        FeatureBlockStatus.AVAILABLE,
        FeatureBlockStatus.EXPERIMENTAL,
    ):
        raise BlockUnavailableError(
            f"block {feature_block_key} is {definition.status.value}"
            + (f": {definition.status_reason}" if definition.status_reason else "")
        )
    envelope = resolved.get(feature_block_key)
    if envelope is None:
        raise BlockUnavailableError(
            f"block {feature_block_key} has no resolution envelope (not activated)"
        )
    return envelope


def feature_block_registry_hash(
    definitions=None,
    resolutions=None,
) -> str:
    registry = definitions if definitions is not None else FEATURE_BLOCK_REGISTRY
    resolved = resolutions if resolutions is not None else FEATURE_BLOCK_RESOLUTION_REGISTRY
    return canonical_contract_sha256(
        {
            "definitions": {
                key: value.model_dump(mode="json") for key, value in sorted(registry.items())
            },
            "resolutions": {
                key: value.model_dump(mode="json") for key, value in sorted(resolved.items())
            },
        }
    )


def with_activated_block(
    *,
    feature_block_key: str,
    resolution_payload: FeatureBlockResolutionPayload,
    definitions=None,
    resolutions=None,
) -> tuple[dict, dict, FeatureBlockResolutionEnvelope]:
    """Pure activation event (revision P1-5): NEVER a bare status flip.

    Returns new (definitions, resolutions, minted envelope) mappings with the
    definition's ``block_version`` bumped, status ``available``, and the first
    ``resolved_feature_block_id`` minted — changing the registry hash and every
    dependent bundle's resolved id. The module-level registries are immutable;
    R5B publishes the returned mappings as the new registry version.
    """

    registry = dict(definitions if definitions is not None else FEATURE_BLOCK_REGISTRY)
    resolved = dict(resolutions if resolutions is not None else FEATURE_BLOCK_RESOLUTION_REGISTRY)
    definition = registry.get(feature_block_key)
    if definition is None:
        raise ValueError(f"unregistered feature block {feature_block_key!r}")
    if definition.status is not FeatureBlockStatus.PLANNED:
        raise ValueError(f"block {feature_block_key} is not planned; nothing to activate")
    new_version = definition.block_version + 1
    if resolution_payload.feature_block_key != feature_block_key:
        raise ValueError("resolution payload key mismatch")
    if resolution_payload.block_version != new_version:
        raise ValueError(
            f"activation must mint block_version {new_version} "
            f"(got {resolution_payload.block_version})"
        )
    envelope = FeatureBlockResolutionEnvelope.from_payload(resolution_payload)
    registry[feature_block_key] = definition.model_copy(
        update={
            "block_version": new_version,
            "status": FeatureBlockStatus.AVAILABLE,
            "status_reason": None,
        }
    )
    resolved[feature_block_key] = envelope
    return registry, resolved, envelope


# ── module-level invariants (fail at import if the partition drifts) ─────────

_AVAILABLE_UNION = (
    *CORE_BASELINE_FEATURES,
    *SESSION_FEATURES,
    *_STRUCTURE_PRIMARY,
    *_DISPLACEMENT,
    *_LIQUIDITY,
)
if len(_AVAILABLE_UNION) != len(set(_AVAILABLE_UNION)):
    raise AssertionError("feature blocks must be disjoint")
if set(_AVAILABLE_UNION) != set(_M3):
    raise AssertionError("available blocks must partition TIER_FEATURE_REGISTRY[M3] exactly")
assert_no_deep_book_identifiers(FEATURE_BLOCK_REGISTRY.keys())
assert_no_deep_book_identifiers(_AVAILABLE_UNION)
assert_no_deep_book_identifiers(mbp1_feature_names())
for _spec in R5B_WINDOW_SPECS:
    assert_no_deep_book_identifiers(_spec.feature_names)


register_identity_pair(
    name="FeatureBlockResolution",
    envelope_cls=FeatureBlockResolutionEnvelope,
    payload_cls=FeatureBlockResolutionPayload,
    id_field="resolved_feature_block_id",
    example_factory=lambda: _RESOLUTIONS["IFVG_SESSION_CONTEXT_V1"].payload,
)
