"""Feature-bundle graph (DELTA_TAXONOMY.md §7; brief §7A.12).

Bundles compose blocks by extension (a graph, not a cumulative enum ladder).
The frozen tier-compat bundles reproduce ``TIER_FEATURE_REGISTRY`` order-exact
so legacy M0–M3 runs participate in comparisons without re-execution; the tier
lane itself is never modified.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar

from pydantic import Field

from ..context_experiment_contracts import ContextFeatureTier
from ..context_feature_view import TIER_FEATURE_REGISTRY
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .feature_blocks import (
    FEATURE_BLOCK_REGISTRY,
    FEATURE_BLOCK_RESOLUTION_REGISTRY,
    BlockUnavailableError,
    resolve_available_block,
)
from .mbp1_source_contract import assert_no_deep_book_identifiers

__all__ = [
    "FeatureBundleDefinition",
    "FeatureBundleResolutionPayload",
    "FeatureBundleResolutionEnvelope",
    "FEATURE_BUNDLE_REGISTRY",
    "FROZEN_TIER_BUNDLES",
    "resolve_bundle_definition",
    "resolve_bundle",
    "bundle_for_tier",
]


class FeatureBundleDefinition(FrozenContract):
    feature_bundle_key: str
    bundle_version: int = Field(ge=1)
    human_name: str
    base_bundle_key: str | None
    included_block_keys: tuple[str, ...]
    excluded_feature_ids: tuple[str, ...] = ()
    manual_include_ids: tuple[str, ...] = ()
    manual_exclude_ids: tuple[str, ...] = ()
    manual_override_registration_id: str | None = None
    required_profile_capabilities: tuple[str, ...] = ()
    required_coverage_gates: ImmutableMap[str, float]


class FeatureBundleResolutionPayload(FrozenContract):
    feature_bundle_key: str
    bundle_version: int = Field(ge=1)
    resolved_block_ids: tuple[str, ...]
    resolved_feature_names: tuple[str, ...]
    resolved_source_identities: tuple[str, ...]


class FeatureBundleResolutionEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "resolved_feature_bundle_id"

    resolved_feature_bundle_id: str = Field(pattern=SHA256_PATTERN)
    payload: FeatureBundleResolutionPayload


def _bundle(
    key: str,
    human_name: str,
    *,
    base: str | None,
    blocks: tuple[str, ...],
) -> FeatureBundleDefinition:
    return FeatureBundleDefinition(
        feature_bundle_key=key,
        bundle_version=1,
        human_name=human_name,
        base_bundle_key=base,
        included_block_keys=blocks,
        required_coverage_gates={},
    )


FEATURE_BUNDLE_REGISTRY: MappingProxyType[str, FeatureBundleDefinition] = MappingProxyType(
    {
        bundle.feature_bundle_key: bundle
        for bundle in (
            _bundle(
                "B0_CORE",
                "Core + session",
                base=None,
                blocks=("IFVG_CORE_BASELINE_V1", "IFVG_SESSION_CONTEXT_V1"),
            ),
            _bundle(
                "B1_CORE_STRUCTURE",
                "Core + structure",
                base="B0_CORE",
                blocks=("IFVG_STRUCTURE_CONTEXT_V1",),
            ),
            _bundle(
                "B2_CORE_ORDER_FLOW",
                "Core + MBP-1 order flow (unresolvable until activation)",
                base="B0_CORE",
                blocks=("IFVG_ORDER_FLOW_MBP1_V1",),
            ),
            _bundle(
                "B3_CORE_STRUCTURE_ORDER_FLOW",
                "Core + structure + MBP-1 order flow (planned)",
                base="B1_CORE_STRUCTURE",
                blocks=("IFVG_ORDER_FLOW_MBP1_V1",),
            ),
            _bundle(
                "B4_CORE_STRUCTURE_LIQUIDITY",
                "Core + structure + liquidity",
                base="B1_CORE_STRUCTURE",
                blocks=("IFVG_LIQUIDITY_CONTEXT_V1",),
            ),
            _bundle(
                "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME",
                "B3 + regime (planned)",
                base="B3_CORE_STRUCTURE_ORDER_FLOW",
                blocks=("IFVG_REGIME_CONTEXT_V1",),
            ),
            _bundle(
                "B6_CORE_STRUCTURE_ORDER_FLOW_EXECUTION_LIQUIDITY",
                "B3 + execution liquidity (planned)",
                base="B3_CORE_STRUCTURE_ORDER_FLOW",
                blocks=("IFVG_EXECUTION_LIQUIDITY_V1",),
            ),
        )
    }
)

#: Frozen tier-compat bundles: immutable, order-exact copies of the tier lane.
FROZEN_TIER_BUNDLES: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "TIER_M0_FROZEN_V1": TIER_FEATURE_REGISTRY[ContextFeatureTier.M0],
        "TIER_M1_FROZEN_V1": TIER_FEATURE_REGISTRY[ContextFeatureTier.M1_PRIMARY],
        "TIER_M1_240_FROZEN_V1": TIER_FEATURE_REGISTRY[
            ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL
        ],
        "TIER_M2_FROZEN_V1": TIER_FEATURE_REGISTRY[ContextFeatureTier.M2],
        "TIER_M3_FROZEN_V1": TIER_FEATURE_REGISTRY[ContextFeatureTier.M3],
    }
)

_TIER_TO_FROZEN = MappingProxyType(
    {
        ContextFeatureTier.M0: "TIER_M0_FROZEN_V1",
        ContextFeatureTier.M1_PRIMARY: "TIER_M1_FROZEN_V1",
        ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL: "TIER_M1_240_FROZEN_V1",
        ContextFeatureTier.M2: "TIER_M2_FROZEN_V1",
        ContextFeatureTier.M3: "TIER_M3_FROZEN_V1",
    }
)

# order-exact equality with the frozen tier lane (import-time invariant;
# explicit raises so `python -O` cannot strip the checks)
for _key, _names in FROZEN_TIER_BUNDLES.items():
    if not (isinstance(_names, tuple) and _names):
        raise AssertionError(f"{_key} must be a non-empty tuple")
if FROZEN_TIER_BUNDLES["TIER_M3_FROZEN_V1"] != TIER_FEATURE_REGISTRY[ContextFeatureTier.M3]:
    raise AssertionError("frozen tier bundle drifted from the tier registry")


def bundle_for_tier(tier: ContextFeatureTier | str) -> str:
    """Map a legacy tier onto its frozen bundle key (cross-lane comparisons)."""

    return _TIER_TO_FROZEN[ContextFeatureTier(tier)]


def resolve_bundle_definition(
    feature_bundle_key: str,
    *,
    registry: MappingProxyType[str, FeatureBundleDefinition] | dict | None = None,
) -> tuple[FeatureBundleDefinition, tuple[str, ...]]:
    """Walk the base chain (cycle-checked); return (definition, ordered blocks)."""

    bundles = registry if registry is not None else FEATURE_BUNDLE_REGISTRY
    definition = bundles.get(feature_bundle_key)
    if definition is None:
        raise ValueError(f"unregistered feature bundle {feature_bundle_key!r}")
    chain: list[FeatureBundleDefinition] = []
    seen: set[str] = set()
    cursor: FeatureBundleDefinition | None = definition
    while cursor is not None:
        if cursor.feature_bundle_key in seen:
            raise ValueError(
                f"feature-bundle base chain has a cycle at {cursor.feature_bundle_key!r}"
            )
        seen.add(cursor.feature_bundle_key)
        chain.append(cursor)
        cursor = (
            bundles.get(cursor.base_bundle_key)
            if cursor.base_bundle_key is not None
            else None
        )
    ordered_blocks: list[str] = []
    for entry in reversed(chain):
        for block_key in entry.included_block_keys:
            if block_key not in ordered_blocks:
                ordered_blocks.append(block_key)
    return definition, tuple(ordered_blocks)


def resolve_bundle(
    feature_bundle_key: str,
    *,
    bundle_registry=None,
    definitions=None,
    resolutions=None,
    allow_experimental: bool = False,
) -> FeatureBundleResolutionEnvelope:
    """Resolve one concrete bundle version; refuses any non-AVAILABLE block."""

    definition, ordered_blocks = resolve_bundle_definition(
        feature_bundle_key, registry=bundle_registry
    )
    block_definitions = definitions if definitions is not None else FEATURE_BLOCK_REGISTRY
    resolved_ids: list[str] = []
    names: list[str] = []
    sources: list[str] = []
    for block_key in ordered_blocks:
        block_definition = block_definitions.get(block_key)
        if block_definition is None:
            raise ValueError(f"bundle references unregistered block {block_key!r}")
        try:
            envelope = resolve_available_block(
                block_key,
                definitions=block_definitions,
                resolutions=resolutions
                if resolutions is not None
                else FEATURE_BLOCK_RESOLUTION_REGISTRY,
                allow_experimental=allow_experimental,
            )
        except BlockUnavailableError as error:
            raise BlockUnavailableError(
                f"bundle {feature_bundle_key} is unresolvable: {error}"
            ) from error
        resolved_ids.append(envelope.resolved_feature_block_id)
        for name in envelope.payload.feature_names:
            if name in definition.excluded_feature_ids:
                continue
            if name not in names:
                names.append(name)
        sources.extend(envelope.payload.source_artifact_refs)
    for name in definition.manual_include_ids:
        if name not in names:
            names.append(name)
    names = [name for name in names if name not in definition.manual_exclude_ids]
    assert_no_deep_book_identifiers(names)
    payload = FeatureBundleResolutionPayload(
        feature_bundle_key=feature_bundle_key,
        bundle_version=definition.bundle_version,
        resolved_block_ids=tuple(resolved_ids),
        resolved_feature_names=tuple(names),
        resolved_source_identities=tuple(dict.fromkeys(sources)),
    )
    return FeatureBundleResolutionEnvelope.from_payload(payload)


register_identity_pair(
    name="FeatureBundleResolution",
    envelope_cls=FeatureBundleResolutionEnvelope,
    payload_cls=FeatureBundleResolutionPayload,
    id_field="resolved_feature_bundle_id",
    example_factory=lambda: resolve_bundle("B0_CORE").payload,
)
