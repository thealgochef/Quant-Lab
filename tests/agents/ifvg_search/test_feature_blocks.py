"""Feature block/bundle/MBP-1 contract suites (DT §6–§7; TEST_MATRIX §3.1/§3.8)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ContextFeatureTier,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import TIER_FEATURE_REGISTRY
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    CORE_BASELINE_FEATURES,
    FEATURE_BLOCK_REGISTRY,
    FEATURE_BLOCK_RESOLUTION_REGISTRY,
    SESSION_FEATURES,
    BlockUnavailableError,
    FeatureBlockResolutionPayload,
    FeatureBlockStatus,
    feature_block_registry_hash,
    resolve_available_block,
    with_activated_block,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (
    FROZEN_TIER_BUNDLES,
    bundle_for_tier,
    resolve_bundle,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    DEEP_BOOK_IDENTIFIER_REGEX,
    R5B_WINDOW_SPECS,
    StageEvidenceCutoff,
    WindowTriggerSemantics,
    assert_no_deep_book_identifiers,
    mbp1_feature_names,
)


def test_available_blocks_partition_m3_exactly() -> None:
    union: list[str] = []
    for key in (
        "IFVG_CORE_BASELINE_V1",
        "IFVG_SESSION_CONTEXT_V1",
        "IFVG_STRUCTURE_CONTEXT_V1",
        "IFVG_DISPLACEMENT_CONTEXT_V2",
        "IFVG_LIQUIDITY_CONTEXT_V1",
    ):
        union.extend(FEATURE_BLOCK_RESOLUTION_REGISTRY[key].payload.feature_names)
    assert len(union) == len(set(union)), "blocks overlap"
    assert set(union) == set(TIER_FEATURE_REGISTRY[ContextFeatureTier.M3])
    assert set(CORE_BASELINE_FEATURES) | set(SESSION_FEATURES) == set(
        TIER_FEATURE_REGISTRY[ContextFeatureTier.M0]
    )


def test_frozen_tier_bundles_are_order_exact() -> None:
    for tier, key in (
        (ContextFeatureTier.M0, "TIER_M0_FROZEN_V1"),
        (ContextFeatureTier.M1_PRIMARY, "TIER_M1_FROZEN_V1"),
        (ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL, "TIER_M1_240_FROZEN_V1"),
        (ContextFeatureTier.M2, "TIER_M2_FROZEN_V1"),
        (ContextFeatureTier.M3, "TIER_M3_FROZEN_V1"),
    ):
        assert FROZEN_TIER_BUNDLES[key] == TIER_FEATURE_REGISTRY[tier]
        assert bundle_for_tier(tier) == key


def test_planned_blocks_fail_closed_everywhere() -> None:
    for key in (
        "IFVG_ORDER_FLOW_MBP1_V1",
        "IFVG_VOLATILITY_CONTEXT_V1",
        "IFVG_REGIME_CONTEXT_V1",
        "IFVG_KEY_LEVEL_CONTEXT_V1",
        "IFVG_EXECUTION_LIQUIDITY_V1",
    ):
        assert FEATURE_BLOCK_REGISTRY[key].status is FeatureBlockStatus.PLANNED
        with pytest.raises(BlockUnavailableError):
            resolve_available_block(key)
    for bundle in (
        "B2_CORE_ORDER_FLOW",
        "B3_CORE_STRUCTURE_ORDER_FLOW",
        "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME",
    ):
        with pytest.raises(BlockUnavailableError, match="unresolvable"):
            resolve_bundle(bundle)


def test_experimental_block_needs_the_explicit_flag() -> None:
    with pytest.raises(BlockUnavailableError, match="experimental"):
        resolve_available_block("IFVG_STRUCTURE_CONTEXT_240_V1")
    envelope = resolve_available_block(
        "IFVG_STRUCTURE_CONTEXT_240_V1", allow_experimental=True
    )
    assert all(
        name.startswith("ctx_structure_14400s_")
        for name in envelope.payload.feature_names
    )


def test_no_deep_book_identifier_anywhere() -> None:
    names = [
        *FEATURE_BLOCK_REGISTRY,
        *(
            name
            for envelope in FEATURE_BLOCK_RESOLUTION_REGISTRY.values()
            for name in envelope.payload.feature_names
        ),
        *mbp1_feature_names(),
        *(spec.feature_window_key for spec in R5B_WINDOW_SPECS),
    ]
    assert_no_deep_book_identifiers(names)
    with pytest.raises(ValueError, match="unrepresentable"):
        assert_no_deep_book_identifiers(("ofl_snap_entry_mbp10_depth",))
    assert DEEP_BOOK_IDENTIFIER_REGEX.search("mbp1") is None
    # the single exemption is the opaque provenance literal, nothing else
    assert_no_deep_book_identifiers(("legacy_verified_replay_source",))


def test_legacy_provenance_is_not_a_block_source_kind() -> None:
    definition = FEATURE_BLOCK_REGISTRY["IFVG_CORE_BASELINE_V1"]
    with pytest.raises(ValueError):
        type(definition).model_validate(
            {
                **definition.model_dump(mode="json"),
                "source_kind": "legacy_verified_replay_source",
            }
        )


def test_activation_is_a_versioned_registry_event() -> None:
    baseline_hash = feature_block_registry_hash()
    baseline_b1 = resolve_bundle("B1_CORE_STRUCTURE")
    planned = FEATURE_BLOCK_REGISTRY["IFVG_ORDER_FLOW_MBP1_V1"]
    payload = FeatureBlockResolutionPayload(
        feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
        block_version=planned.block_version + 1,
        formula_version="ifvg_order_flow_mbp1_v1",
        source_artifact_refs=("a" * 64,),
        source_schema_hash="b" * 64,
        feature_schema_hash="c" * 64,
        materializer_version="mbp1_feature_materializer_v1",
        feature_names=mbp1_feature_names(),
        numeric_features=mbp1_feature_names(),
        categorical_features=(),
        validity_fields=(),
        missing_reason_fields=(),
        source_timeframes=(),
        source_interval_policy="event_stream_v1",
        as_of_policy="stage_evidence_cutoff_v2",
        join_keys=("candidate_id",),
        join_policy="one_to_one_typed_null_on_missing",
        direction_normalization="none",
        session_normalization="none",
        warmup_requirement="exclude_warmup_v1",
        coverage_requirements={},
        mbp1_feature_windows=R5B_WINDOW_SPECS,
    )
    definitions, resolutions, envelope = with_activated_block(
        feature_block_key="IFVG_ORDER_FLOW_MBP1_V1", resolution_payload=payload
    )
    # version bump + first resolved id minted + registry hash changed
    assert definitions["IFVG_ORDER_FLOW_MBP1_V1"].block_version == planned.block_version + 1
    assert definitions["IFVG_ORDER_FLOW_MBP1_V1"].status is FeatureBlockStatus.AVAILABLE
    assert envelope.resolved_feature_block_id not in {
        e.resolved_feature_block_id for e in FEATURE_BLOCK_RESOLUTION_REGISTRY.values()
    }
    assert feature_block_registry_hash(definitions, resolutions) != baseline_hash
    # every dependent bundle gains a NEW resolved id; the logical key is stable
    activated_b2 = resolve_bundle(
        "B2_CORE_ORDER_FLOW", definitions=definitions, resolutions=resolutions
    )
    assert activated_b2.payload.feature_bundle_key == "B2_CORE_ORDER_FLOW"
    assert envelope.resolved_feature_block_id in activated_b2.payload.resolved_block_ids
    # unrelated bundles keep their resolved ids
    assert (
        resolve_bundle(
            "B1_CORE_STRUCTURE", definitions=definitions, resolutions=resolutions
        ).resolved_feature_bundle_id
        == baseline_b1.resolved_feature_bundle_id
    )
    # a wrong-version activation is refused
    with pytest.raises(ValueError, match="block_version"):
        with_activated_block(
            feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
            resolution_payload=payload.model_copy(update={"block_version": 9}),
        )


def test_r5b_window_registry_is_post_trigger_inclusive_and_inf_free() -> None:
    assert len(R5B_WINDOW_SPECS) == 5 + 4  # 5 snapshots + 4 transitions
    for spec in R5B_WINDOW_SPECS:
        assert spec.trigger_semantics is WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE
        assert spec.lower_bound.value == "open"
        assert spec.upper_bound.value == "closed"
        assert spec.minimum_event_count == 1
    with pytest.raises(ValueError, match="unrepresentable"):
        StageEvidenceCutoff(
            stage_id="entry",
            stage_as_of_ts_utc="2026-06-04T14:30:00Z",
            cutoff_kind="exact_source_order_key",
            exact_source_order_key=("+inf", "+inf", 0, 0),
            completed_bar_close_ts_utc=None,
            same_timestamp_policy_id="exclude_all_v1",
            source_evidence_ref=None,
        )


def test_bundle_resolution_is_deterministic() -> None:
    assert (
        resolve_bundle("B4_CORE_STRUCTURE_LIQUIDITY").resolved_feature_bundle_id
        == resolve_bundle("B4_CORE_STRUCTURE_LIQUIDITY").resolved_feature_bundle_id
    )
