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
    """R5B state: MBP-1 is ACTIVE; the other planned blocks still refuse."""

    for key in (
        "IFVG_VOLATILITY_CONTEXT_V1",
        "IFVG_REGIME_CONTEXT_V1",
        "IFVG_KEY_LEVEL_CONTEXT_V1",
        "IFVG_EXECUTION_LIQUIDITY_V1",
    ):
        assert FEATURE_BLOCK_REGISTRY[key].status is FeatureBlockStatus.PLANNED
        with pytest.raises(BlockUnavailableError):
            resolve_available_block(key)
    # bundles that still depend on a planned block refuse exactly as before
    for bundle in (
        "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME",
        "B6_CORE_STRUCTURE_ORDER_FLOW_EXECUTION_LIQUIDITY",
    ):
        with pytest.raises(BlockUnavailableError, match="unresolvable"):
            resolve_bundle(bundle)


def test_mbp1_block_is_active_at_r5b_with_the_research_boundary() -> None:
    """TEST_MATRIX §3.8 'MBP-1 activation state', R5B half: available as a
    NEW block version with its resolution envelope; the research-only
    offline boundary (owner decision R-6) rides the definition."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        MBP1_RESEARCH_BOUNDARY_PATH,
    )

    definition = FEATURE_BLOCK_REGISTRY["IFVG_ORDER_FLOW_MBP1_V1"]
    assert definition.status is FeatureBlockStatus.AVAILABLE
    # R5B activated version 2; the R5B.1 coverage-v2 re-resolution is version 3
    assert definition.block_version == 3
    assert definition.expected_computation_path == MBP1_RESEARCH_BOUNDARY_PATH
    assert definition.can_affect_execution is False
    envelope = resolve_available_block("IFVG_ORDER_FLOW_MBP1_V1")
    assert envelope.payload.formula_version == "ifvg_order_flow_mbp1_formula_v2"
    assert envelope.payload.materializer_version == "mbp1_feature_materializer_v2"
    assert envelope.payload.feature_names == mbp1_feature_names()
    assert envelope.payload.mbp1_feature_windows == R5B_WINDOW_SPECS
    assert envelope.payload.join_policy == "one_to_one_typed_null_on_missing"
    # MBP-1-bearing bundles resolve now — a baseline-vs-MBP-1 study is
    # constructible for the first time
    for bundle in ("B2_CORE_ORDER_FLOW", "B3_CORE_STRUCTURE_ORDER_FLOW"):
        resolved = resolve_bundle(bundle)
        assert envelope.resolved_feature_block_id in resolved.payload.resolved_block_ids


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
    """The PUBLISHED R5B registry is exactly the activation event applied to
    the exported pre-activation (R5 planned) state: version bump, first
    resolved id minted, registry-hash change, dependent bundles re-minted,
    unrelated bundles untouched."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        MBP1_ACTIVATION_ENVELOPE,
        PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY,
        PRE_ACTIVATION_RESOLUTION_REGISTRY,
        PRE_R5B1_FEATURE_BLOCK_REGISTRY,
        PRE_R5B1_RESOLUTION_REGISTRY,
        mbp1_activation_resolution_payload,
    )

    planned = PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY["IFVG_ORDER_FLOW_MBP1_V1"]
    assert planned.status is FeatureBlockStatus.PLANNED
    assert planned.block_version == 1
    assert "IFVG_ORDER_FLOW_MBP1_V1" not in PRE_ACTIVATION_RESOLUTION_REGISTRY
    pre_hash = feature_block_registry_hash(
        PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY, PRE_ACTIVATION_RESOLUTION_REGISTRY
    )
    baseline_b1_pre = resolve_bundle(
        "B1_CORE_STRUCTURE",
        definitions=PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY,
        resolutions=PRE_ACTIVATION_RESOLUTION_REGISTRY,
    )
    # replaying the event over the pre-activation state reproduces the
    # published registry EXACTLY
    definitions, resolutions, envelope = with_activated_block(
        feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
        resolution_payload=mbp1_activation_resolution_payload(),
        definitions=PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY,
        resolutions=PRE_ACTIVATION_RESOLUTION_REGISTRY,
    )
    assert envelope.resolved_feature_block_id == (
        MBP1_ACTIVATION_ENVELOPE.resolved_feature_block_id
    )
    # the activation replay reproduces the R5B state EXACTLY (the published
    # registry is one further event on top — see the R5B.1 test below)
    assert dict(definitions) == dict(PRE_R5B1_FEATURE_BLOCK_REGISTRY)
    assert {k: v.model_dump(mode="json") for k, v in resolutions.items()} == {
        k: v.model_dump(mode="json") for k, v in PRE_R5B1_RESOLUTION_REGISTRY.items()
    }
    # the historical activation payload keeps the v1 formula/materializer
    assert envelope.payload.formula_version == "ifvg_order_flow_mbp1_formula_v1"
    # version bump + first resolved id minted + registry hash changed
    assert definitions["IFVG_ORDER_FLOW_MBP1_V1"].block_version == 2
    assert definitions["IFVG_ORDER_FLOW_MBP1_V1"].status is FeatureBlockStatus.AVAILABLE
    assert envelope.resolved_feature_block_id not in {
        e.resolved_feature_block_id for e in PRE_ACTIVATION_RESOLUTION_REGISTRY.values()
    }
    assert (
        feature_block_registry_hash(
            PRE_R5B1_FEATURE_BLOCK_REGISTRY, PRE_R5B1_RESOLUTION_REGISTRY
        )
        != pre_hash
    )
    assert feature_block_registry_hash() != pre_hash
    # every dependent bundle gains a NEW resolved id; the logical key is stable
    # (resolved against the R5B state — the published registry carries the
    # R5B.1 coverage-v2 re-resolution on top)
    activated_b2 = resolve_bundle(
        "B2_CORE_ORDER_FLOW",
        definitions=PRE_R5B1_FEATURE_BLOCK_REGISTRY,
        resolutions=PRE_R5B1_RESOLUTION_REGISTRY,
    )
    assert activated_b2.payload.feature_bundle_key == "B2_CORE_ORDER_FLOW"
    assert envelope.resolved_feature_block_id in activated_b2.payload.resolved_block_ids
    # unrelated bundles keep their resolved ids across the activation
    assert (
        resolve_bundle("B1_CORE_STRUCTURE").resolved_feature_bundle_id
        == baseline_b1_pre.resolved_feature_bundle_id
    )
    # re-activating the published (already-available) block refuses
    with pytest.raises(ValueError, match="not planned"):
        with_activated_block(
            feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
            resolution_payload=mbp1_activation_resolution_payload(),
        )
    # a wrong-version activation over the planned state is refused
    with pytest.raises(ValueError, match="block_version"):
        with_activated_block(
            feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
            resolution_payload=mbp1_activation_resolution_payload().model_copy(
                update={"block_version": 9}
            ),
            definitions=PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY,
            resolutions=PRE_ACTIVATION_RESOLUTION_REGISTRY,
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


def test_coverage_v2_reresolution_is_a_second_versioned_event() -> None:
    """R5B.1: the coverage-policy correction re-mints the MBP-1 resolution as
    a SECOND versioned event over the R5B state — block v3, new resolved id,
    new registry hash, new B2/B3 bundle ids; B0/B1/B4 untouched; the
    activation event stays byte-identical (v1 formula kept)."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        MBP1_ACTIVATION_ENVELOPE,
        MBP1_COVERAGE_V2_ENVELOPE,
        PRE_R5B1_FEATURE_BLOCK_REGISTRY,
        PRE_R5B1_RESOLUTION_REGISTRY,
        PRE_R6_1_FEATURE_BLOCK_REGISTRY,
        PRE_R6_1_RESOLUTION_REGISTRY,
        FeatureBlockResolutionEnvelope,
        mbp1_coverage_v2_resolution_payload,
        with_reresolved_block,
    )

    definitions, resolutions, envelope = with_reresolved_block(
        feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
        resolution_payload=mbp1_coverage_v2_resolution_payload(),
        definitions=PRE_R5B1_FEATURE_BLOCK_REGISTRY,
        resolutions=PRE_R5B1_RESOLUTION_REGISTRY,
    )
    assert envelope.resolved_feature_block_id == (
        MBP1_COVERAGE_V2_ENVELOPE.resolved_feature_block_id
    )
    # the replayed v2 event IS the R5B.1 state (the R6.1 panel registration is
    # a later, separately provable event over it — see the R6.1 test below)
    assert dict(definitions) == dict(PRE_R6_1_FEATURE_BLOCK_REGISTRY)
    assert {k: v.model_dump(mode="json") for k, v in resolutions.items()} == {
        k: v.model_dump(mode="json") for k, v in PRE_R6_1_RESOLUTION_REGISTRY.items()
    }
    assert set(FEATURE_BLOCK_REGISTRY) == set(definitions) | {"IFVG_CONTEXT_BAR_PANEL_V1"}
    assert definitions["IFVG_ORDER_FLOW_MBP1_V1"].block_version == 3
    assert envelope.payload.block_version == 3
    assert envelope.payload.formula_version == "ifvg_order_flow_mbp1_formula_v2"
    assert envelope.payload.materializer_version == "mbp1_feature_materializer_v2"
    assert envelope.resolved_feature_block_id != (
        MBP1_ACTIVATION_ENVELOPE.resolved_feature_block_id
    )
    assert MBP1_ACTIVATION_ENVELOPE.payload.formula_version == (
        "ifvg_order_flow_mbp1_formula_v1"
    )
    assert feature_block_registry_hash() != feature_block_registry_hash(
        PRE_R5B1_FEATURE_BLOCK_REGISTRY, PRE_R5B1_RESOLUTION_REGISTRY
    )
    for bundle in ("B2_CORE_ORDER_FLOW", "B3_CORE_STRUCTURE_ORDER_FLOW"):
        pre = resolve_bundle(
            bundle,
            definitions=PRE_R5B1_FEATURE_BLOCK_REGISTRY,
            resolutions=PRE_R5B1_RESOLUTION_REGISTRY,
        )
        post = resolve_bundle(bundle)
        assert pre.resolved_feature_bundle_id != post.resolved_feature_bundle_id
        assert envelope.resolved_feature_block_id in post.payload.resolved_block_ids
    for bundle in ("B0_CORE", "B1_CORE_STRUCTURE", "B4_CORE_STRUCTURE_LIQUIDITY"):
        pre = resolve_bundle(
            bundle,
            definitions=PRE_R5B1_FEATURE_BLOCK_REGISTRY,
            resolutions=PRE_R5B1_RESOLUTION_REGISTRY,
        )
        assert pre.resolved_feature_bundle_id == resolve_bundle(bundle).resolved_feature_bundle_id
    # a re-resolution can never target a planned block, a wrong version, or
    # an unchanged identity
    with pytest.raises(ValueError, match="only an available block"):
        with_reresolved_block(
            feature_block_key="IFVG_REGIME_CONTEXT_V1",
            resolution_payload=mbp1_coverage_v2_resolution_payload().model_copy(
                update={"feature_block_key": "IFVG_REGIME_CONTEXT_V1"}
            ),
        )
    with pytest.raises(ValueError, match="block_version"):
        with_reresolved_block(
            feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
            resolution_payload=mbp1_coverage_v2_resolution_payload(),
        )  # the published block is already v3
    unchanged = mbp1_coverage_v2_resolution_payload().model_copy(update={"block_version": 4})
    with pytest.raises(ValueError, match="must change the resolved block identity"):
        with_reresolved_block(
            feature_block_key="IFVG_ORDER_FLOW_MBP1_V1",
            resolution_payload=unchanged,
            definitions=dict(FEATURE_BLOCK_REGISTRY),
            resolutions={
                **FEATURE_BLOCK_RESOLUTION_REGISTRY,
                "IFVG_ORDER_FLOW_MBP1_V1": FeatureBlockResolutionEnvelope.from_payload(
                    unchanged
                ),
            },
        )


# ── R6.1: the context-bar panel block registration (plan §6.A; owner Q3) ─────


def test_context_bar_panel_registration_is_a_versioned_event() -> None:
    """R6.1: the panel block is REGISTERED as its own versioned event over the
    R5B.1 state — replay equality, hash change, B0/B1/B2/B4 ids stable, the
    block contract (HTF_TAP, ``row_id`` join, categorical session state, the
    seven names in order, the two owner-registered intervals), BP0 resolves
    with the panel block only, and a candidate view refuses BP0."""

    from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
        build_bundle_feature_view,
        frozen_tier_for_bundle,
    )
    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
        CONTEXT_BAR_PANEL_FEATURES,
        PANEL_AS_OF_POLICY_ID_V1,
        PANEL_INTERVALS_SECONDS_V1,
    )
    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        CONTEXT_BAR_PANEL_COMPUTATION_PATH,
        CONTEXT_BAR_PANEL_REGISTRATION_ENVELOPE,
        PRE_R6_1_FEATURE_BLOCK_REGISTRY,
        PRE_R6_1_RESOLUTION_REGISTRY,
        AvailabilityStage,
        context_bar_panel_definition,
        context_bar_panel_resolution_payload,
        with_registered_block,
    )
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
        known_cluster_fixture,
    )

    definitions, resolutions, envelope = with_registered_block(
        definition=context_bar_panel_definition(),
        resolution_payload=context_bar_panel_resolution_payload(),
        definitions=PRE_R6_1_FEATURE_BLOCK_REGISTRY,
        resolutions=PRE_R6_1_RESOLUTION_REGISTRY,
    )
    assert envelope.resolved_feature_block_id == (
        CONTEXT_BAR_PANEL_REGISTRATION_ENVELOPE.resolved_feature_block_id
    )
    assert dict(definitions) == dict(FEATURE_BLOCK_REGISTRY)
    assert {k: v.model_dump(mode="json") for k, v in resolutions.items()} == {
        k: v.model_dump(mode="json") for k, v in FEATURE_BLOCK_RESOLUTION_REGISTRY.items()
    }
    assert feature_block_registry_hash() != feature_block_registry_hash(
        PRE_R6_1_FEATURE_BLOCK_REGISTRY, PRE_R6_1_RESOLUTION_REGISTRY
    )
    for bundle in (
        "B0_CORE",
        "B1_CORE_STRUCTURE",
        "B2_CORE_ORDER_FLOW",
        "B4_CORE_STRUCTURE_LIQUIDITY",
    ):
        pre = resolve_bundle(
            bundle,
            definitions=PRE_R6_1_FEATURE_BLOCK_REGISTRY,
            resolutions=PRE_R6_1_RESOLUTION_REGISTRY,
        )
        assert pre.resolved_feature_bundle_id == resolve_bundle(bundle).resolved_feature_bundle_id
    # the block contract
    definition = FEATURE_BLOCK_REGISTRY["IFVG_CONTEXT_BAR_PANEL_V1"]
    assert definition.status is FeatureBlockStatus.AVAILABLE
    assert definition.block_version == 1
    assert definition.availability_stage is AvailabilityStage.HTF_TAP
    assert definition.source_kind == "replay_chart_bars"
    assert definition.expected_computation_path == CONTEXT_BAR_PANEL_COMPUTATION_PATH
    assert "research_only_offline" in definition.experimental_flags
    payload = envelope.payload
    assert payload.feature_names == tuple(CONTEXT_BAR_PANEL_FEATURES)
    assert payload.join_keys == ("row_id",)
    assert payload.categorical_features == ("cbp_session_state",)
    assert payload.validity_fields == ("cbp_valid",)
    assert payload.missing_reason_fields == ("cbp_missing_reason",)
    assert payload.as_of_policy == PANEL_AS_OF_POLICY_ID_V1
    assert payload.source_timeframes == tuple(PANEL_INTERVALS_SECONDS_V1) == (300, 900)
    assert_no_deep_book_identifiers(payload.feature_names)
    assert not set(payload.feature_names) & set(TIER_FEATURE_REGISTRY[ContextFeatureTier.M3])
    # BP0 resolves with the panel block only; it is no frozen tier and no candidate view
    bp0 = resolve_bundle("BP0_CONTEXT_BAR_PANEL")
    assert bp0.payload.resolved_block_ids == (envelope.resolved_feature_block_id,)
    assert bp0.payload.resolved_feature_names == tuple(CONTEXT_BAR_PANEL_FEATURES)
    assert frozen_tier_for_bundle(bp0.payload.resolved_feature_names) is None
    with pytest.raises(ValueError, match="missing from the immutable candidate view"):
        build_bundle_feature_view(known_cluster_fixture(k=3, n=60).view, bp0)
    # a registration event is one-time and version-1 only
    with pytest.raises(ValueError, match="already registered"):
        with_registered_block(
            definition=context_bar_panel_definition(),
            resolution_payload=context_bar_panel_resolution_payload(),
        )
    with pytest.raises(ValueError, match="AVAILABLE block"):
        with_registered_block(
            definition=context_bar_panel_definition().model_copy(
                update={"feature_block_key": "X_V1", "status": FeatureBlockStatus.PLANNED}
            ),
            resolution_payload=context_bar_panel_resolution_payload().model_copy(
                update={"feature_block_key": "X_V1"}
            ),
        )
    with pytest.raises(ValueError, match="block_version 1"):
        with_registered_block(
            definition=context_bar_panel_definition().model_copy(
                update={"feature_block_key": "X_V1", "block_version": 2}
            ),
            resolution_payload=context_bar_panel_resolution_payload().model_copy(
                update={"feature_block_key": "X_V1", "block_version": 2}
            ),
        )
