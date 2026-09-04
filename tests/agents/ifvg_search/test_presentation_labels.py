"""UI-3 §6.5 — the human-label registry and the distinct availability chips
(pure): every registered key of the profile / bundle / block / objective /
model / algorithm registries has a human label; the technical key is never
replaced, only accompanied; proposed / ratified / offline / planned /
implemented / blocked / experimental statuses are distinct."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ArtifactPreparationStatus,
    ContextFeatureTier,
    ProfileCapabilityStatus,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    FEATURE_BLOCK_REGISTRY,
    FeatureBlockStatus,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (
    FEATURE_BUNDLE_REGISTRY,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import MODEL_PROTOCOL_REGISTRY
from alpha_lab.agents.data_infra.ifvg.ml.regime_algorithms import (
    REGIME_ALGORITHM_REGISTRY,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import RegimeRole, RegimeStatus
from alpha_lab.agents.data_infra.ifvg.ml.regime_study import COMPARISON_CLASSES
from alpha_lab.agents.data_infra.ifvg.presentation.labels import (
    AVAILABILITY_CHIPS,
    AvailabilityKind,
    availability_chip,
    availability_for_algorithm,
    availability_for_block_status,
    availability_for_stamp,
    label_for,
    technical_key_for,
)
from alpha_lab.agents.data_infra.ifvg.presentation.review_vocabulary import (
    VERDICT_LABELS,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import UiStatus
from alpha_lab.agents.data_infra.ifvg.profiles import PROFILE_BUILDERS
from alpha_lab.agents.data_infra.ifvg.search.charter import OBJECTIVE_DIRECTIONS


def test_every_registry_key_has_a_human_label_and_the_key_stays_recoverable() -> None:
    registries = {
        "profile": tuple(PROFILE_BUILDERS),
        "bundle": tuple(FEATURE_BUNDLE_REGISTRY),
        "block": tuple(FEATURE_BLOCK_REGISTRY),
        "objective": tuple(OBJECTIVE_DIRECTIONS),
        "model_protocol": tuple(MODEL_PROTOCOL_REGISTRY),
        "algorithm": tuple(REGIME_ALGORITHM_REGISTRY),
        "block_status": tuple(status.value for status in FeatureBlockStatus),
        "regime_status": tuple(status.value for status in RegimeStatus),
        "regime_role": tuple(role.value for role in RegimeRole),
        "profile_capability": tuple(status.value for status in ProfileCapabilityStatus),
        "preparation_status": tuple(status.value for status in ArtifactPreparationStatus),
        "comparison_class": tuple(COMPARISON_CLASSES),
        "feature_tier": tuple(tier.value for tier in ContextFeatureTier),
        "verdict": tuple(VERDICT_LABELS),
        "stamp": ("proposed_protocol_default", "registered_storage_budget", "custom"),
    }
    for kind, keys in registries.items():
        for key in keys:
            label = label_for(kind, key)
            assert label != key, (kind, key)  # a human label exists
            assert label.strip(), (kind, key)
            assert technical_key_for(kind, label) == key, (kind, key)
    assert label_for("bundle", "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME") == (
        "Core + structure + order flow + regime"
    )
    assert label_for("objective", "payout_probability_per_rolling_30d") == (
        "Payout probability per rolling 30 days"
    )
    assert label_for("stamp", "proposed_protocol_default") == (
        "Proposed default — owner ratification required"
    )
    assert label_for("verdict", "not_applicable") == VERDICT_LABELS["not_applicable"]
    # an unregistered key is returned unchanged — never invented, never hidden
    assert label_for("bundle", "B99_UNKNOWN") == "B99_UNKNOWN"
    assert technical_key_for("bundle", "no such label") is None


def test_availability_chips_are_distinct_by_glyph_and_word() -> None:
    assert set(AVAILABILITY_CHIPS) == set(AvailabilityKind)
    chips = {availability_chip(kind) for kind in AvailabilityKind}
    assert len(chips) == len(AvailabilityKind)  # every chip distinct
    for kind, (status, text) in AVAILABILITY_CHIPS.items():
        assert isinstance(status, UiStatus)
        assert text.strip() and availability_chip(kind).endswith(text)
    proposed_status, _ = AVAILABILITY_CHIPS[AvailabilityKind.PROPOSED]
    ratified_status, _ = AVAILABILITY_CHIPS[AvailabilityKind.RATIFIED]
    assert proposed_status is UiStatus.WARNING and ratified_status is UiStatus.PASS
    assert AVAILABILITY_CHIPS[AvailabilityKind.PLANNED][0] is UiStatus.NOT_SELECTED
    assert AVAILABILITY_CHIPS[AvailabilityKind.BLOCKED][0] is UiStatus.BLOCKED
    assert "offline" in availability_chip(AvailabilityKind.RESEARCH_ONLY_OFFLINE).lower()
    assert "disabled" in availability_chip(AvailabilityKind.PLANNED).lower()


def test_availability_derives_from_the_registries_not_from_wording() -> None:
    assert availability_for_block_status("available") is AvailabilityKind.IMPLEMENTED
    assert availability_for_block_status("planned") is AvailabilityKind.PLANNED
    assert availability_for_block_status("experimental") is AvailabilityKind.EXPERIMENTAL
    assert availability_for_block_status("blocked_missing_source") is AvailabilityKind.BLOCKED
    assert availability_for_block_status("superseded") is AvailabilityKind.SUPERSEDED
    kmeans = REGIME_ALGORITHM_REGISTRY["kmeans_v1"]
    spectral = REGIME_ALGORITHM_REGISTRY["spectral_clustering_train_only_v1"]
    assert availability_for_algorithm(kmeans) is AvailabilityKind.IMPLEMENTED
    assert availability_for_algorithm(spectral) is AvailabilityKind.PLANNED
    assert availability_for_stamp("proposed_protocol_default") is AvailabilityKind.PROPOSED
    assert availability_for_stamp("owner_ratified") is AvailabilityKind.RATIFIED
    assert availability_for_stamp("registered_storage_budget") is AvailabilityKind.IMPLEMENTED
