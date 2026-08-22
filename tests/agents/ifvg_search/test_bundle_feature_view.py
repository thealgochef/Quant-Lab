"""Bundle feature-view tests (PHASED R5; TEST_MATRIX §3.8 MBP-1 activation
state, R5 half: the planned block refuses at every study-construction seam)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ContextFeatureTier,
)
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    BundleFeatureViewEnvelope,
    build_bundle_feature_view,
    frozen_tier_for_bundle,
    resolve_available_bundle_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    BlockUnavailableError,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from tests.agents.ifvg_search.pipeline_fixture import build_mini_view


@pytest.fixture(scope="module")
def view():
    view, _labels = build_mini_view()
    return view


def test_b0_core_view_scopes_the_frame_to_the_bundle(view):
    envelope, frame = resolve_available_bundle_view(view, "B0_CORE")
    assert isinstance(envelope, BundleFeatureViewEnvelope)
    payload = envelope.payload
    assert payload.feature_bundle_key == "B0_CORE"
    assert payload.view_id == view.view_id
    assert payload.candidate_count == len(view.frame)
    assert set(payload.resolved_feature_names) <= set(frame.columns)
    assert "candidate_id" in frame.columns  # identity columns survive


def test_mbp1_bundles_refuse_before_r5b(view):
    """No baseline-vs-MBP-1 study is constructible in R5: every bundle
    carrying `IFVG_ORDER_FLOW_MBP1_V1` refuses with the planned status."""

    for bundle_key in ("B2_CORE_ORDER_FLOW", "B3_CORE_STRUCTURE_ORDER_FLOW"):
        with pytest.raises(BlockUnavailableError, match="planned"):
            resolve_available_bundle_view(view, bundle_key)


def test_missing_view_columns_refuse_fail_closed(view):
    envelope = resolve_bundle("B0_CORE")
    narrowed = view.frame.drop(columns=["risk_ticks"])
    import dataclasses

    broken = dataclasses.replace(view, frame=narrowed)
    with pytest.raises(ValueError, match="missing from the immutable candidate view"):
        build_bundle_feature_view(broken, envelope)


def test_b0_core_maps_onto_the_frozen_m0_tier(view):
    envelope, _frame = resolve_available_bundle_view(view, "B0_CORE")
    assert (
        frozen_tier_for_bundle(envelope.payload.resolved_feature_names)
        is ContextFeatureTier.M0
    )


def test_unmatched_feature_sets_have_no_tier_mapping():
    assert frozen_tier_for_bundle(("direction", "made_up_feature")) is None
