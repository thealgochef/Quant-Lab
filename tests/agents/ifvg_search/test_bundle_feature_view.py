"""Bundle feature-view tests (PHASED R5/R5B; TEST_MATRIX §3.8 MBP-1
activation state: since R5B the block is active — MBP-1-bearing bundle views
require the exact materialized evidence and join one-to-one with typed
nulls; a resolved bundle without its evidence still fails closed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ContextFeatureTier,
)
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    BundleFeatureViewEnvelope,
    build_bundle_feature_view,
    frozen_tier_for_bundle,
    mbp1_block_keys_in_bundle,
    resolve_available_bundle_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (
    resolve_bundle,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    mbp1_feature_names,
)
from tests.agents.ifvg_search.mbp1_fixture import feature_artifact_for_frame
from tests.agents.ifvg_search.pipeline_fixture import build_mini_view


@pytest.fixture(scope="module")
def view():
    view, _labels = build_mini_view()
    return view


def _mbp1_metrics_for(view, *, drop_last: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"candidate_id": view.frame["candidate_id"].astype(str)})
    for name in mbp1_feature_names():
        frame[name] = rng.normal(0.0, 1.0, size=len(frame))
    if drop_last:
        frame = frame.iloc[:-1]
    return frame


def _mbp1_evidence_for(view, *, drop_last: bool = False, anchor_salt: str = "a"):
    """(verified envelope, full-schema frame) over the view's candidates."""

    return feature_artifact_for_frame(
        _mbp1_metrics_for(view, drop_last=drop_last), anchor_salt=anchor_salt
    )


def test_b0_core_view_scopes_the_frame_to_the_bundle(view):
    envelope, frame = resolve_available_bundle_view(view, "B0_CORE")
    assert isinstance(envelope, BundleFeatureViewEnvelope)
    payload = envelope.payload
    assert payload.feature_bundle_key == "B0_CORE"
    assert payload.view_id == view.view_id
    assert payload.candidate_count == len(view.frame)
    assert set(payload.resolved_feature_names) <= set(frame.columns)
    assert "candidate_id" in frame.columns  # identity columns survive


def test_mbp1_bundle_views_require_the_exact_evidence(view):
    """R5B: MBP-1-bearing bundles RESOLVE, but a view over them without the
    materialized feature frame AND its artifact envelope fails closed —
    order-flow evidence is never inferred."""

    for bundle_key in ("B2_CORE_ORDER_FLOW", "B3_CORE_STRUCTURE_ORDER_FLOW"):
        assert mbp1_block_keys_in_bundle(resolve_bundle(bundle_key))
    envelope, full_frame = _mbp1_evidence_for(view)
    # the M0-complete mini view satisfies B2's v2 half exactly, so the ONLY
    # missing evidence is the order-flow frame/envelope — refused fail-closed
    with pytest.raises(ValueError, match="frame AND its artifact envelope"):
        resolve_available_bundle_view(view, "B2_CORE_ORDER_FLOW")
    with pytest.raises(ValueError, match="frame AND its artifact envelope"):
        resolve_available_bundle_view(
            view, "B2_CORE_ORDER_FLOW", mbp1_features=full_frame
        )
    # a bundle whose v2/v3 half is ALSO unsatisfied refuses on those columns
    # first — the MBP-1 join can never paper over missing view evidence
    with pytest.raises(ValueError, match="missing from the immutable candidate view"):
        resolve_available_bundle_view(
            view,
            "B3_CORE_STRUCTURE_ORDER_FLOW",
            mbp1_features=full_frame,
            mbp1_feature_artifact=envelope,
        )


def test_mbp1_bundle_view_verifies_and_binds_the_artifact(view):
    """Review F1: the frame must HASH to the pinned envelope's table hash —
    the binding is rehashed at the seam, never caller-asserted — and two
    different evidence artifacts mint two different view identities."""

    envelope, full_frame = _mbp1_evidence_for(view)
    view_env, frame = resolve_available_bundle_view(
        view,
        "B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
    )
    assert view_env.payload.mbp1_feature_artifact_id == (
        envelope.mbp1_feature_artifact_id
    )
    assert view_env.payload.candidate_count == len(view.frame)
    assert set(mbp1_feature_names()) <= set(frame.columns)
    # a tampered frame refuses against the same envelope (rehash mismatch)
    tampered = full_frame.copy()
    tampered.loc[0, "ofl_snap_entry_bid_sz"] = 999.0
    with pytest.raises(ValueError, match="verified, never asserted"):
        resolve_available_bundle_view(
            view,
            "B2_CORE_ORDER_FLOW",
            mbp1_features=tampered,
            mbp1_feature_artifact=envelope,
        )
    # a DIFFERENT evidence artifact (distinct recipe over the same frame)
    # mints a DIFFERENT view identity — the evidence is pinned, never ambient
    other_envelope, other_frame = _mbp1_evidence_for(view, anchor_salt="b")
    other, _ = resolve_available_bundle_view(
        view,
        "B2_CORE_ORDER_FLOW",
        mbp1_features=other_frame,
        mbp1_feature_artifact=other_envelope,
    )
    assert other.bundle_feature_view_id != view_env.bundle_feature_view_id


def test_mbp1_bundle_view_refuses_cohort_misalignment(view):
    """Review F9: a view candidate with NO ROW in the feature artifact is a
    cohort misalignment (the materializer preserves every anchored row, so
    absence is never typed-null evidence) — refused fail-closed."""

    envelope, full_frame = _mbp1_evidence_for(view, drop_last=True)
    with pytest.raises(ValueError, match="cohort misalignment"):
        resolve_available_bundle_view(
            view,
            "B2_CORE_ORDER_FLOW",
            mbp1_features=full_frame,
            mbp1_feature_artifact=envelope,
        )


def test_mbp1_bundle_view_carries_typed_null_values(view):
    """Typed-null VALUES (the registered missingness) survive the join with
    the row preserved — the cohort never changes (acceptance §7A.19.14)."""

    metrics = _mbp1_metrics_for(view)
    for name in mbp1_feature_names():
        metrics.loc[metrics.index[-1], name] = float("nan")
    envelope, full_frame = feature_artifact_for_frame(metrics)
    view_env, frame = resolve_available_bundle_view(
        view,
        "B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
    )
    assert view_env.payload.candidate_count == len(view.frame)
    assert len(frame) == len(view.frame)
    last = frame.iloc[-1]
    assert all(pd.isna(last[name]) for name in mbp1_feature_names())


def test_non_mbp1_bundle_refuses_an_inert_evidence_claim(view):
    envelope, full_frame = _mbp1_evidence_for(view)
    with pytest.raises(ValueError, match="inert evidence claim"):
        resolve_available_bundle_view(
            view,
            "B0_CORE",
            mbp1_features=full_frame,
            mbp1_feature_artifact=envelope,
        )


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
