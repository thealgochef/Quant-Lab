"""Typed axis-value registry suites (TEST_MATRIX §3.1; P0-5/P0-20)."""

from __future__ import annotations

import pytest
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    AxisAuthorizationError,
    AxisClassification,
    CompositeAxisValue,
    assert_axes_authorized,
    registry_sha256,
    resolve_axis_overrides,
)


def test_registry_covers_every_section_field_except_profile_name() -> None:
    section_fields = set(IfvgSmcSection.model_fields) - {"profile_name"}
    assert set(SEARCH_AXIS_REGISTRY_V1) == section_fields


def test_typed_values_round_trip_none_and_tuples() -> None:
    none_value = AXIS_VALUE_REGISTRY_V1["parent_retest_timeout_1m_bars.none"]
    assert none_value.payload is None
    assert none_value.capability_status == "available"
    bounded = AXIS_VALUE_REGISTRY_V1["parent_retest_timeout_1m_bars.480"]
    assert bounded.payload == 480
    sessions = SEARCH_AXIS_REGISTRY_V1["enabled_entry_sessions"]
    baseline = AXIS_VALUE_REGISTRY_V1[sessions.baseline_value_id]
    assert tuple(baseline.payload) == ("asia", "london", "ny")
    reloaded = type(baseline).model_validate(baseline.model_dump(mode="json"))
    assert reloaded.model_dump(mode="json") == baseline.model_dump(mode="json")


def test_unregistered_value_is_refused() -> None:
    with pytest.raises(AxisAuthorizationError, match="not registered"):
        assert_axes_authorized(
            {"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.999"},
            require_ratified=False,
        )


def test_pending_ratification_refused_for_real_but_accepted_synthetic() -> None:
    ids = {"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.480"}
    with pytest.raises(AxisAuthorizationError, match="ratification"):
        assert_axes_authorized(ids, require_ratified=True)
    assert_axes_authorized(ids, require_ratified=False)  # synthetic marker path


def test_inert_and_legacy_fields_are_blocked_before_enumeration() -> None:
    for axis in (
        "break_even_enabled",
        "legacy_candidate_row_limit",
        "parent_reaction_window_1m_bars_max",
        "resolver_policy",
        "retest_trigger",
    ):
        spec = SEARCH_AXIS_REGISTRY_V1[axis]
        assert spec.classification is AxisClassification.BLOCKED
        with pytest.raises(AxisAuthorizationError, match="not searchable"):
            assert_axes_authorized(
                {axis: spec.baseline_value_id}, require_ratified=False
            )


def test_parent_fill_axes_blocked_pending_owner_policy_review() -> None:
    for axis in ("parent_full_fill_invalidation", "parent_structural_invalidation"):
        spec = SEARCH_AXIS_REGISTRY_V1[axis]
        value = AXIS_VALUE_REGISTRY_V1[spec.baseline_value_id]
        assert value.capability_status == "blocked_pending_owner_policy_review"
        with pytest.raises(AxisAuthorizationError):
            assert_axes_authorized({axis: spec.baseline_value_id}, require_ratified=False)


def test_locked_invariants_are_never_searchable() -> None:
    for axis in ("runnable", "execution_enabled", "anchor_policy", "causality_entry"):
        assert (
            SEARCH_AXIS_REGISTRY_V1[axis].classification
            is AxisClassification.LOCKED_INVARIANT
        )
        with pytest.raises(AxisAuthorizationError):
            assert_axes_authorized(
                {axis: SEARCH_AXIS_REGISTRY_V1[axis].baseline_value_id},
                require_ratified=False,
            )


def test_searchable_baselines_have_a_uniform_pending_posture() -> None:
    # owner decision 2: EVERY value of a searchable axis — including the
    # accepted doc-default baseline — needs value-level ratification.
    for key, spec in SEARCH_AXIS_REGISTRY_V1.items():
        if spec.classification is not AxisClassification.APPROVED_SEARCH_AXIS:
            continue
        baseline = AXIS_VALUE_REGISTRY_V1[spec.baseline_value_id]
        if baseline.capability_status == "available":
            assert baseline.owner_ratification_status == "pending", key


def test_value_bound_to_another_axis_is_refused() -> None:
    # the registered-values check fires first in authorization; the expansion
    # helper independently verifies axis-key binding (defense in depth)
    with pytest.raises(AxisAuthorizationError, match="not registered for axis|belongs to"):
        assert_axes_authorized(
            {"tp_r_multiple": "enable_shorts.true"}, require_ratified=False
        )
    with pytest.raises(AxisAuthorizationError, match="belongs to"):
        resolve_axis_overrides({"tp_r_multiple": "enable_shorts.true"})


def test_dependent_composite_member_is_not_standalone_searchable() -> None:
    spec = SEARCH_AXIS_REGISTRY_V1["entry_parent_distance_ticks_max"]
    assert spec.classification is AxisClassification.MEASUREMENT_ONLY
    with pytest.raises(AxisAuthorizationError, match="not searchable"):
        assert_axes_authorized(
            {"entry_parent_distance_ticks_max": spec.baseline_value_id},
            require_ratified=False,
        )


def test_composite_group_resolves_atomically() -> None:
    value = AXIS_VALUE_REGISTRY_V1["entry_near_parent.within_40"]
    assert isinstance(value, CompositeAxisValue)
    overrides = resolve_axis_overrides({"entry_near_parent": "entry_near_parent.within_40"})
    assert overrides == {"entry_near_parent": True, "entry_parent_distance_ticks_max": 40}


def test_resolved_overrides_validate_into_a_section() -> None:
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config

    overrides = resolve_axis_overrides(
        {"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.480"}
    )
    resolved = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": overrides,
        }
    )
    assert resolved.section.parent_retest_timeout_1m_bars == 480
    assert resolved.section.qualification_mode.value == "custom_profile"


def test_timeframe_axes_change_capture_artifacts() -> None:
    for axis in ("htf_timeframes", "parent_timeframes"):
        assert SEARCH_AXIS_REGISTRY_V1[axis].changes_capture_artifacts is True
    assert SEARCH_AXIS_REGISTRY_V1["sl_buffer_ticks"].changes_capture_artifacts is False


def test_registry_hash_is_deterministic_and_sensitive() -> None:
    assert registry_sha256() == registry_sha256()
    trimmed_axes = dict(SEARCH_AXIS_REGISTRY_V1)
    trimmed_axes.pop("sl_buffer_ticks")
    assert registry_sha256(axes=trimmed_axes) != registry_sha256()


def test_ui_never_gets_raw_section_overrides() -> None:
    # the ONLY expansion route is registered value ids; a raw dict is refused
    with pytest.raises(AxisAuthorizationError):
        resolve_axis_overrides({"parent_retest_timeout_1m_bars": "raw:{'x':1}"})
