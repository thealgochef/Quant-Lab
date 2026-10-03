"""A1 registry names and atomic entry-schedule payloads."""

import pytest

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    CompositeAxisValue,
    resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import axis_group_for


def test_menthorq_values_and_slot_presets_follow_existing_registry_contract():
    expected = {
        "menthorq_context_version": {"none": None, "eod_v1": "menthorq_eod_v1"},
        "regime_gate_policy": {
            "off": "off", "positive_only": "positive_only", "negative_only": "negative_only",
        },
        "regime_unknown_policy": {"allow": "allow", "block": "block"},
        "nearest_support_gex1_block": {"false": False, "true": True},
        "nearest_support_universe": {"all_19": "all_19", "studied_8": "studied_8"},
    }
    for key, values in expected.items():
        assert len(SEARCH_AXIS_REGISTRY_V1[key].registered_values) == len(values)
        for token, payload in values.items():
            value = AXIS_VALUE_REGISTRY_V1[f"{key}.{token}"]
            assert value.payload == payload
            assert (value.capability_status, value.owner_ratification_status) == (
                "available", "pending",
            )
    for token, window in (
        ("slot_s1_0830_1000", ("08:30", "10:00")),
        ("slot_s2_1000_1200", ("10:00", "12:00")),
        ("slot_s3_1200_1330", ("12:00", "13:30")),
        ("slot_s4_1330_1510", ("13:30", "15:10")),
        ("midsession_1000_1330", ("10:00", "13:30")),
    ):
        value = AXIS_VALUE_REGISTRY_V1[f"enabled_entry_sessions.{token}"]
        assert isinstance(value, CompositeAxisValue)
        assert value.member_mapping() == {
            "enabled_entry_sessions": ("asia", "london", "ny"),
            "entry_schedule_policy": "explicit_windows_v1",
            "entry_schedule_timezone": "America/Chicago",
            "entry_schedule_windows": (window,),
        }


def test_nearest_support_universe_is_context_dependent_and_uses_existing_risk_group():
    key = "nearest_support_universe"
    spec = SEARCH_AXIS_REGISTRY_V1[key]
    assert spec.dependencies == ("menthorq_context_version",)
    assert spec.baseline_value_id == f"{key}.all_19"
    assert spec.requires_full_sequential_replay
    assert not spec.changes_capture_artifacts
    assert axis_group_for(key) == "Risk Admissibility"
    overrides = resolve_axis_overrides({key: f"{key}.studied_8"})
    with pytest.raises(ValueError, match="menthorq_gate_requires_eod_context"):
        resolve_profile_config({"section_overrides": overrides})
    overrides.update(resolve_axis_overrides({
        "menthorq_context_version": "menthorq_context_version.eod_v1",
    }))
    resolved = resolve_profile_config({"section_overrides": overrides})
    assert resolved.section.nearest_support_universe == "studied_8"
    assert resolved.effective_config[key] == "studied_8"
