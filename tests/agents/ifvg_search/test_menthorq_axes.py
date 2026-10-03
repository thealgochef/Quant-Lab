"""A1 registry names and atomic entry-schedule payloads."""

from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    CompositeAxisValue,
)


def test_menthorq_values_and_slot_presets_follow_existing_registry_contract():
    expected = {
        "menthorq_context_version": {"none": None, "eod_v1": "menthorq_eod_v1"},
        "regime_gate_policy": {
            "off": "off", "positive_only": "positive_only", "negative_only": "negative_only",
        },
        "regime_unknown_policy": {"allow": "allow", "block": "block"},
        "nearest_support_gex1_block": {"false": False, "true": True},
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
