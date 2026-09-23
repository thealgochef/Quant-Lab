"""Exact baseline readings must not hide nulls, booleans, or numeric units."""

import pytest
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

from alpha_lab.agents.data_infra.ifvg.presentation.axis_values import format_axis_value


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("parent_retest_timeout_1m_bars", 60, "60 1m bars"),
        ("opposing_timeout_1m_bars", 90, "90 1m bars"),
        ("inversion_timeout_1m_bars", 120, "120 1m bars"),
        ("post_inversion_expiry_1m_bars_max", 80, "80 1m bars"),
        ("parent_reaction_window_parent_bars", 40, "40 parent bars"),
        ("htf_registry_max_age_days", 1, "1 day"),
        ("htf_registry_max_age_days", 5, "5 days"),
        ("max_executed_trades_per_day", 1, "1 trade per trading day"),
        ("max_executed_trades_per_day", 3, "3 trades per trading day"),
        ("sl_buffer_ticks", 0, "0 ticks"),
        ("tp_r_multiple", 1.0, "1 R"),
        ("tp_r_multiple", 1.25, "1.25 R"),
        ("enable_shorts", False, "Disabled"),
        ("enable_longs", True, "Enabled"),
        ("parent_retest_timeout_1m_bars", None, "No timeout (unbounded)"),
        ("opposing_timeout_1m_bars", None, "No timeout (unbounded)"),
        ("inversion_timeout_1m_bars", None, "No timeout (unbounded)"),
        ("htf_registry_max_age_days", None, "No age limit"),
        ("max_executed_trades_per_day", None, "No trade cap"),
        ("entry_parent_distance_ticks_max", None, "No distance cap"),
        ("non_runnable_reason", None, "Not set (None)"),
        ("legacy_candidate_row_limit", None, "Not set (None)"),
        ("outside_session_policy", "keep_waiting", "keep_waiting"),
        ("htf_timeframes", ["1H", "4H"], "[one hour, four hours]"),
        (
            "enabled_entry_sessions",
            ("asia", "london", "ny"),
            "Original three windows - 3:00 PM to 12:45 AM, 1:00 AM to 6:00 AM, "
            "7:00 AM to 1:00 PM Chicago time",
        ),
        ("enabled_entry_sessions", (), "[]"),
    ],
)
def test_exact_payload_readings(key, value, expected):
    assert format_axis_value(key, value) == expected


def test_nested_mapping_matches_frozen_mapping_without_losing_keys():
    payload = {"sessions": {"asia": {"start": "16:00", "crosses_midnight": True}}}
    frozen = (("sessions", (("asia", (("start", "16:00"), ("crosses_midnight", True))),)),)
    expected = "{sessions: {asia: {start: 16:00; crosses_midnight: Enabled}}}"
    assert format_axis_value("session_scheme", payload) == expected
    assert format_axis_value("session_scheme", frozen) == expected


def test_composite_members_keep_their_own_units_and_disabled_value():
    payload = (("entry_near_parent", False), ("entry_parent_distance_ticks_max", 40))
    assert format_axis_value("entry_near_parent", payload) == (
        "{entry_near_parent: Disabled; entry_parent_distance_ticks_max: 40 ticks}"
    )


def test_every_actual_baseline_field_has_a_nonempty_reading():
    for key, value in default_ifvg_smc_section().model_dump(mode="json").items():
        assert format_axis_value(key, value).strip(), key
