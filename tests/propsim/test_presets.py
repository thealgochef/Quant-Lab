"""Preset registry: presets are data; unknown names fail loud."""

from __future__ import annotations

import pytest

from alpha_lab.propsim import PRESETS, ruleset_from_preset


def test_topstep_50k_ratified_parameters():
    rs = ruleset_from_preset("topstep_50k")
    assert rs.starting_balance == 50_000.0
    assert rs.profit_target == 3_000.0
    assert rs.trail_amount == 2_000.0
    assert rs.trail_style == "eod_floor_realtime_breach"
    assert rs.trail_locks_at_start is True
    assert rs.dll_amount == 1_000.0
    assert rs.dll_hard is False  # soft: a DLL touch halts the day
    assert rs.consistency_pct == 50.0
    assert rs.min_days is None
    assert rs.max_eval_days is None
    assert rs.point_value == 20.0


def test_apex_50k_eod_ratified_parameters():
    rs = ruleset_from_preset("apex_50k_eod")
    assert rs.starting_balance == 50_000.0
    assert rs.profit_target == 3_000.0
    assert rs.trail_amount == 2_000.0
    assert rs.trail_style == "eod_floor_realtime_breach"
    assert rs.trail_locks_at_start is True  # ⚠ verify-lock
    assert rs.dll_amount == 1_000.0
    assert rs.dll_hard is True  # ⚠ verify-soft-vs-hard
    assert rs.consistency_pct is None
    assert rs.min_days is None
    assert rs.max_eval_days == 30
    assert rs.point_value == 20.0


def test_apex_50k_intraday_ratified_parameters():
    rs = ruleset_from_preset("apex_50k_intraday")
    assert rs.trail_style == "intraday_peak_trail"
    assert rs.trail_locks_at_start is True  # ⚠ verify-lock
    assert rs.dll_amount is None
    assert rs.consistency_pct is None
    assert rs.max_eval_days == 30
    assert (rs.starting_balance, rs.profit_target, rs.trail_amount) == (
        50_000.0, 3_000.0, 2_000.0,
    )


def test_tpt_50k_test_ratified_parameters():
    rs = ruleset_from_preset("tpt_50k_test")
    assert rs.trail_style == "eod_floor_realtime_breach"
    assert rs.trail_locks_at_start is True
    assert rs.dll_amount is None
    assert rs.consistency_pct == 50.0
    assert rs.min_days == 5
    assert rs.max_eval_days is None  # no expiry
    assert (rs.starting_balance, rs.profit_target, rs.trail_amount) == (
        50_000.0, 3_000.0, 2_000.0,
    )


def test_unverified_apex_parameters_carry_the_dashboard_note():
    """The two ⚠ parameters stay flagged until dashboard-confirmed."""
    from alpha_lab.propsim import presets

    assert "VERIFY AT DASHBOARD" in presets.__doc__
    assert "verify-lock" in presets.__doc__
    assert "verify-soft-vs-hard" in presets.__doc__


def test_unknown_preset_lists_known_names():
    with pytest.raises(ValueError, match="Unknown propsim preset 'nope'") as excinfo:
        ruleset_from_preset("nope")
    for known in sorted(PRESETS):
        assert known in str(excinfo.value)
