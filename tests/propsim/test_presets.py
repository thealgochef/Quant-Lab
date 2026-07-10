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
    assert rs.dll_soft is True
    assert rs.consistency_pct == 50.0
    assert rs.min_days is None
    assert rs.point_value == 20.0


def test_unknown_preset_lists_known_names():
    with pytest.raises(ValueError, match="Unknown propsim preset 'nope'") as excinfo:
        ruleset_from_preset("nope")
    for known in sorted(PRESETS):
        assert known in str(excinfo.value)
