"""Task A1 pure values, identity and final-candidate evidence gates."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

import test_ifvg_v2_characterization as script
from strategy_core.strategies.ifvg_smc.menthorq_levels import (
    LEVEL_COLUMN_NAMES,
    MenthorqLevelSnapshot,
    derive_menthorq_values,
    evaluate_menthorq_entry_gates,
    slot_chicago_for,
)
from strategy_core.strategies.ifvg_smc.reducer import IfvgReducer, IfvgReducerConfig
from strategy_core.strategies.ifvg_smc.replay import DayOrchestrator, IfvgLevelInputs
from strategy_core.strategies.ifvg_smc.section import (
    MENTHORQ_NEUTRAL_PROFILE_FIELDS,
    default_ifvg_smc_section,
    ifvg_profile_hash,
)
from strategy_core.structures.fvg import GapDirection

CT = ZoneInfo("America/Chicago")


def ct(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=CT).astimezone(UTC)


def snapshot(**changes) -> MenthorqLevelSnapshot:
    values = dict(
        trading_date=date(2026, 1, 13),
        source_eod_date=date(2026, 1, 12),
        source_file_sha256="a" * 64,
        levels={**dict.fromkeys(LEVEL_COLUMN_NAMES), "HVL": 2504., "GEX 1": 2500.},
        implied_move_points=100.,
        regime="positive",
        context_available=True,
    )
    values.update(changes)
    return MenthorqLevelSnapshot(**values)


def section(**changes):
    base = default_ifvg_smc_section()
    return base.model_validate({**base.model_dump(), **changes})


@pytest.mark.parametrize("day", ["2026-01-13", "2026-03-08"])
@pytest.mark.parametrize(("wall", "expected"), [
    ("08:29:59", "outside_cash"), ("08:30:00", "s1_0830_1000"),
    ("10:00:00", "s2_1000_1200"), ("12:00:00", "s3_1200_1330"),
    ("13:30:00", "s4_1330_1510"), ("15:10:00", "outside_cash"),
])
def test_slot_half_open_boundaries_and_dst(day, wall, expected):
    assert slot_chicago_for(ct(f"{day}T{wall}")) == expected


def test_snapshot_is_immutable_and_nearest_ties_preserve_order_and_equality():
    levels = {**dict.fromkeys(LEVEL_COLUMN_NAMES), "Put Support": 2500., "HVL": 2504.,
              "GEX 1": 2500., "GEX 2": 2508., "GEX 3": 2508.}
    snap = snapshot(levels=levels)
    levels["GEX 1"] = 2490.
    with pytest.raises(TypeError):
        snap.levels["GEX 1"] = 2490.
    values = derive_menthorq_values(snap, 2504., ct("2026-01-13T10:00"),
                                   bar_open_points=2500., prior_cash_close_points=2510.)
    assert values.hvl_side == "at"
    assert values.nearest_below_names == "Put Support|GEX 1"
    assert values.nearest_below_points == 2500.
    assert values.nearest_below_distance_points == 4.
    assert values.nearest_below_distance_implied == .04
    assert values.nearest_support_is_gex1 is True
    assert values.nearest_above_names == "GEX 2|GEX 3"
    assert values.nearest_above_points == 2508.
    assert values.nearest_above_distance_points == 4.
    assert values.nearest_above_distance_implied == .04
    assert values.opening_move_signed == -.1 and values.opening_move_abs == .1
    equal = derive_menthorq_values(snap, 2500., ct("2026-01-13T10:00"))
    assert equal.hvl_side == "below"
    assert equal.nearest_below_names is None and equal.nearest_support_is_gex1 is None
    assert equal.nearest_above_names == "HVL"
    assert derive_menthorq_values(snap, 2510., ct("2026-01-13T10:00")).hvl_side == "above"


@pytest.mark.parametrize("implied", [None, 0., -1.])
def test_null_or_nonpositive_implied_move_and_absent_prior_close(implied):
    values = derive_menthorq_values(snapshot(implied_move_points=implied), 2504.,
                                   ct("2026-01-13T10:00"), bar_open_points=2500.,
                                   prior_cash_close_points=2490.)
    assert values.nearest_below_distance_points == 4.
    assert values.nearest_below_distance_implied is None
    assert values.opening_move_signed is None and values.opening_move_abs is None
    absent = derive_menthorq_values(snapshot(), 2504., ct("2026-01-13T10:00"),
                                   bar_open_points=2500.)
    assert absent.opening_move_signed is None and absent.opening_move_abs is None


def test_unavailable_context_and_no_below_remain_null():
    snap = snapshot(context_available=False, unavailable_reason="no_level_row")
    values = derive_menthorq_values(snap, 2504., ct("2026-01-13T10:00"))
    assert values.slot_chicago == "s2_1000_1200"
    assert values.hvl_side is None and values.nearest_support_is_gex1 is None
    gates = evaluate_menthorq_entry_gates(snapshot(levels={"HVL": 2504.}), 2504.,
                                         ct("2026-01-13T10:00"), nearest_support_gex1_block=True)
    assert gates.block_reasons == () and gates.nearest_support_gate_blocked is None


def test_neutral_profile_hash_and_each_nondefault_field():
    base = default_ifvg_smc_section()
    assert ifvg_profile_hash(base) == "e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd"
    historical = base.model_dump()
    for key in MENTHORQ_NEUTRAL_PROFILE_FIELDS:
        historical.pop(key)
    assert ifvg_profile_hash(base.model_validate(historical)) == ifvg_profile_hash(base)
    context = section(menthorq_context_version="menthorq_eod_v1")
    assert ifvg_profile_hash(context) != ifvg_profile_hash(base)
    with pytest.raises(ValueError, match="unsupported_menthorq_context_version"):
        section(menthorq_context_version="unsupported")
    for key, value in (("regime_gate_policy", "positive_only"),
                       ("regime_unknown_policy", "block"),
                       ("nearest_support_gex1_block", True)):
        changed = context.model_validate({**context.model_dump(), key: value})
        assert ifvg_profile_hash(changed) != ifvg_profile_hash(context)
        with pytest.raises(ValueError, match="menthorq_gate_requires_eod_context"):
            section(**{key: value})


def final_entry(monkeypatch, *, start="2026-01-13T09:54", snap=None, session="ny", **changes):
    monkeypatch.setattr(script, "_T0", ct(start))
    monkeypatch.setattr(script, "_DAY", ct(start).astimezone(CT).date())
    cfg = IfvgReducerConfig.from_section(
        section(menthorq_context_version="menthorq_eod_v1", **changes),
        tick_size=.25, strategy_id="ifvg_smc", strategy_version="2",
    )
    reducer = IfvgReducer(cfg)
    script._drive_to_inversion(reducer)
    bar = script._bar(5, 10015, 10019, 9985, 10016)
    gap = script._fvg(60, GapDirection.BULLISH, 10014, 10015,
                      confirmed=bar.availability_ts_utc, ident="menthorq-entry")
    step = replace(script._step(bar, new_fvgs={60: (gap,)}, session_doc=session), menthorq=snap)
    return reducer, reducer.step(step), bar


@pytest.mark.parametrize(("regime", "policy", "support", "unknown_policy", "available", "expected"), [
    ("negative", "positive_only", False, "allow", True, ("regime_gate",)),
    ("positive", "negative_only", False, "allow", True, ("regime_gate",)),
    ("positive", "positive_only", False, "allow", True, ()),
    ("positive", "off", True, "allow", True, ("nearest_support_gex1",)),
    ("negative", "positive_only", True, "allow", True, ("regime_gate", "nearest_support_gex1")),
    ("unknown", "positive_only", False, "allow", True, ()),
    ("unknown", "positive_only", False, "block", True, ("context_unavailable",)),
    ("positive", "positive_only", True, "block", False, ("context_unavailable",)),
    ("positive", "positive_only", True, "allow", False, ()),
])
def test_reducer_gates_keep_blocked_candidate_geometry_and_no_decision(
    monkeypatch, regime, policy, support, unknown_policy, available, expected,
):
    snap = snapshot(regime=regime, context_available=available,
                    unavailable_reason=None if available else "no_level_row")
    reducer, emissions, _ = final_entry(monkeypatch, snap=snap, regime_gate_policy=policy,
                                       regime_unknown_policy=unknown_policy,
                                       nearest_support_gex1_block=support)
    candidate = next(e.record for e in emissions if e.kind == "entry_candidate"
                     and e.record.entry_family == "fresh_fvg_continuation")
    assert candidate.block_reasons == expected
    assert candidate.geometry is not None and candidate.entry_ticks == 10016
    assert candidate.trigger_evidence_id and candidate.trigger_cursor
    decisions = [e for e in emissions if e.kind == "eligible_decision"]
    assert bool(decisions) is not bool(expected)
    assert reducer.phase == ("S4" if expected else "S5")
    if expected:
        assert any(e.kind == "geometry_dossier" and e.record.candidate_id == candidate.candidate_id
                   for e in emissions)


def test_asia_entry_unknown_block_policy_does_not_apply_outside_hours(monkeypatch):
    snap = snapshot(context_available=False, unavailable_reason="after_1700", regime="unknown")
    _, emissions, bar = final_entry(monkeypatch, start="2026-01-12T17:54", session="asia",
                                    snap=snap, regime_gate_policy="positive_only",
                                    regime_unknown_policy="block", nearest_support_gex1_block=True)
    candidate = next(e.record for e in emissions if e.kind == "entry_candidate"
                     and e.record.entry_family == "fresh_fvg_continuation")
    assert candidate.in_doc_session == "asia" and candidate.block_reasons == ()
    assert any(e.kind == "eligible_decision" for e in emissions)
    exported = evaluate_menthorq_entry_gates(snap, candidate.entry_ticks * .25,
                                            bar.availability_ts_utc, regime_gate_policy="positive_only",
                                            regime_unknown_policy="block", nearest_support_gex1_block=True)
    assert exported.regime_gate_blocked is None and exported.nearest_support_gate_blocked is None
    assert exported.gate_status == "not_applicable_outside_hours"


def test_shorts_enabled_bypasses_both_gates_and_session_reasons_stay_first(monkeypatch):
    snap = snapshot(regime="negative")
    _, emissions, bar = final_entry(monkeypatch, snap=snap, enable_shorts=True,
                                    regime_gate_policy="positive_only", nearest_support_gex1_block=True)
    assert any(e.kind == "eligible_decision" for e in emissions)
    gates = evaluate_menthorq_entry_gates(snap, 2504., bar.availability_ts_utc,
                                         enable_shorts=True, regime_gate_policy="positive_only",
                                         nearest_support_gex1_block=True)
    assert gates.gate_status == "not_applicable_shorts_enabled"
    assert gates.regime_gate_blocked is None and gates.nearest_support_gate_blocked is None
    _, blocked, _ = final_entry(monkeypatch, snap=snap, enabled_entry_sessions=("asia",),
                                regime_gate_policy="positive_only", nearest_support_gex1_block=True)
    candidate = next(e.record for e in blocked if e.kind == "entry_candidate"
                     and e.record.entry_family == "fresh_fvg_continuation")
    assert candidate.block_reasons[:3] == ("out_of_session", "regime_gate", "nearest_support_gex1")


def test_existing_level_handoff_supplies_snapshot_at_actual_availability(monkeypatch):
    seen = []
    snap = snapshot()
    monkeypatch.setattr(script, "_T0", ct("2026-01-13T09:54"))
    bar = script._bar(0, 10030, 10032, 10015, 10028)
    def levels_for(ts):
        seen.append(ts)
        return IfvgLevelInputs((), snap)
    orch = DayOrchestrator(section=section(menthorq_context_version="menthorq_eod_v1"),
                           seed=None, levels_for=levels_for)
    inputs = []
    monkeypatch.setattr(orch._reducer, "step", lambda inp: inputs.append(inp) or ())
    orch.on_decision_bar(bar)
    assert seen == [bar.availability_ts_utc]
    assert inputs[0].menthorq is snap and inputs[0].levels == ()
