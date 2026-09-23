"""Independent source checks must detect changed prices, timing and policy claims."""

from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from strategy_core.candles.exchange_calendar import CME_EQUITY_INDEX_FUTURES_ETH_SCHEDULE
from strategy_core.types import Bar, BarKind, CloseReason

from alpha_lab.agents.data_infra.ifvg.gap_validity_evidence import (
    ORIGINAL,
    OWN_CLOSE,
    validate_gap_validity_events,
)


def _bar(ident, seconds, close, *, start=None):
    start = start or datetime(2026, 2, 23, 15, tzinfo=UTC)
    end = start + timedelta(seconds=seconds)
    return Bar(
        timeframe_ticks=seconds, trading_day=date(2026, 2, 23), bar_index=1,
        bar_id=ident, open_ts_utc=start, close_ts_utc=end - timedelta(seconds=1),
        open_ticks=100050, high_ticks=100090, low_ticks=99990, close_ticks=close,
        volume=1, trade_count=1, is_complete=True, is_partial=False,
        close_reason=CloseReason.COMPLETE, kind=BarKind.TIME,
        logical_open_ts_utc=start, logical_close_ts_utc=end,
    )


def _event(*, policy=OWN_CLOSE, seconds=14400, close=99999, bullish=True):
    source = _bar("source", seconds, close)
    decision = _bar("decision", 60, close,
                    start=source.availability_ts_utc - timedelta(minutes=1))
    beyond = close < 100000 if bullish else close > 100080
    invalidates = policy == OWN_CLOSE and beyond
    row = {
        "event_kind": "own_timeframe_close_check", "fvg_fvg_id": "gap",
        "fvg_role": "htf", "fvg_timeframe_seconds": seconds,
        "fvg_direction": "bullish" if bullish else "bearish",
        "fvg_gap_low_ticks": 100000, "fvg_gap_high_ticks": 100080,
        "fvg_confirmed_ts_utc": source.logical_open_ts_utc - timedelta(hours=1),
        "selected_policy": policy, "validity_before": True,
        "validity_after": not invalidates, "finalization_status": "complete",
        "eligible_own_timeframe_check": True, "wick_traversed": False,
        "original_wick_invalid": False,
        "invalidation_reason": "own_timeframe_close_beyond_far_boundary" if invalidates else None,
        "policy_invalidated_ts_utc": source.availability_ts_utc if invalidates else None,
    }
    for prefix, bar in (("source_bar_", source), ("decision_bar_", decision)):
        row.update({prefix + key: value for key, value in asdict(bar).items()})
    return row, [source, decision]


def _verify(rows, bars):
    return validate_gap_validity_events(
        pd.DataFrame(rows), source_bars=bars,
        schedule=CME_EQUITY_INDEX_FUTURES_ETH_SCHEDULE,
    )


@pytest.mark.parametrize("seconds", [3600, 14400])
@pytest.mark.parametrize("bullish,close,invalidates", [
    (True, 99999, 1), (True, 100000, 0), (True, 100020, 0),
    (False, 100081, 1), (False, 100080, 0), (False, 100020, 0),
])
def test_boundary_rule_independently_matches_prices(seconds, bullish, close, invalidates):
    row, bars = _event(seconds=seconds, bullish=bullish, close=close)
    assert _verify([row], bars)["policy_invalidated_gaps"] == invalidates


def test_original_does_not_adopt_close_rule():
    row, bars = _event(policy=ORIGINAL)
    assert _verify([row], bars)["policy_invalidated_gaps"] == 0


@pytest.mark.parametrize("field,value,error", [
    ("source_bar_close_ticks", 100005, "differs from source"),
    ("fvg_timeframe_seconds", 3600, "another timeframe"),
    ("validity_after", True, "differs from source prices"),
    ("fvg_role", "parent", "another role"),
    ("finalization_status", "not_finalized", "finalization status"),
    ("policy_invalidated_ts_utc", "2026-02-23T17:00:00Z", "invalidation time"),
])
def test_evidence_tampering_fails_closed(field, value, error):
    row, bars = _event()
    row[field] = value
    with pytest.raises(ValueError, match=error):
        _verify([row], bars)


def test_duplicate_close_rejected():
    row, bars = _event(close=100020)
    with pytest.raises(ValueError, match="duplicated"):
        _verify([row, row], bars)


def test_developing_source_cannot_claim_finality():
    row, bars = _event()
    source = replace(bars[0], is_complete=False, is_partial=True,
                     close_reason=CloseReason.END_OF_DAY,
                     close_ts_utc=bars[0].close_ts_utc - timedelta(hours=2))
    row.update({"source_bar_" + key: value for key, value in asdict(source).items()})
    with pytest.raises(ValueError, match="finalization status"):
        _verify([row], [source, bars[1]])


def test_arbitrary_tail_at_bucket_edge_still_needs_session_closure():
    row, bars = _event(seconds=3600)
    source = replace(bars[0], is_complete=False, is_partial=True,
                     close_reason=CloseReason.END_OF_DAY)
    row.update({"source_bar_" + key: value for key, value in asdict(source).items()})
    row["finalization_status"] = "finalized_session_tail"
    with pytest.raises(ValueError, match="finalization status"):
        _verify([row], [source, bars[1]])


def test_no_future_close_or_formation_reuse():
    row, bars = _event()
    row["fvg_confirmed_ts_utc"] = bars[0].availability_ts_utc
    with pytest.raises(ValueError, match="formation or past"):
        _verify([row], bars)
    row, bars = _event()
    decision = replace(
        bars[1], logical_close_ts_utc=bars[1].availability_ts_utc - timedelta(minutes=1),
    )
    row.update({"decision_bar_" + key: value for key, value in asdict(decision).items()})
    with pytest.raises(ValueError, match="unavailable"):
        _verify([row], [bars[0], decision])


def test_own_close_terminal_counter_reconciles():
    from alpha_lab.agents.data_infra.ifvg.audit_contracts import (
        AuditTable,
        reconcile_funnel_to_audit,
    )

    reason = "invalidated_htf_own_timeframe_close"
    tables = {AuditTable.SLOT_DEATH: pd.DataFrame([{"death_reason": reason}])}
    assert reconcile_funnel_to_audit({"2026-02-23": {reason: 1}}, tables)["passed"]
    with pytest.raises(ValueError, match=reason):
        reconcile_funnel_to_audit({}, tables)


@pytest.mark.parametrize("policy", [ORIGINAL, OWN_CLOSE])
def test_real_capture_flattening_validates_complete_chronological_chain(policy):
    from types import SimpleNamespace

    from strategy_core.strategies.ifvg_smc.replay import run_day
    from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

    from alpha_lab.agents.data_infra.ifvg.capture_driver import flatten_audit_emissions
    from alpha_lab.agents.data_infra.ifvg.gap_validity_evidence import build_gap_validity_evidence
    from tests.agents.test_ifvg_fsm_audit_contracts import _DAY0, _three_days

    section = default_ifvg_smc_section()
    if "htf_gap_invalidation_policy" not in type(section).model_fields:
        pytest.skip("requires the process-local study Core")
    section = section.model_copy(update={"htf_gap_invalidation_policy": policy})
    audit_frames, bars_by_day, seed = {}, {}, None
    for index, by_tf in enumerate(_three_days()):
        day = _DAY0 + timedelta(days=index)
        result = run_day(
            by_tf, section=section, seed=seed, trading_day=day,
            dataset_exhausted=index == 2, audit_capture_mode="fsm_audit_v1",
        )
        seed = result.end_seed
        audit_frames[day] = flatten_audit_emissions(result.audit_emissions, entering_seed_hash=None)
        bars_by_day[day] = [bar for group in by_tf.values() for bar in group]
    tables, receipt = build_gap_validity_evidence(
        SimpleNamespace(audit_frames=audit_frames, bars_by_day=bars_by_day),
    )
    assert receipt["passed"] and receipt["events"] > 0
    assert not tables["gap_validity_events"].empty
