"""Independent checks reject closure-spanning intervals and invented time fills."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.daily_close_evidence import audit_position_intervals


def inputs(day="2026-01-16", *, overnight=False):
    deadline = pd.Timestamp(f"{day} 15:55", tz="America/Chicago").tz_convert("UTC")
    close = deadline + pd.Timedelta(minutes=5)
    reopening = (
        deadline + pd.Timedelta(days=2, hours=1, minutes=5)
        if deadline.weekday() == 4
        else close + pd.Timedelta(hours=1)
    )
    entry = deadline - pd.Timedelta(minutes=5)
    if overnight:
        entry = deadline - pd.Timedelta(hours=17)
    timestamps = pd.date_range(entry + pd.Timedelta(minutes=1), deadline, freq="min")
    bars = pd.DataFrame(
        {
            "bar_id": [f"one-minute-{i}" for i in range(len(timestamps))],
            "timeframe_seconds": 60,
            "available_ts_utc": timestamps,
            "logical_open_ts_utc": timestamps - pd.Timedelta(minutes=1),
            "logical_close_ts_utc": timestamps,
            "close_ts_utc": timestamps - pd.Timedelta(milliseconds=1),
            "open_ticks": 100,
            "high_ticks": 104,
            "low_ticks": 99,
            "close_ticks": 103,
            "is_complete": True,
            "is_partial": False,
            "close_reason": "complete",
        }
    )
    plan = pd.DataFrame(
        [
            {
                "schedule_id": day,
                "local_date": day,
                "deadline_ts_utc": deadline,
                "market_close_ts_utc": close,
                "reopen_ts_utc": reopening,
            }
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "trade_id": "actual-trade",
                "status": "resolved",
                "resolution": "scheduled_close",
                "entry_ts_utc": entry,
                "resolution_ts_utc": deadline,
                "direction": "LONG",
                "entry_ticks": 100,
                "stop_ticks": 90,
                "target_ticks": 110,
                "exit_ticks": 103,
                "realized_ticks": 3,
                "scheduled_exit_schedule_id": day,
                "scheduled_exit_deadline_ts_utc": deadline,
            }
        ]
    )
    return trades, plan, bars


def test_exact_deadline_and_weekend_lock_are_intervals():
    facts, audit = audit_position_intervals(*inputs())
    assert facts["passed"] and facts["scheduled_exits"] == 1
    assert (
        audit.iloc[0].reopen_ts_utc.tz_convert("America/Chicago").strftime("%a %I:%M %p")
        == "Sun 05:00 PM"
    )
    assert audit.position_interval_violations.sum() == 0


def test_exit_before_four_pm_still_fails_when_it_held_through_weekend():
    trades, plan, bars = inputs()
    trades["resolution_ts_utc"] = pd.Timestamp("2026-01-19 09:00", tz="America/Chicago")
    with pytest.raises(ValueError, match="prohibited position/entry intervals"):
        audit_position_intervals(trades, plan, bars)


def test_open_session_midnight_hold_is_allowed():
    facts, _ = audit_position_intervals(*inputs("2026-01-20", overnight=True))
    assert facts["executions_checked"] == 1


def test_missing_planned_boundary_is_failure_even_when_flat():
    trades, plan, bars = inputs()
    with pytest.raises(ValueError, match="missing required planned exit source coverage"):
        audit_position_intervals(trades.iloc[:0], plan, bars.iloc[:-1])


@pytest.mark.parametrize("field,delta", [
    ("close_ts_utc", pd.Timedelta(0)),
    ("logical_open_ts_utc", -pd.Timedelta(minutes=2)),
    ("logical_close_ts_utc", pd.Timedelta(minutes=1)),
])
def test_boundary_price_requires_exact_complete_half_open_minute(field, delta):
    trades, plan, bars = inputs()
    bars.loc[bars.index[-1], field] = plan.deadline_ts_utc.iloc[0] + delta
    with pytest.raises(ValueError, match="missing required planned exit source coverage"):
        audit_position_intervals(trades, plan, bars)


def test_entry_at_deadline_or_unresolved_position_is_not_compliant():
    trades, plan, bars = inputs()
    trades["entry_ts_utc"] = plan.deadline_ts_utc.iloc[0]
    trades["resolution_ts_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="prohibited position/entry intervals"):
        audit_position_intervals(trades, plan, bars)
    with pytest.raises(ValueError, match="unresolved position"):
        audit_position_intervals(trades.assign(status="open"), plan, bars)


def test_stop_first_at_deadline_takes_precedence_over_target_and_clock():
    trades, plan, bars = inputs()
    bars.loc[bars.index[-1], ["low_ticks", "high_ticks"]] = [89, 111]
    with pytest.raises(ValueError, match="stop-first"):
        audit_position_intervals(trades, plan, bars)
    facts, _ = audit_position_intervals(
        trades.assign(resolution="stop", exit_ticks=90, realized_ticks=-10),
        plan,
        bars,
    )
    assert facts["scheduled_exits"] == 0


@pytest.mark.parametrize("mutation", ["price", "realized", "partial", "future_source", "costs"])
def test_source_or_accounting_tampering_fails(mutation):
    trades, plan, bars = inputs()
    if mutation == "price":
        trades["exit_ticks"] = 104
    elif mutation == "realized":
        trades["realized_ticks"] = 10
    elif mutation == "partial":
        bars.loc[bars.index[-1], "is_partial"] = True
    elif mutation == "future_source":
        bars.loc[bars.index[-1], "close_ts_utc"] += pd.Timedelta(seconds=2)
    else:
        trades["net_r"] = 1.0
    with pytest.raises(ValueError):
        audit_position_intervals(trades, plan, bars)


def test_orphan_orders_and_premature_protection_removal_fail():
    trades, plan, bars = inputs()
    events = pd.DataFrame(
        [
            {
                "event_kind": "deadline_position_closed",
                "position_before": True,
                "position_after": False,
                "trade_id": "actual-trade",
                "pending_entry_order_count": 0,
                "resolution": "scheduled_close",
                "schedule_id": plan.schedule_id.iloc[0],
                "planned_deadline_ts_utc": plan.deadline_ts_utc.iloc[0],
                "actual_exit_ts_utc": plan.deadline_ts_utc.iloc[0],
                "exit_ticks": 103,
                "source_bar_bar_id": bars.bar_id.iloc[-1],
                "entry_lock_after": True,
                "protective_order_disposition": "retained_until_position_resolved_then_removed",
            }
        ]
    )
    facts, _ = audit_position_intervals(trades, plan, bars, forced_events=events)
    assert facts["forced_events_checked"] == 1
    with pytest.raises(ValueError, match="planned deadline coverage"):
        audit_position_intervals(trades, plan, bars, forced_events=events.iloc[:0])
    with pytest.raises(ValueError, match="planned deadline coverage"):
        audit_position_intervals(trades, plan, bars, forced_events=pd.concat([events, events]))
    with pytest.raises(ValueError, match="stale entry order"):
        audit_position_intervals(
            trades, plan, bars, forced_events=events.assign(pending_entry_order_count=1)
        )
    with pytest.raises(ValueError, match="protection"):
        audit_position_intervals(
            trades,
            plan,
            bars,
            forced_events=events.assign(protective_order_disposition="removed_before_exit"),
        )
