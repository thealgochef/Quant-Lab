"""Independent source-price and position-interval audit for daily-flat research.

This module never calls the engine's deadline or execution predicate. Its inputs
are the frozen planned calendar and original one-minute bars, not inferred daily
last rows. Historical captures without this policy remain readable separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _utc(series, field):
    if series.isna().any():
        raise ValueError(f"daily-close audit requires non-null {field}")
    for value in series:
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError(f"daily-close audit requires timezone-aware {field}")
    return pd.to_datetime(series, utc=True, errors="raise")


def audit_position_intervals(trades, schedule, source_bars, *, forced_events=None):
    """Return (facts, closure rows); raise on missing evidence or violations.

    Lock intervals are [planned deadline, legal reopening). A position may exit
    at the deadline, but may not enter there. Every planned deadline requires
    its actual complete one-minute price even when that profile was already flat.
    This establishes historical replay compliance, not broker delivery guarantees.
    """
    required_plan = {"schedule_id", "deadline_ts_utc", "market_close_ts_utc", "reopen_ts_utc"}
    if missing := required_plan - set(schedule):
        raise ValueError(f"daily-close schedule lacks {sorted(missing)}")
    plan = schedule.copy()
    if plan.empty or plan.schedule_id.isna().any() or plan.schedule_id.duplicated().any():
        raise ValueError("daily-close schedule must contain unique planned identities")
    for column in ("deadline_ts_utc", "market_close_ts_utc", "reopen_ts_utc"):
        plan[column] = _utc(plan[column], column)
    if not (
        (plan.deadline_ts_utc < plan.market_close_ts_utc)
        & (plan.market_close_ts_utc < plan.reopen_ts_utc)
    ).all():
        raise ValueError("invalid planned deadline/closure/reopening order")
    plan = plan.sort_values("deadline_ts_utc").reset_index(drop=True)
    if (plan.deadline_ts_utc < plan.reopen_ts_utc.shift()).fillna(False).any():
        raise ValueError("daily-close schedule contains overlapping lock intervals")

    bars = source_bars.copy()
    timeframe = "timeframe_seconds" if "timeframe_seconds" in bars else "timeframe_ticks"
    bars = bars.loc[bars[timeframe].eq(60)].copy()
    available = "available_ts_utc" if "available_ts_utc" in bars else "logical_close_ts_utc"
    for column in (available, "close_ts_utc", "logical_open_ts_utc", "logical_close_ts_utc"):
        bars[column] = _utc(bars[column], column)
    if bars.bar_id.duplicated().any() or bars[available].duplicated().any():
        raise ValueError("daily-close source has duplicated one-minute identities")
    bars = bars.sort_values(available).set_index(available, drop=False)

    work = trades.copy()
    required_trade = {
        "trade_id",
        "status",
        "resolution",
        "entry_ts_utc",
        "resolution_ts_utc",
        "entry_ticks",
        "stop_ticks",
        "target_ticks",
        "realized_ticks",
        "direction",
    }
    if missing := required_trade - set(work):
        raise ValueError(f"daily-close trades lack {sorted(missing)}")
    if work.trade_id.isna().any() or work.trade_id.duplicated().any():
        raise ValueError("daily-close trades repeat or omit an execution identity")
    if not work.status.eq("resolved").all():
        raise ValueError("daily-close replay has an unresolved position")
    for column in ("entry_ts_utc", "resolution_ts_utc"):
        work[column] = _utc(work[column], column)
    if not (work.resolution_ts_utc > work.entry_ts_utc).all():
        raise ValueError("daily-close replay has invalid position intervals")
    ordered = work.sort_values("entry_ts_utc")
    if (ordered.entry_ts_utc <= ordered.resolution_ts_utc.shift()).fillna(False).any():
        raise ValueError("daily-close replay has overlapping positions")

    rows = []
    for item in plan.to_dict("records"):
        deadline, reopening = item["deadline_ts_utc"], item["reopen_ts_utc"]
        coverage = deadline in bars.index
        source = bars.loc[deadline] if coverage else None
        if coverage:
            coverage = bool(
                source.is_complete
                and not source.is_partial
                and source.close_reason == "complete"
                and source.logical_open_ts_utc == deadline - pd.Timedelta(minutes=1)
                and source.logical_close_ts_utc == deadline
                and deadline - pd.Timedelta(minutes=1) <= source.close_ts_utc < deadline
            )
        prohibited = work.loc[(work.entry_ts_utc < reopening) & (work.resolution_ts_utc > deadline)]
        locked_entries = work.loc[(work.entry_ts_utc >= deadline) & (work.entry_ts_utc < reopening)]
        rows.append(
            {
                **item,
                "deadline_chicago": deadline.tz_convert("America/Chicago").strftime(
                    "%B %d, %Y %I:%M %p %Z"
                ),
                "reopen_chicago": reopening.tz_convert("America/Chicago").strftime(
                    "%B %d, %Y %I:%M %p %Z"
                ),
                "source_coverage": "complete_boundary_minute"
                if coverage
                else "missing_or_incomplete_boundary_minute",
                "deadline_source_bar_id": source.bar_id if source is not None else None,
                "position_interval_violations": len(prohibited),
                "late_entry_violations": len(locked_entries),
                "violating_trade_ids": tuple(prohibited.trade_id.astype(str)),
                "scheduled_exit_count": int(
                    (
                        work.resolution.eq("scheduled_close") & work.resolution_ts_utc.eq(deadline)
                    ).sum()
                ),
                "flat_interval_start_utc": deadline,
                "flat_interval_end_utc": reopening,
            }
        )
    audit = pd.DataFrame(rows)
    if not audit.source_coverage.eq("complete_boundary_minute").all():
        failed = audit.loc[
            ~audit.source_coverage.eq("complete_boundary_minute"), "schedule_id"
        ].tolist()
        raise ValueError(f"missing required planned exit source coverage: {failed[:8]}")
    if audit.position_interval_violations.sum() or audit.late_entry_violations.sum():
        failed = audit.loc[
            (audit.position_interval_violations + audit.late_entry_violations).gt(0), "schedule_id"
        ].tolist()
        raise ValueError(
            f"prohibited position/entry intervals across planned closures: {failed[:8]}"
        )

    schedules = plan.set_index("schedule_id")
    scheduled_ids = set()
    for row in work.itertuples(index=False):
        path = bars.loc[(bars.index > row.entry_ts_utc) & (bars.index <= row.resolution_ts_utc)]
        if path.empty or row.resolution_ts_utc not in path.index:
            raise ValueError(f"exit lacks actual one-minute source: {row.trade_id}")
        long = str(row.direction).lower() == "long"
        if str(row.direction).lower() not in {"long", "short"}:
            raise ValueError("unknown execution direction")
        stop = path.low_ticks.le(row.stop_ticks) if long else path.high_ticks.ge(row.stop_ticks)
        target = (
            path.high_ticks.ge(row.target_ticks) if long else path.low_ticks.le(row.target_ticks)
        )
        hits = path.loc[stop | target]
        if not hits.empty:
            first_time = hits.index[0]
            actual_reason = "stop" if bool(stop.loc[first_time]) else "target"
            actual_price = row.stop_ticks if actual_reason == "stop" else row.target_ticks
            if row.resolution != actual_reason or row.resolution_ts_utc != first_time:
                raise ValueError(
                    f"stop-first source replay disagrees with execution: {row.trade_id}"
                )
        else:
            if row.resolution != "scheduled_close":
                raise ValueError(f"protective exit has no source touch: {row.trade_id}")
            schedule_id = getattr(row, "scheduled_exit_schedule_id", None)
            if schedule_id not in schedules.index:
                raise ValueError("scheduled exit has no frozen planned calendar identity")
            expected_deadline = schedules.loc[schedule_id, "deadline_ts_utc"]
            declared = pd.Timestamp(getattr(row, "scheduled_exit_deadline_ts_utc", None))
            if (
                declared.tzinfo is None
                or declared != expected_deadline
                or row.resolution_ts_utc != expected_deadline
            ):
                raise ValueError("scheduled exit does not occur at its predeclared deadline")
            actual_price = path.loc[expected_deadline, "close_ticks"]
            scheduled_ids.add(str(row.trade_id))
        if getattr(row, "exit_ticks", actual_price) != actual_price:
            raise ValueError("actual exit price differs from source execution convention")
        realized = (actual_price - row.entry_ticks) * (1 if long else -1)
        if row.realized_ticks != realized:
            raise ValueError("realized price movement differs from actual exit")
        if hasattr(row, "net_r"):
            expected = (realized * 0.25 - 0.514) / (abs(row.entry_ticks - row.stop_ticks) * 0.25)
            if not np.isclose(row.net_r, expected, rtol=0, atol=1e-10):
                raise ValueError("exit costs or normalized outcome disagree")

    events_checked = 0
    if forced_events is not None:
        events = forced_events.copy()
        if (
            "schedule_id" not in events
            or events.schedule_id.duplicated().any()
            or set(events.schedule_id.astype(str)) != set(plan.schedule_id.astype(str))
        ):
            raise ValueError("mandatory-close events omit or duplicate planned deadline coverage")
        if not events.empty:
            required = {
                "event_kind",
                "position_before",
                "position_after",
                "trade_id",
                "pending_entry_order_count",
                "protective_order_disposition",
                "resolution",
                "schedule_id",
                "planned_deadline_ts_utc",
                "actual_exit_ts_utc",
                "exit_ticks",
                "source_bar_bar_id",
                "entry_lock_after",
            }
            if missing := required - set(events):
                raise ValueError(f"forced-exit evidence lacks {sorted(missing)}")
            closed = events.loc[events.event_kind.eq("deadline_position_closed")]
            scheduled = closed.loc[closed.resolution.eq("scheduled_close")]
            if (
                closed.trade_id.duplicated().any()
                or set(scheduled.trade_id.astype(str)) != scheduled_ids
            ):
                raise ValueError("forced-exit evidence omits or duplicates a scheduled execution")
            if (
                closed.position_after.astype(bool).any()
                or not closed.position_before.astype(bool).all()
            ):
                raise ValueError("forced exit did not confirm flat position state")
            if events.pending_entry_order_count.ne(0).any():
                raise ValueError("forced exit retained a stale entry order")
            if not closed.protective_order_disposition.eq(
                "retained_until_position_resolved_then_removed"
            ).all():
                raise ValueError("forced exit removed protection before position closure")
            if not events.entry_lock_after.astype(bool).all():
                raise ValueError("deadline did not lock new entries")
            executions = work.set_index("trade_id")
            for event in events.to_dict("records"):
                if event["schedule_id"] not in schedules.index:
                    raise ValueError("forced-exit event has an unknown planned schedule")
                planned = schedules.loc[event["schedule_id"], "deadline_ts_utc"]
                if pd.Timestamp(event["planned_deadline_ts_utc"]) != planned:
                    raise ValueError("forced-exit event changed its planned deadline")
                if event["source_bar_bar_id"] != bars.loc[planned, "bar_id"]:
                    raise ValueError("forced-exit event changed its boundary source candle")
                if event["event_kind"] == "deadline_position_closed":
                    trade = executions.loc[event["trade_id"]]
                    if (
                        trade.resolution_ts_utc != planned
                        or pd.Timestamp(event["actual_exit_ts_utc"]) != planned
                        or event["resolution"] != trade.resolution
                        or event["exit_ticks"] != trade.exit_ticks
                    ):
                        raise ValueError("forced-exit lifecycle differs from the actual execution")
                elif event["event_kind"] == "deadline_flat_lock":
                    if bool(event["position_before"]) or bool(event["position_after"]):
                        raise ValueError("flat lock event conceals an open position")
                else:
                    raise ValueError("unknown mandatory-close audit event")
            events_checked = len(events)
        elif scheduled_ids:
            raise ValueError("scheduled executions have no forced-exit audit events")
    return {
        "passed": True,
        "closure_intervals": len(plan),
        "executions_checked": len(work),
        "scheduled_exits": len(scheduled_ids),
        "position_interval_violations": 0,
        "late_entry_violations": 0,
        "missing_boundary_prices": 0,
        "forced_events_checked": events_checked,
        "execution_convention": (
            "Finalized one-minute bar; protective stop-first, then its closing price "
            "at the planned deadline."
        ),
        "operational_boundary": (
            "Historical synchronous execution only; no broker transport or live fill guarantee."
        ),
    }, audit


def build_forced_exit_evidence(capture):
    """Persist and source-check retained close events; full coverage is separate."""
    from .contracts import RecordTable

    frames = capture.audit_frames or {}
    parts = [
        frame.loc[frame.kind.eq("forced_exit_event")]
        for frame in frames.values()
        if not frame.empty
    ]
    parts = [part.loc[:, part.notna().any()].copy() for part in parts if not part.empty]
    if not parts:
        return {}, {"available": False, "reason": "No mandatory-close events recorded"}
    events = pd.concat(parts, ignore_index=True)
    # Flat-only profiles still have a declared nullable execution schema.
    # These are explicit absent executions, not missing/invented price facts.
    for field in ("trade_id", "setup_id", "resolution", "actual_exit_ts_utc", "exit_ticks"):
        if field not in events:
            events[field] = None
    if events.schedule_id.duplicated().any():
        raise ValueError("mandatory-close audit duplicated a planned deadline")
    bars = {bar.bar_id: bar for day in capture.bars_by_day.values() for bar in day}
    trades = capture.tables[RecordTable.EXECUTED_TRADE].set_index("trade_id")
    count = 0
    for event in events.to_dict("records"):
        source = bars.get(event["source_bar_bar_id"])
        if source is None or source.timeframe_ticks != 60:
            raise ValueError("mandatory-close audit lacks its exact one-minute source")
        deadline = pd.Timestamp(event["planned_deadline_ts_utc"])
        if (
            source.availability_ts_utc != deadline
            or not source.is_complete
            or source.is_partial
            or source.close_reason.value != "complete"
            or source.logical_open_ts_utc != deadline - pd.Timedelta(minutes=1)
            or source.logical_close_ts_utc != deadline
            or not deadline - pd.Timedelta(minutes=1) <= source.close_ts_utc < deadline
        ):
            raise ValueError("mandatory-close audit used an unavailable or incomplete price candle")
        for field in ("open_ticks", "high_ticks", "low_ticks", "close_ticks"):
            if event["source_bar_" + field] != getattr(source, field):
                raise ValueError("mandatory-close source candle prices were changed")
        if bool(event["position_after"]) or not bool(event["entry_lock_after"]):
            raise ValueError("mandatory-close event did not confirm flat and entry locked")
        if int(event["pending_entry_order_count"]) != 0:
            raise ValueError("mandatory-close event retained stale entry orders")
        if event["event_kind"] == "deadline_position_closed":
            trade = trades.loc[event["trade_id"]]
            if (
                pd.Timestamp(event["actual_exit_ts_utc"]) != deadline
                or pd.Timestamp(trade.resolution_ts_utc) != deadline
                or event["exit_ticks"] != trade.exit_ticks
                or event["resolution"] != trade.resolution
            ):
                raise ValueError("mandatory-close event differs from the executed trade")
            if trade.resolution == "scheduled_close" and trade.exit_ticks != source.close_ticks:
                raise ValueError("scheduled close differs from the known one-minute closing price")
            if (
                event["protective_order_disposition"]
                != "retained_until_position_resolved_then_removed"
            ):
                raise ValueError("mandatory-close audit removed protection before exit")
            count += 1
        elif event["event_kind"] != "deadline_flat_lock" or bool(event["position_before"]):
            raise ValueError("unknown or inconsistent mandatory-close event")
    return {"forced_exit_events": events}, {
        "available": True,
        "passed": True,
        "events": len(events),
        "position_closures": count,
        "scope": (
            "Retained deadline events checked against original bars and executed trades; "
            "independent full position-interval and planned-boundary coverage audits "
            "are required separately."
        ),
    }
