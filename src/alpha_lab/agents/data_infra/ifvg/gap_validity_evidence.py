"""Source-backed checks for starting-gap validity, separate from physical fills.

Historical audit companions need not contain this additive table. New runtimes
that emit it must pass source, timing and policy checks before publication.
This verifier does not call the engine's gap-validity implementation.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

ORIGINAL = "execution_wick_full_fill_v1"
OWN_CLOSE = "own_timeframe_close_v1"


def _finalization(bar, schedule) -> str:
    if bar.is_complete and not bar.is_partial and bar.close_reason in ("complete", None):
        return "complete"
    if bar.close_reason != "end_of_day" or bar.logical_close_ts_utc is None:
        return "not_finalized"
    observed_end = bar.close_ts_utc.replace(second=0, microsecond=0) + timedelta(minutes=1)
    if observed_end > bar.logical_close_ts_utc:
        return "not_finalized"
    cursor = observed_end + timedelta(minutes=1)
    if str(schedule.slot(cursor).status) in {"eligible", "outside_schedule_coverage"}:
        return "not_finalized"
    while cursor <= bar.logical_close_ts_utc:
        if str(schedule.slot(cursor).status) in {"eligible", "outside_schedule_coverage"}:
            return "not_finalized"
        cursor += timedelta(minutes=1)
    return "finalized_session_tail"


def validate_gap_validity_events(events: pd.DataFrame, *, source_bars, schedule) -> dict:
    """Check every retained event against its exact original source candle.

Coverage here means all retained events, not proof that no observer event was
omitted. Export verification separately reconstructs applicable source closes.
"""
    if events.empty:
        return {"available": True, "passed": True, "events": 0,
                "scope": "source validation of all retained events"}
    required = {
        "event_kind", "fvg_fvg_id", "fvg_timeframe_seconds", "fvg_direction",
        "fvg_gap_low_ticks", "fvg_gap_high_ticks", "fvg_confirmed_ts_utc",
        "source_bar_bar_id", "decision_bar_bar_id", "selected_policy",
        "fvg_role",
        "validity_before", "validity_after", "finalization_status",
        "eligible_own_timeframe_check", "wick_traversed", "original_wick_invalid",
    }
    if missing := required - set(events):
        raise ValueError(f"gap validity evidence is missing {sorted(missing)}")
    bars = {bar.bar_id: bar for bar in source_bars}
    if len(bars) != len(source_bars):
        raise ValueError("duplicate source bar IDs in gap validity evidence")
    checked_closes = set()
    physical_traversals = set()
    policy_invalidated = set()
    finality = {}
    for row in events.to_dict("records"):
        gap_id = row["fvg_fvg_id"]
        policy = row["selected_policy"]
        if policy not in {ORIGINAL, OWN_CLOSE}:
            raise ValueError("unknown starting-gap invalidation policy")
        if row["fvg_role"] != "htf":
            raise ValueError("starting-gap validity evidence leaked into another role")
        source = bars.get(row["source_bar_bar_id"])
        decision = bars.get(row["decision_bar_bar_id"])
        if source is None or decision is None:
            raise ValueError("gap validity source or decision candle is absent")
        for prefix, bar in (("source_bar_", source), ("decision_bar_", decision)):
            for field in ("timeframe_ticks", "open_ticks", "high_ticks", "low_ticks", "close_ticks",
                          "is_complete", "is_partial", "close_reason"):
                saved = row.get(prefix + field)
                actual = getattr(bar, field)
                if saved != actual and not (pd.isna(saved) and actual is None):
                    raise ValueError(f"gap validity {prefix}{field} differs from source")
            for field in ("open_ts_utc", "close_ts_utc", "logical_open_ts_utc",
                          "logical_close_ts_utc"):
                saved, actual = row.get(prefix + field), getattr(bar, field)
                if actual is None and pd.isna(saved):
                    continue
                if pd.Timestamp(saved) != pd.Timestamp(actual):
                    raise ValueError(f"gap validity {prefix}{field} differs from source")
        confirmed = pd.Timestamp(row["fvg_confirmed_ts_utc"])
        if source.availability_ts_utc <= confirmed:
            raise ValueError("gap validity reused a formation or past candle")
        if source.availability_ts_utc > decision.availability_ts_utc:
            raise ValueError("gap validity used an unavailable candle")
        if decision.timeframe_ticks != 60:
            raise ValueError("gap validity decision is not a one-minute candle")
        low, high = int(row["fvg_gap_low_ticks"]), int(row["fvg_gap_high_ticks"])
        bullish = row["fvg_direction"] == "bullish"
        if row["fvg_direction"] not in {"bullish", "bearish"} or low >= high:
            raise ValueError("invalid starting-gap geometry")
        before = bool(row["validity_before"])
        if before != (gap_id not in policy_invalidated):
            raise ValueError("gap validity continuity or no-resurrection check failed")
        invalidates = False
        if row["event_kind"] == "own_timeframe_close_check":
            key = (gap_id, source.bar_id)
            if key in checked_closes:
                raise ValueError("duplicated own-timeframe closing check")
            checked_closes.add(key)
            if source.timeframe_ticks != int(row["fvg_timeframe_seconds"]):
                raise ValueError("gap validity used another timeframe's candle")
            if source.bar_id not in finality:
                finality[source.bar_id] = _finalization(source, schedule)
            status = finality[source.bar_id]
            if row["finalization_status"] != status:
                raise ValueError("gap validity finalization status differs from source")
            eligible = status != "not_finalized"
            if bool(row["eligible_own_timeframe_check"]) != eligible:
                raise ValueError("gap validity counted a developing candle as final")
            beyond = source.close_ticks < low if bullish else source.close_ticks > high
            invalidates = policy == OWN_CLOSE and eligible and beyond
        elif row["event_kind"] == "physical_wick_traversal":
            if source.timeframe_ticks != 60 or source.bar_id != decision.bar_id:
                raise ValueError("physical traversal did not use the decision candle")
            overlaps = source.high_ticks >= low and source.low_ticks <= high
            traverses = source.low_ticks <= low if bullish else source.high_ticks >= high
            if not overlaps or not traverses or not row["wick_traversed"]:
                raise ValueError("physical wick traversal disagrees with source prices")
            if gap_id in physical_traversals:
                raise ValueError("duplicated first physical wick traversal")
            physical_traversals.add(gap_id)
            invalidates = policy == ORIGINAL
        else:
            raise ValueError("unknown gap validity event")
        if invalidates:
            policy_invalidated.add(gap_id)
            expected_reason = ("execution_wick_full_fill" if policy == ORIGINAL
                               else "own_timeframe_close_beyond_far_boundary")
            if row.get("invalidation_reason") != expected_reason:
                raise ValueError("gap policy invalidation reason differs from source")
            if pd.Timestamp(row.get("policy_invalidated_ts_utc")) != source.availability_ts_utc:
                raise ValueError("gap policy invalidation time differs from availability")
        if bool(row["validity_after"]) != (gap_id not in policy_invalidated):
            raise ValueError("gap validity policy result differs from source prices")
        if bool(row["original_wick_invalid"]) != (gap_id in physical_traversals):
            raise ValueError("physical traversal was hidden or rewritten")
    return {
        "available": True, "passed": True, "events": len(events),
        "own_timeframe_checks": len(checked_closes),
        "physical_wick_traversals": len(physical_traversals),
        "policy_invalidated_gaps": len(policy_invalidated),
        "scope": "source validation of all retained events",
    }


def build_gap_validity_evidence(capture):
    from strategy_core.candles.exchange_calendar import CME_EQUITY_INDEX_FUTURES_ETH_SCHEDULE

    parts = [frame.loc[frame.kind.eq("gap_validity_event")]
             for frame in capture.audit_frames.values() if not frame.empty]
    parts = [part.loc[:, part.notna().any()].copy() for part in parts if not part.empty]
    if not parts:
        return {}, {"available": False, "reason": "No starting-gap validity events recorded"}
    events = pd.concat(parts, ignore_index=True)
    bars = [bar for day in capture.bars_by_day.values() for bar in day]
    report = validate_gap_validity_events(
        events, source_bars=bars, schedule=CME_EQUITY_INDEX_FUTURES_ETH_SCHEDULE,
    )
    return {"gap_validity_events": events}, report
