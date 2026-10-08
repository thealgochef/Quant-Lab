"""Point-in-time phase features, shared by label shadows and online inference.

Inputs are the owning Core geometry and completed bars. Output includes the
availability/source receipt for every declared field; outcomes are not inputs.
"""

from __future__ import annotations

import math
from datetime import date

from alpha_lab.propsim.funded.clock import CHICAGO, from_ns, to_ns

MINUTE_NS = 60_000_000_000


def structural_snapshot(setup, geometry, *, tick_size: float) -> dict:
    """Called at the actual executable candidate before any account fill."""
    parent, opposing = geometry.parent, geometry.opposing
    separation = (
        max(
            0,
            parent.gap_low_ticks - opposing.gap_high_ticks,
            opposing.gap_low_ticks - parent.gap_high_ticks,
        )
        * tick_size
    )
    gap = geometry.entry_fvg
    return {
        "setup_id": setup.setup_id,
        "htf_id": geometry.htf.fvg_id,
        "htf_timeframe": str(geometry.htf.timeframe_seconds),
        "parent_timeframe": str(parent.timeframe_seconds),
        "parent_width_points": (parent.gap_high_ticks - parent.gap_low_ticks) * tick_size,
        "opposing_width_points": (opposing.gap_high_ticks - opposing.gap_low_ticks) * tick_size,
        "entry_gap_width_points": None
        if gap is None
        else (gap.gap_high_ticks - gap.gap_low_ticks) * tick_size,
        "separation_points": separation,
        "parent_midpoint": (parent.gap_high_ticks + parent.gap_low_ticks) * tick_size / 2,
        "entry_price": geometry.entry_ticks * tick_size,
        "stop_price": geometry.stop_ticks * tick_size,
        "lock_ns": None if setup.lock_ts_utc is None else to_ns(setup.lock_ts_utc),
        "inversion_ns": None if setup.inversion_ts_utc is None else to_ns(setup.inversion_ts_utc),
        "frozen_max_distance_points": None
        if setup.opposing_distance_ticks_max is None
        else setup.opposing_distance_ticks_max * tick_size,
        "geometry_cursor": geometry.feature_as_of_cursor,
    }


def build_features(
    *,
    job: str,
    decision_ns: int,
    entry_ns: int,
    decision_price: float,
    trading_day: str,
    deadline_ns: int,
    structure: dict,
    completed_bars: list,
    context: dict,
    definitions: list[dict],
    tick_size: float = 0.25,
    checkpoint: dict | None = None,
) -> dict:
    if job not in {"ENTRY", "CONTINUATION"}:
        raise ValueError("unknown prediction job")
    names = [r["name"] for r in definitions if job in r["jobs"]]
    values = dict.fromkeys(names)
    provenance = {
        n: {"known_at_ns": decision_ns, "source": None, "missing_reason": "source_unavailable"}
        for n in names
    }

    def put(name, value, known, source, reason="source_unavailable"):
        if name not in values:
            return
        if known > decision_ns:
            raise ValueError(f"future feature source: {name}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"nonfinite source: {name}")
        values[name] = value
        provenance[name] = {
            "known_at_ns": int(known),
            "source": source,
            "missing_reason": reason if value is None else None,
        }

    risk = abs(structure["entry_price"] - structure["stop_price"])
    if risk <= 0 or not math.isfinite(risk) or deadline_ns < decision_ns or entry_ns > decision_ns:
        raise ValueError("invalid entry risk, decision time or mandatory deadline")
    sid = structure["geometry_cursor"]
    put("initial_risk_points", risk, entry_ns, sid)
    for name in ("htf_timeframe", "parent_timeframe", "frozen_max_distance_points"):
        put(name, structure.get(name), entry_ns, sid)
    for name, raw in (
        ("parent_width_r", "parent_width_points"),
        ("opposing_width_r", "opposing_width_points"),
        ("entry_gap_width_r", "entry_gap_width_points"),
        ("opposing_parent_separation_r", "separation_points"),
    ):
        value = structure.get(raw)
        put(name, None if value is None else value / risk, entry_ns, sid)
    midpoint = structure.get("parent_midpoint")
    put(
        "entry_parent_midpoint_r",
        None if midpoint is None else (structure["entry_price"] - midpoint) / risk,
        entry_ns,
        sid,
    )
    for name, source in (
        ("lock_to_entry_minutes", "lock_ns"),
        ("inversion_to_entry_minutes", "inversion_ns"),
    ):
        stamp = structure.get(source)
        if stamp is not None and stamp > entry_ns:
            raise ValueError("structural event after entry")
        put(name, None if stamp is None else (entry_ns - stamp) / MINUTE_NS, entry_ns, sid)
    clock = from_ns(decision_ns).astimezone(CHICAGO)
    minute = (
        clock.hour * 60
        + clock.minute
        + clock.second / 60
        + (decision_ns % 1_000_000_000) / MINUTE_NS
    )
    put("clock_sin", math.sin(2 * math.pi * minute / 1440), decision_ns, "Chicago_clock")
    put("clock_cos", math.cos(2 * math.pi * minute / 1440), decision_ns, "Chicago_clock")
    put(
        "minutes_to_mandatory_close",
        (deadline_ns - decision_ns) / MINUTE_NS,
        decision_ns,
        "frozen_session_calendar",
    )
    put(
        "trading_weekday",
        date.fromisoformat(trading_day).strftime("%a"),
        decision_ns,
        "frozen_session_calendar",
    )
    bars = [
        b
        for b in completed_bars
        if b.trading_day.isoformat() == trading_day
        and to_ns(b.availability_ts_utc) <= decision_ns
        and b.timeframe_ticks == 60
    ]
    stamps = [to_ns(b.availability_ts_utc) for b in bars]
    if stamps != sorted(set(stamps)):
        raise ValueError("completed bars must have unique chronological availability")

    def window(count):
        chosen = bars[-count:]
        ns = stamps[-count:]
        return (
            chosen
            if len(chosen) == count
            and all(b - a == MINUTE_NS for a, b in zip(ns, ns[1:], strict=False))
            else []
        )

    selected = window(6)
    if selected:
        put(
            "close_change_5_r",
            (selected[-1].close_ticks - selected[0].close_ticks) * tick_size / risk,
            stamps[-1],
            [b.bar_id for b in selected],
        )
    selected = window(15)
    if selected:
        put(
            "range_15_r",
            (max(b.high_ticks for b in selected) - min(b.low_ticks for b in selected))
            * tick_size
            / risk,
            stamps[-1],
            [b.bar_id for b in selected],
        )
    selected = window(16)
    if selected:
        closes = [b.close_ticks for b in selected]
        denominator = sum(abs(b - a) for a, b in zip(closes, closes[1:], strict=False))
        put(
            "signed_efficiency_15",
            (closes[-1] - closes[0]) / denominator if denominator else 0.0,
            stamps[-1],
            [b.bar_id for b in selected],
        )
    gamma, levels = context.get("gamma", {}), context.get("levels", {})
    if to_ns(context["decision_time_utc"]) > decision_ns:
        raise ValueError("future context snapshot")
    if levels.get("status") == "selected":
        known = to_ns(levels["nominal_eligible_from_utc"])
        source = levels["level_set_id"]
        items = levels["items"]
        put("level_set_age_hours", (decision_ns - known) / 3_600_000_000_000, known, source)
        lower = items.get("1D Min", {}).get("price")
        upper = items.get("1D Max", {}).get("price")
        if lower is not None and upper is not None and upper > lower:
            move = (upper - lower) / 2
            put("implied_move_points", move, known, source)
            put("risk_to_implied_move", risk / move, known, source)
            separation = structure.get("separation_points")
            put(
                "separation_to_implied_move",
                None if separation is None else separation / move,
                max(known, entry_ns),
                source,
            )
            put(
                "price_in_implied_range", (decision_price - lower) / (2 * move), decision_ns, source
            )
        hvl = items.get("HVL", {}).get("price")
        put(
            "hvl_distance_r",
            None if hvl is None else (decision_price - hvl) / risk,
            decision_ns,
            source,
        )
        usable = [
            (v["price"], name, v["gex"])
            for name, v in items.items()
            if v.get("price") is not None
            and v.get("gex") is not None
            and math.isfinite(v["price"])
            and math.isfinite(v["gex"])
        ]
        for direction in ("upper", "lower"):
            selected = [
                v
                for v in usable
                if (v[0] >= decision_price if direction == "upper" else v[0] <= decision_price)
            ]
            if selected:
                price, name, gex = min(selected, key=lambda v: (abs(v[0] - decision_price), v[1]))
                put(
                    f"nearest_{direction}_distance_r",
                    abs(price - decision_price) / risk,
                    decision_ns,
                    {"level_set": source, "level": name},
                )
                put(f"nearest_{direction}_gex", gex, known, {"level_set": source, "level": name})
    if gamma.get("status") == "selected":
        known = to_ns(gamma["nominal_eligible_from_utc"])
        source = gamma["source_sha256"]
        value, age = gamma["value"], gamma.get("positive_run_age")
        if value <= 0:
            age = 0
        put("total_net_gamma", value, known, source)
        put("positive_report_age", age, known, source, "left_censored_positive_run")
        phase = (
            "negative"
            if value < 0
            else "zero"
            if value == 0
            else "early_positive"
            if age is not None and age <= 5
            else "established_positive"
            if age is not None and age >= 6
            else "unknown"
        )
        put("gamma_state", phase, known, source)
        put("gamma_report_age_hours", (decision_ns - known) / 3_600_000_000_000, known, source)
    else:
        put("gamma_state", "unknown", decision_ns, "asof_lookup")
    if job == "CONTINUATION":
        if checkpoint is None:
            raise ValueError("continuation requires observed checkpoint evidence")
        put(
            "minutes_entry_to_target",
            (decision_ns - entry_ns) / MINUTE_NS,
            decision_ns,
            checkpoint["event_id"],
        )
        put(
            "mae_so_far_r",
            max(0, structure["entry_price"] - checkpoint["observed_min_price"]) / risk,
            decision_ns,
            checkpoint["event_id"],
        )
        put(
            "checkpoint_overshoot_r",
            (decision_price - checkpoint["target_price"]) / risk,
            decision_ns,
            checkpoint["event_id"],
        )
    return {"values": values, "provenance": provenance, "decision_ns": decision_ns}
