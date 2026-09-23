"""Additive HTF selection evidence; historical audit tables stay readable."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pandas as pd

KINDS = ("htf_selection_observation", "htf_selection_universe", "htf_zone_created")


def validate_selection_evidence(tables, *, expected_bar_ids, taps):
    """Fail closed on missing/duplicate observations, broken ranks or tap joins."""
    observations, universes, creations = (tables[k] for k in KINDS)
    key = "stamp_source_bar_id"
    expected = set(expected_bar_ids)
    if observations.empty or observations[key].duplicated().any():
        raise ValueError("missing or duplicate HTF selection observations")
    if set(observations[key]) != expected or len(observations) != len(expected):
        raise ValueError("HTF selection observation coverage differs from actual replay bars")
    if not creations.empty and creations.created_zone_id.duplicated().any():
        raise ValueError("duplicate HTF creation events")
    lookup = {}
    for row in universes.itertuples():
        if hashlib.sha256(row.zones_json.encode()).hexdigest() != row.universe_id:
            raise ValueError("HTF universe hash mismatch")
        zones = json.loads(row.zones_json)
        if row.universe_id in lookup and lookup[row.universe_id] != zones:
            raise ValueError("conflicting HTF universe contents")
        if len({z["zone"]["fvg_id"] for z in zones}) != len(zones):
            raise ValueError("duplicate zone in pre-cap universe")
        ordered = sorted(
            zones,
            key=lambda z: (
                z["zone"]["timeframe_seconds"],
                z["zone"]["confirmed_ts_utc"],
                z["zone"]["fvg_id"],
            ),
            reverse=True,
        )
        if zones != ordered:
            raise ValueError("HTF universe ranking differs from governing order")
        ranks = Counter()
        for z in zones:
            ranks[z["zone"]["timeframe_seconds"]] += 1
            if z["precap_rank_per_timeframe"] != ranks[z["zone"]["timeframe_seconds"]]:
                raise ValueError("HTF pre-cap rank is incorrect")
        lookup[row.universe_id] = zones
    actual_taps = Counter()
    actual_selected = {}
    if not taps.empty:
        for row in taps.itertuples():
            actual_taps[(row.tap_cursor, row.fvg_fvg_id)] += 1
            if row.selected:
                if row.tap_cursor in actual_selected:
                    raise ValueError("multiple selected HTF zones on one bar")
                actual_selected[row.tap_cursor] = row.fvg_fvg_id
    observed_taps = Counter()
    observed_selected = {}
    for row in observations.itertuples():
        zones = lookup.get(row.universe_id)
        if zones is None:
            raise ValueError("missing HTF universe")
        policy = getattr(row, "direction_selection_policy", "mixed_direction_rank_v1")
        if policy not in {"mixed_direction_rank_v1", "enabled_before_rank_v1"}:
            raise ValueError("unknown HTF direction selection policy")
        eligible = [z for z in zones if policy == "mixed_direction_rank_v1" or (
            getattr(row, "enable_longs", True) if z["zone"]["direction"] == "bullish"
            else getattr(row, "enable_shorts", False))]
        ranks = Counter()
        view = []
        for z in eligible:
            ranks[z["zone"]["timeframe_seconds"]] += 1
            if ranks[z["zone"]["timeframe_seconds"]] <= row.selection_cap:
                view.append(z["zone"]["fvg_id"])
        if view != json.loads(row.inventory_view_ids_json):
            raise ValueError("HTF view differs from cap and per-timeframe rank")
        if row.inventory_live_count != len(zones):
            raise ValueError("HTF live count mismatch")
        ids = json.loads(row.observed_tap_ids_json)
        if row.scan_mode == "not_evaluated" and ids:
            raise ValueError("bypassed scan cannot report actual tap decisions")
        if sum(json.loads(row.observed_drop_counts_json).values()) != len(ids):
            raise ValueError("HTF tap outcome counts do not reconcile")
        for value in (row.conflict_predicate, row.direction_enabled_predicate):
            if value not in {"true", "false", "not_evaluated"}:
                raise ValueError("invalid tri-state predicate")
        for ident in ids:
            observed_taps[(row.stamp_source_bar_cursor, ident)] += 1
        if row.observed_selected_id:
            observed_selected[row.stamp_source_bar_cursor] = row.observed_selected_id
    if observed_taps != actual_taps or observed_selected != actual_selected:
        raise ValueError("HTF observations and canonical tap events do not reconcile")
    return {
        "passed": True,
        "actual_replay_bars": len(expected),
        "observations": len(observations),
        "unique_universes": len(lookup),
        "creation_events": len(creations),
        "tap_events": sum(actual_taps.values()),
        "selected_events": len(actual_selected),
    }


def build_selection_evidence(capture):
    """Extract sidecars only when the imported Core supplies the observer."""
    from strategy_core.strategies.ifvg_smc.reducer import IfvgReducer

    if not hasattr(IfvgReducer, "_selection_audit_stamp"):
        return {}, {"available": False, "reason": "runtime has no HTF selection observer"}
    tables = {}
    for kind in KINDS:
        parts = []
        for frame in capture.audit_frames.values():
            if frame.empty:
                continue
            part = frame.loc[frame.kind.eq(kind)]
            if len(part):
                part = part.loc[:, part.notna().any()].copy()
                parts.append(part)
        tables[kind] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    trace = capture.trace_audit_rows
    taps = trace.loc[trace.kind.eq("htf_tap")]
    result = validate_selection_evidence(
        tables,
        expected_bar_ids=[
            bar.bar_id
            for bars in capture.bars_by_day.values()
            for bar in bars
            if bar.timeframe_ticks == 60
        ],
        taps=taps,
    )
    return tables, result
