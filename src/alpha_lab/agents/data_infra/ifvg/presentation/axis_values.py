"""Readable, exact search-axis payloads without importing the search registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["format_axis_value"]


_NULL_LABELS = {
    "opposing_min_gap_ticks": "Use the general minimum gap",
    "setup_timeout_1m_bars": "No setup lifetime limit (unbounded)",
    "parent_retest_timeout_1m_bars": "No timeout (unbounded)",
    "opposing_timeout_1m_bars": "No timeout (unbounded)",
    "inversion_timeout_1m_bars": "No timeout (unbounded)",
    "htf_registry_max_age_days": "No age limit",
    "max_executed_trades_per_day": "No trade cap",
    "parent_htf_distance_ticks_max": "No distance cap",
    "opposing_parent_distance_ticks_max": "No distance cap",
    "entry_parent_distance_ticks_max": "No distance cap",
    "ltf_registry_max_live": "No live LTF zone cap",
}

# Units describe the supplied value, never a copied baseline/default value.
_UNITS = {
    "opposing_min_gap_ticks": ("tick", "ticks"),
    "setup_timeout_1m_bars": ("1m bar", "1m bars"),
    "parent_retest_timeout_1m_bars": ("1m bar", "1m bars"),
    "opposing_timeout_1m_bars": ("1m bar", "1m bars"),
    "inversion_timeout_1m_bars": ("1m bar", "1m bars"),
    "post_inversion_expiry_1m_bars_max": ("1m bar", "1m bars"),
    "parent_reaction_window_1m_bars_max": ("1m bar", "1m bars"),
    "parent_reaction_window_parent_bars": ("parent bar", "parent bars"),
    "htf_registry_max_age_days": ("day", "days"),
    "max_executed_trades_per_day": (
        "trade per trading day", "trades per trading day",
    ),
    "min_gap_ticks_capture": ("tick", "ticks"),
    "parent_htf_distance_ticks_max": ("tick", "ticks"),
    "opposing_parent_distance_ticks_max": ("tick", "ticks"),
    "entry_parent_distance_ticks_max": ("tick", "ticks"),
    "sl_buffer_ticks": ("tick", "ticks"),
    "tp_r_multiple": ("R", "R"),
    "swing_strength_bars": ("bar", "bars"),
    "swing_pool_max": ("swing", "swings"),
    "ltf_registry_max_live": ("live LTF zone", "live LTF zones"),
    "htf_selection_max_per_timeframe": (
        "HTF zone per timeframe", "HTF zones per timeframe",
    ),
}

_POLICY_LABELS = {
    "holding_policy": {
        "legacy_unrestricted_v1": "Historical unrestricted holding (legacy)",
        "scheduled_daily_close_v1": (
            "Mandatory daily close - 3:55 PM Chicago time (five-minute buffer)"
        ),
    },
    "enabled_entry_sessions": {
        "asia": "Asia - 3:00 PM to 12:45 AM Chicago time",
        "london": "London - 1:00 AM to 6:00 AM Chicago time",
        "ny": "New York - 7:00 AM to 1:00 PM Chicago time",
        "ny_0700_1030": "Legacy morning - 6:00 AM to 9:30 AM Chicago time (historical)",
    },
    "htf_gap_invalidation_policy": {
        "execution_wick_full_fill_v1": "One-minute wick reaches the far edge (original rule)",
        "own_timeframe_close_v1": "Candle closes beyond the gap on its own timeframe",
    },
    "parent_replacement_policy": {
        "highest_tf_newest": "Higher timeframe, then newer parent (highest_tf_newest)",
        "preserve_selected": "Preserve the selected parent (preserve_selected)",
    },
    "parent_retest_depth_policy": {
        "any_live_touch": "Any live first touch (any_live_touch)",
        "strictly_before_ce": "First touch strictly before the midpoint (strictly_before_ce)",
    },
}


def format_axis_value(axis_key: str, value: Any) -> str:
    """Show a payload's actual value and units, including explicit null meaning.

    Frozen key/value tuples and ordinary mappings render identically. Policy
    strings and nested field names remain exact so the displayed configuration
    can be checked against the persisted section.
    """
    if value is None:
        return _NULL_LABELS.get(axis_key, "Not set (None)")
    if isinstance(value, str):
        if axis_key in {"htf_timeframes", "parent_timeframes"}:
            return {
                "1m": "one minute", "3m": "three minutes", "5m": "five minutes",
                "10m": "ten minutes", "15m": "fifteen minutes", "30m": "thirty minutes",
                "1H": "one hour", "4H": "four hours",
            }.get(value, value)
        return _POLICY_LABELS.get(axis_key, {}).get(value, value or 'Empty string ("")')
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, (int, float)):
        number = str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        if axis_key in _UNITS:
            singular, plural = _UNITS[axis_key]
            return f"{number} {singular if value == 1 else plural}"
        return number
    if isinstance(value, Mapping):
        members = value.items()
    elif isinstance(value, tuple) and value and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ):
        members = value
    elif isinstance(value, (list, tuple)):
        if axis_key == "enabled_entry_sessions" and tuple(value) == ("asia", "london", "ny"):
            return (
                "Original three windows - 3:00 PM to 12:45 AM, 1:00 AM to 6:00 AM, "
                "7:00 AM to 1:00 PM Chicago time"
            )
        return "[" + ", ".join(format_axis_value(axis_key, item) for item in value) + "]"
    else:
        return str(value) if str(value) else 'Empty string ("")'
    return "{" + "; ".join(
        f"{key}: {format_axis_value(str(key), item)}" for key, item in members
    ) + "}"
