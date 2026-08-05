"""Fail-closed v1 configuration migration."""

from __future__ import annotations

from copy import deepcopy

__all__ = ["migrate_v1_experiment_config"]


_CANONICAL_DOC_DEFAULTS = {
    "min_gap_ticks": 4,
    "parent_htf_distance_ticks_max": 80,
    "opposing_parent_distance_ticks_max": 80,
    "post_inversion_expiry_1m_bars_max": 80,
}

_V1_TO_V2_SECTION_KEYS = {
    "min_gap_ticks": "min_gap_ticks_capture",
    "parent_htf_distance_ticks_max": "parent_htf_distance_ticks_max",
    "opposing_parent_distance_ticks_max": (
        "opposing_parent_distance_ticks_max"
    ),
    "post_inversion_expiry_1m_bars_max": (
        "post_inversion_expiry_1m_bars_max"
    ),
}


def migrate_v1_experiment_config(raw: dict) -> dict:
    """Translate identity/caps without promoting legacy rows to executions."""
    source = deepcopy(raw)
    doc = source.get("doc_defaults") or {}
    apply = bool(doc.get("apply", False))
    filters = source.get("filters") or {}
    legacy_cap = filters.get("max_trades_per_day")
    values = {key: doc.get(key, default) for key, default in _CANONICAL_DOC_DEFAULTS.items()}
    canonical = values == _CANONICAL_DOC_DEFAULTS
    section_overrides = {
        _V1_TO_V2_SECTION_KEYS[key]: value
        for key, value in values.items()
        if value != _CANONICAL_DOC_DEFAULTS[key]
    }

    if not apply:
        mode = "broad_capture"
        profile_name = "ifvg_v1_legacy_candidate_stream"
        execution_enabled = False
        base_profile_name = profile_name
        section_overrides = {}
    elif canonical:
        mode = "doc_default"
        profile_name = "ifvg_v2_doc_default_fresh_static_1r"
        execution_enabled = True
        base_profile_name = profile_name
    else:
        mode = "custom_profile"
        profile_name = "ifvg_v2_custom_migrated"
        execution_enabled = True
        base_profile_name = "ifvg_v2_doc_default_fresh_static_1r"
    return {
        "migration_schema_version": 2,
        "source_schema": "ifvg_v1",
        "qualification_mode": mode,
        "profile_name": profile_name,
        "base_profile_name": base_profile_name,
        "section_overrides": section_overrides,
        "execution_enabled": execution_enabled,
        "legacy_candidate_row_limit": legacy_cap,
        "max_executed_trades_per_day": None,
        "legacy_selected_read_only": True,
        "legacy_outcomes_read_only": True,
        "raw_v1_config": source,
    }
