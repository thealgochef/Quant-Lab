"""Fixed-value controls used only by the existing study Configuration step."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import (
    LEGACY_MORNING,
    export_fixed_configuration,
    import_fixed_configuration,
)
from alpha_lab.agents.data_infra.ifvg.presentation.axis_values import format_axis_value
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    CompositeAxisValue,
)

FIXED_AXES = (
    "htf_selection_max_per_timeframe",
    "htf_gap_invalidation_policy",
    "htf_direction_selection_policy",
    "enabled_entry_sessions", "holding_policy",
    "parent_timeframes", "opposing_min_gap_ticks",
    "parent_retest_timeout_1m_bars", "opposing_timeout_1m_bars",
    "htf_registry_max_age_days", "setup_timeout_1m_bars",
    "parent_reaction_window_parent_bars", "parent_htf_distance_ticks_max",
    "opposing_parent_distance_ticks_max", "parent_replacement_policy",
    "parent_retest_depth_policy",
)
KEEP_BASELINE = "__keep_baseline__"


def render_fixed_settings(st, payload, baseline_name, *, key_prefix="ifvg_fixed_"):
    st.markdown("**Configuration settings — one replay**")
    st.caption(
        "Choose one value per setting. Other settings retain the selected baseline. "
        "These fixed choices do not add baseline comparisons or combinations."
    )
    base = resolve_profile_config({"profile_name": baseline_name}).effective_config
    selected = dict(payload.get("fixed_axis_value_ids") or {})
    # Imports are applied before widgets are instantiated; they remain ordinary
    # editable fixed choices and cannot import approval or start a replay.
    with st.expander("Import or export one configuration"):
        raw = st.text_area("Paste exported configuration JSON", key=f"{key_prefix}import_json")
        if st.button("Apply imported configuration", key=f"{key_prefix}import_apply"):
            try:
                selected = import_fixed_configuration(raw, baseline_name)
                for axis in FIXED_AXES:
                    if axis in SEARCH_AXIS_REGISTRY_V1:
                        default = (
                            SEARCH_AXIS_REGISTRY_V1[axis].baseline_value_id
                            if axis == "htf_gap_invalidation_policy" else KEEP_BASELINE
                        )
                        st.session_state[f"{key_prefix}{baseline_name}_{axis}"] = (
                            selected.get(axis, default)
                        )
                st.success("Configuration imported. Review the choices and save the draft.")
            except (ValueError, TypeError, KeyError, PermissionError) as error:
                st.error(str(error))
    for axis in FIXED_AXES:
        spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
        if spec is None:
            continue
        explicit_policy = axis == "htf_gap_invalidation_policy"
        options = list(spec.registered_values) if explicit_policy else [
            KEEP_BASELINE, *spec.registered_values,
        ]
        inherited = next((
            value_id for value_id in spec.registered_values
            if getattr(AXIS_VALUE_REGISTRY_V1[value_id], "payload", object()) == base[axis]
        ), spec.baseline_value_id)
        stored = selected.get(axis, inherited if explicit_policy else KEEP_BASELINE)
        if stored not in options:
            st.error(f"Saved value {stored!r} is unavailable; review this configuration.")
            continue

        def label(value, axis=axis):
            if value == KEEP_BASELINE:
                return "Keep baseline: " + format_axis_value(axis, base[axis])
            registered = AXIS_VALUE_REGISTRY_V1[value]
            if isinstance(registered, CompositeAxisValue):
                return registered.human_label
            return format_axis_value(axis, registered.payload)

        widget_key = f"{key_prefix}{baseline_name}_{axis}"
        help_text = spec.description
        if explicit_policy and st.session_state.get(widget_key, stored) == (
            "htf_gap_invalidation_policy.execution_wick_full_fill_v1"
        ):
            help_text = (
                "The original one-minute wick/full-fill rule invalidates starting "
                "one-hour and four-hour gaps. Existing boundary, overlap and later-bar "
                "requirements are preserved. Smaller-pattern rules and protective "
                "stops are unchanged."
            )
        value = st.selectbox(
            spec.human_label, options, index=options.index(stored), format_func=label,
            key=widget_key, help=help_text,
        )
        if value == KEEP_BASELINE:
            selected.pop(axis, None)
        elif not (
            explicit_policy and axis not in (payload.get("fixed_axis_value_ids") or {})
            and selected.get("enabled_entry_sessions") == LEGACY_MORNING
            and value == inherited
        ):
            selected[axis] = value
        if axis == "enabled_entry_sessions" and value == "enabled_entry_sessions.ny_0700_1030":
            st.caption(
                "This saved legacy preset still means 6:00 AM to 9:30 AM Chicago time. "
                "To use the corrected morning, create an explicit copy and select "
                "Morning - 7:00 AM to 10:30 AM Chicago time."
            )
        if axis == "holding_policy":
            st.caption(
                "Mandatory daily-close research requires the 3:55 PM Chicago preset. "
                "Overnight holding is allowed within the open session. The daily and "
                "weekend locks prevent carrying positions across market closures; "
                "the end of an entry window does not itself close a position."
            )
    missing = set(selected) - set(SEARCH_AXIS_REGISTRY_V1)
    if missing:
        st.error("This runtime lacks saved settings: " + ", ".join(sorted(missing)))
    else:
        st.download_button(
            "Export this configuration",
            export_fixed_configuration(baseline_name, selected),
            file_name="ifvg_configuration.json", mime="application/json",
            key=f"{key_prefix}export",
        )
    return selected
