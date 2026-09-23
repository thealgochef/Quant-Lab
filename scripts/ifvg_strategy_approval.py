"""Owner review and approval of an exact strategy search; never launches work."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import ifvg_study_wizard as w

from alpha_lab.agents.data_infra.ifvg.presentation.axis_values import format_axis_value
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import date_scope
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import DECISION_ID, DECISION_KEYS
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval_review import (
    prepare_strategy_approval_review,
    record_strategy_approval,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import owner_authorization_bundle_from_store


def _configuration_rows(intent):
    axes = dict(intent["axes"])
    rows = []
    for index, combination in enumerate(product(*axes.values()), start=1):
        row = {"Configuration": index}
        for axis, value_id in zip(axes, combination, strict=True):
            value = AXIS_VALUE_REGISTRY_V1[value_id]
            payload = getattr(value, "payload", None)
            if not hasattr(value, "payload"):
                payload = resolve_axis_overrides({axis: value_id})
            row[SEARCH_AXIS_REGISTRY_V1[axis].human_label] = format_axis_value(axis, payload)
        rows.append(row)
    return rows


def _review_terms(st, intent):
    policy = intent["date_policy"]
    warmup = set(policy["warmup_dates"])
    evidence = [day for day in policy["replay_dates"] if day not in warmup]
    st.write(f"Research dates: {date_scope(evidence)} · {len(warmup)} fixed warmup days")
    st.dataframe(_configuration_rows(intent), hide_index=True, width="stretch")
    costs = intent["cost_policy"]
    points = costs["cost_points_round_turn"]
    dollars = points * costs["dollars_per_point"]
    st.write(f"Round-trip cost: {points:g} points (${dollars:.2f} per contract).")
    baseline = w.resolve_profile_config({"profile_name": intent["baseline_profile_name"]})
    st.caption(
        f"Baseline target: {format_axis_value('tp_r_multiple', baseline.section.tp_r_multiple)}. "
        f"Random seed: {intent['seed']}. "
        "Settings outside the comparison keep their baseline values."
    )
    with st.expander("Thresholds and other baseline settings"):
        gates = intent["objective_policy"]["feasibility_gates"]
        st.dataframe(
            [
                {"Threshold": label, "Value": str(gates[name]), "Units": unit}
                for name, label, unit in w._STRATEGY_GATE_ROWS
                if name in gates
            ],
            hide_index=True,
            width="stretch",
        )
        robustness = intent["objective_policy"]["robustness_gates"]
        st.write("Additional consistency thresholds:")
        st.dataframe(
            [
                {"Threshold": name.replace("_", " "),
                 "Value": "Not required" if value is None else str(value)}
                for name, value in robustness.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.write(
            "Ranking objective: "
            + ", ".join(value.replace("_", " ") for value in
                        intent["objective_policy"]["pareto_objectives"])
        )
        st.dataframe(
            [
                {"Setting": spec.human_label,
                 "Baseline value": format_axis_value(axis, baseline.effective_config[axis])}
                for axis, spec in SEARCH_AXIS_REGISTRY_V1.items()
                if w.axis_renders_widget(spec) and axis not in intent["axes"]
            ],
            hide_index=True,
            width="stretch",
        )


def render_strategy_approval(st, draft, resolution) -> tuple[bool, str | None]:
    """Return whether this path is handled and its current launch blocker."""
    resolved = resolution.resolved
    if (
        draft.mode_id not in {"fsm_config_search", "single_configuration"}
        or resolved is None
        or resolved.purpose is not w.RunPurpose.DEVELOPMENT_RESEARCH
        or resolution.flow.includes("prop_contracts")
    ):
        return False, None
    st.subheader("Study approval")
    try:
        fields = w._charter_fields(draft, resolution.roots, resolution)
        review = prepare_strategy_approval_review(
            fields, resolution.requirement_set, Path(resolution.roots["store_root"])
        )
        bundle = (
            owner_authorization_bundle_from_store(
                Path(resolution.roots["store_root"]),
                resolution.requirement_set,
                charter_intent_sha256=review.intent_sha256,
            )
            if resolved.freeze_allowed and not review.blockers
            else None
        )
        exact_approval_ready = bundle is not None and all(
            getattr(bundle.decision_refs.get(key), "decision_id", None) == DECISION_ID
            for key in DECISION_KEYS
        )
    except Exception as error:
        st.error(f"The approval review could not be prepared: {w.sanitize_error(error)}")
        return True, "Resolve the approval review error above before running."
    if review.blockers:
        st.error("Resolve these items before approving or running this study:")
        for blocker in review.blockers:
            st.write(f"• {blocker}")
        return True, "Resolve the approval checklist above before running."
    _review_terms(st, review.intent)
    st.caption(
        f"Local data check: metadata and warmup continuity checked for "
        f"{review.cache_checked_dates} dates. The runner verifies the full inputs at launch."
    )
    prefix = (
        f"ifvg_strategy_approval_{draft.draft_id}_{review.intent_sha256}_{review.evidence_sha256}"
    )
    with st.expander("Exact request and approval scope"):
        st.write(
            "This approval covers the selected settings and values, dates and warmup, "
            "costs, thresholds, seed, and recorded software versions. Changing the "
            "request requires matching approval."
        )
        st.download_button(
            "Download exact study request",
            data=json.dumps(review.intent, indent=2, sort_keys=True),
            file_name="strategy-study-request.json",
            mime="application/json",
            key=prefix + "_download",
        )
    if exact_approval_ready:
        st.success("This exact study is approved. Click Run study when you are ready.")
        return True, None
    st.info(
        "One approval is missing for this exact study. Review the settings above, "
        "enter your name and save your approval. Then Run study becomes available."
    )
    statement = (
        "I approve this exact strategy study with the displayed configurations, research "
        "dates and warmup, costs, thresholds, seed, and recorded software versions. "
        "Saving this approval does not start the backtest."
    )
    st.write(statement)
    author = st.text_input("Reviewer name", key=prefix + "_author").strip()
    confirmed = st.checkbox("I approve this exact strategy study", key=prefix + "_confirm")
    if st.button(
        "Save study approval", key=prefix + "_save", disabled=not author or not confirmed
    ):
        try:
            current_resolution = w._resolve_for_draft(draft, resolution.roots)
            current_fields = w._charter_fields(draft, resolution.roots, current_resolution)
            record_strategy_approval(
                review,
                current_fields=current_fields,
                requirement_set=current_resolution.requirement_set,
                store_root=Path(current_resolution.roots["store_root"]),
                author=author,
                approval_statement=statement,
                approved_at=w._utc_now(),
            )
        except Exception as error:
            st.error(f"Approval could not be completed: {w.sanitize_error(error)}")
        else:
            st.rerun()
    return True, "Save the exact study approval above before running."
