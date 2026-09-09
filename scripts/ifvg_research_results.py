"""Question-first results with metric registries and compatibility-gated deltas."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px

from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import describe
from alpha_lab.agents.data_infra.ifvg.presentation.results_presentation import (
    prop_readings,
    strategy_readings,
)
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (
    PRIMARY_METRICS,
    axis_value_name,
    configuration_name,
)


def metric_help(key):
    spec = describe(key)
    direction = {
        "higher_better": "Higher is better",
        "lower_better": "Lower is better",
        "descriptive": "Descriptive; no preferred direction",
        "target": "Closer to the target is better",
    }[spec.directionality]
    return f"{spec.definition} {direction}. Units: {spec.unit}."


def research_sentence(value):
    from ifvg_research_review import words

    text = str(value or "")
    # Registry explanations can contain nested source citations. Remove the
    # entire citation while preserving thresholds and statistical parentheses.
    source = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")
    start, depth, pieces, cursor = None, 0, [], 0
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None and source.search(text[start : index + 1]):
                pieces.append(text[cursor:start].rstrip())
                cursor = index + 1
    pieces.append(text[cursor:])
    text = "".join(pieces)
    text = text.replace("proposed_protocol_default", "proposed default")
    return words(text)


def metric_cards(st, readings, keys=PRIMARY_METRICS):
    by_key = {reading.technical_key: reading for reading in readings}
    columns = st.columns(min(len(keys), 4))
    for column, key in zip(columns, keys, strict=False):
        reading = by_key.get(key)
        if reading is None:
            continue
        column.metric(reading.human_name, reading.display, help=metric_help(key))
        column.caption(reading.chip)


def reading_table(readings):
    return [
        {
            "Metric": reading.human_name,
            "Value": reading.display,
            "Evidence": reading.chip,
            "Interpretation": research_sentence(reading.interpretation),
            "Limitations": research_sentence(reading.caveat),
        }
        for reading in readings
    ]


def render_search_results(st, study, roots):
    from ifvg_results_tab import load_results_bundle

    try:
        bundle = load_results_bundle(
            store_root=study.store_root, state_root=Path(roots["state_root"]), search_id=study.key
        )
    except Exception:
        st.error(
            "Saved result evidence failed verification. This study cannot support a "
            "conclusion until that evidence is restored."
        )
        return
    render_bundle(st, bundle, study=study, roots=roots)


def load_pipeline_bundle(study):
    """Adapt exact persisted pipeline outputs without running an evaluator."""
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import SearchFrontierEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import PipelineResultEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
    from alpha_lab.agents.data_infra.ifvg.study_presentation import human_config_name
    from alpha_lab.agents.data_infra.ifvg.study_providers import (
        _stage_sidecar,
        load_charter,
        load_child_metrics,
    )

    state = study.state or {}
    charter = load_charter(study.store_root, study.charter_id)
    children = list(state.get("children") or ())
    metrics = {}
    if charter:
        for row in children:
            identity = row.get("core_replay_id")
            if identity:
                metrics[identity] = load_child_metrics(
                    study.store_root, identity, charter.payload.cost_policy
                )
    frontier = None
    limitations = []
    result_id = (state.get("publication") or {}).get("pipeline_result_id")
    final_checks = (state.get("stages") or {}).get("15_verify_and_publish", {}).get("in_plan")
    if result_id:
        result = load_verified_envelope(
            study.store_root, "search_results", result_id, PipelineResultEnvelope
        )
        if result.payload.pipeline_semantic_id != study.key:
            raise ValueError("Pipeline result belongs to another study")
        if result.payload.frontier_id:
            frontier = load_verified_envelope(
                study.store_root,
                "frontiers",
                result.payload.frontier_id,
                SearchFrontierEnvelope,
            ).payload.frontier
        if final_checks and (
            result.payload.control_flow_gates is None
            or not result.payload.control_flow_gates.passed
        ):
            frontier = None
            limitations.append(
                "The workflow's final evidence checks have not passed. These "
                "measurements do not establish a completed research conclusion."
            )
    elif final_checks:
        limitations.append(
            "The workflow's final verified report is missing. Its research "
            "conclusion is unresolved."
        )
    vectors = {}
    for stage in ("12_run_prop_historical_replays", "13_run_bootstrap_and_stress"):
        saved = _stage_sidecar(study.store_root, state, stage, "prop_vectors.json") or {}
        for core, simulations in saved.items():
            vectors.setdefault(core, {}).update(simulations)
    return {
        "charter": charter,
        "state": state,
        "children": children,
        "metrics_by_child": metrics,
        "frontier": frontier,
        "limitations": limitations,
        "prop_vectors": {
            core: {
                name: {"payout_reliability_vector": vector} for name, vector in simulations.items()
            }
            for core, simulations in vectors.items()
        },
        "names": {
            row["core_replay_id"]: human_config_name(row.get("axis_value_ids") or {})
            for row in children
            if row.get("core_replay_id")
        },
    }


def render_bundle(st, bundle, *, study=None, roots=None):
    for limitation in bundle.get("limitations", ()):
        st.warning(limitation)
    children = [row for row in bundle["children"] if row.get("core_replay_id")]
    charter = bundle.get("charter")
    if charter is None:
        st.warning(
            "The study's saved settings could not be verified. Its results cannot be interpreted."
        )
        return
    if not children:
        st.warning(
            "The study finished without evaluable configurations. No performance "
            "conclusion is available."
        )
        return
    frontier = bundle.get("frontier")
    if frontier is None:
        st.warning(
            "The final selection evidence is missing. Individual measurements below "
            "do not establish a study verdict."
        )
    elif not frontier.feasible_ids:
        st.warning(
            "No configuration met all of this study's required criteria. Inspect "
            "the measurements and failed criteria below."
        )
    else:
        st.success(
            f"{len(frontier.feasible_ids)} configuration(s) met the study's proposed criteria."
        )
    st.caption(
        "Actual executed strategy · Exploratory estimates from the selected "
        "research period. Passing proposed criteria does not establish future "
        "performance."
    )
    ids = [row["core_replay_id"] for row in children]
    names = {
        identity: f"{index + 1}. {configuration_name(children[index].get('axis_value_ids') or {})}"
        for index, identity in enumerate(ids)
    }
    select_key = "ifvg_research_results_configuration"
    if st.session_state.get(select_key) not in ids:
        st.session_state.pop(select_key, None)
    selected = st.selectbox("Configuration", ids, format_func=names.get, key=select_key)
    if study is not None and roots is not None:
        from ifvg_rules import render_strategy_rules

        with st.expander("Rules for this configuration"):
            render_strategy_rules(
                st,
                study,
                roots,
                selected_core_replay_id=selected,
                show_heading=False,
            )
    policy = charter.payload.objective_policy
    readings = strategy_readings(bundle["metrics_by_child"].get(selected), policy.feasibility_gates)
    if bundle["metrics_by_child"].get(selected) is None:
        st.warning(
            "This configuration has no verified costed measurements. Its performance is unresolved."
        )
    prop_required = bool(charter.payload.authorized_firm_contract_ids)
    if prop_required:
        prop = prop_readings(
            bundle["prop_vectors"].get(selected, {}), policy.prop_feasibility_gates
        )
        metric_cards(
            st,
            prop,
            (
                "first_payout_probability_60d",
                "expected_net_payout_90d",
                "breach_probability_90d",
                "p10_net_payout_90d",
            ),
        )
        st.caption(
            "Simulated prop outcomes · Values summarize the worst outcome across "
            "the saved simulations."
        )
        if not bundle["prop_vectors"].get(selected):
            st.warning(
                "Required prop simulation evidence is missing; prop feasibility is unresolved."
            )
    else:
        metric_cards(st, readings)
    metrics = bundle["metrics_by_child"].get(selected)
    interval = getattr(metrics, "net_expectancy_bootstrap_ci95", None)
    if interval is not None:
        low, high = interval
        st.caption(f"Net expectancy 95% interval: [{low:+.3f}, {high:+.3f}] R")
        if low <= 0 <= high:
            st.warning(
                "The interval includes zero; a positive edge is not established by this sample."
            )
        if not policy.feasibility_gates.require_bootstrap_ci_excludes_zero:
            st.caption("Excluding zero is not a required gate in this study.")
    if metrics is not None:
        from alpha_lab.agents.data_infra.ifvg.search.gates import evaluate_strategy_gates

        gate_report = evaluate_strategy_gates(metrics, policy.feasibility_gates)
        for check in gate_report.checks:
            if not check.passed:
                st.warning(research_sentence(check.explanation))
    if len(children) > 1:
        _comparison(st, bundle, children, names)
    else:
        table = reading_table(readings[:4])
        st.dataframe(table, hide_index=True, width="stretch")
    if st.checkbox("Show research details", key="ifvg_research_result_details"):
        table = reading_table(readings)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download research metrics",
            pd.DataFrame(table).to_csv(index=False),
            file_name="study-metrics.csv",
            mime="text/csv",
        )
        if prop_required:
            st.dataframe(reading_table(prop), hide_index=True, width="stretch")
            _prop_details(
                st, bundle["prop_vectors"].get(selected, {}), policy.prop_feasibility_gates
            )
        if len(children) > 1:
            rows = []
            for child in children:
                metrics = bundle["metrics_by_child"].get(child["core_replay_id"])
                rows.append(
                    {
                        "Configuration": names[child["core_replay_id"]],
                        **{
                            describe(key).human_name: getattr(metrics, key, None)
                            for key in PRIMARY_METRICS
                        },
                    }
                )
            st.subheader("Configuration sensitivity")
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "These are exploratory measurements. Missing fold or stress "
                "evidence cannot establish consistency."
            )
    _review_link(st, bundle, selected)


def _prop_details(st, summaries, gates):
    from ifvg_research_review import words

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import human_name

    if not summaries:
        return
    keys = list(summaries)
    labels = {
        key: f"{index + 1}. {human_name(key, 'Saved firm scenario')}"
        for index, key in enumerate(keys)
    }
    selected = st.selectbox("Firm and scenario", keys, format_func=labels.get)
    summary = summaries[selected]
    mode = summary.get("simulation_mode")
    st.caption("Simulation scope: " + (words(mode) if mode else "Not recorded"))
    st.dataframe(
        reading_table(prop_readings({selected: summary}, gates)), hide_index=True, width="stretch"
    )
    curve = summary.get("survival_curve") or {}
    days, survival = curve.get("days", ()), curve.get("survival", ())
    if days and len(days) == len(survival):
        table = pd.DataFrame({"Day": days, "Survival probability": survival})
        st.plotly_chart(px.line(table, x="Day", y="Survival probability"), width="stretch")
        st.dataframe(table, hide_index=True, width="stretch")
    samples = summary.get("payout_samples") or {}
    if samples:
        horizon = st.selectbox("Payout horizon", list(samples), format_func=words)
        table = pd.DataFrame({"Simulated payout ($)": samples[horizon]})
        st.plotly_chart(px.histogram(table, x="Simulated payout ($)"), width="stretch")
        st.dataframe(table, hide_index=True, width="stretch")


def _comparison(st, bundle, children, names):
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
        AXIS_VALUE_REGISTRY_V1,
        SEARCH_AXIS_REGISTRY_V1,
    )
    from alpha_lab.agents.data_infra.ifvg.search.lineage import derive_lineage_validity

    baseline = next(
        (row for row in children if row.get("comparison_role") == "baseline"), children[0]
    )
    choices = [row["core_replay_id"] for row in children if row is not baseline]
    selected = st.selectbox(
        "Compare with baseline", choices, format_func=names.get, key="ifvg_research_comparison"
    )
    challenger = next(row for row in children if row["core_replay_id"] == selected)
    left_values, right_values = (
        baseline.get("axis_value_ids") or {},
        challenger.get("axis_value_ids") or {},
    )
    changed = sorted(
        key
        for key in set(left_values) | set(right_values)
        if left_values.get(key) != right_values.get(key)
    )
    differences = []
    for key in changed:
        spec = SEARCH_AXIS_REGISTRY_V1.get(key)
        left = AXIS_VALUE_REGISTRY_V1.get(left_values.get(key))
        right = AXIS_VALUE_REGISTRY_V1.get(right_values.get(key))
        differences.append(
            {
                "Setting": spec.human_label if spec else "Unregistered setting",
                "Baseline": axis_value_name(left.value_id) if left else "Unavailable",
                "Challenger": axis_value_name(right.value_id) if right else "Unavailable",
            }
        )
    if differences:
        st.dataframe(differences, hide_index=True, width="stretch")
    compatible, _reason = derive_lineage_validity(tuple(changed))
    if not compatible and changed:
        st.warning(
            "These configurations have incompatible populations. Only setting "
            "differences can be interpreted; performance differences are unavailable."
        )
        return
    left = bundle["metrics_by_child"].get(baseline["core_replay_id"])
    right = bundle["metrics_by_child"].get(selected)
    if left is None or right is None:
        st.warning(
            "One comparison side is missing its costed evaluation. Performance "
            "differences are unavailable."
        )
        return
    rows = []
    for key in PRIMARY_METRICS:
        a, b = getattr(left, key, None), getattr(right, key, None)
        rows.append(
            {
                "Metric": describe(key).human_name,
                "Baseline": a,
                "Challenger": b,
                "Difference": b - a if a is not None and b is not None else None,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "Both sides use the same study dates and cost policy. Differences are "
        "descriptive; paired uncertainty is not established by this table."
    )
    chart_rows = [
        {"Configuration": name, "Net expectancy (R)": value}
        for name, value in (
            ("Baseline", getattr(left, "net_expectancy_r", None)),
            ("Challenger", getattr(right, "net_expectancy_r", None)),
        )
    ]
    st.plotly_chart(
        px.bar(
            pd.DataFrame(chart_rows),
            x="Configuration",
            y="Net expectancy (R)",
            color="Configuration",
        ),
        width="stretch",
        key="ifvg_research_primary_comparison",
    )


def _review_link(st, bundle, selected):
    if bundle.get("search_id"):
        if st.button("Review this configuration's trades", type="primary"):
            st.session_state["ifvg_search_review_pending"] = (bundle["search_id"], selected)
            st.session_state["ifvg_study_v1_open_review"] = True
            st.rerun()
        return
    # A replay link is offered only when the selected child supplies an exact
    # supporting case and a corresponding pair. No baseline fallback is allowed.
    child = next((row for row in bundle["children"] if row.get("core_replay_id") == selected), {})
    case = next(
        (
            (kind, child.get(kind))
            for kind in ("candidate_id", "trade_id", "setup_id")
            if child.get(kind)
        ),
        None,
    )
    if case and child.get("pair_label"):
        if st.button("Inspect supporting trade", type="primary"):
            from ifvg_ui_common import queue_replay_drilldown

            if queue_replay_drilldown(st, case[0], case[1], pair_label=child["pair_label"]):
                st.session_state["ifvg_study_v1_open_review"] = True
                st.rerun()
    else:
        st.caption(
            "A supporting trade link is not saved for this configuration. Trade "
            "review remains available for separately verified evidence."
        )
