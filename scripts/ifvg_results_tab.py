"""Results and History surfaces for the study workspace (R4; FUX §§17–24, 29).

``render_results`` renders the common frame (search picker by full identity,
Summary/Analyst/Audit disclosure, persistent development badge, explicit
result-scope and gross/cost/net labels), the overview cards with the exact
no-pass copy, the payout-reliability frontier with its always-present
selectbox twin, the sensitivity heatmap with glyph classes and a table twin,
the firm matrix / survival / payout views, and the indexed configuration
explorer. ``render_history`` renders the five History sections with the
draft-vs-immutable separation, annotation-only edits, and Clone as New
Search. Every load is exact-ID + manifest-verified; absent artifacts render
their §31 state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_results_charts import (
    build_firm_matrix_figure,
    build_frontier_figure,
    build_payout_distribution_figure,
    build_sensitivity_heatmap,
    build_survival_figure,
)
from ifvg_ui_common import (
    STATE_PREFIX,
    dev_only_badge,
    disclosure_level,
    display_metric,
    identity_block,
    paginate_controls,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    status_badge,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.presentation.flows import FLOW_STEP_TITLES
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    PURPOSE_LABELS,
    PURPOSE_NAMESPACE_CLASS,
    RunPurpose,
)
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    DraftError,
    archive_draft,
    bulk_archive_empty_untitled_drafts,
    clone_draft,
    delete_draft_permanently,
    is_empty_untitled_draft,
    list_drafts,
    restore_draft,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    EXPLORER_PRESETS,
    EXPLORER_STICKY_COLUMNS,
    FIRM_MATRIX_METRICS,
    HEATMAP_METRICS,
    PAYOUT_HORIZONS,
    child_row_presentation,
    heatmap_cell_class,
    human_config_name,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    catalog_annotations,
    list_search_runs,
    load_charter,
    load_child_metrics,
    load_frontier_for_state,
    load_prop_vectors,
    load_search_state,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    NO_PASS_SENTENCE,
    ResultScope,
    StudyStatusKey,
)

__all__ = ["render_results", "render_history", "load_results_bundle"]

_RES = f"{STATE_PREFIX}res_"
_RESULTS_SEARCH_KEY = f"{STATE_PREFIX}results_search_id"
_SELECTED_CONFIG_KEY = f"{STATE_PREFIX}selected_config"

#: simulation_mode → ResultScope (FUX §5.2 — captions derive from the
#: PERSISTED mode, never a hardcoded scope).
_MODE_TO_SCOPE: Mapping[str, ResultScope] = {
    "historical_closed_trade": ResultScope.PROP_HISTORICAL_CLOSED_TRADE,
    "historical_1m_scenario": ResultScope.PROP_1M_SCENARIO,
    "historical_ordered_event_replay": ResultScope.PROP_ORDERED_EVENT_REPLAY,
    "day_block_bootstrap": ResultScope.BOOTSTRAP_SIMULATION,
    "stress": ResultScope.STRESS_SIMULATION,
}


def _prop_scope_caption(vectors: Mapping[str, Mapping[str, dict]]) -> str:
    """The distinct §5.2 scope labels of the persisted prop artifacts."""

    from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: PLC0415
        RESULT_SCOPE_LABELS,
    )

    modes = {
        str(summary.get("simulation_mode", ""))
        for summaries in vectors.values()
        for summary in summaries.values()
    }
    labels = sorted(
        RESULT_SCOPE_LABELS[_MODE_TO_SCOPE[mode]]
        for mode in modes
        if mode in _MODE_TO_SCOPE
    )
    return " · ".join(labels) if labels else "no persisted prop artifacts"


_STRATEGY_COLUMNS: Mapping[str, str] = {
    "trade count": "executed_trades",
    "net E[R]": "net_expectancy_r",
    "realized payoff ratio": "realized_payoff_ratio",
    "profit factor": "profit_factor",
    "max DD R": "max_drawdown_r",
    "trade frequency": "trade_frequency",
    "setup occupancy": "setup_occupancy",
}

_PROP_COLUMNS: Mapping[str, str] = {
    "first payout": "first_payout_probability_60d",
    "3 payouts": "three_payout_probability",
    "expected payout": "expected_net_payout_90d",
    "Q10 payout": "p10_net_payout_90d",
    "replacement cost": "expected_replacement_cost",
}


def load_results_bundle(
    *, store_root: Path, state_root: Path, search_id: str
) -> dict[str, Any]:
    """Everything the Results surfaces render, loaded exact-ID-only."""

    state = load_search_state(state_root, search_id)
    charter = load_charter(store_root, search_id)
    frontier_envelope = load_frontier_for_state(store_root, state)
    children = list((state or {}).get("children") or ())
    metrics_by_child: dict[str, Any] = {}
    if charter is not None:
        for child in children:
            core_id = str(child.get("core_replay_id") or "")
            if not core_id:
                continue
            metrics = load_child_metrics(
                store_root, core_id, charter.payload.cost_policy
            )
            if metrics is not None:
                metrics_by_child[core_id] = metrics
    prop_vectors = load_prop_vectors(store_root)
    frontier = frontier_envelope.payload.frontier if frontier_envelope else None
    names = {
        str(child.get("core_replay_id") or ""): human_config_name(
            child.get("axis_value_ids") or {}
        )
        for child in children
    }
    return {
        "search_id": search_id,
        "state": state,
        "charter": charter,
        "frontier": frontier,
        "children": children,
        "metrics_by_child": metrics_by_child,
        "prop_vectors": prop_vectors,
        "names": names,
    }


def _vector_value(
    summaries: Mapping[str, dict[str, Any]], attribute: str, *, worst: bool = True
) -> float | None:
    """Worst-across-simulations value for one vector attribute (D-#18)."""

    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        OBJECTIVE_DIRECTIONS,
    )

    values = [
        summary.get("payout_reliability_vector", {}).get(attribute)
        for summary in summaries.values()
    ]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if not worst:
        return float(values[0])
    direction = OBJECTIVE_DIRECTIONS.get(attribute, "maximize")
    return float(min(values) if direction == "maximize" else max(values))


def _config_selector(
    st_module, bundle: Mapping[str, Any], *, key: str, eligible: Sequence[str]
) -> str | None:
    """The always-present accessible selectbox twin (FUX §19)."""

    names = bundle["names"]
    options = {
        f"{names.get(core_id, core_id[:12])} · {core_id[:12]}…": core_id
        for core_id in eligible
    }
    if not options:
        return None
    sanitize_select(st_module, key, list(options))
    preselected = st_module.session_state.get(_SELECTED_CONFIG_KEY)
    index = next(
        (
            position
            for position, core_id in enumerate(options.values())
            if core_id == preselected
        ),
        0,
    )
    chosen = st_module.selectbox(
        "Selected configuration", list(options), index=index, key=key
    )
    selected = options[chosen]
    st_module.session_state[_SELECTED_CONFIG_KEY] = selected
    return selected


def render_results(st_module=st, *, roots: Mapping[str, Any]) -> None:
    dev_only_badge(st_module)
    store_root = Path(roots["store_root"])
    state_root = Path(roots["state_root"])
    runs = [
        run
        for run in list_search_runs(
            state_root, store_root, store_roots=_known_store_roots(roots)
        )
        if not run.archived
    ]
    if not runs:
        # UI-1 (plan §6.7): an empty job root is the NO_RUNS state — nothing
        # is missing or corrupt
        render_empty_state(
            st_module,
            "no_runs",
            detail="no completed or running searches exist in the job root",
        )
        return
    labels = {f"{run.display_name} · {run.search_id[:12]}…": run for run in runs}
    preselected = st_module.session_state.get(_RESULTS_SEARCH_KEY)
    index = next(
        (
            position
            for position, run in enumerate(labels.values())
            if run.search_id == preselected
        ),
        0,
    )
    sanitize_select(st_module, f"{_RES}run", list(labels))
    chosen = st_module.selectbox(
        "Search / run", list(labels), index=index, key=f"{_RES}run"
    )
    run = labels[chosen]
    identity_block(st_module, "Full search identity", run.search_id)
    level = disclosure_level(st_module, key=f"{_RES}disclosure")
    # UI-1 (plan F-01): the run's OWN store (located by its exact charter id)
    # is the read location, and the verification badge keys on the ARTIFACT
    # (a synthetic-marker or verification charter), never on a selector
    run_store = Path(run.store_root) if run.store_root else store_root
    bundle = load_results_bundle(
        store_root=run_store, state_root=state_root, search_id=run.search_id
    )
    _artifact_scope_caption(st_module, run)
    roots = {**roots, "store_root": run_store}

    _render_overview(st_module, bundle)
    if level in ("analyst", "audit"):
        _render_frontier(st_module, bundle)
        _render_heatmap(st_module, bundle)
        _render_firm_views(st_module, bundle)
    _render_explorer(st_module, bundle, level)
    if level == "audit":
        _render_audit_block(st_module, bundle)

    with st_module.expander("Comparison, insights, and account timeline"):
        from ifvg_results_compare import (  # noqa: PLC0415
            render_account_timeline,
            render_comparison,
            render_insights_for_bundle,
        )

        render_comparison(st_module, roots=roots, bundle=bundle)
        render_insights_for_bundle(st_module, bundle)
        render_account_timeline(st_module, roots=roots)


def _known_store_roots(roots: Mapping[str, Any]) -> list[Path]:
    """The default read root plus every deployed store by class (the run's
    charter is located by exact id across them; the stores are never listed)."""

    known: list[Path] = [Path(roots["store_root"])]
    for root in dict(roots.get("store_roots") or {}).values():
        if Path(root) not in known:
            known.append(Path(root))
    return known


def _artifact_scope_caption(st_module, run) -> None:
    """The artifact-derived scope: the store's verified namespace class and
    whether the artifact is verification-only (badge from the artifact)."""

    if run.verification_only:
        verification_badge(st_module)
    namespace = run.namespace_class or "unmarked"
    scope = (
        "verification-only artifact (a synthetic fixture or the verification "
        "slice) — never research evidence"
        if run.verification_only
        else "owner-authorized development artifact"
        if run.verification_only is False
        else "artifact scope unresolved (charter not located in a known store)"
    )
    st_module.caption(f"Artifact scope: store namespace class `{namespace}` · {scope}")


def _render_overview(st_module, bundle: Mapping[str, Any]) -> None:
    """FUX §18 — the four cards or the exact no-pass sentence."""

    frontier = bundle["frontier"]
    children = bundle["children"]
    names = bundle["names"]
    st_module.subheader("Overview")
    st_module.caption(
        f"Result scope: **{_prop_scope_caption(bundle['prop_vectors'])}** "
        "(prop cards) · **Actual Executed Strategy** (strategy metrics); "
        "gross, costed, and net values are labeled separately below."
    )
    phase = str((bundle.get("state") or {}).get("phase") or "")
    if frontier is None or not frontier.feasible_ids:
        if phase != "search_complete":
            # FUX §18's no-pass sentence is a TERMINAL verdict — an
            # in-progress run must never display it (adversarial F19).
            st_module.info(
                f"Run in progress (phase: {phase or 'unknown'}) — the "
                "overview verdict appears when the search completes."
            )
            return
        st_module.markdown(f"### {NO_PASS_SENTENCE}")
        reasons: dict[str, int] = {}
        for child in children:
            reason = child.get("failure_reason")
            if reason:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        if reasons:
            st_module.write("Dominant failure reasons (children stopped per gate):")
            st_module.table(
                {
                    "failure reason": list(reasons.keys()),
                    "children stopped": [str(count) for count in reasons.values()],
                }
            )
        else:
            st_module.caption(
                "No failure reasons recorded — the run completed without "
                "producing evaluable children."
            )
        return
    cards = st_module.columns(4)
    champions = dict(frontier.per_objective_champions)
    representative = frontier.development_exploratory_representative_id

    def _card(column, title: str, core_id: str | None) -> None:
        with column:
            st_module.markdown(f"**{title}**")
            if core_id:
                st_module.write(names.get(core_id, core_id[:12] + "…"))
                st_module.code(core_id, language=None)
            else:
                st_module.caption("not available")

    _card(cards[0], "Development Exploratory Representative", representative)
    _card(
        cards[1],
        "Highest Expected Payout",
        champions.get("expected_net_payout_90d"),
    )
    _card(
        cards[2],
        "Highest Payout Reliability",
        champions.get("payout_probability_per_rolling_30d"),
    )
    _card(cards[3], "Lowest Breach Risk", champions.get("breach_probability_90d"))
    st_module.caption(
        "These are ranking-dimension titles, not a declaration of a "
        "universally best strategy. Tie-break trace and dominance edges are "
        "in Audit disclosure."
    )


def _render_frontier(st_module, bundle: Mapping[str, Any]) -> None:
    st_module.subheader("Payout-reliability frontier")
    st_module.caption(
        f"Result scope: **{_prop_scope_caption(bundle['prop_vectors'])}**"
    )
    frontier = bundle["frontier"]
    vectors = bundle["prop_vectors"]
    names = bundle["names"]
    if frontier is None:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="no persisted frontier exists for this search yet",
        )
        return
    rows = []
    for core_id in frontier.feasible_ids:
        summaries = vectors.get(core_id, {})
        expected = _vector_value(summaries, "expected_net_payout_90d")
        reliability = _vector_value(
            summaries, "payout_probability_per_rolling_30d"
        )
        breach = _vector_value(summaries, "breach_probability_90d")
        lifetime = _vector_value(summaries, "median_account_lifetime_days")
        if expected is None or reliability is None or breach is None:
            continue
        rows.append(
            {
                "config_id": core_id,
                "name": names.get(core_id, core_id[:12] + "…"),
                "expected_net_payout_90d": expected,
                "payout_probability_per_rolling_30d": reliability,
                "breach_probability_90d": breach,
                "median_account_lifetime_days": lifetime,
                "feasible": True,
                "firm_context": ", ".join(sorted(summaries)) or "—",
                "evidence_scope": next(
                    (
                        str(summary.get("simulation_mode"))
                        for summary in summaries.values()
                    ),
                    "—",
                ),
            }
        )
    if not rows:
        # These configurations PASSED the strategy gate; what is absent is
        # the persisted simulation artifact — never render the §16.4
        # gate-skip sentence here (adversarial F9a).
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail=(
                "no persisted account-simulation summaries exist for the "
                "feasible configurations (the R5 pipeline persists them)"
            ),
        )
        return
    selected = _config_selector(
        st_module,
        bundle,
        key=f"{_RES}frontier_twin",
        eligible=[row["config_id"] for row in rows],
    )
    figure, omissions = build_frontier_figure(rows, selected_id=selected)
    # FUX §19 primary interaction: selecting a point updates the page (the
    # selectbox above stays the always-present accessible/AppTest twin).
    try:
        event = st_module.plotly_chart(
            figure,
            width="stretch",
            key=f"{_RES}frontier_fig",
            on_select="rerun",
            selection_mode="points",
        )
        points = getattr(getattr(event, "selection", None), "points", None) or []
        for point in points:
            customdata = point.get("customdata") or []
            if customdata and customdata[0] != selected:
                st_module.session_state[_SELECTED_CONFIG_KEY] = customdata[0]
                st_module.rerun()
    except TypeError:
        # older Streamlit without on_select: the selectbox twin carries the
        # complete semantics (FUX §4.1 fallback)
        st_module.plotly_chart(
            figure, width="stretch", key=f"{_RES}frontier_fig"
        )
    for line in omissions.summary_lines():
        st_module.caption(f"Omission: {line}")


def _render_heatmap(st_module, bundle: Mapping[str, Any]) -> None:
    st_module.subheader("Parameter-sensitivity heatmap")
    children = bundle["children"]
    metrics_by_child = bundle["metrics_by_child"]
    vectors = bundle["prop_vectors"]
    axes = sorted(
        {
            axis
            for child in children
            for axis in (child.get("axis_value_ids") or {})
        }
    )
    if len(axes) < 1:
        st_module.caption("No searched axes — the heatmap needs at least one.")
        return
    row_axis = st_module.selectbox(
        "Row axis", axes, key=f"{_RES}heat_row"
    )
    col_options = ["(single axis)"] + [axis for axis in axes if axis != row_axis]
    col_axis = st_module.selectbox(
        "Column axis", col_options, key=f"{_RES}heat_col"
    )
    metric_label = st_module.selectbox(
        "Metric", list(HEATMAP_METRICS), key=f"{_RES}heat_metric"
    )
    attribute = HEATMAP_METRICS[metric_label]
    from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: PLC0415
        RESULT_SCOPE_LABELS,
    )

    is_prop_metric = attribute in (
        "expected_net_payout_90d",
        "breach_probability_90d",
        "payout_probability_per_rolling_30d",
    )
    scope_label = (
        _prop_scope_caption(vectors)
        if is_prop_metric
        else RESULT_SCOPE_LABELS[ResultScope.ACTUAL_EXECUTED_STRATEGY]
    )
    st_module.caption(
        f"Result scope: **{scope_label}** · net values (post-cost)"
    )
    # One entry per child first; then AGGREGATE per (row, col) coordinate —
    # collapsing children into a cell is reported, never silent (F5).
    per_child = []
    for child in children:
        core_id = str(child.get("core_replay_id") or "")
        values = child.get("axis_value_ids") or {}
        metrics = metrics_by_child.get(core_id)
        value = None
        if metrics is not None and hasattr(metrics, attribute):
            value = getattr(metrics, attribute)
        elif core_id in vectors:
            value = _vector_value(vectors[core_id], attribute)
        per_child.append(
            {
                "row_value": values.get(row_axis, "—"),
                "col_value": (
                    values.get(col_axis, "—")
                    if col_axis != "(single axis)"
                    else "·"
                ),
                "value": value,
                "samples": int(
                    getattr(metrics, "executed_trades", 0) if metrics else 0
                ),
                "blocked": child.get("state") == "blocked",
                "failed": child.get("failure_reason") is not None
                and value is None,
                "knife_edge": str(child.get("failure_reason") or "")
                == "knife_edge",
                "config": bundle["names"].get(core_id, core_id[:12]),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in per_child:
        grouped.setdefault(
            (str(entry["row_value"]), str(entry["col_value"])), []
        ).append(entry)
    cells = []
    aggregated_notes: list[str] = []
    for (row_value, col_value), members in sorted(grouped.items()):
        numeric = [m["value"] for m in members if m["value"] is not None]
        value = sum(numeric) / len(numeric) if numeric else None
        samples = sum(m["samples"] for m in members)
        cell_class = heatmap_cell_class(
            value=value,
            sample_count=samples,
            blocked=all(m["blocked"] for m in members),
            failed=bool(members) and all(m["failed"] for m in members),
            knife_edge=any(m["knife_edge"] for m in members),
        )
        evidence = (
            f"mean of {len(numeric)} configuration(s)"
            if len(members) > 1
            else "single configuration"
        )
        if len(members) > 1:
            aggregated_notes.append(
                f"cell ({row_value}, {col_value}): {len(members)} "
                "configurations aggregated (mean)"
            )
        cells.append(
            {
                "row_value": row_value,
                "col_value": col_value,
                "value": value,
                "cell_class": cell_class,
                "sample_count": samples,
                "configs": ", ".join(sorted(m["config"] for m in members)),
                "evidence_reason": evidence,
            }
        )
    figure, omissions = build_sensitivity_heatmap(
        cells,
        row_axis=row_axis,
        col_axis=col_axis if col_axis != "(single axis)" else "—",
        metric_label=metric_label,
        metric_key=attribute,
    )
    st_module.plotly_chart(figure, width="stretch", key=f"{_RES}heat_fig")
    for note in aggregated_notes:
        st_module.caption(f"Omission/aggregation: {note}")
    for line in omissions.summary_lines():
        st_module.caption(f"Omission: {line}")
    st_module.markdown(
        "**Table twin** (same cells: status + value + samples + evidence)"
    )
    st_module.dataframe(
        [
            {
                "configs": cell["configs"],
                row_axis: cell["row_value"],
                (col_axis if col_axis != "(single axis)" else "—"): cell[
                    "col_value"
                ],
                "status": cell["cell_class"],
                "value": display_metric(cell["value"]),
                "samples": cell["sample_count"],
                "evidence reason": cell["evidence_reason"],
            }
            for cell in cells
        ],
        width="stretch",
        hide_index=True,
    )


def _render_firm_views(st_module, bundle: Mapping[str, Any]) -> None:
    vectors = bundle["prop_vectors"]
    names = bundle["names"]
    st_module.subheader("Firm compatibility matrix")
    st_module.caption(
        f"Result scope: **{_prop_scope_caption(vectors)}** · net of fees"
    )
    relevant = {
        core_id: summaries
        for core_id, summaries in vectors.items()
        if core_id in names
    }
    if not relevant:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail=(
                "no account simulations are persisted for this search's "
                "configurations (the R5 pipeline persists them)"
            ),
        )
        return
    metric_label = st_module.selectbox(
        "Matrix metric", list(FIRM_MATRIX_METRICS), key=f"{_RES}firm_metric"
    )
    attribute = FIRM_MATRIX_METRICS[metric_label]
    cells = []
    for core_id, summaries in relevant.items():
        for firm_label, summary in summaries.items():
            value = summary.get("payout_reliability_vector", {}).get(attribute)
            cells.append(
                {
                    "config_name": names.get(core_id, core_id[:12]),
                    "firm_label": firm_label,
                    "value": value,
                    "status_text": "no data" if value is None else "",
                    "universal": True,  # one universal strategy per charter
                }
            )
    figure, omissions = build_firm_matrix_figure(
        cells, metric_label=metric_label, metric_key=attribute
    )
    st_module.plotly_chart(figure, width="stretch", key=f"{_RES}firm_fig")
    for line in omissions.summary_lines():
        st_module.caption(f"Omission: {line}")

    st_module.subheader("Account survival curves")
    selected = _config_selector(
        st_module,
        bundle,
        key=f"{_RES}survival_twin",
        eligible=list(relevant),
    )
    curves = []
    if selected:
        for firm_label, summary in relevant.get(selected, {}).items():
            curve = summary.get("survival_curve")
            if curve:
                curves.append(
                    {
                        "label": firm_label,
                        "days": curve.get("days", ()),
                        "survival": curve.get("survival", ()),
                    }
                )
        mode = next(
            (
                str(summary.get("simulation_mode"))
                for summary in relevant.get(selected, {}).values()
            ),
            "—",
        )
        st_module.caption(
            f"Simulation mode / path capability: **{mode}** — assumed 1m "
            "paths are always a scenario/approximation, never exact "
            "historical chronology."
        )
    if curves:
        figure, omissions = build_survival_figure(curves)
        st_module.plotly_chart(
            figure, width="stretch", key=f"{_RES}surv_fig"
        )
        table_rows = []
        for curve in curves:
            days = list(curve["days"])
            survival = list(curve["survival"])
            for horizon in (30, 60, 90):
                value = next(
                    (
                        s
                        for d, s in zip(days, survival, strict=False)
                        if d >= horizon
                    ),
                    survival[-1] if survival else None,
                )
                table_rows.append(
                    {
                        "firm/policy": curve["label"],
                        "horizon (days)": horizon,
                        "P(alive)": display_metric(value, percent=True),
                    }
                )
        st_module.dataframe(table_rows, width="stretch", hide_index=True)
    else:
        st_module.caption(
            "No survival-curve blocks are persisted for this configuration."
        )

    st_module.subheader("Payout distributions")
    horizon = st_module.selectbox(
        "Horizon", PAYOUT_HORIZONS, key=f"{_RES}payout_horizon"
    )
    samples: list[float] = []
    if selected:
        for summary in relevant.get(selected, {}).values():
            block = (summary.get("payout_samples") or {}).get(horizon)
            if block:
                samples.extend(float(value) for value in block)
    if samples:
        ordered = sorted(samples)

        def _quantile(fraction: float) -> float:
            index = min(
                len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1)))
            )
            return ordered[index]

        quantiles = {
            "mean": sum(ordered) / len(ordered),
            "median": _quantile(0.5),
            "p10": _quantile(0.10),
            "p90": _quantile(0.90),
        }
        st_module.markdown(
            f"**Lower tail first:** P10 = {display_metric(quantiles['p10'])} · "
            f"mean {display_metric(quantiles['mean'])} · median "
            f"{display_metric(quantiles['median'])} · P90 "
            f"{display_metric(quantiles['p90'])}"
        )
        figure, omissions = build_payout_distribution_figure(
            ordered, horizon_label=horizon, quantiles=quantiles
        )
        st_module.plotly_chart(
            figure, width="stretch", key=f"{_RES}payout_fig"
        )
        st_module.caption(
            "Net of fees, withdrawals, and replacements per the simulated "
            "account policy set; simulation mode and path fidelity are "
            "captioned above."
        )
    else:
        st_module.caption(
            "No payout-sample blocks are persisted for this configuration "
            "and horizon."
        )


def _render_explorer(st_module, bundle: Mapping[str, Any], level: str) -> None:
    st_module.subheader("Configuration explorer")
    st_module.caption(
        "Result scope: **Actual Executed Strategy** (Strategy preset) · "
        f"**{_prop_scope_caption(bundle['prop_vectors'])}** (Prop preset); "
        "gross vs net is explicit per column."
    )
    children = bundle["children"]
    metrics_by_child = bundle["metrics_by_child"]
    vectors = bundle["prop_vectors"]
    names = bundle["names"]
    frontier = bundle["frontier"]
    frontier_ids = tuple(frontier.frontier_ids) if frontier else ()
    feasible_ids = tuple(frontier.feasible_ids) if frontier else ()
    representative = (
        frontier.development_exploratory_representative_id if frontier else None
    )
    preset = st_module.radio(
        "Column preset",
        list(EXPLORER_PRESETS),
        horizontal=True,
        key=f"{_RES}preset",
    )
    rows = []
    for child in children:
        core_id = str(child.get("core_replay_id") or "")
        presentation = child_row_presentation(
            child,
            feasible_ids=feasible_ids,
            frontier_ids=frontier_ids,
            representative_id=representative,
            config_name=names.get(core_id),
        )
        rank = (
            frontier_ids.index(core_id) + 1 if core_id in frontier_ids else None
        )
        row: dict[str, Any] = {
            "rank": rank if rank is not None else "—",
            "config name": presentation.config,
            "status": presentation.as_row()["Status"],
            "changed parameters": ", ".join(
                sorted(presentation.axis_value_ids)
            )
            or "baseline",
        }
        metrics = metrics_by_child.get(core_id)
        if preset == "Strategy":
            for label, attribute in _STRATEGY_COLUMNS.items():
                value = getattr(metrics, attribute, None) if metrics else None
                if value is None and metrics is not None:
                    stats = getattr(metrics, "trade_stats", None) or {}
                    value = (
                        stats.get(attribute) if isinstance(stats, dict) else None
                    )
                row[label] = display_metric(value)
        elif preset == "Prop":
            summaries = vectors.get(core_id, {})
            row["firm"] = ", ".join(sorted(summaries)) or "—"
            row["risk policy"] = next(
                (
                    str(summary.get("risk_policy_label", "—"))
                    for summary in summaries.values()
                ),
                "—",
            )
            for label, attribute in _PROP_COLUMNS.items():
                row[label] = display_metric(
                    _vector_value(summaries, attribute)
                )
            breach = _vector_value(summaries, "breach_probability_90d")
            # survival is 1 − P(breach ≤ 90d): the SURVIVAL quantity, never
            # the breach probability under a survival title (F1)
            row["90-day survival"] = display_metric(
                None if breach is None else 1.0 - breach, percent=True
            )
            row["fees"] = display_metric(
                next(
                    (
                        summary.get("total_fees")
                        for summary in summaries.values()
                        if summary.get("total_fees") is not None
                    ),
                    None,
                )
            )
        else:  # Robustness
            row["outer folds passed"] = "— (schema-reserved; exploratory lane)"
            row["stress tests passed"] = "— (no stress suite persisted)"
            row["neighbor stability"] = (
                "knife-edge"
                if str(child.get("failure_reason") or "") == "knife_edge"
                else "—"
            )
            summaries = vectors.get(core_id, {})
            row["worst-firm result"] = display_metric(
                _vector_value(summaries, "expected_net_payout_90d")
            )
            row["concentration warning"] = (
                "yes"
                if metrics is not None
                and (
                    getattr(metrics, "top_day_pnl_share", 0) > 0.4
                    or getattr(metrics, "top_setup_pnl_share", 0) > 0.25
                )
                else "no"
            )
        row["core_replay_id"] = core_id
        rows.append(row)
    rows.sort(
        key=lambda row: (
            isinstance(row["rank"], str),
            row["rank"] if isinstance(row["rank"], int) else 0,
            row["config name"],
        )
    )
    start, end = paginate_controls(st_module, len(rows), key=f"{_RES}explorer")
    page = rows[start:end]
    # FUX §4.1 pinned-column fallback: the selected configuration's
    # identity/details repeat ABOVE the table, and the identity columns
    # lead every preset.
    detail_labels = {
        f"{row['config name']} · {row['core_replay_id'][:12]}…": row
        for row in page
        if row.get("core_replay_id")
    }
    selected_row = None
    if detail_labels:
        sanitize_select(st_module, f"{_RES}exp_detail", list(detail_labels))
        chosen = st_module.selectbox(
            "Row detail (repeated above the table)",
            list(detail_labels),
            key=f"{_RES}exp_detail",
        )
        selected_row = detail_labels[chosen]
        identity_block(
            st_module, "core_replay_id", selected_row["core_replay_id"]
        )
        st_module.write(
            f"Changed parameters (first): {selected_row['changed parameters']}"
        )
    display_columns = [
        column
        for column in (list(EXPLORER_STICKY_COLUMNS) + list(page[0].keys())
        if page
        else list(EXPLORER_STICKY_COLUMNS))
        if column != "core_replay_id"
    ]
    seen: list[str] = []
    for column in display_columns:
        if column not in seen:
            seen.append(column)
    st_module.dataframe(
        [
            {column: row.get(column, "—") for column in seen}
            for row in page
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.caption(
        "The identity columns (rank / config name / status / changed "
        "parameters) lead every preset; wide tables scroll horizontally and "
        "the selected configuration's details repeat above the table."
    )
    if selected_row is not None:
        row = selected_row
        if level == "audit":
            child = next(
                (
                    child
                    for child in children
                    if child.get("core_replay_id") == row["core_replay_id"]
                ),
                {},
            )
            st_module.markdown("**Technical full diff (Audit)**")
            st_module.json(child.get("axis_value_ids") or {})


def _render_audit_block(st_module, bundle: Mapping[str, Any]) -> None:
    st_module.subheader("Audit")
    charter = bundle["charter"]
    frontier = bundle["frontier"]
    if charter is not None:
        identity_block(st_module, "search_id", charter.search_id)
        st_module.write(
            f"Baseline: `{charter.payload.baseline_profile_name}` · "
            f"section hash `{charter.payload.baseline_section_config_hash[:16]}…`"
        )
        st_module.write(
            "Commits: SC "
            f"`{charter.payload.strategy_core_commit[:12]}` · QL "
            f"`{charter.payload.quant_lab_commit[:12]}`"
        )
        st_module.write(
            f"Cost policy: {charter.payload.cost_policy.model_dump()}"
        )
    if frontier is not None:
        st_module.markdown("**Tie-break trace**")
        st_module.code("\n".join(frontier.tie_break_trace) or "—", language="text")
        st_module.markdown("**Dominance edges (dominator → dominated)**")
        st_module.code(
            "\n".join(
                f"{left[:12]}… → {right[:12]}…"
                for left, right in frontier.dominance_edges
            )
            or "—",
            language="text",
        )


_PURPOSE_FILTER_ALL = "All purposes"
_PURPOSE_FILTER_UNRESOLVED = "Unresolved purpose"
_STORE_FILTER_ALL = "All stores"
_STORE_FILTER_OPTIONS = (_STORE_FILTER_ALL, "research", "test", "unmarked")


def _draft_purpose_label(draft) -> str:
    purpose = (draft.purpose_annotation or {}).get("purpose")
    try:
        return PURPOSE_LABELS[RunPurpose(str(purpose))]
    except ValueError:
        return _PURPOSE_FILTER_UNRESOLVED


def _draft_store_class(draft) -> str:
    purpose = (draft.purpose_annotation or {}).get("purpose")
    try:
        return PURPOSE_NAMESPACE_CLASS[RunPurpose(str(purpose))]
    except ValueError:
        return "unmarked"


def _draft_step_label(draft) -> str:
    if draft.current_step_key in FLOW_STEP_TITLES:
        return FLOW_STEP_TITLES[str(draft.current_step_key)]
    return f"step {int(draft.step_index) + 1}"


def _run_purpose_label(annotations: Mapping[str, Mapping[str, Any]], search_id: str) -> str:
    entry = annotations.get(search_id) or {}
    payload = entry.get("purpose")
    purpose = payload.get("purpose") if isinstance(payload, Mapping) else payload
    try:
        return PURPOSE_LABELS[RunPurpose(str(purpose))]
    except ValueError:
        return _PURPOSE_FILTER_UNRESOLVED


def _history_filters(st_module) -> tuple[str, str, bool]:
    """UI-2 (plan §5.1 / §7): READ-ONLY listing filters — they narrow this
    page only and never decide where a charter freezes or launches."""

    columns = st_module.columns([2, 2, 1])
    purpose_options = [
        _PURPOSE_FILTER_ALL,
        *(PURPOSE_LABELS[purpose] for purpose in RunPurpose),
        _PURPOSE_FILTER_UNRESOLVED,
    ]
    with columns[0]:
        sanitize_select(st_module, f"{_RES}hist_purpose_filter", purpose_options)
        purpose_filter = st_module.selectbox(
            "Filter by purpose",
            purpose_options,
            key=f"{_RES}hist_purpose_filter",
            help=(
                "Read-only listing filter over the presentation annotation; it never "
                "changes where a charter freezes or launches (the purpose derives that)."
            ),
        )
    with columns[1]:
        sanitize_select(st_module, f"{_RES}hist_store_filter", list(_STORE_FILTER_OPTIONS))
        store_filter = st_module.selectbox(
            "Filter by store namespace class",
            list(_STORE_FILTER_OPTIONS),
            key=f"{_RES}hist_store_filter",
            help=(
                "Read-only listing filter over each run's OWN verified store namespace "
                "class (its charter located by exact id); never a selector of authority."
            ),
        )
    with columns[2]:
        show_archived = st_module.checkbox(
            "Show archived drafts",
            key=f"{_RES}hist_show_archived",
            help=(
                "The archived / advanced view: restore, or permanently delete a never-frozen "
                "draft with its exact typed name."
            ),
        )
    return str(purpose_filter), str(store_filter), bool(show_archived)


def _draft_matches(draft, purpose_filter: str, store_filter: str) -> bool:
    if purpose_filter != _PURPOSE_FILTER_ALL and _draft_purpose_label(draft) != purpose_filter:
        return False
    return store_filter == _STORE_FILTER_ALL or _draft_store_class(draft) == store_filter


def _render_draft_row(st_module, draft, draft_root: Path) -> None:
    columns = st_module.columns([3, 1, 1, 1])
    with columns[0]:
        st_module.write(
            f"✎ **{draft.display_name}** · {draft.mode_id} · "
            f"{_draft_step_label(draft)} · updated {draft.updated_at_utc} · "
            f"{draft.status} · purpose {_draft_purpose_label(draft)} / store "
            f"`{_draft_store_class(draft)}`"
        )
    with columns[1]:
        if st_module.button("Open", key=f"{_RES}open_{draft.draft_id[:8]}"):
            from ifvg_study_tab import request_route  # noqa: PLC0415

            st_module.session_state[f"{STATE_PREFIX}draft_id"] = draft.draft_id
            request_route(st_module, "new_study")
            st_module.rerun()
    with columns[2]:
        if st_module.button("Clone", key=f"{_RES}clone_{draft.draft_id[:8]}"):
            clone = clone_draft(draft)
            save_draft(draft_root, clone)
            st_module.success(f"Cloned to {clone.display_name}")
    with columns[3]:
        if st_module.button(
            "Archive",
            key=f"{_RES}archive_{draft.draft_id[:8]}",
            help="Reversible: hides the draft from this list; nothing is deleted.",
        ):
            try:
                archive_draft(draft_root, draft.draft_id)
                st_module.rerun()
            except DraftError as error:
                st_module.warning(sanitize_error(error))


def _render_archived_row(st_module, draft, draft_root: Path) -> None:
    columns = st_module.columns([3, 1, 2])
    with columns[0]:
        st_module.write(
            f"▣ **{draft.display_name}** · {draft.mode_id} · archived "
            f"{draft.archived_at_utc or ''} · purpose {_draft_purpose_label(draft)}"
        )
    with columns[1]:
        if st_module.button("Restore", key=f"{_RES}restore_{draft.draft_id[:8]}"):
            try:
                restore_draft(draft_root, draft.draft_id)
                st_module.rerun()
            except DraftError as error:
                st_module.warning(sanitize_error(error))
    with columns[2]:
        if not draft.never_frozen:
            st_module.caption(
                "frozen / launched provenance — never deletable (catalog archive flag only)"
            )
            return
        typed = st_module.text_input(
            "Type the exact draft name to delete it permanently",
            key=f"{_RES}del_name_{draft.draft_id[:8]}",
            help=(
                "Owner Q2: permanent deletion only for a draft that was never frozen and "
                "never launched, only from this archived view, only with its exact name."
            ),
        )
        if st_module.button(
            "Delete draft permanently",
            key=f"{_RES}del_{draft.draft_id[:8]}",
            disabled=str(typed or "") != draft.display_name,
        ):
            try:
                delete_draft_permanently(draft_root, draft.draft_id, confirm_name=str(typed))
                st_module.rerun()
            except DraftError as error:
                st_module.warning(sanitize_error(error))


def render_history(st_module=st, *, roots: Mapping[str, Any]) -> None:
    dev_only_badge(st_module)
    draft_root = Path(roots["draft_root"])
    store_root = Path(roots["store_root"])
    state_root = Path(roots["state_root"])
    purpose_filter, store_filter, show_archived = _history_filters(st_module)

    st_module.subheader("Drafts")
    all_drafts = list_drafts(draft_root, include_archived=True)
    # owner Q2: the one-time bulk archive of the R4-era empty untitled files
    empties = [draft for draft in all_drafts if is_empty_untitled_draft(draft)]
    if empties and st_module.button(
        f"Archive empty untitled drafts ({len(empties)})",
        key=f"{_RES}bulk_archive",
        help="One-time migration: archives (never deletes) every empty, unnamed step-0 draft.",
    ):
        try:
            bulk_archive_empty_untitled_drafts(draft_root)
            st_module.rerun()
        except DraftError as error:
            st_module.warning(sanitize_error(error))
    drafts = [
        draft
        for draft in all_drafts
        if draft.status == "draft"
        and not draft.archived
        and _draft_matches(draft, purpose_filter, store_filter)
    ]
    if not drafts:
        st_module.caption("No mutable drafts match the filters.")
    for draft in drafts:
        _render_draft_row(st_module, draft, draft_root)
    if show_archived:
        st_module.subheader("Archived drafts")
        archived = [
            draft
            for draft in all_drafts
            if draft.archived and _draft_matches(draft, purpose_filter, store_filter)
        ]
        if not archived:
            st_module.caption("No archived drafts match the filters.")
        for draft in archived:
            _render_archived_row(st_module, draft, draft_root)

    frozen_drafts = {
        d.frozen_search_id: d
        for d in all_drafts
        if d.status == "frozen" and d.frozen_search_id
    }
    runs = list_search_runs(state_root, store_root, store_roots=_known_store_roots(roots))
    annotations_by_store: dict[str, Mapping[str, Mapping[str, Any]]] = {}

    def _annotations_for(run) -> Mapping[str, Mapping[str, Any]]:
        key = str(run.store_root or store_root)
        if key not in annotations_by_store:
            try:
                annotations_by_store[key] = catalog_annotations(Path(key))
            except Exception:  # noqa: BLE001 — an unreadable catalog annotates nothing
                annotations_by_store[key] = {}
        return annotations_by_store[key]

    def _run_matches(run) -> bool:
        if purpose_filter != _PURPOSE_FILTER_ALL and (
            _run_purpose_label(_annotations_for(run), run.search_id) != purpose_filter
        ):
            return False
        return store_filter == _STORE_FILTER_ALL or (
            (run.namespace_class or "unmarked") == store_filter
        )

    runs = [run for run in runs if _run_matches(run)]
    running = [
        run for run in runs if run.phase not in ("search_complete",) and not run.archived
    ]
    completed = [
        run for run in runs if run.phase == "search_complete" and not run.archived
    ]
    superseded = [run for run in runs if run.archived]

    def _run_row(run, *, immutable: bool) -> None:
        columns = st_module.columns([3, 1, 1, 1])
        with columns[0]:
            reused = 0
            state = load_search_state(state_root, run.search_id)
            if state:
                reused = sum(
                    1
                    for child in state.get("children") or ()
                    if child.get("state") == "reused"
                )
            reuse_note = (
                f" · {reused} verified reuse hit(s)" if reused else ""
            )
            scope_note = (
                " · verification-only"
                if run.verification_only
                else " · development"
                if run.verification_only is False
                else ""
            )
            st_module.write(
                f"🔒 **{run.display_name}** · {run.phase} · "
                f"{run.child_count} children{reuse_note} · store namespace "
                f"`{run.namespace_class or 'unmarked'}`{scope_note} · purpose "
                f"{_run_purpose_label(_annotations_for(run), run.search_id)}"
            )
            st_module.code(run.search_id, language=None)
        with columns[1]:
            if st_module.button("Results", key=f"{_RES}hist_res_{run.search_id[:8]}"):
                from ifvg_study_tab import request_route  # noqa: PLC0415

                st_module.session_state[_RESULTS_SEARCH_KEY] = run.search_id
                request_route(st_module, "results")
                st_module.rerun()
        with columns[2]:
            draft = frozen_drafts.get(run.search_id)
            if draft is not None:
                if st_module.button(
                    "Clone as New Search", key=f"{_RES}hist_clone_{run.search_id[:8]}"
                ):
                    clone = clone_draft(draft)
                    save_draft(draft_root, clone)
                    from ifvg_study_tab import request_route  # noqa: PLC0415

                    st_module.session_state[f"{STATE_PREFIX}draft_id"] = (
                        clone.draft_id
                    )
                    request_route(st_module, "new_study")
                    st_module.rerun()
            else:
                st_module.caption(
                    "clone unavailable — originating draft not found"
                )
        with columns[3]:
            new_name = st_module.text_input(
                "display name",
                value=run.display_name,
                key=f"{_RES}hist_name_{run.search_id[:8]}",
                label_visibility="collapsed",
            )
            run_store = Path(run.store_root) if run.store_root else store_root
            if new_name != run.display_name and st_module.button(
                "Rename", key=f"{_RES}hist_rn_{run.search_id[:8]}"
            ):
                append_catalog_event(
                    run_store,
                    kind="display_name",
                    artifact_id=run.search_id,
                    payload={"display_name": new_name},
                )
                st_module.rerun()
            # UI-2: the catalog archive FLAG (a mutable annotation; never a delete)
            flag_label = "Restore run" if run.archived else "Archive run (catalog flag)"
            if st_module.button(
                flag_label,
                key=f"{_RES}hist_arch_{run.search_id[:8]}",
                help="A mutable catalog annotation: the immutable run is never deleted.",
            ):
                append_catalog_event(
                    run_store,
                    kind="archive",
                    artifact_id=run.search_id,
                    payload=not run.archived,
                )
                st_module.rerun()
        if immutable:
            st_module.caption(
                "Frozen research evidence: no delete or overwrite control "
                "exists; display names, notes and the archive flag are mutable "
                "annotations only."
            )

    st_module.subheader("Frozen / Running Studies")
    if not running:
        st_module.caption("None.")
    for run in running:
        _run_row(run, immutable=True)

    st_module.subheader("Completed Studies")
    if not completed:
        st_module.caption("None.")
    for run in completed:
        _run_row(run, immutable=True)

    st_module.subheader("Superseded Studies")
    if not superseded:
        st_module.caption("None (catalog archive flag).")
    for run in superseded:
        status_badge(st_module, StudyStatusKey.SUPERSEDED)
        _run_row(run, immutable=True)

    st_module.subheader("Legacy Read-Only Results")
    st_module.caption(
        "Legacy context-lane experiments remain in the Context Research "
        "panel's read-only section with their original reports and "
        "mandatory caveats. No rerun, modification, or promotion control "
        "exists for them anywhere in this workspace."
    )
