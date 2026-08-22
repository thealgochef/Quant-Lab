"""Comparison ribbon, deterministic insights, and the account timeline (R4).

Implements FUX §§25–28: the dimension-diff ribbon rendered BEFORE metrics,
the four comparison panels (Parameter Diff always; Funnel/Strategy/Prop
deltas only where the compatibility contract permits them), the seven-
category deterministic insight panel with exact ``EvidenceRef`` actions, and
the account timeline consuming totally ordered ``PropAccountEventEnvelope``
streams. Cross-profile population deltas surface ONLY through
``study_providers.prepare_cross_profile_deltas`` (which persists both sides'
lineage-uniqueness reports first — the R2→R4 obligation); populations
without an exact basis render ``not_comparable`` and never fuzz.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_results_charts import (
    build_account_timeline_figure,
    build_funnel_delta_figure,
)
from ifvg_ui_common import (
    STATE_PREFIX,
    display_metric,
    identity_block,
    queue_replay_drilldown,
    render_empty_state,
    result_scope_caption,
    sanitize_error,
    sanitize_select,
)

from alpha_lab.agents.data_infra.ifvg.search.insights import (
    InsightCategory,
    InsightPanel,
)
from alpha_lab.agents.data_infra.ifvg.search.insights import (
    render_insight_panel as build_insight_panel,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    changed_axis_labels,
    computation_path_chip,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    list_catalogued_envelope_ids,
    load_account_simulation_events,
)
from alpha_lab.agents.data_infra.ifvg.study_status import ResultScope

__all__ = [
    "render_comparison",
    "render_insight_panel",
    "render_insights_for_bundle",
    "render_account_timeline",
]

_CMP = f"{STATE_PREFIX}cmp_"

_JUMPABLE_KINDS = {
    "setup": "setup_id",
    "trade": "trade_id",
    "candidate": "candidate_id",
    "decision": "decision_id",
}


def _pick_child(
    st_module, bundle: Mapping[str, Any], *, label: str, key: str
) -> Mapping[str, Any] | None:
    children = bundle["children"]
    names = bundle["names"]
    options = {
        f"{names.get(str(child.get('core_replay_id')), '?')} · "
        f"{str(child.get('core_replay_id'))[:12]}…": child
        for child in children
        if child.get("core_replay_id")
    }
    if not options:
        return None
    sanitize_select(st_module, key, list(options))
    chosen = st_module.selectbox(label, list(options), key=key)
    return options[chosen]


def render_comparison(
    st_module=st,
    *,
    roots: Mapping[str, Any],
    bundle: Mapping[str, Any],
    population_reports: Mapping[str, Any] | None = None,
    funnel_delta: Mapping[str, Any] | None = None,
) -> None:
    """Baseline/arbitrary comparison (FUX §25).

    ``population_reports`` / ``funnel_delta`` are injected by callers that
    already ran the lineage-gated delta pipeline
    (``prepare_cross_profile_deltas``); when absent, the membership panels
    render their honest unavailable/not-comparable states — never an
    approximation.
    """

    st_module.subheader("Comparison")
    baseline = _pick_child(
        st_module, bundle, label="Reference configuration", key=f"{_CMP}base"
    )
    challenger = _pick_child(
        st_module, bundle, label="Compared configuration", key=f"{_CMP}chal"
    )
    if baseline is None or challenger is None:
        st_module.caption("Two configurations are required.")
        return
    base_id = str(baseline.get("core_replay_id"))
    chal_id = str(challenger.get("core_replay_id"))
    base_values = dict(baseline.get("axis_value_ids") or {})
    chal_values = dict(challenger.get("axis_value_ids") or {})
    changed = sorted(
        key
        for key in set(base_values) | set(chal_values)
        if base_values.get(key) != chal_values.get(key)
    )
    frozen = sorted(
        key for key in set(base_values) & set(chal_values) if key not in changed
    )
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: PLC0415
        SEARCH_AXIS_REGISTRY_V1,
    )
    from alpha_lab.agents.data_infra.ifvg.search.lineage import (  # noqa: PLC0415
        derive_lineage_validity,
    )
    from alpha_lab.agents.data_infra.ifvg.study.contrasts import (  # noqa: PLC0415
        CONTRAST_BOOTSTRAP_PROTOCOL_ID,
    )

    lineage_valid, lineage_reason = derive_lineage_validity(tuple(changed))
    same_profile = base_id == chal_id or not changed
    if population_reports:
        bases = {
            str(getattr(report, "match_basis", "not_comparable"))
            for report in population_reports.values()
        }
        match_basis = ", ".join(sorted(bases))
    elif same_profile:
        match_basis = "native_id_exact"
    elif lineage_valid:
        match_basis = "profile_independent_lineage_exact (pending delta build)"
    else:
        match_basis = "not_comparable"
    replay_chip = (
        "Requires New Sequential Replay"
        if any(
            SEARCH_AXIS_REGISTRY_V1[key].requires_full_sequential_replay
            for key in changed
            if key in SEARCH_AXIS_REGISTRY_V1
        )
        else "Analysis Filter"
    )
    st_module.markdown("**Dimension-diff ribbon**")
    st_module.table(
        {
            "delta type": ["config_diff" if changed else "identical semantics"],
            "changed dimensions": [", ".join(changed) or "none"],
            "frozen dimensions": [
                ", ".join(frozen[:6]) + ("; …" if len(frozen) > 6 else "")
                or "all shared"
            ],
            "computation path": [replay_chip],
            "compatibility": [
                "comparable" if lineage_valid or same_profile else "config diff only"
            ],
            "cohort identity": ["full population (no cohort filter)"],
            "uncertainty method": [CONTRAST_BOOTSTRAP_PROTOCOL_ID],
            "development status": ["Development Exploratory Representative lane"],
            "match_basis": [match_basis],
        }
    )
    if not lineage_valid and not same_profile:
        st_module.warning(
            "Populations are not comparable: "
            + (lineage_reason or "no exact lineage basis")
            + " — only the parameter diff is shown; membership claims are "
            "disabled, never approximated."
        )

    panels = st_module.tabs(
        ["Parameter Diff", "Funnel Delta", "Strategy Delta", "Prop Delta"]
    )
    with panels[0]:
        rows = []
        for key in sorted(set(base_values) | set(chal_values)):
            spec = SEARCH_AXIS_REGISTRY_V1.get(key)
            rows.append(
                {
                    "axis": key,
                    "reference": base_values.get(key, "—"),
                    "compared": chal_values.get(key, "—"),
                    "changed": "yes" if key in changed else "no",
                    "computation path": computation_path_chip(spec)
                    if spec
                    else "—",
                }
            )
        if rows:
            st_module.dataframe(rows, width="stretch", hide_index=True)
        else:
            st_module.caption("Both configurations are the exact baseline.")
    with panels[1]:
        if funnel_delta is not None:
            totals = funnel_delta.get("totals") or ()
            st_module.plotly_chart(
                build_funnel_delta_figure(list(totals)),
                width="stretch",
                key=f"{_CMP}fd_fig",
            )
            for row in funnel_delta.get("affected", ()):  # exact IDs (FUX §25)
                _evidence_action(
                    st_module, row["kind"], row["identifier"], key_hint="fd"
                )
        elif not lineage_valid and not same_profile:
            render_empty_state(st_module, "lineage_not_comparable")
        else:
            render_empty_state(
                st_module,
                "artifact_unavailable",
                detail=(
                    "funnel tables for this pair are not loaded; deltas "
                    "surface once the lineage-gated delta build runs "
                    "(prepare_cross_profile_deltas)"
                ),
            )
    # §25 rule 1 / DT §3.1: an incompatible (config-diff-only) pair renders
    # the parameter diff ONLY — the metric-delta panels are suppressed with
    # the explicit notice, never populated (adversarial F4c).
    comparable = lineage_valid or same_profile
    suppressed_notice = (
        "Suppressed: this pair is config-diff-only (no exact comparison "
        "basis) — metric deltas are disabled, never approximated."
    )
    with panels[2]:
        if not comparable:
            st_module.warning(suppressed_notice)
            base_metrics = chal_metrics = None
        else:
            result_scope_caption(
                st_module, ResultScope.ACTUAL_EXECUTED_STRATEGY
            )
            base_metrics = bundle["metrics_by_child"].get(base_id)
            chal_metrics = bundle["metrics_by_child"].get(chal_id)
        if not comparable:
            pass
        elif base_metrics is None or chal_metrics is None:
            render_empty_state(
                st_module,
                "artifact_unavailable",
                detail="a costed evaluation is missing for one side",
            )
        else:
            rows = []
            for attribute in (
                "executed_trades",
                "gross_expectancy_r",
                "net_expectancy_r",
                "profit_factor",
                "max_drawdown_r",
                "time_under_water_days",
                "top_day_pnl_share",
                "top_setup_pnl_share",
            ):
                left = getattr(base_metrics, attribute, None)
                right = getattr(chal_metrics, attribute, None)
                delta = (
                    right - left
                    if isinstance(left, (int, float))
                    and isinstance(right, (int, float))
                    else None
                )
                rows.append(
                    {
                        "metric": attribute,
                        "basis": "gross"
                        if attribute.startswith("gross")
                        else "net",
                        "reference": display_metric(left),
                        "compared": display_metric(right),
                        "delta": display_metric(delta),
                    }
                )
            st_module.dataframe(rows, width="stretch", hide_index=True)
    with panels[3]:
        vectors = bundle["prop_vectors"]
        if not comparable:
            st_module.warning(suppressed_notice)
            base_vec: dict = {}
            chal_vec: dict = {}
        else:
            modes = {
                str(summary.get("simulation_mode", ""))
                for summaries in vectors.values()
                for summary in summaries.values()
            }
            scope = ResultScope.BOOTSTRAP_SIMULATION
            if modes == {"historical_closed_trade"}:
                scope = ResultScope.PROP_HISTORICAL_CLOSED_TRADE
            elif modes == {"historical_1m_scenario"}:
                scope = ResultScope.PROP_1M_SCENARIO
            result_scope_caption(st_module, scope)
            base_vec = vectors.get(base_id, {})
            chal_vec = vectors.get(chal_id, {})
        if not comparable:
            pass
        elif not base_vec or not chal_vec:
            render_empty_state(
                st_module,
                "artifact_unavailable",
                detail=(
                    "persisted account simulations are missing for at least "
                    "one side (the R5 pipeline persists them)"
                ),
            )
        else:
            rows = []
            firms = sorted(set(base_vec) | set(chal_vec))
            for firm in firms:
                left = (
                    base_vec.get(firm, {})
                    .get("payout_reliability_vector", {})
                    .get("expected_net_payout_90d")
                )
                right = (
                    chal_vec.get(firm, {})
                    .get("payout_reliability_vector", {})
                    .get("expected_net_payout_90d")
                )
                rows.append(
                    {
                        "firm/policy": firm,
                        "reference E[payout 90d]": display_metric(left),
                        "compared E[payout 90d]": display_metric(right),
                        "delta": display_metric(
                            right - left
                            if left is not None and right is not None
                            else None
                        ),
                    }
                )
            st_module.dataframe(rows, width="stretch", hide_index=True)

    if population_reports:
        st_module.markdown("**Population membership (exact lineage basis)**")
        for kind, report in sorted(population_reports.items()):
            basis = getattr(report, "match_basis", "not_comparable")
            if basis == "not_comparable":
                st_module.caption(
                    f"{kind}: not comparable — "
                    f"{getattr(report, 'match_basis_reason', '')}"
                )
                continue
            added = tuple(getattr(report, "added_keys", ()))
            removed = tuple(getattr(report, "removed_keys", ()))
            st_module.write(
                f"{kind}: +{len(added)} added · −{len(removed)} removed · "
                f"jaccard {display_metric(getattr(report, 'jaccard', None))} "
                f"· basis {basis}"
            )


def _evidence_action(st_module, kind: str, identifier: str, *, key_hint: str) -> None:
    """One exact evidence action: jumpable kinds queue the verifier jump."""

    jump_kind = _JUMPABLE_KINDS.get(kind)
    columns = st_module.columns([3, 1])
    with columns[0]:
        st_module.code(f"{kind}: {identifier}", language=None)
    with columns[1]:
        if jump_kind is not None:
            if st_module.button(
                "Open exact",
                key=f"{_CMP}ev_{key_hint}_{kind}_{identifier[:12]}",
            ):
                queue_replay_drilldown(st_module, jump_kind, identifier)
        elif kind == "simulation":
            if st_module.button(
                "Open timeline",
                key=f"{_CMP}ev_{key_hint}_{kind}_{identifier[:12]}",
            ):
                st_module.session_state[f"{_CMP}timeline_id"] = identifier
        else:
            st_module.caption("identity above")


def render_insight_panel(st_module, panel: InsightPanel) -> None:
    """The seven fixed categories, verbatim deterministic text (FUX §26)."""

    st_module.subheader("Deterministic insights")
    st_module.caption(
        "Persisted deterministic template renders — causally neutral, "
        "match-basis-aware. An AI-written summary is out of scope and never "
        "replaces this panel."
    )
    for category in InsightCategory:
        insights = panel.by_category(category)
        st_module.markdown(f"**{category.value}**")
        if not insights:
            st_module.caption("no insight in this category for this selection")
            continue
        for position, insight in enumerate(insights):
            if insight.suppressed:
                st_module.caption(
                    f"suppressed: {insight.suppression_reason or 'not comparable'}"
                )
                continue
            st_module.write(insight.text)
            if insight.match_basis:
                st_module.caption(f"match_basis: {insight.match_basis}")
            for ref in insight.evidence:
                _evidence_action(
                    st_module,
                    ref.kind,
                    ref.identifier,
                    key_hint=f"{category.name[:4]}{position}",
                )


def render_insights_for_bundle(st_module, bundle: Mapping[str, Any]) -> None:
    """Deterministic panel for the selected configuration vs the baseline.

    The text derives ONLY from persisted immutable artifacts (metrics,
    frontier), so every render reproduces the identical wording; the
    pipeline's S14 stage persists these panels into the insights store at
    R5 — until then the derivation source is labeled here.
    """

    frontier = bundle["frontier"]
    metrics_by_child = bundle["metrics_by_child"]
    names = bundle["names"]
    children = bundle["children"]
    selected = st_module.session_state.get(f"{STATE_PREFIX}selected_config")
    baseline_child = next(
        (
            child
            for child in children
            if child.get("comparison_role") == "baseline"
        ),
        None,
    )
    child_id = selected or (
        frontier.development_exploratory_representative_id if frontier else None
    )
    if not child_id or child_id not in metrics_by_child:
        st_module.caption(
            "No persisted metrics for the selected configuration — insights "
            "need the costed evaluation artifact."
        )
        return
    baseline_id = (
        str(baseline_child.get("core_replay_id")) if baseline_child else None
    )
    baseline_metrics = (
        metrics_by_child.get(baseline_id) if baseline_id else None
    )
    child = next(
        (c for c in children if c.get("core_replay_id") == child_id), {}
    )
    try:
        panel = build_insight_panel(
            changed_axis_labels=changed_axis_labels(
                child.get("axis_value_ids") or {}
            ),
            child_id=child_id,
            metrics=metrics_by_child[child_id],
            baseline_metrics=baseline_metrics,
            sample_caveat=(
                "five-day/verification-scale samples support no research "
                "interpretation"
                if len(children) <= 8
                else None
            ),
        )
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Insight build failed: {sanitize_error(error)}")
        return
    st_module.caption(
        f"Derived deterministically from persisted artifacts for "
        f"{names.get(child_id, child_id[:12])}; S14 persistence lands with "
        "the R5 pipeline."
    )
    render_insight_panel(st_module, panel)


def render_account_timeline(st_module=st, *, roots: Mapping[str, Any]) -> None:
    """The FUX §28 account timeline over ordered event envelopes."""

    st_module.subheader("Account timeline")
    store_root = Path(roots["store_root"])
    catalogued = list_catalogued_envelope_ids(store_root, "account_simulations")
    options = {"(enter an exact id)": None} | {
        f"{name} · {sim_id[:12]}…": sim_id for sim_id, name in catalogued
    }
    sanitize_select(st_module, f"{_CMP}timeline_pick", list(options))
    chosen = st_module.selectbox(
        "Account simulation", list(options), key=f"{_CMP}timeline_pick"
    )
    simulation_id = options[chosen]
    typed = st_module.text_input(
        "Exact account_simulation_id (64-hex)",
        value=str(st_module.session_state.get(f"{_CMP}timeline_id", "")),
        key=f"{_CMP}timeline_typed",
        help="Short ids are display-only; only the full identity resolves.",
    ).strip()
    if typed:
        simulation_id = typed
    if not simulation_id:
        st_module.caption(
            "Choose a catalogued simulation or paste an exact identity."
        )
        return
    try:
        loaded = load_account_simulation_events(store_root, simulation_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Timeline unavailable: {sanitize_error(error)}")
        return
    if loaded is None:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="no account simulation exists under that exact identity",
        )
        return
    summary, events = loaded
    mode = str(summary.get("simulation_mode", "—"))
    scope = (
        ResultScope.PROP_1M_SCENARIO
        if "scenario" in mode
        else ResultScope.PROP_HISTORICAL_CLOSED_TRADE
        if mode == "historical_closed_trade"
        else ResultScope.BOOTSTRAP_SIMULATION
    )
    result_scope_caption(st_module, scope)
    if "scenario" in mode:
        st_module.caption(
            "Assumed 1-minute intrabar paths are a scenario/approximation — "
            "never exact historical chronology."
        )
    identity_block(st_module, "account_simulation_id", simulation_id)
    windows = tuple(
        (str(window["start"]), str(window["end"]))
        for window in summary.get("payout_eligibility_windows", ())
    )
    figure, omissions = build_account_timeline_figure(
        events, eligibility_windows=windows
    )
    st_module.plotly_chart(
        figure, width="stretch", key=f"{_CMP}timeline_fig"
    )
    for line in omissions.summary_lines():
        st_module.caption(f"Omission: {line}")
    st_module.markdown(
        "**Event table** (total order by `event_ordinal` — the envelope "
        "order, never chart-timestamp sorting)"
    )
    table = [
        {
            "ordinal": event.get("event_ordinal"),
            "ts (UTC)": event.get("event_ts_utc"),
            "type": event.get("event_type"),
            "phase": event.get("account_phase"),
            "trade": (event.get("source_trade_id") or "—")[:16],
            "setup": (event.get("source_setup_id") or "—")[:16],
            "event_id": event.get("event_id"),
        }
        for event in events
    ]
    st_module.dataframe(table, width="stretch", hide_index=True)
    ordinals = [str(event.get("event_ordinal")) for event in events]
    if not ordinals:
        return
    sanitize_select(st_module, f"{_CMP}timeline_row", ordinals)
    chosen_ordinal = st_module.selectbox(
        "Open linked evidence for event ordinal",
        ordinals,
        key=f"{_CMP}timeline_row",
    )
    event = next(
        event
        for event in events
        if str(event.get("event_ordinal")) == chosen_ordinal
    )
    identity_block(st_module, "event_id", str(event.get("event_id") or ""))
    for kind, source_key in (
        ("trade", "source_trade_id"),
        ("setup", "source_setup_id"),
        ("candidate", "source_candidate_id"),
        ("decision", "source_decision_id"),
    ):
        identifier = event.get(source_key)
        if identifier:
            _evidence_action(
                st_module, kind, str(identifier), key_hint=f"tl{chosen_ordinal}"
            )
    if event.get("source_path_event_id"):
        identity_block(
            st_module,
            "source_path_event_id",
            str(event.get("source_path_event_id")),
        )
