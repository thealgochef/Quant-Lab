"""Active Runs monitor for the study workspace (R4; FUX §16).

``render_active_runs`` polls the orchestrator's atomic ``search_state.json``
checkpoints through a five-second ``st.fragment`` wrapped around the plain,
AppTest-callable ``_render_monitor_body`` — with an always-present manual
Refresh fallback. It renders the parent phase checklist, the five
keyboard-operable funnel buttons (a Plotly funnel accompanies them, never
replaces them), the exact child table with pagination and skipped-stage
explanations, sanitized row detail with identities and actions, the
confirmed safe-cancel sentinel, and the CLI escape hatch when a status file
is missing. Reading status never loads child artifacts (FUX §33).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_results_charts import build_funnel_figure
from ifvg_ui_common import (
    STATE_PREFIX,
    cli_escape_hatch,
    dev_only_badge,
    identity_block,
    paginate_controls,
    queue_replay_drilldown,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    status_badge,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    SEARCH_PHASES,
    request_safe_cancel,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    ACTIVE_RUNS_COLUMNS,
    FUNNEL_STAGE_LABELS,
    child_row_presentation,
    children_for_stage,
    funnel_counts,
    human_config_name,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    list_search_runs,
    load_charter,
    load_frontier_for_state,
    load_search_state,
)

__all__ = ["render_active_runs", "_render_monitor_body"]

_MON = f"{STATE_PREFIX}mon_"
_SEARCH_KEY = f"{STATE_PREFIX}monitor_search_id"

#: Parent phases shown in the checklist, in orchestrator order (the failed/
#: cancelled terminals render as their own status line).
_CHECKLIST_PHASES = tuple(
    phase for phase in SEARCH_PHASES if phase not in ("failed", "cancelled")
)


def _known_store_roots(roots: Mapping[str, Any]) -> list[Path]:
    known: list[Path] = [Path(roots["store_root"])]
    for root in dict(roots.get("store_roots") or {}).values():
        if Path(root) not in known:
            known.append(Path(root))
    return known


def render_active_runs(st_module=st, *, roots: Mapping[str, Any]) -> None:
    dev_only_badge(st_module)
    runs = list_search_runs(
        Path(roots["state_root"]),
        Path(roots["store_root"]),
        store_roots=_known_store_roots(roots),
    )
    if not runs:
        # UI-1 (plan §6.7): an empty job root is the NO_RUNS state — nothing
        # is missing or corrupt
        render_empty_state(
            st_module,
            "no_runs",
            detail="no search jobs exist in the job root yet",
        )
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_search_job.py status --search-id <search_id>",
            reason="inspect a job outside the workspace",
        )
        return
    labels = {f"{run.display_name} · {run.search_id[:12]}…": run for run in runs}
    preselected = st_module.session_state.get(_SEARCH_KEY)
    options = list(labels)
    index = next(
        (
            position
            for position, run in enumerate(labels.values())
            if run.search_id == preselected
        ),
        0,
    )
    sanitize_select(st_module, f"{_MON}run", options)
    chosen = st_module.selectbox(
        "Active search", options, index=index, key=f"{_MON}run"
    )
    selected_run = labels[chosen]
    search_id = selected_run.search_id
    # UI-1: the run's OWN store (exact charter location) is the read root and
    # the artifact decides the scope badge — never a session selector
    run_roots = {
        **roots,
        "store_root": Path(selected_run.store_root)
        if selected_run.store_root
        else Path(roots["store_root"]),
    }
    if selected_run.verification_only:
        verification_badge(st_module)
    st_module.caption(
        f"Artifact scope: store namespace class `{selected_run.namespace_class or 'unmarked'}`"
        + (
            " · verification-only artifact"
            if selected_run.verification_only
            else " · owner-authorized development artifact"
            if selected_run.verification_only is False
            else " · scope unresolved (charter not located in a known store)"
        )
    )
    st_module.button("Refresh", key=f"{_MON}refresh", help="Manual poll fallback")
    fragment = getattr(st_module, "fragment", None)
    if callable(fragment):

        @fragment(run_every="5s")
        def _auto_body() -> None:
            _render_monitor_body(st_module, roots=run_roots, search_id=search_id)

        _auto_body()
    else:
        _render_monitor_body(st_module, roots=run_roots, search_id=search_id)


def _render_monitor_body(
    st_module, *, roots: Mapping[str, Any], search_id: str
) -> None:
    """Plain body — reads atomic status summaries only, never artifacts."""

    state_root = Path(roots["state_root"])
    store_root = Path(roots["store_root"])
    state = load_search_state(state_root, search_id)
    identity_block(st_module, "Search charter id", search_id)
    if state is None:
        render_empty_state(
            st_module,
            "artifact_missing",
            detail="the job status file is missing or unreadable",
        )
        cli_escape_hatch(
            st_module,
            f"python scripts/ifvg_search_job.py status --search-id {search_id}",
            reason="the status file could not be read",
        )
        return
    phase = str(state.get("phase") or "unknown")
    children = list(state.get("children") or ())
    frontier_envelope = load_frontier_for_state(store_root, state)
    feasible_ids: tuple[str, ...] = ()
    frontier_ids: tuple[str, ...] = ()
    representative: str | None = None
    if frontier_envelope is not None:
        frontier = frontier_envelope.payload.frontier
        feasible_ids = tuple(frontier.feasible_ids)
        frontier_ids = tuple(frontier.frontier_ids)
        representative = frontier.development_exploratory_representative_id

    st_module.markdown("**Phase checklist**")
    reached = (
        _CHECKLIST_PHASES.index(phase) if phase in _CHECKLIST_PHASES else -1
    )
    checklist = []
    for position, name in enumerate(_CHECKLIST_PHASES):
        glyph = "✓" if position <= reached else "·"
        checklist.append(f"{glyph} {name}")
    st_module.code("\n".join(checklist), language="text")
    if phase in ("failed", "cancelled"):
        st_module.error(f"Run terminal state: {phase}")
    notes = state.get("phase_notes") or {}
    for note_key, note in sorted(notes.items()):
        if note_key != "frontier_id":
            st_module.caption(f"{note_key}: {note}")

    counts = funnel_counts(
        children,
        feasible_ids=feasible_ids if frontier_envelope is not None else None,
        frontier_ids=frontier_ids if frontier_envelope is not None else None,
    )
    st_module.markdown("**Parent progress** (FUX §16.2)")
    st_module.table(
        {
            "Profiles Generated": [str(counts.generated)],
            "Replays Completed": [str(counts.replay_valid)],
            "Strategy-Gate Passes": [str(counts.strategy_pass)],
            "Prop-Feasible Configs": [
                str(counts.prop_feasible)
                if counts.prop_feasible is not None
                else "–"
            ],
            "Robust Finalists": [
                str(counts.robust) if counts.robust is not None else "–"
            ],
        }
    )
    st_module.markdown("**Funnel** (buttons filter the child table)")
    stage_key = f"{_MON}stage"
    button_columns = st_module.columns(len(FUNNEL_STAGE_LABELS))
    for column, label, button_label in zip(
        button_columns, FUNNEL_STAGE_LABELS, counts.as_button_labels(), strict=True
    ):
        with column:
            if st_module.button(button_label, key=f"{_MON}funnel_{label}"):
                st_module.session_state[stage_key] = label
    selected_stage = st_module.session_state.get(stage_key, "Generated")
    st_module.caption(f"Filter: {selected_stage}")
    st_module.plotly_chart(
        build_funnel_figure(
            FUNNEL_STAGE_LABELS,
            (
                counts.generated,
                counts.replay_valid,
                counts.strategy_pass,
                counts.prop_feasible,
                counts.robust,
            ),
        ),
        width="stretch",
        key=f"{_MON}funnel_fig",
    )

    filtered = children_for_stage(
        children,
        selected_stage,
        feasible_ids=feasible_ids,
        frontier_ids=frontier_ids,
    )
    charter = load_charter(store_root, search_id)
    prop_wired = bool(
        charter is not None and charter.payload.authorized_firm_contract_ids
    )
    rows = [
        child_row_presentation(
            child,
            feasible_ids=feasible_ids,
            frontier_ids=frontier_ids,
            representative_id=representative,
            prop_wired=prop_wired,
            config_name=human_config_name(child.get("axis_value_ids") or {}),
        )
        for child in filtered
    ]
    st_module.markdown("**Children**")
    if not rows:
        st_module.caption("No children in this stage.")
        return
    start, end = paginate_controls(st_module, len(rows), key=f"{_MON}children")
    page_rows = rows[start:end]
    st_module.dataframe(
        [row.as_row() for row in page_rows],
        width="stretch",
        hide_index=True,
        column_order=list(ACTIVE_RUNS_COLUMNS),
    )

    detail_labels = {
        f"#{row.ordinal} · {row.config}": row for row in page_rows
    }
    sanitize_select(st_module, f"{_MON}detail", list(detail_labels))
    detail_choice = st_module.selectbox(
        "Child detail", list(detail_labels), key=f"{_MON}detail"
    )
    row = detail_labels[detail_choice]
    status_badge(st_module, row.status_key)
    st_module.write(f"Configuration: **{row.config}**")
    if row.axis_value_ids:
        st_module.table(
            {
                "changed parameter": list(row.axis_value_ids.keys()),
                "registered value id": list(row.axis_value_ids.values()),
            }
        )
    identity_block(st_module, "core_replay_id", row.core_replay_id)
    # FUX §16.5 / CS §12: the membership + costed-evaluation linkage — both
    # ids are deterministic functions of the state row + charter, so the
    # detail shows them even before the artifacts are opened.
    try:
        from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
            SearchChildMembership,
        )
        from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: PLC0415
            SearchChildMembershipEnvelope,
        )

        if row.comparison_role in ("baseline", "challenger", "neighbor"):
            membership = SearchChildMembershipEnvelope.from_payload(
                SearchChildMembership(
                    parent_search_id=search_id,
                    child_ordinal=row.ordinal,
                    axis_value_ids=dict(row.axis_value_ids),
                    core_replay_id=row.core_replay_id,
                    comparison_role=row.comparison_role,  # type: ignore[arg-type]
                )
            )
            identity_block(st_module, "membership_id", membership.membership_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.caption(f"membership id unavailable: {sanitize_error(error)}")
    if charter is not None and row.core_replay_id:
        from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
            costed_evaluation_id_for,
        )

        identity_block(
            st_module,
            "costed_evaluation_id (core replay × cost policy)",
            costed_evaluation_id_for(
                row.core_replay_id, charter.payload.cost_policy
            ),
        )
    st_module.caption(f"Comparison role: {row.comparison_role or '—'}")
    if row.human_explanation:
        st_module.write(f"Explanation: {sanitize_error(row.human_explanation)}")
    child_state = next(
        (
            child
            for child in filtered
            if child.get("core_replay_id") == row.core_replay_id
        ),
        {},
    )
    st_module.caption(
        "Attempt history: "
        f"{int(child_state.get('replay_invocations') or 0)} replay "
        f"invocation(s); state {child_state.get('state', '—')}"
    )
    action_columns = st_module.columns([1, 1, 2])
    with action_columns[0]:
        if st_module.button("Open in Results", key=f"{_MON}open_results"):
            from ifvg_study_tab import request_route  # noqa: PLC0415

            st_module.session_state[f"{STATE_PREFIX}results_search_id"] = search_id
            st_module.session_state[f"{STATE_PREFIX}results_child_id"] = (
                row.core_replay_id
            )
            request_route(st_module, "results")
            st_module.rerun()
    with action_columns[1], st_module.expander("Open Replay (exact id)"):
        kind = st_module.selectbox(
            "Id kind",
            ("candidate_id", "decision_id", "trade_id", "setup_id"),
            key=f"{_MON}jump_kind",
        )
        value = st_module.text_input(
            "Exact identifier", key=f"{_MON}jump_value"
        )
        if st_module.button("Queue exact jump", key=f"{_MON}jump_btn"):
            if value.strip():
                queue_replay_drilldown(st_module, kind, value.strip())
            else:
                st_module.warning("An exact identifier is required.")
    with action_columns[2]:
        _safe_cancel_controls(st_module, state_root, search_id, phase)


def _safe_cancel_controls(
    st_module, state_root: Path, search_id: str, phase: str
) -> None:
    """Confirmed safe-cancel: sentinel honored at child boundaries only."""

    if phase in ("search_complete", "failed", "cancelled"):
        st_module.caption(
            "Run is terminal; completed children remain immutable and "
            "reusable."
        )
        return
    confirm = st_module.checkbox(
        "I understand the run stops at the NEXT safe child boundary and "
        "completed children stay immutable/reusable",
        key=f"{_MON}cancel_confirm",
    )
    if st_module.button(
        "Request Safe Cancel", key=f"{_MON}cancel", disabled=not confirm
    ):
        try:
            request_safe_cancel(state_root, search_id)
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.error(f"Cancel request failed: {sanitize_error(error)}")
            return
        st_module.success(
            "Safe-cancel sentinel written; the orchestrator honors it at "
            "the next child boundary."
        )
