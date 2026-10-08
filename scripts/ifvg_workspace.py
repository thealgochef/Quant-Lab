"""Focused IFVG research navigation. Only the selected surface executes."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from pathlib import Path

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (
    StudySummary,
    configuration_name,
    elapsed,
    human_name,
    load_studies,
    search_activity,
    search_progress,
)
from alpha_lab.agents.data_infra.ifvg.presentation.workspace_mode import technical_details_enabled

_NAV = "ifvg_workspace_destination"
_SCREEN = "ifvg_workspace_screen"
_SELECTED = "ifvg_workspace_selected_study"


def workspace_roots():
    from ifvg_pipeline_tab import PIPELINE_STATE_ROOT
    from ifvg_study_tab import workspace_roots as existing_roots

    roots = existing_roots()
    repo = Path(roots["repo_root"])
    return {
        **roots,
        "pipeline_state_root": PIPELINE_STATE_ROOT,
        "context_catalog": repo / "data/ifvg_experiments/context_v1_catalog.json",
        "context_run_root": repo / "data/ifvg_experiments/context_v1",
    }


def _go(st_module, screen, key=None):
    st_module.session_state[_SCREEN] = screen
    if key is not None:
        st_module.session_state[_SELECTED] = key
    st_module.rerun()


#: screens of the redesigned shell and which rail item they belong to
_NEW_STUDY_SCREENS = ("new", "new_funded", "approve_funded", "context", "research_new")
_REDESIGNED_SCREENS = ("list", "funded", "funded_detail", "new_funded", "approve_funded")


def rail_destination(st_module, clicked):
    """A rail click: My studies → the library; New study → the funded setup."""

    st_module.session_state[_NAV] = clicked
    # a new destination: the old page address (a draft, a result) no longer applies
    with contextlib.suppress(Exception):  # headless tests have no page address
        st_module.query_params.clear()
    if clicked == "My studies":
        st_module.session_state[_SCREEN] = "list"
    elif clicked == "New study":
        from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

        st_module.session_state.pop(SESSION_DRAFT_KEY, None)
        st_module.session_state.pop(f"{STATE_PREFIX}draft_id", None)
        st_module.session_state[_SCREEN] = "new_funded"
    st_module.rerun()


#: the earlier private name, kept for callers written before it was public
_rail_destination = rail_destination


def render_workspace(st_module=st, *, roots=None):
    roots = roots or workspace_roots()
    from ifvg_lab_nav import apply_deep_link, sync_url
    from ifvg_lab_ui import inject_theme, rail

    inject_theme(st_module)
    apply_deep_link(roots, st_module)
    pending_replay = st_module.session_state.pop("ifvg_study_v1_open_review", False)
    if pending_replay:
        st_module.session_state[_NAV] = "Trade review"
    # Existing freeze handlers queue an exact selected run and a route.
    pending = st_module.session_state.pop("ifvg_study_v1_pending_route", None)
    if pending in ("Active Runs", "Results"):
        from ifvg_ui_common import STATE_PREFIX

        key = st_module.session_state.get(f"{STATE_PREFIX}monitor_search_id")
        st_module.session_state[_SELECTED] = key
        st_module.session_state[_SCREEN] = "detail"
        st_module.session_state[_NAV] = "My studies"
    destination = st_module.session_state.get(_NAV, "My studies")
    screen = st_module.session_state.get(_SCREEN, "list")
    if destination != "Trade review":
        # the rail follows the screen (a launch from New study lands on the study page)
        destination = "New study" if screen in _NEW_STUDY_SCREENS else "My studies"
        st_module.session_state[_NAV] = destination
    active = destination
    note = st_module.session_state.pop("ifvg_lab_v1_link_note", None)
    if note:
        from ifvg_lab_ui import show

        from alpha_lab.agents.data_infra.ifvg.presentation.lab.html import note as note_block

        show(note_block(note, "orange"), st_module)
    clicked = rail(active, st_module)
    if clicked:
        rail_destination(st_module, clicked)
    if destination == "Trade review":
        if screen == "ml_phase" and st_module.session_state.get("ifvg_ml_phase_pointer"):
            from ifvg_lab_ml_phase import render_ml_phase
            from ifvg_lab_nav import funded_context

            pointer = st_module.session_state["ifvg_ml_phase_pointer"]
            funded_context(pointer["report_id"], st_module)["ml_view"] = "Decision review"
            st_module.session_state["mlphase_ml_view"] = "Decision review"
            render_ml_phase(st_module, roots)
            sync_url(st_module)
            return
        from ifvg_lab_trade_review import render_trade_review_page

        render_trade_review_page(st_module, roots)
        sync_url(st_module)
        return
    if screen not in _REDESIGNED_SCREENS and st_module.button(
            "← My studies", key="ifvg_workspace_back",
            help="Return to the list of every study."):
        st_module.session_state[_NAV] = "My studies"
        _go(st_module, "list")
    if screen == "ml_phase":
        from ifvg_lab_ml_phase import render_ml_phase

        render_ml_phase(st_module, roots)
    elif screen == "funded":
        from ifvg_lab_funded import render_funded_results

        render_funded_results(st_module, roots)
    elif screen == "funded_detail":
        from ifvg_lab_funded import render_funded_detail

        render_funded_detail(st_module, roots)
    elif screen == "new_funded":
        from ifvg_lab_new_funded import render_funded_setup

        render_funded_setup(st_module, roots)
    elif screen == "approve_funded":
        from ifvg_lab_new_funded import render_funded_review

        render_funded_review(st_module, roots)
    elif screen == "new":
        from ifvg_research_wizard import render_new_study

        render_new_study(st_module, roots=roots)
    elif screen == "context":
        from ifvg_research_context import render_context_study

        render_context_study(st_module)
    elif screen == "research_new":
        from ifvg_research_pipeline import render_research_configuration

        render_research_configuration(st_module, roots=roots)
    elif screen == "research_group":
        from ifvg_research_pipeline import render_research_group

        render_research_group(
            st_module,
            roots=roots,
            group_id=st_module.session_state.get("ifvg_workspace_research_group"),
        )
    else:
        studies, issues = load_studies(roots)
        for issue in issues:
            st_module.warning(issue)
        if screen == "detail":
            study = next(
                (row for row in studies if row.key == st_module.session_state.get(_SELECTED)), None
            )
            if study is None:
                st_module.warning(
                    "This study is unavailable. Return to My studies and refresh the list."
                )
            else:
                render_study(st_module, study, roots)
        else:
            from ifvg_lab_library import render_library

            render_library(st_module, studies, roots)
    sync_url(st_module)


def render_my_studies(st_module, studies, roots):
    from ifvg_rules import render_strategy_rules

    st_module.header("My studies")
    st_module.write("Start a research question, resume a study, or inspect its evidence.")
    if st_module.button("New study", type="primary", key="ifvg_workspace_new"):
        from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

        st_module.session_state.pop(SESSION_DRAFT_KEY, None)
        st_module.session_state.pop(f"{STATE_PREFIX}draft_id", None)
        _go(st_module, "new")
    from ifvg_research_pipeline import render_research_group_cards

    render_research_group_cards(st_module, roots)
    with st_module.expander("Filter studies"):
        name = st_module.text_input("Find a study", key="ifvg_workspace_filter")
        status = st_module.multiselect(
            "Status", sorted({row.status for row in studies}), key="ifvg_workspace_status"
        )
        archived = st_module.checkbox("Show archived studies", key="ifvg_workspace_archived")
    visible = [
        row
        for row in studies
        if row.archived == archived
        and name.lower() in row.name.lower()
        and (not status or row.status in status)
    ]
    if not visible:
        st_module.info(
            "No studies match these filters."
            if studies
            else "Your studies will appear here. Start with Evaluate, Compare, or Search."
        )
        return
    page_count = (len(visible) + 9) // 10
    page = (
        int(
            st_module.number_input(
                "Page",
                min_value=1,
                max_value=max(page_count, 1),
                value=1,
                key="ifvg_workspace_page",
            )
        )
        if page_count > 1
        else 1
    )
    for row in visible[(page - 1) * 10 : page * 10]:
        with st_module.container(border=True):
            st_module.subheader(row.name)
            st_module.write(row.question)
            render_strategy_rules(st_module, row, roots, compact=True)
            st_module.caption(
                f"{row.dates} · {row.status}"
                + (" · Scope unresolved" if row.scope == "unresolved" else "")
            )
            if st_module.button(
                "Inspect archived study" if row.archived else row.next_action,
                key=f"ifvg_open_{row.kind}_{row.key}",
            ):
                if (
                    row.kind == "draft"
                    and row.status == "Draft"
                    and not row.archived
                    and row.scope != "unresolved"
                ):
                    _open_draft(st_module, row.draft)
                _go(st_module, "detail", row.key)
            if (
                row.kind == "draft"
                and row.status == "Draft"
                and not row.archived
                and row.scope != "unresolved"
                and st_module.button("Details and actions", key=f"ifvg_details_{row.key}")
            ):
                _go(st_module, "detail", row.key)


def _open_draft(st_module, draft):
    from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

    st_module.session_state.pop(SESSION_DRAFT_KEY, None)
    st_module.session_state[f"{STATE_PREFIX}draft_id"] = draft.draft_id
    _go(st_module, "new")


#: funded comparison runs whose saved result the redesigned screens can show
_FUNDED_RESULT_STATUSES = ("Completed", "Incomplete", "Failed")


def funded_result_target(study: StudySummary, roots, *, app: str | None = None):
    """The saved result of a funded comparison run, or None while it has none."""

    from ifvg_lab_nav import current_app

    state = study.state or {}
    result_id = state.get("result_id")
    if study.kind != "funded_comparison" or not result_id \
            or study.status not in _FUNDED_RESULT_STATUSES:
        return None
    target = {"study_key": study.key, "result_id": str(result_id),
            "plan_id": str(state.get("plan_id") or study.key),
            "store_root": str(study.store_root or roots["store_root"]), "name": study.name,
            "status": study.status, "dates": study.dates,
            "app": app or current_app(roots)}
    for key in ("external_review_only", "catalog_binding", "versions", "qualification"):
        if key in state:
            target[key] = state[key]
    return target


def render_study(st_module, study: StudySummary, roots):
    from ifvg_rules import render_strategy_rules

    st_module.header(study.name)
    st_module.write(study.question)
    st_module.caption(f"{study.dates} · {study.status}")
    render_strategy_rules(st_module, study, roots)
    if study.scope == "unresolved":
        st_module.warning(
            "Scope unresolved. The saved evidence does not establish whether this "
            "is a research study. Its results cannot support a research conclusion."
        )
        if (
            study.kind == "draft"
            and study.draft.never_frozen
            and st_module.button("Continue and resolve scope")
        ):
            _open_draft(st_module, study.draft)
    elif study.kind == "funded":
        from ifvg_funded_study import render_funded_study

        render_funded_study(st_module, study, roots)
        return
    elif study.kind == "funded_comparison":
        target = funded_result_target(study, roots)
        if target is not None:
            # a saved result opens on the redesigned funded results screen
            from ifvg_lab_nav import open_funded_results

            open_funded_results(target, st_module)
            return
        from ifvg_funded_comparison_study import render_comparison_study

        render_comparison_study(st_module, study, roots)
        return
    elif study.kind == "context":
        from ifvg_research_context import render_context_study

        render_context_study(st_module, run_id=study.key)
    elif study.kind == "draft":
        if study.archived:
            st_module.info("Restore this study or clone its saved settings to continue research.")
        elif study.status == "Draft":
            if st_module.button("Continue", type="primary"):
                _open_draft(st_module, study.draft)
        elif study.status == "Ready" and not study.archived:
            from alpha_lab.agents.data_infra.ifvg.study_providers import locate_charter_store

            store = locate_charter_store(
                study.charter_id, [Path(p) for p in roots["store_roots"].values()]
            )
            if store:
                _run_action(st_module, replace(study, store_root=store), roots)
            else:
                st_module.warning(
                    "The saved configuration could not be verified. Running is "
                    "unavailable until its evidence is restored."
                )
        else:
            st_module.warning(
                "Workflow progress was not recorded. Clone this study to configure "
                "and launch a new workflow, or restore its saved progress."
            )
    elif study.status == "Completed":
        if study.kind == "search":
            from ifvg_research_results import render_search_results

            render_search_results(st_module, study, roots)
        else:
            from ifvg_research_pipeline import render_pipeline_results

            render_pipeline_results(st_module, study, roots)
    elif study.status == "Running":
        _live_progress(st_module, study, roots)
    else:
        _progress(st_module, study, roots)
    _manage_study(st_module, study, roots)


@st.fragment(run_every="5s")
def _live_progress(st_module, study, roots):
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (
        pipeline_status,
        search_status,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import read_pipeline_state
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_search_state

    try:
        if study.kind == "pipeline":
            state = read_pipeline_state(Path(roots["pipeline_state_root"]), study.key)
            status = pipeline_status(state)
        else:
            state = load_search_state(Path(roots["state_root"]), study.key)
            status = search_status(state)
    except Exception:
        state, status = None, "Evidence unavailable"
    if status != study.status:
        st_module.rerun()
    _progress(st_module, replace(study, state=state, status=status), roots)


def _progress(st_module, study, roots):
    state = study.state or {}
    stop_requested = False
    if study.status == "Running" and study.kind != "pipeline":
        from alpha_lab.agents.data_infra.ifvg.search.orchestrator import cancellation_requested

        stop_requested = cancellation_requested(Path(roots["state_root"]), study.key)
    if study.status == "Running":
        activity = search_activity(state)
        if study.kind == "pipeline":
            from ifvg_research_pipeline import activity_label

            activity = activity_label(state.get("current_stage"))
        st_module.subheader(activity)
        if study.kind != "pipeline" and state.get("phase") == "charter_frozen":
            notes = state.get("phase_notes") or {}
            try:
                checked = int(notes["preparation_completed_configurations"])
                planned = int(notes["preparation_planned_configurations"])
            except (KeyError, ValueError, TypeError):
                checked, planned = -1, 0
            if 0 <= checked <= planned and planned > 0:
                st_module.caption(
                    f"Input checks: {checked} of {planned} configuration input sets verified."
                )
    elif study.status in ("Failed", "Interrupted", "Blocked"):
        st_module.warning(
            {
                "Failed": (
                    "The study stopped after an error. Its partial results do not "
                    "establish a completed outcome."
                ),
                "Interrupted": (
                    "The study is paused. Saved progress has been retained. "
                    "Work continues only when you choose Resume study."
                ),
                "Blocked": (
                    "A required part of the study is unavailable. The study cannot yet "
                    "answer its research question."
                ),
            }[study.status]
        )
    elif study.status != "Ready":
        st_module.warning(
            "Progress is unavailable or unresolved. Refresh after the evidence has been restored."
        )
    if study.kind == "pipeline":
        rows = [row for row in (state.get("stages") or {}).values() if row.get("in_plan")]
        completed = sum(row.get("status") in ("completed", "reused") for row in rows)
    else:
        rows = list(state.get("children") or ())
        counts = search_progress(state)
        completed = counts.completed + counts.reused
    if rows:
        progress_text = (
            f"{completed} of {len(rows)} work items complete"
            if study.kind == "pipeline"
            else f"{counts.completed} completed · {counts.reused} reused "
            f"· {counts.total} configurations total"
        )
        st_module.progress(completed / len(rows), text=progress_text)
        if study.kind != "pipeline":
            st_module.caption(
                f"{counts.attempted} of {counts.total} configurations attempted "
                "(including saved results and failures)."
            )
            st_module.caption(
                f"Failed: {counts.failed} · Running: {counts.running} · Queued: {counts.queued} "
                f"· Stopped: {counts.stopped} · Blocked: {counts.blocked}"
                + (f" · Unresolved: {counts.unresolved}" if counts.unresolved else "")
            )
            if counts.failed:
                st_module.warning(
                    f"{counts.failed} configurations failed. These are not completed results."
                )
            if study.status == "Running" and not (counts.running or counts.queued):
                st_module.caption(
                    "No configurations are currently queued or marked running. "
                    "The saved study phase is still in progress."
                )
    elif study.kind != "pipeline" and state.get("phase") == "charter_frozen":
        st_module.caption(
            "Configuration counts become available after input verification. "
            "No replay completion has been recorded yet."
        )
    timer_label = "Current attempt elapsed time" if state.get("attempt_started_at_utc") else (
        "Elapsed time"
    )
    st_module.caption(f"{timer_label}: {elapsed(state)}")
    if st_module.button("Refresh progress", key="ifvg_workspace_refresh"):
        st_module.rerun()
    if study.status == "Running":
        if stop_requested:
            st_module.info(
                "Stop requested. Current work will finish at a safe boundary. "
                "Completed results will be kept."
            )
        if st_module.button(
            "Stop after current work", key="ifvg_workspace_cancel", disabled=stop_requested,
            help="Finish the current work at a safe boundary and save progress for a later resume.",
        ):
            try:
                if study.kind == "pipeline":
                    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
                        request_pipeline_cancel,
                    )

                    request_pipeline_cancel(Path(roots["pipeline_state_root"]), study.key)
                else:
                    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
                        request_safe_cancel,
                    )

                    request_safe_cancel(Path(roots["state_root"]), study.key)
                st_module.info("Stop requested. Current work will reach a safe stopping point.")
            except Exception:
                st_module.error("The stop request could not be saved. Refresh and try again.")
    elif study.status in ("Ready", "Failed", "Interrupted", "Blocked") and not study.archived:
        _run_action(st_module, study, roots)
    failures = [
        row.get("failure_reason") for row in state.get("children", ()) if row.get("failure_reason")
    ]
    if failures:
        from alpha_lab.agents.data_infra.ifvg.presentation.workspace import failure_explanation

        for reason in dict.fromkeys(failures):
            st_module.info(failure_explanation(reason))
    if rows and st_module.checkbox("Show work details", key="ifvg_workspace_work_details"):
        from ifvg_research_pipeline import activity_label

        if study.kind == "pipeline":
            table = [
                {
                    "Activity": activity_label(key),
                    "Status": str(row.get("status", "Unknown")).replace("_", " ").capitalize(),
                }
                for key, row in (state.get("stages") or {}).items()
                if row.get("in_plan")
            ]
        else:
            table = [
                {
                    "Configuration": configuration_name(row.get("axis_value_ids") or {}),
                    "Status": str(row.get("state", "Unknown")).replace("_", " ").capitalize(),
                }
                for row in rows
            ]
        st_module.dataframe(table, hide_index=True, width="stretch")


def _run_action(st_module, study, roots):
    from ifvg_research_pipeline import launch_existing

    label = "Run" if study.status == "Ready" else "Resume study"
    if st_module.button(label, type="primary", key="ifvg_workspace_run"):
        launch_existing(st_module, study, roots)


def _manage_study(st_module, study, roots):
    from alpha_lab.agents.data_infra.ifvg.ifsm_replication import (
        CORRECTED_MORNING,
        corrected_morning_copy,
        uses_legacy_morning,
    )
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import AXIS_VALUE_REGISTRY_V1
    from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
    from alpha_lab.agents.data_infra.ifvg.study_drafts import (
        archive_draft,
        clone_draft,
        restore_draft,
        save_draft,
    )

    if not study.draft and not study.store_root:
        return
    with st_module.expander("Study actions"):
        if study.draft and uses_legacy_morning(study.draft):
            st_module.caption(
                "Legacy morning - 6:00 AM to 9:30 AM Chicago time (historical). "
                "Create a separate corrected copy to use 7:00 AM to 10:30 AM Chicago time."
            )
            if st_module.button(
                "Create corrected morning copy", key="ifvg_workspace_correct_morning",
                disabled=CORRECTED_MORNING not in AXIS_VALUE_REGISTRY_V1,
            ):
                clone = corrected_morning_copy(study.draft)
                save_draft(Path(roots["draft_root"]), clone)
                _open_draft(st_module, clone)
        if study.draft and st_module.button("Clone study", key="ifvg_workspace_clone"):
            clone = clone_draft(study.draft)
            clone.display_name = f"{study.name} (copy)"
            save_draft(Path(roots["draft_root"]), clone)
            _open_draft(st_module, clone)
        if not study.archived:
            name = st_module.text_input(
                "Study name", value=study.name, key=f"ifvg_rename_{study.key}"
            )
            if st_module.button(
                "Save name", disabled=not name.strip() or human_name(name, "") != name.strip()
            ):
                try:
                    if study.draft and study.draft.never_frozen:
                        study.draft.display_name = name.strip()
                        save_draft(Path(roots["draft_root"]), study.draft)
                    elif study.store_root and study.charter_id:
                        append_catalog_event(
                            study.store_root,
                            kind="display_name",
                            artifact_id=study.charter_id,
                            payload={"display_name": name.strip()},
                        )
                    else:
                        st_module.warning(
                            "Renaming is unavailable until this study's saved record is restored."
                        )
                        return
                    st_module.rerun()
                except Exception:
                    st_module.error("The name could not be saved. Refresh and try again.")
        if (
            study.status != "Running"
            and (study.draft or study.store_root)
            and st_module.button("Restore study" if study.archived else "Archive study")
        ):
            try:
                if study.draft:
                    action = restore_draft if study.archived else archive_draft
                    action(Path(roots["draft_root"]), study.draft.draft_id)
                elif study.store_root and study.charter_id:
                    append_catalog_event(
                        study.store_root,
                        kind="archive",
                        artifact_id=study.charter_id,
                        payload=not study.archived,
                    )
                _go(st_module, "list")
            except Exception:
                st_module.error("The study could not be updated. Refresh and try again.")


def render_developer(st_module=st):
    if not technical_details_enabled():
        return
    st_module.title("Developer")
    selected = st_module.radio(
        "Developer area",
        (
            "Verification Center",
            "Data & system health",
            "Research diagnostics",
            "Trade diagnostics",
        ),
        horizontal=True,
    )
    roots = workspace_roots()
    if selected == "Verification Center":
        from ifvg_verification_center import render_verification_center

        render_verification_center(st_module, roots=roots)
    elif selected == "Data & system health":
        from ifvg_research_health import render_health

        render_health(st_module)
    elif selected == "Trade diagnostics":
        from ifvg_lab_tab import render_ifvg_replay_tab

        render_ifvg_replay_tab(st_module)
    else:
        from ifvg_study_tab import render_ifvg_study_tab

        render_ifvg_study_tab(st_module)
