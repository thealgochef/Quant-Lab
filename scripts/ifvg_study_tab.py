"""Experiments sub-navigation router for the study workspace (R4; FUX §3).

``render_ifvg_study_tab`` owns the session-state-backed horizontal radio:

    New Study | Active Runs | Results | History | Context Research

Only the selected sub-surface executes (hidden panels never poll jobs or
build charts — FUX §3.3); ``Context Research`` delegates to the existing
M0–M3 experiments panel verbatim (FUX-IA-003); programmatic navigation
(wizard freeze/launch → Active Runs; monitor row → Results; History clone →
New Study) goes through :func:`request_route`, applied before the radio is
instantiated on the next run.

Import launches nothing; every root is a module attribute so tests inject
temporary directories.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from ifvg_ui_common import STATE_PREFIX, sanitize_select

from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_STORE_ROOT,
    SEARCH_TEST_STORE_ROOT,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import STUDY_DRAFT_ROOT
from alpha_lab.agents.data_infra.ifvg.study_status import (
    ROUTE_LABELS,
    StudyWorkspaceRoute,
)

__all__ = [
    "render_ifvg_study_tab",
    "request_route",
    "ROUTE_KEY",
    "NAMESPACE_KEY",
    "workspace_roots",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Module-level roots — tests monkeypatch these with tmp directories.
STORE_ROOT_RESEARCH = _REPO_ROOT / SEARCH_STORE_ROOT
STORE_ROOT_VERIFICATION = _REPO_ROOT / SEARCH_TEST_STORE_ROOT
STATE_ROOT = _REPO_ROOT / "data/ifvg_search_jobs"
DRAFT_ROOT = _REPO_ROOT / STUDY_DRAFT_ROOT

ROUTE_KEY = f"{STATE_PREFIX}route"
_PENDING_ROUTE_KEY = f"{STATE_PREFIX}pending_route"
NAMESPACE_KEY = f"{STATE_PREFIX}namespace"

_NAMESPACES = {
    "Research (search/v1)": "research",
    "Verification / synthetic (search_test/v1)": "verification",
}


def request_route(st_module, route: StudyWorkspaceRoute | str) -> None:
    """Queue a programmatic sub-route change (applied on the next run)."""

    st_module.session_state[_PENDING_ROUTE_KEY] = ROUTE_LABELS[
        StudyWorkspaceRoute(route)
    ]


def workspace_roots(st_module) -> dict[str, Path]:
    """The active store/state/draft roots for the selected namespace."""

    namespace_label = st_module.session_state.get(
        NAMESPACE_KEY, next(iter(_NAMESPACES))
    )
    namespace = _NAMESPACES.get(namespace_label, "research")
    store_root = (
        STORE_ROOT_RESEARCH if namespace == "research" else STORE_ROOT_VERIFICATION
    )
    return {
        "store_root": store_root,
        "state_root": STATE_ROOT,
        "draft_root": DRAFT_ROOT,
        "namespace": namespace,  # type: ignore[dict-item]
    }


def render_ifvg_study_tab(st_module=st, context_research=None) -> None:
    """The Experiments surface: sub-nav radio + exactly one executed route."""

    route_labels = [ROUTE_LABELS[route] for route in StudyWorkspaceRoute]
    pending = st_module.session_state.pop(_PENDING_ROUTE_KEY, None)
    if pending in route_labels:
        st_module.session_state[ROUTE_KEY] = pending
    sanitize_select(st_module, ROUTE_KEY, route_labels)
    selected = st_module.radio(
        "Experiments workspace",
        route_labels,
        horizontal=True,
        key=ROUTE_KEY,
        label_visibility="collapsed",
    )
    if selected == ROUTE_LABELS[StudyWorkspaceRoute.CONTEXT_RESEARCH]:
        if context_research is None:
            from ifvg_lab_tab import render_ifvg_experiments_tab  # noqa: PLC0415

            context_research = render_ifvg_experiments_tab
        context_research(st_module)
        return

    namespace_labels = list(_NAMESPACES)
    sanitize_select(st_module, NAMESPACE_KEY, namespace_labels)
    st_module.radio(
        "Artifact namespace",
        namespace_labels,
        horizontal=True,
        key=NAMESPACE_KEY,
        help=(
            "Research reads data/ifvg_datasets/search/v1; Verification / "
            "synthetic reads the search_test namespace every verification "
            "and synthetic fixture publishes into. Verification artifacts "
            "are never research evidence."
        ),
    )
    roots = workspace_roots(st_module)

    if selected == ROUTE_LABELS[StudyWorkspaceRoute.NEW_STUDY]:
        from ifvg_study_wizard import render_new_study  # noqa: PLC0415

        render_new_study(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.ACTIVE_RUNS]:
        from ifvg_active_runs_tab import render_active_runs  # noqa: PLC0415

        render_active_runs(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.RESULTS]:
        from ifvg_results_tab import render_results  # noqa: PLC0415

        render_results(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.HISTORY]:
        from ifvg_results_tab import render_history  # noqa: PLC0415

        render_history(st_module, roots=roots)
