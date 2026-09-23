"""Dedicated UI process; use run_ifsm_research_ui.py to select the pinned engine."""

from pathlib import Path

import streamlit as st


def main():
    import ifvg_study_tab as study_tab
    import strategy_core
    from run_ifsm_research_ui import CORE, WORKSPACE

    st.set_page_config(page_title="IFVG Lab", layout="wide")
    if not Path(strategy_core.__file__).resolve().is_relative_to(CORE):
        st.error("Start this page with: python scripts/run_ifsm_research_ui.py")
        st.stop()
    study_tab.STORE_ROOT_RESEARCH = WORKSPACE / "search/v1"
    study_tab.STORE_ROOT_VERIFICATION = WORKSPACE / "search_test/v1"
    study_tab.STATE_ROOT = WORKSPACE / "jobs"
    study_tab.DRAFT_ROOT = WORKSPACE / "drafts"
    study_tab.VERIFICATION_CENTER_ROOT = WORKSPACE / "verification_center"
    from ifvg_workspace import render_workspace, workspace_roots

    roots = workspace_roots()
    roots.update({
        "pipeline_state_root": WORKSPACE / "pipeline_jobs",
        "context_catalog": WORKSPACE / "context_catalog.json",
        "context_run_root": WORKSPACE / "context_runs",
    })
    render_workspace(st, roots=roots)


if __name__ == "__main__":
    main()
