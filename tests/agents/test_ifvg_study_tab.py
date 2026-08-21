"""FUX-IA-001..003: shell order, exact sub-navigation, single-route
execution, unchanged Context Research delegation (TEST_MATRIX §3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_study_tab as study_tab  # noqa: E402


def _app() -> None:
    import ifvg_study_tab as study_tab
    import streamlit as st

    study_tab.render_ifvg_study_tab(st)


def _patched_roots(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", tmp_path / "research")
    monkeypatch.setattr(
        study_tab, "STORE_ROOT_VERIFICATION", tmp_path / "verification"
    )
    monkeypatch.setattr(study_tab, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", tmp_path / "drafts")


def test_subnav_is_the_exact_horizontal_radio(monkeypatch, tmp_path) -> None:
    """FUX-IA-002: New Study / Active Runs / Results / History / Context
    Research, session-state-backed, horizontal."""

    _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    nav = at.radio[0]
    assert nav.options == [
        "New Study",
        "Active Runs",
        "Results",
        "History",
        "Context Research",
    ]
    assert nav.value == "New Study"
    assert at.session_state[study_tab.ROUTE_KEY] == "New Study"


def test_only_the_selected_route_executes(monkeypatch, tmp_path) -> None:
    """FUX-IA-002/§3.3: hidden panels never render, poll, or build charts."""

    _patched_roots(monkeypatch, tmp_path)
    calls: list[str] = []
    import ifvg_active_runs_tab as monitor
    import ifvg_results_tab as results
    import ifvg_study_wizard as wizard

    monkeypatch.setattr(
        wizard, "render_new_study", lambda st, *, roots: calls.append("wizard")
    )
    monkeypatch.setattr(
        monitor,
        "render_active_runs",
        lambda st, *, roots: calls.append("monitor"),
    )
    monkeypatch.setattr(
        results, "render_results", lambda st, *, roots: calls.append("results")
    )
    monkeypatch.setattr(
        results, "render_history", lambda st, *, roots: calls.append("history")
    )
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert calls == ["wizard"]
    calls.clear()
    at.radio[0].set_value("Results").run()
    assert calls == ["results"]
    calls.clear()
    at.radio[0].set_value("History").run()
    assert calls == ["history"]


def test_context_research_delegates_verbatim(monkeypatch, tmp_path) -> None:
    """FUX-IA-003: the existing M0–M3 renderer is called as-is."""

    _patched_roots(monkeypatch, tmp_path)
    seen: list[object] = []

    def _app_with_spy() -> None:
        import ifvg_study_tab as study_tab
        import streamlit as st

        def _context(st_module) -> None:
            st_module.write("m0m3-panel-marker")

        study_tab.render_ifvg_study_tab(st, context_research=_context)

    at = apptest.AppTest.from_function(_app_with_spy, default_timeout=60)
    at.run()
    at.radio[0].set_value("Context Research").run()
    assert not at.exception
    assert any("m0m3-panel-marker" in str(block.value) for block in at.markdown) or any(
        "m0m3-panel-marker" in str(getattr(block, "value", ""))
        for block in at.get("text")
    )
    # the namespace selector belongs to the study routes, not Context Research
    assert len(at.radio) == 1
    assert seen == []


def test_default_delegate_is_the_existing_experiments_panel(
    monkeypatch, tmp_path
) -> None:
    _patched_roots(monkeypatch, tmp_path)
    import ifvg_lab_tab as lab

    called: list[str] = []
    monkeypatch.setattr(
        lab, "render_ifvg_experiments_tab", lambda st_module: called.append("m0m3")
    )
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.radio[0].set_value("Context Research").run()
    assert called == ["m0m3"]


def test_programmatic_route_request_applies_next_run(monkeypatch, tmp_path) -> None:
    _patched_roots(monkeypatch, tmp_path)
    import ifvg_active_runs_tab as monitor
    import ifvg_study_wizard as wizard

    monkeypatch.setattr(wizard, "render_new_study", lambda st, *, roots: None)
    monkeypatch.setattr(
        monitor, "render_active_runs", lambda st, *, roots: None
    )
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    at.session_state[f"{study_tab.STATE_PREFIX}pending_route"] = "Active Runs"
    at.run()
    assert at.session_state[study_tab.ROUTE_KEY] == "Active Runs"


def test_top_level_shell_order_is_unchanged(monkeypatch, tmp_path) -> None:
    """FUX-IA-001 — the lab shell keeps Experiments · Replay / Verifier ·
    Data & Audit, with Experiments now delegating to the study workspace."""

    _patched_roots(monkeypatch, tmp_path)
    import ifvg_lab_tab as lab

    monkeypatch.setattr(
        lab, "render_ifvg_replay_tab", lambda st_module: None
    )
    monkeypatch.setattr(
        lab, "render_ifvg_data_audit_tab", lambda st_module: None
    )

    def _shell() -> None:
        import ifvg_lab_tab as lab

        lab.render_ifvg_lab_tab()

    at = apptest.AppTest.from_function(_shell, default_timeout=60)
    at.run()
    assert not at.exception
    assert [tab.label for tab in at.tabs[:3]] == [
        "Experiments",
        "Replay / Verifier",
        "Data & Audit",
    ]
