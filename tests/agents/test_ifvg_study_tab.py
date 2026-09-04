"""FUX-IA-001..003 (as amended by UI-1): shell order, exact sub-navigation
(Start · Verify Implementation · New Study · Active Runs · Results · History ·
Context Research), single-route execution, unchanged Context Research
delegation, the namespace radio GONE, Start cards deriving the purpose."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_study_tab as study_tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.study_drafts import (  # noqa: E402
    DraftNotFoundError,
    load_draft,
)

_ROUTES = [
    "Start",
    "Verify Implementation",
    "New Study",
    "Active Runs",
    "Results",
    "History",
    "Context Research",
]


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


def _text(at) -> str:
    return "\n".join(
        [str(b.value) for b in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(h.value) for h in at.subheader]
        + [str(w.value) for w in at.warning]
        + [str(e.value) for e in at.error]
    )


def test_subnav_is_the_exact_horizontal_radio(monkeypatch, tmp_path) -> None:
    """FUX-IA-002 as amended: Start first, Verify Implementation second."""

    _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    nav = at.radio[0]
    assert nav.options == _ROUTES
    assert nav.value == "Start"
    assert at.session_state[study_tab.ROUTE_KEY] == "Start"


def test_namespace_radio_is_gone(monkeypatch, tmp_path) -> None:
    """UI-1 (plan F-01 / owner Q1): no mutable namespace selector exists on
    any route; the namespace derives from the purpose."""

    _patched_roots(monkeypatch, tmp_path)
    assert not hasattr(study_tab, "NAMESPACE_KEY")
    for route in ("Start", "Verify Implementation", "New Study", "Results", "History"):
        at = apptest.AppTest.from_function(_app, default_timeout=120)
        at.session_state[study_tab.ROUTE_KEY] = route
        at.run()
        assert not at.exception, route
        labels = [radio.label for radio in at.radio]
        assert "Artifact namespace" not in labels, route
        assert all("namespace" not in (label or "").lower() for label in labels), route
    roots = study_tab.workspace_roots()
    assert set(roots["store_roots"]) == {"research", "test"}
    verification = study_tab.roots_for_purpose(roots, "implementation_verification")
    assert verification["store_root"] == tmp_path / "verification"
    assert verification["namespace_class"] == "test"
    research = study_tab.roots_for_purpose(roots, "development_research")
    assert research["store_root"] == tmp_path / "research"
    assert research["namespace_class"] == "research"


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
    assert calls == []  # Start renders the task cards only
    at.radio[0].set_value("New Study").run()
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
    # no other selector belongs to Context Research
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


def test_start_cards_derive_purpose_namespace_and_create_annotated_drafts(
    monkeypatch, tmp_path
) -> None:
    """Plan §5.2: nine task cards; each derives purpose, family and namespace;
    a card creates a draft carrying the purpose annotation and routes to New
    Study; the full-scope card is visible-disabled until readiness is ready."""

    _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception
    text = _text(at)
    assert [card.card_id for card in study_tab.TASK_CARDS] == [
        "verify_implementation",
        "review_setups",
        "evaluate_one",
        "compare_with_baseline",
        "fsm_search",
        "feature_model_evidence",
        "prop_feasibility",
        "advanced_end_to_end",
        "inspect_health",
    ]
    for card in study_tab.TASK_CARDS:
        assert card.title in text, card.card_id
    assert "namespace class `test`" in text
    assert "namespace class `research`" in text
    assert "Store namespace: **unmarked**" in text  # never guessed from the path
    buttons = {button.key: button for button in at.button}
    full = buttons[f"{study_tab._START}advanced_end_to_end"]
    assert full.disabled is False  # visible; readiness gates the freeze, not the draft
    assert "authorization readiness: store_unmarked" in text
    prop = buttons[f"{study_tab._START}prop_feasibility"]
    assert prop.disabled  # no first_party_verified contract exists
    compare = buttons[f"{study_tab._START}compare_with_baseline"]
    compare.click().run()
    assert not at.exception
    draft_id = at.session_state[f"{study_tab.STATE_PREFIX}draft_id"]
    # UI-2 (owner Q2): the card creates a SESSION draft — no file is written
    assert not (tmp_path / "drafts" / draft_id).exists()
    with pytest.raises(DraftNotFoundError):
        load_draft(tmp_path / "drafts", draft_id)
    draft = at.session_state[study_tab.SESSION_DRAFT_KEY]
    assert draft["draft_id"] == draft_id
    assert draft["mode_id"] == "single_configuration"
    assert draft["purpose_annotation"]["purpose"] == "development_research"
    assert draft["purpose_annotation"]["derivation"] == "card_selected"
    assert draft["steps"]["objective"]["question_id"] == "compare_one_with_baseline"
    assert draft["steps"]["validation"]["run_scope"] == "full_authorized_development"
    assert draft["steps"]["validation"]["evidence_class"] == "real"
    assert draft["steps"]["validation"]["worker_limit"] == 1
    assert draft["display_name"].startswith("Compare one configuration with the baseline — 2026-")
    assert at.session_state[study_tab.ROUTE_KEY] == "New Study"


def test_verify_implementation_renders_typed_readiness(monkeypatch, tmp_path) -> None:
    """The Verification Center surface renders the verified-namespace state
    and the TYPED VerificationAuthorizationRef readiness — never a Boolean —
    and starts an Implementation Verification draft in the test namespace."""

    _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[study_tab.ROUTE_KEY] = "Verify Implementation"
    at.run()
    assert not at.exception
    text = _text(at)
    assert "Verification Center" in text
    assert "VERIFICATION ONLY" in text
    assert "Semantic store namespace: **unmarked**" in text
    assert "(VerificationAuthorizationRef): **store_unmarked**" in text
    assert "21/R-5:verification_fixture_authorization" in text
    assert "Verification authorization missing" in text
    labels = [button.label for button in at.button]
    assert "Start a verification draft" in labels
    assert not any(
        word in label.lower()
        for label in labels
        for word in ("sign", "produce seed", "publish", "activate")
    )
    next(b for b in at.button if b.label == "Start a verification draft").click().run()
    assert not at.exception
    draft_id = at.session_state[f"{study_tab.STATE_PREFIX}draft_id"]
    assert not (tmp_path / "drafts" / draft_id).exists()  # session-only (owner Q2)
    draft = at.session_state[study_tab.SESSION_DRAFT_KEY]
    assert draft["purpose_annotation"]["purpose"] == "implementation_verification"
    assert draft["steps"]["validation"]["run_scope"] == "verification_5d"
    assert draft["steps"]["validation"]["evidence_class"] == "synthetic_fixture"


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
