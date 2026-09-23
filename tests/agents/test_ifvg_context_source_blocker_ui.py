"""Fresh legacy forms show the B0 blocker; saved-study navigation stays usable."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import ifvg_lab_tab as lab  # noqa: E402
import ifvg_research_context as context_ui  # noqa: E402


@pytest.mark.parametrize("surface", ["research_context", "lab_experiments"])
def test_new_context_form_shows_b0_source_route_without_fitting(monkeypatch, surface):
    apptest = pytest.importorskip("streamlit.testing.v1")
    calls = []

    def refused_view(pair, **kwargs):
        calls.append(kwargs)
        assert not kwargs.get("allow_legacy_partial_projection", False)
        raise lab.ArtifactVerificationError(
            "unmapped advertised B0 features: exact selected Core stage audit source "
            "and decision bars are required"
        )

    def no_fit(*args, **kwargs):
        pytest.fail("A missing B0 source must not reach model fitting")

    monkeypatch.setattr(lab, "_load_selected_pair", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(lab, "build_candidate_feature_view", refused_view)
    monkeypatch.setattr(lab, "run_and_catalog_context_experiment", no_fit)
    monkeypatch.setattr(lab, "_render_capability_registry", lambda st: None)
    monkeypatch.setattr(lab, "glossary_expander", lambda st: None)
    monkeypatch.setattr(lab, "_legacy_read_only", lambda st: None)
    monkeypatch.setattr(lab, "_run_history", lambda st: st.info("Saved history remains available"))

    if surface == "research_context":
        def app():
            import ifvg_research_context
            import streamlit as st

            ifvg_research_context.render_context_study(st)
    else:
        def app():
            import ifvg_lab_tab

            ifvg_lab_tab.render_ifvg_experiments_tab()

    rendered = apptest.AppTest.from_function(app, default_timeout=20).run()
    assert not rendered.exception
    assert calls == [{}]
    assert len(rendered.error) == 1
    message = rendered.error[0].value
    assert "verified B0 source evidence" in message
    assert "selected Core stage records and decision bars" in message
    assert "New study → More study types → Feature and model study" in message
    assert "saved context studies remain readable" in message
    assert not any("Run" in button.label for button in rendered.button)
    if surface == "lab_experiments":
        assert any("Saved history remains available" in item.value for item in rendered.info)


def test_saved_context_result_does_not_rebuild_legacy_feature_view(monkeypatch):
    apptest = pytest.importorskip("streamlit.testing.v1")
    stored = object()
    loads = []

    def load(run_id, **kwargs):
        loads.append(run_id)
        return stored

    def no_fresh_source(*args, **kwargs):
        pytest.fail("Reading an existing result must not prepare a new candidate view")

    def render(st, result):
        assert result is stored
        st.success("Existing context result loaded")

    monkeypatch.setattr(lab, "load_context_experiment_run", load)
    monkeypatch.setattr(lab, "build_candidate_feature_view", no_fresh_source)
    monkeypatch.setattr(lab, "_load_selected_pair", no_fresh_source)
    monkeypatch.setattr(context_ui, "render_context_result", render)

    def app():
        import ifvg_research_context
        import streamlit as st

        ifvg_research_context.render_context_study(st, run_id="preserved-context-run")

    rendered = apptest.AppTest.from_function(app, default_timeout=20).run()
    assert not rendered.exception and not rendered.error
    assert loads == ["preserved-context-run"]
    assert rendered.success[0].value == "Existing context result loaded"
