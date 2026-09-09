"""Rules remain attached to the visible study and selected result."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import (  # noqa: E402
    RuleDescription,
)
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary  # noqa: E402


def _study(key="study", **kwargs):
    return StudySummary(
        key=key,
        kind="search",
        name=f"Study {key}",
        question="Which settings perform consistently?",
        dates="2026-01-13",
        status="Completed",
        scope="research",
        **kwargs,
    )


def _description(**kwargs):
    return RuleDescription(
        status="available",
        preview_bullets=(
            "If a zone is touched, watch for a setup.",
            "Wait for confirmation.",
            "Compare the waiting limits.",
        ),
        detail_bullets=(
            "If a zone is touched, watch for a setup.",
            "If confirmed, buy with a stop and target.",
        ),
        variations=("Compare waiting for 240, 360, or 480 one-minute candles.",),
        **kwargs,
    )


def _cards_app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_my_studies(st, ifvg_workspace._RULES_STUDIES, {})


def _details_app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_study(st, ifvg_workspace._RULES_STUDY, {})


def _results_app():
    import ifvg_research_results as results
    import streamlit as st

    results.render_bundle(st, results._RULES_BUNDLE, study=results._RULES_STUDY, roots={})


def test_cards_load_rules_only_for_current_page(monkeypatch):
    import ifvg_rules
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    rows = [_study(str(index)) for index in range(11)]
    calls = []

    def load(study, roots, selected_core_replay_id=None):
        calls.append(study.key)
        return _description()

    monkeypatch.setattr(ifvg_rules, "load_study_rules", load)
    monkeypatch.setattr(ifvg_workspace, "_RULES_STUDIES", rows, raising=False)
    at = AppTest.from_function(_cards_app).run()
    assert not at.exception
    assert calls == [str(index) for index in range(10)]
    previews = [item.value for item in at.markdown if item.value.startswith("- ")]
    assert len(previews) == 10
    assert all(len(text.splitlines()) == 3 for text in previews)
    assert not any("What this study changes" in item.value for item in at.markdown)
    calls.clear()
    at.number_input(key="ifvg_workspace_page").set_value(2).run()
    assert not at.exception
    assert calls == ["10"]


def test_study_page_shows_common_rules_and_variations_once(monkeypatch):
    import ifvg_research_results
    import ifvg_rules
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_workspace, "_RULES_STUDY", _study(), raising=False)
    monkeypatch.setattr(ifvg_rules, "load_study_rules", lambda *args: _description())
    monkeypatch.setattr(ifvg_research_results, "render_search_results", lambda *args: None)
    at = AppTest.from_function(_details_app).run()
    assert not at.exception
    text = "\n".join(item.value for item in at.markdown)
    assert text.count("**Strategy rules**") == 1
    assert text.count("**What this study changes**") == 1
    assert text.count("If confirmed, buy with a stop and target.") == 1
    assert "240, 360, or 480" in text


def test_selected_result_updates_its_exact_rules(monkeypatch, tmp_path):
    import ifvg_research_results as results
    import ifvg_rules
    from ifvg_results_tab import load_results_bundle
    from streamlit.testing.v1 import AppTest

    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(tmp_path, with_prop=False, with_contract=False)
    bundle = load_results_bundle(
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        search_id=fixture["search_id"],
    )
    children = [row["core_replay_id"] for row in bundle["children"]]
    calls = []

    def load(study, roots, selected_core_replay_id=None):
        calls.append((study.key, selected_core_replay_id))
        index = children.index(selected_core_replay_id)
        return replace(
            _description(),
            detail_bullets=(f"If confirmed, apply rule {index + 1}.",),
            variations=(),
        )

    monkeypatch.setattr(ifvg_rules, "load_study_rules", load)
    monkeypatch.setattr(results, "_RULES_BUNDLE", bundle, raising=False)
    monkeypatch.setattr(results, "_RULES_STUDY", _study("exact"), raising=False)
    at = AppTest.from_function(_results_app, default_timeout=60).run()
    assert not at.exception
    assert "Rules for this configuration" in [item.label for item in at.expander]
    assert calls[-1] == ("exact", children[0])
    at.selectbox(key="ifvg_research_results_configuration").select(children[1]).run()
    assert not at.exception
    assert calls[-1] == ("exact", children[1])
    text = "\n".join(item.value for item in at.markdown)
    assert "If confirmed, apply rule 2." in text
    assert "If confirmed, apply rule 1." not in text


def test_unconfigured_draft_preview_is_readable(monkeypatch, tmp_path):
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft

    draft = new_draft("single_configuration", display_name="Not configured yet")
    study = replace(_study(), kind="draft", status="Draft", draft=draft)
    monkeypatch.setattr(ifvg_workspace, "_RULES_STUDIES", [study], raising=False)
    at = AppTest.from_function(_cards_app).run()
    assert not at.exception
    text = "\n".join(item.value for item in (*at.markdown, *at.caption))
    assert "Choose a strategy configuration to see its rules." in text
    assert "Preview" in text


def test_pipeline_rules_distinguish_model_and_account_results(monkeypatch):
    import ifvg_research_pipeline
    import ifvg_rules
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        ifvg_workspace, "_RULES_STUDY", replace(_study(), kind="pipeline"), raising=False
    )
    monkeypatch.setattr(ifvg_rules, "load_study_rules", lambda *args: _description())
    monkeypatch.setattr(ifvg_research_pipeline, "render_pipeline_results", lambda *args: None)
    at = AppTest.from_function(_details_app).run()
    assert not at.exception
    assert any(
        "Model predictions and account simulations are evaluated separately." in item.value
        for item in at.caption
    )
