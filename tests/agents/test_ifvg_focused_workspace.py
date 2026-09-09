"""Focused workspace acceptance: boundaries, scientific states and parent Replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.replay_selection import (
    ReplaySelection,  # noqa: E402
)
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (  # noqa: E402
    _draft_scope,
    load_studies,
    pipeline_status,
    search_status,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft, save_draft  # noqa: E402


@pytest.fixture
def roots(tmp_path):
    return {
        "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store", "test": tmp_path / "test"},
        "draft_root": tmp_path / "drafts",
        "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "repo_root": tmp_path,
    }


def test_startup_flag_cannot_be_changed_by_session_environment(monkeypatch):
    before = workspace_mode.DEVELOPER_MODE
    monkeypatch.setenv("QUANT_LAB_DEVELOPER_MODE", "0" if before else "1")
    assert workspace_mode.DEVELOPER_MODE is before
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    with pytest.raises(PermissionError), workspace_mode.developer_area():
        pytest.fail("Disabled Developer surface executed")


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, "Evidence unavailable"),
        ({"phase": "search_complete"}, "Completed"),
        ({"phase": "failed"}, "Failed"),
        ({"phase": "cancelled"}, "Interrupted"),
        ({"phase": "replays", "progress": 100}, "Running"),
        ({"phase": "nonsense", "progress": 100}, "Status unresolved"),
    ],
)
def test_search_lifecycle_comes_from_persisted_state(state, expected):
    assert search_status(state) == expected


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["completed", "pending"], "Status unresolved"),
        (["completed", "blocked"], "Blocked"),
        (["completed", "failed"], "Failed"),
        (["completed", "running"], "Running"),
        (["completed", "reused"], "Completed"),
        ([], "Status unresolved"),
    ],
)
def test_pipeline_completion_is_not_a_progress_bar(statuses, expected):
    state = {
        "stages": {str(i): {"in_plan": True, "status": status} for i, status in enumerate(statuses)}
    }
    state["stages"]["not planned"] = {"in_plan": False, "status": "pending"}
    assert pipeline_status(state) == expected


def test_legacy_scope_is_unresolved_and_verification_is_not_research(roots):
    draft = new_draft("single_configuration", display_name="Legacy study")
    draft.steps["validation"] = {"run_scope": "full_authorized_development"}
    assert _draft_scope(draft) == "unresolved"
    save_draft(roots["draft_root"], draft)
    verification = new_draft("single_configuration", display_name="Fixture study")
    verification.steps["validation"] = {"run_scope": "verification_5d"}
    save_draft(roots["draft_root"], verification)
    rows, issues = load_studies(roots)
    assert not issues
    assert [row.name for row in rows] == ["Legacy study"]
    assert rows[0].scope == "unresolved"
    rows, _ = load_studies(roots, include_verification=True)
    assert len(rows) == 2


@pytest.mark.parametrize(
    "selection",
    [
        ReplaySelection("setup", "setup-with-no-candidates"),
        ReplaySelection("empty"),
        ReplaySelection("unavailable", evidence_available=False),
        ReplaySelection("candidate", "selected-exact-candidate"),
    ],
)
def test_parent_replay_never_falls_through(selection, monkeypatch):
    import ifvg_lab_tab as lab
    import ifvg_verifier_tab as verifier

    calls = []
    pair = SimpleNamespace(v3=SimpleNamespace(tables={}))
    monkeypatch.setattr(lab, "_load_selected_pair", lambda *a, **k: (pair, {}))
    monkeypatch.setattr(
        verifier, "render_verifier_section", lambda *a: calls.append("verifier") or selection
    )

    class Screen:
        def subheader(self, _text):
            pass

        def selectbox(self, *args, **kwargs):
            pytest.fail("A second selector must never replace the selected setup or empty result")

    assert lab.render_ifvg_replay_tab(Screen()) == selection
    assert calls == ["verifier"]


def _workspace_app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def test_open_switch_and_configure_do_not_launch_or_render_developer(monkeypatch, roots):
    import ifvg_lab_tab
    import ifvg_study_wizard
    import ifvg_verification_center
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)

    def forbidden(*args, **kwargs):
        pytest.fail("Hidden technical renderer or launch handler executed")

    monkeypatch.setattr(ifvg_verification_center, "render_verification_center", forbidden)
    monkeypatch.setattr(ifvg_lab_tab, "render_ifvg_data_audit_tab", forbidden)
    monkeypatch.setattr(ifvg_study_wizard, "_spawn_search_job", forbidden)
    at = AppTest.from_function(_workspace_app, default_timeout=60).run()
    assert not at.exception
    assert at.radio[0].options == ["My studies", "Trade review"]
    assert not at.code and not at.json
    at.button(key="ifvg_workspace_new").click().run()
    assert not at.exception
    next(button for button in at.button if button.label == "Configure study").click().run()
    assert not at.exception
    assert not at.code and not at.json
    assert not list(roots["draft_root"].glob("*/draft.json"))
    forbidden_words = ("namespace", "authorization_class", "search_id", "S05", "skipped steps")
    text = " ".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "info", "error")
        for element in at.get(kind)
    )
    assert not any(word in text for word in forbidden_words)
    next(button for button in at.button if button.label == "Next").click().run()
    assert not at.exception
    saved = list(roots["draft_root"].glob("*/draft.json"))
    assert len(saved) == 1
    payload = json.loads(saved[0].read_text())
    assert payload["current_step_key"] == "baseline"
    assert "ifvg_v2_" not in " ".join(at.selectbox[0].options)


def test_normal_draft_management_is_reachable_from_my_studies(monkeypatch, roots):
    import ifvg_workspace
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    draft = start_draft_from_card(
        next(card for card in TASK_CARDS if card.card_id == "evaluate_one"), roots
    )
    draft.display_name = "Manage this study"
    save_draft(roots["draft_root"], draft)
    at = AppTest.from_function(_workspace_app, default_timeout=60).run()
    assert not at.exception
    at.button(key=f"ifvg_details_{draft.draft_id}").click().run()
    assert not at.exception
    labels = {button.label for button in at.button}
    assert {"Continue", "Clone study", "Save name", "Archive study"} <= labels
    assert not at.code and not at.json


def test_result_explanations_hide_nested_sources_but_keep_evidence():
    from ifvg_research_results import research_sentence

    explanation = (
        "Brier score is 0.3153; does not beat the boundary 0.2622 "
        "(context_statistics.binary_prediction_report (the persisted candidate report); "
        "key 'reference_brier_score'). The 95% interval (-0.36 R, 0.24 R) crosses zero."
    )
    assert research_sentence(explanation) == (
        "Brier score is 0.3153; does not beat the boundary 0.2622. "
        "The 95% interval (-0.36 R, 0.24 R) crosses zero."
    )


def test_normal_chart_hides_identity_and_preserves_prices():
    import plotly.graph_objects as go
    from ifvg_research_review import research_figure

    identity = "a" * 64
    original = go.Figure(
        go.Scatter(
            x=[1, 2],
            y=[100, 101],
            hovertext=[f"FVG {identity}<br>price 100", "price 101"],
            meta={"id": identity},
        )
    )
    original.update_layout(title=identity, meta={"id": identity})
    safe = research_figure(original, title="Trade evidence")
    assert identity not in safe.to_json()
    assert identity in original.to_json()
    assert list(safe.data[0].y) == [100, 101]


def test_human_research_tables_preserve_zero_false_and_nullable_fields():
    from ifvg_research_context import _human_frame

    frame = pd.DataFrame(
        {
            "reason": pd.Series([pd.NA, "no_oos_predictions"], dtype="string"),
            "observed": pd.Series([0, False], dtype=object),
            "candidate_id": ["a" * 64, "b" * 64],
        }
    )
    table = _human_frame(frame)
    assert "Candidate id" not in table
    assert list(table["Observed"]) == ["0", "False"]
    assert table.loc[0, "Reason"] == ""


def test_health_unknown_and_conflicting_reports_never_pass():
    from ifvg_research_health import health_reading

    assert health_reading({})[0] == "Unknown"
    assert health_reading({"count": 0})[0] == "Unknown"
    assert health_reading({"status": "verified", "count": 0})[0] == "Unknown"
    assert health_reading({"passed": True, "valid": False})[0] == "Failed"
    assert health_reading({"passed": True})[0] == "Passed"


def test_setup_parent_integration_no_candidate_links(monkeypatch):
    import ifvg_lab_tab as lab
    import ifvg_verifier_tab as verifier
    from streamlit.testing.v1 import AppTest

    from tests.agents.test_ifvg_setup_verifier_tab import (
        _bundle,
        _entry,
        _fake_ctx,
        _make_evidence,
        _setups_frame,
    )

    monkeypatch.setattr(
        lab,
        "_load_selected_pair",
        lambda *a, **k: (SimpleNamespace(v3=SimpleNamespace(tables={})), _entry()),
    )
    monkeypatch.setattr(verifier, "_discover_setup_bundle", lambda *a: _bundle())
    monkeypatch.setattr(verifier, "_cached_setup_context", lambda *a: _fake_ctx())
    monkeypatch.setattr(verifier, "_cached_setup_frame", lambda *a: _setups_frame())
    monkeypatch.setattr(
        verifier,
        "_cached_setup_evidence",
        lambda artifact, setup, stage, ctx: _make_evidence(setup, stage),
    )
    monkeypatch.setattr(verifier, "list_reviews", lambda **kwargs: pd.DataFrame())

    def app():
        import ifvg_lab_tab
        import streamlit as st

        st.session_state["selection"] = ifvg_lab_tab.render_ifvg_replay_tab(st)

    at = AppTest.from_function(app, default_timeout=60)
    at.session_state["ifvg_context_v1_selection_mode"] = "setup"
    at.run()
    assert not at.exception
    assert at.session_state["selection"].kind == "setup"
    assert not at.code and not at.json
    assert not any(widget.label == "Exact candidate ID" for widget in at.selectbox)
    assert any("produced no entry opportunity" in str(item.value) for item in at.info)
    at.selectbox(key="ifvg_context_v1_setup_candidate_less").set_value("with candidates").run()
    assert not at.exception
    assert at.session_state["selection"].kind == "setup"
    at.multiselect(key="ifvg_context_v1_setup_terminal_reason").set_value(
        ["invalidated_htf_filled"]
    ).run()
    assert not at.exception
    assert at.session_state["selection"].kind == "empty"


def test_corrupt_progress_does_not_hide_other_studies(roots):
    good, broken = "a" * 64, "b" * 64
    for identity, content in ((good, '{"phase":"failed"}'), (broken, "{broken")):
        directory = roots["state_root"] / identity
        directory.mkdir(parents=True)
        (directory / "search_state.json").write_text(content)
    rows, _issues = load_studies(roots)
    by_key = {row.key: row for row in rows}
    assert set(by_key) == {good, broken}
    assert by_key[broken].status == "Evidence unavailable"
    assert by_key[broken].scope == "unresolved"
    assert by_key[good].status == "Failed"


def test_frozen_verification_scope_ignores_mutable_draft_annotation(roots):
    from alpha_lab.agents.data_infra.ifvg.study_drafts import mark_frozen
    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(
        roots["repo_root"] / "verification", with_prop=False, with_contract=False
    )
    roots["store_roots"]["test"] = fixture["store_root"]
    draft = new_draft("single_configuration", display_name="Mislabeled fixture")
    draft.steps["validation"] = {"run_scope": "full_authorized_development"}
    save_draft(roots["draft_root"], draft)
    mark_frozen(roots["draft_root"], draft, search_id=fixture["search_id"])
    assert not load_studies(roots)[0]
    assert load_studies(roots, include_verification=True)[0][0].scope == "verification"


def _research_results_app():
    import ifvg_research_results as results
    import streamlit as st

    results.render_bundle(st, results._TEST_BUNDLE)


def _visible_text(at):
    values = []
    for kind in ("markdown", "caption", "warning", "info", "error", "success", "subheader"):
        values.extend(str(item.value) for item in at.get(kind))
    for widget in at.selectbox:
        values.extend(widget.options)
    for metric in at.metric:
        values.extend([metric.label, str(metric.proto.help)])
    for table in at.dataframe:
        values.append(table.value.to_csv(index=False))
    return "\n".join(values)


def test_strategy_results_have_four_relevant_metrics_and_no_technical_exports(
    monkeypatch, tmp_path
):
    import re

    import ifvg_research_results as results
    from ifvg_results_tab import load_results_bundle
    from streamlit.testing.v1 import AppTest

    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(tmp_path, with_prop=False, with_contract=False)
    bundle = load_results_bundle(
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        search_id=fixture["search_id"],
    )
    monkeypatch.setattr(results, "_TEST_BUNDLE", bundle, raising=False)
    at = AppTest.from_function(_research_results_app, default_timeout=60).run()
    assert not at.exception
    assert [metric.label for metric in at.metric] == [
        "Net expectancy",
        "Profit factor",
        "Maximum drawdown",
        "Executed trades",
    ]
    assert not at.code and not at.json
    text = _visible_text(at)
    assert not re.search(r"\bprop\b", text.lower())
    assert not re.search(r"\b[0-9a-f]{32,}\b|\b\w+_\w+\b", text)
    at.checkbox(key="ifvg_research_result_details").check().run()
    assert not at.exception
    assert not re.search(r"\b[0-9a-f]{32,}\b|\b\w+_\w+\b", _visible_text(at))


@pytest.mark.parametrize(
    "card_id",
    [
        "evaluate_one",
        "compare_with_baseline",
        "fsm_search",
        "prop_feasibility",
        "feature_model_evidence",
        "advanced_end_to_end",
    ],
)
def test_each_guided_flow_renders_its_applicable_steps(monkeypatch, roots, card_id):
    import ifvg_research_wizard as wizard
    import ifvg_study_wizard as existing
    import ifvg_workspace
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    draft = start_draft_from_card(
        next(card for card in TASK_CARDS if card.card_id == card_id), roots
    )
    draft.display_name = "Guided research"
    flow = existing._resolve_for_draft(draft, roots).flow
    monkeypatch.setattr(existing, "_spawn_search_job", lambda *_: pytest.fail("Unexpected launch"))

    def app():
        import ifvg_research_wizard
        import ifvg_workspace
        import streamlit as st

        ifvg_research_wizard.render_new_study(st, roots=ifvg_workspace._TEST_ROOTS)

    for step in flow.steps:
        draft.current_step_key = step.key
        save_draft(roots["draft_root"], draft)
        at = AppTest.from_function(app, default_timeout=60)
        at.session_state[wizard._DRAFT] = draft.draft_id
        at.run()
        assert not at.exception, step.key
        assert not at.code and not at.json
        assert "S05" not in _visible_text(at)
        assert "namespace" not in _visible_text(at).lower()


def test_model_workflow_inconclusive_metrics_remain_unavailable():
    from streamlit.testing.v1 import AppTest

    def app():
        import ifvg_research_pipeline
        import streamlit as st

        ifvg_research_pipeline._model_details(
            st,
            {
                "reference_prevalence_v1": {
                    "prediction_report": {"count": 0, "auc": None},
                    "fold_reports": [],
                }
            },
            {},
        )

    at = AppTest.from_function(app).run()
    assert not at.exception
    assert any("inconclusive" in item.value for item in at.warning)
    assert all(metric.value not in ("0", "0.0", "0.00") for metric in at.metric)


def test_router_registers_developer_only_at_startup_and_executes_selected_page(monkeypatch):
    import dashboard

    called, pages = [], []

    class Shell:
        def set_page_config(self, **kwargs):
            pass

        def Page(self, function, **kwargs):  # noqa: N802 - Streamlit API
            page = SimpleNamespace(run=function, title=kwargs["title"])
            pages.append(page)
            return page

        def navigation(self, registered):
            return next(page for page in registered if page.title == "ML Training")

    monkeypatch.setattr(dashboard, "st", Shell())
    for name in (
        "_ifvg_workspace",
        "_ml_workspace",
        "_compatibility_workspace",
        "_strategy_workspace",
        "_developer_workspace",
    ):
        monkeypatch.setattr(dashboard, name, lambda name=name: called.append(name))
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    dashboard.main()
    assert called == ["_ml_workspace"]
    assert [page.title for page in pages] == [
        "IFVG Lab",
        "ML Training",
        "Dashboard Compatibility",
        "Strategy Analysis",
    ]
    pages.clear()
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", True)
    dashboard.main()
    assert pages[-1].title == "Developer"
    assert called == ["_ml_workspace", "_ml_workspace"]


def test_unresolved_exact_candidate_link_cannot_open_another_case(monkeypatch):
    import ifvg_lab_tab as lab
    import ifvg_verifier_tab as verifier
    from streamlit.testing.v1 import AppTest

    from alpha_lab.agents.data_infra.ifvg import replay_chart_provider as provider
    from tests.agents.test_ifvg_setup_verifier_tab import _entry

    monkeypatch.setattr(lab, "_load_selected_pair", lambda *a, **k: (object(), _entry()))
    monkeypatch.setattr(verifier, "read_replay_chart_catalog", lambda *_: {"chart": {}})
    monkeypatch.setattr(verifier, "find_replay_artifact", lambda *_: "chart")
    monkeypatch.setattr(verifier, "_cached_replay_context", lambda *_: object())
    monkeypatch.setattr(
        verifier,
        "_cached_candidate_frame",
        lambda *_: pd.DataFrame({"candidate_id": ["unrelated"]}),
    )

    def missing(*args, **kwargs):
        raise ValueError("Exact candidate not present")

    monkeypatch.setattr(provider, "resolve_selection", missing)

    def app():
        import ifvg_lab_tab
        import streamlit as st

        st.session_state["selection"] = ifvg_lab_tab.render_ifvg_replay_tab(st)

    at = AppTest.from_function(app)
    at.session_state[verifier._PENDING_JUMP_KEY] = ("candidate_id", "missing-exact-case")
    for _ in range(2):
        at.run()
        assert not at.exception
        assert at.session_state["selection"].kind == "unavailable"
        assert not at.selectbox
        assert at.session_state[verifier._PENDING_JUMP_KEY][1] == "missing-exact-case"
    monkeypatch.setattr(verifier, "_cached_candidate_frame", lambda *_: pd.DataFrame())
    at.button(key="ifvg_context_v1_clear_candidate_jump").click().run()
    assert not at.exception
    assert at.session_state["selection"].kind == "empty"
    assert verifier._PENDING_JUMP_KEY not in at.session_state


def test_candidate_filters_do_not_restore_a_previous_unrelated_selection():
    from streamlit.testing.v1 import AppTest

    def app():
        import ifvg_verifier_tab
        import pandas as pd
        import streamlit as st

        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "previous-case",
                    "entry_session": "ny",
                    "trading_day": "2026-01-06",
                    "is_warmup": False,
                    "m3_qualifying": False,
                    "executed": True,
                    "blocked": False,
                    "censored": False,
                    "resolution": "target",
                }
            ]
        )
        st.session_state["filtered"] = ifvg_verifier_tab._filtered_candidates(st, frame)

    at = AppTest.from_function(app)
    at.session_state["ifvg_context_v1_candidate"] = "previous-case"
    at.run()
    at.multiselect(key="ifvg_context_v1_verifier_outcome").set_value(["loss"]).run()
    assert not at.exception
    assert at.session_state["filtered"].empty


def _actions_app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace._manage_study(st, ifvg_workspace._TEST_STUDY, ifvg_workspace._TEST_ROOTS)


def test_draft_rename_archive_restore_and_clone_keep_scientific_settings(monkeypatch, roots):
    from dataclasses import replace

    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary
    from alpha_lab.agents.data_infra.ifvg.study_drafts import list_drafts, load_draft

    draft = new_draft("single_configuration", display_name="Original study")
    draft.steps = {"baseline": {"baseline_profile_name": "exact-profile"}}
    save_draft(roots["draft_root"], draft)
    summary = StudySummary(
        key=draft.draft_id,
        kind="draft",
        name=draft.display_name,
        question="Research question",
        dates="Dates not selected",
        status="Draft",
        scope="research",
        draft=draft,
    )
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(ifvg_workspace, "_TEST_STUDY", summary, raising=False)
    at = AppTest.from_function(_actions_app).run()
    at.text_input[0].set_value("Renamed study")
    next(button for button in at.button if button.label == "Save name").click().run()
    assert not at.exception
    saved = load_draft(roots["draft_root"], draft.draft_id)
    assert saved.display_name == "Renamed study"
    assert saved.steps == draft.steps
    next(button for button in at.button if button.label == "Archive study").click().run()
    archived = load_draft(roots["draft_root"], draft.draft_id)
    assert archived.archived
    monkeypatch.setattr(
        ifvg_workspace, "_TEST_STUDY", replace(summary, archived=True, draft=archived)
    )
    at.run()
    next(button for button in at.button if button.label == "Restore study").click().run()
    assert not load_draft(roots["draft_root"], draft.draft_id).archived
    at.button(key="ifvg_workspace_clone").click().run()
    assert not at.exception
    drafts = list_drafts(roots["draft_root"])
    assert len(drafts) == 2
    clone = next(row for row in drafts if row.draft_id != draft.draft_id)
    assert clone.steps == saved.steps
    assert clone.never_frozen and not clone.archived


def test_running_study_only_requests_safe_cancellation_on_click(monkeypatch, roots):
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary

    identity = "a" * 64
    summary = StudySummary(
        key=identity,
        kind="search",
        name="Running study",
        question="Question",
        dates="Dates",
        status="Running",
        scope="research",
        state={"phase": "replays", "children": [{"state": "running"}]},
    )
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(ifvg_workspace, "_TEST_STUDY", summary, raising=False)

    def app():
        import ifvg_workspace
        import streamlit as st

        ifvg_workspace._progress(st, ifvg_workspace._TEST_STUDY, ifvg_workspace._TEST_ROOTS)

    at = AppTest.from_function(app).run()
    sentinel = roots["state_root"] / identity / "cancel.requested"
    assert not at.exception and not sentinel.exists()
    at.button(key="ifvg_workspace_cancel").click().run()
    assert not at.exception and sentinel.exists()
    assert not any(button.label == "Run" for button in at.button)


def test_pipeline_result_adapter_verifies_persisted_evidence(tmp_path):
    from dataclasses import replace

    from ifvg_research_results import load_pipeline_bundle

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import StudySummary
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import read_pipeline_state, run_pipeline
    from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

    fixture = build_pipeline_fixture(tmp_path)
    save_charter(fixture["store_root"], fixture["charter"])
    run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    identity = fixture["semantic"].pipeline_semantic_id
    state = read_pipeline_state(fixture["state_root"], identity)
    study = StudySummary(
        key=identity,
        kind="pipeline",
        name="Workflow",
        question="Question",
        dates="Dates",
        status="Completed",
        scope="verification",
        store_root=fixture["store_root"],
        state=state,
        charter_id=fixture["charter"].search_id,
    )
    bundle = load_pipeline_bundle(study)
    assert bundle["children"]
    assert bundle["metrics_by_child"]
    missing = {**state, "publication": {}}
    incomplete = load_pipeline_bundle(replace(study, state=missing))
    assert incomplete["frontier"] is None
    assert incomplete["limitations"]
