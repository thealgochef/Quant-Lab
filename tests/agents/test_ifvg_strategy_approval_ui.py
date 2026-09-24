"""Normal Configure study approval workflow; disposable stores and no replays."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (  # noqa: E402
    initialize_store_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import STORE  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.study_drafts import (  # noqa: E402
    load_draft,
    save_draft,
)


def _app():
    import ifvg_research_wizard
    import ifvg_strategy_approval
    import streamlit as st

    ifvg_research_wizard.render_new_study(st, roots=ifvg_strategy_approval._TEST_ROOTS)


def _button(at, label):
    return next(widget for widget in at.button if widget.label == label)


def _reviewer(at):
    assert any(widget.label == "Reviewer name" for widget in at.text_input), _text(at)
    return next(widget for widget in at.text_input if widget.label == "Reviewer name")


def _confirmation(at):
    assert any(widget.label == "I approve this exact strategy study" for widget in at.checkbox), (
        _text(at)
    )
    return next(
        widget for widget in at.checkbox if widget.label == "I approve this exact strategy study"
    )


def _text(at):
    text = "\n".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "info", "error", "success")
        for element in at.get(kind)
    )
    return text + "\n" + "\n".join(element.value.to_json() for element in at.dataframe)


def _approval_paths(roots):
    return list((roots["store_root"] / STORE).glob("*/envelope.json"))


@pytest.fixture
def configured_study(monkeypatch, tmp_path):
    import ifvg_research_wizard
    import ifvg_strategy_approval
    import ifvg_study_wizard as wizard
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash
    from streamlit.testing.v1 import AppTest

    from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service

    roots = {
        "store_root": tmp_path / "research",
        "store_roots": {"research": tmp_path / "research", "test": tmp_path / "test"},
        "draft_root": tmp_path / "drafts",
        "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "repo_root": tmp_path,
    }
    initialize_store_namespace(
        roots["store_root"], namespace_class="research", store_instance_id="c" * 32
    )
    monkeypatch.setattr(ifvg_strategy_approval, "_TEST_ROOTS", roots, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Reviewing or saving study approval must never start a study")

    monkeypatch.setattr(wizard, "_spawn_search_job", forbidden)
    monkeypatch.setattr(wizard, "_freeze_and_launch", forbidden)

    def fixture_cache_metadata(store_root, configs, dates):
        assert Path(store_root) == roots["store_root"]
        assert configs
        return tuple(dates), len(dates), [{"fixture_dates": list(dates)}]

    monkeypatch.setattr(service, "_cache_review", fixture_cache_metadata)
    draft = start_draft_from_card(
        next(card for card in TASK_CARDS if card.card_id == "fsm_search"), roots
    )
    draft.display_name = "Parent staleness with opposing timeout comparison"
    profile = "ifvg_v2_doc_default_fresh_static_1r"
    section = wizard.canonicalize_section(
        wizard.resolve_profile_config({"profile_name": profile}).section
    )
    draft.steps["baseline"] = {
        "baseline_profile_name": profile,
        "baseline_section_config_hash": ifvg_profile_hash(section),
        "baseline_blocked_reason": None,
    }
    draft.steps["search_space"] = {
        "mode_id": draft.mode_id,
        "axis_selections": {
            "parent_retest_timeout_1m_bars": [
                f"parent_retest_timeout_1m_bars.{value}" for value in (240, 360, 480)
            ],
            "opposing_timeout_1m_bars": ["opposing_timeout_1m_bars.90"],
        },
        "interpretation": wizard.INTERPRETATIONS[2],
    }
    draft.steps["validation"].update(
        {"real_dates": ["2026-01-13", "2026-01-14"], "seed": 7, "worker_limit": 1}
    )
    # Two (or, after the date-change case, one) evaluated dates: keep the
    # independent-day threshold reachable so approval is the step under test.
    draft.steps["benchmarks"] = {"strategy_gates": {"min_independent_days": 1}}
    draft.steps["review"] = {"n_children": 8}
    draft.current_step_key = "review"
    save_draft(roots["draft_root"], draft)

    def render():
        at = AppTest.from_function(_app, default_timeout=60)
        at.session_state[ifvg_research_wizard._DRAFT] = draft.draft_id
        at.run()
        assert not at.exception
        assert at.header[0].value == "Configure study"
        return at

    return roots, draft, render


def test_review_requires_explicit_name_and_confirmation_without_writing(configured_study):
    roots, _draft, render = configured_study
    original_files = set(roots["store_root"].rglob("*"))
    at = render()
    assert not _confirmation(at).value
    assert not _reviewer(at).value
    assert _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert "240" in _text(at) and "90" in _text(at)
    assert "2026" in _text(at)
    assert "Running is unavailable until the required authorization" not in _text(at)
    assert not _approval_paths(roots)
    assert set(roots["store_root"].rglob("*")) == original_files

    _confirmation(at).check().run()
    assert _button(at, "Save study approval").disabled
    _confirmation(at).uncheck().run()
    _reviewer(at).set_value("Test owner").run()
    assert _button(at, "Save study approval").disabled
    _confirmation(at).check().run()
    assert not _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert not _approval_paths(roots)
    assert not roots["state_root"].exists()


def test_saving_exact_approval_enables_run_without_launch(configured_study):
    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import runner_entry_key_for_charter

    roots, draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    _button(at, "Save study approval").click().run()
    assert not at.exception
    assert len(_approval_paths(roots)) == 1
    assert not _button(at, "Run study").disabled
    assert wizard._resolve_for_draft(draft, roots).readiness.status == "ready"
    charter = SearchCharterEnvelope.from_payload(wizard._assemble_charter(draft, roots))
    assert runner_entry_key_for_charter(charter) == "search_strategy_development_v1"
    assert not roots["state_root"].exists()
    assert not (roots["store_root"] / "charters").exists()
    assert not (roots["store_root"] / "core_replays").exists()


def test_single_configuration_approval_renders_and_saves_only_one_child(configured_study):
    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children

    roots, draft, render = configured_study
    draft.mode_id = "single_configuration"
    draft.steps["objective"]["mode_id"] = "single_configuration"
    draft.steps["objective"]["question_id"] = "evaluate_one_configuration"
    draft.steps["baseline"]["fixed_axis_value_ids"] = {
        "parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.240",
        "opposing_timeout_1m_bars": "opposing_timeout_1m_bars.90",
    }
    draft.steps["search_space"] = {}
    draft.steps["review"] = {"n_children": 1}
    save_draft(roots["draft_root"], draft)
    at = render()
    assert _button(at, "Run study").disabled
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    _button(at, "Save study approval").click().run()
    assert not at.exception
    assert not _button(at, "Run study").disabled
    charter = SearchCharterEnvelope.from_payload(wizard._assemble_charter(draft, roots))
    assert charter.payload.search_mode == "single_configuration"
    assert dict(charter.payload.axes) == {
        "parent_retest_timeout_1m_bars": ("parent_retest_timeout_1m_bars.240",),
        "opposing_timeout_1m_bars": ("opposing_timeout_1m_bars.90",),
    }
    children = enumerate_children(
        charter,
        store_root=roots["store_root"],
        identity_resolver=lambda spec: spec.resolved_section_config_hash,
    )
    assert len(children) == 1
    assert not roots["state_root"].exists()
    assert not (roots["store_root"] / "core_replays").exists()


def test_ready_legacy_owner_decisions_still_require_exact_study_approval(
    configured_study, monkeypatch
):
    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg import study_providers
    from alpha_lab.agents.data_infra.ifvg.search.authorization import OwnerDecisionEvidenceRef
    from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import DECISION_KEYS

    roots, draft, render = configured_study
    legacy_ref = OwnerDecisionEvidenceRef(
        decision_id="25+28+29+30:regime_feature_eligibility",
        decision_artifact_id="d" * 64,
        content_hash="d" * 64,
        author="Historical test owner",
        approved_at="2026-09-08T00:00:00+00:00",
        effective_from="2026-09-08T00:00:00+00:00",
        reviewed_evidence_refs=("historical-test-evidence",),
    )
    # Model a legacy evidence reader covering the same decision keys, while
    # retaining the real namespace, readiness and bundle assembly behavior.
    monkeypatch.setattr(
        study_providers,
        "_owner_decision_evidence",
        lambda *args, **kwargs: {
            key: (legacy_ref.decision_artifact_id, legacy_ref) for key in DECISION_KEYS
        },
    )
    resolution = wizard._resolve_for_draft(draft, roots)
    assert resolution.readiness.status == "ready"
    assert resolution.resolved.freeze_allowed

    at = render()
    assert not _confirmation(at).value
    assert _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert "This exact study is approved" not in _text(at)
    assert not _approval_paths(roots)
    assert not roots["state_root"].exists()


def test_saved_approval_does_not_unlock_a_changed_study(configured_study):
    import ifvg_study_wizard as wizard

    roots, draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    _button(at, "Save study approval").click().run()
    approval_path = _approval_paths(roots)[0]
    original_approval = approval_path.read_bytes()

    saved = load_draft(roots["draft_root"], draft.draft_id)
    saved.steps["validation"]["seed"] = 19
    save_draft(roots["draft_root"], saved)
    at.run()
    assert not at.exception
    assert not _confirmation(at).value
    assert _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert wizard._resolve_for_draft(saved, roots).readiness.status == "missing"
    assert _approval_paths(roots) == [approval_path]
    assert approval_path.read_bytes() == original_approval


def test_missing_prepared_data_blocks_run_with_a_specific_explanation(
    configured_study, monkeypatch
):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service

    roots, _draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    _button(at, "Save study approval").click().run()
    assert not _button(at, "Run study").disabled

    def missing_metadata(*args, **kwargs):
        raise ValueError("Prepared bars or levels are missing for 2026-01-13; prepare data first.")

    monkeypatch.setattr(service, "_cache_review", missing_metadata)
    at.run()
    assert not at.exception
    assert "Prepared bars or levels are missing for 2026-01-13" in _text(at)
    assert _button(at, "Run study").disabled
    assert all(widget.label != "Save study approval" for widget in at.button)
    assert len(_approval_paths(roots)) == 1
    assert not roots["state_root"].exists()


@pytest.mark.parametrize("change", ["axes", "dates", "seed"])
def test_changed_exact_settings_require_new_unchecked_confirmation(configured_study, change):
    roots, draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    old_key = _confirmation(at).key
    saved = load_draft(roots["draft_root"], draft.draft_id)
    if change == "axes":
        saved.steps["search_space"]["axis_selections"]["opposing_timeout_1m_bars"] = [
            "opposing_timeout_1m_bars.120"
        ]
    elif change == "dates":
        saved.steps["validation"]["real_dates"] = ["2026-01-13"]
    else:
        saved.steps["validation"]["seed"] = 19
    save_draft(roots["draft_root"], saved)
    at.run()
    assert not at.exception
    assert _confirmation(at).key != old_key
    assert not _confirmation(at).value
    assert _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert not _approval_paths(roots)


def test_changed_reviewed_metadata_requires_new_confirmation(configured_study, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service

    roots, _draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()
    old_key = _confirmation(at).key

    def replacement_metadata(store_root, configs, dates):
        assert Path(store_root) == roots["store_root"]
        return tuple(dates), len(dates), [{"fixture_dates": list(dates), "revision": 2}]

    monkeypatch.setattr(service, "_cache_review", replacement_metadata)
    at.run()
    assert not at.exception
    assert _confirmation(at).key != old_key
    assert not _confirmation(at).value
    assert _button(at, "Save study approval").disabled
    assert _button(at, "Run study").disabled
    assert not _approval_paths(roots)


def test_approval_failure_explains_the_actionable_reason(configured_study, monkeypatch):
    import ifvg_strategy_approval as approval_ui

    roots, _draft, render = configured_study
    at = render()
    _reviewer(at).set_value("Test owner").run()
    _confirmation(at).check().run()

    def changed_inputs(*args, **kwargs):
        raise PermissionError("Study settings changed after preview; review the updated settings.")

    monkeypatch.setattr(approval_ui, "record_strategy_approval", changed_inputs)
    _button(at, "Save study approval").click().run()
    assert not at.exception
    assert "Study settings changed after preview" in _text(at)
    assert _button(at, "Run study").disabled
    assert not _approval_paths(roots)
