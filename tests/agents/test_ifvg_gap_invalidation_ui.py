"""Actual IFSM form and exact authoring/worker contracts for larger-gap validity.

AppTest is headless application-driven verification, never a browser screenshot.
All saved drafts here are disposable; synthetic enumeration never runs market data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import (
    export_fixed_configuration,
    fixed_axis_values,
    import_fixed_configuration,
)
from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import describe_strategy
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    AxisAuthorizationError,
    assert_axes_authorized,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    SearchCharterEnvelope,
    SearchCharterPayload,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children
from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft, save_draft
from tests.agents.test_ifsm_replication_ui import (
    BASELINE,
    ROOT,
    _draft,
    _fixed_widget,
    _open,
    _roots,
    _save,
    apptest,
    wizard,
)

FIELD = "htf_gap_invalidation_policy"
ORIGINAL = "execution_wick_full_fill_v1"
OWN_CLOSE = "own_timeframe_close_v1"
LABEL = "One-hour / four-hour gap invalidation"
CHOICES = [
    "One-minute wick reaches the far edge (original rule)",
    "Candle closes beyond the gap on its own timeframe",
]

pytestmark = [
    pytest.mark.usefixtures("developer_presentation"),
    pytest.mark.skipif(FIELD not in SEARCH_AXIS_REGISTRY_V1, reason="Requires supporting Core"),
]


def _vid(policy):
    return f"{FIELD}.{policy}"


def _worker_children(draft, roots):
    fields = wizard._charter_fields(draft, roots)
    fields["owner_authorization"] = wizard.SyntheticAuthorizationMarker()
    charter = SearchCharterEnvelope.from_payload(SearchCharterPayload(**fields))
    loaded = SearchCharterEnvelope.model_validate_json(charter.model_dump_json())
    children = enumerate_children(
        loaded, identity_resolver=lambda spec: spec.resolved_section_config_hash,
    )
    return fields, children


@pytest.mark.parametrize("policy", [ORIGINAL, OWN_CLOSE])
def test_fixed_policy_saved_reloaded_worker_loaded_one_profile(monkeypatch, tmp_path, policy):
    roots = _roots(monkeypatch, tmp_path)
    draft = _draft()
    app = _open(draft, roots)
    widget = _fixed_widget(app, FIELD)
    assert widget.label == LABEL
    assert widget.options == CHOICES
    widget.set_value(_vid(policy)).run()
    if policy == OWN_CLOSE:
        assert "A four-hour gap waits for a completed four-hour candle." in (
            _fixed_widget(app, FIELD).proto.help
        )
    else:
        assert "original one-minute wick/full-fill rule" in _fixed_widget(app, FIELD).proto.help
    stored = _save(app, draft, roots)
    # repair R7: choosing the rule already shown (the inherited original rule) is not an
    # edit, so nothing is written; the missing field resolves to the original rule
    expected = {FIELD: _vid(policy)} if policy == OWN_CLOSE else {}
    assert fixed_axis_values(stored) == expected
    reopened = _open(stored, roots)
    assert _fixed_widget(reopened, FIELD).value == _vid(policy)
    fields, children = _worker_children(stored, roots)
    assert fields["max_child_count"] == 1
    assert fields["axes"] == {key: (value,) for key, value in expected.items()}
    assert len(children) == 1
    child = resolve_profile_config({
        "profile_name": BASELINE, "section_overrides": children[0].section_overrides,
    })
    assert child.section.htf_gap_invalidation_policy == policy
    base = resolve_profile_config({"profile_name": BASELINE}).effective_config
    actual = child.effective_config
    excluded = {FIELD, "qualification_mode"}
    assert {key: value for key, value in actual.items() if key not in excluded} == {
        key: value for key, value in base.items() if key not in excluded
    }
    assert not roots["research"].exists()


def test_actual_ifsm_page_save_reopen_edit_and_import(monkeypatch, tmp_path):
    import run_ifsm_research_ui as launcher
    import strategy_core

    from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode

    monkeypatch.setattr(launcher, "CORE", Path(strategy_core.__file__).resolve().parents[2])
    monkeypatch.setattr(launcher, "WORKSPACE", tmp_path)
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    # main() intentionally uses actual root wiring, but each location is temporary.
    import ifvg_study_tab as study_tab
    for attribute in (
        "STORE_ROOT_RESEARCH", "STORE_ROOT_VERIFICATION", "STATE_ROOT", "DRAFT_ROOT",
        "VERIFICATION_CENTER_ROOT",
    ):
        monkeypatch.setattr(study_tab, attribute, tmp_path / attribute)
    app = apptest.AppTest.from_file(str(ROOT / "scripts/ifsm_research_ui.py"), default_timeout=120)
    app.run()
    assert not app.exception
    app.button(key="ifvg_workspace_new").click().run()
    next(widget for widget in app.radio
         if widget.label == "What would you like to research?").set_value("Evaluate").run()
    next(button for button in app.button if button.label == "Configure study").click().run()
    next(button for button in app.button if button.label == "Next").click().run()
    assert not app.exception
    labels = [widget.label for widget in app.selectbox]
    assert labels.index(LABEL) == labels.index("HTF selection cap per timeframe") + 1
    assert _fixed_widget(app, FIELD).options == CHOICES
    _fixed_widget(app, FIELD).set_value(_vid(OWN_CLOSE)).run()
    next(button for button in app.button if button.label == "Save draft").click().run()
    draft_id = app.session_state[f"{wizard.STATE_PREFIX}draft_id"]
    stored = load_draft(tmp_path / "drafts", draft_id)
    # repair R7: a new Evaluate study starts on S0_D80_W1_P1's saved settings
    from tests.agents.test_ifsm_replication_ui import _named_start

    start = _named_start()[0]
    assert fixed_axis_values(stored) == {**start, FIELD: _vid(OWN_CLOSE)}
    exported = export_fixed_configuration(BASELINE, fixed_axis_values(stored))

    reopened = apptest.AppTest.from_file(
        str(ROOT / "scripts/ifsm_research_ui.py"), default_timeout=120,
    )
    reopened.session_state["ifvg_workspace_screen"] = "new"
    reopened.session_state[f"{wizard.STATE_PREFIX}draft_id"] = draft_id
    reopened.run()
    assert not reopened.exception
    assert _fixed_widget(reopened, FIELD).value == _vid(OWN_CLOSE)
    _fixed_widget(reopened, FIELD).set_value(_vid(ORIGINAL)).run()
    next(button for button in reopened.button if button.label == "Save draft").click().run()
    assert fixed_axis_values(load_draft(tmp_path / "drafts", draft_id)) == {
        **start, FIELD: _vid(ORIGINAL)}
    next(widget for widget in reopened.text_area
         if widget.label == "Paste exported configuration JSON").set_value(exported).run()
    next(button for button in reopened.button
         if button.label == "Apply imported configuration").click().run()
    assert not reopened.exception
    assert _fixed_widget(reopened, FIELD).value == _vid(OWN_CLOSE)
    next(button for button in reopened.button if button.label == "Save draft").click().run()
    assert fixed_axis_values(load_draft(tmp_path / "drafts", draft_id)) == {
        **start, FIELD: _vid(OWN_CLOSE)}
    assert not (tmp_path / "search/v1").exists()


def test_reopening_saved_settings_without_the_rule_never_adds_it(monkeypatch, tmp_path):
    """Repair R7: a saved configuration like drafts f3099b0b… and 708c133a… (fixed
    settings without the gap rule, saved before it existed) reopens byte-identical;
    an owner edit still saves the configuration's rule explicitly."""

    from tests.agents.test_ifsm_replication_ui import _wizard_app

    roots = _roots(monkeypatch, tmp_path)
    fixed = {"opposing_timeout_1m_bars": "opposing_timeout_1m_bars.90",
             "parent_htf_distance_ticks_max": "parent_htf_distance_ticks_max.160",
             "parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.240"}
    draft = _draft(fixed=fixed)
    path = save_draft(roots["drafts"], draft)
    before = path.read_bytes()
    app = apptest.AppTest.from_function(_wizard_app, default_timeout=120)
    app.session_state[f"{wizard.STATE_PREFIX}draft_id"] = draft.draft_id
    app.run()
    assert not app.exception
    assert _fixed_widget(app, FIELD).value == _vid(ORIGINAL)  # shown, not written
    app.run()  # refresh
    assert path.read_bytes() == before
    axis = "htf_selection_max_per_timeframe"
    _fixed_widget(app, axis).set_value(f"{axis}.2").run()
    stored = _save(app, draft, roots)
    assert fixed_axis_values(stored) == {**fixed, f"{axis}": f"{axis}.2", FIELD: _vid(ORIGINAL)}


def test_legacy_missing_field_retains_original_without_rewriting_saved_draft(tmp_path):
    draft = _draft()
    save_draft(tmp_path, draft)
    path = tmp_path / draft.draft_id / "draft.json"
    before = path.read_bytes()
    loaded = load_draft(tmp_path, draft.draft_id)
    assert fixed_axis_values(loaded) == {}
    _, children = _worker_children(loaded, {"store_root": tmp_path / "store"})
    section = resolve_profile_config({
        "profile_name": BASELINE, "section_overrides": children[0].section_overrides,
    }).section
    assert section.htf_gap_invalidation_policy == ORIGINAL
    assert path.read_bytes() == before


@pytest.mark.parametrize("policy", [ORIGINAL, OWN_CLOSE])
def test_export_import_preserves_exact_configuration_without_approval(policy):
    selected = {
        FIELD: _vid(policy),
        "htf_selection_max_per_timeframe": "htf_selection_max_per_timeframe.2",
    }
    exported = export_fixed_configuration(BASELINE, selected)
    assert import_fixed_configuration(exported, BASELINE) == selected
    assert json.loads(exported)["effective_section"][FIELD] == policy
    assert "owner_authorization" not in json.loads(exported)
    assert AXIS_VALUE_REGISTRY_V1[_vid(policy)].owner_ratification_status == "pending"
    with pytest.raises(AxisAuthorizationError, match="ratification"):
        assert_axes_authorized({FIELD: _vid(policy)})


@pytest.mark.parametrize("change", ["policy", "extra_authority", "baseline_hash"])
def test_import_rejects_drift_or_unrecognized_authority(change):
    exported = json.loads(export_fixed_configuration(BASELINE, {FIELD: _vid(OWN_CLOSE)}))
    if change == "policy":
        exported["effective_section"][FIELD] = ORIGINAL
    elif change == "extra_authority":
        exported["owner_authorization"] = {"approved": True}
    else:
        exported["baseline_section_config_hash"] = "0" * 64
    with pytest.raises(ValueError):
        import_fixed_configuration(json.dumps(exported), BASELINE)


def test_policy_is_visible_in_rules_and_has_distinct_effective_identity(tmp_path):
    identities = []
    for policy in (ORIGINAL, OWN_CLOSE):
        draft = _draft(fixed={FIELD: _vid(policy)})
        _, children = _worker_children(draft, {"store_root": tmp_path / "store"})
        identities.append(children[0].resolved_section_config_hash)
        section = resolve_profile_config({"section_overrides": {FIELD: policy}}).effective_config
        description = describe_strategy(section)
        assert description.status == "available"
        text = " ".join(description.detail_bullets)
        if policy == OWN_CLOSE:
            assert "strictly below its bottom edge" in text
            assert "strictly above its top edge" in text
            assert "A wick-traversed one-hour or four-hour starting gap remains available" in text
            assert "protective stops are unchanged" in text
        else:
            assert "one-minute wick completely fills" in text
    assert identities[0] != identities[1]


def test_advanced_comparison_enumerates_only_two_requested_policies(tmp_path):
    draft = _draft(mode="fsm_config_search", question="find_robust_fsm")
    draft.steps["search_space"]["axis_selections"] = {FIELD: [_vid(OWN_CLOSE)]}
    fields, children = _worker_children(draft, {"store_root": tmp_path / "store"})
    assert fields["axes"] == {FIELD: (_vid(ORIGINAL), _vid(OWN_CLOSE))}
    assert len(children) == 2
    assert {child.section_overrides[FIELD] for child in children} == {ORIGINAL, OWN_CLOSE}


def test_session_summary_uses_chicago_am_pm_and_explicit_dst_seasons():
    from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import _chicago_session_clock

    assert _chicago_session_clock("08:00", "America/New_York") == "7:00 AM"
    assert _chicago_session_clock("18:00", "America/New_York") == "5:00 PM"
    assert _chicago_session_clock("12:00", "UTC") == "6:00 AM in winter / 7:00 AM in summer"


@pytest.mark.parametrize("policy", [ORIGINAL, OWN_CLOSE])
def test_normal_approval_review_accepts_exact_policy_without_granting_authority(tmp_path, policy):
    from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
    from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service

    draft = _draft(fixed={FIELD: _vid(policy)})
    draft.steps["validation"]["warmup_dates"] = list(FROZEN_WARMUP_DATES)
    # two evaluated dates: keep the independent-day threshold reachable (R5)
    draft.steps["benchmarks"] = {"strategy_gates": {"min_independent_days": 2}}
    fields = wizard._charter_fields(draft, {"store_root": tmp_path / "store"})
    requirements = wizard._requirement_set_for_draft(draft, "full_authorized_development")
    intent = service._typed_intent(fields)
    rows, _, dates = service._validate_request(intent, requirements)
    assert len(rows) == 1
    assert rows[0]["section_overrides"] == {FIELD: policy}
    assert tuple(dates[:len(FROZEN_WARMUP_DATES)]) == FROZEN_WARMUP_DATES
    assert AXIS_VALUE_REGISTRY_V1[_vid(policy)].owner_ratification_status == "pending"
    assert not (tmp_path / "store").exists()
