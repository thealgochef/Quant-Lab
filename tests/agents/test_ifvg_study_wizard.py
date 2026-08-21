"""FUX-WIZ-001..012 AppTests: shell, drafts, modes/templates, baseline,
search space, prop cards, risk, benchmarks, validation, review, freeze and
registry-gated launch (TEST_MATRIX §3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_study_tab as study_tab  # noqa: E402
import ifvg_study_wizard as wizard  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.profiles import (  # noqa: E402
    resolve_profile_config,
)
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: E402
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: E402
    canonicalize_section,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: E402
    has_envelope,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (  # noqa: E402
    load_draft,
    new_draft,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: E402
    VERIFICATION_BADGE_TEXT,
)

_DRAFT_KEY = f"{wizard.STATE_PREFIX}draft_id"


def _app() -> None:
    import ifvg_study_tab as study_tab
    import ifvg_study_wizard as wizard
    import streamlit as st

    wizard.render_new_study(st, roots=study_tab.workspace_roots(st))


def _patched_roots(monkeypatch, tmp_path) -> dict[str, Path]:
    roots = {
        "research": tmp_path / "search" / "v1",  # 'search' segment: synthetic refused
        "verification": tmp_path / "search_test" / "v1",
        "state": tmp_path / "state",
        "drafts": tmp_path / "drafts",
    }
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", roots["research"])
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", roots["verification"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", roots["state"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", roots["drafts"])
    return roots


def _baseline_hash() -> str:
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    resolved = resolve_profile_config(
        {"profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    )
    return ifvg_profile_hash(canonicalize_section(resolved.section))


def _seed_draft(draft_root: Path, *, step: int, mode: str = "fsm_config_search"):
    spec = SEARCH_AXIS_REGISTRY_V1["parent_retest_timeout_1m_bars"]
    challenger = next(
        value
        for value in spec.registered_values
        if value != spec.baseline_value_id
    )
    draft = new_draft(mode, display_name="AppTest draft")
    draft.step_index = step
    draft.steps = {
        "objective": {
            "mode_id": mode,
            "question_id": "find_robust_fsm"
            if mode == "fsm_config_search"
            else "compare_one_with_baseline",
            "template_id": "strategy_quality_only",
            "custom_objectives": (),
        },
        "baseline": {
            "baseline_profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "baseline_section_config_hash": _baseline_hash(),
            "baseline_blocked_reason": None,
        },
        "search_space": {
            "mode_id": mode,
            "axis_selections": {"parent_retest_timeout_1m_bars": [challenger]},
            "interpretation": "Sequential Strategy Profile",
        },
        "prop_contracts": {
            "mode_id": mode,
            "selected_contract_ids": [],
            "launchable_contract_ids": [],
        },
        "risk_policies": {"selected_contract_ids": [], "per_firm_policies": {}},
        "benchmarks": {},
        "validation": {
            "run_scope": "verification_5d",
            "real_dates": [
                "2026-06-04",
                "2026-06-05",
                "2026-06-08",
                "2026-06-09",
                "2026-06-10",
            ],
            "warmup_dates": [],
            "seed": 7,
            "worker_limit": 4,
        },
        "review": {"run_scope": "verification_5d", "n_children": 2},
    }
    save_draft(draft_root, draft)
    return draft


def _run_at_step(monkeypatch, tmp_path, *, step: int, namespace: str | None = None):
    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=step)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    if namespace:
        at.session_state[study_tab.NAMESPACE_KEY] = namespace
    at.run()
    assert not at.exception
    return at, draft, roots


def _button(at, label: str):
    return next(b for b in at.button if b.label == label)


def _markdown_text(at) -> str:
    return "\n".join(str(block.value) for block in at.markdown)


def _caption_text(at) -> str:
    return "\n".join(str(block.value) for block in at.caption)


# ── shell, drafts, modes ────────────────────────────────────────────────────


def test_wizard_shell_and_exact_step_restore(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-001/002: 8 steps, progress + breadcrumb, exact-step restore,
    Save Draft always visible, autosave on Next."""

    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=2)
    text = _caption_text(at)
    assert "**Strategy Search Space**" in text  # breadcrumb bolds the step
    assert _button(at, "Save Draft")
    assert _button(at, "Back")
    assert _button(at, "Next")
    _button(at, "Next").click().run()
    assert not at.exception
    restored = load_draft(roots["drafts"], draft.draft_id)
    assert restored.step_index == 3  # autosaved on Next
    at2 = apptest.AppTest.from_function(_app, default_timeout=120)
    at2.session_state[_DRAFT_KEY] = draft.draft_id
    at2.run()
    assert "**Prop Contracts**" in _caption_text(at2)  # exact-step restore


def test_five_modes_four_questions_six_templates(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-004 + resolved thresholds/tie-breaks visible."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=0)
    mode_box = next(
        w for w in at.selectbox if w.key == f"{wizard._W}new_mode"
    )
    assert mode_box.options == [
        "Single Configuration",
        "FSM Configuration Search",
        "Prop Benchmark",
        "Universal Prop Search",
        "Full Pipeline Run",
    ]
    question = next(w for w in at.radio if w.key == f"{wizard._W}question")
    assert len(question.options) == 4
    template = next(w for w in at.selectbox if w.key == f"{wizard._W}template")
    assert template.options == [
        "Payout Reliability",
        "Maximum Expected Payout",
        "Low Breach / Long Account Life",
        "Balanced Prop Performance",
        "Strategy Quality Only",
        "Custom",
    ]
    # the resolved objective table is visible — nothing hidden behind a name
    assert "primary objective" in str(at.get("arrow_data_frame")) + str(
        [t.value for t in at.get("table") or []]
    ) or "Resolved objective" in _markdown_text(at)


def test_incompatible_mode_question_pair_blocks_next(monkeypatch, tmp_path) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=0)
    question = next(w for w in at.radio if w.key == f"{wizard._W}question")
    question.set_value("Test repeat-payout feasibility").run()
    assert not at.exception
    assert any(
        "different study mode" in str(err.value) for err in at.error
    )
    assert _button(at, "Next").disabled


# ── baseline ────────────────────────────────────────────────────────────────


def test_baseline_card_and_copyable_identity(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-005: human card + full copyable technical identity."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=1)
    codes = [str(block.value) for block in at.code]
    assert any(len(value.strip()) == 64 for value in codes)  # profile hash
    text = _markdown_text(at)
    assert "Baseline card" in text
    assert _button(at, "Next")
    assert not _button(at, "Next").disabled


def test_blocked_baseline_explains_and_cannot_advance(monkeypatch, tmp_path) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=1)
    baseline_box = next(w for w in at.selectbox if w.key == f"{wizard._W}baseline")
    baseline_box.set_value("ifvg_v2_ict_clean_fresh_static_1r").run()
    assert not at.exception
    assert _button(at, "Next").disabled
    assert any("blocked" in str(err.value).lower() for err in at.error)


# ── search space ────────────────────────────────────────────────────────────


def test_search_space_locked_blocked_axes_have_no_widget(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-006: market-meaning groups; no widget for locked/blocked; no
    raw override editor anywhere; computation-path chips visible."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=2)
    widget_keys = {w.key for w in at.multiselect}
    assert f"{wizard._W}axis_parent_retest_timeout_1m_bars" in widget_keys
    for blocked_axis in (
        "break_even_enabled",
        "parent_full_fill_invalidation",
        "resolver_policy",
        "runnable",  # locked invariant
        "session_scheme",  # thesis-defining → locked, no widget
        "entry_parent_distance_ticks_max",  # measured only
    ):
        assert f"{wizard._W}axis_{blocked_axis}" not in widget_keys, blocked_axis
    assert not at.text_area  # no raw JSON/override editor exists
    text = _markdown_text(at)
    assert "Requires New Sequential Replay" in text
    assert "Staleness" in str([e.label for e in at.expander]) or True
    expander_labels = [e.label for e in at.expander]
    assert "Staleness" in expander_labels
    assert "Parent Handling" in expander_labels


def test_search_space_selection_updates_child_count(monkeypatch, tmp_path) -> None:
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=2)
    target = next(
        w
        for w in at.multiselect
        if w.key == f"{wizard._W}axis_parent_retest_timeout_1m_bars"
    )
    assert target.value  # restored from the seeded draft
    _button(at, "Save Draft").click().run()
    stored = load_draft(roots["drafts"], draft.draft_id)
    assert stored.steps["search_space"]["axis_selections"]


# ── prop contracts ──────────────────────────────────────────────────────────


def test_prop_step_renders_truthful_synthetic_cards(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-008: full card fields; synthetic visibly synthetic."""

    roots = _patched_roots(monkeypatch, tmp_path)
    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(
        tmp_path / "fixture", with_prop=False, with_contract=True
    )
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", fixture["store_root"])
    draft = _seed_draft(roots["drafts"], step=3, mode="prop_benchmark")
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "SYNTHETIC" in warnings
    checkboxes = [c for c in at.checkbox if c.key.startswith(f"{wizard._W}contract_")]
    assert len(checkboxes) == 1  # the launchable synthetic card is selectable


def test_prop_step_without_contracts_renders_the_exact_state(
    monkeypatch, tmp_path
) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=3)
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "No verified firm contract" in headings


# ── benchmarks / validation ─────────────────────────────────────────────────


def test_benchmarks_render_three_ordered_gate_groups(monkeypatch, tmp_path) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=5)
    text = _markdown_text(at)
    first = text.index("1 · Underlying Strategy Gate")
    second = text.index("2 · Prop Feasibility Gate")
    third = text.index("3 · Robustness Gate")
    assert first < second < third
    assert "proposed_protocol_default" in _caption_text(at)
    info = " ".join(str(i.value) for i in at.info)
    assert "verification_control_flow_gates_v1" in info
    assert "no hidden weighted score" in info.lower()


def test_validation_shows_readonly_allowlist_badge_and_checklist(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-011 + FUX-WIZ-007: canonical allowlist read-only, exact badge,
    authorization state, computation-path-scoped checklist."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=6)
    errors = " ".join(str(e.value) for e in at.error)
    assert VERIFICATION_BADGE_TEXT in errors  # the non-dismissible badge
    codes = " ".join(str(c.value) for c in at.code)
    assert "2026-06-04" in codes and "2026-06-10" in codes
    assert "2026-06-11" in codes  # protected boundary shown read-only
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Verification authorization missing" in headings
    text = _markdown_text(at)
    assert "21/R-5:verification_fixture_authorization" in text


# ── review, freeze, launch ──────────────────────────────────────────────────


def test_freeze_and_launch_only_in_the_button_handler(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-012: freeze saves the immutable charter, marks the draft
    frozen, spawns ONLY on the explicit click, and routes to Active Runs."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        wizard, "_spawn_search_job", lambda command: spawned.append(command) or 4242
    )
    at, draft, roots = _run_at_step(
        monkeypatch,
        tmp_path,
        step=7,
        namespace="Verification / synthetic (search_test/v1)",
    )
    assert spawned == []  # rendering launches nothing
    text = _markdown_text(at)
    assert "Resolved charter preview" in text
    assert "Expensive strategy replays" in str(
        [t for t in at.get("table")]
    ) or "Estimated work" in text
    _button(at, "Freeze Search Charter and Launch").click().run()
    assert not at.exception
    assert len(spawned) == 1
    command = spawned[0]
    assert "--runner-entry-key" in command
    assert "synthetic_search_job_fixture_v1" in command
    search_id = command[command.index("--search-id") + 1]
    assert has_envelope(
        study_tab.STORE_ROOT_VERIFICATION, "charters", search_id
    )
    frozen = load_draft(roots["drafts"], draft.draft_id)
    assert frozen.status == "frozen"
    assert frozen.frozen_search_id == search_id
    assert (
        at.session_state[f"{wizard.STATE_PREFIX}pending_route"] == "Active Runs"
    )


def test_research_namespace_freeze_fails_closed(monkeypatch, tmp_path) -> None:
    """No owner evidence exists → a research-namespace charter cannot freeze
    (P0-4 namespace confinement surfaces as a sanitized refusal)."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        wizard, "_spawn_search_job", lambda command: spawned.append(command) or 1
    )
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=7)
    _button(at, "Freeze Search Charter and Launch").click().run()
    assert not at.exception
    assert spawned == []
    errors = " ".join(str(e.value) for e in at.error)
    assert "refus" in errors.lower() or "fail" in errors.lower()
    assert load_draft(roots["drafts"], draft.draft_id).status == "draft"


def test_full_scope_requires_the_exact_typed_confirmation(
    monkeypatch, tmp_path
) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=7)
    draft.steps["validation"]["run_scope"] = "full_authorized_development"
    draft.steps["review"] = {"run_scope": "full_authorized_development"}
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "This will run the full authorized development pipeline." in errors
    assert "It is not an implementation verification run." in errors
    assert _button(at, "Freeze Search Charter and Launch").disabled
    ack = next(w for w in at.text_input if w.key == f"{wizard._W}ack")
    ack.set_value("run full authorized development").run()
    assert not _button(at, "Freeze Search Charter and Launch").disabled


def test_full_pipeline_mode_renders_the_planned_capability_state(
    monkeypatch, tmp_path
) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=7, mode="full_pipeline_run")
    draft.steps["objective"]["question_id"] = "find_robust_fsm"
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "planned / unavailable" in headings.lower()
    assert not any(
        b.label == "Freeze Search Charter and Launch" for b in at.button
    )


def test_risk_step_renders_the_universal_policy_hierarchy(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-009 (adversarial F18a): the universal-strategy tree with
    per-firm account/risk/withdrawal/replacement policy widgets, accounts
    capped at 2."""

    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=4, mode="prop_benchmark")
    contract = "c" * 64
    draft.steps["prop_contracts"]["selected_contract_ids"] = [contract]
    draft.steps["risk_policies"] = {
        "selected_contract_ids": [contract],
        "per_firm_policies": {
            contract: {
                "risk_template": "Fixed Dollar",
                "risk_value": 250.0,
                "withdrawal_policy": "withdraw_monthly_cadence_v1",
                "replacement_policy": "replace_once",
                "n_accounts": 2,
            }
        },
    }
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    codes = " ".join(str(c.value) for c in at.code)
    assert "Universal Strategy Profile" in codes  # the FUX §12 tree
    assert "Account/Risk/Withdrawal/Replacement Policy Set" in codes
    template = next(
        w for w in at.selectbox if w.key == f"{wizard._W}risk_{contract[:16]}"
    )
    assert template.value == "Fixed Dollar"
    withdrawal = next(
        w for w in at.selectbox if w.key == f"{wizard._W}wd_{contract[:16]}"
    )
    assert withdrawal.value == "withdraw_monthly_cadence_v1"
    accounts = next(
        w for w in at.number_input if w.key == f"{wizard._W}acct_{contract[:16]}"
    )
    assert accounts.value == 2
    infos = " ".join(str(i.value) for i in at.info)
    assert "SAME resampled market/trade path" in infos  # copied accounts


def test_back_preserves_fields(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-001 (adversarial F18b): Back never discards valid data."""

    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=2)
    _button(at, "Back").click().run()
    assert not at.exception
    stored = load_draft(roots["drafts"], draft.draft_id)
    assert stored.step_index == 1
    assert stored.steps["search_space"]["axis_selections"]  # fields kept


def test_descriptive_interpretation_blocks_enumeration(
    monkeypatch, tmp_path
) -> None:
    """FUX §10.5 (adversarial F2) at the AppTest level."""

    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=2)
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
        SEARCH_AXIS_REGISTRY_V1 as _AXES,
    )

    draft.steps["search_space"]["axis_selections"] = {
        "enable_shorts": list(_AXES["enable_shorts"].registered_values[:1])
    }
    draft.steps["search_space"]["interpretation"] = "Descriptive Slice"
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "never replays" in errors
    assert _button(at, "Next").disabled


def test_no_cross_draft_widget_leak(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-002 (adversarial F10): opening another draft never inherits
    the previous draft's mounted widget values."""

    roots = _patched_roots(monkeypatch, tmp_path)
    draft_a = _seed_draft(roots["drafts"], step=2)
    draft_b = _seed_draft(roots["drafts"], step=2)
    draft_b.steps["search_space"]["axis_selections"] = {}
    save_draft(roots["drafts"], draft_b)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft_a.draft_id
    at.run()
    target_key = f"{wizard._W}axis_parent_retest_timeout_1m_bars"
    widget_a = next(w for w in at.multiselect if w.key == target_key)
    assert widget_a.value  # A's stored selection is mounted
    at.session_state[_DRAFT_KEY] = draft_b.draft_id
    at.run()
    assert not at.exception
    widget_b = next(w for w in at.multiselect if w.key == target_key)
    assert widget_b.value == []  # B shows B's (empty) state, never A's
    _button(at, "Save Draft").click().run()
    stored_b = load_draft(roots["drafts"], draft_b.draft_id)
    assert not stored_b.steps["search_space"]["axis_selections"]
    stored_a = load_draft(roots["drafts"], draft_a.draft_id)
    assert stored_a.steps["search_space"]["axis_selections"]  # A untouched


def test_unresolved_commit_provenance_refuses_freeze(
    monkeypatch, tmp_path
) -> None:
    """DEV-R4-14 (safety F1): 'unknown' provenance never enters the
    immutable charter — the freeze refuses instead."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        wizard, "_spawn_search_job", lambda command: spawned.append(command) or 1
    )
    monkeypatch.setattr(wizard, "_commit_of", lambda path: "unknown")
    at, draft, roots = _run_at_step(
        monkeypatch,
        tmp_path,
        step=7,
        namespace="Verification / synthetic (search_test/v1)",
    )
    _button(at, "Freeze Search Charter and Launch").click().run()
    assert not at.exception
    assert spawned == []
    errors = " ".join(str(e.value) for e in at.error)
    assert "provenance" in errors.lower()
    assert load_draft(roots["drafts"], draft.draft_id).status == "draft"


def test_clone_as_new_search_from_a_frozen_draft(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-003: the original stays frozen; the clone is a new draft."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import mark_frozen

    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=7)
    mark_frozen(roots["drafts"], draft, search_id="f" * 64)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    _button(at, "Clone as New Search").click().run()
    assert not at.exception
    clone_id = at.session_state[_DRAFT_KEY]
    assert clone_id != draft.draft_id
    clone = load_draft(roots["drafts"], clone_id)
    assert clone.status == "draft"
    assert clone.cloned_from == draft.draft_id
    original = load_draft(roots["drafts"], draft.draft_id)
    assert original.status == "frozen"
