"""FUX-WIZ-001..012 AppTests as amended by UI-1 and UI-2: shell, drafts, modes /
questions / templates, baseline, search space, prop cards, risk, benchmarks,
validation, review, the purpose card, charter satisfiability, authorization
by the actual path, honest launch and the sequential-V1 runtime truth
(TEST_MATRIX §3.11; plan §10) — plus UI-2's goal-conditional flows (skipped
steps carry visible reasons), the prop objective that blocks instead of
disappearing, the session-only draft (no file until Save / the first valid
Next), the Saved / Unsaved chip with autosave, the required draft name, the
duplicate warning and the archived-draft state (owner Q2)."""

from __future__ import annotations

import json
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
    archive_draft,
    load_draft,
    new_draft,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: E402
    catalog_annotations,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: E402
    VERIFICATION_BADGE_TEXT,
)

_DRAFT_KEY = f"{wizard.STATE_PREFIX}draft_id"
_FREEZE = "Freeze Search Charter and Launch"


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


def _annotation(purpose: str) -> dict:
    return {
        "schema_version": 1,
        "purpose": purpose,
        "derivation": "card_selected",
        "owner_confirmed": True,
        "updated_at": "2026-09-04T00:00:00+00:00",
    }


def _seed_draft(
    draft_root: Path,
    *,
    step: int,
    mode: str = "fsm_config_search",
    purpose: str | None = "implementation_verification",
    run_scope: str | None = None,
    template: str = "strategy_quality_only",
):
    spec = SEARCH_AXIS_REGISTRY_V1["parent_retest_timeout_1m_bars"]
    challenger = next(
        value
        for value in spec.registered_values
        if value != spec.baseline_value_id
    )
    draft = new_draft(mode, display_name="AppTest draft")
    draft.step_index = step
    if purpose is not None:
        draft.purpose_annotation = _annotation(purpose)
    scope = run_scope or (
        "verification_5d"
        if purpose in (None, "implementation_verification")
        else "full_authorized_development"
    )
    draft.steps = {
        "objective": {
            "mode_id": mode,
            "question_id": "find_robust_fsm"
            if mode in ("fsm_config_search", "full_pipeline_run")
            else "compare_one_with_baseline",
            "template_id": template,
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
            "run_scope": scope,
            "evidence_class": "synthetic_fixture" if scope == "verification_5d" else "real",
            "real_dates": [
                "2026-06-04",
                "2026-06-05",
                "2026-06-08",
                "2026-06-09",
                "2026-06-10",
            ]
            if scope == "verification_5d"
            else ["2026-01-13", "2026-01-14"],
            "warmup_dates": [],
            "seed": 7,
            "worker_limit": 1,
        },
        "review": {"run_scope": scope, "n_children": 2},
    }
    save_draft(draft_root, draft)
    return draft


def _run_at_step(monkeypatch, tmp_path, *, step: int, **seed_kwargs):
    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=step, **seed_kwargs)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    return at, draft, roots


def _button(at, label: str):
    return next(b for b in at.button if b.label == label)


def _markdown_text(at) -> str:
    return "\n".join(str(block.value) for block in at.markdown)


def _caption_text(at) -> str:
    return "\n".join(str(block.value) for block in at.caption)


def _headings(at) -> str:
    return " ".join(str(h.value) for h in at.subheader)


def _errors(at) -> str:
    return " ".join(str(e.value) for e in at.error)


def _tables(at) -> str:
    return str([t.value.to_dict() for t in at.table])


def _state_writing_spawn(spawned: list[list[str]], *, pid: int = 4242):
    """A fake detached spawn that persists the worker's first checkpoint —
    the honest launch reports success only once this file exists."""

    def _spawn(command: list[str]) -> int:
        spawned.append(command)
        search_id = command[command.index("--search-id") + 1]
        state_root = Path(command[command.index("--state-root") + 1])
        directory = state_root / search_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "search_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "search_id": search_id,
                    "phase": "children_enumerated",
                    "phase_notes": {},
                    "children": [],
                }
            ),
            encoding="utf-8",
        )
        return pid

    return _spawn


# ── shell, drafts, modes ────────────────────────────────────────────────────


def test_wizard_shell_and_exact_step_restore(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-001/002 as amended by UI-2: the goal-derived flow (an FSM
    search walks Goal → Baseline → Search axes → Strategy gates → Validation →
    Review), progress + breadcrumb, exact-step restore by the stored step
    key, Save Draft always visible, autosave on Next."""

    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=2)
    text = _caption_text(at)
    assert "**Search axes**" in text  # breadcrumb bolds the flow step
    assert "Prop Contracts" not in text  # skipped by the flow (no prop objective)
    assert _button(at, "Save Draft")
    assert _button(at, "Back")
    assert _button(at, "Next")
    _button(at, "Next").click().run()
    assert not at.exception
    restored = load_draft(roots["drafts"], draft.draft_id)
    assert restored.step_index == 3  # autosaved on Next (the flow position)
    assert restored.current_step_key == "benchmarks"  # exact restore key
    at2 = apptest.AppTest.from_function(_app, default_timeout=120)
    at2.session_state[_DRAFT_KEY] = draft.draft_id
    at2.run()
    assert "**Strategy gates**" in _caption_text(at2)  # exact-step restore


def test_five_modes_five_questions_six_templates(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-004 as amended (owner Q4: Evaluate is a separate question)."""

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
    purpose_box = next(w for w in at.selectbox if w.key == f"{wizard._W}new_purpose")
    assert purpose_box.options == [
        "Implementation Verification",
        "Development Research",
        "Full Authorized Development",
    ]
    question = next(w for w in at.radio if w.key == f"{wizard._W}question")
    assert question.options[:2] == [
        "Evaluate one configuration",
        "Compare one configuration with the baseline",
    ]
    assert len(question.options) == 5
    template = next(w for w in at.selectbox if w.key == f"{wizard._W}template")
    assert template.options == [
        "Payout Reliability",
        "Maximum Expected Payout",
        "Low Breach / Long Account Life",
        "Balanced Prop Performance",
        "Strategy Quality Only",
        "Custom",
    ]
    assert "Resolved objective" in _markdown_text(at)


def test_incompatible_mode_question_pair_blocks_next(monkeypatch, tmp_path) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=0)
    question = next(w for w in at.radio if w.key == f"{wizard._W}question")
    question.set_value("Test repeat-payout feasibility").run()
    assert not at.exception
    assert "different study mode" in _errors(at)
    assert _button(at, "Next").disabled


def test_goal_card_shows_purpose_scope_namespace_and_evidence(
    monkeypatch, tmp_path
) -> None:
    """Plan §7 New Study: the goal card is fixed at the top of every step with
    the derived purpose, run scope, namespace class, evidence class and the
    typed authorization readiness; the namespace id is never guessed."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=2)
    tables = _tables(at)
    assert "Implementation Verification" in tables
    assert "verification_5d" in tables
    assert "'namespace class': {0: 'test'}" in tables
    assert "synthetic_fixture" in tables
    assert "synthetic_marker" in tables
    assert "not_required" in tables
    assert VERIFICATION_BADGE_TEXT in _errors(at)
    assert "Store namespace: **unmarked**" in _caption_text(at)
    assert "search_test" not in _tables(at)  # a path never names authority


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
    assert "blocked" in _errors(at).lower()


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
    expander_labels = [e.label for e in at.expander]
    assert "Staleness" in expander_labels
    assert "Parent Handling" in expander_labels
    # plan F-12: the ambiguous "~N" caption is replaced by the configuration sentence
    captions = _caption_text(at)
    assert "2 configurations: baseline + 1 challenger" in captions
    assert "~" not in captions.split("configurations")[0][-10:]


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
    """FUX-WIZ-008: full card fields; synthetic visibly synthetic. The cards
    come from the PURPOSE's store (Implementation Verification → the test
    store), never from a session selector."""

    roots = _patched_roots(monkeypatch, tmp_path)
    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(
        tmp_path / "fixture", with_prop=False, with_contract=True
    )
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", fixture["store_root"])
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
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=3, mode="prop_benchmark")
    assert "No verified firm contract" in _headings(at)
    assert _button(at, "Next").disabled  # Prop Benchmark needs a contract


def test_conditional_flow_skips_steps_with_visible_reasons(monkeypatch, tmp_path) -> None:
    """Plan §5.4 / §7: Evaluate one configuration walks Goal → Configuration →
    Strategy gates → Validation → Review; the skipped steps are LISTED with
    their reason on the goal card, never rendered empty."""

    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(
        roots["drafts"], step=0, mode="single_configuration", purpose="development_research"
    )
    draft.steps["objective"]["question_id"] = "evaluate_one_configuration"
    draft.steps["search_space"]["axis_selections"] = {}
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    captions = _caption_text(at)
    assert "Configuration" in captions and "Strategy gates" in captions
    assert "Search axes" not in captions and "Challenger" not in captions
    text = _markdown_text(at)
    assert "Strategy Search Space — skipped:" in text
    assert "no comparison" in text
    assert "Prop Contracts — skipped: no prop objective" in text
    assert "Risk Policies — skipped: no prop objective" in text
    assert "Step 1 of 5" in captions
    # walking the flow never lands on a skipped step
    for _ in range(4):
        if _button(at, "Next").disabled:
            break
        _button(at, "Next").click().run()
        assert not at.exception
        assert "**Prop Contracts**" not in _caption_text(at)
        assert "**Strategy Search Space**" not in _caption_text(at)


def test_prop_objective_blocks_instead_of_disappearing(monkeypatch, tmp_path) -> None:
    """Plan §7 / F-04: a strategy goal with a SELECTED prop objective keeps the
    contract step in its flow; with no verified contract that step blocks
    Next with the contract-workflow action, and the objective is never
    rewritten or hidden."""

    at, _draft, _roots = _run_at_step(
        monkeypatch, tmp_path, step=3, template="payout_reliability"
    )
    assert "No verified firm contract" in _headings(at)
    assert "**Firm contracts**" in _caption_text(at)  # the step is IN the flow
    errors = _errors(at)
    assert "payout_probability_per_rolling_30d" in errors  # echoed unchanged
    assert "first_party_verified" in errors
    assert "never rewritten" in errors
    assert _button(at, "Next").disabled


# ── benchmarks / validation ─────────────────────────────────────────────────


def test_benchmarks_render_the_gate_groups_of_the_flow(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-010 as amended by UI-2: the three groups keep their order; a
    strategy-only goal lists the prop group as skipped with its reason
    (never rendered empty); a prop objective brings it back."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=5)
    text = _markdown_text(at)
    first = text.index("1 · Underlying Strategy Gate")
    third = text.index("3 · Robustness Gate")
    assert first < third
    assert "2 · Prop Feasibility Gate" not in text
    assert "Prop Feasibility Gate — skipped: no prop objective" in _caption_text(at)
    assert "proposed_protocol_default" in _caption_text(at)
    info = " ".join(str(i.value) for i in at.info)
    assert "verification_control_flow_gates_v1" in info
    assert "no hidden weighted score" in info.lower()
    at2, _draft2, _roots2 = _run_at_step(
        monkeypatch, tmp_path / "prop", step=5, template="payout_reliability"
    )
    text = _markdown_text(at2)
    first = text.index("1 · Underlying Strategy Gate")
    second = text.index("2 · Prop Feasibility Gate")
    third = text.index("3 · Robustness Gate")
    assert first < second < third


def test_validation_shows_readonly_allowlist_badge_and_checklist(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-011 + FUX-WIZ-007 as amended: the run scope is DERIVED (no
    scope radio), the canonical allowlist is read-only, the badge is exact,
    the VerificationAuthorizationRef readiness is TYPED, and the
    computation-path-scoped checklist is shown."""

    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=6)
    assert VERIFICATION_BADGE_TEXT in _errors(at)  # the non-dismissible badge
    codes = " ".join(str(c.value) for c in at.code)
    assert "2026-06-04" in codes and "2026-06-10" in codes
    assert "2026-06-11" in codes  # protected boundary shown read-only
    assert "Verification authorization missing" in _headings(at)
    text = _markdown_text(at)
    assert "21/R-5:verification_fixture_authorization" in text
    assert "**Run scope:** `verification_5d`" in text
    radio_labels = [radio.label for radio in at.radio]
    assert "Run scope" not in radio_labels  # derived from the purpose (owner Q1)
    evidence = next(w for w in at.radio if w.key == f"{wizard._W}evidence")
    assert evidence.value.startswith("Synthetic fixture")
    captions = _caption_text(at)
    assert "store_unmarked" in captions or "missing" in captions


def test_worker_control_is_absent_or_fixed_to_one(monkeypatch, tmp_path) -> None:
    """HARDENING-BACKEND §4.6 / plan F-13: no operative worker control; the
    sequential V1 truth is stated."""

    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=6)
    assert not at.slider
    labels = [w.label for w in at.number_input]
    assert not any("worker" in (label or "").lower() for label in labels)
    captions = _caption_text(at)
    assert "sequential_children_v1" in captions
    assert "effective workers: 1" in captions
    _button(at, "Save Draft").click().run()
    assert load_draft(roots["drafts"], draft.draft_id).steps["validation"]["worker_limit"] == 1


def test_development_dates_are_validated_field_by_field(monkeypatch, tmp_path) -> None:
    """Plan F-08: the frozen warmup prefix is read-only and every evidence
    date is checked against the backend logical-day contract at the field —
    never as a generic freeze failure."""

    at, _draft, _roots = _run_at_step(
        monkeypatch, tmp_path, step=6, purpose="development_research"
    )
    codes = " ".join(str(c.value) for c in at.code)
    assert "2026-01-01" in codes and "2026-01-12" in codes  # the frozen prefix
    dates = next(w for w in at.text_area if w.key == f"{wizard._W}full_dates")
    dates.set_value("2026-01-17\n2026-06-11").run()
    assert not at.exception
    errors = _errors(at)
    assert "not a logical trading day" in errors
    assert "outside the development evidence window" in errors
    assert _button(at, "Next").disabled
    dates.set_value("2026-01-13\n2026-01-14").run()
    assert "real_dates" not in _errors(at)
    assert "Owner authorization is not ready" in _headings(at)


# ── review, satisfiability, freeze, launch ──────────────────────────────────


def test_review_shows_satisfiability_report_and_unrewritten_objective(
    monkeypatch, tmp_path
) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=7)
    text = _markdown_text(at)
    assert "Resolved charter preview" in text
    assert "Charter satisfiability" in text
    tables = _tables(at)
    assert "FSM search enumerates at least one challenger" in tables
    assert "✓ PASS" in tables and "✕ FAIL" not in tables
    assert "objective (selected; never rewritten)" in tables
    assert "net_expectancy_r" in tables
    assert "2 configurations: baseline + 1 challenger" in _caption_text(at)
    assert "Blocked by contract" in tables and "S11" in tables
    assert "Estimated work" in text
    assert not _button(at, _FREEZE).disabled


@pytest.mark.parametrize(
    ("mutate", "rule"),
    [
        (lambda d: d.steps["search_space"].update(axis_selections={}),
         "fsm_search_has_a_challenger"),
        (
            lambda d: d.steps["objective"].update(template_id="payout_reliability"),
            "prop_objective_requires_a_verified_contract",
        ),
    ],
)
def test_contradictory_drafts_fail_before_freeze(monkeypatch, tmp_path, mutate, rule) -> None:
    """Plan F-04: a zero-axis search and a prop objective without a verified
    contract are refused at Review with their reason; Freeze is disabled and
    the objective is echoed unchanged (never rewritten)."""

    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=7)
    mutate(draft)
    save_draft(roots["drafts"], draft)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    assert rule in _errors(at)
    assert "✕ FAIL" in _tables(at)
    assert _button(at, _FREEZE).disabled
    if rule == "prop_objective_requires_a_verified_contract":
        assert "payout_probability_per_rolling_30d" in _tables(at)  # unchanged
        assert "never rewritten" in _errors(at) or "never rewritten" in _tables(at)


def test_freeze_and_launch_only_in_the_button_handler(
    monkeypatch, tmp_path
) -> None:
    """FUX-WIZ-012 as amended: freeze saves the immutable charter into the
    PURPOSE's store (Implementation Verification → the test store), records
    the purpose annotation, spawns ONLY on the explicit click, and routes to
    Active Runs only after the worker's state exists."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(wizard, "_spawn_search_job", _state_writing_spawn(spawned))
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=7)
    assert spawned == []  # rendering launches nothing
    _button(at, _FREEZE).click().run()
    assert not at.exception
    assert len(spawned) == 1
    command = spawned[0]
    assert "--runner-entry-key" in command
    assert "synthetic_search_job_fixture_v1" in command
    assert command[command.index("--store-root") + 1] == str(roots["verification"])
    search_id = command[command.index("--search-id") + 1]
    assert has_envelope(roots["verification"], "charters", search_id)
    assert not has_envelope(roots["research"], "charters", search_id)
    frozen = load_draft(roots["drafts"], draft.draft_id)
    assert frozen.status == "frozen"
    assert frozen.frozen_search_id == search_id
    annotation = catalog_annotations(roots["verification"])[search_id]["purpose"]
    assert annotation["purpose"] == "implementation_verification"
    assert annotation["evidence_class"] == "synthetic_fixture"
    success = " ".join(str(s.value) for s in at.success)
    assert "state persisted" in success
    assert (
        at.session_state[f"{wizard.STATE_PREFIX}pending_route"] == "Active Runs"
    )


def test_launch_refuses_unregistered_runner_before_spawn(monkeypatch, tmp_path) -> None:
    """Plan F-02: outside the development checkout the synthetic key is
    unregistered — the typed runner_unavailable state renders and NO worker
    is spawned (never a reported success that dead-ends)."""

    from alpha_lab.agents.data_infra.ifvg.search import runner_registry

    spawned: list[list[str]] = []
    monkeypatch.setattr(wizard, "_spawn_search_job", _state_writing_spawn(spawned))
    monkeypatch.setattr(runner_registry, "_DEVELOPMENT_ENTRIES", {})
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=7)
    _button(at, _FREEZE).click().run()
    assert not at.exception
    assert spawned == []
    assert "No registered executor is available in this process" in _headings(at)
    assert not at.success
    assert f"{wizard.STATE_PREFIX}pending_route" not in at.session_state
    frozen = load_draft(roots["drafts"], draft.draft_id)
    assert frozen.status == "frozen"  # the charter froze; only the launch refused
    codes = " ".join(str(c.value) for c in at.code)
    assert "--runner-entry-key synthetic_search_job_fixture_v1" in codes


def test_launch_reports_started_only_after_state_exists(monkeypatch, tmp_path) -> None:
    """Plan F-02: a spawned worker that never persists state is the typed
    launch_not_started state — not a success, no route to Active Runs."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(
        wizard, "_spawn_search_job", lambda command: spawned.append(command) or 99
    )
    monkeypatch.setattr(wizard, "LAUNCH_STATE_WAIT_SECONDS", 0.4)
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=7)
    _button(at, _FREEZE).click().run()
    assert not at.exception
    assert len(spawned) == 1
    assert "Launch requested — no persisted state yet" in _headings(at)
    assert not at.success
    assert f"{wizard.STATE_PREFIX}pending_route" not in at.session_state
    captions = _caption_text(at)
    assert "pid 99" in captions


def test_research_purpose_cannot_freeze_without_ready_owner_authorization(
    monkeypatch, tmp_path
) -> None:
    """Owner Q1 / plan F-01: a Development Research draft resolves to the
    research namespace; with no verified namespace and no owner evidence the
    typed readiness blocks the freeze BEFORE eight steps — no 'Freeze failed'."""

    spawned: list[list[str]] = []
    monkeypatch.setattr(wizard, "_spawn_search_job", _state_writing_spawn(spawned))
    at, draft, roots = _run_at_step(
        monkeypatch, tmp_path, step=7, purpose="development_research"
    )
    tables = _tables(at)
    assert "Development Research" in tables
    assert "'namespace class': {0: 'research'}" in tables
    assert "owner_authorization_bundle" in tables
    assert _button(at, _FREEZE).disabled
    errors = _errors(at)
    assert "freeze_readiness" in errors
    assert "verified store namespace" in errors or "readiness" in errors
    assert "Freeze failed" not in errors
    assert spawned == []
    assert load_draft(roots["drafts"], draft.draft_id).status == "draft"


def test_full_scope_requires_the_exact_typed_confirmation_and_readiness(
    monkeypatch, tmp_path
) -> None:
    """FUX §15 + plan §5.2: the full-scope warning and typed acknowledgement
    stay; the control remains disabled until the typed owner readiness is
    ready (never a Boolean, never enabled by the phrase alone)."""

    at, _draft, _roots = _run_at_step(
        monkeypatch, tmp_path, step=7, purpose="full_authorized_development"
    )
    errors = _errors(at)
    assert "This will run the full authorized development pipeline." in errors
    assert "It is not an implementation verification run." in errors
    assert "typed_acknowledgement" in errors
    assert _button(at, _FREEZE).disabled
    ack = next(w for w in at.text_input if w.key == f"{wizard._W}ack")
    ack.set_value("run full authorized development").run()
    errors = _errors(at)
    assert "typed_acknowledgement" not in errors  # the phrase is accepted …
    assert "freeze_readiness" in errors  # … but readiness still blocks
    assert _button(at, _FREEZE).disabled


def test_ambiguous_legacy_purpose_blocks_freeze_until_confirmed(
    monkeypatch, tmp_path
) -> None:
    """Plan §5.6: a legacy full-scope draft without an annotation is
    purpose_unresolved (Development Research vs Full Authorized Development
    share the scope); confirming records a presentation annotation only."""

    at, draft, roots = _run_at_step(
        monkeypatch,
        tmp_path,
        step=7,
        purpose=None,
        run_scope="full_authorized_development",
    )
    assert "Run purpose unresolved" in _headings(at)
    assert _button(at, _FREEZE).disabled
    box = next(w for w in at.selectbox if w.key == f"{wizard._W}confirm_purpose")
    box.set_value("Development Research").run()
    _button(at, "Confirm purpose").click().run()
    assert not at.exception
    stored = load_draft(roots["drafts"], draft.draft_id)
    assert stored.purpose_annotation["purpose"] == "development_research"
    assert stored.purpose_annotation["derivation"] == "owner_confirmed"
    assert "Run purpose unresolved" not in _headings(at)
    assert "Development Research" in _tables(at)


def test_legacy_verification_draft_derives_its_purpose_unambiguously(
    monkeypatch, tmp_path
) -> None:
    at, _draft, _roots = _run_at_step(monkeypatch, tmp_path, step=7, purpose=None)
    assert "Run purpose unresolved" not in _headings(at)
    assert "Implementation Verification" in _tables(at)
    assert not _button(at, _FREEZE).disabled


def test_full_pipeline_mode_renders_the_operator_workflow(
    monkeypatch, tmp_path
) -> None:
    """R5 flip of the R4 planned-capability assertion: mode-5 step 8 delegates
    to the real §30 workflow — no planned state, no search-freeze button, the
    pipeline phase radio present (full coverage in test_ifvg_pipeline_tab.py)."""

    import ifvg_pipeline_tab as pipeline_tab

    roots = _patched_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_tab, "PIPELINE_STATE_ROOT", tmp_path / "pipeline_jobs"
    )
    draft = _seed_draft(roots["drafts"], step=7, mode="full_pipeline_run")
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    assert "planned / unavailable" not in _headings(at).lower()
    assert not any(b.label == _FREEZE for b in at.button)
    phase_options = {tuple(radio.options) for radio in at.radio}
    assert any("Monitor" in options for options in phase_options)


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
    assert "never replays" in _errors(at)
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
    monkeypatch.setattr(wizard, "_spawn_search_job", _state_writing_spawn(spawned))
    monkeypatch.setattr(wizard, "_commit_of", lambda path: "unknown")
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=7)
    _button(at, _FREEZE).click().run()
    assert not at.exception
    assert spawned == []
    assert "provenance" in _errors(at).lower()
    assert load_draft(roots["drafts"], draft.draft_id).status == "draft"


def test_clone_as_new_search_from_a_frozen_draft(monkeypatch, tmp_path) -> None:
    """FUX-WIZ-003: the original stays frozen; the clone is a new draft that
    keeps the purpose annotation (derivation: cloned)."""

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
    assert clone.purpose_annotation["purpose"] == "implementation_verification"
    assert clone.purpose_annotation["derivation"] == "cloned"
    original = load_draft(roots["drafts"], draft.draft_id)
    assert original.status == "frozen"


# ── UI-2 drafts (owner Q2): session-only, Saved / Unsaved chip, autosave ────


def test_session_only_draft_writes_no_file_until_save_or_valid_next(
    monkeypatch, tmp_path
) -> None:
    """Owner Q2: Start new draft creates a SESSION draft (no file); the goal
    card and purpose annotation are live; the first Save Draft persists it
    with the annotation; afterwards the chip reads Saved."""

    roots = _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception
    mode_box = next(w for w in at.selectbox if w.key == f"{wizard._W}new_mode")
    mode_box.set_value("Single Configuration").run()
    _button(at, "Start new draft").click().run()
    assert not at.exception
    draft_id = at.session_state[_DRAFT_KEY]
    assert not (roots["drafts"] / draft_id).exists()  # nothing written
    text = _markdown_text(at)
    assert "Not saved yet" in text and "no file" in text.lower()
    assert "Implementation Verification" in _tables(at)  # the annotation lives in the session
    name = next(w for w in at.text_input if w.key == f"{wizard._W}name")
    assert "Single Configuration" in name.value  # the proposed default name (goal — date)
    assert "2026-" in name.value
    _button(at, "Save Draft").click().run()
    assert not at.exception
    stored = load_draft(roots["drafts"], draft_id)
    assert stored.purpose_annotation["purpose"] == "implementation_verification"
    assert stored.steps["objective"]["mode_id"] == "single_configuration"
    assert stored.current_step_key == "objective"
    assert "Saved" in _markdown_text(at)
    assert "Not saved yet" not in _markdown_text(at)


def test_first_valid_next_persists_a_session_draft(monkeypatch, tmp_path) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    mode_box = next(w for w in at.selectbox if w.key == f"{wizard._W}new_mode")
    mode_box.set_value("Single Configuration").run()
    _button(at, "Start new draft").click().run()
    draft_id = at.session_state[_DRAFT_KEY]
    assert not (roots["drafts"] / draft_id).exists()
    assert not _button(at, "Next").disabled
    _button(at, "Next").click().run()
    assert not at.exception
    stored = load_draft(roots["drafts"], draft_id)
    assert stored.step_index == 1 and stored.current_step_key == "baseline"
    assert f"{wizard.STATE_PREFIX}session_draft" not in at.session_state


def test_draft_name_is_required_before_the_first_persistence(monkeypatch, tmp_path) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    _button(at, "Start new draft").click().run()
    draft_id = at.session_state[_DRAFT_KEY]
    name = next(w for w in at.text_input if w.key == f"{wizard._W}name")
    name.set_value("").run()
    _button(at, "Save Draft").click().run()
    assert not at.exception
    assert "Draft name required" in _errors(at)
    assert not (roots["drafts"] / draft_id).exists()


def test_identical_unsaved_draft_warns_about_the_persisted_duplicate(
    monkeypatch, tmp_path
) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    existing = _seed_draft(
        roots["drafts"], step=0, mode="single_configuration", purpose="implementation_verification"
    )
    existing.steps["objective"]["question_id"] = "evaluate_one_configuration"
    existing.steps["baseline"] = {}
    save_draft(roots["drafts"], existing)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    mode_box = next(w for w in at.selectbox if w.key == f"{wizard._W}new_mode")
    mode_box.set_value("Single Configuration").run()
    _button(at, "Start new draft").click().run()
    assert not at.exception
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "identical" in warnings and existing.display_name in warnings
    assert not (roots["drafts"] / at.session_state[_DRAFT_KEY]).exists()  # never a block


def test_persisted_draft_autosaves_and_shows_the_saved_chip(monkeypatch, tmp_path) -> None:
    at, draft, roots = _run_at_step(monkeypatch, tmp_path, step=2)
    assert "Saved" in _markdown_text(at)
    target = next(
        w for w in at.multiselect if w.key == f"{wizard._W}axis_parent_retest_timeout_1m_bars"
    )
    target.set_value([]).run()  # a change on a PERSISTED draft autosaves
    assert not at.exception
    stored = load_draft(roots["drafts"], draft.draft_id)
    assert not stored.steps["search_space"]["axis_selections"]
    assert "Autosaved" in _markdown_text(at)


def test_archived_draft_cannot_be_edited_from_the_wizard(monkeypatch, tmp_path) -> None:
    roots = _patched_roots(monkeypatch, tmp_path)
    draft = _seed_draft(roots["drafts"], step=2)
    archive_draft(roots["drafts"], draft.draft_id)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    assert "This draft is archived" in _headings(at)
    labels = [b.label for b in at.button]
    assert "Next" not in labels and _FREEZE not in labels
    assert "restore" in " ".join(str(i.value) for i in at.info).lower()
