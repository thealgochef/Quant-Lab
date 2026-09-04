"""FUX-RES-001..006 + FUX-HIST-001 AppTests: common frame, overview cards,
exact no-pass copy, frontier twin, heatmap glyph/table twin, firm/survival/
payout views, explorer presets, immutable History (TEST_MATRIX §3.11)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_results_tab as results  # noqa: E402
import ifvg_study_tab as study_tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: E402
    DEV_BADGE_TEXT,
    NO_PASS_SENTENCE,
)
from tests.agents.ifvg_search.study_ui_fixture import (  # noqa: E402
    build_completed_search,
)


@pytest.fixture(scope="module")
def completed_search(tmp_path_factory) -> dict:
    return build_completed_search(tmp_path_factory.mktemp("results_ui"))


def _patch(monkeypatch, fixture: dict) -> None:
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", fixture["store_root"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", fixture["state_root"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", fixture["draft_root"])


def _results_app() -> None:
    import ifvg_results_tab as results
    import ifvg_study_tab as study_tab
    import streamlit as st

    results.render_results(st, roots=study_tab.workspace_roots(st))


def _history_app() -> None:
    import ifvg_results_tab as results
    import ifvg_study_tab as study_tab
    import streamlit as st

    results.render_history(st, roots=study_tab.workspace_roots(st))


def _run(monkeypatch, fixture: dict, app=_results_app, **session):
    _patch(monkeypatch, fixture)
    at = apptest.AppTest.from_function(app, default_timeout=120)
    for key, value in session.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def _text(at) -> str:
    return "\n".join(
        [str(b.value) for b in at.markdown]
        + [str(c.value) for c in at.caption]
        + [str(h.value) for h in at.subheader]
    )


def test_common_frame_picker_disclosure_badge_scopes(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-001: full-identity picker, disclosure, dev badge, scope and
    gross/cost/net labels."""

    at = _run(monkeypatch, completed_search)
    warnings = " ".join(str(w.value) for w in at.warning)
    assert DEV_BADGE_TEXT in warnings  # persistent development badge
    codes = " ".join(str(c.value) for c in at.code)
    assert completed_search["search_id"] in codes  # full identity, copyable
    disclosure = next(
        w for w in at.radio if w.key == f"{results._RES}disclosure"
    )
    assert disclosure.options == ["Summary", "Analyst", "Audit"]
    text = _text(at)
    assert "Result scope:" in text
    # the caption DERIVES from the persisted simulation_mode — the fixture
    # is a historical closed-trade replay, so Bootstrap must NOT be claimed
    assert "Prop Historical Closed-Trade Replay" in text
    assert "Bootstrap Simulation" not in text
    assert "Actual Executed Strategy" in text


def test_overview_cards_and_summary_answers(monkeypatch, completed_search) -> None:
    """FUX-RES-002: the exact four ranking-dimension cards."""

    at = _run(monkeypatch, completed_search)
    text = _text(at)
    assert "Development Exploratory Representative" in text
    assert "Highest Expected Payout" in text
    assert "Highest Payout Reliability" in text
    assert "Lowest Breach Risk" in text
    assert "ranking-dimension titles" in text
    codes = " ".join(str(c.value) for c in at.code)
    assert completed_search["representative"] in codes


def test_no_pass_state_renders_the_exact_sentence(monkeypatch, tmp_path) -> None:
    """FUX-RES-002/STATE-001: exact copy + dominant failure reasons."""

    fixture = build_completed_search(tmp_path, with_prop=False, with_contract=False)
    # strip the frontier pointer so the view sees a no-pass run
    import json

    state_file = (
        fixture["state_root"] / fixture["search_id"] / "search_state.json"
    )
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["phase_notes"].pop("frontier_id", None)
    for child in state["children"]:
        if child.get("failure_reason") is None:
            child["failure_reason"] = "insufficient_trades"
            child["explanation"] = "strategy gate failed"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    at = _run(monkeypatch, fixture)
    text = _text(at)
    assert NO_PASS_SENTENCE in text
    assert "Dominant failure reasons" in text  # per-gate stopped counts follow


def test_frontier_has_the_accessible_selectbox_twin(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-003: exact axes and the always-present selectbox twin."""

    at = _run(
        monkeypatch,
        completed_search,
        **{f"{results._RES}disclosure": "Analyst"},
    )
    twin = next(
        w for w in at.selectbox if w.key == f"{results._RES}frontier_twin"
    )
    assert twin.options  # the representative is selectable without the chart
    text = _text(at)
    assert "Payout-reliability frontier" in text


def test_heatmap_controls_glyphs_and_table_twin(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-004: axis/metric pickers, glyph classes, table twin."""

    at = _run(
        monkeypatch,
        completed_search,
        **{f"{results._RES}disclosure": "Analyst"},
    )
    metric = next(w for w in at.selectbox if w.key == f"{results._RES}heat_metric")
    assert set(metric.options) == {
        "net E[R]",
        "trade count",
        "maximum drawdown",
        "expected payout",
        "breach probability",
        "payout reliability",
    }
    twin_frames = [frame.value for frame in at.dataframe]
    assert any("status" in frame.columns for frame in twin_frames)
    joined = "".join(str(frame.to_dict()) for frame in twin_frames)
    assert any(glyph in joined for glyph in ("stable_plateau", "failed_region"))


def test_firm_survival_and_payout_views(monkeypatch, completed_search) -> None:
    """FUX-RES-005: metric toggles, horizons, P10 prominence."""

    at = _run(
        monkeypatch,
        completed_search,
        **{f"{results._RES}disclosure": "Analyst"},
    )
    matrix_metric = next(
        w for w in at.selectbox if w.key == f"{results._RES}firm_metric"
    )
    assert list(matrix_metric.options) == [
        "P(3 payouts before breach)",
        "expected payout",
        "breach probability",
        "first-payout probability",
    ]
    horizon = next(
        w for w in at.selectbox if w.key == f"{results._RES}payout_horizon"
    )
    assert horizon.options == ["30-day", "60-day", "90-day", "lifetime"]
    text = _text(at)
    assert "Lower tail first" in text and "P10" in text
    assert "scenario/approximation" in text or "Simulation mode" in text


def test_explorer_presets_pagination_and_names(
    monkeypatch, completed_search
) -> None:
    """FUX-RES-006: exact presets, identity columns first, baseline-diff
    names, indexed pagination."""

    at = _run(monkeypatch, completed_search)
    preset = next(w for w in at.radio if w.key == f"{results._RES}preset")
    assert preset.options == ["Strategy", "Prop", "Robustness"]
    explorer_frames = [frame.value for frame in at.dataframe]
    explorer = next(
        frame for frame in explorer_frames if "config name" in frame.columns
    )
    assert list(explorer.columns)[:4] == [
        "rank",
        "config name",
        "status",
        "changed parameters",
    ]
    names = set(explorer["config name"])
    assert "Baseline (doc-default)" in names
    assert any(name != "Baseline (doc-default)" for name in names)
    assert "net E[R]" in explorer.columns
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Rows 1–4 of 4" in captions


def test_prop_preset_shows_true_survival_values(
    monkeypatch, completed_search
) -> None:
    """FUX §24 (adversarial F1): the '90-day survival' column displays the
    SURVIVAL quantity (1 − breach), never the breach probability."""

    at = _run(monkeypatch, completed_search)
    preset = next(w for w in at.radio if w.key == f"{results._RES}preset")
    preset.set_value("Prop").run()
    assert not at.exception
    frames = [frame.value for frame in at.dataframe]
    explorer = next(
        frame for frame in frames if "90-day survival" in frame.columns
    )
    values = set(explorer["90-day survival"])
    # fixture: breach_probability_90d = 0.20 → survival 80.0%
    assert "80.0%" in values
    assert "0.20" not in values and "20.0%" not in values


def test_missing_simulations_render_artifact_unavailable_not_gate_skip(
    monkeypatch, tmp_path
) -> None:
    """Adversarial F9a: feasible configs WITHOUT persisted simulations show
    'Artifact unavailable' — never the §16.4 gate-skip sentence."""

    fixture = build_completed_search(
        tmp_path, with_prop=False, with_contract=False
    )
    at = _run(
        monkeypatch,
        fixture,
        **{f"{results._RES}disclosure": "Analyst"},
    )
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Artifact unavailable" in headings
    text = _text(at)
    frontier_section = text[text.index("Payout-reliability frontier") :]
    assert "Not run — strategy gate failed" not in frontier_section


def test_in_progress_run_never_shows_the_no_pass_verdict(
    monkeypatch, completed_search, tmp_path
) -> None:
    """FUX §18 (adversarial F19): the no-pass sentence is a TERMINAL verdict."""

    import json

    state_root = tmp_path / "progress_state"
    source = completed_search["state_root"] / completed_search["search_id"]
    target = state_root / completed_search["search_id"]
    target.mkdir(parents=True)
    state = json.loads(
        (source / "search_state.json").read_text(encoding="utf-8")
    )
    state["phase"] = "replays"
    state["phase_notes"] = {}
    (target / "search_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    fixture = dict(completed_search, state_root=state_root)
    at = _run(monkeypatch, fixture)
    text = _text(at)
    assert NO_PASS_SENTENCE not in text
    infos = " ".join(str(i.value) for i in at.info)
    assert "Run in progress" in infos


def test_heatmap_single_axis_view_aggregates_honestly(
    monkeypatch, completed_search
) -> None:
    """FUX §33 (adversarial F5): collapsing children into one cell is a
    reported aggregation (mean + n), never a silent keep-last."""

    at = _run(
        monkeypatch,
        completed_search,
        **{f"{results._RES}disclosure": "Analyst"},
    )
    captions = " ".join(str(c.value) for c in at.caption)
    assert "configurations aggregated (mean)" in captions
    frames = [frame.value for frame in at.dataframe]
    twin = next(
        frame for frame in frames if "evidence reason" in frame.columns
    )
    assert any("mean of" in str(value) for value in twin["evidence reason"])


def test_explorer_pagination_over_64_children(monkeypatch, tmp_path) -> None:
    """FUX-RES-006/PERF-001 (adversarial F18c): indexed pagination over a
    64-child state — no full-artifact loads (state-only fixture)."""

    import json

    search_id = "9" * 64
    state_root = tmp_path / "big_state"
    (state_root / search_id).mkdir(parents=True)
    children = [
        {
            "ordinal": index,
            "core_replay_id": f"{index:064x}",
            "axis_value_ids": {"parent_retest_timeout_1m_bars": f"v{index}"},
            "comparison_role": "challenger" if index else "baseline",
            "state": "completed",
            "failure_reason": "insufficient_trades" if index % 2 else None,
            "explanation": "",
            "replay_invocations": 1,
        }
        for index in range(64)
    ]
    (state_root / search_id / "search_state.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "search_id": search_id,
                "phase": "search_complete",
                "phase_notes": {},
                "children": children,
            }
        ),
        encoding="utf-8",
    )
    fixture = {
        "store_root": tmp_path / "empty_store",
        "state_root": state_root,
        "draft_root": tmp_path / "drafts",
    }
    at = _run(monkeypatch, fixture)
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Rows 1–25 of 64" in captions


def test_history_separates_sections_and_keeps_frozen_immutable(
    monkeypatch, completed_search, tmp_path
) -> None:
    """FUX-HIST-001: five sections; no delete/overwrite for frozen evidence;
    clone supported; rename is a catalog annotation only; duplicate
    semantics render as verified reuse."""

    import json

    from alpha_lab.agents.data_infra.ifvg.study_drafts import (
        mark_frozen,
        new_draft,
        save_draft,
    )

    # an isolated state copy with one REUSED child (duplicate semantics)
    state_root = tmp_path / "hist_state"
    source = completed_search["state_root"] / completed_search["search_id"]
    target = state_root / completed_search["search_id"]
    target.mkdir(parents=True)
    state = json.loads(
        (source / "search_state.json").read_text(encoding="utf-8")
    )
    state["children"][0]["state"] = "reused"
    (target / "search_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    fixture = dict(completed_search, state_root=state_root)

    draft_root = completed_search["draft_root"]
    mutable = new_draft("fsm_config_search", display_name="History mutable")
    save_draft(draft_root, mutable)
    frozen = new_draft("fsm_config_search", display_name="History frozen")
    save_draft(draft_root, frozen)
    mark_frozen(draft_root, frozen, search_id=completed_search["search_id"])

    at = _run(monkeypatch, fixture, app=_history_app)
    headings = [str(h.value) for h in at.subheader]
    assert headings == [
        "Drafts",
        "Frozen / Running Studies",
        "Completed Studies",
        "Superseded Studies",
        "Legacy Read-Only Results",
    ]
    labels = [b.label for b in at.button]
    assert "Clone as New Search" in labels
    assert "Discard" not in labels  # discard appears only after confirmation
    forbidden = ("delete", "overwrite", "sealed", "recapture", "promote")
    for label in labels:
        for word in forbidden:
            assert word not in label.lower()
    text = _text(at)
    assert "mutable annotations only" in text
    assert "verified reuse" in text  # duplicate semantics shown as reuse
    assert "Legacy" in text and "no rerun" in text.lower()


# ── UI-2 History lifecycle (owner Q2; plan §7 History) ──────────────────────


def _history_at(monkeypatch, completed_search, tmp_path, **session):

    state_root = tmp_path / "hist_state"
    source = completed_search["state_root"] / completed_search["search_id"]
    target = state_root / completed_search["search_id"]
    if not target.exists():
        target.mkdir(parents=True)
        (target / "search_state.json").write_text(
            (source / "search_state.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
    fixture = dict(completed_search, state_root=state_root, draft_root=tmp_path / "drafts")
    return _run(monkeypatch, fixture, app=_history_app, **session), fixture


def _history_button(at, label: str):
    return next(b for b in at.button if b.label == label)


def test_history_archive_restore_and_typed_delete(monkeypatch, completed_search, tmp_path) -> None:
    """Owner Q2: Archive is the reversible action (hidden by default, restorable,
    never a delete); permanent deletion exists only in the archived view, only
    for never-frozen drafts, only with the exact typed name."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import (
        list_drafts,
        load_draft,
        mark_frozen,
        new_draft,
        save_draft,
    )

    draft_root = tmp_path / "drafts"
    mutable = new_draft("fsm_config_search", display_name="Archive me")
    mutable.steps["objective"] = {"question_id": "find_robust_fsm"}
    save_draft(draft_root, mutable)
    frozen = new_draft("fsm_config_search", display_name="History frozen")
    save_draft(draft_root, frozen)
    mark_frozen(draft_root, frozen, search_id=completed_search["search_id"])
    at, _fixture = _history_at(monkeypatch, completed_search, tmp_path)
    labels = [b.label for b in at.button]
    assert "Archive" in labels and "Discard" not in labels
    assert not any("delete" in label.lower() for label in labels)  # not in the default view
    _history_button(at, "Archive").click().run()
    assert not at.exception
    stored = load_draft(draft_root, mutable.draft_id)
    assert stored.archived is True and stored.steps["objective"]["question_id"] == "find_robust_fsm"
    assert "Archive me" not in _text(at)  # hidden from the default listing
    assert [d.draft_id for d in list_drafts(draft_root)] == [frozen.draft_id]  # provenance only
    # the archived view: Restore, and the typed permanent delete for never-frozen drafts
    at.checkbox(key=f"{results._RES}hist_show_archived").check().run()
    assert not at.exception
    assert "Archived drafts" in " ".join(str(h.value) for h in at.subheader)
    assert "Archive me" in _text(at)
    delete = _history_button(at, "Delete draft permanently")
    assert delete.disabled  # nothing typed yet
    typed = at.text_input(key=f"{results._RES}del_name_{mutable.draft_id[:8]}")
    typed.set_value("archive me").run()  # not exact
    assert _history_button(at, "Delete draft permanently").disabled
    assert (draft_root / mutable.draft_id / "draft.json").exists()
    _history_button(at, "Restore").click().run()
    assert not at.exception
    assert load_draft(draft_root, mutable.draft_id).archived is False
    assert "Archive me" in _text(at)
    # archive again and delete with the exact name
    _history_button(at, "Archive").click().run()
    typed = at.text_input(key=f"{results._RES}del_name_{mutable.draft_id[:8]}")
    typed.set_value("Archive me").run()
    delete = _history_button(at, "Delete draft permanently")
    assert not delete.disabled
    delete.click().run()
    assert not at.exception
    assert not (draft_root / mutable.draft_id).exists()
    # frozen provenance is never offered for deletion anywhere
    assert load_draft(draft_root, frozen.draft_id).status == "frozen"
    text = _text(at)
    assert "never deletable" in text.lower() or "no delete or overwrite control" in text


def test_history_bulk_archives_empty_untitled_drafts_and_deletes_nothing(
    monkeypatch, completed_search, tmp_path
) -> None:
    from alpha_lab.agents.data_infra.ifvg.study_drafts import list_drafts, new_draft, save_draft

    draft_root = tmp_path / "drafts"
    empties = [new_draft("fsm_config_search"), new_draft("prop_benchmark")]
    for draft in empties:
        save_draft(draft_root, draft)
    named = new_draft("fsm_config_search", display_name="Keep me")
    save_draft(draft_root, named)
    at, _fixture = _history_at(monkeypatch, completed_search, tmp_path)
    button = _history_button(at, "Archive empty untitled drafts (2)")
    button.click().run()
    assert not at.exception
    live = {d.draft_id for d in list_drafts(draft_root)}
    assert live == {named.draft_id}
    everything = {d.draft_id for d in list_drafts(draft_root, include_archived=True)}
    assert everything == live | {d.draft_id for d in empties}
    assert all((draft_root / d.draft_id / "draft.json").exists() for d in empties)
    assert not any(b.label.startswith("Archive empty untitled drafts") for b in at.button)


def test_history_filters_are_read_only_and_runs_carry_the_archive_flag(
    monkeypatch, completed_search, tmp_path
) -> None:
    """Plan §5.1 / §7: History's purpose / store filters are READ-ONLY (they
    narrow the listing and never change where a charter freezes); immutable
    runs gain the catalog archive flag (never a delete)."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft, save_draft
    from alpha_lab.agents.data_infra.ifvg.study_providers import catalog_annotations

    draft_root = tmp_path / "drafts"
    research = new_draft("single_configuration", display_name="Research draft")
    research.purpose_annotation = {"purpose": "development_research"}
    save_draft(draft_root, research)
    verification = new_draft("single_configuration", display_name="Verification draft")
    verification.purpose_annotation = {"purpose": "implementation_verification"}
    save_draft(draft_root, verification)
    at, fixture = _history_at(monkeypatch, completed_search, tmp_path)
    text = _text(at)
    assert "Research draft" in text and "Verification draft" in text
    purpose_filter = at.selectbox(key=f"{results._RES}hist_purpose_filter")
    assert "read-only" in (purpose_filter.help or "").lower()
    assert not any("namespace" in (radio.label or "").lower() for radio in at.radio)
    assert completed_search["search_id"] in " ".join(str(c.value) for c in at.code)
    # a fresh listing under each read-only filter (the filters narrow; nothing else)
    filtered, _fixture = _history_at(
        monkeypatch,
        completed_search,
        tmp_path,
        **{f"{results._RES}hist_purpose_filter": "Implementation Verification"},
    )
    text = _text(filtered)
    assert "Verification draft" in text and "Research draft" not in text
    by_store, _fixture = _history_at(
        monkeypatch,
        completed_search,
        tmp_path,
        **{f"{results._RES}hist_store_filter": "research"},
    )
    # the fixture store is unmarked: its run is not a `research` namespace run
    assert completed_search["search_id"] not in " ".join(str(c.value) for c in by_store.code)
    # the archive flag on an immutable run (a catalog annotation, never a delete)
    _history_button(at, "Archive run (catalog flag)").click().run()
    assert not at.exception
    assert catalog_annotations(fixture["store_root"])[completed_search["search_id"]]["archived"]
    headings = [str(h.value) for h in at.subheader]
    assert "Superseded Studies" in headings
    _history_button(at, "Restore run").click().run()
    assert not catalog_annotations(fixture["store_root"])[completed_search["search_id"]]["archived"]
