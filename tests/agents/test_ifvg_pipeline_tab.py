"""Full Pipeline Run AppTests (FUX-PIPE-001..006; TEST_MATRIX §3.11 R5 rows).

Configure's capability matrix, Preview's exact disclosure, launch confined
to the button handler, the 16-stage monitor with semantic id + attempt
history, checkpoint resume/retry semantics, and the verify-then-activate
publication boundary — all over the synthetic pipeline fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_pipeline_tab as pipeline_tab  # noqa: E402
import ifvg_study_tab as study_tab  # noqa: E402
import ifvg_study_wizard as wizard  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: E402
    run_pipeline,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft  # noqa: E402
from tests.agents.ifvg_search.pipeline_fixture import (  # noqa: E402
    build_pipeline_fixture,
)
from tests.agents.test_ifvg_study_wizard import (  # noqa: E402
    _patched_roots,
    _seed_draft,
)

_PIPE = pipeline_tab._PIPE
_DRAFT_KEY = f"{wizard.STATE_PREFIX}draft_id"


def _app() -> None:
    import ifvg_pipeline_tab as pipeline_tab
    import ifvg_study_tab as study_tab
    import streamlit as st

    pipeline_tab.render_pipeline_run(st, roots=study_tab.workspace_roots(st))


def _caption_text(at) -> str:
    return "\n".join(str(block.value) for block in at.caption)


def _dataframe_dump(at) -> str:
    return "\n".join(frame.value.to_string() for frame in at.dataframe)


def _seed_mode5_draft(roots) -> object:
    draft = _seed_draft(roots["drafts"], step=7, mode="full_pipeline_run")
    draft.steps["objective"]["question_id"] = "find_robust_fsm"
    save_draft(roots["drafts"], draft)
    return draft


def _run(monkeypatch, tmp_path, *, phase: str | None = None, draft=True):
    roots = _patched_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_tab, "PIPELINE_STATE_ROOT", tmp_path / "pipeline_jobs"
    )
    seeded = _seed_mode5_draft(roots) if draft else None
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    if phase:
        at.session_state[f"{_PIPE}phase_radio"] = phase
    at.run()
    assert not at.exception
    return at, roots, seeded


@pytest.fixture(scope="module")
def completed_pipeline(tmp_path_factory):
    """One completed synthetic 16-stage run on module-scoped tmp roots."""

    tmp_root = tmp_path_factory.mktemp("pipeline_tab")
    fixture = build_pipeline_fixture(tmp_root)
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    save_charter(fixture["store_root"], fixture["charter"])
    save_or_reuse_envelope(fixture["store_root"], "pipeline_specs", fixture["semantic"])
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    return {**fixture, "result": result, "tmp_root": tmp_root}


def _run_over_completed(monkeypatch, completed, *, phase: str):
    roots = {
        "research": completed["tmp_root"] / "search" / "v1",
        "verification": completed["store_root"],
        "state": completed["tmp_root"] / "state_ui",
        "drafts": completed["tmp_root"] / "drafts_ui",
    }
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", roots["research"])
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", roots["verification"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", roots["state"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", roots["drafts"])
    monkeypatch.setattr(
        pipeline_tab, "PIPELINE_STATE_ROOT", completed["state_root"]
    )
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[f"{_PIPE}phase_radio"] = phase
    at.run()
    assert not at.exception
    return at


def _state_writing_spawn(spawned: list[list[str]], *, pid: int = 4242):
    """A fake detached spawn that persists the worker's first state file —
    the honest launch reports success only once it exists."""

    import json

    def _spawn(command: list[str]) -> int:
        spawned.append(command)
        pipeline_id = command[command.index("--pipeline-id") + 1]
        state_root = Path(command[command.index("--state-root") + 1])
        directory = state_root / pipeline_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "pipeline_semantic_id": pipeline_id,
                    "run_scope": "verification_5d",
                    "current_stage": "00_validate_inputs",
                    "stages": {},
                    "attempts": [],
                    "children": [],
                }
            ),
            encoding="utf-8",
        )
        return pid

    return _spawn


# ── FUX-PIPE-001 — Configure: complete fields, capability-scoped entries ─────


def test_configure_renders_capability_scoped_fields(monkeypatch, tmp_path) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Configure")
    radio_options = {tuple(radio.options) for radio in at.radio}
    assert any("Full 16-stage pipeline" in " ".join(options) for options in radio_options)
    dump = _dataframe_dump(at)
    # planned/blocked entries VISIBLE with status/reason (never hidden);
    # since the R5B activation the ORDER_FLOW bundles left this table and
    # only the still-planned regime/execution-liquidity bundles remain
    assert "B5_CORE_STRUCTURE_ORDER_FLOW_REGIME" in dump
    assert "planned" in dump
    assert "spectral_clustering_train_only_v1" in dump
    assert "post-V1" in dump
    assert "ifvg_context_gam_v1" in dump
    # UI-1 / HARDENING-BACKEND §4.6: no operative worker control; the
    # sequential V1 runtime truth is stated
    assert not at.slider
    captions = _caption_text(at)
    assert "sequential_children_v1" in captions
    assert "effective workers: 1" in captions


def test_full_plan_exposes_available_bundle_and_model_selectors(
    monkeypatch, tmp_path
) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Configure")
    plan_radio = next(radio for radio in at.radio if radio.key == f"{_PIPE}plan")
    plan_radio.set_value(pipeline_tab._FULL_PLAN_LABEL).run()
    assert not at.exception
    bundle = next(box for box in at.selectbox if box.key == f"{_PIPE}bundle")
    assert "B0_CORE" in bundle.options
    # R5B: the activated order-flow bundles are selectable
    assert "B2_CORE_ORDER_FLOW" in bundle.options
    assert "B3_CORE_STRUCTURE_ORDER_FLOW" in bundle.options
    model = next(box for box in at.selectbox if box.key == f"{_PIPE}model")
    assert "ifvg_context_catboost_binary_v1" in model.options
    assert "ifvg_context_gam_v1" not in model.options
    # R6.1: the bundle-aware rung has no frozen-tier wiring — not offered here
    assert "ifvg_context_catboost_bundle_v1" not in model.options


def test_mbp1_bundle_selection_pins_logistic_and_shows_the_boundary(
    monkeypatch, tmp_path
) -> None:
    """FUX §35 R5B: bundle selection after the versioned activation — an
    MBP-1-bearing selection restricts the model protocol to the one
    bundle-parametrized wiring and renders the persistent
    research_only_offline label."""

    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Configure")
    plan_radio = next(radio for radio in at.radio if radio.key == f"{_PIPE}plan")
    plan_radio.set_value(pipeline_tab._FULL_PLAN_LABEL).run()
    bundle = next(box for box in at.selectbox if box.key == f"{_PIPE}bundle")
    bundle.set_value("B2_CORE_ORDER_FLOW").run()
    assert not at.exception
    model = next(box for box in at.selectbox if box.key == f"{_PIPE}model")
    # R6.1 (§6.J): the logistic protocol AND the bundle-aware CatBoost rung
    assert list(model.options) == [
        "ifvg_context_logistic_l2_v1",
        "ifvg_context_catboost_bundle_v1",
    ]
    warnings = "\n".join(str(block.value) for block in at.warning)
    assert "research_only_offline" in warnings
    assert "bundle-aware CatBoost rung (research-only)" in _caption_text(at)
    assert "tier-locked" in _caption_text(at)


# ── FUX-PIPE-002 — Preview: exact counts and disclosure ─────────────────────


def test_preview_shows_counts_estimates_reuse_and_readiness(
    monkeypatch, tmp_path
) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Preview")
    dump = _dataframe_dump(at)
    assert "Resolved dates" in dump
    assert "Strategy child count" in dump
    assert "Prop simulation count" in dump
    assert "Runtime estimate by phase" in dump
    assert "Storage estimate by phase" in dump
    assert "resolved at launch" in dump  # DEV-R4-6 reuse disclosure
    assert "New artifacts expected" in dump
    assert "Stage-plan readiness" in "\n".join(
        str(block.value) for block in at.markdown
    )
    assert "available" in dump


# ── FUX-PIPE-003 — Launch only in the handler ───────────────────────────────


def test_render_and_import_never_launch(monkeypatch, tmp_path) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        pipeline_tab, "_spawn_pipeline_job", lambda command: spawned.append(command) or 1
    )
    for phase in ("Configure", "Preview", "Launch", "Monitor", "Publish"):
        _run(monkeypatch, tmp_path / phase.replace(" ", "_"), phase=phase)
    assert spawned == []


def test_launch_button_freezes_and_spawns_exactly_once(monkeypatch, tmp_path) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(pipeline_tab, "_spawn_pipeline_job", _state_writing_spawn(spawned))
    at, roots, _draft = _run(monkeypatch, tmp_path, phase="Launch")
    captions = _caption_text(at)
    assert "Purpose **Implementation Verification**" in captions
    assert "namespace class `test`" in captions
    launch = next(
        b for b in at.button if b.label == "Freeze Pipeline Specification and Launch"
    )
    launch.click().run()
    assert not at.exception
    assert len(spawned) == 1
    command = spawned[0]
    assert "--runner-entry-key" in command
    assert command[command.index("--runner-entry-key") + 1] == (
        "pipeline_synthetic_fixture_v1"
    )
    # the purpose's store (Implementation Verification → the test store) and
    # the sequential V1 worker count — never a namespace radio, never > 1
    assert command[command.index("--store-root") + 1] == str(roots["verification"])
    assert command[command.index("--max-workers") + 1] == "1"
    success = " ".join(str(s.value) for s in at.success)
    assert "launched detached" in success
    assert "state persisted" in success


def test_pipeline_launch_refuses_unregistered_runner_and_silent_worker(
    monkeypatch, tmp_path
) -> None:
    """Plan F-02 at the pipeline level: an unregistered executor is refused
    BEFORE any spawn; a spawned worker that never persists state is the typed
    launch_not_started state, never a success."""

    from alpha_lab.agents.data_infra.ifvg.search import runner_registry

    spawned: list[list[str]] = []
    monkeypatch.setattr(pipeline_tab, "_spawn_pipeline_job", _state_writing_spawn(spawned))
    monkeypatch.setattr(runner_registry, "_DEVELOPMENT_ENTRIES", {})
    at, _roots, _draft = _run(monkeypatch, tmp_path / "unregistered", phase="Launch")
    next(
        b for b in at.button if b.label == "Freeze Pipeline Specification and Launch"
    ).click().run()
    assert not at.exception
    assert spawned == []
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "No registered executor is available in this process" in headings
    assert not at.success

    monkeypatch.undo()
    silent: list[list[str]] = []
    monkeypatch.setattr(
        pipeline_tab, "_spawn_pipeline_job", lambda command: silent.append(command) or 7
    )
    monkeypatch.setattr(pipeline_tab, "LAUNCH_STATE_WAIT_SECONDS", 0.4)
    at, _roots, _draft = _run(monkeypatch, tmp_path / "silent", phase="Launch")
    next(
        b for b in at.button if b.label == "Freeze Pipeline Specification and Launch"
    ).click().run()
    assert not at.exception
    assert len(silent) == 1
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Launch requested — no persisted state yet" in headings
    assert not at.success


def test_verification_scope_shows_the_nondismissible_badge(
    monkeypatch, tmp_path
) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Launch")
    errors = " ".join(str(e.value) for e in at.error)
    assert "VERIFICATION ONLY" in errors


# ── FUX-PIPE-004 — Monitor: 16 stages, semantic id, attempts ────────────────


def test_monitor_renders_all_sixteen_stages_and_attempt_history(
    monkeypatch, completed_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Monitor")
    dump = _dataframe_dump(at)
    for stage_token in ("00 ·", "11 ·", "15 ·"):
        assert stage_token in dump
    assert "Blocked" in dump  # S11's designed terminal state
    assert "Completed" in dump
    code_blocks = " ".join(str(block.value) for block in at.code)
    assert completed_pipeline["result"].pipeline_semantic_id in code_blocks
    assert "Execution-attempt history" in "\n".join(
        str(block.value) for block in at.markdown
    )


def test_monitor_consumes_the_persisted_comparison_results(
    monkeypatch, completed_pipeline
) -> None:
    """DEV-R4-16 consumption half (adversarial M-2): the Monitor renders the
    PERSISTED S14 ComparisonResultEnvelopes — per-kind match_basis rows and
    the lineage-report evidence links — via exact-ID loads only."""

    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Monitor")
    dump = _dataframe_dump(at)
    assert "profile_independent_lineage_exact" in dump
    assert "not_comparable" in dump  # the trade kind stays truthfully typed
    captions = _caption_text(at)
    assert "Evidence links (lineage-uniqueness reports):" in captions


def test_monitor_renders_the_section304_operational_fields(
    monkeypatch, completed_pipeline
) -> None:
    """FUX §30.4 (adversarial M-3): elapsed / remaining / workers / warnings
    render as operational annotations."""

    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Monitor")
    captions = _caption_text(at)
    assert "Elapsed: started" in captions
    assert "0 stages remaining (run terminal)" in captions
    assert "Workers configured:" in captions
    assert "Warnings: none recorded" in captions


def test_monitor_ladder_panel_shows_rungs_planned_and_s11(
    monkeypatch, completed_pipeline
) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.decision_policies import (
        S11_BLOCKED_REASON,
    )

    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Monitor")
    dump = _dataframe_dump(at)
    assert "reference_prevalence_v1" in dump
    assert "ifvg_context_logistic_l2_v1" in dump
    assert "ifvg_context_gam_v1" in dump  # planned rung stays visible
    captions = _caption_text(at)
    assert S11_BLOCKED_REASON in captions
    # R5-FIX finding 5: this run has 0 OOS rows — the parity claim is NOT
    # evaluable and must never read "parity held over 0 OOS rows"
    assert "parity not evaluable" in captions
    assert "parity held" not in captions


def test_ladder_frame_is_arrow_safe_with_nullable_dtypes() -> None:
    """R5-FIX finding 1: the ladder table's numeric columns carry nullable
    dtypes (Int64/Float64) with pd.NA for the planned rung — pyarrow
    serializes the frame without the automatic-fix fallback that produced
    the smoke run's 17 tracebacks."""

    import pandas as pd
    import pyarrow as pa

    diagnostics = {
        "rungs": {
            "reference_prevalence_v1": {
                "prediction_report": {
                    "count": 12,
                    "brier_score": 0.21,
                    "brier_skill_score": 0.02,
                    "auc": 0.5321,
                    "status": "ok",
                }
            },
            "ifvg_context_logistic_l2_v1": {
                "prediction_report": {
                    "count": 0,
                    "status": "insufficient_class_coverage",
                    "auc_reason": "no_oos_predictions",
                }
            },
        },
        "parity": {"identical_rows": True, "oos_row_count": 0},
    }
    frame = pipeline_tab._ladder_frame(diagnostics)
    assert str(frame["OOS rows"].dtype) == "Int64"
    assert str(frame["Brier"].dtype) == "Float64"
    assert str(frame["Brier skill"].dtype) == "Float64"
    assert str(frame["Rung"].dtype) == "string"
    assert str(frame["AUC"].dtype) == "string"
    assert str(frame["Status"].dtype) == "string"
    # the planned GAM row is NA in numeric columns, never a string
    planned = frame[frame["Rung"] == "ifvg_context_gam_v1"].iloc[0]
    assert pd.isna(planned["OOS rows"]) and pd.isna(planned["Brier"])
    # the exact regression: Arrow conversion succeeds directly
    pa.Table.from_pandas(frame)


def test_parity_caption_wording_for_zero_and_nonzero_rows() -> None:
    """R5-FIX finding 5: 'held' only with rows to hold over."""

    held = pipeline_tab._parity_caption({"oos_row_count": 236})
    assert "parity held over 236 OOS rows" in held
    empty = pipeline_tab._parity_caption({"oos_row_count": 0})
    assert "not evaluable" in empty
    assert "held" not in empty


def test_monitor_without_runs_renders_the_dedicated_state(
    monkeypatch, tmp_path
) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Monitor", draft=False)
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "No pipeline runs exist yet" in headings
    code_blocks = " ".join(str(block.value) for block in at.code)
    assert "ifvg_pipeline_job.py status" in code_blocks


# ── FUX-PIPE-005 — Resume / Retry semantics ─────────────────────────────────


def test_resume_retry_offers_operational_clone_and_new_attempt(
    monkeypatch, completed_pipeline
) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        pipeline_tab,
        "_spawn_pipeline_job",
        lambda command: spawned.append(command) or 77,
    )
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Resume / Retry")
    captions = _caption_text(at)
    assert "SAME" in captions and "new pipeline specification" in captions
    assert not at.slider  # no operative worker control (sequential V1)
    assert "effective workers: 1" in captions
    retry = next(b for b in at.button if b.label == "Resume / Retry (new attempt)")
    retry.click().run()
    assert not at.exception
    assert len(spawned) == 1
    assert "resume" in spawned[0]
    assert spawned[0][spawned[0].index("--max-workers") + 1] == "1"
    assert spawned[0][spawned[0].index("--store-root") + 1] == str(
        completed_pipeline["store_root"]
    )


# ── FUX-PIPE-006 — Publish: gates first; verification cannot activate ───────


def test_publish_runs_gates_and_refuses_verification_activation(
    monkeypatch, completed_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Publish")
    captions = _caption_text(at)
    assert "prepared_not_published" in captions
    assert "never activate" in captions
    # UI-1: the run's store namespace is read from the artifact's own store
    # (an unmarked tmp store is reported as such — never guessed)
    assert "Store namespace: **unmarked**" in captions
    assert "verification-only artifact" in captions
    gates_button = next(b for b in at.button if b.label == "Run Publication Gates")
    gates_button.click().run()
    assert not at.exception
    dump = _dataframe_dump(at)
    for gate in (
        "all_planned_stages_terminal",
        "no_failed_stages",
        "pipeline_result_persisted",
        "control_flow_gates_passed",
    ):
        assert gate in dump
    # the gate cache is bound to the pipeline id, the store namespace and the
    # state digest (plan F-03)
    pipeline_id = completed_pipeline["result"].pipeline_semantic_id
    cache_keys = [
        key
        for key in at.session_state.filtered_state
        if str(key).startswith(f"{_PIPE}gate_results_")
    ]
    assert len(cache_keys) == 1
    assert pipeline_id in cache_keys[0] and "_unmarked_" in cache_keys[0]
    activate = next(
        b for b in at.button if b.label == "Publish and Activate Catalog Entry"
    )
    assert activate.disabled  # verification scope can never activate


# ── fallbacks + wizard integration ──────────────────────────────────────────


def test_fragment_fallback_preserves_monitor_semantics(
    monkeypatch, completed_pipeline
) -> None:
    """FUX-A11Y-004 (R5 row): without st.fragment the SAME plain body renders
    through the manual Refresh fallback."""

    roots = {
        "research": completed_pipeline["tmp_root"] / "search" / "v1",
        "verification": completed_pipeline["store_root"],
        "state": completed_pipeline["tmp_root"] / "state_ui2",
        "drafts": completed_pipeline["tmp_root"] / "drafts_ui2",
    }
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", roots["research"])
    monkeypatch.setattr(study_tab, "STORE_ROOT_VERIFICATION", roots["verification"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", roots["state"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", roots["drafts"])
    monkeypatch.setattr(
        pipeline_tab, "PIPELINE_STATE_ROOT", completed_pipeline["state_root"]
    )

    def _app_no_fragment() -> None:
        import ifvg_pipeline_tab as pipeline_tab  # noqa: F811
        import ifvg_study_tab as study_tab
        import streamlit as st

        class _NoFragment:
            def __getattr__(self, name):
                if name == "fragment":
                    raise AttributeError(name)
                return getattr(st, name)

        pipeline_tab.render_pipeline_run(
            _NoFragment(), roots=study_tab.workspace_roots(st)
        )

    at = apptest.AppTest.from_function(_app_no_fragment, default_timeout=120)
    at.session_state[f"{_PIPE}phase_radio"] = "Monitor"
    at.run()
    assert not at.exception
    assert "Refresh" in [b.label for b in at.button]
    assert "Blocked" in _dataframe_dump(at)  # the same plain body rendered


def test_wizard_mode5_step8_renders_the_pipeline_surface(
    monkeypatch, tmp_path
) -> None:
    """R5 flip of the R4 planned-capability test: mode-5 step 8 now renders
    the real §30 workflow (no planned state, no search freeze button)."""

    roots = _patched_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline_tab, "PIPELINE_STATE_ROOT", tmp_path / "pipeline_jobs"
    )
    draft = _seed_mode5_draft(roots)

    def _wizard_app() -> None:
        import ifvg_study_tab as study_tab
        import ifvg_study_wizard as wizard
        import streamlit as st

        wizard.render_new_study(st, roots=study_tab.workspace_roots(st))

    at = apptest.AppTest.from_function(_wizard_app, default_timeout=120)
    at.session_state[_DRAFT_KEY] = draft.draft_id
    at.run()
    assert not at.exception
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "planned / unavailable" not in headings.lower()
    assert not any(
        b.label == "Freeze Search Charter and Launch" for b in at.button
    )
    phase_options = {tuple(radio.options) for radio in at.radio}
    assert any("Monitor" in options and "Publish" in options for options in phase_options)


# ── R5B — the MBP-1 Order Flow panel (FUX §35 R5B rows) ─────────────────────


@pytest.fixture(scope="module")
def mbp1_pipeline(tmp_path_factory):
    """One completed synthetic MBP-1 (B2) 16-stage run on module tmp roots."""

    tmp_root = tmp_path_factory.mktemp("pipeline_tab_mbp1")
    fixture = build_pipeline_fixture(tmp_root, mbp1=True)
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    save_charter(fixture["store_root"], fixture["charter"])
    save_or_reuse_envelope(fixture["store_root"], "pipeline_specs", fixture["semantic"])
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    return {**fixture, "result": result, "tmp_root": tmp_root}


def test_mbp1_panel_renders_availability_and_the_persistent_boundary(
    monkeypatch, mbp1_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, mbp1_pipeline, phase="Monitor")
    warnings = "\n".join(str(block.value) for block in at.warning)
    assert "research_only_offline" in warnings
    dump = _dataframe_dump(at)
    # availability: the activated block at version 2 with a resolved id
    assert "IFVG_ORDER_FLOW_MBP1_V1" in dump
    assert "available" in dump
    # the window registry drill-down table
    assert "ofl_win_inversion_entry" in dump
    assert "post_trigger_inclusive" in dump
    # both registry hashes (pre/post activation) render as identities
    codes = "\n".join(str(block.value) for block in at.code)
    assert len([c for c in codes.splitlines() if len(c.strip()) == 64]) >= 2


def test_mbp1_panel_renders_the_stamped_coverage_policy_defaults(
    monkeypatch, mbp1_pipeline
) -> None:
    """R5B.1 review F10: the R-6-family coverage defaults are surfaced as
    proposed_protocol_default rows, never as silent constants."""

    at = _run_over_completed(monkeypatch, mbp1_pipeline, phase="Configure")
    dump = _dataframe_dump(at)
    assert "completeness_evidence_required" in dump
    assert "expected_span_source" in dump
    assert "verified_physical_partition_intersect_authorized_session_v2" in dump
    assert "bad_book_scope_fallback" in dump
    assert "proposed_protocol_default" in dump
    assert "before research use" in dump


def test_mbp1_panel_autofills_and_renders_coverage_and_comparison(
    monkeypatch, mbp1_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, mbp1_pipeline, phase="Monitor")
    dump = _dataframe_dump(at)
    # coverage evidence (per-day + per-window) from the exact-ID auto-fill
    assert "2026-01-13" in dump
    # R5B.1: evidence-based coverage facts; sequence jumps are diagnostics
    assert "Completeness" in dump and "evidenced_complete" in dump
    assert "Sequence jumps (diagnostic)" in dump
    assert "Sequence gaps" not in dump
    assert "ofl_snap_entry" in dump
    # the controlled comparison renders both arms with resolved identities
    assert "Baseline+MBP-1" in dump
    assert "B2_CORE_ORDER_FLOW" in dump
    body = "\n".join(str(block.value) for block in at.markdown)
    assert "not evaluable" in body or "not_evaluable" in _caption_text(at) + dump


def test_mbp1_panel_drilldown_requires_exact_ids(monkeypatch, mbp1_pipeline) -> None:
    import ifvg_mbp1_panels as mbp1_panels

    at = _run_over_completed(monkeypatch, mbp1_pipeline, phase="Monitor")
    candidate_input = next(
        box for box in at.text_input if box.key == f"{mbp1_panels._MBP1}candidate_id"
    )
    candidate_input.set_value("pcand_0000").run()
    assert not at.exception
    dump = _dataframe_dump(at)
    assert "completed_bar_boundary" in dump
    assert "✓ valid" in dump
    # a wrong exact id renders the sanitized unavailable state — no fuzzy
    candidate_input.set_value("no_such_candidate").run()
    assert not at.exception
    body = " ".join(str(block.value) for block in at.markdown) + _caption_text(at)
    assert "no fuzzy" in body


def test_mbp1_panel_with_no_runs_renders_manual_input_guidance(
    monkeypatch, tmp_path
) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Configure")
    captions = _caption_text(at)
    assert "mbp1_coverage_report_id" in captions
    assert "controlled_feature_study_id" in captions


# ── R6 — the Regime Lane panel (FUX §35 R6 rows) ────────────────────────────


@pytest.fixture(scope="module")
def regime_persisted(completed_pipeline):
    """Persisted KMeans regime runs (fixture 2) in the completed pipeline's
    verification store: a HEALTHY run (protocol, every fold fit, assessment,
    a first promotion decision), an UNDER-SAMPLED run, and a panel-grain
    protocol (grain identity only)."""

    from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
    from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        ObservationGranularity,
        RegimePromotionDecision,
        RegimePromotionDecisionEnvelope,
        RegimeRole,
        RegimeStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
        resolve_kmeans_protocol,
        run_regime_protocol,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        persist_regime_assessment,
        persist_regime_fit,
        persist_regime_promotion,
        persist_regime_protocol,
    )
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
        known_cluster_fixture,
    )

    bundle = resolve_bundle("B0_CORE").resolved_feature_bundle_id
    root = completed_pipeline["store_root"]
    results = {}
    for label, n, winsorization in (
        ("healthy", 600, "none"),
        ("undersampled", 170, "clip_p01_p99_train_fitted_v1"),  # a DISTINCT protocol
    ):
        fixture = known_cluster_fixture(k=3, n=n)
        folds = build_context_folds(
            fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
        )
        protocol = resolve_kmeans_protocol(
            input_feature_bundle_ref=bundle,
            resolved_input_features=fixture.regime_input_features,
            winsorization_policy=winsorization,
        )
        run = run_regime_protocol(
            fixture.view.frame,
            folds,
            protocol,
            source_artifact_ids=(fixture.view.view_id,),
            bootstrap_refits=3,
        )
        persist_regime_protocol(root, protocol)
        for fold_fit in run.fold_fits:
            persist_regime_fit(
                root,
                fold_fit,
                run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
                observation_frame=fixture.view.frame,
            )
        persist_regime_assessment(root, run.assessment)
        decision = RegimePromotionDecisionEnvelope.from_payload(
            RegimePromotionDecision(
                resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
                role=RegimeRole.DESCRIPTIVE_ONLY,
                status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_status=RegimeStatus.PLANNED,
                previous_decision_ref=None,
                capability_assessment_ref=run.assessment.regime_capability_assessment_id,
                owner_ratification_ref=None,
                decided_at="2026-08-26T00:00:00Z",
            )
        )
        persist_regime_promotion(root, decision)
        results[label] = {
            "protocol_id": protocol.resolved_regime_protocol_id,
            "assessment_id": run.assessment.regime_capability_assessment_id,
            "fit_id": run.fold_fits[0].fit_envelope.regime_fit_id,
            "decision_id": decision.regime_promotion_decision_id,
        }
    # R6.1 (§6.A grain/bundle-key coherence): the panel grain references the
    # panel bundle (row_id join) and the panel features
    panel_protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=resolve_bundle("BP0_CONTEXT_BAR_PANEL").resolved_feature_bundle_id,
        resolved_input_features=("cbp_realized_range_12", "cbp_realized_volatility_12"),
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=300,
        panel_source_artifact_id="b" * 64,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )
    persist_regime_protocol(root, panel_protocol)
    results["panel"] = {"protocol_id": panel_protocol.resolved_regime_protocol_id}
    return results


def _regime_input(at, name: str):
    import ifvg_regime_panels as regime_panels

    return next(box for box in at.text_input if box.key == f"{regime_panels._REG}{name}")


def _everything(at) -> str:
    return (
        _dataframe_dump(at)
        + _caption_text(at)
        + "\n".join(str(block.value) for block in at.markdown)
        + "\n".join(str(getattr(block, "value", "")) for block in at.subheader)
        + "\n".join(str(getattr(block, "value", "")) for block in at.code)
    )


def test_regime_panel_renders_registry_stamps_and_the_spectral_warning(
    monkeypatch, mbp1_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, mbp1_pipeline, phase="Configure")
    dump = _dataframe_dump(at)
    # every algorithm visible; kmeans implemented; post-V1 planned-disabled
    assert "kmeans_v1" in dump and "implemented (V1)" in dump
    for planned in (
        "minibatch_kmeans_v1",
        "gaussian_mixture_v1",
        "spectral_clustering_train_only_v1",
        "nystrom_kmeans_v1",
    ):
        assert planned in dump
    assert "post_v1_regime_expansion" in dump
    warnings = "\n".join(str(block.value) for block in at.warning)
    assert "Training-only exploratory clustering" in warnings
    # proposal stamps surfaced with the ratification requirement
    assert "proposed_protocol_default" in dump
    assert "before feature-eligible" in dump
    assert "fixed_cluster_count" in dump
    assert "minimum_training_observations_decision_row" in dump


def test_regime_model_card_renders_coverage_occupancy_and_stability(
    monkeypatch, completed_pipeline, regime_persisted
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")
    _regime_input(at, "protocol_id").set_value(
        regime_persisted["healthy"]["protocol_id"]
    ).run()
    _regime_input(at, "assessment_id").set_value(
        regime_persisted["healthy"]["assessment_id"]
    ).run()
    assert not at.exception
    dump = _dataframe_dump(at)
    assert "kmeans_v1" in dump
    assert "candidate_stage_row" in dump
    assert "fixed_k — proposed_protocol_default" in dump
    assert "centroid_predict_v1" in dump
    assert "k-means++_n_init_10_v1" in dump
    # coverage / per-fold coverage / fit identities / occupancy / stability
    assert "OOS assignment coverage" in dump
    assert "regime 0" in dump and "regime 2" in dump
    assert "Bootstrap aligned AMI" in dump
    assert "Temporal transitions counted" in dump
    assert "scaled_input_features_v1" in dump
    assert regime_persisted["healthy"]["fit_id"] in dump
    body = "\n".join(str(block.value) for block in at.markdown)
    assert "Per-fold coverage" in body
    assert "Centroid profiles" in body
    assert "Per-cluster bootstrap agreement" in body
    assert "NOMINAL" in body  # never regime 2 > regime 1
    assert "✓ passed" in body
    assert "nothing here can promote" in body
    codes = "\n".join(str(getattr(block, "value", "")) for block in at.code)
    assert regime_persisted["healthy"]["assessment_id"] in codes  # identity block


def test_regime_sample_adequacy_blocked_state(
    monkeypatch, completed_pipeline, regime_persisted
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")
    _regime_input(at, "protocol_id").set_value(
        regime_persisted["undersampled"]["protocol_id"]
    ).run()
    _regime_input(at, "assessment_id").set_value(
        regime_persisted["undersampled"]["assessment_id"]
    ).run()
    assert not at.exception
    everything = _everything(at)
    assert "promotion is blocked" in everything
    assert "k is never shrunk" in everything
    body = "\n".join(str(block.value) for block in at.markdown)
    assert "✕ failed" in body
    assert "sample_adequacy" in body


def test_regime_panel_grain_identity_renders_the_panel_fields(
    monkeypatch, completed_pipeline, regime_persisted
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")
    _regime_input(at, "protocol_id").set_value(
        regime_persisted["panel"]["protocol_id"]
    ).run()
    assert not at.exception
    dump = _dataframe_dump(at)
    assert "context_bar_panel" in dump
    assert "interval 300s" in dump
    assert "completed_bars_last_at_or_before_v1" in dump


def test_regime_assignment_and_stratification_views_render_by_exact_fit_id(
    monkeypatch, completed_pipeline, regime_persisted
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")
    _regime_input(at, "protocol_id").set_value(
        regime_persisted["healthy"]["protocol_id"]
    ).run()
    _regime_input(at, "fit_id").set_value(regime_persisted["healthy"]["fit_id"]).run()
    assert not at.exception
    body = "\n".join(str(block.value) for block in at.markdown)
    assert "Assignment view" in body
    assert "Coverage by partition" in body
    assert "Stratification of the assignment frame" in body
    assert "Regime timeline" in body
    dump = _dataframe_dump(at)
    assert "train" in dump and "test" in dump
    assert "Margin d2−d1" in dump
    assert "Training feature matrix hash" in dump
    # a fit of ANOTHER protocol is refused as unavailable (exact reason)
    _regime_input(at, "fit_id").set_value(regime_persisted["undersampled"]["fit_id"]).run()
    assert not at.exception
    assert "the fit references a different regime protocol id" in _caption_text(at)
    assert "Assignment view" in "\n".join(str(block.value) for block in at.markdown)
    assert "Coverage by partition" not in "\n".join(str(block.value) for block in at.markdown)


def test_regime_promotion_view_renders_role_and_status(
    monkeypatch, completed_pipeline, regime_persisted
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")
    _regime_input(at, "protocol_id").set_value(
        regime_persisted["healthy"]["protocol_id"]
    ).run()
    _regime_input(at, "decision_id").set_value(
        regime_persisted["healthy"]["decision_id"]
    ).run()
    assert not at.exception
    dump = _dataframe_dump(at)
    assert "descriptive_only" in dump
    assert "first decision" in dump
    assert "feature-eligible and beyond are unreachable" in dump
    captions = _caption_text(at)
    assert "unrepresentable in V1" in captions


def test_regime_panel_bogus_id_renders_the_unavailable_state(
    monkeypatch, completed_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, completed_pipeline, phase="Configure")

    def _unavailable_count() -> int:
        return sum(
            "Artifact unavailable" in str(getattr(block, "value", ""))
            for block in at.subheader
        )

    before = _unavailable_count()  # the Configure draft empty state may already show one
    _regime_input(at, "protocol_id").set_value("f" * 64).run()
    assert not at.exception
    assert _unavailable_count() == before + 1
    assert "missing search-store entry" in _caption_text(at)


def test_regime_panel_exposes_no_control_that_promotes_launches_or_retrains() -> None:
    """FUX §35 R6 / kickoff §9: the Regime Lane is read-only — no button,
    form, toggle, or select exists in the panel source, and nothing in it
    can promote, launch, rank, or retrain."""

    source = (Path(__file__).resolve().parents[2] / "scripts" / "ifvg_regime_panels.py").read_text(
        encoding="utf-8"
    )
    for control in (".button(", ".form(", ".toggle(", ".selectbox(", ".form_submit_button("):
        assert control not in source, control
    for verb in ("promote(", "launch(", "retrain(", "rank(", "subprocess", "session_state["):
        assert verb not in source, verb
    # R6.1 (§6.H): the stability gate is `minimum_bootstrap_aligned_ami_mean` —
    # the word "advisory" is gone from UI code
    tab_path = Path(__file__).resolve().parents[2] / "scripts" / "ifvg_pipeline_tab.py"
    tab_source = tab_path.read_text(encoding="utf-8")
    assert "advisory" not in source.lower()
    assert "advisory" not in tab_source.lower()


# ── R6.1 — the regime study surfaces (plan §6.E / §6.G; FUX §35 R6.1 rows) ──


@pytest.fixture(scope="module")
def regime_pipeline(tmp_path_factory):
    """One completed synthetic candidate-grain regime study (descriptive
    classes) on module tmp roots — S05–S10 regime artifacts, the
    deterministic S10 decisions, and the S14 stratified reports."""

    tmp_root = tmp_path_factory.mktemp("pipeline_tab_regime")
    fixture = build_pipeline_fixture(tmp_root, regime_study="candidate")
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    save_charter(fixture["store_root"], fixture["charter"])
    save_or_reuse_envelope(fixture["store_root"], "pipeline_specs", fixture["semantic"])
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    assert result.stage_statuses["14_build_frontier_and_insights"] == "completed"
    return {**fixture, "result": result, "tmp_root": tmp_root}


def _regime_state(completed) -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import read_pipeline_state

    return read_pipeline_state(completed["state_root"], completed["result"].pipeline_semantic_id)


def _regime_diagnostics(completed) -> dict:
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_regime_diagnostics

    diagnostics = load_regime_diagnostics(completed["store_root"], _regime_state(completed))
    assert diagnostics is not None
    return diagnostics


def _preset_run(monkeypatch, tmp_path, *, phase: str, preset: dict):
    """`_run` with session-state presets applied before the first script run."""

    roots = _patched_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_tab, "PIPELINE_STATE_ROOT", tmp_path / "pipeline_jobs")
    _seed_mode5_draft(roots)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state[f"{_PIPE}phase_radio"] = phase
    for key, value in preset.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    return at


def _markdown(at) -> str:
    return "\n".join(str(block.value) for block in at.markdown)


def test_configure_regime_section_grains_algorithm_stamps_and_supervised_fields(
    monkeypatch, tmp_path
) -> None:
    """Configure (R6.1 §6.E): the regime checkbox, the grain radio with the
    sparsity caption, the single executable algorithm + planned entries +
    the mandatory spectral warning, the fixed-k / stage / input captions,
    bootstrap refits, the class multiselect, and — for supervised classes —
    the supervised fields plus the frozen exact authority ids."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import DESCRIPTIVE_CLASSES

    at = _preset_run(
        monkeypatch,
        tmp_path,
        phase="Configure",
        preset={f"{_PIPE}plan": pipeline_tab._FULL_PLAN_LABEL},
    )
    # the panel bundle is never offered as a candidate view
    bundle = next(box for box in at.selectbox if box.key == f"{_PIPE}bundle")
    assert "BP0_CONTEXT_BAR_PANEL" not in bundle.options
    checkbox = next(box for box in at.checkbox if box.key == f"{_PIPE}regime_on")
    assert checkbox.label == "Include a regime study (V1 KMeans, development)"
    checkbox.check().run()
    assert not at.exception
    grain = next(radio for radio in at.radio if radio.key == f"{_PIPE}regime_grain")
    assert list(grain.options) == list(pipeline_tab._REGIME_GRAINS)
    captions = _caption_text(at)
    assert "sample-adequacy floor is 150" in captions
    assert "Fixed k = 3" in captions and "decision 29" in captions
    assert "entry_decision" in captions
    assert "distance_to_htf_ticks" in captions
    algorithm = next(box for box in at.selectbox if box.key == f"{_PIPE}regime_algorithm")
    assert list(algorithm.options) == ["kmeans_v1"]
    dump = _dataframe_dump(at)
    assert "spectral_clustering_train_only_v1" in dump and "planned" in dump
    warnings = "\n".join(str(block.value) for block in at.warning)
    assert "Training-only exploratory clustering" in warnings
    refits = next(box for box in at.number_input if box.key == f"{_PIPE}regime_refits")
    assert int(refits.value) == 50
    classes = next(box for box in at.multiselect if box.key == f"{_PIPE}regime_classes")
    assert tuple(classes.value) == tuple(DESCRIPTIVE_CLASSES)
    assert not any(box.key == f"{_PIPE}regime_decision_id" for box in at.text_input)
    # supervised classes require the supervised fields + the frozen exact ids
    classes.set_value([*classes.value, "feature_only"]).run()
    assert not at.exception
    keys = {box.key for box in at.text_input}
    assert {
        f"{_PIPE}regime_decision_id",
        f"{_PIPE}regime_owner_id",
        f"{_PIPE}regime_assessment_id",
    } <= keys
    assert "FEATURE_ELIGIBLE" in _caption_text(at)
    assert "promote --to feature_eligible" in _caption_text(at)
    assert "Clone as model-bearing study" in _markdown(at)
    # the panel grain uses the stamped panel feature set
    grain.set_value(pipeline_tab._REGIME_GRAIN_PANEL_5M).run()
    assert not at.exception
    captions = _caption_text(at)
    assert "panel_feature_set_v1" in captions and "cbp_realized_range_12" in captions
    assert "cbp_session_state is the block-declared categorical" in captions
    assert "floor 300" in captions


def test_configure_strategy_plan_disables_the_regime_study(monkeypatch, tmp_path) -> None:
    at, _roots, _draft = _run(monkeypatch, tmp_path, phase="Configure")
    checkbox = next(box for box in at.checkbox if box.key == f"{_PIPE}regime_on")
    assert checkbox.disabled
    assert "requires the full 16-stage plan" in _caption_text(at)


def test_preview_shows_regime_rows_readiness_and_the_floor(monkeypatch, tmp_path) -> None:
    at = _preset_run(
        monkeypatch,
        tmp_path,
        phase="Preview",
        preset={f"{_PIPE}plan": pipeline_tab._FULL_PLAN_LABEL, f"{_PIPE}regime_on": True},
    )
    dump = _dataframe_dump(at)
    assert "Regime study" in dump and "kmeans_v1" in dump and "candidate_stage_row" in dump
    assert "Regime sample-adequacy floor" in dump
    assert "150 training observations per fold" in dump
    assert "Regime comparison classes" in dump and "cohort_descriptive" in dump
    assert "descriptive — S10 derives the status" in dump
    assert "Regime protocol id (resolved)" in _caption_text(at)
    codes = "\n".join(str(getattr(block, "value", "")) for block in at.code)
    assert len([c for c in codes.splitlines() if len(c.strip()) == 64]) >= 1
    assert "Stage-plan readiness" in _markdown(at)
    assert "blocked_capability" not in dump
    assert "This plan references unavailable capabilities" not in "\n".join(
        str(block.value) for block in at.error
    )


def test_monitor_regime_panel_shows_substeps_decisions_and_reports(
    monkeypatch, regime_pipeline
) -> None:
    at = _run_over_completed(monkeypatch, regime_pipeline, phase="Monitor")
    body = _markdown(at)
    assert "Sub-steps (D14" in body
    assert "Decisions (deterministic; nothing here promotes)" in body
    assert "Stratified reports (S14" in body
    assert "Clone as model-bearing study (two-pass workflow)" in body
    dump = _dataframe_dump(at)
    assert "S09a" in dump and "S09b" in dump and "S09c" in dump
    assert "not requested (descriptive study)" in dump
    assert "stratification_ready" in dump and "s10_structural" in dump
    assert "cohort_descriptive" in dump
    diagnostics = _regime_diagnostics(regime_pipeline)
    codes = "\n".join(str(getattr(block, "value", "")) for block in at.code)
    assert diagnostics["resolved_regime_protocol_id"] in codes
    assert diagnostics["regime_capability_assessment_id"] in codes
    assert "propose" in _caption_text(at) and "promote --to feature_eligible" in _caption_text(at)


def test_regime_lane_autofills_exact_ids_and_renders_stratified_results(
    monkeypatch, regime_pipeline
) -> None:
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_regime_report_index

    at = _run_over_completed(monkeypatch, regime_pipeline, phase="Configure")
    diagnostics = _regime_diagnostics(regime_pipeline)
    assert _regime_input(at, "protocol_id").value == diagnostics["resolved_regime_protocol_id"]
    assert _regime_input(at, "assessment_id").value == (
        diagnostics["regime_capability_assessment_id"]
    )
    assert _regime_input(at, "fit_id").value == diagnostics["regime_fit_ids"][0]
    assert _regime_input(at, "decision_id").value == (
        diagnostics["decisions"][-1]["regime_promotion_decision_id"]
    )
    report_id = _regime_input(at, "report_id").value
    assert len(report_id) == 64
    captions = _caption_text(at)
    assert "auto-filled" in captions
    body = _markdown(at)
    # per-fold stability + the renamed gate; the model card + assignment +
    # promotion + stratified views all render from the auto-filled ids
    assert "Per-fold bootstrap stability" in body
    assert "Assignment view" in body and "Promotion decision" in body
    assert "Stratified result view" in body
    dump = _dataframe_dump(at)
    assert "minimum_bootstrap_aligned_ami_mean" in dump
    assert "protocol_wide_minimum_fold_mean_v1" in dump
    assert "Candidate-event transition matrix" in body
    assert "stratification_ready" in dump
    assert "no counterfactual claim" in captions and "never a selection input" in captions
    assert "advisory" not in _everything(at)
    # every persisted class renders by its exact report id
    index = load_regime_report_index(regime_pipeline["store_root"], _regime_state(regime_pipeline))
    markers = {
        "cohort_descriptive": "Cohort descriptive strata",
        "stratified_frontier": "Stratified frontier",
        "stratified_prop": "Stratified prop events",
    }
    for comparison_class, ids in index["reports_by_class"].items():
        _regime_input(at, "report_id").set_value(ids[0]).run()
        assert not at.exception
        assert markers[comparison_class] in _markdown(at)
        assert comparison_class in _dataframe_dump(at)
        if comparison_class == "stratified_prop":
            # the D15 report-local summary renders from its verified sidecar
            assert "Event-regime summary" in _markdown(at)
            assert "registered_storage_budget" in _caption_text(at)
    # a descriptive run delivers no modeled class; the index distinguishes
    # S09c deliveries from typed refusals (never both for one class)
    assert dict(index.get("delivered_by", {})) == {}
    assert set(index["refusals"]) <= {"feature_only", "cohort_model"}
    # a report of ANOTHER protocol is refused with the exact reason
    _regime_input(at, "protocol_id").set_value("f" * 64).run()
    assert not at.exception
    assert "missing search-store entry" in _caption_text(at)


def test_regime_lane_surfaces_a_store_integrity_failure(
    monkeypatch, regime_pipeline, tmp_path
) -> None:
    """Safety review S6 pattern: a tampered persisted sidecar renders the
    integrity note — never a cosmetic blank."""

    import shutil

    from alpha_lab.agents.data_infra.ifvg.search.store import envelope_destination

    store_copy = tmp_path / "store"
    state_copy = tmp_path / "state"
    shutil.copytree(regime_pipeline["store_root"], store_copy)
    shutil.copytree(regime_pipeline["state_root"], state_copy)
    state = _regime_state(regime_pipeline)
    result_id = state["stages"]["10_generate_predictions_and_diagnostics"]["stage_result_id"]
    sidecar = (
        envelope_destination(store_copy, "pipeline_stage_results", result_id)
        / "regime_diagnostics.json"
    )
    sidecar.write_bytes(b'{"tampered": true}\n')
    tampered = {
        **regime_pipeline,
        "store_root": store_copy,
        "state_root": state_copy,
        "tmp_root": tmp_path,
    }
    at = _run_over_completed(monkeypatch, tampered, phase="Configure")
    errors = "\n".join(str(block.value) for block in at.error)
    assert "failed store verification" in errors
    assert _regime_input(at, "protocol_id").value == ""


def test_regime_owner_decision_view_renders_provenance_values_and_window(
    monkeypatch, regime_pipeline
) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        RegimePromotionDecision,
        RegimePromotionDecisionEnvelope,
        RegimeRole,
        RegimeStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        load_regime_assessment,
        load_regime_protocol,
        persist_regime_promotion,
    )
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
        synthetic_owner_decision_fixture,
    )

    root = regime_pipeline["store_root"]
    diagnostics = _regime_diagnostics(regime_pipeline)
    protocol = load_regime_protocol(root, diagnostics["resolved_regime_protocol_id"])
    assessment = load_regime_assessment(root, diagnostics["regime_capability_assessment_id"])
    owner = synthetic_owner_decision_fixture(root, protocol=protocol, assessment=assessment)
    eligible = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=diagnostics["decisions"][-1]["regime_promotion_decision_id"],
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at="2026-08-28T14:00:00+00:00",
        )
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    at = _run_over_completed(monkeypatch, regime_pipeline, phase="Configure")
    _regime_input(at, "decision_id").set_value(eligible.regime_promotion_decision_id).run()
    assert not at.exception
    dump = _dataframe_dump(at)
    assert "feature_eligible" in dump
    assert "present (verified owner-decision artifact" in dump
    assert "synthetic_test_authorization_v1" in dump
    assert "Effective window" in dump and "open-ended" in dump
    assert "25:regime_algorithm_baseline" in dump
    assert "algorithm_parameters_hash" in dump and "kmeans_v1" in dump
    assert "Owner decision artifact" in _markdown(at)
    assert "lawful in the synthetic_fixture run scope only" in _caption_text(at)
    codes = "\n".join(str(getattr(block, "value", "")) for block in at.code)
    assert owner.owner_decision_artifact_id in codes
    # a bare 64-hex reference renders the unavailable state, never a blank
    bare = RegimePromotionDecisionEnvelope.from_payload(
        eligible.payload.model_copy(update={"owner_ratification_ref": "e" * 64})
    )
    assert bare.payload.owner_ratification_ref == "e" * 64  # unpersistable by design


def test_launch_and_preview_verify_the_frozen_regime_authority_before_persisting(
    monkeypatch, tmp_path
) -> None:
    """Adversarial R6.1 S7: a model-bearing plan whose frozen authority is
    absent from THIS store is refused by the Preview's readiness (exact
    reason, not "available") and by the Launch handler BEFORE the charter /
    spec envelopes are persisted and before any job is spawned."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import DESCRIPTIVE_CLASSES

    preset = {
        f"{_PIPE}plan": pipeline_tab._FULL_PLAN_LABEL,
        f"{_PIPE}regime_on": True,
        f"{_PIPE}regime_classes": [*DESCRIPTIVE_CLASSES, "feature_only"],
        f"{_PIPE}regime_decision_id": "1" * 64,
        f"{_PIPE}regime_owner_id": "2" * 64,
        f"{_PIPE}regime_assessment_id": "3" * 64,
    }
    at = _preset_run(monkeypatch, tmp_path, phase="Preview", preset=preset)
    dump = _dataframe_dump(at)
    assert "blocked_capability" in dump
    assert "frozen regime authority refused" in dump
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        pipeline_tab, "_spawn_pipeline_job", lambda command: spawned.append(command) or 4242
    )
    at = _preset_run(monkeypatch, tmp_path, phase="Launch", preset=preset)
    launch = next(b for b in at.button if b.label == "Freeze Pipeline Specification and Launch")
    launch.click().run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "frozen regime authority refused" in errors
    assert spawned == []
    assert not list(tmp_path.rglob("pipeline_specs"))
    assert not list(tmp_path.rglob("charters"))
    assert "launched detached" not in " ".join(str(s.value) for s in at.success)
