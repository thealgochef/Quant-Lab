"""Research UI authority, exact source selection and partial-evidence routing."""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import ifvg_research_pipeline as ui  # noqa: E402

CORE_A, CORE_B = "a" * 64, "b" * 64


def _source(core_id, label, target, cost):
    return {
        "core_replay_id": core_id,
        "label": label,
        "profile_hash": "c" * 64,
        "v2_dataset_id": "d" * 64,
        "tp_r_multiple": target,
        "cost_points": cost,
        "replay_dates": ["2026-01-12", "2026-01-13", "2026-01-14"],
        "warmup_dates": ["2026-01-12"],
        "evaluation_dates": ["2026-01-13", "2026-01-14"],
        "blockers": [],
    }


def _configuration_app():
    import ifvg_research_pipeline
    import streamlit as st

    ifvg_research_pipeline.render_research_configuration(
        st, roots=ifvg_research_pipeline._TEST_ROOTS
    )


@pytest.fixture
def configuration(monkeypatch, tmp_path):
    calls = []
    sources = [_source(CORE_A, "Parent 240", 1.0, 0.514), _source(CORE_B, "Parent 480", 2.0, 0.75)]

    def preflight(root, source_ids, spec):
        request = {"source_core_replay_ids": list(source_ids), "study_spec": spec}
        plan_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        calls.append(("preflight", request))
        return {
            "plan_id": plan_id,
            "request": request,
            "subjects": [source for source in sources if source["core_replay_id"] in source_ids],
            "cells": [
                {"label": "Parent 240", "lane": "R5", "core_replay_id": source_id}
                for source_id in source_ids
            ],
            "ready": True,
            "blockers": [],
            "warnings": [],
            "planned_stages": ["00_validate_inputs", "09_train_models", "15_verify_and_publish"],
        }

    def freeze(root, plan, *, authorization_statement, author):
        calls.append(("freeze", plan, authorization_statement, author))
        return {"group_id": "e" * 64}

    def launch(*args):
        calls.append(("launch", args))
        return {"group_id": "e" * 64}

    api = SimpleNamespace(
        list_source_subjects=lambda root: sources,
        build_research_preflight=preflight,
        freeze_research_group=freeze,
        launch_research_group=launch,
    )
    monkeypatch.setattr(ui, "_research_api", lambda: api)
    monkeypatch.setattr(
        ui,
        "_TEST_ROOTS",
        {"store_root": tmp_path / "store", "pipeline_state_root": tmp_path / "state"},
        raising=False,
    )
    return api, calls


def _selected_app():
    at = AppTest.from_function(_configuration_app, default_timeout=60).run()
    assert not at.exception
    at.multiselect(key=ui._RESEARCH + "sources").set_value([CORE_A]).run()
    assert not at.exception
    return at


def test_render_and_review_do_not_authorize_or_launch(configuration):
    _, calls = configuration
    at = _selected_app()
    at.run()
    assert not at.exception
    assert all(call[0] == "preflight" for call in calls)
    request = calls[-1][1]
    assert request["source_core_replay_ids"] == [CORE_A]
    regime = request["study_spec"]["regime_study"]
    assert regime["panel_interval_seconds"] == 300
    assert regime["resolved_cluster_count"] == 3
    assert regime["comparison_classes_requested"] == ["cohort_descriptive"]
    assert regime["owner_decision_artifact_id"] is None
    assert not request["study_spec"]["mbp1_comparison"]
    assert next(
        button for button in at.button if button.label == "Authorize and start research"
    ).disabled


def test_authorization_binds_exact_plan_and_resets_when_sources_change(configuration):
    _, calls = configuration
    at = _selected_app()
    next(value for value in at.text_input if value.label == "Reviewer name").set_value(
        "Research owner"
    ).run()
    next(
        value for value in at.checkbox if value.label == "I authorize this exact research plan"
    ).check().run()
    assert not next(
        button for button in at.button if button.label == "Authorize and start research"
    ).disabled
    at.multiselect(key=ui._RESEARCH + "sources").set_value([CORE_B]).run()
    assert not at.exception
    assert not next(
        value for value in at.checkbox if value.label == "I authorize this exact research plan"
    ).value
    assert next(
        button for button in at.button if button.label == "Authorize and start research"
    ).disabled
    assert not any(call[0] in {"freeze", "launch"} for call in calls)


def test_explicit_authorize_and_start_freezes_then_dispatches_once(configuration):
    _, calls = configuration
    at = _selected_app()
    next(value for value in at.text_input if value.label == "Reviewer name").set_value(
        "Research owner"
    ).run()
    next(
        value for value in at.checkbox if value.label == "I authorize this exact research plan"
    ).check().run()
    next(
        button for button in at.button if button.label == "Authorize and start research"
    ).click().run()
    assert not at.exception
    mutations = [call for call in calls if call[0] != "preflight"]
    assert [call[0] for call in mutations] == ["freeze", "launch"]
    frozen = mutations[0]
    assert frozen[1]["request"]["source_core_replay_ids"] == [CORE_A]
    assert frozen[1]["plan_id"] in frozen[2] and CORE_A in frozen[2]
    assert frozen[3] == "Research owner"
    assert at.session_state["ifvg_workspace_research_group"] == "e" * 64


def test_missing_regime_authority_does_not_fall_back_to_descriptive(configuration, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg import study_providers

    _, calls = configuration
    monkeypatch.setattr(study_providers, "list_catalogued_envelope_ids", lambda *args: ())
    at = _selected_app()
    previous_count = len(calls)
    at.radio(key=ui._RESEARCH + "regime_mode").set_value(
        "Use approved regimes as model features"
    ).run()
    assert not at.exception
    assert len(calls) == previous_count
    assert any("No approved regime decision" in item.value for item in at.error)
    assert not any(item.label == "Authorize and start research" for item in at.button)


def test_linked_regime_features_plan_uses_materializable_base_bundle(configuration, monkeypatch):
    from ifvg_pipeline_tab import _research_stage_plan

    from alpha_lab.agents.data_infra.ifvg import study_providers
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
    from alpha_lab.agents.data_infra.ifvg.search.research_data import LABEL_POLICY
    from alpha_lab.agents.data_infra.ifvg.search.research_runs import ResearchRequest

    _, calls = configuration
    monkeypatch.setattr(
        study_providers, "list_catalogued_envelope_ids", lambda *args: (("d" * 64, "Approved"),)
    )
    at = _selected_app()
    at.radio(key=ui._RESEARCH + "regime_mode").set_value(
        "Use approved regimes as model features"
    ).run()
    assert not at.exception
    spec = calls[-1][1]["study_spec"]
    request = ResearchRequest.model_validate(spec)
    regime = RegimeStudyRequest.model_validate(spec["regime_study"])
    assert regime.supervised_bundle_key == "B0_CORE"
    assert regime.authority_refs == ("d" * 64,) * 3
    assert (
        regime.stage_plan_problems(
            feature_bundle_ids=request.feature_bundle_ids,
            stage_values=tuple(stage.value for stage in _research_stage_plan()),
            label_policy_id=LABEL_POLICY,
            model_protocol_id=request.model_protocol_id,
        )
        == []
    )


def test_candidate_regime_grain_uses_numeric_registry_and_registered_floor(configuration):
    from ifvg_pipeline_tab import _regime_candidate_numeric_inputs, _research_stage_plan

    from alpha_lab.agents.data_infra.ifvg.context_model import categorical_features_for
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import sample_adequacy_minimum
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
    from alpha_lab.agents.data_infra.ifvg.search.research_runs import ResearchRequest

    _, calls = configuration
    at = _selected_app()
    at.radio(key=ui._RESEARCH + "regime_grain").set_value("Candidate entry decisions").run()
    assert not at.exception
    spec = calls[-1][1]["study_spec"]
    request = ResearchRequest.model_validate(spec)
    regime = RegimeStudyRequest.model_validate(spec["regime_study"])
    assert regime.observation_granularity.value == "candidate_stage_row"
    assert regime.panel_interval_seconds is None and regime.panel_as_of_policy_id is None
    assert regime.input_feature_bundle_key == "B0_CORE"
    assert not categorical_features_for(regime.resolved_input_features)
    options = at.multiselect(key=ui._RESEARCH + "regime_inputs_B0_CORE").options
    assert options == list(_regime_candidate_numeric_inputs("B0_CORE"))
    assert sample_adequacy_minimum(regime.observation_granularity) == 150
    assert any("150 training observations per fold" in item.value for item in at.caption)
    assert (
        regime.stage_plan_problems(
            feature_bundle_ids=request.feature_bundle_ids,
            stage_values=tuple(stage.value for stage in _research_stage_plan()),
            label_policy_id="configured-labels",
            model_protocol_id=request.model_protocol_id,
        )
        == []
    )


def test_linked_followup_preserves_precursor_numeric_protocol(configuration):
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest

    _, calls = configuration
    precursor = RegimeStudyRequest(
        observation_granularity="candidate_stage_row",
        observation_stage="entry_decision",
        input_feature_bundle_key="B0_CORE",
        resolved_input_features=("distance_to_htf_ticks", "opposing_size_ticks"),
        winsorization_policy="clip_p01_p99_train_fitted_v1",
        bootstrap_refits=17,
        resolved_cluster_count=3,
        stratified_reporting_requested=True,
        comparison_classes_requested=("feature_only",),
        supervised_bundle_key="B0_CORE",
        regime_promotion_decision_id="c" * 64,
        owner_decision_artifact_id="d" * 64,
        required_capability_assessment_id="e" * 64,
    ).model_dump(mode="json")
    at = AppTest.from_function(_configuration_app, default_timeout=60)
    at.session_state["ifvg_research_seed"] = {
        "source_core_replay_ids": [CORE_A],
        "regime_study": precursor,
    }
    at.run()
    assert not at.exception
    assert calls[-1][1]["study_spec"]["regime_study"] == precursor
    assert not any(item.label == "Regime observations" for item in at.radio)
    assert not any(item.label == "Requested stability refits per fold" for item in at.number_input)


def test_failed_preflight_cannot_be_authorized(configuration):
    api, calls = configuration
    original = api.build_research_preflight

    def blocked(*args):
        result = original(*args)
        result.update(ready=False, blockers=["The exact saved source changed."])
        return result

    api.build_research_preflight = blocked
    at = _selected_app()
    assert any("exact saved source changed" in item.value for item in at.error)
    assert not any(item.label == "I authorize this exact research plan" for item in at.checkbox)
    assert not any(call[0] in {"freeze", "launch"} for call in calls)


def test_inherited_targets_and_costs_remain_per_subject():
    subjects = [_source(CORE_A, "Parent 240", 1.0, 0.514), _source(CORE_B, "Parent 480", 2.0, 0.75)]
    rows = ui._inherited_source_rows(subjects)
    assert [(row["Target (R)"], row["Round-trip cost (points)"]) for row in rows] == [
        (1.0, 0.514),
        (2.0, 0.75),
    ]
    assert ui._default_evaluation_window(subjects) == (date(2026, 1, 13), date(2026, 1, 14))
    subjects[1]["evaluation_dates"] = ["2026-02-02"]
    assert ui._default_evaluation_window(subjects) is None


def test_research_preset_omits_replay_and_prop_stages():
    from ifvg_pipeline_tab import _research_stage_plan

    positions = [int(stage.value.split("_", 1)[0]) for stage in _research_stage_plan()]
    assert positions == [*range(11), 14, 15]


def _group_app():
    import ifvg_research_pipeline
    import streamlit as st

    ifvg_research_pipeline.render_research_group(
        st, roots=ifvg_research_pipeline._TEST_ROOTS, group_id="e" * 64
    )


def test_incomplete_cells_expose_existing_stage_evidence(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg import study_providers

    observed = []
    cell = {
        "core_replay_id": CORE_A,
        "pipeline_semantic_id": "f" * 64,
        "label": "Parent 240",
        "lane": "R5B",
        "status": "failed",
        "reason": "Model fit unavailable",
        "state": {
            "stages": {"05_materialize_feature_views": {"in_plan": True, "status": "completed"}},
            "research_evidence": {
                "status": "insufficient_evidence",
                "oos_rows": 0,
                "valid_oos_folds": 0,
                "regime_gates_passed": False,
                "reasons": ["No valid out-of-sample folds"],
            },
        },
    }
    api = SimpleNamespace(
        read_research_group=lambda *args: {
            "display_name": "Partial evidence",
            "status": "failed",
            "cells": [cell],
        }
    )
    monkeypatch.setattr(ui, "_research_api", lambda: api)
    monkeypatch.setattr(
        ui,
        "_TEST_ROOTS",
        {"store_root": tmp_path / "store", "pipeline_state_root": tmp_path / "state"},
        raising=False,
    )
    monkeypatch.setattr(study_providers, "load_ladder_diagnostics", lambda *args: None)
    monkeypatch.setattr(
        study_providers,
        "mbp1_stage_evidence_defaults",
        lambda *args: observed.append(args[2]) or ({"coverage_report_id": "c" * 64}, None),
    )
    monkeypatch.setattr(study_providers, "regime_stage_evidence_defaults", lambda *args: ({}, None))
    import ifvg_mbp1_panels

    monkeypatch.setattr(
        ifvg_mbp1_panels,
        "render_mbp1_order_flow",
        lambda st, *, roots, default_ids: st.write(
            "Persisted coverage " + default_ids["coverage_report_id"]
        ),
    )
    at = AppTest.from_function(_group_app, default_timeout=60).run()
    assert not at.exception
    assert observed == [cell["pipeline_semantic_id"]]
    assert any("incomplete" in value.value for value in at.warning)
    assert any("Research evidence is insufficient" in value.value for value in at.warning)
    assert any("Operational advisory" in value.value for value in at.caption)
    at.radio(key=ui._RESEARCH + "evidence_panel").set_value("Order flow").run()
    assert not at.exception
    assert any("Persisted coverage " + "c" * 64 in value.value for value in at.markdown)


def test_partial_model_input_checkpoints_are_exact_loaded(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.ml import research_evidence

    request_id = "a" * 64
    (tmp_path / "research_model_inputs" / request_id).mkdir(parents=True)
    seen = []
    monkeypatch.setattr(
        research_evidence,
        "load_research_inputs",
        lambda root, identity: (
            seen.append(identity)
            or {
                "request": {"runner": "ladder"},
                "candidate_features": pd.DataFrame({"candidate_id": ["candidate-1"]}),
                "labels": pd.DataFrame({"candidate_id": ["candidate-1"], "target_r": [2.0]}),
                "fold_assignment": pd.DataFrame(),
                "fold_definitions": [],
            }
        ),
    )
    state = {
        "stages": {
            "09_train_models": {
                "status": "failed",
                "research_model_input_ids": [request_id, "../outside"],
            }
        }
    }
    saved = ui._load_research_tables(tmp_path, state)
    assert seen == [request_id]
    assert saved["model_inputs"][0]["labels"].iloc[0]["target_r"] == 2.0
    assert not saved["errors"]


@pytest.mark.parametrize("sidecar", ["typed", "legacy", "corrupt"])
def test_mbp_provider_prefers_verified_typed_controlled_study(monkeypatch, tmp_path, sidecar):
    from alpha_lab.agents.data_infra.ifvg import study_providers
    from alpha_lab.agents.data_infra.ifvg.search import pipeline, store

    stage_id, request_id, study_id = "a" * 64, "b" * 64, "c" * 64
    state = {
        "stages": {
            "05_materialize_feature_views": {"stage_result_id": stage_id},
            "09_train_models": {
                "stage_result_id": "d" * 64,
                "output_artifact_ids": [study_id if sidecar == "legacy" else request_id, study_id],
            },
        }
    }
    monkeypatch.setattr(pipeline, "read_pipeline_state", lambda *args: state)
    monkeypatch.setattr(
        study_providers,
        "load_sidecar_bytes",
        lambda *args: json.dumps({"__mbp1_evidence__": {"coverage_report_id": "e" * 64}}).encode(),
    )

    def load(*args):
        if sidecar == "corrupt":
            raise store.SearchStoreError("manifest mismatch")
        return {"controlled_feature_study_id": study_id} if sidecar == "typed" else None

    monkeypatch.setattr(store, "load_json_sidecar", load)
    defaults, note = study_providers.mbp1_stage_evidence_defaults(tmp_path, tmp_path, "f" * 64)
    if sidecar == "corrupt":
        assert note and "controlled_study_id" not in defaults
    else:
        assert not note
        assert defaults["controlled_study_id"] == study_id


def _eligibility_app():
    import ifvg_research_pipeline
    import streamlit as st

    ifvg_research_pipeline._render_regime_feature_authorization(
        st,
        ifvg_research_pipeline._TEST_ROOTS,
        ifvg_research_pipeline._TEST_ELIGIBILITY_GROUP,
        {"pipeline_semantic_id": "f" * 64, "core_replay_id": "a" * 64},
    )


@pytest.fixture
def eligibility(monkeypatch, tmp_path):
    review = {
        "ready": True,
        "blockers": [],
        "assessment_id": "a" * 64,
        "protocol_id": "b" * 64,
        "review_id": "c" * 64,
        "input_features": ["return_5m", "range_5m"],
        "requested_bootstrap_refits": 50,
        "applied_bootstrap_refits": 50,
        "gates": {"stability": True},
        "sample_floors": {"minimum_train_observations": 30},
        "evidence": {"K": 3},
    }
    calls = []

    def promote(*args, **kwargs):
        calls.append(kwargs)
        return {
            "regime_promotion_decision_id": "d" * 64,
            "owner_decision_artifact_id": "e" * 64,
            "required_capability_assessment_id": "a" * 64,
        }

    monkeypatch.setattr(
        ui,
        "_research_api",
        lambda: SimpleNamespace(
            read_research_regime_eligibility=lambda *args: review,
            promote_research_regime=promote,
        ),
    )
    monkeypatch.setattr(ui, "_TEST_ROOTS", {"store_root": tmp_path}, raising=False)
    monkeypatch.setattr(
        ui,
        "_TEST_ELIGIBILITY_GROUP",
        {
            "group_id": "1" * 64,
            "display_name": "Regimes",
            "study_spec": {"evaluation_start": "2026-01-13", "evaluation_end": "2026-01-14"},
        },
        raising=False,
    )
    return review, calls


def _open_eligibility():
    at = AppTest.from_function(_eligibility_app, default_timeout=60).run()
    next(
        item for item in at.button if item.label == "Review regime feature eligibility"
    ).click().run()
    assert not at.exception
    return at


def test_regime_eligibility_review_does_not_approve_and_changes_reset_consent(eligibility):
    review, calls = eligibility
    at = _open_eligibility()
    assert not calls
    at.text_input[0].set_value("Research owner").run()
    at.checkbox[0].check().run()
    assert not next(
        item for item in at.button if item.label == "Approve regime feature use"
    ).disabled
    review["review_id"] = "2" * 64
    at.run()
    assert not at.exception
    assert not at.checkbox[0].value
    assert next(item for item in at.button if item.label == "Approve regime feature use").disabled
    assert not calls


def test_regime_eligibility_failed_gates_cannot_be_approved(eligibility):
    review, calls = eligibility
    review.update(ready=False, blockers=["Insufficient bootstrap stability evidence"])
    at = _open_eligibility()
    assert any("Insufficient bootstrap" in item.value for item in at.error)
    assert not at.checkbox
    assert not any(item.label == "Approve regime feature use" for item in at.button)
    assert not calls


def test_explicit_eligibility_approval_seeds_exact_followup_without_launch(eligibility):
    review, calls = eligibility
    at = _open_eligibility()
    at.text_input[0].set_value("Research owner").run()
    at.checkbox[0].check().run()
    next(item for item in at.button if item.label == "Approve regime feature use").click().run()
    assert not at.exception
    assert len(calls) == 1
    assert calls[0]["review_id"] == review["review_id"]
    assert CORE_A in calls[0]["approval_statement"]
    seed = at.session_state["ifvg_research_seed"]
    assert seed["source_core_replay_ids"] == [CORE_A]
    assert seed["evaluation_start"] == "2026-01-13"
    assert seed["base_bundle_key"] == "B0_CORE" and not seed["mbp1_comparison"]
    assert seed["regime_study"]["owner_decision_artifact_id"] == "e" * 64
    assert seed["regime_study"]["supervised_bundle_key"] == "B0_CORE"
    assert at.session_state["ifvg_workspace_screen"] == "research_new"


def test_mbp_receipt_preview_and_import_bind_scoped_subject(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search import research_mbp1, research_subject

    calls = []
    subject = SimpleNamespace(subject_id="c" * 64)
    monkeypatch.setattr(research_subject, "bind_research_subject", lambda *args, **kwargs: subject)
    monkeypatch.setattr(
        research_mbp1,
        "import_research_mbp1_receipt",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"coverage_evidence_id": "d" * 64},
    )
    roots = {"store_root": tmp_path / "store", "repo_root": tmp_path}
    row = {"core_replay_id": CORE_A, "subject_id": "c" * 64, "evaluation_dates": ["2026-01-13"]}
    receipt = {"source_document": {}, "owner_review": {}}
    ui._import_mbp_receipt(roots, row, receipt, dry_run=True)
    ui._import_mbp_receipt(roots, row, receipt, dry_run=False)
    assert [call[1]["dry_run"] for call in calls] == [True, False]
    assert all(call[0][0] is subject for call in calls)
    row["subject_id"] = "e" * 64
    with pytest.raises(ValueError, match="differs from the reviewed"):
        ui._import_mbp_receipt(roots, row, receipt, dry_run=False)
    assert len(calls) == 2
