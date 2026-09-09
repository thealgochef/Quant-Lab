"""Explicit R6 promotion from checksum-verified temporary synthetic evidence."""

import json
import shutil
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    build_regime_oos_assignment_artifact,
    candidate_as_of_frame,
    candidate_fold_oos_assignment,
    consulted_assignment_frame,
    save_regime_oos_assignment,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    load_regime_fit_assignments,
    load_regime_promotion,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import load_owner_decision
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PipelineSemanticIdentity,
    PipelineStageResultEnvelope,
    PipelineStageResultPayload,
    QuantLabPipelineStage,
)
from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import save_research_labels
from alpha_lab.agents.data_infra.ifvg.search.research_regimes import (
    promote_research_regime,
    read_research_regime_eligibility,
)
from alpha_lab.agents.data_infra.ifvg.search.research_runs import (
    ResearchGroupEnvelope,
    ResearchGroupPayload,
    ResearchRequest,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
from tests.agents.data_infra.ifvg.test_regime_supervised_studies import lane  # noqa: F401
from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture


def _save_stage(root, semantic_id, stage, outputs, filename, record):
    envelope = PipelineStageResultEnvelope.from_payload(
        PipelineStageResultPayload(
            pipeline_semantic_id=semantic_id, stage=stage, output_artifact_ids=tuple(outputs)
        )
    )
    save_or_reuse_envelope(
        root,
        "pipeline_stage_results",
        envelope,
        extra_files={filename: json.dumps(record, sort_keys=True).encode()},
    )
    return envelope


@pytest.fixture(scope="module")
def reviewed_lane(lane, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("review_regime") / "store"
    shutil.copytree(lane["root"], root)
    fixture = build_pipeline_fixture(root.parent, regime_study="candidate")
    request = fixture["regime_request"].model_copy(
        update={"comparison_classes_requested": ("cohort_descriptive", "stratified_frontier")}
    )
    research_request = ResearchRequest(
        display_name="Synthetic review",
        evaluation_start=lane["fixture"].trading_days[0],
        evaluation_end=lane["fixture"].trading_days[-1],
        regime_study=request,
    )
    subject_id, approval_id = "a" * 64, "b" * 64
    semantic = PipelineSemanticIdentity.from_payload(
        fixture["semantic"].payload.model_copy(
            update={
                "source_artifact_ids": (subject_id, approval_id),
                "regime_study": request,
                "feature_bundle_ids": research_request.feature_bundle_ids,
            }
        )
    )
    save_or_reuse_envelope(root, "pipeline_specs", semantic)
    pipeline_id = semantic.pipeline_semantic_id
    group = ResearchGroupEnvelope.from_payload(
        ResearchGroupPayload(
            plan_id="c" * 64,
            research_approval_id=approval_id,
            display_name="Synthetic review",
            request_json=research_request.model_dump_json(),
            cells_json=json.dumps(
                [{"subject_id": subject_id, "pipeline_semantic_id": pipeline_id}]
            ),
        )
    )
    save_or_reuse_envelope(root, "research_groups", group)
    labels = lane["labels"].copy()
    labels["is_warmup"] = False
    label = save_research_labels(
        root,
        subject=SimpleNamespace(
            subject_id=subject_id, evaluation_dates=lane["fixture"].trading_days
        ),
        view=lane["fixture"].view,
        labels=labels,
        label_policy_id=lane["label_policy_id"],
    )
    s07 = _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S07_DERIVE_LABELS,
        (label.research_label_id, label.payload.label_artifact_id),
        "research_labels.json",
        {
            "research_label_id": label.research_label_id,
            "label_artifact_id": label.payload.label_artifact_id,
            "research_subject_id": subject_id,
        },
    )
    schedule_id = lane["schedule"].fold_schedule_id
    fold_set_id = lane["fold_set"].fold_set_artifact_id
    s08 = _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S08_BUILD_FOLDS,
        (schedule_id, fold_set_id),
        "fold_sample_adequacy.json",
        {"fold_schedule_id": schedule_id, "candidate_fold_set_artifact_id": fold_set_id},
    )
    fits = {
        fit.fold_index: load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id)
        for fit in lane["run"].fold_fits
    }
    view = lane["fixture"].view
    stage = lane["protocol"].payload.observation_stage
    oos_frame = candidate_fold_oos_assignment(
        consulted_assignment_frame(fits), tuple(view.frame["candidate_id"])
    )
    oos, data = build_regime_oos_assignment_artifact(
        oos_frame,
        protocol=lane["protocol"],
        verified_fit_assignments=fits,
        regime_fold_set_id=lane["fold_set"].payload.fold_set_id,
        fold_schedule_id=schedule_id,
        candidate_as_of=candidate_as_of_frame(view.frame, stage=stage),
        candidate_as_of_source_ref="bundle_feature_view:" + view.view_id,
        candidate_as_of_stage=stage,
    )
    save_regime_oos_assignment(root, oos, data)
    fit_ids = lane["run"].assessment.payload.regime_fit_ids
    record = {
        "resolved_regime_protocol_id": lane["protocol"].resolved_regime_protocol_id,
        "regime_capability_assessment_id": lane["run"].assessment.regime_capability_assessment_id,
        "regime_fit_ids": list(fit_ids),
        "regime_oos_assignment_id": oos.regime_oos_assignment_id,
        "fold_set_artifact_id": fold_set_id,
        "fold_schedule_id": schedule_id,
    }
    s09 = _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S09_TRAIN_MODELS,
        (record["regime_capability_assessment_id"], *fit_ids, oos.regime_oos_assignment_id),
        "regime_run.json",
        {"S09a": record},
    )
    ready_id = lane["ready"].regime_promotion_decision_id
    diagnostics = {
        **record,
        "request": request.model_dump(mode="json"),
        "candidate_fold_set_artifact_id": fold_set_id,
        "regime_fold_set_artifact_id": fold_set_id,
        "decisions": [{"regime_promotion_decision_id": ready_id}],
        "final_status": "stratification_ready",
    }
    s10 = _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
        (ready_id,),
        "regime_diagnostics.json",
        diagnostics,
    )
    return {
        "root": root,
        "group_id": group.research_group_id,
        "pipeline_id": pipeline_id,
        "s07": s07,
        "s08": s08,
        "s09": s09,
        "s10": s10,
        "diagnostics": diagnostics,
    }


@pytest.fixture
def evidence(reviewed_lane, tmp_path):
    root = tmp_path / "store"
    shutil.copytree(reviewed_lane["root"], root)
    return {**reviewed_lane, "root": root}


def test_readiness_is_read_only_and_owner_approval_binds_exact_evidence(evidence, monkeypatch):
    import joblib
    from sklearn.cluster import KMeans

    def never(*args, **kwargs):
        raise AssertionError("eligibility review fitted or deserialized a model")

    monkeypatch.setattr(joblib, "load", never)
    monkeypatch.setattr(KMeans, "fit", never)
    root, group_id, pipeline_id = (evidence[name] for name in ("root", "group_id", "pipeline_id"))
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    review = read_research_regime_eligibility(root, group_id, pipeline_id)
    assert review["ready"], review["blockers"]
    assert review["requested_bootstrap_refits"] == review["applied_bootstrap_refits"] == 3
    assert review["sample_floors"]["minimum_training_observations"] == 150
    assert before == {path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    receipt = promote_research_regime(
        root,
        group_id,
        pipeline_id,
        author="Synthetic test reviewer",
        approval_statement="Approve these exact synthetic protocol values.",
        review_id=review["review_id"],
    )
    owner = load_owner_decision(root, receipt["owner_decision_artifact_id"])
    promotion = load_regime_promotion(root, receipt["regime_promotion_decision_id"])
    assert owner.payload.provenance == "owner_signed"
    assert owner.payload.authorized_transitions == ("stratification_ready->feature_eligible",)
    assert review["review_id"] in owner.payload.reviewed_evidence_refs
    assert (
        promotion.payload.previous_decision_ref
        == review["evidence"]["stratification_ready_decision_id"]
    )
    assert promotion.payload.capability_assessment_ref == review["assessment_id"]
    assert receipt["required_capability_assessment_id"] == review["assessment_id"]
    assert promotion.payload.role.value == "feature_generator"
    assert promotion.payload.status.value == "feature_eligible"


def test_unknown_cell_blank_approval_and_changed_review_refuse(evidence):
    root, group_id, pipeline_id = (evidence[name] for name in ("root", "group_id", "pipeline_id"))
    assert not read_research_regime_eligibility(root, group_id, "f" * 64)["ready"]
    for author, statement, review_id in (
        ("", "approve", None),
        ("owner", "", None),
        ("owner", "approve", "f" * 64),
    ):
        with pytest.raises(PermissionError):
            promote_research_regime(
                root,
                group_id,
                pipeline_id,
                author=author,
                approval_statement=statement,
                review_id=review_id,
            )


@pytest.mark.parametrize("reason", ["corrupt_sidecar", "ambiguous_s10", "absent_labels"])
def test_incomplete_or_ambiguous_precursor_cannot_promote(evidence, reason):
    root, group_id, pipeline_id = (evidence[name] for name in ("root", "group_id", "pipeline_id"))
    if reason == "corrupt_sidecar":
        path = root / "pipeline_stage_results" / evidence["s09"].stage_result_id / "regime_run.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif reason == "ambiguous_s10":
        _save_stage(
            root,
            pipeline_id,
            QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
            (*evidence["s10"].payload.output_artifact_ids, "f" * 64),
            "regime_diagnostics.json",
            evidence["diagnostics"],
        )
    else:
        # Remove one explicitly named temporary fixture sidecar, not source data.
        sidecar = next((root / "research_labels").glob("*/labels.parquet"))
        sidecar.unlink()
    review = read_research_regime_eligibility(root, group_id, pipeline_id)
    assert not review["ready"] and review["blockers"]
    with pytest.raises(PermissionError, match="blocked"):
        promote_research_regime(
            root, group_id, pipeline_id, author="owner", approval_statement="Approve."
        )


def test_passing_coverage_without_stability_cannot_authorize_features(evidence):
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        RegimeCapabilityAssessmentEnvelope,
        RegimePromotionDecisionEnvelope,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        load_regime_assessment,
        persist_regime_assessment,
        persist_regime_promotion,
    )

    root, group_id, pipeline_id = (evidence[name] for name in ("root", "group_id", "pipeline_id"))
    record = dict(evidence["diagnostics"])
    assessment = load_regime_assessment(root, record["regime_capability_assessment_id"])
    failed = RegimeCapabilityAssessmentEnvelope.from_payload(
        assessment.payload.model_copy(
            update={
                "gates_passed": False,
                "gate_failures": ("synthetic_stability_failure",),
                "stability": assessment.payload.stability.model_copy(
                    update={
                        "stability_gates_passed": False,
                        "gate_failures": ("synthetic_stability_failure",),
                    }
                ),
            }
        )
    )
    persist_regime_assessment(root, failed)
    previous = load_regime_promotion(root, record["decisions"][0]["regime_promotion_decision_id"])
    ready = RegimePromotionDecisionEnvelope.from_payload(
        previous.payload.model_copy(
            update={"capability_assessment_ref": failed.regime_capability_assessment_id}
        )
    )
    # Stratification requires coverage + OOS and can legitimately exist when
    # stability has failed. It must never imply eligibility for model features.
    persist_regime_promotion(root, ready)
    for key in ("s09", "s10"):
        source = (root / "pipeline_stage_results" / evidence[key].stage_result_id).resolve()
        target = (root / ("fixture_prior_" + key)).resolve()
        assert root.resolve() in source.parents and target.parent == root.resolve()
        source.rename(target)
    record["regime_capability_assessment_id"] = failed.regime_capability_assessment_id
    record["decisions"] = [{"regime_promotion_decision_id": ready.regime_promotion_decision_id}]
    _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S09_TRAIN_MODELS,
        (
            failed.regime_capability_assessment_id,
            *record["regime_fit_ids"],
            record["regime_oos_assignment_id"],
        ),
        "regime_run.json",
        {"S09a": record},
    )
    _save_stage(
        root,
        pipeline_id,
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
        (ready.regime_promotion_decision_id,),
        "regime_diagnostics.json",
        record,
    )
    review = read_research_regime_eligibility(root, group_id, pipeline_id)
    assert not review["ready"]
    assert review["gates"]["coverage_gates_passed"]
    assert not review["gates"]["stability_gates_passed"]
    assert "synthetic_stability_failure" in review["blockers"]
    before = set((root / "owner_decisions").iterdir())
    with pytest.raises(PermissionError, match="blocked"):
        promote_research_regime(
            root, group_id, pipeline_id, author="owner", approval_statement="Approve."
        )
    assert set((root / "owner_decisions").iterdir()) == before
