"""R6.1 workstream E — the regime study inside the 16-stage pipeline (plan
§6.E; D1, D4, D14; §9.1 ``test_model_bearing_request_binds_exact_promotion_and_owner_artifacts``,
``test_s14_performs_zero_fitting``; §9.2 ``test_pipeline_regime.py``).

Descriptive candidate / panel studies reach the 16 terminal states, persist
every regime artifact through the verified stores, derive the deterministic
S10 decisions from their own assessment, and REUSE every stage on a second
attempt; a panel plan without the verified replay-chart seam refuses at S00;
readiness blocks the regime stages with the exact reason; model-bearing
requests freeze the exact promotion / owner / assessment ids into the
semantic identity and refuse absent or mismatched authority.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
    load_context_bar_panel_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
    load_regime_fold_features,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    load_regime_oos_assignment,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    load_regime_assessment,
    load_regime_promotion,
    load_regime_protocol,
    persist_regime_promotion,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_study import (
    RegimeStudyRequest,
    RegimeStudyStagePlanFacts,
    regime_study_block_reason,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import _child_evaluation_envelope
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    synthetic_owner_decision_fixture,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PipelineSemanticIdentity,
    PipelineSemanticSpecPayload,
    QuantLabPipelineStage,
    StageStatus,
    derive_stage_plan_readiness,
    read_pipeline_state,
    run_pipeline,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline_regime import (
    PANEL_CHART_SELECTION_POLICY_ID,
)
from alpha_lab.agents.data_infra.ifvg.search.store import has_envelope, load_sidecar_bytes
from tests.agents.ifvg_search.pipeline_fixture import (
    LABEL_POLICY_ID,
    build_pipeline_fixture,
    build_regime_study_request,
    foreign_context_bar_source,
)

S = QuantLabPipelineStage


def _run(fixture, **overrides):
    return run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
        **overrides,
    )


def _sidecar(fixture, stage: QuantLabPipelineStage, name: str) -> dict:
    state = read_pipeline_state(fixture["state_root"], fixture["result"].pipeline_semantic_id)
    entry = state["stages"][stage.value]
    return json.loads(
        load_sidecar_bytes(
            fixture["store_root"], "pipeline_stage_results", entry["stage_result_id"], name
        )
    )


def _stage(fixture, stage: QuantLabPipelineStage) -> dict:
    state = read_pipeline_state(fixture["state_root"], fixture["result"].pipeline_semantic_id)
    return state["stages"][stage.value]


_APPROVED_AT = "2026-08-26T00:00:00+00:00"
_DECIDED_AT = "2026-08-28T13:00:00+00:00"


def _frozen_authority(completed, *, approved_at: str = _APPROVED_AT, decided_at: str = _DECIDED_AT):
    """The owner half of the two-pass workflow over a completed DESCRIPTIVE run:
    a synthetic owner decision bound to the run's protocol + assessment and the
    FEATURE_ELIGIBLE promotion (synthetic scope) — returns the exact
    ``(promotion decision id, owner artifact id, assessment id)`` a
    model-bearing request freezes, plus the loaded protocol / assessment."""

    root = completed["store_root"]
    diagnostics = _sidecar(
        completed, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    assert diagnostics["final_status"] == "stratification_ready", diagnostics["final_status"]
    protocol = load_regime_protocol(root, diagnostics["resolved_regime_protocol_id"])
    assessment = load_regime_assessment(root, diagnostics["regime_capability_assessment_id"])
    ready = load_regime_promotion(
        root, diagnostics["decisions"][-1]["regime_promotion_decision_id"]
    )
    owner = synthetic_owner_decision_fixture(
        root,
        protocol=protocol,
        assessment=assessment,
        approved_at=approved_at,
        effective_from=approved_at,
    )
    eligible = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=ready.regime_promotion_decision_id,
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at=decided_at,
        )
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    authority = (
        eligible.regime_promotion_decision_id,
        owner.owner_decision_artifact_id,
        assessment.regime_capability_assessment_id,
    )
    return authority, owner, protocol, assessment, eligible


def _supervised_over(completed, shape: str, *, hard_id_encoding: str = "none"):
    """A model-bearing fixture frozen to ``completed``'s authority, sharing its
    tmp root: the same store (the frozen authority lives there) and — for the
    panel shapes — the same replay-chart seam directory."""

    authority, *_ = _frozen_authority(completed)
    tmp_root = completed["store_root"].parent
    supervised = build_pipeline_fixture(
        tmp_root,
        regime_study=shape,
        regime_authority=authority,
        regime_hard_id_encoding=hard_id_encoding,
    )
    assert supervised["store_root"] == completed["store_root"]
    return supervised


@pytest.fixture(scope="module")
def completed_candidate(tmp_path_factory):
    fixture = build_pipeline_fixture(
        tmp_path_factory.mktemp("pipeline_regime_candidate"), regime_study="candidate"
    )
    return {**fixture, "result": _run(fixture)}


@pytest.fixture(scope="module")
def completed_panel(tmp_path_factory):
    fixture = build_pipeline_fixture(
        tmp_path_factory.mktemp("pipeline_regime_panel"), regime_study="panel"
    )
    return {**fixture, "result": _run(fixture)}


def _assert_terminal(result) -> None:
    statuses = result.stage_statuses
    assert len(statuses) == 16
    terminal = {StageStatus.COMPLETED.value, StageStatus.REUSED.value, StageStatus.BLOCKED.value}
    assert set(statuses.values()) <= terminal, statuses
    blocked = [stage for stage, status in statuses.items() if status == "blocked"]
    assert blocked == [S.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value]


def test_descriptive_candidate_study_reaches_sixteen_terminal_states(completed_candidate):
    _assert_terminal(completed_candidate["result"])
    assert completed_candidate["semantic"].payload.model_protocol_id is None
    assert completed_candidate["semantic"].payload.regime_study.descriptive_classes == (
        "cohort_descriptive",
        "stratified_prop",
        "stratified_frontier",
    )


def test_s05_persists_the_view_and_the_protocol_as_the_verified_observation(
    completed_candidate,
):
    record = _sidecar(
        completed_candidate, S.S05_MATERIALIZE_FEATURE_VIEWS, "bundle_feature_views.json"
    )
    observation = record["__regime_observation__"]
    assert observation["source_kind"] == "bundle_feature_view"
    assert observation["observation_granularity"] == "candidate_stage_row"
    assert observation["frozen_authority_verified"] is False
    view_id = record["__bundle_views_persisted__"]["B0_CORE"]
    assert observation["artifact_id"] == view_id
    assert has_envelope(completed_candidate["store_root"], "bundle_feature_views", view_id)
    protocol = load_regime_protocol(
        completed_candidate["store_root"], observation["resolved_regime_protocol_id"]
    )
    assert protocol.payload.observation_granularity.value == "candidate_stage_row"
    outputs = _stage(completed_candidate, S.S05_MATERIALIZE_FEATURE_VIEWS)["output_artifact_ids"]
    assert observation["resolved_regime_protocol_id"] in outputs


def test_s06_and_s08_emit_the_adequacy_facts_and_persist_the_fold_artifacts(completed_candidate):
    preview = _sidecar(
        completed_candidate, S.S06_VALIDATE_FEATURE_COVERAGE, "regime_sample_adequacy_preview.json"
    )
    assert preview["floors"]["minimum_training_observations"] == 150
    assert preview["floors"]["floor_is_flat_per_grain"] is True
    assert preview["rows_with_inputs"] == 600
    folds = _sidecar(completed_candidate, S.S08_BUILD_FOLDS, "fold_sample_adequacy.json")
    assert folds["candidate_folds_labeled"] is True
    assert has_envelope(
        completed_candidate["store_root"], "fold_schedules", folds["fold_schedule_id"]
    )
    assert has_envelope(
        completed_candidate["store_root"], "fold_sets", folds["candidate_fold_set_artifact_id"]
    )
    assert folds["sample_adequacy_preview"]["expected_gate_outcome"] == "pass"
    outputs = _stage(completed_candidate, S.S08_BUILD_FOLDS)["output_artifact_ids"]
    assert folds["fold_schedule_id"] in outputs
    assert folds["candidate_fold_set_artifact_id"] in outputs


def test_s09a_fits_persist_and_s10_derives_stratification_ready_deterministically(
    completed_candidate,
):
    run = _sidecar(completed_candidate, S.S09_TRAIN_MODELS, "regime_run.json")["S09a"]
    assert run["gates_passed"] is True and run["gate_failures"] == []
    assert len(run["regime_fit_ids"]) >= 2
    assert "fits_reused" not in run  # an attempt fact never enters the semantic sidecar
    root = completed_candidate["store_root"]
    assessment = load_regime_assessment(root, run["regime_capability_assessment_id"])
    assert set(assessment.payload.regime_fit_ids) == set(run["regime_fit_ids"])
    oos = load_regime_oos_assignment(root, run["regime_oos_assignment_id"])
    assert oos.payload.assignment_source == "candidate_fold_oos"
    diagnostics = _sidecar(
        completed_candidate, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    assert diagnostics["authority_source"] == "s10_structural"
    assert diagnostics["final_status"] == "stratification_ready"
    assert [d["status"] for d in diagnostics["decisions"]] == [
        "descriptive_only",
        "stratification_ready",
    ]
    assert diagnostics["sub_steps"] == ["S09a"]
    assert diagnostics["decided_at"].endswith("Z")
    for decision in diagnostics["decisions"]:
        stored = load_regime_promotion(root, decision["regime_promotion_decision_id"])
        assert stored.payload.decided_at == diagnostics["decided_at"]
        assert stored.payload.owner_ratification_ref is None
    assert (
        load_regime_promotion(root, diagnostics["decisions"][-1]["regime_promotion_decision_id"])
        .payload.status
        is RegimeStatus.STRATIFICATION_READY
    )
    s15 = _stage(completed_candidate, S.S15_VERIFY_AND_PUBLISH)
    assert "gates passed" in s15["explanation"]


def test_second_attempt_reuses_every_regime_stage(completed_candidate):
    first = read_pipeline_state(
        completed_candidate["state_root"], completed_candidate["result"].pipeline_semantic_id
    )
    first_ids = {
        stage: entry["stage_result_id"]
        for stage, entry in first["stages"].items()
        if entry["stage_result_id"]
    }
    second = _run(completed_candidate, operational_retry_reason="regime reuse proof")
    second_state = read_pipeline_state(
        completed_candidate["state_root"], second.pipeline_semantic_id
    )
    second_ids = {
        stage: entry["stage_result_id"]
        for stage, entry in second_state["stages"].items()
        if entry["stage_result_id"]
    }
    assert second_ids == first_ids
    for stage, status in second.stage_statuses.items():
        if stage == S.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value:
            assert status == StageStatus.BLOCKED.value
        else:
            assert status == StageStatus.REUSED.value, (stage, status)
    run = json.loads(
        load_sidecar_bytes(
            completed_candidate["store_root"],
            "pipeline_stage_results",
            second_state["stages"][S.S09_TRAIN_MODELS.value]["stage_result_id"],
            "regime_run.json",
        )
    )["S09a"]
    # the sidecar is byte-identical across attempts (an attempt fact such as
    # fits_reused never enters it); the second attempt reproduced every fit
    # by verification and the runner marked the stage REUSED
    assert run["regime_fit_ids"]
    assert "reused by reproduction" in second_state["stages"][S.S09_TRAIN_MODELS.value][
        "explanation"
    ] or second_state["stages"][S.S09_TRAIN_MODELS.value]["status"] == "reused"


def test_descriptive_panel_study_materializes_the_panel_from_the_verified_chart(completed_panel):
    _assert_terminal(completed_panel["result"])
    record = _sidecar(completed_panel, S.S05_MATERIALIZE_FEATURE_VIEWS, "bundle_feature_views.json")
    observation = record["__regime_observation__"]
    assert observation["source_kind"] == "context_bar_panel"
    assert observation["observation_granularity"] == "context_bar_panel"
    assert observation["chart_id"] is not None
    root = completed_panel["store_root"]
    assert has_envelope(root, "context_bar_panels", observation["artifact_id"])
    assert "BP0_CONTEXT_BAR_PANEL" not in record["__bundle_views_persisted__"]
    source = completed_panel["wiring"].context_bar_source
    assert source.requested == [observation["chart_id"]]
    # adversarial R6.1 S1 / F13: the panel binds the LOADED artifact of exactly
    # the chart S05 asked for, selected under the declared policy from the S04
    # chart of the lowest core replay id, and the pair identity is recorded
    assert observation["replay_chart_artifact_id"] == observation["chart_id"]
    assert observation["chart_id"] in source.written
    assert observation["panel_chart_selection_policy_id"] == PANEL_CHART_SELECTION_POLICY_ID
    assert observation["panel_source_core_replay_id"] == min(
        row["core_replay_id"]
        for row in read_pipeline_state(
            completed_panel["state_root"], completed_panel["result"].pipeline_semantic_id
        )["children"]
    )
    assert len(observation["replay_chart_source_pair_sha256"]) == 64
    panel = load_context_bar_panel_artifact(root, observation["artifact_id"])
    assert panel.payload.replay_chart_artifact_id == observation["chart_id"]
    protocol = load_regime_protocol(root, observation["resolved_regime_protocol_id"])
    assert protocol.payload.panel_source_artifact_id == observation["artifact_id"]
    folds = _sidecar(completed_panel, S.S08_BUILD_FOLDS, "fold_sample_adequacy.json")
    assert has_envelope(root, "fold_sets", folds["panel_fold_set_artifact_id"])
    assert folds["panel_fold_set_id"] != folds["candidate_fold_set_id"]
    run = _sidecar(completed_panel, S.S09_TRAIN_MODELS, "regime_run.json")["S09a"]
    oos = load_regime_oos_assignment(root, run["regime_oos_assignment_id"])
    assert oos.payload.assignment_source == "panel_pit"
    assert oos.payload.panel_context.context_bar_panel_artifact_id == observation["artifact_id"]
    assert oos.payload.fold_schedule_id == folds["fold_schedule_id"]
    diagnostics = _sidecar(
        completed_panel, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    assert diagnostics["final_status"] in {"stratification_ready", "descriptive_only"}
    assert diagnostics["gates"]["minimum_training_observations_gate"] == 300


def test_panel_plan_without_the_verified_chart_seam_fails_s00(tmp_path):
    fixture = build_pipeline_fixture(tmp_path, regime_study="panel")
    fixture["wiring"] = dataclasses.replace(fixture["wiring"], context_bar_source=None)
    result = _run(fixture)
    assert result.stage_statuses[S.S00_VALIDATE_INPUTS.value] == StageStatus.FAILED.value
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    assert "no context_bar_source is wired" in state["stages"][S.S00_VALIDATE_INPUTS.value][
        "explanation"
    ]


def test_readiness_blocks_the_regime_stages_with_the_exact_reason(tmp_path):
    fixture = build_pipeline_fixture(tmp_path, regime_study="candidate")
    spec = fixture["semantic"].payload
    leaky = spec.model_copy(
        update={
            "regime_study": spec.regime_study.model_copy(
                update={"resolved_input_features": ("ofl_snap_entry",)}
            )
        }
    )
    report = derive_stage_plan_readiness(leaky)
    assert not report.launchable
    blocked = {entry.stage for entry in report.entries if entry.state == "blocked_capability"}
    assert blocked == {
        S.S05_MATERIALIZE_FEATURE_VIEWS,
        S.S06_VALIDATE_FEATURE_COVERAGE,
        S.S08_BUILD_FOLDS,
        S.S09_TRAIN_MODELS,
        S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
        S.S14_BUILD_FRONTIER_AND_INSIGHTS,
    }
    assert "outside the referenced feature bundle" in report.entry(S.S09_TRAIN_MODELS).reason
    assert derive_stage_plan_readiness(spec).launchable
    # the structural rules live in the semantic spec itself
    with pytest.raises(ValueError, match="must be one of the plan's feature bundles"):
        spec.model_copy(
            update={
                "regime_study": spec.regime_study.model_copy(
                    update={"input_feature_bundle_key": "B1_CORE_STRUCTURE"}
                )
            }
        ).model_validate(
            spec.model_copy(
                update={
                    "regime_study": spec.regime_study.model_copy(
                        update={"input_feature_bundle_key": "B1_CORE_STRUCTURE"}
                    )
                }
            ).model_dump(mode="json")
        )


def test_model_bearing_request_binds_exact_promotion_and_owner_artifacts(
    completed_candidate, tmp_path
):
    """§9.1: the same scientific inputs with different / absent promotion or
    owner refs produce different / refused semantic ids; readiness
    verified-loads the frozen authority; no latest-status lookup exists."""

    root = completed_candidate["store_root"]
    diagnostics = _sidecar(
        completed_candidate, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    protocol = load_regime_protocol(root, diagnostics["resolved_regime_protocol_id"])
    assessment = load_regime_assessment(root, diagnostics["regime_capability_assessment_id"])
    ready = load_regime_promotion(
        root, diagnostics["decisions"][-1]["regime_promotion_decision_id"]
    )
    owner = synthetic_owner_decision_fixture(root, protocol=protocol, assessment=assessment)
    eligible = ready.model_copy(
        update={
            "regime_promotion_decision_id": "0" * 64,  # replaced by from_payload below
        }
    )
    eligible = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=ready.regime_promotion_decision_id,
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at="2026-08-28T12:00:00+00:00",
        )
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    authority = (
        eligible.regime_promotion_decision_id,
        owner.owner_decision_artifact_id,
        assessment.regime_capability_assessment_id,
    )
    supervised = build_pipeline_fixture(
        tmp_path / "supervised", regime_study="candidate_supervised", regime_authority=authority
    )
    descriptive_id = completed_candidate["semantic"].pipeline_semantic_id
    supervised_id = supervised["semantic"].pipeline_semantic_id
    assert supervised_id != descriptive_id
    # a different owner artifact → a different semantic id
    other_owner = synthetic_owner_decision_fixture(
        root,
        protocol=protocol,
        assessment=assessment,
        approved_at="2026-08-27T00:00:00+00:00",
        effective_from="2026-08-27T00:00:00+00:00",
    )
    other = build_pipeline_fixture(
        tmp_path / "other",
        regime_study="candidate_supervised",
        regime_authority=(authority[0], other_owner.owner_decision_artifact_id, authority[2]),
    )
    assert other["semantic"].pipeline_semantic_id != supervised_id
    # absent refs are refused by the request itself
    with pytest.raises(ValueError, match="requires the frozen authority ids"):
        build_regime_study_request("candidate_supervised", authority=None)
    with pytest.raises(ValueError, match="must freeze the EXACT"):
        RegimeStudyRequest(
            input_feature_bundle_key="B0_CORE",
            resolved_input_features=("distance_to_htf_ticks",),
            stratified_reporting_requested=True,
            comparison_classes_requested=("feature_only",),
            supervised_bundle_key="B0_CORE",
            regime_promotion_decision_id=authority[0],
        )
    # readiness verified-loads the frozen authority from THIS store
    spec = supervised["semantic"].payload
    assert derive_stage_plan_readiness(spec).launchable  # no store: structural only
    assert derive_stage_plan_readiness(
        spec, store_root=root, run_scope="synthetic_fixture"
    ).launchable
    wrong_assessment = spec.model_copy(
        update={
            "regime_study": spec.regime_study.model_copy(
                update={"required_capability_assessment_id": "9" * 64}
            )
        }
    )
    report = derive_stage_plan_readiness(
        wrong_assessment, store_root=root, run_scope="synthetic_fixture"
    )
    assert not report.launchable
    assert "frozen regime authority refused" in report.entry(S.S09_TRAIN_MODELS).reason
    # the real run scope refuses the synthetic owner provenance
    real_scope = derive_stage_plan_readiness(
        spec, store_root=root, run_scope="verification_5d"
    )
    assert not real_scope.launchable
    assert "synthetic_fixture run scope" in real_scope.entry(S.S09_TRAIN_MODELS).reason
    # the other owner artifact does NOT authorize the frozen decision
    mismatch = derive_stage_plan_readiness(
        other["semantic"].payload, store_root=root, run_scope="synthetic_fixture"
    )
    assert not mismatch.launchable
    assert "different owner-decision artifact" in mismatch.entry(S.S09_TRAIN_MODELS).reason


def test_semantic_identity_carries_the_regime_study(tmp_path):
    fixture = build_pipeline_fixture(tmp_path, regime_study="candidate")
    plain = build_pipeline_fixture(tmp_path / "plain")
    assert fixture["semantic"].pipeline_semantic_id != plain["semantic"].pipeline_semantic_id
    payload = fixture["semantic"].payload
    reloaded = PipelineSemanticIdentity.from_payload(
        PipelineSemanticSpecPayload.model_validate(payload.model_dump(mode="json"))
    )
    assert reloaded.pipeline_semantic_id == fixture["semantic"].pipeline_semantic_id
    assert plain["semantic"].payload.regime_study is None


@pytest.mark.parametrize(
    ("shape", "hard_id_encoding"),
    [
        ("candidate", "none"),
        ("candidate_supervised", "fit_local_categorical_v1"),
        ("panel_supervised", "fit_local_categorical_v1"),
    ],
)
def test_s14_performs_zero_fitting(
    shape, hard_id_encoding, completed_candidate, completed_panel, tmp_path, monkeypatch
):
    """§9.1: with every estimator's ``fit`` monkeypatched to raise DURING S14,
    the stage still completes and persists the stratified reports from the
    persisted S09/S10/S12/S13 artifacts (D14: S14 performs zero fitting) —
    proven on the descriptive candidate study AND on the model-bearing
    candidate / panel studies, where S09b/S09c DID fit earlier in the run
    (adversarial R6.1 F4); the supervised shapes run under the categorical
    hard-id encoding (F11: one vocabulary, ``fit_local_categorical_v1``)."""

    from catboost import CatBoostClassifier
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (
        load_regime_stratified_report,
    )
    from alpha_lab.agents.data_infra.ifvg.search import pipeline as pipeline_module

    if shape == "candidate":
        fixture = build_pipeline_fixture(tmp_path, regime_study="candidate")
    else:
        base = completed_panel if shape.startswith("panel") else completed_candidate
        fixture = _supervised_over(base, shape, hard_id_encoding=hard_id_encoding)
    original = pipeline_module._STAGE_EXECUTORS[S.S14_BUILD_FRONTIER_AND_INSIGHTS]
    fit_calls: list[str] = []

    def _forbidden(name):
        def _fit(self, *args, **kwargs):
            fit_calls.append(name)
            raise AssertionError(f"{name}.fit was called during S14")

        return _fit

    def _s14_without_fitting(context):
        with pytest.MonkeyPatch.context() as inner:
            inner.setattr(KMeans, "fit", _forbidden("KMeans"))
            inner.setattr(LogisticRegression, "fit", _forbidden("LogisticRegression"))
            inner.setattr(CatBoostClassifier, "fit", _forbidden("CatBoostClassifier"))
            inner.setattr(Pipeline, "fit", _forbidden("Pipeline"))
            return original(context)

    executors = dict(pipeline_module._STAGE_EXECUTORS)
    executors[S.S14_BUILD_FRONTIER_AND_INSIGHTS] = _s14_without_fitting
    monkeypatch.setattr(pipeline_module, "_STAGE_EXECUTORS", executors)
    result = _run(fixture)
    fixture["result"] = result
    assert result.stage_statuses[S.S14_BUILD_FRONTIER_AND_INSIGHTS.value] == (
        StageStatus.COMPLETED.value
    )
    assert fit_calls == []
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    record = json.loads(
        load_sidecar_bytes(
            fixture["store_root"],
            "pipeline_stage_results",
            state["stages"][S.S14_BUILD_FRONTIER_AND_INSIGHTS.value]["stage_result_id"],
            "regime_stratified_reports.json",
        )
    )
    assert record["fitting_performed"] is False
    # the three descriptive classes are report envelopes; the modeled classes
    # (feature_only / cohort_model) are S09c deliverables, never S14 envelopes
    assert set(record["reports_by_class"]) == {
        "cohort_descriptive",
        "stratified_prop",
        "stratified_frontier",
    }
    if shape == "candidate":
        assert record["refusals"] == {}
    else:
        run = _sidecar(fixture, S.S09_TRAIN_MODELS, "regime_run.json")
        assert run["S09b"]["hard_id_encoding"] == "fit_local_categorical_v1"
        assert run["S09c"]["regime_controlled_study_id"]
        artifact = load_regime_fold_features(
            fixture["store_root"], run["S09b"]["regime_fold_feature_artifact_id"]
        )
        assert artifact.payload.hard_id_encoding == "fit_local_categorical_v1"
        # under the categorical encoding the fit-local id IS a model feature
        assert artifact.payload.categorical_model_features
        assert set(artifact.payload.categorical_model_features) <= set(
            artifact.payload.model_feature_names
        )
        assert all(
            name.endswith("_local_id") for name in artifact.payload.categorical_model_features
        )
    for report_id in record["report_ids"]:
        envelope = load_regime_stratified_report(fixture["store_root"], report_id)
        assert envelope.payload.gate.regime_promotion_decision_id == (
            record["regime_promotion_decision_id"]
        )
    outputs = state["stages"][S.S14_BUILD_FRONTIER_AND_INSIGHTS.value]["output_artifact_ids"]
    assert set(record["report_ids"]) <= set(outputs)


def test_model_bearing_run_executes_s09b_s09c_on_the_frozen_authority(
    completed_candidate, tmp_path
):
    """The two-pass workflow end to end: the descriptive run's assessment →
    a synthetic owner decision + the FEATURE_ELIGIBLE promotion (synthetic
    scope) → a NEW frozen model-bearing run (feature_only + cohort_model)
    over the same store: S05 verifies the resolved protocol against the
    frozen decision, S09a reproduces the exact required assessment, S09b
    persists the fold-local feature artifact, S09c persists the controlled
    regime study and the cohort model study, S10 carries the frozen
    authority, S14 reports the descriptive classes and records the modeled
    classes as S09c deliverables, S15 reloads everything."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_cohort_model import (
        load_regime_cohort_model_study,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_controlled_study import (
        load_regime_controlled_study,
    )

    root = completed_candidate["store_root"]
    authority, owner, protocol, assessment, eligible = _frozen_authority(completed_candidate)
    supervised = build_pipeline_fixture(
        tmp_path / "supervised", regime_study="candidate_supervised", regime_authority=authority
    )
    supervised["store_root"] = root  # the frozen authority lives in THIS store
    result = _run(supervised)
    supervised["result"] = result
    _assert_terminal(result)
    observation = _sidecar(
        supervised, S.S05_MATERIALIZE_FEATURE_VIEWS, "bundle_feature_views.json"
    )["__regime_observation__"]
    assert observation["frozen_authority_verified"] is True
    assert observation["resolved_regime_protocol_id"] == protocol.resolved_regime_protocol_id
    run = _sidecar(supervised, S.S09_TRAIN_MODELS, "regime_run.json")
    assert run["S09a"]["regime_capability_assessment_id"] == (
        assessment.regime_capability_assessment_id
    )
    fold_feature_id = run["S09b"]["regime_fold_feature_artifact_id"]
    assert load_regime_fold_features(root, fold_feature_id).payload.resolved_regime_protocol_id == (
        protocol.resolved_regime_protocol_id
    )
    controlled = load_regime_controlled_study(root, run["S09c"]["regime_controlled_study_id"])
    assert controlled.payload.regime_fold_feature_artifact_id == fold_feature_id
    cohort = load_regime_cohort_model_study(root, run["S09c"]["regime_cohort_model_study_id"])
    assert cohort.payload.regime_fold_feature_artifact_id == fold_feature_id
    s10 = _sidecar(
        supervised, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    assert s10["authority_source"] == "frozen_owner_evidence"
    assert s10["final_status"] == "feature_eligible"
    assert s10["frozen_authority"]["regime_promotion_decision_id"] == (
        eligible.regime_promotion_decision_id
    )
    assert s10["sub_steps"] == ["S09a", "S09b", "S09c"]
    assert s10["fold_feature_artifact_id"] == fold_feature_id
    assert "feature_only" in s10["paired_deltas"]
    reports = _sidecar(
        supervised, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json"
    )
    assert reports["regime_promotion_decision_id"] == eligible.regime_promotion_decision_id
    assert reports["authority_source"] == "frozen_owner_evidence"
    assert {"cohort_descriptive", "stratified_prop", "stratified_frontier"} <= set(
        reports["reports_by_class"]
    ), reports["refusals"]
    assert reports["fitting_performed"] is False
    assert supervised["semantic"].pipeline_semantic_id != (
        completed_candidate["semantic"].pipeline_semantic_id
    )


def test_s05_refuses_a_seam_that_serves_another_verified_chart(tmp_path):
    """Adversarial R6.1 S1: a ``context_bar_source`` that ignores the requested
    chart id and serves ANOTHER verified artifact (the reviewer's probe shape)
    fails S05 closed — nothing downstream binds a chart the run never asked
    for, and no panel artifact is persisted."""

    fixture = build_pipeline_fixture(tmp_path, regime_study="panel")
    foreign = foreign_context_bar_source(
        tmp_path / "foreign_chart", fixture["regime_fixture"].trading_days
    )
    fixture["wiring"] = dataclasses.replace(fixture["wiring"], context_bar_source=foreign)
    result = _run(fixture)
    assert result.stage_statuses[S.S05_MATERIALIZE_FEATURE_VIEWS.value] == StageStatus.FAILED.value
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    explanation = state["stages"][S.S05_MATERIALIZE_FEATURE_VIEWS.value]["explanation"]
    assert "the panel binds the chart it asked for" in explanation
    assert foreign.artifact_id[:12] in explanation
    assert not (fixture["store_root"] / "context_bar_panels").exists()
    for stage in (S.S08_BUILD_FOLDS, S.S09_TRAIN_MODELS, S.S14_BUILD_FRONTIER_AND_INSIGHTS):
        assert result.stage_statuses[stage.value] == StageStatus.PENDING.value


def test_s08_derives_the_schedule_once_from_the_candidate_view_days(tmp_path):
    """Adversarial R6.1 F7: the candidate view's OBSERVED trading days are the
    ONE day source of S08 — a label builder that omits the last day (the
    reviewer's probe) yields a thinner last fold under the SAME persisted
    schedule instead of a schedule/window mismatch failure."""

    fixture = build_pipeline_fixture(tmp_path, regime_study="candidate")
    labels = fixture["labels"]
    last_day = max(labels["trading_day"].astype(str))
    thinner = labels[labels["trading_day"].astype(str) != last_day].copy()
    fixture["wiring"] = dataclasses.replace(
        fixture["wiring"], label_builder=lambda _view: (LABEL_POLICY_ID, thinner.copy())
    )
    result = _run(fixture)
    fixture["result"] = result
    assert result.stage_statuses[S.S08_BUILD_FOLDS.value] == StageStatus.COMPLETED.value
    assert result.stage_statuses[S.S09_TRAIN_MODELS.value] == StageStatus.COMPLETED.value
    folds = _sidecar(fixture, S.S08_BUILD_FOLDS, "fold_sample_adequacy.json")
    assert folds["trading_days_source"] == "candidate_view_observed_days"
    assert folds["candidate_folds_labeled"] is True
    assert len(folds["authorized_trading_days"]) == len(fixture["regime_fixture"].trading_days)
    assert folds["days_without_labels"] == [last_day]
    assert has_envelope(fixture["store_root"], "fold_sets", folds["candidate_fold_set_artifact_id"])


def test_s02_reused_children_are_adopted_only_by_verified_reproduction(
    completed_candidate, tmp_path
):
    """Adversarial R6.1 S4: a reused child's re-derived tables become this
    run's evidence ONLY when they reproduce the persisted costed evaluation
    of THIS cost policy; a cost policy with no persisted evaluation leaves the
    reproduction unverifiable — the tables are not adopted, the row says so,
    and no evaluation is published from them."""

    root = completed_candidate["store_root"]
    second = build_pipeline_fixture(
        tmp_path / "second", regime_study="candidate", regime_bootstrap_refits=4
    )
    second["store_root"] = root
    assert second["semantic"].pipeline_semantic_id != (
        completed_candidate["semantic"].pipeline_semantic_id
    )
    result = _run(second)
    assert result.stage_statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == (
        StageStatus.COMPLETED.value
    )
    rows = read_pipeline_state(second["state_root"], result.pipeline_semantic_id)["children"]
    assert rows
    for row in rows:
        assert row["state"] == "reused"
        assert row["replay_invocations"] == 1
        assert "verified reuse by reproduction" in row["explanation"]
        assert "reproduced its persisted costed evaluation" in row["explanation"]
    # a DIFFERENT cost policy over the same immutable replays: nothing persisted
    # can verify the reproduction → the re-derived tables are NOT adopted
    charter = second["charter"]
    other_payload = charter.payload.model_copy(
        update={
            "cost_policy": charter.payload.cost_policy.model_copy(
                update={"cost_points_round_turn": 1.25}
            )
        }
    )
    other_charter = SearchCharterEnvelope.from_payload(other_payload)
    other_spec = second["semantic"].payload.model_copy(
        update={
            "search_charter_id": other_charter.search_id,
            "cost_policy_sha256": canonical_contract_sha256(other_payload.cost_policy),
        }
    )
    third = {
        **second,
        "charter": other_charter,
        "semantic": PipelineSemanticIdentity.from_payload(other_spec),
        "state_root": tmp_path / "third_state",
    }
    result = _run(third)
    rows = read_pipeline_state(third["state_root"], result.pipeline_semantic_id)["children"]
    assert rows
    for row in rows:
        assert row["state"] == "reused"
        assert row["replay_invocations"] == 1
        assert "reproduction unverifiable" in row["explanation"]
        assert "not this run's evidence" in row["explanation"]
        evaluation = _child_evaluation_envelope(row["core_replay_id"], other_payload.cost_policy)
        assert not has_envelope(root, "costed_evaluations", evaluation.costed_evaluation_id)


def test_model_bearing_panel_run_executes_s09b_s09c_pit_on_the_frozen_authority(
    completed_panel,
):
    """The two-pass workflow on the PANEL grain (adversarial R6.1 F4): the
    descriptive panel run's assessment → owner evidence + FEATURE_ELIGIBLE →
    a frozen ``panel_supervised`` run over the same store and chart seam:
    S05 re-materializes the SAME panel (the protocol pins its id) and verifies
    the frozen authority, S09a reproduces the required assessment, S09b
    persists the PIT fold-local features under the panel fold set, S09c the
    two modeled studies, S10 carries the frozen authority, S14 reports the
    descriptive classes with zero fitting, S15 reloads everything."""

    supervised = _supervised_over(completed_panel, "panel_supervised")
    result = _run(supervised)
    supervised["result"] = result
    _assert_terminal(result)
    observation = _sidecar(
        supervised, S.S05_MATERIALIZE_FEATURE_VIEWS, "bundle_feature_views.json"
    )["__regime_observation__"]
    descriptive = _sidecar(
        completed_panel, S.S05_MATERIALIZE_FEATURE_VIEWS, "bundle_feature_views.json"
    )["__regime_observation__"]
    assert observation["frozen_authority_verified"] is True
    assert observation["artifact_id"] == descriptive["artifact_id"]  # the same persisted panel
    assert observation["replay_chart_artifact_id"] == observation["chart_id"]
    assert observation["resolved_regime_protocol_id"] == descriptive["resolved_regime_protocol_id"]
    folds = _sidecar(supervised, S.S08_BUILD_FOLDS, "fold_sample_adequacy.json")
    run = _sidecar(supervised, S.S09_TRAIN_MODELS, "regime_run.json")
    assert run["S09a"]["regime_capability_assessment_id"] == (
        supervised["regime_request"].required_capability_assessment_id
    )
    assert run["S09b"]["regime_fold_set_artifact_id"] == folds["panel_fold_set_artifact_id"]
    assert run["S09b"]["candidate_fold_set_artifact_id"] == folds["candidate_fold_set_artifact_id"]
    assert run["S09b"]["hard_id_encoding"] == "none"
    artifact = load_regime_fold_features(
        supervised["store_root"], run["S09b"]["regime_fold_feature_artifact_id"]
    )
    assert artifact.payload.observation_granularity.value == "context_bar_panel"
    assert run["S09c"]["regime_controlled_study_id"]
    assert run["S09c"]["regime_cohort_model_study_id"]
    s10 = _sidecar(
        supervised, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    assert s10["authority_source"] == "frozen_owner_evidence"
    assert s10["final_status"] == "feature_eligible"
    assert s10["sub_steps"] == ["S09a", "S09b", "S09c"]
    reports = _sidecar(
        supervised, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json"
    )
    assert reports["fitting_performed"] is False
    assert {"cohort_descriptive", "stratified_prop", "stratified_frontier"} <= set(
        reports["reports_by_class"]
    )
    assert "gates passed" in _stage(supervised, S.S15_VERIFY_AND_PUBLISH)["explanation"]


def test_supervised_classes_without_s07_or_a_label_policy_are_refused(
    completed_candidate, tmp_path
):
    """§9.2 (adversarial R6.1 F4): ``feature_only`` / ``cohort_model`` requests
    need 07_derive_labels + a label policy — refused by the request's
    structural rules, by readiness, and by the semantic spec itself; a
    descriptive request needs neither (it runs S09a only)."""

    authority, *_ = _frozen_authority(completed_candidate)
    request = build_regime_study_request("candidate_supervised", authority=authority)
    full = tuple(stage.value for stage in S)
    without_s07 = tuple(value for value in full if value != S.S07_DERIVE_LABELS.value)
    facts = {
        "feature_bundle_ids": ("B0_CORE",),
        "stage_values": without_s07,
        "label_policy_id": LABEL_POLICY_ID,
        "model_protocol_id": "ifvg_context_logistic_l2_v1",
    }
    problems = request.stage_plan_problems(**facts)
    assert any("require 07_derive_labels" in problem for problem in problems), problems
    reason = regime_study_block_reason(request, RegimeStudyStagePlanFacts(**facts))
    assert reason is not None and "require 07_derive_labels" in reason
    problems = request.stage_plan_problems(
        **{**facts, "stage_values": full, "label_policy_id": None}
    )
    assert any("require 07_derive_labels" in problem for problem in problems), problems
    assert build_regime_study_request("candidate").stage_plan_problems(**facts) == []
    # the semantic spec refuses the plan itself (07 is also S08's frozen prerequisite)
    payload = build_pipeline_fixture(
        tmp_path, regime_study="candidate_supervised", regime_authority=authority
    )["semantic"].payload.model_dump(mode="json")
    payload["stage_plan"] = [
        value for value in payload["stage_plan"] if value != "07_derive_labels"
    ]
    with pytest.raises(ValueError, match="07_derive_labels"):
        PipelineSemanticSpecPayload.model_validate(payload)
