"""R6.1 adversarial round — the stratified-report status gate (F2 / S6 / S3).

``resolve_report_gate`` re-derives the STRATIFICATION_READY structural gate
from the LOADED assessment (D6; the status label is never trusted, whoever
minted the decision), re-runs the full owner authorization under the run
scope with the store-owned supersession chain for the modeled classes, and
confines the ``synthetic_fixture`` scope to test namespaces (P0-4 mirror).
Every root is a synthetic tmp root; no real data path exists here.
"""

from __future__ import annotations

import shutil

import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    REGIME_PROMOTION_STORE,
    persist_regime_assessment,
    persist_regime_promotion,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_gate import (
    RegimeStatusRefusalError,
    resolve_report_gate,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (
    CLASS_MINIMUM_STATUS,
    RegimeStratificationClass,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_study import (
    RegimeStudyRequest,
    build_s10_decisions,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    synthetic_owner_decision_fixture,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_candidate_source,
)

_DECIDED_AT = "2026-08-28T12:00:00+00:00"
_MODELED = tuple(
    cls
    for cls in RegimeStratificationClass
    if CLASS_MINIMUM_STATUS[cls] is RegimeStatus.FEATURE_ELIGIBLE
)


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    """A lawful synthetic chain: S10's DESCRIPTIVE_ONLY → STRATIFICATION_READY
    over a PASSING assessment with an OOS assignment, then FEATURE_ELIGIBLE
    ratified by a synthetic owner artifact in the synthetic scope."""

    root = tmp_path_factory.mktemp("report_gate")
    source = persisted_candidate_source(root)
    result = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=2,
    )
    assessment = result.run.assessment
    assert assessment.payload.gates_passed
    assert assessment.payload.coverage.coverage_gates_passed
    assert assessment.payload.oos_assignment_available
    request = RegimeStudyRequest(
        input_feature_bundle_key="B0_CORE",
        resolved_input_features=REGIME_INPUT_FEATURES,
        stratified_reporting_requested=True,
        comparison_classes_requested=("cohort_descriptive",),
    )
    decisions = build_s10_decisions(
        root,
        protocol=result.protocol,
        assessment=assessment,
        request=request,
        decided_at=_DECIDED_AT,
    )
    ready = decisions[-1]
    assert ready.payload.status is RegimeStatus.STRATIFICATION_READY
    owner = synthetic_owner_decision_fixture(root, protocol=result.protocol, assessment=assessment)
    eligible = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=result.protocol.resolved_regime_protocol_id,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=ready.regime_promotion_decision_id,
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at=_DECIDED_AT,
        )
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    return {
        "root": root,
        "protocol": result.protocol,
        "protocol_id": result.protocol.resolved_regime_protocol_id,
        "assessment": assessment,
        "fit_ids": tuple(assessment.payload.regime_fit_ids),
        "first": decisions[0],
        "ready": ready,
        "eligible": eligible,
        "owner": owner,
    }


def _gate(lane, root=None, **overrides):
    kwargs = dict(
        protocol_id=lane["protocol_id"],
        decision_id=lane["eligible"].regime_promotion_decision_id,
        owner_decision_artifact_id=lane["owner"].owner_decision_artifact_id,
        comparison_class=_MODELED[0],
        fit_ids=lane["fit_ids"],
        run_scope="synthetic_fixture",
    )
    kwargs.update(overrides)
    return resolve_report_gate(root or lane["root"], **kwargs)


@pytest.mark.parametrize("comparison_class", _MODELED)
def test_modeled_classes_re_run_owner_authorization_under_the_run_scope(lane, comparison_class):
    """S6: the owner half is the FULL authorization (provenance-vs-scope,
    supersession, effectivity, values, transition) — not a two-field match."""

    resolved = _gate(lane, comparison_class=comparison_class)
    assert resolved.gate.authority_source == "frozen_owner_evidence"
    assert resolved.gate.status_at_report is RegimeStatus.FEATURE_ELIGIBLE
    assert resolved.owner_decision is not None
    assert (
        resolved.owner_decision.owner_decision_artifact_id
        == lane["owner"].owner_decision_artifact_id
    )
    # the same synthetic artifact in the default (real) run scope refuses
    with pytest.raises(RegimeStatusRefusalError, match="lawful only in the synthetic_fixture"):
        _gate(lane, comparison_class=comparison_class, run_scope="full_authorized_development")
    with pytest.raises(RegimeStatusRefusalError, match="lawful only in the synthetic_fixture"):
        _gate(lane, comparison_class=comparison_class, run_scope="verification_5d")


def test_descriptive_classes_take_the_structural_gate_and_load_no_owner(lane):
    resolved = resolve_report_gate(
        lane["root"],
        protocol_id=lane["protocol_id"],
        decision_id=lane["ready"].regime_promotion_decision_id,
        owner_decision_artifact_id=None,
        comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
        fit_ids=lane["fit_ids"],
    )
    assert resolved.gate.authority_source == "s10_structural"
    assert resolved.owner_decision is None
    assert resolved.gate.owner_decision_artifact_id is None


def test_a_superseded_owner_artifact_is_refused_at_the_gate(lane, tmp_path):
    """S6 / S2: the gate consults the store-owned supersession chain — a
    replacement ratified later revokes the frozen artifact's authority."""

    root = tmp_path / "superseded"
    shutil.copytree(lane["root"], root)
    synthetic_owner_decision_fixture(
        root,
        protocol=lane["protocol"],
        assessment=lane["assessment"],
        supersedes=lane["owner"].owner_decision_artifact_id,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
    )
    with pytest.raises(RegimeStatusRefusalError, match="superseded by a later decision"):
        _gate(lane, root=root)


def test_the_structural_gate_is_rederived_from_the_loaded_assessment(tmp_path):
    """F2 / D6: a STRATIFICATION_READY decision minted around the store API
    over an assessment whose coverage gates FAIL is refused whoever minted
    it — the gate never trusts the status label; the store refuses it too."""

    root = tmp_path / "d6"
    fixture = known_cluster_fixture(k=3, n=170)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=resolve_bundle("B0_CORE").resolved_feature_bundle_id,
        resolved_input_features=fixture.regime_input_features,
        winsorization_policy="clip_p01_p99_train_fitted_v1",
    )
    small = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=3,
    )
    assert not small.assessment.payload.coverage.coverage_gates_passed
    persist_regime_protocol(root, small.protocol)
    persist_regime_assessment(root, small.assessment)
    first = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=small.protocol.resolved_regime_protocol_id,
            role=RegimeRole.DESCRIPTIVE_ONLY,
            status=RegimeStatus.DESCRIPTIVE_ONLY,
            previous_status=RegimeStatus.PLANNED,
            previous_decision_ref=None,
            capability_assessment_ref=small.assessment.regime_capability_assessment_id,
            owner_ratification_ref=None,
            decided_at=_DECIDED_AT,
        )
    )
    persist_regime_promotion(root, first)
    ready = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=small.protocol.resolved_regime_protocol_id,
            role=RegimeRole.STRATIFICATION_ONLY,
            status=RegimeStatus.STRATIFICATION_READY,
            previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
            previous_decision_ref=first.regime_promotion_decision_id,
            capability_assessment_ref=small.assessment.regime_capability_assessment_id,
            owner_ratification_ref=None,
            decided_at=_DECIDED_AT,
        )
    )
    with pytest.raises(ValueError, match="structurally requires .* \\(D6\\)"):
        persist_regime_promotion(root, ready)
    # minted around the store API (a pre-fix path, a foreign tool): the gate
    # still refuses it from the LOADED assessment
    save_or_reuse_envelope(root, REGIME_PROMOTION_STORE, ready)
    with pytest.raises(RegimeStatusRefusalError, match="does not pass the structural gates"):
        resolve_report_gate(
            root,
            protocol_id=small.protocol.resolved_regime_protocol_id,
            decision_id=ready.regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
            fit_ids=tuple(small.assessment.payload.regime_fit_ids),
        )


def test_the_synthetic_scope_is_confined_to_test_namespaces_at_the_gate(lane, tmp_path):
    """S3 (P0-4 mirror): a research-namespace root refuses the synthetic
    scope BEFORE any load — nothing needs to exist under it."""

    research = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    research.mkdir(parents=True)
    with pytest.raises(RegimeStatusRefusalError, match="confined to test namespaces"):
        _gate(lane, root=research)
