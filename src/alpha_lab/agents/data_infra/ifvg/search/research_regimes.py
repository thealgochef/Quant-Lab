"""Verified descriptive-regime review and explicit owner-signed eligibility.

Reading never fits or deserializes a model. Promotion changes authority only;
the reviewed numerical protocol, assessment, labels and fold evidence stay fixed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..context_folds import build_context_folds
from ..ml.fold_set_artifact import fold_set_id, load_fold_schedule, load_fold_set_artifact
from ..ml.regime_contracts import (
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from ..ml.regime_oos_assignment import (
    load_regime_oos_assignment,
    load_regime_oos_assignment_frame,
)
from ..ml.regime_store import (
    load_regime_assessment,
    load_regime_fit_assignments,
    load_regime_promotion,
    load_regime_protocol,
    persist_regime_promotion,
)
from .identities import canonical_contract_sha256
from .owner_decisions import (
    OwnerDecisionArtifactEnvelope,
    OwnerDecisionArtifactPayload,
    build_owner_decision_proposal,
    expected_decision_values,
    freeze_decision_values,
    persist_owner_decision,
    plain_decision_values,
    require_owner_decision_namespace,
)
from .pipeline import (
    PipelineSemanticIdentity,
    PipelineStageResultEnvelope,
    QuantLabPipelineStage,
)
from .research_artifacts import load_research_labels
from .store import load_sidecar_bytes, load_verified_envelope

__all__ = ["read_research_regime_eligibility", "promote_research_regime"]


def _cell(root, group_id, pipeline_id):
    from .research_runs import ResearchGroupEnvelope, ResearchRequest  # noqa: PLC0415

    group = load_verified_envelope(root, "research_groups", group_id, ResearchGroupEnvelope)
    matching = [
        row
        for row in json.loads(group.payload.cells_json)
        if row["pipeline_semantic_id"] == pipeline_id
    ]
    if len(matching) != 1:
        raise ValueError("eligibility requires one exact pipeline cell of this research group")
    cell = matching[0]
    semantic = load_verified_envelope(root, "pipeline_specs", pipeline_id, PipelineSemanticIdentity)
    request = ResearchRequest.model_validate_json(group.payload.request_json)
    if (
        cell["subject_id"] not in semantic.payload.source_artifact_ids
        or group.payload.research_approval_id not in semantic.payload.source_artifact_ids
        or request.regime_study != semantic.payload.regime_study
        or request.feature_bundle_ids != semantic.payload.feature_bundle_ids
    ):
        raise ValueError("research group cell differs from its exact pipeline specification")
    regime = request.regime_study
    if regime is None or regime.requires_supervision:
        raise ValueError("eligibility review requires a completed descriptive regime precursor")
    return group, cell, semantic


def _stages(root, pipeline_id):
    required = {
        QuantLabPipelineStage.S07_DERIVE_LABELS,
        QuantLabPipelineStage.S08_BUILD_FOLDS,
        QuantLabPipelineStage.S09_TRAIN_MODELS,
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
    }
    found = {stage: [] for stage in required}
    for path in sorted((root / "pipeline_stage_results").glob("*/envelope.json")):
        # Discovery metadata cannot authorize anything. Every matching entry is
        # exact-loaded with all manifest checks before its IDs are consulted.
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("payload", {}).get("pipeline_semantic_id") != pipeline_id:
            continue
        envelope = load_verified_envelope(
            root, "pipeline_stage_results", path.parent.name, PipelineStageResultEnvelope
        )
        if envelope.payload.stage in found:
            found[envelope.payload.stage].append(envelope)
    for stage, entries in found.items():
        if len(entries) != 1:
            raise ValueError(
                f"eligibility requires one verified {stage.value} result; found {len(entries)}"
            )
    return {stage: entries[0] for stage, entries in found.items()}


def _sidecar(root, stage, filename, evidence):
    raw = load_sidecar_bytes(root, "pipeline_stage_results", stage.stage_result_id, filename)
    evidence[f"{stage.stage_result_id}/{filename}"] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _require_outputs(stage, *ids):
    if not set(ids) <= set(stage.payload.output_artifact_ids):
        raise ValueError(
            f"{stage.payload.stage.value} sidecar names artifacts absent from its outputs"
        )


def _verified_review(root, group_id, pipeline_id, result):
    group, cell, semantic = _cell(root, group_id, pipeline_id)
    namespace = require_owner_decision_namespace(root)
    stages = _stages(root, pipeline_id)
    s07 = stages[QuantLabPipelineStage.S07_DERIVE_LABELS]
    s08 = stages[QuantLabPipelineStage.S08_BUILD_FOLDS]
    s09 = stages[QuantLabPipelineStage.S09_TRAIN_MODELS]
    s10 = stages[QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS]
    evidence = result["evidence"]
    evidence.update(
        {
            "research_group_id": group_id,
            "pipeline_semantic_id": pipeline_id,
            "research_subject_id": cell["subject_id"],
            "store_namespace_id": namespace.store_namespace_id,
            "stage_result_ids": {stage.value: env.stage_result_id for stage, env in stages.items()},
        }
    )
    run = _sidecar(root, s09, "regime_run.json", evidence)["S09a"]
    diagnostics = _sidecar(root, s10, "regime_diagnostics.json", evidence)
    request = semantic.payload.regime_study
    if diagnostics["request"] != request.model_dump(mode="json"):
        raise ValueError("S10 diagnostics differ from the frozen regime request")
    protocol_id = run["resolved_regime_protocol_id"]
    assessment_id = run["regime_capability_assessment_id"]
    if (
        diagnostics["resolved_regime_protocol_id"] != protocol_id
        or diagnostics["regime_capability_assessment_id"] != assessment_id
    ):
        raise ValueError("S09 and S10 do not describe the same exact regime assessment")
    protocol = load_regime_protocol(root, protocol_id)
    assessment = load_regime_assessment(root, assessment_id)
    payload = assessment.payload
    if payload.resolved_regime_protocol_id != protocol_id:
        raise ValueError("capability assessment names another regime protocol")
    for name in (
        "algorithm_key",
        "observation_granularity",
        "observation_stage",
        "panel_interval_seconds",
        "panel_as_of_policy_id",
        "resolved_input_features",
        "resolved_cluster_count",
        "winsorization_policy",
    ):
        if getattr(protocol.payload, name) != getattr(request, name):
            raise ValueError(f"reviewed regime protocol differs from the requested {name}")
    result.update(
        assessment_id=assessment_id,
        protocol_id=protocol_id,
        input_features=list(protocol.payload.resolved_input_features),
        requested_bootstrap_refits=payload.stability.bootstrap_refits_per_fold_requested,
        applied_bootstrap_refits=payload.stability.bootstrap_refits_per_fold_applied,
        gates={
            "gates_passed": payload.gates_passed,
            "gate_failures": list(payload.gate_failures),
            "coverage_gates_passed": payload.coverage.coverage_gates_passed,
            "stability_gates_passed": payload.stability.stability_gates_passed,
            "oos_assignment_available": payload.oos_assignment_available,
        },
        sample_floors=plain_decision_values(expected_decision_values(protocol, assessment)),
    )
    if request.bootstrap_refits != payload.stability.bootstrap_refits_per_fold_requested:
        raise ValueError("assessment bootstrap request differs from the frozen research request")
    if not (
        payload.gates_passed
        and payload.coverage.coverage_gates_passed
        and payload.stability.stability_gates_passed
        and payload.oos_assignment_available
    ):
        result["blockers"].append(
            "regime coverage, occupancy, stability and OOS gates must all pass"
        )
        result["blockers"].extend(payload.gate_failures)
    if payload.stability.bootstrap_refits_per_fold_applied < 1:
        result["blockers"].append("no bootstrap stability refits were applied")

    label_record = _sidecar(root, s07, "research_labels.json", evidence)
    label_id = label_record["research_label_id"]
    label_envelope, labels = load_research_labels(root, label_id)
    _require_outputs(s07, label_id, label_envelope.payload.label_artifact_id)
    if (
        label_envelope.payload.research_subject_id != cell["subject_id"]
        or label_envelope.payload.label_policy_id != semantic.payload.label_policy_id
        or label_record["research_subject_id"] != cell["subject_id"]
        or label_record["label_artifact_id"] != label_envelope.payload.label_artifact_id
    ):
        raise ValueError("precursor labels differ from the exact research subject or policy")
    if (
        labels.empty
        or "is_warmup" not in labels
        or labels["is_warmup"].isna().any()
        or labels["is_warmup"].astype(bool).any()
    ):
        raise ValueError("precursor requires nonempty, explicitly post-warmup labels")
    fold_record = _sidecar(root, s08, "fold_sample_adequacy.json", evidence)
    schedule_id = diagnostics["fold_schedule_id"]
    candidate_fold_id = diagnostics["candidate_fold_set_artifact_id"]
    regime_fold_id = diagnostics["regime_fold_set_artifact_id"]
    _require_outputs(s08, schedule_id, candidate_fold_id, regime_fold_id)
    schedule = load_fold_schedule(root, schedule_id)
    candidate_set, candidate_folds = load_fold_set_artifact(root, candidate_fold_id)
    regime_set, _ = load_fold_set_artifact(root, regime_fold_id)
    if (
        fold_record["fold_schedule_id"] != schedule_id
        or fold_record["candidate_fold_set_artifact_id"] != candidate_fold_id
        or not candidate_set.payload.labeled
        or candidate_set.payload.fold_schedule_id != schedule_id
        or regime_set.payload.fold_schedule_id != schedule_id
        or regime_set.payload.fold_set_id != payload.fold_set_id
        or run["fold_set_artifact_id"] != regime_fold_id
        or run["fold_schedule_id"] != schedule_id
    ):
        raise ValueError("precursor label/regime folds do not share the exact recorded schedule")
    rebuilt = build_context_folds(
        labels,
        authorized_trading_days=schedule.payload.authorized_trading_days,
        purge_from_logical_test_start=True,
    )
    if fold_set_id(rebuilt) != candidate_set.payload.fold_set_id:
        raise ValueError("precursor candidate folds do not reproduce from the saved labels")
    if not any(fold.valid for fold in candidate_folds.folds):
        result["blockers"].append("precursor labels have no valid supervised candidate fold")
    evidence.update(
        research_label_id=label_id,
        fold_schedule_id=schedule_id,
        candidate_fold_set_artifact_id=candidate_fold_id,
        regime_fold_set_artifact_id=regime_fold_id,
    )

    fit_ids = tuple(run["regime_fit_ids"])
    if set(fit_ids) != set(payload.regime_fit_ids) or fit_ids != tuple(
        diagnostics["regime_fit_ids"]
    ):
        raise ValueError("S09/S10 fit membership differs from the assessment")
    _require_outputs(s09, assessment_id, *fit_ids, run["regime_oos_assignment_id"])
    for fit_id in fit_ids:
        fit = load_regime_fit_assignments(root, fit_id)
        if fit.envelope.payload.resolved_regime_protocol_id != protocol_id:
            raise ValueError("reviewed fit belongs to another regime protocol")
    oos = load_regime_oos_assignment(root, run["regime_oos_assignment_id"])
    load_regime_oos_assignment_frame(root, oos)
    if (
        oos.payload.resolved_regime_protocol_id != protocol_id
        or set(oos.payload.regime_fit_ids) != set(fit_ids)
        or oos.payload.fold_schedule_id != schedule_id
        or oos.payload.regime_fold_set_id != payload.fold_set_id
        or diagnostics["regime_oos_assignment_id"] != oos.regime_oos_assignment_id
    ):
        raise ValueError("descriptive OOS evidence differs from the reviewed fits and schedule")
    evidence.update(
        regime_fit_ids=list(fit_ids), regime_oos_assignment_id=oos.regime_oos_assignment_id
    )

    decisions = [
        load_regime_promotion(root, entry["regime_promotion_decision_id"])
        for entry in diagnostics["decisions"]
    ]
    _require_outputs(s10, *(decision.regime_promotion_decision_id for decision in decisions))
    ready = [
        decision
        for decision in decisions
        if decision.payload.status is RegimeStatus.STRATIFICATION_READY
    ]
    if len(ready) != 1 or diagnostics["final_status"] != RegimeStatus.STRATIFICATION_READY.value:
        result["blockers"].append("S10 must end at its exact STRATIFICATION_READY decision")
        previous = None
    else:
        previous = ready[0]
        if (
            previous.payload.resolved_regime_protocol_id != protocol_id
            or previous.payload.capability_assessment_ref != assessment_id
        ):
            raise ValueError("STRATIFICATION_READY decision names another protocol or assessment")
        evidence["stratification_ready_decision_id"] = previous.regime_promotion_decision_id
    result["ready"] = not result["blockers"]
    result["review_id"] = canonical_contract_sha256(
        {"version": "research_regime_eligibility_review_v1", **result}
    )
    return protocol, assessment, previous, namespace


def read_research_regime_eligibility(root, group_id, pipeline_id) -> dict:
    """Return a verified review, or explicit blockers; perform no writes/fits."""
    result = {
        "ready": False,
        "blockers": [],
        "assessment_id": None,
        "protocol_id": None,
        "input_features": [],
        "requested_bootstrap_refits": None,
        "applied_bootstrap_refits": None,
        "gates": {},
        "sample_floors": {},
        "evidence": {},
        "review_id": None,
    }
    try:
        _verified_review(Path(root), group_id, pipeline_id, result)
    except (OSError, ValueError, KeyError, TypeError, PermissionError) as error:
        result["ready"] = False
        result["blockers"].append(str(error))
        result["review_id"] = None
    return result


def promote_research_regime(
    root, group_id, pipeline_id, *, author, approval_statement, review_id=None
) -> dict:
    """Persist the caller's explicit approval of this exact passing review."""
    if (
        not isinstance(author, str)
        or not isinstance(approval_statement, str)
        or not author.strip()
        or not approval_statement.strip()
    ):
        raise PermissionError("explicit owner author and approval statement are required")
    root = Path(root)
    review = read_research_regime_eligibility(root, group_id, pipeline_id)
    if not review["ready"]:
        raise PermissionError("regime eligibility blocked: " + "; ".join(review["blockers"]))
    if review_id is not None and review_id != review["review_id"]:
        raise PermissionError("regime evidence changed after review; refresh before approving")
    protocol = load_regime_protocol(root, review["protocol_id"])
    assessment = load_regime_assessment(root, review["assessment_id"])
    previous = load_regime_promotion(root, review["evidence"]["stratification_ready_decision_id"])
    approved_at = datetime.now(UTC).isoformat()
    if datetime.fromisoformat(previous.payload.decided_at.replace("Z", "+00:00")) > datetime.now(
        UTC
    ):
        raise PermissionError("the precursor decision is dated after the current approval time")
    proposal = build_owner_decision_proposal(
        protocol, assessment, store_namespace_id=review["evidence"]["store_namespace_id"]
    )
    proposal.pop("_proposal_note")
    proposal.update(
        author=author.strip(),
        approved_at=approved_at,
        effective_from=approved_at,
        rationale=approval_statement.strip(),
        reviewed_evidence_refs=(
            review["assessment_id"],
            review["protocol_id"],
            group_id,
            pipeline_id,
            review["review_id"],
            *review["evidence"]["stage_result_ids"].values(),
        ),
    )
    proposal["decision_values"] = freeze_decision_values(proposal["decision_values"])
    owner = OwnerDecisionArtifactEnvelope.from_payload(OwnerDecisionArtifactPayload(**proposal))
    promotion = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=review["protocol_id"],
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=previous.regime_promotion_decision_id,
            capability_assessment_ref=review["assessment_id"],
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at=approved_at,
        )
    )
    persist_owner_decision(root, owner, recorded_at=approved_at)
    persist_regime_promotion(root, promotion)
    load_regime_promotion(root, promotion.regime_promotion_decision_id)
    return {
        "regime_promotion_decision_id": promotion.regime_promotion_decision_id,
        "owner_decision_artifact_id": owner.owner_decision_artifact_id,
        "required_capability_assessment_id": review["assessment_id"],
        "review_id": review["review_id"],
    }
