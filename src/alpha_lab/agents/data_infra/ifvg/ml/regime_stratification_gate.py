"""The status gate every stratified report is built under (R6.1 §6.G; D4).

``resolve_report_gate`` loads the EXACT promotion decision, its capability
assessment, the protocol, and — for FEATURE_ELIGIBLE-or-later classes — the
exact owner artifact, from the verified store, and refuses (with
``RegimeStatusRefusalError``, whose text always ends in "nothing here
promotes") another protocol, a fit set the assessment does not cover, an
owner artifact that is not the decision's own ratification reference, and
any status/role below the class minimum. No "latest decision" lookup exists:
every id is supplied exactly by the caller and verified-loaded.

Adversarial round (F2 / S6 / S3): the structural gate is RE-DERIVED from the
loaded assessment — a STRATIFICATION_READY-or-later decision whose assessment
fails the coverage gates or has no OOS assignment is refused whoever minted
it (D6); the owner half re-runs ``assert_owner_decision_authorizes`` under
the caller's ``run_scope`` with the store-owned supersession chain
(provenance-vs-scope, supersession, effectivity, decision values,
transition); and the ``synthetic_fixture`` scope is confined to test
namespaces (P0-4 mirror).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..search.owner_decisions import (
    OwnerDecisionArtifactEnvelope,
    OwnerDecisionRefusalError,
    assert_owner_decision_authorizes,
    assert_run_scope_lawful_for_root,
    load_owner_decision,
    load_supersession_chain,
    transition_key,
)
from ..search.store import SearchStoreError
from .regime_contracts import (
    RegimeCapabilityAssessmentEnvelope,
    RegimePromotionDecisionEnvelope,
    RegimeProtocolEnvelope,
    RegimeStatus,
    assert_lawful_role,
)
from .regime_store import load_regime_assessment, load_regime_promotion, load_regime_protocol
from .regime_stratified_contracts import (
    CLASS_MINIMUM_STATUS,
    CLASS_ROLE,
    RegimeReportGate,
    RegimeStratificationClass,
    status_meets,
    status_rank,
)

__all__ = ["RegimeStatusRefusalError", "ResolvedReportGate", "resolve_report_gate"]

_SUFFIX = "; nothing here promotes"


class RegimeStatusRefusalError(PermissionError):
    """A stratified report cannot be built under the supplied authority."""


@dataclass(frozen=True)
class ResolvedReportGate:
    gate: RegimeReportGate
    protocol: RegimeProtocolEnvelope
    decision: RegimePromotionDecisionEnvelope
    assessment: RegimeCapabilityAssessmentEnvelope
    owner_decision: OwnerDecisionArtifactEnvelope | None


def _refuse(message: str) -> RegimeStatusRefusalError:
    return RegimeStatusRefusalError(message + _SUFFIX)


def resolve_report_gate(
    root: Path,
    *,
    protocol_id: str,
    decision_id: str,
    owner_decision_artifact_id: str | None,
    comparison_class: RegimeStratificationClass,
    fit_ids: tuple[str, ...],
    run_scope: str = "full_authorized_development",
) -> ResolvedReportGate:
    """Exact-ID loads → the lawful ``RegimeReportGate`` for ``comparison_class``.

    ``run_scope`` is the owner-evidence scope the report is built under
    (the pipeline's run scope; ``synthetic_fixture`` only under the synthetic
    charter marker and only in a test namespace).
    """

    root = Path(root)
    comparison_class = RegimeStratificationClass(comparison_class)
    minimum = CLASS_MINIMUM_STATUS[comparison_class]
    role = CLASS_ROLE[comparison_class]
    try:
        assert_run_scope_lawful_for_root(root, run_scope)
    except OwnerDecisionRefusalError as error:
        raise _refuse(str(error)) from error
    try:
        protocol = load_regime_protocol(root, protocol_id)
    except SearchStoreError as error:
        raise _refuse("the regime protocol is not a verified entry of this store") from error
    try:
        decision = load_regime_promotion(root, decision_id)
    except SearchStoreError as error:
        raise _refuse("the promotion decision is not a verified entry of this store") from error
    if decision.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id:
        raise _refuse("the promotion decision belongs to another regime protocol")
    try:
        assessment = load_regime_assessment(root, decision.payload.capability_assessment_ref)
    except SearchStoreError as error:
        raise _refuse(
            "the decision's capability assessment is not a verified entry of this store"
        ) from error
    if assessment.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id:
        raise _refuse("the capability assessment belongs to another regime protocol")
    uncovered = sorted(set(fit_ids) - set(assessment.payload.regime_fit_ids))
    if not fit_ids or uncovered:
        raise _refuse(
            "the report's fit set is not covered by the decision's capability assessment"
            + (f" ({len(uncovered)} uncovered fit id(s))" if uncovered else " (empty fit set)")
        )
    status = RegimeStatus(decision.payload.status)
    if not status_meets(status, minimum):
        raise _refuse(
            f"{comparison_class.value} requires status {minimum.value} or later; the "
            f"decision status is {status.value}"
        )
    # D6 (F2): the structural gate is re-derived from the LOADED assessment —
    # the status label alone is never trusted, whoever minted the decision.
    if status_meets(status, RegimeStatus.STRATIFICATION_READY) and not (
        assessment.payload.coverage.coverage_gates_passed
        and assessment.payload.oos_assignment_available
    ):
        raise _refuse(
            f"the decision carries status {status.value} but its capability assessment "
            "does not pass the structural gates STRATIFICATION_READY requires "
            "(coverage gates AND an OOS assignment; D6)"
        )
    try:
        assert_lawful_role(role, status)
    except ValueError as error:
        raise _refuse(f"role {role.value} is not lawful at status {status.value}: {error}") from (
            error
        )
    owner: OwnerDecisionArtifactEnvelope | None = None
    needs_owner = status_rank(minimum) >= status_rank(RegimeStatus.FEATURE_ELIGIBLE)
    if needs_owner:
        if owner_decision_artifact_id is None:
            raise _refuse(
                f"{comparison_class.value} requires the exact owner decision artifact id"
            )
        if decision.payload.owner_ratification_ref != owner_decision_artifact_id:
            raise _refuse(
                "the supplied owner decision artifact is not the decision's own "
                "ratification reference"
            )
        try:
            owner = load_owner_decision(root, owner_decision_artifact_id)
        except SearchStoreError as error:
            raise _refuse(
                "the owner decision artifact is not a verified entry of this store"
            ) from error
        except OwnerDecisionRefusalError as error:
            raise _refuse(f"owner evidence refused: {error}") from error
        if owner.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id:
            raise _refuse("the owner decision artifact ratifies another regime protocol")
        if owner.payload.capability_assessment_id != assessment.regime_capability_assessment_id:
            raise _refuse("the owner decision artifact reviewed another capability assessment")
        # S6: the owner half re-runs the full authorization under the run
        # scope with the store-owned supersession chain — a superseded,
        # expired, mis-valued, or synthetic-in-a-real-scope artifact refuses.
        try:
            assert_owner_decision_authorizes(
                owner,
                protocol_envelope=protocol,
                assessment_envelope=assessment,
                transition=transition_key(decision.payload.previous_status, status),
                as_of=decision.payload.decided_at,
                run_scope=run_scope,  # type: ignore[arg-type]
                supersession_chain=load_supersession_chain(root),
            )
        except OwnerDecisionRefusalError as error:
            raise _refuse(f"owner evidence refused: {error}") from error
    elif owner_decision_artifact_id is not None and (
        decision.payload.owner_ratification_ref not in (None, owner_decision_artifact_id)
    ):
        raise _refuse(
            "the supplied owner decision artifact is not the decision's own "
            "ratification reference"
        )
    gate = RegimeReportGate(
        regime_promotion_decision_id=decision.regime_promotion_decision_id,
        owner_decision_artifact_id=owner_decision_artifact_id if needs_owner else None,
        capability_assessment_id=assessment.regime_capability_assessment_id,
        status_at_report=status,
        role_at_report=role,
        minimum_status_required=minimum,
        authority_source="frozen_owner_evidence" if needs_owner else "s10_structural",
    )
    return ResolvedReportGate(
        gate=gate,
        protocol=protocol,
        decision=decision,
        assessment=assessment,
        owner_decision=owner,
    )
