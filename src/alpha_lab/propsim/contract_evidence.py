"""Field-level contract-evidence compiler (CS §5.5; P0-17 + P1-A).

Source documents → per-field evidence rows → compilation report → owner
review → supersession. Every normalized contract field carries first-party
provenance, a locator, and a conflict state; runtime uses only frozen
compiled envelopes and never scrapes live pages. The verification-status
ladder is one-way and synthetic evidence can NEVER produce
``first_party_verified``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    OwnerDecisionEvidenceRef,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    FrozenContract,
    canonical_contract_sha256,
)
from alpha_lab.propsim.firm_contracts import PropFirmContractPayload

__all__ = [
    "PropContractSourceDocument",
    "PropRuleEvidence",
    "PropContractDraft",
    "PropContractCompilationReport",
    "PropContractReviewDecision",
    "PropContractSupersession",
    "ContractStatusError",
    "SupersededContractError",
    "assert_contract_not_superseded",
    "compile_contract_draft",
    "advance_verification_status",
    "REQUIRED_EVIDENCE_FIELDS",
]


class ContractStatusError(PermissionError):
    """An unlawful verification-status transition was attempted."""


class PropContractSourceDocument(FrozenContract):
    source_document_id: str
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    provenance_url: str
    retrieved_at_utc: str
    effective_from: str | None
    effective_to: str | None
    document_kind: Literal[
        "official_rules", "official_faq", "checkout_screen", "official_terms"
    ]

    @property
    def is_synthetic(self) -> bool:
        return self.provenance_url.startswith("synthetic://")


class PropRuleEvidence(FrozenContract):
    field_path: str
    source_document_id: str
    locator: str
    normalized_value_json: str
    reviewer: str | None
    conflict_status: Literal[
        "none", "conflicting_sources", "missing_effective_date", "unresolved"
    ]


class PropContractDraft(FrozenContract):
    contract_payload: PropFirmContractPayload
    evidence_rows: tuple[PropRuleEvidence, ...]


class PropContractCompilationReport(FrozenContract):
    draft_hash: str = Field(pattern=SHA256_PATTERN)
    field_coverage_complete: bool
    unresolved_conflicts: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    passed: bool


class PropContractReviewDecision(FrozenContract):
    draft_hash: str = Field(pattern=SHA256_PATTERN)
    owner_decision_ref: OwnerDecisionEvidenceRef
    verdict: Literal["approved", "rejected", "needs_revision"]


class PropContractSupersession(FrozenContract):
    prior_firm_contract_id: str = Field(pattern=SHA256_PATTERN)
    replacement_firm_contract_id: str = Field(pattern=SHA256_PATTERN)
    reason: str
    effective_at: str


class SupersededContractError(PermissionError):
    """A retired (superseded) firm contract id was offered for new work."""


def assert_contract_not_superseded(
    firm_contract_id: str,
    supersessions: tuple[PropContractSupersession, ...],
) -> None:
    """Refuse a retired contract id for any NEW simulation/authorization.

    Supersession retires the PRIOR id: existing immutable artifacts that cite
    it remain readable, but no new work may bind to it — the replacement id
    must be used instead.
    """

    for supersession in supersessions:
        if supersession.prior_firm_contract_id == firm_contract_id:
            raise SupersededContractError(
                f"firm contract {firm_contract_id[:12]}… was superseded "
                f"({supersession.reason!r}, effective {supersession.effective_at}); "
                f"use {supersession.replacement_firm_contract_id[:12]}… instead"
            )


#: The normalized field paths every compiled contract must evidence — the
#: result-changing rule surface (structural/reference fields are excluded).
REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "evaluation.starting_balance",
    "evaluation.profit_target",
    "evaluation.trail_amount",
    "evaluation.trail_style",
    "evaluation.trail_locks_at_start",
    "evaluation.dll_amount",
    "evaluation.dll_hard",
    "evaluation.consistency_pct",
    "evaluation.min_days",
    "evaluation.max_eval_days",
    "evaluation.breach_observation_policy",
    "evaluation.max_contracts",
    "funded.trail_amount",
    "funded.trail_style",
    "funded.dll_amount",
    "funded.breach_observation_policy",
    "payout.waiting_period",
    "payout.min_between_payouts",
    "payout.split_pct_trader",
    "payout.payout_cap_per_period",
    "payout.post_payout_buffer_rule",
    "fees.evaluation_fee",
    "fees.activation_fee",
    "fees.recurring_fee",
    "fees.reset_fee",
)


def compile_contract_draft(
    draft: PropContractDraft,
    documents: tuple[PropContractSourceDocument, ...],
) -> PropContractCompilationReport:
    """Field-coverage + conflict compilation; passes ONLY when complete/clean.

    Every required field path needs at least one evidence row whose source
    document exists; any non-``none`` conflict state blocks. The report never
    upgrades a verification status — it is the precondition for one.
    """

    known_documents = {document.source_document_id for document in documents}
    evidenced = {row.field_path for row in draft.evidence_rows}
    orphans = sorted(
        row.field_path
        for row in draft.evidence_rows
        if row.source_document_id not in known_documents
    )
    required = set(REQUIRED_EVIDENCE_FIELDS)
    if draft.contract_payload.funded is None:
        required = {path for path in required if not path.startswith("funded.")}
    if draft.contract_payload.payout is None:
        required = {path for path in required if not path.startswith("payout.")}
    missing = sorted(required - evidenced)
    conflicts = sorted(
        f"{row.field_path}:{row.conflict_status}"
        for row in draft.evidence_rows
        if row.conflict_status != "none"
    )
    unresolved = tuple(conflicts + [f"orphan_source:{path}" for path in orphans])
    complete = not missing
    return PropContractCompilationReport(
        draft_hash=canonical_contract_sha256(draft),
        field_coverage_complete=complete,
        unresolved_conflicts=unresolved,
        source_document_ids=tuple(sorted(known_documents)),
        passed=complete and not unresolved,
    )


_LADDER: tuple[str, ...] = (
    "synthetic_fixture_verified",
    "first_party_evidence_compiled",
    "owner_reviewed",
    "first_party_verified",
)


def advance_verification_status(
    current: str,
    target: str,
    *,
    documents: tuple[PropContractSourceDocument, ...],
    compilation: PropContractCompilationReport | None = None,
    review: PropContractReviewDecision | None = None,
) -> str:
    """One-way ladder enforcement (P1-A). Synthetic evidence caps the ladder.

    - any → ``superseded`` is always lawful (retirement);
    - synthetic documents can never support anything beyond
      ``synthetic_fixture_verified``;
    - ``first_party_evidence_compiled`` requires a PASSED compilation over
      exclusively first-party documents;
    - ``owner_reviewed`` requires an approving owner review of that draft;
    - ``first_party_verified`` requires the full chain (compiled + reviewed)
      and is unreachable for synthetic evidence, with no override.
    """

    if target == "superseded":
        return target
    if target not in _LADDER:
        raise ContractStatusError(f"unknown verification status {target!r}")
    if current == "superseded":
        raise ContractStatusError("a superseded contract can never advance")
    if current in _LADDER and _LADDER.index(target) < _LADDER.index(current):
        raise ContractStatusError(
            f"the verification ladder is one-way ({current} → {target} refused)"
        )
    if compilation is not None:
        # the ladder judges the EXACT document set the compilation cited —
        # presenting a different (or empty) set cannot launder synthetic
        # evidence past the cap
        presented = {document.source_document_id for document in documents}
        cited = set(compilation.source_document_ids)
        if not cited <= presented:
            raise ContractStatusError(
                "the compilation cites source documents that were not "
                f"presented for status advancement: {sorted(cited - presented)}"
            )
    synthetic = any(document.is_synthetic for document in documents)
    if synthetic and target != "synthetic_fixture_verified":
        raise ContractStatusError(
            "synthetic fixture evidence can never support "
            f"{target!r} — synthetic contracts prove the machinery only"
        )
    if target in (
        "first_party_evidence_compiled",
        "owner_reviewed",
        "first_party_verified",
    ) and (compilation is None or not compilation.passed):
        raise ContractStatusError(
            f"{target!r} requires a PASSED field-level compilation"
        )
    if target in ("owner_reviewed", "first_party_verified"):
        if review is None or review.verdict != "approved":
            raise ContractStatusError(
                f"{target!r} requires an approving owner review"
            )
        if compilation is not None and review.draft_hash != compilation.draft_hash:
            raise ContractStatusError(
                "the owner review is for a DIFFERENT draft than the compilation"
            )
    return target
