"""Computation-path-scoped owner authorization (CONTRACTS_AND_SCHEMAS.md §3.2).

A run fails only on decisions its actual computation path requires: a
strategy-only search never demands prop or regime decisions; prop studies add
the firm/fidelity/risk/withdrawal/clock set; regime studies add grain/count/
stability decisions; model-gated replay adds ``RejectedCandidatePolicy``. The
real verification slice requires a real :class:`VerificationAuthorizationRef`;
the typed synthetic marker is confined to fully synthetic fixtures.

HARDENING-BACKEND §4.1 / §4.2 (F-11 / F-12): every real authorization bundle
and every verification authorization binds the SEMANTIC ``store_namespace_id``
of the store it authorizes and the ``supersession_head_witness``
(``{store_namespace_id, line_count, head_sha256}``) it was signed against;
:func:`validate_owner_authorization` verifies both against the store when
``store_root`` is given (publication and launch seams always pass it) — a
missing, shorter, or different current head is refused. Local rollback
detection only, never cryptographic owner authenticity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, model_validator

from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .store_namespace import (
    StoreNamespaceError,
    SupersessionHeadWitness,
    assert_namespace_deployment_coherent,
    require_store_namespace,
)

if TYPE_CHECKING:
    from ..study.computation_path import ComputationPath

__all__ = [
    "OwnerDecisionEvidenceRef",
    "AuthorizationRequirement",
    "AuthorizationRequirementSetPayload",
    "AuthorizationRequirementSetEnvelope",
    "OwnerAuthorizationBundle",
    "SyntheticAuthorizationMarker",
    "VerificationAuthorizationRef",
    "AuthorizationError",
    "derive_authorization_requirements",
    "assert_authorization_bound_to_store",
    "validate_owner_authorization",
    "RunScope",
]

RunScope = Literal["synthetic_fixture", "verification_5d", "full_authorized_development"]

#: The decision key a verification run satisfies with a real
#: :class:`VerificationAuthorizationRef` (owner decisions 21 + R-5).
VERIFICATION_FIXTURE_DECISION_KEY = "21/R-5:verification_fixture_authorization"


class OwnerDecisionEvidenceRef(FrozenContract):
    decision_id: str
    decision_artifact_id: str
    content_hash: str = Field(pattern=SHA256_PATTERN)
    author: str
    approved_at: str
    effective_from: str
    reviewed_evidence_refs: tuple[str, ...]


class AuthorizationRequirement(FrozenContract):
    decision_key: str
    reason: str
    required_for_stage_ids: tuple[str, ...]
    required_for_dimension_ids: tuple[str, ...]
    required_for_scope: tuple[str, ...]


class AuthorizationRequirementSetPayload(FrozenContract):
    requirements: tuple[AuthorizationRequirement, ...]


class AuthorizationRequirementSetEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "requirement_set_id"

    requirement_set_id: str = Field(pattern=SHA256_PATTERN)
    payload: AuthorizationRequirementSetPayload


class OwnerAuthorizationBundle(FrozenContract):
    requirement_set_id: str = Field(pattern=SHA256_PATTERN)
    decision_refs: ImmutableMap[str, OwnerDecisionEvidenceRef]
    #: HARDENING-BACKEND §4.1 / §4.2: the semantic namespace the bundle
    #: authorizes and the supersession head it was signed against.
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness

    @model_validator(mode="after")
    def _witness_names_the_namespace(self):
        if self.supersession_head_witness.store_namespace_id != self.store_namespace_id:
            raise ValueError("the supersession head witness names another store namespace")
        return self


class SyntheticAuthorizationMarker(FrozenContract):
    kind: Literal["synthetic_test_authorization_v1"] = "synthetic_test_authorization_v1"


class VerificationAuthorizationRef(FrozenContract):
    """The owner's immutable approval of the real ≤5-day verification fixture."""

    verification_policy_id: str
    approved_allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    coverage_matrix_artifact_id: str = Field(pattern=SHA256_PATTERN)
    seed_snapshot_id: str = Field(pattern=SHA256_PATTERN)
    approved_by: str
    approved_at: str
    content_hash: str = Field(pattern=SHA256_PATTERN)
    #: HARDENING-BACKEND §4.1 / §4.2: the semantic namespace of the
    #: verification store (``search_test/v1``, a ``test`` namespace) and the
    #: supersession head the owner signed against.
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness

    @model_validator(mode="after")
    def _witness_names_the_namespace(self):
        if self.supersession_head_witness.store_namespace_id != self.store_namespace_id:
            raise ValueError("the supersession head witness names another store namespace")
        return self


class AuthorizationError(PermissionError):
    """A launch was refused because required owner evidence is absent/invalid."""


def _requirement(
    key: str,
    reason: str,
    *,
    stages: tuple[str, ...] = (),
    dimensions: tuple[str, ...] = (),
    scope: tuple[str, ...] = (),
) -> AuthorizationRequirement:
    return AuthorizationRequirement(
        decision_key=key,
        reason=reason,
        required_for_stage_ids=stages,
        required_for_dimension_ids=dimensions,
        required_for_scope=scope,
    )


def derive_authorization_requirements(
    run_scope: RunScope,
    study_dimensions: tuple[str, ...],
    computation_path: ComputationPath | None,
    enabled_pipeline_stages: tuple[str, ...],
    selected_firms: tuple[str, ...],
) -> AuthorizationRequirementSetEnvelope:
    """Yield exactly the decision set a run actually needs — nothing more."""

    requirements: list[AuthorizationRequirement] = []
    stages = tuple(enabled_pipeline_stages)
    dimensions = tuple(study_dimensions)

    if run_scope == "synthetic_fixture":
        # Synthetic fixtures require the typed marker at the charter layer,
        # never owner evidence.
        return AuthorizationRequirementSetEnvelope.from_payload(
            AuthorizationRequirementSetPayload(requirements=())
        )

    if run_scope == "verification_5d":
        requirements.append(
            _requirement(
                VERIFICATION_FIXTURE_DECISION_KEY,
                "the real five-day slice requires the owner-approved coverage "
                "matrix, allowlist sign-off, and VerificationAuthorizationRef",
                scope=("verification_5d",),
            )
        )
        return AuthorizationRequirementSetEnvelope.from_payload(
            AuthorizationRequirementSetPayload(requirements=tuple(requirements))
        )

    strategy_search = computation_path is not None and computation_path.full_strategy_replay
    strategy_dimensions = tuple(
        dim for dim in dimensions if dim.startswith("strategy_profile.")
    )
    if strategy_search and strategy_dimensions:
        requirements.extend(
            (
                _requirement(
                    "1:first_search_axes",
                    "the exact authorized FSM search axes are an owner decision",
                    dimensions=strategy_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "2:axis_values",
                    "every searched value (including the unbounded baseline) needs "
                    "value-level ratification",
                    dimensions=strategy_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "7:strategy_gate_thresholds",
                    "research strategy-gate thresholds must be owner-ratified",
                    dimensions=strategy_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "R-2:authorization_workflow",
                    "the owner-authorization bundle workflow itself must be ratified",
                    scope=("full_authorized_development",),
                ),
            )
        )

    prop_path = computation_path is not None and (
        computation_path.prop_resimulation or computation_path.bootstrap_resimulation
    )
    prop_dimensions = tuple(
        dim
        for dim in dimensions
        if dim.startswith(("prop_contract.", "risk_policy.", "payout_policy.", "portfolio_policy."))
    )
    if prop_path and (prop_dimensions or selected_firms):
        requirements.extend(
            (
                _requirement(
                    "5:firms_in_v1",
                    "which firm contracts enter a real study is an owner decision",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "6:contract_evidence_compiler",
                    "real prop use requires first_party_verified compiled contracts",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "8:prop_gate_thresholds",
                    "payout-reliability benchmark thresholds must be owner-ratified",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "R-3:withdrawal_policies",
                    "the withdrawal-policy set authorized for real studies",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "R-4:path_capability_matrix",
                    "the per-rule path-capability and accepted-fidelity matrix",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "10:clock_policy",
                    "clock semantics under resampling (SimulatedClockPolicy) acceptance",
                    dimensions=prop_dimensions,
                    scope=("full_authorized_development",),
                ),
            )
        )

    regime_dimensions = tuple(dim for dim in dimensions if dim.startswith("regime."))
    regime_feature_intent = any(
        dim.startswith("regime.feature") or dim == "regime.feature_eligibility"
        for dim in dimensions
    )
    if regime_dimensions and regime_feature_intent:
        requirements.extend(
            (
                _requirement(
                    "28:regime_grain",
                    "the first regime observation grain and stage",
                    dimensions=regime_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "29:cluster_counts",
                    "cluster counts are never selected by trading results",
                    dimensions=regime_dimensions,
                    scope=("full_authorized_development",),
                ),
                _requirement(
                    "30:occupancy_stability_gates",
                    "occupancy/confidence/stability gates are proposed defaults",
                    dimensions=regime_dimensions,
                    scope=("full_authorized_development",),
                ),
            )
        )

    gated = (
        computation_path is not None and computation_path.model_gated_sequential_replay
    ) or any(stage.endswith("11_run_frozen_model_gated_replays") for stage in stages)
    if gated:
        requirements.append(
            _requirement(
                "R-1:rejected_candidate_policy",
                "model-gated execution is unavailable until RejectedCandidatePolicy "
                "is owner-ratified and sequential golden tests pass",
                stages=tuple(s for s in stages if s.endswith("11_run_frozen_model_gated_replays")),
                scope=("full_authorized_development",),
            )
        )

    return AuthorizationRequirementSetEnvelope.from_payload(
        AuthorizationRequirementSetPayload(requirements=tuple(requirements))
    )


def assert_authorization_bound_to_store(
    store_root: Path,
    *,
    store_namespace_id: str,
    supersession_head_witness: SupersessionHeadWitness,
    expected_namespace_class: str | None = None,
) -> None:
    """HARDENING-BACKEND §4.1 / §4.2: the authorization must name THIS store's
    verified namespace and its witness must equal the CURRENT supersession
    head (missing, shorter, or different refuses). HARDENING-BACKEND-FIX §10:
    the head witness alone is insufficient — the COMPLETE chain from the
    genesis anchor to the current head is proven, every superseded and
    replacement owner decision verified-loaded, before the authority is
    accepted (the ONE shared proof every real seam runs). Typed
    :class:`AuthorizationError` carrying the namespace ``reason``."""

    from .owner_decisions import verify_complete_owner_authority_chain  # noqa: PLC0415

    try:
        namespace = require_store_namespace(
            Path(store_root), expected_class=expected_namespace_class
        )
        # RA-09: the deployment check (a ``test`` namespace under a
        # research-looking path) runs at EVERY bound seam — defense in depth
        assert_namespace_deployment_coherent(Path(store_root), namespace)
        if namespace.store_namespace_id != store_namespace_id:
            raise StoreNamespaceError(
                "store_namespace_identity_mismatch",
                "the authorization names another store namespace "
                f"({store_namespace_id[:12]}… ≠ {namespace.store_namespace_id[:12]}…)",
            )
        verify_complete_owner_authority_chain(
            Path(store_root),
            expected_head_witness=supersession_head_witness,
            store_namespace_id=store_namespace_id,
        )
    except StoreNamespaceError as error:
        refusal = AuthorizationError(f"owner authorization is not bound to this store: {error}")
        refusal.reason = error.reason  # type: ignore[attr-defined]
        raise refusal from error


def validate_owner_authorization(
    bundle: OwnerAuthorizationBundle,
    requirement_set: AuthorizationRequirementSetEnvelope,
    *,
    as_of_utc: str,
    superseded_artifact_ids: tuple[str, ...] = (),
    store_root: Path | None = None,
) -> None:
    """Fail closed on absent, stale, superseded, or inconsistent evidence.
    With ``store_root`` (every publication / launch seam passes it) the
    bundle's namespace and supersession-head witness are verified against
    the store as well; without it only the store-independent binding runs
    (a draft-time check — never a launch)."""

    if bundle.requirement_set_id != requirement_set.requirement_set_id:
        raise AuthorizationError(
            "authorization bundle references a different requirement set"
        )
    if store_root is not None:
        assert_authorization_bound_to_store(
            Path(store_root),
            store_namespace_id=bundle.store_namespace_id,
            supersession_head_witness=bundle.supersession_head_witness,
        )
    problems: list[str] = []
    for requirement in requirement_set.payload.requirements:
        ref = bundle.decision_refs.get(requirement.decision_key)
        if ref is None:
            problems.append(f"missing evidence for {requirement.decision_key}")
            continue
        if ref.decision_artifact_id in superseded_artifact_ids:
            problems.append(f"superseded evidence for {requirement.decision_key}")
        if ref.effective_from > as_of_utc:
            problems.append(
                f"evidence for {requirement.decision_key} is not yet effective"
            )
    if problems:
        raise AuthorizationError(
            "owner authorization is incomplete: " + "; ".join(sorted(problems))
        )


def _example_requirement_set() -> AuthorizationRequirementSetPayload:
    return AuthorizationRequirementSetPayload(
        requirements=(
            _requirement(
                "1:first_search_axes",
                "example",
                dimensions=("strategy_profile.parent_retest_timeout_1m_bars",),
                scope=("full_authorized_development",),
            ),
        )
    )


register_identity_pair(
    name="AuthorizationRequirementSet",
    envelope_cls=AuthorizationRequirementSetEnvelope,
    payload_cls=AuthorizationRequirementSetPayload,
    id_field="requirement_set_id",
    example_factory=_example_requirement_set,
)
