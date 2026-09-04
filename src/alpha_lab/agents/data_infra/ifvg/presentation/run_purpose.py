"""Presentation-only run purpose (UI-1; plan §5.6, §8, owner Q1).

One user-facing purpose authority replaces the independent "Artifact
namespace" and "Run scope" radios:

* ``RunPurpose`` — *Implementation Verification* → ``verification_5d`` in the
  ``test`` namespace (``search_test/v1``); *Development Research* →
  ``full_authorized_development`` in the ``research`` namespace with a
  restricted question-derived stage plan and publication unavailable by
  default; *Full Authorized Development* → ``full_authorized_development``,
  ``research``, the full owner-authorized stage plan, publication a separate
  eligibility gate. ``RunScope`` values are unchanged.
* ``EvidenceClass`` — the actual computation path: a *synthetic fixture*
  (the typed :class:`SyntheticAuthorizationMarker`, confined to
  Implementation Verification in a ``test`` namespace) or *real* evidence (a
  validated ``VerificationAuthorizationRef`` for the real ≤5-day slice; a
  computation-path-scoped ``OwnerAuthorizationBundle`` for research scopes).
* ``RunPurposeAnnotation`` — the MUTABLE, non-semantic record persisted with
  a draft (and as a catalog annotation after freeze). It never enters a
  charter or pipeline identity and cannot bypass the ``RunScope``,
  namespace, stage-plan or authorization validators.
* ``resolve_draft_purpose`` — a stored annotation wins; otherwise a purpose
  is derived ONLY when the legacy run scope (+ stage plan + authorization
  kind) yields one unambiguous purpose; otherwise ``purpose_unresolved`` —
  the owner confirms before the draft can freeze.

The semantic ``store_namespace_id`` comes ONLY from the verified
``StoreNamespaceEnvelope`` of the store (``namespace_state_for_store``); a
local filesystem path never defines authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "AUTHORIZATION_READINESS_STATUSES",
    "NAMESPACE_STATUSES",
    "PURPOSE_LABELS",
    "PURPOSE_DESCRIPTIONS",
    "PURPOSE_RUN_SCOPE",
    "PURPOSE_NAMESPACE_CLASS",
    "PURPOSE_RESULT_LABEL",
    "PURPOSE_STAGE_PLAN_POLICY",
    "PURPOSE_PUBLICATION_AVAILABLE",
    "EvidenceClass",
    "NamespaceState",
    "PurposeResolution",
    "PurposeResolutionStatus",
    "ResolvedPurpose",
    "RunPurpose",
    "RunPurposeAnnotation",
    "derive_purpose_from_legacy",
    "namespace_state_for_store",
    "resolve_draft_purpose",
    "resolve_purpose",
]

ANNOTATION_SCHEMA_VERSION = 1


class RunPurpose(StrEnum):
    IMPLEMENTATION_VERIFICATION = "implementation_verification"
    DEVELOPMENT_RESEARCH = "development_research"
    FULL_AUTHORIZED_DEVELOPMENT = "full_authorized_development"


class EvidenceClass(StrEnum):
    """The actual computation path's evidence class."""

    SYNTHETIC_FIXTURE = "synthetic_fixture"
    REAL = "real"


class PurposeResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    PURPOSE_UNRESOLVED = "purpose_unresolved"


PURPOSE_LABELS: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: "Implementation Verification",
        RunPurpose.DEVELOPMENT_RESEARCH: "Development Research",
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: "Full Authorized Development",
    }
)

PURPOSE_DESCRIPTIONS: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: (
            "Prove the implementation over the exact baseline in the test "
            "namespace (search_test/v1): a synthetic fixture proves the machinery; "
            "the real ≤5-day slice needs the owner's VerificationAuthorizationRef. "
            "Verification results are never research evidence and never publish."
        ),
        RunPurpose.DEVELOPMENT_RESEARCH: (
            "Answer one research question over the authorized development data "
            "in the research namespace (search/v1) with a restricted, "
            "question-derived stage plan. Publication is unavailable by default "
            "and separately gated; every real run needs the computation-path-scoped "
            "owner authorization bundle."
        ),
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: (
            "Run the full owner-authorized stage plan over the authorized "
            "development data in the research namespace (search/v1). Publication "
            "is a separate eligibility gate; the launch needs the typed "
            "acknowledgement and a ready owner authorization bundle."
        ),
    }
)

#: Owner Q1: the RunScope values are unchanged.
PURPOSE_RUN_SCOPE: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: "verification_5d",
        RunPurpose.DEVELOPMENT_RESEARCH: "full_authorized_development",
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: "full_authorized_development",
    }
)

PURPOSE_NAMESPACE_CLASS: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: "test",
        RunPurpose.DEVELOPMENT_RESEARCH: "research",
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: "research",
    }
)

PURPOSE_RESULT_LABEL: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: (
            "Verification-only result — not research evidence"
        ),
        RunPurpose.DEVELOPMENT_RESEARCH: (
            "Development research result (Development Exploratory Representative)"
        ),
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: (
            "Full authorized development result (Development Exploratory Representative)"
        ),
    }
)

PURPOSE_STAGE_PLAN_POLICY: Mapping[RunPurpose, str] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: "verification_baseline",
        RunPurpose.DEVELOPMENT_RESEARCH: "question_derived_restricted",
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: "full_owner_authorized",
    }
)

PURPOSE_PUBLICATION_AVAILABLE: Mapping[RunPurpose, bool] = MappingProxyType(
    {
        RunPurpose.IMPLEMENTATION_VERIFICATION: False,
        RunPurpose.DEVELOPMENT_RESEARCH: False,
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT: True,
    }
)

_STAGE_PLAN_KINDS = frozenset(PURPOSE_STAGE_PLAN_POLICY.values())

#: The typed authorization readiness states the providers derive (plan
#: §6.1 / §9 Phase 1): the UI renders every one of them and never collapses
#: them into a Boolean "present".
AUTHORIZATION_READINESS_STATUSES: tuple[str, ...] = (
    "not_required",  # a fully synthetic fixture needs the typed marker only
    "ready",
    "missing",
    "stale_head",
    "superseded",
    "wrong_namespace",
    "wrong_head",
    "wrong_profile",
    "wrong_source",
    "not_effective",
    "store_unmarked",
    "store_corrupt",
    "store_incoherent",
    "unavailable",  # the lookup itself failed (sanitized detail)
)

NAMESPACE_STATUSES: tuple[str, ...] = (
    "verified",
    "unmarked",
    "corrupt",
    "deployment_incoherent",
    "class_mismatch",
    "unavailable",
)

_AUTHORIZATION_CLASS = Literal[
    "synthetic_marker", "verification_authorization_ref", "owner_authorization_bundle"
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunPurposeAnnotation:
    """Mutable presentation record — never part of a semantic identity."""

    purpose: RunPurpose
    derivation: str  # card_selected | owner_confirmed | derived_from_legacy_scope | cloned
    owner_confirmed: bool
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "purpose": self.purpose.value,
            "derivation": str(self.derivation),
            "owner_confirmed": bool(self.owner_confirmed),
            "updated_at": str(self.updated_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> RunPurposeAnnotation | None:
        """``None`` for an absent or unreadable annotation (never a guess)."""

        if not isinstance(payload, Mapping):
            return None
        try:
            purpose = RunPurpose(str(payload.get("purpose")))
        except ValueError:
            return None
        return cls(
            purpose=purpose,
            derivation=str(payload.get("derivation") or "card_selected"),
            owner_confirmed=bool(payload.get("owner_confirmed", False)),
            updated_at=str(payload.get("updated_at") or ""),
        )


def derive_purpose_from_legacy(
    *,
    run_scope: str | None,
    stage_plan_kind: str | None = None,
    authorization_kind: str | None = None,
) -> RunPurpose | None:
    """The one unambiguous purpose a legacy draft/charter implies, or ``None``.

    ``verification_5d`` (and the synthetic fixture scope) map only to
    Implementation Verification. ``full_authorized_development`` is shared by
    Development Research and Full Authorized Development: it resolves only
    when the stage-plan kind (and, for the full plan, the owner-bundle
    authorization kind) disambiguates it; otherwise the draft is
    ``purpose_unresolved`` and the owner confirms it explicitly.
    """

    if run_scope in ("verification_5d", "synthetic_fixture"):
        return RunPurpose.IMPLEMENTATION_VERIFICATION
    if run_scope != "full_authorized_development":
        return None
    if stage_plan_kind == "question_derived_restricted":
        return RunPurpose.DEVELOPMENT_RESEARCH
    if (
        stage_plan_kind == "full_owner_authorized"
        and authorization_kind == "owner_authorization_bundle"
    ):
        return RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
    return None


@dataclass(frozen=True)
class PurposeResolution:
    status: PurposeResolutionStatus
    purpose: RunPurpose | None
    annotation: RunPurposeAnnotation | None
    reason: str


def resolve_draft_purpose(
    annotation_payload: Mapping[str, Any] | None,
    *,
    run_scope: str | None,
    stage_plan_kind: str | None = None,
    authorization_kind: str | None = None,
    now: str | None = None,
) -> PurposeResolution:
    """Plan §5.6: stored annotation → legacy derivation → ``purpose_unresolved``."""

    stored = RunPurposeAnnotation.from_dict(annotation_payload)
    if stored is not None:
        return PurposeResolution(
            status=PurposeResolutionStatus.RESOLVED,
            purpose=stored.purpose,
            annotation=stored,
            reason="stored purpose annotation",
        )
    derived = derive_purpose_from_legacy(
        run_scope=run_scope,
        stage_plan_kind=stage_plan_kind,
        authorization_kind=authorization_kind,
    )
    if derived is not None:
        return PurposeResolution(
            status=PurposeResolutionStatus.RESOLVED,
            purpose=derived,
            annotation=RunPurposeAnnotation(
                purpose=derived,
                derivation="derived_from_legacy_scope",
                owner_confirmed=False,
                updated_at=now or _utc_now(),
            ),
            reason=f"derived unambiguously from the legacy run scope {run_scope!r}",
        )
    return PurposeResolution(
        status=PurposeResolutionStatus.PURPOSE_UNRESOLVED,
        purpose=None,
        annotation=None,
        reason=(
            "the draft carries no purpose annotation and its legacy run scope "
            f"{run_scope!r} does not derive one unambiguous purpose; the owner must "
            "confirm the purpose before this draft can freeze or launch"
        ),
    )


@dataclass(frozen=True)
class NamespaceState:
    """The store's SEMANTIC namespace as verified from its envelope."""

    namespace_class: str | None
    store_namespace_id: str | None
    status: str  # one of NAMESPACE_STATUSES
    detail: str

    def __post_init__(self) -> None:
        if self.status not in NAMESPACE_STATUSES:
            raise ValueError(f"unregistered namespace status {self.status!r}")


def namespace_state_for_store(
    store_root: Path, *, expected_class: str | None = None
) -> NamespaceState:
    """Verified-load the store namespace; typed states, never a path guess."""

    from ..search.store_namespace import (  # noqa: PLC0415
        StoreNamespaceError,
        assert_namespace_deployment_coherent,
        load_store_namespace,
    )

    root = Path(store_root)
    try:
        envelope = load_store_namespace(root)
    except StoreNamespaceError as error:
        if error.reason == "store_namespace_missing":
            return NamespaceState(
                namespace_class=None,
                store_namespace_id=None,
                status="unmarked",
                detail=(
                    "the store carries no STORE_NAMESPACE.json — an unmarked store has no "
                    "semantic authority (initialize it explicitly with its intended class)"
                ),
            )
        return NamespaceState(
            namespace_class=None,
            store_namespace_id=None,
            status="corrupt",
            detail=f"the store namespace envelope failed verification ({error.reason})",
        )
    except OSError as error:  # pragma: no cover - filesystem failure surfaced typed
        return NamespaceState(
            namespace_class=None,
            store_namespace_id=None,
            status="unavailable",
            detail=f"the store namespace could not be read ({type(error).__name__})",
        )
    namespace_class = str(envelope.payload.namespace_class)
    try:
        assert_namespace_deployment_coherent(root, envelope)
    except StoreNamespaceError as error:
        return NamespaceState(
            namespace_class=namespace_class,
            store_namespace_id=envelope.store_namespace_id,
            status="deployment_incoherent",
            detail=str(error),
        )
    if expected_class is not None and namespace_class != expected_class:
        return NamespaceState(
            namespace_class=namespace_class,
            store_namespace_id=envelope.store_namespace_id,
            status="class_mismatch",
            detail=(
                f"the store is a {namespace_class!r} namespace; the purpose requires "
                f"{expected_class!r}"
            ),
        )
    return NamespaceState(
        namespace_class=namespace_class,
        store_namespace_id=envelope.store_namespace_id,
        status="verified",
        detail=f"verified {namespace_class} namespace {envelope.store_namespace_id[:12]}…",
    )


@dataclass(frozen=True)
class ResolvedPurpose:
    purpose: RunPurpose
    label: str
    run_scope: str
    namespace_class: str
    store_namespace_id: str | None
    namespace_status: str
    namespace_detail: str
    evidence_class: EvidenceClass
    authorization_class: _AUTHORIZATION_CLASS
    authorization_readiness: str
    authorization_detail: str
    stage_plan_policy: str
    publication_available: bool
    result_label: str
    freeze_allowed: bool
    freeze_block_reason: str | None


def resolve_purpose(
    purpose: RunPurpose | str,
    *,
    evidence_class: EvidenceClass | str | None = None,
    namespace: NamespaceState | None = None,
    authorization_readiness: str | None = None,
    authorization_detail: str = "",
) -> ResolvedPurpose:
    """Derive scope, namespace class, authorization class, publication
    availability and the freeze verdict from the purpose and the ACTUAL
    computation path. Nothing here asserts readiness — it is consumed from the
    providers' typed derivation and rendered as-is."""

    purpose = RunPurpose(purpose)
    if evidence_class is None:
        evidence = (
            EvidenceClass.SYNTHETIC_FIXTURE
            if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
            else EvidenceClass.REAL
        )
    else:
        evidence = EvidenceClass(evidence_class)
    if (
        evidence is EvidenceClass.SYNTHETIC_FIXTURE
        and purpose is not RunPurpose.IMPLEMENTATION_VERIFICATION
    ):
        raise ValueError(
            "a synthetic fixture is confined to Implementation Verification in a test "
            f"namespace; {PURPOSE_LABELS[purpose]} requires real owner authorization"
        )
    if evidence is EvidenceClass.SYNTHETIC_FIXTURE:
        authorization_class: _AUTHORIZATION_CLASS = "synthetic_marker"
    elif purpose is RunPurpose.IMPLEMENTATION_VERIFICATION:
        authorization_class = "verification_authorization_ref"
    else:
        authorization_class = "owner_authorization_bundle"
    if authorization_readiness is None:
        readiness = "not_required" if authorization_class == "synthetic_marker" else "missing"
    else:
        readiness = str(authorization_readiness)
        if readiness not in AUTHORIZATION_READINESS_STATUSES:
            raise ValueError(f"unregistered authorization readiness {readiness!r}")
    namespace_class = PURPOSE_NAMESPACE_CLASS[purpose]
    store_namespace_id = namespace.store_namespace_id if namespace is not None else None
    namespace_status = namespace.status if namespace is not None else "unavailable"
    namespace_detail = (
        namespace.detail if namespace is not None else "the store namespace was not resolved"
    )

    block: str | None = None
    if authorization_class == "synthetic_marker":
        # the typed marker freezes into a test namespace (or an unmarked tmp
        # store — a charter is a frozen request, never a launch); a research
        # namespace refuses it at save_charter (P0-4)
        if namespace is not None and namespace.status in ("corrupt", "deployment_incoherent"):
            block = f"the store namespace is {namespace.status}: {namespace.detail}"
        elif namespace is not None and namespace.status == "class_mismatch":
            block = namespace.detail
    else:
        if namespace is None or namespace.status != "verified":
            block = (
                "a real charter requires a verified store namespace of class "
                f"{namespace_class!r}; the store is {namespace_status}"
            )
        elif readiness != "ready":
            block = (
                f"owner authorization readiness is {readiness!r}"
                + (f": {authorization_detail}" if authorization_detail else "")
            )
    return ResolvedPurpose(
        purpose=purpose,
        label=PURPOSE_LABELS[purpose],
        run_scope=PURPOSE_RUN_SCOPE[purpose],
        namespace_class=namespace_class,
        store_namespace_id=store_namespace_id,
        namespace_status=namespace_status,
        namespace_detail=namespace_detail,
        evidence_class=evidence,
        authorization_class=authorization_class,
        authorization_readiness=readiness,
        authorization_detail=authorization_detail,
        stage_plan_policy=PURPOSE_STAGE_PLAN_POLICY[purpose],
        publication_available=PURPOSE_PUBLICATION_AVAILABLE[purpose],
        result_label=PURPOSE_RESULT_LABEL[purpose],
        freeze_allowed=block is None,
        freeze_block_reason=block,
    )
