"""Verified owner-decision evidence for the regime lane (R6.1 §6.F; D5).

R6 gated FEATURE_ELIGIBLE by a 64-hex *format* check on
``owner_ratification_ref``. R6.1 replaces that with a VERIFIED-LOADED
artifact: :class:`OwnerDecisionArtifactEnvelope` (store ``owner_decisions``;
its id IS the ``OwnerDecisionEvidenceRef.content_hash``) binds the EXACT
``resolved_regime_protocol_id`` and ``capability_assessment_id`` it ratifies
and covers decisions **25/28/29/30**: the algorithm key plus the exact
pinned KMeans parameter snapshot/hash (25), grain / interval / stage (28),
k (29), and the occupancy / rows / sample floors and the stability minimum
(30). Every value is re-verified against the loaded registry, protocol, and
assessment (:func:`assert_owner_decision_authorizes`).

HARDENING-BACKEND (§4.1 / §4.2 / §4.3; F-11 / F-12 / F-13): authority is
SEMANTIC, never path-derived — every owner-decision artifact binds the
``store_namespace_id`` of the store it lives in (``search.store_namespace``;
an unmarked store can carry no owner decision), synthetic provenance is
lawful only in a ``test``-class namespace (the pathname heuristic
``research_namespace_root`` survives as a defense-in-depth deployment check
only), and supersession is an IMMUTABLE record chain
(``search.supersession_chain``: one content-addressed
``owner_decision_supersessions/<record-id>/`` entry per replacement, a
mandatory head pointer that commits to the whole chain from the namespace's
genesis anchor, publication under the liveness-aware
``search.owner_decision_lock``). The record is published BEFORE the
replacement decision (a crash in between leaves the chain fail-closed until
the replacement is re-persisted, which is idempotent); a replacement may
never carry weaker provenance than the prior; nothing ever writes into an
existing ``owner_decisions/<decision-id>/`` directory.

:func:`build_owner_decision_proposal` / :func:`render_owner_decision_proposal_markdown`
emit DRAFTS with placeholders that fail validation (they cannot be
persisted by accident) — the owner's ratification workflow replaces the
placeholders; nothing here is an authorization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..ml.regime_algorithms import (
    REGIME_ALGORITHM_REGISTRY,
    pinned_parameters_hash,
    resolve_regime_algorithm_entry,
)
from ..ml.regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    RegimeCapabilityAssessmentEnvelope,
    RegimeProtocolEnvelope,
    RegimeStatus,
    sample_adequacy_minimum,
)
from .authorization import OwnerDecisionEvidenceRef, RunScope
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
    register_identity_pair,
)
from .store import (
    SearchStoreError,
    envelope_destination,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .store_namespace import (
    OWNER_DECISION_STORE,
    SUPERSESSIONS_HEAD_FILE,
    StoreNamespaceEnvelope,
    StoreNamespaceError,
    initialize_test_namespace,
    namespace_class_of,
    path_looks_like_research_store,
    require_store_namespace,
)
from .supersession_chain import (
    OWNER_DECISION_SUPERSESSION_STORE,
    SupersessionRecord,
    load_supersession_records,
    publish_supersession,
)

__all__ = [
    "OWNER_DECISION_STORE",
    "OWNER_DECISION_SUPERSESSION_STORE",
    "SUPERSESSIONS_HEAD_FILE",
    "research_namespace_root",
    "require_owner_decision_namespace",
    "assert_run_scope_lawful_for_root",
    "REGIME_DECISION_ID",
    "REGIME_DECISION_KEYS",
    "AUTHORIZED_TRANSITIONS",
    "OwnerDecisionArtifactPayload",
    "OwnerDecisionArtifactEnvelope",
    "OwnerDecisionRefusalError",
    "SupersessionRecord",
    "expected_decision_values",
    "freeze_decision_values",
    "plain_decision_values",
    "persist_owner_decision",
    "load_owner_decision",
    "load_supersession_chain",
    "assert_owner_decision_authorizes",
    "build_owner_decision_proposal",
    "render_owner_decision_proposal_markdown",
    "synthetic_owner_decision_fixture",
]

#: A replacement may never carry WEAKER provenance than the prior it revokes.
_PROVENANCE_RANK: dict[str, int] = {"synthetic_test_authorization_v1": 0, "owner_signed": 1}
REGIME_DECISION_ID = "25+28+29+30:regime_feature_eligibility"
REGIME_DECISION_KEYS: tuple[str, ...] = (
    "25:regime_algorithm_baseline",
    "28:regime_grain",
    "29:cluster_counts",
    "30:occupancy_stability_gates",
)
AUTHORIZED_TRANSITIONS: tuple[str, ...] = (
    "stratification_ready->feature_eligible",
    "feature_eligible->model_feature",
)
_PLACEHOLDER = "<OWNER_TO_FILL>"
_VALUE_KEYS: tuple[str, ...] = (
    "algorithm_key",
    "algorithm_parameters_hash",
    "algorithm_parameters",
    "observation_granularity",
    "panel_interval_seconds",
    "observation_stage",
    "fixed_cluster_count",
    "minimum_cluster_occupancy_fraction",
    "minimum_cluster_rows_per_fold",
    "minimum_bootstrap_aligned_ami_mean",
    "minimum_training_observations",
)


class OwnerDecisionRefusalError(PermissionError):
    """The owner artifact does not authorize the requested transition."""


def _parse_instant(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry an explicit UTC offset")
    return parsed


#: A decision value is a scalar or a (deep-immutable) map of scalars — the
#: exact pinned KMeans parameter snapshot rides as the nested map.
DecisionScalar = str | int | float | bool | None
DecisionValue = DecisionScalar | ImmutableMap[str, DecisionScalar]


class OwnerDecisionArtifactPayload(FrozenContract):
    decision_id: Literal["25+28+29+30:regime_feature_eligibility"] = REGIME_DECISION_ID
    #: HARDENING-BACKEND §4.1: the SEMANTIC namespace this artifact belongs to
    #: (the store's verified ``store_namespace_id``) — never a path.
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    decision_keys: tuple[str, ...]
    decision_values: ImmutableMap[str, DecisionValue]
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    authorized_transitions: tuple[str, ...] = Field(min_length=1)
    provenance: Literal["owner_signed", "synthetic_test_authorization_v1"]
    author: str = Field(min_length=1)
    approved_at: str
    effective_from: str
    effective_to: str | None
    reviewed_evidence_refs: tuple[str, ...] = Field(min_length=1)
    supersedes: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self):
        missing = sorted(set(REGIME_DECISION_KEYS) - set(self.decision_keys))
        if missing:
            raise ValueError(f"the regime decision artifact must cover keys {missing}")
        unknown = sorted(set(self.authorized_transitions) - set(AUTHORIZED_TRANSITIONS))
        if unknown:
            raise ValueError(f"unregistered authorized transitions {unknown}")
        absent = sorted(set(_VALUE_KEYS) - set(self.decision_values))
        if absent:
            raise ValueError(f"decision_values lacks required entries {absent}")
        if self.capability_assessment_id not in self.reviewed_evidence_refs:
            raise ValueError(
                "reviewed_evidence_refs must contain the capability assessment id"
            )
        for value in self.decision_values.values():
            if isinstance(value, str) and _PLACEHOLDER in value:
                raise ValueError("a proposal placeholder cannot be persisted as a decision")
            if isinstance(value, dict | list | set):
                raise ValueError(
                    "decision_values must not carry plain mutable containers (nested "
                    "maps are ImmutableMap; the identity payload is deep-immutable)"
                )
        if _PLACEHOLDER in self.author or _PLACEHOLDER in self.rationale:
            raise ValueError("a proposal placeholder cannot be persisted as a decision")
        approved = _parse_instant(self.approved_at, field="approved_at")
        effective = _parse_instant(self.effective_from, field="effective_from")
        if effective < approved:
            raise ValueError("effective_from precedes approved_at")
        if self.effective_to is not None:
            until = _parse_instant(self.effective_to, field="effective_to")
            if until <= effective:
                raise ValueError("effective_to must follow effective_from")
        return self


class OwnerDecisionArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "owner_decision_artifact_id"

    owner_decision_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: OwnerDecisionArtifactPayload

    def evidence_ref(self) -> OwnerDecisionEvidenceRef:
        """The bare reference type the authorization layer carries."""

        return OwnerDecisionEvidenceRef(
            decision_id=self.payload.decision_id,
            decision_artifact_id=self.owner_decision_artifact_id,
            content_hash=self.owner_decision_artifact_id,
            author=self.payload.author,
            approved_at=self.payload.approved_at,
            effective_from=self.payload.effective_from,
            reviewed_evidence_refs=self.payload.reviewed_evidence_refs,
        )


def research_namespace_root(root: Path) -> bool:
    """DEFENSE-IN-DEPTH ONLY (HARDENING-BACKEND §4.1): the pre-hardening
    pathname heuristic (a resolved path with a ``search`` segment that is
    not ``search_test``). It may refuse a suspicious deployment; it never
    defines or grants authority — that is the store's verified namespace."""

    return path_looks_like_research_store(root)


def _namespace_refusal(error: StoreNamespaceError) -> OwnerDecisionRefusalError:
    refusal = OwnerDecisionRefusalError(f"owner-decision store refused: {error}")
    refusal.reason = error.reason  # type: ignore[attr-defined]
    return refusal


def require_owner_decision_namespace(root: Path) -> StoreNamespaceEnvelope:
    """The store's verified semantic namespace — an unmarked or corrupt store
    can carry no owner decision (typed refusal)."""

    try:
        return require_store_namespace(root)
    except StoreNamespaceError as error:
        raise _namespace_refusal(error) from error


def assert_run_scope_lawful_for_root(root: Path, run_scope: str) -> None:
    """The ``synthetic_fixture`` run scope is a test-namespace scope: it may
    never unlock synthetic authority in the research namespace (S3). The
    SEMANTIC rule reads the store's verified namespace class (a ``research``
    namespace refuses the scope; an unmarked store can carry no owner
    artifact, so the scope unlocks nothing there); the pathname check stays
    as defense in depth."""

    if str(run_scope) != "synthetic_fixture":
        return
    if path_looks_like_research_store(root):
        raise OwnerDecisionRefusalError(
            "the synthetic_fixture run scope is confined to test namespaces; a "
            "research-looking store path refuses it (P0-4, defense in depth)"
        )
    try:
        namespace_class = namespace_class_of(root)
    except StoreNamespaceError as error:
        raise _namespace_refusal(error) from error
    if namespace_class == "research":
        raise OwnerDecisionRefusalError(
            "the synthetic_fixture run scope is confined to test namespaces; the "
            "research namespace refuses it (P0-4)"
        )


def _assert_provenance_lawful_for_root(
    root: Path, payload: OwnerDecisionArtifactPayload, namespace: StoreNamespaceEnvelope
) -> None:
    if payload.provenance != "synthetic_test_authorization_v1":
        return
    if namespace.payload.namespace_class != "test":
        raise OwnerDecisionRefusalError(
            "synthetic_test_authorization_v1 owner decisions are confined to test namespaces; "
            f"the {namespace.payload.namespace_class} namespace refuses them (P0-4)"
        )
    if path_looks_like_research_store(root):
        raise OwnerDecisionRefusalError(
            "synthetic_test_authorization_v1 owner decisions are confined to test namespaces; "
            "a research-looking store path refuses them (P0-4, defense in depth)"
        )


def freeze_decision_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Nested maps become ``ImmutableMap`` (the identity payload is
    deep-immutable); scalars pass through."""

    return {
        key: (ImmutableMap(dict(value)) if isinstance(value, Mapping) else value)
        for key, value in values.items()
    }


def plain_decision_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON-friendly (plain dict) form of ``decision_values``."""

    return {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in values.items()
    }


def expected_decision_values(
    protocol: RegimeProtocolEnvelope, assessment: RegimeCapabilityAssessmentEnvelope
) -> dict[str, Any]:
    """The values decisions 25/28/29/30 must state for THIS protocol and
    assessment — derived from the registry, protocol, and assessment (the
    exact pinned parameter snapshot rides as an ``ImmutableMap``)."""

    payload = protocol.payload
    entry = resolve_regime_algorithm_entry(payload.algorithm_key)
    grain = ObservationGranularity(payload.observation_granularity)
    return {
        "algorithm_key": payload.algorithm_key,
        "algorithm_parameters_hash": pinned_parameters_hash(entry),
        "algorithm_parameters": ImmutableMap(dict(entry.pinned_parameters)),
        "observation_granularity": grain.value,
        "panel_interval_seconds": payload.panel_interval_seconds,
        "observation_stage": payload.observation_stage.value,
        "fixed_cluster_count": int(payload.resolved_cluster_count),
        "minimum_cluster_occupancy_fraction": float(
            assessment.payload.minimum_cluster_occupancy_gate
        ),
        "minimum_cluster_rows_per_fold": int(assessment.payload.minimum_cluster_rows_gate),
        "minimum_bootstrap_aligned_ami_mean": float(
            assessment.payload.stability.minimum_bootstrap_aligned_ami_mean_applied
        ),
        "minimum_training_observations": int(
            assessment.payload.coverage.minimum_training_observations_gate
        ),
    }


def _chain_refusal(message: str) -> OwnerDecisionRefusalError:
    return OwnerDecisionRefusalError(f"{message}; the chain fails closed")


def load_owner_decision(root: Path, artifact_id: str) -> OwnerDecisionArtifactEnvelope:
    """Exact-ID verified load. The artifact must belong to THIS store's
    verified namespace (an unmarked store refuses; an artifact copied in
    from another namespace refuses) and a synthetic-provenance artifact is
    lawful only in a ``test`` namespace — at load as well as at persist."""

    root = Path(root)
    namespace = require_owner_decision_namespace(root)
    envelope = load_verified_envelope(
        root, OWNER_DECISION_STORE, artifact_id, OwnerDecisionArtifactEnvelope
    )
    if envelope.payload.store_namespace_id != namespace.store_namespace_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact belongs to another store namespace "
            f"({envelope.payload.store_namespace_id[:12]}… ≠ "
            f"{namespace.store_namespace_id[:12]}…); authority never follows a copy"
        )
    _assert_provenance_lawful_for_root(root, envelope.payload, namespace)
    return envelope


def load_supersession_chain(root: Path) -> tuple[SupersessionRecord, ...]:
    """Every recorded supersession — the immutable record chain verified
    from the head back to the namespace's genesis anchor, each record backed
    by a VERIFIED replacement artifact whose ``supersedes`` names the
    superseded id — fail closed on any inconsistency."""

    root = Path(root)
    try:
        records = load_supersession_records(root)
    except StoreNamespaceError as error:
        raise _chain_refusal(f"supersession chain unverifiable ({error})") from error
    for record in records:
        try:
            replacement = load_owner_decision(root, record.replacement_artifact_id)
        except SearchStoreError as error:
            raise _chain_refusal(
                "supersession record names a replacement that is not a verified store "
                f"entry ({record.replacement_artifact_id[:12]}…)"
            ) from error
        if replacement.payload.supersedes != record.superseded_artifact_id:
            raise _chain_refusal(
                "supersession record is not backed by its replacement artifact "
                f"(replacement {record.replacement_artifact_id[:12]}… does not name "
                f"{record.superseded_artifact_id[:12]}… as superseded)"
            )
    return records


def persist_owner_decision(
    root: Path,
    envelope: OwnerDecisionArtifactEnvelope,
    *,
    recorded_at: str,
    supersession_reason: str = "owner decision superseded by a later decision",
) -> OwnerDecisionArtifactEnvelope:
    """Persist (save-or-reuse). The artifact must name THIS store's verified
    namespace; synthetic provenance is confined to ``test`` namespaces. A
    supersession requires the prior to load verified, to name the same
    protocol, and to carry no stronger provenance than the replacement; the
    immutable supersession record is published (under the liveness-aware
    lock; idempotent on an identical replay, refused on a divergent one)
    BEFORE the replacement is published, so a crash in between leaves the
    chain fail-closed until this call is repeated."""

    root = Path(root)
    payload = envelope.payload
    namespace = require_owner_decision_namespace(root)
    if payload.store_namespace_id != namespace.store_namespace_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact names another store namespace "
            f"({payload.store_namespace_id[:12]}… ≠ {namespace.store_namespace_id[:12]}…)"
        )
    _assert_provenance_lawful_for_root(root, payload, namespace)
    if payload.supersedes is not None:
        try:
            prior = load_owner_decision(root, payload.supersedes)
        except SearchStoreError as error:
            raise OwnerDecisionRefusalError(
                "supersession refused: the prior decision is not a verified store entry"
            ) from error
        if prior.payload.resolved_regime_protocol_id != payload.resolved_regime_protocol_id:
            raise OwnerDecisionRefusalError(
                "supersession refused: the prior decision names a different regime protocol"
            )
        if _PROVENANCE_RANK[payload.provenance] < _PROVENANCE_RANK[prior.payload.provenance]:
            raise OwnerDecisionRefusalError(
                f"supersession refused: a {payload.provenance} decision cannot supersede a "
                f"{prior.payload.provenance} decision (weaker provenance)"
            )
        try:
            publish_supersession(
                root,
                superseded_decision_id=payload.supersedes,
                replacement_decision_id=envelope.owner_decision_artifact_id,
                reason=supersession_reason,
                effective_at=recorded_at,
                owner_evidence_ref=envelope.owner_decision_artifact_id,
            )
        except StoreNamespaceError as error:
            raise _chain_refusal(f"supersession refused ({error})") from error
    stored, _reused = save_or_reuse_envelope(root, OWNER_DECISION_STORE, envelope)
    return stored


def _values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(actual, Mapping)):
            return False
        return json.dumps(dict(expected), sort_keys=True, default=str) == json.dumps(
            dict(actual), sort_keys=True, default=str
        )
    return expected == actual


def assert_owner_decision_authorizes(
    artifact: OwnerDecisionArtifactEnvelope,
    *,
    protocol_envelope: RegimeProtocolEnvelope,
    assessment_envelope: RegimeCapabilityAssessmentEnvelope,
    transition: str,
    as_of: str,
    run_scope: RunScope,
    supersession_chain: tuple[SupersessionRecord, ...] = (),
) -> None:
    """Exact refusals, in order: provenance scope; superseded; effectivity;
    protocol; assessment reviewed; decision values 25/28/29/30; transition;
    passing assessment."""

    payload = artifact.payload
    if payload.provenance == "synthetic_test_authorization_v1" and run_scope != "synthetic_fixture":
        raise OwnerDecisionRefusalError(
            "owner decision refused: synthetic_test_authorization_v1 provenance is lawful "
            "only in the synthetic_fixture run scope"
        )
    superseded = {record.superseded_artifact_id for record in supersession_chain}
    if artifact.owner_decision_artifact_id in superseded:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact is superseded by a later decision"
        )
    instant = _parse_instant(as_of, field="as_of")
    if _parse_instant(payload.effective_from, field="effective_from") > instant:
        raise OwnerDecisionRefusalError("owner decision refused: the decision is not yet effective")
    if payload.effective_to is not None and (
        _parse_instant(payload.effective_to, field="effective_to") <= instant
    ):
        raise OwnerDecisionRefusalError("owner decision refused: the decision has expired")
    if payload.resolved_regime_protocol_id != protocol_envelope.resolved_regime_protocol_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact ratifies a different regime protocol"
        )
    if payload.capability_assessment_id != assessment_envelope.regime_capability_assessment_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact reviewed a different capability assessment"
        )
    if assessment_envelope.payload.resolved_regime_protocol_id != (
        protocol_envelope.resolved_regime_protocol_id
    ):
        raise OwnerDecisionRefusalError(
            "owner decision refused: the assessment does not belong to the protocol"
        )
    expected = expected_decision_values(protocol_envelope, assessment_envelope)
    mismatches = sorted(
        key
        for key, value in expected.items()
        if not _values_equal(value, payload.decision_values.get(key))
    )
    if mismatches:
        raise OwnerDecisionRefusalError(
            "owner decision refused: decision values disagree with the protocol / "
            f"registry / assessment on {mismatches}"
        )
    if transition not in payload.authorized_transitions:
        raise OwnerDecisionRefusalError(
            f"owner decision refused: transition {transition!r} is not authorized by the artifact"
        )
    if not assessment_envelope.payload.gates_passed:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the reviewed capability assessment did not pass its gates"
        )


def transition_key(previous: RegimeStatus, proposed: RegimeStatus) -> str:
    return f"{RegimeStatus(previous).value}->{RegimeStatus(proposed).value}"


# ── proposals (drafts that cannot persist) ───────────────────────────────────


def build_owner_decision_proposal(
    protocol: RegimeProtocolEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    *,
    transition: str = AUTHORIZED_TRANSITIONS[0],
    store_namespace_id: str | None = None,
) -> dict[str, Any]:
    """A draft payload with owner placeholders — it FAILS validation. The
    ``store_namespace_id`` is the target store's verified namespace id when
    known (else a placeholder the owner must replace)."""

    if transition not in AUTHORIZED_TRANSITIONS:
        raise ValueError(f"unregistered transition {transition!r}")
    values = plain_decision_values(expected_decision_values(protocol, assessment))
    return {
        "decision_id": REGIME_DECISION_ID,
        "store_namespace_id": store_namespace_id or _PLACEHOLDER,
        "decision_keys": list(REGIME_DECISION_KEYS),
        "decision_values": values,
        "resolved_regime_protocol_id": protocol.resolved_regime_protocol_id,
        "capability_assessment_id": assessment.regime_capability_assessment_id,
        "authorized_transitions": [transition],
        "provenance": "owner_signed",
        "author": _PLACEHOLDER,
        "approved_at": _PLACEHOLDER,
        "effective_from": _PLACEHOLDER,
        "effective_to": None,
        "reviewed_evidence_refs": [assessment.regime_capability_assessment_id],
        "supersedes": None,
        "rationale": _PLACEHOLDER,
        "_proposal_note": (
            "PROPOSAL — nothing here is an authorization; placeholders fail "
            "validation until the owner completes the ratification workflow"
        ),
    }


def render_owner_decision_proposal_markdown(proposal: dict[str, Any], *, title: str) -> str:
    values = proposal["decision_values"]
    lines = [
        f"# {title}",
        "",
        "**PROPOSAL — nothing here is an authorization.** The owner's ratification "
        "workflow replaces every placeholder; a payload carrying a placeholder cannot "
        "be persisted as an owner decision.",
        "",
        f"- decision_id: `{proposal['decision_id']}`",
        f"- store_namespace_id: `{proposal['store_namespace_id']}` (the target store's "
        "verified semantic namespace; never a path)",
        f"- resolved_regime_protocol_id: `{proposal['resolved_regime_protocol_id']}`",
        f"- capability_assessment_id: `{proposal['capability_assessment_id']}`",
        f"- authorized_transitions: {proposal['authorized_transitions']}",
        "",
        "| Decision | Key | Proposed value (proposed_protocol_default) |",
        "|---|---|---|",
        f"| 25 | algorithm_key | `{values['algorithm_key']}` |",
        f"| 25 | algorithm_parameters_hash | `{values['algorithm_parameters_hash']}` |",
        "| 25 | algorithm_parameters | "
        f"`{json.dumps(dict(values['algorithm_parameters']), sort_keys=True)}` |",
        f"| 28 | observation_granularity | `{values['observation_granularity']}` |",
        f"| 28 | panel_interval_seconds | `{values['panel_interval_seconds']}` |",
        f"| 28 | observation_stage | `{values['observation_stage']}` |",
        f"| 29 | fixed_cluster_count | `{values['fixed_cluster_count']}` |",
        "| 30 | minimum_cluster_occupancy_fraction | "
        f"`{values['minimum_cluster_occupancy_fraction']}` |",
        f"| 30 | minimum_cluster_rows_per_fold | `{values['minimum_cluster_rows_per_fold']}` |",
        "| 30 | minimum_bootstrap_aligned_ami_mean | "
        f"`{values['minimum_bootstrap_aligned_ami_mean']}` |",
        f"| 30 | minimum_training_observations | `{values['minimum_training_observations']}` |",
        "",
        "Owner fields to complete: `author`, `approved_at` (ISO-8601 with offset), "
        "`effective_from`, optional `effective_to`, `rationale`; provenance stays "
        "`owner_signed`.",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in proposal.items() if not k.startswith("_")},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def synthetic_owner_decision_fixture(
    root: Path,
    *,
    protocol: RegimeProtocolEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    transitions: tuple[str, ...] = AUTHORIZED_TRANSITIONS,
    approved_at: str = "2026-08-28T00:00:00+00:00",
    effective_from: str = "2026-08-28T00:00:00+00:00",
    effective_to: str | None = None,
    supersedes: str | None = None,
    value_overrides: dict[str, Any] | None = None,
    persist: bool = True,
) -> OwnerDecisionArtifactEnvelope:
    """A SYNTHETIC-provenance decision (lawful in the synthetic run scope
    only) over the exact protocol + assessment. An UNMARKED test root is
    explicitly initialized as a ``test`` namespace here (the class is stated
    by this fixture, never inferred from the path); a marked ``research``
    root refuses."""

    root = Path(root)
    try:
        if namespace_class_of(root) is None:
            # adversarial RA-06: production code never marks a store implicitly —
            # the fixture initializes ONLY a root that lives under the process
            # temp directory (pytest's tmp roots); any other unmarked root
            # (e.g. the repository's real verification store) requires the
            # operator's explicit `ifvg_store_namespace.py init`
            if not _under_temp_directory(root):
                raise StoreNamespaceError(
                    "store_namespace_missing",
                    f"{root} is unmarked and outside the temp directory; the synthetic fixture "
                    "never initializes a namespace implicitly — run the explicit init",
                )
            initialize_test_namespace(root)
        namespace = require_store_namespace(root)
    except StoreNamespaceError as error:
        raise _namespace_refusal(error) from error
    values = expected_decision_values(protocol, assessment)
    values.update(value_overrides or {})
    payload = OwnerDecisionArtifactPayload(
        store_namespace_id=namespace.store_namespace_id,
        decision_keys=REGIME_DECISION_KEYS,
        decision_values=freeze_decision_values(values),
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        capability_assessment_id=assessment.regime_capability_assessment_id,
        authorized_transitions=tuple(transitions),
        provenance="synthetic_test_authorization_v1",
        author="synthetic_fixture",
        approved_at=approved_at,
        effective_from=effective_from,
        effective_to=effective_to,
        reviewed_evidence_refs=(assessment.regime_capability_assessment_id,),
        supersedes=supersedes,
        rationale="synthetic test authorization over the exact assessment",
    )
    envelope = OwnerDecisionArtifactEnvelope.from_payload(payload)
    if persist:
        return persist_owner_decision(root, envelope, recorded_at=approved_at)
    return envelope


def _under_temp_directory(root: Path) -> bool:
    import tempfile  # noqa: PLC0415

    try:
        resolved = Path(root).resolve()
        temp = Path(tempfile.gettempdir()).resolve()
    except OSError:
        return False
    return temp == resolved or temp in resolved.parents


def _example_payload() -> OwnerDecisionArtifactPayload:
    entry = REGIME_ALGORITHM_REGISTRY["kmeans_v1"]
    return OwnerDecisionArtifactPayload(
        store_namespace_id="c" * 64,
        decision_keys=REGIME_DECISION_KEYS,
        decision_values={
            "algorithm_key": "kmeans_v1",
            "algorithm_parameters_hash": pinned_parameters_hash(entry),
            "algorithm_parameters": ImmutableMap(dict(entry.pinned_parameters)),
            "observation_granularity": "candidate_stage_row",
            "panel_interval_seconds": None,
            "observation_stage": "entry_decision",
            "fixed_cluster_count": 3,
            "minimum_cluster_occupancy_fraction": 0.05,
            "minimum_cluster_rows_per_fold": 25,
            "minimum_bootstrap_aligned_ami_mean": float(
                REGIME_PROPOSED_DEFAULTS["minimum_bootstrap_aligned_ami_mean"]["value"]
            ),
            "minimum_training_observations": sample_adequacy_minimum(
                ObservationGranularity.CANDIDATE_STAGE_ROW
            ),
        },
        resolved_regime_protocol_id="a" * 64,
        capability_assessment_id="b" * 64,
        authorized_transitions=AUTHORIZED_TRANSITIONS,
        provenance="synthetic_test_authorization_v1",
        author="example",
        approved_at="2026-08-28T00:00:00+00:00",
        effective_from="2026-08-28T00:00:00+00:00",
        effective_to=None,
        reviewed_evidence_refs=("b" * 64,),
        supersedes=None,
        rationale="example",
    )


register_identity_pair(
    name="OwnerDecisionArtifact",
    envelope_cls=OwnerDecisionArtifactEnvelope,
    payload_cls=OwnerDecisionArtifactPayload,
    id_field="owner_decision_artifact_id",
    example_factory=_example_payload,
)

_ = (canonical_contract_sha256, envelope_destination)  # re-exported helpers for the CLI
