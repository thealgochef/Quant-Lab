"""Explicit, immutable owner approval of one exact strategy-search request.

Approval is local owner evidence, not cryptographic authentication or proof of
successful verification. It grants no prop/model/regime authority. The shared
axis registry remains pending; only a matching charter gets a ratified view.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from .authorization import OwnerDecisionEvidenceRef, assert_authorization_bound_to_store
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from .store import load_verified_envelope, save_or_reuse_envelope
from .store_namespace import require_store_namespace

STORE = "strategy_search_approvals"
DECISION_ID = "strategy_search_approval_v1"
DECISION_KEYS = (
    "1:first_search_axes",
    "2:axis_values",
    "7:strategy_gate_thresholds",
    "R-2:authorization_workflow",
)


def charter_intent(payload) -> dict:
    values = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else dict(payload)
    values.pop("owner_authorization", None)
    # ImmutableMap's wire representation is a tuple of pairs; normalize the
    # pure wizard mapping and a validated charter to the same semantic shape.
    if "axes" in values:
        values["axes"] = dict(values["axes"])
    # Normalize nested contracts as well as already serialized content.
    from pydantic import TypeAdapter  # noqa: PLC0415

    return TypeAdapter(dict).dump_python(values, mode="json")


def charter_intent_hash(payload) -> str:
    return canonical_contract_sha256(charter_intent(payload))


class StrategySearchApprovalPayload(FrozenContract):
    decision_id: Literal["strategy_search_approval_v1"] = DECISION_ID
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    requirement_set_id: str = Field(pattern=SHA256_PATTERN)
    charter_intent_sha256: str = Field(pattern=SHA256_PATTERN)
    approved_charter_json: str
    artifact_provenance_dates: tuple[str, ...]
    author: str = Field(min_length=1)
    approved_at: str
    effective_from: str
    reviewed_evidence_refs: tuple[str, ...] = Field(min_length=1)
    approval_statement: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_scope(self):
        content = json.loads(self.approved_charter_json)
        from ..development_access import PERMITTED_DEVELOPMENT_DATES  # noqa: PLC0415

        if (
            tuple(sorted(set(self.artifact_provenance_dates))) != self.artifact_provenance_dates
            or not set(self.artifact_provenance_dates) <= set(PERMITTED_DEVELOPMENT_DATES)
            or not set(content.get("date_policy", {}).get("replay_dates", ()))
            <= set(self.artifact_provenance_dates)
        ):
            raise ValueError("artifact provenance dates must cover the permitted replay dates")
        if "owner_authorization" in content:
            raise ValueError("approval content must exclude its own authorization")
        if charter_intent_hash(content) != self.charter_intent_sha256:
            raise ValueError("approved charter content does not match its hash")
        if content.get("search_mode") != "fsm_config_search" or not content.get("axes"):
            raise ValueError("this approval supports strategy configuration searches only")
        for key in (
            "authorized_firm_contract_ids",
            "authorized_risk_policy_ids",
            "authorized_withdrawal_policy_ids",
            "source_artifact_ids",
        ):
            if content.get(key):
                raise ValueError("strategy approval grants no prop/model/source-artifact authority")
        if content.get("date_policy", {}).get("access_policy_id") != (
            "development_explicit_dates_before_path_v2"
        ):
            raise ValueError("strategy approval requires the development date policy")
        approved = datetime.fromisoformat(self.approved_at.replace("Z", "+00:00"))
        effective = datetime.fromisoformat(self.effective_from.replace("Z", "+00:00"))
        if approved.tzinfo is None or effective.tzinfo is None or effective < approved:
            raise ValueError("approval timestamps must be aware and effective after approval")
        return self


class StrategySearchApprovalEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "strategy_search_approval_id"
    strategy_search_approval_id: str = Field(pattern=SHA256_PATTERN)
    payload: StrategySearchApprovalPayload

    def evidence_ref(self) -> OwnerDecisionEvidenceRef:
        return OwnerDecisionEvidenceRef(
            decision_id=DECISION_ID,
            decision_artifact_id=self.strategy_search_approval_id,
            content_hash=self.strategy_search_approval_id,
            author=self.payload.author,
            approved_at=self.payload.approved_at,
            effective_from=self.payload.effective_from,
            reviewed_evidence_refs=self.payload.reviewed_evidence_refs,
        )


def load_strategy_approval(root: Path, artifact_id: str, *, as_of_utc: str | None = None):
    namespace = require_store_namespace(root, expected_class="research")
    envelope = load_verified_envelope(root, STORE, artifact_id, StrategySearchApprovalEnvelope)
    if envelope.payload.store_namespace_id != namespace.store_namespace_id:
        raise PermissionError("strategy approval belongs to another namespace")
    now = (
        datetime.fromisoformat(as_of_utc.replace("Z", "+00:00")) if as_of_utc else datetime.now(UTC)
    )
    if datetime.fromisoformat(envelope.payload.effective_from.replace("Z", "+00:00")) > now:
        raise PermissionError("strategy approval is not yet effective")
    return envelope


def persist_strategy_approval(root: Path, envelope: StrategySearchApprovalEnvelope):
    namespace = require_store_namespace(root, expected_class="research")
    if envelope.payload.store_namespace_id != namespace.store_namespace_id:
        raise PermissionError("strategy approval belongs to another namespace")
    # Prove the existing authority chain before publishing, including genesis.
    from .owner_decisions import verify_complete_owner_authority_chain  # noqa: PLC0415

    verify_complete_owner_authority_chain(root)
    return save_or_reuse_envelope(root, STORE, envelope)


def ratified_registry_for_charter(payload, root, registry, *, as_of_utc=None):
    """Load exact approval refs and match ALL request fields before ratification."""
    bundle = payload.owner_authorization
    refs = getattr(bundle, "decision_refs", {})
    relevant = [ref for ref in refs.values() if ref.decision_id == DECISION_ID]
    if not relevant:
        return registry
    if root is None:
        raise PermissionError("strategy approval requires its verified store")
    assert_authorization_bound_to_store(
        Path(root),
        store_namespace_id=bundle.store_namespace_id,
        supersession_head_witness=bundle.supersession_head_witness,
        expected_namespace_class="research",
    )
    approval = load_strategy_approval(
        Path(root),
        relevant[0].decision_artifact_id,
        as_of_utc=as_of_utc,
    )
    if any(refs.get(key) != approval.evidence_ref() for key in DECISION_KEYS):
        raise PermissionError("strategy approval must cover every required decision exactly")
    if bundle.requirement_set_id != approval.payload.requirement_set_id:
        raise PermissionError("strategy approval names a different requirement set")
    if charter_intent_hash(payload) != approval.payload.charter_intent_sha256:
        raise PermissionError("study settings changed after approval")
    from .axis_registry import registry_sha256  # noqa: PLC0415

    if payload.locked_invariants_registry_sha256 != registry_sha256(values=registry):
        raise PermissionError("axis registry changed after approval")
    updated = dict(registry)
    for values in payload.axes.values():
        for value in values:
            if value not in updated:
                raise PermissionError("approved value is no longer registered")
            updated[value] = updated[value].model_copy(
                update={
                    "owner_ratification_status": "ratified",
                    "ratification_evidence_ref": approval.strategy_search_approval_id,
                }
            )
    return updated


def _example_approval_payload():
    from ..development_access import FROZEN_WARMUP_DATES  # noqa: PLC0415
    from .charter import DatePolicy, _example_charter_payload  # noqa: PLC0415

    example = _example_charter_payload()
    fields = charter_intent(
        example.model_copy(
            update={
                "date_policy": DatePolicy(
                    replay_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
                    warmup_dates=FROZEN_WARMUP_DATES,
                    access_policy_id="development_explicit_dates_before_path_v2",
                ),
            }
        )
    )
    return StrategySearchApprovalPayload(
        store_namespace_id="a" * 64,
        requirement_set_id="b" * 64,
        charter_intent_sha256=charter_intent_hash(fields),
        approved_charter_json=json.dumps(fields, sort_keys=True),
        artifact_provenance_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
        author="schema example",
        approved_at="2026-09-08T00:00:00+00:00",
        effective_from="2026-09-08T00:00:00+00:00",
        reviewed_evidence_refs=("schema-example-only",),
        approval_statement="Schema example only.",
    )


register_identity_pair(
    name="StrategySearchApproval",
    envelope_cls=StrategySearchApprovalEnvelope,
    payload_cls=StrategySearchApprovalPayload,
    id_field="strategy_search_approval_id",
    example_factory=_example_approval_payload,
)
