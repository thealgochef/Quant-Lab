"""Decision-policy contracts and registry (`ML_REGIME_CONTRACT_PLAN.md` §7;
`CONTRACTS_AND_SCHEMAS.md` §9; TEST_MATRIX §3.10 "Decision-policy and
schedule envelopes").

Logical registry keys are distinct from resolved identities: a
``decision_policy_key`` names the option; a ``resolved_decision_policy_id``
hashes one exact :class:`DecisionPolicyPayload`. In V1 the ONLY executable
policy is ``none_diagnostic_only_v1`` — every execution-affecting policy is
registered planned, requires an owner-ratified
:class:`RejectedCandidatePolicy` (owner decision R-1), and pipeline stage
S11 stays BLOCKED with the exact registered reason until that ratification
plus the sequential golden tests exist.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

from pydantic import Field, model_validator

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from ..study.study_cell import REGISTERED_DECISION_POLICY_KEYS

__all__ = [
    "S11_BLOCKED_REASON",
    "RejectedCandidatePolicy",
    "DecisionPolicyStatus",
    "DecisionPolicyRegistryEntry",
    "DECISION_POLICY_REGISTRY",
    "DecisionPolicyUnavailableError",
    "DecisionPolicyPayload",
    "DecisionPolicyEnvelope",
    "WalkForwardModelScheduleEntry",
    "WalkForwardModelSchedulePayload",
    "WalkForwardModelScheduleEnvelope",
    "ModelGatedReplayRequest",
    "resolve_decision_policy_entry",
    "resolve_baseline_decision_policy",
]

#: The exact S11 blocked reason (matches the computation-path requirement in
#: ``search/authorization.py`` and the dimension registry verbatim).
S11_BLOCKED_REASON = (
    "model-gated execution is unavailable until RejectedCandidatePolicy "
    "is owner-ratified and sequential golden tests pass"
)


class RejectedCandidatePolicy(StrEnum):
    REJECT_KEEP_SETUP_WAITING = "reject_keep_setup_waiting"
    REJECT_TERMINATE_SETUP_MISSED = "reject_terminate_setup_missed"
    REJECT_CONSUME_ONE_SHOT_TRIGGER = "reject_consume_one_shot_trigger"
    REJECT_RESET_SETUP = "reject_reset_setup"


class DecisionPolicyStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"


class DecisionPolicyRegistryEntry(FrozenContract):
    decision_policy_key: str
    status: DecisionPolicyStatus
    execution_affecting: bool
    reason: str | None = None


DECISION_POLICY_REGISTRY: MappingProxyType[str, DecisionPolicyRegistryEntry] = (
    MappingProxyType(
        {
            "none_diagnostic_only_v1": DecisionPolicyRegistryEntry(
                decision_policy_key="none_diagnostic_only_v1",
                status=DecisionPolicyStatus.AVAILABLE,
                execution_affecting=False,
            ),
            **{
                key: DecisionPolicyRegistryEntry(
                    decision_policy_key=key,
                    status=DecisionPolicyStatus.PLANNED,
                    execution_affecting=True,
                    reason=S11_BLOCKED_REASON,
                )
                for key in (
                    "fixed_probability_threshold_v1",
                    "expected_r_threshold_v1",
                    "abstention_band_v1",
                    "top_n_per_day_v1",
                    "confidence_margin_v1",
                    "regime_conditioned_threshold_v1",
                )
            },
        }
    )
)

if tuple(DECISION_POLICY_REGISTRY) != REGISTERED_DECISION_POLICY_KEYS:
    raise AssertionError(
        "the decision-policy registry must cover exactly the study-cell "
        "registered decision policy keys, in order"
    )


class DecisionPolicyUnavailableError(PermissionError):
    """A planned decision policy was requested for execution (S11 blocked)."""


class DecisionPolicyPayload(FrozenContract):
    """Hashed decision-policy content (non-self-referential — §0.1)."""

    decision_policy_key: str
    parameters: ImmutableMap[str, Any]
    resolved_regime_protocol_id: str | None
    rejected_candidate_policy: RejectedCandidatePolicy | None
    rejected_candidate_policy_ratification_ref: str | None
    requires_frozen_model: bool
    requires_model_gated_sequential_replay: bool
    produces_new_trade_stream_hash: bool

    @model_validator(mode="after")
    def _flags_agree_with_key(self):
        if self.decision_policy_key not in DECISION_POLICY_REGISTRY:
            raise ValueError(
                f"decision policy key {self.decision_policy_key!r} is not registered"
            )
        entry = DECISION_POLICY_REGISTRY[self.decision_policy_key]
        flags = (
            self.requires_frozen_model,
            self.requires_model_gated_sequential_replay,
            self.produces_new_trade_stream_hash,
        )
        if not entry.execution_affecting:
            if self.rejected_candidate_policy is not None:
                raise ValueError(
                    "none_diagnostic_only_v1 carries no rejected-candidate policy"
                )
            if self.rejected_candidate_policy_ratification_ref is not None:
                raise ValueError(
                    "none_diagnostic_only_v1 carries no ratification reference"
                )
            if any(flags):
                raise ValueError(
                    "a diagnostic-only policy cannot require a frozen model, a "
                    "model-gated sequential replay, or a new trade stream"
                )
        else:
            if self.rejected_candidate_policy is None:
                raise ValueError(
                    "every execution-affecting decision policy requires a "
                    "rejected-candidate policy (owner decision R-1)"
                )
            if not self.rejected_candidate_policy_ratification_ref:
                raise ValueError(
                    "every execution-affecting decision policy requires the "
                    "owner's rejected-candidate-policy ratification reference"
                )
            if not all(flags):
                raise ValueError(
                    "every execution-affecting decision policy requires a frozen "
                    "model, a model-gated sequential replay, and a new trade "
                    "stream hash"
                )
        if (
            self.decision_policy_key == "regime_conditioned_threshold_v1"
            and self.resolved_regime_protocol_id is None
        ):
            raise ValueError(
                "regime_conditioned_threshold_v1 requires a resolved regime "
                "protocol id"
            )
        return self


class DecisionPolicyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "resolved_decision_policy_id"

    resolved_decision_policy_id: str = Field(pattern=SHA256_PATTERN)
    payload: DecisionPolicyPayload


class WalkForwardModelScheduleEntry(FrozenContract):
    date_from: str
    date_to: str
    frozen_model_fit_id: str

    @model_validator(mode="after")
    def _ordered(self):
        if self.date_to < self.date_from:
            raise ValueError("schedule entry dates are reversed")
        return self


class WalkForwardModelSchedulePayload(FrozenContract):
    entries: tuple[WalkForwardModelScheduleEntry, ...]
    coverage_policy_id: str

    @model_validator(mode="after")
    def _entries_chronological(self):
        starts = [entry.date_from for entry in self.entries]
        if starts != sorted(starts):
            raise ValueError("schedule entries must be chronological")
        return self


class WalkForwardModelScheduleEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "resolved_model_schedule_id"

    resolved_model_schedule_id: str = Field(pattern=SHA256_PATTERN)
    payload: WalkForwardModelSchedulePayload


class ModelGatedReplayRequest(FrozenContract):
    """S11's ONLY consumable input — policies emit requests, never launches.

    A request is constructible (the option space is real) but no executor
    exists in V1: S11 stays blocked with :data:`S11_BLOCKED_REASON`.
    """

    cell_id: str = Field(pattern=SHA256_PATTERN)
    frozen_model_fit_id: str
    resolved_decision_policy_id: str = Field(pattern=SHA256_PATTERN)
    resolved_model_schedule_id: str = Field(pattern=SHA256_PATTERN)


def resolve_decision_policy_entry(decision_policy_key: str) -> DecisionPolicyRegistryEntry:
    entry = DECISION_POLICY_REGISTRY.get(decision_policy_key)
    if entry is None:
        raise ValueError(
            f"unknown decision policy key {decision_policy_key!r}; registered: "
            f"{sorted(DECISION_POLICY_REGISTRY)}"
        )
    return entry


def resolve_baseline_decision_policy() -> DecisionPolicyEnvelope:
    """The REAL resolved id for the diagnostic-only baseline (DEV-R1-5 closure).

    R1–R4 accepted the ``none_decision_policy_v1`` sentinel because no
    resolver existed; from R5 the pipeline pins this envelope's 64-hex id.
    The sentinel remains lawful for strategy-only study cells.
    """

    payload = DecisionPolicyPayload(
        decision_policy_key="none_diagnostic_only_v1",
        parameters={},
        resolved_regime_protocol_id=None,
        rejected_candidate_policy=None,
        rejected_candidate_policy_ratification_ref=None,
        requires_frozen_model=False,
        requires_model_gated_sequential_replay=False,
        produces_new_trade_stream_hash=False,
    )
    return DecisionPolicyEnvelope.from_payload(payload)


def _example_decision_policy() -> DecisionPolicyPayload:
    return resolve_baseline_decision_policy().payload


def _example_schedule() -> WalkForwardModelSchedulePayload:
    return WalkForwardModelSchedulePayload(
        entries=(
            WalkForwardModelScheduleEntry(
                date_from="2026-03-02",
                date_to="2026-03-06",
                frozen_model_fit_id="a" * 64,
            ),
        ),
        coverage_policy_id="uncovered_dates_refuse_v1",
    )


register_identity_pair(
    name="DecisionPolicy",
    envelope_cls=DecisionPolicyEnvelope,
    payload_cls=DecisionPolicyPayload,
    id_field="resolved_decision_policy_id",
    example_factory=_example_decision_policy,
)
register_identity_pair(
    name="WalkForwardModelSchedule",
    envelope_cls=WalkForwardModelScheduleEnvelope,
    payload_cls=WalkForwardModelSchedulePayload,
    id_field="resolved_model_schedule_id",
    example_factory=_example_schedule,
)
