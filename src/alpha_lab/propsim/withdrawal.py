"""Trader withdrawal behavior — separate from the firm contract (CS §5.2; P0-15).

The firm's :class:`~alpha_lab.propsim.firm_contracts.PayoutPolicy` states what
is PERMITTED; this policy states what the trader CHOOSES. Both identities
enter every account/portfolio simulation — the firm contract alone never
determines trader behavior.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from alpha_lab.propsim.calendar import DayCountBasis, DurationRule

__all__ = [
    "WithdrawalPolicyPayload",
    "WithdrawalPolicyEnvelope",
    "plan_withdrawal_request",
    "REQUEST_MAX_AT_ELIGIBILITY",
]


class WithdrawalPolicyPayload(FrozenContract):
    policy_key: str
    behavior: Literal[
        "request_at_first_eligibility_max",
        "retain_minimum_buffer",
        "fixed_cadence",
        "partial_fixed_amount",
        "max_allowed_each_period",
    ]
    minimum_buffer_retained: float | None
    cadence: DurationRule | None
    partial_amount: float | None
    post_payout_derisk_ref: str | None

    @model_validator(mode="after")
    def _behavior_fields_present(self):
        if self.behavior == "retain_minimum_buffer" and self.minimum_buffer_retained is None:
            raise ValueError("retain_minimum_buffer requires minimum_buffer_retained")
        if self.behavior == "fixed_cadence" and self.cadence is None:
            raise ValueError("fixed_cadence requires a cadence duration rule")
        if self.behavior == "partial_fixed_amount" and self.partial_amount is None:
            raise ValueError("partial_fixed_amount requires partial_amount")
        return self


class WithdrawalPolicyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "withdrawal_policy_id"

    withdrawal_policy_id: str = Field(pattern=SHA256_PATTERN)
    payload: WithdrawalPolicyPayload


def plan_withdrawal_request(
    policy: WithdrawalPolicyPayload,
    *,
    payout_available: float,
    current_buffer: float,
    elapsed_since_last_payout: int | None,
) -> float:
    """The trader's requested amount RIGHT NOW (0.0 = no request).

    The caller has already established firm-side ELIGIBILITY; this function
    encodes only the trader's chosen behavior. Amounts are pre-split (the
    firm's split applies downstream). ``elapsed_since_last_payout`` is
    expressed in the cadence rule's OWN day-count basis (the caller computes
    it through the typed simulated clock — P0-14: bases are never silently
    reinterpreted as trading days).
    """

    if payout_available <= 0:
        return 0.0
    if policy.behavior in ("request_at_first_eligibility_max", "max_allowed_each_period"):
        return payout_available
    if policy.behavior == "retain_minimum_buffer":
        retained = policy.minimum_buffer_retained or 0.0
        return max(0.0, min(payout_available, current_buffer - retained))
    if policy.behavior == "fixed_cadence":
        cadence = policy.cadence
        if cadence is None:  # pragma: no cover - validator guarantees
            return 0.0
        if (
            elapsed_since_last_payout is None  # never paid → cadence starts now
            or elapsed_since_last_payout >= cadence.count
        ):
            return payout_available
        return 0.0
    if policy.behavior == "partial_fixed_amount":
        return min(payout_available, policy.partial_amount or 0.0)
    raise ValueError(f"unknown withdrawal behavior {policy.behavior!r}")  # pragma: no cover


REQUEST_MAX_AT_ELIGIBILITY = WithdrawalPolicyPayload(
    policy_key="request_at_first_eligibility_max_v1",
    behavior="request_at_first_eligibility_max",
    minimum_buffer_retained=None,
    cadence=DurationRule(count=0, basis=DayCountBasis.TRADING_DAY),
    partial_amount=None,
    post_payout_derisk_ref=None,
)


register_identity_pair(
    name="WithdrawalPolicy",
    envelope_cls=WithdrawalPolicyEnvelope,
    payload_cls=WithdrawalPolicyPayload,
    id_field="withdrawal_policy_id",
    example_factory=lambda: REQUEST_MAX_AT_ELIGIBILITY,
)
