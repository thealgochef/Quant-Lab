"""Per-trade risk sizing policies (CS §5.4) — every family, typed skip reasons.

``size_position`` converts one trade's per-contract risk into a contract
count under the account's live state; a trade that cannot be sized lawfully
is SKIPPED with an exact reason (never silently forced to one contract).
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)

__all__ = [
    "RiskPolicyFamily",
    "PostLossRule",
    "PropRiskPolicyPayload",
    "PropRiskPolicyEnvelope",
    "SizingDecision",
    "size_position",
    "FIXED_ONE_NQ_RISK_POLICY",
]


class RiskPolicyFamily(StrEnum):
    FIXED_DOLLAR = "fixed_dollar"
    PCT_START_BUFFER = "pct_start_buffer"
    PCT_CURRENT_BUFFER = "pct_current_buffer"
    FIXED_NQ = "fixed_nq"
    FIXED_MNQ = "fixed_mnq"
    NQ_MNQ_ADAPTIVE = "nq_mnq_adaptive"
    CUSTOM_CONTRACTS = "custom_contracts"


class PostLossRule(FrozenContract):
    consecutive_losses: int = Field(ge=1)
    risk_multiplier: float = Field(gt=0.0, le=1.0)
    recovery_wins: int = Field(ge=1)


class PropRiskPolicyPayload(FrozenContract):
    policy_version: str
    family: RiskPolicyFamily
    initial_risk_dollars: float | None
    risk_pct: float | None
    fixed_contracts: int | None
    instrument: Literal["NQ", "MNQ", "adaptive"] = "NQ"
    point_value_nq: float = 20.0
    point_value_mnq: float = 2.0
    max_contracts: int | None
    daily_stop_dollars: float | None
    daily_stop_r: float | None
    max_current_buffer_usage_pct: float | None
    min_remaining_buffer_dollars: float | None
    post_loss_adjustment: PostLossRule | None
    post_payout_derisk_factor: float | None
    skip_if_min_contract_exceeds_budget: bool = True


class PropRiskPolicyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "risk_policy_id"

    risk_policy_id: str = Field(pattern=SHA256_PATTERN)
    payload: PropRiskPolicyPayload


class SizingDecision(FrozenContract):
    contracts: int
    point_value: float
    instrument: str
    risk_budget_dollars: float | None
    skip_reason: str | None

    @property
    def skipped(self) -> bool:
        return self.contracts == 0


def _budget(
    policy: PropRiskPolicyPayload,
    *,
    start_buffer: float,
    current_buffer: float,
    consecutive_losses: int,
    recovery_wins_since: int,
    payouts_taken: int,
) -> float | None:
    if policy.family is RiskPolicyFamily.FIXED_DOLLAR:
        budget = policy.initial_risk_dollars
    elif policy.family is RiskPolicyFamily.PCT_START_BUFFER:
        budget = (policy.risk_pct or 0.0) / 100.0 * start_buffer
    elif policy.family is RiskPolicyFamily.PCT_CURRENT_BUFFER:
        budget = (policy.risk_pct or 0.0) / 100.0 * max(current_buffer, 0.0)
    else:
        budget = None  # contract-count families carry no dollar budget
    if budget is None:
        return None
    adjustment = policy.post_loss_adjustment
    if (
        adjustment is not None
        and consecutive_losses >= adjustment.consecutive_losses
        and recovery_wins_since < adjustment.recovery_wins
    ):
        budget *= adjustment.risk_multiplier
    if policy.post_payout_derisk_factor is not None and payouts_taken > 0:
        budget *= policy.post_payout_derisk_factor
    return budget


def size_position(
    policy: PropRiskPolicyPayload,
    *,
    risk_points: float,
    start_buffer: float,
    current_buffer: float,
    day_realized_pnl: float,
    consecutive_losses: int = 0,
    recovery_wins_since: int = 0,
    payouts_taken: int = 0,
    phase_max_contracts: int | None = None,
    micro_scaling_table: tuple[tuple[float, int], ...] = (),
) -> SizingDecision:
    """One trade's contract count under the live account state, fail-closed."""

    def _skip(reason: str, point_value: float, instrument: str) -> SizingDecision:
        return SizingDecision(
            contracts=0,
            point_value=point_value,
            instrument=instrument,
            risk_budget_dollars=None,
            skip_reason=reason,
        )

    if risk_points <= 0:
        return _skip("nonpositive_risk_points", policy.point_value_nq, "NQ")

    # daily stop gates (dollar and R forms) refuse further risk today
    if (
        policy.daily_stop_dollars is not None
        and day_realized_pnl <= -policy.daily_stop_dollars
    ):
        return _skip("daily_stop_dollars_reached", policy.point_value_nq, "NQ")
    budget = _budget(
        policy,
        start_buffer=start_buffer,
        current_buffer=current_buffer,
        consecutive_losses=consecutive_losses,
        recovery_wins_since=recovery_wins_since,
        payouts_taken=payouts_taken,
    )
    if (
        policy.daily_stop_r is not None
        and budget is not None
        and budget > 0
        and day_realized_pnl <= -policy.daily_stop_r * budget
    ):
        return _skip("daily_stop_r_reached", policy.point_value_nq, "NQ")

    if policy.family in (RiskPolicyFamily.FIXED_NQ, RiskPolicyFamily.CUSTOM_CONTRACTS):
        contracts, point_value, instrument = (
            policy.fixed_contracts or 0,
            policy.point_value_nq,
            "NQ",
        )
    elif policy.family is RiskPolicyFamily.FIXED_MNQ:
        contracts, point_value, instrument = (
            policy.fixed_contracts or 0,
            policy.point_value_mnq,
            "MNQ",
        )
    elif policy.family is RiskPolicyFamily.NQ_MNQ_ADAPTIVE:
        if budget is None:
            budget = policy.initial_risk_dollars or 0.0
        per_nq = risk_points * policy.point_value_nq
        if budget >= per_nq:
            contracts = math.floor(budget / per_nq)
            point_value, instrument = policy.point_value_nq, "NQ"
        else:
            per_mnq = risk_points * policy.point_value_mnq
            contracts = math.floor(budget / per_mnq) if per_mnq > 0 else 0
            point_value, instrument = policy.point_value_mnq, "MNQ"
            if contracts == 0:
                return _skip("budget_below_one_micro", point_value, instrument)
    else:  # dollar-budget families sized in NQ contracts
        point_value, instrument = policy.point_value_nq, "NQ"
        if budget is None or budget <= 0:
            return _skip("no_risk_budget", point_value, instrument)
        per_contract = risk_points * point_value
        contracts = math.floor(budget / per_contract) if per_contract > 0 else 0
        if contracts == 0:
            if policy.skip_if_min_contract_exceeds_budget:
                return _skip("min_contract_exceeds_budget", point_value, instrument)
            contracts = 1

    if contracts <= 0:
        return _skip("zero_contracts_configured", point_value, instrument)

    # buffer-usage guards
    per_contract_risk = risk_points * point_value
    if policy.max_current_buffer_usage_pct is not None:
        allowed = policy.max_current_buffer_usage_pct / 100.0 * max(current_buffer, 0.0)
        if per_contract_risk * contracts > allowed:
            contracts = math.floor(allowed / per_contract_risk) if per_contract_risk else 0
            if contracts == 0:
                return _skip("buffer_usage_cap", point_value, instrument)
    if policy.min_remaining_buffer_dollars is not None and (
        current_buffer - per_contract_risk * contracts
        < policy.min_remaining_buffer_dollars
    ):
        affordable = math.floor(
            (current_buffer - policy.min_remaining_buffer_dollars) / per_contract_risk
        ) if per_contract_risk else 0
        if affordable <= 0:
            return _skip("min_remaining_buffer", point_value, instrument)
        contracts = min(contracts, affordable)

    # firm caps: micro-scaling table (buffer → allowed contracts) then phase cap
    if micro_scaling_table:
        allowed = 0
        for threshold, table_contracts in sorted(micro_scaling_table):
            if current_buffer >= threshold:
                allowed = table_contracts
        if allowed == 0:
            return _skip("micro_scaling_zero_allowance", point_value, instrument)
        contracts = min(contracts, allowed)
    if policy.max_contracts is not None:
        contracts = min(contracts, policy.max_contracts)
    if phase_max_contracts is not None:
        contracts = min(contracts, phase_max_contracts)
    if contracts <= 0:
        return _skip("capped_to_zero", point_value, instrument)
    return SizingDecision(
        contracts=contracts,
        point_value=point_value,
        instrument=instrument,
        risk_budget_dollars=budget,
        skip_reason=None,
    )


#: The one-contract parity policy: AccountWalk under it must reproduce the
#: existing EvaluationWalk exactly (§16.4 parity row).
FIXED_ONE_NQ_RISK_POLICY = PropRiskPolicyPayload(
    policy_version="fixed_one_nq_v1",
    family=RiskPolicyFamily.FIXED_NQ,
    initial_risk_dollars=None,
    risk_pct=None,
    fixed_contracts=1,
    instrument="NQ",
    max_contracts=None,
    daily_stop_dollars=None,
    daily_stop_r=None,
    max_current_buffer_usage_pct=None,
    min_remaining_buffer_dollars=None,
    post_loss_adjustment=None,
    post_payout_derisk_factor=None,
)


register_identity_pair(
    name="PropRiskPolicy",
    envelope_cls=PropRiskPolicyEnvelope,
    payload_cls=PropRiskPolicyPayload,
    id_field="risk_policy_id",
    example_factory=lambda: FIXED_ONE_NQ_RISK_POLICY,
)
