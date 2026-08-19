"""Firm rules, payout/fees, and the rule→capability matrix (CS §5.2).

The firm contract states what is PERMITTED (rules, thresholds, observation
policies); trader choices (withdrawal behavior) live in the separate
withdrawal policy; adverse-first/favorable-first ordering is NEVER embedded
here — it belongs only to an explicit scenario policy. Every rule declares
its required path capabilities and accepted fidelity classes and fails
closed when the evidence cannot support it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from alpha_lab.propsim.calendar import DayCountBasis, DurationRule
from alpha_lab.propsim.trade_path import (
    PathCapability,
    PropRulePathRequirement,
    TradePathFidelity,
)

__all__ = [
    "AccountPhase",
    "PhaseRules",
    "PayoutPolicy",
    "FeeSchedule",
    "PropFirmContractPayload",
    "PropFirmContractEnvelope",
    "default_rule_path_requirements",
    "SYNTHETIC_FIXTURE_FIRM",
]


class AccountPhase(StrEnum):
    EVALUATION = "evaluation"
    FUNDED = "funded"
    BREACHED = "breached"
    EXPIRED = "expired"
    RETIRED = "retired"


class PhaseRules(FrozenContract):
    starting_balance: float
    profit_target: float | None
    trail_amount: float
    trail_style: Literal[
        "eod_floor_realtime_breach", "intraday_peak_trail", "static_floor"
    ]
    trail_locks_at_start: bool
    dll_amount: float | None
    dll_hard: bool
    consistency_pct: float | None
    min_days: DurationRule | None
    max_eval_days: DurationRule | None
    unrealized_equity_counts_for_breach: bool
    breach_observation_policy: Literal[
        "continuous_ordered_path", "event_updates_only", "eod_only"
    ]
    max_contracts: int = Field(ge=1)
    micro_scaling_table: tuple[tuple[float, int], ...] = ()


class PayoutPolicy(FrozenContract):
    waiting_period: DurationRule
    min_between_payouts: DurationRule
    min_winning_days: DurationRule | None
    winning_day_min_pnl: float | None
    payout_cap_per_period: float | None
    split_pct_trader: float = Field(gt=0.0, le=100.0)
    min_payout: float | None
    max_payout: float | None
    withdrawal_reduces_threshold: bool
    post_payout_buffer_rule: Literal[
        "threshold_unchanged",
        "threshold_resets_to_balance_minus_trail",
        "locked_at_starting_balance",
    ]
    payout_processing: DurationRule | None


class FeeSchedule(FrozenContract):
    evaluation_fee: float = Field(ge=0.0)
    activation_fee: float = Field(ge=0.0)
    recurring_fee: float = Field(ge=0.0)
    recurring_period: DurationRule | None
    reset_fee: float = Field(ge=0.0)


class PropFirmContractPayload(FrozenContract):
    firm: str
    account_type: str
    account_size_label: str
    contract_version: str
    effective_date: str
    source_status: Literal[
        "official_document", "checkout_screen", "user_reported", "unverified"
    ]
    source_references: tuple[str, ...]
    verification_status: Literal[
        "synthetic_fixture_verified",
        "first_party_evidence_compiled",
        "owner_reviewed",
        "first_party_verified",
        "superseded",
    ]
    currency: Literal["USD"] = "USD"
    timezone: str = "America/New_York"
    trading_day_boundary: Literal["18:00_ET_roll"] = "18:00_ET_roll"
    evaluation: PhaseRules
    funded: PhaseRules | None
    payout: PayoutPolicy | None
    fees: FeeSchedule
    rule_path_requirements: tuple[PropRulePathRequirement, ...]
    breach_reasons_supported: tuple[str, ...]
    account_expiration: DurationRule | None

    @model_validator(mode="after")
    def _funded_needs_payout(self):
        if self.funded is not None and self.payout is None:
            raise ValueError("a funded phase requires a payout policy")
        return self


class PropFirmContractEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "firm_contract_id"

    firm_contract_id: str = Field(pattern=SHA256_PATTERN)
    payload: PropFirmContractPayload
    contract_evidence_bundle_id: str


_ALL_HISTORICAL = (
    TradePathFidelity.CLOSED_TRADE_ONLY,
    TradePathFidelity.OHLC_1M_UNORDERED,
    TradePathFidelity.ASSUMED_1M_INTRABAR_PATH,
    TradePathFidelity.ORDERED_MBP1_EVENT_PATH,
)


def default_rule_path_requirements(
    phase: PhaseRules, *, phase_id: str
) -> tuple[PropRulePathRequirement, ...]:
    """The rule→capability/accepted-class matrix for one phase's rules.

    Chronology-sensitive breach observation (`continuous_ordered_path` with
    unrealized equity counting) requires actual market-price chronology and
    accepts assumed 1m paths only as explicitly-permitted SCENARIOS;
    extrema-order-insensitive observation accepts unordered OHLC; EOD-only
    breach needs only the closed-trade result chronology.
    """

    requirements: list[PropRulePathRequirement] = []
    if phase.breach_observation_policy == "continuous_ordered_path":
        requirements.append(
            PropRulePathRequirement(
                rule_id=f"{phase_id}.breach_observation_continuous",
                required_path_capabilities=(
                    PathCapability.MARKET_PRICE_CHRONOLOGY,
                    PathCapability.UNREALIZED_MARK_TO_MARKET,
                ),
                accepted_fidelity_classes=(
                    TradePathFidelity.ASSUMED_1M_INTRABAR_PATH,
                    TradePathFidelity.ORDERED_MBP1_EVENT_PATH,
                ),
                scenario_use_permitted=True,
            )
        )
    elif phase.breach_observation_policy == "event_updates_only":
        requirements.append(
            PropRulePathRequirement(
                rule_id=f"{phase_id}.breach_observation_event_updates",
                required_path_capabilities=(
                    (
                        PathCapability.UNREALIZED_MARK_TO_MARKET,
                        PathCapability.REALIZED_PNL_CHRONOLOGY,
                    )
                    if phase.unrealized_equity_counts_for_breach
                    else (PathCapability.REALIZED_PNL_CHRONOLOGY,)
                ),
                accepted_fidelity_classes=(
                    (
                        TradePathFidelity.OHLC_1M_UNORDERED,
                        TradePathFidelity.ASSUMED_1M_INTRABAR_PATH,
                        TradePathFidelity.ORDERED_MBP1_EVENT_PATH,
                    )
                    if phase.unrealized_equity_counts_for_breach
                    else _ALL_HISTORICAL
                ),
                scenario_use_permitted=True,
            )
        )
    else:  # eod_only
        requirements.append(
            PropRulePathRequirement(
                rule_id=f"{phase_id}.breach_observation_eod",
                required_path_capabilities=(PathCapability.CLOSED_TRADE_RESULT,),
                accepted_fidelity_classes=_ALL_HISTORICAL,
                scenario_use_permitted=True,
            )
        )
    if phase.dll_amount is not None:
        requirements.append(
            PropRulePathRequirement(
                rule_id=f"{phase_id}.daily_loss_limit",
                required_path_capabilities=(PathCapability.REALIZED_PNL_CHRONOLOGY,),
                accepted_fidelity_classes=_ALL_HISTORICAL,
                scenario_use_permitted=True,
            )
        )
    return tuple(requirements)


def _synthetic_phase(
    *, funded: bool = False, starting_balance: float = 50_000.0
) -> PhaseRules:
    return PhaseRules(
        starting_balance=starting_balance,
        profit_target=None if funded else 3_000.0,
        trail_amount=2_000.0,
        trail_style="eod_floor_realtime_breach",
        trail_locks_at_start=True,
        dll_amount=1_000.0,
        dll_hard=False,
        consistency_pct=None if funded else 50.0,
        min_days=(
            None
            if funded
            else DurationRule(count=2, basis=DayCountBasis.TRADING_DAY)
        ),
        max_eval_days=None,
        unrealized_equity_counts_for_breach=False,
        breach_observation_policy="event_updates_only",
        max_contracts=5,
        micro_scaling_table=((1_000.0, 2), (2_000.0, 5)),
    )


#: The synthetic contract fixture the R3 gate compiles + simulates end to end.
#: verification_status can NEVER exceed synthetic_fixture_verified.
SYNTHETIC_FIXTURE_FIRM = PropFirmContractPayload(
    firm="synthetic_fixture_firm",
    account_type="two_phase_trailing",
    account_size_label="50k",
    contract_version="synthetic-v1",
    effective_date="2026-01-01",
    source_status="unverified",
    source_references=("synthetic://fixture/contract-v1",),
    verification_status="synthetic_fixture_verified",
    evaluation=_synthetic_phase(),
    funded=_synthetic_phase(funded=True),
    payout=PayoutPolicy(
        waiting_period=DurationRule(count=3, basis=DayCountBasis.TRADING_DAY),
        min_between_payouts=DurationRule(count=2, basis=DayCountBasis.TRADING_DAY),
        min_winning_days=DurationRule(count=2, basis=DayCountBasis.WINNING_DAY),
        winning_day_min_pnl=100.0,
        payout_cap_per_period=2_000.0,
        split_pct_trader=90.0,
        min_payout=250.0,
        max_payout=None,
        withdrawal_reduces_threshold=False,
        post_payout_buffer_rule="threshold_unchanged",
        payout_processing=None,
    ),
    fees=FeeSchedule(
        evaluation_fee=150.0,
        activation_fee=100.0,
        recurring_fee=50.0,
        recurring_period=DurationRule(count=10, basis=DayCountBasis.TRADING_DAY),
        reset_fee=80.0,
    ),
    rule_path_requirements=(
        *default_rule_path_requirements(_synthetic_phase(), phase_id="evaluation"),
        *default_rule_path_requirements(
            _synthetic_phase(funded=True), phase_id="funded"
        ),
    ),
    breach_reasons_supported=("trailing_floor", "daily_loss_limit"),
    account_expiration=None,
)


register_identity_pair(
    name="PropFirmContract",
    envelope_cls=PropFirmContractEnvelope,
    payload_cls=PropFirmContractPayload,
    id_field="firm_contract_id",
    example_factory=lambda: SYNTHETIC_FIXTURE_FIRM,
    extra_envelope_fields=("contract_evidence_bundle_id",),
)
