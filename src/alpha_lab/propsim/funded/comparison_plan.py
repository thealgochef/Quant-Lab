"""Frozen plan, owner approval and result envelopes for the configuration comparison.

The study configurator freezes an immutable :class:`FundedComparisonPlanEnvelope`
(every resolved configuration, the firms, size, dates, execution model and the
processing clock). A historical run additionally needs a stored
:class:`FundedComparisonApprovalEnvelope` naming that exact plan id; the worker
refuses to start without it. Approvals are recorded only from an explicit owner
action or an owner statement quoted in the approval; nothing here manufactures
one. The worker saves one :class:`FundedComparisonResultEnvelope` whose payload
binds the exact result bytes by SHA-256.
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
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, ProcessingClockPolicy
from alpha_lab.propsim.funded.plan import OwnerDecisionRef
from alpha_lab.propsim.funded.position_walk import EXECUTION_MODEL_ID, EXECUTION_MODEL_TEXT
from alpha_lab.propsim.funded.profiles import (
    MYFUNDEDFUTURES_PROFILE,
    TAKEPROFITTRADER_PROFILE,
    FundedFirmProfile,
)

__all__ = [
    "COMPARISON_PLAN_SCHEMA",
    "COMPARISON_QUESTION",
    "COMPARISON_DECISIONS",
    "COMPARISON_LIMITATIONS",
    "ComparisonConfigurationRef",
    "ComparisonSourceRef",
    "ExecutionModelRef",
    "FundedComparisonPlanPayload",
    "FundedComparisonPlanEnvelope",
    "FundedComparisonApprovalPayload",
    "FundedComparisonApprovalEnvelope",
    "FundedComparisonResultPayload",
    "FundedComparisonResultEnvelope",
    "PLAN_STORE",
    "APPROVAL_STORE",
    "RESULT_STORE",
]

COMPARISON_PLAN_SCHEMA = "funded_comparison_plan_v1"
PLAN_STORE = "funded_comparison_plans"
APPROVAL_STORE = "funded_comparison_approvals"
RESULT_STORE = "funded_comparison_results"

COMPARISON_QUESTION = (
    "Which tested strategy configuration produces the most simulated cash received after "
    "the cost of every funded account used, over the same selected historical period?"
)

COMPARISON_DECISIONS: tuple[OwnerDecisionRef, ...] = (
    OwnerDecisionRef(
        decided_on="2026-09-23", subject="Comparison mode",
        decision="Each configuration and each selected firm is a separate comparison with "
        "at most one live funded account at a time. A failed account is replaced by a fresh "
        "funded account under the same configuration, charged $102 (TakeProfitTrader) or "
        "$125 (MyFundedFutures), with no credit limit. No five-account group, monthly "
        "credits, growth or copied accounts in this mode.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-23", subject="Objective",
        decision="Rank by cash received after the split minus every account purchase. Large "
        "payouts count even if the account later fails; trade count, survival and win rate "
        "are not substitutes.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-23", subject="Payout processing is not bypassed",
        decision="A payout-processing account is alive; it is never replaced or supplemented "
        "and trades nothing until processing completes.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Payout processing clock",
        decision="Two business days after the request date, skipping weekends and US Federal "
        "Reserve bank holidays, paid at 4:00 PM Chicago.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Withdrawal policy",
        decision="Stop entries once realized profit after costs is at least $500 above the "
        "retained $2,100; request the full surplus at the end of that trading day.",
        status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Costs",
        decision="$5.14 per mini at entry and $5.14 at exit, deducted inside the account "
        "only.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="MyFundedFutures equality before the lock",
        decision="Before the floor locks, equity at or below the floor fails (assumption). "
        "After it locks at +$100, only equity strictly below +$100 fails.",
        status="assumption"),
)

COMPARISON_LIMITATIONS: tuple[str, ...] = (
    "One historical path (January 13 – June 10, 2026). The results compare configurations "
    "on that path; they are not probabilities of future payouts.",
    "A replacement account is assumed to be available at once when an account fails (no "
    "evaluation or purchase delay is modeled).",
    "Entries fill at the confirming one-minute candle's closing trade, as the strategy "
    "records them. Stops fill at the first recorded trade at or through the stop; this is "
    "not broker execution and ignores queue position and liquidity.",
    "An entry the strategy would have made, refused only because of payout protection or "
    "processing, discards that setup; the strategy may form new setups afterwards.",
    "The period ends at the last day's 4:00 PM Chicago close. A payout paid at exactly that "
    "time counts as received; anything later is shown as pending.",
    "Where a minute's recorded trades do not rebuild the strategy's candle exactly, that "
    "minute uses a labeled one-minute approximation (losing side first).",
    "The firm terms are the owner's simulation terms, not a claim of complete compliance "
    "with each live program. MyFundedFutures' equality rule before its floor locks is an "
    "explicit assumption.",
)


class ComparisonConfigurationRef(FrozenContract):
    name: str
    display_name: str
    axis_value_ids: tuple[tuple[str, str], ...]
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    approval_id: str = Field(pattern=SHA256_PATTERN)


class ComparisonSourceRef(FrozenContract):
    kind: Literal["verified_study_package"] = "verified_study_package"
    package_run_id: str = Field(pattern=SHA256_PATTERN)
    package_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    package_root_name: str
    title: str
    warmup_dates: tuple[str, ...]
    evaluation_dates: tuple[str, ...]
    cutoff_utc: str


class ExecutionModelRef(FrozenContract):
    model_id: Literal["ordered_prints_stop_market_v2"] = EXECUTION_MODEL_ID
    description: str = EXECUTION_MODEL_TEXT
    price_evidence_policy: Literal["ordered_trade_prints_with_labeled_minute_fallback"] = (
        "ordered_trade_prints_with_labeled_minute_fallback")
    refused_entry_policy: Literal["discard_refused_setup"] = "discard_refused_setup"
    replacement_policy: Literal["immediate_fresh_account_same_configuration"] = (
        "immediate_fresh_account_same_configuration")


class FundedComparisonPlanPayload(FrozenContract):
    plan_schema: Literal["funded_comparison_plan_v1"] = COMPARISON_PLAN_SCHEMA
    mode: Literal["single_account_configuration_comparison"] = (
        "single_account_configuration_comparison")
    purpose: Literal["historical_comparison", "engineering_sample"]
    question: str = COMPARISON_QUESTION
    source: ComparisonSourceRef
    configurations: tuple[ComparisonConfigurationRef, ...] = Field(min_length=1)
    firm_profiles: tuple[FundedFirmProfile, ...] = (TAKEPROFITTRADER_PROFILE,
                                                   MYFUNDEDFUTURES_PROFILE)
    instrument: Literal["mini", "micro"]
    quantity: int = Field(gt=0)
    cost_per_side_cents: int = Field(ge=0)
    processing: ProcessingClockPolicy = TWO_BUSINESS_DAYS_FED_1600
    execution_model: ExecutionModelRef = ExecutionModelRef()
    owner_decisions: tuple[OwnerDecisionRef, ...] = COMPARISON_DECISIONS
    limitations: tuple[str, ...] = COMPARISON_LIMITATIONS

    @model_validator(mode="after")
    def _distinct(self) -> FundedComparisonPlanPayload:
        names = [c.name for c in self.configurations]
        hashes = [c.resolved_section_config_hash for c in self.configurations]
        if len(set(names)) != len(names) or len(set(hashes)) != len(hashes):
            raise ValueError("each configuration must be listed once")
        firms = [p.firm_key for p in self.firm_profiles]
        if not firms or len(set(firms)) != len(firms):
            raise ValueError("select at least one firm, each once")
        return self


class FundedComparisonPlanEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_plan_id"

    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedComparisonPlanPayload


class FundedComparisonApprovalPayload(FrozenContract):
    approval_schema: Literal["funded_comparison_approval_v1"] = "funded_comparison_approval_v1"
    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    approved_on: str
    approved_by: Literal["owner"] = "owner"
    channel: Literal["study_screen", "claude_conversation"]
    statement: str = Field(min_length=20)
    scope: str = Field(min_length=10)


class FundedComparisonApprovalEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_approval_id"

    funded_comparison_approval_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedComparisonApprovalPayload


class FundedComparisonResultPayload(FrozenContract):
    result_schema: Literal["funded_comparison_result_v1"] = "funded_comparison_result_v1"
    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    funded_comparison_approval_id: str | None = None
    result_json_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_passed: bool
    engine_version: str


class FundedComparisonResultEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_result_id"

    funded_comparison_result_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedComparisonResultPayload


def _example_plan() -> FundedComparisonPlanPayload:
    return FundedComparisonPlanPayload(
        purpose="engineering_sample",
        source=ComparisonSourceRef(
            package_run_id="a" * 64, package_manifest_sha256="b" * 64,
            package_root_name="example", title="example",
            warmup_dates=("2026-01-01",), evaluation_dates=("2026-01-13",),
            cutoff_utc="2026-01-13T21:00:00Z"),
        configurations=(ComparisonConfigurationRef(
            name="S0_D160_W1_P0", display_name="example",
            axis_value_ids=(("opposing_parent_distance_ticks_max",
                             "opposing_parent_distance_ticks_max.160"),),
            resolved_section_config_hash="c" * 64, approval_id="d" * 64),),
        instrument="mini", quantity=1, cost_per_side_cents=514,
    )


register_identity_pair(
    name="FundedComparisonPlan", envelope_cls=FundedComparisonPlanEnvelope,
    payload_cls=FundedComparisonPlanPayload, id_field="funded_comparison_plan_id",
    example_factory=_example_plan,
)
register_identity_pair(
    name="FundedComparisonApproval", envelope_cls=FundedComparisonApprovalEnvelope,
    payload_cls=FundedComparisonApprovalPayload, id_field="funded_comparison_approval_id",
    example_factory=lambda: FundedComparisonApprovalPayload(
        funded_comparison_plan_id="a" * 64, approved_on="2026-09-23",
        channel="study_screen", statement="Example owner approval statement text.",
        scope="example scope"),
)
register_identity_pair(
    name="FundedComparisonResult", envelope_cls=FundedComparisonResultEnvelope,
    payload_cls=FundedComparisonResultPayload, id_field="funded_comparison_result_id",
    example_factory=lambda: FundedComparisonResultPayload(
        funded_comparison_plan_id="a" * 64, result_json_sha256="b" * 64,
        validation_passed=True, engine_version="funded_comparison_engine_v1"),
)


# ── version 2: variations with their own size and exit rule ───────────────────

COMPARISON_PLAN_SCHEMA_V2 = "funded_comparison_plan_v2"


class ComparisonVariantRef(FrozenContract):
    """One configuration of a v2 plan, with its own position size and exit rule."""

    name: str
    display_name: str
    axis_value_ids: tuple[tuple[str, str], ...]
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    exit_policy: Literal["fixed_target_v1", "scale_out_half_breakeven_hold_to_close_v1"]
    instrument: Literal["mini", "micro"]
    quantity: int = Field(gt=0)
    #: exact cost per contract per fill in tenths of a cent ($5.14 = 5140, $0.514 = 514)
    cost_per_contract_mills: int = Field(ge=0)
    in_verified_study: bool
    #: verified study configuration whose cached charts (a superset) and approved
    #: read policy supply this configuration's day inputs
    cache_configuration: str

    @model_validator(mode="after")
    def _sizes(self) -> ComparisonVariantRef:
        if self.exit_policy != "fixed_target_v1":
            if self.quantity % 2:
                raise ValueError("the scale-out exit needs an even number of contracts")
            target = dict(self.axis_value_ids).get("tp_r_multiple", "tp_r_multiple.1.0")
            if target != "tp_r_multiple.1.0":
                raise ValueError("the scale-out exit takes its half at 1R in this study")
        return self


class CoreSourceRef(FrozenContract):
    """The exact Strategy-Core source the worker must import."""

    base_commit: str
    branch: str
    patch_sha256: str = Field(pattern=SHA256_PATTERN)
    description: str


class FundedComparisonPlanPayloadV2(FrozenContract):
    plan_schema: Literal["funded_comparison_plan_v2"] = COMPARISON_PLAN_SCHEMA_V2
    mode: Literal["single_account_configuration_comparison"] = (
        "single_account_configuration_comparison")
    purpose: Literal["historical_comparison", "engineering_sample"]
    question: str = COMPARISON_QUESTION
    source: ComparisonSourceRef
    base_configuration: str
    variants: tuple[ComparisonVariantRef, ...] = Field(min_length=1)
    firm_profiles: tuple[FundedFirmProfile, ...] = (TAKEPROFITTRADER_PROFILE,
                                                   MYFUNDEDFUTURES_PROFILE)
    processing: ProcessingClockPolicy = TWO_BUSINESS_DAYS_FED_1600
    execution_model: ExecutionModelRef = ExecutionModelRef()
    core_source: CoreSourceRef
    owner_decisions: tuple[OwnerDecisionRef, ...] = COMPARISON_DECISIONS
    limitations: tuple[str, ...] = COMPARISON_LIMITATIONS

    @model_validator(mode="after")
    def _distinct(self) -> FundedComparisonPlanPayloadV2:
        names = [v.name for v in self.variants]
        hashes = [v.resolved_section_config_hash for v in self.variants]
        if len(set(names)) != len(names) or len(set(hashes)) != len(hashes):
            raise ValueError("each configuration must be listed once")
        firms = [p.firm_key for p in self.firm_profiles]
        if not firms or len(set(firms)) != len(firms):
            raise ValueError("select at least one firm, each once")
        return self

    # the v1 names the worker and screen already use
    @property
    def configurations(self) -> tuple[ComparisonVariantRef, ...]:
        return self.variants


class FundedComparisonPlanEnvelopeV2(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_plan_id"

    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedComparisonPlanPayloadV2


def _example_plan_v2() -> FundedComparisonPlanPayloadV2:
    v1 = _example_plan()
    return FundedComparisonPlanPayloadV2(
        purpose="engineering_sample", source=v1.source, base_configuration="S0_D160_W1_P0",
        variants=(ComparisonVariantRef(
            name="S0-T1-H14-P1-L-SO", display_name="example",
            axis_value_ids=(("tp_r_multiple", "tp_r_multiple.1.0"),),
            resolved_section_config_hash="c" * 64,
            exit_policy="scale_out_half_breakeven_hold_to_close_v1", instrument="micro",
            quantity=10, cost_per_contract_mills=514, in_verified_study=False,
            cache_configuration="S0_D160_W1_P0"),),
        core_source=CoreSourceRef(base_commit="7" * 40, branch="example",
                                  patch_sha256="d" * 64, description="example"),
    )


register_identity_pair(
    name="FundedComparisonPlanV2", envelope_cls=FundedComparisonPlanEnvelopeV2,
    payload_cls=FundedComparisonPlanPayloadV2, id_field="funded_comparison_plan_id",
    example_factory=_example_plan_v2,
)
