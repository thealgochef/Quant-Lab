"""The frozen funded-payout study plan and the saved result envelope.

The workspace saves an immutable :class:`FundedPayoutPlanEnvelope` before a
worker starts; the worker reads ONLY that envelope (never screen state). The
worker saves one :class:`FundedPayoutResultEnvelope`, whose payload binds the
exact result table bytes by SHA-256; the screen and the review folder both
read that verified result.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from alpha_lab.propsim.funded.clock import (
    TWO_BUSINESS_DAYS_FED_1600,
    ProcessingClockPolicy,
)
from alpha_lab.propsim.funded.profiles import (
    MYFUNDEDFUTURES_PROFILE,
    TAKEPROFITTRADER_PROFILE,
    FundedFirmProfile,
)

__all__ = [
    "PLAN_SCHEMA",
    "PriceEvidencePolicy",
    "StrategySourceRef",
    "OwnerDecisionRef",
    "FundedPayoutPlanPayload",
    "FundedPayoutPlanEnvelope",
    "FundedPayoutResultPayload",
    "FundedPayoutResultEnvelope",
    "MATERIAL_LIMITATIONS",
    "PILOT_OWNER_DECISIONS",
    "FUNDED_QUESTION",
]

PLAN_SCHEMA = "funded_payout_plan_v1"

PriceEvidencePolicy = Literal[
    "ordered_trade_prints_required",
    "ordered_trade_prints_with_labeled_minute_fallback",
    "minute_bars_adverse_first",
    "minute_bars_favorable_first",
]

FUNDED_QUESTION = (
    "Which firm's funded operation produced the most cash received after all "
    "modeled account costs, running the same selected strategy and size on the "
    "same market period — and what risk, spending and waiting did that involve?"
)


class StrategySourceRef(FrozenContract):
    kind: Literal["verified_study_package"] = "verified_study_package"
    package_run_id: str = Field(pattern=SHA256_PATTERN)
    package_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    package_root_name: str
    profile_id: str
    description: str
    executions_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_first_day: str
    evaluation_last_day: str
    executions: int = Field(ge=0)


class OwnerDecisionRef(FrozenContract):
    decided_on: str
    subject: str
    decision: str
    status: Literal["owner_confirmed", "owner_approved_for_pilot", "assumption"]


PILOT_OWNER_DECISIONS: tuple[OwnerDecisionRef, ...] = (
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Payout processing clock",
        decision="Two business days after the request date, skipping weekends and US "
        "Federal Reserve bank holidays, paid at 4:00 PM Chicago.",
        status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Pilot strategy",
        decision="Daily-close control S0_D160_W1_P0 on the same January 13 – June 10, "
        "2026 period; no new dates and no sweep.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Pilot size",
        decision="One mini Nasdaq-100 contract in both firms.", status="owner_confirmed"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Credits, growth and booking",
        decision="Initial five credits buy the first five accounts; five credits on the "
        "first Chicago day of later months; unused credits carry; growth tested on the "
        "received after-split wallet after receipts post, with positive net cash, at most "
        "one five-account purchase per firm per receipt day; gross debited once at "
        "request and after-split cash credited once at completion.",
        status="owner_approved_for_pilot"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Costs and mark price",
        decision="$5.14 per mini at entry and $5.14 at exit; open-position equity marked "
        "at the last traded price from ordered trade prints when available.",
        status="owner_approved_for_pilot"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="MyFundedFutures equality before the lock",
        decision="Before the floor locks, equity at or below the floor fails (explicit "
        "assumption). After it locks at +$100, only equity strictly below +$100 fails.",
        status="assumption"),
    OwnerDecisionRef(
        decided_on="2026-09-22", subject="Purpose",
        decision="The pilot validates the simulator. It does not select the final strategy "
        "or position size.", status="owner_confirmed"),
)

MATERIAL_LIMITATIONS: tuple[str, ...] = (
    "Accounts follow the strategy's one shared signal sequence. When an account sits out "
    "(payout pause or a slot waiting for a credit), setups that a separately paused copy "
    "of the strategy might have formed while the shared strategy was in a trade are not "
    "created.",
    "Stops and targets fill at the strategy's recorded prices. Only the loss-limit "
    "liquidation uses the first recorded trade at or through the limit (gap-through). "
    "The price evidence counts stop exits whose triggering trade was worse than the "
    "stop. This is not broker execution.",
    "Where the recorded trades reach the other exit level first inside the minute the "
    "study resolved, the study's one-minute stop-first result is kept and counted.",
    "The five copies in a firm trade the same signals on one market path. They are not "
    "independent experiments, and no probability of future payouts is implied.",
    "The firm terms are the owner's simulation terms, not a claim of complete "
    "compliance with each live program.",
    "MyFundedFutures' equality rule before its floor locks is an explicit assumption.",
    "One historical period (January 13 – June 10, 2026). This pilot validates the "
    "simulator; it does not select the final strategy or position size.",
)


class FundedPayoutPlanPayload(FrozenContract):
    plan_schema: Literal["funded_payout_plan_v1"] = PLAN_SCHEMA
    purpose: Literal["pilot_validation", "engineering_sample"]
    question: str = FUNDED_QUESTION
    source: StrategySourceRef
    instrument: Literal["mini", "micro"]
    quantity: int = Field(gt=0)
    cost_per_side_cents: int = Field(ge=0)
    processing: ProcessingClockPolicy = TWO_BUSINESS_DAYS_FED_1600
    firm_profiles: tuple[FundedFirmProfile, ...] = (
        TAKEPROFITTRADER_PROFILE, MYFUNDEDFUTURES_PROFILE,
    )
    price_evidence_policy: PriceEvidencePolicy
    owner_decisions: tuple[OwnerDecisionRef, ...]
    limitations: tuple[str, ...] = MATERIAL_LIMITATIONS
    authorized_scope: str


class FundedPayoutPlanEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_plan_id"

    funded_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedPayoutPlanPayload


class FundedPayoutResultPayload(FrozenContract):
    result_schema: Literal["funded_payout_result_v1"] = "funded_payout_result_v1"
    funded_plan_id: str = Field(pattern=SHA256_PATTERN)
    result_json_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_passed: bool
    engine_version: str


class FundedPayoutResultEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_result_id"

    funded_result_id: str = Field(pattern=SHA256_PATTERN)
    payload: FundedPayoutResultPayload


def _example_plan() -> FundedPayoutPlanPayload:
    return FundedPayoutPlanPayload(
        purpose="engineering_sample",
        source=StrategySourceRef(
            package_run_id="a" * 64, package_manifest_sha256="b" * 64,
            package_root_name="example", profile_id="S0_D160_W1_P0",
            description="example", executions_sha256="c" * 64,
            evaluation_first_day="2026-01-13", evaluation_last_day="2026-06-10",
            executions=0,
        ),
        instrument="mini", quantity=1, cost_per_side_cents=514,
        price_evidence_policy="ordered_trade_prints_required",
        owner_decisions=PILOT_OWNER_DECISIONS,
        authorized_scope="example",
    )


register_identity_pair(
    name="FundedPayoutPlan",
    envelope_cls=FundedPayoutPlanEnvelope,
    payload_cls=FundedPayoutPlanPayload,
    id_field="funded_plan_id",
    example_factory=_example_plan,
)
register_identity_pair(
    name="FundedPayoutResult",
    envelope_cls=FundedPayoutResultEnvelope,
    payload_cls=FundedPayoutResultPayload,
    id_field="funded_result_id",
    example_factory=lambda: FundedPayoutResultPayload(
        funded_plan_id="a" * 64, result_json_sha256="b" * 64,
        validation_passed=True, engine_version="funded_engine_v1",
    ),
)
