"""Owner-defined funded firm profiles (specification section 2).

These are OWNER-DEFINED SIMULATION TERMS, not a claim of complete live-program
compliance. Sourced first-party mechanics fill only the loss-threshold timing
and lock level; they never overwrite an owner-specified value. Every value that
is an assumption rather than a documented rule says so in ``assumptions``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import FrozenContract

__all__ = [
    "Comparator",
    "RuleSource",
    "FundedFirmProfile",
    "TAKEPROFITTRADER_PROFILE",
    "MYFUNDEDFUTURES_PROFILE",
    "FIRM_PROFILES",
    "INSTRUMENTS",
    "InstrumentSpec",
    "breaches",
]

Comparator = Literal["at_or_below", "below"]


def breaches(equity_cents: int, floor_cents: int, comparator: Comparator) -> bool:
    """The compiled loss-floor comparison (exact integer cents)."""

    if comparator == "at_or_below":
        return equity_cents <= floor_cents
    return equity_cents < floor_cents


class RuleSource(FrozenContract):
    source_id: str
    title: str
    url: str
    checked_on: str
    supports: str


class InstrumentSpec(FrozenContract):
    """One tradable product. Tick value is exact cents per tick per contract."""

    key: Literal["mini", "micro"]
    label: str
    symbol_root: str
    tick_size_points: str = "0.25"
    tick_value_cents: int
    mini_equivalent_tenths: int  # a mini = 10 tenths, a micro = 1 tenth


INSTRUMENTS: dict[str, InstrumentSpec] = {
    "mini": InstrumentSpec(
        key="mini", label="E-mini Nasdaq-100 (NQ)", symbol_root="NQ",
        tick_value_cents=500, mini_equivalent_tenths=10,
    ),
    "micro": InstrumentSpec(
        key="micro", label="Micro E-mini Nasdaq-100 (MNQ)", symbol_root="MNQ",
        tick_value_cents=50, mini_equivalent_tenths=1,
    ),
}


class FundedFirmProfile(FrozenContract):
    profile_version: str
    firm_key: Literal["takeprofittrader", "myfundedfutures"]
    firm_name: str
    account_label: str
    provenance: Literal["owner_defined_simulation_terms"] = "owner_defined_simulation_terms"
    nominal_balance_cents: int = 5_000_000
    acquisition_cost_cents: int = Field(gt=0)
    loss_allowance_cents: int = 200_000
    #: WHEN the floor moves. Enforcement is ALWAYS against current equity on
    #: every ordered observation while a position is open.
    threshold_update: Literal["intraday_peak_equity", "session_close_balance"]
    floor_lock_cents: int
    comparator_before_lock: Comparator
    comparator_after_lock: Comparator
    trader_share_pct: int = Field(gt=0, le=100)
    minimum_gross_request_cents: int = 50_000
    retained_cushion_cents: int = 210_000
    max_mini_equivalent_tenths: int = Field(gt=0)
    monthly_credits: int = 5
    initial_accounts: int = 5
    capacity_step: int = 5
    max_capacity: int = 20
    growth_share_bps: int = 2_500
    consistency_rule: Literal["none"] = "none"
    sources: tuple[RuleSource, ...]
    assumptions: tuple[str, ...]

    @property
    def max_minis_label(self) -> str:
        return f"{self.max_mini_equivalent_tenths // 10} minis or equivalent"

    @property
    def cost_of_next_block_cents(self) -> int:
        return self.acquisition_cost_cents * self.capacity_step

    @property
    def growth_threshold_cents(self) -> int:
        # cost <= share * wallet  <=>  wallet >= cost * 10000 / bps (exact)
        return -(-self.cost_of_next_block_cents * 10_000 // self.growth_share_bps)

    def comparator_for(self, floor_cents: int) -> Comparator:
        if floor_cents >= self.floor_lock_cents:
            return self.comparator_after_lock
        return self.comparator_before_lock


_CHECKED = "2026-09-22"

TAKEPROFITTRADER_PROFILE = FundedFirmProfile(
    profile_version="takeprofittrader_funded_owner_v3",
    firm_key="takeprofittrader",
    firm_name="TakeProfitTrader",
    account_label="$50,000 funded account",
    acquisition_cost_cents=10_200,
    threshold_update="intraday_peak_equity",
    floor_lock_cents=0,
    comparator_before_lock="at_or_below",
    comparator_after_lock="at_or_below",
    trader_share_pct=80,
    max_mini_equivalent_tenths=60,
    sources=(
        RuleSource(
            source_id="S1", title="TakeProfitTrader PRO Account Rules",
            url="https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15171769361053-PRO-Account-Rules",
            checked_on=_CHECKED,
            supports="Peak includes realized and unrealized gains; floor stops at the "
            "starting balance; reaching the floor liquidates the account.",
        ),
        RuleSource(
            source_id="S2", title="TakeProfitTrader: How to keep track of your drawdown",
            url="https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15171820366109-How-to-Keep-Track-Of-Your-Drawdown",
            checked_on=_CHECKED,
            supports="Intraday trailing explanation.",
        ),
        RuleSource(
            source_id="S4", title="TakeProfitTrader: withdraw from PRO account to wallet",
            url="https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15172253980061-How-to-Withdraw-from-PRO-Account-to-the-Wallet",
            checked_on=_CHECKED,
            supports="Gross amount versus 80% received; flat with no working orders.",
        ),
        RuleSource(
            source_id="S7", title="TakeProfitTrader PRO profit split and withdrawal rules",
            url="https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15172219527581-PRO-Account-Profit-Split-Withdrawal-Rules",
            checked_on=_CHECKED,
            supports="Public buffer/split differ from the owner's retained $2,100 rule; the "
            "owner rule is used deliberately.",
        ),
    ),
    assumptions=(
        "Acquisition cost $102, 80% trader share, six-mini limit, $2,100 retained "
        "cushion and $500 gross minimum are owner-defined simulation terms.",
        "Touching the floor (equity at or below it) fails the account, per the "
        "documented touch-liquidates rule.",
        "Open-position equity is marked at the last traded price.",
    ),
)

MYFUNDEDFUTURES_PROFILE = FundedFirmProfile(
    profile_version="myfundedfutures_funded_owner_v3",
    firm_key="myfundedfutures",
    firm_name="MyFundedFutures",
    account_label="$50,000 funded account",
    acquisition_cost_cents=12_500,
    threshold_update="session_close_balance",
    floor_lock_cents=10_000,
    comparator_before_lock="at_or_below",
    comparator_after_lock="below",
    trader_share_pct=90,
    max_mini_equivalent_tenths=30,
    sources=(
        RuleSource(
            source_id="S3", title="MyFundedFutures Rapid end-of-day $50K",
            url="https://help.myfundedfutures.com/en/articles/16158363-rapid-eod-50k-a-comprehensive-look",
            checked_on=_CHECKED,
            supports="End-of-day ratchet locking at +$100; falling below $100 at the "
            "locked floor is a breach; $500 minimum; 90% share.",
        ),
        RuleSource(
            source_id="S8", title="MyFundedFutures end-of-day drawdown explained",
            url="https://help.myfundedfutures.com/en/articles/8348565-end-of-day-eod-drawdown-explained",
            checked_on=_CHECKED,
            supports="Open-position losses count in failure checks (evaluation-framed "
            "page; not used to import evaluation terms).",
        ),
    ),
    assumptions=(
        "Acquisition cost $125, 90% trader share, three-mini limit, $2,100 retained "
        "cushion and $500 gross minimum are owner-defined simulation terms.",
        "BEFORE the floor locks, equity at or below the floor fails the account. This "
        "is an explicit pilot assumption approved by the owner on September 22, 2026; "
        "no precise source establishes the pre-lock equality rule.",
        "AFTER the floor locks at +$100, only equity strictly below +$100 fails "
        "(documented 'falling below' condition).",
        "The floor moves only from the realized balance at the scheduled session "
        "close (4:00 PM Chicago, or the earlier scheduled close), before that day's "
        "withdrawal debit.",
        "Open-position equity is marked at the last traded price.",
    ),
)

FIRM_PROFILES: dict[str, FundedFirmProfile] = {
    TAKEPROFITTRADER_PROFILE.firm_key: TAKEPROFITTRADER_PROFILE,
    MYFUNDEDFUTURES_PROFILE.firm_key: MYFUNDEDFUTURES_PROFILE,
}
