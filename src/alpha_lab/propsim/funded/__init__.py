"""Funded-only payout simulation (funded-payout Version 3 specification).

Two independent firm instances (TakeProfitTrader and MyFundedFutures) replay
the SAME strategy executions on the same market clock. Each instance owns its
accounts, monthly purchase credits, acquisition costs, payout wallet and
growth. Accounts start directly at funded status; there is no evaluation.

The package extends ``alpha_lab.propsim`` for the event-time funded lane. The
legacy day-block :class:`~alpha_lab.propsim.account.AccountWalk` stays
unchanged for historical evaluation/funded studies: it walks closed trades per
day and refuses payout-processing delays, so it cannot represent intraday
open-equity enforcement, a two-day processing pause or per-firm budgets.

Money is integer US cents and prices are integer ticks throughout; floats
never enter the ledgers.
"""

from alpha_lab.propsim.funded.profiles import (
    FIRM_PROFILES,
    MYFUNDEDFUTURES_PROFILE,
    TAKEPROFITTRADER_PROFILE,
    FundedFirmProfile,
)

__all__ = [
    "FIRM_PROFILES",
    "MYFUNDEDFUTURES_PROFILE",
    "TAKEPROFITTRADER_PROFILE",
    "FundedFirmProfile",
]
