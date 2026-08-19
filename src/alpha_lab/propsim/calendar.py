"""Typed day-count and simulated-clock semantics (CS §5.2; P0-14).

Every firm time rule is a :class:`DurationRule` over a typed
:class:`DayCountBasis` under a :class:`SimulatedClockPolicy`; a rule whose
basis the active clock cannot represent (e.g., a calendar-month recurring fee
under day-block bootstrap without a synthetic calendar) FAILS CLOSED with
:class:`UnsupportedCalendarRuleError` — durations are never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    FrozenContract,
    ImmutableMap,
)

__all__ = [
    "DayCountBasis",
    "DurationRule",
    "FirmCalendarPolicy",
    "SimulatedClockPolicy",
    "SimulatedClock",
    "UnsupportedCalendarRuleError",
    "DEFAULT_FIRM_CALENDAR",
    "HISTORICAL_CLOCK_POLICY",
    "BOOTSTRAP_CLOCK_POLICY",
]


class UnsupportedCalendarRuleError(ValueError):
    """The active clock cannot represent this rule's day-count basis."""


class DayCountBasis(StrEnum):
    TRADING_DAY = "trading_day"
    WINNING_DAY = "winning_day"
    BUSINESS_DAY = "business_day"
    CALENDAR_DAY = "calendar_day"
    CALENDAR_MONTH = "calendar_month"
    FIRM_DEFINED_PAYOUT_PERIOD = "firm_defined_payout_period"


class DurationRule(FrozenContract):
    count: int = Field(ge=0)
    basis: DayCountBasis
    firm_calendar_policy_id: str | None = None


class FirmCalendarPolicy(FrozenContract):
    policy_id: str
    trading_day_boundary: Literal["18:00_ET_roll"] = "18:00_ET_roll"
    business_day_definition: str = "mon_fri_excluding_holidays"
    holiday_handling: str = "holidays_are_non_business_days"
    holidays: tuple[str, ...] = ()


class SimulatedClockPolicy(FrozenContract):
    policy_id: str
    advancement_rules: ImmutableMap[str, str]

    _SUPPORTED_MARK: ClassVar[str] = "supported"

    def supports(self, basis: DayCountBasis) -> bool:
        return self.advancement_rules.get(basis.value) == self._SUPPORTED_MARK


DEFAULT_FIRM_CALENDAR = FirmCalendarPolicy(policy_id="firm_calendar_18et_v1")

#: Historical replay has real dates: every basis is representable.
HISTORICAL_CLOCK_POLICY = SimulatedClockPolicy(
    policy_id="historical_calendar_clock_v1",
    advancement_rules={basis.value: "supported" for basis in DayCountBasis},
)

#: Day-block bootstrap resamples trading days WITHOUT a real calendar: only
#: trading/winning-day (and firm payout-period counters defined in trading
#: days) bases are representable; business/calendar bases FAIL CLOSED.
BOOTSTRAP_CLOCK_POLICY = SimulatedClockPolicy(
    policy_id="bootstrap_trading_day_clock_v1",
    advancement_rules={
        DayCountBasis.TRADING_DAY.value: "supported",
        DayCountBasis.WINNING_DAY.value: "supported",
        DayCountBasis.FIRM_DEFINED_PAYOUT_PERIOD.value: "supported",
        DayCountBasis.BUSINESS_DAY.value: "unrepresentable_without_calendar",
        DayCountBasis.CALENDAR_DAY.value: "unrepresentable_without_calendar",
        DayCountBasis.CALENDAR_MONTH.value: "unrepresentable_without_calendar",
    },
)


@dataclass
class SimulatedClock:
    """Advances over played trading days and answers typed duration queries.

    ``advance(day, winning)`` is called once per played trading day IN ORDER
    (historical dates, or synthetic bootstrap ordinals mapped onto a
    deterministic weekday-only synthetic calendar when the policy supports
    it). ``elapsed(basis)`` / ``satisfied(rule)`` refuse unsupported bases.
    """

    policy: SimulatedClockPolicy
    calendar: FirmCalendarPolicy = field(default_factory=lambda: DEFAULT_FIRM_CALENDAR)
    _days: list[date] = field(default_factory=list)
    _winning: list[bool] = field(default_factory=list)

    def advance(self, day: date, *, winning: bool) -> None:
        if self._days and day <= self._days[-1]:
            raise ValueError("the simulated clock only advances forward")
        self._days.append(day)
        self._winning.append(winning)

    @property
    def current_day(self) -> date | None:
        return self._days[-1] if self._days else None

    @property
    def played_days(self) -> int:
        return len(self._days)

    def _require(self, basis: DayCountBasis) -> None:
        if not self.policy.supports(basis):
            raise UnsupportedCalendarRuleError(
                f"day-count basis {basis.value!r} is not representable under "
                f"clock policy {self.policy.policy_id!r} "
                f"({self.policy.advancement_rules.get(basis.value)})"
            )

    def _business_days_between(self, start: date, end: date) -> int:
        holidays = {date.fromisoformat(day) for day in self.calendar.holidays}
        count = 0
        cursor = start
        while cursor < end:
            cursor += timedelta(days=1)
            if cursor.weekday() < 5 and cursor not in holidays:
                count += 1
        return count

    def elapsed_since(self, since_index: int, basis: DayCountBasis) -> int:
        """Elapsed units from the played day at ``since_index`` to now."""

        self._require(basis)
        if not self._days:
            return 0
        since_index = max(0, since_index)
        if basis in (DayCountBasis.TRADING_DAY, DayCountBasis.FIRM_DEFINED_PAYOUT_PERIOD):
            return len(self._days) - 1 - since_index
        if basis is DayCountBasis.WINNING_DAY:
            return sum(1 for flag in self._winning[since_index + 1 :] if flag)
        start, end = self._days[since_index], self._days[-1]
        if basis is DayCountBasis.CALENDAR_DAY:
            return (end - start).days
        if basis is DayCountBasis.BUSINESS_DAY:
            return self._business_days_between(start, end)
        if basis is DayCountBasis.CALENDAR_MONTH:
            return (end.year - start.year) * 12 + (end.month - start.month)
        raise UnsupportedCalendarRuleError(basis.value)  # pragma: no cover

    def satisfied(self, rule: DurationRule, *, since_index: int = 0) -> bool:
        return self.elapsed_since(since_index, rule.basis) >= rule.count

    def total(self, basis: DayCountBasis) -> int:
        self._require(basis)
        if basis is DayCountBasis.WINNING_DAY:
            return sum(1 for flag in self._winning if flag)
        if basis in (DayCountBasis.TRADING_DAY, DayCountBasis.FIRM_DEFINED_PAYOUT_PERIOD):
            return len(self._days)
        return self.elapsed_since(0, basis)
