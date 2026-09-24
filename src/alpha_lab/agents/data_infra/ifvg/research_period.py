"""The permitted research period for new studies (repair R8, owner-authorized).

On September 23, 2026 the owner authorized extending the permitted research
window back to the earliest date of the locally stored market data; June 11,
2026 onward stays protected exactly as before. This module is the one place
that states the extended window and resolves a start/end date range to
logical trading days with the existing trading calendar.

Unchanged on purpose: ``PERMITTED_DEVELOPMENT_DATES`` (the date list existing
preparation jobs replay and whose hash their audits record), the registered
calendar closures, and the prepared-input allowlist (2026 only). Prepared
study inputs exist only for 2026; preparing earlier days is a separate,
not-yet-authorized step (under the current cache layout it would overwrite
the 2026 inputs), so an extended-period study can be configured, saved and
validated but not run until that step is decided.

Warmup follows today's rule: evidence starting on or after January 13, 2026
keeps the frozen ten-day warmup; an earlier start uses the ten store days
(every calendar day except Saturday) immediately before the first evidence day.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from .development_access import FROZEN_WARMUP_DATES
from .search.trading_calendar import (
    logical_trading_days,
    physical_partitions_for,
)

__all__ = [
    "EARLIEST_LOCAL_MARKET_DATE",
    "EXTENDED_HISTORY_CLOSURES",
    "FROZEN_EVIDENCE_FIRST_DAY",
    "LAST_PERMITTED_DAY",
    "LOCAL_DATA_BLOCKS",
    "PROTECTED_FROM_DAY",
    "ResolvedResearchRange",
    "earliest_evidence_day",
    "evidence_day_error",
    "local_day_has_data",
    "resolve_research_range",
    "warmup_dates_for",
]

#: Earliest locally stored NQ day file (determined September 23, 2026 from the
#: date-addressed local store; no date on or after June 11, 2026 is consulted).
EARLIEST_LOCAL_MARKET_DATE = "2021-12-02"
#: The continuous blocks of local day files found on September 23, 2026.
LOCAL_DATA_BLOCKS: tuple[tuple[str, str], ...] = (
    ("2021-12-02", "2022-03-10"),
    ("2025-06-02", "2026-06-10"),
)
FROZEN_EVIDENCE_FIRST_DAY = "2026-01-13"
LAST_PERMITTED_DAY = "2026-06-10"
PROTECTED_FROM_DAY = "2026-06-11"
WARMUP_STORE_DAYS = 10
#: Market closures in the extended period that the registered calendar (which
#: covers 2026 only) does not list, each with its local-file evidence. They are
#: excluded from resolved ranges; the registered calendar is not changed.
EXTENDED_HISTORY_CLOSURES: dict[str, str] = {
    "2021-12-24": "market closed (Christmas observed): the stored file holds 8 records, all "
                  "from the previous day",
    "2025-12-25": "market closed (Christmas Day): the stored file holds only the evening "
                  "reopen that belongs to the next trading day, like the registered "
                  "January 1, 2026 closure",
}
_SOURCE_FILES = ("mbp10.parquet", "mbp1.parquet", "trades.parquet")


def _store_days_before(day: str, count: int) -> tuple[str, ...]:
    cursor = date.fromisoformat(day) - timedelta(days=1)
    out: list[str] = []
    while len(out) < count:
        if cursor.weekday() != 5:  # no Saturday ever carries a day file
            out.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return tuple(reversed(out))


def warmup_dates_for(first_evidence_day: str) -> tuple[str, ...]:
    """The warmup days a study starting on ``first_evidence_day`` replays first."""

    if first_evidence_day >= FROZEN_EVIDENCE_FIRST_DAY:
        return FROZEN_WARMUP_DATES
    return _store_days_before(first_evidence_day, WARMUP_STORE_DAYS)


@lru_cache(maxsize=1)
def earliest_evidence_day() -> str:
    """The first logical trading day whose ten warmup store days are all stored."""

    for day in logical_trading_days(EARLIEST_LOCAL_MARKET_DATE, FROZEN_EVIDENCE_FIRST_DAY):
        if warmup_dates_for(day)[0] >= EARLIEST_LOCAL_MARKET_DATE:
            return day
    return FROZEN_EVIDENCE_FIRST_DAY  # pragma: no cover


def evidence_day_error(day: str) -> str | None:
    """Plain reason a date cannot be an evidence day of a new study (None if it can)."""

    try:
        date.fromisoformat(str(day))
    except ValueError:
        return f"{day!r} is not a date (use YYYY-MM-DD)"
    day = str(day)
    if day >= PROTECTED_FROM_DAY:
        return (f"{day} lies outside the development evidence window: June 11, 2026 onward "
                "is protected and can never be used")
    first = earliest_evidence_day()
    if day < first:
        return (f"{day} lies outside the development evidence window: it is before {first}, "
                "the first day with ten stored warmup days before it (local market data "
                f"starts {EARLIEST_LOCAL_MARKET_DATE})")
    if day not in logical_trading_days(day, day) or day in EXTENDED_HISTORY_CLOSURES:
        return f"{day} is not a logical trading day (weekend or market closure)"
    return None


def local_day_has_data(repo_root: Path) -> Callable[[str], bool]:
    """Date-addressed existence check of a stored day file (no directory listing).

    A path is never constructed for June 11, 2026 or later.
    """

    base = Path(repo_root) / "data" / "databento" / "NQ"

    def check(day: str) -> bool:
        if str(day) >= PROTECTED_FROM_DAY:
            raise PermissionError("protected dates are never checked")
        folder = base / str(day)
        return any((folder / name).is_file() for name in _SOURCE_FILES)

    return check


@dataclass(frozen=True)
class ResolvedResearchRange:
    start: str
    end: str
    trading_days: tuple[str, ...]
    warmup_dates: tuple[str, ...]
    #: (day, plain reason) for every calendar trading day left out
    excluded: tuple[tuple[str, str], ...]
    #: problems that prevent saving this range (empty when it can be saved)
    problems: tuple[str, ...]
    #: notes the owner should see before using the range
    warnings: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.problems and bool(self.trading_days)


def resolve_research_range(start: str, end: str, *,
                           day_has_data: Callable[[str], bool] | None = None
                           ) -> ResolvedResearchRange:
    """Resolve a start/end date range to evidence days with the existing calendar.

    Days are left out, with their reason, when the calendar has no session,
    when the market was closed (:data:`EXTENDED_HISTORY_CLOSURES`), or when a
    stored day file needed for that trading day is missing locally.
    """

    problems: list[str] = []
    warnings: list[str] = []
    for label, value in (("start", start), ("end", end)):
        try:
            date.fromisoformat(str(value))
        except ValueError:
            return ResolvedResearchRange(str(start), str(end), (), (), (), (
                f"The {label} date {value!r} is not a date.",), ())
    if end >= PROTECTED_FROM_DAY:
        problems.append("The end date must be June 10, 2026 or earlier: June 11, 2026 onward "
                        "is protected and can never be used.")
    first_allowed = earliest_evidence_day()
    if start < first_allowed:
        problems.append(f"The start date must be {first_allowed} or later: local market data "
                        f"starts {EARLIEST_LOCAL_MARKET_DATE} and a study first replays ten "
                        "warmup days.")
    if end < start:
        problems.append("The end date is before the start date.")
    if problems:
        return ResolvedResearchRange(start, end, (), (), (), tuple(problems), ())
    calendar = logical_trading_days(start, end)
    kept: list[str] = []
    excluded: list[tuple[str, str]] = []
    for day in calendar:
        if day in EXTENDED_HISTORY_CLOSURES:
            excluded.append((day, EXTENDED_HISTORY_CLOSURES[day]))
            continue
        if day_has_data is not None and day < FROZEN_EVIDENCE_FIRST_DAY:
            missing = [p.physical_utc_date for p in physical_partitions_for(day)
                       if not day_has_data(p.physical_utc_date)]
            if missing:
                excluded.append((day, "; ".join(
                    f"no stored market data for {m}" if m == day else
                    f"no stored market data for the previous evening ({m}) that opens this "
                    "trading day" for m in missing)))
                continue
        kept.append(day)
    warmup = warmup_dates_for(kept[0]) if kept else ()
    if kept and day_has_data is not None and kept[0] < FROZEN_EVIDENCE_FIRST_DAY:
        absent = [d for d in warmup if not day_has_data(d)]
        if absent:
            problems.append("These warmup days have no stored market data: "
                            + ", ".join(absent) + ". Choose a later start date.")
    no_data = sum(1 for _d, reason in excluded if reason.startswith("no stored market data"))
    if no_data > WARMUP_STORE_DAYS:
        warnings.append(f"{no_data} trading days in this range have no stored market data "
                        "(local data is stored for "
                        + " and ".join(f"{a} to {b}" for a, b in LOCAL_DATA_BLOCKS)
                        + "). The strategy would carry its state across that gap.")
    first_replayed = (warmup or kept)[0] if kept else None  # warmup days replay first
    if first_replayed is not None and first_replayed < "2026-01-01":
        warnings.append("Prepared study inputs exist only from January 1, 2026, and this range "
                        f"first replays {first_replayed} (its warmup days come first). It can "
                        "be saved and checked, but it cannot run until preparing the earlier "
                        "days is separately authorized.")
    if not kept and not problems:
        problems.append("No trading day with stored market data lies in this range.")
    return ResolvedResearchRange(start, end, tuple(kept), tuple(warmup), tuple(excluded),
                                 tuple(problems), tuple(warnings))


def dates_match_range(dates: Iterable[str], resolved: ResolvedResearchRange) -> bool:
    return tuple(dates) == resolved.trading_days
