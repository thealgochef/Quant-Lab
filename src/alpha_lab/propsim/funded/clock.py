"""Time helpers and the payout-processing clock (specification section 5).

All engine timestamps are integer nanoseconds since the Unix epoch (UTC).
Durations are computed on absolute time; Chicago wall-clock time is only
used where a rule is DEFINED in local terms (business-day payment time,
monthly grant boundaries) and for display.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from alpha_lab.agents.data_infra.ifvg.search.identities import FrozenContract

__all__ = [
    "CHICAGO",
    "NS",
    "HOUR_NS",
    "to_ns",
    "from_ns",
    "iso_utc",
    "chicago_label",
    "chicago_date",
    "chicago_local_ns",
    "month_grant_times",
    "ProcessingClockPolicy",
    "ELAPSED_48_HOURS",
    "TWO_BUSINESS_DAYS_FED_1600",
    "FED_HOLIDAYS_2025_2027",
    "processing_due_ns",
]

CHICAGO = ZoneInfo("America/Chicago")
NS = 1_000_000_000
HOUR_NS = 3600 * NS


def to_ns(value: str | datetime) -> int:
    """UTC nanoseconds from an ISO string/aware datetime (exact, no float)."""

    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip().replace(" ", "T", 1)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        frac_ns = 0
        # keep nanosecond precision that datetime.fromisoformat would drop
        if "." in text:
            head, rest = text.split(".", 1)
            digits = ""
            for char in rest:
                if char.isdigit():
                    digits += char
                else:
                    break
            tail = rest[len(digits):]
            frac_ns = int((digits + "000000000")[:9])
            text = head + tail
        stamp = datetime.fromisoformat(text)
        if stamp.tzinfo is None:
            raise ValueError(f"timestamp without a timezone: {value!r}")
        base = int(stamp.astimezone(UTC).timestamp()) * NS
        return base + frac_ns
    if stamp.tzinfo is None:
        raise ValueError("naive datetime")
    stamp = stamp.astimezone(UTC)
    return int(stamp.replace(microsecond=0).timestamp()) * NS + stamp.microsecond * 1000


def from_ns(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns // NS, tz=UTC) + timedelta(
        microseconds=(ts_ns % NS) // 1000
    )


def iso_utc(ts_ns: int) -> str:
    whole = datetime.fromtimestamp(ts_ns // NS, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    frac = ts_ns % NS
    return f"{whole}.{frac:09d}Z" if frac else f"{whole}Z"


def chicago_label(ts_ns: int, *, seconds: bool = False) -> str:
    """Owner-facing Chicago time: 'March 06, 2026 04:00 PM CST'."""

    local = from_ns(ts_ns).astimezone(CHICAGO)
    fmt = "%B %d, %Y %I:%M:%S %p %Z" if seconds else "%B %d, %Y %I:%M %p %Z"
    return local.strftime(fmt)


def chicago_date(ts_ns: int) -> date:
    return from_ns(ts_ns).astimezone(CHICAGO).date()


def chicago_local_ns(day: date, clock: time) -> int:
    return to_ns(datetime.combine(day, clock, tzinfo=CHICAGO))


def month_grant_times(start_ns: int, end_ns: int) -> list[tuple[str, int]]:
    """(grant_id 'YYYY-MM', first-of-month Chicago midnight) strictly after start.

    The start month's grant is the initial five-credit grant, issued at the
    simulation start; later months grant at 12:00 AM Chicago on day one.
    """

    out: list[tuple[str, int]] = []
    first = chicago_date(start_ns).replace(day=1)
    year, month = first.year, first.month
    while True:
        month += 1
        if month == 13:
            year, month = year + 1, 1
        at = chicago_local_ns(date(year, month, 1), time(0, 0))
        if at > end_ns:
            return out
        if at > start_ns:
            out.append((f"{year:04d}-{month:02d}", at))


#: US Federal Reserve bank holidays (the owner-approved processing-holiday
#: calendar, September 22, 2026). Saturday holidays are NOT moved to Friday
#: (Federal Reserve practice); Sunday holidays move to Monday.
FED_HOLIDAYS_2025_2027: tuple[str, ...] = (
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19",
    "2026-09-07", "2026-10-12", "2026-11-11", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-05-31", "2027-07-05",
    "2027-09-06", "2027-10-11", "2027-11-11", "2027-11-25",
)


class ProcessingClockPolicy(FrozenContract):
    policy_id: str
    basis: Literal["elapsed_48_hours", "two_business_days"]
    #: business-day mode only: local payment clock time (HH:MM, Chicago)
    payment_time_chicago: str | None
    holiday_calendar_id: str | None
    holidays: tuple[str, ...]
    owner_selected: bool
    description: str


ELAPSED_48_HOURS = ProcessingClockPolicy(
    policy_id="payout_processing_elapsed_48h_v1",
    basis="elapsed_48_hours",
    payment_time_chicago=None,
    holiday_calendar_id=None,
    holidays=(),
    owner_selected=False,
    description="48 elapsed hours from the end-of-day request, weekends included.",
)

TWO_BUSINESS_DAYS_FED_1600 = ProcessingClockPolicy(
    policy_id="payout_processing_two_business_days_fed_1600_v1",
    basis="two_business_days",
    payment_time_chicago="16:00",
    holiday_calendar_id="us_federal_reserve_bank_holidays_2025_2027_v1",
    holidays=FED_HOLIDAYS_2025_2027,
    owner_selected=True,
    description=(
        "Second business day after the request date (the request date is not day "
        "one), skipping weekends and US Federal Reserve bank holidays; paid at "
        "4:00 PM Chicago. Owner-selected on September 22, 2026."
    ),
)


def processing_due_ns(request_ns: int, policy: ProcessingClockPolicy) -> int:
    if policy.basis == "elapsed_48_hours":
        return request_ns + 48 * HOUR_NS
    if policy.payment_time_chicago is None:
        raise ValueError("business-day processing requires an explicit payment time")
    holidays = {date.fromisoformat(day) for day in policy.holidays}
    cursor = chicago_date(request_ns)
    counted = 0
    while counted < 2:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5 and cursor not in holidays:
            counted += 1
    hour, minute = (int(part) for part in policy.payment_time_chicago.split(":"))
    if cursor.year > 2027 or cursor.year < 2025:
        raise ValueError(
            f"processing date {cursor} is outside the frozen holiday calendar"
        )
    return chicago_local_ns(cursor, time(hour, minute))
