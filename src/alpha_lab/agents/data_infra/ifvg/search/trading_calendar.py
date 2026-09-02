"""Logical trading-day semantics for the verification program (§5.1; F-22).

**The date-domain finding.** The R1 window scan scored "consecutive
store-day" windows over the accepted dataset's physical partition dates —
every UTC calendar date that carries a Databento day file. That set holds
every weekday AND every Sunday (the Sunday file holds the Sunday 18:00 ET
Globex open that belongs to MONDAY's trading day), so the February
candidate ``2026-02-06 … 02-11`` names ``2026-02-08``, a Sunday: as a
trading-day id its stream would be ``[Sat 18:00 ET, Sun 18:00 ET)`` — no
market at all. Physical partition dates are NOT trading-day ids.

**Corrected semantics.** A logical trading day is the Strategy-Core
trading-day id ``td`` whose canonical stream is
``[td−1 18:00 ET, td 18:00 ET)`` (``strategy_core.constants.SESSION_TIMEZONE``
/ ``TRADING_DAY_BOUNDARY``); ``DatabentoParquetSource.for_trading_day``
composes it from the two physical UTC-date partitions ``td−1`` and ``td``.
Under the registered policy ``cme_globex_18et_weekday_v1`` a logical
trading day is a weekday (Mon–Fri) that is not a registered FULL-closure
day; partial-session holidays are trading days. The closures inside the
permitted window were verified against the accepted dataset's own evidence
(the FSM-audit day funnel covers all 114 weekdays 2026-01-02 … 06-10 —
MLK, Presidents' Day, Good Friday and Memorial Day included — and does NOT
cover 2026-01-01, which emitted nothing), so the only registered closure is
New Year's Day. Good Friday 2026-04-03 carried a full session (9 lifecycle
rows, 1 candidate, 1 decision, 1 executed trade): the evidence contradicts
a closure and the policy follows the evidence.

**Seed chains follow store days.** The accepted chain replayed every
physical store day (weekdays + Sundays, from the cold start 2026-01-01),
so a profile-matching seed must be produced over the same store-day
sequence (:func:`store_day_chain`); the verification ALLOWLIST is the
logical-day sequence. Both are exposed here so no caller mixes the domains.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from strategy_core.constants import SESSION_TIMEZONE, TRADING_DAY_BOUNDARY

from .identities import SHA256_PATTERN, FrozenContract

__all__ = [
    "TRADING_CALENDAR_POLICY_ID",
    "PERMITTED_WINDOW_FIRST_DAY",
    "PERMITTED_WINDOW_LAST_DAY",
    "PROTECTED_BUFFER_DAY",
    "SEALED_START_DAY",
    "CANONICAL_CHAIN_START_DAY",
    "REGISTERED_FULL_CLOSURES",
    "EVIDENCE_VERIFIED_SESSIONS",
    "TradingCalendarPolicy",
    "CME_GLOBEX_18ET_WEEKDAY_V1",
    "PhysicalPartition",
    "SourcePartitionRef",
    "VerificationTradingDayRef",
    "is_logical_trading_day",
    "logical_trading_days",
    "next_logical_trading_day",
    "previous_logical_trading_day",
    "assert_consecutive_logical_days",
    "consecutive_logical_windows",
    "physical_partitions_for",
    "session_bounds_utc",
    "store_day_chain",
    "trading_day_ref_from_inventory",
    "inventory_from_permitted_source_hashes",
    "source_kind_of_partition_file",
]

TRADING_CALENDAR_POLICY_ID = "cme_globex_18et_weekday_v1"
PERMITTED_WINDOW_FIRST_DAY = "2026-01-01"
PERMITTED_WINDOW_LAST_DAY = "2026-06-10"
PROTECTED_BUFFER_DAY = "2026-06-11"
SEALED_START_DAY = "2026-06-12"
#: The first store day of the accepted chain — the cold start every
#: profile-matching seed chain must reproduce.
CANONICAL_CHAIN_START_DAY = "2026-01-01"
MAX_VERIFICATION_WINDOW_DAYS = 5

#: Registered FULL closures inside the permitted window, each with its
#: evidence (the accepted dataset's typed tables + the FSM-audit day funnel).
REGISTERED_FULL_CLOSURES: tuple[tuple[str, str], ...] = (
    (
        "2026-01-01",
        "New Year's Day (CME Globex equity index futures closed): absent from the "
        "FSM-audit day funnel (no bars, no HTF taps) and zero emissions in every "
        "typed table of the accepted dataset; also the chain's cold-start day",
    ),
)
#: Holidays inside the window whose SESSION is proven by the evidence (they
#: are trading days; partial hours do not change the trading-day id).
EVIDENCE_VERIFIED_SESSIONS: Mapping[str, str] = {
    "2026-01-19": "Martin Luther King Jr. Day — funnel htf_taps=18 (partial session)",
    "2026-02-16": "Presidents' Day — funnel htf_taps=72, setups_born=1 (partial session)",
    "2026-04-03": (
        "Good Friday — 9 lifecycle rows, 1 candidate, 1 decision, 1 executed trade, "
        "funnel htf_taps=78: a full session; NOT a closure (evidence over assumption)"
    ),
    "2026-05-25": "Memorial Day — funnel htf_taps=19, parent_candidates=96 (partial session)",
}

SourceKind = Literal["mbp1", "mbp10", "trades"]
PartitionKey = Literal["prev_utc_date", "utc_date"]


class TradingCalendarPolicy(FrozenContract):
    policy_id: Literal["cme_globex_18et_weekday_v1"] = TRADING_CALENDAR_POLICY_ID
    weekday_only: Literal[True] = True
    full_closure_days: tuple[str, ...]
    session_timezone: str = SESSION_TIMEZONE
    boundary_local_time: str = TRADING_DAY_BOUNDARY.isoformat(timespec="minutes")

    @model_validator(mode="after")
    def _closures_are_dates(self):
        for day in self.full_closure_days:
            date.fromisoformat(day)
        if tuple(sorted(set(self.full_closure_days))) != self.full_closure_days:
            raise ValueError("full_closure_days must be unique and chronological")
        return self


CME_GLOBEX_18ET_WEEKDAY_V1 = TradingCalendarPolicy(
    full_closure_days=tuple(day for day, _reason in REGISTERED_FULL_CLOSURES)
)


def _day(value: str) -> date:
    return date.fromisoformat(str(value))


def is_logical_trading_day(
    day: str, *, policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1
) -> bool:
    parsed = _day(day)
    return parsed.weekday() < 5 and day not in policy.full_closure_days


def logical_trading_days(
    start: str, end: str, *, policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1
) -> tuple[str, ...]:
    """Every logical trading day in ``[start, end]`` (inclusive)."""

    first, last = _day(start), _day(end)
    if last < first:
        raise ValueError("the calendar range is reversed")
    out: list[str] = []
    cursor = first
    while cursor <= last:
        text = cursor.isoformat()
        if is_logical_trading_day(text, policy=policy):
            out.append(text)
        cursor += timedelta(days=1)
    return tuple(out)


def next_logical_trading_day(
    day: str, *, policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1
) -> str:
    cursor = _day(day) + timedelta(days=1)
    for _ in range(31):
        text = cursor.isoformat()
        if is_logical_trading_day(text, policy=policy):
            return text
        cursor += timedelta(days=1)
    raise ValueError(f"no logical trading day within 31 days after {day}")  # pragma: no cover


def previous_logical_trading_day(
    day: str, *, policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1
) -> str:
    cursor = _day(day) - timedelta(days=1)
    for _ in range(31):
        text = cursor.isoformat()
        if is_logical_trading_day(text, policy=policy):
            return text
        cursor -= timedelta(days=1)
    raise ValueError(f"no logical trading day within 31 days before {day}")  # pragma: no cover


def assert_consecutive_logical_days(
    days: Iterable[str],
    *,
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
    max_days: int | None = None,
) -> tuple[str, ...]:
    """Refuse anything but a non-empty run of CONSECUTIVE logical trading
    days (a physical partition date, a gap, or a reordered day is refused)."""

    ordered = tuple(str(day) for day in days)
    if not ordered:
        raise ValueError("a logical-day window cannot be empty")
    if max_days is not None and len(ordered) > max_days:
        raise ValueError(f"a logical-day window admits at most {max_days} days")
    for day in ordered:
        if not is_logical_trading_day(day, policy=policy):
            raise ValueError(
                f"{day} is not a logical trading day under {policy.policy_id} (a physical "
                "partition date or a registered closure is not a trading-day id)"
            )
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if next_logical_trading_day(previous, policy=policy) != current:
            raise ValueError(f"{previous} → {current} are not consecutive logical trading days")
    return ordered


def consecutive_logical_windows(
    days: Iterable[str],
    length: int,
    *,
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
) -> tuple[tuple[str, ...], ...]:
    """Every ``length``-day window of consecutive logical trading days
    inside the ordered sequence ``days`` (a verification window is 1–5 days)."""

    if not 1 <= int(length) <= MAX_VERIFICATION_WINDOW_DAYS:
        raise ValueError(
            f"a verification window admits at most five days (1–{MAX_VERIFICATION_WINDOW_DAYS})"
        )
    ordered = tuple(str(day) for day in days)
    windows: list[tuple[str, ...]] = []
    for start in range(len(ordered) - int(length) + 1):
        candidate = ordered[start : start + int(length)]
        try:
            assert_consecutive_logical_days(candidate, policy=policy)
        except ValueError:
            continue
        windows.append(candidate)
    return tuple(windows)


@dataclass(frozen=True)
class PhysicalPartition:
    physical_utc_date: str
    relative_logical_partition_key: str


def physical_partitions_for(td: str) -> tuple[PhysicalPartition, PhysicalPartition]:
    """The ordered physical UTC-date partitions composing logical day ``td``:
    ``td−1`` (the prior evening from the 18:00 ET roll) and ``td``."""

    day = _day(td)
    previous = (day - timedelta(days=1)).isoformat()
    return (
        PhysicalPartition(
            physical_utc_date=previous, relative_logical_partition_key="prev_utc_date"
        ),
        PhysicalPartition(
            physical_utc_date=day.isoformat(), relative_logical_partition_key="utc_date"
        ),
    )


def session_bounds_utc(
    td: str, *, policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1
) -> tuple[str, str]:
    """``[td−1 18:00 ET, td 18:00 ET)`` as UTC ISO instants (DST-aware)."""

    zone = ZoneInfo(policy.session_timezone)
    day = _day(td)
    boundary = TRADING_DAY_BOUNDARY
    open_local = datetime.combine(day - timedelta(days=1), boundary, tzinfo=zone)
    close_local = datetime.combine(day, boundary, tzinfo=zone)
    return (
        open_local.astimezone(UTC).isoformat(),
        close_local.astimezone(UTC).isoformat(),
    )


def store_day_chain(first: str, last: str) -> tuple[str, ...]:
    """The physical STORE-DAY chain ``[first, last]``: every calendar day
    except Saturdays (no Saturday ever carries a day file; the Sunday file
    holds the Sunday 18:00 ET open). This is the sequence the accepted chain
    replayed and the sequence a profile-matching seed chain must reproduce."""

    start, end = _day(first), _day(last)
    if end < start:
        raise ValueError("the store-day chain range is reversed")
    if start.weekday() == 5 or end.weekday() == 5:
        raise ValueError("a store-day chain never starts or ends on a Saturday")
    out: list[str] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() != 5:
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(out)


class SourcePartitionRef(FrozenContract):
    physical_utc_date: str
    relative_logical_partition_key: PartitionKey
    source_kind: SourceKind
    content_sha256: str = Field(pattern=SHA256_PATTERN)


class VerificationTradingDayRef(FrozenContract):
    """One logical trading day mapped to its exact ordered physical
    partitions (content hashes come from the accepted inventory, never from
    a disk read)."""

    logical_trading_day: str
    session_open_ts_utc: str
    session_close_ts_utc: str
    ordered_source_partition_refs: tuple[SourcePartitionRef, ...]

    @model_validator(mode="after")
    def _maps_exactly(self):
        day = self.logical_trading_day
        if not is_logical_trading_day(day):
            raise ValueError(f"{day} is not a logical trading day")
        expected = physical_partitions_for(day)
        refs = self.ordered_source_partition_refs
        if len(refs) != 2 or tuple(
            (ref.physical_utc_date, ref.relative_logical_partition_key) for ref in refs
        ) != tuple((p.physical_utc_date, p.relative_logical_partition_key) for p in expected):
            raise ValueError(
                "ordered_source_partition_refs must be exactly (td−1 prev_utc_date, td utc_date)"
            )
        open_ts, close_ts = session_bounds_utc(day)
        if (self.session_open_ts_utc, self.session_close_ts_utc) != (open_ts, close_ts):
            raise ValueError("session bounds do not match the 18:00 ET trading-day roll")
        return self


def source_kind_of_partition_file(filename: str) -> SourceKind:
    stem = str(filename).split("/")[-1].split(".")[0]
    if stem not in ("mbp1", "mbp10", "trades"):
        raise ValueError(f"unregistered source partition file {filename!r}")
    return stem  # type: ignore[return-value]


def inventory_from_permitted_source_hashes(
    entries: Iterable[tuple[str, str] | list],
) -> dict[str, tuple[str, str]]:
    """``{physical_utc_date: (source_kind, content_sha256)}`` from an accepted
    manifest's ``permitted_source_hashes`` (``["<date>/<file>", sha]``). Two
    files for one date are refused (the accepted inventory has exactly one)."""

    inventory: dict[str, tuple[str, str]] = {}
    for entry in entries:
        name, digest = str(entry[0]), str(entry[1])
        day, _slash, filename = name.partition("/")
        date.fromisoformat(day)
        if day in inventory:
            raise ValueError(f"the inventory carries two partition files for {day}")
        inventory[day] = (source_kind_of_partition_file(filename), digest)
    return inventory


def trading_day_ref_from_inventory(
    td: str,
    inventory: Mapping[str, tuple[str, str]],
    *,
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
) -> VerificationTradingDayRef | None:
    """The exact ref of logical day ``td``, or ``None`` when either physical
    partition is not in the (already-authorized) inventory — the day is then
    not hash-addressable."""

    if not is_logical_trading_day(td, policy=policy):
        raise ValueError(f"{td} is not a logical trading day under {policy.policy_id}")
    refs: list[SourcePartitionRef] = []
    for partition in physical_partitions_for(td):
        entry = inventory.get(partition.physical_utc_date)
        if entry is None:
            return None
        kind, digest = entry
        refs.append(
            SourcePartitionRef(
                physical_utc_date=partition.physical_utc_date,
                relative_logical_partition_key=partition.relative_logical_partition_key,  # type: ignore[arg-type]
                source_kind=kind,  # type: ignore[arg-type]
                content_sha256=digest,
            )
        )
    open_ts, close_ts = session_bounds_utc(td, policy=policy)
    return VerificationTradingDayRef(
        logical_trading_day=td,
        session_open_ts_utc=open_ts,
        session_close_ts_utc=close_ts,
        ordered_source_partition_refs=tuple(refs),
    )
