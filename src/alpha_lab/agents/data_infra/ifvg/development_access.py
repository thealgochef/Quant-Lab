"""Fail-closed source access for the IFVG development preparation chain.

The policy enumerates calendar dates in memory and authorizes a date before a
path is constructed.  It never lists the source parent directory.  Audit
counters are derived from append-only access events, including separate label
and context row consumption.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .data_access import DataAccessAudit, ExplorationDataPolicy

__all__ = [
    "DEVELOPMENT_CUTOFF_UTC",
    "FROZEN_WARMUP_DATES",
    "PRIOR_RESEARCH_DATES",
    "EXPOSED_DEVELOPMENT_DATES",
    "PERMITTED_DEVELOPMENT_DATES",
    "SourceDateClass",
    "AccessOperation",
    "DevelopmentAccessEvent",
    "DevelopmentAccessAudit",
    "DevelopmentDataAccess",
    "DevelopmentReplayPolicy",
    "VerificationReplayPolicy",
    "calendar_dates",
]

DEVELOPMENT_CUTOFF_UTC = "2026-06-10T21:00:00Z"
FROZEN_WARMUP_DATES = (
    "2026-01-01",
    "2026-01-02",
    "2026-01-04",
    "2026-01-05",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
    "2026-01-09",
    "2026-01-11",
    "2026-01-12",
)


def calendar_dates(first: str, last: str) -> tuple[str, ...]:
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    if end < start:
        raise ValueError("calendar date range is reversed")
    return tuple(
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    )


PRIOR_RESEARCH_DATES = calendar_dates("2026-01-13", "2026-04-30")
EXPOSED_DEVELOPMENT_DATES = calendar_dates("2026-05-01", "2026-06-10")
PERMITTED_DEVELOPMENT_DATES = (
    *FROZEN_WARMUP_DATES,
    *PRIOR_RESEARCH_DATES,
    *EXPOSED_DEVELOPMENT_DATES,
)


class SourceDateClass(StrEnum):
    WARMUP = "warmup"
    PRIOR_RESEARCH = "prior_research"
    EXPOSED_DEVELOPMENT = "exposed_development_only"
    PROTECTED_BUFFER = "protected_buffer"
    SEALED = "sealed"
    OUTSIDE_POLICY = "outside_policy"


class AccessOperation(StrEnum):
    PATH_CONSTRUCTION = "path_construction"
    METADATA = "metadata"
    EXISTENCE = "existence"
    OPEN = "open"
    ROW = "row"
    LABEL_BAR = "label_bar"
    CONTEXT_ROW = "context_row"
    JOB_DISCOVERY = "job_discovery"


def _classify(day: str) -> SourceDateClass:
    parsed = date.fromisoformat(day)
    if day in FROZEN_WARMUP_DATES:
        return SourceDateClass.WARMUP
    if date(2026, 1, 13) <= parsed <= date(2026, 4, 30):
        return SourceDateClass.PRIOR_RESEARCH
    if date(2026, 5, 1) <= parsed <= date(2026, 6, 10):
        return SourceDateClass.EXPOSED_DEVELOPMENT
    if parsed == date(2026, 6, 11):
        return SourceDateClass.PROTECTED_BUFFER
    if parsed >= date(2026, 6, 12):
        return SourceDateClass.SEALED
    return SourceDateClass.OUTSIDE_POLICY


@dataclass(frozen=True, slots=True)
class DevelopmentAccessEvent:
    sequence: int
    operation: AccessOperation
    source_date: str
    source_class: SourceDateClass
    units: int = 1


@dataclass(slots=True)
class DevelopmentAccessAudit:
    events: list[DevelopmentAccessEvent] = field(default_factory=list)
    denied_dates: dict[str, int] = field(default_factory=dict)

    def record(self, operation: AccessOperation, day: str, *, units: int = 1) -> None:
        if units < 0:
            raise ValueError("audit units cannot be negative")
        self.events.append(
            DevelopmentAccessEvent(
                sequence=len(self.events),
                operation=operation,
                source_date=day,
                source_class=_classify(day),
                units=units,
            )
        )

    def deny(self, day: str) -> None:
        self.denied_dates[day] = self.denied_dates.get(day, 0) + 1

    def _counts(self) -> dict[str, dict[str, int]]:
        result = {
            source_class.value: {operation.value: 0 for operation in AccessOperation}
            for source_class in SourceDateClass
        }
        for event in self.events:
            result[event.source_class.value][event.operation.value] += event.units
        return result

    def as_dict(self) -> dict[str, Any]:
        counts = self._counts()
        protected = {
            source_class.value: counts[source_class.value]
            for source_class in (
                SourceDateClass.PROTECTED_BUFFER,
                SourceDateClass.SEALED,
            )
        }
        event_chain = "\n".join(
            f"{event.sequence}:{event.operation.value}:{event.source_date}:"
            f"{event.source_class.value}:{event.units}"
            for event in self.events
        )
        return {
            "policy": "development_explicit_dates_before_path_v2",
            "cutoff_utc": DEVELOPMENT_CUTOFF_UTC,
            "authorized_date_sha256": hashlib.sha256(
                "\n".join(PERMITTED_DEVELOPMENT_DATES).encode("utf-8")
            ).hexdigest(),
            "event_count": len(self.events),
            "event_chain_sha256": hashlib.sha256(event_chain.encode("utf-8")).hexdigest(),
            "counts_by_source_class": counts,
            "protected_counters": protected,
            "denied_dates": dict(sorted(self.denied_dates.items())),
        }

    def assert_zero_protected(self) -> None:
        counts = self._counts()
        violations = {
            source_class.value: {
                operation: amount
                for operation, amount in counts[source_class.value].items()
                if amount
            }
            for source_class in (
                SourceDateClass.PROTECTED_BUFFER,
                SourceDateClass.SEALED,
            )
        }
        violations = {key: value for key, value in violations.items() if value}
        if violations:
            raise AssertionError(f"protected IFVG source access occurred: {violations}")


class DevelopmentDataAccess:
    """Authorize exact development dates and perform only date-addressed I/O."""

    def __init__(
        self,
        *,
        allowed_dates: Iterable[str] = PERMITTED_DEVELOPMENT_DATES,
        audit: DevelopmentAccessAudit | None = None,
    ) -> None:
        ordered = tuple(str(day) for day in allowed_dates)
        if ordered != tuple(sorted(ordered)) or len(ordered) != len(set(ordered)):
            raise ValueError("allowed dates must be unique and chronological")
        protected = [day for day in ordered if _classify(day) in {
            SourceDateClass.PROTECTED_BUFFER,
            SourceDateClass.SEALED,
        }]
        if protected:
            raise PermissionError("protected dates cannot enter the development allowlist")
        self.allowed_dates = ordered
        self._allowed = frozenset(ordered)
        self.audit = audit or DevelopmentAccessAudit()

    def authorize(self, day: str) -> None:
        if day not in self._allowed:
            self.audit.deny(day)
            raise PermissionError("source date is outside the IFVG development contract")

    def construct_path(self, day: str, factory: Callable[[str], Path]) -> Path:
        self.authorize(day)
        path = Path(factory(day))
        self.audit.record(AccessOperation.PATH_CONSTRUCTION, day)
        return path

    def exists(self, day: str, factory: Callable[[str], Path]) -> bool:
        path = self.construct_path(day, factory)
        self.audit.record(AccessOperation.EXISTENCE, day)
        return path.exists()

    def parquet_metadata(self, day: str, factory: Callable[[str], Path]) -> Any:
        path = self.construct_path(day, factory)
        self.audit.record(AccessOperation.METADATA, day)
        return pq.read_metadata(path)

    def read_parquet(
        self,
        day: str,
        factory: Callable[[str], Path],
        **kwargs: Any,
    ) -> pd.DataFrame:
        path = self.construct_path(day, factory)
        self.audit.record(AccessOperation.OPEN, day)
        frame = pd.read_parquet(path, **kwargs)
        self.audit.record(AccessOperation.ROW, day, units=len(frame))
        return frame

    def record_label_bars(self, day: str, count: int) -> None:
        self.authorize(day)
        self.audit.record(AccessOperation.LABEL_BAR, day, units=count)

    def record_context_rows(self, day: str, count: int) -> None:
        self.authorize(day)
        self.audit.record(AccessOperation.CONTEXT_ROW, day, units=count)

    def record_job_discovery(self, day: str) -> None:
        self.authorize(day)
        self.audit.record(AccessOperation.JOB_DISCOVERY, day)

    def existing_paths(
        self,
        factory: Callable[[str], Path],
        *,
        dates: Iterable[str] | None = None,
    ) -> Iterator[tuple[str, Path]]:
        # This loop is the only discovery mechanism: no parent-directory list.
        requested = self.allowed_dates if dates is None else tuple(dates)
        for day in requested:
            self.authorize(day)
            path = self.construct_path(day, factory)
            self.audit.record(AccessOperation.EXISTENCE, day)
            if path.exists():
                yield day, path

    def assert_safe(self) -> None:
        self.audit.assert_zero_protected()


class DevelopmentReplayPolicy(ExplorationDataPolicy):
    """Compatibility adapter for the existing sequential artifact driver.

    Construction is limited to dates first discovered by
    :class:`DevelopmentDataAccess`; callers cannot add June 11 or sealed dates.
    """

    _ifvg_development_policy_v2 = True
    policy_id = "development_explicit_dates_before_path_v2"

    def __init__(
        self,
        replay_dates: Iterable[str],
        *,
        development_audit: DevelopmentAccessAudit | None = None,
    ) -> None:
        ordered = tuple(str(day) for day in replay_dates)
        if ordered != tuple(sorted(ordered)) or len(ordered) != len(set(ordered)):
            raise ValueError("replay dates must be unique and chronological")
        if tuple(day for day in ordered if day < "2026-01-13") != FROZEN_WARMUP_DATES:
            raise ValueError("development replay requires the frozen ten-date warmup")
        if any(day not in PERMITTED_DEVELOPMENT_DATES for day in ordered):
            raise PermissionError("development replay contains a non-permitted date")
        super().__init__(audit=DataAccessAudit(), allowlist=frozenset(ordered))
        self.development_audit = development_audit or DevelopmentAccessAudit()

    def resolve_source_path(
        self,
        day: str,
        path_factory: Callable[[str], Path],
    ) -> Path:
        path = super().resolve_source_path(day, path_factory)
        self.development_audit.record(AccessOperation.PATH_CONSTRUCTION, day)
        return path

    def record_metadata_access(self, day: str) -> None:
        super().record_metadata_access(day)
        self.development_audit.record(AccessOperation.METADATA, day)

    def record_file_open(self, day: str, *, rows: int = 0) -> None:
        # Reimplement instead of calling the base method, whose row callback is
        # virtual and would double-record in the extended audit.
        self.authorize_date(day)
        self.audit.file_opens += 1
        self.audit.file_opens_by_date[day] = self.audit.file_opens_by_date.get(day, 0) + 1
        self.audit.rows_read += int(rows)
        self.audit.rows_read_by_date[day] = (
            self.audit.rows_read_by_date.get(day, 0) + int(rows)
        )
        self.development_audit.record(AccessOperation.OPEN, day)
        if rows:
            self.development_audit.record(AccessOperation.ROW, day, units=rows)

    def record_rows_read(self, day: str, *, rows: int) -> None:
        super().record_rows_read(day, rows=rows)
        self.development_audit.record(AccessOperation.ROW, day, units=rows)

    def assert_zero_forbidden_access(self) -> None:
        super().assert_zero_forbidden_access()
        self.development_audit.assert_zero_protected()

    def audit_dict(self) -> dict[str, Any]:
        return self.development_audit.as_dict()


class VerificationReplayPolicy(DevelopmentReplayPolicy):
    """The third trusted policy class: one frozen ≤5-day verification fixture.

    Consumed by the real baseline vertical slice only
    (``ifvg_prop_robust_config_search_v1`` §6 / TEST_MATRIX §1 Path A).
    Compared to :class:`DevelopmentReplayPolicy` it authorizes at most five
    real trading days, requires NO real warmup prefix (the slice starts from a
    verified profile-matching seed snapshot instead), and still fails closed
    before path construction for any protected, sealed, or off-allowlist date.
    A sixth date, an off-allowlist date, or a rotated window is refused at
    construction — before any source path exists.
    """

    _ifvg_development_policy_v2 = False
    _ifvg_verification_policy_v1 = True
    policy_id = "verification_fixed_allowlist_max5_v1"

    def __init__(
        self,
        replay_dates: Iterable[str],
        *,
        development_audit: DevelopmentAccessAudit | None = None,
    ) -> None:
        ordered = tuple(str(day) for day in replay_dates)
        if not ordered:
            raise ValueError("the verification allowlist cannot be empty")
        if ordered != tuple(sorted(ordered)) or len(ordered) != len(set(ordered)):
            raise ValueError("verification dates must be unique and chronological")
        if len(ordered) > 5:
            raise PermissionError(
                "the verification fixture admits at most five real trading days"
            )
        forbidden = [
            day
            for day in ordered
            if _classify(day)
            not in {
                SourceDateClass.WARMUP,
                SourceDateClass.PRIOR_RESEARCH,
                SourceDateClass.EXPOSED_DEVELOPMENT,
            }
        ]
        if forbidden:
            raise PermissionError(
                "verification dates must lie inside the permitted development "
                f"window; refused before path construction: {forbidden}"
            )
        # Deliberately bypass DevelopmentReplayPolicy.__init__ (which demands
        # the frozen ten-date warmup prefix): the slice warm-starts from a
        # verified seed snapshot, never from real warmup replay days.
        ExplorationDataPolicy.__init__(
            self, audit=DataAccessAudit(), allowlist=frozenset(ordered)
        )
        self.allowed_dates = ordered
        self.development_audit = development_audit or DevelopmentAccessAudit()

    def audit_dict(self) -> dict[str, Any]:
        payload = self.development_audit.as_dict()
        payload["policy"] = self.policy_id
        payload["authorized_date_sha256"] = hashlib.sha256(
            "\n".join(self.allowed_dates).encode("utf-8")
        ).hexdigest()
        return payload
