"""Pre-I/O source allowlist for the IFVG v2 exploration repair."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "EXPLORATION_DATE_ALLOWLIST",
    "EXPOSED_VALIDATION_START",
    "EXPOSED_VALIDATION_END",
    "SEALED_START",
    "DataAccessAudit",
    "ExplorationDataPolicy",
    "allowlist_sha256",
    "require_fixed_exploration_allowlist",
    "discover_allowlisted_source_files",
    "hash_allowlisted_source_files",
]

EXPOSED_VALIDATION_START = "2026-05-01"
EXPOSED_VALIDATION_END = "2026-06-10"
SEALED_START = "2026-06-12"


def _days() -> frozenset[str]:
    return frozenset(
        {
            "2026-01-01",
            "2026-01-02",
            *{f"2026-01-{day:02d}" for day in range(4, 10)},
            *{f"2026-01-{day:02d}" for day in range(11, 17)},
            *{f"2026-01-{day:02d}" for day in range(18, 24)},
            *{f"2026-01-{day:02d}" for day in range(25, 31)},
        }
    )


EXPLORATION_DATE_ALLOWLIST = _days()


def allowlist_sha256(days: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(day) for day in days))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bump(mapping: dict[str, int], day: str, amount: int = 1) -> None:
    mapping[day] = mapping.get(day, 0) + amount


@dataclass
class DataAccessAudit:
    path_constructions: int = 0
    metadata_accesses: int = 0
    file_opens: int = 0
    rows_read: int = 0
    denied_dates: dict[str, int] = field(default_factory=dict)
    path_constructions_by_date: dict[str, int] = field(default_factory=dict)
    metadata_accesses_by_date: dict[str, int] = field(default_factory=dict)
    file_opens_by_date: dict[str, int] = field(default_factory=dict)
    rows_read_by_date: dict[str, int] = field(default_factory=dict)

    @property
    def opened_dates(self) -> dict[str, int]:
        """Compatibility alias used by older diagnostics."""
        return self.file_opens_by_date

    def as_dict(
        self,
        *,
        allowlist: Iterable[str] = EXPLORATION_DATE_ALLOWLIST,
    ) -> dict:
        allowed = tuple(sorted(str(day) for day in allowlist))
        return {
            "policy": "explicit_allowlist_before_path_v1",
            "allowlist": list(allowed),
            "allowlist_sha256": allowlist_sha256(allowed),
            "protected_ranges": {
                "exposed_validation": [
                    EXPOSED_VALIDATION_START,
                    EXPOSED_VALIDATION_END,
                ],
                "june_11": ["2026-06-11", "2026-06-11"],
                "sealed": [SEALED_START, None],
            },
            "protected_path_constructions": 0,
            "protected_metadata_accesses": 0,
            "protected_file_opens": 0,
            "protected_rows_read": 0,
            "path_constructions": self.path_constructions,
            "metadata_accesses": self.metadata_accesses,
            "file_opens": self.file_opens,
            "rows_read": self.rows_read,
            "denied_dates": dict(sorted(self.denied_dates.items())),
            "path_constructions_by_date": dict(
                sorted(self.path_constructions_by_date.items())
            ),
            "metadata_accesses_by_date": dict(
                sorted(self.metadata_accesses_by_date.items())
            ),
            "file_opens_by_date": dict(sorted(self.file_opens_by_date.items())),
            "rows_read_by_date": dict(sorted(self.rows_read_by_date.items())),
        }


class ExplorationDataPolicy:
    """Authorize a date before path construction, metadata, or file access."""

    def __init__(
        self,
        *,
        audit: DataAccessAudit | None = None,
        allowlist: frozenset[str] = EXPLORATION_DATE_ALLOWLIST,
    ) -> None:
        self.audit = audit or DataAccessAudit()
        self.allowlist = frozenset(allowlist)

    def authorize_date(self, day: str) -> None:
        if day not in self.allowlist:
            _bump(self.audit.denied_dates, day)
            raise PermissionError(
                f"IFVG v2 exploration source date {day} is not allowlisted"
            )

    def audit_dict(self) -> dict:
        return self.audit.as_dict(allowlist=self.allowlist)

    def authorize_dates(self, days: Iterable[str]) -> tuple[str, ...]:
        """Validate a requested chain without constructing any source path."""
        result = tuple(days)
        for day in result:
            self.authorize_date(day)
        if len(result) != len(set(result)):
            raise ValueError("IFVG v2 source dates must be unique")
        if result != tuple(sorted(result)):
            raise ValueError("IFVG v2 source dates must be chronological")
        return result

    def resolve_source_path(
        self,
        day: str,
        path_factory: Callable[[str], Path],
    ) -> Path:
        self.authorize_date(day)
        path = Path(path_factory(day))
        self.audit.path_constructions += 1
        _bump(self.audit.path_constructions_by_date, day)
        return path

    def record_metadata_access(self, day: str) -> None:
        self.authorize_date(day)
        self.audit.metadata_accesses += 1
        _bump(self.audit.metadata_accesses_by_date, day)

    def record_file_open(self, day: str, *, rows: int = 0) -> None:
        self.authorize_date(day)
        self.audit.file_opens += 1
        _bump(self.audit.file_opens_by_date, day)
        self.record_rows_read(day, rows=rows)

    def record_rows_read(self, day: str, *, rows: int) -> None:
        self.authorize_date(day)
        self.audit.rows_read += int(rows)
        _bump(self.audit.rows_read_by_date, day, int(rows))

    def read_metadata(
        self,
        day: str,
        path_factory: Callable[[str], Path],
        reader: Callable[[Path], object],
    ) -> object:
        path = self.resolve_source_path(day, path_factory)
        self.record_metadata_access(day)
        return reader(path)

    def read_parquet(
        self,
        day: str,
        path_factory: Callable[[str], Path],
        **kwargs,
    ) -> pd.DataFrame:
        path = self.resolve_source_path(day, path_factory)
        self.authorize_date(day)
        self.record_file_open(day)
        frame = pd.read_parquet(path, **kwargs)
        self.record_rows_read(day, rows=len(frame))
        return frame

    def assert_zero_forbidden_access(self) -> None:
        touched = (
            set(self.audit.path_constructions_by_date)
            | set(self.audit.metadata_accesses_by_date)
            | set(self.audit.file_opens_by_date)
            | set(self.audit.rows_read_by_date)
        )
        forbidden = sorted(touched - self.allowlist)
        if forbidden:
            raise AssertionError(
                "forbidden IFVG source dates reached path/metadata/file access: "
                f"{forbidden}"
            )

    def assert_only_allowlisted_rows(self) -> None:
        self.assert_zero_forbidden_access()
        if set(self.audit.rows_read_by_date) - self.allowlist:
            raise AssertionError("rows from a forbidden IFVG date were read")


def require_fixed_exploration_allowlist(
    policy: ExplorationDataPolicy,
) -> None:
    """Prevent a caller from broadening the repair replay's authorized scope."""
    fixed_repair = policy.allowlist == EXPLORATION_DATE_ALLOWLIST
    trusted_development = bool(
        getattr(policy, "_ifvg_development_policy_v2", False)
    ) and all(
        "2026-01-01" <= day <= "2026-06-10" and day != "2026-06-11"
        for day in policy.allowlist
    )
    # Third trusted class (ifvg_prop_robust_config_search_v1): the frozen
    # ≤5-day verification fixture. Same hard window, plus the day-count cap
    # and a non-empty bound (an empty allowlist can never be "trusted").
    trusted_verification = (
        bool(getattr(policy, "_ifvg_verification_policy_v1", False))
        and 1 <= len(policy.allowlist) <= 5
        and all(
            "2026-01-01" <= day <= "2026-06-10" and day != "2026-06-11"
            for day in policy.allowlist
        )
    )
    if not fixed_repair and not trusted_development and not trusted_verification:
        raise PermissionError(
            "IFVG v2 repair replay requires the exact fixed exploration allowlist"
        )


def discover_allowlisted_source_files(
    policy: ExplorationDataPolicy,
    *,
    data_dir: Path,
    symbol: str,
    dates: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Resolve at most one priority source file for each authorized date."""
    from strategy_core.data.databento_parquet import DAY_FILE_PRIORITY

    requested = tuple(sorted(dates or policy.allowlist))
    policy.authorize_dates(requested)
    result: dict[str, Path] = {}
    for day in requested:
        folder = policy.resolve_source_path(
            day,
            lambda value: Path(data_dir) / symbol / value,
        )
        policy.record_metadata_access(day)
        for filename, _schema in DAY_FILE_PRIORITY:
            candidate = folder / filename
            if candidate.exists():
                result[day] = candidate
                break
    return result


def hash_allowlisted_source_files(
    policy: ExplorationDataPolicy,
    files: dict[str, Path],
) -> tuple[tuple[str, str], ...]:
    """SHA-256 exact permitted inputs, with every file open audited."""
    hashed: list[tuple[str, str]] = []
    for day, path in sorted(files.items()):
        policy.authorize_date(day)
        policy.record_file_open(day)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashed.append((f"{day}/{path.name}", digest.hexdigest()))
    return tuple(hashed)
