"""Bounded MenthorQ v02 EOD context for the IFSM MFFU research batch.

This is a separate nominal overnight selection policy. It does not alter the
existing daytime ``menthorq_levels`` lookup or claim measured publication time.
Only report/set identities are examined for rows in a mixed-date CSV before
deciding whether their measurements are within the authorized cutoff.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

POLICY_ID = "mq_eod_asof_nominal_2200_chicago_v01"
MAX_AGE_CALENDAR_DAYS = 7
_CHICAGO = ZoneInfo("America/Chicago")
_UTC = UTC
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_ZIP_ROOT = "MenthorQ_Research_Data_v02/data/canonical/end_of_day/"
_TABLE_MEMBERS = {
    "total_gamma": _ZIP_ROOT + "metrics/total_gamma_by_report_date.csv",
    "level_sets": _ZIP_ROOT + "levels/level_sets.csv",
    "gamma_long": _ZIP_ROOT + "levels/gamma_long.csv",
}
_TABLE_RELATIVE = {
    key: Path(member.removeprefix("MenthorQ_Research_Data_v02/"))
    for key, member in _TABLE_MEMBERS.items()
}


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("MenthorQ decision and cutoff times must be offset-aware")
    return value.astimezone(_UTC)


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _day(raw: str, field: str) -> date:
    if not isinstance(raw, str) or not _DATE_RE.fullmatch(raw):
        raise ValueError(f"{field} must be an ISO date")
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {raw}") from exc


def _number(raw: str | None, field: str) -> float | None:
    if raw is None:
        raise ValueError(f"missing {field} column")
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"nonfinite {field}")
    return value


def _identity(row: Mapping[str, str], *, pointer: bool = True) -> tuple[str, str, str]:
    fields = ("source_path", "source_sha256", "source_json_pointer")
    values = tuple(row.get(name, "") for name in fields)
    if not values[0] or not re.fullmatch(r"[0-9a-f]{64}", values[1]):
        raise ValueError("source path and SHA-256 identity are required")
    if pointer and not values[2]:
        raise ValueError("source JSON pointer is required")
    return values


def nominal_eligible_from(report_day: date) -> datetime:
    """The unverified 10 PM Chicago schedule anchor for a vendor report date."""
    return datetime.combine(report_day, time(22), _CHICAGO).astimezone(_UTC)


def level_set_eligible_from(requested_day: date, report_day: date) -> datetime:
    """Requested-date convention: both report and R-minus-one must have released."""
    return max(
        nominal_eligible_from(report_day),
        nominal_eligible_from(requested_day - timedelta(days=1)),
    )


@dataclass(frozen=True)
class _Gamma:
    report_day: date
    eligible_from: datetime
    value: float | None
    source: tuple[str, str, str]
    positive_run_lower_bound: int
    positive_age_known: bool


@dataclass(frozen=True)
class _Level:
    name: str
    price: float | None
    gex: float | None
    source_pointer: str


@dataclass(frozen=True)
class _LevelSet:
    set_id: str
    requested_day: date
    report_day: date
    eligible_from: datetime
    source: tuple[str, str, str]
    levels: tuple[_Level, ...]


class EodAsOfIndex:
    """Immutable, cutoff-bounded index with independent gamma and level clocks."""

    def __init__(
        self,
        *,
        gamma_rows: Iterable[Mapping[str, str]],
        level_set_rows: Iterable[Mapping[str, str]],
        gamma_long_rows: Iterable[Mapping[str, str]],
        cutoff_utc: datetime,
        bundle_sha256: str,
        table_sha256: Mapping[str, str],
        ticker: str = "NQ1!",
    ) -> None:
        cutoff = _utc(cutoff_utc)
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
            raise ValueError("bundle_sha256 must be a SHA-256 hex digest")
        if set(table_sha256) != set(_TABLE_MEMBERS) or any(
            not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in table_sha256.values()
        ):
            raise ValueError("all three table SHA-256 identities are required")
        if not ticker:
            raise ValueError("ticker is required")
        self.cutoff_utc = cutoff
        self.bundle_sha256 = bundle_sha256
        self.table_sha256 = MappingProxyType(dict(table_sha256))
        self.ticker = ticker

        # Decide scope from the date identity before interpreting gex or other values.
        gamma_input: dict[date, tuple[float | None, tuple[str, str, str]]] = {}
        for row in gamma_rows:
            report_day = _day(row.get("vendor_report_date", ""), "vendor_report_date")
            if nominal_eligible_from(report_day) > cutoff:
                continue
            if row.get("historical_publication_time"):
                raise ValueError("historical gamma publication evidence needs policy review")
            if report_day in gamma_input:
                raise ValueError(f"conflicting/duplicate gamma report: {report_day}")
            gamma_input[report_day] = (_number(row.get("gex"), "gex"), _identity(row))

        gamma: list[_Gamma] = []
        run = 0
        boundary_observed = False
        for report_day, (value, source) in sorted(gamma_input.items()):
            if value is not None and value > 0:
                run += 1
            else:
                run = 0
                boundary_observed = True
            gamma.append(
                _Gamma(
                    report_day,
                    nominal_eligible_from(report_day),
                    value,
                    source,
                    run,
                    boundary_observed,
                )
            )
        self._gamma = tuple(gamma)
        self._gamma_times = tuple(item.eligible_from for item in gamma)

        # The requested date chooses the set; its own report date determines age.
        sets_by_id: dict[str, tuple[date, date, datetime, tuple[str, str, str], int]] = {}
        requested_seen: set[date] = set()
        for row in level_set_rows:
            if (
                row.get("ticker") != ticker
                or row.get("level_type") != "gamma_levels"
                or row.get("kind") != "eod"
            ):
                continue
            requested = _day(row.get("requested_date", ""), "requested_date")
            report = _day(row.get("vendor_report_date", ""), "vendor_report_date")
            eligible = level_set_eligible_from(requested, report)
            if eligible > cutoff:
                continue
            if report > requested:
                raise ValueError("level set report date cannot follow its requested date")
            if row.get("historical_publication_time"):
                raise ValueError("historical level publication evidence needs policy review")
            if row.get("data_partition") == "later_reference_only":
                raise ValueError("later-reference level set became eligible within cutoff")
            set_id = row.get("level_set_id", "")
            if not set_id or set_id in sets_by_id or requested in requested_seen:
                raise ValueError(f"duplicate/conflicting level set for requested date {requested}")
            requested_seen.add(requested)
            count_text = row.get("level_count", "")
            if not count_text.isdecimal() or int(count_text) < 1:
                raise ValueError(f"invalid level_count for {set_id}")
            sets_by_id[set_id] = (requested, report, eligible, _identity(row), int(count_text))

        levels_by_id: dict[str, dict[str, _Level]] = {set_id: {} for set_id in sets_by_id}
        for row in gamma_long_rows:
            set_id = row.get("level_set_id", "")
            if set_id not in sets_by_id:
                continue  # Out-of-scope measurement columns are not interpreted.
            if row.get("historical_publication_time"):
                raise ValueError("historical level publication evidence needs policy review")
            requested, report, _, set_source, _ = sets_by_id[set_id]
            if (
                _day(row.get("requested_date", ""), "requested_date") != requested
                or _day(row.get("vendor_report_date", ""), "vendor_report_date") != report
                or row.get("ticker") != ticker
                or row.get("level_type") != "gamma_levels"
                or row.get("kind") != "eod"
            ):
                raise ValueError(f"level row identity disagrees with set {set_id}")
            source = _identity(row)
            if source[:2] != set_source[:2]:
                raise ValueError(f"level row raw source disagrees with set {set_id}")
            name = row.get("level_name", "")
            if not name or name in levels_by_id[set_id]:
                raise ValueError(f"duplicate/empty level name in set {set_id}")
            levels_by_id[set_id][name] = _Level(
                name,
                _number(row.get("value"), "value"),
                _number(row.get("gex"), "gex"),
                source[2],
            )

        level_sets: list[_LevelSet] = []
        for set_id, (requested, report, eligible, source, expected_count) in sets_by_id.items():
            levels = levels_by_id[set_id]
            if len(levels) != expected_count:
                raise ValueError(f"level count mismatch for set {set_id}")
            level_sets.append(
                _LevelSet(set_id, requested, report, eligible, source, tuple(levels.values()))
            )
        level_sets.sort(key=lambda item: (item.eligible_from, item.requested_day))
        self._level_sets = tuple(level_sets)
        self._level_times = tuple(item.eligible_from for item in level_sets)
        best: _LevelSet | None = None
        best_prefix: list[_LevelSet] = []
        for item in level_sets:
            if best is None or item.requested_day > best.requested_day:
                best = item
            best_prefix.append(best)
        self._level_best_prefix = tuple(best_prefix)

    def snapshot(self, decision_time_utc: datetime) -> dict[str, Any]:
        """Return a JSON-ready decision record; no state changes on repeated reads."""
        now = _utc(decision_time_utc)
        result: dict[str, Any] = {
            "policy_id": POLICY_ID,
            "decision_time_utc": _stamp(now),
            "historical_publication_verified": False,
            "bundle_sha256": self.bundle_sha256,
            "table_sha256": dict(self.table_sha256),
            "ticker": self.ticker,
        }
        if now > self.cutoff_utc:
            return dict(result, status="outside_authorized_cutoff", gamma=None, levels=None)
        result["status"] = "within_cutoff"
        gamma_pos = bisect_right(self._gamma_times, now) - 1
        result["gamma"] = self._gamma_record(self._gamma[gamma_pos], now) if gamma_pos >= 0 else {
            "status": "no_eligible_report", "sign": "unknown", "value": None,
            "positive_run_age": None, "positive_run_lower_bound": 0,
        }
        level_pos = bisect_right(self._level_times, now) - 1
        result["levels"] = (
            self._level_record(self._level_best_prefix[level_pos], now)
            if level_pos >= 0 else {"status": "no_eligible_set", "items": {}}
        )
        return result

    def policy_context_fields(self, decision_time_utc: datetime) -> dict[str, Any]:
        """Project a decision into Core's context-field vocabulary.

        The returned UTC datetime and mappings are meant for a Core context
        constructor; ``snapshot`` is the JSON-ready audit/worker record. A
        left-censored six-plus positive run passes its observed lower bound as
        an age with ``positive_run_age_is_lower_bound=True``. It cannot satisfy
        an early-positive (1..5) condition.
        """
        record = self.snapshot(decision_time_utc)
        gamma = record["gamma"] or {}
        levels = record["levels"] or {}
        items = levels.get("items", {})
        positive_age = gamma.get("positive_run_age")
        lower_bound = gamma.get("positive_run_lower_bound", 0)
        age_is_lower_bound = positive_age is None and lower_bound >= 6
        if age_is_lower_bound:
            positive_age = lower_bound

        def eligible_time(value: str | None) -> datetime | None:
            return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

        def core_status(value: str | None) -> str | None:
            return "available" if value == "selected" else value

        return {
            "decision_ts_utc": _utc(decision_time_utc),
            "total_net_gex": gamma.get("value"),
            "positive_run_age": positive_age,
            "positive_run_age_is_lower_bound": age_is_lower_bound,
            "level_prices": {name: row["price"] for name, row in items.items()},
            "level_gex": {name: row["gex"] for name, row in items.items()},
            "gamma_status": core_status(gamma.get("status", record["status"])),
            "levels_status": core_status(levels.get("status", record["status"])),
            "gamma_report_date": gamma.get("report_date"),
            "level_requested_date": levels.get("requested_date"),
            "level_report_date": levels.get("report_date"),
            "gamma_eligible_from_utc": eligible_time(gamma.get("nominal_eligible_from_utc")),
            "level_eligible_from_utc": eligible_time(levels.get("nominal_eligible_from_utc")),
            "gamma_source_id": gamma.get("source_sha256"),
            "level_source_id": levels.get("level_set_id"),
        }

    @staticmethod
    def _gamma_record(item: _Gamma, now: datetime) -> dict[str, Any]:
        age = (now.astimezone(_CHICAGO).date() - item.report_day).days
        if age > MAX_AGE_CALENDAR_DAYS:
            status = "stale"
        elif item.value is None:
            status = "null"
        else:
            status = "selected"
        sign = "unknown"
        if status == "selected":
            sign = "positive" if item.value > 0 else "negative" if item.value < 0 else "neutral"
        known_age = (
            item.positive_run_lower_bound
            if item.positive_age_known and sign == "positive"
            else None
        )
        lower_bound = item.positive_run_lower_bound if sign == "positive" else 0
        phase = (
            "early_positive" if known_age is not None and 1 <= known_age <= 5
            else "established_positive" if lower_bound >= 6
            else "positive_age_unknown" if sign == "positive"
            else "none"
        )
        return {
            "status": status,
            "report_date": item.report_day.isoformat(),
            "nominal_eligible_from_utc": _stamp(item.eligible_from),
            "age_calendar_days": age,
            "source_path": item.source[0],
            "source_sha256": item.source[1],
            "source_json_pointer": item.source[2],
            "value": item.value if status == "selected" else None,
            "sign": sign,
            "positive_run_age": known_age,
            "positive_run_lower_bound": lower_bound,
            "positive_phase": phase,
        }

    @staticmethod
    def _level_record(item: _LevelSet, now: datetime) -> dict[str, Any]:
        age = (now.astimezone(_CHICAGO).date() - item.report_day).days
        stale = age > MAX_AGE_CALENDAR_DAYS
        return {
            "status": "stale" if stale else "selected",
            "level_set_id": item.set_id,
            "requested_date": item.requested_day.isoformat(),
            "report_date": item.report_day.isoformat(),
            "nominal_eligible_from_utc": _stamp(item.eligible_from),
            "age_calendar_days": age,
            "source_path": item.source[0],
            "source_sha256": item.source[1],
            "source_json_pointer": item.source[2],
            "items": {
                level.name: {
                    "price": level.price,
                    "gex": level.gex,
                    "source_json_pointer": level.source_pointer,
                }
                for level in item.levels
            } if not stale else {},
        }


def load_v02_eod_asof_zip(
    path: Path | str,
    *,
    cutoff_utc: datetime,
    expected_archive_sha256: str | None = None,
    ticker: str = "NQ1!",
) -> EodAsOfIndex:
    """Verify immutable archive bytes and stream its three canonical EOD tables.

    Hashing the whole archive and each whole member is an integrity-only read.
    CSV streams are mixed-date containers: future row text is read to inspect
    identities, while measurement fields are parsed only for eligible rows.
    No CSV, raw market tape, or extracted artifact is written by this loader.
    """
    archive = Path(path)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    archive_hash = digest.hexdigest()
    if expected_archive_sha256 is not None and archive_hash != expected_archive_sha256:
        raise ValueError("MenthorQ archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as package:
        table_hashes: dict[str, str] = {}
        for key, member in _TABLE_MEMBERS.items():
            digest = hashlib.sha256()
            with package.open(member) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            table_hashes[key] = digest.hexdigest()
        with (
            package.open(_TABLE_MEMBERS["total_gamma"]) as gamma_file,
            package.open(_TABLE_MEMBERS["level_sets"]) as sets_file,
            package.open(_TABLE_MEMBERS["gamma_long"]) as levels_file,
            io.TextIOWrapper(gamma_file, encoding="utf-8-sig", newline="") as gamma_text,
            io.TextIOWrapper(sets_file, encoding="utf-8-sig", newline="") as sets_text,
            io.TextIOWrapper(levels_file, encoding="utf-8-sig", newline="") as levels_text,
        ):
            return EodAsOfIndex(
                gamma_rows=csv.DictReader(gamma_text),
                level_set_rows=csv.DictReader(sets_text),
                gamma_long_rows=csv.DictReader(levels_text),
                cutoff_utc=cutoff_utc,
                bundle_sha256=archive_hash,
                table_sha256=table_hashes,
                ticker=ticker,
            )


def load_v02_eod_asof_root(
    root: Path | str,
    *,
    cutoff_utc: datetime,
    bundle_sha256: str,
    expected_table_sha256: Mapping[str, str],
    ticker: str = "NQ1!",
) -> EodAsOfIndex:
    """Load an extracted immutable v02 root after verifying exact table bytes.

    ``root`` is the directory containing ``data/canonical``. The supplied
    bundle identity is provenance; this function verifies each behavioral
    table against the caller's pinned hashes, not an unextracted ZIP's bytes.
    Full-table reads are for integrity only; the CSV pass parses eligible
    measurements after checking each row's requested/report date identity.
    """
    source_root = Path(root).resolve(strict=True)
    paths = {
        key: (source_root / relative).resolve(strict=True)
        for key, relative in _TABLE_RELATIVE.items()
    }
    if any(not path.is_relative_to(source_root) for path in paths.values()):
        raise ValueError("MenthorQ table path escapes the selected source root")
    table_hashes: dict[str, str] = {}
    for key, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        table_hashes[key] = digest.hexdigest()
    if table_hashes != dict(expected_table_sha256):
        raise ValueError("MenthorQ extracted table SHA-256 mismatch")
    with (
        paths["total_gamma"].open("r", encoding="utf-8-sig", newline="") as gamma_file,
        paths["level_sets"].open("r", encoding="utf-8-sig", newline="") as sets_file,
        paths["gamma_long"].open("r", encoding="utf-8-sig", newline="") as levels_file,
    ):
        return EodAsOfIndex(
            gamma_rows=csv.DictReader(gamma_file),
            level_set_rows=csv.DictReader(sets_file),
            gamma_long_rows=csv.DictReader(levels_file),
            cutoff_utc=cutoff_utc,
            bundle_sha256=bundle_sha256,
            table_sha256=table_hashes,
            ticker=ticker,
        )
