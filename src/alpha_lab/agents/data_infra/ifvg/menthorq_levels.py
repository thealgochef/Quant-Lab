"""A1 run-time EOD lookup; source files and derived values are not dataset tables.

CSV parsing is cached by immutable source bytes and the fixed availability policy.
Each run receives its own provider, including its replay-local prior-cash-close map.
No source-selected contract is exposed by the current day-artifact seam, so the
snapshot's instrument and roll fields remain null rather than inferring a roll.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from strategy_core.constants import RESEARCH_SESSION_SCHEME
from strategy_core.decisions.sessions import trading_day_for
from strategy_core.strategies.ifvg_smc.menthorq_levels import (
    LEVEL_COLUMN_NAMES,
    MenthorqLevelSnapshot,
    derive_menthorq_values,
)
from strategy_core.types import SessionScheme

if TYPE_CHECKING:
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

    from .day_artifacts import DayArtifacts

__all__ = [
    "DEFAULT_LEVELS_PATH",
    "DEFAULT_REGIME_PATH",
    "FORMULA_VERSION",
    "SCHEMA_VERSION",
    "MenthorqLevels",
    "load_menthorq_levels",
    "menthorq_provider_for_section",
]

DEFAULT_LEVELS_PATH = Path(r"C:\menthorq\data\eod_gamma_levels_daily_wide.csv")
DEFAULT_REGIME_PATH = Path(r"C:\menthorq\data\daily_total_dealer_gamma_and_regime.csv")
SCHEMA_VERSION = 1
FORMULA_VERSION = "menthorq_eod_v1"
_CHICAGO = ZoneInfo("America/Chicago")
_LEVEL_COLUMNS = ("trading_date", "source_eod_date", *LEVEL_COLUMN_NAMES)
_REGIME_COLUMNS = (
    "trading_date", "source_eod_date", "total_net_gex", "regime", "gex_percentile_1y"
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


@dataclass(frozen=True)
class _LevelRow:
    source_eod_date: date
    levels: Mapping[str, float | None]
    implied_move_points: float | None


@dataclass(frozen=True)
class _RegimeRow:
    regime: str | None
    total_net_gex: float | None
    gex_percentile_1y: float | None


@dataclass(frozen=True)
class _ParsedSources:
    levels: Mapping[date, _LevelRow]
    regimes: Mapping[date, _RegimeRow]


# Hashes + versioned formula/schema + the specified availability policy. A path
# is deliberately absent, so replacing a file at one path cannot hit stale data.
_SOURCE_CACHE: dict[tuple[str, str, int, str, str, int, int], _ParsedSources] = {}


def _date_cell(value: str, *, field: str, row_number: int) -> date:
    if not _ISO_DATE.fullmatch(value):
        raise ValueError(f"MenthorQ row {row_number}: {field} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"MenthorQ row {row_number}: invalid {field}") from exc


def _number_cell(value: str, *, field: str, row_number: int) -> float | None:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"MenthorQ row {row_number}: unparseable {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"MenthorQ row {row_number}: non-finite {field}")
    return number


def _csv_rows(payload: bytes, expected_columns: tuple[str, ...]):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
    if tuple(reader.fieldnames or ()) != expected_columns:
        raise ValueError(f"MenthorQ CSV schema must be {expected_columns!r}")
    seen: set[date] = set()
    for row_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"MenthorQ row {row_number}: malformed column count")
        day = _date_cell(row["trading_date"], field="trading_date", row_number=row_number)
        if day in seen:
            raise ValueError(f"MenthorQ duplicate trading_date: {day.isoformat()}")
        seen.add(day)
        source_day = _date_cell(
            row["source_eod_date"], field="source_eod_date", row_number=row_number
        )
        if source_day >= day:
            raise ValueError(
                f"MenthorQ row {row_number}: source_eod_date must precede trading_date"
            )
        yield row_number, day, source_day, row


def _parse_sources(level_payload: bytes, regime_payload: bytes) -> _ParsedSources:
    level_rows: dict[date, _LevelRow] = {}
    for row_number, day, source_day, row in _csv_rows(level_payload, _LEVEL_COLUMNS):
        levels = {
            name: _number_cell(row[name], field=name, row_number=row_number)
            for name in LEVEL_COLUMN_NAMES
        }
        low, high = levels["1D Min"], levels["1D Max"]
        implied = (high - low) / 2 if low is not None and high is not None else None
        if implied is not None and implied <= 0:
            implied = None
        level_rows[day] = _LevelRow(source_day, MappingProxyType(levels), implied)
    regime_rows: dict[date, _RegimeRow] = {}
    for row_number, day, _, row in _csv_rows(regime_payload, _REGIME_COLUMNS):
        regime = row["regime"] if row["regime"].strip() else None
        if regime not in {"positive", "negative", None}:
            raise ValueError(f"MenthorQ row {row_number}: invalid regime {regime!r}")
        regime_rows[day] = _RegimeRow(
            regime,
            _number_cell(row["total_net_gex"], field="total_net_gex", row_number=row_number),
            _number_cell(
                row["gex_percentile_1y"], field="gex_percentile_1y", row_number=row_number
            ),
        )
    return _ParsedSources(MappingProxyType(level_rows), MappingProxyType(regime_rows))


class MenthorqLevels:
    """One run's lookup over verified source rows and completed prior-day bars."""

    schema_version = SCHEMA_VERSION
    formula_version = FORMULA_VERSION

    def __init__(self, parsed: _ParsedSources, source_file_sha256: Mapping[str, str]):
        self._parsed = parsed
        self.source_file_sha256 = MappingProxyType(dict(source_file_sha256))
        self._cash_closes: dict[date, float | None] = {}

    def snapshot(self, ts_utc: datetime) -> MenthorqLevelSnapshot:
        if ts_utc.tzinfo is None or ts_utc.utcoffset() is None:
            raise ValueError("MenthorQ lookup requires an aware UTC timestamp")
        chicago = ts_utc.astimezone(_CHICAGO)
        day = chicago.date()
        local_time = chicago.time()
        reason = (
            "before_0600" if local_time < time(6)
            else "after_1700" if local_time >= time(17)
            else "no_level_row" if day not in self._parsed.levels
            else None
        )
        outside = reason in {"before_0600", "after_1700"}
        row = None if outside else self._parsed.levels.get(day)
        regime = None if outside else self._parsed.regimes.get(day)
        return MenthorqLevelSnapshot(
            trading_date=day,
            source_eod_date=row.source_eod_date if row else None,
            source_file_sha256=self.source_file_sha256[DEFAULT_LEVELS_PATH.name],
            levels=(row.levels if row else {name: None for name in LEVEL_COLUMN_NAMES}),
            regime=regime.regime if regime and regime.regime is not None else "unknown",
            total_net_gex=regime.total_net_gex if regime else None,
            gex_percentile_1y=regime.gex_percentile_1y if regime else None,
            implied_move_points=row.implied_move_points if row else None,
            selected_instrument_id=None,
            roll_flag=None,
            context_available=reason is None,
            unavailable_reason=reason,
        )

    def register_day_artifacts(self, artifacts: DayArtifacts, tick_size: float) -> None:
        """Record completed 1m bars only; never scan an unrequested source date."""
        from .day_artifacts import cash_close_points_from_artifacts

        day = date.fromisoformat(artifacts.date_str)
        if any(bar.trading_day == day for bar in artifacts.bars):
            self._cash_closes[day] = cash_close_points_from_artifacts(artifacts, tick_size)

    def prior_cash_close_for(
        self, ts_utc: datetime, *, session_scheme: SessionScheme = RESEARCH_SESSION_SCHEME
    ) -> float | None:
        """Prior day with bars, keyed by Core's logical 18:00 ET day boundary."""
        day = trading_day_for(ts_utc, session_scheme)
        if day is None:
            return None
        prior_days = [prior for prior in self._cash_closes if prior < day]
        return self._cash_closes[max(prior_days)] if prior_days else None

    def derived(
        self, ts_utc: datetime, price_points: float, *,
        bar_open_points: float | None = None,
        prior_cash_close_points: float | None = None,
    ):
        return derive_menthorq_values(
            self.snapshot(ts_utc), price_points, ts_utc,
            bar_open_points=bar_open_points,
            prior_cash_close_points=prior_cash_close_points,
        )


def load_menthorq_levels(
    levels_path: Path | str = DEFAULT_LEVELS_PATH,
    regime_path: Path | str = DEFAULT_REGIME_PATH,
) -> MenthorqLevels:
    """Read each source once; reuse parsing only for identical bytes and policy."""
    levels_payload = Path(levels_path).read_bytes()
    regime_payload = Path(regime_path).read_bytes()
    levels_hash = hashlib.sha256(levels_payload).hexdigest()
    regime_hash = hashlib.sha256(regime_payload).hexdigest()
    key = (levels_hash, regime_hash, SCHEMA_VERSION, FORMULA_VERSION, "America/Chicago", 6, 17)
    parsed = _SOURCE_CACHE.get(key)
    if parsed is None:
        parsed = _parse_sources(levels_payload, regime_payload)
        _SOURCE_CACHE[key] = parsed
    return MenthorqLevels(parsed, {
        DEFAULT_LEVELS_PATH.name: levels_hash,
        DEFAULT_REGIME_PATH.name: regime_hash,
    })


def menthorq_provider_for_section(
    section: IfvgSmcSection, provider: MenthorqLevels | None = None
) -> MenthorqLevels | None:
    """Disabled sections never open the external CSVs or alter the level seam."""
    if section.menthorq_context_version is None:
        return None
    if section.menthorq_context_version != FORMULA_VERSION:
        raise ValueError("unsupported menthorq_context_version")
    return provider if provider is not None else load_menthorq_levels()
