"""A1 source lookup over the supplied CSV fixtures and synthetic boundary rows."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from strategy_core.strategies.ifvg_smc.menthorq_levels import LEVEL_COLUMN_NAMES
from strategy_core.strategies.ifvg_smc.replay import IfvgLevelInputs
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section
from strategy_core.types import Bar, BarKind

from alpha_lab.agents.data_infra.ifvg.day_artifacts import (
    DayArtifacts,
    DaySeeds,
    cash_close_points_from_artifacts,
    levels_for_from_frame,
)
from alpha_lab.agents.data_infra.ifvg.menthorq_levels import (
    DEFAULT_LEVELS_PATH,
    DEFAULT_REGIME_PATH,
    load_menthorq_levels,
    menthorq_provider_for_section,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "menthorq_a1"
_LEVELS = _FIXTURES / "eod_gamma_levels_daily_wide.csv.fixture"
_REGIMES = _FIXTURES / "daily_total_dealer_gamma_and_regime.csv.fixture"
_CHICAGO = ZoneInfo("America/Chicago")


def _ts(day: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}").replace(tzinfo=_CHICAGO).astimezone(UTC)


def _fixture_row(path: Path, day: str = "2026-01-13") -> dict[str, str]:
    return next(row for row in csv.DictReader(io.StringIO(path.read_text()))
                if row["trading_date"] == day)


def _write_rows(path: Path, columns, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sources(tmp_path: Path, *, day: str = "2026-01-13", level_changes=None,
             regime_changes=None, include_levels=True, include_regime=True):
    level = _fixture_row(_LEVELS)
    regime = _fixture_row(_REGIMES)
    for row in (level, regime):
        row["trading_date"] = day
        row["source_eod_date"] = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    level.update(level_changes or {})
    regime.update(regime_changes or {})
    levels_path = _write_rows(tmp_path / "levels.csv", level, [level] if include_levels else [])
    regimes_path = _write_rows(tmp_path / "regimes.csv", regime, [regime] if include_regime else [])
    return levels_path, regimes_path


def test_supplied_source_schema_values_hashes_and_read_only_rows():
    provider = load_menthorq_levels(_LEVELS, _REGIMES)
    snapshot = provider.snapshot(_ts("2026-01-13", "10:00"))
    assert provider.schema_version == 1
    assert provider.formula_version == "menthorq_eod_v1"
    assert provider.source_file_sha256 == {
        DEFAULT_LEVELS_PATH.name: hashlib.sha256(_LEVELS.read_bytes()).hexdigest(),
        DEFAULT_REGIME_PATH.name: hashlib.sha256(_REGIMES.read_bytes()).hexdigest(),
    }
    assert tuple(snapshot.levels) == LEVEL_COLUMN_NAMES
    assert snapshot.source_eod_date == date(2026, 1, 12)
    assert snapshot.regime == "positive"
    assert snapshot.total_net_gex == pytest.approx(3267281.2453690064)
    assert snapshot.levels["HVL"] == 25850.0
    assert snapshot.implied_move_points == pytest.approx((26231.15 - 25653.35) / 2)
    assert snapshot.selected_instrument_id is None
    assert snapshot.roll_flag is None
    with pytest.raises(TypeError):
        snapshot.levels["HVL"] = 1.0


@pytest.mark.parametrize("day", ["2026-01-13", "2026-03-08"])
@pytest.mark.parametrize("hhmm,reason", [
    ("05:59", "before_0600"), ("06:00", None),
    ("16:59", None), ("17:00", "after_1700"),
])
def test_availability_boundaries_standard_and_dst_change_day(tmp_path, day, hhmm, reason):
    provider = load_menthorq_levels(*_sources(tmp_path, day=day))
    snapshot = provider.snapshot(_ts(day, hhmm))
    assert snapshot.trading_date == date.fromisoformat(day)
    assert snapshot.context_available is (reason is None)
    assert snapshot.unavailable_reason == reason
    if reason is not None:
        assert snapshot.source_eod_date is None
        assert snapshot.regime == "unknown"
        assert snapshot.total_net_gex is None
        assert snapshot.gex_percentile_1y is None
        assert snapshot.implied_move_points is None
        assert all(value is None for value in snapshot.levels.values())


def test_rows_never_fill_adjacent_dates_and_regime_is_independent(tmp_path):
    provider = load_menthorq_levels(*_sources(tmp_path))
    missing = provider.snapshot(_ts("2026-01-14", "10:00"))
    assert not missing.context_available
    assert missing.unavailable_reason == "no_level_row"
    assert missing.regime == "unknown"
    assert missing.source_eod_date is None
    levels_path, regime_path = _sources(tmp_path, include_levels=False)
    independently_known = load_menthorq_levels(levels_path, regime_path).snapshot(
        _ts("2026-01-13", "10:00")
    )
    assert independently_known.regime == "positive"
    assert not independently_known.context_available
    assert independently_known.unavailable_reason == "no_level_row"
    levels_path, regime_path = _sources(tmp_path, include_regime=False)
    unknown = load_menthorq_levels(levels_path, regime_path).snapshot(
        _ts("2026-01-13", "10:00")
    )
    assert unknown.context_available
    assert unknown.regime == "unknown"


@pytest.mark.parametrize("source", ["levels", "regime"])
def test_duplicate_trading_dates_rejected_in_each_source(tmp_path, source):
    paths = _sources(tmp_path)
    index = 0 if source == "levels" else 1
    payload = paths[index].read_text()
    paths[index].write_text(payload + payload.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="duplicate trading_date"):
        load_menthorq_levels(*paths)


@pytest.mark.parametrize("source,field,value,match", [
    ("levels", "HVL", "not-a-number", "unparseable HVL"),
    ("regime", "total_net_gex", "NaN", "non-finite total_net_gex"),
    ("levels", "trading_date", "20260113", "must be an ISO date"),
    ("regime", "trading_date", "2026-13-01", "invalid trading_date"),
    ("levels", "source_eod_date", "2026-01-13", "must precede trading_date"),
    ("regime", "source_eod_date", "2026-01-14", "must precede trading_date"),
    ("regime", "regime", "neutral", "invalid regime"),
])
def test_invalid_source_cells_rejected(tmp_path, source, field, value, match):
    changes = {field: value}
    paths = _sources(
        tmp_path, level_changes=changes if source == "levels" else None,
        regime_changes=changes if source == "regime" else None,
    )
    with pytest.raises(ValueError, match=match):
        load_menthorq_levels(*paths)


@pytest.mark.parametrize("source", [0, 1])
def test_exact_source_column_names_and_order_required(tmp_path, source):
    paths = _sources(tmp_path)
    payload = paths[source].read_text()
    paths[source].write_text(payload.replace("source_eod_date", "source_date", 1))
    with pytest.raises(ValueError, match="CSV schema"):
        load_menthorq_levels(*paths)


@pytest.mark.parametrize("low,high", [("", "26231.15"), ("26231.15", "26231.15"),
                                     ("26231.15", "26000")])
def test_null_and_nonpositive_implied_move_stays_null(tmp_path, low, high):
    provider = load_menthorq_levels(*_sources(
        tmp_path, level_changes={"1D Min": low, "1D Max": high, "HVL": ""},
        regime_changes={"total_net_gex": "", "gex_percentile_1y": "", "regime": ""},
    ))
    snapshot = provider.snapshot(_ts("2026-01-13", "10:00"))
    assert snapshot.context_available
    assert snapshot.levels["HVL"] is None
    assert snapshot.regime == "unknown"
    assert snapshot.total_net_gex is None
    assert snapshot.gex_percentile_1y is None
    assert snapshot.implied_move_points is None


def test_cache_follows_content_not_filename_and_run_history_is_independent(tmp_path):
    paths = _sources(tmp_path)
    first = load_menthorq_levels(*paths)
    same = load_menthorq_levels(*paths)
    assert first is not same
    assert first._parsed is same._parsed
    paths = _sources(tmp_path, level_changes={"HVL": "25900.0"})
    changed = load_menthorq_levels(*paths)
    ts = _ts("2026-01-13", "10:00")
    assert changed.snapshot(ts).levels["HVL"] == 25900
    assert first.snapshot(ts).levels["HVL"] == 25850
    assert first.source_file_sha256 != changed.source_file_sha256
    first.register_day_artifacts(_artifacts("2026-01-12", [_bar("2026-01-12", "15:10", 100)]), .25)
    assert first.prior_cash_close_for(ts) == 25
    assert same.prior_cash_close_for(ts) is None


def test_naive_timestamp_rejected_and_disabled_section_does_not_load(monkeypatch):
    provider = load_menthorq_levels(_LEVELS, _REGIMES)
    with pytest.raises(ValueError, match="aware UTC"):
        provider.snapshot(datetime(2026, 1, 13, 10))
    monkeypatch.setattr(
        "alpha_lab.agents.data_infra.ifvg.menthorq_levels.load_menthorq_levels",
        lambda: pytest.fail("disabled context opened CSVs"),
    )
    assert menthorq_provider_for_section(default_ifvg_smc_section(), provider) is None


def _bar(day: str, hhmm: str, close_ticks: int, *, complete=True, timeframe=60):
    ts = _ts(day, hhmm)
    return Bar(
        timeframe_ticks=timeframe, trading_day=date.fromisoformat(day), bar_index=0,
        bar_id=f"{day}:{hhmm}:{timeframe}", open_ts_utc=ts - timedelta(seconds=60),
        close_ts_utc=ts, open_ticks=close_ticks, high_ticks=close_ticks, low_ticks=close_ticks,
        close_ticks=close_ticks, volume=1, trade_count=1, is_complete=complete,
        is_partial=not complete, kind=BarKind.TIME,
    )


def _artifacts(day: str, bars):
    return DayArtifacts(day, bars, {}, DaySeeds(None, None, None, None), None, None, ())


def test_prior_cash_close_uses_completed_one_minute_and_logical_day():
    provider = load_menthorq_levels(_LEVELS, _REGIMES)
    prior = _artifacts("2026-01-12", [
        _bar("2026-01-12", "15:09", 90), _bar("2026-01-12", "15:10", 100),
        _bar("2026-01-12", "15:11", 110), _bar("2026-01-12", "15:10", 200, complete=False),
        _bar("2026-01-12", "15:10", 300, timeframe=180),
    ])
    assert cash_close_points_from_artifacts(prior, .25) == 25
    provider.register_day_artifacts(prior, .25)
    # 18:00 Chicago is already the next logical 18:00 ET day; the CSV date differs.
    evening = _ts("2026-01-12", "18:00")
    assert provider.snapshot(evening).trading_date == date(2026, 1, 12)
    assert provider.prior_cash_close_for(evening) == 25
    provider.register_day_artifacts(_artifacts("2026-01-13", []), .25)
    assert provider.prior_cash_close_for(_ts("2026-01-14", "10:00")) == 25
    snapshot_ts = _ts("2026-01-13", "10:00")
    derived = provider.derived(snapshot_ts, 25850, bar_open_points=26,
                               prior_cash_close_points=provider.prior_cash_close_for(snapshot_ts))
    assert derived.opening_move_signed == pytest.approx(1 / ((26231.15 - 25653.35) / 2))
    assert derived.opening_move_abs == derived.opening_move_signed
    assert provider.derived(snapshot_ts, 25850, bar_open_points=26).opening_move_signed is None
    # The latest earlier logical day has bars but lacks an eligible close: null,
    # rather than silently falling back to a still earlier anchor.
    provider.register_day_artifacts(
        _artifacts("2026-01-13", [_bar("2026-01-13", "16:00", 150)]), .25
    )
    assert provider.prior_cash_close_for(_ts("2026-01-14", "10:00")) is None


def test_exact_level_handoff_preserves_disabled_tuple_and_supplies_requested_snapshot():
    ts = _ts("2026-01-13", "10:00")
    ordinary = ()
    default_lookup = levels_for_from_frame({ts: ordinary})
    assert default_lookup(ts) is ordinary
    assert default_lookup(ts + timedelta(minutes=1)) == ()
    provider = load_menthorq_levels(_LEVELS, _REGIMES)
    lookup = levels_for_from_frame({ts: ordinary}, menthorq_provider=provider)
    inputs = lookup(ts)
    assert isinstance(inputs, IfvgLevelInputs)
    assert inputs.levels is ordinary
    assert inputs.menthorq == provider.snapshot(ts)
    after = lookup(_ts("2026-01-13", "17:00"))
    assert after.levels == ()
    assert after.menthorq.unavailable_reason == "after_1700"
