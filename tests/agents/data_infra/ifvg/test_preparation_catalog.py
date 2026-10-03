"""Task B contract receipts are mutable provenance, never day-table metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from strategy_core.types import Bar, BarKind

from alpha_lab.agents.data_infra.ifvg.capture_driver import capture_single_date
from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import ExplorationDataPolicy
from alpha_lab.agents.data_infra.ifvg.day_artifacts import DayArtifacts, DaySeeds
from alpha_lab.agents.data_infra.ifvg.menthorq_levels import load_menthorq_levels
from alpha_lab.agents.data_infra.ifvg.preparation_catalog import (
    read_preparation_instrument,
    record_preparation_instrument,
)


def _artifacts(day, *, bars=True):
    ts = datetime.fromisoformat(f"{day}T15:00:00+00:00")
    bar = Bar(
        timeframe_ticks=60, trading_day=date.fromisoformat(day), bar_index=0,
        bar_id=f"{day}:1m", open_ts_utc=ts - timedelta(minutes=1), close_ts_utc=ts,
        open_ticks=100, high_ticks=101, low_ticks=99, close_ticks=100,
        volume=1, trade_count=1, is_complete=True, is_partial=False, kind=BarKind.TIME,
    )
    return DayArtifacts(day, [bar] if bars else [], {}, DaySeeds(None, None, None, None),
                        (101, 99) if bars else None, None, ())


def _provider(catalog):
    fixtures = Path(__file__).parent / "fixtures" / "menthorq_a1"
    return load_menthorq_levels(
        fixtures / "eod_gamma_levels_daily_wide.csv.fixture",
        fixtures / "daily_total_dealer_gamma_and_regime.csv.fixture",
        preparation_catalog_paths=(catalog,),
    )


@pytest.mark.parametrize("previous,current,expected", [
    (101, 202, True), (101, 101, False), (None, 202, None), (101, None, None),
])
def test_catalog_fills_selected_contract_and_three_roll_states(
    tmp_path, previous, current, expected,
):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"days": {
        "2026-01-12": {"selected_instrument_id": previous, "raw_symbol": "NQH6"},
        "2026-01-13": {"selected_instrument_id": current, "raw_symbol": "NQH6"},
    }}))
    provider = _provider(catalog)
    provider.register_day_artifacts(_artifacts("2026-01-12"), .25)
    provider.register_day_artifacts(_artifacts("2026-01-13"), .25)
    snapshot = provider.snapshot(datetime(2026, 1, 13, 16, tzinfo=UTC))
    assert snapshot.selected_instrument_id == current
    assert snapshot.roll_flag is expected
    assert provider.roll_flag_for(date(2026, 1, 12)) is None


def test_missing_catalog_entry_is_null_and_empty_day_is_not_roll_predecessor(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"days": {
        "2026-01-12": {"selected_instrument_id": 101},
        "2026-01-13": {"selected_instrument_id": 999},
        "2026-01-14": {"selected_instrument_id": 101},
    }}))
    provider = _provider(catalog)
    provider.register_day_artifacts(_artifacts("2026-01-12"), .25)
    provider.register_day_artifacts(_artifacts("2026-01-13", bars=False), .25)
    provider.register_day_artifacts(_artifacts("2026-01-14"), .25)
    assert provider.roll_flag_for(date(2026, 1, 14)) is False
    provider.register_day_artifacts(_artifacts("2026-01-15"), .25)
    snapshot = provider.snapshot(datetime(2026, 1, 15, 16, tzinfo=UTC))
    assert snapshot.selected_instrument_id is None
    assert snapshot.roll_flag is None
    provider.register_day_artifacts(_artifacts("2026-01-16"), .25)
    assert provider.roll_flag_for(date(2026, 1, 16)) is None


def test_front_month_index_backfill_preserves_existing_files_and_entry(tmp_path, monkeypatch):
    day = "2026-01-13"
    folder = tmp_path / "raw" / "NQ" / day
    folder.mkdir(parents=True)
    raw = folder / "mbp10.parquet"
    pq.write_table(pa.table({
        "ts_event": [datetime(2026, 1, 13, 16, tzinfo=UTC)] * 10,
        "instrument_id": [101, 101, 202, 202, 999, 999, 999, 303, 303, 303],
        "action": ["T"] * 7 + ["A"] * 3,
        "raw_symbol": ["NQH6"] * 2 + ["NQM6"] * 2 + ["NQH6-NQM6"] * 3 + ["NQU6"] * 3,
    }), raw)
    bars = folder / "ifvg_tbars_frozen.parquet"
    levels = folder / "ifvg_levels_frozen.parquet"
    bars.write_bytes(b"immutable A1 bar fixture")
    levels.write_bytes(b"immutable A1 level fixture")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw, bars, levels)}
    catalog = tmp_path / "job" / "catalog.json"
    catalog.parent.mkdir()
    catalog.write_text(json.dumps({"days": {day: {"status": "prepared", "seconds": 1.25}}}))
    # A receipt must never construct events or re-derive bars.
    from strategy_core.data.databento_parquet import DatabentoParquetSource

    monkeypatch.setattr(DatabentoParquetSource, "events", lambda _self: pytest.fail("bar drain"))
    entry = record_preparation_instrument(
        day, data_dir=tmp_path / "raw", symbol="NQ", catalog_path=catalog,
        access_policy=ExplorationDataPolicy(allowlist=frozenset({day})),
    )
    assert entry == {"selected_instrument_id": 202, "raw_symbol": "NQM6"}
    assert json.loads(catalog.read_text())["days"][day] == {
        **entry, "status": "prepared", "seconds": 1.25,
    }
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert read_preparation_instrument(day, (catalog,)) == 202


@pytest.mark.parametrize("source_symbols,raw_symbols,expected", [
    (["NQM6", "NQM6"], None, "NQM6"),
    (["NQM6", "NQM6"], [None, ""], "NQM6"),
    (["NQ", "NQ"], ["NQM6", "NQM6"], "NQM6"),
    (["NQ", "NQ"], None, None),
    (["NQ.c.0", "NQ.c.0"], None, None),
    (["NQH6", "NQM6"], None, None),
    ([b"NQM6", b"NQM6"], None, "NQM6"),
])
def test_raw_contract_symbol_fallback_uses_only_selected_instrument_index(
    tmp_path, monkeypatch, source_symbols, raw_symbols, expected,
):
    day = "2026-01-13"
    folder = tmp_path / "raw" / "NQ" / day
    folder.mkdir(parents=True)
    values = {
        "ts_event": [datetime(2026, 1, 13, 16, tzinfo=UTC)] * 3,
        "instrument_id": [101, 202, 202],
        "action": ["T"] * 3,
        "symbol": [source_symbols[0], *source_symbols],
        "price": [123.0] * 3,
    }
    if raw_symbols is not None:
        values["raw_symbol"] = [raw_symbols[0], *raw_symbols]
    pq.write_table(pa.table(values), folder / "mbp10.parquet")
    actual_parquet = pq.ParquetFile

    class IndexOnlyParquet:
        def __init__(self, *args, **kwargs):
            self.inner = actual_parquet(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read_row_groups(self, groups, *, columns):
            assert set(columns) <= {"instrument_id", "action", "raw_symbol", "symbol"}
            return self.inner.read_row_groups(groups, columns=columns)

        def read(self, *, columns):
            assert set(columns) <= {"instrument_id", "action", "raw_symbol", "symbol"}
            return self.inner.read(columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", IndexOnlyParquet)
    entry = record_preparation_instrument(
        day, data_dir=tmp_path / "raw", symbol="NQ", catalog_path=tmp_path / "catalog.json",
        access_policy=ExplorationDataPolicy(allowlist=frozenset({day})),
    )
    assert entry == {"selected_instrument_id": 202, "raw_symbol": expected}


def test_catalog_allows_same_instrument_symbol_enrichment_and_rejects_other_changes(tmp_path):
    day = "2026-01-13"
    folder = tmp_path / "raw" / "NQ" / day
    folder.mkdir(parents=True)
    raw = folder / "mbp10.parquet"

    def source(instrument, contract):
        pq.write_table(pa.table({
            "ts_event": [datetime(2026, 1, 13, 16, tzinfo=UTC)],
            "instrument_id": [instrument], "action": ["T"], "symbol": [contract],
        }), raw)

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"days": {
        day: {"selected_instrument_id": 202, "raw_symbol": None, "seconds": 1.25},
    }}))
    policy = ExplorationDataPolicy(allowlist=frozenset({day}))

    def record():
        return record_preparation_instrument(
            day, data_dir=tmp_path / "raw", symbol="NQ", catalog_path=catalog,
            access_policy=policy,
        )

    source(202, "NQM6")
    assert record() == {"selected_instrument_id": 202, "raw_symbol": "NQM6"}
    assert json.loads(catalog.read_text())["days"][day]["seconds"] == 1.25
    accepted = catalog.read_bytes()
    assert record()["raw_symbol"] == "NQM6"
    for instrument, contract in ((202, "NQU6"), (303, "NQM6"), (202, "NQ")):
        source(instrument, contract)
        with pytest.raises(ValueError, match="evidence changed"):
            record()
        assert catalog.read_bytes() == accepted


def test_catalog_conflicts_fail_and_missing_catalog_remains_null(tmp_path):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    assert read_preparation_instrument("2026-01-13", (first,)) is None
    for path, instrument in ((first, 101), (second, 202)):
        path.write_text(json.dumps({"days": {
            "2026-01-13": {"selected_instrument_id": instrument},
        }}))
    with pytest.raises(ValueError, match="disagree"):
        read_preparation_instrument("2026-01-13", (first, second))


def test_disabled_capture_has_identical_emissions_and_identity_with_catalog(tmp_path, monkeypatch):
    day = "2026-01-13"
    cfg = IfvgCaptureConfig()
    with_catalog = replace(cfg, preparation_catalog_paths=(tmp_path / "catalog.json",))
    monkeypatch.setattr(
        "alpha_lab.agents.data_infra.ifvg.preparation_catalog.read_preparation_instrument",
        lambda *_args: pytest.fail("disabled context read catalog"),
    )
    before = capture_single_date(day, cfg, artifacts=_artifacts(day), seed=None)
    after = capture_single_date(day, with_catalog, artifacts=_artifacts(day), seed=None)
    pd.testing.assert_frame_equal(before.rows, after.rows, check_exact=True)
    assert before.end_seed == after.end_seed
    assert cfg.profile_hash == with_catalog.profile_hash
    assert cfg.artifacts_tag() == with_catalog.artifacts_tag()
