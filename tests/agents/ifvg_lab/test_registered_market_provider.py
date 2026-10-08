"""Registered market views preserve full-session causality and source integrity."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_lab.agents.data_infra.ifvg.day_artifacts import _META_KEY
from alpha_lab.agents.data_infra.ifvg.manifest import file_sha256
from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
from alpha_lab.agents.data_infra.ifvg.presentation.lab import registered_market as provider


def _source(tmp_path, monkeypatch, *, conflict=False):
    day = "2025-06-16"
    path = tmp_path / "cache/NQ" / day / "ifvg_tbars_tag.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "timeframe_ticks": 60,
            "trading_day": day,
            "kind": "time",
            "bar_id": str(i),
            "logical_open_ts_utc": pd.Timestamp("2025-06-15T22:00:00Z") + pd.Timedelta(minutes=i),
            "logical_close_ts_utc": pd.Timestamp("2025-06-15T22:01:00Z") + pd.Timedelta(minutes=i),
            "open_ticks": 100,
            "high_ticks": 104,
            "low_ticks": 99,
            "close_ticks": 102,
            "volume": 10,
            "is_complete": i == 0,
            "is_partial": i != 0,
            "close_reason": "session_end" if i else None,
        }
        for i in range(2)
    ]
    if conflict:
        duplicate = {**rows[0], "high_ticks": 110}
        rows.append(duplicate)
    metadata = {"source": "saved"}
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    table = table.replace_schema_metadata({_META_KEY: json.dumps(metadata).encode()})
    pq.write_table(table, path)
    catalog = tmp_path / "source_contracts.json"
    catalog.write_text(
        json.dumps({"days": {day: {"selected_instrument_id": 12, "raw_symbol": "NQM5"}}}),
        encoding="utf-8",
    )
    owner = SimpleNamespace(
        owned_dates=(day,),
        registration_id="registered",
        definition={
            "artifact_cache_dir": str(tmp_path / "cache"),
            "artifacts_tag": "tag",
            "catalog_path": str(catalog),
        },
        days={day: {"bars_sha256": file_sha256(path), "metadata": metadata}},
    )
    events = []

    def authorize(value):
        if value != day:
            raise PermissionError("unbound day")
        events.append(("authorize", value))

    def resolve(value, factory):
        authorize(value)
        events.append(("path", value))
        return factory(value)

    access = SimpleNamespace(
        resolve_source_path=resolve,
        record_metadata_access=lambda d: None,
        record_file_open=lambda d: None,
        record_rows_read=lambda d, rows: events.append(("rows", rows)),
    )
    policy = SimpleNamespace(
        authorize_date=authorize,
        registrations=(owner,),
        for_day=lambda d: access,
        audit=SimpleNamespace(denied_dates={}),
        assert_zero_forbidden_access=lambda: None,
    )
    binding = {"dates": [day], "source": "synthetic exact registered source"}
    monkeypatch.setattr(provider, "_verify_sources", lambda plan: (binding, policy))
    plan = SimpleNamespace(
        source=SimpleNamespace(
            kind="verified_task_b_registered_inputs", cutoff_utc="2025-06-15T22:02:00Z"
        )
    )
    return plan, events, tmp_path / "derived"


def test_registered_full_session_bars_preserve_completion_and_contract_receipts(
    tmp_path, monkeypatch
):
    plan, events, output = _source(tmp_path, monkeypatch)
    provider.build_companion(plan, output_root=output)
    minutes = provider.load_companion(plan, root=output)
    assert len(minutes) == 2  # includes a minute with no selected strategy position
    assert list(minutes["close"]) == [25.5, 25.5]
    assert list(minutes["is_complete"]) == [True, False]
    assert list(minutes["is_partial"]) == [False, True]
    assert set(minutes["selected_instrument_id"]) == {12}
    assert set(minutes["raw_symbol"]) == {"NQM5"}
    assert not minutes["roll_flag"].any()
    assert minutes.attrs["source_provenance"]["forbidden_accesses"] == 0
    path_at = next(i for i, event in enumerate(events) if event[0] == "path")
    assert events[path_at - 1][0] == "authorize"


def test_conflicting_minutes_are_not_smoothed_or_deduplicated(tmp_path, monkeypatch):
    plan, _events, output = _source(tmp_path, monkeypatch, conflict=True)
    with pytest.raises(provider.MarketSourceError, match="Conflicting"):
        provider.build_companion(plan, output_root=output)


def test_derived_market_tampering_fails_integrity(tmp_path, monkeypatch):
    plan, _events, output = _source(tmp_path, monkeypatch)
    provider.build_companion(plan, output_root=output)
    (output / "market_minutes.parquet").write_bytes(b"different")
    with pytest.raises(provider.MarketSourceError) as error:
        provider.load_companion(plan, root=output)
    assert error.value.reason == "failed_integrity"


def test_registered_route_never_borrows_a_strategy_package(monkeypatch):
    expected = pd.DataFrame({"close": [1]})
    monkeypatch.setattr(provider, "load_companion", lambda plan: expected)
    monkeypatch.setattr(market, "study_package_root", lambda _: pytest.fail("unrelated package"))
    plan = SimpleNamespace(source=SimpleNamespace(kind="verified_task_b_registered_inputs"))
    assert market.load_study_index_minutes(plan) is expected


def test_protected_date_refused_before_any_source_path(tmp_path):
    source = SimpleNamespace(
        kind="verified_task_b_registered_inputs", warmup_dates=(), evaluation_dates=("2026-06-11",)
    )
    with pytest.raises(PermissionError, match="protected"):
        provider.source_binding(SimpleNamespace(source=source))
    assert not list(tmp_path.iterdir())
