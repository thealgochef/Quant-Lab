from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.search.entry_activity import (
    day_start,
    entry_activity,
    entry_trading_day,
)


def test_activity_report_is_manifest_bound_and_json_safe(tmp_path):
    import hashlib
    import json

    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from alpha_lab.agents.data_infra.ifvg.manifest import save_v2_dataset_immutable
    from alpha_lab.agents.data_infra.ifvg.reporting import build_invariant_audit
    from alpha_lab.agents.data_infra.ifvg.search.entry_activity import entry_activity_payload
    from tests.agents.test_ifvg_v2_reporting_manifest import _identity, _tables

    tables = _tables()
    trades = tables[RecordTable.EXECUTED_TRADE]
    dates = sorted({entry_trading_day(ts) for ts in trades.entry_ts_utc})
    report = entry_activity_payload(
        trades, dates, cutoff_utc=pd.to_datetime(trades.resolution_ts_utc, utc=True).max()
    )
    json.dumps(report, allow_nan=False)
    output = save_v2_dataset_immutable(
        base_dir=tmp_path, identity=_identity(), raw_config={}, effective_config={},
        tables=tables, candidate_report={}, decision_report={}, executed_trade_report={},
        invariant_audit=build_invariant_audit(tables, data_access_audit={}),
        data_access_audit={}, entry_activity_report=report,
    )
    path = output / "entry_activity_report.json"
    assert json.loads(path.read_text()) == report
    manifest = json.loads((output / "manifest.json").read_text())
    entry = next(e for e in manifest["artifacts"] if e["path"].endswith(path.name))
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_entry_day_boundary_and_dst_are_wall_clock():
    assert entry_trading_day("2026-02-10T22:59:59Z") == "2026-02-10"
    assert entry_trading_day("2026-02-10T23:00:00Z") == "2026-02-11"
    assert entry_trading_day("2026-03-08T22:00:00Z") == "2026-03-09"
    assert str(day_start("2026-03-09")) == "2026-03-08 22:00:00+00:00"


def test_explicit_calendar_ties_censoring_and_position_time():
    trades = pd.DataFrame(
        [
            {
                "trade_id": "a",
                "entry_ts_utc": "2026-02-10T17:43:00Z",
                "resolution_ts_utc": "2026-02-12T17:47:00Z",
            },
            {
                "trade_id": "b",
                "entry_ts_utc": "2026-02-16T13:00:00Z",
                "resolution_ts_utc": "2026-02-16T13:04:00Z",
            },
        ]
    )
    result = entry_activity(
        trades,
        ["2026-02-11", "2026-02-12", "2026-02-16", "2026-02-17", "2026-02-18"],
        cutoff_utc="2026-02-18T21:00:00Z",
    )
    assert result["daily"].actual_entries.tolist() == [0, 0, 1, 0, 0]
    assert len(result["summary"]["tied_longest_intervals"]) == 2
    assert result["gaps"].leading_censored.tolist() == [True, False]
    assert result["gaps"].trailing_censored.tolist() == [False, True]
    assert result["elapsed"].flat_seconds.iloc[0] < result["elapsed"].entry_to_entry_seconds.iloc[0]
    assert str(result["gaps"].cohort_start_utc.iloc[0]) == "2026-02-10 23:00:00+00:00"


def test_empty_trajectory_retains_every_date():
    trades = pd.DataFrame(columns=["trade_id", "entry_ts_utc", "resolution_ts_utc"])
    result = entry_activity(trades, ["2026-02-11", "2026-02-12"], cutoff_utc="2026-02-12T21:00:00Z")
    assert result["summary"]["longest_zero_entry_dates"] == 2
    assert result["gaps"].iloc[0].leading_censored
    assert result["gaps"].iloc[0].trailing_censored


@pytest.mark.parametrize("problem", ["duplicate", "future", "naive"])
def test_invalid_execution_records_fail_closed(problem):
    row = {
        "trade_id": "a",
        "entry_ts_utc": "2026-02-10T17:43:00Z",
        "resolution_ts_utc": "2026-02-10T17:47:00Z",
    }
    if problem == "future":
        row["entry_ts_utc"] = "2026-07-01T12:00:00Z"
    if problem == "naive":
        row["entry_ts_utc"] = "2026-02-10T17:43:00"
    trades = pd.DataFrame([row, row] if problem == "duplicate" else [row])
    with pytest.raises(ValueError):
        entry_activity(trades, ["2026-02-11"], cutoff_utc="2026-06-10T21:00:00Z")
