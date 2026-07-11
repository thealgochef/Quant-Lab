"""Tests for the INGEST batch converter (scripts/ingest_databento_batch.py)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _load_module():
    module_name = "_ingest_databento_batch_under_test"
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ingest_databento_batch.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_plan_jobs_parses_filename_date_and_names_mbp1(tmp_path: Path):
    mod = _load_module()
    names = [
        "glbx-mdp3-20260112.mbp-1.dbn.zst",
        "glbx-mdp3-20260111.mbp-1.dbn.zst",
        "metadata.json",  # non-day members ignored
        "condition.json",
    ]
    jobs = mod.plan_jobs(names, out_root=tmp_path)
    assert [j.date_str for j in jobs] == ["2026-01-11", "2026-01-12"]  # sorted
    assert jobs[0].out_path == tmp_path / "2026-01-11" / "mbp1.parquet"
    assert jobs[0].schema_raw == "mbp-1"


def test_plan_jobs_date_range_filters_inclusive(tmp_path: Path):
    mod = _load_module()
    names = [f"glbx-mdp3-2026011{d}.mbp-1.dbn.zst" for d in range(1, 6)]
    jobs = mod.plan_jobs(
        names, out_root=tmp_path, start_date="2026-01-12", end_date="2026-01-14"
    )
    assert [j.date_str for j in jobs] == ["2026-01-12", "2026-01-13", "2026-01-14"]


def test_target_exists_and_loads_rejects_missing_empty_and_corrupt(tmp_path: Path):
    mod = _load_module()
    missing = tmp_path / "missing.parquet"
    assert mod.target_exists_and_loads(missing) is False
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet file")
    assert mod.target_exists_and_loads(corrupt) is False
    good = tmp_path / "good.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(good)
    assert mod.target_exists_and_loads(good) is True


def test_filter_spreads_drops_calendar_spreads_keeps_outrights():
    mod = _load_module()
    df = pd.DataFrame({"symbol": ["NQH6", "NQH6-NQM6", "NQM6", "NQH6-NQH7"], "x": range(4)})
    filtered, dropped = mod.filter_spreads(df)
    assert dropped == 2
    assert list(filtered["symbol"]) == ["NQH6", "NQM6"]
    no_symbol = pd.DataFrame({"x": [1]})
    unchanged, dropped = mod.filter_spreads(no_symbol)
    assert dropped == 0 and len(unchanged) == 1


def _frame(ts_recv: list[datetime]) -> pd.DataFrame:
    df = pd.DataFrame({"price": [25000.0] * len(ts_recv)})
    df.index = pd.DatetimeIndex(pd.to_datetime(ts_recv, utc=True), name="ts_recv")
    return df


def test_sanity_problems_gates_day_window_and_empty():
    mod = _load_module()
    inside = _frame(
        [datetime(2026, 1, 12, 0, 0, tzinfo=UTC), datetime(2026, 1, 12, 23, 59, tzinfo=UTC)]
    )
    fatal, warns = mod.sanity_problems(inside, "2026-01-12")
    assert fatal == []
    assert any("low row count" in w for w in warns)

    outside = _frame([datetime(2026, 1, 13, 0, 0, tzinfo=UTC)])
    fatal, _ = mod.sanity_problems(outside, "2026-01-12")
    assert fatal and "outside day window" in fatal[0]

    fatal, _ = mod.sanity_problems(inside.iloc[:0], "2026-01-12")
    assert fatal == ["0 rows after filtering"]


def _exact_frame(mod) -> pd.DataFrame:
    ts = pd.to_datetime([datetime(2026, 1, 12, 12, tzinfo=UTC)], utc=True)
    data: dict[str, object] = {}
    for name, arrow_type in mod.EXPECTED_MBP1_SCHEMA:
        if name == "ts_recv":
            continue
        if arrow_type == "timestamp[ns, tz=UTC]":
            data[name] = ts
        elif arrow_type == "string":
            data[name] = pd.array(["T"], dtype=str)
        elif arrow_type == "double":
            data[name] = pd.array([25000.0], dtype="float64")
        else:
            data[name] = pd.array([1], dtype=arrow_type)
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex(ts, name="ts_recv")
    return df


def test_schema_problems_pins_store_fingerprint(tmp_path: Path):
    mod = _load_module()
    df = _exact_frame(mod)

    exact = tmp_path / "exact.parquet"
    df.to_parquet(exact)
    assert mod.schema_problems(exact) == []

    wrong = tmp_path / "wrong.parquet"
    df.drop(columns=["symbol"]).to_parquet(wrong)
    problems = mod.schema_problems(wrong)
    assert problems and "symbol" in problems[0]

    retyped = tmp_path / "retyped.parquet"
    df.assign(price=pd.array([25000], dtype="int64")).to_parquet(retyped)
    problems = mod.schema_problems(retyped)
    assert problems and "price" in problems[0]


def test_schema_problems_reports_pure_order_mismatch(tmp_path: Path):
    # Same name:type multiset, different physical order -> its own finding
    # (close-verify fix: this case previously produced an empty message).
    mod = _load_module()
    df = _exact_frame(mod)
    reordered = tmp_path / "reordered.parquet"
    df[list(df.columns[::-1])].to_parquet(reordered)
    assert mod.schema_problems(reordered) == ["column order differs"]

    missing = tmp_path / "missing.parquet"
    df.drop(columns=["symbol"]).to_parquet(missing)
    problems = mod.schema_problems(missing)
    assert "column order differs" not in problems[0]


def test_format_log_line_carries_day_rows_seconds():
    mod = _load_module()
    ok = mod.DayResult("2026-01-12", "OK", 8_934_927, 28.5, "filtered 2,003 spread rows", 201.1)
    line = mod.format_log_line(ok)
    assert line.startswith("2026-01-12 | rows=8,934,927 | 28.5s | OK (201.1 MB)")
    failed = mod.DayResult("2026-01-13", "FAILED", 0, 1.0, "boom")
    assert "FAILED | boom" in mod.format_log_line(failed)
