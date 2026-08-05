"""QL FSM-audit lane contract tests (synthetic chain, no data access).

Drives the flatten → stamp → trace-ordinal → table-assembly → validation →
reconciliation path with the same deterministic three-day fixture Strategy-Core
pins its own audit parity on, proving the QL cut is exact without touching the
store or the accepted artifacts.
"""

from __future__ import annotations

import random
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from strategy_core.candles._ids import make_bar_id
from strategy_core.strategies.ifvg_smc.replay import run_day
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section
from strategy_core.types import Bar, BarKind, CloseReason

from alpha_lab.agents.data_infra.ifvg.audit_contracts import (
    AuditTable,
    reconcile_funnel_to_audit,
    validate_audit_table,
)
from alpha_lab.agents.data_infra.ifvg.capture_driver import (
    flatten_audit_emissions,
    flatten_emissions,
)
from alpha_lab.agents.data_infra.ifvg.dataset import assemble_fsm_audit_tables

_DAY0 = date(2026, 1, 5)
_DAY_START = datetime(2026, 1, 4, 23, 0, tzinfo=UTC)
_BARS_PER_DAY = 240


def _bar_1m(day_idx: int, index: int, o: int, h: int, low: int, c: int) -> Bar:
    day = _DAY0 + timedelta(days=day_idx)
    open_ts = _DAY_START + timedelta(days=day_idx, seconds=60 * index)
    return Bar(
        timeframe_ticks=60,
        trading_day=day,
        bar_index=index,
        bar_id=make_bar_id(60, day, index, BarKind.TIME),
        open_ts_utc=open_ts,
        close_ts_utc=open_ts + timedelta(seconds=59),
        open_ticks=o,
        high_ticks=h,
        low_ticks=low,
        close_ticks=c,
        volume=3,
        trade_count=2,
        is_complete=index < _BARS_PER_DAY - 1,
        is_partial=index >= _BARS_PER_DAY - 1,
        close_reason=(
            CloseReason.COMPLETE if index < _BARS_PER_DAY - 1 else CloseReason.END_OF_DAY
        ),
        kind=BarKind.TIME,
    )


def _aggregate(bars_1m: list[Bar], seconds: int) -> list[Bar]:
    n = seconds // 60
    out: list[Bar] = []
    day = bars_1m[0].trading_day
    for idx, start in enumerate(range(0, len(bars_1m), n)):
        chunk = bars_1m[start : start + n]
        out.append(
            Bar(
                timeframe_ticks=seconds,
                trading_day=day,
                bar_index=idx,
                bar_id=make_bar_id(seconds, day, idx, BarKind.TIME),
                open_ts_utc=chunk[0].open_ts_utc,
                close_ts_utc=chunk[-1].close_ts_utc,
                open_ticks=chunk[0].open_ticks,
                high_ticks=max(b.high_ticks for b in chunk),
                low_ticks=min(b.low_ticks for b in chunk),
                close_ticks=chunk[-1].close_ticks,
                volume=sum(b.volume for b in chunk),
                trade_count=sum(b.trade_count for b in chunk),
                is_complete=True,
                is_partial=False,
                close_reason=CloseReason.COMPLETE,
                kind=BarKind.TIME,
            )
        )
    return out


def _three_days() -> list[dict[int, list[Bar]]]:
    rng = random.Random(5)
    bases = [
        [20000, 20120, 20240, 20360],
        [20300, 20180, 20060, 19940],
        [20000, 20060, 19990, 20050],
    ]
    days: list[dict[int, list[Bar]]] = []
    for day_idx, chunk_bases in enumerate(bases):
        bars_1m = []
        for i in range(_BARS_PER_DAY):
            base = chunk_bases[i // 60]
            o = base + rng.randint(-4, 4)
            c = base + rng.randint(-4, 4)
            h = max(o, c) + rng.randint(0, 4)
            low = min(o, c) - rng.randint(0, 4)
            bars_1m.append(_bar_1m(day_idx, i, o, h, low, c))
        by_tf: dict[int, list[Bar]] = {60: bars_1m}
        for seconds in (180, 300, 600, 900, 1800, 3600, 14400):
            by_tf[seconds] = _aggregate(bars_1m, seconds)
        days.append(by_tf)
    return days


@pytest.fixture(scope="module")
def synthetic_chain():
    section = default_ifvg_smc_section()
    seed = None
    trace_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    day_funnels: dict[str, dict[str, int]] = {}
    chain_dates: list[str] = []
    core_offset = 0
    days = _three_days()
    for day_idx, by_tf in enumerate(days):
        trading_day = _DAY0 + timedelta(days=day_idx)
        date_str = trading_day.isoformat()
        chain_dates.append(date_str)
        result = run_day(
            by_tf,
            section=section,
            seed=seed,
            trading_day=trading_day,
            dataset_exhausted=day_idx == len(days) - 1,
            audit_capture_mode="fsm_audit_v1",
        )
        seed = result.end_seed
        frame = flatten_emissions(result.emissions, entering_seed_hash=None)
        if not frame.empty:
            frame["is_warmup"] = day_idx < 1
            frame["days_of_htf_history"] = day_idx
            frame["evaluation_config_hash"] = "test-eval-hash"
        audit_frame = flatten_audit_emissions(
            result.audit_emissions, entering_seed_hash=None
        )
        if not audit_frame.empty:
            audit_frame["is_warmup"] = day_idx < 1
            audit_frame["days_of_htf_history"] = day_idx
            audit_frame["evaluation_config_hash"] = "test-eval-hash"
            audit_frame["core_trace_ordinal_before_global"] = (
                audit_frame["stamp_core_trace_ordinal_before"].astype(int) + core_offset
            )
            audit_frame["core_trace_ordinal_after_global"] = (
                audit_frame["stamp_core_trace_ordinal_after"].astype(int) + core_offset
            )
        core_offset += len(frame)
        trace_frames.append(frame)
        audit_frames.append(audit_frame)
        day_funnels[date_str] = dict(result.funnel.counters)
    trace = pd.concat(trace_frames, ignore_index=True, sort=False)
    trace["trace_ordinal"] = range(len(trace))
    tables = assemble_fsm_audit_tables(
        trace=trace,
        audit_frames=audit_frames,
        day_funnels=day_funnels,
        chain_dates=tuple(chain_dates),
        warmup_days=1,
    )
    return tables, day_funnels


def test_tables_validate_and_reconcile_exactly(synthetic_chain) -> None:
    tables, day_funnels = synthetic_chain
    report = reconcile_funnel_to_audit(day_funnels, tables)
    assert report["passed"] and not report["failures"]
    # every populated table re-validates standalone.
    for table in AuditTable:
        frame = tables.get(table, pd.DataFrame())
        if not frame.empty:
            validate_audit_table(table, frame)
    assert not tables[AuditTable.HTF_TAP].empty
    assert not tables[AuditTable.FILL_EVENT].empty
    assert not tables[AuditTable.PARENTLESS_STEP].empty
    assert not tables[AuditTable.DAY_FUNNEL].empty


def test_trace_reuse_preserves_interleaving(synthetic_chain) -> None:
    tables, _ = synthetic_chain
    taps = tables[AuditTable.HTF_TAP]
    # trace ordinals are the SAME global ordinals the v2 tables carry — strictly
    # increasing and unique across the reused tables.
    ordinals = pd.concat(
        [
            tables[table]["trace_ordinal"]
            for table in (
                AuditTable.HTF_TAP,
                AuditTable.PARENT_CANDIDATE,
                AuditTable.OPPOSING,
                AuditTable.PARENT_LOCK,
                AuditTable.INVERSION,
                AuditTable.SETUP_RESOLUTION,
            )
            if not tables[table].empty
        ]
    )
    assert ordinals.is_unique
    assert not taps["tap_cursor"].isna().any()


def test_audit_channel_ordering_and_brackets(synthetic_chain) -> None:
    tables, _ = synthetic_chain
    fills = tables[AuditTable.FILL_EVENT]
    assert fills["audit_trace_ordinal"].is_unique
    before = fills["core_trace_ordinal_before_global"].astype(int)
    after = fills["core_trace_ordinal_after_global"].astype(int)
    assert (after == before + 1).all()
    # per-day audit_seq strictly monotone within each source day.
    for _, group in fills.groupby("envelope_trading_day"):
        seqs = group["stamp_audit_seq"].astype(int).tolist()
        assert seqs == sorted(seqs)


def test_parentless_interval_derivation(synthetic_chain) -> None:
    tables, day_funnels = synthetic_chain
    steps = tables[AuditTable.PARENTLESS_STEP]
    intervals = tables[AuditTable.PARENTLESS_INTERVAL]
    assert not intervals.empty
    assert int(intervals["bars_count"].sum()) == int(len(steps))
    total_counter = sum(
        counters.get("parentless_window_live", 0) for counters in day_funnels.values()
    )
    assert int(intervals["bars_count"].sum()) == total_counter
    # ordinal-contiguity: each interval's span equals its bar count.
    spans = (
        intervals["last_step_ordinal"].astype(int)
        - intervals["first_step_ordinal"].astype(int)
        + 1
    )
    assert (spans == intervals["bars_count"].astype(int)).all()
    assert intervals["interval_id"].is_unique
