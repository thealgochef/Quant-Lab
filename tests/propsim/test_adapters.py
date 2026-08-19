"""Adapter tick→point/cost mapping + the sequence-pinning stream hash (§16.4)."""

from __future__ import annotations

import pytest

from alpha_lab.propsim.adapters import (
    account_trades_from_executed_frame,
    gross_trade_stream_hash,
    group_account_trades_by_day,
)
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    make_resolved_trades_frame,
)


def test_tick_to_point_and_cost_mapping_preserves_order_and_links() -> None:
    frame = make_resolved_trades_frame(SYNTHETIC_DAYS)
    trades = account_trades_from_executed_frame(
        frame, tick_size=0.25, cost_points_round_turn=0.5
    )
    assert len(trades) == len(frame)
    # entry-time order with trade-id ties (canonical, storage-order-free)
    entries = [(trade.entry_ts_utc, trade.trade_id) for trade in trades]
    assert entries == sorted(entries)
    first = trades[0]
    row = frame.sort_values(["entry_ts_utc", "trade_id"]).iloc[0]
    assert first.points == pytest.approx(float(row["realized_ticks"]) * 0.25 - 0.5)
    assert first.risk_points == pytest.approx(float(row["risk_ticks"]) * 0.25)
    assert first.mae_pts == pytest.approx(abs(float(row["mae_ticks"])) * 0.25)
    assert first.trade_id == str(row["trade_id"])
    assert first.decision_id == str(row["decision_id"])
    blocks = group_account_trades_by_day(trades)
    assert [day.isoformat() for day, _ in blocks] == list(SYNTHETIC_DAYS)
    assert sum(len(block) for _, block in blocks) == len(trades)


def test_stream_hash_pins_the_sequence_and_refuses_underspecified_frames() -> None:
    frame = make_resolved_trades_frame(SYNTHETIC_DAYS)
    digest = gross_trade_stream_hash(frame)
    # stable across a pure row shuffle (the canonical SEQUENCE is the pin)
    shuffled = frame.sample(frac=1.0, random_state=7)
    assert gross_trade_stream_hash(shuffled) == digest
    # any value change changes the digest
    changed = frame.copy()
    changed.loc[changed.index[0], "realized_ticks"] = 999
    assert gross_trade_stream_hash(changed) != digest
    # a different execution order (different entry time) changes the digest
    reordered = frame.copy()
    reordered.loc[reordered.index[0], "entry_ts_utc"] = "2026-01-15T23:00:00+00:00"
    assert gross_trade_stream_hash(reordered) != digest
    # a frame missing a required hash column is refused, never under-pinned
    with pytest.raises(ValueError, match="requires columns"):
        gross_trade_stream_hash(frame.drop(columns=["direction"]))
