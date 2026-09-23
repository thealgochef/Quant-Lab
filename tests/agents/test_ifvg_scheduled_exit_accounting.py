"""Time exits are actual partial outcomes, with legacy barrier rules intact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (
    compute_strategy_metrics,
    per_trade_net_r,
)
from alpha_lab.agents.data_infra.ifvg.trade_stats import (
    _streaks,
    _validate_and_normalize_executed_trades,
    compute_trade_stats,
)
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame


def scheduled_trade(direction="LONG", realized=12):
    frame = make_resolved_trades_frame(("2026-01-13",), trades_per_day=1)
    entry = int(frame.entry_ticks.iloc[0])
    side = 1 if direction == "LONG" else -1
    return frame.assign(
        direction=direction,
        resolution="scheduled_close",
        realized_ticks=realized,
        entry_ts_utc="2026-01-13T21:50:00Z",
        resolution_ts_utc="2026-01-13T21:55:00Z",
        stop_ticks=entry - side * 40,
        target_ticks=entry + side * 40,
        exit_ticks=entry + side * realized,
        scheduled_exit_deadline_ts_utc="2026-01-13T21:55:00Z",
        scheduled_exit_schedule_id="frozen-calendar-deadline-2026-01-13",
        mfe_ticks=max(realized, 0),
        mae_ticks=min(realized, 0),
    )


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
@pytest.mark.parametrize("realized,label", [(12, "win"), (-8, "loss"), (0, "flat")])
def test_scheduled_partial_outcome_and_one_cost(direction, realized, label):
    frame = scheduled_trade(direction, realized)
    normalized = _validate_and_normalize_executed_trades(frame, tick_size=0.25)
    assert normalized._label.iloc[0] == label
    assert normalized._gross_r.iloc[0] == realized / 40
    net = per_trade_net_r(frame, cost_points=0.514)
    assert net.iloc[0] == pytest.approx((realized * 0.25 - 0.514) / 10)
    stats = compute_trade_stats(frame, cost_points=0.514, evaluation_config_hash="a" * 64)
    assert stats["n"] == 1
    trade = stats["trades"][0]
    assert trade["net_r"] == pytest.approx(net.iloc[0])
    assert trade["pnl_usd"] == pytest.approx((realized * 0.25 - 0.514) * 20)


@pytest.mark.parametrize(
    "field", ["exit_ticks", "scheduled_exit_deadline_ts_utc", "scheduled_exit_schedule_id"]
)
def test_scheduled_exit_requires_actual_price_and_deadline_evidence(field):
    with pytest.raises(ValueError, match="priced deadline evidence"):
        _validate_and_normalize_executed_trades(
            scheduled_trade().drop(columns=field), tick_size=0.25
        )


@pytest.mark.parametrize("price", [np.nan, np.inf, 80000.5, 79960, 80040])
def test_missing_off_grid_and_barrier_crossed_scheduled_prices_fail(price):
    with pytest.raises(ValueError, match="scheduled_close"):
        _validate_and_normalize_executed_trades(
            scheduled_trade().assign(exit_ticks=price), tick_size=0.25
        )


@pytest.mark.parametrize(
    "deadline", ["2026-01-13T21:54:00Z", "2026-01-13T21:56:00Z", "2026-01-13T21:55:00", None]
)
def test_no_late_backdated_naive_or_missing_deadline(deadline):
    frame = scheduled_trade().assign(scheduled_exit_deadline_ts_utc=deadline)
    with pytest.raises(ValueError, match="scheduled_close"):
        _validate_and_normalize_executed_trades(frame, tick_size=0.25)


def test_scheduled_realized_value_cannot_be_forced_to_target_or_stop():
    with pytest.raises(ValueError, match="realized_ticks"):
        _validate_and_normalize_executed_trades(
            scheduled_trade().assign(realized_ticks=40), tick_size=0.25
        )


def test_legacy_barrier_validation_and_costed_metrics_stay_exact():
    legacy = make_resolved_trades_frame(("2026-01-13",))
    before = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: legacy},
        cost_points=0.514,
        evaluation_config_hash="a" * 64,
    )
    priced = legacy.assign(
        exit_ticks=np.where(legacy.resolution.eq("stop"), legacy.stop_ticks, legacy.target_ticks)
    )
    after = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: priced},
        cost_points=0.514,
        evaluation_config_hash="a" * 64,
    )
    assert before.model_dump() == after.model_dump()
    with pytest.raises(ValueError, match="barrier"):
        _validate_and_normalize_executed_trades(priced.assign(exit_ticks=80001), tick_size=0.25)
    with pytest.raises(ValueError, match="barrier"):
        _validate_and_normalize_executed_trades(legacy.assign(realized_ticks=12), tick_size=0.25)


def test_duplicate_exit_is_rejected_before_costs_and_flat_breaks_streak():
    with pytest.raises(ValueError, match="not unique"):
        _validate_and_normalize_executed_trades(pd.concat([scheduled_trade()] * 2), tick_size=0.25)
    assert _streaks(pd.Series(["win", "flat", "win"]))["max_consecutive_wins"] == 1


def test_priced_projection_retains_exit_evidence_and_preserves_old_schema(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        EXECUTED_TRADE_TABLE_PROJECTION_ID,
        SCHEDULED_EXIT_PROJECTION_ID,
        build_executed_trade_table,
        executed_trade_table_id_for,
        load_executed_trade_table,
        save_executed_trade_table,
    )

    raw = scheduled_trade()
    core = "a" * 64
    with pytest.raises(ValueError, match="explicit priced-exit"):
        build_executed_trade_table(core, raw, record_schema_version=2)
    envelope, payload = build_executed_trade_table(
        core,
        raw,
        record_schema_version=2,
        projection_id=SCHEDULED_EXIT_PROJECTION_ID,
    )
    assert envelope.executed_trade_table_id == executed_trade_table_id_for(
        core,
        record_schema_version=2,
        projection_id=SCHEDULED_EXIT_PROJECTION_ID,
    )
    assert envelope.executed_trade_table_id != executed_trade_table_id_for(
        core,
        record_schema_version=2,
        projection_id=EXECUTED_TRADE_TABLE_PROJECTION_ID,
    )
    save_executed_trade_table(tmp_path, envelope, payload)
    saved = load_executed_trade_table(tmp_path, envelope.executed_trade_table_id).frame
    assert saved.exit_ticks.iloc[0] == raw.exit_ticks.iloc[0]
    assert saved.scheduled_exit_schedule_id.iloc[0] == raw.scheduled_exit_schedule_id.iloc[0]
    pd.testing.assert_series_equal(
        per_trade_net_r(raw, cost_points=0.514),
        per_trade_net_r(saved, cost_points=0.514),
    )


def test_frozen_profile_selects_priced_projection_even_with_zero_trades():
    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        SCHEDULED_EXIT_PROJECTION_ID,
        build_executed_trade_table,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import _execution_projection_for

    spec = SimpleNamespace(section_overrides={"holding_policy": "scheduled_daily_close_v1"})
    assert _execution_projection_for(spec) == SCHEDULED_EXIT_PROJECTION_ID
    empty = make_resolved_trades_frame(("2026-01-13",)).iloc[:0]
    envelope, _ = build_executed_trade_table(
        "a" * 64,
        empty,
        record_schema_version=2,
        projection_id=_execution_projection_for(spec),
    )
    assert envelope.row_count == 0
