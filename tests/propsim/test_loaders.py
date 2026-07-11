"""Loader tests over synthetic on-disk fixtures mirroring the real writers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.propsim.loaders import (
    load_executions_trades,
    load_journal_trades,
    load_oos_trades,
    trading_day_for,
)

PRED_A = "aaaaaaaa-0000-0000-0000-000000000001"
PRED_B = "bbbbbbbb-0000-0000-0000-000000000002"
PRED_C = "cccccccc-0000-0000-0000-000000000003"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _executions_fixture(root: Path) -> tuple[Path, Path]:
    """One closed trade (A), one dangling open (B), resets; journal outcome for A."""
    executions = root / "executions"
    journal = root / "journal"
    _write_jsonl(
        executions / "2026-01-05.jsonl",
        [
            {"type": "reset", "mode": "replay", "ts_utc": "2026-01-05T08:35:19+00:00",
             "reason": "replay_reset", "cleared": 0, "cleared_prediction_ids": []},
            {"type": "open", "mode": "replay", "bundle_id": "B",
             "ts_utc": "2026-01-05T14:30:00+00:00", "prediction_id": PRED_A,
             "touch_id": "t-1", "direction": "long", "contracts": 1,
             "point_value": 20.0, "tick_size": 0.25, "entry_price": 15000.0,
             "entry_price_conservative": 15000.25, "tp_price": 15015.0,
             "sl_price": 14985.0, "session": "ny", "level_kind": "pdl",
             "source": "open_setups"},
            {"type": "close", "mode": "replay", "bundle_id": "B",
             "ts_utc": "2026-01-05T14:45:00+00:00", "prediction_id": PRED_A,
             "touch_id": "t-1", "reason": "tp_hit", "direction": "long",
             "contracts": 1, "point_value": 20.0,
             "entry_ts_utc": "2026-01-05T14:30:00+00:00", "entry_price": 15000.0,
             "entry_price_conservative": 15000.25, "exit_price": 15015.0,
             "exit_price_conservative": 15015.0, "points": 15.0,
             "points_conservative": 14.75, "dollars": 300.0,
             "dollars_conservative": 295.0, "session": "ny", "level_kind": "pdl"},
            {"type": "open", "mode": "replay", "bundle_id": "B",
             "ts_utc": "2026-01-05T15:00:00+00:00", "prediction_id": PRED_B,
             "touch_id": "t-2", "direction": "short", "contracts": 1,
             "point_value": 20.0, "tick_size": 0.25, "entry_price": 15020.0,
             "entry_price_conservative": 15019.75, "tp_price": 15005.0,
             "sl_price": 15035.0, "session": "ny", "level_kind": "pdh",
             "source": "open_setups"},
        ],
    )
    _write_jsonl(
        journal / "2026-01-05.jsonl",
        [
            {"type": "prediction", "mode": "replay", "bundle_id": "B",
             "ts_utc": "2026-01-05T14:29:00+00:00", "prediction_id": PRED_A,
             "touch_id": "t-1", "predicted_class": "tradeable_reversal",
             "is_eligible": True, "direction": "long", "session": "ny",
             "level_kind": "pdl"},
            {"type": "outcome", "mode": "replay", "bundle_id": "B",
             "ts_utc": "2026-01-05T14:45:00+00:00", "outcome_id": "o-1",
             "prediction_id": PRED_A, "touch_id": "t-1",
             "resolution_type": "tp_hit", "actual_class": "tradeable_reversal",
             "predicted_class": "tradeable_reversal", "correct": True,
             "max_mfe_pts": 17.5, "max_mae_pts": 2.25, "bars_to_resolution": 3,
             "entry_price": 15000.0},
        ],
    )
    return executions, journal


def test_executions_loader_joins_journal_excursions(tmp_path):
    executions, journal = _executions_fixture(tmp_path)
    loaded = load_executions_trades(executions, journal_dir=journal)
    assert loaded.source == "executions"
    assert len(loaded.trades) == 1  # the dangling open never completed
    trade = loaded.trades[0]
    assert trade.points_optimistic == 15.0
    assert trade.points_conservative == 14.75
    assert trade.mfe_pts == 17.5
    assert trade.mae_pts == 2.25
    assert trade.resolution == "tp_hit"
    assert trade.day.isoformat() == "2026-01-05"
    assert trade.entry_ts == datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    assert loaded.excursions_available
    assert loaded.degradation_reason is None
    assert any("opens without a close" in note and note.endswith(": 1") for note in loaded.notes)


def test_executions_loader_without_journal_degrades(tmp_path):
    executions, _ = _executions_fixture(tmp_path)
    loaded = load_executions_trades(executions, journal_dir=None)
    assert len(loaded.trades) == 1
    assert loaded.trades[0].mae_pts is None
    assert not loaded.excursions_available
    assert "MFE/MAE unavailable" in (loaded.degradation_reason or "")


def test_executions_loader_empty_dir_yields_zero_trades(tmp_path):
    executions = tmp_path / "executions"
    executions.mkdir()
    _write_jsonl(
        executions / "undated.jsonl",
        [{"type": "reset", "mode": "unknown", "ts_utc": None,
          "reason": "live_reset", "cleared": 0, "cleared_prediction_ids": []}],
    )
    loaded = load_executions_trades(executions, journal_dir=None)
    assert loaded.trades == []
    assert not loaded.excursions_available


def test_trading_day_roll_dates_evening_entries_to_the_next_day():
    # 23:30 UTC on Jan 5 = 18:30 ET -> trading day 2026-01-06.
    assert (
        trading_day_for(datetime(2026, 1, 5, 23, 30, tzinfo=UTC)).isoformat()
        == "2026-01-06"
    )
    # 22:59 UTC on Jan 5 = 17:59 ET -> still 2026-01-05.
    assert (
        trading_day_for(datetime(2026, 1, 5, 22, 59, tzinfo=UTC)).isoformat()
        == "2026-01-05"
    )


def test_executions_loader_dedups_replay_rerun_closes(tmp_path):
    """PROPSIM-L2: re-running a replay appends the SAME physical trade under a
    fresh uuid — the loader dedups by fill signature and surfaces the count."""
    executions, journal = _executions_fixture(tmp_path)
    day_file = executions / "2026-01-05.jsonl"
    rows = [json.loads(line) for line in day_file.read_text(encoding="utf-8").splitlines()]
    rerun = [dict(row) for row in rows]
    for row in rerun:
        if row.get("prediction_id"):
            row["prediction_id"] = row["prediction_id"].replace("0000", "1111")
    with day_file.open("a", encoding="utf-8") as handle:
        for row in rerun:
            handle.write(json.dumps(row) + "\n")

    loaded = load_executions_trades(executions, journal_dir=journal)
    assert len(loaded.trades) == 1
    assert any(
        "duplicate closes dropped" in note and note.endswith(": 1")
        for note in loaded.notes
    )


def test_executions_loader_skips_non_finite_points(tmp_path):
    """PS-1 boundary: NaN points on a close row is unparseable, not a trade."""
    executions = tmp_path / "executions"
    _write_jsonl(
        executions / "2026-01-05.jsonl",
        [
            {"type": "close", "ts_utc": "2026-01-05T14:45:00+00:00",
             "prediction_id": PRED_A, "reason": "tp_hit",
             "entry_ts_utc": "2026-01-05T14:30:00+00:00",
             "points": float("nan"), "points_conservative": float("nan")},
        ],
    )
    loaded = load_executions_trades(executions, journal_dir=None)
    assert loaded.trades == []
    assert any("unparseable" in note and note.endswith(": 1") for note in loaded.notes)


def test_journal_loader_dedups_warm_restart_duplicate_outcomes(tmp_path):
    """PROPSIM-L1: warm restarts re-predict the same physical touch under fresh
    ids — outcomes dedup by touch signature (last-write-wins), count surfaced."""
    journal = tmp_path / "journal"
    touch = {
        "ts_utc": "2026-06-16T08:10:55.294315+00:00",
        "level_kind": "asia_high",
        "level_price_ticks": 123440,
        "direction": "short",
    }
    rows = []
    for suffix in ("1", "2", "3"):
        pid = f"dddddddd-0000-0000-0000-00000000000{suffix}"
        rows.append({"type": "prediction", "prediction_id": pid,
                     "is_eligible": False, **touch})
        rows.append({"type": "outcome", "ts_utc": "2026-06-16T08:20:00+00:00",
                     "prediction_id": pid, "resolution_type": "tp_hit",
                     "max_mfe_pts": 16.0, "max_mae_pts": 3.0,
                     "entry_price": 30860.0})
    # A genuinely distinct touch on the same day survives.
    rows.append({"type": "prediction", "prediction_id": PRED_B,
                 "ts_utc": "2026-06-16T10:00:00+00:00", "level_kind": "pdl",
                 "level_price_ticks": 123000, "direction": "long",
                 "is_eligible": False})
    rows.append({"type": "outcome", "ts_utc": "2026-06-16T10:15:00+00:00",
                 "prediction_id": PRED_B, "resolution_type": "sl_hit",
                 "max_mfe_pts": 2.0, "max_mae_pts": 15.0, "entry_price": 30750.0})
    _write_jsonl(journal / "2026-06-16.jsonl", rows)

    loaded = load_journal_trades(journal, tp_points=15.0, sl_points=15.0)
    assert len(loaded.trades) == 2  # 4 outcomes -> 1 deduped touch + 1 distinct
    assert any(
        "duplicate outcomes dropped" in note and note.endswith(": 2")
        for note in loaded.notes
    )


def test_journal_loader_treats_outcomes_as_trades(tmp_path):
    journal = tmp_path / "journal"
    _write_jsonl(
        journal / "2026-01-05.jsonl",
        [
            {"type": "prediction", "ts_utc": "2026-01-05T14:29:00+00:00",
             "prediction_id": PRED_A, "is_eligible": False, "direction": "short"},
            {"type": "outcome", "ts_utc": "2026-01-05T14:45:00+00:00",
             "prediction_id": PRED_A, "resolution_type": "tp_hit",
             "max_mfe_pts": 17.5, "max_mae_pts": 2.25, "entry_price": 15000.0},
            # An outcome with no prediction row: dated by its resolved ts.
            {"type": "outcome", "ts_utc": "2026-01-05T15:45:00+00:00",
             "prediction_id": PRED_C, "resolution_type": "sl_hit",
             "max_mfe_pts": 3.0, "max_mae_pts": 15.0, "entry_price": 15100.0},
        ],
    )
    loaded = load_journal_trades(journal, tp_points=15.0, sl_points=15.0)
    assert loaded.source == "journal"
    assert len(loaded.trades) == 2
    winner, loser = loaded.trades
    assert winner.points_optimistic == 15.0
    assert winner.points_conservative == 14.75  # entry 1 tick adverse
    assert winner.entry_ts == datetime(2026, 1, 5, 14, 29, tzinfo=UTC)  # prediction ts
    assert loser.points_optimistic == -15.0
    assert loser.points_conservative == -15.5  # + entry tick + SL exit tick
    assert loser.entry_ts == datetime(2026, 1, 5, 15, 45, tzinfo=UTC)  # outcome ts
    assert loaded.excursions_available
    assert any("INELIGIBLE" in note for note in loaded.notes)


def _oos_frame(with_outcome_columns: bool) -> pd.DataFrame:
    rows = {
        "fold": [0, 0, 1, 1],
        "timestamp": pd.to_datetime(
            [
                "2025-11-21 09:31:00",
                "2025-11-21 10:02:00",
                "2025-11-24 09:45:00",
                "2025-11-24 13:10:00",
            ]
        ).tz_localize("US/Eastern"),
        "session": ["ny", "ny", "ny", "ny"],
        "label": [
            "tradeable_reversal",
            "trap_reversal",
            "aggressive_blowthrough",
            "tradeable_reversal",
        ],
        "prob_tradeable_reversal": [0.8, 0.4, 0.2, 0.75],
        "gate_0_70_runtime_sessions": [True, False, False, True],
    }
    if with_outcome_columns:
        rows["max_mfe_pts"] = [17.5, 4.0, 2.0, 16.0]
        rows["max_mae_pts"] = [2.25, 15.0, 15.0, 5.0]
        rows["entry_price"] = [15000.0, 15010.0, 15020.0, 15030.0]
        rows["resolution_type"] = ["tp_hit", "sl_hit", "sl_hit", "tp_hit"]
    return pd.DataFrame(rows)


def test_oos_loader_post_p1_columns(tmp_path):
    path = tmp_path / "oos_predictions.parquet"
    _oos_frame(with_outcome_columns=True).to_parquet(path, index=False)
    loaded = load_oos_trades(path, tp_points=15.0, sl_points=15.0)
    assert loaded.source == "oos"
    assert len(loaded.trades) == 4
    assert loaded.excursions_available
    assert loaded.degradation_reason is None
    first = loaded.trades[0]
    assert first.points_optimistic == 15.0
    assert first.points_conservative == 15.0  # idealized: EQUALS optimistic
    assert first.mfe_pts == 17.5
    assert first.mae_pts == 2.25
    assert first.day.isoformat() == "2025-11-21"
    losses = [t for t in loaded.trades if t.resolution == "sl_hit"]
    assert all(t.points_optimistic == -15.0 for t in losses)


def test_oos_loader_pre_p1_degrades_to_realized_only(tmp_path):
    path = tmp_path / "oos_predictions.parquet"
    _oos_frame(with_outcome_columns=False).to_parquet(path, index=False)
    loaded = load_oos_trades(path, tp_points=15.0, sl_points=15.0)
    assert len(loaded.trades) == 4
    assert not loaded.excursions_available
    assert "pre-P1" in (loaded.degradation_reason or "")
    assert all(t.mae_pts is None for t in loaded.trades)
    # resolution still derives from the label via the ratified mapping.
    assert [t.resolution for t in loaded.trades] == [
        "tp_hit", "sl_hit", "sl_hit", "tp_hit",
    ]


def test_oos_loader_gate_column_filters(tmp_path):
    path = tmp_path / "oos_predictions.parquet"
    _oos_frame(with_outcome_columns=True).to_parquet(path, index=False)
    loaded = load_oos_trades(
        path, tp_points=15.0, sl_points=15.0,
        gate_column="gate_0_70_runtime_sessions",
    )
    assert len(loaded.trades) == 2
    assert all(t.resolution == "tp_hit" for t in loaded.trades)


def test_oos_loader_unknown_gate_column_fails_loud(tmp_path):
    path = tmp_path / "oos_predictions.parquet"
    _oos_frame(with_outcome_columns=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="gate column 'nope'"):
        load_oos_trades(path, tp_points=15.0, sl_points=15.0, gate_column="nope")
