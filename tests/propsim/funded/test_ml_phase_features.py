"""Causal source poison and exact feature arithmetic at both decision clocks."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.ml_phase.features import build_features
from alpha_lab.propsim.funded.ml_phase.protocol import load_contracts


def inputs():
    start = datetime(2025, 10, 9, 14, tzinfo=UTC)
    bars = [
        SimpleNamespace(
            availability_ts_utc=start + timedelta(minutes=i),
            trading_day=date(2025, 10, 9),
            timeframe_ticks=60,
            close_ticks=400 + i,
            high_ticks=402 + i,
            low_ticks=398 + i,
            bar_id=f"bar{i}",
        )
        for i in range(16)
    ]
    stamp = to_ns(bars[-1].availability_ts_utc)
    structure = {
        "entry_price": 100.0,
        "stop_price": 90.0,
        "geometry_cursor": "exact-owner",
        "htf_timeframe": "3600",
        "parent_timeframe": "60",
        "parent_width_points": 2.0,
        "opposing_width_points": 3.0,
        "entry_gap_width_points": 1.0,
        "separation_points": 5.0,
        "parent_midpoint": 95.0,
        "lock_ns": stamp - 5 * 60_000_000_000,
        "inversion_ns": stamp - 60_000_000_000,
        "frozen_max_distance_points": 20.0,
    }
    return dict(
        job="ENTRY",
        decision_ns=stamp,
        entry_ns=stamp,
        decision_price=100.0,
        trading_day="2025-10-09",
        deadline_ns=stamp + 60 * 60_000_000_000,
        structure=structure,
        completed_bars=bars,
        context={"decision_time_utc": bars[-1].availability_ts_utc.isoformat()},
        definitions=load_contracts(
            Path(__file__).resolve().parents[3] / "docs/ifsm-mffu-ml-phase-v01"
        )["FEATURE_DEFINITIONS"]["rows"],
    )


def test_future_bars_cannot_change_features_or_provenance():
    args = inputs()
    first = build_features(**args)
    future = SimpleNamespace(**vars(args["completed_bars"][-1]))
    future.availability_ts_utc += timedelta(minutes=1)
    future.close_ticks = future.high_ticks = 1e20
    args["completed_bars"] = [*args["completed_bars"], future]
    assert build_features(**args) == first
    assert first["values"]["close_change_5_r"] == 0.125
    assert first["values"]["range_15_r"] == 0.45
    assert first["values"]["signed_efficiency_15"] == 1.0
    assert first["values"]["total_net_gamma"] is None
    assert first["values"]["gamma_state"] == "unknown"


def test_checkpoint_uses_earlier_completed_bars_not_developing_ohlc():
    args = inputs()
    args.update(
        job="CONTINUATION",
        decision_ns=args["decision_ns"] + 30_000_000_000,
        decision_price=110.25,
        checkpoint={
            "event_id": "ns-and-ordinal",
            "observed_min_price": 98.0,
            "target_price": 110.0,
        },
    )
    features = build_features(**args)
    assert features["values"]["minutes_entry_to_target"] == 0.5
    assert features["values"]["mae_so_far_r"] == 0.2
    assert features["values"]["checkpoint_overshoot_r"] == 0.025
    assert features["provenance"]["range_15_r"]["known_at_ns"] < args["decision_ns"]


def test_gaps_and_other_sessions_do_not_extend_bar_windows():
    args = inputs()
    args["completed_bars"].pop(-2)
    values = build_features(**args)["values"]
    assert values["close_change_5_r"] is None
    assert values["signed_efficiency_15"] is None


def test_vendor_independent_clocks_and_lexical_ties():
    args = inputs()
    args["context"].update(
        gamma={
            "status": "selected",
            "nominal_eligible_from_utc": "2025-10-09T03:00:00Z",
            "source_sha256": "gamma-source",
            "value": -2.0,
            "positive_run_age": None,
        },
        levels={
            "status": "selected",
            "nominal_eligible_from_utc": "2025-10-08T03:00:00Z",
            "level_set_id": "levels-source",
            "items": {
                "1D Min": {"price": 80.0, "gex": None},
                "1D Max": {"price": 120.0, "gex": None},
                "HVL": {"price": 95.0, "gex": 2.0},
                "Z": {"price": 105.0, "gex": 1000.0},
                "A": {"price": 105.0, "gex": 3.0},
            },
        },
    )
    values = build_features(**args)["values"]
    assert values["implied_move_points"] == 20.0
    assert values["nearest_upper_gex"] == 3.0
    assert values["positive_report_age"] == 0
    assert values["level_set_age_hours"] - values["gamma_report_age_hours"] == 24
    args["context"]["gamma"]["nominal_eligible_from_utc"] = "2025-10-10T03:00:00Z"
    with pytest.raises(ValueError, match="future feature source"):
        build_features(**args)
