"""Tests for dashboard utility mode (+15/-30)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import label_event_15_30
from alpha_lab.agents.data_infra.ml.model_evaluator import ModelEvaluator


def _ticks_from_prices(prices: list[float], start: str = "2026-02-20 09:30:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(prices), freq="1s", tz="UTC")
    arr = np.array(prices, dtype=float)
    return pd.DataFrame({
        "ts_event": ts,
        "price": arr,
        "bid_px_00": arr - 0.25,
        "ask_px_00": arr + 0.25,
        "size": np.ones(len(arr)),
    })


def test_label_event_15_30_long_tp_first():
    event_ts = pd.Timestamp("2026-02-20 09:30:00", tz="UTC")
    ticks = _ticks_from_prices([100.0, 104.0, 108.0, 115.5, 114.0])
    out = label_event_15_30(
        direction="LONG",
        event_ts=event_ts,
        forward_ticks=ticks,
        tick_size=1.0,
        tp_ticks=10,
        sl_ticks=30,
    )
    assert out["label_15_30"] == 1
    assert out["outcome_type"] == "tp"


def test_label_event_15_30_short_sl_first():
    event_ts = pd.Timestamp("2026-02-20 09:30:00", tz="UTC")
    ticks = _ticks_from_prices([100.0, 105.0, 115.0, 130.5])
    out = label_event_15_30(
        direction="SHORT",
        event_ts=event_ts,
        forward_ticks=ticks,
        tick_size=1.0,
        tp_ticks=10,
        sl_ticks=20,
    )
    assert out["label_15_30"] == 0
    assert out["outcome_type"] == "sl"


def test_utility_summary_oos_metrics():
    evaluator = ModelEvaluator(n_bootstrap=20, n_permutations=20)
    folds = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 1, 0]),
            "y_pred": np.array([1, 1, 0, 0]),
            "y_prob": np.array([0.9, 0.8, 0.4, 0.2]),
        },
        {
            "fold": 1,
            "y_true": np.array([1, 1, 0, 0]),
            "y_pred": np.array([1, 0, 1, 0]),
            "y_prob": np.array([0.7, 0.3, 0.6, 0.1]),
        },
    ]
    s = evaluator.summarize_utility_15_30_from_oos(folds, tp_points=15.0, sl_points=30.0)
    assert "win_rate_15_30" in s
    assert "expectancy_points_15_30" in s
    assert "profit_factor_15_30" in s
    assert s["n_samples"] == 8


def test_utility_summary_session_filter_ny_rth():
    evaluator = ModelEvaluator(n_bootstrap=20, n_permutations=20)
    folds = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 1, 0]),
            "y_pred": np.array([1, 1, 0, 0]),
            "y_prob": np.array([0.9, 0.8, 0.4, 0.2]),
            "session": np.array(["ny_rth", "ny_rth", "eth", "eth"]),
        },
    ]
    s = evaluator.summarize_utility_15_30_from_oos(
        folds, tp_points=15.0, sl_points=30.0, session_filter="ny_rth",
    )
    assert s["session_filter"] == "ny_rth"
    assert s["n_samples"] == 2
    assert s["n_taken_trades"] == 2
