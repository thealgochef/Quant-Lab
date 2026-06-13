"""W1 P4d: end-to-end stream labeling through the SC drive on real data.

Real-data tests, each capped at a ONE-HOUR slice of one trading day (the W1
test policy). They prove the production labeling path — the canonical SC day
stream driving the same StrategyRuntime/TouchReversalPlugin wiring Trade-Lab
serves with — produces touches, features, and labels end to end, with the
dataset schema intact and the legacy decision stages absent from the call
graph; plus the zero-touch no-op (W3A-READER P0).

Skips cleanly when the Databento NQ store is absent (CI without the data mount).
"""

from __future__ import annotations

import math
import os
import sys
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ENV_DATABENTO_DIR = "QUANT_LAB_DATABENTO_DIR"
_DEFAULT_DATABENTO_DIR = Path(__file__).resolve().parents[2] / "data" / "databento"


def _resolve_databento_dir() -> Path:
    override = os.environ.get(ENV_DATABENTO_DIR)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DATABENTO_DIR


DATA_DIR = _resolve_databento_dir()
SYMBOL = "NQ"
DAY = "2026-02-18"
PREV_DAY = "2026-02-17"

_HAVE_DATA = (DATA_DIR / SYMBOL / DAY).is_dir() and (DATA_DIR / SYMBOL / PREV_DAY).is_dir()

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason=f"Databento NQ store with {PREV_DAY}/{DAY} not present"
)

LEGACY_MODULE = "alpha_lab.agents.data_infra.ml.legacy_decision"


def _hour_one_cap_utc():
    """End of the trading day's FIRST hour: prev-day 19:00 ET as UTC."""
    et = ZoneInfo("US/Eastern")
    from datetime import datetime, time

    prev = date.fromisoformat(PREV_DAY)
    return datetime.combine(prev, time(19, 0), tzinfo=et).astimezone(ZoneInfo("UTC"))


def test_stream_drive_labels_one_hour_slice() -> None:
    from strategy_core import Trade
    from strategy_core.data.databento_parquet import DatabentoParquetSource

    from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
    from alpha_lab.agents.data_infra.ml.engine_decision import process_single_date_stream

    cap_utc = _hour_one_cap_utc()

    # Pre-scan the hour's trade range so the seeded PDH straddles traded prices —
    # guaranteeing at least one touch without assuming anything about the day.
    low = math.inf
    high = -math.inf
    source = DatabentoParquetSource.for_trading_day(DATA_DIR / SYMBOL, date.fromisoformat(DAY))
    for event in source.events():
        if isinstance(event, Trade):
            if event.event_ts_utc > cap_utc:
                break
            points = event.price_points(0.25)
            low = min(low, points)
            high = max(high, points)
    assert high > low, "one-hour slice carried no trades"
    mid_on_grid = round(((low + high) / 2) * 4) / 4

    config = DashboardUtilityConfig(
        bar_type="147t",
        interaction_window_minutes=5,
        tp_points=2.0,  # small barriers so the one-hour forward window resolves
        sl_points=2.0,
        trap_mfe_min=1.0,
        include_approach_features=True,
        approach_window_minutes=15,
        level_proximity_pts=0.5,
    )
    # Seed PDH at the hour's midpoint (PDL far below): available from the day
    # start, so the touch fires inside the slice. Pop the legacy module first so
    # the post-drive assertion is a TRUE call-graph check even when another test
    # in the session (the parity regression) already imported it.
    removed_legacy = sys.modules.pop(LEGACY_MODULE, None)
    try:
        df = process_single_date_stream(
            DAY,
            DATA_DIR,
            SYMBOL,
            config,
            prev_day_hl=(mid_on_grid, low - 50.0),
            events_until_utc=cap_utc,
        )
        # The W1 P4b guarantee: the legacy decision stages are NOT in the call graph.
        assert LEGACY_MODULE not in sys.modules
    finally:
        if removed_legacy is not None:
            sys.modules[LEGACY_MODULE] = removed_legacy

    assert not df.empty, "stream drive produced no labeled rows in the slice"
    expected_columns = {
        "event_ts",
        "date",
        "timestamp",
        "session",
        "decision_time",
        "label_window_end",
        "direction",
        "representative_price",
        "level_type",
        "label",
        "label_encoded",
        "max_mfe",
        "max_mae",
        "int_time_beyond_level",
        "int_time_within_2pts",
        "int_absorption_ratio",
        "app_large_trade_vol_pct",
        "app_avg_trade_size",
        "app_max_spread",
    }
    assert expected_columns <= set(df.columns)
    assert (df["date"] == DAY).all()
    assert set(df["direction"]) <= {"LONG", "SHORT"}
    assert df["label"].notna().all()


def test_stream_drive_zero_touch_day_is_empty_noop() -> None:
    """W3A-READER P0: a day whose levels are never approached yields an empty
    frame without error — the vectorized approach-quote pass must stay a no-op
    on zero touches (its window arrays would be empty and the hull min()/max()
    would raise if entered)."""
    from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
    from alpha_lab.agents.data_infra.ml.engine_decision import process_single_date_stream

    config = DashboardUtilityConfig(
        bar_type="147t",
        interaction_window_minutes=5,
        tp_points=2.0,
        sl_points=2.0,
        trap_mfe_min=1.0,
        include_approach_features=True,  # the guarded pass must be configured ON
        approach_window_minutes=15,
    )
    # Seed PDH/PDL absurdly far above any price in the frozen store (NQ never
    # trades near 200k points): levels EXIST but cannot be touched, so the
    # zero-touch path runs with the approach pass enabled.
    df = process_single_date_stream(
        DAY,
        DATA_DIR,
        SYMBOL,
        config,
        prev_day_hl=(200_000.0, 199_900.0),
        events_until_utc=_hour_one_cap_utc(),
    )
    assert df.empty


def test_build_utility_dataset_refuses_legacy_mode() -> None:
    from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import build_utility_dataset

    with pytest.raises(ValueError, match="legacy"):
        build_utility_dataset([DAY], DATA_DIR, MLPipelineConfig(), use_engine=False)
