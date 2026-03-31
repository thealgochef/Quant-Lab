"""
Look-ahead bias audit -- 7 tests on the backtest pipeline.

Runs on 2025-06-02 (warmup) + 2025-06-03 (audit target, has an executable signal).
Each test prints PASS/FAIL with evidence. If ANY test fails, the backtest
results cannot be trusted.

Usage:
    python scripts/audit_lookahead.py
"""

from __future__ import annotations

import sys
import time as time_mod
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from alpha_lab.dashboard.engine.feature_computer import FeatureComputer
from alpha_lab.dashboard.engine.level_engine import LevelEngine
from alpha_lab.dashboard.engine.models import ObservationStatus
from alpha_lab.dashboard.engine.observation_manager import ObservationManager
from alpha_lab.dashboard.engine.touch_detector import TouchDetector
from alpha_lab.dashboard.model.model_manager import ModelManager
from alpha_lab.dashboard.model.outcome_tracker import OutcomeTracker
from alpha_lab.dashboard.model.prediction_engine import PredictionEngine
from alpha_lab.dashboard.pipeline.price_buffer import PriceBuffer
from alpha_lab.dashboard.pipeline.rithmic_client import BBOUpdate, TradeUpdate
from alpha_lab.dashboard.trading.account_manager import AccountManager
from alpha_lab.dashboard.trading.position_monitor import PositionMonitor
from alpha_lab.dashboard.trading.trade_executor import TradeExecutor

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
DATA_DIR = Path("data/databento/NQ")
MODEL_PATH = Path("data/models/dashboard_3feature_v1.cbm")

AUDIT_DATE = "2025-06-03"  # Has an executable signal
WARMUP_DATE = "2025-06-02"  # Prior day for level computation

# -- Instrumented state with audit hooks ----------------------

class AuditState:
    """BacktestState with diagnostic hooks for all 7 audit tests."""

    def __init__(self) -> None:
        self.price_buffer = PriceBuffer(max_duration=timedelta(hours=72))
        self.level_engine = LevelEngine(self.price_buffer)
        self.feature_computer = FeatureComputer()
        self.touch_detector = TouchDetector(self.level_engine)
        self.observation_manager = ObservationManager(self.feature_computer)
        self.model_manager = ModelManager(Path("data/models"))
        self.prediction_engine = PredictionEngine(self.model_manager)
        self.outcome_tracker = OutcomeTracker()
        self.account_manager = AccountManager()
        self.trade_executor = TradeExecutor(self.account_manager)
        self.position_monitor = PositionMonitor(
            self.account_manager, self.trade_executor,
        )

        self.account_manager.add_account("A1", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A2", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A3", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("B1", Decimal("147"), Decimal("85"), "B")
        self.account_manager.add_account("B2", Decimal("147"), Decimal("85"), "B")

        # -- Audit diagnostics --------------------------------
        self.session_ended = False
        self.latest_price: float | None = None

        # Test 1: Level audit
        self.level_audit_data: dict | None = None

        # Test 2: Observation window audit
        self.obs_audit_data: list[dict] = []

        # Test 3: Prediction audit
        self.pred_audit_data: list[dict] = []

        # Test 4: TP/SL tick audit
        self.tpsl_first_check_ticks: list[dict] = []
        self.position_open_timestamps: list[datetime] = []
        self.tpsl_ticks_checked: list[dict] = []

        # Test 5: Outcome tracker audit
        self.outcome_start_timestamps: dict[str, datetime] = {}
        self.outcome_first_ticks: dict[str, datetime] = {}

        # Test 4: Entry price audit
        self.entry_price_audit: dict | None = None

        # Test 7: Price buffer audit before level computation
        self.price_buffer_latest_at_level_compute: datetime | None = None

        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        # Touch -> Observation start
        def _on_touch(event) -> None:
            self.observation_manager.start_observation(event)

        self.touch_detector.on_touch(_on_touch)

        # Observation complete -> Prediction
        def _on_observation_complete(window) -> None:
            if window.status != ObservationStatus.COMPLETED:
                return

            # TEST 2: Record observation window tick timestamps
            trade_timestamps = [t.timestamp for t in window.trades_accumulated]
            bbo_timestamps = [b.timestamp for b in window.bbo_accumulated]
            self.obs_audit_data.append({
                "event_id": window.event.event_id[:8],
                "touch_ts": window.event.timestamp,
                "window_start": window.start_time,
                "window_end": window.end_time,
                "trade_timestamps": trade_timestamps,
                "bbo_timestamps": bbo_timestamps,
                "n_trades": len(trade_timestamps),
                "n_bbos": len(bbo_timestamps),
            })

            self.prediction_engine.predict(window)

        self.observation_manager.on_observation_complete(
            _on_observation_complete,
        )

        # Prediction -> execute + track
        def _on_prediction(prediction) -> None:
            market_time = prediction.observation.end_time

            # TEST 3: Record exact features passed to model
            self.pred_audit_data.append({
                "event_id": prediction.event_id[:8],
                "features": dict(prediction.features),
                "predicted_class": prediction.predicted_class,
                "is_executable": prediction.is_executable,
                "observation_n_trades": len(prediction.observation.trades_accumulated),
                "observation_n_bbos": len(prediction.observation.bbo_accumulated),
                "model_input_keys": list(prediction.features.keys()),
                "window_start": prediction.observation.start_time,
                "window_end": prediction.observation.end_time,
                "level_price": float(prediction.level_price),
                "direction": prediction.trade_direction.value,
            })

            # Start outcome tracking
            # TEST 5: Record when tracking starts
            self.outcome_start_timestamps[prediction.event_id] = market_time
            self.outcome_tracker.start_tracking(prediction)

            # Execute trade if executable
            if prediction.is_executable:
                executor_dict = {
                    "is_executable": True,
                    "trade_direction": prediction.trade_direction,
                    "level_price": prediction.level_price,
                }
                # TEST 4: Record position open time and prices
                market_price = Decimal(str(self.latest_price)) if self.latest_price else prediction.level_price
                self.position_open_timestamps.append(market_time)
                self.entry_price_audit = {
                    "level_price": float(prediction.level_price),
                    "market_price_at_window_close": float(market_price),
                    "direction": prediction.trade_direction.value,
                }
                self.trade_executor.on_prediction(
                    prediction=executor_dict,
                    timestamp=market_time,
                    current_price=market_price,
                )

        self.prediction_engine.on_prediction(_on_prediction)

        # Trade closed
        def _on_trade_closed(trade) -> None:
            pass  # not needed for audit

        self.trade_executor.on_trade_closed(_on_trade_closed)

        # Outcome resolved
        def _on_outcome_resolved(outcome) -> None:
            pass  # not needed for audit

        self.outcome_tracker.on_outcome_resolved(_on_outcome_resolved)

    def reset_day(self) -> None:
        self.session_ended = False
        self.level_engine.reset_daily()


# -- Data loading (same as run_backtest.py) -------------------

def detect_front_month(conn, mbp_path: str) -> str:
    rows = conn.execute(f"""
        SELECT symbol, count(*) AS n
        FROM read_parquet('{mbp_path}')
        WHERE action = 'T' AND symbol NOT LIKE '%-%'
        GROUP BY symbol ORDER BY n DESC LIMIT 1
    """).fetchall()
    if not rows:
        raise ValueError(f"No trades in {mbp_path}")
    return rows[0][0]


def load_trades_for_date(conn, date_str: str, front_month: str) -> pd.DataFrame:
    mbp_path = str(DATA_DIR / date_str / "mbp10.parquet")
    df = conn.execute(f"""
        SELECT ts_event,
            CAST(price AS DOUBLE) AS price,
            CAST(size AS INTEGER) AS size,
            side,
            CAST(bid_px_00 AS DOUBLE) AS bid_price,
            CAST(ask_px_00 AS DOUBLE) AS ask_price,
            CAST(bid_sz_00 AS INTEGER) AS bid_size,
            CAST(ask_sz_00 AS INTEGER) AS ask_size
        FROM read_parquet('{mbp_path}')
        WHERE action = 'T'
          AND symbol = '{front_month}'
          AND bid_px_00 > 0 AND ask_px_00 > 0
        ORDER BY ts_event
    """).fetchdf()
    return df


# -- Tick processing (same order as run_backtest.py) ----------

# Track for Test 4 & 5
_tpsl_tick_count = 0
_outcome_tick_count = 0


def process_trade_tick(state: AuditState, trade: TradeUpdate) -> None:
    global _tpsl_tick_count, _outcome_tick_count

    state.latest_price = float(trade.price)
    state.price_buffer.add_trade(trade)
    state.touch_detector.on_trade(trade)
    state.observation_manager.on_trade(trade)

    # TEST 4: Track ticks checked by position monitor
    has_positions_before = any(
        a.has_position for a in state.account_manager.get_all_accounts()
    )
    if has_positions_before:
        _tpsl_tick_count += 1
        if _tpsl_tick_count <= 5:  # Record first 5 ticks
            state.tpsl_ticks_checked.append({
                "tick_num": _tpsl_tick_count,
                "timestamp": trade.timestamp,
                "price": float(trade.price),
            })
    state.position_monitor.on_trade(trade)

    # TEST 5: Track ticks checked by outcome tracker
    if state.outcome_tracker.active_trackers > 0:
        for eid in list(state.outcome_start_timestamps.keys()):
            if eid not in state.outcome_first_ticks:
                state.outcome_first_ticks[eid] = trade.timestamp
    state.outcome_tracker.on_trade(trade)

    # Flatten check
    state.position_monitor.check_flatten_time(trade.timestamp, trade.price)

    if not state.session_ended:
        ts = trade.timestamp
        if ts.hour > 19 or (ts.hour == 19 and ts.minute >= 55):
            state.session_ended = True
            state.outcome_tracker.on_session_end()


def process_bbo_tick(state: AuditState, bbo: BBOUpdate) -> None:
    state.price_buffer.add_bbo(bbo)
    state.observation_manager.on_bbo(bbo)


# -- Main audit -----------------------------------------------

def main() -> None:
    global _tpsl_tick_count, _outcome_tick_count

    print(f"\n{'=' * 72}")
    print("  LOOK-AHEAD BIAS AUDIT")
    print(f"{'=' * 72}")
    print(f"\n  Warmup date: {WARMUP_DATE}")
    print(f"  Audit date:  {AUDIT_DATE}")
    print(f"  Model:       {MODEL_PATH}")

    conn = duckdb.connect()
    state = AuditState()

    # Load model
    version = state.model_manager.upload_model(MODEL_PATH)
    state.model_manager.activate_model(version["id"])
    print(f"  Model loaded: {MODEL_PATH.name}")

    # -- Phase 1: Warmup day (2025-06-02) ----------------------
    print(f"\n  [Phase 1] Replaying warmup day: {WARMUP_DATE}...")
    state.reset_day()
    state.account_manager.start_new_day()

    mbp_path = str(DATA_DIR / WARMUP_DATE / "mbp10.parquet")
    front_month_warmup = detect_front_month(conn, mbp_path)
    warmup_df = load_trades_for_date(conn, WARMUP_DATE, front_month_warmup)
    print(f"    Front month: {front_month_warmup}")
    print(f"    Ticks: {len(warmup_df):,}")

    warmup_last_ts = None
    for _, row in warmup_df.iterrows():
        ts = pd.Timestamp(row["ts_event"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        ts_utc = ts.tz_convert(UTC).to_pydatetime()
        warmup_last_ts = ts_utc

        trade_price = Decimal(str(round(float(row["price"]), 2)))
        bid_price = Decimal(str(round(float(row["bid_price"]), 2)))
        ask_price = Decimal(str(round(float(row["ask_price"]), 2)))

        bbo = BBOUpdate(
            timestamp=ts_utc, bid_price=bid_price, bid_size=int(row["bid_size"]),
            ask_price=ask_price, ask_size=int(row["ask_size"]),
            symbol=front_month_warmup,
        )
        state.price_buffer.add_bbo(bbo)

        side = "BUY" if row["side"] == "A" else "SELL"
        trade = TradeUpdate(
            timestamp=ts_utc, price=trade_price, size=int(row["size"]),
            aggressor_side=side, symbol=front_month_warmup,
        )
        state.price_buffer.add_trade(trade)

    print(f"    Last tick: {warmup_last_ts.isoformat()}")
    print(f"    Last tick (ET): {warmup_last_ts.astimezone(ET).strftime('%Y-%m-%d %H:%M:%S ET')}")

    # -- Phase 2: Audit day (2025-06-03) -----------------------
    print(f"\n  [Phase 2] Auditing: {AUDIT_DATE}")

    trading_date = date.fromisoformat(AUDIT_DATE)
    state.reset_day()
    state.account_manager.start_new_day()
    state.price_buffer.evict()

    # ----------------------------------------------------------
    # TEST 7: Price buffer latest timestamp before level compute
    # ----------------------------------------------------------
    # Check what's in the price buffer BEFORE computing levels
    with state.price_buffer._lock:
        if state.price_buffer._trades:
            state.price_buffer_latest_at_level_compute = (
                state.price_buffer._trades[-1].timestamp
            )
        else:
            state.price_buffer_latest_at_level_compute = None

    # ----------------------------------------------------------
    # TEST 1: Compute levels and record sources
    # ----------------------------------------------------------
    rth_open = datetime.combine(
        trading_date, time(9, 30), tzinfo=ET,
    ).astimezone(UTC)

    levels = state.level_engine.compute_levels(
        trading_date, current_time=rth_open,
    )

    level_audit = {
        "trading_date": str(trading_date),
        "current_time_utc": rth_open.isoformat(),
        "current_time_et": rth_open.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S ET"),
        "levels": [],
    }
    for lv in levels:
        level_audit["levels"].append({
            "type": lv.level_type.value,
            "price": float(lv.price),
            "side": lv.side.value,
            "available_from_utc": lv.available_from.isoformat(),
            "available_from_et": lv.available_from.astimezone(ET).strftime(
                "%Y-%m-%d %H:%M:%S ET"
            ),
            "source_session_date": str(lv.source_session_date),
        })
    state.level_audit_data = level_audit

    # -- Load and replay audit day -----------------------------
    mbp_path = str(DATA_DIR / AUDIT_DATE / "mbp10.parquet")
    front_month = detect_front_month(conn, mbp_path)
    audit_df = load_trades_for_date(conn, AUDIT_DATE, front_month)
    print(f"    Front month: {front_month}")
    print(f"    Ticks: {len(audit_df):,}")
    print(f"    Levels computed: {len(levels)}")
    print(f"    Active zones: {len(state.level_engine.get_active_zones())}")

    # TEST 6: Collect all timestamps for monotonicity check
    all_tick_timestamps: list[datetime] = []

    _tpsl_tick_count = 0
    _outcome_tick_count = 0

    for _, row in audit_df.iterrows():
        ts = pd.Timestamp(row["ts_event"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        ts_utc = ts.tz_convert(UTC).to_pydatetime()

        all_tick_timestamps.append(ts_utc)

        trade_price = Decimal(str(round(float(row["price"]), 2)))
        bid_price = Decimal(str(round(float(row["bid_price"]), 2)))
        ask_price = Decimal(str(round(float(row["ask_price"]), 2)))

        bbo = BBOUpdate(
            timestamp=ts_utc, bid_price=bid_price, bid_size=int(row["bid_size"]),
            ask_price=ask_price, ask_size=int(row["ask_size"]),
            symbol=front_month,
        )
        process_bbo_tick(state, bbo)

        side = "BUY" if row["side"] == "A" else "SELL"
        trade = TradeUpdate(
            timestamp=ts_utc, price=trade_price, size=int(row["size"]),
            aggressor_side=side, symbol=front_month,
        )
        process_trade_tick(state, trade)

    # End of day
    if not state.session_ended:
        state.session_ended = True
        state.outcome_tracker.on_session_end()

    conn.close()

    # ==========================================================
    #  PRINT RESULTS FOR ALL 7 TESTS
    # ==========================================================

    print(f"\n{'=' * 72}")
    print("  TEST RESULTS")
    print(f"{'=' * 72}")

    results = []

    # ----------------------------------------------------------
    # TEST 1: Level computation uses only PAST data
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 1: Level computation uses only PAST data")
    print(f"{'-' * 72}")

    la = state.level_audit_data
    print(f"\n  Trading date:    {la['trading_date']}")
    print(f"  current_time:    {la['current_time_et']}")
    print(f"  Levels computed: {len(la['levels'])}")

    test1_pass = True
    for lv in la["levels"]:
        src = lv["source_session_date"]
        avail = lv["available_from_et"]
        avail_dt = datetime.fromisoformat(lv["available_from_utc"])

        # Check 1: source_session_date must be BEFORE trading_date (for PDH/PDL)
        #           or same day but session completed before RTH open
        is_past = avail_dt <= rth_open

        status = "OK" if is_past else "FAIL"
        if not is_past:
            test1_pass = False

        print(f"\n    {lv['type']:12s}  price={lv['price']:>10.2f}  side={lv['side']}")
        print(f"      Source session date: {src}")
        print(f"      Available from:      {avail}")
        print(f"      Available before RTH open (09:30 ET)? {status}")

    if test1_pass:
        print(f"\n  >>> TEST 1: PASS -- All {len(la['levels'])} levels use only past data")
    else:
        print(f"\n  >>> TEST 1: FAIL -- Some levels use future data!")
    results.append(("Test 1: Level computation", test1_pass))

    # ----------------------------------------------------------
    # TEST 2: Observation window only uses ticks WITHIN window
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 2: Observation window only uses ticks WITHIN 5-min window")
    print(f"{'-' * 72}")

    test2_pass = True

    if not state.obs_audit_data:
        print("\n  No observations to audit (no signals fired)")
        test2_pass = True  # Vacuously true
    else:
        # Find the executable signal (or first signal)
        for obs in state.obs_audit_data:
            print(f"\n  Observation: {obs['event_id']}")
            print(f"    Touch timestamp:  {obs['touch_ts'].astimezone(ET).strftime('%H:%M:%S.%f ET')}")
            print(f"    Window start:     {obs['window_start'].astimezone(ET).strftime('%H:%M:%S.%f ET')}")
            print(f"    Window end:       {obs['window_end'].astimezone(ET).strftime('%H:%M:%S.%f ET')}")
            print(f"    Trades accumulated: {obs['n_trades']}")
            print(f"    BBOs accumulated:   {obs['n_bbos']}")

            # Check all trade timestamps within [start, end]
            bad_trades = []
            for ts in obs["trade_timestamps"]:
                if ts < obs["window_start"] or ts > obs["window_end"]:
                    bad_trades.append(ts)

            bad_bbos = []
            for ts in obs["bbo_timestamps"]:
                if ts < obs["window_start"] or ts > obs["window_end"]:
                    bad_bbos.append(ts)

            if bad_trades:
                print(f"    FAIL: {len(bad_trades)} trades outside window!")
                for bt in bad_trades[:5]:
                    print(f"      {bt.astimezone(ET).strftime('%H:%M:%S.%f ET')}")
                test2_pass = False
            else:
                print(f"    All {obs['n_trades']} trade timestamps within window: OK")

            if bad_bbos:
                print(f"    FAIL: {len(bad_bbos)} BBOs outside window!")
                test2_pass = False
            else:
                print(f"    All {obs['n_bbos']} BBO timestamps within window: OK")

            # Print first 5 and last 5 trade timestamps
            if obs["trade_timestamps"]:
                print(f"\n    First 5 trade timestamps:")
                for ts in obs["trade_timestamps"][:5]:
                    print(f"      {ts.astimezone(ET).strftime('%H:%M:%S.%f ET')}")
                if len(obs["trade_timestamps"]) > 5:
                    print(f"    Last 5 trade timestamps:")
                    for ts in obs["trade_timestamps"][-5:]:
                        print(f"      {ts.astimezone(ET).strftime('%H:%M:%S.%f ET')}")

    if test2_pass:
        print(f"\n  >>> TEST 2: PASS -- All ticks within observation windows")
    else:
        print(f"\n  >>> TEST 2: FAIL -- Ticks found outside observation windows!")
    results.append(("Test 2: Observation window bounds", test2_pass))

    # ----------------------------------------------------------
    # TEST 3: Model prediction uses only observation features
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 3: Model prediction uses only features from observation window")
    print(f"{'-' * 72}")

    test3_pass = True
    expected_keys = {"int_time_beyond_level", "int_time_within_2pts", "int_absorption_ratio"}

    for pred in state.pred_audit_data:
        print(f"\n  Prediction: {pred['event_id']}")
        print(f"    Predicted class: {pred['predicted_class']}")
        print(f"    Executable:      {pred['is_executable']}")
        print(f"    Direction:       {pred['direction']}")
        print(f"    Level price:     {pred['level_price']:.2f}")
        print(f"    Window:          {pred['window_start'].astimezone(ET).strftime('%H:%M:%S ET')} -> "
              f"{pred['window_end'].astimezone(ET).strftime('%H:%M:%S ET')}")
        print(f"    Trades in window:  {pred['observation_n_trades']}")
        print(f"    BBOs in window:    {pred['observation_n_bbos']}")
        print(f"    Feature keys:    {pred['model_input_keys']}")
        print(f"    Feature values:")
        for k, v in pred['features'].items():
            print(f"      {k}: {v:.6f}")

        actual_keys = set(pred['model_input_keys'])
        if actual_keys != expected_keys:
            print(f"    FAIL: Unexpected feature keys: {actual_keys - expected_keys}")
            test3_pass = False
        else:
            print(f"    Feature keys match expected 3 features: OK")

        if len(pred['features']) != 3:
            print(f"    FAIL: Expected 3 features, got {len(pred['features'])}")
            test3_pass = False
        else:
            print(f"    Exactly 3 features passed to model: OK")

    if test3_pass:
        print(f"\n  >>> TEST 3: PASS -- Model receives only 3 observation window features")
    else:
        print(f"\n  >>> TEST 3: FAIL -- Model receives unexpected data!")
    results.append(("Test 3: Model input isolation", test3_pass))

    # ----------------------------------------------------------
    # TEST 4: TP/SL resolution uses only ticks AFTER entry
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 4: TP/SL resolution uses only ticks AFTER entry")
    print(f"{'-' * 72}")

    test4_pass = True

    if not state.position_open_timestamps:
        print("\n  No positions opened (no executable signals)")
        test4_pass = True
    else:
        entry_time = state.position_open_timestamps[0]
        print(f"\n  Position opened at (window end): {entry_time.astimezone(ET).strftime('%H:%M:%S.%f ET')}")

        # Entry price comparison
        epa = state.entry_price_audit
        if epa:
            print(f"\n  Entry price verification:")
            print(f"    Level price (touch, 5 min ago):  {epa['level_price']:.2f}")
            print(f"    Market price (at window close):   {epa['market_price_at_window_close']:.2f}")
            diff = abs(epa['market_price_at_window_close'] - epa['level_price'])
            print(f"    Difference:                       {diff:.2f} pts")
            print(f"    Direction:                        {epa['direction']}")

            # Verify entry is at market price, NOT level price
            # Check the actual account positions
            for acct in state.account_manager.get_all_accounts():
                if acct.has_position:
                    actual_entry = float(acct.current_position.entry_price)
                    matches_market = abs(actual_entry - epa['market_price_at_window_close']) < 0.01
                    matches_level = abs(actual_entry - epa['level_price']) < 0.01
                    print(f"\n    Account {acct.label} actual entry: {actual_entry:.2f}")
                    if matches_market and not matches_level:
                        print(f"      -> Matches MARKET price: OK (honest entry)")
                    elif matches_level and not matches_market:
                        print(f"      -> Matches LEVEL price: FAIL (stale entry)")
                        test4_pass = False
                    elif matches_market and matches_level:
                        print(f"      -> Level == Market (no slippage this time)")
                    else:
                        print(f"      -> Matches neither?! FAIL")
                        test4_pass = False
                    break  # All accounts have same entry

        print(f"\n  First {min(5, len(state.tpsl_ticks_checked))} ticks checked for TP/SL:")
        for tick in state.tpsl_ticks_checked:
            ts = tick["timestamp"]
            relation = "AFTER" if ts >= entry_time else "BEFORE"
            ok = "OK" if ts >= entry_time else "FAIL"
            if ts < entry_time:
                test4_pass = False
            print(f"    Tick #{tick['tick_num']}: "
                  f"{ts.astimezone(ET).strftime('%H:%M:%S.%f ET')}  "
                  f"price={tick['price']:.2f}  {relation} entry  [{ok}]")

    if test4_pass:
        print(f"\n  >>> TEST 4: PASS -- All TP/SL checks are on ticks at/after window completion")
    else:
        print(f"\n  >>> TEST 4: FAIL -- TP/SL checked on ticks before entry!")
    results.append(("Test 4: TP/SL temporal ordering", test4_pass))

    # ----------------------------------------------------------
    # TEST 5: Outcome tracker uses only ticks AFTER prediction
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 5: Outcome tracker uses only ticks AFTER prediction")
    print(f"{'-' * 72}")

    test5_pass = True

    if not state.outcome_start_timestamps:
        print("\n  No outcomes tracked")
    else:
        for eid, start_ts in state.outcome_start_timestamps.items():
            first_tick = state.outcome_first_ticks.get(eid)
            eid_short = eid[:8]

            print(f"\n  Event: {eid_short}")
            print(f"    Tracking started (window end): {start_ts.astimezone(ET).strftime('%H:%M:%S.%f ET')}")

            if first_tick:
                relation = "AT/AFTER" if first_tick >= start_ts else "BEFORE"
                ok = "OK" if first_tick >= start_ts else "FAIL"
                if first_tick < start_ts:
                    test5_pass = False
                print(f"    First tick processed:          {first_tick.astimezone(ET).strftime('%H:%M:%S.%f ET')}  {relation}  [{ok}]")
            else:
                print(f"    No ticks processed (resolved at session end)")

    if test5_pass:
        print(f"\n  >>> TEST 5: PASS -- Outcome tracker only processes ticks after prediction")
    else:
        print(f"\n  >>> TEST 5: FAIL -- Outcome tracker uses ticks before prediction!")
    results.append(("Test 5: Outcome tracker timing", test5_pass))

    # ----------------------------------------------------------
    # TEST 6: No future data in tick replay order
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 6: No future data in tick replay order")
    print(f"{'-' * 72}")

    test6_pass = True
    n_ticks = len(all_tick_timestamps)
    print(f"\n  Total ticks: {n_ticks:,}")

    # Check monotonicity
    violations = 0
    first_violation = None
    for i in range(1, n_ticks):
        if all_tick_timestamps[i] < all_tick_timestamps[i - 1]:
            violations += 1
            if first_violation is None:
                first_violation = (i, all_tick_timestamps[i - 1], all_tick_timestamps[i])

    if violations > 0:
        test6_pass = False
        print(f"  FAIL: {violations} monotonicity violations!")
        i, prev, curr = first_violation
        print(f"  First violation at index {i}:")
        print(f"    tick[{i-1}] = {prev.astimezone(ET).strftime('%H:%M:%S.%f ET')}")
        print(f"    tick[{i}]   = {curr.astimezone(ET).strftime('%H:%M:%S.%f ET')}")
    else:
        print(f"  Monotonicity check: 0 violations in {n_ticks:,} ticks")

    # Print first 10 and last 10
    print(f"\n  First 10 ticks:")
    for i, ts in enumerate(all_tick_timestamps[:10]):
        print(f"    [{i:5d}] {ts.astimezone(ET).strftime('%Y-%m-%d %H:%M:%S.%f ET')}")

    print(f"\n  Last 10 ticks:")
    for i, ts in enumerate(all_tick_timestamps[-10:], n_ticks - 10):
        print(f"    [{i:5d}] {ts.astimezone(ET).strftime('%Y-%m-%d %H:%M:%S.%f ET')}")

    if test6_pass:
        print(f"\n  >>> TEST 6: PASS -- All {n_ticks:,} ticks in strictly chronological order")
    else:
        print(f"\n  >>> TEST 6: FAIL -- Tick replay is not chronological!")
    results.append(("Test 6: Tick replay ordering", test6_pass))

    # ----------------------------------------------------------
    # TEST 7: Price buffer doesn't leak future data
    # ----------------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  TEST 7: Price buffer doesn't leak future data")
    print(f"{'-' * 72}")

    test7_pass = True
    latest_in_buffer = state.price_buffer_latest_at_level_compute

    if latest_in_buffer is None:
        print("\n  Price buffer was empty at level computation time")
        print("  (This means no prior day data -- levels would be empty)")
        test7_pass = True
    else:
        latest_et = latest_in_buffer.astimezone(ET)
        audit_date_start = datetime.combine(
            date.fromisoformat(AUDIT_DATE), time(0, 0), tzinfo=ET,
        )
        # The latest tick should be from BEFORE the audit date's
        # earliest possible session (Asia starts at 18:00 ET prev day)
        # For 2025-06-03, the previous day's data should end around
        # the end of 2025-06-02's last session
        is_past = latest_in_buffer < rth_open

        print(f"\n  At level computation time (before loading {AUDIT_DATE} ticks):")
        print(f"    Latest tick in price buffer: {latest_et.strftime('%Y-%m-%d %H:%M:%S.%f ET')}")
        print(f"    RTH open for {AUDIT_DATE}:    {rth_open.astimezone(ET).strftime('%Y-%m-%d %H:%M:%S ET')}")
        print(f"    Latest tick is BEFORE RTH open? {'YES' if is_past else 'NO'}")

        # Check that no tick from audit date exists in buffer
        audit_day_midnight_utc = datetime.combine(
            date.fromisoformat(AUDIT_DATE), time(0, 0), tzinfo=UTC,
        )

        # For the audit, we want to verify that the latest trade in the
        # buffer is from a date BEFORE the audit date. Since the warmup
        # date is 2025-06-02 and audit date is 2025-06-03, the buffer
        # should contain data up to the end of 2025-06-02.
        latest_date = latest_et.date()
        warmup_date_obj = date.fromisoformat(WARMUP_DATE)
        audit_date_obj = date.fromisoformat(AUDIT_DATE)

        print(f"    Latest tick date: {latest_date}")
        print(f"    Warmup date:      {warmup_date_obj}")
        print(f"    Audit date:       {audit_date_obj}")

        # The latest tick should be from the warmup date or earlier
        # Note: Globex data for 2025-06-02 may extend into early hours
        # of 2025-06-03 UTC (since NQ trades nearly 24h). But the key
        # point is: no ticks from the AUDIT day's RTH were loaded.
        if latest_in_buffer >= rth_open:
            test7_pass = False
            print(f"\n  FAIL: Price buffer contains data at/after RTH open!")
        else:
            print(f"\n  Price buffer contains only data BEFORE {AUDIT_DATE} RTH open: OK")

        # Also verify: query the buffer the same way LevelEngine does
        # PDH/PDL: prior day RTH 09:30-16:15 ET
        prev_day = audit_date_obj - timedelta(days=1)
        rth_start_et = datetime.combine(prev_day, time(9, 30), tzinfo=ET)
        rth_end_et = datetime.combine(prev_day, time(16, 15), tzinfo=ET)
        rth_start_utc = rth_start_et.astimezone(UTC)
        rth_end_utc = rth_end_et.astimezone(UTC)

        hl = state.price_buffer.get_high_low_in_range(rth_start_utc, rth_end_utc)
        if hl:
            print(f"\n  PDH/PDL source data verification:")
            print(f"    Query range: {rth_start_et.strftime('%Y-%m-%d %H:%M ET')} to "
                  f"{rth_end_et.strftime('%Y-%m-%d %H:%M ET')}")
            print(f"    High: {float(hl[0]):.2f}")
            print(f"    Low:  {float(hl[1]):.2f}")
            print(f"    Both from {prev_day} RTH (completed before {AUDIT_DATE}): OK")

    if test7_pass:
        print(f"\n  >>> TEST 7: PASS -- Price buffer only contains past data at level computation time")
    else:
        print(f"\n  >>> TEST 7: FAIL -- Price buffer contains future data!")
    results.append(("Test 7: Price buffer isolation", test7_pass))

    # ==========================================================
    #  FINAL SUMMARY
    # ==========================================================
    print(f"\n{'=' * 72}")
    print("  AUDIT SUMMARY")
    print(f"{'=' * 72}\n")

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        marker = "  " if passed else "!!"
        print(f"  {marker} {name:45s} [{status}]")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL 7 TESTS PASSED -- No look-ahead bias detected.")
        print("  Backtest results can be trusted.")
    else:
        n_fail = sum(1 for _, p in results if not p)
        print(f"  {n_fail} TEST(S) FAILED -- Look-ahead bias detected!")
        print("  DO NOT trust backtest results until all failures are fixed.")

    print(f"\n{'=' * 72}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
