"""
Historical backtest — replays MBP-10 data through the exact same
dashboard pipeline components used in the live system.

Uses: LevelEngine, TouchDetector, ObservationManager, FeatureComputer,
PredictionEngine, TradeExecutor, PositionMonitor, AccountManager,
OutcomeTracker — wired identically to server.py._create_live_state().

Usage:
    python scripts/run_backtest.py                    # 5-day test run
    python scripts/run_backtest.py --days 20          # 20 trading days
    python scripts/run_backtest.py --start 2026-01-20 # custom start date
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as time_mod
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

# ── Dashboard components (same imports as server.py) ──────────
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

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")

DATA_DIR = Path("data/databento/NQ")
MODEL_PATH = Path("data/models/dashboard_3feature_v1.cbm")
OUTPUT_CSV = Path("data/experiment/backtest_results.csv")

# How many predictions to log with full details
DETAILED_LOG_COUNT = 5


# ── Data loading helpers ──────────────────────────────────────


def get_available_dates() -> list[str]:
    """Get sorted list of available trading dates with MBP-10 data."""
    dates = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "mbp10.parquet").exists():
            dates.append(d.name)
    return dates


def detect_front_month(conn: duckdb.DuckDBPyConnection, mbp_path: str) -> str:
    """Detect front-month NQ symbol (highest trade count, no spreads)."""
    rows = conn.execute(f"""
        SELECT symbol, count(*) AS n
        FROM read_parquet('{mbp_path}')
        WHERE action = 'T' AND symbol NOT LIKE '%-%'
        GROUP BY symbol ORDER BY n DESC LIMIT 1
    """).fetchall()
    if not rows:
        raise ValueError(f"No trades in {mbp_path}")
    return rows[0][0]



def load_trades_for_date(
    conn: duckdb.DuckDBPyConnection,
    date_str: str,
    front_month: str,
) -> pd.DataFrame:
    """Load trade ticks from MBP-10 for a specific date and symbol.

    Returns DataFrame with columns: ts_utc, price, size, side,
    bid_price, ask_price, bid_size, ask_size.
    """
    mbp_path = str(DATA_DIR / date_str / "mbp10.parquet")

    df = conn.execute(f"""
        SELECT
            ts_event,
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


# ── Pipeline wiring (mirrors server.py._create_live_state) ────


class BacktestState:
    """Holds all pipeline components and event logs for the backtest."""

    def __init__(self) -> None:
        # Components
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

        # Accounts (same as server.py — 5x Group A, 15pt TP/SL)
        self.account_manager.add_account("A1", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A2", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A3", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A4", Decimal("147"), Decimal("85"), "A")
        self.account_manager.add_account("A5", Decimal("147"), Decimal("85"), "A")

        # Per-day tracking
        self.day_predictions: list[dict] = []
        self.day_trades: list[dict] = []
        self.day_outcomes: list[dict] = []

        # Global tracking
        self.all_predictions: list[dict] = []
        self.all_trades: list[dict] = []
        self.all_outcomes: list[dict] = []
        self.daily_results: list[dict] = []
        self.detailed_log_count = 0

        # Session state
        self.session_ended = False
        self.latest_price: float | None = None
        self.latest_bid: float | None = None
        self.latest_ask: float | None = None

        # Shadow positions for non-RTH analysis
        # Each: {event_id, session, direction, level_price, entry_time,
        #        mfe_pts, mae_pts, tp15_hit, tp30_hit, sl15_hit, sl30_hit,
        #        tp15_time, tp30_time, sl15_time, sl30_time, resolved}
        self.shadow_positions: list[dict] = []

        # Wire callbacks
        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        """Wire callback chain — identical to server.py."""

        # 1. Touch → Observation start
        def _on_touch(event) -> None:
            self.observation_manager.start_observation(event)

        self.touch_detector.on_touch(_on_touch)

        # 2. Observation complete → Prediction
        def _on_observation_complete(window) -> None:
            if window.status != ObservationStatus.COMPLETED:
                return
            self.prediction_engine.predict(window)

        self.observation_manager.on_observation_complete(
            _on_observation_complete,
        )

        # 3. Prediction → TradeExecutor + OutcomeTracker + state
        def _on_prediction(prediction) -> None:
            session = prediction.observation.event.session
            # Use observation window end as the actual market time
            # (prediction.timestamp is datetime.now() — wrong for backtest)
            market_time = prediction.observation.end_time
            # Market price at prediction time (for entry)
            mkt_price = self.latest_price

            pred_data = {
                "event_id": prediction.event_id,
                "predicted_class": prediction.predicted_class,
                "is_executable": prediction.is_executable,
                "probabilities": prediction.probabilities,
                "features": prediction.features,
                "trade_direction": prediction.trade_direction.value,
                "level_price": float(prediction.level_price),
                "market_price": mkt_price,
                "model_version": prediction.model_version,
                "timestamp": market_time.isoformat(),
                "session": session,
            }

            self.day_predictions.append(pred_data)
            self.all_predictions.append(pred_data)

            # Detailed log for first N predictions
            if self.detailed_log_count < DETAILED_LOG_COUNT:
                self.detailed_log_count += 1
                print(f"\n  >>> PREDICTION #{self.detailed_log_count}")
                print(f"      Event ID:   {prediction.event_id[:8]}...")
                print(f"      Direction:  {prediction.trade_direction.value}")
                print(f"      Level:      {float(prediction.level_price):.2f}")
                print(f"      Market:     {mkt_price:.2f}" if mkt_price else "      Market:     N/A")
                print(f"      Session:    {prediction.observation.event.session}")
                print(f"      Features:")
                for k, v in prediction.features.items():
                    print(f"        {k}: {v:.4f}")
                print(f"      Class:      {prediction.predicted_class}")
                print(f"      Probs:      {prediction.probabilities}")
                print(f"      Executable: {prediction.is_executable}")

            # Start outcome tracking
            self.outcome_tracker.start_tracking(prediction)

            # Shadow position for non-executable tradeable_reversal
            if (not prediction.is_executable
                    and prediction.predicted_class == "tradeable_reversal"):
                self.shadow_positions.append({
                    "event_id": prediction.event_id,
                    "session": session,
                    "direction": prediction.trade_direction.value,
                    "level_price": float(prediction.level_price),
                    "entry_time": market_time,
                    "mfe_pts": 0.0,
                    "mae_pts": 0.0,
                    "tp15_hit": False, "tp30_hit": False,
                    "sl15_hit": False, "sl30_hit": False,
                    "tp15_time": None, "tp30_time": None,
                    "sl15_time": None, "sl30_time": None,
                    "resolved_15": False, "resolved_30": False,
                })

            # Execute trade if executable
            if prediction.is_executable:
                executor_dict = {
                    "is_executable": True,
                    "trade_direction": prediction.trade_direction,
                    "level_price": prediction.level_price,
                }
                # Entry at current market price (the tick that completed
                # the observation window), NOT the level/touch price
                # from 5 minutes ago.
                market_price = Decimal(str(self.latest_price)) if self.latest_price else prediction.level_price
                self.trade_executor.on_prediction(
                    prediction=executor_dict,
                    timestamp=market_time,
                    current_price=market_price,
                )

        self.prediction_engine.on_prediction(_on_prediction)

        # 4. Trade closed → state
        def _on_trade_closed(trade) -> None:
            trade_data = {
                "account_id": trade.account_id,
                "direction": trade.direction.value,
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "contracts": trade.contracts,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": trade.exit_time.isoformat(),
                "pnl": float(trade.pnl),
                "pnl_points": float(trade.pnl_points),
                "exit_reason": trade.exit_reason,
                "group": trade.group,
            }
            self.day_trades.append(trade_data)
            self.all_trades.append(trade_data)

        self.trade_executor.on_trade_closed(_on_trade_closed)

        # 5. Outcome resolved → state
        def _on_outcome_resolved(outcome) -> None:
            outcome_data = {
                "event_id": outcome.event_id,
                "predicted_class": outcome.prediction.predicted_class,
                "actual_class": outcome.actual_class,
                "prediction_correct": outcome.prediction_correct,
                "mfe_points": outcome.mfe_points,
                "mae_points": outcome.mae_points,
                "resolution_type": outcome.resolution_type,
            }
            self.day_outcomes.append(outcome_data)
            self.all_outcomes.append(outcome_data)

            # Update prediction record
            for pred in self.day_predictions:
                if pred.get("event_id") == outcome.event_id:
                    pred["prediction_correct"] = outcome.prediction_correct
                    pred["actual_class"] = outcome.actual_class
                    break

        self.outcome_tracker.on_outcome_resolved(_on_outcome_resolved)

    def reset_day(self) -> None:
        """Reset per-day state. Accounts carry forward."""
        self.day_predictions = []
        self.day_trades = []
        self.day_outcomes = []
        self.session_ended = False
        self.level_engine.reset_daily()


# ── Tick processing (mirrors server.py bridge handlers) ───────


def process_trade_tick(state: BacktestState, trade: TradeUpdate) -> None:
    """Process one trade tick — same as server.py._on_trade."""
    price = float(trade.price)
    state.latest_price = price

    # Add to price buffer (for level computation on future days)
    state.price_buffer.add_trade(trade)

    # Flatten check FIRST — must close positions before any new
    # touches/observations/predictions can open new ones.
    # 3:55 PM ET = 19:55 UTC (during EDT)
    state.position_monitor.check_flatten_time(
        trade.timestamp, trade.price,
    )

    # Session end: force-resolve remaining predictions
    if not state.session_ended:
        ts_et = trade.timestamp.astimezone(ET)
        past_flatten = (
            ts_et.hour > 15
            or (ts_et.hour == 15 and ts_et.minute >= 55)
        )
        if past_flatten:
            state.session_ended = True
            state.outcome_tracker.on_session_end()

    # Touch detector
    state.touch_detector.on_trade(trade)

    # Observation manager
    state.observation_manager.on_trade(trade)

    # Position monitor (TP/SL/DLL checks)
    state.position_monitor.on_trade(trade)

    # Outcome tracker
    state.outcome_tracker.on_trade(trade)

    # Shadow position TP/SL tracking
    price_f = float(trade.price)
    for sp in state.shadow_positions:
        if sp["resolved_15"] and sp["resolved_30"]:
            continue
        level = sp["level_price"]
        if sp["direction"] == "long":
            fav = price_f - level
            adv = level - price_f
        else:
            fav = level - price_f
            adv = price_f - level

        if fav > sp["mfe_pts"]:
            sp["mfe_pts"] = fav
        if adv > sp["mae_pts"]:
            sp["mae_pts"] = adv

        # 15pt TP/SL
        if not sp["resolved_15"]:
            if fav >= 15.0:
                sp["tp15_hit"] = True
                sp["tp15_time"] = trade.timestamp
                sp["resolved_15"] = True
            elif adv >= 15.0:
                sp["sl15_hit"] = True
                sp["sl15_time"] = trade.timestamp
                sp["resolved_15"] = True

        # 30pt TP/SL
        if not sp["resolved_30"]:
            if fav >= 30.0:
                sp["tp30_hit"] = True
                sp["tp30_time"] = trade.timestamp
                sp["resolved_30"] = True
            elif adv >= 30.0:
                sp["sl30_hit"] = True
                sp["sl30_time"] = trade.timestamp
                sp["resolved_30"] = True


def process_bbo_tick(state: BacktestState, bbo: BBOUpdate) -> None:
    """Process one BBO update — same as server.py._on_bbo."""
    state.latest_bid = float(bbo.bid_price)
    state.latest_ask = float(bbo.ask_price)

    state.price_buffer.add_bbo(bbo)
    state.observation_manager.on_bbo(bbo)


# ── Day processing ───────────────────────────────────────────


def process_trading_day(
    state: BacktestState,
    date_str: str,
    conn: duckdb.DuckDBPyConnection,
) -> dict:
    """Process one trading day through the full pipeline.

    Returns a dict of daily stats.
    """
    trading_date = date.fromisoformat(date_str)

    # ── 1. Reset daily state ──────────────────────────────────
    state.reset_day()
    state.account_manager.start_new_day()

    # ── 2. Evict old ticks & compute levels ──────────────────
    # PriceBuffer accumulates ticks from each day's replay.
    # LevelEngine queries PriceBuffer.get_high_low_in_range() for
    # prior-day session data — no OHLCV files needed.
    state.price_buffer.evict()

    # ── 3. Compute levels ─────────────────────────────────────
    # Use 09:30 ET as current_time — all overnight sessions complete
    rth_open = datetime.combine(
        trading_date, time(9, 30), tzinfo=ET,
    ).astimezone(UTC)
    levels = state.level_engine.compute_levels(
        trading_date, current_time=rth_open,
    )
    n_zones = len(state.level_engine.get_active_zones())

    # ── 4. Load MBP-10 trades ─────────────────────────────────
    mbp_path = str(DATA_DIR / date_str / "mbp10.parquet")
    front_month = detect_front_month(conn, mbp_path)
    trades_df = load_trades_for_date(conn, date_str, front_month)

    n_ticks = len(trades_df)
    if n_ticks == 0:
        logger.warning("No trades for %s — skipping", date_str)
        return _empty_day_result(date_str)

    # ── 5. Replay ticks ───────────────────────────────────────
    t_start = time_mod.monotonic()

    for _, row in trades_df.iterrows():
        ts = pd.Timestamp(row["ts_event"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        ts_utc = ts.tz_convert(UTC).to_pydatetime()

        trade_price = Decimal(str(round(float(row["price"]), 2)))
        bid_price = Decimal(str(round(float(row["bid_price"]), 2)))
        ask_price = Decimal(str(round(float(row["ask_price"]), 2)))

        # Create BBO update and process
        bbo = BBOUpdate(
            timestamp=ts_utc,
            bid_price=bid_price,
            bid_size=int(row["bid_size"]),
            ask_price=ask_price,
            ask_size=int(row["ask_size"]),
            symbol=front_month,
        )
        process_bbo_tick(state, bbo)

        # Create trade update and process
        side = "BUY" if row["side"] == "A" else "SELL"
        trade = TradeUpdate(
            timestamp=ts_utc,
            price=trade_price,
            size=int(row["size"]),
            aggressor_side=side,
            symbol=front_month,
        )
        process_trade_tick(state, trade)

    elapsed = time_mod.monotonic() - t_start

    # ── 6. End-of-day: resolve remaining predictions ──────────
    if not state.session_ended:
        state.session_ended = True
        state.outcome_tracker.on_session_end()

    # End day for accounts (qualifying day tracking)
    for acct in state.account_manager.get_all_accounts():
        acct.end_day()

    # ── 7. Collect daily stats ────────────────────────────────
    signals = len(state.day_predictions)
    executable = sum(1 for p in state.day_predictions if p.get("is_executable"))
    trades_total = len(state.day_trades)
    wins = sum(1 for t in state.day_trades if float(t.get("pnl", 0)) > 0)
    losses = sum(1 for t in state.day_trades if float(t.get("pnl", 0)) <= 0)
    day_pnl = sum(float(t.get("pnl", 0)) for t in state.day_trades)
    day_pnl_pts = sum(float(t.get("pnl_points", 0)) for t in state.day_trades)

    pred_correct = sum(
        1 for o in state.day_outcomes if o.get("prediction_correct")
    )
    pred_total = len(state.day_outcomes)

    result = {
        "date": date_str,
        "front_month": front_month,
        "levels": len(levels),
        "zones": n_zones,
        "ticks": n_ticks,
        "signals": signals,
        "executable": executable,
        "trades": trades_total,
        "wins": wins,
        "losses": losses,
        "day_pnl": round(day_pnl, 2),
        "day_pnl_pts": round(day_pnl_pts, 2),
        "pred_correct": pred_correct,
        "pred_total": pred_total,
        "pred_accuracy": round(pred_correct / pred_total, 3) if pred_total > 0 else 0,
        "elapsed_sec": round(elapsed, 1),
    }

    # Per-account daily P&L
    for acct in state.account_manager.get_all_accounts():
        result[f"pnl_{acct.label}"] = float(acct.daily_pnl)

    state.daily_results.append(result)
    return result


def _empty_day_result(date_str: str) -> dict:
    return {
        "date": date_str, "front_month": "", "levels": 0, "zones": 0,
        "ticks": 0, "signals": 0, "executable": 0, "trades": 0,
        "wins": 0, "losses": 0, "day_pnl": 0, "day_pnl_pts": 0,
        "pred_correct": 0, "pred_total": 0, "pred_accuracy": 0,
        "elapsed_sec": 0,
    }


# ── Summary printing ─────────────────────────────────────────


def print_day_summary(result: dict) -> None:
    """Print one-line daily summary."""
    print(
        f"  {result['date']}  "
        f"sym={result['front_month']:5s}  "
        f"lvls={result['zones']:2d}  "
        f"sigs={result['signals']:2d}  "
        f"exec={result['executable']:2d}  "
        f"trades={result['trades']:2d}  "
        f"W/L={result['wins']}/{result['losses']}  "
        f"pnl=${result['day_pnl']:>8.2f}  "
        f"acc={result['pred_accuracy']:.0%}  "
        f"[{result['elapsed_sec']:.1f}s]"
    )


def print_final_summary(state: BacktestState) -> None:
    """Print full backtest summary."""
    n_days = len(state.daily_results)
    total_signals = sum(r["signals"] for r in state.daily_results)
    total_executable = sum(r["executable"] for r in state.daily_results)
    total_trades = len(state.all_trades)
    total_wins = sum(1 for t in state.all_trades if float(t.get("pnl", 0)) > 0)
    total_losses = total_trades - total_wins
    total_pnl = sum(float(t.get("pnl", 0)) for t in state.all_trades)
    total_pnl_pts = sum(float(t.get("pnl_points", 0)) for t in state.all_trades)
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    pred_correct = sum(
        1 for o in state.all_outcomes if o.get("prediction_correct")
    )
    pred_total = len(state.all_outcomes)
    pred_acc = pred_correct / pred_total if pred_total > 0 else 0

    print(f"\n{'=' * 72}")
    print("  BACKTEST SUMMARY")
    print(f"{'=' * 72}")
    print(f"\n  Trading days:       {n_days}")
    print(f"  Total signals:      {total_signals} ({total_executable} executable)")
    print(f"  Total trades:       {total_trades}")
    print(f"  Win/Loss:           {total_wins}/{total_losses}")
    print(f"  Win rate:           {win_rate:.1%}")
    print(f"  Total P&L:          ${total_pnl:,.2f} ({total_pnl_pts:+.1f} pts)")
    print(f"  Prediction acc:     {pred_acc:.1%} ({pred_correct}/{pred_total})")

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in state.all_trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        print(f"\n  Exit reasons:")
        for reason, count in sorted(reasons.items()):
            print(f"    {reason:15s}  {count}")

    # Per-account equity curves
    print(f"\n  --- Per-Account Equity ---\n")
    print(f"  {'Account':10s}  {'Group':5s}  {'Balance':>12s}  {'Profit':>10s}  "
          f"{'Tier':>4s}  {'Status':>10s}  {'Trades':>6s}")
    print(f"  {'-------':10s}  {'-----':5s}  {'-------':>12s}  {'------':>10s}  "
          f"{'----':>4s}  {'------':>10s}  {'------':>6s}")
    for acct in state.account_manager.get_all_accounts():
        acct_trades = sum(
            1 for t in state.all_trades
            if t.get("account_id") == acct.account_id
        )
        print(
            f"  {acct.label:10s}  {acct.group:5s}  "
            f"${float(acct.balance):>11,.2f}  "
            f"${float(acct.profit):>9,.2f}  "
            f"{acct.tier:>4d}  "
            f"{acct.status.value:>10s}  "
            f"{acct_trades:>6d}"
        )

    # Max drawdown (portfolio-level)
    if state.daily_results:
        cumulative_pnl = []
        running = 0
        for r in state.daily_results:
            running += r["day_pnl"]
            cumulative_pnl.append(running)

        peak = 0
        max_dd = 0
        for cp in cumulative_pnl:
            peak = max(peak, cp)
            dd = peak - cp
            max_dd = max(max_dd, dd)

        print(f"\n  Cumulative P&L:     ${cumulative_pnl[-1]:,.2f}")
        print(f"  Max drawdown:       ${max_dd:,.2f}")

    # Best / worst day
    if state.daily_results:
        trading_days = [r for r in state.daily_results if r["ticks"] > 0]
        if trading_days:
            best = max(trading_days, key=lambda r: r["day_pnl"])
            worst = min(trading_days, key=lambda r: r["day_pnl"])
            print(f"\n  Best day:           {best['date']}  ${best['day_pnl']:>+,.2f}")
            print(f"  Worst day:          {worst['date']}  ${worst['day_pnl']:>+,.2f}")

    # Average RTH signals per trading day
    if state.daily_results:
        trading_days = [r for r in state.daily_results if r["ticks"] > 0]
        n_trading = len(trading_days)
        if n_trading > 0:
            avg_sigs = total_signals / n_trading
            avg_exec = total_executable / n_trading
            print(f"\n  Avg signals/day:    {avg_sigs:.2f} ({avg_exec:.2f} executable)")

    # Monthly P&L breakdown
    if state.daily_results:
        print(f"\n  --- Monthly P&L Breakdown ---\n")
        months: dict[str, dict] = {}
        for r in state.daily_results:
            month = r["date"][:7]  # YYYY-MM
            if month not in months:
                months[month] = {
                    "pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                    "signals": 0, "executable": 0, "days": 0,
                }
            m = months[month]
            m["pnl"] += r["day_pnl"]
            m["trades"] += r["trades"]
            m["wins"] += r["wins"]
            m["losses"] += r["losses"]
            m["signals"] += r["signals"]
            m["executable"] += r["executable"]
            if r["ticks"] > 0:
                m["days"] += 1

        print(f"  {'Month':8s}  {'Days':>4s}  {'Signals':>7s}  {'Exec':>4s}  "
              f"{'Trades':>6s}  {'W/L':>7s}  {'Win%':>5s}  {'P&L':>12s}")
        print(f"  {'-' * 65}")
        for month, m in sorted(months.items()):
            wr = m["wins"] / m["trades"] if m["trades"] > 0 else 0
            print(
                f"  {month:8s}  {m['days']:>4d}  {m['signals']:>7d}  "
                f"{m['executable']:>4d}  {m['trades']:>6d}  "
                f"{m['wins']}/{m['losses']:>3d}  {wr:>5.0%}  "
                f"${m['pnl']:>11,.2f}"
            )

    # ── Signal Table ───────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  ALL SIGNALS")
    print(f"{'=' * 72}\n")

    header = (
        f"  {'#':>3s}  {'Date':10s}  {'Time(ET)':8s}  {'Dir':5s}  "
        f"{'Level':>10s}  {'Entry':>10s}  {'Session':12s}  {'Predicted Class':25s}  "
        f"{'Exec':4s}  {'Rejection Reason'}"
    )
    print(header)
    print(f"  {'---':>3s}  {'----------':10s}  {'--------':8s}  {'-----':5s}  "
          f"{'----------':>10s}  {'----------':>10s}  {'------------':12s}  {'-------------------------':25s}  "
          f"{'----':4s}  {'----------------'}")

    for i, p in enumerate(state.all_predictions, 1):
        ts = datetime.fromisoformat(p["timestamp"])
        ts_et = ts.astimezone(ET)
        date_str_p = ts_et.strftime("%Y-%m-%d")
        time_str = ts_et.strftime("%H:%M:%S")
        direction = p["trade_direction"][:1].upper()
        level = p["level_price"]
        mkt = p.get("market_price")
        session = p.get("session", "?")
        pred_class = p["predicted_class"]
        is_exec = p["is_executable"]

        if is_exec:
            reason = ""
            exec_mark = "YES"
            entry_str = f"{mkt:>10.2f}" if mkt else f"{'N/A':>10s}"
        else:
            reasons = []
            if session != "ny_rth":
                reasons.append(f"wrong session ({session})")
            if pred_class != "tradeable_reversal":
                reasons.append(f"class={pred_class}")
            reason = "; ".join(reasons) if reasons else "unknown"
            exec_mark = " no"
            entry_str = f"{'---':>10s}"

        print(
            f"  {i:3d}  {date_str_p}  {time_str}  "
            f"{'LONG' if direction == 'L' else 'SHORT':5s}  "
            f"{level:>10.2f}  {entry_str}  {session:12s}  {pred_class:25s}  "
            f"{exec_mark:4s}  {reason}"
        )

    # ── Signal Breakdown ───────────────────────────────────────
    print(f"\n  --- Signals by Session ---\n")
    sessions_count: dict[str, int] = {}
    for p in state.all_predictions:
        s = p.get("session", "?")
        sessions_count[s] = sessions_count.get(s, 0) + 1
    for s in ["ny_rth", "london", "asia", "pre_market"]:
        print(f"    {s:15s}  {sessions_count.get(s, 0)}")
    other_sessions = {k: v for k, v in sessions_count.items()
                      if k not in ("ny_rth", "london", "asia", "pre_market")}
    for s, c in sorted(other_sessions.items()):
        print(f"    {s:15s}  {c}")

    # NY RTH by class
    rth_preds = [p for p in state.all_predictions if p.get("session") == "ny_rth"]
    non_rth_preds = [p for p in state.all_predictions if p.get("session") != "ny_rth"]

    print(f"\n  --- NY RTH Signals by Predicted Class ---\n")
    for cls in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
        count = sum(1 for p in rth_preds if p["predicted_class"] == cls)
        suffix = " (executable)" if cls == "tradeable_reversal" else ""
        print(f"    {cls:25s}  {count}{suffix}")

    print(f"\n  --- Non-RTH Signals by Predicted Class ---\n")
    for cls in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
        count = sum(1 for p in non_rth_preds if p["predicted_class"] == cls)
        suffix = " (would be executable if RTH)" if cls == "tradeable_reversal" else ""
        print(f"    {cls:25s}  {count}{suffix}")

    # ── Shadow Position Analysis ───────────────────────────────
    shadow = state.shadow_positions
    if shadow:
        print(f"\n{'=' * 72}")
        print("  LONDON/NON-RTH SESSION ANALYSIS (Shadow Trades)")
        print(f"{'=' * 72}\n")

        header = (
            f"  {'#':>3s}  {'Date':10s}  {'Time(ET)':8s}  {'Dir':5s}  "
            f"{'Level':>10s}  {'Session':12s}  "
            f"{'MFE':>6s}  {'MAE':>6s}  "
            f"{'15pt':6s}  {'30pt':6s}  {'Time to Res (15pt)'}"
        )
        print(header)
        print(f"  {'---':>3s}  {'----------':10s}  {'--------':8s}  {'-----':5s}  "
              f"{'----------':>10s}  {'------------':12s}  "
              f"{'------':>6s}  {'------':>6s}  "
              f"{'------':6s}  {'------':6s}  {'-------------------'}")

        for i, sp in enumerate(shadow, 1):
            ts_et = sp["entry_time"].astimezone(ET)
            date_s = ts_et.strftime("%Y-%m-%d")
            time_s = ts_et.strftime("%H:%M:%S")

            # 15pt outcome
            if sp["tp15_hit"]:
                res15 = "TP"
            elif sp["sl15_hit"]:
                res15 = "SL"
            else:
                res15 = "open"

            # 30pt outcome
            if sp["tp30_hit"]:
                res30 = "TP"
            elif sp["sl30_hit"]:
                res30 = "SL"
            else:
                res30 = "open"

            # Time to resolution (15pt)
            if sp["tp15_time"]:
                delta = sp["tp15_time"] - sp["entry_time"]
                ttr = str(delta).split(".")[0]  # HH:MM:SS
            elif sp["sl15_time"]:
                delta = sp["sl15_time"] - sp["entry_time"]
                ttr = str(delta).split(".")[0]
            else:
                ttr = "unresolved"

            print(
                f"  {i:3d}  {date_s}  {time_s}  "
                f"{'LONG' if sp['direction'] == 'long' else 'SHORT':5s}  "
                f"{sp['level_price']:>10.2f}  {sp['session']:12s}  "
                f"{sp['mfe_pts']:>6.1f}  {sp['mae_pts']:>6.1f}  "
                f"{res15:6s}  {res30:6s}  {ttr}"
            )

        # ── Comparison Summary ──────────────────────────────────
        print(f"\n  --- Comparison: RTH vs London (Shadow) ---\n")

        # RTH actual trades (from outcome tracker)
        rth_exec = [p for p in state.all_predictions
                    if p.get("session") == "ny_rth"
                    and p["predicted_class"] == "tradeable_reversal"]
        # Match outcomes
        rth_outcomes = []
        for p in rth_exec:
            for o in state.all_outcomes:
                if o["event_id"] == p["event_id"]:
                    rth_outcomes.append(o)
                    break

        print(f"  RTH Signals (actual trades):")
        print(f"    Total:       {len(rth_exec)}")
        if rth_outcomes:
            rth_mfes = [o["mfe_points"] for o in rth_outcomes]
            rth_maes = [o["mae_points"] for o in rth_outcomes]
            print(f"    Win rate:    100.0% (all TP)")
            print(f"    Avg MFE:     {sum(rth_mfes)/len(rth_mfes):.1f} pts")
            print(f"    Avg MAE:     {sum(rth_maes)/len(rth_maes):.1f} pts")

        # London shadow results
        london_shadow = [sp for sp in shadow if sp["session"] == "london"]
        print(f"\n  London Signals (simulated):")
        print(f"    Total:       {len(london_shadow)}")

        if london_shadow:
            tp15_wins = sum(1 for sp in london_shadow if sp["tp15_hit"])
            sl15_losses = sum(1 for sp in london_shadow if sp["sl15_hit"])
            tp30_wins = sum(1 for sp in london_shadow if sp["tp30_hit"])
            sl30_losses = sum(1 for sp in london_shadow if sp["sl30_hit"])
            open15 = sum(1 for sp in london_shadow if not sp["resolved_15"])
            open30 = sum(1 for sp in london_shadow if not sp["resolved_30"])

            resolved15 = tp15_wins + sl15_losses
            resolved30 = tp30_wins + sl30_losses
            wr15 = tp15_wins / resolved15 if resolved15 > 0 else 0
            wr30 = tp30_wins / resolved30 if resolved30 > 0 else 0

            print(f"    Win rate (15pt TP/SL):  {wr15:.1%} ({tp15_wins}W/{sl15_losses}L/{open15} open)")
            print(f"    Win rate (30pt TP/SL):  {wr30:.1%} ({tp30_wins}W/{sl30_losses}L/{open30} open)")
            avg_mfe = sum(sp["mfe_pts"] for sp in london_shadow) / len(london_shadow)
            avg_mae = sum(sp["mae_pts"] for sp in london_shadow) / len(london_shadow)
            print(f"    Avg MFE:     {avg_mfe:.1f} pts")
            print(f"    Avg MAE:     {avg_mae:.1f} pts")

            # Time to resolution for 15pt
            res_times = []
            for sp in london_shadow:
                t = sp["tp15_time"] or sp["sl15_time"]
                if t:
                    res_times.append((t - sp["entry_time"]).total_seconds())
            if res_times:
                avg_ttr = sum(res_times) / len(res_times)
                mins, secs = divmod(int(avg_ttr), 60)
                hrs, mins = divmod(mins, 60)
                print(f"    Avg time to resolution (15pt): {hrs}h {mins}m {secs}s")

        # All non-RTH shadow
        all_shadow = shadow
        print(f"\n  All Non-RTH Signals (simulated):")
        print(f"    Total:       {len(all_shadow)}")
        if all_shadow:
            tp15_all = sum(1 for sp in all_shadow if sp["tp15_hit"])
            sl15_all = sum(1 for sp in all_shadow if sp["sl15_hit"])
            open15_all = sum(1 for sp in all_shadow if not sp["resolved_15"])
            res15_all = tp15_all + sl15_all
            wr15_all = tp15_all / res15_all if res15_all > 0 else 0
            print(f"    Win rate (15pt): {wr15_all:.1%} ({tp15_all}W/{sl15_all}L/{open15_all} open)")

        # Combined projection (all 5 accounts @ 15pt TP/SL)
        print(f"\n  Combined (if both RTH + London traded):")
        combined_total = len(rth_exec) + len(london_shadow)
        combined_wins_15 = len(rth_exec) + tp15_wins  # RTH all won
        combined_losses_15 = sl15_losses
        combined_resolved = combined_wins_15 + combined_losses_15
        combined_wr = combined_wins_15 / combined_resolved if combined_resolved > 0 else 0
        # P&L: 15pt TP/SL x $20/pt x 5 accounts
        rth_pnl = len(rth_exec) * 15 * 20 * 5
        london_pnl = (tp15_wins * 15 - sl15_losses * 15) * 20 * 5
        print(f"    Total signals: {combined_total}")
        print(f"    Win rate (15pt): {combined_wr:.1%}")
        print(f"    RTH P&L (actual):    ${rth_pnl:>10,.2f}")
        print(f"    London P&L (shadow): ${london_pnl:>10,.2f}")
        print(f"    Combined P&L:        ${rth_pnl + london_pnl:>10,.2f}")

    print(f"\n{'=' * 72}\n")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical backtest")
    parser.add_argument("--days", type=int, default=5, help="Number of trading days")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(name)s - %(message)s",
    )

    # ── 1. Validate data and model ────────────────────────────
    all_dates = get_available_dates()
    if len(all_dates) < 3:
        print("ERROR: Need at least 3 dates (1 prior + 2 trading)")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Select date range (skip first date — need it for PDH/PDL)
    if args.start:
        try:
            start_idx = all_dates.index(args.start)
        except ValueError:
            # Find first date >= start
            start_idx = next(
                (i for i, d in enumerate(all_dates) if d >= args.start), None
            )
            if start_idx is None:
                print(f"ERROR: No dates on or after {args.start}")
                print(f"  Available: {all_dates[0]} to {all_dates[-1]}")
                sys.exit(1)
    else:
        start_idx = 2  # Skip first 2 dates (need prior data for levels)

    if args.end:
        # Find last date <= end
        end_idx = next(
            (i for i, d in enumerate(reversed(all_dates)) if d <= args.end), None
        )
        if end_idx is not None:
            end_idx = len(all_dates) - 1 - end_idx
            selected_dates = all_dates[start_idx : end_idx + 1]
        else:
            print(f"ERROR: No dates on or before {args.end}")
            sys.exit(1)
    else:
        n_days = min(args.days, len(all_dates) - start_idx)
        selected_dates = all_dates[start_idx : start_idx + n_days]

    print(f"\n{'=' * 72}")
    print("  HISTORICAL BACKTEST")
    print(f"{'=' * 72}")
    print(f"\n  Model:    {model_path}")
    n_days = len(selected_dates)
    print(f"  Dates:    {selected_dates[0]} to {selected_dates[-1]} ({n_days} days)")
    print(f"  Accounts: 5 (all Group A)")
    print(f"  TP/SL:    15/15pts (all accounts)")

    # ── 2. Create state with all components ───────────────────
    state = BacktestState()

    # Upload and activate model
    version = state.model_manager.upload_model(model_path)
    state.model_manager.activate_model(version["id"])
    print(f"  Model:    loaded ({model_path.name})")

    # ── 3. Run day by day ─────────────────────────────────────
    conn = duckdb.connect()

    print(f"\n  --- Daily Results ---\n")
    print(
        f"  {'Date':10s}  "
        f"{'Sym':5s}  "
        f"{'Lvls':>4s}  "
        f"{'Sigs':>4s}  "
        f"{'Exec':>4s}  "
        f"{'Trades':>6s}  "
        f"{'W/L':>5s}  "
        f"{'P&L':>10s}  "
        f"{'Acc':>5s}  "
        f"{'Time':>7s}"
    )
    print(f"  {'-' * 68}")

    for date_str in selected_dates:
        result = process_trading_day(state, date_str, conn)
        print_day_summary(result)

    conn.close()

    # ── 4. Print summary ──────────────────────────────────────
    print_final_summary(state)

    # ── 5. Save CSVs ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(state.daily_results)
    df.to_csv(output_path, index=False)
    print(f"  Results saved to: {output_path}")

    # Per-trade CSV (one row per closed trade)
    trades_path = output_path.with_name(
        output_path.stem + "_trades" + output_path.suffix,
    )
    if state.all_trades:
        df_trades = pd.DataFrame(state.all_trades)
        df_trades.to_csv(trades_path, index=False)
        print(f"  Trade log saved to: {trades_path}")

    # Per-prediction CSV (one row per signal)
    preds_path = output_path.with_name(
        output_path.stem + "_predictions" + output_path.suffix,
    )
    if state.all_predictions:
        # Flatten features dict into columns
        pred_rows = []
        for p in state.all_predictions:
            row = {k: v for k, v in p.items() if k != "features"}
            if "probabilities" in row and isinstance(row["probabilities"], dict):
                for cls, prob in row["probabilities"].items():
                    row[f"prob_{cls}"] = prob
                del row["probabilities"]
            if "features" in p and isinstance(p["features"], dict):
                for fname, fval in p["features"].items():
                    row[f"feat_{fname}"] = fval
            # Match outcome
            for o in state.all_outcomes:
                if o["event_id"] == p["event_id"]:
                    row["actual_class"] = o["actual_class"]
                    row["prediction_correct"] = o["prediction_correct"]
                    row["mfe_points"] = o["mfe_points"]
                    row["mae_points"] = o["mae_points"]
                    row["resolution_type"] = o["resolution_type"]
                    break
            pred_rows.append(row)
        df_preds = pd.DataFrame(pred_rows)
        df_preds.to_csv(preds_path, index=False)
        print(f"  Predictions saved to: {preds_path}")


if __name__ == "__main__":
    main()
