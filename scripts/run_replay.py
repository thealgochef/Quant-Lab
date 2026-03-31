"""
Replay mode entry point — replays historical tick data through the dashboard.

Creates a ReplayClient, injects it into PipelineService, and wires all
components identical to the live dashboard. Day boundary callbacks handle
the same reset sequence as run_backtest.py.

Usage::

    python scripts/run_replay.py --start 2025-06-05 --end 2025-06-10
    python scripts/run_replay.py --speed 50 --port 8001
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from alpha_lab.dashboard.api.server import DashboardState, create_app
from alpha_lab.dashboard.config.settings import DashboardSettings
from alpha_lab.dashboard.engine.feature_computer import FeatureComputer
from alpha_lab.dashboard.engine.level_engine import LevelEngine
from alpha_lab.dashboard.engine.models import ObservationStatus
from alpha_lab.dashboard.engine.observation_manager import ObservationManager
from alpha_lab.dashboard.engine.touch_detector import TouchDetector
from alpha_lab.dashboard.model.model_manager import ModelManager
from alpha_lab.dashboard.model.outcome_tracker import OutcomeTracker
from alpha_lab.dashboard.model.prediction_engine import PredictionEngine
from alpha_lab.dashboard.pipeline.pipeline_service import PipelineService
from alpha_lab.dashboard.pipeline.replay_client import ReplayClient
from alpha_lab.dashboard.pipeline.tick_bar_builder import TickBarBuilder

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("data/databento/NQ")
MODEL_PATH = Path("data/models/dashboard_3feature_v1.cbm")


def _create_replay_state(
    data_dir: Path,
    start_date: str | None,
    end_date: str | None,
    speed: float,
) -> DashboardState:
    """Create a DashboardState wired to a ReplayClient.

    Mirrors _create_live_state() but injects ReplayClient instead of
    DatabentoClient, and wires day-boundary callbacks for proper
    level computation across multi-day replays.
    """
    settings = DashboardSettings()

    # Create replay client and inject into pipeline
    replay_client = ReplayClient(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        speed=speed,
    )
    pipeline = PipelineService(settings, client=replay_client)

    # ── Components (identical to _create_live_state) ──────────
    level_engine = LevelEngine(pipeline._buffer)
    feature_computer = FeatureComputer()
    touch_detector = TouchDetector(level_engine)
    observation_manager = ObservationManager(feature_computer)
    model_manager = ModelManager(settings.model_dir)

    # Auto-load model
    from alpha_lab.dashboard.api.server import _auto_load_model
    _auto_load_model(model_manager, settings.model_dir)
    prediction_engine = PredictionEngine(model_manager)
    outcome_tracker = OutcomeTracker()

    state = DashboardState(
        pipeline=pipeline,
        level_engine=level_engine,
        touch_detector=touch_detector,
        observation_manager=observation_manager,
        model_manager=model_manager,
        prediction_engine=prediction_engine,
        outcome_tracker=outcome_tracker,
        replay_mode=True,
    )

    # Default accounts
    state.account_manager.add_account("A1", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A2", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A3", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A4", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A5", Decimal("147"), Decimal("85"), "A")

    # ── Thread-safe broadcast helper ──────────────────────────
    def _schedule_broadcast(msg: dict) -> None:
        loop = state.event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(
                loop.create_task,
                state.ws_manager.broadcast(msg),
            )

    # ── Tick bar builder ──────────────────────────────────────
    tick_bar_builder = TickBarBuilder()
    state.tick_bar_builder = tick_bar_builder

    def _on_bar_complete(timeframe: str, bar) -> None:
        bar_data = {
            "timestamp": bar.timestamp.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": bar.volume,
        }
        loop = state.event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(
                loop.create_task,
                state.ws_manager.broadcast_bar(timeframe, bar_data),
            )

    def _on_bar_complete_step(timeframe: str, bar) -> None:
        """Signal the replay client that a bar completed (for step mode)."""
        replay_client._bar_complete_event.set()

    tick_bar_builder.on_bar_complete(_on_bar_complete)
    tick_bar_builder.on_bar_complete(_on_bar_complete_step)
    pipeline.register_trade_handler(tick_bar_builder.on_trade)

    # ── Callback chain (mirrors server.py) ────────────────────
    def _on_touch(event) -> None:
        window = observation_manager.start_observation(event)
        if window is not None:
            _schedule_broadcast({
                "type": "observation_started",
                "data": {
                    "event_id": event.event_id,
                    "direction": event.trade_direction.value,
                    "level_price": float(
                        event.level_zone.representative_price,
                    ),
                    "start_time": window.start_time.isoformat(),
                    "end_time": window.end_time.isoformat(),
                    "status": window.status.value,
                    "trades_accumulated": 0,
                },
            })

    touch_detector.on_touch(_on_touch)

    def _on_observation_complete(window) -> None:
        if window.status != ObservationStatus.COMPLETED:
            return
        prediction_engine.predict(window)

    observation_manager.on_observation_complete(_on_observation_complete)

    def _broadcast_session_stats() -> None:
        wins = sum(
            1 for p in state.todays_predictions
            if p.get("prediction_correct")
        )
        losses = sum(
            1 for p in state.todays_predictions
            if p.get("prediction_correct") is not None
            and not p.get("prediction_correct")
        )
        total = wins + losses
        _schedule_broadcast({
            "type": "session_stats",
            "data": {
                "signals_fired": len(state.todays_predictions),
                "wins": wins,
                "losses": losses,
                "accuracy": round(wins / total, 4) if total > 0 else 0,
                "total_trades": len(state.todays_trades),
                "total_pnl": sum(
                    float(t.get("pnl", 0)) for t in state.todays_trades
                ),
            },
        })

    def _on_prediction(prediction) -> None:
        pred_data = {
            "event_id": prediction.event_id,
            "predicted_class": prediction.predicted_class,
            "is_executable": prediction.is_executable,
            "probabilities": prediction.probabilities,
            "features": prediction.features,
            "trade_direction": prediction.trade_direction.value,
            "level_price": float(prediction.level_price),
            "model_version": prediction.model_version,
            "timestamp": prediction.timestamp.isoformat(),
        }
        state.last_prediction = pred_data
        state.todays_predictions.append(pred_data)
        _schedule_broadcast({"type": "prediction", "data": pred_data})
        _broadcast_session_stats()
        outcome_tracker.start_tracking(prediction)

        if prediction.is_executable:
            executor_dict = {
                "is_executable": True,
                "trade_direction": prediction.trade_direction,
                "level_price": prediction.level_price,
            }
            market_price = (
                Decimal(str(state.latest_price))
                if state.latest_price
                else prediction.level_price
            )
            state.trade_executor.on_prediction(
                prediction=executor_dict,
                timestamp=prediction.timestamp,
                current_price=market_price,
            )

    prediction_engine.on_prediction(_on_prediction)

    def _on_trade_opened(pos) -> None:
        acct = state.account_manager.get_account(pos.account_id)
        group = acct.group if acct else "A"
        tp_points = state.position_monitor.get_group_tp(group)
        sl_points = state.position_monitor.get_group_sl(group)
        entry = pos.entry_price
        if pos.direction.value == "long":
            tp_price = float(entry + tp_points)
            sl_price = float(entry - sl_points)
        else:
            tp_price = float(entry - tp_points)
            sl_price = float(entry + sl_points)

        _schedule_broadcast({
            "type": "trade_opened",
            "data": {
                "account_id": pos.account_id,
                "direction": pos.direction.value,
                "entry_price": float(pos.entry_price),
                "contracts": pos.contracts,
                "entry_time": pos.entry_time.isoformat(),
                "tp_price": tp_price,
                "sl_price": sl_price,
            },
        })

    state.trade_executor.on_trade_opened(_on_trade_opened)

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
        state.todays_trades.append(trade_data)
        _schedule_broadcast({"type": "trade_closed", "data": trade_data})

        acct = state.account_manager.get_account(trade.account_id)
        if acct is not None:
            state.equity_snapshots.append({
                "timestamp": trade_data["exit_time"],
                "account_id": trade.account_id,
                "balance": float(acct.balance),
                "profit": float(acct.profit),
                "group": trade.group,
            })
            # Broadcast account update with new balance
            _schedule_broadcast({
                "type": "account_update",
                "data": {
                    "account_id": acct.account_id,
                    "balance": float(acct.balance),
                    "profit": float(acct.profit),
                    "daily_pnl": float(acct.daily_pnl),
                    "group": acct.group,
                    "status": acct.status.value,
                    "has_position": acct.has_position,
                },
            })
        _broadcast_session_stats()

    state.trade_executor.on_trade_closed(_on_trade_closed)

    def _on_outcome_resolved(outcome) -> None:
        for pred in state.todays_predictions:
            if pred.get("event_id") == outcome.event_id:
                pred["prediction_correct"] = outcome.prediction_correct
                pred["actual_class"] = outcome.actual_class
                break
        _schedule_broadcast({
            "type": "outcome_resolved",
            "data": {
                "event_id": outcome.event_id,
                "predicted_class": outcome.prediction.predicted_class,
                "actual_class": outcome.actual_class,
                "prediction_correct": outcome.prediction_correct,
                "mfe_points": outcome.mfe_points,
                "mae_points": outcome.mae_points,
                "resolution_type": outcome.resolution_type,
            },
        })
        _broadcast_session_stats()

    outcome_tracker.on_outcome_resolved(_on_outcome_resolved)

    # ── Bridge handlers (tick stream → all components) ────────
    def _on_trade(trade) -> None:
        try:
            price = float(trade.price)
            state.latest_price = price
            state.ws_manager.update_price(
                price=price,
                bid=state.latest_bid,
                ask=state.latest_ask,
                timestamp=trade.timestamp.isoformat(),
            )
        except Exception:
            logger.exception("Replay _on_trade: price update failed")

        try:
            state.position_monitor.check_flatten_time(
                trade.timestamp, trade.price,
            )
        except Exception:
            logger.exception("Replay _on_trade: flatten check failed")

        if not state.session_ended:
            ts_et = trade.timestamp.astimezone(ET)
            past_flatten = (
                ts_et.hour > 15
                or (ts_et.hour == 15 and ts_et.minute >= 55)
            )
            if past_flatten:
                state.session_ended = True
                outcome_tracker.on_session_end()
                _broadcast_session_stats()

        try:
            touch_detector.on_trade(trade)
        except Exception:
            logger.exception("Replay _on_trade: touch_detector failed")

        try:
            observation_manager.on_trade(trade)
        except Exception:
            logger.exception("Replay _on_trade: observation_manager failed")

        try:
            state.position_monitor.on_trade(trade)
        except Exception:
            logger.exception("Replay _on_trade: position_monitor failed")

        try:
            outcome_tracker.on_trade(trade)
        except Exception:
            logger.exception("Replay _on_trade: outcome_tracker failed")

    def _on_bbo(bbo) -> None:
        try:
            state.latest_bid = float(bbo.bid_price)
            state.latest_ask = float(bbo.ask_price)
            state.ws_manager.update_price(
                price=state.latest_price or 0.0,
                bid=state.latest_bid,
                ask=state.latest_ask,
                timestamp=bbo.timestamp.isoformat(),
            )
        except Exception:
            logger.exception("Replay _on_bbo: price update failed")

        try:
            observation_manager.on_bbo(bbo)
        except Exception:
            logger.exception("Replay _on_bbo: observation_manager failed")

    pipeline.register_trade_handler(_on_trade)
    pipeline.register_bbo_handler(_on_bbo)

    # ── Day boundary handler ──────────────────────────────────
    def _on_day_boundary(date_str: str) -> None:
        """Reset state between replay days (mirrors run_backtest.py)."""
        trading_date = date.fromisoformat(date_str)

        # Reset daily state
        state.todays_trades.clear()
        state.todays_predictions.clear()
        state.last_prediction = None
        state.session_ended = False
        level_engine.reset_daily()
        state.account_manager.start_new_day()
        pipeline._buffer.evict()
        # NOTE: Do NOT reset tick_bar_builder here. Tick bars accumulate
        # continuously across days — a bar spanning midnight is fine.
        # _completed_bars must survive so the REST endpoint can serve them.

        # Compute levels (PDH/PDL from accumulated ticks)
        rth_open = datetime.combine(
            trading_date, time(9, 30), tzinfo=ET,
        ).astimezone(UTC)
        levels = level_engine.compute_levels(
            trading_date, current_time=rth_open,
        )

        # Broadcast level update
        zones_data = []
        for zone in level_engine.get_active_zones():
            zones_data.append({
                "zone_id": zone.zone_id,
                "price": float(zone.representative_price),
                "side": zone.side.value,
                "is_touched": zone.is_touched,
                "levels": [
                    {
                        "type": lv.level_type.value,
                        "price": float(lv.price),
                        "is_manual": lv.is_manual,
                    }
                    for lv in zone.levels
                ],
            })
        if zones_data:
            _schedule_broadcast({
                "type": "level_update",
                "data": {"action": "full_refresh", "levels": zones_data},
            })

        # Broadcast replay day info
        _schedule_broadcast({
            "type": "replay_day",
            "data": {"date": date_str, "levels_count": len(levels)},
        })

        logger.info(
            "Replay day boundary: %s (%d levels)",
            date_str, len(levels),
        )

    replay_client.on_day_boundary(_on_day_boundary)

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay historical data through dashboard")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--speed", type=float, default=10.0, help="Replay speed multiplier")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR), help="Data directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(name)s - %(message)s",
    )

    state = _create_replay_state(
        data_dir=Path(args.data_dir),
        start_date=args.start,
        end_date=args.end,
        speed=args.speed,
    )

    app = create_app(state=state)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
