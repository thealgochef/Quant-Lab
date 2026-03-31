"""
FastAPI server — app factory, DashboardState, and lifespan management.

Creates the FastAPI application that wires together all Phase 1-4
components and exposes them via REST + WebSocket endpoints.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from alpha_lab.dashboard.api.websocket import WebSocketManager
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
from alpha_lab.dashboard.pipeline.tick_bar_builder import TickBarBuilder
from alpha_lab.dashboard.pipeline.rithmic_client import (
    BBOUpdate,
    ConnectionStatus,
    TradeUpdate,
)
from alpha_lab.dashboard.trading.account_manager import AccountManager
from alpha_lab.dashboard.trading.position_monitor import PositionMonitor
from alpha_lab.dashboard.trading.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Central state container holding all Phase 1-4 components.

    Passed to the FastAPI app at creation time. Routes access this
    via ``request.app.state.dashboard``.
    """

    # Required — always available
    account_manager: AccountManager = field(default_factory=AccountManager)
    trade_executor: TradeExecutor | None = None
    position_monitor: PositionMonitor | None = None
    ws_manager: WebSocketManager = field(default_factory=WebSocketManager)

    # Optional — may be None if component not wired
    level_engine: LevelEngine | None = None
    touch_detector: TouchDetector | None = None
    observation_manager: ObservationManager | None = None
    model_manager: ModelManager | None = None
    prediction_engine: PredictionEngine | None = None
    outcome_tracker: OutcomeTracker | None = None

    # Pipeline — live data (None in test mode)
    pipeline: PipelineService | None = None
    tick_bar_builder: TickBarBuilder | None = None
    replay_mode: bool = False

    # Event loop reference for thread-safe WS broadcasting from
    # Databento's background consumer thread. Set during lifespan.
    event_loop: asyncio.AbstractEventLoop | None = field(
        default=None, repr=False,
    )

    # In-memory event logs (today's session)
    todays_trades: list[dict] = field(default_factory=list)
    todays_predictions: list[dict] = field(default_factory=list)
    last_prediction: dict | None = None

    # Latest market data
    latest_price: float | None = None
    latest_bid: float | None = None
    latest_ask: float | None = None
    connection_status: str = "disconnected"

    # Overlay config
    overlay_config: dict[str, bool] = field(default_factory=lambda: {
        "ema_13": True,
        "ema_48": True,
        "ema_200": True,
        "vwap": False,
        "levels": True,
    })

    # Equity curve tracking (snapshots after each trade close)
    equity_snapshots: list[dict] = field(default_factory=list)

    # Session lifecycle — prevents repeated flatten/session-end calls
    session_ended: bool = False

    def __post_init__(self) -> None:
        if self.trade_executor is None:
            self.trade_executor = TradeExecutor(self.account_manager)
        if self.position_monitor is None:
            self.position_monitor = PositionMonitor(
                self.account_manager, self.trade_executor,
            )


async def _preload_tick_bars_from_api(
    tick_bar_builder: TickBarBuilder,
    client: object,
    days: int = 4,
) -> None:
    """Fetch recent trades from Databento Historical API and build tick bars.

    Runs after pipeline.start() so the API key and connection are already
    validated.  Uses DuckDB to efficiently group trades into tick bars.
    """
    import duckdb
    import pandas as pd

    from alpha_lab.dashboard.pipeline.databento_client import DatabentoClient
    from alpha_lab.dashboard.pipeline.price_buffer import OHLCVBar

    if not isinstance(client, DatabentoClient):
        logger.info("Client is not DatabentoClient — skipping tick bar preload")
        return

    logger.info("Fetching last %d days of trades for tick bar preload...", days)
    trades_df = await client.fetch_historical_trades("NQ.c.0", days=days)

    if trades_df.empty:
        logger.warning("No historical trades returned — tick bars will be empty until live data flows")
        return

    logger.info("Building tick bars from %d trades...", len(trades_df))

    conn = duckdb.connect()
    try:
        conn.register("trades_raw", trades_df)

        for tick_count_str, tick_count in [("987t", 987), ("2000t", 2000)]:
            df = conn.execute(f"""
                WITH numbered AS (
                    SELECT *,
                           (row_number() OVER (ORDER BY ts_event) - 1)
                           // {tick_count} AS grp
                    FROM trades_raw
                )
                SELECT
                    max(ts_event) AS ts,
                    first(price ORDER BY ts_event) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price ORDER BY ts_event) AS close,
                    sum(size) AS volume
                FROM numbered
                GROUP BY grp
                HAVING count(*) = {tick_count}
                ORDER BY grp
            """).fetchdf()

            bars = []
            for _, row in df.iterrows():
                ts = row["ts"]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                bars.append(OHLCVBar(
                    timestamp=ts,
                    open=Decimal(str(round(float(row["open"]), 2))),
                    high=Decimal(str(round(float(row["high"]), 2))),
                    low=Decimal(str(round(float(row["low"]), 2))),
                    close=Decimal(str(round(float(row["close"]), 2))),
                    volume=int(row["volume"]),
                ))

            if bars:
                tick_bar_builder.preload_historical(tick_count_str, bars)
                logger.info("Preloaded %d %s bars from %d days of live data", len(bars), tick_count_str, days)
    except Exception:
        logger.exception("Tick bar preload failed — continuing without history")
    finally:
        conn.close()


def _auto_load_model(model_manager: ModelManager, model_dir: Path) -> None:
    """Scan model_dir for .cbm files and auto-upload + activate on startup."""
    cbm_files = sorted(model_dir.glob("*.cbm"))
    if not cbm_files:
        logger.warning("No model file found in %s — upload via Models tab", model_dir)
        return

    # Prefer dashboard_3feature_v1.cbm if it exists
    preferred = model_dir / "dashboard_3feature_v1.cbm"
    chosen = preferred if preferred in cbm_files else cbm_files[0]

    try:
        version = model_manager.upload_model(chosen)
        model_manager.activate_model(version["id"])
        logger.info("Auto-loaded and activated model from %s", chosen)
    except Exception:
        logger.exception("Failed to auto-load model from %s", chosen)


def _create_live_state() -> DashboardState:
    """Create a DashboardState wired to a live data pipeline.

    Loads settings from .env, creates PipelineService (using Databento
    or Rithmic based on DASHBOARD_DATA_SOURCE), and registers bridge
    handlers that funnel tick data into DashboardState + WebSocket.

    Wires the complete signal-to-trade pipeline:
      Trade tick → TouchDetector → ObservationManager → PredictionEngine
                → TradeExecutor + OutcomeTracker
      Every tick → PositionMonitor (TP/SL/DLL) + flatten check
    """
    settings = DashboardSettings()
    pipeline = PipelineService(settings)

    # ── Phase 1: Data + Levels ──────────────────────────────────
    level_engine = LevelEngine(pipeline._buffer)

    # ── Phase 2: Touch Detection + Observation ──────────────────
    feature_computer = FeatureComputer()
    touch_detector = TouchDetector(level_engine)
    observation_manager = ObservationManager(feature_computer)

    # ── Phase 3: Model + Prediction + Outcome Tracking ──────────
    model_manager = ModelManager(settings.model_dir)

    # Auto-load: scan model_dir for .cbm files, upload & activate
    _auto_load_model(model_manager, settings.model_dir)
    prediction_engine = PredictionEngine(model_manager)
    outcome_tracker = OutcomeTracker()

    # ── Create DashboardState with all components ───────────────
    state = DashboardState(
        pipeline=pipeline,
        level_engine=level_engine,
        touch_detector=touch_detector,
        observation_manager=observation_manager,
        model_manager=model_manager,
        prediction_engine=prediction_engine,
        outcome_tracker=outcome_tracker,
    )

    # ── Phase 4: Default Paper Trading Accounts ────────────────
    state.account_manager.add_account("A1", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A2", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A3", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A4", Decimal("147"), Decimal("85"), "A")
    state.account_manager.add_account("A5", Decimal("147"), Decimal("85"), "A")
    logger.info(
        "Created %d default accounts (5xA, 15pt TP/SL)",
        len(state.account_manager.get_all_accounts()),
    )

    # Record initial equity snapshots
    now_iso = datetime.now(UTC).isoformat()
    for acct in state.account_manager.get_all_accounts():
        state.equity_snapshots.append({
            "timestamp": now_iso,
            "account_id": acct.account_id,
            "balance": float(acct.balance),
            "profit": float(acct.profit),
            "group": acct.group,
        })

    # ── Thread-safe broadcast helper ────────────────────────────
    # Trade/BBO callbacks are called from Databento's background
    # consumer thread. Must schedule async WS broadcasts on the
    # event loop thread via call_soon_threadsafe.
    def _schedule_broadcast(msg: dict) -> None:
        loop = state.event_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(
                loop.create_task,
                state.ws_manager.broadcast(msg),
            )

    # ── Tick bar builder ───────────────────────────────────────
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

    tick_bar_builder.on_bar_complete(_on_bar_complete)
    pipeline.register_trade_handler(tick_bar_builder.on_trade)

    # ── Callback chain wiring ───────────────────────────────────

    # 1. Touch → Observation start + WS broadcast
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

    # 2. Observation complete → Prediction
    def _on_observation_complete(window) -> None:
        if window.status != ObservationStatus.COMPLETED:
            return  # Discarded windows don't produce predictions
        # predict() fires _on_prediction callbacks if successful
        prediction_engine.predict(window)

    observation_manager.on_observation_complete(_on_observation_complete)

    # ── Session stats broadcast helper ──────────────────────────
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

    # 3. Prediction → TradeExecutor + OutcomeTracker + state + WS
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

        # Store in state
        state.last_prediction = pred_data
        state.todays_predictions.append(pred_data)

        # WS broadcast
        _schedule_broadcast({"type": "prediction", "data": pred_data})

        # Broadcast updated session stats
        _broadcast_session_stats()

        # Start outcome tracking
        outcome_tracker.start_tracking(prediction)

        # Execute trade if executable
        if prediction.is_executable:
            # Adapter: Prediction -> dict for TradeExecutor interface
            executor_dict = {
                "is_executable": True,
                "trade_direction": prediction.trade_direction,
                "level_price": prediction.level_price,
            }
            # Entry at current market price, not the level/touch price
            market_price = Decimal(str(state.latest_price)) if state.latest_price else prediction.level_price
            state.trade_executor.on_prediction(
                prediction=executor_dict,
                timestamp=prediction.timestamp,
                current_price=market_price,
            )

    prediction_engine.on_prediction(_on_prediction)

    # 4. Trade opened → WS broadcast (includes TP/SL prices)
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

    # 5. Trade closed → state + WS broadcast + equity snapshot
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

        # Record equity snapshot for this account
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

        # Broadcast updated session stats
        _broadcast_session_stats()

    state.trade_executor.on_trade_closed(_on_trade_closed)

    # 6. Outcome resolved → state + WS broadcast + stats
    def _on_outcome_resolved(outcome) -> None:
        # Update the corresponding prediction in todays_predictions
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

        # Broadcast updated session stats
        _broadcast_session_stats()

    outcome_tracker.on_outcome_resolved(_on_outcome_resolved)

    # ── Bridge handlers (tick stream → all components) ──────────
    # Called from Databento's background thread on every tick.

    def _on_trade(trade: TradeUpdate) -> None:
        try:
            price = float(trade.price)
            state.latest_price = price

            # 1. Price update to WS (throttled 1/sec)
            state.ws_manager.update_price(
                price=price,
                bid=state.latest_bid,
                ask=state.latest_ask,
                timestamp=trade.timestamp.isoformat(),
            )
        except Exception:
            logger.exception("Bridge _on_trade: price update failed")

        # Flatten check FIRST — must close positions before any new
        # touches/observations/predictions can open new ones.
        try:
            state.position_monitor.check_flatten_time(
                trade.timestamp, trade.price,
            )
        except Exception:
            logger.exception("Bridge _on_trade: flatten check failed")

        # Session end: force-resolve remaining predictions once past flatten
        try:
            if not state.session_ended:
                from zoneinfo import ZoneInfo
                ts_et = trade.timestamp.astimezone(ZoneInfo("America/New_York"))
                past_flatten = (
                    ts_et.hour > 15
                    or (ts_et.hour == 15 and ts_et.minute >= 55)
                )
                if past_flatten:
                    state.session_ended = True
                    outcome_tracker.on_session_end()
                    _broadcast_session_stats()
                    logger.info("Session ended — force-resolved remaining predictions")
        except Exception:
            logger.exception("Bridge _on_trade: session end failed")

        # Each component is isolated — one failure doesn't block the rest
        try:
            touch_detector.on_trade(trade)
        except Exception:
            logger.exception("Bridge _on_trade: touch_detector failed")

        try:
            observation_manager.on_trade(trade)
        except Exception:
            logger.exception("Bridge _on_trade: observation_manager failed")

        try:
            state.position_monitor.on_trade(trade)
        except Exception:
            logger.exception("Bridge _on_trade: position_monitor failed")

        try:
            outcome_tracker.on_trade(trade)
        except Exception:
            logger.exception("Bridge _on_trade: outcome_tracker failed")

    def _on_bbo(bbo: BBOUpdate) -> None:
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
            logger.exception("Bridge _on_bbo: price update failed")

        try:
            observation_manager.on_bbo(bbo)
        except Exception:
            logger.exception("Bridge _on_bbo: observation_manager failed")

    def _on_connection_status(status: ConnectionStatus) -> None:
        state.connection_status = status.value
        logger.info("Data connection: %s", status.value)
        _schedule_broadcast({
            "type": "connection_status",
            "data": {"status": status.value},
        })
        # Observation may need to discard on disconnect
        observation_manager.on_connection_status(status)

    pipeline.register_trade_handler(_on_trade)
    pipeline.register_bbo_handler(_on_bbo)
    pipeline.register_connection_handler(_on_connection_status)

    # ── Historical backfill callback ────────────────────────────
    def _on_backfill_complete() -> None:
        now = datetime.now(UTC)
        today = now.date()
        levels = level_engine.compute_levels(today, current_time=now)
        logger.info(
            "Computed %d key levels (PDH/PDL, session H/L)", len(levels),
        )

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

    pipeline.register_backfill_callback(_on_backfill_complete)

    return state


def create_app(state: DashboardState | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        state: Pre-configured DashboardState. If None, creates a live
            state with Rithmic pipeline. Pass a custom state for testing.
    """
    # Ensure our app logs are visible alongside uvicorn's
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:     %(name)s - %(message)s",
    )

    if state is None:
        state = _create_live_state()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        ds: DashboardState = app.state.dashboard

        # Store event loop ref for thread-safe broadcasting
        ds.event_loop = asyncio.get_running_loop()

        # Start WebSocket broadcast loop
        ds.ws_manager.start()

        # Start pipeline if wired
        if ds.pipeline is not None:
            try:
                await ds.pipeline.start()
                logger.info("Pipeline started — streaming live data")

                # Preload last 4 days of tick bars from Databento Historical
                # API so the chart has candles immediately on startup.
                if ds.tick_bar_builder is not None:
                    try:
                        await _preload_tick_bars_from_api(
                            ds.tick_bar_builder,
                            ds.pipeline._client,
                            days=4,
                        )
                    except Exception:
                        logger.exception("Tick bar preload failed — chart will populate from live ticks")

            except Exception:
                logger.exception("Pipeline failed to start — running without live data")

        yield

        # Shutdown pipeline
        if ds.pipeline is not None and ds.pipeline.is_running:
            await ds.pipeline.stop()

        await ds.ws_manager.stop()

    app = FastAPI(
        title="Alpha Signal Lab — Dashboard API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store state on the app
    app.state.dashboard = state

    # CORS — allow frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    from alpha_lab.dashboard.api.routes.accounts import router as accounts_router
    from alpha_lab.dashboard.api.routes.config import router as config_router
    from alpha_lab.dashboard.api.routes.data import router as data_router
    from alpha_lab.dashboard.api.routes.levels import router as levels_router
    from alpha_lab.dashboard.api.routes.models import router as models_router
    from alpha_lab.dashboard.api.routes.trading import router as trading_router

    app.include_router(trading_router)
    app.include_router(accounts_router)
    app.include_router(config_router)
    app.include_router(levels_router)
    app.include_router(models_router)
    app.include_router(data_router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        try:
            await state.ws_manager.connect(ws, state)
        except Exception:
            logger.exception("WebSocket backfill/connect failed")
            return

        # Auto-play on first WebSocket connect in replay mode
        if state.replay_mode:
            from alpha_lab.dashboard.pipeline.replay_client import (
                ReplayClient as _RC,
            )
            _client = getattr(state.pipeline, "_client", None)
            if isinstance(_client, _RC) and not _client.replay_complete:
                _client.play()
                logger.info("Replay auto-play triggered on WebSocket connect")

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")
                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                elif msg_type == "subscribe_timeframe":
                    tf = data.get("data", {}).get("timeframe")
                    if tf:
                        state.ws_manager.subscribe_timeframe(ws, tf)
                elif msg_type == "replay_control" and state.replay_mode:
                    from alpha_lab.dashboard.pipeline.replay_client import (
                        ReplayClient,
                    )
                    client = getattr(state.pipeline, "_client", None)
                    if isinstance(client, ReplayClient):
                        payload = data.get("data", {})
                        action = payload.get("action")
                        if action == "play":
                            client.play()
                        elif action == "pause":
                            client.pause()
                        elif action == "step":
                            client.step()
                        elif action == "set_speed":
                            client.set_speed(float(payload.get("speed", 1.0)))
                        elif action == "set_step_mode":
                            client.set_step_mode(bool(payload.get("enabled")))
                        # Send state back so UI can sync
                        await ws.send_json({
                            "type": "replay_state",
                            "data": {
                                "action": action,
                                "paused": not client._pause_event.is_set(),
                                "step_mode": client._step_mode,
                                "speed": client._speed,
                                "replay_complete": client.replay_complete,
                                "current_date": client.current_date,
                            },
                        })
        except WebSocketDisconnect:
            state.ws_manager.disconnect(ws)
        except Exception:
            logger.exception("WebSocket handler error")
            state.ws_manager.disconnect(ws)

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
