# Phase 5 — API Server

**Scope:** FastAPI server with WebSocket for real-time data push and REST endpoints for configuration, account management, model management, and historical data
**Prerequisite:** Phase 4 complete (paper trading engine, all backend logic)
**Produces:** A fully functional API that the frontend (Phase 6) connects to
**Does NOT include:** Frontend UI

---

## 1. What This Phase Builds

**1. WebSocket Server** — Pushes real-time data to connected dashboard clients: price ticks (throttled to 1/sec), OHLCV bar updates, level status, observation window progress, predictions, trade updates, account P&L changes, connection status.

**2. REST API** — Endpoints for configuration (TP/SL, signal mode), account management (add/remove/payout), model management (upload/activate/rollback), historical data queries (trade history, performance stats, equity curves), and manual trading actions (buy/sell/close).

**3. Startup/Backfill Service** — When the dashboard connects, sends all current state and today's history so the frontend renders everything that happened before it connected.

---

## 2. Architecture

```
Frontend (Phase 6)
    ↕ WebSocket (real-time push)
    ↕ REST (actions + queries)
FastAPI Server
    ↕
Pipeline Service (Phase 1)     ← tick stream
Level Engine (Phase 2)         ← active levels
Observation Manager (Phase 2)  ← observation windows
Prediction Engine (Phase 3)    ← predictions
Trade Executor (Phase 4)       ← trade actions
Account Manager (Phase 4)      ← account state
PostgreSQL                     ← persistence
```

---

## 3. Directory Structure (Phase 5 additions)

```
src/alpha_lab/dashboard/
  api/
    __init__.py
    server.py                # FastAPI app, startup/shutdown
    websocket.py             # WebSocket manager + message broadcasting
    routes/
      __init__.py
      trading.py             # Manual trade actions, close-all
      accounts.py            # Account CRUD, payouts
      config.py              # Settings (TP/SL, signal mode, overlays)
      models.py              # Model upload, activate, rollback
      data.py                # OHLCV, trade history, performance stats
      levels.py              # Manual level add/remove
    schemas.py               # Pydantic request/response models

tests/dashboard/
  api/
    __init__.py
    test_websocket.py
    test_routes_trading.py
    test_routes_accounts.py
    test_routes_config.py
    test_routes_data.py
    test_routes_levels.py
```

---

## 4. WebSocket Specification

### 4.1 Connection

```
ws://localhost:8000/ws
```

Single WebSocket connection per dashboard client. On connect, server sends a `backfill` message with current state (see Section 4.3).

### 4.2 Message Format

All WebSocket messages are JSON with a `type` field:

```json
{
    "type": "price_update",
    "data": { ... }
}
```

### 4.3 Message Types (Server → Client)

**`backfill`** — Sent on connection. Contains full current state:
```json
{
    "type": "backfill",
    "data": {
        "connection_status": "connected",
        "latest_price": 21045.75,
        "latest_bid": 21045.50,
        "latest_ask": 21045.75,
        "active_levels": [...],
        "active_observation": null | { event_id, countdown_seconds, features_so_far },
        "last_prediction": null | { ... },
        "open_positions": [...],
        "todays_trades": [...],
        "todays_predictions": [...],
        "session_stats": { signals_fired, wins, losses, accuracy },
        "accounts": [...],
        "config": { group_a_tp, group_b_tp, ... }
    }
}
```

**`price_update`** — Throttled to 1/second:
```json
{
    "type": "price_update",
    "data": {
        "price": 21045.75,
        "bid": 21045.50,
        "ask": 21045.75,
        "timestamp": "2026-03-05T14:30:01Z"
    }
}
```

**`bar_update`** — When a new OHLCV bar closes:
```json
{
    "type": "bar_update",
    "data": {
        "timeframe": "1m",
        "timestamp": "2026-03-05T14:30:00Z",
        "open": 21040.00,
        "high": 21046.25,
        "low": 21039.50,
        "close": 21045.75,
        "volume": 1234
    }
}
```

**`level_update`** — Level added, removed, or touched:
```json
{
    "type": "level_update",
    "data": {
        "action": "touched",  // "added", "removed", "touched"
        "levels": [...]
    }
}
```

**`observation_started`** — New observation window opened:
```json
{
    "type": "observation_started",
    "data": {
        "event_id": "...",
        "level_type": "pdh",
        "level_price": 21045.75,
        "direction": "short",
        "start_time": "...",
        "end_time": "..."
    }
}
```

**`observation_progress`** — Every second during observation:
```json
{
    "type": "observation_progress",
    "data": {
        "event_id": "...",
        "seconds_remaining": 180,
        "trades_accumulated": 450
    }
}
```

**`observation_completed`** / **`observation_discarded`**

**`prediction`** — Model output:
```json
{
    "type": "prediction",
    "data": {
        "event_id": "...",
        "predicted_class": "tradeable_reversal",
        "probabilities": { "tradeable_reversal": 0.85, "trap_reversal": 0.10, "aggressive_blowthrough": 0.05 },
        "features": { "int_time_beyond_level": 45.2, "int_time_within_2pts": 180.5, "int_absorption_ratio": 3.4 },
        "is_executable": true,
        "direction": "short",
        "level_price": 21045.75,
        "timestamp": "..."
    }
}
```

**`trade_opened`** / **`trade_closed`** — Per-account trade events

**`account_update`** — Balance, P&L, tier, or status change

**`outcome_resolved`** — Prediction outcome determined (✓ or ✗)

**`connection_status`** — Rithmic connection changes

**`alert`** — System alerts (DLL warning, account blown, etc.)

### 4.4 Message Types (Client → Server)

Prefer REST for actions. WebSocket from client is only used for:
- `ping` / `pong` keepalive
- `subscribe_timeframe` — switch which timeframe bars are pushed

---

## 5. REST API Specification

### 5.1 Trading Actions

```
POST /api/trading/close-all
    Body: { "reason": "manual" }
    Response: { "closed_trades": [...] }

POST /api/trading/close/{account_id}
    Body: { "reason": "manual" }
    Response: { "closed_trade": {...} }

POST /api/trading/manual-entry
    Body: { "account_id": "...", "direction": "long", "price": 21045.75 }
    Response: { "position": {...} }
```

### 5.2 Account Management

```
GET /api/accounts
    Response: { "accounts": [...], "summary": {...} }

POST /api/accounts
    Body: { "label": "Apex #6", "eval_cost": 167.00, "activation_cost": 79.00, "group": "A" }
    Response: { "account": {...} }

GET /api/accounts/{account_id}
    Response: { "account": {...}, "trade_history": [...], "payout_history": [...] }

POST /api/accounts/{account_id}/payout
    Body: { "amount": 1500.00 }
    Response: { "payout": {...} }
```

### 5.3 Configuration

```
GET /api/config
    Response: { "group_a_tp": 15, "group_b_tp": 30, "group_a_sl": 37.5, ... }

PUT /api/config
    Body: { "group_a_tp": 20, "second_signal_mode": "flip" }
    Response: { "config": {...} }

GET /api/config/overlays
    Response: { "ema_13": true, "ema_48": true, "vwap": false, ... }

PUT /api/config/overlays
    Body: { "vwap": true }
    Response: { "overlays": {...} }
```

### 5.4 Level Management

```
GET /api/levels
    Response: { "zones": [...], "manual_levels": [...] }

POST /api/levels/manual
    Body: { "price": 21000.00 }
    Response: { "level": {...} }

DELETE /api/levels/manual/{price}
    Response: { "deleted": true }
```

### 5.5 Model Management

```
GET /api/models
    Response: { "active": {...}, "versions": [...] }

POST /api/models/upload
    Body: multipart/form-data with .cbm file + optional metrics JSON
    Response: { "version": {...} }

POST /api/models/{version_id}/activate
    Response: { "activated": true }

POST /api/models/{version_id}/rollback
    Response: { "activated": true }
```

### 5.6 Data Queries

```
GET /api/data/ohlcv?timeframe=1m&since=2026-03-05T00:00:00Z
    Response: { "bars": [...] }

GET /api/data/trades?date=2026-03-05
    Response: { "trades": [...] }

GET /api/data/predictions?date=2026-03-05
    Response: { "predictions": [...] }

GET /api/data/performance?period=7d
    Response: { "accuracy": ..., "by_session": {...}, "by_level_type": {...}, ... }

GET /api/data/equity-curve?account_id=...
    Response: { "snapshots": [...] }
```

---

## 6. Test Specifications

### 6.1 `test_websocket.py` (8 tests)

```
"""
Phase 5 — WebSocket Tests

Tests the real-time data push from server to dashboard client.
The WebSocket is the primary data channel for live trading — latency
and reliability directly impact the trader's experience.
"""
```

1. `test_connect_receives_backfill` — New connection immediately receives backfill message
2. `test_price_updates_throttled` — Price updates arrive at most 1/second
3. `test_prediction_pushed_on_signal` — Prediction message pushed when model fires
4. `test_trade_pushed_on_execution` — Trade opened/closed messages pushed
5. `test_observation_progress_updates` — Progress messages during observation window
6. `test_level_update_on_touch` — Level touch event pushed to client
7. `test_connection_status_pushed` — Rithmic connection changes pushed
8. `test_reconnect_receives_fresh_backfill` — Reconnecting client gets new backfill

### 6.2 Route tests (20 tests total across 5 files)

Each route file gets 3-5 tests covering happy path, validation errors, and edge cases. Test with `httpx.AsyncClient` against the FastAPI test client.

---

## 7. Acceptance Criteria

Phase 5 is complete when:

1. **WebSocket pushes real-time data** — Price, bars, predictions, trades, observations all push correctly
2. **Backfill works** — Dashboard connecting mid-session receives all current state and today's events
3. **REST endpoints functional** — All CRUD operations for accounts, config, levels, models work
4. **Manual trading works** — Buy/sell/close-all via API functions correctly
5. **All tests pass**

---

## 8. Dependencies to Add

```toml
fastapi = "*"
uvicorn = "*"
websockets = "*"
python-multipart = "*"   # For file uploads
httpx = "*"              # For testing
```

---

## 9. Notes for Claude Code

- **FastAPI with uvicorn.** The server runs as part of the pipeline service process (single process, async).
- **WebSocket throttling matters.** NQ can produce 10,000+ ticks per minute. The frontend needs 1 price update per second maximum. Aggregate ticks between pushes.
- **Backfill is the tricky part.** When a client connects, it needs OHLCV bars for the chart (48 hours), today's active levels, today's predictions, today's trades, current positions, and account state. This should be a single atomic snapshot.
- **CORS configuration.** If frontend runs on a different port during development, configure CORS middleware.
- **All endpoints are async.** Use SQLAlchemy async sessions throughout.
