# Phase 1 — Data Pipeline Core

**Scope:** Rithmic connection, tick streaming, independent Parquet recording, PostgreSQL operational database
**Prerequisite:** None (first phase)
**Produces:** A running service that connects to Rithmic, streams NQ ticks, records everything to Parquet files, and stores operational data in PostgreSQL
**Does NOT include:** Level detection, observation windows, feature computation, model inference, paper trading, API server, frontend

---

## 1. What This Phase Builds

Three independent components that form the data foundation for all subsequent phases:

**1. Rithmic Client** — Connects to Rithmic via R | Protocol API (WebSocket + protobuf), authenticates with Apex second-login credentials, subscribes to the front-month NQ contract, and streams every trade and BBO update in real time. Handles connection lifecycle, auto-reconnect on drop, and market hours awareness.

**2. Tick Recorder** — An independent process that listens to the Rithmic stream and writes every incoming tick to daily Parquet files. This process has zero dependency on anything else in the system. If it crashes, the dashboard keeps working. If the dashboard crashes, it keeps recording. Same format as the existing Databento Parquet files in `data/databento/NQ/`.

**3. PostgreSQL Database** — The operational database for the live trading system. Stores recent price data (for chart rendering), connection state, configuration, and provides the schema foundation for trade history, account state, and model metadata that later phases will populate.

**4. Pipeline Service** — The orchestrator process that starts the Rithmic client, feeds ticks to the recorder and to an in-memory price buffer, and exposes a simple internal interface that later phases will consume. Manages startup, shutdown, and the overall service lifecycle.

---

## 2. Relevant PRD Sections

Read these sections of `NQ_LIVE_DASHBOARD_PRD.md` for full context:

- **Section 2** (System Architecture) — Two-component design, pipeline runs independently of dashboard
- **Section 3** (Data Sources) — Rithmic second login specs, self-collected historical data, Databento for backfill
- **Section 4** (Instrument) — NQ only, tick size $0.25, tick value $5.00, auto-detect front-month, contract rollover
- **Section 7** (Connection Handling) — Auto-reconnect, no computation while disconnected, connection status tracking
- **Section 9** (Time Rules) — Market hours awareness, hard flatten at 3:55 PM CT
- **Section 13** (Configuration & Persistence) — What persists across sessions, initial setup requirements

---

## 3. Technical Context

### Rithmic API

Rithmic offers R | Protocol API — a WebSocket-based wire protocol using Google Protocol Buffers. It runs on any language/OS. The connection goes through the APEX system with a gateway selection (choose closest, typically Chicago). Credentials come from the Apex dashboard.

Key constraints:
- Rithmic allows only one Market Data session per login — the trader uses a $30/month second login add-on to avoid disconnecting Quantower
- Select NON-PROFESSIONAL for data classification
- System: APEX, Gateway: Chicago Area
- The stream provides: trades (price, size, aggressor side) and BBO updates (best bid, best ask, sizes)
- Rithmic naturally stops sending data when CME Globex is closed (weekends, holidays)
- Historical data available through Rithmic going back to December 2011, limited to 40GB/week

### Python Libraries

Research which Python libraries exist for Rithmic connectivity. Options include:
- `pyrithmic` — community Python wrapper around R | Protocol API
- Direct protobuf implementation using Rithmic's published .proto files
- Any maintained open-source Rithmic Python client

Claude Code should research the current state of available libraries and choose the most reliable, maintained option. If no good library exists, implement direct WebSocket + protobuf communication.

### Existing Project Context

The project lives at `C:\Users\gonza\Documents\Claude-Quant-Lab` with this structure:

```
Claude-Quant-Lab/
  src/alpha_lab/           # Existing Python package
    core/                  # Shared infrastructure
    agents/                # Multi-agent system
    experiment/            # ML hypothesis test (feature computation, key levels, etc.)
  tests/                   # Existing test suite (~750 tests)
  config/                  # YAML configs
  data/                    # Data files including data/databento/NQ/ Parquet files
  pyproject.toml           # Python project config
```

New dashboard code goes under `src/alpha_lab/dashboard/`. New tests go under `tests/dashboard/`.

### Parquet File Format

The existing Databento Parquet files in `data/databento/NQ/` are the reference format. The tick recorder should produce files that are compatible with or similar to this format so that future backtesting can seamlessly use both Databento historical data and self-recorded Rithmic data. Review the existing files to understand the schema before designing the recorder output.

---

## 4. Directory Structure (Phase 1)

```
src/alpha_lab/dashboard/
  __init__.py
  pipeline/
    __init__.py
    rithmic_client.py       # Rithmic connection, auth, streaming, reconnect
    tick_recorder.py         # Independent Parquet file writer
    price_buffer.py          # In-memory recent price buffer for chart data
    pipeline_service.py      # Main orchestrator — starts client, recorder, buffer
  db/
    __init__.py
    models.py                # SQLAlchemy models for PostgreSQL
    connection.py            # Database connection management
    migrations/              # Alembic migrations
  config/
    __init__.py
    settings.py              # Dashboard-specific configuration (Pydantic Settings)

tests/dashboard/
  __init__.py
  conftest.py               # Shared fixtures (mock Rithmic stream, test DB, etc.)
  pipeline/
    __init__.py
    test_rithmic_client.py
    test_tick_recorder.py
    test_price_buffer.py
    test_pipeline_service.py
  db/
    __init__.py
    test_models.py
    test_connection.py
```

---

## 5. Component Specifications

### 5.1 Rithmic Client (`rithmic_client.py`)

**Purpose:** Connect to Rithmic, authenticate, subscribe to NQ front-month, stream ticks.

**Interface:**
```python
class RithmicClient:
    """Connects to Rithmic via R | Protocol API and streams NQ tick data.

    Manages the full connection lifecycle: connect, authenticate, subscribe
    to front-month NQ contract, stream trades and BBO updates, handle
    disconnections with auto-reconnect, and clean shutdown.
    """

    async def connect(self) -> None
        """Establish WebSocket connection and authenticate."""

    async def disconnect(self) -> None
        """Clean shutdown — unsubscribe and close connection."""

    async def subscribe_market_data(self, symbol: str) -> None
        """Subscribe to trades + BBO for the given symbol."""

    def on_trade(self, callback: Callable[[TradeUpdate], None]) -> None
        """Register callback for trade updates."""

    def on_bbo(self, callback: Callable[[BBOUpdate], None]) -> None
        """Register callback for BBO updates."""

    def on_connection_status(self, callback: Callable[[ConnectionStatus], None]) -> None
        """Register callback for connection state changes."""

    @property
    def is_connected(self) -> bool

    @property
    def connection_status(self) -> ConnectionStatus
```

**Data models:**
```python
@dataclass
class TradeUpdate:
    timestamp: datetime          # Exchange timestamp (UTC)
    price: Decimal               # Trade price
    size: int                    # Trade volume
    aggressor_side: str          # 'BUY' or 'SELL'
    symbol: str                  # e.g., 'NQH6'

@dataclass
class BBOUpdate:
    timestamp: datetime          # Exchange timestamp (UTC)
    bid_price: Decimal           # Best bid
    bid_size: int                # Bid volume
    ask_price: Decimal           # Best ask
    ask_size: int                # Ask volume
    symbol: str

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
```

**Behavior:**
- On connection drop: transition to RECONNECTING, attempt reconnect with exponential backoff (1s, 2s, 4s, 8s, max 30s), log each attempt
- On successful reconnect: re-subscribe to market data, resume streaming
- On authentication failure: transition to ERROR, do not retry (bad credentials won't fix themselves)
- Symbol resolution: auto-detect front-month NQ contract. NQ follows quarterly cycle (H=March, M=June, U=September, Z=December). Roll to next contract approximately 1 week before expiration, or use Rithmic's continuous contract symbol if available.
- All timestamps must be in UTC internally, converted to ET only for display

### 5.2 Tick Recorder (`tick_recorder.py`)

**Purpose:** Persist every tick from the Rithmic stream to daily Parquet files. Completely independent — no dependency on the dashboard or any other component.

**Interface:**
```python
class TickRecorder:
    """Records all incoming tick data to daily Parquet files.

    Operates independently of the rest of the system. Receives tick
    callbacks from the Rithmic client and appends them to a daily
    Parquet file. Creates a new file at each date boundary.

    File format is designed to be compatible with the existing
    Databento MBP-10 Parquet files for future unified backtesting.
    """

    def __init__(self, output_dir: Path) -> None
        """output_dir: base directory for Parquet files (e.g., data/rithmic/NQ/)"""

    def record_trade(self, trade: TradeUpdate) -> None
        """Append a trade to today's file."""

    def record_bbo(self, bbo: BBOUpdate) -> None
        """Append a BBO update to today's file."""

    def flush(self) -> None
        """Force write buffered data to disk."""

    def close(self) -> None
        """Flush and close the current file."""
```

**File organization:**
```
data/rithmic/NQ/
  2026-03-03.parquet     # One file per trading day
  2026-03-04.parquet
  ...
```

**Schema (per-row):**
```
timestamp: datetime64[ns, UTC]   # Exchange timestamp
record_type: str                 # 'trade' or 'bbo'
price: float64                   # Trade price or mid-price
bid_price: float64               # Best bid (null for trades)
ask_price: float64               # Best ask (null for trades)
bid_size: int32                  # Bid volume (null for trades)
ask_size: int32                  # Ask volume (null for trades)
trade_size: int32                # Trade volume (null for BBO)
aggressor_side: str              # 'BUY'/'SELL' (null for BBO)
symbol: str                      # Contract symbol
```

**Behavior:**
- Buffer ticks in memory and flush to disk periodically (every 5 seconds or every 1000 ticks, whichever comes first)
- On date boundary (6:00 PM ET = CME Globex day boundary), close current file and open new one
- On shutdown: flush all remaining data before exiting
- On startup: if a file for today already exists, append to it (don't overwrite)
- Use Parquet row group size of ~50,000 rows for efficient reads
- Compress with snappy (matches Databento files)

### 5.3 Price Buffer (`price_buffer.py`)

**Purpose:** Maintain a rolling window of recent price data in memory for chart rendering and level monitoring in later phases.

**Interface:**
```python
class PriceBuffer:
    """In-memory buffer of recent tick data for real-time chart rendering.

    Maintains a rolling window of trades and BBO updates. Provides
    methods to query recent data for OHLCV candle construction.
    Does not persist to disk — this is ephemeral working memory.
    """

    def __init__(self, max_duration: timedelta = timedelta(hours=48)) -> None

    def add_trade(self, trade: TradeUpdate) -> None
    def add_bbo(self, bbo: BBOUpdate) -> None

    @property
    def latest_price(self) -> Decimal | None

    @property
    def latest_bid(self) -> Decimal | None

    @property
    def latest_ask(self) -> Decimal | None

    @property
    def latest_mid(self) -> Decimal | None

    def get_trades_since(self, since: datetime) -> list[TradeUpdate]
    def get_ohlcv(self, timeframe: str, since: datetime) -> list[OHLCVBar]
```

**Behavior:**
- Evict data older than `max_duration` periodically (not on every insert — batch cleanup)
- OHLCV construction uses trade data only (not BBO). Standard candle logic: first trade = open, max = high, min = low, last = close, sum = volume
- Timeframe string format: "1m", "3m", "5m", "10m", "15m", "30m", "1H", "4H", "1D"
- Thread-safe — the pipeline writes while the API server (Phase 5) reads

### 5.4 Pipeline Service (`pipeline_service.py`)

**Purpose:** Orchestrate startup and shutdown of all pipeline components.

**Interface:**
```python
class PipelineService:
    """Main entry point for the data pipeline.

    Starts the Rithmic client, tick recorder, and price buffer.
    Routes tick data from the client to all consumers. Manages
    the overall service lifecycle.

    Later phases will register additional consumers (level engine,
    observation engine, paper trading engine) through this service.
    """

    async def start(self) -> None
        """Start all components and begin streaming."""

    async def stop(self) -> None
        """Graceful shutdown of all components."""

    @property
    def is_running(self) -> bool

    @property
    def connection_status(self) -> ConnectionStatus

    def register_trade_handler(self, handler: Callable) -> None
        """Register additional trade consumers (for later phases)."""

    def register_bbo_handler(self, handler: Callable) -> None
        """Register additional BBO consumers (for later phases)."""
```

**Behavior:**
- On start: initialize DB connection, start Rithmic client, start tick recorder, wire callbacks
- On trade received: fan out to recorder, price buffer, and any registered handlers
- On BBO received: fan out to recorder, price buffer, and any registered handlers
- On connection status change: log it, store in DB for dashboard visibility
- On stop: flush recorder, disconnect Rithmic client, close DB connection
- Designed for `asyncio` — all I/O is async

### 5.5 PostgreSQL Database (`db/`)

**Purpose:** Operational database for the live trading system.

**Tables for Phase 1:**

```sql
-- Configuration storage
CREATE TABLE config (
    key VARCHAR PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Connection status log (for dashboard to show history)
CREATE TABLE connection_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    status VARCHAR NOT NULL,          -- 'connected', 'disconnected', 'reconnecting', 'error'
    details JSONB,                     -- error messages, reconnect attempt count, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Recent OHLCV bars for chart rendering (populated from price buffer periodically)
-- This allows the dashboard to show chart data without querying the price buffer directly
CREATE TABLE ohlcv_bars (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR NOT NULL,        -- '1m', '5m', etc.
    open DECIMAL(12,2) NOT NULL,
    high DECIMAL(12,2) NOT NULL,
    low DECIMAL(12,2) NOT NULL,
    close DECIMAL(12,2) NOT NULL,
    volume BIGINT NOT NULL,
    symbol VARCHAR NOT NULL,
    UNIQUE(timestamp, timeframe, symbol)
);

-- Placeholder tables that later phases will populate
-- (create the schema now so migrations are clean)

CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSONB,                     -- accuracy, precision, fold results
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ
);
```

**Tech:**
- SQLAlchemy 2.0 with async support (asyncpg driver)
- Alembic for schema migrations
- Connection pooling via SQLAlchemy's built-in pool
- Database URL configurable via environment variable or settings file

### 5.6 Configuration (`config/settings.py`)

**Dashboard-specific settings using Pydantic Settings:**

```python
class DashboardSettings(BaseSettings):
    # Rithmic
    rithmic_username: str
    rithmic_password: SecretStr
    rithmic_system: str = "APEX"
    rithmic_gateway: str = "Chicago Area"

    # Databento (for future historical backfill)
    databento_api_key: SecretStr | None = None

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://localhost:5432/alpha_lab_dashboard"

    # Tick recording
    tick_recording_dir: Path = Path("data/rithmic/NQ")

    # Instrument
    symbol: str = "NQ"
    exchange: str = "CME"

    # Price buffer
    price_buffer_hours: int = 48

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DASHBOARD_",
    )
```

---

## 6. Test Specifications

Every test module starts with a docstring explaining what component it covers, which phase it was built in, and the business context.

### 6.1 `test_rithmic_client.py`

```
"""
Phase 1 — Rithmic Client Tests

Tests the Rithmic connection lifecycle, authentication, market data subscription,
and reconnection behavior. The Rithmic client is the foundation of the entire
live trading system — it provides the real-time tick stream that everything
else depends on.

Business context: The trader uses a $30/month Apex second-login add-on to
get a dedicated Rithmic data session. The client must handle connection drops
gracefully because Rithmic disconnects when markets close and may drop during
volatile periods.
"""
```

**Test cases:**

1. `test_connection_lifecycle` — Client transitions through DISCONNECTED → CONNECTING → CONNECTED → DISCONNECTED states correctly
2. `test_authentication_with_valid_credentials` — Successful auth transitions to CONNECTED
3. `test_authentication_failure` — Invalid credentials transition to ERROR, no retry
4. `test_market_data_subscription` — After connecting, subscribing to NQ produces trade and BBO callbacks
5. `test_trade_callback_data_integrity` — Trade updates contain all required fields with correct types
6. `test_bbo_callback_data_integrity` — BBO updates contain all required fields with correct types
7. `test_auto_reconnect_on_drop` — After unexpected disconnect, client transitions to RECONNECTING and attempts reconnection
8. `test_reconnect_exponential_backoff` — Reconnect attempts use exponential backoff (1s, 2s, 4s, 8s, capped at 30s)
9. `test_reconnect_resubscribes` — After successful reconnect, market data subscription is restored
10. `test_connection_status_callback` — Status changes fire the registered callback
11. `test_multiple_callbacks` — Multiple trade/bbo handlers all receive updates
12. `test_clean_disconnect` — `disconnect()` unsubscribes and closes cleanly without errors
13. `test_front_month_detection` — Client resolves "NQ" to the correct front-month contract symbol based on current date

All Rithmic tests use a mock WebSocket/server — they do not require a live Rithmic connection.

### 6.2 `test_tick_recorder.py`

```
"""
Phase 1 — Tick Recorder Tests

Tests the independent Parquet file recording system. The tick recorder runs
as a completely decoupled process that persists every tick from the Rithmic
stream to daily Parquet files. It has zero dependency on the rest of the
dashboard system.

Business context: Every tick received from Rithmic is stored indefinitely,
building a proprietary historical database that replaces Databento for future
backtesting. The files must be compatible with the existing Databento Parquet
files in data/databento/NQ/ for unified analysis.
"""
```

**Test cases:**

1. `test_creates_daily_file` — First trade creates a new Parquet file named `YYYY-MM-DD.parquet`
2. `test_records_trade_fields` — Trade records contain all schema fields with correct values
3. `test_records_bbo_fields` — BBO records contain all schema fields, trade-specific fields are null
4. `test_appends_to_existing_file` — If today's file exists, new data is appended, not overwritten
5. `test_date_boundary_rollover` — At 6:00 PM ET (CME day boundary), a new file is created
6. `test_flush_writes_to_disk` — After `flush()`, data is readable from the Parquet file
7. `test_periodic_flush` — Buffered data is flushed after 5 seconds or 1000 ticks
8. `test_close_flushes_remaining` — `close()` writes all remaining buffered data
9. `test_output_schema` — Parquet file schema matches the specification exactly
10. `test_snappy_compression` — Files are compressed with snappy
11. `test_empty_recorder_close` — Closing a recorder with no data does not create an empty file
12. `test_concurrent_writes` — Multiple threads can record simultaneously without corruption
13. `test_survives_crash_mid_buffer` — If the process dies, previously flushed data is intact (unflushed data is lost — acceptable)

### 6.3 `test_price_buffer.py`

```
"""
Phase 1 — Price Buffer Tests

Tests the in-memory rolling price buffer that provides recent tick data for
chart rendering and level monitoring. This is ephemeral working memory —
it does not persist to disk.

Business context: The dashboard loads with 48 hours of historical data and
shows real-time updates at 1-second refresh. The price buffer holds recent
ticks in memory for fast access. Later phases will use it for level touch
detection and feature computation.
"""
```

**Test cases:**

1. `test_add_and_retrieve_trade` — Added trades are retrievable via `get_trades_since()`
2. `test_latest_price` — `latest_price` returns the most recent trade price
3. `test_latest_bid_ask_mid` — BBO updates correctly set latest bid, ask, and mid-price
4. `test_eviction_by_age` — Data older than `max_duration` is cleaned up
5. `test_ohlcv_1m_construction` — `get_ohlcv("1m", ...)` produces correct 1-minute candles
6. `test_ohlcv_5m_construction` — 5-minute candles aggregate correctly
7. `test_ohlcv_empty_period` — Timeframes with no trades produce no bars (not bars with zero volume)
8. `test_ohlcv_single_trade_in_bar` — A bar with one trade has open=high=low=close
9. `test_thread_safety` — Concurrent reads and writes do not raise exceptions or produce corrupt data
10. `test_ohlcv_uses_trades_only` — BBO updates do not affect OHLCV candle prices

### 6.4 `test_pipeline_service.py`

```
"""
Phase 1 — Pipeline Service Tests

Tests the main orchestrator that starts and coordinates all pipeline
components. The pipeline service is the entry point for the entire
data pipeline — it wires the Rithmic client to the tick recorder
and price buffer, and provides the interface for later phases to
register additional data consumers.

Business context: The pipeline runs 24/7 independently of the dashboard.
If the dashboard is closed, the pipeline continues streaming and recording.
When the dashboard reconnects, it reads current state from PostgreSQL and
the price buffer.
"""
```

**Test cases:**

1. `test_start_initializes_all_components` — `start()` creates and connects Rithmic client, tick recorder, and price buffer
2. `test_stop_shuts_down_cleanly` — `stop()` flushes recorder, disconnects client, closes DB
3. `test_trade_fanout` — Incoming trades are delivered to recorder, price buffer, and registered handlers
4. `test_bbo_fanout` — Incoming BBO updates are delivered to all consumers
5. `test_connection_status_stored_in_db` — Connection status changes are written to the `connection_events` table
6. `test_register_additional_handler` — `register_trade_handler()` adds a consumer that receives subsequent trades
7. `test_service_lifecycle` — Start → receive data → stop → verify all data persisted
8. `test_handles_rithmic_disconnect` — When Rithmic disconnects, service continues running (recorder flushes, buffer retains data)
9. `test_is_running_property` — Correctly reflects whether the service is active

### 6.5 `test_models.py`

```
"""
Phase 1 — Database Model Tests

Tests the PostgreSQL schema and ORM models. The database stores operational
data for the live trading system — connection events, OHLCV bars for chart
rendering, configuration, and provides the schema foundation for trade
history and account state in later phases.

Business context: PostgreSQL is the single source of truth for all
operational state. The dashboard reads from it, the pipeline writes to it.
Both can operate concurrently without conflicts.
"""
```

**Test cases:**

1. `test_config_crud` — Create, read, update, delete config entries
2. `test_config_jsonb_values` — Config values stored as JSONB preserve nested structures
3. `test_connection_event_insert` — Connection events are stored with correct timestamps and status
4. `test_ohlcv_bar_insert` — OHLCV bars are inserted correctly
5. `test_ohlcv_bar_unique_constraint` — Duplicate (timestamp, timeframe, symbol) is rejected
6. `test_model_version_lifecycle` — Create model version, activate, deactivate
7. `test_only_one_active_model` — Activating a model deactivates all others
8. `test_migration_creates_all_tables` — Running Alembic migrations creates the expected schema

---

## 7. Acceptance Criteria

Phase 1 is complete when:

1. **Rithmic client connects and streams** — Given valid credentials, the client connects to Rithmic, subscribes to NQ, and receives trade + BBO updates through callbacks. (Tested with mock server in unit tests; validated manually with real Rithmic connection.)

2. **Tick recorder writes Parquet files** — Every trade and BBO update from the stream is written to daily Parquet files in `data/rithmic/NQ/`. Files are readable by pandas/DuckDB and contain the correct schema.

3. **Price buffer maintains rolling window** — Recent ticks are queryable in memory. OHLCV candle construction produces correct bars for all supported timeframes.

4. **PostgreSQL operational database is running** — All Phase 1 tables exist. Connection events are logged. Config entries are readable/writable. Alembic migrations are clean.

5. **Pipeline service orchestrates everything** — `PipelineService.start()` brings up all components. Trades flow from Rithmic → recorder + buffer + DB. `PipelineService.stop()` shuts down cleanly with no data loss.

6. **Auto-reconnect works** — When the Rithmic connection drops, the client reconnects with exponential backoff and re-subscribes to market data.

7. **All tests pass** — Every test listed in Section 6 passes. No skipped tests, no known failures.

---

## 8. What Claude Code Should Read First

Before writing any code, Claude Code should read these files to understand the existing project:

1. `src/alpha_lab/experiment/` — Scan the directory to understand the existing experiment code structure
2. `data/databento/NQ/` — Examine one Parquet file to understand the existing schema (use `pyarrow` or `pandas`)
3. `pyproject.toml` — Understand existing dependencies and project configuration
4. `src/alpha_lab/core/config.py` — See how the existing project handles configuration
5. `tests/` — Scan test directory structure to understand existing test organization
6. `.env` or `.env.example` — Check for existing environment variable patterns

Then read the test files from prior phases (none for Phase 1, but this becomes critical for Phases 2+).

---

## 9. Dependencies to Add

```toml
# In pyproject.toml [project.dependencies] or equivalent
# Rithmic connectivity (research current best option)
# pyrithmic or protobuf + websockets for direct implementation

# Database
sqlalchemy = ">=2.0"
asyncpg = "*"           # Async PostgreSQL driver
alembic = "*"           # Schema migrations

# Async
asyncio = "*"           # Standard library, but note project uses it

# Parquet (already in project)
pyarrow = "*"           # Already used for existing Parquet files
```

Do not add unnecessary dependencies. The project already has `pandas`, `numpy`, `pyarrow`, and `pydantic`.

---

## 10. TDD Workflow

For every component in this phase:

1. **Read the spec** for the component
2. **Write the test file first** with all test cases from Section 6, including the module docstring
3. **Run tests** — they should all fail (red)
4. **Implement** the component until all tests pass (green)
5. **Refactor** if needed — tests protect you
6. **Move to the next component**

Order of implementation:
1. Database models and migrations (`db/`) — foundation
2. Configuration (`config/settings.py`) — needed by everything
3. Rithmic client (`rithmic_client.py`) — data source
4. Tick recorder (`tick_recorder.py`) — recording layer
5. Price buffer (`price_buffer.py`) — in-memory layer
6. Pipeline service (`pipeline_service.py`) — orchestrator

Each step builds on the previous. Tests from earlier components serve as context for later ones.

---

## 11. Notes for Claude Code

- **Do not touch existing code.** Phase 1 creates new files only. The existing `src/alpha_lab/` code is read-only context.
- **Use async/await throughout.** The pipeline is I/O bound. Use `asyncio` for the main event loop.
- **Decimal for prices.** NQ prices must use `Decimal` internally to avoid floating point drift. Only convert to float at Parquet write boundaries.
- **UTC internally, ET for display.** All timestamps stored and compared in UTC. Conversion to Eastern Time happens only at the presentation layer (Phase 6).
- **The tick recorder is independent.** It receives callbacks but has no reverse dependency. If you remove the recorder, the pipeline still works. If you remove the pipeline, the recorder's tests still pass.
- **Don't over-engineer.** This is Phase 1. The interfaces will be consumed by Phase 2 code. Design them to be extensible but don't build Phase 2 features.
