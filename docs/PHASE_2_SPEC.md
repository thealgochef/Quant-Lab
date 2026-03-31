# Phase 2 — Level Detection & Observation Engine

**Scope:** Key level computation, level touch detection, 5-minute observation windows, feature computation
**Prerequisite:** Phase 1 complete (Rithmic streaming, tick recording, price buffer, PostgreSQL)
**Produces:** A real-time engine that detects when price touches key levels, runs 5-minute observation windows, and computes the 3 validated features
**Does NOT include:** Model inference, paper trading, API server, frontend

---

## 1. What This Phase Builds

The observation engine is the core intelligence layer between raw tick data and model predictions. It converts the continuous tick stream into discrete, labeled events at key price levels.

**1. Level Engine** — Computes key levels from prior completed sessions (PDH, PDL, Asia/London session highs/lows). Levels update automatically as sessions close. Accepts manual levels from the trader. Merges levels within 3 points into zones.

**2. Touch Detector** — Monitors the live tick stream against active levels. Detects first-touch events when price reaches or surpasses a level. Enforces first-touch-only rule per level per day.

**3. Observation Window Manager** — When a touch is detected, opens a 5-minute observation window. Accumulates tick data during the window. Handles edge cases: feed drops mid-window (discard), time cutoffs (3:49 PM CT), multiple simultaneous touches.

**4. Feature Computer** — At the end of each 5-minute window, computes the 3 validated features from the accumulated tick data. Must produce identical values to the batch experiment code for the same input data.

---

## 2. Critical Context: Feature Parity

The experiment validated that only 3 features have predictive power. The live system MUST compute these identically to the batch experiment. Any divergence means the model is receiving inputs it wasn't trained on.

**The 3 features (all computed during the 5-minute interaction window):**

| Feature | Definition | Computation |
|---------|-----------|-------------|
| `int_time_beyond_level` | Seconds price spends past the level in the adverse direction | For LONG events (touching a LOW level): seconds where trade price < level_price. For SHORT events (touching a HIGH level): seconds where trade price > level_price. Computed from trade timestamps — each trade's "duration" extends until the next trade. |
| `int_time_within_2pts` | Seconds price lingers within 2 NQ points of the level | Seconds where abs(trade_price - level_price) <= 2.0. Same duration logic as above. |
| `int_absorption_ratio` | Volume traded at the level divided by volume traded through the level | Volume where abs(trade_price - level_price) <= 2.0 (at level) / volume where abs(trade_price - level_price) > 2.0 (through level). If denominator is zero, use a large default (e.g., 100.0). |

**Feature parity verification:** After implementation, Claude Code must run a comparison test. Take 5 historical events from the experiment results, replay the raw tick data through the live feature computer, and verify the output matches the batch computation within floating-point tolerance (< 0.01 absolute difference per feature).

**Source code reference:** The batch feature computation lives in `src/alpha_lab/experiment/features.py`. The live code in `src/alpha_lab/dashboard/` should import from or replicate the exact logic. If importing directly, ensure no batch-only dependencies bleed in. If replicating, add a cross-validation test that runs both paths on the same data.

---

## 3. Relevant PRD Sections

- **Section 5** (Key Levels) — Auto-detected levels, manual levels, merging rules, first-touch-only
- **Section 6** (Model Behavior) — Observation window mechanics, feature computation, prediction display scope
- **Section 7** (Connection Handling) — Feed drop mid-observation = discard event
- **Section 9** (Time Rules) — Last new observation at 3:49 PM CT, hard flatten at 3:55 PM CT
- **Section 14** (Edge Cases) — Feed drop, time cutoffs, merged levels, manual level deletion mid-observation

---

## 4. Directory Structure (Phase 2 additions)

```
src/alpha_lab/dashboard/
  engine/
    __init__.py
    level_engine.py          # Key level computation + manual level management
    touch_detector.py        # Price vs. level monitoring, first-touch enforcement
    observation_manager.py   # 5-minute window lifecycle
    feature_computer.py      # The 3 validated features
    models.py                # Data models for levels, events, observations

tests/dashboard/
  engine/
    __init__.py
    test_level_engine.py
    test_touch_detector.py
    test_observation_manager.py
    test_feature_computer.py
    test_feature_parity.py   # Cross-validation against experiment code
```

---

## 5. Component Specifications

### 5.1 Data Models (`engine/models.py`)

```python
class LevelType(Enum):
    PDH = "pdh"
    PDL = "pdl"
    ASIA_HIGH = "asia_high"
    ASIA_LOW = "asia_low"
    LONDON_HIGH = "london_high"
    LONDON_LOW = "london_low"
    MANUAL = "manual"

class LevelSide(Enum):
    HIGH = "high"   # PDH, asia_high, london_high — SHORT reversal
    LOW = "low"     # PDL, asia_low, london_low — LONG reversal

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class KeyLevel:
    level_type: LevelType
    price: Decimal
    side: LevelSide
    available_from: datetime        # UTC — when this level becomes active
    source_session_date: date       # Which session produced this level
    is_manual: bool = False
    zone_id: str | None = None      # Set when merged with other levels

@dataclass
class LevelZone:
    zone_id: str
    representative_price: Decimal   # Average price of constituent levels
    levels: list[KeyLevel]
    is_touched: bool = False
    touched_at: datetime | None = None

@dataclass
class TouchEvent:
    event_id: str                   # UUID
    timestamp: datetime             # UTC — exact tick that triggered the touch
    level_zone: LevelZone
    trade_direction: TradeDirection  # LONG if LOW side, SHORT if HIGH side
    price_at_touch: Decimal         # Trade price that triggered detection
    session: str                    # "asia", "london", "ny_rth", "pre_market", "post_market"

class ObservationStatus(Enum):
    ACTIVE = "active"               # Window in progress
    COMPLETED = "completed"         # Window finished, features computed
    DISCARDED_FEED_DROP = "discarded_feed_drop"
    DISCARDED_TIME_CUTOFF = "discarded_time_cutoff"
    DISCARDED_LEVEL_DELETED = "discarded_level_deleted"

@dataclass
class ObservationWindow:
    event: TouchEvent
    start_time: datetime            # Same as touch timestamp
    end_time: datetime              # start_time + 5 minutes
    status: ObservationStatus
    trades_accumulated: list        # TradeUpdate objects during window
    features: dict | None = None    # Computed after window completes
```

### 5.2 Level Engine (`level_engine.py`)

**Purpose:** Compute and manage active key levels for the current trading day.

**Interface:**
```python
class LevelEngine:
    """Computes key levels from prior sessions and manages manual levels.

    Levels are computed from completed prior sessions only — no look-ahead.
    Manual levels are accepted from the trader and treated identically to
    auto-detected levels. Levels within 3 points are merged into zones.

    This engine needs access to the price buffer (Phase 1) for historical
    OHLCV data to compute session highs/lows.
    """

    def __init__(self, price_buffer: PriceBuffer) -> None

    async def compute_levels_for_today(self) -> list[KeyLevel]
        """Compute all auto-detected levels for the current trading day.
        Called once at pipeline startup and again when sessions close."""

    async def on_session_close(self, session: str) -> None
        """Called when a session closes. Computes new levels from the
        completed session and adds them to the active set."""

    def add_manual_level(self, price: Decimal) -> KeyLevel
        """Add a manual level. Automatically determines side based on
        current price (above current = HIGH, below = LOW)."""

    def remove_manual_level(self, price: Decimal) -> bool
        """Remove a manual level. Returns True if found and removed."""

    def get_active_zones(self) -> list[LevelZone]
        """Return all active (untouched) level zones for today."""

    def mark_zone_touched(self, zone_id: str, touched_at: datetime) -> None
        """Mark a zone as touched. It will no longer be active."""

    def reset_daily(self) -> None
        """Clear manual levels and reset touch state. Called at 6 PM ET."""

    @property
    def all_levels(self) -> list[KeyLevel]
```

**Session boundaries (ET):**
- Asia: 18:00 – 01:00
- London: 01:00 – 08:00
- NY RTH: 09:30 – 16:15
- Trading day boundary: 18:00 ET (6 PM)

**Level computation logic:**
- PDH/PDL: High/low of prior completed NY RTH session. Available before current day RTH open.
- Asia High/Low: High/low of completed Asia session (18:00–01:00 ET). Available after 01:00 ET.
- London High/Low: High/low of completed London session (01:00–08:00 ET). Available after 08:00 ET.

**Zone merging:** After computing all levels, merge any levels within 3.0 NQ points of each other into a single zone. The zone's representative price is the average of constituent level prices. Use single-linkage clustering (if A is within 3 pts of B, and B is within 3 pts of C, all three merge even if A and C are > 3 pts apart).

**Startup behavior:** On first startup, compute levels from the price buffer's historical data. If the buffer doesn't have enough history (e.g., first day running), use Databento backfill data or start with only the levels that can be computed.

### 5.3 Touch Detector (`touch_detector.py`)

**Purpose:** Monitor the live tick stream and detect when price first touches an active level zone.

**Interface:**
```python
class TouchDetector:
    """Monitors live trades against active level zones.

    Registers as a trade handler on the pipeline service. For each
    incoming trade, checks whether the trade price touches any active
    (untouched) zone. On first touch, emits a TouchEvent and marks
    the zone as spent for the day.
    """

    def __init__(self, level_engine: LevelEngine) -> None

    def on_trade(self, trade: TradeUpdate) -> TouchEvent | None
        """Process a trade. Returns a TouchEvent if this trade triggers
        a level touch, otherwise None."""

    def on_touch(self, callback: Callable[[TouchEvent], None]) -> None
        """Register callback for touch events."""

    @property
    def active_zone_count(self) -> int
```

**Touch logic:**
- For HIGH-side zones: trade price >= zone representative_price
- For LOW-side zones: trade price <= zone representative_price
- Mixed zones (rare — contains both HIGH and LOW levels): trigger on either direction, determine trade direction from approach (if price came from above → LONG, from below → SHORT)

**First-touch-only enforcement:** Once a zone is touched, it's marked as spent. No subsequent touches of that zone trigger events for the rest of the trading day. Reset happens at 6 PM ET.

**Time filtering:**
- No new touch events after 3:49 PM CT (14:49 CT = 15:49 ET)
- This ensures any 5-minute window completes before the 3:55 PM CT hard flatten

### 5.4 Observation Window Manager (`observation_manager.py`)

**Purpose:** Manage the lifecycle of 5-minute observation windows after touch detection.

**Interface:**
```python
class ObservationManager:
    """Manages 5-minute observation windows after level touches.

    When a TouchEvent fires, opens an observation window that accumulates
    all trades for 5 minutes. At window completion, triggers feature
    computation. Handles edge cases: feed drops, time cutoffs, level
    deletion mid-observation.
    """

    def __init__(self, feature_computer: FeatureComputer) -> None

    def start_observation(self, event: TouchEvent) -> ObservationWindow
        """Open a new 5-minute observation window."""

    def on_trade(self, trade: TradeUpdate) -> None
        """Feed trades to any active observation window."""

    def on_connection_status(self, status: ConnectionStatus) -> None
        """Handle feed drops — discard active windows."""

    def on_level_deleted(self, level_price: Decimal) -> None
        """Handle manual level deletion mid-observation."""

    def on_observation_complete(self, callback: Callable[[ObservationWindow], None]) -> None
        """Register callback for completed observations with features."""

    @property
    def active_observation(self) -> ObservationWindow | None
        """At most one observation can be active at a time."""
```

**Window lifecycle:**
1. Touch detected → window opens, start timer
2. During window: accumulate every trade
3. At 5 minutes: close window, compute features, fire callback
4. If feed drops during window: discard entirely (status = DISCARDED_FEED_DROP), flag on chart
5. If manual level deleted during window: discard (status = DISCARDED_LEVEL_DELETED)
6. Only one observation window can be active at a time. If a new touch fires while a window is active, ignore the new touch (the first observation takes priority).

### 5.5 Feature Computer (`feature_computer.py`)

**Purpose:** Compute the 3 validated features from accumulated trade data within a completed observation window.

**Interface:**
```python
class FeatureComputer:
    """Computes the 3 CatBoost features from observation window tick data.

    These features were validated through the hypothesis experiment with
    86.1% reversal precision. They must be computed identically to the
    batch experiment code in src/alpha_lab/experiment/features.py.
    """

    def compute_features(
        self,
        trades: list[TradeUpdate],
        level_price: Decimal,
        direction: TradeDirection,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[str, float]
        """Compute the 3 features from observation window trades.

        Returns:
            {
                "int_time_beyond_level": float,  # seconds
                "int_time_within_2pts": float,    # seconds
                "int_absorption_ratio": float,    # ratio
            }
        """
```

**Feature computation detail:**

**`int_time_beyond_level`** — Time price spends in the adverse direction past the level.
```
For each consecutive pair of trades (trade_i, trade_{i+1}):
    duration = trade_{i+1}.timestamp - trade_i.timestamp
    
    if direction == LONG:
        if trade_i.price < level_price:
            time_beyond += duration
    elif direction == SHORT:
        if trade_i.price > level_price:
            time_beyond += duration

For the last trade, duration extends to window_end.
Result is total seconds.
```

**`int_time_within_2pts`** — Time price lingers near the level.
```
Same duration logic as above, but condition is:
    abs(trade_i.price - level_price) <= 2.0

Result is total seconds.
```

**`int_absorption_ratio`** — Volume concentration at the level vs. through it.
```
at_level_volume = sum(trade.size for trade in trades
                      if abs(trade.price - level_price) <= 2.0)

through_level_volume = sum(trade.size for trade in trades
                          if abs(trade.price - level_price) > 2.0)

if through_level_volume == 0:
    absorption_ratio = 100.0  # sentinel for pure absorption
else:
    absorption_ratio = at_level_volume / through_level_volume
```

---

## 6. PostgreSQL Tables (Phase 2 additions)

```sql
-- Active levels for today (refreshed daily, read by dashboard)
CREATE TABLE active_levels (
    id SERIAL PRIMARY KEY,
    level_type VARCHAR NOT NULL,         -- 'pdh', 'pdl', 'asia_high', etc.
    price DECIMAL(12,2) NOT NULL,
    side VARCHAR NOT NULL,               -- 'high' or 'low'
    zone_id VARCHAR,
    is_manual BOOLEAN DEFAULT FALSE,
    is_touched BOOLEAN DEFAULT FALSE,
    touched_at TIMESTAMPTZ,
    available_from TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    trading_date DATE NOT NULL
);

-- Observation events (historical record of every touch + outcome)
CREATE TABLE observation_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    level_type VARCHAR NOT NULL,
    level_price DECIMAL(12,2) NOT NULL,
    zone_id VARCHAR,
    trade_direction VARCHAR NOT NULL,     -- 'long' or 'short'
    price_at_touch DECIMAL(12,2) NOT NULL,
    session VARCHAR NOT NULL,
    status VARCHAR NOT NULL,              -- 'completed', 'discarded_feed_drop', etc.
    -- Features (null if discarded)
    int_time_beyond_level FLOAT,
    int_time_within_2pts FLOAT,
    int_absorption_ratio FLOAT,
    -- Prediction (populated by Phase 3)
    prediction VARCHAR,                   -- 'tradeable_reversal', 'trap_reversal', 'aggressive_blowthrough'
    prediction_probabilities JSONB,
    -- Outcome tracking (populated by Phase 3)
    outcome_resolved BOOLEAN DEFAULT FALSE,
    outcome_correct BOOLEAN,
    mfe_points FLOAT,
    mae_points FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    trading_date DATE NOT NULL
);
```

---

## 7. Test Specifications

### 7.1 `test_level_engine.py` (14 tests)

```
"""
Phase 2 — Level Engine Tests

Tests key level computation from historical session data, manual level
management, and zone merging. The level engine determines WHERE the model
looks — incorrect levels mean the model evaluates the wrong price points.

Business context: Key levels (PDH, PDL, session highs/lows) are the
pre-computed trigger points where the CatBoost model evaluates order flow.
The trader also adds manual levels from their own analysis. All levels
within 3 NQ points merge into zones.
"""
```

1. `test_pdh_pdl_computation` — PDH and PDL match the prior RTH session high/low
2. `test_asia_session_high_low` — Asia levels computed from 18:00–01:00 ET window
3. `test_london_session_high_low` — London levels computed from 01:00–08:00 ET window
4. `test_level_available_from_timestamp` — PDH/PDL available before RTH open, Asia levels after 01:00, London after 08:00
5. `test_no_look_ahead` — Levels from incomplete sessions are never returned
6. `test_on_session_close_adds_levels` — When London closes, london_high and london_low appear in active levels
7. `test_add_manual_level` — Manual level is added and appears in active zones
8. `test_remove_manual_level` — Removed manual level disappears from active zones
9. `test_manual_level_side_detection` — Manual level above current price = HIGH side, below = LOW side
10. `test_zone_merging_within_3pts` — Two levels within 3.0 points merge into one zone
11. `test_zone_merging_chain` — Three levels chain-merge via single linkage
12. `test_no_merge_beyond_3pts` — Two levels 3.5 points apart remain separate zones
13. `test_daily_reset` — `reset_daily()` clears manual levels and touch state
14. `test_level_persistence` — Levels written to `active_levels` DB table on computation

### 7.2 `test_touch_detector.py` (12 tests)

```
"""
Phase 2 — Touch Detector Tests

Tests real-time detection of price touching key level zones. The touch
detector converts the continuous tick stream into discrete events that
trigger observation windows.

Business context: A "touch" is the trigger for the entire prediction
pipeline. Missing a touch means missing a trading opportunity. False
touches waste model computation. First-touch-only prevents duplicate
signals on the same level.
"""
```

1. `test_touch_high_level` — Trade at or above a HIGH zone triggers SHORT touch event
2. `test_touch_low_level` — Trade at or below a LOW zone triggers LONG touch event
3. `test_first_touch_only` — Second touch of same zone returns None
4. `test_no_touch_below_zone` — Trade price below a HIGH zone doesn't trigger
5. `test_callback_fires` — Registered callback receives TouchEvent on touch
6. `test_multiple_zones_independent` — Touching zone A doesn't affect zone B
7. `test_time_cutoff_349pm` — Touches after 3:49 PM CT are ignored
8. `test_touch_at_349pm_exactly` — Touch at exactly 3:49 PM CT is allowed (window completes at 3:54)
9. `test_daily_reset_re_enables_zones` — After reset, previously touched zones can be touched again
10. `test_mixed_zone_direction` — Mixed zone uses approach direction to determine trade direction
11. `test_touch_event_fields` — TouchEvent contains all required fields with correct types
12. `test_no_active_zones_no_detection` — When all zones are spent, trades pass through without detection

### 7.3 `test_observation_manager.py` (11 tests)

```
"""
Phase 2 — Observation Window Manager Tests

Tests the 5-minute observation window lifecycle from open to close,
including edge cases like feed drops and manual level deletion.

Business context: The observation window accumulates tick data that
the feature computer uses. Incomplete windows (feed drops) produce
unreliable features and must be discarded. The model was trained on
complete 5-minute windows only.
"""
```

1. `test_start_observation_opens_window` — After start, active_observation is set
2. `test_trades_accumulated_during_window` — Trades added between start and end appear in window
3. `test_window_completes_after_5_minutes` — Callback fires with completed observation after 5 min
4. `test_features_computed_on_completion` — Completed observation has non-None features dict
5. `test_feed_drop_discards_window` — Connection status change to DISCONNECTED discards active window
6. `test_discarded_window_flagged` — Discarded window has status DISCARDED_FEED_DROP
7. `test_level_deletion_discards_window` — Deleting the observed level mid-window discards it
8. `test_only_one_active_window` — Starting a second observation while one is active is rejected
9. `test_no_trades_after_window_close` — Trades arriving after 5 minutes don't accumulate
10. `test_empty_window_still_computes` — Window with zero trades (unlikely but possible) produces features with zero/default values
11. `test_observation_stored_in_db` — Completed observation is written to `observation_events` table

### 7.4 `test_feature_computer.py` (10 tests)

```
"""
Phase 2 — Feature Computer Tests

Tests the computation of the 3 validated CatBoost features from
observation window tick data. Feature parity with the batch experiment
is critical — divergence means the model receives inputs it wasn't
trained on.

Business context: The experiment proved that ONLY these 3 features
(time beyond level, time within 2pts, absorption ratio) have predictive
power. Reversal precision is 86.1% when computed correctly. Wrong
feature values = wrong predictions = lost money.
"""
```

1. `test_time_beyond_level_long` — LONG event: correctly sums time where price < level
2. `test_time_beyond_level_short` — SHORT event: correctly sums time where price > level
3. `test_time_within_2pts` — Correctly sums time where price is within 2.0 points of level
4. `test_absorption_ratio_high_absorption` — Heavy volume at level / light volume through = high ratio
5. `test_absorption_ratio_blowthrough` — Light volume at level / heavy through = low ratio
6. `test_absorption_ratio_zero_through` — Zero through-level volume returns 100.0 sentinel
7. `test_duration_calculation_last_trade` — Last trade's duration extends to window_end
8. `test_empty_trades_returns_zeros` — No trades produces {0.0, 0.0, 0.0}
9. `test_single_trade_at_level` — One trade at the level: time = full window, absorption = 100.0
10. `test_feature_values_are_floats` — All returned values are Python floats, not Decimals

### 7.5 `test_feature_parity.py` (5 tests)

```
"""
Phase 2 — Feature Parity Tests

Validates that the live feature computer produces identical results
to the batch experiment code in src/alpha_lab/experiment/features.py
for the same input data. This is the most important test in the
entire system — if features diverge, the model is useless.

These tests replay historical events from the experiment through the
live feature computer and compare outputs.
"""
```

1. `test_parity_event_1` — Replay historical event 1, compare features within tolerance
2. `test_parity_event_2` — Replay historical event 2
3. `test_parity_event_3` — Replay historical event 3
4. `test_parity_event_4` — Replay historical event 4
5. `test_parity_event_5` — Replay historical event 5

**How to implement:** Load 5 events from `data/experiment/labeled_events.parquet`. For each, load the raw tick data from the corresponding 5-minute window from the Parquet files. Run through both the batch `features.py` and the live `FeatureComputer`. Assert all 3 feature values match within 0.01 absolute tolerance.

---

## 8. Integration with Phase 1

The observation engine registers as a consumer on the Phase 1 pipeline:

```python
# In pipeline_service.py startup (Phase 2 additions):
pipeline.register_trade_handler(touch_detector.on_trade)
pipeline.register_trade_handler(observation_manager.on_trade)
pipeline.register_connection_handler(observation_manager.on_connection_status)
```

**Data flow:**
```
Rithmic tick → Pipeline Service → Touch Detector → Observation Manager → Feature Computer → [Phase 3]
                                → Tick Recorder (unchanged)
                                → Price Buffer (unchanged)
```

---

## 9. Acceptance Criteria

Phase 2 is complete when:

1. **Level engine computes correct levels** — PDH/PDL and session highs/lows match the experiment's `key_levels.parquet` for historical dates
2. **Touch detector fires on first touch only** — Price reaching a level triggers exactly one event per zone per day
3. **Observation windows complete correctly** — 5-minute windows accumulate trades and compute features
4. **Feed drops discard windows** — Disconnection during an observation produces a discarded event, not a partial prediction
5. **Feature parity verified** — All 5 parity tests pass within tolerance
6. **All tests pass** — Every test in Section 7 passes

---

## 10. What Claude Code Should Read First

1. `src/alpha_lab/experiment/features.py` — The batch feature computation (source of truth)
2. `src/alpha_lab/experiment/key_levels.py` — Existing level computation logic
3. `src/alpha_lab/experiment/event_detection.py` — Existing event detection logic
4. `data/experiment/key_levels.parquet` — Computed levels from experiment
5. `data/experiment/labeled_events.parquet` — Events with features for parity testing
6. Phase 1 test files — Understand the interfaces already built
7. `config/settings.yaml` — Session boundary definitions

---

## 11. Notes for Claude Code

- **Feature parity is the #1 priority.** If you have to choose between clean architecture and exact feature reproduction, choose parity. The model was trained on specific feature values — the live code must match them exactly.
- **Use Decimal for price comparisons.** The 2.0-point proximity check and level touch detection must use Decimal arithmetic to avoid floating-point edge cases at NQ's 0.25 tick size.
- **Time zones matter.** Session boundaries are defined in ET. Rithmic timestamps arrive in UTC. All internal computation uses UTC. Conversion to ET happens only for session boundary checks and display.
- **The observation window is exactly 5 minutes.** Not "approximately" — it starts at the touch timestamp and ends exactly 300 seconds later. Use asyncio timers, not polling.
- **One observation at a time.** The PRD does not specify concurrent observations. If price touches two levels within 5 minutes of each other, the first observation takes priority and the second touch is ignored.
