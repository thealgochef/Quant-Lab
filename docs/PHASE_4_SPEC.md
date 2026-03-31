# Phase 4 — Paper Trading Engine

**Scope:** Apex 4.0 account simulation, multi-account orchestration, position management, trade execution
**Prerequisite:** Phase 3 complete (model inference, predictions, outcome tracking)
**Produces:** A paper trading engine that auto-executes trades across 5 simulated Apex accounts when the model predicts "tradeable_reversal" during NY RTH
**Does NOT include:** API server, frontend

---

## 1. What This Phase Builds

**1. Apex Account Simulator** — Full simulation of an Apex 4.0 50K Intraday Trailing Drawdown Performance Account. Tracks balance, trailing drawdown, DLL, tier-based scaling, payout eligibility, 50% consistency rule, and full account lifecycle from creation to retirement (6 payouts) or termination (drawdown breach).

**2. Account Manager** — Manages the portfolio of simulated accounts. Groups them into Group A (3 accounts, 15pt TP) and Group B (2 accounts, 30pt TP). Handles adding, replacing, and retiring accounts.

**3. Trade Executor** — When a prediction is flagged as executable (reversal during NY RTH), places paper trades across all active, eligible accounts simultaneously. Manages TP/SL orders, position monitoring, and the hard 3:55 PM CT flatten.

**4. Position Monitor** — Tracks open positions against the live tick stream. Detects TP/SL hits, DLL breaches, and trailing drawdown violations in real time.

---

## 2. Apex 4.0 Rules Reference

Full rules are in PRD Section 12. Key parameters:

| Parameter | Value |
|-----------|-------|
| Starting balance | $50,000 |
| Trailing drawdown | $2,000 |
| Starting liquidation threshold | $48,000 |
| Safety net | $52,100 |
| Max payouts | 6 per account |
| Min qualifying days between payouts | 5 ($200+ profit each) |
| 50% consistency rule | Best day profit ≤ 50% of total profit |
| Payout split | 100% to trader |

**Tier-based scaling:**

| Tier | Profit Range | Max Contracts | DLL |
|------|-------------|--------------|-----|
| Level 1 | $0–$1,499 | 2 | $1,000 |
| Level 2 | $1,500–$2,999 | 3 | $1,000 |
| Level 3 | $3,000–$5,999 | 4 | $2,000 |
| Level 4 | $6,000+ | 4 | $3,000 |

**Payout caps:** $1,500 / $2,000 / $2,500 / $2,500 / $3,000 / $3,000

**Trailing drawdown mechanics:**
- Starts at $48,000
- Trails upward with peak unrealized balance in real time
- Stops trailing at safety net ($52,100)
- Once safety net reached, liquidation threshold locks at $50,100

---

## 3. Directory Structure (Phase 4 additions)

```
src/alpha_lab/dashboard/
  trading/
    __init__.py
    apex_account.py          # Single account simulation
    account_manager.py       # Multi-account portfolio management
    trade_executor.py        # Trade placement and orchestration
    position_monitor.py      # Live P&L tracking, TP/SL/DLL enforcement

tests/dashboard/
  trading/
    __init__.py
    test_apex_account.py
    test_account_manager.py
    test_trade_executor.py
    test_position_monitor.py
```

---

## 4. Component Specifications

### 4.1 Apex Account Simulator (`apex_account.py`)

**Purpose:** Simulate a single Apex 4.0 50K account with all lifecycle rules.

**Interface:**
```python
class ApexAccount:
    """Simulates one Apex 4.0 50K Intraday Trailing Drawdown PA.

    Tracks balance, trailing drawdown, tier, DLL, payout eligibility,
    and consistency rules. Enforces all Apex 4.0 lifecycle constraints
    from activation through 6 payouts to retirement.
    """

    def __init__(
        self,
        account_id: str,
        label: str,                      # User-assigned name
        eval_cost: Decimal,              # Actual price paid for eval
        activation_cost: Decimal,        # PA activation fee
        group: str,                      # "A" or "B"
    ) -> None

    # --- Balance & State ---
    @property
    def balance(self) -> Decimal
    @property
    def profit(self) -> Decimal              # balance - 50000
    @property
    def status(self) -> AccountStatus        # ACTIVE, BLOWN, RETIRED
    @property
    def tier(self) -> int                    # 1-4
    @property
    def max_contracts(self) -> int           # Based on tier
    @property
    def daily_loss_limit(self) -> Decimal    # Based on tier

    # --- Trailing Drawdown ---
    @property
    def liquidation_threshold(self) -> Decimal
    @property
    def peak_balance(self) -> Decimal
    @property
    def safety_net_reached(self) -> bool
    @property
    def trailing_dd_remaining(self) -> Decimal  # Current balance - liquidation

    # --- Payout ---
    @property
    def payout_number(self) -> int           # 0-6 (next payout number)
    @property
    def qualifying_days(self) -> int         # Days with $200+ profit toward next payout
    @property
    def max_payout_amount(self) -> Decimal   # Based on payout_number
    @property
    def consistency_rule_met(self) -> bool   # Best day ≤ 50% of total profit
    @property
    def payout_eligible(self) -> bool        # All conditions met

    # --- DLL ---
    @property
    def daily_pnl(self) -> Decimal
    @property
    def dll_remaining(self) -> Decimal
    @property
    def dll_locked(self) -> bool             # True if DLL breached today

    # --- Trading ---
    @property
    def has_position(self) -> bool
    @property
    def current_position(self) -> Position | None

    def open_position(self, direction: TradeDirection, entry_price: Decimal,
                      contracts: int = 1) -> Position
    def close_position(self, exit_price: Decimal, reason: str) -> ClosedTrade
    def update_unrealized(self, current_price: Decimal) -> None
        """Update unrealized P&L and trailing drawdown."""

    def start_new_day(self) -> None
        """Reset daily counters (DLL, daily P&L). Called at session open."""

    def request_payout(self, amount: Decimal) -> bool
        """Attempt a payout. Returns False if not eligible."""

    def to_dict(self) -> dict
        """Serialize full account state for API/dashboard."""
```

**Account statuses:**
```python
class AccountStatus(Enum):
    ACTIVE = "active"
    DLL_LOCKED = "dll_locked"       # Locked for rest of day
    BLOWN = "blown"                 # Trailing DD breached
    RETIRED = "retired"             # 6 payouts complete
```

**Position and trade models:**
```python
@dataclass
class Position:
    account_id: str
    direction: TradeDirection
    entry_price: Decimal
    contracts: int
    entry_time: datetime
    unrealized_pnl: Decimal = Decimal("0")
    mfe: Decimal = Decimal("0")
    mae: Decimal = Decimal("0")

@dataclass
class ClosedTrade:
    account_id: str
    direction: TradeDirection
    entry_price: Decimal
    exit_price: Decimal
    contracts: int
    entry_time: datetime
    exit_time: datetime
    pnl: Decimal                    # In dollars
    pnl_points: Decimal             # In NQ points
    exit_reason: str                # 'tp', 'sl', 'flatten', 'manual', 'dll'
    group: str
```

**Trailing drawdown update (called on every tick while position open):**
```python
def update_unrealized(self, current_price):
    # Calculate unrealized P&L
    if position.direction == LONG:
        unrealized = (current_price - position.entry_price) * 20  # NQ $20/point
    else:
        unrealized = (position.entry_price - current_price) * 20

    position.unrealized_pnl = unrealized
    position.mfe = max(position.mfe, unrealized)
    position.mae = min(position.mae, unrealized)

    # Check trailing drawdown
    current_equity = self.balance + unrealized
    if current_equity > self.peak_balance:
        self.peak_balance = current_equity
        if not self.safety_net_reached:
            # Trail the liquidation threshold up
            self._liquidation_threshold = self.peak_balance - Decimal("2000")
            if self.peak_balance >= Decimal("52100"):
                self._safety_net_reached = True
                self._liquidation_threshold = Decimal("50100")  # Lock it

    # Check if blown
    if current_equity <= self._liquidation_threshold:
        self._status = AccountStatus.BLOWN
        # Force close at liquidation price
```

### 4.2 Account Manager (`account_manager.py`)

**Purpose:** Manage the portfolio of simulated Apex accounts.

**Interface:**
```python
class AccountManager:
    """Manages the portfolio of simulated Apex 4.0 accounts.

    Organizes accounts into Group A (smaller TP, consistency builders)
    and Group B (larger TP, runners). Handles adding new accounts,
    replacing blown/retired ones, and aggregate portfolio stats.
    """

    def __init__(self, db_session) -> None

    async def add_account(self, label: str, eval_cost: Decimal,
                          activation_cost: Decimal, group: str) -> ApexAccount

    async def get_active_accounts(self) -> list[ApexAccount]
    async def get_accounts_by_group(self, group: str) -> list[ApexAccount]
    async def get_all_accounts(self) -> list[ApexAccount]  # Including blown/retired

    async def get_tradeable_accounts(self) -> list[ApexAccount]
        """Active accounts not DLL-locked and not currently in a position."""

    def get_portfolio_summary(self) -> dict
        """Aggregate stats: total invested, total withdrawn, net P&L, ROI, etc."""

    async def start_new_day(self) -> None
        """Reset daily counters for all active accounts."""

    async def save_state(self) -> None
        """Persist all account state to PostgreSQL."""

    async def load_state(self) -> None
        """Load account state from PostgreSQL on startup."""
```

**Default configuration:**
- Group A: 3 accounts, TP = 15 points ($300), SL = configurable
- Group B: 2 accounts, TP = 30 points ($600), SL = configurable
- All accounts: 1 contract per trade (Level 1 starting)

### 4.3 Trade Executor (`trade_executor.py`)

**Purpose:** Execute paper trades across accounts when predictions fire.

**Interface:**
```python
class TradeExecutor:
    """Executes paper trades across simulated Apex accounts.

    When the prediction engine produces an executable signal (reversal
    during NY RTH), this executor places trades on all eligible accounts
    simultaneously. Manages the configurable second-signal mode (ignore
    or flip) and the no-hedging constraint.
    """

    def __init__(self, account_manager: AccountManager) -> None

    async def on_prediction(self, prediction: Prediction) -> list[Position]
        """Handle an executable prediction. Opens positions on all eligible
        accounts. Returns the list of opened positions."""

    async def close_all_positions(self, reason: str = "manual") -> list[ClosedTrade]
        """Close all positions across all accounts."""

    async def close_account_position(self, account_id: str,
                                     reason: str = "manual") -> ClosedTrade | None
        """Close position on a single account."""

    async def manual_entry(self, account_id: str, direction: TradeDirection,
                           price: Decimal) -> Position | None
        """Manual trade entry on a specific account."""

    async def hard_flatten(self, current_price: Decimal) -> list[ClosedTrade]
        """3:55 PM CT hard flatten. Close ALL positions, cancel ALL pending.
        Absolute, non-configurable, no exceptions."""

    def on_trade_opened(self, callback: Callable) -> None
    def on_trade_closed(self, callback: Callable) -> None

    # Configuration
    second_signal_mode: str = "ignore"  # "ignore" or "flip"
```

**Execution logic:**
```
on_prediction(prediction):
    if not prediction.is_executable:
        return []

    eligible = account_manager.get_tradeable_accounts()

    for account in eligible:
        # No-hedging check: if account has position in opposite direction
        if account.has_position:
            if second_signal_mode == "ignore":
                continue  # Skip this account
            elif second_signal_mode == "flip":
                close_position(account, current_price, "flip")

        # Check DLL would not be immediately breached
        # Check max contracts for tier
        contracts = min(1, account.max_contracts)  # Always 1 for now

        position = account.open_position(
            direction=prediction.trade_direction,
            entry_price=prediction.level_price,
            contracts=contracts,
        )

    return opened_positions
```

**TP/SL per group:**
```
Group A: TP = level_price ± 15 points, SL = level_price ∓ configurable
Group B: TP = level_price ± 30 points, SL = level_price ∓ configurable
```

**No-hedging absolute constraint:** At no point can one account be long while another is short. The executor must verify direction consistency before opening any position.

### 4.4 Position Monitor (`position_monitor.py`)

**Purpose:** Monitor open positions against live tick stream.

**Interface:**
```python
class PositionMonitor:
    """Monitors open positions and enforces TP/SL/DLL/flatten rules.

    Processes every trade tick against all open positions. Detects
    TP hits, SL hits, DLL breaches, trailing drawdown violations,
    and the hard 3:55 PM CT flatten.
    """

    def __init__(self, account_manager: AccountManager,
                 trade_executor: TradeExecutor) -> None

    def on_trade(self, trade: TradeUpdate) -> list[ClosedTrade]
        """Process a tick against all open positions.
        Returns list of any closed trades."""

    async def check_time_rules(self) -> list[ClosedTrade]
        """Check if hard flatten time reached. Called periodically."""

    def get_group_tp(self, group: str) -> Decimal
    def set_group_tp(self, group: str, points: Decimal) -> None
    def get_group_sl(self, group: str) -> Decimal
    def set_group_sl(self, group: str, points: Decimal) -> None
```

**Per-tick processing:**
```
for each account with open position:
    1. Update unrealized P&L and MFE/MAE
    2. Update trailing drawdown (may blow account)
    3. Check TP hit (based on account's group TP setting)
    4. Check SL hit (based on account's group SL setting)
    5. Check DLL breach (cumulative daily realized + unrealized loss)
    6. If any trigger hit: close position, update account state
```

**Hard flatten (3:55 PM CT = 15:55 ET):**
- Runs on a timer, checked every second in the last 5 minutes of RTH
- At exactly 15:55 ET: close ALL positions on ALL accounts
- This is non-configurable and absolute
- Provides 4-minute buffer before Apex's own forced closure

---

## 5. PostgreSQL Tables (Phase 4 additions)

```sql
-- Simulated Apex accounts
CREATE TABLE apex_accounts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR UNIQUE NOT NULL,
    label VARCHAR NOT NULL,
    group_name VARCHAR NOT NULL,         -- 'A' or 'B'
    eval_cost DECIMAL(10,2) NOT NULL,
    activation_cost DECIMAL(10,2) NOT NULL,
    balance DECIMAL(12,2) NOT NULL DEFAULT 50000.00,
    peak_balance DECIMAL(12,2) NOT NULL DEFAULT 50000.00,
    liquidation_threshold DECIMAL(12,2) NOT NULL DEFAULT 48000.00,
    safety_net_reached BOOLEAN DEFAULT FALSE,
    status VARCHAR NOT NULL DEFAULT 'active',
    tier INTEGER NOT NULL DEFAULT 1,
    payout_number INTEGER NOT NULL DEFAULT 0,
    qualifying_days INTEGER NOT NULL DEFAULT 0,
    total_payouts DECIMAL(12,2) NOT NULL DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    blown_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ
);

-- Trade history
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES apex_accounts(account_id),
    event_id VARCHAR,                    -- Links to observation_events
    direction VARCHAR NOT NULL,
    entry_price DECIMAL(12,2) NOT NULL,
    exit_price DECIMAL(12,2),
    contracts INTEGER NOT NULL DEFAULT 1,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    pnl DECIMAL(12,2),                   -- Dollars
    pnl_points DECIMAL(8,2),             -- NQ points
    exit_reason VARCHAR,
    group_name VARCHAR NOT NULL,
    is_open BOOLEAN DEFAULT TRUE,
    mfe_points DECIMAL(8,2) DEFAULT 0,
    mae_points DECIMAL(8,2) DEFAULT 0,
    trading_date DATE NOT NULL
);

-- Daily account snapshots (for equity curves and analysis)
CREATE TABLE daily_account_snapshots (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES apex_accounts(account_id),
    trading_date DATE NOT NULL,
    opening_balance DECIMAL(12,2) NOT NULL,
    closing_balance DECIMAL(12,2),
    daily_pnl DECIMAL(12,2),
    trades_count INTEGER DEFAULT 0,
    tier INTEGER NOT NULL,
    liquidation_threshold DECIMAL(12,2),
    UNIQUE(account_id, trading_date)
);

-- Payout history
CREATE TABLE payouts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR NOT NULL REFERENCES apex_accounts(account_id),
    payout_number INTEGER NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    balance_before DECIMAL(12,2) NOT NULL,
    balance_after DECIMAL(12,2) NOT NULL,
    requested_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trading configuration
-- (Uses existing config table from Phase 1)
-- Keys: 'group_a_tp', 'group_b_tp', 'group_a_sl', 'group_b_sl', 'second_signal_mode'
```

---

## 6. Test Specifications

### 6.1 `test_apex_account.py` (20 tests)

```
"""
Phase 4 — Apex Account Simulator Tests

Tests the full Apex 4.0 50K account lifecycle simulation. Every rule
that governs real Apex accounts must be accurately replicated — the
entire purpose of paper trading is to validate the strategy under
production constraints before risking real money.

Business context: The trader runs multiple Apex 50K PA accounts with
$2,000 trailing drawdown. A blown account means ~$200 in eval + activation
costs lost. Accurate simulation prevents deploying a strategy that looks
profitable in theory but fails under Apex's specific constraints.
"""
```

1. `test_initial_state` — New account starts at $50K, tier 1, liquidation at $48K
2. `test_open_close_position` — Open and close a winning trade, balance updates correctly
3. `test_profit_updates_tier` — $1,500 profit moves account from tier 1 to tier 2
4. `test_tier_max_contracts` — Tier 1=2, tier 2=3, tier 3=4, tier 4=4
5. `test_dll_tier_1` — DLL is $1,000 at tiers 1-2
6. `test_dll_tier_3` — DLL is $2,000 at tier 3
7. `test_dll_tier_4` — DLL is $3,000 at tier 4
8. `test_dll_breach_locks_account` — Exceeding DLL locks account for the day
9. `test_dll_resets_daily` — `start_new_day()` clears DLL lock
10. `test_trailing_dd_trails_up` — Winning position moves liquidation threshold up
11. `test_trailing_dd_does_not_trail_down` — Losing position doesn't move threshold
12. `test_safety_net_locks_threshold` — Peak balance reaching $52,100 locks liquidation at $50,100
13. `test_blown_on_threshold_breach` — Balance hitting liquidation threshold blows account
14. `test_qualifying_day_200_profit` — Day with $200+ profit counts as qualifying
15. `test_qualifying_day_under_200` — Day with < $200 profit doesn't count
16. `test_payout_eligibility` — 5 qualifying days + sufficient balance + consistency = eligible
17. `test_consistency_rule` — Best day ≤ 50% of total profit required for payout
18. `test_payout_caps` — Payout 1=$1,500, payout 2=$2,000, etc.
19. `test_retirement_after_6_payouts` — Account retires after 6th payout
20. `test_minimum_balance_for_payout` — Balance must be ≥ $52,600 ($52,100 + $500 min payout)

### 6.2 `test_account_manager.py` (8 tests)

```
"""
Phase 4 — Account Manager Tests

Tests portfolio-level account management: grouping, aggregate stats,
state persistence, and daily lifecycle operations.
"""
```

1. `test_add_account_to_group` — Account added to correct group
2. `test_get_tradeable_accounts` — Returns only active, non-DLL-locked, no-position accounts
3. `test_portfolio_summary` — Aggregate stats (total invested, withdrawn, ROI) calculated correctly
4. `test_start_new_day_resets_all` — Resets DLL and daily P&L for all active accounts
5. `test_save_and_load_state` — Account state persists to DB and reloads correctly
6. `test_blown_account_excluded_from_trading` — Blown accounts not returned by get_tradeable
7. `test_retired_account_excluded` — Retired accounts not returned by get_tradeable
8. `test_all_accounts_includes_historical` — get_all_accounts includes blown and retired

### 6.3 `test_trade_executor.py` (11 tests)

```
"""
Phase 4 — Trade Executor Tests

Tests trade execution across multiple accounts, signal handling modes,
and the no-hedging constraint.

Business context: When the model predicts "tradeable_reversal" during
NY RTH, all eligible accounts enter simultaneously. The executor must
handle edge cases like some accounts being DLL-locked, conflicting
signal modes, and the absolute no-hedging rule.
"""
```

1. `test_executable_prediction_opens_positions` — Reversal during RTH opens positions on all eligible accounts
2. `test_non_executable_prediction_no_trades` — Non-executable prediction does nothing
3. `test_ignore_mode_skips_positioned_accounts` — In ignore mode, accounts with positions are skipped
4. `test_flip_mode_closes_then_opens` — In flip mode, existing positions closed before new entry
5. `test_no_hedging_enforced` — Cannot have long and short positions simultaneously
6. `test_close_all_positions` — Closes everything across all accounts
7. `test_close_single_account` — Close one account without affecting others
8. `test_manual_entry` — Manual buy/sell on a specific account works
9. `test_hard_flatten_closes_everything` — 3:55 PM CT flatten closes all, no exceptions
10. `test_dll_locked_excluded` — DLL-locked accounts don't receive new trades
11. `test_trade_recorded_in_db` — Every trade is written to the trades table

### 6.4 `test_position_monitor.py` (10 tests)

```
"""
Phase 4 — Position Monitor Tests

Tests real-time position monitoring against the live tick stream.
The position monitor enforces TP/SL/DLL/flatten rules on every tick.

Business context: Position monitoring runs on every single trade tick.
A missed TP means leaving money on the table. A missed SL means
exceeding risk limits. A missed flatten means violating Apex rules.
"""
```

1. `test_tp_hit_group_a` — Position closed at 15-point profit for Group A
2. `test_tp_hit_group_b` — Position closed at 30-point profit for Group B
3. `test_sl_hit` — Position closed at configurable stop loss
4. `test_unrealized_pnl_updates` — Unrealized P&L tracks with each tick
5. `test_trailing_dd_checked_on_tick` — Trailing drawdown evaluated on every tick
6. `test_account_blown_mid_trade` — Account blown during trade = force close at liquidation
7. `test_dll_breach_mid_trade` — DLL breach = close position, lock account
8. `test_multiple_accounts_independent` — Group A can hit TP while Group B stays open
9. `test_configurable_tp_sl` — Changing TP/SL settings takes effect immediately
10. `test_hard_flatten_timer` — Flatten fires at exactly 15:55 ET

---

## 7. Acceptance Criteria

Phase 4 is complete when:

1. **Apex rules are accurately simulated** — Trailing DD, tiers, DLL, payouts, consistency rule all match the PRD specification
2. **Auto-execution works** — Executable predictions open positions on all eligible accounts
3. **TP/SL enforced on every tick** — No missed exits
4. **Hard flatten is absolute** — 3:55 PM CT closes everything
5. **No hedging** — System prevents opposite-direction positions
6. **State persists** — Account state survives restart
7. **All tests pass** — Every test in Section 6 passes

---

## 8. Notes for Claude Code

- **NQ point value:** 1 NQ point = $20 per contract. $5 per tick (0.25 points). All P&L calculations must use this.
- **Decimal precision is critical.** Account balances, liquidation thresholds, and P&L must use Decimal to avoid rounding errors that could incorrectly trigger or miss a blow.
- **The trailing drawdown trails on unrealized balance.** Not just realized. If an open position pushes equity to $52,100, the safety net is reached even before the trade closes.
- **DLL includes unrealized losses.** If an account has $800 in realized losses today and a $300 unrealized loss, total daily loss is $1,100 — exceeds Level 1 DLL of $1,000.
- **Don't over-optimize position monitoring.** With 5 accounts and 1 position each, checking all on every tick is fine. No batching or sampling needed.
