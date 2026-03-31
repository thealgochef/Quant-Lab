# Phase 6 — Frontend Dashboard

**Scope:** TypeScript/React web application — chart, panels, analysis tab, account management tab, model management tab
**Prerequisite:** Phase 5 complete (API server with WebSocket + REST)
**Produces:** The complete trader-facing dashboard UI
**Does NOT include:** Backend changes (all backend complete in Phases 1-5)

---

## 1. What This Phase Builds

A full-screen web dashboard that connects to the Phase 5 API and displays everything the trader needs for live paper trading. Two main views: Trading View (chart + panels) and Analysis View (full-screen data). Plus Account Management and Model Management tabs.

---

## 2. Tech Stack

- **Framework:** React 18+ with TypeScript
- **Charting:** Lightweight Charts (TradingView open-source) or similar high-performance candlestick library
- **State management:** Zustand or React Context (minimal — most state comes from WebSocket)
- **Styling:** Tailwind CSS — dark theme
- **Build:** Vite
- **WebSocket:** Native browser WebSocket API
- **Data fetching:** fetch API for REST calls

**Location:** Project root as a sibling to `src/`:
```
Claude-Quant-Lab/
  src/alpha_lab/          # Python backend
  dashboard-ui/           # TypeScript frontend
    src/
    package.json
    tsconfig.json
    vite.config.ts
```

---

## 3. Relevant PRD Sections

- **Section 10** (Chart Specification) — Candlestick chart, timeframes, overlays, navigation
- **Section 11** (Dashboard Layout) — Trading View, Analysis View, Account Management, Model Management
- **Section 15** (Notifications) — On-screen visual notifications only

---

## 4. Trading View

### 4.1 Chart (65% of screen width, resizable)

**Candlestick chart:**
- Default timeframe: 1m
- Switchable: 1m, 3m, 5m, 10m, 15m, 30m, 1H, 4H, 1D
- Loads with 48 hours of historical data
- Scrollable to 14 days back
- 1-second refresh rate from WebSocket price updates
- Dark background, standard green/red candles

**Fixed scale (critical — no zoom):**
- Fixed Y-axis range per timeframe
- Fixed candle width per timeframe
- Scroll only: left/right through time, up/down through price
- Auto-scroll lock toggle (lock icon) — when locked, latest bar stays pinned right

**Overlays (togglable):**
- Level lines — horizontal lines at active levels, color-coded by type
- Prediction markers — colored dots at level touch points (green=reversal, orange=trap, red=blowthrough)
- Resolved outcome markers — ✓ or ✗ border added to prediction markers after outcome known
- Observation window shading — semi-transparent overlay during 5-minute windows
- EMA 13, EMA 48, EMA 200
- KAMA
- Session-anchored VWAP

**Overlay computation:** EMAs, KAMA, and VWAP are computed client-side from OHLCV data received via the API. Use the same parameters as the experiment:
- EMA: Standard exponential moving average (periods 13, 48, 200)
- KAMA: Kaufman Adaptive MA (default: period=10, fast=2, slow=30)
- VWAP: Anchored to session open, resets at each new session

### 4.2 Side Panels (adjacent to chart)

**Active Levels Panel:**
- List of active (untouched) level zones
- Each shows: level type label (PDH, PDL, Asia High, etc.), price, side (HIGH/LOW)
- Manual levels marked distinctly
- Touched/spent levels shown grayed out
- Manual level input: text field + "Add" button

**Observation Window Panel:**
- When active: countdown timer (e.g., "3:42 remaining"), progress bar
- Feature accumulation: trades count, current feature estimates updating in real time
- When idle: "Waiting for level touch"

**Most Recent Prediction Panel:**
- Prediction class with color coding
- Probability breakdown (e.g., "Reversal: 85%, Trap: 10%, BT: 5%")
- The 3 feature values
- Execution status ("Executed on 5 accounts" or "Display only — outside RTH")

**Open Positions Panel:**
- Per-account position cards:
  - Account label, group (A/B)
  - Direction (LONG/SHORT) with color
  - Entry price
  - Current unrealized P&L (updating live)
  - MFE / MAE
  - TP target and SL level shown

**Session Stats Panel:**
- Signals fired today
- Predictions breakdown (reversals, traps, blowthroughs)
- Win/loss count
- Session accuracy %

**Overlay Toggles:**
- On/off switches for each overlay

**Account Controls:**
- Close All button (red, prominent)
- Per-account close buttons
- Manual Buy / Sell buttons (for manual entry)

**Connection Status:**
- Green circle = connected
- Yellow = reconnecting
- Red = disconnected
- Status text below indicator

### 4.3 Resize Handle

Draggable divider between chart and panels. Chart default 65% width, minimum 30%. Panels fill remaining space.

---

## 5. Analysis View (Separate Tab)

Full-screen data view. No chart present.

### 5.1 Model Performance

- Accuracy over time (line chart: daily accuracy plotted over days/weeks)
- Precision by prediction class (bar chart)
- Precision by session (Asia, London, NY RTH)
- Precision by level type (PDH/PDL, session highs/lows)
- Confusion matrix visualization

### 5.2 Trade History Log

Sortable, filterable table:

| Column | Content |
|--------|---------|
| Timestamp | When prediction fired |
| Level Type | PDH, PDL, Asia High, Manual, etc. |
| Direction | LONG / SHORT |
| Prediction | Reversal / Trap / Blowthrough |
| Features | 3 feature values (expandable) |
| Entry Price | Level price |
| Exit Price | Per account |
| P&L | Per account (Group A and Group B columns) |
| Correct? | ✓ or ✗ |
| Session | Asia, London, NY RTH |

Filters: date range, prediction class, session, level type, direction, correct/incorrect

### 5.3 Equity Curves

- Per-account equity over time (line chart)
- Aggregate portfolio equity (sum across accounts)
- Overlaid on a single chart with legend

### 5.4 Session-Based Stats

- Performance segmented by session
- Win rate, average win, average loss per session
- Best/worst sessions

---

## 6. Account Management Tab

### 6.1 Summary Section (top)

| Metric | Value |
|--------|-------|
| Total Invested | Sum of all eval + activation costs |
| Total Withdrawn | Sum of all payouts |
| Net P&L | Withdrawn - Invested |
| ROI % | Net P&L / Total Invested |
| Active / Retired / Blown | Count of each |

### 6.2 Per-Account Cards (below)

Expandable cards for each account:

- Account label, group, status badge (Active/Blown/Retired)
- Eval cost, activation cost, total cost
- Current balance, profit
- Current tier, max contracts, DLL
- Trailing DD: liquidation threshold, safety net status, DD remaining
- Payout status: payout number (1-6), qualifying days, amount available
- 50% consistency rule indicator
- "Request Payout" button (if eligible)
- Trade history (expandable sub-table)
- Payout history

### 6.3 Add Account

Button + form: label, eval cost, activation cost, group selection (A or B).

---

## 7. Model Management Tab

- **Active Model:** Version label, training date, accuracy metrics, feature set
- **Upload New Model:** File upload (.cbm) + optional metrics JSON paste/upload
- **Confirmation Step:** After upload, show metrics comparison (new vs current). "Activate" button.
- **Model History:** Table of all versions with metrics, activation dates
- **Rollback:** Button on any historical version to reactivate it

---

## 8. Directory Structure

```
dashboard-ui/
  src/
    App.tsx
    main.tsx
    index.css                    # Tailwind imports
    types/                       # TypeScript type definitions
      api.ts                     # Types matching backend schemas
      chart.ts
      trading.ts
    hooks/
      useWebSocket.ts            # WebSocket connection + message handling
      useApi.ts                  # REST API helpers
      useChart.ts                # Chart data management
    stores/
      tradingStore.ts            # Zustand: positions, predictions, levels
      accountStore.ts            # Zustand: account state
      configStore.ts             # Zustand: settings, overlays
    components/
      layout/
        TradingView.tsx          # Chart + panels layout
        AnalysisView.tsx         # Full-screen analysis
        AccountManagement.tsx
        ModelManagement.tsx
        TabNavigation.tsx
      chart/
        CandlestickChart.tsx     # Main chart component
        ChartOverlays.tsx        # Level lines, prediction markers, EMAs
        TimeframeSelector.tsx
      panels/
        ActiveLevelsPanel.tsx
        ObservationPanel.tsx
        PredictionPanel.tsx
        PositionsPanel.tsx
        SessionStatsPanel.tsx
        OverlayToggles.tsx
        AccountControls.tsx
        ConnectionStatus.tsx
        ManualLevelInput.tsx
      analysis/
        PerformanceCharts.tsx
        TradeHistoryTable.tsx
        EquityCurves.tsx
        SessionStats.tsx
      accounts/
        AccountSummary.tsx
        AccountCard.tsx
        AddAccountForm.tsx
      models/
        ActiveModel.tsx
        ModelUpload.tsx
        ModelHistory.tsx
    utils/
      indicators.ts              # Client-side EMA, KAMA, VWAP computation
      formatters.ts              # Price formatting, time formatting
      constants.ts               # Timeframe configs, colors, etc.
  package.json
  tsconfig.json
  vite.config.ts
  tailwind.config.js
```

---

## 9. Test Specifications

Frontend testing is lighter than backend — focus on integration and critical rendering logic.

### 9.1 Component Tests (React Testing Library)

1. `test_chart_renders_candles` — Chart renders correct number of candles from OHLCV data
2. `test_level_lines_render` — Active levels appear as horizontal lines on chart
3. `test_prediction_markers_render` — Prediction markers appear at correct positions
4. `test_overlay_toggles` — Toggling an overlay on/off adds/removes it from chart
5. `test_position_panel_shows_open` — Open positions display with correct P&L
6. `test_observation_countdown` — Countdown timer decrements correctly
7. `test_connection_status_indicator` — Status indicator reflects current connection state
8. `test_trade_history_filtering` — Filters in trade history table work correctly

### 9.2 WebSocket Integration Tests

1. `test_backfill_populates_state` — Backfill message populates all stores
2. `test_price_updates_chart` — Price updates from WebSocket appear on chart
3. `test_prediction_updates_panel` — New prediction updates prediction panel
4. `test_reconnect_backfill` — After reconnect, state refreshes correctly

### 9.3 Indicator Computation Tests

1. `test_ema_calculation` — EMA values match expected output
2. `test_vwap_calculation` — VWAP anchors to session open and resets
3. `test_kama_calculation` — KAMA responds to volatility changes

---

## 10. Visual Design Guidelines

**Theme:** Dark. Charcoal backgrounds (#1a1a2e or similar), light text, minimal borders. Inspired by professional trading platforms (Quantower, TradingView Pro).

**Color palette:**
- Background: #0f0f1a (darkest), #1a1a2e (panels), #25253d (hover/active)
- Text: #e0e0e0 (primary), #8888aa (secondary)
- Green (bullish/up/buy): #00c853
- Red (bearish/down/sell): #ff1744
- Orange (trap prediction): #ff9100
- Blue (info/neutral): #2979ff
- Yellow (warning/reconnecting): #ffd600

**Typography:** Monospace for prices and numbers (JetBrains Mono or similar). Sans-serif for labels (Inter or system font).

**Spacing:** Dense but readable. This is a trading dashboard, not a marketing site — information density matters more than whitespace.

**Animations:** Minimal. Price updates should feel immediate, not animated. The only animation should be the observation window countdown.

---

## 11. Acceptance Criteria

Phase 6 is complete when:

1. **Chart renders correctly** — Candlesticks display with correct OHLCV, overlays toggle, scrolling works
2. **Real-time updates work** — Price, bars, predictions, trades update via WebSocket within 1 second
3. **All panels functional** — Levels, observations, predictions, positions, stats, controls all work
4. **Analysis tab populated** — Performance charts, trade history, equity curves render with real data
5. **Account management works** — Add, view, payout accounts through the UI
6. **Model management works** — Upload, activate, rollback models through the UI
7. **No zoom on chart** — Fixed scale per timeframe, scroll only
8. **27" 1440p optimized** — Layout fills the screen without scroll or overflow at target resolution

---

## 12. Notes for Claude Code

- **Lightweight Charts is recommended** for the candlestick chart. It's TradingView's open-source library, handles large datasets well, and supports custom overlays. Alternative: uPlot for performance.
- **Do not fetch historical data on every re-render.** Load 48 hours on initial connect (from backfill), then update incrementally from WebSocket bar_update messages.
- **Client-side indicator computation** keeps the API simple. EMAs and VWAP are trivial to compute from OHLCV bars in TypeScript. Don't push these from the backend.
- **Fixed chart scale is non-negotiable.** The trader explicitly requested no zoom. Scrolling only. This is a deliberate design choice for consistent visual pattern recognition.
- **The "no zoom" constraint means pre-defined Y-axis ranges per timeframe.** For 1m chart, you might show 100 points of price range. For 1H, maybe 300 points. These can be configurable in constants but are not interactive.
- **Mobile/responsive is not required.** This runs on a 27" 1440p monitor in a browser. Design for that specific viewport.
- **Keyboard shortcuts are a nice-to-have** but not required for v1. If time permits: Escape = close all, 1-9 = switch timeframes.
