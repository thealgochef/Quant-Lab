# Backtest Findings

Model: `dashboard_3feature_v1.cbm` (CatBoost, 3 features: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`)

---

## 1. Out-of-Sample Backtest (Current)

**Date range:** 2025-06-02 to 2025-11-13 (141 calendar days, ~115 trading days)
**Config:** 5 accounts, all Group A (15pt TP / 15pt SL), RTH-only zone spending

### Headline Results

| Metric | Value |
|--------|-------|
| Total signals | 506 |
| RTH signals | 195 |
| Executable (RTH + tradeable_reversal) | 57 |
| Total trades | 285 (57 signals x 5 accounts) |
| Win/Loss | 160W / 125L |
| Win rate | 56.1% |
| Total P&L | $15,000 (+750 pts) |
| P&L per account | +$3,000 |
| Max drawdown | $9,000 |
| Prediction accuracy | 67.4% (341/506) |
| Avg signals/day | 3.59 (0.40 executable) |
| Best day | 2025-07-10: +$3,000 |
| Worst day | 2025-08-05: -$3,000 |

### Exit Reason Breakdown

| Exit Reason | Count | % |
|-------------|-------|---|
| TP (take profit) | 160 | 56.1% |
| SL (stop loss) | 110 | 38.6% |
| Flatten (3:55 PM) | 15 | 5.3% |

### Monthly P&L Breakdown

| Month | Days | Signals | Exec | Trades | W/L | Win% | P&L |
|-------|------|---------|------|--------|-----|------|-----|
| 2025-06 | 25 | 84 | 5 | 25 | 20/5 | 80% | +$4,500 |
| 2025-07 | 26 | 95 | 15 | 75 | 40/35 | 53% | +$4,500 |
| 2025-08 | 26 | 96 | 13 | 65 | 25/40 | 38% | -$4,500 |
| 2025-09 | 26 | 95 | 13 | 65 | 45/20 | 69% | +$9,000 |
| 2025-10 | 27 | 102 | 11 | 55 | 30/25 | 55% | +$1,500 |
| 2025-11 | 11 | 34 | 0 | 0 | 0/0 | -- | $0 |

**Notable:** August was the only losing month (-$4,500). September was the strongest (+$9,000). November had zero executable signals in 11 trading days.

### Per-Account Final Equity

All 5 accounts identical (same signals, same TP/SL):

| Account | Balance | Profit | Tier | Status | Trades |
|---------|---------|--------|------|--------|--------|
| A1-A5 (each) | $53,000 | +$3,000 | 3 | active | 57 |

### Signal Distribution by Session

| Session | Signals | Predicted tradeable_reversal | Executable |
|---------|---------|------------------------------|-----------|
| ny_rth | 195 | 57 | 57 |
| london | 176 | ~58 | 0 (not traded) |
| pre_market | 133 | ~24 | 0 (not traded) |
| post_market | 2 | ~2 | 0 (not traded) |
| asia | 0 | 0 | 0 |

### Shadow Trade Analysis (Non-RTH)

Shadow positions track what would happen if non-RTH `tradeable_reversal` signals were traded (using level-price entry, not market price -- so results are optimistic):

| Metric | RTH (actual) | London (shadow) | All non-RTH (shadow) |
|--------|-------------|----------------|---------------------|
| Total signals | 57 | 58 | 82 |
| Win rate (15pt) | 56.1% | 60.3% (35W/23L) | 67.1% (55W/27L) |
| Win rate (30pt) | -- | 48.3% (28W/30L) | -- |
| Avg MFE | 30.1 pts | 20.9 pts | -- |
| Avg MAE | 9.6 pts | 18.9 pts | -- |

**Key finding:** London shadow trades have a 60% win rate at 15pt TP/SL, but use level-price entry (optimistic). Real entries would face the same 14-29pt slippage seen in RTH.

---

## 2. Changes Made for This Backtest

### Change 1: RTH-Only First-Touch Rule

Modified `TouchDetector` so non-RTH touches (London/Asia/pre-market) do NOT mark zones as "spent" for RTH:
- During non-RTH: first-touch-only still applies within that session
- When RTH opens: all zones are available regardless of overnight touches
- During RTH: first-touch-only applies normally (zone spent after first RTH touch)

**Impact:** RTH signal count went from 6 (old per-day rule, 31 days) to 195 (RTH-only rule, 141 days) -- from ~0.12/day to ~1.4/day.

### Change 2: Dropped Group B (30pt TP/SL)

Previous backtest proved Group B was unprofitable with realistic entry prices (-$235/account). Changed to 5 accounts, all Group A (15pt TP / 15pt SL).

### Change 3: Market Price Entry (from previous session)

Entries at market price when prediction fires, not level price from 5 minutes earlier.

---

## 3. Prior Findings (Preserved)

### Entry Price Bug (Fixed)

| Metric | Before (level entry) | After (market entry) |
|--------|---------------------|---------------------|
| Win rate | 100% (15/15) | 73.3% (11/15) |
| Total P&L | $6,300 | $2,230 |
| Group A P&L/acct | +$420 | +$900 |
| Group B P&L/acct | +$420 | -$235 |

### First-Touch Diagnostic (Pre-Change)

With the old per-day first-touch rule (31 days analyzed):
- 55% of zones consumed before RTH opens
- Only 6 RTH touches in 31 days (0.19/day)
- Both counterfactuals showed 50 RTH touches possible (+733%)

### Look-Ahead Bias Audit (All 7 Tests PASSED)

| Test | Description | Result |
|------|-------------|--------|
| 1 | Level computation uses only PAST data | PASS |
| 2 | Observation window uses only ticks WITHIN 5-min window | PASS |
| 3 | Model prediction uses only features from observation window | PASS |
| 4 | TP/SL resolution uses only ticks AFTER entry | PASS |
| 5 | Outcome tracker uses only ticks AFTER prediction | PASS |
| 6 | No future data in tick replay order | PASS |
| 7 | Price buffer doesn't leak future data | PASS |

---

## 4. Key Observations & Open Questions

### What's Working
- The system is net profitable over 5+ months ($15,000 across 5 accounts)
- 56.1% win rate with 1:1 R:R yields positive expectancy
- RTH-only zone spending massively increased signal count (0.12/day -> 0.40/day executable)
- No look-ahead bias confirmed by comprehensive audit

### Concerns
- **August drawdown:** 38% win rate, -$4,500. Monthly variance is high.
- **November zero signals:** No executable signals in 11 trading days. The model may not generalize to all market regimes.
- **Max drawdown $9,000:** With 5 accounts at $50k each ($250k total), this is 3.6% portfolio DD -- acceptable but worth monitoring.
- **56% win rate is thin:** With 1:1 R:R, the edge is only 6% over breakeven. Commission and slippage in live trading could erode this.
- **Flatten exits (5.3%):** 15 trades closed at flatten time. These are neither TP nor SL -- they represent unresolved trades that run out of time.

### Open Questions
1. **Should we trade London sessions?** Shadow analysis shows 60% win rate at 15pt. But entry slippage hasn't been tested.
2. **Is 56% win rate enough for live trading?** With commissions (~$4.50/rt/contract x 5 accounts), real win rate needs to be higher.
3. **Why did November produce zero executable signals?** Need to investigate whether this is regime-dependent or data-related.
4. **Asymmetric TP/SL:** Would 20pt TP / 10pt SL or similar improve results? The shadow analysis shows MFE often exceeds 15pt.

---

## 5. Data Files

| File | Description |
|------|-------------|
| `data/experiment/backtest_oos_results.csv` | OOS daily P&L (Jun-Nov 2025) |
| `data/experiment/backtest_results.csv` | Original 30-day backtest results |
| `data/experiment/first_touch_analysis.csv` | Per-day first-touch diagnostic |
| `scripts/run_backtest.py` | Main backtest script |
| `scripts/audit_lookahead.py` | Look-ahead bias audit |
| `scripts/first_touch_analysis.py` | First-touch rule diagnostic |
