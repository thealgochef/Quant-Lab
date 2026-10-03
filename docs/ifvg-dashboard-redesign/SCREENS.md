# Screens

> **Note — September 25, 2026 analytical corrections.** Screen names and wording that
> describe the fixed-floor "payout race", its "full version" and "reordering" the trades are
> superseded: see `docs/ifvg-redesign-fixes/followup-1/CALCULATION_DEFINITIONS.md` (sections 11–14,
> 16 and 17; maintained since follow-up 1, the closeout's copy is the delivered version). The
> layout described here is unchanged. Follow-up 1 also dates Trade review's tap and
> close-through by their candle's close.

One section per screen, in build order. Each panel lists what it shows, where the data comes from, and what existing feature to reuse. "Calc" points to CALCULATIONS.md. Mock images are in `mocks/images/`.

Navigation on every screen: a dark left rail with My studies, Trade review and New study, and the note "All times Chicago, 12-hour" at the bottom.

## 02 Funded results — `02_funded_results.png`, `02b_funded_results_state_myfundedfutures.png`

| Panel | Shows | Source | Reuse |
|---|---|---|---|
| Header | Study name, question, dates, configuration count, firms, comparison mode | Study record | Existing study header |
| Status line | One line: price evidence coverage, money reconciled, micro pricing note, corrections count, approval date; "Details" expands to the existing verification panels | Study validation records | Existing notices, collapsed into one line |
| Selection window | Dates, trading days, leader's net cash, payouts, accounts at the selected firm | Calc: funded results | — |
| Unseen window | June 11, 2026 onward, "Not yet run". The "Plan a confirmation run" button is shown disabled with "Confirmation runs aren't available yet" | None in this task | — |
| Firm switch | TakeProfitTrader / MyFundedFutures; drives this table and every detail view | Shared firm state (repair R2) | — |
| Ranking table | Rank, configuration in two readable lines, net cash, payouts, accounts used, cash per $1, Net R, win rate, worst drawdown, Sharpe, Sortino; leader row tinted; 8 rows then "Show all" | Calc: funded results table | Existing cash ranking and "strategy measures without accounts" tables, merged |
| Checks on the leader | Drop largest payout, result-per-trade range, deflated Sharpe with N, quality gates count | Calc: checks | 95% interval from search results; gates from study settings |
| Findings link | Count badge opening the detail Summary | Calc: findings | — |

## 03–08 Configuration detail (six tabs)

Shared header on every tab: breadcrumb, configuration name, one-line settings summary, tab bar (Summary, Payouts and accounts, Risk and simulation, Trades, Market conditions, Settings and evidence), firm switch on the right.

### 03 Summary — `03_detail_summary.png`, `03b_detail_summary_state_68_percent_range.png`
Headline tiles (net cash, received with payout count, account costs with account count, largest payout, median payout); four-part verdict; findings with severity, one-sentence finding and next step; key measures (average result per trade with a working 68/90/95% switch, Sharpe with Sortino, chance above zero with deflated value, profit factor funded and without accounts, skew and tail weight, beta with R², buy and hold with its first $2,000 breach); concentration bars; quality gates table. Calc: all sections. Reuse the existing four-part verdict layout from the model studies.

### 04 Payouts and accounts — `04_detail_payouts_and_accounts.png`
All existing funded detail panels, kept: payout timing and spending; where the account time went with a stacked bar and refused entries; payouts by month as bars plus the full month table with running net; cumulative payouts, account costs and net cash by day; account replacement history table with why each account was lost; "Show one account" picker with that account's balance against its loss limit by trade and a two-sentence explanation of how it was lost.

### 05 Risk and simulation — `05_detail_risk_and_simulation.png`, `05b_detail_risk_state_shuffle_every_trade.png`
Path count picker and "Run again"; one-line explanation that this reorders the configuration's own trades. Payout race with loss-limit and trigger inputs, three headline numbers, and a stacked chart of paid, still going and died over 40 trades with callouts at 10 and 20; two placeholders for the full-version figures until P5's second step is built. Resampled equity fan with the method switch, sample paths, actual path, labeled end values, and a comparison table for both methods plus what happened. End-value distribution and losing-streak distribution side by side. Drop-from-a-high growth: percentile lines against the $2,000 line, share that touched the limit, and the 10/20/40/100 table. Calc: Risk and simulation. Reuse the block bootstrap from the context studies if its definition matches.

### 06 Trades — `06_detail_trades.png`
Result distribution; performance summary with All, Long and Short columns; two excursion scatters (worst point, best point, against final result) with the counts sentence; trade list with account, entry, exit, entry price, stop, both fills of a partial exit, why it closed, and result. The row where an account's loss limit ended a winning trade is tinted. Each row opens that trade in Trade review.

### 07 Market conditions — `07_detail_market_conditions.png`
Entry day / exit day switch; definition sentence; four condition cards plus the not-enough-history line; profit chart with background shaded by condition and a caption naming the longest stretch and its result; transition table; entry-condition scatter with measure switch and a link-strength badge. Reuse the market regime tab from the feature-and-model studies if its labels can match CALCULATIONS.md; otherwise follow CALCULATIONS.md and log the difference.

### 08 Settings and evidence — `08_detail_settings_and_evidence.png`
Existing panels, kept: configuration settings with signal, mark and fill sources; verification checklist; reporting corrections; what this result can't tell you (the study's stated limitations); owner decisions and assumptions with every decision shown in full, never truncated; download of the study's review folder if that export already exists.

## 09 Trade review — `09_trade_review.png`, `09b_trade_review_state_point_in_time.png`
Source switch (Funded trades, Strategy trades, Verified context, Setups not taken — the last three are the existing sources); study, configuration, firm, account and trade pickers with previous/next; options bar with Full history / Point in time, overlay checkboxes and candle size. Whole-trade chart: candles for the trade's trading day window, four-hour gap zone, in-trade shading, entry marker, stop moved to entry, midnight line, 3:55 PM exit with the result box. Setup chart: one-minute candles around entry with numbered markers 1–5 and a key (parent gap, opposing gap, candle that closed through it, entry with stop, half out). Side panels: what was recorded; how the setup formed with times; price evidence note. Review form: five overall verdicts, four step-by-step judgments, tags with "+ New tag", notes, reviewer, "Save and next trade" and "Save review". Calc: Trade review. Reuse the existing reviewer storage, keyed per firm, account and trade (repair R3).

## 01 My studies — `01_my_studies.png`
Title and "New study"; tabs Funded comparisons, Strategy studies, Model studies, Drafts, Archived with counts; search and sort; funded table with study, what it tested, dates, status, leader net cash per firm, "Open results"; older-method studies tagged "Earlier method" with values greyed; drafts table with a status note (for example the blocked-engine message) and "Continue draft". Only two action labels exist: Open results and Continue draft. Source: both apps' study stores.

## 10 New funded comparison — `10_new_funded_comparison_setup.png`, `10b_setup_state_legacy_baseline_warning.png`
Two-step indicator. Starting configuration picker: S0_D80_W1_P1 named baseline by default; choosing the legacy baseline shows the warning (repair R7). Dates: start and end, resolved trading-day count with warmup, "See every date", June 11 protection line (repair R8). Settings to compare as readable chips with remove and "+ Add": entry hours, profit target, gap rule (new), exit, withdrawal trigger (new), plus fixed settings. Firms, size and costs. Right panel: the plan's multiplication, configuration count, results count, engine note, "Continue to review", "Save draft".

## 11 Review and approve — `11_review_and_approve.png`, `11b_review_state_blocked_engine_and_bad_threshold.png`
Summary tiles; blocked-engine alert when the saved plan needs a capability the current runtime lacks (repair R1), with approval disabled; "What changes between configurations" table first, then "The same for every configuration"; pass/fail checks as editable values with validation messages, including an impossible threshold kept as saved but flagged (repair R5) and the under-water limit marked as needing the owner's decision; approval checkbox and "Record approval and run" using the existing gated approval path.

## Known data gaps

These appear in the mocks but aren't in current stored records. Handle each per DECISION_RULES.md rules 5–7 and list the outcome in `handoff/DATA_GAPS.md`.

1. Setup zones for funded trades — link through `strategy_trade_id` (TASK.md section 4).
2. Unseen-window results — placeholder until a confirmation run exists; out of scope.
3. Payout race full version — needs the funded simulator on resampled orders (TASK.md section 4).
4. Earliest stored date and resolved day count — from repair R8.
5. Session stability, time-block consistency, best setup share — "Not in export" unless the strategy replay stores them.
6. Withdrawal-trigger and gap-rule settings — controls and draft storage only; approval blocked until the engine or simulator supports them.
7. Confirmation-run planning — disabled button; out of scope.
