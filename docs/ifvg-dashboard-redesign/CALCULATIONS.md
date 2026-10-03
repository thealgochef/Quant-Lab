# Calculations

> **Superseded in part — September 25, 2026 analytical corrections.** This file keeps its
> original text. Where it differs, the maintained definitions are
> `docs/ifvg-redesign-fixes/followup-1/CALCULATION_DEFINITIONS.md` since follow-up 1 (September 25,
> evening); section 17 there lists every replaced statement. The closeout's copy,
> `docs/ifvg-redesign-fixes/closeout/CALCULATION_DEFINITIONS.md`, is the delivered version. Replaced
> here: the deflated-Sharpe and
> "chance" wording; the Edge verdict's "Not shown if both fail" (now "Not supported"); the
> Early-losses, Held-half and market-direction findings; the buy-and-hold benchmark (now one
> entry instant, version `buy_and_hold_one_emini_first_open_v2`); the fixed-floor "payout race"
> (now a fixed closed-profit boundary diagnostic); the full version (now conditional resampling
> of recorded trades with the firm's ledger rules, with named populations and clocks);
> "reordering" wording (the resampling draws with replacement); the drop-from-a-high chart (a
> sampled closed-profit drawdown, not account failure); the market labels (now retrospective v1
> plus labels known at entry v1); "the owner is weighing 18–20" (a neutral pending
> decision; the saved 3-day limit is unchanged); and, in follow-up 1, the Trade review zones'
> tap and close-through times (known when their candle closes: "tapped April 12, 5:18 PM"
> names the candle, and the tap was known at 5:19 PM).

Every number on the redesigned screens is defined here. Build these once in a shared calculation module, test them against the reference values, and use them everywhere.

**Reference case** for every test below: funded variation study `5fa65149843484b1` (export v4), configuration `S1-T1-H14-P1-L-SO` ("All open-market hours · Long only · Half at 1R"), firm TakeProfitTrader, research dates January 13 – June 10, 2026 (107 trading days). "Exact" means match to the cent or the shown decimals. Resampled values depend on the random draw; tolerances are given.

## Inputs and shared series

- **Funded trades:** the study's funded trade records for one configuration and one firm, in execution order, across all of that configuration's accounts. Reference: 114 trades.
- **Daily results:** the sum of funded trade net results per trading day, for every trading day in the study calendar, with 0 on days without a trade. Reference: 107 days.
- **Trade path:** cumulative net result by trade, starting at $0 before the first trade.
- **Strategy measures:** the study's stored no-account replay measures (Net R, win rate, drawdown in R, profit factor, days under water). Never recompute these from funded trades.

## Funded results table and headline tiles

| Figure | Definition | Reference |
|---|---|---|
| Net cash | Cash received after the firm's split minus every account purchase | $30,781.88 exact |
| Received after the split | Sum of payouts received | $31,393.88, 13 payouts exact |
| Account costs | Accounts purchased × firm price | $612.00, 6 accounts exact |
| Cash per $1 of accounts | Net cash ÷ account costs | $50.30 |
| Largest · median payout | Of payouts received | $5,610.66 · $2,217.33 |
| Worst drawdown | Largest fall of the trade path from its running high | $3,661 |
| Sharpe | mean(daily) ÷ standard deviation(daily, sample) × √252 | 4.01 |
| Sortino | mean(daily) ÷ √(mean(min(daily, 0)²)) × √252, over all days | 18.82 |
| Net R · win rate | Strategy measures | 73.45 · 58.4% |

Other leaders for checking the ranking (TakeProfitTrader rank 2–4 Sharpe · Sortino · worst drawdown): 3.51 · 13.85 · $6,268; 2.98 · 9.79 · $7,379; 4.00 · 16.69 · $2,659. MyFundedFutures rank 1: $34,818.11, 4.05 · 19.84 · $3,461.

## Checks on the leader

- **Drop the largest payout.** For every configuration at the firm, subtract its largest single payout from net cash and re-rank. Pass if the leader stays rank 1. Reference: pass at both firms (TakeProfitTrader $25,171.22 vs next $21,718.32; MyFundedFutures $28,506.12 vs next $24,870.65).
- **Result per trade range.** Resample the funded trades' net results with replacement, same count, 20,000 times; take the mean of each resample; report the 16th–84th, 5th–95th and 2.5th–97.5th percentiles as the 68%, 90% and 95% ranges. Reference: mean $330.97; 95% $95 to $601; 90% $126 to $551; 68% $201 to $458 (±5% tolerance).
- **Chance the true Sharpe is above zero.** Use the daily (not annualized) Sharpe SR, sample skew g3, and kurtosis g4 (not excess) of the daily results over T days: Φ(SR × √(T−1) ÷ √(1 − g3·SR + (g4−1)/4·SR²)). Reference: skew 3.16, kurtosis 13.81, result above 0.99.
- **Deflated Sharpe.** The same formula with SR replaced by SR − SR0, where SR0 = √V × ((1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))), γ = 0.5772, N = configurations compared in this study at this firm, V = variance of their daily Sharpe ratios. Show N on screen. Reference: N = 64, SR0 annualized 3.15, deflated Sharpe 0.81.
- **Quality gates.** Thresholds come from the study's saved gate settings. Evaluate on the strategy replay (no accounts). Reference: trades 154 (60+ pass), result per trade 0.48R (0R+ pass), profit factor 2.13 (1.1+ pass), drawdown 10.57R (15R max pass), best day's share of profit 14% (40% max pass), trading days under water 40 (fails the saved 3-day limit; the owner is weighing 18–20). If the strategy replay doesn't store a gate's input, show "Not in export". The mock's "days with a trade: 63" was taken from funded trades as a stand-in; use the strategy replay's count when stored.

## Summary: verdict and findings

Default thresholds; log them and let the owner change them later.

| Verdict part | Rule |
|---|---|
| Data integrity | Pass if every position minute used recorded trades or a labeled approximation, and money reconciles. Otherwise Fail. |
| Edge | Holds if the 95% range's low end is above $0 and deflated Sharpe ≥ 0.5. Weak if one of the two fails. Not shown if both fail. |
| Account risk | Watch if worst drawdown exceeds the firm's loss limit or an account was lost before its first payout. Otherwise OK. |
| Sample | Limited if the study covers fewer than 250 trading days or has no unseen-window result. Otherwise Adequate. |

| Finding | Severity | Fires when |
|---|---|---|
| Concentrated result | High | Largest account ≥ 75% of cash received, or five largest trades ≥ 60% of trading profit |
| Early account losses | Medium | Payout race died-first share ≥ 15% |
| Held portion carries the profit | Medium | Trades closed at 3:55 PM made ≥ 100% of total trading profit |
| Little tied to market direction | Info | Beta R² below 0.2 |

Each finding shows one sentence of what happened and a one-line next step. Reference: all four fire (92%, 74%; 21.6%; $44,769 from 13 held trades; R² 0.11).

## Concentration

Largest account's share of cash received (91.6%); five largest trades' share of total trading profit (74.3%); best month's share of net cash (April, 55.6%); largest payout's share of cash received (17.9%); best day's result ÷ sum of all winning days (14.1%). Bars turn orange at 50% and above.

## Tie to the index

- **Beta:** least-squares slope of daily results on the daily E-mini change × $20 over the same days. Reference: 0.067, R² 0.112.
- **Buy and hold one E-mini:** (last trading day's close − first trading day's close) × $20. Reference: +$51,475. Also show the first time its fall from a running high reached $2,000, using one-minute closes. Reference: January 13, 2026, 9:02 AM.

## Payouts and accounts

Existing figures, displayed as stored: payout timing and spending, where the account time went and refused entries, payouts by month with running net, cumulative payouts/costs/net by day, account replacement history, and one account's balance against its loss limit by trade. Reference: Account 1 finished −$663.90 against a limit of −$661.33 on a trade that netted +$477.22; time in trade 268.0 h, payout processing 864.0 h, entries refused while processing 42.

## Trades tab

- **Result distribution:** funded trade net results in bins: under −$750, then $250 steps to $1,000, then $1k–2k, $2k–4k, over $4k. Reference counts: 5, 17, 15, 10, 32, 15, 6, 1, 3, 6, 4.
- **Excursions:** worst point = balance before − lowest equity during the trade; best point = highest equity during the trade − balance before (from the funded trade record's equity fields). Reference: 35 of 47 losers were never $250 in profit; 46 of 67 winners were never $250 against.
- **Performance summary:** gross profit $59,978.74; gross loss −$22,248.16; average win $895.21; average loss −$473.37; largest $7,349.72 / −$950.28; longest streaks 6 / 6; 12 drawdowns averaging 7.8 trades and $1,207; longest 45 trades. Short column shows "—" for long-only configurations.

## Risk and simulation

All resampling uses the funded trades' net results. "Keep streaks together" draws blocks of 10 consecutive trades at random start positions and joins them; "shuffle every trade" draws single trades with replacement. Default 20,000 paths, fixed seed.

- **Payout race (fixed floor version).** Each path starts at $0 and runs up to 200 trades until it reaches the withdrawal trigger (owner input, default +$2,600 = $2,100 cushion + $500) or the loss limit (default −$2,000). Record which and at what trade. Chart the cumulative share paid and died at each trade 0–40. Reference: 78% paid first, 22% died first, typical 8 trades to a payout and 10 to the limit (±2 points, ±1 trade).
- **Payout race — full version.** Same resampled trade orders, run through the existing funded account simulator for the selected firm so its trailing floor, lock, payout protection and processing rules apply. Adds "expected payouts before an account dies" and "expected cash per account bought". See TASK.md section 4.
- **Resampled equity.** For each method: percentiles 5, 25, 50, 75 and 95 of the trade path at every trade up to the actual trade count, ten sample paths, and the actual path. Reference (keep streaks · shuffle): end 5th $14,245 · $15,058; median $38,668 · $36,958; 95th $66,189 · $62,916; bad-case worst drawdown $6,965 · $6,854; finished below $0 0.2% · 0.2% (±5%).
- **Where the trades end up:** distribution of end values in $5,000 bins with the 5th, median, 95th and actual (+$37,731) marked.
- **Losing streaks:** longest run of losing or break-even trades in each path. Typical = median, bad case = 95th percentile. Reference: 6 and 8; actual 6.
- **Drop from a high by trade count:** for trades 0–100, the median, 75th and 95th percentile of each path's worst drop so far, and the share of paths whose drop has reached $2,000. Reference at 10 / 20 / 40 / 100 trades: typical $1,111 / $1,547 / $2,487 / $3,474; reached $2,000 14% / 37% / 65% / 94% (±2 points).

## Market conditions

- **Daily close:** the last one-minute close of each trading day from stored E-mini bars.
- **Labels:** trend is Rising if today's close is above the close 10 trading days earlier, else Falling. Volatility is the sample standard deviation of the last 10 daily percentage changes; Volatile if above the median of that measure across the study's trading days, else Quiet. Days without 10 earlier days are "Not enough history". Stored bars start January 1, 2026, so labels begin January 16.
- **Cards:** per label — trading days, trades, share of trades, net result, win rate, average trade. Trades are labeled by entry day by default; the switch uses exit day.
- **Transition table:** counts and row percentages of consecutive labeled days.
- **Entry conditions:** the default measure is the average true range of the 20 one-minute bars before entry divided by price; correlation with the trade's result in multiples of initial risk. Show "No clear link" when the correlation's size is under 0.1 or not significant. The other switch options (trend, volume, time of day) follow the same pattern; log their exact definitions.

Reference day counts: Rising · quiet 39, Falling · volatile 34, Rising · volatile 18, Falling · quiet 13, not enough history 3. Trades and net: Rising · quiet 41, +$13,546; Rising · volatile 14, +$15,844; Falling · quiet 15, +$56; Falling · volatile 39, +$8,949; not enough history 5, −$664. Stretches: March 18 – April 7 (15 days, falling · volatile) +$10,537. Volatility correlation 0.04.

## Trade review

- **Candles** from stored E-mini bars, aggregated to the chosen size, for the trade's full trading day window.
- **Zones** from the saved setup record linked by `strategy_trade_id`. Reference (April 12, 7:07 PM entry): four-hour gap 24,451.50–25,086.75 confirmed April 8, 1:00 AM; tapped April 12, 5:18 PM; five-minute parent gap 24,937.25–24,966.50 confirmed 7:00 PM; opposing one-minute gap 24,961.75–24,962.50 confirmed 7:03 PM; closed through by the 7:05 PM candle; entry 24,971.00 at 7:07 PM; stop 24,949.75; half out at 24,992.25 at 7:10 PM; last half out at 25,576.25 at 3:55 PM on April 13; +$6,254.72.
- **Point in time** hides every candle, marking and value after the chosen moment, including the result.
