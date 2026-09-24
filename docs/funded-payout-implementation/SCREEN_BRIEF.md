# Completed-results screen

Updated September 23, 2026 for the single-account configuration comparison. The previous brief
(budgeted five-account mode) is preserved under `history/2026-09-23T053828-CDT/`. This brief
applies the owner's presentation requirements; it does not change financial rules.

## Main decision

**Which tested strategy configuration produced the most cash received after the cost of every
funded account used, over the same period?** Show spending, payout timing and losses so the
owner can judge the tradeoff. This is one historical path, not a future success probability.

## Comparison view (first thing on the screen)

- The question, the actual period, the firms, the position size and a visible
  simulated-result and price-evidence status.
- One table of **all** tested configurations, with **TakeProfitTrader** and **MyFundedFutures**
  as separate tabs or clearly separated columns. Never add configurations or firms together
  into a portfolio profit.
  The application shows one firm at a time through a single firm selector that drives both
  the ranking and the detail (SPEC.md A8, A12); a tab choice would not reach the detail.
- Per row: configuration in plain English, payouts received after the split, total account
  costs, net cash earned, largest received payout, accounts purchased, accounts lost before and
  after receiving a payout.
- Unfavorable, zero-payout and zero-trade configurations stay in the table. A failed,
  incomplete or unsupported pair is shown as such, never as a zero.

## Detail view (one selected configuration and firm)

- Strategy settings needed to understand the result, in plain English with full chart
  timeframe names (for example "one-minute chart", "one-hour chart").
- First payout time, payouts by month (including zero months), largest unrecovered account
  spending (maximum shortfall before receipts covered costs), money pending at the cutoff.
- Cumulative receipts, costs and net cash over time.
- Account replacement history: each account's creation, payouts, pauses, failure and the
  replacement that followed.
- Time without trading split into: payout protection, payout processing, trading-hour limits,
  no strategy signal, and account replacement. A payout wait is not a strategy drought.
- Selecting an account shows its balance against the loss limit and payout history; selecting
  a trade shows quantity, Chicago entry/exit times, prices, dollars, initial risk, exit reason
  and whether its prices are recorded prints or a minute-candle approximation.

## Not in this mode

No credit counts, unused credits, growth thresholds, next-five purchase price, capacity or
vacancy widgets. Those belong only to the separate budgeted-mode view that keeps rendering the
September 22 pilot.

## Design boundaries

Keep the established application shell and installed chart components. Plain English, readable
dollars, Chicago dates and 12-hour AM/PM times. No internal hashes, file paths, serialized
settings, schema labels, stack traces or developer counters in the business view or tooltips.
Avoid decorative gradients, oversized empty cards, marketing claims, redundant charts and raw
developer tables. Keyboard access, legible contrast and usable empty/failure states. Material
limitations stay visible.

All shown numbers and the automatic export come from the same immutable verified result. Test
real screen selection, save/reload and navigation, and inspect an actual rendered screenshot.
Sample fixtures are labeled; an empty or sample screen is not evidence that a historical
simulation completed.
