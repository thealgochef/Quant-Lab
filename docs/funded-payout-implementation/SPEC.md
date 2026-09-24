# Funded Payout Lab — specification

Updated September 23, 2026. **Part A is the current research contract** (single-account
configuration comparison, from the owner's scope update in `AGENT_SCOPE_UPDATE.md`).
**Part B is the preserved Version 3 specification of September 22, 2026.** Part B now governs
only the separate, deferred budgeted operating mode (five-account groups, monthly credits,
growth) and describes the completed September 22 pilot. Where Part B's budget, allocation or
copied-account rules conflict with Part A, Part A wins for this mode. Part B's money rules,
loss mechanics, payout policy, clocks, price-evidence principles and export discipline remain
in force for both modes unless Part A changes them explicitly. The byte-exact prior version is
in `history/2026-09-23T053828-CDT/`.

---

# Part A — Single-account configuration comparison (current)

Status: requirements plus the implemented design, recorded September 23, 2026. Whether a
given step is implemented, tested or run is tracked only in TASKS.md.

## A1. Question

**Which tested strategy configuration produces the most simulated cash received after the
cost of every funded account used, over the same selected historical period?** Large payouts
count even if the account later fails. Trade count, inactivity, win rate, survival and
frequent minimum withdrawals are not substitutes. Spending, payout timing and losses are shown
so the owner can judge the tradeoff. One historical path is not a future success probability.

## A2. One configuration, one firm, one live account

- Each configuration resolved in the normal study configurator, for each selected firm, is one
  separate comparison (a *pair*). N configurations x 2 firms = 2N pairs; any N is supported.
- Each pair holds **at most one live funded account**. The first is bought at the evaluation
  start (first session open) and charged $102 (TakeProfitTrader) or $125 (MyFundedFutures).
- On failure the account's liquidation and costs are finalized, the failure is recorded, and a
  fresh funded replacement under the same configuration is bought **at the failure instant**,
  charged once, with **no credit limit** and no requirement to fund it from payouts. No
  evaluation or purchase delay is modeled (research simplification). No purchase at or after
  the cutoff.
- A replacement starts with $0 earned profit, a -$2,000 floor and no payout history. It
  inherits no position, consumed entry, cushion or pending claim. It trades only from a later
  strategy opportunity (A4).
- **A payout-processing account is alive.** It is never replaced or supplemented, and it trades
  nothing until processing completes. The pair keeps being measured during the wait.
- Nothing is rewound or reset: costs and receipts accumulate over the whole period.
- Not in this mode: five-account groups, monthly credits, carryover, vacancies, wallets,
  reinvestment, 25% growth blocks, the twenty-account ceiling, copied accounts, rotation or
  mixed allocation.

## A3. Unchanged financial rules (from Part B and the owner decisions)

$2,000 loss allowance; TakeProfitTrader floor rises with the intraday peak of realized plus
open equity, locks at $0, fails at or below; MyFundedFutures floor moves only from the
scheduled session-close realized balance, locks at +$100, fails at or below before the lock
(approved assumption) and strictly below after it; 80% / 90% trader share; six / three mini
limits (a size is refused, never clipped); $5.14 per mini per fill, posted inside the account
at each fill and never deducted again; mandatory daily and weekend flat. Withdrawal policy:
eligibility from realized profit after costs while alive and flat; once at least $500 gross is
withdrawable above the retained $2,100, no more entries that day; the full surplus is
requested at that day's end-of-day window (MyFundedFutures floor update first); two business
days of processing, paid at 4:00 PM Chicago on the second eligible processing day (US Federal
Reserve holidays); resume only after processing and when market/session rules allow. Received
payouts stay received after a failure; claims pending at the cutoff are not received.

## A4. Account-driven strategy state (implemented design)

Each pair runs its **own** Strategy-Core replay (pinned, unmodified Core) through
`propsim/funded/strategy_driver.py`, stepping one completed one-minute candle at a time over
the study's trusted cached day artifacts (10 warmup + 107 evaluation dates, same read policy
as the approved study). Per candle: timed account events due by the candle's open; the open
position is walked through the candle's ordered observations; the account's refusal reason
(payout protection / processing) is appended to Core's own execution-time admission check;
Core steps the candle.

- **Refused entry:** Core records a blocked candidate; the setup is then **discarded**, so it
  can never fill later from a stale signal. New setups may form afterwards.
- **Account liquidation before the strategy's own exit:** the reducer's position slot is
  cleared after that candle, so Core never continues the dead trade to its old stop or target
  and may form new setups from the next candle (never the liquidation candle itself).
- **Warmup** runs the strategy with no account; its trades are not booked (same as the study).
- Detectors, registries, swings and gap validity continue as Core's own state; nothing resets
  when an account changes. Immutable inputs (candles, levels, prints) are shared between pairs;
  reducers, positions and money never are.
- **Equivalence proof:** each configuration is also replayed with no account; that run must
  reproduce the saved study's trades exactly. Identical trades that different configurations
  produce independently are allowed; no diversity is manufactured.
- Any disagreement between the account and the strategy about an open position stops that
  configuration with an explicit error (it is then shown as not completed).

## A5. Execution model `ordered_prints_stop_market_v2` (versioned)

Entry at the confirming one-minute candle's close (as Core records it). **Stops are
stop-market orders filled at the first recorded trade at or through the stop** (a gap fills
worse, never better). Targets are limit orders filled at the target. The loss limit is checked
on every recorded trade against the floor active at that trade; a breach closes at that trade's
price; on one observation the breach is resolved before the stop, then the target. A position
open at the end of the daily deadline minute closes at that minute's last trade. The exit cost
can itself lose the account.

Price evidence: for every candle a position is open, the front-month MBP-1 trade prints inside
the candle's window are used only if they rebuild Core's candle exactly (open, high, low,
close, print count); otherwise that minute is a labeled one-minute approximation (losing side
first, continuous legs). Counts of exact and approximated minutes, and the stops that filled
worse than the stop price, are reported per pair. The September 22 pilot used
`recorded_strategy_prices_v1` (stops at the recorded price, shared signal); it is kept
unchanged as history and is not overwritten or adjusted.

## A6. Accounting and results

`net cash earned = after-split payouts received - initial and replacement account costs
- other included external costs (none modeled)`. Also per pair: total account spending,
largest cumulative unrecovered spending (maximum shortfall before receipts covered costs),
first payout time, monthly receipts/costs including zero months, pending money at the cutoff,
accounts purchased, accounts lost before and after a payout, refused entries, and time split
into trading, payout protection, payout processing, market closed or locked, and no strategy
signal. Every configuration stays in the comparison, including unfavorable and zero-payout
ones. A configuration that could not complete is shown as not completed with its reason.

## A7. Plan, approval, worker, identity

The configurator stores the selected values in the draft's normal `search_space.axis_selections`
and the funded choices in `review.funded_comparison`. Every combination is resolved through the
normal configurator path (registry value ids -> section overrides -> baseline profile ->
canonical section) and must match an approved configuration of the verified study. The frozen
`FundedComparisonPlan` (store `funded_comparison_plans`) records mode, configurations, firms,
size, costs, clock, execution model, source package and exact dates. A historical run needs a
stored `FundedComparisonApproval` naming that exact plan id, recorded only from the owner's own
action on the screen or an owner statement quoted in the approval. The worker runs one process
per configuration and saves one `FundedComparisonResult` bound by SHA-256; the screen and the
review folder read only that result.

## A8. Screen and review folder

See SCREEN_BRIEF.md. One comparison of all configurations; one firm selector (one firm at a
time) drives both the ranking and the detail per configuration and firm, and the chosen firm,
configuration and account are kept per saved result (A12). No summed totals; no credit/growth
widgets. One compact automatic
review folder per completed study (`reports/funded_comparison/...`), documents and necessary
data only; credit and growth tables are not exported in this mode; actual reviewer findings,
when recorded, are a separate document from the review instructions.

## A9. Acceptance

Through the real configurator, two distinct configurations produce their own results for each
selected firm (four results with both firms). Tests cover: payout then failure; more than five
replacements without a credit stop; no expansion after a large receipt; no substitute during
processing; no same-event replacement trade; full versus resumed equivalence; fees; stop,
target, breach and deadline ordering; equality boundaries; gap fills; zero-payout cases;
pair isolation; the real-Core admission and liquidation seams; and screen/export agreement.
The historical comparison runs only the configurations and size the owner selected or
confirmed, on the saved period. No new dates, feeds or sweeps.

## A10. Variation study (version-2 plans, September 23, 2026)

Owner-selected second run: variations around `S0_D80_W1_P1` from the research notes, plus a
new exit. Differences from A1–A9:

- **Configurations outside the verified study.** Each variant is the base configuration's
  registry value ids with some replaced (new registry values: target 2R/3R, one-hour-only
  gaps, parents without three-minute, `exit_policy`). It is resolved through the normal
  configurator path and frozen with its section hash; its authorization is the owner's
  approval of the exact plan. Variants identical to a study configuration keep the study name.
- **Day inputs.** New chart sets read the verified superset cache of the base configuration
  filtered to their charts (Core aggregates every timeframe independently from the same
  one-minute rows; the level timeline uses one-minute bars only).
- **Equivalence.** The no-account replay must equal the saved study's trades where they
  exist, otherwise Strategy-Core's normal `run_day` chain over the same inputs.
- **Scale-out exit** (`exit_policy = scale_out_half_breakeven_hold_to_close_v1`), built in the
  separate Core branch `funded-scale-out-exit` (from the pinned 7c7111e; the pinned Core
  and the IFSM app pin are unchanged; default behavior and identities unchanged). In Core:
  stop first inside a candle; the target candle exits half; the rest's stop moves to the entry
  price and is checked from the next candle; held to that stop or the daily close; risk and
  R use the initial stop. In the funded account: the half fills at the target (limit) on the
  recorded trades; the break-even stop is a stop-market order; the crossing print still marks
  the rest; a break-even stop inside the scale-out minute clears the strategy's slot without
  counting as a liquidation. Scale-out requires a 1R target.
- **Sizes and costs per configuration** (version-2 plan): fixed-target at one mini ($5.14 per
  fill); scale-out at 10 micros (same exposure) at $0.514 per micro per fill, exact in tenths
  of a cent. Micro positions use the mini (NQ) recorded trades as their price path (declared
  limitation).
- **Core source identity.** The plan freezes the Core base commit and a SHA-256 of the
  branch's uncommitted source changes; the worker refuses any other Core.
- **Strategy measures** per configuration from the no-account replay, with the study's
  definitions (net R after 0.514 points per contract round trip, win rate before costs,
  drawdown, longest run of trading days under water).

## A11. Execution semantics, reporting corrections and evidence (post-run review, September 23, 2026)

Added after the owner's independent review of result `5fa65149…`. The economic result is
unchanged; nothing here reruns it.

- **Two execution policies, never interchangeable.** Strategy measures (R, no account) use
  Strategy-Core's candle rules: stop before target on one candle; after a half exit the
  break-even stop is checked from the next candle. Funded accounts use the recorded exchange
  trades in order: the half exit and a break-even stop can occur in one minute. Trades with the
  same timestamp keep the file's order (timestamp, then sequence, then file row). Each fill
  posts its own exact cost. After the half exit only the remaining contracts are marked, and
  TakeProfitTrader's floor follows realized plus open equity. The loss limit is checked
  before the stop or target on every trade. A gap through a (break-even) stop fills at the
  first trade through it. Payout eligibility is checked only when the whole position is flat.
  Tests: `tests/propsim/funded/test_execution_semantics.py`.
- **Stop-difference measure.** `stop_fill_difference_cents` = max(0, direction × (final stop −
  fill)) × contracts closed at that stop × tick value. It is descriptive, already inside each
  trade's profit and never subtracted again. It counts break-even stops inside the half-exit
  minute.
- **Reporting corrections of saved results.** `apply_reporting_corrections` runs when a saved
  result is loaded, after its hash is verified. It re-derives summary-only fields from the
  saved rows and records each change under `reporting_corrections`:
  `stop_difference_summary_final_quantity_v1` and `daily_close_wording_v1`. The saved bytes,
  money, balances, payouts and ranks never change. The review export lists the corrections and
  checks they are summary-only.
- **Substituted minutes.** A minute uses the labeled approximation when its recorded trades do
  not rebuild the candle exactly. In result `5fa65149` every case was a same-timestamp,
  same-sequence matching event whose first trade differs from the candle's open. The high,
  low, last trade and count agreed. The relevance test is whether any stop, target, break-even
  stop or worst-case loss limit lies inside the minute's range. When none does, no order can
  change an exit, survival or a payout.
- **Micro execution.** Micro positions are proxy executions priced on NQ recorded trades.
  This is a disclosed limitation. No MNQ observations exist locally.
- **Evidence exported per result.**
  - `configuration_bindings.json`: every configurator input, the full effective section
    re-resolved to its frozen hash, size, cost, exit rule, the equivalence population, and
    the study-package and Core source binding.
  - `trading_calendar.csv` and a trade-boundary check against the saved schedule.
  - Optional `approximated_minutes.csv` and `reference_record_reconciliation.csv`.
  - A final payload receipt that re-reads the published folder against its manifest.
- **Version-2 plans from the normal configurator.** The comparison screen's "Variations
  around one configuration" builds `build_variation_plan` from saved selections, which persist
  in the draft. It freezes the imported Core identity. The same selections as the September 23
  study reproduce plan `78406b15…` exactly.
- **Worker source selection.** `job start` launches a version-2 plan's worker only on the
  local checkout whose commit and uncommitted-change hash equal the plan's frozen source
  (`research_core_sources.find_core_checkout`); otherwise it refuses.
- **Application launch.** The IFSM launcher uses the pinned Core by default. For the
  research branch, `--research-core PATH` is an explicit, labeled option: the checkout must be
  based on the pinned commit, and the receipt prints the exact patch hash. It is not a pin
  change.
- **Immutable source snapshot.** An internal immutable snapshot of the research Core's
  uncommitted source is kept (`snapshot_core_source`).

## A12. Saved drafts, launch checks and Trade review (IFVG dashboard repair, September 23, 2026)

Screen-level repairs (`docs/ifvg-dashboard-repairs/TASKS.md`, rows R1–R4). No money rule,
plan, approval, result, export or ledger entry changed.

- **Saved drafts are read as saved** (`propsim/funded/comparison_draft.py`): no default is
  merged in and no value is dropped or substituted. Opening, refreshing or navigating never
  saves; only an owner edit does. When the draft is saved in another window, the open page
  resets to it (keyed on the saved file's SHA-256).
- **A draft the running engine cannot represent opens read-only.** Example: the saved
  half-exit variation draft under the pinned Strategy-Core, which has no exit-rule setting.
  The page shows the saved settings, the engine-independent count (64 configurations), the
  saved plan that matches exactly with its approval and status, and the command that starts
  the application with the research engine (`--research-core`). Editing, approving or
  running it needs that engine; it is never reduced to what the pinned engine offers.
- **Launch** re-reads the draft from disk, rebuilds the plan and requires the identical plan
  id and its stored approval. `job start` refuses a historical plan without a stored owner
  approval before queuing a worker.
- **Results screen:** one firm selector (one firm at a time) drives the ranking and the
  detail. The firm, configuration and account are kept per saved result; another result
  never inherits them. Trades and account events are ordered by their recorded instant.
- **Trade review → Study executions** lists funded comparisons that have a saved result
  (Incomplete and review-only results are labeled; an unavailable one is refused, never
  replaced by another study). A trade shows its recorded funded path (entry, quantity,
  initial stop, half fill, remaining quantity, moved stop, final exit, liquidation and
  replacement) on the strategy's verified bars from the plan's bound study package, in
  Chicago 12-hour AM/PM time (CST/CDT). "Review these trades" carries the configuration,
  firm and account; Back restores them. Funded review notes use their own keys (result,
  pair, account, trade).
- **Limitations:** movement inside a minute is not drawn; configurations that were not in
  the verified strategy study have no gap zones.

---

# Part B — Version 3 specification (September 22, 2026), preserved

Governs the deferred budgeted operating mode and documents the completed September 22 pilot.
Text below is unchanged.

# Funded Payout Lab — revised implementation specification

Prepared September 22, 2026. Version 3.

**Status:** Updated requirements and implementation plan, not implemented software or completed financial results. The owner has confirmed the gross $500 minimum, an end-of-trading-day request, stopping further trading as soon as realized payout eligibility is secured, and a two-day non-trading processing period. This supersedes Version 2's immediate-request/instant-receipt proposal. One owner choice remains: whether two days means 48 elapsed hours including weekends, or two business days excluding weekends and processing holidays. Other proposed accounting/calendar conventions retain their explicitly proposed status. No live trading, purchases, payout requests or economic run has been performed by this document.

## 1. Objective and scope

Compare the cash earned from already-funded accounts after their modeled acquisition and operating costs. Large withdrawals are valuable even when an account subsequently fails. Do not optimize for account survival, payout count, smoothness, or trade frequency as substitutes for cash earned.

Start directly at funded status. Initial and replacement accounts have no earned profits, no payout cushion and no inherited account progress. Do not simulate passing evaluations, buying resets or delayed evaluation completion. Do not discuss automation permissions in this work. This is a simulation; it must not buy accounts, request actual withdrawals or submit live trades.

Use two independent comparison instances on the same market timeline:

- TakeProfitTrader, with its own accounts, purchase credits, costs, payouts, cash balance and growth decisions.
- MyFundedFutures, with its own accounts, purchase credits, costs, payouts, cash balance and growth decisions.

They are alternative operating scenarios, not one ten-account operation. There is no shared pool of ten credits, no cross-instance lending, and no total combining their payouts as though both were financed by one five-credit allowance. Every strategy/sizing scenario receives its own isolated pair of instances; scenarios do not share simulation cash.

## 2. Owner-specified funded profiles

These are **owner-defined simulation terms**, not a claim that every current published rule, account-count cap or retail price has been modeled. Preserve these values. A sourced detail may fill a missing loss-limit mechanic but cannot silently overwrite an owner-specified value. Any excluded public rule or intentional simplification belongs in the run's assumptions, not in a misleading assertion of complete live-program compliance.

| Property | TakeProfitTrader instance | MyFundedFutures instance |
|---|---:|---:|
| Nominal funded account label | $50,000 | $50,000 |
| Modeled acquisition cost per new funded account | $102 | $125 |
| Initial funded accounts | 5 | 5 |
| Fresh starting earned profit | $0 | $0 |
| Starting loss allowance | $2,000 | $2,000 |
| Threshold ratcheting | Intraday, including open-position gains | At the defined end-of-day observation |
| Enforcement against current account equity | During open trades | During open trades, against the currently applicable floor |
| Funded consistency condition in this model | None | None |
| Minimum gross withdrawal request | $500 BEFORE the firm split — confirmed | Same |
| Owner-example retained profit cushion | $2,100 | $2,100 |
| Trader share | 80% | 90% |
| Maximum position | 6 minis or equivalent | 3 minis or equivalent |
| Monthly outside-funded purchase entitlement | 5 accounts | 5 accounts |
| Value of five purchases | $510 | $625 |
| Initial simultaneous-account capacity | 5 | 5 |
| Potential capacity progression | 5, 10, 15, 20 | 5, 10, 15, 20 |
| Once realized payout eligibility is reached | Stop further entries for that account until the payout cycle finishes | Same |
| Request timing | First eligible end-of-trading-day window, while flat | Same |
| Processing lockout | Two days from request; clock basis requires confirmation | Same |
| Behavior after processing completes | Resume the same selected trading/sizing policy at a permitted future opportunity | Same |

One mini Nasdaq contract represents ten corresponding micro contracts by index-dollar exposure; enforce combined whole-contract limits, not two independent mini and micro allowances. Product-specific execution costs must be modeled separately. A micro execution does not become exact merely by dividing a mini trade's dollars by ten. [S5, S6]

The $2,100 retained amount is derived from the owner's examples: $2,650 minus $550 and $5,500 minus $3,400. Do not lower this to $2,000 in the TakeProfitTrader comparison simply because the ordinary public buffer differs. It is the requested withdrawal policy for this experiment.

Acquisition costs are model inputs supplied by the owner, not additional charges to the simulated trading balance. The first five cost $510 or $625 respectively; they are not free account capital and not a personal $250,000 deposit.

## 3. Budget, credits and independent cash ledgers

### Confirmed behavior

Each instance gets five monthly account credits. Unused credits carry forward; failure does not refund a used credit. Surviving accounts carry their full actual trading and payout history into subsequent months without being purchased again. No instance uses another instance's money or credits.

### Proposed accounting/calendar convention

Use the initial five credits to buy the first five funded accounts. The initial partial simulation month receives that initial five-account grant; grant five additional credits on the first day of each following Chicago calendar month. Grant no credits during strategy warmup. Persist grant identity so checkpoint/resume cannot issue it twice. The grant is an allowance, not income, not cash earned, and not an acquisition expense until used. No borrowing from future months.

A credit-funded account purchase consumes exactly one credit and records the appropriate $102/$125 expense. Show credit counts and dollar costs separately. The monthly model is therefore $510 versus $625 of new purchasing capacity, not $500 for both.

Keep these ledgers separately for each instance:

1. Unused outside-funded purchase credits.
2. Cumulative after-split payouts received.
3. Gross withdrawals debited from funded trading accounts.
4. Total acquisition and included operating expenses, with funding source.
5. Available received-payout cash after spending from that cash.
6. Account profit/equity and pending payout claims, neither of which is spendable payout cash.

Proposed available payout wallet:

`wallet = after_split_receipts - purchases_paid_from_wallet - other_wallet_spending`

Primary economic result:

`net_cash_earned = after_split_receipts - all_acquisition_costs - included_external_operating_costs`

The initial and replacement account costs count regardless of whether monthly allowance or retained payouts financed them. A wallet-funded purchase reduces the wallet and enters acquisition costs once; do not subtract it twice when calculating net cash earned. Trading costs already reflected in account profit must not be deducted a second time after payouts. Simulated losses inside a failed $50,000 account are not an additional $50,000 personal cash expense.

No automatic personal-cash distributions are modeled in this version. Available payout cash is held in the instance wallet. Report it as available retained cash, not simultaneously as personal money already spent elsewhere. A future personal-withdrawal policy is separate.

### Replacement

A payout-paused account is alive and continues to occupy one funded-account slot. Do not replace it, consume a credit for it, or create a temporary extra account to evade its pause. Other ready accounts in the same instance continue independently.

On breach, close the position using the applicable executable path, cancel remaining orders and permanently terminate that account. Never keep replaying the failed account to an old stop or target and credit the subsequent outcome.

Proposed replacement funding: consume an available monthly/carried credit first. If no credit is available, preserve a pending replacement vacancy rather than inventing a negative credit or borrowing from another instance. Direct earned-cash one-account replacement outside the five-account growth rule is not silently enabled; any later addition is a separate explicit funding policy.

A replacement begins fresh and participates only from its actual creation cursor onward. It can use legitimately warmed market context, not the old account's position, past entry signal, payout eligibility, profits or high-water mark. Use stable vacancy priority when several accounts fail together.

## 4. Profit-funded growth in blocks of five

### Owner's threshold

For each instance independently:

`cost_of_next_five <= 0.25 * current_available_payout_profit_balance`

| Instance | Cost of five more | Minimum qualifying balance |
|---|---:|---:|
| TakeProfitTrader | $510 | $2,040 |
| MyFundedFutures | $625 | $2,500 |

Deduct the actual purchase cost immediately when the growth purchase executes. The test must not reuse the same pre-purchase balance. A failed or duplicated job cannot buy the block twice. Expansion increases simultaneous capacity in five-account steps, to at most twenty per instance; it does not increase the monthly five-credit grant.

### Explicit proposed interpretation of the balance

Use the current available **after-split received-payout wallet**, after previous reinvestment spending, for the 25% test. Separately require cumulative net cash earned after all account costs to be positive, preserving the owner's prior requirement to expand only after profitability. Do not use open trading profit, requested payments, lifetime receipts before spending, outside allowances, or the other firm's wallet.

Under this interpretation, a wallet of $2,040 can finance one $510 growth purchase and leaves $1,530. A wallet of $2,500 can finance one $625 growth purchase and leaves $1,875. Neither remaining wallet can immediately finance the next same-priced five-account block under the same 25% threshold. All amounts are per instance.

If the owner instead intends the 25% test to use profit AFTER all purchases funded by the external allowance too, select an explicitly different balance basis; do not conflate net earnings with available cash. This plan makes the first basis visible for confirmation/correction, rather than silently imposing the more restrictive one.

### Proposed growth timing

Evaluate growth only after a scheduled payout receipt has actually been posted to that instance wallet, never at eligibility, day-end request, or during processing. Apply the selected daily growth-review convention to those receipt events. For the first version permit at most one five-account expansion per instance payout day; batch same-cursor receipts before evaluating. This makes the 25% decision one budget decision, rather than an order-dependent series of immediate purchases. Do not wait until month end unless that cadence is selected explicitly. Store cadence as a trader policy and label it proposed until frozen.

Buy five funded accounts for that block; the purchase uses payout cash, not five free credits as well. Do not automatically buy extra accounts to fill older replacement vacancies. A capacity of ten need not imply ten currently surviving accounts. At capacity twenty, stop expansion. Keep unused credits and wallet balances available under their existing rules; do not automatically invent another reinvestment mechanism at the cap.

## 5. Secure eligibility, request at day end, wait two days, resume

This section is the selected owner withdrawal policy, not a fresh optimization axis. It replaces Version 2's proposed immediate request and immediate settlement. Both firm instances use it independently at the ACCOUNT level.

### Confirmed monetary rule

Use `earned_account_balance` for account value relative to its nominal starting balance: fresh funded account = $0. A profit balance of $2,650 corresponds to a nominal $52,650 display. Store relative and nominal coordinates explicitly.

The owner confirms the minimum is $500 BEFORE the firm split:

`gross_withdrawable = max(0, realized_earned_account_balance_after_costs - 2100)`

Eligibility requires `gross_withdrawable >= 500`, the account to be alive and flat, and the other modeled eligibility requirements to be satisfied. Request the ENTIRE available gross surplus, not just $500. Preserve a $2,100 realized-profit cushion on every selected-model payout. This is the owner-defined policy even if a public program distinguishes first and later payouts.

`trader_receipt = gross_withdrawable * trader_share`

Use exact currency arithmetic and explicit cent rounding. The firm share and trader receipt sum to the gross withdrawal. No consistency rule, extra qualifying days or payout cap is silently added to this owner-defined scenario.

| Realized profit before request | Gross account debit | Account profit left | TakeProfitTrader receipt after processing | MyFundedFutures receipt after processing |
|---|---:|---:|---:|---:|
| $2,599.99 | $0 | $2,599.99 | $0 | $0 |
| $2,600 | $500 | $2,100 | $400 | $450 |
| $2,650 | $550 | $2,100 | $440 | $495 |
| $5,500 | $3,400 | $2,100 | $2,720 | $3,060 |

The cash-receipt columns are not credited at request time. They are due after the processing interval has completed. This policy remains capable of withdrawing a large overshooting win: a trade closing at $5,500 realized profit does not have its result truncated to the $2,600 eligibility threshold.

### Intraday: secure the payout opportunity

After each actual completed position, post its fills and costs and resolve any account breach FIRST. Then evaluate payout eligibility before considering a new entry on the same decision.

If an alive, flat account qualifies:

1. Mark it **Payout secured — finished trading for the day**.
2. Block new entries immediately, cancel working entry orders, and disarm stale entry signals.
3. Keep the account flat through its first eligible day-end request window. A session reopening does not reset this lock.
4. Continue legitimate background market/context tracking without executing trades on this account.

The lock applies only to that account, not all accounts at the firm and not the other firm's independent instance. A paused account continues to count toward simultaneous account capacity.

A temporary OPEN-POSITION profit that could support a withdrawal is NOT realized eligibility. Do not prematurely close a trade, cap a large winner, or move its target just because marked equity touches $2,600. Any separate 'close the position when payout eligibility becomes attainable' policy would be an execution-changing research choice, not authorized by this version. Protective stops, account failure checks, strategy targets and mandatory daily liquidation continue to govern the actual open trade.

This definition secures profit after it has actually been realized. An entry attempted at the same cursor after the qualifying exit must be rejected. If final fee posting later invalidates eligibility, record the shortfall, keep the no-more-trading-today lock, do not invent a payout or spend its proceeds, and use the explicit next-day release rule rather than silently trading again that day.

### Day end: make one full-surplus request

Use the profile's defined first eligible end-of-trading-day payout window, not civil midnight or the first intraday profitable moment. Respect the earlier applicable mandatory position close. The request must occur while the account is flat with no working orders.

At day end, finish all required trade exits, fee postings and breach checks. For MyFundedFutures, capture and apply the scheduled closing-balance threshold update BEFORE deducting that day's withdrawal. Then recheck eligibility and post the full gross request exactly once. A premature afternoon self-pause does not move the end-of-day threshold early.

**Declared bookkeeping convention:** on request, remove/reserve the FULL gross withdrawal from the account's usable trading balance and create a pending after-split receivable. Do not debit it a second time when paid. The selected first version assumes a valid request settles at the chosen two-day completion time without additional transfer fees or discretionary denial. This is a modeling convention, not a guarantee of real provider timing. Keep an explicit rejected/canceled status for errors or a separately configured processing outcome.

The retained account profit becomes $2,100 after a successful request. Its loss floor never resets downward. Pending receivables are not received money, not eligible growth funding, and not an active account's loss cushion.

If the run ends before the day-end request opportunity, report **Secured, not requested**. If it ends after request but before settlement, report **Processing, not received**. Do not silently extend the primary cash horizon to collect pending payouts.

### Two-day processing period — one clock choice remains

The owner's duration is two days. The following alternatives must not be silently conflated:

- **48 elapsed hours:** request instant plus exactly 48 hours, including weekends. Compute elapsed time on an absolute timestamp, then display Chicago local time. A daylight-saving change does not change elapsed duration.
- **Two business days:** second subsequent permitted processing date, excluding weekends and a frozen processing-holiday calendar, at the declared payment clock time. The request date is not day one. This is not two market sessions; Sunday evening opening is not automatically a business processing date.

Do not hardcode one interpretation under a generic '2 days' label. Show the selected basis and exact resume/payment time in the study configuration, account status and review records. Engineering tests may exercise both, but freeze the owner-selected basis before a definitive financial comparison. There is no third hidden two-trading-day counter.

Start this timer at the end-of-day REQUEST, not when the account became eligible intraday. The account remains flat and cannot place or fill new entry orders for the ENTIRE processing interval. It is a waiting account, not failed, retired, or freely replaceable. Calendar-time events must advance even when the market is closed or there are no market bars.

At completion, post the after-split payment to that instance's received-payout wallet exactly once. Release the payout lock only when processing has actually completed. Trading resumes with the same selected strategy and sizing rule on the next new eligible signal after a permitted market/session opening. Daily-close and weekend locks still override a payout release. Do not replay a missed signal from the waiting period or reset the account's lifetime loss floor.

Once received, the cash can enter the instance's existing 25% growth test. A later failure of the trading account does not erase an earlier received payout. Avoid using this principle to count a merely pending claim as already received.

### Illustrative one-account sequence

A completed trade on Monday brings realized profit to $2,650. Stop new trading on THAT account for the rest of the day. At its eligible day-end window, request $550 gross, retain $2,100 and begin the two-day processing timer. Post $440 for TakeProfitTrader or $495 for MyFundedFutures only when processing completes. Until then the account remains flat, consumes its existing slot, and provides no spending money for expansion. Other accounts may continue. On a normal Monday-to-Wednesday week the two alternative clocks can give the same weekday, but Friday requests and holidays reveal materially different behavior; do not assume they are equivalent.

## 6. Loss-limit mechanics and source evidence

The financial model must separate when a loss threshold MOVES from when it is ENFORCED. Both types require checks while a trade is open. The $2,000 allowance is an account rule, not merely a strategy stop and not an end-of-study statistic.

### TakeProfitTrader — moving intraday threshold

First-party documentation says the peak includes realized and unrealized gains, the floor stops rising at the account's starting balance, and reaching the floor liquidates the account. [S1, S2]

In relative-dollar coordinates, starting floor is -$2,000 and the stop-trailing cap is $0. At each legally observed price/fee event, maintain current realized balance, open-position marked profit/loss and all applicable accrued costs. The running high-water mark never falls.

Representative formula, after confirming the actual mark and cost basis:

`floor_next = min(0, max(floor_previous, peak_equity_seen_so_far - 2000))`

Check `current_equity <= floor_next` under the documented inclusive rule. This is processed on the observed ordered path, not only when the strategy closes a trade. Once the floor reaches zero it remains fixed, even if profits rise further or money is withdrawn. Do not reset it daily.

Synthetic example, ignoring fees only for that test fixture:

- Start $50,000, floor $48,000.
- Open-position profit reaches $1,500: equity $51,500, floor now $49,500.
- Equity subsequently reaches $49,500 before the intended target: account is breached even though the trade is only $500 below its original starting balance.
- A later recovery must not restore the account or produce a payout.

Use separate fixtures including costs and gap-through liquidations; the clean arithmetic example does not redefine real fill prices.

### MyFundedFutures — end-of-day ratchet, intraday enforcement

The selected published end-of-day plan moves its floor at new session-closing highs, never downward, and locks it at +$100 relative to the funded starting point. The firm's separate drawdown explanation explicitly includes open-position losses in failure checks (that general explanation is evaluation-framed; the Rapid funded page supplies the selected funded timing and locked-floor terms). Its funded text describes falling BELOW $100 as a breach at the locked floor. [S3, S8]

Starting relative balance is zero and floor -$2,000. The two independent controls are:

- **Update:** intraday unrealized peaks do not raise the floor. Merely closing a profitable trade during the session does not update it early either; use the scheduled session-close balance, not the highest intraday realized balance.
- **Enforce:** on each ordered current-equity observation, include all open-position profit/loss and accrued costs and compare with the currently active floor. A violation ends the account immediately; it cannot recover before the closing observation and be resurrected.

This is NOT a fresh $2,000 daily loss budget or a per-trade $2,000 stop. Prior losses may leave less than $2,000 of room; the current floor is the governing boundary. After the floor locks, larger retained profit may provide more room. Never reset the floor downward at midnight, on a new session, after a loss, or after a payout.

Required clean arithmetic fixture, excluding fees solely for illustration: start with relative balance $0/floor -$2,000; an open +$1,500 profit leaves that floor unchanged; close the trade at +$1,000 but leave the floor unchanged until the scheduled close; at that close raise the floor to -$1,000. During the following session, relative equity falling below -$1,000 fails immediately while the position is open. Fees-inclusive and exact-boundary fixtures are separate tests.

At the exact profile-defined closing snapshot, compute the new allowable floor from the realized closing balance, capped at +$100. It must remain monotonic. Reconcile scheduled exit, posting of fees, closing observation and payout debit order. Same-time events cannot choose whichever ordering increases survival.

`floor_next_at_close = min(100, max(floor_previous, realized_closing_balance - 2000))`

Withdrawal does not lower the floor. After the floor locks at +$100, an account left at +$2,100 has $2,000 above that floor. The analogous TakeProfitTrader account whose floor has locked at zero has $2,100 above its floor; the same retained payout cushion does not imply identical effective loss room.

**Boundary sign:** Verify the equality rule for every phase of the chosen program before publishing a rules-exact scenario. The cited plan clearly states the locked-floor below-$100 condition; it does not fully specify every pre-lock equality/fee-posting detail. Do not copy TakeProfitTrader's inclusive comparison by convenience or pretend this documentation gap is resolved. Expose the compiled comparator and source/assumption in internal rules and compact validation output.

### Trading and account safety ordering

- Only price observations while the position actually exists can change its open gain, floor or failure state.
- A strategy stop, profit target, scheduled exit and account breach must be resolved on the same causal path. No double exit, double cost or later target after account death.
- A jump through a liquidation boundary fills under the actual execution convention, not automatically at the most favorable threshold price.
- Costs affect the account at their defined events. Do not deduct future fees twice or ignore entry fees when testing remaining allowance.
- Per-account forced liquidation must feed back into that account's strategy state and later opportunity availability.
- Source closes and one-hour/four-hour gap validity are a different subject from funded-account loss limits. Retain the existing own-chart validity implementation.

## 7. Critical accuracy limit: minute candles are not a full live price sequence

The owner's demand for correct live-style intraday loss measurement cannot be satisfied by checking only closed-trade profit or the closing price of each minute.

Minute candles give opening, highest, lowest and closing prices, but do not establish which of the high or low occurred first, or all reversals in between. Different paths with the same candle can raise and cross an intraday floor differently. Maximum favorable and adverse excursion summaries are also insufficient to recover that order. This is a loss of information, not something an assertion test can fix.

Implement an account engine that consumes ordered price/mark events, with timestamp and stable sequence, through the actual strategy/position life cycle. Inspect existing authorized local data for a suitable stream. Record the mark convention, instrument mapping, costs, coverage and executable-price policy. Event-level historical testing is exact relative to those supplied observations and declared fill assumptions; it is not proof of broker execution or unobserved quotes.

The ordinary previous strategy inputs are minute-based. Do not silently fetch new feeds or protected dates. If finer ordered observations are absent:

- Engineering can proceed using deterministic synthetic paths and a declared minute-scenario mode.
- A financial result must be labeled an approximation where within-candle order affects account survival.
- Test alternative paths/mark conventions where feasible and record sensitivity; a pair of open-high-low-close/open-low-high-close assumptions is NOT a complete rigorous bound on every possible within-candle path.
- Do not publish a best-case assumed order as live-accurate, or claim the intraday and end-of-day comparisons are equally resolved when one is more path-sensitive.
- Exactness-required economic publication must fail clearly on insufficient coverage rather than invent a favorable sequence. A separately authorized approximate result remains reviewable with an understandable caveat.

The review output should include the source observations necessary to inspect each actual or ambiguous breach, not a raw tick dump. Full source data remains internally available for targeted audit.

## 8. Existing code: reuse boundaries and implementation work

The earlier CODE_MAP was based on uploaded snapshots, not the latest working tree. Revalidate the normal launcher, current import roots and active pending run. Never reset the current tree to a historical snapshot or change a running study.

### Strategy-Core

Inspect the existing generic execution layer before adding a strategy-specific duplicate. Relevant inspected paths include `src/strategy_core/strategies/ifvg_smc/{reducer,replay,state,section,records}.py`.

Keep detector/confirmation logic and protective/scheduled exits there. Add only tested event interfaces required for account-specific admission, whole-contract size and external account liquidation. Each genuinely different account position/strategy path needs isolated state; shared immutable market features may be reused. Account purchases, monthly credits, splits and cash wallets must not be put into the pattern reducer.

### Quant-Lab

Extend the existing `src/alpha_lab/propsim/` modules, especially:

| Existing area | Required extension |
|---|---|
| `account.py` | Explicit fresh-funded start and fresh-funded replacement; ordered open-equity breach; account/payout state |
| `firm_contracts.py`, `contract_evidence.py` | Owner-defined profile provenance and separate researched threshold mechanics |
| `trade_path.py`, `search_bridge.py` | Verified ordered-path capability and honestly labeled approximate modes; do not pretend a reserved but unsupported mode already works |
| `portfolio.py`, `simulation.py`, `calendar.py` | Separate instance coordinators on a common clock, monthly credits and growth |
| `risk.py` | Whole mini/micro quantity, combined limits, fixed strategy size after payout |
| `withdrawal.py` | Full-surplus greedy request, retained cushion, minimum semantics, daily cycle |
| `prop_metrics.py`, `event_detail.py` | Net received cash, account costs, explicit failure events and compact decision evidence |

Introduce small modules such as `instance.py`, `cash_ledger.py` and `payout_lifecycle.py` only if there is no equivalent current facility. Names are proposals, not instructions to duplicate existing components.

The inspected `AccountWalk` starts accounts in evaluation and refuses payout delays; existing portfolio loops do not provide the new instance wallet. Those are inspection findings to confirm in the current code. Refactor the actual supported account path, not a disconnected demonstrator.

Reuse the existing `scripts/ifvg_workspace.py`, results/pipeline renderers and shared presentation layer for completed results. The normal study wizard must save, reopen and pass both instances and actual settings to the worker. Existing strategy gates for minimum trade frequency/smoothness must not silently discard funded-payout candidates; actual correctness and account constraints remain mandatory.

## 9. Final screen: two separate operating scenarios

Show full firm names, plain English and Chicago times in a 12-hour clock. No unexplained abbreviations, internal source paths, hashes, reducer counters or raw configuration objects.

The final screen begins: **“Two separate funded-account simulations on the same market period.”** Do not show a grand total suggesting ten jointly budgeted accounts. Side-by-side summaries or two comparable tabs are appropriate.

Per instance, primary cards:

- Net cash earned after all account costs.
- Payouts received after the firm's share.
- Largest single received payout.
- Acquisition/operating costs.
- Accounts lost, separated into before first payout and after a payout.

Secondary user information: current active accounts/capacity, unused purchase credits, available payout wallet, gross requested withdrawals still pending, next-five purchase price and growth eligibility threshold. Distinguish trading profit still inside funded accounts from received money.

Charts: cumulative payouts/costs/net cash on a shared date axis; monthly received cash and expenses; account journeys with starts, payouts, failures and replacement links. An account that pays then fails is not rendered as though it earned nothing. Show zero-payment months and partial months.

Selecting an account shows an understandable equity-versus-loss-limit path and actual breach reason. Selecting a trade shows size, price points, dollars, initial risk, close reason and actual chart timeframes. Human text must make a material minute-price approximation visible without exposing implementation jargon.

Comparisons use the same selected strategy/sizing policy on the same timeline but preserve account-specific outcomes. Five copies are not five independent experiments. The count-based allowance gives different dollar spending capacities ($510 versus $625); show cost-adjusted measures alongside absolute cash, not an unearned equal-cash claim.

The screen must additionally show **Trading**, **Payout secured — finished for today**, **Payout processing — trading paused until [Chicago date/time]**, **Ready to trade**, and **Account lost**. Display requested gross amount separately from the after-split cash due and received; the pending timer starts at request. Add separate counts/durations for no trading because of payout protection, processing, session hours, missing entry, missing credit and account failure. A successful payout pause must not be labeled an unexplained strategy drought.

For a shared five-mini strategy, MyFundedFutures is unsupported under the chosen three-mini cap; refuse the scenario or explicitly select another size, never silently clamp it. Separate matched-size comparisons (one/two/three minis) from maximum-size comparisons (six versus three). Do not launch an unrequested sizing grid during implementation.

Older strategy-only results remain accessible and say payouts were not simulated rather than display invented zero payouts. Screen and review export must read the same immutable financial result object.

## 10. Automatic compact review folder — documents and necessary evidence only

Every verified completed funded simulation publishes a separate versioned review folder, optionally zipped. This is part of the application workflow, not a manual agent cleanup job.

Use this consolidated allowlist across BOTH instances and all compared configurations:

```
README.md
QUESTION.md
RESULTS.md
DECISIONS.md
TRADING_RULES.md
RESEARCH_LEDGER.md
DATA_DICTIONARY.md
settings.json
firm_rules.json
rule_sources.csv
run_manifest.json
validation_summary.json
ledger.jsonl
instance_results.csv
account_journeys.csv
cash_ledger.csv
payout_events.csv
credit_events.csv
growth_events.csv
monthly_results.csv
trades.csv
account_events.csv
rule_boundary_evidence.csv
comparison.csv
charts/cash_over_time.png
charts/account_journeys.png
```

All applicable tables carry run/scenario/instance/account identities internally and readable labels. Put one shared logical table in one format, not copies per account or in several formats. Larger necessary tables may use standard Parquet instead of CSV with a documented reason; do not truncate failure evidence to hit a file-count target.

Critical contents:

- `QUESTION.md`: frozen cash objective, exact comparison, input period, assumptions and what would falsify a hypothesis. Preserve the former activity objective as historical, not the current winner criterion.
- `TRADING_RULES.md`: owner-defined terms; starting state; daily close/no-weekend; credit/grant/growth; retained-cushion/minimum/split; payout clock and settlement; exactness limits.
- `ledger.jsonl` and its readable history: cumulative append-only record of completed, failed, proposed and superseded experiments. Do not make a fresh one-run ledger and call it cumulative.
- `cash_ledger.csv`: all actual modeled receipts and purchases, funding origin, before/after balances and reconciliation. Credits are not mixed into earned income.
- `payout_events.csv`: eligibility and stop-trading timestamps; each gross request, firm share, net amount, request/debit/processing/receipt timestamps; selected two-day clock and exact unlock time; post-withdrawal cushion/floor; final pending/rejected states. Separate secured-but-not-requested and processing-at-cutoff claims.
- `credit_events.csv`: grants, purchases, remaining count, event idempotency and carryover.
- `growth_events.csv`: each eligibility decision's wallet basis, net-profit guard, 25% cap, batch cost, result and actual before/after capacity and wallet.
- `trades.csv`: all funded-account executions including account-liquidated trades, costs and actual exits, with both precise machine timestamps and Chicago display columns.
- `rule_boundary_evidence.csv`: meaningful peak/floor transitions, actual/ambiguous breaches, pre/post-payout thresholds, attempted trades during payout protection or processing, and deadline violations, with enough ordered observations to review those claims. Record zero-violation checks compactly; do not dump unchanged mark state on every tick.
- `account_events.csv`: meaningful per-account ready/secured/waiting/processing/released/failure transitions, reason and before/after state, blocked-entry counts and account-slot occupancy during processing; consolidate rather than create one file per event.
- `validation_summary.json`: compact outcomes of financial conservation, no cross-instance money, rule coverage, price evidence, actual executions, resumed-state equivalence, screen/export agreement. No full test logs or technical code dumps.

Full data, source, scripts and engineering receipts stay internally archived. The review folder contains **no scripts, patches, dependency files, notebooks, source trees, raw market dumps, environments, caches, old archives or thousands of single-event files**. A suspected defect gets a targeted deeper export, not routine bulk packaging.

Stage -> verify allowlist/financial joins -> atomically publish. A repack changes export version, not the economic run. If zipped, produce a small external final verification receipt tied to the exact archive and payload-manifest hashes. Verification runs inside the application, not via a script included in the review folder.

## 11. Acceptance tests and implementation order

### Before any financial ranking

1. Confirm current code map, no changes to the running study, owner profile provenance and data fidelity.
2. Implement one funded account with ordered path, the correct independent floor-update and live-enforcement controls, forced exits, realized-eligibility entry lock, day-end split-aware gross request and two-day processing/receipt.
3. Add two independent five-account coordinators, grants/carryover, replacement and block growth.
4. Add final-screen shared presenter and automatic allowlisted review publication.
5. Complete a bounded historical pilot only after financial semantics and data capability are frozen. Do not launch a large search simply because the engine now runs.

### Exact synthetic cases required

- Fresh-funded state has zero earned cushion, floor -$2,000 and one paid acquisition; no evaluation transition/fee.
- TakeProfitTrader open equity 50,000 -> 51,500 -> 49,500 fails at the documented boundary, and later recovery cannot pay.
- Intraday floor stops at starting equity; higher open peaks above the locking point do not keep raising it.
- MyFundedFutures open intraday gains AND a closed profitable trade do not ratchet early; only the next defined session-closing observation does. Existing floor breach liquidates while the position is open. Test reduced remaining allowance after prior losses, no midnight reset, no downward update and the fixed floor after lock.
- Locked MyFundedFutures +$100 boundary uses a deliberately resolved comparator; equality and one-cent crossing both tested.
- Realized balance 2,599.99 not eligible; 2,600 -> gross500; 2,650 -> gross550; 5,500 -> gross3400 under the CONFIRMED pre-split interpretation. Eligibility stops further entries immediately; debit occurs at day-end request, receipt after processing. Verify gross debit, splits, cushion and all wallets.
- Open-position equity reaches 2,600 but its actual closed result does not: no payout qualification or automatic early target. A trade closing at5,500 retains its full outcome and requests3,400, not500.
- The qualifying trade exit resolves before a same-cursor new signal; the new entry is blocked. No new entries/fills after eligibility, during the waiting-to-request phase, or during processing. Cancel working entries; maintain protective orders until actual flat state.
- Day-end MyFundedFutures balance snapshot/floor update precedes the withdrawal debit; the intraday decision to stop does not move that floor early.
- Processing timer starts at request, not intraday eligibility. Exercise both48-hour and two-business-day scheduler fixtures, Friday/weekend, holidays, daylight-saving changes, no-price-event intervals and run cutoff, without choosing the unconfirmed economic setting.
- Pause is per account. Other accounts continue; paused accounts still occupy slots, cannot be replaced as failures, and cannot be bypassed with a free extra account.
- Processing completion credits cash once and releases only the payout lock, not the market/session/daily-close locks. Use the next new eligible signal, never a queued stale signal from the pause.
- Final day payout waiting/requested but not settled remains unreceived at cutoff. Checkpoint/resume preserves locked status, exact due time and pending money without duplicating a request or credit.
- Large paid withdrawal followed by loss retains prior receipt and charges only the next actually purchased replacement.
- Payout pending versus settled cannot finance growth early. The selected model prohibits instant receipt or post-request trading; any legacy zero-delay behavior remains historically separate and cannot enter this run.
- Five initial credits become zero after exactly five purchases, with $510/$625 expense respectively. No warmup grants. New month grant exactly once; unused counts carry.
- All five copies fail together; when credits are absent the instance waits, even when the other instance has abundant money.
- A TakeProfitTrader wallet of 2,039.99 cannot buy the 510 growth group; 2,040 can and leaves1,530. MyFundedFutures 2,499.99 cannot;2,500 can and leaves1,875.
- Wallet debit plus acquisition cost does not double-reduce primary net earnings. Monthly credits are never earnings.
- Capacity 5 ->10 ->15 ->20, same-day grouping and one growth check convention; maximum20 enforced per instance, no inadvertent portfolio-wide20 cap across alternatives.
- Sizes six/three enforced independently; a shared oversized profile cannot be silently clipped. Combined mini/micro exposure verified.
- Stop/target/breach/deadline ordering; no post-liquidation price maxima, no double exit, gap-through pricing, no old entry on replacement.
- Morning window Chicago7:00 AM inclusive to10:30 AM exclusive; before4:00 PM closure with selected buffer, earlier applicable schedule and weekend lock; overnight reopening and monthly scheduling coherent.
- Ambiguous within-minute price sequence affects survival and is never presented as exact. Multiple cloned accounts do not produce independent probabilities.
- Complete vs resumed simulation, ledger idempotency, all dates including no-payment months, missing source coverage, UI/no-payout states and clean export.

No tests or simulations were executed by this planning document. The agent must provide actual completion receipts for implementation work.

## 12. Confirmed choices and one timing decision

Confirmed in the owner's latest reply:

- $500 is the minimum GROSS request before the 80%/90% split.
- Stop further trading on an individual account when it reaches realized payout eligibility intraday.
- Submit the full eligible amount at the first eligible end-of-trading-day window, not immediately intraday.
- Keep that account flat during a two-day processing period.
- Resume the same trading approach only after processing completes.
- MyFundedFutures enforces its established floor during open trades even though that floor only moves at the day-end update.

**One remaining owner timing choice:** Does 'two days' mean **48 elapsed hours including weekends**, or **two business days excluding weekends and processing holidays**? The first interpretation of a Friday request can finish Sunday; the second normally finishes Tuesday, absent a holiday. Market/session/mandatory-close restrictions still apply after either completion.

Do not ask again whether $500 is before the split or whether immediate receipt is intended. Those are resolved. The daily allowance/calendar and growth-wallet/cadence conventions elsewhere remain visible proposed choices rather than hidden implementation details.

### Revision checklist

Version 3 supersedes every Version 2 passage that suggested immediate intraday requests, immediate receipts or trading through payment processing. It preserves both independent five-account instances, cost/credit/growth terms, current chart strategy, hard daily/weekend constraints, the final-screen redesign and the document/data-only review-folder allowlist. Engineering and simulations have not been executed by this plan.

## Research notes and exact sources

Checked September22,2026. The source summary below is distinct from owner-defined simulation assumptions. Preserve dated source metadata in internal evidence. No public account-count cap, news policy or inactivity rule is silently substituted for the selected hypothetical comparison.

- [S1] TakeProfitTrader PRO Account Rules — intraday realized/unrealized high-water mark, cap at starting balance, touch liquidates: https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15171769361053-PRO-Account-Rules
- [S2] TakeProfitTrader tracking explanation: https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15171820366109-How-to-Keep-Track-Of-Your-Drawdown
- [S3] MyFundedFutures Rapid end-of-day50k — end-of-day ratchet, lock at+$100, $500 requested minimum and90%share: https://help.myfundedfutures.com/en/articles/16158363-rapid-eod-50k-a-comprehensive-look
- [S4] TakeProfitTrader account-to-wallet instructions — gross amount versus80% received; flat/no orders; daily availability may differ from instant assumption: https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15172253980061-How-to-Withdraw-from-PRO-Account-to-the-Wallet
- [S5] Exchange micro contract dollar value: https://www.cmegroup.com/markets/equities/nasdaq/micro-e-mini-nasdaq-100.contractSpecs.html
- [S6] Exchange mini contract reference: https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html
- [S8] MyFundedFutures end-of-day drawdown explanation — open equity is included in failure checks; this general page is evaluation-framed and is not used to import evaluation terms into the funded profile: https://help.myfundedfutures.com/en/articles/8348565-end-of-day-eod-drawdown-explained
- [S7] TakeProfitTrader public buffer/split differs from owner's deliberately retained+$2,100 rule: https://takeprofittraderhelp.zendesk.com/hc/en-us/articles/15172219527581-PRO-Account-Profit-Split-Withdrawal-Rules

Missing exact program mechanics must be sourced or separately labeled assumptions, never declared verified merely because a profile contains a number. General public rules do not prove historical user-specific account terms.
