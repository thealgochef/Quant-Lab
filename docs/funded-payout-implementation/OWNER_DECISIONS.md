# Owner decisions

**Maintained by the implementation agent. The owner does not need to edit this file.**

**Current mode (September 23, 2026): single-account configuration comparison.** See the decision log entry of that date at the end of this file. The five-account/credit/growth decisions below remain recorded with their original dates; they now govern only the separate, deferred budgeted operating mode (SPEC.md Part B) and the completed September 22 pilot.

## Confirmed requirements (September 22, 2026; budgeted mode)

Use the full Version 3 specification. Do not ask again about the two separate firms, five funded starts each, separate five-credit monthly allowances, carryover, owner account costs, profit splits, six/three mini limits, correct intratrade failure checks, 25% growth test, gross minimum of $500, retained $2,100 cushion, or the stop/request/processing policy. Large paid withdrawals still count if the account later fails. Evaluation-stage and automation-permission work remain out of scope.

## Processing clock — open owner decision

**Status: RESOLVED September 22, 2026 — two business days (see decision log).**

Ask in the Claude conversation, when the choice is needed: **Should the two-day payout pause mean 48 elapsed hours including weekends, or two business days excluding weekends and processing holidays?** Record the answer here with its actual date. Search existing task decisions for a later answer first; do not repeat a resolved question.

Both clocks begin at the end-of-trading-day REQUEST, not the intraday eligibility lock. No entries occur before processing completes. Received cash and payout-lock release happen only then, and market/session/deadline restrictions still govern resumption. Business-day mode requires an explicit processing calendar and payment time. Build and test both modes while waiting; do not pick whichever produces better financial outcomes.

## Proposed conventions — not silently approved

| Subject | Version 3 proposal |
|---|---|
| Initial credits | First five purchases consume the first month's five credits; no warmup grants. |
| Monthly calendar | Five credits on the first Chicago calendar day of subsequent months; unused credits carry. |
| Replacement | Credit-funded replacement first; no cash-funded single replacement outside the agreed growth policy by default. |
| Growth basis | Available after-split received cash after prior reinvestment, with the previous positive-net-earnings guard. |
| Growth cadence | At most one five-account expansion per instance receipt day, after posted receipts. |
| Payout booking | Gross amount debited/reserved once on request; trader cash credited once after processing; no extra fees/discretionary denial in the initial model. |

Consolidate genuinely unresolved policy choices into one brief request before a financial pilot. Record proposals separately from confirmed owner statements. Do not make the owner re-explain the entire model.

## Pilot prerequisites the agent must resolve

Verify actual source roots and safe task copies; current daily-close baseline; one explicit sizing/strategy choice supported by both firm limits; authorized input identity and dates; ordered-price coverage; costs/mark/fill conventions; sourced or explicitly assumed equality and calendar details. The code map is navigation, not proof of the current implementation. Missing historical data blocks an exact historical claim, not independent engineering work. No new grid or feed purchase is authorized by this package.

## Decision log

No new owner decisions were recorded by the original package. No prior
`docs/funded-payout/` folder or earlier task record exists in the current tree
(checked September 22, 2026), so there was nothing older to reconcile.

### September 22, 2026 — owner answers in the Claude conversation

Asked once, as one consolidated request, before any financial pilot. The owner's
own words are preserved in the conversation; the summary below is exact in
substance.

1. **Processing clock — RESOLVED: two business days.** The pause ends on the
   second business day after the request date, skipping weekends and processing
   holidays; the request date is not day one. The agent proposed, and the owner
   selected, payment at **4:00 PM Chicago** on that date with **US Federal
   Reserve bank holidays** as the processing-holiday calendar. The 48-elapsed-hour
   mode stays implemented and tested but is not used for the definitive pilot.
2. **Pilot strategy — daily-close control `S0_D160_W1_P0`.** Original three
   windows, 160-tick opposing distance, one-tick opposing minimum, no one-minute
   supporting chart; mandatory 3:55 PM Chicago close. Same January 13 – June 10,
   2026 evaluation period, no new dates, no sweep.
3. **Pilot size — one mini Nasdaq contract in both firms** (matched size, within
   both the six-mini and three-mini limits).
4. **Conventions — accepted for this pilot, with the owner's clarifications:**
   - Accept the proposed initial credit, monthly grant, growth-wallet basis and
     one-expansion-per-receipt-day cadence, and the payout booking convention.
     Each firm's accounts, credits and payout money stay separate.
   - Loss checks use actual ordered last-traded-price observations when
     available. Minute-candle assumptions must never be described as exact
     intraday enforcement. Identify the price data used and disclose any
     uncertainty that changes account survival.
   - Modeled cost: **$5.14 at entry and $5.14 at exit per mini**, not charged
     again anywhere else.
   - MyFundedFutures before its floor locks: failing **at or below** the floor is
     an explicit pilot ASSUMPTION unless a precise source establishes it. After
     the floor locks at +$100, keep the documented **below** condition. Test
     equality and one-cent crossing separately.
   - Keep the agreed payout-protection and two-day processing policy unchanged.
   - **The pilot validates the simulator. It does not select the final strategy
     or position size.**

### Proposals now approved for the pilot (were "proposed" above)

| Subject | Approved pilot convention |
|---|---|
| Initial credits | First five purchases consume the first month's five credits; no warmup grants. |
| Monthly calendar | Five credits on the first Chicago calendar day of each later month; unused credits carry. |
| Replacement | Credit-funded replacement first; a vacancy waits for a credit otherwise. |
| Growth basis | Available after-split received wallet after prior reinvestment, plus positive cumulative net cash. |
| Growth cadence | At most one five-account purchase per firm per receipt day, after same-time receipts post. |
| Payout booking | Gross debited once at request; trader share credited once at completion; no extra fees or denial. |
| Mark price | Last traded price (MBP-1 trade prints) for open-position equity. |
| Costs | $5.14 per mini at entry fill, $5.14 per mini at exit fill. |
| MyFundedFutures pre-lock comparator | At or below floor fails (assumption); locked floor: strictly below +$100 fails. |
| Processing | Second business day after request, 4:00 PM Chicago, US Federal Reserve holidays. |

Agent-side engineering choice, not an owner decision: Strategy-Core is used
read-only. Account admission, liquidation, credits and cash stay in Quant-Lab.

### September 23, 2026 — owner scope update: single-account configuration comparison

Source: `AGENT_SCOPE_UPDATE.md` and `UPDATE_AND_CONTINUE.md` in this folder, supplied by
the owner on September 23, 2026. This entry records the change; it does not claim that
any new simulation has completed.

**The research question for this mode:** which tested strategy configuration produces the
most simulated cash received after the cost of every funded account used, over the same
selected historical period? Large payouts count even if the account later fails. Trade
count, account survival, win rate and frequent small withdrawals are not substitutes.

**New for this mode:**

1. Each configuration resolved in the normal study configurator, for each selected firm,
   is one separate comparison with **at most one live funded account at a time**. Two
   firms mean two results per configuration; the application supports whatever number of
   configurations is selected.
2. A failed account is finalized and replaced by a fresh funded account under the same
   configuration, **charged $102 (TakeProfitTrader) or $125 (MyFundedFutures)**, with **no
   monthly-credit limit and no requirement to pay for it from earlier payouts**. No
   evaluation or purchase delay is modeled (a research simplification). The replacement
   trades only from a later valid opportunity; it never re-trades the failed event. No
   purchase after the run's cutoff.
3. Each pair follows its own account-driven strategy state: early liquidation, payout
   pauses and replacements can change later opportunities. A reference trade list may be
   reused only where equivalence for that exact path is demonstrated.
4. `net cash earned = after-split payouts received − initial and replacement account costs
   − other included external costs`. Also report total account spending and the largest
   cumulative shortfall before receipts covered costs (no budget stop).
5. The unresolved execution findings of the pilot are now required work: a stop is not a
   guaranteed fill at its price when recorded prints cross it; price coverage of the 156
   pilot trades does not cover newly generated trades. Corrected conventions are versioned
   and compared, never overwritten.

**Deferred for this mode (preserved for the separate budgeted operating mode):** the
five-account starting group; five monthly purchase credits; credit carryover; the shared
replacement budget and replacement vacancies; reinvestment; the 25% five-account growth
test and its cadence; the twenty-account ceiling; copy trading of one signal stream to
several accounts. There is no account rotation or mixed-strategy allocation choice in this
mode.

**Unchanged and not to be asked again:** $102 / $125 acquisition costs; $2,000 loss
allowance with each firm's floor mechanics (TakeProfitTrader intraday peak, locks at $0,
fails at or below; MyFundedFutures session-close update, locks at +$100, at-or-below
before the lock as the approved assumption, strictly below after it); 80% / 90% trader
share; six / three mini limits; $5.14 per mini per fill; mandatory daily and weekend
flat; $500 gross minimum above the retained $2,100 cushion; stop entries once realized
eligibility is reached; full-surplus request at the day-end window; **two business days,
paid 4:00 PM Chicago, US Federal Reserve holidays**; received payouts stay received after
a failure; pending claims at the cutoff are not received. **A payout-processing account is
alive and is never replaced or supplemented to bypass its pause.**

**Resolved by this update (were open after the pilot):**

- *Shared-signal simplification* — rejected for this mode. Each configuration and firm
  runs its own strategy state (item 3).
- *Stop fills at the recorded stop price* — no longer acceptable as verified history.
  The corrected execution convention is recorded in SPEC.md Part A and versioned against
  the pilot's convention.

**Still open (not decided here):** which exact historical configurations and position size
the definitive comparison runs. The owner's selection through the configurator (or a
confirmed plan) is required before that historical launch; engineering work continues
meanwhile.

### September 23, 2026 — owner selection for the historical comparison

Asked once in the Claude conversation, with the options listed there. The owner selected:
**all 32 approved daily-close configurations (the September 18 study), both TakeProfitTrader
and MyFundedFutures (64 separate results), one mini Nasdaq-100 contract per trade**, on the
saved January 13 – June 10, 2026 period (10 warmup and 107 evaluation dates), with the agreed
$5.14 per fill and the two-business-day, 4:00 PM Chicago processing clock. This is the
bounded authorization for that one plan; it is recorded as a `funded_comparison_approvals`
envelope (channel `claude_conversation`) naming the exact frozen plan. No other size, dates or
configurations are authorized by it.

### September 23, 2026 — owner selection for the variation study (second run)

Asked in the Claude conversation after the first comparison completed. The owner asked to
run "all the config variations you suggested on trying out next that seem promising" from
`reports/IFVG_Configuration_Research_20260923/3_NEW_VARIATIONS_TO_TEST.md`, and chose:

1. **Variations: the 48-configuration set plus the scale-out exit.** Around the best tested
   configuration `S0_D80_W1_P1`: target 1R / 2R / 3R x larger gaps one-hour+four-hour or
   one-hour only x parent charts with or without three-minute (one-minute kept) x long only or
   long and short x original three windows or all open-market hours (48), plus the scale-out
   exit (half at 1R, the rest's stop moved to the entry price and held to that stop or the
   daily close) on the same four non-target axes (16). 64 configurations.
2. **Measurement: both** — the funded single-account cash comparison (TakeProfitTrader and
   MyFundedFutures separately, 128 results) and strategy measures in R from the same replays.
3. **Size:** fixed-target configurations at **one mini** ($5.14 per fill, unchanged); scale-out
   configurations at **10 Micro Nasdaq-100 contracts** (same dollar exposure as one mini; 5 exit
   at 1R, 5 are held) at **$0.514 per micro per fill** (one tenth of the mini cost).

Implementation choices recorded here (agent, not owner decisions): the scale-out exit is built
in a separate editable Strategy-Core checkout (branch `funded-scale-out-exit`, from the pinned
7c7111e; the pinned checkout and the IFSM app pin are unchanged); within a candle the
strategy checks the break-even stop only from the next candle, while the funded account uses
the recorded trades' exact order; the new chart sets reuse the verified superset day caches
filtered to their charts (identical to a fresh build by construction and on a checked day).
Same saved period and dates; no new data.

### September 23, 2026 — owner feedback on the completed variation result (`5fa65149…`)

The owner's feedback file (`FEEDBACK_FOR_AGENT.md`, after an independent review) directs:

- **Preserve this completed result.** The partial-profit exit is the leading development
  candidate under the stated model. It is not verified micro-market execution and not a
  forecast.
- **Not authorized:**
  - a new broad parameter sweep;
  - new data dates;
  - feed purchases;
  - live trading;
  - silent promotion of the research Core branch.
- **Unchanged:**
  - one funded account at a time per configuration and firm;
  - paid replacements;
  - the payout-protection rules;
  - the two-business-day wait.

  Copied accounts, monthly credits, growth and minimum-activity gates stay out.
- Fix the stop-difference summary, publish a new export revision of the same economic run, and
  add regression tests. A new export revision is not a new economic run.
- Keep the approved $0.514 per micro per fill; do not silently replace it. The actual micro fee
  schedule, or explicit sensitivity costs, would be a separate comparison.
- Large one-time payouts are part of the objective. Removing the largest payout is
  sensitivity information, not a rejection test.
- Inspect only authorized local micro observations. None exist; the proxy label stays.

Responses are recorded in TASKS.md (section "Post-run review response") and SPEC.md A11.
