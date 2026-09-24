# Owner decisions

**Maintained by the implementation agent. The owner does not need to edit this file.** This package does not record a new approval or change the financial specification.

## Confirmed requirements

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
