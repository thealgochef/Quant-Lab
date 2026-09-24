# Funded configuration comparison — owner scope update

Status: Requirements for implementation. This document does not claim that changes or new simulations have been completed.

## The question

**Which tested strategy configuration produces the most simulated cash received after the cost of every funded account used, over the same selected historical period?**

Large individual payouts are valuable even if the account subsequently fails. Do not substitute maximum trade count, shortest inactivity, highest winning percentage, longest account survival, or frequent minimum withdrawals for the cash objective. Show spending, payout timing and losses so the owner can evaluate the tradeoff. These are historical research comparisons, not a claim of future returns.

## What this clarification supersedes

For this research mode, the owner explicitly defers the five-account starting group, five monthly purchase credits, credit rollover, shared replacement budget, reinvestment, growth in five-account blocks, and twenty-account ceiling. Preserve their existing implementation and historical records as a separate later operating-budget mode. Do not let those rules restrict or multiply results in this mode.

Copy trading is disabled. Do not broadcast a reference strategy's completed trades to five accounts. There is no account rotation policy or mixed-strategy account allocation to choose in this mode.

## One configuration, one current funded account

For each configuration selected through the existing IFSM study configurator, evaluate one current funded account per selected firm. With both TakeProfitTrader and MyFundedFutures selected, there are two separate comparisons of that configuration. Thirty-two configurations therefore produce sixty-four configuration-and-firm results, not five copies per configuration and not a jointly financed portfolio. Thirty-two is an example/current study size, not an application limit: support the actual selected configuration count.

Each configuration-and-firm pair owns a chronological sequence of accounts:

1. Start one fresh funded account with zero earned profit and the selected firm's initial loss allowance. Charge its acquisition cost.
2. Generate and execute this configuration's own valid opportunities while the current account can trade. Maintain one live funded account and the existing one-position restriction for that pair.
3. Apply the previously agreed funded loss limits, fees, trading hours, mandatory daily/weekend closing, payout protection, and processing pause.
4. When the account fails, finalize its actual liquidation and costs, record the failure permanently, and introduce a fresh funded replacement using the same configuration. Charge the replacement cost. There is no monthly-credit limit or requirement to finance the replacement from earlier payouts.
5. Continue forward on the original calendar. Never rewind the run, reuse the failed entry, reset cumulative costs/receipts, or select a different configuration based on subsequent results.

Replacement availability is a research simplification: no evaluation or acquisition delay is modeled. Replacement trading begins only at a subsequent valid decision/opportunity after termination; no repeated births and retries against the same market event. Do not purchase replacements outside the selected run horizon. Record actual account creation times and all costs.

A replacement starts with fresh account balance, loss threshold and payout history. It does not inherit an old position, consumed entry, profit cushion, or pending claim. Preserve legitimately available historical market context without importing future information. Define and test the setup handling at replacement and resumption; do not silently reset the entire market detector every time an account changes.

## What is NOT removed

Unlimited replacement availability does not mean free replacement or unlimited loss inside one funded account.

Keep the owner-defined acquisition costs: $102 for TakeProfitTrader and $125 for MyFundedFutures. Retain the approved firm-specific loss mechanics, profit shares, contract limits, and the approved split trading costs unless the study explicitly declares a separate change. Do not introduce new public-program rules or re-open automation/evaluation research in this task.

Keep the existing confirmed withdrawal policy:

- Qualify only from realized profit after costs, while alive and flat.
- Once at least $500 gross is withdrawable above the retained $2,100 profit cushion, stop further entries on that account for the day.
- At the agreed end-of-day request window, request the entire eligible surplus, not merely $500.
- Stay paused for the agreed two-business-day processing interval. Receipt is at the recorded 4:00 PM Chicago payment time on the second eligible processing day under the frozen holiday calendar.
- Resume the same configuration only after processing and all market/session restrictions permit a new entry.
- Received payouts remain received if the account later fails. Pending claims at the cutoff remain unreceived.

**A payout-processing account is alive. It must not be replaced or supplemented with another account to bypass its pause.** The configuration continues to be measured over those dates, but executes no new trades during its processing lock. Other configuration comparisons proceed independently.

## Full-period financial accounting

For each configuration-and-firm pair:

`net_cash_earned = after_split_payouts_received - initial_and_replacement_acquisition_costs - other_included_external_operating_costs`

Trading costs already deducted inside the account before withdrawal are not deducted a second time from personal cash. The account's simulated trading losses are not another personal $50,000 cash expense. Count all actual funded purchases, including accounts that never pay. Do not reset cumulative results after a failure.

Report total account spending and the largest cumulative cash shortfall before receipts cover costs, even though there is no budget stop. These reveal how much financing a configuration consumed. Do not describe this as a completed five-credit-budget or scalable portfolio simulation.

Use the same selected calendar and data horizon for every comparable configuration. Retain zero-trade and zero-payout dates and processing waits. Do not omit poor configurations because of historical minimum-trade or smoothness filters; distinguish unfavorable financial outcomes from invalid data or incorrect implementation.

## Execution and integration requirements

Extend the normal study configurator and final results screen. Each resolved configuration must actually reach its funded simulation; do not select one profitable child and treat it as the entire study. Persist exact settings and configuration-to-result mappings. Preserve all older runs and label the prior shared-signal pilot accurately.

A configuration-and-firm pair must follow its actual account-driven trading state. Early liquidation, processing pauses, and replacements can change its later opportunities. Reusing a completed reference trade list is not a substitute for that behavior unless equivalence is demonstrated for the exact path. Common immutable market calculations may be shared for efficiency; mutable position, setup and account state may not leak between comparisons. Matching trades across configurations are permitted when their rules independently produce them; do not manufacture diversity.

The unresolved execution-quality findings remain work to resolve: a stop price is not guaranteed when recorded prices cross it, and coverage of the former 156 reference trades does not establish coverage for other configurations or newly generated trade intervals. Use documented execution rules, authorized ordered price data and explicit evidence limits. Do not silently count favorable assumed fills as verified history. Version and compare corrected assumptions rather than overwrite the old pilot.

## Completed-results screen

Present all tested configurations in one final comparison, with separate firm tabs or clear firm columns. Do not add all hypothetical alternatives into a combined portfolio profit.

Lead with configuration, received payouts after the split, total acquisition costs, net cash earned, largest received payout, number of accounts purchased, and accounts lost before/after receiving a payout. Include first-payout time, payout timing by month, maximum unrecovered account spending, pending money at cutoff, and the distinction between missing signals and deliberate payout-processing pauses in the detail view.

Use plain English, readable dollars, full chart-timeframe names and Chicago dates/times with AM/PM. Show the actual strategy settings needed to understand a result, but no internal hashes, developer counters or source paths. Provide cumulative receipts/costs/net cash and account-replacement history for a selected result. One historical result must not be presented as an established future success probability.

## Acceptance and review handoff

Prove through the existing configurator that multiple distinct configurations produce their own results, and that each selected firm gets exactly one current account per configuration. Tests must cover a profitable payout followed by failure, more than five replacements without a credit stop, no expansion after a large receipt, no substitute account during processing, no same-event replacement trade, full versus resumed equivalence, fees, stop/liquidation ordering and zero-payout cases.

Update the maintained specification, task, owner-decision, screen and review instructions to record this scope as superseding the prior budgeted campaign for this mode. Append a readable change to the cumulative research ledger. The agent makes those updates; the owner should not have to manually merge conflicting instruction files.

Publish one compact automatic review folder for the completed study. Consolidate configuration results, settings, account sequences, cash movements, payout events, actual trades, failure-boundary evidence and concise validation results. Retain the question, decisions, rules, readable research history and actual completed review findings. Exclude scripts, patches, raw dumps and repeated per-account folders. Credit/growth tables are not applicable in this mode and must not be exported as though those features were active.

This scope update defines the requested mode; it is not evidence that a new run is complete and does not authorize live trading, new data dates, unrelated strategy changes, or an unrequested sizing sweep. First establish a correct end-to-end multi-configuration comparison, then execute the selected authorized study using that behavior.
