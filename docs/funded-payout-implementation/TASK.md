# Funded configuration comparison — current task

Updated September 23, 2026. The previous task (implement the funded simulator and run one
five-account pilot) is complete and historical; its text is preserved under
`history/2026-09-23T053828-CDT/`. The current scope is `AGENT_SCOPE_UPDATE.md`, reconciled
into `SPEC.md` Part A.

## Read before editing

`SPEC.md` (Part A is the current contract, Part B the deferred budgeted mode),
`OWNER_DECISIONS.md` (do not re-ask settled choices), `TASKS.md` (current state, open items,
active jobs), `CODE_MAP.md` (verified anchors) and the repository's AGENTS.md. Check running
jobs before any launch or edit that could affect a worker.

## The finish line

The owner selects several strategy configurations in the normal IFSM study configurator and
gets one final comparison answering: *which configuration produced the most cash received
after the cost of every funded account used, over the same period?*

1. **Configurator** — the normal study draft stores the selected configuration values and a
   funded comparison section (firms, size, dates, execution model). It saves and reopens.
2. **Resolved variants** — every selected combination becomes its own frozen configuration in
   the plan. None is dropped for low activity or smoothness.
3. **Per-configuration, per-firm simulation** — for each configuration and each selected firm,
   one live funded account at a time follows that configuration's own strategy state. Failures
   are replaced at $102 / $125 with no credit limit. Payout protection, day-end full-surplus
   request and the two-business-day pause apply. Pairs never share money, positions or setups.
4. **Final comparison** — one results screen listing every configuration with separate firm
   columns or tabs, net cash, receipts, costs, largest payout, account count, failures before
   and after a payout, and a detail view with cash history, monthly results, account journeys,
   pending money and processing waits.
5. **Automatic review folder** — one compact folder per completed study, documents and
   necessary data only, no credit/growth tables.

The completed September 22 pilot stays visible and is labeled as the earlier shared-signal,
five-account check. It is not the current financial answer.

## Execution quality

Resolve the disclosed gaps rather than work around them: stop fills follow the declared
stop-market convention on ordered prints; every newly generated trade gets its own price
evidence (ordered prints where authorized local data exist, otherwise a labeled minute
approximation); corrected conventions are versioned against the pilot's. Show failed,
incomplete or unsupported pairs explicitly.

## Completion evidence

- Two distinct configurator settings survive save/reopen, resolution and worker loading and,
  with both firms, produce four separately identified results.
- Synthetic tests: payout then failure, more than five replacements without a credit stop, no
  expansion after a large receipt, no substitute account during processing, no same-event
  replacement trade, full versus resumed equivalence, fees, stop and liquidation ordering,
  zero-payout cases, and per-pair strategy-state isolation.
- The real results screen and the review folder show the same verified numbers; inspect an
  actual rendered screen.
- A fresh read-only review with `funded-payout-reviewer`; fix reproduced blockers and retest.
- The historical comparison runs only for the configurations and size the owner selected or
  confirmed, on the saved January 13 – June 10, 2026 period. No new dates, feeds or sweeps.
- TASKS.md and the cumulative research ledger record actual completed, failed, blocked and
  unverified states.

No live trades, purchases, withdrawals, remote pushes, forced resets or edits to a running
study are authorized. Finish with **Needs your decision; Implemented and tested; Simulation
results; Not verified; Review files**, giving actual saved paths.
