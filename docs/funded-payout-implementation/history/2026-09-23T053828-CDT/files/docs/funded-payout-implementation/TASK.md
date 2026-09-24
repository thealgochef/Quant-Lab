# Implement the funded-account payout Lab

Implement the funded-account simulator in the existing IFVG Lab, connect it to the normal study workflow, redesign the completed-results screen, and automatically publish the compact review folder. Continue through implementation and verification rather than stopping at a plan.

## Read before editing

Read `SPEC.md`, `OWNER_DECISIONS.md`, `CODE_MAP.md`, and `TASKS.md` in this directory, plus applicable existing instructions in both repositories. `SPEC.md` is the complete unchanged Version 3 specification; this task is not a replacement for its detailed financial rules. Check any older `docs/funded-payout/OWNER_DECISIONS.md` and task record for subsequent owner answers or progress, preserve them, and reconcile only actual dated evidence. Do not make the owner merge files or locate code functions. Record conflicts instead of inventing an approval.

## Deliver the complete outcome

The owner can compare TakeProfitTrader and MyFundedFutures as separate funded-only operations on the same market history. Each has isolated accounts, credits, costs, payout cash and growth. Compare received cash after all modeled costs, including large payouts from accounts that later fail. Present readable cash totals, account journeys and comparisons. Completed runs automatically save the exact documents-and-data-only review folder in the specification.

Inspect the current launcher, actual imports, account simulator, execution path, existing tests and active jobs. Record pre-task baselines and unrelated working changes. Protect the running daily-close/session study; arrange isolated task sources and environment before edits that could affect workers. A clean checkout alone may omit needed uncommitted repairs. Reuse the existing framework, not a parallel demonstration application. Request access to the actual Strategy-Core checkout through normal permissions when needed.

Choose a small coherent integration plan and proceed. Use at most three helpers for independent execution/loss checks, account/payout/budget work, and screen/export work. Give them scoped requirements, source roots and non-overlapping ownership. Agree on shared contracts before parallel edits; verify returned files and tests. Work sequentially when dependencies or unavailable delegation make that safer.

Loss-threshold movement and open-position enforcement are separate controls. Both firms can fail during an open trade. Preserve the realized-eligibility stop, end-of-day gross request and two-day trading pause; requested cash is not received cash. The unresolved processing clock blocks only its definitive economic use, not implementation of both modes, tests, screen or export work. Do not silently turn minute-candle assumptions into an exact live price sequence.

For the screen, use `SCREEN_BRIEF.md` alongside the specification. Preserve the existing application shell. Financial and review figures must come from one verified result, not separate calculations. All bad/no-payout outcomes and material limitations remain visible.

## Completion evidence

- Settings save/reopen, reach the actual worker, change the supported execution path, and retain historical compatibility.
- Funded starts, ordered loss checks, payout protection and processing, isolated budgets, credit carryover, replacement and growth pass positive, boundary, negative and resumed-state tests from the specification.
- The real results screen and automatic review folder use matching verified numbers. Inspect actual rendered screen evidence when available; disclose an unavailable browser rather than claim a visual pass.
- Prepare task-only change evidence against each recorded baseline and use the `funded-payout-reviewer` agent for a fresh read-only check. Fix reproduced blockers and rerun relevant tests. If that agent is unavailable, perform a separately labeled review; do not call self-review independent.
- Once owner-dependent choices, exact pilot settings and price evidence are resolved, complete one bounded authorized same-period funded pilot through the normal workflow. Do not launch a parameter sweep or use protected/new dates. Record an actual blocker if the pilot cannot run; continue other work.
- Update TASKS.md and the persistent research ledger with actual completed, failed, blocked and unverified states. Recheck active jobs before retrying anything.

Ask the owner business questions in the conversation and maintain decision files yourself. Do not send file-creation, text-pasting or settings-merging chores back to them. No live trades, purchases, actual withdrawals, remote pushes, forced resets, new paid feeds or edits to a running study are authorized.

Finish with **Needs your decision; Implemented and tested; Simulation results; Not verified; Review files**. Give actual saved paths. A progress summary, passing synthetic fixture or queued job is not completion.
