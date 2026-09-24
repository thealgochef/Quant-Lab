# Update the task instructions, then continue the funded comparison

Package revision: single-account-per-configuration continuation, September 23, 2026.
Status: instructions only. Copying this package does not change application code, prove implementation, or complete a new simulation.

## Deliverable and governing scope

Use the full `AGENT_SCOPE_UPDATE.md` in this directory as the owner's latest change to the funded research mode. The objective is to compare received cash after every account cost for each strategy configuration over the same market period. A large payout followed by account failure is a valid economic contribution. Account survival, trade count, or frequent small payments are not substitute objectives.

The existing configurator must generate the requested strategy variations; each variation gets its own one-account-at-a-time sequence for each selected firm. Failures are replaced with fresh funded accounts under the same configuration and every purchase is charged. There is no credit allowance, cash-affordability stop, expansion, account rotation, or copied group in this mode. Processing pauses still block that sequence and cannot be bypassed with another account. Two selected firms mean two separate results per variation, never an additive portfolio total.

Preserve the prior operating-budget mode and its completed results separately. This is an extension of the current implementation, not permission to delete working functionality or reclassify historical results as the new mode.

## 1. Inspect and preserve the actual project

Read the applicable root/project instructions, the four task-specific Claude helpers, and the maintained files under `docs/funded-payout-implementation/`. Inspect actual imports, the current branch and uncommitted work, and running jobs. Do not reset a checkout, repoint a worker, or edit read-only archived runtime packages. A needed Strategy-Core integration may be developed in an authorized editable task checkout; the old decision to keep Core read-only was an engineering choice, not a permanent owner prohibition.

Before changing any existing task document or helper, copy its current bytes to a uniquely identified historical snapshot under `docs/funded-payout-implementation/history/<actual-update-time>/`. Record original relative paths and hashes. Preserve a consistent relative-path tree in that snapshot. Never keep backup `.md` rules beneath `.claude/rules/`, where they could remain active instructions. Do not fabricate earlier creation dates or overwrite an existing snapshot.

This package itself overwrites no original kit file. Inspect the user's actual local content rather than reconstructing it from the original empty checklist. Preserve amendments that are unrelated to this scope change.

## 2. Reconcile active documents; do not leave two competing requirements

Make the following task-scoped edits. Existing files that are absent can be created after checking for an equivalent maintained file. The owner does not need to copy, append, or merge text.

| File | Required update |
|---|---|
| `SPEC.md` | Make the independent configuration comparison the current research contract. Keep funded costs, loss rules, payout protection/processing, correct clocks, price-evidence requirements and data-only export. Clearly separate the deferred budgeted operation. |
| `TASK.md` | Point to the current scope and define configurator -> resolved variants -> actual per-configuration/per-firm simulation -> final comparison -> automatic review folder. Remove the single-reference-only pilot as the current finish line. |
| `TASKS.md` | Preserve the completed pilot and its evidence as historical. Add a new, initially open acceptance section for this continuation; do not reset history or mark new work done based on old tests. |
| `OWNER_DECISIONS.md` | Preserve the original decisions with their dates. Add the latest scope change and identify exactly which budget/allocation decisions are deferred for this mode. Keep the two-business-day processing clock and settled monetary rules; do not ask them again. |
| `SCREEN_BRIEF.md` | Show all selected configurations and separate firm results. Remove credit/growth widgets from this mode; retain them only in the separate budgeted-mode view. |
| `REVIEW.md` | Require current configurator coverage, one-account sequences, correct state/price-path execution, costs, payouts and final screen/export agreement. It remains review instructions, not completed review findings. |
| `CODE_MAP.md` | Add verified current locations and integration seams, clearly distinguishing old inspection anchors from what was just inspected. Do not treat old path names as proof of active imports. |
| `GUIDE_NOTES.md` | Keep useful process guidance; correct stale task links and scope statements. Do not imply this update has been economically tested. |

Update the following existing helpers after snapshotting them:

- `.claude/rules/funded-payout-implementation.md`: short task-scoped guidance that points to the latest scope and reconciled specification. No five-account default in this mode. Keep applicable repository safety rules.
- `.claude/skills/funded-payout-implementation/SKILL.md`: preserve its command name as a compatible entry point; route it to this continuation and the maintained current task rather than restarting the old pilot. Keep explicit/manual invocation, normal permissions, and a small body.
- `.claude/skills/review-funded-payout/SKILL.md`: route to the current review contract. Never execute studies as a review side effect.
- `.claude/agents/funded-payout-reviewer.md`: remain restricted to file reading/searching; evaluate the new contract, not whether the five-credit model is the default. A reviewer unable to run tests reports that limitation.

Do not replace root `CLAUDE.md`, `AGENTS.md`, `.claude/settings.json`, `.claude/settings.local.json`, the user's `.settings`, or unrelated rules, agents and skills. An obsolete task-specific sentence discovered in a root instruction should be reported precisely; do not rewrite an entire root file or bypass repository access controls to resolve it.

Append the scope change to the persistent research ledger. Keep readable history concise; leave bulky historical data in the internal store. Update the current scope state in `TASKS.md` so repeated invocation continues the work without making duplicate snapshots, duplicate ledger events, or duplicate runs. A rerun must not silently overwrite local progress from a newer session.

## 3. Continue implementation, using the working financial engine

Extend the existing study configurator and its normal worker/result paths. The owner must be able to select multiple variations in one study and inspect the funded outcomes for every variation. Do not build another disconnected dashboard or merely select one profitable prior child.

A result identity must distinguish configuration, firm, dates, execution model and the single-account comparison mode from the historical copied-account/budget mode. Each pair has at most one live funded account, including payout-paused accounts. It owns its actual setup/position state, acquired-account history, costs, payouts and timer. Independent pairs may share immutable price/detector work, not mutable positions or money.

Unlimited replacement is an availability assumption, not free accounts or the right to trade the same failed event again. A replacement starts fresh after the failed account is resolved, takes future legal opportunities under the same configuration, and retains the accumulated comparison costs/received cash. Do not add an acquisition after the simulation cutoff. Preserve correctly warmed market context without inheriting an old position, consumed signal, cushion or payout history.

Retain the approved $102/$125 acquisition costs; $2,000 firm-specific loss rules; 80%/90% trader shares; six/three-mini caps; $500 gross minimum; $2,100 retained cushion; stop after realized eligibility; request at day end; two-business-day pause and 4:00 PM Chicago receipt using the frozen processing calendar. Fees, threshold equality assumptions and mandatory daily/weekend closure follow the current approved account rules. Do not reinterpret 'no credit limit' as removing firm loss checks or processing waits.

The completed shared-reference pilot is a labeled historical implementation check, not the current financial answer. Resolve the disclosed execution gaps: per-account earlier exits/pauses must affect future strategy opportunities; a stop trigger price is not a guaranteed fill when observations cross it; and the old price check for 156 reference trades does not establish coverage for newly generated paths. Declare and test the execution convention, use authorized ordered prices where available, and show uncertainty where evidence is insufficient. Do not correct payouts by dividing the old copied result or deducting a single aggregate slippage number. Preserve fee accounting without duplicate execution-cost deductions.

Do not silently filter poor or infrequent configurations out using old smoothness/activity preferences. Actual data failures and implementation violations are different from an unfavorable financial outcome. Show failed/incomplete/unsupported pairs explicitly, not fabricated zeros or omitted rows.

## 4. Validate the entire workflow, not just one account's arithmetic

Required engineering evidence includes:

- Two distinct configurator settings survive save/reopen, resolution and actual worker loading as distinct configurations. With both firms they produce four separately identified results; the application supports the user's selected count rather than hard-coding two or thirty-two.
- One current funded account per configuration-and-firm pair, no cross-pair money or trade-state leakage, and no five-copy multiplication.
- More than five losses/replacements without a monthly-credit stop, with every actual purchase charged once. Large payouts create no account expansion.
- A paid account later fails without erasing its receipts. An account in payout processing is not replaced and cannot trade; cutoff-pending money is not received.
- Earlier account liquidation, skipped entries and resumption actually affect later strategy state. Identical independently generated trades are allowed; artificial diversification is not.
- Stop/target/breach/deadline order, fees, equality boundaries, price gaps, original versus resumed equality, duplicate-event prevention, no replayed failed signal, and no changes to old immutable results.
- The final results screen and exported comparisons use the same verified financial values. Inspect an actual rendered screen; templates and test counts alone are not visual evidence.

Use deterministic synthetic fixtures for edge cases. Before historical work, record the exact selected configurations, sizing, source identities and existing authorized date list. Run only the bounded comparison already selected/authorized by the owner through the configurator or a subsequently confirmed plan. If the particular historical selection is genuinely absent, ask for that selection when it is needed; continue all independent implementation and engineering tests. The example '32 configurations' defines expected behavior, not blanket authority to launch an arbitrary new sizing grid or access new dates. A one-configuration pilot alone cannot prove the requested multi-configuration integration.

Preserve the January 13-June 10, 2026 evaluation period, its exact 107 date labels and ten warmup dates whenever that saved study is selected; no protected/new data, feed purchases or live changes. Do not assume one-config path coverage proves full-period coverage for all alternatives. Wait for actual accepted jobs to finish and preserve their real completion/failure receipts.

## 5. Final screen and automatic review folder

Lead with the per-configuration cash question. Present configuration settings in plain English, separate firm columns/tabs, and net cash after all acquisitions, received payouts, largest payout, total costs, account count, and failures before/after a payout. Show selected cash histories, monthly results, replacement journeys, pending money and processing waits. No sum of hypothetical alternatives as portfolio profit. No hidden raw hashes, developer counters, source paths or unexplained shorthand in the business view. Use Chicago dates and AM/PM times.

Publish one compact automatic folder per completed study, not repeated account folders. Include the frozen question, effective settings, funded rules/assumptions, direct results, decisions, concise cumulative ledger, per-configuration results, account journeys, actual trades, cash movements, payouts, meaningful failure/execution evidence, and compact validation outcomes. Include unfavorable/no-payout configurations. Credit and growth tables do not apply in this mode. Actual completed reviewer findings should be distinguishable from the reviewer-instruction template.

No scripts, source patches, environments, raw price dumps, giant nested history lines or redundant data formats in that folder. Keep deeper reproduction material internally. Material omissions must be declared rather than called a complete source-price audit. A package-only rebuild must not trigger a new economic run or overwrite an old result.

## Completion report

Briefly report: which task documents/helpers were reconciled and where their prior versions were preserved; what changed in the application; exact tests and configurator paths exercised; what historical study actually completed; unresolved execution/data limitations; and the final review-folder location. A documentation update is not a completed application change, and a completed engineering test is not a future payout guarantee.

Continue unfinished independent in-scope work instead of stopping after a progress message. Stop for an actual access, data or owner-selection blocker, not because an old checklist says the earlier pilot is Done.
