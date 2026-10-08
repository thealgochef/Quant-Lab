# IFSM partial-exit closing-time repair — v01

## Assignment and finish line

Repair the mandatory-close failure preventing C01 and C02 from completing the existing six-configuration, full-range funded comparison. Diagnose the actual failing state, implement the smallest supported correction, verify it, and complete the missing results through the existing workflow. A diagnosis-only report, passing synthetic tests, or a retry that still stops at the same boundary is not a completed repair.

The owner's current instruction is: **“Let's do the repair first.”** This task is the repair portion of the plan, not authorization for the subsequent gamma-filter experiments or data integration. It authorizes necessary scoped Core/Lab implementation, focused verification, and continuation/re-execution needed to complete the original comparison. Preserve the established approval and access mechanisms; use a newly recorded revision for materially changed source/input/calendar behavior rather than recycling an old approval ID.

Do not start another optimization, change a strategy parameter, install the reconstructed MenthorQ v2 dataset, or broaden the study. Do not commit, push, promote an engine, change global pins, or activate live trading. The source and results have **not** been repaired by creating this task kit.

## 1. Reconcile the current state before editing

Read the repository instructions and the existing task/decision records under `docs/ifsm-correct-config-full-range-v01/`, including `TASK.md` and `ENGINE_INTEGRATION_AND_RESUME_v01.md` where present. Continue their ledger; do not overwrite them or create a competing implementation session. Inspect relevant active workers and current imported source paths before changing shared files. Preserve unrelated work.

Use `PARTIAL_EXIT_CLOSE_REPAIR_REFERENCE_v01.json` beside this file as a navigation and exact-value reference. It contains the original result/plan IDs, all 253 evaluation dates, ten warmup dates, saved effective sections, quantities, fees, account rules, and failed-checkpoint facts extracted from the delivered result. Verify these against the original canonical artifacts locally. This JSON is not a substitute for an approved executable plan or current-source verification.

Historical result: `funded_comparison_ec33a39d953f7308_export_v1`. Four whole-position configurations completed; the two partial-exit versions failed. Their reported error is:

```text
ScheduledCloseCoverageError: position crossed mandatory deadline without an executable boundary bar
```

Both checkpoint summaries report **127 completed replay dates including warmup**, **November 28, 2025 last completed**, and **December 1, 2025 first uncompleted**. This is an investigation boundary, not proof that December 1 contains the missing closing observation. A position can carry an unresolved earlier deadline into a subsequent replay date.

The previous delivery names `ifsm_correct_config_full_range_v01_source_review_v06.zip`, 907127 bytes, SHA-256 `d7b806d151dc390ace8ae4d4361ed3c3d9352fc90b21f53e652073cab873181c`. Reuse its source/review evidence and the scoped `C01_resume_failure.log` / `C02_resume_failure.log` material where available. A missing wrapper ZIP is not itself a blocker when the identical source and required records are available locally. Do not ask the owner to locate functions or build exports that the agent can locate itself.

If a newer repair already exists, verify its exact source, tests, and completed results before doing duplicate work. Do not reset working code to an older tree merely to match this historical reference.

## 2. The comparison that must be completed

| ID | Configuration | Entry schedule | Opposing-pattern distance | Exit | Starting size |
|---|---|---|---|---|---|
| C01 | `S1-T1-H14-P1-L-SO` | All permitted market hours | 80 ticks / 20 points | Half at the original target, remainder at entry-price stop or daily close | Ten micros |
| C02 | `S0-T1-H14-P1-L-SO` | Original three windows | 80 ticks / 20 points | Same partial exit | Ten micros |
| C03 | `S0_D80_W1_P1` | Original three windows | 80 ticks / 20 points | Whole-position original target | One mini |
| C04 | `S1_D80_W1_P1` | All permitted market hours | 80 ticks / 20 points | Whole-position original target | One mini |
| C05 | `S0_D160_W1_P1` | Original three windows | 160 ticks / 40 points | Whole-position original target | One mini |
| C06 | `S1_D160_W1_P1` | All permitted market hours | 160 ticks / 40 points | Whole-position original target | One mini |

The full saved effective sections are authoritative, not this table alone. Preserve one-/four-hour starting gaps with their own-timeframe closing invalidation, two selected starting gaps per timeframe, supporting charts including one minute, finite 240/90 processed-one-minute-bar waits, long-only fresh continuation, and the single active setup/position rule. Do not substitute an old document-default or older 107-date research configuration.

Preserve the exact **June 16, 2025–June 10, 2026** evaluation membership, **253 evaluated dates and ten excluded warmup dates**, continuous initialization, selected contracts, roll inclusion, and already-declared missing inputs. The first logical evaluation day begins on the preceding evening. Final cutoff remains **June 10, 2026 at 4:00 PM Chicago**. No January reset, independent monthly restart, added dates, or discarded difficult dates.

Normal mandatory liquidation remains **3:55 PM Chicago**, with the existing intended buffer before a correctly established earlier closing session. No holding past a required close or over a weekend. Overnight trading remains permissible where the saved strategy permits it. A verified correction to a mistaken historical closing schedule is permitted as a versioned correctness repair; choosing a more convenient liquidation time is not.

Keep three distinct chronological streams per configuration: ordinary strategy, TakeProfitTrader, and MyFundedFutures. Funded streams must preserve their own state and account-driven admissions/exits; do not derive them by walking a newly frozen ordinary trade list. One funded account at a time per configuration/firm, paid replacement after failure, no copying, monthly credits, expansion or reinvestment. Retain all saved prices/proxy disclosures, per-filled-contract fees, account floors and equality rules, payout protection, requests, receipt dates, and processing calendar.

**Keep the current study's passive MenthorQ binding unchanged for this repair:** `menthorq_eod_v1`, regime gate off, nearest-support gate off, unknown allowed, saved hours and universe. Version 2 integration and corrected annotation analysis are separate later work. An immutable old binding remains part of the reproduction; it is not an endorsement of the old date mapping. Do not require the v2 package to complete the closing repair.

## 3. Prove the cause before choosing the fix

For the first actual failure of each partial-exit configuration, recover:

- The failing stream, exact configuration and source identity, selected contract, setup/trade/account identity, entry, prior fills, partial quantity, remaining quantity, protective stop, and applicable account state.
- The last good checkpoint and complete strategy/account state needed to reach the boundary. Distinguish a successful checkpoint write from a state that is actually valid under the corrected policy.
- The deadline being enforced, how it was derived, the date/session it belongs to, exact replay cursor, and whether the prior day already ended with an overdue position.
- Narrow adjacent source observations: bar open/close times, first/last print, completion/finalization flags, order of relevant prices/fills, and source references. Do not infer intrabar order from four candle prices.
- The relevant stack/call site and a concrete before-fix reproduction on the delivered failing source or its faithful isolated copy.

Distinguish missing authorized data, wrong historical schedule, an actual interruption, shortened-bar finalization, checkpoint/day-boundary handling, and a partial-exit integration error. Do not presume any of them merely from the date.

If the evidence points to trading hours or a trading interruption, verify the affected **historical NQ session** using official exchange schedules/notices and the existing data receipts. A holiday calendar and actual feed/trading availability are different evidence. Public schedule verification is allowed; downloading new price history or refreshing current prop-firm terms is not. Record old and corrected calendar/finalization values and their evidence.

Use existing guarded, already authorized inputs. No broad recursive reading under raw-data directories, June 11, 2026 onward observations, the 2021–2022 block, unrelated model data, or new vendor data purchases. Do not import future MenthorQ records from the reconstruction archive as part of this repair.

## 4. Repair the actual defect, not the financial outcome

Necessary task-scoped changes in Core, Lab, calendar interpretation, boundary finalization or checkpoint handling are authorized when supported by the reproduction. Prefer the existing design over a parallel execution engine. Work in an isolated/task-owned runtime and preserve the original source and result versions.

Do not suppress the error, clear an open position without a fill, substitute a whole-position exit, drop the failed date, push liquidation to Monday, or manufacture a boundary candle. Do not pick the last price before a missing interval retrospectively and call that a causally executable closing rule. Preserve distinctions between observed prices and already-permitted approximations.

If an essential historical observation is genuinely absent and the existing execution policy cannot close correctly, report the exact missing evidence and smallest additional modeling/input decision needed. Do not select that new assumption for the owner or label the blocked candidates complete. Engineering work and the supported evidence package should still be delivered.

## 5. Focused verification on final source

Build a regression from the real failure where possible; the new test must fail for the demonstrated reason before the fix and pass afterward. Add only the supporting fixtures needed to cover the changed path. Check:

1. Ordinary and both funded streams, with an open ten-micro position and a five-micro remainder as relevant; the remainder still occupies the position slot.
2. Before/at/after the actual closing deadline, ordinary and shortened sessions, required Friday/weekend closure, and a genuinely missing boundary that must still raise a useful error.
3. Partial exit, stop and forced-close ordering, quantity reconciliation, and fees charged once to actual fills. Preserve the already-declared distinction between ordinary candle resolution and funded ordered-price execution.
4. Day chaining and checkpoint/resume with partial-position and account state, compared with uninterrupted processing from the same valid state. Do not manufacture a flat or fresh account at resume.
5. Worker/save/reopen/approval propagation of the actual source, effective configuration, calendar/input versions, size and costs. Do not accept a correct display name as proof of worker identity.
6. Passive context cannot alter trade or account decisions on affected fixtures.

Run lint, affected tests and relevant neighboring regressions after final functional edits. Follow repository requirements for a broader suite when execution dependencies justify it, but do not repeat unrelated full suites for report wording. Record exact commands, collected/pass/fail/skip counts, compared fields, and source identity; old failures must be evidenced as old, not simply described that way. Zero collected tests is not a pass.

Obtain one focused read-only review of the actual patch and regression evidence. Fix material findings and retest changed paths. Do not commission repeated broad reviews or screenshot matrices of unaffected accepted features.

## 6. Complete the two workers and preserve comparison integrity

Freeze the corrected runtime and plan/calendar/input revisions before the completion runs. Record the repair authority through the existing supported workflow. Do not bypass validation or reuse an approval whose bound inputs no longer match. No further generic permission is needed for the same bounded comparison described here; a genuinely new behavior or new data requirement is different.

Choose the least redundant valid execution route:

- **Verified compatible checkpoint:** continue from it with the complete prior state and history. Demonstrate it precedes any affected state transition; the November 28 checkpoint is not automatically safe.
- **Checkpoint invalid under the fix:** replay from the earliest proven safe point, or from the original warmup if necessary. Preserve the original continuous economic path; never present a fragment as full-range results.
- **Completed C03–C06 cannot be affected:** reuse their immutable results only with a recorded dependency/comparability check and exact source lineage.
- **Shared correction can affect a completed configuration, or non-impact cannot be demonstrated:** run the minimum necessary same-input reference replays from these same six configurations. This is authorized comparability validation, not a new search. Report every resulting difference; do not force the old cash totals to match.

Do not silently combine materially different execution/calendar policies into one six-way ranking. Full-range reruns of all four completed references are not the automatic starting point. Conversely, similar final profit is not sufficient proof of equivalence.

Complete both C01 and C02 through cutoff for ordinary and funded streams. Legitimately failed funded accounts are normal economic outcomes, not technical failure of the configuration. Publish a new immutable aggregate/result revision in the existing Lab, retaining original failures and linking reused versus rerun results. Verify that six strategy results and twelve separate firm results are present and the UI reopens the correct configuration/firm data.

Do not require any minimum payout, trade count or quality-gate pass. This task succeeds through correctness and completion, not a favorable strategy result. Preserve the owner's unresolved three-day time-under-water preference without tuning or using it to conceal completed rows.

## 7. Deliver a compact, independently reviewable closeout

Use the existing full-result export schema. Include the complete actual strategy and funded trades, fills/partial exits, accounts, cash and payout events, daily activity/no-entry intervals, all six configuration bindings, exact date/calendar/input/source identities, validation and accounting tables, QUESTION.md and cumulative research ledger. Preserve full-range and year/month distinctions; do not allocate pooled payouts to gamma buckets.

Add only this repair evidence, extending existing validation/manifest files rather than duplicating them:

| File | Required evidence |
|---|---|
| `repair/REPAIR_REPORT.md` | Question, reproduced cause, precise fix, effect on execution/old results, completion status, remaining limits |
| `repair/failure_boundaries.json` | Both configurations' offending states/streams, deadlines, narrow source observations, reproduction and fixed resolution; real gaps remain explicit |
| `repair/comparability.json` | Per configuration/stream: checkpoint origin or fresh replay, old/new identities, compared fields, observed differences, and why any reused row is valid |
| Existing `validation_summary.json` | Final commands/counts, new failing-before/passing-after regression, final-source checks, resume checks, independent review and precise verification limits |
| Existing manifest and research ledger | Every delivered payload and lineage; earlier task/result entries remain intact |

The previously noted four approximated minutes are a secondary evidence gap, not an established money defect. If their retained minute-level evidence is directly available, include a small `repair/approximated_minutes.csv` naming actual usage, affected minute, cause and whether ordering could change the result. Do not rerun the year, alter its fills, or hold this closing repair open solely to expand that separate audit. If unrecoverable cheaply, list it as a remaining limitation.

Include two or three readable actual screenshots only: completed comparison with correct firm switching and one repaired boundary trade where the existing review can show the evidence. Do not invent a chart from another configuration's setup record; distinguish missing setup provenance from verified fills. Screenshots complement the records, never substitute for them.

Supply a separate small **source-review ZIP** containing task-only patch(es), the affected before/after files and relevant tests/fixtures, required dependency/source bindings, an execution guide for the focused checks, final test receipts and review findings. Include the needed original integration evidence or the already-existing small integration ZIP, without redundant repositories, environments, caches, full-year price dumps, or whole console transcripts. State honestly if the source subset requires the existing project to execute.

Extract and verify both final ZIPs: all manifest members, sizes/hashes, no extra junk, valid relative links. Keep the result ZIP for data/evidence and the source ZIP for code/test review.

## Completion decision

**Complete** means the demonstrated defect is corrected and verified, both partial-exit configurations finish their intended full histories, the six-way aggregate is comparable and reopens in the existing application, all money/activity records reconcile, and compact result plus source evidence is delivered.

**Blocked** means an identified missing observation or new modeling decision genuinely prevents a valid close; include the exact evidence and smallest owner question. It does not mean missing permission for scoped Core edits already authorized above.

Stop after this closeout. Do not start gamma/time filters, the v2 context integration, new entry/exit experiments, a dashboard redesign, a Quantower port, or the old trade-frequency investigation.

## Evidence basis of this assignment

Historical anchors were read from the delivered `funded_comparison_ec33a39d953f7308_export_v1.zip`, especially its configuration bindings, settings, run context, account rules and integration report, together with the independent `FEEDBACK_FOR_AGENT.md` and prior `ENGINE_INTEGRATION_AND_RESUME_v01.md`. Full input/member hashes and exact saved sections are in the companion reference JSON. The implementation instructions above are the new repair-only task; they are not statements that the repair has already been executed.
