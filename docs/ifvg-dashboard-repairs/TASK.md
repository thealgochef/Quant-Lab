# IFVG Lab — repair existing behavior before redesign

## Assignment and finish line

Implement and verify a focused repair of the existing IFVG Lab dashboard. Investigate the current code, reproduce each supported issue safely, fix it, and provide evidence from the actual application. A diagnosis-only report, screenshots without working controls, or passing unit tests without checking the application does not complete this task.

The owner has NOT authorized the broader dashboard redesign or new analytical features in this task. Finish the repairs below and stop. The underlying research question remains: which configuration produces the most received cash after all funded-account acquisition costs? Do not alter that objective or the existing financial experiment to make the screens easier to implement.

The attached screenshot observations establish visible behavior, not its root cause. Record each issue as Reproduced, Fixed and verified, Not reproduced with evidence, Blocked, or Not verified. Do not manufacture a defect or mark an untested repair complete.

## 1. Working boundaries and preservation

Read the latest local instructions and owner decisions, including the maintained `docs/funded-payout-implementation/` records when present. Preserve those documents and their history. Record this as a separate repair task; a prior instruction to continue the full redesign is not authority to expand this task.

Before edits, map both application entry points, their registered study stores, their actual Python environments, imported Strategy-Core versions, and the saved-study/read/preview/approval/dispatch paths. The capture guide identifies `scripts/dashboard.py` and `scripts/run_ifsm_research_ui.py`; verify current locations rather than assuming an old code map is current.

Check active processes and task ownership. Do not edit files being used by another implementation agent or running study. Use an isolated task workspace or coordinate a safe handoff. Preserve existing uncommitted work, including the partial-exit research Core. A new worktree from HEAD alone may omit that work: verify the actual source snapshot used for testing.

Before reproduction, preserve task-relevant source and immutable artifact identities in the internal engineering store. Exercise dangerous reopen/save/approval paths on isolated copies of saved drafts and test stores. Never reproduce a destructive write on the owner's live draft and later restore it over possible concurrent edits.

Keep the following unchanged:
- Completed studies, financial records, saved approvals, source bindings, original configuration membership, notes, and review judgments.
- One funded account at a time per configuration and firm, paid replacements, no copying, monthly-credit limits, growth, or shared funds in the current comparison mode. Preserve the earlier budgeted mode as separate historical functionality.
- Approved costs, quantities, firm terms, stop/partial-exit mechanics, payout protection, processing delays, and trading calendars.
- Existing data access restrictions, verification requirements, and the distinction between ordinary strategy execution and actual funded execution. The only authorized change to data access is the backward extension of the permitted research period in R8; June 11, 2026 onward stays protected exactly as now.
- The default application pin and preserved research-engine source. Do not promote, merge, install over, or hot-swap an engine merely to expose a setting.

No new financial studies, full-period replays, parameter/size sweeps, data downloads or purchases, model training, live trading, or source pin promotion. Tests using synthetic inputs and existing authorized artifacts are allowed. Test worker dispatch by capturing/validating its payload or using an isolated test worker; do not submit a production study. Do not commit or push without a separate owner request.

## 2. Required repairs

### R1 — Preserve a saved experiment when its runtime lacks a setting

Observed case: the saved 64-configuration variation draft is presented as 48 configurations with zero partial exits in the default pinned environment. It becomes the intended 64-configuration study under the research engine. A caption discloses the missing exit setting, but the application presents a different plan and approval area.

Find the actual source of the mismatch across deserialization, registry capability filtering, widget initialization, autosave, plan expansion, and approval/dispatch. Do not assume which layer is responsible.

Required behavior:
1. Load and preserve all saved settings, including unsupported or unknown values. Preserve the saved baseline, exit policies, product/quantity/cost choices, dates, child membership, source binding, and any approval tied to that exact identity. A setting absent from current options is NOT an empty selection or permission to use a default.
2. Distinguish “can display the saved result/plan” from “can execute this plan.” A completed result must remain inspectable without running or importing an incompatible strategy to recompute it.
3. In an incompatible environment, show the saved plan faithfully and explain the missing capability in plain English. Block lossy editing, approval, and launch. Example wording, using the actual saved count: “This study contains 64 configurations and requires the version that supports partial exits. Your saved settings have not been changed.” Never display 64 while sending 48 to the worker.
4. Use an existing trusted compatible launcher/worker route when available, without changing the shared/default pin. An explicit compatible route must carry the exact saved plan, not expand it using a different registry. If unavailable, remain read-only and explain the prerequisite. Do not execute arbitrary code paths taken from a saved artifact.
5. Opening, refreshing, switching pages/tabs, or inspecting a draft must not normalize away its settings or generate a replacement approval. Ordinary navigation may preserve non-economic view state separately; it must not mutate the experiment. Test unchanged save behavior as well as read-only navigation.
6. Use the existing clone flow for intentional changes where practical. Any changed experiment needs a separately identified copy, visible field/count differences, and fresh approval. Do not transfer an approval to changed settings. Do not create a new migration workflow merely for this repair.
7. If the complete original plan cannot be recovered from authoritative saved evidence, report that explicitly and block a guessed executable plan. A similar name or nearby completed run is not proof of its identity.
8. Apply the compatibility and identity check at dispatch as well as in the browser, including stale-page or stale-approval cases. Verify the actual worker source and payload agree with the reviewed plan.
9. Every selected value in a multiselect or saved-plan summary must be readable in full without hovering. Observed: in the funded comparison configurator the chips display the setting name instead of the value, so in three of the four multiselects both selected chips read identically ("Opposing↔pare…", "One-minute opp…", "Parent timefram…"). Show the value itself — by wrapping, a full-value list beside the control, or an equivalent local label change. This is a label repair, not a redesign of the configurator.

Acceptance: verify the real saved variation draft on an isolated copy in compatible and incompatible environments, through both applicable entry points. The compatible plan retains all 64 configurations and both-firm membership; the incompatible one remains unchanged and cannot launch a reduced plan. Preserve any genuinely different saved study counts—64 is a regression case, not a hardcoded application constant. Use synthetic negative cases for unknown values, registry changes, stale approvals, and missing source bindings. Record pre/post hashes and resolved-plan membership, not screenshots alone. Every chip or selection summary on the saved variation draft and the funded comparison configurator shows its complete value, and two different selections never display identical text.

### R2 — Keep firm, configuration, and drilldown selection consistent

Observed case: the ranking tab is MyFundedFutures while the detail selector still shows TakeProfitTrader and its $30,781.88 result.

Use one coherent selected-result context across the existing results page. It must distinguish store/study, economic run, export/result revision where relevant, configuration, firm, and any selected account/trade. Use stable identities, not displayed row positions or labels.

Changing the firm or configuration must update the ranking context, cards, charts, account history, payout table, trade list, links, and any selected-result export together. Preserve the same configuration across firms when it exists; otherwise clear incompatible selections visibly. Do not silently retain an account from another firm or default to a more profitable configuration without indicating the change.

Sorting, filtering, paging, refresh, Back, and navigation between results and review must not change the selected identity unexpectedly. Key caches and session widgets to the appropriate result context; test successive studies and separate browser sessions for stale state. Keep explicit multi-firm comparison views only where already supported and clearly labeled. Never sum alternative configurations into a portfolio.

Preserve the existing automatic full-study export. Label a full-study download differently from any existing selected-result download, so its scope cannot be mistaken. No new export product is required.

Acceptance: inspect the completed 64-configuration run with both firms, including a configuration whose amounts differ. Also switch to the previous 32-configuration result and back. Verify all displayed components against the selected stored result to the cent, including after sort/filter/reload. Financial values are read from the result, never hardcoded in presentation code.

### R3 — Connect funded trades to the existing Trade review

Observed case: the dedicated app has completed funded comparisons but Trade review → Study executions says no saved searches are available.

Find and repair the source-discovery and routing gap. Reuse the existing chart/review component and the application's existing registered stores. Do not build another review application or merge the stores.

The existing execution source selector and result-to-review navigation must reach completed funded trades. Carry the exact run, configuration, firm, account, and trade identity. Do not join on timestamp alone or fall back to an ordinary strategy trade with a similar entry.

Display the recorded funded path: actual entry, initial quantity, initial stop, partial fill when present, remaining quantity, changed stop, final exit, and account liquidation where applicable. Retain the one-hour/four-hour gap geometry and supporting-chart context already available. If an overlay or price path is unavailable, show the available trade facts and an honest missing-evidence message. Do not invent within-minute movement or treat prices after exit as profit captured. Reuse authorized local evidence only.

Preserve ordinary strategy review, verified-context review, rejected/nonexecuted setup review, point-in-time versus full-history behavior, and existing reviewer notes. Point-in-time mode must continue to hide later outcomes. Namespace funded annotations so they cannot overwrite another account's or the reference strategy's review. Navigation back to results must restore the correct context.

Acceptance: open an actual partial-exit winner, a partial-then-stop trade if present, an account-failure exit, and a whole-position trade from the completed saved runs. Prove each matches its recorded prices, quantities, timestamps, and exit reason. Also test absent evidence, zero-trade results, and identical entry timestamps in different firm/configuration/account paths. Smoke-test the two existing review sources and preserve existing saved judgments. Connecting existing data/markers is in scope; new excursion analytics or new account-risk charts are not.

### R4 — Use Chicago time consistently without changing any trading time

Observed case: human-facing trade selectors and chart axes use New York time, and some hover/details show raw machine timestamps.

Use a shared, timezone-aware presentation path for America/Chicago with 12-hour time and AM/PM. Apply it across the IFVG study/review surfaces in both apps: selectors, event tables, axes, tooltips, fills, account events, payout times, and human-readable review summaries. Keep the displayed timezone explicit.

Convert instants; do not relabel them. Preserve the canonical stored timestamps, precision, sequence ordering, trading-date labels, chart-bar anchoring, session-window semantics, approval hashes, payout clocks, and daily deadlines. Do not use a fixed offset. Distinguish ambiguous repeated clock hours where necessary. Sort by the original timestamp, not its formatted string. If an input lacks timezone information, follow its documented source convention or flag ambiguity rather than guessing.

Keep machine-readable export timestamps intact. New human-readable fields or screenshots may use Chicago time without rewriting earlier exports. Correct existing human wording that says no overnight holding: crossing midnight within an open session is allowed; crossing the required daily closure or weekend is not.

Acceptance: test winter and summer conversions, a clock-change boundary with synthetic timestamps, midnight and trading-date boundaries, noon/midnight AM/PM formatting, and alignment between a selected event and its chart marker. The underlying epoch values and execution dates must be unchanged. Existing morning settings must retain 7:00 AM–10:30 AM Chicago semantics, not acquire an extra conversion.

### R5 — Diagnose and validate the impossible day threshold

Observed case: the final review of a saved draft shows a minimum of 2,050 independent trading days for a study with 107 evaluated dates.

Trace the displayed label, widget value/type, saved field, resolved plan, validation, and metric population. Determine whether 2050 was saved, concatenated/rendered incorrectly, parsed incorrectly, or introduced by navigation/autosave. Do not guess 20 or 50. Do not claim completed studies were affected without evidence.

If it is a display/parsing defect, fix the demonstrated cause and add exact round-trip tests. If the saved value really is 2050 and the owner's intended replacement is unknown, retain it, show a clear validation explanation, and leave only that value correction awaiting the owner. Continue all independent repairs.

Before allowing a new affected launch, flag impossible requirements against the actual eligible distinct evaluated-date count for that metric, excluding warmup and respecting the saved access policy. Apply the same validation at the plan/worker boundary. Never clamp a saved threshold or substitute a default silently. Preserve existing historical gate outcomes and comparisons.

Do not use this repair to impose new minimum-activity or smoothness requirements on funded-comparison results. Keep mandatory integrity validation separate from optional strategy preferences. Do not change the old three-day recovery gate just to manufacture a passing result.

Acceptance: test legitimate values and boundary values, a value above the applicable date count, missing dates, warmup exclusion, and the observed saved-draft case. A historic draft remains inspectable even when invalid to launch. A required owner decision must identify the exact field and evidence, not ask the owner to restate all their trading rules.

### R6 — Bound the investigation of the reported five-minute loading stall

The capture guide reports the feature-and-model setup screen took over five minutes; the screenshots alone do not prove the duration or cause.

Measure that existing route once before changes and after any scoped fix, separating cold and warm access where practical. Profile local work rather than repeatedly waiting or triggering preparation/training. If an avoidable repeated scan, redundant artifact load, or similar localized defect is reproduced, repair it without changing the page's capabilities or research semantics. Keep caching tied to source/result identity and access permissions; never return another study's data or bypass verification.

If the delay cannot be reproduced, record measurements and say so. If fixing it requires a new storage/index architecture or substantial redesign, record the narrow evidence and defer that work. Do not mark the delay fixed because a spinner appeared. This is not permission to redesign the study library or optimize an unrelated replay engine.

### R7 — New studies must not default to a baseline that breaks the mandatory close

Observed case: every new-study wizard (Evaluate, Compare, Search, Prop feasibility, Strategy across firms, Full workflow), Trade review and the new context study start from "Baseline (fresh entries, static 1R)". Its holding rule is "Historical unrestricted holding (legacy)", which carries positions across the daily close and weekends. That contradicts the owner's strict prop-firm requirement: flat by 3:55 PM Chicago every trading day and no weekend holding. Its parent retest timeout is "No timeout (unbounded)"; in earlier searches that setting left one setup waiting from April 13, 2026 until the data ended, so those configurations placed no trades after April 10. It also differs from the owner's current selected configuration on the gap cap, gap invalidation, parent timeframes, opposing minimum gap, opposing timeout and parent distance.

Required behavior:
1. Do not edit, rename, re-resolve or delete the existing baseline. Completed studies, drafts, approvals and reviewer notes that reference it must resolve exactly as before.
2. Add the owner's current selected configuration as a named baseline: `S0_D80_W1_P1` from the completed daily-close study, including its mandatory 3:55 PM Chicago daily close. Build it from that configuration's saved, verified identity, not by retyping values. If the saved identity cannot be resolved exactly, report it and do not guess.
3. Make that named baseline the default selection for newly created studies in both apps, including the Trade review research-configuration default. Existing drafts keep their saved baseline.
4. When the legacy baseline is selected or opened, show a plain-English warning, for example: "This baseline holds positions across the daily close and weekends and has no retest time limit. It does not follow the mandatory 3:55 PM Chicago close." Do not block inspection of old studies that used it.
5. A new study whose holding rule is anything other than the mandatory daily close must say so on its review step before approval. Never convert it silently.

Acceptance: an existing draft on the legacy baseline reopens with unchanged settings and identical pre/post hashes. A new study in each wizard starts on the named baseline. The named baseline's resolved settings match `S0_D80_W1_P1`'s saved configuration field for field, including holding rule, gap invalidation, gap cap, parent timeframes, opposing distance, opposing minimum gap, parent retest timeout and opposing timeout. The warning appears on the legacy baseline in both apps.

### R8 — Extend the permitted research period backward (owner-authorized)

Observed case: every study wizard shows "Permitted research period: 2026-01-13 to 2026-06-10", rejects evidence dates outside it, and takes dates typed one per line. The funded configurator takes its dates from a completed strategy study, so a multi-year funded comparison needs a multi-year strategy study first — which the wizards currently refuse. The owner's next research step is a run over roughly two to three years of stored history.

Owner authorization (September 23, 2026): extend the permitted research window to begin at the earliest date of the locally stored, already-authorized market data. June 11, 2026 onward remains protected exactly as now. This authorization covers the permitted window and the date input only.

Required behavior:
1. Determine the actual earliest usable date from the local authorized data and its verification records, and report it. Do not assume two or three years. Do not download, purchase or request new data.
2. Apply the extended window consistently everywhere the permitted period is enforced — the browser, saved-plan validation and the plan/worker boundary. The June 11, 2026 protection must hold at every one of those layers.
3. Replace one-date-per-line entry with a start date and an end date. Resolve the range to valid trading days using the existing trading calendar, excluding holidays and closed sessions, and show the resolved count plus the first and last dates before saving. Handle warmup days exactly as today.
4. Keep the ability to see an exact resolved date list. Existing saved date lists and their approvals must not change.
5. If part of the older data fails existing verification, exclude it with a visible explanation rather than silently.

Acceptance: a new draft can select a range beginning at the reported earliest date and ending on or before June 10, 2026. A range touching June 11, 2026 or later is refused in the browser and at the worker boundary. The resolved trading-day count matches the calendar. An existing saved study's dates and approval hash are unchanged. Do not launch a study to test this.

## 3. Explicitly deferred

Do NOT begin the unified study-browser redesign, reorganize the 26-card block, merge storage, rename all studies, replace the framework, change theme/navigation, rebuild result tabs, or add new graphs, summary scores, market-condition analysis, correlation, Monte Carlo, payout-goal solvers, or new strategy experiments.

The named baseline in R7 exposes an already-completed configuration as a starting point; it is not a new strategy experiment. Allow only the local control/label changes necessary for the repairs—for example, showing the full saved value, a blocking compatibility message, or synchronizing an existing selector. Preserve the broader visual and analytical backlog for the next task. Likewise, do not reopen already corrected financial findings or change the accepted micro-price/cost assumptions.

## 4. Verification and continued work

Maintain `docs/ifvg-dashboard-repairs/TASKS.md` with observed root causes, touched components, current jobs, actual commands/results, evidence paths, and unresolved dependencies. Add a brief cross-reference to the current project task ledger if appropriate; do not overwrite earlier decisions or ask the owner to merge files manually.

For every fix, add a regression that fails on the reproduced behavior and passes after repair. Run relevant existing regression suites. Report exact passed/failed/skipped counts; distinguish newly run tests, reused evidence, test fixtures, and browser observations. Record a known pre-existing failure with a baseline reproduction where available, not a blanket “all passed.” Do not relax tests merely to achieve green output.

Verify both application entry points using isolated stores and actual rendered pages. Prefer existing browser/testing tools. Capture readable real screenshots and their interaction sequence. An exported chart, source-code assertion, or empty mock screen is not proof that the actual funded screen works. If browser verification is unavailable, say Not verified and supply the exact blocker; do not claim complete acceptance. Continue nonblocked work.

Read all current result summaries in the relevant completed comparison fixtures to verify selection membership (64 configurations / 128 firm-results for the variation run; 32 / 64 for the earlier comparison). Browser-check representative paths, not 128 redundant screenshots. Keep historical five-account-operation results separate and unmodified.

Prove immutable economic records, configuration manifests, source bindings, and approvals remain unchanged. Include checks before and after opening/refreshing/navigating and isolated ordinary-save tests. Separate allowed test-copy edits and reviewer notes from immutable financial payloads. Do not rewrite history merely to change display timezone or repair a selector.

Use a fresh read-only reviewer of the task-specific changes against the recorded pre-task state, not blindly against main. Provide the reviewer actual diffs and test evidence internally. Fix demonstrated in-scope findings, retest, and save the review's actual findings, not its instruction template. Independent helpers may investigate nonoverlapping areas; do not let them edit shared files concurrently without ownership coordination.

If a newly found problem would change executed trades or cash, preserve the existing results and report the affected behavior and smallest proposed next step. This task does not authorize a corrected economic rerun. An unavailable engine, source, or owner choice should block only dependent items, not the entire repair effort.

## 5. Compact handoff — exact contents

Create one new repair-review folder and matching ZIP under `reports/ifvg_dashboard_repairs/<repair_id>/` or the project's equivalent established report root. This is a software-repair handoff, not a new funded economic run. Export only:

- `README.md`: scope, source applications, completion status, how to read the evidence.
- `REPAIR_REPORT.md`: R1–R8; reproduced behavior, established cause, smallest fix, remaining limitation, and whether any stored economic record changed.
- `REVIEW_FINDINGS.md`: actual final read-only review, fixes/retests, and remaining issues.
- `validation_summary.json`: test commands/counts and actual results; both app/runtime contexts; browser availability; source snapshot references; no new financial studies launched.
- `study_integrity_checks.csv`: draft/result identity, application/runtime context, saved/resolved counts, approval/payload checks, pre/post digests, and pass/fail. Unknown is not pass.
- `interaction_checks.csv`: repair ID, exact selected study/configuration/firm/account/trade context where applicable, action, expected result, observed result, status, screenshot/test reference. Include clock, threshold, baseline and date-range cases.
- `performance_checks.csv`: measured route, cold/warm condition, duration, measurement basis, changes and limits; no invented timing when not measured.
- `SCREEN_EVIDENCE.md`: indexed before/after or interaction-sequence screenshots, interpretation, viewports, and precisely what each proves.
- `screenshots/`: only the readable actual screens needed for the repairs, normally about 10–15 captures. Include both runtime outcomes for the saved draft, readable chips, both firm selections, funded trade review, Chicago labels, threshold validation, the new-study default baseline with the legacy-baseline warning, and the date-range selector with the June 11 refusal. Explain missing captures; do not replace them with analytical charts.
- `manifest.json`: exact final file inventory, byte sizes, and hashes for every payload except itself, with an explicit counting convention.

Keep scripts, patches, source snapshots, raw market/trade dumps, full stores, long console logs, caches, and environments in the internal engineering store, not this ZIP. Do not include all earlier screenshot slices, review bundles, or the redesign backlog. Verify the final archive itself for links, inventory, and integrity before handing it over.

## Final response and stopping point

Return: (1) repairs completed with evidence, (2) items not reproduced or not verified, (3) confirmation of preserved studies/approvals/economic results and unchanged engine pins, (4) actual test/browser/performance outcomes, (5) the compact review folder and ZIP, and (6) any exact owner decision still necessary.

Use plain English for the owner. Do not call this a redesign, a new profitable run, or validation of future payouts. Stop after the repair handoff; do not proceed into new dashboard features or new trading research without another request.
