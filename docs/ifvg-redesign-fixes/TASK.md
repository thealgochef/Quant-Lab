# IFVG Lab redesign — fixes from the owner's review

## Assignment and finish line

The owner reviewed every file of the redesign handoff (`docs/ifvg-dashboard-redesign/handoff/`, 4 documents and 20 screenshots) on September 24, 2026. This task fixes what that review found. Each fix below says what was observed, what's required, and how it's accepted. Evidence is in `evidence/` and `SOURCE_OBSERVATIONS.md`.

Done means: every fix is Fixed and verified in the running application or files, or recorded as Blocked with the exact blocker; TASKS.md is current; and the handoff in section 5 exists. A plan does not complete this task.

## 1. Boundaries

Everything in the redesign task's section 2 still applies. In short, never:
- launch, re-run, resume or extend a study
- change strategy logic, the engine, account rules, costs, calendars or payout mechanics
- download data, or read anything dated June 11, 2026 or later
- change a saved study, draft, approval, review, export or financial record
- remove an existing feature without a home for it

DECISION_RULES.md and CALCULATIONS.md from the redesign still govern. Browser checks run on an isolated copy of the stores, as in the redesign. Snapshot saved records before and after, and confirm they're identical.

## 2. Fixes

### F1 — One honest test result on the final code

Observed: the terminal summary says "3,761 passed and 7 failed"; `REDESIGN_REPORT.md` says "3,761 passed, 3 skipped" and lists 5 + 2 failures separately. Two tests were edited after that run, so no full run of the final code was reported.

Required: after every other fix in this task is done, run the full test suite once, on the final code, and lint. Report the exact passed, failed and skipped counts from that one run, and name every failing test. Confirm the failures are exactly the five pre-existing ones recorded by the repair task, unchanged.

Acceptance: one set of numbers, from one run, on the code as delivered, appearing identically in the fix report and the final response.

### F2 — Say which payout race the account-risk finding uses

Observed: the Summary finding "Early account losses — In resampling, 22% of fresh accounts hit the loss limit before their first payout" uses the flat-limit race. With TakeProfitTrader's own rules, the same trades give 42% lost before a first payout (58% paid first). The Summary never says which version its number comes from.

Required:
1. When a firm-rules race for this configuration and firm exists at the default seed and path count (in memory or in the lab's own cache — never inside saved study records), the finding uses its "hit the loss limit first" share and says "with <firm>'s own rules".
2. Otherwise the finding keeps the flat-limit figure, says "with a flat $2,000 limit" (or the saved loss allowance), and adds one sentence pointing to Risk and simulation to run the firm's rules.
3. The finding's severity rule (≥ 15%) uses whichever figure is shown.
4. Nothing runs the firm-rules race automatically; it stays on request.

Acceptance: for the leader at TakeProfitTrader, the finding reads 22% with the flat-limit wording before a firm-rules run, and 42% with the firm's-rules wording after one, including after a page reload if the result is cached.

### F3 — Show where the actual cash result ranks under the firm's rules

Observed: after "Run with TakeProfitTrader's own rules", the table shows net cash across reordered runs of $9.8k bad, $23.1k typical and $38.1k good, and "What happened: $30,781.88", but never says where the actual result ranks. Nearby captions say "it wasn't a lucky ordering". That's true for trading profit; under the firm's rules, the actual cash result is well above typical.

Required:
1. Under the firm-rules table, add one sentence: the share of the reordered runs whose net cash was below the actual result (ties count half). Use the same wording rule as the end-distribution caption: between 35% and 65% is "the middle of the pile"; above is "the lucky side"; below is "the unlucky side".
2. Where a caption says the ordering wasn't lucky, make clear it refers to trading profit. For example: "In trading profit, the actual order sits in the middle of the pile."

Acceptance: for the leader at TakeProfitTrader after a firm-rules run, the sentence shows the computed share. It's expected to be on the lucky side, given $30.8k against a $23.1k typical run. The trading-profit caption names what it measures.

### F4 — One draft, one configuration count

Observed: the only draft created during the task, "Funded configuration comparison — September 24, 2026", shows "6 configurations" in My studies in both applications. Its own setup and review pages show 36 configurations (72 separate results). The second state of `11b` shows a 24-configuration draft whose name isn't visible, so it's unclear whether that is the same draft opened on the pinned engine, where the half exit drops out (2 × 2 × 2 × 3 = 24).

Required:
1. One shared function computes a draft's configuration count from the same saved selections the setup page uses, including the withdrawal-trigger and gap-rule settings. My studies, New funded comparison and Review and approve all use it.
2. If the count differs by engine, show the count for the engine the draft was saved with, and say "on the half-exit engine" or "on this engine" where that matters.
3. When approval is blocked, the My studies draft note says so ("Approval blocked: 2 settings need engine support"), not only the engine message.
4. In the fix report, name the draft shown in `11b` state 2 and explain its 24.

Acceptance: the same draft shows the same count, with the same engine wording, on all three screens in both applications.

### F5 — Close the remaining point-in-time leaks

Observed: in point in time, the context line reads "Account 6 · trade 77 of 114 at this firm", so the total tells you how many trades are still to come. The Account picker lists every account the pair ever opens, which reveals future account losses (decision M16 says so).

Required: in point-in-time mode only —
1. The context line leaves out the total ("trade 77 at this firm").
2. The Account picker lists only accounts opened at or before the chosen moment.
3. Check every other visible element in that mode for anything decided after the moment (picker labels, help text, tooltips, captions, links) and hide it the same way.

Full history is unchanged.

Acceptance: at 7:10 PM on the April 12 trade, no count of later trades and no later account appears anywhere on the screen, and switching to Full history restores both.

### F6 — Correct the handoff documents

Observed:
- `DECISIONS_LOG.md` cites four screenshots that aren't in the handoff: `10_setup_new_8637.png` (P7b.1), `11b_review_threshold_pinned.png` (P7b.6), `10_setup_readonly_830868_pinned.png` and `11b_review_blocked_830868_pinned.png` (P7b.9).
- P4b.6 (moment list includes "half out"; default is the half-exit time) contradicts M16 (only moments known before entry; default 7:10 PM) and isn't marked as replaced.
- P3.13 says those tabs open in under 2 seconds, and under 0.1 cached. The report's table says 2.9 and 0.4 (Payouts) and 2.8 and 0.4 (Settings).
- The terminal summary says every number on the leader matches the reference values. Decision F4 records the 90% range's low end at $134 against $126, which is outside the 5% tolerance.

Required: add the four screenshots to `handoff/screenshots/`, or change the references to files that exist. Mark P4b.6 "Replaced by M16". Reconcile P3.13 with the table by stating what each measures. In the report, state plainly that all reference values match within tolerance except the 90% range's low end, a random-draw difference. Append new entries for this task's decisions to the log; don't rewrite earlier entries except to mark them replaced.

Acceptance: every file the log references exists in the handoff, and no two statements in the handoff contradict each other.

### F7 — Put back the explanations that were condensed

Observed: `FEATURE_MAP.csv` row 31 (fact notes: help text on stop slippage, profit inside the account and others) is "Condensed" into the row values. Row 83 (firm terms caption) is "Partly moved, rest reachable" only through the earlier configurator. DECISION_RULES.md rule 12 says nothing disappears.

Required: show every earlier fact note in full on Payouts and accounts, as a help icon on its row or under a "More" section. Show the full firm terms on New funded comparison, in the Firms card's help or under More. Update both rows in `FEATURE_MAP.csv` to where they now live.

Acceptance: every earlier fact note and the full firm terms text can be read on the redesigned screens without opening the earlier configurator.

### F8 — The study name in every breadcrumb

Observed: on the Risk and simulation screen in `extra_05c`, the breadcrumb reads "My studies / Funded comparison / Rank 1 at TakeProfitTrader". The same screen in `05` reads "Funded variation study".

Required: find the route that produces the generic label, and make every route show the saved study name. Routes to check: rail, My studies, Open results, a ranking row, a direct link or reload, the Trade review back link, and the page after a firm-rules run. If the name truly can't be read, the breadcrumb says so in words ("Study name not readable") instead of a generic label.

Acceptance: every route to every detail tab of the reference study shows "Funded variation study".

### F9 — Finish the shared-shell styling

Observed:
- On the earlier wizard in the main application, the progress bar is the framework's default red, not a design color.
- The earlier wizards name new studies with an ISO date ("Evaluate study — 2026-09-24"). The new funded comparison uses words ("… — September 24, 2026").
- The rail's "Other workspaces" links render two ways: "Dashboard Comp" (cut off, bold) on My studies, and "Dashboard Compatibility" (regular weight) on the wizard page.

Required: style the earlier wizards' progress bars and any other remaining framework-red element with the design tokens. Give new study names from every wizard the date in words; existing saved names don't change. Render the rail's other-workspace links one way on every page, with full names.

Acceptance: no framework-red element remains on any page reachable from the rail. New default study names use words. The rail looks identical on every page.

### F10 — Wording and number formats

Observed:
- System-written texts shown on screen still contain code names: "(NQ)" in a limitation, and "Strategy-Core" and "pinned Core" in limitations and notices.
- The drop-growth table's profit row mixes formats: "−$731" beside "−$2.4k", and "$30k" beside "$59.5k".
- At a 1440-pixel-wide window, ranking-table configuration names wrap to three lines, making rows about twice as tall as the mock.
- Date boxes on New funded comparison show "01/13/2026".

Required:
1. For system-written texts (limitations, correction descriptions, notices), apply a documented display wording map, for example "(NQ)" removed, "Strategy-Core" → "the strategy engine", "pinned Core" → "the default engine". Owner decision texts stay word for word; saved files are never edited. List the map in the decisions log.
2. Use one money format per table row: in the drop-growth table, all $k with one decimal ("−$0.7k", "$30.0k").
3. Keep configuration names to at most two lines at 1440 pixels, by widening the column, tightening the second line, or both.
4. Leave the date boxes as they are if the framework can't show words. The resolved range beside them already reads in words; confirm that in the report.

Acceptance: no code name appears in system-written text on the redesigned screens. The drop-growth row uses one format. No ranking name exceeds two lines at 1440 pixels.

### F11 — Two configurators: report, don't remove

Observed: the earlier funded configurator is still reachable under More (`FEATURE_MAP.csv` row 75). The independent review already found one way the two approval paths diverged (fixed in M14).

Required: don't remove it; that's the owner's decision. Instead:
1. List every option or behavior that exists only in the earlier configurator.
2. Add one test that runs the same set of drafts through both approval paths and checks they accept and refuse exactly the same ones.
3. Put the list in the fix report, so the owner can decide whether to retire it.

Acceptance: the test exists and passes, and the list is in the report.

## 3. Out of scope

- New features.
- Engine support for the withdrawal-trigger or gap-rule settings.
- Confirmation runs.
- Free-form tags.
- Changing any saved record.
- Anything in the redesign's DATA_GAPS.md that isn't named above.

## 4. Verification

- Reproduce each finding before fixing it, and record how.
- A test for each fix that fails without it, where the fix is testable.
- In the running application, on the isolated copy: a before and after screenshot for every visible fix (F2, F3, F4, F5, F7, F8, F9, F10).
- F1's single full test run and lint, last.
- Saved records hash-identical before and after.

## 5. Handoff — exact contents

In `docs/ifvg-redesign-fixes/handoff/`:
- `FIX_REPORT.md`:
  - per fix: status (Fixed and verified, Blocked, Not started), what changed, and how it was verified, one short paragraph each
  - F1's exact test counts
  - F4's explanation of the `11b` draft
  - F11's list
- `screenshots/`: before and after for each visible fix, named by fix (for example `F5_before.png`, `F5_after.png`).
- The redesign's `DECISIONS_LOG.md`, `FEATURE_MAP.csv` and `REDESIGN_REPORT.md`, updated in place as F6 and F7 require.

## Final response and stopping point

Reply with:
- which fixes are Fixed and verified, and which are blocked and why
- F1's exact test counts
- anything the owner should decide, at most three items
- the handoff folder path

Then stop.
