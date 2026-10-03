# IFVG Lab redesign — fix report

> September 25, 2026: the theme fix that followed the owner's dark-theme report is in `THEME_FIX_REPORT.md` in this folder.
>
> September 25, 2026 (later): superseded in part by the analytical-corrections closeout, `../closeout/FIX_REPORT.md` (F1–F11 and A1–A11). F2's and F3's wording rules and F1's counts are replaced there; F6's "random-draw difference" is checked there (a seed difference within the same method).
>
> September 25, 2026 (evening): follow-up 1 (`../followup-1/FOLLOWUP_REPORT.md`) narrows that explanation. Changing only the seed under the same method reproduces the difference (seed 11), but the reference's seed was not recorded, so the historical cause is not proven.

September 24, 2026. This report covers fixes F1–F11 from the owner's review (`../TASK.md`). The redesign's `DECISIONS_LOG.md`, `FEATURE_MAP.csv` and `REDESIGN_REPORT.md` are updated in place (`docs/ifvg-dashboard-redesign/handoff/`). This task's decisions are FX1–FX16 in that log. Before-and-after screenshots are in `screenshots/`, named by fix.

- No study was launched, re-run, resumed or extended.
- No strategy, engine, account, cost, calendar or payout rule changed.
- No data was downloaded, and nothing dated June 11, 2026 or later was read.
- Every browser check ran on the redesign's isolated copy of the stores: the pinned engine on one port, the half-exit engine on another, and the main application on a third.
- **Saved records:** all 22,619 saved study, draft, approval, review and export files are hash-identical before and after this task (0 changed, 0 added, 0 removed). The isolated copy's drafts and review ledger are also byte-identical before and after the browser checks.

| Fix | Status |
|---|---|
| F1 One honest test result | Fixed and verified |
| F2 Name the payout race | Fixed and verified |
| F3 Actual cash result's standing | Fixed and verified |
| F4 One configuration count | Fixed and verified |
| F5 Point-in-time leaks | Fixed and verified |
| F6 Handoff documents | Fixed and verified |
| F7 Condensed explanations | Fixed and verified |
| F8 Study name in breadcrumbs | Fixed and verified |
| F9 Shared-shell styling | Fixed and verified |
| F10 Wording and number formats | Fixed and verified |
| F11 Two configurators | Fixed and verified (the list is below; the decision stays yours) |

## F1 — One test result on the final code: Fixed and verified

**The one run on the code as delivered:** `ruff check src tests scripts`, then `python -m pytest -q -p no:cacheprovider -rfs`, started September 24, 2026, 11:13 PM Chicago, and finished at 12:04 AM on September 25.

- **Lint:** clean ("All checks passed!").
- **Tests:** **3,793 passed, 5 failed, 3 skipped**, in 50 minutes 23 seconds.
- **The five failures** are exactly the five pre-existing ones the repair task recorded, unchanged: the frozen CatBoost bundle, the HTF cap experiment, and three capture-scheme default-tag tests.
  1. `tests/agents/data_infra/ifvg/test_catboost_bundle_model.py::test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged`
  2. `tests/agents/ifvg_search/test_htf_cap_experiment.py::test_exact_four_effective_profiles_and_distinct_cap_identities`
  3. `tests/agents/test_ifvg_capture_scheme.py::test_default_tags_regression_locked`
  4. `tests/agents/test_ifvg_capture_scheme.py::test_default_times_round_trip_to_canonical_tags`
  5. `tests/agents/test_ifvg_capture_scheme.py::test_list_profiles_default_first_and_upsert`
- **The three skips** need a checkout this computer doesn't have: two tests need the half-exit engine, and one needs the optional pinned IFSM research checkout.

**An earlier run:** a first full run of this task, started at 10:09 PM, gave 3,792 passed, 6 failed and 3 skipped. The sixth failure was `tests/agents/test_ifvg_study_wizard.py::test_session_only_draft_writes_no_file_until_save_or_valid_next`. It still expected an ISO date in a new study's default name, which fix F9 replaces with words. That assertion was updated to the words form, so the run above is the only one on the final code. The earlier run's log is kept with the internal evidence as superseded.

## F2 — Say which payout race the account-risk finding uses: Fixed and verified

**What changed:**
- The Summary's "Early account losses" finding quotes TakeProfitTrader's own-rules race when one has been run for this configuration and firm at the default draw (seed 20260923, 1,000 paths), and says "with TakeProfitTrader's own rules".
- That result is kept only in the application's memory, beside the Risk tab's own result, never in a saved record.
- Otherwise the finding says "with a flat $2,000 limit" and adds one sentence pointing to Risk and simulation.
- The 15% severity rule, and the leader's finding count on Funded results, use the figure shown.
- Nothing runs the firm's rules automatically.

**Verified:** in the running application, the leader at TakeProfitTrader reads "In resampling with a flat $2,000 limit, 22% …" before a run (`F2_after_flat_limit.png`). After "Run with TakeProfitTrader's own rules" it reads "In resampling with TakeProfitTrader's own rules, 42% of fresh accounts hit the loss limit before their first payout.", including after a page reload (`F2_after_firm_rules_after_reload.png`). Four tests cover it.

**Limit:** a server restart forgets the run, so the finding returns to the flat figure until the button is pressed again.

## F3 — Show where the actual cash result ranks under the firm's rules: Fixed and verified

**What changed:** each firm-rules run now keeps every reordered run's net cash. One sentence under the table gives the share below the actual result, with ties counting half and the same 35–65% wording rule. The end-distribution caption now names what it measures.

**Verified:** for the leader at TakeProfitTrader (1,000 runs, seed 20260923), the table reads: "In net cash under TakeProfitTrader's own rules, 82% of the 1,000 reordered runs ended below the actual $30,781.88 (ties count half), so the real order sits on the lucky side of the pile." The caption reads "In trading profit, the actual result (+$37,731) beat 47% of the 20,000 reorderings — it sits right in the middle of the pile, so in trading profit it wasn't a lucky ordering." (`F3_after.png`). Five tests cover it.

## F4 — One draft, one configuration count: Fixed and verified

**What changed:**
- One shared function counts a draft from its saved selections: the strategy variations (half exit only at 1R) × the saved gap rules × the saved withdrawal triggers.
- My studies, New funded comparison (including its read-only view on the pinned engine) and Review and approve all use it.
- A plan with the half exit runs only on the half-exit engine, so it says "on the half-exit engine" on every screen, whichever engine is running.
- My studies draft notes now say when approval is blocked.

**Verified:** "Funded configuration comparison — September 24, 2026" reads "36 configurations on the half-exit engine" on every screen: My studies in the dedicated application (pinned and half-exit engines) and in the main application, New funded comparison on both engines, and Review and approve on both engines. Its My studies note adds "Approval blocked: 2 settings need engine support." Before the fix, My studies said 6 and the pinned engine's setup said 6 (`F4_before_*.png`, `F4_after_*.png`). Three new tests cover it; two existing tests were updated to the new wording.

**The `11b` draft:** handoff screenshot `11b` state 2 is a different draft: the isolated check draft "Funded configuration comparison — 2026-09-24" (id 4dbcb13a…), created at 1:18 AM Chicago during the redesign's checks and set aside afterwards (decision M7). It had no half exit, so it counts 2 entry-hour choices × 2 targets × 2 gap rules × 3 withdrawal triggers = 24 configurations, 48 separate results, and it held the saved 2,050-day check. It was never the 36-configuration draft opened on the pinned engine.

## F5 — Close the remaining point-in-time leaks: Fixed and verified

**What changed (point in time only):**
- The context line leaves out the total.
- The Account picker lists only accounts opened at or before the moment, from each account's saved opening time.
- The Trade picker lists only trades entered at or before the moment.
- "›" is never disabled, so its state can't reveal whether a later trade exists.

**Checked:** every other element in that mode: pickers, help text, captions, chart legend, setup key, "What was recorded", the review form and links. Nothing else is decided after the moment. Full history is unchanged.

**Verified:** at 7:10 PM on the April 12 trade, the page reads "… TakeProfitTrader · Account 6 · trade 77 at this firm", with no "of 114" anywhere (`F5_after.png`). The headless page test confirms the Trade picker lists 77 trades (before: all 114) and the Account picker lists Accounts 1–6. Switching to Full history restores all 114 trades and "trade 77 of 114 at this firm" (`F5_after_full_history.png`).

For this leader, every account (1–6) opened before 7:10 PM, so the account filter shows its effect in the helper test. The browser's trade list renders only the options in view, so counts come from the headless test.

## F6 — Correct the handoff documents: Fixed and verified

**What changed:**
- The four cited screenshots are now in `handoff/screenshots/`, copied from the redesign's internal evidence: `10_setup_new_8637.png`, `11b_review_threshold_pinned.png`, `10_setup_readonly_830868_pinned.png`, `11b_review_blocked_830868_pinned.png`.
- P4b.6 is marked "Replaced by M16". M16's last sentence, which lists every account, is marked replaced by FX6.
- P3.13 now states what it timed: a headless render of the two tabs, 1.88 s and 0.05 s for Payouts, 1.60 s and 0.01 s for Settings. The report's table times the whole page in a browser from navigation, 2.9/0.4 s and 2.8/0.4 s. The report says the same.
- The report now states plainly that all reference values match within tolerance except the 90% range's low end: $133.51 against $126, 6.0% off, a random-draw difference.
- It also names the three values that differ by definition or rounding, all confirmed by your review:
  - 36 of 47 losers, against 35 in the reference.
  - the 9:03 AM bar stamp, against 9:02 AM.
  - the stretch's $10,536.36, shown as +$10,536, against +$10,537.
- F7's "+$10,537" is marked replaced by FX16, so it no longer contradicts P6.2.
- Decisions FX1–FX16 are appended.

**Verified:** a script checks that every `.png` named in the decisions log, report, data gaps and feature map exists in `handoff/screenshots/` (24 files).

## F7 — Put back the explanations that were condensed: Fixed and verified

**What changed:**
- Payouts and accounts has a new "More: what each figure means (the earlier notes, in full)" section at the bottom. It lists each of the nine earlier fact notes word for word, with its earlier value; earlier "CST" times are rewritten in 12-hour words.
- New funded comparison has a "Firm terms, in full" section under the Firms card. It is built from each firm's saved profile, with the same values as the earlier caption: "TakeProfitTrader: $102 per account, 80% trader share, up to 6 minis or equivalent, $2,000 loss allowance, keeps $2,100 after each payout, $500 minimum gross request. Owner-defined simulation terms." MyFundedFutures reads the same way with its own values.
- Feature map rows 31 and 83 now say where the text lives, with status Moved.

**Verified:** both sections open and read in full in the running application (`F7_after_payouts.png`, `F7_after_setup.png`). Two tests compare the notes with the earlier screen's own source.

## F8 — The study name in every breadcrumb: Fixed and verified

**Found:** the generic label came from a page link opened before review fix M12, which made links read the name from the saved run. Handoff screenshot `extra_05c` was captured at 2:25 AM, before that fix; screenshot `05` was recaptured at 3:23 AM, after it. The label couldn't be reproduced by any route on the current code.

**What changed:** the one remaining fallback, a result that no saved run names, now reads "Study name not readable" instead of "Funded comparison". Trade review's Study picker uses the same words.

**Verified in both applications:** every detail tab shows "My studies / Funded variation study / Rank 1 at TakeProfitTrader" on each route:
- rail → My studies → Open results → ranking row → all six tabs
- direct links to each tab, and a reload
- links naming the wrong application, or none
- the half-exit-engine application
- the Trade review back link
- the page after a firm-rules run

Evidence is in `F8_after_rail_route.png` and `F8_after_risk_after_firm_rules_run.png`. Two tests cover it.

## F9 — Finish the shared-shell styling: Fixed and verified

**What changed:**
- The shared stylesheet replaces the framework's default red with the design blue on every IFVG Lab page with the rail. That covers the earlier wizards' progress bars, the chosen radio, checkbox and toggle, sliders, tags, pills, the tab highlight, focus borders, button hover and the calendar.
- Every lab chart gets a design colorway. Before, the fan chart's hidden band edges picked up the framework red.
- New default names from every wizard use the date in words ("Evaluate study — September 24, 2026"); saved names are unchanged.
- The rail's "Other workspaces" link style ships with the rail itself: full names, regular weight.

**Verified:** a script scanned every visible element for the framework red in both applications. Before the fix it found the red progress bar and the chosen radio. After the fix there were 50 scans of page states. The only hit was the fan chart's hidden band edges on Risk and simulation; they were fixed and the page rescanned clean. The pages covered:
- My studies (all five tabs) and New study.
- The earlier wizard (with its advanced section) and an earlier wizard continued from a saved draft.
- An earlier strategy study's results, a model study's results and a funded study.
- Trade review (all sources), the funded overview and all six detail tabs.
- New funded comparison and Review and approve.

The rail looks the same on My studies and the wizard page (`F9_after_rails_my_studies_and_wizard.png`); the progress bar is blue and the name uses words (`F9_after.png`). The main application's other workspaces don't carry the rail and keep their own styling (decision FX9). Three tests cover it; three older name tests were updated to the words form.

## F10 — Wording and number formats: Fixed and verified

**What changed:**
- **Wording:** a documented display map (decision FX11) removes code names from system-written text. It covers limitations, correction descriptions, verification lines, overview and payouts notices, the strategy-measures note, and the read-only views' engine problems and saved-settings table. Owner decision texts and the saved approval's scope stay word for word, and no saved file is edited.
- **Drop-growth row:** the profit row now uses one money format: "−$2.4k · $1.8k · $11.7k", "−$0.7k · $13.3k · $30.0k".
- **Ranking names:** the Configuration column is wider and the number columns are tighter.
- **Date boxes:** they stay "01/13/2026", because the framework's date box can't show words. The resolved range beside them reads "January 13 – June 10, 2026" and "107 trading days, plus 10 warmup days".

**Verified:** measured in the browser at 1,440 pixels, every one of the 64 names at both firms takes at most two lines (`F10_after_ranking_1440.png`). The only code name left on the redesigned screens is inside your own saved decision text on Settings and evidence ("Built in a separate Strategy-Core branch…"), which stays word for word. Five tests cover F10.

## F11 — Two configurators: report, don't remove: Fixed and verified

**What changed:** the earlier configurator stays reachable. One new test, `test_both_approval_paths_accept_and_refuse_the_same_drafts`, runs ten saved drafts through both approval paths without clicking approval:

1. no New funded comparison settings
2. those settings at today's behavior
3. other withdrawal triggers
4. another gap rule
5. other dates
6. a changed check
7. an impossible check
8. the legacy baseline
9. the half exit
10. an unknown firm

It checks that both paths accept and refuse exactly the same drafts. Drafts 1 and 2 are accepted, plus draft 9 on the half-exit engine.

**Verified:** the test passes.

**Options and behaviors that exist only in the earlier configurator:**
1. The plan kind "Configurations from the completed strategy study": pick configurations of the completed study by its approved values per setting. New funded comparison builds only "Variations around one configuration", and drafts of the other kind open on the earlier configurator.
2. Choosing which verified strategy study to compare. The new page always uses the named baseline's study.
3. Any configuration of that study as the base of a variation plan. The new page offers S0_D80_W1_P1, or the legacy baseline, which blocks approval.
4. For the study-configurations kind: the contract (E-mini or micro), contracts per trade and cost per fill for every configuration. The new page has whole-position and half-exit sizes and costs only.
5. The payout processing pause choice, "48 elapsed hours (engineering comparison only; not owner-selected)". The new page keeps the owner-selected two business days.
6. Recording approval on its own ("Record my approval") and launching later ("Run funded comparison"). Review and approve records approval and runs in one step, through the same checks.
7. A plain list of every configuration's full name under "This plan", with the frozen engine caption at the top. Review and approve shows a table and puts the engine under More.

Both paths share the saved-draft blocker list, the launch check, the read-only view for drafts the engine can't represent, and the reset when another window saves.

## Files changed

- **Screens:** `scripts/ifvg_lab_cache.py`, `ifvg_lab_detail_summary.py`, `ifvg_lab_detail_risk.py`, `ifvg_lab_detail_payouts.py`, `ifvg_lab_detail_settings.py`, `ifvg_lab_funded.py`, `ifvg_lab_library.py`, `ifvg_lab_new_funded.py`, `ifvg_lab_trade_review.py`, `ifvg_lab_ui.py`, `ifvg_research_wizard.py`, `ifvg_funded_comparison_study.py`.
- **Presentation modules** (under `src/alpha_lab/agents/data_infra/ifvg/presentation/lab/`): `firm_race.py`, `format.py`, `funded_measures.py`, `funded_setup.py`, `library.py`, `review_panels.py`, `theme.py`.
- **Draft naming:** `src/alpha_lab/agents/data_infra/ifvg/study_drafts.py`.
- **Tests:** new `tests/agents/ifvg_lab/test_review_fixes.py` (29 tests) and one new test in `test_new_funded.py`.
- **Updated expectations:** `test_library.py`, `test_new_funded.py`, `test_trade_review.py`, `tests/agents/ifvg_search/test_study_drafts.py`, `tests/agents/test_ifvg_study_tab.py`, `tests/agents/test_ifvg_study_wizard.py`.
- **No changes to:** the engine, the funded simulator, the account rules or any saved record.
- **Internal evidence** (never in the repository): `../Claude-Quant-Lab-Research-Artifacts/ifvg-redesign-fixes-20260924/` holds the hash snapshots, full-page captures and page text before and after, the red scans, the capture scripts and the test log.
