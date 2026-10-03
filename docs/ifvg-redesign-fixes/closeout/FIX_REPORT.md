# IFVG Lab — corrections closeout: F1–F11 and A1–A11

September 25, 2026 (Chicago). This closeout answers `../AGENT_RESPONSE_AND_EVIDENCE_REQUEST.md`
and completes `../ANALYTICAL_CORRECTIONS_ADDENDUM.md` inside the same fixes task. It is separate
from the earlier export `../handoff.zip`, which is unchanged. Definitions:
`CALCULATION_DEFINITIONS.md` (this folder). Every file listed in `manifest.json` is in this
folder or its ZIP.

## 1. Scope

**The addendum was not incorporated before.** It was not in the fixes package (its manifest
lists 17 files, dated September 24), and neither the F1–F11 report (1:48 AM) nor the theme
report (2:35 AM) nor the task ledger mentions A1–A11. Its copy on this computer dates from
10:46 AM, September 25. It is now reconciled into the same task: `../TASKS.md` maps every item.
The completed F1–F11 and theme work was kept. Where the addendum conflicts with F2, F3,
`CALCULATIONS.md` or `DECISION_RULES.md`, the addendum wins; the older text is marked
superseded, not rewritten (definitions §17; decisions AC1–AC13 in the redesign's
`DECISIONS_LOG.md`).

**Boundaries kept.**
- No study was launched, re-run or extended, and nothing was approved.
- No account, exit, payout, cost or calendar rule changed, and no engine or funded-simulator
  code was touched.
- No new raw data was read, nothing dated June 11, 2026 or later was opened, and nothing was
  committed or pushed.
- The saved three-day time-under-water limit is unchanged.
- The earlier configurator stays reachable with its approvals.
- Browser checks ran on the redesign's isolated store copy (preservation:
  `preservation_and_access_checks.json`).

## 2. Status of every item

Status words: **Fixed** (implemented and verified), **Qualified/relabelled** (kept, renamed and
limited; not replaced by an exact model), **Deferred** (needs work this task may not do),
**Not verified** (not checked here).

| Item | Status | What is true now | Source (function) | Tests / receipts | Screens |
|---|---|---|---|---|---|
| F1 One honest test result | Fixed | The final lint and full-suite run on the delivered source (§4, step 6). It supersedes the 3,793/5/3 receipt, the theme fix's 3,846/5/3 and this task's own first run (§4, step 3) | — | `validation_summary.json` | — |
| F2 Name the payout race | Qualified/relabelled | The finding names what it quotes: "Early losses in the fixed-boundary diagnostic" before a ledger run, "Early account failures (conditional model)" after one; 15% rule on the figure shown; never runs by itself | `funded_measures.findings`, `ifvg_lab_cache.early_loss_race`, `pair_findings` | `test_review_fixes.py::test_f2_*` | S01, S10 |
| F3 Actual result's standing | Qualified/relabelled | Descriptive percentile with ties half, model and horizon; the lucky/middle/unlucky rule and 35–65% thresholds are removed | `ifvg_lab_detail_risk.firm_standing`, `end_caption`; `resampling.share_below_ties_half` | `test_review_fixes.py::test_f3_*`; `test_ac_risk.py::test_a5_*` | S03, S09 |
| F4 One configuration count | Fixed | One shared count on My studies, New funded comparison and Review and approve; now also asserted equal to the effective plan's membership | `funded_setup.plan_count` | `test_review_fixes.py::test_f4_*`; `test_ac_approval_paths.py::test_a11_the_shared_count_follows_the_saved_membership` | earlier export F4_* (reused) |
| F5 Point-in-time leaks | Fixed (extended by A7) | No later account, trade total or trade; April 12 at 7:10 PM lists 77 trades and Accounts 1–6 | `review_panels.known_accounts`, `known_trades`, `context_line` | `test_review_fixes.py::test_f5_*`; `review_and_provenance_checks.json` | S07 |
| F6 Handoff documents | Qualified/relabelled | The four cited captures exist in the redesign handoff (hashes in `review_and_provenance_checks.json`), but this closeout doesn't rely on them. The $133.51 against $126 explanation, which had been asserted without a check, is now checked: it is a seed difference within the same method (seed 11 reproduces all six reference bounds to the whole dollar; definitions §4) | `funded_measures.bootstrap_mean_ranges` | seed check summarized in `CALCULATION_DEFINITIONS.md` §4 (internal record `interval_investigation.json`) | — |
| F7 Condensed explanations | Fixed (reused claim) | Unchanged by this task | `ifvg_lab_detail_payouts`, `funded_setup` | `test_review_fixes.py::test_f7_*` (full suite) | earlier export F7_* |
| F8 Study name in breadcrumbs | Fixed (reused claim) | Unchanged | `ifvg_lab_nav` | `test_review_fixes.py::test_f8_*` (full suite) | all S screens show "Funded variation study" |
| F9 Shared-shell styling | Fixed (reused claim) | Unchanged; the theme fix follows the app's light or dark theme | `ifvg_lab_ui`, `theme` | `test_review_fixes.py::test_f9_*`, `test_theme.py` (full suite) | earlier export F9_*, T1_* |
| F10 Wording and formats | Fixed (reused claim; wording extended) | Unchanged formats; the drop table row is renamed (A2) | `format`, risk tab | `test_review_fixes.py::test_f10_*` | S03 |
| F11 Two configurators | Fixed, strengthened | The earlier configurator is kept. On ten drafts, both paths accept and refuse the same drafts; for the accepted ones they approve, record and dispatch the identical plan (plan id, membership, approval record, dispatch) on both engines | `ifvg_lab_new_funded`, `ifvg_funded_comparison_study` | `test_ac_approval_paths.py::test_a11_both_paths_approve_record_and_dispatch_the_identical_plan`; matrix in `regression_evidence.json` | — |
| A1 Market labels | Fixed | Retrospective labels kept (numbers unchanged, named `retrospective_daily_close_v1`); new entry-known labels (`entry_known_prior_closes_v1`) from closes completed before each day's open; a Labels switch | `market.condition_labels`, `entry_known_days`, `entry_known_labels` | `test_ac_market.py` (16); future-label invariance in `regression_evidence.json` | S04, S05 |
| A2 Drawdown is not failure | Qualified/relabelled | "Sampled closed-profit drawdown from a previous high"; the 94% is not a failure share; the early-withdrawal claim is removed; the +$2,600/−$2,000 race is a fixed closed-profit boundary diagnostic | `resampling.drawdown_growth`, `payout_race`; risk tab | `test_ac_risk.py::test_a2_*` ($0→+$6,000→+$3,500; $0→−$2,100→+$500) | S02, S03 |
| A3 Conditional firm-ledger race | Qualified/relabelled | "Conditional resampling of recorded trades with <firm>'s ledger rules" (`conditional_firm_ledger_resampling_v1`); four limitations in the title area, Summary finding, cache key and export; both firms conditional; A/B counterexample; parity kept separate. No full-path, regenerated-strategy model was built (that would need new research) | `firm_race` | `test_ac_risk.py::test_a3_*`; `race_evidence/` | S09, S10 |
| A4 Populations and clocks | Fixed | Average payouts among accounts that failed within the horizon (6,240 ÷ 3,395 = 1.84), open accounts and unresolved requests beside it; pooled net cash per purchased account ($5,328) with no mean-of-ratios claim; separate eligibility, request and receipt clocks | `firm_race.path_outcome`, `race_from_outcomes`; risk tab | `test_ac_risk.py::test_a4_*`; `race_evidence/outcomes.csv` | S09 |
| A5 Statistical wording | Fixed | No luck classification; "with replacement" everywhere; probabilistic and deflated Sharpe explained by series and trials; Edge shows "Not supported" when both checks fail; low R² stated as an observation | `resampling`, `funded_measures.verdict`, summary and overview | `test_ac_risk.py::test_a5_*` | S01, S03 |
| A6 Setup identity | Qualified/relabelled | The leader has no own setup record; the matched record of configuration S1_D80_W1_P1 is labelled "Related context …; exact setup identity for this configuration is not established"; it no longer names review steps or point-in-time moments, and the gap and parent judgments are disabled for it; fills unchanged | `setup_records.SetupRecord.identity_established`; review cards | `test_ac_trade_review.py::test_a6_*` | S06–S08 |
| A7 Early-history point in time | Fixed | January 12, 10:31 PM: only Account 1 and one trade at the 10:35 PM cursor; April distinction kept (7:07 PM entry, 7:10 PM cursor, half at 7:10:27.251840803 PM hidden); future records can't change the hidden view (chart range from the schedule); pickers rebuild from the displayed trade's own moment; accounts listed only with a trade by then; forward steps at the last trade move the clock to the end of the study | `review_panels`, `review_chart.chart_window` | `test_ac_trade_review.py::test_a7_*`; `review_and_provenance_checks.json` | S06, S07 |
| A8 Reviewer judgments | Fixed (verified; no behaviour change needed) | A mixed old judgment (entry correct, stop incorrect) survives notes-only and unchanged saves; unknown stays unknown; namespaces kept | `review_panels.review_verdicts`; append-only ledger | `test_ac_trade_review.py::test_a8_*`; isolated UI reproduction | S11a, S11b |
| A9 Held-half attribution | Fixed | Legs recomputed from quantities and per-fill costs: first halves $4,490.68, remainders $40,278.18, whole $44,768.86; the finding quotes the remainders; nothing deducted from cash | `funded_measures.held_to_deadline_legs` | `test_ac_risk.py::test_a9_*` | S01 |
| A10 Benchmark clock and minute record | Fixed | Buy and hold from one entry instant (January 12, 5:00 PM open): +$50,370, first $2,000 fall January 12, 7:41 PM. The April trade links the published minute April 13, 2:03 AM (export v4, hash-checked, identity, interval, uniqueness and count checks) | `market.buy_and_hold`; `minute_companion.link_from_review_folder` | `test_reference_values.py::test_index_tie_and_buy_and_hold`; `test_ac_trade_review.py::test_a10_*` | S01, S08 |
| A11 Research meaning | Fixed | Unavailable ≠ fail; original gates and plans kept; "data points to 18–20" replaced by a neutral pending decision; left-out half exits explained before approval; ten-case matrix with identity | `funded_setup.gate_rows`; `ifvg_lab_new_funded` | `test_ac_approval_paths.py` | — |

## 3. Details that carry over from the earlier handoff

These points are copied here, not just referenced, so this report stands alone.

- **F4, the `11b` draft.** Handoff screenshot `11b` state 2 showed an isolated check draft, not
  the 36-configuration draft: "Funded configuration comparison — 2026-09-24" (id 4dbcb13a…),
  created at 1:18 AM on September 24 during the redesign's checks. It had no half exit, so it
  counted 2 entry-hour choices × 2 targets × 2 gap rules × 3 withdrawal triggers = 24
  configurations and 48 results, with a saved 2,050-day check. It was set aside afterwards.
- **F11, what only the earlier configurator offers** (the owner decides separately whether to
  retire it; nothing is retired here):
  1. The plan kind "Configurations from the completed strategy study".
  2. Choosing which verified strategy study to compare.
  3. Any completed configuration as the base of a variation plan.
  4. Per-configuration contract, quantity and cost.
  5. The 48-elapsed-hour processing pause (an engineering comparison, not the owner's
     setting).
  6. Recording approval without launching.
  7. A plain list of every configuration with the frozen engine caption.

  Both paths share the saved-draft blockers, the launch check, the read-only view for drafts
  the engine can't represent, and the reset when another window saves.
- **Theme fix (September 25, 2:35 AM).** The lab follows the application's light or dark theme,
  both apps open with the rail expanded, and the collapsed rail's reopen button is readable on
  both themes. Its receipt was 3,846 passed, 5 failed (the five pre-existing failures) and
  3 skipped; this closeout's run supersedes that receipt.
- **Point-in-time rule (M16/FX6).** Only moments known before the entry are offered. The
  default is the first five-minute mark after the entry. Pickers list only accounts with a
  trade entered by then (A7 tightened the earlier "opened by then" rule) and trades entered by
  then, and there is no trade total.

## 4. Tests, in the order they ran

All times are Chicago time on September 25, 2026. The complete record is
`validation_summary.json` (`test_sequence`); each step names the source it ran on.

1. **Targeted runs while working (11:07 AM – 12:50 PM).** Each change was followed by its
   affected test files. The integrated targeted runs (the lab tests plus the neighbouring
   funded tests) went from 441 passed and 3 skipped at 11:45 AM to 474 passed and 3 skipped at
   12:50 PM, with ruff clean each time. The three skips need the half-exit engine (step 2).
2. **The half-exit engine.** The default environment imports the pinned engine, which has no
   half-exit setting. In the half-exit application's environment the two half-exit tests that
   the default run skips passed (2 passed). The approval-path matrix passed on both engines
   (pinned: 6 passed, 1 skipped, the half-exit-only case; half-exit: 7 passed).
3. **Full suite, run 1 (12:50 PM – 1:26 PM, tree `97ac9c15…`).** Ruff was clean; pytest gave
   3,958 passed, 6 failed and 4 skipped. Five failures were the pre-existing ones. The sixth,
   `tests/agents/test_funded_payout_workflow.py::test_settings_save_reopen_reach_the_worker_and_publish`
   ("the real …study_drafts.mark_frozen must never run in this test"), came from this task's new
   approval-path test, not from application code. That test's fixture replaces the real draft
   writer with a guard for the length of each test. The earlier configurator's screen module
   (`scripts/ifvg_funded_study.py`) imports the writer by name when it is first loaded. In the
   full run it was first loaded during an approval-path test, so it kept the guard, and the
   later workflow test reached it.
4. **The fix, in the test only.** The fixture now imports the two screens that bind the writer
   by name before it installs its guards, and guards their own bindings too, so pytest restores
   them after each test (7 added lines in `tests/agents/ifvg_lab/test_ac_approval_paths.py`).
   With the old fixture, the two files run together failed the same way (1 failed, 7 passed,
   1 skipped). With the fix they gave 8 passed and 1 skipped on the pinned engine, and 9 passed
   on the half-exit engine. An independent read-only reviewer checked the diagnosis and the fix
   (`REVIEW_FINDINGS.md`, section 4).
5. **Re-freeze and evidence (1:32 – 1:34 PM).** The source was frozen again as tree
   `76bb3b0c…`, which differs from `97ac9c15…` only in that test file. On it, in
   this order: the approval matrix on both engines (6 passed, 1 skipped; 7 passed), the
   half-exit run (2 passed), `regression_evidence.json` (56 records recomputed, 0 problems), the
   race evidence and the provenance probe (identical to run 1's apart from its tree and time).
6. **Final validation, run 2 (1:34 PM – 2:12 PM, tree
   `76bb3b0c514b0d1fba2fcfec17beb1694943b6c7` at start and at end).**
   - `python -m ruff check src tests scripts`: all checks passed (exit 0).
   - `python -m pytest -q -p no:cacheprovider -rfs`: **3,959 passed, 5 failed, 4 skipped** in
     2,300.27 seconds; exit 1 because of the failures below.

   The five failures are the pre-existing ones listed in AGENTS.md. Each fails with the same
   assertion as in the fixes handoff's run, the theme fix's run and run 1:
   - `tests/agents/data_infra/ifvg/test_catboost_bundle_model.py::test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged`
   - `tests/agents/ifvg_search/test_htf_cap_experiment.py::test_exact_four_effective_profiles_and_distinct_cap_identities`
   - `tests/agents/test_ifvg_capture_scheme.py::test_default_tags_regression_locked`
   - `tests/agents/test_ifvg_capture_scheme.py::test_default_times_round_trip_to_canonical_tags`
   - `tests/agents/test_ifvg_capture_scheme.py::test_list_profiles_default_first_and_upsert`

   The four skipped tests:
   - `tests/agents/ifvg_lab/test_ac_approval_paths.py::test_a11_review_explains_the_left_out_half_exits_before_approval`: "needs the half-exit engine: on the pinned engine the half-exit study is read-only and can't be approved"; passed on the half-exit engine (the approval matrix run, 7 passed). This task added it, which is why there are four skips where the theme-fix receipt had three.
   - `tests/agents/ifvg_lab/test_new_funded.py::test_research_engine_opens_the_half_exit_draft_editable_and_reviews_all_64`: "needs the research engine with the half exit"; passed in the half-exit application's environment (the targeted run: 2 passed).
   - `tests/agents/ifvg_search/test_ifsm_replication_values.py::test_pinned_repaired_runtime_registry_supports_exact_research_choices`: "Optional pinned IFSM research checkout is not present on this host"; needs the optional pinned IFSM checkout, which is not on this computer; not run.
   - `tests/agents/test_funded_comparison_saved_drafts.py::test_research_engine_rebuilds_all_64_configurations_and_both_firms`: "needs the research engine with the half exit"; passed in the half-exit application's environment (the targeted run: 2 passed).
7. **After the final validation**, only documentation and closeout files changed (the list is
   in `validation_summary.json`, `changes_after_the_final_tree`). No source or test file changed
   after the suite.

## 5. Independent review

Three read-only reviewers who hadn't written the code reviewed the task diff against the
pre-task source (about 11:53 AM – 12:07 PM). R1 covered analytics and money. R2 covered point in
time, setup identity, reviewer judgments, the minute record and the market labels. R3 covered
the evidence files and the two approval paths. They reported 25 findings: 6 material, 14 minor
and 5 notes, all fixed or stated. The material ones:

- the $126 seed explanation had been withdrawn wrongly;
- the pooled amounts in the definitions were 100 times too small;
- the race evidence used a script-made key instead of the application's;
- the regression evidence didn't compare every expected field;
- two point-in-time faults: pickers built from another trade's moment, and a crash on an
  account with no trade yet.

A fourth reviewer then re-checked the material corrections (about 12:27 – 12:45 PM) and
confirmed five of the six with its own recomputation. The sixth, a stale matrix inside the
regression evidence, was fixed before the final validation, as were four minor points.

The first full-suite run then found the test-isolation fault in step 3 of section 4. A fifth
read-only reviewer checked its diagnosis and fix. It confirmed the import path, and that the fix
covers every module that binds a guarded writer by name today. It also found the test stronger,
not weaker, because no accepted path needs the real writer. It found no leak in the other three
new test files. It suggested a teardown check that would also catch a future module binding a
guard, and named two patches that follow the same pattern without leaking today. These are
recorded as follow-ups, not adopted here (`REVIEW_FINDINGS.md`, section 4).

Every finding, its disposition and its retest is in `REVIEW_FINDINGS.md`.

## 6. Open items and limits

- **Engine work, not done here.** A regenerated full-strategy, full-path firm model was not
  built. Doing it would need new research (a separate proposal), and the conditional model's
  limits are shown instead. The leader's own setup geometry isn't saved: saving each
  variant's setup records would need the funded comparison to store them.
- **Measured before, not re-measured here.** The earlier feature-and-model first-load delay
  (about 4 minutes, repair R6) and the 356 saved children listed as unavailable under the
  pinned engine remain open. This task didn't measure or diagnose that route.
- **Owner decisions still open.**
  - The time-under-water limit (saved 3; the screen proposes no value).
  - Whether to retire the earlier configurator.
- **Trade review.** A replacement account appears in the account picker only once it has a
  trade by the moment, although "What was recorded" can already name its opening (known by
  then). Moving the clock forward past the last trade shows everything, by design.
- **Account risk "Watch".** It still also fires when the closed-profit drawdown across accounts
  exceeds one account's loss allowance (the owner's rule from CALCULATIONS.md); the wording
  names the measure, and the leader also trips it through recorded early losses.
- **The reference ranges' seed** was never recorded; seed 11 reproduces them to the whole dollar, and the application keeps its fixed seed 20260923.
- **Not re-captured.** F7–F10 screens were not re-captured in this closeout (reused claims;
  their tests ran in the full suite).
- **Test hardening follow-ups.** The fifth reviewer's two minor points are recorded in
  `REVIEW_FINDINGS.md`, rows 27 and 28, and were not adopted here: a teardown check for
  leftover guards, and two patches that follow the same pattern without leaking today.
- **The all-128-pair historical-order parity** has no saved per-pair receipt. It is reported
  by the existing test in the final full-suite run, and the two leader pairs' exact parity is
  in `race_evidence/definition.json`.

## 7. What is in this closeout

See `manifest.json`: this report, `CALCULATION_DEFINITIONS.md`, `validation_summary.json`,
`regression_evidence.json`, `review_and_provenance_checks.json`, `race_evidence/definition.json`,
`race_evidence/outcomes.csv`, `preservation_and_access_checks.json`, `REVIEW_FINDINGS.md`,
`SCREEN_EVIDENCE.md` and `screenshots/`. The source-review attachment is a separate ZIP
(`../closeout_source_review.zip`). Scripts, logs, full stores and raw data stay in the internal
engineering folder `../../../../Claude-Quant-Lab-Research-Artifacts/ifvg-analytical-corrections-20260925/`.
