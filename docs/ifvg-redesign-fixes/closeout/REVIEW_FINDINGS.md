# Independent review of the final task changes

Three read-only reviewers who had not written the code reviewed the task diff against the
recorded pre-task source on September 25, 2026 (about 11:53 AM – 12:07 PM Chicago):

- pre-task tree `811f4b3889a53467a58b5c100f8a0075d65ec4b9`, which already contained the
  uncommitted redesign, fixes and theme work;
- reviewed tree `078d8a953867c9548d2df1523ea122a9cf17579f` (label review1; 40 files).

Each reviewer read the owner's addendum and response, the diff, the changed files and the
evidence, and ran only targeted test files with bytecode and cache writing off. None of them
wrote any file.

- **R1:** analytics and money.
- **R2:** point in time, setup identity, reviewer judgments, the minute record and market
  labels.
- **R3:** the evidence files and the two approval paths.

Every finding is listed below with its disposition. Material findings were fixed and
re-reviewed (section 3), and the final validation ran afterwards on the corrected source.

## 1. Findings and dispositions

| # | Reviewer | Severity | Finding | Disposition | Retest |
|---|---|---|---|---|---|
| 1 | R1 | Material | The interim closeout said the $126 reference could not be reproduced and withdrew the seed explanation. But the reference bounds are whole dollars, and with seed 11 the documented method reproduces all six: $95/$601, $126/$551, $201/$458. | Fixed. The investigation now compares all six bounds per seed; seed 11 is the only one of 200 that matches. §4 of the definitions, decision AC12, the FIX_REPORT and the redesign report's notice now say "a seed difference within the same method". | `interval_investigation.json` rerun; `bootstrap_mean_ranges(values, seed=11)` checked |
| 2 | R1, R3 | Material | Definitions §14 gave the pooled amounts 100 times too small ($238,637.28 − $4,482.90). | Fixed: ($23,863,727.91 − $448,290.00) ÷ 4,395 = $5,327.74. The MyFundedFutures amounts ($26,743,318.86 − $436,000.00) ÷ 3,488 = $7,542.24 were added. The screen and the CSV were already right. | recomputed from `outcomes.csv` |
| 3 | R3 | Material | The "cached identity" in `race_evidence/definition.json` used a script-made digest, not the application's key. | Fixed. Each firm now carries the application's exact key, built with `saved_race_binding` and `firm_race_key`: slots 114, cutoff 2026-06-10T21:00Z, digests `18f938a7…` (TakeProfitTrader) and `bc98ede2…` (MyFundedFutures). The key must also agree with the run's own slots and cutoff. | race evidence rerun |
| 4 | R3 | Material | `regression_evidence.json` compared only fields both sides shared. 27 expected fields had no actual, and the empty mismatch list read as full proof. | Fixed. Every expected field now has an actual field of the same name, and missing fields are reported as problems. Test-asserted-only records sit in a separate, labelled group. JSON is ASCII-escaped, which fixes the "â€“" mojibake a reviewer saw. | `regression_evidence.json` regenerated on the final tree: 56 records recomputed, 0 missing or differing fields; 12 test-only records labelled |
| 5 | R2 | Material | In point in time, choosing another account showed an earlier trade with pickers built from the later trade's cursor. April → Account 1 left Accounts 1–6 and five trades listed at the January 12 cursor. | Fixed. In point in time, choosing another trade or account (or a shared-context account) reruns the page on that trade, so every picker is rebuilt from the displayed trade's own moment (April → Account 1 now lists only "All accounts", "Account 1" and the January 12, 10:31 PM trade) | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a7_page_regression_cases[april_trade_then_account_1]`, `[april_trade_then_shared_account_1]`, `[april_trade_then_an_earlier_trade_from_all_accounts]`; with the rerun disabled the first two fail with exactly the reported symptom; re-review ran 12 extra page probes |
| 6 | R2 | Material | In point in time, choosing a listed account with no trade yet crashed the page (`order.index(None)`). Example: trade 18 at 2:00 PM on January 15, Account 2. | Fixed. In point in time an account is listed only if it is the reviewed trade's own or has a trade entered by the moment; an empty account can no longer be chosen, and a defensive note replaces the picker if it ever happens | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a7_page_regression_cases[trade_18_at_2_pm_lists_no_account_without_a_trade]`, `::test_a7_an_account_is_listed_only_with_a_trade_by_the_moment[...]` (3), `::test_a7_page_an_empty_account_shows_a_note_instead_of_crashing` |
| 7 | R1 | Minor | The Edge verdict counted an unavailable check as a failure: "Weak" or "Not supported" with the deflated ratio missing. | Fixed. Each check now passes, fails or is unavailable. "Partly checked" (neutral) means one passes and one is unavailable; "Weak" means at least one fails; "Not supported" means both fail; "Not available" means neither check is possible. The text names the missing check. | `test_ac_risk.py::test_a11_an_unavailable_edge_check_is_never_counted_as_a_failure`, `::test_a11_edge_text_names_the_check_that_was_not_made` |
| 8 | R1 | Minor | The Risk tab's A/B example was misworded: as written, the two paths compress differently, and it said only "can miss" a loss. | Fixed. Both marked paths are quoted in full, with "can miss or create". | `test_ac_risk.py::test_a3_risk_example_quotes_both_marked_paths_and_says_miss_or_create` |
| 9 | R1 | Minor | The probabilistic and deflated Sharpe wording still read as the probability of an event. | Fixed. The Summary note, the docstring and definitions §5 now call it Φ of a z-score: a normal-approximation confidence score, like one minus a one-sided p-value, comparing the observed daily Sharpe ratio with a benchmark. | `test_ac_risk.py::test_a5_summary_sharpe_tiles_note_and_verdict_wording` |
| 10 | R1 | Minor | The overview's deflated card lacked the prior-research caveat. | Fixed. It now reads "…and it doesn't account for research done before this study." | `test_ac_risk.py::test_saved_summary_and_overview_show_the_corrected_wording` |
| 11 | R1, R3 | Minor | The Summary's conditional finding did not name the source-selection limit. | Fixed. It now reads "reuses the recorded trades (already shaped by the historical accounts' entry selection and skipped opportunities) in fixed slots…". | `test_review_fixes.py::test_f2_firm_rules_race_replaces_the_flat_figure_when_it_exists` |
| 12 | R1, R3 | Minor | In `definition.json`, `horizon.last_trading_day` was the last slot's day (June 5), while the screen says June 10. The population behind the failure median and the percentile rule were not stated. `still_going_share` printed as 5.55e-17. | Fixed. There are now separate horizon and slot-day fields, a `summary_statistics_rules` block, and a counted "neither" share. | race evidence rerun |
| 13 | R1 | Note | The Risk tab and the Summary built their cache keys from two cutoff sources. | Fixed. Both use the saved result's slots and cutoff, and a run whose inputs differ is never cached or quoted. | `test_ac_risk.py::test_a3_ledger_key_uses_the_saved_cutoff_and_refuses_a_mismatch` |
| 14 | R1 | Note | Definitions §7 pointed to the wrong section for early losses. | Fixed: it now points to §12 or §14. | — |
| 15 | R1 | Note | "Account risk: Watch" can fire when the closed-profit drawdown across accounts exceeds one account's allowance. | Kept as the owner's CALCULATIONS.md rule. The wording names the measure ("Worst closed-profit drawdown of the trade path across accounts … (each account's loss allowance …)"), and the leader also trips it through recorded early losses. Recorded as an open point in the FIX_REPORT. | — |
| 16 | R3 | Minor | The market negative control was overstated: only the own-close edit changes D's retrospective label. | Fixed. Definitions §15 now says the own close changes the label, later closes move the whole-study threshold, and appending days leaves D's label unchanged in the test case. | — |
| 17 | R3 | Minor | Identity coverage was thin: one distinct plan on the pinned engine, and refused drafts judged through helper models. AC11 overclaimed. | Fixed. Every refused draft is opened on both real pages (approval controls disabled or absent, no writer, nothing captured); four identity drafts added (one firm only; two E-minis at $4.50; five settings varied; half exit with 4 micros at $0.60): 4 distinct plans compared on the pinned engine, 6 on the half-exit engine, all identical across the paths. AC11 reworded to exactly that | `tests/agents/ifvg_lab/test_ac_approval_paths.py::test_a11_both_paths_approve_record_and_dispatch_the_identical_plan` (pinned 6 passed 1 skipped; half-exit 7 passed); two mutation probes made it fail as intended |
| 18 | R3 | Minor | `approval_matrix.json` was not self-contained, and its plan ids are test-only (FAKE_CORE). | Fixed. The printed matrix carries the engine, `fake_core: true` with its identity and a note that plan and approval ids are test-only, and per accepted case the approval id, scope, statement and normalised dispatch command; `approval_matrix.json` adds the tree, receipts and engines, and `regression_evidence.json` embeds it only from the same tree (checked) | — |
| 19 | R3 | Minor | The approval-test fakes were looser than the real writers (store roots, no-write coverage). | Fixed. Captured plans and approvals are keyed by (store root, plan id); a fake approval refuses a plan saved in another store; `find_approval` is scoped to its store; the no-write check snapshots the whole temporary folder | `tests/agents/ifvg_lab/test_ac_approval_paths.py::test_a11_the_capturing_fakes_respect_the_store_root` |
| 20 | R2 | Minor | In point in time, "›" and "Save and next" at the last trade stayed put, which showed that no later trade exists. | Fixed. At the last trade in point in time, "›" and "Save and next trade" move the clock to the end of the study: the page switches to Full history with the note "There is no later recorded trade for this configuration at this firm: the clock moved to the end of the study, so everything is shown." | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a7_page_regression_cases[last_trade_next_moves_the_clock_to_the_end_of_the_study]`, `[last_trade_save_and_next_moves_the_clock_to_the_end_of_the_study]` |
| 21 | R2 | Minor | Point in time defaulted to the full-history leader configuration. | Fixed for a page opened in point in time (it starts on the plan's first configuration); a page switched from Full history keeps the configuration already on screen, which the definitions now state (re-review m2) | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a7_page_regression_cases[no_link_point_in_time_opens_the_plans_first_configuration]`, `[no_link_full_history_opens_the_ranking_leader]` |
| 22 | R2 | Minor | Gap and parent judgments could be saved against related (borrowed) setup context. | Fixed. With only related context, the gap and parent steps are disabled (and visibly dimmed) with the reason in their help, and nothing is saved for them; entry, stop and exit stay reviewable | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a7_page_regression_cases[related_record_disables_the_formation_steps]`, `::test_a6_disabled_formation_steps_save_nothing_whatever_the_control_holds`, `::test_a8_reviewer_judgments_stay_independent[related_record_disables_the_formation_steps|own_record_keeps_every_step|no_record_keeps_every_step]` |
| 23 | R2 | Minor | Minute record honesty: an unknown approximated-minute count read as zero, and a same-time row with a different result gave "none" instead of "conflict". | Fixed. A missing count is said to be not recorded (also on a conflicting record, re-review m1); a same-times row with an exit or result no trade of the pair recorded is a conflict | `tests/agents/ifvg_lab/test_ac_trade_review.py::test_a10_a_missing_approximated_count_never_reads_as_zero[...]` (5), `::test_a10_a_conflicting_record_never_reads_an_unrecorded_count_as_zero`, `::test_a10_minute_companion_links_only_an_exact_unique_trade[same_times_*]` (3) |
| 24 | R2 | Note | The gap legend showed a "Four-hour gap" swatch with no record, and the related setup chart's hover lacked the qualifier. | Fixed: no gap swatch without a zone; the related setup chart's hover reads "Related opposing gap … (context)" (`test_a6_no_gap_swatch_without_a_zone_and_related_hover_is_qualified`) | — |
| 25 | R3 | Note | Some references did not resolve yet: `../closeout/FIX_REPORT.md` and `validation_summary.json`, a stale "(the xfail test below)" comment, and provenance files with no tree anchor. | The closeout files now exist, checked by the manifest. The comment is fixed. The provenance probe records the git blob of every imported module and whether it equals the named tree: the pre-task run matches tree `811f4b3` for all four modules. | — |

## 2. Checked and found correct by the reviewers

- **Received cash.** Nothing requested but not yet received ever counts as received. The
  failed-account average, open accounts and unresolved requests are reported separately, and
  the pooled ratio makes no mean-of-ratios claim. The failure clock covers first accounts
  that failed before any payout.
- **Recomputed from `outcomes.csv` alone** (2,000 rows), exactly equal to the screen: 81.8%
  and 82.3%, 1.84 and 2.48 payouts, $5,327.74 and $7,542.24, the endpoint shares, the
  medians, and the per-row identities (net cash = received − costs; purchased = failed +
  open).
- **Held-half legs.** Reconciled trade by trade on the saved leader: $4,490.68 + $40,278.18
  = $44,768.86, costs included. Nothing is deducted twice.
- **Benchmark.** One entry instant, one price path, one quantity, no costs: +$50,370, first
  $2,000 fall January 12, 7:41 PM.
- **Entry-known labels.** Only earlier days whose close precedes D's 5:00 PM open are used,
  with a threshold from measures known then. Too little history is its own state. The
  versions are separate, and the reference figures reproduce.
- **Point in time.** The chart uses the scheduled window, and the zone label does not depend
  on the exit. The five-largest note, the evidence before the exit, earlier reviews and
  later loss-limit events are hidden. Moments come only from the configuration's own
  record. The April and January picker options are identical before and after the task
  (pre-task versus final probe).
- **Reviewer judgments.** "Not reviewed" writes nothing, the ledger is append-only, no
  consumer treats the latest review as the current judgment, and the namespaces hold.
- **Minute record.** The folder must name the full result id, and the file's SHA-256 must
  match its manifest. Identity, interval, uniqueness, duplicate and count checks apply. The
  April 13, 2:03 AM link reproduces.
- **Approval test.** It compares the whole plan payload and plan id, the content-hashed
  approval record, and the dispatch (launch check, frozen draft digest, worker command). The
  real buttons are pressed in AppTest; the real `_freeze_and_launch` and `dispatch_problem`
  run on capturing fakes; the home-module writers are forbidden; every root is temporary.
  The F11 test was refactored, not weakened.

## 3. Re-review of the material corrections

A fourth read-only reviewer re-checked the corrections (incremental diff from tree
`078d8a95…` to `e6534b1c…`, about 12:27–12:45 PM). It confirmed five of the six material
corrections with its own recomputation: the seed-11 match and its rarity (seeds matching 0, 1,
2, 4 and 6 of the six bounds: 98, 73, 27, 1 and 1); the pooled amounts; the application cache
keys and the Risk tab's identical key; the complete `recomputed_here` records; and the two
point-in-time fixes (65 of 65 tests plus 12 extra page probes, all landing on picker sets
identical to a fresh link to the displayed trade, with no exception). It found one material
evidence problem and four minor ones, all fixed before the final validation:

| Re-review | Severity | Finding | Disposition |
|---|---|---|---|
| M1 | Material | `regression_evidence.json` embedded a stale approval matrix (it had run before the matrix was regenerated). | Fixed: the evidence script now refuses a matrix from another source tree, records its SHA-256 and the tree, and was re-run after the matrix on the final tree. |
| m1 | Minor | A missing count still read as zero on a conflicting minute record. | Fixed, with `test_a10_a_conflicting_record_never_reads_an_unrecorded_count_as_zero`. |
| m2 | Minor | Switching an open page from Full history to Point in time keeps the leader configuration. | Stated plainly in the definitions (§16): the configuration on screen is kept. |
| m3 | Minor | FX6 unmarked; the closeout report repeated the old "accounts opened by then" rule. | FX6 marked superseded by AC7 and listed in §17; the report now says "accounts with a trade entered by then". |
| m4 | Minor | Regression evidence lacked the review follow-up cases and a tree anchor. | Added (account-listing and evidence-wording cases recomputed; page cases as test-only records); every module blob is checked against the final tree. |
| N1–N8 | Notes | Tolerance not stated; no reason shown when the run key refuses a mismatch; per-result trade key across configurations (pre-existing); forward step is to the next trade's default moment; the unavailable-deflation reason too specific; path counts tried at one seed; ledger status; old report said "replaced". | Tolerance stated in the evidence file; the Risk tab now says why the run is off; the Edge text gives example reasons; §4 says path counts were tried at seed 20260923 only; the old report now says "checked"; the ledger statuses are updated; the trade-key and forward-step notes are recorded as known behaviour. |

After these fixes the source was frozen as tree `97ac9c15…` and the evidence was regenerated on
it.

## 4. Found by the final validation, and its check

The first full-suite run on tree `97ac9c15…` (12:50 – 1:26 PM) failed one test beyond the five
pre-existing ones. The cause was this task's new approval-path test (`FIX_REPORT.md`, section 4,
steps 3 and 4). The test was fixed and the source frozen again as tree `76bb3b0c…`. The evidence
was then regenerated in order, and the full suite rerun on that tree (`validation_summary.json`).
A fifth read-only reviewer, who had not written the fix, checked it (about 1:36 – 1:42 PM). It ran
no tests, because the final suite was running.

| # | Found by | Severity | Finding | Disposition | Retest |
|---|---|---|---|---|---|
| 26 | Full suite, run 1 | Material (test isolation) | `tests/agents/test_funded_payout_workflow.py::test_settings_save_reopen_reach_the_worker_and_publish` failed with "the real …study_drafts.mark_frozen must never run in this test". The approval-path fixture guards the real draft writer. `scripts/ifvg_funded_study.py`, first imported during an approval-path test, bound the guard by name and kept it after that test. | Fixed in the test only. The fixture imports the screens that bind the writer by name before installing its guards, and guards their bindings too, so pytest restores them. The reviewer confirmed the import path, and that all three modules that bind a guarded writer by name at import are covered. No accepted path needs the real writer, so the test is stronger, not weaker. | The two files run together: with the old fixture, 1 failed, 7 passed, 1 skipped (reproduced); with the fix, 8 passed, 1 skipped (pinned) and 9 passed (half-exit). Full suite run 2 (`validation_summary.json`). |
| 27 | Fifth reviewer | Minor | The fix names the modules. A future screen that binds a guarded writer by name, and is first imported during an approval-path test, would bring the leak back. Suggested backstop: at teardown, scan the loaded modules for any guard left behind. | Recorded, not adopted in this closeout. Every such binding today is covered (row 26), and adopting it would need another full validation run. Left as a follow-up. | — |
| 28 | Fifth reviewer | Minor | The same pattern, not leaking today. `test_ac_market.py` temporarily replaces `funded_data.ordered_trades`, which four modules bind by name; this is safe only because they are imported before the patch or never reached. The older `tests/agents/test_funded_trade_review.py` patches `comparison_runner.load_comparison_result`, which the comparison screen binds by name. | Recorded; no current path reaches either. Left as a follow-up. | — |
| 29 | Fifth reviewer | Note | Both new guards carry the writer's home-module label, so the message doesn't say which binding was hit (the traceback does). Writers on paths the approval tests never take are not guarded, but every root is temporary and the whole-folder no-write snapshot would catch a write. | Recorded as known behaviour. | — |
