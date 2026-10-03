# Source observations

From the owner's review on September 24, 2026 of all 24 files in the redesign handoff. Screenshot names refer to `docs/ifvg-dashboard-redesign/handoff/screenshots/`. Evidence crops in `evidence/` are cut from those screenshots and are otherwise unchanged.

| Fix | What the review saw | Where | Evidence |
|---|---|---|---|
| F1 | The terminal summary gave "3,761 passed and 7 failed". The report gives "3,761 passed, 3 skipped", with five pre-existing failures and two edited screen tests listed separately. No run after the two tests were edited is reported. | Terminal summary; `REDESIGN_REPORT.md`, Verification | Text only |
| F2 | Summary finding "Early account losses": "In resampling, 22% of fresh accounts hit the loss limit before their first payout." The firm-rules run for the same configuration shows 58% paid first and 42% lost first. | `03_detail_summary.png`; `extra_05c_risk_payout_race_with_firm_rules.png` | `02_summary_early_losses_flat_22_percent.png`, `03_firm_rules_race_actual_vs_typical.png` |
| F3 | The firm-rules table shows net cash bad $9.8k, typical $23.1k, good $38.1k, and "What happened: $30,781.88", with no ranking sentence. The end-distribution caption says the actual order "wasn't a lucky ordering" (trading profit, 47th percentile). | `extra_05c…png`; `05_detail_risk_and_simulation.png` | `03_firm_rules_race_actual_vs_typical.png` |
| F4 | My studies: "Funded configuration comparison — September 24, 2026 · 6 configurations · both firms" (both apps). Setup: "36 configurations × 2 firms = 72 separate results". Review: 36 and 72. `11b` state 2: an unnamed draft with 24 configurations and 48 results. | `01_my_studies.png`; `extra_main_app_my_studies.png`; `10_…setup.png`; `11_review_and_approve.png`; `11b_…png` | `04a_library_draft_6_configurations.png`, `04b_setup_same_draft_36_configurations.png` |
| F5 | Point in time at 7:10 PM: the context line reads "Account 6 · trade 77 of 114 at this firm". Decision M16 states the Account picker lists all of the pair's accounts. | `09b_trade_review_state_point_in_time.png`; `DECISIONS_LOG.md` M16 | `05_point_in_time_count_and_accounts.png` |
| F6 | The log cites `10_setup_new_8637.png` (P7b.1), `11b_review_threshold_pinned.png` (P7b.6), and `10_setup_readonly_830868_pinned.png` and `11b_review_blocked_830868_pinned.png` (P7b.9); none is in the handoff. P4b.6 and M16 describe different point-in-time moment lists and defaults. P3.13: under 2 s and under 0.1 s cached; the report table: 2.9/0.4 (Payouts), 2.8/0.4 (Settings). F4 in the log: 90% low end $134 against $126. | `DECISIONS_LOG.md`; `REDESIGN_REPORT.md` | Text only |
| F7 | Feature map row 31: "Condensed". Row 83: "Partly moved, rest reachable" (full terms only on the earlier configurator). | `FEATURE_MAP.csv` | Text only |
| F8 | Breadcrumb "My studies / Funded comparison / Rank 1 at TakeProfitTrader" on Risk and simulation, against "Funded variation study" on the same screen in `05`. | `extra_05c…png`; `05_…png` | `08a_breadcrumb_generic_label.png`, `08b_breadcrumb_study_name.png` |
| F9 | Earlier wizard: red progress bar; default name "Evaluate study — 2026-09-24". Rail "Other workspaces": "Dashboard Comp" (bold) on My studies against "Dashboard Compatibility" (regular) on the wizard page. | `extra_main_app_earlier_wizard_in_shared_shell.png`; `extra_main_app_my_studies.png` | `09a_earlier_wizard_red_bar_iso_name.png`, `09b_rail_labels_my_studies.png`, `09c_rail_labels_wizard.png` |
| F10 | "(NQ)", "Strategy-Core" and "pinned Core" in limitation and notice text on Settings and evidence. Drop-growth profit row: "−$2.4k · $1.8k · $11.7k", "−$731 · $13.3k · $30k". Ranking names on three lines. Date boxes "01/13/2026" and "06/10/2026". | `08_…png`; `05_…png`; `02_…png`; `10_…png` | `10a_mixed_number_formats.png`, `10b_ranking_names_wrap_three_lines.png` |
| F11 | Earlier configurator reachable under More; M14 records a divergence between the two approval paths, fixed. | `FEATURE_MAP.csv` row 75; `DECISIONS_LOG.md` M14 | Text only |

## Checked and correct (don't change)

The review confirmed these against the reference values or the task. They need no work.

- **Values:** all leader values on Funded results, Summary, Payouts and accounts, Trades, Market conditions and Trade review, at both firms, including cash per $1 to two decimals.
- **Buy and hold:** the $2,000 breach at 9:03 AM (bar close time) is correct.
- **Excursions:** "36 of 47 losers" is correct.
- **Controls:** the firm switch, the 68/90/95% switch, the shuffle method, point in time hiding the exit and result, the legacy warning, and the blocked review page listing its reasons all work.
- **Firm-rules race:** the saved order replays exactly. Its counts are consistent: 3,395 accounts lost plus 1,000 still open equals 4,395 bought.
- **Timing:** the 8-second firm-rules run time matches the report; the "about 12 seconds" on the button is only an estimate made before running.
