# Screen evidence

Final-code captures of the running application, taken 12:51 – 12:52 PM on September 25. The
application was started at 12:51 PM on tree `97ac9c158fea6e08c9aee5c29e56310d49565085`. The
final test suite ran on tree `76bb3b0c514b0d1fba2fcfec17beb1694943b6c7`, which differs from it
only in one test file (`tests/agents/ifvg_lab/test_ac_approval_paths.py`, the fixture fix in
`FIX_REPORT.md` section 4). Every file the application runs is therefore byte-identical in the
two trees (`validation_summary.json`).

**Application.** The dedicated IFSM research application (`run_ifsm_research_ui`) runs with
the pinned Strategy-Core engine on the redesign's isolated store copy, at 127.0.0.1:8661,
light theme, file watcher off, restarted on the final code before capturing.

**Study.** Funded variation study `5fa65149…`, configuration "All open-market hours · Long only ·
Half at 1R" (S1-T1-H14-P1-L-SO), TakeProfitTrader.

**How they were taken.** Headless Chrome at 1,440 pixels wide. Each capture is cropped to
the section it demonstrates; the matching page text was kept internally. No approval,
launch or saved-store write happened. The only write was the S11 review, made on the
isolated copy's review ledger, which was restored byte for byte afterwards (SHA-256
`e861127e…` before and after).

These are final captures, not intermediate ones. The earlier F1–F11 before/after set is in
the unchanged earlier export (`../handoff.zip`) and is not repeated here. The four captures
the old decisions log cited (`10_setup_new_8637.png`, `11b_review_threshold_pinned.png`,
`10_setup_readonly_830868_pinned.png`, `11b_review_blocked_830868_pinned.png`) still exist
in the redesign handoff. This closeout doesn't rely on them, so they aren't included.

| File | What it demonstrates | Items |
|---|---|---|
| `screenshots/S01_summary_findings_before_ledger_run.png` | Verdict and findings before any ledger run: "Early losses in the fixed-boundary diagnostic" (22%, −$2,000 / +$2,600, "not TakeProfitTrader's account rules"); "Held halves carry the profit" ($40,278 held halves against $37,731 total; first halves $4,491, whole trades $44,769); "Low daily linear association with the index" (R² 0.11, 107 days, an observation); the "Probabilistic Sharpe ratio" tile and the note explaining its series and trials; the account-risk text naming the closed-profit drawdown; buy and hold +$50,370 "bought at the January 12, 5:00 PM open; first fell $2,000 from its high January 12, 7:41 PM" | A2, A5, A9, A10, F2 |
| `screenshots/S02_risk_fixed_boundary_diagnostic.png` | The explanation that paths draw with replacement; "Fixed closed-profit boundaries: which is crossed first?" labelled a diagnostic (78% / 22%), its caption saying it is not a payout or account-failure model; the ledger section's heading and limitations line shown before any run | A2, A3, A5 |
| `screenshots/S03_risk_closed_profit_drawdown_and_percentile.png` | "In trading profit, the recorded result (+$37,731) is above 47% of the 20,000 resampled paths' ends (ties count half) … it does not measure luck"; "Sampled closed-profit drawdown from a previous high" (94% by trade 100, "not the share of funded accounts that fail", with the lock) | A2, A5, F3 |
| `screenshots/S04_market_retrospective.png` | Market conditions, "Retrospective labels (retrospective_daily_close_v1)", with its hindsight definition (the default view; numbers unchanged) | A1 |
| `screenshots/S05_market_known_at_entry.png` | The same tab after choosing "Known at entry": "Labels known at entry (entry_known_prior_closes_v1)"; trades labelled by entry day only; cards from pre-open closes | A1 |
| `screenshots/S06_review_january_point_in_time.png` | Early-January point in time. The January 12, 10:31 PM trade is shown at 10:35 PM; the context line reads "Account 1 · trade 1 at this firm", with no total; everything after 10:35 PM is hidden; related context is labelled "exact setup identity … not established". The picker option sets (All accounts / Account 1; one trade) are in `review_and_provenance_checks.json` | A6, A7, F5 |
| `screenshots/S07_review_april_point_in_time.png` | April point in time. Entry at 7:07 PM, cursor 7:10:00 PM; the half exit (7:10:27.251840803 PM) and the result are hidden; there is no "of 114"; related-context labels | A6, A7, F5 |
| `screenshots/S08_review_april_full_history_minute_link.png` | The same trade in full history. The price-evidence line links the published minute: "1 approximated from the one-minute bar: April 13, 2:03 AM — …", with the cause, the effect ("can't change the half exit, the final exit or account survival") and the hash-checked source | A10, A6 |
| `screenshots/S09_risk_after_conditional_run.png` | After "Run conditional resampling with TakeProfitTrader's rules": the limitations line; "Average payouts among accounts that failed within the tested horizon" (1.84: 6,240 payouts ÷ 3,395 accounts; 1,000 accounts still open with 4,616 payouts; no request still processing); "Pooled net cash per purchased account" ($5,328 from $23,863,727.91 − $448,290.00 over 4,395 accounts); the two-model table with separate horizons and eligibility, request and receipt clocks; "the recorded net cash ($30,781.88) is above 82% …"; the historical-order check kept separate; the A/B counterexample | A3, A4, A5 |
| `screenshots/S10_summary_findings_after_ledger_run.png` | The Summary after that run: "Early account failures (conditional model)", 42%, naming all four limitations | A3, F2 |
| `screenshots/S11a_review_form_seeded_mixed_judgment.png` | Isolated ledger seeded with an older mixed review (entry correct, stop incorrect). The form preselects nothing; earlier reviews show "Entry correct, Stop incorrect"; the gap and parent steps are switched off because the zones are related context | A8, A6 |
| `screenshots/S11b_review_form_after_notes_only_save.png` | After a notes-only save through the page (steps left "Not reviewed"): the new review shows "—" for steps, and the older "Entry correct, Stop incorrect" review is unchanged. The ledger lines are in `review_and_provenance_checks.json` | A8 |

The market, review and form screens show the corrected layout, but their data comes from the
saved records. No crop shows raw bars beyond the chart candles the app already draws from the
verified package.
