# Source observations — reference evidence, not additional task authority

Basis: `IFVG_Dashboard_Screenshots_20260923.zip`, its capture guide, and the subsequent `IFVG_Dashboard_Review_and_Redesign` review. The original capture covered two apps, 26 workflow groups, and full pages plus duplicate slices. This repair kit deliberately includes only eight targeted screens. Their bytes are unchanged from the original capture.

The screenshot reviewer did not run the applications or inspect the current code. Visible behavior is not proof of its cause. The capture guide reports that drafts were backed up and restored after navigation; it does not establish permanent corruption of the original drafts or economic results.

| Repair | Observation and boundary of evidence | Screenshot |
|---|---|---|
| R1 | A saved variation draft in the pinned/default app shows 48 configurations, zero half exits, and an approval area. The guide states that the compatible research engine resolves it to 64. Investigate loss of saved semantics; do not presume permanent overwrite. | `evidence/01_saved_draft_reopens_as_48.png` |
| R1 (item 9) | In the funded comparison configurator, selected chips show the setting name rather than the value, so both chips in three of the four multiselects read identically. | `evidence/03_chips_show_setting_name.png` |
| R2 | The ranking tab shows MyFundedFutures while the independently controlled lower detail still shows TakeProfitTrader. Labels exist; this is a context inconsistency, not proof of wrong arithmetic. | `evidence/02_firm_selection_mismatch.png` |
| R3 | The dedicated app's Study executions review says no saved searches are available despite the funded comparisons visible elsewhere in that app. It does not establish missing trade data. | `evidence/06_funded_trade_review_empty.png` |
| R4 | Trade-review selectors and axes use New York time, contrary to the owner's chosen Chicago presentation. Another captured view exposes machine timestamps. Fix conversion/presentation, not trade timing. | `evidence/05_trade_review_wrong_timezone.png` |
| R5 | A saved review displays 2050 required independent trading days for a 107-date study. Saved input, parsing, rendering, and plan behavior have not yet been traced. | `evidence/07_impossible_days_threshold.png` |
| R6 | The capture guide says the feature-and-model setup took over five minutes, showing the previous faded screen while loading. No profiler or independently measured timings were supplied. | Guide section A/12; no timing-proof image included |
| R7 | The new Evaluate wizard's default baseline shows "Historical unrestricted holding (legacy)" and a parent retest timeout of "No timeout (unbounded)". The same baseline is the default in every new-study wizard, Trade review and the new context study. | `evidence/08_default_baseline_legacy_holding.png` |
| R8 | The new Evaluate wizard's validation step shows "Permitted research period: 2026-01-13 to 2026-06-10" and one-date-per-line entry. The same limit appears in every study wizard. | `evidence/09_permitted_research_period.png` |

## Original application map to verify locally

- Main app: `streamlit run scripts/dashboard.py`; its IFVG Lab shows search, evaluate, workflow, context and feature-and-model studies, drafts, and Trade review.
- Dedicated app: `python scripts/run_ifsm_research_ui.py`; the default launch uses a pinned engine and shows funded studies and their drafts. The research engine supporting the partial exit is a distinct saved source context.
- Draft name in the capture guide: “Screen check — reopen the completed variation study.” It may not be a unique identifier. Resolve the exact draft from its saved identity and capture context rather than name alone.
- Existing saved search showing the last review step: “Parent staleness — baseline vs 240 Parent stale Timeout (240) and Opp FVG Timeout (60/90) and Parent reaction window and NY morning session and more (copy).” Do not alter this original just to reproduce R5.
- Completed variation result: `5fa65149843484b1`; version 4 was a reporting/evidence revision of the same economic run. It contains 64 configurations and 128 firm-results.
- Earlier one-account comparison: `92f5a08d3b81ae83`; 32 configurations and 64 firm-results.
- Earlier copied five-account pilot: `330142736fcb00fb`; historical, economically different, not a substitute for the current comparison.

Current sources may have changed since the capture. Reconcile actual local evidence and report nonreproduction honestly. The task does not authorize reverting subsequent valid fixes to match these pictures.

## What is deferred even though present in the original review

The repeated 26 feature/model cards, long wizard arrangement, wide tables, stacked notices, empty risk-policy step, unified study library, broader chart redesign, and new analytics remain later work. Only minimal control/label changes needed to correct the R1–R6 behavior are authorized now. TASK.md, not the wider review, defines this pass.
