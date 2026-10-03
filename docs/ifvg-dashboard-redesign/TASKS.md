# Redesign progress ledger

Maintained by the agent. Status values: Not started, In progress, Done and verified, Partly done, Blocked.

- Task created: September 23, 2026
- Repair task status checked: September 24, 2026, 12:05 AM — R1, R2, R3, R4, R5 (validation), R6 (measured), R8 fixed and verified; R7 partly fixed (named baseline for Evaluate; other study types and the Trade review default wait on owner decisions). No phase is blocked by the repair task (docs/ifvg-dashboard-repairs/TASKS.md, OPEN_DECISIONS.md).
- Owner decisions required: none known at creation. The quality-gate time-under-water limit is still the owner's open decision (saved value 3 days; owner weighing 18–20); show it as needing a decision, don't choose it.
- Work started: September 24, 2026, about 12:00 AM Chicago. No Python or Streamlit process was running; no other agent was editing the project (Codex Node runtimes idle since September 23).
- Pre-task integrity snapshot: 22,619 files hashed (both apps' drafts, stores, job states, funded plans/approvals/results, review ledger, published review folders, `research/core`, `pyproject.toml`) — internal evidence folder `../Claude-Quant-Lab-Research-Artifacts/ifvg-dashboard-redesign-20260924/baseline/pre_task_hashes.json`.
- Browser checks run on an isolated copy of every store (`…/ifvg-dashboard-redesign-20260924/isolated/app/`, market data through a read-only junction), never on the real stores.

| Phase | Acceptance | Status | Evidence |
|---|---|---|---|
| P0 | Code and data map done; FEATURE_MAP.csv lists every existing feature with its new home; gap list recorded | Done and verified | Four read-only investigations; `handoff/FEATURE_MAP.csv` (86 earlier features, none pending); gap list below and in `handoff/DATA_GAPS.md`. |
| P1 | Tokens and components built; shared firm state; Chicago time and money helpers; calculation module passes every reference value | Done and verified | `presentation/lab/` + `scripts/ifvg_lab_ui.py`, `ifvg_lab_nav.py`, `ifvg_lab_cache.py`; `test_reference_values.py` 22 passed. |
| P2 | Funded results overview live; firm switch drives table, leader and detail; leader values match reference | Done and verified | Browser: TakeProfitTrader $30,781.88 · 13 · 6 → MyFundedFutures $34,818.11 · 13 · 4, table and leader change together; detail opens on the same firm. Screenshots 02, 02b. |
| P3 | Summary, Payouts and accounts, Settings and evidence live; every existing funded detail panel placed; verdict and findings follow the rules | Done and verified | Screenshots 03, 03b (68% → $203 to $459), 04, 08; every earlier panel placed (FEATURE_MAP). |
| P4 | Trades tab and Trade review live; trade rows open the exact trade; candles, zones or the zones placeholder; point in time hides everything later; reviews save per firm, account and trade | Done and verified | Screenshots 06, 09, 09b; a Trades row opens that trade; April 12 reference geometry exact; saving checked only in the isolated copy and under test temporary stores. |
| P5a | Payout race (fixed floor), fan with method switch, end distribution, streaks, drawdown growth; fixed seed reproducible; "Run again" works | Done and verified | Screenshots 05, 05b; race 78%/22%, 8 · 10; reproducible; Run again announces its seed. |
| P5b | [Historical-order wording narrowed by follow-up 1, AC16: the check compares the five summary figures, the number of trades and each trade's net result and account-loss flag.] Payout race through the existing funded simulator, with expected payouts and cash per account; or fixed-floor version kept with the exact blocker | Done and verified | `lab/firm_race.py` on the public `PairLedger` API; the saved order replays to all 128 saved results exactly; TakeProfitTrader 58% paid first, 1.84 payouts, $5,328 per account (1,000 paths, 8 s); extra screenshot 05c. TakeProfitTrader labeled approximate. |
| P6 | Market conditions live; day counts and per-condition totals match reference; entry/exit day switch works | Done and verified | Screenshot 07; 39/34/18/13/3 days and all card totals to the dollar; switch works (same counts here, stated). |
| P7 | My studies across both stores; setup with baseline, date range, readable chips, new settings; review with blocked state and validation; nothing launches | Done and verified | Screenshots 01, 10, 10b, 11, 11b (two real states); both applications; approval only through the existing gated path (never clicked); drafts reopen byte-identical. |
| P8 | Consistency sweep done; other wizards get the shared shell only; handoff folder complete | Done and verified | Shared rail/tokens on every workspace screen (extra screenshots of the main app); handoff folder complete. |

## Verification summary

- Calculation reference tests: 22 passed. Redesign tests (`tests/agents/ifvg_lab`, final code): 222 passed, 1 skipped (a research-engine-only test on the pinned engine).
- Affected existing tests plus the redesign tests after the review fixes: 532 passed, 2 skipped; `ruff check src tests scripts` clean.
- Full suite on the final code (3:25–4:03 AM, detached): 3,761 passed, 7 failed, 3 skipped. Five failures are the pre-existing ones recorded by the repair task: the frozen CatBoost bundle hash, the HTF cap experiment, and three capture-scheme default-tag tests. Two failures came from this task: `test_ifvg_daily_close_sessions_ui::test_actual_ifsm_screen_save_reopen_edit_and_import` and `test_ifvg_gap_invalidation_ui::test_actual_ifsm_page_save_reopen_edit_and_import` still clicked New study and expected the earlier chooser. Each now clicks "Other study types" first, with the same assertions, and both files pass: 41 passed.
- Independent read-only review (four lenses: money and boundaries, calculations, navigation and preserved behavior, caching and robustness): no blocker. Ten should-fix and eleven minor findings, all fixed with a test that fails without the fix (decisions M8–M19).
- Load time per screen, isolated app: first open on a fresh server 1.8–3.8 s; cached 0.2–0.7 s.
- Integrity: all 22,619 snapshotted files are identical before and after the task, with 0 changed, 0 added and 0 removed. They cover both applications' drafts, stores and job states, the funded plans, approvals and results, the review ledger, the published review folders, `research/core` and `pyproject.toml`.

## Data gaps found

| Gap | Handling | Status |
|---|---|---|
| Setup zones for funded trades | Link via strategy_trade_id, otherwise by the package's own entry-match key and the same stop to a verified configuration's saved record of the same execution (named on screen); otherwise "Setup zones weren't recorded for this study" | Done; placeholder for trades no verified configuration executed |
| Unseen-window results | Placeholder "Not yet run…"; "Plan a confirmation run" disabled | Placeholder showing (out of scope) |
| Payout race full version | Funded simulator on resampled orders (public ledger API, no simulator change) | Done (on request; TakeProfitTrader labeled approximate) |
| Earliest stored date, resolved day count | From repair R8 (`research_period.resolve_research_range`) | Done (December 14, 2021 earliest start; 107 days for the saved dates) |
| Session stability, time-block, best setup share | Not stored by the strategy replay for the funded variation configurations → "Not in export" | Placeholder showing |
| Days with a trade, best day's share (quality gates) | Not stored by the strategy replay for the funded variation configurations → "Not in export" with the funded-trade figure as a labeled reference | Placeholder showing |
| Withdrawal-trigger and gap-rule settings | Controls and draft only; approval blocked, on Review and approve and in the earlier configurator (the funded simulator uses one fixed $500 trigger; the gap rule is not a funded variation axis) | Done (blocked by design until the engine supports them) |
| Confirmation-run planning | Disabled | Placeholder showing (out of scope) |

## Log

- September 24, 12:00–12:30 AM — P0 reconnaissance; pre-task snapshot; isolated copy (robocopy of both apps' stores and the published review folders; `data/databento` junction).
- 12:30–1:30 AM — P1 calculation layer written and tested against every CALCULATIONS.md reference (22 tests). Design tokens, HTML building blocks, the clickable-HTML component (Streamlit components v2), the rail, deep links, the shared firm state.
- 1:30–2:00 AM — P2 overview, the detail shell and the Summary tab; checked in the running dedicated application (isolated copy) with the Chrome browser and a headless-Chrome full-page capture.
- ~2:05 AM — seven parallel builders started (P3 tabs, Trades tab, Trade review, Risk and simulation with the simulator-backed race, Market conditions, My studies, New funded comparison with Review and approve), each owning separate files and its own isolated app port.
- ~2:05–3:25 AM — builders finished (all seven report done and verified); shared fixes merged (nanosecond display, headless-test fallback, icons, mixed-precision times, switches as radios, deep links); side-by-side screenshots captured; load times measured; four-lens read-only review; three fixers plus the lead fixed every finding; docs updated (ARCHITECTURE.md, ML_TRAINING_WORKBENCH.md, docs/README.md, AGENTS.md).
- 3:25 AM — full suite started detached on the final tree (HEAD 1b9d407 + this task's uncommitted files).
- 4:03 AM — full suite finished (above); two navigation tests updated and passing; post-task integrity snapshot identical to the pre-task snapshot. Task complete.
