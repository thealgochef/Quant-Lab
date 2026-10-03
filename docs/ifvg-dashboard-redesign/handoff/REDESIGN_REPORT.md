# IFVG Lab redesign — report

> **Corrected in part — September 25, 2026 analytical corrections.** See
> `docs/ifvg-redesign-fixes/closeout/FIX_REPORT.md` and `docs/ifvg-redesign-fixes/closeout/CALCULATION_DEFINITIONS.md`. In particular the "random-draw
> difference" explanation of the 90% range's low end is now checked: it is a seed difference
> within the same method (seed 11 reproduces all six reference bounds to the whole dollar), buy and hold is now +$50,370 from one entry instant, the payout races are
> renamed (a closed-profit diagnostic and a conditional model), and the owner's time-under-water
> decision is shown without a suggested value.
>
> **Follow-up 1 — September 25, 2026 (evening).** The maintained definitions are now
> `docs/ifvg-redesign-fixes/followup-1/CALCULATION_DEFINITIONS.md`. The seed explanation above
> is narrowed: changing only the seed reproduces the difference, but the reference's seed was
> not recorded, so the historical cause is not proven. Trade review now dates the tap and the
> close-through by their candle's close (5:19 PM and 7:06 PM, not the mock's 5:18 PM and 7:05 PM).
> The "replaying the saved order reproduces all 128 saved results exactly" line below means the
> five summary figures, the number of trades and each trade's net result and account-loss flag;
> fill times, prices, quantities and setup lineage are not compared.

September 24, 2026. The redesigned screens run inside both existing applications: the main application (`streamlit run scripts/dashboard.py`) and the dedicated application (`python scripts/run_ifsm_research_ui.py`). Every number is computed from the saved study records as CALCULATIONS.md defines it.

- No study was launched, resumed or re-run.
- No strategy, engine, account, cost, calendar or payout rule changed.
- No market data was downloaded, and nothing dated June 11, 2026 or later was read.
- Firms are never added together.

Every browser check ran on an isolated copy of the stores.

Screenshots of each screen beside its mock are in `screenshots/`. Every judgment call is in `DECISIONS_LOG.md`, where each earlier feature now lives is in `FEATURE_MAP.csv`, and every placeholder still showing is in `DATA_GAPS.md`.

## P0 — Map and gap list: Done and verified

- **Built:** a map of both application entry points, their stores, the funded readers, the market-data readers and every panel to reuse. `FEATURE_MAP.csv` places all 86 earlier features. The gap list is in `TASKS.md` and `DATA_GAPS.md`.
- **Left:** nothing.

## P1 — Foundation: Done and verified

- **Built:**
  - design tokens and components;
  - the dark left rail (My studies, Trade review, New study);
  - one shared firm selection per saved result;
  - money, percent, price and Chicago 12-hour formatting;
  - links that reopen the same screen;
  - a calculation module passing all 22 CALCULATIONS.md reference tests;
  - caching.
- **Left:** nothing.

## P2 — Funded results overview (mock 02): Done and verified

- **Built:** the status line with its details, the selection and unseen windows, the merged ranking, the firm switch, checks on the leader and the findings link.
- **Checked:** switching firm changes the table and the leader together, and the detail opens on the same firm. The leader matches the reference ($30,781.88; Sharpe 4.01; Sortino 18.82; worst drawdown $3,661).
- **Reference values:** all CALCULATIONS.md reference values match within their tolerance except the low end of the 90% result-per-trade range: $133.51 (shown $134) against the reference $126, 6.0% off where 5% is allowed. It is a random-draw difference: the 20,000 resamples use the fixed seed 20260923, and the high end and the 68% and 95% ranges are within tolerance (decision F4). Three other figures differ from the reference by definition or rounding, not by draw, and the owner's review confirmed the computed values: 36 of 47 losers were never $250 in profit (reference 35; decision F5); the one-minute bar where buy and hold first fell $2,000 is stamped 9:03 AM, its closing minute (reference 9:02 AM; decision F6); and the March 18 – April 7 stretch made $10,536.36, shown +$10,536 (reference +$10,537; decision FX16). (Corrected by fix F6; the earlier summary said every leader value matched.)
- **Left:** the unseen window stays a placeholder until a confirmation run exists.

## P3 — Summary, Payouts and accounts, Settings and evidence (mocks 03, 04, 08): Done and verified

- **Built:**
  - tiles, the four-part verdict, findings with next steps, and key measures with a working 68/90/95% switch;
  - concentration and quality gates;
  - every earlier funded detail panel, re-homed;
  - the one-account chart with a plain explanation of how the account was lost;
  - the full settings, verification, corrections, limitations and decisions;
  - a download of the published review folder.
- **Left:** two gates and three gate inputs show "Not in export". The strategy replay didn't store them.

## P4 — Trades tab and Trade review (mocks 06, 09, 09b): Done and verified

- **Built:**
  - the result distribution, performance summary, both excursion charts, and the trade list with both fills of a half exit;
  - each trade row opens that exact trade in Trade review;
  - Trade review with four sources, real one-minute candles, setup zones from a verified saved record of the same execution, and numbered setup markers;
  - point in time, which hides everything later, including the result;
  - the full review form, saved per firm, account and trade in the existing ledger.
- **Left:**
  - Trades no verified configuration executed show "Setup zones weren't recorded for this study".
  - "+ New tag" is disabled, because the ledger has a fixed tag list.

## P5a — Risk and simulation, fixed floor (mocks 05, 05b): Done and verified

- **Built:** the payout race with editable loss limit and trigger, the resampled equity fan with the method switch, end values, losing streaks, and drawdown growth with its table.
- **Checked:** the fixed seed reproduces identically. "Run again" picks a new seed and says so. Values are within the stated tolerances.
- **Left:** nothing.

## P5b — Payout race with each firm's own rules: Done and verified

- **Built:** the same resampled orders run through the existing funded simulator, calling only its public ledger. It adds expected payouts before an account dies and expected cash per account bought.
- **Checked:** replaying the saved order reproduces all 128 saved results exactly. Runs on request: 1,000 paths in about 8 seconds.
- **Left:** TakeProfitTrader is labeled approximate, because the saved trades don't record when their high came.

## P6 — Market conditions (mock 07): Done and verified

- **Built:** condition labels, the four cards, the profit chart shaded by condition, the transition table, and the entry-condition scatter with its measure switch and link badge.
- **Checked:** day counts and per-condition totals match the reference. The entry-day/exit-day switch works; here both give the same counts, and the screen says why.
- **Left:** nothing.

## P7 — My studies, New funded comparison, Review and approve (mocks 01, 10, 10b, 11, 11b): Done and verified

- **Built:**
  - My studies across both applications' stores, with the five tabs and only "Open results" and "Continue draft";
  - the setup page with the S0_D80_W1_P1 default and the legacy warning, the date range with the resolved trading-day count, readable chips, and the two new settings;
  - the review page with varying settings first, validated pass/fail checks and the blocked-engine lock.
- **Checked:** nothing launches. Approval goes through the existing gated path. Saved drafts reopen unchanged.
- **Left:** the withdrawal trigger, gap rule, other dates and changed checks are saved but block approval until the simulator or engine supports them. The mock's 48-configuration plan builds 36 configurations, because the half exit exists only at 1R.

## P8 — Consistency sweep and handoff: Done and verified

- **Built:**
  - Chicago 12-hour times, plain-English labels and one placeholder style on every redesigned screen;
  - the earlier wizards and result pages inside the shared rail and tokens, with full-length selection chips;
  - the earlier funded results screen replaced everywhere by the new overview;
  - this handoff folder.
- **Left:** nothing.

## Load time (seconds, isolated application; first open on a freshly started server, then cached)

| Screen | First open | Cached |
|---|---|---|
| My studies | 3.7 | 0.2 |
| Funded results | 3.1 | 0.4 |
| Summary | 3.2 | 0.3 |
| Payouts and accounts | 2.9 | 0.4 |
| Risk and simulation | 3.7 | 0.7 |
| Trades | 2.9 | 0.5 |
| Market conditions | 3.4 | 0.4 |
| Settings and evidence | 2.8 | 0.4 |
| Trade review | 3.8 | 0.5 |
| New funded comparison | 1.8 | 0.3 |
| Review and approve | 2.1 | 0.4 |

The first open includes verifying and loading the 55 MB saved result. These times run in a browser, from navigation until the screen's final content is drawn. Decision P3.13's shorter figures (under 2 seconds, and under 0.1 cached) timed only the two tabs' own render in a headless test, with no browser; both hold, for different spans.

## Verification

- **Calculations:**
  - Every CALCULATIONS.md reference value is tested: 22 tests pass.
  - The redesign test folder has 222 passed and 1 skipped, which is a test that needs the half-exit engine.
- **Full test suite:** the numbers first given here (3,761 passed, 3 skipped, with the failures listed separately) came from a run made before two tests were edited, so no run of the final redesign code was reported. They are superseded by the fix task's one run on the code as delivered, after every fix (see `docs/ifvg-redesign-fixes/handoff/FIX_REPORT.md`, F1): 3,793 passed, 5 failed, 3 skipped; lint clean.
  - The five failures are exactly the pre-existing ones the repair task recorded: the frozen CatBoost bundle, the HTF cap experiment, and three capture-scheme default-tag tests.
- **Independent review:** four read-only reviewers looked at money and boundaries, calculations, navigation, and robustness. They found no blocker.
  - Every finding was fixed, each with a test that fails without the fix.
  - The largest fix: the earlier configurator now refuses approval and launch while a draft's new settings are blocked on Review and approve.
- **In the running application:**
  - every mock screen was captured beside its mock, and every control the mocks show working was clicked;
  - nothing was launched, and approval was never clicked;
  - all of this used an isolated copy of the stores.
- **Saved records:** all 22,619 snapshotted study, draft, approval, review and export files are identical before and after the task.

## For the owner

- In point in time, the default moment for the April 12 trade is 7:10 PM. Its half exit filled at 7:10:27 PM, so the half exit is hidden at that moment; the mock showed it. It appears from the 7:15 PM moment on.
- Open decision, unchanged: the time-under-water limit (saved at 3 days; you're weighing 18–20).
