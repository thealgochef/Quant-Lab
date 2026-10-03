# IFVG Lab — build the redesigned dashboard

## Assignment and finish line

Build the redesigned IFVG Lab screens shown in `mocks/images/`, wired to real study data, inside the existing application. The mocks are the north star for what goes on each screen, in what order, grouped how, and in what words. The owner approved this direction on September 23, 2026.

Done means: each screen in SCREENS.md exists in the running application, shows real values computed as CALCULATIONS.md defines, its controls work, every existing feature still has a home, and the handoff in section 6 exists with screenshots from the actual application next to the matching mock. A plan, a component library without wired data, or passing unit tests without checking the running application does not complete this task.

The research question does not change: which configuration produces the most cash received after every funded-account cost? Account death is a priced cost, not a failure to hide. Aggressive configurations are judged on the same cash measure as conservative ones.

## 1. Prerequisites

Read `docs/ifvg-dashboard-repairs/TASKS.md`. The repair task must be complete before phases that depend on it:

| Depends on repair | Phases blocked until it's fixed and verified |
|---|---|
| R1 saved-plan integrity and readable selections | P7 setup and review |
| R2 firm selection consistency | P2, P3 |
| R3 funded trades reachable in Trade review | P4 |
| R4 Chicago time everywhere | all phases (use its shared time helper) |
| R5 threshold validation | P7 |
| R7 named default baseline | P7 |
| R8 date range and extended research window | P7 |

If a repair isn't marked Fixed and verified, don't redo it here. Build the phases that don't depend on it, and report the blocker.

## 2. Boundaries and preservation

Read the latest local instructions and owner decisions, including `docs/funded-payout-implementation/` and `docs/ifvg-dashboard-repairs/` when present. Preserve them.

Before editing, map both application entry points (the capture guide named `scripts/dashboard.py` and `scripts/run_ifsm_research_ui.py`; verify current locations), their study stores, the funded export readers, the market-data readers, and every existing panel that SCREENS.md says to reuse. Check for other agents or running studies and don't edit files they're using.

Never, in this task:
- launch, re-run, resume or extend any study, including "just to check a number"
- change strategy logic, the engine, account rules, costs, calendars, payout mechanics or any financial record
- download, purchase or request market data
- read, display or compute anything from June 11, 2026 onward
- delete or rewrite completed studies, drafts, approvals, reviews or notes
- remove an existing feature without a home for it (see DECISION_RULES.md rule 12)
- add firms together into one number

Allowed: read-only calculations over stored exports and stored market bars, cached results of those calculations, new display code, shared UI components, and the reorganization shown in the mocks.

## 3. Build phases

Work in this order. Each phase ends with its acceptance checks passing in the running application and a TASKS.md update. Log every judgment call in `handoff/DECISIONS_LOG.md` as you make it.

### P0 — Map and gap list
Map every panel in SCREENS.md to its data source and to any existing code that already produces it. List every value the mocks show that no stored record supports (the known ones are in SCREENS.md "Known data gaps"). Acceptance: `handoff/FEATURE_MAP.csv` lists every existing dashboard feature with its new home, and the gap list is recorded in TASKS.md.

### P1 — Foundation
Shared pieces every later phase uses: the color and type tokens and components in DESIGN_SYSTEM.md; the left navigation (My studies, Trade review, New study); one firm-selection state shared by every funded view; formatting helpers for money, percentages, prices and Chicago 12-hour times; a calculation module implementing CALCULATIONS.md with tests against its reference values; a cache keyed by study, configuration, firm and settings so pages open quickly. Acceptance: calculation tests pass against the reference values; a sample page shows the tokens and components.

### P2 — Funded results overview (mock 02)
Status line, selection and unseen windows, one ranking table with cash and strategy columns, firm switch, checks on the leader, findings link. Acceptance: switching firm changes the table and leader numbers together, and the detail view opened from a row shows the same firm; the leader's values match CALCULATIONS.md reference values.

### P3 — Configuration detail: Summary, Payouts and accounts, Settings and evidence (mocks 03, 04, 08)
Tab bar shared by all six detail tabs. Summary: headline tiles, four-part verdict, findings, key measures with the 68/90/95% switch, concentration, quality gates. Payouts and accounts and Settings and evidence are mostly existing panels, reorganized. Acceptance: every existing panel from the current funded detail view appears in one of these tabs; verdict and findings follow the rules in CALCULATIONS.md.

### P4 — Trades tab and Trade review (mocks 06, 09)
Trades tab: result distribution, excursion charts, performance summary, trade list with both halves of partial exits. Trade review: funded trades as a source, candles from stored bars, setup zones from saved setup records, entry and exit markings, point-in-time mode, setup timeline, full review form. Acceptance: clicking a trade in the Trades tab opens that exact trade in Trade review; point-in-time hides every candle and value after the chosen moment; reviews save per firm, account and trade.

### P5 — Risk and simulation (mock 05)
Payout race, resampled equity with the two-method switch, end distribution, losing streaks, drawdown growth. Then the trailing-floor version of the payout race (CALCULATIONS.md "Payout race — full version"). Acceptance: fixed-seed results are identical on reload; "Run again" changes the seed and says so; results match reference values within the tolerances listed.

### P6 — Market conditions (mock 07)
Condition labels, cards, shaded profit chart, transition table, entry-condition scatter with its switch. Acceptance: day counts and per-condition totals match reference values; the entry-day/exit-day switch works.

### P7 — My studies, New funded comparison, Review and approve (mocks 01, 10, 11)
Library across both apps' study stores with tabs and two action labels. Setup with the named baseline default and legacy warning, date range, readable choices, and the new withdrawal-trigger and gap-rule settings. Review with varying settings first, validated thresholds, and the blocked-engine state. Acceptance: nothing here launches a study; approval still goes through the existing gated path; a saved draft reopens unchanged.

The two new setting axes (withdrawal trigger, gap rule) may not exist in the engine or funded simulator yet. If they don't, build the controls, save them in the draft, and block approval with a plain message naming the missing capability. Don't implement the capability in the engine in this task.

### P8 — Consistency sweep and handoff
Every screen: Chicago 12-hour times, plain-English labels, placeholder style for missing values, no leftover old-style pages reachable from navigation. Other study wizards (Evaluate, Compare, Search, Prop feasibility, Strategy across firms, Full workflow) and strategy and model study result pages get the shared shell, tokens and readable selections only; their options and result content don't change in this task.

## 4. Two things the mocks show that need new plumbing

1. **Setup records for funded trades.** The funded study export has no setup zones. Each funded trade carries `strategy_trade_id`; find where the strategy side saves setup geometry for that trade (earlier studies stored a geometry record with the four-hour gap, parent gap, opposing gap, tap, lock and inversion bars) and link it. If a study never saved geometry, Trade review shows candles, entry, stops and exits, plus the placeholder "Setup zones weren't recorded for this study". Never re-run the strategy to regenerate them.

2. **The payout race with real firm rules.** The mock's version uses a fixed floor. The full version feeds resampled trade orders through the existing funded account simulator so each firm's own trailing floor, lock, payout protection and processing rules apply. Reuse that code; never write a second copy of firm rules. If it can't be called on a synthetic trade sequence without modifying it, keep the fixed-floor version, label it, and report what the simulator would need.

## 5. Verification

- Calculation tests against every reference value in CALCULATIONS.md.
- For each screen: a screenshot of the running application beside the matching mock image, captured with the leader configuration of the funded variation study selected.
- Click checks for every control the mocks show working (firm switch, detail tabs, 68/90/95 switch, method switch, point-in-time, baseline picker, blocked review state).
- Load time for each screen, first open and cached.
- Saved studies, drafts and reviews: hashes before and after match.

## 6. Handoff — exact contents

In `docs/ifvg-dashboard-redesign/handoff/`:
- `REDESIGN_REPORT.md`: per phase — status (Done and verified, Partly done, Blocked, Not started), what was built, what's left, and blockers in one line each. Plain English, no evidence columns.
- `FEATURE_MAP.csv`: every existing feature, its new location, status.
- `DECISIONS_LOG.md`: every rule-based decision — the rule number, what was decided, where it shows.
- `screenshots/`: application screenshot and mock image side by side for each of the 17 mock images.
- `DATA_GAPS.md`: every placeholder still showing, why, and what would fill it.

## Final response and stopping point

Reply with: phases completed and verified; phases partly done or blocked and why; the data gaps still showing placeholders; decisions the owner should look at first (at most five); and the handoff folder path. Then stop.
