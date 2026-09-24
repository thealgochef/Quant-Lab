# Funded payout work — progress

Two parts: **(1) the current single-account configuration comparison** (owner scope update of
September 23, 2026) and **(2) the completed September 22 pilot**, preserved unchanged below as
history. A rerun of the `funded-config-comparison` skill continues from "Current state"; it must
not re-snapshot (the snapshot `history/2026-09-23T053828-CDT/` exists), re-append the ledger
scope event (event id `funded_scope_change_20260923_single_account_configuration_comparison`,
idempotent) or relaunch a plan whose job state is Running or Completed.

## Current state — configuration comparison

Updated: 2026-09-23 (Chicago).
Scope: `AGENT_SCOPE_UPDATE.md` → SPEC.md Part A. Decisions: OWNER_DECISIONS.md entries of
September 23, 2026 (scope change; historical selection; feedback on result 5fa65149).
Current step: **post-run review response complete** (section below). No economic run was
repeated; export version 4 of the same result was published.
Next action: owner decision on the bounded follow-ups listed below. No run, retry or job is
queued.
Owner dependency: the proposed follow-ups need an explicit owner approval (none started).
(Root AGENTS.md sentence noted below.)
Source roots: Quant-Lab `C:/Users/gonza/Documents/Claude-Quant-Lab` (`main`, HEAD `5ece4d6`,
all funded work uncommitted); Strategy-Core pinned checkout
`../Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`
(read only, unmodified). Strategy source: verified daily-close study package
`archived-reports/ifvg_daily_close_sessions_20260918/final_extracted_4e5ce379c54731dd`
(run `a0f66422…`, 32 approved configurations, 10 warmup + 107 evaluation dates), read only.
Active jobs: none (the comparison job finished 2026-09-23 2:45 AM CDT; demo servers stopped).

### Acceptance checklist (this continuation)

| ID | Required outcome | Status | Evidence |
|---|---|---|---|
| C01 | Snapshot task docs/helpers before editing; reconcile docs and helpers | Done | `history/2026-09-23T053828-CDT/MANIFEST.md` (12 files, SHA-256). Rewritten: TASK.md, SCREEN_BRIEF.md, REVIEW.md, the rule, both skills, the reviewer agent. Extended: SPEC.md (Part A + preserved Part B), OWNER_DECISIONS.md, CODE_MAP.md, GUIDE_NOTES.md, this file. ARCHITECTURE.md and docs/ML_TRAINING_WORKBENCH.md gained comparison sections. |
| C02 | Research ledger records the scope change and the run | Done | `data/ifsm_ui_replication/search/funded_payout_research_ledger.jsonl`: `scope_change`, `run_started`, `run_completed`, two `review_folder_published` events. |
| C03 | Per-pair Core replay equivalent to the normal replay without account interference | Done | Historical run: 32 of 32 no-account replays equal the saved study's trades exactly. Real-Core tests in `test_strategy_driver_core.py`. |
| C04 | Account refusal and liquidation change later strategy state correctly | Done | Real-Core tests (refusal, liquidation, B1 sole-reason rule). Review finding B1 fixed before the run. |
| C05 | One live account per pair; replacements charged once, no credit stop; no expansion; no substitute during processing; no same-event replacement trade; pending at cutoff unreceived | Done | `test_pair_comparison.py` (23 tests); result validation checks for all 64 pairs passed. |
| C06 | Execution model v2 and price evidence | Done | Tests; historical run: 69,548 of 69,548 position minutes on exact prints. |
| C07 | Full vs resumed equivalence | Done | Historical run: resumed identical for 32 of 32 configurations (both firms). |
| C08 | Result and independent validation | Done | Validation passed; every check true for all 64 pairs. |
| C09 | Configurator → plan → approval → worker, distinct configurations → identified results | Done | `test_funded_comparison_workflow.py`; historical plan built from the configurator resolution. |
| C10 | Comparison screen | Done | Helper files verified by the lead; `test_funded_comparison_screen.py`. Lead fixes after the browser check: calendar-month chart, target wording, "no entry" label. |
| C11 | Automatic review folder, screen/export agreement | Done | `reports/funded_comparison/funded_comparison_92f5a08d3b81ae83_export_v1/` (automatic) and `_export_v2/` (adds REVIEW_FINDINGS.md; same result). |
| C12 | Rendered screen inspected in a real browser | Done | Headless Chrome over DevTools; screenshots in `../Claude-Quant-Lab-Research-Artifacts/funded-comparison-20260923/screenshots/` (`real_top.png` is the historical result). Minor: long labels push the loss columns into horizontal scroll. |
| C13 | Fresh read-only review; blockers fixed and retested | Done | `funded-payout-reviewer`: 1 blocker (B1) fixed; gaps closed (memory release, bounded workers, Incomplete status, per-pair coverage, plan decisions). Tests after fixes: 113 passed; `ruff check src tests scripts` clean. |
| C14 | Historical comparison of the owner-selected configurations | Done | Plan `9edbbdcd…`, approval (conversation), result `92f5a08d…`, 32 configurations × 2 firms, 1 mini, 33 minutes, 4 workers (~2 GB each). |
| C15 | Final report | Done | Session report of 2026-09-23. |

### Historical result (simulated, one path; plan 9edbbdcd…, result 92f5a08d…)

Every configuration earned positive net cash at both firms. Highest net cash after all
account costs: TakeProfitTrader — S0_D80_W1_P1 $15,510.31 (3 accounts, $306 costs, 19
payouts); MyFundedFutures — S0_D160_W1_P1 $17,759.37 (3 accounts, $375 costs, 21 payouts).
Lowest at both: S3_D160_W1_P0 ($2,021.97 / $2,243.96). Nothing was pending at the cutoff.
Full table: the review folder's `configuration_results.csv`.

### Variation study (second owner run, September 23, 2026) — complete

Owner request: run the promising variations from the research notes on the 107 days; owner
answers: 48-configuration set + scale-out exit, measure both, scale-out at 10 micros, $0.514 per
micro per fill (OWNER_DECISIONS.md, SPEC.md A10).

| ID | Outcome | Status | Evidence |
|---|---|---|---|
| V01 | Scale-out exit in Strategy-Core (separate branch, pinned Core unchanged) | Done | `../Claude-Quant-Lab-Research-Artifacts/strategy-core-scale-out-exit` branch `funded-scale-out-exit` (uncommitted, patch SHA-256 `962ad56a…`); Core suite 691 passed / 3 skipped (pinned: 685 / 3); 6 new scale-out tests. |
| V02 | Registry values (target 2R/3R, one-hour-only gaps, parents without 3-minute, exit rule) | Done | `search/axis_registry.py`; registry-dependent tests 459 passed, 1 failed — the failure (`test_htf_cap_experiment`) predates this work and fails identically with the committed registry. |
| V03 | Account-level half exit, exact micro costs, variation plans (v2), superset cache filtering, strategy measures | Done | `position_walk.py`, `pair_ledger.py`, `pair_engine.py`, `comparison_*` modules; funded tests 129 passed; ruff clean. |
| V04 | Independent review round 2 | Done | 6 blockers fixed before the run (failing-variant handling, initial stop in trade rows, break-even slippage, crossing print after the half exit, micro price-path disclosure, plan decisions); see the review folder's REVIEW_FINDINGS.md. |
| V05 | Historical run | Done | Plan `78406b15…`, approval (conversation), result `5fa65149…`; 64 configurations x 2 firms; 4 workers, 10:43 AM–12:04 PM CDT; validation passed; equivalence 64/64; resumed 64/64; 515,388 of 515,438 position minutes on recorded trades. |
| V06 | Review folder and screen | Done | `reports/funded_comparison/funded_comparison_5fa65149843484b1_export_v1/` (automatic) and `_export_v2/` (adds REVIEW_FINDINGS.md); screenshot `../Claude-Quant-Lab-Research-Artifacts/funded-comparison-20260923/screenshots/variation_study_top.png`. Long configuration names are truncated in table cells (full text on hover and in CSVs). |

Headline (simulated, one path): scale-out configurations took the top 14 places at both firms.
Best: all open-market hours, 1R, one-hour+four-hour gaps, parents with 1- and 3-minute, long
only, scale-out — TakeProfitTrader $30,781.88, MyFundedFutures $34,818.11 net cash, versus
$15,510.31 / $16,519.37 for S0_D80_W1_P1 with the whole-position 1R exit.

### Post-run review response (owner feedback of September 23, 2026 on result 5fa65149)

Source: `FEEDBACK_FOR_AGENT.md` (OWNER_DECISIONS.md, same date). Evidence folder (internal):
`../Claude-Quant-Lab-Research-Artifacts/funded-comparison-20260923/feedback_round_20260923/`
(`scripts/`, `outputs/`, `test_receipts/`, `screenshots/`, `REVIEW_FINDINGS_v4.md`). Review
folder: `reports/funded_comparison/funded_comparison_5fa65149843484b1_export_v4/` (v3 has the
same figures with an unreadable chart; v4 replaces it for reading).

| ID | Outcome | Status | Evidence |
|---|---|---|---|
| R01 | Stop-difference summary: final stop, contracts actually closed there, same-minute break-even stops counted | Done | `comparison_result.stop_fill_difference_cents`; independent recompute: 32 of 128 pairs wrong (all half-exit), 6 undercounted; leader 23 / $225 → 23 / $150; first comparison unaffected (`outputs/stop_summary_discrepancies.csv`). |
| R02 | Correction of the saved result without changing money | Done | `apply_reporting_corrections` on load (hash-verified bytes unchanged); export check `reporting_corrections_are_summary_only`; v4 headline money and ranks identical to v2 for all 128 results; tests `test_stop_difference_summary.py`. |
| R03 | Substituted minutes resolved | Done | `outputs/approximated_intervals.csv`: 97 rows re-walked exactly; 50 configuration minutes = 4 clock minutes, all one same-timestamp matching event whose first trade differs from the candle open; 0 rows where any order can change an exit, survival or payout; 13 target-only (fill time within the minute only). Leader's April 13 and April 16 positions: no level inside the minute. No sensitivity replay needed. |
| R04 | Micro fidelity kept separate and labeled | Done (limitation disclosed, not fixed) | Screen status line and per-configuration traded / signal / mark / execution sources; 3,347 proxy micro positions; only `NQ.FUT` data stored locally (no MNQ); $0.514 fee aggregation verified ($10.28 per round trip). |
| R05 | Execution semantics documented and tested | Done | SPEC.md A11; `test_execution_semantics.py` (same-minute half then stop, same-timestamp order and file ordering, partial fee / remaining mark / intraday floor, breach before break-even, gap-through break-even, payout lock only when flat). Equivalence populations 2 saved controls vs 62 research-Core references labeled. |
| R06 | Full effective configuration, source binding and calendar per result | Done | `configuration_bindings.json` (re-resolved to the frozen hashes), `trading_calendar.csv`, boundary check (986 midnight crossings, 0 outside their trading day; saved schedule, not an independent exchange calendar). |
| R07 | Reference records reconciled | Done | 9,994 = 590 preparation-period + 9,404 evaluated, 0 unexplained (`outputs/reference_record_reconciliation.csv`). |
| R08 | Wording corrections | Done | Daily-close rule (correction `daily_close_wording_v1`, generator fixed); large-payout removal reclassified as sensitivity; median payouts $2,217.33 / $2,494.49 shown; largest $5,610.66 / $6,311.99; other variant MyFundedFutures $6,825.24. |
| R09 | Final payload receipt | Done | `verify_published_folder` after publication; v4: 30 payloads verified, 0 unlisted; manifest states `payload_files`; validation summary states its pre-manifest scope. |
| R10 | Completed-screen evidence | Done | `screenshots/v_list.png`, `v_top.png`, `v_detail.png`, `v_configurator.png` (real browser, app launched with `--research-core`). Setting columns, net cash beside rank, full settings and facts, leader opened by default. |
| R11 | Source snapshot, save/reopen, worker source selection via the normal configurator | Done | Snapshot `../Claude-Quant-Lab-Research-Artifacts/research-core-sources/7c7111e398c0_962ad56a36471c85/`; variation configurator reproduces plan `78406b15…` exactly and shows it completed and approved; `job start` selects the exact checkout or refuses (`test_funded_comparison_workflow.py`); pin unchanged; launcher `--research-core` explicit. |
| R12 | Complete test outcomes | Done | Receipts in `feedback_round_20260923/test_receipts/`: final (`test_receipts/final/`): funded 146 passed (pinned Core) and 146 passed (research Core); registry-dependent plus launcher tests 481 passed, 1 skipped, 1 failed (`test_htf_cap_experiment`, reproduced identically on clean committed HEAD in a temporary worktree); ruff clean; Core suites not rerun (branch source hash unchanged: 691/3 skipped and 685/3 skipped from the earlier receipts). |

Proposed bounded follow-ups (need the owner's explicit approval; none started):

1. Micro cost sensitivity: the same 16 half-exit configurations re-priced at owner-chosen
   alternative micro costs (for example each firm's published micro commission, once the
   owner confirms it), same dates, same rules — a separate comparison.
2. Micro execution evidence: obtain MNQ trade data for the same January 13 – June 10, 2026
   period (owner authorization needed for any download or purchase), then a matched
   same-product, same-size (10 MNQ) whole-exit versus half-exit comparison for the leader.
3. Independent shortened-session calendar check of the saved schedule.
4. Decide whether to commit the research Core branch and how to pin it (no pin change made).

### Engineering runs (not owner results; scratch stores only)

These ran BEFORE the review fix B1 (sole-reason refusals); their figures are superseded by the
historical run above (control after the fix: $11,251.93 / $11,254.17).

- 2026-09-23: control S0_D160_W1_P0, one mini, both firms (`run_configuration`). TakeProfitTrader:
  8 accounts, received $12,307.71, costs $816.00, net $11,491.71. MyFundedFutures: 7 accounts,
  received $12,398.91, costs $875.00, net $11,523.91. Reference 159/159; resumed identical;
  3,614/3,614 minutes on exact prints. Evidence: session scratchpad `control.json`.
- 2026-09-23: worker path, plan `4740e576…` (engineering_sample, scratch store), S0_D160_W1_P0 and
  S3_D160_W1_P0 × both firms, 2 worker processes: Completed, validation passed; control values
  identical to the standalone run. The first attempt (plan `b6df40f9…`) exposed a runner defect —
  a run with zero completed configurations was reported Completed — fixed (a run now needs at least
  one completed configuration) before the second attempt.

### Active jobs and file ownership

- Lead: all `propsim/funded/comparison_*`, `pair_*`, `position_walk.py`, `print_minutes.py`,
  `strategy_driver.py`; `scripts/ifvg_funded_comparison_study.py`, `..._job.py`; hooks in
  `ifvg_research_wizard.py`, `ifvg_workspace.py`, `presentation/workspace.py`, `search/store.py`,
  `ifvg_funded_study.py` (label); tests `test_pair_comparison.py`, `pair_builders.py`,
  `test_strategy_driver_core.py`, `test_funded_comparison_workflow.py`; all task documents.
- Helper (screen/export, 2026-09-23, finished and verified by the lead): `presentation/funded_comparison.py`,
  `scripts/ifvg_funded_comparison_results.py`, `funded_review_package`-style
  `funded_comparison_review.py`, `tests/agents/test_funded_comparison_screen.py`,
  `tests/agents/data_infra/ifvg/test_funded_comparison_review.py`,
  `tests/propsim/funded/comparison_fixture.py`.
- No study workers or Streamlit servers running (checked after the run).

### Root-instruction notes (reported, not rewritten)

- `AGENTS.md`, "Funded payout lane (2026-09-22)" paragraph: "Historical runs need an exact
  authorization record (`PILOT_AUTHORIZATION`)" describes only the earlier budgeted mode. The
  comparison mode uses the stored `funded_comparison_approvals` record instead. Left for the owner
  to update (root instruction file).

---

# Completed pilot record (September 22, 2026) — historical, unchanged

# Funded payout implementation — progress

**Initial status: Not started in this package.** No implementation, tests or financial pilot have been executed by creating these files. The agent must first reconcile any existing progress in the previous task folder and the current repositories.

## Current state

Current step (2026-09-22, 10:45 PM CDT): implementation, pilot, review and fixes are complete.
Next action: the owner decides (1) whether to accept the shared-signal simplification or require per-account strategy replay before any definitive firm ranking, and (2) whether stop exits should keep filling at the recorded stop price. No retry or new run is queued.
Owner dependency: none open for the pilot. The owner resolved the clock (two business days), pilot profile (`S0_D160_W1_P0`), size (one mini) and conventions on 2026-09-22 (see OWNER_DECISIONS.md).
Source roots / task workspaces:
- Quant-Lab: `C:/Users/gonza/Documents/Claude-Quant-Lab`, branch `main`.
- Strategy-Core, imported READ-ONLY: the installed package (site-packages) and the IFSM checkout `../Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/7c7111e398c083cf8e966e2e0c5aac8a41cc12c0` (verified by `run_ifsm_research_ui.py --check`). No Core edits are planned.
Pre-task revisions and existing uncommitted work:
- Quant-Lab HEAD `5ece4d650767d60654a7b9104497850a5ddac96b`. The working tree was clean apart from the untracked task files `.claude/` and `docs/funded-payout-implementation/`. The session-start git snapshot showing a feature branch was outdated.
- Sibling `../Strategy-core` HEAD `6f63a4c7…` has an unrelated modified `V3_COMPATIBILITY_MATRIX.md`. It is not imported and will not be touched.
Active implementation sessions/jobs: none. No python/streamlit processes were running. All `data/ifvg_search_jobs` and `data/ifvg_pipeline_jobs` logs are terminal (last activity 2026-09-11). The IFSM workspace `data/ifsm_ui_replication/search/v1` holds no studies. The daily-close 32-profile study is COMPLETE and was verified at 2026-09-22 5:32 PM CDT (archive `ifvg_daily_close_sessions_a0f66422…_v1.zip`, receipt in `../Claude-Quant-Lab-Research-Artifacts/archived-reports/`). Its sources are archived and are read only.
Evidence location: the internal engineering evidence goes to `../Claude-Quant-Lab-Research-Artifacts/funded-payout-20260922/`. Completed-run review folders are published by the application under `reports/funded_payout/`, which is ignored and never committed.

## Acceptance checklist

| ID | Required outcome | Status | Evidence / actual command / result |
|---|---|---|---|
| F01 | Current source/import map, existing instructions, active workers and safe edit boundaries | Done | Recon recorded above. Launcher `run_ifsm_research_ui.py --check` passed; Core imported read-only from the managed 7c7111e checkout. No active jobs. Baselines are in `../Claude-Quant-Lab-Research-Artifacts/funded-payout-20260922/`baseline/. |
| F02 | Financial rules and price-evidence capability; proposals distinguished from approvals | Done | Owner answers of 2026-09-22 are in OWNER_DECISIONS.md. Price evidence: local MBP-1 prints rebuild every one-minute candle of all 156 pilot positions exactly. Minute-candle modes exist only as labeled approximations. |
| F03 | Fresh-funded start/replacement; costs, contract limits and account-aware trading path | Done | `propsim/funded/{profiles,instance,positions}.py`. Accounts start fresh and funded at $102 or $125, with no evaluation. Size limits are refused, never clipped (6 versus 3 minis, combined mini/micro). Costs are $5.14 per fill. Tests are in `tests/propsim/funded/test_funded_engine.py`. |
| F04 | Correct intraday/current-equity enforcement and separate end-of-day threshold update | Done | TakeProfitTrader's floor moves with the intraday peak of realized plus open equity and locks at 0. MyFundedFutures' floor moves only at the session close and locks at +$100 (below). Both are enforced on every ordered print, with gap-through liquidation. Equality and one-cent cases are tested. |
| F05 | Secured-payout entry lock, day-end gross request, processing pause and one receipt | Done | The secured lock, day-end full-surplus request, 2-business-day or 48-hour clock, single receipt, cutoff states and checkpoint resume are tested. In the pilot, resumed-state equivalence held for both firms. |
| F06 | Independent firm budgets, monthly credits, carryover, replacement vacancies and growth | Done | Budgets are isolated per firm. Credits: initial grant, monthly grants once, carryover. Vacancies wait in order for credits. Growth uses the 25% wallet test and the net-cash guard, at most once per day, capped at 20. Covered by tests and pilot validation. |
| F07 | Normal application settings save/reopen and actual worker propagation | Done | A new study type in the normal workspace. The draft saves and reopens through the real AppTest and the real browser. The frozen plan is the worker's only input. Settings changed in the screen reach the worker (`tests/agents/test_funded_payout_workflow.py`). |
| F08 | Final financial screen, account/trade drilldown, actual visual check and empty/error cases | Done | Results screen, account and trade drilldown, and empty or all-failed states. Real headless-Chrome screenshots are in the session scratchpad `pilot/05_after_publish.png` and `pilot/06_configure_after_fix.png`. A caption shown as math was found in the real screenshot and fixed. |
| F09 | Automatic clean review folder, cumulative ledger, internal checks and atomic publication | Done | The review folder is automatic, uses the exact allowlist, and is staged, verified and then published atomically. The first automatic export was refused by its own check (`documents_contain_no_file_paths`: imported history in RESEARCH_LEDGER.md), so no folder was written. After the fix it was published through the screen's retry button as `reports/funded_payout/funded_payout_330142736fcb00fb_export_v1/`. The cumulative ledger is `data/ifsm_ui_replication/search/funded_payout_research_ledger.jsonl`. |
| F10 | Regression/negative/checkpoint tests; exact commands and passed/failed/skipped counts | Done | Final: 381 passed, 0 failed, 0 skipped (`python -m pytest tests/propsim` plus the funded, workspace, wizard, IFSM-UI, draft and identity files). Ruff is clean. Receipts are in `../Claude-Quant-Lab-Research-Artifacts/funded-payout-20260922/`test_receipts/. |
| F11 | Fresh read-only review of task-specific changes; reproduced blockers fixed and retested | Done | A fresh `funded-payout-reviewer` found no demonstrated money defect. Fixed after review: the stop-gap count is added to the price evidence, the other-exit-first case is listed as a limitation, and the rules show 4:00 PM. The shared-signal deviation stays open for an owner decision. Review inputs are in the evidence folder `review_input/`. |
| F12 | Bounded same-period historical pilot after all prerequisites; no unintended sweep | Done | One authorized pilot through the real UI and worker. Plan `dc4016e9…`, result `33014273…`. January 13 – June 10, 2026, control `S0_D160_W1_P0`, one mini, two business days. No sweep and no new dates. |
| F13 | Screen/internal/export money reconcile; final outcomes and limits reported honestly | Done | The screen and export headline figures agree with each other and with the result cents (verified in export checks). Outcomes are reported in the final summary with their limitations. |

Use Open, In progress, Done, Blocked, or Not verified. A blocker is not Done. For each completed item record supporting paths and actual observations; a helper summary is not test evidence. Zero collected tests is not a passing suite. Keep synthetic checks, reused results, diagnostic analyses and newly completed financial runs distinct.

## Active jobs and file ownership

Record job/session identifier, purpose, working roots, frozen settings, start time, result/receipt location, next check and current status. Record which worker owns each shared file. Check this section and the actual processes before a launch, retry or new writer.

2026-09-22, before the pilot launch:
- Helper 1 (read-only Explore) mapped the study workflow. Done.
- Helper 2 (screen/export) owned `presentation/funded_results.py`, `scripts/ifvg_funded_results.py`, `funded_review_package.py`, their two test files and `tests/propsim/funded/sample_variants.py`. Done. The lead re-ran its 18 tests and they pass. The lead later added the ready-time split rows to the presenter.
- Lead owns everything else: `src/alpha_lab/propsim/funded/*`, `scripts/ifvg_funded_study.py`, `scripts/ifvg_funded_payout_job.py`, the hooks in `ifvg_research_wizard.py` / `ifvg_workspace.py` / `ifvg_rules.py` / `presentation/workspace.py` / `search/store.py`, `pyproject.toml` (matplotlib) and all funded tests.
- No study workers were running before launch (checked 2026-09-22).
- Pilot pre-check: all 156 evaluated control executions have ordered MBP-1 print paths whose rebuilt one-minute candles equal the study's candles exactly. No ordering conflicts. 83 day files.

## Change-review evidence

Record the pre-task baseline of each repository separately, relevant starting uncommitted changes, final touched-file list and internal task-only diffs or before/after files. The read-only reviewer must have those artifacts; do not direct it to compare blindly against main or count unrelated edits as this task's changes.

## Handoff and completion

After material progress, update current/next steps, evidence, owner dependencies and job status. When context is shortened or a session ends, leave enough information to resume without duplicating work. If earlier evidence was wrong, append the correction and its reason rather than erase history.

Final report: Needs your decision; Implemented and tested; Simulation results; Not verified; Review files. A financial pilot stays Blocked or Not run if only engineering checks completed. Update the cumulative project research ledger without fabricating historical facts.

## Pilot outcome (2026-09-22, simulated, one historical path)

| Firm | Net cash after account costs | Received after split | Largest payout | Account costs | Lost before / after a payout |
|---|---:|---:|---:|---:|---|
| TakeProfitTrader | $68,063.90 | $72,653.90 (90 payouts) | $1,783.78 | $4,590.00 | 25 / 5 |
| MyFundedFutures | $76,110.60 | $81,735.60 (90 payouts) | $2,006.75 | $5,625.00 | 25 / 5 |

Nothing was pending at the cutoff. The shared-signal limitation applies.
A separate diagnostic, not part of the economic run: 17 of 67 stop exits had a
triggering print worse than the stop, 28 ticks in total ($140 per mini), with a
largest gap of $30.
