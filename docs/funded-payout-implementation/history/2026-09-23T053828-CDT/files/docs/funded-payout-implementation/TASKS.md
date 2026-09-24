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
