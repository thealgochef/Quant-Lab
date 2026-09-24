# Code map for the funded payout work

## Verified current locations (inspected September 23, 2026)

These anchors were read in the current working tree (Quant-Lab `main`, HEAD `5ece4d6` plus
this task's uncommitted files) and the pinned Strategy-Core checkout
`../Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`
(identical to the installed `strategy_core` package apart from line endings; the IFSM
launcher `scripts/run_ifsm_research_ui.py` puts that checkout first on `PYTHONPATH`). Core is
not modified. The section "Earlier inspection notes" below is the original package's map and
is kept as history; it is not proof of current imports.

### Configuration comparison (current mode)

| Location | Role |
|---|---|
| `scripts/ifvg_research_wizard.py` `_choose_study` / `render_new_study` | "Funded configuration comparison" in More study types; drafts of mode `funded_configuration_comparison` route to the comparison screen. The earlier type is labeled "Funded five-account operation (earlier budgeted mode)". |
| `scripts/ifvg_funded_comparison_study.py` | Configurator screen (per-setting multiselects over the registry value ids, firms, size, clock), plan preview, owner approval action, freeze + detached launch, study listing and result routing. |
| `scripts/ifvg_funded_comparison_job.py` | `start / status / worker / publish` for one frozen plan. `start` refuses a plan already Running or Completed and, before queuing a worker, a historical plan without a stored owner approval (the approval check was added by the IFVG dashboard repair). |
| `scripts/ifvg_workspace.py` `render_study` | `kind == "funded_comparison"` → `render_comparison_study`. |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/workspace.py` `load_studies` | Lists comparison runs (`funded_comparison_jobs/<plan>/state.json`) and unfrozen comparison drafts. `axis_value_name` (setting and value) labels unavailable combinations; `axis_value_text` (the value alone, added by the IFVG dashboard repair) is used by `comparison_draft.value_label` for the "Selected:" line and as the chip fallback when a value has no short variation name (`_chip_label`). |
| `src/alpha_lab/agents/data_infra/ifvg/search/store.py` `SEARCH_STORE_NAMES` | Adds `funded_comparison_plans`, `funded_comparison_approvals`, `funded_comparison_results`. |
| `src/alpha_lab/propsim/funded/comparison_study.py` | Axis choices, selection → approved configurations (unapproved combinations listed), size check, plan building, owner-approval recording. |
| `.../comparison_plan.py` | Plan / approval / result envelopes, question, decisions, limitations. |
| `.../comparison_source.py` | Verified study package → configurations; normal configurator resolution (`resolve_axis_overrides` → `resolve_profile_config` → `canonicalize_section`, hash must match); trusted cached day inputs under the approval's read policy; saved trades for the equivalence check. |
| `.../strategy_driver.py` | `CoreStrategyDriver`: per-pair `DayOrchestrator` chain; seams `IfvgReducer.execution_block_reasons` (per-instance extension) and `IfvgReducer._setup` (cleared for liquidation / refused setups). |
| `.../pair_engine.py` | Candle loop coupling each pair's driver and ledger; consistency errors. |
| `.../pair_ledger.py` | One-account-at-a-time ledger: purchases, replacement at failure, payout lifecycle, day-end floor update, timed events, checkpoint. |
| `.../position_walk.py` | Execution model `ordered_prints_stop_market_v2`. |
| `.../print_minutes.py` | Per-candle exact print reconciliation or labeled approximation. |
| `.../comparison_run.py` | One configuration: reference replay, one run per firm, mid-period resume check. |
| `.../comparison_runner.py` | Worker: plan + approval check, one process per configuration, result, validation, store, ledger, review folder. |
| `.../comparison_result.py`, `.../comparison_describe.py` | The one result (tables, summaries), independent validation; plain-English settings. |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/funded_comparison.py`, `scripts/ifvg_funded_comparison_results.py`, `src/alpha_lab/agents/data_infra/ifvg/funded_comparison_review.py` | Comparison screen presenter, renderer and review folder (see TASKS.md for status). Presenter: `configuration_columns` (the settings that differ, as short columns), `execution_sources`, `proxy_micro_positions`, median payout. Export: `SUPPLEMENT_FILES`, `reporting_corrections.csv`, `verify_published_folder` (final payload receipt). |
| `.../comparison_result.py` `stop_fill_difference_cents`, `apply_reporting_corrections` | Stop-difference measure (final stop, contracts closed there); summary-only corrections of saved results, recorded under `reporting_corrections` (added September 23, post-run review). |
| `.../comparison_runner.py` `load_comparison_result`, `review_supplements`, `republish_comparison_review` | Loading applies the recorded corrections after the hash check; supplements = bindings, calendar, boundary check; republish = the next export version of the same saved result (`job republish`). |
| `.../comparison_evidence.py` | `configuration_bindings`, `trading_calendar_rows`, `trade_boundary_check`. |
| `.../comparison_study.py` `VARIATION_AXES`, `variation_variants`, `core_source_description` | Version-2 variation plans from the configurator (half exit only at 1R; charts must be in the base cache). |
| `.../core_identity.py` `source_identity_at`; `.../research_core_sources.py` | Identity of any Core checkout; `find_core_checkout` (plan → exact local checkout); `snapshot_core_source` (immutable internal snapshot). |
| `scripts/ifvg_funded_comparison_job.py` `worker_environment`, `republish` | Worker launched only on the plan's exact Core source; new export version command. |
| `scripts/ifvg_funded_comparison_study.py` `_render_variations`, `_approve_and_run` | "Variations around one configuration" in the normal configurator: saved/reopened selections, sizes per exit rule, frozen engine shown, already-run plans not relaunched. |
| `scripts/run_ifsm_research_ui.py` `--research-core` | Explicit, labeled research Core on the pinned commit (receipt prints the patch hash); default launch unchanged. |

### IFVG dashboard repair anchors (read September 23, 2026, after the repair)

Screen-level changes from `docs/ifvg-dashboard-repairs/` (R1–R4); behavior and limitations in
SPEC.md A12. No money rule changed.

| Location | Role |
|---|---|
| `src/alpha_lab/propsim/funded/comparison_draft.py` `saved_settings`, `saved_study_selections`, `check_saved_comparison` / `SavedComparisonCheck`, `matching_saved_plans`, `rebuild_saved_plan`, `value_label` | Saved drafts read exactly as saved (no defaults merged, nothing dropped or substituted); what the running engine can represent, with an engine-independent configuration count; saved plans that match the draft exactly; the one plan rebuild shared by the screen and the launch check. |
| `scripts/ifvg_funded_comparison_study.py` `_render_saved_read_only`, `_sync_widget_state` / `_draft_marker`, `_mark_edited` / `_save_edits`, `dispatch_problem`, `_chip_label` | Unrepresentable draft → read-only page (saved settings, count, matching plan with approval and status, the `--research-core` launch command); widget state reset when the saved file's SHA-256 changes; saves only after an owner edit (`on_change`); launch re-reads the draft, rebuilds the plan and requires the identical plan id and its stored approval; chips show the value itself. |
| `scripts/ifvg_funded_comparison_results.py` `selected_context`, `_comparison`, `_detail`, `_accounts`, `_open_trade_review` | One firm selector drives the ranking and the detail (no tabs); firm, configuration and account kept per saved result id (`funded_comparison_v1_selected_context`); widget keys scoped to the result; "Review these trades in Trade review" carries plan, result, configuration, firm and account. |
| `.../presentation/funded_comparison.py` `_instant` | Trades and account events ordered by recorded instant, not by text. |
| `scripts/ifvg_search_review.py` `render_trade_review` | Trade review → Study executions adds funded comparisons of the application's registered store; an unavailable funded target is refused with a message, never replaced by another study. |
| `.../presentation/funded_trade_review.py` (`REVIEWABLE_STATUSES`, `funded_review_sources`, `pair_trades`, `funded_trade_facts`, `review_keys`, `plan_strategy_package`, `strategy_trade_links`); `scripts/ifvg_funded_trade_review.py` (`render_funded_trade_review`, `_back_to_result`) | Lists Completed, Incomplete and Failed-with-result comparisons (labeled); recorded funded path on the plan's bound verified study bars; gap zones only for configurations of the verified study; funded review-ledger keys (result, pair, account, trade); Back restores configuration, firm and account. |
| `.../presentation/chicago_time.py` (`utc_instant`, `chicago_label`, `chicago_wall`, `style_chicago_axis`) | Shared Chicago display helper: 12-hour AM/PM, CST/CDT named, naive times refused unless their convention is named, chart positions to the microsecond. |
| `tests/agents/test_funded_comparison_saved_drafts.py`, `tests/agents/test_funded_trade_review.py`, `tests/agents/test_funded_comparison_screen.py` (updated from tab to selector assertions) | Repair tests for R1–R3. |

### Strategy-Core seams used (pinned 7c7111e, read only)

| Core anchor | Use |
|---|---|
| `strategies/ifvg_smc/replay.py` `DayOrchestrator` (`on_higher_tf_bar`, `on_decision_bar`, `finalize_day`, `finalize_dataset`, `end_seed`) | Stepped exactly as `run_day` does (equivalence tested). |
| `strategies/ifvg_smc/reducer.py` `IfvgReducer.execution_block_reasons` | Called by Core at the entry fill; the driver appends the account refusal reason. |
| `reducer.py` `_setup` (single slot, phase `S4` watching entries, `S5` in trade) | Cleared after a liquidation candle or a refused entry. |
| `reducer.py` stop-first per candle, entry at the confirming close | Matched by the account walk's candle alignment. |

### Earlier funded lane (budgeted mode, September 22 pilot)

`src/alpha_lab/propsim/funded/{profiles,clock,paths,positions,instance,campaign,result,plan,sources,price_evidence,runner}.py`,
`scripts/ifvg_funded_study.py`, `scripts/ifvg_funded_payout_job.py`, `scripts/ifvg_funded_results.py`,
`presentation/funded_results.py`, `funded_review_package.py`. Unchanged apart from the
earlier-mode label on its screen. `profiles.py`, `clock.py`, `instance.split_gross`,
`positions.boundary_offset_ticks`, `price_evidence.load_print_day`, `sources.open_verified_package`
and `runner.{read_state,write_state,append_ledger}` are reused by the comparison mode.

---

## Earlier inspection notes (September 22, 2026 package; historical)

Prepared September 22, 2026. Inspection notes, not a claim that changes are implemented.

## What was inspected

1. The connected public repositories `thealgochef/Quant-Lab` and `thealgochef/Strategy-Core` were located through the connected GitHub tool.
2. Quant-Lab's `main` architecture document describes the older June workbench. Its remote `platform-refactor` branch points at `58cbaa2c368abfe86cdde146d9c18450739ff211`, dated August 11, 2026. This is not assumed to match the newer local application.
3. The actual source files listed below were read from the `source/policy/` snapshot in the supplied `ifvg_no_entry_drought_40a54b336722dcd9d2bbaac63e8e09f731829671664c66a5f36a906ff34d8716.zip`.
4. The later gap-invalidation archive's `implementation/source_map.json` was also read. It identifies the normal launcher as `scripts/run_ifsm_research_ui.py`, the actual screen as `scripts/ifsm_research_ui.py`, Quant-Lab's local root as `C:/Users/gonza/Documents/Claude-Quant-Lab`, and its then-imported Core tree as `C:/Users/gonza/Documents/Strategy-Core-gap-invalidation`. These are historical inspection anchors, not instructions to reset or edit an old checkout.
5. The older Library document `QUANT_LAB_FSM_PROP_SEARCH_DASHBOARD_IMPLEMENTATION_PLAN_V1.md` was used as architectural background. It is explicitly a plan, not evidence all its proposed features exist. Its evaluation-stage scope and old research priorities are superseded for the new funded-only lane.

The currently running daily-close/session study and any unpushed working-tree edits were not inspected. The implementer must map the current imported files and preserve that running study before modifying code. Do not ask the owner to identify the functions.

## Existing Quant-Lab surfaces and observed gaps

Paths below are relative to Quant-Lab. Line references describe the inspected snapshot, not future line numbers.

| Existing file | Observed responsibility | Required action |
|---|---|---|
| `scripts/ifsm_research_ui.py` | `main()` launches the existing IFVG workspace after asserting its Core import. | Retain the launcher and established workspace. Do not create an unrelated dashboard. |
| `scripts/ifvg_workspace.py` | `render_study()` routes completed search studies to `ifvg_research_results.render_search_results()` and completed pipeline studies to `ifvg_research_pipeline.render_pipeline_results()`. | Add the payout-oriented final view to both completion paths through one shared presenter. |
| `scripts/ifvg_research_results.py` | Existing `metric_cards`, `render_bundle`, `_prop_details`, `_comparison`, and trade-review links. | Keep reusable navigation, replace primary metric hierarchy, and remove technical internals from user-facing results. |
| `scripts/ifvg_research_pipeline.py` | `render_pipeline_results()` and broader study orchestration screens. | Render the same immutable payout-result presentation; do not duplicate calculations here. |
| `scripts/ifvg_study_wizard.py` | `_step_prop`, `_step_risk`, objective/validation/review steps and `_freeze_and_launch`. | Expose funded start, account budget, position/exit, withdrawal and growth choices. Confirm actual current wizard entry because the workspace also references `ifvg_research_wizard`. |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` and `presentation/` | Human-readable settings, child-result explanations, workspace names. | Reuse translations; introduce payout-specific result models and plain-English labels. |
| `src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py` | `_run_prop_modes`, `_stage_s12_prop_historical`, `_stage_s13_bootstrap_stress`, `_stage_s14_frontier_insights`, `_stage_s15_verify_publish`. | Attach new funded-only runs, common budget, appropriate validity gates, immutable presentation and export. Current `_run_prop_modes` iterates `context.gates_passed`; old minimum-trade/smoothness gates must not silently remove candidates from the new funded-payout objective. |
| `src/alpha_lab/agents/data_infra/ifvg/search/store.py` | `save_envelope_immutable`, `load_verified_envelope`, typed sidecars and verified reuse. | Extend existing persistence; preserve old identities. Create the review folder from the verified completed result, not from screen state. |
| `src/alpha_lab/propsim/account.py` | Existing account state, fee, breach, payout and replacement events; `AccountWalk`. | Extend, rather than replace with a disconnected simulator. In the inspected source `_start_account()` always starts in evaluation, replacements call it again, and positive payout-processing delays are refused. Add explicit funded initialization/replacement and a true pending/received payout lifecycle. |
| `src/alpha_lab/propsim/firm_contracts.py` | Versioned firm terms, phase rules, payout conditions, source and verification status. | Preserve this separation. Extend unsupported funded details such as precise starting/fixed floor, payout sequence limits, retirement, phase-specific size tiers, eligibility counters and actual receipt timing. Do not fake a passed evaluation. |
| `src/alpha_lab/propsim/contract_evidence.py` | Source evidence compilation and contract verification/supersession. | Reuse for dated first-party rules and explicit owner-defined scenarios. Do not require the owner to research public rules again. |
| `src/alpha_lab/propsim/risk.py` | Fixed contracts, fixed dollars, fractions of remaining allowance, optional daily stops and post-payout sizing. | Extend only missing sizing/after-payout behavior. Distinguish firm-required limits from optional conservative buffers. Do not silently resize a five-contract profile into a three-contract profile. |
| `src/alpha_lab/propsim/withdrawal.py` | Trader withdrawal choices separate from what the firm permits. | Reuse. Add an explicitly defined large-first-payout objective, amount meaning before/after split, pending-request behavior and after-payout continue/reduce/retire choices. |
| `src/alpha_lab/propsim/portfolio.py` | Runs copied accounts on the same ordered day blocks, not independent per-account draws. | Preserve shared-market dependence. Current whole-account loops do not supply one chronological shared cash budget; add an event-time campaign coordinator. |
| `src/alpha_lab/propsim/search_bridge.py` | Builds account simulation inputs and persists results. | Replace or extend the relevant funded lane with account-aware event replay. Inspected `_validate_modes()` explicitly refuses `historical_ordered_event_replay` at this seam. Do not imply an exact ordered-price path is already wired. |
| `src/alpha_lab/propsim/trade_path.py` | Explicit closed-trade, unordered one-minute, assumed within-candle, and ordered-price evidence classes. | Reuse fidelity checks; do not promote assumed ordering into actual history. |
| `src/alpha_lab/propsim/simulation.py`, `calendar.py`, `stress.py`, `prop_metrics.py` | Simulation identities, date clocks, scenarios and account metrics. | Add the funded operation's budget calendar, cash objectives and valid sensitivity reporting. Do not infer future payout probabilities from a single replay or copied accounts. |
| `src/alpha_lab/propsim/event_detail.py` | Persisted, linked account-event evidence. | Reuse internally; expose only compact decision-relevant records in the review folder. |

## Existing Strategy-Core surfaces

Paths are relative to the actually imported Strategy-Core checkout.

| File | Required boundary |
|---|---|
| `src/strategy_core/strategies/ifvg_smc/reducer.py` | Retains the strategy's actual sequential confirmations and setup/position ownership. Integrate explicit external account exits or entry acceptance only through a tested contract. |
| `.../replay.py` | The common chronological market clock and completed-bar delivery; cooperate with per-account execution/exit events without future information. |
| `.../section.py` | Versioned strategy and permitted exit parameters, including point-target choices only when explicitly enabled. |
| `.../state.py` | Save and resume the strategy's correct per-account state; retain pending higher-timeframe delivery and daily-close state. |
| `.../records.py` | Actual entries, fills, exits and origin references required by the payout simulation. |
| `.../gap_validity.py` | Preserve the later tested own-timeframe closing option. No unrelated change is needed for this feature. |

Generic order/position/risk interfaces may exist elsewhere in the current tree. Inspect and reuse them before introducing a strategy-specific duplicate. Firm withdrawal rules and the global purchase budget belong in Quant-Lab, not in the pattern reducer.

## Proposed additions, not files claimed to exist

After the current-tree reconnaissance, fill gaps using small modules such as:

- `alpha_lab/propsim/campaign.py`: shared event clock, account allocation and campaign lifecycle.
- `alpha_lab/propsim/cash_ledger.py`: contribution allowance, purchases, receipts, personal withdrawals and retained cash.
- `alpha_lab/propsim/payout_lifecycle.py`: request, review, payment, rejection/cancellation and associated balance movements.
- `alpha_lab/agents/data_infra/ifvg/presentation/funded_results.py`: one presentation model for completed results.
- `scripts/ifvg_funded_results.py`: final-screen rendering only.
- `alpha_lab/agents/data_infra/ifvg/review_package.py`: strict data-and-document-only export allowlist.

Use different names if an equivalent module already exists. Do not build a second account framework merely to match this proposed directory list.

## Important implementation findings

1. There is already substantial account-simulation infrastructure. This is an extension and integration project, not a blank-slate rewrite.
2. The inspected account engine principally walks completed trades and optional adverse/favorable excursion scenarios. That does not establish exact intraday loss-limit enforcement.
3. Merely changing `_start_account()` to funded is insufficient: independent account loops, processing delays, shared replacement credits, earlier forced exits and changed strategy opportunities also need proper treatment.
4. A funded-only lane must not be blocked by optional historical strategy-score gates that embody the old high-frequency objective. Data correctness and mandatory operating rules still apply.
5. Test and verification code stays in the internal engineering repository/store. The requested automatic review folder must contain no scripts, patches or source-code snapshots.
