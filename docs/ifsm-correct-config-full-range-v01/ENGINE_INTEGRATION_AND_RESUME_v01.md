# Narrow amendment — integrate existing capabilities, then resume the same six-configuration batch

## 1. Authorization and precedence

When the owner sends this amendment to the implementation agent, it authorizes the minimum compatibility work in Strategy-Core and Quant-Lab needed to execute the already requested six configurations with passive MenthorQ annotation. It is not evidence that the integration has been implemented or verified.

Read the existing `TASK.md`, `RUN_REQUEST.json`, `date_scope.json`, and `source_references.json` in this folder. Continue their existing task and progress records; do not start a second competing implementation session or erase the blocked preflight findings.

This amendment supersedes TASK §3's requirement to use an already compatible engine, and its stop restriction where the ONLY missing authority is to integrate the existing partial-exit and passive-context implementations. Minimal Core edits, necessary schema/record/checkpoint adapters, supported runner bindings, and their tests are expressly permitted. TASK §5's annotation requirements and §7's single-batch restriction remain in effect, subject to the focused integration checks below.

The reported absence of an engine containing both capabilities is the starting problem to resolve, not a reason to stop again merely because Core edits are needed. This is permission to combine existing behavior, NOT permission to invent trading behavior or bypass compatibility, identity, approval, or data guards.

## 2. Unchanged experiment

Complete ONE batch, frozen before observing its new economic outcomes:

| ID | Saved configuration | Schedule | Opposing-pattern distance | Exit | Quantity |
|---|---|---|---|---|---|
| C01 | `S1-T1-H14-P1-L-SO` | All permitted market hours | 80 ticks / 20 points | Half at the first target, remainder at entry-price stop or daily close | 10 micros |
| C02 | `S0-T1-H14-P1-L-SO` | Original three windows | 80 ticks / 20 points | Same partial exit | 10 micros |
| C03 | `S0_D80_W1_P1` | Original three windows | 80 ticks / 20 points | Entire position at the first target | 1 mini |
| C04 | `S1_D80_W1_P1` | All permitted market hours | 80 ticks / 20 points | Entire position at the first target | 1 mini |
| C05 | `S0_D160_W1_P1` | Original three windows | 160 ticks / 40 points | Entire position at the first target | 1 mini |
| C06 | `S1_D160_W1_P1` | All permitted market hours | 160 ticks / 40 points | Entire position at the first target | 1 mini |

Use all six complete saved effective configurations, not this summary alone. Keep the finite 240/90 processed-one-minute-bar waits, own-timeframe closing invalidation, selection cap two, one-minute supporting charts and all other specified settings. Do not add the old document-default baseline, substitute whole-position exits for C01/C02, or launch only the four easier configurations.

Reuse the exact 253 evaluated dates, June 16, 2025–June 10, 2026, and the ten excluded warmup dates in `date_scope.json`. Keep its complete ordered membership, source contracts, roll inclusion, missing dates, calendar and cutoff. Each configuration continues through the entire history; no January reset and no borrowed old-baseline state.

Produce the six ordinary strategy results and twelve separate account-driven funded results, comparing TakeProfitTrader and MyFundedFutures independently. Existing account rules, acquisition costs, quantities, trading fees, price-proxy disclosures, full-surplus withdrawals, processing calendar and daily closing deadlines remain unchanged. One funded account at a time per configuration/firm, paid replacements, no copying, monthly credits, growth or reinvestment.

## 3. Integrate only what this batch needs

Use the existing compatibility findings to identify the exact source versions, imports and interfaces that do not fit together. Preserve relevant uncommitted work. Do not repeat a broad repository or market-data investigation just to rediscover a blocker already established.

Prefer the smallest behavior-preserving implementation. Passive annotation may be attached through an observational adapter outside the trading decision path when that preserves the requested records and existing point-in-time lookup rules. Core edits are authorized when needed to support the combination correctly. Do not implement both approaches merely to compare them.

Reuse the already tested partial-exit behavior and existing MenthorQ lookup/annotation implementation. Do not replace either with a simplified rewrite. Work in an isolated, task-owned checkout/runtime, preserving existing engines, global/default pins and historical artifacts. Bind every batch worker to the same frozen compatible source; a notebook/import-path workaround that silently mixes incompatible modules is not acceptable.

MenthorQ remains:

- `menthorq_context_version = menthorq_eod_v1`
- `regime_gate_policy = off`
- `nearest_support_gex1_block = false`
- `regime_unknown_policy = allow`
- `nearest_support_universe = all_19`

Retain prior-day source availability from 6:00 AM to 5:00 PM Chicago and explicit unknown/outside-hours records. Annotation must not select, reject, delay or resize a trade; alter setup state; change a stop; or affect account admission. No new active filters, target caps or model features are needed. Annotation must cover the actual relevant ordinary and funded streams separately, not attach the unrestricted reference's context to a different funded execution.

Using an external observational adapter, if chosen, changes only where the annotation is attached. Record the exact mapping of the above settings to that adapter and bind its identity to the new result. Do not pretend unsupported Core fields were accepted, or describe a run with missing annotation as satisfying the annotation requirement. A context coverage problem is reported independently; it is not permission to remove or modify trades.

## 4. Verify compatibility without creating another research campaign

Use existing tests, deterministic synthetic paths and already authorized, bounded regression fixtures. Any reference comparison must use the same input, initial state, calendar and execution convention on both sides. These are engineering checks, not another full-range study or parameter sweep.

Before launching, demonstrate:

1. **Whole-position preservation:** representative fixed-target behavior on the integrated runtime agrees with its authoritative existing implementation.
2. **Partial-exit preservation:** representative partial-exit behavior agrees with the authoritative half-exit implementation: ten enter, five exit, the remaining five retain the position slot and correct stop, and fees apply to actual filled quantities once. Cover a target then stop within a minute, a protective exit before any partial fill, and mandatory closing. Preserve the documented distinction between ordinary candle-based and funded recorded-price execution; do not force the two policies to become identical.
3. **Passive-context invariance:** in both exit families, annotation off versus on with all gates off preserves decisions, fills, quantities, timestamps, costs and account events on matched fixtures. Include missing context, outside availability and context that would conflict with the trade were a gate active. Added annotation and changed source/provenance identifiers may differ; list those exclusions explicitly rather than demanding byte equality of differently versioned artifacts.
4. **State and worker continuity:** checkpoint/resume and day boundaries preserve behavior, including an open remainder. Prove every actual worker receives its intended exit, full settings, dates, size, costs and imported source. Test save/reopen and the new approval/dispatch binding; identical display names or counts alone are insufficient.

Run focused affected-area tests, relevant existing Core/Lab regressions, and lint after the final functional edits. Use a broader suite where repository instructions or the actual dependency changes require it, but do not repeatedly rerun unrelated suites for wording edits. Record actual failures and skips; never force a predetermined count or weaken expectations to produce a pass. Keep the recent completed-close timing and partial-exit regressions intact.

Obtain one focused read-only review of the integration's actual changes and resolved conflicts, then address material findings and retest affected code. Do not build a large review bureaucracy or investigate the old default's drought as a prerequisite.

A reproducible behavior difference not explained solely by annotation/provenance is a real blocker to resolve. Do not call it harmless because profit is similar. Conversely, do not treat a new honest source hash or schema identity as a trading change by itself.

## 5. Freeze and launch; do not stop at the integration report

After the checks pass, resolve all six complete configurations against the final runtime. Record their historical source bindings, any schema translation, new source/configuration identities and exact differences. Preserve old approval records and create a new approval for the actual final plan through the established workflow, citing the owner's original task plus this amendment. Do not reuse an old approval identity or turn off launch validation.

Assert exactly six configurations and twelve configuration/firm outcomes. Display the resolved plan and proceed with the previously authorized single full-range batch without requesting another generic approval for the same scope. The permission here covers the necessary integration, its focused verification and that same batch—not additional experiments.

Reuse verified work already completed. Resume an identical interrupted job according to the supported checkpoint policy; do not duplicate completed jobs. Do not stop after saying the compatible engine now exists. Complete the batch, publish it in the existing Lab, and provide the review export required by TASK §9. No prediction of a minimum trade count or repetition of the old 107-day profit is required.

## 6. Preserve boundaries and report evidence

No new strategy rules, financial assumptions, configuration search, live activation, platform port, engine promotion, commit or push. No new data preparation/downloads, protected June 11, 2026 onward observations, 2021–2022 data, or broad recursive searches of raw-data directories. Use the existing guarded inputs and preserve all delivered source/result packages and owner records.

Retain the original export contents. Add only a concise `ENGINE_INTEGRATION.md` describing the actual incompatibility, chosen integration, affected files, source identities, behavior-preservation checks and remaining limits. Extend the existing `validation_summary.json` and `configuration_bindings.json` rather than creating duplicate evidence tables. Include actual commands, test scope, compared fields, failures/skips and final imported source. Keep access evidence separate from no-write hashes.

Because code changes may be required, supply a separate small source-review ZIP with the task-only patch, affected before/after source and focused tests/fixtures sufficient for review. Keep raw data, entire repositories, full console histories and environments internal. Check final ZIP manifests after extraction. Do not require fresh copies of earlier unchanged screenshots or financial exports solely to document the integration.

Stop only for a concrete unresolved conflict that would require changing the six strategies' behavior, protected scope or financial rules, or for an actual failed correctness check. Report the exact conflict and smallest additional decision needed. Do not stop solely because TASK §3 formerly prohibited the integration now expressly authorized.

**Finish line:** the compatible implementation is checked, the intended one-batch comparison has run over the unchanged full date range, its real completed/failed outcomes and limits are honestly reported, the existing Lab reopens the result, and the compact review evidence is delivered. Do not investigate or optimize any new drought automatically afterward.
