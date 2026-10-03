# Task A1 — implementation changes

Updated October 3, 2026. Branch in both repositories: `feature/menthorq-level-context`.
Verification receipts and limits are in the companion handoff files.

## Step 0 and Core pairing

- Quant-Lab started on main at `1b9d4077ffc4f68fab51c2ba9333f135e7308fd4`.
  Existing UI work was committed as-is: `2e3efd5d9921e1956d80144ee50d831956c7a360`,
  “Implement IFVG lab UI redesign and review fixes”, 253 files.
  Already on main, its explicit fast-forward merge reported already up to date.
  The push to origin main succeeded before branching.
- Core's clean full-repository worktree started on main at
  `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`; no pending work required a commit.
  The requested branch was created there. Its common repository's unrelated
  dirty platform-refactor worktree was preserved.
- Core implementation commit, published to origin's feature branch:
  `b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2`.
  Worktree: `C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\merge-alignment-20260922\Strategy-Core-main`.
- Quant-Lab's installed Core and prepared verified source now match that exact
  commit. Source identity:
  `afa16bed5733f6d41f090bc8b8e09215464cf9a009ec1c1bc7236973b76ead01`;
  source tree hash:
  `5a53ebeb40c5b5b45ce8d0311be050a23087d08b3803dc27b1df557fb9630559`.
  Only the mandated Core Git pin changes; package version remains 0.1.0 and
  no other dependency changes. Quant-Lab's implementation commit is
  `f4ee8454651abafd97f0b3a1714e41465cf44049`, published to origin's feature branch.
  Final repository heads/statuses are recorded at the end of `run_log.txt`.
  Neither feature branch is merged to main.

## Quant-Lab files

Paths below are relative to Quant-Lab.

| File | Change |
|---|---|
| `.gitattributes` | Apply the existing evidence `-text` pattern to exact A1 CSV fixtures and copied handoff receipts so Git does not convert their bytes; retain original receipt whitespace. |
| `src/alpha_lab/agents/data_infra/ifvg/menthorq_levels.py` | New CSV lookup, exact schema/date validation, content-hash parsed-source cache, aware Chicago availability, immutable Core snapshots, run-local prior cash close history and section provider factory. |
| `src/alpha_lab/agents/data_infra/ifvg/day_artifacts.py` | Optional runtime provider through existing level inputs; completed one-minute cash-close projection. |
| `src/alpha_lab/agents/data_infra/ifvg/capture_driver.py` | Optional provider handoff to replay; retain provider on capture result for direct report reuse. |
| `src/alpha_lab/agents/data_infra/ifvg/dataset.py` | Initialize provider once per enabled run and pass through existing v2/v3 capture helpers. Default-off path unchanged. |
| `src/alpha_lab/agents/data_infra/ifvg/menthorq_reporting.py` | New curated entry context, review CSV export, reconciled candidate/decision/trade grouping, comparison rows and verified saved funded cash consumer. |
| `src/alpha_lab/agents/data_infra/ifvg/experiment.py` | Existing native evaluator optionally attaches runtime grouped reports and session/regime/slot views using the capture provider. |
| `src/alpha_lab/agents/data_infra/ifvg/preparation.py` | Pass capture's provider to direct native report evaluation; write enabled review exports in preparation-job report folders outside immutable datasets, with optional scratch destination. |
| `src/alpha_lab/agents/data_infra/ifvg/search/axis_registry.py` | Four conditional Core-supported field axes and five atomic explicit-window presets; existing available/pending capability pattern. |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` | Assign new registry fields to existing Session Policy / Risk Admissibility presentation groups; keep all registered axes visible without an Other group. |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/strategy_rules.py` | Classify four A1 fields as optional rules with neutral legacy defaults; describe enabled context/gates in the existing entry group while retaining default descriptions. |
| `src/alpha_lab/agents/data_infra/ifvg/profiles.py` | Exclude only neutral A1 values from effective config projections, preserving historical settings shape while typed sections retain the fields. |
| `src/alpha_lab/agents/data_infra/ifvg/search/identities.py` | Use the same neutral map in name-free section hashes, preserving saved canonical profile recognition and hashing every active value. |
| `scripts/ifvg_strategy_approval.py` | Read displayed registry-axis baseline values from the typed section's JSON projection; keep neutral effective settings sparse. Owner-authorized follow-up after the second full run. |
| `pyproject.toml` | Exact Core re-pin, no other dependency changes. |
| `research/core/current.json` | Exact new commit, canonical LF source receipts and source identity. Historical manifest/bundle unchanged. |
| `tests/agents/data_infra/ifvg/test_menthorq_levels.py` | Lookup/cash-close/validation boundaries and source-fixture tests. |
| `tests/agents/data_infra/ifvg/test_menthorq_reporting.py` | Export, grouping/cash reconciliation, authoritative entry timestamp regressions and seven preparation export/destination/forwarding tests. |
| `tests/agents/ifvg_search/test_menthorq_axes.py` | Registry choices, atomic slot payloads and context dependency. |
| `tests/agents/data_infra/ifvg/test_menthorq_rules.py` | Nine pure rule-description cases, including exact unchanged default descriptions and enabled gate explanations. |
| `tests/agents/ifvg_search/test_menthorq_identity_compatibility.py` | Two pure compatibility tests for historical effective settings, canonical naming, registered typed axes and every valid nondefault value. |
| `tests/agents/test_ifvg_strategy_approval_ui.py` | One consumer compatibility case renders the actual approval table, checks all four neutral context defaults, and preserves sparse effective settings and the no-launch/no-approval review behavior. |
| `tests/agents/data_infra/ifvg/fixtures/menthorq_a1/eod_gamma_levels_daily_wide.csv.fixture` | Exact supplied fixture bytes; suffix avoids ignored generated CSV rule. |
| `tests/agents/data_infra/ifvg/fixtures/menthorq_a1/daily_total_dealer_gamma_and_regime.csv.fixture` | Exact supplied regime fixture bytes; portable tests do not require C:\tasks. |
| `AGENTS.md` | Current Core pairing and A1 entry points/limits. |
| `ARCHITECTURE.md` | Runtime provider, Core seam, reports/export and current pin. |
| `docs/ML_TRAINING_WORKBENCH.md` | Registered optional settings, causal review export and launch boundary. |
| `docs/README.md` | A1 page index entry. |
| `docs/pipeline_state.yaml` | Current source pairing and runtime context/export state. |
| `docs/IFSM_MENTHORQ_LEVEL_CONTEXT.md` | New lookup, availability, gates, export, reporting/pooling, parity and reproduction page. |
| `docs/IFVG_PLUGIN_DESIGN_ict_amended_ml_research_revised.md` | Ratified decisions 1–13 addendum; archived context/model scope deferred. |
| `research/core/README.md` | Current exact published feature commit, install/source preparation, historical identity preservation. |
| `handoff/task_a1/` | Requested changes/decision/question records, source receipt, parity/smoke/capacity/test logs, docs and test copies. No replay stores or builders are placed here. |

## Core files

Paths below are relative to Core; every file is in commit `b062bfc`.

| File | Change |
|---|---|
| `src/strategy_core/strategies/ifvg_smc/menthorq_levels.py` | Frozen snapshot, derived values and gate results; pure slot, nearest-level, normalization and gate helpers. |
| `src/strategy_core/strategies/ifvg_smc/section.py` | Four typed fields, named validation and explicit neutral profile exclusions. |
| `src/strategy_core/strategies/ifvg_smc/replay.py` | `IfvgLevelInputs` through existing replay level callback; default tuple supported. |
| `src/strategy_core/strategies/ifvg_smc/plugin.py` | Optional snapshot callback on existing `set_static_levels` seam. |
| `src/strategy_core/strategies/ifvg_smc/reducer.py` | Evaluate final-entry gates after existing execution/session/schedule reasons; retain blocked candidates without decisions. |
| `tests/test_ifvg_menthorq_levels.py` | Thirty pure/reducer/identity cases, including decision 12's Asia-entry case. |
| `docs/DECISIONS.md` | A1 fields and binding decisions 1–13, no A2 contracts. |
| `V3_COMPATIBILITY_MATRIX.md` | Optional runtime seam, neutral hashing and export status. |

## Final names and values

New modules are `ifvg/menthorq_levels.py`, `ifvg/menthorq_reporting.py`, and
Core `ifvg_smc/menthorq_levels.py`. New Core value types are
`MenthorqLevelSnapshot`, `MenthorqDerivedValues`, `MenthorqGateResult` and
`IfvgLevelInputs`. Public helpers include `load_menthorq_levels`,
`menthorq_provider_for_section`, `cash_close_points_from_artifacts`,
`derive_menthorq_values`, `slot_chicago_for`,
`evaluate_menthorq_entry_gates`, `record_context`, `write_context_export`,
`build_menthorq_reports`, `build_comparison_table` and `build_funded_cash_groups`.

The run-local lookup class is `MenthorqLevels`, with `snapshot`, `derived`,
`register_day_artifacts` and `prior_cash_close_for`. Runtime parameters and
attributes are `menthorq_provider` in Quant-Lab, `menthorq_for` on the Core
plugin's level callback, and `menthorq` on Core's level wrapper/step input.
Version constants are `FORMULA_VERSION=menthorq_eod_v1` and `SCHEMA_VERSION=1`;
`LEVEL_COLUMN_NAMES` fixes the exact 19-name source universe. None is a persisted
context record or seed field.

Preparation adds private `_write_preparation_context_export`, optional
`report_root` on both direct and persisted helpers, and the optional returned
`PreparedIfvgPair.context_export_path` (default null). Direct default review path:
`data/ifvg_preparation_jobs/<resolved_section.profile_name>/reports/<v2_artifact_id>/context_export.csv`.
Persisted preparation uses `<selected_job_root>/<job_label-or-profile>/reports/`
as the report root. Explicit relative roots resolve against the repository.
Default-off paths create no report folder. Enabled destinations must stay outside
selected immutable v2/v3 roots. The export's existing run-descriptor keys are
`v2_artifact_id`, `v2_manifest_payload_sha256`, `v3_artifact_id`,
`v3_manifest_payload_sha256`, `section_config_hash`, `evaluation_config_hash`,
`strategy_core_commit` and `strategy_core_source_tree_hash`; these do not create
a new identity contract or enter dataset manifests.

Section axes and payloads:

| Field | Default | Registered value suffixes / payload |
|---|---|---|
| `menthorq_context_version` | null | `none`: null; `eod_v1`: `menthorq_eod_v1` |
| `regime_gate_policy` | `off` | `off`, `positive_only`, `negative_only` |
| `regime_unknown_policy` | `allow` | `allow`, `block` |
| `nearest_support_gex1_block` | false | `false`: false; `true`: true |

Schedule values under existing `enabled_entry_sessions`:
`slot_s1_0830_1000`, `slot_s2_1000_1200`, `slot_s3_1200_1330`,
`slot_s4_1330_1510`, `midsession_1000_1330`. Each payload contains
`enabled_entry_sessions`, `entry_schedule_policy=explicit_windows_v1`,
`entry_schedule_timezone=America/Chicago` and `entry_schedule_windows`
together. Actual windows: 08:30–10:00, 10:00–12:00, 12:00–13:30,
13:30–15:10 and 10:00–13:30. All are half-open.

Named validation: `menthorq_gate_requires_eod_context`,
`unsupported_menthorq_context_version`. Named neutral exclusion map:
`MENTHORQ_NEUTRAL_PROFILE_FIELDS`. New ordered block reasons:
`regime_gate`, `context_unavailable`, `nearest_support_gex1`.
Availability reasons: `before_0600`, `after_1700`, `no_level_row`.
Gate statuses: `evaluated`, `context_unavailable`,
`not_applicable_outside_hours`, `not_applicable_shorts_enabled`.
Regimes: `positive`, `negative`, `unknown`. Slots:
`s1_0830_1000`, `s2_1000_1200`, `s3_1200_1330`,
`s4_1330_1510`, `outside_cash`. HVL side: `above`, `below`, `at`.

Review export columns, in order:
`candidate_id`, `decision_id`, `trade_id`, `availability_ts_utc`,
`entry_price_points`, `tick_size`,
`trading_date`, `source_eod_date`, `source_file_sha256`, `levels`,
`regime`, `total_net_gex`, `gex_percentile_1y`, `implied_move_points`,
`selected_instrument_id`, `roll_flag`, `context_available`,
`unavailable_reason`, `slot_chicago`, `hvl_side`,
`nearest_above_names`, `nearest_above_points`,
`nearest_above_distance_points`, `nearest_above_distance_implied`,
`nearest_below_names`, `nearest_below_points`,
`nearest_below_distance_points`, `nearest_below_distance_implied`,
`nearest_support_is_gex1`, `opening_move_signed`, `opening_move_abs`,
`regime_gate_blocked`, `nearest_support_gate_blocked`, `gate_status`,
`entry_session`. `levels` is JSON with all 19 names in source file order:
Call Resistance, Put Support, HVL, 1D Min, 1D Max, Call Resistance 0DTE,
Put Support 0DTE, HVL 0DTE, Gamma Wall 0DTE, GEX 1 through GEX 10.
The leading JSON comment contains `run_identity`, `formula_version`,
`schema_version`, both `source_file_sha256`, `archival=false` and
`enable_shorts`.

Grouping columns: `entry_session`, `regime`, `slot_chicago`.
Candidate/decision group counts use `count`. Execution metrics:
`trades`, `wins`, `win_rate`, `gross_points`, `net_points`,
`expectancy_points`; comparison additionally `configuration`,
`net_cash_cents` and `roll_flag_unavailable`.
Each `regime.<value>.<metric>` and `slot.<value>.<metric>` split repeats
those metrics and net cash; absent buckets have zero counts/points and null rates.
New funded cash columns are `firm_key`, the three group columns, `events`
and `net_cash_cents`. Smoke cash sentinel: `not_produced_in_a1`.

The optional native report attachment is `menthorq_report`; its report keys are
`pooling_policy`, `evaluation_scope`, `record_counts`, `executed_trade_groups`,
`total`, `open_unresolved_trades`, `roll_days`, `roll_trade_groups`, `comparison`,
`comparison_by_regime`, `comparison_by_slot`, `roll_flag_unavailable`,
`net_cash_cents` and `net_cash_note`. Saved funded grouping returns `groups` and
`totals_cents` with its explicit `pooling_policy`. `MenthorqGateResult` additionally
has `block_reasons`; these are existing candidate-evidence reasons rather than
new export outcome columns.

## Existing patterns and recorded choices

- Runtime lookup lives beside `day_artifacts.py`; Core stays pure. Sources
  cache by immutable bytes/policy, and cash-close history is isolated per run.
- Existing level callback accepts a wrapper only when enabled. No persisted
  day seed, immutable table, context feature registry, manifest or ML schema
  is widened. Default-off identity excludes only the four neutral values.
- Candidates/decisions use final-entry `envelope_ts_utc`, trades their
  `entry_ts_utc`; union-wide structural nulls are preserved. The initial
  smoke reporting failure and its corrected retry are preserved externally.
- Reports reuse post-warmup candidate-entry scope and existing priced execution
  validation, including scheduled-close realized points. The existing cost
  policy's 0.514 points is applied once; no new economics is defined.
- Funded cash follows the verified saved monthly ledger's event aggregation,
  exact cents, unknown session and explicit pooling. New smoke funded cash
  remains gated and uses the owner-approved sentinel.
- Missing instrument identity uses the owner-approved null roll/note fallback;
  no comparison exclusions or inferred rolls.
- Slot presets copy the full existing daytime composite, rather than separate
  mutable window axes. Existing schedule/session/holding semantics remain.
- Existing presentation groups hold the new axes: context version uses Session
  Policy because availability is scheduled; regime, unknown and support gate
  controls use Risk Admissibility. No new group or strategy policy is defined.
  The targeted suite exposed the missing map, and its existing presentation
  test verifies the correction without changing any expectation.
  Gate keys come from the registry's declared context dependency, avoiding a
  second copy of technical names in the presentation layer and retaining its
  existing source-guard contract.
- The final full run exposed the rule-description classifier's complete-field
  guard. The four new fields follow its existing legacy optional-default
  pattern, and private `_menthorq_rule` adds active explanations to the existing
  entry group. No baseline text, group name or policy is changed.
- The same full run exposed two other neutral serialization consumers. Effective
  settings and generated name-free hashes now reuse Core's explicit neutral map,
  following the existing default-exit omission pattern. Historical fixtures and
  identities are preserved; typed validation and registry axes retain all fields.
  These presentation/identity corrections leave the Core replay unchanged.
  The capture harness passes the unchanged typed section directly and does not
  use generated-profile naming or rule presentation. Existing parity, smoke and
  capacity receipts remain valid; their resolved configs retain the fuller
  effective projection that was actually recorded at capture time. They are not
  rewritten to pretend the later serialization fix had already been applied.
- Scratch harness uses the normal bounded v2/v3 capture builders with explicit
  destinations. No new output-root option is needed; default paths unchanged.
  Copied prepared caches keep their original full allowlist receipt for trust,
  while only the requested prefix is opened. No raw market-data read is needed.
- Ordinary preparation uses its existing job folder for the enabled review CSV,
  with the verified v2 ID separating report destinations. The optional
  `report_root` changes only that review destination and supports scratch callers.
  This closes the harness-only export gap without changing replay or datasets.
- Decision 13 excludes only proven context source identity fields for content
  comparison; all IDs and all other columns remain checked. Actual provenance
  is unchanged. Every per-table exclusion/reason is in `parity_report.md`.
- The existing first-30 calendar ends February 23 (January 19 and February 16
  are partial sessions), so the task's February 24 endpoint would mean 31 dates.
  The literal first 30 are used; no arbitrary session removal or 31st run.
- Source fixtures use `.csv.fixture` to preserve exact reference bytes outside
  the generated CSV ignore rule. Handoff contains only the requested light
  reports/receipts/tests/docs; builders, stores, capture traces and caches stay
  under the sibling external working folder.
  The existing evidence `-text` Git attribute preserves fixture and handoff receipt
  bytes across Windows checkouts; it does not alter source CSVs or runtime parsing.
- The final `run_log.txt` is a local generated command receipt, excluded only
  through this checkout's Git metadata. This lets it record the actual final
  pushed HEAD and clean status. Other requested handoff receipts are versioned,
  including explicit additions of the generated CSV and test/capacity logs.

No smoke metric interpretation, Task A2 work, charter creation, fitting, sealed
guard change, failure expectation repair, capacity adjustment or unrelated
dependency update is included.

## Verification status

- New tests: 54 Quant-Lab cases and 30 Core cases pass in their recorded focused
  cycles. Core's IFVG targeted suite passes all 368 cases. The Quant-Lab targeted
  run recorded 1,639 passed / 3 failed / 1 skipped; its new presentation-map
  failure was corrected and the affected existing file then passed all 23 cases.
  The other two targeted failures are in the known set below.
- Ten-date parity passes under decision 13. Both required smoke configurations
  completed over the recorded first 30 dates. The single all-on capacity gate
  passes with unchanged limits. The exact re-pinned Core `--check` passes.
- The first full run recorded 3,973 passed / 51 failed / 7 skipped.
  Forty-six failures and three extra skips traced to the three compatibility
  issues described above. Its original `test_run.log` is preserved unchanged.
- On October 3 the owner granted an exception to section 6 for exactly one
  additional full run, with no other code changes. The same command,
  `python -B -m pytest -q -p no:cacheprovider`, ran from Quant-Lab on the committed
  implementation snapshot. It started at 11:35:05 Chicago and finished at
  12:11:21; pytest reported 2,173.35 seconds. `test_run_2.log`, alongside the first
  receipt, records **4,029 passed / 16 failed / 4 skipped**. The pending-exception
  wording in `questions.md` item 8 describes the earlier state and is superseded
  by this authorization and result. Other handoff files are preserved as the
  owner requested.
- All 54 new Quant-Lab tests pass in the second full run; the existing verbose
  Core receipt proves 30 new Core cases pass. The requested total of 84 new
  passing tests is satisfied. The expected Quant-Lab acceptance count is
  `3,986 + 54 = 4,040`, plus the separately verified 30 Core cases gives 4,070.
  The second full run is short by 11 Quant-Lab passes and has 11 failures beyond
  the documented five. Acceptance was not met at the end of this second run.
  The subsequent owner-authorized fix and scoped acceptance are recorded as
  verification step 3 below.

## Second-run approval-table regression

All 11 additional failures are in
`tests/agents/test_ifvg_strategy_approval_ui.py`:

1. `test_review_requires_explicit_name_and_confirmation_without_writing`
2. `test_saving_exact_approval_enables_run_without_launch`
3. `test_single_configuration_approval_renders_and_saves_only_one_child`
4. `test_ready_legacy_owner_decisions_still_require_exact_study_approval`
5. `test_saved_approval_does_not_unlock_a_changed_study`
6. `test_missing_prepared_data_blocks_run_with_a_specific_explanation`
7. `test_changed_exact_settings_require_new_unchecked_confirmation[axes]`
8. `test_changed_exact_settings_require_new_unchecked_confirmation[dates]`
9. `test_changed_exact_settings_require_new_unchecked_confirmation[seed]`
10. `test_changed_reviewed_metadata_requires_new_confirmation`
11. `test_approval_failure_explains_the_actionable_reason`

The saved traces raise `KeyError: menthorq_context_version` in
`scripts/ifvg_strategy_approval.py:87`. Its baseline table indexes
`baseline.effective_config[axis]` for every displayed registry axis. Historical
effective settings correctly omit neutral A1 fields; the typed section retains
them. This approval-table consumer was missed when applying that compatibility
projection. The closest existing pattern is the registry's typed JSON section
projection: use `baseline.section.model_dump(mode="json")[axis]` for the table
value. The second-run closeout recorded this proposed fix without applying it
because that instruction explicitly said "No other changes." The owner then
authorized the exact fix and one consumer compatibility test; step 3 below
records its implementation and verification.

That second-run receipt-only closeout changed `CHANGES.md` and `test_run_2.log`,
in commit `d0fb7f6cb9d966ce9c42e652ce4f4b033cca78ea`. Both full-run logs retain
their original bytes. The owner's existing untracked `handoff.zip` is preserved
and is not committed or rewritten.

The five known full-suite failures remain untouched:

1. `tests/agents/data_infra/ifvg/test_catboost_bundle_model.py::test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged`
2. `tests/agents/ifvg_search/test_htf_cap_experiment.py::test_exact_four_effective_profiles_and_distinct_cap_identities`
3. `tests/agents/test_ifvg_capture_scheme.py::test_default_tags_regression_locked`
4. `tests/agents/test_ifvg_capture_scheme.py::test_default_times_round_trip_to_canonical_tags`
5. `tests/agents/test_ifvg_capture_scheme.py::test_list_profiles_default_first_and_upsert`

## Verification step 3 — approval-table consumer fix (targeted only)

After `test_run_2.log`, the owner authorized the proposed lookup exactly as
written: each displayed registry axis now reads
`baseline.section.model_dump(mode="json")[axis]`. This follows the registry's
existing typed-section projection. It restores the baseline table without
widening the historical effective configuration or changing approval policy.
No new module, field, value, reason, column or policy is introduced.

One new compatibility test,
`test_approval_baseline_table_retains_neutral_context_defaults`, uses the existing
Configure study AppTest fixture. It verifies the displayed null/off/allow/false
defaults, sparse effective settings, and review's unchanged no-approval/no-launch
behavior. Its source is copied to
`tests/Quant-Lab/tests/agents/test_ifvg_strategy_approval_ui.py` in this handoff.
The cumulative new-test inventory is now 55 Quant-Lab cases and 30 Core cases;
the earlier 84 passing cases retain their recorded receipts, and this run proves
the added case passes.

The grep-style search `rg -n 'ifvg_strategy_approval' tests -g '*.py'` found:

- `tests/agents/test_ifvg_strategy_approval_ui.py`: direct script imports and the
  AppTest consumer, including the new compatibility case.
- `tests/agents/ifvg_search/test_charter_day_threshold.py`: imports the approval
  UI module's shared fixture and helpers; the whole module is included.

The exact scoped command was:

```text
python -B -m pytest -v -p no:cacheprovider tests/agents/test_ifvg_strategy_approval_ui.py tests/agents/ifvg_search/test_charter_day_threshold.py
```

`test_run_3_targeted.log` records **35 passed / 0 failed / 0 skipped in 24.42
seconds**, exit code 0: all 12 approval UI cases (the prior 11 plus the new case)
and all 23 threshold cases pass. This is the third verification step and is a
targeted receipt, not a third full-suite run. The latest owner acceptance is met;
no full-suite pass count is inferred from this scoped run.

This display-only fix changes no Core or replay source, dataset contents,
manifests, gates, capacity limits, dependencies or known failing tests. The existing parity,
smoke, capacity and Core-pin receipts remain valid; none is rerun or rewritten.
Only the script, its one new test, this change record, the copied test and the
new targeted receipt change in this follow-up. Final feature heads and push
results are recorded in the refreshed local `run_log.txt`; Core remains at the
already published `b062bfc` implementation commit. Neither branch is merged to
main. All required handoff files are present.
