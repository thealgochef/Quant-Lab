# R6.1 — Files Touched

Reconciled against the release commit's `git show --name-status`
(**`6c0b60a`**, parent `f3f9ac2` = R5B.1; 87 files: 46 added = 28 src
modules + 1 script + 17 test files / fixtures; 41 modified = 21 src + 2
scripts + 14 tests / fixtures + `docs/DECISIONS.md` + the three staged
shared docs; +25,783 / −566), including the adversarial-fix round. The
commit is path-scoped to the files below (the three shared docs staged via
`stage_shared_docs.py`; `docs/ML_TRAINING_WORKBENCH.md` never committed).

## New source modules (28)

| File | Content |
|---|---|
| `src/.../ifvg/features/context_bar_panel_contract.py` | the frozen formula contract: `PANEL_AS_OF_POLICY_ID_V1` + registry, `PANEL_INTERVALS_SECONDS_V1 = (300, 900)`, `LOOKBACK_BARS = 12`, `MINIMUM_SOURCE_BARS = 13`, `STD_DDOF = 0`, partial-bar / warmup / lookback / reset / session-scheme / intensity-source / `session_of_bar_final_instant_v1` policies, the seven `cbp_*` features with formulas, typed missing reasons, the decision-28 stamps, `assert_panel_interval_registered` |
| `src/.../ifvg/features/context_bar_panel_materializer.py` | `ContextBarPanelArtifact{Payload,Envelope}` (store `context_bar_panels`; Arrow panel + manifest-listed validity sidecar), `materialize_context_bar_panel` (VERIFIED replay-chart artifact only; `bars_tf.parquet` re-read and rehashed), `compute_context_bar_panel_features` (13 complete source bars or all-null with exact offending-bar evidence; contiguity; warmup; formula-NaN stays valid), verify/save/load/reload |
| `src/.../ifvg/features/arrow_tables.py` | deterministic Arrow IPC bytes (pandas metadata stripped) + schema hash helpers shared by the new artifacts |
| `src/.../ifvg/fold_schedules.py` | `FoldScheduleEnvelope` / `derive_fold_schedule` (grain-agnostic 40/5/5/2 windows), `build_candidate_folds_from_schedule` (label-free), `build_context_bar_panel_folds` (panel-native; embargo 2; purge by bar span; stamped floor 300) |
| `src/.../ifvg/ml/fold_set_artifact.py` | the ONE legacy row-population `fold_set_id`, `FoldSetArtifact{Payload,Envelope}` (+ `fold_definitions.json` sidecar; stores `fold_schedules` / `fold_sets`), persist/load (rebuild + legacy-hash + window checks), `assert_same_fold_schedule` |
| `src/.../ifvg/ml/regime_oos_assignment.py` | the DESCRIPTIVE `RegimeOosAssignmentArtifact` (store `regime_oos_assignments`; Arrow sidecar), `candidate_as_of_frame`, the normative panel→candidate PIT rule `assign_panel_regimes_to_candidates` (same day, `<=` completed close, valid row, one-interval staleness, compatible partition, typed nulls incl. `panel_source_bar_incomplete`), `candidate_fold_oos_assignment`, the pattern-checked `candidate_as_of_source_ref` (`CANDIDATE_AS_OF_SOURCE_REF_PATTERN`; review F1), hashes, save/load/verify |
| `src/.../ifvg/ml/regime_sample_adequacy.py` | `preview_sample_adequacy` (per-fold training-row facts under the stamped floor; honesty fields; non-previewable gates) |
| `src/.../ifvg/ml/regime_observation_source.py` | `RegimeObservationSourceRef`, `VerifiedRegimeObservations` (ids from the loaded envelope only; observation-matrix hash), `load_regime_observations`, `assert_source_matches_protocol`, `run_regime_protocol_from_source` |
| `src/.../ifvg/ml/regime_executor.py` | `execute_regime_protocol` — persist protocol → load fold set → load observations → run → fits by verified reuse → assessment → the descriptive OOS artifact (candidate grain or the panel PIT rule); `candidate_as_of_source` is a verified `RegimeObservationSourceRef` (`load_candidate_as_of_source`; a tuple / bare string / panel-kind ref is "not evidence"; the as-of instants come from the LOADED frame) — review F1 |
| `src/.../ifvg/ml/regime_study.py` | `RegimeStudyRequest` (frozen; exact authority refs for model-bearing requests only; `hard_id_encoding: HardIdEncoding`), `stage_plan_problems`, `regime_study_block_reason` (probe protocol + verified frozen authority), `resolve_study_protocol`, `verify_frozen_authority`, `derive_initial_regime_status`, `regime_evidence_as_of`, `build_s10_decisions` |
| `src/.../ifvg/search/pipeline_regime.py` | the regime stage bodies S05/S06/S08/S09/S10/S14/S15 + readiness/wiring helpers used by `search/pipeline.py`; S05 chart selection (`PANEL_CHART_SELECTION_POLICY_ID = panel_chart_lowest_core_replay_id_v1`, `select_panel_chart`) and binding (`_verified_panel_chart`: id + pair refused otherwise; the loaded chart id / policy / `panel_source_core_replay_id` / pair sha256 in the sidecar); S08 schedule from the candidate view's observed days (`trading_days_source`, `days_without_labels`); S09a passes the PERSISTED bundle-view ref — reviews S1/F13/F7/F1 |
| `src/.../ifvg/ml/regime_supervised_stage.py` | S09b (fold-local features persisted) + the status-gated activation + S09c (controlled regime study / cohort model) for model-bearing requests |
| `src/.../ifvg/ml/regime_report_stage.py` | S14's regime half: `StratificationInputs` from persisted artifacts + exact simulation ids + `delivered_by` (this attempt's S09c results, else the run's OWN verified S09 record) + `run_scope`; the D15 loader seam; the panel-grain PIT event assigner verified-loading the OOS assignment → protocol → panel frame → every fit's assignment sidecar by exact id (review F14); prior-attempt report recovery; zero fitting |
| `src/.../ifvg/ml/regime_fold_features.py` | `RegimeFoldFeatureArtifact` (store `regime_fold_features`; fit k → fold k; fit-local `ctx_regime_<p12>_local_*`; canonical id reporting-only), `build_regime_fold_features`, the `RegimeFoldFeatureFrameSource` seam, verify/save/load; the ONE `hard_id_encoding` vocabulary (`HARD_ID_ENCODING_NONE` / `HARD_ID_ENCODING_CATEGORICAL = fit_local_categorical_v1` / `HardIdEncoding`) — review F11 |
| `src/.../ifvg/ml/regime_block_activation.py` | `activate_regime_context_block` / `regime_activation_resolution_payload` — the pure status-gated activation event bound to the exact frozen FEATURE_ELIGIBLE decision, owner artifact, assessment, protocol, fold-feature artifact (`RegimeActivationRefusalError`) |
| `src/.../ifvg/ml/regime_controlled_study.py` | `run_controlled_regime_study` (feature_only; both arms on identical `comparison_row_id`s; paired Brier / log-loss deltas; store `regime_controlled_studies`; `FEATURE_ELIGIBLE_REFUSAL`); `FrozenTree` / `FrozenJson` deep-immutable summaries (ids unchanged) — review F15 |
| `src/.../ifvg/ml/regime_cohort_model.py` | `run_regime_cohort_model_study` (cohort_model per fold × fit-local regime; one-fold fold sets on the pooled D13 identity; `insufficient_regime_partition{…}`; store `regime_cohort_model_studies`) |
| `src/.../ifvg/ml/regime_stratified_contracts.py` | the five classes, minimum statuses / interpretation / roles, `RegimeReportGate`, `RegimeAssignmentEvidenceRef`, `RegimeStratumKey`, per-class bodies, `RegimeStratifiedReport{Payload,Envelope}` (store `regime_stratified_reports`) |
| `src/.../ifvg/ml/regime_stratification_gate.py` | `resolve_report_gate(run_scope=)` — exact-ID loads of the decision / assessment / owner evidence; the structural D6 gate re-derived from the LOADED assessment; the owner branch re-runs `assert_owner_decision_authorizes` under the run scope; `RegimeStatusRefusalError` ("nothing here promotes") — reviews F2/S6 |
| `src/.../ifvg/ml/regime_assignment_sources.py` | verified OOS-artifact loads, `regime_for_trades` (exact one-to-one join), `regime_filter_mask`, `stratum_cohorts` (the first `RegimeFilterRef` consumer) |
| `src/.../ifvg/ml/regime_stratified_strategy.py` | `cohort_descriptive` bodies (`compute_strategy_metrics` per stratum; pooled == child; shares; thin strata; concentration flags; pooled-model skill by regime) |
| `src/.../ifvg/ml/regime_stratified_frontier.py` | `stratified_frontier` bodies (child × regime × metric; `on_frontier` from the pooled frontier only; never a selection input) |
| `src/.../ifvg/ml/regime_stratified_prop.py` | `stratified_prop` bodies (source trade → regime; PIT for historical no-trade events; synthetic clocks never consulted; the D15 loader seam with `EVENT_TYPE_PRECEDENCE` + the event's own `trading_day`; `evidence_not_persisted`) and the streamed, budgeted `account_event_regime_summary.parquet` (schema v1, `account_event_regime_summary_budget_v1`, `EventRegimeSummaryBudgetError` before publication) — review F3/F9 |
| `src/.../ifvg/ml/regime_stratification_service.py` | `build_regime_stratified_reports` (persisted artifacts only; per-class refusals recorded; `delivered_by` verified by exact S09c reload; `run_scope` threaded to the gate; deterministic report ids), persist/load report + detail + the summary sidecar (rehashed) — review F6 |
| `src/.../ifvg/ml/comparison_rows.py` | D13 `comparison_row_id`, the `RegimeFoldFeatureSource` Protocol, the per-fold LEFT join, `with_comparison_row_ids`, default schedule / label ids |
| `src/.../ifvg/ml/catboost_bundle_model.py` | the bundle-aware CatBoost rung `ifvg_context_catboost_bundle_v1` (frozen-lane parameters by value; native NaN + `MISSING_CATEGORY`; registry ∪ block-declared categoricals; resolved protocol hash; fold-local fitting; permutation importance) |
| `src/.../ifvg/search/owner_decisions.py` | `OwnerDecisionArtifact{Payload,Envelope}` (decisions 25/28/29/30; deep-immutable values), store `owner_decisions`, store-owned supersession (hash-chained `SUPERSESSIONS.jsonl` + `SUPERSESSIONS.head`, appended BEFORE publication, stale-lock reclaim, provenance may never weaken, fail closed on any single-file tamper), `assert_run_scope_lawful_for_root` (P0-4 mirror; synthetic provenance refused at persist AND load in the research namespace), `assert_owner_decision_authorizes` (exact refusals), proposals with placeholders that cannot persist, the synthetic fixture — reviews S2/F8, S3, S10 |
| `src/alpha_lab/propsim/event_detail.py` | D15: policies / budgets / schema, the literally streaming ZSTD Parquet writer (one path block at a time, column-wise, straight into the publication directory; preflight + cumulative byte budget before publication; a 32-byte-per-row `event_id` uniqueness index), verified readers — review S12/F16 |

## New scripts (1)

`scripts/ifvg_regime_promotion.py` — the ratification CLI (`chain` with per-row `owner_evidence`, `propose`, `promote --to {stratification_ready, feature_eligible}` through `persist_regime_promotion(run_scope=…)`; no `--decided-at` — `decided_at` derives from verified artifacts so `stratification_ready` reuses S10's decision; `model_feature` refused with the exact shared reason; `(OSError, TimeoutError)` sanitized to exit 2; importing launches nothing).

## Modified source (21)

| File | Change |
|---|---|
| `src/.../ifvg/search/pipeline.py` | `PipelineSemanticSpecPayload.regime_study` + lawful-plan rules; readiness with `store_root` / `run_scope`; `PipelineWiring.context_bar_source`; `_RunContext.regime`; S02 verified reuse by reproduction for stratified reports — re-derived tables adopted ONLY when they reproduce the persisted costed evaluation of this cost policy, otherwise "reproduction unverifiable" (review S4); S04 chart ids; S05/S06/S08 (the labeled folds from the candidate view's observed days — review F7)/S09 (ladder conditional; D13 schedule/label binding; `catboost_bundle` pin)/S10/S14/S15 regime bodies; S12/S13 `account_simulations.json` sidecar + the D15 policy passthrough |
| `src/.../ifvg/ml/regime_contracts.py` | panel protocol tightening (pattern, registered intervals / policies), per-fold stability + transition fields, the renamed gate `minimum_bootstrap_aligned_ami_mean`, the panel / stratification / prop-attribution / hard-id stamps |
| `src/.../ifvg/ml/regime_service.py` | grain/bundle-key coherence, the single-threaded kernel wrapper (lazy `threadpoolctl` import, `THREADPOOLCTL_MISSING_MESSAGE` — review S9), per-fold canonical occupancy, panel-source coherence, registry-pinned bootstrap |
| `src/.../ifvg/ml/regime_diagnostics.py` | per-fold bootstrap (fold seeds; protocol-wide minimum), candidate-event vs panel transition semantics, `transition_matrix_from_sequence` deleted |
| `src/.../ifvg/ml/regime_store.py` | `persist_regime_fit` reuse by reproduction (`return_reuse`); `persist_regime_promotion(run_scope=)` with verified owner evidence from FEATURE_ELIGIBLE, D6 at persistence (STRATIFICATION_READY needs passing coverage gates + OOS), `MODEL_FEATURE_PROMOTION_REFUSAL`, the namespace guard, the monotone `decided_at` chain — reviews F2/S3/S5/S11 |
| `src/.../ifvg/features/feature_blocks.py` | `with_registered_block`, the panel block definition/resolution, `PRE_R6_1_*` exports, `source_kind` `replay_chart_bars`, import-time invariants |
| `src/.../ifvg/features/feature_bundles.py` | `BP0_CONTEXT_BAR_PANEL`, `B7_CORE_REGIME` |
| `src/.../ifvg/features/bundle_feature_view.py` | persisted views (`bundle_feature_views` store; `frame_table_sha256`), `regime_fold_feature_artifact_id`, `bundle_categorical_features` |
| `src/.../ifvg/ml/supervised_ladder.py` | the bundle rung dispatch, `comparison_row_id` parity / paired deltas, fold-local feature passthrough, bundle-path identity fields, the ONE legacy fold-set hash |
| `src/.../ifvg/ml/logistic_model.py` | `comparison_row_id`, fold-local features, block-declared categoricals |
| `src/.../ifvg/ml/controlled_feature_study.py` | three rungs on both arms; cross-arm identity on `comparison_row_id`; the legacy fold hash delegated |
| `src/.../ifvg/ml/model_protocols.py` / `study/study_cell.py` | the `ifvg_context_catboost_bundle_v1` registry entry |
| `src/alpha_lab/propsim/simulation.py` / `search_bridge.py` / `account.py` | D15 policy / storage / schema / budget fields in the simulation identities; the producer-based streaming writer wiring (`account_event_detail_producer`; `walk_summary.json` from the produced counts); `on_simulation_persisted` hook; `trading_day` on account events |
| `src/.../ifvg/search/charter.py` | `SimulationProtocol.event_detail_persistence_policy_id` (default `none_v0`) |
| `src/.../ifvg/search/identities.py` | audit enumeration of every new identity-bearing module |
| `src/.../ifvg/search/store.py` | ten new store names; the sidecar-producer protocol (`ProducedSidecar` / `SidecarProducer` / `write_produced_sidecar`; `save_envelope_immutable(sidecar_producer=)` runs the producer into the temporary publication directory, re-hashes by streaming, refuses lies/strays/ghosts; reuse re-runs it into a scratch directory) — review S12 |
| `src/.../ifvg/study_providers.py` / `study_status.py` | regime diagnostics / run facts / report index / auto-fill providers; the two §31 empty states |
| `scripts/ifvg_pipeline_tab.py` | Configure / Preview / Monitor regime surfaces (the Monitor's S14 table renders `delivered_by` rows beside report ids and refusals — review F6); Preview and the Launch handler call `derive_stage_plan_readiness` / `assert_stage_plan_launchable` with `store_root` + `_effective_run_scope` BEFORE `save_charter` / the `pipeline_specs` envelope / any spawn (review S7); the two-pass clone flow; Regime Lane `default_ids`; the CatBoost bundle caption |
| `scripts/ifvg_regime_panels.py` | auto-filled exact ids, the stratified-report input + per-class renderers, per-fold stability with the renamed gate, grain-specific transitions, the owner-decision view (no controls) |

## New tests (17) and modified tests / fixtures (14)

New: `tests/agents/ifvg_search/test_context_bar_panel.py`, `test_owner_decisions.py`, `test_pipeline_regime.py`, `test_browser_manifest_validator.py`; `tests/agents/data_infra/ifvg/test_fold_schedules.py`, `test_regime_oos_assignment.py`, `test_regime_sample_adequacy.py`, `test_regime_observation_source.py`, `test_regime_fold_features.py`, `test_regime_supervised_studies.py`, `test_regime_stratification.py`, `test_regime_stratification_gate.py` (adversarial round), `test_catboost_bundle_model.py`; `tests/propsim/test_account_event_detail.py`; `tests/agents/test_ifvg_regime_promotion_cli.py`; fixtures `ml_fixtures/synthetic_context_panel.py`, `synthetic_observation_source.py`.
Modified: `test_regime_contracts.py`, `test_regime_service.py`, `test_regime_store.py`, `test_supervised_ladder.py`, `test_controlled_feature_study.py`, `test_ml_registries.py`, `test_feature_blocks.py`, `test_identities.py`, `test_pipeline_job_script.py`, `test_study_status.py`, `pipeline_fixture.py`, `test_ifvg_pipeline_tab.py`, `test_data_infra.py`, `test_databento_provider.py` (hermetic).

## Docs

- `docs/DECISIONS.md` — D-047 + the reservation note (normal commit).
- `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` — R6.1 lane transforms via `stage_shared_docs.py` (user-hunk isolation as every release).
- `docs/ML_TRAINING_WORKBENCH.md` — untouched, uncommitted (user-owned).

## Implementation-progress files (this folder)

`PRE_R6_1_BASELINE.md`, `FILES_TOUCHED.md` (this), `DEVIATIONS.md`
(DEV-R6.1-1…18), `ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (+ the
raw reviewer reports `_review_contract.md` / `_review_safety.md`),
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `TEST_RESULTS.md`, `GATE_SUMMARY.md`,
`stage_shared_docs.py`, `r61_smoke_app.py`, `verify_browser_manifest.py`,
`build_browser_manifest.py`, `browser-smoke/` (25 screenshots +
`MANIFEST.json` v2 + `_screenshots.json` + `_smoke_server.log`),
`DRAFT_OWNER_DECISION_PROPOSALS/` (three drafts — two regenerated over the
smoke store), raw outputs `_midpoint_pytest.txt` / `_final_pytest.txt` /
`_final_pytest_keys_cleared.txt` / `_ruff_and_diffcheck.txt` /
`_surviving_shared_doc_diff.patch`, `R6.1.patch` + `.sha256`,
`_PROGRESS_CHECKPOINT.md` (the resume record); `../DECISIONS_TAKEN.md`
entries 76–100 (#82 / #85 / #86 / #88 amended in place).

## Never modified

The frozen M0–M3 lane (`ifvg/context_model.py` byte-unchanged; golden protocol
hash), Strategy-Core, Trade-Lab, every existing immutable artifact/catalog,
the plan package, the R5B.1 / R6 evidence folders. No new package dependency.
