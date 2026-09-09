# R6.1 — Independent adversarial review, Reviewer 1 (CONTRACT-FIDELITY lens)

Scope: the R6.1 working set relative to HEAD `f3f9ac2` (`git status --short` / `git diff --stat`, 2026-08-29), reviewed read-only against `R6.1_IMPLEMENTATION_PLAN.md` rev 3 (§5 D1–D15, §6.A–6.L, §7, §8, §9), `OWNER_PLAN_REVIEW_CORRECTIONS_2026-08-28.md` (corrections 1–7), `FINAL_PLAN_CORRECTIONS_2026-08-28.md` (rulings 1–8), the FINAL-IMPLEMENTATION-PLAN-DOCS and the kickoff rules; `R6.1/DEVIATIONS.md` challenged. Synthetic fixtures only; no real data path was read; the protected date / sealed range were never touched. Targeted suites run: `test_regime_fold_features.py`, `test_regime_oos_assignment.py`, `test_owner_decisions.py`, `test_fold_schedules.py` (40 passed) and `test_pipeline_regime.py` (12 passed, 121 s). Three adversarial probes were executed on tmp roots (scratchpad `probe_r61.py`; outputs quoted inline).

**Verdict: 0 blockers, 4 majors, 12 minors**

---

## Findings

### F1 — MAJOR — The panel-grain OOS-assignment artifact binds a caller string and an in-memory frame as its candidate as-of provenance

**Evidence.**
- `src/alpha_lab/agents/data_infra/ifvg/ml/regime_executor.py:71` — `candidate_as_of_source: tuple[str, pd.DataFrame] | None`; `:122` `reference, candidate_frame = candidate_as_of_source`; `:140` `as_of_ref = str(reference)`; `:156` `candidate_as_of_source_ref=as_of_ref` into the identity payload.
- `src/alpha_lab/agents/data_infra/ifvg/ml/regime_oos_assignment.py:169` — `RegimeOosAssignmentPayload.candidate_as_of_source_ref: str` (free string, no pattern, no load).
- `src/alpha_lab/agents/data_infra/ifvg/search/pipeline_regime.py:410-415` — S09a passes `(f"bundle_feature_view:{...}", context.bundle_frames[primary])`: the in-memory frame, not the verified-loaded frame of the persisted view.
- Probe B (synthetic `persisted_panel_source`): `execute_regime_protocol(..., candidate_as_of_source=("NOT_AN_ARTIFACT:this-is-a-caller-string", frame))` persisted `candidate_as_of_source_ref = NOT_AN_ARTIFACT:this-is-a-caller-string`; re-running with the same string and a tampered in-memory as-of frame (+1 s on every anchor) minted a second artifact (`0b44ba0b0cd5` vs `375bd8a58496`) under the same "reference".

**Why it violates the plan.** D2 ("`source_artifact_ids` always come from the loaded envelope, never a caller string") and §6.D ("Every bound id comes from a loaded envelope; **it never accepts a frame**"); kickoff rule "a caller-provided string is NOT evidence"; hunt item 9 (identity payload carrying a caller string as evidence). The module docstring (`regime_executor.py:10-11`) claims the opposite of what the signature does. `candidate_as_of_source_hash` binds the *values*, so the artifact is not forgeable in content, but its provenance line is unverifiable: nothing proves the as-of instants came from the persisted bundle view named in the ref.

**Failing scenario.** Any panel study whose S05 in-memory bundle frame diverges from the persisted `bundle_feature_views` sidecar (or any direct executor caller) produces a descriptive OOS artifact whose `candidate_as_of_source_ref` names an artifact that was never loaded; the stratified reports then cite an assignment whose anchor instants cannot be re-derived from the store.

**Fix.** Make the executor take a `RegimeObservationSourceRef`-style `candidate_as_of_source: RegimeObservationSourceRef` (kind `bundle_feature_view`), load it through `load_regime_observations`, derive `as_of` from the loaded frame, and stamp `candidate_as_of_source_ref` from the loaded envelope id (pattern-checked). Add a test that a bare string / in-memory frame is refused.

---

### F2 — MAJOR — D6 is not enforced on the store/CLI path: STRATIFICATION_READY persists over a failing-coverage assessment and the report gate then accepts it as `s10_structural`

**Evidence.**
- `src/alpha_lab/agents/data_infra/ifvg/ml/regime_store.py:551-561` — `persist_regime_promotion` only calls `assert_lawful_promotion(...)`; `regime_contracts.py:577` gates `gates_passed` from FEATURE_ELIGIBLE onward only; no check of `coverage.coverage_gates_passed` / `oos_assignment_available` for STRATIFICATION_READY.
- `scripts/ifvg_regime_promotion.py:156-175` — `promote --to stratification_ready` builds the decision and calls `persist_regime_promotion`; its docstring (`:10-11`) claims "structural gates re-checked at persistence" — they are not.
- `src/alpha_lab/agents/data_infra/ifvg/ml/regime_stratification_gate.py:98-103,146` — `resolve_report_gate` accepts any STRATIFICATION_READY decision and labels the gate `authority_source="s10_structural"` regardless of who minted it.
- The suite itself relies on the gap: `tests/agents/data_infra/ifvg/test_regime_store.py:331` persists `small_second` (STRATIFICATION_READY over the n=170 run whose coverage gates FAIL) without error.
- Probe A (n=170 synthetic run): `coverage_gates_passed=False`, `gate_failures=('sample_adequacy',)`; STRATIFICATION_READY persisted (`9c1c840fb43a`); `resolve_report_gate(..., cohort_descriptive)` **accepted** it with `status_at_report=stratification_ready`, `authority_source=s10_structural`.

**Why it violates the plan.** D6 ("DESCRIPTIVE_ONLY → STRATIFICATION_READY … now structurally requires `coverage_gates_passed` AND `oos_assignment_available`"; rationale "Stratifying over uncovered rows is misleading"), §6.F ("STRATIFICATION_READY may be produced deterministically by S10 … when its ratification-free structural gates pass"), §9.2 `test_owner_decisions.py` row "STRATIFICATION_READY needs coverage gates + OOS" — no such test exists (grep over `test_owner_decisions.py`, `test_regime_store.py`, `test_ifvg_regime_promotion_cli.py`). Only `build_s10_decisions` (`regime_study.py:526-531`) applies D6; every other minting path bypasses it, and the report gate cannot tell them apart.

**Failing scenario.** Owner runs `ifvg_regime_promotion.py promote --to stratification_ready` against a sample-adequacy-failing assessment; a later descriptive run (or the panel's stratified-report inputs) builds `cohort_descriptive` / `stratified_prop` / `stratified_frontier` reports over a protocol whose training folds never met the floor, badged `s10_structural`.

**Fix.** In `persist_regime_promotion`, when `decision.status is STRATIFICATION_READY` require `assessment.payload.coverage.coverage_gates_passed and assessment.payload.oos_assignment_available` (typed refusal); have `resolve_report_gate` re-derive the structural gate from the loaded assessment instead of trusting the status; add the §9.2 test (and fix `test_regime_store.py` (e) to build its failing-assessment chain through a BLOCKED state).

---

### F3 — MAJOR — D15's report-local `account_event_regime_summary.parquet` is absent; S14 instead embeds every prop event as row-oriented JSON in the report detail sidecar (unbounded, unbudgeted)

**Evidence.**
- `grep -rn account_event_regime_summary src/ scripts/ tests/ docs/ ARCHITECTURE.md` → no hits.
- `src/alpha_lab/agents/data_infra/ifvg/ml/regime_stratified_prop.py:303-307` — `detail["simulations"][simulation_id] = {..., "events": events.to_dict(orient="records")}` for every simulation (bootstrap/stress included); `regime_stratification_service.py:271-285` persists that `detail` as `stratified_report_detail.json` (`STRATIFIED_REPORT_DETAIL_SIDECAR`) with no row/byte budget.
- `R6.1/DEVIATIONS.md` records no deviation for this.

**Why it violates the plan.** §6.G "Prop-event detail — versioned, compressed and bounded (D15)": "S14 streams the verified partitions and persists a report-local `account_event_regime_summary.parquet` keyed by the exact regime-assignment evidence; this summary is part of the stratified-report identity"; D15 rationale "Supports exact attribution without unbounded row-oriented JSON or silent storage explosion"; owner correction 6. The implementation re-creates, one artifact downstream, exactly the representation D15 was approved to eliminate: a bootstrap simulation persisted under v2 (bounded ZSTD Parquet, ≤10 M rows) is re-serialized in full as JSON records inside every `stratified_prop` report.

**Failing scenario.** A `stratified_prop` report over one child with the default bootstrap `n_paths` produces a `stratified_report_detail.json` of (paths × events per path) JSON records — tens to hundreds of MB — with no preflight, no budget, no Parquet, and the `detail_sha256` binding a sidecar that a browser panel cannot load.

**Fix.** Replace the per-event JSON with the planned Parquet summary (per path / per regime × event type counts and sums; manifest-listed sha256/bytes/rows/schema; a registered budget) keyed by the `RegimeAssignmentEvidenceRef`; keep only aggregate facts in the JSON detail. Record the interim state in DEVIATIONS if the summary is deferred.

---

### F4 — MAJOR — The §9 test claims are under-delivered: `test_s14_performs_zero_fitting` proves zero fitting only for a descriptive run (three reports, never five), `panel_supervised` is never run end-to-end, and "supervised classes without S07 refused" is untested

**Evidence.**
- `tests/agents/ifvg_search/test_pipeline_regime.py:438-502` — the fit-forbidden monkeypatch wraps S14 of `build_pipeline_fixture(tmp_path, regime_study="candidate")` (descriptive; S09c never runs); `:490-495` asserts `reports_by_class == {cohort_descriptive, stratified_prop, stratified_frontier}` and `refusals == {}`. The model-bearing E2E (`:505-614`) asserts only the self-reported flag `reports["fitting_performed"] is False` (`:611`) — no estimator is monkeypatched there.
- `tests/agents/ifvg_search/pipeline_fixture.py:235` declares the `panel_supervised` shape; `grep -rn panel_supervised tests/ scripts/ R6.1/*.py` finds no consumer — no test or smoke phase runs a supervised panel study through S05–S15 (the only panel-grain fold-feature proof is the unit test `test_panel_grain_uses_fit_k_and_the_candidates_own_partition`).
- `regime_study.py:179-189` (the S07/label-policy rule for supervised classes) — `grep -rln "cohort_model regime studies require|require 07_derive_labels" tests/` → no test.

**Why it violates the plan.** §9.1 row `test_s14_performs_zero_fitting`: "with `KMeans.fit`, `LogisticRegression.fit`, `CatBoostClassifier.fit` (and the sklearn `Pipeline.fit`) monkeypatched to raise during S14, the stage completes and persists **all five reports**"; §9.2 `test_pipeline_regime.py`: "16 terminal states for descriptive candidate / descriptive panel / supervised candidate / **supervised panel** studies; … **supervised classes without S07 refused**"; §11.3 "supervised candidate/panel studies … S14 performs zero fitting". The zero-fitting proof is run precisely on the shape where nothing could fit anyway; the shape where the temptation exists (feature_only/cohort_model present) is only self-attested. Hunt item 11.

**Failing scenario.** A regression that re-fits a rung inside `regime_report_stage` for a model-bearing run (e.g. rebuilding the cohort-model paired deltas at S14) would pass the entire suite. A panel-grain supervised run (S09b PIT fold features + activation under the panel protocol id, whose readiness cannot pre-verify the protocol — `regime_study.py:457-461`) has never been executed through the pipeline.

**Fix.** Parametrize the zero-fitting test over `candidate_supervised` and `panel_supervised` (monkeypatch inside S14 as today); add the `panel_supervised` 16-state E2E; add the S07-refusal test on `PipelineSemanticSpecPayload` / `regime_study_block_reason`; align the assertion with DEV-R6.1-10 (three envelopes + two S09c studies) explicitly.

---

### F5 — MINOR — Test names/claims that exceed what is asserted (hunt 11)

- `tests/agents/data_infra/ifvg/test_regime_supervised_studies.py:422-424` — `test_arms_share_rows_labels_folds_on_comparison_row_id_with_distinct_views` asserts `baseline.view_id == challenger.view_id`; the module docstring (`:3-5`) says "arms with DIFFERENT `view_id`s". The distinct-view proof lives only in `test_catboost_bundle_model.py:187-243` via a synthetic `_CHALLENGER_VIEW_ID` override.
- `test_regime_supervised_studies.py:106` — `_STUDY_PROTOCOLS = (PREVALENCE, LOGISTIC)`: the controlled regime study is unit-tested without the CatBoost bundle rung, although §6.G feature_only specifies "rungs prevalence + logistic + CatBoost bundle" (the rung only runs implicitly in the pipeline E2E).
- §9.1 `test_decision_25_algorithm_snapshot_is_authorized` does not exist by name; its content is covered by the parametrized `test_decision_25_28_29_30_values_are_verified_against_registry_protocol_assessment` (`test_owner_decisions.py:172-204`, cases `algorithm_key`, `algorithm_parameters_hash`, `algorithm_parameters`). Not recorded in DEVIATIONS.

**Fix.** Rename/split the tests to match their assertions; run the regime controlled-study unit test over `DEFAULT_BUNDLE_LADDER_PROTOCOLS`; record the §9.1 name mapping.

---

### F6 — MINOR — S14 records the delivered modeled classes as "refusals" (DEV-R6.1-10) and the plan's five-report claim becomes three

**Evidence.** `regime_stratification_service.py:286-295` — `feature_only`/`cohort_model` always land in `refusals` with the text "… not by the descriptive report service; nothing here promotes"; `regime_report_stage.py:205` persists that dict in `regime_stratified_reports.json`.

**Why.** §6.E S14 row and §9.1 speak of five report envelopes whose identity includes the decision id; DEV-R6.1-10 records the swap, but the recorded state uses the *refusal* vocabulary for classes that succeeded at S09c, so a consumer of the S14 record (UI `regime_status_below_minimum` / refusal renderers) cannot distinguish "built elsewhere" from "refused". **Fix.** Emit a typed `delivered_by: {"feature_only": "<regime_controlled_study_id>", ...}` map (or thin report envelopes referencing the S09c artifacts) instead of a refusal string.

---

### F7 — MINOR — S08's `authorized_trading_days` are the observed view days, and a labeled/view day divergence fails the stage

**Evidence.** `pipeline_regime.py:316-318` derives the schedule from `context.view.frame["trading_day"]`; `pipeline.py:1458-1459` builds the labeled folds from `context.labeled["trading_day"]`; `build_fold_set_artifact` (`fold_set_artifact.py:209`) then demands window equality. Probe C (label builder omitting the last day): S08 `failed` — "fold set carries 2 folds but the schedule has 3"; S09/S10/S14 `pending`. The charter allowlist never enters the schedule.

**Why.** §6.B `FoldSchedulePayload.authorized_trading_days` / `derive_fold_schedule(authorized_trading_days)`; DEV-R6.1-3 records only the fixture shape, not that "authorized" means "observed in the view". Production `context_labels.py:208` keeps censored rows (`binary_target=None`), so the divergence is unlikely on the shipped label path, but any label builder / filtered view that drops a day turns a typed situation into a stage failure. **Fix.** Derive the schedule once from a single day source (the view), pass it to `build_context_folds`, and rename the field or document that the days are the observation days (charter allowlist ∩ observed).

---

### F8 — MINOR — Owner-decision supersession is not crash-safe and ignores the store's own `supersedes` facts

**Evidence.** `owner_decisions.py:348-358` — the replacement is published first, the `SUPERSESSIONS.jsonl` line is appended afterwards and only `if … not already`; `assert_owner_decision_authorizes` (`:397-401`) derives the superseded set from the log alone. A crash between publish and append leaves the prior decision authorized forever (re-persisting the replacement is a reuse and never appends). **Why.** D5 "Supersession is store-owned and every line is backed by a verified replacement artifact" — the reverse direction (a verified replacement whose `supersedes` names the prior) should also count. **Fix.** Append the line before publication (or in the same atomic step) and let the chain loader union log lines with `supersedes` fields of verified artifacts.

---

### F9 — MINOR — The `none_v0` JSON event loader disagrees with the D15 contract it feeds

**Evidence.** `regime_stratified_prop.py:66-75` `_EVENT_PRECEDENCE` (equity_update=0 … replacement=7) vs `propsim/event_detail.py:137-148` `EVENT_TYPE_PRECEDENCE` (fee=0 … breach=7); `regime_stratified_prop.py:113` `trading_day = ts[:10]` (UTC date prefix) although R6.1 added `PropAccountEventEnvelope.trading_day` (`account.py:205-208`) and the audit JSON carries it. **Why.** D15 "exact event timestamp, trading day, clock policy, total-order fields"; §6.G stratified_prop reads both representations through one seam. **Fix.** Import `EVENT_TYPE_PRECEDENCE` and use `record.get("trading_day")`.

---

### F10 — MINOR — The promotion CLI defaults `decided_at` to the wall clock and can mint a second STRATIFICATION_READY beside S10's deterministic one

**Evidence.** `scripts/ifvg_regime_promotion.py:172` `decided_at or datetime.now(UTC).isoformat(...)`. **Why.** D4 "no wall-clock or 'latest decision' lookup enters execution"; §6.F (STRATIFICATION_READY is S10's deterministic decision). The CLI is an owner action, and the id is frozen downstream, but `--to stratification_ready` duplicates S10's decision under a different `decided_at`. **Fix.** Require `--decided-at` (no default) and refuse `--to stratification_ready` (S10-only), or make it reuse the S10 decision by exact id.

---

### F11 — MINOR — `hard_id_encoding` vocabulary mismatch between the request and the artifact

**Evidence.** `regime_study.py:113` `Literal["none", "fit_local_categorical_v1"]`; `regime_fold_features.py:116-117` `("none", "categorical")`; `regime_supervised_stage.py:69` maps by `!= "none"`. `grep -rn fit_local_categorical_v1 tests/` → no test exercises the non-default value. **Fix.** One Literal, one constant; a test through the pipeline with the categorical encoding.

---

### F12 — MINOR — Typed-null vocabulary extended without a deviation record

**Evidence.** `context_bar_panel_contract.py:237-244` adds `panel_source_bar_incomplete` to `PANEL_ASSIGNMENT_MISSING_REASONS`; `regime_oos_assignment.py:145-149` maps `source_bar_incomplete` onto it. **Why.** §6.C / owner correction 2 enumerate `panel_warmup / no_completed_panel_bar / panel_gap / panel_stale` (+ `coverage_gap`); the extension is consistent with final ruling 3 but `DEVIATIONS.md` does not list it. **Fix.** Record it (DEV entry) and add it to the §31 empty-state set if the UI renders it.

---

### F13 — MINOR — S05 silently materializes the panel from the first replay chart only

**Evidence.** `pipeline_regime.py:195` `chart_id = chart_ids[0]`. **Why.** §6.E S04 "records `core_replay_id → chart ids`"; S05 "context_bar_source(chart_id)" — with more than one chart the choice is undeclared and not part of any identity except through the resulting panel id. **Fix.** Refuse `len(chart_ids) != 1` (typed) or pin the chart selection in the request.

---

### F14 — MINOR — S14 consults in-memory run objects rather than persisted artifacts for the panel PIT assigner

**Evidence.** `regime_report_stage.py:66,80` — `panel_event_assigner` uses `regime["panel_frame"]` and `execution.run.assignments` (in-memory) instead of the persisted panel + fit assignments; `build_reports` also reads `context.tables_by_child`. **Why.** §6.E S14 "reads only persisted S09/S10/S12/S13 artifacts"; D14. Content-equal by construction (fits are verified by reproduction), so no result changes, but the "persisted only" invariant is not structurally enforced. **Fix.** Load the panel frame and the fit assignment sidecars by exact id inside the assigner.

---

### F15 — MINOR — Identity payloads carry nested plain dicts inside `ImmutableMap[str, Any]`

**Evidence.** `regime_controlled_study.py:151-153` (`baseline_summary`, `challenger_summary`, `paired_deltas`), `regime_cohort_model.py:116` (`paired_deltas`); values are `jsonable(...)` nested dicts (mutable after identity calculation), while `owner_decisions.py:172-176` refuses exactly that shape for its own payload. **Why.** Hunt 9; CS §0.3 deep immutability. R5B's `ControlledFeatureStudyPayload` is the precedent, so severity stays minor. **Fix.** `deep_freeze` the summaries or type them as `ImmutableMap[str, ImmutableMap[str, _Scalar]]` as `controlled_feature_study.py:78-80` now does.

---

### F16 — MINOR — Evidence-folder integrity: DEVIATIONS claims a reconciliation that has not happened, and three deviations are unrecorded

**Evidence.** `R6.1/DEVIATIONS.md:3` "Reconciled after the adversarial round (`ADVERSARIAL_REVIEW_RESOLUTION.md`)" — the file does not exist (this review is the round); unrecorded: the missing D15 summary (F3), the S08 observed-days semantics (F7), `panel_source_bar_incomplete` (F12), the §9.1 test-name mapping (F5). Also the D15 writer (`event_detail.py:310,406`) materializes every row as a dict and every partition's bytes in memory before publication — at the registered 10 M-row / 2 GiB budget the process exhausts memory before the budget bites ("streaming" is nominal). **Fix.** Reword the header after the round; add the entries; stream partitions to a temp directory and hash as you go.

---

## What holds (verified)

- **D7 / correction 1 (fold-safe features).** `regime_fold_features.py:645-706` consults only fit *k*'s rows for fold *k* (`_fit_rows_by_key` filters `fold_index == k`; the panel path passes `fold_assignments` of fold *k* and `partition_for_candidate`); distances/margins are computed against fit *k*'s own centroids (`regime_service.py:389-410`); `canonical_reporting_cluster_id` is a separate column never in `model_feature_names` (`:202-216`); `test_regime_fold_features_cannot_see_future_outer_folds` perturbs later observations and other fits and proves fold-*k* bytes identical while later folds change, with the descriptive loader monkeypatched to raise (passes).
- **Correction 2 (PIT rule).** `regime_oos_assignment.py:272-408` follows the normative order (same trading day via `classify_session`; latest `close <= as_of`; validity → `panel_warmup`/`panel_gap`; `elapsed > interval` → `panel_stale`; compatible partition → `coverage_gap`); `searchsorted`, no `merge_asof`; duplicates/NaT refused; `test_next_day_warmup_never_inherits_previous_day_regime` and `test_pit_rule_matrix_and_refusals` pass (no row of the next day resolves to a previous-day bar; ties → lexicographically last `row_id`; lowest fold wins in descriptive mode; own partition in fold-feature mode).
- **Correction 3 / D3.** `fold_schedules.py` derives windows exactly like `build_context_folds`; `fold_set_artifact.assert_same_fold_schedule` compares schedule ids and per-fold windows and never `fold_set_id`; `test_same_schedule_different_fold_sets_across_grains` passes; S08 asserts the panel and candidate sets share the schedule (`pipeline_regime.py:366`); the legacy hash is single-sourced (`fold_set_artifact.fold_set_id`, delegated from `regime_service`, `supervised_ladder`, `controlled_feature_study`).
- **Correction 5 / D14.** S09 = S09a (executor) + S09b/S09c only for `requires_supervision` (`pipeline_regime.py:401-480`); a descriptive study pins no model protocol (`pipeline.py:340-346`) and runs S09a only (`sub_steps == ["S09a"]` asserted); S14 fits nothing on the descriptive path (monkeypatched proof) — scope caveat in F4; DEV-R6.1-1 (S07 stays a dependency) is recorded.
- **Correction 6 / D15 (simulation side).** `propsim/event_detail.py`: policy/storage/schema/budget enter `AccountSimulationPayload` / `PortfolioSimulationPayload` (`simulation.py:148-171,211-230`) and `SimulationProtocol` (`charter.py:252-254`); ZSTD Parquet blocks of 250 paths, rows carry `event_ts_utc`/`event_ts_ns`/`trading_day`/`clock_policy_id`/`event_precedence`/`event_ordinal`/ids/source trade/candidate/amount; preflight rows and cumulative bytes refuse before `save_or_reuse_envelope`; manifest lists sha256/bytes/rows/schema per partition; `none_v0` artifacts are never widened (`store.py:312-333` refuses different sidecars; `test_none_v0_artifacts_are_never_widened`); `test_prop_event_detail_capacity_and_exact_order` reproduces the `(path_ordinal, event_ordinal)` total order.
- **Correction 7 / D13.** `comparison_rows.comparison_row_id` = hash{fold_schedule_id, candidate_fold_set_id (legacy population hash), fold_index, candidate_id, label_artifact_id}; emitted by prevalence, logistic and the CatBoost bundle rung; `_assert_identical_rows`, `assert_cross_arm_identity`, `paired_cell_delta_report` key on it; `oos_row_id` retained; ladder ids bind view id, bundle id, fold-feature evidence ref and every rung's resolved hash; `test_comparison_row_ids_are_equal_across_baseline_and_challenger_bundles` proves equal sets across different `view_id`s with disjoint `oos_row_id`s.
- **Final ruling 1 / D1 / D4.** `RegimeStudyRequest` refuses supervised classes without the three exact refs and descriptive requests that carry them (`regime_study.py:226-247`); the request is hashed into `PipelineSemanticSpecPayload` (`pipeline.py:276`); `verify_frozen_authority` verified-loads decision/assessment/owner/protocol and re-runs `assert_owner_decision_authorizes` at readiness (S00), S05 (`expected_protocol_id` = the resolved protocol), S09a (assessment id equality, `pipeline_regime.py:427-435`) and S10; no `latest`/`iterdir`/`glob` lookup exists in the regime, pipeline_regime, owner_decisions, promotion-CLI or panel modules (grep); `regime_stage_evidence_defaults` reads the run's own S10 record by exact stage-result id.
- **Final ruling 4 / D5 (decision 25).** `OwnerDecisionArtifactPayload` requires keys 25/28/29/30 and the eleven values incl. `algorithm_key`, `algorithm_parameters_hash` and the `algorithm_parameters` snapshot (`owner_decisions.py:92-115,138-187`); `expected_decision_values` recomputes them from the registry/protocol/assessment and `assert_owner_decision_authorizes` refuses any mismatch (parametrized test, incl. wrong key/hash/snapshot); verification mutates nothing (`test_lawful_synthetic_chain…`); synthetic provenance refused outside `synthetic_fixture`; a bare 64-hex ref is refused at persistence, at readiness, at activation and at the report gate (`regime_store.py:586-593`, `test_bare_64_hex_reference_is_not_evidence`).
- **Final ruling 3 (panel validity).** `context_bar_panel_materializer.py:349-363` nulls all seven features on any `observed != expected` bar in the 13-bar window, types `source_bar_incomplete`, and writes offending-bar rows to the validity sidecar bound by `validity_table_sha256`; `test_incomplete_source_bar_invalidates_entire_panel_window` pins 13 affected rows and the exact offending id/counts; `MINIMUM_SOURCE_BARS = 13`, `STD_DDOF = 0`, intervals `(300, 900)` refused otherwise (`regime_contracts.py:218-235`).
- **Final ruling 6 / D8.** Model-facing names are the fit-local distance vector, assigned distance, margin and (under `categorical`) the local id; the aligned reporting id is excluded from `model_feature_names` and from the activation payload (`regime_block_activation.py:210-247`); `bundle_rung_categorical_features` unions the frozen registry with block-declared categoricals; the fitted CatBoost model reports them as categoricals (`get_cat_feature_indices`, test).
- **Final ruling 7.** No identity payload hashes its own content hash: table hashes (`feature_table_sha256`, `assignment_table_sha256`, `panel_table_sha256`, `frame_table_sha256`, `detail_sha256`, `fold_definitions_sha256`) are envelope extras; source-document hashes (`bars_tf_sha256`, `replay_chart_manifest_payload_sha256`) are inputs.
- **D9.** `IFVG_REGIME_CONTEXT_V1` stays PLANNED at import (`feature_blocks.py:367-374`); activation is a pure `with_activated_block` event (block_version 2) bound to protocol, fold-feature, assessment, promotion and owner ids; `B7_CORE_REGIME` resolves only under the activation registries.
- **D10 / D11 / §6.H.** Per-fold bootstrap with `rng(7+fold)` and seeds `1000+1000·fold+i` from registry pins minus `random_state` (`regime_diagnostics.py:291-300`); protocol-wide minimum fold mean gated by the renamed `minimum_bootstrap_aligned_ami_mean` (0.5); `bootstrap_undefined_fold`; 50/400 budget via `bootstrap_plan`; candidate-event transitions reset on day/session/7200 s; panel transitions consecutive completed bars; exactly-one-grain validator; "advisory" removed from code (grep); `transition_matrix_from_sequence` deleted.
- **D2 / §6.D.** `load_regime_observations` takes ids from the loaded envelope only; `run_regime_protocol` requires the pinned panel id ∈ `source_artifact_ids`; the executor loads the fold set and refuses grain/key/source mismatches; fits are reused only by reproduction (`regime_store.py:298-328`, second-attempt pipeline test reuses every stage with identical stage-result ids).
- **§6.A block/bundle.** `with_registered_block` is a one-time versioned event; `PRE_R6_1_*` registries exported; import-time invariants (panel names disjoint from M3 and MBP-1, no deep-book identifier, version 1 AVAILABLE, registry hash changed); `BP0_CONTEXT_BAR_PANEL` (no base) and `assert_grain_bundle_coherent`.
- **§6.J.** `ifvg_context_catboost_bundle_v1` registered (AVAILABLE, `nonlinear_challenger_bundle`); resolved hash covers parameters, preprocessing policy, categorical registry, package/Python versions, ordered features, bundle id, registry hash, fold-set and schedule ids, seed, grid, calibration; `CATBOOST_BUNDLE_REFUSAL` now scoped to the tier-locked rung with the mirror refusal; readiness admits the rung on bundle paths only; M0–M3 golden hash test present.
- **§8 stamps** present in `REGIME_PROPOSED_DEFAULTS` and single-sourced from the panel leaf (`panel_minimum_source_bars=13`, `panel_std_ddof=0`, `panel_assignment_max_staleness_intervals=1`, `bootstrap_refits_per_fold=50`, `bootstrap_total_refit_cap=400`, `candidate_event_maximum_gap_seconds=7200`, `minimum_trades_per_regime_stratum=20`, `minimum_training_rows_per_regime_stratum=60`, `prop_event_attribution_policy`, `regime_feature_hard_id_encoding_default=none`).
- **Refusal strings / stamps** match the plan where the plan fixes them: `MODEL_FEATURE_REFUSAL`, "nothing here promotes" suffix, the interval refusal text, `fold_local_fit_partition_assignment_v1`, `completed_bars_last_at_or_before_v1`, both temporal-order policy ids, `regime_oos_assignment_v1`, `ifvg_regime_context_formula_v1`, `regime_stratification_v1`; DEV-R6.1-6 records the `Error`-suffixed exception names.
- **UI** exposes no control that promotes/launches/retrains (`test_regime_panel_exposes_no_control…`), no `session_state` writes in `ifvg_regime_panels.py`; the two §31 states are registered; the Regime Lane auto-fill passes `default_ids=`.
