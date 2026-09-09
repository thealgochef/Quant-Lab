# HARDENING-BACKEND — Files Touched

Reconciled against the release commit's `git show --name-status`
(**`e56f937`**, parent `0c8d528` = R6.1-FIX; tree `104900745106…`; **61
files: 26 added, 35 modified; +13,386 / −721**), including the adversarial
fix round: 7 new src modules + 5 new scripts + 14 new test files (13 modules
+ the `namespace_fixture.py` helper); 14 modified src modules + 2 modified
scripts + 15 modified test files + `pyproject.toml` + `docs/DECISIONS.md` +
the three staged shared docs. The commit is path-scoped to this list;
`docs/ML_TRAINING_WORKBENCH.md` (user-owned) and the pre-existing untracked
local files (data / reports / prompts / the `QL-*` evidence tree) are NOT in
the commit; the three shared docs (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`) are staged as HEAD + HARDENING-BACKEND lane
transforms only (`stage_shared_docs.py`; the surviving worktree diff equals
the R1 pre-existing hunks — `_surviving_shared_doc_diff.patch`).

## New source modules (7)

| File | Workstream | Content |
|---|---|---|
| `src/.../ifvg/search/store_namespace.py` | A | the semantic store namespace (§4.1): `StoreNamespacePayload/Envelope`, `genesis_id_for`, `SupersessionHeadWitness`, the head-pointer primitives (`read_supersession_head`, `write_supersession_head_atomic`, `chain_head_digest`), `initialize_store_namespace` / `initialize_test_namespace`, `load_/require_store_namespace`, `namespace_class_of`, `path_looks_like_research_store` (defense in depth), `assert_namespace_deployment_coherent`, `StoreNamespaceError` (14 typed reasons); identity pair `StoreNamespace` |
| `src/.../ifvg/search/supersession_chain.py` | A | the immutable supersession record chain (§4.2): `OwnerDecisionSupersessionPayload/Envelope` (store `owner_decision_supersessions`), `SupersessionRecord`, `verify_chain_structure`, `load_supersession_records`, `current_supersession_head_witness`, `assert_head_witness_current`, `publish_supersession` (four-step publication under the lock); identity pair `OwnerDecisionSupersession` |
| `src/.../ifvg/search/owner_decision_lock.py` | A | the liveness-aware lock (§4.3): `LockBody`, `current_process_start_token`, `process_liveness` (Win32 `ctypes` / POSIX), `OwnerDecisionLock` (acquire with dead-holder reclaim only, `refresh`, `verify_held`, `release`), `OwnerDecisionLockError` (4 typed reasons) |
| `src/.../ifvg/search/bounded_verification.py` | E | Phase 4 (§6): `preflight_bounded_verification` → `BoundedVerificationPreflight` / `BoundedVerificationRefusalError` (16 reasons); `R1BaselineGatePayload/Envelope` + `build_/save_r1_baseline_gate_report` (store `r1_baseline_gate_reports`); `BoundedComponentResult`, `BoundedReleaseControlFlowPayload/Envelope`, `build_/save_bounded_release_control_flow_report` (store `bounded_release_control_flow_reports`); identity pairs `R1BaselineGateReport`, `BoundedReleaseControlFlowReport` |
| `src/.../ifvg/search/trading_calendar.py` | D | the logical trading-day calendar (§5.1; F-22): `cme_globex_18et_weekday_v1`, `is_logical_trading_day`, `logical_trading_days`, `next_/previous_logical_trading_day`, `assert_consecutive_logical_days`, `consecutive_logical_windows`, `physical_partitions_for`, `session_bounds_utc`, `store_day_chain`, `SourcePartitionRef`, `VerificationTradingDayRef`, `trading_day_ref_from_inventory`, `inventory_from_permitted_source_hashes` |
| `src/.../ifvg/search/verification_window.py` | D | the rebuilt shortlist (§5.1): `LogicalDayCoverage`, `LogicalWindowScore`, `ShortlistEntry`, `R1CandidateAssessment`, `VerificationWindowShortlist`, `build_logical_day_coverage`, `rank_logical_windows`, `build_verification_window_shortlist`, `render_shortlist_markdown`, `shortlist_document` |
| `src/.../ifvg/search/seed_production.py` | D | the separately authorized seed production (§5.3–§5.5; F-16 / F-21): `SeedProductionReplayPolicy`, `SeedProductionAuthorizationPayload/Envelope/Ref`, `SeedProductionRunPayload/Envelope`, `SeedProductionAuthorizationError`, `seed_chain_source_inventory_hash`, `persist_/synthetic_/verify_seed_production_authorization`, `run_seed_production_chain`, `build_seed_production_packet`, `build_verification_authorization_packet`, the markdown renderers; identity pairs `SeedProductionAuthorization`, `SeedProductionRun` |

## Modified source (14)

| File | Workstream | Change |
|---|---|---|
| `src/.../ifvg/features/mbp1_coverage_diagnostic.py` | fix round (RA-01) | `assert_diagnostic_authorized` binds the store namespace (`test`, coherently deployed) + the CURRENT head witness before the coverage matrix loads, before `register_program_allowlist`, before any path; the pathname rule is defense in depth |
| `src/.../ifvg/search/store.py` | A | `SEARCH_STORE_NAMES` + `owner_decision_supersessions`, `seed_production_authorizations`, `seed_production_runs`, `r1_baseline_gate_reports`, `bounded_release_control_flow_reports` |
| `src/.../ifvg/search/identities.py` | A/D/E | `registered_identity_pairs()` imports `store_namespace`, `supersession_chain`, `bounded_verification`, `seed_production` |
| `src/.../ifvg/search/owner_decisions.py` | A | `OwnerDecisionArtifactPayload.store_namespace_id`; `require_owner_decision_namespace`; semantic `assert_run_scope_lawful_for_root`; namespace-bound `load_owner_decision` / `persist_owner_decision` (the record published BEFORE the replacement through `publish_supersession`); chain-backed `load_supersession_chain`; `research_namespace_root` = defense-in-depth alias; the JSONL log, `_GENESIS_SHA256`, the age-based lock helpers and the line-digest machinery removed; proposals carry `store_namespace_id`; `synthetic_owner_decision_fixture` marks unmarked test roots |
| `src/.../ifvg/search/authorization.py` | A | `OwnerAuthorizationBundle` / `VerificationAuthorizationRef` gain `store_namespace_id` + `supersession_head_witness` (validator: the witness names the bundle's namespace); `assert_authorization_bound_to_store`; `validate_owner_authorization(store_root=)` |
| `src/.../ifvg/search/verification.py` | A | `validate_verification_run(..., store_root)` (required; a `test` namespace + current witness); the example run carries the new fields |
| `src/.../ifvg/search/charter.py` | A | `validate_charter(store_root=)`; `save_charter` = semantic synthetic rule + bound real bundles |
| `src/.../ifvg/search/executors.py` | A | `real_verification_context` requires a coherent `test` namespace and a current witness before any path |
| `src/.../ifvg/search/child_replay.py` | A | the canonical-root check runs first; `validate_verification_run(store_root=)` |
| `src/.../ifvg/search/pipeline.py` | A + C | S00: the real verification branch passes `store_root`; the full-development branch requires the charter bundle bound to the store + current witness. `SUPPORTED_CHILD_WORKERS`, `EXECUTION_MODE_V1`, `UNSUPPORTED_WORKER_PARALLELISM_REASON`, `UnsupportedWorkerParallelismError`, `assert_supported_worker_parallelism`, `worker_parallelism_refusal`; `WorkerPolicy.max_workers == 1` (validator; `le=4` removed); `ExecutionAttemptIdentity.effective_workers` / `execution_mode`; `run_pipeline` stamps the receipt |
| `src/.../ifvg/dataset.py` | C | `concat_schema_aligned` (+ `_na_capable`, `_common_with_all_na_entry`) replaces the deprecated audit-frame concat; `numpy` import |
| `src/alpha_lab/propsim/event_detail.py` | B | the streaming writer (iterable of walk pairs; declared preflight counts; no whole-artifact index; the unconditional disk-backed DuckDB uniqueness check; bounded row groups; manifest records `row_group_size` + `event_id_uniqueness`); `numpy` import dropped |
| `src/alpha_lab/propsim/search_bridge.py` | B | `_walk_pairs` generator feeds the writer with the declared counts |
| `src/.../ifvg/ml/regime_stratified_prop.py` | B | the external DuckDB aggregation (intermediate Parquet partitions, memory limit, spill dir, exact counts, canonical ORDER BY, `_RowGroupAlignedWriter`, budgets before publication, temp cleanup); `StratifiedPropResult.summary` lazily parsed |

## Modified scripts (2) + new scripts (5)

| File | Workstream | Change |
|---|---|---|
| `scripts/ifvg_pipeline_job.py` | C | `_refuse_unsupported_parallelism`: `start` / `resume` / `worker` refuse `--max-workers != 1` before job creation / store access (JSON, exit 2) |
| `scripts/ifvg_regime_promotion.py` | A | `propose` stamps the target store's verified `store_namespace_id` (or the placeholder) |
| `scripts/ifvg_store_namespace.py` (new) | A | the one-time explicit namespace migration (`init` intent/confirm, `show`) |
| `scripts/hardening_capacity_benchmark.py` (new) | B | the `HARDENING_CAPACITY_POLICY_V1` harness |
| `scripts/ifvg_verification_window_shortlist.py` (new) | D | rebuilds the logical-day shortlist from already-authorized evidence |
| `scripts/ifvg_seed_production.py` (new) | D | `packet` / `verify-authorization` / `run` |
| `scripts/ifvg_bounded_verification.py` (new) | E | `preflight` / `run` (refuses `fail_before_path` without the owner's persisted authorization) |

## Config (1)

| File | Workstream | Change |
|---|---|---|
| `pyproject.toml` | C | `[tool.pytest.ini_options].filterwarnings = ["error", <one exact rule>]` with the reason / owner / expiry comment |

## New tests (13)

| File | Workstream | Covers |
|---|---|---|
| `tests/agents/ifvg_search/namespace_fixture.py` | A | helpers: `namespace_and_witness`, `verification_authorization_ref`, `owner_authorization_bundle`, dummy witness |
| `tests/agents/ifvg_search/test_store_namespace.py` (9) | A | explicit / idempotent / immutable init; verified load (tamper, malformed, class); identity is not a path hash + relocation; path heuristic = defense in depth; synthetic scope follows the class; unmarked stores carry no authority; missing head = corruption; interrupted init recovery; the CLI |
| `tests/agents/ifvg_search/test_supersession_chain.py` (7) | A | immutable records + head commits to the chain; identical replay reuses / divergent refused; orphan records have no authority; crash before the head update repaired idempotently; rollback / deletion / forgery / rewritten record / foreign head refused; witness rule (missing / shorter / different / relocated); the transition verifier runs inside the lock before any write |
| `tests/agents/ifvg_search/test_owner_decision_lock.py` (8) | A | liveness verdicts; slow live holder never reclaimed; dead pid after timeout; PID reuse; other host / malformed never reclaimed; heartbeat + own-token release; a lost lock aborts before the head moves; crash recovery |
| `tests/agents/ifvg_search/test_bounded_verification.py` (8) | E | the preflight (pass; refusal order; every window defect; seed binding); the R1 gate report contract; the bounded report over the synthetic full-plan fixture; untyped states fail closed; the regime non-fit branch |
| `tests/agents/ifvg_search/test_bounded_verification_script.py` (2) | E | both subcommands refuse `fail_before_path` on a run-less store; the preflight passes on a persisted bound run |
| `tests/agents/ifvg_search/test_trading_calendar.py` (5) | D | the calendar policy, consecutive semantics, partitions + DST session bounds, the store-day chain, the day refs |
| `tests/agents/ifvg_search/test_verification_window.py` (4) | D | ranking order, coverage rows, lexicographic ranking + constraints, the shortlist / R1 assessment / no allowlist marker |
| `tests/agents/ifvg_search/test_seed_production.py` (9) | D | the policy fails before path; payload semantics; verification refusals; synthetic provenance confinement; the synthetic seed-production proof; refusals before any path; unsigned packets; the CLI |
| `tests/agents/test_hardening_warning_policy.py` (11) | C | the schema-aligned concat vs the old result on every warning shape and the real audit frames; the typed append; the registered pytest policy; the typed worker refusal |
| `tests/propsim/test_event_detail_streaming.py` (8) | B | one-shot iterable never materialized; no whole-artifact index; byte-identical to the sequence form; declared counts verified; the external check catches a forged cross-block duplicate (and nothing in memory does); path/ordinal one-to-one; temp dirs cleaned; the bridge passes a generator |
| `tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` (6) | B | external aggregation equals the in-memory reference byte-for-byte; row-group alignment past one group; cross-partition path repetition refused externally; the row budget is the exact external count; temp cleanup; the DuckDB limits |

## Modified tests (14)

`test_owner_decisions.py` (rewritten supersession / lock / namespace tests; explicit test namespaces; `_owner_signed` stamps the namespace id), `test_authorization.py` (bound bundles; the namespace + head test), `test_charter.py` (bundle fields), `test_verification.py` (bound runs; `store_root`; the namespace + head test), `test_verification_slice.py` (the verification store is a marked test namespace; bound ref), `test_study_providers.py`, `test_mbp1_coverage_evidence.py`, `test_child_audit_companion.py` (bound refs), `tests/agents/test_ifvg_regime_promotion_cli.py` + `tests/agents/data_infra/ifvg/test_regime_store.py` (explicit test namespaces in the lane fixtures), `test_pipeline_contracts.py` / `test_pipeline_run.py` / `test_pipeline_job_script.py` (sequential truth), `tests/agents/test_ifvg_context_experiment_engine.py` (the typed append).

## Adversarial fix round (amendments to the files above)

| File | Finding | Amendment |
|---|---|---|
| `search/owner_decision_lock.py` | RA-02 | `_retry_fs`; `release` / `refresh` / reclaim / body read retry transient `OSError`s; typed `lock_release_failed` / `lock_refresh_failed` |
| `search/store_namespace.py` | RA-10 | `_atomic_write_text` retries `os.replace`; typed `atomic_write_failed` |
| `search/authorization.py` | RA-09 | `assert_authorization_bound_to_store` runs `assert_namespace_deployment_coherent` |
| `search/pipeline.py` | RA-03 / B-06 / B-03 | `activate_pipeline_result` verified-loads the charter and requires the bundle bound to the CURRENT head (a synthetic marker never activates); `WorkerPolicy.max_workers` before-validator (strict int, typed before `ge=1`); S08 persists `fold_summary.json` |
| `search/owner_decisions.py` | RA-06 | `synthetic_owner_decision_fixture` auto-marks temp-directory roots only (`_under_temp_directory`) |
| `search/executors.py` | RA-08 | `_verification_run_envelope(pipeline_semantic_id=)` selects by pipeline id, refuses ambiguity; the pipeline entry passes the semantic id |
| `search/seed_production.py` | RA-07 | `_store_entry_counts` / `_assert_only_permitted_stores_changed` post-condition; reason `prohibited_output_written` |
| `search/bounded_verification.py`, `scripts/ifvg_bounded_verification.py` | RA-04 / RA-05 / B-01 / B-03 / B-04 | Fix-E (`_fix_E_NOTES.md`): typed seed reasons (`seed_snapshot_unverifiable`), the real S14 sidecar name + `children_skipped` evidence, `fold_summary.json` reader, panel typed-reason counts, `store_behavior_proofs` + attempt-1 gates in the runner |
| NEW `tests/agents/ifvg_search/test_hardening_fix_round.py` (14) | main lane | RA-01 / RA-02 / RA-03 / RA-06 / RA-08 / RA-09 / RA-10 / B-06 / B-03 (S08 sidecar) proofs |
| `tests/agents/ifvg_search/test_mbp1_coverage_evidence.py` | RA-01 | the "missing coverage matrix" case first proves the namespace refusal, then marks the root as the same namespace instance |
| NEW `tests/agents/ifvg_search/test_pipeline_authority_seams.py` (3) | B-05 | a full-development charter with a real bundle: unmarked store / foreign namespace / stale witness refuse at S00 with no stage run and no child row; a bound bundle passes S00 |
| `tests/agents/ifvg_search/test_bounded_verification.py` (8 → 13) | RA-04 / RA-05 / B-01 / B-03 / B-04 | seed failures typed by cause; the fold outcome from the typed sidecar; typed child skips from the S14 record; a REAL panel-grain regime run (`completed_panel_regime_pipeline`) reads the persisted S14 record, the fold summary and the panel artifact's validity counts; `store_behavior_proofs` on the fixture's table incl. a tampered source |
| `tests/propsim/test_event_detail_streaming.py` (8 → 9) | B-08 | the early `path_count` overrun refuses while streaming, before any flush / manifest |
| `tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` (6 → 7) | B-07 | the byte-budget refusal with a generous row budget (file branch + empty-table branch; temp cleaned; nothing published) |

## Docs

`docs/DECISIONS.md` — D-050 + the reservation note; the three shared docs via `stage_shared_docs.py`; `../DECISIONS_TAKEN.md` #115–#125.
