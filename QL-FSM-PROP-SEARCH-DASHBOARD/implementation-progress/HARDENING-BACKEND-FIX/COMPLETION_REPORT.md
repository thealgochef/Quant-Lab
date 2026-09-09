# HARDENING-BACKEND-FIX — Completion Report

**Feature:** `ifvg_prop_robust_config_search_v1`
**Release type:** compact corrective backend release (plan §1.2: ten corrections; nothing else)
**Plan:** `IMPLEMENTATION_PLAN.md` (this folder; verbatim copy of `HARDENING_BACKEND_FIX_IMPLEMENTATION_PLAN.md`, sha256 `28be05221ef040f8…`)
**Baseline:** parent commit `e56f9376a5b4ba269f7fa11cdc6e37b08200638f` (HARDENING-BACKEND) on `feature/ifvg-prop-robust-config-search-v1`; HEAD had not advanced.
**Final commit:** `a5eee1a` (`a5eee1aa1e04127809eb57fec1e61331a1d692ef`; tree `7ded6e896f72…`; parent `e56f937`; branch `feature/ifvg-prop-robust-config-search-v1`; one release-scoped commit — not pushed, not merged)

```text
implementation_status: complete
backend_dev_complete_for_ui: true
ui_implementation_may_begin: true
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
full_authorized_development_run_completed: false
```

Not performed (by design, plan §1.1 / §1.3): no owner verification run, no seed production, no full pipeline, no strategy search, no real-data model fit or prop simulation, no catalog activation, no live-serving work, no push, no merge. `data/` gained zero files.

---

## 1. Baseline and preflight (plan §2)

| Item | Evidence |
|---|---|
| Branch / HEAD | `feature/ifvg-prop-robust-config-search-v1` @ `e56f937` (expected parent; no later owner commit) |
| Pre-existing user-owned hunks | `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`, `docs/pipeline_state.yaml` — captured as a patch (sha256 `b12b4200379a101e…`) and per-file hashes before implementation; **byte-identical after the release** (the same patch hash and the same four file hashes before the commit; after it the three shared docs carry the lane transforms staged as HEAD + transforms, and the surviving worktree diff is the user's hunks only — §9 item 8) |
| Supplied baseline | prior full suite 2,102 passed / 0 failed, warnings as errors, Ruff clean, `git diff --check` clean — accepted without rerunning |
| Read authority | this plan; `HARDENING-BACKEND/_PROGRESS_CHECKPOINT.md`, `GATE_SUMMARY.md`; the named contract / test-matrix / owner-decision docs. `HARDENING_BACKEND_INDEPENDENT_REVIEW.md` and the R6.1-FIX readiness plan named in §2.4 were **not present in the working tree**; the plan's own defect descriptions were the authority |
| Evidence folder | `implementation-progress\HARDENING-BACKEND-FIX\` (created; no prior-release evidence folder edited) |

Effort control (plan §3): one main agent implemented everything; exactly two read-only reviewers ran once (§13) and their nine in-scope findings (0 blockers, 2 major, 2 medium, 5 minor) were fixed in the one permitted round (§7); no workflows, no nested agents. (The first reviewer pass was lost to a context-window reset before either report was written — `_PROGRESS_CHECKPOINT.md` Reset 1 — and was relaunched once with the same scope; only that relaunch produced reports.)

---

## 2. Corrections delivered (plan §§4–10) — exact files changed

### Source (26 files: 1 added, 25 modified)

| Workstream | File | Change |
|---|---|---|
| A | `src/alpha_lab/agents/data_infra/ifvg/search/file_mutex.py` **(new)** | private stdlib cross-process mutex (`msvcrt.locking` byte-range / `fcntl.flock`); never unlinked; no body, no authority |
| A | `search/owner_decision_lock.py` | token-safe reclamation under the reclaim mutex (`_observe_stale` → `_reclaim`: re-read, re-evaluate age/host/pid/start-token, unlink ONLY the byte-identical dead holder); typed `lock_read_failed` (never absence); `release()` raises `lock_release_failed` when it cannot read/verify its own lock (absent / malformed / vanished / persistent I/O); `_try_create` removes or quarantines the partial file before `lock_create_failed`; token-only release preserved |
| A | `search/store_namespace.py` | one-time init mutex (`owner_decisions/STORE_NAMESPACE.init.mutex`); five-state classification (both absent → deterministic pair through temp files + verified-load; both coherent → idempotent / `store_namespace_divergent`; one present → exact recovery by the identical request (class + explicit instance id, no supersession records) else `incomplete_store_namespace_initialization`; both inconsistent → fail closed); `_verified_genesis_pair` proves `namespace.store_namespace_id == head.store_namespace_id`, `authority_genesis_id == head.head_sha256`, requested class == persisted class; shared head-document helper (recovered bytes are the deterministic bytes); reasons `incomplete_store_namespace_initialization`, `store_namespace_initialization_busy`, and (§10) `supersession_decision_unverifiable`, `supersession_transition_unlawful`, `supersession_chain_divergent` |
| A | `search/supersession_chain.py` | imports the supersession store name from the namespace module (one source of truth) |
| B | `search/trading_calendar.py` | `SourceKind = Literal["mbp1", "trades", "legacy_verified_replay_source"]`; private physical-file resolver (`_PHYSICAL_STEM_RESOLVER`, the only place the historical stem is named); `PhysicalSourceDescriptor` (internal: filename, era, sha256, partition key; `replay_bytes_only`); `assert_public_source_kind`; `physical_source_descriptor`; `physical_descriptors_from_permitted_source_hashes`; `inventory_from_permitted_source_hashes` returns public kinds; `trading_day_ref_from_inventory` refuses a non-public kind |
| B | `search/verification_window.py` | era-boundary docstring names the public kinds only |
| B | `search/seed_production.py` | `seed_chain_source_inventory_hash` refuses a non-public kind before hashing |
| C | `ml/regime_contracts.py` | `validate_native_values` + `FIT_ASSIGNMENT_NATIVE_SPEC` + `SENTINEL_STRINGS` + `RegimeAssignmentEvidenceError` (typed reasons); `_fit_assignment_frame_for_schema` validates natively BEFORE `to_numeric` / `astype`; `validate_assignment_rows` gains the provenance rules (fold + partition together; a fit id only with them; `no_oos_assignment` never names a fit / fold) |
| C | `ml/regime_oos_assignment.py` | three-way semantics (`_typed` with retained provenance, `_retained_invalid_row`); the candidate grain indexes ALL test rows; the panel PIT rule retains the invalid fit row in both modes; `OOS_ASSIGNMENT_NATIVE_SPEC` validated before canonicalization; payload refuses any `assignment_schema_hash` but the registered one; `_assert_candidate_sets_equal` (exact set / count / duplicates); `verify_assignment_table_bytes` decodes the Arrow bytes and proves schema / count / uniqueness / row invariants at save AND load |
| C | `ml/regime_fold_features.py` | `fold_feature_native_spec`; native validation before conversion (no `errors="coerce"`); spine rules in `validate_fold_feature_rows`; `assert_fold_feature_spine_bound` at build and load (rows ⊆ the `FoldFitRef` binding of their fold); an invalid fit row without a typed reason is refused |
| C | `ml/regime_assignment_sources.py` | `regime_for_trades` keeps the known fit / fold of an invalid assignment; only the trade-facing outputs are null |
| D | `search/store.py` | `_validate_manifest_entries` (shared by every probe / load / reuse path), `_sidecar_path` (resolve + compare before opening; symlink escape), size + hash checks, reasons `invalid_store_locator`, `sidecar_path_escape` |
| D | `ml/comparison_rows.py` | `assert_unique_label_candidates` (before every label hash), `assert_exact_label_artifact` |
| D | `ml/controlled_feature_study.py`, `ml/regime_controlled_study.py`, `ml/regime_cohort_model.py` | optional `label_policy_id`: when a persisting caller names the policy the label artifact id is proven to derive exactly from these labels |
| D | `ml/regime_supervised_stage.py`, `search/pipeline.py` | the pipeline's persisting study seams pass the registered `label_policy_id` (S09c and the controlled MBP-1 study) |
| D | `ml/regime_stratification_service.py` | `StratificationEvidenceError`; the persisting service requires every child's `executed_trade_table_id` (`executed_trade_table_required`) |
| E | `search/child_replay.py` | `canonicalize_seed_datetimes` (aware → fresh stdlib-UTC object; naive unchanged; dataclasses / pydantic / named tuples / tuples / lists / dicts rebuilt), `canonical_seed_hash`; `save_seed_snapshot` canonicalizes unconditionally (the one seam) |
| E | `search/seed_production.py` | `_canonical_utc` delegates; the runner hashes and persists the canonical seed (the old "hash changed" refusal removed — the canonical hash IS the identity) |
| F | `propsim/event_detail.py` | `EventDetailBudget.max_rows_per_partition` (omitted from serialization when absent → V1 identities byte-identical), `EVENT_DETAIL_BUDGET_V2` (50,000), `EVENT_DETAIL_PARTITION_BOUND_POLICY_V2`; the writer flushes at the bound even inside one path; partitions keyed `(path_block_id, partition_ordinal)` (`event_detail_block_000000_000.parquet`) with first / last event keys, rows, bytes, sha256, schema hash; cleanup of every written partition on refusal; the reader proves the bound, the boundary keys and the total order across partitions; the V1 budget is loadable, refused by the writer |
| F | `propsim/search_bridge.py` | the V2 budget is the default identity budget |
| F | `scripts/hardening_capacity_benchmark.py` | B1 shapes `normal` / `skewed` / `dense`, the resident-batch gate, shape evaluation + markdown; one markdown robustness fix (a B1-only run) |
| G | `search/owner_decisions.py` | `verify_complete_owner_authority_chain` + `CompleteOwnerAuthorityChain`; `load_supersession_chain` = the complete proof |
| G | `search/authorization.py` | `assert_authorization_bound_to_store` runs the complete proof (charter freeze/load, pipeline launch, activation gate, executors, verification run, MBP-1 diagnostic all bind through it) |
| G | `search/seed_production.py` | `_witness_current` runs the complete proof; the proof's reasons registered |
| G | `search/bounded_verification.py` | the preflight witness step runs the complete proof |
| review | `search/file_mutex.py`, `search/owner_decision_lock.py`, `search/store_namespace.py` | RA-01 a held lock found carrying another writer's token releases as the typed `lock_release_failed` (the foreign lock untouched); RA-02 only the platform's contention errno is contention (`EACCES` / `EDEADLOCK` for `msvcrt.locking`, `EWOULDBLOCK` / `EAGAIN` for `fcntl.flock`) — anything else is `FileMutexError` → the registered `lock_mutex_failed` / `store_namespace_initialization_failed`; RA-04 a read failure right after exclusive creation discards the fresh lock before the typed error |
| review | `search/child_replay.py` | RA-03 `canonicalize_seed_datetimes` covers dict keys, set / frozenset members, pydantic `extra="allow"` values, every dataclass field (`init=False` included, onto a shallow copy) and pandas Timestamps; `numpy.datetime64`, `NaT` and sub-microsecond carriers are refused (`SeedSnapshotError`), never dropped |
| review | `ml/comparison_rows.py`, `ml/controlled_feature_study.py`, `ml/regime_controlled_study.py`, `ml/regime_cohort_model.py` | RB-01 run-level `label_identity_proof` (`exact` / `caller_supplied_unproven` / `content_hash_unpersisted`; not identity-bearing); `assert_persistable_label_proof` at every persisting save — an id passed without the registered policy is in-memory only |
| review | `ml/regime_fold_features.py` | RB-02 `assert_fold_feature_spine_bound`: a fit-bearing fold refuses any fit-less row (build and load) |
| review | `propsim/event_detail.py`, `scripts/hardening_capacity_benchmark.py` | RB-03 partition and manifest paths registered BEFORE their writes and the manifest stage inside the cleanup guard (a partial file and a manifest-stage refusal leave nothing behind; a foreign manifest is never removed); RB-04 the reader binds the per-partition bound to the identity-bound budget and refuses a disagreeing top-level field; the benchmark observes the resident batch at the writer's Parquet seam (`max_resident_rows_observed`) instead of the manifest's own claim |
| review | `ml/regime_oos_assignment.py`, `ml/regime_assignment_sources.py` | RB-05 `validate_consumed_natively`: the PIT rule, the candidate OOS index and the executed-trade projection validate the input columns they consume natively before any `bool` / `astype(bool)` coercion (the kernel's empty fit id for fit-less folds stays the seam's own mapping) |
| docs | `docs/DECISIONS.md` | D-051 recorded (the review round amended into its Decision); reservation note updated |

### Tests (25 files: 2 added, 23 modified) — 60 new test functions (3 parametrized: 31 manifest cases, 7 timezone cases, 6 chain-corruption modes) plus 9 extended existing tests

`tests/agents/ifvg_search/test_owner_decision_lock.py`, `test_store_namespace.py`, `test_trading_calendar.py`, `test_verification_window.py`, `test_seed_production.py`, `test_bounded_verification.py`, `test_public_source_kind_surface.py` **(new)**, `test_store_sidecar_probe.py`, `test_pipeline_evidence_integrity.py`, `test_child_replay.py`, `test_authorization.py`, `test_authority_chain_seams.py` **(new)**, `conftest.py` (seed re-zoning helpers); `tests/agents/data_infra/ifvg/test_regime_oos_assignment.py`, `test_regime_assignment_evidence.py`, `test_regime_fold_features.py`, `test_regime_fold_feature_evidence.py`, `test_r61_fix_review_fixes.py`, `test_label_identity.py`, `test_regime_stratification_evidence.py`, `test_regime_stratification.py`, `test_controlled_feature_study.py` (review round), `test_regime_supervised_studies.py` (review round); `tests/propsim/test_event_detail_streaming.py`, `test_account_event_detail.py`.

Existing expectations changed ONLY where the plan changes the semantics: (i) a class-only re-initialization of a headless marked store and an orphan genesis head that matches no deterministic initialization are now `incomplete_store_namespace_initialization` (§4.2 rule 3) instead of `supersession_head_missing` / silent rewrite; (ii) `mbp10` fixtures name the public legacy literal; (iii) the OOS identity-binding case keeps the candidate sets equal (§6.5); (iv) persisting stratification calls name a persisted executed-trade table (§7.3); (v) a refused event-detail build leaves no partition behind (§9.2 cleanup); (vi) a bundle re-signed over a chain that names non-existent decisions is refused (`supersession_decision_unverifiable`, §10.1) instead of accepted; (vii) review round RB-01: the persisting study tests pass the registered label policy (a save now requires the exact-label proof) and the regime study lane derives its label artifact id from that policy; (viii) review round RB-05: one pre-existing OOS test documenting the kernel's empty fit id for fit-less folds caught the first cut of the consumed-column validation, which therefore excludes `regime_fit_id`.

---

## 3. HB-FIX acceptance assertions (plan §12.2)

| ID | Assertion | Evidence (test) |
|---|---|---|
| HB-FIX-01 | Two stale reclaimers cannot delete the newly acquired live lock | `test_two_stale_reclaimers_cannot_delete_the_winner` (two threads, barrier + event forcing both to observe stale S before either reclaims; the loser re-reads A under the mutex and times out; A survives), `test_many_reclaimers_never_overlap` (4 threads, max concurrent holders = 1), `test_reclaimer_that_observed_old_token_cannot_unlink_new_live_token`, `test_unknown_liveness_remains_non_reclaimable`, `test_persistent_read_error_is_typed_not_absence`, `test_release_read_error_does_not_silently_leave_success`, `test_failed_body_write_cleans_partial_exclusive_lock`; review round: `test_release_of_a_held_lock_replaced_by_another_writer_is_typed` (RA-01), `test_mutex_persistent_io_error_is_not_reported_as_contention` (RA-02), `test_read_failure_after_exclusive_creation_leaves_no_orphan_lock` (RA-04) |
| HB-FIX-02 | Namespace initialization coherent and deterministic under concurrent identical and divergent requests | `test_concurrent_identical_initializers_publish_one_coherent_pair` (6 + 4 threads), `test_concurrent_divergent_initializers_one_wins_other_refuses`, `test_crash_after_genesis_before_namespace_recovers_exactly`, `test_crash_after_namespace_before_genesis_recovers_exactly` (incl. the records-exist refusal), `test_namespace_head_mismatch_never_returns_success`, `test_relocated_store_retains_namespace_identity` |
| HB-FIX-03 | No public `mbp10`; legacy physical files resolve to `legacy_verified_replay_source` | `test_physical_pre_mbp1_file_resolves_to_the_public_legacy_source_kind`, `test_public_source_kind_refuses_the_physical_stem`, `test_physical_descriptor_keeps_truthful_provenance_without_public_capability`, `test_legacy_era_inventory_serializes_only_public_source_kinds`, `test_seed_inventory_hash_refuses_the_physical_stem_and_binds_the_legacy_kind`, the four scan tests in `test_public_source_kind_surface.py` (the literal appears exactly once in the IFVG lane, in the private resolver) |
| HB-FIX-04 | Invalid OOS assignments retain fit / fold / partition / protocol / reason | `test_invalid_oos_rows_retain_exact_provenance_and_no_coverage_is_bare`, `test_panel_descriptive_mode_retains_invalid_fit_provenance`, `test_trade_projection_retains_invalid_assignment_provenance` (extended by RB-05: a `"False"`-string input is refused before the projection) |
| HB-FIX-05 | Malformed bool / integer / ID evidence refused before coercion | `test_malformed_bool_integer_and_id_evidence_is_refused_before_coercion`, `test_oos_and_fold_feature_tables_validate_native_values_too` (`"False"`, `1`, `1.5`, `0.5`, `True`, `"0"`, `"0.1"`, `inf`, `object()`, `None` / `""` / `"None"` / `"nan"` / `"<NA>"` / `"NaN"` / `"null"`, a boolean inside a distance vector, a string in a numeric column of an invalid fold row); review round RB-05: the PIT rule, the candidate OOS index and the trade projection validate their consumed input columns natively before any coercion (`test_oos_and_fold_feature_tables_validate_native_values_too` extended) |
| HB-FIX-06 | Invalid fold-feature rows retain the complete reconciliation spine | `test_invalid_fold_feature_rows_retain_the_complete_reconciliation_spine`, `test_loader_refuses_a_stored_row_outside_the_bound_fold_fit_refs` (both extended by RB-02: a fit-bearing fold refuses a fit-less row at build and at load) |
| HB-FIX-07 | OOS schema identity and candidate populations exact at save and load | `test_wrong_schema_hash_is_refused_at_save_and_a_tampered_schema_at_reload`, `test_candidate_as_of_and_assignment_sets_must_be_exactly_equal`, `test_oos_payload_and_saver_bind_the_registered_schema_hash` |
| HB-FIX-08 | Manifest traversal, symlink escape, duplicate / reserved paths, malformed entries, invalid hashes fail typed | `test_malformed_manifest_entries_fail_typed_on_every_read_path` (23 entry shapes × 7 read paths), `test_a_manifest_without_the_envelope_entry_is_malformed`, `test_a_declared_byte_count_that_disagrees_with_the_file_is_a_hash_mismatch`, `test_symlink_escape_is_refused_before_the_file_is_opened` (skips only where the host refuses symbolic links), `test_invalid_store_locators_are_distinguished_from_absence`, `test_a_traversal_entry_in_a_trade_table_manifest_fails_typed` |
| HB-FIX-09 | Persisted studies require exact label and executed-trade artifact identities | `test_duplicate_label_candidates_are_refused_before_hashing`, `test_persisting_seam_proves_the_exact_label_artifact`, `test_persisted_reports_require_and_verify_the_exact_executed_trade_table`; review round RB-01: `test_every_persisting_save_requires_the_exact_label_proof` (every persisting save requires `label_identity_proof="exact"`; the regime study saves proven in `test_controlled_study_persists_and_reloads` / `test_cohort_model_floor_refusals_persistence_and_bundle_shape`) |
| HB-FIX-10 | Direct and runner seed saves canonicalize every aware timezone to UTC; canonical goldens unchanged | `test_direct_seed_save_canonicalizes_every_aware_timezone_to_utc` (stdlib UTC, pytz UTC / New York, zoneinfo UTC / Tokyo, fixed −05:00 / +09:30: same hash, same snapshot id, byte-identical sidecar, no caller mutation), `test_seed_canonicalization_covers_nested_containers_and_keeps_naive_datetimes`, `test_seed_production_runner_canonicalizes_non_utc_seed_datetimes`; review round RA-03: `test_seed_canonicalization_covers_nested_containers_and_keeps_naive_datetimes` extended (dict keys, sets, pydantic extras, `init=False` fields, pandas Timestamp; `datetime64` / `NaT` / nanoseconds refused) |
| HB-FIX-11 | Event-detail memory bounded by the registered per-partition row limit for normal and skewed lawful shapes | `test_one_path_larger_than_one_partition_is_split_and_reconstructed_exactly`, `test_one_path_block_and_skewed_shapes_obey_the_bound_and_reconstruct`, `test_partitioning_is_independent_of_iterator_chunking_and_repeat_is_byte_identical`, `test_total_row_refusal_publishes_nothing_and_cleans_every_partition`, `test_v1_budget_is_loadable_but_refused_by_the_row_bounded_writer`; review round: `test_cleanup_covers_a_partial_write_and_the_manifest_stage` (RB-03), `test_reader_binds_the_partition_bound_to_the_identity_bound_budget` (RB-04); the focused benchmark (§5 below, rerun over the fixed tree with the independent resident-batch observation) |
| HB-FIX-12 | Every real authority seam refuses an incomplete or corrupt supersession replacement chain | `test_every_seam_refuses_an_incomplete_or_corrupt_replacement_chain` (6 modes × bundle / verification ref / seed authorization / regime chain loader; the structure-only witness rule is shown to still pass the four no-head-move corruptions), `test_effective_time_and_provenance_rules_are_proven_from_the_artifacts`, `test_a_record_recorded_before_its_replacement_was_approved_is_refused`, `test_pipeline_launch_refuses_a_corrupt_replacement_before_any_stage`, `test_every_real_seam_shares_the_complete_proof_call_path` |
| HB-FIX-13 | M0–M3, Strategy-Core, Trade-Lab, immutable artifacts, S11, MBP-1 offline-only boundaries unchanged | no M0–M3 module touched; Strategy-Core clean at pin `a4e3303`; Trade-Lab untouched; `test_r61_fix_goldens.py` green (core replay / account simulation / feature-block registry / protocol goldens); `order_flow_depth_policy="mbp1_only_v1"` untouched; `data/` gained zero files |

---

## 4. Tests and counts

| Run | Result | Raw |
|---|---|---|
| Red-first baseline (the release's test files against the parent `e56f937`, source unchanged) | 68 failed / 131 passed / 5 collection errors | `_red_first_baseline_pytest.txt` |
| Per-workstream targeted regressions during implementation (A–G) + the WS-G follow-up rerun | all green (WS-G: the 2 expected failures → 43 passed on the rerun) | `_targeted_pytest.txt` |
| Focused review round: the direct targeted suites over the fixed tree (24 files) | 295 passed / 1 failed on the first cut (the kernel's empty fit id for fit-less folds — RB-05 narrowed), 25 passed on the rerun of the two OOS files | `_fix_round_pytest.txt` |
| Exact reuse + consumer regression over the fixed tree (plan §12.3 item 9) | 93 passed (6:11) | `_exact_reuse_pytest.txt` |
| **Release-final full suite, environment as-is** | **2195 passed, 0 failed** in 1735.02 s (0:28:55), exit 0, no warnings summary (`filterwarnings = error`) | `_final_pytest.txt` |
| **Release-final full suite, provider keys cleared** (`env -u POLYGON_API_KEY -u DATABENTO_API_KEY`) | **2195 passed, 0 failed** in 1777.70 s (0:29:37), exit 0; both keys absent in the run environment | `_final_pytest_keys_cleared.txt` |

Collection: **2195 tests** (2102 at `e56f937` → +93 net new collected tests: 60 new test functions including the parametrized cases, plus 9 extended existing tests). The counts coincide across the two environments (no credential-conditional test). Session note: the first tool-backgrounded full-suite chain was killed by the harness at ~22 minutes (44 % of the as-is run; nothing else was affected); both suites then ran to completion as one detached process (`_PROGRESS_CHECKPOINT.md`, 04:50Z).

Red-first evidence: the release's test files were run against the parent commit `e56f937` in a throwaway worktree with the source tree unchanged (`_red_first_baseline_pytest.txt`): **68 failed, 131 passed, 5 errors** — the 68 are behavioural failures at run time (the four lock / namespace / OOS / manifest / seed / event-detail families fail on the old behaviour or on a new attribute at call time); the 5 errors are collection errors of files that import symbols this release adds at module level (the two new files `test_authority_chain_seams.py` and `test_public_source_kind_surface.py`, `test_trading_calendar.py` for the new resolver symbols, and `tests/propsim/test_event_detail_streaming.py` / `test_account_event_detail.py` for `EVENT_DETAIL_BUDGET_V2`); the 68 behavioural failures span 15 files (27 manifest-probe cases, 8 seed-canonicalization cases, 7 lock cases, 7 namespace cases, 4 OOS-provenance cases, 3 native-validation cases, 3 seed-production cases, 2 label cases, and one each in the fold-feature, fold-evidence, stratification, bounded-verification, pipeline-integrity, verification-window and review-fix files). Tests were written alongside each workstream's implementation rather than strictly before it; the baseline run is the red evidence.

---

## 5. Capacity numbers (plan §9.3; `CAPACITY_BENCHMARKS.md` / `.json`, `_capacity_benchmark.txt`)

`HARDENING_CAPACITY_POLICY_V1`, partition bound policy `event_detail_partition_row_bound_v2` (`max_rows_per_partition = 50,000` — the benchmark's measured row-group size; no registered ceiling lowered). The benchmark ran once over the pre-review tree (`_capacity_benchmark_before_review_round.txt`: **overall PASS**, minimum available RAM at any run start 8.807 GiB) and was rerun over the FINAL tree after the review round (RB-03 changed the writer's cleanup order, RB-04 added the independent resident-batch observation): `_capacity_benchmark_rerun_attempt1.txt` — **every numerical gate PASS**, every output hash IDENTICAL to the pre-review run (the writer's bytes are unchanged), but **`overall: FAIL` on the harness's environmental precondition only**: the minimum available RAM at a run start was 7.868 GiB (B1) / 7.956 GiB (B2) against the ≥ 8 GiB gate — no suite was running; the host had 8.16 GiB free with the owner's own applications (Chrome, VS Code, Discord, memory compression) holding the rest, and the benchmark process's own baseline pushed the measurement below the gate. The retry (attempt 2, `_capacity_benchmark.txt`, after both full suites with the machine otherwise idle) repeated the result exactly: every numerical gate PASS (B1 1M rows peak RSS +0.338 GiB, B2 +0.481 GiB; identical hashes), minimum available RAM 7.647 GiB (B1) / 7.905 GiB (B2) — `overall: FAIL` on the precondition only. The benchmark harness and `HARDENING_CAPACITY_POLICY_V1` were NOT modified to pass (plan §9.3 forbids lowering a gate autonomously); the owner can reproduce the formal `overall: PASS` artifact on an idle host with the same command. Windows 11, 16 logical CPUs, 31.1 GiB RAM.

| B1 (writer, `normal` 200 events/path; final tree) | 250k | 500k | 1M |
|---|---:|---:|---:|
| peak RSS increase (repeat) | 0.223 (0.230) GiB | 0.278 (0.276) GiB | 0.326 (0.325) GiB |
| Python allocation peak | 35.8 MiB | 35.8 MiB | 35.8 MiB |
| wall | 2.1 s | 4.1 s | 7.8 s |
| partitions / max partition rows / observed resident batch | 5 / 50,000 / 50,000 | 10 / 50,000 / 50,000 | 20 / 50,000 / 50,000 |
| output hash repeat | identical (`97bda11c7c22`) | identical (`db0a25a028db`) | identical (`2c8f09db151d`) |

| B1 shape at 1M rows (final tree) | paths | partitions | observed resident batch | peak RSS increase | Python alloc peak | wall | repeat hash |
|---|---:|---:|---:|---:|---:|---:|---|
| `skewed` (5 paths carry 90 %) | 505 | 21 | 50,000 | 0.326 GiB | 217.5 MiB | 8.7 s | identical (`86f9ef4338d0`) |
| `dense` (250 paths = one block) | 250 | 20 | 50,000 | 0.328 GiB | 35.7 MiB | 8.1 s | identical (`13702335fd42`) |

| Gate (final tree) | measured | limit | result |
|---|---:|---:|---|
| minimum available RAM at run start ≥ 8 GiB (environmental precondition) | 7.868 GiB (B1) / 7.956 GiB (B2) | 8.000 GiB | FAIL (environmental precondition only) |
| 1M-row peak RSS increase ≤ 1.5 GiB | 0.326 GiB (skewed 0.326, dense 0.328) | 1.5 GiB | PASS |
| 1M-row Python allocation peak ≤ 512 MiB | 35.8 MiB (skewed 217.5 — the generator's whale walk results, not the writer; dense 35.7) | 512 MiB | PASS |
| 1M-row wall ≤ 300 s | 7.8 s | 300 s | PASS |
| slope 500k→1M within policy (≤ 1.25 × max(slope 250k→500k, 64 B/row)) | 104.7 B/row | 296.1 B/row | PASS |
| registered 10M-row projection ≤ min(6 GiB, 50 % available) | 1.204 GiB | 3.934 GiB (50 % of the 7.868 GiB measured) | PASS |
| serialized 10M-row projection ≤ 2 GiB | 0.329 GiB | 2 GiB | PASS |
| repeat output byte-identical | identical (every size, every shape; identical to the pre-review run) | identical | PASS |
| maximum resident writer batch (OBSERVED at the writer's Parquet seam — review RB-04) ≤ max_rows_per_partition (every shape, every size) | 50,000 rows | 50,000 rows | PASS |

B2 (the external regime-summary aggregation, not redesigned) re-ran unchanged on the final tree: 1M rows peak RSS +0.478 GiB, Python peak 134.6 MiB, wall 37.5 s, projection 0.834 GiB (limit 3.978 GiB) — every numerical gate PASS; the same environmental precondition miss (7.956 GiB).

## 6. Identity re-mints and preserved goldens (plan §11)

Re-minted (semantics changed; synthetic only — no real artifact existed under the prior contracts):

| Family | Reason |
|---|---|
| Source inventories / verification windows / seed authorizations whose payload serialized `"mbp10"` | the public kind of a historical physical partition is now `legacy_verified_replay_source` (`SourcePartitionRef.source_kind`, `seed_chain_source_inventory_hash`, shortlist `trading_day_refs`) |
| Seed snapshots created from non-UTC datetime representations | the canonical hash is the identity; a UTC-represented seed keeps its existing hash and bytes |
| Account / portfolio simulations under the v2 event-detail policy with the default budget | `EVENT_DETAIL_BUDGET_V2` (the V1 budget's serialization is unchanged, so identities minted under an explicit V1 budget are preserved) |
| Regime OOS assignment / fold-feature artifacts whose invalid rows previously collapsed | invalid rows now carry fit / fold / partition / reason (table bytes and the consulted hash differ only where such rows existed) |
| Event-detail sidecar bytes and detail manifests | partition key `(path_block_id, partition_ordinal)`, new manifest fields (post-materialization facts; the simulation id moves only through the budget) |

Preserved (proven): Strategy-Core pin `a4e3303…` (clean); the fixed M0–M3 lane untouched; R6.1-FIX goldens (`GOLDEN_B0_BUNDLE_ID`, `GOLDEN_CANDIDATE_PROTOCOL_ID`, `GOLDEN_CORE_REPLAY_ID`, `GOLDEN_ACCOUNT_SIMULATION_ID`, `GOLDEN_FEATURE_BLOCK_REGISTRY_HASH`) green; R5B formulas, model protocol parameters, KMeans fit numerical identities (the fit sidecar bytes are unchanged — validation was added, coercion rules were not), prop-firm rule contracts, the S11 blocked reason, `order_flow_depth_policy="mbp1_only_v1"` — all untouched. The review round adds no identity change: `label_identity_proof` is run-level (study payloads and ids unchanged — proven by the same-id assertion in `test_every_persisting_save_requires_the_exact_label_proof`); `lock_mutex_failed` and `store_namespace_initialization_failed` are registered failure vocabulary, not artifact fields; the event-detail partition bytes, manifest bytes and simulation identities are unchanged (only the writer's cleanup order and the reader's bound binding changed — the rerun benchmark's output hashes are the evidence).

---

## 7. Focused review (plan §13; `FOCUSED_REVIEW.md`)

Exactly two read-only reviewers, one pass, over the release diff (sha256 `662bc4423bc3dad0…`, 50 files): Reviewer A (lock / namespace / chain seams / seed canonicalization — `_review_A.md`) and Reviewer B (regime evidence / manifests / source kind / event-detail capacity — `_review_B.md`); each ran four single test files and a handful of synthetic scratch probes, neither edited a repository file, spawned an agent or opened anything under `data/`.

| Reviewer | BLOCKER | MAJOR | MEDIUM | MINOR | Disposition |
|---|---:|---:|---:|---:|---|
| A | 0 | 1 (RA-01 silent release of a held lock replaced by a foreign token) | 0 | 3 (RA-02 non-contention mutex errors typed as contention; RA-03 uncovered datetime carriers; RA-04 orphan lock after a post-creation read failure) | all four FIXED |
| B | 0 | 1 (RB-01 exact-label proof opt-in for direct API callers) | 2 (RB-02 fit-bearing fold accepted fit-less rows; RB-03 cleanup missed a partial file and the manifest stage) | 2 (RB-04 tautological resident-batch gate + loose reader bound; RB-05 input `valid` coerced before validation) | all five FIXED |

Lanes with no finding: namespace initialization (HB-FIX-02), the complete chain proof at every enumerated seam (HB-FIX-12), the central manifest validator on every store read path (HB-FIX-08), the public source-kind boundary (HB-FIX-03), OOS schema identity / candidate-set equality (HB-FIX-07). Both MAJOR items were plan-contract gaps (§4.1 release typing; §7.2 persisted-study proof), not HB-FIX assertion failures. The nine fixes landed once (`patch_review_round.py` / `patch_review_round_tests.py` in the job tmp dir; the code is the commit), the direct targeted suites reran (`_fix_round_pytest.txt`: 295 passed / 1 failed on the first cut — the pre-existing empty-fit-id convention, RB-05 narrowed, the two OOS files 25 passed on the rerun), then the consumer regression and the final gate ran over the fixed tree (§9). Five consolidated carry-forwards are recorded in `FOCUSED_REVIEW.md` (not gates).

---

## 8. Access / scope proof (plan §12.3 item 7)

| Proof | Result |
|---|---|
| protected access counters / sealed access counters | 0 / 0 (no source path was constructed; every test is synthetic under `tmp_path`) |
| new real-data artifact | none — `find data -type f -newer <plan file>` → 0; no `STORE_NAMESPACE.json`, allowlist marker, mutex file or seed / verification store under `data/` |
| seed-production run / bounded real verification / full pipeline / catalog activation | none |
| June 11 / sealed range | never accessed; every date literal in new tests lies inside the synthetic allowlist |
| user-owned hunks | byte-identical to the baseline capture (patch sha256 `b12b4200379a101e…`) |

---

## 9. Final gate (plan §12.3)

| # | Gate item (plan §12.3) | Status | Evidence |
|---|---|---|---|
| 1 | Complete suite, normal environment | **PASS** — 2195 passed / 0 failed (0:28:55) | `_final_pytest.txt` |
| 2 | Complete suite, provider credentials cleared | **PASS** — 2195 passed / 0 failed (0:29:37) | `_final_pytest_keys_cleared.txt` |
| 3 | 0 failed; warnings-as-errors; no new unallowlisted warning | **PASS** — `filterwarnings = error` unchanged from HARDENING-BACKEND (one exact third-party rule); no warnings summary in either raw output | both logs |
| 4 | `python -m ruff check src tests scripts`; `git diff --check` | **PASS** — clean over the whole tree; clean over the tracked and the 3 intent-added new files; no production `assert` in any touched src module | `_ruff_and_diffcheck.txt` |
| 5 | Fixed identity / golden suites | **PASS** — `test_r61_fix_goldens.py` green in every run; the R6.1-FIX goldens, the Strategy-Core pin `a4e3303` (clean) and the M0–M3 lane unchanged (§6) | `_exact_reuse_pytest.txt`, both full suites |
| 6 | Focused capacity benchmark once | **FAIL (environmental precondition only)** — every numerical gate PASS on the final tree with output hashes identical to the pre-review PASS run; the environmental ≥ 8 GiB available-RAM precondition was missed on both reruns over the final tree (7.868 / 7.956 GiB, then 7.647 / 7.905 GiB at run start) because the owner's own applications held the host's memory — the pre-review run met it at 8.807 GiB and the writer's bytes are proven unchanged by the identical output hashes; the harness and `HARDENING_CAPACITY_POLICY_V1` were NOT modified to pass (plan §9.3 forbids lowering a gate autonomously); an owner rerun on an idle host (`python scripts/hardening_capacity_benchmark.py --out-dir <this folder>`) produces the formal PASS artifact | `_capacity_benchmark.txt`, `_capacity_benchmark_rerun_attempt1.txt`, `_capacity_benchmark_before_review_round.txt`, `CAPACITY_BENCHMARKS.md` / `.json` |
| 7 | protected access = 0; sealed access = 0; no new real-data artifact; no seed-production run; no bounded real verification run; no full pipeline run; no catalog activation | **PASS** — §8 (`find data -type f -newer <plan>` → 0 at every checkpoint incl. after the gate; no namespace / mutex / head marker under `data/`) | this report |
| 8 | Surviving pre-existing user-owned hunks byte-identical to the baseline capture | **PASS** — the four user-owned docs' worktree diff sha256 `b12b4200379a101e…` at baseline and before the commit; after the commit (the three shared docs staged as HEAD + lane transforms and replayed onto the worktree) `_surviving_shared_doc_diff.patch` is IDENTICAL to the capture modulo `index` / `@@` header lines | `stage_shared_docs.py`, `_surviving_shared_doc_diff.patch` |
| 9 | Exact reuse on a representative synthetic pipeline after the corrected identities | **PASS** — second attempts reuse every regime stage, reused children adopted only by verified reproduction, orchestrator full-reuse repeat identity, the prop seam, the stratification gate and the external aggregation: 93 passed over the fixed tree | `_exact_reuse_pytest.txt` |
| — | One focused two-reviewer pass; in-scope findings fixed once | **PASS** — 0 blockers; 9 findings FIXED; 5 carry-forwards | `FOCUSED_REVIEW.md`, `_review_A.md`, `_review_B.md` |
| — | Commit boundary: one release-scoped commit parented by `e56f937`; no push, no merge | **PASS** — `a5eee1a` (55 files: the 52 release files of `_commit_file_list.txt` + the 3 shared docs) | `HARDENING-BACKEND-FIX.patch` / `.bundle` |

Plan §16 completion criteria: every box holds, with one honest qualification: the benchmark's formal `overall: PASS` artifact over the final tree is outstanding for the environmental reason in item 6 (every numerical capacity gate passed twice on the final tree and the pre-review run passed the precondition).

Package: `HARDENING-BACKEND-FIX.patch` (`git format-patch --stdout e56f937..a5eee1a`; 447,911 bytes; sha256 `95fbd9463f79c603b3a8e75cf9f9df424017dc802e2a483dac1ddae3f9f7d3af`) + `.sha256`; **`HARDENING-BACKEND-FIX.bundle`** (the prerequisite-complete chain `179a2c9..a5eee1a` = R5B.1 → R6.1 → R6.1-FIX → HARDENING-BACKEND → HARDENING-BACKEND-FIX; requires R6 `179a2c9`; carries the branch head; `git bundle verify` OK; 788,937 bytes; sha256 `68f094be5e506bfafd5910cecb93cd94290736394bfe2eefd2e9b2617af9a650`) + `.sha256`; `_commit_file_list.txt`. Not pushed. Not merged.

---

## 10. Remaining owner actions (unchanged from HARDENING-BACKEND)

1. Select the permanent logical-day verification window from `HARDENING-BACKEND/VERIFICATION_WINDOW_SHORTLIST.md` (the June proposal is ineligible as stated).
2. Sign the `SeedProductionAuthorizationRef`; run the seed-only chain; review the concrete seed.
3. Sign the `VerificationAuthorizationRef`; `init` the real verification store as a `test` namespace (`scripts/ifvg_store_namespace.py init … --confirm`; pass `--store-instance-id` when repeating an interrupted initialization); register the one program allowlist; run `scripts/ifvg_bounded_verification.py run`.

Carry-forwards recorded by this release (not gates): the R6.1-FIX / HARDENING-BACKEND carry-forwards remain; new: (i) direct API callers of the three study runners that pass a `label_artifact_id` without `label_policy_id` keep the R6.1-FIX trust semantics (the pipeline seams always pass the policy — exact); (ii) the five consolidated reviewer carry-forwards in `FOCUSED_REVIEW.md`: non-store manifest families outside §7.1's store scope; the declared-only `replay_bytes_only` capability and the scan regex's dotted-literal blind spot; the untyped `OverflowError` for a huge int and the single-path whale held upstream of the writer; structure-only witness minting for unpersisted synthetic authorizations plus two performance / telemetry notes; the datetime-keyed `DayArtifacts` map.

---

## 11. UI transition decision

```text
backend_dev_complete_for_ui = true
ui_implementation_may_begin = true
owner_verification_still_required = true
formal_release_acceptance = blocked_by_R1_VerificationAuthorizationRef
```

The separate UI/UX implementation branch may rebase onto the finalized backend contracts and begin. The owner seed-production and bounded-verification workflow remains a later action performed through the corrected UI and the existing backend authorization contracts.
