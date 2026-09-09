# HARDENING-BACKEND-FIX — Focused Final Review (plan §13)

**Reviewed diff:** the uncommitted release working tree over parent `e56f9376a5b4ba269f7fa11cdc6e37b08200638f`
(HARDENING-BACKEND) — `git diff HEAD -- src scripts tests docs/DECISIONS.md`, 50 files,
+5,631 / −502, sha256 `662bc4423bc3dad02ac54eef03ea50c0e9458bce09e359f958e16f4b77cd716b`
(`review_diff.patch`). Reviewed commit after the round: `a5eee1a` (`a5eee1aa1e04127809eb57fec1e61331a1d692ef`).
**Reviewers (plan §3.2 — exactly two, one pass, read-only, ≤5 findings + ≤5 carry-forwards, ≤1,500 words each):**
Reviewer A = owner-decision lock, namespace initialization, supersession-chain authority seams, seed
canonicalization (`_review_A.md`, 1,256 words; ran `test_owner_decision_lock` 15, `test_store_namespace` 15,
`test_child_replay` 18, `test_authority_chain_seams` 10 — all passed; 4 scratch probes under the job tmp dir).
Reviewer B = regime assignment / fold evidence, manifest validation, legacy source-kind boundary, event-detail
capacity (`_review_B.md`, 1,391 words; ran `test_store_sidecar_probe` 31, `test_regime_fold_feature_evidence` 5,
`test_event_detail_streaming` 14, `test_regime_oos_assignment` 10 — all passed; 5 scratch probes).
Both reviewers affirmed the access rule ("did not open, list or construct any path under `data/`"; Reviewer A's one
`git status --porcelain` call printed pre-existing untracked entry names under `data/ifvg_datasets/` without
opening them — those entries pre-date the plan file by mtime, see §8 of `COMPLETION_REPORT.md`).
No reviewer spawned an agent, ran a long suite, or edited a repository file other than its own report.

**Verdict:** 0 BLOCKER. Reviewer A: 1 MAJOR (RA-01) + 3 MINOR; Reviewer B: 1 MAJOR (RB-01) + 2 MEDIUM + 2 MINOR.
HB-FIX-01, -02, -03, -07, -08, -10 and -12 were affirmed on the code and the tests as written; the two MAJOR items
are plan-contract gaps (§4.1 release typing, §7.2 persisted-study proof), not assertion failures. All nine
in-scope findings were fixed ONCE in the single permitted round (plan §3.2 / §13); the affected targeted suites
were rerun (`_fix_round_pytest.txt`), then the final gate ran over the fixed tree (`COMPLETION_REPORT.md` §9).

## Findings and dispositions

| ID | Fixed checklist item | Finding | Severity | Disposition | Test proving resolution |
|---|---|---|---|---|---|
| RA-01 | §4.1 "release() must raise a typed failure if it cannot read or verify its own lock" (HB-FIX-01 context) | A writer that HELD the lock and found, at release, a body carrying ANOTHER writer's token returned silently — the one state that means a publication may have overlapped (`verify_held()` on the same state raised `lock_lost`). | MAJOR | **FIXED** — `owner_decision_lock.py::release()`: when `was_held` and the body carries a foreign token → typed `lock_release_failed` ("replaced by a non-conforming actor … a publication may have overlapped"); the foreign lock is left untouched; a no-longer-held release stays silent. | `test_release_of_a_held_lock_replaced_by_another_writer_is_typed` |
| RA-02 | §4.1 "persistent I/O failure is a typed error, never silence" | `ExclusiveFileMutex._try_lock` swallowed EVERY `OSError` and spun to the deadline, so a persistent non-contention failure of the mutex file (EBADF / EIO / ENOSPC) was reported as contention (`lock_held_by_live_holder`, `store_namespace_initialization_busy`). | MINOR | **FIXED** — `file_mutex.py`: only the platform's contention errno is contention (`EACCES` / `EDEADLOCK` for `msvcrt.locking`, probed on this host; `EWOULDBLOCK` / `EAGAIN` for `fcntl.flock`); anything else is the typed `FileMutexError`; the lock maps it to the new registered reason `lock_mutex_failed`, namespace initialization to `store_namespace_initialization_failed`. | `test_mutex_persistent_io_error_is_not_reported_as_contention` |
| RA-03 | §8.2 "every nested value … mappings" / HB-FIX-10 uncovered carriers | Not canonicalized: `pandas.Timestamp` (kept as a subclass the seed sandbox unpickler does not admit), datetime dict KEYS, set / frozenset members, pydantic `extra="allow"` values, `numpy.datetime64` (passed through); a dataclass `init=False` field was silently RESET to its default by `dataclasses.replace`. Not reachable by today's `IfvgDaySeed` graph (dataclasses with init fields, stdlib datetimes and dates only). | MINOR | **FIXED** — `child_replay.py::canonicalize_seed_datetimes`: dict keys and set / frozenset members canonicalized; pydantic extras canonicalized through `model_copy`; every dataclass field (`init=False` included) canonicalized onto a shallow copy; a `pandas.Timestamp` is rebuilt as the stdlib datetime it represents; sub-microsecond precision, `NaT` and `numpy.datetime64` are refused with the typed `SeedSnapshotError` (never dropped). UTC goldens unchanged (`test_direct_seed_save_canonicalizes_every_aware_timezone_to_utc` still green). | `test_seed_canonicalization_covers_nested_containers_and_keeps_naive_datetimes` (extended) |
| RA-04 | §4.1 typed reads / "a lost ordinary lock token aborts before publication" | After a successful `O_EXCL` create, a persistent read failure raised `lock_read_failed` with OUR live lock left on disk and `__exit__` never running — a non-reclaimable orphan denying the store for the life of the process. | MINOR | **FIXED** — `owner_decision_lock.py::acquire()`: on a read failure right after creation the token-safe `_discard_partial_lock()` runs (removed / quarantined) before the typed error is re-raised with the disposition; `_held` stays false. | `test_read_failure_after_exclusive_creation_leaves_no_orphan_lock` |
| RB-01 | §7.2 "Persisted controlled studies require the exact immutable label-artifact identity" (HB-FIX-09) | The exact-label proof was opt-in: `label_artifact_id=<any 64-hex>` without `label_policy_id` skipped `assert_exact_label_artifact`, still stamped `label_identity_source="label_artifact"`, and the three `save_*_study` seams refused only `content_hash_unpersisted` — a study over an arbitrary caller string was persistable outside the pipeline seams. | MAJOR | **FIXED** — a run-level `label_identity_proof` on all three study runs (`exact` only when the registered policy proved the id; `caller_supplied_unproven` for an id without the policy — in-memory use only; `content_hash_unpersisted` for the helper form); `comparison_rows.assert_persistable_label_proof` is called by every persisting save (`save_controlled_feature_study`, `save_regime_controlled_study`, `save_regime_cohort_model_study`) and refuses anything but `exact`. Payloads and identities unchanged (the proof is not identity-bearing); the pipeline seams already pass the policy. | `test_every_persisting_save_requires_the_exact_label_proof`; `test_persisted_studies_require_the_exact_label_artifact_id` (now passes the policy); the refusal checks in `test_controlled_study_persists_and_reloads` and `test_cohort_model_floor_refusals_persistence_and_bundle_shape` |
| RB-02 | §6.4 fold-feature invariants (HB-FIX-06) | A fold whose `FoldFitRef` binds a fit accepted a row typed `no_valid_regime_fit` with a null `regime_fit_id` at build, table-bytes and load time — a known invalid assignment of the applicable fit could be collapsed into fit-less absence and reloaded as lawful (the builder never emits the shape; the hole was verification-side). | MEDIUM | **FIXED** — `regime_fold_features.py::assert_fold_feature_spine_bound` (run at build AND load): a fit-bearing fold refuses any row whose `regime_fit_id` is null (`assignment_provenance_incomplete`, "… name no fit although the artifact binds FoldFitRef …"). | `test_invalid_fold_feature_rows_retain_the_complete_reconciliation_spine` (extended); `test_loader_refuses_a_stored_row_outside_the_bound_fold_fit_refs` (extended with the collapsed-row load case) |
| RB-03 | §9.2 "cleanup after failure" (HB-FIX-11) | (a) The detail-manifest stage ran outside the `try/except BaseException`, so a refusal there left every partition behind in a caller-owned directory; (b) `written.append(path)` ran after `_write_parquet`, so an `OSError` mid-write left a partial partition file behind. | MEDIUM | **FIXED** — `event_detail.py::build_account_event_detail`: the partition path (and the manifest path) is registered BEFORE its write; the whole manifest stage runs inside the cleanup guard; a pre-existing (foreign) manifest is still refused and never removed. | `test_cleanup_covers_a_partial_write_and_the_manifest_stage` |
| RB-04 | §9.3 gate "maximum resident writer batch" (HB-FIX-11) | The benchmark's resident-batch gate compared the manifest's own `max_partition_rows` with the bound (the writer raises above the bound, so the gate could not fail and measured nothing independent); the reader's bound proof consulted the manifest's loose top-level `max_rows_per_partition` instead of the identity-bound budget. | MINOR | **FIXED** — the benchmark observes every flushed table's row count at the writer's Parquet seam (`_write_parquet`, an independent observation) and gates `max_resident_rows_observed ≤ max_rows_per_partition` (`CAPACITY_BENCHMARKS.md` rerun over the fixed tree); the reader binds the bound to `manifest["budget"]["max_rows_per_partition"]` (proven equal to the payload's budget) and refuses a disagreeing top-level field. | `test_reader_binds_the_partition_bound_to_the_identity_bound_budget`; the rerun benchmark gate row |
| RB-05 | §6.1–6.2 (HB-FIX-04 / -05) | `regime_for_trades` and `candidate_fold_oos_assignment` / `assign_panel_regimes_to_candidates` coerced their INPUT `valid` column (`bool(row["valid"])`, `astype(bool)`) before any native validation: a `"False"` string projected as valid (production callers pass verified-loaded frames; direct API use only). | MINOR | **FIXED** — `regime_oos_assignment.validate_consumed_natively` validates the consumed columns of the input frame natively (`native_value_refused`) before the seam reads them, at the panel PIT seam, the candidate OOS index and the executed-trade projection (`regime_assignment_sources.py`); the fit-assignment seams deliberately exclude `regime_fit_id` (the kernel writes an EMPTY fit id for a fold that had no fit and the seam maps it to null itself — one pre-existing test documents that convention and caught the first cut). | `test_trade_projection_retains_invalid_assignment_provenance` (extended); `test_oos_and_fold_feature_tables_validate_native_values_too` (extended) |

Lanes with no finding (both reviewers): namespace initialization (HB-FIX-02), the complete supersession-chain
proof at every real seam (HB-FIX-12 — every caller of `assert_authorization_bound_to_store` /
`load_supersession_chain` enumerated), the central manifest validator and `_sidecar_path` (HB-FIX-08 — every
store read path goes through `_verified_manifest`), the public source-kind boundary (HB-FIX-03 — the stem is
confined to the private resolver; nothing downstream branches on it), OOS schema identity / candidate-set
equality (HB-FIX-07 — the consulted hash is recomputed from the exact verified bytes).

## Carry-forward (out of scope; ≤5 — consolidated from both reviewers, not gates)

1. Non-store manifest families still read `manifest.json` with `json.loads` + `entry["path"]` outside the central
   validator: `ifvg/artifact_io.py`, `context_run_store.py`, `fsm_audit_io.py`, `replay_chart_store.py`,
   `ml/logistic_model.py` (M0–M3 / exploration lanes; outside §7.1's store scope).
2. `PhysicalSourceDescriptor.replay_bytes_only` and `physical_descriptors_from_permitted_source_hashes` have no
   consumer in `src/` — the capability statement is declared, not enforced (the legacy boundary is enforced by
   `mbp1_source_artifact._require_no_legacy_provenance`); the public-surface scan regex misses dotted literals
   such as `"mbp10.parquet"` (none present) and `physical_source_descriptor` resolves a partition key by its
   last segment only.
3. `validate_native_values` raises an untyped `OverflowError` for a Python int beyond float range in a float
   column; a lawful single-path whale (10M events on ONE path) is held upstream in the walk result's `events`
   tuple, outside the writer's bound and unmeasured by the benchmark (skewed shape = 180k events / path).
4. `synthetic_seed_production_authorization(..., persist=False)` mints its witness structure-only (harmless — an
   unpersisted envelope cannot pass `verify_seed_production_authorization`); `verify_complete_owner_authority_chain`
   re-reads the namespace file twice per chain record (performance only); `_reclaim` records `reclaimed_from`
   before the creation retry (telemetry only).
5. `day_artifacts.py` keys one `DayArtifacts` map by datetime — if such a map ever enters `IfvgDaySeed`, the
   dict-key canonicalization landed by RA-03 is what keeps the seed identity stable.
