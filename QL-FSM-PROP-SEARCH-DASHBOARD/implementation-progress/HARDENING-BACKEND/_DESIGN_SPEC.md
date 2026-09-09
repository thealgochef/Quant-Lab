# HARDENING-BACKEND — cross-workstream design spec (implementation record)

Written 2026-09-02 before the parallel workstreams started. Every workstream
codes against the API below; deviations are recorded in `DEVIATIONS.md`.

## Shared core (landed first, main agent)

`search/store_namespace.py` (§4.1, F-11)

- `StoreNamespacePayload{namespace_schema_version=1, namespace_class: research|test, store_instance_id: 32-hex uuid (never a path hash), authority_genesis_id = genesis_id_for(instance, class)}`; `StoreNamespaceEnvelope{store_namespace_id = hash(payload), payload}`; file `<root>/STORE_NAMESPACE.json`, immutable, re-verified (id hashes payload) on every load.
- `initialize_store_namespace(root, *, namespace_class, store_instance_id=None)` — the explicit one-time migration (class is an argument, never inferred); idempotent on identical; `store_namespace_divergent` otherwise; writes the GENESIS head first, then the namespace file.
- `initialize_test_namespace(root)` — fixture helper (class `test`).
- `load_store_namespace(root)` / `require_store_namespace(root, *, expected_class=None)` / `namespace_class_of(root) -> str|None` (None = unmarked; corrupt = typed refusal).
- `path_looks_like_research_store(root)` — DEFENSE-IN-DEPTH only; `assert_namespace_deployment_coherent(root, ns)` refuses a `test` namespace under a research-looking path.
- `SupersessionHeadWitness{store_namespace_id, line_count, head_sha256}`; `genesis_head_witness(ns)`; `chain_head_digest(prior, record_id)`.
- Head pointer `owner_decisions/SUPERSESSIONS.head` = `{head_schema_version: 2, store_namespace_id, record_id|null, line_count, head_sha256}`; genesis: `record_id=null, line_count=0, head_sha256=authority_genesis_id`. Missing head = `supersession_head_missing` (corruption).
- `StoreNamespaceError(PermissionError).reason ∈ STORE_NAMESPACE_FAILURE_REASONS`.

`search/supersession_chain.py` (§4.2, F-12)

- Store `owner_decision_supersessions/<record-id>/`; `OwnerDecisionSupersessionPayload{store_namespace_id, superseded_decision_id, replacement_decision_id, prior_head_record_id|null, prior_head_sha256, prior_line_count, reason, effective_at, owner_evidence_ref}`.
- `verify_chain_structure(root)` walks head → genesis (record loads only, no decision loads); `load_supersession_records(root) -> tuple[SupersessionRecord]` (consumer projection: `superseded_artifact_id`, `replacement_artifact_id`, `recorded_at`, `line_number`, `head_sha256`).
- `current_supersession_head_witness(root)`; `assert_head_witness_current(root, witness)` — refuses missing / shorter (`supersession_head_shorter_than_witness`) / different (`supersession_head_witness_mismatch`).
- `publish_supersession(root, *, superseded_decision_id, replacement_decision_id, reason, effective_at, owner_evidence_ref, verify_transition=None, lock=None)` — under `OwnerDecisionLock`: verify namespace + head + (caller's) transition, save the immutable record (temp dir → verify → atomic publish), re-verify the lock token, atomically replace the head. Identical replay = reuse; divergent = `supersession_divergent_replay`.

`search/owner_decision_lock.py` (§4.3, F-13)

- `OwnerDecisionLock(root, *, wait_seconds=30, heartbeat_timeout_seconds=60, poll_seconds=0.05)` context manager; body `{lock_schema_version, pid, process_start_token, lock_token, host, created_at, heartbeat_at}`; `refresh()` (heartbeat, atomic), `verify_held()` (token must be ours → else `lock_lost`), `release()` (own token only), `reclaimed_from`.
- Reclaim only when heartbeat age > timeout AND `process_liveness(pid, start_token, host) == "dead"` (no such pid / exited / PID reuse); other host / malformed / unassessable = `lock_holder_liveness_unknown` or `lock_body_malformed` → timeout, never reclaimed.
- `current_process_start_token()` (Win32 creation time via ctypes; `/proc/<pid>/stat` starttime on Linux; None elsewhere).

## Authority rules (WS-A applies them to the consumers)

- Owner-decision artifacts: `OwnerDecisionArtifactPayload.store_namespace_id` (required, 64-hex) must equal the store's namespace id at persist AND at load; an unmarked store refuses both (`store_namespace_missing`). Synthetic provenance requires class `test` (semantic) AND not a research-looking path (defense in depth).
- `assert_run_scope_lawful_for_root(root, run_scope)`: `synthetic_fixture` requires class `test` when the store is marked; in an UNMARKED store the synthetic scope can unlock no owner authority (owner artifacts cannot exist there) and only the path check applies; the research class refuses the synthetic scope.
- `load_supersession_chain(root)` keeps its name/signature and returns `SupersessionRecord`s from the immutable chain (consumers unchanged: `record.superseded_artifact_id`).
- `OwnerAuthorizationBundle` gains `store_namespace_id` + `supersession_head_witness: SupersessionHeadWitness`; `validate_owner_authorization(..., store_root=)` verifies both. `VerificationAuthorizationRef` gains the same two fields; `validate_verification_run(..., store_root=)` verifies them. Real charters (`save_charter`) with a bundle must match the store namespace + current witness; synthetic-marker charters are refused in a `research` namespace (semantic) and under a research-looking path (defense in depth).
- Real launches (`executors.real_verification_context`, S00 real scope) require a marked `test` namespace for `search_test/v1` outputs and a current witness.

## WS-B (capacity, §4.4)

Owns `propsim/event_detail.py`, `propsim/search_bridge.py`, `ml/regime_stratified_prop.py`, `scripts/hardening_capacity_benchmark.py`, `tests/propsim/test_account_event_detail.py`, `tests/agents/data_infra/ifvg/test_regime_stratification.py` (prop parts), new tests, `CAPACITY_BENCHMARKS.md` (+ JSON) in this folder.

- `build_account_event_detail(walks, path_records=None, *, total_rows=None, ...)` accepts an ITERABLE of `(path_record, walk_result)` pairs in draw-ordinal order (a generator is never materialized); no whole-artifact `np.concatenate` id index; uniqueness proven by the canonical key (unique path ids × strictly increasing ordinals; the event id must be the deterministic projection of that key, else a DuckDB disk-backed uniqueness check over the written partitions) plus bounded per-block checks; bounded row groups.
- Stratified summary: per-partition group rows → canonical intermediate Parquet partitions in an attempt-local temp dir → exact DuckDB aggregation under an explicit `memory_limit` + `temp_directory` → exact unique-path counts externally → canonical-sort → bytes; temp files cleaned on success/failure.
- `HARDENING_CAPACITY_POLICY_V1` benchmark harness (fresh subprocess per run; native RSS via Win32 `GetProcessMemoryInfo` PeakWorkingSetSize / POSIX `resource`; tracemalloc supplementary) at 250k/500k/1M rows; determinism repeat; slope; projection at the registered maximum; the numerical gate table of §4.4.

## WS-C (§4.5 warnings + §4.6 sequential truth)

Owns `pyproject.toml`, `ifvg/dataset.py` (the :813 concat only), `tests/agents/test_ifvg_context_experiment_engine.py` (:260 only), `search/pipeline.py` (`WorkerPolicy`, `ExecutionAttemptIdentity`, `run_pipeline` attempt receipt only), `scripts/ifvg_pipeline_job.py`, `tests/agents/ifvg_search/test_pipeline_contracts.py`, `test_pipeline_run.py` (the worker-policy test only), `test_pipeline_job_script.py`, new tests, `WARNING_BASELINE.json` + `WARNING_POLICY_REPORT.md` in this folder.

- `SUPPORTED_CHILD_WORKERS = 1`, `EXECUTION_MODE_V1 = "sequential_children_v1"`, typed `UnsupportedWorkerParallelismError(ValueError).reason = "unsupported_worker_parallelism_v1"`; `WorkerPolicy.max_workers` must equal 1; `ExecutionAttemptIdentity.effective_workers: Literal[1]`, `execution_mode: Literal["sequential_children_v1"]`; the job shim refuses `--max-workers != 1` BEFORE job creation (start/resume/worker); receipts persist both fields.
- `filterwarnings = ["error", <one exact third-party rule>]` in pyproject; project-owned warnings fixed with schema-aligned typed frames (never dropping all-null columns; frozen bytes/hashes unchanged).

## WS-D (Phase 3, §5)

Owns new `search/trading_calendar.py`, `search/verification_window.py`, `search/seed_production.py`, `scripts/ifvg_verification_window_shortlist.py`, `scripts/ifvg_seed_production.py`, their tests, `PHASE3_*.md` + `LOGICAL_WINDOW_COVERAGE_SCAN.json` + `VERIFICATION_WINDOW_SHORTLIST.md` in this folder.

- Logical trading day = the Strategy-Core trading-day id `td` whose stream is `[td−1 18:00 ET, td 18:00 ET)`; physical partitions `(td−1, td)` UTC dates; weekday, not a registered full-closure day; `VerificationTradingDayRef{logical_trading_day, session_open_ts_utc, session_close_ts_utc, ordered_source_partition_refs[{physical_utc_date, relative_logical_partition_key, source_kind, content_sha256}]}`.
- Shortlist rebuilt from already-authorized evidence only (accepted v2 dataset + FSM audit funnel); lexicographic ranking of §5.1; contains at least the June proposal, the highest lifecycle/candidate/decision/trade window, the highest audit-day window. NO hashing into `register_program_allowlist`, NO owner selection.
- `SeedProductionReplayPolicy` (fail-before-path; ordered logical chain; June 11 + sealed excluded), `SeedProductionAuthorizationPayload/Envelope/Ref` (binds `store_namespace_id`, `supersession_head_witness`, profile + section hash, chain days, `snapshot_through_day`, `first_intended_verification_day`, access-policy id, expected source-inventory hash, QL/SC identities, seed schema version, chain policy id, `final_day_exhausts_dataset=False`, permitted/prohibited outputs, owner refs/approver/effective time), `SeedProductionRunPayload/Envelope`; `verify_seed_production_authorization` BEFORE any source path; `run_seed_production_chain` (synthetic proof over the conftest chain; outputs = seed snapshot + access audit + run receipt only); unsigned packet builders whose placeholders fail validation.

## WS-E (Phase 4, §6)

After A and D land: `search/bounded_verification.py` (preflight, `R1BaselineGateReport`, `BoundedReleaseControlFlowReport` + builder), `scripts/ifvg_bounded_verification.py`, tests, `PHASE4_*.md`.
