# HARDENING-BACKEND — Phase 2 (plan §4): backend hardening — what the code does

Findings F-11, F-12, F-13, F-17, F-18, F-20 (plan §2) closed; presentation
layer untouched (the UI/UX redesign is a separate plan). Every section states
the code, the tests and the evidence file.

## §4.1 Semantic store namespace (F-11) — `search/store_namespace.py`

- `StoreNamespacePayload{namespace_schema_version=1, namespace_class ∈ {research, test},
  store_instance_id (a 32-hex UUID generated or explicitly supplied ONCE — never an
  absolute-path hash), authority_genesis_id = genesis_id_for(instance, class)}`;
  `StoreNamespaceEnvelope{store_namespace_id = hash(payload)}`; immutable file
  `<root>/STORE_NAMESPACE.json`, re-verified (the id must hash the payload) on EVERY
  authorization load — a rewritten file is `store_namespace_identity_mismatch`, never a class.
- Every owner-decision artifact (`OwnerDecisionArtifactPayload.store_namespace_id`),
  supersession record, authorization bundle (`OwnerAuthorizationBundle`), verification
  authorization (`VerificationAuthorizationRef`) and seed-production authorization
  (`SeedProductionAuthorizationPayload`) references the `store_namespace_id`; persist AND load
  refuse an artifact that names another namespace ("authority never follows a copy").
- An unmarked store has no semantic authority: owner decisions, supersession records,
  owner-evidence promotions, real charters, real authorizations and real launches refuse
  (`store_namespace_missing`). The one-time explicit migration is
  `scripts/ifvg_store_namespace.py init --store-root … --namespace-class research|test`
  (prints the intent; `--confirm` initializes; idempotent on identical; a different class or
  instance is `store_namespace_divergent`); the class is an operator argument, never inferred.
- The pathname heuristic (`path_looks_like_research_store`; the old `research_namespace_root`
  is an alias) is DEFENSE IN DEPTH: it refuses a `test` namespace under a research-looking path
  (`assert_namespace_deployment_coherent`) and synthetic authority under a research-looking path,
  but never defines authority. Relocation preserves the id (`copytree` → identical namespace id
  and witness).
- Consumers: `assert_run_scope_lawful_for_root` (the `synthetic_fixture` scope: a `research`
  namespace refuses; an unmarked store cannot carry owner artifacts so the scope unlocks nothing
  there; the path check stays), `save_charter` (synthetic-marker charters refused in a `research`
  namespace; real bundles must be bound), `executors.real_verification_context` and the pipeline's
  S00 real branches (a coherent `test` namespace + current witness before any path).
- Tests: `test_store_namespace.py` (9): explicit/idempotent/immutable init; verified load
  (tamper, malformed, class mismatch); identity is not a path hash + relocation; path heuristic
  is defense in depth; the synthetic scope follows the class; an unmarked store cannot carry
  owner authority; a missing head is corruption; interrupted-initialization recovery; the CLI.

## §4.2 Immutable supersession chain + head witnesses (F-12) — `search/supersession_chain.py`

- Store `owner_decision_supersessions/<record-id>/` (manifest-verified entries);
  `OwnerDecisionSupersessionPayload{store_namespace_id, superseded_decision_id,
  replacement_decision_id, prior_head_record_id, prior_head_sha256, prior_line_count, reason,
  effective_at, owner_evidence_ref}`. Nothing writes into an existing `owner_decisions/<id>/`
  directory (test-proven: the superseded artifacts' directories stay `envelope.json` +
  `manifest.json`, bytes unchanged).
- The mandatory head `owner_decisions/SUPERSESSIONS.head` =
  `{head_schema_version: 2, store_namespace_id, record_id, line_count, head_sha256}` with
  `head_sha256 = chain_head_digest(prior_head_sha256, record_id)` committing to the whole chain
  from the namespace's `authority_genesis_id` (the explicit GENESIS head written at
  initialization). A missing head is `supersession_head_missing` — corruption, never "no
  supersessions".
- Publication (`publish_supersession`) under the liveness-aware lock: (1) verify namespace,
  current head, and the caller's source/replacement/nondivergence check; (2/3) write + verify
  the immutable record in the store's temporary directory and publish it atomically; the lock
  token is re-verified; (4) atomically replace the head. A failure before (4) leaves an orphan
  record with NO authority (test: the chain and every witness ignore it; the retry reuses it and
  advances the head — exactly one record); a failure after (4) leaves the head naming an
  already verified immutable record. Identical replay = reuse; a divergent record for the same
  transition = `supersession_divergent_replay`.
- `persist_owner_decision` publishes the record BEFORE the replacement decision (a crash in
  between leaves the chain fail-closed — "not a verified store entry" — until the replacement is
  re-persisted, idempotently); `load_supersession_chain` keeps its name and returns the
  structure-verified `SupersessionRecord`s backed by verified replacement artifacts.
- Witness rule: `SupersessionHeadWitness{store_namespace_id, line_count, head_sha256}` on every
  real charter bundle, verification authorization and seed-production authorization;
  `assert_head_witness_current` refuses a missing, SHORTER (`supersession_head_shorter_than_witness`
  — a rolled-back chain) or different (`supersession_head_witness_mismatch`) current head;
  verified at `validate_owner_authorization(store_root=)`, `validate_verification_run(store_root)`,
  `save_charter`, the real executors, S00 and the Phase 4 preflight. Honest boundary: local
  rollback detection, not cryptographic owner authenticity — deleting or rewriting the entire
  store plus every external witness is outside the trust boundary.
- Tests: `test_supersession_chain.py` (7), the rewritten supersession tests of
  `test_owner_decisions.py` (store-owned chain; crash between record and replacement;
  deleted/edited/rolled-back/headless chain; weaker provenance), `test_authorization.py`
  (bundle bound to namespace + current head; moved head; rollback), `test_verification.py`
  (run bound to a `test` namespace + head).

## §4.3 Liveness-aware owner-decision lock (F-13) — `search/owner_decision_lock.py`

- Lock body `{lock_schema_version, pid, process_start_token, lock_token, host, created_at,
  heartbeat_at}`; `OwnerDecisionLock.refresh()` rewrites the heartbeat atomically (before the
  immutable write), `verify_held()` re-reads the token before the head moves (`lock_lost` aborts
  the writer before publication), `release()` unlinks only its own token.
- Reclaim ONLY when the heartbeat exceeds the timeout AND `process_liveness(pid, start_token,
  host) == "dead"`: no such pid, an exited pid, or a live pid whose process-start token differs
  from the recorded one (PID reuse; Win32 creation time via `ctypes` `GetProcessTimes`, Linux
  `/proc/<pid>/stat` starttime). A slow LIVE holder is never reclaimed
  (`lock_held_by_live_holder`); another host, a malformed body, or an unassessable platform is
  never reclaimed (`lock_holder_liveness_unknown` / `lock_body_malformed`) — the waiter times out
  with the typed reason for operator intervention. No new dependency (`psutil` not required).
- Tests: `test_owner_decision_lock.py` (8): liveness verdicts; slow live holder never reclaimed;
  dead pid reclaimed only after the heartbeat timeout; PID reuse detected by the start token;
  other host / malformed never reclaimed; heartbeat keeps the token, release unlinks only its own;
  a stolen lock aborts the writer before the head moves (orphan record, head untouched); crash
  recovery reclaims a dead writer's lock and publishes. Plus the rewritten lock test of
  `test_owner_decisions.py` through `persist_owner_decision`.

## §4.4 Capacity implementation and numerical gates (F-17) — WS-B

See `CAPACITY_BENCHMARKS.md` / `.json` (the measured and the extrapolated tables, every gate)
and `_ws_B_NOTES.md`.

- `propsim/event_detail.py`: `build_account_event_detail` consumes an ITERABLE of
  `(path_record, walk_result)` pairs exactly once (never materialized) with the caller's declared
  preflight counts (verified after streaming); the whole-artifact `np.concatenate` id index is
  gone; uniqueness = the emitter's canonical key argument (`(path_instance_id, event_ordinal)`
  under strictly increasing ordinals per path and distinct path records) + an unconditional
  disk-backed DuckDB distinct check over the WRITTEN partitions (memory-limited, attempt-local
  temp dir, cleaned in `finally`); bounded row groups; the manifest records the uniqueness check.
- `propsim/search_bridge.py`: the bridge feeds a generator in draw-ordinal order (no second
  all-path list). `AccountSimulationRun` itself still holds the walk results (the simulation's
  D15-era design; outside the writer — recorded honestly in `DEVIATIONS.md`).
- `ml/regime_stratified_prop.py`: per-partition group rows → typed intermediate Parquet
  partitions in an attempt-local temp dir → exact DuckDB aggregation (`memory_limit` 512 MiB,
  one thread, spill directory) → exact unique-path counts and the cross-partition
  path-repetition refusal externally → canonical ORDER BY (the SQL form of the sort key) →
  row-group-aligned Parquet writer whose bytes equal `summary_parquet_bytes(table)` → budgets
  checked before publication; temp files cleaned on success and refusal.
- `HARDENING_CAPACITY_POLICY_V1` harness (`scripts/hardening_capacity_benchmark.py`; fresh
  subprocess per run; native RSS via Win32 `GetProcessMemoryInfo` / POSIX `getrusage`;
  `tracemalloc` supplementary): measured 1M-row peak RSS increase B1 +0.32–0.34 GiB (≤ 1.5 GiB),
  B2 +0.48 GiB (≤ 1.5 GiB); 1M-row wall B1 ≈ 8 s, B2 ≈ 38 s (≤ 300 s); byte-identical hashes on
  repeat at every size; the slopes, projections at the registered maxima (10M detail rows / 5M
  summary rows) and artifact-size gates are in `CAPACITY_BENCHMARKS.md`.
- Tests: `tests/propsim/test_event_detail_streaming.py` (8),
  `tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` (6); the R6.1
  `test_account_event_detail.py` suite unchanged and green.

## §4.5 Warning policy (F-18) — WS-C

See `WARNING_POLICY_REPORT.md` / `WARNING_BASELINE.json` and `_ws_C_NOTES.md`.

- Project-owned warnings fixed at the source with schema-aligned typed frames: `dataset.py`
  `concat_schema_aligned` (the `:813` pandas concat `FutureWarning`; reproduces the
  pre-deprecation result exactly, never drops an all-null column — proven on the real synthetic
  FSM-chain audit frames and every warning shape; frozen bytes/hashes unchanged, goldens green)
  and the test-side typed one-row append (`test_ifvg_context_experiment_engine.py:260`).
- `pyproject.toml`: `filterwarnings = ["error", <one exact rule>]` — the ONLY ignore is the
  scikit-learn 1.7.0 / SciPy 1.16 L-BFGS-B `disp`/`iprint` `DeprecationWarning` (exact message
  regex, category, module `sklearn\.linear_model\._logistic`, reason, owner, expiry). No broad
  `DeprecationWarning` suppression. Project-owned warnings remaining: 0 (the full suites of
  `TEST_RESULTS.md` run under warnings-as-errors).
- Tests: `tests/agents/test_hardening_warning_policy.py` (11).

## §4.6 Sequential execution truth (F-20) — WS-C

- `search/pipeline.py`: `SUPPORTED_CHILD_WORKERS = 1`, `EXECUTION_MODE_V1 =
  "sequential_children_v1"`, typed `UnsupportedWorkerParallelismError` (`.reason =
  unsupported_worker_parallelism_v1`); `WorkerPolicy.max_workers` must equal 1 (never coerced;
  the R5 `le=4` ceiling — a claim of parallelism the executor never had — is gone);
  `ExecutionAttemptIdentity.effective_workers: Literal[1]` + `execution_mode:
  Literal["sequential_children_v1"]` persisted in every attempt receipt (operational metadata;
  never scientific identity).
- `scripts/ifvg_pipeline_job.py`: `start` / `resume` / `worker` refuse `--max-workers != 1`
  BEFORE job creation / any store access (JSON refusal, exit 2). The UI slider (1–4) is the UI
  plan's control (`DEVIATIONS.md`). The bounded current-code throughput benchmark of §4.6 is a
  Phase 5 precondition, not part of this release.
- Tests: `test_pipeline_contracts.py`, `test_pipeline_run.py`, `test_pipeline_job_script.py`
  (typed refusals for 2/4/8; receipts state the sequential mode; start/resume refuse before
  spawning; worker refuses before store access).

## Adversarial-round amendments (what changed after the two reviews)

- **Lock liveness (RA-02, MAJOR)**: `release` / `refresh` / the reclaim unlink / the body read
  retry transient sharing violations (40 × 25 ms) and type a persistent failure
  (`lock_release_failed`, `lock_refresh_failed`) — never a silently orphaned lock owned by a live
  pid; the head / namespace atomic replace retries the same way (`atomic_write_failed`, RA-10).
- **The MBP-1 diagnostic seam (RA-01, MAJOR)** binds the store's `test` namespace and the CURRENT
  head witness before the coverage matrix loads, before `register_program_allowlist`, before any
  path (the pathname rule is defense in depth).
- **Publication (RA-03)**: `activate_pipeline_result` re-verifies the charter bundle against the
  CURRENT head after the re-derived gates; a synthetic-marker charter never activates.
- **Every bound seam (RA-09)** runs the deployment-coherence check; **run selection (RA-08)** is by
  pipeline id with ambiguity refused; **the synthetic fixture (RA-06)** marks temp roots only;
  **seed production (RA-07)** has a store post-condition (`prohibited_output_written`).
- **Sequential truth (B-06)**: strict int, typed before `ge=1`. **S08** persists a typed
  `fold_summary.json` (B-03). **Capacity (B-02)**: the B1 gates are stated for the benchmark's
  200 events/path shape; the writer is block-bounded for every shape; a row-based flush guard is
  a future versioned policy change (`CAPACITY_BENCHMARKS.md` "Shape parameters"; DEV-HB-32).
- Dispositions: `ADVERSARIAL_REVIEW_RESOLUTION.md` (20/20; 17 FIXED incl. 2 variants, 2 ACCEPTED-documented, 0 open).

## §4.7 acceptance evidence

`HARDENING-BACKEND.patch` + `.sha256`, the Git bundle + `.sha256`, raw pytest outputs
(`_final_pytest*.txt`), Ruff + `git diff --check` (`_ruff_and_diffcheck.txt`), namespace
migration/relocation evidence (`test_store_namespace.py`, the CLI smoke in `TEST_RESULTS.md`),
supersession rollback/crash evidence (`test_supersession_chain.py`, `test_owner_decisions.py`),
lock liveness evidence (`test_owner_decision_lock.py`), `CAPACITY_BENCHMARKS.md`,
`WARNING_BASELINE.json`, clean-tree proof and zero new real-data artifacts
(`ACCESS_SAFETY_EVIDENCE.md`), `full_pipeline_not_run=true` (`GATE_SUMMARY.md`).
