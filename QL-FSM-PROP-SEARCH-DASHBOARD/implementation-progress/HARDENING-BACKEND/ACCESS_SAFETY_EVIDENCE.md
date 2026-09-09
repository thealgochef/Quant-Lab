# HARDENING-BACKEND — Access-Safety Evidence

Implementer's audit, written 2026-09-02 before the adversarial round;
re-checked after the round (the final `find` / `git status` facts are
repeated in `GATE_SUMMARY.md`). Everything below states what the code does
and what the filesystem shows.

## Protected/sealed counters: ZERO

- **No real source path was ever constructed** by this release's
  implementation, tests, benchmarks, reviewers' probes or evidence runs.
  The only real store touched read-only is the ALREADY-AUTHORIZED accepted
  v2 dataset + FSM audit day funnel (manifest-verified) that the Phase 3
  shortlist rebuild reads exactly as R1's `window_coverage_scan.py` did
  (`LOGICAL_WINDOW_COVERAGE_SCAN.json` records `no_raw_source_reads: true`).
  The Phase 4 runner was executed against the repository's REAL verification
  store root and refused `fail_before_path` (no persisted verification run;
  `PHASE4_BOUNDED_VERIFICATION.md`), constructing no path.
- **`grep` over every NEW source module and script** (`store_namespace.py`,
  `supersession_chain.py`, `owner_decision_lock.py`, `bounded_verification.py`,
  `trading_calendar.py`, `verification_window.py`, `seed_production.py`,
  the five new scripts): no `data/databento`, no `DEFAULT_DATA_DIR`; the single
  `IfvgCaptureConfig(` construction is the seed-production CLI's `run`
  building a config OBJECT for `run_seed_production_chain`, which verifies the
  persisted signed authorization (namespace, witness, profile, chain,
  inventory hash, code identities, effectivity) BEFORE any day artifact or
  source path is touched (`test_seed_production.py`: refusals before any path
  with the capture monkeypatched to fail if reached).
- **Date literals in the new modules**: the window guards
  (`PERMITTED_WINDOW_LAST_DAY = 2026-06-10`, `PROTECTED_BUFFER_DAY = 2026-06-11`,
  `SEALED_START_DAY = 2026-06-12`, `_PROTECTED_BUFFER_DAY`), the plan's existing
  June proposal constant (a documented candidate, now found ineligible), the
  evidence-verified holiday session notes, and September-2026 ISO instants
  used as fixture `approved_at` values — instants from which no path is
  derived. June 11 and the sealed range are refused by every new policy
  (`SeedProductionReplayPolicy`, the preflight's `protected_or_sealed_date`).
- **No production `assert`** in any touched or new `src` module (`grep`
  verified; `pragma: no cover` guards excluded).
- **On disk**: `find data -type f -newermt "2026-09-02 00:00"` → **0 files**
  at the baseline, after every workstream, and at the gate. No `search`,
  `search_test/STORE_NAMESPACE.json`, `owner_decision_supersessions`,
  `seed_production_*`, `r1_baseline_gate_reports` or
  `bounded_release_control_flow_reports` directory exists under `data/`; every
  store a test or benchmark wrote is `tmp_path` / `tempfile.mkdtemp()` rooted
  (the benchmark's synthetic stores and the DuckDB spill directories are
  attempt-local temp directories, removed in `finally`).
- **Strategy-Core clean at the pinned commit
  `a4e3303179ac6a1088aecaaa3482934cf1aec4d7`** (`git status --short` empty);
  **Trade-Lab untouched** — its worktree carries the user's 27 pre-existing
  dirty files (none newer than 2026-09-01). The plan package, `../R5B.1/`,
  `../R6/`, `../R6.1/` and `../R6.1-FIX/` are unmodified.

## No owner action simulated, no real run

- No `register_program_allowlist` call outside the existing R1 slice code
  path (the Phase 4 preflight only READS an existing marker and refuses a
  different one; the shortlist script never calls it — `register_program_allowlist_called: false`).
- No `SeedProductionAuthorizationRef` / `VerificationAuthorizationRef` of
  `owner_signed` provenance exists anywhere; the packets are UNSIGNED and
  their placeholders fail validation; the synthetic-provenance seed-production
  authorization is confined to `test` namespaces (refused elsewhere at persist
  and at load).
- The real verification store root was NOT initialized as a namespace (the
  owner's explicit `scripts/ifvg_store_namespace.py init … --confirm`); every
  namespace initialization in this release happened under `tmp_path`.
- `full_pipeline_not_run=true`: no full-development pipeline, no real seed
  replay, no real ≤5-day run.

## Authority never follows a path or a copy

- Research-versus-test authority is the store's verified namespace envelope;
  a copied owner artifact refuses at load ("another store namespace"); a
  relocated store keeps its identity and witness; a `test` namespace under a
  research-looking path is refused as incoherent; the synthetic scope is
  refused in a `research` namespace and, as defense in depth, under a
  research-looking path (`test_store_namespace.py`, `test_owner_decisions.py`).
- Every real authorization (bundle, verification, seed production) binds the
  namespace id and the CURRENT supersession head; a missing, shorter or
  different head refuses at publication and launch (`test_authorization.py`,
  `test_verification.py`, `test_seed_production.py`, the preflight tests).

## No automated selection or promotion

- Nothing in this release adds a selection, ranking-into-authorization,
  promotion, activation or launch path: the shortlist is evidence with a
  content id (`owner_selection = "NOT PERFORMED"`); S11 stays blocked with its
  registered reason (typed by the bounded report); the bounded report and the
  R1 gate report are `verification_only` / `not_for_research_interpretation`
  / `full_pipeline_not_run` (literal `True`) and `not_research_evidence`;
  research profitability, strategy, payout, feature-selection and promotion
  gates are never applied to the fixture.
- The sequential-execution truth removes a claimed capability (parallel
  children) rather than adding one; the job shim refuses before job creation.

## Determinism and store discipline

- Five new stores (`owner_decision_supersessions`, `seed_production_authorizations`,
  `seed_production_runs`, `r1_baseline_gate_reports`,
  `bounded_release_control_flow_reports`) follow the manifest protocol
  (identity → refuse-if-exists → tmp-dir write → manifest → atomic publish →
  reload → assert), exact-id loads, never listed, relocation-safe, tamper
  fails closed. The namespace file and the head pointer are the two
  deliberately mutable-by-replacement files: the namespace file is immutable
  in content (re-initialization must reproduce it) and self-verifying; the
  head is replaced atomically and commits to the whole chain.
- The capacity benchmark reproduced byte-identical output hashes on repeat at
  every size; the external summary aggregation is byte-identical to the
  in-memory reference and deterministic (single-thread DuckDB, canonical
  ORDER BY, row-group-aligned writer).

## Frozen lanes

`ifvg/context_model.py` and the M0–M3 lane are byte-unchanged; the golden
identities of `PRE_HARDENING_BASELINE.md` reproduce
(`test_r61_fix_goldens.py`); `dataset.concat_schema_aligned` reproduces the
pre-deprecation audit-table bytes (goldens and the FSM audit contract tests
green); Strategy-Core / Trade-Lab untouched; no new package dependency
(`psutil` not required; DuckDB was already a dependency).
