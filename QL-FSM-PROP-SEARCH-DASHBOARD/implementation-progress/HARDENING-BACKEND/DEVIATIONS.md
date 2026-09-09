# HARDENING-BACKEND — Deviations and Scoping Notes

Written 2026-09-02 against `../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3, Phases 2 (§4), 3 (§5) and 4 (§6). Every entry states what the
CODE does. Workstream-local notes with more detail: `_ws_B_NOTES.md`,
`_ws_C_NOTES.md`, `_ws_D_NOTES.md`. Entries added after the adversarial round
are marked.

## Process

### DEV-HB-1 — Three phases in ONE release commit; owner and real-run steps NOT performed

Plan §10 sequences Phase 2 (a `HARDENING-BACKEND` commit), then the owner's
window selection, seed-production signature, seed replay, verification
signature and the real ≤5-day run. This release lands Phase 2 in full and the
CODE/CONTRACT halves of Phases 3 and 4 (calendar, shortlist rebuild, seed
production, packets, preflight, gate reports, runner) in the same commit so
the owner can act on a complete backend. Nothing owner-only was done: no
window was selected or registered (`register_program_allowlist` was never
called; no `VERIFICATION_ALLOWLIST_MARKER.json` exists), no
`SeedProductionAuthorizationRef` / `VerificationAuthorizationRef` exists, no
real seed replay and no real verification ran, the real verification store
(`data/ifvg_datasets/search_test/v1`) was NOT initialized as a namespace
(the owner's explicit `init`), and `data/` gained zero files.

### DEV-HB-2 — Shared core landed before its tests; red-first evidence is by workstream

To let four workstreams run in parallel, the main agent wrote and
smoke-tested `store_namespace.py`, `supersession_chain.py` and
`owner_decision_lock.py` first, then the consumer seams, then the test
modules. The red-first artifact for WS-A is the collection failure of the
R6.1 JSONL-log supersession tests once the mutable log was gone
(`_red_ws_A.txt`); every new WS-A test module carries negative / tamper /
crash cases (a lying head, an orphan record, a stolen lock, a rolled-back
chain). WS-B / WS-C / WS-D wrote their tests red first (`_red_ws_B.txt`,
`_red_ws_C.txt`, `_red_ws_D.txt`).

## Phase 2 — §4.1 namespace (F-11)

### DEV-HB-3 — What "an unmarked store cannot authorize a write" covers

Owner-decision artifacts (persist AND load), supersession records,
owner-evidence promotions (`FEATURE_ELIGIBLE`+ through
`_assert_owner_evidence_authorizes`), real charters (`save_charter` with a
bundle), real verification / seed-production authorizations and every real
launch (executors, S00 real scopes, the Phase 4 preflight) require the marked
namespace. STRUCTURAL regime promotions that consume no owner evidence, and
synthetic-marker charters, remain lawful in an unmarked tmp/test root —
they authorize nothing — because an unmarked store can hold no owner artifact
(persist and load refuse), so the `synthetic_fixture` scope unlocks nothing
there. The `research` class refuses the synthetic scope semantically; the
pathname heuristic (`path_looks_like_research_store`) stays as defense in
depth only. `synthetic_owner_decision_fixture` marks an UNMARKED test root
explicitly as a `test` namespace (the class is stated by the fixture, never
inferred from the path) so the R6.1 test lanes need no per-test change.

### DEV-HB-4 — `store_root` is optional on `validate_owner_authorization` / `validate_charter`

The draft-time validation (`validate_charter` from the wizard / pipeline
tab, before a store exists) runs the store-independent binding only; the
store-bound half (namespace id + current head witness) runs at every
publication and launch seam that has a store: `save_charter`,
`validate_verification_run(store_root)` (REQUIRED), the real executors, S00
(both real scopes) and the Phase 4 preflight. A real charter therefore
cannot be published or launched unbound.

### DEV-HB-5 — Owner-decision artifact ids re-mint (synthetic only)

`OwnerDecisionArtifactPayload.store_namespace_id` is required, so every
owner-decision artifact id moves; likewise every `VerificationRunPayload`
(the ref carries the namespace + witness) and every real charter bundle. No
persisted real artifact exists outside tmp roots. The golden identities of
`PRE_HARDENING_BASELINE.md` are unchanged (`test_r61_fix_goldens.py`).

## Phase 2 — §4.2 supersession chain (F-12)

### DEV-HB-6 — The record is published BEFORE the replacement decision

Plan §4.2 step 1 verifies "the replacement decision". The replacement is
verified as a valid envelope (same protocol as the prior, provenance rank,
`supersedes` naming the prior) but persisted AFTER the record + head so a
crash in between keeps the R6.1 fail-closed semantics (the prior cannot
authorize while its replacement is missing; re-persisting repairs
idempotently with exactly one record). The head digest commits to the whole
chain (`chain_head_digest(prior, record_id)`), so a rolled-back head is
"shorter than the witness" and a replaced head "breaks the chain".

### DEV-HB-7 — "Different current head" refuses even when the chain only grew

`assert_head_witness_current` refuses ANY difference (missing, shorter,
different) — an authorization signed against head N is stale once a
supersession N+1 lands and must be re-signed. This is the conservative
reading of §4.2 ("refuses a missing, shorter, or different current head").

### DEV-HB-8 — Legacy R6.1 chains are not migrated

A pre-hardening store carries `SUPERSESSIONS.jsonl` + a v1 head; the v2
reader refuses it as `supersession_head_malformed` ("a pre-hardening JSONL
log head is not a v2 head; migrate the store"). No real store has a chain
(every R6.1 chain lived in tmp roots), so no migration tool was written.

## Phase 2 — §4.3 lock (F-13)

### DEV-HB-9 — Unassessable holders are never reclaimed

A lock whose body is malformed, whose host differs, or whose platform gives
no liveness verdict times out with `lock_body_malformed` /
`lock_holder_liveness_unknown` and requires operator intervention — age
alone never reclaims (the F-13 defect). The process-start token is the Win32
creation time / Linux `/proc` starttime; where unavailable (`None`) PID
reuse cannot be distinguished from the live process and the holder counts
as alive.

## Phase 2 — §4.4 capacity (F-17) — from `_ws_B_NOTES.md`

### DEV-HB-10 — The disk-backed uniqueness check is unconditional

The writer cannot re-derive a foreign envelope's id projection per row
(the emitter's `account_namespace` is not on the envelope), so the
artifact-level DuckDB distinct check over the written partitions runs on
every publication (memory-limited, attempt-local temp dir), with the
canonical-key argument recorded in the manifest. No whole-artifact
in-memory set exists.

### DEV-HB-11 — `AccountSimulationRun` still holds every walk result

The writer and the bridge no longer build a second all-path list or an
index; the simulation's own result object (`run_account_simulation`,
`build_payout_reliability_vector`) still materializes the walks — the
D15-era design, outside the writer path §4.4 gates. Recorded honestly; a
streaming simulation seam is a future capability.

### DEV-HB-12 — Benchmark environmental precondition

Attempt 1 of the benchmark ran while other workstreams' suites were
executing and measured 7.896 GiB free RAM at the B1 start (the §4.4 minimum
is 8 GiB); every numerical gate passed with the same output hashes. Attempt
2 ran with ≥ 8.75 GiB free at every start and is the recorded result
(`CAPACITY_BENCHMARKS.md`; attempt 1 kept as
`_capacity_benchmark_run_attempt1.txt`). No ceiling was lowered. The B1
synthetic walks are lightweight duck-typed event objects (the writer reads
attributes only), so B1 measures the writer, the store publish and the
production reader — not Pydantic envelope construction.

### DEV-HB-12a — Manifest / detail additions and a last-ulp accumulation note (WS-B)

The event-detail manifest gains `row_group_size` and the
`event_id_uniqueness` record (canonical key, artifact check, memory limit,
partitions / rows checked, distinct ids / paths); the `stratified_prop`
report's JSON `detail["summary"]` gains an `aggregation` block (engine,
limits, intermediate partition count). Stratified-report ids re-mint
(synthetic only). With one thread and insertion order preserved the
external aggregation is deterministic run-to-run, but for a summary key
whose events span ≥ 3 partitions the floating-point accumulation order of
`amount_sum` may differ from the R6.1 in-memory merge at the last ulp
(proven byte-identical to the pandas reference on every fixture; no
persisted real artifact exists).

## Phase 2 — §4.5 warnings / §4.6 sequential truth — from `_ws_C_NOTES.md`

### DEV-HB-13 — The typed worker error rides inside pydantic's `ValidationError`

`WorkerPolicy(max_workers=2)` raises `ValidationError` carrying the
`UnsupportedWorkerParallelismError` (`ctx.error`; `worker_parallelism_refusal`
extracts it); the job shim calls `assert_supported_worker_parallelism`
directly, which raises the typed error before any object exists.

### DEV-HB-14 — The UI worker slider is untouched

`scripts/ifvg_pipeline_tab.py` still offers 1–4 workers; the backend now
refuses > 1 before job creation with the typed reason. The control belongs
to the separate UI/UX plan (this plan makes no Streamlit change).

### DEV-HB-15 — The message regex encodes `:` as `\x3a`

pytest splits an ini `filterwarnings` rule on `:`; the exact sklearn/SciPy
message contains a colon, encoded as `\x3a` in the regex.

## Phase 3 — from `_ws_D_NOTES.md`

### DEV-HB-16 — Seed chains are STORE-day chains; the allowlist is logical

The accepted chain replayed every non-Saturday calendar day (Sundays as
zero-bar reducer steps); a logical-only chain is not proven
emission-identical, so the seed-production authorization binds BOTH the
store-day replay chain and its logical subset, and `snapshot_through_day` is
the physical day before the first verification day. The verification
allowlist itself is a tuple of consecutive LOGICAL trading days (F-22).

### DEV-HB-17 — Good Friday 2026-04-03 is a trading day; the June proposal is INELIGIBLE

The registered full closure inside the window is `2026-01-01` only (the
accepted evidence shows a full Good Friday session). Under the plan's hard
constraint "at least one exact verifier setup target" (a resolved executed
trade whose setup id resolves in the lifecycle table) the June proposal
`2026-06-04…06-10` has zero targets and ranks 70 of 110 — an owner finding
recorded in `VERIFICATION_WINDOW_SHORTLIST.md`, not a selection. The R1
February store-day candidate is `provisional_ineligible_as_stated`
(`2026-02-08` is a Sunday partition); its corrected logical form
`2026-02-06…02-12` ranks 3; rank 1 is `2026-02-04…02-10`.

### DEV-HB-18 — Seed timestamp canonicalization; a latent R1 sandbox defect

A seed produced from Parquet-loaded day artifacts carries `pytz` tzinfo
that the R1 `_SeedSandboxUnpickler` refuses; the seed-production runner
canonicalizes the seed graph to stdlib UTC (same instants; hash-preserving,
self-checked) before persisting. `child_replay.save_seed_snapshot` itself was
NOT changed (R1 code); a direct disk-chain seed saved through it would still
not reload — recommended R1 correction: canonicalize inside
`save_seed_snapshot` (or allow `pytz.UTC` in the sandbox safe-list).

### DEV-HB-19 — Measurable proxies in the coverage rows

Panel/control-flow coverage is the count of window days in the FSM-audit
day funnel; MBP-1 scope evidence and panel coverage are `not_evaluated`
(no such evidence artifact exists for these days). Stated on every row.

## Phase 4

### DEV-HB-20 — "Audit-disabled and audit-enabled modes" are the dual drive of one replay

The baseline slice runs both modes in one dual-drive replay compared by the
neutrality report; `R1BaselineGatePayload` therefore carries the six gates of
the first (fresh) and second (verified reuse) attempts plus the two
audit-mode core-table digests (`audit_modes_equal`) rather than two
single-mode pipeline runs. The last three "also prove" proofs (different
bytes / corrupt manifest / corrupt sidecar fail closed) are the immutable
store's behaviors proven by the store suites on the exact code and recorded
as evidence refs by the runner.

### DEV-HB-21 — The bounded report reads typed states from persisted evidence only

`build_bounded_release_control_flow_report` types each component from the
completed state file and the manifest-verified stage sidecars
(`bundle_feature_views.json`, `fold_sample_adequacy.json`, `regime_run.json`,
`supervised_ladder.json`, the S15 `search_results` envelope) — never from a
research metric; the S05 typed-insufficiency branch reads the stage's
sanitized explanation when S05 fails with an `insufficient*` reason (the
panel materializer's registered vocabulary). The regime non-fit branch is
proven from a persisted `regime_run.json` record (zero fits, typed gate
failures); the full-pipeline synthetic fixture exercises the no-valid-fold,
zero-prediction, verification-only and S11-blocked branches.

### DEV-HB-22 — The `R1-VERIFICATION` evidence folder is produced by the real run, not this release

`scripts/ifvg_bounded_verification.py run` writes `VERIFICATION_RUN.md`,
`R1_BASELINE_GATES.json`, `BOUNDED_RELEASE_CONTROL_FLOW.json`,
`ACCESS_AUDIT.jsonl`, `REUSE_RECEIPT.json`,
`MANIFEST_AND_SIDECAR_HASHES.json`; the pytest/JUnit, verifier deep link and
`full_pipeline_not_run=true` proof of §6.4 join it when the owner's run
happens. No such folder exists yet.

## Amended after the adversarial round (2026-09-02)

The dispositions of every finding are in `ADVERSARIAL_REVIEW_RESOLUTION.md`;
the entries below record what the fix round changed.

### DEV-HB-23 — Lock filesystem steps retry transient sharing violations; persistent failures are typed (RA-02 / RA-10)

Windows refuses `unlink` / `replace` while ANY other handle has the file
open (a polling waiter's `read_text` is enough). `OwnerDecisionLock.release`,
`refresh`, the reclaim unlink and the lock-body read retry a transient
`OSError` with a short backoff (40 × 25 ms); a PERSISTENT failure is a typed
`OwnerDecisionLockError` (`lock_release_failed` — the file still carries the
writer's token, operator intervention required; `lock_refresh_failed` — the
writer aborts before publication), never silence. The head / namespace
`_atomic_write_text` retries `os.replace` the same way and types a persistent
failure `atomic_write_failed`. Mutual exclusion was never violated (the
reviewer's probes observed overlap 0); this closes the liveness defect.

### DEV-HB-24 — The MBP-1 diagnostic seam binds the namespace and the current head (RA-01)

`assert_diagnostic_authorized` calls `assert_authorization_bound_to_store(…,
expected_namespace_class="test")` right after the run-envelope binding checks
— before the coverage matrix loads, before `register_program_allowlist`,
before any source path. The `search_test/v1` pathname rule stays as defense
in depth. The R6.1-FIX test that pointed the diagnostic at an unmarked root
now proves the namespace refusal first and marks that root as the same
namespace instance to reach the matrix refusal.

### DEV-HB-25 — Activation re-verifies the charter bundle against the CURRENT head (RA-03)

`activate_pipeline_result`: after the scope refusal and the re-derived gates,
the charter is verified-loaded from the store; a synthetic-marker charter can
never activate; a real bundle must be bound to this store's namespace and the
CURRENT supersession head (`AuthorizationError` → `PublicationError`). A
supersession that lands between launch and activation stops the activation.

### DEV-HB-26 — The synthetic fixture marks TEMP roots only (RA-06)

`synthetic_owner_decision_fixture` initializes a `test` namespace only for an
unmarked root under the process temp directory (pytest's tmp roots); any
other unmarked root (e.g. the repository's real verification store) is a
typed `store_namespace_missing` refusal — the operator's explicit `init` is
the only way to mark a real store.

### DEV-HB-27 — Verification-run selection is by pipeline id; ambiguity refuses (RA-08)

`executors._verification_run_envelope(store_root, pipeline_semantic_id=…)`
filters the catalogued runs by the pipeline semantic id being launched (the
pipeline entry passes it; the search entry has none) and refuses when more
than one candidate remains — never positional.

### DEV-HB-28 — The deployment-coherence check runs inside the bound check (RA-09)

`assert_authorization_bound_to_store` calls
`assert_namespace_deployment_coherent`, so `validate_verification_run`, the
S00 full-development branch, `save_charter`, the MBP-1 seam and the preflight
all refuse a `test` namespace under a research-looking path.

### DEV-HB-29 — Seed production has a store post-condition (RA-07)

`run_seed_production_chain` snapshots the entry COUNT of every search store
directory before the replay and refuses `prohibited_output_written` if any
store other than `seed_snapshots` / `seed_production_runs` changed (counts
only; no entry id is ever listed). The Quant-Lab / Strategy-Core source
identities are git reads (not source paths) computed before the
authorization loads.

### DEV-HB-30 — Worker counts are strict ints and typed before `ge=1` (B-06)

`WorkerPolicy.max_workers` runs a before-validator: a bool, a str or any
non-int, and every value other than 1 (including 0) raise the typed
`UnsupportedWorkerParallelismError` (`worker_parallelism_refusal` extracts it
from the `ValidationError`).

### DEV-HB-31 — S08 persists a typed `fold_summary.json` (B-03)

The labeled S08 branch persists `{fold_count, valid_fold_count,
invalid_reasons, trading_day_count}` as a stage sidecar; the bounded report
reads it instead of parsing the sanitized explanation. The label-free regime
branch keeps its `fold_sample_adequacy.json`.

### DEV-HB-32 — The B1 numerical gates are stated for the benchmark's shape (B-02)

The event-detail writer is block-bounded (memory ∝ rows per path block =
`path_block_size × events_per_path`, never the artifact); the benchmark's
shape is 200 events/path (50,000 rows per block) and the projection at 10M
rows holds for that shape. `EVENT_DETAIL_BUDGET_V1` caps total rows and the
block size but not events per path, so a production walk with more events
per path scales the per-block resident memory linearly (documented in
`CAPACITY_BENCHMARKS.md`). A row-based flush guard is a future versioned
capacity-policy change (owner approval; splitting a path block across
partitions is outside the current manifest contract) — no ceiling was
changed in this release.

### DEV-HB-33 — Wasted work on a refused oversized simulation (B-09)

A single oversized simulation is aggregated in DuckDB (memory-limited,
spilling) before the exact row-budget refusal — memory-safe, refusal before
publication, wasted work only. Unchanged.

## Deferred cleanups (plan §2, "not a gate") remain open

`threadpoolctl>=3.1` declaration, `candidate_not_in_assignment` in the
reason vocabulary, the obsolete in-memory assignment frame,
`authorized_session_span_ns` binding, `children_without_strata` vs
`gates_passed` — unchanged from R6.1-FIX (DEV-R6.1-FIX-11).
