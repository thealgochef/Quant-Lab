# R6.1-FIX → Hardening → R1 Verification → Full Authorized Development Run — Implementation Plan

**Feature:** `ifvg_prop_robust_config_search_v1` · **Branch:** `feature/ifvg-prop-robust-config-search-v1` · **HEAD at planning:** R6.1 `6c0b60a1dddee89a058bfe015c55be5fbe2ceced` (parent R5B.1 `f3f9ac26fb118c1470aab4b113c107b1d116bddb`, parent R6 `179a2c9d…`) — verified from Git.  
**Prepared:** 2026-09-01 · **Status:** PLAN — revision 3, backend-only correction after independent review; pending owner approval. **Not an authorization.** No code, tests, artifacts, catalogs, seeds, replays, MBP-1 builds, model fits, prop simulations, owner decisions, or pipeline runs were performed while producing or revising it.  
**Supersession:** this revision replaces and invalidates the previously generated revision 2, which incorrectly inserted UI-redesign and browser/accessibility phases into the backend plan. Do not implement from revision 2.  
**Authority:** subordinate to `..\..\FINAL-IMPLEMENTATION-PLAN-DOCS\` and `..\R6.1-CORRECTION-PLAN-DOCS-FINAL\` (both unchanged). Evidence folders `..\R5B.1\`, `..\R6\`, and `..\R6.1\` are never edited. **Scope separation:** this is the backend R6.1 correction, hardening, authorization, bounded-verification, and operator-run plan only. The IFVG Lab UI/UX redesign is governed by its own separate plan and is neither a phase nor an acceptance gate here. This plan makes no Streamlit, chart-presentation, viewport, keyboard, accessibility, or UI-workflow changes.

---

## 1. Executive status

**Complete — authored and synthetically verified, not accepted:** R1–R6, R5B.1 (`f3f9ac2`; `R5B.1.patch` SHA-256 `517b548c…0a32` matches its companion), and R6.1 (`6c0b60a`; `R6.1.patch` SHA-256 `6a166afe…bfe7a` matches). The R6.1 suite passed twice with 1,931 tests and zero failures; Ruff and `git diff --check` were clean. Protected and sealed counters are zero in every release. The fixed M0–M3 lane, Strategy-Core, and Trade-Lab remain unchanged. Acceptance remains `transitively_blocked_by_R1`.

**Phase 1 — compact R6.1-FIX:** bind descriptive and model-facing regime outputs to the exact verified assignment sidecars that determine their bytes; enforce assignment schemas and invalid-row invariants; preserve candidates with a typed missing as-of reason; persist exact executed-trade tables for reused children; fail closed on corrupt prior-stage sidecars; require exact label-artifact identity for persisted studies; account for thin regimes without treating them as statistically reportable; and close the remaining production-wiring and MBP-1 contract defects.

**Phase 2 — backend hardening:** replace path-derived owner authority with an immutable semantic namespace contract; replace mutable-in-place supersession markers with an immutable supersession-record chain and mandatory head witnesses; make owner-decision locks liveness-aware; implement bounded event-detail streaming and external regime-summary aggregation; establish numerical RSS/runtime/capacity gates; make the backend execution contract truthful about sequential V1 operation; and enforce a precise warning policy.

**Phase 3 and Phase 4 — R1 owner-verification unblock and bounded verification:** no compliant profile-matching seed currently exists. The verification window is not selected yet. The current February “store-day” candidate is not eligible for registration until its logical trading-day mapping is established. Seed creation requires its own owner-signed `SeedProductionAuthorizationRef`; the concrete seed must then be reviewed and included in a separate final `VerificationAuthorizationRef`. The real verification remains one exact baseline over one canonical logical allowlist of at most five trading days.

**Phase 5 — Full Authorized Development:** remains a separate, explicit operator action after Phase 1, Phase 2, seed production, bounded real verification, R1 acceptance, and freezing every result-bearing semantic input. The run is invoked only through an explicit owner action against the approved backend operator contract. S11 remains blocked, MBP-1 remains offline/research-only, and live serving/execution remains unavailable.

---

## 2. Recon findings

Severity: **RB** release blocker · **HB** hardening blocker · **OD** owner-decision/authorization blocker · **EG** evidence gap. Status: **C** confirmed · **P** partial · **AF** already fixed · **FP** false positive.

| ID | Status | Sev | File / symbol | Finding | Owning phase |
|---|---|---|---|---|---|
| F-01 | C | RB | `ml/regime_oos_assignment.py` `RegimeOosAssignmentPayload`, `consulted_assignments_hash` | Descriptive assignment identity does not bind every source assignment value or the exact per-fit assignment sidecar/schema. | 1 |
| F-02 | C | RB | `ml/regime_executor.py`; `ml/regime_store.py` | The descriptive artifact is built from `run.assignments`; reuse compares row count and IDs, not the stored assignment values. | 1 |
| F-03 | C | RB | `ml/regime_fold_features.py`; `ml/regime_supervised_stage.py` | Model-facing fold features derive from an in-memory frame; `FoldFitRef` does not bind assignment bytes/schema. | 1 |
| F-04 | C | RB | `regime_oos_assignment.py`; `features/context_bar_panel_contract.py` | A null candidate stage anchor aborts the population although the contract describes typed downstream missingness. | 1 |
| F-05 | C | RB | `regime_oos_assignment.py`; `ml/regime_store.py`; `ml/regime_contracts.py` | A `valid=true` assignment may lack required values; the persisted fit-assignment sidecar has no enforced complete Arrow schema. | 1 |
| F-06 | C | RB | `search/pipeline.py`; `ml/regime_report_stage.py`; `search/identities.py` | A reused child without this cost policy’s persisted evaluation can disappear from S14; no independently reloadable executed-trade table artifact exists. | 1 |
| F-07 | C | RB | prior-stage loaders in `pipeline.py` and `regime_report_stage.py` | Broad exception recovery converts corruption, manifest mismatch, and I/O errors into lawful optional absence. | 1 |
| F-08 | C | RB | `ml/controlled_feature_study.py`; `ml/comparison_rows.py` | A persisted controlled study may bind a fallback label hash narrower than the economic/label columns it consumes. | 1 |
| F-09 | C | RB | `ml/regime_stratified_strategy.py` | Regimes below the reportability floor are omitted from raw net-R concentration accounting. | 1 |
| F-10A | C | RB | `regime_fold_features.py`; `pipeline.py` | Production assertions represent runtime wiring checks and can disappear under optimized Python. | 1 |
| F-10B | C | RB | `regime_stratified_strategy.py` | The normalized executed-trade frame is validated but the raw frame is used for joins, strata, and the binding hash. | 1 |
| F-10C | C | RB | `features/mbp1_coverage_evidence.py` | Positive MBP-1 evidence can be accepted without equality to the full manifest evidence scope/content refs. | 1 |
| F-10D | C | RB | `features/mbp1_feature_materializer.py` → `mbp1_source_contract.py` | Enum `.value` strings are passed into enum-typed identity-bearing fields and emit project-owned Pydantic warnings. | 1 |
| F-11 | C | HB | `search/owner_decisions.py` namespace helpers | Research/test authority is inferred from filesystem pathname components. | 2A |
| F-12 | C | HB | `search/owner_decisions.py` supersession chain | Deleting the log and head can restore a superseded decision; the prior plan’s in-directory marker would mutate an immutable decision artifact. | 2A |
| F-13 | C | HB | `search/owner_decisions.py` lock helpers | Lock expiry is age-only; a slow live holder can be reclaimed and overlapped. | 2A |
| F-14 | AF, except F-10C | — | R5B.1 MBP-1 source-quality path | Sequence jumps are diagnostic only; publisher/channel scope, recovery boundaries, and completeness compilation are implemented. The full scope-equality defect remains F-10C. | — |
| F-15 | FP | — | immutable store, alignment, assignment-source joins | Intra-store divergent sidecars are refused; fold alignment and typed invalid rows already exist. | — |
| F-16 | C | OD | `child_replay.py` seed contracts; current data stores | No compliant profile-matching seed snapshot exists; existing v2/replay-chart artifacts are not restorable Strategy-Core state. | 3 |
| F-17 | C/EG | HB | `propsim/event_detail.py`; `regime_stratified_prop.py` | Registered row/byte ceilings exist, but input materialization, global ID indexing, and cross-partition aggregation have no measured RSS safety proof. | 2A |
| F-18 | C | HB | `pyproject.toml`; R6.1 test output | 386 warnings remain: 378 narrowly attributable third-party deprecations and eight project-owned warnings. | 2A |
| F-20 | C | HB | `search/pipeline.py` execution-attempt validation and executor | `max_workers` can describe unsupported parallelism although the V1 executor is sequential; the backend contract must resolve or accept only one worker and refuse unsupported values. | 2 |
| F-21 | C | OD | `search/authorization.py`; current verification contract | `VerificationAuthorizationRef` requires a concrete seed, but no separate authorization contract exists for the real seed-producing replay. | 3 |
| F-22 | C | OD | R1 `WINDOW_COVERAGE_SCAN.json` / proposed coverage matrix | Current candidates mix logical trading days and physical/store partition dates; the permanent allowlist cannot be selected until that mapping is explicit. | 3 |

**Deferred cleanup, not a gate:** declare `threadpoolctl>=3.1`; register `candidate_not_in_assignment` in the reason vocabulary; remove the obsolete `oos_assignment_frame` after verified-source migration; bind `authorized_session_span_ns` to the session scheme; compare `stratified_frontier.children_without_strata` with `gates_passed`.

---

## 3. Phase 1 — Compact R6.1-FIX

One release-scoped commit, `R6.1-FIX`, parented by `6c0b60a`. Evidence goes under `..\R6.1-FIX\` using the existing release-evidence structure, including `R6.1-FIX.patch`, its SHA-256, and a Git bundle spanning R5B.1 through R6.1-FIX. Tests are written red first. No successful R6.1 component is redesigned. No existing immutable artifact is rewritten; changed semantic identities mint new synthetic/test artifacts beside prior ones. Golden-test `regime_fit_id`, `resolved_regime_protocol_id`, M0–M3 protocol identities, core replay IDs, and simulation IDs.

### 3.1 Assignment-source identity — F-01/F-02

- `ml/regime_store.py`: make `load_regime_fit_assignments` return `VerifiedFitAssignments{regime_fit_id, frame, assignments_sidecar_sha256, assignment_schema_hash}` from manifest-verified bytes under the enforced schema in §3.4.
- The `persist_regime_fit` reuse path compares canonical stored sidecar bytes byte-for-byte with the candidate bytes; matching bytes produce `REUSED`, different bytes under the same fit identity fail closed.
- `ml/regime_executor.py`: after each fit is persisted or verified-reused, exact-load its assignment sidecar. Descriptive candidate/panel assignment construction must consume only these verified frames, never `run.assignments`.
- `RegimeOosAssignmentPayload` gains canonically sorted `regime_fit_assignment_refs: tuple[FitAssignmentRef, ...]`, where each ref contains `regime_fit_id`, assignment-sidecar SHA-256, and assignment-schema hash. Validate that the derived `regime_fit_ids` projection exactly equals the ordered IDs in those refs.
- The consulted-source hash, when retained, covers every consulted row value that can change output: fit ID, row ID, fold, partition, local cluster ID, canonical reporting ID, full canonical distance vector, assigned distance, assignment margin, validity, and missing reason.
- Keep output `assignment_table_sha256` and output schema hash as post-materialization envelope/manifest facts. Add both to `RegimeAssignmentEvidenceRef`; S14 refuses a caller-supplied frame or hash that differs from the verified assignment artifact.
- Re-mint affected descriptive-assignment and stratified-report identities only.

### 3.2 Fold-feature source identity — F-03

- Extend `FoldFitRef` with non-null `assignments_sidecar_sha256` and `assignment_schema_hash` whenever `regime_fit_id` is present; all three fields are null together only for a legitimately absent fit.
- Change `build_regime_fold_features` to accept `Mapping[int, VerifiedFitAssignments]` and reject any unverified in-memory frame.
- S09b obtains those refs from the verified regime execution result. The fold-feature loader re-checks every exact fit assignment ref against the store without listing.
- Re-mint fold-feature, bundle-view, bundle-model, and controlled-study identities whose payload now includes the exact source refs. Preserve fit and protocol identities.

### 3.3 Candidate as-of policy — F-04

Adopt one explicit policy: **preserve the candidate with typed missingness**.

- Register `candidate_as_of_missing` in panel-assignment and fold-feature missing-reason vocabularies.
- `_as_of_ns` returns a null mask. A null source anchor creates one invalid assignment/feature row with the typed reason; an unparseable non-null timestamp remains a hard error.
- Candidate-grain behavior remains unchanged.
- The source hash represents null deterministically.
- Reports count the reason; no candidate silently disappears.

### 3.4 Assignment schemas and cross-field invariants — F-05

- Define immutable Arrow schemas and hashes for persisted fit assignments, descriptive OOS assignments, and model-facing fold assignments.
- For a **valid descriptive row**, require: row/candidate ID; fit ID; fold index; `partition="test"`; local cluster in `[0,k)`; canonical reporting cluster ID; complete finite distance vector of length `k`; finite assigned distance and margin; `missing_reason=null`.
- For a **valid model-facing row**, require: row/candidate/panel linkage; fit ID; fold; train/test partition; local cluster; complete finite distance vector; finite assigned distance/margin. Canonical reporting ID may be null only when the model-facing contract explicitly does not consume it.
- For an **invalid row**, retain the identity/linkage fields needed for reconciliation—row ID, candidate/panel ID, fold, partition, protocol and applicable fit reference. Set only assignment outputs (`local_cluster_id`, canonical ID, distances, assigned distance, margin) to null; require `valid=false` and one registered missing reason.
- Validate on build, save, verified load, and downstream consumption. Remove optional-column fallback branches.

### 3.5 Thin-regime accounting — F-09

Use all valid assigned trades for raw accounting; apply the minimum-trade floor only to interval/reportability metrics.

Definitions:

```text
net_r_by_regime[r] =
    sum(per_trade_net_r for valid assigned trades whose regime == r)

unassigned_net_r =
    sum(per_trade_net_r for invalid or unassigned trades)

abs_net_r_mass =
    sum(abs(net_r_by_regime[r])) over all assigned regimes

abs_net_r_share_by_regime[r] =
    abs(net_r_by_regime[r]) / abs_net_r_mass
    when abs_net_r_mass > 0, otherwise null with reason zero_abs_net_r_mass

top_regime_abs_net_r_share =
    max(abs_net_r_share_by_regime.values())
    when defined

works_only_in_regime =
    true only when overall assigned net R > 0,
    exactly one assigned regime has net R > 0,
    every other assigned regime has net R <= 0,
    and unassigned_trade_count == 0;
    otherwise false, or null with incomplete_assignment_accounting
    when unassigned trades prevent the claim.
```

Persist raw trade counts, `net_r_by_regime`, signed contribution fractions when total assigned net R is nonzero, absolute concentration shares, unassigned counts/R, formula version, and zero-denominator reasons. Do not infer statistical confidence for thin strata.

### 3.6 Exact label identity — F-08

- Make `ControlledFeatureStudyPayload.label_artifact_id` mandatory and 64-hex for every persisted study.
- Unpersisted helper runs may retain a full consumed-column content hash, but must stamp `label_identity_source="content_hash_unpersisted"` and cannot be saved, compared as an immutable study, or promoted.
- Persisting callers accept only `label_identity_source="label_artifact"`.
- The exact label artifact must bind every label/economic column consumed by the study, including candidate ID, binary target, gross/net R, trading day, setup ID, resolution timestamps, and the registered label policy.

### 3.7 Immutable executed-trade table — F-06

Create `executed_trade_tables` as a content-addressed search store.

```text
ExecutedTradeTablePayload
    core_replay_id
    source_record_schema_version
    table_projection_id = "core_executed_trade_exact_v1"
    executed_trade_arrow_schema_hash
    source_table_name = "executed_trade"

ExecutedTradeTableEnvelope
    executed_trade_table_id
    payload
    executed_trade_table_sha256
    source_core_table_hash
    row_count
    byte_size
```

`EXECUTED_TRADE_TABLE_SCHEMA_V1` is the exact ordered executed-trade projection consumed by costed evaluation and S14. It may not be inferred from a frame. S02 persists it after fresh completion and verified reproduction. On reuse, exact-load it by derived ID; verified identical bytes are `REUSED`, different bytes under one semantic ID fail closed, and missing/corrupt evidence creates a typed child-stage failure or `children_skipped` record—never silent omission. S14 iterates the charter child set and binds the table artifact ID/hash in its own identity.

### 3.8 Fail-closed prior-stage sidecars — F-07

Add a typed sidecar probe/load contract:

```text
sidecar_not_produced_for_path
    lawful optional absence

sidecar_missing_but_manifest_declares_it
sidecar_hash_mismatch
manifest_hash_mismatch
envelope_identity_mismatch
malformed_sidecar
unexpected_io_error
    stage failure
```

`has_sidecar` returns false only when a verified manifest proves that the sidecar was not produced for that computation path. All other errors propagate through sanitized typed failures. Persist reload-failure reasons in the S15 record. Never catch broad `Exception` around immutable evidence loads.

### 3.9 Production correctness cleanup — F-10A/B/C/D

- Replace production `assert` statements with `PipelineWiringError`, `SearchStoreError`, or a precise value/contract error.
- Use the normalized ordered executed-trade frame for every join, stratum, computation, and binding hash.
- Require full equality between `Mbp1PartitionEvidence.scope`, the gap-manifest scope, and the complete ordered partition-content refs before a positive completeness claim.
- Pass enum members into enum-typed payloads; do not pass `.value`.
- Warning fixes must preserve exact ordered columns and frozen schema. Construct schema-aligned typed frames and explicit dtypes; **never drop all-null columns** from frozen outputs.
- Golden table bytes/hashes and legacy schemas must remain unchanged.

### 3.10 Phase-1 gate

Required evidence:

- every new identity/tamper/leakage test red before code and green after;
- two full relevant suites, as-is and with provider keys cleared, with zero failures;
- Ruff, `git diff --check`, and exact golden-ID checks;
- double-run pipeline reuse with identical stage-result IDs and zero replay on verified reuse;
- zero protected/sealed access and zero new real-data artifacts;
- R5B.1-to-R6.1-FIX Git bundle and SHA-256;
- one focused two-reviewer adversarial pass;
- `implementation_status=complete`, `acceptance_status=transitively_blocked_by_R1`.

Commit boundary:

```text
R6.1-FIX: bind verified assignment evidence, enforce schemas,
persist executed trades, and fail closed on corrupt sidecars
```

No push and no merge.

---

## 4. Phase 2 — Backend hardening

This phase is limited to backend governance, immutable-evidence integrity, capacity, runtime-contract truthfulness, warnings, and release evidence. Presentation-layer implementation and its acceptance gates are out of scope.

### 4.1 Semantic store namespace — F-11

Implement a path-independent immutable namespace contract:

```text
StoreNamespacePayload
    namespace_schema_version
    namespace_class: research | test
    store_instance_id: stable UUID generated or explicitly supplied once
    authority_genesis_id

StoreNamespaceEnvelope
    store_namespace_id = hash(payload)
    payload
```

Rules:

- `store_instance_id` is not an absolute-path hash. Store relocation preserves semantic identity.
- Every owner-decision artifact, supersession record, authorization bundle, seed-production authorization, and verification authorization references `store_namespace_id`.
- An unmarked existing store cannot authorize a write or real launch. A one-time explicit migration command initializes the namespace after showing the operator the intended class; it never infers class from a path.
- Filesystem naming checks may remain as defense-in-depth configuration checks. They may refuse a suspicious deployment, but they do not define or change semantic authority.
- Namespace envelope bytes are immutable and verified on every authorization load.

### 4.2 Immutable supersession chain and head witnesses — F-12

Do not write into an existing `owner_decisions/<decision-id>/` directory.

Create:

```text
owner_decision_supersessions/<supersession-record-id>/
```

with an immutable `OwnerDecisionSupersessionPayload/Envelope` binding:

```text
store_namespace_id
superseded_decision_id
replacement_decision_id
prior_head_record_id
prior_head_sha256
prior_line_count
reason
effective_at
owner_evidence_ref
```

Publication under the owner-decision lock:

1. verify namespace, current head, source decision, replacement decision, and nondivergence;
2. write and verify the immutable supersession record in a temporary directory;
3. atomically publish that record;
4. atomically replace the required `SUPERSESSIONS.head` pointer with `{record_id, line_count, head_sha256}`;
5. if failure occurs before step 4, the orphan record has no authority; if failure occurs after step 4, the head points only to an already verified immutable record.

Additional rules:

- A namespace is initialized with an explicit genesis head. Missing head is corruption, never “no supersessions.”
- Verified identical replay is idempotent reuse; a divergent record for the same transition is refused.
- Every real charter and authorization bundle records the current `{store_namespace_id, line_count, head_sha256}` witness. Verification refuses a missing, shorter, or different current head.
- The same witness rule applies to strategy, MBP-1, regime, prop, publication, seed-production, and verification authorizations—not only regime promotion.
- This provides local rollback detection, not cryptographic owner authenticity. Deleting or rewriting the entire store plus every external witness remains outside the trust boundary and must be stated honestly.

### 4.3 Liveness-aware owner-decision lock — F-13

The lock contains `pid`, process-start token where available, random lock token, host, creation time, and heartbeat time.

- Reclaim only when the heartbeat exceeds the timeout **and** the recorded process is demonstrably not alive.
- Refresh heartbeat before long verification and before append/commit.
- Every writer re-verifies the lock token before publishing.
- Release unlinks only its own token.
- A lost lock aborts the writer before publication.
- Test a slow live holder, a dead PID, PID reuse where process-start metadata is available, token mismatch, and crash recovery.

### 4.4 Capacity implementation and numerical gates — F-17

Capacity proof uses synthetic data only and does not run a research simulation.

#### Event-detail writer

Before benchmarking:

- change `search_bridge`/event-detail input to an iterator or generator; no all-path event list may be constructed;
- flush bounded Arrow/Parquet row groups;
- remove the whole-artifact `np.concatenate` ID-index spike;
- prove event-ID uniqueness by the existing deterministic identity projection and per-path/account event ordinal. If any source cannot prove uniqueness from its canonical key, use a disk-backed external uniqueness check rather than a global in-memory set.

#### Regime-stratified event summary

- stream verified event-detail partitions;
- write canonical intermediate partitions;
- perform exact external aggregation with DuckDB or an equivalent spill-capable engine under an explicit memory limit and attempt-local temp directory;
- compute exact unique-path counts externally;
- canonical-sort final rows before hashing/publication;
- clean temp files on success/failure without altering immutable outputs.

#### `HARDENING_CAPACITY_POLICY_V1`

Run each benchmark at 250,000, 500,000, and 1,000,000 input rows. Record total process RSS with a native/process monitor; `tracemalloc` is supplementary.

| Gate | B1 event detail | B2 stratified summary |
|---|---:|---:|
| Minimum available RAM at start | 8 GiB | 8 GiB |
| 1M-row peak RSS increase | ≤1.5 GiB | ≤1.5 GiB |
| 1M-row Python allocation peak | ≤512 MiB | ≤768 MiB |
| 1M-row wall time | ≤300 s | ≤300 s |
| Growth from 500k→1M | per-row RSS slope ≤1.25× `max(250k→500k slope, 64 B/row)` | same |
| Projection at registered maximum | ≤`min(6 GiB, 50% of minimum measured available RAM)` | same |
| Serialized artifact | ≤2 GiB at 10M detail rows | ≤256 MiB at 5M summary rows |
| Determinism | byte-identical output/hash on repeat | byte-identical output/hash on repeat |

Hardening cannot pass on extrapolation alone. It passes only when the implementation is block/partition bounded and the projected registered ceiling satisfies every numerical gate. If it does not, either:

1. implement further streaming/externalization and remeasure; or
2. make an explicit versioned capacity-policy change that lowers the supported ceiling to a measured safe bound, with owner approval.

Record measured and extrapolated values separately in `CAPACITY_BENCHMARKS.md`.

### 4.5 Warning policy — F-18

- Fix all project-owned warnings with explicit schemas/dtypes; preserve all frozen columns, table order, bytes, and hashes.
- Add only narrowly scoped third-party warning rules with exact message, category, module, affected package version, reason, owner, and expiry/removal condition.
- No broad `DeprecationWarning` suppression.
- Run the relevant suite under `filterwarnings=error`; every warning not in the exact allowlist fails.
- Evidence reports project-owned warnings = 0 and identifies each remaining third-party warning rule.

### 4.6 Operational truth: sequential execution in V1 — F-20

The current backend executor is sequential. The backend contract must not claim or accept unsupported parallel child execution.

- Define the V1 execution capability as `supported_child_workers=1`.
- Validate `ExecutionAttemptPayload` and every backend launch entry point so `max_workers` must equal 1. A request above 1 fails before job creation with typed reason `unsupported_worker_parallelism_v1`.
- Persist `effective_workers=1` and `execution_mode="sequential_children_v1"` in attempt receipts and runtime evidence; these remain operational metadata and never alter scientific identity.
- Do not silently coerce a requested value greater than 1. The caller must correct and resubmit the attempt.
- A real process pool is a separately versioned future capability, not part of hardening.
- Before the Full Authorized Development run, execute a bounded current-code throughput benchmark over the authorized small fixture and use its observed replay/day and stage timings. Do not rely solely on the older 15-seconds/day planning estimate.

### 4.7 Phase-2 acceptance evidence

One `HARDENING-BACKEND` commit and evidence folder:

```text
HARDENING-BACKEND.patch
Git bundle + SHA-256
raw pytest and JUnit
Ruff over touched backend modules, tests, and backend CLI scripts
git diff --check and diff stat
namespace migration/relocation evidence
supersession rollback/crash evidence
lock liveness evidence
CAPACITY_BENCHMARKS.md
WARNING_BASELINE.json
clean-tree proof
zero new real-data artifacts
full_pipeline_not_run=true
```

Phase 2 is accepted when the backend governance, capacity, warning, runtime-contract, and evidence gates in this section pass. Presentation-layer evidence is not part of this backend acceptance.

---

## 5. Phase 3 — R1 fixture, seed production, and authorization unblock

### 5.1 Canonical date semantics and shortlist — F-22

The permanent verification allowlist is an ordered tuple of at most five **consecutive logical trading-day IDs** accepted by the Strategy-Core/Quant-Lab trading calendar. It is not a list of physical UTC partition dates.

For each logical day, persist a mapping:

```text
VerificationTradingDayRef
    logical_trading_day
    session_open_ts_utc
    session_close_ts_utc
    ordered_source_partition_refs:
        physical UTC date
        relative logical partition key
        publisher/channel/source kind
        content hash / manifest ref
```

Before owner selection, rebuild the coverage report from already-authorized evidence only. Produce a ranked shortlist containing at least:

1. the existing June proposal;
2. the highest lifecycle/candidate/decision/trade-coverage window;
3. the highest complete-audit-day-coverage window.

Hard constraints:

```text
1–5 consecutive logical trading days
all source partitions present and hash-addressable
profile-compatible seed producible through the prior logical day
at least one exact verifier setup target
June 11 and the sealed range excluded
one common window for the entire implementation-verification program
```

Lexicographic ranking, with no hidden score:

```text
required source/replay coverage
→ all required lifecycle path classes represented
→ audit-day coverage
→ executed trades
→ decisions
→ candidates
→ exact verifier targets
→ MBP-1 scope evidence
→ panel/control-flow coverage
→ distance from protected/sealed boundary
```

The current `2026-02-06…02-11` “store-day” candidate is **provisional only**. February 8 is not accepted as a logical trading-day ID unless the calendar mapping proves that exact meaning. Do not hash or call `register_program_allowlist` until the owner selects from the corrected logical-day shortlist.

### 5.2 Seed inventory — F-16

No compliant seed exists. Existing v2, replay-chart, and FSM-audit artifacts contain tables/evidence, not a restorable `IfvgDaySeed` state graph. Derivation from those tables is not allowed.

Lawful options remain:

```text
reuse a verified matching seed — none found
derive from an already-persisted exact terminal state — unavailable
run a separately authorized seed-production chain
remain blocked
```

### 5.3 Separate seed-production authorization — F-21

Implement a dedicated policy and contracts before any real seed-producing replay:

```text
SeedProductionReplayPolicy
SeedProductionAuthorizationPayload / Envelope / Ref
SeedProductionRunPayload / Envelope
```

The authorization payload binds:

```text
store_namespace_id and supersession-head witness
baseline profile name and resolved section hash
ordered logical seed-chain trading days
snapshot_through_day
first intended verification day
access-policy ID
expected ordered source-inventory hash
Quant-Lab and Strategy-Core source identities
seed schema version
chain policy ID
final_day_exhausts_dataset = false
permitted outputs = seed_snapshot + access_audit + run receipt
prohibited outputs = research/search metrics, candidate/trade performance reports,
                     model fitting, feature studies, prop simulation,
                     frontier/insights, research-catalog publication
owner decision refs, approver, effective time, content hash
```

Rules:

- authorization is verified before any source path is constructed;
- exact dates and source hashes only; no listing;
- output goes to a seed-production/test namespace, not the research catalog;
- the seed snapshot is profile-bound, content-addressed, schema-validated, and verified on reload;
- every report states the exact number of seed-chain days and that this is a separately authorized preparation action, not part of the `≤5` verification evidence footprint;
- any profile/source/code/date divergence requires a new authorization.

### 5.4 Two-stage owner flow

```text
A. Select a provisional logical verification window.
B. Create the unsigned seed-production packet.
C. Owner signs SeedProductionAuthorizationRef.
D. Run only the authorized seed-production chain.
E. Publish and verify one seed snapshot and access audit.
F. Owner reviews the concrete seed ID and provenance.
G. Build the final unsigned VerificationAuthorizationRef packet.
H. Owner signs the final verification authorization.
I. Register the one program allowlist and run Phase 4.
```

A failed or rejected seed-production run does not authorize verification.

### 5.5 Final verification packet

Only after a verified seed exists, generate the unsigned packet containing:

```text
corrected logical allowlist and hash
coverage-matrix artifact and ranking trace
baseline profile and section hash
verified seed snapshot ID, hash, schema, and provenance
VerificationReplayPolicy ID
store namespace and supersession-head witness
pipeline/verification semantic identity
expected ReplayInputBundle source-inventory hash
zero protected/sealed requirements
verification_only / not_for_research_interpretation /
full_pipeline_not_run stamps
blank owner approval fields
```

Only the owner may create the final `VerificationAuthorizationRef`.

---

## 6. Phase 4 — Real ≤5-day verification

The real verification starts only after Phase 1, Phase 2, seed production, and the final owner authorization have passed.

### 6.1 Preflight

Before path construction:

- exact-load and verify `StoreNamespaceEnvelope`;
- verify supersession-head witness;
- verify `VerificationAuthorizationRef`;
- verify the permanent logical allowlist and its physical partition mapping;
- exact-load the profile-matching seed;
- reject a sixth day, rotated window, date-domain mismatch, source-inventory mismatch, synthetic marker, profile/seed mismatch, or research-catalog destination;
- lock output to `search_test/v1`.

### 6.2 R1 baseline verification gates

Run the exact baseline in audit-disabled and audit-enabled modes.

Required `verification_control_flow_gates_v1`:

```text
replay_completed
invariants_passed
artifacts_published_and_reloaded
neutrality_passed
verifier_link_resolves
zero_forbidden_counters
```

Also prove:

- native IDs and core table hashes repeat;
- the executed-trade table artifact exact-loads;
- a second identical invocation returns `REUSED` with zero replay invocations;
- identical existing bytes are a pass/reuse;
- same semantic ID with different bytes, missing/corrupt manifest, or corrupt sidecar fails closed.

### 6.3 Release-specific bounded-control-flow gates

Create a separate immutable `BoundedReleaseControlFlowReport`. These checks do not become research evidence.

| Component | Required bounded result |
|---|---|
| MBP-1 diagnostic | Same allowlist/marker; diagnostic completes; current expected state `completeness_unknown` is typed; no feature eligibility or research claim. |
| 5m/15m panel | Verified source identity; materializes or returns a typed insufficiency state; no protected/sealed access. |
| Fold construction | The five-day fixture produces a typed `no_valid_fold`/insufficient-history result under the real fold protocol; no fabricated valid fold. |
| KMeans/regime | If no lawful fit can be made, persist the typed non-fit/capability result; never persist an invalid fit. Any diagnostic fit is verification-only. |
| CatBoost/logistic | No valid fold produces a typed non-fit result and zero fabricated predictions. |
| S14 | Zero estimator fitting; required verification report or an explicit typed skip with evidence. |
| S15 | Verification-only result; no research publication, frontier selection, promotion, or activation. |
| S11 | Remains blocked with the exact registered reason. |

The overall bounded verification passes only when both the six R1 baseline gates and every applicable release-specific gate pass.

### 6.4 Evidence and artifacts

Expected artifacts:

```text
verification_runs
core_replays
executed_trade_tables
audit/replay-chart companions
pipeline_stage_results
context_bar_panels or typed insufficiency records
fold_sets or typed no-valid-fold record
regime capability/non-fit records
MBP-1 coverage diagnostic
BoundedReleaseControlFlowReport
```

The seed snapshot is consumed, not created, in Phase 4.

Evidence folder `..\R1-VERIFICATION\` includes:

```text
VERIFICATION_RUN.md
R1_BASELINE_GATES.json
BOUNDED_RELEASE_CONTROL_FLOW.json
ACCESS_AUDIT.jsonl
pytest/JUnit
manifest and sidecar hashes
reuse receipt
verifier deep link
full_pipeline_not_run=true proof
```

Abort on any authorization, namespace, head-witness, source, date, seed, neutrality, immutable-evidence, or forbidden-access failure.

Research profitability, strategy, payout, feature-selection, or promotion gates are never applied to this fixture.

---

## 7. Phase 5 — Full Authorized Development operator run

This is a separate post-acceptance action.

### 7.1 Preconditions

```text
R6.1-FIX accepted
Phase 2 backend hardening accepted
seed production completed and verified
Phase 4 bounded verification accepted
R1 and dependent releases accepted
clean source tree
required owner decisions current and unsuperseded
first-party verified prop contracts for selected prop stages
resource/storage budget approved
an explicit owner-selected backend invocation entry point
```

The invocation entry point may call the backend contracts, but it cannot alter their semantics, authorization checks, identities, or gates.

### 7.2 Freeze every result-bearing semantic input

The computation-path-specific authorization mechanism still requires only decisions used by the chosen path. Separately, the frozen pipeline specification and its preflight report must enumerate and bind every result-bearing input, whether or not it needs an owner decision:

```text
market universe and contract-roll policy
strategy baseline, axes, and exact values
observation cohort and warmup/date policy
label policy and exact label artifact
feature block/bundle identities
fold schedule and fold-set identities
validation protocol
prevalence/logistic/CatBoost model protocols
decision and calibration policies
regime protocol, fit policy, assignment policy, and promotion evidence
execution and cost policies
firm contracts
risk, withdrawal, replacement, payout, and portfolio policies
path-fidelity and simulated-clock policies
bootstrap and stress scenarios
strategy, prop, and robustness gates
outer evaluation/publication policy
repository/source identities and data lineage
```

No hidden default may affect results.

### 7.3 Resource forecast

Recompute the operator forecast after Phase 2 throughput/capacity measurements. Separate measured values from estimates.

At minimum report:

```text
strategy child count
observed replay seconds/day and projected time/child
sequential execution policy (V1)
feature and panel materialization time
logistic/CatBoost/KMeans time
firm × risk-policy × scenario counts
bootstrap/stress path counts
D15 detail and summary row estimates
peak RSS under the measured hardening policy
disk usage
checkpoint cadence
total wall time and uncertainty range
```

Do not divide by `max_workers`; V1 is sequential.

### 7.4 Backend operator lifecycle

The backend must expose one deterministic operator lifecycle through an approved operator entry point:

```text
resolve and validate the selected computation path
emit a preflight report containing every frozen semantic input
freeze `PipelineSemanticSpec`
create a distinct `ExecutionAttemptIdentity`
explicitly launch the job
persist stage progress and resource evidence
cancel only at a documented safe boundary
resume/retry with the same semantic identity and a new attempt identity
verify immutable stage outputs and final gates
prepare publication only for an eligible research run
activate a research-catalog entry only through a separate approved action
```

Rules:

- no test, import, application startup, catalog read, or preflight may auto-launch the full run;
- verification-scoped results are refused by the publication service;
- retry/resource changes never mint a new scientific identity;
- service/CLI errors are typed and sanitized.

### 7.5 Outputs and interpretation

Expected immutable outputs include:

```text
charters and pipeline specifications
core replays and search-child memberships
executed-trade tables and costed evaluations
audit and chart companions
feature/label views
fold schedules and fold sets
MBP-1 and context-panel artifacts
regime protocols, fits, assignments, capabilities, and fold-local features
prevalence/logistic/CatBoost predictions and comparisons
prop simulations and event-detail artifacts
stratified reports, frontier, robustness, and insights
pipeline result and publication-gate result
```

Boundaries:

```text
no automatic winner selection
no automatic feature/model/threshold/cluster selection
no automatic promotion
no profitability claim before human review
Development Exploratory Representative unless an approved outer evaluation passes
S11 remains blocked
MBP-1 remains research_only_offline
no Trade-Lab serving, live execution, or orders
```

---

## 8. Owner decision and authorization matrix

| Capability | Blocking decisions / evidence | Authoring | Acceptance / real use |
|---|---|---|---|
| R6.1-FIX | none | after plan approval | transitively blocked by R1 until Phase 4 |
| Phase 2 backend hardening | none; one-time namespace initialization is an administrative action | proceeds | must pass numerical, governance, warning, and runtime-contract gates |
| Verification-window selection | decision 21 / R-5 coverage review | no data run | owner selects one corrected logical-day window |
| Seed production | `SeedProductionAuthorizationRef` over exact chain/profile/source/code | contract authoring proceeds | owner signature required before source-path construction |
| R1 verification | concrete verified seed + final `VerificationAuthorizationRef` + current namespace/head witness | code authored | owner signature required; both gate layers must pass |
| Strategy-only full run | decisions 1, 2, 7, R-2 and selected §15 gates | authored | current owner artifacts required |
| MBP-1 study | R1 accepted; coverage policy v2 passes on authorized real evidence; R-6 keeps offline-only boundary | authored | verification control flow alone does not authorize research use |
| Regime descriptive study | R1 accepted; selected protocol and capability evidence | authored | descriptive only unless further status is authorized |
| `feature_only` / `cohort_model` | decisions 25/28/29/30 over the exact real protocol and capability assessment | authored | `FEATURE_ELIGIBLE` only through the owner action |
| Prop benchmark | decisions 5/6/8, R-3, R-4, verified clock/fidelity policies, first-party contracts | authored | synthetic contracts never become real contracts |
| Publishable representative | decision 15 and approved outer evaluation | schema-reserved | otherwise `Development Exploratory Representative` |
| S11 / live serving | R-1 and a separate Strategy-Core parity/serving contract | blocked | unavailable in V1 |

---

## 9. Test and evidence matrix

| Phase | Test / evidence | Fixture | Required assertion | Failure meaning |
|---|---|---|---|---|
| 1 | Descriptive assignment source identity | perturbed verified fit sidecar | source ref/hash/schema changes ID or fails tamper check | artifact ID does not determine output |
| 1 | Verified-source derivation | monkeypatched in-memory assignments | output equals verified stored bytes, not memory object | unverified evidence leak |
| 1 | Fold-feature source identity/leakage | both grains and fold-local fits | exact sidecar refs bind identity; fold `k` uses fit `k` only | unbound feature source or leakage |
| 1 | Typed missing as-of | one null anchor | linked invalid row retained with registered reason | population loss/contract contradiction |
| 1 | Assignment schema/invariants | partial and tampered sidecars | build/save/load refuse invalid cross-field states | invalid assignment accepted |
| 1 | Executed-trade artifact | fresh and reused child, alternate cost policy | exact table persists/loads; missing child gets typed reason; tamper fails | S14 silently drops evidence |
| 1 | Prior-stage sidecar failures | tampered vectors/simulations/deltas/reports/S09c | corruption is typed stage failure; only manifest-proven nonproduction is optional | tamper treated as absence |
| 1 | Thin-regime accounting | sub-floor strata plus unassigned rows | all assigned R enters raw concentration; confidence remains floor-gated | concentration understated |
| 1 | Exact label identity | persisted study without exact label artifact | persistence refused | narrow label identity |
| 1 | MBP-1 scope/enum/normalized-frame checks | mismatched scope and warning fixtures | full scope equality, enum-correct serialization, normalized hash | positive claim or identity drift |
| 1 | Golden identities | existing M0–M3, fit, protocol, core replay | unchanged unless explicitly listed as re-minted | unintended compatibility break |
| 2 | Namespace relocation/migration | research/test temp roots | authority follows `store_namespace_id`, not path; unmarked store fails closed | path-derived authority |
| 2 | Supersession atomicity/rollback | crash before/after head, missing/old head | no decision-dir mutation; orphan has no authority; rollback refused | superseded decision restored |
| 2 | Lock liveness | slow live holder, dead PID, token mismatch | one writer; live holder not reclaimed; lost lock aborts | decision-chain corruption |
| 2 | Capacity B1/B2 | 250k/500k/1M synthetic rows | every numerical gate and deterministic repeat passes | registered ceiling unsupported |
| 2 | Warning policy | suite under warnings-as-errors | zero project warnings; only exact versioned third-party rules | warning regression |
| 2 | Sequential-executor truth | legacy/new execution attempts | backend accepts only effective workers = 1; >1 fails before job creation; receipts state sequential mode | runtime contract claims unsupported parallelism |
| 3 | Date-domain mapping | candidate shortlist | logical days are consecutive and map to exact physical partitions | invalid permanent allowlist |
| 3 | Seed-production authorization | negative and signed synthetic refs | absent/stale/divergent auth fails before path construction | unauthorized real seed replay |
| 3 | Seed-production run | exact authorized chain | only seed/audit/receipt produced; seed reload/profile/source hashes match | seed not trustworthy |
| 4 | Verification preflight | sixth date, rotation, source/seed/profile mismatch | fail before path construction | access boundary broken |
| 4 | R1 baseline gates | authorized slice | all six baseline gates pass, verified reuse exact | R1 not accepted |
| 4 | Bounded release-control gates | same slice | MBP-1/panel/folds/models/S14/S15 reach exact typed safe states | later release not verified |
| 5 | Frozen semantic preview | selected full path | every result-bearing input shown and hash-bound | hidden input changes results |
| 5 | Publication boundary | completed operator fixture | service refuses verification publication; no automatic rank, promotion, or activation | unauthorized catalog activation |

---

## 10. Execution order and final readiness checklist

**Required backend sequence:**

```text
1. Approve revision 3 of this backend-only plan.
2. Implement and verify R6.1-FIX.
3. Implement and accept Phase 2 backend hardening.
4. Resolve logical trading-day mappings and select the permanent verification window.
5. Owner signs SeedProductionAuthorizationRef.
6. Run and verify the seed-only production chain.
7. Owner reviews the concrete seed and signs VerificationAuthorizationRef.
8. Run the real ≤5-day R1 baseline and bounded release-control verification.
9. Accept R1 and dependent releases.
10. Freeze every semantic input for the selected Full Authorized Development path.
11. Explicitly launch the Full Authorized Development pipeline.
12. Perform human research review of immutable results.
```


| Readiness item | Current status | Completion condition |
|---|---|---|
| R1–R6, R5B.1, R6.1 source | implemented; synthetically verified | preserved through all fixes |
| R6.1-FIX | not implemented | Phase-1 gate passes |
| Phase 2 backend hardening | open | governance, capacity, warning, and runtime-contract gates pass |
| Logical allowlist | not selected | owner selects corrected logical-day shortlist |
| Seed-production authorization | absent | signed owner ref over exact chain |
| Profile-matching seed | absent | separately authorized seed snapshot published and verified |
| Final verification authorization | absent | signed ref includes concrete seed and current head witness |
| Real ≤5-day verification | not run | both R1 and release-specific gate layers pass |
| Full-run semantic decisions | proposals/unfrozen | only selected-path decisions current and every result-bearing input frozen |
| Full development run | not run | explicit owner launch completes immutable pipeline |
| Research conclusion | unavailable | human review; exploratory unless outer evaluation passes |
| S11, Trade-Lab serving, live execution | unavailable | remains unavailable under this plan |

**Confirmations:** this revision changes documentation only. It does not authorize or perform a seed replay, verification run, model fit, MBP-1 build, prop simulation, search, catalog mutation, owner decision, or full pipeline. It makes no presentation-layer implementation change. Strategy-Core and Trade-Lab remain read-only; M0–M3 and all existing immutable artifacts remain unchanged; June 11 and the sealed range remain prohibited; one canonical logical ≤5-day fixture serves the full implementation-verification program; any longer seed-production chain is separately disclosed, separately authorized, and excluded from research interpretation.
