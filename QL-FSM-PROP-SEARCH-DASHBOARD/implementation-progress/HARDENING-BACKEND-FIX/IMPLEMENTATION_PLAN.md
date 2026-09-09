# HARDENING-BACKEND-FIX — Focused Backend Correction Implementation Plan

**Feature:** `ifvg_prop_robust_config_search_v1`  
**Repository:** `C:\Users\gonza\Documents\Claude-Quant-Lab`  
**Expected parent:** HARDENING-BACKEND commit `e56f9376a5b4ba269f7fa11cdc6e37b08200638f` on `feature/ifvg-prop-robust-config-search-v1`  
**Release type:** compact corrective release; not another major backend phase  
**Scope:** Quant-Lab backend only  
**Status:** implementation plan awaiting owner approval  
**Implementation output folder:**  
`C:\Users\gonza\Documents\Claude-Quant-Lab\QL-FSM-PROP-SEARCH-DASHBOARD\implementation-progress\HARDENING-BACKEND-FIX\`

This plan directly authorizes implementation after owner approval. **Do not create another implementation plan.** Begin with the bounded baseline checks in §2, write the targeted tests, implement only the corrections in §§4–10, and stop at the release gate in §12.

The release exists to stabilize the backend contracts before the separate UI/UX implementation begins. It does **not** authorize the owner verification run, seed production, a full development pipeline, a strategy search, model fitting on real data, prop simulation on real data, catalog activation, or any live-serving work.

---

## 1. Outcome and release boundary

### 1.1 Required outcome

At the end of this release, the backend must be safe and stable for the UI branch to consume:

```text
implementation_status: complete
backend_dev_complete_for_ui: true
ui_implementation_may_begin: true
formal_acceptance_status: transitively_blocked_by_R1
real_verification_run_completed: false
full_authorized_development_run_completed: false
```

The final two `false` values are expected. Owner fixture selection, seed-production authorization, seed production, `VerificationAuthorizationRef`, and the real bounded verification remain later owner actions. They do not need to be completed before UI code begins.

### 1.2 Included corrections

This one release closes only:

1. token-safe stale owner-decision-lock reclamation;
2. concurrency-safe store-namespace initialization;
3. the public `legacy_verified_replay_source` boundary for historical MBP-10-era physical files;
4. invalid OOS regime-assignment provenance retention;
5. strict pre-coercion assignment and fold-feature validation;
6. complete manifest-entry and path validation;
7. central seed datetime canonicalization;
8. a hard per-partition event-detail memory bound plus worst-lawful-shape capacity proof;
9. complete supersession-chain verification at every real authority seam;
10. the directly related schema/set/identifier integrity gaps enumerated in §7.

### 1.3 Explicit non-goals

Do not:

- redesign the owner-decision, namespace, seed, verification, regime, or prop architecture;
- change Strategy-Core;
- change Trade-Lab;
- change the fixed M0–M3 lane or its modules;
- implement any UI or Streamlit changes;
- implement browser, viewport, keyboard, chart, or accessibility work;
- change strategy rules, profiles, labels, model formulas, KMeans formulas, CatBoost parameters, or prop-firm rules;
- add GMM, spectral clustering, Nyström, model gating, or S11 functionality;
- implement parallel child execution;
- lower any registered capacity ceiling without explicit owner approval;
- modify, widen, migrate, or delete an existing immutable artifact;
- run a full development replay, full pipeline, real model fit, real prop simulation, or strategy search;
- choose the permanent verification window;
- create a real `SeedProductionAuthorizationRef`, seed artifact, or `VerificationAuthorizationRef`;
- access June 11, 2026 or the sealed range;
- push or merge the branch.

Any issue outside this list is a carry-forward unless it directly prevents one of the named acceptance criteria from passing.

---

## 2. Baseline and preflight

The main implementation agent owns this release. Before editing:

1. Verify the branch and HEAD:
   ```text
   branch = feature/ifvg-prop-robust-config-search-v1
   HEAD   = e56f9376a5b4ba269f7fa11cdc6e37b08200638f
   ```
   A later owner-authored commit may be present. Do not reset, clean, stash, or overwrite it. If HEAD has advanced, identify the exact delta and continue only when the named backend seams are still compatible.

2. Capture:
   ```text
   git status --short
   git diff
   git diff --cached
   git log --oneline --decorate -15
   ```

3. Preserve every pre-existing user-owned hunk. Record its hash or patch before implementation and compare it again after the release.

4. Read only the necessary authority and evidence:
   ```text
   implementation-progress\HARDENING-BACKEND\
   implementation-progress\R6.1-FIX\
   R6_1_FIX_HARDENING_READINESS_IMPLEMENTATION_PLAN_BACKEND_ONLY_REVISED_V3.md
   HARDENING_BACKEND_INDEPENDENT_REVIEW.md
   FINAL-IMPLEMENTATION-PLAN-DOCS\CONTRACTS_AND_SCHEMAS.md
   FINAL-IMPLEMENTATION-PLAN-DOCS\TEST_MATRIX.md
   FINAL-IMPLEMENTATION-PLAN-DOCS\OWNER_DECISIONS.md
   ```

5. Establish the supplied baseline without rerunning the full suite:
   ```text
   prior full suite: 2,102 passed, 0 failed
   warning mode: errors
   Ruff: clean
   git diff --check: clean
   ```
   Run only the targeted tests during implementation. Run the complete suite exactly at the final gate in §12.

6. Create:
   ```text
   implementation-progress\HARDENING-BACKEND-FIX\
   ```
   Do not edit the evidence folders of prior releases.

---

## 3. Effort control and subagent restrictions

This is a focused correction run. Long-running dynamic agent workflows are prohibited.

### 3.1 Main-agent responsibility

The main agent must:

- perform the code inspection;
- write the tests;
- implement the fixes;
- run the targeted and final gates;
- reconcile the findings;
- write the completion report.

Do not delegate the implementation to a swarm.

### 3.2 Reviewer budget

At most **two read-only reviewers total** may be used, in **one single review pass** after the targeted implementation and tests are green:

```text
Reviewer A:
    owner-decision lock
    namespace initialization
    supersession-chain authority seams
    seed canonicalization

Reviewer B:
    regime assignment/fold evidence
    manifest validation
    legacy source-kind boundary
    event-detail capacity
```

Binding limits:

- no reviewer may spawn another agent;
- no nested or dynamic workflows;
- no second adversarial round;
- no repeated “review until consensus” loop;
- reviewers may not edit files or run long test suites;
- each reviewer inspects only the changed diff and the named call paths;
- each reviewer returns at most 1,500 words and no more than five findings;
- style, naming, comments, formatting, and unrelated cleanup are out of scope;
- outside-scope observations go into one short carry-forward list and do not expand the release;
- the main agent fixes in-scope blockers once, reruns targeted tests, then proceeds to the final gate.

### 3.3 Ambiguity stop rule

Investigate current contracts and tests first. Ask the owner only when a choice would:

- change scientific or artifact identity beyond the re-mints listed in §11;
- lower a supported capacity ceiling;
- require Strategy-Core, Trade-Lab, M0–M3, or UI changes;
- alter owner-authorization semantics;
- require mutation of an existing immutable artifact.

Ask no more than three concise questions in one batch. Do not assume a new policy.

---

## 4. Workstream A — Owner-decision lock and namespace initialization

### 4.1 Token-safe stale-lock reclamation

**Primary file**

```text
src/alpha_lab/agents/data_infra/ifvg/search/owner_decision_lock.py
```

**Current defect**

The current path can:

```text
read stale lock S
decide S is reclaimable
later unlink the current lock path
```

Two reclaimers can both inspect `S`; the first may replace it with live lock `A`, after which the second can delete `A`.

**Required implementation**

Reuse an existing correct cross-platform mutex primitive if the repository already has one. Otherwise add a private, standard-library-only reclaim mutex in this module:

```text
Windows: msvcrt byte-range exclusive lock
POSIX:   fcntl.flock exclusive lock
```

Use it only around stale-lock reclamation.

Required sequence:

1. Try ordinary `O_CREAT | O_EXCL` acquisition.
2. When the ordinary lock exists and appears stale:
   - acquire the dedicated reclaim mutex;
   - re-read the ordinary lock under that mutex;
   - re-evaluate heartbeat age, host, PID liveness, and process-start token;
   - compare the exact `lock_token` and body that are being reclaimed;
   - unlink only when the still-current body is demonstrably the same stale/dead holder;
   - release the reclaim mutex;
   - retry ordinary acquisition.
3. A stale decision made before acquiring the reclaim mutex grants no authority to unlink.
4. Another host, malformed body, or unknown liveness remains non-reclaimable.
5. A lost ordinary lock token aborts before publication.

Do not “fix” this with a second unlocked read immediately before `unlink`; that remains a TOCTOU.

**Related error behavior**

- Distinguish:
  ```text
  lock file absent
  malformed lock body
  persistent read I/O failure
  ```
- `_read()` must never convert persistent I/O failure into absence.
- `release()` must raise a typed failure if it cannot read or verify its own lock.
- If `_try_create()` succeeds at `O_EXCL` creation but body write/fsync fails, remove or quarantine the partial file before returning a typed failure.
- Preserve token-only release: a process never removes a lock it does not own.

**Tests first**

Extend:

```text
tests/agents/ifvg_search/test_owner_decision_lock.py
```

Required tests:

```text
test_two_stale_reclaimers_cannot_delete_the_winner
test_reclaimer_that_observed_old_token_cannot_unlink_new_live_token
test_persistent_read_error_is_typed_not_absence
test_release_read_error_does_not_silently_leave_success
test_failed_body_write_cleans_partial_exclusive_lock
test_unknown_liveness_remains_non_reclaimable
```

The two-reclaimer test must use real concurrency synchronization—not sequential mocks that cannot reproduce the race.

### 4.2 Atomic and recoverable store-namespace initialization

**Primary file**

```text
src/alpha_lab/agents/data_infra/ifvg/search/store_namespace.py
```

**Required implementation**

Add a one-time initialization mutex independent of the ordinary owner-decision lock.

Under the mutex, classify the store as exactly one of:

```text
both namespace and genesis absent
both present and coherent
namespace present / genesis absent
genesis present / namespace absent
both present but inconsistent
```

Rules:

1. **Both absent**
   - precompute deterministic namespace-envelope and genesis-head bytes;
   - write through temporary files;
   - publish under the initialization mutex;
   - verified-load both objects before returning.

2. **Both present and coherent**
   - the identical request is idempotent reuse;
   - a divergent namespace class or store instance is a typed refusal.

3. **One present**
   - recover only if the existing object exactly matches the deterministic object required by the same requested initialization;
   - otherwise raise `incomplete_store_namespace_initialization` or an equivalent registered typed error;
   - never overwrite conflicting bytes.

4. **Both inconsistent**
   - fail closed;
   - never select one object as authoritative by pathname or modification time.

5. Before success, prove:
   ```text
   namespace.store_namespace_id == genesis.store_namespace_id
   namespace.authority_genesis_id == genesis.record_id
   requested namespace class == persisted namespace class
   ```

6. Relocation must preserve semantic namespace identity. An absolute path must not enter `store_namespace_id`.

**Tests first**

Extend:

```text
tests/agents/ifvg_search/test_store_namespace.py
```

Required tests:

```text
test_concurrent_identical_initializers_publish_one_coherent_pair
test_concurrent_divergent_initializers_one_wins_other_refuses
test_crash_after_genesis_before_namespace_recovers_exactly
test_crash_after_namespace_before_genesis_recovers_exactly
test_namespace_head_mismatch_never_returns_success
test_relocated_store_retains_namespace_identity
```

Use temporary directories only.

---

## 5. Workstream B — Public source-kind boundary

**Primary files**

```text
src/alpha_lab/agents/data_infra/ifvg/search/trading_calendar.py
src/alpha_lab/agents/data_infra/ifvg/search/verification_window.py
src/alpha_lab/agents/data_infra/ifvg/search/seed_production.py
src/alpha_lab/agents/data_infra/ifvg/search/bounded_verification.py
scripts/ifvg_verification_window_shortlist.py
tests/agents/ifvg_search/test_trading_calendar.py
tests/agents/ifvg_search/test_verification_window.py
tests/agents/ifvg_search/test_seed_production.py
tests/agents/ifvg_search/test_bounded_verification.py
```

### 5.1 Required public contract

The public source kind is exactly:

```python
Literal["mbp1", "trades", "legacy_verified_replay_source"]
```

`"mbp10"` must not be a public contract value.

A historical physical file may still be named `mbp10.parquet`. Map that physical source to:

```text
source_kind = legacy_verified_replay_source
```

Keep its truthful physical provenance in an internal descriptor, for example:

```text
physical_filename
physical_schema_era_id
physical_content_sha256
physical_partition_key
```

That descriptor must not imply that MBP-10 features, live depth, search controls, model features, or UI capabilities are supported.

### 5.2 Identity and compatibility

- Re-mint only source-inventory, verification-window, seed-authorization, or verification identities whose semantic payload previously serialized `"mbp10"`.
- Do not mutate old artifacts.
- Do not rename physical source files.
- Do not alter `order_flow_depth_policy="mbp1_only_v1"`.
- The mapping is deterministic and content-preserving.

### 5.3 Tests first

Add or update tests proving:

```text
physical mbp10-era file -> public legacy_verified_replay_source
public SourceKind refuses "mbp10"
verification window/report JSON never emits public source_kind="mbp10"
seed and bounded-verification contracts use the opaque legacy value
MBP-1 feature/source contracts remain unchanged
```

Add one focused public-surface scan. It may permit the literal `mbp10` only in the private physical-file resolver and historical migration fixtures. It must prohibit it from serialized public contracts, registries, providers, study cells, feature bundles, live contracts, and model feature names.

---

## 6. Workstream C — Regime assignment and fold-feature integrity

### 6.1 Preserve invalid OOS provenance

**Primary files**

```text
src/alpha_lab/agents/data_infra/ifvg/ml/regime_oos_assignment.py
src/alpha_lab/agents/data_infra/ifvg/ml/regime_contracts.py
src/alpha_lab/agents/data_infra/ifvg/ml/regime_store.py
tests/agents/data_infra/ifvg/test_regime_oos_assignment.py
tests/agents/data_infra/ifvg/test_regime_assignment_evidence.py
tests/agents/data_infra/ifvg/test_r61_fix_review_fixes.py
```

Build the descriptive assignment index from **all test-partition rows**, not only valid rows.

Required three-way semantics:

```text
A. valid OOS assignment
   retain fit/fold/partition/protocol and assignment outputs

B. invalid OOS assignment
   retain fit/fold/partition/protocol and original typed missing reason
   null only cluster/distance/margin outputs

C. no OOS test row exists
   reason = no_oos_assignment
   no applicable fit/fold linkage exists
```

`no_oos_assignment` is reserved for case C.

Apply the same provenance rule when projecting assignments to executed trades. An invalid assignment must not lose its known fit and fold merely because the trade-facing output is null.

### 6.2 Strict validation before conversion

In `regime_contracts.py`, validate native values before pandas or Arrow canonicalization.

Required behavior:

- Boolean fields accept only actual booleans.
- Integral fields accept only true integral values; `bool` is not accepted as an integer.
- Fractional values such as `1.5` are refused rather than truncated.
- Numeric strings are refused unless the frozen contract explicitly declares strings.
- Float fields reject infinity and invalid objects.
- Required IDs are validated as non-null and non-empty before any `.astype(str)`.
- Sentinel-like strings created from missing objects—`"None"`, `"nan"`, `"<NA>"`—are not accepted as identifiers.
- `errors="coerce"` must not convert malformed evidence into lawful missingness.

Only after validation may the implementation produce canonical Arrow/pandas values.

### 6.3 Descriptive-assignment invariants

For `valid=true`, require:

```text
candidate/row ID
regime fit ID
fold index
partition
resolved protocol
fit-local cluster ID
canonical reporting cluster ID
complete distance vector of registered k
assigned distance
assignment margin
missing_reason is null
```

For an invalid row that came from an applicable fit, require:

```text
candidate/row ID
regime fit ID
fold index
partition
resolved protocol
registered missing reason
```

and require all assignment outputs to be null.

For true no-OOS coverage, require the specific no-coverage reason and no fabricated fit/fold.

### 6.4 Fold-feature invariants

**Primary files**

```text
src/alpha_lab/agents/data_infra/ifvg/ml/regime_fold_features.py
src/alpha_lab/agents/data_infra/ifvg/ml/regime_contracts.py
tests/agents/data_infra/ifvg/test_regime_fold_features.py
tests/agents/data_infra/ifvg/test_regime_fold_feature_evidence.py
```

Every fold-feature row must retain its reconciliation spine:

```text
row/candidate ID
fold index
partition
applicable regime fit ID
resolved protocol
source assignment-table hash
```

When source assignment is invalid, model outputs are null and the exact typed source reason is retained. Do not collapse a known invalid assignment into generic absence.

### 6.5 Schema identity and candidate-set equality

- `RegimeOosAssignmentPayload.assignment_schema_hash` must equal the registered OOS schema hash.
- The saver must decode and validate the actual Arrow schema and row invariants before publication.
- The loader repeats those checks.
- The candidate-as-of input candidate set and assignment output candidate set must be exactly equal:
  ```text
  same IDs
  same count
  no duplicates
  no extras
  no omissions
  ```
- Duplicate candidate IDs are a typed failure.

### 6.6 Tests first

Required adversarial cases:

```text
valid OOS row remains valid with exact provenance
invalid OOS row retains fit/fold/partition/reason
candidate with no test row receives only no_oos_assignment
trade projection retains invalid assignment provenance
"False" cannot serialize as true
fractional fold index is refused
null ID cannot become "None" or "nan"
wrong registered schema hash is refused at save
tampered Arrow schema is refused at reload
candidate-as-of and assignment ID-set mismatch is refused
duplicate candidate IDs are refused
```

---

## 7. Workstream D — Manifest, label, and persisted-report boundaries

### 7.1 Central manifest-entry validator

**Primary file**

```text
src/alpha_lab/agents/data_infra/ifvg/search/store.py
```

Add one shared validator used by every sidecar probe and load path.

Every manifest artifact entry must be:

```text
a mapping/object
with a non-empty relative path
with a valid lowercase 64-hex SHA-256
with required byte/row/schema fields where the store contract requires them
```

Reject:

```text
absolute paths
drive-qualified paths
empty/dot paths
.. traversal
normalized-path collisions
duplicate paths
symlink escape outside the artifact directory
manifest/envelope reserved names
non-mapping entries
missing required keys
invalid hash strings
```

Resolve and compare canonical paths before opening a sidecar. Every malformed-manifest condition must become a registered typed `SidecarLoadError` or store error—never `AttributeError`, `KeyError`, or silent absence.

Distinguish:

```text
store entry genuinely absent
invalid store locator/name/ID
malformed manifest
sidecar lawfully not declared
declared sidecar missing
declared sidecar hash-invalid
```

### 7.2 Exact label identity

**Primary files**

```text
src/alpha_lab/agents/data_infra/ifvg/ml/comparison_rows.py
tests/agents/data_infra/ifvg/test_label_identity.py
```

- Reject duplicate candidate IDs before label-content hashing.
- Persisted controlled studies require the exact immutable label-artifact identity.
- A convenience helper without an exact label artifact may return an ephemeral in-memory result only; it must not publish a persisted controlled-study artifact.
- Every label/economic field consumed by a persisted study remains bound through the exact label artifact.

### 7.3 Executed-trade evidence for stratification

Inspect the current `ChildStratificationInputs` and persistence service.

Required behavior:

- every persisted stratified report must reference and verified-load the exact `executed_trade_table_id`;
- a caller-provided frame without that ID may be used only by an explicitly ephemeral/non-persisting helper;
- the report identity binds the verified executed-trade-table artifact;
- tampered or absent declared trade tables fail closed rather than being silently skipped.

### 7.4 Tests first

Extend:

```text
tests/agents/ifvg_search/test_store_sidecar_probe.py
tests/agents/ifvg_search/test_pipeline_evidence_integrity.py
tests/agents/data_infra/ifvg/test_label_identity.py
tests/agents/data_infra/ifvg/test_regime_stratification_evidence.py
```

Include manifest path traversal, symlink escape, duplicate paths, malformed entry type, bad hash, reserved name, exact label requirement, duplicate label candidate, and missing/tampered trade-table tests.

---

## 8. Workstream E — Central seed canonicalization

**Primary files**

```text
src/alpha_lab/agents/data_infra/ifvg/search/child_replay.py
src/alpha_lab/agents/data_infra/ifvg/search/seed_production.py
tests/agents/ifvg_search/test_seed_production.py
tests/agents/ifvg_search/test_child_audit_companion.py
```

### 8.1 One unavoidable canonicalization seam

Move or delegate recursive datetime canonicalization into the central seed persistence path used by:

```text
save_seed_snapshot
seed-production runner
any direct seed-save caller
```

A caller must not be able to bypass canonicalization by calling `save_seed_snapshot` directly.

### 8.2 Datetime rule

For every nested value:

```text
aware datetime already exactly UTC:
    preserve the instant in canonical UTC representation

aware datetime in any other timezone:
    value.astimezone(UTC)

naive datetime:
    follow the existing seed contract exactly;
    never infer the machine's local timezone
```

Cover recursively:

```text
dataclasses
Pydantic models where applicable
tuples
lists
mappings
nested seed state
```

Fix the current `isinstance(value.tzinfo, timezone)` shortcut; a stdlib UTC−05:00 timezone is not UTC.

### 8.3 Compatibility

- Existing already-canonical UTC seed bytes and golden IDs must remain unchanged.
- Two timezone representations of the same instant must produce identical canonical bytes and seed identity.
- Non-UTC conversion must preserve the represented instant.
- The function must not mutate caller-owned objects.

### 8.4 Tests first

Required cases:

```text
pytz UTC and non-UTC
zoneinfo UTC and non-UTC
datetime.timezone fixed positive and negative offsets
nested dataclass/list/tuple/map
direct save_seed_snapshot
seed-production runner
same instant in different zones -> same artifact
existing canonical UTC golden -> unchanged
naive behavior follows the current frozen contract
```

---

## 9. Workstream F — Event-detail worst-shape capacity bound

**Primary files**

```text
src/alpha_lab/propsim/event_detail.py
src/alpha_lab/propsim/search_bridge.py
scripts/hardening_capacity_benchmark.py
tests/propsim/test_event_detail_streaming.py
```

The external regime-summary aggregation has already passed hardening. Do not redesign it.

### 9.1 Required deterministic row bound

Add a versioned, identity-bearing bound:

```text
max_rows_per_partition = 50_000
```

or the exact existing measured row-group size if the current registered policy names another value.

The writer must flush by row count even when that splits one path block. Do not require all events for 250 paths to coexist in Python lists.

Recommended deterministic partition key:

```text
path_block_id
partition_ordinal_within_path_block
```

Manifest evidence must include:

```text
path block
partition ordinal
row count
first event ordering key
last event ordering key
file bytes
file SHA-256
schema hash
```

Rules:

- preserve exact chronological/event ordering;
- do not split, drop, or deduplicate events;
- do not truncate when total budget is exceeded;
- total row/byte limit still fails before final publication;
- repeat input produces byte-identical partitioning and manifest identity;
- partition naming/order is independent of iterator chunking.

If reconnaissance proves that a hard `max_events_per_path` is already a frozen business contract, stop and ask before substituting it for row-based flushing. Do not silently lower supported event density.

### 9.2 Unit tests

Add fixtures for:

```text
one path larger than one partition
one path block larger than one partition
highly skewed event counts
chunking-independent deterministic output
exact reconstruction across split partitions
hard total-row refusal without partial publication
cleanup after failure
```

### 9.3 Focused numerical benchmark

Run one focused capacity gate after implementation—no repeated tuning loop.

Benchmark:

```text
250,000 rows
500,000 rows
1,000,000 rows
```

Include at least:

```text
normal shape: 200 events/path
skewed shape: one or few very long paths
dense shape: 250 paths whose combined rows exceed one partition many times
```

Acceptance:

```text
1M-row peak RSS increase               <= 1.5 GiB
1M-row Python allocation peak          <= 512 MiB
1M-row wall time                       <= 300 s
500k->1M per-row RSS slope              within HARDENING_CAPACITY_POLICY_V1
registered 10M-row projection          <= min(6 GiB, 50% of measured available RAM)
serialized 10M-row projection          <= 2 GiB
repeat output                          byte-identical
maximum resident writer batch          <= max_rows_per_partition
```

A shape-conditional projection is not sufficient. Every input admitted by the revised policy must obey the hard resident-row bound.

If the implementation cannot satisfy the existing registered ceiling, stop and ask the owner. Do not lower it autonomously.

---

## 10. Workstream G — Complete supersession-chain authority proof

**Primary files to inspect**

```text
src/alpha_lab/agents/data_infra/ifvg/search/supersession_chain.py
src/alpha_lab/agents/data_infra/ifvg/search/owner_decisions.py
src/alpha_lab/agents/data_infra/ifvg/search/authorization.py
src/alpha_lab/agents/data_infra/ifvg/search/seed_production.py
src/alpha_lab/agents/data_infra/ifvg/search/verification.py
src/alpha_lab/agents/data_infra/ifvg/search/bounded_verification.py
src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py
src/alpha_lab/agents/data_infra/ifvg/search/executors.py
src/alpha_lab/agents/data_infra/ifvg/features/mbp1_coverage_diagnostic.py
src/alpha_lab/agents/data_infra/ifvg/ml/regime_store.py
```

### 10.1 Required central proof

A current head witness alone is insufficient. Before any real authority is accepted, verify the complete chain from genesis to current head and verified-load every replacement owner-decision artifact referenced by that chain.

Provide one shared function or one demonstrably shared call path equivalent to:

```text
verify_complete_owner_authority_chain(
    store_namespace_id,
    expected_head_witness,
    owner_decision_store
)
```

It must prove:

```text
head record and digest are current
every chain record exists and hashes correctly
every superseded decision exists
every replacement decision exists and verifies
replacement points to the expected predecessor
namespace and effective-time rules are lawful
no divergent transition exists
```

### 10.2 Required authority seams

Parameterize integration tests across:

```text
OwnerAuthorizationBundle freeze/load
SeedProductionAuthorizationRef freeze/load
VerificationAuthorizationRef freeze/load
MBP-1 real diagnostic authorization
Regime promotion
Pipeline freeze/launch
Publication/activation gate
```

When the head advances but the replacement owner-decision artifact is missing, malformed, hash-invalid, or predecessor-inconsistent, every seam must refuse before data-path construction or publication.

If current code already performs the complete proof at a seam, retain it and add the integration evidence; do not add redundant layers.

---

## 11. Identity, migration, and compatibility rules

### 11.1 Expected re-mints

New identities are expected only where semantics change:

```text
source inventories/windows that serialized public "mbp10"
seed artifacts created from previously noncanonical timezone representations
event-detail artifacts under the new partition-bound policy/version
regime OOS/fold artifacts whose prior invalid-row provenance was incorrect
```

Document each changed identity family and reason in the completion report.

### 11.2 Identities that must remain unchanged

Prove no unintended change to:

```text
Strategy-Core commit/pin
core strategy replay identities unrelated to corrected public source descriptors
fixed M0–M3 identities
accepted v2/v3 artifacts
R5B formula identities
model protocol parameters
KMeans fit numerical identities
prop-firm rule contracts
S11 blocked reason
```

### 11.3 Existing artifacts

- Never rewrite or add files inside an existing content-addressed artifact directory.
- Corrected outputs use new versioned contracts/IDs.
- Old artifacts remain readable only where their original contract permits.
- No migration scans of real artifact roots are part of this implementation.
- No catalog activation occurs.

---

## 12. Tests and release acceptance

### 12.1 Targeted test sequence

Run targeted tests by workstream while implementing:

```text
A:
test_owner_decision_lock.py
test_store_namespace.py

B:
test_trading_calendar.py
test_verification_window.py
test_seed_production.py
test_bounded_verification.py

C/D:
test_regime_oos_assignment.py
test_regime_assignment_evidence.py
test_regime_fold_features.py
test_regime_fold_feature_evidence.py
test_r61_fix_review_fixes.py
test_store_sidecar_probe.py
test_pipeline_evidence_integrity.py
test_label_identity.py
test_regime_stratification_evidence.py

E:
test_seed_production.py
direct seed-save tests

F:
test_event_detail_streaming.py
focused hardening capacity benchmark

G:
test_supersession_chain.py
test_authorization.py
test_pipeline_authority_seams.py
test_bounded_verification.py
test_regime_store.py
test_mbp1_coverage_evidence.py
```

Tests must be behavioral. Import/collection failures do not count as sufficient red-first evidence for the central cases.

### 12.2 Required new acceptance assertions

| ID | Assertion |
|---|---|
| HB-FIX-01 | Two stale reclaimers cannot delete the newly acquired live lock. |
| HB-FIX-02 | Namespace initialization is coherent and deterministic under concurrent identical and divergent requests. |
| HB-FIX-03 | Public serialized source kinds contain no `mbp10`; legacy physical files resolve to `legacy_verified_replay_source`. |
| HB-FIX-04 | Invalid OOS assignments retain exact fit/fold/partition/protocol/reason provenance. |
| HB-FIX-05 | Malformed bool/integer/ID evidence is refused before coercion. |
| HB-FIX-06 | Invalid fold-feature rows retain the complete reconciliation spine. |
| HB-FIX-07 | OOS schema identity and candidate populations are exact and verified at save and load. |
| HB-FIX-08 | Manifest traversal, symlink escape, duplicate/reserved paths, malformed entries, and invalid hashes fail with typed errors. |
| HB-FIX-09 | Persisted studies require exact label and executed-trade artifact identities. |
| HB-FIX-10 | Direct and runner seed saves canonicalize every aware timezone to UTC without changing canonical UTC goldens. |
| HB-FIX-11 | Event-detail memory is bounded by the registered per-partition row limit for normal and skewed lawful shapes. |
| HB-FIX-12 | Every real authority seam refuses an incomplete or corrupt supersession replacement chain. |
| HB-FIX-13 | M0–M3, Strategy-Core, Trade-Lab, immutable artifacts, S11, and MBP-1 offline-only boundaries remain unchanged. |

### 12.3 Final full gate

After targeted tests and the one focused reviewer pass:

1. Run the complete repository suite once in the normal environment:
   ```text
   python -m pytest -q
   ```
2. Run it once with provider credentials cleared.
3. Both runs:
   ```text
   0 failed
   warnings-as-errors
   no new unallowlisted warning
   ```
4. Run:
   ```text
   python -m ruff check src tests scripts
   git diff --check
   ```
5. Run the fixed identity/golden suites.
6. Run the focused capacity benchmark once.
7. Prove:
   ```text
   protected access counters = 0
   sealed access counters = 0
   no new real-data artifact
   no seed-production run
   no bounded real verification run
   no full pipeline run
   no catalog activation
   ```
8. Verify surviving pre-existing user-owned hunks are byte-identical to the baseline capture.
9. Verify exact reuse on a representative synthetic pipeline after the corrected identities are established.

A browser or UI gate is not part of this backend release.

---

## 13. Focused final review

After all targeted tests pass, invoke at most the two reviewers permitted by §3.

The main agent must consolidate the result into one short file:

```text
FOCUSED_REVIEW.md
```

It must contain:

```text
reviewed commit/diff
fixed checklist item
finding
severity
disposition
test proving resolution
```

No additional adversarial round is allowed. If a reviewer discovers a genuine in-scope blocker, fix it once and rerun only the affected targeted tests plus the final gate. Outside-scope findings go into a maximum-five-item carry-forward section.

---

## 14. Commit and evidence package

Use one focused release commit or at most two logically separated commits:

```text
1. HARDENING-BACKEND-FIX: close authority and immutable-evidence gaps
2. HARDENING-BACKEND-FIX: bound event-detail capacity and finalize evidence
```

Do not push or merge.

Create only the necessary evidence under:

```text
implementation-progress\HARDENING-BACKEND-FIX\
```

Required:

```text
IMPLEMENTATION_PLAN.md              # this plan, copied verbatim
COMPLETION_REPORT.md                # one consolidated narrative
FOCUSED_REVIEW.md                   # capped targeted review
HARDENING-BACKEND-FIX.patch
HARDENING-BACKEND-FIX.patch.sha256
HARDENING-BACKEND-FIX.bundle
HARDENING-BACKEND-FIX.bundle.sha256
_targeted_pytest.txt
_final_pytest.txt
_final_pytest_keys_cleared.txt
_ruff_and_diffcheck.txt
_capacity_benchmark.txt
CAPACITY_BENCHMARKS.json
```

Do not generate a large family of overlapping Markdown reports. `COMPLETION_REPORT.md` must cover:

- baseline and final commit;
- exact files changed;
- each HB-FIX acceptance item;
- tests and counts;
- capacity numbers;
- identity re-mints and preserved goldens;
- reviewer findings and resolutions;
- access/scope proof;
- remaining owner actions;
- final UI transition decision.

---

## 15. Stop conditions

Stop and report instead of improvising when:

- the current branch cannot be reconciled with the expected parent without overwriting user work;
- a required correction needs Strategy-Core, Trade-Lab, M0–M3, or UI changes;
- a capacity fix requires lowering the 10M/2-GiB policy rather than bounding implementation;
- an existing immutable artifact would need mutation;
- a source-kind correction requires renaming or rereading real source files;
- complete authority-chain validation changes owner-decision semantics rather than enforcing the existing chain;
- any test requires June 11, the sealed range, or an unapproved real-data path;
- a new dependency appears necessary;
- the reviewer proposes an unrelated architecture expansion.

---

## 16. Final completion criteria and UI handoff

The release is complete only when all of the following are true:

```text
[ ] stale-lock reclamation is serialized and race-tested
[ ] namespace initialization is atomic/recoverable and race-tested
[ ] public source kind is MBP-1/trades/legacy only
[ ] invalid OOS and fold-feature rows retain exact known provenance
[ ] assignment values are validated before conversion
[ ] manifest entries and paths are fully fail-closed
[ ] label/trade evidence is exact for every persisted study
[ ] seed datetime canonicalization is unavoidable and timezone-correct
[ ] event-detail resident memory is hard-bounded for every lawful input
[ ] every real authority seam verifies the complete supersession chain
[ ] full suite passes twice with warnings-as-errors
[ ] Ruff and diff checks pass
[ ] required goldens remain unchanged
[ ] protected/sealed counters remain zero
[ ] no real run or immutable-artifact mutation occurred
[ ] patch, bundle, hashes, focused review, and completion report verify
```

When all boxes pass, the completion report must state:

```text
backend_dev_complete_for_ui = true
ui_implementation_may_begin = true
owner_verification_still_required = true
formal_release_acceptance = blocked_by_R1_VerificationAuthorizationRef
```

At that point the separate UI/UX implementation branch may rebase onto the finalized backend contracts and begin. The owner seed-production and bounded-verification workflow remains a later action performed through the corrected UI and existing backend authorization contracts.
