# R6.1-FIX — Deviations and Scoping Notes

Written before the adversarial round (2026-09-01) against
`../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md` revision 3,
Phase 1 (§3.1–§3.10); reconciled after the round
(`ADVERSARIAL_REVIEW_RESOLUTION.md`): entries DEV-R6.1-FIX-13 onward record
the fix round and the amendments below are marked in place. Every entry
states what the CODE does.

## DEV-R6.1-FIX-1 — Sidecar failure vocabulary: three typed reasons beyond the plan's list

Plan §3.8 names seven states (`sidecar_not_produced_for_path` + six stage
failures). `search/store.py` registers `SIDECAR_LOAD_FAILURE_REASONS` with
NINE failure reasons: the plan's `sidecar_missing_but_manifest_declares_it`,
`sidecar_hash_mismatch`, `manifest_hash_mismatch`,
`envelope_identity_mismatch`, `malformed_sidecar`, `unexpected_io_error`, plus
`store_entry_missing` (the entry DIRECTORY does not exist),
`manifest_missing_for_existing_entry` (an existing entry directory without
its manifest — corruption, never absence; added after the round, review
B-02; `has_envelope` raises it too) and `malformed_manifest` (the manifest is
not JSON / lacks `artifacts`). All are stage failures, never lawful absence;
`SidecarLoadError` refuses an unregistered reason. `has_sidecar` is false ONLY
for the manifest-proven `sidecar_not_produced_for_path` state. Consequence for
the prior-stage recoveries (`_prior_stage_vectors`, `_prior_stage_simulations`,
the lineage maps, `_delivered_by_s09c`; the S14 prior-reports recovery was
deleted after the round — DEV-R6.1-FIX-19): they no
longer consult `has_envelope` first — a state file that names a stage result
whose entry is gone is `store_entry_missing` (typed failure), where R6.1
treated it as absence. DECISIONS_TAKEN #108.

## DEV-R6.1-FIX-2 — The executed-trade table store has no "not produced" manifest; absence is a derived-identity fact

A child whose replay predates R6.1-FIX (or whose fresh completion failed
before the table was published) has NO entry under `executed_trade_tables`,
so there is no verified manifest to prove non-production. Because the table
identity DERIVES from the core replay (`executed_trade_table_id_for`), S02 /
S14 exact-probe that id without listing: `probe_executed_trade_table` maps
`store_entry_missing` — raised ONLY when the entry directory does not exist —
to the typed state `"absent"`; every other probe state (an existing entry
without its manifest, a declared-but-missing sidecar, a hash / manifest /
identity mismatch, a malformed envelope) raises `SidecarLoadError` — classified
by exception type at the detection point, never by message text (reviews
B-02 / B-08) — and is a typed child failure in S02 (`FailureReason.REPLAY`,
sanitized). An absent table is the
typed S14 skip `executed_trade_table_unavailable` (plan §3.7: "a
`children_skipped` record — never silent omission"). DECISIONS_TAKEN #107.

## DEV-R6.1-FIX-3 — S02 reuse semantics amended relative to the R6.1 S4 disposition (#98)

R6.1 adopted a reused child's re-derived tables ONLY when they reproduced the
persisted costed evaluation of THIS cost policy, so a new cost policy over an
immutable replay was "reproduction unverifiable". With the persisted
executed-trade table (§3.7) the reproduction is verified byte-for-byte
against the table under ANY cost policy, and a reused child's costed
evaluation for a cost policy that has none is published from the VERIFIED
persisted table (fresh compute over the verified bytes; zero replay when no
regime re-derivation is needed). "Reproduction unverifiable" now applies only
when neither a persisted table nor a persisted evaluation exists (the third
branch of `test_s02_reused_children_are_adopted_only_by_verified_reproduction`
parks the table to reproduce it). The immutable core replay is never
rewritten; a re-derivation whose bytes differ from the persisted table fails
the child closed. DECISIONS_TAKEN #98 (amended pointer) / #107.

## DEV-R6.1-FIX-4 — F-10D: the enum `.value` strings originated in a test's `model_copy(update=…)`, not in the materializer's payload construction

Recon F-10D located the warning in `mbp1_feature_materializer.py →
mbp1_source_contract.py`. On inspection the materializer's only `.value` uses
write STRING columns of the stage-window evidence table (schema-typed
strings — lawful); the Pydantic serialization warnings came from
`test_mbp1_materializer.py` updating enum-typed `Mbp1FeatureWindowSpec`
fields with raw strings through `model_copy(update=…)`, which bypasses
validation. The fix is structural: `FrozenContract.model_copy` refuses a raw
value for an enum-typed field (`TypeError`), the test passes enum members,
and `test_mbp1_window_specs_serialize_without_warnings` runs under
`warnings.simplefilter("error")`. The remaining project-owned warnings of the
R6.1 baseline belong to Phase 2 (F-18) and are counted in `TEST_RESULTS.md`.
DECISIONS_TAKEN #109.

## DEV-R6.1-FIX-5 — Two builder-internal guards became `RuntimeError`, the five pipeline seam checks `PipelineWiringError`

Plan §3.9 allows `PipelineWiringError`, `SearchStoreError`, or a precise
value/contract error. The five `assert context.wiring.<seam> is not None`
checks in `search/pipeline.py` (S03 audit builder, S04 chart builder, S05
candidate view source, S07 label builder, the MBP-1 evidence seam) raise the
new typed `PipelineWiringError(RuntimeError)`; the two `assert` statements
inside `build_regime_fold_features` (panel as-of / panel inputs, already
refused earlier by typed `ValueError`s) became a `RuntimeError` guarded with
`pragma: no cover`. No `assert` remains in any touched `src` module
(`grep`-verified). DECISIONS_TAKEN #109.

## DEV-R6.1-FIX-6 — Downstream stages are PENDING for a halted attempt (not in the plan text)

While making S15 record reload failures (§3.8), the tamper tests showed that a
halted attempt left the LATER planned stages carrying the prior attempt's
terminal statuses in the state file (stale "completed" entries after a
typed failure). `run_pipeline` now calls `_mark_downstream_not_run` on every
halt: each later planned stage is `PENDING` with "not run: an earlier stage
failed in this attempt" (S11 keeps its registered blocked state). Asserted by
`test_prior_attempt_sidecar_tamper_fails_the_stage_closed` and
`test_lineage_sidecar_tamper_fails_closed_before_the_delta_stage`.
Amended after the round (review B-04): the halt / cancel also resets the
state file's publication block (`_reset_publication_block`) and
`activate_pipeline_result` re-derives the publication gates from the LATEST
attempt before activating anything. DECISIONS_TAKEN #108 / #114.

## DEV-R6.1-FIX-7 — Identities re-minted (synthetic only) and the identities proven unchanged

Re-minted, as the plan allows (§3.1 "affected descriptive-assignment and
stratified-report identities", §3.2 "fold-feature, bundle-view, bundle-model,
and controlled-study identities"): the descriptive OOS assignment
(`regime_oos_assignment_v2`, `regime_fit_assignment_refs`,
`resolved_cluster_count`, the full consulted hash); the fold-feature artifact
(`FoldFitRef` sidecar/schema hashes); every stratified report
(`RegimeAssignmentEvidenceRef.assignment_table_sha256` /
`assignment_schema_hash`, `net_r_accounting`, `executed_trade_table_id`,
`source_metric_refs`); the S07 label artifact id
(`label_artifact_consumed_columns_v2`) and with it the bundle-path ladder /
controlled-study identities that bind it; the controlled-study payload
(`label_identity_source`); the S02 / S07 / S09 / S14 / S15 stage-result ids
(children rows carry `executed_trade_table_id` / `_sha256`; S14 records
`children_evidence` / typed `children_skipped`; S15 records
`reload_failures`). PRESERVED and golden-tested
(`tests/agents/ifvg_search/test_r61_fix_goldens.py`, `_goldens_pytest.txt`):
`resolved_regime_protocol_id`, the R6 golden fold-0 `regime_fit_id`, the
frozen M0 CatBoost `resolved_hash`, `core_replay_id`,
`account_simulation_id`, `feature_block_registry_hash`, the `B0_CORE` bundle
id (values in `PRE_R6_1_FIX_BASELINE.md`). No persisted real artifact exists
outside tmp roots; nothing immutable was rewritten. DECISIONS_TAKEN #110.
Amended after the round: the fix round re-minted the OOS-assignment ids once
more (`candidate_as_of_stage`), the `cohort_descriptive` report ids
(`executed_trade_table_artifact_sha256`), the pipeline result ids
(`reload_failure_reasons`) and the bundle-path ladder ids of HELPER runs
(review RA-05); the golden set was re-run unchanged after the round.

## DEV-R6.1-FIX-8 — The exact label artifact is S07's content-derived stage output, not a new store

Plan §3.6 requires the persisted study to bind "the exact label artifact".
S07 mints `label_artifact_content_id(label_policy_id, labeled)` over the
registered policy and EVERY consumed column (`LABEL_CONSUMED_COLUMNS`:
candidate id, setup id, trading day, entry / resolution instants and
availability flags, binary target, gross / net R) and records it as the S07
output id in the immutable stage result; no separate `label_artifacts` store
was added (the labeled frame is re-derivable from the verified candidate view
under the pinned policy). `ControlledFeatureStudyPayload.label_artifact_id`
is mandatory; `label_identity_source="content_hash_unpersisted"` (the helper
form with `label_policy_id=None`) refuses `save_controlled_feature_study`
with `PermissionError`. DECISIONS_TAKEN #106.

## DEV-R6.1-FIX-9 — `regime_fit_ids` retained as the validated projection of the refs

Plan §3.1 adds `regime_fit_assignment_refs`; the existing `regime_fit_ids`
field stays on `RegimeOosAssignmentPayload` as the DERIVED projection
(validated to equal the refs' ordered ids) so every consumer of the id tuple
(the stratification service, the panel assigner, the UI) is unchanged.

## DEV-R6.1-FIX-10 — `CohortDescriptiveBody` concentration fields mirror the accounting

`works_only_in_regime` stays a tuple (empty unless the accounting's claim is
TRUE, then exactly the one regime id) and `top_regime_abs_net_r_share`
mirrors `net_r_accounting.top_regime_abs_net_r_share`; the R6.1 "at least
two reported regimes" precondition is gone (the raw accounting decides over
ALL valid assigned trades). The per-trade basis is `per_trade_net_r` =
`(realized − cost) / risk` of the normalizer's frame — exactly the vector
`compute_strategy_metrics` averages into `net_expectancy_r`. DECISIONS_TAKEN #105.

## DEV-R6.1-FIX-11 — Deferred cleanups (plan §2, "not a gate") remain open

Not done by design: declaring `threadpoolctl>=3.1` (still a lazy import,
DEV-R6.1-13); registering `candidate_not_in_assignment` in the reason
vocabulary (still a literal in `regime_assignment_sources.py`); removing the
obsolete in-memory assignment frame after the verified-source migration
(`RegimeProtocolRun.assignments` is still produced by the kernel and consumed
by `persist_regime_fit`; every DOWNSTREAM artifact reads verified bytes);
binding `authorized_session_span_ns` to the session scheme; comparing
`stratified_frontier.children_without_strata` with `gates_passed`.

## DEV-R6.1-FIX-12 — One script line changed for the store contract; no presentation change

`scripts/ifvg_regime_panels.py` unpacks `VerifiedFitAssignments` (the new
return type of `load_regime_fit_assignments`) — a type adaptation of the
Assignment view's loader call, not a UI change (plan scope separation: the
UI/UX redesign is a separate plan).

## Amended after the adversarial round (2026-09-01/02)

The dispositions of every finding are in `ADVERSARIAL_REVIEW_RESOLUTION.md`;
the entries below record the deviations and scoping notes the fix round
added (DEV-R6.1-FIX-13 onward). Entries DEV-R6.1-FIX-1/-2/-3/-4/-7 above were
re-read against the post-fix code and stand as written except where an
entry below supersedes a sentence (stated explicitly).

## DEV-R6.1-FIX-13 — `works_only_in_regime`: the plan's predicate verbatim, `assigned_regime_count` persisted (review RA-03)

The plan §3.5 predicate is kept verbatim — a SINGLE assigned regime with
positive net R yields a vacuous TRUE claim — and `RegimeNetRAccounting`
persists `assigned_regime_count` so the vacuity is visible rather than
hidden. NULL (`incomplete_assignment_accounting`) is reserved for the case
where unassigned trades prevent a claim the assigned side supports; a claim
the assigned side refutes is FALSE with no reason even when unassigned
trades exist. `RegimeNetRAccounting._coherent` recomputes every derived
value from `net_r_by_regime` (mass, total, shares, top share, signed
fractions, the zero-denominator reasons, the claim and its regime id) with
`math.isclose(rel_tol=1e-9, abs_tol=1e-9)`; `CohortDescriptiveBody`
requires the regime strata to be exactly the accounting's regimes with
matching trade counts (review RA-02). DECISIONS_TAKEN #111.

## DEV-R6.1-FIX-14 — The stratification service re-verifies the caller's frame against the persisted table (review RA-01)

`build_regime_stratified_reports` no longer binds `executed_trade_table_id`
by assertion: for every child naming a table it exact-loads the artifact,
refuses a table of another core replay, refuses a caller frame whose
projection bytes (`executed_trade_table_bytes(project_executed_trades(...))`)
differ from the artifact's bytes, and consumes the LOADED frame for every
class. `CohortDescriptiveBody.executed_trade_table_artifact_sha256` (the
artifact's projection-bytes hash) is bound beside the normalized-frame hash
(id and artifact hash together or neither). A reordered frame with scratch
columns reproduces; a different row set is refused. Cohort report ids
re-mint (synthetic only). DECISIONS_TAKEN #111.

## DEV-R6.1-FIX-15 — Assignment-row invariants: linkage on invalid rows, arithmetic self-consistency on valid rows (review RA-06)

`validate_assignment_rows` now requires the linkage key on EVERY row
(`row_id` for the `fit` / `model_facing` kinds; `candidate_id` for the
`descriptive` kind, falling back to `row_id` when a fit-schema frame is
validated under the descriptive rule), a lawful non-null `partition`, a
64-hex non-null `regime_fit_id` and a non-negative non-null `fold_index` on
invalid rows (the all-null `_typed()` shape stays lawful), and on valid rows
`assigned_distance == distances[fold_local_cluster_id] == min(distances)`
and `assignment_margin == second_smallest − smallest ≥ 0` within
`_ASSIGNMENT_TOLERANCE = 1e-9` (the kernel's own arithmetic, copied by the
PIT and candidate rules). These checks fire in `_bound_assignments` BEFORE
the estimator-reproduction check, so a forged local id is now refused as an
arithmetic inconsistency (a different message than R6.1's; fail-closed
either way — `test_regime_store.py` accepts both). The loader additionally
requires the sidecar's `resolved_regime_protocol_id` column to equal the
envelope's. DECISIONS_TAKEN #112.

## DEV-R6.1-FIX-16 — `candidate_as_of_stage` on both grains; the as-of hash is strict (review RA-07)

`RegimeOosAssignmentPayload.candidate_as_of_stage: AvailabilityStage` is a
new REQUIRED field on BOTH grains (on the panel grain it is redundant with
`panel_context.candidate_as_of_stage` by design — validator-enforced
equality; on the candidate grain it records which anchor column the hashed
instants came from, so the same rows under a different stage mint a
different id). `candidate_as_of_source_hash` routes through `_as_of_ns`: an
unparseable NON-null anchor is a hard error on every grain, a null anchor
hashes as `None` deterministically. Every OOS-assignment id re-mints
(synthetic only; already re-minted this release). DECISIONS_TAKEN #112.

## DEV-R6.1-FIX-17 — Every helper path binds the FULL consumed-column label hash; the ladder run is stamped (review RA-05)

`run_supervised_ladder`, `run_catboost_bundle_model` and
`run_logistic_fold_models` default `label_artifact_id` to
`label_artifact_content_id(None, labeled)` — never the narrow
`(candidate_id, binary_target)` pair hash — and `SupervisedLadderRun`
carries `label_identity_source` (`label_artifact` when the caller passed an
exact id, `content_hash_unpersisted` otherwise; a caller may declare the
helper form explicitly, and any other value is refused).
`ControlledFeatureStudyPayload.label_identity_source` is REQUIRED (no
self-declared default). The ladder identity still binds the pair hash under
its `label_content_hash` key (unchanged since R5 — the frozen-tier ladder
ids do not move); only the bundle-path ladder ids of HELPER runs (no exact
id) re-mint; in-pipeline every ladder receives S07's exact id, so no
persisted pipeline identity moves. Two R6.1 CatBoost-bundle tests that
pinned the narrow default were updated. DECISIONS_TAKEN #113.

## DEV-R6.1-FIX-18 — The enum copy guard recognises field shapes (review RA-04)

`FrozenContract.model_copy` guards a scalar `Enum` / `Enum | None` field
(member required) and a homogeneous `tuple` / `list` / `set` / `frozenset`
of ONE enum class (every element a member; a string or a non-iterable is
refused); a union of two enums, a mapping and every non-enum annotation
pass through untouched. A structural test walks every registered identity
payload's sequence-of-enum field (`PipelineSemanticSpecPayload.stage_plan`
among the six the reviewer enumerated) and proves a lawful copy succeeds and
a tuple of raw strings is refused. DECISIONS_TAKEN #113.

## DEV-R6.1-FIX-19 — The S14 prior-reports recovery branch is gone; S15 reloads tables and reports (review B-03)

`regime_report_stage._prior_reports_record` and the recovery branch of
`build_reports` were deleted: with persisted executed-trade tables, an
`executed_trade_table_unavailable` child that a prior attempt reported is
store corruption, not lawful reuse, and the prior record could re-report a
child that failed in THIS attempt, discard this attempt's typed skips and
name a table the store no longer holds. S14 always builds from the present
children plus typed `children_skipped`; identical evidence re-mints identical
report ids, which the store reuses. `regime["children_evidence"]` is threaded
to S15 and `s15_regime_reload_failures` reloads
`executed_trade_tables/<id>` (re-checking the recorded sha256) and
`regime_stratified_reports/<id>` for every entry of the S14 record.
`_delivered_by_s09c` stays (tested on a tampered S09 record).
DECISIONS_TAKEN #114.

## DEV-R6.1-FIX-20 — Replay policy under stratified reporting: one verified reproduction per reused child (review B-05; supersedes the "zero replay" reading of §3.10 for stratified runs)

Zero replay holds for NON-stratified verified reuse (every reused child of
the double run carries `replay_invocations == 0`, asserted by
`test_second_attempt_with_different_workers_shares_every_semantic_identity`).
A run that requests stratified reporting re-derives every reused child
exactly ONCE as a verified reproduction against the persisted executed-trade
table — the projection bytes AND the raw core-table hash
(`source_core_table_hash`) must reproduce, else the child fails closed —
because S12/S13 consume raw tables the store does not hold (the DEV-R6.1-8
premise for the tables S14 reads is gone; the premise for the prop stages
remains). `test_second_attempt_reuses_every_regime_stage` asserts
`replay_invocations == 1` per reused child. The §3.10 gate evidence reads:
identical stage-result ids on the double run, zero replay for non-stratified
reuse, one verified reproduction per reused child for stratified runs
(`GATE_SUMMARY.md`). A load-only reuse of prop stages from persisted raw
tables is a hardening candidate, not part of this release.
DECISIONS_TAKEN #114.

## DEV-R6.1-FIX-21 — Reload-failure reasons live on the immutable pipeline result, not on an S15 sidecar (review B-09)

The plan §3.8 asks to "persist reload-failure reasons in the S15 record". They
are persisted as `PipelineResultPayload.reload_failure_reasons`
(`ImmutableMap`, default empty) — part of the content-addressed pipeline
result — rather than as an S15 stage sidecar: two failing attempts with
identical gate booleans but different reasons would share ONE S15
stage-result id with different sidecar bytes, which the immutable store
refuses (observed while implementing). Pipeline result ids re-mint (S15
output; synthetic only). The mutable state file keeps its `reload_failures`
copy for the Monitor. DECISIONS_TAKEN #114.

## DEV-R6.1-FIX-22 — Every costed evaluation derives from the persisted projection; publication moved inside per-child containment (review B-01)

S02 no longer evaluates a fresh child from its raw tables and a reused child
from the projection: after `save_executed_trade_table` the fresh and the
regime-reuse paths exact-load the table back
(`_persist_and_load_executed_trade_table`) and `_child_costed_evaluation`
computes / publishes every costed evaluation from the LOADED projection
frame, so `costed_evaluation_id → strategy_metrics.json` is
provenance-independent (the int-typed synthetic fixture serializes
`80000` from a raw frame and `80000.0` from the projection — the mechanical
root of the finding; the real v2 capture is float-typed and unaffected). The
metrics + lineage publication runs inside per-child containment: a store
refusal is a typed child failure (`FailureReason.INVARIANT`) and the other
children complete. S12/S13 keep consuming the raw `result.tables`.
DECISIONS_TAKEN #114.
