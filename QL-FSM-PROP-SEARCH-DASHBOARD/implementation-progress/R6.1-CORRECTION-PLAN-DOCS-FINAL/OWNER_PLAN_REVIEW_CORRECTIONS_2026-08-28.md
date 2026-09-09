# Owner Plan-Review Corrections for R6.1 (revision 1 → revision 2) — verbatim, 2026-08-28

**Document type:** Input record. The owner's review of `R6.1_IMPLEMENTATION_PLAN.md` revision 1.
Each item is applied in revision 2 (see the plan's §0 revision log for the section mapping).

---

Please make these corrections to the plan

## Issues that block approval

### 1. The proposed regime features are not fold-safe for supervised training

The most serious issue is in D7 and the feature_only workstream.

The plan proposes giving every candidate its one global OOS regime assignment and using those
values as supervised model features. That is appropriate for descriptive OOS stratification, but
not for training logistic or CatBoost models across multiple outer folds.

A training candidate in supervised fold k could receive a regime assignment created by another
regime fold whose training period includes information from after fold k's test interval. That
creates a potential future-data path. It also mixes cluster geometries from multiple KMeans fits
and leaves many training rows without assignments.

The design needs two separate artifacts:

- RegimeOosAssignmentArtifact — descriptive reporting only
- RegimeFoldFeatureArtifact — fold_index, candidate_id, partition=train|test, regime_fit_id

For each supervised fold: fit KMeans on that fold's training data; derive train features from
that fit; derive test features OOS from the same fit; fit logistic/CatBoost using those
fold-local features.

The global OOS assignment must not become the model-training feature source.

This is a scientific correctness blocker.

### 2. Panel-to-candidate assignment can carry a previous-day regime forward

The assignment rule at R6.1_IMPLEMENTATION_PLAN.md:244–260 searches for the latest completed
panel row before the candidate timestamp. It does not normatively require:
`panel.trading_day == candidate.trading_day`.

That means a candidate occurring during the next day's 12-bar warmup could inherit the previous
day's final regime. A current-day source gap could produce the same stale carry-forward.

The assignment contract must require: same trading day; panel row valid; panel close <= candidate
as-of; compatible OOS partition; elapsed time <= registered maximum staleness.

Recommended V1 maximum staleness: panel_interval_seconds.

If no row satisfies those rules, preserve the candidate with a typed null reason: panel_warmup;
no_completed_panel_bar; panel_gap; panel_stale.

Never carry a regime across the 18:00 trading-day reset.

### 3. Panel folds and candidate folds cannot share the same fold_set_id

The plan creates panel fold sets keyed by row_id and candidate fold sets keyed by candidate_id.
It later requires fold_set_id equality for the controlled feature_only comparison. Those IDs
cannot truthfully be equal because their observation populations and keys differ, even when they
use the same train/test day windows.

The correct abstraction is: FoldSchedulePayload; FoldScheduleEnvelope; fold_schedule_id.

fold_schedule_id should identify: authorized days; ordered train/test day windows; step; embargo;
purge policy; boundary policy.

Then: panel_fold_set_id != candidate_fold_set_id; panel.fold_schedule_id == candidate.fold_schedule_id.

Cross-grain studies require equal schedules, not equal row-set hashes. Without this addition,
panel-based regime feature studies may be impossible or may tempt the implementation to weaken
identity checks.

### 4. The MBP-1 correction still invents unsupported recovery semantics

The plan correctly withdraws: sequence jump > 1 = source gap. That is good. Databento defines
sequence as the original venue sequence number, and MBP-1 emits events that update the top price
level—not every original venue message. Raw sequence continuity therefore cannot be used as
symbol-level completeness proof.

However, the new plan says: F_MAYBE_BAD_BOOK opens an uncertainty interval; the next record without
the flag closes it.

Official Databento documentation defines F_MAYBE_BAD_BOOK as meaning an unrecoverable channel gap
was detected. It does not establish that the next unflagged message restores book correctness or
closes a stateful interval.

The plan must require a documented recovery boundary, such as: an explicit verified gap manifest;
a documented vendor recovery/clear event; a documented snapshot recovery rule; another
dataset-specific owner-approved evidence source.

When no verified recovery exists, it must fail closed for the relevant scope rather than assume the
next unflagged record repairs the book.

There is a second issue: Databento's dataset-condition endpoint reports condition at the dataset and
UTC-date level, not at symbol, schema, publisher-channel, or physical-partition granularity.
available means no known dataset issue; it does not prove one NQ MBP-1 partition is complete.

The policy should therefore say: vendor_no_known_dataset_issue; vendor_dataset_degraded — rather
than treating available alone as evidenced_complete.

The coverage calculation must also define: union of overlapping intervals; clipping to the expected
partition/session span; missing head and tail handling; empty span handling;
dataset/publisher/schema/instrument/partition scope.

This is a real-data correctness blocker.

### 5. Model training is being placed in the reporting stage

The pipeline plan puts KMeans fitting in S09, but the feature_only and cohort_model
logistic/CatBoost training is performed through the stratification service under S14. S14 is meant
for frontier, reports, and insights. It should not secretly fit models.

The pipeline should be: S05 materialize observation/features; S06 coverage and adequacy; S08 folds;
S09a KMeans fits and assignments; S09b fold-local regime features; S09c logistic/CatBoost controlled
models; S10 diagnostics and paired deltas; S14 persist/render stratified reports only.

A descriptive KMeans study should not require S07 labels at all. Labels and supervised model stages
should be required only for: feature_only; cohort_model.

This preserves the computation-path-scoped architecture.

### 6. The prop-event sidecar remains an unanswered owner assumption

The README correctly flags account_events_by_path.json as requiring your confirmation.

The sidecar is reasonable and I recommend approving it, but only if it becomes part of a new
versioned immutable account-simulation artifact.

The plan must add: event_detail_persistence_policy_id; account simulation schema/version bump;
sidecar hash, byte size, row count in manifest; path_instance_id + account_id + event_id identity;
new account_simulation_id when policy changes; no widening or mutation of existing artifacts.

Without this, an additional result-bearing sidecar could be added without changing the immutable
identity.

If you decline the sidecar, stratified_prop must be historical-only and bootstrap/stress
attribution must return a typed evidence_not_persisted state.

Do not begin implementation with that question still open.

### 7. Baseline and challenger OOS row identities will not match

The new CatBoost runner proposes: oos_row_id = hash(view_id, fold_index, candidate_id).

A baseline feature view and an MBP-1/regime challenger view have different view_id values. Their
OOS IDs will therefore differ even when they contain exactly the same candidates and folds.

Use a bundle-independent comparison-row identity, for example:
hash(fold_schedule_id, candidate_fold_set_id, fold_index, candidate_id, label_view_id)

The model/prediction artifact identity should separately include the feature-view and protocol IDs.

This is required to enforce the owner's "identical OOS rows across arms" rule.

## Smaller amendments

These are not architectural blockers, but should be fixed in the same revision.

**Panel formula source count** — Several formulas require current bar + 12 prior bars. That is 13
completed source bars. Stamp: minimum_source_bars = 13. Also explicitly pin: population standard
deviation ddof = 0.

**Stability naming** — The plan calls the AMI threshold an "advisory floor" but uses it as a
promotion gate. Use either minimum_bootstrap_aligned_ami_mean or retain advisory semantics and do
not fail promotion on it.

**Required adversarial tests** — Add tests for: future outer-fold leakage into regime features;
previous-day assignment during next-day warmup; same schedule / different panel-candidate fold
sets; F_MAYBE_BAD_BOOK without documented recovery; overlapping declared gap intervals; dataset
condition available but missing partition evidence; S14 performing zero fitting; equal
comparison-row IDs across baseline/challenger bundles.


---

## Revision-3 closure note

Revision 3 preserves every correction above and adds the final contract-closure items documented in
`FINAL_PLAN_CORRECTIONS_2026-08-28.md`: exact promotion identity, physical/channel MBP-1 scope,
complete source-bar validity, decision 25 authorization, bounded Parquet prop-event persistence,
fit-local model cluster features, non-self-referential source-document hashes, and release patch/bundle evidence.
