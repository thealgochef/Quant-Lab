# IFVG Context Experiment v1 Contract

Status: locked implementation contract

## Artifact and capability contracts

Quant-Lab context container, dataset, and manifest schemas are version 4. Artifacts are addressed
only by catalog ID, resolved beneath configured roots, and verified for manifest hash, file hash,
size, row count, schema, PK/FK/as-of integrity, and the exact v3-to-v2 reference.

Capability registry:

| Profile | Status | Reason |
|---|---|---|
| `ifvg_v2_doc_default_fresh_static_1r` | runnable | accepted default |
| `ifvg_v2_ict_clean_fresh_static_1r` | blocked | `same_leg_locality_sweep_semantics_unresolved` |
| `ifvg_v2_ict_clean_pure_retest_static_1r` | blocked | `owner_trigger_not_selected` |
| `ifvg_v2_weak_counter_displacement_research` | analysis_only | research-only construction |
| `ifvg_v1_legacy_candidate_stream` | legacy_read_only | legacy contract |
| canonical-short enabled | blocked | `canonical_short_profile_not_ratified` |

Preparation statuses are `not_prepared`, `preparing`, `failed`, `context_ready`, `superseded`,
`legacy_only`, and `blocked`.

## Arrow registry

`ifvg_context_arrow_v1` declares ordered `pyarrow.Schema` objects for:

1. `context_state`
2. `context_capture`
3. `context_structure_state`
4. `context_structure_delta`
5. `context_displacement_window`
6. `equal_level_pool_lifecycle`
7. `equal_level_pool_member`
8. `equal_level_sweep_link`
9. `context_validity_provenance`
10. `candidate_context_link`
11. `decision_context_link`
12. `trade_context_link`

IDs/enums/hashes/composite cursors are UTF-8; tick prices/counts/ordinal cursors are int64;
normalized measurements are float64; instants are `timestamp[us, tz=UTC]`; flags are boolean;
ID collections are `list<utf8>`. Optional fields have fixed nullable types. Column order and
metadata are canonical. Missing or undeclared columns fail. Empty, all-null, and populated tables
share the exact schema. Every table schema hash and the aggregate schema hash enter identity.

## Exact candidate feature view

One immutable candidate view supplies the same candidate IDs and fold assignment to every tier.
`CandidateStageContextLink` binds candidate ID, stage, exact geometry evidence ID/cursor, and one
unique context capture. `context.as_of_ts <= feature_as_of_ts` is mandatory. Missing, duplicate,
setup-only, nearest-time, row-order, and keep-last joins fail. A legitimately unavailable feature
keeps the candidate with null values plus a registered validity/missing reason.

Tiers:

- M0: explicit causal v2 candidate allowlist.
- M1_PRIMARY: M0 plus local 1m/3m/5m/10m/15m/30m/60m structure primitives; all 240m data and
  influenced aggregates excluded.
- M1_PLUS_240_EXPERIMENTAL: M1 plus explicit 240m fields.
- M2: M1 primary plus formula-v2 displacement. Formula-v1 and superseded artifacts are rejected.
- M3: M2 plus pools and opposing-leg sweep evidence. Without positive qualification variation it
  is `descriptive_only_no_positive_qualification_coverage`.

Observation filters apply only after view construction. Strategy construction changes—session,
short enablement, trigger, and similar controls—require a newly prepared profile.

## Counterfactual labels

R1.0, R1.5, R2.0 and fixed SL/TP overrides are separately hashed derivations. Each starts at the
first eligible bar strictly after entry, and recomputes resolution, MFE, MAE, availability, and
horizon. R1.0 path metrics are never reused for another barrier. BE-managed labels and actual v2
execution outcomes are outside this label contract. Censored rows remain in reports but not in
binary targets. Baseline research cost is `cost_per_trade_r=0.0` and enters label identity.

## Fold protocol

Expanding folds use 40 authorized trading days of training, 5 test days, a 5-day step, and at
least 30 training candidates. Setups stay whole; boundary-crossing setups are excluded. Training
label intervals overlapping the test interval are purged, then a two-authorized-trading-day
embargo is applied. Entry and resolution availability and both training classes are required.
Invalid folds and reasons are retained. Test windows do not overlap, so each candidate has at most
one OOS prediction. If no valid fold remains, descriptive reporting completes with
`insufficient_class_coverage`.

## Model and statistics protocol

Model ID: `ifvg_context_catboost_binary_v1`.

```text
iterations=200; depth=4; learning_rate=0.08; loss_function=Logloss
eval_metric=Logloss; l2_leaf_reg=3; random_seed=7; random_strength=0
bootstrap_type=No; boosting_type=Plain; grow_policy=SymmetricTree
task_type=CPU; thread_count=1; class_weights=None; use_best_model=False
allow_writing_files=False; threshold_report_grid=[0.40,0.50,0.60,0.70]
calibration_policy=raw_probability_diagnostics_v1
```

The resolved configuration, categorical registry, package/Python versions, feature registry, and
manual overrides are hashed. Numeric missingness stays native NaN; categoricals use registered
`__MISSING__`. Search, automatic feature selection, threshold selection, calibration fitting, and
promotion are prohibited.

Each fold includes a training-prevalence reference. Reports include Brier and skill, log loss,
AUC when defined, calibration intercept/slope, ten fixed equal-width reliability bins, threshold
coverage, and gross/net R. Feature importance is descriptive only: fold CatBoost importance and
20-repeat held-out permutation importance with seed 7, coverage, and variance.

Uncertainty uses 10,000 repetitions and seed 7: setup-cluster and trading-day blocks for candidate
statistics, paired trading-day blocks for tier deltas, and trading-day blocks for executed trades.
Fewer than two clusters yields a null interval with a reason.

## Persistence and reporting

Views live at `data/ifvg_datasets/context_views/v1/<view-id>/`; runs live at
`data/ifvg_experiments/context_v1/<full-sha256-id>/`. Writes are atomic, immutable, and refuse
duplicates. Display names and notes belong only to a mutable catalog.

Reports remain separated into candidate research, actual execution, feature coverage, and
reconciliation/audit. Quantitative deltas require identical artifact pair, cohort, labels, folds,
model/calibration/bootstrap protocols, and OOS row IDs; a registered tier comparison may differ
only by tier. Otherwise only a configuration diff is shown.

## Implementation handoff

The schema-4 contract is implemented in:

- `context_experiment_contracts.py` for capabilities, preparation states, tiers, references,
  configs, fold/result identities, and reconciliation;
- `context_schemas.py` for the twelve exact Arrow schemas, per-table hashes, and aggregate hash;
- `artifact_io.py` and `preparation.py` for verified pair loading, atomic preparation,
  locks/checkpoints/cancellation, and crash-safe reuse;
- `context_feature_view.py` for exact candidate-stage geometry/cursor/capture links and tier
  allowlists;
- `context_labels.py`, `context_folds.py`, `context_model.py`, and `context_statistics.py` for
  independent barriers, setup-safe walk-forward partitions, the fixed CatBoost protocol, and
  deterministic diagnostics/bootstrap;
- `context_reporting.py` and `context_run_store.py` for separated reports, complete run identity,
  immutable persistence, verified reload, and compatibility-aware deltas.

Exact-link construction uses the v2 trigger evidence ID and cursor and rejects missing,
duplicate, contradictory, setup-only, or approximate evidence. Entry timestamps are taken from
the exact candidate record rather than a nullable union-column fallback. Fold boundary checks
run before censor/availability filtering, and cohort-specific M3 status requires actual opposing-
leg variation in the selected cohort.

All Quant-Lab tests pass (970 total), including schema, tamper, access, preparation, exact-link,
label/fold/model/statistics, immutable-store, and UI AppTest coverage. No current context run was
created because a verified formula-v2 pair does not exist: January publication is blocked by the
aggregate replay-performance gate. Legacy stored results remain read-only and are not treated as
current schema-4 runs.
