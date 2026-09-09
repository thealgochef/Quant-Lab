# R6 — Deviations and Scoping Notes

Reconciled after the adversarial round (`ADVERSARIAL_REVIEW_RESOLUTION.md`);
every entry states what the CODE does now.

## DEV-R6-1 — Assignment-frame projection of the plan's per-row contract

The plan's `RegimeAssignment` (ML §3) lists `distances: tuple[float, ...]`
(every centroid) and the GMM-only `probabilities`/`log_density`. The V1
persisted frame carries the full `distances` list (the §4 `kmeans_v1`
output — review F13), `assigned_distance`, `assignment_margin` (d2−d1),
and the observation's as-of timestamp `observation_ts_utc`;
`assignment_entropy`/`log_density`/`outlier_score` ride as NaN columns
(schema stable for the post-V1 algorithms); `probabilities` is added by
the GMM release. `RegimeAssignmentColumns` names exactly the persisted
columns. DECISIONS_TAKEN #54.

## DEV-R6-2 — Assessment/promotion envelope wrappers

`RegimeCapabilityAssessment` and `RegimePromotionDecision` are typed in the
plan as bare contracts; persistence requires store keys, so both gained
content-addressed Payload/Envelope wrappers (the coverage-report
precedent), registered with the identity-projection audit. Additive
(DECISIONS_TAKEN #55); CS §0.1 names only RegimeProtocol/RegimeFit as
mandatory pairs, which both exist exactly as specified.

## DEV-R6-3 — Bootstrap-stability engineering protocol

The plan sketches "50 seeded refits, aligned-AMI + per-cluster agreement".
The implementation pins: resamples of the REFERENCE fold's training matrix
only (never test rows — test-pinned, review F16), deterministic seeds
(`default_rng(7)` resampling; refit seeds 1000+i), the EXACT
ascending-local-id Hungarian tie-break (review F8), adjusted mutual
information scoring, and the advisory ≥0.5 floor as a promotion-blocking
(never fit-deleting) stability gate. DECISIONS_TAKEN #56 — an engineering
default under the plan's sketch, unratified like every default.

## DEV-R6-4 — Stratified RESULT views are a study-class deliverable

FUX §35 R6 mandates "coverage, occupancy, stability, assignment, and
stratification views". Delivered on the Regime Lane panel (pipeline-surface
expander beside the MBP-1 panel; FUX-IA-002 keeps the five-entry sub-nav):
coverage (incl. per-fold coverage and typed nulls), nominal-id occupancy,
stability (bootstrap AMI mean/5th percentile, per-cluster agreement, the
OOS-timeline persistence + transition matrix, recurrence, separation,
descriptive silhouette, centroid profiles), the exact-fit-id ASSIGNMENT view
(artifact facts, coverage by partition, assigned-distance/margin quantiles,
the OOS regime timeline), the STRATIFICATION of the assignment frame
(rows/share/mean margin by nominal regime × partition), the model-card
identities (protocol, input bundle, fit ids, fold set, assessment), and the
promotion-decision role/status view. **What is NOT in R6:** stratified
RESULT views — strategy/model/prop METRICS by regime — which are the
comparison classes of ML §5.5 (`cohort_descriptive` regime filter,
`feature_only`, `cohort_model`, stratified frontier/prop reports) and land
with their studies. This is a scoping deviation recorded for the owner
(GATE_SUMMARY gate row), not a delivery claim. DECISIONS_TAKEN #58.

## DEV-R6-5 — Panel→candidate assignment is OOS-only and order-free

`assign_panel_regimes_to_candidates` consults ONLY `partition == "test"`
valid rows (a frozen out-of-sample fit — never a fit that saw the
candidate's future), takes the last COMPLETED bar at or before the as-of
instant (`<=` on bar close), and when several OOS folds cover the same bar
chooses the lowest `fold_index` (deterministic, independent of input row
order); the output carries `regime_fit_id` and `partition`. A bar with no
OOS assignment types `coverage_gap`. Replaces the midpoint "first valid
row" rule (review F4). DECISIONS_TAKEN #57.

## DEV-R6-6 — Pre-acceptance schema extensions (identities differ from the midpoint tree)

The adversarial round added identity-bearing fields: `pinned_parameters_hash`
on the protocol (F12), `training_feature_matrix_hash` + non-empty verified
`source_artifact_ids` on the fit (F1), `previous_decision_ref` + patterned
`owner_ratification_ref` + validated `decided_at` on the promotion decision
(F5/S1), and the stability report's `temporal_order_policy` /
`temporal_transition_count` / `alignment_space` (F3/F9). Every derived
identity computed after these changes differs from the midpoint values; no
persisted regime artifact exists outside test/smoke tmp roots, so nothing
immutable was mutated (the R5B DEV-R5B-3/#51 precedent). DECISIONS_TAKEN #60.

## DEV-R6-7 — No panel materializer / panel fold builder in V1

The `CONTEXT_BAR_PANEL` grain is implemented in the actual schema, the fit
path executes on a panel keyed by `row_id` with hand-built walk-forward
folds (`ContextFoldSet` over trading days; 480-bar synthetic 5m panel,
≥300 training bars per fold — review F14), and the PIT panel→candidate
assignment is test-pinned for 5m and 15m. What R6 does NOT ship: a
completed-bar panel MATERIALIZER from the replay-chart resample set and a
panel-native fold builder (`build_context_folds` takes labeled candidates).
The plan lists the 1m panel builder as planned (§12); the 5m/15m
materializer is recorded here as the same class of post-R6 wiring.
DECISIONS_TAKEN #61.

## DEV-R6-8 — Store integrity is not authenticity (trust boundary)

The manifest protocol verifies INTEGRITY (manifest hash + sha256 of every
sidecar before any byte is interpreted) — not authenticity. A store root is
a trusted local directory: an actor with write access who re-hashes a
sidecar AND the manifest passes verification, and the joblib sidecar is
then unpickled. Mitigations in R6: the UI never unpickles
(`load_regime_fit_assignments` — JSON + Arrow only), `load_sidecar_bytes`
verifies the manifest hash, whitelists the sidecar name, and hashes the
returned bytes (review S6), and `load_regime_fit` is reached only by
`persist_regime_fit`'s own verification and tests. Signing / a non-pickle
serialization is a post-V1 hardening candidate (review S7). DECISIONS_TAKEN
#62.

## Scoping notes (not deviations)

- The five-day real mini-run of ML §9 ("artifact pair → … → one kmeans_v1
  regime fit") is part of the SAME owner-blocked real slice as every real
  half since R1 (DEV-R5-6 / DEV-R5B-1 pattern): the synthetic fixtures
  prove the full chain shape, and the real half waits on the
  `VerificationAuthorizationRef`. Recorded as the standing first blocker.
- Fixture 3 (two-ring spectral graph) and the GMM/Nyström arms of fixture
  4 are the post-V1 regime-expansion release by plan (V3 P1-6) — not
  implemented here by design.
- `inner_train_only_selection`, PCA, kernels, `none_training_only`,
  `gmm_posterior_v1`, `nystrom_transform_kmeans_predict_v1`, and
  `surrogate_logistic_v1` are schema-EXPRESSIBLE (every planned protocol
  can be written down and hashed in V1) and execution-REFUSED
  (`assert_protocol_executable` fails closed before any preprocessing or
  fit — review S3); `fixed_k` + `centroid_predict_v1` under `kmeans_v1` is
  the single executable policy tuple.
