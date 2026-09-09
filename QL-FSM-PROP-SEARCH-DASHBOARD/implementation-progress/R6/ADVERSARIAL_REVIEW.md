# R6 — Adversarial Review (two independent read-only reviewers)

Per kickoff §10: independent reviewers attempted to falsify the release
against `ML_REGIME_CONTRACT_PLAN.md`, `CONTRACTS_AND_SCHEMAS.md`,
`TEST_MATRIX.md`, `PHASED_DELIVERY.md`, `FRONTEND_UX_CONTRACT.md` §35, the
protected/sealed access rules, the kickoff §9 ML/regime constraints, and
the M0–M3/immutable-artifact boundaries. Reviewers produced findings only
(no edits); every probe output quoted below was observed by the reviewer.
Dispositions and post-fix verification: `ADVERSARIAL_REVIEW_RESOLUTION.md`.

Context: a first adversarial round ran before the 2026-08-26 context reset
(its C-series fixes — e.g. the single-source `REGIME_PROPOSED_DEFAULTS`,
review C7/C9 — are already in the code); its reports were lost with the
context, so the round was re-run in full on the post-C-series tree. The
findings below are the surviving, re-verified set.

---

## Reviewer 1 — contract fidelity (verdict: 0 blockers, 7 majors, 9 minors)

All 25 R6 unit tests, the 19 identity-audit tests, and the 4 R6 AppTests
pass; `ruff check` on the R6 set is clean. The findings were obtained by
reading all seven modules, the panel, the diffs and the tests, and by
running probe scripts against the shipped fixtures (outputs quoted
verbatim).

**F1 (MAJOR)** — `regime_fit_id` does not bind the fitted content.
`RegimeFitPayload` hashes only `{protocol id, source_artifact_ids,
fold_index, fit_start, fit_end, training_row_ids_hash}`;
`run_regime_protocol` defaults `source_artifact_ids=()` and every test
passes nothing, and `training_row_ids_hash` hashes candidate IDs only. Two
fits over frames with identical row ids but different feature values
therefore mint the SAME `regime_fit_id` while producing different fitted
parameters — the store's byte backstop catches the collision at persist
time with a confusing "DIFFERENT sidecar content" error instead of the
identity being distinct. Nothing binds `view_id`/`artifact_pair_hash` of
the `CandidateFeatureView` either (the service takes a bare frame), and
`input_feature_bundle_ref` is an unchecked caller string (the resolver
never verifies `resolved_input_features ⊆ bundle`).
Citation: CS §0.1 ("pin an already-materialized upstream evidence artifact
by its **verified ID**"); ML plan §3 `RegimeFitPayload.source_artifact_ids`;
TEST_MATRIX §3.10 "Regime fit provenance"; the R5B F1/F2 precedent ("a
caller-provided string is NOT evidence").
Evidence: probe over fixture 2 with `perturbed[c] = c*3 + 11`: `B1 same
regime_fit_id per fold: [True, True, True]`, `same param payload hash:
[False, False, False]`, `source_artifact_ids in fit payload: ()`; second
persist → `SearchStoreError … exists with DIFFERENT sidecar content`.
Reproduction: run `run_regime_protocol` twice on
`known_cluster_fixture(3,600).view.frame` and on an affine-transformed copy
with the same protocol/folds; compare `fold_fits[i].fit_envelope.regime_fit_id`
and `preprocessing.fitted_parameter_payload_hash`.
Suggested resolution: require a non-empty, verified `source_artifact_ids`
(or a content hash of the observation frame / the view identity) in
`RegimeFitPayload`; refuse `()`; verify that `resolved_input_features`
belong to the referenced bundle.

**F2 (MAJOR)** — The §5.2 stage-leakage rule is opt-in.
`assert_no_regime_leakage` only checks availability stages `if
feature_stage_for:`; `resolve_kmeans_protocol` forwards whatever the caller
supplies (default `None`) and `run_regime_protocol` re-checks WITHOUT stage
info. A feature from an `entry_decision` block is accepted into a protocol
declared at `observation_stage=HTF_TAP`. The feature→block map needed to
enforce this exists in the same package
(`FEATURE_BLOCK_RESOLUTION_REGISTRY` payloads carry `feature_names`).
Citation: ML plan §5.2 ("Prohibited inputs are refused … any feature whose
block availability_stage > the spec observation_stage", acceptance
7B.22-3).
Evidence: `D1 'direction' (block IFVG_CORE_BASELINE_V1, stage
entry_decision) ACCEPTED at observation_stage=htf_tap -> protocol
9bec55b10581`; `C3 run_regime_protocol passes feature_stage_for: False`.
The only test of the stage rule hands the validator an explicit map.
Reproduction: `resolve_kmeans_protocol(input_feature_bundle_ref="a"*64,
resolved_input_features=("direction",),
observation_stage=AvailabilityStage.HTF_TAP)`.
Suggested resolution: derive `feature_stage_for` from the block/resolution
registries inside the resolver (refuse unknown feature names), and re-run
the stage check in `run_regime_protocol`.

**F3 (MAJOR)** — `temporal_persistence` and `transition_matrix` are not
temporal statistics. `build_stability_report` orders valid TRAIN rows by
`["fold_index", "row_id"]` ("row-id (entry-time surrogate) order") and
concatenates across folds. (a) Real candidate ids are content-derived
uuid5 (plan §12 flagged assumption (a)), so lexical order is random; (b)
with expanding-window folds the same candidate appears in up to every
fold's training set, so the "sequence" is 1,375 entries for 518 distinct
rows; (c) fold boundaries create spurious transitions. Because these
numbers live in the hashed assessment, content-identical data yields a
different `regime_capability_assessment_id` depending purely on id
spelling.
Citation: ML plan §3 `RegimeStabilityReport.temporal_persistence;
transition_matrix`; §10 "regime timeline, transition matrix"; CS §0.1
(identity must not depend on non-semantic facts).
Evidence: `A6 train rows: 1375 distinct row ids: 518 ids in >1 fold: 455
max folds per id: 3`; renaming ids to sha256 prefixes on the SAME data:
`B4 persistence with kc_#### ids: 0.330422 | with hashed ids: 0.347162`,
`transition[0] … (0.325893, 0.323661, 0.350446) | hashed: (0.354037,
0.3147, 0.331263)`, `assessment ids equal? False`.
Reproduction: map `candidate_id` through `hashlib.sha256(...)[:16]` in both
the frame and the labels, rebuild folds, rerun, compare
`assessment.payload.stability`.
Suggested resolution: order by the frame's `entry_ts_utc` (available on
every row), compute per fold (or on OOS test rows only) without cross-fold
concatenation, and exclude the statistic from the hashed payload until it
is semantically defined.

**F4 (MAJOR)** — `assign_panel_regimes_to_candidates` ignores `partition`
and is input-order dependent. It picks the first `valid` row for the bar
regardless of whether that row is a TRAIN-partition (in-sample)
assignment, and `lookup.loc[[bar_id]]` preserves the caller's frame order,
not `(row_id, fold_index)` order — so DECISIONS_TAKEN #57 / DEV-R6-5 ("the
deterministic first row in (row_id, fold_index) order") is not what the
code does. The output frame carries no `partition`, so a downstream
consumer cannot apply the plan's own PIT rule (regime features read only
`partition="test", valid=True` rows, §4). "Never a future bar" is honored,
but "never a fit that saw the future" is not.
Citation: Amendment P1-B / ML plan §5.1 ("assign the **frozen** current
regime to candidate stages at their as-of instants"); §4 test-partition-only
consumption; TEST_MATRIX §3.8 "PIT panel→candidate assignment".
Evidence: same bar with rows `[fold3-train, fold0-test]` → `{'fold_index':
3, 'fold_local_cluster_id': 2}`; rows reversed → `{'fold_index': 0,
'fold_local_cluster_id': 1}`; `'partition' in output: False`. The shipped
test's `panel_assignments` frame has no `partition` column at all.
Suggested resolution: require `partition` in `panel_assignments`, choose
only `partition == "test"` rows (or a fold whose training window closed
before the bar), sort the candidates deterministically, and emit
`partition`/`regime_fit_id` in the output.

**F5 (MAJOR)** — The promotion ladder is advisory, not structural.
`RegimePromotionDecision` only requires a non-empty `owner_ratification_ref`
string for `FEATURE_ELIGIBLE+`; the one-step rule and the `gates_passed`
requirement live only in the free function `assert_lawful_promotion`, which
`persist_regime_promotion` never calls and which never loads the referenced
assessment. `decided_at` and `owner_ratification_ref` accept any string; no
`OwnerDecisionEvidenceRef` is required.
Citation: ML plan §1 ("the promotion sequence … is enforced by status
transitions in `regime_contracts.py`"); §5.4 ("promotion requires the
owner's `OwnerDecisionEvidenceRef`"); ACCESS_SAFETY_EVIDENCE ("structurally
requires a PASSING capability assessment AND the owner's ratification
reference — at both … and the `RegimePromotionDecision` contract itself").
Evidence: `C2 PLANNED->MODEL_FEATURE, role=frozen_execution_gate,
ratification='x', decided_at='whenever' persisted: a132e616f223 reused:
False` (with `capability_assessment_ref="b"*64`, an assessment that does
not exist).
Reproduction: construct that `RegimePromotionDecision`, wrap with
`RegimePromotionDecisionEnvelope.from_payload`, call
`persist_regime_promotion(tmp, env)`.
Suggested resolution: make `persist_regime_promotion` load the referenced
assessment and the previous decision and call `assert_lawful_promotion`;
type `owner_ratification_ref` as an `OwnerDecisionEvidenceRef` id (64-hex)
and validate `decided_at` as ISO-8601.

**F6 (MAJOR, borderline)** — Persistence verification does not bind the
assignments to the fit. `persist_regime_fit` verifies only that the
reloaded PREPROCESSING reproduces a caller-chosen slice; it never checks
that `fold_assignments.regime_fit_id`/`fold_index` match the envelope, nor
that the reloaded estimator reproduces the persisted
`fold_local_cluster_id`s. A fit persisted with another fold's assignment
frame publishes as "verified" and reloads with a foreign sidecar. No test
proves reloaded estimator + pipeline == persisted assignments.
Citation: ML plan §5.3 (`_persist_and_verify_regime_fit`), §3.10
("assignments reference exact fit+protocol").
Evidence: `D4 fold-0 fit persisted with fold-1 assignments: ACCEPTED;
sidecar regime_fit_id matches envelope: False fold_index in sidecar: [1]`.
Suggested resolution: refuse frames whose `regime_fit_id`/`fold_index`/
protocol id disagree with the envelope; after reload, re-assign every valid
row with the reloaded pipeline+estimator and assert equality with the
persisted sidecar.

**F7 (MAJOR)** — FUX §35 R6 "assignment" and "stratification" views are not
delivered; DEV-R6-4 is a scoping deviation, not a delivery. The panel
renders aggregate coverage, occupancy, five stability scalars, and the
transition matrix. It renders no per-row/per-fold assignment facts (no
fit-id load, no distance/margin distributions, no regime timeline), no
metrics-by-regime, and — contrary to DEV-R6-4's own wording
("per-fold-coverage rendering") — not `per_fold_coverage` either. §10
model-card fields also missing: role/status (no promotion decision
surfaced), input bundle id, fit ids/windows, the assessment identity block,
centroid profiles (`semantic_descriptors` unrendered),
`per_cluster_agreement`, `bootstrap_aligned_ami_low`. No AppTest exercises
a `CONTEXT_BAR_PANEL` protocol in the card, and no AppTest asserts the
absence of a launch/promote control.
Citation: FUX §35 R6 ("coverage, occupancy, stability, assignment, and
stratification views"); PHASED_DELIVERY R6 gate; ML plan §10 model card +
regime diagnostics.
Evidence: `grep -n "per_fold_coverage|regime_fit_ids|semantic_descriptors|
per_cluster_agreement|bootstrap_aligned_ami_low" scripts/ifvg_regime_panels.py`
→ no hits.
Suggested resolution: either deliver an exact-fit-id assignment view
(persisted `assignments.arrow` → per-fold coverage, margin/distance
distributions, timeline by `entry_ts_utc`) plus centroid profiles, and
record the stratified-RESULT views as an explicit deviation with an
owner-visible gate note; or amend the R6 gate text.

**F8 (MINOR)** — The Hungarian "ascending fold-local id" tie-break is not
implemented. The epsilon added in `regime_alignment.py` is a per-ROW
constant; every complete assignment uses every row once, so it cannot
change which optimal assignment wins. `_aligned_labels` in diagnostics has
no tie-break at all.
Citation: ML plan §5.4 ("ties broken by ascending fold-local id").
Evidence: brute force over 4,000 random 3×3 tie-prone matrices: `trials
with ties: 1711; scipy+row-eps != ascending-local-id choice in 605`.
Suggested resolution: enumerate optimal assignments (k ≤ 5) and pick the
lexicographically smallest, or add a lexicographic `(row, col)` epsilon
and document the exact rule.

**F9 (MINOR)** — Fold feature-widths can differ (indicator columns appear
only when a feature is missing in that fold's TRAIN rows) and
alignment/recurrence silently truncate to `min(width)`; fold-to-fold
centroids are compared in different spaces without a refusal or a flag.
Evidence: `D2 per-fold output feature names: [(0, (…2)), (1, (…2)), (2,
('distance_to_htf_ticks', 'opposing_size_ticks',
'missing_indicator__opposing_size_ticks'))]`; `gates: ()`.
Suggested resolution: align on the shared FEATURE prefix explicitly and
record the alignment space in the report, or fix the indicator set
protocol-wide.

**F10 (MINOR)** — All-missing training rows are imputed INTO the fit (a
synthetic median point with indicator=1 per row) yet reported as
`source_feature_missing`/`valid=False`; `training_row_count` and
`training_row_ids_hash` include them.
Evidence: `A5 fold 0: all-missing ids inside training_row_ids: ['kc_0000',
…, 'kc_0400'] training_row_count=402` while their assignment rows read
`valid: False, missing_reason: source_feature_missing`.
Suggested resolution: exclude all-missing rows from the training matrix
(or state explicitly that they participate) so the fit identity and the
typed-null report agree.

**F11 (MINOR)** — Non-unique `candidate_id` values are accepted by
`fit_regime_preprocessing` (`.loc[list(train_ids)]` returns duplicates), so
the fitted statistics change while `training_row_ids` still reports the
de-duplicated count. `_fit_one_fold` then silently uses `.iloc[0]`.
Evidence: `B2 duplicate-id frame accepted; param hash changed: True |
training_row_ids len: 402 | scaler n_samples_seen: 403 vs base 402`.
Suggested resolution: refuse frames whose `candidate_id` is non-unique.

**F12 (MINOR)** — Protocol identity/schema vs registry: (a)
`initialization_policy` is `Literal["k-means++_n_init_10_v1"]` whereas the
plan types it `str`; the planned GMM/minibatch/Nyström protocols cannot be
expressed without a schema change. (b) The protocol payload embeds no
pinned parameters (only `algorithm_version="1"`), so a registry pin edit
without a version bump keeps the same `resolved_regime_protocol_id` (CS
§0.2). (c) The nested `pinned_parameters["kmeans"]` dict of
`nystrom_kmeans_v1` is mutable in place through `ImmutableMap[str, Any]`
(CS §0.3).
Evidence: `E3 protocol payload fields: [] | algorithm_version: 1`; `E2
nested pinned dict mutated in place: before 10 after 99`.
Suggested resolution: widen `initialization_policy` to the registered set;
hash `entry.pinned_parameters` into the protocol payload; deep-freeze
nested registry values.

**F13 (MINOR)** — Declared assignment schema is inconsistent with the
frame: `RegimeAssignmentColumns` lists `distances` and `probabilities` that
the persisted frame never carries.
Evidence: `A4 declared - actual: ['distances', 'probabilities']`.
Suggested resolution: make the constant match the frame (or emit the
distance vector, which is cheap for KMeans and is what §4 names as an
output).

**F14 (MINOR)** — Gate-value duplication and untested grain path:
`SAMPLE_ADEQUACY_MINIMUMS` restates the values instead of reading
`REGIME_PROPOSED_DEFAULTS` ("the SINGLE source"); `DECISION_ROW → 150` is
not stamped anywhere; the `CONTEXT_BAR_PANEL` (300) fit path is never
executed (no fold builder exists for bar panels and no test runs
`run_regime_protocol` on a panel protocol).
Suggested resolution: derive the minimums from the stamped defaults; add a
panel-grain fit test (or record the absence of a panel fold builder as a
deviation).

**F15 (MINOR)** — Overclaims in the implementer's declarations relative to
the code: ACCESS_SAFETY "structurally requires … at both" (F5);
DEV-R6-5/#57 "(row_id, fold_index) order" (F4); DEV-R6-4
"per-fold-coverage rendering" (F7); `regime_alignment` docstring tie-break
(F8); `regime_diagnostics` "entry-time surrogate" (F3);
`regime_preprocessing` "a pooled or future-leaking fit is unrepresentable
at this seam" — a hand-built `IfvgContextFoldDefinition`/`ContextFoldSet`
with any ids is representable and nothing binds `fold_set_id` to
`build_context_folds` (the signature test is exactly as strong as the
plan's wording and no stronger).

**F16 (MINOR)** — Missing mandated/obvious proofs: no test that the
bootstrap resamples the reference fold's TRAINING matrix only; no
`software_versions`-sensitivity test; no winsorization
(`clip_p01_p99_train_fitted_v1`) test at all, so the persisted clip bounds
and both `transform` clip branches are untested; no k=2 test; no
reloaded-estimator-vs-persisted-assignments test (F6); no AppTest asserting
the regime panel exposes no button/launch control; no AppTest for the
context-panel grain identity string. (The reviewer verified winsorization,
k=2 and reload-equality manually — they work — but the release does not
prove them.)

### Checks attacked that SURVIVED (reviewer 1)

- **Fold-locality of the fit and every stability statistic**: poisoning
  every fold-0 TEST row and every out-of-fold row with `1e9` left fold-0
  centroids, `regime_fit_id`, bootstrap AMI, silhouette, min-separation and
  all fold-0 TRAIN assignments unchanged; the bootstrap resamples
  `train_matrix[sample]` of the reference fold only and `silhouette_score`
  sees only `train_matrix`; `fit_regime_preprocessing` slices
  `fold.train_candidate_ids` internally.
- **Distances/margins**: computed in the scaled space against
  `centroids_scaled` after `preprocessing.transform`; manual argmin agrees
  with `estimator.predict` on `0 / 1551` valid rows; k=2 margins populated.
- **Winsorization end-to-end** (manual): clip bounds fitted on train,
  persisted in `parameters.json`, reload `allclose: True`, and the reloaded
  estimator reproduced all 461 persisted fold-0 `fold_local_cluster_id`s.
- **Identity audit**: `registered_identity_pairs()` enumerates 44 pairs
  including the four regime pairs; `test_identities.py` (19 passed) proves
  payload→id→envelope→reload determinism, tampered-id refusal,
  forbidden-field absence and mutation-adversarial deep immutability; no
  absolute path or self-hash exists in any hashed regime payload;
  `software_versions` is plan-listed and changes the protocol id;
  `preprocessing_pipeline_ref` is validated relative and actually resolved
  through the manifest.
- **Store discipline**: reuse compares sidecar bytes and fails closed on
  different content; joblib/arrow bytes were deterministic (idempotent
  second persist passes); every sidecar (joblib included) loads through
  `load_sidecar_bytes` hash verification and the parameter payload is
  re-hashed against `fitted_parameter_payload_hash`; relocation and tamper
  tests pass; an empty verification slice is refused.
- **P1-C / V1 boundary**: planned keys refuse at
  `assert_regime_algorithm_fittable` and again inside `run_regime_protocol`;
  import-time invariant enforces exactly one implemented algorithm;
  `GaussianMixture`/`SpectralClustering`/`Nystroem`/`MiniBatchKMeans`/
  torch/keras/gym are imported nowhere in the R6 set;
  `IFVG_REGIME_CONTEXT_V1` still refuses resolution.
- **Registry pins**: all six entries match ML plan §4
  parameter-for-parameter; `algorithm_version` is in the protocol
  identity; the mandatory spectral warning text is verbatim and rendered.
- **P1-B schema**: 5m/15m accepted, each missing panel field refused,
  non-panel grains refuse panel fields, `panel_interval_seconds < 60`
  refused; `<=` on completed bar close is exact and the at-close/mid-bar/
  before-first-bar cases are pinned; tz-naive inputs are localized as UTC.
- **Gates**: sample adequacy (min over valid folds), occupancy, per-fold
  cluster rows, empty-cluster, and no-OOS-coverage only append failures —
  no code path shrinks `k`, deletes a fit, selects features/thresholds/k,
  or promotes from a score; silhouette is descriptive only.
- **Separation**: `role`/`status` structurally absent from
  `RegimeProtocolPayload` and `RegimeFitPayload`; a promotion decision
  changes no protocol/fit id (unit + store tests).
- **UI safety**: no `st.button`/`selectbox`/`form` in the panel; every load
  is exact-ID through the 64-hex-validated store layer and every error
  goes through `render_empty_state`/`sanitize_error`; the sample-adequacy
  blocked state, nominal-id wording, proposal-stamp table, dev-only badge,
  and bogus-id "Artifact unavailable" state all render.
- **Typed nulls**: injected all-missing rows survive as
  `source_feature_missing` in every fold; invalid folds type every row
  `fold_invalid`; `rows_typed_null` counts them.
- **Hygiene**: `ruff check` clean on all R6 files; no tracked file under
  `data/` changed; all test writes are tmp-rooted.

---

## Reviewer 2 — safety/access (verdict: 0 blockers, 3 majors, 1 medium, 5 minors; protected/sealed zero-counter **AFFIRMED**)

Scope reviewed: the 7 new `ifvg/ml/regime_*.py` modules,
`scripts/ifvg_regime_panels.py`, the diffs to `search/store.py`,
`search/identities.py`, `scripts/ifvg_pipeline_tab.py`, `docs/DECISIONS.md`,
the 3 new test files + 2 fixtures + 4 AppTests, `R6/stage_shared_docs.py`,
and the R6 evidence documents. Ground truth for dates:
`ARCHITECTURE_MAP.md` (`2026-06-11 → PROTECTED_BUFFER; ≥ 2026-06-12 →
SEALED`), kickoff §9 Data safety, TEST_MATRIX §1–§2 (one ≤5-day allowlist,
candidate `2026-06-04…06-10`, `OWNER_DECISIONS` item 21). All repro scripts
wrote only to `%TEMP%`; no repository file was modified.

**S1 (MAJOR)** — The promotion contract does not enforce what the release
evidence says it enforces. `RegimePromotionDecision` accepts a
ladder-skipping, unassessed promotion with a fabricated ratification
string, and the persistence layer stores it.
Citation: kickoff §9 ML/regime ("No feature, model, threshold, cluster
count, or strategy is automatically selected or promoted"); ML plan §3/§5.4
("promotion requires the owner's `OwnerDecisionEvidenceRef`"; gates to
`FEATURE_ELIGIBLE`); DECISIONS_TAKEN #59 and `ACCESS_SAFETY_EVIDENCE.md`
("structurally requires a PASSING capability assessment AND the owner's
ratification reference — at both `assert_lawful_promotion` and the
`RegimePromotionDecision` contract itself"); D-044 amendment and
`ARCH_APPEND`.
Evidence: the only validator checks `not self.owner_ratification_ref`;
`previous_status` is never compared to `status`,
`capability_assessment_ref` is never resolved, `gates_passed` is not a
field, and `owner_ratification_ref: str | None` has no pattern.
`assert_lawful_promotion` has zero production callers (`grep -rn
assert_lawful_promotion src scripts tests` hits only `regime_contracts.py`
and `test_regime_contracts.py`) and its `gates_passed` is a caller-supplied
bool unbound from any assessment. Repro output: `CONSTRUCTED + PERSISTED +
RELOADED: model_feature planned ratification='x' assessment_ref=ffffffff`;
the helper chain `SUPERSEDED → DESCRIPTIVE_ONLY → STRATIFICATION_READY →
FEATURE_ELIGIBLE → MODEL_FEATURE` also passed with `gates_passed=True,
ref="x"` (SUPERSEDED is not terminal either).
Reproduction: `RegimePromotionDecision(resolved_regime_protocol_id="a"*64,
role=RegimeRole.PREDICTIVE_MODEL, status=RegimeStatus.MODEL_FEATURE,
previous_status=RegimeStatus.PLANNED, capability_assessment_ref="f"*64,
owner_ratification_ref="x", decided_at="t")` →
`RegimePromotionDecisionEnvelope.from_payload` →
`persist_regime_promotion(tmp, env)` → `load_verified_envelope(...)`
succeeds.
Suggested resolution: make the contract validator call
`assert_lawful_promotion(previous_status, status, ...)`; give
`persist_regime_promotion` a required assessment binding and refuse unless
the assessment id/protocol match and `gates_passed` holds for
`FEATURE_ELIGIBLE`+; type `owner_ratification_ref` as a 64-hex
`OwnerDecisionEvidenceRef`. Correct DECISIONS_TAKEN #59, the evidence
file, D-044 and `ARCH_APPEND` to match whatever is actually enforced.
Severity stays MAJOR (not BLOCKER) only because nothing in V1 consumes
promotion decisions: `IFVG_REGIME_CONTEXT_V1` is PLANNED and S11 is
blocked, so the persisted decision cannot reach a bundle or gate.

**S2 (MAJOR)** — The `RegimeRole` ladder is unenforced: a decision with
`role=frozen_execution_gate` (or `decision_policy` /
`execution_gate_candidate`) is constructible and persistable with any
status and no replay evidence.
Citation: ML plan §1 ("The promotion sequence (descriptive → stratification
→ feature-generator → predictive-model input → execution-gate candidate →
frozen gate after full sequential replay) is **enforced by status
transitions in `regime_contracts.py`** — a regime artifact can never become
an FSM guard … execution gate, or prop-risk control without a separately
frozen protocol plus a downstream sequential replay"); the module's own
docstring.
Evidence: `RegimePromotionDecision.role: RegimeRole` has no validator; no
code couples role to status, to `S11_BLOCKED_REASON`, or to a gated-replay
reference. Repro output: `CONSTRUCTED + PERSISTED: role=
frozen_execution_gate status= descriptive_only`; `CONSTRUCTED:
decision_policy model_feature prev= experimental`; `CONSTRUCTED:
execution_gate_candidate model_feature prev= experimental`.
Suggested resolution: add a role ladder mirroring `PROMOTION_SEQUENCE`
(role ≤ status-permitted maximum; `execution_gate_candidate`/
`frozen_execution_gate`/`decision_policy` refused in V1 with the S11
blocked reason, or requiring a `gated_replay_request_ref`). Add a test
that every execution-side role is unrepresentable in V1.

**S3 (MAJOR)** — Planned protocol policies fail **open**:
`run_regime_protocol` honors only `algorithm_key`,
`resolved_cluster_count`, `winsorization_policy`,
`resolved_input_features`, grain and stage; every other policy field is
hashed into `resolved_regime_protocol_id` and then ignored, so a protocol
declaring `inner_train_only_selection`, PCA, a kernel, or
`none_training_only` executes as plain fixed-k centroid-predict KMeans —
including OOS test-row assignments for a `none_training_only` protocol.
Citation: kickoff §5 ML/regime ("planned post-V1 algorithms fail closed");
ML plan §4 P1-C / 7B.22-6 (training-only → "emits **no** test-partition
assignments"); §5.4 (`inner_train_only_selection` "schema-supported
(planned)"); TEST_MATRIX §3.9 (identity truthfulness);
`ACCESS_SAFETY_EVIDENCE.md`.
Evidence: `_fit_one_fold` uses only `winsorization_policy` and
`resolved_cluster_count`; `("test", ...)` is assigned unconditionally; no
reference to `cluster_count_policy`, `dimensionality_reduction_policy`,
`out_of_sample_assignment_policy`, or `kernel_or_affinity_policy` anywhere
in the service. Repro output for each of the five hacked payloads: `RAN …:
fits=3 k=3 test rows assigned=120 gates_passed=True oos_available=True`.
The `REGIME_ASSIGNMENT_MISSING_REASONS` entry `training_only_algorithm` is
never emitted by any path.
Reproduction: `hacked = RegimeProtocolEnvelope.from_payload(
protocol.payload.model_copy(update={"out_of_sample_assignment_policy":
"none_training_only"}))`; `run_regime_protocol(frame, folds, hacked)`
succeeds with 120 valid OOS rows.
Suggested resolution: in `run_regime_protocol` (before any fit) refuse any
payload whose policy tuple is not the single executable V1 tuple with a
planned refusal; add a registry of executable policy values per
algorithm; test that every planned policy value refuses before
preprocessing. Update the evidence wording ("no implementation" →
"refused").

**S4 (MEDIUM)** — Input permission is not structural: the
availability-stage half of `assert_no_regime_leakage` only runs when a
caller supplies `feature_stage_for` (default `None`), `input_feature_bundle_ref`
is a free `str` (no `SHA256_PATTERN`) never resolved against the bundle
registry, and `resolved_input_features` is never checked as a subset of
the referenced bundle.
Citation: ML plan §5.2 (7B.22-3); §3 (`input_feature_bundle_ref` =
"resolved_feature_bundle_id of PERMITTED inputs only").
Evidence: `if feature_stage_for:` in `assert_no_regime_leakage`;
`feature_stage_for: dict | None = None` in the resolver;
`input_feature_bundle_ref: str`. Repro (j):
`resolve_kmeans_protocol(..., observation_stage=AvailabilityStage.PARENT_LOCK,
resolved_input_features=("distance_to_htf_ticks",))` resolved with no map.
Suggested resolution: resolve `input_feature_bundle_ref` through the
bundle registry inside `resolve_kmeans_protocol`, derive
`feature_stage_for` from the block definitions, refuse features outside
the bundle, and pattern the ref as 64-hex.

**S5 (MINOR)** — The FUX source scans do not cover the new panel:
`tests/agents/test_ifvg_study_scans.py::_UI_SCRIPTS` lists eight scripts
and omits `ifvg_regime_panels.py` (and the R5B `ifvg_mbp1_panels.py`).
Citation: FUX-SAFE-001 / FUX-LABEL-001 / FUX §3.2.
Evidence: the reviewer replicated every scan over the panel: `PROBLEMS:
none`; widget calls used are exactly `caption, dataframe, markdown,
text_input, warning, write`.
Suggested resolution: add both panel scripts to `_UI_SCRIPTS`.

**S6 (MINOR)** — `load_sidecar_bytes` (pre-existing) verifies neither
`manifest_payload_sha256` nor the `name` argument, and it hashes a second
on-disk read rather than the bytes it returns. R6 is the first lane to
feed it a *stored, dynamic* name (`artifact.preprocessing_pipeline_ref`),
and `RegimeFitArtifact._relative_ref` is a blacklist looser than the
save-side whitelist `_SIDECAR_NAME_PATTERN`.
Citation: ML plan §5.3 P1-4; R5 safety F5 note in `store.py`.
Evidence: repro (i) — with an appended traversal entry in a copied
manifest (stale `manifest_payload_sha256`), `load_sidecar_bytes(dest,
"regime_fits", fit_id, "../../OUTSIDE_SECRET.bin")` returned
`b'outside-the-entry'`, while `load_verified_envelope` on the same entry
refused ("manifest hash mismatch"). The R6 chain is protected because
`load_regime_fit` calls `load_verified_envelope` first and `_relative_ref`
rejects `..` segments — but it accepts `""`, `"."`, `"./pipeline.joblib"`,
`"sub/pipeline.joblib"`, `"pipeline.joblib/"`, `"envelope.json"`,
`"manifest.json"` (repro (g)). TOCTOU: `data = path.read_bytes()` then
`file_sha256(path)`.
Suggested resolution: in `load_sidecar_bytes`, verify the manifest hash,
`fullmatch` `name` against `_SIDECAR_NAME_PATTERN`, and hash `data` in
memory; mirror the whitelist in `_relative_ref`.

**S7 (MINOR, design note)** — Store integrity is not authenticity: an actor
with write access to a store root who recomputes the sidecar sha and
`manifest_payload_sha256` passes every check, and `joblib.load` unpickles
the bytes.
Citation: kickoff §9 Repository boundaries / immutability (context); no
plan clause requires signing.
Evidence: repro (h2) — after a manifest-consistent re-hash of
`pipeline.joblib`, `load_regime_fit` printed `*** UNPICKLE SIDE EFFECT
EXECUTED (attacker pickle) ***` with `joblib.load calls=1`. All four
*unrehashed* tampers were refused with `joblib.load calls=0` — the hash
gate does run before unpickling. This is the pre-existing store design
(seed snapshots are pickles too); the panel never unpickles (it loads only
JSON envelopes), and `load_regime_fit` is reached only by
`persist_regime_fit`'s own verification and tests.
Suggested resolution: document the store root as a trusted boundary in the
evidence; consider an HMAC/signature or a non-pickle serialization for the
post-V1 hardening.

**S8 (MINOR, truthfulness)** — `persist_regime_fit` publishes before it
verifies: `save_or_reuse_envelope` (`os.replace` inside) runs before the
reload + `np.allclose` check; a failed check raises but leaves a complete,
manifest-valid entry that reloads later with no "verified" marker.
Citation: `regime_store.py` docstring ("a persisted fit that cannot
reproduce itself never publishes as verified"); `ACCESS_SAFETY_EVIDENCE.md`.
Evidence: repro (m) — with `np.allclose` forced false: `persist raised: the
reloaded regime pipeline does not reproduce…`, then `entry published on
disk despite failed verification? -> True`, and `load_regime_fit` on that
root succeeded.
Suggested resolution: verify before publishing, or persist a verification
sidecar that `load_regime_fit` requires; soften the wording otherwise.

**S9 (MINOR)** — `_fit_one_fold` fits preprocessing before refusing a
planned algorithm (`fit_regime_preprocessing` precedes
`assert_regime_algorithm_fittable`). The public `run_regime_protocol`
guards first, so this is reachable only through the private function.
Evidence: repro (f) — `run_regime_protocol refused:
RegimeAlgorithmUnavailableError ; preprocessing fits before refusal = 0`;
`_fit_one_fold (private) refused: … preprocessing fits before refusal =
1`. No file access is involved either way (in-memory rows).
Suggested resolution: move the assertion above the preprocessing fit.

### Verified clean (reviewer 2, evidence-backed)

- **Protected/sealed counters: zero — AFFIRMED.** `grep -n -E
  "2026-06-1[1-9]|2026-06-[23][0-9]|2026-0[7-9]-|…|June 1[1-9]|06/11"` over
  all 13 R6 files: no source-date hit. Every date literal in the set is
  synthetic: `2026-01-05` (bdate range; example), `2026-02-27` (example),
  `2026-01-13` (panel test / mbp1 fixture); the only post-June literal is
  `decided_at="2026-08-26T00:00:00Z"`, a decision timestamp from which no
  path is derived. Grep for `data/|ifvg_datasets|ifvg_experiments|databento|
  search/v1|search_test|os.environ|getenv|listdir|glob|iterdir|rglob|open(|
  read_parquet|read_csv|duckdb` over the R6 set: only `Path(root)`
  pass-throughs in `regime_store.py` and the panel's `_REPO_ROOT` `sys.path`
  insert. On disk: `find data -type f -newermt 2026-08-25` → 0 files;
  `data/ifvg_datasets/search` and `search_test` do not exist; no
  `regime_protocols|regime_fits|regime_assessments|regime_promotions`
  directory anywhere in the repo. All test roots are
  `tmp_path`/`tmp_path_factory`. No real model fit: the only fits are on
  the synthetic fixtures.
- **V1 algorithm boundary.** `grep -E "GaussianMixture|SpectralClustering|
  Nystroem|MiniBatchKMeans|torch|keras|tensorflow|gym|stable_baselines|jax|
  lightgbm|xgboost"` over the 13 files: no hits (7B.22-19). Import-time
  invariant (`_implemented == ["kmeans_v1"]`; every expansion key
  registered). `run_regime_protocol` refuses `gaussian_mixture_v1` before
  any preprocessing fit (repro (f), 0 fits).
- **No automated selection.** No loop over k/features/thresholds anywhere
  in the lane; `run_regime_protocol` takes exactly one protocol; the gates
  only append `gate_failures` and never touch `resolved_cluster_count`
  (test pins 3 centroids under `sample_adequacy` failure); silhouette is
  `silhouette_descriptive` and appears in no gate; `regime_diagnostics.py`
  exposes only `build_stability_report` and
  `transition_matrix_from_sequence` — no retrain/disable/promote consumer.
  `model_copy(update={"status": MODEL_FEATURE})` on a lawful decision is
  refused at `from_payload` (pydantic re-validates the payload, repro (c)).
- **Regime outputs cannot reach execution.** `IFVG_REGIME_CONTEXT_V1` is
  `FeatureBlockStatus.PLANNED` and
  `test_regime_feature_block_still_refuses_in_v1` passes;
  `search/pipeline.py == HEAD` (S11 reason unchanged); grep for regime
  imports outside the lane hits only `identities.py` (audit import), the
  panel, and the pipeline-tab mount; every `regime_`-string hit in
  `study/*`, `search/authorization.py`, `feature_blocks.py` is HEAD content.
- **Store discipline.** `git diff search/store.py` is +5 lines (four store
  names); all four regime stores go through
  `save_or_reuse_envelope`/`load_verified_envelope`; tamper on
  `pipeline.joblib`, `artifact.json`, `assignments.arrow`, `envelope.json`
  each refused before `joblib.load` (repro (h)); 64-hex enforcement is at
  the store layer (`fullmatch`) — `../../x`, uppercase, 63-char,
  trailing-newline, `..`, and empty ids all refused with "envelope id is
  not a valid store key" via `load_regime_protocol` (repro (l)); the panel
  never lists (no `iterdir/listdir/glob` in the script; only exact-ID
  loads).
- **UI safety.** Panel widget calls: `caption, dataframe, markdown,
  text_input, warning, write` — no `button`/`form`/`selectbox`/`toggle`, no
  `session_state` writes, no `subprocess`/`json.loads`/`yaml`; every error
  goes through `sanitize_error`; `git diff scripts/ifvg_pipeline_tab.py` is
  +4 lines (one read-only expander); `_spawn_pipeline_job(` call sites: 3
  at HEAD, 3 in the worktree; `ruff check` on the full R6 set: "All checks
  passed!" (FUX-MOD-001); all 4 R6 AppTests pass; ad-hoc FUX scans pass.
- **Frozen lanes.** `git status --short` on `src/alpha_lab/propsim`,
  `context_feature_view.py`, `context_experiment_contracts.py`,
  `context_folds.py`, `ml/model_trainer.py`: clean. `../Strategy-core`
  (HEAD `a4e3303`): `status --short` empty. `../Trade-Lab`: 27 pre-existing
  entries (18 M, 1 D, 8 ??), newest mtime `2026-07-31 14:19`; `find
  -newermt 2026-08-01` over every dirty path: nothing.
- **Declared file list / shared docs.** `git status` matches
  `FILES_TOUCHED.md` exactly; the plan/progress tree is untracked as in
  every prior release. Diff-of-diffs: the worktree diffs of
  `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
  `docs/ML_TRAINING_WORKBENCH.md` are byte-identical to
  `R1/PRE_EXISTING_DIFF.patch` modulo `index` lines — no R6 text leaked
  into the user-owned files. Both staging anchors exist at HEAD, so
  `stage_shared_docs.py` will not fail; the `ARCH_APPEND` text is accurate
  except for the S1 claim. `docs/DECISIONS.md` change is additive (D-044
  paragraph + reservation note) with the same S1 overstatement.
- **Evidence claims checked.** "Bootstrap resamples the REFERENCE fold's
  training matrix only": TRUE (`reference = sorted(fold_fits)[0]`,
  `indexed.loc[training_row_ids]`, `rng.integers(0, len(train_matrix))`,
  refit `random_state=1000+i`); test rows never enter. "No new package
  dependency": TRUE (`joblib`/`scipy` imported at HEAD in
  `logistic_model.py`/`feature_insight.py`; `pyproject.toml` pins
  `scipy>=1.12`, `scikit-learn>=1.4`). Identity audit registers the four
  regime pairs. Test runs: `test_regime_contracts.py` +
  `test_regime_service.py` + `test_regime_store.py` +
  `test_ifvg_study_scans.py` + `ifvg_search/test_identities.py` → 58
  passed; `test_ifvg_pipeline_tab.py -k regime` → 4 passed.
