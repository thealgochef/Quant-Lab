# ML, Regime Modeling, and Spectral-Clustering Contract Plan

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 items 30–34; §7B of the brief)
**Status:** Plan-only. **REVISED** per `IMPLEMENTATION_PLAN_REVISION_REQUEST.md` (P0-11 gating semantics, P1-1 scope, P1-2 grains, P1-3 proposal defaults, P1-4 portable refs, P1-6 wording). See `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`. Decision-log reservation: D-044 (supervised ladder + regime algorithm registry + spectral restrictions; **V1 active scope = prevalence / logistic / CatBoost / KMeans**).
**Module home:** `src/alpha_lab/agents/data_infra/ifvg/ml/` + test fixtures under `tests/agents/data_infra/ifvg/ml_fixtures/`.
**Dependency check (verified):** scikit-learn 1.7.0 and catboost 1.2.10 are installed; `KMeans`, `MiniBatchKMeans`, `GaussianMixture`, `SpectralClustering`, `Nystroem`, `LogisticRegression`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `scipy.optimize.linear_sum_assignment` all importable. **No new dependencies are required for V1.** No SC changes are required for the V1 ML layer (a future formula-v3 for MBP-1/volatility blocks is an SC change, carried here as planned identity only).

```
ifvg/ml/  model_protocols.py · logistic_model.py · supervised_ladder.py ·
          regime_contracts.py · regime_algorithms.py · regime_preprocessing.py ·
          regime_service.py · regime_store.py · regime_alignment.py ·
          regime_diagnostics.py · calibration_policies.py · decision_policies.py ·
          drift_monitoring.py
```

Binding constraints honored: MBP-1 only; ≤5-day real verification with synthetic ML fixtures; no automated model/feature/threshold/cluster-count selection outside a frozen charter; deep learning and RL absent from V1; fold-local preprocessing everywhere; cluster IDs nominal; direct spectral clustering training-only absent a registered OOS assignment policy; Nyström+KMeans is the production-compatible spectral approximation.

**V1 active scope (revision P1-1; V3 P1-6 boundary ruling).** The complete schema, capability registry, and spectral restrictions below are all designed in V1, but the first released workspace *implements* only: the training-prevalence reference, regularized logistic regression, the fixed CatBoost challenger, **`kmeans_v1`**, and the regime contracts/status/coverage/UI states. **V1 hardens and releases with KMeans. `minibatch_kmeans_v1`, `gaussian_mixture_v1`, `spectral_clustering_train_only_v1` (diagnostics), `nystrom_kmeans_v1`, and the expanded drift-metric set are post-V1** — delivered in the regime-expansion release that follows V1 hardening (`PHASED_DELIVERY.md`); spectral/Nyström are no longer described as part of the first release anywhere in this package. Their registry entries carry `implementation_status="planned"` with pinned parameters preserved verbatim so later activation is an implementation event, not a design change. Rationale: the core search/prop workflow does not depend on them; the effort-control objective favors the smaller V1; every fail-closed "planned/blocked" UI state is still exercised by real registry entries in V1.

**V4 terminology rule:** all authoritative text uses `resolved_regime_protocol_id` (the protocol envelope id); per-fold fit references are `regime_fit_id`. Regime feature names become `ctx_regime_{resolved_protocol_id_prefix}_…`; role/status live in `RegimePromotionDecision`, never in the protocol payload — so a promotion (or demotion) never changes any fit's numerical identity.

---

## 1. ML role taxonomy (brief §7B.1)

`RegimeRole` StrEnum: `descriptive_only`, `stratification_only`, `feature_generator`, `predictive_model`, `decision_policy`, `execution_gate_candidate`, `frozen_execution_gate`, `monitoring_only`. Every assessed regime protocol has exactly one current `RegimePromotionDecision`. Numerical `RegimeFit` artifacts are role-free and remain unchanged when promotion status changes. The promotion sequence (descriptive → stratification → feature-generator → predictive-model input → execution-gate candidate → frozen gate after full sequential replay) is enforced by status transitions in `regime_contracts.py` — a regime artifact can never become an FSM guard, trade-eligibility rule, execution gate, or prop-risk control without a separately frozen protocol plus a downstream sequential replay (the gated-replay request contract, §8).

---

## 2. Supervised model ladder (brief §7B.2) — `model_protocols.py`, `logistic_model.py`, `supervised_ladder.py`

```python
MODEL_PROTOCOL_REGISTRY = MappingProxyType({
  "reference_prevalence_v1":         ModelProtocolEntry(status=AVAILABLE, kind="reference"),
  "ifvg_context_logistic_l2_v1":     ModelProtocolEntry(status=AVAILABLE, kind="interpretable"),
  "ifvg_context_catboost_binary_v1": ModelProtocolEntry(status=AVAILABLE, kind="nonlinear_challenger"),
  "ifvg_context_gam_v1":             ModelProtocolEntry(status=PLANNED,
                                       reason="preregistered_basis_penalty_protocol_not_ratified"),
})
```

**Reference 0 — prevalence (exists).** Per-fold `training_prevalence` already flows into every prediction row; `binary_prediction_report` computes the reference Brier from it. The ladder renders it as a first-class rung with no new fit code. The prevalence reference is mandatory for every fold.

**Model A — `ifvg_context_logistic_l2_v1` (NEW).**

```python
IFVG_CONTEXT_LOGISTIC_PARAMETERS = MappingProxyType({
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 10_000, "tol": 1e-6,
    "fit_intercept": True, "class_weight": None, "random_state": 7, "n_jobs": 1,
})
LOGISTIC_PREPROCESSING_POLICY = MappingProxyType({
    "numeric_imputer": {"strategy": "median", "add_indicator": True},   # fold-local SimpleImputer
    "scaler": {"with_mean": True, "with_std": True},                    # fold-local StandardScaler
    "categorical_encoder": {"kind": "one_hot", "handle_unknown": "ignore",
                            "missing_value": MISSING_CATEGORY},         # fold-local OneHotEncoder
    "categorical_registry": CATEGORICAL_FEATURE_REGISTRY,               # reused from context_model
})

def resolve_logistic_protocol(*, ordered_features, feature_registry_hash,
                              manual_feature_overrides=None) -> ResolvedLogisticProtocol
    # mirrors resolve_context_model_protocol: resolved_hash covers protocol id, parameters,
    # preprocessing policy, categorical registry, package versions {scikit-learn, numpy, pandas}
    # + python_version, ordered/categorical features, feature_registry_hash, overrides hash,
    # threshold grid, calibration_policy="raw_probability_diagnostics_v1".

def run_logistic_fold_models(view, labeled_candidates, folds, *, features, ...) -> run
    # identical loop shape to run_context_fold_models: per valid fold, fit
    # imputer+scaler+encoder on TRAIN ONLY (sklearn Pipeline), fit LogisticRegression,
    # predict_proba on the test rows; emit the SAME prediction-row schema with
    # oos_row_id = canonical hash of {view_id, fold_index, candidate_id} — deliberately
    # model-independent so ladder rows pair exactly; permutation importance (repeats 20, seed 7).
```

**Model B — CatBoost (exists).** `ifvg_context_catboost_binary_v1` reused unchanged; CatBoost-native categorical/NaN handling is its (already-hashed) missingness policy.

**Optional Model C — GAM.** Registry entry only (`planned`); no `pygam` dependency in V1; a future protocol must pin basis/penalty (`n_splines`, `lam`) as Literals.

**Ladder runner** (`supervised_ladder.py`): one view + one `ContextFoldSet` feed every rung; before any delta is computed the runner asserts identical `oos_row_id` sets and identical `(candidate_id, target, training_prevalence)` tuples across rungs (brief §7B.2 identical-rows rule; acceptance 7B.22-1). Paired rung deltas run through the generalized `paired_cell_delta_report` (identical-OOS-ids gate).

**Feature-selection policy (brief §7B.3) — enforcement:** permitted = preregistered block ablations, preregistered minimal bundles, regularization-strength comparisons under a frozen protocol, manual registered include/exclude lists entering immutable identity (`manual_override_registration_id` on the bundle). Prohibited behaviors are code-level refusals:
- Registries are `MappingProxyType`; `resolve_*_protocol` rejects overrides touching features/params/thresholds.
- `run_supervised_ladder(protocols=...)` accepts protocol **ids** only; unknown id → `ValueError`.
- `ProhibitedSelectionError` raised whenever a request enumerates >1 value for a threshold, cluster count, calibrator, or feature subset outside a frozen charter carrying owner ratification (acceptance 7B.22-15).
- No module imports torch/keras/gym; a repo test asserts the absence of deep-learning/RL controls (acceptance 7B.22-19).

Primary metrics (existing `binary_prediction_report`): Brier, Brier skill vs the fold-local prevalence reference, log loss, AUC when defined, calibration intercept/slope, reliability bins, coverage, fold stability, trading-day block uncertainty. Secondary descriptive diagnostics: grouped permutation importance (block-level roll-up of the existing `_permutation_importance`); SHAP is `planned` and excluded from V1 (owner decision 32).

---

## 3. Regime contracts (brief §7B.4) — `regime_contracts.py`

```python
class RegimeStatus(StrEnum):
    PLANNED; DESCRIPTIVE_ONLY; STRATIFICATION_READY; FEATURE_ELIGIBLE; MODEL_FEATURE
    EXPERIMENTAL; BLOCKED_NO_OOS_ASSIGNMENT; BLOCKED_INSUFFICIENT_COVERAGE; SUPERSEDED

class ObservationGranularity(StrEnum):      # Amendment P1-B — the panel grain is in the ACTUAL schema
    CANDIDATE_STAGE_ROW = "candidate_stage_row"   # V1 default — one row per exact candidate at a stage anchor
    DECISION_ROW = "decision_row"                 # joins on eligible_decision
    CONTEXT_BAR_PANEL = "context_bar_panel"       # completed point-in-time context bars; interval is a field
                                                  # (1m is expressed as panel_interval_seconds=60 — there is
                                                  #  no separate CONTEXT_BAR_1M enum value)

class RegimeProtocolPayload(FrozenContract):   # V3 P1-3: the ALGORITHM PROTOCOL — no role/status,
                                               # no fold/fit state. Hashed → resolved_regime_protocol_id.
    algorithm_key: str                      # LOGICAL registry key: "kmeans_v1", "gaussian_mixture_v1", …
                                            # (V3 P1-2 — never confused with the resolved identity)
    algorithm_version: str
    input_feature_bundle_ref: str           # resolved_feature_bundle_id of PERMITTED inputs only
    resolved_input_features: tuple[str, ...]
    observation_granularity: ObservationGranularity
    panel_interval_seconds: int | None      # P1-B: required iff CONTEXT_BAR_PANEL (owner-registered,
                                            # e.g. 300 or 900); must be None for the other grains
    panel_source_artifact_id: str | None    # P1-B: completed-bar panel source; None unless panel grain
    panel_as_of_policy_id: str | None       # P1-B: completed-bars-only as-of rule + candidate-assignment
                                            # policy; None unless panel grain
    observation_stage: AvailabilityStage
    missingness_policy: str                 # "median_impute_with_indicator_v1"
    winsorization_policy: str               # "clip_p01_p99_train_fitted_v1" | "none"
    scaler_policy: str                      # "standard_scaler_v1"
    dimensionality_reduction_policy: str    # "none" | "pca_fixed_components_v1" (planned)
    kernel_or_affinity_policy: str | None
    cluster_count_policy: Literal["fixed_k", "inner_train_only_selection"]
    resolved_cluster_count: int
    random_seed: Literal[7] = 7
    initialization_policy: str
    out_of_sample_assignment_policy: str    # "centroid_predict_v1" | "gmm_posterior_v1"
                                            # | "nystrom_transform_kmeans_predict_v1"
                                            # | "none_training_only" | "surrogate_logistic_v1"
    cluster_label_alignment_policy: Literal["centroid_min_distance_hungarian_v1"]
    fit_scope: Literal["per_training_fold"] # fold-local only
    software_versions: Mapping[str, str]    # immutable-mapping wrapper (CS §0.3)
    formula_version: str

class RegimeProtocolEnvelope(FrozenContract):
    resolved_regime_protocol_id: str        # = canonical_contract_sha256(payload)
    payload: RegimeProtocolPayload

class RegimeFitPayload(FrozenContract):     # V3 P1-3: ONE fold's fit under one protocol
    resolved_regime_protocol_id: str
    source_artifact_ids: tuple[str, ...]
    fold_index: int
    fit_start: str | None; fit_end: str | None
    training_row_ids_hash: str

class RegimeFitEnvelope(FrozenContract):
    regime_fit_id: str
    payload: RegimeFitPayload

class RegimeCapabilityAssessment(FrozenContract):
    resolved_regime_protocol_id: str
    regime_fit_ids: tuple[str, ...]
    fold_set_id: str
    coverage: RegimeCoverageReport            # rows assigned/typed-null, per-fold + OOS coverage
    occupancy: Mapping[int, float]            # canonical ids
    stability: RegimeStabilityReport          # bootstrap AMI, persistence, transitions, recurrence
    oos_assignment_available: bool
    minimum_cluster_occupancy_gate: float     # the gate values applied (proposed_protocol_default)
    minimum_assignment_confidence_gate: float | None
    gates_passed: bool; gate_failures: tuple[str, ...]

class RegimePromotionDecision(FrozenContract):      # V3 P1-3: status changes WITHOUT implying the
    resolved_regime_protocol_id: str                # numerical fit changed
    role: RegimeRole
    status: RegimeStatus                            # descriptive → stratification → feature-eligible → model-feature
    capability_assessment_ref: str
    owner_ratification_ref: str | None              # required before FEATURE_ELIGIBLE (P1-3 proposal stamps)
    decided_at: str

# Panel-grain validation (P1-B): CONTEXT_BAR_PANEL requires an owner-registered
# panel_interval_seconds, a panel_source_artifact_id, completed point-in-time bars only, and a
# candidate-assignment policy; for CANDIDATE_STAGE_ROW / DECISION_ROW all three panel fields must
# be None — any other combination fails validation. Tests: 5m panel; 15m panel; candidate-stage;
# decision-row; invalid panel-field combinations refused; point-in-time assignment from a panel fit
# to candidate stages.

class RegimeFitArtifact(FrozenContract):    # the materialized numerical artifact for one RegimeFitEnvelope
    regime_fit_id: str; fold_index: int
    fitted_parameter_payload_hash: str      # canonical hash of centroids/means/covs/Nyström arrays
    preprocessing_pipeline_ref: str         # store path of the persisted fitted pipeline
    fold_local_cluster_ids: tuple[int, ...]
    centroid_or_component_descriptors: ImmutableMap[int, tuple[tuple[str, float], ...]]
    training_row_ids_hash: str; training_row_count: int
    inertia_or_loglik: float
    software_versions: ImmutableMap[str, str]

class RegimeAssignment(FrozenContract):     # one per (row, fold); persisted as a frame
    resolved_regime_protocol_id: str
    regime_fit_id: str
    row_id: str; fold_index: int; partition: Literal["train", "test"]
    fold_local_cluster_id: int | None
    canonical_reporting_cluster_id: int | None   # reporting only; never a model input
    distances: tuple[float, ...] | None          # kmeans/nystrom
    probabilities: tuple[float, ...] | None      # gmm
    assignment_margin: float | None; assignment_entropy: float | None
    log_density: float | None; outlier_score: float | None
    valid: bool
    missing_reason: str | None
    # typed reasons: source_feature_missing | fold_invalid | training_only_algorithm
    #              | below_confidence_floor | coverage_gap   (rows preserved — 7B.22-10)

class RegimeCoverageReport(FrozenContract):
    rows_total: int; rows_assigned: int; rows_typed_null: ImmutableMap[str, int]
    per_fold_coverage; oos_assignment_coverage
    per_cluster_occupancy                    # canonical ids
    coverage_gates_passed: bool; gate_failures: tuple[str, ...]

class RegimeStabilityReport(FrozenContract):
    bootstrap_stability                      # 50 seeded refits, aligned-AMI + per-cluster agreement
    temporal_persistence; transition_matrix; fold_to_fold_recurrence
    separation                               # centroid separation; optional silhouette/CH/DB (descriptive only)
    semantic_descriptors
    stability_gates_passed: bool; gate_failures: tuple[str, ...]
```

---

## 4. Regime algorithm registry (brief §7B.8) — `regime_algorithms.py`

```python
REGIME_ALGORITHM_REGISTRY = MappingProxyType({
  "kmeans_v1": implementation_status="implemented" (V1 ACTIVE), oos=True,
      pinned={"init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4,
              "random_state": 7, "algorithm": "lloyd"},
      outputs=(cluster id, distance to every centroid, assigned distance, margin d2−d1),

  "minibatch_kmeans_v1": implementation_status="planned" (regime-expansion release), oos=True,
      pinned={"batch_size": 1024, "n_init": 10, "max_no_improvement": 100,
              "random_state": 7, "reassignment_ratio": 0.01},
      constraints={"use_only_when_rows_gt": 100_000},     # deterministic seed + fixed batch protocol

  "gaussian_mixture_v1": implementation_status="planned" (regime-expansion release), oos=True,
      pinned={"covariance_type": "diag", "reg_covar": 1e-6, "max_iter": 200, "n_init": 5,
              "init_params": "kmeans", "random_state": 7, "tol": 1e-3},
      outputs=(component id, probabilities, entropy, log density),
      # covariance policy explicit and bounded — "diag" only

  "spectral_clustering_train_only_v1": implementation_status="planned" (regime-expansion release,
      as training-only diagnostics), oos=False,
      pinned={"affinity": "nearest_neighbors", "n_neighbors": 15, "eigen_solver": "arpack",
              "assign_labels": "kmeans", "random_state": 7, "eigen_tol": "auto"},
      constraints={"max_training_observations": 20_000,
                   "max_affinity_memory_bytes": 512 * 2**20,
                   "min_observation_grain": "candidate_stage_row_or_completed_context_bar",
                   "train_fold_only_affinity": True},     # never tick/MBP-event scale
      default_status_on_fit=BLOCKED_NO_OOS_ASSIGNMENT,    # or DESCRIPTIVE_ONLY when declared

  "nystrom_kmeans_v1": implementation_status="planned" (regime-expansion release), oos=True,
      pinned={"kernel": "rbf", "gamma": <spec field>, "n_components": <spec field>,
              "nystroem_random_state": 7,                  # component sampling seed — identity field
              "kmeans": {"n_init": 10, "max_iter": 300, "random_state": 7, "algorithm": "lloyd"}},
      identity_fields=("kernel", "gamma", "n_components", "nystroem_random_state", "kmeans_config"),
      # pipeline: fold-fitted scaler → fold-fitted Nyström map → fold-fitted KMeans
      #           → transform + assign validation/test rows deterministically

  "surrogate_assignment_logistic_v1": implementation_status="planned", oos=True,
      constraints={"requires_explicit_owner_authorization": True,
                   "secondary_to": "nystrom_kmeans_v1",
                   "persist": ("surrogate_algorithm", "training_data_hash",
                               "training_accuracy", "per_class_uncertainty", "identity")},
})
```

The direct-spectral restrictions and the Nyström design below remain **explicit and binding** even though their implementations are deferred — they are part of the intended architecture (revision P1-1), and every registry entry above ships in V1 with fail-closed refusal + UI-state tests regardless of implementation status.

**Direct-spectral restrictions — V1 enforcement vs deferred implementation (Amendment P1-C, acceptance 7B.22-6):**

*What V1 actually ships:* capability-registry entries and **fail-closed planned/blocked behavior only**. **No direct spectral or Nyström fit implementation is callable in V1** — a request for a planned algorithm is refused with the correct status/reason; training-only spectral results cannot enter a predictive bundle (the bundle resolver refuses `IFVG_REGIME_CONTEXT_V1` whenever the referenced regime model lacks OOS test-row coverage — the regime feature materializer reads only `partition="test", valid=True` assignment rows); and the algorithm identities remain distinct in the registry. Those three behaviors are the V1 test checks.

*What the frozen restrictions bind on the future implementation* (regime-expansion release; the numerical tests land with it): the fit service computes the affinity on training-fold rows only (test rows are never passed in); it enforces the observation cap and memory bound before fitting; it records affinity type, kernel parameters, neighbor count, eigen-solver, label-assignment method, seed, and tolerances into the artifact; it emits **no** test-partition assignments (test rows carry `valid=False, missing_reason="training_only_assignment"`), and the spec status is forced to `BLOCKED_NO_OOS_ASSIGNMENT` (or `DESCRIPTIVE_ONLY`) unless a registered surrogate policy is named; `nystrom_kmeans_v1` and `spectral_clustering_train_only_v1` remain distinct algorithm ids with distinct pinned payloads whose envelope ids are verified on save→reload (acceptance 7B.22-18); the two-ring synthetic fixture proves behavioral distinctness from KMeans at that release.

---

## 5. Observation grains, inputs, preprocessing, gates

### 5.1 Grains (brief §7B.5; revision P1-2)

V1 default: `CANDIDATE_STAGE_ROW` at `observation_stage=ENTRY_DECISION` — rows are the existing `CandidateFeatureView` frame. **Its sample-size limitation is explicit:** the accepted development window contains on the order of only ~10² entry candidates, which may be too sparse for stable clustering; candidate-stage fits that fail the sample-adequacy gate (below) stay `DESCRIPTIVE_ONLY`/`BLOCKED_INSUFFICIENT_COVERAGE` rather than being promoted because fitting merely succeeded.

The architecture therefore also supports a **completed-market-context panel grain**: `CONTEXT_BAR_PANEL` over completed 5m or 15m context bars (or another owner-approved interval — both timeframes already exist in the replay-chart resample set). The fold-fitted regime model may train on that larger point-in-time panel and **assign the frozen current regime to candidate stages** at their as-of instants; the panel interval enters the spec identity (`observation_granularity=CONTEXT_BAR_PANEL`, `panel_interval_seconds`). `DECISION_ROW` joins on decision ids; the 1m panel remains a planned builder. Regime models are never fitted at tick or MBP-event scale (registry constraint).

**Sample-adequacy gate** (applies to every grain): minimum training observations per fold (proposed default ≥ 300 for panel grains, ≥ 150 for candidate-stage), minimum per-cluster occupancy (§5.4), and both-partition coverage — all recorded in `RegimeCoverageReport`; failure blocks promotion, never silently shrinks k.

### 5.2 Inputs (brief §7B.6)

Permitted inputs are features from blocks whose `availability_stage` ≤ the spec's `observation_stage` (point-in-time by construction), drawn from the brief's permitted list (volatility, compression, trend/path efficiency, session state, intensity, spread, MBP-1 imbalance/OFI/depletion/aggression, top-of-book state). Prohibited inputs are refused by `assert_no_regime_leakage(resolved_input_features)` raising `RegimeLeakageError` for any column in:

```python
PROHIBITED_REGIME_INPUTS = ("binary_target", "label", "gross_r", "net_r", "mfe_r", "mae_r",
    "realized_ticks", "resolution", "resolution_ts_utc", "bars_after_entry_to_resolution",
    any "payout"/"breach"/"passed" column,
    any feature whose block availability_stage > the spec observation_stage)
```

(acceptance 7B.22-3).

### 5.3 Fold-local preprocessing (brief §7B.7) — `regime_preprocessing.py`

Fixed per-fold pipeline: `SimpleImputer(median, add_indicator)` → optional train-fitted winsorizer (p01/p99 clips) → `StandardScaler` → optional Nyström kernel map. Every step is fitted on training-fold rows only; the fitted pipeline is part of the artifact, persisted dual-format: (a) canonical JSON parameter payload (medians, clip bounds, scaler mean/scale, Nyström components) whose hash is `fitted_parameter_payload_hash`; (b) a joblib binary for reuse. `_persist_and_verify_regime_fit` saves both, reloads, re-transforms a held fixture slice, and asserts `np.allclose` between reloaded output and fit-time output. The fit API accepts a `fold: IfvgContextFoldDefinition` and slices internally — it cannot receive a pooled frame (a test asserts this), so fitting on full data / pooled OOS / future rows / prop outcomes / sealed data is unrepresentable.

**Portable artifact references (revision P1-4):** `RegimeFitArtifact.preprocessing_pipeline_ref` — and every fitted-artifact reference in this lane — is a **manifest-relative path**, never an absolute filesystem path, accompanied by `{file_sha256, byte_size, artifact_schema_version, software_versions, fitted_parameter_payload_hash}`. Reload verification resolves against the artifact directory root and the checksums; it must not depend on the original machine's paths. A relocation test (copy the artifact directory, reload, re-transform, `np.allclose`) enforces portability.

### 5.4 Cluster count, nominal IDs, alignment, gates (brief §7B.9–7B.12)

- **Cluster count:** `fixed_k` policy (proposed k=3 primary; k∈{4,5} only as separately registered specs — no selection). `inner_train_only_selection` is schema-supported (planned): frozen selection metric + tie-break, per-fold selected count stored; the selector API receives inner-train features only — test folds, candidate outcomes, strategy expectancy, and prop metrics are unavailable to it by construction. Cluster count is never chosen by trading/prop results.
- **Proposal stamps (revision P1-3):** every scientific default in this lane — the KMeans-first algorithm choice, k=3, the ≥25-rows/5%-occupancy gates, the aligned-AMI ≥ 0.5 threshold, the sample-adequacy minimums, and the spectral/GMM implementation timing — is recorded in the spec/registry as `proposed_protocol_default` with `owner_ratification_required_before_feature_eligible=True`. No cluster family becomes a predictive feature or execution gate merely because the engineering implementation exists; promotion requires the owner's `OwnerDecisionEvidenceRef` on the regime grain + cluster policy (charter authorization bundle).
- **Nominal IDs:** artifacts persist `fold_local_cluster_id`; `regime_alignment.py::align_reporting_labels` implements `centroid_min_distance_hungarian_v1` — reference = first valid fold's centroids in scaled space, later folds matched via `scipy.optimize.linear_sum_assignment`, ties broken by ascending fold-local id; outputs `canonical_reporting_cluster_id` + `cluster_semantic_summary` (top-|z| descriptors). Alignment runs in the reporting layer only, after predictions exist; a test proves canonical relabeling never changes any prediction hash (acceptance 7B.22-9). The UI never implies `regime 2 > regime 1`.
- **Promotion gates** (to `FEATURE_ELIGIBLE`): min occupancy (default 5 % of training rows AND ≥25 rows per cluster per fold), OOS assignment exists, source coverage meets the block's requirements, leakage validator passed, bootstrap-stability aligned-AMI ≥ 0.5 (advisory default; owner-tunable), interpretable point-in-time descriptors present. Gate failure keeps the status at `STRATIFICATION_READY` / `DESCRIPTIVE_ONLY` / `BLOCKED_INSUFFICIENT_COVERAGE`. No single internal clustering score (silhouette / Calinski-Harabasz / Davies-Bouldin — kept as descriptive fields) can promote.

### 5.5 Regime research uses (brief §7B.13–7B.15) — comparison-class mapping

| Question | Comparison class | Requirement |
|---|---|---|
| Strategy behavior by regime | `cohort_descriptive` (regime filter) | stratified report; no counterfactual claim |
| Model skill with regime feature | `feature_only` (bundle ± `IFVG_REGIME_CONTEXT_V1`) | identical rows/labels/folds (7B.22-11) |
| Separate models per regime | `cohort_model` | pooled baseline retained; invalid regime folds reported (`insufficient_regime_partition`) |
| Config works only in one regime | stratified frontier views | descriptive |
| Regime-specific executable profile | `strategy_counterfactual` | **full sequential replay** (7B.22-12) |
| Prop metrics by regime | stratified prop reports | descriptive |
| Occupancy drift over time | drift monitoring (§9) | monitoring-only |

Controlled supervised comparisons supported: baseline model / baseline + hard regime ID / baseline + distances-probabilities / baseline + MBP-1 / baseline + MBP-1 + regime outputs — all nonchanged dimensions identical. A regime feature earns inclusion via incremental OOS Brier-skill/calibration improvement, fold stability, coverage, and absence of one-regime/one-day concentration — never because one regime has favorable in-sample expectancy. Distances/probabilities are preferred over the hard ID alone; a hard cluster ID may enter CatBoost as a categorical feature only when the assignment is point-in-time and OOS.

---

## 6. Calibration policies (brief §7B.16) — `calibration_policies.py`

```python
CALIBRATION_POLICY_REGISTRY = MappingProxyType({
  "raw_probability_diagnostics_v1": AVAILABLE,   # existing id; V1 default and only executable policy
  "platt_sigmoid_train_fold_v1":    PLANNED,     # fit inside each training fold on a recorded
                                                 # calibration subset; apply once to that fold's test rows;
                                                 # never fit on pooled OOS and score the same rows
  "isotonic_train_fold_v1":         PLANNED,     # requires calibration-subset support n>=200 per fold
})
```

Raw probability diagnostics remain mandatory regardless of calibrator. Calibrator type is a `model_protocol` dimension, so a calibration change is a `model_protocol` comparison with `requires_model_refit=True`; the fit service accepts only train-fold rows for calibrator fitting (acceptance 7B.22-14).

---

## 7. Decision policies (brief §7B.17) — `decision_policies.py`

Logical registry keys and resolved identities are distinct:

```python
DECISION_POLICY_REGISTRY = MappingProxyType({
  "none_diagnostic_only_v1":         DecisionPolicyRegistryEntry(status=AVAILABLE),
  "fixed_probability_threshold_v1": DecisionPolicyRegistryEntry(status=PLANNED),
  "expected_r_threshold_v1":        DecisionPolicyRegistryEntry(status=PLANNED),
  "abstention_band_v1":              DecisionPolicyRegistryEntry(status=PLANNED),
  "top_n_per_day_v1":                DecisionPolicyRegistryEntry(status=PLANNED),
  "confidence_margin_v1":            DecisionPolicyRegistryEntry(status=PLANNED),
  "regime_conditioned_threshold_v1": DecisionPolicyRegistryEntry(status=PLANNED),
})

class DecisionPolicyPayload(FrozenContract):
    decision_policy_key: str
    parameters: ImmutableMap[str, Any]
    resolved_regime_protocol_id: str | None
    rejected_candidate_policy: RejectedCandidatePolicy | None
    rejected_candidate_policy_ratification_ref: str | None
    requires_frozen_model: bool
    requires_model_gated_sequential_replay: bool
    produces_new_trade_stream_hash: bool
    # none_diagnostic_only_v1 => policy None and flags False;
    # every execution-affecting policy => ratified rejection policy and flags True.

class DecisionPolicyEnvelope(FrozenContract):
    resolved_decision_policy_id: str
    payload: DecisionPolicyPayload

class WalkForwardModelScheduleEntry(FrozenContract):
    date_from: str
    date_to: str
    frozen_model_fit_id: str

class WalkForwardModelSchedulePayload(FrozenContract):
    entries: tuple[WalkForwardModelScheduleEntry, ...]
    coverage_policy_id: str

class WalkForwardModelScheduleEnvelope(FrozenContract):
    resolved_model_schedule_id: str
    payload: WalkForwardModelSchedulePayload
```

A model-gated replay request references only `resolved_decision_policy_id` and `resolved_model_schedule_id`. Candidate rejection is a strategy semantic. Pipeline stage S11 remains blocked until one `RejectedCandidatePolicy` is owner-ratified and sequential golden tests pass. A threshold table alone can never establish execution or prop value.

**ML/prop separation:** the supervised/regime target whitelist is `("binary_target",)`. Payout/pass/breach outcomes cannot be model targets. The hierarchy remains market/setup evidence → trade-quality model → frozen sequential decision policy → executed trade stream → prop realization.

## 8. Drift monitoring (brief §7B.19) — `drift_monitoring.py`

`DriftReport` (frozen; `role: Literal["monitoring_only"]`): explicit reference/current windows + sample counts; per-feature PSI / KS / Wasserstein (scipy.stats); prediction-distribution drift; rolling Brier + calibration slope/intercept; regime occupancy + transition drift; session/direction mix drift; funnel-conversion drift; payoff compression; loss expansion; opportunity-rate change; MBP-1 source-coverage drift; descriptive change-point indicators. Stored immutably. **The module exports report builders only — there is no consumer API that retrains, disables, or promotes anything from a drift alarm**; V1 automation is structurally impossible.

---

## 9. Verification fixtures (brief §7B.21) — `tests/agents/data_infra/ifvg/ml_fixtures/`

All `numpy.random.default_rng(7)`-seeded and deterministic. (Per the revised P1-1 scope: fixtures 1, 2, and the KMeans arm of 4 ship in V1; fixture 3 and the GMM/Nyström arms of 4 ship with the regime-expansion release.)

1. `synthetic_supervised.py::class_balanced_supervised_fixture(n=400)` — two informative Gaussian features + two noise features + one categorical, exact 50/50 classes, synthetic candidate/setup/trading-day ids shaped like the fold-input schema (so `build_context_folds` and both ladder rungs run end-to-end on it). Proves ladder parity, fold-local imputer/scaler behavior, oos_row_id pairing.
2. `synthetic_clusters.py::known_cluster_fixture(k=3, n=600)` — well-separated blobs with known memberships. Proves KMeans/GMM determinism, occupancy gates, Hungarian alignment stability (permuted fold order → stable canonical ids), margins/entropy.
3. `synthetic_spectral_graph.py::two_ring_fixture(n=800)` — concentric rings where KMeans fails and spectral/Nyström succeed. Proves the algorithms are genuinely distinct (guards mislabeling) and exercises the affinity/memory-bound refusals.
4. `synthetic_oos_assignment.py::oos_assignment_fixture()` — train/test blob split with injected missing rows. Proves deterministic OOS transform+assign for kmeans/gmm/nystrom, typed-null preservation, and that spectral-train-only emits no test assignments.

**Five-day real mini-run:** consumes the backend's `VerificationReplayPolicy` (≤5-day allowlist) — this layer never constructs real paths itself. It runs the full chain (artifact pair → bundle view → labels → folds → ladder → one kmeans_v1 regime fit → stores → UI states) and **proves**: pipeline execution, fold-local fitting, artifact identity, save/reload/reuse, missingness handling, OOS assignment behavior, UI rendering including planned/blocked states, and safe failure states. It **never proves model or regime quality** — the mini-run report pins `{"model_quality_claim": "none", "full_pipeline_not_run": true}`, and verification-fit artifacts live under the isolated test namespace. On ≤5 days most folds are invalid (`insufficient_train_candidates`) — itself a verified safe-failure state.

---

## 10. ML dashboard requirements (brief §7B.20) — surfaces the frontend renders

Within each compatible study (rendering specs in `IMPLEMENTATION_PLAN.md` §7):
- **Model ladder**: prevalence / logistic / CatBoost (/ GAM planned) metrics on identical rows/folds.
- **Feature-bundle matrix**: each block, source status, coverage, model-availability.
- **Regime model card**: algorithm, role/status, input bundle, grain, fit window/fold, cluster count, OOS assignment policy, coverage, min occupancy, stability, artifact identity.
- **Regime diagnostics**: occupancy, centroid/component profiles, distance/probability distributions, confidence/entropy, regime timeline, transition matrix, strategy/model/prop metrics by regime.
- **Spectral warning states** (mandatory text): `Training-only exploratory clustering — no OOS assignment; cannot enter a predictive feature bundle` or `OOS assignment supplied by <registered policy>`.
- **Explainability**: grouped block importance first; individual-feature views secondary with small-sample/development-data warnings.
- **Drift**: current vs reference distributions with explicit windows and sample counts.

---

## 11. Acceptance mapping (brief §7B.22 → designed artifact/test)

| # | Criterion | Contract / module | Test |
|---|---|---|---|
| 1 | Ladder on identical rows/folds | `run_supervised_ladder` assertion | ladder parity on fixture 1 |
| 2 | Fold-local preprocessing | train-only pipelines (`regime_preprocessing`, `logistic_model`) | fit-API signature test + leak test |
| 3 | No target/outcome/future regime inputs | `assert_no_regime_leakage` | leakage validator unit test |
| 4 | Deterministic OOS KMeans | pinned `kmeans_v1` | double-run hash equality (fixture 4) |
| 5 | Deterministic GMM probs/entropy OOS | pinned `gaussian_mixture_v1` | fixture 4 (GMM arm — **regime-expansion release**; registry refusal tested in V1) |
| 6 | Spectral blocked from test bundles without registered OOS policy | forced status + bundle-resolver refusal | resolver refusal test (V1 — status-driven, implementation-independent) |
| 7 | Nyström+KMeans OOS deterministic | `nystrom_kmeans_v1` | fixtures 3+4 (**regime-expansion release**; identity fields designed in V1) |
| 8 | Cluster count fixed or inner-train-only | Literal policy + selector API | selector-cannot-see-test test |
| 9 | Nominal IDs; alignment never alters inputs | reporting-only alignment | prediction-hash invariance |
| 10 | Typed missing regime rows | `RegimeAssignment.missing_reason` | fixture 4 |
| 11 | Regime-feature deltas identical rows/labels/folds | `feature_only` equalities | compatibility test |
| 12 | Regime-specific strategy → new sequential profile | dimension flags | computation-path test |
| 13 | Decision policies → full gated replay first | `DecisionPolicyPayload` replay-required flags + request contract | path derivation test |
| 14 | Calibration train-fold-only | registry rules + fit API | calibrator-input test |
| 15 | No automated selection outside a frozen charter | `ProhibitedSelectionError` + frozen registries | refusal tests |
| 16 | 5-day + synthetic fixtures only | fixtures + policy consumption | allowlist audit test |
| 17 | Full pipeline via operator UI only | request-emission only; launch owned by pipeline UI | orchestrator gate test |
| 18 | Distinct spectral/Nyström/KMeans/GMM identities | distinct algorithm ids + embedded verification | V1: registry-identity distinctness test; expansion release: two-ring behavioral mislabel test |
| 19 | No DL/RL controls in V1 | absent imports/registry entries | registry scan test |
| 20 | Every ML insight links to rows/folds/artifacts/uncertainty | `evidence_links` + run/fold/artifact hashes | report-linkage test |

---

## 12. V1 implemented vs planned (schemas-only)

**Implemented in V1** (code + tests + the 5-day mini-run; revised scope per P1-1): all `study/` contracts (see `DELTA_TAXONOMY.md`); the six available feature blocks + frozen tier bundles + partition assertions; `Mbp1SourceContract` schema + guard tests; the logistic protocol + supervised ladder; **`kmeans_v1` only** among clustering algorithms; regime contracts/store/alignment/coverage/status/UI states; calibration + decision-policy registries (baseline diagnostic policy executable); the **core** drift report builders (feature/prediction/calibration distributions + regime occupancy where a regime exists); synthetic fixtures 1 (supervised), 2 (known clusters), and 4 (OOS assignment, KMeans lanes).

**Post-V1 regime-expansion release (V3 P1-6: implementation deferred beyond V1 hardening; schemas, registry entries, pinned parameters, and restrictions all designed in V1):** `minibatch_kmeans_v1`, `gaussian_mixture_v1`, `spectral_clustering_train_only_v1` diagnostics, `nystrom_kmeans_v1`, fixture 3 (two-ring spectral graph) + the GMM/Nyström arms of fixture 4, and the expanded drift-metric set (funnel-conversion, payoff-compression, MBP-1 coverage drift). V1 hardening and the operator run **do not** wait on any of this — availability is capability-gated (`CONTRACTS_AND_SCHEMAS.md` §7).

**Release-scoped availability:** `IFVG_ORDER_FLOW_MBP1_V1` is planned through R5 and is implemented/activated in V1 at mandatory R5B as an offline research-only block. The following remain planned after V1 unless separately released: VOLATILITY, REGIME-feature, KEY_LEVEL, and EXECUTION_LIQUIDITY materializers; GAM; Platt/isotonic calibration; non-baseline decision-policy execution; surrogate spectral assignment; the 1m context-panel builder; and inner-train cluster-count selection. S11 remains blocked until `RejectedCandidatePolicy` is owner-ratified and sequential golden tests pass.

**Publication wording (revision P1-6):** every ML/regime result surface carries the development-only badge, and any selected configuration influenced by model or regime evidence is labeled a **development exploratory** result — the publishable-representative pathway requires the separately frozen outer evaluation protocol (see `IMPLEMENTATION_PLAN.md` §8).

**Flagged assumptions:** (a) v2 record IDs are content-derived uuid5 — confirmed for the SC namespace helpers; a replay-determinism parity test must land before cross-run population deltas ship; (b) `VerificationReplayPolicy` is backend-owned; this layer consumes only its allowlist + authorize-before-path behavior; (c) risk/prop/payout/portfolio identity contracts and the trade-stream-hash definition are backend-owned (`none_*` baselines here); (d) the propsim report schema is treated as an opaque payload by the delta layer.
