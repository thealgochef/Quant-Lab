# Delta Taxonomy — Universal Study-Cell Identity, Dimension Registry, Comparison Contract, Feature Blocks

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 items 23–26; §7A of the brief)
**Status:** Plan-only. **REVISED** per the first revision (P0-9/10/11/19, P1-5/6), Amendment V2 (P0-B/C/D/G, P1-E/G), **the V3 compact patch (P0-1 complete Payload/Envelope + key/resolved conversion of Comparison, DeclaredContrast, FeatureBlock, FeatureBundle; P0-6 concrete strategy-only `DataLineagePayload`; P1-4 window-trigger semantics)**. See `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`. Decision-log reservations: D-042, D-043 (as amended).

**Identity-payload convention (Amendment P0-C):** every ID-producing contract in this layer follows the Payload/Envelope pattern defined in `CONTRACTS_AND_SCHEMAS.md` §0 — the hashed payload never contains its own derived ID, artifact/manifest hashes, display metadata, annotations, or attempt metadata. The identity-projection audit test covers every contract in this document.
**Module homes:** `src/alpha_lab/agents/data_infra/ifvg/study/` and `src/alpha_lab/agents/data_infra/ifvg/features/`. The M0–M3 lane modules are **not modified** — the new lane is additive and the tier registry stays frozen.

```
ifvg/study/     dimension_contracts.py · study_cell.py · comparison_contracts.py ·
                computation_path.py · delta_outputs.py · population_delta.py ·
                funnel_delta.py · contrasts.py
ifvg/features/  feature_blocks.py · feature_bundles.py · mbp1_source_contract.py ·
                bundle_feature_view.py
```

---

## 1. Experiment-dimension registry (brief §7A.3)

### 1.1 `study/dimension_contracts.py`

```python
class DimensionType(StrEnum):        # all 17 classes required by the brief
    DATA_LINEAGE; MARKET_UNIVERSE; STRATEGY_PROFILE; OBSERVATION_COHORT
    LABEL_POLICY; FEATURE_BLOCK; MODEL_PROTOCOL; DECISION_POLICY
    EXECUTION_POLICY; COST_POLICY; RISK_POLICY; PROP_CONTRACT
    PAYOUT_POLICY; PORTFOLIO_POLICY; VALIDATION_PROTOCOL; STRESS_SCENARIO
    ENGINEERING_PROTOCOL

class DimensionCapabilityStatus(StrEnum):    # mirrors the fail-closed ProfileCapabilityStatus pattern
    AVAILABLE; PLANNED; BLOCKED_MISSING_SOURCE; BLOCKED_OWNER_DECISION; EXPERIMENTAL; SUPERSEDED

class OwnerRatificationStatus(StrEnum): RATIFIED; PENDING; NOT_REQUIRED

class ExperimentDimensionSpec(FrozenContract):
    dimension_id: str                    # e.g. "label_policy.candidate_static_r1_v2"
    dimension_type: DimensionType
    human_name: str
    technical_key: str                   # the StudyCellIdentity field it binds to
    description: str
    baseline_value: str
    allowed_values: tuple[str, ...]
    value_schema: str                    # dotted contract-class name or JSON-schema hash
    capability_status: DimensionCapabilityStatus
    capability_reason: str | None = None
    owner_ratification_status: OwnerRatificationStatus
    # required-computation flags
    requires_full_strategy_replay: bool
    requires_feature_materialization: bool
    requires_label_recomputation: bool
    requires_model_refit: bool
    requires_model_gated_sequential_replay: bool
    requires_cost_recomputation: bool
    requires_prop_resimulation: bool
    requires_bootstrap_resimulation: bool
    # population/order effect flags
    can_change_setup_population: bool
    can_change_candidate_population: bool
    can_change_trade_order: bool
    can_change_account_state: bool
    can_change_label_meaning: bool
    compatible_dimensions: tuple[str, ...] = ()
    incompatible_dimensions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    identity_effect: Literal["new_study_cell", "annotation_only"]

EXPERIMENT_DIMENSION_REGISTRY: MappingProxyType[str, ExperimentDimensionSpec]

def resolve_dimension(dimension_id) -> ExperimentDimensionSpec   # unregistered → ValueError (fail closed)
def assert_dimension_usable(spec) -> None
    # DimensionBlockedError unless AVAILABLE (EXPERIMENTAL only with an explicit allow_experimental flag)
```

The registry **fails closed** when an unknown, blocked, or incompatible dimension enters a study (§7A.3 closing requirement).

---

## 2. Canonical study-cell identity (brief §7A.1–7A.2)

`study/study_cell.py` registers **17 dimension classes**: 16 semantic identity dimensions plus one annotation-only engineering dimension. The hashed study-cell payload contains only the 16 semantic dimensions. Field-level mapping to **existing** identities:

| Axis contract | Fields → existing sources |
|---|---|
| `DataLineagePayload` | **Concrete field-level schema (V3 P0-6)** — strategy-only cells carry no context/formula identity merely because fields exist: `core_replay_id: str` · `v2_artifact_id: str` · `v2_manifest_hash: str` · `data_source_id` ("databento_nq_v1") · `dataset_schema_version` (2) · `bar_construction_policy_id` ("time_1m_v1") · `order_flow_depth_policy: Literal["mbp1_only_v1"]` (feature-lane pin; legacy replay provenance is separately scoped per CS §1.1) · **typed optionals, `None` for strategy-only cells:** `context_artifact_id: str \| None` · `context_manifest_hash: str \| None` · `context_formula_id: str \| None` (set to `ifvg_context_formula_v2` **only when a v3 context artifact is actually part of the computation** — never carried by default) · `feature_view_id: str \| None` · `label_view_id: str \| None` · `fold_set_id: str \| None` · `model_fit_id: str \| None` · `regime_fit_id: str \| None`. The former `artifact_pair_hash` survives as a derived convenience for v2+v3 studies only |
| `MarketUniverseIdentity` | `instrument_universe_id: Literal["nq_only_v1"]`, `contract_roll_policy_id` |
| `StrategyProfileIdentity` | `strategy_profile_id` (profile name — **baseline names gated by the fixed `PROFILE_CAPABILITY_REGISTRY`; generated `ifvg_search_profile_<hash16>` names gated by `GeneratedProfileCapability` instead**, Amendment P0-D), `strategy_version`, `profile_hash`, `section_config_hash` (the v2 record envelope), `entry_family`, `direction_policy` ("long_only_v1"; canonical shorts blocked until ratified), `session_policy`, `concurrency_policy` |
| `ObservationCohortIdentity` | `cohort_id` = a **`CohortSpec` hash** (the full public contract — `CONTRACTS_AND_SCHEMAS.md` §9.1: interpretation mode, date/warmup policy, row kind, session/direction/entry-family/HTF/parent/regime/evidence-quality filters, executed-vs-counterfactual scope, minimum coverage; `CONTEXT_COHORT_REGISTRY` keys map onto frozen `CohortSpec` instances), `observation_filters_hash`, `warmup_policy_id`, `date_policy_id` |
| `LabelPolicyIdentity` | `label_policy_id` (`IfvgContextLabelConfig.identity`), `label_family`, `label_derivation_id` (bar-evidence-hash-bearing) |
| `FeatureBundleIdentity` | `feature_bundle_key` (logical) + `resolved_feature_bundle_id` (the resolved-version hash, §7) + block-registry hash (new lane; tier lane keeps its own); `none_feature_bundle_v1` for strategy-only cells |
| `ModelProtocolIdentity` | `model_protocol_key` (`reference_prevalence_v1` \| `ifvg_context_logistic_l2_v1` \| `ifvg_context_catboost_binary_v1` \| `ifvg_context_gam_v1` planned), `resolved_model_protocol_id`, `calibration_policy_id` (`raw_probability_diagnostics_v1`), `resolved_regime_protocol_id \| None` |
| `DecisionPolicyIdentity` | `decision_policy_key` + `resolved_decision_policy_id` (`none_diagnostic_only_v1` baseline) |
| `ExecutionPolicyIdentity` | `execution_policy_id` ("confirmation_close_next1m_stop_first_v1"), `anchor_policy`, `resolver_policy` (envelope fields) |
| `CostPolicyIdentity` | `cost_policy_id` ("gross_zero_cost_v1" for the model lane; costed ids from the search-lane `CostPolicy`), `cost_points` |
| `RiskPolicyIdentity` / `PropContractIdentity` / `PayoutPolicyIdentity` / `PortfolioPolicyIdentity` | backend-owned ids (`CONTRACTS_AND_SCHEMAS.md` §4); `none_*` baselines (`none_v1`, `none_personal_account_v1`, `single_account_v1`) |
| `ValidationProtocolIdentity` | `fold_protocol_id: Literal["ifvg_context_walkforward_40_5_5_2_v1"]` (= the frozen `build_context_folds` protocol), `bootstrap_protocol_id: Literal["setup_and_day_block_bootstrap_10000_seed7_v1"]`, `fold_set_hash` |
| `StressScenarioIdentity` | `stress_scenario_id` ("none_historical_v1" baseline; else `STRESS_SCENARIOS_V1` keys) |
| `EngineeringProtocolAnnotation` | runtime/storage operation details; **not part of the semantic payload** — lives in `StudyCellAnnotation`, keyed by `cell_id`, never hashed (Amendment P0-B). The registry's `ENGINEERING_PROTOCOL` dimension type remains for classification, with `identity_effect="annotation_only"` now meaning "excluded from the semantic payload" |

```python
class StudyCellSemanticPayload(FrozenContract):     # Amendment P0-B: SEMANTIC dimensions only — 16 fields
    data_lineage: DataLineageIdentity
    market_universe: MarketUniverseIdentity
    strategy_profile: StrategyProfileIdentity
    observation_cohort: ObservationCohortIdentity
    label_policy: LabelPolicyIdentity
    feature_bundle: FeatureBundleIdentity
    model_protocol: ModelProtocolIdentity
    decision_policy: DecisionPolicyIdentity
    execution_policy: ExecutionPolicyIdentity
    cost_policy: CostPolicyIdentity
    risk_policy: RiskPolicyIdentity
    prop_contract: PropContractIdentity
    payout_policy: PayoutPolicyIdentity
    portfolio_policy: PortfolioPolicyIdentity
    validation_protocol: ValidationProtocolIdentity
    stress_scenario: StressScenarioIdentity
    # NOTE: EngineeringProtocolIdentity is REMOVED from the hashed payload.

    @model_validator(mode="after")
    def _all_dimensions_registered_and_usable(self): ...
        # resolve every axis value against the registry; assert usable; pairwise
        # incompatibility check; fail closed on unknown/blocked/incompatible.

class StudyCellIdentity(FrozenContract):            # envelope (P0-C)
    cell_id: str                                    # = canonical_contract_sha256(payload)
    payload: StudyCellSemanticPayload

class StudyCellAnnotation(FrozenContract):          # never hashed into cell_id
    cell_id: str
    engineering_protocol: EngineeringProtocolAnnotation   # runtime/storage operation details only
    display_metadata: Mapping[str, Any]

BASELINE_STUDY_CELL: StudyCellIdentity
    # §7A.2: one distinct baseline per dimension —
    # strategy ifvg_v2_doc_default_fresh_static_1r · cohort all-post-warmup-candidates ·
    # label candidate_static_r1_v2 · bundle IFVG_CORE_BASELINE_V1 · model reference_prevalence_v1 ·
    # execution confirmation-close/next-1m-stop-first · cost gross_zero_cost_v1 (model lane) /
    # verified NQ policy (search lane) · realization none_personal_account_v1

def study_cell_from_context_run(stored) -> StudyCellIdentity
    # read-only adapter: maps IfvgContextExperimentConfig/Result into a cell
    # (tier → frozen bundle id, §5.4) so legacy M0–M3 runs participate in comparisons
    # without re-execution.
```

**Semantic-vs-annotation classification rule (P0-B):** if an engineering/environment value can change numerical research output, it is **not** annotation-only — it moves into the owning semantic protocol (model package versions → `ModelProtocolIdentity`; resolver version → `ExecutionPolicyIdentity`/the core replay identity; simulation numeric policy → the owning `AccountSimulationPayload` or `PortfolioSimulationPayload`; bootstrap implementation/protocol → `ValidationProtocolIdentity`). Only runtime and storage operation details (workers, storage roots, page sizes, runtime estimates) remain annotations or execution-attempt evidence. Tests: changing workers/storage-root/page-size/runtime-estimate leaves `cell_id` unchanged; changing a model package/protocol version changes the correct semantic identity; every registry dimension marked `annotation_only` is absent from `StudyCellSemanticPayload`; the UI displays annotations without affecting comparison compatibility.

**Optional artifacts for strategy-only cells (Amendment P1-G):** `DataLineageIdentity`'s v3-context fields and the feature/label/model/regime axes support typed null identities — `none_context_artifact_v1`, `none_feature_bundle_v1`, `none_label_view_v1`, `none_model_protocol_v1`, `none_regime_model_v1` — so a valid FSM or prop study can consist of the core v2 replay + audit/chart companions + costed execution stream + prop simulation **without any v3 context, feature view, label derivation, fold set, model, or regime artifact**. The computation path decides what must be materialized: a strategy-only cell is never blocked by a missing v3 pair; a feature-only study still requires its exact v3/feature refs; a model study still requires labels/folds/model refs. Tests: strategy-only replay + prop sim with no v3/model artifacts passes; missing *required* derived evidence fails while irrelevant missing evidence does not.

**Generated-profile gating (Amendment P0-D):** `StrategyProfileIdentity.strategy_profile_id` is gated by the fixed `PROFILE_CAPABILITY_REGISTRY` **only when it is one of the five pre-existing named profiles (the baseline check)**. Generated canonical profiles (`ifvg_search_profile_<hash16>`) are gated by their own `GeneratedProfileCapability` contract (`CONTRACTS_AND_SCHEMAS.md` §1.4) — runnable baseline + registered/authorized values + intact invariants + valid section; they are never inserted into, and never fail merely for being absent from, the fixed registry.

A result is never described only as "Config A" when identities differ; every comparison view renders **Changed / Frozen / Derived / Unavailable** dimension sets computed against the selected reference (§7A.2, §7A.10 — the reference is switchable in the UI without changing any immutable study).

---

## 3. Comparison contract and computation paths (brief §7A.4–7A.5, §7A.15)

### 3.1 `study/comparison_contracts.py`

`ComparisonClass` carries exactly the brief's 20 values (`feature_only`, `cohort_descriptive`, `cohort_model`, `strategy_counterfactual`, `label_counterfactual`, `model_protocol`, `model_gate_execution`, `execution_policy`, `cost_policy`, `risk_policy`, `prop_realization`, `payout_policy`, `portfolio_policy`, `data_lineage`, `validation_protocol`, `stress_scenario`, `composite_preregistered`, `configuration_neighbor`, `baseline_to_child`, `child_to_child`).

```python
class ComparisonPayload(FrozenContract):       # V3 P0-1: the hashed payload — NO self-id field
    baseline_cell_id: str
    challenger_cell_ids: tuple[str, ...]
    changed_dimension_ids: tuple[str, ...]     # computed, then declared and frozen
    frozen_dimension_ids: tuple[str, ...]
    derived_dimension_ids: tuple[str, ...]
    comparison_class: ComparisonClass
    required_equalities: tuple[str, ...]
    permitted_differences: tuple[str, ...]
    computation_path: ComputationPath
    metric_registry: tuple[str, ...]           # DeltaOutputFamily ids (auto-selected, §4)
    uncertainty_protocol: Literal["setup_and_day_block_bootstrap_10000_seed7_v1",
                                  "day_block_bootstrap_10000_seed7_v1"]
    interpretation: Literal["descriptive", "modeled", "strategy_counterfactual"]   # §7A.14 output

class ComparisonEnvelope(FrozenContract):
    comparison_id: str                         # = canonical_contract_sha256(payload)
    payload: ComparisonPayload

class ComparisonCompatibility(FrozenContract):     # generalizes IfvgContextRunReconciliation
    per_dimension_match: ImmutableMap[str, bool]
    required_equalities_satisfied: bool
    registered_single_axis_delta_only: bool
    compatible_for_metric_delta: bool
    compatibility_status: Literal["compatible", "config_diff_only"]
    incompatibility_reasons: tuple[str, ...]
    differing_fields: tuple[str, ...]

class ComparisonResult(FrozenContract):
    comparison_id: str
    compatibility: ComparisonCompatibility
    delta_reports: ImmutableMap[str, ImmutableMap[str, Any]]  # family_id -> immutable payload
    config_diff: ImmutableMap[str, Any] | None                   # populated iff config_diff_only
    evidence_links: tuple[str, ...]            # run ids / artifact ids / manifest hashes
```

`build_comparison(...)` verifies the declared class matches the observed changed-dimension set (refuses mismatches), evaluates the required equalities, and on failure returns `config_diff_only` with a full field diff and **no** delta families beyond identity + compatibility (§7A.15 last rule; acceptance §7A.19.9).

### 3.2 `study/computation_path.py`

```python
class ComputationPath(FrozenContract):
    full_strategy_replay: bool
    feature_materialization: bool
    label_recomputation: bool
    model_refit: bool
    model_gated_sequential_replay: bool
    cost_recomputation: bool
    prop_resimulation: bool
    bootstrap_resimulation: bool
    reuse_trade_stream_hash: bool     # True when nothing above invalidates the stream

def derive_computation_path(changed: tuple[ExperimentDimensionSpec, ...]) -> ComputationPath
    # Pure OR-fold over each spec's requires_* flags, plus closure rules:
    #   full_strategy_replay        ⇒ cost + prop + bootstrap resimulation
    #   model_gated_sequential_replay ⇒ new trade stream ⇒ prop + bootstrap
    #   label change that alters trade lifetime/occupancy ⇒ full replay
    #   cost-only change with unchanged trade sequence ⇒ cost recomputation only
    # Deterministic, no I/O; unit-tested against the brief §7A.5 table row by row.
```

Registry seeding encodes §7A.5 exactly (feature blocks ⇒ materialize + refit; cohort-descriptive ⇒ nothing; cohort model ⇒ refit; label ⇒ label recompute; FSM/session/direction/entry-family ⇒ full replay; model algo/hyperparams ⇒ refit on identical rows/folds; execution-affecting threshold/abstention ⇒ frozen model + gated replay; lifetime-changing fill/exit/management ⇒ full replay; cost-only ⇒ cost recompute; risk ⇒ account resim; prop contract ⇒ prop resim; payout/withdrawal ⇒ payout resim; portfolio ⇒ portfolio resim; bootstrap/stress ⇒ sim rerun only; data/formula/bar policy ⇒ regenerate everything). The dashboard shows the computation path **before launch** because `ComparisonPayload.computation_path` is materialized at spec-build time.

**Model-gated path precondition (revision P0-11):** a `decision_policy` change routes through `DecisionPolicyEnvelope`, whose `RejectedCandidatePolicy` is a **strategy semantic** (a rejected candidate changes future candidate availability and slot occupancy). The gated-replay computation path is derivable, but it is executable only when the rejected-candidate policy carries owner ratification and the sequential golden tests exist — until then pipeline stage S11 stays `BLOCKED` with the reason text "model-gated execution is unavailable until RejectedCandidatePolicy is owner-ratified and sequential golden tests pass" (`CONTRACTS_AND_SCHEMAS.md` §9.2).

### 3.3 Required equalities per class (§7A.15)

`REQUIRED_EQUALITIES: MappingProxyType[ComparisonClass, tuple[str, ...]]`:

| Class | Required equalities (technical keys) |
|---|---|
| feature_only | artifact_pair_hash, strategy profile.*, cohort_id + observation_filters_hash, label_policy_id + label_derivation_id, fold_set_hash, model_protocol_id, cost_policy_id, **oos_row_ids** |
| cohort_descriptive | artifact_pair_hash, profile, label, data lineage (cohort may differ; result labeled descriptive) |
| cohort_model | as feature_only minus cohort; explicit no-counterfactual label |
| label_counterfactual | artifact_pair_hash, profile, cohort, fold_protocol_id, **candidate_ids** |
| model_protocol | rows/labels/folds/bundle/cost equal |
| model_gate_execution | frozen model hash equal; decision policy differs; requires a **new trade-stream hash** |
| execution_policy / cost_policy | cost class requires the identical trade-stream hash |
| risk_policy | trade_stream_hash, prop_contract_id, withdrawal_policy_id, replacement policy, clock policy, trade-path bundle, and path-capability report |
| prop_realization | trade_stream_hash, risk_policy_id (unless explicitly varied), withdrawal_policy_id, replacement policy, clock policy, and common trade-path bundle/capability report |
| payout_policy | trade_stream_hash, prop_contract_id, risk_policy_id — the varied dimension is the **withdrawal policy** (`WithdrawalPolicyPayload/Envelope`), which is separate from the firm contract |
| portfolio_policy | common correlated market-path id |
| strategy_counterfactual / configuration_neighbor / baseline_to_child / child_to_child | same source data, authorized dates, cost model, label/execution semantics unless explicitly changed, validation protocol (setup/trade identities may differ) |
| data_lineage / validation_protocol / stress_scenario | everything else equal |
| composite_preregistered | charter-frozen list |

---

## 4. Delta output registry (brief §7A.6–7A.8)

### 4.1 `study/delta_outputs.py`

`DeltaOutputFamily` carries the brief's **23** families verbatim: `identity_delta`, `compatibility_delta`, `population_delta`, `funnel_delta`, `sequence_delta`, `timing_delta`, `feature_coverage_delta`, `predictive_delta`, `calibration_delta`, `label_delta`, `economic_delta`, `execution_delta`, `cost_delta`, `risk_delta`, `tail_delta`, `stability_delta`, `concentration_delta`, `prop_delta`, `payout_delta`, `portfolio_delta`, `robustness_delta`, `evidence_quality_delta`, `engineering_delta`.

`DELTA_FAMILY_SELECTION: MappingProxyType[ComparisonClass, tuple[DeltaOutputFamily, ...]]` selects families **automatically** from the comparison class (acceptance §7A.19.8), e.g.:
- `feature_only` → identity, compatibility, feature_coverage, predictive, calibration, stability, concentration, evidence_quality
- `strategy_counterfactual` → identity, compatibility, population, funnel, sequence, timing, economic, tail, stability, concentration, evidence_quality
- `prop_realization` → identity, compatibility, prop, payout, tail, risk, robustness

### 4.2 Producer mapping (existing → reuse; NEW where noted)

| Family | Producer |
|---|---|
| identity / compatibility | `StudyCellIdentity` diff + `ComparisonCompatibility` (NEW) |
| population | NEW `study/population_delta.py` over the v2 `RecordTable` tables |
| funnel | NEW `study/funnel_delta.py` over `DayFunnelRecord.counters` |
| sequence | NEW: ordered trade/decision streams (entry-cursor ordering) → order diff + first divergence |
| timing | NEW: stage waits/durations from lifecycle event cursors + `bars_after_entry_to_resolution` |
| feature_coverage | existing coverage report + NEW per-block roll-up |
| predictive / calibration | existing `binary_prediction_report`; paired deltas via `paired_tier_delta_report` generalized to `paired_cell_delta_report` (same identical-OOS-ids gate) |
| label | existing label derivation frames; NEW agreement/flip table |
| economic | existing `compute_trade_stats` + `executed_trade_uncertainty_report` |
| execution / cost | `compute_trade_stats` cost blocks + search-lane `CostPolicy`; shortfall metrics deferred to the execution-model milestone |
| risk / tail / prop / payout / portfolio / robustness | `alpha_lab.propsim` + the search-lane prop extension (`CONTRACTS_AND_SCHEMAS.md` §4–5) |
| stability | NEW: per-fold metric series + chronological-thirds splits |
| concentration | NEW: session/day/setup concentration (Herfindahl, top-day share) |
| evidence_quality | existing quarantine table + reconciliation report + typed missing-reason counts |
| engineering | job runtime/artifact-size counters from run manifests |

### 4.3 `study/population_delta.py` — exact set relationships with an explicit match basis (§7A.6.E; revision P0-9)

**Verified constraint that reshapes this section:** every native record ID embeds the profile hash (`make_setup_id(profile_hash, htf_fvg_id, tap_cursor)`, SC `records.py:76`), so two different profiles **never share native IDs** even for the same market opportunity. A same-profile repeat test proves ID determinism — it proves nothing about cross-profile commonality. Every population delta therefore declares its **match basis**, and cross-profile matching runs on the profile-independent lineage layer (`CONTRACTS_AND_SCHEMAS.md` §4: `OpportunityLineage`, `NativeLineageMap`, `LineageMatchResult`, `PopulationDeltaMatchBasis`).

```python
class PopulationDeltaReport(FrozenContract):
    entity_kind: Literal["setup", "candidate", "decision", "trade"]
    match_basis: Literal["native_id_exact",                   # same-profile comparisons only
                         "profile_independent_lineage_exact", # cross-profile comparisons
                         "not_comparable"]                    # comparison disabled, reason given
    match_basis_reason: str | None
    common_keys: tuple[str, ...]          # native ids or lineage keys per the declared basis
    added_keys: tuple[str, ...]
    removed_keys: tuple[str, ...]
    jaccard: float | None
    first_divergence: FirstDivergence | None

class FirstDivergence(FrozenContract):
    trading_day: str
    entity_key: str
    side: Literal["added", "removed", "reordered"]
    baseline_cursor: str | None
    challenger_cursor: str | None
    divergence_reason: str | None
    # registered vocabulary: earlier_stale_parent_expiry | different_htf_winner |
    # different_parent_selection | different_fill_invalidation | different_session_gate |
    # slot_became_free | slot_remained_occupied | unattributed
```

Rules:
- Same-profile / same-config comparisons use `native_id_exact` (v2 primary keys).
- Cross-profile comparisons use `profile_independent_lineage_exact` **only** — lineage keys derive from source-stable evidence (FVG ids from bar identity + gap geometry; bar cursors), never from profile-hashed native IDs.
- **No nearest-time, row-order, fuzzy-geometry, or keep-last matching exists anywhere.** Where an exact profile-independent key cannot be constructed for a population type under the changed axes (e.g., the axis alters the FVG-detection threshold or the entry-trigger meaning), that population's comparison is **disabled** as `not_comparable` with a reason — commonality is never inferred.
- Divergence reasons are attributed only where a lifecycle terminal reason maps 1:1; otherwise `unattributed` — never guessed.
- **Lineage payloads and collisions (Amendment P1-E):** the exact per-entity lineage payloads (`SetupLineagePayload` → `CandidateLineagePayload` → `DecisionLineagePayload` → `TradeLineagePayload`) and the one-to-one uniqueness rule are specified in `CONTRACTS_AND_SCHEMAS.md` §4. When two native rows collide onto one lineage key: no dedupe, no keep-first/last, no fuzzy resolution — the entity kind is marked `not_comparable` with a persisted `LineageCollisionRecord`, and the `LineageUniquenessReport` accompanies every replay used in a cross-profile comparison.
- **Pre-ship requirements:** (a) the replay-determinism parity test (same config twice → Jaccard 1.0 on every entity kind, native basis); (b) the lineage-validity test (fvg-id stability across profiles sharing the timeframe set + `min_gap_ticks_capture`); (c) the lineage uniqueness/collision suite — all before any cross-profile delta ships.

### 4.4 `study/funnel_delta.py`

`FunnelDeltaReport` — per-counter `(baseline, challenger, delta)` totals + per-day breakdown + conditional stage-conversion deltas + terminal-reason deltas, built from the union of counter vocabularies (mismatches reported, never silently dropped).

---

## 5. Main effects and interactions (brief §7A.9) — `study/contrasts.py`

```python
class DeclaredContrastPayload(FrozenContract):     # V3 P0-1: hashed payload — no self-id
    charter_id: str                        # frozen search charter
    axis_dimension_ids: tuple[str, ...]    # 1 = main effect, 2 = two-way interaction
    conditioning: Mapping[str, str]        # immutable-mapping wrapper (deep-immutability rule, CS §0.3)
    cell_ids: tuple[str, ...]              # exact compatible child set — the denominator, recorded
    effect_kind: Literal["main_effect", "interaction", "conditional_effect",
                         "neighbor_effect", "firm_specific_effect", "regime_specific_effect"]

class DeclaredContrastEnvelope(FrozenContract):
    contrast_id: str
    payload: DeclaredContrastPayload

class ContrastResult(FrozenContract):
    contrast_id: str
    matched_pairs: tuple[tuple[str, str], ...]
    effect_estimate: ImmutableMap[str, Any]        # per-metric paired delta + bootstrap CI (seed-7 protocol)
    denominator: int
    causal_language_permitted: Literal[False] = False
```

Rules enforced in code: only declared contrasts (the `contrast_id` must exist in the frozen charter — post-hoc contrasts are refused); `UnbalancedDesignError` when the axis grid is not fully crossed over the conditioning slice and no explicit adjustment method is registered; raw child results always retained (contrasts reference cells, never replace them); observational wording only.

---

## 6. Feature-block contract (brief §7A.11) — `features/feature_blocks.py`

The stable registry definition and one resolved materialization are separate objects.

```python
class FeatureBlockStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_OWNER_DECISION = "blocked_owner_decision"
    EXPERIMENTAL = "experimental"
    SUPERSEDED = "superseded"

class AvailabilityStage(StrEnum):
    HTF_TAP = "htf_tap"
    PARENT_LOCK = "parent_lock"
    OPPOSING_CONFIRMATION = "opposing_confirmation"
    INVERSION = "inversion"
    ENTRY_DECISION = "entry_decision"

class FeatureBlockDefinition(FrozenContract):
    feature_block_key: str
    block_version: int
    human_name: str
    status: FeatureBlockStatus
    status_reason: str | None
    feature_family: str
    source_kind: Literal[
        "v2_candidate", "v3_context", "mbp1_parquet", "regime_artifact",
        "bars_1m", "planned_external",
    ]
    availability_stage: AvailabilityStage
    dependencies: tuple[str, ...]
    incompatible_blocks: tuple[str, ...]
    experimental_flags: tuple[str, ...]
    requires_strategy_replay: bool
    requires_feature_materialization: bool
    requires_model_retrain: bool
    can_affect_execution: bool
    expected_computation_path: str

class FeatureBlockResolutionPayload(FrozenContract):
    feature_block_key: str
    block_version: int
    formula_version: str
    source_artifact_refs: tuple[str, ...]
    source_schema_hash: str
    feature_schema_hash: str
    materializer_version: str
    feature_names: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    validity_fields: tuple[str, ...]
    missing_reason_fields: tuple[str, ...]
    source_timeframes: tuple[int, ...]
    source_interval_policy: str
    as_of_policy: str
    join_keys: tuple[str, ...]
    join_policy: Literal["one_to_one_required", "one_to_one_typed_null_on_missing"]
    direction_normalization: str
    session_normalization: str
    warmup_requirement: str
    coverage_requirements: ImmutableMap[str, float]
    mbp1_feature_windows: tuple[Mbp1FeatureWindowSpec, ...] = ()

class FeatureBlockResolutionEnvelope(FrozenContract):
    resolved_feature_block_id: str
    payload: FeatureBlockResolutionPayload

FEATURE_BLOCK_REGISTRY: MappingProxyType[str, FeatureBlockDefinition]
FEATURE_BLOCK_RESOLUTION_REGISTRY: MappingProxyType[str, FeatureBlockResolutionEnvelope]

def resolve_block_definition(feature_block_key: str) -> FeatureBlockDefinition: ...
def resolve_available_block(feature_block_key: str) -> FeatureBlockResolutionEnvelope: ...
```

A planned or blocked definition has no resolution envelope. Activation publishes a new definition version and a new resolved payload; study cells and bundles reference `resolved_feature_block_id`, not status-bearing registry metadata.

### 6.1 Registry seeding — exact partition of the existing tier ladder

The existing feature-name tuples in `context_feature_view.py` are the source of truth; blocks partition them **disjointly**:

| feature_block_key | status through release | feature names |
|---|---|---|
| `IFVG_CORE_BASELINE_V1` | available | `M0_FEATURES` minus `("entry_session", "in_engine_session", "in_doc_session")` |
| `IFVG_SESSION_CONTEXT_V1` | available | the three session features |
| `IFVG_STRUCTURE_CONTEXT_V1` | available | structure features over the primary TFs (60…3600 s) |
| `IFVG_STRUCTURE_CONTEXT_240_V1` | experimental | `ctx_structure_14400s_*` only |
| `IFVG_DISPLACEMENT_CONTEXT_V2` | available | displacement windows × primitives |
| `IFVG_LIQUIDITY_CONTEXT_V1` | available | `ctx_pool_*` + `ctx_sweep_*` |
| `IFVG_VOLATILITY_CONTEXT_V1` | planned | reserved `ctx_vol_*` (source: 1m bars; SC formula v3) |
| `IFVG_ORDER_FLOW_MBP1_V1` | planned through R5; available at R5B | §6.2 |
| `IFVG_REGIME_CONTEXT_V1` | planned | `ctx_regime_{resolved_regime_protocol_id_prefix}_{cluster_id \| dist_k \| prob_k \| entropy \| margin}` + validity/missing-reason fields (depends on an OOS-capable regime artifact) |
| `IFVG_KEY_LEVEL_CONTEXT_V1` | planned | reserved `ctx_keylevel_*` (PDH/PDL/session extremes distances) |
| `IFVG_EXECUTION_LIQUIDITY_V1` | planned | reserved `exliq_*` (shares the MBP-1 source artifact; **separate semantic identity** per §7A.6.L) |

A module-level assertion + unit test proves partition exactness: the union of the available blocks reproduces `TIER_FEATURE_REGISTRY[M3]` feature-for-feature — the frozen tiers are exactly recoverable.

### 6.2 MBP-1 source contract — `features/mbp1_source_contract.py` (brief §1.1A, §7A.6.L)

```python
class Mbp1SourceContract(FrozenContract):
    order_flow_depth_policy: Literal["mbp1_only_v1"] = "mbp1_only_v1"   # the ONLY representable value
    vendor: Literal["databento"] = "databento"
    schema: Literal["mbp-1"] = "mbp-1"
    instrument: str
    contract_roll_policy_id: str
    event_time_field: Literal["ts_event"] = "ts_event"
    receive_time_field: Literal["ts_recv"] = "ts_recv"
    sequence_field: Literal["sequence"] = "sequence"
    sequence_ordering_policy: Literal["ts_event_ts_recv_sequence_source_ordinal_v1"]
        # source_event_order_key = (ts_event, ts_recv, sequence, source_ordinal) — the COMPLETE
        # deterministic total order (revision P0-19). Event time + sequence alone is insufficient
        # (same-timestamp bursts and vendor sequence resets are real). source_ordinal = row ordinal
        # within the source partition, a deterministic final tie-break.
        # Verified caveat: raw mbp1.parquet retains ts_recv, but the current SC reader
        # intentionally discards it (databento_parquet.py:42); the PLANNED MBP-1 materializer
        # decodes ts_recv itself from the parquet — no SC change required.
    stage_cutoff_contract: Literal["stage_evidence_cutoff_v2"]
        # Amendment P0-G — the earlier "(stage_ts, +inf, +inf, +inf)" exclusive-upper-bound rule is
        # WITHDRAWN (it admitted same-timestamp events occurring after the stage decision).
        # Cutoffs are StageEvidenceCutoff objects (CONTRACTS_AND_SCHEMAS.md §10).
        # The comparator is selected by the registered Mbp1FeatureWindowSpec:
        # PRE_TRIGGER_EXCLUSIVE uses <; POST_TRIGGER_INCLUSIVE uses <= on the exact trigger key.
        #   TIMESTAMP_EXCLUSIVE     — include iff ts_event < stage_ts; ALL same-timestamp events
        #                             excluded or typed ambiguous;
        #   COMPLETED_BAR_BOUNDARY  — versioned bar-boundary rule tied to the exact completed bar and
        #                             feed-timestamp semantics; numerically-equal timestamps are not
        #                             assumed known before the bar-close decision.
        # When exact same-timestamp ordering is unavailable: preserve the row, emit typed missing
        # reason "same_timestamp_order_unavailable", never widen the window.
        # V3 P1-4: every MBP-1 feature/window additionally declares its WindowTriggerSemantics
        # (pre_trigger_exclusive | post_trigger_inclusive | completed_bar_as_of) — a FEATURE-DEFINITION
        # choice, not just an ordering rule: approach-imbalance windows exclude the trigger event;
        # an entry-state snapshot may deliberately include the event that made the stage observable
        # (comparator <= on the exact key); transition windows declare their interval bounds
        # ([prev, current) or (prev, current]) explicitly. See CONTRACTS_AND_SCHEMAS.md §10.
    feature_window_specs: tuple[Mbp1FeatureWindowSpec, ...]
        # Complete stage/window definitions; this tuple is included in FeatureBlockResolutionPayload.
    source_partitions: str                 # "data/databento/NQ/<date>/mbp1.parquet"
    coverage_policy: ImmutableMap[str, float]
    gap_semantics: str                     # "sequence_gap_marks_interval_invalid_v1"
    trades_derivation: Literal["trades_from_mbp1_d_p_17"] = "trades_from_mbp1_d_p_17"

MBP1_MISSING_REASONS = (
    "no_mbp1_partition", "sequence_gap", "coverage_below_threshold",
    "stage_outside_coverage", "instrument_roll_boundary",
    "same_timestamp_order_unavailable", "minimum_event_count_not_met",
)
```

Feature naming for the block separates two aggregate kinds:
- **Stage snapshots**: `ofl_snap_{stage}_{metric}`, stage ∈ {htf_tap, parent_lock, opposing, inversion, entry}, metric ∈ {spread_ticks, queue_imbalance, order_count_imbalance, microprice_offset_ticks, bid_sz, ask_sz, bid_ct, ask_ct}.
- **Stage-transition aggregates**: `ofl_win_{from}_{to}_{metric}`, metric ∈ {ofi_sum, aggressive_buy_frac, aggressive_sell_frac, depletion_events, replenishment_events, absorption_score, event_count, quote_intensity, trade_intensity}.

The initial R5B window registry is frozen as follows:

```text
ofl_snap_<stage>:
    from_stage = None
    to_stage = <stage>
    trigger_semantics = POST_TRIGGER_INCLUSIVE
    lower_bound = OPEN
    upper_bound = CLOSED
    minimum_event_count = 1

ofl_win_<from>_<to>:
    from_stage = <from>
    to_stage = <to>
    trigger_semantics = POST_TRIGGER_INCLUSIVE
    lower_bound = OPEN
    upper_bound = CLOSED
    minimum_event_count = 1
```

Thus a stage snapshot represents the top-of-book state when the stage became observable, and a transition aggregate represents `(from_stage, to_stage]`. If an exact triggering-event key is unavailable, timestamp-only evidence excludes all same-timestamp events; a snapshot that therefore cannot be established is retained as typed null with `same_timestamp_order_unavailable`. Changing these window definitions creates a new `FeatureBlockResolutionPayload` and resolved block ID.

Missing MBP-1 evidence preserves the row as typed null + registered reason; it never changes the cohort (acceptance §7A.19.14). **A registry-wide guard test asserts no block/bundle/feature/control identifier matches `mbp[\W_]?(10|\d{2,})`** — per the V3 P0-3 scoped statement: MBP-10 is unavailable to the new order-flow feature, model, UI, and live-streaming contracts (acceptance §7A.19.13); the single opaque replay-provenance literal (`legacy_verified_replay_source`, `CONTRACTS_AND_SCHEMAS.md` §1.1) is the only exemption and is unqueryable from this layer. Mandatory materializer tests (P0-19 + Amendment P0-G): multiple events sharing one `ts_event` split correctly under an exact source-order key; timestamp-only cutoffs exclude **every** same-timestamp event; a same-timestamp event occurring after the stage cannot change any feature; batch/repeat materialization windows are identical; a source scan proves no `+inf` cutoff construction exists. Delivery: the block's materializer stack ships in **Release R5B — Offline MBP-1 Feature Activation** (`PHASED_DELIVERY.md`); until R5B, the block stays `planned` and no baseline-vs-MBP-1 research claim is possible. The activated block is **research-only offline** — it cannot become a live model feature, execution gate, or Trade-Lab serving feature without a later Strategy-Core formula/parity contract and a separately approved sequential model-gated replay.

---

## 7. Feature-bundle contract (brief §7A.12) — `features/feature_bundles.py`

```python
class FeatureBundleDefinition(FrozenContract):
    feature_bundle_key: str                    # LOGICAL registry key (e.g. "B3_CORE_STRUCTURE_ORDER_FLOW")
    bundle_version: int; human_name: str
    base_bundle_key: str | None                # graph edge (composition by extension)
    included_block_keys: tuple[str, ...]       # logical keys; resolution binds concrete versions
    excluded_feature_ids: tuple[str, ...] = ()
    manual_include_ids: tuple[str, ...] = ()
    manual_exclude_ids: tuple[str, ...] = ()
    manual_override_registration_id: str | None   # registered identity for manual lists (§7B.3)
    required_profile_capabilities: tuple[str, ...]
    required_coverage_gates: Mapping[str, float]  # immutable-mapping wrapper (CS §0.3)

class FeatureBundleResolutionPayload(FrozenContract):   # hashed → resolved_feature_bundle_id
    feature_bundle_key: str; bundle_version: int
    resolved_block_ids: tuple[str, ...]        # the resolved_feature_block_id of each included block
    resolved_feature_names: tuple[str, ...]    # deterministic resolution order
    resolved_source_identities: tuple[str, ...]

class FeatureBundleResolutionEnvelope(FrozenContract):
    resolved_feature_bundle_id: str            # one resolved
    payload: FeatureBundleResolutionPayload    # version of one bundle; study cells reference THIS id

def resolve_bundle_definition(feature_bundle_key) -> FeatureBundleDefinition
    # walks the base chain (cycle-checked); refuses any included block whose status is not
    # AVAILABLE (or EXPERIMENTAL with an explicit flag) → BlockUnavailableError with status + reason.
```

Seeded bundles (a **graph**, not a cumulative enum ladder):

```
B0_CORE                                  = CORE + SESSION
B1_CORE_STRUCTURE                        = B0 + STRUCTURE
B2_CORE_ORDER_FLOW                       = B0 + ORDER_FLOW_MBP1        (unresolvable until the block activates)
B3_CORE_STRUCTURE_ORDER_FLOW             = B1 + ORDER_FLOW_MBP1        (planned)
B4_CORE_STRUCTURE_LIQUIDITY              = B1 + LIQUIDITY
B5_CORE_STRUCTURE_ORDER_FLOW_REGIME      = B3 + REGIME                 (planned)
B6_CORE_STRUCTURE_ORDER_FLOW_EXECUTION_LIQUIDITY = B3 + EXECUTION_LIQUIDITY (planned)

# Frozen tier-compat bundles (immutable; assert-equal to TIER_FEATURE_REGISTRY, order-exact):
TIER_M0_FROZEN_V1 · TIER_M1_FROZEN_V1 · TIER_M1_240_FROZEN_V1 · TIER_M2_FROZEN_V1 · TIER_M3_FROZEN_V1
```

**Migration/compat (M0–M3 untouched):** the tier lane (`ContextFeatureTier`, `TIER_FEATURE_REGISTRY`, `features_for_tier`, tier-bearing run identities) is not modified — existing run identities, view ids, and stored artifacts remain valid and reloadable. `bundle_for_tier(tier)` maps tiers to the frozen bundles for cross-lane comparison; an import-time/unit-test assertion guarantees order-exact equality. New studies reference the `feature_bundle_key` + its `resolved_feature_bundle_id`; new-lane run identities carry the resolved id instead of a tier. `features/bundle_feature_view.py::build_bundle_feature_view(pair, bundle)` reuses `build_candidate_feature_view(pair)` and projects `[identity columns, *resolved_feature_names]`; non-v2/v3 source kinds extend the frame by `join_keys` with typed-null missing handling.

**Planned-block activation creates a new versioned identity (revision P1-5; V3 key/resolved form).** Activation is *not* merely a status flip: when a planned block becomes real it gains its actual source schema hash, formula version, feature schema, materializer version, source identities, and coverage contract — so activation publishes a **new `FeatureBlockDefinition` version plus a `FeatureBlockResolutionEnvelope`; its `resolved_feature_block_id` is minted for the first time** (a later change = a new resolved id), changes the block-registry hash, and any bundle resolving the block gains a new `resolved_feature_bundle_id`. What stays fixed: the **logical `feature_block_key`** (e.g. `IFVG_ORDER_FLOW_MBP1_V1` — stable across versions) and the surrounding platform — no schema, contract, or UI *redesign* is required (§16.13); statuses and fail-closed behavior carry over. A test asserts the version bump, the resolved-id mint, and the registry-hash change on a simulated activation.

---

## 8. Session/direction interpretation selector (brief §7A.14; revision P0-10)

The selector is now backed by the public `CohortSpec` contract (`CONTRACTS_AND_SCHEMAS.md` §9.1): the UI's question maps 1:1 onto `CohortSpec.interpretation_mode`, and the mode maps onto a comparison class:

| Selector choice | `InterpretationMode` | Comparison class | Computation |
|---|---|---|---|
| Analysis slice only | `descriptive_slice` | `cohort_descriptive` | filter/report only; labeled descriptive |
| Train a specialized model | `specialized_model` | `cohort_model` | fold-local refit on the declared cohort |
| Build a new sequential strategy profile | `sequential_strategy_profile` | `strategy_counterfactual` | full sequential Strategy-Core replay via a **new child profile built from registered axis values** — never a filter |

Only the third creates a true strategy counterfactual; the `interpretation` field on `ComparisonPayload`, the cohort's own mode, and the dashboard labels enforce the distinction (acceptance §7A.19.3–4). Canonical shorts remain blocked until ratified; the schema supports them without redesign (`enable_shorts` value registered with `owner_ratification_status="pending"`).

---

## 9. Guardrails (brief §7A.18) — enforcement points

| Prohibited behavior | Enforced by |
|---|---|
| FSM + feature + label + model + prop-risk search in one uncontrolled grid | charter axes limited to `APPROVED_SEARCH_AXIS`; ML dimensions enter only via separately frozen comparison specs; `validate_charter` refuses mixed-kind axes |
| Post-hoc cohort slice as strategy counterfactual | `interpretation` labels + `cohort_*` classes never select execution/prop delta families |
| Candidate outcomes as executed-trade performance | existing `trade_stats` executions-only validation + `_reject_performance_payload` |
| Feature-only model metrics as proof of prop value | prop delta families selectable only for classes whose computation path includes prop resimulation |
| Threshold selection after viewing the same OOS curve | `ProhibitedSelectionError` (see `ML_REGIME_CONTRACT_PLAN.md` §6) |
| Silent inclusion of planned/blocked feature blocks | `resolve_bundle` refusal |
| Silent mixing of contract versions | `prop_contract_id` includes `contract_version`; equalities gate |
| Independent simulation of copied accounts | `run_portfolio_replay` has no per-account resampling path |
| Causal wording for descriptive marginal effects | `causal_language_permitted: Literal[False]` + UI wording rules |

---

## 10. Acceptance mapping (brief §7A.19 → designed artifact)

| # | Criterion | Designed enforcement |
|---|---|---|
| 1 | New block without baseline mutation | additive registry + untouched tier lane (partition test) |
| 2 | Feature-only preserves candidate IDs/folds | `REQUIRED_EQUALITIES[feature_only]` incl. oos_row_ids |
| 3 | Cohort studies claim no counterfactual | `interpretation` + class labeling |
| 4 | Session/direction strategy changes force new profiles | dimension `requires_full_strategy_replay` |
| 5 | Label changes create new identities + path metrics | `LabelPolicyIdentity` (config id + bar-evidence-hash derivation id) |
| 6 | Model gate forces gated replay | `DecisionPolicyPayload` replay-required flags (`ML_REGIME_CONTRACT_PLAN.md` §8) |
| 7 | Risk/prop reuse the trade-stream hash | `ComputationPath.reuse_trade_stream_hash` + equalities |
| 8 | Automatic metric-family selection | `DELTA_FAMILY_SELECTION` |
| 9 | Incompatible → config diff only | `ComparisonCompatibility.compatibility_status` |
| 10 | Exact added/removed/common sets | `population_delta.py` |
| 11 | Balanced factorial main/interaction effects | `contrasts.py` (declared-only, denominator recorded) |
| 12 | Dashboard exposes changed/frozen dimensions | `ComparisonPayload` fields consumed by the dimension-diff ribbon |
| 13 | MBP-1 has explicit planned/available states; no MBP-10/deeper-book feature, model, UI, or live-source capability (opaque legacy replay provenance is the sole exception) | `Mbp1SourceContract` Literal + regex guard test |
| 14 | Order-flow missing rows typed | block `join_policy` + `MBP1_MISSING_REASONS` |
| 15 | Every insight links to exact comparison + evidence | `ComparisonResult.evidence_links` + insight `EvidenceRef`s |
