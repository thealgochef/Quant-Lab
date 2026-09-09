# Contracts and Schemas — `ifvg_prop_robust_config_search_v1`

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 items 6, 7, 30)
**Status:** Plan-only. This file is the authoritative field-level contract specification for the search, replay, verification, prop-lifecycle, pipeline, and model-gating boundaries. The study/feature contracts in `DELTA_TAXONOMY.md` and the regime contracts in `ML_REGIME_CONTRACT_PLAN.md` are co-authoritative parts of the same self-contained package. No code exists yet.
**Revision status:** REVISED through the **V4 final contract-closure patch**. This patch closes resolved-identity schemas, verification authorization binding, trade-path/portfolio identity, MBP-1 window definitions, and regime provenance. See `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`.
**Conventions:** frozen pydantic v2 (`frozen=True, extra="forbid"`); enums are `StrEnum`; registries are `MappingProxyType`; hashing via the existing `canonical_contract_sha256`.

---

## 0. Identity conventions (applies to every contract below)

### 0.1 Payload/Envelope (non-self-referential)

Every ID-producing contract uses:

```python
class XPayload(FrozenContract):      # the semantic identity payload — the ONLY thing hashed
    ...
class XEnvelope(FrozenContract):
    x_id: str                        # = canonical_contract_sha256(payload)
    payload: XPayload
```

Pairs exist for every semantic identity, including **StudyCell, Comparison, DeclaredContrast, Cohort, RegimeProtocol, RegimeFit, FeatureBlockResolution, FeatureBundleResolution, DecisionPolicy, WalkForwardModelSchedule, AuthorizationRequirementSet, SearchCharter, VerificationRun, PropFirmContract, RiskPolicy, WithdrawalPolicy, AccountPolicySet, PortfolioPolicy, TradePathArtifact, TradePathBundle, AccountSimulation, PortfolioSimulation, PipelineSemanticSpec, ReplayInputBundle, and CoreStrategyReplay**. The converted feature/study definitions live in `DELTA_TAXONOMY.md`; the regime definitions live in `ML_REGIME_CONTRACT_PLAN.md`. Payloads **exclude**: their own derived ID; hashes/manifests of the artifact being created from that same payload; mutable display metadata; annotations; execution-attempt metadata; non-semantic timestamps; absolute local paths. Artifact content hashes are post-materialization envelope/manifest facts. A downstream semantic payload may pin an already-materialized upstream evidence artifact by its verified ID and manifest hash when exact provenance requires both. The **identity-projection audit test** enumerates every ID-producing contract and enforces all of this plus payload → ID → envelope → reload determinism.

### 0.2 Registry keys vs resolved identities (V3 P0-1 / P1-2)

Two distinct notions, never conflated:
- **Registry key** — a stable logical name: `IFVG_ORDER_FLOW_MBP1_V1`, `fixed_probability_threshold_v1`, `kmeans_v1`. Keys survive versions.
- **Resolved identity** — the hash of one concrete resolved payload (actual formulas, source artifacts, schemas, parameters, materializer versions).

Explicit paired fields wherever both exist: `feature_block_key` / `resolved_feature_block_id` · `feature_bundle_key` / `resolved_feature_bundle_id` · `decision_policy_key` / `resolved_decision_policy_id` · `algorithm_key` / `resolved_regime_protocol_id` · `axis_technical_key` / `RegisteredAxisValue.value_id`. Charters, cells, and simulations reference **resolved identities**; UIs may display keys.

### 0.3 Deep immutability (V3 P1-1)

`frozen=True` prevents field reassignment but does not deep-freeze nested mutables. `ImmutableMap[K, V]` in this plan means a read-only, canonically sorted mapping wrapper whose constructor defensively copies the input and whose serializer emits a tuple of sorted key/value records. Identity-bearing contracts therefore must use, in order of preference: (a) canonical tuple-based records instead of dicts where practical; (b) an immutable-mapping wrapper type with a registered canonical serializer for genuinely map-shaped fields (`parameters`, `conditioning`, coverage maps); (c) where a plain `Mapping` remains, a defensive deep-copy at construction plus identity revalidation immediately before every save/use. **Mutation-adversarial tests** are mandatory: mutate every reachable nested structure after identity calculation and prove the persisted identity/payload is unaffected (`TEST_MATRIX.md` §3.9).

Package placement: `ifvg/search/`, `ifvg/study/`, `ifvg/features/`, `ifvg/ml/`, `propsim/`. Reserved decisions D-039…D-045 stand (D-039 also covers the replay-input bundle; D-042 the envelope/key conventions + generated-profile capability).

---

## 1. Replay identity stack

### 1.1 `ReplayInputBundle` — content-addressing the complete replay input

Reuses the verified native primitives (`data_access.hash_allowlisted_source_files`, day-artifact stamps/manifests, `manifest.read_repository_state`/`source_tree_hash`) — no duplicate hashing.

```python
class ReplaySourcePartitionRef(FrozenContract):
    source_partition_id: str           # V3 P0-4: stable physical/logical key — one trading day is
                                       # assembled from MULTIPLE UTC/source partitions, so the fields
                                       # below alone cannot distinguish two partitions of one day
    source_partition_utc_date: str     # the UTC date of the physical partition file
    source_manifest_id: str | None     # when the partition is manifest-backed
    relative_logical_partition_key: str  # e.g. "prev_utc_date/mbp1" vs "day_utc_date/mbp1"
    trading_day: str
    instrument: str
    contract_symbol: str | None
    source_kind: Literal["mbp1", "trades", "legacy_verified_replay_source"]   # V3 P0-3, policy below
    source_schema_id: str
    content_sha256: str
    byte_size: int
    row_count: int | None

class ReplayDayArtifactRef(FrozenContract):
    trading_day: str
    artifact_kind: Literal["bars", "levels", "calendar", "other_required_input"]
    artifact_id: str
    manifest_payload_sha256: str
    content_sha256: str | None

class ReplayAccessAuthorizationRef(FrozenContract):    # V3 P0-5: DETERMINISTIC PREFLIGHT — may enter identity
    access_policy_id: str
    authorized_date_set_id: str
    expected_source_inventory_hash: str    # authorized dates + expected source hashes + policy, pre-run

class ReplayInputBundlePayload(FrozenContract):        # = ReplayInputContentIdentity
    authorized_date_set_id: str
    ordered_source_partitions: tuple[ReplaySourcePartitionRef, ...]
        # canonical order: (trading_day, source_partition_utc_date, relative_logical_partition_key,
        #                   source_kind, instrument) — includes the V3 P0-4 partition keys
    ordered_day_artifacts: tuple[ReplayDayArtifactRef, ...]   # (trading_day, artifact_kind)
    source_contract_id: str
    source_schema_era_id: str
    access_authorization: ReplayAccessAuthorizationRef

class ReplayInputBundleEnvelope(FrozenContract):
    replay_input_bundle_id: str
    payload: ReplayInputBundlePayload

class ReplayExecutionAccessAudit(FrozenContract):      # V3 P0-5: RUNTIME evidence — NEVER in replay identity
    core_replay_id: str
    execution_attempt_ref: str
    event_chain_sha256: str            # actual path constructions/opens/reads/ordering for ONE attempt
    counters: Mapping[str, int]        # protected/sealed counters (must be zero)
```

Reorder-only inputs are normalized to canonical order; unnormalizable orderings are refused. Absolute paths never appear; identity is unchanged when files move with identical verified content.

**`source_kind` policy (V3 P0-3 — Option B, legacy-scoped):** the development window's early source eras genuinely decode from `mbp10.parquet` (era-boundary behavior verified in the Strategy-Core reader), so historical replay provenance must be able to reference those partitions truthfully. `legacy_verified_replay_source` is that reference — **opaque provenance only**. It cannot be queried by the new order-flow feature layer, bundle materialization, the dashboard, live-source contracts, or model features (guard tests enforce each). The truthful boundary statement, used consistently across the package: **MBP-10 is unavailable to the new order-flow feature, model, UI, and live-streaming contracts; legacy replay provenance may reference historical source eras without exposing deeper-book features.** The MBP-10 identifier-guard regex still applies to every feature/bundle/control/registry namespace — only this one provenance literal is exempt.

### 1.2 `CoreStrategyReplayIdentity`

```python
class CoreStrategyReplayPayload(FrozenContract):
    replay_input_bundle_id: str                    # §1.1 — the exact bytes consumed
    quant_lab_replay_source_identity: str          # source_tree_hash + commit/dirty evidence scoped to the
                                                   # QL code controlling profile resolution, capture driver,
                                                   # dataset chain, replay adapter, table partitioning,
                                                   # seed chaining, resolver invocation
    strategy_core_commit: str
    strategy_core_source_identity: str
    resolved_section_config_hash: str              # ifvg_profile_hash(section-with-canonical-name)
    canonical_profile_id: str                      # §1.3
    warmup_seed_identity: str                      # profile-bound: seed-snapshot id | warmup-policy id
    resolver_policy: Literal["next_1m_bar_stop_first_v1"]
    anchor_policy: str
    capture_schema_version: int
    record_schema_version: int

class CoreStrategyReplayIdentity(FrozenContract):
    core_replay_id: str
    payload: CoreStrategyReplayPayload
```

**Identity rule (test-enforced).** Changes with: source partition bytes; cached bar/level artifact content; source schema/era; QL replay/capture code; SC replay/reducer code; section semantics; seed; resolver; date set. Unchanged for: parent-study membership; display names; cost/risk/prop/payout policies; audit/chart schema; worker count; memory; retry metadata; **runtime access audits** (P0-5).

### 1.3 Canonical semantic profile naming

`name_free_section_hash = sha256(section.model_dump minus "profile_name")`; `canonical_profile_id = f"ifvg_search_profile_{name_free_section_hash[:16]}"`; the section is constructed with that name and hashed by the existing `ifvg_profile_hash`. Verified: `profile_name` participates in Strategy-Core record IDs via the profile hash (SC `section.py:97/324–331`, `records.py:76–85`) — canonical naming makes that participation study-independent. Baselines keep their registered names; display labels live only in the mutable catalog.

### 1.4 `GeneratedProfileCapability`

```python
class ResolvedSearchProfileRef(FrozenContract):
    canonical_profile_id: str
    baseline_profile_id: str               # must be RUNNABLE in the fixed PROFILE_CAPABILITY_REGISTRY
    resolved_section_config_hash: str
    axis_value_ids: Mapping[str, str]      # immutable-mapping wrapper (§0.3)
    owner_authorization_id: str
    profile_capability_id: str

class GeneratedProfileCapability(FrozenContract):
    capability_id: str
    status: Literal["generated_runnable", "blocked_owner_decision", "blocked_invalid_section",
                    "blocked_invariant_failure", "blocked_base_profile"]
    baseline_capability_ref: str
    registry_hash: str
    authorization_ref: str
    validation_report_ref: str | None
    reason: str | None
```

Eligibility: runnable baseline + registered values + values owner-authorized for the run scope + intact locked invariants + valid resolved section + deterministic canonical name/hash. Generated children are never inserted into the fixed registry and never fail for being absent from it; post-replay invariant/neutrality failures block publication; no legacy fixed-tier launcher access.

### 1.5 Membership, companions, costed evaluation (complete schemas — V3 P0-2 merge)

```python
class SearchChildMembership(FrozenContract):
    parent_search_id: str
    child_ordinal: int
    axis_value_ids: Mapping[str, str]          # RegisteredAxisValue.value_id per axis key
    core_replay_id: str
    comparison_role: Literal["baseline", "challenger", "neighbor"]
    # display names live in the mutable catalog, never here

class CoreReplayArtifactReference(FrozenContract):
    core_replay_id: str
    v2_dataset_artifact_id: str
    manifest_payload_sha256: str
    gross_trade_stream_hash: str               # table_content_hash over EXECUTED_TRADE (gross; SC is frictionless)

class FsmAuditArtifactIdentity(FrozenContract):
    core_replay_id: str
    audit_schema_version: int
    audit_contract_fingerprint: str
    neutrality_mechanism_id: str               # §3.3

class ReplayChartArtifactIdentity(FrozenContract):
    core_replay_id: str
    replay_chart_schema_version: int
    range_policy_id: str
    stage_gating_policy_id: str

class CostedEvaluationIdentity(FrozenContract):
    core_replay_id: str
    cost_policy_sha256: str                    # → net metrics, PlannedVsRealizedEdge, costed trade records
```

Derived-layer identities (`FeatureViewIdentity` = the existing content-hashed view id; `LabelViewIdentity` = the existing label-derivation id incl. bar-evidence hash; `FoldSetIdentity` = the fold-set hash; `ModelFitIdentity`/`PredictionSetIdentity` = the resolved-protocol + run-id pattern; `RegimeFitEnvelope` = `ML_REGIME_CONTRACT_PLAN.md` §3) all follow §0.

---

## 2. Search-axis registry (complete schema — V3 P0-2 merge)

```python
class AxisClassification(StrEnum):
    LOCKED_INVARIANT = "locked_correctness_invariant"
    THESIS_DEFINING = "strategy_thesis_defining"
    APPROVED_SEARCH_AXIS = "approved_search_axis"
    RISK_POLICY_AXIS = "risk_policy_axis"
    MEASUREMENT_ONLY = "measurement_only"
    BLOCKED = "blocked"
    EXPERIMENTAL = "experimental"

class DimensionValueSpec(FrozenContract):
    value_schema_id: str
    json_schema_hash: str

class RegisteredAxisValue(FrozenContract):
    value_id: str                          # stable, e.g. "parent_retest_timeout.480" / ".none"
    axis_technical_key: str
    payload: Any                           # typed per DimensionValueSpec: int | None | tuple[str, ...] | record
    human_label: str
    capability_status: Literal["available", "blocked_reducer_hardcoded", "blocked_inert_field",
                               "blocked_legacy_field", "blocked_pending_owner_policy_review", "planned"]
    owner_ratification_status: Literal["ratified", "pending", "not_required"]   # VALUE-specific
    ratification_evidence_ref: str | None
    dependencies: tuple[str, ...]
    incompatibilities: tuple[str, ...]
    expected_replay_effect: str

class CompositeAxisValue(FrozenContract):  # dependent field groups / composite named policies
    value_id: str
    human_label: str
    member_values: Mapping[str, Any]       # e.g. {"entry_near_parent": True, "entry_parent_distance_ticks_max": 40}
    # + the same status/ratification/dependency fields as RegisteredAxisValue

class SearchAxisSpec(FrozenContract):
    technical_key: str                     # exact IfvgSmcSection field (or "risk_policy.*")
    human_label: str
    description: str
    classification: AxisClassification
    value_spec: DimensionValueSpec
    registered_values: tuple[str, ...]     # the ONLY selectable value_ids
    baseline_value_id: str
    requires_full_sequential_replay: bool
    changes_capture_artifacts: bool        # True only for TF-set/tick/scheme axes
    dependencies: tuple[str, ...]
    expected_artifact_effect: str

SEARCH_AXIS_REGISTRY_V1 / AXIS_VALUE_REGISTRY_V1: MappingProxyType
def registry_sha256() -> str
def assert_axes_authorized(axis_value_ids, registry=...) -> None   # fail-closed, value-level
```

Classification results (verified against SC `section.py`): searchable-today axes = the timeout/expiry clocks (`parent_retest_timeout_1m_bars` [baseline value `.none` — unbounded, the accepted doc-default], `opposing_timeout_1m_bars`, `inversion_timeout_1m_bars`, `post_inversion_expiry_1m_bars_max`, `parent_reaction_window_parent_bars`), distances (`parent_htf_distance_ticks_max`, `opposing_parent_distance_ticks_max`, the `entry_near_parent` composite), retention (`htf_registry_max_age_days`, `ltf_registry_max_live`, `htf_selection_max_per_timeframe`), capture geometry (`min_gap_ticks_capture`, `swing_strength_bars`, `swing_pool_max`), trade geometry (`sl_buffer_ticks`, `tp_r_multiple`), session policy (`enabled_entry_sessions`, `outside_session_policy`), `max_executed_trades_per_day`, `enable_shorts` (pending ratification), TF sets (`htf_timeframes`, `parent_timeframes` — expensive, `changes_capture_artifacts=True`). Blocked: inert traps (`break_even_enabled`, `legacy_candidate_row_limit`, `parent_reaction_window_1m_bars_max`), reducer-hardcoded axes (fill model, resolver branching, stop anchor, non-1m gap timeframes, `ifvg_retest` execution, tap-conflict, ranking, intraday flatten), and **`parent_full_fill_invalidation`/`parent_structural_invalidation` = `blocked_pending_owner_policy_review`**. Locked invariants: `runnable`/`execution_enabled`/`non_runnable_reason`/`qualification_mode`, the causality triple, `anchor_policy`, strict body-close inversion, one-active-setup/one-active-trade, no pyramiding, entry-at-confirmation-close, exact PIT evidence. Thesis-locked in v1: `session_scheme`, `doc_sessions`, `entry_families`, `entry_family`, `label_family`, `enable_longs`. The UI never exposes raw `section_overrides`.

---

## 3. Charter, authorization, child replay, and gates

### 3.1 Complete search-charter schemas

```python
class SearchMode(StrEnum):
    SINGLE_CONFIGURATION = "single_configuration"
    FSM_CONFIG_SEARCH = "fsm_config_search"
    PROP_BENCHMARK = "prop_benchmark"
    UNIVERSAL_PROP_SEARCH = "universal_prop_search"

class ResolvedPropGateThresholds(FrozenContract):
    minimum_first_payout_probability_60d: float
    maximum_breach_probability_90d: float
    minimum_expected_net_payout_90d: float | None
    minimum_p10_net_payout_90d: float | None
    maximum_p90_payout_drought_days: int | None
    minimum_three_payout_probability: float | None

class ResolvedRobustnessGateThresholds(FrozenContract):
    maximum_neighbor_expectancy_degradation_r: float | None
    minimum_plateau_width: int | None
    maximum_worst_firm_breach_probability_90d: float | None
    minimum_time_block_sign_consistency: float | None
    minimum_outer_fold_recurrence: float | None  # schema-reserved; null in the exploratory lane

class ObjectivePolicy(FrozenContract):
    feasibility_gates: ResolvedStrategyGateThresholds
    prop_feasibility_gates: ResolvedPropGateThresholds
    robustness_gates: ResolvedRobustnessGateThresholds
    pareto_objectives: tuple[str, ...]
    lexicographic_tie_breaks: tuple[str, ...]
    # No weight fields exist; a hidden weighted score is structurally impossible.

class DatePolicy(FrozenContract):
    replay_dates: tuple[str, ...]
    warmup_dates: tuple[str, ...]
    development_cutoff_utc: Literal["2026-06-10T21:00:00Z"] = "2026-06-10T21:00:00Z"
    access_policy_id: Literal[
        "development_explicit_dates_before_path_v2",
        "verification_fixed_allowlist_max5_v1",
    ]

class SimulationProtocol(FrozenContract):
    modes: tuple[Literal[
        "historical_closed_trade",
        "historical_1m_scenario",
        "historical_ordered_event_replay",
        "day_block_bootstrap",
        "stress",
    ], ...]
    bootstrap_n_paths: int = 10_000
    bootstrap_seed: int = 42
    block_policy: Literal["single_day", "contiguous_5day_block"] = "single_day"
    stress_scenario_ids: tuple[str, ...]
    trade_path_capability_policy_id: str
    clock_policy_id: str

class CostPolicy(FrozenContract):
    cost_points_round_turn: float = 0.514
    dollars_per_point: float = 20.0
    tick_size: float = 0.25

class SearchCharterPayload(FrozenContract):
    contract_name: Literal["ifvg_prop_robust_config_search_v1"]
    schema_version: int = 1
    search_mode: SearchMode
    baseline_profile_name: str
    baseline_section_config_hash: str
    axes: ImmutableMap[str, tuple[str, ...]]
    locked_invariants_registry_sha256: str
    measured_only_fields: tuple[str, ...]
    blocked_capabilities: tuple[str, ...]
    authorized_firm_contract_ids: tuple[str, ...]
    authorized_risk_policy_ids: tuple[str, ...]
    authorized_withdrawal_policy_ids: tuple[str, ...]
    objective_policy: ObjectivePolicy
    date_policy: DatePolicy
    simulation_protocol: SimulationProtocol
    max_child_count: int
    search_algorithm: Literal["deterministic_exhaustive_v1"]
    seed: int
    cost_policy: CostPolicy
    strategy_core_commit: str
    quant_lab_commit: str
    source_artifact_ids: tuple[str, ...]
    owner_authorization: OwnerAuthorizationBundle | SyntheticAuthorizationMarker

class SearchCharterEnvelope(FrozenContract):
    search_id: str
    payload: SearchCharterPayload
```

Freeze means immutable save. Any semantic change creates a new `search_id`; presentation changes remain catalog-only. `validate_charter` enforces registered axes and values, runnable baseline or generated-profile capability, child-count ceiling, contract status, date policy, and computation-path-scoped authorization.

### 3.2 Computation-path-scoped owner authorization

```python
class OwnerDecisionEvidenceRef(FrozenContract):
    decision_id: str
    decision_artifact_id: str
    content_hash: str
    author: str
    approved_at: str
    effective_from: str
    reviewed_evidence_refs: tuple[str, ...]

class AuthorizationRequirement(FrozenContract):
    decision_key: str
    reason: str
    required_for_stage_ids: tuple[str, ...]
    required_for_dimension_ids: tuple[str, ...]
    required_for_scope: tuple[str, ...]

class AuthorizationRequirementSetPayload(FrozenContract):
    requirements: tuple[AuthorizationRequirement, ...]

class AuthorizationRequirementSetEnvelope(FrozenContract):
    requirement_set_id: str
    payload: AuthorizationRequirementSetPayload

class OwnerAuthorizationBundle(FrozenContract):
    requirement_set_id: str
    decision_refs: ImmutableMap[str, OwnerDecisionEvidenceRef]

class SyntheticAuthorizationMarker(FrozenContract):
    kind: Literal["synthetic_test_authorization_v1"]

class VerificationAuthorizationRef(FrozenContract):
    verification_policy_id: str
    approved_allowlist_hash: str
    coverage_matrix_artifact_id: str
    seed_snapshot_id: str
    approved_by: str
    approved_at: str
    content_hash: str


def derive_authorization_requirements(
    run_scope,
    study_dimensions,
    computation_path,
    enabled_pipeline_stages,
    selected_firms,
) -> AuthorizationRequirementSetEnvelope: ...
```

A run fails only on decisions required by its actual computation path. Strategy-only studies do not require prop or regime decisions; prop studies add firm/fidelity/risk/withdrawal/clock decisions; regime studies add grain/count/stability decisions; model-gated replay adds `RejectedCandidatePolicy`. The real verification slice requires `VerificationAuthorizationRef`; the synthetic marker is confined to fully synthetic fixtures.

### 3.3 Child replay worker and audit neutrality

Worker flow: resolve profile → construct the validated section → authorize inputs → run the sequential v2 capture/evaluation → build requested companions → assert zero forbidden access → publish atomically → reload and verify, keyed by `core_replay_id`.

```python
class ChildAuditNeutralityReport(FrozenContract):
    core_replay_id: str
    mechanism: Literal["dual_drive_ab_v1", "single_drive_side_channel_v1"]
    audit_disabled_core_table_hashes: ImmutableMap[str, str] | None
    audit_enabled_core_table_hashes: ImmutableMap[str, str] | None
    tables_equal: bool | None
    mechanism_evidence_refs: tuple[str, ...]
    core_trace_content_hash: str
    audit_stamp_referential_integrity: bool
    passed: bool
```

The accepted doc-default parity gate remains unchanged. Any failed child-neutrality report blocks the audit artifact and child publication.

### 3.4 Strategy, prop, and robustness gate contracts

```python
class ResolvedStrategyGateThresholds(FrozenContract):
    min_executed_trades: int = 30
    min_independent_days: int = 20
    min_net_expectancy_r: float = 0.0
    min_profit_factor: float = 1.1
    max_drawdown_r: float = 15.0
    max_time_under_water_days: int = 45
    min_session_stability_score: float
    min_time_block_sign_consistency: float = 0.6
    max_top_day_pnl_share: float = 0.40
    max_top_setup_pnl_share: float = 0.25
    require_bootstrap_ci_excludes_zero: bool = False

class PlannedVsRealizedEdge(FrozenContract):
    planned_rrr: float
    realized_avg_winner_r: float
    realized_avg_loser_r: float
    realized_payoff_ratio: float
    gross_expectancy_r: float
    cost_r: float
    net_expectancy_r: float
    slippage_commission_drag_r: float
```

All gate values are proposals until owner-ratified. Research gates never run against the five-day verification fixture; that fixture uses `verification_control_flow_gates_v1` only.

## 4. Profile-independent opportunity lineage

```python
class SetupLineagePayload(FrozenContract):
    direction: str
    htf_timeframe_seconds: int
    htf_fvg_id: str                    # bar-identity + gap-geometry derived; profile-independent
    activation_cursor: str

class CandidateLineagePayload(FrozenContract):
    setup_lineage_id: str
    entry_family: str
    entry_trigger_cursor: str
    entry_fvg_id: str | None
    decision_as_of_cursor: str

class DecisionLineagePayload(FrozenContract):
    candidate_lineage_id: str
    decision_kind: str
    decision_cursor: str

class TradeLineagePayload(FrozenContract):
    decision_lineage_id: str
    entry_cursor: str
    entry_policy_id: str

class LineageCollisionRecord(FrozenContract):
    core_replay_id: str; entity_kind: str
    lineage_key: str; native_ids: tuple[str, ...]

class LineageUniquenessReport(FrozenContract):
    core_replay_id: str
    per_entity_kind: Mapping[str, Literal["one_to_one", "collisions_present"]]
    collisions: tuple[LineageCollisionRecord, ...]
```

Rules: payloads are profile-independent, PIT, source-derived, unique within a replay; native→lineage must be one-to-one; collisions ⇒ `not_comparable` + persisted records (never dedupe/keep-first/fuzzy). `match_basis` ∈ {`native_id_exact`, `profile_independent_lineage_exact`, `unmatched`, `not_comparable`} on every population delta. Pre-ship: native-determinism, lineage-validity, and collision suites.

---

## 5. Prop-firm lifecycle and simulation contracts

### 5.1 Trade-path evidence, bundles, and capabilities

```python
class TradePathFidelity(StrEnum):
    CLOSED_TRADE_ONLY = "closed_trade_only"
    OHLC_1M_UNORDERED = "ohlc_1m_unordered"
    ASSUMED_1M_INTRABAR_PATH = "assumed_1m_intrabar_path"
    ORDERED_MBP1_EVENT_PATH = "ordered_mbp1_event_path"
    ORDERED_FILL_EVENT_PATH = "ordered_fill_event_path"

class PathCapability(StrEnum):
    CLOSED_TRADE_RESULT = "closed_trade_result"
    REALIZED_PNL_CHRONOLOGY = "realized_pnl_chronology"
    UNREALIZED_MARK_TO_MARKET = "unrealized_mark_to_market"
    MARKET_PRICE_CHRONOLOGY = "market_price_chronology"
    INTRABAR_EXTREMA_ORDER = "intrabar_extrema_order"
    ORDERED_FILLS = "ordered_fills"

class OhlcBarPathObservation(FrozenContract):
    open_ts_utc: str
    close_ts_utc: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    observed_intrabar_order: Literal["unknown"] = "unknown"

class TradePathEvent(FrozenContract):
    event_id: str
    event_ts_utc: str
    event_ordinal: int
    source_order_key: tuple[str, str, int, int] | None
    kind: Literal["bar_extreme", "bar_close", "quote", "trade", "fill", "entry", "exit"]
    price: float
    quantity: float | None
    generating_policy_id: str | None

class TradePathArtifactPayload(FrozenContract):
    core_replay_id: str
    trade_id: str
    fidelity: TradePathFidelity
    capabilities: tuple[PathCapability, ...]
    observations: tuple[OhlcBarPathObservation, ...] | None
    ordered_events: tuple[TradePathEvent, ...] | None
    intrabar_scenario_policy_id: str | None
    source_identities: tuple[str, ...]

class TradePathArtifactEnvelope(FrozenContract):
    trade_path_artifact_id: str
    payload: TradePathArtifactPayload
    manifest_payload_sha256: str

class TradePathBundlePayload(FrozenContract):
    core_replay_id: str
    gross_trade_stream_hash: str
    ordered_trade_ids: tuple[str, ...]
    ordered_trade_path_artifact_ids: tuple[str, ...]
    bundle_capabilities: tuple[PathCapability, ...]  # capabilities common to every included trade path
    per_trade_capability_report_ids: tuple[str, ...]
    coverage_policy_id: str

class TradePathBundleEnvelope(FrozenContract):
    trade_path_bundle_id: str
    payload: TradePathBundlePayload
    manifest_payload_sha256: str

class PropRulePathRequirement(FrozenContract):
    rule_id: str
    required_path_capabilities: tuple[PathCapability, ...]
    accepted_fidelity_classes: tuple[TradePathFidelity, ...]
    scenario_use_permitted: bool

class PathCapabilityReport(FrozenContract):
    trade_path_bundle_id: str
    available_capabilities: tuple[PathCapability, ...]
    per_rule: ImmutableMap[str, Literal["supported", "scenario_only", "unsupported"]]
    failure_reasons: ImmutableMap[str, tuple[str, ...]]
```

Fidelity classes are not treated as one total ordering. A rule is supported only when its explicit capability set is present and its fidelity class is accepted. Assumed 1-minute intrabar paths remain scenario evidence. Chronology-sensitive historical claims require actual ordered market/fill evidence as specified by that rule.

### 5.2 Firm rules, calendar, withdrawal, and fees

```python
class DayCountBasis(StrEnum):
    TRADING_DAY = "trading_day"
    WINNING_DAY = "winning_day"
    BUSINESS_DAY = "business_day"
    CALENDAR_DAY = "calendar_day"
    CALENDAR_MONTH = "calendar_month"
    FIRM_DEFINED_PAYOUT_PERIOD = "firm_defined_payout_period"

class DurationRule(FrozenContract):
    count: int
    basis: DayCountBasis
    firm_calendar_policy_id: str | None

class FirmCalendarPolicy(FrozenContract):
    policy_id: str
    trading_day_boundary: Literal["18:00_ET_roll"]
    business_day_definition: str
    holiday_handling: str

class SimulatedClockPolicy(FrozenContract):
    policy_id: str
    advancement_rules: ImmutableMap[str, str]

class PhaseRules(FrozenContract):
    starting_balance: float
    profit_target: float | None
    trail_amount: float
    trail_style: Literal["eod_floor_realtime_breach", "intraday_peak_trail", "static_floor"]
    trail_locks_at_start: bool
    dll_amount: float | None
    dll_hard: bool
    consistency_pct: float | None
    min_days: DurationRule | None
    max_eval_days: DurationRule | None
    unrealized_equity_counts_for_breach: bool
    breach_observation_policy: Literal["continuous_ordered_path", "event_updates_only", "eod_only"]
    max_contracts: int
    micro_scaling_table: tuple[tuple[float, int], ...] = ()

class PayoutPolicy(FrozenContract):
    waiting_period: DurationRule
    min_between_payouts: DurationRule
    min_winning_days: DurationRule | None
    winning_day_min_pnl: float | None
    payout_cap_per_period: float | None
    split_pct_trader: float
    min_payout: float | None
    max_payout: float | None
    withdrawal_reduces_threshold: bool
    post_payout_buffer_rule: Literal[
        "threshold_unchanged",
        "threshold_resets_to_balance_minus_trail",
        "locked_at_starting_balance",
    ]
    payout_processing: DurationRule | None

class FeeSchedule(FrozenContract):
    evaluation_fee: float
    activation_fee: float
    recurring_fee: float
    recurring_period: DurationRule | None
    reset_fee: float

class WithdrawalPolicyPayload(FrozenContract):
    policy_key: str
    behavior: Literal[
        "request_at_first_eligibility_max",
        "retain_minimum_buffer",
        "fixed_cadence",
        "partial_fixed_amount",
        "max_allowed_each_period",
    ]
    minimum_buffer_retained: float | None
    cadence: DurationRule | None
    partial_amount: float | None
    post_payout_derisk_ref: str | None

class WithdrawalPolicyEnvelope(FrozenContract):
    withdrawal_policy_id: str
    payload: WithdrawalPolicyPayload

class PropFirmContractPayload(FrozenContract):
    firm: str
    account_type: str
    account_size_label: str
    contract_version: str
    effective_date: str
    source_status: Literal["official_document", "checkout_screen", "user_reported", "unverified"]
    source_references: tuple[str, ...]
    verification_status: Literal[
        "synthetic_fixture_verified",
        "first_party_evidence_compiled",
        "owner_reviewed",
        "first_party_verified",
        "superseded",
    ]
    currency: Literal["USD"] = "USD"
    timezone: str = "America/New_York"
    trading_day_boundary: Literal["18:00_ET_roll"] = "18:00_ET_roll"
    evaluation: PhaseRules
    funded: PhaseRules | None
    payout: PayoutPolicy | None
    fees: FeeSchedule
    rule_path_requirements: tuple[PropRulePathRequirement, ...]
    breach_reasons_supported: tuple[str, ...]
    account_expiration: DurationRule | None

class PropFirmContractEnvelope(FrozenContract):
    firm_contract_id: str
    payload: PropFirmContractPayload
    contract_evidence_bundle_id: str
```

Firm rules state whether unrealized equity counts and when it is observed. Adverse-first or favorable-first ordering is never embedded in the firm contract; it belongs only to an explicit scenario policy. Synthetic contracts test the compiler and engine but never become `first_party_verified`.

### 5.3 Account state and typed event bodies

```python
class AccountPhase(StrEnum):
    EVALUATION = "evaluation"
    FUNDED = "funded"
    BREACHED = "breached"
    EXPIRED = "expired"
    RETIRED = "retired"

class PropAccountState:
    phase: AccountPhase
    balance: float
    equity: float
    high_water_mark: float
    drawdown_floor: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    available_buffer: float
    contract_allowance: int
    winning_day_count: int
    consistency_ok: bool
    payout_eligible: bool
    payout_available: float
    post_payout_floor: float | None
    fees_paid_total: float
    breached: bool
    breach_reason: str | None
    account_age_days: int
    days_in_phase: int

class PropFeeEvent(FrozenContract):
    fee_kind: Literal["evaluation", "activation", "recurring", "reset"]
    amount: float

class PropPayoutEvent(FrozenContract):
    requested_amount: float
    approved_amount: float
    trader_amount: float
    firm_amount: float

class PropBreachEvent(FrozenContract):
    breach_reason: str
    threshold_value: float
    observed_equity: float

class PropReplacementEvent(FrozenContract):
    prior_account_id: str
    replacement_account_id: str
    reset_fee: float

class PropPhaseTransitionEvent(FrozenContract):
    from_phase: AccountPhase
    to_phase: AccountPhase
    reason: str

class PropThresholdRatchetEvent(FrozenContract):
    prior_floor: float
    new_floor: float
    reference_equity: float

class PropDailyHaltEvent(FrozenContract):
    halt_reason: str
    halt_until_trading_day: str

class PropEquityUpdateEvent(FrozenContract):
    prior_equity: float
    new_equity: float
    realized_delta: float
    unrealized_delta: float

PropAccountEventBody = (
    PropFeeEvent | PropPayoutEvent | PropBreachEvent | PropReplacementEvent |
    PropPhaseTransitionEvent | PropThresholdRatchetEvent | PropDailyHaltEvent |
    PropEquityUpdateEvent
)

class PropAccountEventEnvelope(FrozenContract):
    event_id: str
    event_ts_utc: str
    event_ordinal: int
    path_instance_id: str
    account_id: str
    account_ordinal: int
    firm_contract_id: str
    account_phase: AccountPhase
    source_trade_id: str | None
    source_decision_id: str | None
    source_candidate_id: str | None
    source_setup_id: str | None
    source_path_event_id: str | None
    event_type: Literal[
        "fee", "payout", "breach", "replacement", "phase_transition",
        "threshold_ratchet", "daily_halt", "equity_update",
    ]
    event_order_policy_id: str
    payload: PropAccountEventBody
```

`AccountWalk` is deterministic and consumes the ordered trade-path bundle under one account policy. A path event cited by an account event must exist by exact `TradePathEvent.event_id`.

### 5.4 Risk, account legs, portfolio, stress, and simulation identity

```python
class RiskPolicyFamily(StrEnum):
    FIXED_DOLLAR = "fixed_dollar"
    PCT_START_BUFFER = "pct_start_buffer"
    PCT_CURRENT_BUFFER = "pct_current_buffer"
    FIXED_NQ = "fixed_nq"
    FIXED_MNQ = "fixed_mnq"
    NQ_MNQ_ADAPTIVE = "nq_mnq_adaptive"
    CUSTOM_CONTRACTS = "custom_contracts"

class PostLossRule(FrozenContract):
    consecutive_losses: int
    risk_multiplier: float
    recovery_wins: int

class PropRiskPolicyPayload(FrozenContract):
    policy_version: str
    family: RiskPolicyFamily
    initial_risk_dollars: float | None
    risk_pct: float | None
    fixed_contracts: int | None
    instrument: Literal["NQ", "MNQ", "adaptive"]
    point_value_nq: float = 20.0
    point_value_mnq: float = 2.0
    max_contracts: int | None
    daily_stop_dollars: float | None
    daily_stop_r: float | None
    max_current_buffer_usage_pct: float | None
    min_remaining_buffer_dollars: float | None
    post_loss_adjustment: PostLossRule | None
    post_payout_derisk_factor: float | None
    skip_if_min_contract_exceeds_budget: bool = True

class PropRiskPolicyEnvelope(FrozenContract):
    risk_policy_id: str
    payload: PropRiskPolicyPayload

class AccountPolicySetPayload(FrozenContract):
    firm_contract_id: str
    risk_policy_id: str
    withdrawal_policy_id: str
    replacement_policy: Literal["none", "auto_replace_up_to_n"]
    max_replacements: int
    clock_policy_id: str

class AccountPolicySetEnvelope(FrozenContract):
    account_policy_set_id: str
    payload: AccountPolicySetPayload

class PortfolioLeg(FrozenContract):
    leg_id: str
    account_policy_set_id: str
    n_accounts: int

class PortfolioPolicyPayload(FrozenContract):
    legs: tuple[PortfolioLeg, ...]
    copied_market_path_policy: Literal["one_common_correlated_path_v1"]

class PortfolioPolicyEnvelope(FrozenContract):
    portfolio_policy_id: str
    payload: PortfolioPolicyPayload

class StressScenarioSpec(FrozenContract):
    scenario_id: str
    name: str
    params: ImmutableMap[str, float]
    seed_offset: int

class SimulationPathRecord(FrozenContract):
    path_instance_id: str
    sampled_index_sequence_hash: str
    draw_ordinal: int

class AccountSimulationPayload(FrozenContract):
    core_replay_id: str
    gross_trade_stream_hash: str
    costed_evaluation_id: str
    trade_path_bundle_id: str
    trade_path_bundle_manifest_sha256: str
    path_capability_report_id: str
    account_policy_set_id: str
    simulation_mode: str
    intrabar_scenario_policy_id: str | None
    bootstrap_protocol_id: str | None
    stress_scenario_id: str | None
    seed: int
    n_paths: int

class AccountSimulationEnvelope(FrozenContract):
    account_simulation_id: str
    payload: AccountSimulationPayload

class PortfolioSimulationPayload(FrozenContract):
    core_replay_id: str
    gross_trade_stream_hash: str
    costed_evaluation_id: str
    trade_path_bundle_id: str
    trade_path_bundle_manifest_sha256: str
    path_capability_report_id: str
    portfolio_policy_id: str
    resolved_legs: tuple[PortfolioLeg, ...]
    simulation_mode: str
    intrabar_scenario_policy_id: str | None
    bootstrap_protocol_id: str | None
    stress_scenario_id: str | None
    seed: int
    n_paths: int

class PortfolioSimulationEnvelope(FrozenContract):
    portfolio_simulation_id: str
    payload: PortfolioSimulationPayload
```

Every result-changing account policy is attached to a leg or account simulation. Mixed-firm portfolios therefore have a complete identity. Duplicate sampled index sequences are legal; every draw still has a unique `path_instance_id`.

### 5.5 Contract-evidence compiler

```python
class PropContractSourceDocument(FrozenContract):
    source_document_id: str
    content_sha256: str
    provenance_url: str
    retrieved_at_utc: str
    effective_from: str | None
    effective_to: str | None
    document_kind: Literal["official_rules", "official_faq", "checkout_screen", "official_terms"]

class PropRuleEvidence(FrozenContract):
    field_path: str
    source_document_id: str
    locator: str
    normalized_value_json: str
    reviewer: str | None
    conflict_status: Literal["none", "conflicting_sources", "missing_effective_date", "unresolved"]

class PropContractDraft(FrozenContract):
    contract_payload: PropFirmContractPayload
    evidence_rows: tuple[PropRuleEvidence, ...]

class PropContractCompilationReport(FrozenContract):
    draft_hash: str
    field_coverage_complete: bool
    unresolved_conflicts: tuple[str, ...]
    source_document_ids: tuple[str, ...]
    passed: bool

class PropContractReviewDecision(FrozenContract):
    draft_hash: str
    owner_decision_ref: OwnerDecisionEvidenceRef
    verdict: Literal["approved", "rejected", "needs_revision"]

class PropContractSupersession(FrozenContract):
    prior_firm_contract_id: str
    replacement_firm_contract_id: str
    reason: str
    effective_at: str
```

Every normalized field carries first-party provenance, locator, effective interval, reviewer, and conflict state. Runtime uses only frozen compiled envelopes and never scrapes live pages.

### 5.6 Result contracts

```python
class EvaluationFitness(FrozenContract):
    pass_probability: float
    median_days_to_pass: float | None
    breach_probability: float
    expiration_probability: float
    expected_fees_paid: float

class FundedFitness(FrozenContract):
    first_payout_probability_30d: float
    first_payout_probability_60d: float
    three_payout_probability: float
    breach_probability_90d: float
    median_account_lifetime_days: float | None

class CashExtraction(FrozenContract):
    expected_net_payout_90d: float
    p10_net_payout_90d: float
    median_days_between_payouts: float | None
    p90_payout_drought_days: float | None
    expected_replacement_cost: float

class PortfolioFitness(FrozenContract):
    probability_at_least_one_payout_90d: float
    probability_all_accounts_breach_90d: float
    expected_portfolio_net_payout_90d: float
    p10_portfolio_net_payout_90d: float
    payout_concentration_by_firm: tuple[tuple[str, float], ...]

class PayoutReliabilityVector(FrozenContract):
    first_payout_probability_30d: float
    first_payout_probability_60d: float
    three_payout_probability: float
    payout_probability_per_rolling_30d: float
    median_days_between_payouts: float | None
    p90_payout_drought_days: float | None
    expected_net_payout_90d: float
    p10_net_payout_90d: float
    breach_probability_90d: float
    expected_replacement_cost: float
```

Account and portfolio result envelopes reference the applicable simulation ID and these immutable metric blocks. No result-changing constructor-only argument is permitted.

## 6. Verification policy and run identity

The verification program uses one canonical real-data allowlist, at most five authorized trading days in total, reused by every release. The real slice is one exact baseline profile with its profile-matching seed. Multi-child and missing behavioral branches are synthetic.

```python
class VerificationRunPayload(FrozenContract):
    pipeline_semantic_id: str
    verification_policy_id: Literal["verification_fixed_allowlist_max5_v1"]
    verification_authorization: VerificationAuthorizationRef
    allowlist: tuple[str, ...]
    allowlist_hash: str
    seed_snapshot_id: str
    baseline_profile_id: str
    baseline_section_config_hash: str
    coverage_matrix_artifact_id: str
    output_namespace: Literal["search_test/v1"]
    gate_policy_id: Literal["verification_control_flow_gates_v1"]

class VerificationRunEnvelope(FrozenContract):
    verification_run_id: str
    payload: VerificationRunPayload
```

Before any source path is constructed, validation asserts that the authorization reference, allowlist hash, seed snapshot, coverage matrix, profile, and pipeline semantic identity agree exactly. Reports stamp `verification_only=true`, `not_for_research_interpretation=true`, and `full_pipeline_not_run=true`. The synthetic marker is invalid for this run.

## 7. Pipeline contracts

```python
class PipelineRunScope(StrEnum):
    VERIFICATION_5D = "verification_5d"
    FULL_AUTHORIZED_DEVELOPMENT = "full_authorized_development"

class QuantLabPipelineStage(StrEnum):
    S00_VALIDATE_INPUTS = "00_validate_inputs"
    S01_PREPARE_STRATEGY_PROFILES = "01_prepare_strategy_profiles"
    S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS = "02_run_or_reuse_sequential_replays"
    S03_BUILD_OR_REUSE_FSM_AUDIT = "03_build_or_reuse_fsm_audit"
    S04_BUILD_OR_REUSE_REPLAY_CHARTS = "04_build_or_reuse_replay_charts"
    S05_MATERIALIZE_FEATURE_VIEWS = "05_materialize_feature_views"
    S06_VALIDATE_FEATURE_COVERAGE = "06_validate_feature_coverage"
    S07_DERIVE_LABELS = "07_derive_labels"
    S08_BUILD_FOLDS = "08_build_folds"
    S09_TRAIN_MODELS = "09_train_models"
    S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS = "10_generate_predictions_and_diagnostics"
    S11_RUN_FROZEN_MODEL_GATED_REPLAYS = "11_run_frozen_model_gated_replays"
    S12_RUN_PROP_HISTORICAL_REPLAYS = "12_run_prop_historical_replays"
    S13_RUN_BOOTSTRAP_AND_STRESS = "13_run_bootstrap_and_stress"
    S14_BUILD_FRONTIER_AND_INSIGHTS = "14_build_frontier_and_insights"
    S15_VERIFY_AND_PUBLISH = "15_verify_and_publish"

class StageStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    REUSED = "reused"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED_AT_SAFE_BOUNDARY = "cancelled_at_safe_boundary"
    BLOCKED = "blocked"

class WorkerPolicy(FrozenContract):
    max_workers: int
    max_tasks_per_child: int
    memory_budget_bytes: int
    start_method: Literal["spawn"] = "spawn"

class PipelineSemanticSpecPayload(FrozenContract):
    run_scope: PipelineRunScope
    date_allowlist: tuple[str, ...]
    allowlist_hash: str
    warmup_policy_id: str
    search_charter_id: str | None
    source_artifact_ids: tuple[str, ...]
    feature_bundle_ids: tuple[str, ...]
    label_policy_id: str | None
    fold_protocol_id: str | None
    model_protocol_id: str | None
    cost_policy_sha256: str
    account_policy_set_ids: tuple[str, ...]
    portfolio_policy_ids: tuple[str, ...]
    simulation_protocol: SimulationProtocol
    software_commits: ImmutableMap[str, str]
    stage_plan: tuple[QuantLabPipelineStage, ...]

class PipelineSemanticIdentity(FrozenContract):
    pipeline_semantic_id: str
    payload: PipelineSemanticSpecPayload

class ExecutionAttemptIdentity(FrozenContract):
    pipeline_semantic_id: str
    worker_policy: WorkerPolicy
    attempt_number: int
    host_environment: ImmutableMap[str, str]
    started_at: str
    ended_at: str | None
    operational_retry_reason: str | None
```

Verification scope additionally requires a matching `VerificationRunEnvelope`. Stage and result identities key on the semantic pipeline ID; operational retries use new attempt identities without changing scientific identity. S11 remains blocked until a rejected-candidate policy is owner-ratified and sequential golden tests pass. Operator availability is capability-scoped by the selected stage plan.

## 8. Stores and the concurrency-safe catalog (complete)

Immutable stores on the verified manifest protocol (identity → refuse-if-exists → tmp-dir write + PK/FK/reconciliation validation → manifest with `manifest_payload_sha256` + `RepositoryState` → `os.replace` → reload → assert): `core_replays`, `memberships`, `costed_evaluations`, `charters`, `prop_contracts`, `contract_evidence`, `risk_policies`, `withdrawal_policies`, `account_policy_sets`, `portfolio_policies`, `trade_path_artifacts`, `trade_path_bundles`, `account_replays`, `account_simulations`, `portfolio_simulations`, `frontiers`, `insights`, `search_results` under `data/ifvg_datasets/search/v1/` (test namespace `search_test/v1/`). Child v2 replay tables use the existing heavyweight saver; the stores hold envelopes + references. Mutable catalog = lock-guarded append-only `catalog_events.jsonl` (O_EXCL appends; `{event_id, ts, kind: display_name|note|star|archive, artifact_id, payload, writer_pid}`) + deterministic `rebuild_catalog_index(events, manifests)`; never holds research metrics; concurrent-publisher and torn-write recovery tests.

## 9. Cohorts and model-gated semantics

```python
class InterpretationMode(StrEnum):
    DESCRIPTIVE_SLICE = "descriptive_slice"
    SPECIALIZED_MODEL = "specialized_model"
    SEQUENTIAL_STRATEGY_PROFILE = "sequential_strategy_profile"

class RegimeFilterRef(FrozenContract):
    resolved_regime_protocol_id: str
    regime_fit_ids: tuple[str, ...]
    canonical_reporting_cluster_ids: tuple[int, ...]
    assignment_partition: Literal["test_oos"] = "test_oos"

class CohortPayload(FrozenContract):
    interpretation_mode: InterpretationMode
    date_policy_id: str
    warmup_policy: Literal["exclude_warmup_v1", "include_warmup_v1"]
    row_kind: Literal["entry_candidate", "eligible_decision", "executed_trade", "context_bar"]
    session_filter: tuple[str, ...] | None
    direction_filter: Literal["long", "short"] | None
    entry_family_filter: tuple[str, ...] | None
    htf_timeframe_filter: tuple[int, ...] | None
    parent_timeframe_filter: tuple[int, ...] | None
    regime_filter: RegimeFilterRef | None
    evidence_quality_filter: tuple[str, ...] | None
    executed_counterfactual_scope: Literal["executed_only", "counterfactual_only", "both"]
    minimum_coverage: ImmutableMap[str, float]

class CohortEnvelope(FrozenContract):
    cohort_id: str
    payload: CohortPayload

class RejectedCandidatePolicy(StrEnum):
    REJECT_KEEP_SETUP_WAITING = "reject_keep_setup_waiting"
    REJECT_TERMINATE_SETUP_MISSED = "reject_terminate_setup_missed"
    REJECT_CONSUME_ONE_SHOT_TRIGGER = "reject_consume_one_shot_trigger"
    REJECT_RESET_SETUP = "reject_reset_setup"

class DecisionPolicyPayload(FrozenContract):
    decision_policy_key: str
    parameters: ImmutableMap[str, Any]
    rejected_candidate_policy: RejectedCandidatePolicy | None
    rejected_candidate_policy_ratification_ref: str | None
    requires_frozen_model: bool
    requires_model_gated_sequential_replay: bool
    produces_new_trade_stream_hash: bool

    # Validator:
    # - none_diagnostic_only_v1 => no rejection policy, all three flags False;
    # - every execution-affecting policy => rejection policy + ratification ref required,
    #   and all three flags True.

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

class ModelGatedReplayRequest(FrozenContract):
    cell_id: str
    frozen_model_fit_id: str
    resolved_decision_policy_id: str
    resolved_model_schedule_id: str
```

Only `SEQUENTIAL_STRATEGY_PROFILE` creates an executable counterfactual. S11 cannot execute until the resolved decision policy contains an owner-ratified rejection semantic and the sequential golden suite passes.

## 10. MBP-1 ordering, stage cutoffs, and feature-window semantics

Ordering is `source_event_order_key = (ts_event, ts_recv, sequence, source_ordinal)`. The offline R5B materializer reads `ts_recv` directly from MBP-1 parquet; no Strategy-Core change is required.

```python
class StageCutoffKind(StrEnum):
    EXACT_SOURCE_ORDER_KEY = "exact_source_order_key"
    TIMESTAMP_EXCLUSIVE = "timestamp_exclusive"
    COMPLETED_BAR_BOUNDARY = "completed_bar_boundary"
    AMBIGUOUS_SAME_TIMESTAMP = "ambiguous_same_timestamp"

class WindowTriggerSemantics(StrEnum):
    PRE_TRIGGER_EXCLUSIVE = "pre_trigger_exclusive"
    POST_TRIGGER_INCLUSIVE = "post_trigger_inclusive"
    COMPLETED_BAR_AS_OF = "completed_bar_as_of"

class IntervalBound(StrEnum):
    OPEN = "open"
    CLOSED = "closed"

class StageEvidenceCutoff(FrozenContract):
    stage_id: str
    stage_as_of_ts_utc: str
    cutoff_kind: StageCutoffKind
    exact_source_order_key: tuple[str, str, int, int] | None
    completed_bar_close_ts_utc: str | None
    same_timestamp_policy_id: str
    source_evidence_ref: str | None

class Mbp1FeatureWindowSpec(FrozenContract):
    feature_window_key: str
    feature_names: tuple[str, ...]
    from_stage: str | None
    to_stage: str
    lower_bound: IntervalBound
    upper_bound: IntervalBound
    trigger_semantics: WindowTriggerSemantics
    cutoff_policy_id: str
    minimum_event_count: int
    missingness_policy_id: str
```

Every MBP-1 feature belongs to one registered `Mbp1FeatureWindowSpec`, and the complete ordered tuple of window specs is part of the resolved order-flow block identity. `PRE_TRIGGER_EXCLUSIVE` uses `<` on an exact trigger key; `POST_TRIGGER_INCLUSIVE` uses `<=`; timestamp-only cutoffs use strict `ts_event < stage_ts` and exclude every same-timestamp event; completed-bar windows use their registered boundary policy. Missing or ambiguous evidence preserves the candidate with typed nulls, including `same_timestamp_order_unavailable`.

## 11. Study-cell semantic/annotation split (summary; full definitions in `DELTA_TAXONOMY.md` §2)

`StudyCellSemanticPayload` (16 semantic dimensions; engineering protocol excluded) is the only hashed object; `StudyCellAnnotation` (runtime/storage/display) is never hashed; output-changing values belong to their owning semantic protocol. The **concrete strategy-only `DataLineagePayload`** (V3 P0-6) is defined in `DELTA_TAXONOMY.md` §2 — context/feature/label/model/regime references are typed optionals, and a strategy-only cell carries no `ifvg_context_formula_v2` identity.

## 12. Frontier, robustness, insights, providers (complete summaries)

`FrontierResult`: feasible ids, frontier ids, dominance edges, `development_exploratory_representative_id` (publishable status requires an owner-approved outer protocol), highest-payout/reliability/lowest-breach ids, per-firm champions, persisted lexicographic tie-break trace; deterministic O(n²) sort in sorted-id order; the resolved objective policy lives in the charter identity. Robustness: ±1-step neighbor degradation, plateau width, knife-edge warnings, firm sensitivity; `outer_fold_recurrence` schema-reserved. `FailureReason` has **15** values: `replay`, `invariant`, `insufficient_days`, `insufficient_trades`, `negative_expectancy`, `drawdown`, `funded_survival`, `breach`, `fees`, `one_firm`, `stress`, `knife_edge`, `unverified_contract`, `blocked_axis`, `cancelled`. Insights: deterministic template renders with typed categories + `EvidenceRef`s (setup/trade/account-event/child/simulation/axis), match-basis-aware, suppressed for `not_comparable` populations. UI providers surface: registered axis values (never raw overrides), authorization-requirement checklists per run scope, generated-profile capability, trade-path fidelity + scenario labels, verification-authorization state, membership+core-replay linkage, paged child summaries (indexed reads), comparison payloads with `match_basis`, account-event envelopes, pipeline semantic id + attempt history — every error string sanitized.


---

## 13. Frontend presentation contracts

The complete interaction and visual behavior is normative in `FRONTEND_UX_CONTRACT.md`. The following pure contracts live in `src/alpha_lab/agents/data_infra/ifvg/study_status.py` and `study_presentation.py`; they contain no Streamlit objects and are unit-tested independently.

```python
class StudyWorkspaceRoute(StrEnum):
    NEW_STUDY = "new_study"
    ACTIVE_RUNS = "active_runs"
    RESULTS = "results"
    HISTORY = "history"
    CONTEXT_RESEARCH = "context_research"

class DisclosureLevel(StrEnum):
    SUMMARY = "summary"
    ANALYST = "analyst"
    AUDIT = "audit"

class ResultScope(StrEnum):
    CANDIDATE_RESEARCH = "candidate_research"
    ACTUAL_EXECUTED_STRATEGY = "actual_executed_strategy"
    PROP_HISTORICAL_CLOSED_TRADE = "prop_historical_closed_trade"
    PROP_1M_SCENARIO = "prop_1m_scenario"
    PROP_ORDERED_EVENT_REPLAY = "prop_ordered_event_replay"
    BOOTSTRAP_SIMULATION = "bootstrap_simulation"
    STRESS_SIMULATION = "stress_simulation"

class StudyStatusKey(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    QUEUED = "queued"
    RUNNING = "running"
    REPLAY_FAILED = "replay_failed"
    STRATEGY_REJECTED = "strategy_rejected"
    PROP_REJECTED = "prop_rejected"
    ROBUST_FINALIST = "robust_finalist"
    SELECTED_REPRESENTATIVE = "selected_representative"
    SUPERSEDED = "superseded"
    BLOCKED = "blocked"

class StatusPresentation(FrozenContract):
    status_key: StudyStatusKey
    visible_label: str
    glyph: str
    semantic_class: str
    help_text: str

class EmptyStateKey(StrEnum):
    NO_CONFIGURATIONS_PASS = "no_configurations_pass"
    NO_VERIFIED_FIRM_CONTRACT = "no_verified_firm_contract"
    BLOCKED_SEARCH_AXIS = "blocked_search_axis"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    CHILD_REPLAY_FAILED = "child_replay_failed"
    PROP_NOT_RUN_STRATEGY_GATE = "prop_not_run_strategy_gate"
    NO_MODEL_RESULT = "no_model_result"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    PROTECTED_RANGE_REFUSAL = "protected_range_refusal"
    VERIFICATION_AUTHORIZATION_MISSING = "verification_authorization_missing"
    CAPABILITY_PLANNED = "capability_planned"
    LINEAGE_NOT_COMPARABLE = "lineage_not_comparable"

class EmptyStatePresentation(FrozenContract):
    key: EmptyStateKey
    heading: str
    explanation: str
    owning_gate: str | None
    next_action: str | None
```

Rules:

- `StudyStatusKey.SELECTED_REPRESENTATIVE` renders **Development Exploratory Representative** in the development lane; the internal key does not authorize publishable wording.
- Every status is presented with glyph + text + high-contrast semantic styling; color alone is prohibited.
- `ResultScope` is mandatory on result cards/charts/tables and enters presentation snapshots, not scientific identity.
- exact fixed copy, route behavior, fallbacks, wizard fields, chart/table twins, accessibility, viewport QA, and `FUX-*` gates are defined in `FRONTEND_UX_CONTRACT.md`.
- presentation contracts are non-research-bearing; display names/notes remain mutable catalog annotations. They cannot change or replace any artifact identity.
