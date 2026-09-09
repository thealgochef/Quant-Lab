# Agent 2 Revision Amendment V2 — Final Targeted Corrections for the Quant-Lab FSM / Prop Search Plan

**Document type:** Mandatory second-pass planning amendment  
**Target:** Agent 2 revised plan package  
**Status:** Documentation-only correction request  
**Implementation authorization:** None  
**Predecessor:** `IMPLEMENTATION_PLAN_REVISION_REQUEST.md`  
**Basis:** Independent line-by-line re-review of the first revised Agent 2 package  
**Objective:** Correct the remaining identity, authorization, point-in-time, prop-fidelity, regime-schema, MBP-1-delivery, and cross-document consistency defects without redesigning the accepted architecture.

---

# 1. Instructions to Agent 2

The first revision materially improved the plan. Preserve those accepted changes.

This is a **narrow final amendment pass**, not a request to restart planning or restore any superseded design.

Revise the current authoritative documents **in place**. Do not append an errata page while leaving contradictory schemas or milestone text elsewhere.

## 1.1 Documents that must be reviewed and revised

At minimum:

```text
IMPLEMENTATION_PLAN.md
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
ML_REGIME_CONTRACT_PLAN.md
TEST_MATRIX.md
OWNER_DECISIONS.md
PHASED_DELIVERY.md
README.md
REVISION_CHANGELOG.md
```

Revise `ARCHITECTURE_MAP.md` where the corrected replay-input identity, generated-profile capability, MBP-1 materialization, or prop-path capability requires an updated gap or source-of-truth statement.

`FSM-PLAN-DOCUMENT.md` remains provenance-only. It must not become implementation authority.

## 1.2 Documentation-only boundary

Do not:

- modify production code;
- modify tests;
- create a seed;
- run a replay;
- materialize MBP-1 features;
- fit a model or clusterer;
- run a prop simulation;
- mutate an artifact or catalog;
- access new source data;
- access June 11, 2026 or sealed data;
- modify Strategy-Core, Quant-Lab, or Trade-Lab code;
- promote any proposed scientific value into owner authority.

Repository inspection and read-only code verification are permitted.

## 1.3 Required revision outputs

Return:

1. every revised planning document;
2. an updated `REVISION_CHANGELOG.md` containing an **Amendment V2** section;
3. a new `FINAL_CONSISTENCY_AUDIT_V2.md`;
4. the remaining owner-decision list after this amendment;
5. explicit confirmation that no code, tests, artifacts, catalogs, seeds, replays, feature builds, model fits, simulations, or source-data runs occurred.

Stop after the planning documents.

---

# 2. Preserve These Accepted First-Revision Corrections

Do not regress any of the following:

```text
CoreStrategyReplayIdentity separated from SearchChildMembership
stable study-independent canonical profile naming
cost/audit/chart/risk/prop/resource policies excluded from core replay identity
PipelineSemanticIdentity separated from ExecutionAttemptIdentity
typed registered axis values with value-level ratification
parent fill blocked from the first real search
one real baseline five-day slice plus synthetic multi-child verification
profile-matching seed requirement
per-child audit-neutrality report
profile-independent opportunity lineage and explicit match basis
public CohortSpec and interpretation modes
RejectedCandidatePolicy option space with S11 blocked
MBP-1 maximum depth and no MBP-10
trade-path fidelity classes and conservative-approximation labeling
totally ordered account-event envelope
typed day-count semantics
withdrawal policy separated from firm rules
complete prop-simulation policy identity
field-level prop-contract evidence compiler
concurrency-safe rebuildable mutable catalog
bootstrap duplicate paths permitted with unique path-instance IDs
V1 ML scope limited to prevalence/logistic/CatBoost/KMeans
spectral and Nyström contracts preserved as planned capabilities
Development Exploratory Representative wording before outer evaluation
early real baseline vertical slice
fidelity-first prop sequencing
smaller usable releases
```

If any later correction changes the shape of one of these accepted items, preserve its intent and explain the compatible refinement.

---

# 3. Remaining Mandatory P0 Corrections

---

## P0-A — Make core replay identity content-address the complete replay input

### Current defect

The revised `CoreStrategyReplayIdentity` identifies:

```text
date policy / ordered-date hash
artifacts tag / era descriptor
Strategy-Core source
resolved section
seed
resolver
schema versions
```

That is not enough to prove the exact source and day-artifact bytes consumed by the replay.

An access-policy/date hash and an `artifacts_tag()` identify a requested configuration. They do not necessarily identify:

```text
the exact source partitions
the exact cached bar artifacts
the exact level artifacts
the exact per-day manifests
the Quant-Lab replay/capture implementation
```

A source or cached artifact could change while those higher-level tags remain the same.

### Required design

Add a content-addressed input contract, conceptually:

```python
class ReplaySourcePartitionRef(FrozenContract):
    trading_day: str
    instrument: str
    contract_symbol: str | None
    source_kind: str
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

class ReplayInputBundlePayload(FrozenContract):
    authorized_date_set_id: str
    ordered_source_partitions: tuple[ReplaySourcePartitionRef, ...]
    ordered_day_artifacts: tuple[ReplayDayArtifactRef, ...]
    source_contract_id: str
    source_schema_era_id: str
    access_policy_id: str
    access_audit_identity: str

class ReplayInputBundleEnvelope(FrozenContract):
    replay_input_bundle_id: str
    payload: ReplayInputBundlePayload
```

Use repository-native existing manifest/source-hash primitives where possible. Do not invent duplicate file hashing when a verified immutable manifest already provides the content identity.

Revise:

```python
class CoreStrategyReplayIdentity:
    replay_input_bundle_id: str
    quant_lab_replay_source_identity: str
    strategy_core_source_identity: str
    ...
```

`quant_lab_replay_source_identity` must cover the scoped Quant-Lab code that controls:

```text
profile resolution
capture driver
dataset chain
replay adapter
table partitioning
seed chaining
resolver invocation
```

It must include commit/dirty/source-tree evidence under the existing repository-state idiom.

### Identity rule

The core replay ID must change when any of these changes:

```text
source partition bytes
cached bar/level artifact content
source schema/era
Quant-Lab replay/capture code
Strategy-Core replay/reducer code
section semantics
seed
resolver
date set
```

It must not change for:

```text
parent-study membership
display names
cost/risk/prop/payout policies
audit/chart schema
worker count
memory limit
retry metadata
```

### Required tests

Add:

1. source partition content hash changes → new replay-input bundle and core replay ID;
2. one cached day-artifact manifest changes → new ID;
3. Quant-Lab replay source-tree hash changes → new ID;
4. reorder-only changes to a canonically sorted input list are normalized or refused;
5. identical verified input manifests across parent studies reuse one replay;
6. absolute filesystem location changes do not alter identity when the verified manifest content is unchanged.

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
IMPLEMENTATION_PLAN.md
ARCHITECTURE_MAP.md
TEST_MATRIX.md
PHASED_DELIVERY.md
REVISION_CHANGELOG.md
README.md
```

---

## P0-B — Remove annotation-only engineering state from `StudyCellIdentity`

### Current defect

The dimension registry declares:

```text
EngineeringProtocolIdentity.identity_effect = annotation_only
```

while the current `StudyCellIdentity.cell_id` hashes the complete object containing all dimensions.

That means an annotation-only runtime/storage field can still change the semantic study-cell ID.

### Required design

Separate semantic identity from annotation:

```python
class StudyCellSemanticPayload(FrozenContract):
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

class StudyCellIdentity(FrozenContract):
    cell_id: str
    payload: StudyCellSemanticPayload

class StudyCellAnnotation(FrozenContract):
    cell_id: str
    engineering_protocol: EngineeringProtocolAnnotation
    display_metadata: Mapping[str, Any]
```

A function equivalent to:

```python
study_cell_id(payload) = canonical_contract_sha256(payload)
```

must hash only semantic dimensions.

### Classification rule

If an engineering/environment value can change numerical research output, it is not annotation-only. Place it in the relevant semantic protocol:

```text
model package version -> ModelProtocolIdentity
resolver version -> ExecutionPolicy/CoreReplayIdentity
simulation numeric policy -> PropSimulationIdentity
bootstrap implementation/protocol -> ValidationProtocolIdentity
```

Only runtime and storage operation details remain annotations or execution-attempt evidence.

### Required tests

1. changing workers, storage root, page size, or runtime estimate leaves `cell_id` unchanged;
2. changing a model package/protocol version changes the correct semantic identity;
3. every registry dimension marked `annotation_only` is absent from `StudyCellSemanticPayload`;
4. the UI can display engineering annotations without changing comparison compatibility.

### Documents affected

```text
DELTA_TAXONOMY.md
CONTRACTS_AND_SCHEMAS.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
REVISION_CHANGELOG.md
```

---

## P0-C — Adopt one non-self-referential identity payload convention

### Current defect

Several contracts place their own derived IDs or artifact hashes inside the object that may be canonically hashed:

```text
ComparisonSpec.comparison_id
CohortSpec.cohort_id
RegimeModelSpec.regime_model_id
RegimeModelSpec.artifact_hash
other id-bearing specs
```

This creates circular or ambiguous identity rules.

### Required convention

Use one uniform pattern throughout the new lane:

```python
class ComparisonPayload(FrozenContract):
    ...

class ComparisonEnvelope(FrozenContract):
    comparison_id: str
    payload: ComparisonPayload
```

Equivalent pairs are required for at least:

```text
StudyCell
Comparison
Cohort
RegimeModel
RegimeFit
FeatureBlock
FeatureBundle
SearchCharter
PropSimulation
PipelineSemanticSpec
```

Alternatively, define an explicit `identity_payload()` method, but it must be field-auditable and consistent across contracts.

### Identity payloads must exclude

```text
their own derived ID
artifact hash derived from the same payload
manifest payload hash
mutable display metadata
annotation-only fields
execution-attempt metadata
timestamps that do not define research semantics
absolute local paths
```

Artifact content hashes are calculated after materialization and belong in the artifact envelope/manifest, not the pre-run semantic specification.

### Required tests

Add an identity-projection audit that:

1. enumerates every new ID-producing contract;
2. verifies its identity payload contains no self-ID field;
3. verifies no annotation/attempt/display field enters;
4. verifies payload → ID → envelope → reload determinism;
5. verifies an artifact content hash is not used to define the pre-materialization semantic ID that generated it.

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
ML_REGIME_CONTRACT_PLAN.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
```

---

## P0-D — Define a generated-search-profile capability contract

### Current defect

The baseline profile must be checked through the fixed `PROFILE_CAPABILITY_REGISTRY`, which is correct.

A generated child such as:

```text
ifvg_search_profile_<semantic-hash>
```

will not be a pre-existing fixed M0–M3 registry key.

The current study-cell text still implies every `StrategyProfileIdentity.profile_name` is gated by that fixed registry.

### Required design

Add:

```python
class ResolvedSearchProfileRef(FrozenContract):
    canonical_profile_id: str
    baseline_profile_id: str
    resolved_section_config_hash: str
    axis_value_ids: Mapping[str, str]
    owner_authorization_id: str
    profile_capability_id: str

class GeneratedProfileCapability(FrozenContract):
    capability_id: str
    status: Literal[
        "generated_runnable",
        "blocked_owner_decision",
        "blocked_invalid_section",
        "blocked_invariant_failure",
        "blocked_base_profile",
    ]
    baseline_capability_ref: str
    registry_hash: str
    authorization_ref: str
    validation_report_ref: str | None
    reason: str | None
```

### Rules

A generated child is eligible for replay only when:

```text
the baseline profile is registered RUNNABLE
all axis/value IDs are registered
all required values are owner-authorized for the run scope
locked invariants remain unchanged
the resolved section validates
the generated canonical profile name/hash is deterministic
```

The child does **not** need to be inserted into or pre-listed in the fixed M0–M3 profile registry.

After replay:

```text
invariant and audit-neutrality failures block artifact publication
```

### Required tests

1. runnable baseline + ratified values → generated-runnable child;
2. blocked baseline → child blocked;
3. unratified value → child blocked before replay;
4. generated profile absent from fixed registry does not fail merely because it is generated;
5. a generated child cannot enter the legacy fixed-tier M0–M3 launcher unless a separate compatibility adapter explicitly supports it.

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
ARCHITECTURE_MAP.md
```

---

## P0-E — Make owner authorization computation-path scoped

### Current defect

The current fixed `OwnerAuthorizationBundle` requires references for prop, regime, validation, verification, and other decisions even when the selected study does not use those capabilities.

It also assigns the synthetic authorization marker to `verification_5d`, even though the real verification path reads authorized real data.

### Required design

Replace the fixed all-capabilities bundle with:

```python
class AuthorizationRequirement(FrozenContract):
    decision_key: str
    reason: str
    required_for_stage_ids: tuple[str, ...]
    required_for_dimension_ids: tuple[str, ...]
    required_for_scope: tuple[str, ...]

class AuthorizationRequirementSet(FrozenContract):
    requirement_set_id: str
    requirements: tuple[AuthorizationRequirement, ...]

class OwnerAuthorizationBundle(FrozenContract):
    requirement_set_id: str
    decision_refs: Mapping[str, OwnerDecisionEvidenceRef]

class VerificationAuthorizationRef(FrozenContract):
    verification_policy_id: str
    approved_allowlist_hash: str
    coverage_matrix_artifact_id: str
    seed_snapshot_id: str
    approved_by: str
    approved_at: str
    content_hash: str
```

Define:

```python
derive_authorization_requirements(
    run_scope,
    study_dimensions,
    computation_path,
    enabled_pipeline_stages,
    selected_firms,
) -> AuthorizationRequirementSet
```

### Examples

```text
Strategy-only FSM search:
    requires axis/value, locked-invariant, date, baseline/profile, strategy-gate decisions
    does not require firm, withdrawal, prop-gate, or regime decisions

Single frozen prop benchmark:
    adds firm contract, fidelity, risk, withdrawal, clock, and prop-gate decisions

Regime descriptive study:
    requires regime grain/count/stability protocol
    does not require RejectedCandidatePolicy

Model-gated sequential replay:
    requires RejectedCandidatePolicy and frozen model/decision protocol

Verification real slice:
    requires VerificationAuthorizationRef
    never uses SyntheticAuthorizationMarker

Fully synthetic orchestration fixture:
    uses SyntheticAuthorizationMarker
```

### Validation rule

A charter fails only when a decision required by its actual computation path is absent or invalid.

Do not require unrelated owner decisions.

### Required tests

Parameterize study modes and assert the exact requirement set for:

```text
single config
FSM strategy search
feature-only comparison
regime descriptive
model fit
model-gated replay
prop benchmark
universal prop search
verification_5d
synthetic orchestration
```

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
OWNER_DECISIONS.md
PHASED_DELIVERY.md
README.md
REVISION_CHANGELOG.md
```

---

## P0-F — Reclassify the five-day fixture as a release-verification blocker

### Current defect

The fixture allowlist and coverage sign-off are classified only as `BLOCKING-RESEARCH`.

Release 1 cannot close without the real vertical slice.

### Required classification

Add:

```text
BLOCKING-VERIFICATION
```

or:

```text
BLOCKING-RELEASE
```

Definition:

```text
does not block code authoring;
does block the corresponding release acceptance gate and every dependent release
```

Reclassify:

```text
Owner decision 21
R-5 fixture coverage sign-off
VerificationAuthorizationRef approval
```

### Correct wording

Replace statements equivalent to:

```text
none of it blocks the build
```

with:

```text
nothing currently blocks documentation approval or code authoring;
the approved five-day fixture and coverage evidence block Release 1 acceptance,
and Release 1 gates every subsequent release.
```

Do not imply that implementation can be fully verified without the owner-approved real fixture.

### Documents affected

```text
OWNER_DECISIONS.md
README.md
IMPLEMENTATION_PLAN.md
PHASED_DELIVERY.md
TEST_MATRIX.md
REVISION_CHANGELOG.md
```

---

## P0-G — Correct MBP-1 point-in-time stage-cutoff semantics

### Current defect

The event ordering key is now correctly proposed as:

```text
(ts_event, ts_recv, sequence, source_ordinal)
```

But the current stage cutoff uses a timestamp converted to:

```text
(stage_ts, +inf, +inf, +inf)
```

and includes every source event whose key is less than that upper bound.

That admits MBP-1 events with the same event timestamp that arrived or occurred after the stage decision.

### Required design

Add:

```python
class StageCutoffKind(StrEnum):
    EXACT_SOURCE_ORDER_KEY = "exact_source_order_key"
    TIMESTAMP_EXCLUSIVE = "timestamp_exclusive"
    COMPLETED_BAR_BOUNDARY = "completed_bar_boundary"
    AMBIGUOUS_SAME_TIMESTAMP = "ambiguous_same_timestamp"

class StageEvidenceCutoff(FrozenContract):
    stage_id: str
    stage_as_of_ts_utc: str
    cutoff_kind: StageCutoffKind
    exact_source_order_key: tuple[str, str, int, int] | None
    completed_bar_close_ts_utc: str | None
    same_timestamp_policy_id: str
    source_evidence_ref: str | None
```

### Inclusion rules

#### Exact source key available

```text
include source_event_order_key < exact_stage_source_order_key
```

Use the exact source event that made the stage observable, not an artificial `+inf` bound.

#### Only stage timestamp available

Use:

```text
ts_event < stage_ts
```

All events with `ts_event == stage_ts` are excluded or typed as ambiguous/unavailable.

#### Completed bar close

Use a versioned bar-boundary rule tied to the exact completed bar and feed timestamp semantics. Do not assume all events with a numerically equal timestamp were known before the bar-close decision.

### Missingness

When exact same-timestamp ordering is unavailable:

```text
preserve the candidate/stage row
emit typed missing reason:
    same_timestamp_order_unavailable
```

Do not widen the window.

### Required tests

1. multiple events share `ts_event`; only earlier `ts_recv/sequence/source_ordinal` events enter under an exact key;
2. timestamp-only cutoff excludes every same-timestamp event;
3. a same-timestamp event after the stage cannot change a feature;
4. batch/repeat materialization produces identical windows;
5. no `+inf` cutoff construction remains in the source;
6. MBP-1 remains the deepest representable level.

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
TEST_MATRIX.md
IMPLEMENTATION_PLAN.md
PHASED_DELIVERY.md
ARCHITECTURE_MAP.md
REVISION_CHANGELOG.md
```

---

## P0-H — Make prop path-fidelity labels historically truthful

### Current defect

The plan correctly relabels MFE/MAE adverse-first as a conservative approximation.

It still describes:

```text
historical_1m_path
ordered_1m_bar_path
```

as exact under a declared intrabar policy.

A 1-minute OHLC bar does not contain the observed order of high and low. An assumed ordering policy produces a deterministic scenario, not historical event chronology.

### Required fidelity model

Use fidelity classes equivalent to:

```python
class TradePathFidelity(StrEnum):
    CLOSED_TRADE_ONLY = "closed_trade_only"
    OHLC_1M_UNORDERED = "ohlc_1m_unordered"
    ASSUMED_1M_INTRABAR_PATH = "assumed_1m_intrabar_path"
    ORDERED_MBP1_EVENT_PATH = "ordered_mbp1_event_path"
    ORDERED_FILL_EVENT_PATH = "ordered_fill_event_path"
```

Represent a raw 1-minute observation honestly:

```python
class OhlcBarPathObservation(FrozenContract):
    open_ts_utc: str
    close_ts_utc: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    observed_intrabar_order: Literal["unknown"] = "unknown"
```

A scenario expander may generate assumed subevents under:

```text
bar_adverse_extreme_first_v1
bar_favorable_extreme_first_v1
another registered scenario policy
```

Those generated paths remain `ASSUMED_1M_INTRABAR_PATH`.

### Simulation-mode wording

Use:

```text
historical_closed_trade
historical_1m_scenario
historical_ordered_event_replay
day_block_bootstrap
stress
```

Only an actual ordered MBP-1/tick/fill event stream may be described as historical ordered event replay.

### Per-rule capability

Every prop rule must specify its minimum fidelity.

Examples:

```text
rule insensitive to intrabar extrema ordering:
    may accept OHLC_1M_UNORDERED

rule requiring exact intraday peak trail or DLL touch chronology:
    requires ORDERED_MBP1_EVENT_PATH or another actual ordered source

assumed 1m path:
    scenario/approximation only
```

Do not set one universal fidelity floor for all rules.

### Prop identity correction

Add the exact evidence reference:

```python
class PropSimulationIdentity:
    trade_path_artifact_id: str
    trade_path_manifest_sha256: str
    trade_path_fidelity: TradePathFidelity
    intrabar_scenario_policy_id: str | None
    ...
```

A path-policy ID without the actual path artifact identity is insufficient.

### Owner decision R-4

Revise R-4 from:

```text
ordered_1m_bar_path is proposed for any publishable intraday-rule result
```

to:

```text
owner approves a per-rule path-fidelity matrix;
publishable historical claims require actual ordered evidence for every chronology-sensitive rule;
1m assumed paths remain scenario evidence.
```

### Required tests

1. a 1m OHLC record cannot claim observed high/low ordering;
2. two assumed intrabar policies produce two separate scenario identities/results;
3. a chronology-sensitive rule refuses OHLC-only evidence;
4. an insensitive rule may run with lower fidelity;
5. the UI/report uses `scenario` or `approximation`, never `exact historical`, for assumed 1m paths;
6. changing only the actual path artifact changes `PropSimulationIdentity`.

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
OWNER_DECISIONS.md
PHASED_DELIVERY.md
DELTA_TAXONOMY.md
REVISION_CHANGELOG.md
README.md
```

---

# 4. Required P1 Corrections

---

## P1-A — Distinguish synthetic contract verification from first-party contract verification

### Current contradiction

Release 3 is described as delivering:

```text
one real firm contract
```

while its acceptance fixture compiles a synthetic source-document set.

### Required statuses

Add:

```text
synthetic_fixture_verified
first_party_evidence_compiled
owner_reviewed
first_party_verified
superseded
```

A synthetic source bundle can prove:

```text
compiler behavior
conflict handling
field provenance
account-engine integration
```

It cannot make a real firm contract current or verified.

### Release wording

Release 3 gate:

```text
one synthetic contract fixture is compiled and simulated end-to-end
```

Optional real integration gate:

```text
one first-party evidence-compiled and owner-reviewed contract
```

That optional real gate is blocked by owner decisions 5/6 and is not required to prove the code architecture.

### Documents affected

```text
PHASED_DELIVERY.md
CONTRACTS_AND_SCHEMAS.md
OWNER_DECISIONS.md
TEST_MATRIX.md
IMPLEMENTATION_PLAN.md
README.md
```

---

## P1-B — Add the completed context-panel grain to the actual regime schema

### Current contradiction

The narrative supports completed 5m/15m context panels, but the enum/spec currently exposes only candidate, decision, and `CONTEXT_BAR_1M`.

### Required schema

```python
class ObservationGranularity(StrEnum):
    CANDIDATE_STAGE_ROW = "candidate_stage_row"
    DECISION_ROW = "decision_row"
    CONTEXT_BAR_PANEL = "context_bar_panel"

class RegimeModelPayload(FrozenContract):
    observation_granularity: ObservationGranularity
    panel_interval_seconds: int | None
    panel_source_artifact_id: str | None
    panel_as_of_policy_id: str | None
    ...
```

Validation:

```text
CONTEXT_BAR_PANEL:
    requires an owner-registered interval and source artifact
    requires completed point-in-time bars
    requires a candidate-assignment policy

other grains:
    panel fields must be null
```

If 1m remains desirable, represent it as `panel_interval_seconds=60`; do not create contradictory enum and narrative names.

### Required tests

```text
5m panel
15m panel
candidate-stage
decision-row
invalid panel field combinations
point-in-time assignment from panel to candidate stage
```

### Documents affected

```text
ML_REGIME_CONTRACT_PLAN.md
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
TEST_MATRIX.md
OWNER_DECISIONS.md
```

---

## P1-C — Correct planned spectral/Nyström enforcement wording

### Current contradiction

The registry marks direct spectral and Nyström as planned for a later release, but the current text says their restrictions are enforced in code.

### Required V1 wording

```text
V1 ships capability-registry entries and fail-closed planned/blocked UI states.
No direct spectral or Nyström fit implementation is callable in V1.
Their future implementations must satisfy the frozen restrictions and tests.
```

The V1 test checks:

```text
planned algorithm request is refused with the correct status/reason
training-only spectral cannot enter a predictive bundle
algorithm identities remain distinct in the registry
```

The regime-expansion release adds the actual fit and algorithm-specific numerical tests.

### Documents affected

```text
ML_REGIME_CONTRACT_PLAN.md
TEST_MATRIX.md
PHASED_DELIVERY.md
README.md
REVISION_CHANGELOG.md
```

---

## P1-D — Explicitly assign MBP-1 feature materialization to a release

### Current gap

`IFVG_ORDER_FLOW_MBP1_V1` is still planned.

Release 5 is named an MBP-1 feature platform but does not clearly list the source artifact, materializer, stage-window builder, schema, coverage report, or controlled baseline-versus-MBP-1 study.

### Required design decision

Choose and document one of these truthful options.

## Option 1 — Release 5 is contract/integration readiness only

Rename it:

```text
Pipeline runner + MBP-1 contract/integration readiness + supervised ladder
```

State that:

```text
no MBP-1 feature bundle can be activated
no baseline-vs-MBP-1 model comparison can run
```

until a later activation release.

## Option 2 — Release 5 activates the offline research block

Add explicit files/responsibilities equivalent to:

```text
features/mbp1_source_artifact.py
features/mbp1_arrow_schemas.py
features/mbp1_stage_windows.py
features/mbp1_feature_materializer.py
features/mbp1_coverage.py
features/mbp1_feature_join.py
```

Required outputs:

```text
immutable MBP-1 source/coverage artifact
exact stage-cutoff evidence
typed missing reasons
feature table with canonical schema/hash
feature coverage report
exact candidate/stage join
activated IFVG_ORDER_FLOW_MBP1_V1 block version
baseline vs baseline+MBP-1 controlled-study workflow
dashboard coverage and comparison panels
```

### Formula ownership and promotion boundary

The plan must explicitly state whether the V1 MBP-1 block is:

```text
research_only_offline
```

or owned by Strategy-Core.

If QL-only:

```text
cannot be promoted to live/training-serving parity
cannot become an execution gate
requires a later Strategy-Core formula/serving contract
```

If Strategy-Core-owned:

```text
requires a separately reviewed Strategy-Core milestone and pin promotion
```

Trade-Lab live serving remains out of scope.

### Verification

Implementation verification remains:

```text
one fixed <=5-day real source/join/UI control-flow fixture
synthetic feature-value fixtures for formulas
no full-development feature materialization
```

### Documents affected

```text
PHASED_DELIVERY.md
IMPLEMENTATION_PLAN.md
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
ML_REGIME_CONTRACT_PLAN.md
TEST_MATRIX.md
ARCHITECTURE_MAP.md
README.md
OWNER_DECISIONS.md if a formula-ownership decision is required
```

---

## P1-E — Fully specify lineage payloads and collision behavior

### Current gap

The revised plan lists lineage ingredients but not the complete canonical key for each entity.

### Required contracts

Define exact payloads, grounded in current records, equivalent to:

```python
class SetupLineagePayload(FrozenContract):
    direction: str
    htf_timeframe_seconds: int
    htf_fvg_id: str
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
```

The exact fields may differ after code inspection. They must be:

```text
profile-independent
point-in-time
source-derived
unique within a replay
stable across profiles when the underlying opportunity is semantically comparable
```

### Collision rules

For every replay/entity kind:

```text
native rows -> lineage key must be one-to-one
```

When two native rows map to one lineage key:

```text
do not deduplicate
do not keep first/last
do not fuzzy-resolve
mark that entity type not_comparable
persist collision evidence
```

Add:

```text
LineageUniquenessReport
LineageCollisionRecord
```

### Required tests

```text
unique valid mapping
intentional collision
direction/family collision prevention
profile change with same opportunity
entry-thesis change -> not comparable when semantics differ
```

### Documents affected

```text
CONTRACTS_AND_SCHEMAS.md
DELTA_TAXONOMY.md
TEST_MATRIX.md
IMPLEMENTATION_PLAN.md
```

---

## P1-F — Remove unproven “ratified variant” language

### Current contradiction

The owner register says no proposal is owner-approved but describes 240/480 as ratified variants.

### Required wording

Use:

```text
existing prepared or implemented variants
```

unless the document cites an immutable owner-decision artifact that ratified the strategy value.

State the proposed scientific contrast explicitly:

```text
baseline = None / unbounded parent retest
challengers = owner-approved bounded values
```

Do not silently omit the accepted unbounded baseline from a staleness comparison.

### Documents affected

```text
OWNER_DECISIONS.md
IMPLEMENTATION_PLAN.md
CONTRACTS_AND_SCHEMAS.md
README.md
```

---

## P1-G — Make v3/context/model identities optional for strategy-only cells

### Current gap

The canonical study-cell data-lineage description assumes a v2/v3 artifact pair.

A valid FSM or prop study may require:

```text
core v2 replay
audit/chart companions
costed execution stream
prop simulation
```

without:

```text
v3 context
feature view
label derivation
fold set
model
regime
```

### Required design

Use optional references or explicit null identities:

```text
none_context_artifact_v1
none_feature_bundle_v1
none_label_view_v1
none_model_protocol_v1
none_regime_model_v1
```

Prefer typed optional payloads where the dimension is genuinely not part of the computation.

The computation path decides when to materialize:

```text
v3 context
feature view
labels
folds
model
regime
```

A strategy-only cell must not be blocked because a v3 pair does not exist.

### Required tests

1. strategy-only replay + prop simulation with no v3/model artifacts;
2. feature-only study requires exact v3/feature refs;
3. model study requires labels/folds/model refs;
4. missing required derived evidence fails, while irrelevant missing evidence does not.

### Documents affected

```text
DELTA_TAXONOMY.md
CONTRACTS_AND_SCHEMAS.md
IMPLEMENTATION_PLAN.md
TEST_MATRIX.md
```

---

# 5. Required Test-Matrix Additions

Add explicit tests for every correction above.

At minimum:

```text
ReplayInputBundle content sensitivity
Quant-Lab replay source identity sensitivity
portable input-manifest identity
StudyCell annotation exclusion
identity self-field exclusion
GeneratedProfileCapability
computation-path-scoped authorization
real VerificationAuthorizationRef
BLOCKING-VERIFICATION classification
MBP-1 same-timestamp future-event exclusion
timestamp-only conservative cutoff
1m OHLC unordered truthfulness
assumed-path scenario identity
per-rule path-fidelity refusal
trade-path artifact identity in simulation
synthetic-vs-first-party contract statuses
regime context-panel schema/validation
planned spectral capability refusal
MBP-1 block activation or explicit planned-state refusal
lineage one-to-one uniqueness and collision refusal
strategy-only cell without v3
```

The test matrix must identify:

```text
test level
fixture
assertion
failure meaning
release gate
```

No new real dates may be added to verification.

---

# 6. Required Owner-Decision Revisions

Update `OWNER_DECISIONS.md` with at least:

```text
verification allowlist and coverage sign-off -> BLOCKING-VERIFICATION / BLOCKING-RELEASE
R-4 -> per-rule path-fidelity matrix; assumed 1m paths are scenarios, not historical truth
MBP-1 formula ownership / research-only promotion boundary, if not purely engineering
first real search baseline includes None/unbounded unless owner explicitly chooses otherwise
scientific timeout values remain proposed until evidence refs exist
capability-specific authorization rather than one universal blocking list
```

The final summary must distinguish:

```text
blocks code authoring
blocks Release 1 verification
blocks a strategy-only real search
blocks real prop use
blocks regime feature eligibility
blocks model-gated execution
blocks publishable representative status
blocks the full operator pipeline for the selected stage plan
```

Do not say every open decision blocks every possible pipeline.

---

# 7. Required Delivery-Plan Revisions

The release plan must show:

## Release 1

Add:

```text
ReplayInputBundleIdentity
Quant-Lab replay source identity
semantic identity projections
GeneratedProfileCapability
VerificationAuthorizationRef
```

Release 1 acceptance is blocked until the real fixture authorization and coverage sign-off exist.

## Release 2

Add:

```text
lineage uniqueness/collision reports
generated-profile capability in orchestration
```

## Release 3

Rename the gate to synthetic contract/lifecycle verification unless first-party evidence actually exists.

Path-fidelity contracts must use truthful 1m scenario labels.

## Release 5

Resolve whether it:

```text
only prepares MBP-1 contracts
```

or:

```text
activates the offline MBP-1 feature block
```

Do not use “MBP-1 feature platform” ambiguously.

## Release 6

Add the actual `CONTEXT_BAR_PANEL` schema.

Planned spectral/Nyström requests are refused in V1; numerical implementation lands only in the expansion sub-release.

---

# 8. Required Cross-Document Consistency Audit V2

Create `FINAL_CONSISTENCY_AUDIT_V2.md`.

For each topic below, list the exact authoritative sections in all affected documents and confirm the same design is expressed everywhere:

```text
replay-input content identity
Quant-Lab/Strategy-Core source identity
semantic vs annotation identity
non-self-referential ID projection
generated profile capability
authorization requirement derivation
real verification authorization
verification-blocker classification
MBP-1 stage cutoff
MBP-1 feature delivery milestone
trade-path fidelity and scenario wording
path artifact identity
synthetic vs first-party contract status
regime panel grain
planned spectral capability behavior
lineage uniqueness/collisions
optional v3/model artifacts
owner blocking summaries
release dependency graph
```

Also run a document-only search for these superseded phrases and confirm they remain only in changelog “previous design” text or provenance-only documents:

```text
stage_ts +inf cutoff
historical_1m_path exact
ordered_1m_bar_path exact
one real firm contract compiled from synthetic sources
every real charter requires all owner decisions
verification_5d uses synthetic authorization
PROFILE_CAPABILITY_REGISTRY gates generated profile name
cell_id hashes annotation_only engineering protocol
ratified variants
MBP-1 feature platform without materializer
```

The prior statement that no contradictions remain must be replaced by the result of this second audit.

---

# 9. Amendment V2 Acceptance Checklist

The revised package is ready for implementation only when:

## Replay and identity

1. Exact replay input bytes/artifact manifests and Quant-Lab replay code enter core replay identity.
2. Annotation-only engineering fields do not enter the semantic study-cell ID.
3. No ID payload contains its own derived ID/hash.
4. Generated profiles have an authorized capability path separate from the fixed M0–M3 registry.
5. Strategy-only cells do not require irrelevant v3/model artifacts.

## Authorization and verification

6. Required owner decisions are derived from the actual computation path.
7. Real five-day verification uses a real `VerificationAuthorizationRef`.
8. Fixture sign-off is classified as a Release 1 blocker.
9. Synthetic authorization is confined to synthetic fixtures.
10. README and owner summaries no longer imply all open decisions block every run type.

## MBP-1

11. Same-timestamp evidence after the stage is excluded.
12. Timestamp-only stages conservatively exclude ambiguous same-timestamp events.
13. The plan clearly states whether/when `IFVG_ORDER_FLOW_MBP1_V1` becomes an executable research block.
14. MBP-10 remains structurally absent.

## Prop realization

15. 1m OHLC is represented as unordered evidence.
16. Assumed intrabar order is a scenario/approximation, not exact history.
17. Every rule has a minimum path-fidelity requirement.
18. The exact trade-path artifact enters simulation identity.
19. Synthetic and first-party firm-contract statuses are distinct.

## Regime and lineage

20. `CONTEXT_BAR_PANEL` exists in the actual schema with interval/source/as-of fields.
21. Planned spectral/Nyström algorithms are fail-closed in V1, not described as implemented.
22. Lineage keys are fully specified and one-to-one; collisions disable comparison.
23. Proposed scientific values remain unratified until evidence refs exist.

## Delivery and consistency

24. Release gates and dependencies reflect all corrections.
25. The test matrix contains all required new tests.
26. `FINAL_CONSISTENCY_AUDIT_V2.md` finds no remaining authority-document contradiction.
27. No implementation begins from a superseded plan version.

---

# 10. Required Final Response From Agent 2

Return:

1. paths to every revised document;
2. a concise P0-A through P0-H completion table;
3. a concise P1-A through P1-G completion table;
4. the updated owner-blocker matrix by capability/run type;
5. the chosen MBP-1 release/ownership design;
6. the chosen path-fidelity terminology and per-rule policy;
7. the path to `FINAL_CONSISTENCY_AUDIT_V2.md`;
8. confirmation that all superseded phrases were checked;
9. confirmation that no code, tests, artifacts, catalogs, seeds, replays, feature builds, model fits, simulations, or data runs occurred.

Stop after the revised planning documents.
