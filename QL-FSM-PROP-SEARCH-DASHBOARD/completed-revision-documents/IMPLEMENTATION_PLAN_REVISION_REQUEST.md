# Agent 2 Revision Request — Quant-Lab FSM / Prop Search Plan Documents

**Document type:** Mandatory planning-document revision brief  
**Target:** Agent 2 / Claude Fable plan package  
**Status:** Documentation-only amendment request  
**Implementation authorization:** None  
**Source:** Independent line-by-line review of the Agent 2 planning package  
**Objective:** Revise the existing implementation-plan documents so they are internally consistent, technically implementable, and safe to use as the basis for a later coding task.

---

# 1. Instructions to Agent 2

Revise the existing plan package in place. Do **not** merely append an errata page while leaving contradicted contracts elsewhere.

Update every affected statement, schema, test, milestone, owner decision, file map, and README summary so that the final package expresses one coherent design.

## 1.1 Documentation-only boundary

Do not:

- modify production code;
- modify tests;
- generate or regenerate data artifacts;
- create seed snapshots;
- launch a replay, feature build, model fit, prop simulation, or search;
- modify catalogs;
- access June 11, 2026 or any sealed source;
- modify Strategy-Core, Quant-Lab, or Trade-Lab code;
- treat the recommended scientific defaults in the current documents as owner-approved.

You may inspect the repositories and the existing plan files as needed to make the revised plan codebase-specific.

## 1.2 Documents that must be revised

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
```

Update `ARCHITECTURE_MAP.md` only where the revised architecture or newly identified gaps require it.

`FSM-PLAN-DOCUMENT.md` is planning provenance, not implementation authority. It may be updated with a pointer to the final revisions, but it must not override the revised implementation documents.

## 1.3 Required revision outputs

Produce:

1. revised versions of all affected planning documents;
2. `REVISION_CHANGELOG.md`;
3. `REVISED_OPEN_DECISIONS.md` if the revised `OWNER_DECISIONS.md` does not already provide a complete change log;
4. a final cross-document consistency check;
5. an explicit confirmation that no code, tests, artifacts, catalogs, or runtime data were changed.

Stop after revising the documents.

---

# 2. Mandatory P0 Corrections

The following changes are required before the plan is suitable for implementation.

---

## P0-1 — Separate reusable strategy replay identity from parent-study membership

The current `ChildStrategyIdentity` is over-keyed by fields that do not define the underlying strategy replay, including:

```text
parent_search_id
search-derived profile_name
cost_policy_sha256
audit_schema_identity
```

Replace the current design with at least these separate identities/contracts:

```text
CoreStrategyReplayIdentity
SearchChildMembership
CoreReplayArtifactReference

FsmAuditArtifactIdentity
ReplayChartArtifactIdentity

CostedEvaluationIdentity
FeatureViewIdentity
LabelViewIdentity
FoldSetIdentity
ModelFitIdentity
PredictionSetIdentity
RegimeFitIdentity

PropSimulationIdentity

PipelineSemanticIdentity
ExecutionAttemptIdentity
ArtifactContentIdentity
```

### Core strategy replay identity must include

```text
authorized source/date-set identity
bar/day-artifact identity
Strategy-Core commit/source identity
resolved section/profile hash
stable semantic profile identifier
warmup/seed identity
resolver/execution semantics
strategy/capture schema versions
```

### Core strategy replay identity must not include

```text
parent search ID
child ordinal
human display name
cost policy
risk policy
prop contract
payout policy
portfolio policy
audit schema
replay-chart schema
worker count
memory budget
retry number
job root
```

### Search membership

`SearchChildMembership` links:

```text
parent research/search ID
child ordinal
declared axis values
CoreStrategyReplayIdentity
display name
comparison role
```

Changing the parent study must not force a replay when the semantic strategy replay identity is unchanged.

### Required tests

Add tests proving:

1. the same resolved strategy configuration reuses one core replay across two different parent studies;
2. changing cost, risk, prop, payout, or worker policy does not change `CoreStrategyReplayIdentity`;
3. changing the section/profile, date set, seed, source, resolver, or Strategy-Core version does change it;
4. parent membership and child ordinal change only `SearchChildMembership`;
5. a changed audit or replay-chart schema creates new companion-artifact identities without changing the core replay identity.

---

## P0-2 — Use stable semantic profile naming

Do not persist a semantic profile name based on:

```text
<baseline>__search_<search-id>__c<ordinal>
```

That is parent-specific presentation metadata.

Define a stable canonical strategy-profile identifier derived from the resolved strategy semantics, for example:

```text
ifvg_search_profile_<resolved_section_hash_prefix>
```

or another repository-consistent profile identifier whose complete identity is the full section/profile hash.

The UI may show a parent-specific human label such as:

```text
60-Bar Parent Timeout
```

but that display label belongs only to mutable catalog/presentation metadata.

The revised plan must explain whether `profile_name` participates in existing Strategy-Core record IDs. If it does, the plan must guarantee the canonical semantic name is stable across parent studies.

---

## P0-3 — Separate semantic pipeline identity from operational execution attempts

Worker count, memory limits, process-pool settings, retry count, and other resource values are operational.

Define:

```text
PipelineSemanticIdentity
ExecutionAttemptIdentity
```

`PipelineSemanticIdentity` contains the complete research-bearing specification.

`ExecutionAttemptIdentity` contains:

```text
pipeline semantic ID
worker/resource policy
attempt number
host/environment metadata
start/end timestamps
operational retry reason
```

Changing workers or memory after an operational failure must not create a new scientific result identity.

Add a test proving two attempts with different worker counts produce the same semantic stage/result identities when outputs are byte-identical.

---

## P0-4 — Add immutable owner-authorization references to the charter

The current owner-decision document contains blocking decisions, but the search charter does not carry exact immutable approval evidence.

Add a contract such as:

```text
OwnerDecisionEvidenceRef
OwnerAuthorizationBundle
```

Every real research charter must reference the exact approved artifacts for:

```text
search axes and values
locked invariants
parent-fill policy
firm/account contracts
strategy gates
prop gates
verification allowlist
regime grain and cluster policy
validation protocol
```

Required fields:

```text
decision_id
decision_artifact_id
content_hash
author
approved_at
effective_from
reviewed_evidence_refs
```

A real launch must fail closed when any required approval reference is absent, stale, superseded, or inconsistent with the charter.

Synthetic and verification fixtures may use explicitly typed synthetic authorization artifacts.

---

## P0-5 — Replace scalar axis values with a typed axis-value contract

The current scalar union cannot accurately represent all planned values.

The registry must support:

```text
None/null timeout values
tuples of timeframes
sets/lists with canonical ordering
structured policy variants
dependent field groups
composite named policies
```

Define something equivalent to:

```text
DimensionValueSpec
RegisteredAxisValue
CompositeAxisValue
```

Each value needs:

```text
stable value ID
typed payload/schema
human label
capability status
owner-ratification status
dependencies
incompatibilities
expected replay effect
```

Owner ratification must be value-specific, not only one Boolean for the entire axis.

Do not expose arbitrary `section_overrides` in the UI.

---

## P0-6 — Correct the five-day verification design

The binding rule remains:

```text
one frozen allowlist
at most five authorized real trading days total
warmup + evidence <= 5
no full-development implementation gate
```

The current plan incorrectly expects four real child profiles with changed section hashes to start from one doc-default seed snapshot.

A Strategy-Core seed is profile-bound. One baseline seed must not be reused across semantically different child profiles.

### Revised verification split

Use two complementary verification paths.

#### A. Real five-day vertical slice

Run:

```text
one exact baseline profile
one exact matching seed snapshot
one frozen <=5-day allowlist
one complete sequential replay
audit/chart/store/reload/UI control flow
```

This proves:

```text
source access
mid-chain start
seed compatibility
sequential replay
artifact persistence
exact verifier linkage
five-day access safety
```

#### B. Synthetic multi-child study

Use deterministic synthetic fixtures for:

```text
2×2 child enumeration
parent/child orchestration
replay-result reuse
strategy-gate pass/fail
16 prop simulations
frontier
insights
cancel/resume/lock behavior
```

Do not claim that the synthetic child result proves real strategy behavior.

### Optional alternative

Real multi-child verification is allowed only if each child has its own independently verified, profile-matching immutable seed snapshot. The plan must not create those snapshots by secretly replaying the full development history during implementation verification.

### Gate policy

Normal research gates such as:

```text
minimum 30 trades
minimum 20 independent days
```

cannot be expected to pass on a five-day fixture.

Define a separate nonresearch verification-gate policy or use synthetic child results for downstream prop/frontier control-flow tests.

Every verification result must remain stamped:

```text
verification_only=true
not_for_research_interpretation=true
full_pipeline_not_run=true
```

---

## P0-7 — Select the real verification fixture by coverage, not recency alone

Do not retain the proposed June 4–10 fixture merely because it is the last five trading days.

The owner-approved fixture must satisfy a documented coverage matrix using already-authorized evidence:

```text
source partition available
setup activation
parent candidate/lock where available
opposing/inversion where available
entry candidate
execution/resolution where available
audit events
replay-chart coverage
MBP-1 source coverage when the MBP-1 lane is being verified
```

If no one five-day real fixture can cover all paths, use:

```text
one fixed real fixture for source/replay integration
synthetic fixtures for the missing behavioral branches
```

The entire release suite must still use the same real allowlist and must not rotate through multiple windows.

Update `OWNER_DECISIONS.md` so the exact dates are an owner-approved, coverage-evidenced decision rather than an engineering default.

---

## P0-8 — Replace child audit “parity exemption” with same-child audit neutrality proof

Do not rely only on:

```text
parity_exempt_non_baseline=true
```

A child cannot be compared with the accepted doc-default baseline artifact, but audit behavior neutrality must still be proven.

For every child strategy replay, require one of:

### Preferred

```text
core tables from audit-disabled replay
==
core tables from audit-enabled replay
```

### Acceptable single-drive equivalent

A formally demonstrated one-drive mechanism in which:

```text
audit capture is a side channel
core stream identity is independently verified
audit capture cannot alter reducer state or ordering
```

Persist a `ChildAuditNeutralityReport`.

The doc-default accepted baseline parity gate remains intact. Child-specific neutrality is an additional gate, not a replacement.

Update:

```text
IMPLEMENTATION_PLAN.md
CONTRACTS_AND_SCHEMAS.md
TEST_MATRIX.md
PHASED_DELIVERY.md
README.md
```

accordingly.

---

## P0-9 — Add profile-independent opportunity lineage for cross-profile deltas

A same-profile repeat test proves deterministic IDs. It does **not** prove that the same underlying market opportunity retains the same native setup/candidate/trade ID after profile changes.

Define a profile-independent exact lineage layer, derived only from stable source evidence, such as:

```text
HTF opportunity key
tap/activation key
parent opportunity key
opposing opportunity key
inversion opportunity key
entry-trigger opportunity key
trade-opportunity key
```

The precise keys must be grounded in the current Strategy-Core records and exact source cursors.

Every cross-profile delta must state one of:

```text
native_id_exact
profile_independent_lineage_exact
unmatched
not_comparable
```

No nearest-time, row-order, fuzzy-geometry, or keep-last matching is allowed.

If a profile-independent exact key cannot be constructed for a population type, the plan must disable that comparison rather than infer commonality.

Add contracts and tests for:

```text
OpportunityLineage
NativeLineageMap
LineageMatchResult
PopulationDeltaMatchBasis
```

---

## P0-10 — Add a real `CohortSpec`

The plan references cohort IDs and a session/direction interpretation selector but does not define the public cohort contract.

Add:

```text
CohortSpec
CohortIdentity
CohortResult
InterpretationMode
```

Required interpretation modes:

```text
descriptive_slice
specialized_model
sequential_strategy_profile
```

Required cohort dimensions should include, where supported:

```text
date policy
warmup policy
row kind
session
direction
entry family
HTF timeframe
parent timeframe
regime
evidence-quality status
executed/counterfactual scope
```

The UI must explicitly ask whether `Asia only`, `long only`, or similar input means:

```text
analysis slice
specialized model
new sequential strategy
```

Only the third creates an executable counterfactual.

---

## P0-11 — Define complete model-gated FSM semantics before claiming gated replay support

The plan says a model decision policy requires a full sequential replay but does not define what happens when a candidate is rejected.

Add:

```text
RejectedCandidatePolicy
ModelDecisionPolicySpec
ModelGatedReplayRequest
WalkForwardModelSchedule
```

Potential semantics may include:

```text
reject candidate and keep setup waiting
reject candidate and terminate setup as missed
reject candidate and consume one-shot trigger
reject candidate and reset setup
```

Do not authorize one implicitly.

The chosen behavior can change future candidate availability and slot occupancy, so it is a strategy semantic and requires owner approval.

Pipeline stage S11 may remain blocked in V1. The plan must say:

```text
model-gated execution is unavailable until RejectedCandidatePolicy is owner-ratified and sequential golden tests pass
```

A threshold report alone never establishes execution or prop value.

---

## P0-12 — Add exact intratrade path-fidelity contracts for prop simulation

The current `PropTradeRecord` design based on:

```text
realized result
MFE magnitude
MAE magnitude
adverse-first assumption
```

is insufficient for an exact historical simulation of many trailing-drawdown and daily-loss rules.

Add:

```text
TradePathArtifactSpec
TradePathEvent
TradePathFidelity
PropRulePathRequirement
PathCapabilityReport
```

Suggested fidelity classes:

```text
closed_trade_only
ordered_1m_bar_path
ordered_tick_or_mbp1_event_path
ordered_fill_event_path
```

The plan must define:

```text
event timestamps
event ordinals
same-timestamp ordering
price/equity update ordering
fee ordering
threshold-ratchet ordering
trade/fill linkage
account linkage
```

Each prop rule declares its minimum required fidelity.

A simulation must fail closed when the available path cannot determine the required chronology.

Do not call an adverse-first approximation “exact historical.” Label it explicitly as a conservative scenario/approximation.

---

## P0-13 — Add a totally ordered account-event envelope

Every prop account event must carry, as applicable:

```text
event_id
event_ts_utc
event_ordinal
path_instance_id
account_id
account_ordinal
firm_contract_id
account_phase
source_trade_id
source_decision_id
source_candidate_id
source_setup_id
source_path_event_id
event_type
event_order_policy_id
```

Use a deterministic total-order key.

`day` alone is not sufficient for DLL, trailing-floor, fee, payout, breach, or replacement chronology.

---

## P0-14 — Type all calendar/day-count semantics

Replace naked integer “days” with typed duration/count contracts.

Add:

```text
DayCountBasis
DurationRule
FirmCalendarPolicy
SimulatedClockPolicy
```

Supported bases should include, where needed:

```text
trading_day
winning_day
business_day
calendar_day
calendar_month
firm_defined_payout_period
```

Apply them explicitly to:

```text
minimum trading days
winning-day requirements
payout waiting period
days between payouts
payout processing
recurring fees
account expiration
replacement delay
rolling payout windows
```

The bootstrap must define how resampled day blocks advance each type of clock.

A rule that cannot be represented truthfully under the selected bootstrap clock must be marked unsupported and fail closed.

---

## P0-15 — Separate payout/withdrawal behavior from the firm contract

Do not hardcode:

```text
request_at_first_eligibility_max
```

as the only account behavior.

Define a separate immutable policy:

```text
WithdrawalPolicySpec
PayoutBehaviorSpec
```

It must support future controlled comparisons such as:

```text
request immediately at eligibility
retain minimum buffer
fixed payout cadence
partial withdrawal
maximum allowed withdrawal
post-payout de-risking
```

The firm contract defines what is permitted.

The withdrawal policy defines what the simulated trader chooses.

Both IDs must enter `PropSimulationIdentity`.

---

## P0-16 — Complete `PropSimulationIdentity`

The simulation identity must include every result-changing policy, including:

```text
trade-stream hash
trade-path artifact/path-fidelity policy
firm contract
risk policy
withdrawal/payout behavior
replacement policy
maximum replacements
portfolio policy
simulation mode
bootstrap/stress protocol
clock/day-count policy
seed
number of paths
cost policy where applicable
```

No constructor-only argument may alter results without entering identity.

---

## P0-17 — Add a field-level prop-contract evidence compiler

Do not defer prop contract verification to a manual status flag alone.

Plan an offline, deterministic source workflow:

```text
PropContractSourceDocument
PropRuleEvidence
PropContractDraft
PropContractCompilationReport
PropContractReviewDecision
PropContractSupersession
```

Every normalized contract field must retain:

```text
source document ID/hash
official provenance/URL
retrieved timestamp
effective interval
page/section locator
normalized value
reviewer
conflict status
supersession status
```

Runtime jobs consume only the frozen compiled contract artifact. They must not scrape live pages.

For publishable real results:

```text
stale or unverified contract override is prohibited
```

Synthetic test contracts remain allowed in the test namespace.

Update owner decision 6 from “deferred/manual” to a required pre-real-run capability.

---

## P0-18 — Make the mutable catalog concurrency-safe

A single mutable JSON file updated by multiple publishers can lose writes.

Use one of:

```text
transactional SQLite metadata/annotation index
lock-guarded append-only catalog with deterministic rebuild
```

The immutable artifacts remain authoritative.

The mutable index must be rebuildable from manifests and must never hold research metrics as the sole source of truth.

Add concurrent-publisher and crash-recovery tests.

---

## P0-19 — Correct MBP-1 total ordering and stage cutoffs

Use a complete deterministic ordering key such as:

```text
(ts_event, ts_recv, sequence, source_ordinal)
```

Do not use only event time plus sequence.

The stage cutoff must also be represented as a compatible total-order key. Events are included only when:

```text
source_event_order_key < stage_cutoff_order_key
```

or according to another explicitly versioned inclusive/exclusive rule.

Add tests for multiple events sharing the same event timestamp.

MBP-1 remains the maximum supported order-book depth. No MBP-10 or deeper-book support may be introduced.

---

## P0-20 — Remove parent-fill Boolean from the recommended first real search

Do not recommend:

```text
parent_full_fill_invalidation ∈ {true, false}
```

as an initial real search axis.

The existing Boolean does not distinguish:

```text
pre-lock/provisional fill
locked-parent fill
S2/S3/S4 fill
wick-only traversal
body close through
small overshoot
post-inversion thesis destruction
fallback/replacement behavior
```

Revise the recommendation to:

```text
parent-fill behavior locked to the current accepted semantics
status = blocked_pending_owner_policy_review
```

A richer policy requires exact owner semantics and possibly a Strategy-Core contract change.

For the first real search, recommend no replacement without owner approval. A minimal single-axis staleness comparison may be proposed after setup-level review.

For synthetic orchestration acceptance, use generic synthetic axes or a timeout-only real axis plus a synthetic second dimension. Do not let a synthetic 2×2 fixture imply owner authorization of parent-fill changes.

---

## P0-21 — Correct bootstrap duplicate-path semantics

Random day-block bootstrap can legitimately sample identical index sequences more than once.

Do not reject valid duplicate sampled paths.

Instead require:

```text
unique path_instance_id per simulation draw
stored sampled index sequence/hash
deterministic seed reproducibility
common path sequence shared across copied accounts
no accidental duplicate OOS prediction rows
```

Revise the current “No duplicate OOS path” test accordingly.

---

# 3. Required P1 Scope and Research-Quality Revisions

These changes are important for feasibility and scientific clarity.

---

## P1-1 — Narrow the initial ML/regime implementation scope

Preserve the complete schema and capability registry, but reduce the first implementation burden.

Recommended initial active scope:

```text
training-prevalence reference
regularized logistic regression
fixed CatBoost
KMeans
regime contracts/status/coverage/UI
```

Recommended later milestones:

```text
Gaussian mixture
direct spectral training-only diagnostics
Nyström+KMeans
expanded drift monitoring
```

Direct spectral and Nyström concepts must remain explicit in the plan because they are part of the intended architecture, but they need not all be production code in the first released workspace.

If Agent 2 retains GMM or spectral code in V1, justify the incremental value, dependency cost, and test burden relative to the core search/prop workflow.

---

## P1-2 — Support a larger completed-market-context regime grain

Candidate-stage rows may be too sparse for stable clustering.

The architecture should support a point-in-time completed context panel such as:

```text
completed 5m context bars
completed 15m context bars
another owner-approved context interval
```

The fold-fitted regime model may train on that larger panel and assign the frozen current regime to candidate stages.

Candidate-stage and decision-row grains remain supported.

The plan must include sample-adequacy gates and must not treat `k=3` as useful merely because fitting succeeds.

---

## P1-3 — Treat regime defaults as proposals, not owner authority

The following remain proposed protocol defaults:

```text
KMeans/GMM choice
k = 3
minimum 25 rows per cluster
5% occupancy
aligned AMI threshold
spectral implementation timing
```

Mark them as:

```text
proposed_protocol_default
owner_ratification_required_before_feature_eligible
```

No cluster family becomes a predictive feature or execution gate merely because the engineering implementation exists.

---

## P1-4 — Make persisted fitted-pipeline references portable

A fitted preprocessing/model artifact must use a manifest-relative, checksummed reference, not a local absolute filesystem path.

Persist:

```text
artifact-relative path
file hash
byte size
schema/version
software versions
portable parameter payload hash
```

Reload verification must not depend on the original machine path.

---

## P1-5 — Planned feature-block activation creates a new versioned identity

Do not describe activation as only a status flip.

When a planned feature block becomes real, it gains:

```text
source schema hash
formula version
feature schema
materializer version
source identities
coverage contract
```

That requires a new `FeatureBlockSpec` version and registry hash.

The surrounding platform should not require redesign, but the feature identity must change.

---

## P1-6 — Exploratory versus publishable representative

If nested outer evaluation remains deferred, the UI and reports must say:

```text
development exploratory representative
```

Do not present it as a publishable robust representative.

A publishable selected configuration requires a separately frozen outer evaluation protocol, such as nested walk-forward or another owner-approved unseen-data protocol.

Update:

```text
status vocabulary
overview cards
frontier reports
owner decision 15
publication gates
insight wording
```

---

## P1-7 — Add an earlier real vertical slice

Do not defer the first real baseline replay/store/reload integration until the final hardening milestone.

Milestone 2 should include:

```text
one baseline profile
one exact matching seed
one fixed <=5-day real allowlist
core replay
same-child audit-neutrality proof
immutable save/reload/reuse
exact verifier link
zero forbidden access
```

This should gate later prop, UI, and ML milestones.

---

## P1-8 — Resolve path fidelity before implementing the prop engine

The prop milestone must begin with:

```text
TradePathArtifactSpec
PathCapabilityReport
event-order contract
calendar/day-count contract
firm-rule minimum-fidelity matrix
```

Only after these contracts pass should the account state machine be implemented.

---

## P1-9 — Break the release into smaller usable increments

Revise the delivery plan toward:

```text
Release 1:
    contracts, identity, access, immutable stores, baseline real vertical slice

Release 2:
    multi-child FSM search, lineage, exact deltas, verifier integration

Release 3:
    one verified prop contract, exact lifecycle, risk and payout policies

Release 4:
    wizard, active runs, results and account timeline

Release 5:
    MBP-1 feature platform and supervised model ladder

Release 6:
    regime expansion, advanced robustness, optional spectral diagnostics
```

The exact milestone names may follow repository conventions, but implementation should not require the entire research operating system to land before the user receives a usable, verified slice.

---

# 4. Per-Document Revision Requirements

---

## 4.1 `IMPLEMENTATION_PLAN.md`

Revise:

- executive recommendation;
- recommended first study;
- backend module map;
- identity model;
- job/orchestration design;
- prop simulation design;
- search/robustness protocol;
- full-pipeline identity;
- risks;
- expected files;
- milestones;
- confirmations.

Add explicit summaries of:

```text
core replay identity vs membership
profile-independent lineage
five-day real/synthetic split
same-child audit neutrality
path fidelity and rule capability
calendar/day-count semantics
withdrawal policy
field-level prop evidence
CohortSpec
RejectedCandidatePolicy
catalog concurrency
exploratory vs publishable representative
```

---

## 4.2 `CONTRACTS_AND_SCHEMAS.md`

Replace or add field-level contracts for:

```text
DimensionValueSpec
OwnerDecisionEvidenceRef
OwnerAuthorizationBundle

CoreStrategyReplayIdentity
SearchChildMembership
ExecutionAttemptIdentity
PipelineSemanticIdentity

ChildAuditNeutralityReport

OpportunityLineage
NativeLineageMap
LineageMatchResult

CohortSpec
CohortIdentity
CohortResult
InterpretationMode

RejectedCandidatePolicy
ModelDecisionPolicySpec
WalkForwardModelSchedule

TradePathArtifactSpec
TradePathEvent
TradePathFidelity
PropRulePathRequirement
PathCapabilityReport

PropAccountEventEnvelope
DayCountBasis
DurationRule
FirmCalendarPolicy
SimulatedClockPolicy

WithdrawalPolicySpec
PayoutBehaviorSpec

PropContractSourceDocument
PropRuleEvidence
PropContractCompilationReport

Catalog transaction/index contract
```

Correct every affected identity payload.

---

## 4.3 `DELTA_TAXONOMY.md`

Revise:

- study-cell identity fields where necessary;
- cross-profile population matching;
- comparison compatibility rules;
- cohort contract;
- computation paths for model-gated execution;
- execution/prop path-fidelity requirements;
- MBP-1 ordering;
- planned-block versioning;
- exploratory versus publishable statuses.

Add a delta field identifying:

```text
match basis = native exact | lineage exact | unavailable
```

---

## 4.4 `ML_REGIME_CONTRACT_PLAN.md`

Revise:

- initial implemented versus planned algorithm scope;
- regime observation grains;
- sample-adequacy requirements;
- proposed versus owner-ratified defaults;
- portable fitted-pipeline references;
- model-gated candidate rejection semantics;
- S11 blocked-state reason;
- publication wording.

Keep direct spectral restrictions and Nyström design explicit even if implementation is deferred.

---

## 4.5 `TEST_MATRIX.md`

Replace the current real four-child E2E with:

```text
one-profile real five-day vertical slice
synthetic 2×2 multi-child E2E
synthetic strategy/prop/frontier pass/fail paths
```

Add tests for:

```text
core replay reuse across parent studies
stable semantic profile naming
cost/audit/resource independence of core replay ID
execution-attempt identity
profile-seed mismatch refusal
profile-independent lineage
same-child audit on/off parity
CohortSpec interpretations
RejectedCandidatePolicy
ordered intratrade path capability
ambiguous event/bar ordering
DayCountBasis
withdrawal-policy identity
replacement-policy identity
prop source compilation/supersession
concurrent catalog publishing
MBP-1 same-timestamp ordering
planned-block version bump
exploratory representative wording
```

Correct the bootstrap duplicate-path test.

Keep the one fixed <=5-day real-data budget.

---

## 4.6 `OWNER_DECISIONS.md`

Reclassify:

- parent-fill axis: blocked pending precise policy;
- first real search: no implicit parent-fill Boolean;
- verification dates: coverage-based owner decision;
- contract-source ingestion: required before real prop use;
- nested walk-forward: exploratory lane may defer, publishable representative may not;
- regime scientific values: proposals until owner ratification;
- spectral/GMM implementation timing: engineering recommendation, not scientific promotion;
- all existing “ENGINEERING” scientific defaults: distinguish implementation availability from research authorization.

Do not say “nothing blocks implementation” until the P0 architecture revisions in this brief are incorporated.

---

## 4.7 `PHASED_DELIVERY.md`

Revise milestone order to include:

1. contract/identity corrections;
2. early real baseline vertical slice;
3. child search and lineage;
4. path-fidelity contracts;
5. one verified prop lifecycle;
6. UI;
7. MBP-1/model lane;
8. regime expansion;
9. hardening;
10. operator full run.

Update every file list and gate.

---

## 4.8 `README.md`

Add:

```text
revision status
documents superseded
P0 corrections incorporated
owner-unratified decisions
implementation authorization status
```

Do not imply the package is ready for implementation until the revision acceptance checklist below is satisfied.

---

# 5. Revised Acceptance Criteria

The revised planning package is acceptable only when all of the following are true.

## Identity and reuse

1. The same strategy replay can be reused across parent studies.
2. Cost, risk, prop, audit, chart, and resource changes do not invalidate core strategy replay identity.
3. Parent membership, child ordinal, and display names are separate.
4. Operational retries have separate attempt identities.
5. Every real charter carries exact owner-authorization evidence.

## Five-day verification

6. One real baseline replay uses one exact matching seed.
7. Multi-child behavior is verified synthetically unless child-specific seeds exist.
8. The real allowlist is <=5 days and coverage-evidenced.
9. The real verification path does not depend on research gates impossible within five days.
10. No full development run is an implementation gate.

## Sequential and audit correctness

11. Every strategy child receives a full sequential replay.
12. Audit capture is behavior-neutral for every child.
13. Cross-profile population deltas use exact lineage or are disabled.
14. Model-gated execution remains blocked until rejection semantics are approved.

## Prop fidelity

15. Every prop rule declares minimum path fidelity.
16. Unsupported chronology fails closed.
17. Account events have total ordering and exact lineage.
18. Time rules declare their day-count basis.
19. Withdrawal behavior is separate from firm rules.
20. Every result-changing prop policy enters simulation identity.
21. Real contracts compile from field-level first-party evidence.

## Delta and feature architecture

22. `CohortSpec` is explicit.
23. MBP-1 uses the complete ordering key.
24. Planned feature activation creates a new versioned identity.
25. Descriptive cohorts are not called executable counterfactuals.

## ML/regime

26. The first implementation scope is realistically bounded.
27. Candidate-stage sample limitations are explicit.
28. Regime defaults are proposals until ratified.
29. Direct spectral cannot enter OOS feature bundles.
30. Fitted artifacts are portable and checksummed.

## UI and publication

31. Exploratory results are labelled exploratory.
32. A publishable representative requires an approved outer evaluation.
33. The catalog is concurrency-safe and rebuildable.
34. Every decisive delta drills into exact evidence.
35. No implementation begins from superseded plan text.

---

# 6. Required `REVISION_CHANGELOG.md` Format

For every changed document, include:

```text
Document
Section
Previous design
Revised design
Reason
Downstream documents updated
Remaining owner decision
```

Also include a contradiction audit showing that these topics are expressed consistently across all documents:

```text
identity decomposition
verification seed policy
first search recommendation
audit parity
lineage matching
prop path fidelity
calendar semantics
withdrawal policy
contract evidence
cohort semantics
model-gate semantics
ML/regime scope
milestone order
publication status
```

---

# 7. Final Response Required From Agent 2

Return:

1. paths to every revised document;
2. a concise summary of the P0 changes;
3. the list of still-open owner decisions;
4. confirmation that all cross-document contradictions were checked;
5. confirmation that no code, tests, artifacts, catalogs, replay, model fit, or data run occurred.

Stop after the revised planning documents.
