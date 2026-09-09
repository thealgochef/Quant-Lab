# Quant-Lab Robust FSM Configuration Search & Prop-Firm Realization Workspace

**Document type:** Product and research planning brief  
**Revision:** 4 — adds an explicit supervised-ML, unsupervised-regime, spectral-clustering, model-governance, and ML-dashboard contract  
**Status:** Plan-only; no implementation authorized by this document  
**Primary repository:** Claude-Quant-Lab  
**Supporting repository:** Strategy-Core, as the deterministic sequential-replay authority  
**Out of scope:** Trade-Lab live serving, orders, runtime activation, and production deployment  
**Audience:** Codex Sol and Claude Fable, who must inspect the current codebase and produce a codebase-specific implementation plan

**Output:**  The implementation plan should be saved in this folder C:\Users\gonza\Documents\Claude-Quant-Lab\QL-FSM-PROP-SEARCH-DASHBOARD and any supporting documents related to this implementation should also be saved here aswell. 
---

## 1. Instructions to the Planning Agent

This document defines the product intent, research constraints, backend requirements, frontend requirements, and acceptance criteria for a new Quant-Lab research workspace.

Your task is to:

1. Inspect the current Strategy-Core and Claude-Quant-Lab codebases.
2. Identify existing contracts, services, artifacts, UI components, job runners, and chart primitives that can be reused.
3. Produce a detailed, file-specific implementation plan.
4. Explain all required schema, service, artifact, job, reporting, and UI changes.
5. Define a phased delivery sequence, test matrix, migration strategy, performance budgets, and release gates.
6. Separate resolved engineering decisions from genuine owner decisions.
7. Stop after the implementation plan.

Do **not**:

- modify production code;
- modify tests;
- create or regenerate strategy artifacts;
- run a configuration search;
- train or refit a model;
- modify current catalogs;
- access June 11, 2026 or any sealed data;
- modify Trade-Lab;
- silently reinterpret existing M0–M3 experiments as an adaptive-search system.

The implementation plan must be grounded in the actual codebase. Where this brief proposes conceptual names, use the repository’s canonical naming and module boundaries when a better existing abstraction already exists.


## 1.1 Binding Owner Corrections

The following constraints supersede any older design note, prototype, or source document that
suggests a broader order-book depth or a full-data implementation-verification run.

### A. Order-flow scope is MBP-1 only

The user’s Databento subscription supports **MBP-1 for live streaming**. Therefore:

```text
maximum supported live order-book depth = MBP-1
```

The planning and implementation scope must include:

```text
IFVG_ORDER_FLOW_MBP1_V1
```

and must exclude:

```text
MBP-10
multi-level depth
deeper-book incremental feature blocks
controls or contracts that imply live access beyond MBP-1
```

Any older project text that mentions MBP-10 is superseded for this workspace. A deeper-book
feature family may be reconsidered only after the user obtains an appropriate source entitlement
and approves a new, separately versioned contract. The current schema must not reserve misleading
UI controls that imply MBP-10 is available.

MBP-1 may still support derived top-of-book and event-flow measurements, including:

```text
best bid/ask price and size
best-level order count
spread
microprice
queue imbalance
order-count imbalance
event-based OFI
aggressive trade imbalance
best-level add/cancel/depletion/replenishment
quote and trade intensity
absorption and persistence proxies
entry-exhaustion measures
```

All such features must be point-in-time, stage-aligned, source-versioned, and coverage-audited.

### B. Implementation verification is capped at five real trading days

No implementation, integration, benchmark, UI, or release-verification run may execute the full
research pipeline over the full development dataset.

All real-data verification for one implementation/release must use one fixed explicit allowlist
containing **at most five authorized trading days total**:

```text
maximum real verification days = 5
warmup + evidence combined      <= 5 days
```

Requirements:

- The same frozen five-day-or-smaller allowlist is reused across the verification suite.
- Tests may use synthetic fixtures in addition to the real mini-fixture.
- Tests must not rotate through different five-day windows and thereby traverse the full dataset
  cumulatively.
- If a component requires more historical state than the five-day fixture provides, verification
  must use a precomputed verified seed/snapshot or a synthetic warmup fixture; it must not read
  extra real dates.
- Any verification-only model fit must be a functional micro-run, not a research result.
- Verification-only outputs must use a test namespace and must not enter normal research catalogs.
- Protected and sealed access rules remain unchanged and all access counters must remain zero.
- Full-data replay, feature materialization, fold construction, model fitting, prop simulation,
  or search execution is not an implementation acceptance gate.

### C. Full retraining is a post-implementation operator workflow

After implementation and five-day verification are complete, Quant-Lab must provide a
streamlined, standardized UI workflow through which the user can launch the full authorized
pipeline.

The UI-driven full pipeline must be able to orchestrate, as applicable:

```text
source/artifact verification
full authorized sequential replay or child-profile preparation
FSM audit and replay-chart preparation
feature materialization
feature coverage validation
label derivation
fold construction
model training
calibration/diagnostics
optional frozen model-gated replay
prop-contract simulations
robustness simulations
immutable reporting and publication
```

The full run:

- is initiated explicitly by the user from the UI;
- is never triggered automatically by implementation tests, deployment, or catalog loading;
- previews the exact dates, phases, estimated runtime, worker count, and storage;
- is backgrounded, resumable, lock-guarded, checkpointed, and safely cancellable;
- exposes standardized progress and sanitized errors;
- verifies and reuses existing immutable stages when identities match;
- publishes only after all required gates pass;
- remains restricted to authorized development data unless a separately approved protocol changes
  the boundary.

The implementation plan must therefore distinguish:

```text
verification scope: fixed <=5-day mini-pipeline
operator scope: full authorized development pipeline launched through the UI
```

---

# 2. Executive Summary

Quant-Lab currently supports fixed IFVG research artifacts, deterministic M0–M3 feature-tier experiments, exact evidence inspection, reports, and setup/trade replay. It does not yet provide a user-friendly way to:

1. define a controlled FSM configuration search;
2. run many full sequential strategy replays as one parent research study;
3. explain how each FSM rule changes the setup and trade population;
4. apply versioned prop-firm contracts and risk policies to each resulting trade stream;
5. measure repeat-payout reliability, account survival, breach risk, fee drag, and payout droughts;
6. compare configurations through a Pareto frontier and parameter-stability analysis;
7. identify one robust representative configuration from a stable parameter region;
8. navigate from aggregate differences to the exact setup, trade, payout, or breach event on the chart.

The intended product is a new, separately authorized Quant-Lab research lane:

```text
One frozen parent search charter
    ├── many immutable full sequential FSM child replays
    ├── underlying strategy-quality gates
    ├── versioned prop-contract simulations
    ├── firm-specific risk policies
    ├── day-block and stress simulations
    ├── feasible configuration set
    ├── Pareto frontier
    └── robust representative configuration
```

A “single experiment” means one orchestrated, immutable parent study. It does **not** mean one replay can infer the best parameters without evaluating alternative configurations. Every strategy configuration must receive its own full sequential replay because FSM changes can alter setup occupancy, selected structures, candidate order, trade order, and all downstream account-state outcomes.

The dashboard must feel like a guided research workspace rather than a large settings form. A user should be able to:

> Define the research question, freeze the permitted search space, monitor child replays, see which configurations survive strategy and prop-firm benchmarks, understand why they differ, inspect parameter stability, and open every decisive setup or account event in the verifier.

---

# 3. Product Context

## 3.1 Existing Quant-Lab capabilities

The current IFVG Lab conceptually contains:

```text
Experiments
Replay / Verifier
Data & Audit
```

Existing work already provides or is intended to provide:

- immutable v2 candidate/decision/execution artifacts;
- exact-linked formula-v2 context artifacts;
- M0–M3 feature-tier experiments;
- candidate research reports separated from actual execution reports;
- exact candidate/setup/trade identities;
- setup-level replay-chart artifacts;
- deterministic FSM audit evidence;
- point-in-time chart verification;
- immutable experiment history;
- access-safety and artifact-verification reports.

The new search and prop-realization workspace must extend these capabilities rather than replacing them.

## 3.2 Existing research-contract boundary

The current M0–M3 context experiment is a fixed-tier model-ablation protocol. It prohibits adaptive feature, threshold, profile, and date search. The new capability must therefore be a **separate research lane** with:

- a separate contract;
- separate parent and child artifact identities;
- separate search authorization;
- separate reporting;
- no silent reuse of the fixed-tier experiment contract for adaptive profile search.

## 3.3 Current strategy capability boundary

The currently accepted runnable strategy profile remains the repaired doc-default fresh-FVG continuation profile.

The following remain separately blocked or experimental until explicitly ratified:

```text
pure IFVG retest execution
ICT-clean profile
canonical short-enabled profile
240m/Q-40 canonical interpretation
BE-managed label family
broader EQH/EQL tolerance
```

The search UI must surface these capability states and must not permit blocked axes to enter a study.

## 3.4 Current FSM research dependency

The recent FSM audit exposed material policy questions, including:

- a selected parent that waited without a retest for approximately 58 days;
- disabled `parent_retest_timeout`;
- post-inversion parent-fill deaths, including wick-only cases;
- the disabled-direction 4H-winner / bullish-1H-runner-up interaction;
- unresolved HTF-age semantics;
- unresolved 240m/Q-40 anchoring.

The search system can be planned and implemented with synthetic fixtures now. The first broad real-data search must remain blocked until the owner freezes the initial authorized strategy search axes and values after setup-level chart review.

---

# 4. Research Framework

The workspace must preserve five distinct analytical layers.

## 4.1 Edge creation

Question:

> Does the deterministic market mechanism plausibly produce favorable outcomes?

Relevant evidence:

- FSM progression;
- setup geometry;
- displacement and liquidity context;
- entry and barrier construction;
- planned risk-reward.

## 4.2 Edge detection

Question:

> Is positive expectancy supported rather than explained by sample noise, regime concentration, or repeated development-data selection?

Relevant evidence:

- cost-adjusted expectancy in R;
- realized payoff ratio;
- confidence intervals;
- time-block stability;
- setup/day clustering;
- lower-tail results;
- walk-forward consistency;
- minimum independent sample requirements.

## 4.3 Edge preservation

Question:

> Can the strategy survive the path long enough for the edge to manifest?

Relevant evidence:

- maximum drawdown;
- time under water;
- losing-streak distribution;
- setup-slot occupation;
- risk-of-ruin;
- behavioral and operational constraints;
- regime and opportunity-frequency shifts.

## 4.4 Edge extraction

Question:

> Is the return worth the variance, capital usage, opportunity cost, and execution drag?

Relevant evidence:

- realized payoff ratio versus planned RRR;
- commission and slippage drag;
- capital efficiency;
- volatility drag;
- risk-policy sensitivity;
- lower-tail payout.

## 4.5 Prop-firm realization

Question:

> Can the same strategy path satisfy a versioned prop-firm payout contract repeatedly?

Relevant evidence:

- challenge pass probability;
- first-payout probability;
- multiple-payout probability;
- funded-account survival;
- breach probability;
- payout drought;
- fee and replacement drag;
- net cash extracted;
- portfolio-level payout reliability across firms.

These layers must remain visible and separate. A configuration must not be called good merely because it passes an evaluation quickly.

---

# 5. Product Goals

The new workspace must allow a user to:

1. Create one immutable parent search containing a controlled set of FSM child configurations.
2. Lock correctness invariants and search only approved discretionary/profile axes.
3. Preview the exact number of strategy replays and downstream prop simulations before launch.
4. Execute a full sequential Strategy-Core replay for each unique strategy profile.
5. Reuse an existing child replay when its full identity already exists.
6. Apply one or more versioned prop-firm contracts to each qualifying strategy trade stream.
7. Define firm-specific risk policies without changing the underlying strategy profile.
8. Reject configurations that fail underlying strategy-quality gates before prop optimization.
9. Run exact historical, day-block bootstrap, and adverse stress simulations.
10. Show a feasible set and Pareto frontier rather than one opaque winner.
11. Find stable parameter regions and identify knife-edge results.
12. Select one robust representative configuration from the center of a stable region.
13. Explain funnel, strategy, prop, and robustness differences from the baseline.
14. Drill from every aggregate difference into exact setups, trades, payouts, and breaches.
15. Persist every search, child, contract, risk policy, simulation, and report immutably.
16. Keep development, protected, and sealed boundaries explicit and enforced.
17. Display human-readable insights backed by exact artifact evidence.
18. Preserve a complete audit trail of why each configuration passed or failed.

---

# 6. Non-Goals

The first implementation must not:

- place orders;
- activate a live model;
- modify Trade-Lab;
- promote a strategy automatically;
- use sealed data;
- infer strategy changes from post-hoc candidate-table filters;
- treat blocked candidates as executed trades;
- permit pyramiding or multiple active setups;
- search correctness invariants;
- search unratified strategy families;
- optimize model features and FSM rules in the same study;
- optimize the strategy target, model, threshold, risk policy, and FSM rules simultaneously;
- use one hidden weighted score as research truth;
- call development results external validation;
- treat copied prop accounts as independent market paths;
- silently change firm rules without a new contract version;
- silently select the most favorable threshold after results are visible;
- return only `best_config.json` without the feasible set, frontier, and failure reasons.
- include or imply MBP-10/deeper-book live features;
- run full-development retraining or a full-data pipeline as part of implementation verification;
- trigger a full pipeline automatically from tests, startup, deployment, or catalog refresh;

---

# 7. Product and Research Principles

## 7.1 Guided workflow over giant settings form

The user should define a research question through a wizard. Technical configuration remains inspectable but should not be the first interface.

## 7.2 Progressive disclosure

Use three information levels:

```text
Summary
Analyst
Audit
```

The user should understand the result without reading hashes, but every claim must expose its exact evidence.

## 7.3 Human names plus exact identities

Display:

```text
60-Bar Parent Timeout
```

and retain:

```text
profile hash
artifact ID
manifest hash
search ID
contract ID
risk-policy ID
simulation ID
```

with copy controls.

## 7.4 One scientific axis at a time

The system may orchestrate many child configurations, but the search charter must declare the authorized axes. The UI must clearly distinguish:

```text
strategy search axis
firm-specific risk-policy axis
model axis
label axis
analysis-only filter
measured-only field
locked invariant
```

## 7.5 Strategy quality before prop success

A strategy that fails the underlying edge gate must not be promoted because an aggressive risk policy passes a short evaluation.

## 7.6 Full sequential replay

Every strategy configuration must run through the actual one-active-setup, one-active-trade FSM.

No sequence-approximate shortcut may be used for:

- session changes;
- timeout changes;
- fill-policy changes;
- direction changes;
- HTF selection changes;
- daily execution caps;
- any rule that can alter slot occupancy.

## 7.7 Universal strategy, firm-specific realization

The preferred optimization target is:

```text
one universal strategy profile
+
one risk policy per firm contract
```

This keeps the market logic stable while respecting different contract economics.

## 7.8 Robust basin over point optimum

The system must identify configurations that remain competitive across neighboring values, time blocks, firms, and stress scenarios.

A result that works only at one exact parameter value should be flagged as knife-edge.

## 7.9 Deterministic insight generation

Primary insight text should be generated from deterministic templates and exact deltas. An optional AI summary may be added later, but it must not replace the reproducible insight layer.

---


# 7A. Universal Experiment-Dimension and Delta Measurement Contract

The examples in this brief—baseline versus order flow, session comparisons, long-only,
short-only, or different prop firms—are illustrative rather than exhaustive.

The implementation must not hard-code a small list of comparison types. Quant-Lab needs a
general comparison grammar that can represent any controlled research cell, identify exactly
which dimensions changed, determine the required computation path, and select the correct
delta measurements automatically.

## 7A.1 Canonical study-cell identity

Every research result must resolve to a canonical `StudyCellIdentity` composed from orthogonal
dimensions.

Conceptually:

```text
Data Lineage
× Market Universe
× Strategy Construction
× Observation Cohort
× Label / Outcome Policy
× Feature Bundle
× Model Protocol
× Decision / Gate Policy
× Execution / Management Policy
× Cost Model
× Risk / Sizing Policy
× Prop Contract
× Payout / Withdrawal Policy
× Portfolio Topology
× Validation Protocol
× Stress Scenario
```

The exact class and field names should follow existing repository conventions, but the
identity must be capable of distinguishing at least:

```text
data_source_id
data_formula_id
instrument_universe_id
contract_roll_policy_id
bar_construction_policy_id

strategy_profile_id
strategy_version
section_config_hash
entry_family
direction_policy
session_policy
concurrency_policy

cohort_id
warmup_policy_id
date_policy_id
label_policy_id

feature_bundle_id
feature_schema_hash
model_protocol_id
model_artifact_id
decision_policy_id

execution_policy_id
cost_policy_id
risk_policy_id

prop_contract_id
payout_policy_id
portfolio_policy_id

fold_protocol_id
bootstrap_protocol_id
stress_scenario_id
```

A result must never be described only as “Config A” when these identities differ.

## 7A.2 Baseline identity at every layer

The platform must support a distinct baseline for each dimension:

```text
baseline_data_lineage_id
baseline_market_universe_id
baseline_strategy_profile_id
baseline_cohort_id
baseline_label_policy_id
baseline_feature_bundle_id
baseline_model_protocol_id
baseline_decision_policy_id
baseline_execution_policy_id
baseline_cost_policy_id
baseline_risk_policy_id
baseline_prop_contract_id
baseline_payout_policy_id
baseline_portfolio_policy_id
baseline_validation_protocol_id
```

Example:

```text
Strategy baseline:
    ifvg_v2_doc_default_fresh_static_1r

Cohort baseline:
    all post-warmup candidates or all post-warmup sequential executions,
    explicitly selected

Label baseline:
    candidate_static_r1_v2

Feature baseline:
    IFVG_CORE_BASELINE_V1

Model baseline:
    training-prevalence reference, fixed logistic baseline, or fixed CatBoost protocol

Execution baseline:
    confirmation-close entry, next-1m-bar stop-first resolution

Cost baseline:
    verified NQ commission/slippage policy

Realization baseline:
    personal-account execution with no prop constraints
```

Every comparison must display:

```text
Changed dimensions
Frozen dimensions
Derived dimensions
Unavailable dimensions
```

## 7A.3 Experiment-dimension registry

Define a versioned `ExperimentDimensionSpec` registry.

Each dimension must declare:

```text
dimension_id
dimension_type
human_name
technical_key
description

baseline_value
allowed_values
value_schema
capability_status
owner_ratification_status

requires_full_strategy_replay
requires_feature_materialization
requires_label_recomputation
requires_model_refit
requires_model_gated_sequential_replay
requires_cost_recomputation
requires_prop_resimulation
requires_bootstrap_resimulation

can_change_setup_population
can_change_candidate_population
can_change_trade_order
can_change_account_state
can_change_label_meaning

compatible_dimensions
incompatible_dimensions
dependencies
identity_effect
```

Required `dimension_type` classes include:

```text
data_lineage
market_universe
strategy_profile
observation_cohort
label_policy
feature_block
model_protocol
decision_policy
execution_policy
cost_policy
risk_policy
prop_contract
payout_policy
portfolio_policy
validation_protocol
stress_scenario
engineering_protocol
```

The registry must fail closed when an unknown, blocked, or incompatible dimension enters a
study.

## 7A.4 Comparison contract

Define a typed comparison contract such as:

```text
ComparisonSpec
ComparisonCompatibility
ComparisonResult
```

The comparison contract must store:

```text
baseline_cell_id
challenger_cell_ids
changed_dimension_ids
frozen_dimension_ids
derived_dimension_ids

comparison_class
required_equalities
permitted_differences
computation_path
metric_registry
uncertainty_protocol
compatibility_status
incompatibility_reasons
```

Required `comparison_class` values:

```text
feature_only
cohort_descriptive
cohort_model
strategy_counterfactual
label_counterfactual
model_protocol
model_gate_execution
execution_policy
cost_policy
risk_policy
prop_realization
payout_policy
portfolio_policy
data_lineage
validation_protocol
stress_scenario
composite_preregistered
configuration_neighbor
baseline_to_child
child_to_child
```

## 7A.5 Computation-path rules

The system must derive the required work from the changed dimensions.

| Changed dimension | Required computation |
|---|---|
| feature block | materialize exact feature view and refit model |
| cohort only, descriptive | filter/report only; no counterfactual strategy claim |
| cohort-specific model | refit on declared cohort |
| label policy | recompute exact labels and path metrics |
| FSM/session/direction/entry-family rule | full sequential Strategy-Core replay |
| model algorithm or hyperparameters | refit under identical rows/folds |
| model threshold or abstention used for execution | frozen model plus full sequential gated replay |
| fill, exit, or management policy that changes trade lifetime | full sequential replay |
| cost policy only | cost recomputation when trade sequence is unchanged |
| risk sizing only | account/risk resimulation |
| prop contract | prop account-state resimulation |
| payout/withdrawal policy | payout-state resimulation |
| portfolio topology | correlated portfolio resimulation |
| bootstrap or stress protocol | simulation rerun only |
| data/formula/bar-construction policy | regenerate affected evidence and all dependent layers |

The dashboard must show this computation path before launch.

## 7A.6 Comprehensive delta families

The platform must support all of the following delta families. This list is intentionally
broader than the initial examples and must remain extensible.

### A. Data-lineage and source deltas

Examples:

```text
vendor or source snapshot
MBP schema version
bar source
tick versus bar evidence
calendar/formula version
source coverage
roll policy
continuous versus actual contract
1m source version
data-cleaning or gap policy
```

Measurements:

```text
row and event coverage
missing partitions
source gaps
bar/evidence parity
feature availability
changed setup/candidate/trade IDs
formula-value drift
artifact identity drift
```

### B. Market-universe deltas

Examples:

```text
NQ versus MNQ
future support for other instruments
actual contract month
roll-window policy
regular versus extended-hours universe
```

Measurements:

```text
opportunity count
tick/point risk distribution
cost-to-risk ratio
liquidity and spread
strategy expectancy
prop contract feasibility
```

No cross-instrument comparison may silently assume identical tick value, spread, depth, or
contract economics.

### C. Bar-construction and timeframe deltas

Examples:

```text
HTF anchor
timeframe set
parent-timeframe universe
execution timeframe
partial-bar policy
TIME versus tick construction
```

Measurements:

```text
changed FVG identities
changed setup roots
changed selected parents
funnel deltas
trade-set overlap
Q-40 exposure
timing drift
```

### D. FSM and strategy-construction deltas

Examples:

```text
HTF priority
HTF retention
parent priority
parent candidate pool
parent-retest timeout
opposing timeout
inversion timeout
post-inversion expiry
parent fill policy
HTF fill policy
causality
distance/locality
session eligibility
direction enablement
entry family
outside-session policy
daily execution cap
```

Measurements:

```text
raw opportunities
activations
locks
opposing selections
inversions
entry signals
decisions
executions
terminal reasons
slot occupancy
setup duration
candidate/trade additions and removals
trade-order changes
```

### E. Opportunity-population and exact-set deltas

Every strategy comparison must calculate exact set relationships:

```text
common setup IDs
added setup IDs
removed setup IDs
common candidate IDs
added/removed candidate IDs
common decision IDs
added/removed decisions
common trade IDs
added/removed trades
changed trade ordering
Jaccard overlap
first divergence
```

Also calculate why each set changed:

```text
earlier stale-parent expiry
different HTF winner
different parent selection
different fill invalidation
different session gate
slot became free
slot remained occupied
```

### F. Funnel and conversion deltas

Measurements:

```text
stage entry counts
stage advancement rates
terminal-reason counts
conditional conversion rates
first-failure changes
nonterminal replacement changes
parentless duration
stage wait duration
setup-slot utilization
candidate-to-decision conversion
decision-to-trade conversion
```

### G. Temporal and latency deltas

Examples and metrics:

```text
HTF age at tap
tap-to-parent
parent-selection-to-lock
lock-to-opposing
opposing-to-inversion
inversion-to-entry
entry-to-resolution
setup lifetime
time under water
payout drought
days to pass
days to first payout
```

Report both bars and wall/session time where available.

### H. Direction deltas

Required modes:

```text
direction as a model feature
long-only descriptive cohort
short-only descriptive cohort
long-specific model
short-specific model
long-only sequential strategy
short-only sequential strategy
both-direction strategy
setup-direction-normalized pooled model
```

The UI must state whether each comparison is descriptive, modeled, or a true strategy
counterfactual.

Canonical shorts remain blocked until ratified, but the schema must support them without
redesign.

### I. Session and calendar-regime deltas

Required modes:

```text
session as a feature
session as a descriptive cohort
session-specific model
session as an entry gate
outside-session policy
session transition
day of week
month/quarter
holiday/early close
macro-event proximity
```

Measurements:

```text
opportunity distribution
strategy expectancy
trade count
slot crowding
funnel conversion
model performance
payout reliability
concentration by session/day
```

### J. Entry-thesis deltas

Examples:

```text
fresh-FVG continuation
pure IFVG retest
first touch
CE touch
fractional penetration
rejection close
displacement after touch
```

These are separate strategy/profile families when they alter executable entry identity.

Measurements:

```text
candidate overlap
entry timing
risk distribution
payoff ratio
trade frequency
session/regime dependence
prop realization
```

### K. Feature-block deltas

The schema must support orthogonal, independently composable blocks.

Required current and planned blocks include:

```text
IFVG_CORE_BASELINE_V1
IFVG_STRUCTURE_CONTEXT_V1
IFVG_DISPLACEMENT_CONTEXT_V2
IFVG_LIQUIDITY_CONTEXT_V1
IFVG_SESSION_CONTEXT_V1
IFVG_VOLATILITY_CONTEXT_V1
IFVG_ORDER_FLOW_MBP1_V1
IFVG_REGIME_CONTEXT_V1
IFVG_KEY_LEVEL_CONTEXT_V1
IFVG_EXECUTION_LIQUIDITY_V1
```

Future-compatible optional blocks should be possible for:

```text
auction/volume-profile context
VWAP and opening-range context
scheduled-event context
overnight inventory/context
derivatives-positioning context
other owner-approved external evidence
```

These optional blocks must remain `planned` or `blocked` until their source and point-in-time
contracts exist.

### L. Order-flow and microstructure deltas

Order-flow comparisons must support:

```text
baseline versus +MBP-1
structure versus structure+MBP-1
MBP-1 stage snapshots versus MBP-1 stage-transition aggregates
price-only versus price+order-flow
```

Stage anchors:

```text
HTF tap
parent lock
opposing confirmation
inversion
entry decision
```

Measurements:

```text
source/event coverage
spread
queue imbalance
order-count imbalance
microprice
event-based OFI
aggressive flow
depletion/replenishment
absorption
post-inversion persistence
entry exhaustion
model skill delta
calibration delta
sequential gate value
prop realization delta
```

#### MBP-1 source and depth boundary

The order-flow contract must pin:

```text
order_flow_depth_policy = mbp1_only_v1
```

Rules:

- MBP-1 is the deepest live order-book source supported by this project.
- Historical preparation for this feature family must also use no deeper than MBP-1 unless a new
  owner-approved source contract is introduced.
- No feature name, source contract, artifact identity, dashboard control, or acceptance test may
  imply MBP-10 availability.
- The MBP-1 source contract must pin vendor/schema, instrument/contract, roll policy, event time,
  receive time, sequence ordering, source partitions, coverage, and gap semantics.
- Missing MBP-1 evidence preserves the strategy/candidate row as typed null plus a registered
  reason; it does not silently change the cohort.
- Derived execution-liquidity and order-flow blocks may share the same MBP-1 source artifact, but
  they must retain separate semantic identities.

### M. Market-context and regime deltas

Examples:

```text
volatility
compression/expansion
trend/path efficiency
volume/event intensity
regime cluster
session range state
day type
news/event regime
```

Required comparisons:

```text
context as model feature
stratified performance
regime-specific model
regime-specific strategy profile
regime stress scenario
```

A fitted regime model must be trained within each training fold.

### N. Geometry, liquidity, and key-level deltas

Examples:

```text
FVG sizes
zone distances
same-reaction-leg
entry locality
EQH/EQL
PDH/PDL
session highs/lows
weekly highs/lows
sweep/reclaim
nearest liquidity target
```

Measurements:

```text
coverage
near-boundary behavior
funnel restriction
candidate/trade quality
target reachability
session and direction interaction
```

### O. Cohort and segmentation deltas

Examples:

```text
all versus Asia
all versus NY
long versus short
fresh versus retest
1H-rooted versus 4H-rooted
parent timeframe
risk bucket
HTF age bucket
setup-duration bucket
visual-audit verdict
data-quality status
```

Cohort deltas must never be presented as a counterfactual executable strategy unless a separate
sequential replay exists.

### P. Label and outcome-policy deltas

Examples:

```text
R1.0
R1.5
R2.0
fixed points
liquidity target
time stop
partial plus runner
BE-managed
MFE threshold
resolution horizon
```

The platform must distinguish:

```text
counterfactual label-only comparison
actual executable exit-policy comparison
```

If a changed exit affects trade lifetime or slot availability, it requires a full sequential replay.

### Q. Model-protocol deltas

Examples:

```text
prevalence reference
regularized logistic regression
GAM or other interpretable baseline
fixed CatBoost challenger
model hyperparameters
class weights
missingness policy
retraining window
retraining cadence
pooled versus session-specific model
pooled versus direction-specific model
```

Measurements:

```text
Brier
Brier skill
log loss
AUC
calibration intercept/slope
reliability
coverage
fold stability
feature importance stability
prediction drift
```

### R. Feature-selection and regularization deltas

Examples:

```text
full bundle
preregistered minimal bundle
regularized versus unregularized
manual registered include/exclude
group-level ablation
```

The system must prohibit uncontrolled iterative subset search on the same OOS cohort.

### S. Model-decision and gating deltas

Examples:

```text
probability threshold
expected-R threshold
abstention
top-N candidate ranking
daily model gate
confidence margin
ensemble agreement
```

A prediction-table comparison is not enough when the gate changes trade occupancy.

Required outputs:

```text
predictive metrics
selected-row coverage
full sequential gated replay
added/removed executions
slot-occupancy changes
costed strategy results
prop realization
```

### T. Execution-model deltas

Examples:

```text
confirmation-close fill
next-tick fill
market-order slippage
limit-order fill probability
partial fill
entry delay
stop-first versus event ordering
tick versus 1m management
```

Measurements:

```text
fill rate
entry shortfall
stop/target shortfall
missed trades
realized payoff compression
cost in points
cost in candidate-specific R
edge leakage
```

### U. Trade-management deltas

Examples:

```text
static 1R
BE-managed
trailing stop
partial profit
runner
time stop
session-close flattening
```

Measurements:

```text
average win/loss R
realized payoff ratio
expectancy
drawdown
trade lifetime
slot occupancy
prop payout and breach metrics
```

### V. Cost-policy deltas

Examples:

```text
gross
commissions only
commissions plus baseline slippage
stress slippage
spread-aware costs
market-impact estimate
```

Measurements:

```text
gross R
cost R
net R
cost per trade
cost as percentage of risk
cost as percentage of gross edge
break-even win-rate change
```

### W. Risk and position-sizing deltas

Examples:

```text
fixed dollar
fixed contracts
percentage of starting buffer
percentage of current buffer
NQ versus MNQ
adaptive NQ/MNQ
daily shutdown
post-payout de-risking
loss-streak de-risking
```

Measurements:

```text
buffer usage
risk-of-ruin
drawdown
contract utilization
skipped trades
account survival
payout reliability
capital efficiency
```

### X. Prop-contract deltas

Examples:

```text
firm
account size
evaluation versus funded
contract version
drawdown type
daily-loss rule
consistency rule
payout rule
fees
contract limits
```

Measurements:

```text
pass probability
first-payout probability
multiple-payout probability
breach probability
account lifetime
payout drought
fee drag
net cash extraction
```

### Y. Payout and withdrawal-policy deltas

Examples:

```text
withdraw as soon as eligible
minimum buffer retained
fixed payout cadence
maximum permitted payout
partial withdrawal
post-payout risk reduction
```

Measurements:

```text
cash extracted
post-payout breach probability
future payout probability
account lifetime
buffer recovery time
lifetime value
```

### Z. Portfolio-topology deltas

Examples:

```text
one account
multiple accounts at one firm
one account per firm
multiple firms
copy-traded portfolio
firm-specific risk policies
replacement policy
```

Measurements:

```text
portfolio payout distribution
probability of at least one payout
probability all accounts breach
payout concentration
replacement cost
correlated drawdown
firm-contract diversification
```

Copied accounts must share one common strategy path.

### AA. Validation-protocol deltas

Examples:

```text
fold length
training length
embargo
purge
rolling versus expanding
outer walk-forward
bootstrap block length
number of paths
```

These comparisons measure research-protocol sensitivity and must never be used to cherry-pick the
most favorable protocol.

### AB. Stress-scenario deltas

Examples:

```text
lower win rate
compressed average winner
expanded average loss
higher costs
missed winning fills
clustered losses
lower opportunity frequency
session concentration
edge decay
data outages
```

Measurements:

```text
strategy survival
prop survival
frontier movement
parameter-region stability
failure reasons
```

### AC. Time stability and edge-decay deltas

Required views:

```text
chronological thirds
rolling windows
month/quarter
pre/post regime
recent versus historical
opportunity-rate drift
funnel-conversion drift
payoff compression
loss expansion
cost drift
```

The dashboard must distinguish normal variance from a statistically or mechanically meaningful
change in the process.

### AD. Evidence-quality and audit deltas

Examples:

```text
complete versus missing feature source
formula version
exact versus unavailable evidence
visual-review verdict
candidate with versus without complete geometry
```

Measurements:

```text
coverage
missing reasons
quarantine
point-in-time validity
reviewer verdict distribution
metric sensitivity to evidence-quality filters
```

### AE. Engineering and operational deltas

These are not trading-edge claims, but they are required for platform decisions:

```text
replay runtime
memory
artifact size
job throughput
failure/retry rate
dashboard load time
chart render time
storage growth
```

## 7A.7 Delta output registry

For each compatible comparison, Quant-Lab must automatically select the applicable delta output
families.

Required output families:

```text
identity_delta
compatibility_delta
population_delta
funnel_delta
sequence_delta
timing_delta
feature_coverage_delta
predictive_delta
calibration_delta
label_delta
economic_delta
execution_delta
cost_delta
risk_delta
tail_delta
stability_delta
concentration_delta
prop_delta
payout_delta
portfolio_delta
robustness_delta
evidence_quality_delta
engineering_delta
```

## 7A.8 Core metric inventory

### Identity and compatibility

```text
changed dimensions
frozen dimensions
artifact IDs
schema hashes
profile hashes
trade-stream hashes
OOS cohort hashes
fold hashes
contract versions
```

### Population

```text
common/added/removed setups
common/added/removed candidates
common/added/removed decisions
common/added/removed trades
Jaccard overlap
first divergence
```

### Funnel

```text
stage counts
conditional conversion
terminal reasons
replacement counts
parentless duration
slot occupancy
```

### Timing

```text
stage wait times
setup duration
trade duration
time under water
payout drought
days to pass/payout
```

### Strategy economics

```text
planned RRR
realized payoff ratio
gross/net E[R]
profit factor
total R
trade frequency
```

### Tail and path risk

```text
max drawdown
drawdown duration
loss streak
VaR/CVaR or lower quantiles
10th-percentile payout
risk of ruin
```

### Model quality

```text
Brier
Brier skill
log loss
AUC
calibration
coverage
reliability
```

### Feature quality

```text
coverage
missingness
variance
constant/near-constant
fold coverage
long/short coverage
session coverage
stage coverage
source latency
```

### Execution leakage

```text
entry shortfall
stop/target shortfall
cost R
missed fills
payoff compression
```

### Prop realization

```text
pass
payout
survival
breach
fees
lifetime
drought
net cash
```

### Robustness

```text
outer-fold recurrence
stress passes
neighbor stability
worst-firm result
knife-edge score
```

## 7A.9 Main effects and interactions

When the search charter defines a balanced or otherwise interpretable multi-axis design, the
backend should calculate:

```text
matched one-axis contrasts
main effects
two-way interaction effects
conditional effects
neighbor effects
firm-specific effects
regime-specific effects
```

Example:

```text
parent timeout main effect
parent-fill-policy main effect
timeout × parent-fill interaction
timeout effect within Topstep
timeout effect in Asia
```

Rules:

- Use only declared contrasts.
- Record the exact denominator and compatible child set.
- Do not call an observational marginal effect causal.
- Do not calculate a main effect from an unbalanced search without an explicit adjustment method.
- Always retain the raw child results.

## 7A.10 Comparison hierarchy

The dashboard must support several baselines:

```text
canonical baseline
parent configuration
nearest one-axis neighbor
selected robust representative
highest-reliability point
per-firm champion
user-pinned reference
```

The user should be able to switch the comparison reference without changing the immutable study.

## 7A.11 Feature-block contract

Define a `FeatureBlockSpec` containing:

```text
block_id
block_version
human_name
status
feature_family

source_kind
source_artifact_refs
source_schema_hash
feature_schema_hash

feature_names
numeric_features
categorical_features
validity_fields
missing_reason_fields

availability_stage
source_timeframes
source_interval_policy
as_of_policy
join_keys
join_policy

direction_normalization
session_normalization
warmup_requirement
coverage_requirements

dependencies
incompatible_blocks
experimental_flags

requires_strategy_replay
requires_feature_materialization
requires_model_retrain
can_affect_execution
```

Statuses:

```text
available
planned
blocked_missing_source
blocked_owner_decision
experimental
superseded
```

## 7A.12 Feature-bundle contract

Define a composable `FeatureBundleSpec`:

```text
bundle_id
bundle_version
human_name

base_bundle_id
included_block_ids
excluded_feature_ids
manual_include_ids
manual_exclude_ids

resolved_feature_names
resolved_source_identities
bundle_schema_hash

required_profile_capabilities
required_coverage_gates
```

The architecture must support a graph of bundles rather than only a cumulative enum ladder.

Examples:

```text
B0_CORE
B1_CORE_STRUCTURE
B2_CORE_ORDER_FLOW
B3_CORE_STRUCTURE_ORDER_FLOW
B4_CORE_STRUCTURE_LIQUIDITY
B5_CORE_STRUCTURE_ORDER_FLOW_REGIME
B6_CORE_STRUCTURE_ORDER_FLOW_EXECUTION_LIQUIDITY
```

## 7A.13 Cohort contract

Define a `CohortSpec` with:

```text
cohort_id
source_profile_id
source_artifact_id
row_kind
warmup_policy
date policy
session filter
direction filter
entry-family filter
HTF/parent filter
regime filter
evidence-quality filter
executed/counterfactual scope
minimum coverage
```

The UI must label cohort results as descriptive or model-specific unless a corresponding
sequential strategy profile exists.

## 7A.14 Session and direction interpretation selector

When a user selects a session or direction, the UI must ask how it should be interpreted:

```text
Analysis slice only
Train a specialized model
Build a new sequential strategy profile
```

For example:

```text
Asia only
    ├── descriptive cohort
    ├── Asia-specific model
    └── Asia-only executable profile
```

Only the third option creates a true strategy counterfactual.

## 7A.15 Comparison compatibility rules

Quantitative paired deltas require equality of every nonchanged load-bearing dimension.

Examples:

### Feature-only delta

Require identical:

```text
strategy profile
candidate IDs
labels
folds
model protocol
cost policy
```

### Prop-contract delta

Require identical:

```text
trade-stream hash
risk policy, unless explicitly varied
payout policy
simulation paths
```

### Risk-policy delta

Require identical:

```text
trade-stream hash
prop contract
simulation paths
```

### Strategy-profile delta

Allow different setup/trade identities, but require:

```text
same source data
same authorized dates
same cost model
same label/execution semantics unless explicitly changed
same validation protocol
```

Incompatible studies receive a configuration diff only, not a controlled metric delta.

## 7A.16 Dashboard requirements for deltas

Every comparison view must display:

```text
delta type
changed dimensions
frozen dimensions
computation path
compatibility status
sample/cohort identity
uncertainty method
development/validation status
```

Required UI components:

```text
dimension-diff ribbon
feature-block matrix
population-overlap panel
funnel delta
timing delta
strategy economics delta
prop delta
stability/robustness delta
evidence-quality panel
exact affected-setup list
```

## 7A.17 Controlled-study examples

The completed platform must support, at minimum:

```text
Study A:
Core baseline versus core + MBP-1.
Same profile, rows, labels, folds, and model protocol.

Study B:
Core + structure versus core + structure + MBP-1.
Only the MBP-1 feature block changes.

Study C:
Pooled rows versus session-context feature.
Same rows; session becomes an input.

Study D:
Pooled model versus Asia-only model.
Different cohort; clearly labeled; no executable-strategy claim.

Study E:
All-session strategy versus Asia-only strategy.
Separate full sequential replays.

Study F:
Long descriptive cohort versus short descriptive cohort.
No strategy-counterfactual claim.

Study G:
Long-only versus short-only executable strategies.
Separate profiles; blocked until canonical shorts are ratified.

Study H:
R1.0 versus R2.0 counterfactual labels.
Same candidates; separate label identities.

Study I:
Static 1R versus BE-managed executable policy.
Full sequential replay if trade lifetime/occupancy changes.

Study J:
Fixed CatBoost versus regularized logistic model.
Same rows, features, labels, and folds.

Study K:
Frozen model without gate versus threshold-gated execution.
Full sequential gated replay.

Study L:
Confirmation-close fill versus stressed slippage model.
Execution/cost delta.

Study M:
Fixed dollar versus current-buffer risk.
Same trade stream and prop contract.

Study N:
Topstep versus Apex realization.
Same strategy trade-stream hash and risk policy.

Study O:
Immediate payout versus buffer-retention withdrawal policy.
Same contract and trade stream.

Study P:
One account versus multi-firm copied portfolio.
Common correlated market path.

Study Q:
Historical path versus clustered-loss stress.
Same strategy, contract, and risk policy.

Study R:
Formula/version A versus formula/version B.
Explicit data/evidence lineage comparison.

Study S:
Baseline versus nearest one-axis FSM neighbor.
Full sequential replay and exact population/funnel delta.

Study T:
Two-axis timeout × fill-policy search.
Main effects, interaction, robust-basin analysis, and raw child results.
```

## 7A.18 Guardrails

The system must prohibit or warn on:

```text
FSM + feature + label + model + prop-risk search in one uncontrolled grid
post-hoc use of a cohort slice as a strategy counterfactual
use of candidate outcomes as executed-trade performance
use of feature-only model metrics as proof of prop value
selection of a threshold after viewing the same OOS curve
silent inclusion of planned/blocked feature blocks
silent mixing of contract versions
independent simulation of copied accounts
causal wording for descriptive marginal effects
```

## 7A.19 Delta acceptance criteria

A completed implementation must prove:

1. A new registered feature block can be added without mutating baseline artifacts.
2. Feature-only comparisons preserve identical candidate IDs and folds.
3. Session/direction cohort studies do not claim strategy-counterfactual execution.
4. Session/direction strategy changes force new sequential profiles.
5. Label changes create new identities and path metrics.
6. A model gate forces a sequential gated replay before execution/prop claims.
7. Risk and prop changes reuse the same trade-stream hash.
8. All compatible comparisons select the correct metric families automatically.
9. Incompatible comparisons show identity/config differences without misleading deltas.
10. Exact added/removed/common setup, candidate, decision, and trade sets are available.
11. Balanced factorial studies report preregistered main and interaction effects.
12. The dashboard exposes all changed/frozen dimensions.
13. The planned MBP-1 block appears with explicit capability status, and no MBP-10/deeper-book block or control is exposed.
14. Order-flow blocks preserve missing rows with typed reasons.
15. Every insight links to the exact comparison and supporting evidence.



# 7B. Machine Learning, Regime Modeling, and Spectral-Clustering Contract

The earlier sections define a generic `regime cluster`, planned regime feature block, model-protocol
dimension, fixed interpretable and CatBoost baselines, and fold-local fitting. Those references are
not sufficiently precise to govern an implementation.

This section makes the intended ML concepts explicit.

The platform must treat ML as a separately identified research layer. It must never silently turn a
descriptive cluster, feature importance, model probability, or prop-firm outcome into an executable
strategy rule.

## 7B.1 ML role taxonomy

Every fitted statistical artifact must declare one role:

```text
descriptive_only
stratification_only
feature_generator
predictive_model
decision_policy
execution_gate_candidate
frozen_execution_gate
monitoring_only
```

Required promotion sequence for a regime artifact:

```text
descriptive_only
    -> stratification_only
    -> feature_generator
    -> predictive-model input
    -> execution-gate candidate
    -> frozen execution gate after full sequential replay
```

A regime cluster begins as `descriptive_only` or `stratification_only`.

It must not become:

```text
an FSM guard
a trade eligibility rule
a model execution gate
a prop-risk control
```

without a separately frozen protocol and downstream sequential replay.

## 7B.2 Supervised model ladder

The standard supervised comparison ladder must support:

```text
Reference 0:
    fold-local training prevalence

Model A:
    regularized logistic regression

Model B:
    fixed CatBoost challenger

Optional Model C:
    preregistered GAM or another interpretable nonlinear baseline
```

Required rules:

- All models in a controlled comparison use identical rows, labels, folds, costs, and feature
  bundles unless the declared delta specifically changes one of those dimensions.
- The prevalence reference is mandatory for every fold.
- Regularized logistic regression is the default interpretable model.
- CatBoost is the default nonlinear challenger.
- GAM support is optional and must use a fixed, preregistered basis/penalty protocol.
- Any hyperparameter search is a separate, explicitly authorized model-protocol study.
- No implementation plan may silently add automated model selection, Bayesian optimization,
  threshold selection, calibration selection, or feature-subset search.
- Deep LOB neural networks, end-to-end CNN/LSTM/Transformer entry models, and reinforcement
  learning for entries, trade management, sizing, or prop-account control are out of scope for
  this plan.

Primary supervised metrics:

```text
Brier score
Brier skill versus the fold-local prevalence reference
log loss
AUC when defined
calibration intercept
calibration slope
reliability bins
coverage
abstention coverage where applicable
fold stability
trading-day block uncertainty
```

Secondary descriptive diagnostics may include:

```text
grouped permutation importance
fold-native model importance
SHAP for frozen tree-model diagnostics
partial dependence or ICE when coverage is adequate
```

These diagnostics are descriptive. They do not authorize automatic feature selection.

## 7B.3 Feature-selection and regularization policy

Permitted:

```text
preregistered feature-block ablations
preregistered minimal feature bundles
regularization-strength comparisons under a frozen protocol
manual registered include/exclude lists entering immutable identity
```

Prohibited:

```text
iterative top-N selection on the same OOS rows
backward elimination after inspecting the same folds
selecting features from pooled OOS importance and claiming independent validation
automatic feature promotion from SHAP/permutation rank
```

Feature-only studies must preserve exact candidate IDs, labels, folds, and model protocols.

## 7B.4 Regime-model contract

Define a versioned contract, conceptually:

```text
RegimeModelSpec
RegimeFitArtifact
RegimeAssignment
RegimeCoverageReport
RegimeStabilityReport
```

Required `RegimeModelSpec` fields:

```text
regime_model_id
algorithm_id
algorithm_version
role
status

input_feature_bundle_id
resolved_input_features
observation_granularity
observation_stage
source_artifact_ids

fit_scope
fit_start
fit_end
fold_id
training_row_ids_hash

missingness_policy
winsorization_policy
scaler_policy
dimensionality_reduction_policy
kernel_or_affinity_policy

cluster_count_policy
resolved_cluster_count
random_seed
initialization_policy

out_of_sample_assignment_policy
cluster_label_alignment_policy
minimum_cluster_occupancy
minimum_assignment_confidence

software_versions
formula_version
schema_hash
artifact_hash
```

Allowed statuses:

```text
planned
descriptive_only
stratification_ready
feature_eligible
model_feature
experimental
blocked_no_oos_assignment
blocked_insufficient_coverage
superseded
```

## 7B.5 Regime observation granularity

Do not fit a regime model to every MBP event or every raw tick.

Permitted observation grains:

```text
one row per exact IFVG candidate stage
one row per exact decision
one row per completed fixed context interval
one row per completed 1m or coarser market-context bar
```

The observation grain and as-of instant enter identity.

Candidate-stage examples:

```text
HTF tap
parent lock
opposing confirmation
inversion
entry decision
```

A regular market-context panel may be used to increase the regime-training sample, provided every
row uses only completed, point-in-time evidence.

## 7B.6 Regime input features

Permitted point-in-time inputs may include:

```text
realized volatility
short/long volatility ratio
range compression or expansion
trend and path efficiency
directional persistence
session state
session-range percentile
distance from session open
volume
trade intensity
quote intensity
spread
MBP-1 queue imbalance
MBP-1 order-count imbalance
MBP-1 OFI
MBP-1 depletion and replenishment
MBP-1 aggressive-flow imbalance
top-of-book liquidity state
```

Prohibited inputs:

```text
candidate label
trade outcome
MFE
MAE
resolution
future session range
future daily statistics
post-entry evidence when the regime is assigned before entry
prop payout or breach outcome
any feature unavailable at the assignment as-of instant
```

## 7B.7 Fold-local preprocessing

Every fitted preprocessing step must be trained inside the training fold:

```text
missing-value policy
winsorization
scaling
PCA or another projection
kernel approximation
cluster-count selection
clusterer
out-of-sample assignment model
cluster-label alignment reference
```

No transform may be fitted on:

```text
the full dataset
pooled OOS rows
future test rows
prop outcomes
sealed data
```

The complete fitted preprocessing pipeline is part of the regime artifact.

## 7B.8 Supported regime algorithms

The implementation architecture must support an algorithm registry.

### Production baseline — `kmeans_v1`

Use when:

```text
hard cluster assignment is sufficient
the feature space is scaled and reasonably compact
out-of-sample assignment is required
```

Persist:

```text
cluster ID
distance to every centroid
distance to assigned centroid
assignment margin
```

### Large-sample production baseline — `minibatch_kmeans_v1`

Use only when the observation count makes standard KMeans unnecessarily expensive.

It must preserve a deterministic seed and fixed batch protocol.

### Soft-assignment challenger — `gaussian_mixture_v1`

Persist:

```text
component ID
component probabilities
assignment entropy
log density
```

A covariance policy must be explicit and bounded.

### Direct spectral clustering — `spectral_clustering_train_only_v1`

Direct spectral clustering is an **exploratory training-only algorithm** unless a valid
out-of-sample assignment policy is explicitly supplied.

Required constraints:

- Fit only on training-fold observations.
- Never fit an affinity matrix using combined training and test rows.
- Never refit the clustering on each test row or on pooled OOS data.
- Do not run direct spectral clustering at tick or MBP-event scale.
- Bound the maximum training observations and affinity-matrix memory.
- Record affinity type, kernel parameters, neighbor count, eigen-solver, label-assignment method,
  seed, and numerical tolerances.
- If no predeclared out-of-sample assignment is available, the result remains
  `descriptive_only` or `blocked_no_oos_assignment`.
- A training-only spectral cluster ID must not enter the test feature matrix.

### Production-compatible spectral approximation — `nystrom_kmeans_v1`

When a spectral-style nonlinear partition is desired with deterministic out-of-sample assignment,
the preferred first production-compatible approximation is:

```text
fold-fitted scaler
    -> fold-fitted Nyström kernel feature map
    -> fold-fitted KMeans
    -> transform and assign validation/test rows
```

Required identity fields:

```text
kernel
gamma or kernel parameters
number of Nyström components
component sampling seed
KMeans configuration
```

This is a separate algorithm from direct `SpectralClustering` and must not be mislabeled.

### Optional frozen surrogate assignment

A direct spectral-training result may become feature-eligible only if a separately registered
surrogate assignment policy is used:

```text
fit spectral clusters on training rows
fit a simple assignment model on training features -> training cluster IDs
apply the frozen assignment model to test rows
```

The surrogate algorithm, training data, accuracy, uncertainty, and identity must be persisted.
This policy is secondary to `nystrom_kmeans_v1` and must be explicitly authorized.

## 7B.9 Cluster-count policy

Supported policies:

```text
fixed_k
inner_train_only_selection
```

The first implementation should prefer `fixed_k` or a very small preregistered candidate set.

If cluster count is selected:

- selection occurs only inside the inner training data;
- the selection metric and tie-break are frozen;
- the test fold, candidate outcomes, strategy expectancy, and prop metrics are not used;
- the selected count is stored per fold.

Do not choose cluster count by whichever value produces the best trading or prop result.

## 7B.10 Cluster IDs are nominal

Cluster IDs do not have an ordinal meaning.

The artifact must persist:

```text
fold_local_cluster_id
canonical_reporting_cluster_id
cluster_semantic_summary
```

Cross-fold reporting may align labels using a deterministic centroid/component matching policy,
such as minimum-distance assignment with a stable tie-break.

Label alignment is for reporting. It must not retroactively change a fold's model inputs or
predictions.

The UI must never imply:

```text
regime 2 > regime 1
```

unless a separate ordered economic state variable is explicitly defined.

## 7B.11 Regime outputs

Depending on the algorithm, expose:

```text
hard cluster ID
distances to clusters
soft probabilities
assignment margin
assignment entropy
outlier or low-density score
cluster occupancy
cluster centroid/component descriptors
regime persistence
regime transition counts
```

Prefer distances or probabilities over a hard ID alone.

A CatBoost model may receive a hard cluster ID as a categorical feature only when the assignment
is available point-in-time and out-of-sample.

## 7B.12 Regime stability and quality diagnostics

Required diagnostics:

```text
cluster occupancy
minimum and maximum cluster size
feature coverage by cluster
assignment confidence
assignment entropy
centroid/component separation
bootstrap assignment stability
temporal persistence
transition rate
fold-to-fold recurrence
semantic centroid descriptors
out-of-sample assignment coverage
```

Optional diagnostics:

```text
silhouette
Calinski-Harabasz
Davies-Bouldin
adjusted mutual information after label alignment
prediction-strength style stability
```

No single internal clustering score is sufficient for promotion.

A cluster family becomes `feature_eligible` only when:

- occupancy is not pathologically sparse;
- assignments are stable enough for the declared role;
- OOS assignment exists;
- source coverage is adequate;
- no leakage is present;
- the cluster has interpretable point-in-time descriptors.

## 7B.13 Regime research uses

The dashboard and backend must support separate questions:

```text
Does the deterministic strategy behave differently by regime?
Does the model rank candidates better when regime is included as a feature?
Should separate models be fitted by regime?
Does a strategy configuration work only in one regime?
How do prop survival and payout reliability change by regime?
Does regime occupancy drift over time?
```

These are separate comparisons:

```text
stratification-only
regime-feature delta
regime-specific model delta
regime-specific strategy-profile delta
regime stress delta
```

A regime-specific executable profile requires a full sequential replay. A stratified report does
not.

## 7B.14 Supervised use of regime features

Controlled comparisons should include:

```text
baseline model
baseline + hard regime ID
baseline + regime distances/probabilities
baseline + MBP-1
baseline + MBP-1 + regime outputs
```

All nonchanged dimensions must remain identical.

A regime feature should earn inclusion through:

```text
incremental OOS Brier/Brier-skill improvement
calibration improvement
fold stability
coverage
lack of one-regime or one-day concentration
```

It should not be selected because one regime has a favorable in-sample expectancy.

## 7B.15 Regime-specific models

A regime-specific model is permitted only when:

```text
the regime assignment is available before prediction
each training partition has minimum sample and both-class coverage
the model protocol is preregistered
the pooled baseline remains available
invalid regime folds are reported rather than hidden
```

The dashboard must compare:

```text
pooled model
pooled model + regime feature
separate regime-specific models
```

under identical outer test rows where comparison is valid.

## 7B.16 Calibration policy

Raw probability diagnostics remain mandatory.

If a fitted calibrator is used:

- fit it only inside each training fold;
- record the calibration training subset;
- apply it once to the corresponding test rows;
- do not fit on pooled OOS predictions and score the same rows;
- treat calibrator type as a model-protocol dimension.

Supported future calibrators may include:

```text
Platt/logistic calibration
isotonic calibration when sample support is sufficient
```

The first implementation may remain raw-diagnostics-only.

## 7B.17 Model decision policies

Prediction metrics and executable decisions remain separate.

Potential registered policies:

```text
fixed probability threshold
expected-R threshold
abstention band
top-N per day
minimum confidence margin
regime-conditioned threshold
```

Any policy that changes which candidate executes requires:

```text
frozen OOS-capable model
frozen decision policy
full sequential model-gated replay
new trade-stream hash
new prop simulations
```

A threshold table alone cannot establish execution value.

## 7B.18 ML and prop-firm separation

Do not train a regime model or supervised trade model directly on:

```text
payout occurred
challenge passed
account breached
```

as the first trade-quality target.

Those outcomes depend on prior trades, risk sizing, withdrawal policy, and contract state.

The initial hierarchy is:

```text
market/setup features
    -> trade-quality model
    -> frozen sequential decision policy
    -> executed trade stream
    -> prop-account realization
```

Prop metrics evaluate downstream realization. They do not replace predictive validation.

## 7B.19 Drift and edge-decay monitoring

The platform should support monitoring-only metrics:

```text
feature-distribution drift
prediction-distribution drift
calibration drift
regime occupancy drift
regime transition drift
session/direction mix drift
FSM funnel-conversion drift
payoff compression
loss expansion
opportunity-rate change
MBP-1 source-coverage drift
```

Potential statistical diagnostics include:

```text
Wasserstein distance
KS-style distribution comparison
population stability index
change-point indicators
rolling calibration and Brier
```

These remain monitoring evidence. V1 must not automatically retrain, disable, or promote a model
from one drift alarm.

## 7B.20 ML dashboard requirements

Add a dedicated analyst surface within each compatible study.

### Model ladder

Show:

```text
prevalence reference
regularized logistic
CatBoost
optional GAM
```

Metrics appear on identical rows/folds.

### Feature-bundle matrix

Show each block, source status, coverage, and whether it was available to the model.

### Regime model card

Show:

```text
algorithm
role/status
input bundle
observation grain
fit window/fold
cluster count
OOS assignment policy
coverage
minimum occupancy
stability
artifact identity
```

### Regime diagnostics

Required visuals:

```text
cluster occupancy
centroid/component profile
distance/probability distributions
assignment confidence/entropy
regime timeline
regime transition matrix
strategy metrics by regime
model metrics by regime
prop metrics by regime
```

### Spectral warning states

Direct spectral clustering must display:

```text
Training-only exploratory clustering
No OOS assignment — cannot enter predictive feature bundle
```

or:

```text
OOS assignment supplied by <registered policy>
```

### Explainability

Show grouped feature-block importance first. Individual-feature SHAP/permutation views are
secondary and carry small-sample and same-development-data warnings.

### Drift

Show current versus reference feature, prediction, and regime distributions, with explicit data
windows and sample counts.

## 7B.21 Five-day verification policy for ML

Implementation verification remains capped at five real authorized trading days total.

Use:

```text
synthetic class-balanced supervised fixtures
synthetic known-cluster fixtures
synthetic spectral graph fixtures
synthetic OOS-assignment fixtures
a fixed <=5-day real mini-pipeline for source/join/UI control-flow verification
```

The five-day verification must not be interpreted as evidence of model or regime quality.

It must prove:

```text
pipeline execution
fold-local fitting
artifact identity
save/reload/reuse
missingness handling
OOS assignment behavior
UI rendering
safe failure states
```

Full regime fitting, feature materialization, model training, and evaluation over the authorized
development dataset occur only through the post-implementation UI-launched full pipeline.

## 7B.22 ML acceptance criteria

A completed implementation must prove:

1. The prevalence, logistic, and CatBoost ladder can run on identical rows/folds.
2. Every fitted preprocessing step is fold-local.
3. Regime inputs contain no target, outcome, post-entry, or future information.
4. KMeans assignments are deterministic and available OOS.
5. Gaussian-mixture probabilities and entropy are deterministic and available OOS.
6. Direct spectral clustering cannot enter a test feature bundle without a registered OOS
   assignment policy.
7. A Nyström+KMeans pipeline can transform and assign OOS rows deterministically.
8. Cluster count is fixed or selected only inside inner training data.
9. Cluster IDs are treated as nominal and reporting alignment does not alter model inputs.
10. Missing regime evidence preserves rows with typed reasons.
11. Regime-feature deltas preserve identical rows, labels, and folds.
12. Regime-specific strategy changes force a new sequential profile.
13. Model decision policies force a full sequential gated replay before execution or prop claims.
14. Calibration fitting, when enabled, is training-fold-only.
15. No automatic feature, threshold, cluster-count, model, or prop-metric selection occurs outside
    a separately frozen search charter.
16. Five-day verification uses only the fixed mini-fixture and synthetic ML fixtures.
17. The full authorized model/regime pipeline is launched only through the standardized operator UI.
18. Direct spectral, Nyström-spectral approximation, KMeans, and Gaussian-mixture artifacts have
    distinct identities and cannot be mislabeled.
19. Deep learning and reinforcement-learning controls are absent from V1.
20. Every ML insight links to exact rows, folds, artifacts, and uncertainty evidence.


# 8. High-Level Architecture

```text
Verified baseline artifacts
        |
        v
Immutable Search Charter
        |
        v
Authorized Strategy Configuration Enumerator
        |
        +------------------------------+
        |                              |
        v                              v
Full Sequential Replay A          Full Sequential Replay N
        |                              |
        v                              v
Immutable Child Strategy Results and Audit Artifacts
        |
        v
Underlying Strategy-Quality Gates
        |
        v
Versioned Prop Contracts × Firm-Specific Risk Policies
        |
        v
Exact Historical Account Replays
        |
        v
Day-Block Bootstrap + Stress Simulations
        |
        v
Feasible Set + Pareto Frontier + Stability Analysis
        |
        v
Robust Representative + Failure Attribution
        |
        v
Dashboard + Exact Setup/Trade/Account-Event Drill-Down
```

The planning agent must determine which existing Quant-Lab job, artifact, reporting, chart, catalog, and immutable-store abstractions should own each stage.

---

# 9. Backend Requirements

## 9.1 New research-lane contract

Create a separate versioned contract, conceptually:

```text
ifvg_prop_robust_config_search_v1
```

The final name should follow repository conventions.

It must not extend the fixed M0–M3 contract in a way that permits silent adaptive search.

## 9.2 Search charter

Define an immutable search charter contract such as:

```text
IfvgStrategySearchSpec
```

Required content:

```text
search mode
baseline profile reference
authorized strategy axes
allowed values per axis
locked invariants
measured-only fields
blocked capabilities
authorized firms
authorized risk-policy families
underlying strategy gates
prop feasibility gates
robustness gates
objective and tie-break policy
date policy
warmup policy
simulation protocol
bootstrap protocol
stress protocol
maximum child count
search algorithm
seed
software and artifact identities
```

Once launched, the charter is frozen.

Changes require:

```text
Clone as New Search
```

## 9.3 Search-axis registry

Every searchable field must be registered with:

```text
human label
technical key
description
classification
value type
allowed values
baseline value
profile/version effect
whether full sequential replay is required
owner-ratification status
capability status
dependencies
expected artifact effect
```

Classifications:

```text
locked correctness invariant
strategy-thesis-defining
approved search axis
risk-policy axis
measurement-only
blocked
experimental
```

Examples of permanently locked invariants:

```text
one active setup
one active trade
no pyramiding
own-timeframe FVG close discipline
strict body-close inversion
no entry on inversion candle
entry-at-confirmation-close chronology
exact point-in-time evidence
```

Examples of potential future approved axes after owner review:

```text
parent-retest staleness
parent-fill handling
HTF selection policy
HTF age policy
post-inversion expiry
session entry policy
distance and size policies
```

The system must fail closed when an unregistered or blocked axis enters a charter.

## 9.4 Child strategy identity

Every strategy child must have a deterministic identity containing at least:

```text
parent search ID
baseline profile identity
resolved section/config values
strategy version
profile name
section hash
authorized dates
warmup policy
resolver identity
cost policy
Strategy-Core commit
source artifact identities
audit schema identity
```

A child with an existing identical identity should be verified and reused rather than rerun.

## 9.5 Full sequential replay orchestration

For every unique strategy configuration:

1. Build or resolve the exact named profile.
2. Execute the full sequential Strategy-Core replay.
3. Persist separate setup, candidate, decision, execution, audit, and replay-chart artifacts.
4. Verify one-active-setup and one-active-trade invariants.
5. Persist funnel and terminal-reason counts.
6. Persist exact strategy performance after costs.
7. Persist the trade-stream hash.
8. Persist failure status and reason.
9. Never derive strategy execution by filtering an existing candidate table.

The orchestrator must support:

```text
queued
running
completed
failed
reused
cancelled at safe boundary
blocked
```

It must be resumable, lock-guarded, crash-safe, and idempotent.

## 9.6 Parent and child job state

The parent search must expose:

```text
charter frozen
children enumerated
replays queued
replays running
replays completed
underlying-edge passed
prop simulations running
prop-feasible
robustness passed
frontier complete
search complete
```

Every child must expose:

```text
config identity
replay status
strategy-gate status
prop status per firm
robustness status
failure reason
artifact references
```

## 9.6A Verification-run scope and five-day data budget

Define a typed verification policy, conceptually:

```text
VerificationRunSpec
VerificationDataPolicy
```

Required identity fields:

```text
verification_policy_id
fixed_authorized_date_allowlist
allowlist_hash
maximum_real_trading_days = 5
warmup_days
evidence_days
synthetic_fixture_ids
precomputed_seed_ids
software commits
test namespace
```

The verification orchestrator must fail before source-path construction when:

```text
number of real authorized dates > 5
warmup + evidence dates > 5
a date is not on the frozen verification allowlist
a test attempts a different real-data allowlist in the same release verification
a normal research catalog is selected as the destination
```

The five-day verification scope must exercise the complete control flow without claiming research
validity:

```text
profile preparation
sequential replay
audit and replay artifacts
feature materialization
coverage validation
label derivation
fold/model micro-run where a synthetic or class-balanced fixture is needed
prop account replay
immutable save/reload/reuse
dashboard progress and result rendering
```

Verification reports must carry:

```text
verification_only = true
not_for_research_interpretation = true
real_date_count
real_date_allowlist_hash
synthetic_fixture_ids
full_pipeline_not_run = true
```

Performance measurements used for implementation acceptance must also use no more than the same
five real trading days. A separate operator-run performance estimate may be derived later from
phase timings, but no full-data benchmark is permitted during implementation verification.

## 9.6B Standardized full-pipeline run contract

Define a separate operator-facing contract, conceptually:

```text
QuantLabPipelineRunSpec
QuantLabPipelineRunIdentity
QuantLabPipelineStage
QuantLabPipelineRunResult
```

Required run scopes:

```text
verification_5d
full_authorized_development
```

`verification_5d` is used by tests and implementation verification. `full_authorized_development`
is available only through the released operator workflow and must never be invoked automatically.

The full pipeline identity must pin:

```text
run scope
authorized date allowlist and hash
warmup policy
strategy/search charter
source artifact identities
feature bundles
label policy
fold protocol
model protocol
cost policy
prop contracts
risk policies
simulation protocol
software commits
worker and resource policy
```

Standardized stages should include, where applicable:

```text
00_validate_inputs
01_prepare_strategy_profiles
02_run_or_reuse_sequential_replays
03_build_or_reuse_fsm_audit
04_build_or_reuse_replay_charts
05_materialize_feature_views
06_validate_feature_coverage
07_derive_labels
08_build_folds
09_train_models
10_generate_predictions_and_diagnostics
11_run_frozen_model_gated_replays
12_run_prop_historical_replays
13_run_bootstrap_and_stress
14_build_frontier_and_insights
15_verify_and_publish
```

Every stage must expose:

```text
pending
queued
running
checkpointed
completed
reused
failed
cancel_requested
cancelled_at_safe_boundary
blocked
```

Requirements:

- phase-level and date/fold/child-level progress;
- deterministic stage identities and reuse;
- persisted checkpoints;
- safe cancellation and resume;
- sanitized errors;
- explicit retry policy;
- worker and memory controls;
- runtime/storage estimate before launch;
- immutable publication only after final validation;
- no partial artifact becomes a catalog-ready result.

## 9.7 Strategy-quality gate

Before any prop-contract ranking, each child must be evaluated on the underlying executed strategy.

At minimum support:

```text
cost-adjusted expectancy in R
realized payoff ratio
profit factor
maximum drawdown in R
time under water
trade frequency
independent trading-day count
executed trade count
session stability
time-block stability
setup/day block uncertainty
concentration warnings
```

Gate thresholds must be explicit search-charter fields.

The system should support objective templates but must display and persist the resolved thresholds.

## 9.8 Planned versus realized edge measurements

Persist both:

```text
planned RRR
realized average winner R
realized average loser R
realized payoff ratio
gross expectancy R
cost R
net expectancy R
slippage/commission drag
```

Do not represent planned 1R targets as proof of a realized 1.0 payoff ratio.

## 9.9 Prop-firm contract model

Define a versioned contract such as:

```text
PropFirmContractSpec
```

Required metadata:

```text
firm
account type
account size label
contract version
effective date
source status
source references
verification status
currency
timezone
trading-day boundary
```

Required rule support, where applicable:

```text
evaluation versus funded phase
starting balance
profit target
drawdown type
drawdown amount
trailing behavior
high-water-mark behavior
daily loss rule
unrealized-P&L treatment
contract limits
scaling rules
consistency rules
minimum trading days
winning-day requirements
payout waiting period
payout eligibility
payout cap
payout split
withdrawal effect on threshold
post-payout buffer behavior
activation fees
recurring fees
replacement/reset fees
account-expiration rules
breach reasons
```

A stale or unverified contract must be visible and optionally blocked from launch.

Firm internal business models are out of scope. The simulator must model the enforceable contract.

## 9.10 Prop account state machine

Define exact state and event contracts such as:

```text
PropAccountState
PropAccountEvent
PropPayoutEvent
PropBreachEvent
PropReplacementEvent
```

State must include:

```text
phase
balance
equity
high-water mark
current drawdown threshold
daily realized P&L
daily unrealized P&L where required
available drawdown buffer
contract allowance
winning-day count
consistency state
payout eligibility
payout amount available
post-payout threshold
fees paid
breach status
breach reason
account age
```

Every state transition must be deterministic and auditable.

## 9.11 Risk-policy model

Define a separate contract such as:

```text
PropRiskPolicySpec
```

Support named policies such as:

```text
fixed dollar risk
percentage of starting drawdown buffer
percentage of current drawdown buffer
fixed NQ contracts
fixed MNQ contracts
NQ/MNQ adaptive
daily shutdown
risk reduction after payout
risk reduction near drawdown threshold
custom contract count
```

Required controls may include:

```text
initial risk per trade
maximum current-buffer usage
minimum remaining buffer
daily stop
maximum contracts
instrument choice
post-loss risk adjustment
post-payout de-risking
recovery conditions
skip-if-minimum-contract-exceeds-budget
```

Strategy profile identity and risk-policy identity must remain separate.

## 9.12 Exact historical prop replay

For each qualifying child, firm contract, and risk policy:

1. Feed the exact chronological executed-trade stream into the prop state machine.
2. Apply firm rules in the correct event order.
3. Record balance, threshold, fees, payout eligibility, payout, breach, and replacement events.
4. Link every account event back to exact strategy trade and setup IDs.
5. Persist a complete account timeline.
6. Never infer independent market paths for copied accounts.

## 9.13 Correlated portfolio replay

When the same strategy is copied across multiple firms or accounts:

- use one common chronological market/trade path;
- apply different account states and contracts;
- do not bootstrap each copied account independently;
- report contract-state diversification separately from strategy diversification.

## 9.14 Simulation modes

Support three distinct modes.

### Exact historical

Question:

> What happened on this exact chronological path?

### Trading-day block bootstrap

Resample trading days or contiguous day blocks, preserving within-day and serial structure.

Do not independently shuffle trades.

### Stress scenarios

At minimum support frozen scenario definitions for:

```text
win-rate reduction
average-winner compression
average-loss expansion
higher transaction costs
missed winning fills
clustered losing days
lower opportunity frequency
session concentration
edge-decay regime
```

Each simulation must pin:

```text
seed
block policy
number of paths
stress parameters
source child trade-stream hash
firm contract
risk policy
```

## 9.15 Prop metrics

At minimum calculate:

### Evaluation fitness

```text
probability of passing before breach
median trading days to pass
probability of passing within 10/20/30 days
expected attempts before pass
expected fees before pass
breach-reason distribution
```

### Funded-account fitness

```text
probability of first payout before breach
probability of 2/3/5 payouts before breach
median days to first payout
median days between payouts
30/60/90-day survival
expected account lifetime
payout drought distribution
expected payouts before breach
expected net lifetime value
```

### Cash extraction

```text
gross withdrawals
payout split
evaluation fees
activation fees
recurring fees
replacement fees
trading costs
net cash received
fees per payout dollar
```

### Portfolio fitness

```text
probability of at least one payout within 30/60/90 days
probability of at least one payout in each rolling 30-day period
expected total net payout
10th/50th/90th percentile total payout
maximum portfolio payout drought
probability all accounts breach
expected account replacements
payout concentration by firm
```

## 9.16 Payout reliability vector

Do not compress payout consistency into one average.

At minimum show:

```text
P(first payout within 30 days)
P(first payout within 60 days)
P(3 payouts before breach)
P(at least one payout in each rolling 30-day period)
median days between payouts
90th-percentile payout drought
expected 90-day net payout
10th-percentile 90-day net payout
90-day breach probability
expected replacement cost
```

## 9.17 Feasibility gates and ranking

Use:

```text
hard feasibility constraints
then Pareto frontier
then explicit lexicographic tie-breaks
```

Do not use one hidden weighted score.

A representative ranking policy may prioritize:

1. highest worst-firm payout reliability;
2. highest 10th-percentile net payout;
3. lowest breach probability;
4. shortest payout drought;
5. lowest fee drag;
6. widest stable parameter region.

The exact resolved policy must enter search identity.

## 9.18 Pareto frontier

A configuration is dominated when another configuration is no worse across all selected objectives and strictly better in at least one.

The result must retain:

```text
all feasible points
all frontier points
dominance relationships
selected robust representative
highest expected-payout point
highest reliability point
lowest breach point
per-firm champion
```

Do not label one point simply `best`.

## 9.19 Parameter-stability and robust-basin analysis

For every searchable parameter:

```text
neighboring-config degradation
top-config frequency across outer folds
firm-specific sensitivity
interaction with other axes
plateau width
knife-edge warning
```

A robust representative should be selected from the interior of a stable region where practical.

## 9.20 Nested development evaluation

The planner must specify a leakage-safe research design.

Preferred structure:

```text
outer walk-forward fold
    inner development search
    frozen child selected
    one outer-fold evaluation
```

Report:

```text
selected config per outer fold
parameter-region recurrence
outer-fold strategy metrics
outer-fold prop metrics
cross-fold stability
```

The sealed range remains untouched until a later separately approved one-shot evaluation.

## 9.21 Failure attribution

Every rejected child must have one or more structured reasons, for example:

```text
replay failed
invariant failed
insufficient independent days
insufficient executed trades
negative cost-adjusted expectancy
excessive drawdown
challenge pass but funded survival failed
excess breach probability
fees consumed payout
failed one firm
failed robustness stress
knife-edge parameter behavior
unverified prop contract
```

These reasons must power both reports and UI filters.

## 9.22 Deterministic insight engine

Create exact rule-based insights such as:

```text
This configuration produced 11 additional trades because stale parent setups expired earlier.

Trade count increased 21%, but net expectancy declined from +0.14R to +0.06R.

Topstep payout reliability improved, while Apex breach probability increased because of its trailing-threshold behavior.

Results remained stable for timeout values from 45 to 75 bars.

58% of expected payouts came from Asia-session trades on nine trading days.
```

Every insight must contain exact evidence references.

Optional AI summaries may be layered on top later, but the deterministic insight output is authoritative.

## 9.23 Immutable persistence

Persist separately:

```text
search charter
child profile config
child replay result
child funnel/audit/replay references
prop contract
risk policy
historical account replay
simulation result
frontier result
insight result
final search result
```

Requirements:

- full SHA-256 identities;
- atomic saves;
- overwrite refusal;
- verified reload;
- duplicate reuse;
- exact source references;
- separate mutable catalog only for display names, notes, and user annotations.

## 9.24 Data access and safety

Current development boundary:

```text
latest authorized source instant: 2026-06-10T21:00:00Z
June 11: protected
June 12 onward: sealed
```

Requirements:

- permitted dates enumerated before path construction;
- no broad parent-directory listing;
- no protected/sealed `exists`, `stat`, metadata, open, or read;
- no sealed controls in the UI;
- event-derived access counters;
- all protected/sealed counters must remain zero;
- label and account simulation cannot consume unauthorized future bars.

## 9.25 Existing experiment integration

The fixed M0–M3 lane remains separate.

Later, a model-gated strategy may become an authorized child family only after:

- predictive skill beats its reference;
- feature/model protocol is frozen;
- threshold is frozen;
- sequential model-gated replay exists;
- the model is not selected adaptively within the same FSM search.

The initial robust-config search should be deterministic FSM plus prop realization only.

---

# 10. Frontend Requirements

## 10.1 Top-level information architecture

Retain the IFVG Lab top-level tabs:

```text
Experiments
Replay / Verifier
Data & Audit
```

Inside **Experiments**, add:

```text
New Study
Active Runs
Results
History
```

## 10.2 New Study modes

Offer five modes:

```text
Single Configuration
FSM Configuration Search
Prop Benchmark
Universal Prop Search
Full Pipeline Run
```

Definitions:

### Single Configuration

Evaluate one exact frozen profile.

### FSM Configuration Search

Search a small approved set of strategy configurations.

### Prop Benchmark

Apply prop contracts and risk policies to one frozen strategy stream.

### Universal Prop Search

Search one strategy configuration across multiple firms with firm-specific risk policies.

### Full Pipeline Run

Launch the standardized post-implementation pipeline over either:

```text
Verification Fixture — maximum five authorized trading days
Full Authorized Development Data — explicit operator action
```

The page must make the distinction prominent. The verification scope is for functional proof only;
the full authorized scope is the user-operated research run.

## 10.3 Objective-first wizard

The user should first choose the research question:

```text
Compare one configuration with the baseline
Find a robust FSM configuration
Test repeat-payout feasibility
Find one strategy configuration across multiple firms
```

Then use these wizard steps:

```text
1. Objective
2. Baseline
3. Strategy Search Space
4. Prop Contracts
5. Risk Policies
6. Benchmarks
7. Validation
8. Review & Launch
```

## 10.4 Objective templates

Provide templates:

```text
Payout Reliability
Maximum Expected Payout
Low Breach / Long Account Life
Balanced Prop Performance
Strategy Quality Only
Custom
```

Each template must expose:

```text
primary objective
hard constraints
tie-breaks
```

Do not hide resolved thresholds.

## 10.5 Baseline card

Show a human-readable baseline:

```text
Profile
Entry thesis
Direction
Target/label family
FSM concurrency
Development date range
Status
```

Put technical identity behind:

```text
View Technical Identity
Copy Profile Hash
Copy Artifact IDs
```

## 10.6 Search-space editor

Organize parameter cards by market meaning:

```text
Staleness
Parent Handling
HTF Selection
Causality and Locality
Entry Timing
Session Policy
Risk Admissibility
```

Each card must show:

```text
human label
technical key
baseline value
search values
classification
market meaning
existing evidence
status
artifact/profile impact
```

Use visible status labels:

```text
Locked Invariant
Search Axis
Measured Only
Blocked
Experimental
```

The user must not be able to search locked or blocked fields.

## 10.7 Combination preview

Before launch show:

```text
unique strategy profiles
full sequential replays
firm/risk combinations
historical prop replays
bootstrap/stress simulations
estimated runtime
estimated storage
```

Clearly distinguish expensive strategy replays from cheaper downstream prop simulations.

## 10.8 Prop contract cards

Each selected firm/account card should show:

```text
firm
account type
account size label
contract version
effective date
verification status
drawdown rule
daily rule
contracts
payout rules
fees
post-payout behavior
```

Stale or unverified cards should be warned or blocked according to policy.

## 10.9 Risk-policy editor

Support templates:

```text
Fixed Dollar
Percent of Starting Buffer
Percent of Current Buffer
NQ/MNQ Adaptive
Custom Contract Count
```

Display the hierarchy:

```text
Universal Strategy Profile
    ├── Firm A Risk Policy
    ├── Firm B Risk Policy
    └── Firm C Risk Policy
```

## 10.10 Benchmark editor

Separate:

```text
Underlying Strategy Gate
Prop Feasibility Gate
Robustness Gate
```

The user must see why a configuration is rejected and which gates were not run because an earlier gate failed.

## 10.11 Validation screen

Display:

```text
run scope: verification_5d | full_authorized_development
authorized development dates
real-date count
warmup
protected buffer
sealed boundary
search algorithm
outer-fold protocol
block-bootstrap protocol
stress paths
seed
estimated runtime
estimated storage
worker limit
```

Protected and sealed values are read-only.

For `verification_5d`:

```text
real-date count <= 5
warmup + evidence <= 5
verification-only badge is mandatory
research interpretation and catalog activation are prohibited
```

For `full_authorized_development`:

- the user must explicitly select and confirm the scope;
- the UI must show every authorized date or a downloadable resolved allowlist;
- no full run starts during tests or application startup;
- launch requires a frozen pipeline specification.

## 10.12 Freeze and launch

The final review page must show the complete search charter and estimated work.

Launch button:

```text
Freeze Search Charter and Launch
```

After launch, the study is immutable.

## 10.12A Full Pipeline Run UI

The Full Pipeline Run surface must provide a standardized operator workflow.

### Configure

The user selects:

```text
run scope
baseline/profile or parent search
authorized date range
feature bundle
label policy
fold protocol
model protocol
cost policy
prop contracts
risk policies
simulation protocol
worker/resource limits
```

### Preview

Before launch show:

```text
resolved dates and count
planned stages
strategy child count
feature-view count
fold/model count
prop simulation count
estimated runtime by phase
estimated storage
reuse opportunities
new artifacts expected
```

### Launch

Use an explicit action:

```text
Freeze Pipeline Specification and Launch
```

For full scope, require a second confirmation that states:

```text
This will run the full authorized development pipeline.
It is not an implementation verification run.
```

### Monitor

Render:

```text
overall progress
current phase
current date/child/fold
elapsed time
estimated remaining time
worker utilization
checkpoint status
reused stages
warnings
safe-cancel control
```

### Resume and retry

The user can:

```text
resume from verified checkpoint
retry a failed stage
clone a failed run with a changed resource policy
open logs and sanitized failure evidence
```

Changing any research-bearing field creates a new pipeline identity. Retrying an operational failure
with the same frozen identity reuses verified completed stages.

### Publish

The result remains `prepared_not_published` until all final identity, access, schema, model,
simulation, and report gates pass. Publication and catalog activation are explicit final actions.

The implementation test suite must exercise this entire UI and job path only with the
`verification_5d` fixture. It must not launch the full authorized pipeline.

## 10.13 Active Runs page

Show parent progress:

```text
profiles generated
replays completed
strategy-gate passes
prop-feasible configs
robust finalists
```

Use a clickable funnel:

```text
Generated
Replay Valid
Strategy Pass
Prop Feasible
Robust
```

Show a child-status table:

| Config | Replay | Strategy Gate | Prop Simulation | Robustness | Status |
|---|---|---|---|---|---|

Every failure needs a human-readable explanation.

## 10.14 Results overview

The first results view must answer:

1. Did any configuration pass?
2. Which is the robust representative?
3. Which maximizes payout?
4. Which maximizes reliability?
5. Which minimizes breach?
6. Why do they differ?

Use summary cards:

```text
Robust Representative
Highest Expected Payout
Highest Payout Reliability
Lowest Breach Risk
```

If none pass:

```text
No configuration passed all benchmarks.
```

Show dominant failure reasons.

## 10.15 Payout-reliability frontier

Primary chart:

```text
x-axis: expected 90-day net payout
y-axis: probability of at least one payout per 30 days
color: 90-day breach probability
size: expected account lifetime
```

Clicking a point must update the rest of the page.

## 10.16 Parameter-sensitivity heatmap

Support configurable axes and metric, for example:

```text
rows: parent-retest timeout
columns: parent-fill policy
color: P(3 payouts before breach)
```

Allow metrics such as:

```text
net E[R]
trade count
maximum drawdown
expected payout
breach probability
payout reliability
```

Distinguish:

```text
stable plateau
knife-edge point
failed region
insufficient-data cell
blocked cell
```

## 10.17 Firm compatibility matrix

Rows are strategy configs. Columns are firms.

Metric toggle:

```text
P(3 payouts before breach)
expected payout
breach probability
first-payout probability
```

This should make universal versus firm-specific configurations obvious.

## 10.18 Account survival curves

For a selected config, plot account-survival probability over trading days, one line per firm.

## 10.19 Payout distributions

Show:

```text
30-day
60-day
90-day
lifetime
```

Always display:

```text
mean
median
10th percentile
90th percentile
```

The lower tail must be visually prominent.

## 10.20 Configuration explorer

Provide column presets:

### Strategy

```text
trade count
net E[R]
realized payoff ratio
profit factor
max DD R
trade frequency
setup occupancy
```

### Prop

```text
firm
risk policy
first payout
3 payouts
90-day survival
expected payout
Q10 payout
fees
replacement cost
```

### Robustness

```text
outer folds passed
stress tests passed
neighbor stability
worst-firm result
concentration warning
```

Sticky columns:

```text
rank
config name
status
changed parameters
```

## 10.21 Human-readable configuration names

Generate names from the baseline diff:

```text
60-Bar Parent Timeout
Same-Session Parent Expiry
60-Bar Timeout + Close-Aware S4 Fill
```

Show only changed fields first.

## 10.22 Baseline comparison

Provide four panels:

```text
Parameter Diff
Funnel Delta
Strategy Delta
Prop Delta
```

Example funnel deltas:

```text
activations
locks
opposing selections
inversions
executed trades
setup-slot occupancy
terminal reasons
```

Clicking any delta must list the exact affected setups.

## 10.23 Deterministic insight panel

Insight categories:

```text
What Changed
Edge Effect
Prop Effect
Robustness
Concentration Warning
Evidence Quality
Recommended Inspection
```

Each insight must link to supporting evidence.

## 10.24 Exact drill-down

From any result, support:

```text
Open Changed Setups
Open Additional Trades
Open Removed Trades
Open Breach-Causing Path
Open Payout Event
```

Use the existing exact Replay / Verifier lane.

No setup/time fallback is allowed.

## 10.25 Account timeline

For exact historical replay, show:

```text
balance
trailing threshold
daily-loss threshold
payout eligibility
payouts
fees
breaches
replacement events
```

Clicking an account event must open the linked strategy trade or setup.

## 10.26 Progressive disclosure

### Summary

Plain-language cards and pass/fail.

### Analyst

Charts, distributions, sensitivity, and comparisons.

### Audit

Hashes, manifests, contract versions, child IDs, folds, seeds, and raw reports.

## 10.27 Status vocabulary

Use:

```text
Draft
Frozen
Queued
Running
Replay Failed
Strategy Rejected
Prop Rejected
Robust Finalist
Selected Representative
Superseded
Blocked
```

Avoid:

```text
Best
Winner
Production Ready
```

unless a separately defined policy explicitly permits that wording.

## 10.28 UX safeguards

The UI must clearly distinguish:

```text
Analysis Filter
Requires New Sequential Replay
```

It must also distinguish:

```text
candidate research
actual executed strategy
prop historical replay
bootstrap simulation
stress simulation
```

Gross, costed, and net results must be labeled separately.

Development-only results must carry a persistent badge.

## 10.29 Empty, blocked, and failure states

The UI must intentionally render:

```text
no configurations pass
no verified firm contract
blocked search axis
insufficient sample
child replay failed
prop simulation not run because strategy gate failed
no model result
artifact unavailable
protected-range refusal
```

No raw traceback or local filesystem path may be exposed.

## 10.30 Accessibility and responsiveness

The implementation plan must include:

```text
keyboard navigation
screen-reader labels
high-contrast status semantics
responsive tables
desktop/tablet/mobile layouts
large-chart fallback
empty/failure state QA
```

Required viewport QA:

```text
1440×900
1024×768
768×1024
390×844
```

---

# 11. Reporting Requirements

Every parent search should produce:

```text
search charter report
child inventory
strategy-gate report
prop-contract report
risk-policy report
historical prop replay report
bootstrap report
stress report
feasible-set report
Pareto-frontier report
parameter-sensitivity report
failure-attribution report
deterministic-insight report
access-safety report
identity/reconciliation report
final search summary
```

Every child should retain exact links to:

```text
strategy profile
v2 execution artifact
FSM audit artifact
replay-chart artifact
prop simulations
account timelines
setup/trade drill-down
```

---

# 12. Performance and Capacity Requirements

The planning agent must define concrete budgets after codebase inspection.

At minimum benchmark:

```text
child profile enumeration
single sequential replay
parallel replay scheduling
prop simulation per firm/risk policy
10,000-path block bootstrap
stress simulation
frontier computation
dashboard initial load
results filtering
chart rendering
exact drill-down load
artifact storage growth
```

Required safeguards:

- bounded worker count;
- bounded memory;
- no duplicate replay of identical profiles;
- incremental persistence;
- resumable parent jobs;
- cancellation at safe boundaries;
- deterministic reuse;
- UI pagination/virtualization for large child counts;
- no full artifact load into Streamlit memory when indexed reads are possible.

The first version should impose a hard maximum search size.

All implementation performance and capacity gates must use the fixed verification fixture of no
more than five authorized trading days. The plan must not require a full-data benchmark to close
implementation.

The released UI must separately estimate and record full-pipeline timing from actual operator runs.
Those operator measurements may refine future defaults, but they are not substituted into the
five-day implementation-verification identity.

---

# 13. Recommended Implementation Milestones

The planning agent may adjust file-level sequencing after repository inspection, but the product should generally be delivered in this order.

## Milestone 0 — Codebase discovery and contract plan

- Map current experiment, artifact, job, reporting, catalog, and UI architecture.
- Identify reusable immutable-store and job-runner abstractions.
- Produce exact schema and module plan.
- Resolve blockers and owner decisions.
- No code changes.

## Milestone 1 — Search contracts, feature expansion, and pipeline identities

- Search mode and capability registry.
- Search-axis registry.
- Universal experiment-dimension registry.
- MBP-1-only feature-block/source contract.
- Search charter identity.
- Child identity.
- VerificationRunSpec and five-day data policy.
- Full-pipeline run identity and stage contract.
- Immutable parent/child/pipeline stores.
- Synthetic tests.

## Milestone 2 — Sequential child replay orchestrator

- Enumerate small exhaustive spaces.
- Run/reuse full Strategy-Core child replays.
- Persist child funnel, strategy, audit, and replay references.
- Strategy-quality gates.
- Progress and failure reporting.

## Milestone 3 — Prop contract and account-state engine

- Versioned prop contracts.
- Risk policies.
- Exact historical account replay.
- Payout/breach/fee/account-state events.
- Exact trade/setup linkage.

## Milestone 4 — Simulation and robustness

- Trading-day block bootstrap.
- Stress scenarios.
- Feasible set.
- Pareto frontier.
- Parameter sensitivity.
- Robust representative.

## Milestone 5 — New Study, Full Pipeline Run, and Active Runs UI

- Guided research wizard.
- Full Pipeline Run mode.
- Verification-versus-full scope selector.
- Search-space and pipeline-phase preview.
- Freeze and launch.
- Parent, child, phase, date, and fold progress.
- Checkpoint/resume/safe-cancel controls.
- Failure reasons.
- Immutable history.

## Milestone 6 — Results and insight UI

- Summary cards.
- Frontier.
- Heatmap.
- Firm matrix.
- Survival curves.
- Payout distributions.
- Configuration explorer.
- Baseline comparison.
- Deterministic insights.

## Milestone 7 — Deep audit integration

- Funnel-delta setup lists.
- Account timeline.
- Replay links.
- Breach/payout event drill-down.
- Exact audit evidence.

## Milestone 8 — Five-day hardening and release verification

- Full unit and synthetic test suites.
- One fixed real-data verification allowlist of at most five trading days.
- Five-day end-to-end mini-pipeline only.
- Five-day performance/capacity gates.
- AppTest.
- Browser QA.
- Accessibility.
- Protected/sealed access proof.
- Immutable repeat/reuse proof.
- Proof that the full authorized pipeline was not invoked.
- Documentation and operator handoff.

## Milestone 9 — Post-implementation full authorized pipeline run

This is an explicit operator action after implementation acceptance, not an implementation gate.

- Launch from the Quant-Lab UI.
- Use the standardized frozen full-pipeline specification.
- Run or reuse every required stage over the authorized development dataset.
- Monitor/checkpoint/resume through the UI.
- Validate and publish immutable outputs.
- Record actual phase runtimes and storage.
- Do not access June 11 or sealed data.

---

# 14. Initial Search Algorithm

The first authorized implementation should support:

```text
deterministic exhaustive enumeration
```

Reasons:

- FSM variables are often categorical;
- response surfaces can be discontinuous;
- setup occupancy causes abrupt trade-stream changes;
- exhaustive output is auditable;
- small search spaces are sufficient for the first exploratory studies.

A later separately approved version may add:

```text
Sobol or stratified sampling
successive halving
full simulation for finalists
```

The first version must not use an opaque black-box optimizer as the only search method.

---

# 15. Activation and Roadmap Gate

The backend and UI may be implemented using synthetic fixtures, current immutable artifacts, and
one fixed real-data verification allowlist containing no more than five authorized trading days.

No implementation-verification step may run the full authorized development pipeline.

After implementation acceptance, the user may launch the standardized full authorized pipeline
explicitly through the UI. The first broad real-data FSM/prop search must still remain blocked
until:

1. setup-level visual review is complete;
2. stale-parent policy is resolved;
3. parent-fill policy is frozen for the search;
4. authorized strategy search axes and values are owner-approved;
5. all profile identities are named and hash-pinned;
6. prop contracts are verified and versioned;
7. benchmark thresholds are owner-approved;
8. the current development boundary remains enforced.

The sealed range remains untouched.

---

# 16. Acceptance Criteria

## 16.1 Plan quality

The codebase-specific implementation plan must:

- identify exact files, classes, functions, schemas, services, scripts, tests, and UI modules;
- state what is reused versus newly created;
- define migration and compatibility;
- identify owner decisions;
- define performance budgets;
- define a complete test matrix;
- preserve existing M0–M3 behavior.

## 16.2 Research integrity

A completed implementation must prove:

1. Each strategy configuration receives a full sequential replay.
2. No strategy execution is derived from post-hoc candidate filtering.
3. One-active-setup and one-active-trade invariants remain enforced.
4. Strategy and risk-policy identities remain separate.
5. Underlying strategy gates run before prop ranking.
6. Copied accounts share one common trade path.
7. Block bootstrap uses days/contiguous blocks, not independent trade shuffling.
8. Every search axis is registered and authorized.
9. Every child and simulation is immutable and exact-linked.
10. No automatic promotion occurs.
11. No sealed data is accessed.

## 16.3 Backend functional acceptance

Using the fixed verification fixture of at most five authorized trading days, the user can freeze one parent study containing, for example:

```text
2 parent-timeout values
× 2 parent-fill policies
= 4 full strategy replays
```

and:

```text
4 strategy configs
× 2 firms
× 2 risk policies
= 16 prop simulations
```

The system must:

- enumerate exactly four strategy children;
- run/reuse exactly four sequential replays;
- run exactly sixteen prop simulations after strategy gates;
- skip prop simulation for strategy-rejected children;
- persist all identities;
- resume after interruption;
- return identical results on repeat;
- refuse overwrite;
- show exact failure reasons.

## 16.4 Prop engine acceptance

For deterministic fixtures, verify:

- trailing drawdown;
- static drawdown;
- daily loss;
- payout eligibility;
- payout cap;
- post-payout threshold behavior;
- fees;
- breach;
- replacement;
- minimum days;
- consistency rule;
- contract limit;
- exact event ordering.

## 16.5 Simulation acceptance

Verify:

- deterministic seed identity;
- exact historical replay;
- day-block bootstrap;
- stress scenarios;
- common correlated path across accounts;
- no duplicate OOS path;
- stable aggregation;
- lower-tail metrics;
- frontier correctness;
- dominance correctness;
- neighbor-stability calculation.

## 16.6 UI acceptance

A user can:

1. Open IFVG Lab → Experiments → New Study.
2. Select `Universal Prop Search`.
3. Choose an authorized baseline.
4. See locked, searchable, measured, blocked, and experimental parameters.
5. Select a small authorized search space.
6. Select versioned firm contracts.
7. Define firm-specific risk policies.
8. Define explicit strategy, prop, and robustness gates.
9. Review exact combination count and estimated runtime.
10. Freeze and launch the charter.
11. Monitor parent and child progress.
12. See why children failed.
13. Open the results overview.
14. View the frontier, heatmap, firm matrix, survival curves, and payout distribution.
15. Compare one config with the baseline.
16. Click a funnel delta and open exact setups.
17. Click a breach or payout event and open the linked trade.
18. Clone a frozen study without mutating the original.
19. Reload all results from immutable storage.
20. See a clear `No configuration passed` state when appropriate.
21. Open `Full Pipeline Run` and see `verification_5d` and
    `full_authorized_development` as distinct scopes.
22. Launch, monitor, safely cancel, resume, and reload a complete
    `verification_5d` pipeline.
23. Preview the exact phases, dates, work units, runtime, storage, and reuse for a
    `full_authorized_development` run.
24. After implementation acceptance, explicitly launch the full authorized pipeline from the UI
    without changing or bypassing its frozen specification.
25. Verify that tests, application startup, and catalog loading never launch the full pipeline
    automatically.

## 16.7 Insight acceptance

Every generated insight must:

- be reproducible;
- have a typed category;
- cite exact child/config IDs;
- cite exact metrics and deltas;
- link to evidence;
- avoid unsupported causal language;
- distinguish descriptive from counterfactual conclusions;
- disclose sample and concentration limitations.

## 16.8 Safety acceptance

- Every protected and sealed event counter is zero.
- No protected/sealed path is constructed or listed.
- No sealed UI control exists.
- No raw local path or secret appears in the UI.
- Blocked profiles cannot launch.
- Stale/unverified prop contracts are warned or blocked.
- Development results are never labeled validation or production evidence.

## 16.9 Immutability acceptance

- Same charter produces the same search ID.
- Same child config produces the same child ID.
- Same prop simulation produces the same simulation ID.
- Duplicate execution verifies and reuses existing artifacts.
- Existing artifacts cannot be overwritten or deleted from the study UI.
- Display names and notes are stored separately from immutable results.

## 16.10 Performance acceptance

The implementation plan must define and the implementation must pass:

- replay throughput;
- prop-simulation throughput;
- bootstrap wall time;
- memory use;
- artifact size;
- dashboard load time;
- chart interaction time;
- large-search pagination;
- safe worker limits;
- repeated-run variability.

## 16.11 UI QA acceptance

Required:

- repository-native unit and integration tests;
- Streamlit AppTest;
- lint/type checks;
- HTTP health and root checks;
- 1440×900;
- 1024×768;
- 768×1024;
- 390×844;
- keyboard-only navigation;
- empty/blocked/failure states;
- screenshot evidence;
- long-ID and wide-table behavior.

---


## 16.12 Universal delta-architecture acceptance

The implementation must pass a matrix that exercises at least one example of every computation
class:

```text
feature-only
cohort descriptive
cohort-specific model
strategy counterfactual
label counterfactual
model protocol
model-gated execution
execution/cost
risk
prop contract
payout policy
portfolio
stress
data lineage
```

For each example, verify:

```text
correct identity
correct required computation
correct frozen dimensions
correct delta metric families
correct compatibility status
correct artifact reuse or regeneration
correct dashboard label
```

## 16.13 Expansion-feature acceptance

The schema must be able to register, display, and validate:

```text
current baseline
structure
displacement
liquidity
session context
volatility context
planned MBP-1
explicit MBP-10/deeper-book exclusion
planned regime
planned key-level context
planned execution-liquidity context
```

A planned block may remain unavailable, but it must have:

```text
stable block identity
source requirements
point-in-time contract
dependencies
coverage requirements
capability reason
expected computation path
```

No frontend or backend redesign should be required when a planned block becomes available.

## 16.14 Five-day verification-budget acceptance

The implementation must prove:

1. One canonical verification allowlist contains no more than five authorized trading days.
2. Warmup plus evidence dates do not exceed five.
3. Every real-data integration, performance, AppTest-backed job, and end-to-end verification uses
   that same allowlist.
4. No verification test constructs a path for a real date outside the allowlist.
5. Synthetic fixtures and precomputed seeds are used where additional state coverage is required.
6. Verification-only model fits and artifacts are clearly nonresearch and isolated from normal
   catalogs.
7. The verification mini-pipeline exercises every stage needed for orchestration correctness.
8. No full-development replay, feature build, fold build, training run, prop search, or robustness
   simulation executes during implementation verification.
9. The verification report states `full_pipeline_not_run = true`.
10. All protected and sealed counters remain zero.

## 16.15 Standardized full-pipeline UI acceptance

After implementation acceptance, the released UI must allow the user to:

1. Select `Full Authorized Development` explicitly.
2. Inspect and export the exact resolved date allowlist.
3. Select the profile/search, feature bundle, label, fold, model, cost, prop, risk, and simulation
   policies.
4. Preview stages, child/fold/simulation counts, runtime, storage, and reuse.
5. Freeze an immutable pipeline specification.
6. Launch the background job manually.
7. Observe phase/date/child/fold progress.
8. Safely cancel at a boundary.
9. Resume from verified checkpoints.
10. Retry an operational failure without changing research identity.
11. Reuse already-verified stages.
12. Receive sanitized errors and complete reports.
13. Publish only after all required gates pass.
14. Reload the complete result from immutable storage.

The test suite validates this workflow with `verification_5d`; the actual full authorized run is a
post-implementation operator action and is not required to close the code implementation.


## 16.16 Machine-learning and regime-model acceptance

The implementation plan must map every requirement in §7B to:

```text
contract/schema
fit service
artifact store
fold integration
report
dashboard component
test
failure state
```

At minimum, the implementation plan must specify:

1. exact supervised baseline protocols;
2. exact regime algorithm registry;
3. direct spectral-clustering restrictions;
4. a production-compatible spectral approximation;
5. OOS assignment semantics;
6. fold-local preprocessing;
7. cluster-label reporting alignment;
8. stability and occupancy gates;
9. regime-feature/model/profile delta semantics;
10. calibration and decision-policy boundaries;
11. five-day ML verification fixtures;
12. standardized full-pipeline integration.

The implementation may mark spectral and regime computation as a later milestone, but the schemas,
identities, capability states, UI states, and tests for unavailable/planned status must be designed
in V1.


# 17. Open Decisions for the Implementation Planner to Classify

The planner must determine which are blocking and which can be deferred.

1. Exact first authorized FSM search axes.
2. Exact values for parent-retest staleness.
3. Whether parent-fill handling is included in the first search.
4. Whether initial prop contracts cover evaluations, funded accounts, or both.
5. Which firms/account sizes enter v1.
6. Official contract-source ingestion and verification workflow.
7. Exact underlying strategy-gate thresholds.
8. Exact payout-reliability benchmark thresholds.
9. Exact number of bootstrap paths.
10. Exact block-length policy.
11. Whether risk policies can use dynamic current-buffer sizing in v1.
12. Whether account replacement is automatic or a separate scenario.
13. Whether multiple accounts per firm are supported in v1.
14. Whether an optional user-defined weighted score is allowed as a secondary display.
15. Whether nested walk-forward is mandatory in v1 or introduced after the exploratory lane.
16. Search-size limit and worker limit.
17. First-version artifact retention policy.
18. Whether deterministic insight templates are sufficient or an optional AI explanation layer is planned later.
19. How current setup-level verifier artifacts are linked into child search results.
20. How current dirty/unpublished environments are handled before release.
21. Which fixed five-day-or-smaller real-data allowlist becomes the canonical implementation-verification fixture.
22. Which precomputed seeds or synthetic fixtures are required to cover long-history behavior without reading extra real dates.
23. Exact standardized stages and resource defaults for the UI-launched full authorized pipeline.
24. Whether full pipeline publication is one action or a separate verify-then-activate sequence.
25. Which regime algorithm is the first implemented baseline: KMeans only, KMeans + Gaussian mixture, or the full registered set.
26. Whether direct spectral clustering is implemented in V1 as training-only diagnostics or remains planned.
27. Whether `nystrom_kmeans_v1` is required in V1 or delivered in the first regime-expansion milestone.
28. Exact regime observation grain and assignment stage for the first implementation.
29. Fixed cluster counts or the small inner-train-only cluster-count candidate set.
30. Minimum cluster occupancy, assignment-confidence, and stability gates.
31. Whether a GAM baseline is implemented in V1 or remains optional.
32. Whether SHAP is included in V1 or only grouped permutation importance is exposed.

The implementation planner should recommend defaults where architecture and research discipline support them. Only genuine strategy or business decisions should be escalated.

---

# 18. Required Output From Codex Sol or Claude Fable

Produce one implementation-plan Markdown document containing:

1. Executive recommendation.
2. Current codebase architecture map.
3. Gap analysis against this brief.
4. Exact backend module/file changes.
5. Exact frontend module/file changes.
6. Data contracts and schemas.
7. Identity and immutability model.
8. Job/orchestration design.
9. Prop-account simulation design.
10. Search and robustness protocol.
11. Reporting and deterministic-insight design.
12. UI information architecture and component map.
13. Reuse versus rewrite decisions.
14. Migration and catalog strategy.
15. Access-safety design.
16. Performance and capacity plan.
17. Full test matrix.
18. Phased implementation sequence.
19. Acceptance-gate mapping.
20. Open owner decisions.
21. Risks and mitigations.
22. Exact expected files to change.
23. Universal study-cell identity and experiment-dimension registry.
24. Full delta taxonomy, computation-path matrix, and compatibility rules.
25. Feature-block and feature-bundle contracts, including the planned MBP-1-only expansion and explicit exclusion of MBP-10.
26. Main-effect, interaction, neighbor-stability, and exact population-delta design.
27. Five-day verification-data policy, mini-pipeline fixture design, and proof strategy.
28. Standardized UI-launched full-pipeline contract, stages, progress, checkpoint, and publication design.
29. Explicit confirmation that MBP-1 is the maximum supported order-flow depth and MBP-10 is excluded.
30. Supervised-model ladder and immutable model-protocol contracts.
31. Regime-model schemas, algorithm registry, and fold-local fit architecture.
32. Direct spectral-clustering restrictions and OOS-assignment design.
33. Nyström+KMeans spectral-approximation design and identity.
34. Regime stability, reporting-alignment, dashboard, and drift-monitoring design.
35. Explicit confirmation that no code was changed during planning.

Stop after the plan.
