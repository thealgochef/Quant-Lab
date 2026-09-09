# R6 Owner Audit (input to the R6.1 plan) — verbatim, received 2026-08-28

**Document type:** Input record. The owner's review of the R6 release evidence
(`..\R6\`). Answered point by point in
`R6.1_IMPLEMENTATION_PLAN.md` §2 (verification) and §6 (workstreams).

---

The phase R6 was completed and implemented, but theres some major findings after review.
Please create a Phase R6.1 to address the findings after you review this audit. Once you review
this audit line by line create a plan for implementation.

## Major findings

### 1. There is no real context-panel materializer or panel-native fold builder

DEV-R6-7 explicitly states that R6 does not ship:

- a completed 5m/15m context-panel materializer
- a panel-native fold builder

This is not a minor optional feature.

Your accepted development pair contains:

- 132 candidates total
- 125 post-warmup candidates

The default KMeans sample-adequacy floor shown by R6 is: 150 training rows for k=3

Therefore:

- candidate-stage grain: likely blocked
- decision-row grain: even smaller and blocked
- context-panel grain: potentially sufficient, but currently cannot be built from real data

The panel grain was added specifically to solve candidate-stage sparsity. The R6 plan expected
the actual context-panel schema and point-in-time panel-to-candidate assignment path to be available.

Required R6.1 work — add:

- completed 5m or 15m context-panel materializer
- panel source-artifact identity
- panel as-of policy
- panel-native folds
- panel-to-candidate assignment artifact
- real sample-adequacy preview

Without this, the real KMeans lane is likely to render only a blocked-insufficient-sample state.

### 2. Regime fitting is not integrated into the 16-stage operator pipeline

This is visible from the evidence package:

- search/pipeline.py was not modified in R6;
- the R6 smoke helper first runs the existing 16-stage synthetic pipeline;
- it then separately calls a helper to fit and persist regime artifacts;
- those artifact IDs are manually supplied to the UI.

The operator pipeline already defines: S05 materialize feature views; S06 validate feature
coverage; S08 build folds; S09 train models; S10 predictions and diagnostics; S14 frontier and insights.

R6 evidence does not show a pipeline executor that automatically: resolves the regime protocol;
loads or materializes its observation view; obtains the fold set; fits KMeans per training fold;
produces OOS assignments; persists fits and assignments; builds a capability assessment; sends
exact artifact references into downstream reports.

Consequence: the Full Authorized Development pipeline cannot yet be expected to produce R6 regime
artifacts by itself. The panel can display manually supplied artifacts, but that is not the
end-to-end operator workflow.

Required R6.1 work — wire regime processing into the capability-scoped stage plan: S05/S06
observation view and coverage; S08 fold set; S09 KMeans fit and OOS assignment; S10 diagnostics,
alignment, capability assessment; S14 regime-aware stratified reporting when requested. A
strategy-only or prop-only plan should still skip these operations.

### 3. Owner-ratification evidence appears format-checked, not verified-loaded

The fix requires a 64-character hexadecimal owner_ratification_ref before a regime can reach
FEATURE_ELIGIBLE. The evidence shows persist_regime_promotion verifies: the capability assessment;
the previous promotion decision; the ladder transition. It does not demonstrate that it loads and
verifies an immutable OwnerDecisionEvidenceRef and confirms that the decision actually authorizes:
the resolved regime protocol; the observation grain and stage; k=3; the occupancy and stability
thresholds; the feature-eligibility transition. A syntactically valid 64-hex string is not owner
authorization.

Required correction — before feature eligibility can be persisted: verified-load the
owner-decision artifact; verify its hash and supersession state; verify its effective date; verify
the exact decision keys and values; verify it applies to this protocol/grain/k/gate set; bind the
verified evidence ID into the promotion decision. This is not currently an execution risk because
the UI stays descriptive-only and S11 is blocked. It is a governance blocker before real feature promotion.

### 4. "Stratification" does not yet include strategy, model, or prop results by regime

DEV-R6-4 defers: strategy performance by regime; model skill and calibration by regime; prop
metrics by regime; regime-stratified frontier views. The current stratification panel primarily
contains: assigned-row count; cluster share; assignment margin; OOS assignment timeline. Those are
useful assignment diagnostics. They do not answer the core research questions: Does the strategy
work only in one regime? Does a regime feature improve OOS Brier or calibration? Does MBP-1 help
only in certain regimes? Do payout or breach outcomes concentrate in one regime? The design
explicitly called for these comparison classes and descriptive regime uses.

Required R6.1 work — implement adapters for: cohort_descriptive (actual strategy metrics by
frozen OOS regime); feature_only (baseline vs baseline+regime on identical rows/labels/folds);
cohort_model (regime-specific models with pooled baseline retained); stratified_prop
(payout/breach/fees by frozen OOS regime); stratified_frontier (descriptive config behavior by
regime). A regime-specific executable strategy still requires a full sequential replay.

### 5. The complete repository gate is still red

The raw final output says: 2 failed, 1733 passed, 84 warnings, exit code 1. The two failures are
credibly caused by provider API keys being present, and both pass when those keys are cleared.
That diagnoses the problem. It does not make the release-wide command green. Before hardening:
clear POLYGON_API_KEY; clear DATABENTO_API_KEY; rerun the complete pytest suite; retain the full
all-green output.

### 6. Browser evidence is not bound to the final R6 commit

The R6 release commit is 179a2c9. The browser manifest records 7f018f57..., which is the R5B
parent. The manifest includes a source-tree digest and may therefore have captured the dirty R6
tree, but the evidence package does not prove that the digest equals the final committed R6
source. The smoke evidence was prepared before the final commit and then reused.

Required correction — either: rerun browser smoke from the final R6 correction commit; or include
a proof that the manifest source-tree digest matches that commit's scoped tree. The manifest
should also include SHA-256 and byte size for every screenshot and the server log.

## Scientific and interpretation concerns

### Reference-fold-only bootstrap stability

The package records 50 refits of the reference fold's training matrix. The UI presents the
resulting aligned AMI and cluster agreement as general regime stability. A stable first fold does
not prove later folds are stable.

Better policy — run bootstrap stability for every valid fold and report: per-fold aligned AMI;
per-fold cluster agreement; minimum / lower-quantile stability; fold coverage. If the current
design is retained, rename it `reference_fold_bootstrap_stability` and do not let it alone govern
protocol-wide promotion.

### Transition matrices on irregular candidate-stage observations

Candidate-stage assignments are not a regular time series. Two consecutive candidates may be
separated by minutes, hours, overnight boundaries, multiple trading days. Yet the current
transition matrix counts adjacent OOS candidate assignments after timestamp sorting. That is a
candidate-event transition matrix, not necessarily a market-regime transition matrix.

Better policy — for candidate and decision grains: record elapsed time; reset across sessions or
trading days; apply a maximum-gap rule; label it `candidate_event_transition_matrix`. Use ordinary
temporal persistence and transition semantics only on a regular 5m/15m panel.

### Source artifact provenance needs a verified-load proof on the real path

The synthetic smoke passes an in-memory fixture view ID as its source artifact reference. The
training-matrix hash protects the numerical fit, but the evidence does not prove that a real
upstream artifact is verified-loaded before its ID is accepted. The real stage executor must:
load the source artifact through its immutable store; verify manifest and files; obtain the
artifact ID from the loaded envelope; build the feature matrix from those verified bytes; bind
both source ID and matrix hash into the fit.

## Important carry-forward blockers

R6 does not resolve the major R5B real-data concern: raw venue sequence-number jumps must not
automatically be treated as missing MBP-1 data. The prior review concluded that the coverage
contract is not ready for real research until that is corrected.

The R5B model path also remains logistic-only. CatBoost is still refused for the MBP-1 bundle
path, so the system cannot yet answer whether order flow improves the nonlinear CatBoost model.

## R6.1 must close

- real context-panel materialization and folds
- pipeline-stage integration
- verified owner-evidence lookup
- regime performance stratification
- temporal/stability semantics
- all-green suite
- final-commit-bound browser evidence

## Then hardening must close

The final frontend contract still requires the four viewports, keyboard-only operation, populated
and blocked states, long IDs, wide tables, responsive behavior, and screenshot evidence.

## Then real verification

You still need: one owner-approved <=5-day allowlist; a compliant pre-existing profile-matching
seed; signed VerificationAuthorizationRef; real baseline, MBP-1, and regime control-flow
verification; zero protected/sealed access.

## Then the full operator run

Only that run will generate the actual immutable outputs from which you can judge: strategy
configurations; MBP-1 incremental value; KMeans regime behavior; model comparisons; prop-firm
realization.

## Prop-firm decision support

That additionally requires: first-party verified firm contracts; owner-approved
risk/withdrawal/replacement policies; owner-approved strategy and payout gates; real
account-state simulations.

## Live filtering or execution

Not part of V1: S11 remains blocked; no regime execution gate is authorized; MBP-1 is
research_only_offline; Trade-Lab serving is out of scope.
