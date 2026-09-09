# Implementation Kickoff — `ifvg_prop_robust_config_search_v1`

Read this file first:

```text
C:\Users\gonza\Documents\Claude-Quant-Lab\QL-FSM-PROP-SEARCH-DASHBOARD\FINAL-IMPLEMENTATION-PLAN-DOCS\README.md
```

Then read **every authoritative document named by that README**, in its stated reading order. At minimum, read:

```text
IMPLEMENTATION_PLAN.md
FRONTEND_UX_CONTRACT.md
FRONTEND_RETENTION_VERIFICATION.md
PHASED_DELIVERY.md
CONTRACTS_AND_SCHEMAS.md
TEST_MATRIX.md
OWNER_DECISIONS.md
DELTA_TAXONOMY.md
ML_REGIME_CONTRACT_PLAN.md
ARCHITECTURE_MAP.md
the current final consistency-audit document named by README.md
REVISION_CHANGELOG.md
```

`FRONTEND_UX_CONTRACT.md` is the complete normative UI/UX authority; do not treat the shorter frontend summary in `IMPLEMENTATION_PLAN.md` as a substitute.

The `FINAL-IMPLEMENTATION-PLAN-DOCS\` folder is the complete authoritative implementation package for `ifvg_prop_robust_config_search_v1`.

Treat that package as **read-only implementation authority**:

- Do not edit the plan documents during implementation.
- `FSM-PLAN-DOCUMENT.md` is superseded provenance and is never implementation authority.
- The sibling `original-plan-brief\` and `completed-revision-documents\` folders are input/history only. Do not use them to override the final package.
- When a summary in this prompt conflicts with the final package, the final package governs.
- When the final package conflicts with the current codebase on a factual implementation seam, stop and document the conflict rather than silently inventing a third design.

I approve the final authoritative package for **code authoring and bounded implementation verification**.

Implement the complete V1 delivery sequence:

```text
R1
R2
R3
R4
R5
R5B
R6 — V1 KMeans regime lane
hardening
```

Do **not** implement the post-V1 regime expansion in this task:

```text
MiniBatch KMeans
Gaussian mixture
direct spectral-clustering diagnostics
Nyström + KMeans
expanded post-V1 drift suite
```

Their contracts, registry entries, blocked/planned UI states, and fail-closed behavior must exist exactly as specified, but their numerical fit implementations remain post-V1.

Do not execute the Full Authorized Development operator run. Implement its UI and orchestration path, but leave the full run as a separate, explicit post-acceptance owner action.

Proceed release-by-release without routine approval pauses. Stop only for:

1. a genuine semantic or business owner decision not already covered by the package;
2. a factual codebase contradiction that invalidates a load-bearing plan assumption;
3. a gate failure that remains unresolved after real diagnosis and fix attempts;
4. a required Strategy-Core change, which is outside this V1 scope;
5. an overlap with pre-existing user-owned changes that cannot be safely isolated.

---

# 1. Preflight and repository protection

Before changing code:

1. Record, in `implementation-progress\R1\PRE_IMPLEMENTATION_BASELINE.md`:

   ```text
   current branch
   HEAD commit
   git status --short
   installed Strategy-Core location and commit/pin
   Python and relevant dependency versions
   existing test baseline
   existing lint baseline
   existing pip-check issues
   authoritative plan-package filenames
   ```

2. Create a dedicated branch:

   ```text
   feature/ifvg-prop-robust-config-search-v1
   ```

   Use an equivalent unique branch name only when that branch already exists.

3. Do not reset, stash, clean, discard, overwrite, or absorb pre-existing user-owned work.

4. If the Quant-Lab worktree is dirty before implementation:

   - preserve the exact baseline status;
   - stage and commit only files changed by this implementation;
   - do not include unrelated changes in any release commit;
   - stop if a pre-existing modification overlaps a planned file and cannot be safely separated.

5. Treat Strategy-Core and Trade-Lab as read-only. Record their state, but do not edit them.

6. Run the repository’s existing baseline checks before implementation and record exact commands, exit codes, pass/fail counts, and known pre-existing failures. Do not repair unrelated environment or dependency issues unless the final plan explicitly requires it.

7. Do not alter the authoritative plan package. Record any implementation deviation in the release evidence and completion report.

---

# 2. Release execution model

Implement in this order:

```text
R1 → R2 → R3 → R4 → R5 → R5B → R6 → hardening
```

Use read-only exploration/reviewer subagents freely. Avoid parallel writers on overlapping files. A single writer must own each file at a time.

For every release:

1. Read that release’s complete scope and gate from `PHASED_DELIVERY.md`.
2. Read every referenced contract and test requirement.
3. Write or update tests before, or in the same small change as, implementation.
4. Implement only that release’s authorized scope.
5. Run the release-specific tests.
6. Run all required repository-wide regression checks named by the plan.
7. Run:

   ```text
   git diff --check
   ```

8. Run a read-only adversarial review using independent reviewers whose task is to falsify the release against:

   ```text
   CONTRACTS_AND_SCHEMAS.md
   TEST_MATRIX.md
   PHASED_DELIVERY.md
   the protected/sealed access rules
   the existing M0–M3 and immutable-artifact compatibility boundaries
   ```

   Reviewers must produce findings only; they do not edit implementation files.

9. Fix every valid finding.
10. Re-run the affected tests and gate checks.
11. Save release evidence under `implementation-progress\<release>\`.
12. Commit the release on the feature branch with a clear release-scoped message.
13. Do not push and do not merge.

Each release has two separate statuses:

```text
implementation_status
acceptance_status
```

Never label a release `accepted` merely because its code is written.

---

# 3. R0 → R1 entry checks

Resolve these first:

```text
capture-driver mid-chain start from a cached, profile-matching seed
native-ID repeat-replay determinism
```

Use the documented Quant-Lab-only fallback when required.

Do not modify Strategy-Core.

Before real-data authorization exists, use:

```text
code inspection
synthetic characterization
test fixtures
existing already-authorized immutable evidence
```

to implement and validate the seams.

The final real proof remains part of the authorized R1 vertical slice.

---

# 4. R1 verification authorization — critical distinction

This implementation prompt **does not grant** the real five-day fixture authorization and must not be treated as a `VerificationAuthorizationRef`.

Do not fabricate, self-sign, or infer owner authorization.

Before any real source path is constructed, the real R1 vertical slice requires an owner-approved, immutable `VerificationAuthorizationRef` binding:

```text
the one canonical <=5-day allowlist
coverage-matrix artifact
matching baseline seed snapshot
verification-policy identity
pipeline/run identity
owner/approver evidence
```

Until that exists:

1. Implement the complete R1 real-slice path.
2. Build the coverage matrix only from already-authorized existing artifacts, without new raw-source discovery.
3. Produce the proposed allowlist and a draft authorization payload for owner review.
4. Run all synthetic, contract, store, identity, policy, UI, and fail-before-path tests.
5. Do **not** execute the real baseline vertical slice.
6. Mark:

   ```text
   R1 implementation_status = complete, when code/tests permit
   R1 acceptance_status = blocked_verification_authorization
   ```

7. Continue authoring and synthetic verification of later releases, as the plan permits, but:

   - do not call any dependent release accepted;
   - do not activate or publish research results;
   - mark dependent acceptance as transitively blocked by R1;
   - keep the missing owner fixture sign-off as the first item in every downstream gate summary and the final completion report.

If a valid `VerificationAuthorizationRef` already exists in the repository and exactly matches the final contracts, verify it and run the real vertical slice. Never create or approve it yourself.

One canonical real-data allowlist applies to the entire implementation-verification program. Do not rotate or substitute dates by release.

---

# 5. Test-first priorities

At minimum, implement tests first for:

## Identity and immutability

```text
ReplayInputBundle content sensitivity
Quant-Lab and Strategy-Core source identity sensitivity
CoreStrategyReplay reuse and independence
stable canonical profile naming
SearchChildMembership separation
Payload/Envelope identity-projection audit
deep immutability
GeneratedProfileCapability
VerificationRun authorization binding
semantic pipeline identity vs execution attempt
store overwrite refusal and verified reuse
concurrency-safe catalog rebuild/recovery
```

## Sequential FSM research

```text
profile/seed mismatch refusal
native-ID determinism
profile-independent lineage uniqueness and collision refusal
full replay per strategy child
audit-enabled vs audit-disabled neutrality
CohortSpec interpretation modes
no candidate-table execution shortcut
S11 blocked behavior
```

## Prop lifecycle

```text
TradePathBundle and path-event identity
account vs portfolio simulation identities
per-leg firm/risk/withdrawal/replacement policy identity
capability-based path requirements
unordered OHLC vs assumed scenario vs ordered-event evidence
typed day-count/calendar rules
totally ordered account events
field-level contract evidence compilation
synthetic vs first-party verification statuses
rule-by-rule deterministic fixtures
common correlated path across copied accounts
```

## MBP-1

```text
MBP-1-only feature/live/UI/model boundary
opaque legacy replay provenance cannot enter feature materialization
(ts_event, ts_recv, sequence, source_ordinal) ordering
exact stage cutoff and same-timestamp exclusion
per-feature WindowTriggerSemantics
typed missing reasons
R5 readiness refusal before R5B
R5B block activation/version change
```

## ML and regime

```text
identical rows/folds for prevalence/logistic/CatBoost
fold-local preprocessing
KMeans OOS assignment
RegimeProtocol vs RegimeFit vs assessment vs promotion identity
CONTEXT_BAR_PANEL validation
planned post-V1 algorithms fail closed
no direct spectral feature use
no automated feature/threshold/cluster-count selection
```

---


# 6. Frontend and user-experience execution contract

Read `FRONTEND_UX_CONTRACT.md` in full before touching any UI file. Implement every applicable `FUX-*` requirement; do not collapse the UI into a generic form, one monolithic Streamlit file, or a statistics-only dashboard.

Non-negotiable frontend expectations include:

```text
complete five-mode, eight-step wizard
persistent drafts, autosave, Save Draft, History, immutable Clone as New Search
registered-axis cards with exact classifications, evidence, and computation-path chips
full prop/risk/benchmark/validation/authorization/review interactions
Active Runs checklist, keyboard funnel, exact table/detail actions, safe cancel and fallbacks
Results overview, frontier, heatmap, firm matrix, survival, payout distributions, explorer,
comparison ribbon/four panels, deterministic insights, account timeline, exact verifier actions
Full Pipeline Configure/Preview/Launch/Monitor/Resume-Retry/Publish
Summary/Analyst/Audit disclosure and explicit result-scope/gross-cost-net labels
intentional empty/blocked/failure states and sanitized errors
keyboard, screen-reader-visible labels, chart/widget twins, non-color semantics
responsive QA and screenshots at 1440x900, 1024x768, 768x1024, and 390x844
```

For R4, R5, R5B, R6, and hardening:

1. Map changed code and tests to the exact `FUX-*` IDs in the release gate summary.
2. Run the field-level AppTests and source scans in `TEST_MATRIX.md` §3.11.
3. Perform an independent read-only UX adversarial review that attempts to find:
   - missing fields or modes;
   - hidden thresholds or raw override paths;
   - inaccessible chart-only interactions;
   - color-only status semantics;
   - missing empty/failure states;
   - incorrect development/authorization/path-fidelity wording;
   - lost exact-ID drill-down;
   - mobile/tablet clipping or unusable tables;
   - pipeline actions that can launch outside an explicit handler.
4. Fix valid findings before the release implementation checkpoint.
5. Preserve screenshot and keyboard evidence in the relevant release folder.
6. When interactive browser infrastructure is unavailable, leave the relevant acceptance gate open; do not substitute “HTTP 200” or AppTest success for interactive viewport/keyboard evidence.

The existing M0–M3 Context Research panel and top-level Replay / Verifier and Data & Audit routes remain intact.

---

# 7. Progress-document organization

Create:

```text
C:\Users\gonza\Documents\Claude-Quant-Lab\QL-FSM-PROP-SEARCH-DASHBOARD\implementation-progress\
```

with:

```text
R1\
R2\
R3\
R4\
R5\
R5B\
R6\
hardening\
DECISIONS_TAKEN.md
COMPLETION_REPORT.md
```

Every implementation document produced during the work must live in the release folder it belongs to.

Per-release folders must contain, at minimum:

```text
GATE_SUMMARY.md
TEST_RESULTS.md
ADVERSARIAL_REVIEW.md
ADVERSARIAL_REVIEW_RESOLUTION.md
FILES_TOUCHED.md
DEVIATIONS.md
ACCESS_SAFETY_EVIDENCE.md, where applicable
```

Additional scratch analysis and notes belong in the relevant release folder.

The only exceptions are repository-convention files that belong in their established locations:

```text
docs/DECISIONS.md
ARCHITECTURE.md
docs/README.md
docs/pipeline_state.yaml
source code
tests
```

Every release gate summary must list:

```text
commit hash
all repo files touched
all implementation-progress files produced
exact commands run
exit codes
test counts
timings
gate status
open blockers
protected/sealed counters
```

Do not scatter documents elsewhere.

---

# 8. Commit discipline

Commit at each release’s implementation checkpoint.

Use release-scoped commits, for example:

```text
R1: add search identities, verification policy, and stores
R2: add child orchestration, lineage, and deltas
R3: add prop lifecycle contracts and synthetic account engine
...
```

When R1 acceptance is still blocked, the commit message and gate summary must say:

```text
implementation complete; acceptance blocked pending VerificationAuthorizationRef
```

Do not falsely use:

```text
accepted
verified on real fixture
production ready
research ready
```

No push. No merge.

Do not commit generated raw data, normal research artifacts, mutable existing catalogs, or external-source downloads.

Test artifacts must remain in temporary or explicitly test-namespaced storage.

---

# 9. Hard constraints

These are non-negotiable.

## Repository boundaries

- Do not modify Strategy-Core.
- Do not modify Trade-Lab.
- Do not modify the fixed M0–M3 lane modules or semantics.
- Do not mutate any existing immutable artifact.
- Do not mutate any existing immutable or legacy catalog.
- Preserve existing public behavior of the evaluation-only propsim API.
- Prefer additive lifecycle modules; any compatibility edit explicitly permitted by the final plan must be minimal, tested, and logged.
- Do not edit the final implementation-plan package.

## Data safety

- Real-data access occurs only through the existing policies plus the new `VerificationReplayPolicy`.
- Use one canonical owner-authorized allowlist of at most five real trading days.
- Use one baseline profile and its exact profile-matching seed.
- Multi-child implementation verification is synthetic only.
- Never construct, list, stat, open, or read June 11, 2026.
- Never construct, list, stat, open, or read the sealed range.
- All protected/sealed counters remain zero.
- No full-development replay, feature materialization, model fit, configuration search, prop search, bootstrap research run, or operator full-pipeline run occurs during implementation.
- Do not fetch or scrape current prop-firm rules. R3 uses synthetic contract evidence unless owner-supplied first-party evidence already exists and is separately authorized.

## Order flow

- MBP-1 is the maximum supported new order-flow depth.
- New feature, bundle, model, UI, dashboard, and live-source contracts may not expose MBP-10 or deeper depth.
- Opaque `legacy_verified_replay_source` provenance may be retained only where the final plan permits it for existing replay input identity.
- Legacy provenance may not be queried by R5B feature materialization, model features, dashboard controls, or live-source contracts.
- R5B is `research_only_offline`.
- R5B cannot create a live execution gate, Trade-Lab serving feature, or production inference feature.

## ML and regime

- V1 implements prevalence, regularized logistic regression, CatBoost, KMeans, and the planned/blocked registry/UI states.
- MiniBatch KMeans, GMM, direct spectral diagnostics, and Nyström+KMeans remain post-V1.
- S11 remains blocked with the exact reason text defined by the authoritative contracts.
- No model-gated sequential replay is authorized.
- No feature, model, threshold, cluster count, or strategy is automatically selected or promoted.
- Selected development configurations are labeled:

  ```text
  Development Exploratory Representative
  ```

- No profitability, validation, sealed, promotion, production, or live-readiness claim is permitted.

## Prop realization

- Assumed one-minute intrabar paths are always labeled scenario/approximation.
- They are never described as exact historical chronology.
- Only actual ordered event evidence may support ordered historical replay.
- Synthetic contracts remain `synthetic_fixture_verified`, never `first_party_verified`.
- No real prop result is produced without separately authorized first-party contract evidence.

## Documentation and governance

Update, in the same release changes required by repository convention:

```text
docs/DECISIONS.md — D-039 through D-045
ARCHITECTURE.md
docs/README.md
docs/pipeline_state.yaml
```

Do not overwrite unrelated existing decisions or documentation.

---

# 10. Adversarial review standard

For each release, ask independent read-only reviewers to attempt to prove that the implementation violates:

```text
identity projection
immutable reuse
point-in-time evidence
full sequential replay
one-active-setup / one-active-trade
cross-profile lineage
prop path fidelity
calendar semantics
MBP-1 depth and cutoff rules
fold-local ML
protected/sealed access
legacy M0–M3 compatibility
```

The review report must contain:

```text
finding
severity
contract citation
code/test evidence
reproduction steps
resolution
post-fix verification
```

Do not dismiss a finding because the tests are green. Fix the contract violation or prove the finding invalid with exact evidence.

---

# 11. Failure and stop conditions

Do not improvise around the plan.

Stop and report when:

```text
a required change would modify Strategy-Core
an owner-ratified semantic choice is genuinely missing
the current codebase contradicts a load-bearing final-plan assumption
a release gate remains red after real diagnosis/fix attempts
protected/sealed access cannot be proven zero
existing user-owned changes overlap and cannot be isolated
the implementation would require a full-development run to validate
```

A missing `VerificationAuthorizationRef` is not a reason to stop code authoring. It is a reason to keep R1 and every dependent acceptance status blocked.

Where the plan provides an engineering default or `proposed_protocol_default`, implement it and log the choice. It remains unratified for research until the appropriate owner evidence exists.

---

# 12. Hardening

Hardening must include every command and UI check specified by the final package.

At minimum:

```text
full Quant-Lab pytest suite
all targeted new suites
ruff over all covered source/test paths
targeted validation for scripts/UI that current CI does not cover
git diff --check
immutable repeat/reuse tests
access-safety proof
Streamlit AppTest
browser viewport checks at required sizes
keyboard-only navigation
all `FUX-*` empty/blocked/failure states
screenshot evidence at 1440x900, 1024x768, 768x1024, and 390x844
chart/widget fallback parity
wide-table and long-ID responsive behavior
long-ID and wide-table behavior
```

When browser automation is unavailable, do not claim the browser gate passed. Record it as an open hardening gate.

Do not substitute a different browser method when the contract explicitly requires a particular backend or evidence form.

Hardening does not execute the full authorized development pipeline.

---

# 13. Completion report

Finish with:

```text
implementation-progress\COMPLETION_REPORT.md
```

The report must contain:

1. Executive status.
2. Per-release:

   ```text
   implementation_status
   acceptance_status
   commit hash
   files changed
   features delivered
   exact gate evidence
   tests and counts
   adversarial findings and resolutions
   open blockers
   ```

3. The R1 fixture-authorization status as the first blocker when still absent.
4. A dependency map showing which later acceptance statuses remain transitively blocked by R1.
5. All plan deviations and reasons.
6. Consolidated `DECISIONS_TAKEN.md`.
7. Confirmation that:

   ```text
   Strategy-Core unchanged
   Trade-Lab unchanged
   M0–M3 behavior unchanged
   existing immutable artifacts/catalogs unchanged
   protected/sealed counters zero
   no full-development run occurred
   no operator full run occurred
   no real prop claims were produced
   ```

8. The exact owner actions needed next.
9. The exact review order, beginning with:

   ```text
   1. COMPLETION_REPORT.md
   2. R1\GATE_SUMMARY.md
   3. hardening\GATE_SUMMARY.md
   4. release commit map
   5. highest-severity adversarial-review resolutions
   ```

10. A clear final classification:

   ```text
   implementation complete / incomplete
   release acceptance complete / blocked
   research authorization absent / present
   operator full run not executed
   ```

Do not describe the project as complete without distinguishing implementation completion from acceptance and research authorization.
