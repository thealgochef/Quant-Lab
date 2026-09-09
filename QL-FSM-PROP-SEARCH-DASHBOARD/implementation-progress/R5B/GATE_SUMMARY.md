# R5B — Gate Summary

**Release:** R5B — Offline MBP-1 Feature Activation (mandatory; immediately
after R5; owner ruling P1-D, boundary decision R-6)
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R5B cannot be declared accepted until R1's acceptance —
> including the owner-approved coverage matrix, allowlist, seed snapshot,
> and signed `VerificationAuthorizationRef` — passes.

## Commits

- **`7f018f5`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `fb8062f` = R5-FIX) — 35 files changed, +5,963 / −125
  (8 new src + 1 new script + 5 new test files; 11 modified src +
  2 modified scripts + 8 modified test files; `docs/DECISIONS.md`).
  **Message:** `R5B: activate offline MBP-1 features — schemas, PIT
  windows, materializer, controlled study` (+ the acceptance-blocked
  statement).
- Not pushed. Not merged.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R5B lane transforms ONLY
  (`R5B/stage_shared_docs.py`; post-commit `--apply-worktree` replayed the
  same transforms). **Post-commit verification: the surviving worktree
  diff on the four user-owned files is content-identical to
  `../R1/PRE_EXISTING_DIFF.patch`** (74 + 5 + 44 + 28 hunk lines; the only
  differing lines are the three git `index` header hashes, which
  necessarily change as HEAD advances — raw comparison in
  `_surviving_shared_doc_diff.patch`). `docs/ML_TRAINING_WORKBENCH.md`
  untouched and uncommitted. `docs/DECISIONS.md` gains **D-046** (the
  activation decision the R5-era reservation note promised) + the updated
  reservation note (not user-dirty; committed normally).

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against `git status --short`,
including the adversarial-fix round (`search/identities.py` audit-import
list, `study_providers.py` auto-fill provider, `test_identities.py`
type-aware placeholders).

## Implementation-progress files produced

`PRE_R5B_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`
(DEV-R5B-1…6 + scoping notes), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both reviewer
reports), `ADVERSARIAL_REVIEW_RESOLUTION.md` (17 dispositions),
`stage_shared_docs.py`, `r5b_smoke_app.py`, `browser-smoke/`
(4 screenshots + `MANIFEST.json`), raw outputs `_baseline_pytest.txt` /
`_midpoint_pytest.txt` / `_final_pytest.txt` /
`_env_dependent_baseline_check.txt` / `_smoke_server.log` /
`_surviving_shared_doc_diff.patch`, `GATE_SUMMARY.md` (this) +
`../DECISIONS_TAKEN.md` entries 46–53.

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (`7f018f5` content):
full repo 1682 passed, 2 failed** — the two failures are the PRE-EXISTING
environment-dependent provider-key tests (this session's env carries the
API keys; both pass with the keys cleared —
`_env_dependent_baseline_check.txt`), identical to the pre-R5B baseline:
**+78 net new tests, 0 new failures** (11:08; raw `_final_pytest.txt`).
`ruff check src tests scripts` clean; `git diff --check` clean; live
browser smoke over a freshly built B2 16-stage run — **server log has
zero tracebacks and zero deprecation warnings**, 4 screenshots bound to
commit/tree/pipeline/scratch-key by `browser-smoke/MANIFEST.json`.

## Gate status (PHASED_DELIVERY R5B: the owner's 13 deliverables + gate rows → evidence)

| Deliverable / gate item | Status |
|---|---|
| (1) Immutable MBP-1 source/coverage artifact | **PASS** — content-addressed `Mbp1SourceArtifact` (per-partition hashes, first/last order keys, gap intervals — vendor resets are not gaps — gap-adjusted coverage); storage mode is an ENVELOPE fact (review F7), so byte-identical evidence has one id across stored/referenced modes; save refuses sidecars contradicting coverage (review F1) |
| (2) Exact Arrow schemas + schema hashes | **PASS** — four pinned schemas (raw mbp-1 retaining `ts_recv`; normalized with the `source_ordinal` tie-break; the 76-metric feature table + per-window evidence; stage-window evidence) with canonical field/type hashes, field- and type-sensitive by test |
| (3) Exact PIT stage-cutoff support on `(ts_event, ts_recv, sequence, source_ordinal)`; no `+inf`; same-timestamp exclusion | **PASS** — `PRE_TRIGGER_EXCLUSIVE` `<` / `POST_TRIGGER_INCLUSIVE` `<=` on exact keys; strict `ts_event <` for timestamp-only evidence with EVERY same-ts event excluded and typed ambiguity; both key timestamps validated non-infinite (review F6); the widened no-+inf source scan covers the features package + pipeline + ml modules |
| (4) Stage-window construction for the registered lifecycle anchors | **PASS** — the candidate row's five anchors → `COMPLETED_BAR_BOUNDARY` cutoffs under `completed_bar_boundary_exclusive_v1`; the frozen 9-window registry (POST_TRIGGER_INCLUSIVE, OPEN/CLOSED, min-count 1) exactly per DT §6.2; a missing lower anchor refuses rather than widening |
| (5) The offline MBP-1 feature materializer | **PASS** — 76 hand-computed-verified formulas (CKS OFI, aggressor fractions, depletion/replenishment, absorption, intensities, top-of-book snapshots); supplied evidence VERIFIED against the source artifact's coverage hashes before any window (review F1); deterministic batch/repeat; the §3.8 same-ts-after-stage mutation proof at the FEATURE level under an exact-key builder (review F3) |
| (6) Typed missingness + source-coverage reasons | **PASS** — all seven registered reasons emitted under the documented day→anchor→boundary→content precedence; rows always preserved; formula-edge NaN stays VALID; every emitted reason registered by test |
| (7) Exact candidate/stage joins, no nearest-time or row-order fallback | **PASS** — one-to-one on `candidate_id` only; duplicate refusal; the no-merge_asof/nearest source scan; the bundle-view seam refuses cohort misalignment (review F9) and verifies the frame↔artifact binding by rehash (review F1) |
| (8) `IFVG_ORDER_FLOW_MBP1_V1` activation as a new versioned research-only block | **PASS** — the published registry IS `with_activated_block` over the exported pre-activation state: `block_version` 2, first resolved id minted, registry hash changed (both hashes surfaced in the UI), B2/B3 re-minted, B1/B4 stable, replay-the-event equality proof; window-definition changes mint new resolved ids (review F4) |
| (9) Feature coverage/validity reports | **PASS** — per-day/per-window/per-feature `Mbp1CoverageReport` with the permanent `research_only_offline` stamp, content-addressed persist/reload |
| (10) Controlled Baseline vs Baseline+MBP-1 study on identical profile/candidate rows/labels/folds/model protocol | **PASS** — challenger vs its OWN base bundle; bundle-parametrized prevalence+logistic arms (CatBoost tier-locked with the exact reason at ladder/readiness/UI); cross-arm row identity + numerically identical prevalence reference asserted; paired Brier delta under the fixed 10k/seed-7 bootstrap (the planted-signal test proves the joined columns reach the model); ladder ids bind the exact evidence artifact (review F2); pipeline S09 runs + persists it (E2E in the 0-valid-fold safe-failure shape, exactly the R5 ladder pattern) |
| (11) Dashboard support (availability, coverage, bundle selection, comparison deltas, evidence drill-down) | **PASS** — the MBP-1 Order Flow panel (availability + both registry hashes, the frozen window registry, exact-ID coverage/missingness, per-candidate stage-window drill-down with no fuzzy fallback, the controlled comparison) under the persistent `research_only_offline` badge; Configure offers B2/B3 after activation and pins the logistic protocol with the tier-lock caption; auto-fill via the provider layer with the integrity note (review S6); 6 AppTests + the live browser smoke |
| (12) Five-day real control-flow verification + synthetic feature-formula fixtures; no full-development materialization | **PASS (synthetic half) / BLOCKED-honest (real half)** — hand-computed formula fixtures + the full synthetic pipeline E2E; the real half is DEV-R5B-1: the policy-gated real readers exist and refuse correctly (authorize-before-path, deeper-book refusal, column projection), but the real run awaits the owner's `VerificationAuthorizationRef`, exactly like every real half since R1 (DEV-R5-6 precedent); no full-development materialization occurred |
| (13) Immutable save/reload/reuse + exact artifact identities | **PASS** — all four new stores on the manifest protocol; save-or-reuse idempotency; tamper refusals on every sidecar (envelope-fact rehash on load); identity sensitivity proofs (events bytes, anchors incl. `setup_id` [review F8], window definitions, evidence artifact) |
| §3.8 same-timestamp future-event exclusion / timestamp-only conservative cutoff / no-+inf scan | **PASS** (admission level + the F3 materializer-level mutation proof; scan widened) |
| §3.8 MBP-1 activation state (R5B half) + §3.9 legacy-scoping guards re-run | **PASS** — active v2 with the resolution envelope; the R5-era planned-state tests flipped to their two-sided R5B halves; legacy provenance structurally sealed (no `source_kind` ingress — API-surface scan) with the refusal function as its executable statement |
| §3.9 window-trigger semantics; §3.10 typed window definitions + block-resolution identity | **PASS** |
| FUX §35 R5B rows + §30.1 (bundle available as `research_only_offline` after R5B) | **PASS** — AppTest-witnessed + browser smoke; no launch on render (the spawn spies stay zero across all phases incl. the new expander) |
| MBP-10/deeper-feature guards remain green | **PASS** — and STRENGTHENED: regex covers every depth ≠ 1 (review S2); the read seam refuses deeper-book files and column-projects lawful ones (review S1) |

## Carried-obligation closure (R5 → R5B)

1. **DEV-R5-10 (safety F4) — search-shim store-root alignment** — CLOSED:
   `ifvg_search_job.py` passes the worker's `--store-root` to runner-entry
   factories (`factory(charter, store_root=…)`), matching the pipeline
   shim's contract; the registered synthetic factory accepts the keyword;
   the safety reviewer verified no access widening.
2. **The MBP-1 activation event itself (the owner's 13 deliverables)** —
   CLOSED per the table above (real five-day half blocked-honest,
   DEV-R5B-1).
3. **DECISIONS_TAKEN #41 — bundle-parametrized ladders** — CLOSED for the
   R5B lane: tier XOR (bundle features + resolved bundle ref + evidence
   ref); the CatBoost rung stays tier-locked by design.

## Adversarial review

Two independent read-only reviewers (contract fidelity; safety/access):
**0 blockers; 3 majors + 8 minors (contract lens) and 1 medium + 5 minors
(safety lens) — ALL 17 dispositioned** in
`ADVERSARIAL_REVIEW_RESOLUTION.md`: 13 fixed in code/tests (incl. the
evidence↔identity verification chain F1, the ladder evidence binding F2,
the materializer-level same-ts mutation proof F3, the ingestion-side
deeper-book refusal S1, and the provider-layer auto-fill S6), 3 accepted
with documentation (the declared DEV-R5B-1 real-half blocker; the
enumerated out-of-scope files; the pre-existing Trade-Lab worktree state),
1 refuted with precedent (F11 — result artifacts hash their content, per
the frontier/insights/coverage pattern). The safety lens's
protected/sealed verdict: **zero-counter AFFIRMED at the R5B layer**.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R5: blocks R1 acceptance and, transitively, R5B's (and every later
   release's) acceptance.
2. Acceptance-time slice requirements carried by R1/R2/R5 evidence
   (unchanged), now including the real five-day MBP-1 control-flow run
   (DEV-R5B-1): completing the real evidence wiring (anchors from the
   published baseline v2 dataset + the authorized mbp1 partitions under
   `VerificationReplayPolicy`) once the authorization exists.
3. Hardening gates OPEN by design: the four-viewport screenshot matrix,
   keyboard-only navigation, chart/widget fallback parity, long-ID/
   wide-table interactive QA (the R5B desktop browser smoke is recorded
   evidence, not a hardening waiver).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no real source path was ever
constructed; every test ran on synthetic fixtures with tmp-only writes;
the repo's real data namespaces were never written (verified on disk by
the safety reviewer — `search/v1` and `search_test/v1` do not exist, no
namespace written today); authorize-before-path proven with zero
path-factory invocations on refusal; the real MBP-1 seam is unwired in
the real executors (a real MBP-1 plan refuses at S00); Strategy-Core
clean and Trade-Lab untouched by this release (its pre-existing user
worktree changes acknowledged, mtimes ≤ 2026-07-31); M0–M3 and the
evaluation-only propsim API byte-unchanged; `full_pipeline_not_run`
holds; `research_only_offline` is structural (definition-level boundary +
import-time invariant + S11 blocked + zero live/serving controls).
Corroborated by the safety reviewer's independent grep/on-disk audit
(verdict AFFIRMED).
