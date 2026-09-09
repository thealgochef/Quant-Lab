# R6 — Gate Summary

**Release:** R6 — V1 KMeans Regime Lane (PHASED_DELIVERY "Release 6"; V3
P1-3/P1-6; Amendments P1-B/P1-C; the GMM/minibatch/spectral/Nyström
implementations are the post-V1 regime-expansion release)
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R6 cannot be declared accepted until R1's acceptance —
> including the owner-approved coverage matrix, allowlist, seed snapshot,
> and signed `VerificationAuthorizationRef` — passes.

> Session note: R6 was authored across a context reset (2026-08-26 →
> 2026-08-28). The pre-reset tree carried the complete implementation and
> a first adversarial round's C-series fixes; its review reports were lost
> with the context, so the adversarial round was re-run in full on
> 2026-08-28 (`ADVERSARIAL_REVIEW.md`) and every finding dispositioned
> (`ADVERSARIAL_REVIEW_RESOLUTION.md`).

## Commits

- **`179a2c9`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `7f018f5` = R5B) — 22 files changed, +5,928 / −12
  (7 new src + 1 new script + 5 new test/fixture files; 2 modified src +
  1 modified script + 2 modified test files; `docs/DECISIONS.md`;
  the three staged shared docs).
  **Message:** `R6: add the V1 KMeans regime lane — contracts, fold-local
  service, alignment, diagnostics, stores, panel` (+ the acceptance-blocked
  statement).
- Not pushed. Not merged.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R6 lane transforms ONLY
  (`R6/stage_shared_docs.py`; post-commit `--apply-worktree` replayed the
  same transforms). **Post-commit verification: the surviving worktree
  diff on the four user-owned files is content-identical to
  `../R1/PRE_EXISTING_DIFF.patch`** (the only differing lines are the git
  `index` header hashes, which necessarily change as HEAD advances — raw
  comparison in `_surviving_shared_doc_diff.patch`; the diff-of-diffs was
  empty). `docs/ML_TRAINING_WORKBENCH.md` untouched and uncommitted.
  `docs/DECISIONS.md` gains the D-044 KMeans-regime paragraph + the updated
  reservation note (not user-dirty; committed normally).

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against `git status --short`,
including the adversarial-fix round (`search/store.py` sidecar-loader
hardening, `test_ifvg_study_scans.py` scan-list extension).

## Implementation-progress files produced

`PRE_R6_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`
(DEV-R6-1…8 + scoping notes), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both reviewer
reports), `ADVERSARIAL_REVIEW_RESOLUTION.md` (25 dispositions),
`stage_shared_docs.py`, `r6_smoke_app.py`, `browser-smoke/`
(11 screenshots + `MANIFEST.json`), raw outputs `_baseline_pytest.txt` /
`_midpoint_pytest.txt` / `_final_pytest.txt` / `_env_dependent_check.txt`
/ `_smoke_server.log` / `_surviving_shared_doc_diff.patch`,
`GATE_SUMMARY.md` (this) + `../DECISIONS_TAKEN.md` entries 54–66.

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (`179a2c9` content):
full repo 1733 passed, 2 failed** — the two failures are the PRE-EXISTING
environment-dependent provider-key tests (this session's env carries the
API keys; both pass with the keys cleared — `_env_dependent_check.txt`),
identical to the pre-R6 baseline: **+51 net new tests, 0 new failures**
(10:02; raw `_final_pytest.txt`). `ruff check src tests scripts` clean;
`git diff --check` clean; live browser smoke over a freshly built 16-stage
run + persisted regime artifacts — **server log has zero tracebacks and
zero deprecation warnings**, 11 screenshots bound to commit/tree/pipeline/
scratch-key by `browser-smoke/MANIFEST.json`.

## Gate status (PHASED_DELIVERY R6 gate rows → evidence)

| Gate item | Status |
|---|---|
| KMeans fold-local fit/assign/align/coverage/stability on fixtures (2 + 4-KMeans) | **PASS** — per valid fold the pinned `kmeans_v1` fits on training rows only and assigns train+test rows deterministically (double-run hash equality; OOS AMI > 0.99 vs the known memberships; distances to every centroid, margin d2−d1); fold locality proven by test-row poisoning (fit identity, bootstrap AMI, silhouette, separation unchanged); every fit identity pins verified source artifacts + the training-feature-matrix hash (review F1) |
| R6 UI satisfies FUX §35 (coverage / occupancy / stability / assignment / stratification, proposal stamps, sample-adequacy blocks, context-panel identity, planned post-V1 states) | **PASS (assignment-frame views) / SCOPED (stratified RESULT views)** — coverage (+ per-fold), nominal-id occupancy, stability (+ per-cluster agreement, centroid profiles, OOS-timeline transition matrix), the exact-fit-id ASSIGNMENT view (coverage by partition, distance/margin quantiles, the OOS regime timeline), the STRATIFICATION of the assignment frame (regime × partition), the promotion role/status view, proposal stamps, the insufficient-sample blocked state, the panel-grain identity, and planned-disabled algorithm entries with the mandatory spectral warning — 8 AppTests + the live browser smoke; **owner note (DEV-R6-4):** strategy/model/prop METRICS by regime are the ML §5.5 comparison classes and land with their studies, not with the lane |
| Proposal stamps (`proposed_protocol_default`) surfaced in the UI | **PASS** — the nine stamped defaults (incl. the decision-row minimum) with owner-decision numbers and the ratification requirement; the model card stamps fixed-k |
| Sample-adequacy gate blocks under-sampled candidate-stage fits | **PASS** — `sample_adequacy` failure under 150 training rows per fold blocks promotion; every fit still carries exactly k centroids (never shrunk); the blocked state renders with the exact gate/observed values |
| The actual `CONTEXT_BAR_PANEL` schema (interval/source/as-of fields + validation, P1-B) on synthetic 5m/15m panels incl. PIT panel→candidate assignment | **PASS** — all three panel fields required iff the panel grain (each missing field / each non-panel grain refuses; `< 60s` refuses); the 5m/15m fit path executes end to end on a 480-bar synthetic panel with hand-built walk-forward folds (≥300 training bars per fold); the PIT assignment consults only OUT-OF-SAMPLE assignments of the last COMPLETED bar at or before the as-of instant (`<=`), lowest fold wins, independent of input order, gaps typed `coverage_gap` (review F4) |
| Planned spectral/Nyström fit requests refused in V1 with the correct status/reason (P1-C — no fit implementation callable) | **PASS** — every post-V1 key refuses at `assert_regime_algorithm_fittable` and again inside `run_regime_protocol` BEFORE any preprocessing (zero preprocessing fits, test-pinned); every planned protocol POLICY (`inner_train_only_selection`, PCA, kernels, non-centroid OOS policies) refuses the same way (review S3); registry drift (version / pinned-parameter hash) refuses; `GaussianMixture`/`SpectralClustering`/`Nystroem`/`MiniBatchKMeans` imported nowhere; `IFVG_REGIME_CONTEXT_V1` still refuses resolution |
| TEST_MATRIX §3.8 "Regime context-panel schema/validation" + "Planned spectral capability refusal" | **PASS** (rows above) |
| TEST_MATRIX §3.9 "Regime protocol/fit/promotion separation" (status changes leave `resolved_regime_protocol_id` and every `regime_fit_id` unchanged; role/status absent from protocol/fit payloads) | **PASS** — unit + store tests; the promotion ladder is now STRUCTURAL at the contract and at persistence (reviews F5/S1/S2) |
| TEST_MATRIX §3.10 "Regime fit provenance" (assignments reference exact fit+protocol; assessment references exact fit set + fold set; promotion changes no fit id) | **PASS** — assignment frames are bound to their fit at persistence and re-predicted from the exact bytes before publication (review F6); the assessment carries `regime_fit_ids` + `fold_set_id`; reload after promotion under the original fit id |
| ML §11 rows 2/3/4/8/9/10/15/18/19 (fold-local preprocessing; leakage validator; deterministic OOS KMeans; fixed k; nominal ids + prediction-hash invariance; typed missing rows; no automated selection; distinct identities; no DL/RL) | **PASS** — incl. the default-on registry-derived stage rule (review F2), geometry-ranked nominal ids with the exact tie-break (reviews F3/F8/F9), all-missing rows outside the fit (F10), unique keys (F11) |
| V1 boundary (post-V1 expansion not required) | **PASS** — `kmeans_v1` is the single implemented algorithm (import-time invariant); fixture 3 and the GMM/Nyström arms are the expansion release by plan |

## Carried-obligation closure

1. **DECISIONS_TAKEN #57/#59 (pre-reset wording)** — CORRECTED: the
   panel→candidate rule and the promotion-ladder claims now describe what
   the code enforces (reviews F4/F5/F15).
2. **Reviewer S5 (R5B panel outside the FUX scans)** — CLOSED:
   `ifvg_mbp1_panels.py` and `ifvg_regime_panels.py` are under the scans.
3. **Shared store hardening (S6)** — CLOSED for every lane:
   `load_sidecar_bytes` verifies the manifest hash, whitelists the name,
   and hashes the returned bytes.

## Adversarial review

Two independent read-only reviewers (contract fidelity; safety/access):
**0 blockers; 7 majors + 9 minors (contract lens) and 3 majors + 1 medium
+ 5 minors (safety lens) — ALL 25 dispositioned** in
`ADVERSARIAL_REVIEW_RESOLUTION.md`: 23 fixed in code/tests (incl. the
content-bound fit identity F1, the default-on input permission F2/S4, the
OOS-timeline temporal facts F3, the OOS-only panel assignment F4, the
structural promotion + role ladders F5/S1/S2, the assignment↔fit binding
F6, the delivered assignment/stratification/promotion views F7, the exact
Hungarian tie-break F8, fail-closed protocol policies S3, the hardened
sidecar loader S6, verify-before-publish S8), 2 accepted with
documentation (the stratified-RESULT-view scope of F7 → DEV-R6-4; the
store trust boundary S7 → DEV-R6-8, with the no-unpickle UI loader as a
partial fix). The safety lens's protected/sealed verdict: **zero-counter
AFFIRMED at the R6 layer**.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R5B: blocks R1 acceptance and, transitively, R6's (and every later
   release's) acceptance.
2. Acceptance-time slice requirements carried by R1/R2/R5/R5B evidence
   (unchanged), now including the five-day real mini-run of ML §9
   (artifact pair → … → one `kmeans_v1` regime fit → stores → UI states)
   once the authorization exists — the synthetic fixtures prove the
   identical chain shape.
3. Owner ratification of the regime defaults (OWNER_DECISIONS 25–30 — all
   still `proposed_protocol_default`): nothing in the lane can reach
   `FEATURE_ELIGIBLE` without the `OwnerDecisionEvidenceRef`.
4. Hardening gates OPEN by design: the four-viewport screenshot matrix,
   keyboard-only navigation, chart/widget fallback parity, long-ID/
   wide-table interactive QA (the R6 desktop browser smoke is recorded
   evidence, not a hardening waiver); store signing / non-pickle
   serialization as a hardening candidate (DEV-R6-8).
5. Post-R6 wiring recorded as deviations: the 5m/15m panel materializer +
   panel fold builder (DEV-R6-7); stratified RESULT views with the ML §5.5
   study classes (DEV-R6-4).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no real source path was ever
constructed (the lane's inputs are in-memory frames; its only I/O is the
verified store protocol on caller-supplied roots); every test ran on
synthetic fixtures with tmp-only writes; the browser smoke wrote only to
the content-addressed `%TEMP%` scratch; the repo's real data namespaces
were never written (verified on disk by the safety reviewer — no file
under `data/` newer than 2026-08-25; `search/v1` and `search_test/v1` do
not exist; no `regime_*` store directory exists in the repo); no
protected/sealed date literal exists in the R6 set; Strategy-Core clean
and Trade-Lab untouched by this release (its pre-existing user worktree
changes acknowledged, mtimes ≤ 2026-07-31); M0–M3 and the propsim API
byte-unchanged; `full_pipeline_not_run` holds; no regime output can reach
a predictive bundle or an execution surface (planned block + S11 blocked +
execution-side roles unrepresentable). Corroborated by the safety
reviewer's independent grep/on-disk audit (verdict AFFIRMED).
