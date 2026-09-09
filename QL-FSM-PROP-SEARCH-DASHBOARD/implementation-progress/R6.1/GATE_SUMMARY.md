# R6.1 — Gate Summary

**Release:** R6.1 — the regime-lane correction (the second of the two
correction-release commits planned by
`../R6.1-CORRECTION-PLAN-DOCS-FINAL/R6.1_IMPLEMENTATION_PLAN.md` revision 3
— workstreams A–L, decisions D1–D15; owner planning decisions Q1–Q4;
plan-review corrections 1–7; final contract-closure rulings 1–8)
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release; in addition every FEATURE_ELIGIBLE promotion of a real protocol
> stays blocked on the owner's ratification of decisions 25/28/29/30 as a
> persisted owner-decision artifact (the three drafts under
> `DRAFT_OWNER_DECISION_PROPOSALS/` are proposals, not authorizations), and
> the real five-day regime mini-run stays blocked on the real-slice
> authorization (plan §12 blocker 1).

> Session note: the R6.1 implementation crossed three context resets and one
> session rate-limit kill of the four parallel fix workstreams
> (`_PROGRESS_CHECKPOINT.md` is the resume-from-here record). The adversarial
> round ran on the complete post-implementation tree (two independent
> read-only reviewers, `ADVERSARIAL_REVIEW.md`), every finding was
> dispositioned (`ADVERSARIAL_REVIEW_RESOLUTION.md`), the fixes re-ran the
> full suite twice, and only then was the release-scoped commit made.

## Commits

- **`6c0b60a`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `f3f9ac2` = R5B.1; git tree `70e972a0…`) — 87 files changed,
  +25,783 / −566: 46 added (28 src modules incl. `propsim/event_detail.py`,
  1 script, 17 test files / fixtures) + 41 modified (21 src, 2 scripts, 14
  tests / fixtures, `docs/DECISIONS.md` + the three staged shared docs).
  **Message:** `R6.1: regime-lane correction — verified panel/fold/owner-evidence
  seams, frozen model-bearing authority, stratification classes, the
  bundle-aware CatBoost rung` (+ the acceptance-blocked statement).
- Not pushed. Not merged.
- **Path-scoped**: the commit carries exactly the R6.1 file list
  (`FILES_TOUCHED.md`); `docs/ML_TRAINING_WORKBENCH.md` (user-owned) and the
  pre-existing untracked local files (data / reports / prompts / the `QL-*`
  evidence tree) are NOT in the commit.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R6.1 lane transforms ONLY
  (`stage_shared_docs.py`; blob ids `c89b5ec5…` / `39646cc2…` / `b24c174a…`;
  post-commit `--apply-worktree` replayed the same transforms). **Post-commit
  verification: the surviving worktree diff on the four user-owned files is
  content-identical to `../R1/PRE_EXISTING_DIFF.patch`** (diff-of-diffs
  empty modulo `index` header lines — `_surviving_shared_doc_diff.patch`).
  `docs/DECISIONS.md` gains D-047 (amended after the adversarial round) +
  the updated reservation note (committed normally).
- Source-review evidence: `R6.1.patch` (`git format-patch --stdout
  f3f9ac2..6c0b60a`, 1,310,228 bytes) + `R6.1.patch.sha256` (final closure #8).

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against the commit's `--name-status`
after the adversarial-fix round.

## Implementation-progress files produced

`PRE_R6_1_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-R6.1-1…18 +
scoping notes), `TEST_RESULTS.md`, `ACCESS_SAFETY_EVIDENCE.md`,
`ADVERSARIAL_REVIEW.md` (both reviewer reports verbatim; raw
`_review_contract.md` / `_review_safety.md`), `ADVERSARIAL_REVIEW_RESOLUTION.md`
(28 dispositions), `stage_shared_docs.py`, `r61_smoke_app.py`,
`verify_browser_manifest.py`, `build_browser_manifest.py`, `browser-smoke/`
(25 screenshots + `MANIFEST.json` v2 + `_screenshots.json` +
`_smoke_server.log`), `DRAFT_OWNER_DECISION_PROPOSALS/` (three drafts; two
regenerated over the smoke store), raw outputs `_midpoint_pytest.txt` /
`_final_pytest.txt` / `_final_pytest_keys_cleared.txt` /
`_ruff_and_diffcheck.txt` / `_surviving_shared_doc_diff.patch`, `R6.1.patch`
+ `.sha256`, `_PROGRESS_CHECKPOINT.md` (the resume record), `GATE_SUMMARY.md`
(this) + `../DECISIONS_TAKEN.md` entries 76–100 (#82/#85/#86/#88 carry
in-place "AMENDED by the R6.1 adversarial round" pointers).

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (the `6c0b60a` content):
full repo 1931 passed, 0 failed as-is (17:57) and 1931 passed, 0 failed with
the provider keys cleared (18:09)** — the former environment-dependent pair
is hermetic since this release; **+173 net new tests over R5B.1 (1758), 0
failures in either environment.** `ruff check src tests scripts` clean;
`git diff --check` clean over the tracked and the new files. Browser
manifest v2 validated against the commit (`verify_browser_manifest.py` → OK,
26 files, 25 screenshots, committed-tree digest verified).

## Gate status (plan §6 deliverables + owner decisions → evidence)

| Gate item | Status |
|---|---|
| (§6.A / Q3 / rulings 3, 5) Context-bar panel block + materializer: seven `cbp_*` features from a VERIFIED replay-chart artifact only; 300 s / 900 s; 13 complete source bars or all-null with exact evidence; ddof 0; 18:00 ET reset; session of the bar's final instant; `IFVG_CONTEXT_BAR_PANEL_V1` as a versioned event; `BP0_CONTEXT_BAR_PANEL`; grain/bundle-key coherence | **PASS** — `test_context_bar_panel.py`, `test_feature_blocks.py` (registration event, `PRE_R6_1_*`), the panel E2E; S05 now binds the panel to the chart it asked for under a declared selection policy (review S1/F13) |
| (§6.B / correction 3 / D3) Fold schedules as an identity; every fold set a persisted artifact; cross-grain equality on the schedule, never the row-population id; the ONE legacy hash | **PASS** — `test_fold_schedules.py`; S08 derives the schedule ONCE from the candidate view's observed days (review F7; DEV-R6.1-17) |
| (§6.C / correction 2 / D7) Two assignment artifacts: the descriptive OOS assignment (candidate fold OOS / the normative panel→candidate PIT rule with typed nulls, never across 18:00 ET) and the fold-local feature artifact (fit k → fold k) as the ONLY supervised feature source | **PASS** — `test_regime_oos_assignment.py` (PIT matrix, next-day warmup, refusals), `test_regime_fold_features.py` (the leakage test), `test_regime_sample_adequacy.py`; `panel_source_bar_incomplete` recorded (DEV-R6.1-11) |
| (§6.D / D2) Verified observation seam + executor: every input verified-loaded; `source_artifact_ids` from the loaded envelope; fits reused only by reproduction; single-threaded kernel | **PASS** — `test_regime_observation_source.py`, `test_regime_store.py`, the double-run pipeline test; the panel-grain candidate as-of source is a verified bundle-view ref (review F1; #95) |
| (§6.E / D1 / D4 / D14 / ruling 1) The regime study inside the 16-stage pipeline: frozen `RegimeStudyRequest`; descriptive studies S09a only with deterministic S10 decisions at the evidence as-of; model-bearing studies freeze the exact authority (verified at readiness, S05, S09a, S10, activation) and run S09b/S09c; S14 zero fitting; no "latest" lookup | **PASS** — `test_pipeline_regime.py` 19 (descriptive candidate / panel, supervised candidate / PANEL two-pass E2E, zero fitting on all three shapes, S07 refusal, second-attempt reuse, S02 verified reproduction — reviews F4 / S4 / F11); the UI verifies the frozen authority BEFORE persisting (review S7) |
| (§6.F / D5 / D6 / ruling 4) Owner-decision artifacts (decisions 25/28/29/30 incl. the exact KMeans snapshot/hash), store-owned fail-closed supersession, synthetic provenance only in the synthetic scope, FEATURE_ELIGIBLE only against the artifact; STRATIFICATION_READY structural | **PASS** — `test_owner_decisions.py` (+ hash-chained, publish-before supersession; weaker provenance refused — review S2/F8), `test_regime_store.py` (D6 at persistence, MODEL_FEATURE unpersistable, namespace guard, monotone derived `decided_at` — reviews F2 / S3 / S5 / F10 / S11), `test_regime_stratification_gate.py` (owner authorization under the run scope; structural gate from the loaded assessment — reviews S6 / F2), the promotion CLI tests |
| (§6.G / §5.5) The five stratification classes; `RegimeFilterRef` consumed; thin strata typed at the stamped floors; the frontier never a selection input; prop events by source trade then PIT; D15 report-local summary | **PASS** — `test_regime_stratification.py` 15 (incl. the bounded `account_event_regime_summary.parquet`, its budget and tamper refusals — review F3; `delivered_by` — review F6; D15 precedence + own trading day — review F9) |
| (§6.G / D9) Status-gated `IFVG_REGIME_CONTEXT_V1` activation bound to the frozen ids; `B7_CORE_REGIME`; fit-local model features, canonical id reporting-only (ruling 6) | **PASS** — `test_regime_supervised_studies.py` (activation refusals; three-rung bundle ladder on both arms; deep-immutable summaries — reviews F5 / F15), the supervised E2E (`activated block …` at S09c) |
| (§6.G / correction 6 / D15) Prop-event detail: versioned, compressed, bounded; budgets before atomic publication; `none_v0` never widened | **PASS** — `test_account_event_detail.py` 13; the writer streams literally through the store's sidecar-producer protocol (review S12/F16; DEV-R6.1-14) |
| (§6.H / D10 / D11) Per-fold bootstrap with the gate `minimum_bootstrap_aligned_ami_mean` on the protocol-wide minimum; grain-specific transition semantics | **PASS** — `test_regime_service.py`, `test_regime_diagnostics` cases, the panel / candidate transition views in the smoke |
| (§6.J / correction 7 / D13) The bundle-aware CatBoost rung on both arms; `comparison_row_id` bundle-independent; the frozen M0–M3 lane byte / identity unchanged | **PASS** — `test_catboost_bundle_model.py`, `test_controlled_feature_study.py`, `test_ml_registries.py` (golden protocol hash) |
| (§6.K) Evidence hygiene: hermetic provider tests; browser manifest v2 bound to the commit; source-review patch | **PASS** — as-is == keys-cleared counts; `verify_browser_manifest.py` OK (`test_browser_manifest_validator.py`); `R6.1.patch` + sha256 |
| (§6.L) Docs in the same change: D-047, ARCHITECTURE / README / pipeline_state (lane transforms), DECISIONS_TAKEN #76–#100, DEVIATIONS, ACCESS_SAFETY | **PASS** — amended after the adversarial round (reviews F16 / S8) |
| Adversarial round (kickoff §10): two independent read-only reviewers; every finding dispositioned before the commit | **PASS** — 0 blockers; 6 majors + 4 mediums + 18 minors; 26 FIXED in code/tests, 1 FIXED-partial with a recorded deviation (F14: the trade tables have no persisted artifact), 1 ACCEPTED-recorded (F12); the safety lens's protected/sealed verdict AFFIRMED |
| Protected/sealed counters | **ZERO** — `ACCESS_SAFETY_EVIDENCE.md`; re-checked after the smoke: `find data -type f -newermt 2026-08-25` → 0; no regime / owner / fold / search directory under `data/` |

## Adversarial review

Two independent read-only reviewers (contract fidelity F1–F16; safety /
access / immutability S1–S12): **0 blockers; 4 majors + 12 minors (contract
lens) and 2 majors + 4 mediums + 6 minors (safety lens) — ALL 28
dispositioned** in `ADVERSARIAL_REVIEW_RESOLUTION.md`: the verified
candidate as-of provenance (F1), D6 on every path + the owner-authorizing
report gate (F2/S6), the bounded event-regime summary parquet (F3), the §9
test coverage (F4), the S05 chart binding + declared selection (S1/F13), the
hash-chained publish-before supersession (S2/F8), the namespace-confined
synthetic scope (S3), S02 reproduction fail-closed (S4), MODEL_FEATURE
unpersistable (S5), the derived `decided_at` (F10/S11), one hard-id
vocabulary (F11), persisted-only S14 assigner (F14, partial), deep-frozen
summaries (F15), the literally streaming writer + producer protocol (S12),
UI readiness before persisting (S7), stale-lock / sanitized CLI (S10), lazy
`threadpoolctl` (S9), honest test names (F5), `delivered_by` (F6), D15
precedence (F9), the S08 schedule (F7), the docs (F16/S8), and the recorded
`panel_source_bar_incomplete` (F12).

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R6: blocks R1 acceptance and, transitively, R6.1's (and every later
   release's) acceptance; it also gates the real five-day regime mini-run
   and the bounded MBP-1 diagnostic.
2. **Ratification of decisions 25/28/29/30** as persisted owner-decision
   artifacts over the REAL protocol + assessment (the drafts are proposals);
   without them no real protocol can reach FEATURE_ELIGIBLE and no
   model-bearing regime study can be frozen.
3. **Coverage policy v2 on the real fixture** (R5B.1 blocker 2) for real
   MBP-1 research.
4. Hardening candidates open by design: a persisted executed-trade table
   artifact (DEV-R6.1-18), report envelopes for the modeled classes
   (DEV-R6.1-10), `run_scope` inside the promotion payload and a
   listing-free defence against deleting the supersession log AND its head
   together (DEV-R6.1-15; the DEV-R6-8 trust boundary), declaring
   `threadpoolctl` (DEV-R6.1-13), the four-viewport / keyboard-only /
   performance-budget UI gates, store signing / non-pickle serialization
   (DEV-R6-8).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no real source path was ever
constructed; the only real-path reader the regime lane can reach is the
verified replay-chart pair loader behind the `context_bar_source` seam, and
S05 now refuses any artifact that is not the chart it asked for; every store
write in tests is `tmp_path`-rooted and the smoke wrote only under
`%TEMP%\ifvg_r61_smoke\<key>`; `data/` gained zero files; Strategy-Core clean
and Trade-Lab untouched; the frozen M0–M3 lane files are byte-unchanged.
Corroborated by the safety reviewer's independent grep / on-disk audit
(verdict AFFIRMED).
