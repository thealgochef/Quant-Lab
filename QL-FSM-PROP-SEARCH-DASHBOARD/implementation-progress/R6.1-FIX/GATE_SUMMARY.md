# R6.1-FIX — Gate Summary

**Release:** R6.1-FIX — the compact correction of R6.1 after independent
review (Phase 1 of `../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3, owner-approved 2026-09-01; findings F-01…F-10D; §3.1–§3.10)
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21 / R-5) remains the FIRST blocker for every
> release; Phase 2 backend hardening (namespace, supersession chain, lock
> liveness, capacity, warning policy, sequential-executor truth), the
> logical-day window selection, seed production, the real ≤5-day
> verification and the Full Authorized Development run are the plan's later
> phases and are NOT part of this release.

> Session note: the R6.1-FIX implementation crossed one context reset
> (`_PROGRESS_CHECKPOINT.md` is the resume-from-here record). The adversarial
> round ran on the complete post-implementation tree (two independent
> read-only reviewers, `ADVERSARIAL_REVIEW.md`), every finding was
> dispositioned (`ADVERSARIAL_REVIEW_RESOLUTION.md`), the fixes were made by
> three disjoint workstreams plus the main agent, the full suite ran twice
> over the combined tree, and only then was the release-scoped commit made.

## Commits

- **`0c8d528`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `6c0b60a` = R6.1; git tree `6e084715…`) — 51 files changed,
  +6,949 / −643: 11 added (1 src module `search/executed_trade_table.py` +
  10 test files) + 40 modified (23 src, 1 script, 12 tests,
  `docs/DECISIONS.md` + the three staged shared docs).
  **Message:** `R6.1-FIX: bind verified assignment evidence, enforce schemas,
  persist executed trades, and fail closed on corrupt sidecars` (+ the
  acceptance-blocked statement).
- Not pushed. Not merged.
- **Path-scoped**: the commit carries exactly the R6.1-FIX file list
  (`FILES_TOUCHED.md`); `docs/ML_TRAINING_WORKBENCH.md` (user-owned) and the
  pre-existing untracked local files (data / reports / prompts / the `QL-*`
  evidence tree) are NOT in the commit.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R6.1-FIX lane transforms ONLY
  (`stage_shared_docs.py`; blob ids `88eda47b…` / `e1007116…` / `b51f7a4b…`;
  post-commit `--apply-worktree` replayed the same transforms). **Post-commit
  verification: the surviving worktree diff on the four user-owned files is
  content-identical to `../R1/PRE_EXISTING_DIFF.patch`** (diff-of-diffs empty
  modulo `index` header lines — `_surviving_shared_doc_diff.patch`).
  `docs/DECISIONS.md` gains D-049 (amended after the adversarial round) +
  the updated reservation note (committed normally).
- Source-review evidence: `R6.1-FIX.patch` (`git format-patch --stdout
  6c0b60a..0c8d528`, 479,595 bytes) + `R6.1-FIX.patch.sha256`;
  **`R6.1-FIX.bundle`** (the R5B.1 → R6.1 → R6.1-FIX chain, prerequisite R6
  `179a2c9`; 506,631 bytes; `git bundle verify` OK) + `R6.1-FIX.bundle.sha256`
  (plan §3.10 "R5B.1-to-R6.1-FIX Git bundle and SHA-256").

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against the commit's `--name-status`
after the adversarial-fix round.

## Implementation-progress files produced

`PRE_R6_1_FIX_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`
(DEV-R6.1-FIX-1…22, four amended in place), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both reviewer reports
verbatim; raw `_review_contract.md` / `_review_pipeline.md`),
`ADVERSARIAL_REVIEW_RESOLUTION.md` (18 dispositions + the midpoint fixture
note), `stage_shared_docs.py`, raw outputs `_goldens_pytest.txt` /
`_midpoint_pytest.txt` / `_final_pytest.txt` / `_final_pytest_keys_cleared.txt`
/ `_ruff_and_diffcheck.txt` / `_surviving_shared_doc_diff.patch` /
`_red_ws_ab.txt` / `_red_ws_cde.txt` (the tests-red-first collection errors
before the code landed), `R6.1-FIX.patch` + `.sha256`, `R6.1-FIX.bundle` +
`.sha256`, `_PROGRESS_CHECKPOINT.md` (the resume record), `GATE_SUMMARY.md`
(this) + `../DECISIONS_TAKEN.md` entries 101–114 (#98 carries an in-place
"AMENDED by R6.1-FIX #107" pointer).

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (the `0c8d528` content):
full repo 1993 passed, 0 failed as-is (22:28) and 1993 passed, 0 failed with
the provider keys cleared (25:07); +62 net new tests over R6.1 (1931), 0
failures in either environment.** `ruff check src tests scripts` clean;
`git diff --check` clean over the tracked and the intent-added new files;
no `assert` in any touched src module.

## Gate status (plan §3.10 → evidence)

| Gate item | Status |
|---|---|
| Every new identity / tamper / leakage test red before code and green after | **PASS** — the nine WS A–E test modules failed collection before the code landed (`_red_ws_ab.txt` / `_red_ws_cde.txt`); the fix-round tests were written red first per workstream (`ADVERSARIAL_REVIEW_RESOLUTION.md`); all green in both release-final suites |
| Two full relevant suites, as-is and with provider keys cleared, zero failures | **PASS** — 1993 / 0 and 1993 / 0 (`_final_pytest*.txt`) |
| Ruff, `git diff --check`, exact golden-ID checks | **PASS** — `_ruff_and_diffcheck.txt`; `test_r61_fix_goldens.py` (the seven `PRE_R6_1_FIX_BASELINE.md` identities unchanged; `_goldens_pytest.txt` + both full suites) |
| Double-run pipeline reuse with identical stage-result ids and zero replay on verified reuse | **PASS (with the recorded reading)** — identical stage-result ids on both double runs; `replay_invocations == 0` for every reused child of the non-stratified run; a run with stratified reporting performs exactly ONE verified reproduction per reused child (projection bytes AND core-table hash must reproduce) because S12/S13 consume raw tables the store does not hold — DEV-R6.1-FIX-20 (review B-05) |
| Zero protected / sealed access and zero new real-data artifacts | **PASS** — `ACCESS_SAFETY_EVIDENCE.md`; `find data -type f -newermt "2026-09-01 19:00"` → 0 before and after the round; no forbidden store directory under `data/`; both reviewers' access verdicts AFFIRMED |
| R5B.1-to-R6.1-FIX Git bundle and SHA-256 | **PASS** — `R6.1-FIX.bundle` + `.sha256` (verified) |
| One focused two-reviewer adversarial pass | **PASS** — 0 blockers; 1 major (B-01) + 9 mediums + 8 minors, ALL 18 dispositioned: 17 FIXED (one as a variant, B-09; one with the replay policy ACCEPTED-recorded, B-05); the midpoint fixture failure fixed |
| `implementation_status=complete`, `acceptance_status=transitively_blocked_by_R1` | **STATED** — this file, `docs/pipeline_state.yaml` (`R6_1_FIX_compact_correction`), D-049 |
| Commit boundary (plan §3.10 message; one release-scoped commit parented by `6c0b60a`; no push, no merge) | **PASS** — `0c8d528` |

## Findings closed (plan §2 → code)

| Finding | Closed by |
|---|---|
| F-01 / F-02 assignment-source identity | `VerifiedFitAssignments`, byte-for-byte fit reuse, the executor's exact loads, `FitAssignmentRef` / `regime_fit_assignment_refs`, the full consulted hash, `RegimeAssignmentEvidenceRef` table + schema hashes (+ RA-01: the service re-verifies the caller's frame and binds the artifact hash) |
| F-03 fold-feature source identity | `FoldFitRef` sidecar / schema hashes, `build_regime_fold_features(fit_assignments=…)`, the loader's exact re-check, `validate_fold_feature_rows` |
| F-04 candidate as-of policy | `candidate_as_of_missing` (both grains); the strict, stage-bound as-of hash (RA-07) |
| F-05 assignment schemas and invariants | `FIT_ASSIGNMENT_SCHEMA` + `validate_assignment_rows` on build / save / load / consumption; linkage + arithmetic self-consistency (RA-06) |
| F-06 immutable executed-trade table | `executed_trade_tables` (42-column exact projection, derived id); S02 persists / verifies / exact-loads; every costed evaluation from the loaded projection (B-01); S14 charter iteration + typed `children_skipped`; no prior-record recovery (B-03) |
| F-07 fail-closed prior-stage sidecars | the typed probe contract (nine reasons; B-02 / B-08), typed prior-stage recoveries, S15 `reload_failures` + `reload_failure_reasons` (B-09), downstream PENDING + publication reset (B-04) |
| F-08 exact label identity | `label_artifact_content_id`, mandatory `label_artifact_id` + required `label_identity_source`, unpersistable helper runs on every helper path (RA-05) |
| F-09 thin-regime accounting | `RegimeNetRAccounting` over every valid assigned trade, recomputed by its validator, FALSE / NULL semantics, `assigned_regime_count` (RA-02 / RA-03) |
| F-10A / B / C / D | `PipelineWiringError`; the normalized frame; full MBP-1 scope + ref equality; the shape-aware enum copy guard (RA-04) |

## Adversarial review

Two independent read-only reviewers (semantic identity / immutable evidence
/ PIT / access safety RA-01–RA-08; pipeline integration / fail-closed
evidence flow / typed failures / test adequacy B-01–B-10): **0 blockers; 1
major (B-01: one costed-evaluation identity published from two differently
typed frames) + 9 mediums + 8 minors — ALL 18 dispositioned** in
`ADVERSARIAL_REVIEW_RESOLUTION.md`: one canonical costed-evaluation frame
inside per-child containment (B-01), corrupt-never-absent store entries
(B-02), the deleted S14 recovery branch + S15 table / report reloads (B-03),
the publication-block reset + activation re-derivation (B-04), the
core-table-hash check + asserted replay counts (B-05), the mutation-killing
tests (B-06), the one schema-version helper + wiring error (B-07), typed
reasons at the detection point (B-08 / RA-08), immutable reload reasons
(B-09), the parametrized projection test (B-10), the service's table
re-verification + artifact hash (RA-01), the recomputing accounting
validator (RA-02), FALSE-not-NULL claims + `assigned_regime_count` (RA-03),
the shape-aware enum guard (RA-04), full-column helper label identities +
stamps (RA-05), row linkage + arithmetic invariants + the protocol column
(RA-06), the strict stage-bound as-of hash (RA-07). Both reviewers' access
verdicts AFFIRMED.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R6.1: blocks R1 acceptance and, transitively, R6.1-FIX's acceptance.
2. **Phase 2 backend hardening** (plan §4: semantic store namespace,
   immutable supersession chain + head witnesses, liveness-aware lock,
   capacity gates, warning policy, sequential-executor truth) — a separate
   `HARDENING-BACKEND` release; the six remaining project-owned warnings
   belong to it (F-18).
3. **Logical trading-day mapping and window selection; seed production;
   final verification authorization** (plan §5) — owner actions.
4. Hardening candidates recorded by this release, not gates: load-only
   prop-stage reuse from persisted raw tables (DEV-R6.1-FIX-20), the plan's
   deferred cleanups (DEV-R6.1-FIX-11), `ruff format` drift in
   `regime_stratification_service.py` (pre-existing; the gate is `ruff check`).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no real source path was ever
constructed; the release touches no data-access module; every store write in
tests is `tmp_path`-rooted; `data/` gained zero files during the release;
Strategy-Core clean at the pin and Trade-Lab untouched (its dirty files
predate this release); the frozen M0–M3 lane files are byte-unchanged and the
golden identities reproduce. Corroborated by both reviewers' independent
audits (verdicts AFFIRMED).
