# R5B.1 — Gate Summary

**Release:** R5B.1 — MBP-1 source-coverage policy correction (the first of
the two correction-release commits planned by
`../R6.1-CORRECTION-PLAN-DOCS-FINAL/R6.1_IMPLEMENTATION_PLAN.md` revision 3
— workstream I / D12; owner planning decision Q1; plan-review correction 4;
final contract-closure rulings #2/#7)
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release; in addition, real MBP-1 research and R5B acceptance stay blocked
> until coverage policy v2 passes on the real fixture (plan §12 blocker 3 —
> partition-scope evidence manifests or an owner-approved recovery rule are
> prerequisites for any `evidenced_complete` day).

> Session note: the R5B.1 change set was authored before a context reset;
> its first internal review report was lost with the context (its fixes are
> in the code), so the adversarial round was re-run in full by two
> independent read-only reviewers on the post-fix tree
> (`ADVERSARIAL_REVIEW.md`) and every finding dispositioned
> (`ADVERSARIAL_REVIEW_RESOLUTION.md`). The evidence folder was completed
> across a second reset (the resume-from-here record is
> `../R6.1/_PROGRESS_CHECKPOINT.md`).

## Commits

- **`f3f9ac2`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `179a2c9` = R6; git tree `b9fa5b84…`) — 25 files changed,
  +4,444 / −156 (2 new src + 1 new script + 1 new test file; 9 modified
  src + 1 modified script + 7 modified test/fixture files;
  `docs/DECISIONS.md`; the three staged shared docs).
  **Message:** `R5B.1: withdraw sequence-jump gap semantics; evidence-based
  MBP-1 coverage policy v2` (+ the acceptance-blocked statement).
- Not pushed. Not merged.
- **Path-scoped**: the commit carries exactly the R5B.1 file list
  (`FILES_TOUCHED.md`); the concurrent R6.1 worktree files
  (regime lane, panel materializer, fold schedules, owner decisions, …)
  were stashed for the release-final runs and are NOT in this commit.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R5B.1 lane transforms ONLY
  (`stage_shared_docs.py`; post-commit `--apply-worktree` replayed the
  same transforms). **Post-commit verification: the surviving worktree
  diff on the four user-owned files is content-identical to
  `../R1/PRE_EXISTING_DIFF.patch`** (diff-of-diffs empty modulo `index`
  header lines — `_surviving_shared_doc_diff.patch`).
  `docs/ML_TRAINING_WORKBENCH.md` untouched and uncommitted.
  `docs/DECISIONS.md` gains D-048 + the updated reservation note
  (committed normally).
- Source-review evidence: `R5B.1.patch` (`git format-patch --stdout
  179a2c9..f3f9ac2`) + `R5B.1.patch.sha256` (final closure #8).

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against `git status --short` before
the release commit, including the adversarial-fix round.

## Implementation-progress files produced

`PRE_R5B_1_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`
(DEV-R5B.1-1…4 + scoping notes), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both reviewer
reports), `ADVERSARIAL_REVIEW_RESOLUTION.md` (20 dispositions),
`stage_shared_docs.py`, raw outputs `_final_pytest.txt` /
`_final_pytest_keys_cleared.txt` / `_detached_worktree_pytest_as_is.txt` /
`_detached_worktree_pytest_keys_cleared.txt` / `_ruff_and_diffcheck.txt` /
`_surviving_shared_doc_diff.patch`, `R5B.1.patch` + `.sha256`,
`GATE_SUMMARY.md` (this) + `../DECISIONS_TAKEN.md` entries 67–75 (and the
in-place "AMENDED by R5B.1" pointers on #48/#49; `../R5B/DEVIATIONS.md`
DEV-R5B-6 "SUPERSEDED by R5B.1").

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (the `f3f9ac2` tree, main
tree): full repo 1756 passed, 2 failed as-is — the two failures are the
PRE-EXISTING environment-dependent provider-key pair; with the keys removed
the same tree reports 1758 passed, 0 failed** (9:40 / 9:54; raw
`_final_pytest.txt` / `_final_pytest_keys_cleared.txt`): **+23 net new
tests, 0 new failures.** `ruff check src tests scripts` clean;
`git diff --check` clean.

## Gate status (plan §6.I deliverables + owner Q1 items → evidence)

| Gate item | Status |
|---|---|
| (Q1-1) Withdraw "raw venue sequence jump > 1 = source gap" | **PASS** — `Mbp1SourceContract.gap_semantics` is the Literal `mbp1_source_coverage_declared_evidence_v2` (`sequence_gap_marks_interval_invalid_v1` fails validation); `_sequence_gap_intervals` → `Mbp1SequenceJumpDiagnostics` / `Mbp1TsRecvGapDiagnostics`; a giant sequence jump never lowers coverage (test); the rationale (venue channel sequence; top-of-book-only emission) is recorded in the contract, D-048, and ARCHITECTURE |
| (Q1-2) Versioned evidence-based coverage policy at an explicit scope | **PASS** — `Mbp1EvidenceScope` (four scope levels, physical partition key, UTC date, verified expected span); accepted kinds = declared partition gap manifest / `F_MAYBE_BAD_BOOK` (DBN bit 4) / dataset-condition record; `completeness_status ∈ {evidenced_complete, declared_gaps, completeness_unknown}`; positive completeness ONLY through `compile_mbp1_partition_gap_manifest` over a verified `Mbp1CompletenessCompilationReport` bound to the partition content it certifies (reviews F1/S3/S7); dataset conditions map to the five `vendor_*` states and only downgrade |
| (Correction 4) `F_MAYBE_BAD_BOOK` closes only at a documented recovery boundary; fail closed to the partition end | **PASS** — trusted start (manifest / last trusted event / partition start), documented recovery kinds only, `open_uncertainty_to_partition_end`, the flagged record always inside the interval (F5), a boundary beyond the partition end never closes it (F6), scope-bound boundaries/conditions (F7), channel scope only behind a verified map else publisher/physical partition (final closure #2) — §9.1 `test_maybe_bad_book_without_documented_recovery_fails_closed` + `test_multi_utc_partition_denominators_and_channel_wide_bad_book_scope` |
| (D12) Coverage calculation — physical-partition denominator, union/merge/clip, multi-UTC duration weighting, head/tail only when declared, empty span unknown | **PASS** — `intersection(verified span, authorized session span)`; half-open pairwise-disjoint spans (F3); merged once (§9.1 `test_overlapping_declared_gap_intervals_are_merged_once`); stray in-session events refuse the build and uncovered sub-spans type `coverage_evidence_unavailable` (F2); `test_scope_mismatch_empty_span_and_head_tail_handling` |
| (Q1-3) Diagnostics only; never an interval / reason / coverage change | **PASS** — sequence-jump + `ts_recv`-gap diagnostics ride the partition row and the coverage report (`sequence_positive_jump_count`, diagnostic-labelled in the UI); `completeness_inferred_from_sequence_continuity=False` structurally in the diagnostic report |
| (Q1-4) Re-mint every affected identity | **PASS** — normalized schema hash (`publisher_id` + `flags`), source artifacts, `with_reresolved_block` v3 (formula/materializer v2; `PRE_R5B1_*` exported; replay-the-event proof; B2/B3 move, B0/B1/B4 stable), feature artifacts, coverage reports (`coverage_policy_id`), controlled-study ids — DEV-R5B.1-3; nothing immutable mutated |
| (Q1-5) The owner's six synthetic proofs + the §9.1 MBP-1 rows | **PASS** — `test_mbp1_coverage_evidence.py` (18): normal skips never gaps; declared gap → exactly the intersecting windows, never widened/imputed; identical evidence → identical artifacts; no partition-scope evidence → `coverage_evidence_unavailable`; bad-book fails closed; overlaps merged once; `available` without partition evidence is `completeness_unknown`; scope mismatch ignored with a diagnostic; manifest tamper refused; MBP-10 guards green |
| (Q1-6) Bounded real-data diagnostic under the canonical ≤5-day fixture, gated by the R1 real-slice authorization | **PASS (shape) / BLOCKED-honest (real run)** — `scripts/ifvg_mbp1_coverage_diagnostic.py` + `assert_diagnostic_authorized` (verified run envelope + `VerificationAuthorizationRef` allowlist-hash binding, policy id, `search_test/v1` namespace, verified coverage matrix, the one canonical allowlist, `VerificationReplayPolicy`); the gate matrix + CLI refusals are test-pinned; the synthetic path proves the report shape; store-verified evidence enters only via `--evidence-json` (F4/S6). **No real run occurred** (DEV-R5B.1-4) — the real run is an owner action |
| (Q1-7) Real MBP-1 research and R5B acceptance stay blocked until the corrected policy passes | **HELD** — `docs/pipeline_state.yaml` `R5B_1_mbp1_coverage_correction` records `acceptance_status: transitively_blocked_by_R1`; D-048 trade-off states every real MBP-1 day is `completeness_unknown` until partition-scope evidence exists |
| Real read seam clips the UTC-date file to the trading day and the development cutoff (review S1) | **PASS** — `read_mbp1_partition_frame` clips BEFORE normalization/hashing; counts + raw-file sha256 ride the partition row; `test_real_read_clips_to_trading_day_session_and_development_cutoff`; the previous UTC file's evening portion is NOT composed (DEV-R5B.1-1) |
| Synthetic evidence lawful only under the synthetic marker | **PASS** — `evidence_provenance` on every partition row; S05 `assert_evidence_provenance_permitted`; the real builder refuses synthetic provenance; `test_synthetic_evidence_is_refused_outside_the_synthetic_scope` |
| UI: evidence-based coverage view + stamped R-6-family defaults (`proposed_protocol_default`) | **PASS** — two AppTests; the panel is read-only under the FUX source scans (review S5 precedent) |
| Docs in the same change (D-048; DECISIONS_TAKEN #48/#49 amended; DEV-R5B-6 superseded; ARCHITECTURE / README / pipeline_state) | **PASS** — F11 |

## Adversarial review

Two independent read-only reviewers (contract fidelity; safety/access):
**0 blockers; 4 majors + 8 minors (contract lens) and 3 majors + 1 medium
+ 4 minors (safety lens) — ALL 20 dispositioned** in
`ADVERSARIAL_REVIEW_RESOLUTION.md`: every finding FIXED in code/tests (the
content-bound positive completeness F1/S3, the stray-event/uncovered-span
refusals F2, half-open disjoint partitions F3, the store-verified evidence
seam F4/S6, the real-read clipping S1 [+ DEV-R5B.1-1], the R1 real-slice
gate S2/F9, the exact namespace S4, sealed-date unrepresentability S5, the
non-self-referential source hash S7, the interval closure/scope exactness
F5/F6/F7/F12, the superseded-resolution refusal F8, the stamped defaults
F10, the doc pointers F11) or in process (S8 — the path-scoped commit).
The safety lens's protected/sealed verdict: **zero-counter AFFIRMED**.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R6: blocks R1 acceptance and, transitively, R5B.1's (and every later
   release's) acceptance; it ALSO gates the bounded MBP-1 coverage
   diagnostic's real run.
2. **Coverage policy v2 on the real fixture**: real MBP-1 research and R5B
   acceptance stay blocked until partition-scope evidence manifests (or an
   owner-approved recovery rule) exist and the corrected policy passes on
   the canonical ≤5-day fixture (plan §12 blocker 3); the owner-review hash
   → persisted owner-decision artifact binding lands with R6.1's verified
   owner-evidence workstream (DEV-R5B.1-2).
3. R6.1 (the second correction commit): the regime-lane correction — panel
   materializer and block, fold schedules, verified seams, owner evidence,
   stratification, CatBoost bundle rung, pipeline integration, hermetic
   provider tests, commit-bound browser evidence — in progress in the same
   worktree (`../R6.1/_PROGRESS_CHECKPOINT.md`).
4. Hardening gates open by design (four viewports, keyboard-only,
   chart/widget fallbacks, long-ID/wide-table QA, performance budgets,
   store signing / non-pickle serialization — DEV-R6-8).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no real source path was ever
constructed; the only protected/sealed date literals in the R5B.1 set are
refusal probes in one test; the real read seam clips to the authorized
session span and the development cutoff before any artifact is built; the
bounded diagnostic fails before any path without the owner's verified
authorization; `data/` gained zero files (`find data -type f -newermt
2026-08-25` → 0); `search/v1` and `search_test/v1` do not exist; no MBP-1
evidence store directory exists under `data/`; Strategy-Core clean and
Trade-Lab untouched; M0–M3, propsim, and the R6 regime modules
byte-unchanged. Corroborated by the safety reviewer's independent
grep/on-disk audit (verdict AFFIRMED).
