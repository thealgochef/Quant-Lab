# R1 — Gate Summary

**Release:** R1 — Contracts, identities, access, stores + the real baseline vertical slice (path implemented; slice NOT executed)
**implementation_status: complete**
**acceptance_status: blocked_verification_authorization**

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) is the first blocker for this and every
> dependent release.

## Commit

- **Hash:** `3546254` (branch `feature/ifvg-prop-robust-config-search-v1`; parent `58cbaa2`)
- **Message:** `R1: add search identities, verification policy, and stores` (+ acceptance-blocked statement)
- Not pushed. Not merged. 42 files changed, +10,346 / −7.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` committed as HEAD + lane appends only; the
  user's pre-existing uncommitted hunks remain in the worktree and are
  **byte-identical** to `PRE_EXISTING_DIFF.patch` post-commit (151/151 hunk
  lines verified). `docs/ML_TRAINING_WORKBENCH.md` untouched and uncommitted.

## Repo files touched

See `FILES_TOUCHED.md` (21 new source modules, 14 new test files, 3
sanctioned code modifications, 4 doc updates incl. D-039/D-040/D-042/D-043).

## Implementation-progress files produced

`PRE_IMPLEMENTATION_BASELINE.md`, `PRE_EXISTING_DIFF.patch`,
`BASELINE_PYTEST.txt`, `R1_PYTEST_FULL.txt`, `COVERAGE_MATRIX_PROPOSED.json`,
`WINDOW_COVERAGE_SCAN.json`, `coverage_matrix_build.py`,
`window_coverage_scan.py`, `DRAFT_VERIFICATION_AUTHORIZATION.md`,
`FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-R1-1…8), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `GATE_SUMMARY.md` (this file).

## Exact commands, exit codes, counts, timings

| Command | Result | Exit | Time |
|---|---|---|---|
| `python -m pytest -q` (baseline, pre-impl) | 1089 passed, 3 warnings | 0 | 394.6s |
| `python -m pytest tests/agents/ifvg_search -q` (final) | **137 passed** | 0 | 2.6s |
| `python -m pytest -q` (final, post-review-fixes) | **1226 passed** (1089 + 137), 3 pre-existing warnings | 0 | 390.4s |
| `ruff check src tests` (final) | All checks passed | 0 | <5s |
| `git diff --check` | clean | 0 | <1s |
| `python …/coverage_matrix_build.py` | coverage_matrix_id `a4a4835a97e1e35b…` | 0 | ~5s |
| `python …/window_coverage_scan.py` | 124 windows scored; top `2026-02-06…02-11` | 0 | ~5s |

## Gate status (PHASED_DELIVERY R1)

| Gate item | Status |
|---|---|
| Identity tests (reuse/independence/sensitivity, bundle sensitivity + QL source sensitivity + portability, canonical naming, membership separation, companion versioning, semantic/annotation exclusion + identity-projection audit, GeneratedProfileCapability) | **PASS** (synthetic; 137-test lane suite) |
| Typed-axis-value + computation-path-scoped authorization fail-closed | **PASS** |
| Catalog concurrent-publisher + crash-recovery (+ Windows delete-pending contention, stale-lock break) | **PASS** |
| Store save→reload→assert / overwrite refusal / exact-ID / verified reuse | **PASS** |
| R0→R1 item (a) mid-chain seed start | **RESOLVED (synthetic proof)** — `start_after_artifact` fallback; emission-identical restart; `final_day_exhausts_dataset=False` requirement for snapshot prefixes discovered and proven |
| R0→R1 item (b) native-ID replay determinism | **RESOLVED (synthetic proof)** — byte-identical table hashes across replays |
| M0–M3 suite untouched and green | **PASS** (1089 pre-existing tests all green; no lane module modified) |
| **The real vertical slice** (dual-drive on the authorized fixture → neutrality → immutable save/reload/reuse → verifier link → zero forbidden access → stamps) | **OPEN — BLOCKED on `VerificationAuthorizationRef`** (owner decisions 21/R-5). The complete slice path is implemented and composition-tested synthetically (real identity assembly, publication, honest gates); the audit/chart companion builders + exact `setup_id` verifier jump are R2-assigned seams whose gates evaluate open until R2 wires them (DEV-R1-6). |

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5): canonical ≤5-day
   allowlist choice (coverage evidence shows the plan's candidate window is
   funnel-quiet — both lawful options in `DRAFT_VERIFICATION_AUTHORIZATION.md`),
   coverage-matrix sign-off, seed-snapshot production authorization, and the
   signed `VerificationAuthorizationRef`. Blocks R1 acceptance and,
   transitively, every dependent release's acceptance.
2. R2 companion wiring (neutrality-aware audit build, chart companion,
   `setup_id` verifier jump) — required before the slice's
   `artifacts_published_and_reloaded` / `verifier_link_resolves` /
   `invariants_passed` gates can close at acceptance time.

## Protected/sealed counters

**Zero.** No path for 2026-06-11 or ≥2026-06-12 was ever constructed;
no data-store file was created or modified (both adversarial reviewers
independently verified via footprint + mtime sweeps); the only real-data
operations were the two documented read-only loads of already-authorized
immutable artifacts. `full_pipeline_not_run=true` holds for every
verification surface; no full-development replay, feature build, model fit,
prop simulation, or search run occurred.

## Adversarial review

Two independent read-only reviewers; **27 findings (1 blocker, 6 majors,
20 minors) — all fixed** and re-verified (`ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`). No finding was dismissed.
