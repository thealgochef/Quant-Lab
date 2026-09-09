# R2 — Gate Summary

**Release:** R2 — Multi-child FSM search, lineage, exact deltas, verifier integration
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R2 cannot be declared accepted until R1's acceptance — including
> the owner-approved coverage matrix, allowlist, seed snapshot, and signed
> `VerificationAuthorizationRef` — passes.

## Commit

- **Hash:** `e050a25` (branch `feature/ifvg-prop-robust-config-search-v1`; parent `3546254`)
- 36 files changed, +6,486 / −33.
- **Message:** `R2: add child orchestration, lineage, and deltas` (+ the
  acceptance-blocked statement)
- Not pushed. Not merged.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R2 lane transforms ONLY
  (`stage_shared_docs.py`); the user's pre-existing uncommitted hunks remain
  in the worktree, byte-verified post-commit against
  `../R1/PRE_EXISTING_DIFF.patch` (151/151 hunk lines byte-identical across
  all four user-owned files, UTF-8 byte-level comparison). `docs/ML_TRAINING_WORKBENCH.md` untouched
  and uncommitted. `docs/DECISIONS.md` untouched this release (no R2-reserved
  decision id; D-041 is R3's).

## Repo files touched

See `FILES_TOUCHED.md` (11 new source modules incl. the job shim, 9 new test
files, the three sanctioned pre-existing modifications + the DEV-R2-1
dataset.py seam, 3 shared-doc lane appends, plus the adversarial-fix
additions listed there).

## Implementation-progress files produced

`PRE_R2_BASELINE.md`, `R2_PYTEST_FULL.txt`, `FILES_TOUCHED.md`,
`DEVIATIONS.md` (DEV-R2-1…9 + scoping notes), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both verbatim reviewer
reports), `ADVERSARIAL_REVIEW_RESOLUTION.md`, `stage_shared_docs.py`,
`GATE_SUMMARY.md` (this file).

## Exact commands, exit codes, counts, timings

| Command | Result | Exit | Time |
|---|---|---|---|
| `python -m pytest tests/agents/ifvg_search -q` (baseline over inherited drafts) | 137 passed | 0 | 2.7s |
| lane + verifier-tab suites (pre-adversarial checkpoint) | 196 passed | 0 | 5.9s |
| `python -m pytest -q` (full repo, pre-adversarial checkpoint) | 1279 passed, 7 warnings | 0 | 582.0s |
| lane + verifier-tab suites (FINAL, post-adversarial fixes) | **216 passed**, 4 warnings | 0 | 7.2s |
| `python -m pytest -q` (full repo, FINAL) | **1299 passed** (1226 R1-final + 72 new lane + 1 net new tab), 7 pre-existing-pattern warnings | 0 | 439.7s (7:19) |
| `python -m ruff check src tests scripts/ifvg_search_job.py scripts/ifvg_verifier_tab.py` (final) | All checks passed | 0 | <10s |
| `git diff --check` | clean | 0 | <1s |

## Gate status (PHASED_DELIVERY R2)

| Gate item | Status |
|---|---|
| Synthetic 2×2 E2E: enumerate→reuse→gates→frontier→insights | **PASS** — 2×2 over two APPROVED axes under the synthetic marker; 4 enumerated/deduped; gates with human explanations; persisted frontier + 7-category insights; and (post-adversarial) reuse and gates/frontier proven TOGETHER: a fully-reused re-run reproduces the byte-identical frontier and gate outcomes from the published costed evaluations |
| Resume/cancel/lock | **PASS** — safe cancel at child boundaries with sentinel consumption; live-lock contention refusal; nonce-verified release; rename-to-quarantine stale break; resume-after-kill reuses completed children |
| Cross-study reuse via identical verified input manifests | **PASS** — second charter, zero replay invocations, verified-reuse explanations, evaluations reloaded |
| Lineage: native determinism · lineage validity · match basis · not_comparable disabling · one-to-one uniqueness + collision refusal (P1-E) | **PASS** — real-reducer determinism (Jaccard 1.0/kind); cross-profile setup lineage keys IDENTICAL across a timeout variant while native ids are disjoint; native basis refuses cross-profile tables; collisions persisted via the lineage_reports store; validity DERIVED from the axis registry; no-fuzzy source scan |
| Generated-profile capability enforced in orchestration (P0-D) | **PASS** — blocked children never reach the runner; real registry digest bound; per-combination value authorization re-runs fail-closed; `blocked_invariant_failure` reachable (DEV-R2-8 exemption documented) |
| `CohortSpec` interpretation tests | **PASS** — 1:1 mode→class mapping; descriptive/model cohorts can never select execution/prop delta families; sequential mode constructs a canonical child profile, never a filter |
| Strategy-only cells run without v3/model artifacts (P1-G) | **PASS** — R1's `test_study_cell` rows + the whole E2E runs children with `none_*` identities (the prop-sim half of the §3.8 row is R3-gated by the row itself) |
| Funnel-delta → exact-setup drill-through into the verifier | **PASS** — the full chain proven end to end: population-delta first divergence → `queue_jump("setup_id", …)` → setup-mode routing → exact provider resolution (and clean refusal where the entity does not exist); `resolve_selection` setup_id contract (1/0/N candidates) tested |
| Strategy-gate pass/fail explanations | **PASS** — all ELEVEN thresholds with human explanations incl. the bootstrap-CI gate; unmeasurable caps fail closed with trades present; typed first-failure reasons inside the 15-value vocabulary |

## Adversarial review

Two independent read-only reviewers: **1 blocker, 10 distinct majors, 19
minors — all resolved** (fixed in code/tests or explicitly ruled;
`ADVERSARIAL_REVIEW.md` + `ADVERSARIAL_REVIEW_RESOLUTION.md`). The blocker
(neutrality did not block child publication) is closed at all three seams
(worker raise, slice fail-before-publication + §3.3 ordering, orchestrator
duck-check) with empty-store proofs. 20 finding-driven tests added. Four
explicit residuals are recorded in the resolution document.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from R1's
   gate summary: the canonical ≤5-day allowlist choice, coverage-matrix
   sign-off, seed-snapshot production authorization, and the signed
   `VerificationAuthorizationRef`. Blocks R1 acceptance and, transitively,
   R2's (and every later release's) acceptance.
2. Acceptance-time slice requirements now carried by R2 evidence: the real
   slice must bind real repository states into `build_slice_companions`
   (functools.partial) and must require a NON-vacuous verifier link
   (`verifier_link_vacuous_zero_targets == False`) — a funnel-quiet window
   satisfies the machinery but proves no content.
3. R4 obligations recorded in DEVIATIONS: persist lineage-uniqueness reports
   before surfacing cross-profile deltas; registry-gate the job shim's
   `--runner-entry`; interaction-contrast evaluation (declarations already
   frozen and typed-refused).

## Protected/sealed counters

**Zero.** No path for 2026-06-11 or ≥2026-06-12 was ever constructed; every
R2 test ran on synthetic days (2026-01-13…15) inside tmp directories; no
real data-store file was created or modified; both adversarial reviewers
independently verified the date-literal, no-listing, and tmp-only-write
claims (`ACCESS_SAFETY_EVIDENCE.md`). `full_pipeline_not_run=true` holds on
every verification surface; no full-development replay, feature build, model
fit, prop simulation, or real search ran.
