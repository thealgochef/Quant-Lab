# R3 — Gate Summary

**Release:** R3 — Prop lifecycle: fidelity contracts first, then synthetic contract/lifecycle verification
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R3 cannot be declared accepted until R1's acceptance — including
> the owner-approved coverage matrix, allowlist, seed snapshot, and signed
> `VerificationAuthorizationRef` — passes.

## Commit

- **Hash:** `a23893c` (branch `feature/ifvg-prop-robust-config-search-v1`; parent `e050a25` = R2)
- 31 files changed, +7,707 / −35.
- **Message:** `R3: add prop lifecycle contracts and synthetic account engine` (+ the acceptance-blocked statement)
- Not pushed. Not merged.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R3 lane transforms ONLY
  (`R3/stage_shared_docs.py`); the user's pre-existing uncommitted hunks
  remain in the worktree, byte-verified post-commit against
  `../R1/PRE_EXISTING_DIFF.patch` (diff-line comparison identical across all
  four user-owned files). `docs/ML_TRAINING_WORKBENCH.md` untouched and
  uncommitted. `docs/DECISIONS.md` gains D-041 (R3's reserved id) and the
  updated reservation note.

## Resumption note

The R3 modules and five test suites were authored in a prior session
(2026-08-18) that ended before verification, progress docs, or a commit;
this session (2026-08-19) reconstructed the baseline (`PRE_R3_BASELINE.md`),
completed the missing seam tests (which immediately found a real
orchestrator defect — DEV-R3-2), ran the two-reviewer adversarial pass, and
resolved every finding before this single release commit.

## Repo files touched

See `FILES_TOUCHED.md` (13 new propsim modules incl. the production
`search_bridge.py`, 9 new test files, 3 modified search-lane modules,
2 modified search-lane test files, 3 lane-staged shared docs + D-041).

## Implementation-progress files produced

`PRE_R3_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-R3-1…11),
`TEST_RESULTS.md`, `ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`
(both reviewer reports), `ADVERSARIAL_REVIEW_RESOLUTION.md`,
`stage_shared_docs.py`, `GATE_SUMMARY.md` (this file) +
`../DECISIONS_TAKEN.md` entries 17–25.

## Exact commands, exit codes, counts, timings

| Command | Result | Exit | Time |
|---|---|---|---|
| `python -m pytest tests/propsim tests/agents/ifvg_search -q` (resumption baseline) | 296 passed | 0 | 7.3s |
| `python -m pytest -q` (full repo, pre-review checkpoint) | 1341 passed, 7 warnings | 0 | 9:07 |
| lane suites (FINAL, post-review fixes) | **334 passed**, 4 warnings | 0 | 8.0s |
| `python -m pytest -q` (full repo, FINAL) | **1372 passed** (1299 R2-final + 73 net new), 7 pre-existing-pattern warnings | 0 | 8:24 |
| `python -m ruff check src tests scripts/... R3/stage_shared_docs.py` (final) | All checks passed | 0 | <10s |
| `git diff --check` | clean | 0 | <1s |

## Gate status (PHASED_DELIVERY R3)

| Gate item | Status |
|---|---|
| **Synthetic contract/lifecycle verification: one synthetic contract fixture compiled and simulated end-to-end on the R1 baseline stream** | **PASS** — `test_synthetic_contract_e2e.py`: synthetic source documents → PASSED field-level compilation → `synthetic_fixture_verified` (and provably CANNOT advance) → `PropFirmContractEnvelope` → adapters over the R1 baseline synthetic EXECUTED_TRADE stream → artifacts → bundle → capability report → account walk → `PayoutReliabilityVector` → `evaluate_prop_gates` (6 checks) → `run_search` frontier champion on `expected_net_payout_90d` via the PRODUCTION `make_prop_simulator` |
| §16.4 rule matrix (rule-by-rule fixtures) | **PASS** — reviewer 1's coverage table closed: trailing ×3, static, DLL hard/soft, payout eligibility/cap/split, 3 post-payout rules, all four fee kinds incl. RECURRING (trading-day + calendar-month), breach with linked trade, replacement ordinal + reset fee, min days, winning days, consistency denominator (+ funded-phase consistency), contract limits + micro-scaling, exact same-day ordering, `AccountWalk`≡`EvaluationWalk` parity (3 styles × 2 modes), all 7 sizing families + all 11 skip reasons, adapter mapping + sequence-pinning stream hash |
| P0-12/P0-H path-fidelity capability | **PASS** — capability-set + accepted-class support, PER TRADE, artifact-derived fidelities, fail-closed `PathCapabilityReport`, mixed-bundle degradation |
| P0-H 1m scenario truthfulness + scenario identities | **PASS** — `observed_intrabar_order` unrepresentable except `"unknown"`; two policies → two artifact/simulation identities AND two walked results (favorable-first breaches where adverse-first survives); wording scan over all 13 modules |
| P0-13 event-envelope total order | **PASS** — strict unique ordinals, deterministic reruns, exact source links (positive + negative path-event citation), intraday ratchets now evented |
| P0-14 DayCountBasis | **PASS** — clock layer + WALK layer: every declared rule validated at construction; calendar bases work under the historical clock and refuse under bootstrap; withdrawal cadence elapses in its own basis |
| P0-15/16 withdrawal + replacement identity | **PASS** — two-run payout-stream divergence from a withdrawal-only change; constructor-surface audit with ZERO guard exemptions; bootstrap horizon inside the protocol id |
| P0-17/P1-A compilation + status ladder | **PASS** — one-way ladder, synthetic cap bound to the compilation's document set, supersession retires ids (`effective_at` + refusal helper wired into the runner) |
| §3.4/P0-21 simulation acceptance | **PASS** — duplicate sampled sequences legal, unique `path_instance_id` + stored sequence hashes, same-spec determinism, common correlated path (replay AND portfolio bootstrap), deterministic stress ×9, event-based lower-tail metrics |
| Frontier wired to prop metrics | **PASS** — through REAL simulations (production bridge) and through the seam contract tests (ALL-legs, worst-firm flip, scoping, containment, explicit exclusions) |

## Adversarial review

Two independent read-only reviewers (contract-fidelity lens; safety/identity
lens): **4 blockers, 7 majors, 11 minors after dedup — all resolved** in
code/tests before the commit (`ADVERSARIAL_REVIEW.md` +
`ADVERSARIAL_REVIEW_RESOLUTION.md`; residuals listed there are design-scoped,
none blocking). The reviews executed reproductions; the three deepest
resolutions: scenario policies now change walked results, the bootstrap
horizon rides the identity, and calendar bases fail closed at walk
construction.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1/R2: the canonical ≤5-day allowlist choice, coverage-matrix sign-off,
   seed-snapshot production authorization, and the signed
   `VerificationAuthorizationRef`. Blocks R1 acceptance and, transitively,
   R3's (and every later release's) acceptance.
2. Acceptance-time slice requirements carried by R2 evidence (unchanged):
   real repository states bound into `build_slice_companions`; non-vacuous
   verifier link.
3. R4 obligations (unchanged, from R2/DEVIATIONS) + R3 carries forward:
   scenario-mode bridging at the production seam when bar evidence is
   plumbed (DEV-R3-11) and real artifact-store manifests (DEV-R3-9).

## Protected/sealed counters

**Zero.** No path for 2026-06-11 or the sealed range was ever constructed;
every R3 test ran on synthetic days (2026-01-* family + the 2020-01-06
synthetic bootstrap anchor) with tmp-only writes; no real data-store file
was created or modified; no network access; no real prop-firm rules
(synthetic evidence only, capped at `synthetic_fixture_verified`);
Strategy-Core and Trade-Lab untouched; the evaluation-only propsim API is
byte-untouched with its suites green; `full_pipeline_not_run` holds — no
full-development replay, feature build, model fit, prop search, or operator
run occurred. Corroborated by the safety-lens reviewer's independent grep
audit (`ADVERSARIAL_REVIEW.md`).
